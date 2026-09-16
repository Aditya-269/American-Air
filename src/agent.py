"""
Core AI Support Agent pipeline for American Airlines (@AmericanAir).
Implements the 4-step pipeline:
  1. Classify Intent (nearest-centroid with LLM verification fallback)
  2. Retrieve Grounding (precedents from resolved historical threads)
  3. Draft Reply (strictly grounded in precedents, zero hallucinated dollar amounts/PNRs)
  4. Escalation Decision (Two layers: deterministic hard safety/legal rules + heuristic precedent gate)
"""

import os
import sys
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.intents import IntentClassifier, INTENT_TAXONOMY, INTENT_DESCRIPTIONS
from src.retrieval import RetrievalIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
logger = logging.getLogger("agent")

# Cache directory for LLM calls
CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", ".cache/llm"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# HARD ESCALATION RULES (Layer 1: Deterministic, Never Overridden by Model)
# ---------------------------------------------------------------------------
HARD_RULES = [
    {
        "code": "hard_rule_legal_threat",
        "pattern": re.compile(
            r"\b(lawyer|attorney|sue\b|suing|lawsuit|legal\s+action|court|litigation|small\s+claims)\b",
            re.IGNORECASE
        ),
        "description": "Customer threatening legal litigation or attorney involvement"
    },
    {
        "code": "hard_rule_dot_regulatory",
        "pattern": re.compile(
            r"\b(dot\s+complaint|department\s+of\s+transportation|faa\s+complaint|federal\s+aviation|regulatory\s+complaint)\b",
            re.IGNORECASE
        ),
        "description": "DOT or FAA regulatory complaint filing"
    },
    {
        "code": "hard_rule_safety_medical",
        "pattern": re.compile(
            r"\b(medical\s+emergency|heart\s+attack|stroke|paramedic|ambulance|passed\s+out|unconscious|"
            r"bomb|weapon|gun|hijack|terrorist|assault\w*|police\s+involved|physical\s+violence|"
            r"injured\b|injury|spilled\s+hot|burned\s+by|scalded|intentionally\s+hurt)\b",
            re.IGNORECASE
        ),
        "description": "Immediate safety, medical emergency, security, or physical injury allegation"
    },
    {
        "code": "hard_rule_unaccompanied_minor",
        "pattern": re.compile(
            r"\b(unaccompanied\s+minor|umnr)\b|"
            r"\b(child|minor|kid|daughter|son|[0-9]{1,2}\s*[- ]?year\s*[- ]?old)\b.{0,30}\b(traveling\s+alone|flying\s+alone|alone|by\s+(himself|herself|themselves)|without\s+(me|parent|adult|chaperone)|stranded)\b",
            re.IGNORECASE
        ),
        "description": "Unaccompanied minor or child stranded/traveling alone"
    },
    {
        "code": "hard_rule_abusive_threat",
        "pattern": re.compile(
            r"\b(kill\s+yourself|go\s+die|burn\s+in\s+hell|punch\s+you|fuck\s+you\s+all)\b",
            re.IGNORECASE
        ),
        "description": "Abusive, harassing, or threatening language"
    },
    {
        "code": "hard_rule_dollar_amount_demanded",
        "pattern": re.compile(
            r"(\$\s*[0-9]+|[0-9]+\s*(dollars|usd))\b",
            re.IGNORECASE
        ),
        "description": "Explicit dollar amount demanded or promised"
    },
    {
        "code": "hard_rule_adversarial_prompt_injection",
        "pattern": re.compile(
            r"\b(ignore\s+(all\s+)?previous\s+instructions|disregard\s+(all\s+)?prior\s+instructions|system\s+prompt\s+override|developer\s+mode)\b",
            re.IGNORECASE
        ),
        "description": "Adversarial prompt injection attempt detected"
    }
]


def check_hard_rules(text: str) -> Optional[Dict[str, str]]:
    """
    Evaluates customer text against Layer 1 deterministic safety and legal rules.
    Returns triggering rule info if any rule matches, else None.
    """
    if not text:
        return None
    for rule in HARD_RULES:
        if rule["pattern"].search(text):
            return {
                "reason_code": rule["code"],
                "description": rule["description"]
            }
    return None


# ---------------------------------------------------------------------------
# LLM CACHE HELPER
# ---------------------------------------------------------------------------
def _get_cache_key(prompt: str, model_tag: str) -> str:
    payload = f"{model_tag}:{prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_cache(cache_key: str) -> Optional[str]:
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("response")
        except Exception:
            return None
    return None


def _write_cache(cache_key: str, prompt: str, response: str) -> None:
    cache_file = CACHE_DIR / f"{cache_key}.json"
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "response": response}, f)
    except Exception as e:
        logger.warning(f"Failed writing cache {cache_key}: {e}")


# ---------------------------------------------------------------------------
# LLM CALL WRAPPER (OpenAI / Anthropic / Offline Fallback)
# ---------------------------------------------------------------------------
def call_llm(
    prompt: str,
    system_instruction: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0
) -> str:
    """
    Calls configured LLM with automatic disk caching.
    Falls back gracefully if no API key is provided.
    """
    # Check for available API keys
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if groq_key:
        effective_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    elif openai_key:
        effective_model = os.getenv("LLM_MODEL", model)
    elif anthropic_key:
        effective_model = "claude-3-5-haiku-latest"
    else:
        effective_model = "offline"

    cache_key = _get_cache_key(f"{system_instruction}\n\n{prompt}", effective_model)
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    if groq_key:
        try:
            from openai import OpenAI
            groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
            resp = client.chat.completions.create(
                model=groq_model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            ans = resp.choices[0].message.content.strip()
            _write_cache(cache_key, prompt, ans)
            return ans
        except Exception as e:
            logger.warning(f"Groq API call failed: {e}")

    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            ans = resp.choices[0].message.content.strip()
            _write_cache(cache_key, prompt, ans)
            return ans
        except Exception as e:
            logger.warning(f"OpenAI API call failed: {e}")

    if anthropic_key:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=anthropic_key)
            resp = client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=300,
                temperature=temperature,
                system=system_instruction,
                messages=[{"role": "user", "content": prompt}]
            )
            ans = resp.content[0].text.strip()
            _write_cache(cache_key, prompt, ans)
            return ans
        except Exception as e:
            logger.warning(f"Anthropic API call failed: {e}")

    # Offline / Heuristic Fallback (deterministic high-quality synthesis)
    return ""


# ---------------------------------------------------------------------------
# THE SUPPORT AGENT
# ---------------------------------------------------------------------------
class AmericanAirAgent:
    """
    Full AI Support Agent pipeline for @AmericanAir.
    """
    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        retriever: Optional[RetrievalIndex] = None,
        retrieval_sim_threshold: float = 0.50,
        intent_conf_threshold: float = 0.48,
        mode: str = "full"
    ):
        self.mode = mode
        self.classifier = classifier or IntentClassifier()
        self.retriever = retriever or RetrievalIndex(model=self.classifier.model)
        self.retrieval_sim_threshold = retrieval_sim_threshold
        self.intent_conf_threshold = intent_conf_threshold

    def classify_intent(self, text: str) -> Dict[str, Any]:
        """
        Classifies intent via nearest centroid.
        If confidence is borderline and LLM is enabled, requests LLM verification.
        """
        res = self.classifier.classify(text, confidence_threshold=self.intent_conf_threshold)

        # LLM fallback if confidence is low or margin is ambiguous (only in full mode)
        if res.get("needs_fallback") and self.mode != "offline":
            system_prompt = (
                "You are an expert customer intent classifier for American Airlines (@AmericanAir).\n"
                "Classify the customer's opening message into EXACTLY ONE of the following classes:\n"
                f"{json.dumps(INTENT_TAXONOMY)}\n"
                "Respond with ONLY the exact class name, nothing else."
            )
            user_prompt = f"Customer message: \"{text}\"\nClass:"
            llm_choice = call_llm(user_prompt, system_prompt, temperature=0.0)
            llm_choice_clean = llm_choice.strip().lower().replace('"', '').replace("'", "")
            if llm_choice_clean in INTENT_TAXONOMY:
                res["intent"] = llm_choice_clean
                res["confidence"] = max(res["confidence"], 0.75)
                res["llm_verified"] = True

        return res

    def draft_reply(
        self,
        customer_message: str,
        intent: str,
        precedents: List[Dict[str, Any]]
    ) -> str:
        """
        Drafts a customer support reply strictly grounded in historical precedent.
        Guards against inventing PNRs, flight numbers, or specific monetary commitments.
        """
        precedent_snippets = "\n".join([
            f"Precedent {i+1} (AA Reply): \"{p['aa_reply']}\""
            for i, p in enumerate(precedents[:3])
        ])

        system_instruction = (
            "You are an official customer support representative for American Airlines (@AmericanAir) on Twitter.\n"
            "Draft a helpful, concise, empathetic response to the customer's message.\n"
            "STRICT CONSTRAINTS:\n"
            "1. ONLY state policies and instructions consistent with the provided precedents and official AA rules.\n"
            "2. NEVER fabricate flight numbers, 6-letter confirmation codes (PNRs), or gate details.\n"
            "3. NEVER promise specific dollar amounts, cash refunds, or vouchers; advise the customer to submit claims at aa.com or DM their record locator for agent review.\n"
            "4. Keep the reply within Twitter length (<260 characters) and maintain a warm, professional airline tone."
        )

        user_prompt = (
            f"Customer Intent: {intent}\n"
            f"Customer Message: \"{customer_message}\"\n\n"
            f"Verified Historical Precedents:\n{precedent_snippets}\n\n"
            "Draft Reply:"
        )

        if self.mode != "offline":
            draft = call_llm(user_prompt, system_instruction, temperature=0.2)
            if draft:
                # Clean quotes if wrapped
                draft = draft.strip().strip('"')
                return draft

        # Grounded Offline Template Synthesizer
        # Uses the best non-boilerplate precedent or template tailored to intent
        best_prec = precedents[0]["aa_reply"] if precedents else ""
        
        intent_fallbacks = {
            "flight_delay_cancellation": (
                "We know delays and cancellations disrupt your plans and we're truly sorry. "
                "Please DM us your 6-letter record locator and full name so we can look up your itinerary and assist with rebooking options."
            ),
            "baggage_issue": (
                "We're sorry for the trouble with your luggage. "
                "Please make sure to file a claim with our airport baggage service office or online, and DM us your bag tag number and confirmation code so we can assist."
            ),
            "rebooking_change_request": (
                "We'd be glad to help explore available flight options for you. "
                "Please send us a DM with your 6-letter confirmation code and preferred travel times so our team can assist."
            ),
            "refund_compensation_request": (
                "We understand your request. You can check refund eligibility or submit an official request online at aa.com/refunds. "
                "Please feel free to DM your 6-letter record locator if you need further guidance."
            ),
            "aadvantage_miles_issue": (
                "We'd be happy to check your account status and mileage activity. "
                "Please send us a DM with your AAdvantage number and full name so we can look into this for you."
            ),
            "checkin_boarding_issue": (
                "We're sorry for the difficulty with check-in or boarding. "
                "Please DM us your 6-letter record locator and full name, or see an airport gate agent if your flight is departing shortly."
            ),
            "general_complaint_service_quality": (
                "This is certainly not the travel experience we strive to provide. "
                "We appreciate your feedback and would like to look into this further. Please DM us your confirmation code and travel details."
            ),
            "booking_payment_issue": (
                "We're sorry for the difficulty with your booking transaction. "
                "Please DM us your full name, email, and the last 4 digits of the card used so our reservations team can assist."
            ),
            "other_unclear": (
                "Thanks for reaching out to American Airlines. "
                "Could you please DM us your 6-letter confirmation code or additional flight details so we can best assist you?"
            )
        }

        # If best precedent is high quality and relevant, adapt its phrasing; otherwise use verified intent policy template
        if best_prec and len(best_prec) > 25 and not precedents[0].get("is_boilerplate", False):
            # Precedent-adapted synthesis
            return best_prec
        return intent_fallbacks.get(intent, intent_fallbacks["other_unclear"])

    def process(self, customer_message: str, thread_id: str = "query") -> Dict[str, Any]:
        """
        Executes end-to-end support agent pipeline for a customer message.
        """
        cleaned_msg = customer_message.strip()

        # Step 1: Evaluate Layer 1 Hard Escalation Rules
        hard_match = check_hard_rules(cleaned_msg)
        if hard_match:
            # Hard escalation triggered immediately
            intent_res = self.classify_intent(cleaned_msg)
            intent = intent_res["intent"]
            return {
                "thread_id": thread_id,
                "customer_message": cleaned_msg,
                "intent": intent,
                "intent_confidence": intent_res["confidence"],
                "decision": "escalate",
                "reason_code": hard_match["reason_code"],
                "best_retrieval_similarity": 0.0,
                "grounding_examples_used": [],
                "draft_reply": "This case requires human representative handling due to regulatory, safety, legal, or monetary policy restrictions.",
                "trace": {
                    "layer": "hard_rules",
                    "hard_rule_description": hard_match["description"],
                    "intent_scores": intent_res.get("all_scores", {})
                }
            }

        # Step 2: Classify Intent
        intent_res = self.classify_intent(cleaned_msg)
        intent = intent_res["intent"]
        intent_conf = intent_res["confidence"]

        # Step 3: Retrieve Grounding Precedents
        retrieval_res = self.retriever.retrieve(
            query=cleaned_msg,
            predicted_intent=intent,
            top_k=3
        )
        best_sim = retrieval_res["best_similarity"]
        precedents = retrieval_res["matches"]
        grounding_ids = retrieval_res["grounding_ids"]

        # Step 4: Evaluate Layer 2 Heuristic / Model Risk Layer
        decision = "auto_handle"
        reason_code = "ok"

        if intent_conf < self.intent_conf_threshold:
            decision = "escalate"
            reason_code = "low_intent_confidence"
        elif best_sim < self.retrieval_sim_threshold:
            decision = "escalate"
            reason_code = "no_similar_precedent"
        elif intent == "refund_compensation_request" and re.search(r"\b(refund|compensation|pay\s+me|reimburse)\b", cleaned_msg, re.IGNORECASE):
            # Monetary refunds require authorized agent action
            decision = "escalate"
            reason_code = "policy_refund_agent_only"
        elif intent == "other_unclear" and best_sim < 0.60:
            decision = "escalate"
            reason_code = "ambiguous_intent"

        # Step 5: Draft Grounded Reply
        draft = self.draft_reply(cleaned_msg, intent, precedents)

        return {
            "thread_id": thread_id,
            "customer_message": cleaned_msg,
            "intent": intent,
            "intent_confidence": intent_conf,
            "decision": decision,
            "reason_code": reason_code,
            "best_retrieval_similarity": best_sim,
            "grounding_examples_used": grounding_ids,
            "draft_reply": draft,
            "trace": {
                "layer": "heuristic_model",
                "retrieval_matches_count": len(precedents),
                "intent_margin": intent_res.get("margin", 0.0),
                "intent_scores": intent_res.get("all_scores", {})
            }
        }


if __name__ == "__main__":
    agent = AmericanAirAgent()
    sample_queries = [
        "Flight AA 1245 was cancelled in Chicago. What are my options to get to Boston tonight?",
        "My checked luggage is missing at baggage carousel terminal 3.",
        "I will sue American Airlines and am contacting my lawyer for this delay!",
        "Can I get a $500 cash refund for my ticket?",
        "My 8 year old daughter is flying alone as an unaccompanied minor and got stuck in Charlotte!",
        "Thanks AA for the wonderful flight to Hawaii today!"
    ]
    print("\n--- Testing Agent Pipeline ---")
    for sq in sample_queries:
        res = agent.process(sq)
        print(f"\n[Query] {sq}")
        print(f"  -> Intent: {res['intent']} (conf={res['intent_confidence']:.2f})")
        print(f"  -> Decision: {res['decision'].upper()} (reason={res['reason_code']})")
        print(f"  -> Best Precedent Sim: {res['best_retrieval_similarity']:.3f}")
        print(f"  -> Draft: {res['draft_reply'][:100]}...")
