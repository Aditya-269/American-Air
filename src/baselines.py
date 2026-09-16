"""
Baseline implementations for American Airlines customer support agent comparison.
Provides:
  1. TrivialBaseline: Always 'other_unclear', static generic template, always auto-handles.
  2. SimpleBaseline: Keyword/regex intent classifier, static per-intent templates, rule-only escalation.
"""

import re
from typing import Dict, Any, Optional

TRIVIAL_TEMPLATE = "Thanks for reaching out, please DM your confirmation number so we can help."

SIMPLE_INTENT_PATTERNS = [
    ("flight_delay_cancellation", re.compile(r"\b(delay|delayed|cancel|cancelled|cancellation|tarmac|late|stuck)\b", re.IGNORECASE)),
    ("baggage_issue", re.compile(r"\b(bag|bags|baggage|luggage|suitcase|carousel)\b", re.IGNORECASE)),
    ("rebooking_change_request", re.compile(r"\b(rebook|rebooked|change\s+flight|reschedule|switch\s+flight|standby)\b", re.IGNORECASE)),
    ("refund_compensation_request", re.compile(r"\b(refund|money\s+back|reimbursement|compensation|voucher)\b", re.IGNORECASE)),
    ("aadvantage_miles_issue", re.compile(r"\b(aadvantage|miles|frequent\s+flyer|loyalty|points)\b", re.IGNORECASE)),
    ("checkin_boarding_issue", re.compile(r"\b(check\s*in|boarding|boarding\s+pass|gate|seat\s+assignment)\b", re.IGNORECASE)),
    ("booking_payment_issue", re.compile(r"\b(credit\s+card|payment|charged|declined|double\s+charged)\b", re.IGNORECASE)),
    ("general_complaint_service_quality", re.compile(r"\b(rude|horrible|terrible|worst\s+airline|attitude|disrespect)\b", re.IGNORECASE)),
]

SIMPLE_TEMPLATES = {
    "flight_delay_cancellation": "We're sorry for the delay or cancellation. Please check your flight status on aa.com or see an airport agent.",
    "baggage_issue": "We apologize for the baggage issue. Please visit the airport baggage office to file a report.",
    "rebooking_change_request": "To change or rebook your flight, please visit aa.com or call our reservations team.",
    "refund_compensation_request": "For refund or compensation requests, please submit your claim at aa.com/refunds.",
    "aadvantage_miles_issue": "For AAdvantage mileage questions, please visit the AAdvantage section on aa.com.",
    "checkin_boarding_issue": "For check-in or boarding issues, please speak to an agent at the airport check-in desk.",
    "booking_payment_issue": "If you experienced payment difficulties, please verify your bank statement and visit aa.com.",
    "general_complaint_service_quality": "We take service feedback seriously and regret that your travel did not meet expectations.",
    "other_unclear": TRIVIAL_TEMPLATE
}

SIMPLE_ESCALATE_PATTERN = re.compile(r"\b(lawyer|refund|compensation)\b", re.IGNORECASE)


class TrivialBaseline:
    """
    Trivial baseline:
    - Always classifies as 'other_unclear'
    - Always replies with one generic DM template
    - Always auto-handles (never escalates)
    """
    def __init__(self):
        self.name = "trivial_baseline"

    def process(self, customer_message: str, thread_id: str = "query") -> Dict[str, Any]:
        return {
            "thread_id": thread_id,
            "customer_message": customer_message,
            "intent": "other_unclear",
            "intent_confidence": 0.50,
            "decision": "auto_handle",
            "reason_code": "trivial_always_autohandle",
            "best_retrieval_similarity": 0.0,
            "grounding_examples_used": [],
            "draft_reply": TRIVIAL_TEMPLATE,
            "trace": {"baseline": "trivial"}
        }


class SimpleBaseline:
    """
    Simple baseline:
    - Regex keyword-based intent classification
    - Per-intent static template reply (no LLM, no grounding retrieval)
    - Rule-only escalation: escalates if message contains 'lawyer', 'refund', or 'compensation'
    """
    def __init__(self):
        self.name = "simple_baseline"

    def classify(self, text: str) -> str:
        for intent, pattern in SIMPLE_INTENT_PATTERNS:
            if pattern.search(text):
                return intent
        return "other_unclear"

    def process(self, customer_message: str, thread_id: str = "query") -> Dict[str, Any]:
        intent = self.classify(customer_message)
        
        # Rule-only escalation
        if SIMPLE_ESCALATE_PATTERN.search(customer_message):
            decision = "escalate"
            reason_code = "rule_keyword_escalation"
        else:
            decision = "auto_handle"
            reason_code = "rule_keyword_autohandle"

        reply = SIMPLE_TEMPLATES.get(intent, SIMPLE_TEMPLATES["other_unclear"])

        return {
            "thread_id": thread_id,
            "customer_message": customer_message,
            "intent": intent,
            "intent_confidence": 0.70 if intent != "other_unclear" else 0.40,
            "decision": decision,
            "reason_code": reason_code,
            "best_retrieval_similarity": 0.0,
            "grounding_examples_used": [],
            "draft_reply": reply,
            "trace": {"baseline": "simple_rules", "matched_intent": intent}
        }


if __name__ == "__main__":
    triv = TrivialBaseline()
    simp = SimpleBaseline()

    test_q = "My luggage was lost and I demand compensation from my lawyer"
    print("Trivial:", triv.process(test_q))
    print("Simple :", simp.process(test_q))
