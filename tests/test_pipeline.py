"""
Comprehensive unit and integration test suite for the American Airlines Support Agent pipeline.
Verifies data cleaning, intent classification, grounding retrieval, hard escalation rules,
agent decision schema, baselines, and evaluation metrics.
"""

import os
import sys
import pytest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import clean_text, is_boilerplate_reply, parse_conversation_turns
from src.intents import IntentClassifier, INTENT_TAXONOMY, INTENT_DESCRIPTIONS
from src.retrieval import RetrievalIndex
from src.agent import AmericanAirAgent, check_hard_rules
from src.baselines import TrivialBaseline, SimpleBaseline
from src.eval_harness import evaluate_intent, evaluate_escalation


# ---------------------------------------------------------------------------
# DATA INGESTION & TEXT CLEANING TESTS
# ---------------------------------------------------------------------------
def test_clean_text():
    raw = "@AmericanAir @123456 Where is flight AA 123? @AmericanAir"
    cleaned = clean_text(raw)
    assert "@AmericanAir" not in cleaned
    assert "@123456" not in cleaned
    assert "Where is flight AA 123?" in cleaned


def test_boilerplate_detection():
    boilerplate = "We're sorry to hear this, please DM us your 6-letter record locator so we can help."
    specific = "AA Flight 441 from Charlotte is on time and scheduled to depart at 4:30 PM."
    assert is_boilerplate_reply(boilerplate) is True
    assert is_boilerplate_reply(specific) is False


def test_parse_conversation_turns():
    conv = (
        "Customer: Where is my lost bag from flight AA 99?\n"
        "Support: We are checking on that right now for you.\n"
        "Customer: Any update yet?"
    )
    turns = parse_conversation_turns(conv)
    assert len(turns) == 3
    assert turns[0]["role"] == "customer"
    assert turns[1]["role"] == "support"
    assert "lost bag" in turns[0]["text"]


# ---------------------------------------------------------------------------
# INTENT TAXONOMY & CLASSIFICATION TESTS
# ---------------------------------------------------------------------------
def test_intent_taxonomy_completeness():
    assert len(INTENT_TAXONOMY) == 9
    assert "flight_delay_cancellation" in INTENT_TAXONOMY
    assert "baggage_issue" in INTENT_TAXONOMY
    assert "other_unclear" in INTENT_TAXONOMY


def test_intent_classifier():
    classifier = IntentClassifier()
    res = classifier.classify("My checked baggage was completely destroyed and ripped apart at baggage claim")
    assert res["intent"] == "baggage_issue"
    assert res["confidence"] > 0.40
    assert "baggage_issue" in res["all_scores"]


# ---------------------------------------------------------------------------
# RETRIEVAL GROUNDING TESTS
# ---------------------------------------------------------------------------
def test_retrieval_index():
    retriever = RetrievalIndex()
    res = retriever.retrieve("Flight AA 100 was cancelled in Miami", predicted_intent="flight_delay_cancellation", top_k=3)
    assert "matches" in res
    assert len(res["matches"]) <= 3
    assert res["best_similarity"] > 0.30
    assert len(res["grounding_ids"]) <= 3


# ---------------------------------------------------------------------------
# TWO-LAYER ESCALATION HARD RULES TESTS
# ---------------------------------------------------------------------------
def test_hard_rules_legal():
    rule = check_hard_rules("I will contact my attorney and lawyer to sue American Airlines!")
    assert rule is not None
    assert rule["reason_code"] == "hard_rule_legal_threat"


def test_hard_rules_dot():
    rule = check_hard_rules("Filing an official DOT complaint under federal tarmac delay rules.")
    assert rule is not None
    assert rule["reason_code"] == "hard_rule_dot_regulatory"


def test_hard_rules_medical():
    rule = check_hard_rules("Passenger having medical emergency and heart attack on board!")
    assert rule is not None
    assert rule["reason_code"] == "hard_rule_safety_medical"


def test_hard_rules_unaccompanied_minor():
    rule = check_hard_rules("My child is an unaccompanied minor stranded alone at gate 12.")
    assert rule is not None
    assert rule["reason_code"] == "hard_rule_unaccompanied_minor"


def test_hard_rules_dollar_amount():
    rule = check_hard_rules("You owe me a $500 cash reimbursement for this hotel stay.")
    assert rule is not None
    assert rule["reason_code"] == "hard_rule_dollar_amount_demanded"


def test_hard_rules_benign():
    rule = check_hard_rules("Can you please tell me if flight 402 is on time?")
    assert rule is None


# ---------------------------------------------------------------------------
# AGENT PIPELINE & STRUCTURED DECISION OBJECT TESTS
# ---------------------------------------------------------------------------
def test_agent_hard_rule_escalation():
    agent = AmericanAirAgent()
    res = agent.process("I am suing your airline and having my lawyer file today!")
    assert res["decision"] == "escalate"
    assert res["reason_code"] == "hard_rule_legal_threat"
    assert "customer_message" in res
    assert "intent" in res
    assert "draft_reply" in res
    assert "grounding_examples_used" in res


def test_agent_routine_autohandle():
    agent = AmericanAirAgent()
    res = agent.process("Is flight AA 205 on schedule to arrive in Dallas on time?")
    assert res["intent"] in ["flight_delay_cancellation", "other_unclear"]
    assert res["decision"] in ["auto_handle", "escalate"]
    assert len(res["draft_reply"]) > 10
    # Strict anti-hallucination check: draft must not invent specific dollar amounts
    assert "$" not in res["draft_reply"]


# ---------------------------------------------------------------------------
# BASELINE TESTS
# ---------------------------------------------------------------------------
def test_trivial_baseline():
    trivial = TrivialBaseline()
    res = trivial.process("My flight was cancelled and I am suing American Airlines!")
    assert res["intent"] == "other_unclear"
    assert res["decision"] == "auto_handle"
    assert "DM your confirmation number" in res["draft_reply"]


def test_simple_baseline():
    simple = SimpleBaseline()
    res = simple.process("My bag was lost")
    assert res["intent"] == "baggage_issue"
    assert res["decision"] == "auto_handle"

    res_esc = simple.process("My lawyer will handle this refund")
    assert res_esc["decision"] == "escalate"


# ---------------------------------------------------------------------------
# EVALUATION METRICS TESTS
# ---------------------------------------------------------------------------
def test_evaluation_metrics():
    y_true_intent = ["baggage_issue", "flight_delay_cancellation", "other_unclear"]
    y_pred_intent = ["baggage_issue", "flight_delay_cancellation", "baggage_issue"]
    intent_metrics = evaluate_intent(y_true_intent, y_pred_intent)
    assert intent_metrics["accuracy"] == round(2.0 / 3.0, 4)

    y_true_dec = ["escalate", "auto_handle", "escalate"]
    y_pred_dec = ["escalate", "auto_handle", "auto_handle"]
    esc_metrics = evaluate_escalation(y_true_dec, y_pred_dec)
    assert esc_metrics["false_negatives_costly_autohandles"] == 1
    assert esc_metrics["false_positives_unnecessary_escalates"] == 0
    assert esc_metrics["false_autohandle_rate"] == 0.50
