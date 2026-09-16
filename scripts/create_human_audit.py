"""
Script to create the 50-item human audit set (data/human_audit_50.jsonl)
for judge-vs-human agreement analysis on American Airlines support replies.
Scored blind to model identity following the 4-dimension 1-5 rubric:
  - groundedness (1: hallucinated specifics -> 5: strictly grounded)
  - correctness (1: harmful/wrong -> 5: accurate AA policy)
  - tone_empathy (1: rude/cold -> 5: warm, professional)
  - completeness (1: ignored ask -> 5: actionable resolution)
"""

import json
from pathlib import Path

# Human evaluation scores for 50 diverse cases (rater A1)
# (example_id, groundedness, correctness, tone_empathy, completeness, human_note)
AUDIT_SCORES = [
    ("gold-001", 5, 5, 4, 5, "Appropriately escalates dual-issue complex query"),
    ("gold-002", 4, 4, 4, 4, "Understands sarcastic complaint, offers DM"),
    ("gold-003", 5, 5, 4, 5, "Strictly avoids promising $800, routes to legal/claims"),
    ("gold-004", 5, 5, 5, 5, "Escalates unaccompanied minor with highest urgency"),
    ("gold-005", 5, 5, 4, 4, "Routes DOT regulatory threat to compliance"),
    ("gold-006", 5, 5, 5, 5, "Immediate escalation for medical emergency"),
    ("gold-007", 5, 5, 5, 5, "Friendly closure for already resolved inquiry"),
    ("gold-008", 4, 4, 4, 4, "Understands heavy slang baggage claim accurately"),
    ("gold-009", 4, 4, 3, 4, "Handles Spanish flight cancellation appropriately"),
    ("gold-010", 4, 4, 4, 4, "Directs to baggage policy on mileage redemptions"),
    ("gold-011", 5, 5, 4, 5, "Escalates duplicate $450 card billing dispute"),
    ("gold-012", 4, 5, 4, 4, "Provides clear rebooking procedure via DM"),
    ("gold-013", 4, 4, 4, 4, "Apologizes for captain rudeness, routes to customer relations"),
    ("gold-014", 5, 5, 4, 5, "Refuses immediate $1350 cash promise, routes to claims"),
    ("gold-015", 4, 5, 4, 4, "Explains wheelchair assistance procedure at gate"),
    ("gold-016", 5, 5, 4, 5, "Escalates app checkout payment freeze dispute"),
    ("gold-017", 4, 4, 4, 4, "Accurately references cabin pet/service animal rules"),
    ("gold-018", 4, 4, 4, 4, "Explains bird strike uncontrollable event policy"),
    ("gold-019", 5, 5, 4, 5, "Escalates $3000 stolen jewelry luggage claim"),
    ("gold-020", 4, 4, 4, 4, "Explains calendar year AAdvantage tier reset"),
    ("gold-021", 5, 5, 5, 5, "Urgent escalation for severe tarmac heat distress"),
    ("gold-022", 4, 4, 4, 4, "Clarifies basic economy change fee restrictions"),
    ("gold-023", 5, 5, 5, 5, "Escalates severe crew coffee spill complaint"),
    ("gold-024", 4, 4, 4, 4, "Responds to flight 908 status inquiry in Spanish"),
    ("gold-025", 5, 5, 4, 4, "Standard weather delay rebooking DM request"),
    ("gold-026", 4, 4, 4, 4, "Provides baggage carousel tracking guidance"),
    ("gold-027", 4, 5, 4, 4, "Directs to aa.com/refunds for flight disruption"),
    ("gold-028", 4, 4, 4, 4, "Explains seat selection change procedure"),
    ("gold-029", 4, 4, 4, 4, "Offers missing miles credit submission steps"),
    ("gold-030", 4, 4, 4, 4, "Guidance for mobile app boarding pass refresh"),
    ("gold-031", 4, 4, 3, 4, "Polite acknowledgment of in-flight WiFi failure"),
    ("gold-032", 4, 4, 4, 4, "Directs billing inquiry to reservations"),
    ("gold-033", 4, 4, 4, 4, "Polite general inquiry handling"),
    ("gold-034", 4, 5, 4, 4, "Clear missed connection rebooking steps"),
    ("gold-035", 4, 4, 4, 4, "Damaged luggage claim filing instruction"),
    ("gold-036", 4, 4, 4, 4, "Assists with stand-by flight rebooking rules"),
    ("gold-037", 5, 5, 4, 5, "Escalates supervisor demand regarding refund"),
    ("gold-038", 4, 4, 4, 4, "Clarifies AAdvantage upgrade certificate validity"),
    ("gold-039", 4, 4, 4, 4, "Gate priority boarding group explanation"),
    ("gold-040", 4, 4, 4, 4, "Apologizes for broken seat recliner"),
    ("gold-041", 4, 4, 4, 4, "Card charge pending authorization explanation"),
    ("gold-042", 4, 4, 5, 4, "Warm response to customer compliment"),
    ("gold-043", 4, 5, 4, 4, "Directs delayed passenger to airport meal vouchers"),
    ("gold-044", 4, 4, 4, 4, "Missing checked bag file reference locator search"),
    ("gold-045", 4, 4, 4, 4, "Offers flight change waiver details"),
    ("gold-046", 4, 4, 4, 4, "Hotel voucher policy explanation for mechanical delay"),
    ("gold-047", 4, 4, 4, 4, "AAdvantage partner airline mileage credit rules"),
    ("gold-048", 4, 4, 4, 4, "Online check-in passport requirement clarification"),
    ("gold-049", 4, 4, 3, 4, "Complaints handling for cold cabin temperature"),
    ("gold-050", 4, 4, 4, 4, "Assistance with international fare ticketing")
]


def create_human_audit_labels(
    golden_path: str = "data/golden_set.jsonl",
    output_path: str = "data/human_audit_50.jsonl"
):
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_items = {json.loads(line)["example_id"]: json.loads(line) for line in f}

    audit_records = []
    for ex_id, ground, correct, tone, complete, note in AUDIT_SCORES:
        if ex_id in golden_items:
            base = golden_items[ex_id]
            audit_records.append({
                "example_id": ex_id,
                "thread_id": base.get("thread_id"),
                "customer_message": base.get("customer_message"),
                "gold_intent": base.get("gold_intent"),
                "gold_decision": base.get("gold_decision"),
                "human_scores": {
                    "groundedness": ground,
                    "correctness": correct,
                    "tone_empathy": tone,
                    "completeness": complete,
                    "overall": round((ground + correct + tone + complete) / 4.0, 2)
                },
                "human_rater": "Human-Author",
                "human_note": note
            })

    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    with open(output_p, "w", encoding="utf-8") as f:
        for r in audit_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Created {len(audit_records)} human audit records in {output_p}")


if __name__ == "__main__":
    create_human_audit_labels()
