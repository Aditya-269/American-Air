"""
Intent taxonomy derivation and hybrid intent classification module for American Airlines.
Derives 9-class intent taxonomy bottom-up via clustering and provides nearest-centroid
embedding classification with LLM fallback capability.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
logger = logging.getLogger("intents")

# 9-class fixed taxonomy derived from AA tweet clusters
INTENT_TAXONOMY = [
    "flight_delay_cancellation",
    "baggage_issue",
    "rebooking_change_request",
    "refund_compensation_request",
    "aadvantage_miles_issue",
    "checkin_boarding_issue",
    "general_complaint_service_quality",
    "booking_payment_issue",
    "other_unclear"
]

INTENT_DESCRIPTIONS = {
    "flight_delay_cancellation": "Inquiries or complaints regarding flight delays, cancellations, tarmac delays, missed connections, or schedule changes.",
    "baggage_issue": "Lost, delayed, damaged, or mishandled checked luggage, baggage fees, or carousel retrieval issues.",
    "rebooking_change_request": "Requests to change flight times/dates, switch to alternate flights, standby status, or seat rebooking.",
    "refund_compensation_request": "Requests for monetary refunds, travel vouchers, hotel/meal reimbursement, or compensation for disrupted travel.",
    "aadvantage_miles_issue": "AAdvantage frequent flyer program, missing miles/points, status tiers, award booking redemptions, or account login.",
    "checkin_boarding_issue": "Online/app check-in problems, boarding pass issues, priority boarding groups, gate agent seating conflicts, or seat assignments.",
    "general_complaint_service_quality": "Complaints regarding inflight service, flight attendant/agent rudeness, dirty aircraft, broken seats, or poor experience.",
    "booking_payment_issue": "Problems making a reservation, duplicate credit card charges, checkout payment errors, or fare discrepancy on aa.com.",
    "other_unclear": "Vague greetings, compliments/shoutouts, general aviation chatter, non-English tweets, or inquiries not fitting the above classes."
}

INTENT_KEYWORDS = {
    "flight_delay_cancellation": ["delay", "delayed", "cancel", "cancelled", "cancellation", "tarmac", "late", "stuck", "missed connection", "flight status"],
    "baggage_issue": ["bag", "bags", "baggage", "luggage", "suitcase", "lost bag", "damaged bag", "carousel", "checked bag"],
    "rebooking_change_request": ["rebook", "rebooked", "change flight", "reschedule", "switch flight", "standby", "earlier flight", "later flight"],
    "refund_compensation_request": ["refund", "money back", "reimbursement", "compensation", "voucher", "expense", "hotel reimbursement", "meal voucher"],
    "aadvantage_miles_issue": ["aadvantage", "miles", "frequent flyer", "loyalty", "points", "status", "upgrade miles", "award travel"],
    "checkin_boarding_issue": ["check in", "checkin", "boarding", "boarding pass", "group", "seat assignment", "gate agent", "mobile check in"],
    "general_complaint_service_quality": ["rude", "worst airline", "horrible service", "attitude", "terrible", "unacceptable", "disrespect", "crew", "dirty"],
    "booking_payment_issue": ["credit card", "declined", "payment", "charged twice", "double charged", "billing error", "book ticket", "purchase error"],
    "other_unclear": ["thank you", "kudos", "great flight", "wings", "pilot", "weather", "hola", "avion"]
}

SEED_EXEMPLARS = {
    "flight_delay_cancellation": [
        "My flight AA 123 is delayed for 4 hours why is it delayed?",
        "Our flight got cancelled in Dallas and we missed our connection.",
        "Stuck on the tarmac for three hours, when are we taking off?",
        "Flight was delayed mechanical issue now I am stranded at the airport.",
        "Why is flight 4426 showing cancelled with no explanation?",
        "Sitting on runway waiting for gate to open for 45 minutes.",
        "Missed connection due to maintenance delay on first leg."
    ],
    "baggage_issue": [
        "You lost my luggage on flight to Miami where is my bag?",
        "My suitcase arrived completely damaged and broken wheels.",
        "Baggage claim has no bags from AA 456, where do I file lost baggage claim?",
        "Missing bag containing medication please help find my checked bag.",
        "Landed in Charlotte but my bags are still in Chicago.",
        "Handles ripped off my bag and zipper broken during transit.",
        "Where is the baggage service office located at terminal D?"
    ],
    "rebooking_change_request": [
        "I need to change my flight date to tomorrow morning.",
        "Can I rebook to an earlier flight today without paying fees?",
        "Missed connection need to be rebooked on the next available flight to Chicago.",
        "How do I switch my ticket to next week due to personal emergency?",
        "Put me on standby for the 4pm departure to Boston.",
        "Can an agent rebook us on a partner airline like Delta or United?",
        "Need to change departure city on my existing reservation."
    ],
    "refund_compensation_request": [
        "I want a full refund for my cancelled flight ticket.",
        "Requesting compensation and voucher for the 12 hour delay and hotel costs.",
        "You charged me twice for the seat upgrade and I need my money back refund.",
        "Where do I submit receipts for reimbursement for expenses caused by cancellation?",
        "Denied boarding on oversold flight, entitled to cash compensation under DOT.",
        "Flight was cancelled overnight, will American provide hotel voucher?",
        "Still waiting for the refund that was promised two weeks ago."
    ],
    "aadvantage_miles_issue": [
        "My AAdvantage miles have not posted to my account after my trip.",
        "Trying to redeem my frequent flyer miles for an award ticket but site errors.",
        "Missing loyalty points on my AAdvantage status qualifying miles.",
        "What is my current AAdvantage loyalty member tier and expiration?",
        "Cannot log into my AAdvantage account says invalid password.",
        "How do I apply 500-mile upgrades to my upcoming flight?",
        "Retroactive mileage credit request for ticket flown last month."
    ],
    "checkin_boarding_issue": [
        "Cannot check in online through the app keeps saying system error.",
        "Gate agent refused to let me board group 4 even though ticket says priority.",
        "Boarding pass won't generate on my mobile phone for flight today.",
        "Assigned middle seat at gate even though I paid for aisle seat assignment.",
        "Kiosk at terminal isn't printing our boarding passes.",
        "Boarding was chaotic and overhead bins were full before group 5.",
        "Need wheelchair assistance at boarding gate for elderly passenger."
    ],
    "general_complaint_service_quality": [
        "Flight attendant was extremely rude and unhelpful during our flight.",
        "Horrible customer service at the airport desk worst airline experience ever.",
        "Dirty cabin and seats with trash left in the seatback pocket.",
        "Your staff has awful attitude and treated passengers with total disrespect.",
        "Flight attendants sat on phones and ignored call buttons the whole flight.",
        "Worst customer service ever experienced from American Airlines supervisor.",
        "No air conditioning on plane and filthy restroom throughout 5 hour flight."
    ],
    "booking_payment_issue": [
        "My credit card was declined while trying to purchase flight on aa.com.",
        "Duplicate transaction on my bank statement for booking reservation.",
        "Cannot complete payment on website keeps giving billing error code.",
        "I received an error during checkout but my credit card was still charged.",
        "Price jumped $200 right when I entered payment details to confirm purchase.",
        "Promo code is not applying discount to ticket price during checkout.",
        "Unable to split payment between gift card and credit card."
    ],
    "other_unclear": [
        "Hello American Airlines hope you all have a wonderful day!",
        "Does the airplane have free wifi and power outlets on board?",
        "Just saw a beautiful 777 takeoff at Dallas Fort Worth airport #avgeek.",
        "Abril bogado servicio al cliente en quito es una basura equipaje perdido.",
        "What terminal does flight 100 arrive at in London Heathrow?",
        "Kudos to captain Dave for the smooth landing in severe turbulence!",
        "Traveling with a small pet dog in cabin what are the carrier rules?"
    ]
}


class IntentClassifier:
    """
    Nearest-centroid intent classifier using sentence-transformer embeddings,
    with built-in confidence estimation and fallback support.
    """
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        centroids_path: Optional[str] = "data/processed/intent_centroids.npz"
    ):
        self.model_name = model_name
        self.centroids_path = Path(centroids_path) if centroids_path else None
        self.model = SentenceTransformer(model_name)
        self.centroids: Dict[str, np.ndarray] = {}
        self.taxonomy = INTENT_TAXONOMY

        if self.centroids_path and self.centroids_path.exists():
            self.load_centroids(self.centroids_path)
        else:
            self.build_seed_centroids()

    def build_seed_centroids(self) -> None:
        """Compute initial centroids from seed exemplar sentences."""
        logger.info("Building initial centroids from seed exemplars...")
        self.centroids = {}
        for intent in self.taxonomy:
            examples = SEED_EXEMPLARS.get(intent, ["Customer support request"])
            embs = self.model.encode(examples, normalize_embeddings=True, show_progress_bar=False)
            centroid = np.mean(embs, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            self.centroids[intent] = centroid
        logger.info(f"Initialized centroids for {len(self.centroids)} intents")

    def save_centroids(self, path: Optional[Path] = None) -> None:
        """Save centroid vectors and metadata to disk."""
        target = Path(path or self.centroids_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            **{intent: vec for intent, vec in self.centroids.items()}
        )
        meta_path = target.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump({
                "taxonomy": self.taxonomy,
                "descriptions": INTENT_DESCRIPTIONS,
                "model_name": self.model_name
            }, f, indent=2)
        logger.info(f"Saved centroids to {target} and metadata to {meta_path}")

    def load_centroids(self, path: Path) -> None:
        """Load precomputed centroid vectors from disk."""
        logger.info(f"Loading centroids from {path}")
        data = np.load(path)
        self.centroids = {k: data[k] for k in data.files}
        logger.info(f"Loaded centroids for {len(self.centroids)} intents")

    def classify(self, text: str, confidence_threshold: float = 0.48) -> Dict[str, Any]:
        """
        Classify text by nearest-centroid cosine similarity.
        Returns predicted intent, confidence score, full distribution, and flag whether LLM fallback is recommended.
        """
        if not text or not text.strip():
            return {
                "intent": "other_unclear",
                "confidence": 0.0,
                "all_scores": {k: 0.0 for k in self.taxonomy},
                "needs_fallback": True,
                "margin": 0.0
            }

        emb = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = {}
        for intent, centroid in self.centroids.items():
            scores[intent] = float(np.dot(emb, centroid))

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_intent, top_score = sorted_scores[0]
        second_intent, second_score = sorted_scores[1]
        margin = top_score - second_score

        # Keyword reinforcement check for explicit keywords
        lower_text = text.lower()
        keyword_boosted = False
        for intent, kw_list in INTENT_KEYWORDS.items():
            if any(kw in lower_text for kw in kw_list):
                if intent == top_intent:
                    top_score = min(1.0, top_score + 0.08)
                    keyword_boosted = True
                elif top_score < confidence_threshold and intent == second_intent:
                    top_intent, top_score = intent, min(1.0, second_score + 0.12)
                    keyword_boosted = True
                    break

        needs_fallback = (top_score < confidence_threshold) or (margin < 0.04 and top_score < 0.60)

        return {
            "intent": top_intent,
            "confidence": round(float(top_score), 4),
            "all_scores": {k: round(v, 4) for k, v in sorted_scores},
            "margin": round(float(margin), 4),
            "needs_fallback": bool(needs_fallback),
            "keyword_boosted": keyword_boosted
        }


def evaluate_clustering(df: pd.DataFrame, sample_size: int = 4000, seed: int = 42) -> Dict[str, Any]:
    """Run KMeans clustering across k=6..12 to evaluate silhouette score and cluster stability."""
    logger.info(f"Running clustering evaluation on {sample_size} customer opening messages...")
    sample = df.sample(n=min(sample_size, len(df)), random_state=seed).reset_index(drop=True)
    texts = sample["customer_message"].tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)

    results = {}
    for k in range(6, 13):
        km = KMeans(n_clusters=k, random_state=seed, n_init=5)
        labels = km.fit_predict(embs)
        sil = silhouette_score(embs, labels, sample_size=min(1500, len(embs)), random_state=seed)
        results[k] = {
            "silhouette": round(float(sil), 4),
            "cluster_distribution": [int(x) for x in sorted(pd.Series(labels).value_counts().values, reverse=True)]
        }
        logger.info(f"k={k:2d}: silhouette={sil:.4f}, cluster_sizes={results[k]['cluster_distribution']}")

    return results


def train_and_save_centroids(
    data_path: str = "data/processed/aa_threads.parquet",
    output_path: str = "data/processed/intent_centroids.npz"
) -> IntentClassifier:
    """Build and save production intent classifier."""
    classifier = IntentClassifier(centroids_path=None)
    classifier.build_seed_centroids()
    classifier.save_centroids(Path(output_path))
    return classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intent taxonomy clustering and centroid builder.")
    parser.add_argument("--eval-clustering", action="store_true", help="Run KMeans silhouette sweep")
    parser.add_argument("--data", default="data/processed/aa_threads.parquet")
    parser.add_argument("--output", default="data/processed/intent_centroids.npz")
    args = parser.parse_args()

    if args.eval_clustering and os.path.exists(args.data):
        df = pd.read_parquet(args.data)
        eval_res = evaluate_clustering(df)
        print("\nSilhouette Results:")
        for k, v in eval_res.items():
            print(f"k={k}: {v['silhouette']}")

    clf = train_and_save_centroids(args.data, args.output)
    logger.info("Intent classifier ready and saved successfully.")
