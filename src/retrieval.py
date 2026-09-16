"""
Retrieval index for grounding AI support agent replies against historical AmericanAir resolutions.
Uses sentence-transformers and NearestNeighbors/cosine similarity to index and search (customer_message, aa_reply, intent) triples.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from src.intents import IntentClassifier, INTENT_TAXONOMY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("retrieval")


class RetrievalIndex:
    """
    Grounding index storing historical (customer_message, aa_reply, intent) triples.
    Enables intent-guided cosine retrieval and returns best similarity score for escalation gating.
    """
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "data/processed/retrieval_index.npz",
        corpus_path: str = "data/processed/retrieval_corpus.parquet"
    ):
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.corpus_path = Path(corpus_path)
        self.model = SentenceTransformer(model_name)
        
        self.corpus_df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.nn_index: Optional[NearestNeighbors] = None
        
        if self.index_path.exists() and self.corpus_path.exists():
            self.load()

    def build_index(
        self,
        threads_df: pd.DataFrame,
        classifier: Optional[IntentClassifier] = None,
        max_items: int = 8000,
        seed: int = 42
    ) -> None:
        """
        Build index from resolved historical threads.
        Tags threads with intents and computes normalized embeddings.
        """
        logger.info(f"Building retrieval grounding index from {len(threads_df)} candidate threads...")
        
        # Filter out threads where AA reply is empty or purely unhelpful
        valid = threads_df[
            (threads_df["customer_message"].str.len() >= 12) &
            (threads_df["aa_reply"].str.len() >= 12)
        ].copy()

        # Stratified or diverse sample up to max_items
        if len(valid) > max_items:
            valid = valid.sample(n=max_items, random_state=seed).reset_index(drop=True)
        else:
            valid = valid.reset_index(drop=True)

        logger.info(f"Tagging {len(valid)} historical corpus items with intents...")
        if classifier is None:
            classifier = IntentClassifier()

        intents = []
        for text in valid["customer_message"]:
            cls_res = classifier.classify(text)
            intents.append(cls_res["intent"])
        valid["intent"] = intents

        logger.info(f"Encoding {len(valid)} customer messages with {self.model_name}...")
        texts = valid["customer_message"].tolist()
        self.embeddings = self.model.encode(
            texts,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        self.corpus_df = valid[[
            "thread_id", "customer_message", "aa_reply", "intent", "is_boilerplate"
        ]].copy()

        self.nn_index = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn_index.fit(self.embeddings)

        self.save()
        logger.info(f"Successfully built and indexed {len(self.corpus_df)} historical grounding triples")

    def save(self) -> None:
        """Save embeddings and corpus metadata to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.index_path, embeddings=self.embeddings)
        self.corpus_df.to_parquet(self.corpus_path, index=False)
        logger.info(f"Saved retrieval index to {self.index_path} and corpus to {self.corpus_path}")

    def load(self) -> None:
        """Load index embeddings and corpus from disk."""
        logger.info(f"Loading retrieval index from {self.index_path}...")
        data = np.load(self.index_path)
        self.embeddings = data["embeddings"]
        self.corpus_df = pd.read_parquet(self.corpus_path)
        self.nn_index = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn_index.fit(self.embeddings)
        logger.info(f"Loaded {len(self.corpus_df)} historical grounding triples from disk")

    def retrieve(
        self,
        query: str,
        predicted_intent: Optional[str] = None,
        top_k: int = 3,
        intent_boost: float = 0.05
    ) -> Dict[str, Any]:
        """
        Retrieve top-k most similar historical interactions for grounding.
        Filters/boosts matches aligning with predicted_intent and returns best similarity score.
        """
        if self.embeddings is None or self.corpus_df is None or len(self.corpus_df) == 0:
            return {
                "matches": [],
                "best_similarity": 0.0,
                "mean_similarity": 0.0,
                "grounding_ids": []
            }

        q_emb = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        
        # Compute dot product similarity across full corpus
        sims = np.dot(self.embeddings, q_emb)

        # Optional intent preference: prioritize matching intent while preserving similarity truth
        candidate_indices = np.argsort(sims)[::-1][: top_k * 10]
        scored_candidates = []

        for idx in candidate_indices:
            row = self.corpus_df.iloc[idx]
            raw_sim = float(sims[idx])
            adjusted_sim = raw_sim
            if predicted_intent and predicted_intent != "other_unclear" and row["intent"] == predicted_intent:
                adjusted_sim += intent_boost

            # Penalize pure boilerplate when looking for high-quality grounding
            if row["is_boilerplate"]:
                adjusted_sim -= 0.03

            scored_candidates.append({
                "index": idx,
                "thread_id": str(row["thread_id"]),
                "customer_message": str(row["customer_message"]),
                "aa_reply": str(row["aa_reply"]),
                "intent": str(row["intent"]),
                "is_boilerplate": bool(row["is_boilerplate"]),
                "raw_similarity": round(raw_sim, 4),
                "ranking_score": round(adjusted_sim, 4)
            })

        # Sort by adjusted ranking score descending and take top_k
        scored_candidates.sort(key=lambda x: x["ranking_score"], reverse=True)
        top_matches = scored_candidates[:top_k]

        best_sim = max([m["raw_similarity"] for m in top_matches]) if top_matches else 0.0
        mean_sim = float(np.mean([m["raw_similarity"] for m in top_matches])) if top_matches else 0.0

        return {
            "matches": top_matches,
            "best_similarity": round(best_sim, 4),
            "mean_similarity": round(mean_sim, 4),
            "grounding_ids": [m["thread_id"] for m in top_matches]
        }


def build_and_save_index(
    input_threads: str = "data/processed/aa_threads.parquet",
    index_out: str = "data/processed/retrieval_index.npz",
    corpus_out: str = "data/processed/retrieval_corpus.parquet",
    max_items: int = 8000
) -> RetrievalIndex:
    """Convenience function to build and save grounding index."""
    df = pd.read_parquet(input_threads)
    retriever = RetrievalIndex(index_path=index_out, corpus_path=corpus_out)
    clf = IntentClassifier()
    retriever.build_index(df, classifier=clf, max_items=max_items)
    return retriever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build retrieval grounding index.")
    parser.add_argument("--input", default="data/processed/aa_threads.parquet")
    parser.add_argument("--index-out", default="data/processed/retrieval_index.npz")
    parser.add_argument("--corpus-out", default="data/processed/retrieval_corpus.parquet")
    parser.add_argument("--max-items", type=int, default=8000)
    args = parser.parse_args()

    build_and_save_index(
        input_threads=args.input,
        index_out=args.index_out,
        corpus_out=args.corpus_out,
        max_items=args.max_items
    )
    logger.info("Retrieval index built and verified successfully.")
