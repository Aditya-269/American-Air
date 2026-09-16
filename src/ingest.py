"""
Data ingestion and brand filtering pipeline for American Airlines (@AmericanAir).
Supports both Kaggle raw twcs.csv (chunked processing) and conversation parquet mirrors.
"""

import os
import re
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ingest")

BOILERPLATE_PATTERN = re.compile(
    r"(please\s+(dm|send\s+(us\s+)?a?\s*dm|direct\s+message)|"
    r"dm\s+(us\s+)?your\s+(record\s+locator|confirmation|booking\s+reference)|"
    r"send\s+us\s+a\s+direct\s+message|"
    r"follow\s+and\s+dm|"
    r"reach\s+out\s+via\s+dm)",
    re.IGNORECASE
)

MENTION_CLEAN_PATTERN = re.compile(r"(@americanair\b|@[0-9]+\b)", re.IGNORECASE)


def clean_text(text: str) -> str:
    """
    Clean tweet text by removing routing mentions (@AmericanAir, anonymized @12345)
    and normalizing whitespace while preserving semantic punctuation and emojis.
    """
    if not isinstance(text, str):
        return ""
    # Remove routing mentions
    text = MENTION_CLEAN_PATTERN.sub("", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_boilerplate_reply(text: str) -> bool:
    """Flag boilerplate replies asking for DM/record locator."""
    if not text:
        return False
    return bool(BOILERPLATE_PATTERN.search(text))


def parse_conversation_turns(conv_str: str) -> List[Dict[str, str]]:
    """
    Parse a conversation block into role-tagged turns.
    Expected format lines: 'Customer: ...' and 'Support: ...'
    """
    turns = []
    current_role = None
    current_lines = []

    for line in conv_str.splitlines():
        line = line.strip()
        if not line:
            continue
        lower_line = line.lower()
        if lower_line.startswith("customer:"):
            if current_role and current_lines:
                turns.append({"role": current_role, "text": clean_text(" ".join(current_lines))})
            current_role = "customer"
            current_lines = [line[len("customer:"):].strip()]
        elif lower_line.startswith("support:"):
            if current_role and current_lines:
                turns.append({"role": current_role, "text": clean_text(" ".join(current_lines))})
            current_role = "support"
            current_lines = [line[len("support:"):].strip()]
        else:
            if current_lines:
                current_lines.append(line)

    if current_role and current_lines:
        turns.append({"role": current_role, "text": clean_text(" ".join(current_lines))})

    return [t for t in turns if t["text"]]


def process_conversation_dataframe(df: pd.DataFrame, max_threads: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """
    Extract valid AmericanAir threads where customer first message and AA first reply are both present.
    """
    records = []
    for row in df.itertuples(index=False):
        conv_text = getattr(row, "conversation", "")
        thread_id = str(getattr(row, "conversation_id", f"tid_{len(records)}"))
        
        turns = parse_conversation_turns(conv_text)
        cust_turns = [t["text"] for t in turns if t["role"] == "customer"]
        sup_turns = [t["text"] for t in turns if t["role"] == "support"]

        # Keep only threads where customer first message and AA first reply are both present
        if not cust_turns or not sup_turns:
            continue

        cust_first = cust_turns[0]
        sup_first = sup_turns[0]

        # Minimum content filter
        if len(cust_first) < 8 or len(sup_first) < 8:
            continue

        records.append({
            "thread_id": thread_id,
            "customer_message": cust_first,
            "aa_reply": sup_first,
            "turns_count": len(turns),
            "num_customer_turns": len(cust_turns),
            "num_support_turns": len(sup_turns),
            "is_boilerplate": is_boilerplate_reply(sup_first),
            "raw_turns": turns
        })

    result_df = pd.DataFrame(records)
    logger.info(f"Extracted {len(result_df)} valid threads with customer opening and AA reply")

    if max_threads and len(result_df) > max_threads:
        result_df = result_df.sample(n=max_threads, random_state=seed).reset_index(drop=True)
        logger.info(f"Subsampled down to {len(result_df)} threads (seed={seed})")

    return result_df


def ingest_from_parquet(parquet_path: str, max_threads: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """Ingest AmericanAir threads from conversation parquet file."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    logger.info(f"Loading from parquet: {parquet_path}")
    table = pq.read_table(parquet_path)
    mask = pc.equal(table["company"], "AmericanAir")
    aa_table = table.filter(mask)
    logger.info(f"Found {len(aa_table)} AmericanAir threads in raw parquet")
    
    df = aa_table.to_pandas()
    return process_conversation_dataframe(df, max_threads=max_threads, seed=seed)


def ingest_from_csv(csv_path: str, max_threads: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """
    Ingest from Kaggle twcs.csv using chunked streaming.
    Does not load full 3M rows at once into memory.
    """
    logger.info(f"Loading raw Kaggle CSV in chunks from: {csv_path}")
    aa_tweets = []
    target_in_responses = set()

    # Pass 1: Find AmericanAir tweets
    for chunk in pd.read_csv(csv_path, chunksize=100_000, usecols=["tweet_id", "author_id", "in_response_to_tweet_id", "text", "created_at", "inbound"]):
        aa_subset = chunk[chunk["author_id"].astype(str).str.lower() == "americanair"]
        if not aa_subset.empty:
            aa_tweets.append(aa_subset)
            for parent_id in aa_subset["in_response_to_tweet_id"].dropna().unique():
                target_in_responses.add(parent_id)

    if not aa_tweets:
        logger.warning("No AmericanAir tweets found in CSV")
        return pd.DataFrame()

    all_aa = pd.concat(aa_tweets, ignore_index=True)
    logger.info(f"Found {len(all_aa)} AA tweets. Looking for {len(target_in_responses)} customer parent tweets...")

    # Pass 2: Find customer parent tweets
    customer_tweets = []
    for chunk in pd.read_csv(csv_path, chunksize=100_000, usecols=["tweet_id", "author_id", "text", "created_at", "inbound"]):
        matched = chunk[chunk["tweet_id"].isin(target_in_responses)]
        if not matched.empty:
            customer_tweets.append(matched)

    all_cust = pd.concat(customer_tweets, ignore_index=True)
    logger.info(f"Found {len(all_cust)} customer opening tweets")

    # Pair customer first tweet with AA first reply
    merged = pd.merge(
        all_cust,
        all_aa,
        left_on="tweet_id",
        right_on="in_response_to_tweet_id",
        suffixes=("_customer", "_aa")
    )

    records = []
    for _, row in merged.iterrows():
        cust_txt = clean_text(str(row["text_customer"]))
        aa_txt = clean_text(str(row["text_aa"]))
        if len(cust_txt) < 8 or len(aa_txt) < 8:
            continue
        records.append({
            "thread_id": f"tw_{row['tweet_id_customer']}",
            "customer_message": cust_txt,
            "aa_reply": aa_txt,
            "turns_count": 2,
            "num_customer_turns": 1,
            "num_support_turns": 1,
            "is_boilerplate": is_boilerplate_reply(aa_txt),
            "raw_turns": [
                {"role": "customer", "text": cust_txt},
                {"role": "support", "text": aa_txt}
            ]
        })

    result_df = pd.DataFrame(records)
    if max_threads and len(result_df) > max_threads:
        result_df = result_df.sample(n=max_threads, random_state=seed).reset_index(drop=True)
    return result_df


def run_ingest(
    input_path: Optional[str] = None,
    output_parquet: str = "data/processed/aa_threads.parquet",
    output_jsonl: str = "data/processed/aa_threads.jsonl",
    max_threads: int = 25000,
    seed: int = 42
) -> pd.DataFrame:
    """Main execution function for ingestion pipeline."""
    # Auto-detect input path if not specified
    if not input_path:
        if os.path.exists("data/raw/twitter_support.parquet"):
            input_path = "data/raw/twitter_support.parquet"
        elif os.path.exists("data/raw/twcs.csv"):
            input_path = "data/raw/twcs.csv"
        else:
            raise FileNotFoundError("Neither data/raw/twitter_support.parquet nor data/raw/twcs.csv found.")

    logger.info(f"Running ingestion with input={input_path}, max_threads={max_threads}")
    if input_path.endswith(".parquet"):
        df = ingest_from_parquet(input_path, max_threads=max_threads, seed=seed)
    else:
        df = ingest_from_csv(input_path, max_threads=max_threads, seed=seed)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    
    # Save parquet (for fast vectorized reading)
    # Note: store raw_turns as json string for parquet compatibility
    export_df = df.copy()
    export_df["raw_turns_json"] = export_df["raw_turns"].apply(json.dumps)
    export_df_save = export_df.drop(columns=["raw_turns"])
    export_df_save.to_parquet(output_parquet, index=False)
    logger.info(f"Saved {len(export_df_save)} threads to {output_parquet}")

    # Save jsonl
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(df)} threads to {output_jsonl}")

    # Log summary statistics
    avg_turns = df["turns_count"].mean()
    boilerplate_pct = (df["is_boilerplate"].sum() / len(df)) * 100.0 if len(df) > 0 else 0.0
    avg_cust_len = df["customer_message"].apply(lambda x: len(x.split())).mean()
    avg_reply_len = df["aa_reply"].apply(lambda x: len(x.split())).mean()

    stats_summary = (
        f"\n==================== INGESTION SUMMARY ====================\n"
        f"Total Filtered AA Threads : {len(df):,}\n"
        f"AA Reply Rate            : 100.0% (guaranteed by filter)\n"
        f"Average Turns Per Thread : {avg_turns:.2f}\n"
        f"Boilerplate Reply Rate   : {boilerplate_pct:.2f}% (flagged for audit)\n"
        f"Avg Customer Msg Words   : {avg_cust_len:.1f}\n"
        f"Avg AA Reply Words       : {avg_reply_len:.1f}\n"
        f"Output Parquet Path      : {output_parquet}\n"
        f"Output JSONL Path        : {output_jsonl}\n"
        f"==========================================================="
    )
    logger.info(stats_summary)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and filter AmericanAir conversation threads.")
    parser.add_argument("--input", type=str, default=None, help="Path to input parquet or twcs.csv")
    parser.add_argument("--output-parquet", type=str, default="data/processed/aa_threads.parquet")
    parser.add_argument("--output-jsonl", type=str, default="data/processed/aa_threads.jsonl")
    parser.add_argument("--max-threads", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_ingest(
        input_path=args.input,
        output_parquet=args.output_parquet,
        output_jsonl=args.output_jsonl,
        max_threads=args.max_threads,
        seed=args.seed
    )
