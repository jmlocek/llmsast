#!/usr/bin/env python3
"""create_rag_db.py – Tworzenie bazy RAG w Qdrant z ostatnich N rekordów DiverseVul.

Zgodne z wymaganiami: instructions/tworzenierag.md
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

load_dotenv()

# Ensure UTF-8 output even when stdout/stderr are redirected on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Embeddingi – LM Studio / OpenAI-compatible
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "https://10.147.18.200:1234/v1")
LM_STUDIO_KEY = os.getenv("LM_STUDIO_KEY", "Bearer LLM-abcde")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "text-embedding-nomic-embed-text-v1.5")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "768"))  # Nomic v1.5

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "diversevul_kb")

# Dane
DEFAULT_INPUT = "datasets/diversevul_20230702.json"
DEFAULT_RAG_SIZE = 20000
BATCH_SIZE = 50
MAX_EMBED_CHARS = 32000


def init_qdrant(url: str) -> QdrantClient:
    print(f"[Init] Connecting to Qdrant: {url} ...")
    try:
        client = QdrantClient(url=url)
        client.get_collections()
        print("[Init] Qdrant connected.")
        return client
    except Exception as e:
        print(f"ERROR: Cannot connect to Qdrant: {e}")
        sys.exit(1)


def init_embedding_client() -> OpenAI:
    api_key = LM_STUDIO_KEY
    if api_key.lower().startswith("bearer "):
        api_key = api_key.split(" ", 1)[1]

    http_client = httpx.Client(verify=False, timeout=120.0)
    client = OpenAI(base_url=LM_STUDIO_URL, api_key=api_key, http_client=http_client)

    print(f"[Init] Embedding endpoint: {LM_STUDIO_URL}")
    print(f"[Init] Embedding model: {EMBEDDING_MODEL_ID}")
    print(f"[Init] Expected vector size: {VECTOR_SIZE}")
    return client


def get_embedding(client: OpenAI, text: str) -> list[float]:
    safe_text = text[:MAX_EMBED_CHARS].replace("\n", " ")
    response = client.embeddings.create(input=[safe_text], model=EMBEDDING_MODEL_ID)
    return response.data[0].embedding


def validate_embedding(client: OpenAI) -> None:
    print("[Validation] Checking embedding model...")
    try:
        probe = get_embedding(client, "probe")
        if len(probe) != VECTOR_SIZE:
            print(f"ERROR: Wrong embedding vector size: {len(probe)} (expected {VECTOR_SIZE})")
            sys.exit(1)
        print(f"[Validation] OK (size={len(probe)}).")
    except Exception as e:
        print(f"ERROR: Embedding failed: {e}")
        sys.exit(1)


def deterministic_point_id(record: dict) -> str:
    key = f"{record.get('project', '')}|{record.get('commit_id', '')}|{record.get('hash', '')}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def load_last_n_records(input_path: str, n: int) -> tuple[list[dict], int]:
    print(f"[Data] Counting records in {input_path}...")
    total_lines = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1

    print(f"[Data] Total records: {total_lines}")

    start_line = 0 if total_lines <= n else (total_lines - n)

    records: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start_line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"[Data] Loaded tail records for RAG: {len(records)}")
    return records, total_lines


def setup_collection(client: QdrantClient, name: str, recreate: bool) -> None:
    exists = client.collection_exists(name)
    if recreate and exists:
        print(f"[Qdrant] Deleting existing collection '{name}'...")
        client.delete_collection(name)
        exists = False

    if not exists:
        print(f"[Qdrant] Creating collection '{name}' (size={VECTOR_SIZE}, distance=COSINE)...")
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
    else:
        print(f"[Qdrant] Collection '{name}' exists - upserting.")


def build_payload(record: dict, origin_line: int) -> dict:
    target_raw = record.get("target", 0)
    try:
        target = int(target_raw)
    except Exception:
        target = 0

    return {
        "code": record.get("func", ""),
        "target": target,
        "is_vulnerable": target == 1,
        "cwe": record.get("cwe", []),
        "project": record.get("project", ""),
        "commit_id": record.get("commit_id", ""),
        "hash": str(record.get("hash", "")),
        "size": record.get("size", 0),
        "message": record.get("message", ""),
        "origin_line": origin_line,
    }


def save_manifest(
    output_dir: Path,
    input_path: str,
    total_records: int,
    rag_size: int,
    collection: str,
    indexed: int,
    skipped: int,
    rag_ids: list[str],
) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input_file": input_path,
        "total_records_in_file": total_records,
        "rag_size_requested": rag_size,
        "records_indexed": indexed,
        "records_skipped": skipped,
        "collection_name": collection,
        "split_method": "tail (ostatnie N rekordów w kolejności pliku)",
        "embedding_model": EMBEDDING_MODEL_ID,
        "vector_size": VECTOR_SIZE,
    }

    manifest_path = output_dir / "rag_diversevul_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[Artifact] Manifest: {manifest_path}")

    ids_path = output_dir / "rag_train_ids.json"
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(rag_ids, f)
    print(f"[Artifact] IDs ({len(rag_ids)}): {ids_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tworzenie bazy RAG w Qdrant z ostatnich N rekordów DiverseVul."
    )
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--rag-size", type=int, default=DEFAULT_RAG_SIZE)
    parser.add_argument("--qdrant-url", type=str, default=QDRANT_URL)
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--max-fail-rate", type=float, default=0.05)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TWORZENIE BAZY RAG - DiverseVul -> Qdrant")
    print("=" * 60 + "\n")

    qdrant = init_qdrant(args.qdrant_url)
    embed_client = init_embedding_client()
    validate_embedding(embed_client)

    records, total_in_file = load_last_n_records(args.input, args.rag_size)
    if not records:
        print("ERROR: No records loaded.")
        sys.exit(1)

    setup_collection(qdrant, args.collection, args.recreate)

    print(f"[Indexing] Indexing {len(records)} records (batch={BATCH_SIZE})...")
    points_batch: list[models.PointStruct] = []
    indexed_ids: list[str] = []
    indexed_count = 0
    skipped_count = 0

    start_line_in_file = total_in_file - len(records) + 1

    for i, record in enumerate(tqdm(records, desc="Embedding + upsert", ascii=True)):
        origin_line = start_line_in_file + i
        try:
            code = record.get("func", "")
            if not code:
                skipped_count += 1
                continue

            vector = get_embedding(embed_client, code)
            point_id = deterministic_point_id(record)
            point = models.PointStruct(id=point_id, vector=vector, payload=build_payload(record, origin_line))
            points_batch.append(point)
            indexed_ids.append(point_id)
            indexed_count += 1

            if len(points_batch) >= BATCH_SIZE:
                qdrant.upsert(collection_name=args.collection, points=points_batch)
                points_batch = []
        except Exception:
            skipped_count += 1
            continue

    if points_batch:
        qdrant.upsert(collection_name=args.collection, points=points_batch)

    print("-" * 40)
    print(f"Indexed: {indexed_count}")
    print(f"Skipped: {skipped_count}")

    fail_rate = skipped_count / len(records) if records else 0
    if fail_rate > args.max_fail_rate:
        print(f"ERROR: Fail rate ({fail_rate:.1%}) exceeds threshold ({args.max_fail_rate:.0%}).")
        sys.exit(1)

    output_dir = Path("rag")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(
        output_dir=output_dir,
        input_path=args.input,
        total_records=total_in_file,
        rag_size=args.rag_size,
        collection=args.collection,
        indexed=indexed_count,
        skipped=skipped_count,
        rag_ids=indexed_ids,
    )

    info = qdrant.get_collection(args.collection)
    print(f"[Qdrant] points_count={info.points_count}")
    print("SUCCESS")


if __name__ == "__main__":
    main()