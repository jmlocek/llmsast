#!/usr/bin/env python3
"""
create_rag_db.py – Tworzenie bazy RAG w Qdrant z ostatnich N rekordów DiverseVul.

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

# ---------------------------------------------------------------------------
# KONFIGURACJA (domyślne wartości, nadpisywane przez .env lub CLI)
# ---------------------------------------------------------------------------

load_dotenv()

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
MAX_EMBED_CHARS = 32000  # limit znaków dla embeddingu

# ---------------------------------------------------------------------------
# INICJALIZACJA KLIENTÓW
# ---------------------------------------------------------------------------


def init_qdrant(url: str) -> QdrantClient:
    """Łączy się z Qdrant i weryfikuje połączenie."""
    print(f"[Init] Łączenie z Qdrant: {url} ...")
    try:
        client = QdrantClient(url=url)
        client.get_collections()  # szybki test
        print("[Init] Połączono z Qdrant.")
        return client
    except Exception as e:
        print(f"\n❌ BŁĄD POŁĄCZENIA Z QDRANT: {e}")
        print("Upewnij się, że kontener działa: podman ps | grep qdrant")
        sys.exit(1)


def init_embedding_client() -> OpenAI:
    """Tworzy klienta OpenAI-compatible do embeddingów (LM Studio)."""
    # Usuwamy prefix 'Bearer ' jeśli jest (OpenAI client sam go doda)
    api_key = LM_STUDIO_KEY
    if api_key.lower().startswith("bearer "):
        api_key = api_key.split(" ", 1)[1]

    http_client = httpx.Client(verify=False, timeout=120.0)
    client = OpenAI(base_url=LM_STUDIO_URL, api_key=api_key, http_client=http_client)

    print(f"[Init] Embedding endpoint: {LM_STUDIO_URL}")
    print(f"[Init] Embedding model: {EMBEDDING_MODEL_ID}")
    print(f"[Init] Expected vector size: {VECTOR_SIZE}")
    return client


# ---------------------------------------------------------------------------
# FUNKCJE POMOCNICZE
# ---------------------------------------------------------------------------


def get_embedding(client: OpenAI, text: str) -> list[float]:
    """Generuje embedding dla tekstu używając LM Studio."""
    safe_text = text[:MAX_EMBED_CHARS].replace("\n", " ")
    response = client.embeddings.create(input=[safe_text], model=EMBEDDING_MODEL_ID)
    return response.data[0].embedding


def validate_embedding(client: OpenAI) -> None:
    """Waliduje, że model embeddingów działa i zwraca właściwy rozmiar wektora."""
    print("[Validation] Sprawdzanie modelu embeddingów...")
    try:
        probe = get_embedding(client, "probe test")
        if len(probe) != VECTOR_SIZE:
            print(f"❌ Nieprawidłowy rozmiar wektora: {len(probe)} (oczekiwano {VECTOR_SIZE})")
            print(f"   Sprawdź EMBEDDING_MODEL_ID={EMBEDDING_MODEL_ID}")
            sys.exit(1)
        print(f"[Validation] OK – wektor ma {len(probe)} wymiarów.")
    except Exception as e:
        print(f"❌ Błąd embeddingu: {e}")
        sys.exit(1)


def deterministic_point_id(record: dict) -> str:
    """Tworzy deterministyczny ID punktu na podstawie pól rekordu."""
    # Używamy project + commit_id + hash (unikalne dla rekordu)
    key = f"{record.get('project', '')}|{record.get('commit_id', '')}|{record.get('hash', '')}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def load_last_n_records(input_path: str, n: int) -> tuple[list[dict], int]:
    """
    Wczytuje ostatnie N rekordów z pliku JSONL (tryb strumieniowy).
    Zwraca (lista rekordów, całkowita liczba linii w pliku).
    """
    print(f"[Data] Zliczanie rekordów w {input_path}...")
    total_lines = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1

    print(f"[Data] Plik zawiera {total_lines} rekordów.")

    if total_lines <= n:
        print(f"⚠️  Plik ma mniej niż {n} rekordów – wczytuję wszystkie.")
        start_line = 0
    else:
        start_line = total_lines - n

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= start_line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip malformed lines

    print(f"[Data] Wczytano {len(records)} rekordów do RAG (tail).")
    return records, total_lines


def setup_collection(client: QdrantClient, name: str, recreate: bool) -> None:
    """Tworzy lub odtwarza kolekcję w Qdrant."""
    exists = client.collection_exists(name)

    if recreate and exists:
        print(f"[Qdrant] Usuwanie istniejącej kolekcji '{name}'...")
        client.delete_collection(name)
        exists = False

    if not exists:
        print(f"[Qdrant] Tworzenie kolekcji '{name}' (size={VECTOR_SIZE}, distance=COSINE)...")
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
    else:
        print(f"[Qdrant] Kolekcja '{name}' już istnieje – dopisuję punkty (upsert).")


def build_payload(record: dict, origin_line: int) -> dict:
    """Buduje payload dla punktu Qdrant."""
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
    """Zapisuje manifest i listę ID rekordów RAG."""
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
    print(f"[Artifact] Zapisano manifest: {manifest_path}")

    ids_path = output_dir / "rag_train_ids.json"
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(rag_ids, f)
    print(f"[Artifact] Zapisano listę ID ({len(rag_ids)}): {ids_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Tworzenie bazy RAG w Qdrant z ostatnich N rekordów DiverseVul."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Ścieżka do pliku JSONL (domyślnie: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--rag-size",
        type=int,
        default=DEFAULT_RAG_SIZE,
        help=f"Liczba ostatnich rekordów do RAG (domyślnie: {DEFAULT_RAG_SIZE})",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=QDRANT_URL,
        help=f"URL Qdrant (domyślnie: {QDRANT_URL})",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help=f"Nazwa kolekcji Qdrant (domyślnie: {COLLECTION_NAME})",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Usuń i utwórz kolekcję od nowa",
    )
    parser.add_argument(
        "--max-fail-rate",
        type=float,
        default=0.05,
        help="Maksymalny dopuszczalny odsetek błędów (domyślnie: 0.05 = 5%%)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TWORZENIE BAZY RAG – DiverseVul → Qdrant")
    print("=" * 60 + "\n")

    # 1. Inicjalizacja klientów
    qdrant = init_qdrant(args.qdrant_url)
    embed_client = init_embedding_client()
    validate_embedding(embed_client)

    # 2. Wczytanie danych
    records, total_in_file = load_last_n_records(args.input, args.rag_size)
    if not records:
        print("❌ Brak rekordów do zaindeksowania.")
        sys.exit(1)

    # 3. Setup kolekcji
    setup_collection(qdrant, args.collection, args.recreate)

    # 4. Indeksowanie
    print(f"\n[Indexing] Indeksowanie {len(records)} rekordów (batch={BATCH_SIZE})...")

    points_batch: list[models.PointStruct] = []
    indexed_ids: list[str] = []
    indexed_count = 0
    skipped_count = 0

    # origin_line = numer linii w pliku (1-based), dla rekordów z tail
    start_line_in_file = total_in_file - len(records) + 1

    for i, record in enumerate(tqdm(records, desc="Embedding + upsert")):
        origin_line = start_line_in_file + i
        try:
            code = record.get("func", "")
            if not code:
                skipped_count += 1
                continue

            vector = get_embedding(embed_client, code)
            point_id = deterministic_point_id(record)

            point = models.PointStruct(
                id=point_id,
                vector=vector,
                payload=build_payload(record, origin_line),
            )
            points_batch.append(point)
            indexed_ids.append(point_id)
            indexed_count += 1

            if len(points_batch) >= BATCH_SIZE:
                qdrant.upsert(collection_name=args.collection, points=points_batch)
                points_batch = []

        except Exception as e:
            skipped_count += 1
            # opcjonalnie: tqdm.write(f"⚠️  Błąd rekordu {origin_line}: {e}")
            continue

    # Ostatni batch
    if points_batch:
        qdrant.upsert(collection_name=args.collection, points=points_batch)

    # 5. Podsumowanie
    print("\n" + "-" * 40)
    print(f"✅ Zaindeksowano: {indexed_count}")
    print(f"⚠️  Pominięto:     {skipped_count}")

    fail_rate = skipped_count / len(records) if records else 0
    if fail_rate > args.max_fail_rate:
        print(f"❌ Odsetek błędów ({fail_rate:.1%}) przekracza próg ({args.max_fail_rate:.0%})!")
        sys.exit(1)

    # 6. Zapisz artefakty
    output_dir = Path(args.input).parent.parent / "rag"
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

    # 7. Weryfikacja kolekcji
    info = qdrant.get_collection(args.collection)
    print(f"\n[Qdrant] Kolekcja '{args.collection}' – punktów: {info.points_count}")
    print("\n🎉 SUKCES!")


if __name__ == "__main__":
    main()
