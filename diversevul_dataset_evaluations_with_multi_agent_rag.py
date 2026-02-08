#!/usr/bin/env python3
"""
diversevul_dataset_evaluations_with_multi_agent_rag.py

Evaluation script for DiverseVul dataset using MULTI AGENT mode + RAG.
RAG context is injected into:
  - Vulnerability Hunter agent (Step 2)
  - FP Remover / Risk Verifier agent (Step 3)

For each sample, retrieves 1 nearest VULNERABLE + 1 nearest SAFE example
from Qdrant and attaches them to the prompts for agents 2 and 3.
"""

import argparse
import hashlib
import json
import os
import sys

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from agents_logic import CodeAgents

load_dotenv()

# Ensure UTF-8 output even when stdout/stderr are redirected on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# CONFIG (from create_rag_db.py / .env)
# ---------------------------------------------------------------------------
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "https://10.147.18.200:1234/v1")
LM_STUDIO_KEY = os.getenv("LM_STUDIO_KEY", "Bearer LLM-abcde")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "text-embedding-nomic-embed-text-v1.5")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "768"))
MAX_EMBED_CHARS = int(os.getenv("MAX_EMBED_CHARS", "32000"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "diversevul_kb")

# Evaluation defaults
DEFAULT_JSON_FILE = "datasets/diversevul_20230702.json"
DEFAULT_PROGRESS_FILE = "diversevul_multi_agent_rag_progress.json"
DEFAULT_RAG_TRAIN_IDS_FILE = "rag/rag_train_ids.json"
MAX_SAMPLES = 1100


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------
def extract_simple_verdict_and_report(text: str) -> dict:
    """
    Format:
      has_vulnerability: [true/false]
      [Report text...]
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Response is empty or not a string")

    newline_index = text.find("\n")
    if newline_index == -1:
        first_line = text.strip()
        report = "[No report found]"
    else:
        first_line = text[:newline_index].strip()
        report = text[newline_index + 1:].strip()

    key = "has_vulnerability:"
    if not first_line.lower().startswith(key):
        raise ValueError(f"Response missing '{key}' on the first line.")

    bool_str = first_line[len(key):].strip().lower()
    if bool_str == "true":
        has_vuln = True
    elif bool_str == "false":
        has_vuln = False
    else:
        raise ValueError(f"Invalid boolean value: '{bool_str}' found on first line.")

    return {"has_vulnerability": has_vuln, "report": report}


def save_progress(progress_file: str, data: dict) -> None:
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_progress(progress_file: str) -> dict | None:
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def deterministic_point_id(record: dict) -> str:
    """Must match create_rag_db.py: sha256(project|commit_id|hash)[:32]"""
    key = f"{record.get('project', '')}|{record.get('commit_id', '')}|{record.get('hash', '')}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def load_rag_train_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(str(x) for x in data) if isinstance(data, list) else set()


# ---------------------------------------------------------------------------
# EMBEDDING CLIENT (LM Studio / OpenAI-compatible)
# ---------------------------------------------------------------------------
def normalize_lm_studio_key(key: str) -> str:
    if key.lower().startswith("bearer "):
        return key.split(" ", 1)[1].strip()
    return key.strip()


def init_embedding_client() -> OpenAI:
    http_client = httpx.Client(verify=False, timeout=120.0)
    return OpenAI(
        base_url=LM_STUDIO_URL,
        api_key=normalize_lm_studio_key(LM_STUDIO_KEY),
        http_client=http_client,
    )


def get_embedding(client: OpenAI, text: str) -> list[float]:
    safe_text = (text or "")[:MAX_EMBED_CHARS].replace("\n", " ")
    resp = client.embeddings.create(model=EMBEDDING_MODEL_ID, input=safe_text)
    return resp.data[0].embedding


def validate_embedding(client: OpenAI) -> None:
    vec = get_embedding(client, "embedding-probe")
    if len(vec) != VECTOR_SIZE:
        raise RuntimeError(
            f"VECTOR_SIZE mismatch: got {len(vec)}, expected {VECTOR_SIZE}. "
            f"Check EMBEDDING_MODEL_ID/VECTOR_SIZE."
        )
    print(f"[Embedding] Validation OK (size={len(vec)})")


# ---------------------------------------------------------------------------
# QDRANT RAG RETRIEVAL (balanced: 1 vulnerable + 1 safe)
# ---------------------------------------------------------------------------
def init_qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def qdrant_vector_search(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    query_vector: list[float],
    query_filter: qdrant_models.Filter | None,
    limit: int,
    with_payload: bool = True,
):
    """Compatibility wrapper across qdrant-client versions."""
    if hasattr(qdrant, "search"):
        return qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
        )

    if hasattr(qdrant, "query_points"):
        resp = qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
        )
        return list(getattr(resp, "points", []) or [])

    raise AttributeError(
        "Unsupported qdrant-client version: expected QdrantClient.search or QdrantClient.query_points"
    )


def retrieve_balanced_context(
    qdrant: QdrantClient,
    embed_client: OpenAI,
    code: str,
    max_chars_per_example: int = 2000,
) -> str:
    """
    Retrieves 1 nearest VULNERABLE example + 1 nearest SAFE example from Qdrant.
    Uses filter on payload field `is_vulnerable`.
    """
    query_vec = get_embedding(embed_client, code)

    # Search for nearest vulnerable example
    vuln_hits = qdrant_vector_search(
        qdrant,
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vec,
        query_filter=qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="is_vulnerable",
                    match=qdrant_models.MatchValue(value=True),
                )
            ]
        ),
        limit=1,
        with_payload=True,
    )

    # Search for nearest safe example
    safe_hits = qdrant_vector_search(
        qdrant,
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vec,
        query_filter=qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="is_vulnerable",
                    match=qdrant_models.MatchValue(value=False),
                )
            ]
        ),
        limit=1,
        with_payload=True,
    )

    parts: list[str] = []

    # Format vulnerable example
    if vuln_hits:
        hit = vuln_hits[0]
        payload = hit.payload or {}
        ex_code = (payload.get("code") or "")[:max_chars_per_example]
        project = payload.get("project", "")
        cwe = payload.get("cwe", [])
        parts.append(
            f"=== Similar VULNERABLE example (score={hit.score:.3f}, project={project}, cwe={cwe}) ===\n"
            f"```c\n{ex_code}\n```"
        )

    # Format safe example
    if safe_hits:
        hit = safe_hits[0]
        payload = hit.payload or {}
        ex_code = (payload.get("code") or "")[:max_chars_per_example]
        project = payload.get("project", "")
        parts.append(
            f"=== Similar SAFE example (score={hit.score:.3f}, project={project}) ===\n"
            f"```c\n{ex_code}\n```"
        )

    return "\n\n".join(parts)


def build_rag_preamble_for_hunter(rag_context: str) -> str:
    """
    Builds RAG preamble for the Vulnerability Hunter agent.
    Emphasizes looking for real vulnerabilities despite similar safe examples.
    """
    if not rag_context:
        return ""
    
    return (
        "=== RAG CONTEXT (Reference Examples) ===\n"
        "Below are similar code examples from a knowledge base. One is VULNERABLE and one is SAFE.\n"
        "Use them to understand vulnerability patterns, but focus on finding REAL issues in the target code.\n"
        "Do NOT assume the code is safe just because a similar safe example exists.\n\n"
        f"{rag_context}\n\n"
        "=== END RAG CONTEXT ===\n\n"
    )


def build_rag_preamble_for_fp_remover(rag_context: str) -> str:
    """
    Builds RAG preamble for the FP Remover / Risk Verifier agent.
    Helps distinguish real vulnerabilities from false positives.
    """
    if not rag_context:
        return ""
    
    return (
        "=== RAG CONTEXT (Reference Examples) ===\n"
        "Below are similar code examples from a knowledge base. One is VULNERABLE and one is SAFE.\n"
        "Use them to help distinguish real vulnerabilities from false positives.\n"
        "Compare the vulnerability patterns - if the target code lacks the dangerous pattern, it may be a false positive.\n"
        "If the target code matches the vulnerable pattern, confirm it as a real vulnerability.\n\n"
        f"{rag_context}\n\n"
        "=== END RAG CONTEXT ===\n\n"
    )


# ---------------------------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------------------------
def evaluate_diversevul_dataset_with_multi_agent_rag(
    json_file: str,
    progress_file: str,
    rag_train_ids_file: str,
    max_samples: int,
) -> None:
    # Resume support
    progress = load_progress(progress_file)
    if progress:
        tp = progress.get("tp", 0)
        fp = progress.get("fp", 0)
        tn = progress.get("tn", 0)
        fn = progress.get("fn", 0)
        current_index = progress.get("current_index", 0)
        print("Resuming from saved progress...")
    else:
        tp = fp = tn = fn = 0
        current_index = 0
        print("Starting fresh...")

    print("--- Initializing agents ---")
    print("--- Running in MULTI AGENT + RAG mode ---")
    agents = CodeAgents()

    print("--- Initializing RAG (Qdrant + Embeddings) ---")
    qdrant = init_qdrant()
    embed_client = init_embedding_client()
    validate_embedding(embed_client)

    # Load IDs that are in RAG KB (to exclude from evaluation)
    rag_train_ids = load_rag_train_ids(rag_train_ids_file)
    if rag_train_ids:
        print(f"Loaded RAG train IDs: {len(rag_train_ids)} (will exclude from evaluation)")
    else:
        print(f"Warning: RAG train IDs not found/empty: {rag_train_ids_file}. Evaluation may overlap with RAG KB.")

    # Load and split dataset (excluding RAG records)
    print(f"Loading dataset from {json_file}...")
    vulnerable_functions: list[dict] = []
    safe_functions: list[dict] = []

    with open(json_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
                continue

            # Exclude records that were inserted into RAG KB
            pid = deterministic_point_id(data)
            if pid in rag_train_ids:
                continue

            target = data.get("target")
            if target == 1:
                vulnerable_functions.append(data)
            elif target == 0:
                safe_functions.append(data)

            # Stop early once we have enough candidates
            if len(vulnerable_functions) >= max_samples and len(safe_functions) >= max_samples:
                break

    vulnerable_functions = vulnerable_functions[:max_samples]
    safe_functions = safe_functions[:max_samples]

    total_vuln = len(vulnerable_functions)
    total_safe = len(safe_functions)
    total_samples = total_vuln + total_safe

    print("--- Successfully loaded dataset ---")
    print(f"Total vulnerable functions (limited): {total_vuln}")
    print(f"Total safe functions (limited): {total_safe}")
    print(f"Total samples to evaluate: {total_samples}")
    print(f"RAG collection: {QDRANT_COLLECTION}")

    # Process samples alternating between vulnerable and safe
    for idx in range(current_index, total_samples):
        if idx % 2 == 0:
            vuln_idx = idx // 2
            if vuln_idx >= total_vuln:
                print(f"\nSkipping index {idx + 1} (no more vulnerable functions)")
                progress = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "current_index": idx + 1}
                save_progress(progress_file, progress)
                continue
            func_data = vulnerable_functions[vuln_idx]
            is_vulnerable = True
            print(f"\n--- Processing Vulnerable Function {vuln_idx + 1}/{total_vuln} (Overall: {idx + 1}/{total_samples}) ---")
        else:
            safe_idx = idx // 2
            if safe_idx >= total_safe:
                print(f"\nSkipping index {idx + 1} (no more safe functions)")
                progress = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "current_index": idx + 1}
                save_progress(progress_file, progress)
                continue
            func_data = safe_functions[safe_idx]
            is_vulnerable = False
            print(f"\n--- Processing Safe Function {safe_idx + 1}/{total_safe} (Overall: {idx + 1}/{total_samples}) ---")

        code = func_data.get("func", "")
        if not isinstance(code, str) or not code.strip():
            print(f"Skipping sample {idx + 1} (no code found)")
            progress = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "current_index": idx + 1}
            save_progress(progress_file, progress)
            continue

        try:
            # Retrieve balanced RAG context (1 vulnerable + 1 safe)
            print("  Retrieving RAG context...")
            rag_context = retrieve_balanced_context(qdrant, embed_client, code)
            
            # Build RAG preambles for both Hunter and FP Remover
            rag_preamble_hunter = build_rag_preamble_for_hunter(rag_context)
            rag_preamble_fp = build_rag_preamble_for_fp_remover(rag_context)

            # ===== MULTI-AGENT PIPELINE =====
            
            # Step 1: Context Analysis (no RAG - just understands the code)
            print("  Step 1: Context Analysis...")
            context_summary = agents.analyze_code(code)

            # Step 2: Vulnerability Detection (WITH RAG)
            print("  Step 2: Vulnerability Detection (with RAG)...")
            augmented_code_for_hunter = rag_preamble_hunter + f"=== CODE TO ANALYZE ===\n```c\n{code}\n```"
            vuln_list_string = agents.find_vulnerabilities(
                context_summary=context_summary,
                input_code=augmented_code_for_hunter,
            )

            # Step 3: Risk Verification / FP Removal (WITH RAG)
            print("  Step 3: Risk Verification (with RAG)...")
            # Prepend RAG context to the input code for the FP remover
            augmented_code_for_fp = rag_preamble_fp + f"=== CODE TO ANALYZE ===\n```c\n{code}\n```"
            verdict_string = agents.verify_risk_and_fp(
                vuln_list=vuln_list_string,
                context_summary=context_summary,
                input_code=augmented_code_for_fp,
            )

            verdict_data = extract_simple_verdict_and_report(verdict_string)
            detected_vuln = verdict_data.get("has_vulnerability")

            if is_vulnerable:
                if detected_vuln is True:
                    tp += 1
                    print("Result: CORRECTLY detected vulnerability (TP)")
                elif detected_vuln is False:
                    fn += 1
                    print("Result: FAILED to detect vulnerability (FN)")
                else:
                    print("Parsed response did not contain a valid 'has_vulnerability' boolean value.")
            else:
                if detected_vuln is True:
                    fp += 1
                    print("Result: INCORRECTLY detected vulnerability (FP)")
                elif detected_vuln is False:
                    tn += 1
                    print("Result: CORRECTLY ignored safe code (TN)")
                else:
                    print("Parsed response did not contain a valid 'has_vulnerability' boolean value.")

        except Exception as e:
            print(f"An error occurred during function analysis: {e}", file=sys.stderr)

        # Save progress after each sample
        progress = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "current_index": idx + 1}
        save_progress(progress_file, progress)

    # --- Calculate and Print Final Metrics ---
    print("\n--- Evaluation Complete ---")

    total_positive_cases = tp + fn
    total_negative_cases = tn + fp

    if total_positive_cases == 0 and total_negative_cases == 0:
        print("No valid samples were processed.")
        return

    print(f"Total Vulnerable Samples Tested: {total_positive_cases}")
    print(f"Total Safe Samples Tested:       {total_negative_cases}")
    print("---")
    print(f"True Positives (TP):  {tp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print("---")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    positive_accuracy = tp / total_positive_cases if total_positive_cases > 0 else 0.0
    negative_accuracy = tn / total_negative_cases if total_negative_cases > 0 else 0.0
    overall_accuracy = (tp + tn) / (total_positive_cases + total_negative_cases)

    print(f"Overall Accuracy:  {overall_accuracy:.2%} (Correct classifications / Total cases)")
    print(f"Positive Accuracy: {positive_accuracy:.2%} (Recall / Sensitivity)")
    print(f"Negative Accuracy: {negative_accuracy:.2%} (Specificity)")
    print("---")
    print(f"Precision: {precision:.2%}  (How trustworthy are the 'vulnerable' alerts?)")
    print(f"Recall:    {recall:.2%}  (How many real bugs did we find?)")
    print(f"F1-Score:  {f1_score:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DiverseVul with MULTI AGENT + RAG (RAG injected into Hunter & FP Remover)"
    )
    parser.add_argument("--json-file", default=DEFAULT_JSON_FILE)
    parser.add_argument("--progress-file", default=DEFAULT_PROGRESS_FILE)
    parser.add_argument("--rag-train-ids-file", default=DEFAULT_RAG_TRAIN_IDS_FILE)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES, help="Per class (vuln and safe)")
    args = parser.parse_args()

    evaluate_diversevul_dataset_with_multi_agent_rag(
        json_file=args.json_file,
        progress_file=args.progress_file,
        rag_train_ids_file=args.rag_train_ids_file,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
