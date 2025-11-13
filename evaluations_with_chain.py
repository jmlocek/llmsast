import pandas as pd
import sys
import json
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents_logic import CodeAgents, AgentConfig
from llmsastchain import CodeAgents

def extract_json_from_string(text):
    """
    Extracts the first JSON object from a string that might be
    wrapped in Markdown (e.g., ```json ... ```) or other text.
    """
    if not isinstance(text, str):
        raise json.JSONDecodeError("Input was not a string", str(text), 0)
        
    try:
        # Znajdź pierwszą klamrę otwierającą
        start_index = text.find('{')
        # Znajdź ostatnią klamrę zamykającą
        end_index = text.rfind('}')
        
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_string = text[start_index : end_index + 1]
            
            # Spróbuj sparsować wyciągnięty podciąg
            return json.loads(json_string)
        else:
            # Jeśli nie znaleziono { lub }, to nie jest poprawny JSON
            raise json.JSONDecodeError(
                "Could not find a valid JSON object block (missing '{' or '}') in the response.", 
                text, 
                0
            )
            
    except json.JSONDecodeError as e:
        print(f"--- ERROR during JSON extraction ---", file=sys.stderr)
        print(f"Failed to parse text: {text}", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        print(f"-------------------------------------", file=sys.stderr)
        # Rzuć wyjątek ponownie, aby został przechwycony przez główną pętlę try/except
        raise e


def save_progress(progress_file, data):
    with open(progress_file, 'w') as f:
        json.dump(data, f)

def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return None

def evaluate_vulnerability_fix_dataset():
    """
    Uruchamia ewaluację na zbiorze danych w trybie Multi AGENT.
    """
    print("--- Initializing agents ---")
    print("--- Running in MULTI AGENT mode ---")

    agents = CodeAgents()

    csv_file = "datasets/vulnerability_fix_dataset.csv"
    progress_file = "multi_agent_progress.json"

    # Load progress if available
    progress = load_progress(progress_file)
    if progress:
        print("Resuming from saved progress...")
        tp = progress.get("tp", 0)
        fp = progress.get("fp", 0)
        tn = progress.get("tn", 0)
        fn = progress.get("fn", 0)
        start_index = progress.get("current_index", 0)
    else:
        print("Starting fresh...")
        tp = fp = tn = fn = 0
        start_index = 0

    try:
        df = pd.read_csv(csv_file)

        if 'vulnerable_code' not in df.columns or 'fixed_code' not in df.columns:
            print(f"Error: CSV must contain 'vulnerable_code' and 'fixed_code' columns.", file=sys.stderr)
            return

        total_samples = len(df)
        print(f"--- Successfully loaded {total_samples} samples from {csv_file} ---")

        for index, row in df.iloc[start_index:].iterrows():
            print(f"\n--- Processing Sample {index + 1} of {total_samples} ---")

            # --- 1. Test kodu PODATNEGO (Test pozytywny) ---
            vuln_code = row['vulnerable_code']
            expected_vulnerability = row.get('vulnerability_type', 'N/A') # Użyj .get() dla bezpieczeństwa
            
            if not isinstance(vuln_code, str) or not vuln_code.strip():
                print(f"Skipping row {index + 1} (no vulnerable code found)")
                continue

            print(f"Analyzing vulnerable code for: {expected_vulnerability}")
            try:
                analyzed_code = agents.analyze_code(vuln_code)
                vulnerabilities = agents.find_vulnerabilities(analyzed_code)
                verdict = agents.verify_risk_and_fp(vuln_list=vulnerabilities, context_summary=analyzed_code)

                print("\n--- Code Context Summary ---")
                print(analyzed_code)
                print("\n--- Potential Vulnerabilities ---")
                print(vulnerabilities)
                print("\n--- Final Risk & FP Assessment ---")
                print(verdict)

                if verdict.get("has_vulnerability") is True:
                    if expected_vulnerability:  # Assuming `expected_vulnerability` indicates a real vulnerability
                        tp += 1
                        print("Result: CORRECTLY detected vulnerability (TP)")
                    else:
                        fp += 1
                        print("Result: FALSELY detected vulnerability (FP)")
                elif verdict.get("has_vulnerability") is False:
                    if expected_vulnerability:
                        fn += 1
                        print("Result: FAILED to detect vulnerability (FN)")
                    else:
                        tn += 1
                        print("Result: CORRECTLY identified no vulnerability (TN)")
                else:
                    print("Result: Inconclusive or invalid response.")

            except Exception as e:
                print(f"Error processing sample {index + 1}: {e}", file=sys.stderr)

            # Save progress periodically
            if (index + 1) % 10 == 0:
                save_progress(progress_file, {
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "current_index": index + 1
                })

    except FileNotFoundError:
        print(f"Error: The file {csv_file} was not found.", file=sys.stderr)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {csv_file} is empty.", file=sys.stderr)
    except KeyboardInterrupt:
        print("Execution interrupted. Saving progress...")
        save_progress(progress_file, {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "current_index": index + 1
        })
        print("Progress saved. You can resume later.")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)