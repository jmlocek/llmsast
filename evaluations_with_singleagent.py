import pandas as pd
import sys
import json
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents_logic import CodeAgents, AgentConfig

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


def evaluate_vulnerability_fix_dataset_single():
    """
    Uruchamia ewaluację na zbiorze danych w trybie SINGLE AGENT.
    """
    print("--- Initializing agents ---")
    print("--- Running in SINGLE AGENT mode ---")
        
    agents = CodeAgents()  
    
    csv_file = "datasets/vulnerability_fix_dataset.csv"

    # Liczniki dla metryk
    tp = 0  # True Positives
    fp = 0  # False Positives
    tn = 0  # True Negatives
    fn = 0  # False Negatives

    try:
        df = pd.read_csv(csv_file)
        
        if 'vulnerable_code' not in df.columns or 'fixed_code' not in df.columns:
            print(f"Error: CSV must contain 'vulnerable_code' and 'fixed_code' columns.", file=sys.stderr)
            return
        
        total_samples = len(df)
        print(f"--- Successfully loaded {total_samples} samples from {csv_file} ---")
        
        for index, row in df.iterrows():
            print(f"\n--- Processing Sample {index + 1} of {total_samples} ---")
            
            # --- 1. Test kodu PODATNEGO (Test pozytywny) ---
            vuln_code = row['vulnerable_code']
            expected_vulnerability = row.get('vulnerability_type', 'N/A') # Użyj .get() dla bezpieczeństwa
            
            if not isinstance(vuln_code, str) or not vuln_code.strip():
                print(f"Skipping row {index + 1} (no vulnerable code found)")
                continue

            print(f"Analyzing vulnerable code for: {expected_vulnerability}")
            verdict_string = "" # Inicjalizacja do raportowania błędów
            try:
                # Uruchom tryb pojedynczego agenta
                verdict_string = agents.analyze_code_single(f"vulnerable_code: {vuln_code}")
                
                # 2. PARSOWANIE JSON
                verdict_vuln = extract_json_from_string(verdict_string)
                
                print(verdict_vuln)
                if verdict_vuln.get('has_vulnerability'): 
                    tp += 1
                    print("Result: CORRECTLY detected vulnerability (TP)")
                else:
                    fn += 1
                    print("Result: FAILED to detect vulnerability (FN)")

            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON response from agent: {verdict_string}", file=sys.stderr)
                fn += 1 # Licz jako błąd wykrycia
            except KeyError:
                print(f"Error: 'has_vulnerability' key missing from agent response: {verdict_string}", file=sys.stderr)
                fn += 1 # Licz jako błąd wykrycia
            except Exception as e:
                print(f"An unknown error occurred during vuln analysis: {e}", file=sys.stderr)
                fn += 1 # Licz jako błąd wykrycia

            # --- 2. Test kodu NAPRAWIONEGO (Test negatywny / Test FP) ---
            fixed_code = row['fixed_code']
            
            if not isinstance(fixed_code, str) or not fixed_code.strip():
                print(f"Skipping FP test for row {index + 1} (no fixed code found)")
                continue

            print("Analyzing fixed code (expecting no vulnerability)...")
            verdict_string_fixed = "" # Inicjalizacja do raportowania błędów
            try:
                # Uruchom tryb pojedynczego agenta
                verdict_string_fixed = agents.analyze_code_single(f"fixed_code: {fixed_code}")

                # 2. PARSOWANIE JSON
                verdict_fixed = extract_json_from_string(verdict_string_fixed)
                
                # 3. Dostęp do klucza słownika
                if verdict_fixed.get('has_vulnerability'):
                    fp += 1
                    print("Result: INCORRECTLY detected vulnerability (FP)")
                else:
                    tn += 1
                    print("Result: CORRECTLY ignored safe code (TN)")
                    
            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON response from agent: {verdict_string_fixed}", file=sys.stderr)
                fp += 1 # Licz błędną odpowiedź na 'bezpiecznym' pliku jako Fałszywy Pozytyw
            except KeyError:
                print(f"Error: 'has_vulnerability' key missing from agent response: {verdict_string_fixed}", file=sys.stderr)
                fp += 1 # Również licz jako FP
            except Exception as e:
                print(f"An unknown error occurred during fixed analysis: {e}", file=sys.stderr)
                fp += 1 # Również licz jako FP

        # --- 3. Oblicz i wydrukuj końcowe metryki ---
        print("\n--- Evaluation Complete ---")
        
        # Sprawdź, czy jakiekolwiek testy zostały uruchomione
        total_positive_cases = tp + fn
        total_negative_cases = tn + fp
        
        if total_positive_cases == 0 and total_negative_cases == 0:
            print("No valid samples were processed.")
            return

        print(f"Total Vulnerable Samples Tested: {total_positive_cases}")
        print(f"Total Fixed Samples Tested:     {total_negative_cases}")
        print("---")
        print(f"True Positives (TP):  {tp}")
        print(f"False Negatives (FN): {fn}")
        print(f"True Negatives (TN):  {tn}")
        print(f"False Positives (FP): {fp}")
        print("---")

        # Oblicz metryki
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Dokładność dla zadania "podatny"
        positive_accuracy = tp / total_positive_cases if total_positive_cases > 0 else 0.0
        # Dokładność dla zadania "naprawiony" (FP)
        negative_accuracy = tn / total_negative_cases if total_negative_cases > 0 else 0.0
        # Ogólna dokładność
        overall_accuracy = (tp + tn) / (total_positive_cases + total_negative_cases) if (total_positive_cases + total_negative_cases) > 0 else 0.0


        print(f"Overall Accuracy:  {overall_accuracy:.2%} (Correct classifications / Total cases)")
        print(f"Positive Accuracy: {positive_accuracy:.2%} (Recall / Sensitivity)")
        print(f"Negative Accuracy: {negative_accuracy:.2%} (Specificity)")
        print("---")
        print(f"Precision: {precision:.2%}  (How trustworthy are the 'vulnerable' alerts?)")
        print(f"Recall:    {recall:.2%}  (How many real bugs did we find?)")
        print(f"F1-Score:  {f1_score:.2%}")

    except FileNotFoundError:
        print(f"Error: The file {csv_file} was not found.", file=sys.stderr)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {csv_file} is empty.", file=sys.stderr)
    except Exception as e:
        print(f"An unhandled error occurred: {e}", file=sys.stderr)