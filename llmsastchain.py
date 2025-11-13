# agents.py
import os
import pandas as pd
import sys
import json
import re
import requests
import lmstudio as lms
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

script_path = os.path.realpath(__file__)
BASE_DIR = os.path.dirname(script_path)
PROMPTS_DIR = os.path.join(BASE_DIR, "agent_prompts")

def load_prompt_from_file(filename: str) -> str:
    """Wczytuje zawartość pliku z folderu PROMPTS_DIR."""
    file_path = os.path.join(PROMPTS_DIR, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku promptu. Upewnij się, że plik istnieje.", file=sys.stderr)
        print(f"Oczekiwana ścieżka: {file_path}", file=sys.stderr)
    except Exception as e:
        print(f"Błąd podczas wczytywania promptu {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

SYSTEM_PROMPT_ONLY_ONE_AGENT = load_prompt_from_file("single_agent.txt")
SYSTEM_PROMPT_CONTEXT = load_prompt_from_file("context_agent.txt")
SYSTEM_PROMPT_VULN = load_prompt_from_file("vuln_hunter_agent.txt")
SYSTEM_PROMPT_RISK_FP = load_prompt_from_file("fp_remover_agent.txt")


@dataclass(frozen=True)
class AgentConfig:
    base_url: str = "http://10.147.18.100:1234/v1"  # Base URL for LM Studio
    model: str = "ibm/granite-4-h-tiny"  # Default model
    api_key: str = ""  # API key if required
    temperature: float = 0.2  # Default temperature



class CodeAgents:

    def __init__(self, config: AgentConfig | None = None) -> None:
        cfg = config or AgentConfig()

        self._config = AgentConfig(
            base_url=cfg.base_url,
            model=cfg.model,
            api_key=cfg.api_key,
            temperature=cfg.temperature,
        )

        # Zbuduj łańcuchy
        self._context_chain = self._make_chain(
            system_prompt=SYSTEM_PROMPT_CONTEXT,
            human_template="Here is the source code to analyze:\n\n```{input_code}```",
        )

        self._vuln_chain = self._make_chain(
            system_prompt=SYSTEM_PROMPT_VULN,
            human_template=(
                "Based on the following inputs, identify potential vulnerabilities:\n\n"
                "Code Context Summary:\n{context_summary}\n\n"
            ),
        )

        self._risk_fp_chain = self._make_chain(
            system_prompt=SYSTEM_PROMPT_RISK_FP,
            human_template=(
                "Critically evaluate the following potential vulnerabilities:\n\n"
                "Vulnerabilities List:\n{vuln_list}\n\n"
                "Code Context Summary:\n{context_summary}\n\n"
            ),
        )

    def _call_local_model(self, prompt: str) -> str:
        """Wywołanie lokalnego serwera LLM Studio z poprawnym endpointem i formatem zapytania."""
        try:
            url = "http://10.147.18.100:1234/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self._config.model,
                "messages": [{"role": "user", "content": prompt}],
            }
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Local model request failed: {e}")

    def _make_chain(self, *, system_prompt: str, human_template: str):
        # Zawsze używaj lokalnego serwera
        class LocalChain:
            def __init__(self, parent, system_prompt, human_template):
                self.parent = parent
                self.system_prompt = system_prompt
                self.human_template = human_template

            def invoke(self, variables: dict):
                try:
                    human = self.human_template.format(**variables) if variables else self.human_template
                except Exception:
                    human = self.human_template
                full_prompt = f"SYSTEM:\n{self.system_prompt}\n\nUSER:\n{human}"
                return self.parent._call_local_model(full_prompt)

        return LocalChain(self, system_prompt, human_template)

    # ============ Publiczny interfejs ============
    def analyze_code(self, input_code: str) -> str:
        """Zwraca 2–3 akapity podsumowania kontekstu kodu."""
        return self._context_chain.invoke({"input_code": input_code})
    
    def find_vulnerabilities(
        self, context_summary: str) -> str:
        """Zwraca listę potencjalnych luk bezpieczeństwa na podstawie podsumowań i diffu."""
        return self._vuln_chain.invoke(
            {
                "context_summary": context_summary,
            }
        )

    def verify_risk_and_fp(self, *, vuln_list: str, context_summary: str) -> str:
        """Uruchamia Weryfikatora Ryzyka i Fałszywych Pozytywów (Agent 3).
        Przyjmuje wynik Agenta 2 plus kontekst i zwraca przefiltrowaną ocenę."""
        return self._risk_fp_chain.invoke({
        "vuln_list": vuln_list,
        "context_summary": context_summary,        }
    )


def run_short_code_dataset():
    print("--- Initializing agents ---")
    agents = CodeAgents()  
    
    csv_file = "datasets/vulnerability_fix_dataset.csv"

    counter = 0
    
    try:
        df = pd.read_csv(csv_file)
        
        if 'vulnerable_code' not in df.columns:
            print(f"Error: Column 'vulnerable_code' not found in {csv_file}", file=sys.stderr)
        else:
            print(f"--- Successfully loaded {len(df)} samples from {csv_file} ---")
            
            for index, row in df.iterrows():
                sample_code = row['vulnerable_code']
                
                # Skip rows where the code is missing
                if not isinstance(sample_code, str) or not sample_code.strip():
                    print(f"\n--- Skipping row {index + 1} (no code found) ---")
                    continue
                
                print(f"\n--- Processing Sample {index + 1} of {len(df)} ---")
                
                print("--- Running analysis ---")
                analyzed_code = agents.analyze_code(sample_code)
                vulnerabilities = agents.find_vulnerabilities(
                    analyzed_code,
                )
                verdict = agents.verify_risk_and_fp(
                    vuln_list=vulnerabilities,
                    context_summary=analyzed_code,
                )   

                # --- Print the results for this specific sample ---
                print("\n--- Code Context Summary ---")
                print(analyzed_code)
                print("\n--- Potential Vulnerabilities ---")
                print(vulnerabilities)
                print("\n--- Final Risk & FP Assessment ---")
                print(verdict)
                print(f"--- Finished Processing Sample {index + 1} ---")

                counter += 1
                if counter >= 1:
                    print("\n--- Reached processing limit of 3 samples. Exiting. ---")
                    break

    except FileNotFoundError:
        print(f"Error: The file {csv_file} was not found.", file=sys.stderr)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {csv_file} is empty.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)




def extract_json_from_string(text):
    """
    Extracts the first JSON object from a string that might be
    wrapped in Markdown (e.g., ```json ... ```) or other text.
    """
    if not isinstance(text, str):
        raise json.JSONDecodeError("Input was not a string", str(text), 0)
        
    try:
        # Find the first opening brace
        start_index = text.find('{')
        # Find the last closing brace
        end_index = text.rfind('}')
        
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_string = text[start_index : end_index + 1]
            
            # Now, attempt to parse this extracted substring
            return json.loads(json_string)
        else:
            # If we couldn't find a { or }, it's not valid JSON
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
        # Re-raise the exception to be caught by the main loop's try/except
        raise e


def evaluate_vulnerability_fix_dataset():
    print("--- Initializing agents ---")
    agents = CodeAgents()  
    
    csv_file = "datasets/vulnerability_fix_dataset.csv"

    # Counters for our metrics
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
            
            # --- 1. Test the VULNERABLE code (Positive Test) ---
            vuln_code = row['vulnerable_code']
            expected_vulnerability = row.get('vulnerability_type', 'N/A') # Use .get() for safety
            
            if not isinstance(vuln_code, str) or not vuln_code.strip():
                print(f"Skipping row {index + 1} (no vulnerable code found)")
                continue

            print(f"Analyzing vulnerable code for: {expected_vulnerability}")
            verdict_string = "" # Initialize for error reporting
            try:
                # Run agents on vulnerable code
                verdict_string = agents.verify_risk_and_fp(
                    # Pass a modified string to help the placeholder distinguish
                    vuln_list=agents.find_vulnerabilities(agents.analyze_code(f"vulnerable_code: {vuln_code}")),
                    context_summary=None, 
                )
                
                # 2. PARSE THE JSON STRING (*** THIS IS THE FIX ***)
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
                fn += 1 # Count as a failure to detect
            except KeyError:
                print(f"Error: 'has_vulnerability' key missing from agent response: {verdict_string}", file=sys.stderr)
                fn += 1 # Count as a failure to detect
            except Exception as e:
                print(f"An unknown error occurred during vuln analysis: {e}", file=sys.stderr)
                fn += 1 # Count as a failure to detect

            # --- 2. Test the FIXED code (Negative Test / FP Test) ---
            fixed_code = row['fixed_code']
            
            if not isinstance(fixed_code, str) or not fixed_code.strip():
                print(f"Skipping FP test for row {index + 1} (no fixed code found)")
                continue

            print("Analyzing fixed code (expecting no vulnerability)...")
            verdict_string_fixed = "" # Initialize for error reporting
            try:
                # Run agents on fixed code
                verdict_string_fixed = agents.verify_risk_and_fp(
                    # Pass a modified string to help the placeholder distinguish
                    vuln_list=agents.find_vulnerabilities(agents.analyze_code(f"fixed_code: {fixed_code}")),
                    context_summary=None,
                )
                
                # 2. PARSE THE JSON STRING (*** THIS IS THE FIX ***)
                verdict_fixed = extract_json_from_string(verdict_string_fixed)
                
                # 3. Access the dictionary key
                if verdict_fixed.get('has_vulnerability'):
                    fp += 1
                    print("Result: INCORRECTLY detected vulnerability (FP)")
                else:
                    tn += 1
                    print("Result: CORRECTLY ignored safe code (TN)")
                    
            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON response from agent: {verdict_string_fixed}", file=sys.stderr)
                fp += 1 # Count a malformed response on a 'safe' file as a False Positive
            except KeyError:
                print(f"Error: 'has_vulnerability' key missing from agent response: {verdict_string_fixed}", file=sys.stderr)
                fp += 1 # Also count as FP
            except Exception as e:
                print(f"An unknown error occurred during fixed analysis: {e}", file=sys.stderr)
                fp += 1 # Also count as FP

        # --- 3. Calculate and Print Final Metrics ---
        print("\n--- Evaluation Complete ---")
        
        # Check if any tests were actually run
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

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Accuracy for the "vulnerable" task
        positive_accuracy = tp / total_positive_cases if total_positive_cases > 0 else 0.0
        # Accuracy for the "fixed" (FP) task
        negative_accuracy = tn / total_negative_cases if total_negative_cases > 0 else 0.0
        # Overall accuracy
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

if __name__ == "__main__":
    evaluate_vulnerability_fix_dataset()