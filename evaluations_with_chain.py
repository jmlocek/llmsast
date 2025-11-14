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
        print("--- Initializing agents ---")
        agents = CodeAgents()

        csv_file = "datasets/vulnerability_fix_dataset.csv"

        try:
            df = pd.read_csv(csv_file)

            if 'vulnerable_code' not in df.columns or 'fixed_code' not in df.columns:
                print(f"Error: CSV must contain 'vulnerable_code' and 'fixed_code' columns.", file=sys.stderr)
                return

            total_samples = len(df)
            print(f"--- Successfully loaded {total_samples} samples from {csv_file} ---")

            for index, row in df.iloc[start_index:].iterrows():
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
                    # 1. Generate context ONCE and store it
                    context_summary_vuln = agents.analyze_code(vuln_code)

                    vuln_list_string = agents.find_vulnerabilities(
                        context_summary=context_summary_vuln,
                        input_code=vuln_code  
                    )

                    verdict_string = agents.verify_risk_and_fp(
                        vuln_list=vuln_list_string,
                        context_summary=context_summary_vuln,
                        input_code=vuln_code  
                    )
                    
                    # 2. PARSE THE JSON STRING (*** THIS IS THE FIX ***)
                    verdict_vuln = extract_json_from_string(verdict_string)
                    
                    print(verdict_vuln)
                    if verdict_vuln.get('has_vulnerability') is True:
                        tp += 1
                        print("Result: CORRECTLY detected vulnerability (TP)")
                    elif verdict_vuln.get('has_vulnerability') is False:
                        fn += 1
                        print("Result: FAILED to detect vulnerability (FN)")
                    else:
                        print("Parsed JSON did not contain a valid 'has_vulnerability' boolean value.")

                except json.JSONDecodeError:
                    print(f"Error: Could not parse JSON response from agent: {verdict_string}", file=sys.stderr)
                except KeyError:
                    print(f"Error: 'has_vulnerability' key missing from agent response: {verdict_string}", file=sys.stderr)
                except Exception as e:
                    print(f"An unknown error occurred during vuln analysis: {e}", file=sys.stderr)

                # --- 2. Test the FIXED code (Negative Test / FP Test) ---
                fixed_code = row['fixed_code']
                
                if not isinstance(fixed_code, str) or not fixed_code.strip():
                    print(f"Skipping FP test for row {index + 1} (no fixed code found)")
                    continue

                print("Analyzing fixed code (expecting no vulnerability)...")
                verdict_string_fixed = "" # Initialize for error reporting
                try:
                    # Run agents on fixed code
                    # 1. Generate context ONCE and store it
                    context_summary_vuln = agents.analyze_code(vuln_code)

                    vuln_list_string = agents.find_vulnerabilities(
                        context_summary=context_summary_vuln,
                        input_code=vuln_code  # <-- ADD THIS
                    )

                    verdict_string = agents.verify_risk_and_fp(
                        vuln_list=vuln_list_string,
                        context_summary=context_summary_vuln,
                        input_code=vuln_code  # <-- ADD THIS
                    )
                    
                    verdict_fixed = extract_json_from_string(verdict_string_fixed)
                    
                    # 3. Access the dictionary key
                    if verdict_fixed.get('has_vulnerability') is True:
                        fp += 1
                        print("Result: INCORRECTLY detected vulnerability (FP)")
                    elif verdict_fixed.get('has_vulnerability') is False:
                        tn += 1
                        print("Result: CORRECTLY ignored safe code (TN)")
                    else:
                        print("Parsed JSON did not contain a valid 'has_vulnerability' boolean value.")    

                except json.JSONDecodeError:
                    print(f"Error: Could not parse JSON response from agent: {verdict_string_fixed}", file=sys.stderr)
                except KeyError:
                    print(f"Error: 'has_vulnerability' key missing from agent response: {verdict_string_fixed}", file=sys.stderr)
                except Exception as e:
                    print(f"An unknown error occurred during fixed analysis: {e}", file=sys.stderr)

                # Save progress after processing each sample
                progress = {
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "current_index": index + 1
                }
                save_progress(progress_file, progress)

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
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)