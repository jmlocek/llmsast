import pandas as pd
import sys
import json
import os
from agents_logic import CodeAgents, AgentConfig

def extract_simple_verdict_and_report(text: str) -> dict:
    """
    Extracts the verdict and report from the simple text format.
    Format:
    has_vulnerability: [true/false]
    [Report text...]
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Response is empty or not a string")

    try:
        # Find the first line break
        newline_index = text.find('\n')
        
        if newline_index == -1:
            first_line = text.strip()
            report = "[No report found]"
        else:
            first_line = text[:newline_index].strip()
            report = text[newline_index+1:].strip()

        # Parse the first line
        key = "has_vulnerability:"
        if not first_line.lower().startswith(key):
            raise ValueError(f"Response missing '{key}' on the first line.")

        bool_str = first_line[len(key):].strip().lower()
        
        has_vuln = False
        if bool_str == 'true':
            has_vuln = True
        elif bool_str == 'false':
            has_vuln = False
        else:
            raise ValueError(f"Invalid boolean value: '{bool_str}' found on first line.")

        return {
            "has_vulnerability": has_vuln,
            "report": report
        }

    except Exception as e:
        print(f"--- ERROR during Simple Verdict extraction ---", file=sys.stderr)
        print(f"Failed to parse text: {text}", file=sys.stderr)
        raise e # Re-raise to be caught by main loop

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
    Uruchamia ewaluację na zbiorze danych w trybie SINGLE AGENT.
    """
    csv_file = "datasets/vulnerability_fix_dataset.csv"
    progress_file = "single_agent_progress_granite-tiny.json"

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
        print("--- Running in SINGLE AGENT mode ---")
        agents = CodeAgents()

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

                if not isinstance(vuln_code, str) or not vuln_code.strip():
                    print(f"Skipping row {index + 1} (no vulnerable code found)")
                    continue

                verdict_string = "" # Initialize for error reporting
                try:
                    print("Analyzing vuln code (expecting vulnerability)...")

                    # Use single agent mode
                    verdict_string = agents.analyze_code_single(vuln_code)
                    
                    verdict_data = extract_simple_verdict_and_report(verdict_string)
                    
                    if verdict_data.get('has_vulnerability') is True:
                        tp += 1
                        print("Result: CORRECTLY detected vulnerability (TP)")
                    elif verdict_data.get('has_vulnerability') is False:
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
                    # Use single agent mode for fixed code
                    verdict_string_fixed = agents.analyze_code_single(fixed_code)
                    
                    verdict_data_fixed = extract_simple_verdict_and_report(verdict_string_fixed)
                    
                    if verdict_data_fixed.get('has_vulnerability') is True:
                        fp += 1
                        print("Result: INCORRECTLY detected vulnerability (FP)")
                    elif verdict_data_fixed.get('has_vulnerability') is False:
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
