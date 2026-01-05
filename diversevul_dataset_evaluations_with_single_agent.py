import json
import sys
import os
from agents_logic_old import CodeAgents, AgentConfig

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
        raise e

def save_progress(progress_file, data):
    with open(progress_file, 'w') as f:
        json.dump(data, f)

def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return None

# Limit to first 1100 samples of each type
MAX_SAMPLES = 1100

def evaluate_diversevul_dataset():
    """
    Evaluates the DiverseVul dataset in SINGLE AGENT mode.
    Splits vulnerable and non-vulnerable functions and tests them one by one.
    """
    json_file = "datasets/diversevul_20230702.json"
    progress_file = "diversevul_single_agent_progress.json"

    # Load progress if available
    progress = load_progress(progress_file)
    if progress:
        print("Resuming from saved progress...")
        tp = progress.get("tp", 0)
        fp = progress.get("fp", 0)
        tn = progress.get("tn", 0)
        fn = progress.get("fn", 0)
        current_index = progress.get("current_index", 0)
    else:
        print("Starting fresh...")
        tp = fp = tn = fn = 0
        current_index = 0

    try:
        print("--- Initializing agents ---")
        print("--- Running in SINGLE AGENT mode ---")
        agents = CodeAgents()

        # Load and split the dataset
        print(f"Loading dataset from {json_file}...")
        vulnerable_functions = []
        safe_functions = []
        
        with open(json_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if data.get('target') == 1:
                        vulnerable_functions.append(data)
                    elif data.get('target') == 0:
                        safe_functions.append(data)
                    else:
                        print(f"Warning: Unknown target value on line {line_num}: {data.get('target')}", file=sys.stderr)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
                    continue
        
        # Limit to first MAX_SAMPLES of each type
        vulnerable_functions = vulnerable_functions[:MAX_SAMPLES]
        safe_functions = safe_functions[:MAX_SAMPLES]
        
        total_vuln = len(vulnerable_functions)
        total_safe = len(safe_functions)
        total_samples = total_vuln + total_safe
        
        print(f"--- Successfully loaded dataset ---")
        print(f"Total vulnerable functions (limited): {total_vuln}")
        print(f"Total safe functions (limited): {total_safe}")
        print(f"Total samples to evaluate: {total_samples}")
        
        # Process samples alternating between vulnerable and safe
        # current_index tracks progress through all samples (0 to total_samples-1)
        # Even indices (0, 2, 4, ...) = vulnerable, Odd indices (1, 3, 5, ...) = safe
        for idx in range(current_index, total_samples):
            # Alternate between vulnerable and safe
            if idx % 2 == 0:
                # Even index: Processing vulnerable function
                vuln_idx = idx // 2
                if vuln_idx >= total_vuln:
                    # No more vulnerable functions, skip
                    print(f"\nSkipping index {idx + 1} (no more vulnerable functions)")
                    progress = {
                        "tp": tp,
                        "fp": fp,
                        "tn": tn,
                        "fn": fn,
                        "current_index": idx + 1
                    }
                    save_progress(progress_file, progress)
                    continue
                func_data = vulnerable_functions[vuln_idx]
                is_vulnerable = True
                print(f"\n--- Processing Vulnerable Function {vuln_idx + 1}/{total_vuln} (Overall: {idx + 1}/{total_samples}) ---")
            else:
                # Odd index: Processing safe function
                safe_idx = idx // 2
                if safe_idx >= total_safe:
                    # No more safe functions, skip
                    print(f"\nSkipping index {idx + 1} (no more safe functions)")
                    progress = {
                        "tp": tp,
                        "fp": fp,
                        "tn": tn,
                        "fn": fn,
                        "current_index": idx + 1
                    }
                    save_progress(progress_file, progress)
                    continue
                func_data = safe_functions[safe_idx]
                is_vulnerable = False
                print(f"\n--- Processing Safe Function {safe_idx + 1}/{total_safe} (Overall: {idx + 1}/{total_samples}) ---")
            
            code = func_data.get('func', '')
            
            if not isinstance(code, str) or not code.strip():
                print(f"Skipping sample {idx + 1} (no code found)")
                # Still save progress to move forward
                progress = {
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "current_index": idx + 1
                }
                save_progress(progress_file, progress)
                continue
            
            try:
                # Use single agent mode
                verdict_string = agents.analyze_code_single(code)
                verdict_data = extract_simple_verdict_and_report(verdict_string)
                
                detected_vuln = verdict_data.get('has_vulnerability')
                
                if is_vulnerable:
                    # Processing vulnerable code
                    if detected_vuln is True:
                        tp += 1
                        print("Result: CORRECTLY detected vulnerability (TP)")
                    elif detected_vuln is False:
                        fn += 1
                        print("Result: FAILED to detect vulnerability (FN)")
                    else:
                        print("Parsed response did not contain a valid 'has_vulnerability' boolean value.")
                else:
                    # Processing safe code
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
            progress = {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "current_index": idx + 1
            }
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
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Accuracy for the "vulnerable" task
        positive_accuracy = tp / total_positive_cases if total_positive_cases > 0 else 0.0
        # Accuracy for the "safe" task
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
        print(f"Error: The file {json_file} was not found.", file=sys.stderr)
    except Exception as e:
        print(f"An unhandled error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    evaluate_diversevul_dataset()
