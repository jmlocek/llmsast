import json
import sys
from agents_logic_old import CodeAgents

def extract_simple_verdict_and_report(text: str) -> dict:
    """
    Extracts the verdict and report from the simple text format.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Response is empty or not a string")

    try:
        newline_index = text.find('\n')
        
        if newline_index == -1:
            first_line = text.strip()
            report = "[No report found]"
        else:
            first_line = text[:newline_index].strip()
            report = text[newline_index+1:].strip()

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


def debug_evaluation():
    """
    Debug script to check what the model is actually returning
    """
    json_file = "datasets/diversevul_20230702.json"

    print("--- Initializing agents ---")
    agents = CodeAgents()

    # Load dataset
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
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
                continue
    
    print(f"Loaded {len(vulnerable_functions)} vulnerable and {len(safe_functions)} safe functions")
    
    # Test on first vulnerable sample
    print("\n" + "="*80)
    print("TESTING ON VULNERABLE CODE (should return has_vulnerability: true)")
    print("="*80)
    
    if vulnerable_functions:
        vuln_code = vulnerable_functions[0].get('func', '')
        print(f"\nCode snippet (first 500 chars):\n{vuln_code[:500]}...")
        
        # Multi-agent mode
        print("\n--- MULTI-AGENT MODE ---")
        print("\nStep 1: Context Analysis...")
        context_summary = agents.analyze_code(vuln_code)
        print(f"Context Summary:\n{context_summary}\n")
        
        print("\nStep 2: Vulnerability Detection...")
        vuln_list_string = agents.find_vulnerabilities(
            context_summary=context_summary,
            input_code=vuln_code
        )
        print(f"Vulnerabilities Found:\n{vuln_list_string}\n")
        
        print("\nStep 3: Risk Verification...")
        verdict_string = agents.verify_risk_and_fp(
            vuln_list=vuln_list_string,
            context_summary=context_summary,
            input_code=vuln_code
        )
        print(f"Final Verdict (RAW):\n{verdict_string}\n")
        
        try:
            verdict_data = extract_simple_verdict_and_report(verdict_string)
            print(f"Parsed Result: {verdict_data['has_vulnerability']}")
            print(f"Report: {verdict_data['report']}")
        except Exception as e:
            print(f"ERROR parsing verdict: {e}")
        
        # Single-agent mode
        print("\n--- SINGLE-AGENT MODE ---")
        single_verdict = agents.analyze_code_single(vuln_code)
        print(f"Single Agent Verdict (RAW):\n{single_verdict}\n")
        
        try:
            single_data = extract_simple_verdict_and_report(single_verdict)
            print(f"Parsed Result: {single_data['has_vulnerability']}")
            print(f"Report: {single_data['report']}")
        except Exception as e:
            print(f"ERROR parsing verdict: {e}")
    
    # Test on first safe sample
    print("\n" + "="*80)
    print("TESTING ON SAFE CODE (should return has_vulnerability: false)")
    print("="*80)
    
    if safe_functions:
        safe_code = safe_functions[0].get('func', '')
        print(f"\nCode snippet (first 500 chars):\n{safe_code[:500]}...")
        
        # Single-agent mode (faster for testing)
        print("\n--- SINGLE-AGENT MODE ---")
        safe_verdict = agents.analyze_code_single(safe_code)
        print(f"Single Agent Verdict (RAW):\n{safe_verdict}\n")
        
        try:
            safe_data = extract_simple_verdict_and_report(safe_verdict)
            print(f"Parsed Result: {safe_data['has_vulnerability']}")
            print(f"Report: {safe_data['report']}")
        except Exception as e:
            print(f"ERROR parsing verdict: {e}")


if __name__ == "__main__":
    debug_evaluation()
