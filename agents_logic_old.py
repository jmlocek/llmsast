import os
import sys
import requests
from dataclasses import dataclass
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
SYSTEM_PROMPT_ONLY_ONE_AGENT_BAD = load_prompt_from_file("single_agent_bad.txt")
SYSTEM_PROMPT_CONTEXT = load_prompt_from_file("context_agent.txt")
SYSTEM_PROMPT_VULN = load_prompt_from_file("vuln_hunter_agent.txt")
SYSTEM_PROMPT_RISK_FP = load_prompt_from_file("fp_remover_agent.txt")


@dataclass(frozen=True)
class AgentConfig:
    model: str = "gpt-oss-20b"
    temperature: float = 0.2
    base_url: str = "http://10.147.18.100:1234/v1" 

class CodeAgents:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self._config = config or AgentConfig()

        # Zbuduj łańcuchy
        self._context_chain = self._make_chain(
            system_prompt=SYSTEM_PROMPT_CONTEXT,
            human_template="Here is the source code to analyze:\n\n```{input_code}```",
        )

# In agents_logic.py

        self._vuln_chain = self._make_chain(
            system_prompt=SYSTEM_PROMPT_VULN,
            human_template=(
                "Based on the following inputs, identify potential vulnerabilities:\n\n"
                "Code Context Summary:\n{context_summary}\n\n"
                "Source Code:\n```{input_code}```\n\n"  # <-- ADD THIS
            ),
        )

        self._risk_fp_chain = self._make_chain(
            system_prompt=SYSTEM_PROMPT_RISK_FP,
            human_template=(
                "Critically evaluate the following potential vulnerabilities:\n\n"
                "Vulnerabilities List:\n{vuln_list}\n\n"
                "Code Context Summary:\n{context_summary}\n\n"
                "Source Code:\n```{input_code}```\n\n"  # <-- ADD THIS
            ),
        )

        self._single_agent_chain = self._make_chain(
            system_prompt=SYSTEM_PROMPT_ONLY_ONE_AGENT,
            human_template="Here is the source code to analyze:\n\n```{input_code}```",
        )

    def _make_chain(self, *, system_prompt: str, human_template: str):
        # Nested function for the LLM call
        def local_llm_call(prompt: str) -> str:
            try:
                response = requests.post(
                    url=f"{self._config.base_url}/chat/completions",
                    json={
                        "model": self._config.model,
                        "messages": [{"role": "system", "content": system_prompt}, # Note: Added system prompt to messages
                                     {"role": "user", "content": prompt}],
                        "temperature": self._config.temperature,
                    },
                    timeout=600,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                raise RuntimeError(f"Local LLM call failed: {e}")

        class LocalChain:
            def __init__(self, system_prompt, human_template):
                self.system_prompt = system_prompt
                self.human_template = human_template

            def invoke(self, variables: dict):
                prompt = self.human_template.format(**variables)
                full_prompt = f"SYSTEM:\n{self.system_prompt}\n\nUSER:\n{prompt}"
                return local_llm_call(full_prompt)

        return LocalChain(system_prompt, human_template)

    # ============ Publiczny interfejs ============
    def analyze_code(self, input_code: str) -> str:
        """Zwraca 2–3 akapity podsumowania kontekstu kodu."""
        return self._context_chain.invoke({"input_code": input_code})
    
    def find_vulnerabilities(
            self, context_summary: str, input_code: str ) -> str:
            """Zwraca listę potencjalnych luk bezpieczeństwa."""
            return self._vuln_chain.invoke(
                {
                    "context_summary": context_summary,
                    "input_code": input_code 
                }
            )

    def verify_risk_and_fp(
            self, *, vuln_list: str, context_summary: str, input_code: str) -> str:
            """Uruchamia Weryfikatora Ryzyka i Fałszywych Pozytywów (Agent 3)."""
            return self._risk_fp_chain.invoke({
                "vuln_list": vuln_list,
                "context_summary": context_summary,
                "input_code": input_code # <-- THIS IS THE FIX
            })

    def analyze_code_single(self, input_code: str) -> str:
        """Uruchamia pojedynczego agenta do analizy kodu."""
        return self._single_agent_chain.invoke({"input_code": input_code})