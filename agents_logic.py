import os
import sys
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
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    api_key: str | None = None  # domyślnie weź z ENV


class CodeAgents:
    def __init__(self, config: AgentConfig | None = None) -> None:
        load_dotenv()
        api_key = (config.api_key if config else None) or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Set it in your environment or .env file."
            )

        self._config = config or AgentConfig(api_key=api_key)

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

        self._single_agent_chain = self._make_chain(
            system_prompt=SYSTEM_PROMPT_ONLY_ONE_AGENT,
            human_template="Here is the source code to analyze:\n\n```{input_code}```",
        )

    def _make_chain(self, *, system_prompt: str, human_template: str):
        llm = ChatGoogleGenerativeAI(
            model=self._config.model,
            temperature=self._config.temperature,
            api_key=self._config.api_key,
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_template)]
        )
        return prompt | llm | StrOutputParser()

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

    def analyze_code_single(self, input_code: str) -> str:
        """Uruchamia pojedynczego agenta do analizy kodu."""
        return self._single_agent_chain.invoke({"input_code": input_code})