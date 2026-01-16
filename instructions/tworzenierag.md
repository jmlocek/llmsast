# Tworzenie bazy RAG z DiverseVul (ostatnie 20 000 rekordów)

## Cel
Celem programu jest **zbudowanie bazy wiedzy RAG** w Qdrant na podstawie fragmentu zbioru **DiverseVul**.

- **Zbiór do bazy RAG (train/RAG KB):** ostatnie **20 000** rekordów z pliku `datasets/diversevul_20230702.json`.
- **Zbiór testowy:** wszystkie pozostałe rekordy (program może wygenerować artefakty ułatwiające powtarzalny podział, ale **nie uruchamia ewaluacji**).

Program ma **tylko**:
1) wczytać dane,
2) wybrać końcowe 20k rekordów,
3) policzyć embeddingi,
4) wgrać punkty do Qdrant,
5) zapisać metadane o podziale.

## Kontekst repo
W repo istnieje wstępny skrypt: `rag/create_rag_db.py` (obecnie korzysta z embeddingów przez LM Studio i zapisuje do Qdrant po HTTP).
Ten dokument opisuje wymagania dla docelowego programu (może być rozwinięciem istniejącego skryptu lub nowym plikiem).

---

## Definicje
- **DiverseVul**: zbiór rekordów (JSON Lines) zawierających m.in. pole `func` (kod funkcji), `target` (etykieta), `cwe` (lista CWE), `project`, `commit_id`, `hash`, `size`, `message`.
- **Rekord**: jedna linia JSON w pliku (JSONL).
- **Ostatnie 20 000**: deterministycznie rozumiane jako **ostatnie 20 000 linii/rekordów w kolejności pliku** (tj. „tail”).
- **Baza RAG**: kolekcja w Qdrant zawierająca wektory embeddingów oraz payload z metadanymi rekordu.

---

## Wymagania funkcjonalne

### 1) Wejście danych
- Program MUSI wczytać dane z pliku JSONL: domyślnie `datasets/diversevul_20230702.json`.
- Program MUSI wspierać parametr CLI do wskazania pliku wejściowego (np. `--input`).
- Program MUSI działać w trybie strumieniowym (iteracja po liniach) lub z użyciem narzędzia typu pandas; preferowany jest tryb strumieniowy dla stabilności pamięci.

### 2) Selekcja rekordów do RAG
- Program MUSI wybrać dokładnie ostatnie `N=20000` rekordów (parametryzowane np. `--rag-size 20000`).
- Jeśli plik ma mniej niż `N` rekordów: program powinien wczytać wszystkie i ostrzec użytkownika.
- Program MUSI zachować deterministyczność podziału (ta sama wersja pliku → ten sam zestaw rekordów w RAG).

### 3) Reprezentacja dokumentu do embeddingu
- Dla każdego rekordu program MUSI wygenerować tekst do embeddingu oparty o pole `func`.
- Program POWINIEN umożliwiać warianty tekstu (flagą), np.:
  - `code_only`: tylko `func`
  - `code_plus_meta`: `func` + lekkie metadane (`project`, `cwe`, `message`) w jednym stringu
- Program MUSI zabezpieczyć długość wejścia do modelu embeddingów (np. limit znaków / tokenów, obcięcie, normalizacja białych znaków).

### 4) Generowanie embeddingów
Program MUSI wspierać co najmniej jeden sposób generowania embeddingów (wybór przez flagę lub konfigurację):

**Opcja A (zgodna z aktualnym skryptem): LM Studio / OpenAI-compatible embeddings**
- Program łączy się z serwerem embeddingów przez OpenAI-compatible API (np. LM Studio).
- Konfiguracja (zgodna z obecnym skryptem): `LM_STUDIO_URL`, `LM_STUDIO_KEY`, `EMBEDDING_MODEL_ID`.
- Uwaga: biblioteka OpenAI sama dodaje nagłówek `Authorization: Bearer ...`, więc jeśli podasz klucz w formie `Bearer XYZ`, program powinien usunąć prefiks `Bearer `.

Wymagania dot. modelu embeddingów dla tego projektu (konkretna konfiguracja):
- Domyślny `EMBEDDING_MODEL_ID`: `text-embedding-nomic-embed-text-v1.5`
- Oczekiwany rozmiar wektora (`VECTOR_SIZE`): `768` (Nomic v1.5)
- Domyślny `LM_STUDIO_URL`: `https://10.147.18.200:1234/v1`
- Domyślny `LM_STUDIO_KEY`: `Bearer LLM-abcde` (prefiks `Bearer ` może zostać odcięty przed przekazaniem do klienta)

**Opcja B (rekomendowana jako alternatywa): lokalny model embeddingów bez serwera**
- Program liczy embeddingi lokalnie (np. `sentence-transformers` lub `fastembed`).
- Zaleta: brak zależności od zewnętrznego endpointu i TLS.

W obu opcjach:
- Program MUSI weryfikować zgodność wymiaru wektora (`vector_size`) z konfiguracją kolekcji Qdrant.
- Program POWINIEN obsłużyć retry/backoff dla błędów sieciowych (jeśli embeddingi są z API).

### 5) Inicjalizacja i zapis do Qdrant
- Program MUSI łączyć się z Qdrant po HTTP (np. `http://localhost:6333`).
- Program MUSI tworzyć kolekcję, jeśli nie istnieje.
- Program MUSI umożliwić tryb:
  - `--recreate`: usuń i utwórz kolekcję od nowa
  - bez `--recreate`: dopisuj punkty (upsert)
- Konfiguracja kolekcji:
  - `vectors.size = VECTOR_SIZE`
  - `distance = COSINE` (domyślnie)

### 6) Struktura payload
Każdy punkt w Qdrant MUSI zawierać payload co najmniej:
- `code` (z `func`)
- `is_vulnerable` (z `target` jako bool/int)
- `cwe` (lista; jeśli brak, to `[]`)

Każdy punkt POWINIEN zawierać również (jeśli dostępne):
- `project`, `commit_id`, `hash`, `size`, `message`
- `origin`/`origin_index`/`origin_line` — informacja umożliwiająca identyfikację rekordu w pliku

### 7) Identyfikatory i powtarzalność
- Program MUSI nadawać stabilny identyfikator punktu albo:
  - deterministyczny (np. hash z `project+commit_id+hash`), albo
  - losowy UUID (wtedy program MUSI zapisać mapowanie do artefaktu).

Rekomendacja: deterministyczne `point_id` (ułatwia rerun i deduplikację).

### 8) Artefakty wyjściowe (split + metryki)
Program MUSI zapisać do `rag/` (lub `results/`) co najmniej:
- plik manifestu RAG (np. `rag_diversevul_manifest.json`) zawierający:
  - ścieżkę wejścia, datę uruchomienia, liczbę rekordów, `rag_size`, nazwę kolekcji,
  - informację jak zdefiniowano „ostatnie 20k” (tail z kolejności pliku),
  - statystyki: ile rekordów wgrano, ile pominięto, ile błędów embeddingu.
- plik listy identyfikatorów rekordów w RAG (np. `rag_train_ids.jsonl` lub `.json`).

Opcjonalnie (ale bardzo przydatne):
- `test_filter.json`/`test_exclusion_ids.json` (lista rekordów, które NIE są w RAG i powinny trafić do testów).

---

## Wymagania niefunkcjonalne
- **Wydajność:** batchowanie upsertów do Qdrant (np. 50–200 punktów/batch) oraz batchowanie embeddingów, jeśli API/model pozwala.
- **Odporność:** program nie powinien przerywać całego procesu przez błąd pojedynczego rekordu (log + skip), ale powinien zwrócić niezerowy kod wyjścia, jeśli odsetek błędów przekracza próg (np. `--max-fail-rate`).
- **Logowanie:** czytelne logi postępu (tqdm lub log co X rekordów), plus podsumowanie na końcu.
- **Konfiguracja:** możliwość ustawienia parametrów przez `.env` lub zmienne środowiskowe oraz flagi CLI.
- **Powtarzalność:** ten sam input + config → ta sama kolekcja (przy deterministycznych ID) i ta sama lista rekordów RAG.

---

## Interfejs CLI (propozycja)
Przykładowe uruchomienie:

- `python -m rag.build_diversevul_qdrant --input datasets/diversevul_20230702.json --rag-size 20000 --qdrant-url http://localhost:6333 --collection diversevul_kb --recreate`

Minimalne flagi:
- `--input` (ścieżka do JSONL)
- `--rag-size` (domyślnie 20000)
- `--qdrant-url` (domyślnie `http://localhost:6333`)
- `--collection` (domyślnie np. `diversevul_kb`)
- `--recreate` (bool)

---

## Qdrant w Podman (Windows)
Założenie: Qdrant działa w kontenerze uruchomionym lokalnie i ma wystawione porty.

### 1) Start maszyny Podman (jeśli potrzeba)
- `podman machine init`
- `podman machine start`

### 2) Uruchom Qdrant
Rekomendowany run (z trwałym wolumenem):
- `podman volume create qdrant_data`
- `podman run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_data:/qdrant/storage qdrant/qdrant:latest`

### 3) Szybka weryfikacja
- `curl http://localhost:6333/collections`

Program powinien przed startem indeksowania wykonać prosty check połączenia do Qdrant i czytelnie wypisać błąd, jeśli Qdrant nie odpowiada.

---

## Walidacja poprawności
Program POWINIEN wykonać kontrolę:
- czy embedding ma oczekiwany rozmiar `VECTOR_SIZE`
- czy liczba punktów w kolekcji po indeksowaniu zwiększyła się o ~N (w zależności od błędów)
- czy payload zawiera wymagane pola

---

## Uwagi / alternatywne propozycje
- Qdrant jest dobrym wyborem (szybki, produkcyjny, filtrowanie payload). Alternatywy do prototypu:
  - FAISS (lokalnie, bez serwera) — ale brak filtrowania payload jak w Qdrant.
  - Chroma — prostszy start, ale Qdrant jest bardziej „serwerowy”.

W tym projekcie rekomendowane jest pozostanie przy Qdrant (Podman) + embeddingi lokalne lub LM Studio, zależnie od infrastruktury.
