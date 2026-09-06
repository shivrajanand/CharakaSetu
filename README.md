# 🌿 CharakaSetu

### AI-powered retrieval of classical Ayurvedic knowledge from the Charaka Samhita

**CharakaSetu** is a retrieval-only prototype that connects an English patient symptom or clinical complaint with relevant passages from the **Charaka Samhita**.

The system is designed to help users **find relevant classical textual evidence**, not to diagnose conditions or recommend treatments.

> **Important:** CharakaSetu does not generate medical advice, diagnoses, or treatment recommendations. It retrieves existing passages from the source corpus.

---

## How it works

```text
Patient complaint
       ↓
JinaColBERT-v2
       ↓
Late-interaction / MaxSim retrieval
       ↓
Ranked Charaka Samhita passages
       ↓
Sanskrit + IAST + English translation
       ↓
Chapter, verse and source information
```

The application is built with **Streamlit**.

The retriever is separated from the application layer so that other retrieval approaches can be evaluated or added later without redesigning the interface.

---

## Project structure

```text
charaksetu/
├── app.py
│
├── rag/
│   ├── corpus.py
│   ├── retriever.py
│   └── pipeline.py
│
├── utils/
│   └── formatting.py
│
├── data/
│   └── rag-corpus/
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Main components

* **`app.py`** — Streamlit application and user interface
* **`rag/corpus.py`** — corpus loading and validation
* **`rag/retriever.py`** — retriever interface and JinaColBERT-v2 implementation
* **`rag/pipeline.py`** — connects query, retrieval and corpus lookup
* **`utils/formatting.py`** — result presentation
* **`data/rag-corpus/`** — Charaka Samhita retrieval corpus

---

## Corpus

CharakaSetu uses an existing `rag-corpus`; it does not create or modify the underlying knowledge source.

The corpus contains:

```text
verse_id
chapter_title
sthana
heading_path
devanagari
iast
translation
verse_numbers_list
source_url
```

The searchable representation currently combines:

```text
translation + heading_path
```

---

## Retrieval model

The current prototype uses:

**JinaColBERT-v2**

Retrieval follows a ColBERT-style late-interaction approach:

1. Encode the query into token-level embeddings.
2. Encode corpus passages into token-level embeddings.
3. Compute token-level similarities.
4. Apply MaxSim across document tokens.
5. Rank passages by the resulting score.

The current implementation uses a fixed **32-token query width**, as required by the model.

Model outputs are converted to CPU FP32 NumPy arrays before NumPy-based scoring.

---

## Current retrieval results

Pilot evaluation on the current test set:

| Metric    |      Score |
| --------- | ---------: |
| Recall@1  |     22.22% |
| Recall@3  |     37.65% |
| Recall@5  |     41.98% |
| Recall@10 | **53.70%** |
| MRR       |      0.511 |
| nDCG@10   |  **0.435** |

These are **pilot results**, not the final benchmark.

Other retrieval approaches, including BM25, dense embeddings and hybrid retrieval, are being evaluated separately as part of the research work.

---

## Running locally

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
```

### 2. Add the corpus

Place the existing corpus under:

```text
data/rag-corpus/
```

Supported formats depend on the corpus loader implementation.

### 3. Start the application

```bash
streamlit run app.py
```

The application loads the corpus, initializes JinaColBERT-v2 and prepares the document representations for retrieval.

---

## Using CharakaSetu

Enter a patient's symptoms or clinical complaint in English.

For example:

```text
The patient complains of persistent fatigue, weakness,
poor appetite, heaviness in the body and excessive sleepiness.
```

Select the desired number of results and retrieve the relevant passages.

Each result provides the available:

* Chapter
* Sthana
* Section
* Sanskrit text
* IAST transliteration
* English translation
* Verse citation
* Source

---

## Design direction

CharakaSetu is currently a **Phase-1 retrieval prototype**.

The architecture intentionally keeps the retriever independent from the application so that future versions can investigate:

```text
BM25
Dense Retrieval
JinaColBERT
BM25 + JinaColBERT
Reranking
```

Future research may extend the system beyond retrieval, but such functionality is not part of the current prototype.

---

## Scope

### Included

* English clinical query input
* Classical Ayurvedic text retrieval
* JinaColBERT-v2
* Top-K retrieval
* Sanskrit and English evidence display
* Source attribution
* Streamlit interface

### Not included

* Diagnosis
* Treatment recommendation
* LLM-generated answers
* Medical advice
* Autonomous clinical decision making
* Knowledge graph
* Fine-tuning
* Agentic workflows
* Chat memory

---

## Attribution

**Made by [Sanganaka, IIT Kharagpur](https://sanganaka-iitkgp.github.io/).**

---

### CharakaSetu

*Bridging modern clinical queries with classical Ayurvedic knowledge.*
