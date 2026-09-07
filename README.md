# CharakaSetu

### Semantic retrieval of classical Ayurvedic knowledge from the Charaka Samhita

CharakaSetu is a retrieval-focused research prototype that connects an English patient symptom or clinical complaint with relevant passages from the **Charaka Samhita**.

The system uses **JinaColBERT-v2** with ColBERT-style late interaction to retrieve relevant verses from the corpus. It is designed to surface **source-grounded classical textual evidence**, rather than generate medical advice.

> **Important:** CharakaSetu does not diagnose conditions, recommend treatments, or provide medical advice. It retrieves existing passages from the source corpus for research and information retrieval purposes.

---

## Retrieval Pipeline

```text
Patient Query
     ↓
JinaColBERT-v2
     ↓
Late Interaction / MaxSim
     ↓
Ranked Verse IDs
     ↓
Corpus Lookup
     ↓
Relevant Charaka Samhita Passages
```

### Pipeline stages

**1. Patient Query**

The user provides an English description of symptoms, a clinical complaint, or relevant clinical notes.

**2. JinaColBERT-v2**

The query is encoded using JinaColBERT-v2 for semantic retrieval.

**3. MaxSim**

ColBERT-style late interaction compares query and document token embeddings. MaxSim produces a relevance score for each candidate passage.

**4. Ranked Verse IDs**

Candidate passages are ranked according to their retrieval scores and the top-K verse IDs are selected.

**5. Corpus Lookup**

The selected verse IDs are used to retrieve the corresponding records from the Charaka Samhita corpus.

The final result contains the available Sanskrit text, IAST transliteration, English translation, and source metadata.

---

## Offline Document Embedding

Document embeddings are **precomputed before the application is used**.

The Charaka Samhita corpus is encoded once using JinaColBERT-v2 and the resulting document representations are stored in a local cache.

```text
Charaka Samhita Corpus
          ↓
    JinaColBERT-v2
          ↓
Document Embeddings
          ↓
.embedding_cache/
jina-colbert-v2.pkl
```

This prevents the Streamlit application from having to encode the entire corpus every time it starts.

The embeddings can be generated using:

```bash
python scripts/precompute_embeddings.py
```

For example:

```bash
python scripts/precompute_embeddings.py \
    --device cuda \
    --batch-size 32
```

The cache is written under:

```text
data/rag-corpus/.embedding_cache/jina-colbert-v2.pkl
```

The application automatically uses the cache when it matches the current corpus verse IDs.

This makes the expensive document-encoding step an **offline, one-time operation**, while runtime retrieval only needs to encode the incoming query and perform MaxSim scoring against the cached document representations.

---

## Retrieval Model

The current system uses:

**JinaColBERT-v2**

JinaColBERT-v2 follows the ColBERT paradigm of **late interaction** rather than representing an entire document with a single embedding.

At retrieval time:

1. The patient query is encoded into token-level embeddings.
2. Document embeddings are loaded from the precomputed cache.
3. Query and document token representations are compared.
4. MaxSim computes the passage-level relevance score.
5. Passages are ranked by their scores.
6. The corresponding verse IDs are returned.

The current implementation uses a fixed **32-token query width**, as required by the model.

Model outputs are converted to CPU FP32 NumPy arrays before NumPy-based scoring.

---

## Corpus

CharakaSetu uses an existing `rag-corpus` as its knowledge source. The retrieval system does not generate or modify the underlying textual knowledge.

Each corpus record contains:

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

The current searchable representation combines:

```text
translation + heading_path
```

The corpus therefore allows semantic retrieval using English clinical descriptions while preserving access to the corresponding Sanskrit source material.

---

## Results

Pilot evaluation on the current test set:

| Metric    |      Score |
| --------- | ---------: |
| Recall@1  |     22.22% |
| Recall@3  |     37.65% |
| Recall@5  |     41.98% |
| Recall@10 | **53.70%** |
| MRR       |      0.511 |
| nDCG@10   |  **0.435** |

These are **pilot retrieval results**, not a final benchmark.

The project also evaluates alternative retrieval approaches, including:

* BM25
* Dense embedding retrieval
* ColBERT-based retrieval
* Hybrid retrieval

The current prototype uses **JinaColBERT-v2** as the selected retrieval approach.

---

## Application

The application is built using **Streamlit**.

The architecture separates the retrieval layer from the application layer, allowing different retrieval methods to be evaluated without redesigning the user interface.

```text
Streamlit UI
     ↓
Retrieval Pipeline
     ↓
JinaColBERT-v2 Retriever
     ↓
Precomputed Document Embeddings
     ↓
Ranked Verse IDs
     ↓
Charaka Samhita Corpus
```

---

## Project Structure

```text
charaksetu/
│
├── app.py
│
├── rag/
│   ├── corpus.py
│   ├── retriever.py
│   └── pipeline.py
│
├── scripts/
│   └── precompute_embeddings.py
│
├── utils/
│   └── formatting.py
│
├── data/
│   └── rag-corpus/
│       └── .embedding_cache/
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Main components

| Component                          | Purpose                                               |
| ---------------------------------- | ----------------------------------------------------- |
| `app.py`                           | Streamlit application and user interface              |
| `rag/corpus.py`                    | Corpus loading and validation                         |
| `rag/retriever.py`                 | Retriever interface and JinaColBERT-v2 implementation |
| `rag/pipeline.py`                  | Query, retrieval and corpus lookup pipeline           |
| `scripts/precompute_embeddings.py` | Offline document embedding and cache generation       |
| `utils/formatting.py`              | Result formatting and presentation                    |
| `data/rag-corpus/`                 | Charaka Samhita retrieval corpus                      |

---

## Running Locally

### 1. Create an environment

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the corpus

Place the retrieval corpus under:

```text
data/rag-corpus/
```

### 4. Precompute document embeddings

Run this once on the machine with the available compute:

```bash
python scripts/precompute_embeddings.py
```

For GPU-based encoding:

```bash
python scripts/precompute_embeddings.py --device cuda --batch-size 32
```

### 5. Start the application

```bash
streamlit run app.py
```

At startup, the application can load the precomputed document representations rather than re-encoding the entire corpus.

---

## Example Query

A user can enter a clinical complaint such as:

```text
The patient complains of persistent fatigue, weakness,
poor appetite, heaviness in the body and excessive sleepiness.
```

The system retrieves the most relevant passages from the Charaka Samhita.

Each retrieved result can contain:

* Sanskrit text
* IAST transliteration
* English translation
* Chapter
* Sthana
* Section / heading
* Verse citation
* Source information

---

## Research Scope

CharakaSetu is currently a **Phase-1 retrieval prototype**.

The research focuses on evaluating how effectively different information retrieval approaches can connect modern English clinical descriptions with relevant passages from classical Ayurvedic literature.

### Retrieval approaches under investigation

```text
BM25
   │
Dense Embeddings
   │
ColBERT
   │
JinaColBERT-v2
   │
Hybrid Retrieval
```

JinaColBERT-v2 is currently used in the prototype because of its late-interaction retrieval architecture and semantic matching capability.

Future work may investigate reranking and other retrieval architectures.

---

## Scope

### Included

* English clinical query input
* Charaka Samhita retrieval
* JinaColBERT-v2
* Late-interaction / MaxSim scoring
* Precomputed document embeddings
* Top-K retrieval
* Verse-level corpus lookup
* Sanskrit and English evidence display
* Source attribution
* Streamlit interface

### Not Included

* Diagnosis
* Treatment recommendation
* LLM-generated medical answers
* Medical advice
* Autonomous clinical decision making
* Knowledge graphs
* Fine-tuning
* Agentic workflows
* Chat memory

---

## Attribution

**Made by [Sanganaka, IIT Kharagpur](https://sanganaka-iitkgp.github.io/).**

---

### CharakaSetu

*Bridging modern clinical queries with classical Ayurvedic knowledge.*
