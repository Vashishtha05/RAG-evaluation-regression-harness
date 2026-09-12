#  RAG Evaluation & Regression Harness

<p align="center">
  <img src="https://img.shields.io/badge/RAG-Evaluation-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Testing-Regression-black?style=for-the-badge">
  <img src="https://img.shields.io/badge/LLM-Judges-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-Quality%20Engineering-yellow?style=for-the-badge">
</p>

<p align="center">
  A testable evaluation and regression framework for measuring Retrieval-Augmented Generation (RAG) quality across retrieval, generation, and abstention.
</p>

---

## 📌 Overview

RAG Evaluation & Regression Harness treats **RAG quality as a testable software contract**.

The system keeps **retrieval quality** separate from **generation quality**, making it possible to diagnose whether a failing answer is caused by poor document retrieval or poor answer generation.

The project provides:

* A verified golden evaluation dataset
* Swappable RAG pipeline components
* Deterministic retrieval metrics
* LLM-based generation judges
* Judge calibration against human labels
* SQLite-based evaluation storage
* Baselines and regression detection
* Pull-request and nightly evaluation workflows
* Streamlit dashboard support

The initial evaluation corpus is the **public-domain Constitution of the United States**.

---

## ✨ Features

* 🎯 Golden dataset with verified evaluation cases
* 🔍 Precision@k, Recall@k, and MRR retrieval metrics
* 🧠 Faithfulness, answer relevance, and correctness evaluation
* 🚫 Dedicated abstention evaluation for unanswerable questions
* ⚖️ Human-vs-LLM judge calibration
* 📊 Variance-based regression thresholds
* 🗄️ SQLite evaluation results store
* 🔄 Swappable retrieval and generation components
* 🚦 Automated quality gates
* 🔀 Pull-request evaluation
* 🌙 Scheduled full evaluation
* 📈 Streamlit evaluation dashboard

---

## 📊 Evaluation Dataset

The golden dataset contains **50 evaluation cases**, including **48 verified cases**.

The cases cover:

* Single-chunk factual questions
* Multi-chunk questions
* Deliberately unanswerable questions

The `verified` field identifies the human-reviewed evaluation core.

Future generated candidates must not be promoted into the verified subset without human review.

---

## 🧮 Metrics

The evaluation system separates metrics based on the cases where they are meaningful.

| Metric Family | Metrics                                     | Evaluation Cases   |
| ------------- | ------------------------------------------- | ------------------ |
| Retrieval     | Precision@k, Recall@k, MRR                  | Answerable cases   |
| Abstention    | Abstention score                            | Unanswerable cases |
| Generation    | Faithfulness, Answer Relevance, Correctness | All cases          |

### 🔍 Retrieval

Retrieval metrics are evaluated only on **answerable cases**.

Unanswerable questions do not contain a relevant chunk to retrieve, so including them would produce misleading retrieval scores.

### 🚫 Abstention

Abstention is evaluated only on **unanswerable cases**.

* `1.0` → Pipeline correctly declines to answer from the corpus
* `0.0` → Pipeline incorrectly answers

The default offline answerer never abstains, producing a baseline abstention score of `0.0`.

### 🧠 Generation

Generation metrics are evaluated across the complete golden set:

* Faithfulness
* Answer relevance
* Correctness

---

## ⚙️ Tech Stack

| Technology     | Usage                       |
| -------------- | --------------------------- |
| Python         | Core development            |
| RAG Pipeline   | Retrieval and generation    |
| Embeddings     | Document representation     |
| Vector Store   | Similarity search           |
| FAISS          | Optional vector backend     |
| OpenAI API     | LLM and embedding models    |
| SQLite         | Evaluation results storage  |
| Pytest         | Automated testing           |
| Streamlit      | Evaluation dashboard        |
| GitHub Actions | CI and scheduled evaluation |

---

## 🏗️ Build Order

The project is developed incrementally as a software-quality system:

1. 🗂️ Golden dataset
2. 🔌 Swappable RAG pipeline
3. 📐 Deterministic retrieval metrics
4. 🧠 LLM judges
5. ⚖️ Judge calibration
6. 🗄️ SQLite results store
7. 📊 Baselines and regression gates
8. 🚦 Pull-request CI and nightly evaluation
9. 📈 Streamlit dashboard

---

## 🚀 Getting Started

### 1️⃣ Create Virtual Environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2️⃣ Install Dependencies

```powershell
pip install -e ".[runtime,dev]"
```

### 3️⃣ Run Tests

```powershell
pytest
```

### 4️⃣ Validate Dataset

```powershell
python -m rag_eval.cli validate-dataset
```

---

## ▶️ Run Evaluation

To run the quality evaluation locally, persist the candidate run, and generate a Markdown report:

```powershell
python -m rag_eval.cli evaluate `
  --store results.sqlite3 `
  --run-id local `
  --baseline-file data/baseline.json `
  --threshold 0.05 `
  --report-file rag-quality-report.md
```

The quality gate:

* Compares the candidate run against a baseline
* Detects metric degradation
* Applies the configured threshold
* Exits with a non-zero status when a regression exceeds the allowed tolerance

The quality gate operates independently of persistence.

`--store` only controls whether the candidate run is written to SQLite.

---

## 🔄 Vector Store Options

The evaluation pipeline supports multiple retrieval backends.

### Deterministic Vector Backend

```powershell
python -m rag_eval.cli evaluate --vector-store vector
```

### FAISS

Install the vector extra:

```powershell
pip install -e ".[vector]"
```

Then run:

```powershell
python -m rag_eval.cli evaluate --vector-store faiss
```

### OpenAI LLM + Embeddings

```powershell
python -m rag_eval.cli evaluate `
  --llm openai:gpt-4o-mini `
  --embedding-model openai:text-embedding-3-small
```

An OpenAI API key is required for this configuration.

Prompt templates can be selected using:

```text
--prompt-template
```

---

## ⚖️ Judge Calibration

LLM judges must be calibrated against human-reviewed labels before their scores are trusted.

Calibration data is stored in:

```text
data/calibration.json
```

Run the deterministic heuristic judge:

```powershell
python -m rag_eval.cli calibrate --judge heuristic
```

Run the OpenAI judge:

```powershell
python -m rag_eval.cli calibrate --judge openai:gpt-4o-mini
```

The heuristic judge is deterministic and does not require an API key.

The OpenAI judge requires the appropriate LLM extra and API key.

Human agreement determines whether the judge is measuring meaningful quality or introducing excessive evaluation noise.

---

## 📈 Regression Thresholds

A single flat threshold assumes every metric has the same stability.

However:

* Deterministic retrieval metrics are generally stable
* LLM-judged metrics can vary between repeated runs

The framework can derive **per-metric tolerances from repeated same-configuration runs**.

Generate thresholds:

```powershell
python -m rag_eval.cli suggest-thresholds `
  --store results.sqlite3 `
  --runs run-a,run-b,run-c `
  --z 2 `
  --out data/thresholds.json
```

Use the generated thresholds:

```powershell
python -m rag_eval.cli evaluate `
  --baseline-file data/baseline.json `
  --threshold-file data/thresholds.json
```

`--threshold-file` overrides the global `--threshold` on a per-metric basis.

This allows noisy metrics to have wider tolerances while stable metrics maintain tighter regression limits.

---

## 🚦 Quality Gates

The quality gate can operate using either:

### Standalone Baseline

```text
--baseline-file data/baseline.json
```

### Stored Baseline Run

```text
--baseline <run-id>
--store results.sqlite3
```

The evaluation exits with a **non-zero status** whenever a metric drops beyond its configured tolerance.

Pull requests evaluate:

```text
10 verified cases
```

The scheduled workflow evaluates:

```text
Complete golden set
```

---

## 📂 Project Structure

```text
├── data/
│   ├── corpus/
│   │   ├── source documents
│   │   └── chunk manifests
│   ├── golden.json
│   ├── calibration.json
│   ├── baseline.json
│   └── baseline-pr.json
│
├── src/
│   └── rag_eval/
│       └── package components
│
├── tests/
│   └── focused component tests
│
├── .github/
│   └── workflows/
│       ├── pull-request evaluation
│       └── nightly evaluation
│
├── results.sqlite3
├── rag-quality-report.md
└── LICENSE
```

---

## 🔄 Evaluation Flow

```text
┌──────────────────────────┐
│     Golden Dataset       │
│  Questions + Labels      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│      RAG Pipeline        │
│                          │
│ Chunk → Embed → Retrieve │
│          → Generate      │
└────────────┬─────────────┘
             │
             ├──────────────────┐
             ▼                  ▼
┌─────────────────────┐  ┌─────────────────────┐
│ Retrieval Metrics   │  │ Generation Judges   │
│                     │  │                     │
│ Precision@k         │  │ Faithfulness        │
│ Recall@k            │  │ Relevance           │
│ MRR                 │  │ Correctness         │
└──────────┬──────────┘  └──────────┬──────────┘
           │                        │
           └───────────┬────────────┘
                       ▼
             ┌───────────────────┐
             │ Regression Gate   │
             │                   │
             │ Baseline +        │
             │ Thresholds        │
             └─────────┬─────────┘
                       ▼
             ┌───────────────────┐
             │ CI / Nightly Run  │
             │ + SQLite + Report │
             └───────────────────┘
```

---

## 🧪 Testing

Run the complete test suite:

```powershell
pytest
```

The test suite contains focused checks for individual RAG evaluation components.

Dataset validation can be performed independently:

```powershell
python -m rag_eval.cli validate-dataset
```

---

## 📊 Results & Reporting

Evaluation results can be persisted in:

```text
results.sqlite3
```

Markdown quality reports can be generated using:

```text
--report-file
```

Example:

```text
rag-quality-report.md
```

The stored results support baseline comparisons, regression analysis, and variance-based threshold generation.

---

## 🔀 CI & Scheduled Evaluation

The project includes GitHub Actions workflows for automated evaluation.

### Pull Requests

Pull requests run a faster evaluation against the first **10 verified cases**.

### Nightly Evaluation

The scheduled workflow evaluates the **complete golden dataset**.

This provides a fast development feedback loop while retaining a comprehensive scheduled quality check.

---

## 🔮 Future Extensions

* 📈 Expand the Streamlit evaluation dashboard
* 🧑‍⚖️ Increase human-labeled calibration data
* 🔍 Add additional retrieval metrics
* 🧠 Add additional generation judges
* 📊 Improve regression visualization
* 🔄 Add more RAG backends
* 🧪 Expand the golden evaluation dataset

---

## 📄 License

Released under the **MIT License**.

