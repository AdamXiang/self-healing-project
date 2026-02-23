# 🩺 Self-Healing NLP Pipeline

![Python](https://img.shields.io/badge/Python-3.9--3.12-blue?logo=python&logoColor=white)
![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-3.0-017CEE?logo=apacheairflow&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-llama3.2-black?logo=ollama&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-POC-yellow)

> An Airflow 3.0 batch NLP pipeline that **automatically diagnoses and repairs dirty input data** before LLM inference, and **gracefully degrades** when the inference engine is unavailable — so the pipeline always completes, no matter what.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Design Decisions & Trade-offs](#-design-decisions--trade-offs)
- [Prerequisites](#-prerequisites)
- [Setup & Installation](#-setup--installation)
- [Usage Guide](#-usage-guide)
- [Folder Structure](#-folder-structure)
- [Known Limitations & Future Work](#-known-limitations--future-work)

---

## 🔍 Overview

Real-world review datasets are messy. `null` values, wrong data types, special characters, and oversized text fields are the norm — not the exception. Feeding this raw data directly into an LLM inference engine causes silent failures, unparseable outputs, and full pipeline crashes.

This project implements a **six-stage Airflow DAG** that solves this with three core subsystems:

1. **Self-Healing Layer** — Intercepts and repairs 5 categories of data anomalies before any inference occurs, tagging each record with full observability metadata (`was_healed`, `error_type`, `action_taken`).
2. **LLM Inference Layer** — Sends cleansed text to a locally-hosted Ollama model with a two-stage fallback parser and per-item retry logic. If the Ollama host is entirely unreachable, the batch completes in **Degraded Mode** instead of failing.
3. **Observability Layer** — Aggregates results into a structured health report with four severity tiers: `HEALTHY`, `WARNING`, `DEGRADED`, and `CRITICAL`.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Airflow DAG Execution Flow                   │
│                                                                 │
│  ┌───────────┐    ┌────────────┐    ┌─────────────────────┐     │
│  │load_model │    │load_reviews│    │diagnose_and_heal    │     │
│  │           │    │            │    │                     │     │
│  │ Validate  │    │ Stream     │    │ ┌─────────────────┐ │     │
│  │ Ollama    │    │ JSONL via  │    │ │ null → placeholder│ │   │
│  │ connection│    │ islice()   │    │ │ wrong_type → str  │ │   │
│  │ + test    │    │ (O(1) mem) │    │ │ empty → placeholder││   │
│  │ prompt    │    │            │    │ │ special_chars → ✂│ │   │
│  └─────┬─────┘    └─────┬──────┘   │ │ too_long → trunc │ │     │
│        │                │          │ └─────────────────┘ │      │
│        │                │          └──────────┬──────────┘      │
│        │                │                     │                 │
│        └────────────────┼─────────────────────┘                 │
│                         ▼                                       │
│              ┌─────────────────────┐                            │
│              │batch_analyze_       │                            │
│              │sentiment            │◄── model_info (XCom)       │
│              │                     │                            │
│              │ Per-item retry(x3)  │                            │
│              │ Two-stage parser    │                            │
│              │ Degraded fallback   │                            │
│              └──────────┬──────────┘                            │
│                         │                                       │
│              ┌──────────▼──────────┐   writes JSON to disk      │
│              │aggregate_results    │──────────────────────►     │
│              │                     │   /output/*.json           │
│              │ Sentiment dist.     │                            │
│              │ Star correlation    │   returns lightweight      │
│              │ Avg confidence      │   summary to XCom ◄──      │
│              └──────────┬──────────┘                            │
│                         │                                       │
│              ┌──────────▼──────────┐                            │
│              │generate_health_     │                            │
│              │report               │                            │
│              │                     │                            │
│              │ HEALTHY             │                            │
│              │ WARNING (>50% healed│                            │
│              │ DEGRADED (>0 degrade│                            │
│              │ CRITICAL (>10% deg.)│                            │
│              └─────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

| Stage | Input | Output | Key Mechanism |
|---|---|---|---|
| `load_model` | Ollama config | model_info dict | Fail-fast validation + test prompt |
| `load_reviews` | JSONL file | raw reviews list | `itertools.islice` for O(1) memory |
| `diagnose_and_heal_batch` | raw reviews | healed reviews | 5-strategy healing + `was_healed` flag |
| `batch_analyze_sentiment` | healed reviews | analyzed results | Per-item retry + degraded fallback |
| `aggregate_results` | results | summary (no `results` key) | Writes bulk data to disk; XCom gets only metadata |
| `generate_health_report` | summary | health report | 4-tier severity classification |

---

## 🛠 Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| **Orchestration** | Apache Airflow 3.0 (TaskFlow API) | Native `@dag` / `@task` decorators; clean dependency graph |
| **LLM Inference** | Ollama + llama3.2 | On-premise deployment — zero data egress, no per-token cost |
| **Data Format** | JSONL input / JSON output | Schema-flexible, streamable for large datasets |
| **Runtime** | Python 3.12 | Upper bound of `apache-airflow-providers-fab` compatibility window |
| **Async DB** | asyncpg | Async PostgreSQL driver for Airflow metadata store |

---

## ⚖️ Design Decisions & Trade-offs

### 1. On-Premise Ollama vs. Cloud LLM APIs

**Decision:** Run `llama3.2` locally via Ollama instead of calling OpenAI or Anthropic APIs.

**Why it matters in production:**
- **Data Privacy & Compliance:** Real customer reviews contain PII. Routing them through third-party cloud APIs creates significant GDPR/data residency risk. Local inference keeps data entirely within your network perimeter.
- **Cost Predictability:** Cloud APIs bill per token — costs scale linearly (and unpredictably) with data volume. A locally-hosted model trades upfront hardware cost for fixed, predictable inference cost regardless of batch size.

**Trade-off accepted:** Single-machine inference throughput is the bottleneck. This is a deliberate POC-phase decision.

---

### 2. XCom as Signal Channel, Not Data Store

**Anti-pattern avoided:** Passing large lists of dicts between tasks via XCom.

XCom serializes return values into the Airflow **Metadata Database** (Postgres/MySQL). At `batch_size=100`, this is manageable. At `batch_size=10,000`, it will bloat the database and degrade the Airflow scheduler.

**Pattern applied in `aggregate_results`:**

```python
# ✅ Bulk data written to disk
with open(output_file, 'w') as f:
    json.dump(summary, f)

# ✅ Only lightweight metadata returned to XCom
return {k: v for k, v in summary.items() if k != 'results'}
```

**Rule of thumb:** XCom carries *pointers*, not *payloads*. For production scale, the pointer should be an S3 URI or a file path.

---

### 3. LLM Output Parser — Defensive by Design

LLM outputs are non-deterministic. `_parse_ollama_response()` implements two-stage graceful degradation:

```
Stage 1: Strict JSON deserialization
    → Strip markdown code fences (``` json ... ```)
    → json.loads() + enum boundary validation

Stage 2: Heuristic substring matching (fallback)
    → if 'POSITIVE' in upper_text → score: 0.75
    → if 'NEGATIVE' in upper_text → score: 0.75

Final fallback: NEUTRAL with score: 0.5
```

**Lesson:** Treat LLM outputs with the same skepticism you'd apply to user input — validate and sanitize everything.

---

### 4. Degraded Mode — Fail Gracefully, Not Hard

If the Ollama host is unreachable, `_analyze_with_ollama()` catches the connection error and routes the entire batch through `_created_degraded_results()`, which assigns `NEUTRAL` sentiment and `status: 'degraded'` to every record.

The DAG completes successfully. Downstream dashboards can filter on `status == 'degraded'` to accurately report infrastructure outages without data loss.

---

## 📋 Prerequisites

- **Python 3.9 – 3.12** (strictly enforced by `apache-airflow-providers-fab>=3.0.0`)

  > ⚠️ **Why this range?** Python 3.8 reached EOL in October 2024 (lower bound). Python 3.13 introduced breaking changes to C extensions that core Airflow dependencies (SQLAlchemy, Flask-AppBuilder) have not yet fully absorbed (upper bound).

- **Ollama** installed and running locally → [Install Ollama](https://ollama.com)
- **llama3.2** model pulled: `ollama pull llama3.2`
- **Apache Airflow 3.0+**

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/self-healing-pipeline.git
cd self-healing-pipeline
```

### 2. Create a virtual environment with Python 3.12

```bash
# Using uv (recommended)
uv venv --python 3.12
source .venv/bin/activate

# Or with standard venv
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Network troubleshooting:** If you encounter `Connection refused (os error 61)` during installation, check for residual proxy environment variables from VPN software:
> ```bash
> env | grep -i proxy
> unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
> ```

### 4. Configure Airflow home directory

```bash
# Set AIRFLOW_HOME to the project root — do this before any airflow commands
export AIRFLOW_HOME=$(pwd)
```

> ⚠️ If `AIRFLOW_HOME` is not set, Airflow defaults to `~/airflow` and your DAG will not be discovered.

### 5. Initialize the Airflow database

```bash
airflow db migrate
# Expected output: "Database migrating done!"
```

### 6. Disable example DAGs

In `airflow.cfg`, set:

```ini
load_examples = False
```

### 7. Configure environment variables (optional overrides)

```bash
export PIPELINE_BASE_DIR="/path/to/your/project"
export PIPELINE_INPUT_FILE="/path/to/your/reviews.json"
export PIPELINE_OUTPUT_DIR="/path/to/output"
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="llama3.2"
```

### 8. Start Airflow

```bash
airflow standalone
```

Copy the admin password from the startup log:

```
Simple auth manager | Password for user 'admin': <your-password>
```

Navigate to `http://localhost:8080` and log in.

---

## 📖 Usage Guide

### Triggering the DAG

1. In the Airflow UI, locate `self_healing_pipeline_dag`.
2. Click **Trigger DAG w/ config**.
3. Adjust parameters as needed:

| Parameter | Default | Description |
|---|---|---|
| `input_file` | `./input/academic_dataset_review.json` | Path to JSONL review file |
| `output_dir` | `./output` | Directory for JSON output files |
| `batch_size` | `100` | Number of records per batch |
| `offset` | `0` | Starting line offset in the input file |
| `ollama_model` | `llama3.2` | Ollama model name |

### Reading the Health Report

The final task `generate_health_report` logs a structured JSON report:

```json
{
  "pipeline": "self_healing_pipeline",
  "health_status": "HEALTHY",
  "metrics": {
    "total_processed": 100,
    "success_count": 87,
    "healed_count": 13,
    "degraded_count": 0,
    "success_rate": 0.87,
    "healed_rate": 0.13,
    "degraded_rate": 0.0
  },
  "sentiment_distribution": {
    "POSITIVE": 42,
    "NEGATIVE": 31,
    "NEUTRAL": 27
  }
}
```

**Health Status Thresholds:**

| Status | Condition | Meaning |
|---|---|---|
| `HEALTHY` | `degraded == 0` AND `healed ≤ 50%` | All systems nominal |
| `WARNING` | `healed > 50%` | High upstream data drift or corruption |
| `DEGRADED` | `degraded > 0` | Partial inference engine failure |
| `CRITICAL` | `degraded > 10%` | Severe backend outage |

> **Note:** These thresholds are hardcoded for POC purposes. Production deployments should expose them as Airflow Variables, calibrated against downstream SLA requirements with data consumers.

### Output Files

Batch results are persisted to:

```
output/sentiment_analysis_summary_<YYYYMMDD-HHMMSS>_<offset>.json
```

---

## 📁 Folder Structure

```
self-healing/
├── dags/
│   └── agentic_pipeline_dag.py   # Core DAG definition + all helper functions
├── input/
│   └── academic_dataset_review.json  # Sample JSONL review dataset
├── output/                           # Generated at runtime
│   └── sentiment_analysis_summary_*.json
├── pyproject.toml                    # Project metadata (asyncpg dependency)
├── requirements.txt                  # Full dependency list
└── README.md
```

---

## 🔭 Known Limitations & Future Work

| Limitation | Current State | Production Path |
|---|---|---|
| **Inference throughput** | Single Ollama instance; ~minutes per 100-record batch | Replace with `vLLM` for high-throughput batching; distribute across multiple Ollama workers |
| **Health thresholds** | Hardcoded in DAG code | Externalize to Airflow Variables; negotiate thresholds against SLA with data consumers |
| **XCom data isolation** | `aggregate_results` writes to local disk | Migrate to S3/GCS; XCom carries only the object URI |
| **Alerting** | Health status logged only | Integrate with PagerDuty, Slack webhook, or Airflow callbacks on `CRITICAL` status |
| **Healing coverage** | 5 rule-based strategies | Add ML-based anomaly detection for unknown corruption patterns |
| **Parallel processing** | Sequential per-item inference | Implement Airflow dynamic task mapping for parallel batch processing |


## 🙏 Acknowledgments

**CodeWithYu** provides this useful and insight data engineering project - Self-Healing Project
