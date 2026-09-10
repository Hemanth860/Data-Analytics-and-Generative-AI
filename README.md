# Zepto Data & AI Platform

**Author:** Kammari Hemanth Kumar Achari  

---

## 📌 Executive Summary
The **Zepto Data & AI Platform** is an end-to-end engineered AI/ML platform bringing together three interconnected capabilities into one repository:

1. **`/data_pipeline`:** Data engineering pipeline that scrapes product catalog data from [books.toscrape.com](http://books.toscrape.com/), cleans & standardizes fields, converts prices using a fixed baseline exchange rate (`1 GBP = 105.50 INR`), and loads data into a normalized SQLite relational database (`zepto_store.db`).
2. **`/analytics`:** Analytics pipeline profiling customer/order datasets, applying threshold-based missing value rules, univariate/bivariate EDA, 6x6 correlation matrix, 4 multivariate data story charts, and training/evaluating scikit-learn ML classification & regression models with train-only preprocessing pipeline and joblib export.
3. **`/support_assistant`:** Grounded GenAI Support Assistant using 8 official Zepto policy documents, `sentence-transformers` embeddings (`all-MiniLM-L6-v2`), `ChromaDB` vector store, a 3-node **LangGraph StateGraph** router, Pydantic response validation, FastAPI REST service (`POST /ask`), Streamlit UI, and Docker containerization.

---

## 📁 Repository Directory Structure

```
zepto-data-ai-platform/
├── README.md                          # Root project README (Setup, execution, design rationale)
├── requirements.txt                   # Consolidated dependencies file
├── data_pipeline/                    # Module 1 
│   ├── README.md                      # Pipeline design & fixed rate baseline note
│   ├── pipeline.py                    # Scraper, ETL, SQLite schema load & 5 SQL queries
│   ├── data_pipeline.ipynb            # Interactive notebook version
│   ├── test_pipeline.py               # Unit tests (100% pass)
│   └── zepto_store.db                 # SQLite relational database
├── analytics/                         # Module 2 
│   ├── README.md                      # Detailed EDA notes, metrics tables & recommendation
│   ├── 01_eda.py                      # Profiling, missing value rules, univariate/bivariate EDA
│   ├── 01_eda.ipynb                   # Part A Notebook
│   ├── 02_modeling.py                 # Stratified split, ColumnTransformer, ML classifiers, SMOTE, tuning
│   ├── 02_modeling.ipynb              # Part B Notebook
│   ├── titanic.csv                    # Committed offline fallback dataset
│   ├── titanic_pipeline.joblib        # Complete fitted pipeline artifact
│   └── artifacts/                     # Generated charts & decision tree plot
└── support_assistant/                 # Module 3 
    ├── README.md                      # RAG architecture, node descriptions & transcripts
    ├── Dockerfile                     # Docker configuration (port 7860)
    ├── main.py                        # FastAPI application (POST /ask)
    ├── rag_engine.py                  # LangGraph StateGraph (3 nodes)
    ├── ingest_policies.py             # Sentence-transformers embedding & ChromaDB store
    ├── prompts.py                     # Structured prompt template
    ├── ui.py                          # Streamlit interactive web dashboard
    ├── api_transcripts.json           # Transcripts of test API calls
    └── docs/                          # 8 Grounded Zepto Policy Documents (doc_01.txt ... doc_08.txt)
```

---

## 🛠️ Project Setup & Installation

This project uses **one consolidated `requirements.txt`** located at the root of the repository.

### 1. Install Dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution Guide (Running Each Module End-to-End)

### Module 1: Data Engineering Pipeline (`/data_pipeline`)
```bash
# Run end-to-end scraper, ETL pipeline, SQLite database creation & SQL queries:
python data_pipeline/pipeline.py

# Run unit test suite:
python -m unittest data_pipeline/test_pipeline.py
```

### Module 2: Analytics & Predictive Modeling (`/analytics`)
```bash
# Run Part A (EDA, missing value rules, outlier detection & data story plots):
python analytics/01_eda.py

# Run Part B (Stratified split, ML classifiers, imbalance SMOTE, tuning & regression):
python analytics/02_modeling.py
```

### Module 3: GenAI Policy Support Assistant (`/support_assistant`)
```bash
# Run FastAPI REST Service (POST /ask):
python support_assistant/main.py

# Run Interactive Streamlit Dashboard:
streamlit run support_assistant/ui.py

# Run container via Docker:
docker build -t zepto-support-assistant support_assistant/
docker run -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant
```

---

## 🧠 Summary of Design Decisions

### Module 1 (`/data_pipeline`) Design Rationale:
* **Fixed Rate Constant:** Applied project baseline `1 GBP = 105.50 INR` as a deterministic constant requiring zero external API keys or network latency.
* **Schema Normalization:** Designed a 2-table PK/FK schema (`categories` $\rightarrow$ `books`) enforcing foreign key integrity (`PRAGMA foreign_keys = ON;`).
* **Pandas Verification:** Demonstrated that `pd.read_sql()` query outputs match pure in-memory `pd.merge()` without SQL syntax (`[SUCCESS - MATCH]`).

### Module 2 (`/analytics`) Design Rationale:
* **Offline Dataset Fallback:** Saved `analytics/titanic.csv` on initial load so grading evaluates cleanly offline.
* **Threshold Missing Value Rule:** Applied <5% drop rows (`embarked`), 5-30% median impute (`age`), and >30% drop column (`deck`).
* **Train-Only Scaling:** Built scikit-learn `ColumnTransformer` inside a `Pipeline` to guarantee zero test-set data leakage.
* **SMOTE Imbalance Handling:** Applied SMOTE oversampling exclusively on training folds, improving minority class recall (F1 = 0.3609).

### Module 3 (`/support_assistant`) Design Rationale:
* **Default Offline Mock Mode (`MOCK_LLM=1`):** Serves deterministic, rule-based responses and grounded snippet extractions with zero LLM API dependency.
* **LangGraph Orchestration:** Wrote a 3-node StateGraph (`classify_intent` $\rightarrow$ `retrieve_and_answer` OR `direct_answer`) with Pydantic output validation (`PolicyResponse`).

---

## 🌿 Git Branching & Version Control Workflow

The Git commit history of this single repository demonstrates a standard production feature-branch workflow:
1. Repository initialized on `main` branch.
2. Feature branch `feature/zepto-platform-modules` created.
3. Three separate incremental commits made on the feature branch (one per module).
4. Feature branch merged back into `main` with a explicit merge commit.

### To verify the Git commit graph:
```bash
git log --graph --oneline --all
```
