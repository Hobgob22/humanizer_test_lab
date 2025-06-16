# Humanizer Test Bench

## Overview

Humanizer Test Bench is a comprehensive toolkit for rewriting text through multiple LLM “humanizer” prompts/models and evaluating how AI detectors rate the output. It combines OpenAI and Google Gemini for rewriting, GPTZero and Sapling for detection, plus built-in quality, semantic, and statistical analyses.

## Features

- **Multi-Provider Humanization**  
  - OpenAI (`gpt-4o`, `gpt-4.1`, fine-tuned models)  
  - Google Gemini Flash  
- **AI Detection**  
  - GPTZero & Sapling wrappers with caching and retry logic  
- **Quality & Semantic Checks**  
  - Citation & structure validation via Gemini  
  - Descriptive statistics (mean, percentiles, histograms)  
- **Interfaces**  
  - **Streamlit UI** (`src/ui.py`) with live logs and charts  
  - **CLI** (`src/cli.py`) for batch processing  
- **Robust Pipeline**  
  - Caching, rate-limiting, retry policies, and SQLite storage  
- **DevOps-Ready**  
  - Docker & Docker Compose configurations  
  - `Makefile` convenience targets  
  - systemd service for production

## Setup

### 1. Prerequisites

- **Git**  
- **Python 3.10+**  
- **Docker Engine 20.10+** & **Docker Compose 2.0+**  
- **Make** (optional, but recommended for shortcuts)  

### 2. Clone & Environment

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
````

Copy the example environment file and populate your API keys:

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```ini
APP_AUTH_KEY=your_ui_password

# OpenAI
OPENAI_API_KEY=sk-...
HUMANIZER_OPENAI_API_KEY=sk-...

# Detectors
GPTZERO_API_KEY=...
SAPLING_API_KEY=...

# Google Gemini
GEMINI_API_KEY=...

# (Optional) Tuning defaults
REHUMANIZE_N=5
ZERO_SHOT_THRESHOLD=0.10
MIN_WORDS_PARAGRAPH=15
MAX_ITER=5
```

### 3. Python Virtual Environment (Optional)

If you want to run locally without Docker:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # for testing & linters
pre-commit install
```

### 4. Initialize Project Directories

```bash
make init
```

This will ensure `data/`, `cache/`, `logs/`, and `results/` folders exist.

### 5. Development Server (Hot Reload)

```bash
make dev
```

* Streams logs to your terminal.
* UI available at [http://localhost:8501](http://localhost:8501).
* Code changes in `src/` reload automatically.

To run in background:

```bash
make dev-d
```

### 6. Production Server

```bash
make prod
```

* Brings up the production image on port 8501.
* For nginx-backed HTTPS, use:

```bash
make prod-nginx
```

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for detailed production and SSL setup.

### 7. Building & Cleaning

* **Build all images:**

  ```bash
  make build
  ```
* **Build dev/prod image only:**

  ```bash
  make build-dev
  make build-prod
  ```
* **Stop & remove containers:**

  ```bash
  make clean
  ```
* **Prune Docker system:**

  ```bash
  make docker-clean
  ```

### 8. Useful Make Targets

* **View logs:**

  ```bash
  make logs-dev     # development logs
  make logs-prod    # production logs
  ```
* **Open shell:**

  ```bash
  make shell        # dev container
  make shell-prod   # prod container
  ```
* **Run CLI inside container:**

  ```bash
  make cli ARGS="--folder data/ai_texts --models gpt-4o,gpt-4.1 --iters 5"
  ```
* **Tests & QA:**

  ```bash
  make test         # run pytest
  make lint         # flake8 & mypy
  make format       # black & isort
  ```
* **Backup & Restore:**

  ```bash
  make backup       # archive data/results
  make restore      # list and restore backups
  ```
* **Environment Check:**

  ```bash
  make check-env
  ```
* **Dependencies Update:**

  ```bash
  make update-deps
  ```
* **Resource Monitoring:**

  ```bash
  make stats        # one-shot docker stats
  make monitor      # live docker stats
  ```

## Usage

### Streamlit UI

```bash
streamlit run src/ui.py
```

Log in with your `APP_AUTH_KEY`, configure inputs/models/iterations, and start a run.

### CLI

```bash
python -m src.cli \
  --folder data/ai_texts \
  --models gpt-4o,gpt-4.1,gemini-2.0-flash \
  --iters 5 \
  --out results/out.json
```

Run `python -m src.cli --help` for all available options.

## License

MIT License — see [LICENSE](LICENSE) for details.
