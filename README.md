# Humanizer Test Bench

A comprehensive toolkit for testing, evaluating, and benchmarking AI-driven text "humanization" pipelines. This project provides a streamlined workflow for rewriting documents with leading LLMs (OpenAI, Gemini) and assessing their AI-detection scores (GPTZero, Sapling).

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CI](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/<your-repo>/actions)

## Overview

In an era of advanced language models, distinguishing between human and AI-generated text is a growing challenge. This "Humanizer Test Bench" was built to systematically explore this space. It allows you to take any text, process it through various "humanizer" prompts and models, and immediately see how AI detectors perceive the result.

## Features

-   **Multi-Provider Humanization**:
    -   Rewrite documents and paragraphs using **OpenAI** (e.g., `gpt-4o`, `gpt-4-turbo`) and **Google Gemini** models.
-   **AI Content Detection**:
    -   Score rewritten text for AI-generated content using **GPTZero** and **Sapling**.
-   **Quality & Semantic Analysis**:
    -   Perform quality checks and calculate semantic similarity scores to ensure the rewritten text retains its original meaning.
-   **Dual Interfaces**:
    -   **Streamlit UI**: A rich, interactive dashboard for hands-on experiments.
    -   **CLI**: A powerful command-line tool for batch processing and automation.
-   **Robust Pipeline**:
    -   Built-in caching, API rate-limiting, and SQLite result storage for efficient and repeatable experiments.

## Getting Started

### Prerequisites

-   Python 3.10+
-   Git

### 1. Installation

First, clone the repository and navigate into the project directory.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

Next, create and activate a virtual environment.

```bash
# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# On Windows
python -m venv .venv
.\.venv\Scripts\activate
```

Finally, install the required dependencies.

```bash
pip install -r requirements.txt
```

### 2. Configuration

The application requires API keys to interact with external services.

Copy the example environment file:
```bash
cp .env.example .env
```

Now, open the `.env` file in your editor and add your API keys. **The application will not function without them.**

```ini
OPENAI_API_KEY="your_openai_key"
GEMINI_API_KEY="your_gemini_key"
GPTZERO_API_KEY="your_gptzero_key"
SAPLING_API_KEY="your_sapling_key"
```

## Usage

This tool can be run as an interactive web app or as a command-line tool.

### Streamlit UI

For interactive analysis and single-document processing, launch the Streamlit dashboard.

```bash
streamlit run src/ui.py
```

### Command-Line Interface (CLI)

For batch processing, use the CLI. The example below processes all `.docx` files in a folder using two different models and five rewrite iterations per document.

```bash
python -m src.cli \
  --input-folder data/ai_texts \
  --models gpt-4o,gemini-1.5-pro \
  --iterations 5
```

For a full list of commands and options, run:
```bash
python -m src.cli --help
```

## Project Structure

The repository is organized to separate concerns, making it easy to extend and maintain.

```
.
├── data/                  # Sample input documents (.docx)
├── src/
│   ├── cache.py           # Caching for API calls
│   ├── cli.py             # Command-line interface logic
│   ├── config.py          # Environment variable loading
│   ├── detectors/         # AI detection API clients (GPTZero, Sapling)
│   ├── docx_utils.py      # Utilities for reading/writing .docx files
│   ├── evaluation/        # Semantic similarity and quality checks
│   ├── humanizers/        # Core text rewriting logic
│   ├── metrics.py         # Scoring and metric calculation functions
│   ├── models.py          # Wrappers for OpenAI and Gemini models
│   ├── pipeline.py        # Orchestrates the end-to-end workflow
│   ├── prompts.py         # Prompt templates for humanization
│   ├── rate_limiter.py    # API rate-limiting utilities
│   ├── results_db.py      # SQLite database for storing results
│   └── ui.py              # Streamlit web interface
├── .env.example           # Example environment file
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Contributing

Contributions are highly welcome! To contribute:

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/your-amazing-feature`).
3.  Commit your changes (`git commit -m "feat: Add some amazing feature"`).
4.  Push to your branch (`git push origin feature/your-amazing-feature`).
5.  Open a Pull Request.

For major changes, please open an issue first to discuss what you would like to change. Please ensure your code follows the existing style and that all tests pass.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
