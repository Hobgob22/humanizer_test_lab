# Humanizer Test Bench (Test Lab v2)

## Overview

Humanizer Test Bench is a comprehensive platform for evaluating text humanization models. It rewrites text through multiple LLM providers, evaluates AI detection scores, and provides detailed analytics on model performance. The platform features a modern **React** frontend with a **FastAPI** backend.

## Features

### 🧪 Writing Profile Lab (v2)
- Generate a structured **writing profile** from a sample (multi-provider structured output)
- Copy/paste profile payloads between pages for reproducible experiments
- Humanize a text using a selected writing profile with **user/system prompt injection modes**
- Built-in **stop buttons** and resilient error handling for long-running actions

### 🎯 User-Style Humanization Tests
- Dedicated model section for “user-style” experiments
- Paste a writing profile and choose injection mode (system/user)
- Optional **Style Adherence** evaluator (Gemini 2.5 Flash) to score how well outputs adopt the target style

### 🤖 Multi-Provider Humanization
- **OpenAI** (gpt-4.1, gpt-4o, gpt-4.1-mini, 15+ fine-tuned models)
- **Claude** (Sonnet 4, Sonnet 3.7, Haiku 3.5)
- **Google Gemini** (2.0 Flash, 2.5 Flash, 2.5 Pro)
- Support for custom dynamic prompts and system configurations

### 🔍 AI Detection & Quality Analysis
- **GPTZero** and **Sapling** AI detection with caching
- **Gemini-based Quality Checks** (semantic meaning, citations, grammar)
- **Gemini-based Style Adherence** (optional; shown only when present in results)
- **Statistical Analysis** (mean, median, percentiles, zero-shot success rates)

### 💻 Modern Web Interface
- **React + Vite** frontend with real-time updates
- **FastAPI** backend with WebSocket support
- **Interactive Dashboard** with job monitoring
- **Comprehensive Analytics** with charts and data tables
- **Document Browser** for detailed result exploration

### 🚀 Production-Ready
- **Docker** & **Docker Compose** configurations
- **Fly.io** deployment ready
- **SQLite + Turso** dual-database support
- Automatic backups and disaster recovery
- Health checks and monitoring

## Architecture

```
┌─────────────────┐
│  React Frontend │  ← Vite, TailwindCSS, React Router
│  (Port 5173)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Backend│  ← REST API + WebSocket
│  (Port 8000)    │
└────────┬────────┘
         │
         ├─────────→ OpenAI API
         ├─────────→ Anthropic Claude API
         ├─────────→ Google Gemini API
         ├─────────→ GPTZero API
         ├─────────→ Sapling API
         │
         └─────────→ SQLite / Turso DB
```

## Quick Start

### Prerequisites
- **Python 3.11+**
- **Node.js 22+** and **npm**
- **Docker** (optional, for containerized deployment)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd humanizer_test_lab
```

### 2. Set Up Environment

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` and set your API keys:

```bash
# Required
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
CLAUDE_API_KEY=...

# Optional but recommended
GPTZERO_API_KEY=...
SAPLING_PRIMARY_API_KEY=...

# Authentication
APP_AUTH_KEY=your-secure-password

# Database (optional - uses local SQLite if not set)
TURSO_DATABASE_URL=...
TURSO_AUTH_TOKEN=...
```

### 3. Development Setup

#### Backend

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run FastAPI server with hot reload
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The application will be available at:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 4. Production Build

```bash
# Build React frontend
cd frontend
npm run build
cd ..

# Run FastAPI server (serves React build automatically)
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000 to access the application.

## Docker Deployment

### Development

```bash
docker-compose up
```

### Production

```bash
# Build and run production container
docker build -t humanizer-test-bench .
docker run -p 8000:8000 --env-file .env humanizer-test-bench
```

## Fly.io Deployment

```bash
# Install Fly.io CLI
curl -L https://fly.io/install.sh | sh

# Login to Fly.io
fly auth login

# Create app (first time only)
fly launch

# Deploy
fly deploy

# Set secrets
fly secrets set OPENAI_API_KEY=sk-...
fly secrets set GEMINI_API_KEY=...
fly secrets set CLAUDE_API_KEY=...
fly secrets set APP_AUTH_KEY=your-secure-password
```

## Usage

### Creating a New Benchmark Run

1. Navigate to **New Run** page
2. Configure run settings:
   - Run name
   - Document folders (AI texts, Human texts, etc.)
   - Models to test
   - Number of iterations
   - Detection options (GPTZero, Sapling)
3. Click **Start Run**
4. Monitor progress in **Job Status** page

### Analyzing Results

- **Benchmark Analysis**: Compare models with statistical summaries and charts
- **Document Browser**: Explore individual document results
- **Preview Results**: Quick model screening and rankings
- **Job Status**: Real-time job monitoring with logs

## Project Structure

```
humanizer_test_lab/
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/    # Reusable UI components
│   │   ├── pages/         # Main application pages
│   │   ├── lib/          # Utilities and API client
│   │   └── App.jsx       # Root component
│   ├── package.json
│   └── vite.config.js
├── src/                   # Python backend
│   ├── api/              # FastAPI application
│   │   ├── routes/       # API endpoints
│   │   ├── main.py      # FastAPI app entry
│   │   └── models.py    # Pydantic schemas
│   ├── humanizers/       # LLM provider wrappers
│   ├── detectors/        # AI detection services
│   ├── evaluation/       # Quality checking
│   ├── job_manager.py   # Background job processing
│   ├── pipeline.py      # Main processing pipeline
│   ├── results_db.py    # Database layer
│   └── models.py        # Model registry
├── data/                 # Document storage
│   ├── ai_texts/
│   ├── human_texts/
│   ├── ai_paras/
│   └── human_paras/
├── results/             # Run results and database
├── Dockerfile          # Production container
├── Dockerfile.dev      # Development container
├── docker-compose.yml  # Docker Compose config
├── fly.toml           # Fly.io configuration
└── README.md
```

## API Documentation

Once the backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Key Endpoints

- `GET /api/health` - Health check
- `POST /api/jobs/` - Create new benchmark job
- `GET /api/jobs/` - List all jobs
- `GET /api/jobs/{job_id}` - Get job details
- `POST /api/jobs/{job_id}/cancel` - Cancel running job
- `GET /api/runs/` - List completed runs
- `GET /api/runs/{run_name}` - Load run data
- `POST /api/statistics/` - Compute statistics
- `WS /api/ws` - WebSocket for real-time updates

## Configuration

### Model Registry

Models are configured in `src/models.py`. The registry includes:
- Vanilla LLMs (GPT-4, Claude, Gemini)
- Fine-tuned models
- Dynamic prompt models with customizable system prompts

### Rate Limiting

Rate limits are configured per provider:
- OpenAI: 1500 req/min
- Gemini: 700 req/min
- Claude: 700 req/min
- GPTZero: 500 req/min
- Sapling: 120,000 chars/2 min

## Development

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Run dev server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Backend Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run with auto-reload
python -m uvicorn src.api.main:app --reload

# Run tests (if available)
pytest
```

## Troubleshooting

### Frontend won't build
- Ensure Node.js 22+ is installed
- Delete `node_modules` and `package-lock.json`, then run `npm install`
- Check for TypeScript/ESLint errors

### Backend API errors
- Verify all required API keys are set in `.env`
- Check Python version (3.11+ required)
- Ensure all dependencies are installed

### Database issues
- Local SQLite is used by default in `results/runs.sqlite`
- For Turso, verify `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN`
- Check file permissions on `results/` directory

## Migration from Streamlit

This application was migrated from Streamlit to React + FastAPI for improved performance and deployment flexibility. All functionality has been preserved:

- ✅ Job creation and monitoring
- ✅ Real-time progress updates via WebSocket
- ✅ Benchmark analysis with statistics and charts
- ✅ Document browsing and detailed results
- ✅ Model comparison and preview results

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Add your license here]

## Support

For issues and questions, please open an issue on GitHub.

---

**Built with**: React, Vite, TailwindCSS, FastAPI, Python, OpenAI, Claude, Gemini
