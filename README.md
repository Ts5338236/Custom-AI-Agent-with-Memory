# Custom AI Agent with Memory - Production Platform

A scalable, intelligent AI agent platform built with FastAPI, LangChain, and Next.js.

## 🚀 Key Features
- **Multi-Agent Orchestration**: Planner, Executor, and Reviewer agents for complex tasks.
- **Hybrid Memory**: FAISS-based vector search + SQL-based Knowledge Graph for deep personalization.
- **Security & Privacy**: JWT Auth, RBAC, and automated PII Redaction.
- **Resilience**: Built-in retries, circuit breakers, and response caching.
- **Observability**: Deep tracing of AI reasoning and Prometheus monitoring.
- **Streaming UI**: Modern React chat interface with real-time SSE updates.

## 🛠 Tech Stack
- **Backend**: Python (FastAPI, SQLModel, SQLAlchemy)
- **AI**: LangChain, OpenAI (GPT-4o)
- **Database**: SQLite (Local Dev) / PostgreSQL (Production)
- **Vector Store**: FAISS
- **Frontend**: Next.js 14, Tailwind CSS

## ⚙️ Setup Instructions

### 1. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # Add your OPENAI_API_KEY
python -m uvicorn app.main:app --reload
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 🧪 Testing & Evaluation
Run the automated benchmark suite:
```bash
cd backend
pytest tests/test_prompts.py
```

## 🐳 Docker Deployment
```bash
docker-compose up --build
```

## 📄 Documentation
- **Architecture**: [architecture_whitepaper.md](./architecture_whitepaper.md)
- **API Docs**: Available at `http://localhost:8000/docs` (Swagger UI)

---
Developed as a production-grade AI solution.
