# 🚀 AgentForge – Agentic RAG + Multimodal AI Platform
### *Project for AI Engineer Role at Madeline & Co.*

AgentForge is a production-ready backend system that showcases practical AI engineering skills across:

- Agentic reasoning (planner + tool use)
- RAG (Retrieval-Augmented Generation)
- Multimodal pipelines (Text + Image + PDF → Text)
- Scalable backend design (FastAPI, PostgreSQL, Qdrant, Docker)
- Local LLM inference using Ollama (LLaMA 3.1)

This project demonstrates end-to-end AI system building with clean architecture, evaluation tooling, and deployment support.

---

# ✨ Features Overview

## ✅ 1. Ingestion Pipelines
Supports uploading:

- Text (`.txt`)
- PDF → converted to text using PyPDF (in-memory, no temporary file)
- Images → captioned + embedded
- Audio (module stub present)
- Video (module stub present)

Each ingestion runs:

- Text chunking  
- CLIP embeddings  
- Captioning for images  
- Stores metadata (PostgreSQL)  
- Stores embeddings (Qdrant)  

Endpoint:

POST /ingest/ingest

---

## ✅ 2. RAG Engine (Vector Search + LLM)
- Uses Qdrant for vector retrieval  
- ROUGE-L scoring  
- Similarity metrics  
- Cost estimation  
- Context assembly  
- Answer generation using LLaMA 3.1 (Ollama)  

Endpoint:

POST /query/

---

## ✅ 3. Agentic Workflow
Two fully implemented agent systems:

### 🔹 A. Controller Agent  
A rule-based agent with:

- Planning  
- RAG Tool  
- Calculator Tool  
- Web scraper tool  
- LLaMA synthesis  

### 🔹 B. LangGraph-style Agent  
Implements LangGraph principles:

- State machine (plan → act → final)  
- Executes tools step-by-step  
- Produces structured reasoning trace  
- Outputs normalized JSON  

Endpoint:

POST /agent/

---

## ✅ 4. Multimodal Query Engine
Supports full multimodal flow (required by assignment):

✔️ Combine image + text document (PDF)  
✔️ Retrieve context  
✔️ Fuse into final answer via LLaMA  

Example:

“Based on this PDF and this image, explain how wind turbines generate clean energy.”

Endpoint:

POST /multimodal/

---

## ✅ 5. Evaluation Notebook (Required)
The notebook:

- Runs 6 agent tasks  
- Compares Controller vs LangGraph agent  
- Measures latency  
- Computes RAG similarity / ROUGE  
- Logs results in a dataframe  

File:

evaluation.ipynb

---

## ✅ 6. Production-Level Engineering
- FastAPI + Pydantic  
- PostgreSQL (dockerized)  
- Qdrant Vector DB (dockerized)  
- Ollama LLaMA model  
- Dockerfile + docker-compose.yml  
- Latency logging (file logs)  
- Modular folder architecture  

---

# 🗂 Project Structure

AGENTFORGE/
│
├── app/
│   ├── api/                    # FastAPI routes
│   ├── agents/                 # Controller + LangGraph agent
│   ├── core/                   # Schemas, configs
│   ├── eval/                   # Cost + evaluation metrics
│   ├── multimodal/             # Image / PDF processors
│   ├── services/               # Llama, embeddings, RAG, Qdrant
│   ├── utils/                  # Latency, chunking
│   └── main.py                 # FastAPI entrypoint
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── evaluation.ipynb
├── README.md
└── examples/

---

# ⚡ Installation (Dockerized)

### 1. Ensure Docker Desktop is running.

### 2. Build & launch the entire stack:

docker compose up –build

### 3. Access FastAPI docs:

http://localhost:8000

Swagger will list all routes.

---

# 🔌 API Endpoints

## Health Check

GET /health/health

## Ingest File

POST /ingest/ingest

## RAG Query

POST /query/

## Agent (Controller or LangGraph)

POST /agent/
{
“task”: “Summarize the story”,
“engine”: “controller”
}

## Multimodal Query

POST /multimodal/

---

# 🧪 Evaluation

Open:

evaluation.ipynb

Generates:

- Answer quality  
- RAG metrics  
- Latency  
- Controller vs LangGraph comparison  
- Multimodal evaluation  

---

# 🛠 Tech Stack

- Python 3.11 (inside container)
- FastAPI + Pydantic
- LangChain & LangGraph-style planning
- Qdrant vector DB
- PostgreSQL + Alembic
- Transformers (CLIP)
- Sentence Transformers
- PyPDF
- Uvicorn
- Docker + Docker Compose

---

# 🚀 Deployment (Local / Container)

Everything is containerized.

Start entire system:

docker compose up –build

Stop system:

docker compose down

---

# 🎥 Video Demo (5 minutes)
Include:

1. Architecture overview  
2. Ingestion demo  
3. RAG query demo  
4. Agent demo  
5. Multimodal demo  
6. Docker launch  

---
