# Nexus - Build a Vector Database from Scratch in Java + React

A fully working **Vector Database** built from scratch in **Java** with a **React** web UI.
Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms side-by-side, plus a **RAG pipeline** powered by a local LLM via Ollama.

---

## What This Project Does

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW, KD-Tree, Brute Force - run all three and compare speed |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories: CS, Math, Food, Sports |
| **2D PCA Scatter Plot** | Live visualization of semantic space in the React UI |
| **Real Document Embedding** | Paste any text and Ollama embeds it with `nomic-embed-text` |
| **RAG Pipeline** | Ask questions about your documents, retrieve context with HNSW, answer with local LLM |
| **Full REST API** | CRUD endpoints for insert, delete, search, benchmark, hnsw-info, docs, and RAG |

---

## How It Works

```text
Your Text
    |
    v
Ollama (nomic-embed-text)          <- converts text to an embedding vector
    |
    v
HNSW Index (Java)                  <- indexes the vector in a multilayer graph
    |
    v
Semantic Search                    <- finds nearest neighbors in vector space
    |
    v
Ollama (llama3.2)                  <- reads retrieved chunks and generates an answer
    |
    v
Answer
```

**HNSW (Hierarchical Navigable Small World)** builds a multilayer graph where higher layers are progressively sparser. Searches start from the top layer and move down toward the closest neighborhood, which makes search much faster than scanning every vector.

---

## Prerequisites

You need these installed on Windows:

1. **Java 17 or newer**
2. **Node.js + npm**
3. **Git**
4. **Ollama** for document embedding and RAG features

The demo vector search works without Ollama. The **Documents** tab needs `nomic-embed-text`, and the **Ask AI** tab needs `llama3.2`.

---

## Step-by-Step Setup (Windows)

### Step 1 - Install Java

Install Java 17 or newer. You can use Microsoft OpenJDK, Eclipse Temurin, or any Java 17+ distribution.

Verify in PowerShell:

```powershell
java -version
javac -version
```

You should see version `17` or newer.

### Step 2 - Install Node.js

Install Node.js from:

```text
https://nodejs.org
```

Verify in PowerShell:

```powershell
node -v
npm -v
```

### Step 3 - Install Git

Download Git for Windows:

```text
https://git-scm.com/download/win
```

Verify:

```powershell
git --version
```

### Step 4 - Install Ollama (Local AI Models)

1. Go to `https://ollama.com`
2. Download and install Ollama for Windows
3. Launch the Ollama desktop app once
4. Open a new PowerShell and pull the two required models:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

If PowerShell does not recognize `ollama`, use the installed executable directly:

```powershell
& 'C:\Users\shiva\AppData\Local\Programs\Ollama\ollama.exe' pull nomic-embed-text
& 'C:\Users\shiva\AppData\Local\Programs\Ollama\ollama.exe' pull llama3.2
```

Verify Ollama:

```powershell
ollama list
```

Or, if using the full path:

```powershell
& 'C:\Users\shiva\AppData\Local\Programs\Ollama\ollama.exe' list
```

If Ollama is not already running, start the desktop app or run:

```powershell
ollama serve
```

> Minimum specs for Ollama: 8GB RAM recommended.

---

## Build and Run

Run all commands from the project folder:

```powershell
cd "C:\Users\shiva\Documents\New project"
```

### Step 1 - Install Frontend Dependencies

```powershell
cd frontend
npm install
```

### Step 2 - Build the React UI

```powershell
npm run build
```

This creates:

```text
frontend/dist/
```

The Java server serves this built React app automatically.

### Step 3 - Compile the Java Backend

Go back to the project root:

```powershell
cd ..
javac -d backend/out backend/src/main/java/com/yourownai/App.java
```

### Step 4 - Start the Server

```powershell
java -cp backend/out com.yourownai.App
```

You should see:

```text
=== Nexus Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

Open:

```text
http://localhost:8080
```

Quick health checks:

```powershell
Invoke-WebRequest -Uri 'http://localhost:8080/stats' -UseBasicParsing
Invoke-WebRequest -Uri 'http://localhost:8080/status' -UseBasicParsing
```

In `/status`, `ollamaAvailable` should be `true` when Ollama is running.

### If Port 8080 Is Busy

You can run on another port:

```powershell
java -cp backend/out com.yourownai.App 18080
```

Then open:

```text
http://localhost:18080
```

### Fast Restart After Code Changes

If only Java changed:

```powershell
javac -d backend/out backend/src/main/java/com/yourownai/App.java
java -cp backend/out com.yourownai.App
```

If only React changed:

```powershell
cd frontend
npm run build
cd ..
java -cp backend/out com.yourownai.App
```

---

## Development Mode

For frontend development, run the Java backend first from the project root:

```powershell
java -cp backend/out com.yourownai.App
```

Then start Vite in another terminal:

```powershell
cd frontend
npm run dev
```

Open the Vite URL, usually:

```text
http://localhost:5173
```

When running through Vite, the React app calls the API at `http://localhost:8080`, so keep the Java backend running.

---

## Using the Application

### Tab 1: Search (Demo Vectors)

- Type any concept in the search box: `binary tree`, `sushi`, `basketball`, `calculus`
- Choose your algorithm: **HNSW**, **KD-Tree**, or **Brute Force**
- Choose distance metric: **Cosine**, **Euclidean**, or **Manhattan**
- Click **SEARCH** to see the top matches
- Click **COMPARE ALL ALGOS** to run all 3 algorithms and compare their speed

The scatter plot shows all 20 vectors projected into 2D using PCA in the React frontend. The categories form visual clusters: CS, Math, Food, and Sports.

### Tab 2: Documents (Real Embeddings)

This uses Ollama to generate real embeddings from any text.

1. Type a document title, such as `Operating Systems Notes`
2. Paste notes, textbook content, or any long text
3. Click **EMBED & INSERT**
4. Long documents are split into overlapping 250-word chunks
5. Each chunk gets an embedding and is stored in the document index

### Tab 3: Ask AI (RAG Pipeline)

1. Insert documents in the Documents tab
2. Ask a question about those documents
3. Click **ASK AI**

Behind the scenes:

```text
1. Your question is embedded with nomic-embed-text
2. HNSW finds the most similar document chunks
3. Retrieved chunks are added to the prompt
4. llama3.2 generates the final answer
```

---

## REST API Reference

The server exposes the same API at:

```text
http://localhost:8080
```

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |
| `GET` | `/stats` | Database statistics |

### Document and RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document |
| `GET` | `/doc/list` | none | List stored document chunks |
| `DELETE` | `/doc/delete/:id` | none | Delete document chunk |
| `POST` | `/doc/search` | `{"question":"...","k":3}` | Retrieve matching chunks |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| `GET` | `/status` | none | Ollama status and model info |

### Example: Search via curl

```powershell
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

### Example: Insert a Demo Vector

```powershell
curl -X POST http://localhost:8080/insert `
  -H "Content-Type: application/json" `
  -d '{"metadata":"Heap priority queue data structure","category":"cs","embedding":[0.9,0.8,0.7,0.7,0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.1,0.1,0.1,0.1]}'
```

### Example: Insert a Document

```powershell
curl -X POST http://localhost:8080/doc/insert `
  -H "Content-Type: application/json" `
  -d '{"title":"Dynamic Programming Notes","text":"Dynamic programming solves problems by storing answers to overlapping subproblems."}'
```

### Example: Ask a Question

```powershell
curl -X POST http://localhost:8080/doc/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"What is dynamic programming?","k":3}'
```

---

## Project Structure

```text
New project/
├── backend/
│   ├── src/main/java/com/yourownai/App.java
│   └── out/                         <- compiled Java classes, generated locally
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   └── styles.css
│   ├── dist/                        <- React production build, generated locally
│   └── node_modules/                <- npm dependencies, generated locally
├── .gitignore
└── README.md
```

### Backend Architecture

```text
BruteForce          O(N*d)      Exact baseline
KDTree              O(log N)    Exact, axis-aligned partitioning
HNSW                O(log N)    Approximate, multilayer graph search

Nexus               Unified interface over all 3 algorithms for 16D demo vectors
DocumentDB          HNSW + brute force index for real Ollama embeddings
OllamaClient        Java HTTP client for /api/embeddings and /api/generate
Api                 Built-in Java HttpServer exposing REST routes and React static files
```

### Frontend Architecture

```text
React App           Main UI state and API calls
Canvas Scatter      2D PCA visualization of vectors
Documents Tab       Document insertion and chunk listing
Ask AI Tab          RAG question flow and retrieved context display
Vite                Development server and production build
```

---

## Algorithm Deep Dive

### HNSW (Hierarchical Navigable Small World)

Each node is inserted into a multilayer graph. Every node gets a random maximum layer. Layer 0 contains all nodes and has the most connections. Higher layers contain fewer nodes and act as long-distance navigation layers.

**Insert:**
Start at the top layer, greedily find a nearby entry point, then descend layer by layer. At each layer, run a beam-style search and connect to the nearest neighbors.

**Search:**
Start at the top layer, move toward closer nodes, then descend. At layer 0, expand candidate neighbors and return the nearest matches.

**Why it is fast:**
The top layers help jump near the right region quickly. Layer 0 performs the detailed local search.

### KD-Tree

KD-Tree recursively splits space by cycling through vector dimensions. During search, it can skip subtrees that cannot contain a closer point than the best current candidate.

**Strength:**
Works well for low-dimensional vectors like the 16D demo set.

**Weakness:**
In high dimensions, pruning becomes less effective.

### Brute Force

Brute force compares the query vector to every stored vector.

**Strength:**
Always exact and simple.

**Weakness:**
Runtime grows linearly with the number of vectors.

---

## Common Issues

| Problem | Fix |
|---|---|
| `Ollama: OFF` in the header | Launch the Ollama desktop app, then refresh the browser |
| `ollama` is not recognized in PowerShell | Use `C:\Users\shiva\AppData\Local\Programs\Ollama\ollama.exe` directly |
| Document insert says Ollama unavailable | Pull `nomic-embed-text` and check `http://localhost:8080/status` |
| Ask AI fails or returns model error | Pull `llama3.2` |
| AI answers are slow | Local LLM generation can take time on CPU |
| `java` or `javac` not found | Install Java 17+ and add it to PATH |
| `npm` not found | Install Node.js |
| React build fails before dependencies install | Run `npm install` inside `frontend` |
| Port 8080 already in use | Use another port: `java -cp backend/out com.yourownai.App 18080` |

### Find What Is Using Port 8080

```powershell
netstat -ano | findstr 8080
```

Then stop the process if needed:

```powershell
taskkill /PID <PID> /F
```

### Use a Smaller/Faster LLM

If `llama3.2` is too slow, pull a smaller model:

```powershell
ollama pull llama3.2:1b
```

Then edit `backend/src/main/java/com/yourownai/App.java`:

```java
final String genModel = "llama3.2:1b";
```

Recompile and restart:

```powershell
javac -d backend/out backend/src/main/java/com/yourownai/App.java
java -cp backend/out com.yourownai.App
```

---

## Build Verification

These commands were used to verify the port:

```powershell
javac -d backend/out backend/src/main/java/com/yourownai/App.java
cd frontend
npm run build
```

Smoke-tested endpoints:

```text
GET /
GET /stats
GET /search?v=...&k=3&metric=cosine&algo=hnsw
```

---

## License

MIT - use this however you want.
