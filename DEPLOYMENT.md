# Free Deployment Guide

This project can be deployed for free on Render as a Docker web service.

## What Works on Free Hosting

The hosted app supports:

- React frontend
- Java backend
- Demo vector search
- `/stats`, `/items`, `/search`, `/benchmark`, `/hnsw-info`

RAG/document AI requires an external Ollama-compatible API. Free web hosts usually cannot run local Ollama models inside the same service reliably.

## Files Added for Deployment

```text
Dockerfile
.dockerignore
render.yaml
```

The Docker build:

1. Installs and builds the React frontend.
2. Compiles the Java backend.
3. Runs the Java app using Render's `PORT` environment variable.

## Render Deployment Steps

1. Push this project to GitHub.
2. Go to `https://render.com`.
3. Click **New**.
4. Choose **Blueprint** if using `render.yaml`, or choose **Web Service**.
5. Connect your GitHub repository.
6. Select the free plan.
7. Deploy.

If creating a Web Service manually:

```text
Runtime: Docker
Plan: Free
```

No build/start command is needed when using the Dockerfile.

## Environment Variables

For demo-only deployment, no environment variables are required.

For hosted Ollama-compatible AI, set:

```text
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=your_api_key_here
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_GEN_MODEL=llama3.2
```

If you do not set `OLLAMA_API_KEY`, the hosted app will still run, but `/status` may show Ollama unavailable and document/RAG features may fail.

## Test After Deploy

Replace `YOUR_RENDER_URL` with your Render URL:

```text
https://YOUR_RENDER_URL/
https://YOUR_RENDER_URL/stats
https://YOUR_RENDER_URL/status
https://YOUR_RENDER_URL/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw
```

## Important Limitations

- Render free services can sleep when inactive.
- In-memory inserted vectors/documents are lost when the service restarts.
- Full RAG depends on an external model API.
- This is a demo deployment, not production infrastructure.
