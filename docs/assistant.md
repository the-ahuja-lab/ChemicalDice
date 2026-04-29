# *CDI Bot*

*Chemical Dice Integrator — Conversational Molecular Embedding Platform*

**CDI Bot** is a fully containerised, LLM-powered web application that gives researchers and chemists a natural-language interface to the **Chemical Dice Integrator (CDI)**.

> [!TIP]
> **Watch the CDI Bot in action:**
> <iframe width="100%" height="400" src="https://www.youtube.com/embed/3NaBBTviEsA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

CDI's core embedding pipeline exists as a Python library and a REST API, but lacked any accessible interface for non-programmer users. Researchers needed to write code to generate embeddings, understand API contracts, and manually manage files. The goal was to eliminate that friction entirely:

* Allow any researcher to generate molecular embeddings by simply typing in natural language or uploading a spreadsheet

* Remove dependency on a locally installed Ollama instance — the LLM runs inside the container, making deployment a single command on any machine

* Provide a microservice interface so ML pipelines can call the CDI API programmatically without the chat layer

**How We Built It**

**Architecture —** Three services managed by Supervisor inside one Docker image: Ollama (LLM server, port 11434), FastAPI backend (port 8001), and Streamlit frontend (port 8501). Ollama starts first; FastAPI waits for its health check; Streamlit starts after a brief delay.

**LLM Pipeline —** Every user message triggers two sequential Ollama calls. Step 1 generates a natural-language conversational reply using a CDI-specific system prompt with few-shot examples. Step 2 is a near-deterministic intent classifier (temperature 0.05) that outputs a structured JSON object with fields *intent* (run\_file | run\_smiles | chat) and *smiles*. No regex or keyword lists — the LLM does all parsing.

**Model Baking —** The chosen LLM (default: llama3.1:8b) is pulled from Ollama's registry during docker build and stored as an image layer. Runtime requires zero internet access and zero model downloads.

**CDI Integration —** Single-SMILES requests hit the CDI REST API (chemicaldice.ahujalab.iiitd.edu.in). Batch file requests invoke the *ChemicalDice* Python library directly. Results are checkpointed as CSV and served via a download endpoint.

**Features**

| Feature | Description |
| :---- | :---- |
| **💬  Conversational Chat UI** | Streamlit-based dark-themed interface with animated message bubbles |
| **🔬  Single SMILES Embedding** | Paste any SMILES string; get a CDI vector instantly via the CDI REST API |
| **📂  Batch File Embedding** | Upload CSV / Excel / TSV / JSON — auto-detected, converted, and processed |
| **🤖  LLM-Driven Intent Engine** | Two-step Ollama pipeline: chat reply \+ structured JSON intent \+ SMILES extraction |
| **⚙️  Microservice Tab** | Direct API access — no chat needed; ideal for programmatic integration |
| **⬇️  CSV Download** | Embeddings exported as a ready-to-use CSV with SMILES \+ CDI\_0…CDI\_n columns |
| **🐳  Single Docker Image** | One \`docker build\`, one \`docker run\` — Ollama \+ FastAPI \+ Streamlit bundled |

**How the Public Gets It**

| Docker Image | Single self-contained image (\~8 GB); run with one command on any Linux host |
| :---- | :---- |
| **Default Model** | LLM baked in at build time (default: llama3.1:8b); swappable via \--build-arg |
| **Ports Exposed** | 8501 — Streamlit UI    |    8001 — FastAPI REST API (/docs for Swagger) |
| **GPU Support** | Pass \--gpus all for NVIDIA acceleration; falls back to CPU automatically |
| **Model Switching** | docker build \--build-arg LLM\_MODEL=\<tag\> — no code changes required |

**Run using a prebuilt Docker image**

docker pull ahujalab/chemicaldice-app:v1

docker run \--gpus all \--name chemicaldice-app \-p 8001:8001 \-p 8501:8501 ahujalab/chemicaldice-app:v1

### **⚙️ Requirements (must have)**

* Docker installed  
* NVIDIA GPU  
* NVIDIA Container Toolkit installed (for `--gpus all`)  
* \~20 GB free disk space  
* At least 8 GB RAM (16 GB recommended)

