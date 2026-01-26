# GEMINI.md - Context & Instructions

## Project Overview
**AA-LLM-Course** is a comprehensive educational repository for the "Advanced Applications of Large Language Models" course (COSC 650, Fall 2025). It serves as both a curriculum archive and a working lab notebook for building LLM-powered applications.

**Core Themes:**
*   **LLM Foundations:** History, architecture, and API interaction (OpenAI, Anthropic).
*   **Prompt Engineering:** Advanced techniques (Chain-of-Thought, structured outputs).
*   **RAG (Retrieval-Augmented Generation):** Vector embeddings, indexing, and evaluation.
*   **Agents & Assistants:** Building autonomous agents with LangChain and Streamlit.
*   **Responsible AI:** Ethics, evaluation, and safety.

**Key Technologies:**
*   **Language:** Python 3.10+
*   **Frameworks:** LangChain, LlamaIndex, Streamlit
*   **Tools:** JupyterLab, VS Code
*   **APIs:** OpenAI, Azure OpenAI, Anthropic, Tavily

## Repository Structure

The repository is organized into chronological modules and standalone resources:

*   **`Module 1` - `Module 5`**: The core curriculum, split into weekly folders. Each contains:
    *   Jupyter Notebooks for hands-on labs.
    *   Python scripts for building agents (esp. Module 4).
    *   Supporting data and prompts.
*   **`Notebooks/`**: A reorganized collection of the module notebooks, grouped by theme (Setup, API Usage, RAG, etc.) for ad-hoc workshops.
*   **`Papers/`**: A large collection (400+) of PDF research papers cited in the course.
*   **`Markdown/`**: Reference cards (Model Cards, Prompt Templates) and notes.
*   **`Diagrams/` & `LLM-Images/`**: Visual assets and the code to generate them.

## Setup & Usage

### 1. Environment Setup
The project relies on a Python virtual environment and environment variables for API keys.

```bash
# Create and activate virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
# Note: specific requirements are often found in individual module folders.
# Check for requirements.txt in the specific week/module you are working on.
```

### 2. Configuration (`.env`)
This project uses a central `.env` file to manage secrets for OpenAI, Azure, LangChain, etc.

1.  Copy the example file: `cp env-example.txt .env`
2.  Edit `.env` and fill in your API keys (OpenAI, Azure, LangChain, etc.).

### 3. Running Labs
*   **Notebooks:** Launch JupyterLab in the root or specific directory:
    ```bash
    jupyter lab
    ```
*   **AI Assistants (Streamlit):**
    Navigate to `Module 4 AI Assistants & Agents/Week 12 .../7-Building-AI-Assistants/` (check exact path) and run:
    ```bash
    streamlit run AA-15.py  # (or other agent script)
    ```

## Development Conventions

*   **Modularity:** The codebase is designed to be modular. Exercises are often self-contained within their weekly folders.
*   **Paths:** Notebooks may reference data in `../Data` or `../LLM-Images`. Be mindful of relative paths when running notebooks from different locations.
*   **Evaluation:** There is a strong emphasis on evaluation (Module 3 & responsible AI). Use the provided evaluation notebooks to benchmark RAG pipelines and prompts.

## Key Resources
*   **`env-example.txt`**: Template for all required environment variables.
*   **`Markdown/Cards/`**: Reusable reference cards for models and datasets.
*   **`Papers/`**: Extensive library of LLM research papers.
