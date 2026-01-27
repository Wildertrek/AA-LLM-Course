# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an educational repository for COSC 650 "Advanced Applications of Large Language Models" (Fall 2025). It contains a complete curriculum covering LLM foundations, prompt engineering, RAG systems, and AI agents/assistants. The repository serves as both a course archive and working lab environment.

## Environment Setup

### Initial Setup
```bash
# Create virtual environment (Python 3.10+ required)
python3 -m venv .venv
source .venv/bin/activate

# Configure environment variables
cp env-example.txt .env
# Edit .env with your API keys
```

### Running Jupyter Notebooks
```bash
# Launch from repository root
jupyter lab

# Or from specific module directory
cd "Module 1 Foundations & LLM Interaction "
jupyter lab
```

### Running Streamlit AI Assistants
```bash
# Navigate to the assistants directory
cd "Module 4 AI Assistants & Agents /Week 12 AI Assistant Implementation & Basic UI/7-Building-AI-Assistants"

# Install dependencies
pip install -r requirements.txt

# Run a specific assistant
streamlit run app.py
# Or versioned assistants: AA-37.py, AA-39.py, AA-40.py
```

### Testing Commands
Most notebooks are self-contained. To verify environment setup:
```bash
# Check Python version
python --version  # Should be 3.10+

# Verify key packages
python -c "import langchain; import openai; import anthropic; print('Dependencies OK')"
```

## Repository Architecture

### Module Structure
The repository uses **two parallel organizations**:
1. **Module folders** (`Module 1` through `Module 5`) - Organized by course week/schedule
2. **Notebooks folder** - Same content reorganized thematically for workshops

**Important**: Module directories have trailing spaces in their names (e.g., `"Module 1 Foundations & LLM Interaction "` with space at end). Always quote paths in shell commands.

### Key Directories

- **Module 1-5**: Weekly curriculum materials with nested structure:
  - `Week X [Topic]/` - Contains notebooks and Python scripts for that week
  - `LangChain/` subdirectories - LangChain-specific implementations
  - `LM-Notebooks/` - Historical language model notebooks (BoW → Transformers)

- **Notebooks/**: Theme-based organization mirroring module content:
  - `1-Setup-Foundations/` - Environment setup and LLM history
  - `2-API-Usage-LLM-Interaction-and-Evaluation/` - API usage patterns
  - `3-Prompt-Engineering-Best-Practices/` - Prompting techniques
  - `4-Retrieval-Augmented Generation/` - RAG implementations
  - `7-Building-AI-Assistants/` - Agent/assistant builds with Streamlit

- **Supporting Assets**:
  - `Papers/` - 400+ research PDFs (agents, RAG, reasoning, ethics)
  - `Markdown/Cards/` - Reusable reference cards (model cards, dataset cards, agent templates)
  - `Markdown/Prompt-Engineering-Adjustments-Table.md` - Prompt refinement guide
  - `Data/` - Sample corpora (e.g., `pg2600.txt` for retrieval demos)
  - `LLM-Images/` & `Diagrams/` - Visual assets and diagram generation code

### Technology Stack

**LLM Providers**: OpenAI (GPT-4o, GPT-4o-mini), Anthropic (Claude 3.5 Sonnet), Google (Gemini 2.0 Flash), xAI (Grok-2), Azure OpenAI

**Frameworks**:
- LangChain - Primary agent/chain framework (used extensively in Modules 1, 3, 4)
- LlamaIndex - Alternative RAG framework (examples in Module 3)
- Streamlit - UI layer for assistants (Module 4)

**Vector Stores**: FAISS, ChromaDB (Module 3 RAG implementations)

**Key Libraries**: `langchain-openai`, `langchain-anthropic`, `langchain-community`, `openai`, `anthropic`, `google-generativeai`, `python-dotenv`, `streamlit`

## Environment Variables

The `.env` file at repository root provides all API keys and credentials. Critical variables:

```bash
# Core LLM APIs
OPENAI_API_KEY=           # or OPENAI_API_COURSE_KEY
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=           # For Gemini
GROK_API_KEY=             # xAI Grok (also XAI_API_KEY)

# LangChain tracing
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=true

# Search/tools
TAVILY_API_KEY=           # Web search tool

# Azure (used in some RAG examples)
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_KEY=
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_API_VERSION=2024-06-01
AZURE_SEARCH_SERVICE_ENDPOINT=
AZURE_SEARCH_ADMIN_KEY=
```

**Note**: The repository contains a populated `.env` file with active credentials. When working with sensitive operations, be mindful of credential exposure.

## Code Patterns & Conventions

### LLM Initialization Pattern
The codebase consistently uses this pattern (see `app.py`):
```python
from dotenv import load_dotenv
import os

load_dotenv()

def get_env_var(var: str):
    value = os.getenv(var)
    if value is None:
        raise ValueError(f"{var} not found in environment variables.")
    return value

# Initialize models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

gpt4o_chat = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=get_env_var("OPENAI_API_KEY"))
claude_chat = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
```

### LangChain Abstraction Levels
Module 1 demonstrates LangChain progression (in `Week 1.../LangChain/`):
1. `0_simple-graph.ipynb` - Basic LangGraph
2. `1_chain.ipynb` - Simple chains
3. `2_router.ipynb` - Conditional routing
4. `3_agent.ipynb` - ReAct-style agents
5. `4_agent-memory.ipynb` - Memory integration
6. `5_deployment.ipynb` - Production patterns

### RAG Pipeline Architecture
Module 3 implements standard RAG flow:
1. **Document Loading & Chunking** (Week 8) - Fixed-size, recursive, semantic chunking strategies
2. **Embedding & Indexing** (Week 7-8) - Vector generation and storage (FAISS/ChromaDB)
3. **Retrieval** (Week 8) - Similarity search with configurable metrics
4. **Generation** (Week 9) - Context injection and LLM completion
5. **Evaluation** (Week 9) - Retrieval metrics (precision/recall), generation metrics (faithfulness/relevance)

Key notebook: `Module 3.../Week 9.../Week 4 - Retrieval-Augmented Generation.ipynb` contains integrated pipeline.

### AI Assistant Evolution
Module 4 contains iterative assistant builds (`AA-15.py` → `AA-40.py`):
- Early versions: Basic LLM interaction
- Mid versions: Tool integration (web search, APIs)
- Later versions: Memory, multi-turn conversations, Streamlit UI
- `app.py`: Multi-model selector with GPT-4o, Claude, Gemini, Grok

Pattern: ReAct and Plan-and-Execute agent architectures using LangChain agents.

## Working with This Repository

### Path Handling
- **Module directories have trailing spaces** - always quote: `cd "Module 1 Foundations & LLM Interaction "`
- Notebooks reference relative paths: `../Data/`, `../LLM-Images/`
- When running notebooks from different locations, verify data paths resolve correctly

### Notebook Organization
- Notebooks are **duplicated** between `Module X/Week Y/` and `Notebooks/` directories
- The `Notebooks/` organization is theme-based and useful for non-linear exploration
- Prefer module folders when following course sequence; use `Notebooks/` for ad-hoc work

### Dependencies
- **No central requirements.txt** - dependencies are specified per module/week
- Check for `requirements.txt` in specific week folders or notebook comments
- Core dependencies in `Notebooks/7-Building-AI-Assistants/requirements.txt`:
  ```
  anthropic==0.49.0
  google-generativeai==0.8.4
  langchain-anthropic==0.3.9
  langchain-openai==0.2.12
  openai==1.57.4
  streamlit==1.43.1
  python-dotenv==1.0.1
  ```

### Common Operations
- **Add new agent capabilities**: Reference `Module 4.../7-Building-AI-Assistants/` versioned scripts
- **Modify RAG pipeline**: Start with `Module 3.../Week 9.../Week 4 - Retrieval-Augmented Generation.ipynb`
- **Update prompt templates**: Check `Markdown/Prompt-Engineering-Adjustments-Table.md` for patterns
- **Add research references**: Place PDFs in `Papers/` directory

### Evaluation Focus
The course emphasizes systematic evaluation:
- Module 3 Week 9 contains RAG evaluation notebooks
- Use model-based evaluation (LLM-as-judge) patterns from Module 1 Week 3
- Metrics: BLEU/ROUGE (limitations acknowledged), precision/recall, faithfulness, relevance

## File Navigation Tips

When searching for specific content:
- **LLM history/fundamentals**: `Module 1.../Week 1.../1-History-of-Language-Models.ipynb`
- **API interaction patterns**: `Module 1.../Week 2.../` (check exact week name)
- **Prompt engineering techniques**: `Module 2.../` or `Notebooks/3-Prompt-Engineering-Best-Practices/`
- **RAG implementations**: `Module 3.../` or `Notebooks/4-Retrieval-Augmented Generation/`
- **Agent/assistant code**: `Module 4.../Week 12.../7-Building-AI-Assistants/`
- **Reference materials**: `Markdown/Cards/`, `Papers/`
- **Visual assets**: `LLM-Images/`, `Diagrams/`
