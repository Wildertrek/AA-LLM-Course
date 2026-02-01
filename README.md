# Advanced Applications of Large Language Models: Assistants, Agents & RAG

This repository accompanies COSC 650 (Fall 2025) and aggregates every slide, notebook, diagram, and reading packet referenced throughout the semester. It doubles as a working lab notebook: each module folder mirrors the course schedule, and supporting data, markdown cards, and images sit alongside the instructional material so you can reproduce the hands-on exercises without hunting for assets in external drives.

## Course Snapshot
- **Course Number:** COSC 650
- **Semester:** Fall 2025
- **Credit Hours:** 3
- **Instructors:** Jens Gregor / Joseph Raetano

## How to Use This Repository
The material is intentionally modular. Each week in the syllabus maps to its own directory that contains notebooks, sample APIs, or agent builds you can run locally. Supporting references (papers, diagrams, prompt cards) live in their own folders so you can pull them into lecture decks, LMS modules, or tutorials.

### Quick Start
1. Clone or download this repository.
2. Create a virtual environment (Python 3.10+ recommended): `python3 -m venv .venv && source .venv/bin/activate`.
3. Install the dependencies required by the notebook or lab you are running (requirements are specified directly in the relevant module folders or notebooks).
4. Duplicate `env-example.txt` to `.env` (or export the values in your shell) before running any RAG or agent labs.
5. Launch JupyterLab or VS Code with the virtual environment selected and open the notebook or script for the week you are teaching.
6. To run the AI Assistant (Module 4), navigate to the assistant directory and run `streamlit run app.py`.

### Environment Variables
`env-example.txt` captures every secret used across the Azure Search, OpenAI, LangChain, and Anthropic exercises. Copy it to `.env` and replace placeholder values with your credentials. The labs automatically read from `.env`, so keeping one canonical file at the repo root prevents drift between modules.

### Recommended Tooling
- JupyterLab/Notebook for the historical model walkthroughs and RAG explorations.
- VS Code or another IDE for the `7-Building-AI-Assistants` scripts that use LangChain and Streamlit.
- Diagramming/notetaking app for incorporating items from `Diagrams/` or `Markdown/`.

## Repository Map
> **⚠️ Important Note on Paths:** Some directory names contain trailing spaces (specifically `Module 1` and `Module 4`). When navigating via CLI, ensure you quote the paths (e.g., `cd "Module 1 Foundations & LLM Interaction "`).

| Path | Description | Highlights |
| --- | --- | --- |
| `Module 1 Foundations & LLM Interaction /` | Weeks 1-3 notebooks covering setup, the history of language models, and LangChain basics. | Historical LM notebooks, LangChain starter graph/chain/agent demos. |
| `Module 2 Advanced Prompt Engineering/` | Prompting labs and examples used in Weeks 4-6. | Prompt chains, advanced prompting notebooks. |
| `Module 3 Retrieval-Augmented Generation (RAG)/` | Complete RAG curriculum for Weeks 7-9. | Chunking/indexing labs, LangChain RAG implementations, evaluation notebooks. |
| `Module 4 AI Assistants & Agents /` | Agentic AI content for Weeks 10-12 plus the Streamlit assistant build. | `7-Building-AI-Assistants/` Python scripts and notebooks demonstrating Plan-and-Execute flows. |
| `Module 5 Project Integration & Wrap-up/` | Capstone guidance and responsible AI materials for Weeks 13-15. | Project integration notes, final presentation scaffolding. |
| `Notebooks/` | Stand-alone Jupyter notebooks grouped by theme for ad hoc workshops. | `1-Setup...`, `2-API...`, `3-Prompt...`, `4-Retrieval...`, `7-Building...`, `a14-Responsible...`. |
| `Markdown/` | Reusable reference cards, prompt templates, and notes. | `Cards/` directory with model, dataset, and agent cards; prompt adjustment tables. |
| `Data/` | Sample corpora referenced in the notebooks. | `pg2600.txt` for retrieval demos. |
| `LLM-Images/` | Figures used throughout slides and cards. | Model architecture diagrams, prompt heuristics, OWASP risk graphics. |
| `Diagrams/` | Data for generated diagrams plus notebooks that build them. | `Diagrams.ipynb`, AI threat crosswalk CSVs, heatmaps. |
| `Papers/` | Curated PDF library supporting lectures and literature reviews. | 400+ PDFs spanning agents, RAG, reasoning, and responsible AI. |
| `tree_output.txt` | Snapshot of the repository tree for quick diffing or syllabus references. | Helpful when generating LMS upload manifests. |
| `GEMINI.md` / `CLAUDE.md` | Context files for AI coding assistants. | Setup instructions and architectural overviews for agents. |

## Agent Instructions & Context
This repository includes specialized context files to assist AI coding agents (and humans using them):
- **`GEMINI.md`**: Context and instructions for Google Gemini (CLI).
- **`CLAUDE.md`**: Guidance for Claude Code, including environment setup, architectural details, and warnings about directory quirks.
- **`.claude/`**: Configuration for Claude.

## Working Through the Modules
- **Module 1 – Foundations & LLM Interaction:** Start with `Week 1 History & Fundamentals of LLMs/` for environment setup and historical notebooks (BoW through Transformer). `Week 2` and `Week 3` folders layer in API usage and evaluation labs, and the nested `LangChain/` folder shows the same workflow through LangChain abstraction levels.
- **Module 2 – Advanced Prompt Engineering:** Each week folder corresponds to the prompting topics outlined below. Many activities have complementary references inside `Markdown/Prompt-Engineering-Adjustments-Table.md`.
- **Module 3 – Retrieval-Augmented Generation:** Contains end-to-end RAG walkthroughs, chunking experiments, vector database labs, and evaluation notebooks (see `Week 9 Enhancing the RAG Pipeline - Generation & Evaluation/Week 4 - Retrieval-Augmented Generation.ipynb` for the integrated pipeline).
- **Module 4 – AI Assistants & Agents:** Includes conceptual material plus runnable assistants. The `Week 12 AI Assistant Implementation & Basic UI/7-Building-AI-Assistants` directory houses iterative Python builds (e.g., `AA-37.py` through `AA-40.py` and `app.py`) demonstrating how to add tools, memory, and Streamlit UIs.
- **Module 5 – Project Integration & Wrap-up:** Use this folder during the final third of the semester for integration checkpoints, responsible AI deep dives, and presentation prep.

## Key Assets & References
- **Markdown Cards:** Located in `Markdown/Cards/` for quick inclusion in slide decks or LMS modules (model cards, dataset cards, agent templates, etc.).
- **Images & Figures:** Pull from `LLM-Images/` or `Diagrams/heatmap_by_framework.png` when you need ready-to-use visuals.
- **Notebooks vs. Modules:** The `Notebooks/` directory mirrors the module content but is reorganized for workshops and hackathons; it is safe to copy subsets without the full course tree.
- **Research Library:** `Papers/` stores the PDFs cited in lectures, literature reviews, and optional readings; file names match paper titles for easy search.

## Curriculum Outline

### Course Description
This course provides an intensive, hands-on immersion into the practical applications of Generative AI and Large Language Models (LLMs), with a primary focus on Retrieval-Augmented Generation (RAG) and building LLM-powered Agents/Assistants. Students gain experience through projects focused on prompt engineering, RAG pipeline construction and evaluation, and creating autonomous AI assistants capable of executing tasks. The curriculum emphasizes interaction with LLM APIs, integration into applications using Streamlit, leveraging LangChain, and foundational ethical AI practices.

### Course Objectives
1. Develop a deep, practical understanding of LLMs and their core application patterns in RAG and Agents.
2. Master and apply techniques such as advanced prompt engineering, retrieval-augmented generation pipeline design, and agent construction using frameworks like LangChain.
3. Design and implement LLM-based RAG systems and AI assistants for real-world tasks.
4. Build autonomous LLM-powered AI assistants capable of executing multi-step workflows and integrate them with basic user interfaces (Streamlit).
5. Apply fundamental ethical considerations and responsible AI practices in LLM application development.

---

### Weekly Breakdown

This schedule expands on the modules with existing materials, dedicating more time to each core concept.

**Module 1: Foundations & LLM Interaction (Weeks 1-3)**

* **Week 1: History & Fundamentals of LLMs**
    * **Topics:** Deep dive into the history of Language Models (BoW, TF-IDF, RNNs, LSTMs, Attention, Transformers); Key terminology (Tokens, Embeddings, Prompts); Survey of major LLM families (GPT, Llama, Claude, Gemini/PaLM) and architectures.
    * **Hands-on:** Environment setup; Running historical model notebooks; In-depth tokenization exercises; Basic embedding generation and visualization.
* **Week 2: LLM APIs & Basic Interaction**
    * **Topics:** Interacting with core LLM APIs (OpenAI, Azure OpenAI, Anthropic if applicable); Understanding key API parameters (temperature, top_p, max_tokens, etc.) and their effects; Structured API calls and response handling.
    * **Hands-on:** Focused labs on calling different LLM APIs for generation tasks; Comparing outputs based on parameter changes; Error handling and best practices for API usage.
* **Week 3: Evaluating LLM Outputs**
    * **Topics:** Need for evaluation; Intrinsic vs. Extrinsic evaluation; Common metrics (BLEU, ROUGE, Perplexity limitations); Introduction to model-based evaluation (using LLMs to evaluate LLMs); Designing evaluation test cases.
    * **Hands-on:** Implementing evaluation scripts; Setting up basic model-based evaluation pipelines; Analyzing evaluation outputs; Discussing limitations of current metrics.

**Module 2: Advanced Prompt Engineering (Weeks 4-6)**

* **Week 4: Core Prompting Techniques**
    * **Topics:** Zero-shot, Few-shot prompting; Role prompting; Crafting clear instructions and constraints; Using delimiters and structured formats (e.g., Markdown, JSON).
    * **Hands-on:** Designing and testing prompts for various tasks (summarization, Q&A, classification); Comparing zero-shot vs. few-shot performance.
* **Week 5: Advanced Prompting Patterns & Iteration**
    * **Topics:** Chain-of-Thought (CoT), Self-Consistency, Tree-of-Thoughts (conceptual); Step-by-step prompting; Techniques for reducing hallucinations; Iterative prompt refinement process.
    * **Hands-on:** Implementing CoT prompts; Iteratively refining prompts based on output analysis; Developing rubrics for prompt quality.
* **Week 6: Prompt Chaining & Structured Outputs**
    * **Topics:** Breaking down complex tasks into prompt chains; Managing context between prompts; Ensuring consistent structured outputs (JSON, XML); Prompt templating engines (e.g., Jinja, LangChain templates).
    * **Hands-on:** Building multi-step prompt chains using basic Python or LangChain expression language; Enforcing structured output formats; Using prompt templates for reusability.

**Module 3: Retrieval-Augmented Generation (RAG) (Weeks 7-9)**

* **Week 7: RAG Fundamentals & Vector Embeddings**
    * **Topics:** Core RAG architecture; Why RAG? (vs. Fine-tuning); Deep dive into vector embeddings - models (Sentence Transformers, OpenAI Ada), properties, use cases; Similarity and distance metrics (Cosine, Euclidean, Dot Product).
    * **Hands-on:** Generating and comparing embeddings from different models; Calculating vector similarity; Setting up a basic vector store (e.g., FAISS, ChromaDB).
* **Week 8: Building the RAG Pipeline - Indexing & Retrieval**
    * **Topics:** Document loading and chunking strategies (fixed size, recursive, semantic); Indexing pipelines; Vector database choices and tradeoffs; Core retrieval algorithms (similarity search).
    * **Hands-on:** Implementing different chunking methods; Loading and indexing documents into a vector store; Performing vector searches; Evaluating retrieval relevance.
* **Week 9: Enhancing the RAG Pipeline - Generation & Evaluation**
    * **Topics:** Combining retrieved context with the original prompt for the LLM; Context stuffing strategies; Evaluating RAG systems (Retrieval metrics: Precision, Recall; Generation metrics: Faithfulness, Relevance); Introduction to RAG frameworks (LangChain RAG chains).
    * **Hands-on:** Building an end-to-end RAG chain using LangChain; Evaluating the RAG system on a Q&A task; Experimenting with different context combination methods.

**Module 4: AI Assistants & Agents (Weeks 10-12)**

* **Week 10: Introduction to Agentic AI**
    * **Topics:** Concept of LLM-powered agents; Core components (LLM Brain, Memory, Tools - conceptual); Agent loops (Observe-Think-Act); Agentic patterns (ReAct, basic Plan-and-Execute).
    * **Hands-on:** Implementing a very simple ReAct-style agent using LangChain or raw API calls; Designing the thought process for a simple task.
* **Week 11: Building Task-Specific Agents & Memory**
    * **Topics:** Designing agents for specific tasks (e.g., information gathering, simple automation); Short-term and long-term memory concepts for agents (conceptual overview); Introduction to agent frameworks (LangChain Agents, brief look at Autogen concepts).
    * **Hands-on:** Developing specific agents (e.g., Weather Agent, API Spec Agent); Implementing basic conversation buffer memory.
* **Week 12: AI Assistant Implementation & Basic UI**
    * **Topics:** Architecting a more cohesive AI Assistant integrating multiple capabilities (e.g., RAG + Agent tools); Handling multi-turn conversations; Basic UI integration strategies using Streamlit.
    * **Hands-on:** Integrating RAG capabilities into an agent/assistant; Developing a simple Streamlit interface for the assistant; Managing conversation state in the UI.

**Module 5: Project Integration & Wrap-up (Weeks 13-15)**

* **Week 13: Project Work Session & Integration**
    * **Topics:** Dedicated time for students to integrate RAG and Agent components into their final projects. Troubleshooting and refinement.
    * **Hands-on:** Intensive project development; Instructor and peer support.
* **Week 14: Responsible AI & Final Presentations I**
    * **Topics:** Review of ethical considerations (bias, fairness, transparency) specifically in the context of RAG and Agents; Strategies for mitigating risks; Responsible deployment considerations.
    * **Hands-on:** Analyzing course projects for potential ethical issues; Preparing final presentations.
    * **Project Presentations:** Begin final project demonstrations.
* **Week 15: Final Presentations II & Course Conclusion**
    * **Topics:** Continuation and conclusion of final project presentations; Peer review and feedback.
    * **Project Presentations:** Final Q&A and evaluation.
    * **Course Review:** Summary of key takeaways; Discussion of limitations and future directions within RAG and Agents.

---

### Assessment Methods
* Module Projects (Prompting, RAG, AI Assistants & Agents): 60%
* Participation and Weekly Assignments: 10%
* Peer Reviews and Interaction: 10%
* Final Integrated Project Presentation: 20%

### Textbooks and Resources
* Primary Textbook: *Building LLMs for Production, Enhancing LLM Abilities and Reliability with Prompting, Fine-Tuning, and RAG* by Louis-François Bouchard.
* Possible other Textbook: *Large Language Models: A Deep Dive: Bridging Theory and Practice*
* Supplementary book materials: [https://towardsai.net/book](https://towardsai.net/book)
* Supplementary materials including research articles, online courses, and tutorials.

### Software and Tools
* Programming languages: Python
* AI frameworks/APIs: OpenAI (GPT-4o), Azure OpenAI, Anthropic (Claude 3.5 Sonnet), Google (Gemini 2.0 Flash), xAI (Grok-2)
* LLM frameworks: LangChain, LlamaIndex
* Application frameworks: Streamlit
* Vector Stores: FAISS, ChromaDB (Examples)
* Cloud services and AI platforms: Azure (Optional)
