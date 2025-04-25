---

**Course Title:** Advanced Applications of Large Language Models: Assistants, Agents & RAG
**Course Number:** COSC 650
**Semester:** Fall 2025
**Credit Hours:** 3
**Instructors:** *Jens Gregor / Joseph Raetano

### **Course Description:**
This course provides an intensive, hands-on immersion into the practical applications of Generative AI and Large Language Models (LLMs), with a primary focus on Retrieval-Augmented Generation (RAG) and building LLM-powered Agents/Assistants. Students will gain deep practical experience through individual projects focused on prompt engineering, RAG pipeline construction and evaluation, and creating autonomous AI assistants capable of executing tasks. The curriculum emphasizes hands-on interaction with LLM APIs, integration into basic applications using Streamlit, leveraging the LangChain framework, and foundational ethical AI practices.

### **Course Objectives:**
1.  Develop a deep, practical understanding of LLMs and their core application patterns in RAG and Agents.
2.  Master and apply techniques such as advanced prompt engineering, retrieval-augmented generation pipeline design, and agent construction using frameworks like LangChain.
3.  Design and implement LLM-based RAG systems and AI assistants for real-world tasks.
4.  Build autonomous LLM-powered AI assistants capable of executing multi-step workflows and integrate them with basic user interfaces (Streamlit).
5.  Apply fundamental ethical considerations and responsible AI practices in LLM application development.

---

### **Weekly Breakdown:**

This schedule expands on the modules with existing materials, dedicating more time to each core concept.

**Module 1: Foundations & LLM Interaction (Weeks 1-3)**

* **Week 1: History & Fundamentals of LLMs**
    * **Topics:** Deep dive into the history of Language Models (BoW, TF-IDF, RNNs, LSTMs, Attention, Transformers); Key terminology (Tokens, Embeddings, Prompts); Survey of major LLM families (GPT, Llama, Claude, Gemini/PaLM) and architectures.
    * **Hands-on:** Environment setup; Running historical model notebooks; In-depth tokenization exercises; Basic embedding generation and visualization.
* **Week 2: LLM APIs & Basic Interaction**
    * **Topics:** Interacting with core LLM APIs (OpenAI, Azure OpenAI, Anthropic if applicable); Understanding key API parameters (temperature, top\_p, max\_tokens, etc.) and their effects; Structured API calls and response handling.
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

### **Assessment Methods:**
* Module Projects (Prompting, RAG, AI Assistants & Agents): 60%
* Participation and Weekly Assignments: 10%
* Peer Reviews and Interaction: 10%
* Final Integrated Project Presentation: 20%

### **Textbooks and Resources:**
* Primary Textbook: *Building LLMs for Production, Enhancing LLM Abilities and Reliability with Prompting, Fine-Tuning, and RAG* by Louis-François Bouchard.
* Possible other Textbook: *Large Language Models: A Deep Dive: Bridging Theory and Practice*
* Supplementary book materials: [https://towardsai.net/book](https://towardsai.net/book)
* Supplementary materials including research articles, online courses, and tutorials.

### **Software and Tools:**
* Programming languages: Python
* AI frameworks/APIs: OpenAI, Azure OpenAI, Anthropic (Optional)
* LLM frameworks: LangChain, LlamaIndex
* Application frameworks: Streamlit
* Vector Stores: FAISS, ChromaDB (Examples)
* Cloud services and AI platforms: Azure (Optional)

---