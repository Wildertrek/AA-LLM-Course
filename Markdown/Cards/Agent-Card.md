Below is a **generic AI Agent Card** template, drawing on key elements from the AI Agent Index. It’s designed to capture both **technical** and **safety** details of an “agentic” AI system, as well as provide essential context for users, developers, auditors, and policymakers. You can adapt, add, or remove fields as needed for your organization or project.

---

# **AI Agent Card**

## **1. Basic Information**
- **Agent Name:**  
  A concise, descriptive name or version (e.g., “WebNavigator v2.1,” “CodeCritic 2025”).  
- **Short Description / Primary Function:**  
  In a single sentence, what is the agent meant to do?  
- **Intended Uses (Stated by Developer):**  
  - What tasks or domains is the agent built for? (e.g., “Software debugging,” “Customer email triage,” “Scientific experiment planning”)  
- **Date(s) of Release or Deployment:**  
  - Indicate each relevant milestone, if multiple versions exist.

---

## **2. Developer / Organization**
- **Legal Name & Contact Info:**  
  - The main organization, lab, or individual responsible for development.  
  - Include website or primary contact channel (e.g., email, GitHub Issues, helpdesk portal).  
- **Entity Type:**  
  - E.g., “Private company,” “University research group,” “Open-source community project.”  
- **Country / Region:**  
  - Where the developer is primarily based or legally registered.  
- **Safety Policies or Responsible AI Statements:**  
  - Link to any publicly available policy, code of conduct, or safety framework relevant to this agent.

---

## **3. System Components & Architecture**
- **Base / Backend Model(s):**  
  - Which underlying model(s) power this agent? (e.g., “GPT-4,” “Llama 3.2 90B,” “Custom BERT variant.”)  
  - Note if the model is privately hosted, open source, or licensed from a third party.  
- **Reasoning, Planning, & Memory Implementation:**  
  - How does the agent “think” or structure its process? (e.g., chain-of-thought prompting, reflection-based loops, custom planning code)  
  - Describe any memory or knowledge store (e.g., “Vector DB for extended context,” “Local ephemeral memory,” “Persistent logs”).  
- **Observation Space:**  
  - What can the agent “see” or read while it operates? (e.g., text from user, web pages, OS-level logs, sensor data)  
- **Action Space (Tools & API Calls):**  
  - Which actions can the agent directly perform in the real or digital world? (e.g., making web requests, writing to a local file system, controlling a robot)  
- **User Interface:**  
  - How do end-users interact with the agent? (CLI, web UI, Slack bot, embedded in an IDE, etc.)  
- **Development Cost & Compute (If Known):**  
  - Any publicly stated R&D spend, training compute, or other noteworthy costs. (e.g., “Trained for 1 week on 8 A100 GPUs,” “No data publicly disclosed.”)

---

## **4. Guardrails & Oversight**
- **Accessibility of Components:**  
  - **Weights:** Are the model parameters publicly available, partially available, or proprietary?  
  - **Training / Fine-tuning Data:** Is the dataset open or closed? Are there any data statements or documentation?  
  - **Code:** Is the agent’s code open source, partially open (e.g., a GitHub repo missing certain modules), or fully proprietary?  
  - **Scaffolding:** Are custom prompts, chain-of-thought logs, memory modules, or planning logic available or documented?  
  - **Documentation:** Is there a dedicated user manual or developer guide?  
- **Controls & Guardrails:**  
  - Prompt filtering, integrated policy constraints, forced human-in-the-loop for high-stakes tasks, or usage rate limiting.  
- **Customer / Usage Restrictions:**  
  - Are there user verification steps, paywalls, or other restrictions to prevent misuse?  
  - E.g., “Enterprise customers only,” “No public API,” “Requires ID verification.”  
- **Monitoring & Shutdown Procedures:**  
  - Mechanisms to pause, stop, or override the agent if it exhibits harmful or unintended behavior (e.g., panic button, resource usage monitors, kill switch).

---

## **5. Evaluations**
- **Notable Benchmarks or Competitions:**  
  - Any recognized leaderboards or official tests (e.g., “SWE-Bench,” “WebArena,” “AgentHarm,” “GAIA benchmark”).  
- **Bespoke Testing & Demos:**  
  - In-house tests or demonstrations (e.g., “Tested on 500 GitHub issues,” “Simulated phishing attacks”).  
- **Internal Safety or Risk Assessments (If Disclosed):**  
  - Type and scope: e.g., “We performed red-teaming for cybersecurity vulnerabilities,” “We tested generative outputs for bias.”  
  - Key findings or reported metrics (if any).  
- **Publicly Reported External Red-Teaming or Auditing:**  
  - **Personnel:** Who did it (e.g., security firm, academic group)?  
  - **Scope, Methods, & Access:** How thorough or “white-box” was the evaluation?  
  - **Findings:** Major vulnerabilities or improvements discovered.

---

## **6. Ecosystem & Interoperability**
- **Integrations & Compatibility:**  
  - Official partnerships (e.g., with CRM systems, code repositories, robotics platforms).  
  - Known community-driven plugins or expansions.  
- **Usage Statistics & Patterns:**  
  - If known or disclosed, mention monthly active users, typical domains or industries using the agent, or notable adopters.  
- **Dependencies:**  
  - Critical external services or APIs the agent relies on (e.g., “Requires AWS S3 for memory storage,” “Needs a stable web scraping tool”).

---

## **7. Known Limitations & Future Plans**
- **Coverage & Capability Gaps:**  
  - Does the agent fail on particular tasks, languages, or domains?  
- **Planned Updates & Versioning Strategy:**  
  - Frequency and nature of future releases (e.g., “Monthly updates,” “Beta release Q2 2025,” “No scheduled improvements”).  
- **Open Issues or Technical Debt:**  
  - List any major open problems or design shortcomings.  
- **Long-Term Maintenance or Deprecation Plan:**  
  - Who owns ongoing support, or if EOL (end-of-life) is anticipated, specify when.

---

## **8. Additional Notes**
- **Context or Historical Notes:**  
  - Unique origin story, special licensing constraints, or interesting user community details.  
- **References & Further Reading:**  
  - Links to academic papers, blog posts, or key press releases describing the agent.  
- **Citation Info (If Applicable):**  
  - If you require references or citations (e.g., an APA or BibTeX format) for external parties who build upon or study the agent.

---

### **How to Use This Agent Card**
1. **Link to Other Cards:**  
   - Embed or hyperlink your **Model Card** (for the foundation model), **Dataset Card** (if fine-tuning data is used), **AI Use-case Card** (explaining the broader application).  
2. **Keep It Updated:**  
   - Revision logs and version numbers ensure that users always see current information.  
3. **Encourage Feedback & Corrections:**  
   - Provide a quick feedback channel (e.g., GitHub issues, contact forms) for reporting inaccuracies or newly discovered risks.

---

This **AI Agent Card** is designed to be the “one-stop shop” for understanding your AI agent’s capabilities, architecture, intended uses, and risk management. By maintaining a well-structured and regularly updated Agent Card, you provide critical transparency, helping stakeholders—from developers to end-users—understand how best to work with, evaluate, and govern your agentic AI system.