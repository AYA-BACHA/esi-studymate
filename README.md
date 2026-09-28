# ESI StudyMate

A simple AI study assistant for university students, built after completing **ThirdUni Week 3: AI Agents and LangChain**.

---

## What is ESI StudyMate?

**ESI StudyMate** is an AI study assistant created for students at the Higher National School of Computer Science (ESI). 

When studying complex courses like Operating Systems, Computer Architecture, or Algorithms, students often have specific questions about their lecture slides and notes. 

With StudyMate:
1. You upload your course material (such as a PDF or text file of lecture notes).
2. The system indexes the content so the AI can search it.
3. You can chat with StudyMate about the course, and it answers using facts directly from the uploaded material.
4. If a concept requires a mathematical calculation (like computing memory block allocation or page table sizes), StudyMate uses its calculator tool.

---

## How It Works

StudyMate uses a standard **RAG (Retrieval-Augmented Generation)** pipeline:

```text
Course PDF / Text
       ↓
Split into Chunks (RecursiveCharacterTextSplitter)
       ↓
Embeddings (all-MiniLM-L6-v2)
       ↓
Vector Store (ChromaDB)
       ↓
Student asks a question
       ↓
Retrieve relevant chunks (Course Retriever)
       ↓
Agent receives the retrieved context
       ↓
Answer with source reference
```

The AI does not make up facts about your course. It only knows what is present in the uploaded material. If the topic is not covered in your documents, it clearly tells you.

---

## The Agent

StudyMate uses LangChain's `create_agent` pattern:

```python
agent = create_agent(
    model=model,
    tools=[search_course_material, calculator],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory,
)
```

The agent has access to two tools:
- **`search_course_material`**: Queries the Chroma vector store for paragraphs matching the student's question.
- **`calculator`**: Evaluates mathematical expressions accurately (e.g., `ceil(2500 / 512)` or `2**32 / 4096`).

When you ask a question, the agent decides whether to call a tool or answer directly.

---

## Conversation Memory

StudyMate includes **short-term conversation memory** using LangGraph's `MemorySaver` checkpointer.

This allows you to ask natural follow-up questions:

> **Student:** "What is an interrupt?"  
> **StudyMate:** Explains what interrupts are and mentions hardware vs software interrupts.  
> **Student:** "Can you explain the second type?"  
> **StudyMate:** Understands that "second type" refers to software interrupts/traps from the previous message.

---

## Technologies Used

Only the essential technologies taught in Week 3 are used:

- **Python 3**: Core language
- **LangChain & LangGraph**: Agent creation (`create_agent`), tool decorator (`@tool`), and memory (`MemorySaver`)
- **ChromaDB**: Local vector database for storing course document embeddings
- **Streamlit**: Simple web interface for uploading files and chatting
- **PyPDF**: Extracting text from course PDF documents
- **Pytest**: Running automated tests

---

## What I Learned (Week 3 Concepts)

This project helped me understand the fundamental building blocks of AI agents:

1. **Difference between an LLM and an Agent**: An LLM only predicts words based on its static training. An agent can reason, choose actions, and call external tools to get fresh information or perform calculations.
2. **Tool Use (`@tool`)**: How to define functions that the model can invoke when it needs external abilities (retrieval and math).
3. **RAG Pipeline**: How documents are broken into chunks, turned into vector embeddings, and retrieved using semantic similarity search.
4. **Agent Memory & State**: How to maintain conversation history across multiple turns using checkpointers so the agent remembers previous questions.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure your API key
Copy `.env.example` to `.env` and add your Google API key:
```bash
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

### 3. Run the application
```bash
streamlit run app.py
```

### 4. Run the tests
```bash
pytest -v
```

---

## Project Structure

```text
esi-studymate/
│
├── app.py              # Simple Streamlit web interface
├── agent.py            # StudyMate agent definition (create_agent + memory)
├── rag.py              # PDF loading, chunking, and Chroma vector store
├── tools.py            # @tool definitions (course search & calculator)
├── requirements.txt    # Project dependencies
├── README.md           # Documentation
├── pytest.ini          # Test configuration
├── .env.example        # Environment variable template
├── .gitignore          # Files to ignore in git
│
├── data/               # Course documents and Chroma storage
└── tests/              # Simple unit tests for tools, rag, and agent
```
