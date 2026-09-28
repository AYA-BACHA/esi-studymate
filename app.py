"""ESI StudyMate - Streamlit Application.

A simple AI study assistant interface that lets students upload course materials
and chat with an agent powered by Google Gemini that retrieves relevant course information and uses tools.
"""

from pathlib import Path
import streamlit as st
from rag import load_and_index_document, vector_store
from agent import get_agent

# Page configuration
st.set_page_config(
    page_title="ESI StudyMate",
    layout="centered",
)

st.title("ESI StudyMate")
st.caption("AI Study Assistant for University Students (Powered by Google Gemini)")

# Sidebar for document management
with st.sidebar:
    st.header("Course Material")

    uploaded_file = st.file_uploader(
        "Upload a course PDF or text file",
        type=["pdf", "txt"],
        help="Upload lecture slides, course notes, or tutorials.",
    )

    if uploaded_file is not None:
        save_path = Path("./data") / uploaded_file.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        key = f"indexed_{uploaded_file.name}"
        if key not in st.session_state:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                num_chunks = load_and_index_document(str(save_path))
                st.session_state[key] = True
                st.success(f"Indexed {uploaded_file.name} into {num_chunks} chunks!")

    # Quick button to load sample OS notes
    if st.button("Load Sample OS Course Notes", use_container_width=True):
        sample_path = "./data/Operating_Systems_Notes.txt"
        with st.spinner("Indexing sample notes..."):
            chunks_count = load_and_index_document(sample_path)
            st.success(f"Indexed sample course notes ({chunks_count} chunks)!")

    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Initialize conversation messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your course..."):
    # Add student message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate agent response
    with st.chat_message("assistant"):
        with st.spinner("StudyMate is thinking..."):
            agent = get_agent()
            config = {"configurable": {"thread_id": "student-study-session"}}

            try:
                response = agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config=config,
                )
                raw_answer = response["messages"][-1].content
                if isinstance(raw_answer, list):
                    texts = [item.get("text", "") for item in raw_answer if isinstance(item, dict) and "text" in item]
                    answer = "\n".join(texts) if texts else str(raw_answer)
                else:
                    answer = str(raw_answer)
            except Exception as e:
                err_str = str(e)
                if "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str:
                    answer = (
                        "⚠️ **Google Gemini Server High Demand (503)**\n\n"
                        "Google's servers are temporarily experiencing high traffic for this model.\n\n"
                        "💡 **Quick fix:** We've set `GEMINI_MODEL=gemini-flash-lite-latest` in `.env` which is fast and currently active. Please try asking your question again!"
                    )
                elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str:
                    answer = (
                        "⚠️ **Daily Quota Reached for Model (429)**\n\n"
                        "The free tier quota for this specific model has reached its temporary limit.\n\n"
                        "💡 **Quick fix:** Use `GEMINI_MODEL=gemini-flash-lite-latest` in your `.env` file, which has higher quota and is working right now."
                    )
                elif "API_KEY_INVALID" in err_str or "placeholder-key" in err_str or "API key not valid" in err_str:
                    answer = (
                        "⚠️ **Invalid or Missing Google API Key**\n\n"
                        "Please make sure your `.env` file contains a valid Google AI Studio Gemini API key:\n\n"
                        "```bash\n"
                        "GOOGLE_API_KEY=your_key_here\n"
                        "GEMINI_MODEL=gemini-flash-lite-latest\n"
                        "```\n\n"
                        "👉 You can get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey)."
                    )
                elif "PERMISSION_DENIED" in err_str or "denied access" in err_str:
                    answer = (
                        "⚠️ **Google API Permission Denied (403)**\n\n"
                        "This Google API key was denied access by Google's servers.\n\n"
                        "👉 Please generate a fresh free Gemini key at [Google AI Studio](https://aistudio.google.com/app/apikey) and put it into your `.env` file."
                    )
                else:
                    answer = f"Error: {e}\n\nPlease verify your GOOGLE_API_KEY and GEMINI_MODEL in `.env`."

            st.markdown(answer)

    # Save assistant message to UI history
    st.session_state.messages.append({"role": "assistant", "content": answer})
