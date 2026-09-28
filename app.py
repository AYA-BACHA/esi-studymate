"""ESI StudyMate - Streamlit Application.

A simple AI study assistant interface that lets students upload course materials
and chat with an agent that retrieves relevant course information and uses tools.
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
st.caption("AI Study Assistant for University Students")

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
                answer = response["messages"][-1].content
            except Exception as e:
                answer = f"Error: {e}. Please ensure your OPENAI_API_KEY is set in your .env file."

            st.markdown(answer)

    # Save assistant message to UI history
    st.session_state.messages.append({"role": "assistant", "content": answer})
