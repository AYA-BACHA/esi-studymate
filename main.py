"""CLI Entrypoint for ESI StudyMate.

Allows direct command-line interaction and one-command Streamlit startup.
"""

import sys
import subprocess
from pathlib import Path
from app.agents.router import execute_agent
from app.rag.indexer import document_indexer
from app.rag.vectorstore import course_vector_store
from app.memory.conversation import conversation_memory


def run_cli():
    """Interactive terminal session."""
    print("=" * 65)
    print("ESI StudyMate — Academic Study Assistant")
    print("=" * 65)

    stats = course_vector_store.get_stats()
    print(f"[*] Chroma Vector DB: {stats['total_chunks']} chunks indexed.")
    if stats["total_chunks"] == 0:
        print("[*] Loading default sample OS course documents...")
        from data.sample_data_loader import create_sample_documents
        ch3, ch4 = create_sample_documents()
        document_indexer.index_file(ch3)
        document_indexer.index_file(ch4)
        print("[+] Default course documents indexed.")

    print("\nEnter a query (or 'exit' to quit, 'clear' to reset memory):")
    print("-" * 65)

    while True:
        try:
            query = input("\nStudent > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("\nSession ended.")
                break
            if query.lower() == "clear":
                conversation_memory.clear()
                print("[*] Conversation memory cleared.")
                continue

            contextual_q = conversation_memory.contextualize_query(query)
            conversation_memory.add_message(role="user", content=query)

            result = execute_agent(
                query=contextual_q,
                teaching_style="standard",
                existing_messages=conversation_memory.get_messages(),
                summary=conversation_memory.summary,
            )

            print(f"\n[Agent: {result.route_decision.upper()}]")
            print(result.final_response)

            conversation_memory.add_message(
                role="assistant",
                content=result.final_response,
                sources=result.sources,
            )

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break


def launch_streamlit():
    """Launch Streamlit web application."""
    ui_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(ui_path)]
    subprocess.run(cmd)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ui":
        launch_streamlit()
    else:
        run_cli()
