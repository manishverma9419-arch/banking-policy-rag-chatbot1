
import streamlit as st

from rag_engine import RAGEngine, RAGError


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Banking Policy RAG Chatbot",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Banking Policy RAG Chatbot")

st.caption(
    "Ask questions about banking policies, customer grievances, "
    "fraud risk, KYC directions, and the RBI Ombudsman Scheme."
)


# =====================================================
# CACHED ENGINE
# =====================================================

@st.cache_resource(show_spinner=False)
def load_rag_engine():
    return RAGEngine()


# =====================================================
# INITIALIZE APPLICATION
# =====================================================

try:
    with st.spinner(
        "Initializing the policy assistant. "
        "The first run may download the local embedding model..."
    ):
        engine = load_rag_engine()

except Exception as error:
    st.error("The banking policy assistant could not start.")

    st.code(str(error))

    st.markdown(
        """
        **Troubleshooting**

        - Check that the four PDFs exist inside the `data` directory.
        - Check that `.env` contains a valid `GEMINI_API_KEY`.
        - Check your internet connection for the initial model download.
        - If Gemini quota is exhausted, check your API usage and billing.
        """
    )

    st.stop()


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.header("About")

    st.write(
        "This assistant retrieves relevant passages from your "
        "local banking-policy documents and uses Gemini to "
        "generate answers."
    )

    st.write(
        f"**Indexed chunks:** {len(engine.records)}"
    )

    st.write(
        "**Embeddings:** Local Sentence Transformers"
    )

    st.write(
        "**Answer generation:** Gemini API"
    )

    st.divider()

    st.subheader("Available documents")

    documents = sorted({
        record["source"]
        for record in engine.records
    })

    for document in documents:
        st.write(f"📄 {document}")

    st.divider()

    if st.button(
        "Clear chat history",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# =====================================================
# CHAT HISTORY
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================================
# DISPLAY PREVIOUS MESSAGES
# =====================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View retrieved sources"):
                for source in message["sources"]:
                    page = (
                        f", page {source['page']}"
                        if source["page"] is not None
                        else ""
                    )

                    st.markdown(
                        f"- **{source['source']}**{page} "
                        f"(similarity: {source['score']:.3f})"
                    )


# =====================================================
# QUESTION INPUT
# =====================================================

question = st.chat_input(
    "Ask a question about banking policy..."
)


# =====================================================
# PROCESS QUESTION
# =====================================================

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Searching policy documents and preparing your answer..."
            ):
                answer, matches = engine.answer_question(question)

            st.markdown(answer)

            with st.expander("View retrieved sources"):
                for source in matches:
                    page = (
                        f", page {source['page']}"
                        if source["page"] is not None
                        else ""
                    )

                    st.markdown(
                        f"- **{source['source']}**{page} "
                        f"(similarity: {source['score']:.3f})"
                    )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": matches,
            })

        except Exception as error:
            st.error(
                "Sorry, I could not answer this question."
            )

            st.info(str(error))

            st.warning(
                "Your question has not been answered successfully. "
                "Check the error above before retrying."
            )
