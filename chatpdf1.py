import os
import shutil
from pathlib import Path

import pdfplumber
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_DIR = Path("faiss_index")

# Gemini models
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.6-flash"


# ============================================================
# API KEY CHECK
# ============================================================

def check_api_key():
    """Check whether the Google API key is configured."""

    if not GOOGLE_API_KEY:
        st.error(
            "❌ GOOGLE_API_KEY is missing.\n\n"
            "Create a .env file in the same folder as chatpdf1.py "
            "and add:\n\n"
            "GOOGLE_API_KEY=your_new_api_key_here"
        )
        return False

    return True


# ============================================================
# GEMINI EMBEDDINGS
# ============================================================

@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Create and cache the Gemini embedding model."""

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


# ============================================================
# GEMINI CHAT MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def get_chat_model():
    """Create and cache the Gemini chat model."""

    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY,
    )


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def get_pdf_documents(pdf_files):
    """
    Extract text from uploaded PDF files.

    Each PDF page becomes a LangChain Document so that
    filename and page number can be preserved.
    """

    documents = []

    for pdf_file in pdf_files:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                total_pages = len(pdf.pages)
                progress_text = st.empty()

                for page_number, page in enumerate(pdf.pages, start=1):
                    progress_text.info(
                        f"📖 Reading {pdf_file.name} "
                        f"(page {page_number}/{total_pages})..."
                    )

                    page_text = page.extract_text()

                    if page_text:
                        page_text = page_text.strip()

                        if page_text:
                            documents.append(
                                Document(
                                    page_content=page_text,
                                    metadata={
                                        "source": pdf_file.name,
                                        "page": page_number,
                                    },
                                )
                            )

                progress_text.empty()

        except Exception as exc:
            st.warning(
                f"⚠️ Could not read '{pdf_file.name}': {exc}"
            )

    return documents


# ============================================================
# TEXT CHUNKING
# ============================================================

def get_text_chunks(documents):
    """Split PDF documents into smaller chunks for retrieval."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
    )

    return splitter.split_documents(documents)


# ============================================================
# CREATE FAISS VECTOR STORE
# ============================================================

def get_vector_store(chunks):
    """
    Create a new FAISS vector database.

    The old index is deleted first so that old embeddings
    are never mixed with the current embedding model.
    """

    if not chunks:
        raise ValueError(
            "No text was extracted from the PDF.\n\n"
            "If your PDF contains scanned images rather than "
            "selectable text, OCR is required."
        )

    # Remove existing FAISS index.
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)

    st.info(
        f"🔄 Creating embeddings for {len(chunks)} text chunks..."
    )

    vector_store = FAISS.from_documents(
        chunks,
        embedding=get_embeddings(),
    )

    vector_store.save_local(str(INDEX_DIR))

    return len(chunks)


# ============================================================
# LOAD FAISS VECTOR STORE
# ============================================================

def load_vector_store():
    """Load the saved FAISS index."""

    if not INDEX_DIR.exists():
        return None

    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True,
    )


# ============================================================
# RAG PROMPT
# ============================================================

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a PDF question-answering assistant.

Answer the user's question using ONLY the supplied PDF context.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not guess.
3. Do not invent information.
4. If the answer is not present in the supplied context,
   say exactly:

   "Answer is not available in the uploaded PDF(s)."

5. Give a clear and concise answer.
6. If useful, mention the PDF filename and page number.
7. Do not invent page numbers or sources.

PDF CONTEXT:

{context}
""",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


# ============================================================
# FORMAT RETRIEVED DOCUMENTS
# ============================================================

def format_context(docs):
    """Convert retrieved documents into text for Gemini."""

    parts = []

    for doc in docs:
        source = doc.metadata.get(
            "source",
            "Unknown PDF",
        )

        page = doc.metadata.get(
            "page",
            "Unknown page",
        )

        parts.append(
            f"""
[Source: {source}, Page: {page}]

{doc.page_content}
"""
        )

    return "\n\n-------------------------\n\n".join(parts)


# ============================================================
# ANSWER USER QUESTION
# ============================================================

def answer_question(question, k=5):
    """
    Search FAISS and ask Gemini to answer using the
    retrieved PDF context.
    """

    vector_store = load_vector_store()

    if vector_store is None:
        raise FileNotFoundError(
            "No PDF has been processed yet. "
            "Upload a PDF and click 'Submit & Process'."
        )

    # Retrieve the most relevant chunks.
    docs = vector_store.similarity_search(
        question,
        k=k,
    )

    if not docs:
        return (
            "Answer is not available in the uploaded PDF(s).",
            [],
        )

    # Build the PDF context.
    context = format_context(docs)

    # Build the Gemini prompt.
    messages = PROMPT.format_messages(
        context=context,
        question=question,
    )

    # Ask Gemini.
    response = get_chat_model().invoke(messages)

    answer = response.content

    # Handle possible list-style responses safely.
    if isinstance(answer, list):
        answer = "".join(
            item.get("text", "")
            if isinstance(item, dict)
            else str(item)
            for item in answer
        )

    return str(answer).strip(), docs


# ============================================================
# CLEAR FAISS INDEX
# ============================================================

def clear_index():
    """Delete the existing FAISS index."""

    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)

        st.success(
            "🗑️ FAISS index deleted successfully."
        )

        st.rerun()


# ============================================================
# MAIN STREAMLIT APP
# ============================================================

def main():

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title("📚 PDF RAG Chatbot")

    st.write(
        "Upload one or more PDF files and ask questions "
        "about their contents."
    )

    st.divider()

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    with st.sidebar:

        st.header("📁 PDF Documents")

        pdf_files = st.file_uploader(
            "Upload your PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        st.write("")

        process_button = st.button(
            "🚀 Submit & Process",
            type="primary",
            use_container_width=True,
        )

        st.divider()

        # ----------------------------------------------------
        # PROCESS PDF BUTTON
        # ----------------------------------------------------

        if process_button:

            if not pdf_files:
                st.warning(
                    "⚠️ Please upload at least one PDF."
                )

            elif not check_api_key():
                st.stop()

            else:

                try:

                    # ----------------------------------------
                    # PROCESSING STATUS
                    # ----------------------------------------

                    with st.status(
                        "Processing PDFs...",
                        expanded=True,
                    ) as status:

                        # ------------------------------------
                        # STEP 1: EXTRACT TEXT
                        # ------------------------------------

                        st.write(
                            "📖 Extracting text from PDFs..."
                        )

                        documents = get_pdf_documents(
                            pdf_files
                        )

                        if not documents:
                            raise ValueError(
                                "No readable text was found "
                                "in the uploaded PDF files."
                            )

                        st.write(
                            f"✅ Extracted text from "
                            f"{len(documents)} pages."
                        )

                        # ------------------------------------
                        # STEP 2: CHUNK TEXT
                        # ------------------------------------

                        st.write(
                            "✂️ Splitting text into chunks..."
                        )

                        chunks = get_text_chunks(
                            documents
                        )

                        st.write(
                            f"✅ Created {len(chunks)} "
                            f"text chunks."
                        )

                        # ------------------------------------
                        # STEP 3: CREATE EMBEDDINGS
                        # ------------------------------------

                        st.write(
                            "🧠 Creating Gemini embeddings..."
                        )

                        chunk_count = get_vector_store(
                            chunks
                        )

                        # ------------------------------------
                        # PROCESSING COMPLETE
                        # ------------------------------------

                        status.update(
                            label="✅ PDF processing completed!",
                            state="complete",
                            expanded=False,
                        )

                    st.success(
                        f"🎉 Done! "
                        f"{len(documents)} pages were processed "
                        f"into {chunk_count} searchable chunks."
                    )

                except Exception as exc:

                    st.error(
                        f"❌ Processing failed:\n\n{exc}"
                    )

        # ----------------------------------------------------
        # INDEX STATUS
        # ----------------------------------------------------

        st.divider()

        st.subheader("📊 Index Status")

        if INDEX_DIR.exists():

            st.success(
                "🟢 FAISS index is ready"
            )

        else:

            st.info(
                "🔴 No FAISS index yet"
            )

        # ----------------------------------------------------
        # CLEAR INDEX
        # ----------------------------------------------------

        if INDEX_DIR.exists():

            if st.button(
                "🗑️ Clear Index",
                use_container_width=True,
            ):
                clear_index()

    # --------------------------------------------------------
    # QUESTION AREA
    # --------------------------------------------------------

    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: What is the main conclusion "
            "of this document?"
        ),
    )

    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    if question:

        if not check_api_key():
            st.stop()

        with st.spinner(
            "🔎 Searching the PDF and generating answer..."
        ):

            try:

                answer, docs = answer_question(
                    question
                )

                # --------------------------------------------
                # DISPLAY ANSWER
                # --------------------------------------------

                st.subheader("🤖 Answer")

                st.write(answer)

                # --------------------------------------------
                # DISPLAY SOURCES
                # --------------------------------------------

                if docs:

                    st.divider()

                    with st.expander(
                        "📖 View Retrieved Sources"
                    ):

                        for i, doc in enumerate(
                            docs,
                            start=1,
                        ):

                            source = doc.metadata.get(
                                "source",
                                "Unknown PDF",
                            )

                            page = doc.metadata.get(
                                "page",
                                "Unknown page",
                            )

                            st.markdown(
                                f"### {i}. {source} — Page {page}"
                            )

                            st.write(
                                doc.page_content
                            )

                            if i < len(docs):
                                st.divider()

            except FileNotFoundError as exc:

                st.warning(
                    f"⚠️ {exc}"
                )

            except Exception as exc:

                st.error(
                    "❌ An error occurred while "
                    "answering the question:"
                )

                st.exception(exc)


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()