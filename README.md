# RAG-based PDF Interaction Chatbot

## 1. Create/activate a fresh virtual environment

Windows PowerShell:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
```

If you already have a `venv`, it is recommended to recreate it rather than
trying to repair a mixture of old and new LangChain packages.

## 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Add the API key

Copy `.env.example` to `.env` and put your Google AI API key in it:

```text
GOOGLE_API_KEY=your_google_api_key_here
```

Do not upload `.env` to GitHub.

## 4. Run

```powershell
streamlit run chatpdf1.py
```

## 5. Important

The old `faiss_index` from the previous project was created with an older
embedding setup. Delete it before the first run, or simply upload PDFs and
click **Submit & Process**; the application automatically replaces the old
index.

The application uses:

- `gemini-embedding-001` for embeddings
- `gemini-2.5-flash` for answer generation
- FAISS for local vector search
- `pdfplumber` for PDF text extraction
