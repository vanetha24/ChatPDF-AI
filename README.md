📚 PDF RAG Chatbot
🤖 Ask Questions About Your PDFs Using Generative AI
A Retrieval-Augmented Generation (RAG) based PDF chatbot that allows users to upload PDF documents and ask questions about their content.
The application extracts text from PDFs, splits it into meaningful chunks, converts the chunks into vector embeddings using Google Gemini, stores them in FAISS, and retrieves the most relevant information to generate accurate answers.

🚀 Features
•	 Upload one or multiple PDF files
•	 Extract text from PDF documents
•	 Split documents into smaller text chunks
•	 Generate embeddings using Google Gemini
•	 Store and search embeddings using FAISS
•	 Generate answers using Gemini
•	 Display retrieved PDF sources and page numbers
•	 Clear and rebuild the FAISS index
•	 Simple and interactive Streamlit interface
•	 Answers are restricted to the uploaded PDF context

🛠️ Tech Stack
Technology	Purpose
-> Python	Core programming language
-> Streamlit	Web application interface
-> LangChain	RAG application framework
-> Google Gemini	Embeddings & AI responses
-> FAISS	Vector similarity search
-> pdfplumber	PDF text extraction
-> python-dotenv	Environment variable management

=> How It Works
                PDF Upload
                     │
                     ▼
              Text Extraction
                     │
                     ▼
               Text Chunking
                     │
                     ▼
            Gemini Embeddings
                     │
                     ▼
                FAISS Index
                     │
                     │
               User Question
                     │
                     ▼
              Similarity Search
                     │
                     ▼
           Relevant PDF Chunks
                     │
                     ▼
               Google Gemini
                     │
                     ▼
                  Answer
                     │
                     ▼
               Source Pages

🔄 RAG Pipeline
The application follows these main steps:
1️. Upload PDF
Users upload one or more PDF documents through the Streamlit interface.
2️. Extract Text
Text is extracted page-by-page using pdfplumber.
3️. Chunk the Text
Large text is divided into smaller chunks using LangChain's RecursiveCharacterTextSplitter.
4️. Generate Embeddings
Each text chunk is converted into a numerical vector using:
gemini-embedding-001
5️. Store in FAISS
The generated vectors are stored in a FAISS vector database for efficient similarity search.
6️. Ask a Question
The user enters a question related to the uploaded documents.
7️. Retrieve Relevant Information
FAISS searches for the most relevant chunks based on semantic similarity.
8️. Generate the Answer
The retrieved context is provided to Google Gemini, which generates the final answer.
9️. Show Sources
The application displays the retrieved PDF filename and page number so the user can verify the information.

💻 Installation
1. Clone the repository
git clone YOUR_GITHUB_REPOSITORY_URL
2. Open the project folder
cd RagBased-pdfInteraction-chatbot
3. Create a virtual environment
python -m venv venv
4. Activate the virtual environment
Windows:
venv\Scripts\activate
5. Install dependencies
pip install -r requirements.txt

🔑 Configure Gemini API Key
Create a .env file in the project directory:
GOOGLE_API_KEY=your_google_api_key_here
⚠️ Important: Never upload your .env file or your real API key to GitHub.
The .env file should be included in .gitignore.

▶️ Run the Application
Start the Streamlit application:
python -m streamlit run chatpdf1.py
Then open the local URL shown in the terminal.
Usually:
http://localhost:8501

📖 How to Use
1.	Launch the application.
2.	Upload one or more PDF files.
3.	Click 🚀 Submit & Process.
4.	Wait for the PDF processing to finish.
5.	Enter your question.
6.	Click/submit the question.
7.	The chatbot retrieves relevant information from the PDFs.
8.	Gemini generates the answer.
9.	Expand 📖 View Retrieved Sources to see the supporting PDF pages.

📁 Project Structure
RagBased-pdfInteraction-chatbot/
│
├── 📄 chatpdf1.py
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 .env.example
├── 📄 .gitignore
│
└── 📁 faiss_index/
    ├── index.faiss
    └── index.pkl

🎯 Project Objectives
This project demonstrates practical implementation of:
•	Retrieval-Augmented Generation (RAG)
•	Large Language Models (LLMs)
•	Vector embeddings
•	Semantic search
•	Vector databases
•	Document processing
•	Prompt engineering
•	LangChain
•	Generative AI
•	Streamlit application development

🔮 Future Improvements
Some possible improvements include:
•	 Deploy the application online
•	 Add chat history and conversational memory
•	 Improve retrieval using hybrid search
•	 Add metadata filtering
•	 Support additional document formats
•	 Add user authentication
•	 Add evaluation metrics for RAG quality
•	 Add OCR support for scanned PDFs

👨‍💻 Author : VANETHA A C K
Connect With Me
•	 LinkedIn: www.linkedin.com/in/vanetha24
•	 GitHub: https://github.com/vanetha24
•	 Email: vvanetha633@gmail.com 

⭐ If You Like This Project
If you found this project useful, consider giving it a ⭐ on GitHub!

