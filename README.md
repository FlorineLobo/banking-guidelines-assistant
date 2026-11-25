# Banking Guidelines Assistant

### AI-Powered Document Intelligence using Retrieval-Augmented Generation (RAG)

An AI-powered application that allows users to upload banking regulations, financial guidelines, and reports in PDF format and ask questions using natural language.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant passages from uploaded documents and generate evidence-grounded answers with document names and page references.

---

## 1. Project Overview

Banking and financial documents often contain lengthy regulatory instructions, compliance requirements, capital adequacy rules, and financial information.

Manually searching through these documents can be time-consuming.

The Banking Guidelines Assistant addresses this problem by combining semantic search and Large Language Model (LLM) generation to help users retrieve relevant information from multiple PDF documents.

### Key Features

- Upload and manage multiple PDF documents.
- Extract text page by page using PyMuPDF.
- Split extracted text into overlapping chunks.
- Generate semantic embeddings using Sentence Transformers.
- Store and search document embeddings using FAISS.
- Retrieve relevant passages using hybrid retrieval.
- Compare information across multiple documents.
- Generate answers using Google's Gemini API.
- Display source document names and page references.
- Abstain when retrieved evidence is insufficient.
- Manage uploaded PDFs through the Streamlit interface.
- Rebuild the knowledge base when documents are added or removed.

---

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.11 |
| Frontend | Streamlit |
| PDF Processing | PyMuPDF |
| Embedding Model | Sentence Transformers – all-MiniLM-L6-v2 |
| Vector Index | FAISS |
| LLM | Google Gemini |
| Numerical Processing | NumPy |
| Environment Configuration | python-dotenv |
| Testing | pytest |

---

## 3. System Architecture

The application follows a Retrieval-Augmented Generation architecture.

### Document Ingestion Pipeline

1. **PDF Upload:** Users upload one or more PDF documents.
2. **Text Extraction:** PyMuPDF extracts text from individual PDF pages.
3. **Chunking:** Extracted text is divided into smaller overlapping chunks.
4. **Metadata Preservation:** Each chunk retains its document name, page number, and chunk ID.
5. **Embedding Generation:** Sentence Transformers converts text chunks into numerical vectors.
6. **Vector Indexing:** FAISS stores the vectors for efficient similarity search.

### Question Answering Pipeline

1. **User Query:** The user enters a natural-language question.
2. **Query Embedding:** The embedding model converts the question into a vector.
3. **Retrieval:** FAISS searches for semantically similar passages.
4. **Hybrid Ranking:** Semantic retrieval and keyword-based retrieval are combined using Reciprocal Rank Fusion (RRF).
5. **Context Construction:** Relevant passages and their metadata are assembled into a prompt.
6. **Answer Generation:** Gemini generates an answer using the retrieved evidence.
7. **Citation Validation:** The application checks whether cited document and page references exist among the retrieved passages.
8. **Response:** The application displays the answer, source references, and expandable retrieved evidence.

---

## 4. RAG Architecture

### Document Ingestion

```text
             PDF Documents
                    |
                    v
           PyMuPDF Extraction
                    |
                    v
              Text Chunking
          (800 characters,
             150 overlap)
                    |
                    v
          Sentence Transformers
             Embedding Model
                    |
                    v
                FAISS
                    |
                    v
          Stored Chunk Metadata
```

### Question Answering

```text
             User Question
                    |
                    v
             Query Embedding
                    |
                    v
          FAISS Similarity Search
                    |
                    v
            Keyword Retrieval
                    |
                    v
        Reciprocal Rank Fusion
                    |
                    v
          Relevant Text Chunks
                    |
                    v
           Prompt Construction
                    |
                    v
               Gemini API
                    |
                    v
           Citation Validation
                    |
                    v
         Answer + Page References
```

---

## 5. Installation and Setup

### Prerequisites

- Python 3.11
- Git
- Google Gemini API key

### Step 1: Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd banking-guidelines-assistant
```

### Step 2: Create a virtual environment

```bash
python3 -m venv banking-env
```

Activate it on macOS/Linux:

```bash
source banking-env/bin/activate
```

On Windows:

```bash
banking-env\Scripts\activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure environment variables

Create a `.env` file in the project root.

Add the following configuration:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_configured_gemini_model
```

Replace the placeholders with your actual API credentials and configured Gemini model.

**Important:** Never commit your `.env` file or expose your API key publicly. The `.env.example` file provides the configuration template.

### Step 5: Add PDF documents

Place your permitted PDF documents inside:

```text
data/uploads/
```

The application supports multiple PDF documents.

The sample banking PDFs used during development are not included in the Git repository. You must provide your own permitted PDF documents for testing.

### Step 6: Run the application

```bash
streamlit run app.py
```

Open the local Streamlit URL displayed in your terminal.

---

## 6. Using the Application

1. Upload one or more banking PDF documents using the sidebar.
2. Click **Build / Refresh Knowledge Base** to prepare the knowledge base.
3. Select either all documents or an individual document as the search scope.
4. Enter a question in the chat interface or select a suggested question.
5. Review the generated answer and its source references.
6. Expand retrieved evidence to inspect the passages supplied to the AI.
7. Upload or remove documents and refresh the knowledge base when the document collection changes.

---

## 7. Sample Questions

The application can be tested using questions such as:

| Question | Expected Source / Behaviour |
|---|---|
| What is the global speculative-grade default rate forecast by the end of 2026? | Moody's Annual Default Study |
| When do the RBI Directions come into force? | ECL India Guidelines |
| Compare the implementation dates of RBI and Jordan. | RBI and Central Bank of Jordan documents |
| What is the minimum CAR specified by RBI? | Should abstain if sufficient supporting evidence is unavailable |

---

## 8. Project Structure

```text
banking-guidelines-assistant/
│
├── app.py
├── build_knowledge_base.py
├── requirements.txt
├── pytest.ini
├── README.md
├── .gitignore
├── .env.example
│
├── ingestion/
│   ├── __init__.py
│   ├── pdf_loader.py
│   ├── chunker.py
│   └── document_manager.py
│
├── rag/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── generator.py
│   ├── prompts.py
│   └── knowledge_base_manager.py
│
├── data/
│   ├── uploads/
│   └── processed/
│
├── tests/
│   ├── test_chunker.py
│   ├── test_cross_document_retrieval.py
│   ├── test_pdf_loader.py
│   └── test_qa.py
│
└── development_scripts/
    └── Development and debugging scripts
```

**Note:**

- `.env` is created locally and is excluded from version control.
- Uploaded PDFs are stored locally in `data/uploads/`.
- Generated FAISS indexes and processed chunk metadata are stored in `data/processed/`.
- The generated knowledge base can be rebuilt from the uploaded PDFs.
- Development and debugging scripts are kept separately from the main application and automated test suite.

---

## 9. Running Tests

Run the automated test suite from the project root:

```bash
python -m pytest -v
```

The test suite covers:

- PDF text extraction.
- Missing and invalid PDF handling.
- Text chunking.
- Cross-document retrieval.
- Question answering.
- Handling questions with insufficient supporting information.

**Note:** The document-dependent integration tests require PDF files to be placed in `data/uploads/` before running the test suite. The sample banking documents are excluded from version control. Users must provide their own permitted PDF documents for testing.

---

## 10. Design Considerations

### Why RAG?

RAG allows the application to retrieve relevant information from user-provided documents before generating an answer.

This grounds responses in document evidence instead of relying solely on the LLM's pre-trained knowledge.

### Why Sentence Transformers?

The `all-MiniLM-L6-v2` model generates compact 384-dimensional embeddings and can run locally.

This avoids sending every document chunk to an external embedding API.

### Why FAISS?

FAISS provides efficient similarity search over numerical vectors and supports persistent vector indexes.

### Why Hybrid Retrieval?

Semantic search captures meaning, while keyword retrieval helps identify exact terms, dates, and regulatory phrases.

Reciprocal Rank Fusion combines the rankings from these retrieval methods.

### Why Gemini?

Gemini is used as the answer-generation model. It receives the retrieved passages and is instructed to answer using only the supplied evidence.

### Why Abstention?

In banking and regulatory contexts, unsupported answers can be misleading.

The application is instructed to abstain when the retrieved passages do not contain sufficient information.

### Why Preserve Document Metadata?

Each text chunk retains its original document name and PDF page number.

This allows retrieved passages to be traced back to their source documents and supports page-level references in