# Career Intelligence Assistant

Career Intelligence Assistant is a small RAG application for comparing a resume against one or more job descriptions. The goal is to help a candidate ask practical questions like:

- Am I a good fit for this role?
- What skills am I missing?
- Which job description matches my background best?
- How should I prepare for interviews?

I chose this use case because it is easy to understand, useful in a real workflow, and gives enough room to show retrieval, grounding, prompt design, and sensible trade-offs without making the project unnecessarily large.

## What the App Does

1. Upload or paste a resume.
2. Add one or more job descriptions.
3. Ask a question about role fit, skill gaps, resume improvements, or interview preparation.
4. Split the resume and job descriptions into chunks.
5. Retrieve the chunks most relevant to the question.
6. Generate an answer using only the retrieved context.
7. Show the evidence snippets used for the answer.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Add your Groq API key to `.env`:

```bash
GROQ_API_KEY=your_key_here
GROQ_CHAT_MODEL=llama-3.1-8b-instant
```

## Architecture

```text
Streamlit UI
  |
  |-- resume upload or pasted text
  |-- job description inputs
  |-- user question
  |
  v
Document processing
  |
  |-- text extraction for txt/pdf
  |-- cleanup
  |-- chunking with source metadata
  |
  v
Retrieval
  |
  |-- TF-IDF vectorization
  |-- cosine similarity ranking
  |
  v
LLM answer
  |
  |-- Groq chat model
  |-- prompt with retrieved context
  |-- answer grounded in resume and job evidence
  |
  v
Answer + evidence snippets
```

The code is intentionally split into a few small files:

- `app.py` handles the Streamlit interface and user flow.
- `document_processing.py` handles file extraction, cleanup, and chunking.
- `rag_service.py` handles retrieval, prompt construction, and answer generation.

## RAG Approach

The app uses a straightforward retrieval flow:

1. Clean the resume and job description text.
2. Split the text into overlapping chunks.
3. Convert the question and chunks into local TF-IDF vectors.
4. Rank chunks by cosine similarity.
5. Pass a small set of top chunks to the Groq chat model as context.
6. Ask the model to answer only from that context.

I kept the retrieval layer simple on purpose. For this assignment, the important part is that the app is understandable, working, and grounded in the uploaded material. TF-IDF is not as semantically strong as embeddings, but it is transparent, fast, free to run locally, and good enough for a small resume/job-description demo. The app also keeps a fixed context budget before calling the model so large resumes or job descriptions do not make the request too large. A separate vector database with embedding search would be useful later, but it would add moving parts that are not needed for the first version.

## Key Decisions

**Streamlit instead of a larger frontend/backend stack**

Streamlit made sense here because the core challenge is not frontend complexity. The core challenge is building a clear AI workflow: ingestion, retrieval, prompt design, and evidence display. Streamlit lets the project stay focused on that.

**In-memory retrieval**

The current version builds a local TF-IDF index at analysis time and keeps data in session state. This is enough for a candidate-facing demo where the user uploads a resume and a few job descriptions. It also keeps the setup simple for reviewers.

**Groq for answer generation**

I used Groq for the chat model because it is fast, easy to set up, and has a generous free tier. Since Groq is used only for generation here, the retrieval step stays local.

**Evidence shown beside the answer**

The app shows the retrieved chunks used for the response. This is important because career advice can become vague quickly. Evidence snippets make the answer easier to trust and easier to inspect.

**PDF support included, but kept basic**

The app supports `.pdf` and `.txt` uploads. PDF extraction is handled with `pypdf`, which works for normal text-based PDFs. Scanned resumes would need OCR, which I would add only if the product required it.

## Guardrails

The assistant is instructed to answer only from the provided resume and job description context. If the information is not present, it should say what is missing instead of inventing details.

The app also validates basic input before running analysis:

- resume is required
- at least one job description is required
- a question is required
- readable document text is required

## Production Plan

If I were taking this beyond an assignment, I would make these changes:

- Move embeddings into a persistent vector store such as pgvector, Pinecone, or Weaviate.
- Replace TF-IDF retrieval with embedding-based retrieval for better semantic matching.
- Add user accounts so each user can save resumes, jobs, and previous analyses.
- Store uploaded files securely with encryption and retention controls.
- Add OCR for scanned PDFs.
- Cache embeddings so the same documents are not re-embedded repeatedly.
- Add observability around latency, retrieval quality, API errors, and token usage.
- Add automated tests for chunking, retrieval ranking, and prompt assembly.
- Add clearer privacy messaging because resumes contain sensitive personal data.

For cloud deployment, the simplest path is Streamlit Community Cloud. For a higher-scale production version, I would likely use AWS or GCP with managed storage, a hosted vector database, secrets management, and structured logging.

## Engineering Standards Followed

- Kept the implementation small and readable.
- Separated UI, document processing, and RAG logic.
- Avoided unnecessary frameworks and abstractions.
- Kept setup simple for local review.
- Used environment variables for secrets.
- Displayed evidence instead of returning unsupported advice.
- Chose practical defaults but left the Groq model configurable.

## What I Would Improve With More Time

The biggest improvement would be better evaluation. I would create a small test set of resumes, job descriptions, and expected answer traits, then check whether retrieval is pulling the right evidence before judging the final answer quality.

I would also improve the UI around comparison. For example, if a user uploads three job descriptions, the app could show a side-by-side fit summary before the chat response.

Finally, I would add persistence. Right now the app is session-based, which is fine for a demo, but a real user would expect saved analyses and the ability to come back later.

## How I Used AI Tools

I used AI assistance as a coding partner for scaffolding, checking implementation options, and speeding up repetitive edits. I kept the product scope, architecture choices, and trade-offs deliberately simple because the assignment values clear engineering judgment more than complexity.

I reviewed the generated structure while building and kept the implementation focused on a working core flow rather than adding extra features that would make the demo harder to reason about.
