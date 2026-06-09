from dataclasses import dataclass

from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    source_type: str
    source_name: str
    text: str
    score: float


SYSTEM_PROMPT = """
You are a career intelligence assistant. Use only the provided resume and job description context.
If the context does not support an answer, say what is missing instead of guessing.
Be specific, practical, and honest. Focus on role fit, skill gaps, resume improvements, and interview preparation.
""".strip()


MAX_CHUNK_TEXT_LENGTH = 280
MAX_CONTEXT_LENGTH = 1200


def trim_text(text, max_length):
    if len(text) <= max_length:
        return text

    return text[:max_length].rsplit(" ", 1)[0].strip()


def retrieve_chunks(question, chunks, top_k=3):
    texts = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([question] + texts)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    ranked_chunks = []

    for chunk, score in zip(chunks, scores):
        ranked_chunks.append(
            RetrievedChunk(
                id=chunk.id,
                source_type=chunk.source_type,
                source_name=chunk.source_name,
                text=chunk.text,
                score=float(score),
            )
        )

    ranked_chunks.sort(key=lambda item: item.score, reverse=True)
    return ranked_chunks[:top_k]


def build_context(retrieved_chunks):
    sections = []
    remaining_length = MAX_CONTEXT_LENGTH

    for index, chunk in enumerate(retrieved_chunks, start=1):
        chunk_text = trim_text(chunk.text, min(MAX_CHUNK_TEXT_LENGTH, remaining_length))
        section = (
            f"Source {index}: {chunk.source_name}\n"
            f"Type: {chunk.source_type}\n"
            f"Text:\n{chunk_text}"
        )

        if len(section) > remaining_length:
            break

        sections.append(section)
        remaining_length -= len(section)

        if remaining_length <= 0:
            break

    return "\n\n---\n\n".join(sections)


def generate_answer(client, question, retrieved_chunks, chat_model):
    context = build_context(retrieved_chunks)
    user_prompt = f"""
Question:
{question}

Context:
{context}

Answer briefly with:
1. Fit summary
2. Evidence
3. Gaps
4. Next steps
""".strip()

    response = client.chat.completions.create(
        model=chat_model,
        temperature=0.2,
        max_tokens=700,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content.strip()


def answer_question(
    api_key,
    question,
    chunks,
    chat_model="llama-3.1-8b-instant",
):
    client = Groq(api_key=api_key)
    retrieved_chunks = retrieve_chunks(question, chunks)
    answer = generate_answer(client, question, retrieved_chunks, chat_model)
    return answer, retrieved_chunks
