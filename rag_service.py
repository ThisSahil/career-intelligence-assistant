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


def retrieve_chunks(question, chunks, top_k=7):
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

    for index, chunk in enumerate(retrieved_chunks, start=1):
        sections.append(
            f"Source {index}: {chunk.source_name}\n"
            f"Type: {chunk.source_type}\n"
            f"Text:\n{chunk.text}"
        )

    return "\n\n---\n\n".join(sections)


def generate_answer(client, question, retrieved_chunks, chat_model):
    context = build_context(retrieved_chunks)
    user_prompt = f"""
Question:
{question}

Context:
{context}

Answer with these sections:
1. Direct answer
2. Supporting evidence
3. Skill gaps or risks
4. Recommended next steps
""".strip()

    response = client.chat.completions.create(
        model=chat_model,
        temperature=0.2,
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
    chat_model="compound-beta",
):
    client = Groq(api_key=api_key)
    retrieved_chunks = retrieve_chunks(question, chunks)
    answer = generate_answer(client, question, retrieved_chunks, chat_model)
    return answer, retrieved_chunks
