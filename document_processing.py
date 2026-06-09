import re
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    source_type: str
    source_name: str
    text: str


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_upload(uploaded_file):
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if file_name.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return clean_text("\n\n".join(pages))

    return clean_text(file_bytes.decode("utf-8", errors="ignore"))


def chunk_text(text, source_type, source_name, chunk_size=450, overlap=75, id_prefix=None):
    cleaned = clean_text(text)

    if not cleaned:
        return []

    chunks = []
    start = 0

    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end].strip()

        if chunk:
            chunk_id = f"{id_prefix or source_type}-{len(chunks) + 1}"
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    source_type=source_type,
                    source_name=source_name,
                    text=chunk,
                )
            )

        if end == len(cleaned):
            break

        start = max(end - overlap, 0)

    return chunks


def build_document_chunks(resume_text, jobs):
    chunks = chunk_text(resume_text, "resume", "Resume", id_prefix="resume")

    for index, job in enumerate(jobs, start=1):
        title = clean_text(job.get("title") or f"Job {index}")
        description = job.get("description", "")
        chunks.extend(chunk_text(description, "job", title, id_prefix=f"job-{index}"))

    return chunks
