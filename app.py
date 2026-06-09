import os

import streamlit as st
from dotenv import load_dotenv

from document_processing import build_document_chunks, extract_text_from_upload
from rag_service import answer_question


load_dotenv()


st.set_page_config(
    page_title="Career Intelligence Assistant",
    page_icon="CA",
    layout="wide",
)


if "jobs" not in st.session_state:
    st.session_state.jobs = [{"title": "Job 1", "description": ""}]

if "question" not in st.session_state:
    st.session_state.question = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []


def add_job():
    next_number = len(st.session_state.jobs) + 1
    st.session_state.jobs.append({"title": f"Job {next_number}", "description": ""})


def remove_job(index):
    if len(st.session_state.jobs) > 1:
        st.session_state.jobs.pop(index)


def use_sample_question(question):
    st.session_state.question = question


def get_config_value(name, default=None):
    value = os.getenv(name)

    if value:
        return value

    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


st.title("Career Intelligence Assistant")
st.caption(
    "Upload or paste your resume, compare it against job descriptions, "
    "and ask grounded career-fit questions."
)


with st.sidebar:
    st.header("How it works")
    st.write("1. Add your resume.")
    st.write("2. Add one or more job descriptions.")
    st.write("3. Ask a question about fit, gaps, or interview prep.")
    st.write("4. Review the answer and supporting evidence.")
    st.divider()
    st.caption("Model")
    st.write(get_config_value("GROQ_CHAT_MODEL", "compound-beta"))


left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("Resume")
    resume_file = st.file_uploader("Upload resume", type=["txt", "pdf"])
    resume_text = st.text_area(
        "Or paste resume text",
        height=260,
        placeholder="Paste your resume here...",
    )

with right_col:
    st.subheader("Job Descriptions")

    for index, job in enumerate(st.session_state.jobs):
        with st.container(border=True):
            job["title"] = st.text_input(
                "Job title",
                value=job["title"],
                key=f"job_title_{index}",
            )
            job["description"] = st.text_area(
                "Job description",
                value=job["description"],
                height=180,
                placeholder="Paste the job description here...",
                key=f"job_description_{index}",
            )

            if len(st.session_state.jobs) > 1:
                st.button(
                    "Remove this job",
                    key=f"remove_job_{index}",
                    on_click=remove_job,
                    args=(index,),
                )

    st.button("Add another job", on_click=add_job)


st.divider()
st.subheader("Ask a Question")

sample_col_1, sample_col_2, sample_col_3 = st.columns(3)

with sample_col_1:
    st.button(
        "What skills am I missing?",
        on_click=use_sample_question,
        args=("What skills am I missing for this role?",),
        use_container_width=True,
    )

with sample_col_2:
    st.button(
        "Which job fits me best?",
        on_click=use_sample_question,
        args=("Which job description is the best fit for my resume?",),
        use_container_width=True,
    )

with sample_col_3:
    st.button(
        "How should I prepare?",
        on_click=use_sample_question,
        args=("How should I prepare for interviews for this role?",),
        use_container_width=True,
    )

question = st.text_area(
    "Your question",
    key="question",
    height=110,
    placeholder="Ask about role fit, skill gaps, resume improvements, or interview preparation...",
)

if st.button("Analyze", type="primary"):
    has_resume = bool(resume_text.strip()) or resume_file is not None
    has_job = any(job["description"].strip() for job in st.session_state.jobs)
    has_question = bool(question.strip())

    if not has_resume:
        st.warning("Add your resume before running the analysis.")
    elif not has_job:
        st.warning("Add at least one job description before running the analysis.")
    elif not has_question:
        st.warning("Ask a question before running the analysis.")
    else:
        try:
            uploaded_resume_text = (
                extract_text_from_upload(resume_file) if resume_file else ""
            )
            final_resume_text = uploaded_resume_text or resume_text
            st.session_state.chunks = build_document_chunks(
                final_resume_text,
                st.session_state.jobs,
            )
        except Exception as error:
            st.error(f"Could not read the uploaded file: {error}")
            st.stop()

        if not st.session_state.chunks:
            st.warning("No readable text was found in the provided documents.")
            st.stop()

        resume_chunks = [
            chunk for chunk in st.session_state.chunks if chunk.source_type == "resume"
        ]
        job_chunks = [
            chunk for chunk in st.session_state.chunks if chunk.source_type == "job"
        ]

        api_key = get_config_value("GROQ_API_KEY")

        if not api_key:
            st.error("Add GROQ_API_KEY to your .env file before running analysis.")
            st.stop()

        answer_col, evidence_col = st.columns([2, 1], gap="large")

        with answer_col:
            with st.spinner("Analyzing your resume and job descriptions..."):
                try:
                    answer, retrieved_chunks = answer_question(
                        api_key=api_key,
                        question=question,
                        chunks=st.session_state.chunks,
                        chat_model=get_config_value(
                            "GROQ_CHAT_MODEL",
                            "compound-beta",
                        ),
                    )
                except Exception as error:
                    st.error(f"Analysis failed: {error}")
                    st.stop()

            st.success(
                f"Prepared {len(st.session_state.chunks)} chunks "
                f"({len(resume_chunks)} from resume, {len(job_chunks)} from jobs)."
            )
            st.markdown("### Answer")
            st.markdown(answer)

        with evidence_col:
            st.markdown("### Evidence Used")
            for chunk in retrieved_chunks:
                label = f"{chunk.source_name} · {chunk.score:.2f}"
                with st.expander(label):
                    st.write(chunk.text)
