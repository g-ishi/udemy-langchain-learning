from dotenv import load_dotenv

load_dotenv()

from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from graph.ingestion import retriever


def test_foo() -> None:
    assert 1 == 1


def test_retrieval_grade_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )  # type: ignore

    assert res.binary_score == "yes"
