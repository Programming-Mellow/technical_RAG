"""
LLM-as-judge evaluation harness for the technical RAG system.

Metrics (no ground-truth required — uses LM Studio as the judge LLM):
  - Faithfulness:     answer is grounded in retrieved context only
  - Answer Relevancy: answer directly addresses the question asked

Refusal accuracy is checked separately for out-of-corpus queries.

Usage:
    python evaluate.py

Requires LM Studio running at localhost:1234 and ChromaDB populated via ingest.py.
"""

import re
from dataclasses import dataclass

import pandas as pd

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
TOP_K = 8
REFUSAL_PHRASE = "I don't have enough information"

# Judge prompts mirror the RAGAS faithfulness / answer-relevancy metrics.
# Keeping context under ~3000 chars so it fits within a 4K local model context.
_FAITHFULNESS_JUDGE = """You are an evaluation assistant. Score how faithfully the answer stays within the provided context.

1.0 = Every claim in the answer is directly supported by the context
0.5 = Some claims are supported but others go beyond the context
0.0 = The answer contradicts or ignores the context entirely

Context (excerpt):
{context}

Answer:
{answer}

Respond with ONLY a single decimal number between 0.0 and 1.0. No explanation."""

_RELEVANCY_JUDGE = """You are an evaluation assistant. Score how well the answer addresses the question.

1.0 = The answer directly and completely addresses the question
0.5 = The answer partially addresses the question
0.0 = The answer is unrelated to the question

Question: {question}

Answer:
{answer}

Respond with ONLY a single decimal number between 0.0 and 1.0. No explanation."""

PROMPT_TEMPLATE = """You are a technical documentation assistant. Answer questions using ONLY the context provided below.

Rules:
- If the answer is not in the provided context, respond with: "I don't have enough information in the documentation to answer that question."
- Do not use your general knowledge to supplement the context.
- Cite the source document name when relevant.
- Be concise and accurate.
- Do not infer or extrapolate beyond what is written.

Context:
{context}

Question: {question}"""


@dataclass
class TestCase:
    question: str
    category: str
    expect_refusal: bool = False


TEST_SUITE: list[TestCase] = [
    # ── Factual / core pillars ───────────────────────────────────────────
    TestCase(
        "What are the five pillars of the AWS Well-Architected Framework?",
        "factual",
    ),
    TestCase(
        "What are the design principles for the Security pillar of the Well-Architected Framework?",
        "factual",
    ),
    TestCase(
        "What disaster recovery strategies does AWS recommend, and how do they differ in RTO and RPO?",
        "factual",
    ),
    TestCase(
        "What are the key cost optimization design principles in the Well-Architected Framework?",
        "factual",
    ),
    TestCase(
        "What best practices does AWS recommend to reduce the environmental impact of cloud workloads?",
        "sustainability",
    ),
    # ── Domain-specific lenses ───────────────────────────────────────────
    TestCase(
        "What unique considerations does the Machine Learning lens add to the Well-Architected Framework?",
        "lens-ml",
    ),
    TestCase(
        "What additional security controls are recommended for financial services workloads on AWS?",
        "lens-fsi",
    ),
    TestCase(
        "What are the tenancy isolation models recommended for SaaS applications on AWS?",
        "lens-saas",
    ),
    TestCase(
        "What compliance and security considerations does the healthcare industry lens add?",
        "lens-healthcare",
    ),
    # ── Architecture patterns ────────────────────────────────────────────
    TestCase(
        "What is cell-based architecture and how does it reduce the blast radius of failures?",
        "architecture",
    ),
    TestCase(
        "How does the serverless lens address operational excellence differently from traditional workloads?",
        "architecture",
    ),
    # ── Generative AI ────────────────────────────────────────────────────
    TestCase(
        "What responsible AI principles should guide generative AI workloads on AWS?",
        "gen-ai",
    ),
    TestCase(
        "How does the generative AI lens address the Reliability pillar?",
        "gen-ai",
    ),
    # ── Out-of-corpus: should trigger the refusal response ───────────────
    TestCase(
        "What is the best recipe for pasta carbonara?",
        "out-of-corpus",
        expect_refusal=True,
    ),
    TestCase(
        "How do I deploy a Kubernetes cluster on Azure?",
        "out-of-corpus",
        expect_refusal=True,
    ),
]


def format_docs(docs: list) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}, page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


def build_components():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    llm = ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key="lm-studio",
        model="local-model",
        temperature=0.1,
    )
    prompt = ChatPromptTemplate.from_messages([("human", PROMPT_TEMPLATE)])
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return retriever, chain, embeddings, llm


def run_test_suite(retriever, chain) -> list[dict]:
    results = []
    for i, tc in enumerate(TEST_SUITE, 1):
        print(f"  [{i:02d}/{len(TEST_SUITE)}] [{tc.category}] {tc.question[:70]}...")
        # Retrieve docs separately so RAGAS can inspect the raw context
        docs = retriever.invoke(tc.question)
        answer = chain.invoke(tc.question)
        results.append({
            "question": tc.question,
            "answer": answer,
            "contexts": [d.page_content for d in docs],
            "category": tc.category,
            "expect_refusal": tc.expect_refusal,
            "refused": REFUSAL_PHRASE.lower() in answer.lower(),
        })
    return results


def _parse_score(text: str, default: float = 0.5) -> float:
    """Extract the first valid 0.0–1.0 decimal from a judge LLM response."""
    match = re.search(r"\b(1\.0+|0\.\d+|[01])\b", text.strip())
    return min(max(float(match.group()), 0.0), 1.0) if match else default


def score_with_llm_judge(results: list[dict], llm) -> pd.DataFrame:
    faith_chain = (
        ChatPromptTemplate.from_messages([("human", _FAITHFULNESS_JUDGE)])
        | llm
        | StrOutputParser()
    )
    rel_chain = (
        ChatPromptTemplate.from_messages([("human", _RELEVANCY_JUDGE)])
        | llm
        | StrOutputParser()
    )

    rows = []
    for i, r in enumerate(results, 1):
        print(f"  [{i:02d}/{len(results)}] judging: {r['question'][:65]}...")
        # Truncate context to ~3000 chars so it fits in a 4K local model window
        context_str = "\n\n".join(r["contexts"])[:3000]
        faith_score = _parse_score(faith_chain.invoke({"context": context_str, "answer": r["answer"]}))
        rel_score = _parse_score(rel_chain.invoke({"question": r["question"], "answer": r["answer"]}))
        rows.append({
            "question": r["question"],
            "faithfulness": faith_score,
            "response_relevancy": rel_score,
        })

    return pd.DataFrame(rows)


def print_report(results: list[dict], scores_df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("RAGAS EVALUATION REPORT")
    print("=" * 80)

    for result, (_, row) in zip(results, scores_df.iterrows()):
        f_score = row.get("faithfulness", float("nan"))
        ar_score = row.get("response_relevancy", float("nan"))
        print(f"\n  [{result['category']}] {result['question']}")
        print(f"    Faithfulness:     {f_score:.3f}  |  Answer Relevancy: {ar_score:.3f}")

        if result["expect_refusal"]:
            status = "PASS - correctly refused" if result["refused"] else "FAIL - should have refused"
            print(f"    Refusal check:    {status}")

        preview = result["answer"][:180].replace("\n", " ")
        print(f"    Answer preview:   {preview}{'...' if len(result['answer']) > 180 else ''}")

    refusal_cases = [r for r in results if r["expect_refusal"]]
    correct_refusals = sum(1 for r in refusal_cases if r["refused"])
    in_corpus = [r for r in results if not r["expect_refusal"]]

    print("\n" + "─" * 80)
    print("AGGREGATE")
    print(f"  Faithfulness (avg):      {scores_df['faithfulness'].mean():.3f}   (1.0 = fully grounded, 0.0 = hallucinated)")
    print(f"  Answer Relevancy (avg):  {scores_df['response_relevancy'].mean():.3f}   (1.0 = on-topic, 0.0 = off-topic)")
    if refusal_cases:
        print(f"  Refusal accuracy:        {correct_refusals}/{len(refusal_cases)} out-of-corpus queries correctly refused")
    print("=" * 80 + "\n")


def main() -> None:
    print("Loading embeddings and ChromaDB...")
    retriever, chain, embeddings, llm = build_components()
    print(f"\nRunning {len(TEST_SUITE)} test queries...\n")
    results = run_test_suite(retriever, chain)
    print("\nScoring with LLM-as-judge (LM Studio)...")
    scores_df = score_with_llm_judge(results, llm)
    print_report(results, scores_df)


if __name__ == "__main__":
    main()
