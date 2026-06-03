"""
RAG query loop: ChromaDB retrieval → LM Studio (OpenAI-compatible API).

Prerequisites:
  1. Run `python ingest.py` first to populate ChromaDB.
  2. LM Studio running with a model loaded and the local server enabled at
     localhost:1234 (Server tab → Start Server).

Usage:
    python query.py
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
TOP_K = 5

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

prompt = ChatPromptTemplate.from_messages([
    ("human", PROMPT_TEMPLATE),
])


def format_docs(docs: list) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}, page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


def build_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    # LM Studio exposes an OpenAI-compatible endpoint — no real API key needed.
    llm = ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key="lm-studio",
        model="local-model",  # LM Studio ignores this field; the loaded model is used
        temperature=0.1,
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main() -> None:
    print("Loading embeddings and ChromaDB ...")
    chain = build_chain()
    print("Ready. LM Studio must be running at localhost:1234.\n")
    print("Type 'quit' or press Ctrl-C to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break

        response = chain.invoke(query)
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()
