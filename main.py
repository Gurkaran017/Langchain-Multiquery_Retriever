from vector_store import get_vector_store
from langchain.retrievers.multi_query import MultiQueryRetriever
import config

def main():
    vectorstore = get_vector_store()
    llm = config.llm  # Use the LLM from config

    # Create retrievers
    similarity_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm
    )

    # Query
    query = "How to improve energy levels and maintain balance?"

    # Retrieve results
    similarity_results = similarity_retriever.invoke(query)
    multiquery_results= multiquery_retriever.invoke(query)    

    for i, doc in enumerate(similarity_results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content)

    print("*"*150)

    for i, doc in enumerate(multiquery_results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content)


if __name__ == "__main__":
    main()
