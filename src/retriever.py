"""
retriever.py - Retriever Construction
=====================================
Implements the ChromaDB retriever to get similar documents back
based on the users query. Introduces filtering by metadata.

Retriever is a Runnable with signature:
  str -> List[Document]

Because it is a Runnable it can compose with other Runnables with
the pipe (|) operator.
"""
from typing import Optional, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

# Create multiple different retrievers
def build_similarity_retriever(vectorstore: Chroma, k: int = 6):
  """
  Converts the vectorstore into a retriever
  """
  return vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": k}
  )


def build_mmr_retriever(vectorstore: Chroma, k: int = 6, lambda_mult: float = 0.6):
  """
  MMR balances relevance with diversity. Top-k alone might return similar results.
  """
  return vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
      "k": k,
      "fetch_k": k * 4,
      "lambda_mult": lambda_mult
    }
  )


def build_filtered_retriever(
  vectorstore: Chroma,
  k:      int = 6,
  min_rating: Optional[int] = None,
  roast:  Optional[str] = None,
  aroma:  Optional[int] = None,
  body:   Optional[int] = None,
  flavor: Optional[int] = None,
):
  """
  Filtered retrievers account for the users specified filters and only retrieves
  documents that fulfil those requirements from metadata. 
  """
  conditions = []
  if min_rating:
    conditions.append({"rating": {"gte": min_rating}})
  if roast:
    conditions.append({"roast_level": {"eq": roast}})
  if aroma:
    conditions.append({"aroma": {"gte": aroma}})
  if body:
    conditions.append({"body": {"gte": body}})
  if flavor:
    conditions.append({"flavor": {"gte": flavor}})
  
  search_kwargs = {"k": k}
  if conditions:
    search_kwargs['filter'] = {"$and": conditions} if len(conditions) > 1 else conditions[0]
  
  return vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs=search_kwargs
  )


def build_multiquery_retriever(vectorstore: Chroma, llm, k: int = 4, n_variants: int = 3):
  """
  To improve the accuracy of retrieved documents. Multiquery generates summaries of the existing
  query and passes them to the retriever to get different chunks that essentially mean the same thing.
  It generates N paraphrases and retrieves a subset of documents, then deduplicates them across all queries.

  This improves recall for ambiguous queries or those that have similar phrasing.
  """
  base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

  query_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
    You are a coffee expert. Generate exactly {n_variants} alternate phrasings of the user's coffee query
to improve search recall. Each phrasing should capture a different aspect of the request or use different
vocabulary. Output one question per line, no numbering, no extra text.
    """),
    ("human", "{question}")
  ])

  query_gen_chain = query_gen_prompt | llm | StrOutputParser()

  def _multi_retrieve(query: str) -> List[Document]:
    raw = query_gen_chain.invoke({"question": query})
    variants = [q.strip() for q in raw.strip().splitlines() if q.strip()]
    all_queries = [query] + variants[:n_variants]

    seen, results = set(), []
    for q in all_queries:
      for doc in base_retriever.invoke(q):
        if doc.page_content not in seen:
          seen.add(doc.page_content)
          results.append(doc)
    return results
  
  return RunnableLambda(_multi_retrieve)


def build_preference_aware_retriever(vectorestore: Chroma, k: int = 4):
  """
  Injects the users prefered beans into the context
  """
  def _preference_aware_retriever(inputs: dict) -> List[Document]:
    question = inputs.get("question", "")
    liked_beans = inputs.get("liked_beans", [])

    if liked_beans:
      preference_context = (
        f"The user has enjoyed these coffees: {', '.join(liked_beans)}."
        f"Based on this preference please recommend coffees similar to: {question}"
      )
    else:
      preference_context = question
    
    fetch_k = k + len(liked_beans)
    search_kwargs = {"k": fetch_k, "fetch_k": fetch_k * 3, "lambda_mult": 0.7}

    if liked_beans:
      search_kwargs["filter"] = {"name": {"$nin": liked_beans}}
    
    base_retriever = vectorestore.as_retriever(
      search_type="mmr",
      search_kwargs=search_kwargs
    )

    docs = base_retriever.invoke(preference_context)

    # Secondary filter in case $nin doesn't work
    liked_set = set(liked_beans)
    filtered = [doc for doc in docs if doc.metadata.get("name") not in liked_set]
    
    return base_retriever.invoke(preference_context)
  
  return RunnableLambda(_preference_aware_retriever)