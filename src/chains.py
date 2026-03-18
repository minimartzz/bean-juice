"""
chains.py - Chain Construction
==============================
Creates the LangChain chains to transform and find documents.
Every component - prompts, models, retrievers, parsers - is a Runnable.
Runnables connect with | to form pipelines. The resulting chain is a
Runnable itself.

Components:
"""
from typing import List
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
  RunnableParallel,
  RunnablePassthrough,
  RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
# Memory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.documents import Document

# ========================================
# 1. Document Formatter
# ========================================
def _format_docs(docs: List[Document]) -> str:
  parts = []
  for i, doc in enumerate(docs, 1):
    name = doc.metadata.get("name", "Unknown")
    location = doc.metadata.get("location", "Unknown")
    origin = doc.metadata.get("origin", "Unknown")
    roaster = doc.metadata.get("roaster", "Unknown")
    parts.append(f"[{i}] {name} ({origin}) By: {roaster}\n{doc.page_content}")
  return "\n\n---\n\n".join(parts)

format_docs = RunnableLambda(_format_docs)

# ========================================
# 2. Simple Recommendation Chain
# ========================================
def build_recommendation_chain(retriever, llm):
  """
  ── Basic LCEL pipe chain ─────────────────────────────────────────────────
  The simplest production-grade RAG chain. Uses only langchain_core.

  Data flow:
    question: str
      -> RunnableParallel runs both branches simultaneously
    {
      "context":  retriever → format_docs  (str)
      "question": RunnablePassthrough()    (str, unchanged)
    }
      -> ChatPromptTemplate fills {context} and {question}
    List[BaseMessage]
      -> LLM generates a response
    AIMessage
      -> StrOutputParser extracts the text
    str
  """
  prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a world-class coffee expert and sommelier.
Your task is to recommend coffee beans from the provided catalog based on the user's preferences.

For each recommendation:
1. Name the coffee and explain WHY it matches their request
2. Describe the key flavor notes and what makes it special
3. Suggest the best brew method for their preference
4. Mention the price and roaster

Be enthusiastic, specific, and educational. Explain coffee concepts when relevant.
Always recommend between 2-4 options with clear reasoning.

If the user mentions coffees they've enjoyed, use those as anchors to explain
how your recommendations compare and contrast.

Available coffees:
{context}"""),
      ("human", "{question}")
    ])

  chain = (
    RunnableParallel({
      "context": retriever | format_docs,
      "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
  )
  
  return chain

# ========================================
# 3. User Preference Chain
# ========================================
def build_preference_chain(preference_retriever, llm):
  prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert coffee sommelier with deep knowledge of
specialty coffee origins, processing methods, and flavor science.

The user has told you which coffees they've enjoyed. Use this to understand
their palate and make targeted recommendations from the catalog.

When making recommendations:
- Explicitly reference the coffees they've liked and explain the connection
- Use sensory language (acidity, body, sweetness, flavor notes)
- Educate them about why certain origins/processes produce certain flavors
- Suggest brew methods that will highlight the best qualities

Coffees the user has enjoyed: {liked_beans}

Available catalog:
{context}"""),
    ("human", "{question}")
  ])

  def _extract_liked_beans(inputs: dict) -> str:
    liked = inputs.get("liked_beans", [])
    return ", ".join(liked) if liked else "No preferences specified yet"
  
  chain = (
    RunnableParallel({
      "context": preference_retriever | format_docs,
      "question": lambda x: x["question"],
      "liked_beans": RunnableLambda(_extract_liked_beans)
    })
    | prompt
    | llm
    | StrOutputParser()
  )

  return chain

# ========================================
# 4. Structured Output Chain
# ========================================
class CoffeeRecommendation(BaseModel):
  """Pydantic model for structured output"""
  name: str               = Field(description="Name of coffee beans")
  roaster_location: str   = Field(description="Location of roaster")
  origin: str             = Field(description="Country of origin for coffee beans")
  match_reason: str       = Field(description="Why this coffee was selected")
  flavor_notes: List[str] = Field(description="Key flavor notes")
  bean_details: List[str] = Field(description="Details about the bean like Aroma, Body, Aftertaste, With Milk")
  brew_method: str        = Field(description="Recommended brew method")
  price: float            = Field(description="Price of bean")
  confidence: str         = Field(description="High / Medium / Low match confidence")

class RecommendationResponse(BaseModel):
  """Strcutured response containing multiple recommendations"""
  recommendations: List[CoffeeRecommendation] = Field(
    description="List of 2-4 coffee recommendations"
  )
  summary: str = Field(description="A one-paragraph summary of the recommendations")
  palate_analysis: str = Field(
    description="Brief analysis of what the user's preferences reveal about their palate"
  )

def build_structured_chain(retriever, llm):
  structured_llm = llm.with_structured_output(RecommendationResponse)

  prompt