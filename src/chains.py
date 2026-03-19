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
  """
  Extends basic chain to access input of both question and user's bean
  preferences. Liked beans can be retrieved from the user profile and dded
  as context to the prompt.

  Input: {"question": str, "liked_beans": List[str]}
  Output: str (recommendation text)
  """
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
  """
  Structured output forces the LLM to return JSON that conforms to the Pydantic schema.
  It uses the model's native function calls to guarantee the output shape 

  Returns a Pydantic object instead of a string. Good for rendering structured data in
  a UI, or storing in a database. Enable with fallback in case larger models fail
  """
  structured_llm = llm.with_structured_output(RecommendationResponse)

  prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a coffee expert. Analyse the user's query and recommend 3-4 coffess from the catalog
Return structed recommendations.

Available coffees:
{context}
    """),
    ("human", "{question}")
  ])

  chain = (
    RunnableParallel({
      "context": retriever | format_docs,
      "question": RunnablePassthrough()
    })
    | prompt
    | structured_llm
  )
  return chain

# ========================================
# 5. Memory
# ========================================
# In-memory session store
_session_store: dict[str, BaseChatMessageHistory] = {}

def _get_session_history(session_id: str) -> BaseChatMessageHistory:
  """
  Calls function with session_id to retrieve the history of the conversation.
  Use RunnableWithMessageHistory to instantiate the runner + BaseChatMessageHistory
  to implement the base
  """
  if session_id not in _session_store:
    _session_store[session_id] = ChatMessageHistory()
  return _session_store[session_id]


def build_conversational_chain(retriever, llm):
  """
  Builds a conversational LLM which retrieves users message history and injects it
  into the current chat. Allows the LLM to derive more information based on historical
  conversations.

  Utilises the _get_session_history and replaces the history variable in the prompt
  """
  prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly, knowledgeable coffee sommelier. You remeber the
    conversation history and build on previous exchanges. Use the users's evolving preferences
    to refine the recommendation over time

    Coffees they've enjoyed: {liked_beans}

    Current context:
    {context}"""
    ),
    # Injects the historical messages here with the key history
    MessagesPlaceholder(variable_name="history",)
    ("human", "{question}")
  ])

  base_chain = (
    RunnableParallel({
      "context": retriever | format_docs,
      "question": lambda x: x['question'],
      "liked_beans": lambda x: ", ".join(x.get("liked_beans", [])) or "No preference specified yet.",
      "history": lambda x: x['history']
    })
    | prompt
    | llm
    | StrOutputParser()
  )

  # RunnableMessageWithHistory
  # Wraps the chain and injects the message history and pass new query message down
  # input_message_key: which input key contains the current users message
  # history_messages_key: which prompt variable receives the message history
  conversational_chain = RunnableWithMessageHistory(
    base_chain,
    _get_session_history,
    input_messages_key="question",
    history_factory_config="history"
  )
  return conversational_chain
