"""
app.py - Coffee Bean Recommender
================================
Main application entry point. Combines all modules into a Gradio application.

Run with:
  python app.py

Requires Ollama running with qwen3:0.6b:
  ollama pull qwen3:0.6b && ollama serve
"""
import uuid
import os
import gradio as gr
import duckdb
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

from ingestion import load_vectorstore, build_vectorstore
from retriever import build_mmr_retriever, build_preference_aware_retriever
from chains import build_recommendation_chain, build_preference_chain, build_conversational_chain
load_dotenv()

# ========================================
# Startup
# ========================================
def initialise():
  """
  Load/ Build vector store
    -> Initialise LLM for responses
    -> Initialise retriever
    -> Setup different RAG chains
  """
  chroma_dir = os.environ["CHROMA_DIR"]
  lang_model = os.environ["LANG_MODEL"]

  if not os.path.exists(chroma_dir):
    print(f"[APP] No vector store found - building from data")
    vectorstore = build_vectorstore(reset=True)
  else:
    vectorstore = load_vectorstore()

  # LLM instatiation
  # Can be swapped out for any iteration of LLM from any provider
  # temperature=0: Deterministic output for retrieval
  llm = OllamaLLM(model=lang_model, temperature=0)
  mmr_retriever = build_mmr_retriever(vectorstore, k=5)
  pref_retriever = build_preference_aware_retriever(vectorstore, k=5)

  # Build chain
  simple_chain = build_recommendation_chain(mmr_retriever, llm)
  preference_chain = build_preference_chain(pref_retriever, llm)
  conversational_chain = build_conversational_chain(pref_retriever, llm)

  return simple_chain, preference_chain, conversational_chain

print("[APP] Initialising...")
simple_chain, preference_chain, conversational_chain = initialise()

# ========================================
# Handler Functions
# ========================================
def handle_simple_query(query: str) -> str:
  """Run the basic MMR-retrieval chain"""
  if not query.strip():
    return "Please enter a query to get recommendations."
  return simple_chain.invoke(query)

def handle_preference_query(query: str, liked_beans: list) -> str:
  """Run the preference-aware chain with liked beans context"""
  if not query.strip():
    return "Please enter a query to get recommendations."
  return preference_chain.invoke({
    "question": query,
    "liked_beans": liked_beans
  })

def handle_conversation(message: str, liked_beans: list, history: list, session_id: str):
  """Run a turn of the conversational chain with persistent history"""
  if not message.strip():
    return history, ""

  response = conversational_chain.invoke(
    {"question": message, "liked_beans": liked_beans},
    config={"configurable": {"session_id": session_id}}
  )
  history.append({"role": "user", "content": message})
  history.append({"role": "assistant", "content": response})
  return history, ""

# ========================================
# App Details
# ========================================
duckdb_warehouse = os.environ['DB_PATH']
con = duckdb.connect(duckdb_warehouse)
BEAN_NAMES = [item[0] for item in con.sql("SELECT DISTINCT bean FROM coffee.reviews;").fetchall() if item[0] is not None]

# ========================================
# Gradio UI
# ========================================
# ---- Stylesheet ------------------------
THEME = gr.themes.Base(
  primary_hue="amber",
  secondary_hue="orange",
  neutral_hue="stone",
  font=[gr.themes.GoogleFont("DM Serif Display"), gr.themes.GoogleFont("DM Sans"), "serif"],
).set(
  body_background_fill="#1a1208",
  body_background_fill_dark="#1a1208",
  body_text_color="#f5e6c8",
  body_text_color_dark="#f5e6c8",
  block_background_fill="#2a1e0f",
  block_background_fill_dark="#2a1e0f",
  block_border_color="#6b4c2a",
  block_border_color_dark="#6b4c2a",
  block_label_text_color="#c9a96e",
  input_background_fill="#1e1508",
  input_background_fill_dark="#1e1508",
  input_border_color="#6b4c2a",
  button_primary_background_fill="#c17f24",
  button_primary_background_fill_hover="#d4921e",
  button_primary_text_color="#1a1208",
)
CSS = """
.container { max-width: 900px; margin: 0 auto; }
.title-block { text-align: center; padding: 2rem 0 1rem; }
h1 { font-family: 'DM Serif Display', serif !important; color: #e8c57a !important; font-size: 2.4rem !important; }
h3 { color: #c9a96e !important; }
.tab-nav button { font-family: 'DM Sans', sans-serif !important; }
.prose { color: #d4b896 !important; line-height: 1.7; }
footer { display: none !important; }
"""

# ---- Frontend ------------------------
with gr.Blocks(title="Coffee Recommender") as site:
  # Grant each sessions a unique ID
  session_id = gr.State(str(uuid.uuid4()))

  gr.HTML("""
  <div class="title-block">
    <h1>☕ Coffee Bean Recommender</h1>
    <p style="color:#a07850; font-family:'DM Sans',sans-serif; font-size:1.05rem;">
      Specialty coffee recommendations powered by LangChain &amp; RAG
    </p>
  </div>
  """)

  with gr.Tabs():
    # ---- Tab 1: Simple Query ------------
    with gr.Tab("🔍 Quick Recommend"):
      gr.Markdown(
        "**Simple semantic search** — describe what you're looking for and get "
        "matched recommendations from the catalog. Uses MMR retrieval for diverse results.",
        elem_classes=["prose"]
      )
      with gr.Row():
        simple_input = gr.Textbox(
          label="What are you looking for?",
          placeholder="e.g. 'a bright fruity light roast for pour-over' or 'low acidity espresso bean'",
          lines=2,
        )
      simple_btn    = gr.Button("Find Coffees →", variant="primary")
      simple_output = gr.Markdown(label="Recommendations")

      gr.Examples(
        examples=[
          ["I want something floral and delicate, like drinking tea"],
          ["A bold, chocolatey espresso with low acidity"],
          ["Fruity natural process coffees with berry notes"],
          ["Something unusual and complex I've never tried before"],
          ["I prefer coffees with strong aroma and bold flavors"],
        ],
        inputs=simple_input,
      )
      simple_btn.click(handle_simple_query, inputs=simple_input, outputs=simple_output)

    # ---- Tab 2: Preference-Aware ------------
    with gr.Tab("❤️ My Preferences"):
      gr.Markdown(
        "**Preference-aware recommendations** — select coffees you've enjoyed and "
        "describe what you're after. The chain uses your taste history to find better matches.",
        elem_classes=["prose"]
      )
      liked_selector = gr.CheckboxGroup(
        choices=BEAN_NAMES,
        label="Coffees I've enjoyed",
        info="Select any beans you've tried and liked",
      )
      pref_input = gr.Textbox(
        label="What are you looking for today?",
        placeholder="e.g. 'something similar but with more chocolate' or 'explore a new origin'",
        lines=2,
      )
      pref_btn    = gr.Button("Get Personalised Recommendations →", variant="primary")
      pref_output = gr.Markdown(label="Personalised Recommendations")

      gr.Examples(
        examples=[
          ["Something similar but from a different country"],
          ["I want to try a natural process version of what I like"],
          ["Find me something more exotic and unusual"],
          ["What would pair well with these on a tasting flight?"],
        ],
        inputs=pref_input,
      )
      pref_btn.click(
        handle_preference_query,
        inputs=[pref_input, liked_selector],
        outputs=pref_output,
      )
      
    # ---- Tab 3: Historical Messages ------------
    with gr.Tab("💬 Chat with an Expert"):
      gr.Markdown(
        "**Conversational RAG with memory** — have a multi-turn dialogue with a coffee "
        "expert. The chain remembers your conversation history and builds on previous exchanges.",
        elem_classes=["prose"]
      )
      conv_liked = gr.CheckboxGroup(
        choices=BEAN_NAMES,
        label="Coffees I've enjoyed (optional)",
        scale=1,
      )
      chatbot     = gr.Chatbot(height=420, label="Coffee Sommelier")
      chat_input  = gr.Textbox(
        label="Ask anything about coffee",
        placeholder="e.g. 'What makes Ethiopian coffees so floral?' or 'Recommend something for cold brew'",
        lines=1,
      )
      chat_btn = gr.Button("Send", variant="primary")

      chat_btn.click(
        handle_conversation,
        inputs=[chat_input, conv_liked, chatbot, session_id],
        outputs=[chatbot, chat_input],
      )
      chat_input.submit(
        handle_conversation,
        inputs=[chat_input, conv_liked, chatbot, session_id],
        outputs=[chatbot, chat_input],
      )
  
    with gr.Tab("ℹ️ About"):
      gr.Markdown("""
### Dataset

This app used coffee reviews taken from www.coffeereview.com with a catalog of 8000+ user reviews of
coffee beans, covering diverse origins,processes, roast levels and flavor profiles.
      """)
  
  gr.HTML("""
  <p style="text-align:center; color:#6b4c2a; font-size:0.85rem; padding:1rem 0; font-family:'DM Sans',sans-serif;">
      Powered by LangChain · ChromaDB · HuggingFace Embeddings · Qwen3 via Ollama
  </p>
  """)

if __name__ == "__main__":
  site.launch(
    server_name="0.0.0.0",
    server_port=7860,
    theme=THEME,
    css=CSS
  )


