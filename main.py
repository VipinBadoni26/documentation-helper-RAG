import streamlit as st
from backend import run_llm
from dotenv import load_dotenv
import os

# Load environment variables
DOTENV_PATH = "Langchain/documentation-helper/.env"
load_dotenv(DOTENV_PATH)

# --- PAGE CONFIG ---
st.set_page_config(page_title="LangChain Doc Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 LangChain Documentation Chatbot")
st.write("Ask me anything about the LangChain documentation!")

# --- USER INPUT ---
query = st.text_input("Enter your question:", placeholder="e.g. What is a LangChain Chain?")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Generating response..."):
            try:
                response = run_llm(query)
                st.markdown("### 🧠 Answer:")
                st.write(response["result"])

                with st.expander("📚 Retrieved Documents"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Document {i+1}:** {doc.metadata.get('source', 'unknown')}")
                        st.write(doc.page_content[:600] + "...")
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
    else:
        st.warning("Please enter a question first.")
