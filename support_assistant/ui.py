"""
Streamlit Web UI for Zepto GenAI Policy Assistant (/support_assistant/ui.py)
Author: AI/ML Engineer (B.Tech Capstone Project)
"""

import sys
import os
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from rag_engine import run_support_assistant

st.set_page_config(
    page_title="Zepto AI Policy Support Assistant",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Zepto AI Policy Support Assistant")
st.markdown("""
Welcome to **Zepto's AI Customer Support Portal**. Ask any question regarding Zepto's delivery, return, cancellation, pass membership, or support policies below.
""")

st.sidebar.header("⚙️ Configuration & Info")
st.sidebar.markdown("**Mode:** Graded Offline Mock Baseline (`MOCK_LLM=1`)")
st.sidebar.markdown("**Vector Store:** ChromaDB (`zepto_policies`)")
st.sidebar.markdown("**Embedding Model:** `all-MiniLM-L6-v2`")
st.sidebar.markdown("**Corpus:** 8 Grounded Zepto Policy Documents")

query = st.text_input("Enter your customer question:", placeholder="e.g. What is Zepto's return policy for damaged items?")

if st.button("Submit Query", type="primary"):
    if query.strip():
        with st.spinner("Processing query through LangGraph RAG pipeline..."):
            response = run_support_assistant(query)
            
            st.subheader("💡 Assistant Response")
            st.info(response.answer)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Retrieved Source Documents:**")
                if response.sources:
                    for src in response.sources:
                        st.success(f"📄 {src}")
                else:
                    st.write("None (General query answered directly)")
                    
            with col2:
                st.markdown("**Confidence Score:**")
                st.metric("Confidence", f"{response.confidence * 100:.1f}%")
    else:
        st.warning("Please enter a question.")
