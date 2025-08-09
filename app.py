# app.py

import streamlit as st
from main import load_or_create_vectordb, rag_query, web_search, generate_report

st.set_page_config(page_title=" Financial RAG Analyst", layout="wide")
st.title(" Financial Insights Generator")

st.markdown(
    "Upload company documents and ask financial questions. "
    "The app uses RAG + web search to generate a financial report."
)

query = st.text_input("Enter your financial question:", placeholder="e.g., Provide financial insights about AMEX")

if query:
    with st.spinner("Processing your request..."):
        vectordb = load_or_create_vectordb(["documents/American-Express-2024-Annual-Report.pdf"])
        rag_info = rag_query(query, vectordb)
        web_info = web_search(f"latest news and market data about {query}")
        report = generate_report(rag_info, web_info, query)

        # Save report to file
        with open("final_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

    st.success("✅ Report generated successfully!")
    st.markdown(report)

    # Download button
    st.download_button(
        label="📥 Download Report as TXT",
        data=report,
        file_name="financial_report.txt",
        mime="text/plain"
    )
