from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from datetime import datetime


# Step 1: Load and chunk documents
def load_and_chunk_docs(file_paths):
    documents = []
    for path in file_paths:
        loader = PyMuPDFLoader(path)
        documents += loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    return chunks


# Step 2: Create embeddings and vector DB
def create_vector_db(chunks, persist_directory='./vector_db'):
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb


# Step 3: RAG query using Ollama Mistral
def rag_query(query, vectordb):
    llm = Ollama(model="mistral")
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
    return qa_chain.run(query)


# Step 4: Web Search for latest info
def web_search(query):
    search = DuckDuckGoSearchRun()
    return search.run(query)


# Step 5: Generate financial report with advanced journalistic capabilities
def generate_report(rag_info, web_info, query):
    llm = Ollama(model="mistral")
    current_datetime = datetime.now()
    template = PromptTemplate.from_template(
        """
        You are an elite research analyst in the financial services domain.
        Your expertise encompasses:

        - Deep investigative financial research and analysis
        - Fact-checking and source verification
        - Data-driven reporting and visualization
        - Expert interview synthesis
        - Trend analysis and future predictions
        - Complex topic simplification
        - Ethical practices
        - Balanced perspective presentation
        - Global context integration

        Research Phase:
        - Utilize internal and external data provided
        - Prioritize recent publications and expert opinions

        Analysis Phase:
        - Cross-reference facts
        - Identify emerging trends

        Writing Phase:
        - Craft an attention-grabbing headline
        - Structure in Financial Report style with executive summary, key findings, impact analysis, and future outlook
        - Include relevant quotes and statistics clearly

        Quality Control:
        - Verify facts and ensure narrative readability

        Internal Data:
        {rag_info}

        External Market Data:
        {web_info}

        # Financial Report on {query}

        ## Executive Summary

        ## Background & Context

        ## Key Findings

        ## Impact Analysis

        ## Future Outlook

        ## Expert Insights

        ## Sources & Methodology

        ---
        Research conducted by Financial Agent
        Published: {current_datetime:%Y-%m-%d}
        Last Updated: {current_datetime:%H:%M:%S}
        """
    )
    prompt = template.format(rag_info=rag_info, web_info=web_info, query=query, current_datetime=current_datetime)
    report = llm.invoke(prompt)
    return report


# Main Function (Example Usage)
if __name__ == '__main__':
    paths = ["documents/American-Express-2024-Annual-Report.pdf"]
    chunks = load_and_chunk_docs(paths)
    vectordb = create_vector_db(chunks)

    query = "Provide financial insights about AMEX."
    rag_info = rag_query(query, vectordb)
    web_info = web_search("latest news and market data about AMEX")

    final_report = generate_report(rag_info, web_info, query)

    print(final_report)

