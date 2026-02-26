import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline


st.set_page_config(page_title="Demo Deployment", layout="wide")
st.title("Demo Deployment")


uploaded_file = st.file_uploader("Upload your PDF", type="pdf")


@st.cache_resource
def build_vectorstore(pdf_path):

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


if uploaded_file:


    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("Document uploaded successfully!")

    vectorstore = build_vectorstore("temp.pdf")

    query = st.text_input("Ask your question")

    if query:


        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5}
        )

        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])


        prompt = f"""
You are an expert assistant.

Answer in 2-3 sentences.
Use ONLY the context below.
If the answer is not in the context, say:
"I don't know based on the document."

Context:
{context}

Question:
{query}

Answer:
"""


        pipe = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_new_tokens=256
        )


        with st.spinner("Thinking..."):
            result = pipe(prompt)

        full_output = result[0]["generated_text"]


        if "Answer:" in full_output:
            answer = full_output.split("Answer:")[-1].strip()
        else:
            answer = full_output.strip()


        st.markdown("###  Answer")
        st.write(answer)