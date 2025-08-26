import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings


load_dotenv()


def get_pdf_text(pdf_docs):
    '''
        Extracts and concatenates text content from a list of PDF files.
    '''
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages: # get content of each page.
            text += page.extract_text()
    return text

def get_text_chunks(text):
    '''
        Splits a large text string into smaller overlapping chunks for easier processing.
    '''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    '''
        Creates a vector store from a list of text chunks using OpenAI embeddings
        and saves it locally using FAISS.

    '''
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")



def get_conversational_chain():

    '''
        Creates a conversational QA chain using OpenAI's GPT model with a custom prompt.

        The chain takes a context and a question, and generates a detailed answer
        based only on the provided context. If the answer is not present in the context,
        it responds with "answer is not available in the context".
    '''

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    '''
        Retrieves relevant documents from a local FAISS vector store based on the user's question,
        runs a conversational QA chain on those documents, and displays the response.
    '''
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question},
        return_only_outputs=True
    )
    print(response)
    st.write("Reply:", response['output_text'])


def main():
    '''
        Launches a Streamlit app that allows users to interact with uploaded PDF files
        by asking questions and receiving answers based on the content.
    '''
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()





