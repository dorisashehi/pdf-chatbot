# PDF Chatbot

About this web app: **This project is a **Streamlit app** that allows you to upload one or more PDF files and then **ask questions** about their content.
It uses **LangChain**, **OpenAI embeddings**, and **FAISS** to create a vector database of your documents and provide context-aware answers. **

## Required Features

The following **required** functionality is completed:

- [x] **Upload multiple PDF files.**
- [x] **Extract and split text into overlapping chunks for efficient processing.**
- [x] **Generate embeddings using **OpenAI’s `text-embedding-3-large`** model.**
- [x] **Store embeddings locally in a **FAISS vector database**.**
- [x] **Ask natural language questions about your PDFs and get precise answers.**
- [x] **If the answer is not in the document, the app will say: `"answer is not available in the context"`.**

## Installation

- Clone this repository
- git clone https://github.com/your-username/chat-with-pdf.git
- cd chat-with-pdf
- python -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt
- Create a .env file in the project root and add your OpenAI API key: OPENAI_API_KEY=your_openai_api_key_here
- Run the Streamlit app: streamlit run app.py
- venv/ and faiss_index/ should be added to .gitignore so they aren’t tracked by Git.

# Technologies

- [Streamlit](https://streamlit.io/) – for the interactive UI.
- [PyPDF2](https://pypi.org/project/PyPDF2/) – to extract text from PDFs.
- [LangChain](https://www.langchain.com/) – for text splitting, embeddings, and conversational chains.
- [FAISS](https://faiss.ai/) – for fast vector similarity search.
- [OpenAI API](https://platform.openai.com/) – for embeddings & GPT-powered Q&A.
- [python-dotenv](https://pypi.org/project/python-dotenv/) – to manage environment variables.
