import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import base64
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to encode image to Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Function to read PDF content
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if the answer is not in provided context just say, "answer is not available in the context", 
    don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Allow dangerous deserialization here
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Main Streamlit application
def main():
    # Configure Page
    st.set_page_config(
        page_title="Chat with PDF",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # CSS for custom styling
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bg_image_path = os.path.join(current_dir, "image.png")
    bg_image_base64 = get_base64_image(bg_image_path)

    st.markdown(
        f"""
        <style>
        body {{
            background-image: url("data:image/png;base64,{bg_image_base64}");
            background-size: cover;
            background-attachment: fixed;
            color: black;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: darkgreen !important;
        }}
        .stSidebar {{
            background-color: #f0f0f0 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header Section
    st.title("📄 Chat with Your PDF using Gemini")
    st.markdown(
        """
        Upload your PDF documents and get answers to your questions with AI. 
        Powered by **Gemini** for intelligent and fast insights. 🎉
        """
    )

    # Question Input Section
    st.markdown("### 🤔 Ask a Question")
    user_question = st.text_input("Type your question here and press Enter", "")

    if user_question:
        with st.spinner("Searching for answers..."):
            user_input(user_question)

    # Sidebar for File Uploads
    with st.sidebar:
        st.header("📂 File Upload")
        st.write("Upload one or more PDF files to start processing:")
        pdf_docs = st.file_uploader(
            "Drag and drop your files here", accept_multiple_files=True, type=["pdf"]
        )

        if st.button("📥 Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing your files..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("🎉 Files processed successfully!")
            else:
                st.error("⚠️ Please upload at least one PDF file to proceed.")

    # Footer
    st.markdown(
        """
        ---
        👨‍💻 Created by **Anand Khanna** | Powered by [LangChain](https://langchain.com/) and [Streamlit](https://streamlit.io/)
        """
    )

if __name__ == "__main__":
    main()
