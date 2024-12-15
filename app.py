#--- Importing Libraries
import streamlit as st # For building the interactive web interface.
from dotenv import load_dotenv # Loads environment variables (e.g., API keys).
from PyPDF2 import PdfReader # Extracts text from PDF files.
# langchain : Provides tools for natural language processing, text splitting, embeddings, and conversational AI chains.
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS # Used for efficient similarity search.
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template # Custom HTML templates for styling.
from langchain.llms import HuggingFaceHub

#---() This function takes a list of PDF files (pdf_docs) and extracts the text from each page using PyPDF2.
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#---() The function splits the extracted text into chunks of a specific size (1000 characters) with some overlap (200 characters) using CharacterTextSplitter from langchain. This helps in efficient processing and retrieval of relevant information from the PDFs during user queries.
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

#---() Converts the text chunks into vector embeddings using OpenAIEmbeddings and stores these embeddings in a FAISS index. FAISS (Facebook AI Similarity Search) is used for efficient similarity search, allowing the model to quickly find relevant text chunks for answering queries.
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings() # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

#---() This function sets up the ConversationalRetrievalChain which connects a ChatOpenAI model with the FAISS vector store for conversational AI capabilities.
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI() # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# The function handles user input by passing the question to the conversation chain. It stores the chat history in st.session_state so that the conversation can be maintained across interactions.
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Create a scrollable chat container
    with st.container():
        st.markdown("""
            <style>
                .chat-container {
                    max-height: 500px;
                    overflow-y: scroll;
                    padding-right: 10px;
                    padding-left: 10px;
                    border: 1px solid #ccc;
                    border-radius: 8px;
                    margin-top: 20px;
                }
                .message {
                    margin-bottom: 10px;
                }
            </style>
        """, unsafe_allow_html=True)
    
    # Display chat history in the scrollable container
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.chat_message("user").markdown(message.content)
            else:
                st.chat_message("assistant").markdown(message.content)
        st.markdown('</div>', unsafe_allow_html=True)

# The main function sets up the Streamlit interface, loads environment variables, and initializes session states for conversation and chat history.
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                # Show success notification after processing
                st.success("PDFs processed successfully! You can now ask questions about your documents.")

# This block ensures that the main function runs when the script is executed.
if __name__ == '__main__':
    main()
