# Import necessary libraries
import os
import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")



# Initialize Streamlit app
st.title("YouTube Video Summarizer")
st.write("Enter a YouTube video URL and click 'Generate Summary' to get the video summary.")

# Input field for YouTube URL
# input_url = "https://www.youtube.com/watch?v=YvJ4C08pFJQ"
input_url = st.text_input("Enter YouTube Video URL")

# Load llm model using the Groq API
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

if st.button("Generate Summary"):
    if input_url:
        st.write("### Processing...")

        print("input_url", input_url, type(input_url))
        # Load the YouTube video transcript
        loader = YoutubeLoader.from_youtube_url(input_url, add_video_info=False)
        transcript = loader.load()

        print("transcript", transcript)
        # Split the transcript into smaller chunks for better processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
        texts = text_splitter.split_documents(transcript)
        print(len(texts))

        
        # Map Phase - Summarize Individual Chunks
        system_prompt = "You are a YouTube summarizer. Your objective is to help the viewer understand the overall content clearly."
        map_template = "Write a concise summary of the following: {docs}."
        map_prompt = ChatPromptTemplate([("system", system_prompt), ("human", map_template)])

        # Define the LLM chain for mapping (individual chunk summarization)
        map_chain = LLMChain(llm=llm, prompt=map_prompt)

        # Reduce Phase - Combine Summaries into a Final Summary
        reduce_template = """
        The following is a set of summaries:
        {docs}

        Take these summaries and try to generate a consolidated summary as follows:
        1. List the main topics covered in the content.
        2. Provide a brief explanation for each topic.
        3. List the technologies or tools mentioned in the video.
        4. Provide the source of information if mentioned.
        5. Give the final conclusion and key takeaways.
        """
        reduce_prompt = ChatPromptTemplate([("system", system_prompt), ("human", reduce_template)])

        # Define the LLM chain for reduction (combining individual summaries)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        # Define a StuffDocumentsChain to take multiple summaries and combine them into a final summary
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )

        # Define ReduceDocumentsChain to handle cases where the document exceeds token limits
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4000,
        )

        # Map-Reduce Chain - Full Processing Pipeline
        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=False,
        )

        # Run the summarization process
        st.markdown("### Video Summary")
        if len(texts) == 0:
            st.error("No input received")
        else:
            result = map_reduce_chain.invoke(texts)
            # Display the final summary
            st.markdown(result["output_text"], unsafe_allow_html=True)
    else:
        st.error("Please enter a valid YouTube URL.")