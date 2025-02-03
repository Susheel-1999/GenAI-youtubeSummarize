# Import necessary libraries
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_groq import ChatGroq
from langchain.chains import LLMChain

os.environ['GROQ_API_KEY'] = 'gsk_ZQmaJyX5NlpxGx8MRKPqWGdyb3FYUGrS4mpW9BMY2tHYrwXo'

input_url = "https://www.youtube.com/watch?v=YvJ4C08pFJQ"

# Load llm model using the groq api
llm = ChatGroq(
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Load the YouTube video transcript
loader = YoutubeLoader.from_youtube_url(input_url, add_video_info=False)
transcript = loader.load()

# Split the transcript into smaller chunks for better processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
texts = text_splitter.split_documents(transcript)

# --------------------------------------
# Map Phase - Summarize Individual Chunks
# --------------------------------------

# Define a system prompt for role specification
system_prompt = "You are a YouTube summarizer. Your objective is to help the viewer understand the overall content clearly."

# Define a template for summarizing each chunk
map_template = "Write a concise summary of the following: {docs}."
map_prompt = ChatPromptTemplate([ ("system", system_prompt),
                                  ("human", map_template)])

# Define the LLM chain for mapping (individual chunk summarization)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# --------------------------------------
# Reduce Phase - Combine Summaries into a Final Summary
# --------------------------------------

# Define a structured template for the final summarized output
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

# Create a ChatPromptTemplate for reduction
reduce_prompt = ChatPromptTemplate([("system", system_prompt),
                                    ("human", reduce_template)])

# Define the LLM chain for reduction (combining individual summaries)
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# --------------------------------------
# Combining Documents - Final Summarization Process
# --------------------------------------

# Define a StuffDocumentsChain to take multiple summaries and combine them into a final summary
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)

# Define ReduceDocumentsChain to handle cases where the document exceeds token limits
reduce_documents_chain = ReduceDocumentsChain(
    # Final combination chain
    combine_documents_chain=combine_documents_chain,
    # If documents exceed token limit, collapse them first
    collapse_documents_chain=combine_documents_chain,
    # Maximum number of tokens to process before collapsing
    token_max=4000,
)

# --------------------------------------
# Map-Reduce Chain - Full Processing Pipeline
# --------------------------------------

# Define a Map-Reduce chain that first maps individual documents, then reduces them
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain to summarize each chunk
    llm_chain=map_chain,
    # Reduce chain to combine summaries
    reduce_documents_chain=reduce_documents_chain,
    # Variable name where the documents are stored in the prompt
    document_variable_name="docs",
    # Do not return intermediate map summaries, only the final output
    return_intermediate_steps=False,
)

# Run the full summarization process
result = map_reduce_chain.invoke(texts)

# Print the final summarized output
print(result["output_text"])
