
import re
import pinecone
from secret import PINECONE_API, HUGGINGFACEHUB_API_TOKEN
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from dotenv import load_dotenv

load_dotenv()



# Function to read documents
def load_docs(directory):
    loader = PyPDFDirectoryLoader(directory)
    files = loader.load()
    return files


# Passing the directory to the 'load_docs' function
folder = 'Docs/'
documents = load_docs(folder)
print('Length of documents:')
print(len(documents))


# This function will split the documents into chunks
def split_docs(docs, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(docs)
    return docs


split = split_docs(documents)
print('\nLength of split documents:')
print(len(split))

# Hugging Face LLM for creating Embeddings for documents/Text
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
text = "Hello Buddy"
query_result = embeddings.embed_query(text)
print('\nText to be embedded:')
print(text)
print('Length of embeddings:')
print(len(query_result))  # length of embeddings used in Pinecone 'Dimensions' parameter of index

"""
Pinecone allows for data to be uploaded into a vector database and true semantic 
search can be performed.

Not only is conversational data highly unstructured, but it can also be complex. 
Vector search and vector databases allows for similarity searches.

We will initialize Pinecone and create a Pinecone index by passing our documents, 
embeddings model and mentioning the specific INDEX which has to be used
Vector databases are designed to handle the unique structure of vector embeddings, 
which are dense vectors of numbers that represent text. They are used in machine 
learning to capture the meaning of words and map their semantic meaning.

These databases index vectors for easy search and retrieval by comparing values and 
finding those that are most similar to one another, making them ideal for natural 
language processing and AI-driven applications.
"""
pinecone.init(
    api_key=PINECONE_API,  # import from secret.py file
    environment="us-west4-gcp-free"
)

index_name = "mcq-creator"
index = Pinecone.from_documents(split, embeddings, index_name=index_name)


# This function will help us in fetching the top relevent documents from our vector store - Pinecone
def get_similar_docs(query, k=2):
    similar_docs = index.similarity_search(query, k=k)
    return similar_docs


llm = HuggingFaceHub(repo_id="bigscience/bloom",
                     model_kwargs={"temperature": 1e-10},
                     huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

chain = load_qa_chain(llm, chain_type="stuff")


# This function will help us get the answer to the question that we raise
def get_answer(query):
    relevant_docs = get_similar_docs(query)
    print(relevant_docs)
    response = chain.run(input_documents=relevant_docs, question=query)
    return response


our_query = "How is India's economy?"
print('\nQuery:')
print(our_query)
answer = get_answer(our_query)
print(answer)

response_schemas = [
    ResponseSchema(name="question", description="Question generated from provided input text data."),
    ResponseSchema(name="choices", description="Available options for a multiple-choice question in comma separated."),
    ResponseSchema(name="answer", description="Correct answer for the asked question.")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# This helps us fetch the instructions the langchain creates to fetch the response in desired format
format_instructions = output_parser.get_format_instructions()
print('\nFormat instructions:')
print(format_instructions)

# create ChatGPT object
chat_model = ChatOpenAI()

prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("""When a text input is given by the user, please generate multiple choice questions 
        from it along with the correct answer. 
        \n{format_instructions}\n{user_prompt}""")
    ],
    input_variables=["user_prompt"],
    partial_variables={"format_instructions": format_instructions}
)

final_query = prompt.format_prompt(user_prompt=answer)
print(final_query)

final_query_output = chat_model(final_query.to_messages())
print(final_query_output.content)

# Let's extract JSON data from Markdown text that we have
markdown_text = final_query_output.content
json_string = re.search(r'{(.*?)}', markdown_text, re.DOTALL).group(1)
print(json_string)
