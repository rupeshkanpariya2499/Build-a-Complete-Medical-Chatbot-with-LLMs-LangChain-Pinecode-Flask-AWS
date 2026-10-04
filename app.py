from flask import Flask, render_template, request
from langchain_core.documents import Document
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import wikipedia
import os


app = Flask(__name__)
wikipedia.set_user_agent("MedicalChatbot/1.0")

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_embeddings()

index_name = os.getenv("PINECONE_INDEX_NAME", "medicalchatbot-openai")

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatGroq(
    model="openai/gpt-oss-20b")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)


def get_wikipedia_documents(query):
    """Return short Wikipedia summaries relevant to the user's question."""
    wikipedia_documents = []
    try:
        titles = wikipedia.search(query, results=2)
        for title in titles:
            try:
                summary = wikipedia.summary(title, sentences=4, auto_suggest=False)
                wikipedia_documents.append(
                    Document(
                        page_content=summary,
                        metadata={"source": f"Wikipedia: {title}"},
                    )
                )
            except Exception as error:
                print(f"Wikipedia page lookup failed for {title}: {error}")
                continue
    except Exception as error:
        print(f"Wikipedia lookup failed: {error}")

    return wikipedia_documents

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    pdf_documents = retriever.invoke(msg)
    wikipedia_documents = get_wikipedia_documents(msg)
    response = question_answer_chain.invoke(
        {"input": msg, "context": pdf_documents + wikipedia_documents}
    )
    print("Response : ", response)
    return str(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port= 8000, debug=True)