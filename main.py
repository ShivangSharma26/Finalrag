"""
Improved Hybrid Chatbot using RAG and General-purpose Responses

This script defines an enhanced chatbot that can intelligently switch between:
1. RAG-based responses (retrieval-augmented generation) for domain-specific queries
2. General-purpose LLM responses for open-domain questions

Key Components:
- Loads and parses a dataset from JSONL format into LangChain `Document` objects.
- Initializes a Chroma vectorstore backed by HuggingFace sentence embeddings for semantic retrieval.
- Uses Gemini's 1.5 Flash model for language generation via `ChatGoogleGenerativeAI`.
- Two response pipelines:
    a. RAG Chain: Retrieves relevant documents and uses context to answer queries related to Maker Bhavan Foundation and similar domains.
    b. General Chain: Provides intelligent, open-ended responses to general user queries using conversation history.
- Utilizes `ConversationBufferMemory` to maintain context across chat turns.
- A list of predefined domain-related keywords helps the chatbot determine whether to use RAG or general logic.

Execution starts from the `chat_interface()` method that enables real-time conversation through a command-line interface.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory

load_dotenv('.env.local')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DOMAIN_KEYWORDS = [
    'maker bhavan', 'iitgn', '3d printing', 'workshop', 'faculty lead',
    'shivang sharma', 'abhi raval', 'pratik mutha', 'aniruddh mali',
    'invention factory', 'prototyping', 'inventx', 'tinkerers’ lab',
    'vishwakarma award', 'leap program', 'skill builder', 'sprint workshop',
    'summer student fellowship', 'industry engagement', 'maker competition',
    'electronics prototyping', 'pcb milling', 'metal 3d printing',
    'fused deposition modeling', 'sla printing', 'laser cutting',
    'vacuum forming', 'cnc', 'digital fabrication', 'interactive design lounge',
    'collaborative classroom', 'project-based learning', 'active learning',
    'experiential education', 'reverse engineering', 'safety training',
    'project officer', 'project management', 'mentorship', 'incubation',
    'startup support', 'student startups', 'patent filing', 'innovation challenge',
    'maker culture', 'hands-on learning', 'hemant kanakia',
    'damayanti bhattacharya', 'maker bhavan foundation', 'science awareness program',
    'indian academic makerspaces summit', 'iam summit', 'center for essential skills',
    'design thinking', 'entrepreneurship', 'collaboration',
    'interdisciplinary learning', 'mechanical fabrication',
    'electronics prototyping zone', 'equipment booking',
    'technology transfer office', 'project proposal', 'industry collaboration',
    'faculty collaboration', 'mentor', 'patron', 'project officer', 'project management'
]

def process_jsonl_to_documents(file_path):
    documents = []
    data = pd.read_json(file_path, lines=True)
    for _, row in data.iterrows():
        system_prompt = row['messages'][0]['content']
        question = row['messages'][1]['content']
        answer = row['messages'][2]['content']
        documents.append(Document(
            page_content=f"Question: {question}\nAnswer: {answer}",
            metadata={"source": "maker_bhavan_dataset", "topic": system_prompt}
        ))
    return documents

class ImprovedChatbot:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            input_key="question"
        )
        self.vectorstore = self.initialize_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.7
        )
        self.rag_chain = self.create_rag_chain()
        self.general_chain = self.create_general_chain()

    def initialize_vectorstore(self):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db_path = "./chroma_db"
        if os.path.exists(db_path):
            return Chroma(persist_directory=db_path, embedding_function=embeddings)
        else:
            docs = process_jsonl_to_documents("dataset.jsonl")
            return Chroma.from_documents(docs, embeddings, persist_directory=db_path)

    def create_rag_chain(self):
        prompt = PromptTemplate.from_template(
            "You are Maker Bhavan's official assistant. Use this context to answer naturally:\n"
            "Context: {context}\n"
            "Chat History: {chat_history}\n"
            "Question: {question}\n"
            "Answer:"

        )
        return (
            RunnablePassthrough.assign(
                context=lambda x: self.retriever.get_relevant_documents(x["question"]),
                chat_history=lambda x: self._format_chat_history(x.get("chat_history", []))
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def create_general_chain(self):
        prompt = PromptTemplate.from_template(
            "You are a helpful assistant who can talk about anything.\n"
            "{chat_history}\n"
            "User: {question}\n"
            "Assistant:"
        )
        return (
            RunnablePassthrough.assign(
                chat_history=lambda x: self._format_chat_history(x.get("chat_history", []))
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_chat_history(self, chat_history):
        return "\n".join(
            f"User: {msg.content}" if msg.type == "human" else f"Assistant: {msg.content}"
            for msg in chat_history[-8:]
        )

    def is_domain_query(self, query):
        return any(keyword in query.lower() for keyword in DOMAIN_KEYWORDS)

    def chat_interface(self):
        print("🤖 Enhanced Hybrid Chatbot Initialized (Type 'exit' to quit)")
        while True:
            query = input("\nYou: ")
            if query.lower() == 'exit':
                break
            try:
                memory_data = {
                    "question": query,
                    "chat_history": self.memory.load_memory_variables({}).get("chat_history", [])
                }
                if self.is_domain_query(query):
                    print("\nBot (RAG): ", end="", flush=True)
                    response = self.rag_chain.invoke(memory_data)
                else:
                    print("\nBot (General): ", end="", flush=True)
                    response = self.general_chain.invoke(memory_data)
                self.memory.save_context({"question": query}, {"answer": response})
                print(response)
            except Exception as e:
                print(f"Error: {str(e)}")
# FastAPI server setup
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Initialize chatbot instance
chatbot = ImprovedChatbot()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (optional, helpful if connecting to a frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request/response schema
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    response: str
    mode: str  # "RAG" or "General"

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        memory_data = {
            "question": request.question,
            "chat_history": chatbot.memory.load_memory_variables({}).get("chat_history", [])
        }
        if chatbot.is_domain_query(request.question):
            response = chatbot.rag_chain.invoke(memory_data)
            mode = "RAG"
        else:
            response = chatbot.general_chain.invoke(memory_data)
            mode = "General"
        chatbot.memory.save_context({"question": request.question}, {"answer": response})
        return ChatResponse(response=response, mode=mode)
    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}", mode="Error")
@app.get("/")
def root():
    return {"message": "Server is running shiavng is og as always!"}
