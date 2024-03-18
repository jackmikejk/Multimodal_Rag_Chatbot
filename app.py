#/*Contributed by Michael Jackson Maria Louis and Swathi Murali*/
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
from typing import Optional
import os
import uuid
import base64
from fastapi import FastAPI, Request, Form, Response, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import json
from dotenv import load_dotenv
load_dotenv()
#Contributed by Michael Jackson Maria Louis - FastAPI for frontend and backend interaction
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)

prompt_template = """You are a vehicle specification providing bot and an expert in giving information about Electrical Vehicals.
Answer the question based only on the following context, which can include text, images and tables:
{context}
Question: {question}
Don't answer if you are not sure and decline to answer and say "Sorry, I don't have much information about it."
Just return the helpful answer in as much as detailed possible.
Answer:
"""

qa_chain = LLMChain(llm=ChatOpenAI(model="gpt-4", openai_api_key = openai_api_key, max_tokens=1024),
                    prompt=PromptTemplate.from_template(prompt_template))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("sample.html", {"request": request})

#contributed by Swathi Murali for the backend endpoint
@app.post("/get_answer")
async def get_answer(question: str = Form(...), context: Optional[str] = Form("")):
    # Parse the existing context
    new_context = context

    relevant_docs = db.similarity_search(question)
    relevant_images = []
    for d in relevant_docs:
        doc_type = d.metadata['type']
        if doc_type == 'text':
            new_context += '[text]' + d.metadata['original_content'] + "||"
        elif doc_type == 'table':
            new_context += '[table]' + d.metadata['original_content'] + "||"
        elif doc_type == 'image':
            new_context += '[image]' + d.page_content + "||"
            relevant_images.append(d.metadata['original_content'])
    
    # Assuming qa_chain.run can process the extended context including the separators "||"
    result = qa_chain.run({'context': new_context, 'question': question})

    response_data = {
        "result": result,
        "context": new_context  # Return the updated context with each response
    }

    if relevant_images:
        # If there are images, return the first image along with the text result and the updated context
        response_data["relevant_images"] = relevant_images[0]

    return JSONResponse(response_data)