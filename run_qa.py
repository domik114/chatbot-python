from similarity_search import similarity_search, vector_db, embedding_model
# from langchain import HuggingFacePipeline, LLMChain, OpenAI, PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
import gradio as gr
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated, List
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

load_dotenv()

llm_model = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
# llm_model= HuggingFacePipeline.from_model_id("google/flan-t5-base", task="text2text-generation", pipeline_kwargs={"max_new_tokens": 1000})
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
LLM_CHAIN = LLMChain(prompt=QA_CHAIN_PROMPT, llm=llm_model, verbose=False)

def generate_answer(question: str) -> str:
    context_docs = similarity_search(question)
    context = "\n\n".join(doc.page_content for doc in context_docs)
    answer = LLM_CHAIN.run(question=question, context=context)
    return answer

def main():
    answer = generate_answer("ile jest problemów z silnikiem?")
    print(answer)
    # demo = gr.Interface(fn=generate_answer, inputs="text", outputs="text")
    # demo.launch()

@app.get("/", response_class=RedirectResponse)
def redirect_to_docs():
    return RedirectResponse(url="/docs", status_code=302)

@app.post("/question", response_class=JSONResponse)
async def recommend_books(question: Annotated[str, Query(max_length=50)]):
    answer = generate_answer(question)
    return JSONResponse(content=answer, status_code=200)

if __name__ == '__main__':
    # main()
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")