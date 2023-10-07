# Databricks notebook source
# MAGIC %pip install Jinja2==3.0.3 fastapi==0.100.0 uvicorn nest_asyncio databricks-cli gradio==3.37.0 nest_asyncio

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./util/install-llm-libraries

# COMMAND ----------

# MAGIC %run ./util/notebook-config

# COMMAND ----------

import gradio as gr

import re
import time
import pandas as pd

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import PromptTemplate

from util.embeddings import load_vector_db
from util.mptbot import HuggingFacePipelineLocal, TGILocalPipeline
from util.qabot import *
from langchain.chat_models import ChatOpenAI


from langchain import LLMChain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS

# COMMAND ----------

# from huggingface_hub import login
# access_token_read = dbutils.secrets.get(scope="hugging_phase", key="llama")
# login(token = access_token_read)

# COMMAND ----------

def load_vector_db(embeddings_model = 'intfloat/e5-large-v2',
                   config = None,
                   n_documents = 5):
  '''
  Function to retrieve the vector store created
  '''
  if config['model_id'] == 'openai' :
    embeddings = OpenAIEmbeddings(model=config['embedding_model'])
  else:
    if "instructor" in config['embedding_model']:
      embeddings = HuggingFaceInstructEmbeddings(model_name= config['embedding_model'])
    else:
      embeddings = HuggingFaceEmbeddings(model_name= config['embedding_model'])
  
  vector_store = FAISS.load_local(embeddings=embeddings, folder_path=config['vector_store_path'])
  retriever = vector_store.as_retriever(search_kwargs={'k': n_documents}) # configure retrieval mechanism
  return retriever

# Retrieve the vector database:
retriever = load_vector_db(config['embedding_model'],
                           config,
                           n_documents = 5)

# COMMAND ----------

# from huggingface_hub import login
# access_token_read = dbutils.secrets.get(scope="hugging_phase", key="llama")
# login(token = access_token_read)

# COMMAND ----------

# define system-level instructions
system_message_prompt = SystemMessagePromptTemplate.from_template(config['template'])
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

# define model to respond to prompt
llm = TGILocalPipeline.from_model_id(
    model_id=config['model_id'],
    model_kwargs =config['model_kwargs'],
    pipeline_kwargs= config['pipeline_kwargs'])

# Instatiate the QABot
qabot = QABot(llm, retriever, chat_prompt)

# COMMAND ----------

question="what is Unilever's business model in 2012"
x = qabot.get_answer(question) 
x

# COMMAND ----------

import json
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()


# Create the Gradio Template
def respond(question, chat_history):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H%M%S")
    info = qabot.get_answer(question)
    
    chat_history.append((question,info['answer']))
    
    # create the output file  
    output_dict = {"question":question , "answer": info['answer']}
    
    with open(path, "w+") as f:
        json.dump(output_dict, f)
    
    return "", chat_history , info['vector_doc'], info['source']

# COMMAND ----------

print(respond("What is Unilever's revenue for 2022?"))

# COMMAND ----------

from flask import Flask, jsonify, request

app = Flask("llama2-13b-chat")

@app.route('/', methods=['POST'])
def serve_falcon_70b_instruct():
    resp = respond(**request.json)
    return jsonify(resp)


# COMMAND ----------

from dbruntime.databricks_repl_context import get_context
ctx = get_context()

port = "7777"
driver_proxy_api = f"https://{ctx.browserHostName}/driver-proxy-api/o/0/{ctx.clusterId}/{port}"

print(f"""
driver_proxy_api = '{driver_proxy_api}'
cluster_id = '{ctx.clusterId}'
port = {port}
""")


# COMMAND ----------

app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)

# COMMAND ----------

!ps aux | grep 'python'
