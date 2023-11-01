# Databricks notebook source
# MAGIC %pip install Jinja2==3.0.3 fastapi==0.100.0 uvicorn nest_asyncio databricks-cli gradio==3.37.0 nest_asyncio

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./util/install-llm-libraries

# COMMAND ----------

!pip install text_generation

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
from util.mptbot import TGILocalPipeline
from util.qabot import *
from langchain.chat_models import ChatOpenAI


from langchain import LLMChain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS

# COMMAND ----------

def load_vector_db(embeddings_model = 'BAAI/bge-large-en',
                   config = None,
                   n_documents = 5):
  '''
  Function to retrieve the vector store created
  '''
  embeddings = HuggingFaceEmbeddings(model_name= config['embedding_model'])
  
  vector_store = FAISS.load_local(embeddings=embeddings, folder_path=config['vector_store_path'])
  retriever = vector_store.as_retriever(search_kwargs={'k': n_documents}) # configure retrieval mechanism
  return retriever

# Retrieve the vector database:
retriever = load_vector_db(config['embedding_model'],
                           config,
                           n_documents = 5)

# COMMAND ----------

# define system-level instructions
system_message_prompt = SystemMessagePromptTemplate.from_template(config['template'])
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

# define the model to respond to prompt
llm = TGILocalPipeline.from_model_id(
    model_id=config['model_id'],
    model_kwargs =config['model_kwargs'],
    pipeline_kwargs= config['pipeline_kwargs'])

# Instatiate the QABot
qabot = QABot(llm, retriever, chat_prompt)

# COMMAND ----------

question="what happens if I lose my keys?"

# COMMAND ----------

# DBTITLE 1,Let's see an example response
x = qabot.get_answer(question) 
x

# COMMAND ----------

# DBTITLE 1,Let's create our API
import json
from transformers import AutoTokenizer


from datetime import datetime

def respond(prompt, **kwargs):
    
    start = datetime.now()
    dt_string = start.strftime("%d-%m-%Y-%H%M%S")
    
    # get no of tokens in prompt
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en")
    
    tokens_prompt = len(tokenizer(prompt).input_ids)
    
    # get answer form llm 
    info = qabot.get_answer(prompt)
    
    # calculate inference time
    end = datetime.now()
    difference = end - start
    
    seconds = difference.total_seconds()

    # create the output file  
    output_dict = {"question": prompt , 
                   "answer": info['answer'], 
                   "prompt_tokens": tokens_prompt,
                   "source": info['source'],
                   "vector_doc": info['vector_doc'],
                   "inference_time_sec": seconds
                   }
    
    return output_dict

# COMMAND ----------

print(respond("what happens if I lose my keys?"))

# COMMAND ----------

from flask import Flask, jsonify, request

app = Flask("llama2-13b-chat")

@app.route('/', methods=['POST'])
def serve_llama2_13b_instruct():
    resp = respond(**request.json)
    return jsonify(resp)


# COMMAND ----------

from dbruntime.databricks_repl_context import get_context
ctx = get_context()

port = "7778"
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

# COMMAND ----------


