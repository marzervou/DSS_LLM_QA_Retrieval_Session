# Databricks notebook source
# Databricks notebook source
# MAGIC %pip install Jinja2==3.0.3 fastapi==0.100.0 uvicorn nest_asyncio databricks-cli gradio==3.37.0 nest_asyncio

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./utils/llm_libraries.py

# COMMAND ----------

# MAGIC %run ./utils/config.py

# COMMAND ----------

import re
import time
import pandas as pd

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import PromptTemplate

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS

from utils.embeddings import load_vector_db
from utils.mptbot import HuggingFacePipelineLocal, TGILocalPipeline
from utils.qabot import *
from langchain.chat_models import ChatOpenAI
from utils.app import DatabricksApp

from langchain import LLMChain


# COMMAND ----------

from huggingface_hub import login
access_token_read = dbutils.secrets.get(scope="hugging_phase", key="llama")
login(token = access_token_read)

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

qabot.get_answer("What is Unilever's revenue in 2020?")

# COMMAND ----------

import json
from transformers import AutoTokenizer


from datetime import datetime

def respond(prompt, **kwargs):
    
    start = datetime.now()
    dt_string = start.strftime("%d-%m-%Y-%H%M%S")
    
    # get no of tokens in prompt
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
    
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
                   "response_tokens": info['no_of_tokens'],
                   "source": info['source'],
                   "vector_doc": info['vector_doc'],
                   "inference_time_sec": seconds
                   }
    
    #write the file out
    path = "/dbfs/FileStore/mz_poc/llm_responses/{}.json".format(dt_string)
    with open(path, "w+") as f:
        json.dump(output_dict, f)
    
    return output_dict


# COMMAND ----------

print(respond("What is Unilever's revenue for 2022?"))

# COMMAND ----------

from flask import Flask, jsonify, request

app = Flask("llama2-13b-chat")

@app.route('/', methods=['POST'])
def serve_falcon_70b_instruct():
    # question = request.args.get('question')
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

# COMMAND ----------

# kill the gradio process
# ! kill -9  $(ps aux | grep 'databricks/python_shell/scripts/db_ipykernel_launcher.py' | awk '{print $2}')

# COMMAND ----------



# COMMAND ----------
