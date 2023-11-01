# Databricks notebook source
# DBTITLE 1,Install Required Libraries
# MAGIC %run "./util/install-prep-libraries"

# COMMAND ----------

# DBTITLE 1,Import Required Functions
import json

import pyspark.sql.functions as fn
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS

from util.pre_process import preprocess_using_langchain , preprocess_using_formrecognizer

# COMMAND ----------

# MAGIC %run "./util/notebook-config"

# COMMAND ----------

# MAGIC %md ##Step 1: Load the Raw Data to Table

# COMMAND ----------

if config['use_azure_formrecognizer'] == True:
  df = preprocess_using_formrecognizer(config)
else:
  df = preprocess_using_langchain(config)

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md We can persist this data to a table as follows:

# COMMAND ----------

# DBTITLE 1,Save Data to Table
# save data to table
_ = (
  spark.createDataFrame(df)
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable(config['use-case'])
  )

# count rows in table
print(spark.table(config['use-case']).count())

# COMMAND ----------

# MAGIC %md ##Step 2: Prepare Data for Indexing
# MAGIC
# MAGIC While there are many fields avaiable to us in our newly loaded table, the fields that are relevant for our application are:
# MAGIC
# MAGIC * text - Documentation text or knowledge base response which may include relevant information about user's question
# MAGIC * source - the url pointing to the online document

# COMMAND ----------

# DBTITLE 1,Retrieve Raw Inputs
raw_inputs = (
  spark
    .table(config['use-case'])
    .selectExpr(
      'full_text',
      'source'
      )
  ) 

display(raw_inputs)

# COMMAND ----------

# MAGIC %md Please note that we are specifying overlap between our chunks.  This is to help avoid the arbitrary separation of words that might capture a key concept. 
# MAGIC
# MAGIC We have set our overlap size very small for this demonstration but you may notice that overlap size does not neatly translate into the exact number of words that will overlap between chunks. This is because we are not splitting the content directly on words but instead on byte-pair encoding tokens derived from the words that make up the text.  You can learn more about byte-pair encoding [here](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt) but just note that its a frequently employed mechanism for compressing text in many LLM algorithms.

# COMMAND ----------

# DBTITLE 1,Chunking Configurations
chunk_size = 512
chunk_overlap = 50

# COMMAND ----------

# DBTITLE 1,Divide Inputs into Chunks
@fn.udf('array<string>')
def get_chunks(text):

  # instantiate tokenization utilities
  text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  
  # split text into chunks
  return text_splitter.split_text(text)


# split text into chunks
chunked_inputs = (
  raw_inputs
    .withColumn('chunks', get_chunks('full_text')) # divide text into chunks
    .drop('full_text')
    .withColumn('num_chunks', fn.expr("size(chunks)"))
    .withColumn('chunk', fn.expr("explode(chunks)"))
    .drop('chunks')
    .withColumnRenamed('chunk','text')
  )

  # display transformed data
display(chunked_inputs)

# COMMAND ----------

# MAGIC %md ##Step 4: Create Vector Store
# MAGIC
# MAGIC With our data divided into chunks, we are ready to convert these records into searchable embeddings. Our first step is to separate the content that will be converted from the content that will serve as the metadata surrounding the document:

# COMMAND ----------

# DBTITLE 1,Separate Inputs into Searchable Text & Metadata
# convert inputs to pandas dataframe
inputs = chunked_inputs.toPandas()

# extract searchable text elements
text_inputs = inputs['text'].to_list()

# extract metadata
metadata_inputs = (
  inputs
    .drop(['text','num_chunks'], axis=1)
    .to_dict(orient='records')
  )

# COMMAND ----------

# MAGIC %md Next, we will initialize the vector store into which we will load our data.  If you are not familiar with vector stores, these are specialized databases that store text data as embeddings and enable fast searches based on content similarity.  We will be using the [FAISS vector store](https://faiss.ai/) developed by Facebook AI Research. It's fast and lightweight, characteristics that make it ideal for our scenario.
# MAGIC
# MAGIC The key to setting up the vector store is to configure it with an embedding model that it will used to convert both the documents and any searchable text to an embedding (vector). You have a wide range of choices avaialble to you as you consider which embedding model to employ.  Some popular models include the [sentence-transformer](https://huggingface.co/models?library=sentence-transformers&sort=downloads) family of models available on the HuggingFace hub as well as the [OpenAI embedding models](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings):
# MAGIC
# MAGIC **NOTE** The OpenAI API key used by the OpenAIEmbeddings object is specified in an environment variable set during the earlier `%run` call to get configuration variables.

# COMMAND ----------

# DBTITLE 1,Load Vector Store
# identify embedding model that will generate embedding vectors
embeddings = HuggingFaceEmbeddings(model_name=config['embedding_model'])

# instantiate vector store object
vector_store = FAISS.from_texts(
  embedding=embeddings, 
  texts=text_inputs, 
  metadatas=metadata_inputs)

# COMMAND ----------

# MAGIC %md So that we make use of our vector store in subsequent notebooks, let's persist it to storage:

# COMMAND ----------

# DBTITLE 1,Persist Vector Store to Storage
vector_store.save_local(folder_path=config['vector_store_path'])

# COMMAND ----------

dbutils.notebook.exit("exit")

# COMMAND ----------


