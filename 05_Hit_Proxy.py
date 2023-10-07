# Databricks notebook source
# Databricks notebook source
import requests
import json

def request_llamav2_13b(question):
    token = "dapi00308e6d4b31b20e88ada51931d2c2fb-2"
    url = 'https://adb-3744970589451902.2.azuredatabricks.net/driver-proxy-api/o/0/0724-082317-yu51nqbk/7777'
    
    headers = {
        "Content-Type": "application/json",
        "Authentication": f"Bearer {token}"
        }
    
    data = {
     "prompt": question
     }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.text






# COMMAND ----------

request_llamav2_13b("What initiatives did Unilever take to reduce its environmental impact in 2020 according to the 2020 Annual Report")

