# Databricks notebook source
token = "dapi68f7b044a5644c2c75c2ebda6128624d"

# COMMAND ----------

# DBTITLE 1,Hit the API
import requests
import json

def request_llamav2_13b(question, token):
    token = token
    url = 'https://adb-984752964297111.11.azuredatabricks.net/driver-proxy-api/o/0/0918-192555-elmkeq9x/7778'
    
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

import json

json.loads(request_llamav2_13b("what is the duration for the policy?", token=token))


# COMMAND ----------

# what is limit of the misfueling cost covered in the policy?
# what is the name of policy holder?
# what is the duration for the policy?
# what is the duration for the policy bought by the policy holder mentioned in the policy schedule / Validation schedule?
# "what is the vehicle age covered by the policy?"
# "what are the regions covered by the policy?"
# what happens if I lose my keys?

# COMMAND ----------


