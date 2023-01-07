import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://192.168.1.61:8000/"

testdata_path = '/testdata/testdata.csv'

#Call each API endpoint and store the responses
response1 = requests.post(URL + "prediction" + "?data_path=" + testdata_path)
response2 = requests.get(URL + "scoring")
response3 = requests.get(URL + "summarystats")
response4 = requests.get(URL + "diagnostics")

#combine all API responses
responses = [response1, response2, response3, response4]

#write the responses to your workspace
with open('config.json','r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'])

# check if apireturns file exists add counter to file name
if os.path.exists(model_path + "/apireturns.txt"):
    counter = 2
    while os.path.exists(model_path + "/apireturns" + str(counter) + ".txt"):
        counter += 1
    model_path = model_path + "/apireturns" + str(counter) + ".txt"
else:
    model_path = model_path + "/apireturns.txt"

with open(model_path, "w") as f:
    for response in responses:
        f.write(response.text)
