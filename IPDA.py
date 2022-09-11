import json

cityflow_config_file='data/cityflow.config'

configData = json.load(open(cityflow_config_file))
flowFile_path = configData['dir'] + configData['flowFile']


json_data = json.load(open(flowFile_path, "r"))
file_out = open("data/hangzhou/hangzhou_0.json", "w")
i=0
for vehicle in json_data:
    print(vehicle)
    json_data[i]["startTime"] = 0
    i=i+1
file_out.write(json.dumps(json_data))
file_out.close()

