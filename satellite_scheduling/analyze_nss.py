import json
import numpy as np

with open("satellite_scheduling/scenarios_larger/nss_results_25.json", "r") as f:
    data = json.load(f)


cleaned_data = []
for item in data:
    cleaned_item = dict()
    cleaned_item["percentSatPerIter"] = item["percentSatPerIter"]
    cleaned_item["messages"] = item["messages"]
    cleaned_item["percentSatisfied"] = item["percentSatisfied"]
    cleaned_item["requestsInCampaign"] = item["requestsInCampaign"]
    cleaned_item["evaluatedPolicy"] = item["evaluatedPolicy"]
    cleaned_data.append(cleaned_item)

print(np.mean(cleaned_item["percentSatisfied"]))
print(np.mean(cleaned_item["messages"]))
