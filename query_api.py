import requests
import json

inference_item = {
                        "age": 30,
                        "workclass": "Private",
                        "fnlwgt": 345898,
                        "education": "HS-grad",
                        "education-num": 9,
                        "marital-status": "Never-married",
                        "occupation": "Craft-repair",
                        "relationship": "Not-in-family",
                        "race": "Black",
                        "sex": "Male",
                        "capital-gain": 0,
                        "capital-loss": 0,
                        "hours-per-week": 46,
                        "native-country": "United-States"
                    }
response = requests.post('https://income-model-deployment.onrender.com/infer/', data=json.dumps(inference_item))

print(response.status_code)
print(response.json())