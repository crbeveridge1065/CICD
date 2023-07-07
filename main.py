from fastapi import FastAPI, Body
import pickle
from pydantic import BaseModel, Field
import pandas as pd
from src.ml.data import process_data
from src.ml.model import train_model, inference, compute_model_metrics

# Declare the data object with its components and their type.
class DataItem(BaseModel):
    age: float
    workclass: str
    fnlwgt: float
    education: str
    education_num: float = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(alias='capital-gain')
    capital_loss: float = Field(alias='capital-loss')
    hours_per_week: float = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')


    class Config:
        schema_extra = {
            "examples": {
                "normal": {
                    "summary": "A normal example",
                    "description": "A **normal** item works correctly.",
                    "value": {
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
                }
            }
        }


# Instantiate the app.
app = FastAPI()

# Load the model and encoder and lb
model_filename = 'model/mlp_model.pkl'
model = pickle.load(open(model_filename, 'rb'))

encoder_filename = 'model/encoder.pkl'
encoder = pickle.load(open(encoder_filename, 'rb'))

lb_filename = 'model/encoder.pkl'
lb = pickle.load(open(lb_filename, 'rb'))

# Features that are categorical
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

input_features = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello!, this model is to be used to predict whether a person's income exceeds $50K/yr. \
                        An inference value of 0 indicates <=50k/yr, and a value of 1 indicates >50k/yr."}

@app.post("/infer/")
async def create_item(
        item: DataItem = Body(
            None,
            examples=DataItem.Config.schema_extra["examples"],
            )
        ):

    global encoder
    global lb
    global model

    # put data into list
    data = [item.age, item.workclass, item.fnlwgt, item.education, item.education_num, item.marital_status,
            item.occupation, item.relationship, item.race, item.sex, item.capital_gain, item.capital_loss,
            item.hours_per_week, item.native_country]

    # create pandas dataframe
    df = pd.DataFrame([data], columns=input_features)

    # Process the data with same encoder and lb as used in training
    X_item, _, encoder, lb = process_data(df, cat_features, training=False, encoder=encoder, lb=lb)

    return {"inference:": str(inference(model, X_item)[0])}