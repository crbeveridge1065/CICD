from fastapi import FastAPI, Body
import pickle
from pydantic import BaseModel, Field

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

# Load the model and encoder
model_filename = 'model/mlp_model.pkl'
model = pickle.load(open(model_filename, 'rb'))

encoder_filename = 'model/encoder.pkl'
encoder = pickle.load(open(encoder_filename, 'rb'))


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello!, this model is to be used to predict whether a person's income exceeds $50K/yr."}

@app.post("/infer/")
async def create_item(
        item: DataItem = Body(
            None,
            examples=DataItem.Config.schema_extra["examples"],
            )
        ):


    return item