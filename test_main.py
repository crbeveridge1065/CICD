from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_get_path():
    r = client.get("")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello!, this model is to be used to predict whether a person's income exceeds  \
                        $50K/yr. An inference value of 0 indicates <=50k/yr, and a value of 1 indicates >50k/yr."}


def test_post_infer_0():
    r = client.post("/infer/", json={
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
        })

    assert r.status_code == 200
    assert r.json() == {"inference": "0"}

def test_post_infer_1():
    r = client.post("/infer/", json={
        "age": 38,
        "workclass": "Private",
        "fnlwgt": 139180,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Divorced",
        "occupation": "Prof-specialty",
        "relationship": "Unmarried",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 15020,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {"inference": "1"}
