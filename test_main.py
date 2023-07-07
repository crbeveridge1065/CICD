from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_get_path():
    r = client.get("")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello!, this model is to be used to predict whether a person's income exceeds $50K/yr. \
                        An inference value of 0 indicates <=50k/yr, and a value of 1 indicates >50k/yr."}