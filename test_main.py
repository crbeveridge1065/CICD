from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_get_path():
    r = client.get("/infer/")
    assert r.status_code == 200
    assert r.json() == {"fetch": "Fetched 1 of 42"}