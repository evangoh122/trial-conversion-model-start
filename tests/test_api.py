from fastapi.testclient import TestClient
 
from trial_conversion_model.api.main import app
 
client = TestClient(app)
 
 
def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"