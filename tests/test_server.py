from fastapi.testclient import TestClient

from server.app import app


def test_server():
    # the 'with' block runs startup, which loads the checkpoint
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["model_loaded"] is True

        r = client.post("/generate", json={"prompt": "Hello", "max_tokens": 20})
        assert r.status_code == 200
        assert isinstance(r.json()["text"], str)

        r = client.post("/generate", json={"prompt": "", "max_tokens": 20})
        assert r.status_code == 422   # empty prompt rejected


if __name__ == "__main__":
    test_server()
    print("server OK")
