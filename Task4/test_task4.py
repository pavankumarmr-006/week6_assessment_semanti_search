
from fastapi.testclient import TestClient

from task4 import app


# Create test client

client = TestClient(app)


# API key

headers = {

    "x-api-key": "my-secret-key"

}


# --------------------------------
# TEST 1: Health endpoint
# --------------------------------

def test_health():

    response = client.get("/health")

    assert response.status_code == 200


# --------------------------------
# TEST 2: Health message
# --------------------------------

def test_health_message():

    response = client.get("/health")

    data = response.json()

    assert data["status"] == "running"


# --------------------------------
# TEST 3: Wrong API key
# --------------------------------

def test_wrong_api_key():

    response = client.post(

        "/search",

        json={

            "query": "leave policy",

            "top_k": 5

        },

        headers={

            "x-api-key": "wrong-key"

        }

    )

    assert response.status_code == 401


# --------------------------------
# TEST 4: Invalid top_k
# --------------------------------

def test_invalid_top_k():

    response = client.post(

        "/search",

        json={

            "query": "leave policy",

            "top_k": 30

        },

        headers=headers

    )

    assert response.status_code == 422


# --------------------------------
# TEST 5: Missing query
# --------------------------------

def test_missing_query():

    response = client.post(

        "/search",

        json={

            "top_k": 5

        },

        headers=headers

    )

    assert response.status_code == 422


# --------------------------------
# TEST 6: Valid search
# --------------------------------

def test_valid_search():

    response = client.post(

        "/search",

        json={

            "query": "How many annual leaves?",

            "top_k": 5

        },

        headers=headers

    )

    assert response.status_code == 200

    data = response.json()

    assert "results" in data

    assert "query" in data
