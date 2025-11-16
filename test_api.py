"""
Simple test script to verify the API is working.
Run this after starting the server.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_ask_post():
    """Test /ask endpoint with POST."""
    print("Testing /ask endpoint (POST)...")
    question = "When is Layla planning her trip to London?"
    response = requests.post(
        f"{BASE_URL}/ask",
        json={"question": question}
    )
    print(f"Question: {question}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_ask_get():
    """Test /ask endpoint with GET."""
    print("Testing /ask endpoint (GET)...")
    question = "How many cars does Vikram Desai have?"
    response = requests.get(
        f"{BASE_URL}/ask",
        params={"question": question}
    )
    print(f"Question: {question}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_insights():
    """Test /insights endpoint."""
    print("Testing /insights endpoint...")
    response = requests.get(f"{BASE_URL}/insights")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Member Data QA API")
    print("=" * 50)
    print()
    
    try:
        test_health()
        test_ask_post()
        test_ask_get()
        test_insights()
        print("=" * 50)
        print("All tests completed!")
        print("=" * 50)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API.")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Error: {e}")

