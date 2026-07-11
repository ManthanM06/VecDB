import requests
import random
import time
from concurrent.futures import ThreadPoolExecutor

# Config
BASE_URL = "http://localhost:8080"
DIMENSIONS = 128
NUM_VECTORS = 10000

def insert_vector(vector_id):
    # Generate a random 128-dimensional vector
    vector = [random.uniform(-1.0, 1.0) for _ in range(DIMENSIONS)]
    payload = {
        "id": vector_id,
        "vector": vector
    }
    try:
        response = requests.post(f"{BASE_URL}/insert", json=payload)
        return response.status_code == 200
    except Exception as e:
        return False

if __name__ == "__main__":
    print(f"Blasting {NUM_VECTORS} vectors to VecDB Server (50 concurrent threads)...")
    start_time = time.time()

    # Send concurrent HTTP POST requests
    successful_inserts = 0
    with ThreadPoolExecutor(max_workers=50) as executor:
        results = executor.map(insert_vector, range(NUM_VECTORS))
        successful_inserts = sum(results)

    end_time = time.time()
    print(f"Inserted {successful_inserts}/{NUM_VECTORS} vectors in {end_time - start_time:.2f} seconds.")

    print("\n Testing Search Endpoint...")
    query_vector = [random.uniform(-1.0, 1.0) for _ in range(DIMENSIONS)]
    
    search_start = time.time()
    search_response = requests.post(f"{BASE_URL}/search", json={"k": 5, "vector": query_vector})
    search_end = time.time()

    if search_response.status_code == 200:
        print(f"Search HTTP Latency: {(search_end - search_start) * 1000:.2f} ms")
        print("Top 5 Results:", search_response.json())
    else:
        print("Search failed:", search_response.text)

    print("\n Triggering Disk Save...")
    save_response = requests.post(f"{BASE_URL}/save")
    if save_response.status_code == 200:
        print("Database saved successfully to production_database.vec")
    else:
        print("Save failed:", save_response.text)