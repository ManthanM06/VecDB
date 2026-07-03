import numpy as np
from sklearn.neighbors import NearestNeighbors
import json
import os

def generate_ground_truth(num_vectors = 10000, num_queries = 100, dim = 128, k = 10):
    # use float32 as float in c++ are 32 bits
    base_vectors = np.random.rand(num_vectors, dim).astype(np.float32)
    query_vectors = np.random.rand(num_queries, dim).astype(np.float32)

    # L2 normalization (for cosine similarity)
    # divide each vector with its magnitude 
    base_vectors = base_vectors / np.linalg.norm(base_vectors, axis=1,keepdims=True)
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1,keepdims=True)

    print("Calculating exact nearest neighbouts")
    knn = NearestNeighbors(n_neighbors=k , algorithm="brute", metric='cosine')
    knn.fit(base_vectors)

    distances,indices  = knn.kneighbors(query_vectors)

    print("formating data to export...")
    dataset = {
        "base_vectors" : base_vectors.tolist(),
        "queries": []
    }

    for i in range(num_queries):
        dataset["queries"].append({
            "query_vector": query_vectors[i].tolist(),
            "expected_indices": indices[i].tolist(),
            "expected_distances": distances[i].tolist(),
        })
    

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    output_dir = os.path.join(project_root, "tests", "test_data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "ground_truth.json")

    print(f"saving to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(dataset, f)

    print("Done")

if __name__ == "__main__":
    generate_ground_truth()
