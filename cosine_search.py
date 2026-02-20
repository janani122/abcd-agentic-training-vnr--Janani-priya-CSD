import numpy as np

def normalize(v):
    return v / np.linalg.norm(v)

def cosine_search(query, vectors, top_k=5):
    vectors = np.array([normalize(v) for v in vectors])
    query = normalize(np.array(query))

    scores = vectors @ query
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(i, scores[i]) for i in top_indices]


# example usage
data = [
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 1],
    [-1, 0, -1]
]

query = [1, 0, 1]
print(cosine_search(query, data))