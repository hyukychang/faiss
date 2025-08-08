import faiss
import numpy as np
from collections import Counter


# Set dimension of the vectors
d = 128  # Example: 128-dimensional vectors

# Create some training data (1000 random vectors)
# nb = 100000
nb = 10000
np.random.seed(42)
xb = np.random.random((nb, d)).astype('float32')

# Create HNSW index
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)  # 32 = number of neighbors in the HNSW graph (M)
index.hnsw.efConstruction = 40      # Higher = more accurate, slower to build

# Add vectors to the index
index.add(xb)

# Set query parameters
index.hnsw.efSearch = 50  # Higher = more accurate search

# Perform a search
nq = 1  # number of queries
xq = np.random.random((nq, d)).astype('float32')
k = 4   # number of nearest neighbors

D, I = index.search(xq, k)  # D = distances, I = indices

# ----------------------
# Print HNSW Stats
# ----------------------

# Access the level of each node
levels = faiss.vector_to_array(index.hnsw.levels)
assert len(levels) == nb

max_level = max(levels)
level_hist = Counter(levels)

print("📊 HNSW Index Stats:")
print(f"- Total vectors: {nb}")
print(f"- Max layer (Lmax): {max_level}")
print(f"- Layer histogram (level -> count):")
for lvl in sorted(level_hist, reverse=True):
    print(f"  Level {lvl}: {level_hist[lvl]} vectors")

# Print results
for i in range(nq):
    print(f"Query {i}:")
    print("  Neighbors:", I[i])
    print("  Distances:", D[i])
