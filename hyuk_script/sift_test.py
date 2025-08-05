from collections import Counter
import faiss


# import the existing SIFT1M index
# index_name = "sift1m_hnsw_ip.index"
index_name = "sift1m_hnsw.index"
print(f"Loading index from {index_name}...")
index: faiss.HNSW = faiss.read_index(index_name)
# print the index stats
print("Index stats:")
# Access the level of each node
levels = faiss.vector_to_array(index.hnsw.levels)
assert len(levels) == 1_000_000  # SIFT1M has 1 million vectors

max_level = max(levels)
level_hist = Counter(levels)

print("📊 HNSW Index Stats:")
print(f"- Total vectors: {len(levels)}")
print(f"- Max layer (Lmax): {max_level}")
print(f"- Layer histogram (level -> count):")
for lvl in sorted(level_hist, reverse=True):
    print(f"  Level {lvl}: {level_hist[lvl]} vectors")

# Perform a search
from faiss.contrib.datasets import DatasetSIFT1M

ds = DatasetSIFT1M()
query = ds.get_queries()
num_query = query.shape[0]  # number of queries
ks = [1,5,10,50,100]

ground_truth = ds.get_groundtruth()

recall = {k: 0 for k in ks}

for k in ks:
    print(f"Search for k={k}:")
    D, I = index.search(query, k)  # D = distances, I = indices
    for i in range(num_query):
        # Calculate recall
        correct = set(ground_truth[i][:k])
        retrieved = set(I[i])
        recall[k] += len(correct.intersection(retrieved)) / len(correct)
    recall[k] /= num_query
    print("-" * 40)

# print recall with table format
print("Recall results:")
print(f"{'k':<5} | {'Recall':<10}")
print("-" * 40)
for k in ks:
    print(f"{k:<5} | {recall[k]:<10.4f}")
print("-" * 40)
print("Done.")