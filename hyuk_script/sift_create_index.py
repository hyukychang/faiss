import faiss
from faiss.contrib.datasets import DatasetSIFT1M

ds = DatasetSIFT1M()

databse =ds.get_database()
dim = databse.shape[1]  # 128 for SIFT1M
print(f"Database shape: {databse.shape}, Dimension: {dim}")

M = 32
distance_metric = faiss.METRIC_INNER_PRODUCT  # L2 distance for SIFT1M
index = faiss.IndexHNSWFlat(dim, M, distance_metric)  # 32 = M (neighbors per node)
index.hnsw.efConstruction = 40

print("Adding vectors to HNSW index...")
index.add(databse)  # 1 million base vectors
print("Vectors added to index.")
print("saving index...")
# Optional: Save index to disk
faiss.write_index(index, "sift1m_hnsw_ip.index")


