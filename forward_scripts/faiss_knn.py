import numpy as np
import faiss


class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.gpu_index_ivf = None
        self.y = None
        self.k = k
        self.res = faiss.StandardGpuResources()

    def fit(self, X, y):
        nlist = 100
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index_ivf = faiss.IndexIVFFlat(self.index, X.shape[1], nlist, faiss.METRIC_L2)
        self.gpu_index_ivf = faiss.index_cpu_to_gpu(self.res, 0, self.index_ivf)
        assert not self.gpu_index_ivf.is_trained
        self.gpu_index_ivf.train(X.astype(np.float32))
        assert self.gpu_index_ivf.is_trained
        self.gpu_index_ivf.add(X.astype(np.float32))
        # print(f"Trained:{self.index.is_trained}")
        self.y = y

    def predict(self, X):
        distances, indices = self.gpu_index_ivf.search(X.astype(np.float32), k=self.k)
        # print(distances)
        # print(indices)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions