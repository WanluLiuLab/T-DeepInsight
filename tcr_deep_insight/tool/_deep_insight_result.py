import scanpy as sc 

class TCRDeepInsightClusterResult:
    def __init__(
        self,
        _data: sc.AnnData,
        _cluster_labels: str = None,
    ):
        self._data = _data 
        self._cluster_labels = _cluster_labels

    @property
    def data(self):
        return self._data
    
    @property
    def cluster_labels(self):
        return self._cluster_labels

    