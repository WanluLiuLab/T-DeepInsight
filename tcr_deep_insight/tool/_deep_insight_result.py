import os
from typing import Dict, List, Optional
import pandas as pd
import scanpy as sc 
import numpy as np 

from ..utils._compat import Literal
from ..utils._logger import Colors
from ..utils._utilities import FLATTEN
from ..utils._definitions import TRAB_DEFINITION

class TCRDeepInsightClusterResult:
    def __init__(
        self,
        _data: sc.AnnData,
        _tcr_adata: sc.AnnData = None,
        _gex_adata: Optional[sc.AnnData] = None,
        _cluster_label: Optional[str] = None,
    ):
        self._data = _data 
        self._tcr_adata = _tcr_adata
        self._gex_adata = _gex_adata
        self._cluster_label = _cluster_label

    @property
    def data(self):
        return self._data
    
    @property
    def cluster_label(self):
        return self._cluster_label

    @cluster_label.setter
    def cluster_label(self, cluster_label: str):
        assert(cluster_label in self._data.obs.columns)
        self._cluster_label = cluster_label

    @property
    def tcr_adata(self):
        return self._tcr_adata

    @tcr_adata.setter
    def tcr_adata(self, tcr_adata: sc.AnnData):
        self._tcr_adata = tcr_adata

    @property
    def gex_adata(self):
        return self._gex_adata
    
    @gex_adata.setter
    def gex_adata(self, gex_adata: sc.AnnData):
        self._gex_adata = gex_adata

    def __repr__(self) -> str:
        return  f'{Colors.GREEN}TCRDeepInsightClusterResult{Colors.NC} object containing {Colors.CYAN}{self.data.shape[0]}{Colors.NC} clusters'
    

    def save_to_disk(self, save_path, save_tcr_data=True, save_gex_data=True):
        """
        Save the cluster result to disk

        :param save_path: the path to save the cluster result
        :param save_tcr_data: whether to save the tcr data
        :param save_gex_data: whether to save the gex data
        """
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self._data.write_h5ad(os.path.join(save_path, 'cluster_data.h5ad'))
        if save_tcr_data:
            self._tcr_adata.write_h5ad(os.path.join(save_path, 'tcr_data.h5ad'))
        if save_gex_data:
            self._gex_adata.write_h5ad(os.path.join(save_path, 'gex_data.h5ad'))
        if self._cluster_label is not None:
            with open(os.path.join(save_path, 'cluster_label.txt'), 'w') as f:
                f.write(self._cluster_label)

    @classmethod
    def load_from_disk(
        cls, 
        save_path: str, 
        tcr_data_path: Optional[str] = None,
        gex_adata_path: Optional[str] = None
    ):
        """
        Load the cluster result from disk

        :param save_path: the path to load the cluster result
        """
        data = sc.read_h5ad(os.path.join(save_path, 'cluster_data.h5ad'))
        if tcr_data_path is not None:
            print('Loading tcr data from {}'.format(tcr_data_path))
            tcr_adata = sc.read_h5ad(tcr_data_path)
        else: 
            if os.path.exists(os.path.join(save_path, 'tcr_data.h5ad')):
                tcr_adata = sc.read_h5ad(os.path.join(save_path, 'tcr_data.h5ad'))
            else:
                tcr_adata = None

        if gex_adata_path is not None:
            print('Loading gex data from {}'.format(gex_adata_path))
            gex_adata = sc.read_h5ad(gex_adata_path)
        else:
            if os.path.exists(os.path.join(save_path, 'gex_data.h5ad')):
                gex_adata = sc.read_h5ad(os.path.join(save_path, 'gex_data.h5ad'))
            else:
                gex_adata = None

        cluster_label = None
        if os.path.exists(os.path.join(save_path, 'cluster_label.txt')):
            with open(os.path.join(save_path, 'cluster_label.txt'), 'r') as f:
                cluster_label = f.read()
        return cls(data, tcr_adata, gex_adata, cluster_label)

    def get_tcrs_for_cluster(
        self,
        label: str,
        rank: int = 0, 
        rank_by: Literal["tcr_similarity", "disease_specificity"] = 'tcr_similarity',
        min_tcr_number: int = 4,
        min_individual_number: int = 2,
        min_tcr_similarity_score: Optional[float] = None,
        min_disease_specificity_score: Optional[float] = None,
        return_background_tcrs: bool = False,
        additional_label_key_values: Optional[Dict[str, List[str]]] = None
    ):
        """
        Get the tcrs for a specific cluster

        :param label: the cluster label
        :param rank: the rank of the tcrs to return
        :param rank_by: the metric to rank the tcrs
        :param min_tcr_number: the minimum number of unique tcrs in the cluster
        :param min_individual_number: the minimum number of individuals in the cluster
        :param min_tcr_similarity_score: the minimum tcr similarity score
        :param min_disease_specificity_score: the minimum disease specificity score
        :param return_background_tcrs: whether to return other tcrs in the cluster
        :param additional_label_key_values: additional label key values to filter the cluster

        :return: a dictionary containing the tcrs and their metadata

        """
        return self._get_tcrs_for_cluster(
            label, 
            rank_by,
            rank, 
            min_tcr_number=min_tcr_number, 
            min_individual_number=min_individual_number,
            min_tcr_similarity_score = min_tcr_similarity_score,
            min_disease_specificity_score = min_disease_specificity_score,
            return_background_tcrs=return_background_tcrs,                
            additional_label_key_values=additional_label_key_values
        )
    
    def to_pandas_dataframe(self):
        ret = []
        cluster_labels = []
        for i in range(len(self.data)):
            ret.append(
                list(map(lambda x: x.split("="), filter(lambda x: x != '-', self.data.obs.iloc[i].loc[
                    list(filter(lambda z: 'TCRab' in z, self.data.obs.columns))
                ].to_numpy())))
            )
            cluster_labels.append([self.data.obs.iloc[i].loc["cluster_index"]] * len(ret[-1]))
        df = pd.DataFrame(FLATTEN(ret), columns = TRAB_DEFINITION + ['individual'])
        df['cluster_index'] = FLATTEN(cluster_labels)
        return df


    def _get_tcrs_for_cluster(
        self, 
        label: str, 
        rank_by: Literal["tcr_similarity", "disease_specificity"] = 'tcr_similarity',
        rank: int = 0,
        min_tcr_number: int = 4,
        min_individual_number: int = 2,
        min_tcr_similarity_score: Optional[float] = None,
        min_disease_specificity_score: Optional[float] = None,
        return_background_tcrs: bool = False,
        additional_label_key_values: Optional[Dict[str, List[str]]] = None
    ):
        if rank_by not in ['tcr_similarity', 'disease_specificity']:
            raise ValueError('rank_by must be one of ["tcr_similarity", "disease_specificity"], got {}'.format(rank_by))
        
        min_tcr_similarity_score = min_tcr_similarity_score if min_tcr_similarity_score is not None else -1
        min_disease_specificity_score = min_disease_specificity_score if min_disease_specificity_score is not None else -1
        if additional_label_key_values is not None:
            additional_indices = np.bitwise_and.reduce(
                [
                    np.array(self.data.obs[key] == value) for key, value in additional_label_key_values.items()
                ]
            )
        else:
            additional_indices = np.ones(len(self.data.obs), dtype=bool)
            
        result_tcr = self.data.obs[
            np.array(self.data.obs[self.cluster_label] == label) & 
            np.array(self.data.obs['count'] >= min_tcr_number) &
            np.array(self.data.obs['number_of_individuals'] >= min_individual_number) &
            np.array(self.data.obs['tcr_similarity_score'] >= min_tcr_similarity_score) &
            np.array(self.data.obs['disease_specificity_score'] >= min_disease_specificity_score) &
            additional_indices
        ]

        if rank_by == 'tcr_similarity':
            result = result_tcr.sort_values(
                'tcr_similarity_score', 
                ascending=False
            )
        elif rank_by == 'disease_specificity':
            result = result_tcr.sort_values(
                'disease_specificity_score', 
                ascending=False
            )

        tcrs = list(filter(lambda x: x != '-', result.iloc[rank,1:41])) 
        cdr3a, cdr3b, trav, traj, trbv, trbj, individual = list(map(
            list, 
            np.array(list(map(lambda x: x.split("="), tcrs))).T
        ))

        if return_background_tcrs:
            return {
                'cluster_index': result['cluster_index'][rank],
                'tcrs': tcrs,
                'cdr3a': cdr3a,
                'cdr3b': cdr3b,
                'trav': trav,
                'traj': traj,
                'trbv': trbv,
                'trbj': trbj,
                'individual': individual
            }, self.get_tcrs_by_cluster_index(result['cluster_index'][rank], len(tcrs))
        else:
            return {
                'cluster_index': result['cluster_index'][rank],
                'tcrs': tcrs,
                'cdr3a': cdr3a,
                'cdr3b': cdr3b,
                'trav': trav,
                'traj': traj,
                'trbv': trbv,
                'trbj': trbj,
                'individual': individual
            }
    
    def get_tcrs_gex_embedding_coordinates(self, use_rep: str = 'X_umap') -> Dict[str, np.ndarray]:
        """
        Get the gex embedding coordinates for each tcr

        :param use_rep: the representation to use. Default: 'X_umap'

        :return: a dictionary containing the gex embedding coordinates for each tcr
        """
        result_tcr = self.data.obs
        coordinates = {}
        import tqdm
        tcrs2int = dict(zip(self.gex_adata.obs['tcr'], range(len(self.gex_adata))))
        for i in tqdm.trange(len(result_tcr)):
            tcrs = list(result_tcr.iloc[i,1:41])
            coordinates[result_tcr.index[i]] = self.gex_adata.obsm[use_rep][
                list(map(tcrs2int.get, list(filter(lambda x: x != '-', tcrs))))
            ]
        return coordinates

            
    def get_tcrs_by_cluster_index(
        self, 
        cluster_index: int,
        _n_after: int = 0
    ) -> List[str]:
        """
        Get the tcrs for a specific cluster

        :param cluster_index: the cluster index
        :param _n_after: the number of tcrs to skip
        """
        _result = self.tcr_adata.obs.iloc[
            self.data.uns['I'][int(cluster_index)]
        ]
        cdr3a = list(_result['CDR3a'])[_n_after:]
        cdr3b = list(_result['CDR3b'])[_n_after:]
        trav = list(_result['TRAV'])[_n_after:]
        traj = list(_result['TRAJ'])[_n_after:]
        trbv = list(_result['TRBV'])[_n_after:]
        trbj = list(_result['TRBJ'])[_n_after:]
        tcrs = list(_result['tcr'])[_n_after:]
        labels = None
        if self.cluster_label:
            labels = list(_result[self.cluster_label])[_n_after:]

        individual = list(_result['individual'])[_n_after:]

        return {
            'cluster_index': cluster_index,
            'tcrs': tcrs,
            'cdr3a': cdr3a,
            'cdr3b': cdr3b,
            'trav': trav,
            'traj': traj,
            'trbv': trbv,
            'trbj': trbj,
            'individual': individual,
            'labels': labels
        }
