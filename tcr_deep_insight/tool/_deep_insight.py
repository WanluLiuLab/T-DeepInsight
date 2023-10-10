import torch
import scanpy as sc 
from sklearn.decomposition import PCA, KernelPCA

# Faiss
import faiss

from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
import datasets

# Built-in
import time
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.utils import class_weight
from collections import Counter
from itertools import chain
from copy import deepcopy
import json
from typing import Any, Callable, Mapping, Union, Iterable, Tuple, Optional, Mapping
import os
from pathlib import Path
import warnings
import umap
from transformers import (
    BertConfig,
    PreTrainedModel,
    BertForMaskedLM
)

# TCR Deep Insight

from ._deep_insight_result import TDIResult
from ..utils._compat import Literal
from ..model._model import VAEMixin, TRABModelMixin
from ..model._collator import TCRabCollatorForVJCDR3
from ..model._tokenizer import TCRabTokenizerForVJCDR3, TCRabTokenizerForCDR123
from ..model._trainer import TCRabTrainer
from ..model._model_utils import to_embedding_tcr_only_from_pandas_v2
from ..model._primitives import *
from ..model._config import get_config, get_human_config, get_mouse_config
from ..utils._compat import Literal
from ..utils._tensor_utils import one_hot, get_k_elements
from ..utils._decorators import typed
from ..utils._loss import LossFunction
from ..utils._logger import mt, get_tqdm
from ..utils._distributions import multivariate_gaussian_p_value
from ..utils._parallelizer import Parallelizer
from ..utils._utilities import random_subset_by_key, euclidean

from ..utils._definitions import (
    TRA_DEFINITION_ORIG,
    TRAB_DEFINITION_ORIG,
    TRB_DEFINITION_ORIG,
    TRA_DEFINITION,
    TRAB_DEFINITION,
    TRB_DEFINITION,
)
from ..utils._logger import mt
from ..utils._utilities import default_aggrf, default_pure_criteria
from ..utils._utilities import nearest_neighbor_eucliean_distances
from ..utils._decorators import deprecated
MODULE_PATH = Path(__file__).parent
warnings.filterwarnings("ignore")

@deprecated(
    ymd=(2023, 11, 1), 
    optional_message="Auto pretraining will be deprecated after 2023-11-1 and will be replaced by a unified pipeline"
)
@typed({
    "gex_adata": sc.AnnData,
    "max_epoch": int,
    "lr": float,
    "device": str,
})
def pretrain_gex_embedding(
    gex_adata: sc.AnnData, 
    batch_key: str = 'sample_name',
    max_epoch=16, 
    lr=1e-4, 
    device='cuda:0', 
    vae_model: Optional[VAEMixin] = None,
    state_dict_save_path: Optional[str] = None,
    highly_variable_genes_kwargs: Optional[Mapping] = None,
    vae_model_config: Optional[Mapping] = None,
    vae_model_train_kwargs: Optional[Mapping] = None
):
    """
    Pretrain GEX embedding using the adata as reference

    :param gex_adata: AnnData object containing GEX data
    :param batch_key: Batch key. Default: 'sample_name'
    :param max_epoch: Maximum number of epochs. Default: 16
    :param lr: Learning rate. Default: 1e-4
    :param device: Device to use. Default: 'cuda:0'
    :param vae_model: Pretrained VAE model. Default: None
    :param state_dict_save_path: Path to save the state dict. Default: None
    :param highly_variable_genes_kwargs: Keyword arguments for highly_variable_genes. Default: None
    :param vae_model_config: Configuration for VAE model. Default: None
    :param vae_model_train_kwargs: Keyword arguments for VAE model training. Default: None

    :return: Pretrained VAE model
    
    """
    mt("Building VAE model...")
    if highly_variable_genes_kwargs is None:
        sc.pp.highly_variable_genes(
            gex_adata, flavor='seurat_v3', batch_key=batch_key, inplace=False
        )
    else:
        if 'batch_key' in highly_variable_genes_kwargs:
            highly_variable_genes_kwargs.pop('batch_key')
        if 'inplace' in highly_variable_genes_kwargs:
            highly_variable_genes_kwargs.pop('inplace')
        if 'flavor' in highly_variable_genes_kwargs:
            highly_variable_genes_kwargs.pop('flavor')
        sc.pp.highly_variable_genes(
            gex_adata, 
            flavor='seurat_v3', 
            batch_key=batch_key, 
            inplace=False, 
            **highly_variable_genes_kwargs
        )
    if vae_model_config is None:
        vae_model = VAEMixin(
            adata=gex_adata[:,gex_adata.var['highly_variable']],
            batch_key=batch_key,
            device=device
        )
    else:
        if 'batch_key' in vae_model_config:
            vae_model_config.pop('batch_key')
        if 'device' in vae_model_config:
            vae_model_config.pop('device')
        if 'adata' in vae_model_config:
            vae_model_config.pop('adata')
        vae_model = VAEMixin(
            adata=gex_adata[:,gex_adata.var['highly_variable']],
            batch_key=batch_key,
            device=device,
            **vae_model_config
        )

    if vae_model_train_kwargs is None:
        vae_model.fit(
            max_epoch=max_epoch, 
            lr=lr
        )
    else:
        if 'max_epoch' in vae_model_train_kwargs:
            vae_model_train_kwargs.pop('max_epoch')
        if 'lr' in vae_model_train_kwargs:
            vae_model_train_kwargs.pop('lr')
        vae_model.fit(
            max_epoch=max_epoch, 
            lr=lr,
            **vae_model_train_kwargs
        )

    vae_model_state_dict = vae_model.state_dict()
    X_gex = vae_model.get_latent_embedding()
    gex_adata.obsm["X_gex"] = X_gex
    if state_dict_save_path is not None:
        torch.save(vae_model_state_dict, state_dict_save_path)
    else:
        state_dict_save_path = time.ctime().replace(" ","_").replace(":","_") + "_VAE_state_dict.ckpt"
        mt("saving VAE model weights to..." + state_dict_save_path)
        torch.save(vae_model_state_dict, state_dict_save_path)
    return vae_model

# @deprecated
@typed({
    "gex_adata": sc.AnnData,
    "gex_reference_adata": sc.AnnData,
})
def get_pretrained_gex_embedding(
    gex_adata: sc.AnnData,
    gex_reference_adata: sc.AnnData,
    transfer_label: Optional[str] = None,
    checkpoint_path: Optional[str] = os.path.join(MODULE_PATH, '../data/pretrained_weights/vae_gex_all_cd4_cd8.ckpt'),
    extra_train_epochs: Optional[int] = 0,
    reference_proportion: Optional[float] = 0,
    query_multiples: Optional[float] = 9.,
    device: str = 'cpu'
):
    """
    Get GEX embedding from pretrained VAE model
    
    .. note::
        This method modifies the `gex_adata` inplace.


    .. note::
        If `reference_proportion` is set to 0, `query_multiples` will be used to determine the number of cells to use as reference

    :param gex_adata: AnnData object containing GEX data
    :param gex_reference_adata: AnnData object containing reference GEX data
    :param transfer_label: Label to transfer. Default: None
    :param checkpoint_path: Path to pretrained VAE model. Default: None
    :param extra_train_epochs: Number of extra training epochs. Default: 0
    :param reference_proportion: Proportion of reference cells to use. Default: 0
    :param query_multiples: Number of query cells to use. Default: 9
    :param device: Device to use. Default: 'cpu'

    :return: The vae model

    """
    mt("Loading Reference adata...")
    if reference_proportion > 0:
        n = int(len(gex_reference_adata) * reference_proportion)
    elif query_multiples > 0:
        n = int(len(gex_adata) * query_multiples)
    else:
        raise ValueError()
    
    reference = random_subset_by_key(gex_reference_adata, 'sample_name', n)


    mt("Building VAE model...")
    model = VAEMixin(
        adata=reference,
        batch_key='sample_name',
        constrain_latent_embedding=True,
        device=device
    )
    mt("Loading VAE model checkpoints...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.transfer(gex_adata[:, reference.var.index], batch_key='sample_name', times_of_new=9)
    if extra_train_epochs > 0:
        mt("Training VAE model for extra {} epochs...".format(extra_train_epochs))
        model.fit(max_epoch=extra_train_epochs, lr=1e-4)

    X_gex = model.get_latent_embedding()
    
    gex_adata.obsm["X_gex"] = X_gex[-len(gex_adata):]
    if transfer_label is not None:
        mt("Transfering cell type labels...")
        nn = KNeighborsClassifier(n_neighbors=13)
        nn.fit(
            gex_reference_adata.obsm["X_gex"], 
            gex_reference_adata.obs[transfer_label]
        )
        gex_adata.obs[transfer_label] = nn.predict(gex_adata.obsm["X_gex"])

    return model

# @deprecated
def get_pretrained_gex_umap(gex_adata, gex_reference_adata, **kwargs):
    """
    Get UMAP embedding from pretrained VAE model

    .. note::
        This method modifies the `gex_adata` inplace.

    :param gex_adata: AnnData object containing GEX data
    :param gex_reference_adata: AnnData object containing reference GEX data
    :param kwargs: Keyword arguments to pass to UMAP

    """
    if gex_reference_adata is not None:
        z = umap.UMAP(**kwargs).fit_transform(np.vstack([
            gex_reference_adata.obsm["X_gex"],
            gex_adata.obsm["X_gex"]])
        )
        gex_adata.obsm["X_umap"] = z[-len(gex_adata):]
        gex_reference_adata.obsm["X_umap"] = z[:-len(gex_adata)]
    else:
        gex_adata.obsm["X_umap"] = umap.UMAP(**kwargs).fit_transform(gex_adata.obsm["X_gex"])

def pretrain_tcr_embedding(
        tcr_adata: sc.AnnData, 
        species: Literal['human','mouse'] = 'human',
        max_train_sequence=100000, 
        max_epoch=50,
        show_progress=True, 
        n_per_batch=256, 
        shuffle=True,
        encoding: Literal['vjcdr3','cdr123'] = 'vjcdr3'
    ):
    """
    Pretrain TCR embedding using the adata as reference

    :param tcr_adata: AnnData object containing TCR data
    :param species: Species. Default: 'human'. Options: 'human', 'mouse'
    :param max_train_sequence: Maximum number of training sequences. Default: 100000
    :param max_epoch: Maximum number of epochs. Default: 50
    :param show_progress: Whether to show progress. Default: True
    :param n_per_batch: Number of sequences per batch. Default: 256
    :param shuffle: Whether to shuffle. Default: True
    """
    if encoding == 'vjcdr3':
        tcr_tokenizer = TCRabTokenizerForVJCDR3(
            tra_max_length=48, 
            trb_max_length=48,
            species=species
        )
    elif encoding == 'cdr123':
        tcr_tokenizer = TCRabTokenizerForCDR123(
            tra_max_length=48, 
            trb_max_length=48,
            species=species
        )
    else:
        raise ValueError("Invalid encoding")
    
    tcr_dataset = tcr_tokenizer.to_dataset(
    ids=tcr_adata.obs.index,
        alpha_v_genes=list(tcr_adata.obs['TRAV']),
        alpha_j_genes=list(tcr_adata.obs['TRAJ']),
        beta_v_genes=list(tcr_adata.obs['TRBV']),
        beta_j_genes=list(tcr_adata.obs['TRBJ']),
        alpha_chains=list(tcr_adata.obs['CDR3a']),
        beta_chains=list(tcr_adata.obs['CDR3b']),
    )

    tcr_model = TRABModelMixin(
        get_human_config() if species == 'human' else get_mouse_config(),
        labels_number=1
    ).to("cuda:0")

    tcr_collator = TCRabCollatorForVJCDR3(48, 48, mlm_probability=0)

    tcr_model_optimizer = torch.optim.AdamW(tcr_model.parameters(), lr=5e-5)

    tcr_model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        tcr_model_optimizer, 
        mode='min', 
        factor=0.1, 
        patience=5
    )

    tcr_model_trainer = TCRabTrainer(
        tcr_model, 
        collator=tcr_collator, 
        train_dataset=tcr_dataset['train'], 
        test_dataset=tcr_dataset['test'], 
        optimizers=(tcr_model_optimizer, tcr_model_scheduler), 
        device='cuda:0'
    )

    tcr_model_trainer.fit(
        max_train_sequence=max_train_sequence, 
        max_epoch=max_epoch,
        show_progress=show_progress, 
        n_per_batch=n_per_batch, 
        shuffle=shuffle
    )

    return tcr_model


def get_pretrained_tcr_embedding(
    tcr_adata: sc.AnnData, 
    bert_config: Mapping[str, Any],
    encoding: Literal['vjcdr3','cdr123'] = 'vjcdr3',
    species: Literal['human','mouse'] = 'human',
    checkpoint_path=os.path.join(MODULE_PATH, '../data/pretrained_weights/bert_tcr_768.ckpt'),
    pca_path: Optional[str] = None, 
    use_pca:bool=True, 
    use_kernel_pca: bool = False,
    pca_n_components: int = 64,
    device='cuda:0',
    mask_tr: Literal['none','tra','trb'] = 'none',
    pooling: Literal["cls", "mean", "max", "pool", "trb", "tra", "weighted"] = "mean",
    pooling_weight: Tuple[float] = (0.1,0.9),
    n_per_batch: int = 256,
):
    """
    Get TCR embedding from pretrained BERT model

    .. note::
        This method modifies the `tcr_adata` inplace.

    :param tcr_adata: AnnData object containing TCR data
    :param bert_config: BERT config
    :param checkpoint_path: Path to pretrained BERT model. Default: None
    :param pca_path: Path to PCA model. Default: None
    :param use_pca: Whether to use PCA. Default: True

    """
    mt("Building BERT model")
    tcr_model = TRABModelMixin(
        bert_config,
        pooling_cls_position=1,
        labels_number=1,
        pooling=pooling,
        pooling_weight=pooling_weight
    )
    tcr_model.to(device)
    if encoding == 'vjcdr3':
        tcr_tokenizer = TCRabTokenizerForVJCDR3(
            tra_max_length=48, 
            trb_max_length=48,
            species=species
        )
    elif encoding == 'cdr123':
        tcr_tokenizer = TCRabTokenizerForCDR123(
            tra_max_length=60, 
            trb_max_length=60,
            species=species
        )
        tcr_model.pooling_cls_position = 0
    else:
        raise ValueError("Invalid encoding")
    
    mt("Loading BERT model checkpoints...")
    try:
        tcr_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except RuntimeError as e:
        mt("Failed to load the full pretrained BERT model. Please make sure the checkpoint path is correct.")

    
    mt("Computing TCR Embeddings...")
    all_embedding = to_embedding_tcr_only_from_pandas_v2(
        tcr_model,
        tcr_adata.obs,
        tcr_tokenizer,
        device,
        n_per_batch=n_per_batch,
        mask_tr=mask_tr
    )

    if use_pca and os.path.exists(pca_path):
        mt("Loading PCA model...")
        pca = load(pca_path)
        if use_kernel_pca:
            assert(isinstance(pca, KernelPCA), "PCA model is not a KernelPCA model")
        else:
            assert(isinstance(pca, PCA), "PCA model is not a PCA model")
            
        mt("Performing PCA...")
        all_embedding_pca = pca.transform(all_embedding)
    elif use_pca:
        if use_kernel_pca:
            pca = KernelPCA(
                n_components=pca_n_components, 
                kernel="rbf", 
                gamma=10, 
                fit_inverse_transform=True, 
                alpha=0.1
            )
            all_embedding_pca = np.array(pca.fit_transform(all_embedding))
        else:
            pca = PCA(n_components=pca_n_components).fit(all_embedding)
            all_embedding_pca = np.array(pca.transform(all_embedding))
        if pca_path is not None:
            mt("Saving PCA model...")
            dump(pca, pca_path)
    else:
        all_embedding_pca = all_embedding


    tcr_adata.obsm["X_tcr"] = all_embedding
    tcr_adata.obsm["X_tcr_pca"] = all_embedding_pca

def _prepare_tcr_embedding(
    tcr_adata: sc.AnnData,
    layer_norm: bool = True,
    use_gex: bool = True,
    _tcr_embedding_weight: float = 6.,
) -> Tuple[np.ndarray, int, int]:
    if layer_norm:
        # LayerNorm for TCR and GEX
        ln_1 = torch.nn.LayerNorm(tcr_adata.obsm["X_tcr_pca"].shape[1])
        X_tcr_pca = ln_1(torch.tensor(tcr_adata.obsm["X_tcr_pca"]).float()).detach().numpy()
        ln_2 = torch.nn.LayerNorm(tcr_adata.obsm["X_gex"].shape[1])
        X_gex = ln_2(torch.tensor(tcr_adata.obsm["X_gex"]).float()).detach().numpy()
        if not use_gex:
            all_tcr_gex_embedding = X_tcr_pca
        else:
            all_tcr_gex_embedding = np.hstack([
                X_tcr_pca,
                X_gex
            ])
    else:
        X_tcr_pca = tcr_adata.obsm["X_tcr_pca"]
        X_gex = tcr_adata.obsm["X_gex"]
        if not use_gex:
            all_tcr_gex_embedding = tcr_adata.obsm["X_tcr_pca"]
        else:
            all_tcr_gex_embedding = np.hstack([
                tcr_adata.obsm["X_tcr_pca"],
                _tcr_embedding_weight*tcr_adata.obsm["X_gex"]
            ])
    return all_tcr_gex_embedding, X_tcr_pca.shape[1], X_gex.shape[1]


def cluster_tcr(
    tcr_adata: sc.AnnData,
    label_key: str = None,  
    include_label_keys: Optional[Iterable[str]] = None,
    gpu=0,
    pure_label: bool = True,
    pure_criteria: Callable = default_pure_criteria,
    layer_norm: bool = True,
    max_distance: float = 25,
    max_cluster_size: int = 40,
    use_gex: bool = True,
    filter_intersection_fraction: float = 0.7,
    k: int = 3
):
    """
    Cluster TCRs by joint TCR-GEX embedding.

    :param tcr_adata: AnnData object containing TCR data
    :param label_key: Key of the label to cluster
    :param gpu: GPU ID. Default: 0
    :param layer_norm: Whether to use LayerNorm. Default: True
    :param max_distance: Maximum distance. Default: 25
    :param max_cluster_size: Maximum cluster size for dTCR clusters. Default: 40

    :return: sc.AnnData containing clustered TCRs.

    .. note::
        The `gpu` parameter indicates GPU to use for clustering. If `gpu` is 0, CPU is used.

    """
    all_tcr_gex_embedding, tcr_dim, gex_dim = _prepare_tcr_embedding(
        tcr_adata,
        layer_norm=layer_norm,
        use_gex=use_gex
    )
    result = _cluster_tcr_by_label_core(
        tcr_adata.obs, 
        all_tcr_gex_embedding=all_tcr_gex_embedding, 
        query_tcr_gex_embedding=all_tcr_gex_embedding, 
        tcr_dim=tcr_dim,
        gex_dim=gex_dim,
        label_key=label_key, 
        include_label_keys=include_label_keys,
        gpu=gpu,
        pure_label=pure_label,
        pure_criteria=pure_criteria,
        max_distance=max_distance,
        max_cluster_size=max_cluster_size,
        k=k,
        filter_intersection_fraction=filter_intersection_fraction,
    )
    return result

def cluster_tcr_from_reference(
    tcr_adata: sc.AnnData,
    tcr_reference_adata: sc.AnnData,
    label_key: str = None,  
    include_label_keys: Optional[Iterable[str]] = None,
    gpu=0,
    layer_norm: bool = True,
    pure_label: bool = True,
    pure_criteria: Callable = lambda x,y,z: Counter(x).most_common()[0][1] / len(x) > y and Counter(x).most_common()[0][0] == z,
    max_distance:float = 25.,
    max_cluster_size: int = 40,
    use_gex: bool = True,
) -> TDIResult:
    """
    Cluster TCRs from reference. 

    :param tcr_adata: AnnData object containing TCR data
    :param tcr_reference_adata: AnnData object containing reference TCR data
    :param label_key: Key of the label to use for clustering
    :param reference_path: Path to reference TCR data. Default: None
    :param gpu: GPU to use. Default: 0
    :param layer_norm: Whether to use LayerNorm. Default: True
    :param max_distance: Maximum distance of the most dissimilar TCR in a dTCR cluster . Default: 25
    :param max_cluster_size: Maximum cluster size for dTCR clusters. Default: 40
    :param use_gex: Whether to use GEX embedding. Default: True

    :return: TDIResult object containing clustered TCRs.

    .. note::
        The `gpu` parameter indicates GPU to use for clustering. If `gpu` is 0, CPU is used.
    
    """
    all_tcr_gex_embedding_reference, tcr_dim, gex_dim = _prepare_tcr_embedding(
        tcr_reference_adata,
        layer_norm=layer_norm,
        use_gex=use_gex
    )
    all_tcr_gex_embedding_query, tcr_dim, gex_dim = _prepare_tcr_embedding(
        tcr_adata,
        layer_norm=layer_norm,
        use_gex=use_gex
    )

    df = pd.concat([
        tcr_reference_adata.obs,
        tcr_adata.obs
    ])

    result = _cluster_tcr_by_label_core(
        df, 
        all_tcr_gex_embedding=all_tcr_gex_embedding_reference, 
        query_tcr_gex_embedding=all_tcr_gex_embedding_query, 
        tcr_dim=tcr_dim,
        gex_dim=gex_dim,
        label_key=label_key, 
        include_label_keys=include_label_keys,
        gpu=gpu,
        pure_label=pure_label,
        pure_criteria=pure_criteria,
        max_distance=max_distance,
        max_cluster_size=max_cluster_size,
    )
    return result

@typed({
    "df": pd.DataFrame,
    "all_tcr_gex_embedding": np.ndarray,
    "query_tcr_gex_embedding": np.ndarray,
    "tcr_dim": int,
    "gex_dim": int,
    "label_key": str,
    "gpu": int,
    "pure_label": bool,
    "pure_criteria": Callable,
    "calculate_tcr_gex_distance": bool,
    "max_distance": float,
    "max_cluster_size": int,
})
def _cluster_tcr_by_label_core(
    df, 
    *,
    all_tcr_gex_embedding, 
    query_tcr_gex_embedding, 
    tcr_dim, 
    gex_dim,
    label_key = None,
    include_label_keys: Optional[List[str]] = None,
    gpu = 0,
    pure_label: bool = True,
    pure_criteria: Callable = lambda x,z: Counter(x).most_common()[0][1] / len(x) > 0.7 and Counter(x).most_common()[0][0] == z,
    max_distance = 25,
    max_cluster_size = 40,
    calculate_tcr_gex_distance: bool = False,
    n_jobs: int = 1,
    k: int = 3,
    filter_intersection_fraction: float = 0.7
) -> TDIResult:
    kmeans = faiss.Kmeans(
        all_tcr_gex_embedding.shape[1],
        all_tcr_gex_embedding.shape[0],
        niter=20,
        verbose=True,
        gpu=gpu
    )
    kmeans.train(all_tcr_gex_embedding)
    D, I = kmeans.index.search(query_tcr_gex_embedding, max_cluster_size)

    CAT_STORAGE_SIZE = 10
    NUM_CAT_STORAGE = int(max_cluster_size / CAT_STORAGE_SIZE)

    _result = [[] for _ in range(NUM_CAT_STORAGE)]

    if label_key is not None:
        label_map = dict(zip(range(len(df)), df[label_key]))
    else:
        label_map = dict(zip(range(len(df)), ['undefined']*len(df)))
        mt("Warning: No label key is provided.")
        pure_label = False

    mt("Iterative select TCRs as clustering anchors")
    
    def FLATTEN(x): 
        return [i for s in x for i in s]
    def par_func(data, queue):
        ret = []
        for i in data:
            if queue is not None:
                queue.put(None)
            label = np.array([label_map[x] for x in I[i]])
            mp = np.argwhere(D[i] > max_distance)
            if len(mp) > 0:
                init_j = mp[0][0]
            else:
                init_j = max_cluster_size

            for j in list(range(2, max(2,init_j+1)))[::-1]:
                pure_criteria_pass = pure_criteria(label[:j], label[0])
                if (pure_criteria_pass or (not pure_label)):
                    d = label[0]
                    if pure_label:
                        pomc = list(filter(lambda x: label[x] == d, range(0, j)))
                        comp = list(filter(lambda x: label[x] != d, range(j, max_cluster_size)))[:j]
                    else: 
                        pomc = list(range(0, j))
                        comp = list(range(j, max_cluster_size))[:j]
                    cluster_size = len(pomc)

                    ret.append(((cluster_size-1) // CAT_STORAGE_SIZE, [
                            label[0]] + \
                            list( I[i][pomc]) + \
                                [-1] * ((CAT_STORAGE_SIZE * (((cluster_size-1) // CAT_STORAGE_SIZE)+ 1)) - len(pomc)
                            ) + \
                            [cluster_size] + \
                            [D[i][pomc].mean(), # mean distance for disease group
                            D[i][comp].mean(), # mean distance for non-disease group,
                            D[i][comp][:k].mean() - D[i][pomc][0], # distance between the first non-disease and the first disease
                            i
                    ]))
            return ret 
    
    if n_jobs > 1:
        p = Parallelizer(n_jobs=n_jobs)
        ret = p.parallelize(
            map_func=par_func, 
            map_data=list(range(I.shape[0])),
            reduce_func=FLATTEN,
            progress=True,
        )()
    else:
        ret = []
        pbar = get_tqdm()(total=I.shape[0], bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i in range(I.shape[0]):
            label = np.array([label_map[x] for x in I[i]])
            mp = np.argwhere(D[i] > max_distance)
            if len(mp) > 0:
                init_j = mp[0][0]
            else:
                init_j = max_cluster_size

            for j in list(range(2, max(2,init_j+1)))[::-1]:
                pure_criteria_pass = pure_criteria(label[:j], label[0])
                if (pure_criteria_pass or (not pure_label)):
                    d = label[0]
                    if pure_label:
                        pomc = list(filter(lambda x: label[x] == d, range(0, j)))
                        comp = list(filter(lambda x: label[x] != d, range(j, max_cluster_size)))[:j]
                    else: 
                        pomc = list(range(0, j))
                        comp = list(range(j, max_cluster_size))[:j]
                    cluster_size = len(pomc)

                    ret.append(((cluster_size-1) // CAT_STORAGE_SIZE, [
                            label[0]] + \
                            list( I[i][pomc]) + \
                                [-1] * ((CAT_STORAGE_SIZE * (((cluster_size-1) // CAT_STORAGE_SIZE)+ 1)) - len(pomc)
                            ) + \
                            [cluster_size] + \
                            [D[i][pomc].mean(), 
                            D[i][comp].mean(),
                            D[i][comp][:k].mean() - D[i][pomc][-1],
                            i
                    ]))
                    break

            pbar.update(1)
        pbar.close()

    for i in ret:
        _result[i[0]].append(i[1])

    all_result_tcr = []
    all_tcr_sequence = df['tcr'].to_numpy()

    for ii in range(len(_result)):
        
        result = pd.DataFrame(_result[ii])
        max_cluster_size_ = (ii+1) * CAT_STORAGE_SIZE

        if result.shape[0] == 0:
            continue
        # remove singletons clusters

        result = result[result.iloc[:,max_cluster_size_+1] > 1]
        result = result.sort_values(max_cluster_size_+1, ascending=False)
        if result.shape[0] == 0:
            continue


        result.index = list(range(len(result)))
        selected_indices = set()
        appeared_clusters = set()
        appeared_clusters_2 = {}
        indices_mapping = {}


        for i in range(len(result)):
            t = tuple(sorted(list(filter(lambda x: x >= 0, result.iloc[i,1:max_cluster_size_+1].to_numpy()))))

            if t in appeared_clusters:
                pbar.update(1)
                continue

            flag = False
            st = set(t)
            for j in t:
                if j in appeared_clusters_2.keys():
                    for k in appeared_clusters_2[j]:
                        sk = set(k)
                        # if st.issubset(sk):
                        if len(st.intersection(sk)) / len(st) > 0.9:
                            flag = True
                        elif (len(st.intersection(sk)) / len(st) > filter_intersection_fraction and \
                              len(st.intersection(sk)) / len(sk) > filter_intersection_fraction and \
                              k in appeared_clusters
                            ):
                            if (result.iloc[indices_mapping[k], max_cluster_size_+2] > result.iloc[i,max_cluster_size_+2] and \
                                len(st) >= len(sk)
                            ):
                                # this cluster is a subset of a already selected cluster
                                # and the selected cluster has a higher mean distance
                                # remove the selected cluster
                                appeared_clusters.remove(k)
                                if indices_mapping[k] in selected_indices:
                                    selected_indices.remove(indices_mapping[k])
                                for j in k:
                                    if k in appeared_clusters_2[j]:
                                        appeared_clusters_2[j].remove(k)
                                break
                            else:
                                # this cluster is a subset of a already selected cluster, skip
                                flag = True
                                break

            if flag:
                pbar.update(1)
                continue

            appeared_clusters.add(t)
            for j in t:
                if j not in appeared_clusters_2:
                    appeared_clusters_2[j] = [t]
                else:
                    appeared_clusters_2[j].append(t)

            selected_indices.add(i)
            indices_mapping[t] = i

        result = result.iloc[list(selected_indices)]

        if include_label_keys is not None and label_key is not None:
            result = result[result.iloc[:,0].isin(include_label_keys)]

        result_tcr = result.copy()
        result_cell_number = result.copy()

        for i in list(range(1, max_cluster_size_+1))[::-1]:
            result_tcr.iloc[:,i] = result_tcr.iloc[:,i].apply(
                lambda x: all_tcr_sequence[x] if x >= 0 else '-'
            )

            
        result_tcr['number_of_individuals'] =list(
            map(
                lambda x: len(
                    np.unique(
                        list(
                            map(
                                lambda z: z.split("=")[-1],
                                filter(lambda x: x != "-", x),
                            )
                        )
                    )
                ),
                result_tcr.iloc[:,1:max_cluster_size_+1].to_numpy(),
            )
        )

        result_tcr['number_of_unique_tcrs'] = list(
            map(
                lambda x: len(
                    np.unique(
                        list(
                            map(
                                lambda z: "=".join(z.split("=")[:-1]),
                                filter(lambda x: x != "-", x),
                            )
                        )
                    )
                ),
                result_tcr.iloc[:, 1:max_cluster_size_+1].to_numpy(),
            )
        )
        
        
        if 'number_of_cells' in df.columns:
            all_number_of_cell = df['number_of_cells'].to_numpy()
            for i in list(range(1, max_cluster_size_+1))[::-1]:
                result_cell_number.iloc[:,i] = result_cell_number.iloc[:,i].apply(
                    lambda x: all_number_of_cell[x] if x >= 0 else '-'
                )
            result_tcr['number_of_cells'] = result_cell_number.iloc[:,1:max_cluster_size_+1].sum(axis=1)

        

        result_tcr.columns = (
            [label_key]
            + [f"TCRab{x}" for x in range(1, result_tcr.shape[1]-8)]
            + [
                "number_of_tcrs",
                "mean_distance",
                "mean_distance_other",
                "distance_difference",
                "cluster_index",
                "number_of_individuals",
                "number_of_unique_tcrs"
            ] + (["number_of_cells"] if 'number_of_cells' in df.columns else [])
        )

        # result_tcr = result_tcr[result_tcr['number_of_unique_tcrs'] > 1]
        # result_tcr = result_tcr[result_tcr['mean_distance'] > 1e-3]

        result_tcr['disease_association_score'] = result_tcr['distance_difference']
        result_tcr['tcr_similarity_score'] = 100 - result_tcr['mean_distance']

        if calculate_tcr_gex_distance:
            D_gex = nearest_neighbor_eucliean_distances(
                query_tcr_gex_embedding[:,tcr_dim:],
                I,
                result_tcr['cluster_index']
            )
            D_tcr = nearest_neighbor_eucliean_distances(
                query_tcr_gex_embedding[:,:tcr_dim],
                I,
                result_tcr['cluster_index']
            )

            result_tcr['mean_distance_gex'] = D_gex.mean(axis=1)
            result_tcr['mean_distance_tcr'] = D_tcr.mean(axis=1)

        result_tcr = result_tcr.drop(columns=['mean_distance','mean_distance_other','distance_difference'])
        if not pure_label:
            result_tcr = result_tcr.drop(columns=[label_key, 'disease_association_score'])
        tcrs = list(map(','.join, result_tcr.loc[:,[f"TCRab{x}" for x in range(1, max_cluster_size_+1)]].to_numpy()))
        result_tcr = result_tcr.loc[:,list(filter(lambda x: not x.startswith("TCRab"), result_tcr.columns))]
        result_tcr.insert(0, "TCRab", tcrs)
        all_result_tcr.append(result_tcr)

    result_tcr = pd.concat(all_result_tcr)
    return TDIResult(
        sc.AnnData(
            obs=result_tcr,
            uns={
                "I": I, 
                "D": D, 
                "max_cluster_size":max_cluster_size,
                "max_distance":max_distance,
            }
        ),
        _cluster_label = label_key
    )

@typed({
    "reference_adata": sc.AnnData,
    "tcr_cluster_adata": sc.AnnData,
    "label_key": str,
    "map_function": Callable
})
def inject_labels_for_tcr_cluster_adata(
    reference_adata: sc.AnnData, 
    tcr_cluster_adata: sc.AnnData, 
    label_key: str, 
    map_function: Callable = default_aggrf
):
    """
    Inject labels for tcr_cluster_adata based on reference_adata
    :param reference_adata: sc.AnnData. Reference AnnData object containing labels
    :param tcr_cluster_adata: sc.AnnData
    :param label_key: str. Key of the label to use for clustering in reference_adata.obs.columns
    :param map_function: Callable. Default: function that returns the most frequent label otherwise "Ambigious"
    :return: sc.AnnData


    """
    # Get a list of lists of TCRs
    if 'tcr' not in reference_adata.obs.columns:
        raise ValueError("tcr column not found in reference_adata.obs. Please run `tdi.pp.update_anndata` first.")
    
    tcr_list = tcr_cluster_adata.obs.loc[
        :,
        list(
            filter(
                lambda x: x.startswith("TCRab"), 
                tcr_cluster_adata.obs.columns
            )
        )
    ].to_numpy()

    if "tcr" not in reference_adata.obs.columns:
        raise ValueError("tcr column not found in reference_adata.obs")
    
    # For each list of TCRs, find the label of the most abundant TCR
    labels = []
    tcr2int = dict(zip(reference_adata.obs['tcr'], range(len(reference_adata.obs['tcr']))))
    for tcrs in tcr_list:
        tcrs = list(filter(lambda x: x != '-', tcrs))
        if len(tcrs) == 0:
            labels.append("NA")
        else:
            # tcrs contains a list of TCRs
            # reference_adata.obs.loc[] finds the rows of reference_adata.obs where the TCRs are listed
            # reference_adata.obs.loc[... , label_key] gets the label_key column for those rows
            # map_function gets the aggregate function
            # map_function(reference_adata.obs.loc[... , label_key]) calls the aggregate function on the label_key column
            # labels.append(...) stores the result in the labels list
            labels.append(map_function(reference_adata.obs.iloc[
                list(map(lambda x: tcr2int[x], tcrs)),
            ][label_key])) 

    tcr_cluster_adata.obs[label_key] = labels

@typed({
    "tcr_adata": sc.AnnData,
    "tcr_reference_adata": sc.AnnData,
    "gpu": int,
    "max_tcr_distance": float,
    "layer_norm": bool
})
def query_tcr_without_gex(
    tcr_query_adata: sc.AnnData,
    tcr_reference_adata: sc.AnnData,
    gpu=0,
    max_tcr_distance: float = 20.,
    layer_norm: bool = True,
):
    if layer_norm:
        ln_1 = torch.nn.LayerNorm(tcr_reference_adata.obsm["X_tcr_pca"].shape[1])
        X_tcr_pca_reference = ln_1(torch.tensor(tcr_reference_adata.obsm["X_tcr_pca"], dtype=torch.float32)).detach().cpu().numpy()
        X_tcr_pca_query = ln_1(torch.tensor(tcr_query_adata.obsm["X_tcr_pca"], dtype=torch.float32)).detach().cpu().numpy()
    else:
        X_tcr_pca_reference = tcr_reference_adata.obsm["X_tcr_pca"]
        X_tcr_pca_query = tcr_query_adata.obsm["X_tcr_pca"]

    mt("computing pairwise disease of TCRs...")
    kmeans = faiss.Kmeans(
        X_tcr_pca_query.shape[1],
        X_tcr_pca_query.shape[0],
        niter=20,
        verbose=True,
        gpu=gpu
    )
    # ignore faiss warnings
    kmeans.cp.min_points_per_centroid = 1
    kmeans.cp.max_points_per_centroid = 1000000000
    kmeans.train(X_tcr_pca_reference.astype(np.float32))

    D, I = kmeans.index.search(X_tcr_pca_query.astype(np.float32), 40)

    I[D > max_tcr_distance] = -1
    tcr_col_index = tcr_reference_adata.obs.columns.get_loc("tcr")
    result = []
    
    for i in I:
        result.append(
            json.dumps(
                list(
                    map(
                        lambda z: tcr_reference_adata.obs.iloc[z, tcr_col_index],
                        filter(lambda x: x != -1, i),
                    )
                )
            )
        )
    tcr_query_adata.obs["tcr_query"] = result
