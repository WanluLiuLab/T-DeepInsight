import torch
import scanpy as sc 
from sklearn.decomposition import PCA

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
import tqdm
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

from ._deep_insight_result import TCRDeepInsightClusterResult
from ..utils._compat import Literal
from ..model._model import VAEMixin, TRABModelMixin
from ..model._collator import TCRabCollator
from ..model._tokenizer import TCRabTokenizer
from ..model._trainer import TCRabTrainer
from ..model._model_utils import to_embedding_tcr_only_from_pandas_v2
from ..model._primitives import *
from ..model._config import get_config, get_human_config, get_mouse_config

from ..utils._tensor_utils import one_hot, get_k_elements
from ..utils._decorators import typed
from ..utils._loss import LossFunction
from ..utils._logger import mt
from ..utils._distributions import multivariate_gaussian_p_value

from ..utils._utilities import random_subset_by_key, euclidean
from ..utils._compat import Literal
from ..utils._definitions import (
    TRA_DEFINITION_ORIG,
    TRAB_DEFINITION_ORIG,
    TRB_DEFINITION_ORIG,
    TRA_DEFINITION,
    TRAB_DEFINITION,
    TRB_DEFINITION,
)
from ..utils._logger import mt
from ..utils._utilities import default_aggrf


MODULE_PATH = Path(__file__).parent
warnings.filterwarnings("ignore")


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
        shuffle=True
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
    tcr_tokenizer = TCRabTokenizer(
        tra_max_length=48, 
        trb_max_length=48,
        species=species
    )
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

    tcr_collator = TCRabCollator(48, 48, mlm_probability=0)

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
    checkpoint_path=os.path.join(MODULE_PATH, '../data/pretrained_weights/bert_tcr_768.ckpt'),
    pca_path: Optional[str] = None, 
    use_pca:bool=True, 
    device='cuda:0'
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
        pooling='trb',
        pooling_cls_position=1,
        labels_number=1
    )
    tcr_model.to(device)
    tokenizer = TCRabTokenizer(
        tra_max_length=48,
        trb_max_length=48,
    )
    mt("Loading BERT model checkpoints...")
    tcr_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    mt("Computing TCR Embeddings...")
    all_embedding = to_embedding_tcr_only_from_pandas_v2(
        tcr_model,
        tcr_adata.obs,
        tokenizer,
        device,
        mask_tr='tra'
    )

    if use_pca and os.path.exists(pca_path):
        mt("Loading PCA model...")
        pca = load(pca_path)
        mt("Performing PCA...")
        all_embedding_pca = np.array(pca.transform(all_embedding))
    elif use_pca:
        pca = PCA(n_components=64).fit(all_embedding)
        all_embedding_pca = np.array(pca.transform(all_embedding))
        if pca_path is not None:
            mt("Saving PCA model...")
            dump(pca, pca_path)
    else:
        all_embedding_pca = all_embedding


    tcr_adata.obsm["X_tcr"] = all_embedding
    tcr_adata.obsm["X_tcr_pca"] = all_embedding_pca

def cluster_tcr(
    tcr_adata: sc.AnnData,
    label_key: str,  
    gpu=0,
    layer_norm: bool = True,
    max_distance: float = 25,
    max_cluster_size: int = 40,
    no_gex: bool = False
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
    if layer_norm:
        # LayerNorm for TCR and GEX
        ln_1 = torch.nn.LayerNorm(tcr_adata.obsm["X_tcr_pca"].shape[1])
        X_tcr_pca = ln_1(torch.tensor(tcr_adata.obsm["X_tcr_pca"]).float()).detach().numpy()
        ln_2 = torch.nn.LayerNorm(tcr_adata.obsm["X_gex"].shape[1])
        X_gex = ln_2(torch.tensor(tcr_adata.obsm["X_gex"]).float()).detach().numpy()
        if no_gex:
            all_tcr_gex_embedding = X_tcr_pca
        else:
            all_tcr_gex_embedding = np.hstack([
                X_tcr_pca,
                X_gex
            ])
    else:
        if no_gex:
            all_tcr_gex_embedding = tcr_adata.obsm["X_tcr_pca"]
        else:
            all_tcr_gex_embedding = np.hstack([
                tcr_adata.obsm["X_tcr_pca"],
                6*tcr_adata.obsm["X_gex"]
            ])
    result = _cluster_tcr_by_label(
        tcr_adata.obs, 
        all_tcr_gex_embedding, 
        all_tcr_gex_embedding, 
        label_key, 
        gpu,
        max_distance,
        max_cluster_size
    )
    return result

@typed({
    "tcr_adata": sc.AnnData,
    "tcr_reference_adata": sc.AnnData,
    "label_key": str,
    "gpu": int,
    "layer_norm": bool,
    "max_distance": float,
    "max_cluster_size": int
})
def cluster_tcr_from_reference(
    tcr_adata: sc.AnnData,
    tcr_reference_adata: sc.AnnData,
    label_key: str,  
    gpu=0,
    layer_norm: bool = True,
    max_distance:float = 25.,
    max_cluster_size: int = 40,
    no_gex: bool = False
) -> TCRDeepInsightClusterResult:
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
    :param no_gex: Whether to use GEX embedding. Default: False

    :return: TCRDeepInsightClusterResult object containing clustered TCRs.

    .. note::
        The `gpu` parameter indicates GPU to use for clustering. If `gpu` is 0, CPU is used.
    
    """
    if layer_norm:
        # LayerNorm for TCR and GEX
        ln_1 = torch.nn.LayerNorm(tcr_reference_adata.obsm["X_tcr_pca"].shape[1])
        X_tcr_pca = ln_1(torch.tensor(tcr_reference_adata.obsm["X_tcr_pca"])).detach().numpy()
        ln_2 = torch.nn.LayerNorm(tcr_reference_adata.obsm["X_gex"].shape[1])
        X_gex = ln_2(torch.tensor(tcr_reference_adata.obsm["X_gex"])).detach().numpy()

        if no_gex:
            all_tcr_gex_embedding_reference = X_tcr_pca
        else:
            all_tcr_gex_embedding_reference = np.hstack([
                X_tcr_pca,
                X_gex
            ])
        X_tcr_pca = ln_1(torch.tensor(tcr_adata.obsm["X_tcr_pca"])).detach().numpy()
        X_gex = ln_2(torch.tensor(tcr_adata.obsm["X_gex"])).detach().numpy()

        if no_gex:
            all_tcr_gex_embedding_query = X_tcr_pca
        else:
            all_tcr_gex_embedding_query = np.hstack([
                X_tcr_pca,
                X_gex
            ])

        all_tcr_gex_embedding = np.vstack([
            all_tcr_gex_embedding_reference,
            all_tcr_gex_embedding_query
        ])
    else:
        # No LayerNorm for TCR and GEX. 
        if no_gex:
            all_tcr_gex_embedding_reference = tcr_reference_adata.obsm["X_tcr_pca"]
        else:
            all_tcr_gex_embedding_reference = np.hstack([
                tcr_reference_adata.obsm["X_tcr_pca"],
                6*tcr_reference_adata.obsm["X_gex"]
            ])
        if no_gex:
            all_tcr_gex_embedding_query = tcr_adata.obsm["X_tcr_pca"]
        else:
            all_tcr_gex_embedding_query = np.hstack([
                tcr_adata.obsm["X_tcr_pca"],
                6*tcr_adata.obsm["X_gex"]
            ])

        all_tcr_gex_embedding = np.vstack([
            all_tcr_gex_embedding_reference,
            all_tcr_gex_embedding_query
        ])

    df = pd.concat([
        tcr_reference_adata.obs,
        tcr_adata.obs
    ])

    result = _cluster_tcr_by_label(
        df, 
        all_tcr_gex_embedding, 
        all_tcr_gex_embedding_query, 
        label_key, 
        gpu,
        max_distance,
        max_cluster_size
    )
    return result

@typed({
    "df": pd.DataFrame,
    "all_tcr_gex_embedding": np.ndarray,
    "query_tcr_gex_embedding": np.ndarray,
    "label_key": str,
    "gpu": int,
    "max_distance": float,
    "max_cluster_size": int
})
def _cluster_tcr_by_label(
        df, 
        all_tcr_gex_embedding, 
        query_tcr_gex_embedding, 
        label_key, 
        gpu = 0,
        max_distance = 25,
        max_cluster_size = 40
) -> TCRDeepInsightClusterResult:
    kmeans = faiss.Kmeans(
        all_tcr_gex_embedding.shape[1],
        all_tcr_gex_embedding.shape[0],
        niter=20,
        verbose=True,
        gpu=gpu
    )
    kmeans.train(all_tcr_gex_embedding)
    D, I = kmeans.index.search(query_tcr_gex_embedding, max_cluster_size)

    _result = []
    label_map = dict(zip(range(len(df)), df[label_key]))
    labels = np.unique(df[label_key])
    mt("Iterative anchoring...")
    for i in tqdm.trange(I.shape[0]):
        label = np.array([label_map[x] for x in I[i]])
        for j in list(range(1, max_cluster_size+1))[::-1]:
            # if len(np.unique(label[:j])) == 1 and label[0] in labels:
            # Update: 
            if len(np.unique(label[:j])) == 1 and label[0] in labels and D[i][j-1] < max_distance:
                d = np.unique(label[:j])[0]
                comp = list(filter(lambda x: label[x] != d, range(j, max_cluster_size)))[:j]
                # comp = list(range(j, min(40, 2*j)))
                # comp = list(range(j, min(40, 2*j))) 
                # clust, comp = all_tcr_gex_embedding[I[i][:j]], all_tcr_gex_embedding[I[i][comp]]
                # a,b = clust.mean(axis=0), comp.mean(axis=0)
                # euc_distance = euclidean(a,b)
                # pval = multivariate_gaussian_p_value(clust, comp)


                _result.append([label[0]] + \
                        list( I[i][:j]) + \
                        [-1] * (max_cluster_size-j) + \
                        [j] + \
                        [D[i][1:j].mean(), 
                        D[i][comp].mean()
                ])

                break
    
    result = pd.DataFrame(_result)
    
    result['cluster_index'] = result.index

    result = result[result.iloc[:,max_cluster_size+1] > 1]
    result = result.sort_values(max_cluster_size+1, ascending=False)

    result.index = list(range(len(result)))
    all_indices = list(np.unique(result.iloc[:,1:41].to_numpy()))
    all_indices.remove(-1)
    all_indices = set(all_indices)
    selected_indices = []
    mt("Remove repetitive clusters...")
    for i in tqdm.trange(len(result)):
        if result.iloc[i,1] in all_indices:
            selected_indices.append(i)
            all_indices.remove(result.iloc[i,1])
        for a in result.iloc[i,2:max_cluster_size+1].to_numpy():
            if a >= 0 and a in all_indices:
                all_indices.remove(a)

    result = result.iloc[selected_indices]

    result_tcr = result.copy()

    result_individual = result.copy()
    for i in list(range(1, max_cluster_size+1))[::-1]:
        result_tcr.iloc[:,i] = result_tcr.iloc[:,i].apply(
            lambda x: df.iloc[x,:]['tcr'] if x >= 0 else '-'
        )
    for i in list(range(1, max_cluster_size+1))[::-1]:
        result_individual.iloc[:,i] = result_individual.iloc[:,i].apply(
            lambda x: df.iloc[x,:]['individual'] if x >= 0 else '-'
        )

    result_tcr['number_of_individuals'] = list(
        map(
            lambda x: len(np.unique(list(filter(lambda x: x != '-', x)))), 
            result_individual.iloc[:,1:41].to_numpy()
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
            result_tcr.iloc[:, 1:41].to_numpy(),
        )
    )



    result_tcr.columns = (
        [label_key]
        + [f"TCRab{x}" for x in range(1, 41)]
        + [
            "number_of_tcrs",
            "mean_distance",
            "mean_distance_other",
            "cluster_index",
            "number_of_individuals",
        ]
    )
    result_individual['number_of_individuals'] = result_tcr['number_of_individuals']
    result_individual.columns = (
        [label_key]
        + [f"individual{x}" for x in range(1, 41)]
        + [
            "number_of_tcrs",
            "mean_distance",
            "mean_distance_other",
            "cluster_index",
            "number_of_individuals",
        ]
    )

    # result_tcr = result_tcr[result_tcr['number_of_unique_tcrs'] > 1]
    # result_tcr = result_tcr[result_tcr['mean_distance'] > 1e-3]

    result_tcr['disease_association_score'] = result_tcr['mean_distance_other'] - result_tcr['mean_distance']

    result_tcr['tcr_similarity_score'] = 100 - result_tcr['mean_distance']

    return TCRDeepInsightClusterResult(
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

    kmeans = faiss.Kmeans(
        X_tcr_pca_query.shape[1],
        X_tcr_pca_query.shape[0],
        niter=20,
        verbose=True,
        gpu=gpu
    )

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
