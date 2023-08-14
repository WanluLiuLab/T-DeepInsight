# Pytorch
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.distributions import kl_divergence as kld
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Hugginface Transformers
from transformers import (
    BertConfig,
    PreTrainedModel,
    BertForMaskedLM
)

# Third Party
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
import datasets
import numpy as np
from pathlib import Path
import umap

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
from typing import Callable, Mapping, Union, Iterable, Tuple, Optional, Mapping
import os
import warnings

# Package
from ._primitives import *
from ._tokenizer import TCRabTokenizer
from ._config import get_config

from ..utils._tensor_utils import one_hot, get_k_elements, get_last_k_elements
from ..utils._decorators import typed
from ..utils._loss import LossFunction
from ..utils._logger import mt

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
from ..utils._umap import (
    umap_is_installed, 
    cuml_is_installed,
    get_default_umap_reducer,
    get_default_cuml_reducer
)

MODULE_PATH = Path(__file__).parent
warnings.filterwarnings("ignore")

class TRABModelMixin(nn.Module):
    def __init__(self,
        bert_config: BertConfig,
        pooling: Literal["cls", "mean", "max", "pool", "trb", "tra", "weighted"] = "mean",
        pooling_cls_position: int = 0,
        pooling_weight = (0.1,0.9),
        hidden_layers: Iterable[int] = [512,192],
        labels_number: int = 1,
    ) -> None:
        """
        TRABModel is a BERT model that takes in a αβTCR sequence
        :param bert_config: BertConfig
        :param pooling: Pooling method, one of "cls", "mean", "max", "pool", "trb", "tra", "weighted"
        :param pooling_cls_position: Position of the cls token
        :param pooling_weight: Weight of the cls token
        :param hidden_layers: Hidden layers of the classifier
        :param labels_number: Number of labels

        :example:
            >>> from tcr_deep_insight as tdi
            >>> model = tdi.model.TCRabModel(
            >>>    tdi.model.config.get_human_config(),
            >>>    labels_number=4
            >>> )
        """
        super(TRABModelMixin, self).__init__()
        self.model = BertForMaskedLM(bert_config)
        self.pooler = nn.Sequential(
            nn.Linear(bert_config.hidden_size,bert_config.hidden_size),
            nn.Tanh()
        )

        self.pooling = pooling
        self.pooling_cls_position = pooling_cls_position
        self.pooling_weight = pooling_weight

        self.config = bert_config

        self.labels_number = labels_number

        self.fct = nn.Sequential(
            nn.Linear(self.config.hidden_size, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0],hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], labels_number)
        )

    def forward(self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        token_type_ids: torch.Tensor,
        output_hidden_states = True,
    ):
        '''
        Forward pass of the model

        :param input_ids: Input ids
        :param attention_mask: Attention mask
        :param labels: Labels
        :param token_type_ids: Token type ids

        :return: Output of the model
        '''

        output = self.model.forward(
                input_ids = input_ids,
                attention_mask = attention_mask,
                return_dict = True,
                token_type_ids = token_type_ids,
                labels = labels,
                output_hidden_states = output_hidden_states
        )
        hidden_states = None
        hidden_states_length = int(output.hidden_states[-1].shape[1]/2)
        if output_hidden_states:
            if self.pooling == "mean":
                hidden_states = output.hidden_states[-1][:,1:,:].mean(1)
            elif self.pooling == "max":
                hidden_states = output.hidden_states[-1][:,1:,:].max(1)
            elif self.pooling == "cls":
                hidden_states = output.hidden_states[-1][:,self.pooling_cls_position,:]
            elif self.pooling == 'pool':
                hidden_states = self.pooler(output.hidden_states[-1][:,self.pooling_cls_position,:])
            elif self.pooling == 'tra':
                if self.pooling_cls_position == 1:
                    hidden_states = output.hidden_states[-1][
                        :,
                        torch.hstack([
                            torch.tensor([self.pooling_cls_position]),
                            torch.arange(2,hidden_states_length),
                        ]),
                        :
                    ].mean(1) + output.hidden_states[-1][:, 0] + output.hidden_states[-1][:, hidden_states_length]
                else:
                    hidden_states = output.hidden_states[-1][
                        :,
                        torch.arange(1,hidden_states_length),
                        :
                    ].mean(1)
            elif self.pooling == 'trb':
                if self.pooling_cls_position == 1:
                    hidden_states = output.hidden_states[-1][
                        :,
                        torch.hstack([
                            torch.tensor([self.pooling_cls_position]),
                            torch.arange(hidden_states_length+2,hidden_states_length*2),
                        ]),
                        :
                    ].mean(1) + output.hidden_states[-1][:, 0] + output.hidden_states[-1][:, hidden_states_length]
                else:
                    hidden_states = output.hidden_states[-1][
                        :,
                        torch.arange(hidden_states_length+1,hidden_states_length*2),
                        :
                    ].mean(1)
            elif self.pooling == 'weighted':
                if self.pooling_cls_position == 1:
                    hidden_states = (output.hidden_states[-1][
                        :,
                        torch.hstack([
                            torch.tensor([self.pooling_cls_position]),
                            torch.arange(2,hidden_states_length)
                        ]),
                        :
                    ] * self.pooling_weight[0] + output.hidden_states[-1][
                        :,
                        torch.hstack([
                            torch.tensor([self.pooling_cls_position]),
                            torch.arange(hidden_states_length+2,hidden_states_length*2),
                        ]),
                        :
                    ] * self.pooling_weight[1]).mean(1) + output.hidden_states[-1][:, 0] + output.hidden_states[-1][:, hidden_states_length]
                else:
                    hidden_states = (output.hidden_states[-1][
                        :,
                        torch.arange(hidden_states_length+1,hidden_states_length*2),
                        :
                    ] * self.pooling_weight[0] + output.hidden_states[-1][
                        :,
                        torch.arange(1,hidden_states_length)
                        :
                    ] * self.pooling_weight[1]).mean(1)
            else:
                raise ValueError("Unrecognized pool strategy")

        prediction_out = self.fct(hidden_states)

        return {
            "output": output,
            "hidden_states": hidden_states,
            "prediction_out": prediction_out,
        }
    
class VAEMixin(ReparameterizeLayerBase, MMDLayerBase):
    """Vanilla single-cell VAE"""
    def __init__(self, *,
       adata: Optional[sc.AnnData] = None,
       hidden_stacks: List[int] = [128], 
       n_latent: int = 10,
       n_batch: int = 0,
       n_label: int = 0,
       n_categorical_covariate: Optional[Iterable[int]] = None,
       batch_key = None,
       label_key = None,
       categorical_covariate_keys: Optional[Iterable[str]] = None,
       dispersion:  Literal["gene", "gene-batch", "gene-cell"] = "gene-cell", 
       log_variational: bool = True,
       bias: bool = True,
       use_batch_norm: bool = True,
       use_layer_norm: bool = False,
       batch_hidden_dim: int = 8,
       batch_embedding: Literal["embedding", "onehot"] = "onehot",
       reconstruction_method: Literal['mse', 'zg', 'zinb'] = 'zinb',
       constrain_latent_method: Literal['mse', 'normal'] = 'mse',
       constrain_latent_embedding: bool = False,
       constrain_latent_key: str = 'X_gex',
       encode_libsize: bool = False,
       decode_libsize: bool = True,
       dropout_rate: float = 0.1,
       activation_fn: nn.Module = nn.ReLU,
       inject_batch: bool = True,
       inject_label: bool = False,
       inject_categorical_covariate: bool = True,
       use_attention: bool = False,
       mmd_key: Optional[Literal['batch','label']] = None,
       device: Union[str, torch.device] = "cpu"
    ) -> None:
        """
        :param adata: AnnData. If provided, initialize the model with the adata.
        :param hidden_stacks: List[int]. Number of hidden units in each layer. Default: [128] (one hidden layer with 128 units)
        :param n_latent: int. Number of latent dimensions. Default: 10
        :param n_batch: int. Number of batch. Default: 0
        :param n_label: int. Number of label. Default: 0
        :param n_categorical_covariate: Optional[Iterable[int]]. Number of categorical covariate. Default: None
        :param batch_key: str. Batch key in adata.obs. Default: None
        :param label_key: str. Label key in adata.obs. Default: None
        :param categorical_covariate_keys: Optional[Iterable[str]]. Categorical covariate keys in adata.obs. Default: None
        :param dispersion: Literal["gene", "gene-batch", "gene-cell"]. Dispersion method. Default: "gene-cell"
        :param log_variational: bool. If True, log the variational distribution. Default: True
        :param bias: bool. If True, use bias in the linear layer. Default: True
        :param use_batch_norm: bool. If True, use batch normalization. Default: True
        :param use_layer_norm: bool. If True, use layer normalization. Default: False
        :param batch_hidden_dim: int. Number of hidden units in the batch embedding layer. Default: 8
        :param batch_embedding: Literal["embedding", "onehot"]. Batch embedding method. Default: "onehot"
        :param constrain_latent_method: Literal['mse', 'normal']. Method to constrain the latent embedding. Default: 'mse'
        :param constrain_latent_embedding: bool. If True, constrain the latent embedding. Default: False
        :param constrain_latent_key: str. Key to the data to constrain the latent embedding. Default: 'X_gex'
        :param encode_libsize: bool. If True, encode the library size. Default: False
        :param dropout_rate: float. Dropout rate. Default: 0.1
        :param activation_fn: nn.Module. Activation function. Default: nn.ReLU
        :param inject_batch: bool. If True, inject batch information. Default: True
        :param inject_label: bool. If True, inject label information. Default: False
        :param inject_categorical_covariate: bool. If True, inject categorical covariate information. Default: True
        :param use_attention: bool. If True, use attention. Default: False
        :param mmd_key: Optional[Literal['batch','label']]. If provided, use MMD loss. Default: None
        :param device: Union[str, torch.device]. Device to use. Default: "cpu"

        :example:
            >>> from tcr_deep_insight as tdi
            >>> model = tdi.model.VAEModel(
            >>>    adata,
            >>>    batch_key = 'batch',
            >>>    label_key = 'cell_type',
            >>>    categorical_covariate_keys = ['sample'],
            >>> )
        """

        super(VAEMixin, self).__init__()
        self.adata = adata
        self.in_dim = adata.shape[1] if adata else -1
        self.n_hidden = hidden_stacks[-1]
        self.n_latent = n_latent
        self.n_categorical_covariate = n_categorical_covariate
        self._hidden_stacks = hidden_stacks 

        if n_batch > 0 and not batch_key:
            raise ValueError("Please provide a batch key if n_batch is greater than 0")
        if n_label > 0 and not label_key:
            raise ValueError("Please provide a label key if n_batch is greater than 0")

        self.label_key = label_key
        self.batch_key = batch_key
        self.categorical_covariate_keys = categorical_covariate_keys
        self.n_batch = n_batch
        self.n_label = n_label
        self.new_adata_key = 'new_adata'
        self.new_adata_code = None
        self.log_variational = log_variational
        self.mmd_key = mmd_key
        self.reconstruction_method = reconstruction_method
        self.constrain_latent_embedding = constrain_latent_embedding
        self.constrain_latent_method = constrain_latent_method
        self.constrain_latent_key = constrain_latent_key
        self.device=device

        self.initialize_dataset()

        self.batch_embedding = batch_embedding
        if batch_embedding == "onehot":
            batch_hidden_dim = self.n_batch
        self.batch_hidden_dim = batch_hidden_dim
        self.inject_batch = inject_batch
        self.inject_label = inject_label
        self.inject_categorical_covariate = inject_categorical_covariate
        self.encode_libsize = encode_libsize
        self.decode_libsize = decode_libsize
        self.dispersion = dispersion
        self.use_attention = use_attention

        self.fcargs = dict(
            bias           = bias, 
            dropout_rate   = dropout_rate, 
            use_batch_norm = use_batch_norm, 
            use_layer_norm = use_layer_norm,
            activation_fn  = activation_fn,
            device         = device
        )
       
        

        #############################
        # Model Trainable Variables #
        #############################

        if self.dispersion == "gene":
            self.px_rate = torch.nn.Parameter(torch.randn(self.in_dim))
        elif self.dispersion == "gene-batch":
            self.px_rate = torch.nn.Parameter(torch.randn(self.in_dim, self.n_batch))
        elif self.dispersion == "gene-cell":
            pass 
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        ############
        # ENCODERS #
        ############

        self.encoder = SAE(
            self.in_dim if not self.encode_libsize else self.in_dim + 1,
            stacks = hidden_stacks,
            # n_cat_list = [self.n_batch] if self.n_batch > 0 else None,
            cat_dim = batch_hidden_dim,
            cat_embedding = batch_embedding,
            encode_only = True,
            **self.fcargs
        )

        # The latent cell representation z ~ Logisticnormal(0, I)
        self.z_mean_fc = nn.Linear(self.n_hidden, self.n_latent)
        self.z_var_fc = nn.Linear(self.n_hidden, self.n_latent)
        self.z_transformation = nn.Softmax(dim=-1)

        ############
        # DECODERS #
        ############

        if self.n_categorical_covariate_ is not None and self.inject_categorical_covariate:
            if self.n_batch > 0 and self.n_label > 0 and inject_batch and inject_label:
                decoder_n_cat_list = [self.n_batch, self.n_label, *self.n_categorical_covariate]
            elif self.n_batch > 0 and inject_batch:
                decoder_n_cat_list = [self.n_batch, *self.n_categorical_covariate]
            elif self.n_label > 0 and inject_label:
                decoder_n_cat_list = [self.n_label, *self.n_categorical_covariate]
            else:
                decoder_n_cat_list = None
        else:
            if self.n_batch > 0 and self.n_label > 0 and inject_batch and inject_label:
                decoder_n_cat_list = [self.n_batch, self.n_label]
            elif self.n_batch > 0 and inject_batch:
                decoder_n_cat_list = [self.n_batch]
            elif self.n_label > 0 and inject_label:
                decoder_n_cat_list = [self.n_label]
            else:
                decoder_n_cat_list = None
        
        self.decoder_n_cat_list = decoder_n_cat_list

        self.decoder = FCLayer(
            in_dim = self.n_latent, 
            out_dim = self.n_hidden,
            n_cat_list = decoder_n_cat_list,
            cat_dim = batch_hidden_dim,
            cat_embedding = batch_embedding,
            use_layer_norm=False,
            use_batch_norm=True,
            dropout_rate=0,
            device=device
        )

        self.px_rna_rate_decoder = nn.Linear(self.n_hidden, self.in_dim)
        self.px_rna_scale_decoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.in_dim),
            nn.Softmax(dim=-1)
        )
        self.px_rna_dropout_decoder = nn.Linear(self.n_hidden, self.in_dim)
        
        if self.n_label > 0:
            self.att = FCSelfAttention(dim = 1, dim_head = 5, n_heads = 5)
            self.fc = nn.Sequential(
                nn.Linear(self.n_latent, self.n_label),
            )

        self._trained = False 

        self.to(device)

    def initialize_dataset(self):
        mt("Initializing dataset into memory")
        if self.batch_key is not None:
            n_batch_ = len(np.unique(self.adata.obs[self.batch_key]))
            if self.n_batch != n_batch_:
                mt(f"warning: the provided n_batch={self.n_batch} does not match the number of batch in the adata.")

                mt(f"         setting n_batch to {n_batch_}")
                self.n_batch = n_batch_
            self.batch_category = pd.Categorical(self.adata.obs[self.batch_key])
            self.batch_category_summary = dict(Counter(self.batch_category))
            
        if self.label_key is not None:
            n_label_ = len(np.unique(list(filter(lambda x: x != self.new_adata_key, self.adata.obs[self.label_key]))))

            if self.n_label != n_label_:
                mt(f"warning: the provided n_label={self.n_label} does not match the number of batch in the adata.")

                mt(f"         setting n_label to {n_label_}")
                self.n_label = n_label_
            self.label_category = pd.Categorical(self.adata.obs[self.label_key])
            self.label_category_summary = dict(Counter(list(filter(lambda x: x != self.new_adata_key, self.label_category))))

            self.label_category_weight = len(self.label_category) / len(self.label_category.categories) * torch.tensor([self.label_category_summary[x] for x in list(filter(lambda x: x != self.new_adata_key, self.label_category.categories))], dtype=torch.float64).to(self.device)

            if self.new_adata_key in self.label_category.categories:
                self.new_adata_code = list(self.label_category.categories).index(self.new_adata_key)

        self.n_categorical_covariate_ = None
        if self.categorical_covariate_keys is not None:
            self.n_categorical_covariate_ = [len(np.unique(self.adata.obs[x])) for x in self.categorical_covariate_keys]
            if self.n_categorical_covariate == None or len(self.n_categorical_covariate_) != len(self.n_categorical_covariate):
                mt(f"warning: the provided n_categorical_covariate={self.n_categorical_covariate} does not match the number of categorical covariate in the adata.")
                mt(f"         setting n_categorical_covariate to {self.n_categorical_covariate_}")
                self.n_categorical_covariate = self.n_categorical_covariate_
            else:
                for e,(i,j) in enumerate(zip(self.n_categorical_covariate_, self.n_categorical_covariate)):
                    if i != j:
                        mt(f"n_categorical_covariate {self.categorical_covariate_keys[e]} does not match the number in the adata.")
                        mt(f"         setting n_categorical_covariate {e} to {i}")
                        self.n_categorical_covariate[e] = i
            self.categorical_covariate_category = [pd.Categorical(self.adata.obs[x]) for x in self.categorical_covariate_keys]
            self.categorical_covariate_category_summary = [dict(Counter(x)) for x in self.categorical_covariate_category]

        X = self.adata.X
        if self.log_variational:
            self.max_gene_exp = np.log(np.max(X)+1)
        else:
            self.max_gene_exp = np.max(X)

        self._n_record = X.shape[0]
        self._indices = np.array(list(range(self._n_record)))
        batch_categories, label_categories = None, None
        categorical_covariate_categories = None

        if self.batch_key is not None:
            if self.batch_key not in self.adata.obs.columns:
                raise ValueError(f"batch_key {self.batch_key} is not found in AnnData obs")
            batch_categories = np.array(self.batch_category.codes)
        if self.label_key is not None:
            if self.label_key not in self.adata.obs.columns:
                raise ValueError(f"label_key {self.label_key} is not found in AnnData obs")
            label_categories = np.array(self.label_category.codes)
        if self.categorical_covariate_keys is not None:
            for e,i in enumerate(self.categorical_covariate_keys):
                if i not in self.adata.obs.columns:
                    raise ValueError(f"categorical_covariate_keys {i} is not found in AnnData obs")
            categorical_covariate_categories = [np.array(x.codes) for x in self.categorical_covariate_category]


        if self.constrain_latent_embedding and self.constrain_latent_key in self.adata.obsm.keys():
            P = self.adata.obsm[self.constrain_latent_key]
            if categorical_covariate_categories is not None:
                if batch_categories is not None and label_categories is not None:
                    _dataset = list(zip(X, P, batch_categories, label_categories, *categorical_covariate_categories))
                elif batch_categories is not None:
                    _dataset = list(zip(X, P, batch_categories, *categorical_covariate_categories))
                elif label_categories is not None:
                    _dataset = list(zip(X, P, label_categories, *categorical_covariate_categories))
                else:
                    _dataset = list(zip(X, P, *categorical_covariate_categories))
            else:
                if batch_categories is not None and label_categories is not None:
                    _dataset = list(zip(X, P, batch_categories, label_categories))
                elif batch_categories is not None:
                    _dataset = list(zip(X, P, batch_categories))
                elif label_categories is not None:
                    _dataset = list(zip(X, P, label_categories))
                else:
                    _dataset = list(zip(X, P))
        else:
            if categorical_covariate_categories is not None:
                if batch_categories is not None and label_categories is not None:
                    _dataset = list(zip(X, batch_categories, label_categories, *categorical_covariate_categories))
                elif batch_categories is not None:
                    _dataset = list(zip(X, batch_categories, *categorical_covariate_categories))
                elif label_categories is not None:
                    _dataset = list(zip(X, label_categories, *categorical_covariate_categories))
                else:
                    _dataset = list(zip(X, *categorical_covariate_categories))
            else:
                if batch_categories is not None and label_categories is not None:
                    _dataset = list(zip(X, batch_categories, label_categories))
                elif batch_categories is not None:
                    _dataset = list(zip(X, batch_categories))
                elif label_categories is not None:
                    _dataset = list(zip(X, label_categories))
                else:
                    _dataset = list(X)
        
        
        _shuffle_indices = list(range(len(_dataset)))
        np.random.shuffle(_shuffle_indices)
        self._dataset = np.array([_dataset[i] for i in _shuffle_indices])
        self._shuffle_indices = np.array(
            [x for x, _ in sorted(zip(range(len(_dataset)), _shuffle_indices), key=lambda x: x[1])]
        )
        self._shuffled_indices_inverse = _shuffle_indices


    def as_dataloader(
        self, 
        subset_indices: Union[torch.tensor, np.ndarray] = None,
        n_per_batch: int = 128, 
        train_test_split: bool = False,
        random_seed: bool = 42,
        validation_split: bool = .2,
        shuffle: bool = True
    ):
        indices = subset_indices if subset_indices is not None else self._indices
        if shuffle:
            np.random.shuffle(indices)
        if train_test_split:
            np.random.seed(random_seed)
            split = int(np.floor(validation_split * self._n_record))
            if split % n_per_batch == 1:
                n_per_batch -= 1
            elif (self._n_record - split) % n_per_batch == 1:
                n_per_batch += 1
            train_indices, val_indices = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)
            return DataLoader(indices, n_per_batch,  sampler = train_sampler), DataLoader(indices, n_per_batch, sampler = valid_sampler)
        if len(indices) % n_per_batch == 1:
            n_per_batch -= 1
        return DataLoader(indices, n_per_batch, shuffle = shuffle)


    def encode(self, X: torch.tensor, batch_index: torch.tensor = None, eps: float = 1e-4):
        # Encode for hidden space
        # if batch_index is not None and self.inject_batch:
        #    X = torch.hstack([X, batch_index])
        libsize = torch.log(X.sum(1))
        if self.log_variational:
            X = torch.log(1+X)
        q = self.encoder.encode(torch.hstack([X,libsize.unsqueeze(1)])) if self.encode_libsize else self.encoder.encode(X)
        q_mu = self.z_mean_fc(q)
        q_var = torch.exp(self.z_var_fc(q)) + eps
        z = Normal(q_mu, q_var.sqrt()).rsample()
        H = dict(
            q = q,
            q_mu = q_mu, 
            q_var = q_var,
            z = z
        )

        return H 

    def decode(self, 
        H: Mapping[str, torch.tensor], 
        lib_size:torch.tensor, 
        batch_index: torch.tensor = None, 
        label_index: torch.tensor = None,
        categorical_covariate_index: torch.tensor = None,
        eps: float = 1e-4
    ):
        z = H["z"] # cell latent representation

        if categorical_covariate_index is not None and self.inject_categorical_covariate:
            if batch_index is not None and label_index is not None and self.inject_batch and self.inject_label:
                z = torch.hstack([z, batch_index, label_index, *categorical_covariate_index])
            elif batch_index is not None and self.inject_batch:
                z = torch.hstack([z, batch_index, *categorical_covariate_index])
            elif label_index is not None and self.inject_label:
                z = torch.hstack([z, label_index, *categorical_covariate_index])
        else:
            if batch_index is not None and label_index is not None and self.inject_batch and self.inject_label:
                z = torch.hstack([z, batch_index, label_index])
            elif batch_index is not None and self.inject_batch:
                z = torch.hstack([z, batch_index])
            elif label_index is not None and self.inject_label:
                z = torch.hstack([z, label_index])

        # eps to prevent numerical overflow and NaN gradient
        px = self.decoder(z)
        h = None

        if self.use_attention:
            h = rearrange(z, 'n (d z)-> n d z', z = 1)
            h = rearrange(self.att(h), 'n d z -> n (d z)')

        px_rna_scale = self.px_rna_scale_decoder(px) 
        if self.decode_libsize:
            px_rna_scale = px_rna_scale * lib_size.unsqueeze(1) 
        
        if self.dispersion == "gene-cell":
            px_rna_rate = self.px_rna_rate_decoder(px) ## In logits 
        elif self.dispersion == "gene-batch":
            px_rna_rate = F.linear(one_hot(batch_index, self.n_batch), self.px_rate)
        elif self.dispersion == "gene":
            px_rna_rate = self.px_rate

        px_rna_dropout = self.px_rna_dropout_decoder(px)  ## In logits
    
        R = dict(
            h = h,
            px = px,
            px_rna_scale = px_rna_scale,
            px_rna_rate = px_rna_rate,
            px_rna_dropout = px_rna_dropout
        )
        return R

    def forward(
        self,  
        X: torch.tensor, 
        lib_size: torch.tensor, 
        batch_index: torch.tensor = None,
        label_index: torch.tensor = None,
        categorical_covariate_index: torch.tensor = None,
        P: torch.tensor = None,
        reduction: str = "sum"
    ):
        H = self.encode(X, batch_index)
        q_mu = H["q_mu"]
        q_var = H["q_var"]
        mean = torch.zeros_like(q_mu)
        scale = torch.ones_like(q_var)
        kldiv_loss = kld(Normal(q_mu, q_var.sqrt()), Normal(mean, scale)).sum(dim = 1)
        prediction_loss = torch.tensor(0.)

        R = self.decode(H, lib_size, batch_index, label_index, categorical_covariate_index)

        if self.reconstruction_method == 'zinb':
            reconstruction_loss = LossFunction.zinb_reconstruction_loss(
                X,
                mu = R['px_rna_scale'],
                theta = R['px_rna_rate'].exp(), 
                gate_logits = R['px_rna_dropout'],
                reduction = reduction
            )
        elif self.reconstruction_method == 'zg':
            reconstruction_loss = LossFunction.zi_gaussian_reconstruction_loss(
                X,
                mean=R['px_rna_scale'],
                variance=R['px_rna_rate'].exp(),
                gate_logits=R['px_rna_dropout'],
                reduction=reduction
            )
        elif self.reconstruction_method == 'mse':
            reconstruction_loss = nn.functional.mse_loss(
                X,
                R['px_rna_scale'],
                reduction=reduction
            )

        if self.n_label > 0:
            criterion = nn.CrossEntropyLoss(weight=self.label_category_weight)
            if self.use_attention:
                prediction = self.fc(R['h'])
            else:
                prediction = self.fc(H['z'])

            if self.new_adata_code and self.new_adata_code in label_index:
                prediction_index = (label_index != self.new_adata_code).squeeze() 
                prediction_loss = criterion(prediction[prediction_index], one_hot(label_index[prediction_index], self.n_label))
            else:
                prediction_loss = criterion(prediction, one_hot(label_index, self.n_label))

        latent_constrain = torch.tensor(0.)
        if self.constrain_latent_embedding and P is not None:
            # Constrains on cells with no PCA information will be ignored
            latent_constrain_mask = P.mean(1) != 0
            if self.constrain_latent_method == 'mse':
                latent_constrain = (
                    nn.MSELoss(reduction='none')(P, q_mu).sum(1) * latent_constrain_mask 
                ).sum() / len(list(filter(lambda x: x != 0, P.detach().cpu().numpy().mean(1))))
            elif self.constrain_latent_method == 'normal':
                latent_constrain = (
                    kld(Normal(q_mu, q_var.sqrt()), Normal(P, torch.ones_like(P))).sum(1) * latent_constrain_mask
                ).sum() / len(list(filter(lambda x: x != 0, P.detach().cpu().numpy().mean(1))))

        mmd_loss = torch.tensor(0.)
        if self.mmd_key:
            if self.mmd_key == 'batch':
                mmd_loss = self.MMDLoss(H['q'], batch_index.detach().cpu().numpy())
            elif self.mmd_key == 'label':
                mmd_loss = self.MMDLoss(H['q'], label_index.detach().cpu().numpy())
            else:
                raise ValueError('mmd_key should be one of batch, label')
            
        loss_record = {
            "reconstruction_loss": reconstruction_loss,
            "prediction_loss": prediction_loss,
            "kldiv_loss": kldiv_loss,
            "mmd_loss": mmd_loss,
            "latent_constrain_loss": latent_constrain
        }
        return H, R, loss_record

    def calculate_metric(self, X_test, kl_weight, pred_weight, mmd_weight):
        epoch_total_loss = 0
        epoch_reconstruction_loss = 0
        epoch_kldiv_loss = 0
        epoch_prediction_loss = 0
        epoch_mmd_loss = 0
        b = 0
        with torch.no_grad():
            for b, X in enumerate(X_test):
                X = self._dataset[X.cpu().numpy()]
                batch_index, label_index, categorical_covariate_index = None, None, None
                if self.n_categorical_covariate_ is not None:
                    if self.n_batch > 0 and self.n_label > 0:
                        if self.constrain_latent_embedding:
                            X, P, batch_index, label_index, categorical_covariate_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2), get_last_k_elements(X,3), get_k_elements(X,4)
                        else:
                            X, batch_index, label_index, categorical_covariate_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2), get_last_k_elements(X,3)
                    elif self.n_batch > 0:
                        if self.constrain_latent_embedding:
                            X, P, batch_index, categorical_covariate_index = get_k_elements(X,0), get_k_elements(X,1),  get_k_elements(X,2), get_last_k_elements(X,3)
                        else:
                            X, batch_index, categorical_covariate_index = get_k_elements(X,0), get_k_elements(X,1),  get_last_k_elements(X,2)
                    elif self.n_label > 0:
                        if self.constrain_latent_embedding:
                            X, P, label_index, categorical_covariate_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2), get_last_k_elements(X,3)
                        else: 
                            X, label_index, categorical_covariate_index = get_k_elements(X,0), get_k_elements(X,1), get_last_k_elements(X,2)
                    categorical_covariate_index = list(np.vstack(categorical_covariate_index).T.astype(int))
                else:
                    if self.n_batch > 0 and self.n_label > 0:
                        if self.constrain_latent_embedding:
                            X, P, batch_index, label_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2), get_k_elements(X,3)
                        else:
                            X, batch_index, label_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2)
                    elif self.n_batch > 0:
                        if self.constrain_latent_embedding:
                            X, P, batch_index = get_k_elements(X,0), get_k_elements(X,1),  get_k_elements(X,2)
                        else:
                            X, batch_index = get_k_elements(X,0), get_k_elements(X,1)
                    elif self.n_label > 0:
                        if self.constrain_latent_embedding:
                            X, P, label_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2)
                        else: 
                            X, label_index = get_k_elements(X,0), get_k_elements(X,1)
                
                X = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, X))))


                if self.n_label > 0:
                    label_index = torch.tensor(label_index)
                    if not isinstance(label_index, torch.FloatTensor):
                        label_index = label_index.type(torch.FloatTensor)
                    label_index = label_index.to(self.device).unsqueeze(1)
                if self.n_batch > 0:
                    batch_index = torch.tensor(batch_index)
                    if not isinstance(batch_index, torch.FloatTensor):
                        batch_index = batch_index.type(torch.FloatTensor)
                    batch_index = batch_index.to(self.device).unsqueeze(1)  
                if self.n_categorical_covariate_ is not None:
                    for i in range(len(categorical_covariate_index)):
                        categorical_covariate_index[i] = torch.tensor(categorical_covariate_index[i])
                        if not isinstance(categorical_covariate_index[i], torch.FloatTensor):
                            categorical_covariate_index[i] = categorical_covariate_index[i].type(torch.FloatTensor)
                        categorical_covariate_index[i] = categorical_covariate_index[i].to(self.device).unsqueeze(1)

                if not isinstance(X, torch.FloatTensor):
                    X = X.type(torch.FloatTensor)
                X = X.to(self.device)   

                
                lib_size = X.sum(1).to(self.device) 

                H, R, L = self.forward(X, lib_size, batch_index, label_index, categorical_covariate_index)

                reconstruction_loss = L['reconstruction_loss']
                prediction_loss = pred_weight * L['prediction_loss'] 
                kldiv_loss = kl_weight * L['kldiv_loss']    
                mmd_loss = mmd_weight * L['mmd_loss']

                avg_reconstruction_loss = reconstruction_loss.sum()
                avg_kldiv_loss = kldiv_loss.sum()
                avg_mmd_loss = mmd_loss

                epoch_reconstruction_loss += avg_reconstruction_loss.item()
                epoch_kldiv_loss += avg_kldiv_loss.item()
                if self.n_label > 0:
                    epoch_prediction_loss += prediction_loss.sum().item() 

                epoch_mmd_loss += avg_mmd_loss
                epoch_total_loss += (avg_reconstruction_loss + avg_kldiv_loss + avg_mmd_loss).item()
        return {
            "epoch_reconstruction_loss":  epoch_reconstruction_loss / (b+1),
            "epoch_kldiv_loss": epoch_kldiv_loss / (b+1),
            "epoch_mmd_loss": epoch_mmd_loss / (b+1),
            "epoch_total_loss": epoch_total_loss / (b+1),   
        }

    
    def fit(self,
            max_epoch:int = 35, 
            n_per_batch:int = 128,
            kl_weight: float = 1.,
            pred_weight: float = 1.,
            mmd_weight: float = 1.,
            constrain_weight: float = 1.,
            optimizer_parameters: Iterable = None,
            validation_split: float = .2,
            lr: bool = 1e-3,
            lr_schedule: bool = False,
            lr_factor: float = 0.6,
            lr_patience: int = 30,
            lr_threshold: float = 0.0,
            lr_min: float = 1e-6,
            n_epochs_kl_warmup: Union[int, None] = 400,
            weight_decay: float = 1e-6,
            random_seed: int = 12,
            subset_indices: Union[torch.tensor, np.ndarray] = None,
            pred_last_n_epoch: int = 10,
            pred_last_n_epoch_fconly: bool = False,
            reconstruction_reduction: str = 'sum',
        ):
        self.train()
        if n_epochs_kl_warmup:
            n_epochs_kl_warmup = min(max_epoch, n_epochs_kl_warmup)
            kl_warmup_gradient = kl_weight / n_epochs_kl_warmup
            kl_weight_max = kl_weight
            kl_weight = 0.

        if optimizer_parameters is None:
            optimizer = optim.AdamW(self.parameters(), lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(optimizer_parameters, lr, weight_decay=weight_decay)

        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=lr_patience,
            factor=lr_factor,
            threshold=lr_threshold,
            min_lr=lr_min,
            threshold_mode="abs",
            verbose=True,
        ) if lr_schedule else None 

        labels=None

        from tqdm import tqdm
        best_state_dict = None
        best_score = 0
        current_score = 0
        pbar = tqdm(range(max_epoch), desc="Epoch")
        loss_record = {
                "epoch_reconstruction_loss": 0,
                "epoch_kldiv_loss": 0,
                "epoch_prediction_loss": 0,
                "epoch_mmd_loss": 0,
                "epoch_total_loss": 0
        }
        for epoch in range(1, max_epoch+1):
            self._trained = True
            pbar.desc = "Epoch {}".format(epoch)
            epoch_total_loss = 0
            epoch_reconstruction_loss = 0
            epoch_kldiv_loss = 0
            epoch_prediction_loss = 0
            epoch_mmd_loss = 0
            X_train, X_test = self.as_dataloader(
                n_per_batch=n_per_batch, 
                train_test_split = True, 
                validation_split = validation_split, 
                random_seed=random_seed,
                subset_indices=subset_indices
            )
            if self.n_label > 0 and epoch+1 == max_epoch - pred_last_n_epoch:
                mt("saving transcriptome only state dict")
                self.gene_only_state_dict = deepcopy(self.state_dict())
                if  pred_last_n_epoch_fconly:
                    optimizer = optim.AdamW(chain(self.att.parameters(), self.fc.parameters()), lr, weight_decay=weight_decay)
            
            for b, X in enumerate(X_train):
                P = None
                X = self._dataset[X.cpu().numpy()]
                batch_index, label_index, categorical_covariate_index = None, None, None
                if self.n_batch > 0 or self.n_label > 0:
                    if not isinstance(X, Iterable) and len(X) > 1:
                        raise ValueError()
                    if self.n_categorical_covariate_ is not None:
                        if self.n_batch > 0 and self.n_label > 0:
                            if self.constrain_latent_embedding:
                                X, P, batch_index, label_index, categorical_covariate_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2), get_last_k_elements(X,3), get_k_elements(X,4)
                            else:
                                X, batch_index, label_index, categorical_covariate_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2), get_last_k_elements(X,3)
                        elif self.n_batch > 0:
                            if self.constrain_latent_embedding:
                                X, P, batch_index, categorical_covariate_index = get_k_elements(X,0), get_k_elements(X,1),  get_k_elements(X,2), get_last_k_elements(X,3)
                            else:
                                X, batch_index, categorical_covariate_index = get_k_elements(X,0), get_k_elements(X,1),  get_last_k_elements(X,2)
                        elif self.n_label > 0:
                            if self.constrain_latent_embedding:
                                X, P, label_index, categorical_covariate_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2), get_last_k_elements(X,3)
                            else: 
                                X, label_index, categorical_covariate_index = get_k_elements(X,0), get_k_elements(X,1), get_last_k_elements(X,2)
                        categorical_covariate_index = list(np.vstack(categorical_covariate_index).T.astype(int))
                    else:
                        if self.n_batch > 0 and self.n_label > 0:
                            if self.constrain_latent_embedding:
                                X, P, batch_index, label_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2), get_k_elements(X,3)
                            else:
                                X, batch_index, label_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2)
                        elif self.n_batch > 0:
                            if self.constrain_latent_embedding:
                                X, P, batch_index = get_k_elements(X,0), get_k_elements(X,1),  get_k_elements(X,2)
                            else:
                                X, batch_index = get_k_elements(X,0), get_k_elements(X,1)
                        elif self.n_label > 0:
                            if self.constrain_latent_embedding:
                                X, P, label_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2)
                            else: 
                                X, label_index = get_k_elements(X,0), get_k_elements(X,1)


                X = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, X))))
                if self.constrain_latent_embedding:
                    P = torch.tensor(np.array(P)).type(torch.FloatTensor).to(self.device)

                if self.n_label > 0:
                    label_index = torch.tensor(label_index)
                    if not isinstance(label_index, torch.FloatTensor):
                        label_index = label_index.type(torch.FloatTensor)
                    label_index = label_index.to(self.device).unsqueeze(1)
                if self.n_batch > 0:
                    batch_index = torch.tensor(batch_index)
                    if not isinstance(batch_index, torch.FloatTensor):
                        batch_index = batch_index.type(torch.FloatTensor)
                    batch_index = batch_index.to(self.device).unsqueeze(1)
                if self.n_categorical_covariate_ is not None:
                    for i in range(len(categorical_covariate_index)):
                        categorical_covariate_index[i] = torch.tensor(categorical_covariate_index[i])
                        if not isinstance(categorical_covariate_index[i], torch.FloatTensor):
                            categorical_covariate_index[i] = categorical_covariate_index[i].type(torch.FloatTensor)
                        categorical_covariate_index[i] = categorical_covariate_index[i].to(self.device).unsqueeze(1)

                if not isinstance(X, torch.FloatTensor):
                    X = X.type(torch.FloatTensor)
                X = X.to(self.device)

                
                lib_size = X.sum(1).to(self.device)

                H, R, L = self.forward(X, lib_size, batch_index, label_index, categorical_covariate_index, P, reduction=reconstruction_reduction)

                reconstruction_loss = L['reconstruction_loss']
                prediction_loss = pred_weight * L['prediction_loss'] 
                kldiv_loss = kl_weight * L['kldiv_loss']    
                mmd_loss = mmd_weight * L['mmd_loss']

                avg_reconstruction_loss = reconstruction_loss.sum()  / n_per_batch
                avg_kldiv_loss = kldiv_loss.sum()  / n_per_batch
                avg_mmd_loss = mmd_loss / n_per_batch

                epoch_reconstruction_loss += avg_reconstruction_loss.item()
                epoch_kldiv_loss += avg_kldiv_loss.item()
                if self.n_label > 0:
                    epoch_prediction_loss += prediction_loss.sum().item()

                if epoch > max_epoch - pred_last_n_epoch:
                    loss = avg_reconstruction_loss + avg_kldiv_loss + avg_mmd_loss + prediction_loss.sum()
                else: 
                    loss = avg_reconstruction_loss + avg_kldiv_loss + avg_mmd_loss
    
                if self.constrain_latent_embedding:
                    loss += constrain_weight * L['latent_constrain_loss']

                epoch_total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({
                    'reconst': '{:.2e}'.format(loss_record["epoch_reconstruction_loss"]),
                    'kldiv': '{:.2e}'.format(loss_record["epoch_kldiv_loss"]),
                    'mmd': '{:.2e}'.format(loss_record["epoch_mmd_loss"]),
                    'step': f'{b} / {len(X_train)}'
                })
            loss_record = self.calculate_metric(X_test, kl_weight, pred_weight, mmd_weight)
            if lr_schedule:
                scheduler.step(loss_record["epoch_total_loss"])
            pbar.set_postfix({
                'reconst': '{:.2e}'.format(loss_record["epoch_reconstruction_loss"]),
                'kldiv': '{:.2e}'.format(loss_record["epoch_kldiv_loss"]),
                'mmd': '{:.2e}'.format(loss_record["epoch_mmd_loss"]),
            })
            pbar.update(1)
            if n_epochs_kl_warmup:
                kl_weight = min( kl_weight + kl_warmup_gradient, kl_weight_max)
            random_seed += 1
        if current_score < best_score:
            mt("restoring state dict with best performance")
            self.load_state_dict(best_state_dict)
        pbar.close()
        self.trained_state_dict = deepcopy(self.state_dict())

    
    @torch.no_grad()
    def get_latent_embedding(self, latent_key: Literal["z", "q_mu"] = "q_mu", n_per_batch: int = 128) -> np.ndarray:
        X = self.as_dataloader(subset_indices = list(range(len(self._dataset))), shuffle=False, n_per_batch=n_per_batch)
        Zs = []
        for x in X:
            x = self._dataset[x.cpu().numpy()]
            batch_index = None
            label_index = None
            if self.n_batch > 0 or self.n_label > 0:
                if not isinstance(x, Iterable) and len(x) > 1:
                    raise ValueError()
                if self.n_batch > 0 and self.n_label > 0:
                    X, batch_index, label_index = get_k_elements(x,0), get_k_elements(x,1), get_k_elements(X,2)
                elif self.n_batch > 0:
                    X, batch_index = get_k_elements(x,0), get_k_elements(x,1)
                elif self.n_label > 0:
                    X, label_index = get_k_elements(x,0), get_k_elements(x,1)
                        
            if self.n_label > 0:
                label_index = torch.tensor(label_index)
                if not isinstance(label_index, torch.FloatTensor):
                    label_index = label_index.type(torch.FloatTensor)
                label_index = label_index.to(self.device).unsqueeze(1)
            if self.n_batch > 0:
                batch_index = torch.tensor(batch_index)
                if not isinstance(batch_index, torch.FloatTensor):
                    batch_index = batch_index.type(torch.FloatTensor)
                batch_index = batch_index.to(self.device).unsqueeze(1)
                
            X = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, X))))
            if not isinstance(X, torch.FloatTensor):
                X = X.type(torch.FloatTensor)
            X = X.to(self.device)
                    
            H = self.encode(X, batch_index if batch_index != None else None)
            Zs.append(H[latent_key].detach().cpu().numpy())
        return np.vstack(Zs)[self._shuffle_indices]

    @torch.no_grad()
    def get_normalized_expression(self, k = 'px_rna_scale') -> np.ndarray:
        X = self.as_dataloader(subset_indices = list(range(len(self._dataset))), shuffle=False)
        Zs = []
        for x in X:
            batch_index = None
            if self.n_batch > 0 or self.n_label > 0:
                if not isinstance(x, list) and len(x) > 1:
                    raise ValueError()
                if self.n_batch > 0 and self.n_label > 0:
                    x, batch_index, label_index = x
                    batch_index = batch_index.unsqueeze(1)
                    label_index = label_index.unsqueeze(1)
                elif self.n_batch > 0:
                    x, batch_index = x
                    batch_index = batch_index.unsqueeze(1)
                elif self.n_label > 0:
                    x, label_index = x
                    label_index = label_index.unsqueeze(1)

            _,R,_ = self.forward(
                x.to(self.device), 
                x.sum(1).to(self.device), 
                batch_index.to(self.device),
                label_index.to(self.device)
            )
            Zs.append(R[k].detach().cpu().numpy())
        return np.vstack(Zs)[self._shuffle_indices]


    def to(self, device:str):
        super(VAEMixin, self).to(device)
        self.device=device 
        return self

    def transfer(self, 
        new_adata: sc.AnnData, 
        batch_key: str, 
        concat_with_original: bool = True,
        fraction_of_original: Optional[float] = None,
        times_of_new: Optional[float] = None
    ):
        new_batch_category = new_adata.obs[batch_key]
        original_batch_dim = self.batch_hidden_dim
        new_n_batch = len(np.unique(new_batch_category))

        if self.batch_embedding == "embedding":
            original_embedding_weight = self.decoder.cat_embedding[0].weight
        
        new_adata.obs[self.batch_key] = new_adata.obs[batch_key]
        
        original_batch_categories = self.batch_category.categories

        if fraction_of_original is not None:
            old_adata = random_subset_by_key(
                self.adata, 
                key = batch_key, 
                n = int(len(self.adata) * fraction_of_original)
            )
        elif times_of_new is not None:
            old_adata = random_subset_by_key(
                self.adata, 
                key = batch_key, 
                n = int(len(new_adata) * times_of_new)
            )
        else:
            old_adata = self.adata

        old_adata.obs['_transfer_label'] = 'reference'
        new_adata.obs['_transfer_label'] = 'query'

        if concat_with_original:
            self.adata = sc.concat([old_adata, new_adata])
        else:
            self.adata = new_adata
        
        self.initialize_dataset()

        if self.batch_embedding == "onehot":
            self.batch_hidden_dim = self.n_batch
        
        if self.n_categorical_covariate_ is not None and self.inject_categorical_covariate:
            if self.n_batch > 0 and self.n_label > 0 and self.inject_batch and self.inject_label:
                decoder_n_cat_list = [self.n_batch, self.n_label, *self.n_categorical_covariate]
            elif self.n_batch > 0 and self.inject_batch:
                decoder_n_cat_list = [self.n_batch, *self.n_categorical_covariate]
            elif self.n_label > 0 and self.inject_label:
                decoder_n_cat_list = [self.n_label, *self.n_categorical_covariate]
            else:
                decoder_n_cat_list = None
        else:
            if self.n_batch > 0 and self.n_label > 0 and self.inject_batch and self.inject_label:
                decoder_n_cat_list = [self.n_batch, self.n_label]
            elif self.n_batch > 0 and self.inject_batch:
                decoder_n_cat_list = [self.n_batch]
            elif self.n_label > 0 and self.inject_label:
                decoder_n_cat_list = [self.n_label]
            else:
                decoder_n_cat_list = None
        
        
        self.decoder_n_cat_list = decoder_n_cat_list
        
        original_weight = torch.tensor(self.decoder._fclayer[0].weight)

        self.decoder = FCLayer(
            in_dim = self.n_latent, 
            out_dim = self.n_hidden,
            n_cat_list = self.decoder_n_cat_list,
            cat_dim = self.batch_hidden_dim,
            cat_embedding = self.batch_embedding,
            use_layer_norm=False,
            use_batch_norm=True,
            dropout_rate=0,
            device=self.device
        )


        if self.batch_embedding == 'embedding':
            new_embedding = nn.Embedding(self.n_batch + new_n_batch, self.batch_hidden_dim).to(self.device)
            original_category_index = [list(self.batch_category.categories).index(x) for x in original_batch_categories]
            new_embedding_weight = new_embedding.weight.detach()
            new_embedding_weight[original_category_index] = original_embedding_weight.detach()
            new_embedding.weight = nn.Parameter(new_embedding_weight)
            new_embedding = new_embedding.to(self.device)
            self.decoder.cat_embedding[0] = new_embedding

        new_weight = torch.tensor(self.decoder._fclayer[0].weight)
        new_weight[:,:(self.n_latent + original_batch_dim)] = original_weight[:,:(self.n_latent + original_batch_dim)]
        self.decoder._fclayer[0].weight = nn.Parameter(new_weight)
        self.to(self.device)

    def transfer_umap(self, reference_adata: sc.AnnData, use_cuml: bool = False):
        """
        Transfer umap embedding from reference_adata to self.adata

        :param reference_adata: sc.AnnData
        """

        s = set(reference_adata.obs.index)
        s = list(filter(lambda x: x in s, self.adata.obs.index))
        self.adata.obsm["X_umap"] = np.zeros((len(self.adata), 2))
        ss = set(s)
        indices = list(map(lambda x: x in ss, self.adata.obs.index))
        self.adata.obsm["X_umap"][indices] = reference_adata[s].obsm["X_umap"]


        if 'X_gex' not in self.adata.obsm.keys():
            mt("X_gex is not found in model.adata.obsm. Calculating latent embedding")
            Z = self.get_latent_embedding()
            self.adata.obsm["X_gex"] = Z 
            mt("latent embedding calculation finished")

        x = self.adata.obsm["X_umap"][indices]
        mt("Fitting reference UMAP")
        if use_cuml:
            raise NotImplementedError()
        else:
            reducer = get_default_umap_reducer(
                init = x,
                target_metric = 'euclidean', 
                n_epochs = 10
            )

        reducer.fit(self.adata.obsm["X_gex"][indices], y=x)
        Z = self.adata.obsm["X_gex"]
        mt("Transforming UMAP")
        self.adata.obsm["X_umap"] = reducer.transform(Z)
        

    def transfer_label(self, reference_adata: sc.AnnData, label_key: str, method: Literal['knn'] = 'knn', **method_kwargs):
        """
        Transfer label from reference_adata to self.adata

        :param reference_adata: sc.AnnData
        :param label_key: str
        """
        s = set(reference_adata.obs.index)
        s = list(filter(lambda x: x in s, self.adata.obs.index))

        self.adata.obs[label_key] = np.nan
        ss = set(s)
        indices = list(map(lambda x: x in ss, self.adata.obs.index))
        self.adata.obs[label_key][indices] = reference_adata[s].obs[label_key]

        if 'X_gex' not in self.adata.obsm.keys():
            Z = self.get_latent_embedding()
            self.adata.obsm["X_gex"] = Z

        if method == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(**method_kwargs)
            knn.fit(
                self.adata.obsm["X_gex"][indices], self.adata.obs.loc[indices, label_key]
            )
            self.adata.obs.loc[self.adata.obs['_transfer_label'] == 'query', label_key] = knn.predict(
                self.adata.obsm["X_gex"][self.adata.obs['_transfer_label'] == 'query']
            )
        else:
            raise NotImplementedError()
