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

from ._model import VAEMixin


class OrthoLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(OrthoLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = F.normalize(x, dim=-1)  # Normalize the output vector to make it orthonormal
        return x
    
class TransitionVAE(VAEMixin):
    def __init__(self, *args, **kwargs):
        super(TransitionVAE, self).__init__(*args, **kwargs)
        self.fc_t = nn.Sequential(
            nn.Linear(self._hidden_stacks[-1], self.n_latent),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_latent),
            nn.Linear(self.n_latent, self.n_latent),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_latent),
            nn.Linear(self.n_latent, self.n_latent)
        )
        self.fc_d = nn.Sequential(
            nn.Linear(self.n_latent, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.ReLU()
        )
        self.to(self.device)

    def initialize_relations(self, relations):
        self.relations = relations

    
    def fit(self,
            max_epoch:int = 35, 
            n_per_batch:int = 128,
            reconstruction_weight: float = 1.,
            kl_weight: float = 1.,
            pred_weight: float = 1.,
            mmd_weight: float = 1.,
            constrain_weight: float = 1.,
            transition_weight: float = 1.,
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
        epoch_transition_loss_final = 0
        for epoch in range(1, max_epoch+1):
            self._trained = True
            pbar.desc = "Epoch {}".format(epoch)
            epoch_total_loss = 0
            epoch_reconstruction_loss = 0
            epoch_kldiv_loss = 0
            epoch_prediction_loss = 0
            epoch_transition_loss = 0
            X_train, X_test = self.as_dataloader(
                n_per_batch=n_per_batch, 
                train_test_split = True, 
                validation_split = validation_split, 
                random_seed=random_seed,
                subset_indices=subset_indices
            )

            n_steps = len(X_train)
            relations_n_per_batch = int(len(self.relations) / n_steps)

            relations_dataloader = DataLoader(
                self.relations, 
                batch_size=relations_n_per_batch, 
                shuffle=True
            )

            if self.n_label > 0 and epoch+1 == max_epoch - pred_last_n_epoch:
                mt("saving transcriptome only state dict")
                self.gene_only_state_dict = deepcopy(self.state_dict())
                if  pred_last_n_epoch_fconly:
                    optimizer = optim.AdamW(chain(self.att.parameters(), self.fc.parameters()), lr, weight_decay=weight_decay)
            
            for b, (X,R) in enumerate(zip(X_train, relations_dataloader)):
                P = None
                source_P = None
                target_P = None
                X = self._dataset[X.cpu().numpy()]

                source, target = R[:,0], R[:,1]
                source = source.cpu().numpy().astype(int)
                target = target.cpu().numpy().astype(int)
                source_X = self._dataset[self._shuffle_indices[source]]
                target_X = self._dataset[self._shuffle_indices[target]] 

                batch_index, label_index, categorical_covariate_index = None, None, None
                source_batch_index, source_label_index, source_categorical_covariate_index = None, None, None
                target_batch_index, target_label_index, target_categorical_covariate_index = None, None, None
                if self.n_batch > 0 or self.n_label > 0:
                    if not isinstance(X, Iterable) and len(X) > 1:
                        raise ValueError()
                    if self.n_categorical_covariate_ is not None:
                        if self.n_batch > 0 and self.n_label > 0:
                            if self.constrain_latent_embedding:
                                X, P, batch_index, label_index, categorical_covariate_index = (
                                    get_k_elements(X,0), 
                                    get_k_elements(X,1), 
                                    get_k_elements(X,2), 
                                    get_last_k_elements(X,3), 
                                    get_k_elements(X,4)
                                )
                                source_X, source_P, source_batch_index, source_label_index, source_categorical_covariate_index = (
                                    get_k_elements(source_X,0), 
                                    get_k_elements(source_X,1), 
                                    get_k_elements(source_X,2), 
                                    get_last_k_elements(source_X,3), 
                                    get_k_elements(source_X,4)
                                )
                                target_X, target_P, target_batch_index, target_label_index, target_categorical_covariate_index = (
                                    get_k_elements(target_X,0), 
                                    get_k_elements(target_X,1), 
                                    get_k_elements(target_X,2), 
                                    get_last_k_elements(target_X,3), 
                                    get_k_elements(target_X,4)
                                )
                            else:
                                X, batch_index, label_index, categorical_covariate_index = (
                                    get_k_elements(X,0), 
                                    get_k_elements(X,1),
                                    get_k_elements(X,2), 
                                    get_last_k_elements(X,3)
                                )
                                source_X, source_batch_index, source_label_index, source_categorical_covariate_index = (
                                    get_k_elements(source_X,0), 
                                    get_k_elements(source_X,1),
                                    get_k_elements(source_X,2), 
                                    get_last_k_elements(source_X,3)
                                )
                                target_X, target_batch_index, target_label_index, target_categorical_covariate_index = (
                                    get_k_elements(target_X,0), 
                                    get_k_elements(target_X,1),
                                    get_k_elements(target_X,2), 
                                    get_last_k_elements(target_X,3)
                                )
                        elif self.n_batch > 0:
                            if self.constrain_latent_embedding:
                                X, P, batch_index, categorical_covariate_index = (
                                    get_k_elements(X,0), 
                                    get_k_elements(X,1),  
                                    get_k_elements(X,2), 
                                    get_last_k_elements(X,3)
                                )
                                source_X, source_P, source_batch_index, source_categorical_covariate_index = (
                                    get_k_elements(source_X,0), 
                                    get_k_elements(source_X,1),  
                                    get_k_elements(source_X,2), 
                                    get_last_k_elements(source_X,3)
                                )
                                target_X, target_P, target_batch_index, target_categorical_covariate_index = (
                                    get_k_elements(target_X,0), 
                                    get_k_elements(target_X,1),  
                                    get_k_elements(target_X,2), 
                                    get_last_k_elements(target_X,3)
                                )
                            else:
                                X, batch_index, categorical_covariate_index = (
                                    get_k_elements(X,0), 
                                    get_k_elements(X,1), 
                                    get_last_k_elements(X,2)
                                )
                                source_X, source_batch_index, source_categorical_covariate_index = (
                                    get_k_elements(source_X,0), 
                                    get_k_elements(source_X,1), 
                                    get_last_k_elements(source_X,2)
                                )
                                target_X, target_batch_index, target_categorical_covariate_index = (
                                    get_k_elements(target_X,0), 
                                    get_k_elements(target_X,1), 
                                    get_last_k_elements(target_X,2)
                                )
                        elif self.n_label > 0:
                            if self.constrain_latent_embedding:
                                X, P, label_index, categorical_covariate_index = (
                                    get_k_elements(X,0), 
                                    get_k_elements(X,1), 
                                    get_k_elements(X,2), 
                                    get_last_k_elements(X,3)
                                )
                                source_X, source_P, source_label_index, source_categorical_covariate_index = (
                                    get_k_elements(source_X,0), 
                                    get_k_elements(source_X,1), 
                                    get_k_elements(source_X,2), 
                                    get_last_k_elements(source_X,3)
                                )
                                target_X, target_P, target_label_index, target_categorical_covariate_index = (
                                    get_k_elements(target_X,0), 
                                    get_k_elements(target_X,1), 
                                    get_k_elements(target_X,2), 
                                    get_last_k_elements(target_X,3)
                                )
                            else: 
                                X, label_index, categorical_covariate_index = (
                                    get_k_elements(X,0), 
                                    get_k_elements(X,1), 
                                    get_last_k_elements(X,2)
                                )
                                source_X, source_label_index, source_categorical_covariate_index = (
                                    get_k_elements(source_X,0), 
                                    get_k_elements(source_X,1), 
                                    get_last_k_elements(source_X,2)
                                )
                                target_X, target_label_index, target_categorical_covariate_index = (
                                    get_k_elements(target_X,0), 
                                    get_k_elements(target_X,1), 
                                    get_last_k_elements(target_X,2)
                                )
                        categorical_covariate_index = list(np.vstack(categorical_covariate_index).T.astype(int))
                        source_categorical_covariate_index = list(np.vstack(source_categorical_covariate_index).T.astype(int))
                        target_categorical_covariate_index = list(np.vstack(target_categorical_covariate_index).T.astype(int))
                    else:
                        if self.n_batch > 0 and self.n_label > 0:
                            if self.constrain_latent_embedding:
                                X, P, batch_index, label_index = (
                                    get_k_elements(X,0), 
                                    get_k_elements(X,1), 
                                    get_k_elements(X,2), 
                                    get_k_elements(X,3)
                                )
                                source_X, source_P, source_batch_index, source_label_index = (
                                    get_k_elements(source_X,0), 
                                    get_k_elements(source_X,1), 
                                    get_k_elements(source_X,2), 
                                    get_k_elements(source_X,3)
                                )
                                target_X, target_P, target_batch_index, target_label_index = (
                                    get_k_elements(target_X,0), 
                                    get_k_elements(target_X,1), 
                                    get_k_elements(target_X,2), 
                                    get_k_elements(target_X,3)
                                )
                            else:
                                X, batch_index, label_index = (
                                    get_k_elements(X,0), 
                                    get_k_elements(X,1), 
                                    get_k_elements(X,2)
                                )
                                source_X, source_batch_index, source_label_index = (
                                    get_k_elements(source_X,0), 
                                    get_k_elements(source_X,1), 
                                    get_k_elements(source_X,2)
                                )
                                target_X, target_batch_index, target_label_index = (
                                    get_k_elements(target_X,0), 
                                    get_k_elements(target_X,1), 
                                    get_k_elements(target_X,2)
                                )
                        elif self.n_batch > 0:
                            if self.constrain_latent_embedding:
                                X, P, batch_index = (
                                    get_k_elements(X,0), 
                                    get_k_elements(X,1),  
                                    get_k_elements(X,2)
                                )
                                source_X, source_P, source_batch_index = (
                                    get_k_elements(source_X,0), 
                                    get_k_elements(source_X,1),  
                                    get_k_elements(source_X,2)
                                )
                                target_X, target_P, target_batch_index = (
                                    get_k_elements(target_X,0), 
                                    get_k_elements(target_X,1),  
                                    get_k_elements(target_X,2)
                                )
                            else:
                                X, batch_index = (
                                    get_k_elements(X,0), 
                                    get_k_elements(X,1)
                                )
                                source_X, source_batch_index = (
                                    get_k_elements(source_X,0), 
                                    get_k_elements(source_X,1)
                                )
                                target_X, target_batch_index = (
                                    get_k_elements(target_X,0), 
                                    get_k_elements(target_X,1)
                                )
                        elif self.n_label > 0:
                            if self.constrain_latent_embedding:
                                X, P, label_index = (
                                    get_k_elements(X,0), 
                                    get_k_elements(X,1), 
                                    get_k_elements(X,2)
                                )
                                source_X, source_P, source_label_index = (
                                    get_k_elements(source_X,0), 
                                    get_k_elements(source_X,1), 
                                    get_k_elements(source_X,2)
                                )
                                target_X, target_P, target_label_index = (
                                    get_k_elements(target_X,0), 
                                    get_k_elements(target_X,1), 
                                    get_k_elements(target_X,2)
                                )

                            else: 
                                X, label_index = get_k_elements(X,0), get_k_elements(X,1)
                                source_X, source_label_index = get_k_elements(source_X,0), get_k_elements(source_X,1)
                                target_X, target_label_index = get_k_elements(target_X,0), get_k_elements(target_X,1)

                X = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, X))))
                source_X = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, source_X))))
                target_X = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, target_X))))
                if self.constrain_latent_embedding:
                    P = torch.tensor(np.array(P)).type(torch.FloatTensor).to(self.device)
                    source_P = torch.tensor(np.array(source_P)).type(torch.FloatTensor).to(self.device)

                if self.n_label > 0:
                    label_index = torch.tensor(label_index)
                    source_label_index = torch.tensor(source_label_index)
                    target_label_index = torch.tensor(target_label_index)
                    if not isinstance(label_index, torch.FloatTensor):
                        label_index = label_index.type(torch.FloatTensor)
                    if not isinstance(source_label_index, torch.FloatTensor):
                        source_label_index = source_label_index.type(torch.FloatTensor)
                    if not isinstance(target_label_index, torch.FloatTensor):
                        target_label_index = target_label_index.type(torch.FloatTensor)
                    label_index = label_index.to(self.device).unsqueeze(1)
                    source_label_index = source_label_index.to(self.device).unsqueeze(1)
                    target_label_index = target_label_index.to(self.device).unsqueeze(1)
                if self.n_batch > 0:
                    batch_index = torch.tensor(batch_index)
                    source_batch_index = torch.tensor(source_batch_index)
                    target_batch_index = torch.tensor(target_batch_index)
                    if not isinstance(batch_index, torch.FloatTensor):
                        batch_index = batch_index.type(torch.FloatTensor)
                    if not isinstance(source_batch_index, torch.FloatTensor):
                        source_batch_index = source_batch_index.type(torch.FloatTensor)
                    if not isinstance(target_batch_index, torch.FloatTensor):
                        target_batch_index = target_batch_index.type(torch.FloatTensor)
                    batch_index = batch_index.to(self.device).unsqueeze(1)
                    source_batch_index = source_batch_index.to(self.device).unsqueeze(1)
                    target_batch_index = target_batch_index.to(self.device).unsqueeze(1)
                if self.n_categorical_covariate_ is not None:
                    for i in range(len(categorical_covariate_index)):
                        categorical_covariate_index[i] = torch.tensor(categorical_covariate_index[i])
                        source_categorical_covariate_index[i] = torch.tensor(source_categorical_covariate_index[i])
                        target_categorical_covariate_index[i] = torch.tensor(target_categorical_covariate_index[i])
                        if not isinstance(categorical_covariate_index[i], torch.FloatTensor):
                            categorical_covariate_index[i] = categorical_covariate_index[i].type(torch.FloatTensor)
                        if not isinstance(source_categorical_covariate_index[i], torch.FloatTensor):
                            source_categorical_covariate_index[i] = source_categorical_covariate_index[i].type(torch.FloatTensor)
                        if not isinstance(target_categorical_covariate_index[i], torch.FloatTensor):
                            target_categorical_covariate_index[i] = target_categorical_covariate_index[i].type(torch.FloatTensor)
                        categorical_covariate_index[i] = categorical_covariate_index[i].to(self.device).unsqueeze(1)
                        source_categorical_covariate_index[i] = source_categorical_covariate_index[i].to(self.device).unsqueeze(1)
                        target_categorical_covariate_index[i] = target_categorical_covariate_index[i].to(self.device).unsqueeze(1)

                if not isinstance(X, torch.FloatTensor):
                    X = X.type(torch.FloatTensor)
                if not isinstance(source_X, torch.FloatTensor):
                    source_X = source_X.type(torch.FloatTensor)
                if not isinstance(target_X, torch.FloatTensor):
                    target_X = target_X.type(torch.FloatTensor)
                X = X.to(self.device)
                source_X = source_X.to(self.device)
                target_X = target_X.to(self.device)

                lib_size = X.sum(1).to(self.device)
                source_lib_size = source_X.sum(1).to(self.device)
                target_lib_size = target_X.sum(1).to(self.device)

                _, _, L = self.forward(
                    X, 
                    lib_size, 
                    batch_index, 
                    label_index, 
                    categorical_covariate_index, 
                    P, 
                    reduction=reconstruction_reduction
                )

                source_H, source_R, source_L = self.forward(
                    source_X, 
                    source_lib_size, 
                    source_batch_index, 
                    source_label_index, 
                    source_categorical_covariate_index, 
                    source_P, 
                    reduction=reconstruction_reduction
                )

                target_H, target_R, target_L = self.forward(
                    target_X, 
                    target_lib_size, 
                    target_batch_index, 
                    target_label_index, 
                    target_categorical_covariate_index, 
                    target_P, 
                    reduction=reconstruction_reduction
                )

                T = self.fc_t(source_H['q'])
                D = self.fc_d(source_H['q_mu']) + 1

                '''
                avg_transition_loss = -Normal(
                    target_H['q_mu'], target_H["q_var"].sqrt()
                ).log_prob(
                    source_H['q_mu'] + (T * D)
                ).sum(1).mean()
                '''
                
                avg_transition_loss = transition_weight * nn.MSELoss(reduction='sum')(target_H['q_mu'], source_H['q_mu'] + T) / n_per_batch

                reconstruction_loss = reconstruction_weight * L['reconstruction_loss']
                prediction_loss = pred_weight * L['prediction_loss'] 
                kldiv_loss = kl_weight * L['kldiv_loss']    
                mmd_loss = mmd_weight * L['mmd_loss']

                avg_reconstruction_loss = reconstruction_loss.sum()  / n_per_batch
                avg_kldiv_loss = kldiv_loss.sum()  / n_per_batch
                avg_mmd_loss = mmd_loss / n_per_batch

                epoch_reconstruction_loss += avg_reconstruction_loss.item()
                epoch_kldiv_loss += avg_kldiv_loss.item()
                epoch_transition_loss += avg_transition_loss.item()
                if self.n_label > 0:
                    epoch_prediction_loss += prediction_loss.sum().item()

                if epoch > max_epoch - pred_last_n_epoch:
                    loss = avg_reconstruction_loss + avg_kldiv_loss + avg_mmd_loss + prediction_loss.sum() + avg_transition_loss
                else: 
                    loss = avg_reconstruction_loss + avg_kldiv_loss + avg_mmd_loss + avg_transition_loss
    
                if self.constrain_latent_embedding:
                    loss += constrain_weight * L['latent_constrain_loss']

                epoch_total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({
                    'step': f'{b} / {len(X_train)}',
                    'transition_loss': '{:.2e}'.format(epoch_transition_loss_final),
                })
            epoch_transition_loss_final = epoch_transition_loss
            '''
            loss_record = self.calculate_metric(X_test, kl_weight, pred_weight, mmd_weight)
            if lr_schedule:
                scheduler.step(loss_record["epoch_total_loss"])
            pbar.set_postfix({
                'reconst': '{:.2e}'.format(loss_record["epoch_reconstruction_loss"]),
                'kldiv': '{:.2e}'.format(loss_record["epoch_kldiv_loss"]),
                'mmd': '{:.2e}'.format(loss_record["epoch_mmd_loss"]),
            })
            '''
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
    def get_latent_embedding_transition(self, latent_key: Literal["z", "q_mu"] = "q_mu", n_per_batch: int = 128) -> np.ndarray:
        X = self.as_dataloader(subset_indices = list(range(len(self._dataset))), shuffle=False, n_per_batch=n_per_batch)
        Zs = []
        Ts, Ds = [], []
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
            Ts.append(self.fc_t(H['q']).detach().cpu().numpy())
            Ds.append(self.fc_d(H['q_mu']).detach().cpu().numpy() + 1)
        return np.vstack(Zs)[self._shuffle_indices], np.vstack(Ts)[self._shuffle_indices], np.vstack(Ds)[self._shuffle_indices]
    