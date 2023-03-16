import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Union, Mapping, Iterable, List, Optional
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch import nn, einsum

from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat

import numpy as np

from ..utils._loss import LossFunction
from ..utils._logger import  mt
from ..utils._definitions import FCDEF
from ..utils._tensor_utils import one_hot
from ..utils._compat import Literal

class ReparameterizeLayerBase(nn.Module):
    """Base layer for VAE reparameterization"""
    def __init__(self) -> None:
        """
        @brief ReparameterizeLayer Defining methods for reparameterize tricks
        """
        super(ReparameterizeLayerBase, self).__init__()

    def reparameterize(
        self, 
        mu: torch.Tensor, 
        var: torch.Tensor
    ) -> torch.Tensor :
        untran_z = Normal(mu, var.sqrt()).rsample()
        return untran_z

    def reparameterize_transformation(
        self, 
        transfunc, 
        mu: torch.Tensor, 
        var: torch.Tensor
    ) -> torch.Tensor:
        z = self.reparameterize(mu, var)
        ztrans = transfunc(z)
        return ztrans, z

class MMDLayerBase:
    """Base layer for Maximum-mean descrepancy calculation"""
    def HierarchicalMMDLoss(
        self, 
        z: torch.Tensor, 
        cat: np.array, 
        hierarchical_weight: Iterable[float]
    ) -> torch.Tensor:
        if len(cat.shape) <= 1:
            raise ValueError("Illegal category array")
        if len(z) != cat.shape[0]:
            raise ValueError("Dimension of z {} should be equal to dimension of category {}".format(len(z), cat.shape[0]))
        if len(hierarchical_weight) != cat.shape[1]:
            raise ValueError("Dimension of hierarchical_weight {} should be equal to dimension of category {}".format(len(hierarchical_weight), cat.shape[1]))

        if cat.shape[1] < 2:
            cat = cat.flatten()
            return self.MMDLoss(z, cat)
        loss = 0
        zs = []
        for i in np.unique(cat[:, 0]):
            idx = list(map(lambda t:t[0], filter(lambda x:x[1] == i, enumerate(cat[:,0]))))
            loss += self.HierarchicalMMDLoss(z[idx], cat[idx,1:], hierarchical_weight[1:])
            zs.append(z[idx])
        for i in range(len(np.unique(cat[:,0]))):
            for j in range(i+1, len(np.unique(cat[:,0]))):
                loss += LossFunction.mmd_loss(
                    zs[i], zs[j]
                )
        return loss

    def MMDLoss(self, z: torch.Tensor, cat: np.array) -> torch.Tensor:
        zs = []
        loss = 0
        for i in np.unique(cat):
            idx = list(map(lambda z:z[0], filter(lambda x:x[1] == i, enumerate(cat))))
            zs.append(z[idx])
        for i in range(len(np.unique(cat))):
            for j in range(i+1,len(np.unique(cat))):
                loss += LossFunction.mmd_loss(
                    zs[i], zs[j]
                )
        return loss


class FCLayer(nn.Module):
    """FCLayer Fully-Connected Layers for a neural network """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_cat_list: Iterable[int] = None,
        cat_dim: int = 8,
        cat_embedding: Literal["embedding", "onehot"] = "onehot",
        bias: bool = True,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        activation_fn: nn.Module = nn.ReLU,
        activation_dim: int = None,
        device: str = "cuda"
    ):
        super(FCLayer, self).__init__()
        if n_cat_list is not None:
            # Categories
            if not all(map(lambda x: x > 1, n_cat_list)):
                raise ValueError("category list contains values less than 1")
            self.n_category = len(n_cat_list)
            self._cat_dim = cat_dim
            self.cat_dimension = self.n_category * cat_dim # Total dimension of categories using one-hot encoding
            self.n_cat_list = n_cat_list
            if cat_embedding == "embedding":
                self.cat_embedding = nn.ModuleList(
                    [nn.Embedding(n, cat_dim) for n in n_cat_list]
                )
            else: 
                self.cat_embedding = [
                    lambda x: one_hot(x.unsqueeze(1), n) for n in n_cat_list
                ]

        else:
            # No categories will be included
            self.n_category = 0
            self.n_cat_list = None
        
        self._fclayer = nn.Sequential(
                *list(filter(lambda x:x, 
                        [
                            nn.Linear(in_dim, out_dim, bias=bias) 
                            if self.n_category == 0 
                            else nn.Linear(in_dim + self.cat_dimension, out_dim, bias=bias),
                            nn.BatchNorm1d(out_dim, momentum=0.01, eps=0.001) if use_batch_norm else None,
                            nn.LayerNorm(out_dim, elementwise_affine=False) if use_layer_norm else None,
                            activation_fn(dim=activation_dim) if activation_dim else activation_fn() if activation_fn else None,
                            nn.Dropout(p = dropout_rate) if dropout_rate > 0 else None
                        ]
                    )
                )
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device 

    def forward(self, X: torch.Tensor, cat_list: torch.Tensor =  None):
        category_embedding = []
        if self.n_category > 0:
            if cat_list != None:
                if (len(cat_list) != self.n_category):
                    raise ValueError("Number of category list should be equal to {}".format(self.n_category))

                for i, (n_cat, cat) in enumerate(zip(self.n_cat_list, cat_list)):
                    assert(n_cat > 1)
                    category_embedding.append(self.cat_embedding[i](cat))
            else:
                if X.shape[1] != self.in_dim + self.n_category:
                    raise ValueError("Dimension of X should be equal to {} + {} if cat_list is provided".format(self.in_dim, self.n_category))
                cat_list = X[:, -self.n_category:].type(torch.LongTensor).T.to(self.device)
                for i, (n_cat, cat) in enumerate(zip(self.n_cat_list, cat_list)):
                    assert(n_cat > 1)
                    category_embedding.append(self.cat_embedding[i](cat))
               
            category_embedding = torch.hstack(category_embedding).to(self.device)
            return self._fclayer(torch.hstack([X[:,:self.in_dim], category_embedding]))
        else:
            return self._fclayer(X)

    def to(self, device:str):
        super(FCLayer, self).to(device)
        self.device=device 
        return self


class PredictionLayerBase(nn.Module):
    """Prediction layer base """
    def __init__(self, *, in_dim:int, n_pred_category: int):
        super(PredictionLayer, self).__init__()
        self.in_dim = in_dim
        self.n_pred_category = n_pred_category
        self.decoder = FCLayer(
            in_dim = in_dim,
            out_dim = n_pred_category,
            bias = False,
            dropout_rate = 0,
            use_batch_norm = False,
            use_layer_norm = False,
            activation_fn = nn.ReLU,
        )

    def forward(self, X: torch.Tensor):
        return nn.Softmax(-1)( self.decoder(X) )

class SAE(nn.Module):
    ''' Stacked Autoencoders. 
        Fitting includes stacked fitting and fine-tuning:
            Fine-tuning step removes the decoder and use clustering method
            to fine-tune the encoder.
        parameters:
            dim:    int 
            stacks: Iterable[int]
            n_cat_list: Iterable[int]
            cat_dim: int
    '''
    def __init__(
            self, 
            dim:int, 
            stacks:Iterable[int] = [512, 128, 64], 
            n_cat_list: Iterable[int] = None,
            cat_dim: int = 8,
            bias: bool = True,
            dropout_rate: float = 0.1,
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
            cat_embedding: Literal["embedding", "onthot"] = "onehot",
            activation_fn: nn.Module = nn.ReLU,
            encode_only: bool = False,
            decode_only: bool = False,
            device="cuda"
    ):
        super(SAE, self).__init__()
        fcargs = dict(
            bias=bias, 
            dropout_rate=dropout_rate, 
            use_batch_norm=use_batch_norm, 
            use_layer_norm=use_layer_norm,
            activation_fn=activation_fn,
            device=device,
            cat_embedding = cat_embedding
        )
        self.dim = dim
        self.num_layers = len(stacks)
        self.n_cat_list = n_cat_list
        self.cat_dim = cat_dim
        self.n_category = len(n_cat_list) if n_cat_list != None else 0
        self.stacks = stacks
        layers = [None] * len(stacks)
        self.n_layers = len(stacks)
        if (encode_only & decode_only):
            raise ValueError("SAE instance cannot be both encode and decode only")
        for i,j in enumerate(stacks):
            if i == 0:
                layers[i] = [FCLayer(dim, 
                             stacks[i], 
                             n_cat_list, 
                             cat_dim,
                             **fcargs)
                             if not decode_only 
                             else None, 
                             FCLayer(stacks[i], dim, **fcargs) 
                             if not encode_only 
                             else None]
            else:
                layers[i] = [FCLayer(stacks[i-1], stacks[i], **fcargs)
                             if not decode_only 
                             else None, 
                             FCLayer(stacks[i], stacks[i-1], **fcargs) 
                             if not encode_only 
                             else None ]
        layers = [i for s in layers for i in s]
        self.layers = nn.ModuleList(layers)
        self.device = device
        self.loss = []
        self.encode_only = encode_only
        self.decode_only = decode_only

    def get_layer(self, codec:str, layer:int):
        i = 0 if codec == FCDEF.ENCODER else 1
        return self.layers[layer * 2 + i]

    def encode(self, x: torch.Tensor):
        '''
        encode features in the nth layer 
        '''
        if self.decode_only:
            raise TypeError("This is an decoder-only SAE instance")
        h = None
        for i in range(self.num_layers):
            layer = self.get_layer(FCDEF.ENCODER, i)
            if i == self.num_layers - 1:
                if i == 0:
                    h = layer(x)
                else:
                    h = layer(h)
            else:
                if i == 0: 
                    h = layer(x)
                else:
                    h = layer(h)
        return h
    
    def decode(self, z: torch.Tensor):
        '''
        decode features in the nth layer 
        '''
        if self.encode_only:
            raise TypeError("This is an encoder-only SAE instance")
        h = None
        for i in range(self.num_layers):
            layer = self.get_layer(FCDEF.DECODER, self.num_layers - 1 - i)
            if i == self.num_layers - 1:
                if i == 0:
                    h = layer(z)
                else:
                    h = layer(h)
            else:
                if i == 0:
                    h = layer(z)
                else:
                    h = layer(h)
        return h

    def forward(self, x: torch.Tensor):
        z = self.encode(x.view(-1, self.dim))
        return self.decode(z), z

    def fit(self, X_train, max_epoch, minibatch = True, lr = 1e-3):
        optimizer = optim.Adam(self.parameters(), lr, weight_decay = 1e-3);
        scheduler =  ReduceLROnPlateau(optimizer, mode="min", patience=10)
        if minibatch:
            for epoch in range(1, max_epoch+1):
                epoch_total_loss = 0
                for batch_index, X in enumerate(X_train):
                    if X.device.type != self.device:
                        X = X.to(self.device)
                    recon_batch, hidden_batch = self.forward(X)
                    loss = LossFunction.mse(recon_batch, X)
                    epoch_total_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss)
                    del(X) # Reduce memory 
                    torch.cuda.empty_cache() 

                mt('Epoch: {} Average loss: {:.8f}'.format(epoch, epoch_total_loss))
                self.loss.append(epoch_total_loss)
        else:
            for epoch in range(1, max_epoch+1):
                epoch_total_loss = 0
                if X_train.device.type != self.device:
                    X_train = X_train.to(self.device)
                recon_batch, hidden_batch = self.forward(X_train)
                loss = LossFunction.mse(recon_batch, X_train)
                epoch_total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                torch.cuda.empty_cache() 
                mt('Epoch: {} Average loss: {:.8f}'.format(epoch, epoch_total_loss))
                self.loss.append(epoch_total_loss)
    def to(self, device:str):
        super(SAE, self).to(device)
        self.device=device 
        return self

class PositionalEncoding(nn.Module):
    def __init__(self, *,
                       n_hiddens: int, 
                       dropout: float, 
                       max_len:int=1000
    ) -> None:
        super(PositionalEncoding, self).__init__()
        assert(n_hiddens % 2 == 0)
        self.n_hiddens = n_hiddens
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, n_hiddens))
        X = rearrange(torch.arange(max_len, dtype=torch.float32),
            '(n d) -> n d', d = 1) / torch.pow(1e4, torch.arange(
            0, n_hiddens, 2, dtype=torch.float32) / n_hiddens)
        self.P[:,:,0::2] = torch.sin(X)
        self.P[:,:,1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :X.shape[2]].to(X.device)
        return self.dropout(X)

class FCSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head,
        n_heads
    ):
        super().__init__()
        inner_dim = dim_head * n_heads
        self.n_heads = n_heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, X, mask = None):
        """ X.shape = (batch, tokens, dim)
        """
        h = self.n_heads
        q, k, v = self.to_qkv(X).chunk(3, dim = -1)
        # q.shape = (batch, head, tokens, inner_dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        # attention.shape = (batch, head, tokens, tokens)
        attention = einsum('b h n d, b h m d -> b h n m', k, q) * self.scale
        if mask != None:
            if mask.shape != attention.shape:
                mask = rearrange(mask, 'n h -> n () h ()')
            attention = attention.masked_fill(mask, -torch.finfo(attention.dtype).max)
        attention = torch.softmax(attention, dim=-1)
        # out.shape = (batch, head, inner_dim, tokens)
        out = einsum('b h n e, b h e d -> b h d n', attention, v)
        # out.shape = (batch, tokens, head * inner_dim)
        out = rearrange(out, 'b h d n -> b n (h d)')
        return self.to_out(out)

class AddNorm(nn.Module):
    """Residual connection followed by layer normalization.""" 
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs) 
        self.dropout = nn.Dropout(dropout) 
        self.ln = nn.LayerNorm(normalized_shape)
    
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

# reference: https://github.com/daiquocnguyen/Graph-Transformer/blob/master/UGformerV2_PyTorch/UGformerV2.py
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_dim, out_dim, activation = torch.relu, bias:bool = False) -> None:
        super(GraphConvolution, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 
        self.weight = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.bn = nn.BatchNorm1d(out_dim)
        self.has_bias = bias

    def forward(self, X, A):
        Z = self.weight(X)
        output = torch.mm(A, Z)
        output = self.bn(output)
        return self.activation(output)

# reference: https://github.com/daiquocnguyen/Graph-Transformer/blob/master/UGformerV2_PyTorch/UGformerV2.py    
class FullyConnectedGraphTransformer(nn.Module):
    def __init__(self, feature_dim,
                       ff_hidden_size,
                       n_self_att_layers,
                       dropout,
                       n_GNN_layers,
                       n_head, device = "cuda"
                       ) -> None:
        super(FullyConnectedGraphTransformer, self).__init__()
        self.feature_dim = feature_dim
        self.ff_hidden_size = ff_hidden_size
        self.n_self_att_layers = n_self_att_layers
        self.n_GNN_layers = n_GNN_layers
        self.n_head = n_head
        self.GNN_layers = nn.ModuleList()
        self.selfAtt_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.node_embedding = nn.Linear(feature_dim * n_GNN_layers, feature_dim, bias = True)
        self.device = device
        for _ in range(self.n_GNN_layers):
            encoder_layer = nn.TransformerEncoderLayer(self.feature_dim, 
                nhead=self.n_head, 
                dim_feedforward=self.ff_hidden_size,
                dropout=0.5)
            self.selfAtt_layers.append(
                nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.n_self_att_layers).to(device))
            self.GNN_layers.append(GraphConvolution(self.feature_dim, self.feature_dim, torch.relu, True).to(device))
            self.dropouts.append(nn.Dropout(dropout))
    
    def reset_parameters(self):
        for i in self.selfAtt_layers:
            i.reset_parameters()
        self.prediction.reset_parameters()
        for i in self.dropouts:
            i.reset_parameters()

    def forward(self, X, A):
        Xs = []
        for i in range(self.n_GNN_layers):
            # self attention over all nodes
            X = X.unsqueeze(1)
            X = self.selfAtt_layers[i](X)
            X = X.squeeze(1)
            X = self.GNN_layers[i](X, A)
            Xs.append(X)
        X = torch.hstack(Xs)
        X = self.node_embedding(X)
        return X

    def to(self, device:str):
        super(FullyConnectedGraphTransformer, self).to(device)
        self.device=device 
        return self