import torch
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence as kldiv
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import rbf_kernel

from torch.autograd import Variable

from ._logger import *
from ._distributions import *

# Reference: https://github.com/tim-learn/ATDOC/blob/main/loss.py
class MMD(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X - f_of_Y
        loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss

class LossFunction:
    @staticmethod
    def mse(recon_x:   torch.tensor, 
            x:         torch.tensor, 
            reduction: str = "sum"):
        """
        The reconstruction error in the form of mse
        """
        return F.mse_loss(recon_x, x, reduction = reduction)

    @staticmethod
    def bce(recon_x:   torch.tensor, 
            x:         torch.tensor, 
            reduction: str = "sum"):
        """
        The reconstruction error in the form of mse
        """
        return F.binary_cross_entropy(recon_x, x)

    @staticmethod
    def vae_mse(recon_x: torch.tensor, 
                x:       torch.tensor, 
                mu:      torch.tensor, 
                var:     torch.tensor, 
                kld_weight: float = 0.5):
        """
        The KL-divergence of the latent probability z
        KL(q || p) = -∫ q(z) log [ p(z) / q(z) ] 
                = -E[log p(z) - log q(z)] 
        """
        MSE = F.mse_loss(recon_x, x, reduction = "sum")
        KLD = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1), dim = 0)

        return MSE + KLD


        
    @staticmethod
    def mmd_loss(x: torch.tensor, y: torch.tensor):
        return MMD()(x,y)
        
    @staticmethod
    def zinb_reconstruction_loss(X:            torch.tensor, 
                                 total_counts: torch.tensor = None,
                                 logits:       torch.tensor = None,
                                 mu:           torch.tensor = None,
                                 theta:        torch.tensor = None,
                                 gate_logits:  torch.tensor = None,
                                 reduction:    str = "sum"):
        if ((total_counts == None) and (logits == None)):
            if ((mu == None) and (theta == None )):
                raise ValueError
            logits = (mu / theta).log()
            total_counts = theta + 1e-6
            znb = ZeroInflatedNegativeBinomial(
                total_count=total_counts, 
                logits=logits,
                gate_logits=gate_logits
            )   
        else: 
            znb = ZeroInflatedNegativeBinomial(
                total_count=total_counts, 
                logits=logits, 
                gate_logits=gate_logits
            )
        if reduction == "sum":
            reconst_loss = -znb.log_prob(X).sum(dim = 1)
        elif reduction == "mean":
            reconst_loss = -znb.log_prob(X).mean(dim = 1)
        return reconst_loss

    @staticmethod
    def nb_reconstruction_loss(X:            torch.tensor, 
                                 total_counts: torch.tensor = None,
                                 logits:       torch.tensor = None,
                                 mu:           torch.tensor = None,
                                 theta:        torch.tensor = None):
        if ((total_counts == None) and (logits == None)):
            if ((mu == None) and (theta == None )):
                raise ValueError
            logits = (mu + 1e-6) - (theta + 1e-6).log()
            total_counts = theta 
        
        nb = NegativeBinomial(
            total_count=total_counts, 
            logits=logits, 
        )
        reconst_loss = -nb.log_prob(X).sum(dim = 1)
        return reconst_loss

    

    @staticmethod
    def kld(q: torch.tensor,
            p: torch.tensor):
        kl_loss = kldiv(q.log(), p, reduction="sum", log_target=False)
        kl_loss.requires_grad_(True)
        return kl_loss

    @staticmethod
    def kl1(mu:  torch.tensor, 
            var: torch.tensor):
        return kldiv(Normal(mu, torch.sqrt(var)), Normal(0, 1)).sum(dim=1)

    @staticmethod
    def kl2(mu1: torch.tensor, 
           var1: torch.tensor, 
           mu2:  torch.tensor, 
           var2: torch.tensor):
        return kldiv(Normal(mu1, var1.sqrt()), Normal(mu2, var2.sqrt()))

    @staticmethod
    def soft_cross_entropy(pred:         torch.tensor, 
                           soft_targets: torch.tensor):
        # use nn.CrossEntropyLoss if not using soft labels
        logsoftmax = nn.LogSoftmax(dim=1)
        return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

    

def trvae_partition(data, partitions, num_partitions):
    res = []
    partitions = partitions.flatten()
    for i in range(num_partitions):
        indices = torch.nonzero((partitions == i), as_tuple=False).squeeze(1)
        res += [data[indices]]
    return res

def pairwise_distance(x, y):
    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output
    
def gaussian_kernel_matrix(x, y, alphas):
    """Computes multiscale-RBF kernel between x and y.

       Parameters
       ----------
       x: torch.Tensor
            Tensor with shape [batch_size, z_dim].
       y: torch.Tensor
            Tensor with shape [batch_size, z_dim].
       alphas: Tensor

       Returns
       -------
       Returns the computed multiscale-RBF kernel between x and y.
    """

    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)

    alphas = alphas.view(alphas.shape[0], 1)
    beta = 1. / (2. * alphas)

    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def mmd_loss_calc(source_features, target_features):
    """Initializes Maximum Mean Discrepancy(MMD) between source_features and target_features.

       - Gretton, Arthur, et al. "A Kernel Two-Sample Test". 2012.

       Parameters
       ----------
       source_features: torch.Tensor
            Tensor with shape [batch_size, z_dim]
       target_features: torch.Tensor
            Tensor with shape [batch_size, z_dim]

       Returns
       -------
       Returns the computed MMD between x and y.
    """
    alphas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    alphas = Variable(torch.FloatTensor(alphas)).to(device=source_features.device)

    cost = torch.mean(gaussian_kernel_matrix(source_features, source_features, alphas))
    cost += torch.mean(gaussian_kernel_matrix(target_features, target_features, alphas))
    cost -= 2 * torch.mean(gaussian_kernel_matrix(source_features, target_features, alphas))

    return cost

def trvae_mmd(y, c, n_conditions, beta = 1, boundary = None):
    """Initializes Maximum Mean Discrepancy(MMD) between every different condition.

       Parameters
       ----------
       n_conditions: integer
            Number of classes (conditions) the data contain.
       beta: float
            beta coefficient for MMD loss.
       boundary: integer
            If not 'None', mmd loss is only calculated on #new conditions.
       y: torch.Tensor
            Torch Tensor of computed latent data.
       c: torch.Tensor
            Torch Tensor of condition labels.

       Returns
       -------
       Returns MMD loss.
    """

    # partition separates y into num_cls subsets w.r.t. their labels c
    conditions_mmd = trvae_partition(y, c, n_conditions)

    loss = torch.tensor(0.0, device=y.device)
    if boundary is not None:
        for i in range(boundary):
            for j in range(boundary, n_conditions):
                if conditions_mmd[i].size(0) < 2 or conditions_mmd[j].size(0) < 2:
                    continue
                loss += mmd_loss_calc(conditions_mmd[i], conditions_mmd[j])
    else:
        for i in range(len(conditions_mmd)):
            if conditions_mmd[i].size(0) < 1:
                continue
            for j in range(i):
                if conditions_mmd[j].size(0) < 1 or i == j:
                    continue
                loss += mmd_loss_calc(conditions_mmd[i], conditions_mmd[j])

    return beta * loss
