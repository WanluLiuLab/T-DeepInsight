import os
import argparse
import sys
import numpy as np
import random
import pandas as pd
from typing import Iterable
import torch
import torch.nn.functional as F
from collections import Counter
import scanpy as sc
from ._logger import Colors

def multi_values_dict(keys, values):
    ret = {}
    for k,v in zip(keys, values):
        if k not in ret.keys():
            ret[k] = [v]
        else:
            ret[k].append(v)
    return ret


def random_subset_by_key(adata, key, n):
    from collections import Counter
    counts = {k:v/len(adata) for k,v in Counter(adata.obs[key]).items()}
    ns = [(k,int(v*n)) for k,v in counts.items()]
    adatas = []
    for k,v in ns:
        view = adata[adata.obs[key] == k]
        adatas.append(
            view[np.random.choice(list(range(len(view))), v, replace=False)]
        )
    return sc.concat(adatas)


def exists(x):
    return x != None

def sliceSimutaneuously(a, index):
    return pd.DataFrame(a).iloc[index,index].to_numpy()

def mask_split(tensor, indices):
    sorter = torch.argsort(indices)
    _, counts = torch.unique(indices, return_counts=True)
    return torch.split(tensor[sorter], counts.tolist())


def print_version():
    print(Colors.YELLOW)
    print('Python VERSION:{}\n'.format(Colors.NC), sys.version)
    print(Colors.YELLOW)
    print('PyTorch VERSION:{}\n'.format(Colors.NC), torch.__version__)
    print(Colors.YELLOW)
    print('CUDA VERSION{}\n'.format(Colors.NC))
    from subprocess import call
    try: call(["nvcc", "--version"])
    except: pass
    print(Colors.YELLOW)
    print('CUDNN VERSION:{}\n'.format(Colors.NC), torch.backends.cudnn.version())
    print(Colors.YELLOW)
    print('Number CUDA Devices:{}\n'.format(Colors.NC), torch.cuda.device_count())
    try:
        print('Devices             ')
        call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    except FileNotFoundError:
        # There is no nvidia-smi in this machine
        pass
    if torch.cuda.is_available():
        print('Active CUDA Device: GPU', torch.cuda.current_device())
        print ('Available devices     ', torch.cuda.device_count())
        print ('Current cuda device   ', torch.cuda.current_device())
    else:
        # cuda not available
        pass

def read_tsv(path, header:bool = True, skip_first_line: bool = False, return_pandas: bool = True):
    result = []
    if os.path.exists(path):
        f = open(path)
        if skip_first_line:
            line = f.readline()
        header_length = None
        if header:
            header = f.readline().strip().split('\t')
            header_length = len(header)

        while 1:
            line = f.readline()
            if not line:
                break
            line = line.strip().split('\t')
            if not header_length:
                header_length = len(line)
            result.append(line[:header_length])
        f.close()
        if return_pandas:
            if header:
                return pd.DataFrame(result, columns = header)
            else:
                return pd.DataFrame(result)
        else:
            return result
    else:
        it = iter(path.split('\n'))
        if skip_first_line:
            line = next(it)
        header_length = None
        if header:
            header = next(it).strip().split('\t')
            header_length = len(header)

        while 1:
            try:
                line = next(it)
                if not line:
                    break
                line = line.strip().split('\t')
                if not header_length:
                    header_length = len(line)
                result.append(line[:header_length])
            except:
                break 
        if return_pandas:
            if header:
                return pd.DataFrame(list(filter(lambda x: len(x) == 125, result)), columns = header)
            else:
                return pd.DataFrame(list(filter(lambda x: len(x) == 125, result)))
        else:
            return result
