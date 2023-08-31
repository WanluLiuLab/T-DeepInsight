import torch

umap_is_installed = False
try:
    from umap import UMAP as cpuUMAP
    umap_is_installed = True
except ImportError:
    pass


cuml_is_installed = False

try:
    import cuml
    from cuml.manifold import UMAP as cumlUMAP
    cuml_is_installed = True
except ImportError:
    pass


def get_default_umap_reducer(**kwargs):
    if 'min_dist' not in kwargs:
        kwargs['min_dist'] = 0.1
    if 'spread' not in kwargs:
        kwargs['spread'] = 1.5
    if 'local_connectivity' not in kwargs:
        kwargs['local_connectivity'] = 1.5
    if 'verbose' not in kwargs:
        kwargs['verbose'] = True
    if 'repulsion_strength' not in kwargs:
        kwargs['repulsion_strength'] = 1.2
    return cpuUMAP(
        **kwargs
    )

def get_constrained_umap_reducer(**kwargs):
    if 'min_dist' not in kwargs:
        kwargs['min_dist'] = 0.1
    if 'spread' not in kwargs:
        kwargs['spread'] = 1.5
    if 'local_connectivity' not in kwargs:
        kwargs['local_connectivity'] = 1.5
    if 'verbose' not in kwargs:
        kwargs['verbose'] = True
    if 'repulsion_strength' not in kwargs:
        kwargs['repulsion_strength'] = 1.2
    return cpuUMAP(
        target_metric='euclidean',
        **kwargs
    )

def get_default_cuml_reducer(**kwargs):
    if 'min_dist' not in kwargs:
        kwargs['min_dist'] = 0.1
    if 'spread' not in kwargs:
        kwargs['spread'] = 1.5
    if 'local_connectivity' not in kwargs:
        kwargs['local_connectivity'] = 1.5
    if 'verbose' not in kwargs:
        kwargs['verbose'] = True
    if 'repulsion_strength' not in kwargs:
        kwargs['repulsion_strength'] = 1.2

    if torch.cuda.is_available():
        return cumlUMAP(
            **kwargs
        )
    else:
        raise Exception('CUDA is not available. Please install CUDA and cuml to use cumlUMAP or use get_default_umap_reducer')

def get_default_reducer():
    if torch.cuda.is_available():
        return get_default_cuml_reducer()
    else:
        return get_default_umap_reducer()
    
