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
    return cpuUMAP(
        min_dist=0.1, 
        spread=1.5, 
        local_connectivity=1.5, 
        verbose=True, 
        repulsion_strength=1.2,
        **kwargs
    )

def get_constrained_umap_reducer():
    return cpuUMAP(
        min_dist=0.1, 
        spread=1.5, 
        local_connectivity=1.5, 
        verbose=True, 
        repulsion_strength=1.2,
        target_metric='euclidean',
    )

def get_default_cuml_reducer(**kwargs):
    if torch.cuda.is_available():
        return cumlUMAP(
            min_dist=0.1, 
            spread=1.5, 
            local_connectivity=1.5, 
            verbose=True, 
            repulsion_strength=1.2,
            **kwargs
        )
    else:
        raise Exception('CUDA is not available. Please install CUDA and cuml to use cumlUMAP or use get_default_umap_reducer')

def get_default_reducer():
    if torch.cuda.is_available():
        return get_default_cuml_reducer()
    else:
        return get_default_umap_reducer()
    
