from scanpy.tools import *
from ._deep_insight import *
from ._deep_insight_result import *
import umap as umap

cuml_is_installed = False
try:
    import cuml
    cuml_is_installed = True
except ImportError:
    pass

