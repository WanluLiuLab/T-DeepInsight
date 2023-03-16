# TCR-DeepInsight

The emergence of single-cell immune profiling technology has led to the production of a large amount of data on single-cell gene expression (GEX) and T cell receptor (TCR), which has great potential for studying TCR biology and identifying effective TCRs. However, one of the major challenges is the lack of a reference atlas that provides easy access to these datasets. On the other hand, the use of TCR engineering in disease immunotherapy is rapidly advancing, and single-cell immune profiling data can be a valuable resource for identifying functional TCRs. Nevertheless, the lack of efficient computational tools to integrate and identify functional TCRs is a significant obstacle in this field.

## Methods 
Hardware requirement for TCR-DeepInsight includes
1. RAM: >16Gb for larger dataset
2. VRAM of CUDA-enabled GPU: >8Gb 


Operation System requirements for running TCR-DeepInsight include the installation of Python3 (Python3.8 used for development) and several PyPI packages. You can create a running environment using CONDA

```shell
conda create -n tcrdeepinsight python=3.8 -y    
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install faiss-gpu transformers==4.17.0 datasets==1.18.4 scikit-learn==0.24.1 scanpy==1.8.1 anndata==0.8.0 matplotlib=3.3.4 einops==0.4.1 biopython==1.79
```

## Data 

`Cellranger-6.1.2` is used for the processing of raw 10x GEX and TCR sequencing data as described in [Wu et al., 2021](https://academic.oup.com/nar/article/50/D1/D1244/6381136). However, TCR-DeepInsight is compatible with cellranger with version > 3.0.0.

Integration of GEX and TCR data is performed by `scanpy` and `scirpy`. Specifically, TCR-DeepInsight requires cells with full-length TCR and complement information of CDR3α, TRAV, TRAJ, CDR3β, TRBV, and TRBJ.

Example code for integrating and filtering GEX and TCR data:

```python
import scanpy as sc
import scirpy as ir
ir.pp.merge_with_ir(gex_adata, tcr_adata)
if isinstance(tcr_adata_opt, sc.AnnData):
    gex_adata.obs['is_cell'] = 'None'
    gex_adata.obs['high_confidence'] = 'None'
    tcr_adata_opt.obs['is_cell'] = 'None'
    tcr_adata_opt.obs['high_confidence'] = 'None'
    ir.pp.merge_airr_chains(gex_adata, tcr_adata_opt)
elif isinstance(tcr_adata_opt, Iterable):
    for i in tcr_adata_opt:
        gex_adata.obs['is_cell'] = 'None'
        gex_adata.obs['high_confidence'] = 'None'
        i.obs['is_cell'] = 'None'
        i.obs['high_confidence'] = 'None'
        ir.pp.merge_airr_chains(gex_adata, i)
ir.tl.chain_qc(gex_adata)
if self.check_valid_vdj:
    gex_adata = gex_adata[
        list(map(lambda x: x in ["single pair", "extra VJ", "extra VDJ"], gex_adata.obs["chain_pairing"]))
    ].copy()
```

## Usage

Please see the Jupyter Notebook for TCR-DeepInsight.