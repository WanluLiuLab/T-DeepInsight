import scanpy as sc
import matplotlib.pyplot as plt 

def plot_selected_tcrs(
    adata,
    tcrs,
    color_key,
    color,
    **kwargs
):
    fig,ax=createFig()
    fig.set_size_inches(3,3)

    ax.scatter(
        tcr_adata.obsm["X_umap"][:,0],
        tcr_adata.obsm["X_umap"][:,1],
        s=0.1,
        color=list(map(lambda x: reannotated_prediction_palette[x], tcr_adata.obs['reannotated_prediction'])),
        linewidths=0,
        alpha=0.2,
    )

    obsm = tcr_adata[
            np.array(list(map(lambda x: x in tcrs,tcr_adata.obs['tcr'])))
    ].obsm["X_umap"]

    ax.scatter(
        obsm[:,0],
        obsm[:,1],
        s=10,
        marker='*',
        color='red',
    )
