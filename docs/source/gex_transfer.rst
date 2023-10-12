Transfering Multi-source Gene Expression (GEX)
==============================================

This is a repository for the code for transfering multi-source gene expression (GEX) data using the VAE model from the TCR-DeepInsight package.

.. code-block:: python
  :linenos:

  import t_deep_insight as tdi 

  # Load the data
  query_adata = tdi.read_h5ad("query_adata.h5ad")

The `adata` is a :class:`anndata.AnnData` object with raw GEX count matrix stored in adata.X.
To transfer the GEX data, we first need to build a VAE model with previously trained model parameters weights.
We need to make sure that the number of genes in the new data is the same as the number of genes in the training data. If not, please see the `Retraining Multi-source GEX Data <gex_retraining.html>`_ tutorial for how to transfer GEX data with different number of genes.

.. code-block:: python
  :linenos:

  tdi.model.VAEModel.setup_anndata(query_adata, "model.pt")
  model_state_dict = torch.load("model.pt")
  vae_model = tdi.model.VAEModel(
    adata=query_adata,
    batch_key="sample_name", 
    batch_embedding='embedding', 
    device='cuda:0', 
    batch_hidden_dim=64
  )
  vae_model.partial_load_state_dict(model_state_dict['model_state_dict'])

Getting the transfered latent embedding
---------------------------------------

.. code-block:: python
  :linenos:

  query_adata.obsm['X_gex'] = vae_model.get_latent_embedding()


Mapping the UMAP representation to the reference
------------------------------------------------




.. code-block:: python
  :linenos:
  
  query_adata.obsm['X_umap'] = tdi.ut.transfer_umap(
    reference_adata.obsm['X_gex'],
    reference_adata.obsm['X_umap'],
    query_adata.obsm['X_gex']
    method = 'knn'
  ) 


