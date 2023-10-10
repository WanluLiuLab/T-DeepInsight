Integrating Multi-source Gene Expression (GEX)
==============================================

This is a repository for the code for integrating multi-source gene expression (GEX) data using the VAE model from the TCR-DeepInsight package.


.. code-block:: python
    :linenos 

    # import tcr_deep_insight as tdi 

    # Load the data
    adata = tdi.data.human_gex_reference_v2()


The `adata` is a :class:`anndata.AnnData` object with raw GEX count matrix stored in adata.X.


Training the VAE model using batch key
--------------------------------------

The following use `sample_name` as the batch key. The batch index is converted to **one-hot encoding** for the decoder part of the model to remove the batch effect.


.. code-block::python 
  :linenos

  # Subset 500,000 cells for training
  vae_model = tdi.model.VAEModel(
    adata=tdi.ut.random_subset_by_key(gex_reference_adata, key='sample_name', n = 500000),
    batch_key="sample_name", 
    batch_embedding='onehot',
    device='cuda:0', 
  )


The following use `sample_name` as the batch key. The batch index is converted to **64-dimensional embedding** for the decoder part of the model to remove the batch effect.


.. code-block::python 
  :linenos

  # Subset 500,000 cells for training
  vae_model = tdi.model.VAEModel(
    adata=tdi.ut.random_subset_by_key(gex_reference_adata, key='sample_name', n = 500000),
    batch_key="sample_name", 
    batch_embedding='embedding', 
    device='cuda:0', 
    batch_hidden_dim=64,
  )

Training the VAE model using batch key and categorical covariates (e.g. `study_name`)
-------------------------------------------------------------------------------------


.. code-block::python 
  :linenos

  # Subset 500,000 cells for training
  vae_model = tdi.model.VAEModel(
    adata=tdi.ut.random_subset_by_key(gex_reference_adata, key='sample_name', n = 500000),
    batch_key="sample_name", 
    categorical_covariate_keys=['study_name'],
    batch_embedding='embedding', 
    device='cuda:0', 
    batch_hidden_dim=64,
  )

Training the VAE model using batch key and label key (e.g. `cell_type`)
-----------------------------------------------------------------------


.. code-block::python 
  :linenos

  # Subset 500,000 cells for training
  vae_model = tdi.model.VAEModel(
    adata=tdi.ut.random_subset_by_key(gex_reference_adata, key='sample_name', n = 500000),
    batch_key="sample_name", 
    label_key='cell_type',
    batch_embedding='embedding', 
    device='cuda:0', 
    batch_hidden_dim=64,
  )

Re-Training the VAE model using a different set of genes (e.g. highly variable genes)
-------------------------------------------------------------------------------------

The `X_gex` is the VAE embedding of the GEX data. The `constrain_latent_embedding` and `constrain_latent_key` arguments constrain the VAE embedding to be close to the `X_gex` embedding. This is useful when the VAE model is trained on a different subset of genes (e.g. highly variable genes) and we want to use the VAE embedding of the full set of genes.


.. code-block::python 
  :linenos
  adata = adata[:,adata.var.highly_variable]
  # Subset 500,000 cells for training
  vae_model = tdi.model.VAEModel(
    adata=tdi.ut.random_subset_by_key(gex_reference_adata, key='sample_name', n = 500000),
    batch_key="sample_name", 
    batch_embedding='embedding', 
    device='cuda:0', 
    batch_hidden_dim=64,
    constrain_latent_embedding=True,
    constrain_latent_key='X_gex'
  )