Training T cell receptor (TCR) seqeunce by BERT
===============================================

This is a tutorial for training TCR sequence by BERT. We use the huARdbV2 reference dataset as an example.

.. code-block:: python
  :linenos:
  
  import t_deep_insight as tdi
  import torch 

  gex_reference_adata = tdi.data.human_gex_reference_v2()
  tdi.pp.update_anndata(gex_reference_adata)
  tcr_reference_adata = tdi.unique_tcr_by_individual(gex_reference_adata)

  tcr_reference_adata.obs['id'] = list(range(len(tcr_reference_adata.obs)))
  tcr_reference_adata = tcr_reference_adata[tcr_reference_adata.obs['TRBV']  != 'TRBV20OR9-2']
  tcr_dataset = tcr_tokenizer.to_dataset(
    tcr_reference_adata.obs
  )

  tcr_model = tdi.model.TCRModel(
    BertConfig.from_dict(tdi.model.config.get_config(
        hidden_size=384,
        intermediate_size=768
    )),
    labels_number=1
  ).to("cuda:2")

  tcr_collator = tdi.model.default_collator(
    species='human',
    tra_max_length=48,
    trb_max_length=48,
  )

  tcr_model_optimizer = torch.optim.AdamW(tcr_model.parameters(), lr=1e-4)

  tcr_model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    tcr_model_optimizer, 
    mode='min', 
    factor=0.1, 
    patience=5
  )
  tcr_model_trainer = tdi.model.TCRModelTrainer(
    tcr_model, 
    collator=tcr_collator, 
    train_dataset=tcr_dataset['train'], 
    test_dataset=tcr_dataset['test'], 
    optimizers=(tcr_model_optimizer, tcr_model_scheduler), 
    device='cuda:2'
  )
  tcr_model_trainer.fit(
    max_epoch=5, 
    show_progress=True, 
    n_per_batch=128, 
    shuffle=True
  )
  torch.save(
    tcr_model.state_dict(), 
    "./tcr_deep_insight/data/pretrained_weights/human_bert_tcr_v2_384_768.ckpt"
  )
  
Obtaining TCR sequence embedding
--------------------------------

We can obtain the TCR sequence embedding by the following code:

.. code-block:: python
  :linenos:
  
  tdi.tl.get_pretrained_tcr_embedding(
    tcr_adata=tcr_reference_adata,
    bert_config=tdi.model.config.get_human_config(),
    checkpoint_path='./tcr_deep_insight/data/pretrained_weights/human_bert_tcr_v2_384_768.ckpt',
    pca_path='./tcr_deep_insight/data/pretrained_weights/human_bert_tcr_pca_v2.pkl',
    use_pca=True,
    encoding='cdr123',
    mask_tr='tra', # or 'trb'. If chose 'tra', the CDR3α will be masked. If chose 'trb', the CDRβ  sequence will be masked.
  )

