API
===

Preprocessing
-------------

**VDJPreprocessingV1Human** and **VDJPreprocessingV1Mouse** are the main 
preprocessing classes for human and mouse data, respectively. They are
used to preprocess raw data and update the AnnData object with the
preprocessed data from CellRanger.

.. autoclass:: t_deep_insight.pp.VDJPreprocessingV1Human
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: t_deep_insight.pp.VDJPreprocessingV1Mouse
    :members:
    :undoc-members:
    :show-inheritance:


.. autofunction:: t_deep_insight.pp.update_anndata
.. autofunction:: t_deep_insight.pp.unique_tcr_by_individual

.. autodata:: t_deep_insight.pp.HSAP_REF_DATA

.. autodata:: t_deep_insight.pp.MMUS_REF_DATA

Data
----


.. autofunction:: t_deep_insight.data.human_gex_reference_v2
.. autofunction:: t_deep_insight.data.human_tcr_reference_v2
.. autofunction:: t_deep_insight.data.human_gex_reference_v2_cd4
.. autofunction:: t_deep_insight.data.human_gex_reference_v2_cd8
.. autofunction:: t_deep_insight.data.human_gex_reference_v2_zheng_2020_cd4
.. autofunction:: t_deep_insight.data.human_gex_reference_v2_zheng_2020_cd8
.. autofunction:: t_deep_insight.data.human_gex_multi_atlas_v1_cd4
.. autofunction:: t_deep_insight.data.human_gex_multi_atlas_v1_cd8
.. autofunction:: t_deep_insight.data.mouse_gex_reference_v1
.. autofunction:: t_deep_insight.data.mouse_tcr_reference_v1

Tool
----

.. autofunction:: t_deep_insight.tl.pretrain_tcr_embedding
.. autofunction:: t_deep_insight.tl.get_pretrained_tcr_embedding
.. autofunction:: t_deep_insight.tl.cluster_tcr
.. autofunction:: t_deep_insight.tl.cluster_tcr_from_reference

.. autoclass:: t_deep_insight.tl.TDIResult
    :members:
    :undoc-members:
    :show-inheritance:


Model
-----

.. autoclass:: t_deep_insight.model.TCRabModel
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: t_deep_insight.model.VAEModel
    :members:
    :undoc-members:
    :show-inheritance:


Plotting
--------

.. autofunction:: t_deep_insight.pl.set_plotting_params
.. autofunction:: t_deep_insight.pl.create_fig
.. autofunction:: t_deep_insight.pl.create_subplots
.. autofunction:: t_deep_insight.pl.plot_cdr3_sequence
.. autofunction:: t_deep_insight.pl.plot_selected_tcrs

Utilities
---------

.. autofunction:: t_deep_insight.utils.transfer_umap