API
===

Preprocessing
-------------

**VDJPreprocessingV1Human** and **VDJPreprocessingV1Mouse** are the main 
preprocessing classes for human and mouse data, respectively. They are
used to preprocess raw data and update the AnnData object with the
preprocessed data from CellRanger.

.. autoclass:: tcr_deep_insight.pp.VDJPreprocessingV1Human
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: tcr_deep_insight.pp.VDJPreprocessingV1Mouse
    :members:
    :undoc-members:
    :show-inheritance:


.. autofunction:: tcr_deep_insight.pp.update_anndata
.. autofunction:: tcr_deep_insight.pp.unique_tcr_by_individual

.. autodata:: tcr_deep_insight.pp.HSAP_REF_DATA

.. autodata:: tcr_deep_insight.pp.MMUS_REF_DATA

Data
----

.. autofunction:: tcr_deep_insight.data.human_gex_reference_v1
.. autofunction:: tcr_deep_insight.data.mouse_gex_reference_v1
.. autofunction:: tcr_deep_insight.data.human_tcr_reference_v1
.. autofunction:: tcr_deep_insight.data.mouse_tcr_reference_v1

Tool
----

.. autofunction:: tcr_deep_insight.tl.pretrain_gex_embedding
.. autofunction:: tcr_deep_insight.tl.get_pretrained_gex_embedding
.. autofunction:: tcr_deep_insight.tl.get_pretrained_tcr_embedding
    .. autofunction:: tcr_deep_insight.tl.cluster_tcr
.. autofunction:: tcr_deep_insight.tl.cluster_tcr_from_reference



Model
-----

.. autoclass:: tcr_deep_insight.model.TCRabModel
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: tcr_deep_insight.model.VAEModel
    :members:
    :undoc-members:
    :show-inheritance:

Plotting
--------

.. autofunction:: tcr_deep_insight.pl.set_plotting_params
.. autofunction:: tcr_deep_insight.pl.createFig 
.. autofunction:: tcr_deep_insight.pl.createSubplots
.. autofunction:: tcr_deep_insight.pl.plot_cdr3_sequence
.. autofunction:: tcr_deep_insight.pl.plot_gex_tcr_selected_tcrs
.. autofunction:: tcr_deep_insight.pl.plot_gex_selected_tcrs