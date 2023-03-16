#!/usr/bin/env python
# coding: utf-8

# # TCR-DeepInsigt Vignette

# The emergence of single-cell immune profiling technology has led to the production of a large amount of data on single-cell gene expression (GEX) and T cell receptor (TCR), which has great potential for studying TCR biology and identifying effective TCRs. However, one of the major challenges is the lack of a reference atlas that provides easy access to these datasets. On the other hand, the use of TCR engineering in disease immunotherapy is rapidly advancing, and single-cell immune profiling data can be a valuable resource for identifying functional TCRs. Nevertheless, the lack of efficient computational tools to integrate and identify functional TCRs is a significant obstacle in this field.

# In[1]:


import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['font.family'] = "Arial"


# Once the single-cell immune profiling datasets are processed by CellRanger and the GEX and TCR information are integrated by Scanpy, and Scirpy, you would get a datasets including:
# 1) The raw gene expression matrix
# 2) The Full-length TCR sequence for each single-cell
# 
# 
# And you would provide the sample name as well as the individual name to the dataset

# We use an example datasets from Sun et al., 2022 of Gastric Cancer here
# 
# Sun, K., Xu, R., Ma, F., Yang, N., Li, Y., Sun, X., Jin, P., Kang, W., Jia, L., Xiong, J., et al. (2022). scRNA-seq of gastric tumor shows complex intercellular interaction with an alternative T cell exhaustion trajectory. Nat Commun 13, 4943. 10.1038/s41467-022-32627-z.
# 

# In[123]:


adata = sc.read_h5ad("./tcr_deep_insight/data/example_query.h5ad")


# In[124]:


adata


# In[125]:


adata.obs


# In[73]:


from tcr_deep_insight.model._model import *
tdi = TCRDeepInsight(adata)
tdi.get_pretrained_gex_embedding(transfer_label='cell_type_3', query_multiples=9)
tdi.unique_tcr_by_individual()
tdi.get_pretrained_tcr_embedding('cuda:3')
tdi.tcr_adata.obs['disease_type_1'] = 'GC'
result = tdi.cluster_tcr_from_reference(label_key='disease_type_1', gpu=3)


# In[74]:


tdi.get_pretrained_gex_umap(min_dist=0.5)


# In[75]:


from tcr_deep_insight.utils._definitions import reannotated_prediction_palette
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
fig.set_size_inches(5,5)
sc.pl.umap(adata, color='cell_type', palette=reannotated_prediction_palette,ax=ax)

fig,ax=plt.subplots()
fig.set_size_inches(5,5)
sc.pl.umap(tdi.gex_reference, color='cell_type_3', palette=reannotated_prediction_palette,ax=ax)


# In[89]:


indices = np.random.choice(list(range(tdi.tcr_reference.shape[0])), tdi.tcr_adata.shape[0] * 9, replace=False)
z = umap.UMAP(min_dist=0.5).fit_transform(np.vstack([
    tdi.tcr_reference.obsm["X_gex"][indices],
    tdi.tcr_adata.obsm['X_gex']
]))


# In[77]:


m = tdi.gex_adata.obs.groupby('tcr').agg({'cell_type':list})
m = dict(zip(m.index, m['cell_type']))


# In[94]:


tdi.tcr_adata.obs['cell_type'] = list(map(lambda x: 
      mapf(m[x]),
      tdi.tcr_adata.obs['tcr']
))


# In[99]:


tcr_reference_to_plot = tdi.tcr_reference[indices]
tcr_reference_to_plot.obsm["X_umap"] = z[:-tdi.tcr_adata.shape[0]]
tdi.tcr_adata.obsm["X_umap"] = z[-tdi.tcr_adata.shape[0]:]
from tcr_deep_insight.utils._definitions import reannotated_prediction_palette
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
fig.set_size_inches(5,5)
sc.pl.umap(tdi.tcr_adata, color='cell_type', palette=reannotated_prediction_palette,ax=ax)

fig,ax=plt.subplots()
fig.set_size_inches(5,5)
sc.pl.umap(tcr_reference_to_plot, color='cell_subtype', palette=reannotated_prediction_palette,ax=ax)


# In[100]:


tcr_reference_to_plot


# In[76]:


from tcr_deep_insight.utils._definitions import reannotated_prediction_palette
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
fig.set_size_inches(5,5)
sc.pl.umap(adata, color='sample_name', palette=sc.pl.palettes.godsnot_102,ax=ax)


# In[78]:


result_tcr = result.obs[result.obs['mean_distance'] < 600]


# In[79]:


def FLATTEN(x): return [i for s in x for i in s]
def mapf(i):
    if len(np.unique(i)) == 1:
        return i[0]
    if len(i) == 2:
        return "Ambiguous"
    else:
        c = Counter(i)
        return sorted(c.items(), key=lambda x: -x[1])[0][0]
result_tcr['cell_type'] = list(map(lambda x: mapf(FLATTEN(list(map(lambda i: m[i], list(filter(lambda z: z != '-', x)))))), result_tcr.iloc[:,1:21].to_numpy()))


# In[80]:


result_tcr = result_tcr[
    np.array(result_tcr['count'] > 2) &
    np.array(result_tcr['number_of_individuals'] > 1) & 
    np.array(result_tcr['cell_type'] != 'MAIT')
].sort_values("number_of_individuals")


# In[81]:


result_tcr['disease_specificity_score'] = result_tcr['mean_distance_other'] - result_tcr['mean_distance']
result_tcr['tcr_similarity_score'] = 600-result_tcr['mean_distance']


# In[104]:


from tcr_deep_insight.utils._definitions import reannotated_prediction_palette
def createFig(figsize=(8, 4)):
    fig,ax=plt.subplots()           
    ax.spines['right'].set_color('none')     
    ax.spines['top'].set_color('none')
    #ax.spines['bottom'].set_color('none')     
    #ax.spines['left'].set_color('none')
    for line in ax.yaxis.get_ticklines():
        line.set_markersize(5)
        line.set_color("#585958")
        line.set_markeredgewidth(0.5)
    for line in ax.xaxis.get_ticklines():
        line.set_markersize(5)
        line.set_markeredgewidth(0.5)
        line.set_color("#585958")
    ax.set_xbound(0,10)
    ax.set_ybound(0,10)
    fig.set_size_inches(figsize)
    return fig,ax
fig,ax=createFig()
ax.scatter(
    result_tcr['tcr_similarity_score'],
    result_tcr['disease_specificity_score'],
    c=list(map(lambda x: 
               reannotated_prediction_palette.get(x[0]) if x[1]**2 +x[2]**2>300**2 else '#D7D7D7', 
               zip(result_tcr['cell_type'], result_tcr['disease_specificity_score'], result_tcr['tcr_similarity_score']))),
    s=result_tcr['count'] * 6, 
    linewidths=0
)

fig.savefig("../20230315_GC.pdf")


# In[122]:


tcrs = set(result_tcr.sort_values("tcr_similarity_score", ascending=False).iloc[1])

fig,ax=createFig()
fig.set_size_inches(3,3)

ax.scatter(
    tdi.gex_adata.obsm["X_umap"][:,0],
    tdi.gex_adata.obsm["X_umap"][:,1],
    s=0.5,
    color=list(map(lambda x: reannotated_prediction_palette[x], tdi.gex_adata.obs['cell_type'])),
    linewidths=0,
)

obsm = tdi.gex_adata[
        np.array(list(map(lambda x: x in tcrs,tdi.gex_adata.obs['tcr'])))
].obsm["X_umap"]

ax.scatter(
    obsm[:,0],
    obsm[:,1],
    s=10,
    marker='*',
    color='red'
)


# In[117]:


tcrs = set(result_tcr.sort_values("tcr_similarity_score", ascending=False).iloc[1])

fig,ax=createFig()
fig.set_size_inches(3,3)

ax.scatter(
    tdi.tcr_adata.obsm["X_umap"][:,0],
    tdi.tcr_adata.obsm["X_umap"][:,1],
    s=0.5,
    color=list(map(lambda x: reannotated_prediction_palette[x], tdi.tcr_adata.obs['cell_type'])),
    linewidths=0,
)

obsm = tdi.tcr_adata[
        np.array(list(map(lambda x: x in tcrs,tdi.tcr_adata.obs['tcr'])))
].obsm["X_umap"]

ax.scatter(
    obsm[:,0],
    obsm[:,1],
    s=10,
    marker='*',
    color='red'
)

