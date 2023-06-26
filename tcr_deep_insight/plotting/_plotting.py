from collections import Counter
from typing import Optional
import scanpy as sc
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib.patches import Rectangle
import numpy as np 
import logomaker
import os
from ..utils._utilities import seqs2mat, mafft_alignment
from ._palette import godsnot_102

from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
import warnings
from pathlib import Path
MODULE_PATH = Path(__file__).parent
warnings.filterwarnings("ignore")

# Path to your OTF font file
font_path =  os.path.join(MODULE_PATH, "fonts", "Arial.ttf")

# Create a FontProperties object
# Load the custom font
arial_font = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = arial_font.get_name()


def set_plotting_params(
    dpi: int,
    fontsize: int = 12,
    fontfamily: str = "Arial",
    linewidth: float = 0.5,
):
    """
    Set default plotting parameters for matplotlib.

    :param dpi: dpi for saving figures
    :param fontsize: default fontsize
    :param fontfamily: default fontfamily
    :param linewidth: default linewidth
    
    """
    plt.rcParams['figure.dpi'] = dpi 
    plt.rcParams['savefig.dpi'] = dpi 
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['axes.linewidth'] = linewidth
    plt.rcParams['font.family'] = fontfamily
    mpl.rcParams['pdf.fonttype'] = 42 # saving text-editable pdfs
    mpl.rcParams['ps.fonttype'] = 42 # saving text-editable pdfs

amino_acids_color_scale = {
    'R': "#4363AE",
    'H': "#8282D1",
    'K': "#4164AE",
    'D': "#E61D26",
    'E': "#E61D26",
    'S': "#F8971D",
    'T': "#F8971D",
    'N': "#4FC4CC",
    'Q': "#4FC4CC",
    'C': "#E5E515",
    'G': "#ECEDEE",
    'P': "#DC9682",
    'A': "#C8C7C7",
    'V': "#148340",
    'I': "#148340",
    'L': "#148340",
    'M': "#E5E515",
    'F': "#3B5CAA",
    'Y': "#4164AE",
    'W': "#148340",
    '-': "#F7F7F7",
    '.': "#F7F7F7",
}
"""
Amino acids color scale for logo plots.
reference link: http://yulab-smu.top/ggmsa/articles/guides/Color_schemes_And_Font_Families.html
"""


def createFig(figsize=(8, 4)):
    """
    Create a figure with a single axis.

    :param figsize: figure size. Default: (8, 4)
    """
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

def createSubplots(nrow,ncol, figsize=(8,8),gridspec_kw={}):
    """
    Create a figure with multiple axes.

    :param nrow: number of rows
    :param ncol: number of columns
    :param figsize: figure size. Default: (8, 8)
    :param gridspec_kw: gridspec_kw. Default: {}
    """
    fig,axes=plt.subplots(nrow, ncol, gridspec_kw=gridspec_kw)
    for ax in axes.flatten():
        ax.spines['right'].set_color('none')     
        ax.spines['top'].set_color('none')
        for line in ax.yaxis.get_ticklines():
            line.set_markersize(5)
            line.set_color("#585958")
            line.set_markeredgewidth(0.5)
        for line in ax.xaxis.get_ticklines():
            line.set_markersize(5)
            line.set_markeredgewidth(0.5)
            line.set_color("#585958")
    fig.set_size_inches(figsize)
    return fig,axes


def piechart(
    ax, 
    anno, 
    cm_dict, 
    radius=1, 
    width=1, 
    setp=False, 
    show_anno=False
):
    kw = dict(arrowprops=dict(arrowstyle="-"), zorder=0, va="center")
    pie, _ = ax.pie(anno.values(),
                    radius=radius,
                    colors=[cm_dict[p] for p in anno.keys()],
                    # wedgeprops=dict(width=width, edgecolor='w')
                    )
    if setp:
        plt.setp(pie, width=width, edgecolor='w')
    for i, p in enumerate(pie):
        theta1, theta2 = p.theta1, p.theta2
        center, r = p.center, p.r
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        x = r * np.cos(np.pi / 180 * (theta1+theta2)/2) + center[0]
        y = r * np.sin(np.pi / 180 * (theta1+theta2)/2) + center[1]
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "arc3, rad=0"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        percentage = anno[list(anno.keys())[i]] / sum(list(anno.values()))
        if show_anno:
            if anno[list(anno.keys())[i]] / sum(list(anno.values())) > 0.005:
                ax.annotate(list(anno.keys())[i] + ", " + str(round(percentage * 100, 2)) + "%", xy=(x, y), xytext=(
                    x*1.2, y*1.2), horizontalalignment=horizontalalignment, size=6, fontweight=100)
    return pie


def mafft_alignment(sequences):
    result = []
    import sys
    from Bio.Align.Applications import MafftCommandline
    import tempfile
    with tempfile.NamedTemporaryFile() as temp:
        temp.write('\n'.join(list(map(lambda x: '>seq{}\n'.format(x[0]) + x[1], enumerate(sequences)))).encode())
        temp.seek(0)
        mafft_cline = MafftCommandline(input=temp.name)
        stdout,stderr=mafft_cline()
    for i,j in enumerate(stdout.split("\n")):
        if i % 2 != 0:
            result.append(j.replace("-","."))
    return result

def plot_cdr3_sequence(sequences, alignment = False):
    """
    Plot CDR3 sequences.

    :param sequences: a list of CDR3 sequences
    :param alignment: whether to align the sequences. Default: False
    
    """
    if alignment:
        sequences = mafft_alignment(sequences)
    fig,ax=createFig()
    for i,s in enumerate(sequences):
        for j,c in enumerate(s):
            ax.add_patch(Rectangle((1+j*0.2, 0.7*i+.9), 0.2,0.6, facecolor = amino_acids_color_scale[c]))
        ax.text(x=1.015,y=0.7*i+1,s=s,fontfamily='Droid Sans Mono for Powerline', size=12)
    ax.spines['right'].set_color('none')     
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')     
    ax.spines['left'].set_color('none')
    ax.set_ybound(0, 0.7*i+1)
    return fig, ax 

def plot_gex_selected_tcrs(
    adata,
    tcrs,
    color,
    palette,
    **kwargs
):
    """
    Plot the tcrs on the umap of the gex data

    :param gex_adata: sc.AnnData
    :param color: str
    :param tcrs: list
    :param palette: dict
    :return: fig, ax

    .. note::
        You should have `mafft` installed in your system to use this function
    """
    fig,ax=createFig()
    fig.set_size_inches(3,3)

    ax.scatter(
        adata.obsm["X_umap"][:,0],
        adata.obsm["X_umap"][:,1],
        s=0.1,
        color=list(map(lambda x: palette[x], adata.obs[color])),
        linewidths=0,
        alpha=0.2,
    )

    obsm = adata[
        np.array(list(map(lambda x: x in tcrs,adata.obs['tcr'])))
    ].obsm["X_umap"]

    ax.scatter(
        obsm[:,0],
        obsm[:,1],
        s=10,
        marker='*',
        color='red',
    )

def plot_gex_tcr_selected_tcrs(
    gex_adata: sc.AnnData,
    color: str,
    tcrs: list,
    palette: Optional[dict] = None
):
    """
    Plot the tcrs on the umap of the gex data, with the TCRs as a pie chart and logo plot

    :param gex_adata: sc.AnnData
    :param color: str
    :param tcrs: list
    :param palette: dict (optional) 
    :return: fig, ax

    .. note::
        You should have `mafft` installed in your system to use this function
    """
    if palette is None:
        if len(set(gex_adata.obs[color])) <= 20:
            palette = sc.pl.palettes.default_20
        elif len(set(gex_adata.obs[color])) <= 28:
            palette = sc.pl.palettes.default_28
        else:
            palette = sc.pl.palettes.default_102
            
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['font.family'] = "Arial"

    gs_kw = dict(width_ratios=[1,1], height_ratios=[6, 1, 1])
    fig, axes = plt.subplot_mosaic([[0,0],
                                [3,4],
                                [1,2]],
                                gridspec_kw=gs_kw, figsize=(5, 7),
                                layout="constrained")

    logomaker.Logo(seqs2mat(mafft_alignment(list(map(lambda x: x.split("=")[0], tcrs)))), ax=axes[1])
    logomaker.Logo(seqs2mat(mafft_alignment(list(map(lambda x: x.split("=")[1], tcrs)))), ax=axes[2])

    axes[0].scatter(
        gex_adata.obsm["X_umap"][:,0],
        gex_adata.obsm["X_umap"][:,1],
        s=0.5,
        color=list(map(lambda x: palette[x], gex_adata.obs[color])),
        linewidths=0,
    )

    obsm = gex_adata[
            np.array(list(map(lambda x: x in tcrs, gex_adata.obs['tcr'])))
    ].obsm["X_umap"]

    axes[0].scatter(
        obsm[:,0],
        obsm[:,1],
        s=10,
        marker='*',
        color='red'
    )

    obs = gex_adata[
            np.array(list(map(lambda x: x in tcrs, gex_adata.obs['tcr'])))
    ].obs

    piechart(
        axes[3],
        Counter(obs['TRAV']),
        show_anno=True,
        cm_dict=dict(zip(Counter(obs['TRAV']).keys(), godsnot_102))
    )


    piechart(
        axes[4],
        Counter(obs['TRBV']),
        show_anno=True,
        cm_dict=dict(zip(Counter(obs['TRBV']).keys(), godsnot_102))
    )

    for ax in axes.values():
        ax.spines['right'].set_color('none') 
        ax.spines['top'].set_color('none')
        ax.set_xticks([])
        ax.set_yticks([])

        for line in ax.yaxis.get_ticklines():
            line.set_markersize(5)
            line.set_color("#585958")
            line.set_markeredgewidth(0.5)
        for line in ax.xaxis.get_ticklines():
            line.set_markersize(5)
            line.set_markeredgewidth(0.5)
            line.set_color("#585958")

    axes[1].set_title("CDR3a")
    axes[2].set_title("CDR3b")
    axes[3].set_title("TRAV")
    axes[4].set_title("TRBV")
