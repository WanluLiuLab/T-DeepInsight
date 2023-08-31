import setuptools 

setuptools.setup(
    name="TCRDeepInsight",
    version="0.0.2",
    author="Ziwei Xue",
    author_email="xueziweisz@gmail.com",
    description="TCRDeepInsight: a deep learning framework for single-cell GEX/TCR  analysis",
    long_description="TCRDeepInsight: a deep learning framework for single-cell GEX/TCR  analysis",
    package_dir = {'': 'tcr_deep_insight'},
    packages=setuptools.find_packages("tcr_deep_insight", exclude=[
        "*reference*",
        "*pretrained_weights*",
        "*docs*"
    ]),
    install_requires=[
        'faiss-gpu==1.7.2',
        'transformers==4.17.0',
        'datasets==1.18.4',
        'anndata==0.8.0',
        'scanpy==1.8.1',
        'scikit-learn==0.24.1 ',
        'matplotlib==3.3.4 ',
        'einops==0.4.1 ',
        'biopython==1.79',
        'seaborn==0.12.2',
        'torch==1.13.1+cu117',
        'torchvision==0.14.1+cu117',
        'torchaudio==0.13.1+cu117',
    ],
    dependency_links=['https://download.pytorch.org/whl/cu117'],
    include_package_data=False,
)
