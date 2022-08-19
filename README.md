# Mixture of Input-Output Hidden Markov Models for Heterogeneous Disease Progression Modeling
This repository contains the code to reproduce the experiments in:
```
@InProceedings{ceritli2022,
    title = {Mixture of Input-Output Hidden Markov Models for Heterogeneous Disease Progression Modeling},
    author = {Ceritli, Taha and Creagh, Andrew P. and Clifton, David A.},
    booktitle={Proceedings of the 1st Workshop on Healthcare AI and COVID-19,
ICML 2022}
```

Note that we extend the codebase in https://github.com/kseverso/DiseaseProgressionModeling-HMM 
and apply the data preprocessing steps in https://github.com/kseverso/Discovery-of-PD-States-using-ML. The repository is structured following the template provided in *[The Turing Way](https://the-turing-way.netlify.app/welcome)*.

---

*Contents:* <a href="#introduction"><b>Introduction</b></a> | <a href="#installation"><b>Installation</b></a> | <a href="#experiments"><b>Experiments</b></a> | <a href="#contributing"><b>Contributing</b></a> | <a href="#notes"><b>License</b></a>

---

## Introduction

## Installation
You can set up the necessary virtual environment via the following code:
```
conda create --name mIOHMM python=3.8.8
conda activate mIOHMM
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install scipy==1.6.1
pip install scikit-learn==0.24.1
pip install seaborn
pip install pandas
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=mIOHMM
```

## Experiments
You can reproduce the experiments via the command line as follows:
```
python -m experiments.synthetic
python -m experiments.real
```

You could also use the [jupyter notebook](https://github.com/tahaceritli/mIOHMM/blob/main/notebooks/Synthetic%20Data.ipynb) 
to reproduce synthetic data experiments. We provide another [jupyter notebook](https://github.com/tahaceritli/mIOHMM/blob/main/notebooks/Real%20Data.ipynb) for real data experiments to reproduce the figures 
and tables used in the paper using the trained models. 
<!---
## Repo Structure

Inspired by [Cookie Cutter Data Science](https://github.com/drivendata/cookiecutter-data-science)

```
├── LICENSE
├── README.md          <- The top-level README for users of this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── outputs            <- Figures and Tables generated using the trained models.
│
├── src                <- Source code for use in this project.
│   │
│   ├── utils.py       <- Scripts to download or generate data
│   │
│   ├── piomhmm.py     <- Class definitions for the model
└──
```
--->

## Contributing
If you encounter an issue in mIOHMM, please [open an 
issue](https://help.github.com/en/github/managing-your-work-on-github/creating-an-issue) 
or [submit a pull 
request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request). 

## License

This work is licensed under the MIT license (code) and Creative Commons Attribution 4.0 International license (for documentation).
You are free to share and adapt the material for any purpose, even commercially,
as long as you provide attribution (give appropriate credit, provide a link to the license,
and indicate if changes were made) in any reasonable manner, but not in any way that suggests the
licensor endorses you or your use, and with no additional restrictions.