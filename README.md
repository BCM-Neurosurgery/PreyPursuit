# PreyPursuit

code for running modeling for:
[Complementary roles for hippocampus and anterior cingulate in composing continuous choice](https://www.biorxiv.org/content/10.1101/2025.03.17.643774v1.abstract)

## Contents
* [Package Installation](#package-installation)
* [Controller Modeling](#controller-modeling)
* [Switch Detection](#switch-detection)
* [GLM modeling](#glm-modeling)
* [dPCA](#dpca)
* [Clustering](#clustering-analysis)

## Package Installation
To run controller modeling with example data, you need to have the required python packages installed. This can be done using the included conda environment config as below.

```{bash}
conda env create -f environment.yml
conda activate pac_control_env
```

## Controller Modeling
First, you will want to generate controller modeling results for subsequent analysis
To do this, make sure you are in the main directory and execute the following script
```{bash}
python -m scripts.run_controller_modeling -c pv 
```
**pv** is an example controller model we can run. you can specify any controller from the following list:
- p
- pv
- pf
- pvi
- pif
- pvf
- pvif

**NOTE**: all example data will by default be stored in the example_data/ directory. You can generally change the output directory directly in the code if needed.

## Switch Detection

## GLM modeling
To run GLM modeling with example data, execute the following script
```{bash}
python -m scripts.run_glm_modeling -p YFD
```
^ can optionally specify controller type to use for wt with `-m <c>`

## dPCA

## Clustering analysis

All modeling code (except dPCA) was written by Justin Fine, with modifications by Assia Chericoni and Taha Ismail
dPCA was written by Wieland Brendel



