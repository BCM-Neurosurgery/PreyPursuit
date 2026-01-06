# PreyPursuit

code for running modeling for:
[Complementary roles for hippocampus and anterior cingulate in composing continuous choice](https://www.biorxiv.org/content/10.1101/2025.03.17.643774v1.abstract)

## Contents
* [Data availability](#data-availability)
* [Authorship Statement](#authorship-statement)
* [Package Installation](#package-installation)
* [Controller Modeling](#controller-modeling)
* [Recovery Simulations](#recovery-simulations)
* [Switch Detection](#switch-detection)
* [GLM modeling](#glm-modeling)
* [dPCA](#dpca)
* [Clustering](#clustering-analysis)

## Data Availability
This repository includes example behavioral and neural data from one participant,
provided to demonstrate the full analysis pipeline.

## Authorship Statement
All behavioral modeling code was originally written by Justin Fine, with subsequent
modifications and extensions by Assia Chericoni and Taha Ismail.
The dPCA implementation is based on the method introduced by Brendel et al.,
with task-specific adaptations.

## Package Installation
This repository includes Python and MATLAB scripts for behavioral modeling and
neural population analyses.
The code can be run on major operating systems (macOS, Linux, Windows) after
installing the required dependencies. 

This can be done with the included conda environment config and installed as below.
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

The controller modeling pipeline has a runtime of approximately 15
minutes and outputs trial-wise predicted wttrajectories for each controller class,
together with the corresponding evidence lower bound (ELBO) values.

## Recovery Simulations
A demo to run simualtions for model recovery analyses is provided in `demo/simulation_demo.ipynb`.
Because running full recovery analyses (e.g., 30 simulations × 30 trials) requires
several days of computation, we additionally provide the results of an example
simulation, together with MATLAB scripts for reproducing the figures, in `plotting/simulation`.

## Switch Detection
Switch detection is performed using the functions in the `ChangeOfMind/functions/processing.py` script.
An example implementation is provided in the demo/ folder as a Jupyter
notebook (`demo/plot_switch_detection.ipynb`). This demo can be run directly on the output of the
behavioral modeling pipeline (i.e., trial-wise w<sub>t</sub> time series).

## GLM modeling
To run GLM modeling with example data, execute the following script
```{bash}
python -m scripts.run_glm_modeling -p YFD
```
^ can optionally specify controller type to use for w<sub>t</sub> with `-m <c>`

## dPCA
a demo of the dPCA analysis is included in `demo/dPCA_demo_Final.ipynb`. It is not intetended to generate interpretable results, but only to show how the dPCA analysis is implemented.

## Clustering analysis

