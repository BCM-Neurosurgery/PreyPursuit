# PreyPursuit

## Overview
code for running modeling for:
[Complementary roles for hippocampus, anterior cingulate and orbitofrontal cortex in composing continuous choice](https://www.biorxiv.org/content/10.1101/2025.03.17.643774v2)

## Contents
* [Data availability](#data-availability)
* [Authorship Statement](#authorship-statement)
* [Package Installation](#package-installation)
* [Controller Modeling](#behavioral-modeling)
* [Recovery Simulations](#model-recovery-analysis)
* [Switch Detection](#switch-analysis)
* [GLM modeling](#neural-gam-modeling)
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

## System Requirements

## Hardware
No special hardware requirements beyond a standard workstation.

## Software
The code can be run on major operating systems (macOS, Linux, Windows) after
installing the required dependencies. 
- Python (3.11), see `environment.yml` for a complete list of required packages
- MATLAB (for plotting and selected analysis)

## Package Installation
Install the required Python packages using the provided conda environment file:

```{bash}
conda env create -f environment.yml
conda activate pac_control_env
```

## Repo Contents
### Behavioral Modeling
To run an example of the controller modeling pipeline:
```{bash}
python -m scripts.run_controller_modeling -c pv 
```
**pv** example controller class. you can specify any controller from the following list:
- p
- pv
- pf
- pvi
- pif
- pvf

**NOTE**: all results will by default be stored in the `example_data/` directory. 

The controller modeling pipeline has a runtime of approximately 15
minutes and outputs trial-wise predicted *w<sub>t</sub>* trajectories for each specified controller class.

### Model Recovery Analysis
Code for simulation-based model recovery analysis is provided in `demo/simulation_demo.ipynb`.
Due to long runtimes, example results and MATALB plotting scripts are included in `plotting/simulation`.

### Switch Analysis
Switch detection is performed using the functions in the `ChangeOfMind/functions/processing.py` script.
An example implementation is provided in the demo/ folder as a Jupyter
notebook (`demo/plot_switch_detection.ipynb`). This demo can be run directly on the output of the
behavioral modeling pipeline (i.e., trial-wise *w<sub>t</sub>* time series).
The GLM used for the switch statistics can be found under `legacy/MAIN_GLM_all_subj.ipynb`.

### Neural GAM modeling
To run GAM modeling with example data, execute the following script
```{bash}
python -m scripts.run_glm_modeling -p YFD
```
^ can optionally specify controller type to use for *w<sub>t</sub>* with `-m <c>`

can plot proportion of tuned neurons, NNMF results, and fischer information results using the plotting notebook included under `plotting/plot_nmf_fi.ipynb`

### dPCA
Neural population analysis include condition-based and *w<sub>t</sub>*-binned dPCA. Full dPCA analysis can be found under `src/dPCA` and run with the `DPCAPipeline`.
Example scripts with synthetic-data demos are provided in `demo/dPCA_demo_Final.ipynb`. 

### Clustering analysis
Neural population analysis for clustering ramp. Full clustering analysis can be found under `kmeans.ipynb`.

