# PanTumorUSSeg

# Installation

This codebase is tested on Ubuntu 20.04.2 LTS with python 3.10. Follow the below steps to create environment and install dependencies.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -n pantumorseg python=3.10 -y

# Activate the environment
conda activate pantumorseg

# Install torch (requires version >= 2.1.2) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

```
* Clone PanTumorUSSeg code repository and install requirements
```bash
# Clone MaPLe code base
git clone https://github.com/HealthX-Lab/PanTumorUSSeg

cd PanTumorUSSeg/
# Install requirements

pip install -e .
```

# Data
* Place dataset under `data` like the following:
```
data/
|–– EUS/
|   |–– train/
|   |   |–– images/
|   |   |–– masks/
|   |–– val/
|   |   |–– images/
|   |   |–– masks/
|   |–– test/
|   |   |–– images/
|   |   |–– masks/
```

# Training and Evaluation
* Run the training and evaluation script

```bash
bash scripts/pipeline.sh EUS outputs
```

* You can change some design settings in the config:
```
|   |–– configs/
|   |   |–– EUS.yaml/
```
