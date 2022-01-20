# ACDC
## Description

This repository stores the scripts and helpers needed for training and evaluating different deep learning models used to enhance SLAM.

## Requirments

###### CUDA = v11.1
###### CuDNN >= v8.2.1
###### Python > 3.8
###### Poetry > 0.12

## Installation

To install locally under an isolated virtual environment use poetry. The
following command will install the specified versions of the dependencies
specified in `poetry.lock`. If the latter doesn't exist yet, it will create it
based on the version specifications found in `pyproject.toml`.
```bash
➜ cd <path/to/local/repo>
➜ poetry install

```
If you plan to use deformable convolutions (for example for semantic/panoptic segmentation), please run:
```bash
➜ cd <path/to/local/repo>
➜ poetry shell

```

## Dataset

TODO: How to download the dataset

## Inference

This is the command you need to use for inference:
```bash
➜ cd <path/to/local/repo>
➜ poetry run inference <PATH_TO_CONFIG> <PATH_TO_MODEL_WEIGHTS> -o <OUTPUT_FOLDER> [-cpu] [--workers N] [--seed N]
```

## Evaluation

This is the command you need to use for evaluation:
```bash
➜ cd <path/to/local/repo>
➜ poetry run evaluate <PATH_TO_CONFIG> <RESULT_FOLDER> [--workers N]
```

## Models

TODO: List of models to download