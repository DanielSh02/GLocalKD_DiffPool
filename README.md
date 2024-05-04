# GLocalKD_DiffPool

This repository contains an implementation of a GCN-based model for graph anomaly detection. It is built on combining the GLocalKD model developed by Ma et al. (2022) with the DiffPool architecture proposed by Ying et al. (2019). The code is based off of the implementation of GLocalKD by Ma available [here](https://github.com/RongrongMa/GLocalKD).

The implementation of the DiffPool model with GCNs can be found in `gcn.py`. The training function is written in `main.py`. Testing of the model can be done with different parameters by editing the runtime section at the end of `main.py`.
