# Non-Line-of-Sight 3D Object Reconstruction via mmWave Surface Normal Estimation

Created by: Laura Dodds, Tara Boroushaki, Kaichen Zhou, and Fadel Adib

[[Paper](https://laura-dodds.github.io/papers/mmNorm.pdf)] [[Slides](https://laura-dodds.github.io/data/mmNorm_final.pptx)]

https://github.com/user-attachments/assets/a7fdcd05-8660-4291-83be-f5629ae9cc83

## Requirements

This code relies on a GPU for computation. 

This repository has been tested on Ubuntu 22.04 with Python 3.10.  

## Installation

1. Clone this reposity
2. Run `python3 -m venv mmwave_venv` to create a new virtual environment
3. Run `source mmwave_venv/bin/activate` to source the virtual environment
4. Run `python3 setup.py --install` to run the install. 
      - Note: You can run `python3 setup.py` to re-compile the C++/CUDA code without re-installing packages. 

## Accessing Data
You can access demo data here: https://www.dropbox.com/scl/fi/keir3o11s9sux95bzwccu/sample_data.zip?rlkey=81siwq9m990h0id5esmmhe63h&st=c82an2s1&dl=0

To use this data, download the zip file and extract it in the main folder of the repo. There should be a `sample_data/` folder next to the `src/` folder. 

We are currently working on providing public access to our full dataset. If you would like access to the full dataset sooner, please email us. 

## Visualizing Reconstruction
Data can be visualized by running `cd src/utils && python3 visualization.py`. The sample data provided above contains pre-processed data that can be visualized without any processing. More details can be found in the documentation of that python file.

## Processing data
To process the sample data, run `cd src/data_processing && ./mmwave_reconstruction.sh`.

## Codebase Organization
Our codebase is organized as follows:

All the code to process the data is contained in `src/data_processing`. The `mmwave_reconstruction.sh` script is the main script to run the processing. It calls three different python files responsible for each of the three steps in mmNorm:
1. `mmwave_normal_estimation.py` estimates surface normal vector fields from mmWave signals (Section 2 in the paper).
2. `sdf.py` computes RSDF from the estimated normal vector fields (Section 3 in the paper).
3. `isosurface_optimization.py` runs the isosurface optimization to select the final isosurface and construct the overall point cloud (Section 4 in the paper). 

## Codebase Acknowledgement

This repository and dataset are built on our prior work, MITO. Please refer to that work and original code repository for more information on the dataset structure, SAR image processing, and more.   

## Citing mmNorm
If you found mmNorm helpful for your research, please consider citing us with the following BibTeX entry:
```
@inproceedings{dodds_mmnorm,
author = {Dodds, Laura and Boroushaki, Tara and Zhou, Kaichen and Adib, Fadel},
title = {Non-Line-of-Sight 3D Object Reconstruction via mmWave Surface Normal Estimation},
year = {2025},
doi = {10.1145/3711875.3729138},
url = {https://doi.org/10.1145/3711875.3729138},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
booktitle = {Proceedings of the 23rd International Conference on Mobile Systems, Applications, and Services},
series = {MobiSys '25}
}
```

If you found the mmWave dataset helpful for your research, please also consider citing our prior work, MITO, with the following BibTeX entry: 
```
@misc{dodds2025mitoenablingnonlineofsightperception,  
      title={MITO: Enabling Non-Line-of-Sight Perception using Millimeter-waves through Real-World Datasets and Simulation Tools},   
      author={Laura Dodds and Tara Boroushaki and Fadel Adib},  
      year={2025},  
      eprint={2502.10259},  
      archivePrefix={arXiv},  
      primaryClass={cs.CV},  
      url={https://arxiv.org/abs/2502.10259},   
}
```


