# MITO: Enabling Non-Line-of-Sight Perception using Millimeter-waves through Real-World Datasets and Simulation Tools

Created by: Laura Dodds, Tara Boroushaki, Kaichen Zhou, and Fadel Adib

[[Paper](http://arxiv.org/abs/2502.10259)] 

https://github.com/user-attachments/assets/b99041ab-b8b2-481a-a487-7e895c477a07


## ??


This repository and dataset are built on our prior work, MITO. Please refer to that work and original code repository for more information on the dataset structure, ..., and more.   

## Installation
This repository is tested on Ubuntu 22.04 with Python 3.10. 

1. Clone this reposity
2. Run `python3 -m venv mmwave_venv` to create a new virtual environment
3. Run `source mmwave_venv/bin/activate` to source the virtual environment
4. Run `python3 setup.py --install` to run the install. Please note:
    1. This installs numpy 1.24.3, which is required for PSBody. If you are not planning to run the simulation, you can use a different version of numpy by changing requirements.txt.
    2. This setup will install requirements as needed, depending on the settings in param.json. It will only install GPU requirements if use_cuda is true, and it will only install simulation/segmentation requirements if use_simulation/use_segmentation are true. Please update params.json accordingly before running python3 setup.py

## Accessing Data
Our data is stored on a hugging face repository. To clone a copy of this repository, please follow these steps:

1. Make sure you run the following commands from the outermost directory of this repo (i.e., within the MITO_Codebase folder)
2. Run `huggingface-cli login`
3. Follow the URL printed in the terminal to create a new hugging face token (or use an existing one)
4. Enter your hugging face token. You can choose whether to save this token as a git credential.  
5. Run `git clone https://huggingface.co/datasets/SignalKinetics/MITO_Dataset`

You should now have a folder called `MITO_Dataset` which contains the processed files for all objects. If you would like access to the raw signal data, please email us.

## Visualizing Data
Data can be visualized by running `cd src/utils && python3 visualization.py`. More details can be found in the documentation of that python file (or in the tutorial listed below).

## Classifier
The final classifier weights can be found in `src/classification/checkpoints`. Run `test_classifier.py` to see the result for the final weights on the test set. Run `train_classifier.py` to train the classifier. 

### Pretrained Weights
Pretrained weights are available [here](https://drive.google.com/file/d/1WzkjCBq-tK-8Il-dCjyOuNZ_oeNgd8AV/view?usp=sharing).
This includes a zip file of trained weights for our full classifier and two microbenchmarks (for using only edge or only specular simulations). To test with pretrained weights, create a folder `src/classification/checkpoints` and extract the zip file within that folder beofre running `test_classifier.py`.  

## Running the Simulation
The simulation can be run with the `run_simulation.sh` file. See the file (or the tutorial listed below) for more details. 

## Tutorials 
We provide the following tutorials to introduce different features of this repository. All tutorials are in the `tutorials\` folder.

### 1. Loading and Visualizing the Dataset
This tutorial introduces the dataset (contents and structure) and shows how to download, access, and visualize the data. If your goal is to build new models using the previously processed images, this tutorial should be sufficient for your goals. The remainder of the tutorials show more advanced functionality (e.g., building models on this dataset or simulating/processing new images.)

### 2. Simulating new mmWave Images
This tutorial shows how to use our open-source simulation tool to produce synthetic images for any 3D mesh. This can be used to produce more synthetic data than we have released in our initial dataset release. 

### 3. (COMING SOON) Segmenting mmWave Images
This tutorial shows our approach for segmenting mmWave images using the SegmentAnything model (https://github.com/facebookresearch/segment-anything). This is a good example of using our data in downstream models.

### 4. (COMING SOON) Classifying mmWave Images
This tutorial shows our approach for classifying mmWave images, using a custom classification network. This is a good example of buidling custom models with our dataset. 

### 5. (COMING SOON) Understanding mmWave Imaging
This is an advanced tutorial explaining in-depth how mmWave imaging works. 


## Citing mmNorm
If you found mmNorm helpful for your research, please consider citing us with the following BibTeX entry:
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

If you found the mmWave dataset helpful for your research, please consider citing our prior work, MITO, with the following BibTeX entry: 
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


