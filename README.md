# Tomato-Ripeness-Classification-via-Texture-Adaptive-Pooling-in-CNNs
By Rut Yatigul<sup>[1]</sup>, Wisanu Jitviriya<sup>[1]</sup>, 
Eiji Hayashi<sup>[2]</sup>, Aran Blattler<sup>[1]</sup> and 
Teppakorn Sittiwanchai<sup>[1]</sup> 

King Mongkut's University of Technology North Bangkok<sup>[1]</sup>  and Kyushu Institute of Technology<sup>[2]</sup>

## Implementation
In this repository, both the Enhanced Squeeze-and-Excitation Network and the Extended CNNs are implemented using Python and PyTorch.

### Dataset
The tomato ripeness classification dataset can be obtained from the original distributor 'THESIS TOMATO' at [16th run uploaded images e50 Computer Vision Model](https://universe.roboflow.com/thesis-tomato/16th-run-uploaded-images-e50). 
This dataset is utilized under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

### Required Python Libraries:
- torch
- torchvision
- scikit-learn
- numpy
- ptflops
- import-ipynb
- Pillow
- matplotlib
- pandas
- scikit-fuzzy

### Hardware
The models were trained on a Dell G15-5521 Gaming Laptop with the following specifications:
- CPU: 12th Gen Intel(R) Core(TM) i9-12900H
- GPU: NVIDIA GeForce RTX 3090 (24GB VRAM) (External GPU connected via Razer Core X Chroma)
- Memory: 16 GB (DDR5)
- Storage: 1TB NVMe SSD
- OS: Ubuntu 22.04

### Note
- The seed should be fixed at 42 to ensure reproducibility.
- To review previous training results or train a new models, please access the file named **"Architecture"_kfold_test_split.ipynb** located in the (**training models**) folder.
- **starter_"Architecture".ipynb** is used to centralize the training hyperparameters, ensuring that all models use consistent settings. It is not intended to be the file used to initiate the training process.

