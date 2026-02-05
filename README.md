# **MCSDR: Multi-Condition Sequential Diffusion Regression for LVEF Estimation**

This is the official code implementation for the paper **"MCSDR: Generative Regression for Left Ventricular Ejection Fraction Estimation from Echocardiography Video"** (Currently Under Peer Review).

## **📖 Introduction**

MCSDR is a generative regression framework based on Diffusion Models, designed to precisely estimate Left Ventricular Ejection Fraction (LVEF) from echocardiogram videos. Unlike traditional deterministic regression methods, MCSDR models the regression task as a denoising process from Gaussian noise to the target LVEF value.

**Core Features:**

* **Generative Regression**: Models the conditional distribution ![][image1] of LVEF using diffusion models, capturing the uncertainty of predictions.  
* **Multi-Condition Fusion**: Supports Multi-modal inputs, including Vision-only or Vision \+ Tabular data (e.g., Age, Sex, Weight, etc.).  

## **🏆 Performance & Model Zoo**

### **Pretrained Backbone**

We use **UniFormer-S** as our feature extractor. 

* **UniFormer-S Weights**: [Download via Google Drive](https://drive.google.com/file/d/1ZNF4lMTLEnaZyhVHHTUW_MctB7kDGP6o/view?usp=sharing)

### **Best Checkpoints**

We achieved SOTA performance on three public datasets. We provide the pretrained weights for the best models below:

| Dataset | Modality | MAE ↓ | RMSE ↓ | R2 ↑ | Checkpoint                                                                                        |
| :---- | :---- | :---- | :---- | :---- |:--------------------------------------------------------------------------------------------------|
| **EchoNet-Dynamic** | Vision Only | **3.31** | **4.22** | **0.88** | [Download](https://drive.google.com/file/d/1CW7_MrWGZ1531QAojGCt4v_cZHnEhnJQ/view?usp=sharing)                                 |
| **EchoNet-Pediatric** | Vision Only | 4.15 | 5.80 | 0.74 | [Download](https://drive.google.com/file/d/1jzUDhHXgeaRPXk543imzgMJKwNlIKY4c/view?usp=sharing)                                 |
| **EchoNet-Pediatric** | Vision \+ Text | **4.13** | **5.55** | **0.77** | [Download](https://drive.google.com/file/d/1ISSNApa-D2ieMJ3IijglD5pN41A1uSXR/view?usp=sharing)                                 |
| **CAMUS** | Vision Only | 6.22 | 8.31 | 0.58 | [Download](https://drive.google.com/file/d/1oxNmLuSOaqeXmCxSGIUl1EIYMSlB5eE1/view?usp=drive_link) |
| **CAMUS** | Vision \+ Text | **5.73** | **6.50** | **0.66** | [Download](https://drive.google.com/file/d/1FLMskU37D9gTI_1bimp9kszuyGSbsxvR/view?usp=sharing)    |
**Note**: The tabular data inputs (Text/Tabular) contained in different datasets are as follows:

* Pediatric: Age, Sex, Weight, Height  
* CAMUS: Age, Sex  

## **🛠️ Installation**

1. **Clone the Repository**  
   git clone \[https://github.com/lvmarch/MCSDR.git](https://github.com/lvmarch/MCSDR.git)  
   cd MCSDR

2. **Create Environment**  
   We recommend using Python 3.9+ and PyTorch 1.12+.  
   conda create \-n mcsdr python=3.9  
   conda activate mcsdr  
   pip install \-r requirements.txt

## **📂 Data Preparation**

Please organize your data according to the following structure. The code defaults to reading a CSV file index, which must contain FileName and EF columns, as well as tabular columns for multi-modal input (e.g., Age, Sex).

/Your/Data/Path/  
├── Videos/  
│   ├── 0X1A2B3C.avi  
│   ├── ...  
└── FileList.csv  \# Columns: FileName, EF, Split, \[Age, Sex...\]

Modify the data\_folder and file\_list\_path parameters in the configuration files (configs/) to point to your data path.

## **🚀 Training**

### **Run Training**

Use train.py and specify the corresponding configuration file. For example, training on EchoNet-Dynamic:
 
python train.py \--config configs/dynamic/stage2\_uniformer\_diffusion.yaml

**Configuration Notes**:

* Set use\_tabular\_data: True in the YAML file to enable multi-modal training.

## **📊 Evaluation**

### **Basic Evaluation**

Calculate MAE, RMSE, R2, and CRPS (Continuous Ranked Probability Score).

python evaluate.py \\  
  \--config configs/dynamic/eval\_uniformer\_diffusion.yaml \\  
  \--checkpoint save/dynamic\_train/best.pth \\  
  \--batch\_size 32


## **📁 Directory Structure**

MCSDR/  
├── configs/            \# Training and evaluation configs for each dataset  
├── src/  
│   ├── data/           \# Data loading and augmentation  
│   ├── models/         \# UniFormer, Diffusion Head, MLP Head  
│   ├── engine/         \# Training logic  
│   ├── losses/         \# RnC Loss, Diffusion Loss  
│   └── utils/          \# Plotting and metrics tools  
├── train.py            \# Training entry point  
├── evaluate.py         \# Evaluation entry point

## **🙏 Acknowledgements**

Parts of the code logic in this project are referenced or based on the following outstanding open-source projects. Special thanks to:

* [**CoReEcho**](https://github.com/BioMedIA-MBZUAI/CoReEcho): Referenced its contrastive learning-based ultrasound video representation learning framework and UniFormer implementation.  
* [**denoising-diffusion-pytorch**](https://github.com/lucidrains/denoising-diffusion-pytorch): Referenced the core implementation architecture of the Diffusion Model.  
* [**ddim**](https://github.com/ermongroup/ddim): Referenced its Denoising Diffusion Implicit Models (DDIM) accelerated sampling strategy.

## **📜 License**

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## **Citation**

If you find this code useful for your research, please consider citing our paper:

To be updated soon...


[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAZCAYAAACclhZ6AAADyElEQVR4Xu1XTUhUURR+gwZGRUXZoDPjnR9IrCBjaBNRm4KEapHRxqB2QbSuTQsJ3IaItYggCsIo22UGCQluJCEoioIQEiJJCEGojdj0ffPOfd535g3zbIYW4QeH9+75u+fce+6573neOv5DtLa2bta8GEh0dnZu4VML4iCfz2/VvLqRzWa7c7mc0fxa4ALA9r4xpk3L4gB2vQ1NiEkAbzU/DupNBkjA9mVDEioWixsQzOOOjo5BLYuDBiTD3ZnG/Fe9vyzVMtLp9EY4eQFnIxg2a3kcNCIZVEYS9h/hp1/LYiOTyeyDk0VQn5bFBRZjO+xH60nG80ttGPShvb19pxZ6qVRqB5WqjQmsxCU4WMLK7Hf5FnTMMtR8F0wCNFktGcyxzRkmMG5xxgG4oKBlLM7hkIAMCL6BxhFQBs+7oPegz+xaVk/KY5bb7NqLrBuyBZKbLMZnQJ9wYHfLODIZjLvg4zWeS4jnEWgv3l+BftCHq0ugSg6C/4sLHDAlkQegAVAJwouO7Bp5oB6O8ZwFPfQidoz1C9kV8VGeAPZ5jOdBU3K3RCXTjPfbRgLGs8v4CUzyfNEfaNhTc4qfOS5wwMSEx8HsAU2Iky4rk50oMSlxEDZe1btAO2kOizxb5GN8DOPfoCGrK0EEySSTyU14v5yV8oLtac4JGuAYu3yEjcfaW4ifOdCYllEYWkE+OaZjBHVKdCKTsYD8J2jUk07HgCSwoGFIEO7OhAD+kNiUq6EaxA+TmdQyCvUK2u0OdovGNZIJSkzGY6698GolQ9k8S1TLXIgfJsOyrxCWjHPQjN8tyAvuFLzPgCZYGoHhKlj7oU7HoGjDduzwKpJhidnuafzkgznwfgj2RatrIX4qF1cOKsvpvLDodBnU6+oZvwS+YuK0yxfQJjgvOf+Tp2TUnSRBBMmwhEXvOf0av0LK5wVowvstL+KClvO4YnQ5imAcNA16AppCxkdDSl5wOFeor2WE8ds5t/6p8Vt9qMREJ5SMJPAG9Mz4O8+OSDvGMRN1+AnptJXlaPyD2scLj3eIbHkF4DgFvS/U1zJBU6FQ2EUfoE4GrX8TdDKEndfqSktu07YWWOgW45/H8GeVtEa25dAKVkECKzEI3Wn3q1UmH3HPC1fOKdsAUcmsFfJZ9R1JnQgJjLRf0B329JCwChikcX4BWCpw/I4BQnYOzwU877k2Fo1IxviX9wHNZ2BnLbGMtDwK8hvQ736HYbUKxj9zN+Frj6dubIt6k2FFGNWY6gYTQdAnNb8WpCRv2Bt/rYDtdc1bxzr+If4A5FwpsdEzYfAAAAAASUVORK5CYII=>