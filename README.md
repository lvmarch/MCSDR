# **MCSDR: Multi-Condition Sequential Diffusion Regression for LVEF Estimation**

This is the official code implementation for the paper **"MCSDR: Generative Regression for Left Ventricular Ejection Fraction Estimation from Echocardiography Video"** (Currently Under Peer Review).

## **📖 Abstract**

Left Ventricular Ejection Fraction (LVEF) estimation from echocardiography presents an ill-posed inverse problem, where intrinsic speckle noise, artifacts, and limited viewing angles can lead to ambiguity.Prevailing deep learning approaches typically formulate this task as deterministic regression, training models to minimize mean squared error. While effective, this paradigm generally encourages models to learn the conditional mean of the target variable, which may not fully capture the complex, multimodal nature of the true posterior distribution or the inherent aleatoric uncertainty.In this work, we explore a transition from deterministic regression to Generative Regression. We introduce MCSDR (Multimodal Conditional Score-based Diffusion for Regression), a probabilistic framework designed to model the continuous posterior distribution of LVEF conditioned on echocardiogram videos and patient demographic priors.Distinct from traditional methods that output a single point estimate, MCSDR learns a score function to refine a prediction from noise iteratively, guided by a Multimodal Conditional Score Network (MCSN). This architecture integrates visual, spatiotemporal, and patient demographics (e.g., age, sex) to constrain the solution space.Experiments on three public datasets (EchoNet-Dynamic, EchoNet-Pediatric, and CAMUS) demonstrate that our generative approach outperforms deterministic baselines in terms of accuracy. Crucially, qualitative analysis reveals that our model's generative trajectories exhibit distinct behaviors for confident versus ambiguous cases, offering a new layer of interpretability for AI-assisted diagnosis. 

**Core Features:**

* **Generative Regression**: Models the conditional distribution ![][image1] of LVEF using diffusion models, capturing the uncertainty of predictions.  
* **Multi-Condition Fusion**: Supports Multi-modal inputs, including Vision-only or Vision \+ Tabular data (e.g., Age, Sex, Weight, etc.).  

## **🏆 Performance & Model Zoo**

### **Pretrained Backbone**

We use **UniFormer-S** as our feature extractor. 

* **UniFormer-S Weights**: [Download via Google Drive](https://drive.google.com/file/d/1ZNF4lMTLEnaZyhVHHTUW_MctB7kDGP6o/view?usp=sharing)

### **Best Checkpoints**

We achieved SOTA performance on three public datasets. We provide the pretrained weights for the best models below:

| Dataset | Modality                      | MAE ↓ | RMSE ↓ | R2 ↑ | Checkpoint                                                                                        |
| :---- |:------------------------------| :---- | :---- | :---- |:--------------------------------------------------------------------------------------------------|
| **EchoNet-Dynamic** | Vision Only                   | **3.76** | **4.81** | **0.84** | [Download](https://drive.google.com/file/d/1CW7_MrWGZ1531QAojGCt4v_cZHnEhnJQ/view?usp=sharing)                                 |
| **EchoNet-Pediatric** | Vision Only                   | 4.26 | 5.88 | 0.74 | [Download](https://drive.google.com/file/d/1jzUDhHXgeaRPXk543imzgMJKwNlIKY4c/view?usp=sharing)                                 |
| **EchoNet-Pediatric** | Vision \+ Clinical attributes | **4.22** | **5.59** | **0.77** | [Download](https://drive.google.com/file/d/1ISSNApa-D2ieMJ3IijglD5pN41A1uSXR/view?usp=sharing)                                 |
| **CAMUS** | Vision Only                   | 6.22 | 8.31 | 0.58 | [Download](https://drive.google.com/file/d/1eMUA2VbE_YNb2016aQwqv4ef-gXfPgcp/view?usp=sharing) |
| **CAMUS** | Vision \+ Clinical attributes | **5.73** | **7.50** | **0.66** | [Download](https://drive.google.com/file/d/1FLMskU37D9gTI_1bimp9kszuyGSbsxvR/view?usp=sharing)    |

**Note**: The tabular data inputs (Clinical attributes) contained in different datasets are as follows:

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
├── configs/            
├── src/  
│   ├── data/           
│   ├── models/         
│   ├── engine/         
│   ├── losses/         
│   └── utils/          
├── train.py            
├── evaluate.py         

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
