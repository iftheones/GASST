# GASST
The open source code will be available here after the publication of the paper GASST:Geometrically Adaptive Spatial-Spectral Selective Cross-Scale Transformer for Hyperspectral Image Classification.

<img width="1776" height="720" alt="GASST" src="https://github.com/user-attachments/assets/c2bc0c75-25d2-4fb6-a274-50e3b6b451d1" />


## 🚀Let's Get Started!
### `A. Installation`
**Step 1: Clone the repository:**

Clone this repository and navigate to the project directory:
```bash
git clone https://github.com/iftheones/GASST.git
cd GASST
```

**Step 2: Environment Setup:**

It is recommended to set up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:

***Create and activate a new conda environment***

```bash
conda create -n GASST
conda activate GASST
```

***Install dependencies***

Our method uses python 3.8, pytorch 1.13, other environments are in requirements.txt
```bash
pip install -r requirements.txt
```

### `B. Data Preparation`

Download HSI classification dataset from [Google Drive](https://drive.google.com/drive/folders/1iPFLdrAFUveqwCtMpf5859pQhGXN_z4J?usp=drive_link) or [Baidu Drive (百度网盘)](https://pan.baidu.com/s/1bSqq-Uv3AC5qfRmqxbMjfg?pwd=2025) and put it under the [dataset] folder. It will have the following structure: 
```
${DATASET_ROOT}   # Dataset root directory
├── datasets
│   │
│   ├── pu        # Pavia University data
│   │   ├──PaviaU.mat
│   │   ├──PaviaU_gt.mat
│   │
│   ├── houston13  # Houston 2013 data
│   │   ├──GRSS2013.mat
│   │   ├──GRSS2013_gt.mat 
│   │
│   ├── ip         # Indian Pines data	
│   │   ├──Indian_pines_corrected.mat
│   │   ├──Indian_pines_gt.mat 
│   │     
│   ├── whuhh     # Whu-HongHu data
│   │   ├──WHU_Hi_HongHu.mat
│   │   ├──WHU_Hi_HongHu_gt.mat 
│   │
│   ├── other HSI Datasets   
│   │   ├ ... 
│   │    

```

### `C. Performance Evaluation`
- The following commands show how to train and evaluate DSFormer for HSI classification:
```bash
python main.py --model GASST --dataset_name pu --num_run 10 --epoch 500 --device 0 --dataID 1 --patch_size 10 --k 2/5 --train_num 30 --group_num 4 --ps 2
python main.py --model GASST --dataset_name ip --num_run 10 --epoch 500 --device 1 --dataID 4 --patch_size 10 --k 4/5 --train_num 50 --group_num 4 --ps 2
python main.py --model GASST --dataset_name houston13 --num_run 10 --epoch 500 --device 2 --dataID 3 --patch_size 10 --k 3/5 --train_num 50 --group_num 4 --ps 2
python main.py --model GASST --dataset_name whuhh --num_run 10 --epoch 500 --device 3 --dataID 7 --patch_size 10 --k 3/5 --train_num 50 --group_num 4 --ps 2
```

## 📜Reference

if you find it useful for your research, please consider giving this repo a ⭐ and citing our paper! We appreciate your support！😊
