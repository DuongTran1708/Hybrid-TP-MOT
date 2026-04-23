# PBVS 2026 - Thermal Pedestrian Multiple Object Tracking Challenge (TP-MOT)

## Automation Lab, Sungkyunkwan University

## Team: SKKU-AutoLab

## Paper accepted at the 22nd IEEE Workshop on Perception Beyond the Visible Spectrum (PBVS), CVPR 2026. [Paper](asset/PBVS-5.pdf)

---

#### I. Installation

1. Download & install Miniconda or Anaconda from https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html


2. Open new Terminal, create new conda environment named **pbvs25_mot** and activate it with following commands:

```shell
conda create --name hybrid-tp-mot python=3.10 -y

conda activate hybrid-tp-mot

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

bash setup.sh
```

---


#### II. Data preparation

##### a. Data download

Go to the website of PBVS Thermal MOT Challenge to get the dataset.

- https://pbvs-workshop.github.io/challenge.html

##### b. Video data import

Add video files to **Hybrid-TP-MOT/data**.

The program folder structure should be as following:

```
Hybrid-TP-MOT
├── data
│   ├──tmot_dataset
...
```

---

#### III. Reference

##### a. Check weight

Download weight from release of github [release](https://github.com/DuongTran1708/Hybrid-TP-MOT/releases/tag/v0.0.1)

The folder structure should be as following:
```
Hybrid-TP-MOT
├── models_zoo
│   ├──pbvs26_tmot
│   │   ├──yolov11s_tmot_v2.0_1920_1GPU
│   │   │   ├──weight
│   │   │   │   └──best.pt
│   │   ├──solider_reid
│   │   │   ├──rswin_tiny_reid.yml
│   │   │   └──rswin_tiny_reid_NEW.pth
```


##### b. Run inference

And the running script to get the result

```shell
bash run_official.sh
```

##### c. Get the result
After more than 5-10 minutes, we get the result:
```
Hybrid-TP-MOT
├── data
│   ├──tmot_dataset
│   │   ├──output_pbvs26
│   │   │   ├──tracking
│   │   │   │   └──hybridtpmot
```

#### IV. Data Processing

The data processing code is in the folder **Hybrid-TP-MOT/utilities**.

```
utilities/dataset_MOT_to_reid.py
utilities/dataset_MOT_to_pose_bbox_segmen_reid.py
```


#### V. Citation

```
@InProceedings{DuongTran_2026_CVPRW,
    author    = {Tran, Duong Nguyen-Ngoc and Pham, Long Hoang and Pham-Nam Ho, Quoc and Tran, Chi Dai and Huynh, Ngoc Doan-Minh and Nguyen, Huy-Hung and Jeon, Jae Wook},
    title     = {A Hybrid Data-Centric Framework for Thermal Multiple-Object Tracking with Complex Motion Patterns},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops},
    month     = {June},
    year      = {2026}
}
```

#### VI. Acknowledgement

Most of the code is adapted from [Mon](https://github.com/phlong3105/mon).

This repository also features code from
[Ultralytics](https://github.com/ultralytics/ultralytics),
[SOLIDER](https://github.com/tinyvision/SOLIDER),
[mmengine](https://github.com/open-mmlab/mmengine)