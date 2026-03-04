# PBVS 2026 - Thermal Pedestrian Multiple Object Tracking Challenge (TP-MOT)

## Automation Lab, Sungkyunkwan University

## Team: SKKU-AutoLab

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

Download weight from release of github [release](.)

The folder structure should be as following:
```
Hybrid-TP-MOT
├── models_zoo
│   ├──pbvs26_tmot
│   │   ├──yolov11s_tmot_v2.0_1920_1GPU
│   │   │   ├──weight
│   │   │   │   └──best.pt
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
```

#### IV. Citation

#### V. Acknowledgement

Most of the code is adapted from [Mon](https://github.com/phlong3105/mon).

This repository also features code from
[Ultralytics](https://github.com/ultralytics/ultralytics),
[SOLIDER](https://github.com/tinyvision/SOLIDER),
[mmengine](https://github.com/open-mmlab/mmengine)