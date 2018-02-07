## Data Science Bowl

Code for 2018 data science bowl, using UNet and soft dice loss.

Because of my poor laptop, i haven't run the project so there may be many bugs.

FPN and other model will be implement, i'll run this project after next term begins.

### data overview
```
data
├── stage1_sample_submission.csv 
├── stage1_test 
├── stage1_train 
└── stage1_train_labels.csv 
```

 - (256, 256, 3)      334 
 - (256, 320, 3)      112 
 - (520, 696, 3)       92 
 - (360, 360, 3)       91 
 - (1024, 1024, 3)     16 
 - (512, 640, 3)       13 
 - (603, 1272, 3)       6 
 - (260, 347, 3)        5 
 - (1040, 1388, 3)      1 

### project structure
```
Kaggle18
├── config.py 
├── data 
│   ├── dataset.py 
│   ├── __init__.py 
│   └── Resize.py 
├── main.py 
├── model 
│   ├── BasicModule.py 
│   ├── checkpoints 
│   ├── __init__.py 
│   └── UNet.py 
├── README.md 
└── utils 
    ├── __init__.py 
    ├── Loss.py 
    ├── saved_loss.csv 
    ├── saved_lr.csv 
    └── util.py 
```


### Acknowledgement
thanks to [Andrea](https://www.kaggle.com/asindico/bowl-of-nuclei) for overview of data.

thanks to [Yun Chen](https://www.kaggle.com/cloudfall/pytorch-tutorials-on-dsb2018) for data preprocess and net model.

thanks to [Stephen Bailey](https://www.kaggle.com/stkbailey/teaching-notebook-for-total-imaging-newbies) for run-length encoding algorithm.
