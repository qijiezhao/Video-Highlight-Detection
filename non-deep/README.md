## VHD with non-deep models

### Requirements:
The following libraries are required to run the codes.
- sklearn
- multiprocessing
- itertools
- xgboost

They each supports: machine learning tools, multi-process reading data, Loop testing to tune parameters, GBDT lib.

### Run the code:

1. Configurations:
    
```
    Mannually set PATHs and hyper-params in configs.py as following:
    1,PROJECT_ROOT         the path to project
    2,BAIDU_VH_ROOT        the path to dataset
    3,MIN_FEATURE          the mini value of feature data( for computing svm kernels)
    4,METRICS              Use AP here
    5,ALTER                smoothing params
    6,GRID_SEARCH_PARAMS   grid search params
```

2. train the baidu dataset:

```
sh scripts/watershed_train.sh
```

3. test the baidu dataset(computing result on both val and test):

```
sh scripts/watershed_test.sh

```

###  Results:

**Single model's performance** (sample rate: 1:100 on train set):

tested on validation set

model name | AP
---|---
(1) Logistic regression | 81.7%
(2) XGBoost | 82.8%
(3) SVM (linear kernel) | 81.5%
fusion (1) and (2) | 83.2%

Tips: SVM with x2 kernel or rbf kernel may bring higher result, but we should consider more of efficiency here.

**Detection performance**

highlights score curves are generated from above, here we smooth them with multiple filters and then watersh them to final results.

set: IoU=0.5

filter length| AP | recall
---|---|---
4 | 0.41| 0.56
8 | 0.49| 0.51
12| 0.50| 0.52
20| 0.45| 0.51
40| 0.39| 0.42
80| 0.31| 0.38
fusion 8,12,20,40,80 + TBP|0.45|0.81
fusion 4,8,12,20,40,80 + TBP|0.43|0.85

Tips: NMS param threshold value is slightly tuned from 0.3~0.7.

Else, TBP is method: **T**hrow false positive **B**y **P**rior.
