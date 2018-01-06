# Video-Highlights-Detection

In this repo, Video Highlight Detection (VHD) are talked about.

[**Baidu_VH**](http://ai.baidu.com/broad/introduction) is the first large-scale dataset about VHD, main experiments are based on it.




Proposed codes :
- [x] prepare
- [ ] main experiments
- - [x] non-deep models
- - [ ] deep models
- [ ] ablations
- [ ] TBA

-----
Tips:

Non-deep models now have achieved 42% mAP (IoU ranges from 0.5 to 0.95 on BaiduVH dataset, each gap is 0.05)on the validtion set.
more information about non-deep, please refer to the 'non-deep' directory.
Performance on deep models is going to be proposed later.

### Result records:

- Thumos14, Temporal Actionness Grouping. mAP = 0.309

This is implemented by modified version of xiong's TAG. The experimental report/ relevant paper will propose soon. 



- Baidu VHD [preliminary stage](https://www.kesci.com/apps/home/competition/5a41bca63bf3464aab731a31/leaderboard):mAP = 33.9% 

Note that we only take xgb+lr yet(non-deep), and only 1% training data is used. We will refresh our result soon, by replacing xgb+lr with a mlp slightly and using full training set.

