# DS-PASS
Detail-Sensitive Panoramic Annular Semantic Segmentation

# PASS Dataset

## GoogleDrive
[**New! Google Drive Download Link**](https://drive.google.com/file/d/1A_P2u5HUbrHZnKJYAOL2f7JLxxj69LqB/view?usp=sharing)

## BaiduYun
For Validation (Most important files):

[**Unfolded Panoramas for Validation**](https://pan.baidu.com/s/1lsd_CN9u4uSCp-KmE2pn9Q),
(400 images)

[**Annonations**](https://pan.baidu.com/s/1XJ6fFq60UwTZui456AQlPw), (400 annotation images)

[Groundtruth](https://pan.baidu.com/s/1RkrxtYu5Y1UzBvzn8aBugg)

There are 400 panoramas with annotations. Please use the Annotations data for evaluation.

In total, there are 1050 panoramas. Complete Panoramas:

[All Unfolded Panoramas](https://pan.baidu.com/s/16BLZArMyVfP_dEYnshEicQ)

RAW Panoramas: [RAW1](https://pan.baidu.com/s/1LBTQnVHcL0TKoY7njtPiBg),
               [RAW2](https://pan.baidu.com/s/1B_kaC8uu531exuXMlCE6_A),
               [RAW3](https://pan.baidu.com/s/1car_7_dH58wKWDjM6brhlQ)


## Panorama Sequences Captured by Instrumented Vehicle
[Sequence1](https://pan.baidu.com/s/17L1-of4f80-sJqcsha_umw)
[Sequence2](https://pan.baidu.com/s/1-sCfhJPrm8YlFOmad90ljw)
[Sequence3](https://pan.baidu.com/s/1CfgSTD3jJR9tnE79oAu4BA)

## Panorama Sequences Captured by Mobile Robot
[Sequences](https://pan.baidu.com/s/15lIseRZkZgtF4UhCthlsUA)

# Code Usage
Download the Model (model_superbest.pth) from

[**Trained-SwaftNet-Model BaiduYun**](https://pan.baidu.com/s/1GHgv8cLA-LzsgtqGaYAm6Q)

or

[**Trained-SwaftNet-Model GoogleDrive**](https://drive.google.com/file/d/1dNx3FbeqJjrB7-LrhlLnKvdlZ69_0sZ0/view?usp=sharing)

```
python3.6 eval_cityscapes_color_1.py --datadir /home/kailun/Downloads/DS-PASS-master/eval_swaftnet/data/ --subset val --loadDir ../eval_swaftnet/ --loadWeights model_superbest.pth --loadModel swaftnet.py
```

![Example segmentation](example_segmentation.jpg?raw=true "Example segmentation")

[![Video](dspass_youtube.jpg?raw=true)](https://youtu.be/pvhpxbdrA_4 "Video on Youtube")

# Publications
If you use our code or dataset, please consider citing any of the following papers:

**DS-PASS: Detail-Sensitive Panoramic Annular Semantic Segmentation through SwaftNet for Surrounding Sensing.**
K. Yang, X. Hu, H. Chen, K. Xiang, K. Wang, R. Stiefelhagen.
In IEEE Intelligent Vehicles Symposium (**IV**), Las Vegas, NV, United States, October 2020. 
[[**PDF**](https://arxiv.org/pdf/1909.07721)]
[[**VIDEO**](https://youtu.be/pvhpxbdrA_4)]

**Helping the Blind to Get through COVID-19: Social Distancing Assistant Using Real-Time Semantic Segmentation on RGB-D Video.**
M. Martinez, K. Yang, A. Constantinescu, R. Stiefelhagen.
Sensors, 2020.
[[**PDF**]](https://www.mdpi.com/1424-8220/20/18/5202/pdf)
