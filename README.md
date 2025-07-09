# qa_headpose_regression
## Head Pose Regression Using Quantization Aware MobilenetV2

State of the art head pose detection ( yaw, roll, pitch ) dated to @2020-2021

Estimate three angles using a face image, the angles correspond to yaw, roll, pitch. ( note that the face image should be cropped using some tool, lets say mtcnn or hog )

Consider that some of this is inspired from FSANet at the time [FSANet](https://github.com/shamangary/FSA-Net), while imporving and speeding up inference, the model achieves the same inferece time as FSANet on GPU (3ms), using only tflite and 8 cpu cores.

## UPDATE
i have rewritten the original tensorflow code with torch. for now same results aren't guaranteed but i'll work on it. Original code is located in the [IPYNB](./train_pose_predictor.ipynb)

to train the model again simply:

### 1. download the datasets
```bash
gdown 1j6GMx33DCcbUOS8J3NHZ-BMHgk7H-oC_
unzip data.zip
mv data.zip archive
```

### 2. train
```bash
python train.py
```

observe the results

## Examples
( YAW PITCH ROLL order ):  
![image](https://github.com/KiLJ4EdeN/head_pose_regression/assets/57220509/a47f5bb7-4766-47a2-8e9b-512815e47332)
```
Model Prediction: [ 47. -45. -39.]
Ground Truth: [ 37.76773363 -51.35657112 -34.43247194]
```



![image](https://github.com/KiLJ4EdeN/head_pose_regression/assets/57220509/878b12c6-c328-4a6b-b3b6-74d0618e31b2)
```
Model Prediction: [ 68. 12.  4. ]
Ground Truth: [ 71.62278432 13.5477656   9.57131228 ]
```

## Results
```
BIWI
YAW: 3.8884045387677673
PITCH: 4.948589362227766
ROLL: 2.656535269831331

MAE: 3.8311763902756213
=========================
AFLW
YAW: 4.029069105922883
PITCH: 5.747231110398772
ROLL: 3.975032158809743

MAE: 4.583777458377132
```

#### Comparison with SOTA

<img src="https://github.com/shamangary/FSA-Net/blob/master/FSANET_table1.png" height="220"/><img src="https://github.com/shamangary/FSA-Net/blob/master/FSANET_table2.png" height="220"/><img src="https://github.com/shamangary/FSA-Net/blob/master/FSANET_table3.png" height="220"/>

```BibTeX
@misc{head_pose_regression,
  title =        {Head Pose Regression Using Quantization Aware MobilenetV2},
  author =       {Abdolkarim Saeedi},
  publisher =    {GitHub},
  howpublished = {\url{[https://github.com/KiLJ4EdeN/qa_headpose_regression](https://github.com/KiLJ4EdeN/qa_headpose_regression)https://github.com/KiLJ4EdeN/qa_headpose_regression}},
  year =         {2024}
}
```
