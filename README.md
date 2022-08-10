# Face Occlusion in the Wild
Kakao Pay AI Engineer 채용과제입니다.

------

## Introduction
- 얼굴의 가림 유무를 분류하는 모델입니다.
- Mobilefacenet 을 사용하였으며, Accuracy 기준 93.9 % 의 성능을 달성하였습니다.
- Implementation 은 [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) 를 참조하였습니다. 아래는 원본 repository 의 설명입니다.
------
- This repo is a reimplementation of Arcface[(paper)](https://arxiv.org/abs/1801.07698), or Insightface[(github)](https://github.com/deepinsight/insightface)
- For models, including the pytorch implementation of the backbone modules of Arcface and MobileFacenet
- Codes for transform MXNET data records in Insightface[(github)](https://github.com/deepinsight/insightface) to Image Datafolders are provided
- Pretrained models are posted, include the [MobileFacenet](https://arxiv.org/abs/1804.07573) and IR-SE50 in the original paper
------

## Datasets & Performance
[Mobilefacenet @ GoogleDrive](https://drive.google.com/drive/folders/1uYuCbup6C4r26yRMpFp_3O8cUczsgiVI?usp=sharing)

| Scratch | Pretraining | Pretraining + Class Imbalance | Pretraining + Class Imbalance + Data Augmentation | 
|----|----|----|----|
| 0.468 | 0.469 | 0.908 | 0.939                                             |

## How to use

- clone

  ```
  git clone https://github.com/windforces-student/face-occlusion-in-the-wild.git
  ```

### Data Preparation

#### Prepare Facebank
- Google drive 에서 다운로드받은 데이터를 다음과 같은 구조로 배치합니다.
```text
data/facebank/
        ---> train_v2/
            ---> 0_0_0
            ---> ...
        ---> test/
            ---> 0_0_0
            ---> ...
```

#### Download the pretrained model
```text
model_2022-08-05-22-09_accuracy:0.939209726443769_step:759_final_v7.onnx
```
#### Check inference results in `Export To ONNX.ipynb`
- Inference 성능은 위의 ipynb 에서 확인하십시요.

------

## References
- This repo is mainly inspired by [deepinsight/insightface](https://github.com/deepinsight/insightface) and [InsightFace_TF](https://github.com/auroua/InsightFace_TF)
