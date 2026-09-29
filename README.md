# 🫁 흉부 X-ray 폐렴 분류 (Pneumonia Classification)

> 데이콘 · 오즈코딩 [초격차] AI 헬스케어 딥러닝 트랙 해커톤

흉부 X-ray 이미지를 입력받아 **정상(NORMAL, 0)** 과 **폐렴(PNEUMONIA, 1)** 을 분류하는 딥러닝 모델입니다.
ImageNet 사전학습 **EfficientNet-B0** 를 전이학습하고, 5-Fold 앙상블과 TTA, 검증셋 기반 threshold 튜닝으로 예측을 만듭니다. Grad-CAM으로 모델이 판단 근거로 삼은 영역도 시각화합니다.

---

## 📁 데이터

| 파일/폴더 | 설명 |
|---|---|
| `train/` | 학습용 흉부 X-ray 이미지 5,216장 (`train_0001.png` ~ `train_5216.png`) |
| `test/` | 평가용 흉부 X-ray 이미지 624장 (`test_0001.png` ~ `test_0624.png`) |
| `train.csv` | `file_name`, `label` (정상: 0, 폐렴: 1) |
| `test.csv` | `file_name` |
| `sample_submission.csv` | 제출 양식 (`file_name`, `label`) |

- 학습 데이터는 **폐렴 : 정상 ≈ 3 : 1** 로 클래스 불균형이 있습니다.
- 이 데이터 계열(Kermany et al., 2018 소아 흉부 X-ray)은 학습셋과 테스트셋의 클래스 분포가 달라서, 모델이 폐렴을 **과잉 예측**하기 쉽습니다.

---

## 🗂️ 프로젝트 구조

```
.
├── pneumonia_classification.ipynb   # 전체 파이프라인 노트북
├── README.md
├── data/                            # 대회 데이터 (직접 배치)
│   ├── train/
│   ├── test/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
└── outputs/                         # 실행 시 자동 생성
    ├── efficientnet_b0_fold{0-4}.pth
    ├── submission_efficientnet_b0_384_thr{X.XX}.csv
    └── test_prob_efficientnet_b0.csv
```

---

## ⚙️ 실행 환경

- Python 3.9+
- **GPU 권장** (CUDA 자동 감지, 없으면 CPU로 실행되지만 매우 느림)
- 주요 라이브러리

```bash
pip install torch torchvision opencv-python-headless scikit-learn pandas numpy matplotlib tqdm
```

**Google Colab 에서 실행하는 경우**
`런타임 → 런타임 유형 변경 → 하드웨어 가속기: GPU` 를 선택하세요.

---

## 🚀 실행 방법

1. 대회 데이터를 `data/` 폴더에 넣습니다. 다른 위치라면 노트북의 `CFG.DATA_DIR` 를 수정합니다.
2. `pneumonia_classification.ipynb` 를 위에서부터 순서대로 실행합니다.
3. `outputs/submission_*.csv` 파일을 데이콘에 제출합니다.

> 💡 처음에는 `N_FOLDS = 1`, `IMG_SIZE = 256` 으로 전체 흐름이 도는지 빠르게 확인한 뒤 설정을 키우는 것을 추천합니다.

---

## 🔬 파이프라인

```
X-ray 이미지
   │  CLAHE 대비 보정 → 흑백을 3채널로 복제
   ▼
Augmentation (소폭 회전·이동·확대, 밝기·대비)
   ▼
EfficientNet-B0 (ImageNet 사전학습, 출력층 2-class로 교체)
   ▼
5-Fold 학습 → 폴드별 best AUC 가중치 저장 + OOF 예측 수집
   ▼
OOF 기준 threshold 튜닝
   ▼
테스트 추론: 5 모델 × TTA 2종 = 10개 예측 평균 → threshold 적용
   ▼
submission.csv  +  Grad-CAM 시각화
```

### 1. 전처리 & 증강
- **CLAHE**(대비 제한 히스토그램 평활화)로 폐 음영 대비를 강화합니다.
- 흑백 이미지를 3채널로 복제해 ImageNet 사전학습 가중치를 그대로 활용합니다.
- 의료 영상 특성을 고려해 증강은 약하게 적용합니다. **좌우 반전은 심장 위치가 바뀌므로 사용하지 않습니다.**

### 2. 모델
| 항목 | 내용 |
|---|---|
| 기본 백본 | EfficientNet-B0 (약 5.3M 파라미터) |
| 사전학습 | ImageNet (torchvision `DEFAULT` 가중치) |
| 출력층 | 1000-class → 2-class Linear 교체 |
| 학습 방식 | 백본 전체 fine-tuning |
| 선택 가능 백본 | `efficientnet_b0`, `resnet50`, `densenet121`, `convnext_tiny` |

### 3. 학습 설정
| 항목 | 값 |
|---|---|
| 입력 크기 | 384 × 384 |
| 검증 | Stratified 5-Fold |
| Loss | 클래스 가중치(역빈도) + Label Smoothing(0.05) CrossEntropy |
| Optimizer | AdamW (lr 3e-4, weight decay 1e-4) |
| Scheduler | 1 epoch warmup + Cosine decay |
| Epoch | 최대 15 (Early stopping patience 4, 기준: val AUC) |
| 기타 | AMP(mixed precision), Gradient clipping(2.0) |

### 4. Threshold 튜닝
불균형 데이터에서는 0.5가 최적 기준이 아닌 경우가 많습니다.
OOF(Out-of-Fold) 예측 확률로 **F1(macro)을 최대화하는 threshold** 를 찾아 테스트 예측에 적용합니다.
평가지표가 다르다면 노트북의 `METRIC` 을 `'accuracy'` 또는 `'f1'` 로 바꾸면 됩니다.

### 5. 추론
- **Fold 앙상블**: 5개 폴드 모델의 예측 확률을 평균합니다.
- **TTA**: 원본 + 약간 확대(center crop) 이미지 2종을 평균합니다.
- 확률값을 `test_prob_*.csv` 로 따로 저장해, 다른 모델과 앙상블하거나 threshold만 바꿔 재제출할 수 있습니다.

### 6. Grad-CAM
검증 이미지에 대해 모델이 폐렴 판단 근거로 삼은 영역을 히트맵으로 시각화합니다.
폐 실질이 아닌 글자·마커·이미지 가장자리에 활성화가 몰린다면 shortcut learning을 의심해볼 수 있습니다.

---

## 🛠️ 주요 설정 (`CFG`)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `DATA_DIR` | `./data` | 데이터 폴더 경로 |
| `MODEL_NAME` | `efficientnet_b0` | 백본 모델 |
| `IMG_SIZE` | `384` | 입력 해상도 |
| `BATCH_SIZE` | `32` | GPU 메모리 부족 시 16으로 감소 |
| `EPOCHS` | `15` | 최대 학습 epoch |
| `N_FOLDS` | `5` | `1` 이면 단일 holdout(검증 15%) |
| `RUN_FOLDS` | `None` | 특정 폴드만 실행 (예: `[0]`) |
| `USE_TTA` | `True` | 테스트 시 TTA 사용 여부 |

---

## 📊 결과

| 모델 | IMG_SIZE | Folds | OOF AUC | OOF F1(macro) | Threshold | Public Score |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | 384 | 5 | - | - | - | - |

> 실행 후 결과를 채워주세요.

---

## 💡 개선 아이디어

- **해상도 상향**: `IMG_SIZE` 448~512 (미세한 음영 포착)
- **백본 앙상블**: `densenet121`, `convnext_tiny` 등을 각각 학습 후 `test_prob_*.csv` 평균
- **폐 영역 집중**: 폐 분할(U-Net) 마스크로 ROI crop 후 분류
- **도메인 사전학습 가중치**: CheXpert / NIH ChestX-ray14 기반 (예: TorchXRayVision)
- **불균형 대응 비교**: Focal Loss, WeightedRandomSampler
- **Threshold 미세 조정**: 제출 점수를 보며 `BEST_THR` 주변 조정

> ⚠️ 외부에서 테스트셋과 같은 원본 이미지를 구해 학습에 사용하는 것은 대회 규칙 위반 소지가 있으므로 사용하지 않습니다.

---

## 📚 참고

- Kermany, D. S., et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell, 172(5).
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
- Rajpurkar, P., et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* arXiv:1711.05225.
- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
