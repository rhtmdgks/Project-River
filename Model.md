# Project River – EEG Jamo Classification Pipeline  
**Version 1.0 – Confirmed Architecture**

본 문서는 ㄱ, ㄴ, ㄷ, ㄹ, ㅇ, ㅏ, ㅓ, ㅡ, ㅣ (총 9 classes)의 한국어 자모 상상(Imagined Jamo) EEG 데이터를  
Mac mini M2 환경에서 효율적으로 학습하기 위한 **최종 확정 모델 구조와 알고리즘 설계**를 기술한다.

---

# 1. Core Principle  
저채널·저SNR EEG(Muse 2)에 대해 과도하게 복잡한 딥러닝 모델은 거의 성능이 나오지 않는다.  
따라서 Project River는 **특징 공학(Feature Engineering) 우선 → 경량 CNN(EEGNet) 적용**이라는  
실용주의적 Feasibility-First 전략을 채택한다.

---

# 2. Overview of the Training Pipeline

1. Raw EEG CSV 로드  
2. 신호 전처리 (Filtering → Artifact Reduction)  
3. 시간–주파수 변환 (STFT/WT)  
4. Band Power(Delta–Gamma) 추출  
5. Feature Tensor 구축  
6. EEGNet 기반 경량 CNN 학습  
7. Cross-validation 기반 성능 검증  
8. 실시간 추론용 경량화(선택)

---

# 3. Signal Pre-processing

## 3.1 Filtering
- Band-pass filter: **0.5–45 Hz**  
- Notch filter: **60 Hz (전력선 잡음 제거)**  
- 필요 시 smoothing(window=250–500 ms)

## 3.2 Artifact Reduction
- ICA 또는 회귀 기반 EOG 제거  
- Accelerometer/Gyro 기반 Motion Artifact 마스킹  
- HeadBandOn 상태값으로 불량 구간 제거

---

# 4. Feature Engineering (Mandatory)

원시 시계열(raw signal) 대신 **주파수 영역 기반의 정량적 feature vector**를 사용한다.

## 4.1 Time–Frequency Transform
- STFT(Welch) 또는 Wavelet Transform

## 4.2 Band Power Extraction
각 채널(TP9, AF7, AF8, TP10)에 대해 다음 대역 파워를 계산:

| Band  | Range (Hz) | 의미 |
|-------|------------|------|
| Delta | 0.5–4      | 안정/저각성 |
| Theta | 4–8        | 인지/언어 상상 관련 |
| Alpha | 8–12       | 시/주의 억제 패턴 |
| Beta  | 12–30      | 인지 작업 활성 |
| Gamma | 30–45      | 상상 발화 시 coherence 변동 |

→ 최종 feature shape: **(5 bands × 4 channels = 20 features)**  
→ 시간 윈도우 단위로 sliding하여 multiple samples 생성

Band Power는 Muse 4채널 EEG에서 가장 안정적이며,  
Imagined Speech 연구에서 **Theta·Gamma 대역이 가장 중요한 분리 특성**으로 보고된다.

---

# 5. Model Architecture Decision

## 5.1 선택된 모델: **EEGNet (경량 CNN)**  
장점:
- 저채널 EEG에 최적화  
- Depthwise & Separable Conv 구조로 연산량 극도로 감소  
- Mac Mini M2에서 매우 빠른 학습 속도  
- 소규모 데이터셋에서도 일반 CNN보다 과적합이 적음  
- 실시간 배포 가능성 높음

단점:
- 매우 복잡한 패턴을 잡지는 못하지만  
  **Band Power 기반 입력에서는 오히려 가장 안정적**

결론: **Project River의 1st Generation 모델로 확정**

---

# 6. Data Augmentation (Recommended)

데이터 부족을 보완하기 위해 다음 기법을 적용:

- Gaussian noise 추가  
- Temporal Jittering (±50–100 ms shift)  
- Random Scaling  
- Time Masking (SpecAugment 변형)

---

# 7. Training Strategy

- Optimizer: Adam  
- Loss: CrossEntropy  
- Learning Rate: 1e-3 → cosine decay  
- Batch Size: 16–64  
- Epochs: 50–200 (early stopping 적용)  
- Evaluation: Stratified 5-fold cross-validation

초기 목표 성능:  
- 9-class 분류 정확도 **≥ 50%**  
  (baseline random chance 11% 대비 4–5배 향상)

---

# 8. Deployment Considerations

- Apple Silicon(M2)의 GPU(ANE) 기반 추론 가속 지원  
- CoreML 변환 가능 (추후 iOS/Swift 기반 UI 구성 가능)  
- 실시간 시스템에서 latency < 50 ms 달성 가능

---

# 9. Final Recommendation

**신호 품질 확보 → Band Power 특징 추출 → EEGNet 입력 → 경량 학습**  
이라는 흐름은 Mac Mini M2, Muse 2 EEG, 한국어 자모 분류라는  
Project River의 모든 제약 조건을 만족시키는 최적의 설계이다.

