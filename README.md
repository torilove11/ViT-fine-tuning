# 🐾 ViT Fine-tuning

이 프로젝트는 Vision Transformer (ViT) 모델을 Stanford Dogs Dataset에 맞춰 Fine-tuning하여 120종의 강아지 품종을 분류하는 모델을 구축하는 것을 목표로 합니다.

단순한 분류를 넘어, 학습된 모델을 활용하기 위해 Google Gemini와 연동한 'AI 강아지 관상소' 웹 데모(Gradio)를 구현하여 End-to-End 파이프라인을 완성했습니다.

---
### 🎯 프로젝트 목표 (Project Objective)
- ViT Fine-tuning: ImageNet-21k로 사전 학습된 google/vit-base-patch16-224-in21k 모델을 Downstream Task(견종 분류)에 최적화합니다.
- High Accuracy: 120개 클래스의 미세한 특징을 구분하여 높은 분류 정확도를 달성합니다.
- Multimodal Application: Vision 모델(ViT)의 예측 결과와 LLM(Gemini)의 스토리텔링을 결합한 창의적인 서비스를 구현합니다.
- ---
### 📊 모델 학습 요약 (Training Summary)

#### 1. Model Architecture
- Base Model: google/vit-base-patch16-224-in21k
- Input Size: 224 x 224 pixels
- Patch Size: 16 x 16

#### 2. Dataset
- Name: Stanford Dogs Dataset (maurice-fp/stanford-dogs)
- Classes: 120 Breeds
- Data Split: Train / Test

#### 3. Hyperparameters
학습 효율과 성능 최적화를 위해 다음과 같은 파라미터를 설정했습니다.
| Parameter        | Value | Description                     |
|------------------|-------|---------------------------------|
| Epochs           | 10    | 충분한 학습을 위해 10 Epoch 설정          |
| Batch Size       | 32    | GPU 메모리 효율성을 고려한 배치 크기          |
| Learning Rate    | 5e-5  | 사전 학습된 가중치를 파괴하지 않도록 낮은 학습률 적용  |
| Weight Decay     | 0.01  | 과적합 방지를 위한 가중치 감쇠 적용            |
| Warmup Ratio     | 0.1   | 학습 초기 안정화를 위한 Warmup 스케줄링       |
| Precision        | FP16  | 학습 속도 향상을 위한 Mixed Precision 사용 |
| label smoothing  | 0.1   | 모델의 과잉 확신을 억제하여 일반화 성능 개선       |

#### 4. Data Preprocessing & Augmentation
일반화 성능을 높이기 위해 학습 데이터에 다양한 변환을 적용했습니다.
- Train: RandomResizedCrop(224), RandomHorizontalFlip, RandomRotation(15), ColorJitter
- Test: Resize(256) → CenterCrop(224)

#### 5. Model performance
| Accuracy | Loss   |
|----------|--------|
| 88.85%   | 1.1695 |
---
### 🛠️ 기술 스택 (Tech Stack)
| Category        | Technology                                             |
|-----------------|--------------------------------------------------------|
| Deep Learning   | Python, PyTorch, Hugging Face Transformers, Accelerate |
| Model           | Vision Transformer (ViT), Google Gemini 1.5 Flash      |
| Data Processing | Hugging Face Datasets, ViTImageProcessor               |
| Serving & UI    | Gradio                                                 |
| Tools           | TensorBoard (Logging)                                  |
---
### 📂 프로젝트 구조 (Directory Structure)
```
├── src/
│   ├── config.py       # 학습 파라미터 (Epochs=10, LR=5e-5 등) 설정
│   ├── data.py         # Stanford Dogs 데이터셋 로드 및 Augmentation 정의
│   ├── model.py        # ViT 모델 로드 및 Label Mapping (id2label)
│   └── utils.py        # Accuracy Metric 계산 및 Collator 함수
├── train.py            # Hugging Face Trainer 기반 Fine-tuning 실행 스크립트
├── eval.py             # 저장된 모델 로드 및 Test Set 평가
├── infer.py            # 단일 이미지 예측(Inference) 및 Top-3 추출 로직
├── app.py              # Gradio + Gemini API 연동 웹 애플리케이션
└── requirements.txt    # 의존성 라이브러리 목록
```
---
### 🧪 활용 예시 (Application Demo)
학습된 모델의 실용성을 검증하기 위해 Gradio를 활용한 ****'AI 강아지 관상소'****를 제작했습니다.
1. Image Classification: ViT 모델이 업로드된 사진을 분석하여 상위 3개 견종 확률을 계산합니다.
2. Generative AI Integration: 1순위로 예측된 견종과 이미지를 Gemini 1.5 Flash에 전달합니다.
3. Storytelling: Gemini가 "관상가" 페르소나를 가지고 강아지의 성격과 운세를 재미있게 풀이해줍니다.