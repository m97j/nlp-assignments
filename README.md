
---

# 🗣️ NLP Assignments

이 저장소는 3개의 자연어처리 과제(NLP1, NLP2, NLP3)의 구현 코드를 모아둔 포트폴리오입니다.  
각 과제는 **GitHub에는 코드만** 포함되어 있으며, **상세 실행 결과와 분석 보고서는 Kaggle Notebook**에서 확인할 수 있습니다.

---

## 📁 Repository Structure
```
NLP/
  assignment1/   # Byte Pair Encoding (BPE) Tokenizer
  assignment2/   # Hangul Automata (자소 → 음절 조합기)
  assignment3/   # KoELECTRA 기반 문장 분류 (Fine-tuning)
```

---

## 📌 Assignment 1 — Byte Pair Encoding (BPE) Tokenizer
**목표**  
- BPE 알고리즘을 이용해 서브워드 단위 토크나이저 구현  
- 최대 어휘 크기 제한 및 학습/추론 모드 지원

**구현 내용**  
- 학습 모드: 코퍼스 파일 로드 → BPE 병합 규칙 학습 → 어휘 저장(pickle)  
- 추론 모드: 저장된 병합 규칙 로드 → 입력 텍스트 토큰화 → 결과 저장

**실행 방법**
```bash
# 학습
python bpe_tokenizer.py --train ./data/corpus.txt --max_vocab 3000 --vocab ./model/vocab.pkl

# 추론
python bpe_tokenizer.py --infer ./model/vocab.pkl --input ./data/input.txt --output ./data/output.txt
```

**상세 결과 및 분석**  
🔗 [Kaggle Notebook - NLP1](https://www.kaggle.com/nlp1-link)

---

## 📌 Assignment 2 — Hangul Automata (자소 → 음절 조합기)
**목표**  
- 한글 자모 입력을 받아 완성형 음절로 변환하는 오토마타 구현  
- 초성·중성·종성 상태 전이 및 복합 종성 처리 지원

**구현 내용**  
- 초성/중성/종성 리스트 정의  
- 상태 기반 자소 처리 로직 구현  
- `<` 입력 시 백스페이스 기능 지원

**실행 방법**
```bash
python hangul_automata.py
# 콘솔에서 자소 입력 → 변환된 완성형 한글 출력
```

**상세 결과 및 분석**  
🔗 [Kaggle Notebook - NLP2](https://www.kaggle.com/nlp2-link)

---

## 📌 Assignment 3 — KoELECTRA 기반 문장 분류 (Fine-tuning)
**목표**  
- 한국어 사전학습 모델(KoELECTRA)을 활용한 문장 분류 모델 파인튜닝  
- Stratified K-Fold 교차검증(5-Fold) 적용  
- Weighted F1-score 기준 성능 최적화

**구현 내용**  
- 데이터 전처리(소문자 변환, 공백 정리)  
- Hugging Face Transformers 기반 토크나이징 및 모델 로드  
- Trainer API를 활용한 학습/평가/추론 파이프라인 구성  
- OOF 예측 및 Test 예측 결과 생성

**실행 방법** *(GPU 환경 강력 권장 — CPU 실행 시 매우 오래 걸릴 수 있음)*
```bash
python koelectra_classifier.py \
    --train ./data/train.csv \
    --test ./data/test.csv \
    --output ./submission.csv
```
- `--epochs`, `--batch_size`, `--model_name` 등 하이퍼파라미터를 CLI에서 조정 가능  
- `--output` 경로에 제출용 CSV 파일이 생성됨

**상세 결과 및 분석**  
🔗 [Kaggle Notebook - NLP3](https://www.kaggle.com/code/nrmx202/mjk-nlp)

---

## 📂 Data
- 모든 과제의 데이터는 수업 자료 또는 교내 Private Dataset으로 **공개 불가**
- 동일한 형식의 예시 데이터는 공개 코퍼스(Kaggle, AI Hub 등)를 참고 가능
- Kaggle Notebook에서는 Private Dataset을 사용하여 실행 후 결과만 공개

---

## 📜 License
이 저장소의 코드는 자유롭게 참고 가능하나, 데이터는 포함되어 있지 않습니다.

---
