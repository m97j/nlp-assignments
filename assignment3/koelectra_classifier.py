import argparse
import pandas as pd
import numpy as np
import torch
import random
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# 시드 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clean_text(text):
    return text.lower().strip()

def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

def main():
    parser = argparse.ArgumentParser(description="KoELECTRA Sentence Classifier (Fine-tuning)")
    parser.add_argument("--train", required=True, help="Path to training CSV file")
    parser.add_argument("--test", required=True, help="Path to test CSV file")
    parser.add_argument("--output", required=True, help="Path to save submission CSV file")
    parser.add_argument("--model_name", default="monologg/koelectra-base-discriminator", help="HuggingFace model name")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # GPU 권장 안내
    if not torch.cuda.is_available():
        print("⚠️ GPU 환경에서 실행을 권장합니다. 현재 CPU로 실행됩니다.")

    set_seed(args.seed)

    # 데이터 로드
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    # 텍스트 전처리
    train_df['text'] = train_df['text'].map(clean_text)
    test_df['text'] = test_df['text'].map(clean_text)

    # 라벨 매핑
    label_list = sorted(train_df['label'].unique())
    label2id = {v: i for i, v in enumerate(label_list)}
    id2label = {i: v for i, v in enumerate(label_list)}
    train_df['label'] = train_df['label'].map(label2id)

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof_preds = np.zeros((len(train_df), len(label_list)))
    test_preds = np.zeros((len(test_df), len(label_list)))

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        print(f"\n====== Fold {fold+1} ======")
        train_ds = Dataset.from_pandas(train_df.iloc[train_idx][['text', 'label']])
        val_ds = Dataset.from_pandas(train_df.iloc[val_idx][['text', 'label']])
        test_ds = Dataset.from_pandas(test_df[['text']])

        train_ds = train_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        val_ds = val_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        test_ds = test_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True)

        train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])

        # 모델 로드
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(label_list)
        )

        # 학습 파라미터
        training_args = TrainingArguments(
            output_dir=f'./result_{fold}',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            num_train_epochs=args.epochs,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            logging_dir=f'./logs_{fold}',
            save_total_limit=1,
            logging_steps=50,
            report_to="none",
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)
            return {"f1": f1_score(labels, preds, average='weighted')}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # OOF 예측
        val_preds = trainer.predict(val_ds).predictions
        oof_preds[val_idx] = val_preds

        # Test 예측
        test_preds += trainer.predict(test_ds).predictions / skf.n_splits

    # 최종 제출 파일 생성
    final_preds = np.argmax(test_preds, axis=1)
    submit = pd.DataFrame({
        'id': range(len(final_preds)),
        'label': [id2label[x] for x in final_preds]
    })
    submit.to_csv(args.output, index=False)
    print(f" Submission saved to {args.output}")

if __name__ == "__main__":
    main()
