# %% [markdown]
# This notebook inspects optimal hyperparameters for classification models finetuning.

# %%

# import torch
# from numba import cuda
# cuda.select_device(0)
# cuda.close()
# cuda.select_device(0)
# torch.cuda.empty_cache()

# %%
label_set = ["Negative", "Positive", "Neutral"]
STR_TO_NUM = {k: i for i, k in enumerate(label_set)}
NUM_TO_STR = {i: k for i, k in enumerate(label_set)}

import pandas as pd

df = pd.read_json("bcs_polsent.jsonl", orient="records", lines=True)
df["label"] = df.label.apply(lambda s: STR_TO_NUM[s])
df = df[["sentence", "label", "split"]].rename(
    columns={"sentence": "text", "label": "labels"}
)
train = df[df.split == "train"].drop(columns=["split"])
dev = df[df.split == "dev"].drop(columns=["split"])
test = df[df.split == "test"].drop(columns=["split"])


# %%
from transformers import AutoTokenizer
from datasets import Dataset

train = Dataset.from_pandas(train)
dev = Dataset.from_pandas(dev)


# %%
def train_and_eval(model_name, num_epoch, batch_size):

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=512)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_train = train.map(preprocess_function, batched=True)
    tokenized_dev = dev.map(preprocess_function, batched=True)

    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    from transformers import (
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    import numpy as np

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=8,
        seed=np.random.randint(1000),
        save_steps=-1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    def predict(model, text):
        import torch

        inputs = tokenizer(
            "Hello, my dog is cute", return_tensors="pt"
        ).input_ids.cuda()
        logits = model(inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return int(predicted_ids[0])

    y_pred = [NUM_TO_STR[predict(model, i)] for i in dev["text"]]
    y_true = [NUM_TO_STR[i] for i in dev["labels"]]

    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

    f1 = f1_score(y_true, y_pred, labels=label_set, average="macro")
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=label_set)
    del model
    with open("results.csv", "a") as f:
        f.write(f'{model_name},{batch_size},{num_epoch},{f1},"{y_true}","{y_pred}"\n')


models = [
    "xlm-roberta-base",
    "xlm-roberta-large",
    "classla/bcms-bertic",
    "EMBEDDIA/crosloengual-bert",
]
epochs = [1, 2, 3, 4, 5, 6, 7, 9, 15, 30, 60]
batch_sizes = [8, 16]

for current_model in models:
    for batch_size in batch_sizes:
        for epoch in epochs:
            for i in range(5):
                train_and_eval(current_model, epoch, batch_size)
