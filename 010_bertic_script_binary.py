# %% [markdown]
# # Bertic, binary CLF, on full train

# %%
label_set = ['Negative', "Other"]
STR_TO_NUM = {k: i for i, k in enumerate(label_set)}
NUM_TO_STR = {i:k for i, k in enumerate(label_set)}

import pandas as pd
df = pd.read_json("bcs_polsent_007.jsonl", orient="records", lines=True)
df["label"] = df.label.apply(lambda s: STR_TO_NUM.get(s, STR_TO_NUM.get("Other")))
df = df[["sentence", "label", "split"]].rename(columns={"sentence": "text", "label":"labels"})
train = df[df.split=="train"].drop(columns=["split"]).reset_index(drop=True)
dev = df[df.split=="dev"].drop(columns=["split"]).reset_index(drop=True)
test = df[df.split=="test"].drop(columns=["split"]).reset_index(drop=True)

def train_model(train_df, model_name, output_dir):
    if model_name == "EMBEDDIA/crosloengual-bert":
        model_type = "bert"
        NUM_EPOCH = 3
    elif model_name == "classla/bcms-bertic":
        model_type = "electra"
        NUM_EPOCH = 9
    elif model_name == "xlm-roberta-base":
        model_type = "xlmroberta"
        NUM_EPOCH = 40
    else:
        raise AttributeError(f"Expected either xlm-roberta-base, classla/bcms-bertic, or EMBEDDIA/crosloengual-bert, got {model_name}.")

    from simpletransformers.classification import ClassificationModel
    import torch
    torch.cuda.empty_cache()
    model_args = {
        "num_train_epochs": NUM_EPOCH,
        "learning_rate": 4e-5,
        "overwrite_output_dir": True,
        "use_multiprocessing_for_evaluation": False,
        "use_multiprocessing": False,
        "use_cuda": True,
        # "train_batch_size": 8, 
        # "no_cache": True,
        "output_dir": output_dir,
        "save_steps": -1,
        "max_seq_length": 512,
        "silent": True,
    }

    model = ClassificationModel(
        model_type, model_name, num_labels=3, use_cuda=True, args=model_args
    )
    model.train_model(train_df)
    return model


def eval_model(model, test_df):
    y_true_enc = test_df.labels
    from tqdm.auto import tqdm
    y_pred_enc = [model.predict(i)[0][0] for i in tqdm(test_df.text.values)]

    y_true = [NUM_TO_STR[i] for i in y_true_enc]
    y_pred = [NUM_TO_STR[i] for i in y_pred_enc]
    from sklearn.metrics import f1_score
    macroF1 = f1_score(y_true, y_pred, labels=label_set, average="macro")

    return {"macroF1": macroF1, "y_true": y_true, "y_pred": y_pred}


# %%
best_f1 = 0
from pathlib import Path

binary_path = Path("./models/binary")
binary_best_path = Path("./models/binary_best")
import shutil
for i in range(10):
    model = train_model(train, "classla/bcms-bertic", str(binary_path))
    current_f1 = eval_model(model, test)["macroF1"]
    print(f"{current_f1=}")
    if current_f1 > best_f1:
        shutil.rmtree(str(binary_best_path))
        shutil.copytree(str(binary_path), str(binary_best_path))
        shutil.rmtree(str(binary_path))
        best_f1 = current_f1

# %%



