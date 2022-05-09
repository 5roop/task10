
# %%

# import torch
# from numba import cuda
# cuda.select_device(0)
# cuda.close()
# cuda.select_device(0)
# torch.cuda.empty_cache()

# %%
label_set = ['Negative', 'Positive', 'Neutral']
STR_TO_NUM = {k: i for i, k in enumerate(label_set)}
NUM_TO_STR = {i:k for i, k in enumerate(label_set)}

import pandas as pd
df = pd.read_json("bcs_polsent.jsonl", orient="records", lines=True)
df["label"] = df.label.apply(lambda s: STR_TO_NUM[s])
df = df[["sentence", "label", "split"]].rename(columns={"sentence": "text", "label":"labels"})
train = df[df.split=="train"].drop(columns=["split"])
dev = df[df.split=="dev"].drop(columns=["split"])
test = df[df.split=="test"].drop(columns=["split"])


# %%
def train_model(train_df, model_name, model_type, batch_size, NUM_EPOCHS=5):
    batch_size=8
    from simpletransformers.classification import ClassificationModel
    import torch
    torch.cuda.empty_cache()
    model_args = {
        "num_train_epochs": NUM_EPOCHS,
        "learning_rate": 4e-5,
        "overwrite_output_dir": True,
        "train_batch_size": 8, 
        "no_save": True,
        "no_cache": True,
        "overwrite_output_dir": True,
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
    macroF1 = f1_score(y_true, y_pred, labels=test_df.labels.unique().tolist(), average="macro")

    return {"macroF1": macroF1, "y_true": y_true, "y_pred": y_pred}


# %%

for i in range(3):
    for epoch in [3,5,7,9,11,13,15,18,22, 33, 40, 45, 50]:
        for batch in [8]:
            for modeltype, modelname in zip(
                [ 
                    # "bert", 
                "xlmroberta", 
                # "electra"
                ],
                [
                    # "EMBEDDIA/crosloengual-bert",
                    "xlm-roberta-base",
                    # "classla/bcms-bertic",
                ],
            ):
                print(f"training {modelname},{modeltype},{epoch}")
                model = train_model(train, modelname, modeltype, batch, NUM_EPOCHS=epoch)
                print("Model trained. Evaluating.")
                
                stats = eval_model(model, dev)
                stats["eval_split"] = "dev"
                stats["model_name"] = modelname
                stats["epoch"] = epoch
                filename = "002_results_dev_test_xlmrobertabase2.jsonl"
                with open(filename, "a") as f:
                    f.write(f"{stats}\n")

                stats = eval_model(model, test)
                stats["eval_split"] = "test"
                stats["model_name"] = modelname
                stats["epoch"] = epoch
                with open(filename, "a") as f:
                    f.write(f"{stats}\n")
                del model

# %%



