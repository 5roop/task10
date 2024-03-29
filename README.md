 
Remarks:
* Use only first sheet
* Prepare a json dataset to push on CLARIN repo (train, dev, test), check distributions of `GOLD` label and `party`
* Where gold==Hard_disagreement: take value of reconciliation_hard
* Label set: {Positive, Neutral, Negative}
* Metric: macro F1


# Addendum 2022-05-03T13:20:15

The first iteration of the dataset is prepared and available [here](bcm_polsent.jsonl). The construction was done in a repeatable fashion and extensively described in the [notebook](001_dataset_preparation.ipynb) than generated the dataset.

# Meeting notes 2022-05-03T13:37:10

* Sentiment label distribution should be twice as important as label distribution, perform another split
* Hyperparams: keep batch size constant (8 AND 16), learning rate default, check the optimal epoch number.


# Addendum 2022-05-04T10:36:07

Hyperparameters identified so far:
* Fastext: epoch: 16, macroF1: 0.59
* Fastext with embeddings: epoch: 5, macroF1: 0.66


Performance on test:
* Fasttext:  macro f1 on test: 0.4715 +/- 0.0090, sample of 50
* Fasttext with embeddings: macro f1 with embeddings on test: 0.6312 +/- 0.0043, sample of 10

# Addendum 2022-05-04T16:06:52

Supposedly optimal hyperparameters:

| model_name                 | model_type   |   epochs |   batch_size |   macroF1 |
|:---------------------------|:-------------|---------:|-------------:|----------:|
| EMBEDDIA/crosloengual-bert | bert         |       15 |            8 |  0.848155 |
| classla/bcms-bertic        | electra      |       15 |            8 |  0.856857 |
| xlm-roberta-base           | xlmroberta   |       15 |            8 |  0.82363  |



# Addendum 2022-05-05T08:15:42

Performance on test data: 
| model_name                 | macroF1                           |
|:---------------------------|:----------------------------------|
| classla/bcms-bertic        | 0.7925 +/- 0.0126, sample size: 6 |
| EMBEDDIA/crosloengual-bert | 0.7683 +/- 0.0066, sample size: 6 |
| xlm-roberta-base           | 0.7620 +/- 0.0108, sample size: 6 |

# Addendum 2022-05-06T08:58:16

New hyperparams: go forward with 9 epochs. Repeat sweep for cse bert. 

# Addendum 2022-05-06T13:01:18

Bertic trained. Results:

|                                                  | ('macroF1', 'macroF1_stats')   |   ('macroF1', 'len') |
|:-------------------------------------------------|:-------------------------------|---------------------:|
| ('classla/bcms-bertic', 'train', 'test')         | 0.7837 ± 0.0125                |                    6 |
| ('classla/bcms-bertic', 'train', 'test_HR')      | 0.8143 ± 0.0148                |                    6 |
| ('classla/bcms-bertic', 'train', 'test_SRB')     | 0.7380 ± 0.0201                |                    6 |
| ('classla/bcms-bertic', 'train_HR', 'test_HR')   | 0.8020 ± 0.0103                |                    6 |
| ('classla/bcms-bertic', 'train_HR', 'test_SRB')  | 0.7495 ± 0.0250                |                    6 |
| ('classla/bcms-bertic', 'train_SRB', 'test_HR')  | 0.8247 ± 0.0155                |                    6 |
| ('classla/bcms-bertic', 'train_SRB', 'test_SRB') | 0.7553 ± 0.0128                |                    6 |



# Meeting notes 2022-05-09T14:11:16

TO DO:
* ✓ Add HR,SRB matrix for CSE Bert
* ✓ Add fastext with embeddings to table 1 (model comparison on train-test)
* ✓ Drop fasttext matrices. 
* Add dataset creation procedure in bullets.



# Meeting notes: 2022-05-17T15:50:43

* Resplit, stratify by country. ✓
* Retrain. - will after GPU is free.


# Meeting notes 2022-05-19T11:28:15

Figure 1: Best  bertic CM - replace with average performance. Title to be changed to sth like 'row normalized Bertic performance'
Replace the tables and figures to represent the new data results.

Binary experiments: Negative vs. Other
Just overall results: no interparliamentary results.
Confusion matrices.

For labelset in {full, binary}:
    train on full train
    evaluate on {full, HR, BiH, SRB}{test}

expected table: check overleaf


# Addendum 2022-05-27T15:32:35
TO DO: replot confusion matrices with raw numbers, keep the old available

# Addendum 2022-05-31T09:55:35

When trying to train and save Bertic models on ternary and binary full data, I noticed that the F1 is terrible (was 0.8, now 0.15). I have yet to debug this issue. So far I've tried removing the cache, next step will be downgrading the dataset to `bcs_polsent.jsonl`.

None of that worked. I'll try downgrading the simpletransformers.


# Addendum 2022-06-01T08:31:11
None of the above worked. After updating, the training just takes forever, I'm not sure anything is happening. The `ps` command shows the processes have been running for 9 seconds, regardless of when we run it. No output is produced in output directories, suggesting that the model doesn't start training.



# Addendum 2022-06-01T09:13:57
I fiddled with the model args a bit. I don't know what I did, but I managed to fix it. For my troubles the universe also fixed evaluation speeds for me, now they are super fast. 

# Addendum 2022-06-02T07:11:51
The models are published on huggingface under the same urls as in the paper: [ternary](https://huggingface.co/classla/bcms-bertic-parlasent-bcs-ter) and [binary](https://huggingface.co/classla/bcms-bertic-parlasent-bcs-bi)

# Addendum 2022-06-06T13:05:16

The paper is submitted, and also available on arxiv. The next step is to prepare the dataset for publication in a way that corresponds to the setup described in the paper.