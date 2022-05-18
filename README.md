 
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
* Retrain.
