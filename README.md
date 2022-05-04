 
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

