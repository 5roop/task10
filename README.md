 
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
