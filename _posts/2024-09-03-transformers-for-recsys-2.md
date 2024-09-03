---
title: Transformers for Recommender Systems - Part 2
description: >-
  Improve model to reduce duplicates and improve training performance 
date: 2024-09-03
categories: [Blog, Tutorial]
tags: [AI, Machine Learning, RecSys, Transformers]
pin: true
math: false
author: ks
---

This is continuation of the previous post on [Transformers for Recommender Systems](../transformers-for-recsys-1). In this post, we will improve the model to reduce duplicates and improve training performance. In addition, we will continue to test metrics to see if our model has improved. We will look at offline training and validation metrics.

All the code is available on my Github repository - [Recformers](https://github.com/kapilsh/recformer).


## Model with Repetition Penalty

We trained a new model where we added penalty for the model whenever the model returns a prediction that already exists in the input tokens.

### New Loss function

The major change in the new model is the loss function. In the new loss function, we calculate the standard cross-entropy loss and add a penalty term for tokens that are repeated from the input sequence. The penalty term is controlled by a hyperparameter `penalty_weight`. Higher the penalty weight, higher the penalty for duplicates.

Here is the loss function that we defined for the model to penalize duplicates.

```python
class DedupCrossEntropyLoss(nn.Module):
    """
    Custom loss function that combines cross-entropy loss with a penalty term
    for tokens that are repeated from the input sequence.
    """
    def __init__(self, penalty_weight = 1.0):
        super(DedupCrossEntropyLoss, self).__init__()
        self.penalty_weight = penalty_weight
        self.ce_loss = nn.CrossEntropyLoss()
        logger.info(f"Penalty weight: {penalty_weight}")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, input_tokens: torch.Tensor):
        # Calculate the standard cross-entropy loss
        ce_loss = self.ce_loss(logits, labels)
        if self.penalty_weight == 0:
            return ce_loss
        token_output = torch.argmax(logits, dim=1)
        duplicated_masks = torch.eq(input_tokens, token_output.unsqueeze(-1)).any(dim=-1).float()
        penalty = duplicated_masks * self.penalty_weight
        loss = ce_loss + penalty.mean()
        return loss
```

### Model training diff

I played around with a few different learning rates and penalty factor and finally landed on the numbers below. We show the diff from the original model trained in previous post below:

```diff
-> % git --no-pager diff training_config.yaml 
diff --git a/movielens-ntp/training_config.yaml b/movielens-ntp/training_config.yaml
index 235e7b5..9dff23a 100644
--- a/movielens-ntp/training_config.yaml
+++ b/movielens-ntp/training_config.yaml
@@ -2,12 +2,13 @@ trainer_config:
   data_dir: ./data/ml-1m
   model_dir: ./models
   batch_size: 512
-  starting_learning_rate: 0.0005
+  starting_learning_rate: 0.0008
   learning_rate_decay: 0.95
   device: cuda
   num_epochs: 1000
   validation_fraction: 0.15
   tensorboard_dir: ./runs
+  penalize_duplicates_factor: 0.2
 
 movie_transformer_config:
   context_window_size: 5

```

Finally, we trained the model:

```shell
python model_train.py --config_file=./training_config.yaml --penalize-duplicates
```

Next, we will evaluate our newly trained model. Let's define a few simple functions first!

### Imports

```python
import torch
import torch.nn.functional as F
from model_train import run_model_training, load_config, get_model_config
from data import MovieLensSequenceDataset
from torch.utils.data import DataLoader
from tbparse import SummaryReader
import plotly.express as px
from eval import (
    get_model_predictions,
    get_model_predictions,
    calculate_metrics,
    calculate_relevance,
)
import pandas as pd
import numpy as np
import torch.nn as nn
from movielens_transformer import MovieLensTransformer
```

### Load model artifacts

```python
def load_model_artifacts(model_file: str, config_file: str):
    config_file = "./training_config.yaml"
    config = load_config(config_file)
    sequence_length = config["movie_transformer_config"]["context_window_size"]
    batch_size = config["trainer_config"]["batch_size"]
    valid_dataset = MovieLensSequenceDataset(
        movies_file="./data/ml-1m/movies.dat",
        users_file="./data/ml-1m/users.dat",
        ratings_file="./data/ml-1m/ratings.dat",
        sequence_length=sequence_length,
        window_size=1,
        is_validation=True,
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    with open(model_file, "rb") as f:
        model_state_dict = torch.load(f, weights_only=True)
        model_config = get_model_config(config, valid_dataset)
    trained_model = MovieLensTransformer(model_config)
    trained_model.load_state_dict(model_state_dict)
    return config, trained_model, valid_dataloader
```

### Model predictions

```python
def predict_next_movie(model, movie_ids, user_ids):
    model.eval()
    logits = model(movie_ids=movie_ids, user_ids=user_ids)
    probabilities = F.softmax(logits, dim=-1)
    predicted_movie_ids = torch.argmax(probabilities, dim=-1)
    return predicted_movie_ids
```

### Read Tensorboard logs

```python
def read_tensorboard_logs(log_file: str):
    reader = SummaryReader(log_file)
    metrics = reader.scalars
    return metrics
```

## Comparison

Let's define a dictionary where we save the model files and tensorboard logs for the baseline model and the model with duplication penalty.

```python
models = {
    "baseline": (
        "./models/model_1000e_512_32_32_4_4.pth",
        "./models/events.out.tfevents.1724703053.kapilsh-dev-big.514232.0",
    ),
    "with_duplication_penalty": (
        "./models/model_1000e_512_32_32_4_4_w_pen.pth",
        "./models/events.out.tfevents.1725245582.kapilsh-dev-big.1079645.0",
    ),
}
```

### Populate ranking metrics

This function is almost identical to the one we used in the previous post. We calculate the ranking metrics - `MRR`, `MAP`, and `NDCG` for the models.

```python
def populate_ranking_metrics(
    model_name, model, valid_dataloader, sequence_length, k_values
):
    metrics = {}

    for k in k_values:
        # we would torch.cat these later
        model_relevances = []
        model_scores = []

        for batch in valid_dataloader:
            (
                movie_id_tokens,
                rating_id_tokens,
                user_id_tokens,
                output_movie_id_tokens,
                output_rating_id_tokens,
            ) = batch
            model_predictions = get_model_predictions(
                model, movie_id_tokens, user_id_tokens, n=k
            )
            model_relevance = calculate_relevance(
                model_predictions.predictions, output_movie_id_tokens
            )
            model_relevances.append(model_relevance)
            model_scores.append(model_predictions.scores)

        model_relevances_tensor = torch.cat(model_relevances)
        model_scores_tensor = torch.cat(model_scores)

        # Calculate the metrics
        model_metrics = calculate_metrics(model_relevances_tensor, model_scores_tensor)

        metrics[k] = model_metrics

    metrics_df = pd.DataFrame(
        [
            {
                "k": k,
                f"MRR_{model_name}": v.MRR,
                f"MAP_{model_name}": v.MAP,
                f"NDCG_{model_name}": v.NDCG,
            }
            for k, v in metrics.items()
        ]
    )
    return metrics_df
```

### Let's plot the loss curves and print out the ranking metrics

```python
k_values = [3, 5, 10]

for model_name, (model_file, tb_file) in models.items():
    print(f"======= Model: {model_name} =======")
    config, model, valid_dataloader = load_model_artifacts(
        model_file, "./training_config.yaml"
    )
    metrics = read_tensorboard_logs(tb_file)
    metrics_by_run = metrics.groupby(["step", "tag"]).mean().unstack()["value"]
    metrics_by_run.columns.name = None
    fig = px.line(metrics_by_run, title=f"Model: {model_name}")
    fig.show()

    print("======= Ranking Metrics =======")
    ranking_metrics = populate_ranking_metrics(
        model_name,
        model,
        valid_dataloader,
        config["movie_transformer_config"]["context_window_size"],
        k_values,
    )
    print(ranking_metrics)
```

### Baseline Model

#### Loss curves

![Baseline Model loss](/assets/movielens-ntp/baseline_loss.png)

#### Ranking Metrics

|   |  k |       MRR |      MAP |      NDCG |
|--:|---:|----------:|---------:|----------:|
| 0 |  3 | 0.0738384 | 0.113106 | 0.0838683 |
| 1 |  5 | 0.0869578 | 0.167913 |  0.106887 |
| 2 | 10 |  0.098746 | 0.265052 |  0.137353 |

### Model with Duplication Penalty

#### Loss curves

![Model with Duplication Penalty loss](/assets/movielens-ntp/duplication_penalty_loss.png)

#### Ranking Metrics

|   |  k |       MRR |      MAP |      NDCG |
|--:|---:|----------:|---------:|----------:|
| 0 |  3 | 0.0713242 | 0.109596 | 0.0811038 |
| 1 |  5 | 0.0832133 | 0.161164 |  0.102398 |
| 2 | 10 |  0.095786 | 0.259384 |  0.133728 |

> Based on these results, the new model seems worse! Let's check if the dedup logic is actually working. We will check how many predicted values are duplicated in baseline vs duplication_penalty model to root-cause.
{: .prompt-warning}


Let's define a function to get the duplicated movies for a given model.

```python
def get_duplicated_movies(k: int, model: nn.Module, valid_dataloader: DataLoader):
    duplicated_movies = []
    for i, batch in enumerate(valid_dataloader):
        movie_id_tokens, rating_ids, user_id_tokens, movie_targets, rating_targets = (
            batch
        )
        with torch.no_grad():
            # batch x num_tokens
            output = model(movie_id_tokens, user_id_tokens)
        output_probabilites = F.softmax(output, dim=-1)
        _, top_tokens = output_probabilites.topk(k, dim=-1)

        for i in range(movie_id_tokens.shape[0]):
            input_tokens = movie_id_tokens[i]
            output_tokens = top_tokens[i]
            concat_tensor, counts = torch.cat([input_tokens, output_tokens]).unique(
                return_counts=True
            )
            intersection = concat_tensor[torch.where(counts.gt(1))]
            if intersection.shape[0] > 0:
                duplicated_movies.append(intersection)

    return torch.cat(duplicated_movies)
```

Now lets check the duplicated movies for both models for different values of k:

```python
k_values = [3, 5, 10]

for model_name, (model_file, tb_file) in models.items():
    config, model, valid_dataloader = load_model_artifacts(
        model_file, "./training_config.yaml"
    )

    for k in k_values:
        duplicated_movies = get_duplicated_movies(k, model, valid_dataloader)
        average_duplications = duplicated_movies.shape[0] / (
            k * len(valid_dataloader.dataset)
        )
        print(f"[{model_name}] Average Duplications @ {k}: ", average_duplications)
```

```
[baseline] Average Duplications @ 3:  0.1470205989104597
[baseline] Average Duplications @ 5:  0.13704773640936546
[baseline] Average Duplications @ 10:  0.11827639845659683

[with_duplication_penalty] Average Duplications @ 3:  0.14305713358221067
[with_duplication_penalty] Average Duplications @ 5:  0.1336653952280474
[with_duplication_penalty] Average Duplications @ 10:  0.11541450251582536
```

**✅ We do see fewer repetitions.** 

So, our penalty term is working. We might need to tweak some hyperparameters or let the model train longer. At this point, we might just push ahead make some other changes to the model to improve training efficiency for learning and performance.

For example,

- **Learning rate scheduler:** We slightly increased learning rate in penalty and that made model learn faster. We could consider a learning rate scheduler to have higher learning rate initially and decay it as the model trains
- **Compilation:** `torch.compile` should significantly improve our model training performance

I will check-point the repo now. We will work on the above improvements next.

## Model Improvements

### Update Learning rate and apply `torch.compile`

I updated the model with a learning rate scheduler to `CosineAnnealingLR`. Learning rate schedulers provide a way for us to adjust learning rate during the training run. See [How to adjust learning rate](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) for details.

In addition, to improve the model training performance, I leveraged `torch.compile` to compile the model before model training and validation. GPU utilization was fairly poor with the previous model so I increased the batch size, and made some more tweaks to the hyperparameters.

```diff

$ git --no-pager diff d931432667e82d955b8bd4c3d803f9b1974ac924..5411b7d65be81e67a5db140f5deebf9548f67e9d -- ./training_config.yaml
diff --git a/movielens-ntp/training_config.yaml b/movielens-ntp/training_config.yaml
index 9dff23a..f6e285f 100644
--- a/movielens-ntp/training_config.yaml
+++ b/movielens-ntp/training_config.yaml
@@ -1,14 +1,14 @@
 trainer_config:
   data_dir: ./data/ml-1m
   model_dir: ./models
-  batch_size: 512
-  starting_learning_rate: 0.0008
-  learning_rate_decay: 0.95
+  batch_size: 4096
+  starting_learning_rate: 0.0005
+  learning_rate_decay: 0.99
   device: cuda
-  num_epochs: 1000
+  num_epochs: 500
   validation_fraction: 0.15
   tensorboard_dir: ./runs
-  penalize_duplicates_factor: 0.2
+  penalize_duplicates_factor: 0.1

```

### Changes to training script

Here we add the learning rate scheduler and update the learning rate after each epoch. In addition, we compile the model before training and validation.

```diff
$ git --no-pager diff d931432667e82d955b8bd4c3d803f9b1974ac924..5411b7d65be81e67a5db140f5deebf9548f67e9d -- ./model_train.py                                                                                                                                                 
diff --git a/movielens-ntp/model_train.py b/movielens-ntp/model_train.py                                                                
index 49b9a63..2e8e0c5 100644     
--- a/movielens-ntp/model_train.py  
+++ b/movielens-ntp/model_train.py           
@@ -12,6 +12,7 @@ import numpy as np                                                                                                    
 from tqdm import trange
 from torch.utils.tensorboard import SummaryWriter
 from loss import DedupCrossEntropyLoss                                                                                                 
+from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR                                                      
                                                                    
                      
 def load_config(config_file: str):
@@ -103,6 +104,8 @@ def get_model_config(config: dict, dataset: Dataset) -> MovieLensTransformerConf
                                                       
                                                                                                                                        
 def run_model_training(config: dict, penalize_duplicates: bool = False):                                                               
+    torch.set_float32_matmul_precision("high")                                                                                         
+                               
     device = config["trainer_config"]["device"]
     if device == "cuda" and not torch.cuda.is_available():
         raise ValueError("CUDA is not available")
@@ -134,12 +137,19 @@ def run_model_training(config: dict, penalize_duplicates: bool = False):
         model.parameters(), lr=config["trainer_config"]["starting_learning_rate"]
     )
  
+    scheduler = CosineAnnealingLR(
+        optimizer=optimizer,
+        T_max=100,
+        eta_min=0.000001,
+    )
     best_validation_loss = np.inf
  
     writer = SummaryWriter(
         log_dir=config["trainer_config"]["tensorboard_dir"], flush_secs=30
     )
  
+    compiled_model = torch.compile(model, fullgraph=True, mode="max-autotune")
+
     for epoch in range(config["trainer_config"]["num_epochs"]):
         model.train()
         total_loss = 0.0
@@ -157,7 +167,7 @@ def run_model_training(config: dict, penalize_duplicates: bool = False):
                 movie_ids,
                 user_ids,
                 movie_targets,
-                model,
+                compiled_model,
                 optimizer,
                 criterion,
                 device,
@@ -190,7 +200,7 @@ def run_model_training(config: dict, penalize_duplicates: bool = False):
                 movie_ids,
                 user_ids,
                 movie_targets,
-                model,
+                compiled_model,
                 criterion,
                 device,
             )
+        # update the learning rate
+        scheduler.step()
+        logger.info(f"Learning rate: {scheduler.get_last_lr()[0]}")
+
 
```

### Results with torch.compile and learning rate scheduler

Following the above changes, I trained the model for 500 epochs (as in the config).  Let's look at the model results after these changes. 

```python
model_file = "./models/model_500e_4098_32_32_4_4_compiled_cos_ann_lr.pth"
tb_file = "./models/events.out.tfevents.1725310210.kapilsh-dev-big.1156831.0"
model_name = "compiled_cos_ann_lr_with_duplication_penalty"
config, model, valid_dataloader = load_model_artifacts(
    model_file, "./training_config.yaml"
)
metrics = read_tensorboard_logs(tb_file)
metrics_by_run = metrics.groupby(["step", "tag"]).mean().unstack()["value"]
metrics_by_run.columns.name = None
fig = px.line(metrics_by_run, title=f"Model: {model_name}")
fig.show()

print("======= Ranking Metrics =======")
ranking_metrics = populate_ranking_metrics(
    model_name,
    model,
    valid_dataloader,
    config["movie_transformer_config"]["context_window_size"],
    k_values,
)
print(ranking_metrics)
```

#### Loss curves

![Compiled Model loss](/assets/movielens-ntp/duplication_penalty_compiled_loss.png)

#### Ranking Metrics

|   |  k |       MRR |      MAP |      NDCG |
|--:|---:|----------:|---------:|----------:|
| 0 |  3 | 0.0734953 | 0.112638 | 0.0834948 |
| 1 |  5 | 0.0852096 | 0.164966 |  0.104846 |
| 2 | 10 | 0.0983773 | 0.265874 |  0.137209 |


> It seems there isn't any improvement in the model after these changes. Although, my training run finished in half the time - so that's a win! 🚀🚀🚀 
> 
> If we look at the ranking metrics, they are in the same ballpark as the previous run. In fact, they look slightly worse.
{: .prompt-info}

Let's look at the duplication metric next.

```python
for k in k_values:
    duplicated_movies = get_duplicated_movies(k, model, valid_dataloader)
    average_duplications = duplicated_movies.shape[0] / (
        k * len(valid_dataloader.dataset)
    )
    print(f"[{model_name}] Average Duplications @ {k}: ", average_duplications)
```

```
[with_duplication_penalty] Average Duplications @ 3:  0.14272585985539563
[with_duplication_penalty] Average Duplications @ 5:  0.1329353334684776
[with_duplication_penalty] Average Duplications @ 10:  0.11556051084532738
```

These metrics also seem slightly worse that the previous run. 📉📉📉 

It is possible that the model needs to train further since the validation loss is still decreasing in the last few epochs. I did not want to waste away the learning from this training run so I added support for model to be able to start training from a checkpoint.

```diff
$ git --no-pager diff 5411b7d65be81e67a5db140f5deebf9548f67e9d model_train.py
diff --git a/movielens-ntp/model_train.py b/movielens-ntp/model_train.py
index 2e8e0c5..99b0861 100644
--- a/movielens-ntp/model_train.py
+++ b/movielens-ntp/model_train.py
@@ -1,4 +1,5 @@
 import os
+from typing import Optional
 import click
 from movielens_transformer import MovieLensTransformer, MovieLensTransformerConfig
 from torch.utils.data import DataLoader, Dataset
@@ -103,7 +104,7 @@ def get_model_config(config: dict, dataset: Dataset) -> MovieLensTransformerConf
     return model_config
 
 
-def run_model_training(config: dict, penalize_duplicates: bool = False):
+def run_model_training(config: dict, penalize_duplicates: bool = False, checkpoint: Optional[str] = None):
     torch.set_float32_matmul_precision("high")
 
     device = config["trainer_config"]["device"]
@@ -122,7 +123,11 @@ def run_model_training(config: dict, penalize_duplicates: bool = False):
 
     model = MovieLensTransformer(config=model_config)
 
-    init_weights(model)
+    if not checkpoint:
+        init_weights(model)
+    else:
+        model.load_state_dict(torch.load(checkpoint))
+    
     model.to(device)
 
     criterion = DedupCrossEntropyLoss(
@@ -238,9 +243,10 @@ def run_model_training(config: dict, penalize_duplicates: bool = False):
     is_flag=True,
     help="penalize duplicates in the output token",
 )
-def main(config_file: str, penalize_duplicates: bool):
+@click.option("--checkpoint", help="checkpoint filename", required=False)
+def main(config_file: str, penalize_duplicates: bool, checkpoint: Optional[str] = None):
     config = load_config(config_file)
-    run_model_training(config, penalize_duplicates)
+    run_model_training(config, penalize_duplicates, checkpoint)

```

Let's look at its loss metrics after running training run from the checkpoint.

```shell
$ python model_train.py --config_file=./training_config.yaml --penalize-duplicates --checkpoint=models/model_500e_4098_32_32_4_4_compiled_cos_ann_lr.pth
```

```python
model_file = (
    "./models/model_500e_4098_32_32_4_4_compiled_cos_ann_lr_from_checkpoint.pth"
)
tb_file = "./models/events.out.tfevents.1725330223.kapilsh-dev-big.1178851.0"
model_name = "compiled_cos_ann_lr_with_duplication_penalty_from_checkpoint"
metrics = read_tensorboard_logs(tb_file)
metrics_by_run = metrics.groupby(["step", "tag"]).mean().unstack()["value"]
metrics_by_run.columns.name = None
fig = px.line(metrics_by_run, title=f"Model: {model_name}")
fig.show()
```

#### Loss curves

![Compiled Model from checkpoint loss](/assets/movielens-ntp/duplication_penalty_compiled_from_checkpoint_loss.png)

We see that the model loss improved slightly but not a lot. I wanted to test the duplication metric as well, however, it seems currently it is not possible to save/load compiled_model without some effort. See discussion [forum](https://discuss.pytorch.org/t/how-to-save-load-a-model-with-torch-compile/179739) and [on github issue](https://github.com/pytorch/pytorch/issues/101107#issuecomment-1542688089).

So, I will continue iterating on this model for now. We have just used the movie id sequences here. There is more potential to learn from user features, user-movie features, etc. So, we will tackle that next.

## Summary

In this post, we improved the model to reduce duplicates and improve training performance. We added a penalty for the model whenever the model returns a prediction that already exists in the input tokens. 

- We trained the model with a new loss function and compared the results with the previous model. 
- We also looked at the duplication metric to see if the penalty term is working and introduced learning rate scheduler and `torch.compile` to improve training performance.
- We added support for model to be able to start training from a checkpoint. 

### Results

- The new model seems slightly worse than the previous model but we didn't spend a lot of time on hyperparameter tuning.
- The duplication metric shows that the penalty term is working and we have fewer repetitions in the output tokens.
