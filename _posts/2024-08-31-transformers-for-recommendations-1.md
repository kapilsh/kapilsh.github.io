---
title: Transformers for Recommendations
description: >-
  Exploring a simple transformer model for sequence modelling in recommendation systems 
date: 2024-08-31
categories: [Blog, Tutorial]
tags: [AI, Machine Learning, RecSys, Transformers]
pin: true
math: false
author: ks
---

In this post, we will train a transformer model for recommendation systems from scratch. We will use the [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) dataset. We will leverage some of the code that we already have for transformers in my previous posts - [Post 1](../transformers-from-scratch) and [Post 2](../exploring-gpt2).

We will adopt the previous code and modify it to suit the recommendation system use case.

You can refer to my Github repository for the code - [Recformers](https://github.com/kapilsh/recformer).

## Setup

Let's start by installing the required libraries. I have provided a [`requirements.txt`](https://github.com/kapilsh/recformer/blob/main/movielens-ntp/requirements.txt) file in the repository. This was generated via command:

```bash
conda list -e > requirements.txt
```
Let's do all the imports first. It will include some of the modules that we haven't defined yet. We will in the upcoming sections.

```python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from imageio.v3 import imread
from collections import Counter
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from data import MovieLensSequenceDataset
from movielens_transformer import (
    MovieLensTransformer,
    MovieLensTransformerConfig,
    TransformerConfig,
)
import torch.nn.functional as F
from dacite import from_dict
import pprint
from model_train import run_model_training, load_config, get_model_config
from eval import (
    get_model_predictions,
    get_popular_movies,
    get_popular_movie_predictions,
    calculate_relevance,
    calculate_metrics,
)
from sklearn.metrics import ndcg_score
```

## Movielens Dataset

We will use the MovieLens-1M dataset for this tutorial. You can download the dataset from [here](https://grouplens.org/datasets/movielens/1m/). The dataset contains 1 million ratings from 6000 users on 4000 movies. The dataset also contains movie metadata like genres. Let's explore the dataset.

There are 3 main files in the dataset:
- `users.dat` - Contains user information
- `ratings.dat` - Contains user ratings for movies
- `movies.dat` - Contains movie information

```python
users = pd.read_csv(
    "./data/ml-1m/users.dat",
    sep="::",
    names=["user_id", "sex", "age_group", "occupation", "zip_code"],
    encoding="ISO-8859-1",
    engine="python",
    dtype={
        "user_id": np.int32,
        "sex": "category",
        "age_group": "category",
        "occupation": "category",
        "zip_code": str,
    },
)

ratings = pd.read_csv(
    "./data/ml-1m/ratings.dat",
    sep="::",
    names=["user_id", "movie_id", "rating", "unix_timestamp"],
    encoding="ISO-8859-1",
    engine="python",
    dtype={
        "user_id": np.int32,
        "movie_id": np.int32,
        "rating": np.int8,
        "unix_timestamp": np.int32,
    },
)

movies = pd.read_csv(
    "./data/ml-1m/movies.dat",
    sep="::",
    names=["movie_id", "title", "genres"],
    encoding="ISO-8859-1",
    engine="python",
    dtype={"movie_id": np.int32, "title": str, "genres": str},
)
```

```python
print(users.head())
print(ratings.head())
print(movies.head())
```

### Users

|    |   user_id | sex   |   age_group |   occupation |   zip_code |
|---:|----------:|:------|------------:|-------------:|-----------:|
|  0 |         1 | F     |           1 |           10 |      48067 |
|  1 |         2 | M     |          56 |           16 |      70072 |
|  2 |         3 | M     |          25 |           15 |      55117 |
|  3 |         4 | M     |          45 |            7 |      02460 |
|  4 |         5 | M     |          25 |           20 |      55455 |


### Ratings

|    |   user_id |   movie_id |   rating |   unix_timestamp |
|---:|----------:|-----------:|---------:|-----------------:|
|  0 |         1 |       1193 |        5 |      9.78301e+08 |
|  1 |         1 |        661 |        3 |      9.78302e+08 |
|  2 |         1 |        914 |        3 |      9.78302e+08 |
|  3 |         1 |       3408 |        4 |      9.783e+08   |
|  4 |         1 |       2355 |        5 |      9.78824e+08 |

### Movies

|    |   movie_id | title                              | genres                       |
|---:|-----------:|:-----------------------------------|:-----------------------------|
|  0 |          1 | Toy Story (1995)                   | Animation|Children's|Comedy  |
|  1 |          2 | Jumanji (1995)                     | Adventure|Children's|Fantasy |
|  2 |          3 | Grumpier Old Men (1995)            | Comedy|Romance               |
|  3 |          4 | Waiting to Exhale (1995)           | Comedy|Drama                 |
|  4 |          5 | Father of the Bride Part II (1995) | Comedy                       |

We also downloaded the movie posters for the dataset from [a Kaggle dataset](https://www.kaggle.com/datasets/ghrzarea/movielens-20m-posters-for-machine-learning). It will be nice to show posters of the movies watched and recommended :). I symlinked these posters to `./data/posters`.

```python
image_urls = [os.path.join("./data/posters", f"{i}.jpg") for i in movies.movie_id]
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, (ax, img) in enumerate(
    zip(axs.flat, np.array(image_urls)[[5, 10, 15]].tolist())
):
    ax.imshow(imread(img))  # Display the image
    ax.axis("off")  # Turn off axis
plt.show()
```

![Explorations Posters](/assets/movielens-ntp/exploration_posters.png)

## Tokenization

Let's now tokenize the user_ids and movie_ids such that both map on to 0:N space, where N is the number of unique entities.

```python
# user ids
unique_user_ids = users["user_id"].unique()
unique_user_ids.sort()

# movie ids
unique_movie_ids = movies["movie_id"].unique()
unique_movie_ids.sort()

# tokenization
user_id_tokens = {user_id: i for i, user_id in enumerate(unique_user_ids)}
movie_id_tokens = {movie_id: i for i, movie_id in enumerate(unique_movie_ids)}
```

## Sequence Generation

We will generate sequences of user_ids and movie_ids for each user. In this part, we will order the movies by timestamp and get full sequence for each user.

```python
ratings["user_id_tokens"] = ratings["user_id"].map(user_id_tokens)
ratings["movie_id_tokens"] = ratings["movie_id"].map(movie_id_tokens)

ratings_ordered = (
    ratings[["user_id_tokens", "movie_id_tokens", "unix_timestamp", "rating"]]
    .sort_values(by="unix_timestamp")
    .groupby("user_id_tokens")
    .agg(list)
    .reset_index()
)

print(ratings_ordered.head(2))
```

|    |   user_id_tokens | movie_id_tokens                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | unix_timestamp                                               | rating               |
|---:|-----------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------|:---------------------|
|  0 |                0 | [3117, 1672, 1250, 1009, 2271, ...]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | [978300019, 978300055, 978300055, 978300055, 978300103, ...] | [4, 4, 5, 5, 3, ...] |
|  1 |                1 | [1180, 1199, 1192, 2648, 1273, ...]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | [978298124, 978298151, 978298151, 978298196, 978298261, ...] | [4, 3, 4, 3, 5, ...] |

For the model training, we will feed a sequence movie_ids to the model and predict the next movie_id. Let's assume we have a `sequence_length = 5`. Let's first generate a sequence for one user_id and make sure we generate the correct sequences. Then we can expand that to all the data.

```python

sequence_length = 5
min_sequence_length = 1
window_size = 1

sample_data = ratings_ordered.iloc[0]
sample_movie_ids = torch.tensor(sample_data.movie_id_tokens, dtype=torch.int32)
sample_ratings = torch.tensor(sample_data.rating, dtype=torch.int8)
sample_movie_sequences = (
    sample_movie_ids.ravel().unfold(0, sequence_length, 1).to(torch.int32)
)
sample_rating_sequences = (
    sample_ratings.ravel().unfold(0, sequence_length, 1).to(torch.int8)
)

print(sample_movie_sequences)
```
which produces:

```
tensor([[3117, 1672, 1250, 1009, 2271],
        [1672, 1250, 1009, 2271, 1768],
        ...
        [ 773, 2225, 2286, 1838, 1526],
        [2225, 2286, 1838, 1526,   47]], dtype=torch.int32)
```

This looks good. The movie ids are shifted by 1 in the sequence as we move row by row.

Let's expand on this and generate sequences for all the users.

```python
def generate_sequences(row, sequence_length, window_size):
    movie_ids = torch.tensor(row.movie_id_tokens, dtype=torch.int32)
    ratings = torch.tensor(row.rating, dtype=torch.int8)
    movie_sequences = (
        movie_ids.ravel().unfold(0, sequence_length, window_size).to(torch.int32)
    )
    rating_sequences = (
        ratings.ravel().unfold(0, sequence_length, window_size).to(torch.int8)
    )
    return (movie_sequences, rating_sequences)

for i, row in ratings_ordered.iterrows():
    movie_sequences, rating_sequences = generate_sequences(
        row, sequence_length, window_size
    )
    print(movie_sequences)
    print(rating_sequences)
    break
```

```
tensor([[3117, 1672, 1250, 1009, 2271],
        ...
        [2225, 2286, 1838, 1526,   47]], dtype=torch.int32)
tensor([[4, 4, 5, 5, 3],
        ...
        [4, 5, 4, 4, 5]], dtype=torch.int8)
```

I have converted all the scrappy code into a proper `Dataset` class. We will call this `MovieLensSequenceDataset`. Refer to the code in the repository (`data.py`) for the full implementation.

```python
dataset = MovieLensSequenceDataset(
    movies_file="./data/ml-1m/movies.dat",
    users_file="./data/ml-1m/users.dat",
    ratings_file="./data/ml-1m/ratings.dat",
    sequence_length=5,
    window_size=1,
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
batch = next(iter(dataloader))
for t in batch:
    print(t)
    print(t.shape)
    print("-----------------------------------")
```

```
2024-09-01 06:43:22.424 | INFO     | data:__init__:89 - Creating MovieLensSequenceDataset with validation set: %s
2024-09-01 06:43:22.425 | INFO     | data:read_movielens_data:12 - Reading data from files
2024-09-01 06:43:24.184 | INFO     | data:_add_tokens:140 - Adding tokens to data
2024-09-01 06:43:24.220 | INFO     | data:_generate_sequences:159 - Generating sequences
2024-09-01 06:43:24.899 | INFO     | data:__init__:110 - Train data length: 883277
2024-09-01 06:43:24.899 | INFO     | data:__init__:111 - Validation data length: 98812
tensor([[3883, 3117, 1672, 1250, 1009],
        [3117, 1672, 1250, 1009, 2271],
        [1672, 1250, 1009, 2271, 1768],
        [1250, 1009, 2271, 1768, 3339]], dtype=torch.int32)
torch.Size([4, 5])
-----------------------------------
tensor([[0, 4, 4, 5, 5],
        [4, 4, 5, 5, 3],
        [4, 5, 5, 3, 5],
        [5, 5, 3, 5, 4]], dtype=torch.int8)
torch.Size([4, 5])
-----------------------------------
tensor([0, 0, 0, 0], dtype=torch.int32)
torch.Size([4])
-----------------------------------
tensor([2271, 1768, 3339, 1189], dtype=torch.int32)
torch.Size([4])
-----------------------------------
tensor([3, 5, 4, 4], dtype=torch.int8)
torch.Size([4])
-----------------------------------
```

## Model

### Model Definition

Let build the transformer based recommendation model now. We can reuse most of the code that I wrote for my previous post on [Exploring GPT](https://www.kapilsharma.dev/posts/exploring-gpt2/). We will skip defining the `CausalMultiHeadAttention`, `MLP`, `TransformerEncoderLayer` (defined as `GPT2Layer` in previous post), and `Transformer` (defined as `GPT2` in the previous post). Instead, let me just show the differences between what the GPT2 for language model defined vs here.

We will skip the `output_layer` in the `Transformer` i.e. just remove it. We need an interaction layer between the movies and users in rec sys model. Hence, the output layer will be after the interaction. This interaction layer will be defined as an MLP (similar to the one we defined in [DLRM exploration](https://www.kapilsharma.dev/posts/cuda-mode-fusing-kernels-talk/)).

```python
class InteractionMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super(InteractionMLP, self).__init__()
        fc_layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                fc_layers.append(nn.Linear(input_size, hidden_size))
            else:
                fc_layers.append(nn.Linear(hidden_sizes[i - 1], hidden_size))
            fc_layers.append(nn.ReLU())
        fc_layers.append(
            nn.Linear(
                hidden_sizes[-1] if hidden_sizes else input_size,
                output_size,
                bias=False,
            )
        )
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor):
        return self.fc_layers(x)

```

### Final Movielens RecSys Model

```python
class MovieLensTransformer(nn.Module):
    def __init__(self, config: MovieLensTransformerConfig):
        super().__init__()
        self.movie_transformer = Transformer(config.movie_transformer_config)
        self.user_embedding = nn.Embedding(
            config.num_users, config.user_embedding_dimension
        )

        self.output_layer = InteractionMLP(
            config.movie_transformer_config.embedding_dimension
            + config.user_embedding_dimension,
            config.interaction_mlp_hidden_sizes,
            config.movie_transformer_config.vocab_size,
        )

    def forward(self, movie_ids: torch.Tensor, user_ids: torch.Tensor):
        movie_embeddings = self.movie_transformer(movie_ids)
        user_embeddings = self.user_embedding(user_ids)
        embeddings = torch.cat([movie_embeddings, user_embeddings], dim=-1)
        return self.output_layer(embeddings)  # returns logits
```

### Quick Test

Next, let's test the model that we defined to see if we can run a sample batch through it.

Consider a sample model with 2 transformer encoder layers with 2 heads each, embedding dimension of 32, and a user embedding dimension of 32. We will define the model config as below:

```python
movie_ids, ratings, user_ids, output_movie_ids, output_ratings = batch
config_json = {
    "movie_transformer_config": {
        "vocab_size": len(dataset.metadata.movie_id_tokens),
        "context_window_size": 5,
        "embedding_dimension": 32,
        "num_layers": 2,
        "num_heads": 2,
        "dropout_embeddings": 0.1,
        "dropout_attention": 0.1,
        "dropout_residual": 0.1,
        "layer_norm_epsilon": 1e-5,
    },
    "user_embedding_dimension": 32,
    "num_users": len(dataset.metadata.user_id_tokens),
    "interaction_mlp_hidden_sizes": [16],
}

config = from_dict(data_class=MovieLensTransformerConfig, data=config_json)
```

Now, let's construct this model:

```python
model = MovieLensTransformer(config)
model
```

```
MovieLensTransformer(
  (movie_transformer): Transformer(
    (token_embedding): Embedding(3885, 32)
    (positional_embedding): Embedding(5, 32)
    (embedding_dropout): Dropout(p=0.1, inplace=False)
    (layers): ModuleList(
      (0-1): 2 x TransformerEncoderLayer(
        (layer_norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (attention): CausalMultiHeadAttention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (out): Linear(in_features=32, out_features=32, bias=True)
          (residual_dropout): Dropout(p=0.1, inplace=False)
        )
        (layer_norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (activation): GELU(approximate='tanh')
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (layer_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
  )
  (user_embedding): Embedding(6040, 32)
  (output_layer): InteractionMLP(
    (fc_layers): Sequential(
      (0): Linear(in_features=320, out_features=16, bias=True)
      (1): ReLU()
      (2): Linear(in_features=16, out_features=3885, bias=False)
    )
  )
)
```
```python
logits = model(movie_ids=movie_ids, user_ids=user_ids)
logits.shape

# torch.Size([4, 3885])
```

### Model Graph

For quick visualization, I used digraph to generate the model graph:

![Model Graph](/assets/movielens-ntp/model-graph.png)

## Training

We are able to run the model end to end! Now, let's setup the weight initialization for the model.

```python
def init_weights(model: MovieLensTransformer):
    initrange = 0.1
    model.movie_transformer.token_embedding.weight.data.uniform_(-initrange, initrange)
    model.user_embedding.weight.data.uniform_(-initrange, initrange)
    for name, p in model.output_layer.named_parameters():
        if "weight" in name:
            p.data.uniform_(-initrange, initrange)
        elif "bias" in name:
            p.data.zero_()
```

### Train step

Now that we have the model defined and weights initialization done, we should be able to define a train step. Let's follow the sage advice from karpathy and try to overfit one batch.

![Overfit one batch](./extras/karpathy_overfit_one_batch.png)

### Overfit one batch

```python
print(batch)
```

```
[tensor([[3883, 3117, 1672, 1250, 1009],
         [1672, 1250, 1009, 2271, 1768],
         [1250, 1009, 2271, 1768, 3339],
         [2271, 1768, 3339, 1189, 2735]], dtype=torch.int32),
 tensor([[0, 4, 4, 5, 5],
         [4, 5, 5, 3, 5],
         [5, 5, 3, 5, 4],
         [3, 5, 4, 4, 5]], dtype=torch.int8),
 tensor([0, 0, 0, 0], dtype=torch.int32),
 tensor([2271, 3339, 1189,  257], dtype=torch.int32),
 tensor([3, 4, 4, 4], dtype=torch.int8)]
```

We will use CrossEntropyLoss for the loss function and Adam for the optimizer. Refer to previous posts for more context on loss function.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


movie_ids, user_ids, output_movie_ids = (
    movie_ids.to(device),
    user_ids.to(device),
    output_movie_ids.to(device),
)

epochs = 10000

for epoch in range(epochs):
    optimizer.zero_grad()
    logits = model(movie_ids=movie_ids, user_ids=user_ids)
    loss = criterion(
        logits.view(-1, logits.shape[-1]), output_movie_ids.view(-1).long()
    )
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

```
Epoch 0, Loss: 8.441213607788086
Epoch 1000, Loss: 0.0002911832998506725
Epoch 2000, Loss: 7.801931496942416e-05
Epoch 3000, Loss: 3.2334930438082665e-05
Epoch 4000, Loss: 1.6748752386774868e-05
Epoch 5000, Loss: 7.867780368542299e-06
Epoch 6000, Loss: 4.321327196521452e-06
Epoch 7000, Loss: 2.2947760953684337e-06
Epoch 8000, Loss: 1.1324875686113955e-06
Epoch 9000, Loss: 7.152555099310121e-07
```

Let's now generate the next token from this overfit model to check if it matches the batch target tensor.

```python
model.eval()
logits = model(movie_ids=movie_ids, user_ids=user_ids)
print(logits)

# logits -> token probabilities
probabilities = F.softmax(logits, dim=-1)
print(torch.argmax(probabilities, dim=-1))
print(output_movie_ids)
```

```
tensor([[-10.5723, -12.5562,  -5.7962,  ...,  -7.3221, -13.8760,  -4.5341],
        [-15.7002, -18.1513,  -2.9636,  ..., -14.8108, -15.4972,  -3.4746],
        [-10.2821, -15.7702,  -6.5699,  ...,  -8.4564,  -5.2607,  -3.0306],
        [ -6.7581,  -7.9105,  -3.9927,  ...,  -9.7447,  -5.7057,  -7.2526]],
       device='cuda:0', grad_fn=<MmBackward0>)
tensor([2271, 3339, 1189,  257], device='cuda:0')
tensor([2271, 3339, 1189,  257], device='cuda:0', dtype=torch.int32)
```

They match perfectly!

Nice! So, we can see the model has fully overfit the specific batch and model training run passes the smell check.

Next, we will define the full training loop. We will also add a validation loop to test of validation loss and checkpoint model when validation loss improves.

### Train Step

Based on the test above, we can define the train step as below:

```python
def train_step(
    movie_ids: torch.Tensor,
    user_ids: torch.Tensor,
    movie_targets: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
):
    movie_ids, user_ids, movie_targets = (
        movie_ids.to(device),
        user_ids.to(device),
        movie_targets.to(device),
    )
    optimizer.zero_grad()
    output = model(movie_ids, user_ids)
    loss = criterion(output.view(-1, output.size(-1)), movie_targets.view(-1).long())
    loss.backward()
    optimizer.step()
    return loss.item()
```


### Validation step

Similarly, we can define the validation step to test the model on the validation set:

```python
def validation_step(
    movie_ids: torch.Tensor,
    user_ids: torch.Tensor,
    movie_targets: torch.Tensor,
    model: nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
):
    movie_ids, user_ids, movie_targets = (
        movie_ids.to(device),
        user_ids.to(device),
        movie_targets.to(device),
    )
    with torch.no_grad():
        output = model(movie_ids, user_ids)
        loss = criterion(
            output.view(-1, output.size(-1)), movie_targets.view(-1).long()
        )
    return loss.item()
```

### Dataset

```python
def get_dataset(config) -> Dataset:
    movies_file = os.path.join(config["trainer_config"]["data_dir"], "movies.dat")
    users_file = os.path.join(config["trainer_config"]["data_dir"], "users.dat")
    ratings_file = os.path.join(config["trainer_config"]["data_dir"], "ratings.dat")

    dataset = MovieLensSequenceDataset(
        movies_file=movies_file,
        users_file=users_file,
        ratings_file=ratings_file,
        sequence_length=config["movie_transformer_config"]["context_window_size"],
        window_size=1,  # next token prediction with sliding window of 1
    )
    return dataset
```


Using all the code above, full training loop has been defined in [`model_train.py`](https://github.com/kapilsh/recformer/blob/main/movielens-ntp/model_train.py) in the repo. Most of this code is boilerplate. So, I will not go into details here.


> NOTE: I have added tensorflow logging to the training loop. We will use it later to visualize the loss values.
{: .prompt-info}

```python

def run_model_training(config: dict):
    device = config["trainer_config"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    dataset = get_dataset(config)
    train_dataloader = DataLoader(
        dataset, batch_size=config["trainer_config"]["batch_size"], shuffle=True
    )
    validation_dataloader = DataLoader(
        dataset, batch_size=config["trainer_config"]["batch_size"], shuffle=False
    )

    model_config = get_model_config(config, dataset)

    model = MovieLensTransformer(config=model_config)

    init_weights(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["trainer_config"]["starting_learning_rate"]
    )

    best_validation_loss = np.inf

    writer = SummaryWriter(
        log_dir=config["trainer_config"]["tensorboard_dir"], flush_secs=30
    )

    for epoch in range(config["trainer_config"]["num_epochs"]):
        model.train()
        total_loss = 0.0

        pbar = trange(len(train_dataloader))
        pbar.ncols = 150
        for i, (
            movie_ids,
            rating_ids,
            user_ids,
            movie_targets,
            rating_targets,
        ) in enumerate(train_dataloader):
            loss = train_step(
                movie_ids,
                user_ids,
                movie_targets,
                model,
                optimizer,
                criterion,
                device,
            )
            total_loss += loss
            pbar.update(1)
            pbar.set_description(
                f"[Epoch = {epoch}] Current training loss (loss = {np.round(loss, 4)})"
            )
            pbar.refresh()

        pbar.close()
        train_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch}, Loss: {np.round(train_loss, 4)}")
        writer.add_scalar("loss/train", train_loss, epoch)

        model.eval()
        total_loss = 0.0

        pbar = trange(len(validation_dataloader))
        pbar.ncols = 150
        for i, (
            movie_ids,
            rating_ids,
            user_ids,
            movie_targets,
            rating_targets,
        ) in enumerate(validation_dataloader):
            loss = validation_step(
                movie_ids,
                user_ids,
                movie_targets,
                model,
                criterion,
                device,
            )
            total_loss += loss
            pbar.update(1)
            pbar.set_description(
                f"[Epoch = {epoch}] Current validation loss (loss = {np.round(loss, 4)})"
            )
            pbar.refresh()

        pbar.close()

        validation_loss = total_loss / len(validation_dataloader)
        logger.info(f"Validation Loss: {np.round(validation_loss, 4)}")
        writer.add_scalar("loss/validation", validation_loss, epoch)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            if not os.path.exists(config["trainer_config"]["model_dir"]):
                os.makedirs(config["trainer_config"]["model_dir"])
            torch.save(
                model.state_dict(),
                os.path.join(config["trainer_config"]["model_dir"], "model.pth"),
            )
```

Let's test this with a single epoch run!

```python
config_file = "./training_config.yaml"
config = load_config(config_file)
config["trainer_config"]["num_epochs"] = 1
```

```python
run_model_training(config)
```

```
2024-09-01 05:34:43.708 | INFO     | data:__init__:89 - Creating MovieLensSequenceDataset with validation set: %s
2024-09-01 05:34:43.708 | INFO     | data:read_movielens_data:12 - Reading data from files
2024-09-01 05:34:45.536 | INFO     | data:_add_tokens:140 - Adding tokens to data
2024-09-01 05:34:45.572 | INFO     | data:_generate_sequences:159 - Generating sequences
2024-09-01 05:34:46.203 | INFO     | data:__init__:110 - Train data length: 884050
2024-09-01 05:34:46.203 | INFO     | data:__init__:111 - Validation data length: 98039
2024-09-01 05:34:46.204 | INFO     | model_train:get_model_config:98 - Model config:
 ========== 
MovieLensTransformerConfig(movie_transformer_config=TransformerConfig(vocab_size=3885, context_window_size=5, embedding_dimension=32, num_layers=4, num_heads=4, dropout_embeddings=0.1, dropout_attention=0.1, dropout_residual=0.1, layer_norm_epsilon=1e-05), user_embedding_dimension=32, num_users=6040, interaction_mlp_hidden_sizes=[16]) 
 ==========
[Epoch = 0] Current training loss (loss = 7.295): 100%|██████████████████████████████████████████████████████████| 1727/1727 [00:11<00:00, 155.17it/s]
2024-09-01 05:34:57.343 | INFO     | model_train:run_model_training:164 - Epoch 0, Loss: 7.4376
[Epoch = 0] Current validation loss (loss = 7.2114): 100%|███████████████████████████████████████████████████████| 1727/1727 [00:06<00:00, 256.52it/s]
2024-09-01 05:35:04.077 | INFO     | model_train:run_model_training:197 - Validation Loss: 7.1828
```

Awesome! It is working. Let us now run a full training run for 1000 epochs. I ran the full training run and collected the loss values for training and validation using tensorboard. Let's look at these loss values:

```python
from tbparse import SummaryReader

log_dir = config["trainer_config"]["tensorboard_dir"]
reader = SummaryReader(log_dir)
metrics = reader.scalars
print(metrics.head())
metrics.groupby(["step", "tag"]).mean().unstack()["value"].plot()
```

|    |   step | tag        |   value |
|---:|-------:|:-----------|--------:|
|  0 |      0 | loss/train | 7.41048 |
|  1 |      0 | loss/train | 7.37051 |
|  2 |      0 | loss/train | 7.38262 |
|  3 |      0 | loss/train | 7.34704 |
|  4 |      0 | loss/train | 7.4013  |

### Loss Values

![Training curve](/assets/movielens-ntp/training-curve.png)

We can see that the model has learnt a lot already. Looking at the last few loss values, it seems that the model is still learning but it is good enough for now for our next steps.

## Model Predictions

Trained model is available at [`movielens-ntp/models/model_1000e_512_32_32_4_4.pth`](https://github.com/kapilsh/recformer/blob/main/movielens-ntp/models/model_1000e_512_32_32_4_4.pth). We will use this to generate next movie prediction.

### Next Token Prediction

A simple way to get the topmost movie recommended is to get the token corresponding to the maximum probability. We apply softmax to the logits to get the probability and use `argmax` to get the corresponding token.

Let's look at some sample code:

```python
trained_model_state_dict = torch.load("./models/model_1000e_512_32_32_4_4.pth")
model_config = get_model_config(config, test_dataset)
trained_model = MovieLensTransformer(model_config)
trained_model.load_state_dict(trained_model_state_dict)


def predict_next_movie(model, movie_ids, user_ids):
    model.eval()
    logits = model(movie_ids=movie_ids, user_ids=user_ids)
    probabilities = F.softmax(logits, dim=-1)
    predicted_movie_ids = torch.argmax(probabilities, dim=-1)
    print(predicted_movie_ids)
    return predicted_movie_ids
    
# test_batch from dataloader
movie_ids, rating_ids, user_ids, movie_targets, rating_targets = test_batch
predicted_movie_ids = predict_next_movie(trained_model, movie_ids, user_ids)
```

```
tensor([2381, 2847,    2,   93, 2300, 3460,  588,  583,  651, 2401, 2789, 2428,
         584, 1575, 1023,  351])
```

Let's visualize these prediction using our movie posters:

```python
merged = torch.cat([movie_ids, predicted_movie_ids.view(-1, 1)], dim=1)
merged
```

```
tensor([[1763, 2379, 2632, 1352, 1984, 2381],
        [1852, 2847, 1081,   31, 1543, 2847],
        [1050,  228,  516, 2655,  817,    2],
        [1356, 1556,  349, 2951,  372,   93],
        [1931, 1019, 2233, 1429, 2178, 2300],
        [1548,  363, 1444,  986, 2916, 3460],
        [ 188, 2872,  427, 2315, 1278,  588],
        [1854,  360, 2428,  230, 2358,  583],
        [2559, 1534, 2126, 1084, 1499,  651],
        [1085, 2401, 1942, 3379, 1932, 2401],
        [2063, 2557, 1291,  510, 2429, 2789],
        [1819, 1430,  267,  205, 1564, 2428],
        [1482, 1726, 3088, 2252, 2286,  584],
        [2269, 2038,  589, 3878, 1575, 1575],
        [ 158, 2335,  542, 2125,  453, 1023],
        [ 431, 3495, 1471, 1600, 1969,  351]])
```

### Get Poster paths

```python
rows, columns = merged.shape
movie_image_files = []

token_to_movie = {v: k for k, v in test_dataset.metadata.movie_id_tokens.items()}
for row in range(rows):
    movie_image_files.append(
        [
            f"./data/posters/{token_to_movie[image_token.item()]}.jpg"
            for image_token in merged[row]
        ]
    )
```

## Movie Recommendations

```python
for image_row in movie_image_files:
    fig, axs = plt.subplots(1, columns, figsize=(18, 5))
    for i, (ax, img) in enumerate(zip(axs.flat, image_row)):
        if os.path.exists(img):
            ax.imshow(imread(img))
        if i == columns - 1:
            ax.patch.set_edgecolor("red")
            ax.patch.set_linewidth(3)
        # ax.axis("off")  # Turn off axis
        ax.set_aspect("equal")
        ax.set_title(os.path.basename(img).split(".")[0])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
```

![Movie Recommendation 0](/assets/movielens-ntp/Movielens_RecSys_60_0.png)
![Movie Recommendation 1](/assets/movielens-ntp/Movielens_RecSys_60_1.png)
![Movie Recommendation 2](/assets/movielens-ntp/Movielens_RecSys_60_2.png)
![Movie Recommendation 3](/assets/movielens-ntp/Movielens_RecSys_60_3.png)
![Movie Recommendation 4](/assets/movielens-ntp/Movielens_RecSys_60_4.png)
![Movie Recommendation 5](/assets/movielens-ntp/Movielens_RecSys_60_5.png)
![Movie Recommendation 6](/assets/movielens-ntp/Movielens_RecSys_60_6.png)
![Movie Recommendation 7](/assets/movielens-ntp/Movielens_RecSys_60_7.png)
![Movie Recommendation 8](/assets/movielens-ntp/Movielens_RecSys_60_8.png)
![Movie Recommendation 9](/assets/movielens-ntp/Movielens_RecSys_60_9.png)
![Movie Recommendation 10](/assets/movielens-ntp/Movielens_RecSys_60_10.png)
![Movie Recommendation 11](/assets/movielens-ntp/Movielens_RecSys_60_11.png)
![Movie Recommendation 12](/assets/movielens-ntp/Movielens_RecSys_60_12.png)
![Movie Recommendation 13](/assets/movielens-ntp/Movielens_RecSys_60_13.png)
![Movie Recommendation 14](/assets/movielens-ntp/Movielens_RecSys_60_14.png)
![Movie Recommendation 15](/assets/movielens-ntp/Movielens_RecSys_60_15.png)

Based on my limited movie watching experience, I can see some good recommendations in the above list. 

We can see that model learns to recommend movies based on the user's previous movie watching history for example sci-fi movies are followed by sci-fi movies.


## Sources

- [Images](https://www.kaggle.com/datasets/ghrzarea/movielens-20m-posters-for-machine-learning)
- [BST for Recommender Systems](https://arxiv.org/pdf/1905.06874)
- [BERT4Rec](https://towardsdatascience.com/build-your-own-movie-recommender-system-using-bert4rec-92e4e34938c5)
- [Metrics](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems#ranking-quality-with-evidently)
- [Personalized Recommendations with Transformers](https://medium.com/hepsiburada-data-science/personalized-recommendations-with-transformers-11c13cff2be)
