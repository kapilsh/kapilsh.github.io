---
title: Transformers from Scratch
description: >-
  Another article in the wild on writing transformers from scratch
date: 2024-07-02
categories: [Blog, Tutorial]
tags: [AI, Transformers, Pytorch]
pin: true
author: ks
---

So, there are plenty of "Transformers from Scratch" articles and videos out there in the wild and I would be lying if I said I did not look at any of them before writing my own. So why this?

My motivation to share was purely just for demonstration as to how I wrote my own small module for transformers "from scratch".

To start with, let's import the libraries and modules we will use.

```python
import torch
from torch import nn
import math
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
```

# Attention is all you need

Below is one of the most snapshoted diagram in all of Machine Learning. Without this, any new blog post is incomplete. So, here it is: 

![transformers](/assets/c82eda57-1ae2-4540-b29d-5d7190535a30.png){: width="500" }

# Parameters and descriptions

Now comes the important stuff. What are the main parameters that the Transformers use? I describe some of them below.

- **Vocab Size**: All the tokens that can be part of the input sequence. Sequences are tokenized based on the vocab
- **Sequence Length (S)**: Length of the sentence or sequence fed into the tranformer
- **Embedding Size (E)**: Size of the embedding for each node in the sequence above

S X E dimension for one sequence (S = d_model in the paper)

B X S X E for a batch size of B

From paper:

> To facilitate these residual connections, all sub-layers in the model, as well as the embedding
layers, produce outputs of dimension dmodel = 512.

- **Number of Heads**: Number of heads in the attention layer - output from embedding layer is split into these heads
- **Head dimension**: Can be calculated from embedding dimension and number of heads: `embedding_size // num_heads` (same as d_k in the paper)

From paper:

> We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of
queries and keys of dimension d_k, and values of dimension d_v

### Scaled Dot Product Attention

![Screenshot 2024-06-06 at 10.50.31 AM.png](/assets/42949b60-8cb3-4cc9-ba38-e64af0dfe566.png)

- **Dropout**: Dropout rate after attention layer
- **Feed Forward Hidden size**: Hidden dimension for feed forward layer (effectively an MLP)

# Encoder

### Encoder Overview

Best resource I found that illustrates the internals of Transformer architecture is [Jay Alammar's Blog Post - The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/). Below is the encoder internals from the post.

![Screenshot 2024-06-06 at 5.27.39 PM.png](/assets/015a8f26-68c3-4c34-8f18-1f75353fc899.png)

Below is the pseudo-code for how Encoder layer processes inputs and passes them from one layer to another 

- Input -> Sparse Embedding
- X = Sparse Embedding + Position Embedding
- Calculate A_out = MultiHeadAttn(X)
- A_out = Dropout(A_out)
- L_out = LayerNorm(A_out + X)
- F_out = FeedForwardNN(L_out)
- F_out = Dropout(F_out)
- out = LayerNorm(F_out + L_out)

# Exploration

For exploration, I will use 1984 book text. I have used this one of my previous posts as well to generate new chapters using LSTM (haha I know). For tokenizing, I will use the GPT2 tokenizer.

```python

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Read the novel from the text file
with open('./data/1984.txt', 'r', encoding='utf-8') as file:
    novel_text = file.read()

encoded_input = tokenizer(novel_text, return_tensors='pt')
encoded_input

# output
# {'input_ids': tensor([[14126,   352,   628,  ..., 10970, 23578,   198]]), 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]])}
```

Let's set sequence length to be something reasonable for testing and exploration:

```python
sequence_length = 10
```

For generating the inputs to the models, I use a window length of 1 to slide the sequence along to generate several sequences from our input text.

```python
model_inputs = encoded_input["input_ids"].ravel().unfold(0, sequence_length, 1).to(torch.int32)
model_inputs

# output
# tensor([[14126,   352,   628,  ...,   257,  6016,  4692],
#        [  352,   628,   198,  ...,  6016,  4692,  1110],
#        [  628,   198,   198,  ...,  4692,  1110,   287],
#        ...,
#        [ 2739,   257,  3128,  ...,   198,   198, 10970],
#        [  257,  3128,   355,  ...,   198, 10970, 23578],
#        [ 3128,   355, 32215,  ..., 10970, 23578,   198]], dtype=torch.int32)

```

```python
batch = model_inputs[:4, :]
batch

# output
# tensor([[14126,   352,   628,   198,   198,  1026,   373,   257,  6016,  4692],
#        [  352,   628,   198,   198,  1026,   373,   257,  6016,  4692,  1110],
#        [  628,   198,   198,  1026,   373,   257,  6016,  4692,  1110,   287],
#        [  198,   198,  1026,   373,   257,  6016,  4692,  1110,   287,  3035]],
#       dtype=torch.int32)
```



# Position Encoding

Quoting from the original paper, positional encoding are just functions over the sequence and embedding dimensions. 

> NOTE: GPT2 onwards, positional encoding is usually added as learnable embedding instead. 

![Screenshot 2024-06-19 at 6.22.53 AM.png](/assets/35a50179-5b16-4056-a785-47c506c2c28b.png)

Let's generate this step by step. First, we need to generate the indices.

```python
embedding_dimension = 6 # d_model
even_index = torch.arange(0, embedding_dimension, 2)
odd_index = torch.arange(1, embedding_dimension, 2)
print(even_index)
# tensor([0, 2, 4])

print(odd_index)
# tensor([1, 3, 5])
```

From here, we should be able to generate 

```python
denominator = torch.pow(10000, even_index / embedding_dimension)
print(denominator) # for odd, denominator is the same
# tensor([  1.0000,  21.5443, 464.1590])
```

Now, we can generate the indices along the sequence dimension.

```python
positions = torch.arange(0, sequence_length, 1).reshape(sequence_length, 1)
print(positions)
# tensor([[0],
#         [1],
#         [2],
#         [3],
#         [4],
#         [5],
#         [6],
#         [7],
#         [8],
#         [9]])
```

We should have everything now to calculate the position embeddings.

```python
even_pe = torch.sin(positions / denominator)
odd_pe = torch.cos(positions / denominator)
stacked = torch.stack([even_pe, odd_pe], dim=2)
position_embeddings = torch.flatten(stacked, start_dim=1, end_dim=2)

print(position_embeddings)
#tensor([[ 0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  1.0000],
#        [ 0.8415,  0.5403,  0.0464,  0.9989,  0.0022,  1.0000],
#        [ 0.9093, -0.4161,  0.0927,  0.9957,  0.0043,  1.0000],
#        [ 0.1411, -0.9900,  0.1388,  0.9903,  0.0065,  1.0000],
#        [-0.7568, -0.6536,  0.1846,  0.9828,  0.0086,  1.0000],
#        [-0.9589,  0.2837,  0.2300,  0.9732,  0.0108,  0.9999],
#        [-0.2794,  0.9602,  0.2749,  0.9615,  0.0129,  0.9999],
#        [ 0.6570,  0.7539,  0.3192,  0.9477,  0.0151,  0.9999],
#        [ 0.9894, -0.1455,  0.3629,  0.9318,  0.0172,  0.9999],
#        [ 0.4121, -0.9111,  0.4057,  0.9140,  0.0194,  0.9998]])
```


# Input Layer

We now define the input layer i.e. the layer where sequences are converted to embeddings and then combined (added to) with position embeddings.

```python
class InputLayer(nn.Module):
    def __init__(self, 
                 num_embeddings: int, 
                 sequence_length: int, 
                 embedding_dimension: int):
        super().__init__()
        self.sequence_length = sequence_length
        self.embedding_dimension = embedding_dimension
        self.input_embeddings = nn.Embedding(num_embeddings=num_embeddings, 
                                             embedding_dim=embedding_dimension)
        self.position_embeddings = self._init_position_embedding(sequence_length, embedding_dimension)
        print(sequence_length, embedding_dimension)

    @staticmethod
    def _init_position_embedding(sl: int, ed: int) -> torch.Tensor:
        even_index = torch.arange(0, embedding_dimension, 2)
        odd_index = torch.arange(1, embedding_dimension, 2)
        denominator = torch.pow(10000, even_index / embedding_dimension)
        even_pe = torch.sin(positions / denominator)
        odd_pe = torch.cos(positions / denominator)
        stacked = torch.stack([even_pe, odd_pe], dim=2)
        return torch.flatten(stacked, start_dim=1, end_dim=2)
        
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        emb = self.input_embeddings(input)
        return emb + self.position_embeddings
input_layer = InputLayer(num_embeddings=len(tokenizer.get_vocab()), sequence_length=sequence_length, embedding_dimension=embedding_dimension)
```

We had generated a batch previously. Now we can pass the batch through the input layer to output embeddings that feed into the attention layers. 

```python
attention_input = input_layer(batch)
print(attention_input.size())
# torch.Size([4, 10, 6])
```

# Multi Head Attention

```python
num_heads = 2
head_dimension = embedding_dimension // num_heads
if embedding_dimension % num_heads != 0:
    raise ValueError("embedding_dimension should be divisible by num_heads")
```


```python
# split up input into Q, K, V
qkv_layer = nn.Linear(embedding_dimension, 3 * embedding_dimension)
with torch.no_grad():
    qkv_layer.weight.fill_(2)

qkv = qkv_layer(attention_input)
```


```python
qkv.size()
```




    torch.Size([4, 10, 18])




```python
input_shape = attention_input.shape
input_shape = input_shape[:-1] + (num_heads, 3 * head_dimension)
input_shape
```




    torch.Size([4, 10, 2, 9])




```python
qkv = qkv.reshape(*input_shape)
qkv = qkv.permute(0, 2, 1, 3)
```


```python
qkv.size() # batch_size, num_head, sequence_length, 3 * head_dim
```




    torch.Size([4, 2, 10, 9])




```python
q, k, v = qkv.chunk(3, dim=-1)
```


```python
for masked in [True, False]:
    if masked:
        mask = torch.full((sequence_length, sequence_length),
                          -torch.inf)
        mask = torch.triu(mask, diagonal=1)
    else:
        mask = torch.zeros((sequence_length, sequence_length))
    print(mask)
```

    tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
            [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
            [0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
            [0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
            [0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf],
            [0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf],
            [0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf],
            [0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., -inf],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])


## Scaled Dot Product Attention


```python
# here we take the query and key head blocks and do the scaled multiply

scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dimension)

# here we add the masks for backward looking bias (for encoder the masks are all 0 since we use both forward and backward context)
scores += mask

# softmax(QK/sqrt(d_model))
attention = torch.softmax(scores, dim=-1)
```


```python
attention.size()
```




    torch.Size([4, 2, 10, 10])




```python
# last part in the attention calculation where we multiply with values
values = torch.matmul(attention, v)
```


```python
values.size()
```




    torch.Size([4, 2, 10, 3])




```python
values = values.transpose(1, 2).contiguous()
print("After transposing the head and sequence dims", values.size())
values = values.view(*values.shape[:2], -1)
print("After concating all the head dims", values.size())
```

    After transposing the head and sequence dims torch.Size([4, 10, 2, 3])
    After concating all the head dims torch.Size([4, 10, 6])



```python
# final linear layer in the attention

attn_linear_layer = nn.Linear(embedding_dimension, embedding_dimension)
attn_out = attn_linear_layer(values_out)
print("output from attention has size", attn_out.size())
attn_out
```

    output from attention has size torch.Size([4, 10, 6])





    tensor([[[-3.9014, -2.5143,  0.5328,  2.1713, 11.6392, -4.5111],
             [-3.9261, -2.5294,  0.5359,  2.1826, 11.7051, -4.5360],
             [-3.9235, -2.5278,  0.5356,  2.1814, 11.6980, -4.5333],
             [-2.9408, -1.9877,  0.5475,  1.6745,  8.8760, -3.4817],
             [ 1.7852,  1.0139, -0.2578, -0.4185, -3.4584,  1.2042],
             [-3.9260, -2.5294,  0.5359,  2.1826, 11.7050, -4.5360],
             [ 1.8151,  1.0338, -0.2649, -0.4310, -3.5336,  1.2330],
             [-3.9244, -2.5284,  0.5357,  2.1818, 11.7005, -4.5343],
             [-3.9097, -2.5193,  0.5338,  2.1751, 11.6612, -4.5194],
             [-3.9217, -2.5267,  0.5354,  2.1806, 11.6933, -4.5315]],
    
            [[-3.5661, -2.3060,  0.4857,  2.0187, 10.7494, -4.1742],
             [-3.5661, -2.3060,  0.4857,  2.0187, 10.7494, -4.1742],
             [-3.5613, -2.3031,  0.4852,  2.0165, 10.7365, -4.1694],
             [-2.7212, -1.8198,  0.4495,  1.6014,  8.3896, -3.2900],
             [-3.5661, -2.3060,  0.4857,  2.0187, 10.7494, -4.1742],
             [ 2.9829,  1.7587, -0.4278, -0.9626, -6.6335,  2.4065],
             [-3.5660, -2.3060,  0.4857,  2.0187, 10.7494, -4.1742],
             [-3.5659, -2.3059,  0.4857,  2.0186, 10.7490, -4.1740],
             [-3.5661, -2.3060,  0.4857,  2.0187, 10.7494, -4.1742],
             [-3.5660, -2.3060,  0.4857,  2.0187, 10.7494, -4.1742]],
    
            [[-3.5589, -2.3016,  0.4847,  2.0155, 10.7305, -4.1671],
             [-3.5504, -2.2963,  0.4836,  2.0115, 10.7078, -4.1585],
             [-3.5410, -2.2905,  0.4824,  2.0072, 10.6826, -4.1490],
             [-3.5623, -2.3036,  0.4852,  2.0170, 10.7394, -4.1704],
             [ 3.6319,  2.1614, -0.5183, -1.2581, -8.3559,  3.0586],
             [-3.5402, -2.2900,  0.4823,  2.0069, 10.6806, -4.1482],
             [-3.5421, -2.2911,  0.4825,  2.0077, 10.6855, -4.1500],
             [-3.5621, -2.3035,  0.4852,  2.0169, 10.7388, -4.1702],
             [-3.5605, -2.3026,  0.4850,  2.0162, 10.7347, -4.1687],
             [-3.5519, -2.2973,  0.4838,  2.0122, 10.7119, -4.1600]],
    
            [[-5.0136, -3.2044,  0.6877,  2.6777, 14.5916, -5.6288],
             [-5.0149, -3.2052,  0.6878,  2.6783, 14.5950, -5.6301],
             [-5.0154, -3.2055,  0.6879,  2.6785, 14.5963, -5.6306],
             [ 3.1944,  1.8899, -0.4573, -1.0589, -7.1948,  2.6190],
             [-4.9563, -3.1697,  0.6814,  2.6509, 14.4371, -5.5705],
             [-3.7136, -2.4995,  0.7240,  1.9989, 10.8286, -4.2287],
             [-5.0154, -3.2055,  0.6879,  2.6785, 14.5963, -5.6306],
             [-5.0154, -3.2055,  0.6879,  2.6785, 14.5963, -5.6306],
             [-5.0154, -3.2055,  0.6879,  2.6785, 14.5963, -5.6306],
             [-5.0154, -3.2055,  0.6879,  2.6785, 14.5963, -5.6306]]],
           grad_fn=<ViewBackward0>)




```python
attn_out.size()
```




    torch.Size([4, 10, 6])



# Layer Normalization


```python
# there is already a layer norm implementation in pytorch

layer_norm = nn.LayerNorm(embedding_dimension)
layer_norm_out = layer_norm(attn_out + attention_input) # attention_input is the residual connection
layer_norm_out.size()
```




    torch.Size([4, 10, 6])




```python
layer_norm_out[:, :, 0]
```




    tensor([[-0.9511, -0.7062, -0.7170, -0.5669,  1.0509, -0.6046,  0.6238, -0.7322,
             -0.8901, -0.8219],
            [-0.8372, -0.7635, -0.5182, -0.5417, -0.5061,  0.6271, -0.8802, -0.9871,
             -0.7650, -0.6345],
            [-0.9223, -0.5743, -0.5162, -0.3550,  0.7457, -0.9374, -1.1227, -0.8897,
             -0.5797, -1.0181],
            [-0.7742, -0.6618, -0.4229,  0.8975, -0.8579, -1.1751, -1.0069, -0.7410,
             -0.9509, -0.9719]], grad_fn=<SelectBackward0>)




```python
# let;s calculate it from scratch
attn_out_mean = attn_out.mean(-1, keepdim=True)
attn_out_var = torch.pow(attn_out - attn_out_mean, 2).mean(-1, keepdims=True)
attn_out_std = torch.sqrt(attn_out_var + layer_norm.eps)
ln_out = (attn_out - attn_out_mean) / attn_out_std
ln_out[:, :, 0]
```




    tensor([[-0.8153, -0.8156, -0.8156, -0.8085,  1.0473, -0.8156,  1.0442, -0.8156,
             -0.8154, -0.8156],
            [-0.8104, -0.8104, -0.8104, -0.7993, -0.8104,  0.9674, -0.8104, -0.8104,
             -0.8104, -0.8104],
            [-0.8103, -0.8102, -0.8101, -0.8104,  0.9485, -0.8101, -0.8101, -0.8104,
             -0.8104, -0.8102],
            [-0.8271, -0.8271, -0.8271,  0.9603, -0.8267, -0.8244, -0.8271, -0.8271,
             -0.8271, -0.8271]], grad_fn=<SelectBackward0>)



# Feed Forward Network

![Screenshot 2024-06-19 at 6.13.30 AM.png](/assets/5972f9cb-b802-47a6-8b50-af30a2425a16.png)

> The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality
df f = 2048.


```python
class FeedForwardNetwork(nn.Module):
    def __init__(self, 
                 embedding_dimension: int, 
                 hidden_dimension: int, 
                 drop_prop: float):
        super().__init__()
        self.linear1 = nn.Linear(in_features=embedding_dimension, out_features=hidden_dimension)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout()
        self.linear2 = nn.Linear(in_features=hidden_dimension, out_features=embedding_dimension)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
```


```python
ffn_hidden_dimension = 12
dropout_prob = 0.1
ffn = FeedForwardNetwork(embedding_dimension=embedding_dimension, hidden_dimension=ffn_hidden_dimension, drop_prop=dropout_prob)
```


```python
ffn_out = ffn(layer_norm_out)
ffn_out.size()
```




    torch.Size([4, 10, 6])



At this point, ffn output also goes through a layer normalization


```python
ffn_layer_norm = nn.LayerNorm(embedding_dimension)
ffn_layer_norm_out = ffn_layer_norm(ffn_out + layer_norm_out) # layer_norm_out is the residual connection
ffn_layer_norm_out.size()
```




    torch.Size([4, 10, 6])




```python
ffn_layer_norm_out
```




    tensor([[[-1.0527, -0.0532, -1.0837,  0.7020,  1.7722, -0.2845],
             [-0.8055,  0.0955, -1.1206,  0.5460,  1.8595, -0.5748],
             [-0.5474, -0.9311, -0.3291,  0.6749,  1.9132, -0.7805],
             [-0.7260, -0.5811, -0.5319,  0.6078,  1.9803, -0.7491],
             [ 1.3078, -0.3787, -0.0302,  0.2983, -1.8908,  0.6937],
             [-0.6731, -0.1974, -1.1219,  0.3222,  2.0078, -0.3376],
             [ 1.0256,  0.1397, -0.3442,  0.0828, -1.9343,  1.0305],
             [-0.9266, -0.5781, -0.6862,  0.5453,  1.9815, -0.3359],
             [-1.0120, -0.3394, -0.9822,  0.7689,  1.8023, -0.2378],
             [-0.6715, -1.1744, -0.0743,  0.6736,  1.8344, -0.5877]],
    
            [[-0.5824, -0.4211, -0.4398,  0.4363,  2.0189, -1.0118],
             [-0.7856, -0.0313, -1.2074,  0.8620,  1.7004, -0.5381],
             [-0.3629, -0.9726,  0.0042,  0.4350,  1.9198, -1.0235],
             [-0.6208, -0.6086, -0.5741,  0.7985,  1.8758, -0.8708],
             [-0.6826, -0.5313, -0.8994,  0.2657,  2.0782, -0.2306],
             [ 0.8581,  0.1424, -0.0482, -0.1513, -1.9653,  1.1643],
             [-0.7377, -1.0379, -0.0577,  0.4414,  1.9631, -0.5712],
             [-1.0824, -0.1500, -1.0562,  0.7706,  1.7450, -0.2270],
             [-0.5958, -1.1300, -0.1412,  0.6829,  1.8531, -0.6691],
             [-0.3126, -1.0439,  0.1087,  0.7184,  1.7042, -1.1748]],
    
            [[-0.6357, -0.5490, -0.5320,  0.8365,  1.8409, -0.9606],
             [-0.4226, -0.8586,  0.0503,  0.4322,  1.9120, -1.1132],
             [-0.5779, -0.3562, -0.7749,  0.7005,  1.9077, -0.8992],
             [-0.5519, -0.6373, -0.8964,  0.2708,  2.0833, -0.2685],
             [ 1.0269,  0.0101, -0.2513,  0.0264, -1.9146,  1.1025],
             [-0.7957, -1.0977, -0.0192,  0.4780,  1.9223, -0.4876],
             [-1.2022, -0.0847, -1.0130,  0.7763,  1.6993, -0.1757],
             [-1.0415, -0.3640, -0.8110,  0.7806,  1.8326, -0.3966],
             [-0.7404, -0.4590, -0.5343,  0.6912,  1.9238, -0.8813],
             [-0.8868, -1.0669,  0.1662,  0.6033,  1.8133, -0.6290]],
    
            [[-0.6372, -0.7121, -0.0500,  0.4062,  1.9842, -0.9911],
             [-0.4743, -0.7758, -0.0496,  0.3689,  1.9822, -1.0515],
             [-0.4491, -0.3044, -1.1540,  0.3238,  2.0196, -0.4359],
             [ 1.1559, -0.0035, -0.4572,  0.2203, -1.8697,  0.9541],
             [-0.7136, -1.1454,  0.0262,  0.4643,  1.9143, -0.5457],
             [-1.1798, -0.0699, -1.0778,  0.9135,  1.6035, -0.1895],
             [-0.7400, -0.7336, -0.2556,  0.7215,  1.8860, -0.8783],
             [-0.6360, -0.8736,  0.1273,  0.5372,  1.8666, -1.0215],
             [-1.1143, -0.2475, -0.7123,  0.5862,  1.9142, -0.4264],
             [-1.0495, -0.0475, -0.9295,  0.8658,  1.7225, -0.5618]]],
           grad_fn=<NativeLayerNormBackward0>)



# Decoder

![Screenshot 2024-06-19 at 6.51.29 AM.png](/assets/3784c45f-74a7-438c-9b3e-e941e7029ae7.png)

- Input -> Sparse Embedding
- X = Sparse Embedding + Position Embedding (Masked)
- Calculate A_out = MultiHeadAttn(X) - This is self attention
- A_out = Dropout(A_out)
- L_out = LayerNorm(A_out + X)
- C_out = MultiHeadCrossAttn(L_out, encoder_out) - this is the cross attention
- C_out = Dropout(A_out)
- LC_out = LayerNorm(C_out + L_out)
- F_out = FeedForwardNN(LC_out)
- F_out = Dropout(F_out)
- out = LayerNorm(F_out + LC_out)

