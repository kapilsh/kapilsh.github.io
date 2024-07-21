---
title: Logits to Text
description: >-
  How to generate text from a transformer model
date: 2024-07-19
categories: [Blog, Tutorial]
tags: [AI, Transformers, Pytorch, Machine Learning, LLM]
pin: false
math: true
author: ks
---

Generating text from logits for a Transformer model involves converting the output logits into human-readable text. In this post, we will cover the simplest way to generate text i.e. Greedy Approach but I briefly cover other techniques for the text generation. 

If you are looking for more, Huggingface has a [great blog post](https://huggingface.co/blog/how-to-generate) on text generation using different techniques. 

Okay let's go. Here's a step-by-step outline of the process:

1. **Obtain the Logits:** These are the raw scores output by the Transformer model before applying any probability function. Typically, the shape of the logits is `[batch_size, sequence_length, vocab_size]`.

2. **Apply a Softmax Function:** Convert logits to probabilities. Softmax is typically used to convert the logits into probabilities for each token in the vocabulary.

```python
import torch.nn.functional as F
probabilities = F.softmax(logits, dim=-1)
```

3. **Sample or Argmax:** Convert probabilities to actual token indices. There are different strategies to do this:

  - **Greedy Decoding (Argmax):** Choose the token with the highest probability.

```python
predicted_indices = torch.argmax(probabilities, dim=-1)
```

  - **Sampling:** Sample from the probability distribution.

```python
predicted_indices = torch.multinomial(probabilities, num_samples=1).squeeze()
```
4. **Convert Token Indices to Text:** Use a tokenizer to convert the token indices back to text.

```python
# Assuming you have a tokenizer that can convert indices back to tokens
generated_text = tokenizer.decode(predicted_indices, skip_special_tokens=True)
```
5. Here's a complete example assuming you have the `logits` from a Transformer model and a tokenizer:

```python
import torch
import torch.nn.functional as F

# Example logits tensor
logits = torch.randn(1, 10, 50257)  # [batch_size, sequence_length, vocab_size]

# Apply softmax to get probabilities
probabilities = F.softmax(logits, dim=-1)

# Greedy decoding
predicted_indices = torch.argmax(probabilities, dim=-1)

# Assuming you have a tokenizer that can decode the indices to text
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Convert indices to text
generated_text = tokenizer.decode(predicted_indices[0], skip_special_tokens=True)

print(generated_text)
```

I [previously wrote a post](../exploring-gpt2/) on creating GPT2 model from scratch. We can use that model to generate some text. 

```python
gpt2_model = GPT2.from_pretrained()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "Hello, I'm a language model,"
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)

with torch.no_grad():
    my_gpt_logits = gpt2_model(encoded_input['input_ids'])
    print(my_gpt_logits)

probabilities = F.softmax(my_gpt_logits, dim=-1)
predicted_indices = torch.argmax(probabilities, dim=-1)
print("Argmax sampling: ", predicted_indices)
generated_text = tokenizer.decode(predicted_indices[0], skip_special_tokens=True)
print("Generated text: \'", generated_text, "\'")
```

```
{'input_ids': tensor([[15496,    11,   314,  1101,   257,  3303,  2746,    11]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]])}
Argmax sampling:  tensor([[  11,  314, 1101,  407, 1310, 5887,   13,  290]])
Generated text: ' , I'm not little expert. and '
```

That looks coherent enough!

## Notes on other Strategies:
- **Temperature Scaling:** You might want to scale the logits before applying softmax to control the randomness of the predictions. Lower temperature makes the model more confident (less random), while a higher temperature makes it less confident (more random).
```python
temperature = 1.0
scaled_logits = logits / temperature
probabilities = F.softmax(scaled_logits, dim=-1)
```

- **Top-k and Top-p Sampling:** These are advanced sampling methods that can help to produce more coherent and diverse text by limiting the sampling pool to the top-k or top-p tokens.

- **Beam Search:** This is another decoding method that keeps track of multiple hypotheses (beams) during the generation process and selects the most likely sequence overall.

## Sources
- [Huggingface Blog](https://huggingface.co/blog/how-to-generate)
