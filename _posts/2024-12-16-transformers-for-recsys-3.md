---
title: Transformers for Recommender Systems - Part 3
description: >-
  Regularization to improve duplication penalty loss 
date: 2024-12-16
categories: [Blog, Tutorial]
tags: [AI, Machine Learning, RecSys, Transformers]
pin: true
math: false
author: ks
---

This is a small update and a continuation of the previous posts [Transformers for Recommender Systems - Part 1](../transformers-for-recsys-1) and [Transformers for Recommender Systems - Part 2](../transformers-for-recsys-2). 

In this post, we improved the model further by improving duplication penalty loss via regularization. This was a pretty small change such that we use median instead of mean to calculate the penalty loss. 

> Median essentially helps in reducing the impact of outliers via L1 regularization.

### Updated Loss Function

![Regularized Penalty Loss](/assets/recformer_regularized_penalty_loss.png)

This improved the train loss from 5.33 -> 5.31 and validation loss from 5.26 -> 5.24.

All the code is available on [Github](https://github.com/kapilsh/recformer/commit/1489cf6929f6f12ae349c56ca010cfb9008efc0c).
