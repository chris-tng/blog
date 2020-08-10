---

title: "Exploration of recent advancements in Semi-supervised Learning for NLP (part1)"
summary: "Exploration of recent advancements in Semi-supervised Learning for NLP (part1)"
layout: post
toc: true
comments: true
hide: false
search_exclude: false
categories: [semi-supervised,consistency-training,nlp]
---

As practitioners, I bet many would share the chronic pain of lack of data. And yet, we want to show off our shiny deep neural networks that achieves 95% accuracy. But where to get the data? Here I show you a pain reliever. It's called `semi-supervised learning`. It may not be the cure since you still need some labeled data to start off, but it's there to help. This post is my attempt to explore recent progress in semi-supervised learning with a focus in NLP.

A first-order intuition about semi-supervised learning is that it's a regularization method in disguise. You may ask why we need regularization. It's because training deep learning models on a small amount of labeled data is prone to overfitting. In such regime, it's important to have strong regularization. That's why semi-supervised learning uses unlabeled data to regularize training.

In this post, I will go over two regularization methods: `consistency training` and `mixup`.

`Consistency Training` says that a robust learning system should produce the same output given small perturbations of input. This means adding a small amount of noise and forcing the system to be noise-invariant actually helps training. Researches has shown that the best kind of noises is data augmentation.

`Mixup` says that neural network should behave linearly in-between training samples.



### Experiment on YELP dataset







- Vanilla Consistency training
- Manifold Mixup







- UDA + Pseudo-labeling

- MixMatch

  

  