---
title: "Understanding linear attention in Transformers are RNNs"
summary: "Understanding linear attention in Transformers are RNNs"
layout: post
toc: true
comments: true
hide: false
search_exclude: false
categories: [transformer,nlp,self-attention,rnn]
---

Recently, I came across an interesting paper [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) (Update: it was accepted in ICML 2020). In the theme of efficient transformer, this paper claims to reduce the complexity of `self-attention` from quadratic to linear in terms of sequence length. If that doesn't sound exciting enough:

> Our linear transformers achieve similar performance to vanilla transformers and they are up to 4000x faster on autoregressive prediction of very long sequences

That sounds awesome and shiny. Let's see what magic lies underneath.

### Self-attention

A short revisit to `self-attention`: given 3 matrices Q, K, V corresponding to queries, keys and values. Scaled-dot product attention is computed as 

$$ V' = softmax (\frac{QK^T}{\sqrt{d}}) V $$

Assume $Q$ of shape $(N, d)$, $K$ and $V$ are both of shape $(N, d)$, where $N$ is the sequence length, $d$ is model size. The attention computation becomes a bottleneck since $QK^T$ is of shape $(N, N)$.

Denote $v'_i$ is the i-th row of $V'$. Decomposing the above formula, we see that it's computed as a linear combination of rows of $V$ :

$$ v'_i = \frac{\sum_j \alpha_j v_j}{\sum_j \alpha_j} $$

$$ \alpha_i = sim (q_i, k_j) = \exp ( \frac{q_i . k_j^T}{\sqrt{d}}) $$ 

Here is where it gets interesting. Note that instead of using `exponential` as the similarity function, we can use any kernel. Reminder

> A kernel is a function that corresponds to a dot product in very high dimensional space.

Using this property of kernel:  given a kernel K, we have $K(q_i, k_j) = \phi(q_i)^T \phi(k_j)$ where $\phi$ is the mapping to "high" dimensional space, called `feature map`. Our attention can be written as:

$$ v'_i = \frac{\sum_j sim(q_i, k_j) v_j}{\sum_j sim(q_i, k_j)} $$

$$ = \frac{\sum_j \phi(q_i)^T \phi(k_j) v_j}{\sum_j \phi(q_i)^T \phi(k_j)} $$

$$ = \frac{\phi(q_i)^T \sum_j \phi(k_j) v_j}{\phi(q_i)^T \sum_j \phi(k_j)} $$

In short

$$ v'_j = \frac{\phi(q_i)^T (\sum_j \phi(k_j)^T v_j )}{ \phi(q_i)^T \sum_j \phi(k_j)^T} $$

Basically the trick is that by kernelizing, we could compute the $KV$ product first, which results in a $(d, d)$ matrix, much cheaper than the $QK^T$ product of shape $(N, N)$. In many practical cases, we want our transformer to process long sequences, so $N$ >> $d$  hence the saveup.

In terms of computation, the scaled-dot product attention would require $\mathcal{O}(d N^2)$ operations while the kernelized attention requires $$ \mathcal{O}(Nd^2) $$ operations. Assume we have sequences of length 4000 with model of size 1000. Standard attention would need $16e+9$ operations, kernelized one needs $4e+9$ operations, so theoretically 4 times speedup.

### Experiment

Let's call our new attention - kernelized attention. For our experiment, we'll use `elu + 1` as our feature map

```python
def elu(x, alpha = 0.1):
    mask = (x > 0.).float()
    return mask*x + (1-mask)*alpha* (x.exp() - 1)

def phi(x, alpha = 0.1):
    return elu(x, alpha) + 1
```

Our standard scaled-dot product attention

```python
import math

def query_key_value_attn(q, k, v):
    "q, k and v are batch-major of shape (batch_sz, seq_len, feature_sz)"
    d_model = q.size(-1)
    sim = q @ k.transpose(-2, -1) / math.sqrt(d_model)
    return sim.softmax(-1) @ v
```

Kernelized attention

```python
def kernel_attn(q, k, v, phi):
    assert k.size(1) == v.size(1), f"Key and Value MUST have the same seq len"
    # project onto feature space
    k_proj_T = phi(k.transpose(-2, -1))
    q_proj = phi(q)
    s = k_proj_T @ v
    m = k_proj_T.sum(-1, keepdim=True)
    return (q_proj @ s) / (q_proj @ m)
```

On a `V100 GPU`, with `seq_len = 4000`, `d_model = 1024` I see a speedup around 2.5 times (2.24ms vs 5.57ms), which is not bad.

In terms of memory, the difference is significant, when `d_model` is small compared to `seq_len`. I use `d_model = 64` so the whole ting can fit into 16GB of GPU memory

| Seq Len | Kernelized | Scaled-dot |
|-|-|-|
| 4096    |    12 MB   |   132 MB   |
| 4096*6  |    72 MB   |   4096 MB  |



### Notes

We can see kernelized attention effectively scales linearly with respect to sequence length.

It's all nice and exciting at this point. This could be our poor man's `self-attention` from now on. But keep it mind that applying masking to kernelized attention is not trivial, that could result in huge performance loss if the computation is not vectorized. Personally, I have tried to implement vectorized arbitrary masking but unsuccessful so far. Nevertheless, it's a cool trick in our toolbox.

