---
title: "First Look at Gradient Checkpointing in Pytorch"
summary: "First Look at Gradient Checkpointing in Pytorch"
layout: post
toc: true
comments: true
hide: false
search_exclude: false
categories: [pytorch]

---

If you are one of those like me, a DL practitioner who couldn't afford to rent a super duper 8-GPU Titan RTX or don't have access to such compute. You might be interested in `gradient checkpoint`, a simple technique to trade computation for memory. In this post, I'll explore gradient checkpointing in Pytorch.

In brief, `gradient checkpointing` is a trick to save memory by recomputing the intermediate activations during backward. Think of it like "lazy" backward. Layer activations are not saved for backpropagation but recomputed when necessary. To use it in pytorch:

```python
import torch.utils.checkpoint as cp

# Original:
out = self.my_block(inp1, inp2, inp3)

# With checkpointing:
out = cp.checkpoint(self.my_block, inp1, inp2, inp3)
```

That looks surprisingly simple. Wondering what magic lies underneath? Let's dive in.

### Forward pass

Imagine the following forward pass: input x goes through layers one by one, results are intermediate activations h1, h2. Normally, h1 and h2 are tracked by Autograd engine for backpropagation. 

```python
x ---> [ layer1 ] ---> [ layer2 ] ---> 
                   h1              h2  
```

The trick is to detach it from the `computation graph` so they do not consume memory.

```python
with torch.no_grad():
    h2 = layer2(layer1(x))
    return h2
```

Encapsulating this into a gradient checkpointing block which produces the output but doesn't save any intermediate states

```python
x ---> [    gradient ckpt   ] ---> 
                               h2  
```

### Backward pass

```python
# NORMAL
x <--- [ layer1 ] <--- [ layer2 ] <---
   dx              dh1             dh2

# GRAD CKPT
x <--- [     gradient ckpt      ] <---
   dx                              dh2
```

During the backward pass, the gradient checkpointing block needs to return $$ \frac{dL}{dx} $$.

Since it's detached from the computation graph, it needs to recompute intermediate states to produce gradient for input x. The trick is to redo the forward pass with grad-enabled and compute the gradient of activations with respect to input x.

```python
detach_x = x.detach()
with torch.enable_grad():
    h2 = layer2(layer1(detach_x))
torch.autograd.backward(h2, dh2)
return detach_x.grad
```

### Putting it together

Using what we learnt so far, we can create our own version of gradient checkpointing.

```python
def detach_variable(inputs):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue
            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)

class CkptFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, func, *args):
        ctx.func = func
        ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = func(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        inputs = ctx.saved_tensors
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            outputs = ctx.func(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
                      for inp in detached_inputs)
        return (None, ) + grads
```

Let's see how much memory it saves. 

We'll create a 40 layers neural networks with hidden state of 1024 neurons

```python
def clones(module, N):
    return nn.Sequential(*[copy.deepcopy(module) for i in range(N)])

class SubLayer(nn.Module):

    def __init__(self, hidden_sz):
        super().__init__()
        self.lin = nn.Linear(hidden_sz, hidden_sz)
        self.out = nn.Tanh()

    def forward(self, x):
        x = self.lin(x)
        return self.out(x)

class DummyModel(nn.Module):

    def __init__(self, input_sz, hidden_sz, N, use_ckpt=False):
        super().__init__()
        self.lin1 = nn.Linear(input_sz, hidden_sz)
        self.out1 = nn.Sigmoid()
        self.layers = clones(SubLayer(hidden_sz), N)
        self.out3 = nn.Softmax(dim=-1)
        self.use_ckpt = use_ckpt

    def forward(self, x):
        x1 = self.lin1(x)
        x1 = self.out1(x1)
        if self.use_ckpt:
        	x2 = CkptFunc.apply(self.layers, x1)
        else:
            x2 = self.layers(x1)
        x3 = self.out3(x2)
        return x3
    
model = DummyModel(input_sz=64, hidden_sz=1024, use_ckpt=True, N=40)

x = torch.randn(512, 64)
y = model(x)
```

Result is encouraging: memory consumption with grad ckpt: `166.5352 (MB)` vs without `244.5352 (MB)`. This is 30% saving in memory.

That's it. Congratulations! You just learn something really cool for your toolbox.

### References

- [Pytorch code for gradient ckpt](https://github.com/pytorch/pytorch/blob/master/torch/utils/checkpoint.py)