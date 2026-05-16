---
title: "microVIT - A Vision Transformer in Pure Python"
description: "A walkthrough of microVIT, a single-file pure Python implementation of a Vision Transformer trained on MNIST using a hand-rolled autograd engine and no external dependencies."
pubDate: "2026-03-20"
tags: ["deep-learning", "machine-learning", "vision-transformer", "autograd", "transformers"]
---

This started from Karpathy's [microGPT](https://karpathy.github.io/2026/02/12/microgpt/), which distilled a GPT into 200 lines. This post applies the same idea to the vision side: a Vision Transformer (ViT) trained for image classification, no PyTorch, no NumPy, pure Python.

microVIT is one Python file, about 250 lines, that takes MNIST digits as input and classifies them. It has patch extraction, patch projection, a CLS token, positional embeddings, multi-head self-attention, feed-forward blocks, and a classification head. The autograd engine is borrowed from microGPT.

The full source is [microVIT.py](https://github.com/lab176344/writeup/blob/main/src/content/blog/microvit/microVIT.py).

## What a ViT Is

A Vision Transformer turns an image into a sequence and runs a transformer encoder over that sequence.

The image is split into fixed-size patches. Each patch is flattened and projected into an embedding vector, the same way a text token is mapped into an embedding in a language model. A learned CLS token is prepended to the patch sequence, positional embeddings are added, and the whole sequence is processed by self-attention blocks. For classification, the final CLS representation is passed to a linear head.

That is the main change from a CNN. The model does not process the image as a 2D grid of sliding convolution windows. It processes it as a sequence of patch tokens.

Before 2020, CNNs were the dominant architecture for vision tasks.

CNNs (ResNet, EfficientNet, etc.) have an inductive bias baked in: locality and translation equivariance. A convolutional kernel at position `(r, c)` only ever sees a small neighborhood. The network has to stack many layers to get global context, and global dependencies have to flow through intermediate representations.

A vanilla ViT has much weaker built-in locality and translation bias. Every patch can attend to every other patch at every layer, directly. This usually requires more data, because the model has to learn the spatial structure that CNNs get from their architecture. The benefit is that the model is not limited to local features, and at large scale this works very well.

The original ViT paper (Dosovitskiy et al., 2020) showed this gap: when pretrained at large scale, ViT-L/16 trained on ImageNet-21k outperformed ResNet-152 on transfer benchmarks. The key enabler was pretraining on a large enough corpus that the model could learn the geometric relationships that CNNs get from their architecture.

microVIT runs on 100 MNIST images. It is not meant to generalize. I use this setup because it keeps every operation small enough to read.

## Parameters

```python
n_embed = 16
n_head = 4
head_size = n_embed // n_head
n_layer = 2
patch_size = 7
image_size = 28
n_patches = (image_size // patch_size) ** 2
n_classes = 10
```

With these numbers: 16 patches per image (4×4 grid of 7×7 patches), 16-dim embeddings, 4 attention heads of size 4 each, 2 transformer layers. The state dict:

```python
W_PATCH = matrix(patch_size * patch_size, n_embed)  # 49 × 16
CLS_TOKEN = matrix(1, n_embed)                       # 1 × 16
WPE = matrix(n_patches + 1, n_embed)                 # 17 × 16
CLS_HEAD = matrix(n_embed, n_classes)                # 16 × 10

for i in range(n_layer):
    for k in ["WQ", "WK", "WV", "WO"]:
        state_dict[f"{k}{i}"] = matrix(n_embed, n_embed)   # 16 × 16
    state_dict[f"F1{i}"] = matrix(n_embed, 4 * n_embed)    # 16 × 64
    state_dict[f"F2{i}"] = matrix(4 * n_embed, n_embed)    # 64 × 16
```

Total parameters: `(49×16) + (1×16) + (17×16) + (16×10) + 2×(4×16×16 + 16×64 + 64×16)` = 784 + 16 + 272 + 160 + 2×(1024 + 1024 + 1024) = 1232 + 6144 = **7376 parameters**.

## Architecture

Pixels go in, 10 class logits come out. The pipeline: extract 16 non-overlapping 7×7 patches from the 28×28 image, project each from 49 raw pixel values down to a 16-dim embedding via `WP`, prepend a learned CLS token to make a 17-token sequence, add positional embeddings, run through 2 transformer blocks, then normalize the CLS token and project it to 10 logits via `CLS_HEAD`.

Only the CLS token is passed to the classification head; some ViT variants average-pool the patch tokens instead. Here, the CLS token is a learned vector that attends to all patches at every layer, accumulating a global summary of the image. The patch tokens are discarded after the last transformer layer. They have had two rounds of attention to contribute their information to the CLS representation.

## Step 1: Patch Extraction

```python
patches = []
for r in range(0, image_size, patch_size):
    for c in range(0, image_size, patch_size):
        patch = []
        for pr in range(patch_size):
            start_px = (r + pr) * image_size + c
            end_px = start_px + patch_size
            patch.extend(pixels[start_px:end_px])
        patches.append(linear([Value(p) for p in patch], state_dict["WP"]))
```

The image is stored as a flat list of 784 floats, row-major. For patch at `(r, c)`, row `pr` within the patch starts at pixel `(r + pr) * 28 + c`. Collecting `patch_size` pixels from that offset gives one row of the 7×7 block.

After the inner loop, `patch` is a flat 49-element vector. `linear(patch, W_PATCH)` computes `patch @ W_PATCH`, projecting the 49-dim raw pixel vector into a 16-dim embedding. The result is appended to `patches`.

Patches are extracted left-to-right, top-to-bottom: `p00` is the top-left 7×7 block, `p15` is the bottom-right.

## Step 2: CLS Token and Positional Embeddings

The CLS token is the key architectural difference between a ViT and an encoder that just average-pools its patch outputs. It's a learned 16-dim vector prepended to the patch sequence before any attention:

```python
x_seq = state_dict["CLS"] + patches
```

Because attention is global, every position attends to every other position. The CLS token accumulates information from all 16 patches over the transformer layers. By the final layer, its representation is a 16-dim summary of the image. That is what gets classified.

Average pooling over patch tokens also works in practice, but the CLS token gives the model a dedicated aggregation vector.

After prepending the CLS token, positional embeddings are added:

```python
x_seq = [
    [xi + pei for xi, pei in zip(x, pe)]
    for x, pe in zip(x_seq, state_dict["WPE"])
]
```

`WPE` is a 17×16 matrix. Row 0 is the positional embedding for the CLS slot. Rows 1–16 are for patches p00–p15. The addition is element-wise over the 16-dim vectors.

Without positional embeddings, attention is permutation-invariant. Swapping p02 with p13 would produce the same output. The positional embeddings give the model information about where each patch came from. The original ViT used learned 1D position embeddings, same as this implementation. Later variants use learned 2D, relative, sinusoidal, or RoPE-style encodings depending on the architecture.

## Step 3: Multi-Head Self-Attention

This is the transformer block. For each layer `i`:

```python
x_norm = [rmsnorm(x) for x in x_seq]
k_all = [linear(xt, state_dict[f"WK{i}"]) for xt in x_norm]
v_all = [linear(xt, state_dict[f"WV{i}"]) for xt in x_norm]

new_x_seq = []
for t in range(len(x_seq)):
    q = linear(x_norm[t], state_dict[f"WQ{i}"])
    x_attn_heads = []
    for h in range(n_head):
        hs = h * head_size       # head start index
        qh = q[hs : hs + head_size]
        scores = [
            sum(qh[j] * k[hs + j] for j in range(head_size)) / (head_size**0.5)
            for k in k_all
        ]
        weights = softmax(scores)
        out = [
            sum(weights[idx] * v_all[idx][hs + j] for idx in range(len(weights)))
            for j in range(head_size)
        ]
        x_attn_heads.extend(out)
    x_attn_out = linear(x_attn_heads, state_dict[f"WO{i}"])
    new_x_seq.append([a + b for a, b in zip(x_attn_out, x_seq[t])])
```

There are four details to notice:

**Pre-norm vs post-norm.** `rmsnorm` is applied before the attention (`x_norm = [rmsnorm(x) for x in x_seq]`), not after. This is a pre-norm block: normalize the input to the sublayer, run attention or the FFN, then add the residual connection. The `rmsnorm` function here is the minimal version of RMS normalization; it has no learned scale parameter.

**Attention is bidirectional.** Notice there is no `range(t + 1)` here. Every token attends to every other token, including tokens later in the sequence. In microGPT's causal decoder, the bound was `range(t + 1)` to enforce causality. In a ViT encoder, there is no causal structure. The CLS token at position 0 attends to patch p15 at position 16, and vice versa. This is standard ViT behavior: full bidirectional attention.

**Scaled dot-product.** `/ (head_size**0.5)` scales the dot products before softmax. Without scaling, as `head_size` grows, dot products grow in magnitude, pushing softmax into saturation where gradients vanish. The `1/sqrt(d)` factor keeps the expected variance of the dot product close to 1 regardless of head size.

**Multi-head.** The query, key, and value projections are full 16×16 matrices but the attention computation slices them into 4 heads of size 4. Each head attends over the full sequence but in a 4-dim subspace. `x_attn_heads` collects the 4-dim output of each head sequentially, resulting in a 16-dim concatenated vector that gets projected through `WO`.

The residual add at the end (`[a + b for a, b in zip(x_attn_out, x_seq[t])]`) ensures gradients flow directly from the loss to early layers without passing through the attention nonlinearities.

## Step 4: Feed-Forward Block

After attention, each position is independently processed through a two-layer MLP:

```python
x_norm = [rmsnorm(x) for x in x_seq]
new_x_seq = []
for j, x in enumerate(x_norm):
    ff = linear(x, state_dict[f"F1{i}"])    # 16 → 64
    ff = [f.gelu() for f in ff]
    ff = linear(ff, state_dict[f"F2{i}"])   # 64 → 16
    new_x_seq.append([a + b for a, b in zip(ff, x_seq[j])])
```

The expansion ratio is 4×, so the hidden dimension is `4 * n_embed = 64`. This matches the standard transformer FFN design from "Attention Is All You Need", where the expansion factor is also 4×.

GELU is used as the activation. The closed-form approximation:

```python
def gelu(self):
    x = self.value
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
    t = math.tanh(inner)
    out = 0.5 * x * (1.0 + t)
    grad = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * (
        math.sqrt(2.0 / math.pi) * (1.0 + 3.0 * 0.044715 * x**2)
    )
    return Value(out, (self,), (grad,))
```

The forward pass computes the GELU approximation directly. The local gradient for the `_local_grads` field is computed analytically with the chain rule, rather than relying on operator composition. This avoids building a deep subgraph for a single activation. Instead of chaining `tanh`, `mul`, `add`, etc., the gradient of the whole GELU is precomputed and stored as one scalar at construction time.

## Step 5: Classification Head

After the final transformer layer, only the CLS token (position 0) is passed to the classification head:

```python
return linear(rmsnorm(x_seq[0]), state_dict["CLS_HEAD"])
```

`x_seq[0]` is the CLS token after 2 rounds of attention over all 16 patch tokens. `rmsnorm` normalizes it, then `linear(_, CLS_HEAD)` projects from 16 dims to 10 logits, one per MNIST class.

This is why the CLS token exists. Attention gives it two layers of access to every patch. By the time it exits the transformer, it holds a summary of whatever the attention mechanism found useful to retain from the spatial arrangement of pixel values. The classification head then reads that summary.

## Autograd Engine

The `Value` class is a scalar node in a computation graph:

```python
class Value:
    __slots__ = ("value", "grad", "_children", "_local_grads")

    def __init__(self, value, children=(), local_grads=()):
        self.value = value
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads
```

Every operation creates a new `Value` with references to its inputs (`_children`) and their local gradients (`_local_grads`). For multiplication:

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.value * other.value, (self, other), (other.value, self.value))
```

The local grad of `a * b` with respect to `a` is `b`, and with respect to `b` is `a`. These are captured at construction time.

Backprop is a topological sort followed by a reverse pass:

```python
def backward(self):
    topo, visited = [], set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)

    build_topo(self)
    self.grad = 1.0
    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad
```

`build_topo` does a DFS from the loss node, appending nodes in post-order (children before parents). Reversing that gives topological order from root to leaves. The chain rule: `child.grad += local_grad * v.grad` accumulates the upstream gradient (from `v`) scaled by the local gradient of this edge.

The `+=` is important. A `Value` node can be a child of multiple parent nodes. For example, one activation can feed the query, key, value, and residual branches, and one layer weight is reused for every token passed through that layer. Each parent contributes independently to the gradient, so those contributions accumulate.

For a forward pass over one MNIST image at these dimensions, the graph has on the order of tens of thousands of scalar `Value` nodes. The topological sort and reverse pass walk all of them. This is why the training loop is slow. There is no vectorization, no batching, no BLAS. Every multiply is a Python function call that allocates a new object.

## Training

```python
images, labels = load_mnist(100)
for epoch in range(n_epochs):
    epoch_loss = 0.0
    for i in range(len(images)):
        logits = forward(images[i])
        loss = -softmax(logits)[labels[i]].log()
        loss.backward()
        for p in params:
            p.value -= learning_rate * p.grad
            p.zero_grad()
        epoch_loss += loss.value
```

Loss is cross-entropy: `-log(p_correct)`. `softmax(logits)` converts the 10 raw logits to a probability distribution. Indexing by the true label gives the probability assigned to the correct class. Taking the negative log turns minimization into "maximize the probability of the correct class".

The update is vanilla SGD with `lr = 0.05`. No Adam, no weight decay, no learning rate schedule. `p.zero_grad()` resets gradients after each step. Without this, gradients from previous examples accumulate into the next step's update.

After 10 epochs on 100 images:

```text
Epoch 0 | Step   0 | Label: 5 | Pred: 3 | Loss: 2.3026
Epoch 0 | Step  20 | Label: 1 | Pred: 1 | Loss: 1.8734
Epoch 0 | Step  40 | Label: 3 | Pred: 3 | Loss: 1.4102
...
Epoch 0 complete. Avg Loss: 1.8821
Epoch 1 complete. Avg Loss: 1.2543
...
Epoch 9 complete. Avg Loss: 0.1847
```

Loss drops consistently across epochs. The model is memorizing the training set, which is expected. 100 examples and 7376 parameters is not a setup that will generalize. That is not the goal here.

## How This Connects to Production ViTs

The microVIT architecture follows the same pattern as ViT-B/16, just at much smaller scale:

| Component | microVIT | ViT-B/16 |
|---|---|---|
| Patch size | 7×7 | 16×16 |
| Image size | 28×28 | 224×224 |
| Patches per image | 16 | 196 |
| Embedding dim | 16 | 768 |
| Attention heads | 4 | 12 |
| Head dim | 4 | 64 |
| Transformer layers | 2 | 12 |
| FFN hidden dim | 64 | 3072 |
| Normalization | RMSNorm-style, no learned scale | LayerNorm |
| Parameters | ~7K | ~86M |

The structure is the same. The shapes are different. ViT-B/16 has 196 image patches plus a CLS token in 768-dim space with 12 heads of size 64 across 12 layers. It uses LayerNorm where this toy implementation uses RMSNorm-style normalization. The main pieces still line up: scaled dot-product attention, pre-norm residual blocks, GELU FFN with 4× expansion, residual adds, and a CLS token for classification.

The main differences are:

**Scale and pretraining.** ViT at ImageNet scale requires either a massive dataset (JFT-300M in the original paper) or stronger training recipes such as distillation, augmentation, or DINO/MAE-style self-supervised pretraining. The inductive biases that CNNs get for free, local connectivity and weight sharing, have to be learned from data in a ViT.

**Patch embedding as convolution.** Production ViTs implement the patch projection as a strided convolution with kernel size = stride = patch size, which handles the pixel gathering and linear projection in a single GPU-efficient op. Functionally equivalent to the explicit loop and matrix multiply in microVIT.

**Absolute vs relative position.** The learned absolute positional embeddings here are the same style used in the original ViT. Many later variants explore 2D sinusoidal embeddings, relative position encodings, or RoPE-style approaches to improve transfer between different resolutions.

**Flash Attention.** ViT-B/16 has 196 patch tokens plus the CLS token, so each attention head works over a `197×197` attention matrix. That is manageable in float32. At longer sequences, or with higher resolution inputs, the quadratic memory cost becomes the bottleneck. FlashAttention rewrites the attention computation to avoid materializing the full attention matrix, using tiled SRAM operations to reduce HBM access. The math is the same; the implementation avoids the memory wall.
