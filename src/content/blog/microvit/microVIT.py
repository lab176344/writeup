import gzip
import math
import os
import random
import urllib.request

# Autograd engine adapted from Andrej Karpathy's microGPT:
# https://karpathy.github.io/2026/02/12/microgpt/

n_embed = 16
n_head = 4
head_size = n_embed // n_head
n_layer = 2
patch_size = 7
image_size = 28
assert head_size * n_head == n_embed, (
    "embedding dimension must be divisible by number of heads"
)
n_patches = (image_size // patch_size) ** 2
n_classes = 10
n_epochs = 10
learning_rate = 0.05

random.seed(42)


class Value:
    __slots__ = ("value", "grad", "_children", "_local_grads")

    def __init__(self, value, children=(), local_grads=()):
        self.value = value
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.value + other.value, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.value * other.value, (self, other), (other.value, self.value))

    def __pow__(self, other):
        return Value(self.value**other, (self,), (other * self.value ** (other - 1),))

    def log(self):
        return Value(
            math.log(self.value + 1e-10), (self,), (1.0 / (self.value + 1e-10),)
        )

    def exp(self):
        return Value(math.exp(self.value), (self,), (math.exp(self.value),))

    def relu(self):
        return Value(max(0, self.value), (self,), (1.0 if self.value > 0 else 0.0,))

    def tanh(self):
        t = math.tanh(self.value)
        return Value(t, (self,), (1.0 - t**2,))

    def gelu(self):
        x = self.value
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
        t = math.tanh(inner)
        out = 0.5 * x * (1.0 + t)
        grad = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * (
            math.sqrt(2.0 / math.pi) * (1.0 + 3.0 * 0.044715 * x**2)
        )
        return Value(out, (self,), (grad,))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __lt__(self, other):
        return self.value < (other.value if isinstance(other, Value) else other)

    def __gt__(self, other):
        return self.value > (other.value if isinstance(other, Value) else other)

    def zero_grad(self):
        self.grad = 0.0

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


def download_mnist():
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"]
    for f in files:
        if not os.path.exists(f):
            opener = urllib.request.build_opener()
            opener.addheaders = [("User-agent", "Mozilla/5.0")]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(base_url + f, f)


def load_mnist(num_images=100):
    download_mnist()
    with gzip.open("train-images-idx3-ubyte.gz", "rb") as f:
        f.read(16)
        images = [[b / 255.0 for b in f.read(784)] for _ in range(num_images)]
    with gzip.open("train-labels-idx1-ubyte.gz", "rb") as f:
        f.read(8)
        labels = [b for b in f.read(num_images)]
    return images, labels


def matrix(m, n):
    return [[Value(random.gauss(0, 0.1)) for _ in range(n)] for _ in range(m)]


def linear(x, w):
    return [sum(xi * wi for xi, wi in zip(x, w_col)) for w_col in zip(*w)]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def softmax(x):
    max_x = max(xi.value for xi in x)
    exps = [(xi - max_x).exp() for xi in x]
    total = sum(exps)
    return [e / total for e in exps]


W_PATCH = matrix(patch_size * patch_size, n_embed)
CLS_TOKEN = matrix(1, n_embed)
WPE = matrix(n_patches + 1, n_embed)
CLS_HEAD = matrix(n_embed, n_classes)

state_dict = {"WP": W_PATCH, "CLS": CLS_TOKEN, "WPE": WPE, "CLS_HEAD": CLS_HEAD}
for i in range(n_layer):
    for k in ["WQ", "WK", "WV", "WO"]:
        state_dict[f"{k}{i}"] = matrix(n_embed, n_embed)
    state_dict[f"F1{i}"] = matrix(n_embed, 4 * n_embed)
    state_dict[f"F2{i}"] = matrix(4 * n_embed, n_embed)

params = [p for mat in state_dict.values() for row in mat for p in row]


def forward(pixels):
    patches = []
    for r in range(0, image_size, patch_size):
        for c in range(0, image_size, patch_size):
            patch = []
            for pr in range(patch_size):
                start_px = (r + pr) * image_size + c
                end_px = start_px + patch_size
                patch.extend(pixels[start_px:end_px])
            patches.append(linear([Value(p) for p in patch], state_dict["WP"]))

    x_seq = state_dict["CLS"] + patches
    x_seq = [
        [xi + pei for xi, pei in zip(x, pe)] for x, pe in zip(x_seq, state_dict["WPE"])
    ]

    for i in range(n_layer):
        x_norm = [rmsnorm(x) for x in x_seq]
        k_all = [linear(xt, state_dict[f"WK{i}"]) for xt in x_norm]
        v_all = [linear(xt, state_dict[f"WV{i}"]) for xt in x_norm]
        new_x_seq = []
        for t in range(len(x_seq)):
            q = linear(x_norm[t], state_dict[f"WQ{i}"])
            x_attn_heads = []
            for h in range(n_head):
                hs = h * head_size
                qh = q[hs : hs + head_size]
                scores = [
                    sum(qh[j] * k[hs + j] for j in range(head_size)) / (head_size**0.5)
                    for k in k_all
                ]
                weights = softmax(scores)
                out = [
                    sum(
                        weights[idx] * v_all[idx][hs + j] for idx in range(len(weights))
                    )
                    for j in range(head_size)
                ]
                x_attn_heads.extend(out)
            x_attn_out = linear(x_attn_heads, state_dict[f"WO{i}"])
            new_x_seq.append([a + b for a, b in zip(x_attn_out, x_seq[t])])
        x_seq = new_x_seq
        x_norm = [rmsnorm(x) for x in x_seq]
        new_x_seq = []
        for j, x in enumerate(x_norm):
            ff = linear(x, state_dict[f"F1{i}"])
            ff = [f.gelu() for f in ff]
            ff = linear(ff, state_dict[f"F2{i}"])
            new_x_seq.append([a + b for a, b in zip(ff, x_seq[j])])
        x_seq = new_x_seq

    return linear(rmsnorm(x_seq[0]), state_dict["CLS_HEAD"])


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
        if i % 20 == 0:
            pred = logits.index(max(logits))
            print(
                f"Epoch {epoch} | Step {i:3d} | Label: {labels[i]} | Pred: {pred} | Loss: {loss.value:.4f}"
            )
    print(f"Epoch {epoch} complete. Avg Loss: {epoch_loss / len(images):.4f}")
