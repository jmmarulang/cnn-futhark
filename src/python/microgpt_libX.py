"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import math     # math.log, math.exp

# Let there be Autograd to recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

def to_grad(x) : return x.grad

def to_data(x) : return x.data

def to_val(x) : return Value(x)

def reset_grad(x) : x.grad = 0

class GPT():
    def __init__(self, docs, state_dict):
        self.docs = docs
        self.uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
        self.BOS = len(self.uchars) # token id for a special Beginning of Sequence (BOS) token
        self.vocab_size = len(self.uchars) + 1 # total number of unique tokens, +1 is for BOS
        self.vocab = self.uchars + ["end"]

        self.n_layer = 1
        self.n_embd = 16     # width of the network (embedding dimension)
        self.block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
        self.n_head = 4      # number of attention heads
        self.head_dim = self.n_embd // self.n_head # derived dimension of each head
        self.state_dict = state_dict
        self.params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]

        # Let there be Adam, the blessed optimizer and its buffers
        self.learning_rate, self.beta1, self.beta2, self.eps_adam = 0.01, 0.85, 0.99, 1e-8
        self.m = [0.0] * len(self.params) # first moment buffer
        self.v = [0.0] * len(self.params) # second moment buffer

    def linear(self, x, w):
        return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

    def softmax(self, logits):
        max_val = max(val.data for val in logits)
        exps = [(val - max_val).exp() for val in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def rmsnorm(self, x):
        ms = sum(xi * xi for xi in x) / len(x)
        scale = (ms + 1e-5) ** -0.5
        return [xi * scale for xi in x]

    def to_data(self, x) : return x.data

    def forward(self, token_id, pos_id, keys, values):
        tok_emb = self.state_dict['wte'][token_id] # token embedding
        pos_emb = self.state_dict['wpe'][pos_id] # position embedding
        x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
        x = self.rmsnorm(x) # note: not redundant due to backward pass via the residual connection

        for li in range(self.n_layer):
            assert self.n_layer == 1
            # 1) Multi-head Attention block
            x_residual = x
            x = self.rmsnorm(x)
            q = self.linear(x, self.state_dict['wqry'])
            k = self.linear(x, self.state_dict['wkey'])
            v = self.linear(x, self.state_dict['wval'])
            keys[li].append(k)
            values[li].append(v)
            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs:hs+self.head_dim]
                k_h = [ki[hs:hs+self.head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs+self.head_dim] for vi in values[li]]
                attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(self.head_dim)) / self.head_dim**0.5 for t in range(len(k_h))]
                attn_weights = self.softmax(attn_logits)
                head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(self.head_dim)]
                x_attn.extend(head_out)
            x = self.linear(x_attn, self.state_dict['wout'])
            x = [a + b for a, b in zip(x, x_residual)]
            # 2) MLP block
            x_residual = x
            x = self.rmsnorm(x)
            x = self.linear(x, self.state_dict['wup'])
            x = [xi.relu() for xi in x]
            x = self.linear(x, self.state_dict['wdown'])
            x = [a + b for a, b in zip(x, x_residual)]

        logits = self.linear(x, self.state_dict['wvoc'])
        return logits

    def forward_seq(self, seq_ids):
        n = min(self.block_size, len(seq_ids))
        keys, vals = [[]], [[]]
        seq_logits = []

        for pos_id in range(n):
            tok_id = seq_ids[pos_id]
            logits = self.forward(tok_id, pos_id, keys, vals)
            seq_logits.append(logits)

        return seq_logits

    def train(self, num_steps):
        seq_losses = []
        # Repeat in sequence
        for step in range(num_steps):

            # Take single document, tokenize it, surround it with BOS special token on both sides
            doc = self.docs[step % len(self.docs)]
            tokens = [self.BOS] + [self.uchars.index(ch) for ch in doc] + [self.BOS]
            n = min(self.block_size, len(tokens) - 1)

            # Forward the token sequence through the model, building up the computation graph all the way to the loss
            keys, values = [[] for _ in range(self.n_layer)], [[] for _ in range(self.n_layer)]
            losses = []
            for pos_id in range(n):
                token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
                logits = self.forward(token_id, pos_id, keys, values)
                probs = self.softmax(logits)
                loss_t = -probs[target_id].log()
                losses.append(loss_t)
            loss = (1 / self.block_size) * sum(losses) # final average loss over the document sequence. May yours be low.

            # Backward the loss, calculating the gradients with respect to all model parameters
            loss.backward()

            # Adam optimizer update: update the model parameters based on the corresponding gradients
            lr_t = self.learning_rate * (1 - step / num_steps) # linear learning rate decay
            for i, p in enumerate(self.params):
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
                m_hat = self.m[i] / (1 - self.beta1 ** (step + 1))
                v_hat = self.v[i] / (1 - self.beta2 ** (step + 1))
                p.data -= lr_t * m_hat / (v_hat ** 0.5 + self.eps_adam)
                p.grad = 0

            print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

            seq_losses.append(loss)

        return seq_losses
