import os       # os.path.exists
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
        return Value(self.data * other.data, (self, other), (
            other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (
                other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (
                math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (
                float(self.data > 0),))
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

def to_num(x) : return x.data

def to_val(x) : return Value(x)

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def forward_tok(wdic, tok_id, pos_id, keys, vals, ah = 4, hd = 4):
    tok_emb = wdic['wte'][tok_id] # token embedding
    pos_emb = wdic['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection
    # 1) Multi-head Attention block
    x_residual = x
    x = rmsnorm(x)
    q = linear(x, wdic['wqry'])
    k = linear(x, wdic['wkey'])
    v = linear(x, wdic['wval'])
    keys.append(k)
    vals.append(v)
    x_attn = []
    for h in range(ah):
        hs = h * hd
        q_h = q[hs:hs+hd]
        k_h = [ki[hs:hs+hd] for ki in keys]
        v_h = [vi[hs:hs+hd] for vi in vals]
        attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(hd)) / hd**0.5
                       for t in range(len(k_h))]
        attn_weights = softmax(attn_logits)
        head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(hd)]
        x_attn.extend(head_out)
    x = linear(x_attn, wdic['wout'])
    x = [a + b for a, b in zip(x, x_residual)]
    # 2) MLP block
    x_residual = x
    x = rmsnorm(x)
    x = linear(x, wdic['wup'])
    x = [xi.relu() for xi in x]
    x = linear(x, wdic['wdown'])
    x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, wdic['wvoc'])
    return logits

def forward_seq(wdic, seq_ids, sl = 16, ah = 4, hd = 4):
    n = min(sl, len(seq_ids))
    keys, vals = [], []
    mlogits = []

    for pos_id in range(n):
        tok_id = seq_ids[pos_id]
        logits = forward_tok(wdic, tok_id, pos_id, keys, vals, ah, hd)
        mlogits.append(logits)

    return mlogits

def cal_loss(wdic, seq_ids, sl = 16, ah = 4, hd = 4):
    n = min(sl, len(seq_ids) - 1)
    keys, vals = [], []
    mprobs = []
    losses = []

    for pos_id in range(n):
        tok_id, target_id = seq_ids[pos_id], seq_ids[pos_id + 1]
        logits = forward_tok(wdic, tok_id, pos_id, keys, vals, ah, hd)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()

        mprobs.append(probs)
        losses.append(loss_t)
    # Instead of dividing by n we divide by sl!!!
    loss = (1 / sl) * sum(losses)

    return loss, losses

# def update_kvs(wdic, tok_id, pos_id, keys, vals):
#     tok_emb = wdic['wte'][tok_id] # token embedding
#     pos_emb = wdic['wpe'][pos_id] # position embedding
#     x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
#     x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection
#     # 1) Multi-head Attention block
#     x = rmsnorm(x)
#     k = linear(x, wdic['wkey'])
#     v = linear(x, wdic['wval'])
#     keys.append(k)
#     vals.append(v)