"""
A PyTorch equivalent of microgpt.py: train and run inference for a character-level GPT.
"""

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import numpy as np
import matplotlib.pyplot as plt

seed = 1
torch.manual_seed(seed)
random.seed(seed)

# Dataset
futhark = "futhark/microgpt"
# print(futhark)

# Data
file = open('input-mgpt/input.txt')
docs = [line.strip() for line in file if line.strip()]
# random.shuffle(docs)

# Tokenizer
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
vocab = uchars + ["end"]

# Hyperparameters
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head
learning_rate = 0.01

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(n_embd, head_dim, bias=False)
        self.query = nn.Linear(n_embd, head_dim, bias=False)
        self.value = nn.Linear(n_embd, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_dim)
        q = self.query(x) # (B, T, head_dim)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, head_dim)
        out = wei @ v # (B, T, T) @ (B, T, head_dim) -> (B, T, head_dim)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(n_head)])
        self.proj = nn.Linear(head_dim * n_head, n_embd, bias=False)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa_heads = MultiHeadAttention()
        self.ffwd = FeedForward()

    def forward(self, x):
        x = x + self.sa_heads(F.rms_norm(x, (n_embd,)))
        x = x + self.ffwd(F.rms_norm(x, (n_embd,)))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Better init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # torch.nn.init.normal_(module.weight, mean=0.0, std=0.08)
            torch.nn.init.constant_(module.weight, 0.5)
        elif isinstance(module, nn.Embedding):
            # torch.nn.init.normal_(module.weight, mean=0.0, std=0.08)
            torch.nn.init.constant_(module.weight, 0.5)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        x = F.rms_norm(x, (n_embd,))
        x = self.blocks(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

model = GPT()
print(f"num params: {sum(p.numel() for p in model.parameters())}")

num_steps = 10_000

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.85, 0.99), eps=1e-8)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_steps)

model.train()
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    x = torch.tensor([tokens[:n]], dtype=torch.long)
    y = torch.tensor([tokens[1:n+1]], dtype=torch.long)

    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}", end='\r')


doc = list("marulanda")
dl = len(doc) + 2

# sequence ids
python_tokens = [BOS] + [vocab.index(ch) for ch in doc] + [BOS]


# def softmax(logits):
#     max_val = max(val for val in logits)
#     exps = [np.exp(val - max_val) for val in logits]
#     total = np.sum(exps)
#     return [e / total for e in exps]

# Torch
torch_tokens = torch.tensor([python_tokens], dtype=torch.long)
model.eval()
with torch.no_grad():
    torch_logits, _ = model(torch_tokens)
torch_logits = torch_logits.numpy()[0]
# torch_probs = np.array([softmax(logits) for logits in torch_logits])


# Probs
while True:
    torch_index = int(input("torch index <- "))
    if (torch_index == -1):
            break
    barWidth = 0.25
    torch_data = torch_logits[torch_index]

    br1 = np.arange(len(torch_data))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    plt.bar(br1, torch_data, width=barWidth, label="torch")
    plt.xticks([r + barWidth for r in range(len(torch_data))], vocab)
    plt.xlabel('next token', fontsize = 12)
    plt.legend()
    plt.savefig('torch_' + "".join(doc) + "_seed_" + str(seed) + "_index_" + str(torch_index) + "_iter_" + str(num_steps) + '_.png')
    plt.show()


# torch_probs = np.array([softmax(logits) for logits in torch_logits])

# print("\n--- inference (new, hallucinated names) ---")
# model.eval()
# temperature = 0.5
# with torch.no_grad():
#     for sample_idx in range(20):
#         idx = torch.tensor([[BOS]], dtype=torch.long)
#         sample = []
#         for _ in range(block_size):
#             logits, _ = model(idx)
#             logits = logits[:, -1, :] / temperature
#             probs = F.softmax(logits, dim=-1)
#             idx_next = torch.multinomial(probs, num_samples=1)
#             if idx_next.item() == BOS:
#                 break
#             idx = torch.cat((idx, idx_next), dim=1)
#             sample.append(uchars[idx_next.item()])
#         print(f"sample {sample_idx+1:2d}: {''.join(sample)}")