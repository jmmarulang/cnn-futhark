# import microgpt
import numpy as np
# import string
import matplotlib.pyplot as plt
import time
import random
# import argparse
import futhark_server
# import logging
seed = 40
random.seed(seed)

def softmax(logits):
    max_val = max(val for val in logits)
    exps = [np.exp(val - max_val) for val in logits]
    total = np.sum(exps)
    return [e / total for e in exps]

futhark = "futhark/microgpt"
# print(futhark)

# Data
file = open('input-mgpt/input.txt')
docs = [line.strip() for line in file if line.strip()]
random.shuffle(docs)

# Tokenizer
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
vocab = uchars + ["end"]

# Initialize the parameters, to store the knowledge of the model
ed = 16     # width of the network (embedding dimension)
sl = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
ah = 4      # number of attention heads
hd = ed // ah # derived dimension of each head
big_num = 1000000000000000

dimdic = {'wte' : (vocab_size, ed), 'wpe' : (sl, ed),
          'wqry' : (ed, ed), 'wkey' : (ed, ed), 'wval' : (ed, ed),
          'wout' : (ed, ed), 'wup' : (4 * ed, ed), 'wdown' :(ed, 4 * ed),
          'wvoc': (vocab_size, ed)}

ran_matrix = lambda nout, nin, std=0.08: \
    np.array([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)])

const_matrix = lambda num, nout, nin, std=0.08: \
    np.array([[num for _ in range(nin)] for _ in range(nout)])

fwdic = {}
fmdic = {}
fvdic = {}
pmdic = {}
pvdic = {}

for k , dim in dimdic.items():
    # fwdic[k] = ran_matrix(*dim)
    fwdic[k] = const_matrix(0.5, *dim)
    fmdic[k] = np.zeros(dim)
    fvdic[k] = np.zeros(dim)
    pmdic[k] = np.zeros(dim)
    pvdic[k] = np.zeros(dim)

ones = np.ones((sl,sl))
cau_mask = (ones - np.tril(ones))

num_steps = 5

# -------------------------------------
# TRAINING FUT

print("Hold on to your morses")

# Preprocessing
masks = np.zeros((num_steps, sl, sl)).astype(np.float64)
dls = np.zeros((num_steps)).astype(np.int64)
seqs = np.zeros((num_steps, sl)).astype(np.int64, copy=False)

for step in range(num_steps):
    # doc lengths
    doc = docs[step % len(docs)]
    dl = len(doc) + 2
    dls[step] = dl
    # Masking
    pad_mask = np.ones((sl,sl))
    for i in range(dl):
        pad_mask[i][ 0 : dl] = 0
    mask = np.where(cau_mask + pad_mask >= 1, 1, 0).astype(np.float64)
    mask = -1*mask*big_num
    masks[step] = mask

with futhark_server.Server(futhark) as server:
    server.put_value('num_steps',
                     np.array(num_steps).astype(np.int64, copy=False))
    for k , data in fwdic.items():
        server.put_value(k, data)
    server.cmd_call('to_params', 'p', *fwdic.keys())
    server.cmd_call('zero_params', 'mp')
    server.cmd_call('zero_params', 'vp')
    server.put_value('masks', masks)
    server.put_value('dls', dls)

    # start timer
    start = time.time()
    # Tokenization
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        dl = len(doc) + 2
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        # Padding
        ftokens = tokens + ([BOS] * (sl - dl))
        seqs[step] = ftokens
    server.put_value('seqs', seqs)
    server.cmd_call('train', 'p_mp_vp', 'p', 'mp', 'vp', 'masks',
                    'dls', 'seqs')
    end = time.time()
    print("fgrad time", end - start)
    p_mp_vp = server.get_value('p_mp_vp')

for i , k in enumerate(dimdic.keys()):
    fwdic[k] = p_mp_vp[i]
    fmdic[k] = p_mp_vp[i + 9]
    fmdic[k] = p_mp_vp[i + 18]

try:
    np.save("fwdic.npy", fwdic, allow_pickle=True)
    file = open('fwdic.txt', 'wt')
    file.write(str(fwdic))
    file.close()
except :
    print("It refused")

# -------------------------------------
# DEF TORCH

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch.microgpt_torch_lib as mt

torch.manual_seed(seed)
device = torch.device("cpu")

model = mt.GPT()
model.to(device)
model.vocan_size = vocab_size
learning_rate = 0.01

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.85, 0.99), eps=1e-8)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=1000)

# # -------------------------------------
# # TRAINING TORCH

model.train()

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(sl, len(tokens) - 1)

    x = torch.tensor([tokens[:n]], dtype=torch.long, device= device) # change input devices
    y = torch.tensor([tokens[1:n+1]], dtype=torch.long, device= device)

    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

#-------------------------------------
# PROBS

input
# doc = list("wakuntchapinka")
doc = list("jairo")
dl = len(doc) + 2

# sequence ids
ptokens = [BOS] + [vocab.index(ch) for ch in doc] + [BOS]
# add padding
ftokens = ptokens + ([BOS] * (sl - dl))
# to numpy
ftokens = np.array(ftokens)
print("".join(doc))

pad_mask = np.ones((sl,sl))
for i in range(dl):
    pad_mask[i][ 0 : dl] = 0

# print(pad_mask)
mask = np.where(cau_mask + pad_mask >= 1, 1, 0).astype(np.float64)

mask = -1*mask*big_num

with futhark_server.Server(futhark) as server:
    server.put_value('tokens', ftokens)
    server.put_value('mask', mask)
    for k , data in fwdic.items():
        server.put_value(k, data)
    server.cmd_call('to_params', 'fparams', *fwdic.keys())
    server.cmd_call('forward_seq', 'fmlogits', 'fparams', 'tokens', 'mask')
    fmlogits = server.get_value('fmlogits')
mfprobs = np.array([softmax(logits) for logits in fmlogits])
mfprobs = mfprobs[: dl]

with torch.no_grad():
    idx = torch.tensor([ftokens], dtype=torch.long) #source of error?
    mplogits, _ = model(idx)
    mplogits = mplogits[:, -1, :]
    mpprobs = F.softmax(mplogits, dim=-1)

# # #---------

barWidth = 0.25
lfprobs = mfprobs[0]
lpprobs = mpprobs[0]

br1 = np.arange(len(lfprobs))
br2 = [x + barWidth for x in br1]
plt.bar(br1, lfprobs, width=barWidth, label="futhark")
plt.bar(br2, lpprobs, width=barWidth, label="python")
plt.xticks([r + barWidth for r in range(len(lfprobs))], vocab)
plt.xlabel('next token probability', fontsize = 12)
plt.legend()
# plt.savefig('lprobs_' + "".join(doc) + "_seed" + str(seed) +  '_.png')
plt.show()