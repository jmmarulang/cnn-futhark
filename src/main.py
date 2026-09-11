import numpy as np
import matplotlib.pyplot as plt
import time
import random
import futhark_server
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch.microgpt_torch_lib as mt


seed = 40
random.seed(seed)

precision = np.float32

def softmax(logits):
    max_val = max(val for val in logits)
    exps = [np.exp(val - max_val) for val in logits]
    total = np.sum(exps)
    return [e / total for e in exps]

futhark = "futhark/microgpt"

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
bs = 1
num_steps = 30000
num_batches = 30000
assert num_batches == num_steps / bs
sl = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
ed = 16     # width of the network (embedding dimension)
ah = 4      # number of attention heads
hd = ed // ah # derived dimension of each head
big_num = 1000000000000000

dimdic = {'wte' : (vocab_size, ed), 'wpe' : (sl, ed),
          'wqry' : (ed, ed), 'wkey' : (ed, ed), 'wval' : (ed, ed),
          'wout' : (ed, ed), 'wup' : (4 * ed, ed), 'wdown' :(ed, 4 * ed),
          'wvoc': (vocab_size, ed)}

# Build matrix of randon numbers
ran_matrix = lambda nout, nin, std=0.08: \
    np.array([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)]).astype(precision)

# Build constant matrix
const_matrix = lambda num, nout, nin, std=0.08: \
    np.array([[num for _ in range(nin)] for _ in range(nout)]).astype(precision)

fwdic = {}
fmdic = {}
fvdic = {}
pmdic = {}
pvdic = {}

# Initial weights
for k , dim in dimdic.items():
    # fwdic[k] = ran_matrix(*dim)
    fwdic[k] = const_matrix(0.5, *dim) # constant so we can compare with pytorch
    fmdic[k] = np.zeros(dim)
    fvdic[k] = np.zeros(dim)
    pmdic[k] = np.zeros(dim)
    pvdic[k] = np.zeros(dim)

ones = np.ones((sl,sl))
cau_mask = (ones - np.tril(ones)).astype(precision)

# -------------------------------------
# DEF TORCH

torch.manual_seed(seed)
device_type = "cpu"
device = torch.device(device_type)

model = mt.GPT()
model.to(device)
model.vocan_size = vocab_size
learning_rate = 0.01

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.85, 0.99), eps=1e-8)

# -------------------------------------
# TRAINING DATA
masks = np.zeros((num_batches, bs, sl, sl)).astype(precision)
# DANGER : Not sure if sl is correct
dls = np.ones((num_batches)).astype(np.int64)*sl
batches = np.zeros((num_batches, bs, sl)).astype(np.int64)

step = 0
for i in range(num_batches):
    for j in range(bs):
        doc = docs[step % len(docs)]; step += 1
        doc = doc[:sl - 2]
        seq = [BOS] + [vocab.index(ch) for ch in doc] + [BOS]
        seq = seq + ([BOS] * (sl - len(seq)))

        batches[i][j] = seq

        mask = np.where(cau_mask >= 1, 1, 0)
        masks[i][j] = -1*mask*big_num

batches = np.array(batches).astype(np.int64)
masks = np.array(masks).astype(precision)

# -------------------------------------
# TRAINING FUT

print("Hold on to your morses")

with futhark_server.Server(futhark) as server:
    for k , data in fwdic.items():
        server.put_value(k, data)
    server.cmd_call('to_params', 'p', *fwdic.keys())
    server.cmd_call('zero_params', 'mp')
    server.cmd_call('zero_params', 'vp')
    server.put_value('masks', masks)
    server.put_value('dls', dls)

    # train
    # start timer
    start = time.perf_counter ()
    server.put_value('seqs', batches)
    server.cmd_call('train', 'p_mp_vp', 'p', 'mp', 'vp', 'masks',
                    'dls', 'seqs')
    end = time.perf_counter ()
    print("futhark grad time", end - start)
    p_mp_vp = server.get_value('p_mp_vp')

# save weights
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

# # # -------------------------------------
# # # # TRAINING TORCH

# model.train()
# # start timer
# start = time.perf_counter ()
# for i in range(num_batches):
#     batch = batches[i]
#     n = sl - 1

#     x = torch.tensor(batch[: , :n], dtype=torch.long, device= device)
#     y = torch.tensor(batch[: , 1:n+1], dtype=torch.long, device= device)

#     with torch.autocast(device_type= device_type, dtype=torch.bfloat16):
#         logits, loss = model(x, y)

#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

#     print(f"batch {i+1:4d} / {num_batches:4d} | loss {loss.item():.4f}", end='\r')

# end = time.perf_counter ()
# print("torch grad time", end - start)

#-------------------------------------
# PROBS

# input
# dl = int(input("sl >> ")) + 2
dl = 5 + 2
docs = ["jairo"]*bs
tseqs = []
fseqs = []
masks = []
for i in range(bs):
    # doc = input(str(i) + " doc >> ")
    doc = docs[i]
    seq = [BOS] + [vocab.index(ch) for ch in doc] + [BOS]
    assert (len(seq) == dl)
    tseqs.append(seq)
    seq = seq + ([BOS] * (sl - dl))
    assert (len(seq) == sl)
    fseqs.append(seq)

    pad_mask = np.ones((sl,sl))
    for i in range(dl):
        pad_mask[i][ 0 : dl] = 0
    mask = np.where(cau_mask + pad_mask >= 1, 1, 0)
    mask = -1*mask*big_num
    masks.append(mask)

fseqs = np.array(fseqs).astype(np.int64)
masks = np.array(masks).astype(precision)

with futhark_server.Server(futhark) as server:
    server.put_value('seqs', fseqs)
    server.put_value('masks', masks)
    for k , data in fwdic.items():
        server.put_value(k, data.astype(precision))
    server.cmd_call('to_params', 'params', *fwdic.keys())
    server.cmd_call('forward', 'logits', 'params', 'seqs', 'masks')
    f_batch_seq_logits = server.get_value('logits')
    server.put_value('dl', np.array(dl))
    server.cmd_call('loss', 'floss', 'dl', 'params', 'seqs', 'masks')
    floss = server.get_value('floss')

# DANGER : dl or dl - 1 ?
f_batch_seq_logits = \
    np.array([seq_logits[: dl - 1] for seq_logits in f_batch_seq_logits])

t_batch_loss = []
with torch.no_grad():
    x = []
    y = []
    n = min(sl, dl - 1)
    for i in range(bs):
        tokens = tseqs[i]
        x.append(tokens[:n])
        y.append(tokens[1:n+1])
    x = np.array(x)
    y = np.array(y)
    x = torch.tensor(x, dtype=torch.long, device= device)
    y = torch.tensor(y, dtype=torch.long, device= device)
    seq_logits, seq_loss = model(x, y)
    seq_logits = seq_logits.numpy()
    # DANGER : May not be reshaped in the proper order
    t_batch_seq_logits = seq_logits.reshape(bs, dl - 1, vocab_size)
    t_batch_loss.append(seq_loss.numpy())
tloss = np.mean(t_batch_loss)

print("futhark final loss" , floss)
print("torch   final loss" , tloss)

fprobs = [[softmax(logits) for logits in seq_logits]
          for seq_logits in f_batch_seq_logits]

tprobs = [[softmax(logits) for logits in seq_logits]
          for seq_logits in t_batch_seq_logits]

# # ---------

barWidth = 0.25
lfprobs = fprobs[0][-1]
lpprobs = tprobs[0][-1]

br1 = np.arange(len(lfprobs))
br2 = [x + barWidth for x in br1]
plt.bar(br1, lfprobs, width=barWidth, label="futhark")
plt.bar(br2, lpprobs, width=barWidth, label="pytorch")
plt.xticks([r + barWidth for r in range(len(lfprobs))], vocab)
plt.xlabel('next token probability', fontsize = 12)
plt.legend()
# plt.savefig('lprobs_' + "".join(doc) + "_seed" + str(seed) +  '_.png')
plt.show()