import microgpt
import numpy as np
import string
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,"/home/jmmg1c24/Documents/Github Repos/cnn-futhark/src/purePython")
import microgptlib as mp
import random
seed = 42
random.seed(seed)
# import argparse
# import os       # os.path.exists
# import math     # math.log, math.exp

def softmax(logits):
    max_val = max(val for val in logits)
    exps = [np.exp(val - max_val) for val in logits]
    total = np.sum(exps)
    return [e / total for e in exps]

# Futhark call
mgpt = microgpt.microgpt()

alphabet = list(string.ascii_lowercase)
assert len(alphabet) == 26

BOS = len(alphabet)
vocab = alphabet + ["end"]
vocab_size = len(alphabet) + 1 # total number of unique tokens, +1 is for BOS

# Initialize the parameters, to store the knowledge of the model
ed = 16     # width of the network (embedding dimension)
sl = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
ah = 4      # number of attention heads
hd = 4 # derived dimension of each head
big_num = 700

# cau_mask = -np.ones((sl, sl))*big_num
cau_mask = -np.triu(np.ones((sl,sl)))*big_num

ran_matrix = lambda nout, nin, std=0.08: np.array([[(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)])
fwdic = {'wte': ran_matrix(vocab_size, ed), 'wpe': ran_matrix(sl, ed), 'wvoc': ran_matrix(vocab_size, ed)}
fwdic['wqry'] = ran_matrix(ed, ed)
fwdic['wkey'] = ran_matrix(ed, ed)
fwdic['wval'] = ran_matrix(ed, ed)
fwdic['wout'] = ran_matrix(ed, ed)
fwdic['wup'] = ran_matrix(4 * ed, ed)
fwdic['wdown'] = ran_matrix(ed, 4 * ed)
# params = [p for mat in fwdic.values() for row in mat for p in row] # flatten params into a single list[Value]
# print(f"num params: {len(params)}")

pwdic = { k : np.vectorize(mp.to_val)(v) for k, v in fwdic.items()}

fparams = mgpt.make_params(fwdic['wte'], fwdic['wpe'], fwdic['wqry'], fwdic['wkey'], fwdic['wval'], fwdic['wout'], fwdic['wup'], fwdic['wdown'], fwdic['wvoc'])

# input
doc = list("wakuntchapinka")

# doc = [alphabet[i] for i in random.sample(range(BOS), sl - 2)]

seq_ids = np.array([BOS] + [alphabet.index(ch) for ch in doc] + [BOS])

assert len(seq_ids) == 16

pad_mask = np.zeros((sl,sl))

mask = cau_mask + pad_mask

mflogits = mgpt.main(fparams, seq_ids, mask)
mfprobs = np.array([softmax(logits) for logits in mflogits])

mplogits = mp.forward_seq(pwdic, seq_ids)
mplogits = np.array([[val.data for val in logits] for logits in mplogits])
mpprobs = np.array([softmax(logits) for logits in mplogits])

barWidth = 0.25
lfprobs = mfprobs[-1]
lpprobs = mpprobs[-1]

# br1 = np.arange(len(lfprobs))
# br2 = [x + barWidth for x in br1]
# plt.bar(br1, lfprobs, width=barWidth, label="futhark")
# plt.bar(br2, lpprobs, width=barWidth, label="python")
# plt.xticks([r + barWidth for r in range(len(lfprobs))], vocab)
# plt.xlabel('next token probability', fontsize = 12)
# plt.legend()
# # plt.savefig('lprobs_' + "".join(doc) + "_seed" + str(seed) +  '_.png')
# plt.show()

# mse = np.array([np.mean([np.pow(mplogits[i][j] - mflogits[i][j], 2) for j in range(ed)]) for i in range(sl)])

# print(mse)

# plt.plot(mse, '-o')
# plt.xlabel('token position', fontsize = 12)
# plt.ylabel('mean square error', fontsize = 12)
# plt.savefig('mse_' + "".join(doc) + "_seed" + str(seed) +  '_.png')
# plt.show()

# print(np.sum(np.isnan(flogits)))

# print(logits)

# print(mgpt.main((1,1,1,1,1,1,1,1,1), tok_ids, 1))

# def main(): 
#     print(BOS)
#     mgpt = microgpt.microgpt()
#     doc = "Wakuntchapinka"
#     seq = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]