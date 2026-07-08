import microgpt
import numpy as np
import string
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,"/home/jmmg1c24/Documents/Github Repos/cnn-futhark/src/purePython")
import microgptlib as mp
import time
import random
seed = 40
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

ones = np.ones((sl,sl))
tri = (ones - np.tril(ones))
cau_mask = -1 * tri * big_num

ran_matrix = lambda nout, nin, std=0.08: np.array(
    [[(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)])
fwdic = {'wte': ran_matrix(vocab_size, ed), 'wpe': ran_matrix(sl, ed),
         'wvoc': ran_matrix(vocab_size, ed)}
fwdic['wqry'] = ran_matrix(ed, ed)
fwdic['wkey'] = ran_matrix(ed, ed)
fwdic['wval'] = ran_matrix(ed, ed)
fwdic['wout'] = ran_matrix(ed, ed)
fwdic['wup'] = ran_matrix(4 * ed, ed)
fwdic['wdown'] = ran_matrix(ed, 4 * ed)
# pparams = [p for mat in fwdic.values() for row in mat for p in row] # flatten params into a single list[Value]
# print(f"num params: {len(params)}")

pwdic = { k : np.vectorize(mp.to_val)(v) for k, v in fwdic.items()}

fparams = mgpt.make_params(fwdic['wte'], fwdic['wpe'], fwdic['wqry'],
                           fwdic['wkey'], fwdic['wval'], fwdic['wout'],
                           fwdic['wup'], fwdic['wdown'], fwdic['wvoc'])

# input
doc = list("wakuntchapinka")

# doc = [alphabet[i] for i in random.sample(range(BOS), sl - 2)]

seq_ids = np.array([BOS] + [alphabet.index(ch) for ch in doc] + [BOS])
asl = len(seq_ids)
print("".join(doc))

assert asl == 16

pad_mask = np.zeros((sl,sl))

mask = cau_mask + pad_mask

# start = time.time()
# fmlogits = mgpt.forward_seq(fparams, seq_ids, mask)
# mfprobs = np.array([softmax(logits) for logits in fmlogits])
# end = time.time()
# print("mfprobs", end - start)

# start = time.time()
# pmlogits = mp.forward_seq(pwdic, seq_ids)
# pmlogits = np.array([[val.data for val in logits] for logits in pmlogits])
# mpprobs = np.array([softmax(logits) for logits in pmlogits])
# end = time.time()
# print("mpprobs", end - start)

# mse = np.array([np.mean([np.pow(pmlogits[i][j] - fmlogits[i][j], 2) for j in range(ed)]) for i in range(sl)])
# pos = 0
# print(big_num, pos, mse[pos])

ftarget_ids = np.array([seq_ids[pos_id + 1] for pos_id in range(asl - 1)])
ftarget = np.array([[1 if (n < (asl - 1) and ftarget_ids[n] == m) else 0
                     for m in range(vocab_size)]
                     for n in range(sl)]).astype(np.float64)

# start = time.time()
# floss, flosses = mgpt.cal_loss(fparams, seq_ids, ftarget, mask)
# end = time.time()
# print("floss", end - start)
# flosses = flosses[: asl - 1]

start = time.time()
ploss, plosses = mp.cal_loss(pwdic, seq_ids)
end1 = time.time()
print("ploss", end1 - start)

ploss.backward()
end = time.time()
print("pgrad_loss", end - start)

ploss, plosses = ploss.data, [aloss.data for aloss in plosses]

pdwdic = { k : np.vectorize(mp.to_grad)(v) for k, v in pwdic.items()}
# print(pdwdic['wvoc'])

# start = time.time()
# fgrad = mgpt.grad_loss(fparams, seq_ids, ftarget, mask)
# end = time.time()
# print("fgrad_loss", end - start)

# fdwdic = {}
# fdwdic['wpe'] = fgrad[0]
# fdwdic['wqry'] = fgrad[1]
# fdwdic['wkey'] = fgrad[2]
# fdwdic['wval'] = fgrad[3]
# fdwdic['wout'] = fgrad[4]
# fdwdic['wup'] = fgrad[5]
# fdwdic['wdown'] = fgrad[6]
# fdwdic['wvoc'] = fgrad[7]
# fdwdic['wseq'] = fgrad[8]

try:
    np.save("fdwdic.npy", fdwdic, allow_pickle=True)
    np.save("pdwdic.npy", pdwdic, allow_pickle=True)
    file = open('fdwdic.txt', 'wt')
    file.write(str(fdwdic))
    file.close()
    file = open('pdwdic.txt', 'wt')
    file.write(str(pdwdic))
    file.close()
except :
    print("It refused")

fdwdic = np.load("fdwdic.npy", allow_pickle=True).item()
pdwdic = np.load("pdwdic.npy", allow_pickle=True).item()

fdwdic_flat = { k : v.flatten() for k, v in fdwdic.items()}
pdwdic_flat = { k : v.flatten() for k, v in pdwdic.items()}


#-----------------------------------------------------

# abse_loss = [np.abs(afloss - aploss) for (afloss , aploss) in zip(flosses, plosses)]
# plt.plot(abse_loss, '-o')
# plt.xlabel('token position', fontsize = 12)
# plt.ylabel('abs error', fontsize = 12)
# # plt.savefig('mse_' + "".join(doc) + "_seed" + str(seed) +  '_.png')
# plt.show()


# plt.plot(plosses, '-o')
# plt.plot(flosses, '-o')
# plt.xlabel('token position', fontsize = 12)
# plt.ylabel('loss', fontsize = 12)
# # plt.savefig('mse_' + "".join(doc) + "_seed" + str(seed) +  '_.png')
# plt.show()


# barWidth = 0.25
# lfprobs = mfprobs[0]
# lpprobs = mpprobs[0]

# br1 = np.arange(len(lfprobs))
# br2 = [x + barWidth for x in br1]
# plt.bar(br1, lfprobs, width=barWidth, label="futhark")
# plt.bar(br2, lpprobs, width=barWidth, label="python")
# plt.xticks([r + barWidth for r in range(len(lfprobs))], vocab)
# plt.xlabel('next token probability', fontsize = 12)
# plt.legend()
# # plt.savefig('lprobs_' + "".join(doc) + "_seed" + str(seed) +  '_.png')
# plt.show()

# plt.plot(mse, '-o')
# plt.xlabel('token position', fontsize = 12)
# plt.ylabel('mean square error', fontsize = 12)
# # plt.savefig('mse_' + "".join(doc) + "_seed" + str(seed) +  '_.png')
# plt.show()

key = "wpe"
fdata = fdwdic[key]
pdata = pdwdic[key]
mse = np.array([np.mean([np.pow(fdata[i][j] - pdata[i][j], 2) for j in range(fdata.shape[1])]) for i in range(fdata.shape[0])])
plt.plot(mse, '-o')
plt.xlabel(key + ' weight position', fontsize = 12)
plt.ylabel('mse', fontsize = 12)
# plt.savefig('mse_' + "".join(doc) + "_seed" + str(seed) +  '_.png')
plt.show()