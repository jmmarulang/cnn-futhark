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
big_num = 1000000000000

ones = np.ones((sl,sl))
cau_mask = (ones - np.tril(ones))
print(cau_mask)
# cau_mask = -1 * tri * big_num

dimdic = {'wpe' : (sl, ed), 'wqry' : (ed, ed), 'wkey' : (ed, ed),
          'wval' : (ed, ed), 'wout' : (ed, ed), 'wup' : (4 * ed, ed),
          'wdown' :(ed, 4 * ed), 'wvoc': (vocab_size, ed),
          'wte' : (vocab_size, ed)}

ran_matrix = lambda nout, nin, std=0.08: np.array(
    [[(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)])

fwdic = {}
fmdic = {}
fvdic = {}
pmdic = {}
pvdic = {}

for k , dim in dimdic.items():
    fwdic[k] = ran_matrix(*dim)
    fmdic[k] = np.zeros(dim)
    fvdic[k] = np.zeros(dim)
    pmdic[k] = np.zeros(dim)
    pvdic[k] = np.zeros(dim)
pwdic = { k : np.vectorize(mp.to_val)(v) for k, v in fwdic.items()}

fparams = mgpt.make_params(fwdic['wte'], fwdic['wpe'], fwdic['wqry'],
                           fwdic['wkey'], fwdic['wval'], fwdic['wout'],
                           fwdic['wup'], fwdic['wdown'], fwdic['wvoc'])

# input
doc = list("wakuntchapinka")
# doc = list("jairo")
asl = len(doc) + 2
# doc = [alphabet[i] for i in random.sample(range(BOS), sl - 2)]

# sequence ids
seq_ids = [BOS] + [alphabet.index(ch) for ch in doc] + [BOS]
print(seq_ids)
# add padding
seq_ids = seq_ids + ([BOS] * (sl - asl - 2))
print(seq_ids)
# to numpy
seq_ids = np.array(seq_ids)
print("".join(doc))

pad_mask = np.ones((sl,sl))
for i in range(asl):
    pad_mask[i][ 0 : asl] = 0

# print(pad_mask)
mask = np.where(cau_mask + pad_mask >= 1, 1, 0).astype(np.float64)
mask = -1*mask*big_num

#-------------------------------------
# PROBS

start = time.time()
fmlogits = mgpt.forward_seq(fparams, seq_ids, mask)
mfprobs = np.array([softmax(logits) for logits in fmlogits])
end = time.time()
print("mfprobs", end - start)

start = time.time()
pmlogits = mp.forward_seq(pwdic, seq_ids)
pmlogits = np.array([[val.data for val in logits] for logits in pmlogits])
mpprobs = np.array([softmax(logits) for logits in pmlogits])
end = time.time()
print("mpprobs", end - start)

#-------------------------------------
# LOSS

# ftarget_ids = np.array([seq_ids[pos_id + 1] for pos_id in range(asl - 1)])
# ftarget = np.array([[1 if (n < (asl - 1) and ftarget_ids[n] == m) else 0
#                      for m in range(vocab_size)]
#                      for n in range(sl)]).astype(np.float64)

# # start = time.time()
# # floss, flosses = mgpt.cal_loss(fparams, seq_ids, ftarget, mask)
# # end = time.time()
# # print("floss", end - start)
# # flosses = flosses[: asl - 1]

# # start = time.time()
# ploss, plosses = mp.cal_loss(pwdic, seq_ids)
# # end1 = time.time()
# # print("ploss", end1 - start)

# ploss.backward()
# # end = time.time()
# # print("pgrad_loss", end - start)

# ploss, plosses = ploss.data, [aloss.data for aloss in plosses]

#-------------------------------------
# GRADIENT

# pdwdic = { k : np.vectorize(mp.to_grad)(v) for k, v in pwdic.items()}

# start = time.time()
# fgrad = mgpt.grad_loss(fparams, seq_ids, ftarget, mask)
# end = time.time()
# print("fgrad_loss", end - start)

# fdwdic = {}
# for i , k in enumerate(dimdic.keys()):
#     print(k , i)
#     fdwdic[k] = fgrad[i]

#-------------------------------------
# TO FILE

try:
    np.save("fdwdic.npy", fdwdic, allow_pickle=True)
    file = open('fdwdic.txt', 'wt')
    file.write(str(fdwdic))
    file.close()
except :
    print("It refused")

try:
    np.save("pdwdic.npy", pdwdic, allow_pickle=True)
    file = open('pdwdic.txt', 'wt')
    file.write(str(pdwdic))
    file.close()
except :
    print("It refused")

fdwdic = np.load("fdwdic.npy", allow_pickle=True).item()
pdwdic = np.load("pdwdic.npy", allow_pickle=True).item()


#-------------------------------------
# UPDATE WEIGHTS

# mp.update(pwdic, pdwdic, pmdic, pvdic, 0)

# mp.update(fwdic, fdwdic, fmdic, fvdic, 0)


#-----------------------------------------------------
# PLOT

# abse_loss = [np.abs(afloss - aploss) for (afloss , aploss) in zip(flosses, plosses)]
# print(np.mean(abse_loss))
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

# br1 = np.arange(len(flosses))
# br2 = [x + barWidth for x in br1]
# plt.bar(br1, flosses, width=barWidth, label="futhark")
# plt.bar(br2, plosses, width=barWidth, label="python")
# plt.xticks([r + barWidth for r in range(len(flosses))], [i for i in range(sl - 1)])
# plt.xlabel('next token loss', fontsize = 12)
# plt.legend()
# # plt.savefig('lprobs_' + "".join(doc) + "_seed" + str(seed) +  '_.png')
# plt.show()

# barWidth = 0.25
# lfprobs = mfprobs[-1]
# lpprobs = mpprobs[-1]

# br1 = np.arange(len(lfprobs))
# br2 = [x + barWidth for x in br1]
# plt.bar(br1, lfprobs, width=barWidth, label="futhark")
# plt.bar(br2, lpprobs, width=barWidth, label="python")
# plt.xticks([r + barWidth for r in range(len(lfprobs))], vocab)
# plt.xlabel('next token probability', fontsize = 12)
# plt.legend()
# # plt.savefig('lprobs_' + "".join(doc) + "_seed" + str(seed) +  '_.png')
# plt.show()

mse = np.array([np.mean([np.pow(pmlogits[i][j] - fmlogits[i][j], 2) for j in range(ed)]) for i in range(sl)])
plt.plot(mse, '-o')
plt.xlabel('token position', fontsize = 12)
plt.ylabel('mean square error', fontsize = 12)
# plt.savefig('mse_' + "".join(doc) + "_seed" + str(seed) +  '_.png')
plt.show()

# key = "wte"
# index = [0]
# fdata = fdwdic[key][index][0]
# pdata = pdwdic[key][index][0]
# print(fdata)
# barWidth = 0.25

# abse = [np.abs(afloss - aploss) for (afloss , aploss) in zip(fdata, pdata)]
# print(np.mean(abse))
# plt.plot(abse, '-o')
# plt.xlabel('weight', fontsize = 12)
# plt.ylabel('abs error', fontsize = 12)
# # plt.savefig('mse_' + "".join(doc) + "_seed" + str(seed) +  '_.png')
# plt.show()

# key = "wte"
# index = [0]
# fdata = fdwdic[key][index][0]
# pdata = pdwdic[key][index][0]
# barWidth = 0.25

# br1 = np.arange(len(fdata))
# br2 = [x + barWidth for x in br1]
# plt.bar(br1, fdata, width=barWidth, label="futhark")
# plt.bar(br2, pdata, width=barWidth, label="python")
# plt.xticks([r + barWidth for r in range(len(fdata))], range(sl))
# plt.xlabel('position', fontsize = 12)
# plt.ylabel('weight', fontsize = 12)
# plt.legend()
# # plt.savefig('lprobs_' + "".join(doc) + "_seed" + str(seed) +  '_.png')
# plt.show()