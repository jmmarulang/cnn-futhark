# import microgpt
import numpy as np
import string
import matplotlib.pyplot as plt
import sys
# sys.path.insert(0,"/home/jmmg1c24/Documents/Github Repos/cnn-futhark/src/purePython")
# import microgptlib as mp
import time
import random
import argparse
import futhark_server
import logging
seed = 40
random.seed(seed)

def softmax(logits):
    max_val = max(val for val in logits)
    exps = [np.exp(val - max_val) for val in logits]
    total = np.sum(exps)
    return [e / total for e in exps]

def update(wdic, dwdic, mdic, vdic, step, num_steps, learning_rate = 0.01, beta1 = 0.85, beta2 = 0.99, eps_adam = 1e-8):
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for k , dp in dwdic.items():
        mdic[k] = beta1 * mdic[k] + (1 - beta1) * dp
        vdic[k] = beta2 * vdic[k] + (1 - beta2) * dp ** 2
        m_hat = mdic[k] / (1 - beta1 ** (step + 1))
        v_hat = vdic[k] / (1 - beta2 ** (step + 1))
        wdic[k] -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)

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
          'wqry' : (ed, ed), 'wkey' : (ed, ed), 'wval' : (ed, ed), 'wout' : (ed, ed),
          'wup' : (4 * ed, ed), 'wdown' :(ed, 4 * ed), 'wvoc': (vocab_size, ed),
          }

ran_matrix = lambda nout, nin, std=0.08: np.array([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)])

fwdic = {}
fmdic = {}
fvdic = {}

for k , dim in dimdic.items():
    # pwdic[k] = ran_matrix(*dim)
    fwdic[k] = ran_matrix(*dim)
    fmdic[k] = np.zeros(dim)
    fvdic[k] = np.zeros(dim)

ones = np.ones((sl,sl))
cau_mask = (ones - np.tril(ones))

#-------------------------------------
# TRAINING FUT

num_steps = 500

mask_l = []
ftarget_l = []

for step in range(num_steps):
    doc = list(docs[step % len(docs)])
    asl = len(doc) + 2
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]

    pad_mask = np.ones((sl,sl))
    for i in range(asl):
        pad_mask[i][ 0 : asl] = 0
    mask = np.where(cau_mask + pad_mask >= 1, 1, 0).astype(np.float64)
    mask = -1*mask*big_num
    mask_l.append(mask)

    ftarget_id = np.array([tokens[pos_id + 1] for pos_id in range(asl - 1)])
    ftarget = np.array([[1 if (n < (asl - 1) and ftarget_id[n] == m) else 0
                        for m in range(vocab_size)]
                        for n in range(sl)]).astype(np.float64)
    ftarget_l.append(ftarget)

fdwdic = {}
with futhark_server.Server(futhark) as server:
    start = time.time()
    for step in range(num_steps):
        doc = list(docs[step % len(docs)])
        asl = len(doc) + 2
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]

        # Padding
        ftokens = tokens + ([BOS] * (sl - asl))

        ftokens = np.array(ftokens)

        server.put_value('tokens', ftokens)
        server.put_value('mask', mask_l[step])
        server.put_value('ftarget', ftarget_l[step])
        for k , data in fwdic.items():
            server.put_value(k, data)
        server.cmd_call('make_params', 'fparams', *fwdic.keys())
        server.cmd_call('grad_loss', 'grad',
                        'fparams', 'tokens',  'ftarget' ,'mask')
        fgrad = server.get_value('grad')

        fdwdic = {}
        for i , k in enumerate(dimdic.keys()):
            fdwdic[k] = fgrad[i]

        update(fwdic, fdwdic, fmdic, fvdic, step, num_steps)

        server.cmd_free('tokens', 'mask', 'ftarget', 'fparams','grad')
        server.cmd_free(*fwdic.keys())

end = time.time()
print("fgrad time", end - start)

try:
    np.save("fwdic.npy", fwdic, allow_pickle=True)
    file = open('fwdic.txt', 'wt')
    file.write(str(fwdic))
    file.close()
except :
    print("It refused")