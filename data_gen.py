import math
from stat import filemode
import pandas as pd
import numpy as np
import os
import tqdm
import glob
import re

p_f = re.compile('\d+\.\d+')
p_d = re.compile('\d+')
p_txPower = re.compile('^\d+\.\d+txPower')
p_distance = re.compile('txPower-\d+\.\d+m')
p_bw = re.compile('m\d+BW')
p_subbw = re.compile('BW\d+sub_BW')
p_suboff = re.compile('sub_BW\d+sub_off')
p_numIntfBS = re.compile('\d+_intfBS*')
p_x = re.compile('paras_\d+\.csv')

def match(filename):
    res = {}
    # res['txPower'] = float(re.search(p_f, p_txPower.search(filename).group()).group())
    # res['distance'] = float(re.search(p_f, p_distance.search(filename).group()).group())
    # res['bw'] = int(re.search(p_d, p_bw.search(filename).group()).group())
    # res['subbw'] = int(re.search(p_d, p_subbw.search(filename).group()).group())
    # res['suboff'] = int(re.search(p_d, p_suboff.search(filename).group()).group())
    # res['num_intfBS'] = int(re.search())
    res['num_intfBS'] = (int(p_d.search(p_numIntfBS.search(filename).group()).group()))
    return res

def gen(folder, minmax):
    # num_intfBS = match(folder)['num_intfBS'] 
    if folder[-1] != '/':
        folder = folder + '/'
    files = os.listdir(folder)
    idxes = set()
    for f in files:
        if p_x.search(f) != None:
            idxes.add(int(p_d.search(f).group()))
    N = len(idxes)
    x = np.empty((N, 5))
    yhat = np.empty((N, 1))
    j = 0
    for i in tqdm.tqdm(idxes):
        px = folder + "paras_" + str(i) + ".csv"
        dfx = pd.read_csv(px)
        xi = dfx[dfx.columns[0:5]].to_numpy()
        x[j] = xi
        py = folder + str(i) + "-DL-SINR.csv"
        dfy = pd.read_csv(py)
        sinrs = dfy["SINR"].unique()
        assert(len(sinrs) == 1)
        sinr = sinrs[0]
        yhat[j] = math.log10(sinr) * 10
        j = j + 1
    np.save(folder[0:folder.rfind('/', 0, folder.rfind('/'))] + '/x.npy', x)
    np.save(folder[0:folder.rfind('/', 0, folder.rfind('/'))] + '/y.npy', yhat)

folder = "/home/rui/work/static-sinr-model/data/train/model-data/"

if __name__ == '__main__':
    gen(folder, None)
    for (dirpath, dirnames, filenames) in os.walk(
        folder[0:folder.rfind('/', 0, folder.rfind('/'))]):
        for f in filenames:
            if f.endswith('.npy'):
                os.system('ls -ahl ' + dirpath + '/' + f)