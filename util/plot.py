from statistics import correlation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import torch
from scipy.optimize import curve_fit, nnls
import math
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from data_loader import StaticSinrDataset
from model import StaticSINRModel 

# plt.rcParams['text.usetex'] = True 

test_folder = '/home/rui/work/static-sinr-model/data/test/'
test_dataset = StaticSinrDataset(test_folder)

test_dataloader  = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=None,
                                                batch_sampler=None)

model_path = '/home/rui/work/static-sinr-model/check_points/2022-12-18 16:53:26/epoch_994_357.78326416015625'
model = StaticSINRModel()
model.load_state_dict(torch.load(model_path))

criterion = torch.nn.MSELoss(reduction='sum')

model.eval()
sum_test_loss = 0.0

ys = []
yhats = []

with torch.no_grad():
    for idx, data in enumerate(test_dataloader):
        x = data['x'].float()
        y = data['y'].float()
        yhat = model(x)
        loss = criterion(yhat, y.data)
        sum_test_loss = sum_test_loss + loss
        ys.append(y)
        yhats.append(yhat)

test_loss = sum_test_loss / len(test_dataloader)
print('Loss: ', test_loss)

ys = torch.stack(ys)
yhats = torch.stack(yhats)
 
ys = ys.detach().numpy()
yhats = yhats.detach().numpy()

def cor(y, yhat):
    ybar = y.mean()
    yhatbar = yhat.mean()
    numerator = np.sum((y - ybar) * (yhat - yhatbar))
    denominator = np.sqrt(np.sum(np.power(y - ybar, 2)) * np.sum(np.power(yhat - yhatbar, 2)))
    return numerator / denominator

correlation = cor(ys, yhats)

print(ys.shape)
print(yhats.shape)

def line_func(x, w, b):
    return x * w + b

lopt, lcov = curve_fit(line_func, ys.reshape(-1), yhats.reshape(-1))
print(lopt)

fig, ax = plt.subplots(facecolor=(1, 1, 1))
ax.scatter(yhats, ys, c='#2b6a99', s=10, alpha=0.4)
ax.plot(yhats, line_func(yhats, *lopt), c='#f16c23', ls='dotted', label=r'yhat='+str(lopt[0])+'y+'+str(lopt[1]))
# ax.plot(y, y, c='gray', ls='dotted')#, label=r'\hat{y}='+str(lopt[0])+'y+'+str(lopt[1]))
ax.set_xlabel(r'$\hat{y}$', fontsize=15)
ax.set_ylabel(r'$y$', fontsize=15)
# ax.legend()
fig.suptitle("Loss (MSE) = " + '{:.4f}'.format(test_loss.item()) + " Corr = " + '{:.4f}'.format(correlation))
# fig.suptitle("Loss (MSE)" + test_loss+ "Corr = " + str(correlation))
fig.tight_layout()
fig.savefig('test2.png', dpi=300)
# plt.show()






