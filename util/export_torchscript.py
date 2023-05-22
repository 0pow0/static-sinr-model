import torch
import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
print(currentdir, parentdir)
sys.path.insert(0, parentdir) 

from model import StaticSINRModel

p_model_state_dict = '/home/rui/work/static-sinr-model/check_points/2022-12-18 16:53:26/epoch_994_357.78326416015625'
model = StaticSINRModel() 
print('1')
model.load_state_dict(torch.load(p_model_state_dict))
print('2')

model.eval()

x = torch.rand(10, 5)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
# traced_script_module = torch.jit.trace(model, x)

# p_traced_model = '/home/rui/work/static-sinr-model/check_points/2022-12-18 16:53:26/traced_static-sinr-model.pt'
# traced_script_module.save(p_traced_model)

x = torch.tensor([[10.0, 20.0, 30.0, 40.0, 5.0]])
print(model(x))