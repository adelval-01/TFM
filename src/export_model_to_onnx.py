import sys
import os
import pickle
import torch
import gzip
import json

workspace_dir = '/home/adelval/BTS/TFM/afterburner8k_win20/'
sys.path.append(workspace_dir + 'src/net1')
sys.path.append(workspace_dir + 'src/train')
sys.path.append(workspace_dir + 'src/eval')
from vvtk_net.v1.utils import *
from vvtk_net.v1.transforms import *
from vvtk_net.v1.transforms_fe import *
from vvtk_net.v1.datafeed import *
from vvtk_net.v1.layers_pytorch import *
from vvtk_net.config import Configuration
from eval_utils import *


# Load configuration parameters
with open('metrics/config.json') as json_file:
    cfg = json.load(json_file)

# Load model dimensions and weights for windowing

def load_obj(file):
    if not isinstance(file,str):
        return pickle.load(f)

    root,ext = os.path.splitext(file)
    if ext == '.gz':
        with gzip.open(file, 'rb') as f:
            return pickle.load(f)
    else:
        with open(file, 'rb') as f:
            return pickle.load(f)


input_dim, output_dim = load_obj(workspace_dir + 'data/model/dimensions.pkl') 


print('  input_dim: %s' % str(input_dim))
print('  output_dim: %s' % str(output_dim))

# Load net for windowing
sys.path.append( workspace_dir + 'src/net1')
from net_snr import Net_snr 
net_snr = Net_snr(input_dim, output_dim, cuda=True, single_gpu=True)
net_snr.load( workspace_dir + 'data/model/theta_last')
net_snr.set_mode_train(False)
net_snr.model.eval()

# Export model to onnx
calls = 1

model_dir = os.path.join('.','models')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_file = os.path.join(model_dir, 'net_snr_w{}_s{}_{}to{}_d{}.onnx').format(
    int(cfg['analysis_window_length']*1000),
    int(cfg['analysis_window_shift']*1000),
    int(cfg['min_windows']),
    int(cfg['max_windows']),
    int(cfg['diezmation_factor']))
print('Model file: %s' % model_file)

print('Exporting net_snr model to onnx')
print('Calls: %i, Windows: %i' % (calls, cfg['max_windows']))
tensor = torch.ones([calls, cfg['max_windows'] , input_dim], dtype=torch.float32).cuda()
torch.onnx.export(net_snr.model, tensor, model_file,
        export_params=True,
        verbose=False,
        do_constant_folding = False,  
        input_names = ['input'],
        output_names = ['output']
    )
print('Model exported to %s' % model_file)
