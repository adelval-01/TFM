# import time, os, sys
# import numpy as np
# import soundfile as sf
# import pickle
# import warnings
# import gzip
# import scipy.io
# from scipy import signal
# from scipy.io import wavfile
# import scipy.special as sc

# workspace_dir = '/home/adelval/BTS/TFM/afterburner8k_win20/'

# sys.path.append(workspace_dir + 'src/net1')
# sys.path.append(workspace_dir + 'src/train')
# sys.path.append(workspace_dir + 'src/eval')
# from vvtk_net.v1.utils import *
# from vvtk_net.v1.transforms import *
# from vvtk_net.v1.transforms_fe import *
# from vvtk_net.v1.datafeed import *
# from vvtk_net.v1.layers_pytorch import *
# from vvtk_net.config import Configuration
# from eval_utils import *


# imports from run_test
from __future__ import print_function
from __future__ import division
import time, os, sys
import numpy as np
import soundfile as sf
import pickle
import warnings
import gzip
import scipy.io
from scipy import signal
from scipy.io import wavfile
from scipy.io import loadmat
# import matplotlib.pyplot as plt

warnings.simplefilter('ignore')

import torch
import torch.utils.data as data

# 8k Net trained
# workspace_dir = '/home/adelval/BTS/TFM/afterburner8k/'
workspace_dir = '/home/adelval/BTS/TFM/afterburner8k_win20/'
# 16k Net trained
# workspace_dir = '/home/adelval/BTS/TFM/test/'


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

#======================= READS AND LOADS ===========================#

def read_audio(f):
    x, fs = sf.read(f)
    return np.array(x * 2**15, dtype=np.int16), fs


def read_pkl(f):
    with open(f, 'rb') as file: 
        return pickle.load(file)


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

#===================================================================#

#======================== FEATURE EXTRACTION =======================#

def offset(x):
    return signal.lfilter([1.0, -1.0],[1, -0.99899,], x)


def optimized_preemphasis(x, alpha=0.97):
    x = np.array(x, dtype=np.float64)
    x[1:] = x[1:] - alpha * x[:-1]
    x[0] = x[0] * (1 - alpha)
    return x


def fft(X, NFFT):
    Xfreq = np.abs(np.fft.fft(X, NFFT)) # Computes the FFT along each row of X
    Xfreq = np.asarray(Xfreq, dtype=np.float32)
    return Xfreq[:(NFFT // 2)] # Return Half the Spectrum, only the first half is meaningful for real-valued signals


def window_fft(x, hamming_win, nfft):
    x = offset(x)
    x = optimized_preemphasis(x) 
    x = x*hamming_win

    return fft(x, nfft)


def log_psd(Xfft, eps=1e-8):
    """
    Computes the log-scaled power spectral density (PSD) from the FFT output.
    
    Parameters:
        Xfft (np.ndarray): FFT output, shape (N, nfft//2)
        scale (float): Multiplier for log scale, typically 2.0 for power (or 10/20 for dB)
        eps (float): Small constant to avoid log(0)
    
    Returns:
        np.ndarray: Log-scaled PSD
    """
    Xfft = np.asarray(Xfft, dtype=np.float32)
    X_log_psd = 2 * np.log10(np.abs(Xfft) + eps)
    return X_log_psd


#FB MFCC Filter Bank Mel-Frequency Cepstral Coefficients
def fb_etsi(F, B, fs):
    StFreq = 64.0
    fb = np.zeros((F, B))
    # /* Constants for calculation*/
    start_mel = 2595.0 * np.log10(1.0 + StFreq / 700)
    fs_per_2_mel = 2595.0 * np.log10(1.0 + (fs / 2) / 700)
    for b in range(B):
        # /* Calculating mel-scaled frequency and the corresponding FFT-bin */
        # /* number for the lower edge of the band                          */
        freq = 700 * (np.power(10.0, (start_mel + (b) / (B + 1) * (fs_per_2_mel - start_mel)) / 2595) - 1.0)
        f1 = (2 * F * freq / fs + 0.5)
        # /* Calculating mel-scaled frequency for the upper edge of the band */
        freq = 700 * (
        np.power(10.0, (start_mel + (b + 2) / (B + 1) * (fs_per_2_mel - start_mel)) / 2595) - 1.0)
        # /* Calculating and storing the length of the band in terms of FFT-bins*/
        f3 = (2 * F * freq / fs + 0.5)
        f2 = (f1 + f3) / 2
        f3 = min(f3, F - 1)
        s = 0.0
        f1 = int(f1)
        f2 = int(f2)
        f3 = int(f3)
        for f in range(f1, f2):
            fb[f, b] = f * (1 / (f2 - f1)) - f1 / (f2 - f1)
            s += fb[f, b]
        for f in range(f2, f3):
            fb[f, b] = f * (-1 / (f3 - f2)) + f3 / (f3 - f2)
            s += fb[f, b]
        for f in range(f1, f3):
            fb[f, b] /= s  # //normalization
    return fb


def f_base_dct(N):
    b = np.zeros((N, N))
    for n in range(N):
        if n == 0:
            kn = np.sqrt(1 / N)
        else:
            kn = np.sqrt(2 / N)
        for m in range(N):
            b[m, n] = kn * np.cos((2 * m + 1) * n * np.pi / (2 * N))
    return b



def frame_fb_mfcc(Xfft, Xb, fb, dct, mu, std, X_buffer):

    start_xb_dot = time.time()
    # Xb = np.log(np.dot(Xfft, fb ) + 1.0)
    Xb = np.dot(Xfft, fb) + 1.0
    Xb = (Xfft @ fb) + 1.0
    # print(f'Xb time:', (time.time() - start_xb_dot)*1000)
    start_xb_log = time.time()
    Xb = np.log(Xb)
    # print(f'Xb log time:', (time.time() - start_xb_log)*1000)
    start_xc = time.time()
    Xc = np.dot(Xb, dct)             
    # print('Xc time:', (time.time() - start_xc)*1000)           

    # X = np.concatenate( [Xb, Xc] )
    
    # X = np.empty(Xb.shape[0] + Xc.shape[0], dtype=np.float32)
    X_buffer[:Xb.shape[0]] = Xb
    X_buffer[Xb.shape[0]:] = Xc
    
    # X = np.asarray(X, dtype=np.float32)

    X_buffer -= mu
    X_buffer /= std + 1e-6
    
    return X_buffer

#====================================================================#

#======================== MASK EVALUATION ===========================#

def net_eval(net_snr, frame_concat):

    net_snr.set_mode_train(False)

    x = frame_concat
    n_frames, fft_fb = x.shape
    x = x.reshape(1, n_frames, fft_fb)

    snr = net_snr.predict(x)
    snr = to_numpy(snr.squeeze())

    return snr


def apply_filter(data, filt, frame=640, shift=160, nfft=1024):

    win = np.sqrt(np.hanning(frame))
    win = np.array(win, dtype=np.float32)
    
    yw = np.zeros(data.size)

    it = 1
    for i in range(0,it):
        xw = data[i*shift : i*shift+frame] * win
        Xfft = np.fft.fft(xw, nfft)
        Xfft = Xfft[0:int(nfft/2+1)]
        outf = Xfft * filt[:,i]
        fliped = np.flip(np.conj(outf[1:int(nfft/2)]), axis=0)
        outw = np.concatenate((outf, fliped), axis=0)
        outw = np.real(np.fft.ifft(outw, nfft, axis=0))

        yw[i*shift : i*shift+frame] = yw[i*shift : i*shift+frame] + outw[0:frame]*win

    return yw


def noiseReduction(data, snr_net, fs, frame, shift, nfft, gmin):
    
    # Add un-audible noise to avoid signals with 0
    data = data + 1e-7*np.random.rand(data.size) 
    snr_net = snr_net.reshape(-1,1) # para darle 2-D
    snr_net = np.concatenate((snr_net, snr_net[int(nfft/2-1):, :]), axis=0)
    # VAD
    ini = int(np.floor((nfft*300)/fs))
    out = int(np.ceil((nfft*2500)/fs))
    E = np.mean(snr_net[ini:out,:], axis=0)
    vad = 1/(1 + np.exp(-90*(E-0.05)))

    difference = 1 - snr_net
    difference[difference < 1e-7] = 1e-7
    gamma = 1 / difference
    upsilon = gamma * snr_net
    g = np.exp(0.5 * sc.exp1(upsilon)) * snr_net
    g[g > 1] = 1
    gmin = gmin/2
    gtotal = np.power(g, snr_net) * np.power((gmin * snr_net), (1 - snr_net))
    filt = np.power(gtotal,vad) * np.power(1e-3,1-vad)
    yw = apply_filter(data, filt, frame, shift, nfft)

    return yw, filt

#====================================================================#

def compute_vad(x, fs=16000, N=0.025, M=0.010, nfft=512):  
    maxamplitude, th, lambd, n, coef_floor, alpha  = 0.75, 4, 0.6, 4, 0.22, 0.9
    x = maxamplitude * (x / (np.abs(x).max()))
    
    frame = int(N * fs)
    shift = int(M * fs)     
    Nframes = int(1 + np.ceil((x.size-shift)/shift))  
        
    x = preemphasis(offset(x))
    X = windowing2(x, fs, N, M) * np.hamming(frame)    
    Xfft = np.abs(np.fft.fft(X, nfft))
    Xfft = Xfft[:, 1:int(nfft/2+1)]
    
    # init
    LTSE = np.zeros(Xfft.shape)
    r_est = np.zeros(Xfft.shape)
    r_lt = np.zeros(Xfft.shape)
    LTSD = np.zeros(Nframes)
    vad = np.zeros(Nframes)

    LTSE[0] = np.max(Xfft[0:n+1],axis=0)
    suelo = np.ones(int(nfft/2)) * (coef_floor * np.power(np.mean(Xfft), 2))
    r_est[0] = np.mean(np.power(Xfft[0:n],2),0)
    r_lt[0] = r_est[0]
    LTSD[0] = 10*np.log10(1/(nfft)*sum(np.power(LTSE[0],2)/r_est[0]))
    vad[0] = int(LTSD[0] > th)

    # LTSE & LTSD => vad
    for j in range(1,Nframes-1):
        a = np.max([j - n, 1])
        aa = np.min([j + n, Nframes + 1])
        LTSE[j] = np.max(Xfft[np.max([j-n, 0]):np.min([j+n, Nframes+1])+1],axis=0)

        if vad[j-1]:
            r_est[j] = r_est[j-1]
        else:
            r_est[j] = lambd*r_est[j-1]+(1-lambd)*np.power(Xfft[j],2) #MCRA sustituir...ver el de C

        r_lt[j] = alpha*r_lt[j-1]+(1-alpha)*np.min([np.power(Xfft[j],2),r_est[j]],0)
        LTSD[j] = 10*np.log10(1/(nfft)*sum(np.power(LTSE[j],2)/np.max([r_lt[j],suelo],0)))
        vad[j] = int(LTSD[j] > th)

    vadsamples = np.kron(vad, np.ones(shift))
    vadsamples = np.reshape(vadsamples, (1,np.product(vadsamples.shape)))[0]
    vadsamples = vadsamples[0:x.size]
    vadsamples = np.asarray(vadsamples, dtype=np.int16)

    return vadsamples, vad   