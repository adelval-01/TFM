from __future__ import print_function
from __future__ import division
import time, os, sys
import numpy as np
import soundfile as sf
import pickle
import warnings
import gzip
import scipy.io
from scipy.io import wavfile
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
import logging

warnings.simplefilter('ignore')

import torch
import torch.utils.data as data

# 8k Net trained
workspace_dir = '/home/adelval/BTS/TFM/afterburner8k/'
# 16k Net trained
# workspace_dir = '/home/adelval/BTS/TFM/test/'

sys.path.append(workspace_dir + 'src/net')
sys.path.append(workspace_dir + 'src/train')
sys.path.append(workspace_dir + 'src/eval')
from vvtk_net.v1.utils import *
from vvtk_net.v1.transforms import *
from vvtk_net.v1.transforms_fe import *
from vvtk_net.v1.datafeed import *
from vvtk_net.v1.layers_pytorch import *
from vvtk_net.config import Configuration
from eval_utils import *





def offset(x):
    return signal.lfilter([1.0, -1.0],[1, -0.99899,], x)

def preemphasis(x):
    #return signal.lfilter([1.0, -0.97],1, x)    
    x0 = x[0] * (1.0 - 0.97)
    for i in reversed(range(len(x))): 
        x[i] = x[i] - x[i-1] * 0.97
    x[0] = x0
    return x

# Divide x into overlapping frames of fixed length without extending to slide last frame
def windowing2(x, fs=16000, Ns=0.025, Ms=0.010, Mw=20):
    # Mw limit the number of windows
    N = int(Ns * fs)            # Number of samples in each window 640
    M = int(Ms * fs)            # Step size (number of samples between window starts) 160
    n = (len(x) + M - 1) // M   # Number of frames 23
    # print("Number of frames", n)    
    # T = (n - 1) * M + N         # Total signal length needed to fit the frames
    xa = x.copy()
    # Se ignora el padding porque la ventana se deslizará
    # if T > len(x):
    #     print("rellena con ceros")
    #     xa.resize(T, refcheck=False)    # Pad the signal with zeros
    m = np.arange(0, Mw*M, M)          # Starting indices of each frame
    ind = np.arange(N).reshape(-1, 1) + m.reshape(1, -1) # 2D matrix whit rows correspnding to the samples whitin a window and columns the starting indeces each time
    xa = xa[ind.astype(int).T].astype(np.float32)
    # print(f'Window frames {xa[19,:5]}')

    return xa

def hamming(X):
    w = np.hamming(X.shape[1])
    return X * w

def fft(X, NFFT=None):
    if NFFT is None:
        NFFT = int(2 ** np.ceil(np.log2(X.shape[1])))
    Xfreq = np.abs(np.fft.fft(X, NFFT, axis=1)) # Computes the FFT along each row of X
    Xfreq = np.asarray(Xfreq, dtype=np.float32)
    return Xfreq[:, :(NFFT // 2)] # Return Half the Spectrum, only the first half is meaningful for real-valued signals


def frame_fft(data, fs, w, m, nfft, max_windows):
    
    N =[int(wi * fs) for wi in w]
    F =[int(2 ** np.ceil(np.log2( Ni)))//2 for Ni in N]

    x = offset(data)

    # Emphasis to increase the amplitude of high freq
    x = preemphasis(x)
    # print(f'0 frame   {x[:10]}')
    # print(f'1 frame   {x[1*int(m*fs):1*int(m*fs)+10]}')
    # print(f'2 frame   {x[2*int(m*fs):2*int(m*fs)+10]}')
    # print(f'3 frame   {x[3*int(m*fs):3*int(m*fs)+10]}')
    # print(f'20 frame  {x[19*int(m*fs):19*int(m*fs)+10]}')

    X = hamming(windowing2(x, fs=fs, Ns=w[0], Ms=m, Mw=max_windows))
    # Dimesions of X are (N,n) having in row the signal windowed with hamming of win length
    Xfft = fft(X, nfft[0])
    # print(f"la shape de Xfft es {Xfft.shape}")
    X = Xfft * Xfft # Power spectrum
    X = np.asarray(X, dtype=np.float32)
    
    return X    # Returns the PSD of the frame processed


# Log scale for PSD
def log_scale(frame_psd):
    scale=1.
    eps=1e-8
    x = frame_psd    
    x = np.abs(x)
    x = scale * np.log10(x + eps)  
    # print("Vector con Power Spectral Density in log scale del frame: \n", x[:10])

    return x   


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


def frame_fb_mfcc(data, fs, B, w, m, nfft, max_windows):

    N =[int(wi * fs) for wi in w]
    F = [int(nffti/2) for nffti in nfft]
    fb = [ fb_etsi(Fi, Bi, fs) for Fi, Bi in zip(F, B)]
    dct = [ f_base_dct(Bi) for Bi in B] 


    x = offset(data)
    x = preemphasis(x)
    
    X = hamming(windowing2(x, fs=fs, Ns=w[0], Ms=m, Mw=max_windows))
    
    Xfft = fft(X, nfft[0])
    # print(f'la shape en FCMFCC de Xfft es {Xfft.shape}')
    Xb = np.log(Xfft.dot( fb[0] ) + 1)
    Xc = Xb.dot(dct[0])                                
    
    X = np.concatenate( [Xb, Xc], 1 )
    
    X = np.asarray(X, dtype=np.float32)
    
    # print("El tamaño de x08k2 es: ",XX.shape)
    # print("Vector con FilterBank MFCC del frame: \n", X[:10])
    
    return X

def read_pkl(f):
    with open(f, 'rb') as file:  # Open the file in binary read mode
        return pickle.load(file)

def norm_fb_frame(frame_fbmfcc):
    # Normalization of fbmfcc
    # file = workspace_dir + 'data/model/fe1_norm1.pkl'  
    file = workspace_dir + 'data/model/fe1_norm1_rn.pkl' 
    x = frame_fbmfcc
    mu, std = read_pkl(file)
    x -= mu
    x /= std + 1e-6
    # print("Vector con FB MFCC normalizado del frame: \n", x[:10])

    return x
    

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



# Model inference
def net_eval(frame_concat, net_snr):

    net_snr.set_mode_train(False)
    x = frame_concat
    n_frames, fft_fb = x.shape
    x = x.reshape(1, n_frames, fft_fb)
    #x = x.unsqueeze(0) # Batch dimension
    # print(f'Las dimensiones de la muestra a evaluar son: {x.shape}')

    snr = net_snr.predict(x)
    snr = to_numpy(snr.squeeze())
    # print(f'La máscara del frame caculado es de {snr.shape}')
    
    #scipy.io.savemat(f, mdict={'snr': snr})
    # x = to_numpy(x.squeeze())
    # snr = to_numpy(snr.squeeze())
    return snr



# Aplicar mascara a frame 

def apply_filter(data, filt, frame=640, shift=160, nfft=1024):

    win = np.sqrt(np.hanning(frame))
    win = np.array(win, dtype=np.float32)
    
    yw = np.zeros(data.size)
    #print(f'La ventana se desplazara {it} veces')
    it = 1
    for i in range(0,it): #Por la cara, eliiminar bucle
        logging.debug(f'El frame sin enventanado con rn resulta {data[:10]}')
        xw = data[i*shift : i*shift+frame] * win
        logging.debug(f'El frame enventanado resulta {xw[:10]}')
        Xfft = np.fft.fft(xw, nfft)
        Xfft = Xfft[0:int(nfft/2+1)]
        logging.debug(f'El tamaño de la transformada del frame es {Xfft.shape}')
        logging.debug(f'La transformada del frame es {Xfft[:10]}')
        outf = Xfft * filt[:,i]
        logging.debug(f'El tamaño de la salida del filtro sera {outf.shape}')
        logging.debug(f'La salida del filtro es {outf[:10]}')
        fliped = np.flip(np.conj(outf[1:int(nfft/2)]), axis=0)
        outw = np.concatenate((outf, fliped), axis=0)
        outw = np.real(np.fft.ifft(outw, nfft, axis=0))
        logging.debug(f'El frame mejorado resulta sin OLA es {outw[:10]}')
        #print(f'La salida de la ifft tendra {outw.shape} samples') # de las que nos quedamos con 640 porque el resto son relleno
        yw[i*shift : i*shift+frame] = yw[i*shift : i*shift+frame] + outw[0:frame]*win
        logging.debug(f'El frame mejorado resulta {yw[:10]}')


    return yw

# ----------------------------------------------------------------------------------------------------------------------
def noiseReduction(data, snr_net, fs, frame, shift, nfft, gmin):
    
    # Add un-audible noise to avoid signals with 0
    data = data + 1e-7*np.random.rand(data.size) 
    snr_net = snr_net.reshape(-1,1) # para darle 2-D
    # Load snr_net
    snr_net = np.concatenate((snr_net, snr_net[int(nfft/2-1):, :]), axis=0)
    #snr_net = np.concatenate((snr_net, snr_net[int(nfft/2-1):, :]))
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
    logging.debug(f'El filtro será {filt[:10]}')
    yw = apply_filter(data, filt, frame, shift, nfft)

    return yw, filt
