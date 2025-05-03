import os
import gc
import sys
import json
import time
import numpy as np
from scipy.io import wavfile

import model_utils as mu

import torch
import torch.utils.data as data

import onnx
import onnxruntime

# 8k Net trained
workspace_dir = './afterburner8k_win20/'

reduced_net = False

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

# Load configuration parameterss
with open('metrics/config.json') as json_file:
    cfg = json.load(json_file)

print(f'\n   |=====================================================================|')
print(  f'   |              STARTING SPEECH ENHANCEMENT LATENCY TEST               |')
print(  f'   |=====================================================================|')

#======================= PYTORCH MODEL LOAD ===========================#
print(f'\n  Loading Pytorch model')
input_dim, output_dim = load_obj(workspace_dir + 'data/model/dimensions.pkl') 

from net_snr import Net_snr 
net_snr = Net_snr(input_dim, output_dim, cuda=True, single_gpu=True)
net_snr.load( workspace_dir + 'data/model/theta_last')

#========================= ONNX MODEL LOAD =============================#

model_file = os.path.join('.','models', 'net_snr_w{}_s{}_{}.onnx').format(
    int(cfg['analysis_window_length']*1000),
    int(cfg['analysis_window_shift']*1000),
    int(cfg['max_windows']))

print(f'\n  Loading ONNX model from {model_file}')
onnx_model = onnx.load(model_file)
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession(model_file, providers=["CUDAExecutionProvider"])

#=======================================================================#

# Parámetros
fs=cfg["fs"]
B=32
w=cfg["analysis_window_length"]
m=cfg["analysis_window_shift"]
nfft=cfg["nfft"]
gmin = cfg["gmin"]
min_windows = cfg["min_windows"]
max_windows = cfg["max_windows"]
diezmation_factor = cfg["diezmation_factor"]
WARM_UP_FACTOR = cfg["warm_up_factor"]

#======================== MODEL WARM UP ===========================#
if cfg["warm_up_factor"] > 0:

    print('\n  WARMING UP MODEL...')

    # Pytorch warm up
    num_windows = min_windows
    net_snr.model.eval()
    for n in range(WARM_UP_FACTOR*(max_windows - min_windows + 1)):
        if n%WARM_UP_FACTOR == 0:
            num_windows += 1
        # print(f'  Warm up {n} of {WARM_UP_FACTOR*(max_windows - min_windows)}')
        progresive_warm_up_tensor = torch.randn([1,num_windows-1, input_dim], dtype=torch.float32).cpu().numpy()
        # print(progresive_warm_up_tensor.shape)
        start_time = time.time()
        net_snr.predict(progresive_warm_up_tensor)
        # print(f'  PyTorch model warm up inference time: {(time.time() - start_time) * 1000} ms')

    # ONNX warm up (ONLY FOR STATIONAY WINDOWING)
    warm_up_tensor = torch.randn([1, max_windows, input_dim], dtype=torch.float32).cpu().numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: warm_up_tensor}

    for _ in range(WARM_UP_FACTOR):
        start_time = time.time()
        ort_session.run(None, ort_inputs)
        # print(f'  ONNX model warm up inference time: {(time.time() - start_time) * 1000} ms')

    print('  WARM UP DONE')

#==================================================================#



# Cálculo de los filtros
N = int(w * fs)
F = int(nfft/2)
fb_time = time.time()
fb = mu.fb_etsi(F, B, fs).astype(np.float32)
print(f'FB : {fb.shape} y {fb.dtype}')
# print(f'El tiempo de cálculo de los filtros es {(time.time() - fb_time)*1000} ms')
dct_time = time.time()
dct = mu.f_base_dct(B).astype(np.float32)
print(f'DCT: {dct.shape} y {dct.dtype}')
# print(f'El tiempo de cálculo de las bases dct es {(time.time() - dct_time)*1000} ms')
# Media y desviación para la normalización
file = workspace_dir + 'data/model/fe1_norm1.pkl'
mu_, std = read_pkl(file)
mu_ = mu_.astype(np.float32)
std = std.astype(np.float32)
print(mu_.dtype)
print(std.dtype)

# Ventana de hamming
hamming_win = np.hamming(fs * w)

# x_test = ['./afterburner8k/data/audio/minitest_8k/5-CH0_C01_stadium_15dB.wav']
x_test = ['./afterburner8k/data/audio/minitest_8k/5-CH0_C01_stadium_15dB.wav',
          './afterburner8k/data/audio/minitest_8k/6-CH0_C01_traffic_15dB.wav', 
          './afterburner8k/data/audio/minitest_8k/7-CH0_C01_city_5dB.wav',
          './afterburner8k/data/audio/minitest_8k/10-CH0_C01_airport_10dB.wav',
          './afterburner8k/data/audio/minitest_8k/12-CH0_C01_babies_5dB.wav']


print(f'\n|-----------------------------INITIAL PARAMETERS-----------------------------|')
print(f'  Frecuencia de muestreo: {cfg["fs"]} Hz')
frame_size = 0.01  # 10 ms
frame_samples = int(cfg["frame_length"] * cfg["fs"])  # Muestras por frame
print(f'  Tamaño de frame: {frame_samples} samples')
shift_size = m  # 10 ms
shift_samples = int(cfg["analysis_window_shift"] * fs)  # Desplazamiento entre frames 160
print(f'  Desplazamiento: {shift_samples} samples')
window_size = w  # 40 ms (640 muestras)
window_samples = int(cfg["analysis_window_length"] * fs)  # Muestras por ventana 640
print(f'  Tamaño de ventana: {window_samples} samples')
window_inference_min = int((min_windows + (window_size/frame_size)-1) * frame_samples)
window_inference_max = int((max_windows + (window_size/frame_size)-1) * frame_samples)
print(f'  Buffer progresivo: {window_inference_min} - {window_inference_max} samples')
print(f'|----------------------------------------------------------------------------|\n')


for audio_file in x_test:
    print(f'\n|================================AUDIO {x_test.index(audio_file)+1}===================================|')
    # Cargar el audio
    print(f'  Audio file: {audio_file}')
    audio, fs = mu.read_audio(audio_file)
    if fs != cfg['fs']:
        print(f'  No se puede procesar el audio, frecuencia de muestreo del audio {fs}kHz != {cfg["fs"]}kHz configurada')
        break
    print(f'  Duración del audio: {len(audio)/cfg["fs"]}s - {len(audio)} samples')

    output_enh_dir = os.path.join('.','audios','enhanced')
    if not os.path.exists(output_enh_dir):
        os.makedirs(output_enh_dir)
    output_enh_file = os.path.basename(audio_file).replace('.wav','_w{}_s{}_{}to{}_d{}.wav').format(
        int(cfg['analysis_window_length']*1000),
        int(cfg['analysis_window_shift']*1000),
        int(min_windows),
        int(max_windows),
        int(diezmation_factor))
    output_enh_file = os.path.join(output_enh_dir, output_enh_file)
    print(f'  Enhanced Audio: {output_enh_file}')


    # Inicializar variables
    start_audio_processing = time.time()
    it = 0
    buffer_frame = np.zeros(0)  # Buffer de ventana recibida
    Xfft_windows_list = []
    X_mfcc_buffer = np.empty(64, dtype=np.float32)
    Xb_preallocated = np.empty(32, dtype=np.float32)
    snr_frame_mask = np.ones((512,min_windows)) # Inicializado con la duración de la ventana de inferencia
    yenh = np.zeros(len(audio)) # Inicializado con la duración del audio original

    # Variables globales para el cálculo de los tiempos
    accum_offset = 0
    accum_preemphasis = 0
    accum_windowing = 0
    accum_fft = 0
    accum_log = 0
    accum_fb_mfcc = 0
    accum_norm = 0 
    accum_inf = 0
    diff_acum = 0
    diff_inf_acum = 0
    dropout_rate = 0
    accum_delay = 0
    latencies = []

    # CALCULO DE LA MÁSCARA SNR 
    for n_frame in range(int((len(audio)/fs)*100)):
        # Obtener el frame de audio
        frame = audio[n_frame * shift_samples: n_frame * shift_samples + frame_samples]

        # print(f'\n{n_frame} frame de {len(frame)} --> {frame[:10]}')
        buffer_frame = np.concatenate([buffer_frame, frame])
        
        start_time = time.time()
        if len(buffer_frame) >= window_samples:
            work_window = buffer_frame[:window_samples]

            start_fft = time.time()
            Xfft = mu.window_fft(work_window, hamming_win, nfft)
            accum_fft += (time.time()-start_fft)

            start_log = time.time()
            log_psd_Xfft = mu.log_psd(Xfft)
            accum_log += (time.time()-start_log)

            start_fb = time.time()
            fb_windows = mu.frame_fb_mfcc(Xfft, Xb_preallocated, fb, dct, mu_, std, X_mfcc_buffer)
            if time.time()-start_fb > 0.001:
                print(f'WARNING El tiempo {n_frame} de cálculo de la FB MFCC es: {(time.time()-start_fb)*1000} ms')
            accum_fb_mfcc += (time.time()-start_fb)

            windows_concat = np.concatenate((log_psd_Xfft,fb_windows))
            Xfft_windows_list.append(windows_concat)

            if n_frame-3 < max_windows:   
                if n_frame-3 >= min_windows:
                    # print("Reached MIN WINDOW --> STRATING INFERENCE")
                    transformed_windows = np.vstack(Xfft_windows_list)

                    start_prof = time.time()
                    if(n_frame % diezmation_factor == 0):
                        snr_frame_mask = mu.net_eval(net_snr, transformed_windows)
                        snr_frame_mask = snr_frame_mask.T
                        # print(f'Time {n_frame}: {(time.time()-start_prof)*1000} ms')
                    accum_inf += (time.time()-start_prof)
            # Si el buffer alcanza o excede las 20 ventanas para hacer la inferencia
            else:
                Xfft_windows_list.pop(0)
                transformed_windows = np.vstack(Xfft_windows_list)

                start_prof = time.time()
                if(n_frame % diezmation_factor == 0):
                    # snr_frame_mask = mu.net_eval(net_snr, transformed_windows)
                    # snr_frame_mask = snr_frame_mask.T
                    transformed_windows = transformed_windows.reshape(1, max_windows, 576)
                    ort_inputs = {ort_session.get_inputs()[0].name: transformed_windows}
                    snr_frame_mask = ort_session.run(None, ort_inputs)[0]
                    snr_frame_mask = snr_frame_mask.squeeze().T
                    # print(f'Time onnx {n_frame}: {(time.time()-start_prof)*1000} ms')
                accum_inf += (time.time()-start_prof)

            buffer_frame = buffer_frame[shift_samples:]

            # Aqui haría la evaluacion con la máscara pertinente (para las primeras 3 ventanas sin máscara calculada)
            # cnt = int(n_frame - w/m) Ajustar al tamaño de la ventana
            cnt = n_frame - 3
            # print(f'EVALUATION OF WINDOW {cnt}')
            x = np.array(work_window, dtype=np.float32) / 2 ** 15 # 0.04 * fs = 640 samples

            xenh, filt = mu.noiseReduction(x, snr_frame_mask[:,-1], fs, window_samples, shift_samples, nfft, gmin)

            slice_size = min(len(yenh) - cnt * shift_samples, window_samples)

            yenh[cnt * shift_samples : cnt * shift_samples + slice_size] += xenh[0:slice_size]


        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)    
        # diff_acum += end_time - start_time
        if (end_time - start_time) > frame_size:
            dropout_rate += 1
        # print(f'Tiempo de procesamiento del frame {n_frame} completo es de {(end_time-start_time)*1000} ms')

    total_audio_processing = time.time() - start_audio_processing
    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    tail_latency_99 = np.percentile(latencies, 99)

    rtf = mean_latency/ (cfg["frame_length"]*1000)
    if rtf < 1.0:
        color = "\033[92m"  # Green
    else:
        color = "\033[91m"  # Red

    print(f'\n|-----------------------------FRAME TIME STATS-------------------------------|')
    print(f'  Tiempo medio de procesamiento de la FFT: {((accum_fft/(n_frame-6))*1000):.4f} ms')
    print(f'  Tiempo medio de procesamiento de la escala log: {((accum_log/(n_frame-6))*1000):.4f} ms')
    print(f'  Tiempo medio de procesamiento de la FB MFCC: {((accum_fb_mfcc/(n_frame-6))*1000):.4f} ms')
    print(f'  Tiempo medio de procesamiento de la inferencia: {((accum_inf/(n_frame-6))*1000):.4f} ms')
    print(f'  Tiempo medio de procesamiento total: {(mean_latency):.4f} ms')
    print(f'  Desviación estándar de la latencia media: ±{std_latency:.4f} ms')
    print(f'\n|-----------------------------TOTAL TIME STATS-------------------------------|')
    print(f'  Tiempo total de procesamiento del audio: {(total_audio_processing):.4f} s')
    print(f'  REAL TIME FACTOR (RTF): {color}{rtf:.4f}\033[0m')
    print(f'  Frames con delay > {int(cfg["frame_length"]*1000)}ms: {(dropout_rate/n_frame*100):.2f}%')
    print(f"  99th Percentile Latency: {tail_latency_99:.4f} ms")
    print(f'|----------------------------------------------------------------------------|\n')

    yenh = yenh/3     # 4 because in OverLapAdd we sum 4 times the frame

    yenh = np.array(yenh*(2 ** 15), dtype=np.int16)     # set int16 wav format

    wavfile.write(output_enh_file,fs,yenh)