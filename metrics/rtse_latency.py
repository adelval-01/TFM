import os
import gc
import sys
import json
import time
import numpy as np
from scipy.io import wavfile
from scipy.io import savemat

import model_utils as mu

import torch
import torch.utils.data as data

import onnx
import onnxruntime

# 8k Net trained
workspace_dir = './afterburner8k_win20/'

sys.path.append(workspace_dir + 'src/net1')
sys.path.append(workspace_dir + 'src/train')
sys.path.append(workspace_dir + 'src/eval')
from vvtk_net.v1.utils import *
from vvtk_net.v1.transforms import *
from vvtk_net.v1.transforms_fe import *
from vvtk_net.v1.datafeed import *
from vvtk_net.v1.layers_pytorch import *
from eval_utils import *


def load_nets(workspace_dir):
    #======================= PYTORCH MODEL LOAD ===========================#
    print(f'\n  Loading Pytorch model')
    input_dim, output_dim = load_obj(workspace_dir + 'data/model/dimensions.pkl') 

    from net_snr import Net_snr 
    net_snr = Net_snr(input_dim, output_dim, cuda=True, single_gpu=True)
    net_snr.load( workspace_dir + 'data/model/theta_last')

    #========================= ONNX MODEL LOAD =============================#

    model_file = os.path.join('.','models', 'net_snr_w{}_s{}_{}.onnx').format(
        # int(cfg['analysis_window_length']*1000),
        # int(cfg['analysis_window_shift']*1000),
        # int(cfg['max_windows']))
        int(40),
        int(10),
        int(20))

    print(f'\n  Loading ONNX model from {model_file}')
    onnx_model = onnx.load(model_file)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(model_file, providers=["CUDAExecutionProvider"])

    return net_snr, ort_session

def run_rtse_latency(cfg, config_num, mw_len, df_len, workspace_dir, net_snr, ort_session):
    # Load configuration parameterss
    print(f'\n   |=====================================================================|')
    print(  f'   |              STARTING SPEECH ENHANCEMENT LATENCY TEST {config_num+1}             |')
    print(  f'   |=====================================================================|')

    #=======================================================================#
    if mw_len > 1:
        i_var_param = mw_len
    elif df_len > 1:
        i_var_param = df_len
    else:
        i_var_param = 1

    for i_param in range(i_var_param):
        # Parámetros
        fs=cfg["fs"]
        B=cfg["number_of_mel_filters"]
        w=cfg["analysis_window_length"]
        m=cfg["analysis_window_shift"]
        nfft=cfg["nfft"]
        gmin = cfg["gmin"]
        if mw_len > 1:
            min_windows = cfg["min_windows"][i_param]
        else:
            min_windows = cfg["min_windows"]
        max_windows = cfg["max_windows"]
        if df_len > 1:
            diezmation_factor = cfg["diezmation_factor"][i_param]
        else:
            diezmation_factor = cfg["diezmation_factor"]
        warm_up_factor = cfg["warm_up_factor"]

        #==================================================================#

        # Cálculo de los filtros
        N = int(w * fs)
        F = int(nfft/2)
        fb_time = time.time()
        fb = mu.fb_etsi(F, B, fs).astype(np.float32)
        # print(f'El tiempo de cálculo de los filtros es {(time.time() - fb_time)*1000} ms')
        dct_time = time.time()
        dct = mu.f_base_dct(B).astype(np.float32)
        # print(f'El tiempo de cálculo de las bases dct es {(time.time() - dct_time)*1000} ms')

        # Media y desviación para la normalización
        file = workspace_dir + 'data/model/fe1_norm1.pkl'
        mu_, std = read_pkl(file)
        mu_ = mu_.astype(np.float32)
        std = std.astype(np.float32)

        # Ventana de hamming
        hamming_win = np.hamming(fs * w)

        # x_test = ['./afterburner8k/data/audio/minitest_8k/5-CH0_C01_stadium_15dB.wav']

        file_list_ = cfg['audio_list']
        x_test = []

        for file_list in file_list_:
            x_test += read_list_str(file_list)

        print(f'\n  Audio files to process: {x_test}')


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
        print(f'  Warming up factor: {warm_up_factor}')
        print(f'  Diezmation factor: {diezmation_factor}')
        print(f'|----------------------------------------------------------------------------|\n')

        
                #======================== MODEL WARM UP ===========================#
        if warm_up_factor > 0 and config_num == 0:
            input_dim, output_dim = load_obj(workspace_dir + 'data/model/dimensions.pkl')
            print('\n  WARMING UP MODEL...')

            # Pytorch warm up
            num_windows = min_windows
            net_snr.model.eval()
            for n in range(warm_up_factor*(max_windows - min_windows + 1)):
                if n%warm_up_factor == 0:
                    num_windows += 1
                # print(f'  Warm up {n} of {warm_up_factor*(max_windows - min_windows)}')
                progresive_warm_up_tensor = torch.randn([1,num_windows-1, input_dim], dtype=torch.float32).cpu().numpy()
                # print(progresive_warm_up_tensor.shape)
                start_time = time.time()
                net_snr.predict(progresive_warm_up_tensor)
                # print(f'  PyTorch model warm up inference time: {(time.time() - start_time) * 1000} ms')

            # ONNX warm up (ONLY FOR STATIONAY WINDOWING)
            warm_up_tensor = torch.randn([1, max_windows, input_dim], dtype=torch.float32).cpu().numpy()
            ort_inputs = {ort_session.get_inputs()[0].name: warm_up_tensor}

            for _ in range(warm_up_factor):
                start_time = time.time()
                ort_session.run(None, ort_inputs)
                # print(f'  ONNX model warm up inference time: {(time.time() - start_time) * 1000} ms')

            print('  WARM UP DONE')

        #==================================================================#
        
        all_latency_data = {}

        for audio_file in x_test:
            print(f'\n|================================AUDIO {x_test.index(audio_file)+1}===================================|')
            # Cargar el audio
            print(f'  Audio file: {audio_file}')
            audio, fs = mu.read_audio(audio_file)
            if fs != cfg['fs']:
                print(f'  No se puede procesar el audio, frecuencia de muestreo del audio {fs}kHz != {cfg["fs"]}kHz configurada')
                break
            print(f'  Duración del audio: {len(audio)/cfg["fs"]}s - {len(audio)} samples')

            # #------------------------SNR PRE-----------------------------#
            # pre_vad = compute_vad(audio, fs, w, m, nfft)
            # snr_prev = int(wada_snr(audio, fs, pre_vad))
            # print('snr(wada)=%idB, file: %s' % (snr_prev, audio_file))
            # #------------------------------------------------------------#

            output_enh_dir = os.path.join('.','data','audio','enhanced')
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
            preprocess_latencies = []
            inference_latencies = []
            total_latencies = []
            

            # CALCULO DE LA MÁSCARA SNR 
            for n_frame in range(int((len(audio)/fs)*100)):
                # Obtener el frame de audio
                frame = audio[n_frame * shift_samples: n_frame * shift_samples + frame_samples]

                # print(f'\n{n_frame} frame de {len(frame)} --> {frame[:10]}')
                buffer_frame = np.concatenate([buffer_frame, frame])
                
                start_time = time.time()
                if len(buffer_frame) >= window_samples:
                    # Preprocessing: Feature extraction
                    start_preprocess = time.time()

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
                    preprocess_latencies.append((time.time()-start_preprocess)*1000)

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
                            inference_latencies.append((time.time()-start_prof)*1000)
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
                        inference_latencies.append((time.time()-start_prof)*1000)

                    buffer_frame = buffer_frame[shift_samples:]

                    # Aqui haría la evaluacion con la máscara pertinente (para las primeras 3 ventanas sin máscara calculada)
                    # cnt = int(n_frame - w/m) Ajustar al tamaño de la ventana
                    cnt = n_frame - 3
                    # print(f'EVALUATION OF WINDOW {cnt}')
                    x = np.array(work_window, dtype=np.float32) / 2 ** 15 

                    xenh, filt = mu.noiseReduction(x, snr_frame_mask[:,-1], fs, window_samples, shift_samples, nfft, gmin)

                    slice_size = min(len(yenh) - cnt * shift_samples, window_samples)

                    yenh[cnt * shift_samples : cnt * shift_samples + slice_size] += xenh[0:slice_size]


                end_time = time.time()
                total_latencies.append((end_time - start_time) * 1000)    
                # diff_acum += end_time - start_time
                if (end_time - start_time) > frame_size:
                    dropout_rate += 1
                # print(f'Tiempo de procesamiento del frame {n_frame} completo es de {(end_time-start_time)*1000} ms')

            total_audio_processing = time.time() - start_audio_processing
            total_latencies = np.array(total_latencies)
            mean_latency = np.mean(total_latencies)
            std_latency = np.std(total_latencies)
            tail_latency_99 = np.percentile(total_latencies, 99)

            rtf = mean_latency/ (cfg["frame_length"]*1000)
            if rtf < 1.0:
                color = "\033[92m"  # Green
            else:
                color = "\033[91m"  # Red

            print(f'\n|-----------------------------FRAME TIME STATS-------------------------------|')
            print(f'  Tiempo medio de procesamiento de la FFT: {((accum_fft/(n_frame-6))*1000):.4f} ms')
            print(f'  Tiempo medio de procesamiento de la escala log: {((accum_log/(n_frame-6))*1000):.4f} ms')
            print(f'  Tiempo medio de procesamiento de la FB MFCC: {((accum_fb_mfcc/(n_frame-6))*1000):.4f} ms')
            print(f'  Tiempo medio de extracción de features: {np.mean(preprocess_latencies):.4f} ms')
            print(f'  Tiempo medio de procesamiento de la inferencia: {((accum_inf/(n_frame-6))*1000):.4f} ms')
            print(f'  Tiempo medio de procesamiento de la inferencia: {(np.mean(inference_latencies)):.4f} ms')
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

            #--------------------------SNR POST--------------------------#
            # post_vad = compute_vad(yenh, fs, w, m, nfft)
            # snr_post = int(wada_snr(yenh_clipped, fs, pre_vad))
            # print('snr(wada)=%idB, file: %s' % (snr_post, output_enh))
            #------------------------------------------------------------#

            wavfile.write(output_enh_file,fs,yenh)

            latency_data = {
            f"f_{os.path.basename(audio_file.replace('-','_').replace('.','_'))}_total": np.array(total_latencies),
            f"f_{os.path.basename(audio_file.replace('-','_').replace('.','_'))}_inference": np.array(inference_latencies),
            f"f_{os.path.basename(audio_file.replace('-','_').replace('.','_'))}_preprocess": np.array(preprocess_latencies),
            }

            # Append to a global dictionary (you'd define this once before your for-loop)
            all_latency_data.update(latency_data)


        # Save time profiling into mat file to be plotted in MATLAB
        mat_file = os.path.join('.','metrics','time_profiling','latency_se_{}to{}_d{}_wu{}.mat').format(
            int(min_windows),
            int(max_windows),
            int(diezmation_factor),
            int(warm_up_factor))
        if not os.path.exists(os.path.dirname(mat_file)):
            os.makedirs(os.path.dirname(mat_file))
        savemat(mat_file, all_latency_data)
        print(f'\n  Latency data saved to {mat_file}')



if __name__ == '__main__':

    configs = []
    if len(sys.argv) > 1:
        # Load specific config file
        configs.append(sys.argv[1])
    else:
        exclude_configs = ['config-3.json', 'config-4.json', 'config-5.json', 'config-6.json']
        for filename in os.listdir('./metrics/configs'):
            if filename.endswith('.json') and filename not in exclude_configs:
                config = os.path.join('./metrics/configs', filename)
                configs.append(config)

    net_snr, ort_session = load_nets(workspace_dir)

    print(f"Loaded {len(configs)} config(s).")
    # Example use
    for config_num, config in enumerate(configs):
        print(f"\nConfig {config_num+1}:\n{json.dumps(config, indent=2)}")
        with open(config) as json_file:
            cfg = json.load(json_file)
        if isinstance(cfg['min_windows'], list):
            mw_len = len(cfg['min_windows'])
        else:
            mw_len = 1
        if isinstance(cfg['diezmation_factor'], list):
            df_len = len(cfg['diezmation_factor'])
        else: 
            df_len = 1
        run_rtse_latency(cfg, config_num, mw_len, df_len, workspace_dir, net_snr, ort_session)