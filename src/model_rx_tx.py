import os
import sys
import wave
import time
import json
import asyncio
import logging
import numpy as np
from scipy.io import wavfile
from livekit import rtc, api
from signal import SIGINT, SIGTERM

import model_utils as mu

import torch
import torch.utils.data as data

import onnx
import onnxruntime

SAMPLE_RATE = 8000
FRAME_DURATION_MS = 10
NUM_CHANNELS = 1
FORMAT = 2 # 16-bit PCM
WAV_FILE = "./audios/livekit/audio_received_lk.wav"
WAV_ENH = "./audios/livekit/audio_enhanced_lk.wav"

# Load configuration parameterss
with open('metrics/config.json') as json_file:
    cfg = json.load(json_file)

# Load model dimensions and weights
# 8k Net trained
workspace_dir = './afterburner8k_win20/'

# 16k Net trained
# workspace_dir = '/home/adelval/BTS/TFM/test/'

sys.path.append( workspace_dir + 'src/net1')

#======================= PYTORCH MODEL LOAD ===========================#
print(f'  \nLoading Pytorch model')
input_dim, output_dim = mu.load_obj(workspace_dir + 'data/model/dimensions.pkl') 

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

#======================== MODEL WARM UP ===========================#
if cfg["warm_up_factor"] > 0:

    print('\n  WARMING UP MODEL...')

    # Pytorch warm up
    num_windows = cfg["min_windows"]
    net_snr.model.eval()
    for n in range(cfg["warm_up_factor"]*(cfg["max_windows"] - cfg["min_windows"] + 1)):
        if n%cfg["warm_up_factor"] == 0:
            num_windows += 1
        # print(f'  Warm up {n} of {WARM_UP_FACTOR*(max_windows - min_windows)}')
        progresive_warm_up_tensor = torch.randn([1,num_windows-1, input_dim], dtype=torch.float32).cpu().numpy()
        # print(progresive_warm_up_tensor.shape)
        start_time = time.time()
        net_snr.predict(progresive_warm_up_tensor)
        # print(f'  PyTorch model warm up inference time: {(time.time() - start_time) * 1000} ms')

    # ONNX warm up (ONLY FOR STATIONAY WINDOWING)
    warm_up_tensor = torch.randn([1, cfg["max_windows"], input_dim], dtype=torch.float32).cpu().numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: warm_up_tensor}

    for _ in range(cfg["warm_up_factor"]):
        start_time = time.time()
        ort_session.run(None, ort_inputs)
        # print(f'  ONNX model warm up inference time: {(time.time() - start_time) * 1000} ms')

    print('  WARM UP DONE')

#=======================================================================#

# Initializate wav file
def setup_wav_file():
    os.makedirs(os.path.dirname(WAV_FILE), exist_ok=True)
    wav_file = wave.open(WAV_FILE, 'wb')
    wav_file.setnchannels(NUM_CHANNELS)
    wav_file.setsampwidth(FORMAT)
    wav_file.setframerate(SAMPLE_RATE)
    return wav_file

async def main(room_1: rtc.Room, room_2: rtc.Room) -> None:
    """
    Main function to connect participant to 2 rooms. From the fisrt room, the participant will receive audio frames
    and send them to the second room.
    Args:
        room_1: rtc.Room object --> Room where the participant will receive audio frames
        room_2: rtc.Room object --> Room where the participant will send audio frames
    """
    
    wav = setup_wav_file()                          # Initialize wav file to save audio frames
    stop_processing = asyncio.Event()               # Event to signal when to stop processing

    @room_1.on("participant_disconnected")
    def on_participant_disconnect(participant: rtc.Participant, *_):
        logging.info("participant disconnected: %s", participant.identity)

    @room_1.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logging.info("track subscribed: %s", publication.sid)
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print("Subscribed to an Audio Track")
            _audio_stream = rtc.AudioStream(
                track,
                sample_rate=SAMPLE_RATE, 
                num_channels=NUM_CHANNELS
            )
            # audio_stream is an async iterator that yields AudioFrame

        # Start an async task to handle the audio frames
        asyncio.create_task(process_audio_stream(_audio_stream, source))
        
    @room_1.on("track_unpublished")
    def on_track_unpublished(
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logging.info("Track unpublished: %s", publication.sid)
        stop_processing.set()                       # Signal to stop audio processing

    async def process_audio_stream(audio_stream, source):
        """
        Process audio frames from the audio stream and send them to the source
        Args:
            audio_stream: rtc.AudioStream object --> Audio stream from the subscribed track
            source: rtc.AudioSource object --> Audio source to publish the audio frames
        """

        try:
            #--------------------------PARAMETERS FOR SE MODEL------------------------------------#
        
            fs=cfg["fs"]
            B=cfg["number_of_mel_filters"]
            w=cfg["analysis_window_length"]
            m=cfg["analysis_window_shift"]
            nfft=cfg["nfft"]
            gmin = cfg["gmin"]
            min_windows = cfg["min_windows"]
            max_windows = cfg["max_windows"]
            diezmation_factor = cfg["diezmation_factor"]
            warm_up_factor = cfg["warm_up_factor"]


            # Cálculo de los filtros
            N = int(w * fs)
            F = int(nfft/2)
            fb_time = time.time()
            fb = mu.fb_etsi(F, B, fs)
            # print(f'El tiempo de cálculo de los filtros es {(time.time() - fb_time)*1000} ms')
            dct_time = time.time()
            dct = mu.f_base_dct(B)
            # print(f'El tiempo de cálculo de las bases dct es {(time.time() - dct_time)*1000} ms')

            # Media y desviación para la normalización
            file = workspace_dir + 'data/model/fe1_norm1.pkl'
            mu_, std = mu.read_pkl(file)
            mu_ = mu_.astype(np.float32)
            std = std.astype(np.float32)

            # Ventana de hamming
            hamming_win = np.hamming(fs * w)
            
            #-------------------------------------------------------------------------------------#
            
            n_frame = 0     # Counter for frames
            it = 0          # Counter for windows
            buffer_frame = np.zeros(0)                      # Buffer de ventana recibida
            Xfft_windows_list = []
            X_mfcc_buffer = np.empty(64, dtype=np.float32)
            Xb_preallocated = np.empty(32, dtype=np.float32)
            snr_frame_mask = np.ones((512,min_windows))     # Inicializado con la duración de la ventana de inferencia
            yenh = np.zeros(int(fs*60))                      # Inicializado con la duración del audio original


            logging.debug(f'\n|-----------------------------INITIAL PARAMETERS-----------------------------|')
            logging.debug(f'  Frecuencia de muestreo: {cfg["fs"]} Hz')
            frame_size = 0.01  # 10 ms
            frame_samples = int(cfg["frame_length"] * cfg["fs"])  # Muestras por frame
            logging.debug(f'  Tamaño de frame: {frame_samples} samples')
            shift_size = m  # 10 ms
            shift_samples = int(cfg["analysis_window_shift"] * fs)  # Desplazamiento entre frames 160
            logging.debug(f'  Desplazamiento: {shift_samples} samples')
            window_size = w  # 40 ms (640 muestras)
            window_samples = int(cfg["analysis_window_length"] * fs)  # Muestras por ventana 640
            logging.debug(f'  Tamaño de ventana: {window_samples} samples')
            window_inference_min = int((min_windows + (window_size/frame_size)-1) * frame_samples)
            window_inference_max = int((max_windows + (window_size/frame_size)-1) * frame_samples)
            logging.debug(f'  Buffer progresivo: {window_inference_min} - {window_inference_max} samples')
            logging.debug(f'|----------------------------------------------------------------------------|\n')

            #------------------------------PARAMETERS SEND FRAMES---------------------------------#
            samples_per_channel = SAMPLE_RATE * FRAME_DURATION_MS // 1000                       # Calculate samples per frame
            audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, samples_per_channel) # Prepare the audio frame
            audio_data_tx = np.frombuffer(audio_frame.data, dtype=np.int16)                     # Maps the audio frame data to a numpy array for easier manipulation
            #-------------------------------------------------------------------------------------#

            async for event in audio_stream:
                # Check if participant has unpublished the track
                if stop_processing.is_set():
                    logging.info("Stopping audio processing as track is unpublished.")
                    break

                print(f"Processing frame {n_frame}...")
                audio_data_rx = np.frombuffer(event.frame.data, dtype=np.int16)
                wav.writeframes(audio_data_rx) 

                ## PROCESS AUDIO DATA
                # Concatenar el frame recibido al buffer `buffer_frame`
                buffer_frame = np.concatenate([buffer_frame, audio_data_rx])

                if len(buffer_frame) >= window_samples:
                    work_window = buffer_frame[:window_samples]

                    # start_fft = time.time()
                    Xfft = mu.window_fft(work_window, hamming_win, nfft)
                    # accum_fft += (time.time()-start_fft)

                    # start_log = time.time()
                    log_psd_Xfft = mu.log_psd(Xfft)
                    # accum_log += (time.time()-start_log)

                    # start_fb = time.time()
                    fb_windows = mu.frame_fb_mfcc(Xfft, Xb_preallocated, fb, dct, mu_, std, X_mfcc_buffer)
                    # accum_fb_mfcc += (time.time()-start_fb)

                    windows_concat = np.concatenate((log_psd_Xfft,fb_windows))
                    Xfft_windows_list.append(windows_concat)

                    if n_frame-3 < max_windows:   
                        if n_frame-3 >= min_windows:
                            # print("Reached MIN WINDOW --> STRATING INFERENCE")
                            transformed_windows = np.vstack(Xfft_windows_list)

                            # start_prof = time.time()
                            if(n_frame % diezmation_factor == 0):
                                snr_frame_mask = mu.net_eval(net_snr, transformed_windows)
                                snr_frame_mask = snr_frame_mask.T
                                # print(f'Time {n_frame}: {(time.time()-start_prof)*1000} ms')
                            # accum_inf += (time.time()-start_prof)
                    # Si el buffer alcanza o excede las 20 ventanas para hacer la inferencia
                    else:
                        Xfft_windows_list.pop(0)
                        transformed_windows = np.vstack(Xfft_windows_list)

                        # start_prof = time.time()
                        if(n_frame % diezmation_factor == 0):
                            transformed_windows = transformed_windows.reshape(1, max_windows, 576)
                            ort_inputs = {ort_session.get_inputs()[0].name: transformed_windows}
                            snr_frame_mask = ort_session.run(None, ort_inputs)[0]
                            snr_frame_mask = snr_frame_mask.squeeze().T
                            # print(f'Time onnx {n_frame}: {(time.time()-start_prof)*1000} ms')
                        # accum_inf += (time.time()-start_prof)

                    buffer_frame = buffer_frame[shift_samples:]

                    # Aqui haría la evaluacion con la máscara pertinente (para las primeras 3 ventanas sin máscara calculada)
                    # cnt = int(n_frame - w/m) Ajustar al tamaño de la ventana
                    cnt = n_frame - 3
                    # print(f'EVALUATION OF WINDOW {cnt}')

                    x = np.array(work_window, dtype=np.float32) / 2 ** 15   # Audio window to be enhanced

                    xenh, filt = mu.noiseReduction(x, snr_frame_mask[:,-1], fs, window_samples, shift_samples, nfft, gmin)

                    slice_size = min(len(yenh) - cnt * shift_samples, window_samples)

                    yenh[cnt * shift_samples : cnt * shift_samples + slice_size] += xenh[0:slice_size]
  
                    #-------------------------------------------------------------------------------------#
                    # Publish audio frames to the track in room_2 with no delay
                    await asyncio.ensure_future(publish_frames(
                        source, 
                        audio_frame, 
                        audio_data_tx, 
                        ((yenh[cnt * frame_samples : cnt * frame_samples + frame_samples]/3)*2**15).astype(np.int16))
                    )
                n_frame += 1

            logging.info(f"Audio stream processing completed. {n_frame} frames processed.")

        except Exception as e:
            logging.error(f"Error processing audio stream: {e}")
        
        finally:
            # Save the WAV file even if an error occurs
            if yenh is not None:
                try:
                    logging.debug(f'El audio mejorado tiene una longitud de {len(yenh)} y {yenh.dtype}')
                    # for i in range(200):
                    #     logging.debug(f'{i} La señal mejorada es {np.int16((2**15)*yenh[i*160:10+i*160])}')
                    wavfile.write(WAV_ENH, SAMPLE_RATE, np.array((yenh/2)*(2 ** 15), dtype=np.int16))
                    logging.info("Enhanced audio saved successfully.")
                except Exception as save_error:
                    logging.error(f"Failed to save enhanced audio: {save_error}")
            wav.close()
    
    room_id_1 = input("Please enter an id for the receiver room: ")
    token_1 = (
        api.AccessToken('API4bcDob32kABX','fWCQds2YzguBZJbVgdXbPCodqYY0jcHviHqIkwDZ7yV')
        .with_identity("python-model")
        .with_name("Python Model")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                # room="TFM-room",
                room=room_id_1,
            )
        )
        .to_jwt()
    )
    room_id_2 = input("Please enter an id for the sender room: ")
    token_2 = (
        api.AccessToken('API4bcDob32kABX','fWCQds2YzguBZJbVgdXbPCodqYY0jcHviHqIkwDZ7yV')
        .with_identity("python-model")
        .with_name("Python Model")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                # room="TFM-room",
                room=room_id_2,
            )
        )
        .to_jwt()
    )

    url = "wss://test-tfm-3lii83j0.livekit.cloud"   # Livekit project URL   

    logging.info("Connecting to %s", url)
    try:
        logging.info("Connecting to room %s...", room_2.name)
        await room_2.connect(
            url,
            token_2,
            options=rtc.RoomOptions(
                auto_subscribe=False,
            ),
        )
        logging.info("Connected to room %s", room_2.name)

        # Publish a track
        source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
        track = rtc.LocalAudioTrack.create_audio_track("audio_wav", source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        publication = await room_2.local_participant.publish_track(track, options)
        logging.info(f"Track {publication.sid} published by {room_2.local_participant.identity}")

        logging.info("Connecting to room %s...", room_1.name)
        await room_1.connect(
            url,
            token_1,
            options=rtc.RoomOptions(
                auto_subscribe=True,
            ),
        )
        logging.info("Connected to room %s", room_1.name)
    except rtc.ConnectError as e:
        logging.error("Failed to connect to the room: %s", e)
        return
    
    

async def publish_frames(source: rtc.AudioSource, audio_frame:rtc.AudioFrame, audio_data: np.ndarray, frame: np.ndarray):
    """
    Send audio frames through the source
    Args:   
        source: rtc.AudioSource object --> Audio source to publish the audio frames
        audio_frame: rtc.AudioFrame object --> Audio frame to send
        audio_data: np.ndarray --> Audio data mapped to audio frame to send
        frame: np.ndarray --> Frame to send

    """
    #source.clear_queue()

    np.copyto(audio_data, frame)

    # Capture frame to send it to the track
    # logging.info(f"Capturing frame {list(audio_frame.data[:10])}")
    await source.capture_frame(audio_frame)
    # time.sleep(0.001) # Study the effect of delay

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("./logs/model_rx_tx.log"), logging.StreamHandler()],
    )

    loop_1 = asyncio.get_event_loop()
    room_1 = rtc.Room(loop=loop_1)
    loop_2 = asyncio.get_event_loop()
    room_2 = rtc.Room(loop=loop_2)

    async def cleanup_1():
        await room_1.disconnect()
        loop_1.stop()
    
    async def cleanup_2():
        await room_2.disconnect()
        loop_2.stop()

    asyncio.ensure_future(main(room_1, room_2))
    for signal in [SIGINT, SIGTERM]:
        loop_1.add_signal_handler(signal, lambda: asyncio.ensure_future(cleanup_1()))
        loop_2.add_signal_handler(signal, lambda: asyncio.ensure_future(cleanup_2()))

    try:
        loop_1.run_forever()
        loop_2.run_forever()
    finally:
        loop_1.close()
        loop_2.close()