import asyncio
import logging
from signal import SIGINT, SIGTERM
import os
import wave
import sys
from scipy.io import wavfile

import numpy as np
from livekit import rtc, api

import model_utils as mu

SAMPLE_RATE = 16000
NUM_CHANNELS = 1
FORMAT = 2 # 16-bit PCM
WAV_FILE = "BTS/TFM/audios/livekit/audio_received.wav"
WAV_ENH = "BTS/TFM/audios/livekit/audio_enhanced_.wav"
WAV_ENH_NORM = "BTS/TFM/audios/livekit/audio_enhanced_norm.wav"

# Load model dimensions and weights

# 8k Net trained
# workspace_dir = '/home/adelval/BTS/TFM/afterburner8k/'

# 16k Net trained
workspace_dir = '/home/adelval/BTS/TFM/test/'
input_dim, output_dim = mu.load_obj(workspace_dir + 'data/model/dimensions.pkl') 

print('  input_dim: %s' % str(input_dim))
print('  output_dim: %s' % str(output_dim))

sys.path.append( workspace_dir + 'src/net')

from net_snr import Net_snr
# from net_snr_original import Net_snr

net_snr = Net_snr(input_dim, output_dim, cuda=True)
net_snr.load_theta( workspace_dir + 'data/model/theta_last')

# Initializate wav file
def setup_wav_file():
    # Crear el directorio si no existe
    os.makedirs(os.path.dirname(WAV_FILE), exist_ok=True)
    wav_file = wave.open(WAV_FILE, 'wb')
    wav_file.setnchannels(NUM_CHANNELS)
    wav_file.setsampwidth(FORMAT)
    wav_file.setframerate(SAMPLE_RATE)
    return wav_file

async def main(room: rtc.Room) -> None:
    wav = setup_wav_file()
    stop_processing = asyncio.Event()  # Event to signal when to stop processing

    @room.on("participant_disconnected")
    def on_participant_disconnect(participant: rtc.Participant, *_):
        logging.info("participant disconnected: %s", participant.identity)

    @room.on("track_subscribed")
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
        asyncio.create_task(process_audio_stream(_audio_stream))
        
    @room.on("track_unpublished")
    def on_track_unpublished(
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logging.info("Track unpublished: %s", publication.sid)
        stop_processing.set()  # Signal to stop audio processing

    async def process_audio_stream(audio_stream):
        try:

            ## Parameters for the SE model ##
            #-------------------------------------------------------------------------------------#
            n_frame = 0     # Counter for frames
            it = 0          # Counter for windows

            fs = SAMPLE_RATE
            B=[32]
            w=[0.040]
            m=0.010
            nfft=[1024]
            gmin = 0.0562

            frame_size = 0.01  # 10 ms
            frame_samples = int(frame_size * fs)  # Muestras por frame
            logging.debug(f'Un frame tiene una duración de {frame_samples} samples')
            shift_size = m  # 10 ms
            shift_samples = int(shift_size * fs)  # Desplazamiento entre frames 160
            logging.debug(f'Desplazamiento de {shift_samples} samples')
            window_size = w[0]  # 40 ms (640 muestras)
            window_samples = int(window_size * fs)  # Muestras por ventana 640
            logging.debug(f'La ventana tiene una duración de {window_samples} samples')
            min_windows = 4
            max_windows = 20
            window_inference_min = int((min_windows + (window_size/frame_size)-1) * frame_samples)
            window_inference_max = int((max_windows + (window_size/frame_size)-1) * frame_samples)
            logging.debug(f'El buffer retendra hasta {window_inference_max} samples')
            #-------------------------------------------------------------------------------------#
            
            buffer_frame = np.zeros(0)  # Buffer de ventana recibida

            snr_frame_mask = np.ones((512,min_windows)) # Inicializado con la duración de la ventana de inferencia
            yenh = np.zeros(100000) # Inicializado con la duración del audio original

            async for event in audio_stream:

                # Check if participant has unpublished the track
                if stop_processing.is_set():
                    logging.info("Stopping audio processing as track is unpublished.")
                    break

                n_frame += 1
                audio_data = np.frombuffer(event.frame.data, dtype=np.int16) # Receive samples of 160
                logging.debug(f'{n_frame-1} Se reciben estos frames {list(event.frame.data[:10])}')
                # logging.debug(f'{n_frame-1} Se reciben estos frames {audio_data[:10]}')
                wav.writeframes(audio_data)  # Save to WAV to compare
                
                ## PROCESS AUDIO DATA
                # Concatenar el frame recibido al buffer `buffer_frame`
                buffer_frame = np.concatenate([buffer_frame, audio_data])

                if len(buffer_frame) >= window_samples:
                    # Extraer las primeras 640 muestras como ventana completa
                    if len(buffer_frame) < window_inference_max:
                        work_window = buffer_frame[it*shift_samples:it*shift_samples+window_samples]
                        logging.debug(f"Frame number: {n_frame} y buffer len {int(len(buffer_frame)/frame_samples)}")
                        logging.debug(f"Ventana acumulada {work_window[:10]}")
                        logging.debug(f"Ventana acumulada {work_window[160:170]}")
                        logging.debug(f"Ventana acumulada {work_window[320:330]}")
                        logging.debug(f"Ventana acumulada {work_window[480:490]}")
                        it += 1 # Number of windows received
                        if len(buffer_frame) >= window_inference_min:
                            logging.debug("Reached MIN WINDOW --> STRATING INFERENCE")
                            work_inf_frames = buffer_frame[:window_inference_max]
                            logging.debug(f'0 frame  {work_inf_frames[:10]}')
                            logging.debug(f'1 frame  {work_inf_frames[frame_samples:frame_samples+10]}')
                            logging.debug(f'2 frame  {work_inf_frames[2*frame_samples:2*frame_samples+10]}')
                            logging.debug(f'3 frame  {work_inf_frames[3*frame_samples:3*frame_samples+10]}')
                            fft_windows = mu.frame_fft(work_inf_frames, fs, w, m, nfft, it)
                            logging.debug(f'0 Vector 2D con FFT para cada frame por row  {fft_windows[0,:10]}')
                            logging.debug(f'1 Vector 2D con FFT para cada frame por row  {fft_windows[1,:10]}')
                            logging.debug(f'2 Vector 2D con FFT para cada frame por row  {fft_windows[2,:10]}')
                            logging.debug(f'3 Vector 2D con FFT para cada frame por row  {fft_windows[3,:10]}')
                            fft_windows_log = mu.log_scale(fft_windows)
                            logging.debug(f'0 Vector 2D con FFT para cada frame por row  {fft_windows_log[0,:10]}')
                            logging.debug(f'1 Vector 2D con FFT para cada frame por row  {fft_windows_log[1,:10]}')
                            logging.debug(f'2 Vector 2D con FFT para cada frame por row  {fft_windows_log[2,:10]}')
                            logging.debug(f'3 Vector 2D con FFT para cada frame por row  {fft_windows_log[3,:10]}')
                            fb_windows = mu.frame_fb_mfcc(work_inf_frames, fs, B, w, m, nfft, it)
                            logging.debug(f'0 Filtro Mel {fb_windows[0,:5]}')
                            logging.debug(f'1 Filtro Mel {fb_windows[1,:5]}')
                            logging.debug(f'2 Filtro Mel {fb_windows[2,:5]}')
                            logging.debug(f'3 Filtro Mel {fb_windows[3,:5]}')
                            fb_windows_norm = mu.norm_fb_frame(fb_windows)
                            logging.debug(f'0 Filtro Mel normalizado {fb_windows_norm[0,:5]}')
                            logging.debug(f'1 Filtro Mel normalizado {fb_windows_norm[1,:5]}')
                            logging.debug(f'2 Filtro Mel normalizado {fb_windows_norm[2,:5]}')
                            logging.debug(f'3 Filtro Mel normalizado {fb_windows_norm[3,:5]}')
                            windows_concat = np.concatenate( (fft_windows_log,fb_windows_norm), 1 )
                            logging.debug(f'0 La concatenacion resulta  {windows_concat[0,:5]}  y {windows_concat[0,512:517]}')
                            logging.debug(f'1 La concatenacion resulta  {windows_concat[1,:5]}  y {windows_concat[1,512:517]}')
                            logging.debug(f'2 La concatenacion resulta  {windows_concat[2,:5]}  y {windows_concat[2,512:517]}')
                            logging.debug(f'3 La concatenacion resulta  {windows_concat[3,:5]}  y {windows_concat[3,512:517]}')
                            snr_frame_mask = mu.net_eval(windows_concat, net_snr)
                            logging.debug(snr_frame_mask.shape)
                            logging.debug(snr_frame_mask[0,:10])
                            logging.debug(snr_frame_mask[1,:10])
                            logging.debug(snr_frame_mask[2,:10])
                            logging.debug(snr_frame_mask[3,:10])
                            snr_frame_mask = snr_frame_mask.T
                            logging.debug(f'La máscara {it} calculada es de {snr_frame_mask.shape}')
                            logging.debug(f'Frames restantes en el buffer {len(buffer_frame)/frame_samples}')

                    # Si el buffer alcanza o excede las 20 ventanas para hacer la inferencia
                    else:
                        work_window = buffer_frame[window_inference_max-window_samples:] # Los últimos frames del buffer
                        logging.debug(f"Frame number: {n_frame}")
                        logging.debug(f"Ventana acumulada {work_window[:10]}")
                        logging.debug(f"Ventana acumulada {work_window[160:170]}")
                        logging.debug(f"Ventana acumulada {work_window[320:330]}")
                        logging.debug(f"Ventana acumulada {work_window[480:490]}")
                        logging.debug("Reached MAX WINDOW --> Starting Inference")
                        work_inf_frames = buffer_frame[:window_inference_max]
                        logging.debug(f'0 frame  {work_inf_frames[:10]}')
                        logging.debug(f'1 frame  {work_inf_frames[frame_samples:frame_samples+10]}')
                        logging.debug(f'2 frame  {work_inf_frames[2*frame_samples:2*frame_samples+10]}')
                        logging.debug(f'3 frame  {work_inf_frames[3*frame_samples:3*frame_samples+10]}')
                        logging.debug(f'23 frame {work_inf_frames[22*frame_samples:22*frame_samples+10]}')
                        fft_windows = mu.frame_fft(work_inf_frames, fs, w, m, nfft, max_windows)
                        logging.debug(f'0 Vector 2D con FFT para cada frame por row  {fft_windows[0,:10]}')
                        logging.debug(f'1 Vector 2D con FFT para cada frame por row  {fft_windows[1,:10]}')
                        logging.debug(f'2 Vector 2D con FFT para cada frame por row  {fft_windows[2,:10]}')
                        logging.debug(f'3 Vector 2D con FFT para cada frame por row  {fft_windows[3,:10]}')
                        logging.debug(f'20 Vector 2D con FFT para cada frame por row {fft_windows[19,:10]}')
                        fft_windows_log = mu.log_scale(fft_windows)
                        logging.debug(f'0 Vector 2D con FFT para cada frame por row  {fft_windows_log[0,:10]}')
                        logging.debug(f'1 Vector 2D con FFT para cada frame por row  {fft_windows_log[1,:10]}')
                        logging.debug(f'2 Vector 2D con FFT para cada frame por row  {fft_windows_log[2,:10]}')
                        logging.debug(f'3 Vector 2D con FFT para cada frame por row  {fft_windows_log[3,:10]}')
                        logging.debug(f'20 Vector 2D con FFT para cada frame por row {fft_windows_log[19,:10]}')
                        fb_windows = mu.frame_fb_mfcc(work_inf_frames, fs, B, w, m, nfft, max_windows)
                        logging.debug(f'0 Filtro Mel {fb_windows[0,:5]}')
                        logging.debug(f'1 Filtro Mel {fb_windows[1,:5]}')
                        logging.debug(f'2 Filtro Mel {fb_windows[2,:5]}')
                        logging.debug(f'3 Filtro Mel {fb_windows[3,:5]}')
                        logging.debug(f'19 Filtro Mel {fb_windows[19,:5]}')
                        fb_windows_norm = mu.norm_fb_frame(fb_windows)
                        logging.debug(f'0 Filtro Mel normalizado {fb_windows_norm[0,:5]}')
                        logging.debug(f'1 Filtro Mel normalizado {fb_windows_norm[1,:5]}')
                        logging.debug(f'2 Filtro Mel normalizado {fb_windows_norm[2,:5]}')
                        logging.debug(f'3 Filtro Mel normalizado {fb_windows_norm[3,:5]}')
                        logging.debug(f'19 Filtro Melnormalizado {fb_windows_norm[19,:5]}')
                        windows_concat = np.concatenate( (fft_windows_log,fb_windows_norm), 1 )
                        logging.debug(f'0 La concatenacion resulta  {windows_concat[0,:5]}  y {windows_concat[0,512:517]}')
                        logging.debug(f'1 La concatenacion resulta  {windows_concat[1,:5]}  y {windows_concat[1,512:517]}')
                        logging.debug(f'2 La concatenacion resulta  {windows_concat[2,:5]}  y {windows_concat[2,512:517]}')
                        logging.debug(f'3 La concatenacion resulta  {windows_concat[3,:5]}  y {windows_concat[3,512:517]}')
                        logging.debug(f'19 La concatenacion resulta {windows_concat[19,:5]} y {windows_concat[19,512:517]}')

                        snr_frame_mask = mu.net_eval(windows_concat, net_snr)
                        logging.debug(snr_frame_mask.shape)
                        logging.debug(snr_frame_mask[0,:10])
                        logging.debug(snr_frame_mask[1,:10])
                        logging.debug(snr_frame_mask[2,:10])
                        logging.debug(snr_frame_mask[3,:10])
                        logging.debug(snr_frame_mask[19,:10])
                        snr_frame_mask = snr_frame_mask.T
                        #Desplazar las muestras en `buffer_frame` para la próxima ventana
                        buffer_frame = buffer_frame[shift_samples:]
                        logging.debug(f'Frames restantes en el buffer {len(buffer_frame)/frame_samples}')

                    # Aqui haría la evaluacion con la máscara pertinente (para las primeras 3 ventanas sin máscara calculada)
                    cnt = n_frame - 3
                    logging.debug(f'EVALUATION OF WINDOW {cnt}')
                    x = np.array(work_window, dtype=np.float32) / 2 ** 15 # 0.04 * fs = 640 samples
                    logging.debug(f'El frame sin enventanado resulta {x[:10]}')
                    logging.debug(f'Se le aplica la mascara {snr_frame_mask[:10,-1]}')
                    xenh, filt = mu.noiseReduction(x, snr_frame_mask[:,-1], fs, window_samples, shift_samples, nfft[0], gmin)
                    logging.debug(f'{n_frame} Frames en xenh {xenh[:10]}')
                    logging.debug(f'Las dimensiones del filtro son {filt.shape}')
                    logging.debug(f'Window processed {cnt} {xenh.shape} {yenh[cnt*shift_samples : cnt*shift_samples+window_samples].shape}')
                    slice_size = min(len(yenh) - cnt * shift_samples, window_samples)
                    if slice_size < 0:
                        slice_size = 0
                    yenh[cnt * shift_samples : cnt * shift_samples + slice_size] += xenh[0:slice_size]
                    logging.debug(f'{n_frame} Frames en yenh {yenh[cnt * shift_samples : cnt * shift_samples + 10]}')
            
            max_amplitude = np.max(np.abs(yenh))
            if max_amplitude > 0:
                yenh_norm = (yenh / max_amplitude)

            wavfile.write(WAV_ENH_NORM,fs,yenh_norm)

            logging.info(f"Audio stream processing completed. {n_frame} frames processed.")

        except Exception as e:
            logging.error(f"Error processing audio stream: {e}")
        
        finally:
            # Save the WAV file even if an error occurs
            if yenh is not None:
                try:
                    logging.debug(f'El audio mejorado tiene una longitud de {len(yenh)} y {yenh.dtype}')
                    for i in range(200):
                        logging.debug(f'{i} La señal mejorada es {np.int16((2**15)*yenh[i*160:10+i*160])}')
                    wavfile.write(WAV_ENH, SAMPLE_RATE, yenh)
                    logging.info("Enhanced audio saved successfully.")
                except Exception as save_error:
                    logging.error(f"Failed to save enhanced audio: {save_error}")
    

    token = (
        api.AccessToken('API4bcDob32kABX','fWCQds2YzguBZJbVgdXbPCodqYY0jcHviHqIkwDZ7yV')
        .with_identity("python-model")
        .with_name("Python Model")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room="TFM-room",
            )
        )
        .to_jwt()
    )
    url = "wss://test-tfm-3lii83j0.livekit.cloud"

    logging.info("connecting to %s", url)
    try:
        await room.connect(
            url,
            token,
            options=rtc.RoomOptions(
                auto_subscribe=True,
            ),
        )
        logging.info("connected to room %s", room.name)
    except rtc.ConnectError as e:
        logging.error("failed to connect to the room: %s", e)
        return


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("BTS/TFM/logs/model.log"), logging.StreamHandler()],
    )

    loop = asyncio.get_event_loop()
    room = rtc.Room(loop=loop)

    async def cleanup():
        await room.disconnect()
        loop.stop()

    asyncio.ensure_future(main(room))
    for signal in [SIGINT, SIGTERM]:
        loop.add_signal_handler(signal, lambda: asyncio.ensure_future(cleanup()))

    try:
        loop.run_forever()
    finally:
        loop.close()