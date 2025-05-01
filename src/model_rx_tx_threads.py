import os
import sys
import wave
import time
import asyncio
import queue
import logging
import numpy as np
from scipy.io import wavfile
from livekit import rtc, api
from signal import SIGINT, SIGTERM

import model_utils as mu

SAMPLE_RATE = 8000
FRAME_DURATION_MS = 10
NUM_CHANNELS = 1
FORMAT = 2 # 16-bit PCM
WAV_FILE = "BTS/TFM/audios/livekit/audio_received_lk.wav"
WAV_ENH = "BTS/TFM/audios/livekit/audio_enhanced_lk.wav"

frame_queue = asyncio.Queue()

# Load model dimensions and weights
# 8k Net trained
workspace_dir = '/home/adelval/BTS/TFM/afterburner8k_win20/'

# 16k Net trained
# workspace_dir = '/home/adelval/BTS/TFM/test/'
input_dim, output_dim = mu.load_obj(workspace_dir + 'data/model/dimensions.pkl') 

print('  input_dim: %s' % str(input_dim))
print('  output_dim: %s' % str(output_dim))

sys.path.append( workspace_dir + 'src/net')

from net_snr import Net_snr
# from net_snr_original import Net_snr

net_snr = Net_snr(input_dim, output_dim, cuda=True)
net_snr.load_theta( workspace_dir + 'data/model/theta_last_rn')

# Function to send audio frames as a coroutine
async def start_sending_frames(source, audio_frame, audio_data, frame_queue, send_rate=0.005, loop_thread=None):
    """
    Continuously send frames from the frame queue at a constant rate.
    Starts only when there are at least 500 elements in the queue.
    Args:
        source: rtc.AudioSource object --> Audio source to publish the audio frames
        audio_frame: rtc.AudioFrame object --> Audio frame to send
        audio_data: np.ndarray --> Audio data mapped to audio frame to send
        frame_queue: queue.Queue --> Queue holding enhanced frames
        send_rate: float --> Frame sending interval in seconds
    """
    logging.info("Thread to send frames started.")

    while True:
        print(f"Queue size {frame_queue.qsize()}")
        if frame_queue.qsize() >= 500: 
            while True:
                try:
                    frame = await frame_queue.get()  # Wait for a frame
                    np.copyto(audio_data, frame)
                    await source.capture_frame(audio_frame)
                    logging.info(f"Enhanced frame consumed {frame_queue.qsize()}")
                    logging.info(f"Sending frame at {time.time()}")
                    await asyncio.sleep(send_rate)  # Maintain sending rate
                except Exception as e:
                    print(f"Error during frame sending: {e}")
        # To yield control back to the event loop, allowing other tasks to run.
        await asyncio.sleep(0)
        

# Initializate wav file
def setup_wav_file():
    os.makedirs(os.path.dirname(WAV_FILE), exist_ok=True)
    wav_file = wave.open(WAV_FILE, 'wb')
    wav_file.setnchannels(NUM_CHANNELS)
    wav_file.setsampwidth(FORMAT)
    wav_file.setframerate(SAMPLE_RATE)
    return wav_file

async def main(room_1: rtc.Room, room_2: rtc.Room, loop_3) -> None:
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
        print("Audio processing task started")


        #------------------------------PARAMETERS SEND FRAMES---------------------------------#
        samples_per_channel = SAMPLE_RATE * FRAME_DURATION_MS // 1000                           # Calculate samples per frame
        audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, samples_per_channel)     # Prepare the audio frame
        audio_data_tx = np.frombuffer(audio_frame.data, dtype=np.int16)                         # Maps the audio frame data to a numpy array for easier manipulation
        #-------------------------------------------------------------------------------------#
        asyncio.create_task(start_sending_frames(source, audio_frame, audio_data_tx, frame_queue))
        print("Sending frames task started")
        
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
            n_frame = 0     # Counter for frames
            it = 0          # Counter for windows

            fs = SAMPLE_RATE
            B=[32]
            w=[0.040]
            m=0.010
            nfft=[1024]
            gmin = 0.0562
            diezmation_factor = 1

            frame_size = 0.01       # ms
            frame_samples = int(frame_size * fs)  # Muestras por frame
            logging.debug(f'Un frame tiene una duración de {frame_samples} samples')
            shift_size = m          # ms
            shift_samples = int(shift_size * fs)  # Desplazamiento entre frames 160
            logging.debug(f'Desplazamiento de {shift_samples} samples')
            window_size = w[0]      # 40 ms (640 muestras)
            window_samples = int(window_size * fs)  # Muestras por ventana 640
            logging.debug(f'La ventana tiene una duración de {window_samples} samples')
            min_windows = 4
            max_windows = 20
            window_inference_min = int((min_windows + (window_size/frame_size)-1) * frame_samples)
            window_inference_max = int((max_windows + (window_size/frame_size)-1) * frame_samples)
            logging.debug(f'El buffer retendra hasta {window_inference_max} samples')
            #-------------------------------------------------------------------------------------#


            buffer_frame = np.zeros(0)                      # Buffer de ventana recibida

            snr_frame_mask = np.ones((512,min_windows))     # Inicializado con la duración de la ventana de inferencia
            yenh = np.zeros(640000)                         # Inicializado con la duración del audio original


            async for event in audio_stream:
                # Check if participant has unpublished the track
                if stop_processing.is_set():
                    logging.info("Stopping audio processing as track is unpublished.")
                    break

                n_frame += 1
                audio_data_rx = np.frombuffer(event.frame.data, dtype=np.int16)
                wav.writeframes(audio_data_rx) 

                ## PROCESS AUDIO DATA
                # Concatenar el frame recibido al buffer `buffer_frame`
                buffer_frame = np.concatenate([buffer_frame, audio_data_rx])

                if len(buffer_frame) >= window_samples:

                    if len(buffer_frame) < window_inference_max:
                        work_window = buffer_frame[it*shift_samples:it*shift_samples+window_samples]
                        logging.debug(f"Frame number: {n_frame} y buffer len {int(len(buffer_frame)/frame_samples)}")
                        it += 1                                                           # Number of windows received
                        if len(buffer_frame) >= window_inference_min:
                            logging.debug("Reached MIN WINDOW --> STRATING INFERENCE")
                            work_inf_frames = buffer_frame[:window_inference_max]
                            fft_windows = mu.frame_fft(work_inf_frames, fs, w, m, nfft, it)
                            fft_windows_log = mu.log_scale(fft_windows)
                            fb_windows = mu.frame_fb_mfcc(work_inf_frames, fs, B, w, m, nfft, it)
                            fb_windows_norm = mu.norm_fb_frame(fb_windows)
                            windows_concat = np.concatenate( (fft_windows_log,fb_windows_norm), 1 )
                            # Avoid blocking the event loop by creating a paralell thread
                            loop = asyncio.get_event_loop()
                            snr_frame_mask = await loop.run_in_executor(None, mu.net_eval, windows_concat, net_snr)
                            snr_frame_mask = snr_frame_mask.T
                            logging.debug(f'Frames restantes en el buffer {len(buffer_frame)/frame_samples}')

                    # Si el buffer alcanza o excede las 20 ventanas para hacer la inferencia
                    else:
                        work_window = buffer_frame[window_inference_max-window_samples:] # Los últimos frames del buffer
                        logging.debug("Reached MAX WINDOW --> Starting Inference")
                        work_inf_frames = buffer_frame[:window_inference_max]
                        fft_windows = mu.frame_fft(work_inf_frames, fs, w, m, nfft, max_windows)
                        fft_windows_log = mu.log_scale(fft_windows)
                        fb_windows = mu.frame_fb_mfcc(work_inf_frames, fs, B, w, m, nfft, max_windows)
                        fb_windows_norm = mu.norm_fb_frame(fb_windows)
                        windows_concat = np.concatenate( (fft_windows_log,fb_windows_norm), 1 )
                        # Avoid blocking the event loop by creating a paralell thread
                        if(n_frame % diezmation_factor == 0):
                            loop = asyncio.get_event_loop()
                            snr_frame_mask = await loop.run_in_executor(None, mu.net_eval, windows_concat, net_snr)
                            snr_frame_mask = snr_frame_mask.T
                        # Desplazar las muestras en `buffer_frame` para la próxima ventana
                        buffer_frame = buffer_frame[shift_samples:]
                        logging.debug(f'Frames restantes en el buffer {len(buffer_frame)/frame_samples}')

                    #---------------------------------EVALUATION OF WINDOW---------------------------------#
                    cnt = n_frame - 3 # Automatize
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
                    # logging.info(f'Frame {cnt} processed')
                    #-------------------------------------------------------------------------------------#

                    # Put audio enhanced frames in the shared queue
                    await frame_queue.put((np.clip(yenh[cnt*frame_samples : cnt*frame_samples+frame_samples]*2**15, -32768, 32767)).astype(np.int16))
                    logging.info(f"Enhanced frame produced {frame_queue.qsize()}")
                    await asyncio.sleep(0)

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
                    wavfile.write(WAV_ENH, SAMPLE_RATE, np.clip(yenh*2**15, -32768, 32767).astype(np.int16))
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
        handlers=[logging.FileHandler("BTS/TFM/logs/consumer_wave.log"), logging.StreamHandler()],
    )

    loop_1 = asyncio.get_event_loop()
    room_1 = rtc.Room(loop=loop_1)
    loop_2 = asyncio.get_event_loop()
    room_2 = rtc.Room(loop=loop_2)
    
    loop_3 = asyncio.get_event_loop()

    async def cleanup_1():
        await room_1.disconnect()
        loop_1.stop()
    
    async def cleanup_2():
        await room_2.disconnect()
        loop_2.stop()

    asyncio.ensure_future(main(room_1, room_2, loop_3))
    for signal in [SIGINT, SIGTERM]:
        loop_1.add_signal_handler(signal, lambda: asyncio.ensure_future(cleanup_1()))
        loop_2.add_signal_handler(signal, lambda: asyncio.ensure_future(cleanup_2()))

    try:
        loop_1.run_forever()
        loop_2.run_forever()
        loop_3.run_forever()
    finally:
        loop_1.close()
        loop_2.close()
        loop_3.close()