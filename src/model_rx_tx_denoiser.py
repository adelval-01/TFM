"""
LiveKit Dual-Room Audio Processing

Description:
    This script integrates with LiveKit to facilitate real-time audio processing between two rooms.
    - **Room 1**: Waits for a participant to connect and publish an audio track.
    - When an audio track is detected, a speech processing module is triggered to clean the audio (noise reduction).
    - **Room 2**: The cleaned audio is published as a new track.

    This implementation ensures high-quality, real-time audio enhancement from participant in Room 1 for participants in Room 2.

Author: Aitor del Val Allueva
Date: 2025-03-14  
Version: 1.0  

Requirements:
    - Python 3.9
    - livekit SDK
    - Speech processing library (e.g., noise reduction, filtering)

Usage:
    Run the script and ensure proper LiveKit credentials are configured.
"""

import os
import wave
import time
import asyncio
import logging
import numpy as np
from scipy.io import wavfile
from livekit import rtc, api
from signal import SIGINT, SIGTERM

import torch
from denoiser import pretrained
from denoiser.dsp import convert_audio

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 10
NUM_CHANNELS = 1
FORMAT = 2 # 16-bit PCM
WAV_FILE = "BTS/TFM/audios/livekit/audio_received_denoiser.wav"
WAV_ENH = "BTS/TFM/audios/livekit/audio_enhanced_denoiser.wav"

# Initializate wav file
def setup_wav_file():
    os.makedirs(os.path.dirname(WAV_FILE), exist_ok=True)
    wav_file = wave.open(WAV_FILE, 'wb')
    wav_file.setnchannels(NUM_CHANNELS)
    wav_file.setsampwidth(FORMAT)
    wav_file.setframerate(SAMPLE_RATE)
    return wav_file

async def publish_track(room: rtc.Room):
    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("callback", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    publication = await room.local_participant.publish_track(track, options)
    logging.info(f"Callback track {publication.sid} published by {room.local_participant.identity}")


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
    call_sid = None

    @room_1.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        logging.info(
            "participant connected: %s %s %s %s", participant.sid, participant.identity, participant.metadata, participant.name
        )
        #-------------------------- ANSWER THE CALL ---------------------------------#
        # Publish a track
        asyncio.create_task(publish_track(room_1))
        #-----------------------------------------------------------------------------#

    @room_1.on("participant_disconnected")
    def on_participant_disconnect(participant: rtc.Participant, *_):
        logging.info("participant disconnected: %s", participant.identity)

    @room_1.on("track_published")
    def on_track_published(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        publication.set_subscribed(True)


    @room_1.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        global call_sid
        call_sid = publication.sid
        logging.info("track subscribed: %s", call_sid)
        if (track.kind == rtc.TrackKind.KIND_AUDIO):
            print("Subscribed to an Audio Track")
            _audio_stream = rtc.AudioStream(track,sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS)
            # audio_stream is an async iterator that yields AudioFrame

        # Start an async task to handle the audio frames
        asyncio.create_task(process_audio_stream(_audio_stream, source))
        
    @room_1.on("track_unpublished")
    def on_track_unpublished(
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        global call_sid
        logging.info("Track unpublished: %s", publication.sid)
        if publication.sid == call_sid:
            stop_processing.set()                       # Signal to stop audio processing

    async def process_audio_stream(audio_stream, source):
        """
        Process audio frames from the audio stream and send them to the source
        Args:
            audio_stream: rtc.AudioStream object --> Audio stream from the subscribed track
            source: rtc.AudioSource object --> Audio source to publish the audio frames
        """

        try:
            it = 0

            #------------------------------PARAMETERS SEND FRAMES---------------------------------#
            samples_per_channel = SAMPLE_RATE * FRAME_DURATION_MS // 1000                         # Calculate samples per frame
            audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, samples_per_channel)   # Prepare the audio frame
            audio_data_tx = np.frombuffer(audio_frame.data, dtype=np.int16)                       # Maps the audio frame data to a numpy array for easier manipulation
            #-------------------------------------------------------------------------------------#

            #------------------------------BUFFERING PARAMETERS-----------------------------------#
            frame_size = 0.01   # s
            frame_samples = int(frame_size * SAMPLE_RATE)  # Samples per frame 
            shift_size = 0.01   # s
            shift_samples = int(shift_size * SAMPLE_RATE)  # Windows shifting
            logging.debug(f'Desplazamiento de {shift_samples} samples')
            window_size = 0.04  # s
            window_samples = int(window_size * SAMPLE_RATE)  # Samples per window 640
            logging.debug(f'La ventana tiene una duración de {window_samples} samples')

            buffer_frame = np.zeros(0, dtype=np.float32)
            denoised_signal = np.zeros(0)
            denoised_signal = np.zeros(64000000, dtype=np.float32)

            # window = torch.hann_window(window_samples, periodic=False, dtype=torch.float32)
            window = np.hamming(window_samples)
            #-------------------------------------------------------------------------------------#

            #-------------------------------- DENOISER -------------------------------------------#
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f'Using device: {device}')
            model = pretrained.dns48().to(device)
            #-------------------------------------------------------------------------------------#
            
            accum_time = 0

            async for event in audio_stream:
                # Check if we should stop processing
                if stop_processing.is_set():
                    logging.info("Stopping audio processing as track is unpublished.")
                    break
                
                audio_data_rx = np.frombuffer(event.frame.data, dtype=np.int16)
                wav.writeframes(audio_data_rx) 


                #-------------------------------- PROCESS AUDIO ----------------------------------------#
                
                buffer_frame = np.concatenate([buffer_frame, audio_data_rx])


                if len(buffer_frame) >= window_samples:
                    work_window = buffer_frame[:window_samples]
                    # print(work_window.shape, work_window.dtype)
                    
                    
                    with torch.no_grad():
                        # work_window = convert_audio(work_window.to(device), SAMPLE_RATE, model.sample_rate, model.chin)
                        work_window_tensor = torch.tensor(work_window[None]).to(device)
                        # print(work_window_tensor.shape, work_window_tensor.dtype)
                        start_time = time.time()
                        denoised_window = model(torch.tensor(work_window[None]).to(device))[0]
                        # print(f'Tiempo de procesamiento de la ventana {time.time() - start_time}')
                        denoised_window = denoised_window.squeeze(0)
                        accum_time += time.time() - start_time
                        # print(f'Denoised signal {denoised_window.shape} {denoised_window.dtype}')
                    # denoised_signal = np.concatenate([denoised_signal, denoised_window.data[:shift_samples].cpu().numpy()])
                    # print(denoised_signal.shape, denoised_signal.dtype)
                    denoised_signal[it*shift_samples:it*shift_samples+window_samples] += denoised_window.data.cpu().numpy() * window

                    buffer_frame = buffer_frame[shift_samples:]
                    # print(buffer_frame.shape, buffer_frame.dtype)
                    
                    denoised_frame = denoised_signal[it*shift_samples:it*shift_samples+shift_samples].astype(np.int16)
                    # print(denoised_frame.shape, denoised_frame.dtype)
                    it += 1

                    # Publish audio frames to the track in room_2
                    await asyncio.ensure_future(publish_frames(source, audio_frame, audio_data_tx, denoised_frame))
                    # if(i>=2):
                    #     delay = time.time() - past_time
                    #     print(f"Frame {it} at {delay}")
                    # past_time = time.time()

                #---------------------------------------------------------------------------------------#
            
            print(f"Total frames are {it} and the average time per window is {(accum_time/it*1000):.4f} ms")
            logging.info("Audio stream processing completed.")

        except Exception as e:
            logging.error(f"Error processing audio stream: {e}")

        finally:
            # Save the WAV file even if an error occurs
            if denoised_signal is not None:
                try:
                    logging.debug(f'El audio mejorado tiene una longitud de {len(denoised_signal)} y {denoised_signal.dtype}')
                    # for i in range(200):
                    #     logging.debug(f'{i} La señal mejorada es {np.int16((2**15)*yenh[i*160:10+i*160])}')
                    wavfile.write(WAV_ENH, SAMPLE_RATE, denoised_signal.astype(np.int16))
                    logging.info("Enhanced audio saved successfully.")
                except Exception as save_error:
                    logging.error(f"Failed to save enhanced audio: {save_error}")
            wav.close()
            # exit(0)
    
    room_id_1 = input("Please enter an id for the receiver room: ")
    token_1 = (
        api.AccessToken('API4bcDob32kABX','fWCQds2YzguBZJbVgdXbPCodqYY0jcHviHqIkwDZ7yV')
        .with_identity("python-consumer")
        .with_name("Python Consumer")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_id_1,
            )
        )
        .to_jwt()
    )
    room_id_2 = input("Please enter an id for the sender room: ")
    token_2 = (
        api.AccessToken('API4bcDob32kABX','fWCQds2YzguBZJbVgdXbPCodqYY0jcHviHqIkwDZ7yV')
        .with_identity("python-consumer")
        .with_name("Python Consumer")
        .with_grants(
            api.VideoGrants(
                room_join=True,
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
                auto_subscribe=False,
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
    # time.sleep(0.008) # Study the effect of delay

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("BTS/TFM/logs/consumer_wave.log"), logging.StreamHandler()],
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