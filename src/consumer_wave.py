import asyncio
import logging
from signal import SIGINT, SIGTERM
import os

import wave
import pyaudio


import numpy as np
from livekit import rtc, api

SAMPLE_RATE = 16000
NUM_CHANNELS = 1
FORMAT = pyaudio.paInt16

# Initializate wav file
def setup_wav_file():
    # Crear el directorio si no existe
    wav_file = "audios/audio_received.wav"
    os.makedirs(os.path.dirname(wav_file), exist_ok=True)
    wav_file = wave.open(wav_file, 'wb')
    wav_file.setnchannels(NUM_CHANNELS)
    wav_file.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wav_file.setframerate(SAMPLE_RATE)
    return wav_file

async def main(room: rtc.Room) -> None:
    wav = setup_wav_file()

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
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            _video_stream = rtc.VideoStream(track)
            # video_stream is an async iterator that yields VideoFrame
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            print("Subscribed to an Audio Track")
            _audio_stream = rtc.AudioStream(track,sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS)
            # audio_stream is an async iterator that yields AudioFrame

        # Start an async task to handle the audio frames
        asyncio.create_task(process_audio_stream(_audio_stream, 0))

    async def process_audio_stream(audio_stream, i):
        """Async function to process audio frames from the audio stream."""
        async for event in audio_stream:
            # Convert bytes to numpy array for analysis (assuming 16-bit PCM)
            audio_data = np.frombuffer(event.frame.to_wav_bytes(), dtype=np.int16)

            # Write non-silent frames to wav file
            if (np.count_nonzero(np.abs(audio_data[22:]) > 10)) > 0:    # Remove 22 header long
                wav.writeframes(audio_data[22:])
    

    token = (
        api.AccessToken('API4bcDob32kABX','fWCQds2YzguBZJbVgdXbPCodqYY0jcHviHqIkwDZ7yV')
        .with_identity("python-consumer")
        .with_name("Python Consumer")
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

    """
    @room.on("track")
    async def on_track(audio_stream):
        print("Recibiendo audio...")
        # Leer datos de audio conforme llegan
        async for frame in audio_stream:
            # Escribe los datos de audio en el archivo WAV
            wav_file.writeframes(frame)
        print("Audio terminado.")

    
    # receive a track
    audio_stream = rtc.AudioStream.from_participant(
        participant=rtc.RemoteParticipant,
        track_source=rtc.TrackSource.SOURCE_MICROPHONE,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
    )  
    """  

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("logs/consumer_wave.log"), logging.StreamHandler()],
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