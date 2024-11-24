import asyncio
import logging
from signal import SIGINT, SIGTERM
import os
import wave
import time

import numpy as np
from livekit import rtc, api

SAMPLE_RATE = 48000
NUM_CHANNELS = 1
FRAME_DURATION_MS = 10  # Frame duration in milliseconds


audio_wav = "TFM/audios/audio_1.wav"
#audio_wav = "TFM/audios/5-CH0_C01_stadium_15dB.wav"
# ensure LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are set


async def main(room: rtc.Room) -> None:
    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        logging.info(
            "participant connected: %s %s %s %s", participant.sid, participant.identity, participant.metadata, participant.name
        )
    
    @room.on("participant_disconnected")
    def on_participant_disconnect(participant: rtc.Participant, *_):
        logging.info("participant disconnected: %s", participant.identity)

    token = (
        api.AccessToken('API4bcDob32kABX','fWCQds2YzguBZJbVgdXbPCodqYY0jcHviHqIkwDZ7yV')
        .with_identity("python-publisher")
        .with_name("Python Publisher")
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

    # publish a track
    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("audio_wav", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    publication = await room.local_participant.publish_track(track, options)
    logging.debug("published track %s", publication.sid)

    # Wait for Enter key press
    await asyncio.to_thread(input)
    print('Publishing wav file')
    try:
        future = asyncio.ensure_future(publish_wav_frames(source, audio_wav))
        await future
    finally:
        time.sleep(1) # Compesate the inminent close of the track
        await room.local_participant.unpublish_track(track.sid, stop_on_unpublish=True)
        logging.info("Unpublished track %s", publication.sid)




async def publish_wav_frames(source: rtc.AudioSource, wav_file_path: str):
    """Read a .wav file and send its audio frames through the source."""
    
    source.clear_queue()
    # Open the .wav file
    with wave.open(wav_file_path, 'rb') as wav_file:
        # Ensure the .wav file's format matches the stream's expected sample rate and channels
        wav_sample_rate = wav_file.getframerate()
        wav_channels = wav_file.getnchannels()
        
        if wav_sample_rate != SAMPLE_RATE or wav_channels != NUM_CHANNELS:
            raise ValueError(f"Expected .wav file with {SAMPLE_RATE} Hz and {NUM_CHANNELS} channel(s), "
                             f"but got {wav_sample_rate} Hz and {wav_channels} channel(s).")

        # Calculate samples per frame
        samples_per_channel = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # For 10 ms frame duration

        # Prepare the audio frame
        audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, samples_per_channel)
        audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)

        # Read and send audio frames from the .wav file
        i = 0
        while True:
            # Read raw audio data from the file (in bytes)
            raw_data = wav_file.readframes(samples_per_channel)
            if not raw_data:
                break  # End of file reached
            
            # Convert raw audio data to numpy array and fill the audio frame
            wav_samples = np.frombuffer(raw_data, dtype=np.int16)
            
            # Check if the data is shorter than expected
            if len(wav_samples) < samples_per_channel:
                # Pad with zeros if necessary
                wav_samples = np.pad(wav_samples, (0, samples_per_channel - len(wav_samples)), 'constant')
            i +=1
                     

            np.copyto(audio_data, wav_samples)

            # Capture frame to send it to the track
            await source.capture_frame(audio_frame)
        print(f"Total frames of 10ms are {i}")   
    print("Finished publishing .wav audio file.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("TFM/logs/publish_wave.log"), logging.StreamHandler()],
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