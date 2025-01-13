import os
import wave
import time
import asyncio
import logging
import numpy as np
from livekit import rtc, api
from signal import SIGINT, SIGTERM

SAMPLE_RATE = 8000
FRAME_DURATION_MS = 10
NUM_CHANNELS = 1
FORMAT = 2 # 16-bit PCM
WAV_FILE = "BTS/TFM/audios/audio_received.wav"

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
            _audio_stream = rtc.AudioStream(track,sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS)
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
            i = 0

            samples_per_channel = SAMPLE_RATE * FRAME_DURATION_MS // 1000                       # Calculate samples per frame
            audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, samples_per_channel) # Prepare the audio frame
            audio_data_tx = np.frombuffer(audio_frame.data, dtype=np.int16)                     # Maps the audio frame data to a numpy array for easier manipulation

            async for event in audio_stream:
                # Check if we should stop processing
                if stop_processing.is_set():
                    logging.info("Stopping audio processing as track is unpublished.")
                    break
                i += 1
                audio_data_rx = np.frombuffer(event.frame.data, dtype=np.int16)
                wav.writeframes(audio_data_rx) 

                # Publish audio frames to the track in room_2
                await asyncio.ensure_future(publish_frames(source, audio_frame, audio_data_tx, audio_data_rx))


            print(f"Total frames are {i}")
            logging.info("Audio stream processing completed.")
        except Exception as e:
            logging.error(f"Error processing audio stream: {e}")
    
    room_id_1 = input("Please enter an id for the receiver room: ")
    token_1 = (
        api.AccessToken('API4bcDob32kABX','fWCQds2YzguBZJbVgdXbPCodqYY0jcHviHqIkwDZ7yV')
        .with_identity("python-consumer")
        .with_name("Python Consumer")
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
        .with_identity("python-consumer")
        .with_name("Python Consumer")
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
    time.sleep(0.001) # Study the effect of delay

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