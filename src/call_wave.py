import asyncio
import logging
from signal import SIGINT, SIGTERM
import os
import wave
import time
import sys

import numpy as np
from livekit import rtc, api
from livekit.protocol.sip import CreateSIPOutboundTrunkRequest, SIPOutboundTrunkInfo,  ListSIPOutboundTrunkRequest

SAMPLE_RATE = 16000
NUM_CHANNELS = 1
FRAME_DURATION_MS = 10  # Frame duration in milliseconds


# audio_wav = "BTS/TFM/audios/audio_1.wav"
audio_wav = sys.argv[1]
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


    room_id = input("Please enter a value for the variable: ")
    logging.info("Trying to connect to room %s", room_id)
    token = (
        api.AccessToken('API4bcDob32kABX','fWCQds2YzguBZJbVgdXbPCodqYY0jcHviHqIkwDZ7yV')
        .with_identity("python-publisher")
        .with_name("Python Publisher")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_id,
                # room="TFM-room",
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

    #--------------------------------- SIP -------------------------------------------#
    livekit_api = api.LiveKitAPI(
        url="wss://test-tfm-3lii83j0.livekit.cloud",
        api_key="APIaRjG6ptSKLTy",
        api_secret="VAovqneYo91unfwHFCGmlJgevds1CZDUIbeufPUuThoE"
    )

    # Call a participant
    # To obtain trunkID
    # rules = await livekit_api.sip.list_sip_outbound_trunk(
    #     ListSIPOutboundTrunkRequest()
    # )
    # print(f"{rules}")

    # `create_sip_participant` starts dialing the user
    user_identity = "phone_callee"
    phone_number = "+34976214883"
    # out_trunk_id = input("Please enter the id of the outbound trunk: ")
    out_trunk_id = "ST_sEbW5d8fhh3E"
    await livekit_api.sip.create_sip_participant(
        api.CreateSIPParticipantRequest(
            room_name=room_id,
            sip_trunk_id=out_trunk_id,
            sip_call_to=phone_number,
            participant_identity=user_identity,
        )
    )

    # a participant is created as soon as we start dialing
    # participant = await room.wait_for_participant(identity=user_identity)
    print(f"Press enter to send the DTMF code to the participant")
    await asyncio.to_thread(input)
    # publishes extension in DTMF
    await room.local_participant.publish_dtmf(code=8, digit='8')
    await room.local_participant.publish_dtmf(code=2, digit='2')
    await room.local_participant.publish_dtmf(code=2, digit='2')
    await room.local_participant.publish_dtmf(code=7, digit='7')
    #---------------------------------------------------------------------------------#

    # publish a track
    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("audio_wav", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    publication = await room.local_participant.publish_track(track, options)
    logging.debug("published track %s", publication.sid)

    # Wait for Enter key press
    print('Press Enter to publish the wav file')
    await asyncio.to_thread(input)
    print('Publishing wav file')
    try:
        future = asyncio.ensure_future(publish_wav_frames(source, audio_wav))
        await future
    finally:
        # await asyncio.to_thread(input)
        time.sleep(2) # Compesate the inminent close of the track
        await room.local_participant.unpublish_track(track.sid, stop_on_unpublish=True)
        await source.aclose()
        
        logging.info("Unpublished track %s", publication.sid)



async def publish_wav_frames(source: rtc.AudioSource, wav_file_path: str):
    """Read a .wav file and send its audio frames through the source."""
    
    #source.clear_queue()
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
        # Maps the audio frame data to a numpy array for easier manipulation
        audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)

        # Read and send audio frames from the .wav file
        i = 0
        while True:
            # Read raw audio data from the file (in bytes)
            raw_data = wav_file.readframes(samples_per_channel)
            # logging.info(f'raw frame {i} is {raw_data[:10]}') 
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
            # logging.info(f"Capturing frame {list(audio_frame.data[:10])}")
            await source.capture_frame(audio_frame)
            # time.sleep(0.005) # Study the effect of delay
        print(f"Total frames of 10ms are {i}")   
    print("Finished publishing .wav audio file.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("BTS/TFM/logs/publish_wave.log"), logging.StreamHandler()],
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