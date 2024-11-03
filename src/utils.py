import pyaudio
import wave

def play_audio(filename):
    # Open the .wav file for playback
    with wave.open(filename, 'rb') as wf:
        audio = pyaudio.PyAudio()

        # Open a stream for playback
        stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)

        print("Playing audio...")

        # Read and play back audio in chunks
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)

        print("Playback finished.")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()