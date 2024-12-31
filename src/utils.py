import subprocess

def download_audio_files(remote_host, remote_directory, local_directory, username):
    try:
        # Construct the SCP command
        scp_command = f"scp {username}@{remote_host}:{remote_directory}/*.wav {local_directory}"
        subprocess.run(scp_command, shell=True, check=True)
        print("Files downloaded successfully to:", local_directory)
    except subprocess.CalledProcessError as e:
        print("Error during file transfer:", str(e))

# Configuration
remote_host = "your.remote.server.com"
username = "your_username"
remote_directory = "/path/to/remote/directory"
local_directory = "/path/to/local/directory"

download_audio_files(remote_host, remote_directory, local_directory, username)
