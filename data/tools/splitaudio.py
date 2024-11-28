from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import sys
# Load the .wav file
if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print("Usage: python splitaudio.py <in_file_path> <out_file_path>")
        sys.exit(1)
    
    in_file_path = sys.argv[1]#"/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/A-PiMod/2013_10_Christoph/01_02_EL_LN_UN_VV_YADA.wav"
    out_file_path = sys.argv[2]
    
    audio = AudioSegment.from_wav(in_file_path)

    # Directory to save the split audio files
    os.makedirs(out_file_path, exist_ok=False)

    # Split the audio on silence (you can adjust min_silence_len and silence_thresh)
    chunks = split_on_silence(audio,    
                            min_silence_len=1000,  # Minimum silence length in milliseconds
                            silence_thresh=-40)    # Silence threshold in dBFS

    # Save each non-silent chunk as a new .wav file
    for i, chunk in enumerate(chunks):
        output_path = os.path.join(output_dir, f"recording_{i + 1}.wav")
        chunk.export(output_path, format="wav")
        print(f"Saved: {output_path}")

    print(f"Splitting completed. {len(chunks)} files saved to {output_dir}.")
