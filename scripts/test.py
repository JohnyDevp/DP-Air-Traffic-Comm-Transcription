import argparse

# do parse args
parser = argparse.ArgumentParser(description='Evaluate transcription accuracy in WER or CER.')
parser.add_argument('--datasets', type=str, required=True, help='Path to the dataset file with both audio and transcription as are provided in finetuning dataset')
args = parser.parse_args()

print(args.datasets.replace('[','').replace(']','').split(','))