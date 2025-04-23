from transformers import WhisperForConditionalGeneration, WhisperProcessor
import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Visualize the evaluation of sentences.')
    argparser.add_argument('--model', type=str, required=True, help='Model name or path')
    
    args=argparser.parse_args()
    
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    