from multiprocessing import process
import re
import torch
from torch.utils.data import DataLoader
from jiwer import wer, cer
import os 
import argparse
from datasets import load_from_disk, concatenate_datasets, Dataset
from typing import Any, Dict, List, Union
from dataclasses import dataclass
from transformers import WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
import evaluate
from tqdm import tqdm
import json


class PrepareDatasetAsInput:
    
    def __init__(self, feature_extractor, tokenizer_en, tokenizer_fr):
        self.feature_extractor = feature_extractor
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
            
    def prepare_dataset(self, batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids **** CHANGED FROM **sentence** TO **transcription**
        # if french, than use french tokenizer, english otherwise
        tokenizer = self.tokenizer_en
        if "lang" in batch:
            if batch["lang"] == "fr":
                tokenizer = self.tokenizer_fr

        batch["labels_fullts"] = tokenizer(batch["full_ts"]).input_ids
        batch["labels_shortts"] = tokenizer(batch["short_ts"]).input_ids

        return batch

    def prepare_dataset_with_prompt(self,batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode prompts to prompt ids - we assume that the dataset has a column `"prompt"` that contains the prompt for each example
        prompt_ids = []
        if 'prompt_fullts' in batch:
            prompt_ids = self.tokenizer_en.get_prompt_ids(batch["prompt_fullts"]).tolist() # YOU NEED TO ADD TOLIST() because array cant be combined with list in the next lines
        if 'prompt_shortts' in batch:
            prompt_ids = self.tokenizer_en.get_prompt_ids(batch["prompt_shortts"]).tolist() # YOU NEED TO ADD TOLIST() because array cant be combined with list in the next lines
            
        # encode target text to label ids **** CHANGED FROM **sentence** TO **transcription**
        # if french, than use french tokenizer, english otherwise
        tokenizer = self.tokenizer_en
        if "lang" in batch:
            if batch["lang"] == "fr":
                tokenizer = self.tokenizer_fr

        token_ids = tokenizer(batch["full_ts"]).input_ids

        batch["labels_fullts"] = token_ids # building labels ids with prompt and tokens together
        batch["labels_shortts"] = tokenizer(batch["short_ts"]).input_ids

        return batch
    
    def prepare_dataset_self_prompt(self,batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids **** CHANGED FROM **sentence** TO **transcription**
        # if french, than use french tokenizer, english otherwise
        tokenizer = self.tokenizer_en
        if "lang" in batch:
            if batch["lang"] == "fr":
                tokenizer = self.tokenizer_fr
        
        # make prompt from the lables
        batch['fullts_prompt_ids'] = self.tokenizer_en.get_prompt_ids(batch["full_ts"]).tolist() # YOU NEED TO ADD TOLIST() because array cant be combined with list in the next lines
        batch['shortts_prompt_ids'] = self.tokenizer_en.get_prompt_ids(batch["short_ts"]).tolist() # YOU NEED TO ADD TOLIST() because array cant be combined with list in the next lines
        
        batch["labels_fullts"] = tokenizer(batch["full_ts"]).input_ids # building labels ids with prompt and tokens together
        batch["labels_shortts"] = tokenizer(batch["short_ts"]).input_ids

        return batch

class ComputeMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.metric = evaluate.load("wer")
        
    def compute_metrics(self,pred_text, reference_text):
        pred_ids = pred_text
        label_ids = reference_text

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    def compute_metrics_from_text(self,pred_str,label_str):

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

@dataclass
class EvaluationSetup:
    metric : str
    datasets : str
    datasets_basedir : str
    model : str
    output_file : str
    transcription_name_in_ds : str
    prompt_name_in_ds : str = None
    overwrite : bool = False
    separate_ds : bool = False
    use_prompt : bool = True
    self_prompt : bool = False

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        batch["labels"] = labels_batch
        
        return batch

def build_dataset(ds_list : list[str], prepare_dataset_fn, path_to_ds :str, separate_ds=False, ts='fullts') -> list[Dataset]|Dataset:  
    allds_test = []
    # find all datasets to be tested
    for ds_name in ds_list:
        ds = None
        match ds_name:
            case 'atco_en_ruzyne':
                ds = load_from_disk(os.path.join(path_to_ds,"atco/en_ruzyne_test_ds"))
            case 'atco_en_stefanik':
                ds = load_from_disk(os.path.join(path_to_ds,"atco/en_stefanik_test_ds"))
            case 'atco_en_zurich':
                ds = load_from_disk(os.path.join(path_to_ds,"atco/en_zurich_test_ds"))
            case 'atco_fr':
                ds = load_from_disk(os.path.join(path_to_ds,"atco/fr_test_ds"))
            case 'atco_other_lang':
                ds = load_from_disk(os.path.join(path_to_ds,"atco/other_lang_test_ds"))
            case 'hiwire_fr':
                ds = load_from_disk(os.path.join(path_to_ds,"hiwire/hwir_fr_test_ds"))
            case 'hiwire_gr':
                ds = load_from_disk(os.path.join(path_to_ds,"hiwire/hwir_gr_test_ds"))
            case 'hiwire_sp':
                ds = load_from_disk(os.path.join(path_to_ds,"hiwire/hwir_sp_test_ds"))
            case 'malorca':
                ds = load_from_disk(os.path.join(path_to_ds,"malorca/malorca_test_ds"))
            case 'nato':
                ds = load_from_disk(os.path.join(path_to_ds,"nato/nato_test_ds"))
            case 'uwb':
                ds = load_from_disk(os.path.join(path_to_ds,"uwb/uwb_test_ds"))
        if (ds):
            allds_test.append(ds)
    
    # prepare the datasets to be ready for model
    for idx,ds in enumerate(allds_test):
        allds_test[idx] = ds.map(prepare_dataset_fn, remove_columns=ds.column_names, num_proc=1)
        allds_test[idx] = allds_test[idx].rename_column('labels_fullts' if ts == 'fullts' else 'labels_shortts','labels')
    
    # return either concatenated datasets or list of datasets
    if (separate_ds):
        return allds_test
    else:
        return concatenate_datasets(allds_test)
    
def compute(test_ds : list[Dataset]|Dataset, model, processor, metric, batch_size=3, use_prompt=False):        
    # define collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )
    
    if (isinstance(test_ds, Dataset)):
        # setup the dataloader
        dataloader = DataLoader(test_ds, batch_size=batch_size, collate_fn=data_collator)
        
        # Iterate over batches
        all_preds = []
        all_lables = []
        for batch in tqdm(dataloader):
            input_features = batch["input_features"].cuda()
            out = model.generate(input_features).detach().cpu()
            all_preds.extend(processor.batch_decode(out, skip_special_tokens=True))
            all_lables.extend(processor.batch_decode(batch["labels"]['input_ids'], skip_special_tokens=True))
            # Free memory
            del batch
            torch.cuda.empty_cache()
        
        # compute the wer
        wer=metric.compute_metrics_from_text(all_preds, all_lables)   
        
        return wer

def setup_model_processor(model_path) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    # setup the model
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    if torch.cuda.is_available():
        model.cuda()
    # setting the parameters of model for correct working
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    processor = WhisperProcessor.from_pretrained(model_path)
    
    return model, processor

def main(args):
    if (args.metric == 'wer'):
        # handle the output file
        if (args.overwrite):
            f= open(args.output_file, 'w')
        else: 
            if os.path.exists(args.output_file):
                print("Output file already exists. Use --overwrite to overwrite it.")
                exit(1)
            else:
                f= open(args.output_file, 'w')
        
        # run the evaluation
        # ===========================================
        
        # setup model and the processor
        model, processor = setup_model_processor(args.model)
        tokenizer_fr = WhisperTokenizer.from_pretrained(args.model, language="French") # just for the french tokenizer
        prep_ds_cls = PrepareDatasetAsInput(processor.feature_extractor, processor.tokenizer, tokenizer_fr)
        
        # split the datasets list (; separated) and build the dataset
        ds_list = args.datasets#.split(';')
        print(ds_list)
        built_ds = build_dataset(
            ds_list, 
            prep_ds_cls.prepare_dataset, 
            args.datasets_basedir, 
            args.separate_ds, 
            ts='fullts'
        )
        print(built_ds)
        exit(2)
        metric = ComputeMetrics(processor.tokenizer)
        # compute the wer
        out=compute(built_ds, model, processor, metric, batch_size=3)
        # ===========================================
        
        print(out)
        # write the output
        f.write(str(out))
        f.close()
    elif (args.metric == 'cer'):
        metric = cer

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Evaluate transcription accuracy in WER or CER.')
    parser.add_argument('--setup', type=str, required=True, help='Type of evaluation metric')
    evaluation_setup = parser.parse_args()
    
    # load the setup
    with open(evaluation_setup.setup, 'r') as f:
        setup = json.load(f)
    evaluation_setup = EvaluationSetup(**setup)
    
 
    # run the evaluation
    print (evaluation_setup)
    # exit(2)
    main(evaluation_setup)