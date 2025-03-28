import json
from click import prompt
from regex import D
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
import numpy as np


class PrepareDatasetAsInput:
    
    def __init__(self, feature_extractor, tokenizer_en, tokenizer_fr):
        self.feature_extractor = feature_extractor
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
    
    def set_transcription_name(self, transcription_name):
        self.transcription_name = transcription_name
    
    def set_prompt_name(self, prompt_name):
        self.prompt_name = prompt_name
               
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

        batch['labels'] = tokenizer(batch[self.transcription_name]).input_ids

        return batch
    
    def prepare_dataset_with_prompt(self,batch):
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
        
        # encode prompts to prompt ids - we assume that the dataset has a column `"prompt"` that contains the prompt for each example
        prompt_ids = tokenizer.get_prompt_ids(batch[self.prompt_name]).tolist() # YOU NEED TO ADD TOLIST() because array cant be combined with list in the next lines

        batch["prompt_ids"] = prompt_ids
        batch["labels"] = tokenizer(batch[self.transcription_name]).input_ids # building labels ids with prompt and tokens together
        
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
        prompt_ids = self.tokenizer_en.get_prompt_ids(batch[self.transcription_name]).tolist() # YOU NEED TO ADD TOLIST() because array cant be combined with list in the next lines

        batch['prompt_ids'] = prompt_ids 
        batch['labels'] = tokenizer(batch[self.transcription_name]).input_ids # building labels ids with prompt and tokens together

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
class DataCollatorSpeechSeq2SeqWithPaddingWOPrompt:
    processor: Any
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

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPaddingWITHPROMPT:
    processor : Any
    decoder_start_token_id: int
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # ==================================================================================
        # WORKING CALL, USED IN PROMPT-TEST-3A, CORRECT, ONLY WITH 448 MAX LENGTH
        # ==================================================================================

        # split inputs and labels since they have to be of different lengths and need
        # different padding methods

        # dataloader returns a list of features which we convert to a dict
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        prompt_features = [{"prompt_ids": feature["prompt_ids"]} for feature in features]
        
        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        # shift labels to the right to get decoder input ids
        labels = labels_batch["input_ids"]
        
        labels_mask = labels_batch.attention_mask
        
        # replace padding with -100 to ignore correctly when computing the loss
        labels = labels.masked_fill(labels_mask.ne(1), -100)

        batch["labels"] = labels
        batch["prompt_ids"] = prompt_features['prompt_ids']

        return batch
    
    def myoldcall_stillworking(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # copy the labels as in its form now (with both prompt and the transcript itself) it should be the input to the decoder
        batch['decoder_input_ids'] = labels_batch["input_ids"].clone()[:, :-1]

        # replace padding in labels with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)[:, 1:]
        
        # then mask out the prompt in the labels
        bos_index = np.argmax(labels==self.decoder_start_token_id, axis=1)
        prompt_mask = np.arange(labels.shape[1]) < bos_index.numpy()[:, np.newaxis]
        labels = torch.tensor(np.where(prompt_mask, -100, labels))

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways (actually it's not needed to cut it here, because it the bos
        # tokeon is appended just if decode_input_ids is not provided)
        # STILL WE LET IT BE THERE AS PART OF ORIGINAL CODE
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        # and at last we create attention mask, to tell the decoder to use the correct part from the labels
        # attention_mask = torch.tensor(np.where(decoder_input_ids != tokenizer_en.pad_token_id, 1 , 0))
        # batch["attention_mask"] = attention_mask

        return batch

@dataclass
class EvaluationSetup:
    metric : str
    datasets : str
    datasets_basedir : str
    model : str
    output_file : str
    transcription_name_in_ds : str
    prompt_name_in_ds : str = None
    eval_description : str = ""
    overwrite : bool = False
    separate_ds : bool = False
    use_prompt : bool = True
    self_prompt : bool = False

def build_dataset(ds_list : list[str], prepare_dataset_fn, path_to_ds :str, separate_ds=False) -> list[Dataset]|Dataset:  
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
        # allds_test[idx] = allds_test[idx].rename_column('labels_fullts' if transcription == 'fullts' else 'labels_shortts','labels')
    
    # return either concatenated datasets or list of datasets
    if (separate_ds):
        return allds_test
    else:
        return concatenate_datasets(allds_test)
    
def compute(test_ds : list[Dataset]|Dataset, model, processor, metric, batch_size=3, use_prompt=False):        
    # define collator
    if (not use_prompt):
        data_collator = DataCollatorSpeechSeq2SeqWithPaddingWOPrompt(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id
        )
    else:
        data_collator = DataCollatorSpeechSeq2SeqWithPaddingWITHPROMPT(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id
        )
    
    if (isinstance(test_ds, Dataset) and not use_prompt):
        # setup the dataloader
        dataloader = DataLoader(test_ds, batch_size=batch_size, collate_fn=data_collator)
        
        # Iterate over batches
        all_preds = []
        all_lables = []
        for batch in tqdm(dataloader):
            input_features = batch["input_features"].cuda()
            preds = model.generate(input_features).detach().cpu()
            preds = model(input_features, labels=batch["labels"].cuda())
            print(preds.loss)
            
            exit(1)
            all_preds.extend(processor.batch_decode(preds, skip_special_tokens=True))
            all_lables.extend(processor.batch_decode(batch["labels"], skip_special_tokens=True))
            # Free memory
            del batch
            torch.cuda.empty_cache()
        
        # compute the wer
        wer=metric.compute_metrics_from_text(all_preds, all_lables)   
        
        return wer
    elif (isinstance(test_ds, Dataset) and use_prompt):
        # WHEN using PROMPT, so far we can test only one sample at a time with one promp
        # because yet we cannot handle different prompts for different samples in the same batch
        # setup the dataloader
        # dataloader = DataLoader(test_ds, batch_size=1, collate_fn=data_collator)
        
        # Iterate over batches
        all_preds = []
        all_lables = []
        loss = []
        for d in tqdm(test_ds):
            input_features = torch.tensor(d["input_features"]).unsqueeze(0).cuda()
            prompt_ids = torch.tensor(d["prompt_ids"]).cuda()
            decoder_input_ids = torch.tensor(prompt_ids + ).unsqueeze(0).cuda()
            
            outputs = model(input_features, labels=d["labels"].cuda())
            
            preds = model.generate(input_features, prompt_ids=prompt_ids).detach().cpu()
            
            all_preds.extend([processor.tokenizer.decode(preds[0], skip_special_tokens=True)])
            all_lables.extend([processor.tokenizer.decode(d["labels"], skip_special_tokens=True)])
        
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

def main(evaluation_setup : EvaluationSetup):
    if (evaluation_setup.metric == 'wer'):
        # handle the output file
        if (evaluation_setup.overwrite):
            f= open(evaluation_setup.output_file, 'w')
        else: 
            if os.path.exists(evaluation_setup.output_file):
                print("Output file already exists. Use --overwrite to overwrite it.")
                exit(1)
            else:
                f= open(evaluation_setup.output_file, 'w')
        
        # run the evaluation
        # ===========================================
        
        # setup model and the processor
        model, processor = setup_model_processor(evaluation_setup.model)
        tokenizer_fr = WhisperTokenizer.from_pretrained(evaluation_setup.model, language="French") # just for the french tokenizer
        prep_ds_cls = PrepareDatasetAsInput(processor.feature_extractor, processor.tokenizer, tokenizer_fr)
        
        # set up the prompt and transcription names in the given dataset, that will be used
        prep_ds_cls.set_transcription_name(evaluation_setup.transcription_name_in_ds)
        prep_ds_cls.set_prompt_name(evaluation_setup.prompt_name_in_ds)
        
        ds_list = evaluation_setup.datasets
        
        if evaluation_setup.use_prompt:
            if evaluation_setup.self_prompt:
                prepare_fn = prep_ds_cls.prepare_dataset_self_prompt
            else:
                prepare_fn = prep_ds_cls.prepare_dataset_with_prompt
        else:
            prepare_fn = prep_ds_cls.prepare_dataset
        built_ds = build_dataset(
            ds_list, 
            prepare_fn,
            evaluation_setup.datasets_basedir, 
            evaluation_setup.separate_ds
        )
        metric = ComputeMetrics(processor.tokenizer)
        # compute the wer
        out=compute(built_ds, model, processor, metric, batch_size=3, use_prompt=evaluation_setup.use_prompt)
        # ===========================================
    
        print(out)
        # write the output
        f.write("******** Evaluation setup ********\n")
        f.write(evaluation_setup.__str__())
        f.write("\n")
        f.write("******** Evaluation description ********\n")
        f.write(evaluation_setup.eval_description)
        f.write("\n")
        f.write("******** Evaluation results ********\n")
        f.write(str(out))
        f.close()
        
    elif (evaluation_setup.metric == 'cer'):
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
    
    print('******** Evaluation setup ********')
    print(evaluation_setup.__str__())
    
    # run the evaluation
    main(evaluation_setup)