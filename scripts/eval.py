from io import TextIOWrapper
from arrow import get
import torch
from torch.utils.data import DataLoader
from jiwer import wer, cer
import os, json, argparse, time
from datasets import load_from_disk, concatenate_datasets, Dataset
from typing import Any, Dict, List, Union
from dataclasses import dataclass
from transformers import WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig
import evaluate
from tqdm import tqdm
from glob import glob
import numpy as np


class PrepareDatasetAsInput:

    def __init__(self, feature_extractor, tokenizer_en, tokenizer_fr, prompt_version=None):
        self.feature_extractor = feature_extractor
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr

    def set_transcription_name(self, transcription_name):
        self.transcription_name = transcription_name

    def set_prompt_name(self, prompt_name):
        self.prompt_name = prompt_name

    def prepare_dataset(self, batch):
        if (self.transcription_name is None):
            raise ValueError("Transcription name is not set. Please set it using set_transcription_name() method.")
        
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        features = self.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"],
            truncation=False,
            padding='max_length',
            return_attention_mask=True
        )

        # add input features to batch
        batch["input_features"] = features.input_features[0]
        
        # add attention mask to batch, potentially for data augmentation
        batch['attention_mask'] = features.attention_mask[0]

        # encode target text to label ids **** CHANGED FROM **sentence** TO **transcription**
        # if french, than use french tokenizer, english otherwise
        tokenizer = self.tokenizer_en
        if "lang" in batch:
            if batch["lang"] == "fr":
                tokenizer = self.tokenizer_fr

        batch['labels'] = tokenizer(batch[self.transcription_name]).input_ids

        return batch

    def prepare_dataset_with_prompt(self,batch):
        if (self.transcription_name is None or self.prompt_name is None):
            raise ValueError("Transcription name is not set. Please set it using set_transcription_name() method.")
        
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        features = self.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"],
            truncation=False,
            padding='max_length',
            return_attention_mask=True
        )

        # add input features to batch
        batch["input_features"] = features.input_features[0]
        
        # add attention mask to batch, potentially for data augmentation
        batch['attention_mask'] = features.attention_mask[0]

        # encode target text to label ids **** CHANGED FROM **sentence** TO **transcription**
        # if french, than use french tokenizer, english otherwise
        tokenizer = self.tokenizer_en
        if "lang" in batch:
            if batch["lang"] == "fr":
                tokenizer = self.tokenizer_fr

        # encode prompts to prompt ids - we assume that the dataset has a column `"prompt"` that contains the prompt for each example
        prompt_ids = tokenizer.get_prompt_ids(batch[self.prompt_name]).tolist() # YOU NEED TO ADD TOLIST() because array cant be combined with list in the next lines

        batch["labels"] = prompt_ids + tokenizer(batch[self.transcription_name]).input_ids # building labels ids with prompt and tokens together

        return batch

    def prepare_dataset_self_prompt(self,batch):
        if (self.transcription_name is None):
            raise ValueError("Transcription name is not set. Please set it using set_transcription_name() method.")
        
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        features = self.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"],
            truncation=False,
            padding='max_length',
            return_attention_mask=True
        )

        # add input features to batch
        batch["input_features"] = features.input_features[0]
        
        # add attention mask to batch, potentially for data augmentation
        batch['attention_mask'] = features.attention_mask[0]
        # encode target text to label ids **** CHANGED FROM **sentence** TO **transcription**
        # if french, than use french tokenizer, english otherwise
        tokenizer = self.tokenizer_en
        if "lang" in batch:
            if batch["lang"] == "fr":
                tokenizer = self.tokenizer_fr

        # make prompt from the lables
        prompt_ids = self.tokenizer_en.get_prompt_ids(batch[self.transcription_name]).tolist() # YOU NEED TO ADD TOLIST() because array cant be combined with list in the next lines

        batch['labels'] = prompt_ids + tokenizer(batch[self.transcription_name]).input_ids # building labels ids with prompt and tokens together

        return batch
  
class ComputeMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.metric = evaluate.load("wer")
        
    def compute_metrics(self,pred_text, reference_text) -> float:
        pred_ids = pred_text
        label_ids = reference_text

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return wer
    
    def compute_metrics_from_text(self,pred_str,label_str) -> float:

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return wer

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

        # FOR USE ONLY IF WANT DATA AUGMENTATION
        batch['attention_mask'] = torch.tensor([mask['attention_mask'] for mask in features])

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
        
        # get the decoder input ids, by removing the last token (this is the 'shift' operation)
        decoder_input_ids = labels[:, :-1]

        # shift the labels to the left, to match work as prediction
        labels = labels[:, 1:]
        labels_mask = labels_batch.attention_mask[:, 1:]

        # replace padding with -100 to ignore correctly when computing the loss
        labels = labels.masked_fill(labels_mask.ne(1), -100)

        # replace initial prompt tokens with -100 to ignore correctly when computing the loss
        bos_index = torch.argmax((labels == self.decoder_start_token_id).long(), dim=1)
        prompt_mask = torch.arange(labels.shape[1]) < bos_index[:, None]
        labels = torch.where(prompt_mask, -100, labels)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids
   
        # FOR USE ONLY IF WANT DATA AUGMENTATION
        batch['attention_mask'] = torch.tensor([mask['attention_mask'] for mask in features])
        
        return batch

@dataclass
class EvaluationSetup:
    r"""
        Class that stores the evalution parameters passed to script

        `same_processor` : bool

    """
    metric : str
    datasets : str
    datasets_basedir : str
    models : str | list
    output_file : str
    transcription_name_in_ds : str
    checkpoints_eval : bool = False
    batch_size : int = 1
    same_processor : bool = True
    prompt_name_in_ds : str = None
    eval_description : str = ""
    overwrite : bool = False
    separate_ds : bool = False
    use_prompt : bool = True
    self_prompt : bool = False


from rapidfuzz import process
import re
class EvalCallsigns:
    wer_metric : ComputeMetrics
    
    def __init__(self, metric : ComputeMetrics):
        self.wer_metric = metric
    
    def __call__(self, transcription : str, callsigns : dict[str,int]):
        if (isinstance(callsigns,str)):
            callsigns = [callsigns]

        for callsign,num_of_occurences in callsigns.items():
            wer = self.find_lowest_wer(callsign, num_of_occurences, transcription)
    
    def find_lowest_wer(self, callsign : str, num_of_occurences : int, transcription : str) -> tuple[int,float]:
        callsign_norm = re.sub(r'\s+',' ',callsign.strip().lower()).split(' ')
        transcription_norm = re.sub(r'\s+',' ',transcription.strip().lower()).split(' ')
        # arange a list where wer will be stored
        wer_list = np.zeros(len(transcription) - len(callsign_norm) + 1)
        # move a window with callsign through the transcription, compute wer and store
        for idx in range(0,len(transcription) - len(callsign_norm) + 1):
            # check if the callsign is in the transcription
            cal_wer = self.wer_metric.compute_metrics_from_text(
                callsign_norm, transcription_norm[idx:idx+len(callsign_norm)]
            )
            wer_list[idx] = cal_wer
        
        # return as many lowest wer as num_of_occurences
        return sorted(wer_list)[0:num_of_occurences]
        
    def __obtain_callsign_from_transcription(self, callsigns, callsigns_pos, transcription : str) -> tuple[int,float]:
        ts_cor = re.sub(transcription.strip().lower(),r'\s+')
        ts_arr = ts_cor.split(' ')
        
        totally_correct_callsigns = 0
        wer_callsigns = .0
        for callsign in callsigns:
            # count all totally correct callsigns
            totally_correct_callsigns += ts_cor.count(callsign.strip().lower())
            
            callsign_arr = callsign.strip().lower().split(' ')    
            
            # check if the callsign is in the transcription
            for cal in callsign_arr:
                process.extractOne(cal, ts_arr, scorer=80)
        
        return totally_correct_callsigns, wer_callsigns
    
    def __search_for_callsign(self, callsign):
        # search for the callsign in the dataset
        # if found, return True
        # if not found, return False
        pass

def cuda_clean(model=None):
  if model:
    model.to('cpu')
    del model
  torch.cuda.synchronize()
  torch.cuda.empty_cache()
  
def build_dataset(ds_list : list[str], prepare_dataset_fn, path_to_ds :str, separate_ds=False) -> dict[str,Dataset]|Dataset:  
    allds_test = {}
    # find all datasets to be tested
    for ds_name in ds_list:
        ds = None
        match ds_name:
            case 'atco_en_ruzyne':
                if (os.path.exists(os.path.join(path_to_ds,"atco/en_ruzyne_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"atco/en_ruzyne_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"en_ruzyne_test_ds"))
            case 'atco_en_stefanik':
                if (os.path.exists(os.path.join(path_to_ds,"atco/en_stefanik_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"atco/en_stefanik_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"en_stefanik_test_ds"))
            case 'atco_en_zurich':
                if (os.path.exists(os.path.join(path_to_ds,"atco/en_zurich_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"atco/en_zurich_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"en_zurich_test_ds"))
            case 'atco_fr':
                if (os.path.exists(os.path.join(path_to_ds,"atco/fr_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"atco/fr_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"fr_test_ds"))
            case 'atco_other_lang':
                if (os.path.exists(os.path.join(path_to_ds,"atco/other_lang_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"atco/other_lang_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"other_lang_test_ds"))
            case 'hiwire_fr':
                if (os.path.exists(os.path.join(path_to_ds,"hiwire/hwir_fr_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"hiwire/hwir_fr_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"hwir_fr_test_ds"))
            case 'hiwire_gr':
                if (os.path.exists(os.path.join(path_to_ds,"hiwire/hwir_gr_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"hiwire/hwir_gr_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"hwir_gr_test_ds"))
            case 'hiwire_sp':
                if (os.path.exists(os.path.join(path_to_ds,"hiwire/hwir_sp_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"hiwire/hwir_sp_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"hwir_sp_test_ds"))
            case 'malorca':
                if (os.path.exists(os.path.join(path_to_ds,"malorca/malorca_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"malorca/malorca_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"malorca_test_ds"))
            case 'nato':
                if (os.path.exists(os.path.join(path_to_ds,"nato/nato_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"nato/nato_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"nato_test_ds"))
            case 'uwb':
                if (os.path.exists(os.path.join(path_to_ds,"uwb/uwb_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"uwb/uwb_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"uwb_test_ds"))
        if (ds):
            allds_test[ds_name]=ds
    
    # prepare the datasets to be ready for model
    for key in allds_test:
        allds_test[key] = allds_test[key].map(prepare_dataset_fn, remove_columns=ds.column_names, num_proc=1)
    
    # return either concatenated datasets or list of datasets
    if (separate_ds):
        return allds_test
    else:
        return concatenate_datasets([allds_test[key] for key in allds_test])

def compute(test_ds : dict[str,Dataset]|Dataset, model, processor, metric, batch_size=3, use_prompt=False, compute_callsign_wer=False, compute_runway_wer=False) -> dict[str,dict]:
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

    # run the evaluation
    if (isinstance(test_ds, Dataset) and not use_prompt):
        print(f"SINGLE DATASET, NOT PROMPT, batch size {batch_size}")
        # setup the dataloader
        dataloader = DataLoader(test_ds, batch_size=batch_size, collate_fn=data_collator)

        # Iterate over batches
        all_preds = []
        all_lables = []
        all_loss = []
        for batch in tqdm(dataloader):
            input_features = batch["input_features"].cuda()

            preds = model.generate(input_features)
            with torch.no_grad():
              outputs = model(input_features, labels=batch["labels"].cuda())

            # detach from gpu
            input_features.detach().cpu()
            preds = preds.detach().cpu()

            # TODO FIX ALL LOSSES IN OTHER PLACES, as it is here
            all_loss.extend(outputs.loss.detach().cpu().repeat(len(batch)))
            all_preds.extend(processor.batch_decode(preds, skip_special_tokens=True))
            all_lables.extend(processor.batch_decode(batch["labels"], skip_special_tokens=True))


        # compute the wer
        wer=metric.compute_metrics_from_text(all_preds, all_lables)
        loss=torch.mean(torch.tensor(all_loss,dtype=float))

        print(f"allds > wer: {wer} loss: {loss}") # PRINTOUT

        return {'allds':{'wer':wer,'loss':loss}}

    elif (isinstance(test_ds, dict) and not use_prompt):
        print(f"MULTIPLE DATASETS, NOT PROMPT, batch size {batch_size}")
        out = {}
        for ds_name, ds in test_ds.items():
            # setup the dataloader
            dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=data_collator)

            # Iterate over batches
            all_preds = []
            all_lables = []
            all_loss = []
            for batch in tqdm(dataloader):
                input_features = batch["input_features"].cuda()
                labels=batch["labels"].cuda()
                
                preds = model.generate(input_features)
                with torch.no_grad():
                  outputs = model(input_features, labels=labels)

                # detach from gpu
                input_features.detach().cpu()
                labels.detach().cpu()
                preds = preds.detach().cpu()

                all_loss.extend(outputs.loss.detach().cpu().repeat(len(batch)))
                all_preds.extend(processor.batch_decode(preds, skip_special_tokens=True))
                all_lables.extend(processor.batch_decode(batch["labels"], skip_special_tokens=True))


            # compute the wer
            wer=metric.compute_metrics_from_text(all_preds, all_lables)
            loss=torch.mean(torch.tensor(all_loss,dtype=float))
            print(f"{ds_name} > wer: {wer} loss: {loss}") # PRINTOUT
            out[ds_name] = {'wer':wer,'loss':loss}

        return out

    elif (isinstance(test_ds, Dataset) and use_prompt):
        print(f"SINGLE DATASET, USE PROMPT, batch size {batch_size}")
        # WHEN using PROMPT, so far we can test only one sample at a time with one promp
        # because yet we cannot handle different prompts for different samples in the same batch
        # setup the dataloader
        dataloader = DataLoader(test_ds, batch_size=batch_size, collate_fn=data_collator)


        # Iterate over batches
        all_preds = []
        all_lables = []
        all_loss = []
        for batch in tqdm(dataloader):

            # pass everything to CUDA
            for key in batch:
                if (key == 'prompt_ids'): continue
                batch[key] = batch[key].cuda()

            # loop through the batch and compute the predicitons
            # it is necessary because model.generate() cannot handle multiple input features with multiple prompt
            for idx in range(batch_size):
                # check whether the index doesnt go above the number of items in current batch
                if (idx >= len(batch['input_features'])): break

                input_features = batch["input_features"][idx].unsqueeze(0)
                prompt_ids = torch.tensor(batch["prompt_ids"][idx]['prompt_ids']).cuda()

                preds = model.generate(input_features, prompt_ids=prompt_ids)
                # detach the predictions from the gpu
                preds = preds.detach().gpu()

                # extend all the predictions and labels for later wer computation
                all_preds.extend([processor.tokenizer.decode(preds[0], skip_special_tokens=True)])
                all_lables.extend([processor.tokenizer.decode(batch["labels"][idx], skip_special_tokens=True)])
                
                # compute the wer for callsigns

            # compute the loss
            with torch.no_grad():
                outputs = model(input_features = batch['input_features'], labels=batch['labels'], decoder_input_ids=batch['decoder_input_ids'])

            # append the loss result to the whole result obtained so far
            all_loss.extend(outputs.loss.detach().cpu().repeat(len(batch)))

        # compute the wer
        wer=metric.compute_metrics_from_text(all_preds, all_lables)
        loss = torch.mean(torch.tensor(all_loss),dtype=float)
        print(f"allds > wer: {wer} loss: {loss}") # PRINTOUT
        return {'allds':{'wer':wer, 'loss':loss}}

    elif (isinstance(test_ds, dict) and use_prompt):
        print(f"MULTIPLE DATASETS, USE PROMPT, batch size {batch_size}")
        # WHEN using PROMPT, so far we can test only one sample at a time with one promp
        # because yet we cannot handle different prompts for different samples in the same batch
        # setup the dataloader
        out = {}
        for ds_name,ds in test_ds.items():
            dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=data_collator)

            # Iterate over batches
            all_preds = []
            all_lables = []
            all_loss = []
            for batch in tqdm(dataloader):

                # pass everything to CUDA
                for key in batch:
                    if (key == 'prompt_ids'): continue
                    batch[key] = batch[key].cuda()

                # loop through the batch and compute the predicitons
                # it is necessary because model.generate() cannot handle multiple input features with multiple prompt
                for idx in range(batch_size):
                    # check whether the index doesnt go above the number of items in current batch
                    if (idx >= len(batch['input_features'])): break

                    input_features = batch["input_features"][idx].unsqueeze(0)
                    prompt_ids = torch.tensor(batch["prompt_ids"][idx]['prompt_ids']).cuda()
                    preds = model.generate(input_features, prompt_ids=prompt_ids)
                    preds = preds.detach().cpu()

                    # compute loss together for all batch
                    all_preds.extend([processor.tokenizer.decode(preds[0], skip_special_tokens=True)])
                    all_lables.extend([processor.tokenizer.decode(batch["labels"][idx], skip_special_tokens=True)])

                # compute the loss
                with torch.no_grad():
                    outputs = model(input_features = batch['input_features'], labels=batch['labels'], decoder_input_ids=batch['decoder_input_ids'])
                
                # append the loss result to the whole result obtained so far
                all_loss.extend(outputs.loss.detach().cpu().repeat(len(batch)))

            # compute the wer and loss for current dataset
            wer=metric.compute_metrics_from_text(all_preds, all_lables)
            loss = torch.mean(torch.tensor(all_loss),dtype=float)
            print(f"{ds_name} > wer: {wer} loss: {loss}") # PRINTOUT
            out[ds_name] = {'wer':wer, 'loss':loss}

        return out
    else:
        raise ValueError('Wrong dataset format')

def setup_model_processor(model_path) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    # setup the model
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    if torch.cuda.is_available():
        model.cuda()
    # setting the parameters of model for correct working
    
    model.generation_config = GenerationConfig.from_pretrained('openai/whisper-medium')
    
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    
    
    
    processor = WhisperProcessor.from_pretrained(model_path,language="English", task="transcribe")
    
    return model, processor

def result_printout(f_desc : TextIOWrapper, out_dict : dict[str, dict], evaluation_setup : EvaluationSetup):
    for k,v in out_dict.items():
        print(f"DATASET: {k} | WER: {v['wer']} LOSS: {v['loss']}") # PRINTOUT

    # write the output
    f_desc.write("******** Evaluation results ********\n")
    for k,v in out_dict.items():
        f_desc.write(f"DATASET: {k} | WER: {v['wer']} LOSS: {v['loss']}\n")

    f_desc.flush()
           
def main(evaluation_setup : EvaluationSetup):
    if (evaluation_setup.metric == 'wer'):
        # handle the output file
        if (evaluation_setup.overwrite):
            out_file= open(evaluation_setup.output_file, 'w')
        else:
            out_file= open(evaluation_setup.output_file, 'a')

        # print to file evaluation info
        out_file.write(f'#### EVALUATION STARTED - TIME {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} ####\n')
        out_file.write("******** Evaluation setup ********\n")
        out_file.write(evaluation_setup.__str__())
        out_file.write("\n")
        out_file.write("******** Evaluation description ********\n")
        out_file.write(evaluation_setup.eval_description)
        out_file.write("\n\n")
        out_file.flush()
        
        def get_current_models_in_checkpoint(checkpoints_dir) -> set[str]:
            all_models=[]
            for check_model in sorted(glob(checkpoints_dir+'/checkpoint-*'),key=lambda x: int(x.split('-')[-1])):
                all_models.append(check_model)
            return set(all_models)
        
        # handle multiple models
        if (isinstance(evaluation_setup.models,list) or evaluation_setup.checkpoints_eval):
            # build the list of paths to all checkpoints
            if(evaluation_setup.checkpoints_eval):
                evaluation_setup.same_processor = True # must be same processor, no matter what was set in setup
                if (not isinstance(evaluation_setup.models,str)):
                    checkpoints_dir = evaluation_setup.models[0] # the first and hopefully one element of array
                else:
                    checkpoints_dir = evaluation_setup.models
                # load all checkpoints to be evaluated, sorted 
                evaluation_setup.models = sorted(
                    get_current_models_in_checkpoint(checkpoints_dir),
                    key=lambda x: int(x.split('-')[-1])
                )
                print(evaluation_setup.models)
            # go through each model, clear the memory and see the results
            first_model_in_serie = True

            # these setups can change over iterations through the list of models
            # when models are same, than this setup is stored and reused
            prep_ds_cls : PrepareDatasetAsInput     = None
            built_ds : dict[Dataset]|Dataset        = None
            metric : ComputeMetrics                 = None
            tokenizer_fr : WhisperTokenizer         = None

            for model_path in evaluation_setup.models:
                out_file.write(f'#### EVAL MODEL {model_path} ####\n')
                out_file.flush()

                print(f'*** Model path {model_path} ***')

                model, processor = setup_model_processor(model_path)

                if (not evaluation_setup.same_processor or first_model_in_serie):
                    tokenizer_fr = WhisperTokenizer.from_pretrained(model_path, language="French") # just for the french tokenizer
                    # above are the only conditions when new data parsing is necessary
                    # prepare new handler for dataset
                    prep_ds_cls = PrepareDatasetAsInput(processor.feature_extractor, processor.tokenizer, tokenizer_fr)
                    # set up the prompt and transcription names in the given dataset, that will be used
                    prep_ds_cls.set_transcription_name(evaluation_setup.transcription_name_in_ds)
                    prep_ds_cls.set_prompt_name(evaluation_setup.prompt_name_in_ds)

                    # choose function for dataset parsing
                    if evaluation_setup.use_prompt:
                        if evaluation_setup.self_prompt:
                            prepare_fn = prep_ds_cls.prepare_dataset_self_prompt
                        else:
                            prepare_fn = prep_ds_cls.prepare_dataset_with_prompt
                    else:
                        prepare_fn = prep_ds_cls.prepare_dataset

                    built_ds = build_dataset(
                        evaluation_setup.datasets,
                        prepare_fn,
                        evaluation_setup.datasets_basedir,
                        evaluation_setup.separate_ds
                    )

                    metric = ComputeMetrics(processor.tokenizer)

                # compute the wer and loss
                out_dict = compute(built_ds, model, processor, metric, batch_size=evaluation_setup.batch_size, use_prompt=evaluation_setup.use_prompt)

                # print results to the file
                result_printout(out_file,out_dict,evaluation_setup)

                # new lines at the end of the file
                out_file.write('\n\n')
                out_file.flush()

                # clean the memory
                cuda_clean(model)

                if first_model_in_serie:
                    first_model_in_serie = False

                # update all models ... that can change over time 
                if (evaluation_setup.checkpoints_eval):
                    new_models = sorted(
                        get_current_models_in_checkpoint(checkpoints_dir) - set(evaluation_setup.models),
                        key=lambda x: int(x.split('-')[-1])
                    )
                    if (len(new_models) > 0):
                        out_file.write(f'#### NEW CHECKPOINTS FOUND ####\n')
                        out_file.write(f'{new_models}\n')
                        
                    evaluation_setup.models.extend(new_models)
                    
            # close file
            out_file.close()
        else:
            # run the evaluation of single model
            # ==========================================
            model_path = evaluation_setup.models
            
            out_file.write(f'#### EVAL MODEL {model_path} ####\n')
            out_file.flush()
            print(f'*** Model path {model_path} ***')

            # setup model and the processor
            model, processor = setup_model_processor(model_path)
            tokenizer_fr = WhisperTokenizer.from_pretrained(evaluation_setup.models, language="French") # just for the french tokenizer
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

            # compute the wer and loss
            out_dict = compute(built_ds, model, processor, metric, batch_size=evaluation_setup.batch_size, use_prompt=evaluation_setup.use_prompt)

            result_printout(out_file, out_dict, evaluation_setup)

            # new lines at the end of the file
            out_file.write('\n\n')

            out_file.close()

    elif (evaluation_setup.metric == 'cer'):
        metric = cer

def parse_args():
    parser = argparse.ArgumentParser(description="Parse evaluation configuration")

    parser.add_argument('--setup', type=str, required=False, default=None, help='Type of evaluation metric')
    
    parser.add_argument("--metric", type=str, default="wer")
    parser.add_argument("--datasets", nargs='+')
    parser.add_argument("--datasets_basedir", type=str, default="./data/")
    parser.add_argument("--models", nargs='+')
    parser.add_argument("--checkpoints_eval", action="store_true")  # default is False
    parser.add_argument("--same_processor", action="store_true")
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--separate_ds", action="store_true")  # default is False
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_description", type=str, default="")
    parser.add_argument("--use_prompt", action="store_true")
    parser.add_argument("--self_prompt", action="store_true")
    parser.add_argument("--transcription_name_in_ds", type=str, default="full_ts")
    parser.add_argument("--prompt_name_in_ds", type=str, default="prompt_fullts_1G_4B")

    return parser.parse_args()

def build_config(args):
    return {
        "metric": args.metric,
        "datasets": args.datasets,
        "datasets_basedir": args.datasets_basedir,
        "models": args.models,
        "checkpoints_eval": args.checkpoints_eval,
        "same_processor": args.same_processor,
        "output_file": args.output_file,
        "separate_ds": args.separate_ds,
        "overwrite": args.overwrite,
        "batch_size": args.batch_size,
        "eval_description": args.eval_description,
        "use_prompt": args.use_prompt,
        "self_prompt": args.self_prompt,
        "transcription_name_in_ds": args.transcription_name_in_ds,
        "prompt_name_in_ds": args.prompt_name_in_ds
    }

if __name__ == '__main__':
    # parse args
    args = parse_args()
    
    # load the setup
    if args.setup is not None:
        with open(args.setup, 'r') as f:
            setup = json.load(f)
    else:
        setup = build_config(args)

    # load the setup
    evaluation_setup = EvaluationSetup(**setup)
    
    print('******** Evaluation setup ********')
    print(evaluation_setup.__str__())
    
    # run the evaluation
    main(evaluation_setup)