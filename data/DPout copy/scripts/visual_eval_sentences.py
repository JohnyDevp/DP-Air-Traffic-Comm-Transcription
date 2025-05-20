import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import argparse, os 
from datasets import load_from_disk, concatenate_datasets, Dataset

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

        # STORE THE CALLSIGNS FOR EVALUATION
        if ("prompt-data" in batch):
            batch['long_callsigns'] =  batch['prompt-data']['long_callsigns']
            batch['short_callsigns'] =  batch['prompt-data']['short_callsigns']
        else :
            batch['long_callsigns'] = {}
            batch['short_callsigns'] = {}
            
        # encode target text to label ids **** CHANGED FROM **sentence** TO **transcription**
        # if french, than use french tokenizer, english otherwise
        tokenizer = self.tokenizer_en
        if "lang" in batch:
            if batch["lang"] == "fr":
                tokenizer = self.tokenizer_fr

        batch['labels'] = tokenizer(batch[self.transcription_name]).input_ids
        batch[self.transcription_name] = batch[self.transcription_name] # add the full transcription to the batch
        
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

        # STORE THE CALLSIGNS FOR EVALUATION
        if ("prompt-data" in batch):
            batch['long_callsigns'] = batch['prompt-data']['long_callsigns']
            batch['short_callsigns'] = batch['prompt-data']['short_callsigns']
        else :
            batch['long_callsigns'] = {}
            batch['short_callsigns'] = {}
            
        # encode target text to label ids **** CHANGED FROM **sentence** TO **transcription**
        # if french, than use french tokenizer, english otherwise
        tokenizer = self.tokenizer_en
        if "lang" in batch:
            if batch["lang"] == "fr":
                tokenizer = self.tokenizer_fr

        # encode prompts to prompt ids - we assume that the dataset has a column `"prompt"` that contains the prompt for each example
        prompt_ids = tokenizer.get_prompt_ids(batch[self.prompt_name]).tolist() # YOU NEED TO ADD TOLIST() because array cant be combined with list in the next lines

        batch["labels"] = prompt_ids + tokenizer(batch[self.transcription_name]).input_ids # building labels ids with prompt and tokens together
        batch['prompt_ids'] = prompt_ids # add the prompt ids to the batch
        batch['transcription'] = batch[self.transcription_name] # add the full transcription to the batch
        return batch

  
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
    for key,ds in allds_test.items():
        allds_test[key] = ds.map(prepare_dataset_fn, remove_columns=ds.column_names, num_proc=1)
    
    # return either concatenated datasets or list of datasets
    if (separate_ds):
        return allds_test
    else:
        return concatenate_datasets([allds_test[key] for key in allds_test])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Visualize the evaluation of sentences.')
    argparser.add_argument('--model', type=str, required=True, help='Model name or path')
    argparser.add_argument('--dataset', nargs='+', required=True, help='List of dataset names to transcribe. Possible only [atco_en_ruzyne, atco_en_stefanik, atco_en_zurich, atco_fr, atco_other_lang, hiwire_fr, hiwire_gr, hiwire_sp, malorca, nato, uwb]')
    argparser.add_argument('--dataset_root', type=str, required=True, help='Path to the dataset root')
    argparser.add_argument('--output_file', type=str, required=True, help='Output directory for the results')
    argparser.add_argument('--use_prompt', action='store_true', required=False, help='Whether to use prompt for evaluation')
    argparser.add_argument('--prompt_name', type=str, required=False, help='Prompt name to use for evaluation')
    argparser.add_argument('--transcription_name', type=str, required=True, help='Transcription name to use for evaluation')
    args=argparser.parse_args()
    
    # build the datasets
    processor = WhisperProcessor.from_pretrained(args.model)
    tokenizer_en = processor.tokenizer
    pd = PrepareDatasetAsInput(processor.feature_extractor, tokenizer_en, tokenizer_en)
    pd.set_transcription_name(args.transcription_name)
    if (args.use_prompt):
        pd.set_prompt_name(args.prompt_name) 
        datasets_dict = build_dataset(args.dataset,pd.prepare_dataset_with_prompt, args.dataset_root, separate_ds=True)
    else:
        datasets_dict = build_dataset(args.dataset,pd.prepare_dataset, args.dataset_root, separate_ds=True)
    
    # setup output file
    if os.path.dirname(args.output_file) != '' and not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))
    file = open(args.output_file,'a')
    
    # load the model up
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.to('cuda')
    for name, ds in datasets_dict.items():
        file.write(f"Evaluating {name} dataset\n")
        file.write(args.__str__() + '\n')
        file.write('**************************\n')
        for item in tqdm(ds):
            prompt_ids = torch.tensor(item['prompt_ids']).cuda()
            input_features = torch.tensor(item['input_features']).unsqueeze(0).cuda()
            preds=model.generate(input_features=input_features,prompt_ids=prompt_ids)
            file.write(f"O: {item['transcription']}\nP: {processor.decode(preds[0], skip_special_tokens=True)}\n\n")
            
        file.write('\n##############################################################\n\n')
    file.close()