### splitaudio.py

reads one audio file and splits it to multiple wav files according to the voice activity - long silence treated as separator between communications

### synthesize.py

reads **metadata.json** of some group of files and join them with the audio wav, resulting in saving datasetdict, consisting of train and test datasets, which items are of two values: [labels,input_features]
