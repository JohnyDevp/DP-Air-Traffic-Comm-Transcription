## SPLIT
- TEST
    - CA dir (15 wavs, +ts 15)
    - DE dir (445 wavs, +ts 445)
    - NL dir (19 wavs, +ts 19)
    - UK dir (7 wavs, +ts 7)
- TRAIN
    - CA dir (15 wavs, +ts 15)
    - DE dir (445 wavs, +ts 445)
    - NL dir (19 wavs, +ts 19)
    - UK dir (7 wavs, +ts 7)

- ALL 
    - 486 wavs used


## NOTES
-   .sph files to .wav files converted through **sox** using command 
```bash
$ for file in */*/*; do if [[ "${file##*.}" == "sph" ]]; then sox $file ${file%.*}.wav; fi; done
```
