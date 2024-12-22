## SPLIT

-   TEST (callsigns in square brackets)

    -   UK dir (used: 49Fem[L0H]+49Male[H7V,O6N])
    -   CA dir (used: 121Fem[AW,M2D]+116Male[RT,CB])
    -   NL dir (used: 46Fem[PAVO,PAFF]+51Male[AMAA,31])
    -   DE dir (used: 60Male[2EZ,B6J,Y4B,A2W,7KD,V1Q,L2F]) (no female present in DE)

-   TRAIN

    -   UK dir (used: 259 wavs)
    -   CA dir (used: 614 wavs)
    -   NL dir (used: 272 wavs)
    -   DE dir (used: 228 wavs)

-   ALL
    -   original split is:
        -   UK dir (orig: 7 wavs, +ts 7)
        -   CA dir (orig: 15 wavs, +ts 15)
        -   NL dir (orig: 19 wavs, +ts 19)
        -   DE dir (orig: 445 wavs, +ts 445)

## DESC

In this dataset is used this number of wavs (same as speechs), which was made by splitting the original wavs according to the speechs

-   dataset UK: 357 speechs, 52.93m\*
-   dataset CA: 851 speechs, 106.48m\*
-   dataset NL: 369 speechs, 56.4m\*
-   dataset DE: 288 speechs, 27.9m\*

\* adduced minutes with noise included in wavs

## NOTES
- WAV LENGTH OK (<30s, for the own-cut files), the cuts are around the speech segments, no noise should be included there
    - original wavs are not used...
    
-   **!!! IMPORTANT !!!** only texts between synctimes \< 30s were converted and used, so the number of potential usable wavs are greater, but it will be hard to match the speaking text with the transcription

-   .sph files to .wav files converted through **sox** using command

```bash
$ for file in */*/*; do if [[ "${file##*.}" == "sph" ]]; then sox $file ${file%.*}.wav; fi; done
```
