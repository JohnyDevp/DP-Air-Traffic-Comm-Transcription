## SPLIT DATA

-   TEST

    -   RUZYNE 47/78 (there are a few more wavs (6), but without transcription, so they are skipped)
    -   STEFANIK 50/88
    -   ZURICH 413/413 (a few more wavs are without ts (3))

-   TRAIN

    -   everything, what is not in test

-   ALL
    2117 wavs (from those 36 has \<non_english\> tag set to 1 and 8 are longer than 30s)

## SPLIT DATA_nonEN

-   TEST

    -   RUZYNE 47/78 (there are a few more wavs (6), but without transcription, so they are skipped)
    -   STEFANIK 50/88
    -   ZURICH 413/413 (a few more wavs are without ts (3))

-   TRAIN

    -   everything, what is not in test

-   ALL
    -   158 wavs (2 are longer than 30s, SION ds)

### DIRS in ATCO folder on disc

-   _DATA_ data ONLY IN ENGLISH, derived from _DATA(Copy)-original_ (original _DATA_)
-   _DATA(Copy)-original_ original folder, originally named _DATA_
-   _DATA-data-nonEN_ data from original _DATA_, that are NOT in ENGLISH
-   _DATA-longer30s-cuts_ cuts of wavs from the originally _DATA_ folder, that were longer than 30s
-
-   _DATA_nonEN_ data NOT IN ENGLISH, derived from _DATA_nonEN(Copy)-original_ (original _DATA_nonEN_)
-   _DATA_nonEN(Copy)-original_ original folder, originally named _DATA_nonEN_
-   _DATA_nonEN-datanonen-EN_ data from original _DATA_nonEN_, that are IN ENGLISH
-   _DATA_nonEN-longer30s-cuts_ cuts of wavs from the originally _DATA_nonEN_ folder, that were longer than 30s

\* NOTE: as english were considered wavs, with more than half segments containing <non_english> tag set to 0

### NOTES

-   the speakers in test set for ZURICH are **unseen** (not in train!)
-   the RUZYNE and STEFANIK test sets are **partly seen** in train
-   a few wavs has empty transcripts, so they are skipped (from ZURICH, RUZYNE, BRNO)
