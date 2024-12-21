## SPLIT

-   TEST (3961 speeches, 650 wavs)

    -   in file _test_wavs.out_ you can find list of wav files (original) with the speeches, that are supposed to be in test set; first column shows the original wavname, second shows number of usable speeches in it (the number of files that will be created from the original wav)

-   TRAIN (10706 speeches, 2006 wavs)

## DESC

-   total stm transcriptions (lines in _stm_ file) is 30205, 15537 are blank, 1 is too long in audio (>30s)
-   total usable transcriptions (speeches) are 14667, total usable original wavs 2656
-   3961 test speeches, 10706 train speeches

## NOTES

-   **!!! WARNING !!!** the transcripts (_stm_ file)were mix of shorts (like numbers as digits, but not all of them, e.g. "1 thousand 5 hundred")

    -   the problem comes especially with the decimal points, which can be pronounced in many ways
    -   in the reverse translation is expected english
