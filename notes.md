## QUESTIONS 16.4.

Budou stacit zkracene prepisy bez experimentu? Ci s nima?

## QUESTIONS 9.4.

Jak se vyporadat s tim ze mi nejde pouzit CUDA na clusteru... pritom mi to fungovalo

Experimenty hotove - natrenovane VANMED i VANMED_ALLDS na ATCO_EN, lr: 1e-5, wr:0.12 - viz grafy - batch size:gas pro ALLDS na 4:20 - lepsi rust vykazuje rozhodne ten ALLDS, ale otazka no... - natrenovany VANMED na ATCO_en, lr: 6.25e-6, wr: 0.09... lr vychazi z clanku, 40x mensi - zmena batch size na - bohuzel neotestovany, protoze cluster... - natrenovany VANMED na ATCO_en, lr: 6.25-6, wr: 0.12, zmena jen u lr - testovani v procesu

Jak se chovat s tou lr, wr, pripadne batchsize/gas, dropout

Attention maska nefunguje, pouze pro data augmentation... tak otazka jestli taky zkusit experiment

Kdyz se podivam, tak MALORCA je naucena perfektne, jen 15 epoch, vynikajici vysledky u sebe, co s tim - mam ziskat prompt i pro malorcu - mam ji pouzivat pro trenink, neni to ATCO male? (jen 3h trenink)
Zde take otazka teda, jestli se vubec da naucit to ATCO lepe, xnevar00 taky prezentovala nauceni se na ATCO datech a WER 19, ale kdyz jsem predhodil ta svoje data, tak WER byla kolem 40... takze nepouzitelne

Uz zacit s tim promptem? Mam extrahovane bezpecne CALLSIGN a RUNWAY.

Co short forma? Bude stacit pro splneni zadani to jen natrenovat a vyhodnotit treba s nejlepsimi parametry pro full prompt? A callsignu uz se venovat jen na full casti?

## TODO

-   make SGE script
-   run evaluation
-   run training for shortts

# Literature to be studied

https://arxiv.org/abs/2212.04356 -- robust speech recognition  
https://arxiv.org/abs/1706.03762 -- attention is all you need  
https://cdn.openai.com/papers/whisper.pdf -- noise robust models  
https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-024-00349-3 -- explanation on training whisper with low resources

https://dspace.mit.edu/bitstream/handle/1721.1/145275/BadrinathBalakrishnan-TRR2021.pdf?sequence=2&isAllowed=y -- air traffic control transcription

https://aim.rlp.cz/vfrmanual/actual/enr_6_en.html -- radiotelephony phraseology

### book of natural language processing

https://web.stanford.edu/~jurafsky/slp3/ed3bookaug20_2024.pdf

### prompting whisper texts

https://arxiv.org/html/2312.08079
https://arxiv.org/pdf/2406.02649

# current write ideas

-   kompozice whisperu (viz poznamky v tabletu)

# todo project ideas

-   stahnout model whisperu :heavy_check_mark:
-   dostudovat whisper clanek
-   nacist vic o tranformerech
