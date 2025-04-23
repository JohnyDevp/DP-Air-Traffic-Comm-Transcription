WORKING ON FULLTS

-   WER na vetach
    -   odkud zlepseni (jen callsign, jina mista ve vete)
    -   AG4B vs zadny prompt
-   natrenovat na spatnych a evaluace na dobry a obracene
-   zkusit nahodna slova (uplna nahoda)
-   PROMPT otestuj bez PROMPTU
-   shortts -> smaz entry odevsad, .,? smazat , lower case mezery

### AZ SE VRATIS TAK MUSIS

-   pridej vsude ignore_case, obecne nechceme evaluovat lower/upper case

### FULLTS VANILLA MEDIUM, experiments of 5 parameters

-   hotovo: learning rate, warmup, dropout
-   aktualni stav:

### FULLTS ALLDS MEDIUM, experiments of 5 parameters

-   hotovo: learning rate, warmup
-   aktualni stav: dropout train

### FULLTS VANILLA MEDIUM, experiments with MALORCA+ATCO

-   hotovo: 30 epoch ev+tr
-   aktualni stav: NO CONTINUE

### FULLTS ALLDS MEDIUM, experiments with MALORCA+ATCO

-   hotovo: 30 epoch ev+tr
-   aktualni stav: NO CONTINUE

# WORKING ON SHORTTS

### SHORTTS VANILLA MEDIUM, finetune for 8 epochs of alldata

-   hotovo: train
-   aktualni stav: eval

## NAZEV KAZDE SLOZKY

je vzdy ve formě [startovací model]_[na jakem DS se trenovalo]_[pripadne vymezeni jazyka]\_[dodatek]
dodatek muze byt

-   nop - no prompt (implicitne True)
-   p - prompt (implicitne False)
-   atmask - pouzita attention maska pro augmentaci (implicitne False)

## MOJE POZNAMKY K JEDNOTLIVYM SOUBORUM

### allds-atco-nop

whisper med natrenovany na vsem krome UWB, otestovany na celem ATCO, bez promptu a bez masky

### atco-atco-nop

whisper med natrenovany na celem ATCO, otestovany na celem ATCO, bez promptu a bez masky
chybi tam otestovane nejake prostredni epochy, ale asi je to jedno

### NEWatco-atco-nop_loss_wer.png

jen obrazek z nove natrenovaneho whisper medium na celem ATCO a otestovanem na celem ATCO, chova se to dost podobne jako bez toho

### atco_eval_atco_NOmask.txt

ponechan dataset ATCO jen EN (cizojazycne nepouzity) pro trenovani i testovani
nevyuzita attention_mask, kvuli tomu to i zaroven bylo trenovane (soubezne byl trenovan atco_eval_atco_wmask.txt)

### atco_eval_atco_wmask.txt

to same jako atco_eval_atco_NOmask.txt, prave s jedinym rozdilem, ze byla pouzita attention_mask, zda se ze bez zmeny
