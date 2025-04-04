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
