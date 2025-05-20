## AUTOR
Jan Holáň, xholan11
xholan11@stud.fit.vutbr.cz

    

## SPUŠTĚNÍ
### OBECNÉ UPOZORNĚNÍ
- Očekává se existence kořenové složky **MALORCA** se surovými daty, ve které je očekávána struktura
    - **MALORCA/DATA_ATC/VIENNA/DATA/dev12**, složka obsahující informace rozdělení trénovací sady
    - **MALORCA/DATA_ATC/VIENNA/DATA/test**, složka obsahující informace rozdělení testovací sady  
    - **MALORCA/DATA_ATC/VIENNA/WAV_FILES**, složka s audii a přepisy
- za proměnné typu DISK_ROOT se očekává dosazení **path/to/folder/containing/MALORCA_core_folder**, bez MALORCA
- místo změn proměnné DISK_ROOT přímo ve skriptu je možné použít parametr skriptu
    - např.: `$ python nazevskriptu.py path/to/folder/containing/MALORCA_core_folder` 

### makemetadata.py
- jeho spuštěním se vytvoří soubory obsahující metadata souborů určených k trénování a testování. Ve vyhotovených souborech se bude nacházet pole záznamů ve formátu *json* (pole "prompt" nebylo pro tento dataset vytvořeno):
```json
[
    {
        "audio": "MALORCA/DATA_ATC/VIENNA/WAV_FILES/LOWW02/BALAD_20160615_B3/2016-06-15__10-44-12-29.wav",
        "full_ts": "speedbird six nine seven no atc speed limit",
        "short_ts": "baw697 no atc speed limit",
        "prompt": null
    }
]
```

- pro správné spuštění skriptu editujte proměnnou **DISK_ROOT** v sekci pod podmínkou 
```python
if __name__ == "__main__":
```
na konci souboru, která ukazuje na místo, kde je uložena složka se surovými daty UWB. Je také potřeba zkontrolovat další proměnné (pole *inputs*) v této sekci, které značí cesty k surovým datům UWB, zda skutečně tyto cesty existují.

- dále je doporučené spouštět skript v přiloženém prostředí venv312, které obsahuje python3.12 a potřebné balíčky
- vytvořeny budou soubory:
    - metadata_dev12.json
    - metadata_test.json
- **VYTVOŘENÉ SOUBORY JSOU TRUECASE, K JEJICH PŘEVEDENÍ DO LOWERCASE FORMY JE TŘEBA POUŽÍT NÁSTROJ lowercaser.py**, který se nachází ve složce *tools* (vygenerovaný jazykovým modelem ChatGPT)

### makedataset.py
- jeho spuštěním vzniknou datasety, které mohou být načtené knihovnou *datasets* (může být použita funkce tohoto balíčku `load_from_disk()`)
- editujte proměnnou **DISK_ROOT** aby ukazovala na kořenový adresář se složkou se surovými daty UWB.
- dále je doporučené spouštět skript v přiloženém prostředí venv312, které obsahuje python3.12 a potřebné balíčky
- vytvořeny budou soubory:
    - malorca_test_ds
    - malorca_train_ds


## ROZDĚLENÍ
- MALORCA již byla rozdělena na test a train set; zde je pouze souhrn a odkaz na jednotlivé složky
-   TEST
    -   TEST složka (1557 wav souborů)

-   TRÉNINK

    -   DEV\_1 složka (3987)
    -   DEV\_2 složka (2348)

-   CELKEM
    -   použito 7892 wav
    -   dalších 8037 wav zůstává bez časových značek ve složce DEV_3

## POZNÁMKY

-   DÉLKA wav souborů v pořádku (<30s, jeden soubor má 31.49s _LOWW38/BALAD_20160924_E3/2016-09-24\_\_10-24-47-31.wav_)
-   možný šum v použitých nahrávkách (důvod viz bod níže)
-   POUŽITÉ WAVY JSOU PŮVODNÍ – žádné střihy podle řečových segmentů, protože ty nebyly nikde specifikovány

-   DEV\_3 složka malorca dataset nebyl použit
