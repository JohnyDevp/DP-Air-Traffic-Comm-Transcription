## AUTOR
Jan Holáň, xholan11
xholan11@fit.vutbr.cz

## SPUŠTĚNÍ
### POSTUP
Data ve složce A-PiMod byla ve formě dvou wav souborů, z nichž první (01_01_EL_LN_UJ_VV_YADA.wav) byl po prozkoumání takřka identický s druhým (01_02_EL_LN_UN_VV_YADA.wav). Navíc nebyly k dispozici ani žádné transkripce. Z toho důvodu je pro přesnou reprodukci vytváření datasetu nutné:
1. použít skript splitaudio.py (vygenerovaný ChatGPT)
2. použít skript makemetadata.py a následně makedataset.py
Popis skriptů a jejich použití následuje.
### splitaudio.py
- skript slouží pro rozdělení dlouhého audia na menší části podle *ticha*
- spuštění:
```bash
python splitaudio.py ROOT_FOLDER/A-PiMod/2013_10_Christoph/01_02_EL_LN_UN_VV_YADA.wav ROOT_FOLDER/A-PiMod/2013_10_Christoph/OUT_FOLDER
```
kde **ROOT_FOLDER** je cesta k adresáři A-PiMod. Dále samozřejmě, při změně struktury zbylé cesty, je potřeba provést úpravy. První argument udává cestu ke zpracovávanému souboru, druhý složku, kam bude uložen výstup.
- spuštění je doporučené v přiloženém virtuálním prosředí *venv312*

### makemetadata.py
- jeho spuštěním se vytvoří soubory obsahující metadata souborů určených k trénování a testování. Ve vyhotovených souborech se bude nacházet pole záznamů ve formátu *json* (pole "prompt" nebylo pro tento dataset vytvořeno):
```json
[
    {
        "audio": "/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/A-PiMod/2013_10_Christoph/01_02_EL_LN_UN_VV_YADA/recording_51.wav",
        "full_ts": "signing up and ready to copy skye two seven six skye two two seven six",
        "short_ts": "signing up and ready to copy skye 276 skye 2276",
        "prompt": null
    },
]
```

- pro správné spuštění skriptu editujte proměnnou **DISK_ROOT** v sekci pod podmínkou 
```python
if __name__ == "__main__":
```
na konci souboru, která ukazuje na místo, kde je uložena složka se surovými daty UWB. Je také potřeba zkontrolovat proměnnou **DIRS**, což je cesta přímo k vytvořeným wav souborům z originálního souboru pomocí splitaudio.py skriptu (použitý v předchozím kroku)

- dále je doporučené spouštět skript v přiloženém prostředí venv312, které obsahuje python3.12 a potřebné balíčky
- vytvořeny budou soubory:
    - metadata_train.json
- **VYTVOŘENÉ SOUBORY JSOU TRUECASE, K JEJICH PŘEVEDENÍ DO LOWERCASE FORMY JE TŘEBA POUŽÍT NÁSTROJ lowercaser.py**, který se nachází ve složce *tools* (vygenerovaný jazykovým modelem ChatGPT)

### makedataset.py
- jeho spuštěním vzniknou datasety, které mohou být načtené knihovnou *datasets* (může být použita funkce tohoto balíčku `load_from_disk()`)
- editujte proměnnou **DISK_ROOT** aby ukazovala na kořenový adresář se složkou se surovými daty UWB.
- dále je doporučené spouštět skript v přiloženém prostředí venv312, které obsahuje python3.12 a potřebné balíčky
- vytvořeny budou soubory:
    - apimod_train_ds


## ROZDĚLENÍ
-   TEST

    -   zde nevytvořeno

-   TRÉNINK

    -   105 wav souborů automaticky přepsáno pomocí modelu BUT-FIT/whisper-ATC-czech-full z HF

## POZNÁMKY

-   DÉLKA WAV souborů OK (<30s, pro vlastní rozdělení původního wav)
-   TICHO BY MĚLO BÝT VYLOUČENO, HLUK MOŽNÝ U POUŽITÝCH WAV
-   téměř 13,5 minuty
-   použity pouze rozdělené wav soubory ve složce 01_02_EL_LN_UN_VV_YADA, ve složce 01_01_EL_LN_UJ_VV_YADA vypadají wav soubory téměř stejně a je jich méně, takže nebyly využity
