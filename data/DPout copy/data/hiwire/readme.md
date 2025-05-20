## AUTOR

Jan Holáň, xholan11
xholan11@stud.fit.vutbr.cz

## SPUŠTĚNÍ

### makemetadata.py

-   jeho spuštěním se vytvoří soubory obsahující metadata souborů určených k trénování a testování. Ve vyhotovených souborech se bude nacházet pole záznamů ve formátu _json_ (pole "prompt" nebylo pro tento dataset vytvořeno):

```json
[
	{
        "audio": "HIWIRE_ELDA_S0293/speechdata/LN/FR/FMHF/FMHF_012_LN.wav",
        "full_ts": "select vhf3 one two three",
        "short_ts": "select vhf3 123",
        "prompt": null
    }
]
```

-   pro správné spuštění skriptu editujte proměnnou **ROOT_DIR** v sekci pod podmínkou

```python
if __name__ == "__main__":
```

na konci souboru, která ukazuje na místo, kde je uložena složka se surovými daty UWB. Je také potřeba zkontrolovat další proměnné (disk_path_to_be_excluded, FOLDER_NAME), zda odpovídají požadavkům - jejich použití je vysvětleno v komentářích a k jejich změně by nemělo dojít.

-   dále je doporučené spouštět skript v přiloženém prostředí venv312, které obsahuje python3.12 a potřebné balíčky
-   vytvořeny budou soubory:
    -   metadata_hwir_fr_test.json
    -   metadata_hwir_gr_test.json
    -   metadata_hwir_sp_test.json
    -   metadata_hwir_train.json
- **VYTVOŘENÉ SOUBORY JSOU TRUECASE, K JEJICH PŘEVEDENÍ DO LOWERCASE FORMY JE TŘEBA POUŽÍT NÁSTROJ lowercaser.py**, který se nachází ve složce *tools* (vygenerovaný jazykovým modelem ChatGPT)
### makedataset.py

-   jeho spuštěním vzniknou datasety, které mohou být načtené knihovnou _datasets_ (může být použita funkce tohoto balíčku `load_from_disk()`)
-   editujte proměnnou **DISK_ROOT** aby ukazovala na kořenový adresář se složkou se surovými daty UWB.
-   dále je doporučené spouštět skript v přiloženém prostředí venv312, které obsahuje python3.12 a potřebné balíčky
-   vytvořeny budou soubory:
    -   hwir_fr_test_ds
    -   hwir_gr_test_ds
    -   hwir_sp_test_ds
    -   hwir_train_ds

## ROZDĚLENÍ

-   TEST
    -   FR 800/3100 (4Muži, 4Ženy)
    -   GR 400/1433 (2Muži, 2Ženy)
    -   SP 300/999 (2Muži, 1Žena)
-   TRÉNINK

    -   vše, co není v testu

-   CELKEM
    -   5532 wavů

## POZNÁMKY

-   DÉLKA WAV souborů V POŘÁDKU (<30s, pro původní ... řečové segmenty nejsou zahrnuty, tzn. že ve wavech použitých pro trénink může být šum)
-   MOŽNÝ ŠUM V POUŽITÝCH WAV SOUBORECH
-   V dodaných datech dostupná pouze složka LN, tudíž podle dokumentace by měly být dostupné další složky
