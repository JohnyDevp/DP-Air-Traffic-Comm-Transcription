## AUTOR
Jan Holáň, xholan11
xholan11@fit.vutbr.cz

## SPUŠTĚNÍ
### makemetadata.py
- jeho spuštěním se vytvoří soubory obsahující metadata souborů určených k trénování a testování. Ve vyhotovených souborech se bude nacházet pole záznamů ve formátu *json* (pole "prompt" nebylo pro tento dataset vytvořeno):
```json
[
    {
        "audio": "n4_nato_speech_LDC2006S13/data/CA/CA_Audio_Speech_Segments/0_C2B_male.wav",
        "full_ts": "this is charlie two bravo, roger out",
        "short_ts": "this is C2B, roger out",
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
    - metadata_UK_test.json
    - metadata_UK_train.json  
    - metadata_NL_test.json
    - metadata_NL_train.json  
    - metadata_DE_test.json
    - metadata_DE_train.json  
    - metadata_CA_test.json
    - metadata_CA_train.json  

### makedataset.py
- jeho spuštěním vzniknou datasety, které mohou být načtené knihovnou *datasets* (může být použita funkce tohoto balíčku `load_from_disk()`)
- editujte proměnnou **DISK_ROOT** aby ukazovala na kořenový adresář se složkou se surovými daty UWB.
- dále je doporučené spouštět skript v přiloženém prostředí venv312, které obsahuje python3.12 a potřebné balíčky
- vytvořeny budou soubory:
    - nato_test_ds
    - nato_train_ds

## ROZDĚLENÍ

-   TEST (volací znaky, podle kterých došlo k rozdělení, jsou v hranatých závorkách)

    -   UK složka (použito: 49Žen[L0H]+49Mužů[H7V,O6N])
    -   CA složka (použito: 121Žen[AW,M2D]+116Mužů[RT,CB])
    -   NL složka (použito: 46Žen[PAVO,PAFF]+51Mužů[AMAA,31])
    -   DE složka (použito: 60Mužů[2EZ,B6J,Y4B,A2W,7KD,V1Q,L2F]) (žádné ženy v DE nejsou)

-   TRÉNINK

    -   UK složka (použito: 259 wav)
    -   CA složka (použito: 614 wav)
    -   NL složka (použito: 272 wav)
    -   DE složka (použito: 228 wav)

-   CELKEM
    -   původní rozdělení je:
        -   UK složka (původně: 7 wav, +ts 7)
        -   CA složka (původně: 15 wav, +ts 15)
        -   NL složka (původně: 19 wav, +ts 19)
        -   DE složka (původně: 445 wav, +ts 445)

## POPIS

V tomto datasetu je použito toto množství wav (shodné jako počet řečí), které byly vytvořeny rozdělením původních wav podle projevů:

-   dataset UK: 357 projevů, 52.93m\*
-   dataset CA: 851 projevů, 106.48m\*
-   dataset NL: 369 projevů, 56.4m\*
-   dataset DE: 288 projevů, 27.9m\*

\* uvedené minuty zahrnují i šum obsažený ve wavech

## POZNÁMKY
- DÉLKA wav V POŘÁDKU (<30s pro vlastnoručně vystřižené soubory), střihy jsou kolem řečových segmentů, neměl by být zahrnut žádný šum ve výsledných nahrávkách
    - původní wav soubory nejsou použity...
    
-   **!!! DŮLEŽITÉ !!!** byly převedeny a použity pouze texty mezi synctimes \< 30s, takže počet potenciálně použitelných wav je větší, ale bylo by obtížné přiřadit mluvený text k přepisu (nutno případně využít časové značky)

-   .sph soubory byly převedeny na .wav soubory pomocí **sox** příkazem:

```bash
$ for file in */*/*; do if [[ "${file##*.}" == "sph" ]]; then sox $file ${file%.*}.wav; fi; done
```
