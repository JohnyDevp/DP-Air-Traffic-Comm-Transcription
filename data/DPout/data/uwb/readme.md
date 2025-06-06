## AUTOR

Jan Holáň, xholan11
xholan11@stud.fit.vutbr.cz


## SPUŠTĚNÍ

### OBECNÉ UPOZORNĚNÍ
- Očekává se existence kořenové složky **UWB_ATCC** se surovými daty, ve které je očekávána struktura
    - **UWB_ATCC/audio**, složka obsahující audio 
    - **UWB_ATCC/stm/stm**, soubor s přepisy
- za proměnné typu DISK_ROOT se očekává dosazení **path/to/folder/containing/uwb_core_folder**, bez UWB_ATCC
- místo změn proměnné DISK_ROOT přímo ve skriptu je možné použít parametr skriptu
    - např.: `$ python nazevskriptu.py path/to/folder/containing/uwb_core_folder` 

### makemetadata.py

-   jeho spuštěním se vytvoří soubory obsahující metadata souborů určených k trénování a testování. Ve vyhotovených souborech se bude nacházet pole záznamů ve formátu _json_ (pole "prompt" nebylo pro tento dataset vytvořeno):

```json
[
	{
		"audio": "UWB_ATCC/audio_split/ACCU-07R4Pv_52.wav",
		"full_ts": "Lufthansa four juliet alpha one three two .",
		"short_ts": "Lufthansa 4JA132.",
		"prompt": null
	}
]
```

-   pro správné spuštění skriptu editujte proměnnou **DISK_ROOT** v sekci pod podmínkou na konci souboru

```python
if __name__ == "__main__":
```

, která ukazuje na místo, kde je uložena složka se surovými daty UWB. Je také potřeba zkontrolovat další proměnné v této sekci, které značí cesty k surovým datům UWB, zda skutečně existuje.

-   dále je doporučené spouštět skript v přiloženém prostředí venv312, které obsahuje python3.12 a potřebné balíčky
-   vytvořeny budou soubory:
    -   metadata_test.json
    -   metadata_train.json
-   **VYTVOŘENÉ SOUBORY JSOU TRUECASE, K JEJICH PŘEVEDENÍ DO LOWERCASE FORMY JE TŘEBA POUŽÍT NÁSTROJ lowercaser.py**, který se nachází ve složce _tools_ (vygenerovaný jazykovým modelem ChatGPT)

### makedataset.py

-   jeho spuštěním vzniknou datasety, které mohou být načtené knihovnou _datasets_ (může být použita funkce tohoto balíčku `load_from_disk()`)
-   editujte proměnnou **DISK_ROOT** aby ukazovala na kořenový adresář se složkou se surovými daty UWB.
-   dále je doporučené spouštět skript v přiloženém prostředí venv312, které obsahuje python3.12 a potřebné balíčky
-   vytvořeny budou soubory:
    -   uwb_test_ds
    -   uwb_train_ds

## ROZDĚLENÍ

-   TEST (3961 řečí, 650 wavů)

    -   v souboru _test_wavs.out_ najdete seznam wav souborů (původních) s řečmi, které mají být v testovací sadě; první sloupec ukazuje původní název wav souboru, druhý ukazuje počet použitelných řečí v něm (počet souborů, které budou vytvořeny z původního wavu)

-   TRAIN (10706 řečí, 2006 wavů)
    -   všechny ostatní soubory, které nejsou v _test_wavs.out_

## POPIS

-   celkový počet stm přepisů (řádky v _stm_ souboru) je 30205, 15537 je prázdných, 1 je příliš dlouhý v audionahrávce (>30s)
-   celkový počet použitelných přepisů (řečí) je 14667, celkový počet použitelných původních wavů je 2656
-   3961 testovacích řečí, 10706 trénovacích řečí

## POZNÁMKY

-   DÉLKA WAVŮ V POŘÁDKU (<30s, pro vlastnoručně vystřižené wavy)

    -   původní wavy nebyly použity, střih byl proveden podle časových značek v _stm_ souboru s kontrolou délky <30s

-   **!!! VAROVÁNÍ !!!** přepisy (_stm_ soubor) byly směsí plných forem slov a zkrácenin (např. čísla jako číslice, ale ne všechna, např. "1 thousand 5 hundred")

    -   problém nastává zejména u desetinných čísel, která mohou být vyslovována mnoha způsoby
    -   u zpětného překladu se očekává angličtina
