 

## UPOZORNĚNÍ

VŠECHNY SKRIPTY PRACUJÍ IN-SITU. TEDY BĚHEM SPOUŠTĚNÍ VŠECH ŠESTI NÁSLEDUJÍCÍCH SKRIPTŮ SE PRACUJE POUZE SE SEDMI VYTVOŘENÝMI SOUBORY S METADATY. BĚHEM PRÁCE SKRIPTŮ DOJDE KE ZMĚNĚ STRUKTURY SLOŽKY S DATY!!!!

## SPUŠTĚNÍ

úprava ATCO2 datasetů je relativně náročná a prošla mnoha stupni. Každý skript je očíslován podle pořadí v jakém má být spuštěn pro korektní přípravu datasetů.

### 1_separating.py

Skript, jehož prostřednictvím jsou surová data rozdělená následovně

-   neanglická data ze složky DATA se přesouvají do DATA-data-nonEN
-   anglická data ze složky DATA_nonEN se přesouvají do DATA_nonEN-datanonen-EN

### 2_makemetadata.py

-   jeho spuštěním se vytvoří soubory obsahující metadata souborů určených k trénování a testování. Ve vyhotovených souborech se bude nacházet pole záznamů ve formátu _json_. Vytvořené pole prompt v sobě nese informace, které jsou v dalším skriptu použity pro vytvoření použitelných promptů:

```json
[
	{
		"audio": "ATCO2-ASRdataset-v1_final/DATA/LKPR_RUZYNE_Tower_134_560MHz_20201026_143955.wav",
		"full_ts": "Dream Team Four Zero Four Kilo taxi charlie lima and quebec    behind follow me car    to stand Sierra three",
		"short_ts": "GAC404K taxi CL 000 Q behind follow me car 00 1000 S 3",
		"prompt": {
			"waypoints": [],
			"short_callsigns": [],
			"long_callsigns": []
		}
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
-   **VYTVOŘENÉ SOUBORY JSOU TRUECASE, K JEJICH PŘEVEDENÍ DO LOWERCASE FORMY JE TŘEBA POUŽÍT NÁSTROJ lowercaser.py**, který se nachází ve složce _tools_ (vygenerovaný jazykovým modelem ChatGPT)

### makedataset.py

-   jeho spuštěním vzniknou datasety, které mohou být načtené knihovnou _datasets_ (může být použita funkce tohoto balíčku `load_from_disk()`)
-   editujte proměnnou **DISK_ROOT** aby ukazovala na kořenový adresář se složkou se surovými daty UWB.
-   dále je doporučené spouštět skript v přiloženém prostředí venv312, které obsahuje python3.12 a potřebné balíčky
-   vytvořeny budou soubory:
    -   hwir_fr_test_ds
    -   hwir_gr_test_ds
    -   hwir_sp_test_ds
    -   hwir_train_ds

## ROZDĚLENÍ DAT

-   **TEST**

    -   číslo v závorce udává počet surových nahrávek, číslo bez závorky potom počet nahrávek, které jsou zpracovávány v rámci trénování a testování (toto číslo je větší, protože některé nahrávky byly rozstříhané kvůli příliš velké délce)
    -   **RUZYNĚ** 70(50)/114(94) _(je zde několik dalších wavů (6), ale bez přepisu, takže jsou přeskočeny)_
    -   **ŠTEFÁNIK** 53/103(99)
    -   **ZÜRICH** 412/412 _(několik dalších wavů je bez přepisu (3))_
    -   **FR** 33/125(113)
    -   **JINÉ JAZYKY** 40/40

-   **TRÉNINK**
    -   **FR** 92(80)/125(113)
    -   **JINÉ JAZYKY** 0/0
    -   **EN TRÉNOVACÍ SADA** 1579(1554)
        -   **RUZYNĚ** 44/114(94)
        -   **ŠTEFÁNIK** 50(46)/103(99)
        -   **ZÜRICH** 0/0

### SLOŽKY v adresáři ATCO na disku

-   `_DATA_` data POUZE V ANGLIČTINĚ, odvozeno z `_DATA(Copy)-original_` (původní `_DATA_`)
-   `_DATA-original_` původní složka, původně pojmenována `_DATA_`
-   `_DATA-data-nonEN_` data z původního `_DATA_`, která NEJSOU v ANGLIČTINĚ
-   `_DATA-longer30s-cuts_` střihy wavů z původní složky `_DATA_`, které byly delší než 30 sekund
-   `_DATA_nonEN_` data NE v ANGLIČTINĚ, odvozená z `_DATA_nonEN(Copy)-original_` (původní `_DATA_nonEN_`)
-   `_DATA_nonEN-original_` původní složka, původně pojmenována `_DATA_nonEN_`
-   `_DATA_nonEN-datanonen-EN_` data z původního `_DATA_nonEN_`, která JSOU v ANGLIČTINĚ
-   `_DATA_nonEN-longer30s-cuts_` střihy wavů z původní složky `_DATA_nonEN_`, které byly delší než 30 sekund

\* **POZNÁMKA**: za anglické byly považovány wav soubory, kde více než polovina segmentů měla nastavený tag `<non_english>` na 0

### POZNÁMKY

-   mluvčí v testovací sadě pro ZÜRICH jsou **neznámí** v trénovací sadě
-   testovací sady RUZYNĚ a ŠTEFÁNIK jsou **částečně známé** z trénovací sady
-   některé wav soubory mají prázdné přepisy, takže jsou přeskočeny (ze ZÜRICH, RUZYNĚ, BRNO)
