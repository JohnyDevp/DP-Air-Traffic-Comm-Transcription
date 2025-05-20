## AUTOR

Jan Holáň, xholan11
xholan11@stud.fit.vutbr.cz

# POPIS

V jednotlivých složkách se nacházejí skripty pro přípravu dat. Je nutno poznamenat, že nejsou uživatelsky přívětivé. Všechny skripty počítají s organizací jednotlivých složek s daty takovou, jaká byla dodána pro diplomovou práci.

Každý skript vyžaduje v zásadě **pouze změnu** cesty ke kořenové složce, ve které jsou obsaženy složky s jednotlivými datasety

Skripty v jednotlivých složkách jsou očíslovány v takovém pořadí, v jakém je očekáváno jejich spuštění.

# PROSTŘEDÍ

Předpokládané použití skriptů je očekáváno v prostředí python3.12 s balíčky definovanými ve složce env

# OBECNÉ DOPORUČENÍ PRO SPOUŠTĚNÍ

Pro každý skript je poznamenané, které proměnné bude třeba změnit. Pokud zůstává struktura adresáře stejná, bude vždy stačit změnit proměnnout DISK_ROOT, která vede k adresáři, který obsahuje všechny složky s jednotlivými datasety

-   místo změn proměnné DISK_ROOT přímo ve skriptu je možné použít parametr skriptu
    -   např.: `$ python nazevskriptu.py path/to/folder/containing/uwb_core_folder`, vždy ale s cestou bez konečného názvu složky

# OČEKÁVANÉ NÁZVY JEDNOTLIVÝCH SLOŽEK S DATASETY

-   pokud jsou tyto složky přejmenované, je potřebné najít patřičná místa ve skriptech a přejmenovat je tam

-   ATCO2-ASRdataset-v1_final
-   HIWIRE_ELDA_S0293
-   MALORCA
-   n4_nato_speech_LDC2006S13
-   UWB_ATCC
