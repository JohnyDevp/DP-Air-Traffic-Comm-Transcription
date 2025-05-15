### AUTOR

Jan Holáň, xholan11
xholan11@fit.vutbr.cz

### Popis

Ve složce _tools_ se nachází

-   trojice slovníků, které sloužily pro zpracování volacích znaků, které se vyskytovaly v jednotlivých promluvách letecké komunikace. Vešekeré informace v nich obsažené byly vygenerovány pomocným skriptem _callsigns_extractor.py_ (lze spustit v dodaném virtuálním prostředí **venv312**) ze stránky ["https://en.wikipedia.org/wiki/List_of_airline_codes"]("https://en.wikipedia.org/wiki/List_of_airline_codes")
    -   airline_icao.json
    -   callsigns_icao.json
    -   icao_callsigns.json
-   slovník, který obsahuje seznam dostupných volacích značek a také jejich zkrácené formy
    -   global_vocab.json
-   pole náhodných českých slov, ze kterých byly generovány prompty, které měly za úkol předávat modelu slova zaručeně neobsažená v promluvách
    -   random_czech_words.json
-   lowercaser.py - pomocný nástroj (vygenerovaný modelem ChatGPT), který jako argument přijme pole json souborů (očekávaný formát je takový, v jakém jsou v této práci generované metadata soubory pro datasety) a převede vše, kromě pole _audio_ do **lowercase** formy.
    !!! Pozor, nástroj pracuje s jedním a týž souborem, který přepíše výsledkem !!!
    -   použití: `python lowercaser.py metadata_fr_test.json metadata_fr_train.json`
