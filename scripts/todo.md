### WHAT IS HAPPENING NOW
- trenovani na colabu modelu base s testovanim attention masky, tady na notasu se testuje model base bez vyuziti attention masky

### TODO
zitra zjisti, jak vypada ten prompt_ids pri pouziti generate - potrebujes vedet, jak vlastne vypada ten prompt 

### WHAT TO DO
generate() potrebuje prompt urcite, a input_features ... kdyz toto pripravis v tomm datacollator,tak by i trainer evaluate mohl fungovat. nicmene realita je stejne takova, ze se ten model, respektive to generate, pro kazdy