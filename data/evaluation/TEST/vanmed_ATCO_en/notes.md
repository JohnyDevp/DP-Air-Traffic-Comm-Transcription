### Moje poznamky
jedna se o whisper medium, natrenovany na 15+15 epoch datasetu ATCO EN cast.
nebyl pouzity prompt ani nic dalsiho
btw vychazi se (tzn 1. iterace byla trenovana) z toho, jak jsem trenoval jeste s domenkou, ze attention maska je normalne pouzivana, akorat ze neni. Jeji pouziti je mozne pouze kdyz se v konfiguraci modelu povoli data augmentation, coz je defaultne false. Whisper implicitne neumoznuje maskovani vstupu. A u decoder ids neni maska treba, tam je pouzito pro ty labely -100 tokeny.