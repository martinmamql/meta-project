#!/bin/bash

for i in 1 2 3 4 5 6 7 8 9
do 
    bash recola_arousal.sh
    bash recola_valence.sh
    bash sewa_arousal.sh
    bash sewa_valence.sh
    bash iemocap_arousal.sh
    bash iemocap_valence.sh
    bash mosi_sentiment.sh
    bash mosei_happiness.sh
    bash mosei_sentiment.sh
done

