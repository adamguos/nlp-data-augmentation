#!/bin/bash
dir="eda_nlp/data"
for i in 50 100 500 1000 1456 ;
do for j in 0 1 2 3 4 5 6 7 8 9;
	do python eda_nlp/code/augment.py --input=$dir/reddit_$i/$j.txt;
    done;
done;
