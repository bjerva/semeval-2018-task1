#!/usr/bin/env python

from sys import argv
import os.path

base = r'''#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task 2
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3GB
#SBATCH --job-name={0}
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=wqx727@alumni.ku.dk

srun python3 -O semeval2018 \
        --train data/2018-EI-reg-En-anger-train.txt data/2018-EI-reg-En-fear-train.txt data/2018-EI-reg-En-joy-train.txt data/2018-EI-reg-En-sadness-train.txt  \
        --dev data/2018-EI-reg-En-anger-dev.txt data/2018-EI-reg-En-fear-dev.txt data/2018-EI-reg-En-joy-dev.txt data/2018-EI-reg-En-sadness-dev.txt --test data/2018-EI-reg-En-anger-dev.txt data/2018-EI-reg-En-fear-dev.txt data/2018-EI-reg-En-joy-dev.txt data/2018-EI-reg-En-sadness-dev.txt \
        --tag dropout_{0} \
        --rnn \
        --dropout {0} \
        --epochs 10 \
        --char-embedding-dim 400 \
        --rnn-dim 250 \
        --resnet 4 \
        --embeddings embeddings.txt \
        --aux data/2018-E-c-En-train.txt data/2018-E-c-En-dev.txt \
        --word-embedding-dim 400 \
        --bn \
        --words \
        --chars \
        --early-stopping 4 \
        > /dev/null 2>&1
'''


dropouts = [0.2, 0.4, 0.6]
#for i, language in enumerate(languages):
for j, dropout in enumerate(dropouts):
    #if size == 'high' and language == 'scottish-gaelic':
    #    continue # No 'high' data'
    #if not os.path.exists('./all/task1/{0}-train-{1}'.format(language, size)):
    #    continue
    with open('./slurms/{0}.slrm'.format(dropout), 'w') as out_f:
        out_f.write(base.format(dropout))
