#!/usr/bin/env bash


# Get and unpack bAbI
cd data
wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
tar -xzvf tasks_1-20_v1-2.tar.gz
rm tasks_1-20_v1-2.tar.gz

mv tasks_1-20_v1-2/* ./
rm -r tasks_1-20_v1-2

cd ..

# Create all seq2seq datasets

mkdir data/seq2seq
for TASK in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
	python3 2seq2seq.py --task=$TASK
done
