#!/usr/bin/env bash
python code/train.py --bs=8 --lr=1e-3 --epoch=20

python code/inference.py --bs=8
