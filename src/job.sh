#!/bin/bash

module load tensorflow/0.12.1
source ../../tensorflow/bin/activate

func ()
{
local arch=cnn_word
local partition=human_annot_only
local name=${arch}_${partition}_impression

#python create_partition.py --partition $partition

python classifier.py \
    --runtype train \
    --arch $arch \
    --partition $partition \
    --name $name

python classifier.py \
    --runtype test \
    --arch $arch \
    --partition $partition \
    --name $name
}

func
