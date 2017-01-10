#!/bin/bash
#
# This shell script does the following
# 1) Saves randomly generated partion of data
# 2) Trains model on data
# 3) Tests model on testing subset of data
# 4) Prints error analysis results


module load tensorflow/0.12.1
source ../../tensorflow/bin/activate

func ()
{
local arch=cnn_word
local partition=human_annot_only
local name=${arch}_${partition}_impression

#python create_partition.py --partition $partition
<<COMMENT
python classifier.py \
    --runtype train \
    --arch $arch \
    --partition $partition \
    --name $name
COMMENT

python classifier.py \
    --runtype test \
    --arch $arch \
    --partition $partition \
    --name $name \
    -error_analysis
}

func
