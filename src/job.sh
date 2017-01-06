#!/bin/bash

module load tensorflow/0.9.0
source ../../tensorflow/bin/activate

func ()
{
local arch=cnn_word
local name=$arch

#python classifier.py --runtype train --arch $arch --name $name
python classifier.py --runtype test --arch $arch --name $name
}

func
