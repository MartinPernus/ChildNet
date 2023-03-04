#!/bin/bash

mkdir -p checkpoints
cd checkpoints

echo "Downloading pretrained models ..."
wget -O checkpoints.tar.xz https://www.dropbox.com/s/p33r4091ynjt869/checkpoints.tar.xz?dl=1

echo "Extracting ..."
tar -xf checkpoints.tar.xz 
rm checkpoints.tar.xz

cd ..
echo "Done!"
