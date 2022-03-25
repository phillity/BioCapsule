#!/bin/bash

echo 'Downloading and extracting VGGFace2_Visulaize, LFW and RS features!'

wget https://cs.iupui.edu/~xzou/Bio-Capsule-Research-2022/features.tar.gz
tar -xvzf features.tar.gz
rm features.tar.gz
