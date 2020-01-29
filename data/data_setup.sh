#!/bin/bash

echo 'Downloading and extracting LFW features!'

wget https://cs.iupui.edu/~phillity/lfw_arcface_feat.npz
wget https://cs.iupui.edu/~phillity/lfw_facenet_feat.npz

echo 'Downloading and extracting RS features!'

wget https://cs.iupui.edu/~phillity/rs_arcface_feat.npz
wget https://cs.iupui.edu/~phillity/rs_facenet_feat.npz
