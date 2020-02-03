#!/bin/bash

echo 'Downloading and extracting LFW and RS features!'

wget https://cs.iupui.edu/~phillity/features.tar.gz
tar -xvzf features.tar.gz
rm features.tar.gz

echo 'Downloading and extracting VGGFace2 ArcFace features!'

wget https://cs.iupui.edu/~phillity/vggface2_arcface_dataset.hdf5

echo 'Downloading and extracting VGGFace2 FaceNet features!'

wget https://cs.iupui.edu/~phillity/vggface2_facenet_dataset.hdf5
