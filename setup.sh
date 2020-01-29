#!/bin/bash

dir=$(dirname "$0")

echo 'Downloading and extracting face models!'

cd $dir"/src/face_models/"
bash face_setup.sh
cd $dir

echo 'Downloading and extracting RS and LFW images!'

cd $dir"/images/"
bash images_setup.sh
cd $dir

echo 'Downloading and extracting RS and LFW features!'

cd $dir"/data/"
bash data_setup.sh
cd $dir