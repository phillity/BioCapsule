#!/bin/bash

mkdir models

echo 'Downloading and extracting arcface model!'

wget https://cs.iupui.edu/~xzou/Bio-Capsule-Research-2022/model-r100-ii.tar.gz
tar -xvzf model-r100-ii.tar.gz
rm model-r100-ii.tar.gz
mv model-r100-ii models/arcface

echo 'Downloading and extracting facenet model!'

wget https://cs.iupui.edu/~xzou/Bio-Capsule-Research-2022/model-ir-v1.tar.gz
tar -xvzf model-ir-v1.tar.gz
rm model-ir-v1.tar.gz
mv model-ir-v1 models/facenet

echo 'Downloading and extracting mtcnn model!'

wget https://cs.iupui.edu/~xzou/Bio-Capsule-Research-2022/model-mtcnn.tar.gz
tar -xvzf model-mtcnn.tar.gz
rm model-mtcnn.tar.gz
mv model-mtcnn models/mtcnn

echo 'Downloading and extracting retinaface model!'

wget https://cs.iupui.edu/~xzou/Bio-Capsule-Research-2022/retinaface.tar.gz
tar -xvzf retinaface.tar.gz
rm retinaface.tar.gz
mv retinaface models/retinaface
