#!/bin/bash

echo 'Downloading and extracting face models!'

wget https://cs.iupui.edu/~phillity/model-r100-ii.tar.gz
tar -xvzf model-r100-ii.tar.gz
rm model-r100-ii.tar.gz

wget https://cs.iupui.edu/~phillity/model-mtcnn.tar.gz
tar -xvzf model-mtcnn.tar.gz
rm model-mtcnn.tar.gz

wget https://cs.iupui.edu/~phillity/model-ir-v1.tar.gz
tar -xvzf model-ir-v1.tar.gz
rm model-ir-v1.tar.gz