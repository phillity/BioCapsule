#!/bin/bash

echo 'Downloading and extracting VGGFace2_Visualize images!'

wget https://cs.iupui.edu/~xzou/Bio-Capsule-Research-2022/vggface2_visualize.tar.gz
tar -xvzf vggface2_visualize.tar.gz
rm vggface2_visualize.tar.gz

echo 'Downloading and extracting LFW images!'

wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
wget http://vis-www.cs.umass.edu/lfw/pairs.txt
wget http://vis-www.cs.umass.edu/lfw/people.txt
tar -xvzf lfw.tgz
rm lfw.tgz

echo 'Downloading and extracting GTDB images!'

wget http://www.anefian.com/research/gt_db.zip
unzip gt_db.zip
rm gt_db.zip
mv gt_db gtdb
cd gtdb
find . -name "*.jbf" -type f -delete
cd ..

echo 'Downloading and extracting RS images!'

wget https://cs.iupui.edu/~xzou/Bio-Capsule-Research-2022/rs.tar.gz
tar -xvzf rs.tar.gz
rm rs.tar.gz
