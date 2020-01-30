#!/bin/bash

echo 'Downloading and extracting LFW images!'

wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
wget http://vis-www.cs.umass.edu/lfw/pairs.txt
wget http://vis-www.cs.umass.edu/lfw/people.txt

tar -xvzf lfw.tgz
rm lfw.tgz

echo 'Downloading and extracting RS images!'

wget https://cs.iupui.edu/~phillity/rs.tar.gz

tar -xvzf rs.tar.gz
rm rs.tar.gz
