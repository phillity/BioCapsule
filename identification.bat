@ECHO OFF
ECHO Activate biocapsule environment
call activate biocapsule
ECHO Run identification for lfw_small
python src/identification.py -d lfw_small -f facenet -b underlying
python src/identification.py -d lfw_small -f facenet -b same
python src/identification.py -d lfw_small -f arcface -b underlying
python src/identification.py -d lfw_small -f arcface -b same
ECHO Run identification for color_feret
python src/identification.py -d color_feret -f facenet -b underlying
python src/identification.py -d color_feret -f facenet -b same
python src/identification.py -d color_feret -f arcface -b underlying
python src/identification.py -d color_feret -f arcface -b same
PAUSE