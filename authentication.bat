@ECHO OFF
ECHO Activate biocapsule environment
call activate biocapsule
ECHO Run authentication for lfw_small
python src/authentication.py -d lfw_small -f facenet -b underlying -t 16
python src/authentication.py -d lfw_small -f facenet -b same -t 16
python src/authentication.py -d lfw_small -f arcface -b underlying -t 16
python src/authentication.py -d lfw_small -f arcface -b same -t 16
ECHO Run authentication for color_feret
python src/authentication.py -d color_feret -f facenet -b underlying -t 16
python src/authentication.py -d color_feret -f facenet -b same -t 16
python src/authentication.py -d color_feret -f arcface -b underlying -t 16
python src/authentication.py -d color_feret -f arcface -b same -t 16
PAUSE