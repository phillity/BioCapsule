@ECHO OFF
ECHO Activate biocapsule environment
call activate biocapsule
ECHO Run verification for lfw
python src/verification.py -f facenet -b underlying
python src/verification.py -f facenet -b same
python src/verification.py -f arcface -b underlying
python src/verification.py -f arcface -b same
PAUSE