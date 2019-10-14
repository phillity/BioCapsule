@ECHO OFF
ECHO Activate biocapsule environment
call activate biocapsule
ECHO Run arcface extraction for lfw_small
python src/feat_extract.py -d lfw_small -m arcface -gpu 0
ECHO Run facenet extraction for lfw_small
python src/feat_extract.py -d lfw_small -m facenet -gpu 0
PAUSE