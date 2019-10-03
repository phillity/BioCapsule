@ECHO OFF
ECHO Activate biocapsule environment
call activate biocapsule
ECHO Run arcface extraction for caltech
python src/feat_extract.py -d caltech -m arcface
ECHO Run facenet extraction for caltech
python src/feat_extract.py -d caltech -m facenet
ECHO Run arcface extraction for gt_db
python src/feat_extract.py -d gt_db -m arcface
ECHO Run facenet extraction for gt_db
python src/feat_extract.py -d gt_db -m facenet
ECHO Run arcface extraction for orl
python src/feat_extract.py -d orl -m arcface
ECHO Run facenet extraction for orl
python src/feat_extract.py -d orl -m facenet
ECHO Run arcface extraction for lfw
python src/feat_extract.py -d lfw -m arcface
ECHO Run facenet extraction for lfw
python src/feat_extract.py -d lfw -m facenet
ECHO Run arcface extraction for color_feret
python src/feat_extract.py -d color_feret -m arcface
ECHO Run facenet extraction for color_feret
python src/feat_extract.py -d color_feret -m facenet
PAUSE