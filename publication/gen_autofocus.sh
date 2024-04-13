# Openframe
BEADS_DIR="/home/miguel/Projects/data/11042024_beads4Miguel/11042024_beads4Miguel_beads/";
OUTDIR="/home/miguel/Projects/smlm_z/autofocus/VIT_autofocus2";
PX_SIZE=86;
ZSTEP=10;

BRIGHTNESS=0
GAUSS=0


source ~/anaconda3/etc/profile.d/conda.sh
MODEL_DIR=$OUTDIR/out_roll_alignment
OUT_NUP=$OUTDIR/out_roll_alignment/out_nup
# echo "Preparing data...";
# conda activate picasso;
# echo python3 ~/publication/prep_data.py $BEADS_DIR -z $ZSTEP -px $PX_SIZE --debug;
mkdir -p $OUTDIR;
cp $BEADS_DIR/combined/* $OUTDIR/;

set -e 

conda activate smlm_z;

echo "Training model...";
cd $OUTDIR && echo $PWD &&
python3 /home/miguel/Projects/smlm_z/publication/train_model.py --project autofocus --norm frame --system openframe --dataset $BEADS_DIR -o $MODEL_DIR --aug-brightness $BRIGHTNESS --aug-gauss $GAUSS;