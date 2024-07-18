# Openframe
# BEADS_DIR="/home/miguel/Projects/data/11042024_beads4Miguel/11042024_beads4Miguel_beads/";
# OUTDIR="/home/miguel/Projects/smlm_z/autofocus/VIT_autofocus2";
# PX_SIZE=86;
# ZSTEP=10;


# Zeiss
BEADS_DIR="/home/miguel/Projects/data/20240415_beads_AF_Miguel/stack_before_1/";
OUTDIR="/home/miguel/Projects/smlm_z/autofocus/VIT_autofocus_zeiss4";
PX_SIZE=106;
ZSTEP=10;
NORM="frame-min"

BRIGHTNESS=0.2
GAUSS=0.05


source ~/anaconda3/etc/profile.d/conda.sh
MODEL_DIR=$OUTDIR/out_roll_alignment
OUT_NUP=$OUTDIR/out_roll_alignment/out_nup
echo "Preparing data...";
conda activate picasso;
# python3 /home/miguel/Projects/smlm_z/publication/prep_data.py $BEADS_DIR -z $ZSTEP -px $PX_SIZE --debug -g 10000;
mkdir -p $OUTDIR;
cp $BEADS_DIR/combined/* $OUTDIR/;

set -e 

conda activate smlm_z;

echo "Training model...";
cd $OUTDIR && echo $PWD && wandb offline && 
python3 /home/miguel/Projects/smlm_z/publication/train_model.py --project autofocus --norm $NORM --system openframe --dataset $BEADS_DIR -o $MODEL_DIR --aug-brightness $BRIGHTNESS --aug-gauss $GAUSS;