# FD-LOCO
BEADS_DIR="/media/Data/smlm_z_data/fd-loco/Astigmatism_beads_stacks_2um";
OUTDIR="/home/miguel/Projects/smlm_z/publication/VIT_fd-loco";
PX_SIZE=110;
ZSTEP=10;
DEFAULT_LOCS='/home/miguel/Projects/data/fd-loco/roi_startpos_810_790_split.ome_locs.hdf5';
DEFAULT_SPOTS='/home/miguel/Projects/data/fd-loco/roi_startpos_810_790_split.ome_spots.hdf5';
PICKED='/home/miguel/Projects/data/fd-loco/roi_startpos_810_790_split.ome_locs_picked.hdf5';

BRIGHTNESS=0
GAUSS=0

source ~/anaconda3/etc/profile.d/conda.sh
MODEL_DIR=$OUTDIR/out_roll_alignment
OUT_NUP=$OUTDIR/out_roll_alignment/out_nup


source ~/anaconda3/etc/profile.d/conda.sh

# echo "Preparing data...";
# conda activate picasso;
# echo python3 ~/publication/prep_data.py $BEADS_DIR -z $ZSTEP -px $PX_SIZE --debug;
mkdir -p $OUTDIR;
cp $BEADS_DIR/combined/* $OUTDIR/;

set -e 



conda activate smlm_z;

# echo "Training model...";
# cd $OUTDIR && echo $PWD &&
# python3 /home/miguel/Projects/smlm_z/publication/train_model.py --norm frame --system ries --dataset $BEADS_DIR -o $MODEL_DIR --aug-brightness $BRIGHTNESS --aug-gauss $GAUSS;



# echo "Localising experimental data..."
# cd $OUTDIR/out_roll_alignment && echo $PWD &&
# python3 /home/miguel/Projects/smlm_z/publication/localise_exp_sample.py -l $DEFAULT_LOCS -s $DEFAULT_SPOTS -p $PICKED -mo . -o out_nup;
# cd $OUTDIR/out_roll_alignment/out_nup && echo $PWD &&
# python3 /home/miguel/Projects/smlm_z/publication/render_nup.py -l $OUTDIR/out_roll_alignment/out_nup/locs_3d.hdf5 -px $PX_SIZE -p $PICKED;

# echo "Generate Ries comparison data"
# python3 /home/miguel/Projects/smlm_z/publication/fd_loco_accuracy_comparison.py $OUTDIR;
