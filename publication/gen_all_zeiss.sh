# Zeiss
BEADS_DIR="/media/Data/smlm_z_data/20231121_nup_miguel_zeiss/stacks/";
OUTDIR="/home/miguel/Projects/smlm_z/publication/VIT_Zeiss";
PX_SIZE=106;
ZSTEP=10;
DEFAULT_LOCS='/media/Data/smlm_z_data/20231121_nup_miguel_zeiss/FOV1/storm_1/storm_1_MMStack_Default.ome_locs_undrifted.hdf5';
DEFAULT_SPOTS='/media/Data/smlm_z_data/20231121_nup_miguel_zeiss/FOV1/storm_1/storm_1_MMStack_Default.ome_spots.hdf5';
PICKED='/media/Data/smlm_z_data/20231121_nup_miguel_zeiss/FOV1/storm_1/storm_1_MMStack_Default.ome_locs_picked.hdf5';


source ~/anaconda3/etc/profile.d/conda.sh

# echo "Preparing data...";
# conda activate picasso;
# echo python3 /home/miguel/Projects/smlm_z/publication/prep_data.py $BEADS_DIR -z $ZSTEP -px $PX_SIZE --debug;
# mkdir $OUTDIR;
# cp $BEADS_DIR/combined/* $OUTDIR/;

conda activate smlm_z;

# echo "Training model...";
# cd $OUTDIR && echo $PWD &&
# python3  /home/miguel/Projects/smlm_z/publication/train_model_debug.py -o out_roll_alignment --aug_ratio 2 --brightness 0.01 --gauss 0.001;

echo "Localising experimental data..."
cd $OUTDIR/out_roll_alignment && echo $PWD &&
python3 /home/miguel/Projects/smlm_z/publication/localise_exp_sample.py -l $DEFAULT_LOCS -s $DEFAULT_SPOTS  -px $PX_SIZE -p $PICKED -mo . -o out_nup;
cd $OUTDIR/out_roll_alignment/out_nup && echo $PWD &&
python3 /home/miguel/Projects/smlm_z/publication/render_nup.py -l $OUTDIR/out_roll_alignment/out_nup/locs_3d.hdf5 -px $PX_SIZE -p $PICKED;

