# Openframe
BEADS_DIR="/data/mdb119/data/openframe_grid_data/training_data";
# OUTDIR="~/publication/VIT_openframe_tmp";
PX_SIZE=86;
ZSTEP=10;
LOCS_NUP='/data/mdb119/data/openframe_grid_data/nup_data/storm_1_MMStack_Default.ome_locs_undrifted.hdf5';
SPOTS_NUP='/data/mdb119/data/openframe_grid_data/nup_data/storm_1_MMStack_Default.ome_spots.hdf5';
PICKED_LOCS_NUP='/data/mdb119/data/openframe_grid_data/nup_data/storm_1_MMStack_Default.ome_locs_undrifted_picked_4.hdf5';

source ~/anaconda3/etc/profile.d/conda.sh

# echo "Preparing data...";
# conda activate picasso;
# echo python3 ~/publication/prep_data.py $BEADS_DIR -z $ZSTEP -px $PX_SIZE --debug;
mkdir $OUTDIR;
cp $BEADS_DIR/combined/* $OUTDIR/;


conda activate smlm_z;

echo "Training model...";
cd $OUTDIR && echo $PWD &&
python3  ~/publication/train_model_debug.py --dataset openframe -o out_roll_alignment --aug_ratio 2 --brightness $BRIGHTNESS --gauss $GAUSS;

echo "Localising NUP data..."
cd $OUTDIR/out_roll_alignment && echo $PWD &&
python3 ~/publication/localise_exp_sample.py -l $LOCS_NUP -s $SPOTS_NUP -p $PICKED_LOCS_NUP -px $PX_SIZE -mo . -o out_nup;
cd $OUTDIR/out_roll_alignment/out_nup && echo $PWD &&
python3 ~/publication/render_nup.py -l $OUTDIR/out_roll_alignment/out_nup/locs_3d.hdf5 -px $PX_SIZE -df -os 10 -k 0.5 -df;
# cd $OUTDIR/out_roll_alignment && echo $PWD &&
# python3 ~/publication/localise_exp_sample.py -l $LOCS_NUP -s $SPOTS_NUP -px $PX_SIZE -mo . -o out_nup_full_fov;


# LOCS_MITO="/media/Data/smlm_z_data/20231212_miguel_openframe/mitochondria/FOV2/storm_1/storm_1_MMStack_Default.ome_locs_undrift.hdf5";
# SPOTS_MITO="/media/Data/smlm_z_data/20231212_miguel_openframe/mitochondria/FOV2/storm_1/storm_1_MMStack_Default.ome_spots.hdf5";
# PICKED_MITO="/media/Data/smlm_z_data/20231212_miguel_openframe/mitochondria/FOV2/storm_1/storm_1_MMStack_Default.ome_locs_undrift_picked.hdf5";

# echo "Localising mitochondria data..."
# cd $OUTDIR/out_roll_alignment && echo $PWD &&
# python3 ~/publication/localise_exp_sample.py -l $LOCS_MITO -s $SPOTS_MITO -px $PX_SIZE -p $PICKED_MITO -mo . -o out_mitochondria;


# LOCS_TUB="/media/Data/smlm_z_data/20231212_miguel_openframe/tubulin/FOV1/storm_1/storm_1_MMStack_Default.ome_locs_undrifted.hdf5";
# SPOTS_TUB="/media/Data/smlm_z_data/20231212_miguel_openframe/tubulin/FOV1/storm_1/storm_1_MMStack_Default.ome_spots.hdf5";
# PICKED_TUB="/media/Data/smlm_z_data/20231212_miguel_openframe/tubulin/FOV1/storm_1/storm_1_MMStack_Default.ome_locs_undrifted_picked.hdf5";
# echo "Localising tubulin data..."
# cd $OUTDIR/out_roll_alignment && echo $PWD &&
# python3 ~/publication/localise_exp_sample.py -l $LOCS_TUB -s $SPOTS_TUB -px $PX_SIZE -p $PICKED_TUB -mo . -o out_tubulin;
