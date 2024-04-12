# Openframe
BEADS_DIR="/home/miguel/Projects/data/20230601_MQ_celltype/20230601_MQ_celltype_beads";
OUTDIR="/home/miguel/Projects/smlm_z/publication/VIT_openframe";
PX_SIZE=86;
ZSTEP=10;
LOCS_NUP='/home/miguel/Projects/data/20230601_MQ_celltype/nup/fov2/storm_1/storm_1_MMStack_Default.ome_locs.hdf5';
SPOTS_NUP='/home/miguel/Projects/data/20230601_MQ_celltype/nup/fov2/storm_1/storm_1_MMStack_Default.ome_spots.hdf5';
PICKED_LOCS_NUP='/home/miguel/Projects/data/20230601_MQ_celltype/nup/fov2/storm_1/storm_1_MMStack_Default.ome_locs_undrifted_picked_4.hdf5';

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
python3 /home/miguel/Projects/smlm_z/publication/train_model.py --norm frame --system openframe --dataset $BEADS_DIR -o $MODEL_DIR --aug-brightness $BRIGHTNESS --aug-gauss $GAUSS;

python3 /home/miguel/Projects/smlm_z/publication/ -mo $MODEL_DIR --norm frame --zstep 10 --datasets /home/miguel/Projects/data/20231020_20nm_beads_10um_range_10nm_step /home/miguel/Projects/data/20230601_MQ_celltype/20230601_MQ_celltype_beads /home/miguel/Projects/data/all_openframe_beads/20231205_miguel_mitochondria /home/miguel/Projects/data/all_openframe_beads/20231212_miguel_openframe /home/miguel/Projects/data/20231128_tubulin_miguel

echo "Localising NUP data..."
cd $MODEL_DIR && echo $PWD &&
python3 ~/publication/localise_exp_sample.py -l $LOCS_NUP -s $SPOTS_NUP -p $PICKED_LOCS_NUP -px $PX_SIZE -mo . -o $OUT_NUP;
cd $OUT_NUP && echo $PWD &&
python3 ~/publication/render_nup.py -l $OUT_NUP/locs_3d.hdf5 -px $PX_SIZE -df -os 10 -k 0.5 -df;
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
