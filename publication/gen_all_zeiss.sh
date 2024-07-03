# Zeiss
# BEADS_DIR="/media/Data/smlm_z_data/20231121_nup_miguel_zeiss/stacks/";
# OUTDIR="/home/miguel/Projects/smlm_z/publication/VIT_Zeiss";
# OUTDIR='/home/miguel/Projects/smlm_z/autofocus/VIT_zeiss/out_resnet_fov-max_aug11'
# OUTDIR="/home/miguel/Projects/smlm_z/autofocus/VIT_zeiss_lowsnr_data/out_resnet_fov-max_aug11"
# OUTDIR='/home/miguel/Projects/smlm_z/autofocus/VIT_zeiss_lowsnr_data/out_33'
# OUTDIR='/home/miguel/Projects/smlm_z/autofocus/VIT_zeiss_lowsnr_data/out_efficientnet_corrected_rescale'
# OUTDIR='/home/miguel/Projects/smlm_z/autofocus/VIT_zeiss_lowsnr_data/out_mobilenet_corrected_rescale_3'
# OUTDIR='/home/miguel/Projects/smlm_z/autofocus/VIT_zeiss_lowsnr_data/out_efficientnet_corrected_rescale2'
OUTDIR='/home/miguel/Projects/smlm_z/autofocus/VIT_zeiss_lowsnr_data/out_24_nvidia6_repeat_5'
# Best
# OUTDIR='/home/miguel/Projects/smlm_z/autofocus/VIT_zeiss_lowsnr_data/out_24_nvidia6'



# PX_SIZE=106;
# ZSTEP=10;

NUP_OUTDIR='out_nup'
DEFAULT_LOCS='/media/Backup/smlm_z_data/20240625_NUP_ifluor647/FOV1/storm_1/storm_1_MMStack_Default.ome_locs_undrift.hdf5';
DEFAULT_SPOTS='/media/Backup/smlm_z_data/20240625_NUP_ifluor647/FOV1/storm_1/storm_1_MMStack_Default.ome_spots.hdf5';
PICKED='/media/Backup/smlm_z_data/20240625_NUP_ifluor647/FOV1/storm_1/storm_1_MMStack_Default.ome_locs_undrift_picked.hdf5';
KDE_FACTOR='0.5'

# NUP_OUTDIR='out_nup2'
# DEFAULT_LOCS='/media/Backup/smlm_z_data/20240625_NUP_ifluor647/FOV2/storm_1/storm_1_MMStack_Default.ome_locs_undrift.hdf5';
# DEFAULT_SPOTS='/media/Backup/smlm_z_data/20240625_NUP_ifluor647/FOV2/storm_1/storm_1_MMStack_Default.ome_spots.hdf5';
# PICKED='/media/Backup/smlm_z_data/20240625_NUP_ifluor647/FOV2/storm_1/storm_1_MMStack_Default.ome_locs_undrift_picked.hdf5';
# KDE_FACTOR='0.25'

set -e

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

# echo "Localising experimental data..."
cd $OUTDIR && echo $PWD &&
python3 /home/miguel/Projects/smlm_z/publication/localise_exp_sample.py -l $DEFAULT_LOCS -s $DEFAULT_SPOTS -mo . -o $NUP_OUTDIR;


python3 /home/miguel/Projects/smlm_z/publication/undrift_z.py $OUTDIR/$NUP_OUTDIR/locs_3d.hdf5 --rcc 500;

# cd $OUTDIR/out_nup && echo $PWD &&
# python3 /home/miguel/Projects/smlm_z/publication/render_nup.py -l $OUTDIR/$NUP_OUTDIR/locs_3d.hdf5 -p $PICKED;

cd $OUTDIR/$NUP_OUTDIR && echo $PWD &&
python3 /home/miguel/Projects/smlm_z/publication/render_nup.py -l $OUTDIR/$NUP_OUTDIR/locs_3d_undrift_z.hdf5 -p $PICKED --kde-factor $KDE_FACTOR;


# RCC, n_good, mean_sep, sep_std
# 100, 18, 98, 35.36
# 250, 31, 95.7, 34.98
# 500, 32, 97, 37
# 750, 28, 95.6. 34.4
# 1000, 22, 97.57, 36.581