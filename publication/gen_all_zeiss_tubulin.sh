# Zeiss
# BEADS_DIR="/media/Data/smlm_z_data/20231121_nup_miguel_zeiss/stacks/";
# OUTDIR="/home/miguel/Projects/smlm_z/publication/VIT_Zeiss";
# OUTDIR='/home/miguel/Projects/smlm_z/autofocus/VIT_zeiss/out_resnet_fov-max_aug11'
# OUTDIR="/home/miguel/Projects/smlm_z/autofocus/VIT_zeiss_lowsnr_data/out_resnet_fov-max_aug11"
# OUTDIR='/home/miguel/Projects/smlm_z/autofocus/VIT_zeiss_lowsnr_data/out_33'
# OUTDIR='/home/miguel/Projects/smlm_z/autofocus/VIT_zeiss_lowsnr_data/out_efficientnet_corrected_rescale'
# OUTDIR='/home/miguel/Projects/smlm_z/autofocus/VIT_zeiss_lowsnr_data/out_mobilenet_corrected_rescale_3'
# OUTDIR='/home/miguel/Projects/smlm_z/autofocus/VIT_zeiss_lowsnr_data/out_efficientnet_corrected_rescale2'
# OUTDIR='/home/miguel/Projects/smlm_z/autofocus/VIT_zeiss/out_resnet_fov-max_aug11'

# OUTDIR='/home/miguel/Projects/smlm_z/autofocus/VIT_zeiss_lowsnr_data/out_71_nvidia6'
# Best
OUTDIR='/home/miguel/Projects/smlm_z/publication/models/zeiss_red_beads/out_24_nvidia6_bak'


# PX_SIZE=106;
# ZSTEP=10;

NUP_OUTDIR='out_tub_alt'
DEFAULT_LOCS='/media/Backup/smlm_z_data/20240315_tubulin/Well1/ROI1/storm_1/storm_1_MMStack_Default.ome_locs_alt_undrift.hdf5'
DEFAULT_SPOTS='/media/Backup/smlm_z_data/20240315_tubulin/Well1/ROI1/storm_1/storm_1_MMStack_Default.ome_spots_alt.hdf5'
PICKED='/home/miguel/Projects/smlm_z/publication/models/zeiss_red_beads/out_24_nvidia6_bak/out_tub_alt/locs_3d_undrift_z_picked.hdf5'
# DEFAULT_LOCS='/media/Backup/smlm_z_data/20240315_tubulin/Well1/ROI1/storm_1/storm_1_MMStack_Default.ome_locs_undrift.hdf5';
# DEFAULT_SPOTS='/media/Backup/smlm_z_data/20240315_tubulin/Well1/ROI1/storm_1/storm_1_MMStack_Default.ome_spots.hdf5';
# PICKED='/media/Backup/smlm_z_data/20240315_tubulin/Well1/ROI1/storm_1/storm_1_MMStack_Default.ome_locs_undrift_picked.hdf5';
# PICKED='/media/Backup/smlm_z_data/20240315_tubulin/Well1/ROI1/storm_1/storm_1_MMStack_Default.ome_locs_undrift_picked_rect.hdf5';
# PICKED_UNDRIFTED_FILE='/home/miguel/Projects/smlm_z/publication/models/zeiss_red_beads/out_24_nvidia6_bak/out_tub/locs_3d_undrift_z_rect.hdf5';
KDE_FACTOR='0.25'

# Poor imaging?
# NUP_OUTDIR='out_tub2'
# DEFAULT_LOCS='/media/Backup/smlm_z_data/20240315_tubulin/Well1/ROI1/storm_2/storm_2_MMStack_Default.ome_locs_undrift.hdf5';
# DEFAULT_SPOTS='/media/Backup/smlm_z_data/20240315_tubulin/Well1/ROI1/storm_2/storm_2_MMStack_Default.ome_spots.hdf5';
# KDE_FACTOR='0.25'

# NUP_OUTDIR='out_tub3'
# DEFAULT_LOCS='/media/Backup/smlm_z_data/20240315_tubulin/Well1/ROI2/storm_1/storm_1_MMStack_Default.ome_locs_undrift.hdf5';
# DEFAULT_SPOTS='/media/Backup/smlm_z_data/20240315_tubulin/Well1/ROI2/storm_1/storm_1_MMStack_Default.ome_spots.hdf5';
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
# python3  /home/miguel/Projects/smlm_z/publication/train_model.py --activation=linear --architecture=mobilenet --aug-brightness=0.08834745796580779 --aug-gauss=0.001723547986120866 --aug-poisson-lam=511.6073576291673 --batch_size=64 --dataset=20240625_NUP_ifluor647 --dense1=1024 --dense2=32 --learning_rate=0.000379076200436412 --norm=fov-max --system=zeiss;

# echo "Localising experimental data..."
# cd $OUTDIR && echo $PWD &&
# python3 /home/miguel/Projects/smlm_z/publication/localise_exp_sample.py -l $DEFAULT_LOCS -s $DEFAULT_SPOTS -mo . -o $NUP_OUTDIR;


# echo $OUTDIR/$NUP_OUTDIR/locs_3d.hdf5
# python3 /home/miguel/Projects/smlm_z/publication/undrift_z.py $OUTDIR/$NUP_OUTDIR/locs_3d.hdf5 --n-frames 500;

cd $OUTDIR/$NUP_OUTDIR && echo $PWD && wandb offline &&
# python3 /home/miguel/Projects/smlm_z/publication/render_tub.py -l $OUTDIR/$NUP_OUTDIR/locs_3d_undrift_z.hdf5 -p $PICKED --kde-factor $KDE_FACTOR --max-frame 13000;
# python3 /home/miguel/Projects/smlm_z/publication/render_tub.py -l $OUTDIR/$NUP_OUTDIR/locs_3d_undrift_z_picked.hdf5 --kde-factor $KDE_FACTOR;
python3 /home/miguel/Projects/smlm_z/publication/render_tub.py -l $PICKED --kde-factor $KDE_FACTOR --max-frame 13000;



# RCC, n_good, mean_sep, sep_std
# 100, 18, 98, 35.36
# 250, 31, 95.7, 34.98
# 500, 32, 97, 37
# 750, 28, 95.6. 34.4
# 1000, 22, 97.57, 36.581