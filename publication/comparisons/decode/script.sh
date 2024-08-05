

# python3 convert_result.py;
# Run 3D drift correction from picked fiduciary beads in Picasso Render GUI
set -e
python3 ../remap_nup_groups.py /home/miguel/Projects/smlm_z/publication/models/zeiss_red_beads/out_24_nvidia6_bak/out_nup_alt_2_pic_updated/nup_renders3/nup.hdf5 emitter_remapped_undrift_picked.hdf5;