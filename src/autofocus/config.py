import glob

# Level of detail of DWT transform. Higher value -> higher memory footprint and potentially better performance.
dwt_level = 6


base_data_path = '/home/miguel/Projects/uni/data/autofocus'

cfgs = {
    'slit_50nm': {
        'glob_path' : '{base_data_path}/slit/202006*_*um_{vs}nm_AF_74*/stack*/MMStack_Pos1.ome.tif',
        'z_voxel': 50,
        'row_avg': False
    },
    'slit_1000nm': {
        'glob_path' : '{base_data_path}/slit/202006*_*um_{vs}nm_AF_74*/stack*/MMStack_Pos1.ome.tif',
        'z_voxel': 1000,
        'row_avg': True
    },
    'cylindrical_200nm': {
        'glob_path' : '{base_data_path}/cylindrical_lenses_openframe/*_*um_{vs}nm_cylindrical_lenses/*.tif',
        'z_voxel': 200,
        'row_avg': False
    },
    # Not working (due to corrupted tiff files?)
    # 'cylindrical_50nm': {
    #     'glob_path' : '{base_data_path}/cylindrical_lenses_openframe/*_*um_{vs}nm_cylindrical_lenses/*.tif',
    #     'z_voxel': 50,
    #     'row_avg': True
    # }
}


def get_cfg_images(cfg):
    vs = cfg['z_voxel']
    glob_path = cfg['glob_path']
    img_glob = glob_path.format(base_data_path=base_data_path, vs=vs)
    images = glob.glob(img_glob)
    print(f'Found {len(images)} images.')
    return images
