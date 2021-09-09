from final_project.smlm_3d.util import get_base_data_path

bpath = get_base_data_path()

dataset_configs = {
    'olympus': {
        'training': {
            'bpath': get_base_data_path() / 'bead_3D_STORM_three_instruments' / '20210723_Olympus_beads' / 'glass_beads_agarose_1',
            'img': 'glass_beads_agarose_1_MMStack_Default.ome.tif',
            'csv': 'glass_beads_agarose_1_MMStack_Default_filtered.csv',
            'wl': 647,
            'na': 1.3,
            'voxel_sizes': (50, 65, 65),
            'reverse_stack': False,
        },
        'sphere_ground_truth': {
            'bpath': get_base_data_path() / 'bead_3D_STORM_three_instruments' / '20210723_Olympus_beads' / '1mm_beads_agarose_centre_1',
            'img': '1mm_beads_agarose_centre_1_MMStack_Default.ome.tif',
            'csv': '1mm_beads_agarose_centre_1_MMStack_Default_substack_filtered.csv',
            'wl': 647,
            'na': 1.3,
            'voxel_sizes': (50, 65, 65),
            'reverse_stack': False,
        },
        'sphere': {
            'bpath': get_base_data_path() / 'bead_3D_STORM_three_instruments' / '20210723_Olympus_beads' / '1mm_beads_agarose_centre_1',
            'img': '1mm_beads_agarose_centre_1_MMStack_Default_substack.tif',
            'csv': '1mm_beads_agarose_centre_1_MMStack_Default_substack_filtered.csv',
            'wl': 647,
            'na': 1.3,
            'voxel_sizes': (2000, 65, 65),
            'reverse_stack': False,
        },
    },
    'openframe': {
        'training': {
            'bpath': get_base_data_path() / 'bead_3D_STORM_three_instruments' / '20200723_OpenFrame_3D_beads' / 'glass_beads_agarose_1',
            'img': 'glass_beads_agarose_1_MMStack_Default.ome.tif',
            'csv': 'glass_beads_agarose_1_MMStack_Default_filtered.csv',
            'wl': 635,
            'na': 1.4,
            'voxel_sizes': (50, 85.5, 85.5),
            'reverse_stack': True,
        },
        'sphere_ground_truth': {
            'bpath': get_base_data_path() / 'bead_3D_STORM_three_instruments' / '20200723_OpenFrame_3D_beads' / 'glass_beads_1mm_agarose_center_1',
            'img': 'glass_beads_1mm_agarose_center_1_MMStack_Default.ome.tif',
            'csv': 'glass_beads_1mm_agarose_center_1_substack_filtered.csv',
            'wl': 635,
            'na': 1.4,
            'voxel_sizes': (50, 85.5, 85.5),
            'reverse_stack': False,
        },
        'sphere': {
            'bpath': get_base_data_path() / 'bead_3D_STORM_three_instruments' / '20200723_OpenFrame_3D_beads' / 'glass_beads_1mm_agarose_center_1',
            'img': 'glass_beads_1mm_agarose_center_1_MMStack_substack.ome.tif',
            'csv': 'glass_beads_1mm_agarose_center_1_substack.csv',
            'wl': 635,
            'na': 1.4,
            'voxel_sizes': (2000, 85.5, 85.5),
            'reverse_stack': False,
        }
    },
    'other':{
        'training':{
            'bpath': get_base_data_path() / 'bead_3D_STORM_three_instruments' / '20210723_beads_1mm' / '1mm_bead_agarose_1',
            'img': '1mm_bead_agarose_1_MMStack_Pos0.ome.tif',
            'csv': '1mm_bead_agarose_1_MMStack_Pos0_filtered.csv',
            'wl': 635,
            'na': 'unknown',
            'voxel_sizes': (50, 106, 106),
            'reverse_stack': False,
        },
        'sphere_ground_truth':{
            'bpath': get_base_data_path() / 'bead_3D_STORM_three_instruments' / '20210723_beads_1mm' / '1mm_bead_surface_center_1',
            'img': '1mm_bead_surface_center_1_MMStack_Pos0.ome.tif',
            'csv': '1mm_bead_surface_center_1_MMStack_Pos0_substack_filtered.csv',
            'wl': 635,
            'na': 'unknown',
            'voxel_sizes': (50, 106, 106),
            'reverse_stack': True,
        },
        'sphere':{
            'bpath': get_base_data_path() / 'bead_3D_STORM_three_instruments' / '20210723_beads_1mm' / '1mm_bead_surface_center_1',
            'img': '1mm_bead_surface_center_1_MMStack_Pos0_substack.ome.tif',
            'csv': '1mm_bead_surface_center_1_MMStack_Pos0_substack_filtered.csv',
            'wl': 635,
            'na': 'unknown',
            'voxel_sizes': (2000, 106, 106),
            'reverse_stack': True,
        },
        'single_slice':{
            'bpath': get_base_data_path() / 'bead_3D_STORM_three_instruments' / '20210723_beads_1mm' / '1mm_bead_surface_center_1',
            'img': 'single_slice.ome.tif',
            'csv': 'single_slice.csv',
            'wl': 635,
            'na': 'unknown',
            'voxel_sizes': (0, 106, 106),
            'reverse_stack': True,
        },
    }
}
