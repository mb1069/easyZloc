import pandas as pd
from tifffile import imread, imwrite
import os
import matplotlib.pyplot as plt

from final_project.smlm_3d.data.datasets import TrainingDataSet, ExperimentalDataSet
from final_project.smlm_3d.util import get_base_data_path
from final_project.smlm_3d.workflow_v2 import train_model, measure_error

res_file = os.path.join(os.path.dirname(__file__), 'results', 'z_step.csv')

def sub_sample_image(impath, src_zstep, target_zstep):
    if src_zstep == target_zstep:
        return impath
    new_impath = str(impath).replace('.ome.tif', f'_zstep_{target_zstep}.ome.tif')
    if os.path.exists(new_impath):
        return new_impath
    img = imread(impath)
    img = img[::target_zstep//src_zstep]
    imwrite(new_impath, img, compress=6)
    return new_impath
    

def main():
    z_range = 2000

    train_dpath = get_base_data_path() / 'experimental' / 'other' / '1mm_bead_june6_' / '635_red_stack_5_1'
    _train_img = '635_red_stack_5_1_MMStack_Pos0.ome.tif'
    train_csv = '635_red_stack_5_1_MMStack_Pos0.csv'
    _voxel_sizes = (10, 106, 106)

    # errors = dict()
    # for z_step in range(10, 150, 10):
    #     train_img = sub_sample_image(train_dpath / _train_img, _voxel_sizes[0], target_zstep=z_step)

    #     voxel_sizes = list(_voxel_sizes)
    #     voxel_sizes[0] = z_step
    #     train_dataset = TrainingDataSet(train_dpath, train_img, train_csv, voxel_sizes, z_range)

    #     model = train_model(train_dataset.data)
    #     z_step_errors = measure_error(model, train_dataset.data['test'])
    #     errors[z_step] = z_step_errors

    # min_len = min([len(v) for v in errors.values()])
    # errors = {k:v[0:min_len] for k, v in errors.items()}
    
    # df = pd.DataFrame.from_records(errors)
    # df.to_csv(res_file, index=False)

    df = pd.read_csv(res_file)
    df.boxplot(showfliers=False)
    # plt.yscale('log')
    plt.xlabel('Z-localisation accuracy (nm)')
    plt.ylabel('Sampling Z-step in calibration bead stack (nm)')
    plt.show()

if __name__ == '__main__':
    main()