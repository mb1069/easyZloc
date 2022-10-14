import os

psf_modelling_file = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'psf_models.csv')
storm_data_dir = os.path.join(os.path.dirname(__file__), 'raw_data', 'storm')
matlab_data_dir = os.path.join(os.path.dirname(__file__), 'raw_data', 'matlab_data')

jonny_data_dirs = ['/Volumes/Samsung_T5/uni/smlm/647_wf/',
                   '/data/mdb119/smlm/smlm_z/cnnSTORM/src/raw_data/jonny_data/',
                   '/home/miguel/Projects/uni/data/smlm_3d/experimental']
for d in jonny_data_dirs:
    if os.path.exists(d):
        jonny_data_dir = d
        break

res_file = os.path.join(os.path.dirname(__file__), '..', '..', 'tmp', 'res.csv')

weights_file = os.path.join(os.path.dirname(__file__), os.pardir, 'experiments', 'results', 'weights.h5')