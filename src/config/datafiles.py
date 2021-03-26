import os

psf_modelling_file = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'psf_models.csv')
storm_data_dir = os.path.join(os.path.dirname(__file__), 'raw_data', 'storm')
matlab_data_dir = os.path.join(os.path.dirname(__file__), 'raw_data', 'matlab_data')

jonny_data_dir = '/Volumes/Samsung_T5/uni/smlm/647_wf/'
if not os.path.exists(jonny_data_dir):
    jonny_data_dir = '/data/mdb119/smlm/smlm_z/cnnSTORM/src/raw_data/jonny_data/'
