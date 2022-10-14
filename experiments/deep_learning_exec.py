from data.datasets import TrainingDataSet, ExperimentalDataSet
from config.datafiles import res_file
from config.datasets import dataset_configs
from data.visualise import scatter_3d, show_psf_axial
from workflow_v2 import eval_model
from experiments.deep_learning import load_model
import numpy as np
import matplotlib.pyplot as plt


z_range = 1000

dataset = 'paired_bead_stacks'


sub_datasets = {k: [] for k in ['training', 'experimental']}
for k in sub_datasets:
    exp_dataset = ExperimentalDataSet(dataset_configs[dataset][k], transform_data=False)
    model = load_model()
    

    # coords = exp_dataset.predict_dataset(model, 200)
    coords = exp_dataset.estimate_ground_truth()
    scatter_3d(coords)
quit()

for sub_dataset in sub_datasets:
    exp_dataset = TrainingDataSet(dataset_configs[dataset][sub_dataset], z_range, transform_data=False, add_noise=False, lazy=True)
    exp_dataset.prepare_debug()


    for i in range(exp_dataset.total_emitters):
        try:
            psf = exp_dataset.debug_emitter(i, z_range)[0][:, :, :, np.newaxis]
        except RuntimeError:
            continue
        pred = model.predict(psf).squeeze()
        pred = pred - pred[0]
        truth = np.array([i*10 for i in range(len(pred))])
        acc = abs(pred-truth)
        sub_datasets[sub_dataset].extend(acc)

labels, vals = zip(*sub_datasets.items())
plt.boxplot(vals, showfliers=False)
plt.xticks([1, 2], labels)
plt.show()
        