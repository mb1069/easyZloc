"""
This is a boilerplate pipeline 'classify'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_train_val_test, train_classifier, eval_classifier, check_data, augment_datasets, norm_psfs


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_train_val_test,
            inputs=['X', 'y', 'parameters'],
            outputs=['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'data_splits'],
            name='split_train_val_test',
            tags=['training']
        ),
        node(
            func=augment_datasets,
            inputs=['X_train', 'y_train', 'parameters'],
            outputs=['X_train_aug', 'y_train_aug'],
            name='augment_data',
            tags=['training']
        ),
        node(
            func=norm_psfs,
            inputs=['X_train_aug', 'X_val', 'X_test', 'X'],
            outputs=['X_train_aug_norm', 'X_val_norm', 'X_test_norm', 'X_norm'],
            name='norm_psfs',
            tags=['training']
        ),
        node(
            func=check_data,
            inputs=['X_train_aug_norm', 'y_train_aug', 'X_val_norm', 'y_val', 'X_test_norm', 'y_test'],
            outputs='data_plots',
            name='check_data',
            tags=['training'],
        ),
        node(
            func=train_classifier,
            inputs=['X_train_aug_norm', 'y_train_aug', 'X_val_norm', 'y_val', 'parameters'],
            outputs=['model', 'training_plot'],
            name='train_classifier',
            tags=['training']
        ),
        node(
            func=eval_classifier,
            inputs=['model', 'X_norm', 'y', 'X_train_aug_norm', 'y_train_aug', 'X_val_norm', 'y_val', 'X_test_norm', 'y_test', 'data_splits'],
            outputs=['clf_metrics', 'clf_figures', 'scatter_figures'],
            name='eval_classifier',
            tags=['training']
        )
    ],
        inputs=['X', 'y'],
        outputs=['data_plots', 'model', 'clf_metrics', 'clf_figures', 'scatter_figures'],
    )
