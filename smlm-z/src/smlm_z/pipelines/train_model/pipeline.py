"""
This is a boilerplate pipeline 'classify'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_train_val_test, train_classifier, eval_classifier, check_data, augment_datasets


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_train_val_test,
            inputs=['X', 'y', 'parameters'],
            outputs=['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'],
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
            func=check_data,
            inputs=['X_train_aug', 'y_train_aug', 'X_val', 'y_val', 'X_test', 'y_test'],
            outputs='data_plots',
            name='check_data',
            tags=['training'],
        ),
        node(
            func=train_classifier,
            inputs=['X_train_aug', 'y_train_aug', 'X_val', 'y_val', 'parameters'],
            outputs=['model', 'training_plot'],
            name='train_classifier',
            tags=['training']
        ),
        node(
            func=eval_classifier,
            inputs=['model', 'X_train_aug', 'y_train_aug', 'X_val', 'y_val', 'X_test', 'y_test'],
            outputs=['clf_metrics', 'clf_figures', 'scatter_figures'],
            name='eval_classifier',
            tags=['training']
        )
    ],
        inputs=['X', 'y'],
        outputs=['data_plots', 'model', 'clf_metrics', 'clf_figures', 'scatter_figures'],
    )
