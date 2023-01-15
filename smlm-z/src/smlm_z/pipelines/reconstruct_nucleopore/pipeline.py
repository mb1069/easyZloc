"""
This is a boilerplate pipeline 'reconstruct_nucleopore'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from ..preprocessing.nodes import norm_images, resize_stacks

from .nodes import cluster_locs, predict_z, recreate_sample, norm_exp_coordinates

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=norm_images,
            inputs='nanopore_spots',
            outputs='norm_spots',
            name='norm_exp_stacks',
            tags=['training', 'reconstruct'],
        ),
        node(
            func=resize_stacks,
            inputs=['norm_spots', 'params:model_input_shape'],
            outputs='resized_spots',
            name='resize_exp_spots',
            tags=['training', 'reconstruct'],
        ),
        node(
            func=cluster_locs,
            inputs=['nanopore_locs', 'parameters'],
            outputs='clustered_locs',
            name='cluster_locs',
            tags=['training', 'reconstruct']
        ),
        node(
            func=norm_exp_coordinates,
            inputs=['nanopore_locs', 'parameters'],
            outputs='norm_df',
            name='norm_exp_coordinates',
            tags=['training', 'reconstruct']
        ),
        node(
            func=predict_z,
            inputs=['model', 'resized_spots', 'norm_df'],
            outputs='z_pos',
            name='predict_z',
            tags=['training', 'reconstruct']
        ),
        node(
            func=recreate_sample,
            inputs=['z_pos', 'clustered_locs'],
            outputs=['nanopore_plot_3d', 'nanopore_gauss_model_fits'],
            name='recreate_sample',
            tags=['training', 'reconstruct']
        )
    ],
    inputs=['model', 'nanopore_spots', 'nanopore_locs'],
    outputs=['nanopore_plot_3d', 'nanopore_gauss_model_fits']
    )
