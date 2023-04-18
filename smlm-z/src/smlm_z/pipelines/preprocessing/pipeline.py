"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import extract_training_stacks, resize_stacks, norm_coordinates, align_stacks, stacks_to_training_data, norm_images, merge_model_inputs, trim_stacks
from .model_psf import model_and_sim_beadstacks


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=norm_coordinates,
            inputs=['locs', 'parameters'],
            outputs='norm_coords',
            name='norm_coordinates',
            tags=['training']
        ),
        node(
            func=model_and_sim_beadstacks,
            inputs=['raw_stacks', 'params:psf_modelling_params'],
            outputs='training_stacks',
            name='model_psfs',
            tags=['training'],
        ),
        node(
            func=stacks_to_validation_data,
            inputs=['raw_stacks', 'norm_coords', 'parameters'],
            outputs=['val_psfs', 'val_xy_coords', 'val_z_coords'],
            name='stacks_to_val_data',
            tags=['training']
        ),
        node(
            func=stacks_to_training_data,
            inputs=['training_stacks', 'norm_coords', 'offsets', 'parameters'],
            outputs=['psfs', 'xy_coords', 'z_coords'],
            name='stacks_to_training_data',
            tags=['training']
        ),
        node(
            func=resize_stacks,
            inputs=['psfs', 'params:model_input_shape'],
            outputs='resized_psfs',
            name='resize_stacks',
            tags=['training', 'inference']
        ),
        node(
            func=norm_images,
            inputs='resized_psfs',
            outputs='norm_psfs',
            name='norm_psfs',
            tags=['training'],
        ),
        node(
            func=trim_stacks,
            inputs=['norm_psfs', 'xy_coords', 'z_coords', 'parameters'],
            outputs=['trim_psfs', 'trim_xy_coords', 'trim_z_coords'],
            name='trim_stacks',
            tags=['training', 'inference']
        ),
        node(
            func=merge_model_inputs,
            inputs=['trim_psfs', 'trim_xy_coords', 'trim_z_coords'],
            outputs=['X', 'y'],
            name='merge_model_inputs',
            tags=['training', 'inference']
        )
    ],
    inputs=['raw_stacks', 'locs'],
    outputs=['training_stacks', 'X', 'y', 'offsets_plot']
)
