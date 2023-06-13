"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import resize_stacks, norm_coordinates, align_stacks, stacks_to_training_data, norm_images, merge_model_inputs, trim_stacks
from .mask_psfs import mask_psfs

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
            func=align_stacks,
            inputs=['training_stacks', 'norm_coords', 'parameters'],
            outputs=['offsets', 'offsets_plot'],
            name='align_psfs',
            tags=['training']
        ),
        node(
            func=stacks_to_training_data,
            inputs=['training_stacks', 'norm_coords', 'offsets', 'parameters'],
            outputs=['psfs', 'xy_coords', 'z_coords'],
            name='stacks_to_training_data',
            tags=['training']
        ),
        # node(
        #     func=mask_psfs,
        #     inputs=['psfs', 'parameters'],
        #     outputs='masked_psfs',
        #     name='mask_psfs',
        #     tags=['training', 'inference'],
        # ),
        node(
            func=resize_stacks,
            inputs=['psfs', 'params:model_input_shape'],
            outputs='resized_psfs',
            name='resize_stacks',
            tags=['training', 'inference']
        ),
        # node(
        #     func=norm_images,
        #     inputs='resized_psfs',
        #     outputs='norm_psfs',
        #     name='norm_psfs',
        #     tags=['training'],
        # ),
        node(
            func=trim_stacks,
            inputs=['resized_psfs', 'xy_coords', 'z_coords', 'parameters'],
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
    inputs=['training_stacks', 'locs'],
    outputs=['X', 'y', 'offsets_plot']
)
