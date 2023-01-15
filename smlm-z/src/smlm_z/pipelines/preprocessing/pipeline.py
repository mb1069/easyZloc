"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import extract_training_stacks, resize_stacks, norm_coordinates, align_stacks, stacks_to_training_data, norm_images, merge_model_inputs

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=extract_training_stacks,
            inputs=['spots', 'bead_stack', 'parameters'],
            outputs='training_stacks',
            name='extract_stacks',
            tags=['training', 'inference']
        ),
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
            func=norm_images,
            inputs='training_stacks',
            outputs='norm_stacks',
            name='norm_stacks',
            tags=['training'],
        ),
        node(
            func=stacks_to_training_data,
            inputs=['norm_stacks', 'norm_coords', 'offsets'],
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
            func=merge_model_inputs,
            inputs=['resized_psfs', 'xy_coords', 'z_coords'],
            outputs=['X', 'y'],
            name='merge_model_inputs',
            tags=['training', 'inference']
        )
    ],
    inputs=['spots', 'locs', 'bead_stack'],
    outputs=['X', 'y', 'offsets_plot']
)
