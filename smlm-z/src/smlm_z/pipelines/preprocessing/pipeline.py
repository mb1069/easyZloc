"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import extract_training_stacks, resize_stacks, norm_coordinates, align_stacks, stacks_to_training_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=extract_training_stacks,
            inputs=['spots', 'bead_stack', 'params:picasso'],
            outputs='training_stacks',
            name='extract_stacks',
            tags=['training', 'inference']
        ),
        node(
            func=resize_stacks,
            inputs=['training_stacks', 'params:model_input_shape'],
            outputs='resized_stacks',
            name='resize_stacks',
            tags=['training', 'inference']
        ),
        node(
            func=norm_coordinates,
            inputs=['locs', 'bead_stack'],
            outputs='norm_coords',
            name='norm_coordinates',
            tags=['training']
        ),
        node(
            func=align_stacks,
            inputs=['training_stacks', 'norm_coords'],
            outputs='offsets',
            name='align_psfs',
            tags=['training']
        ),
        node(
            func=stacks_to_training_data,
            inputs=['resized_stacks', 'norm_coords', 'offsets'],
            outputs=['X', 'y'],
            name='stacks_to_training_data',
            tags=['training']
        )
    ],
    inputs=['spots', 'locs', 'bead_stack'],
    outputs=['X', 'y']
    )
