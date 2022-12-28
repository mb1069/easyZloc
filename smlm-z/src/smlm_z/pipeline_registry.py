"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines.preprocessing import create_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    ml_pipeline = create_pipeline()
    inference_pipeline = ml_pipeline.only_nodes_with_tags("inference")
    training_pipeline_ml = create_pipeline(
        training=ml_pipeline.only_nodes_with_tags("training"),
        inference=inference_pipeline,
        input_name="instances",
    )

    return {
        "__default__": training_pipeline_ml,
    }
