"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines import preprocessing as pre, classify as clf

from kedro_mlflow.pipeline import pipeline_ml_factory

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pre_pipeline = pre.create_pipeline()
    clf_pipeline = clf.create_pipeline()

    pipeline = pre_pipeline + clf_pipeline

    # training_pipeline_ml = pipeline_ml_factory(
    #     training=pipeline.only_nodes_with_tags("training"),
    #     inference=pipeline.only_nodes_with_tags("inference"),
    #     input_name=["spots", "bead_stack"],
    # )

    return {
        "__default__": pipeline,
    }
