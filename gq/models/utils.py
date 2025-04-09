import sys
from gq.utils import ModelConfiguration
from .model import Model


def create_model(params: ModelConfiguration) -> Model:
    """initialize model wih controlled randomness through iterative initializations based on params.init_no

    Args:
        params (ModelConfiguration): all values specifying the model's configuration

    Returns:
        Model: model objects as specified by params.model_name
    """

    model = getattr(sys.modules[__package__], params.model_name)  # type: ignore
    m = None

    for _ in range(params.init_no):
        m = model(params)

    return m  # type: ignore
