"""Neural architectures required by the assignment."""

from .birnn import BiRNNNextWord
from .factory import create_model
from .textcnn import TextCNNNextWord

__all__ = ["BiRNNNextWord", "TextCNNNextWord", "create_model"]

