import torch

from ufc_next_word.config import ModelConfig
from ufc_next_word.models import create_model


def test_both_models_return_one_score_per_vocabulary_word() -> None:
    config = ModelConfig(max_context_length=12)
    inputs = torch.zeros((4, config.max_context_length), dtype=torch.long)

    for model_name in ("birnn", "textcnn"):
        model = create_model(model_name, vocab_size=101, padding_idx=0, config=config)
        assert model(inputs).shape == (4, 101)

