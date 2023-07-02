from essentialmix.core.lm.base import TextCompletionModel
from essentialmix.core.lm.torch_lm import TorchLanguageModel, TorchTextCompletionModel
from essentialmix.core.registry.registry import Registry
from essentialmix.models.gpt2 import GPT2LanguageModel

# Base
TEXT_COMPLETION_MODEL_REGISTRY = Registry[TextCompletionModel](TextCompletionModel)
TEXT_COMPLETION_MODEL_REGISTRY.register(TorchTextCompletionModel)

# Torch
TORCH_LANGUAGE_MODEL_REGISTRY = Registry[TorchLanguageModel](GPT2LanguageModel)
TORCH_LANGUAGE_MODEL_REGISTRY.register(GPT2LanguageModel)
