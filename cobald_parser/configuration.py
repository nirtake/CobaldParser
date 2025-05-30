from transformers import PretrainedConfig


class CobaldParserConfig(PretrainedConfig):
    model_type = "cobald_parser"

    def __init__(
        self,
        encoder_model_name: str = None,
        null_classifier_hidden_size: int = 0,
        lemma_classifier_hidden_size: int = 0,
        morphology_classifier_hidden_size: int = 0,
        dependency_classifier_hidden_size: int = 0,
        misc_classifier_hidden_size: int = 0,
        deepslot_classifier_hidden_size: int = 0,
        semclass_classifier_hidden_size: int = 0,
        activation: str = 'relu',
        dropout: float = 0.1,
        consecutive_null_limit: int = 0,
        vocabulary: dict[dict[int, str]] = {},
        # LoRA params (новое!)
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: list = None,
        **kwargs
    ):
        print("DEBUG (encoder): use_lora:", use_lora)
        print("DEBUG (encoder): lora_target_modules:", lora_target_modules)
        self.encoder_model_name = encoder_model_name
        self.null_classifier_hidden_size = null_classifier_hidden_size
        self.consecutive_null_limit = consecutive_null_limit
        self.lemma_classifier_hidden_size = lemma_classifier_hidden_size
        self.morphology_classifier_hidden_size = morphology_classifier_hidden_size
        self.dependency_classifier_hidden_size = dependency_classifier_hidden_size
        self.misc_classifier_hidden_size = misc_classifier_hidden_size
        self.deepslot_classifier_hidden_size = deepslot_classifier_hidden_size
        self.semclass_classifier_hidden_size = semclass_classifier_hidden_size
        self.activation = activation
        self.dropout = dropout
        # The serialized config stores mappings as strings,
        # e.g. {"0": "acl", "1": "conj"}, so we have to convert them to int.
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.vocabulary = {
            column: {int(k): v for k, v in labels.items()}
            for column, labels in vocabulary.items()
        }
        super().__init__(**kwargs)