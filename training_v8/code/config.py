from dataclasses import dataclass, field


@dataclass
class TransformerCfg:
    name: str
    hf_id: str
    max_length: int = 256
    epochs: int = 3
    batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    freeze_embeddings: bool = False


@dataclass
class Config:
    output_dir: str = "artifacts"
    seed: int = 42
    device: str = "cpu"
    num_threads: int | None = None

    n_folds: int = 7
    val_size: float = 0.1

    transformers: list = field(default_factory=lambda: [
        TransformerCfg(name="rubert_tiny2", hf_id="cointegrated/rubert-tiny2",
                       max_length=128, epochs=3, batch_size=32, lr=5e-5),
        TransformerCfg(name="minilm", hf_id="microsoft/Multilingual-MiniLM-L12-H384",
                       max_length=128, epochs=3, batch_size=32, lr=2e-5, warmup_ratio=0.1,
                       freeze_embeddings=True),
    ])

    min_val_auc: float = 0.80
    max_retries: int = 2

    max_attack_ratio: float | None = 3.0
    protected_sources: tuple = ("PromptSC", "synthetic_domain")

    stride: int = 32

    threshold_mode: str = "recall"
    target_recall: float = 0.85
    gray_recall: float = 0.95
    min_precision: float = 0.40
    min_recall: float = 0.75
    target_fpr: float = 0.01
    gray_fpr: float = 0.05
    min_ru_benign_for_threshold: int = 200

    early_benign: float = 0.02
    early_attack: float = 0.98

    prompt_guard_id: str = "meta-llama/Llama-Prompt-Guard-2-86M"
    prompt_guard_low: float = 0.3
    prompt_guard_high: float = 0.7
