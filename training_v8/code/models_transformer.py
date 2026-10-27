import shutil

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    XLMRobertaTokenizerFast,
)


def load_tokenizer(hf_id_or_dir: str):
    if "Multilingual-MiniLM" in hf_id_or_dir:
        return XLMRobertaTokenizerFast.from_pretrained(hf_id_or_dir)
    return AutoTokenizer.from_pretrained(hf_id_or_dir)


class WeightedTrainer(Trainer):

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(outputs.logits.float(), labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def _softmax_pos(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(-1, keepdims=True)
    e = np.exp(logits)
    return e[:, 1] / e.sum(-1)


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    try:
        auc = roc_auc_score(labels, _softmax_pos(logits))
    except ValueError:
        auc = 0.0
    return {"roc_auc": auc}


class _MpsCacheCallback(TrainerCallback):

    def __init__(self, every: int = 50):
        self.every = every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every == 0:
            torch.mps.empty_cache()

    def on_evaluate(self, args, state, control, **kwargs):
        torch.mps.empty_cache()


def _warmup_kwargs(ratio: float) -> dict:
    import inspect
    params = inspect.signature(TrainingArguments.__init__).parameters
    return {"warmup_ratio": ratio} if "warmup_ratio" in params else {"warmup_steps": ratio}


def _length_grouping_kwargs() -> dict:
    import inspect
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "train_sampling_strategy" in params:
        return {"train_sampling_strategy": "group_by_length"}
    if "group_by_length" in params:
        return {"group_by_length": True}
    return {}


def _split_val(texts, labels, groups, val_size, seed):
    if groups is None:
        return train_test_split(texts, labels, test_size=val_size, stratify=labels, random_state=seed)
    from sklearn.model_selection import StratifiedGroupKFold
    n_splits = max(2, round(1 / val_size))
    tr_idx, va_idx = next(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
                          .split(texts, labels, groups))
    pick = lambda idx, xs: [xs[i] for i in idx]
    return pick(tr_idx, texts), pick(va_idx, texts), pick(tr_idx, labels), pick(va_idx, labels)


def fine_tune(cfg, texts, labels, out_dir: str, seed: int = 42, val_size: float = 0.1,
              device: str = "cpu", groups=None):
    tok = load_tokenizer(cfg.hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.hf_id, num_labels=2,
        id2label={0: "benign", 1: "attack"}, label2id={"benign": 0, "attack": 1},
    )
    if getattr(cfg, "freeze_embeddings", False):
        emb = model.get_input_embeddings()
        emb.weight.requires_grad = False
        print(f"  словарные эмбеддинги заморожены ({emb.weight.numel() / 1e6:.0f}M параметров не обучаются)")

    tr_x, va_x, tr_y, va_y = _split_val(list(texts), list(labels), groups, val_size, seed)

    pad = "max_length" if device == "mps" else False

    def make_ds(x, y):
        ds = Dataset.from_dict({"text": x, "label": y})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=cfg.max_length, padding=pad),
                      batched=True, remove_columns=["text"])

    counts = np.bincount(tr_y, minlength=2)
    class_weights = torch.tensor(len(tr_y) / (2 * np.maximum(counts, 1)), dtype=torch.float)

    ckpt_dir = out_dir + "_ckpt"
    args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        **_warmup_kwargs(cfg.warmup_ratio),
        **_length_grouping_kwargs(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        seed=seed,
        use_cpu=(device == "cpu"),
        fp16=(device == "cuda"),
        dataloader_pin_memory=(device == "cuda"),
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=make_ds(tr_x, tr_y),
        eval_dataset=make_ds(va_x, va_y),
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
                  + ([_MpsCacheCallback()] if device == "mps" else []),
        class_weights=class_weights,
    )
    trainer.train()
    model = _reload_best(trainer, device)
    model.best_val_auc = float(trainer.state.best_metric or 0.0)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    return model, tok


_LEGACY = (("LayerNorm.gamma", "LayerNorm.weight"), ("LayerNorm.beta", "LayerNorm.bias"))


def _modern(key: str) -> str:
    for old, new in _LEGACY:
        if key.endswith(old):
            return key[: -len(old)] + new
    return key


def _reload_best(trainer, device):
    best = trainer.state.best_model_checkpoint
    if not best:
        return trainer.model
    from safetensors.numpy import load_file
    model = AutoModelForSequenceClassification.from_pretrained(best)
    file_w = {_modern(k): v for k, v in load_file(f"{best}/model.safetensors").items()}
    loaded = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()
              if not k.endswith("position_ids")}
    bad = [k for k, v in loaded.items()
           if k not in file_w or v.shape != file_w[k].shape or not np.allclose(v, file_w[k])]
    if bad:
        raise RuntimeError(f"Лучший чекпойнт загрузился не полностью: {bad[:5]}")
    print(f"  лучший чекпойнт: {best.rsplit('/', 1)[-1]} (ROC-AUC {trainer.state.best_metric:.4f}), "
          f"веса сверены с файлом")
    return model.to(trainer.model.device).eval()


@torch.no_grad()
def predict_proba(model, tok, texts, max_length: int, stride: int, batch_size: int = 32) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    texts = list(texts)
    stride = min(stride, max_length // 4)
    is_mps = device.type == "mps"
    pad = "max_length" if is_mps else True
    out = np.zeros(len(texts), dtype=np.float32)
    for s in range(0, len(texts), batch_size):
        batch = texts[s:s + batch_size]
        enc = tok(batch, truncation=True, max_length=max_length, stride=stride,
                  return_overflowing_tokens=True, padding=pad, return_tensors="pt")
        mapping = np.asarray(enc.pop("overflow_to_sample_mapping"))
        enc = {k: v.to(device) for k, v in enc.items()}
        probs = torch.softmax(model(**enc).logits.float(), -1)[:, 1].cpu().numpy()
        agg = np.zeros(len(batch), dtype=np.float32)
        np.maximum.at(agg, mapping, probs)
        out[s:s + len(batch)] = agg
        if is_mps and (s // batch_size) % 50 == 0:
            torch.mps.empty_cache()
    return out
