from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

@dataclass
class TraceResult:
#result container
    model_name: str
    model_id: str
    num_layers: int
    num_facts: int
    mask_patch: list[float] = field(default_factory=list)
    context_patch: list[float] = field(default_factory=list)
    clean_minus_corrupt: float = 0.0

    def peak_layer(self, curve: str = "mask_patch") -> tuple[int, float]:
        scores = getattr(self, curve)
        idx = int(np.argmax(scores))
        return idx, scores[idx]

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "num_layers": self.num_layers,
            "num_facts": self.num_facts,
            "mask_patch": self.mask_patch,
            "context_patch": self.context_patch,
            "clean_minus_corrupt": self.clean_minus_corrupt,
        }


# token position helpers
def _find_subtoken_index(token_ids: list[int], target_ids: list[int]) -> int:
    for i in range(len(token_ids) - len(target_ids) + 1):
        if token_ids[i : i + len(target_ids)] == target_ids:
            return i
    return len(token_ids) - 1


def _find_mask_index(token_ids: list[int], mask_id: int) -> int:
    for i, tid in enumerate(token_ids):
        if tid == mask_id:
            return i
    return len(token_ids) - 1


# get architecture style
def _get_layers(model) -> list:
    """Return the list of transformer layers from supported architectures."""
    # ModernBERT
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    # RoBERTa / BERT
    if hasattr(model, "roberta"):
        return list(model.roberta.encoder.layer)
    if hasattr(model, "bert"):
        return list(model.bert.encoder.layer)
    # DeBERTa
    if hasattr(model, "deberta"):
        return list(model.deberta.encoder.layer)
    # ALBERT
    if hasattr(model, "albert"):
        return list(model.albert.encoder.albert_layer_groups)
    raise ValueError(
        "Unsupported architecture. Expected model.model.layers, "
        "model.roberta.encoder.layer, model.bert.encoder.layer, or similar."
    )


def _get_mlp(layer):
    # get mlp submodule from a single layer
    if hasattr(layer, "mlp"):
        return layer.mlp  # ModernBERT
    if hasattr(layer, "intermediate"):
        return layer.intermediate  # BERT / RoBERTa
    if hasattr(layer, "output"):
        return layer.output  # none from the above
    raise ValueError(f"Cannot find MLP in layer: {type(layer)}")


# main tracer
class CausalTracer:
    def __init__(
        self,
        model_name: str,
        model_id: str,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{self.model_name}] tracing on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_id, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        self._layers = _get_layers(self.model)
        self.num_layers = len(self._layers)


    def _get_activations(self, inputs: dict) -> tuple[torch.Tensor, dict]:
        activations: dict[str, torch.Tensor] = {}

        def _hook(name):
            def fn(module, inp, out):
                act = out[0] if isinstance(out, tuple) else out
                activations[name] = act.detach()
            return fn

        handles = []
        for i, layer in enumerate(self._layers):
            handles.append(_get_mlp(layer).register_forward_hook(_hook(f"layer_{i}")))

        with torch.no_grad():
            logits = self.model(**inputs).logits

        for h in handles:
            h.remove()

        return logits, activations

    def _patch_run(
        self,
        corrupt_inputs: dict,
        clean_acts: dict,
        patch_pairs: list[tuple[int, int]],
        layer_idx: int,
    ) -> torch.Tensor:
        # run corrupt prompt with on layer patched and get the logits

        def _hook(module, inp, out):
            act = out[0] if isinstance(out, tuple) else out
            clean_act = clean_acts[f"layer_{layer_idx}"]
            for ci, ri in patch_pairs:
                if ri < act.shape[1] and ci < clean_act.shape[1]:
                    act[:, ri, :] = clean_act[:, ci, :]
            if isinstance(out, tuple):
                return (act,) + out[1:]
            return act

        handle = _get_mlp(self._layers[layer_idx]).register_forward_hook(_hook)
        with torch.no_grad():
            logits = self.model(**corrupt_inputs).logits
        handle.remove()
        return logits

    def _resolve_index(self, prompt: str, subject: str) -> int:
        ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
        if "[MASK]" in prompt and self.tokenizer.mask_token_id is not None:
            return _find_mask_index(ids, self.tokenizer.mask_token_id)
        sub_ids = self.tokenizer.encode(subject, add_special_tokens=False)
        return _find_subtoken_index(ids, sub_ids)

    def _find_token(self, prompt: str, token_str: str) -> int:
        ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
        tok_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
        return _find_subtoken_index(ids, tok_ids)

    def trace(self, dataset: list[dict], show_progress: bool = True) -> TraceResult:
        mask_scores = np.zeros(self.num_layers)
        ctx_scores = np.zeros(self.num_layers)
        clean_minus_corrupt = 0.0

        items = tqdm(dataset, desc=self.model_name) if show_progress else dataset

        for item in items:
            clean_prompt = item.get("mask_prompt", item["clean_prompt"])
            corrupt_prompt = item.get("corrupt_mask_prompt", item["corrupt_prompt"])
            target_str = item["target"]
            clean_subject = item.get("clean_subject", target_str)
            corrupt_subject = item.get("corrupt_subject", target_str)

            # Tokenise
            corr_inputs = self.tokenizer(
                corrupt_prompt, return_tensors="pt"
            ).to(self.device)
            clean_inputs = self.tokenizer(
                clean_prompt, return_tensors="pt"
            ).to(self.device)

# resolve positions
            target_idx = self._resolve_index(corrupt_prompt, corrupt_subject)
            clean_target_idx = self._resolve_index(clean_prompt, clean_subject)

            ctx_from = item.get("context_from")
            ctx_to = item.get("context_to")
            if ctx_from and ctx_to:
                clean_ctx_idx = self._find_token(clean_prompt, ctx_from)
                corrupt_ctx_idx = self._find_token(corrupt_prompt, ctx_to)
            else:
                clean_ctx_idx = clean_target_idx
                corrupt_ctx_idx = target_idx

            # target token id
            target_id = self.tokenizer.encode(target_str, add_special_tokens=False)[0]

            # Baselines
            with torch.no_grad():
                base_logits = self.model(**corr_inputs).logits
            base_prob = torch.softmax(base_logits[0, target_idx, :], dim=-1)[
                target_id
            ].item()

            clean_logits, clean_acts = self._get_activations(clean_inputs)
            clean_prob = torch.softmax(
                clean_logits[0, clean_target_idx, :], dim=-1
            )[target_id].item()
            clean_minus_corrupt += clean_prob - base_prob

            # patch every layer
            for layer in range(self.num_layers):
                mask_logits = self._patch_run(
                    corr_inputs, clean_acts,
                    [(clean_target_idx, target_idx)], layer,
                )
                ctx_logits = self._patch_run(
                    corr_inputs, clean_acts,
                    [(clean_ctx_idx, corrupt_ctx_idx)], layer,
                )
                mask_prob = torch.softmax(
                    mask_logits[0, target_idx, :], dim=-1
                )[target_id].item()
                ctx_prob = torch.softmax(
                    ctx_logits[0, target_idx, :], dim=-1
                )[target_id].item()

                mask_scores[layer] += mask_prob - base_prob
                ctx_scores[layer] += ctx_prob - base_prob

        n = len(dataset)
        return TraceResult(
            model_name=self.model_name,
            model_id=self.model_id,
            num_layers=self.num_layers,
            num_facts=n,
            mask_patch=(mask_scores / n).tolist(),
            context_patch=(ctx_scores / n).tolist(),
            clean_minus_corrupt=clean_minus_corrupt / n,
        )

    def release(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
