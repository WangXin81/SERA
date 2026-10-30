import argparse
import copy
import importlib.util
import json
import os
import random
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from sera import clip_encoder as clip

try:
    from sera.low_rank.utils import apply_lora, mark_only_lora_as_trainable, get_lora_parameters
    LORA_AVAILABLE = True
    LORA_IMPORT_ERROR = None
except Exception as exc:
    apply_lora = mark_only_lora_as_trainable = get_lora_parameters = None
    LORA_AVAILABLE = False
    LORA_IMPORT_ERROR = exc
from datasets.caltech101 import Caltech101
from datasets.dtd import DescribableTextures
from datasets.eurosat import EuroSAT
from datasets.fgvc import FGVCAircraft
from datasets.food101 import Food101
from datasets.imagenet import ImageNetFSL, imagenet_classes, imagenet_templates
from datasets.oxford_flowers import OxfordFlowers
from datasets.oxford_pets import OxfordPets
from datasets.stanford_cars import StanfordCars
from datasets.sun397 import SUN397
from datasets.ucf101 import UCF101
from datasets.utils import DatasetWrapper


BACKBONES = {
    "rn50": "RN50",
    "rn101": "RN101",
    "vit_b32": "ViT-B/32",
    "vit_b16": "ViT-B/16",
    "vit_l14": "ViT-L/14",
    "vit_l14_336": "ViT-L/14@336px",
}

DATASET_SPECS = {
    "caltech101": (Caltech101, ("Caltech101", "caltech-101")),
    "dtd": (DescribableTextures, ("dtd", "DTD")),
    "eurosat": (EuroSAT, ("eurosat", "EuroSAT")),
    "fgvc_aircraft": (FGVCAircraft, ("fgvc_aircraft", "fgvc")),
    "fgvc": (FGVCAircraft, ("fgvc_aircraft", "fgvc")),
    "food101": (Food101, ("Food101", "food-101")),
    "oxford_flowers": (OxfordFlowers, ("Flower102", "oxford_flowers")),
    "oxford_pets": (OxfordPets, ("OxfordPets", "oxford_pets")),
    "stanford_cars": (StanfordCars, ("StanfordCars", "stanford_cars")),
    "sun397": (SUN397, ("sun397", "SUN397")),
    "ucf101": (UCF101, ("UCF101", "ucf101")),
    "imagenet": (ImageNetFSL, ("imagenet/images", "imagenet")),
}

VOCAB_FREE_SIMPLE_TEMPLATES = [
    "a photo of a {}.",
    "a clear photo of a {}.",
    "a close-up photo of a {}.",
    "a cropped photo of a {}.",
]

PROMPT_ENSEMBLE_TEMPLATES = [
    "a photo of a {}.",
    "a clear photo of a {}.",
    "a close-up photo of a {}.",
    "a cropped photo of a {}.",
    "a photo of the {}.",
    "a centered photo of a {}.",
    "a detailed photo of a {}.",
    "a natural image of {}.",
]


@dataclass
class ModelConfig:
    dataset_name: str
    shots: int
    lr: float
    seed: int
    train_epoch: int
    batch_size: int
    backbone: str
    num_classes: int
    loss_lambda: Sequence[float]
    fuse_type: int


def parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


PAPER_MODULES = {
    "unified_evidence_construction": "Unified Evidence Construction",
    "residual_evidence_adaptation": "Residual Evidence Adaptation",
    "validation_protected_evidence_routing": "Validation-protected Evidence Routing",
    "multi_view_evidence_aggregation": "Multi-view Evidence Aggregation",
}


def paper_evidence_descriptor(name: str) -> Dict[str, str]:
    exact = {
        "clip": ("Unified Evidence Construction / Base Alignment Evidence", "Semantic Prior Evidence"),
        "mlp": ("Unified Evidence Construction / Base Alignment Evidence", "Visual Adaptation Evidence"),
        "ada": ("Unified Evidence Construction / Base Alignment Evidence", "Residual Logit Evidence"),
        "total": ("Unified Evidence Construction / Base Alignment Evidence", "Aligned Fusion Evidence"),
        "prompt_ensemble": ("Unified Evidence Construction / Text-aligned Evidence", "Prompt Evidence"),
        "prompt_ensemble_adapted": ("Unified Evidence Construction / Text-aligned Evidence", "Aligned Prompt Evidence"),
        "text_residual": ("Unified Evidence Construction / Text-aligned Evidence", "Text Residual Evidence"),
        "residual_evidence": ("Residual Evidence Adaptation", "Residual Adaptation Evidence"),
    }
    if name in exact:
        category, paper_name = exact[name]
    elif name.startswith("text_prior_ridge"):
        category, paper_name = "Unified Evidence Construction / Text-aligned Evidence", "Text-prior Regression Evidence"
    elif name.startswith("ridge_clip"):
        category, paper_name = "Unified Evidence Construction / Support-adaptive Evidence", "Support Regression Evidence"
    elif name.startswith("adaptive_cache_"):
        category, paper_name = "Unified Evidence Construction / Support-adaptive Evidence", "Trainable Support-cache Evidence"
    elif name.startswith("selected_feature_cache"):
        category, paper_name = "Unified Evidence Construction / Support-adaptive Evidence", "Selected-feature Cache Evidence"
    elif name.startswith("svd_proto"):
        category, paper_name = "Unified Evidence Construction / Subspace Evidence", "Subspace Prototype Evidence"
    elif name.startswith("svd_stack"):
        category, paper_name = "Unified Evidence Construction / Subspace Evidence", "Subspace Regression Evidence"
    elif name.startswith("svd_text_prior"):
        category, paper_name = "Unified Evidence Construction / Subspace Evidence", "Subspace Text-prior Evidence"
    elif name.startswith("sim_raw"):
        category, paper_name = "Unified Evidence Construction / Vocabulary-free Relation Evidence", "Direct Relation Evidence"
    elif name.startswith(("sim_z", "sim_teacher", "sim_select")):
        category, paper_name = "Unified Evidence Construction / Vocabulary-free Relation Evidence", "Relational Mapping Evidence"
    elif name.startswith("stack_ridge"):
        category, paper_name = "Unified Evidence Construction / Composite Evidence", "Composite Regression Evidence"
    elif name.startswith("feature_scale"):
        category, paper_name = "Unified Evidence Construction / Text-aligned Evidence", "Feature-scaled Text Evidence"
    else:
        category, paper_name = "Other Evidence", name
    return {
        "internal_name": name,
        "paper_name": paper_name,
        "category": category,
    }


def build_paper_evidence_registry(names: Iterable[str]) -> Dict[str, Dict[str, str]]:
    return {name: paper_evidence_descriptor(name) for name in sorted(set(names))}


def paperize_selection_key(selection_key: object) -> object:
    if not isinstance(selection_key, str):
        return selection_key
    if selection_key == "base_alignment_fusion":
        return "Base Alignment Beta Fusion"
    if selection_key.startswith("single:"):
        internal = selection_key.split(":", 1)[1]
        return "single:" + paper_evidence_descriptor(internal)["paper_name"]
    if selection_key.startswith("prob:"):
        parts = []
        for item in selection_key[5:].split(","):
            internal, weight = item.rsplit("=", 1)
            paper_name = paper_evidence_descriptor(internal)["paper_name"]
            parts.append(f"{paper_name}={weight}")
        return "prob:" + ",".join(parts)
    return selection_key


def parse_args():
    parser = argparse.ArgumentParser(description="SERA: Support Evidence Routing and Adaptation")
    parser.add_argument("--dataset", default="dtd", choices=sorted(DATASET_SPECS))
    parser.add_argument("--root_path", default="./data")
    parser.add_argument("--model_cache_dir", default="./pretrained/clip")
    parser.add_argument("--backbone", default="vit_b16", choices=sorted(BACKBONES))
    parser.add_argument("--shots", "--n_shots", default=16, type=int)
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--loss_lambda", default="1,1,1,1,1")
    parser.add_argument("--fuse_type", default=2, type=int, choices=[1, 2, 3])
    parser.add_argument("--validation_mode", default="official_val", choices=["official_val", "support"])
    parser.add_argument("--selection_mode", "--evidence_routing_mode", dest="selection_mode", default="official_val", choices=["official_val", "support_only", "fixed"], help="Evidence routing protocol: official validation, support-internal selection, or fixed global fusion.")
    parser.add_argument(
        "--evidence_pool_profile",
        default="balanced",
        choices=["balanced", "wide", "compact50", "compact_core", "text_guided", "minimal"],
        help="Default SERA evidence pool. balanced is the paper-facing SERA pool.",
    )
    parser.add_argument("--disable_peft_warmup", action="store_true")
    parser.add_argument("--enable_residual_reference_warmup", action="store_true")
    parser.add_argument("--enable_residual_evidence_adaptation", dest="enable_residual_evidence", action="store_true")
    parser.add_argument("--enable_validation_protected_evidence_routing", dest="enable_validation_protected_routing", action="store_true")
    parser.add_argument("--evidence_routing_guard_margin", dest="evidence_routing_guard_margin", metavar="MARGIN", default=0.0, type=float)
    parser.add_argument("--enable_support_reliability_router", "--enable_support_evidence_routing", dest="enable_support_reliability_router", action="store_true")
    parser.add_argument("--support_router_margin", default=0.0, type=float)
    parser.add_argument("--enable_prototype_compactness_gate", "--enable_compactness_check", dest="enable_prototype_compactness_gate", action="store_true")
    parser.add_argument("--enable_shot_aware_router", "--enable_shot_aware_calibration", dest="enable_shot_aware_router", action="store_true")
    parser.add_argument("--enable_text_prior_preservation", "--enable_text_prior_retention", dest="enable_text_prior_preservation", action="store_true")
    parser.add_argument("--residual_evidence_iters_per_shot", dest="residual_evidence_iters_per_shot", metavar="ITERS", default=500, type=int)
    parser.add_argument("--residual_evidence_batch_size", dest="residual_evidence_batch_size", metavar="BATCH", default=32, type=int)
    parser.add_argument("--lora_warmup_iters_per_shot", default=250, type=int)
    parser.add_argument("--lora_warmup_lr", default=2e-4, type=float)
    parser.add_argument("--lora_warmup_weight_decay", default=1e-2, type=float)
    parser.add_argument("--lora_position", default="all", choices=["bottom", "mid", "up", "half-up", "half-bottom", "all", "top3"])
    parser.add_argument("--lora_encoder", default="both", choices=["text", "vision", "both"])
    parser.add_argument("--lora_params", default="q,k,v")
    parser.add_argument("--lora_rank", default=2, type=int)
    parser.add_argument("--lora_alpha", default=1, type=int)
    parser.add_argument("--lora_dropout", default=0.25, type=float)
    parser.add_argument("--peft_warmup_epochs", default=3, type=int)
    parser.add_argument("--peft_warmup_lr", default=2e-5, type=float)
    parser.add_argument("--peft_warmup_weight_decay", default=1e-4, type=float)
    parser.add_argument("--peft_train_visual_proj", action="store_true")
    parser.add_argument("--disable_vocab_free", action="store_true")
    parser.add_argument("--source_prompt_type", default="imagenet_text_ensemble", choices=["imagenet_text_ensemble", "imagenet_text_full"])
    parser.add_argument("--sim_lambdas", default="0.03,0.1,0.3")
    parser.add_argument("--sim_topk", default="128,256,512")
    parser.add_argument("--sim_teacher_weights", default="0.0,0.1,0.3")
    parser.add_argument("--disable_extra_adapters", action="store_true")
    parser.add_argument("--enable_feature_scale_evidence", action="store_true")
    parser.add_argument("--feature_scale_lambdas", default="0.01,0.0625,0.1,0.3")
    parser.add_argument("--feature_scale_epochs", default=120, type=int)
    parser.add_argument("--feature_scale_lr", default=0.05, type=float)
    parser.add_argument("--feature_scale_kl_weight", default=0.5, type=float)
    parser.add_argument(
        "--disable_subspace_evidence",
        "--disable_svd_adapter",
        dest="disable_svd_adapter",
        action="store_true",
    )
    parser.add_argument("--svd_ranks", default="8,16,32")
    parser.add_argument("--svd_lambdas", default="0.01,0.03,0.1")
    parser.add_argument("--svd_alphas", default="0.25,0.5")
    parser.add_argument(
        "--disable_composite_evidence",
        "--disable_stack_adapter",
        dest="disable_stack_adapter",
        action="store_true",
    )
    parser.add_argument(
        "--disable_auxiliary_support_evidence",
        action="store_true",
        help="Disable the trainable support-cache, selected-feature cache, and text-residual evidence families.",
    )
    parser.add_argument("--stack_lambdas", default="0.1,0.3,1.0")
    parser.add_argument("--stack_top_branches", default=16, type=int)
    parser.add_argument("--ridge_lambdas", default="0.01,0.03,0.1")
    parser.add_argument("--adaptive_cache_betas", default="5.5,10.0")
    parser.add_argument("--adaptive_cache_alphas", default="1.0,2.0")
    parser.add_argument("--adaptive_cache_epochs", default=40, type=int)
    parser.add_argument("--adaptive_cache_lr", default=0.01, type=float)
    parser.add_argument("--text_residual_epochs", default=80, type=int)
    parser.add_argument("--text_residual_lr", default=0.01, type=float)
    parser.add_argument("--selected_feature_counts", default="128,256")
    parser.add_argument("--fusion_grid_step", default=0.05, type=float)
    parser.add_argument("--enable_multi_view_evidence_aggregation", dest="enable_multi_view_evidence_aggregation", action="store_true")
    parser.add_argument("--save_json", default="")
    parser.add_argument("--save_fixed_companion_json", default="")
    parser.add_argument(
        "--save_compact_ablation_json",
        default="",
        help="Save same-run evidence-pool ablations that share one trained model and one set of logits.",
    )
    parser.add_argument(
        "--save_uec_ablation_json",
        default="",
        help="Save paired UEC evidence-family ablations from the same trained model and logits.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def convert_image_to_rgb(image):
    return image.convert("RGB")


def transform_train(size: int):
    return Compose([
        T.RandomResizedCrop(size=size, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def transform_test(size: int):
    return Compose([
        Resize(size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(size),
        convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class FiveCropEvalTransform:
    def __init__(self, size: int):
        self.resize = Resize(size, interpolation=InterpolationMode.BICUBIC)
        self.five_crop = T.FiveCrop(size)
        self.to_tensor = ToTensor()
        self.normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    def __call__(self, image):
        image = convert_image_to_rgb(image)
        image = self.resize(image)
        crops = self.five_crop(image)
        return torch.stack([self.normalize(self.to_tensor(crop)) for crop in crops], dim=0)


def transform_multi_view_evidence(size: int):
    return FiveCropEvalTransform(size)


def resolve_dataset_root(root_path: str, dataset_name: str) -> Path:
    _, candidates = DATASET_SPECS[dataset_name]
    root = Path(root_path)
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    tried = ", ".join(str(root / c) for c in candidates)
    raise FileNotFoundError(f"Could not find dataset {dataset_name}. Tried: {tried}")


def build_dataset(dataset_name: str, root_path: str, shots: int):
    cls, _ = DATASET_SPECS[dataset_name]
    dataset_root = resolve_dataset_root(root_path, dataset_name)
    return dataset_root, cls(str(dataset_root), shots)


def make_loader(items, transform, batch_size: int, num_workers: int, shuffle: bool, drop_last: bool = False):
    return DataLoader(
        DatasetWrapper(items, input_size=224, transform=transform, is_train=shuffle),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


def clean_name(name: str) -> str:
    return name.replace("_", " ")


@torch.no_grad()
def encode_texts(model, texts: Iterable[str], device: torch.device, batch_size: int = 256) -> torch.Tensor:
    texts = list(texts)
    chunks = []
    for start in range(0, len(texts), batch_size):
        tokens = clip.tokenize(texts[start:start + batch_size]).to(device)
        feats = model.encode_text(tokens).float()
        feats = F.normalize(feats, dim=-1)
        chunks.append(feats.cpu())
    return torch.cat(chunks, dim=0).to(device)


@torch.no_grad()
def build_class_text_features(model, classnames: Sequence[str], templates: Sequence[str], cache_path: Path, device: torch.device) -> torch.Tensor:
    if cache_path.exists():
        return torch.load(cache_path, map_location=device).float()
    prompts = []
    for classname in classnames:
        prompts.extend([template.format(clean_name(classname)) for template in templates])
    feats = encode_texts(model, prompts, device)
    feats = feats.view(len(classnames), len(templates), -1).mean(dim=1)
    feats = F.normalize(feats, dim=-1).T.contiguous()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats.cpu(), cache_path)
    return feats.float()


@torch.no_grad()
def build_vocab_free_features(model, prompt_type: str, cache_path: Path, device: torch.device) -> torch.Tensor:
    if cache_path.exists():
        return torch.load(cache_path, map_location=device).float()
    templates = VOCAB_FREE_SIMPLE_TEMPLATES if prompt_type == "imagenet_text_ensemble" else imagenet_templates
    prompts = []
    for classname in imagenet_classes:
        prompts.extend([template.format(clean_name(classname)) for template in templates])
    feats = encode_texts(model, prompts, device)
    feats = feats.view(len(imagenet_classes), len(templates), -1).mean(dim=1)
    feats = F.normalize(feats, dim=-1).T.contiguous()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats.cpu(), cache_path)
    return feats.float()


def freeze_for_base_alignment(model):
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "adapter" in name:
            param.requires_grad = True


def freeze_for_visual_peft(model, train_visual_proj: bool = False) -> int:
    for param in model.parameters():
        param.requires_grad = False
    trainable = 0
    for name, param in model.named_parameters():
        is_visual_norm = name.startswith("visual.") and (
            ".ln_" in name or ".ln_pre" in name or ".ln_post" in name
        )
        is_visual_proj = train_visual_proj and name == "visual.proj"
        is_scale = name == "logit_scale"
        if is_visual_norm or is_visual_proj or is_scale:
            param.requires_grad = True
            trainable += param.numel()
    return trainable


def train_visual_peft_warmup(
    model,
    text_feats: torch.Tensor,
    loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    train_visual_proj: bool = False,
):
    if epochs <= 0:
        return
    trainable = freeze_for_visual_peft(model, train_visual_proj=train_visual_proj)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        print("PEFT warmup skipped: no trainable visual LayerNorm parameters found")
        return
    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay), eps=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, int(epochs) * len(loader)))
    text_feats = F.normalize(text_feats.detach().float(), dim=0).to(device)
    print(f"PEFT warmup trainable_params={trainable} epochs={epochs} lr={lr}")
    for epoch in range(int(epochs)):
        model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        iterator = tqdm(loader, desc=f"peft warmup {epoch}", leave=False)
        for images, labels in iterator:
            images = images.to(device)
            labels = labels.to(device)
            image_feats, _, _ = model.encode_image(images)
            image_feats = F.normalize(image_feats.float(), dim=-1)
            logits = model.logit_scale.exp().float() * (image_feats @ text_feats)
            loss = F.cross_entropy(row_calibrate(logits), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.item())
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += int(labels.numel())
            iterator.set_postfix(loss=loss.item(), acc=correct / max(1, total))
        print(f"peft_epoch={epoch} train_loss={loss_sum / max(1, len(loader)):.4f} train_acc={correct / max(1, total):.4f}")



@torch.no_grad()
def build_class_text_features_uncached(model, classnames: Sequence[str], templates: Sequence[str], device: torch.device) -> torch.Tensor:
    prompts = []
    for classname in classnames:
        prompts.extend([template.format(clean_name(classname)) for template in templates])
    feats = encode_texts(model, prompts, device)
    feats = feats.view(len(classnames), len(templates), -1).mean(dim=1)
    feats = F.normalize(feats, dim=-1).T.contiguous()
    return feats.float()


def _image_feature_tensor(image_output):
    return image_output[0] if isinstance(image_output, (tuple, list)) else image_output


def train_residual_reference_warmup(model, dataset, loader, args, backbone_name: str, device: torch.device) -> bool:
    if not args.enable_residual_reference_warmup:
        return False
    if not LORA_AVAILABLE:
        print(f"Residual warmup skipped: low-rank module import failed: {LORA_IMPORT_ERROR}")
        return False
    params = [p.strip() for p in str(args.lora_params).split(",") if p.strip()]
    lora_args = SimpleNamespace(
        encoder=args.lora_encoder,
        position=args.lora_position,
        params=params,
        r=args.lora_rank,
        alpha=args.lora_alpha,
        dropout_rate=args.lora_dropout,
        backbone=backbone_name,
    )
    print(
        f"Residual warmup enabled encoder={lora_args.encoder} position={lora_args.position} "
        f"params={params} r={lora_args.r} alpha={lora_args.alpha}"
    )
    apply_lora(lora_args, model)
    model.float()
    model.to(device)
    mark_only_lora_as_trainable(model)
    lora_params = get_lora_parameters(model)
    if not lora_params:
        print("Residual warmup skipped: no residual adapter parameters found")
        return False
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=float(args.lora_warmup_lr),
        weight_decay=float(args.lora_warmup_weight_decay),
        eps=1e-4,
    )
    total_iters = max(1, int(args.lora_warmup_iters_per_shot) * int(args.shots))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    class_prompts = [dataset.template[0].format(clean_name(c)) for c in dataset.classnames]
    count_iters = 0
    while count_iters < total_iters:
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        iterator = tqdm(loader, desc=f"REA warmup {count_iters}/{total_iters}", leave=False)
        for images, labels in iterator:
            images = images.to(device)
            labels = labels.to(device)
            if args.lora_encoder in {"text", "both"}:
                tokens = clip.tokenize(class_prompts).to(device)
                text_features = model.encode_text(tokens).float()
                text_features = F.normalize(text_features, dim=-1).T.contiguous()
            else:
                with torch.no_grad():
                    tokens = clip.tokenize(class_prompts).to(device)
                    text_features = model.encode_text(tokens).float()
                    text_features = F.normalize(text_features, dim=-1).T.contiguous()
            image_output = model.encode_image(images)
            image_features = _image_feature_tensor(image_output).float()
            image_features = F.normalize(image_features, dim=-1)
            logits = model.logit_scale.exp().float() * (image_features @ text_features)
            loss = F.cross_entropy(row_calibrate(logits), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.item())
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += int(labels.numel())
            count_iters += 1
            iterator.set_postfix(loss=loss.item(), acc=correct / max(1, total))
            if count_iters >= total_iters:
                break
        print(
            f"residual_warmup_iters={count_iters}/{total_iters} "
            f"train_loss={loss_sum / max(1, len(loader)):.4f} train_acc={correct / max(1, total):.4f}"
        )
    return True


def load_reference_clip_for_lora():
    """Load the vendored reference CLIP package used by residual adaptation."""
    repo_clip = Path(__file__).resolve().parent.parent / "third_party" / "reference_clip" / "clip"
    init_file = repo_clip / "__init__.py"
    if not init_file.exists():
        raise FileNotFoundError(f"Reference CLIP package for residual adaptation not found: {init_file}")
    module_name = "residual_reference_clip"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(init_file),
        submodule_search_locations=[str(repo_clip)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def build_lora_single_template_text_features(ref_clip, model, classnames: Sequence[str], template: str, device: torch.device) -> torch.Tensor:
    texts = [template.format(clean_name(classname)) for classname in classnames]
    tokens = ref_clip.tokenize(texts).to(device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        text_features = model.encode_text(tokens)
    return F.normalize(text_features.float(), dim=-1)


@torch.no_grad()
def collect_lora_clip_logits(ref_clip, model, loader, text_features: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    chunks = []
    text_features = F.normalize(text_features.float(), dim=-1).to(device)
    for images, _ in tqdm(loader, desc="residual evidence eval", leave=False):
        images = images.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            image_features = model.encode_image(images)
        image_features = F.normalize(image_features.float(), dim=-1)
        chunks.append((100.0 * image_features @ text_features.T).detach().cpu())
    return torch.cat(chunks, dim=0).float()


def train_residual_evidence_branch(dataset, args, backbone_name: str, cache_dir: Path, device: torch.device):
    if not args.enable_residual_evidence:
        return None
    if not LORA_AVAILABLE:
        print(f"Residual Evidence Adaptation skipped: low-rank module import failed: {LORA_IMPORT_ERROR}")
        return None

    ref_clip = load_reference_clip_for_lora()
    lora_model, lora_preprocess = ref_clip.load(
        backbone_name,
        device=device,
        jit=False,
        download_root=str(cache_dir),
    )
    lora_model.eval()
    image_size = int(getattr(lora_model.visual, "input_resolution", 224))
    train_transform = Compose([
        T.RandomResizedCrop(size=image_size, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    branch_batch_size = int(args.residual_evidence_batch_size)
    train_loader = make_loader(dataset.train_x, train_transform, branch_batch_size, args.num_workers, shuffle=True)
    support_loader = make_loader(dataset.train_x, lora_preprocess, branch_batch_size, args.num_workers, shuffle=False)
    if args.validation_mode == "official_val":
        val_loader = make_loader(dataset.val, lora_preprocess, branch_batch_size, args.num_workers, shuffle=False)
    else:
        val_loader = support_loader
    test_loader = make_loader(dataset.test, lora_preprocess, branch_batch_size, args.num_workers, shuffle=False)

    params = [p.strip() for p in str(args.lora_params).split(",") if p.strip()]
    lora_args = SimpleNamespace(
        encoder=args.lora_encoder,
        position=args.lora_position,
        params=params,
        r=args.lora_rank,
        alpha=args.lora_alpha,
        dropout_rate=args.lora_dropout,
        backbone=backbone_name,
    )
    print(
        f"Residual Evidence Adaptation enabled encoder={lora_args.encoder} position={lora_args.position} "
        f"params={params} r={lora_args.r} alpha={lora_args.alpha}"
    )
    apply_lora(lora_args, lora_model)
    lora_model.to(device)
    mark_only_lora_as_trainable(lora_model)
    lora_params = get_lora_parameters(lora_model)
    if not lora_params:
        print("Residual Evidence Adaptation skipped: no residual adapter parameters found")
        return None

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=float(args.lora_warmup_lr),
        weight_decay=float(args.lora_warmup_weight_decay),
        betas=(0.9, 0.999),
    )
    total_iters = max(1, int(args.residual_evidence_iters_per_shot) * int(args.shots))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    single_template = dataset.template[0]
    count_iters = 0
    while count_iters < total_iters:
        lora_model.train()
        loss_sum, correct, total = 0.0, 0, 0
        iterator = tqdm(train_loader, desc=f"residual adaptation {count_iters}/{total_iters}", leave=False)
        for images, labels in iterator:
            images = images.to(device)
            labels = labels.to(device)
            if args.lora_encoder in {"text", "both"}:
                text_features = build_lora_single_template_text_features(
                    ref_clip, lora_model, dataset.classnames, single_template, device
                )
            else:
                with torch.no_grad():
                    text_features = build_lora_single_template_text_features(
                        ref_clip, lora_model, dataset.classnames, single_template, device
                    )
            if args.lora_encoder in {"vision", "both"}:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    image_features = lora_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                        image_features = lora_model.encode_image(images)
            image_features = F.normalize(image_features.float(), dim=-1)
            logits = 100.0 * image_features @ text_features.T
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.item())
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += int(labels.numel())
            count_iters += 1
            iterator.set_postfix(loss=loss.item(), acc=correct / max(1, total))
            if count_iters >= total_iters:
                break
        print(
            f"residual_adaptation_iters={count_iters}/{total_iters} "
            f"train_loss={loss_sum / max(1, len(train_loader)):.4f} train_acc={correct / max(1, total):.4f}"
        )

    text_features = build_lora_single_template_text_features(ref_clip, lora_model, dataset.classnames, single_template, device)
    support_logits = collect_lora_clip_logits(ref_clip, lora_model, support_loader, text_features, device)
    val_logits = collect_lora_clip_logits(ref_clip, lora_model, val_loader, text_features, device)
    test_logits = collect_lora_clip_logits(ref_clip, lora_model, test_loader, text_features, device)
    del lora_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "support": {"residual_evidence": support_logits},
        "val": {"residual_evidence": val_logits},
        "test": {"residual_evidence": test_logits},
    }

def forward_logits(model, images: torch.Tensor, text_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
    image_feats, mlp_logits, fused_feats = model.encode_image(images)
    image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
    clip_logits = model.logit_scale.exp() * (image_feats @ text_feats.to(image_feats.dtype))
    ada_logits = clip_logits + model.adapter["adapter_3"](
        model.adapter["adapter_1"](clip_logits) + model.adapter["adapter_2"](fused_feats)
    )
    weight = model.adapter["g_weight"](fused_feats)
    total_logits = weight * mlp_logits + (1.0 - weight) * ada_logits
    return {
        "clip": clip_logits.float(),
        "mlp": mlp_logits.float(),
        "ada": ada_logits.float(),
        "total": total_logits.float(),
        "image_features": image_feats.float(),
        "fused_features": fused_feats.float(),
    }


def base_alignment_loss(labels, logits: Dict[str, torch.Tensor], loss_lambda: Sequence[float]) -> torch.Tensor:
    clip_logits = logits["clip"]
    mlp_logits = logits["mlp"]
    ada_logits = logits["ada"]
    total_logits = logits["total"]
    return (
        F.cross_entropy(mlp_logits, labels) * loss_lambda[0]
        + F.cross_entropy(ada_logits, labels) * loss_lambda[1]
        + F.cross_entropy(total_logits, labels) * loss_lambda[2]
        + F.l1_loss(mlp_logits, clip_logits) * loss_lambda[3]
        + F.l1_loss(ada_logits, clip_logits) * loss_lambda[4]
    )


def train_base_alignment(model, text_feats, loader, config: ModelConfig, device: torch.device):
    freeze_for_base_alignment(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.lr / 10, weight_decay=1e-4, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, config.train_epoch * len(loader)))
    for epoch in range(config.train_epoch):
        model.adapter.train()
        model.visual.adapter.train()
        loss_sum, correct, total = 0.0, 0, 0
        iterator = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
        for images, labels in iterator:
            images = images.to(device)
            labels = labels.to(device)
            logits = forward_logits(model, images, text_feats)
            loss = base_alignment_loss(labels, logits, config.loss_lambda)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.item())
            correct += int((logits["mlp"].argmax(dim=-1) == labels).sum().item())
            total += int(labels.numel())
            iterator.set_postfix(loss=loss.item(), acc=correct / max(1, total))
        print(f"epoch={epoch} train_loss={loss_sum / max(1, len(loader)):.4f} train_acc={correct / max(1, total):.4f}")


@torch.no_grad()
def collect_outputs(model, text_feats, loader, device: torch.device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    model.eval()
    chunks: Dict[str, List[torch.Tensor]] = {"clip": [], "mlp": [], "ada": [], "total": [], "image_features": []}
    labels_out = []
    for images, labels in tqdm(loader, desc="evaluate", leave=False):
        if images.dim() == 5:
            batch_size, num_aug = images.shape[:2]
            flat = images.view(batch_size * num_aug, *images.shape[2:]).to(device)
            out_flat = forward_logits(model, flat, text_feats)
            out = {}
            for key in chunks:
                value = out_flat[key].view(batch_size, num_aug, -1).float()
                if key == "image_features":
                    value = F.normalize(value, dim=-1).mean(dim=1)
                    value = F.normalize(value, dim=-1)
                else:
                    value = value.mean(dim=1)
                out[key] = value
        else:
            out = forward_logits(model, images.to(device), text_feats)
        for key in chunks:
            chunks[key].append(out[key].detach().cpu())
        labels_out.append(labels.long().cpu())
    return {key: torch.cat(value, dim=0).float() for key, value in chunks.items()}, torch.cat(labels_out, dim=0).long()


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels.long(), num_classes=num_classes).float()


def robust_centers(features: torch.Tensor, labels: torch.Tensor, num_classes: int, sharpness: float = 8.0) -> torch.Tensor:
    centers = []
    for cls in range(num_classes):
        cls_features = features[labels == cls]
        raw = F.normalize(cls_features.mean(dim=0), dim=-1)
        sims = cls_features @ raw
        weights = torch.softmax(sharpness * sims, dim=0)
        centers.append(F.normalize((weights[:, None] * cls_features).sum(dim=0), dim=-1))
    return torch.stack(centers, dim=0)


def row_calibrate(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    centered = logits.float() - logits.float().mean(dim=-1, keepdim=True)
    scale = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return centered / scale


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1).cpu() == labels.cpu()).float().mean().item())


def standardize(train_source: torch.Tensor, eval_source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = train_source.mean(dim=0, keepdim=True)
    std = train_source.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
    return (train_source - mean) / std, (eval_source - mean) / std


def solve_source_ridge(source: torch.Tensor, labels: torch.Tensor, num_classes: int, lambda_reg: float, teacher=None, teacher_weight: float = 0.0):
    targets = one_hot(labels, num_classes).to(source.device)
    if teacher is None or teacher_weight <= 0:
        if source.shape[0] < source.shape[1]:
            eye_n = torch.eye(source.shape[0], device=source.device, dtype=source.dtype)
            dual = torch.linalg.solve(source @ source.T + lambda_reg * eye_n, targets)
            return source.T @ dual
        eye = torch.eye(source.shape[-1], device=source.device, dtype=source.dtype)
        return torch.linalg.solve(source.T @ source + lambda_reg * eye, source.T @ targets)
    eye = torch.eye(source.shape[-1], device=source.device, dtype=source.dtype)
    lhs = source.T @ source + (lambda_reg + teacher_weight) * eye
    rhs = source.T @ targets + teacher_weight * teacher.to(source.device).float()
    return torch.linalg.solve(lhs, rhs)


def solve_prior_ridge(features: torch.Tensor, labels: torch.Tensor, prior: torch.Tensor, lambda_reg: float, prior_weight: float):
    num_classes = prior.shape[1]
    targets = one_hot(labels, num_classes).to(features.device)
    eye = torch.eye(features.shape[-1], device=features.device, dtype=features.dtype)
    lhs = features.T @ features + (float(lambda_reg) + float(prior_weight)) * eye
    rhs = features.T @ targets + float(prior_weight) * prior.to(features.device).float()
    return torch.linalg.solve(lhs, rhs)


def train_text_residual_weights(
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    base_text_feats: torch.Tensor,
    epochs: int,
    lr: float,
    residual_scale: float,
    reg_weight: float,
) -> torch.Tensor:
    x = F.normalize(support_features.float(), dim=-1)
    y = support_labels.long()
    base = F.normalize(base_text_feats.float(), dim=0)
    residual = torch.nn.Parameter(torch.zeros_like(base))
    optimizer = torch.optim.AdamW([residual], lr=float(lr), weight_decay=0.0)
    for _ in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        weights = F.normalize(base + float(residual_scale) * residual, dim=0)
        logits = 100.0 * (x @ weights)
        loss = F.cross_entropy(row_calibrate(logits), y) + float(reg_weight) * residual.pow(2).mean()
        loss.backward()
        optimizer.step()
    return F.normalize(base + float(residual_scale) * residual.detach(), dim=0)


def cache_logits(eval_features: torch.Tensor, support_features: torch.Tensor, support_labels: torch.Tensor, num_classes: int, beta: float):
    affinity = eval_features @ support_features.T
    return torch.exp(float(beta) * (affinity - 1.0)) @ one_hot(support_labels, num_classes).to(eval_features.device)


def train_adaptive_cache_keys(
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    base_text_feats: torch.Tensor,
    beta: float,
    alpha: float,
    epochs: int,
    lr: float,
    key_reg: float = 0.02,
) -> torch.Tensor:
    x = F.normalize(support_features.float(), dim=-1)
    y = support_labels.long()
    num_classes = base_text_feats.shape[1]
    values = one_hot(y, num_classes).to(x.device)
    base_logits = (100.0 * (x @ F.normalize(base_text_feats.float(), dim=0))).detach()
    keys = torch.nn.Parameter(x.clone())
    optimizer = torch.optim.AdamW([keys], lr=float(lr), weight_decay=0.0)
    for _ in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        norm_keys = F.normalize(keys, dim=-1)
        cache = torch.exp(float(beta) * (x @ norm_keys.T - 1.0)) @ values
        logits = base_logits + float(alpha) * cache
        loss = F.cross_entropy(row_calibrate(logits), y) + float(key_reg) * (norm_keys - x).pow(2).mean()
        loss.backward()
        optimizer.step()
    return F.normalize(keys.detach(), dim=-1)


def clip_feature_indices(text_feats: torch.Tensor, support_features: torch.Tensor, support_labels: torch.Tensor, feat_num: int) -> torch.Tensor:
    num_classes = text_feats.shape[1]
    means, variances = [], []
    for cls in range(num_classes):
        cls_features = support_features[support_labels == cls]
        means.append(cls_features.mean(dim=0))
        variances.append(cls_features.var(dim=0, unbiased=False))
    means = torch.stack(means, dim=0)
    variances = torch.stack(variances, dim=0)
    between = (means - support_features.mean(dim=0, keepdim=True)).pow(2).mean(dim=0)
    within = variances.mean(dim=0).clamp_min(1e-4)
    text_var = F.normalize(text_feats.float().T, dim=-1).var(dim=0, unbiased=False)
    score = between / within + 0.25 * text_var
    k = max(1, min(int(feat_num), support_features.shape[-1]))
    return torch.topk(score, k=k, dim=0).indices


def build_extra_adapter_logits(
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    eval_features: torch.Tensor,
    base_text_feats: torch.Tensor,
    aux_text_feats: torch.Tensor,
    ridge_lambdas: Sequence[float],
    adaptive_cache_betas: Sequence[float],
    adaptive_cache_alphas: Sequence[float],
    adaptive_cache_epochs: int,
    adaptive_cache_lr: float,
    text_residual_epochs: int,
    text_residual_lr: float,
    selected_feature_counts: Sequence[int],
    enable_auxiliary_support_evidence: bool = True,
    trained_adaptive_cache_keys: Dict[Tuple[float, float], torch.Tensor] = None,
    trained_text_residual: torch.Tensor = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[float, float], torch.Tensor], torch.Tensor]:
    sf = F.normalize(support_features.float(), dim=-1)
    ef = F.normalize(eval_features.float(), dim=-1)
    labels = support_labels.long()
    num_classes = base_text_feats.shape[1]
    base_text = F.normalize(base_text_feats.float(), dim=0)
    aux_text = F.normalize(aux_text_feats.float(), dim=0)
    centers = robust_centers(sf, labels, num_classes)
    center_text = F.normalize(centers.T, dim=0)
    logits: Dict[str, torch.Tensor] = {}

    logits["prompt_ensemble"] = 100.0 * (ef @ aux_text)
    adapted_aux = F.normalize(0.65 * aux_text + 0.35 * center_text, dim=0)
    logits["prompt_ensemble_adapted"] = 100.0 * (ef @ adapted_aux)

    for lam in ridge_lambdas:
        weights = solve_source_ridge(sf, labels, num_classes, lam)
        logits[f"ridge_clip_l{lam:g}"] = 100.0 * (ef @ weights)
        prior_weights = solve_prior_ridge(sf, labels, base_text, lam, prior_weight=0.3)
        logits[f"text_prior_ridge_l{lam:g}"] = 100.0 * (ef @ prior_weights)

    adaptive_cache_keys = {} if trained_adaptive_cache_keys is None else dict(trained_adaptive_cache_keys)
    if enable_auxiliary_support_evidence:
        if trained_text_residual is None:
            trained_text_residual = train_text_residual_weights(
                sf,
                labels,
                base_text,
                text_residual_epochs,
                text_residual_lr,
                residual_scale=1.0,
                reg_weight=0.02,
            )
        logits["text_residual"] = 100.0 * (ef @ trained_text_residual)

        for beta in adaptive_cache_betas:
            for alpha in adaptive_cache_alphas:
                key = (float(beta), float(alpha))
                if key not in adaptive_cache_keys:
                    adaptive_cache_keys[key] = train_adaptive_cache_keys(sf, labels, base_text, beta, alpha, adaptive_cache_epochs, adaptive_cache_lr)
                base_logits = 100.0 * (ef @ base_text)
                cache = cache_logits(ef, adaptive_cache_keys[key], labels, num_classes, beta)
                logits[f"adaptive_cache_b{beta:g}_a{alpha:g}"] = base_logits + float(alpha) * cache

        for feat_num in selected_feature_counts:
            idx = clip_feature_indices(base_text, sf, labels, feat_num)
            sf_sel = F.normalize(sf[:, idx], dim=-1)
            ef_sel = F.normalize(ef[:, idx], dim=-1)
            text_sel = F.normalize(base_text[idx], dim=0)
            logits[f"selected_feature_cache_f{int(feat_num)}"] = (
                100.0 * (ef_sel @ text_sel)
                + 2.0 * cache_logits(ef_sel, sf_sel, labels, num_classes, beta=10.0)
            )

    return logits, adaptive_cache_keys, trained_text_residual


def train_feature_scale_scales(
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    text_feats: torch.Tensor,
    lambdas: Sequence[float],
    epochs: int,
    lr: float,
    kl_weight: float,
    logit_scale: float = 100.0,
) -> Dict[float, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sf = F.normalize(support_features.float(), dim=-1).to(device)
    labels = support_labels.long().to(device)
    text = F.normalize(text_feats.float(), dim=0).to(device)
    base_logits = float(logit_scale) * (sf @ text)
    base_probs = torch.softmax(row_calibrate(base_logits).detach(), dim=-1)
    scales: Dict[float, torch.Tensor] = {}
    for lam in lambdas:
        scale = torch.nn.Parameter(torch.zeros(sf.shape[-1], device=device))
        optimizer = torch.optim.AdamW([scale], lr=float(lr), weight_decay=0.0, eps=1e-6)
        for _ in range(max(1, int(epochs))):
            gated = F.normalize(sf * torch.exp(scale.clamp(-2.0, 2.0)), dim=-1)
            logits = float(logit_scale) * (gated @ text)
            calibrated = row_calibrate(logits)
            ce = F.cross_entropy(calibrated, labels)
            kl = F.kl_div(F.log_softmax(calibrated, dim=-1), base_probs, reduction="batchmean")
            reg = scale.pow(2).mean()
            loss = ce + float(kl_weight) * kl + float(lam) * reg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scales[float(lam)] = scale.detach().cpu()
    return scales


def apply_feature_scale_logits(
    eval_features: torch.Tensor,
    text_feats: torch.Tensor,
    scales: Dict[float, torch.Tensor],
    logit_scale: float = 100.0,
) -> Dict[str, torch.Tensor]:
    ef = F.normalize(eval_features.float(), dim=-1)
    text = F.normalize(text_feats.float(), dim=0)
    logits: Dict[str, torch.Tensor] = {}
    for lam, scale in scales.items():
        gated = F.normalize(ef * torch.exp(scale.float().clamp(-2.0, 2.0)), dim=-1)
        logits[f"feature_scale_l{lam:g}"] = float(logit_scale) * (gated @ text)
    return logits


def class_subspace_basis(
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    text_feats: torch.Tensor,
    max_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_classes = text_feats.shape[1]
    centers = robust_centers(support_features, support_labels, num_classes)
    text = F.normalize(text_feats.float().T, dim=-1)
    anchors = F.normalize(0.55 * centers + 0.45 * text, dim=-1)
    centered = anchors - anchors.mean(dim=0, keepdim=True)
    rank = max(1, min(int(max_rank), centered.shape[0] - 1, centered.shape[1]))
    _, _, vh = torch.linalg.svd(centered.float(), full_matrices=False)
    basis = F.normalize(vh[:rank].T.contiguous(), dim=0)
    return basis, centers


def build_svd_adapter_logits(
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    eval_features: torch.Tensor,
    base_text_feats: torch.Tensor,
    ranks: Sequence[int],
    lambdas: Sequence[float],
    alphas: Sequence[float],
) -> Dict[str, torch.Tensor]:
    sf = F.normalize(support_features.float(), dim=-1)
    ef = F.normalize(eval_features.float(), dim=-1)
    labels = support_labels.long()
    num_classes = base_text_feats.shape[1]
    base_text = F.normalize(base_text_feats.float(), dim=0)
    valid_ranks = sorted({max(1, int(r)) for r in ranks})
    max_rank = min(max(valid_ranks), num_classes - 1, sf.shape[-1])
    basis_full, centers = class_subspace_basis(sf, labels, base_text, max_rank)
    center_text = F.normalize(centers.T, dim=0)
    logits: Dict[str, torch.Tensor] = {}
    for rank in valid_ranks:
        rank = min(rank, basis_full.shape[1])
        if rank <= 0:
            continue
        basis = basis_full[:, :rank]
        sf_proj = sf @ basis
        ef_proj = ef @ basis
        text_proj = base_text.T @ basis
        center_proj = centers @ basis

        for alpha in alphas:
            proto = F.normalize((1.0 - float(alpha)) * base_text.T + float(alpha) * centers, dim=-1)
            full_score = ef @ proto.T
            sub_score = ef_proj @ (proto @ basis).T
            logits[f"svd_proto_k{rank}_a{alpha:g}"] = 100.0 * (full_score + float(alpha) * sub_score)

        compact_train = torch.cat([
            sf @ base_text,
            sf @ center_text,
            sf_proj,
            sf_proj @ text_proj.T,
            sf_proj @ center_proj.T,
        ], dim=1)
        compact_eval = torch.cat([
            ef @ base_text,
            ef @ center_text,
            ef_proj,
            ef_proj @ text_proj.T,
            ef_proj @ center_proj.T,
        ], dim=1)
        compact_train, compact_eval = standardize(compact_train, compact_eval)
        for lam in lambdas:
            mapping = solve_source_ridge(compact_train, labels, num_classes, float(lam))
            logits[f"svd_stack_k{rank}_l{lam:g}"] = 100.0 * (compact_eval @ mapping)
            sub_mapping = solve_source_ridge(
                sf_proj,
                labels,
                num_classes,
                float(lam),
                teacher=text_proj.T,
                teacher_weight=0.15,
            )
            logits[f"svd_text_prior_k{rank}_l{lam:g}"] = 100.0 * (ef_proj @ sub_mapping)
    return logits


def branch_feature_matrix(logits_by_name: Dict[str, torch.Tensor], names: Sequence[str]) -> torch.Tensor:
    return torch.cat([row_calibrate(logits_by_name[name]).float() for name in names], dim=1)


def standardize_with_train(train_source: torch.Tensor, *eval_sources: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    mean = train_source.mean(dim=0, keepdim=True)
    std = train_source.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
    return tuple((source - mean) / std for source in (train_source, *eval_sources))


def build_stack_adapter_logits(
    support_branches: Dict[str, torch.Tensor],
    support_labels: torch.Tensor,
    val_branches: Dict[str, torch.Tensor],
    test_branches: Dict[str, torch.Tensor],
    lambdas: Sequence[float],
    top_branches: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], dict]:
    common = [name for name in support_branches if name in val_branches and name in test_branches]
    if not common:
        return {}, {}, {"stack_enabled": False, "reason": "no common branches"}
    support_scores = {name: accuracy(support_branches[name], support_labels) for name in common}
    names = [
        name for name, _ in sorted(support_scores.items(), key=lambda kv: kv[1], reverse=True)
    ][: max(1, int(top_branches))]
    num_classes = int(support_branches[names[0]].shape[1])
    train_x = branch_feature_matrix(support_branches, names)
    val_x = branch_feature_matrix(val_branches, names)
    test_x = branch_feature_matrix(test_branches, names)
    train_x, val_x, test_x = standardize_with_train(train_x, val_x, test_x)
    val_logits, test_logits = {}, {}
    for lam in lambdas:
        mapping = solve_source_ridge(train_x, support_labels.long(), num_classes, float(lam))
        val_logits[f"stack_ridge_l{lam:g}"] = 100.0 * (val_x @ mapping)
        test_logits[f"stack_ridge_l{lam:g}"] = 100.0 * (test_x @ mapping)
    return val_logits, test_logits, {
        "stack_enabled": True,
        "branches": names,
        "support_branch_acc": support_scores,
    }


def source_teacher(support_features: torch.Tensor, labels: torch.Tensor, text_feats: torch.Tensor, source_proto: torch.Tensor, visual_weight: float, text_weight: float):
    num_classes = text_feats.shape[1]
    centers = []
    for cls in range(num_classes):
        cls_features = support_features[labels == cls]
        centers.append(F.normalize(cls_features.mean(dim=0), dim=-1))
    centers = torch.stack(centers, dim=0)
    vocab = F.normalize(source_proto.float(), dim=0)
    text = F.normalize(text_feats.float().T, dim=-1)
    relation = visual_weight * (centers @ vocab) + text_weight * (text @ vocab)
    return F.normalize(relation, dim=-1).T.contiguous()


def select_source_indices(source: torch.Tensor, labels: torch.Tensor, teacher: torch.Tensor, feat_num: int):
    num_classes = teacher.shape[1]
    means, vars_ = [], []
    for cls in range(num_classes):
        cls_source = source[labels == cls]
        means.append(cls_source.mean(dim=0))
        vars_.append(cls_source.var(dim=0, unbiased=False))
    means = torch.stack(means, dim=0)
    vars_ = torch.stack(vars_, dim=0)
    between = (means - source.mean(dim=0, keepdim=True)).pow(2).mean(dim=0)
    within = vars_.mean(dim=0).clamp_min(1e-4)
    score = between / within + 0.25 * teacher.var(dim=1, unbiased=False)
    k = max(1, min(int(feat_num), source.shape[-1]))
    return torch.topk(score, k=k, dim=0).indices


def build_sim_logits(
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    eval_features: torch.Tensor,
    text_feats: torch.Tensor,
    source_proto: torch.Tensor,
    lambdas: Sequence[float],
    topks: Sequence[int],
    teacher_weights: Sequence[float],
) -> Dict[str, torch.Tensor]:
    support_features = F.normalize(support_features.float(), dim=-1)
    eval_features = F.normalize(eval_features.float(), dim=-1)
    source_proto = F.normalize(source_proto.float(), dim=0)
    num_classes = text_feats.shape[1]
    support_source = support_features @ source_proto
    eval_source = eval_features @ source_proto
    logits = {}
    for lam in lambdas:
        mapping = solve_source_ridge(support_source, support_labels, num_classes, lam)
        logits[f"sim_raw_l{lam:g}"] = 100.0 * (eval_source @ mapping)
        train_z, eval_z = standardize(support_source, eval_source)
        mapping_z = solve_source_ridge(train_z, support_labels, num_classes, lam)
        logits[f"sim_z_l{lam:g}"] = 100.0 * (eval_z @ mapping_z)
        teacher = source_teacher(support_features, support_labels, text_feats, source_proto, 0.6, 0.4)
        for tw in teacher_weights:
            if tw <= 0:
                continue
            mapping_t = solve_source_ridge(train_z, support_labels, num_classes, lam, teacher=teacher, teacher_weight=tw)
            logits[f"sim_teacher_l{lam:g}_w{tw:g}"] = 100.0 * (eval_z @ mapping_t)
            for topk in topks:
                idx = select_source_indices(train_z, support_labels, teacher, topk)
                mapping_s = solve_source_ridge(train_z[:, idx], support_labels, num_classes, lam, teacher=teacher[idx], teacher_weight=tw)
                logits[f"sim_select_l{lam:g}_w{tw:g}_k{topk}"] = 100.0 * (eval_z[:, idx] @ mapping_s)
    return logits


def official_beta_fusion(val_logits: Dict[str, torch.Tensor], val_labels: torch.Tensor, test_logits: Dict[str, torch.Tensor], step: float):
    best_beta, best_acc = 0.0, -1.0
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for beta in grid:
        candidate = float(beta) * val_logits["mlp"] + (1.0 - float(beta)) * val_logits["ada"]
        acc = accuracy(candidate, val_labels)
        if acc > best_acc:
            best_acc, best_beta = acc, float(beta)
    out = best_beta * test_logits["mlp"] + (1.0 - best_beta) * test_logits["ada"]
    return out, {"base_alignment_fusion": best_beta, "base_alignment_fusion_val_acc": best_acc}


def prob_fusion(logits_by_name: Dict[str, torch.Tensor], weights: Dict[str, float]) -> torch.Tensor:
    probs = None
    for name, weight in weights.items():
        branch = torch.softmax(row_calibrate(logits_by_name[name]), dim=-1)
        probs = weight * branch if probs is None else probs + weight * branch
    return probs.clamp_min(1e-12).log()


def generate_weight_grid(names: Sequence[str], step: float):
    names = list(names)
    if len(names) == 1:
        yield {names[0]: 1.0}
        return
    units = max(1, int(round(1.0 / step)))
    if len(names) == 2:
        for i in range(units + 1):
            yield {names[0]: i / units, names[1]: 1.0 - i / units}
        return
    for i in range(units + 1):
        for j in range(units + 1 - i):
            k = units - i - j
            yield {names[0]: i / units, names[1]: j / units, names[2]: k / units}


def fixed_global_fusion(test_logits: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, dict]:
    if "total" in test_logits and "residual_evidence" in test_logits:
        weights = {"total": 0.60, "residual_evidence": 0.40}
        logits = prob_fusion(test_logits, weights)
        paper_weights = {"Aligned Fusion Evidence": 0.60, "Residual Adaptation Evidence": 0.40}
    elif "total" in test_logits:
        weights = {"total": 1.0}
        logits = test_logits["total"]
        paper_weights = {"Aligned Fusion Evidence": 1.0}
    elif "ada" in test_logits:
        weights = {"ada": 1.0}
        logits = test_logits["ada"]
        paper_weights = {"Residual Logit Evidence": 1.0}
    else:
        first = sorted(test_logits)[0]
        weights = {first: 1.0}
        logits = test_logits[first]
        paper_weights = {first: 1.0}
    return logits, {
        "fusion": "fixed_global",
        "selection_key": "fixed:" + ",".join(f"{k}={v:.2f}" for k, v in paper_weights.items()),
        "weights": weights,
        "paper_weights": paper_weights,
        "selection_val_acc": None,
        "branch_val_acc": {},
    }

def select_fusion_by_mode(
    selection_mode: str,
    selection_branches: Dict[str, torch.Tensor],
    selection_labels: torch.Tensor,
    test_branches: Dict[str, torch.Tensor],
    step: float,
) -> Tuple[torch.Tensor, dict]:
    if selection_mode == "fixed":
        logits, info = fixed_global_fusion(test_branches)
        info["selection_mode"] = "fixed"
        info["official_val_used_for_selection"] = False
        info["support_only_selection_used"] = False
        info["fixed_selection_used"] = True
        return logits, info
    logits, info = choose_final_fusion(selection_branches, selection_labels, test_branches, step)
    info["selection_mode"] = selection_mode
    info["official_val_used_for_selection"] = selection_mode == "official_val"
    info["support_only_selection_used"] = selection_mode == "support_only"
    info["fixed_selection_used"] = False
    return logits, info


def normalize_evidence_pool_profile(value: str) -> str:
    aliases = {
        "default": "wide",
        "full_pool": "wide",
        "no_composite_subspace": "compact50",
        "sera_balanced_pool": "balanced",
        "sera_text_guided_pool": "text_guided",
        "sera_minimal_pool": "minimal",
    }
    return aliases.get(str(value), str(value))


def filter_evidence_pool(
    branches: Dict[str, torch.Tensor],
    *,
    remove_composite: bool,
    remove_subspace: bool,
    remove_auxiliary_support: bool,
    pool_profile: str = "wide",
) -> Dict[str, torch.Tensor]:
    pool_profile = normalize_evidence_pool_profile(pool_profile)
    base_alignment_names = {"clip", "mlp", "ada", "total"}
    text_light_names = {
        "prompt_ensemble",
        "prompt_ensemble_adapted",
        "text_prior_ridge_l0.03",
    }
    minimal_vocab_free_names = {
        "sim_raw_l0.1",
        "sim_raw_l0.3",
        "sim_z_l0.1",
        "sim_z_l0.3",
        "sim_teacher_l0.1_w0.1",
        "sim_select_l0.1_w0.1_k256",
    }
    filtered = {}
    for name, logits in branches.items():
        if pool_profile == "balanced":
            if name not in base_alignment_names and not name.startswith("sim_"):
                continue
        elif pool_profile == "text_guided":
            if name not in base_alignment_names and name not in text_light_names and not name.startswith("sim_"):
                continue
        elif pool_profile == "minimal":
            if name not in base_alignment_names and name not in text_light_names and name not in minimal_vocab_free_names:
                continue
        elif pool_profile == "compact50":
            remove_composite = True
            remove_subspace = True
        elif pool_profile == "compact_core":
            remove_composite = True
            remove_subspace = True
            remove_auxiliary_support = True
        if remove_composite and name.startswith("stack_ridge"):
            continue
        if remove_subspace and name.startswith("svd_"):
            continue
        if remove_auxiliary_support and (
            name.startswith("adaptive_cache_")
            or name.startswith("selected_feature_cache")
            or name == "text_residual"
        ):
            continue
        filtered[name] = logits
    return filtered


def evaluate_compact_pool_variant(
    args,
    support_branches: Dict[str, torch.Tensor],
    val_branches: Dict[str, torch.Tensor],
    test_branches: Dict[str, torch.Tensor],
    support_labels: torch.Tensor,
    val_labels: torch.Tensor,
    test_labels: torch.Tensor,
    residual_evidence,
    *,
    remove_composite: bool,
    remove_subspace: bool,
    remove_auxiliary_support: bool,
    pool_profile: str = "default",
) -> dict:
    base_support = filter_evidence_pool(
        support_branches,
        remove_composite=remove_composite,
        remove_subspace=remove_subspace,
        remove_auxiliary_support=remove_auxiliary_support,
        pool_profile=pool_profile,
    )
    base_val = filter_evidence_pool(
        val_branches,
        remove_composite=remove_composite,
        remove_subspace=remove_subspace,
        remove_auxiliary_support=remove_auxiliary_support,
        pool_profile=pool_profile,
    )
    base_test = filter_evidence_pool(
        test_branches,
        remove_composite=remove_composite,
        remove_subspace=remove_subspace,
        remove_auxiliary_support=remove_auxiliary_support,
        pool_profile=pool_profile,
    )
    if args.selection_mode == "support_only":
        selection_branches, selection_labels = base_support, support_labels
    else:
        selection_branches, selection_labels = base_val, val_labels
    base_logits, base_info = select_fusion_by_mode(
        args.selection_mode,
        selection_branches,
        selection_labels,
        base_test,
        args.fusion_grid_step,
    )
    final_logits = base_logits
    final_info = dict(base_info)
    final_test_branches = base_test

    if residual_evidence is not None:
        candidate_support = dict(base_support)
        candidate_val = dict(base_val)
        candidate_test = dict(base_test)
        candidate_support.update(residual_evidence["support"])
        candidate_val.update(residual_evidence["val"])
        candidate_test.update(residual_evidence["test"])
        if args.selection_mode == "support_only":
            candidate_selection, candidate_labels = candidate_support, support_labels
        else:
            candidate_selection, candidate_labels = candidate_val, val_labels
        candidate_logits, candidate_info = select_fusion_by_mode(
            args.selection_mode,
            candidate_selection,
            candidate_labels,
            candidate_test,
            args.fusion_grid_step,
        )
        base_score = float(base_info.get("selection_val_acc", -1.0) or -1.0)
        candidate_score = float(candidate_info.get("selection_val_acc", -1.0) or -1.0)
        if args.selection_mode == "fixed":
            use_candidate = True
        else:
            use_candidate = (
                not args.enable_validation_protected_routing
                or candidate_score >= base_score + float(args.evidence_routing_guard_margin)
            )
        if use_candidate:
            final_logits = candidate_logits
            final_info = dict(candidate_info)
            final_test_branches = candidate_test
            final_info["guard_decision"] = (
                "residual_evidence_selected"
                if "residual_evidence" in str(candidate_info.get("selection_key", ""))
                else "baseline_equivalent_selected"
            )
        else:
            final_info = dict(base_info)
            final_info["guard_decision"] = "unified_evidence_fallback"
        final_info["base_selection_key"] = base_info.get("selection_key")
        final_info["base_selection_val_acc"] = base_score
        final_info["candidate_selection_key"] = candidate_info.get("selection_key")
        final_info["candidate_selection_val_acc"] = candidate_score

    return {
        "final_acc": accuracy(final_logits, test_labels),
        "selection_key": final_info.get("selection_key"),
        "paper_selection_key": paperize_selection_key(final_info.get("selection_key")),
        "guard_decision": final_info.get("guard_decision"),
        "base_branch_count": len(base_test),
        "candidate_branch_count": len(final_test_branches),
        "removed": {
            "composite_evidence": remove_composite,
            "subspace_evidence": remove_subspace,
            "auxiliary_support_evidence": remove_auxiliary_support,
        },
        "pool_profile": pool_profile,
        "branch_names": sorted(final_test_branches),
    }


def filter_uec_evidence_family(
    branches: Dict[str, torch.Tensor],
    family_mode: str,
) -> Dict[str, torch.Tensor]:
    """Keep exactly one UEC evidence family; shared training remains unchanged."""
    base_alignment_names = {"clip", "mlp", "ada", "total"}
    if family_mode == "base_alignment_only":
        return {name: logits for name, logits in branches.items() if name in base_alignment_names}
    if family_mode == "vocabulary_free_relation_only":
        return {name: logits for name, logits in branches.items() if name.startswith("sim_")}
    raise ValueError(f"Unsupported UEC evidence family mode: {family_mode}")


def evaluate_uec_evidence_family_variant(
    args,
    support_branches: Dict[str, torch.Tensor],
    val_branches: Dict[str, torch.Tensor],
    test_branches: Dict[str, torch.Tensor],
    support_labels: torch.Tensor,
    val_labels: torch.Tensor,
    test_labels: torch.Tensor,
    residual_evidence,
    family_mode: str,
) -> dict:
    variant = evaluate_compact_pool_variant(
        args,
        filter_uec_evidence_family(support_branches, family_mode),
        filter_uec_evidence_family(val_branches, family_mode),
        filter_uec_evidence_family(test_branches, family_mode),
        support_labels,
        val_labels,
        test_labels,
        residual_evidence,
        remove_composite=False,
        remove_subspace=False,
        remove_auxiliary_support=False,
        pool_profile="balanced",
    )
    variant["uec_evidence_family_mode"] = family_mode
    variant["shared_feature_training_unchanged"] = True
    variant["removed_from_routing_pool"] = (
        "vocabulary_free_relation_evidence"
        if family_mode == "base_alignment_only"
        else "base_alignment_evidence"
    )
    return variant


def choose_final_fusion(val_logits: Dict[str, torch.Tensor], val_labels: torch.Tensor, test_logits: Dict[str, torch.Tensor], step: float):
    candidates = {}
    if {"mlp", "ada"}.issubset(val_logits) and {"mlp", "ada"}.issubset(test_logits):
        base_alignment_logits, base_alignment_info = official_beta_fusion(val_logits, val_labels, test_logits, step)
        candidates["base_alignment_fusion"] = (
            accuracy(base_alignment_logits, val_labels) if "base_alignment_fusion" in val_logits else base_alignment_info["base_alignment_fusion_val_acc"],
            {"kind": "materialized", "logits": base_alignment_logits, "info": {"fusion": "base_alignment_fusion", **base_alignment_info}},
        )

    branch_scores = {name: accuracy(logits, val_labels) for name, logits in val_logits.items()}
    sorted_names = [name for name, _ in sorted(branch_scores.items(), key=lambda kv: kv[1], reverse=True)]
    base_pool = []
    for name in sorted_names:
        if name not in base_pool:
            base_pool.append(name)
        if len(base_pool) >= 8:
            break
    for name in base_pool:
        candidates[f"single:{name}"] = (
            branch_scores[name],
            {"kind": "single", "name": name, "info": {"fusion": f"single:{name}", "weights": {name: 1.0}}},
        )
    for size in (2, 3):
        pool = base_pool[: min(len(base_pool), 6)]
        for start in range(0, max(1, len(pool) - size + 1)):
            names = pool[start:start + size]
            if len(names) < size:
                continue
            for weights in generate_weight_grid(names, step):
                val_candidate = prob_fusion(val_logits, weights)
                acc = accuracy(val_candidate, val_labels)
                key = "prob:" + ",".join(f"{k}={v:.2f}" for k, v in weights.items())
                if key not in candidates or acc > candidates[key][0]:
                    candidates[key] = (
                        acc,
                        {"kind": "prob", "weights": dict(weights), "info": {"fusion": "prob", "weights": dict(weights)}},
                    )
    best_key, best = max(candidates.items(), key=lambda kv: kv[1][0])
    spec = best[1]
    if spec["kind"] == "materialized":
        final_logits = spec["logits"]
    elif spec["kind"] == "single":
        final_logits = test_logits[spec["name"]]
    else:
        final_logits = prob_fusion(test_logits, spec["weights"])
    info = spec["info"]
    info["selection_key"] = best_key
    info["selection_val_acc"] = best[0]
    info["branch_val_acc"] = branch_scores
    return final_logits, info


def build_fixed_companion_outputs(
    base_test_branches: Dict[str, torch.Tensor],
    candidate_test_branches: Dict[str, torch.Tensor],
    residual_evidence_available: bool,
    stack_info: dict,
    args,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], dict]:
    base_logits, base_info = fixed_global_fusion(base_test_branches)
    base_info["selection_mode"] = "fixed"
    base_info["official_val_used_for_selection"] = False
    base_info["support_only_selection_used"] = False
    base_info["fixed_selection_used"] = True
    base_info["stack_info"] = stack_info
    final_logits = base_logits
    final_branches = base_test_branches
    fusion_info = dict(base_info)

    if residual_evidence_available:
        candidate_logits, candidate_info = fixed_global_fusion(candidate_test_branches)
        candidate_info["selection_mode"] = "fixed"
        candidate_info["official_val_used_for_selection"] = False
        candidate_info["support_only_selection_used"] = False
        candidate_info["fixed_selection_used"] = True
        candidate_info["stack_info"] = stack_info
        final_logits = candidate_logits
        final_branches = candidate_test_branches
        fusion_info = dict(candidate_info)
        if "residual_evidence" in str(candidate_info.get("selection_key", "")):
            fusion_info["guard_decision"] = "residual_evidence_selected"
        else:
            fusion_info["guard_decision"] = "baseline_equivalent_selected"
        fusion_info["base_selection_key"] = base_info.get("selection_key")
        fusion_info["base_selection_val_acc"] = -1.0
        fusion_info["candidate_selection_key"] = candidate_info.get("selection_key")
        fusion_info["candidate_selection_val_acc"] = -1.0
        fusion_info["evidence_routing_guard_margin"] = float(args.evidence_routing_guard_margin)
        fusion_info["validation_protected_routing_enabled"] = args.enable_validation_protected_routing

    fusion_info["selection_mode"] = "fixed"
    fusion_info["official_val_used_for_selection"] = False
    fusion_info["support_only_selection_used"] = False
    fusion_info["fixed_selection_used"] = True
    fusion_info["query_leakage"] = False
    fusion_info["test_label_used"] = False
    fusion_info["router_safety_policy"] = "fixed_selection_protocol"
    fusion_info["sera_module_policy"] = "SERA_balanced_default"
    fusion_info["support_reliability_router_enabled"] = args.enable_support_reliability_router
    fusion_info["prototype_compactness_gate_enabled"] = args.enable_prototype_compactness_gate
    fusion_info["shot_aware_router_enabled"] = args.enable_shot_aware_router
    fusion_info["text_prior_preservation_enabled"] = args.enable_text_prior_preservation
    return final_logits, final_branches, fusion_info


def save_json(path: str, payload: dict):
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    args.evidence_pool_profile = normalize_evidence_pool_profile(args.evidence_pool_profile)
    if args.selection_mode == "official_val":
        sera_protocol_name = "SERA_val"
    elif args.selection_mode == "support_only":
        sera_protocol_name = "SERA_support_only"
    elif args.selection_mode == "fixed":
        sera_protocol_name = "SERA_support_fixed" if args.validation_mode == "support" else "SERA_fixed"
    else:
        sera_protocol_name = f"SERA_{args.selection_mode}"
    disable_extra_adapters = args.disable_extra_adapters or args.evidence_pool_profile == "balanced"
    disable_svd_adapter = args.disable_svd_adapter or args.evidence_pool_profile in {
        "compact50",
        "compact_core",
        "balanced",
        "text_guided",
        "minimal",
    }
    disable_stack_adapter = args.disable_stack_adapter or args.evidence_pool_profile in {
        "compact50",
        "compact_core",
        "balanced",
        "text_guided",
        "minimal",
    }
    disable_auxiliary_support_evidence = args.disable_auxiliary_support_evidence or args.evidence_pool_profile in {
        "compact_core",
        "balanced",
        "text_guided",
        "minimal",
    }
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root, dataset = build_dataset(args.dataset, args.root_path, args.shots)
    num_classes = dataset.num_classes
    backbone_name = BACKBONES[args.backbone]
    cache_dir = Path(args.model_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = ModelConfig(
        dataset_name=args.dataset,
        shots=args.shots,
        lr=args.lr,
        seed=args.seed,
        train_epoch=args.epochs,
        batch_size=args.batch_size,
        backbone=backbone_name,
        num_classes=num_classes,
        loss_lambda=parse_float_list(args.loss_lambda),
        fuse_type=args.fuse_type,
    )
    model, preprocess = clip.load(backbone_name, device=device, download_root=str(cache_dir), num_classes=num_classes, config=config)
    model.eval()
    image_size = int(getattr(model.visual, "input_resolution", 224))

    text_cache = cache_dir / "text_feats" / f"{args.dataset}_{backbone_name.replace('/', '_').replace('-', '_')}.pt"
    text_feats = build_class_text_features(model, dataset.classnames, dataset.template, text_cache, device).to(device)

    train_loader = make_loader(dataset.train_x, transform_train(image_size), args.batch_size, args.num_workers, shuffle=True)
    lora_was_trained = train_residual_reference_warmup(model, dataset, train_loader, args, backbone_name, device)
    if lora_was_trained:
        model.eval()
        text_feats = build_class_text_features_uncached(model, dataset.classnames, dataset.template, device).to(device)
    if not args.disable_peft_warmup:
        train_visual_peft_warmup(
            model,
            text_feats,
            train_loader,
            epochs=args.peft_warmup_epochs,
            lr=args.peft_warmup_lr,
            weight_decay=args.peft_warmup_weight_decay,
            device=device,
            train_visual_proj=args.peft_train_visual_proj,
        )
    train_base_alignment(model, text_feats, train_loader, config, device)
    residual_evidence = train_residual_evidence_branch(dataset, args, backbone_name, cache_dir, device)

    eval_transform = transform_multi_view_evidence(image_size) if args.enable_multi_view_evidence_aggregation else preprocess
    if args.enable_multi_view_evidence_aggregation:
        print(f"Multi-view Evidence Aggregation enabled for support/val/test evaluation: views=5 image_size={image_size}")
    support_eval_loader = make_loader(dataset.train_x, eval_transform, args.batch_size, args.num_workers, shuffle=False)
    if args.validation_mode == "official_val":
        val_loader = make_loader(dataset.val, eval_transform, args.batch_size, args.num_workers, shuffle=False)
    else:
        val_loader = support_eval_loader
    test_loader = make_loader(dataset.test, eval_transform, args.batch_size, args.num_workers, shuffle=False)

    support_logits, support_labels = collect_outputs(model, text_feats, support_eval_loader, device)
    val_logits, val_labels = collect_outputs(model, text_feats, val_loader, device)
    test_logits, test_labels = collect_outputs(model, text_feats, test_loader, device)

    val_branches = {k: v for k, v in val_logits.items() if k in {"clip", "mlp", "ada", "total"}}
    test_branches = {k: v for k, v in test_logits.items() if k in {"clip", "mlp", "ada", "total"}}
    support_branches = {k: v for k, v in support_logits.items() if k in {"clip", "mlp", "ada", "total"}}

    if not disable_extra_adapters:
        aux_text_cache = cache_dir / "text_feats" / f"{args.dataset}_{backbone_name.replace('/', '_').replace('-', '_')}_prompt_ensemble.pt"
        aux_text_feats = build_class_text_features(model, dataset.classnames, PROMPT_ENSEMBLE_TEMPLATES, aux_text_cache, device).cpu()
        ridge_lambdas = parse_float_list(args.ridge_lambdas)
        adaptive_cache_betas = parse_float_list(args.adaptive_cache_betas)
        adaptive_cache_alphas = parse_float_list(args.adaptive_cache_alphas)
        selected_feature_counts = parse_int_list(args.selected_feature_counts)
        extra_val, adaptive_cache_keys, text_residual_weights = build_extra_adapter_logits(
            support_features=support_logits["image_features"],
            support_labels=support_labels,
            eval_features=val_logits["image_features"],
            base_text_feats=text_feats.cpu(),
            aux_text_feats=aux_text_feats,
            ridge_lambdas=ridge_lambdas,
            adaptive_cache_betas=adaptive_cache_betas,
            adaptive_cache_alphas=adaptive_cache_alphas,
            adaptive_cache_epochs=args.adaptive_cache_epochs,
            adaptive_cache_lr=args.adaptive_cache_lr,
            text_residual_epochs=args.text_residual_epochs,
            text_residual_lr=args.text_residual_lr,
            selected_feature_counts=selected_feature_counts,
            enable_auxiliary_support_evidence=not disable_auxiliary_support_evidence,
        )
        extra_support, _, _ = build_extra_adapter_logits(
            support_features=support_logits["image_features"],
            support_labels=support_labels,
            eval_features=support_logits["image_features"],
            base_text_feats=text_feats.cpu(),
            aux_text_feats=aux_text_feats,
            ridge_lambdas=ridge_lambdas,
            adaptive_cache_betas=adaptive_cache_betas,
            adaptive_cache_alphas=adaptive_cache_alphas,
            adaptive_cache_epochs=args.adaptive_cache_epochs,
            adaptive_cache_lr=args.adaptive_cache_lr,
            text_residual_epochs=args.text_residual_epochs,
            text_residual_lr=args.text_residual_lr,
            selected_feature_counts=selected_feature_counts,
            enable_auxiliary_support_evidence=not disable_auxiliary_support_evidence,
            trained_adaptive_cache_keys=adaptive_cache_keys,
            trained_text_residual=text_residual_weights,
        )
        extra_test, _, _ = build_extra_adapter_logits(
            support_features=support_logits["image_features"],
            support_labels=support_labels,
            eval_features=test_logits["image_features"],
            base_text_feats=text_feats.cpu(),
            aux_text_feats=aux_text_feats,
            ridge_lambdas=ridge_lambdas,
            adaptive_cache_betas=adaptive_cache_betas,
            adaptive_cache_alphas=adaptive_cache_alphas,
            adaptive_cache_epochs=args.adaptive_cache_epochs,
            adaptive_cache_lr=args.adaptive_cache_lr,
            text_residual_epochs=args.text_residual_epochs,
            text_residual_lr=args.text_residual_lr,
            selected_feature_counts=selected_feature_counts,
            enable_auxiliary_support_evidence=not disable_auxiliary_support_evidence,
            trained_adaptive_cache_keys=adaptive_cache_keys,
            trained_text_residual=text_residual_weights,
        )
        support_branches.update(extra_support)
        val_branches.update(extra_val)
        test_branches.update(extra_test)

    if args.enable_feature_scale_evidence:
        feature_scales = train_feature_scale_scales(
            support_logits["image_features"],
            support_labels,
            text_feats.cpu(),
            lambdas=parse_float_list(args.feature_scale_lambdas),
            epochs=args.feature_scale_epochs,
            lr=args.feature_scale_lr,
            kl_weight=args.feature_scale_kl_weight,
            logit_scale=float(model.logit_scale.exp().detach().cpu()),
        )
        support_branches.update(apply_feature_scale_logits(
            support_logits["image_features"],
            text_feats.cpu(),
            feature_scales,
            logit_scale=float(model.logit_scale.exp().detach().cpu()),
        ))
        val_branches.update(apply_feature_scale_logits(
            val_logits["image_features"],
            text_feats.cpu(),
            feature_scales,
            logit_scale=float(model.logit_scale.exp().detach().cpu()),
        ))
        test_branches.update(apply_feature_scale_logits(
            test_logits["image_features"],
            text_feats.cpu(),
            feature_scales,
            logit_scale=float(model.logit_scale.exp().detach().cpu()),
        ))

    if not disable_svd_adapter:
        svd_ranks = parse_int_list(args.svd_ranks)
        svd_lambdas = parse_float_list(args.svd_lambdas)
        svd_alphas = parse_float_list(args.svd_alphas)
        svd_support = build_svd_adapter_logits(
            support_logits["image_features"],
            support_labels,
            support_logits["image_features"],
            text_feats.cpu(),
            svd_ranks,
            svd_lambdas,
            svd_alphas,
        )
        svd_val = build_svd_adapter_logits(
            support_logits["image_features"],
            support_labels,
            val_logits["image_features"],
            text_feats.cpu(),
            svd_ranks,
            svd_lambdas,
            svd_alphas,
        )
        svd_test = build_svd_adapter_logits(
            support_logits["image_features"],
            support_labels,
            test_logits["image_features"],
            text_feats.cpu(),
            svd_ranks,
            svd_lambdas,
            svd_alphas,
        )
        support_branches.update(svd_support)
        val_branches.update(svd_val)
        test_branches.update(svd_test)

    if not args.disable_vocab_free:
        source_cache = cache_dir / "text_feats" / f"vocab_free_{args.source_prompt_type}_{backbone_name.replace('/', '_').replace('-', '_')}.pt"
        source_proto = build_vocab_free_features(model, args.source_prompt_type, source_cache, device).cpu()
        lambdas = parse_float_list(args.sim_lambdas)
        topks = [int(v.strip()) for v in args.sim_topk.split(",") if v.strip()]
        teacher_weights = parse_float_list(args.sim_teacher_weights)
        sim_val = build_sim_logits(
            support_logits["image_features"],
            support_labels,
            val_logits["image_features"],
            text_feats.cpu(),
            source_proto,
            lambdas,
            topks,
            teacher_weights,
        )
        sim_support = build_sim_logits(
            support_logits["image_features"],
            support_labels,
            support_logits["image_features"],
            text_feats.cpu(),
            source_proto,
            lambdas,
            topks,
            teacher_weights,
        )
        sim_test = build_sim_logits(
            support_logits["image_features"],
            support_labels,
            test_logits["image_features"],
            text_feats.cpu(),
            source_proto,
            lambdas,
            topks,
            teacher_weights,
        )
        support_branches.update(sim_support)
        val_branches.update(sim_val)
        test_branches.update(sim_test)

    stack_info = {"stack_enabled": False}
    if not disable_stack_adapter:
        stack_val, stack_test, stack_info = build_stack_adapter_logits(
            support_branches=support_branches,
            support_labels=support_labels,
            val_branches=val_branches,
            test_branches=test_branches,
            lambdas=parse_float_list(args.stack_lambdas),
            top_branches=args.stack_top_branches,
        )
        val_branches.update(stack_val)
        test_branches.update(stack_test)

    raw_support_branches = dict(support_branches)
    raw_val_branches = dict(val_branches)
    raw_test_branches = dict(test_branches)
    support_branches = filter_evidence_pool(
        raw_support_branches,
        remove_composite=False,
        remove_subspace=False,
        remove_auxiliary_support=False,
        pool_profile=args.evidence_pool_profile,
    )
    val_branches = filter_evidence_pool(
        raw_val_branches,
        remove_composite=False,
        remove_subspace=False,
        remove_auxiliary_support=False,
        pool_profile=args.evidence_pool_profile,
    )
    test_branches = filter_evidence_pool(
        raw_test_branches,
        remove_composite=False,
        remove_subspace=False,
        remove_auxiliary_support=False,
        pool_profile=args.evidence_pool_profile,
    )

    if args.selection_mode == "support_only":
        selection_branches = support_branches
        selection_labels = support_labels
    else:
        selection_branches = val_branches
        selection_labels = val_labels

    base_final_logits, base_fusion_info = select_fusion_by_mode(
        args.selection_mode,
        selection_branches,
        selection_labels,
        test_branches,
        args.fusion_grid_step,
    )
    base_fusion_info["stack_info"] = stack_info
    base_val_acc_raw = base_fusion_info.get("selection_val_acc", None)
    base_val_acc = float(base_val_acc_raw) if base_val_acc_raw is not None else -1.0

    final_logits = base_final_logits
    fusion_info = dict(base_fusion_info)
    final_val_branches = val_branches
    final_test_branches = test_branches

    if residual_evidence is not None:
        candidate_val_branches = dict(val_branches)
        candidate_test_branches = dict(test_branches)
        candidate_val_branches.update(residual_evidence["val"])
        candidate_test_branches.update(residual_evidence["test"])
        if args.selection_mode == "support_only":
            candidate_selection_branches = dict(support_branches)
            candidate_selection_branches.update(residual_evidence["support"])
            candidate_selection_labels = support_labels
        else:
            candidate_selection_branches = candidate_val_branches
            candidate_selection_labels = val_labels
        candidate_logits, candidate_fusion_info = select_fusion_by_mode(
            args.selection_mode,
            candidate_selection_branches,
            candidate_selection_labels,
            candidate_test_branches,
            args.fusion_grid_step,
        )
        candidate_fusion_info["stack_info"] = stack_info
        candidate_val_acc_raw = candidate_fusion_info.get("selection_val_acc", None)
        candidate_val_acc = float(candidate_val_acc_raw) if candidate_val_acc_raw is not None else -1.0
        guard_margin = float(args.evidence_routing_guard_margin)
        if args.selection_mode == "fixed":
            use_candidate = True
        else:
            use_candidate = (
                not args.enable_validation_protected_routing
                or candidate_val_acc >= base_val_acc + guard_margin
            )
        final_val_branches = candidate_val_branches
        final_test_branches = candidate_test_branches
        if use_candidate:
            final_logits = candidate_logits
            fusion_info = dict(candidate_fusion_info)
            if "residual_evidence" in str(candidate_fusion_info.get("selection_key", "")):
                fusion_info["guard_decision"] = "residual_evidence_selected"
            else:
                fusion_info["guard_decision"] = "baseline_equivalent_selected"
        else:
            fusion_info = dict(base_fusion_info)
            fusion_info["guard_decision"] = "unified_evidence_fallback"
        fusion_info["base_selection_key"] = base_fusion_info.get("selection_key")
        fusion_info["base_selection_val_acc"] = base_val_acc
        fusion_info["candidate_selection_key"] = candidate_fusion_info.get("selection_key")
        fusion_info["candidate_selection_val_acc"] = candidate_val_acc
        fusion_info["evidence_routing_guard_margin"] = guard_margin
        fusion_info["validation_protected_routing_enabled"] = args.enable_validation_protected_routing

    fusion_info["selection_mode"] = args.selection_mode
    fusion_info["official_val_used_for_selection"] = args.selection_mode == "official_val"
    fusion_info["support_only_selection_used"] = args.selection_mode == "support_only"
    fusion_info["fixed_selection_used"] = args.selection_mode == "fixed"
    fusion_info["query_leakage"] = False
    fusion_info["test_label_used"] = False
    fusion_info["router_safety_policy"] = "official_validation_protected_fallback" if args.selection_mode == "official_val" else f"{args.selection_mode}_selection_protocol"
    fusion_info["sera_module_policy"] = "SERA_balanced_default"
    fusion_info["support_reliability_router_enabled"] = args.enable_support_reliability_router
    fusion_info["prototype_compactness_gate_enabled"] = args.enable_prototype_compactness_gate
    fusion_info["shot_aware_router_enabled"] = args.enable_shot_aware_router
    fusion_info["text_prior_preservation_enabled"] = args.enable_text_prior_preservation

    metrics = {f"{name}_acc": accuracy(logits, test_labels) for name, logits in final_test_branches.items()}
    metrics["final_acc"] = accuracy(final_logits, test_labels)

    print("\n================================")
    print(" Final Results (SERA)")
    print("================================")
    print(f"dataset={args.dataset} root={dataset_root}")
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]:.4f}")
    print(f"fusion: {fusion_info}")
    print("================================\n")

    paper_registry = build_paper_evidence_registry(final_test_branches.keys())
    summary = {
        "version": "SERA",
        "dataset": "fgvc_aircraft" if args.dataset == "fgvc" else args.dataset,
        "backbone": args.backbone,
        "shots": args.shots,
        "seed": args.seed,
        "validation_mode": args.validation_mode,
        "uses_extra_training_data": False,
        "uses_query_images_for_training": False,
        "uses_query_labels_for_adaptation": False,
        "uses_query_labels_for_selection": False,
        "selection_mode": args.selection_mode,
        "sera_protocol_name": sera_protocol_name,
        "official_val_used_for_selection": args.selection_mode == "official_val",
        "support_only_selection_used": args.selection_mode == "support_only",
        "fixed_selection_used": args.selection_mode == "fixed",
        "query_leakage": False,
        "test_label_used": False,
        "uses_val_labels_for_selection": args.selection_mode == "official_val" and args.validation_mode == "official_val",
        "vocab_free_enabled": not args.disable_vocab_free,
        "peft_warmup_enabled": not args.disable_peft_warmup,
        "feature_scale_evidence_enabled": args.enable_feature_scale_evidence,
        "evidence_pool_profile": args.evidence_pool_profile,
        "balanced_evidence_pool_default": args.evidence_pool_profile == "balanced",
        "svd_adapter_enabled": not disable_svd_adapter,
        "stack_adapter_enabled": not disable_stack_adapter,
        "auxiliary_support_evidence_enabled": not disable_auxiliary_support_evidence,
        "evidence_pool_configuration": {
            "unified_evidence_construction": True,
            "base_alignment_evidence_enabled": True,
            "vocabulary_free_relation_evidence_enabled": not args.disable_vocab_free,
            "residual_evidence_adaptation_enabled": bool(args.enable_residual_evidence),
            "composite_evidence_enabled": not disable_stack_adapter,
            "subspace_evidence_enabled": not disable_svd_adapter,
            "auxiliary_support_evidence_enabled": not disable_auxiliary_support_evidence,
        },
        "residual_reference_warmup_enabled": args.enable_residual_reference_warmup,
        "residual_evidence_enabled": args.enable_residual_evidence,
        "validation_protected_routing_enabled": args.enable_validation_protected_routing,
        "evidence_routing_guard_margin": args.evidence_routing_guard_margin,
        "multi_view_evidence_aggregation_enabled": args.enable_multi_view_evidence_aggregation,
        "residual_reference_warmup_iters_per_shot": args.lora_warmup_iters_per_shot,
        "residual_evidence_iters_per_shot": args.residual_evidence_iters_per_shot,
        "residual_evidence_batch_size": args.residual_evidence_batch_size,
        "sera_modules": {
            "unified_evidence_construction": True,
            "residual_evidence_adaptation": bool(args.enable_residual_evidence),
            "validation_protected_evidence_routing": True,
            "multi_view_evidence_aggregation": bool(args.enable_multi_view_evidence_aggregation),
        },
        "sera_module_roles": {
            "unified_evidence_construction": "core",
            "residual_evidence_adaptation": "core",
            "validation_protected_evidence_routing": "core",
            "multi_view_evidence_aggregation": "core",
        },
        "sera_aliases": {
            "evidence_pool_profile": args.evidence_pool_profile,
            "evidence_routing_mode": args.selection_mode,
            "enable_residual_evidence_adaptation": args.enable_residual_evidence,
            "enable_validation_protected_evidence_routing": args.enable_validation_protected_routing,
            "enable_multi_view_evidence_aggregation": args.enable_multi_view_evidence_aggregation,
        },
        "paper_modules": PAPER_MODULES,
        "paper_evidence_registry": paper_registry,
        "sera_selection_summary": {
            "selection_key": fusion_info.get("selection_key"),
            "paper_selection_key": paperize_selection_key(fusion_info.get("selection_key")),
            "base_selection_key": fusion_info.get("base_selection_key"),
            "paper_base_selection_key": paperize_selection_key(fusion_info.get("base_selection_key")),
            "candidate_selection_key": fusion_info.get("candidate_selection_key"),
            "paper_candidate_selection_key": paperize_selection_key(fusion_info.get("candidate_selection_key")),
            "guard_decision": fusion_info.get("guard_decision"),
            "router_safety_policy": fusion_info.get("router_safety_policy"),
        },
        "metric_means": {"final": metrics["final_acc"], **metrics},
        "fusion_info": fusion_info,
        "args": vars(args),
    }
    if args.save_compact_ablation_json:
        compact_specs = {
            "wide_pool": (False, False, False, "wide"),
            "compact50_pool": (False, False, False, "compact50"),
            "compact_core_pool": (False, False, False, "compact_core"),
            "balanced_pool": (False, False, False, "balanced"),
            "text_guided_pool": (False, False, False, "text_guided"),
            "minimal_pool": (False, False, False, "minimal"),
        }
        compact_variants = {}
        for variant_name, removed in compact_specs.items():
            compact_variants[variant_name] = evaluate_compact_pool_variant(
                args,
                raw_support_branches,
                raw_val_branches,
                raw_test_branches,
                support_labels,
                val_labels,
                test_labels,
                residual_evidence,
                remove_composite=removed[0],
                remove_subspace=removed[1],
                remove_auxiliary_support=removed[2],
                pool_profile=removed[3],
            )
        full_acc = compact_variants["wide_pool"]["final_acc"]
        for variant in compact_variants.values():
            variant["delta_vs_full_pp"] = 100.0 * (variant["final_acc"] - full_acc)
        compact_payload = {
            "version": "SERA",
            "dataset": summary["dataset"],
            "backbone": args.backbone,
            "shots": args.shots,
            "seed": args.seed,
            "selection_mode": args.selection_mode,
            "same_run_shared_training_and_logits": True,
            "query_leakage": False,
            "test_label_used": False,
            "variants": compact_variants,
        }
        save_json(args.save_compact_ablation_json, compact_payload)
        summary["compact_ablation_json"] = args.save_compact_ablation_json
        summary["compact_ablation_variants"] = compact_variants
    if args.save_uec_ablation_json:
        uec_variants = {}
        for family_mode in ("base_alignment_only", "vocabulary_free_relation_only"):
            variant = evaluate_uec_evidence_family_variant(
                args,
                raw_support_branches,
                raw_val_branches,
                raw_test_branches,
                support_labels,
                val_labels,
                test_labels,
                residual_evidence,
                family_mode,
            )
            variant["delta_vs_full_pp"] = 100.0 * (variant["final_acc"] - metrics["final_acc"])
            uec_variants[family_mode] = variant
        uec_payload = {
            "version": "SERA_UEC_INTERNAL_ABLATION",
            "dataset": summary["dataset"],
            "backbone": args.backbone,
            "shots": args.shots,
            "seed": args.seed,
            "selection_mode": args.selection_mode,
            "same_run_shared_training_and_logits": True,
            "shared_feature_training_unchanged": True,
            "full_final_acc": metrics["final_acc"],
            "query_leakage": False,
            "test_label_used": False,
            "variants": uec_variants,
        }
        save_json(args.save_uec_ablation_json, uec_payload)
        summary["uec_ablation_json"] = args.save_uec_ablation_json
        summary["uec_ablation_variants"] = uec_variants
    save_json(args.save_json, summary)
    if args.save_fixed_companion_json:
        if args.selection_mode != "support_only":
            raise ValueError("--save_fixed_companion_json is only valid with --selection_mode support_only")
        fixed_logits, fixed_test_branches, fixed_fusion_info = build_fixed_companion_outputs(
            base_test_branches=test_branches,
            candidate_test_branches=final_test_branches,
            residual_evidence_available=residual_evidence is not None,
            stack_info=stack_info,
            args=args,
        )
        fixed_metrics = {
            f"{name}_acc": accuracy(logits, test_labels)
            for name, logits in fixed_test_branches.items()
        }
        fixed_metrics["final_acc"] = accuracy(fixed_logits, test_labels)
        fixed_summary = copy.deepcopy(summary)
        fixed_summary["selection_mode"] = "fixed"
        fixed_summary["official_val_used_for_selection"] = False
        fixed_summary["support_only_selection_used"] = False
        fixed_summary["fixed_selection_used"] = True
        fixed_summary["uses_val_labels_for_selection"] = False
        fixed_summary["sera_selection_summary"] = {
            "selection_key": fixed_fusion_info.get("selection_key"),
            "base_selection_key": fixed_fusion_info.get("base_selection_key"),
            "candidate_selection_key": fixed_fusion_info.get("candidate_selection_key"),
            "guard_decision": fixed_fusion_info.get("guard_decision"),
            "router_safety_policy": fixed_fusion_info.get("router_safety_policy"),
        }
        fixed_summary["metric_means"] = {"final": fixed_metrics["final_acc"], **fixed_metrics}
        fixed_summary["fusion_info"] = fixed_fusion_info
        fixed_summary["args"]["selection_mode"] = "fixed"
        fixed_summary["args"]["save_json"] = args.save_fixed_companion_json
        fixed_summary["args"]["save_fixed_companion_json"] = ""
        save_json(args.save_fixed_companion_json, fixed_summary)


if __name__ == "__main__":
    main()
