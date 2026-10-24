"""
Training loop with ontology-guided hybrid loss.
Loss = λ_CE * CE(one-hot) + λ_uniform * KL(uniform) + λ_onto * KL(onto_soft)
     + λ_proto * MSE(proto_sim, onto_sim)
"""
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from . import config
from .model import AgroOntoVLM
from .ontology import compute_similarity_matrix


def hybrid_loss(logits_cls, logits_clip, labels, onto_sim, num_classes,
                lambda_ce, lambda_uniform, lambda_onto):
    """
    Compute hybrid loss combining:
      1) Cross-entropy with one-hot targets
      2) KL divergence toward uniform distribution
      3) KL divergence toward ontology soft targets
    """
    # 1) Standard cross-entropy
    ce_loss = F.cross_entropy(logits_cls, labels)

    # Combine logits (average of cls head and CLIP-style)
    logits = 0.5 * logits_cls + 0.5 * logits_clip
    log_probs = F.log_softmax(logits, dim=-1)

    # 2) KL toward uniform
    uniform = torch.ones(num_classes, device=logits.device) / num_classes
    uniform = uniform.unsqueeze(0).expand_as(log_probs)
    kl_uniform = F.kl_div(log_probs, uniform, reduction="batchmean")

    # 3) KL toward ontology soft targets
    # Gather the ontology row for each sample's true label
    onto_targets = onto_sim[labels]  # (B, K)
    kl_onto = F.kl_div(log_probs, onto_targets, reduction="batchmean")

    loss = lambda_ce * ce_loss + lambda_uniform * kl_uniform + lambda_onto * kl_onto
    return loss, ce_loss.item(), kl_uniform.item(), kl_onto.item()


def prototype_regulariser(image_features, labels, onto_sim, num_classes):
    """
    MSE between class-prototype cosine similarity and ontology similarity.
    Prototype = mean of normalised image features per class in the batch.
    """
    device = image_features.device
    # Compute per-class prototypes from current batch
    prototypes = torch.zeros(num_classes, image_features.shape[-1], device=device)
    counts = torch.zeros(num_classes, device=device)
    for i in range(num_classes):
        mask = labels == i
        if mask.sum() > 0:
            prototypes[i] = image_features[mask].mean(dim=0)
            counts[i] = 1
    # Normalise
    norms = prototypes.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    prototypes = prototypes / norms
    # Only compute for classes present in batch
    present = counts > 0
    if present.sum() < 2:
        return torch.tensor(0.0, device=device)

    proto_sim = prototypes @ prototypes.T  # (K, K)
    # Mask to present classes only
    mask_2d = present.unsqueeze(0) & present.unsqueeze(1)
    mse = F.mse_loss(proto_sim[mask_2d], onto_sim[mask_2d])
    return mse


def train_one_epoch(model, loader, optimizer, scaler, onto_sim, num_classes,
                    lambda_ce, lambda_uniform, lambda_onto, lambda_proto,
                    device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with autocast():
            logits_cls, logits_clip, img_feat = model(images)
            loss, ce, klu, klo = hybrid_loss(
                logits_cls, logits_clip, labels, onto_sim, num_classes,
                lambda_ce, lambda_uniform, lambda_onto)
            # Prototype graph regulariser
            proto_loss = prototype_regulariser(img_feat, labels, onto_sim,
                                               num_classes)
            loss = loss + lambda_proto * proto_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]],
            config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        preds = logits_cls.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with autocast():
            logits_cls, _, _ = model(images)
        preds = logits_cls.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def run_training(dataset_name, lambda_onto, results_dir=None):
    """
    Full training run for one dataset × one lambda_onto value.
    Returns dict with per-epoch metrics.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}  |  λ_onto: {lambda_onto}  |  Device: {device}")
    print(f"{'='*60}")

    if device == "cpu":
        print("WARNING: Running on CPU — this will be very slow!")

    # ── Data ───────────────────────────────────────────────────────
    from .dataset import build_loaders, get_clip_preprocess
    preprocess = get_clip_preprocess()
    train_loader, val_loader, class_names, num_classes = build_loaders(
        dataset_name, preprocess)

    # ── Ontology ───────────────────────────────────────────────────
    onto_sim = compute_similarity_matrix(
        dataset_name, class_names=class_names, num_classes=num_classes)
    onto_sim = onto_sim.to(device)

    # ── Model ──────────────────────────────────────────────────────
    model = AgroOntoVLM(
        num_classes=num_classes,
        class_names=class_names,
        clip_model_name=config.CLIP_MODEL,
        lora_rank=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_blocks=config.LORA_BLOCKS,
        device=device)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total_params:,} "
          f"({100*trainable/total_params:.2f}%)")

    # ── Optimiser & scheduler ──────────────────────────────────────
    param_groups = model.get_param_groups(
        config.LR_LORA, config.LR_HEAD, config.LR_DELTA, config.LR_LOGIT_SCALE)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS)
    scaler = GradScaler()

    # Adjusted loss weights
    lambda_ce = 1.0 - lambda_onto - config.LAMBDA_UNIFORM
    lambda_uniform = config.LAMBDA_UNIFORM
    lambda_proto = config.LAMBDA_PROTO

    # ── Training ───────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "epoch": []}
    best_val = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, onto_sim, num_classes,
            lambda_ce, lambda_uniform, lambda_onto, lambda_proto, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_acc"].append(round(val_acc, 2))

        print(f"  Epoch {epoch}/{config.EPOCHS}  "
              f"loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
              f"val_acc={val_acc:.2f}%  [{elapsed:.1f}s]"
              + ("  ★" if val_acc >= best_val else ""))

    result = {
        "dataset": dataset_name,
        "lambda_onto": lambda_onto,
        "lambda_ce": round(lambda_ce, 3),
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "final_val_acc": history["val_acc"][-1],
        "epochs": config.EPOCHS,
        "trainable_params": trainable,
        "history": history,
    }

    # Save results
    if results_dir is None:
        results_dir = config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir,
                            f"{dataset_name}_lambda{lambda_onto}.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  ✓ Results saved: {out_file}")
    print(f"  ✓ Best val_acc: {best_val:.2f}% at epoch {best_epoch}")

    return result
