"""
CLIP ViT-B/32 with LoRA adapters on blocks 8-11.
Trainable: LoRA params + classification head + text delta + logit scale.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer wrapping an existing nn.Linear."""

    def __init__(self, original: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_f = original.in_features
        out_f = original.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze the original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    @property
    def in_features(self):
        return self.original.in_features

    @property
    def out_features(self):
        return self.original.out_features

    def forward(self, x):
        base = self.original(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora


def _patch_block_attention(block, lora_in):
    """
    Monkey-patch the attention method on a ResidualAttentionBlock so that
    the in_proj LoRA is actually used during the forward pass.

    Without this patch, nn.MultiheadAttention accesses in_proj_weight
    directly through F.multi_head_attention_forward, bypassing any
    LoRA wrapper stored on the module.
    """
    attn = block.attn
    num_heads = attn.num_heads
    head_dim = attn.embed_dim // num_heads

    def patched_attention(x):
        # x shape: (seq_len, batch, embed_dim)
        L, N, E = x.shape

        # Move attn_mask to correct device/dtype if present
        attn_mask = block.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)

        # QKV through LoRA-adapted in_proj
        qkv = lora_in(x)                     # (L, N, 3*E)
        q, k, v = qkv.chunk(3, dim=-1)       # each (L, N, E)

        # Reshape to (N*num_heads, L, head_dim)
        q = q.contiguous().view(L, N * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(L, N * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(L, N * num_heads, head_dim).transpose(0, 1)

        # Scaled dot-product attention (PyTorch 2.0+)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask)

        # Reshape back: (N*num_heads, L, head_dim) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(L, N, E)

        # Output projection (already LoRA-wrapped via inject_lora)
        attn_output = attn.out_proj(attn_output)
        return attn_output

    block.attention = patched_attention


def inject_lora(model, rank, alpha, block_indices):
    """
    Inject LoRA adapters into specified transformer blocks:
      - in_proj (combined Q, K, V projection)
      - out_proj (attention output projection)
      - c_fc, c_proj (MLP layers)
    """
    transformer = model.visual.transformer
    lora_params = []

    for idx in block_indices:
        block = transformer.resblocks[idx]
        attn = block.attn

        # ── in_proj LoRA (Q, K, V combined) ────────────────────────
        if hasattr(attn, "in_proj_weight") and attn.in_proj_weight is not None:
            d = attn.in_proj_weight.shape[1]
            in_proj_linear = nn.Linear(d, 3 * d,
                                       bias=attn.in_proj_bias is not None)
            in_proj_linear.weight = nn.Parameter(attn.in_proj_weight)
            if attn.in_proj_bias is not None:
                in_proj_linear.bias = nn.Parameter(attn.in_proj_bias)
            lora_in = LoRALinear(in_proj_linear, rank, alpha)
            attn._lora_in_proj = lora_in
            lora_params.extend([lora_in.lora_A, lora_in.lora_B])

            # Patch block.attention to route through lora_in
            _patch_block_attention(block, lora_in)

        # ── out_proj LoRA ──────────────────────────────────────────
        lora_out = LoRALinear(attn.out_proj, rank, alpha)
        attn.out_proj = lora_out
        lora_params.extend([lora_out.lora_A, lora_out.lora_B])

        # ── MLP LoRA ──────────────────────────────────────────────
        mlp = block.mlp
        lora_fc1 = LoRALinear(mlp.c_fc, rank, alpha)
        lora_fc2 = LoRALinear(mlp.c_proj, rank, alpha)
        mlp.c_fc = lora_fc1
        mlp.c_proj = lora_fc2
        lora_params.extend([lora_fc1.lora_A, lora_fc1.lora_B])
        lora_params.extend([lora_fc2.lora_A, lora_fc2.lora_B])

    return lora_params


class AgroOntoVLM(nn.Module):
    """
    CLIP ViT-B/32 + LoRA + learnable class-text delta + classification head.
    """

    def __init__(self, num_classes, class_names, clip_model_name="ViT-B/32",
                 lora_rank=8, lora_alpha=16, lora_blocks=None, device="cuda"):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

        # Load CLIP
        model, _ = clip.load(clip_model_name, device=device, jit=False)
        self.clip = model.float()  # ensure float32

        # Freeze everything
        for p in self.clip.parameters():
            p.requires_grad = False

        # Inject LoRA
        if lora_blocks is None:
            lora_blocks = [8, 9, 10, 11]
        self.lora_params = inject_lora(self.clip, lora_rank, lora_alpha,
                                       lora_blocks)

        # Text embeddings for class prompts (frozen base + learnable delta)
        with torch.no_grad():
            prompts = [f"a photo of a {name.replace('_', ' ')}"
                       for name in class_names]
            tokens = clip.tokenize(prompts).to(device)
            text_features = self.clip.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.register_buffer("text_base", text_features)
        self.text_delta = nn.Parameter(torch.zeros_like(text_features))

        # Learnable logit scale (initialised from CLIP's)
        self.logit_scale = nn.Parameter(
            self.clip.logit_scale.detach().clone())

        # Linear classification head
        embed_dim = text_features.shape[-1]  # 512 for ViT-B/32
        self.classifier = nn.Linear(embed_dim, num_classes)

    def encode_image(self, images):
        """Extract image features through LoRA-adapted visual encoder."""
        features = self.clip.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def forward(self, images):
        """
        Returns:
            logits_cls: classification head logits  (B, K)
            logits_clip: CLIP-style cosine similarity logits  (B, K)
            image_features: normalised image embeddings  (B, D)
        """
        image_features = self.encode_image(images)

        # Text features with delta
        text_features = self.text_base + self.text_delta
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # CLIP-style logits
        scale = self.logit_scale.exp()
        logits_clip = scale * (image_features @ text_features.T)

        # Classification head logits
        logits_cls = self.classifier(image_features)

        return logits_cls, logits_clip, image_features

    def get_param_groups(self, lr_lora, lr_head, lr_delta, lr_scale):
        """Separate parameter groups with different learning rates."""
        return [
            {"params": self.lora_params, "lr": lr_lora},
            {"params": [self.text_delta], "lr": lr_delta},
            {"params": [self.logit_scale], "lr": lr_scale},
            {"params": self.classifier.parameters(), "lr": lr_head},
        ]
