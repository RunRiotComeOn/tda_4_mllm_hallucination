"""
TRAK-CA: Cross-Attention-Only TRAK for MLLM Hallucination Attribution
======================================================================

Novelty over vanilla TRAK:
  Standard TRAK computes gradients over ALL trainable parameters (self-attention
  + cross-attention LoRA weights).  This includes self-attention layers that
  primarily capture language priors / co-occurrence statistics — their gradient
  signal is dominated by "language bias" rather than visual grounding.

  TRAK-CA restricts the gradient collection to **Cross-Attention (CA) layers
  only** — the Vision-to-Language adapter where visual features are injected
  into the text decoder:

    Score(z_train, z_test) = phi_test^T (Phi^T Phi + lambda I)^{-1} phi_i
    where phi = P^T nabla_{theta_CA} L(z)

    theta_CA in {crossattention.self.query, crossattention.self.value}

  By filtering out self-attention / MLP gradient noise, TRAK-CA produces
  attributions that focus on *visual* hallucinations (objects the model
  "sees" but aren't there) rather than *linguistic* hallucinations (objects
  the model predicts from language co-occurrence).

Pipeline:
  1. LoRA fine-tune BLIP on MS COCO Detection (object categories)
  2. Generate object lists and detect hallucinated objects
  3. Compute TRAK-CA influence scores (cross-attention gradients only)
  4. Output top-k most influential training samples

Designed for 8 GB VRAM (FP16, gradient checkpointing).
"""

import torch
import torch.nn as nn
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import gc
import os
import re

# ──────────────────────────────────────────
# 0.  Environment setup
# ──────────────────────────────────────────

OUTPUT_DIR = "results_blip_trak_ca_10k"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True

# ──────────────────────────────────────────
# 1.  Load BLIP + LoRA
# ──────────────────────────────────────────

model_name = "Salesforce/blip-image-captioning-base"

processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to(device)

# Freeze all base parameters
for param in model.parameters():
    param.requires_grad = False

# Insert LoRA into decoder self-attention & cross-attention
# NOTE: LoRA is applied to BOTH self-attn and cross-attn during training
# (needed for good fine-tuning), but for TRAK-CA attribution we only
# collect gradients from the cross-attention subset.
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    # Matches text_decoder.bert.encoder.layer.*.attention.self.{query,value}
    # and crossattention.self.{query,value} — vision encoder uses fused "qkv"
    # so these patterns only hit the text decoder.
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model.gradient_checkpointing_enable()

# ──────────────────────────────────────────
# 1b. Identify Cross-Attention LoRA parameters
# ──────────────────────────────────────────
#
# In BLIP's text decoder (BertLMHeadModel), parameter names follow:
#   - Self-attention:  ...layer.X.attention.self.{query,value}.lora_...
#   - Cross-attention: ...layer.X.crossattention.self.{query,value}.lora_...
#
# We filter by "crossattention" to isolate the Vision-to-Language adapter.

def is_cross_attention_param(name: str) -> bool:
    """Check if a parameter belongs to a cross-attention LoRA adapter."""
    return "crossattention" in name and ("lora_A" in name or "lora_B" in name)


# Enumerate and report
ca_param_names = [n for n, p in model.named_parameters()
                  if p.requires_grad and is_cross_attention_param(n)]
sa_param_names = [n for n, p in model.named_parameters()
                  if p.requires_grad and not is_cross_attention_param(n)]

ca_param_count = sum(p.numel() for n, p in model.named_parameters()
                     if p.requires_grad and is_cross_attention_param(n))
all_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n--- TRAK-CA Parameter Filter ---")
print(f"  Total trainable (LoRA) params : {all_param_count:,}")
print(f"  Cross-attention LoRA params   : {ca_param_count:,}  "
      f"({ca_param_count / all_param_count * 100:.1f}%)")
print(f"  Self-attention LoRA params    : {all_param_count - ca_param_count:,}  "
      f"(excluded from TRAK)")
print(f"  CA LoRA modules ({len(ca_param_names)}):")
for n in ca_param_names:
    print(f"    {n}")

# ──────────────────────────────────────────
# 2.  Load MS COCO Detection (full train + val)
# ──────────────────────────────────────────

train_raw_dataset = load_dataset("detection-datasets/coco", split="train")
eval_raw_dataset = load_dataset("detection-datasets/coco", split="val")
category_names = train_raw_dataset.features["objects"]["category"].feature.names

# TRAIN_SIZE = len(train_raw_dataset)
# EVAL_SIZE = len(eval_raw_dataset)
TRAIN_SIZE = 10000
EVAL_SIZE = 1000

# Truncate actual datasets to match the requested sizes
train_raw_dataset = train_raw_dataset.select(range(TRAIN_SIZE))
eval_raw_dataset = eval_raw_dataset.select(range(EVAL_SIZE))

print(f"Train set: {TRAIN_SIZE} samples")
print(f"Eval set:  {EVAL_SIZE} samples")
print(f"Category names ({len(category_names)}): {category_names[:5]} ...")


def get_object_label(example):
    """Build a deduplicated, sorted object category string for one image."""
    cats = example["objects"]["category"]
    unique_names = sorted(set(category_names[c] for c in cats))
    return "; ".join(unique_names)  # e.g. "car; person; skateboard"

# ──────────────────────────────────────────
# 2b. Preprocessing
# ──────────────────────────────────────────

def _extract_label(example):
    cats = example["objects"]["category"]
    unique_names = sorted(set(category_names[c] for c in cats))
    return {"_label": "; ".join(unique_names)}

print("Extracting train labels ...")
_train_with_labels = train_raw_dataset.map(_extract_label, keep_in_memory=True)
train_raw_labels = _train_with_labels["_label"]

print("Extracting eval labels ...")
_eval_with_labels = eval_raw_dataset.map(_extract_label, keep_in_memory=True)
eval_raw_labels = _eval_with_labels["_label"]


class LazyCocoDataset(torch.utils.data.Dataset):
    """Lazy preprocessing — only processes one sample at a time on __getitem__,
    so the full preprocessed dataset never sits in memory or on disk."""

    def __init__(self, hf_dataset, raw_labels):
        self.hf_dataset = hf_dataset
        self.raw_labels = raw_labels

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        image = example["image"]
        label = self.raw_labels[idx]

        inputs = processor(
            images=image,
            text=label,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}


train_dataset = LazyCocoDataset(train_raw_dataset, train_raw_labels)
eval_dataset = LazyCocoDataset(eval_raw_dataset, eval_raw_labels)

print(f"Train dataset: {len(train_dataset)}, Eval dataset: {len(eval_dataset)}")

# ──────────────────────────────────────────
# 2c. DataLoader with proper collation
# ──────────────────────────────────────────

def collate_fn(batch):
    """Stack tensors; all are already padded to max_length."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}

BATCH_SIZE = 16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 20e9 else 8
print(f"Using batch size: {BATCH_SIZE}")

loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=collate_fn, num_workers=0)

# ──────────────────────────────────────────
# 3.  LoRA fine-tuning (ALL LoRA params — self-attn + cross-attn)
# ──────────────────────────────────────────

NUM_EPOCHS = 2

model.train()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-4,
)

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    print(f"\n=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")
    for batch in tqdm(loader, desc=f"Epoch {epoch + 1}"):
        batch_gpu = {k: v.to(device, dtype=torch.float16)
                     if v.dtype == torch.float32 else v.to(device)
                     for k, v in batch.items()}

        outputs = model(**batch_gpu, labels=batch_gpu["input_ids"])
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss += loss.item()

    avg = epoch_loss / len(loader)
    print(f"  avg loss = {avg:.4f}")
    torch.cuda.empty_cache()

print("\nLoRA fine-tuning done.")
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_before_cleaning"))
print(f"LoRA checkpoint saved to {OUTPUT_DIR}/lora_before_cleaning")

# ──────────────────────────────────────────
# 4.  Detect hallucinations on eval set
# ──────────────────────────────────────────

def parse_gt_objects(text: str) -> set:
    """Parse object names from a '; '-separated GT label string."""
    return {obj.strip().lower() for obj in text.split(";") if obj.strip()}


# Build a lowercase lookup set for all COCO category names
_category_names_lower = [name.lower() for name in category_names]


def extract_generated_objects(text: str) -> set:
    """Extract known COCO category names from (possibly freeform) generated text.

    The model may generate structured 'a; b; c' lists or freeform captions.
    We match every known category that appears as a whole word in the text.
    """
    text_lower = text.lower()
    found = set()
    for name_lower in _category_names_lower:
        # Whole-word boundary check to avoid 'car' matching inside 'card'
        if re.search(r'\b' + re.escape(name_lower) + r'\b', text_lower):
            found.add(name_lower)
    return found


model.eval()
hallucination_cases = []

print(f"\nScanning {len(eval_dataset)} eval samples for hallucinations ...")
for i in tqdm(range(len(eval_dataset)), desc="Hallucination detection"):
    sample = eval_dataset[i]
    image = sample["pixel_values"].unsqueeze(0).to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values=image, max_length=64)
        generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)

    gt_label = eval_raw_labels[i]
    gen_objects = extract_generated_objects(generated_text)
    gt_objects = parse_gt_objects(gt_label)

    hallucinated = gen_objects - gt_objects
    if hallucinated:
        hallucination_cases.append({
            "eval_index": i,
            "generated": generated_text,
            "gt_label": gt_label,
            "hallucinated_objects": hallucinated,
        })

print(f"Hallucination cases found: {len(hallucination_cases)} / {len(eval_dataset)}")

if not hallucination_cases:
    print("No hallucinations detected. Try more samples or fewer epochs.")
    exit(0)

# Save hallucination eval sample images
print(f"\nSaving {len(hallucination_cases)} hallucination eval images to {OUTPUT_DIR}/ ...")
for case_item in hallucination_cases:
    eidx = case_item["eval_index"]
    eval_raw_dataset[eidx]["image"].save(
        os.path.join(OUTPUT_DIR, f"halluc_eval_{eidx}.png"))

# ──────────────────────────────────────────
# 5.  Helper functions for TRAK-CA
# ──────────────────────────────────────────

# TRAK hyperparameters
TRAK_PROJ_DIM = 2048        # projection dimension k
TRAK_LAMBDA = 1e-3          # ridge regularization parameter lambda


def hallucination_loss(eval_index: int, hallucinated_tokens: set):
    """
    Compute a loss that focuses on the hallucinated tokens.

    Strategy: run a forward pass with the model's own generated sequence as
    the target, then sum the log-probabilities at positions corresponding to
    hallucinated tokens.  This gives a scalar whose gradient w.r.t. LoRA
    parameters indicates how training data pushed probability mass onto
    those hallucinated tokens.
    """
    sample = eval_dataset[eval_index]
    image = sample["pixel_values"].unsqueeze(0).to(device)

    # Generate token ids (must be in eval mode to avoid gradient-checkpointing
    # conflicts with the attention mask during generation)
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(pixel_values=image, max_length=64)
    model.train()

    # Forward pass WITH gradient (need grad for TRAK)
    outputs = model(pixel_values=image, input_ids=generated_ids, labels=generated_ids)
    logits = outputs.logits  # (1, seq_len, vocab_size)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    # Map hallucinated nouns to token positions
    token_ids = generated_ids[0].tolist()
    loss_terms = []

    for pos, tid in enumerate(token_ids):
        decoded_token = processor.tokenizer.decode([tid]).strip().lower()
        for hall_noun in hallucinated_tokens:
            if hall_noun in decoded_token or decoded_token in hall_noun:
                loss_terms.append(log_probs[0, pos, tid])
                break

    if not loss_terms:
        # Fallback: use full sequence cross-entropy loss
        return outputs.loss

    return torch.stack(loss_terms).sum()


def get_ca_lora_grad_projected(loss, proj_matrix_gpu):
    """Backprop, collect ONLY cross-attention LoRA gradients, project on GPU.

    This is the core of TRAK-CA: we discard self-attention gradients
    (which carry language-bias signal) and keep only cross-attention
    gradients (which carry visual-grounding signal).

    The gradient vector g_CA has dimension d_CA << d_all, so:
      - The projection matrix P is smaller (d_CA x k)
      - The projection is faster
      - The resulting attributions are focused on visual grounding
    """
    model.zero_grad()
    loss.backward(retain_graph=False)

    grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if is_cross_attention_param(name):
                grads.append(param.grad.detach().flatten())  # stay on GPU

    if not grads:
        return torch.zeros(proj_matrix_gpu.shape[1])

    g_ca = torch.cat(grads).float()         # (d_CA,) on GPU, float32
    phi = proj_matrix_gpu.T @ g_ca           # (k,) on GPU — fast matmul
    return phi.cpu()                         # only k-dim transferred to CPU


# ──────────────────────────────────────────
# 6.  TRAK-CA for ALL hallucination cases
# ──────────────────────────────────────────
#
# Same TRAK algorithm, but with gradient space restricted to L_CA:
#
#   phi(z) = P^T  nabla_{theta_CA} L(z)        (CA-only projected gradient)
#   Phi    = [phi_1, ..., phi_n]^T              (n x k matrix)
#   G^{-1} = (Phi^T Phi + lambda I)^{-1}       (k x k)
#   score(z_test, z_i) = phi_test^T G^{-1} phi_i
#
# Because d_CA is much smaller than d_all, the projection is more
# efficient and the signal-to-noise ratio is higher — MLP and
# self-attention gradient noise is completely eliminated.

MAX_FILTERED = 500   # cap per-hallucinated-object to avoid excessive computation
from collections import defaultdict

# --- 6a. Collect all unique training indices needed across all cases ----------

train_labels_lower = [l.lower() for l in train_raw_labels]
object_to_train_indices = {}   # halluc_object -> list of train indices

all_halluc_objects = set()
for case in hallucination_cases:
    all_halluc_objects |= case["hallucinated_objects"]

print(f"\nUnique hallucinated objects across all cases: {len(all_halluc_objects)}")

for obj in sorted(all_halluc_objects):
    indices = [i for i, lab in enumerate(train_labels_lower) if obj in lab]
    object_to_train_indices[obj] = indices[:MAX_FILTERED]

# Unique training indices that need gradient computation
all_train_indices = set()
for indices in object_to_train_indices.values():
    all_train_indices.update(indices)
all_train_indices = sorted(all_train_indices)
print(f"Unique training samples to compute gradients for: {len(all_train_indices)}")

# --- 6b. Initialize random projection matrix (d_CA x k) ----------------------

d_ca = ca_param_count  # cross-attention LoRA param count only
k = min(TRAK_PROJ_DIM, d_ca)  # projection dim cannot exceed full dim
print(f"\nTRAK-CA: cross-attention gradient dim d_CA = {d_ca}, projection dim k = {k}")
print(f"         (vanilla TRAK would use d_all = {all_param_count})")

# P ∈ R^{d_CA × k}, entries ~ N(0, 1/k) for variance-preserving projection
torch.manual_seed(42)  # reproducibility
proj_matrix_gpu = (torch.randn(d_ca, k) / (k ** 0.5)).to(device)
print(f"Projection matrix: {proj_matrix_gpu.shape} "
      f"({proj_matrix_gpu.nbytes / 1e6:.1f} MB, on {device})")

# --- 6c. Compute projected training gradients (CA-only) -----------------------

model.train()
train_idx_to_pos = {idx: pos for pos, idx in enumerate(all_train_indices)}
n_train = len(all_train_indices)
Phi = torch.zeros(n_train, k)  # Phi in R^{n x k}

GRAD_BATCH_SIZE = BATCH_SIZE

print(f"\nComputing projected CA training gradients (grad_batch={GRAD_BATCH_SIZE}) ...")
for batch_start in tqdm(range(0, n_train, GRAD_BATCH_SIZE),
                        desc="Train gradients (TRAK-CA)"):
    batch_end = min(batch_start + GRAD_BATCH_SIZE, n_train)
    batch_indices = all_train_indices[batch_start:batch_end]

    # Load and collate mini-batch
    samples = [train_dataset[idx] for idx in batch_indices]
    keys = samples[0].keys()
    batch_gpu = {
        k_: torch.stack([s[k_] for s in samples]).to(
            device, dtype=torch.float16 if samples[0][k_].dtype == torch.float32 else None
        ) if samples[0][k_].dtype == torch.float32
        else torch.stack([s[k_] for s in samples]).to(device)
        for k_ in keys
    }

    # Per-sample forward + backward for individual CA gradients
    for j, idx in enumerate(batch_indices):
        pos = batch_start + j
        single = {k_: v[j:j+1] for k_, v in batch_gpu.items()}
        outputs = model(**single, labels=single["input_ids"])
        Phi[pos] = get_ca_lora_grad_projected(outputs.loss, proj_matrix_gpu)

    if batch_end % 200 == 0:
        gc.collect()
        torch.cuda.empty_cache()

print(f"Projected gradient matrix Phi: {Phi.shape} ({Phi.nbytes / 1e6:.1f} MB)")

# --- 6d. Pre-compute the TRAK kernel inverse G^{-1} = (Phi^T Phi + lambda I)^{-1}

print("\nComputing TRAK-CA kernel inverse (Phi^T Phi + lambda I)^{-1} ...")
PhiTPhi = Phi.T @ Phi  # (k, k)
G_inv = torch.linalg.inv(PhiTPhi + TRAK_LAMBDA * torch.eye(k))  # (k, k)
print(f"Kernel matrix: {PhiTPhi.shape}, condition number: {torch.linalg.cond(PhiTPhi).item():.2e}")

# Pre-compute G^{-1} Phi^T for efficient per-test scoring
G_inv_PhiT = G_inv @ Phi.T  # (k, n_train)

del PhiTPhi
gc.collect()

# --- 6e. Per-case: compute test CA projected gradient + TRAK-CA scores --------

train_influence_agg = defaultdict(float)
per_case_results = []

print(f"\nComputing TRAK-CA scores for {len(hallucination_cases)} hallucination cases ...")
for ci, case in enumerate(hallucination_cases):
    case_eval_idx = case["eval_index"]
    hall_tokens = case["hallucinated_objects"]

    # Test gradient for this hallucination case (CA-only, projected on GPU)
    test_loss = hallucination_loss(case_eval_idx, hall_tokens)
    phi_test = get_ca_lora_grad_projected(test_loss, proj_matrix_gpu)  # (k,)

    # TRAK-CA scores: phi_test^T G^{-1} Phi^T
    all_scores = phi_test @ G_inv_PhiT  # (n_train,)

    # Gather relevant training indices for this case's hallucinated objects
    filtered_indices = set()
    for obj in hall_tokens:
        filtered_indices.update(object_to_train_indices.get(obj, []))
    filtered_indices = sorted(filtered_indices)

    # Extract scores for filtered indices
    case_influences = []
    for idx in filtered_indices:
        pos = train_idx_to_pos[idx]
        score = all_scores[pos].item()
        case_influences.append((idx, score))

    case_influences.sort(key=lambda x: -x[1])

    # Accumulate all positive-influence samples for this case
    for idx, score in case_influences:
        if score > 0:
            train_influence_agg[idx] += score

    per_case_results.append({
        "case": case,
        "influences": case_influences,
    })

    if (ci + 1) % 100 == 0 or ci == 0:
        print(f"  [{ci+1}/{len(hallucination_cases)}] "
              f"eval #{case_eval_idx}  halluc={hall_tokens}  "
              f"filtered={len(filtered_indices)}")

# Free TRAK matrices
del proj_matrix_gpu, Phi, G_inv, G_inv_PhiT
gc.collect()
torch.cuda.empty_cache()

# ──────────────────────────────────────────
# 7.  Identify harmful training samples
# ──────────────────────────────────────────

harmful_set = set(train_influence_agg.keys())
harmful_sorted = sorted(train_influence_agg.items(), key=lambda x: -x[1])

print("\n" + "=" * 60)
print(f"HARMFUL TRAINING SAMPLES (positive influence): {len(harmful_set)} / {TRAIN_SIZE}")
print("=" * 60)

for rank, (idx, agg_score) in enumerate(harmful_sorted[:15], 1):
    print(f"  #{rank}  train_idx={idx}  agg_influence={agg_score:.4f}  "
          f"objects: {train_raw_labels[idx]}")

# Save top-10 most harmful training sample images
print(f"\nSaving top-10 harmful training sample images to {OUTPUT_DIR}/ ...")
for rank, (idx, agg_score) in enumerate(harmful_sorted[:10], 1):
    train_raw_dataset[idx]["image"].save(
        os.path.join(OUTPUT_DIR, f"top{rank}_train_{idx}.png"))

# ──────────────────────────────────────────
# 8.  Write first-round summary
# ──────────────────────────────────────────

summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("Hallucination TDA (TRAK-CA) — First Round Summary\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Method: TRAK-CA (Cross-Attention gradients only)\n")
    f.write(f"  Gradient scope: crossattention.self.{{query,value}} LoRA\n")
    f.write(f"  CA params: {ca_param_count:,} / {all_param_count:,} total "
            f"({ca_param_count / all_param_count * 100:.1f}%)\n\n")

    f.write(f"Train size: {TRAIN_SIZE}   Eval size: {EVAL_SIZE}\n")
    f.write(f"TRAK projection dim: {k}   lambda: {TRAK_LAMBDA}\n")
    f.write(f"Hallucination cases (before cleaning): "
            f"{len(hallucination_cases)} / {EVAL_SIZE}\n")
    f.write(f"Harmful training samples to remove: {len(harmful_set)}\n\n")

    for ci, res in enumerate(per_case_results, 1):
        c = res["case"]
        f.write(f"--- Case {ci}: eval_idx={c['eval_index']} ---\n")
        f.write(f"  Generated  : {c['generated']}\n")
        f.write(f"  GT label   : {c['gt_label']}\n")
        f.write(f"  Hallucinated: {', '.join(c['hallucinated_objects'])}\n")
        top5 = res["influences"][:5]
        for r, (idx, sc) in enumerate(top5, 1):
            f.write(f"    top-{r} train_idx={idx}  influence={sc:.4f}  "
                    f"objects: {train_raw_labels[idx]}\n")
        f.write("\n")

    f.write("--- Aggregated Harmful Training Samples (top-15) ---\n")
    for rank, (idx, agg_score) in enumerate(harmful_sorted[:15], 1):
        f.write(f"  #{rank}  train_idx={idx}  agg_influence={agg_score:.4f}  "
                f"objects: {train_raw_labels[idx]}\n")

print(f"First-round summary written to {summary_path}")

# ──────────────────────────────────────────
# 9.  Remove harmful samples & retrain
# ──────────────────────────────────────────

clean_indices = [i for i in range(TRAIN_SIZE) if i not in harmful_set]
print(f"\nCleaned training set: {len(clean_indices)} / {TRAIN_SIZE} "
      f"(removed {len(harmful_set)})")

# Build cleaned training dataset (lazy, same as original)
clean_train_raw = train_raw_dataset.select(clean_indices)
clean_raw_labels = [train_raw_labels[i] for i in clean_indices]
clean_train_ds = LazyCocoDataset(clean_train_raw, clean_raw_labels)

clean_loader = DataLoader(clean_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=0)

# Reload a fresh BLIP + LoRA (reset all weights)
del model
gc.collect()
torch.cuda.empty_cache()

model = BlipForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16,
).to(device)
for param in model.parameters():
    param.requires_grad = False
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()

# Retrain on cleaned data
model.train()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-4,
)

print(f"\n{'=' * 60}")
print("RETRAINING on cleaned dataset ...")
print(f"{'=' * 60}")

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    print(f"\n=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")
    for batch in tqdm(clean_loader, desc=f"Retrain epoch {epoch + 1}"):
        batch_gpu = {k: v.to(device, dtype=torch.float16)
                     if v.dtype == torch.float32 else v.to(device)
                     for k, v in batch.items()}

        outputs = model(**batch_gpu, labels=batch_gpu["input_ids"])
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss += loss.item()

    avg = epoch_loss / len(clean_loader)
    print(f"  avg loss = {avg:.4f}")
    torch.cuda.empty_cache()

print("\nRetraining done.")
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_after_cleaning"))
print(f"LoRA checkpoint saved to {OUTPUT_DIR}/lora_after_cleaning")

# ──────────────────────────────────────────
# 10. Re-evaluate hallucinations after cleaning
# ──────────────────────────────────────────

model.eval()
hallucination_cases_after = []

print(f"\nRe-scanning {len(eval_dataset)} eval samples for hallucinations ...")
for i in tqdm(range(len(eval_dataset)), desc="Re-eval hallucination detection"):
    sample = eval_dataset[i]
    image = sample["pixel_values"].unsqueeze(0).to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values=image, max_length=64)
        generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)

    gt_label = eval_raw_labels[i]
    gen_objects = extract_generated_objects(generated_text)
    gt_objects = parse_gt_objects(gt_label)

    hallucinated = gen_objects - gt_objects
    if hallucinated:
        hallucination_cases_after.append({
            "eval_index": i,
            "generated": generated_text,
            "gt_label": gt_label,
            "hallucinated_objects": hallucinated,
        })

# ──────────────────────────────────────────
# 11. Before vs After comparison
# ──────────────────────────────────────────

n_before = len(hallucination_cases)
n_after = len(hallucination_cases_after)
rate_before = n_before / EVAL_SIZE * 100
rate_after = n_after / EVAL_SIZE * 100

print(f"\n{'=' * 60}")
print("BEFORE vs AFTER DATA CLEANING  (TRAK-CA)")
print(f"{'=' * 60}")
print(f"  Attribution method  : TRAK-CA (cross-attention only)")
print(f"  CA params used      : {ca_param_count:,} / {all_param_count:,}")
print(f"  Training samples    : {TRAIN_SIZE} -> {len(clean_indices)} "
      f"(removed {len(harmful_set)})")
print(f"  Hallucination cases : {n_before} -> {n_after}")
print(f"  Hallucination rate  : {rate_before:.1f}% -> {rate_after:.1f}%")
if n_before > 0:
    reduction = (n_before - n_after) / n_before * 100
    print(f"  Reduction           : {reduction:.1f}%")
print(f"{'=' * 60}")

if hallucination_cases_after:
    print("\nRemaining hallucination cases:")
    for c in hallucination_cases_after:
        print(f"  eval #{c['eval_index']}: generated='{c['generated']}' "
              f"halluc={c['hallucinated_objects']}")

# Append comparison to summary
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("\n\n" + "=" * 60 + "\n")
    f.write("After Data Cleaning — Re-evaluation  (TRAK-CA)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Attribution method  : TRAK-CA (cross-attention only)\n")
    f.write(f"CA params used      : {ca_param_count:,} / {all_param_count:,}\n")
    f.write(f"Training samples    : {TRAIN_SIZE} -> {len(clean_indices)} "
            f"(removed {len(harmful_set)})\n")
    f.write(f"Hallucination cases : {n_before} -> {n_after}\n")
    f.write(f"Hallucination rate  : {rate_before:.1f}% -> {rate_after:.1f}%\n")
    if n_before > 0:
        f.write(f"Reduction           : {reduction:.1f}%\n")
    f.write("\n")

    if hallucination_cases_after:
        f.write("Remaining hallucination cases:\n")
        for c in hallucination_cases_after:
            f.write(f"  eval #{c['eval_index']}: generated='{c['generated']}' "
                    f"halluc={c['hallucinated_objects']}\n")
    else:
        f.write("All hallucinations eliminated!\n")

print(f"\nFull summary updated: {summary_path}")
print("Done.")
