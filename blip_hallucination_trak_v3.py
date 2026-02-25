"""
TRAK-CA v3: Full-Dataset Cross-Attention TRAK for MLLM Hallucination Attribution
=================================================================================

Improvement over v2:
  v2 used a label-based "suspect pre-filter" (MAX_FILTERED=500) that only
  computed gradients for training samples whose labels contain the
  hallucinated object name.  This is fundamentally flawed:

    - It turns "attribution" into "verification" — asking "is it one of these
      500 suspects?" instead of "who did it?"
    - It misses the real culprits: a "mop" sample that looks like a "dog",
      a noisy background that triggers a false detection, or co-occurrence
      patterns from semantically unrelated images.
    - The kernel matrix G = Phi^T Phi only reflects the filtered subset,
      not the true training distribution, distorting the ridge regression.

  v3 fixes this by computing TRAK-CA over the **entire training set**:

    1. Compute phi_i = P^T nabla_{theta_CA} L(z_i)  for ALL N training samples
    2. Build Phi in R^{N x k}  from the full training set
    3. G^{-1} = (Phi^T Phi + lambda I)^{-1}  reflects the true covariance
    4. For each hallucination case, score ALL N training samples:
         score_i = phi_test^T G^{-1} phi_i

  This is the theoretically correct TRAK formulation.  The computational
  cost is O(N) gradient computations (one-time), which is feasible because:
    - CA-only gradients have small dimension d_CA
    - Random projection reduces storage to k floats per sample
    - The kernel inverse is only k x k (fast)
    - Scoring is a single matrix-vector product

Pipeline:
  1. LoRA fine-tune BLIP on MS COCO Detection (object categories)
  2. Generate object lists and detect hallucinated objects
  3. Compute full-dataset TRAK-CA influence scores (CA gradients, all samples)
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
from collections import defaultdict
import random
import gc
import os
import re

# ──────────────────────────────────────────
# 0.  Environment setup
# ──────────────────────────────────────────

OUTPUT_DIR = "results_blip_trak_ca_v3_0.05"
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

for param in model.parameters():
    param.requires_grad = False

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
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

def is_cross_attention_param(name: str) -> bool:
    """Check if a parameter belongs to a cross-attention LoRA adapter."""
    return "crossattention" in name and ("lora_A" in name or "lora_B" in name)


ca_param_names = [n for n, p in model.named_parameters()
                  if p.requires_grad and is_cross_attention_param(n)]

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
# 2.  Load MS COCO Detection
# ──────────────────────────────────────────

train_raw_dataset = load_dataset("detection-datasets/coco", split="train")
eval_raw_dataset = load_dataset("detection-datasets/coco", split="val")
category_names = train_raw_dataset.features["objects"]["category"].feature.names

# TRAIN_SIZE = len(train_raw_dataset)
# EVAL_SIZE = len(eval_raw_dataset)
TRAIN_SIZE = 10000
EVAL_SIZE = 1000

train_raw_dataset = train_raw_dataset.select(range(TRAIN_SIZE))
eval_raw_dataset = eval_raw_dataset.select(range(EVAL_SIZE))

print(f"Train set: {TRAIN_SIZE} samples")
print(f"Eval set:  {EVAL_SIZE} samples")
print(f"Category names ({len(category_names)}): {category_names[:5]} ...")

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
# 2c. DataLoader
# ──────────────────────────────────────────

def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}

# BATCH_SIZE = 16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 20e9 else 8
BATCH_SIZE = 32
print(f"Using batch size: {BATCH_SIZE}")

loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=collate_fn, num_workers=0)

# ──────────────────────────────────────────
# 3.  LoRA fine-tuning
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
    return {obj.strip().lower() for obj in text.split(";") if obj.strip()}

_category_names_lower = [name.lower() for name in category_names]

def extract_generated_objects(text: str) -> set:
    text_lower = text.lower()
    found = set()
    for name_lower in _category_names_lower:
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

TRAK_PROJ_DIM = 2048
TRAK_LAMBDA = 1e-3
TOP_K_PER_CASE = 50   # report top-k per case (full ranking is computed)
REMOVE_FRACTION = 0.05  # remove top 5% most harmful training samples


def hallucination_loss(eval_index: int, hallucinated_tokens: set):
    """Compute a loss focused on hallucinated token positions."""
    sample = eval_dataset[eval_index]
    image = sample["pixel_values"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(pixel_values=image, max_length=64)
    model.train()

    outputs = model(pixel_values=image, input_ids=generated_ids, labels=generated_ids)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    token_ids = generated_ids[0].tolist()
    loss_terms = []

    for pos, tid in enumerate(token_ids):
        decoded_token = processor.tokenizer.decode([tid]).strip().lower()
        for hall_noun in hallucinated_tokens:
            if hall_noun in decoded_token or decoded_token in hall_noun:
                loss_terms.append(log_probs[0, pos, tid])
                break

    if not loss_terms:
        return outputs.loss

    return torch.stack(loss_terms).sum()


def get_ca_lora_grad_projected(loss, proj_matrix_gpu):
    """Backprop, collect ONLY cross-attention LoRA gradients, project on GPU."""
    model.zero_grad()
    loss.backward(retain_graph=False)

    grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if is_cross_attention_param(name):
                grads.append(param.grad.detach().flatten())  # stay on GPU

    if not grads:
        return torch.zeros(proj_matrix_gpu.shape[1])

    g_ca = torch.cat(grads).float()
    phi = proj_matrix_gpu.T @ g_ca
    return phi.cpu()


# ──────────────────────────────────────────
# 6.  Full-dataset TRAK-CA
# ──────────────────────────────────────────
#
# v3: NO label-based pre-filtering.  We compute projected gradients for
# every single training sample, build the full kernel, and score against
# the entire training set for each hallucination case.
#
#   phi_i  = P^T nabla_{theta_CA} L(z_i)   for i = 1, ..., N  (ALL samples)
#   Phi    = [phi_1, ..., phi_N]^T          in R^{N x k}
#   G^{-1} = (Phi^T Phi + lambda I)^{-1}   in R^{k x k}
#   scores = phi_test^T  G^{-1}  Phi^T      in R^{N}
#
# The kernel G now reflects the TRUE training distribution, not a biased
# subset.  Attribution can discover unexpected culprits (e.g., a "mop"
# sample causing "dog" hallucinations via texture similarity).

# --- 6a. Initialize random projection matrix (d_CA x k) ----------------------

d_ca = ca_param_count
k = min(TRAK_PROJ_DIM, d_ca)
print(f"\nTRAK-CA v3 (full-dataset): d_CA = {d_ca}, projection dim k = {k}")
print(f"Will compute gradients for ALL {TRAIN_SIZE} training samples.")

torch.manual_seed(42)
proj_matrix_gpu = (torch.randn(d_ca, k) / (k ** 0.5)).to(device)
print(f"Projection matrix: {proj_matrix_gpu.shape} "
      f"({proj_matrix_gpu.nbytes / 1e6:.1f} MB, on {device})")

# --- 6b. Compute projected CA gradients for ALL training samples ---------------
#
# This is the most expensive step.  For TRAIN_SIZE=10000 with BLIP-base CA-only
# gradients, each iteration is: forward + backward + project.
# With the CA-only filter, d_CA is small (~50% of d_all), so the projection
# matmul is fast.  The bottleneck is the per-sample forward/backward.
#
# We accumulate Phi^T Phi incrementally to avoid storing the full Phi matrix
# when TRAIN_SIZE is very large.  For moderate sizes (<=50k), we also store
# Phi for per-sample scoring.

model.train()

# Strategy: store full Phi in CPU memory (N x k float32 ~ N * k * 4 bytes)
# For N=10000, k=2048: ~80 MB — easily fits in RAM.
Phi = torch.zeros(TRAIN_SIZE, k)

# Also accumulate Phi^T Phi incrementally (k x k) for numerical stability
# with very large datasets.  For moderate N this is equivalent to Phi.T @ Phi.
PhiTPhi_acc = torch.zeros(k, k)

GRAD_BATCH_SIZE = BATCH_SIZE

print(f"\nComputing projected CA gradients for ALL {TRAIN_SIZE} training samples "
      f"(grad_batch={GRAD_BATCH_SIZE}) ...")
for batch_start in tqdm(range(0, TRAIN_SIZE, GRAD_BATCH_SIZE),
                        desc="Full-dataset train gradients (TRAK-CA)"):
    batch_end = min(batch_start + GRAD_BATCH_SIZE, TRAIN_SIZE)
    batch_indices = list(range(batch_start, batch_end))

    # Load and collate mini-batch onto GPU
    samples = [train_dataset[idx] for idx in batch_indices]
    keys_list = list(samples[0].keys())
    batch_gpu = {
        k_: torch.stack([s[k_] for s in samples]).to(
            device, dtype=torch.float16 if samples[0][k_].dtype == torch.float32 else None
        ) if samples[0][k_].dtype == torch.float32
        else torch.stack([s[k_] for s in samples]).to(device)
        for k_ in keys_list
    }

    # Per-sample forward + backward for individual CA gradients
    for j in range(len(batch_indices)):
        idx = batch_indices[j]
        single = {k_: v[j:j+1] for k_, v in batch_gpu.items()}
        outputs = model(**single, labels=single["input_ids"])
        phi_i = get_ca_lora_grad_projected(outputs.loss, proj_matrix_gpu)
        Phi[idx] = phi_i
        # Incremental rank-1 update: Phi^T Phi += phi_i phi_i^T
        PhiTPhi_acc += phi_i.unsqueeze(1) @ phi_i.unsqueeze(0)

    if (batch_end % 500 == 0) or batch_end == TRAIN_SIZE:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [{batch_end}/{TRAIN_SIZE}] completed")

print(f"Projected gradient matrix Phi: {Phi.shape} ({Phi.nbytes / 1e6:.1f} MB)")

# --- 6c. Compute the full-dataset TRAK kernel inverse -------------------------

print("\nComputing full-dataset TRAK-CA kernel inverse (Phi^T Phi + lambda I)^{-1} ...")
G_inv = torch.linalg.inv(PhiTPhi_acc + TRAK_LAMBDA * torch.eye(k))
cond = torch.linalg.cond(PhiTPhi_acc).item()
print(f"Kernel matrix: ({k}, {k}), condition number: {cond:.2e}")

# Pre-compute G^{-1} Phi^T for efficient scoring: (k, N)
G_inv_PhiT = G_inv @ Phi.T

del PhiTPhi_acc
gc.collect()

# --- 6d. Per-case: score ALL training samples ---------------------------------
#
# For each hallucination case:
#   scores = phi_test^T  G^{-1}  Phi^T    (k,) @ (k, N) = (N,)
# This gives a score for EVERY training sample — no pre-filtering.

train_influence_agg = defaultdict(float)
per_case_results = []

print(f"\nComputing TRAK-CA scores for {len(hallucination_cases)} hallucination cases "
      f"(scoring ALL {TRAIN_SIZE} training samples per case) ...")
for ci, case in enumerate(hallucination_cases):
    case_eval_idx = case["eval_index"]
    hall_tokens = case["hallucinated_objects"]

    # Test gradient (CA-only, projected on GPU)
    test_loss = hallucination_loss(case_eval_idx, hall_tokens)
    phi_test = get_ca_lora_grad_projected(test_loss, proj_matrix_gpu)

    # Score ALL training samples
    all_scores = phi_test @ G_inv_PhiT   # (TRAIN_SIZE,)

    # Build full ranking (descending by score)
    sorted_indices = torch.argsort(all_scores, descending=True)

    # Top-K for this case
    case_influences = []
    for rank_pos in range(min(TOP_K_PER_CASE, TRAIN_SIZE)):
        idx = sorted_indices[rank_pos].item()
        score = all_scores[idx].item()
        case_influences.append((idx, score))

    # Accumulate ALL positive-influence samples (not just top-K)
    positive_mask = all_scores > 0
    positive_indices = positive_mask.nonzero(as_tuple=True)[0]
    for pos in positive_indices:
        idx = pos.item()
        train_influence_agg[idx] += all_scores[idx].item()

    per_case_results.append({
        "case": case,
        "influences": case_influences,
        "n_positive": positive_indices.numel(),
    })

    if (ci + 1) % 100 == 0 or ci == 0:
        top1_idx, top1_score = case_influences[0]
        print(f"  [{ci+1}/{len(hallucination_cases)}] "
              f"eval #{case_eval_idx}  halluc={hall_tokens}  "
              f"positive={positive_indices.numel()}/{TRAIN_SIZE}  "
              f"top1: train[{top1_idx}]={top1_score:.4f} "
              f"({train_raw_labels[top1_idx]})")

# Free TRAK matrices
del proj_matrix_gpu, Phi, G_inv, G_inv_PhiT
gc.collect()
torch.cuda.empty_cache()

# ──────────────────────────────────────────
# 7.  Identify harmful training samples
# ──────────────────────────────────────────

harmful_sorted = sorted(train_influence_agg.items(), key=lambda x: -x[1])

# Only remove the top REMOVE_FRACTION of training samples by aggregated score,
# instead of ALL samples with any positive influence (which can remove everything).
max_remove = max(1, int(TRAIN_SIZE * REMOVE_FRACTION))
harmful_sorted_top = harmful_sorted[:max_remove]
harmful_set = set(idx for idx, _ in harmful_sorted_top)

print("\n" + "=" * 60)
print(f"HARMFUL TRAINING SAMPLES (top {REMOVE_FRACTION*100:.0f}%): "
      f"{len(harmful_set)} / {TRAIN_SIZE}  "
      f"(total positive-influence: {len(train_influence_agg)})")
print("=" * 60)

for rank, (idx, agg_score) in enumerate(harmful_sorted[:20], 1):
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
    f.write("Hallucination TDA (TRAK-CA v3 Full-Dataset) — Summary\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Method: TRAK-CA v3 (Cross-Attention gradients, FULL dataset)\n")
    f.write(f"  NO label-based pre-filtering — all {TRAIN_SIZE} samples scored\n")
    f.write(f"  Gradient scope: crossattention.self.{{query,value}} LoRA\n")
    f.write(f"  CA params: {ca_param_count:,} / {all_param_count:,} total "
            f"({ca_param_count / all_param_count * 100:.1f}%)\n\n")

    f.write(f"Train size: {TRAIN_SIZE}   Eval size: {EVAL_SIZE}\n")
    f.write(f"TRAK projection dim: {k}   lambda: {TRAK_LAMBDA}\n")
    f.write(f"Kernel condition number: {cond:.2e}\n")
    f.write(f"Hallucination cases (before cleaning): "
            f"{len(hallucination_cases)} / {EVAL_SIZE}\n")
    f.write(f"Harmful training samples to remove: {len(harmful_set)}\n\n")

    for ci, res in enumerate(per_case_results, 1):
        c = res["case"]
        f.write(f"--- Case {ci}: eval_idx={c['eval_index']} ---\n")
        f.write(f"  Generated  : {c['generated']}\n")
        f.write(f"  GT label   : {c['gt_label']}\n")
        f.write(f"  Hallucinated: {', '.join(c['hallucinated_objects'])}\n")
        f.write(f"  Positive-influence samples: {res['n_positive']} / {TRAIN_SIZE}\n")
        top5 = res["influences"][:5]
        for r, (idx, sc) in enumerate(top5, 1):
            f.write(f"    top-{r} train_idx={idx}  influence={sc:.4f}  "
                    f"objects: {train_raw_labels[idx]}\n")
        f.write("\n")

    f.write("--- Aggregated Harmful Training Samples (top-20) ---\n")
    for rank, (idx, agg_score) in enumerate(harmful_sorted[:20], 1):
        f.write(f"  #{rank}  train_idx={idx}  agg_influence={agg_score:.4f}  "
                f"objects: {train_raw_labels[idx]}\n")

print(f"First-round summary written to {summary_path}")

# ──────────────────────────────────────────
# 9.  Remove harmful samples & retrain
# ──────────────────────────────────────────

clean_indices = [i for i in range(TRAIN_SIZE) if i not in harmful_set]
print(f"\nCleaned training set: {len(clean_indices)} / {TRAIN_SIZE} "
      f"(removed {len(harmful_set)})")

clean_train_raw = train_raw_dataset.select(clean_indices)
clean_raw_labels = [train_raw_labels[i] for i in clean_indices]
clean_train_ds = LazyCocoDataset(clean_train_raw, clean_raw_labels)

clean_loader = DataLoader(clean_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=0)

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
print("BEFORE vs AFTER DATA CLEANING  (TRAK-CA v3 Full-Dataset)")
print(f"{'=' * 60}")
print(f"  Attribution method  : TRAK-CA v3 (CA-only, full-dataset, no pre-filter)")
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

with open(summary_path, "a", encoding="utf-8") as f:
    f.write("\n\n" + "=" * 60 + "\n")
    f.write("After Data Cleaning — Re-evaluation  (TRAK-CA v3)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Attribution method  : TRAK-CA v3 (full-dataset, no pre-filter)\n")
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
