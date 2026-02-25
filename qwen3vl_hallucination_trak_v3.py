"""
TRAK-CA v3: Full-Dataset Visual Merger TRAK for Qwen3-VL Hallucination Attribution
===================================================================================

Improvement over v2:
  v2 used a label-based "suspect pre-filter" (MAX_FILTERED=500) that only
  computed gradients for training samples whose labels contain the
  hallucinated object name.  This is fundamentally flawed:

    - It turns "attribution" into "verification" -- asking "is it one of these
      500 suspects?" instead of "who did it?"
    - It misses the real culprits: a "mop" sample that looks like a "dog",
      a noisy background that triggers a false detection, or co-occurrence
      patterns from semantically unrelated images.
    - The kernel matrix G = Phi^T Phi only reflects the filtered subset,
      not the true training distribution, distorting the ridge regression.

  v3 fixes this by computing TRAK-CA over the **entire training set**:

    1. Compute phi_i = P^T nabla_{theta_merger} L(z_i)  for ALL N training samples
    2. Build Phi in R^{N x k}  from the full training set
    3. G^{-1} = (Phi^T Phi + lambda I)^{-1}  reflects the true covariance
    4. For each hallucination case, score ALL N training samples:
         score_i = phi_test^T G^{-1} phi_i

  This is the theoretically correct TRAK formulation.  The computational
  cost is O(N) gradient computations (one-time), which is feasible because:
    - Merger-only gradients have small dimension d_merger
    - Random projection reduces storage to k floats per sample
    - The kernel inverse is only k x k (fast)
    - Scoring is a single matrix-vector product

Pipeline:
  1. LoRA fine-tune Qwen3-VL on MS COCO Detection + unfreeze visual merger
  2. Generate descriptions and detect hallucinated objects
  3. Compute full-dataset TRAK-CA influence scores (merger gradients, all samples)
  4. Remove harmful samples, retrain, re-evaluate

Designed for A100 (BF16, gradient checkpointing, SDPA).
Set DEBUG_TRAIN_SIZE / DEBUG_EVAL_SIZE env vars for small-subset runs.
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from collections import defaultdict
import random
import json
import gc
import os
import re

# ──────────────────────────────────────────
# 0.  Environment setup
# ──────────────────────────────────────────

OUTPUT_DIR = "results_qwen3vl_trak_ca_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True

# ──────────────────────────────────────────
# 1.  Load Qwen3-VL-2B + LoRA + Unfreeze Merger
# ──────────────────────────────────────────

model_name = "Qwen/Qwen3-VL-2B-Instruct"

processor = Qwen3VLProcessor.from_pretrained(model_name)
processor.tokenizer.padding_side = "left"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to(device)

for param in model.parameters():
    param.requires_grad = False

# LoRA on LM decoder self-attention (same as original TracIn version)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Additionally unfreeze the Visual Merger (Vision-to-Language Adapter).
# The merger is small (~few Linear + LayerNorm layers) and is the bridge
# between the vision encoder and the LM decoder.  We need its gradients
# for TRAK-CA attribution.
merger_unfrozen = 0
for name, param in model.named_parameters():
    if "merger" in name:
        param.requires_grad = True
        merger_unfrozen += 1

model.gradient_checkpointing_enable()

# ──────────────────────────────────────────
# 1b. Identify Visual Merger parameters
# ──────────────────────────────────────────

def is_merger_param(name: str) -> bool:
    """Check if a parameter belongs to the visual merger (Vision-to-Language adapter)."""
    return "merger" in name


# Enumerate and report
merger_param_names = [n for n, p in model.named_parameters()
                      if p.requires_grad and is_merger_param(n)]
non_merger_param_names = [n for n, p in model.named_parameters()
                          if p.requires_grad and not is_merger_param(n)]

merger_param_count = sum(p.numel() for n, p in model.named_parameters()
                         if p.requires_grad and is_merger_param(n))
all_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n--- TRAK-CA Parameter Filter (Visual Merger) ---")
print(f"  Total trainable params        : {all_param_count:,}")
print(f"  Visual merger params (for CA)  : {merger_param_count:,}  "
      f"({merger_param_count / all_param_count * 100:.1f}%)")
print(f"  LM decoder LoRA params         : {all_param_count - merger_param_count:,}  "
      f"(excluded from TRAK)")
print(f"  Merger modules ({len(merger_param_names)}):")
for n in merger_param_names:
    print(f"    {n}")

if merger_param_count == 0:
    print("\n  WARNING: No merger parameters found! Check model architecture.")
    print("  Listing all trainable params for debugging:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"    {n}  ({p.numel():,})")

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
# 2b. Extract labels
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

# ──────────────────────────────────────────
# 2c. Chat-format helpers for Qwen3-VL
# ──────────────────────────────────────────

TRAIN_SYSTEM_PROMPT = "You are an object detection assistant. List the object categories visible in the image."
TRAIN_USER_PROMPT = "List all object categories in this image, separated by semicolons."

IM_START_ID = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")


def build_train_messages(image, label_text):
    return [
        {"role": "system", "content": TRAIN_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": TRAIN_USER_PROMPT},
        ]},
        {"role": "assistant", "content": label_text},
    ]


def build_inference_messages(image):
    return [
        {"role": "system", "content": TRAIN_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": TRAIN_USER_PROMPT},
        ]},
    ]


def mask_labels(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Create labels from input_ids: mask everything before assistant content with -100."""
    labels = input_ids.clone()
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)

    for i in range(labels.shape[0]):
        ids = labels[i]
        ids[ids == pad_token_id] = -100
        positions = (input_ids[i] == IM_START_ID).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            assistant_start = positions[-1].item()
            mask_end = min(assistant_start + 2, ids.shape[0])
            ids[:mask_end] = -100

    return labels.squeeze(0) if input_ids.dim() == 1 else labels


def prepare_single(image, label_text):
    """Prepare a single sample and move to device."""
    messages = build_train_messages(image, label_text)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt",
    )
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    inputs["labels"] = mask_labels(inputs["input_ids"], pad_id)
    return {k: v.to(device) for k, v in inputs.items()}


def prepare_batch(images, label_texts):
    """Prepare a batch of samples using the processor's native batching."""
    all_texts = []
    all_image_inputs = []
    for img, lab in zip(images, label_texts):
        msgs = build_train_messages(img, lab)
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        all_texts.append(text)
        img_inp, _ = process_vision_info(msgs)
        all_image_inputs.extend(img_inp)

    inputs = processor(
        text=all_texts,
        images=all_image_inputs,
        padding=True,
        return_tensors="pt",
    )
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    inputs["labels"] = mask_labels(inputs["input_ids"], pad_id)
    return {k: v.to(device) for k, v in inputs.items()}


# ──────────────────────────────────────────
# 3.  LoRA fine-tuning (ALL trainable params: LoRA + merger)
# ──────────────────────────────────────────

NUM_EPOCHS = 2
BATCH_SIZE = 16

model.train()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-4,
)

train_order = list(range(TRAIN_SIZE))


def run_training(dataset, raw_labels, n_samples, desc_prefix="Epoch"):
    """Generic training loop for a dataset."""
    order = list(range(n_samples))
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        print(f"\n=== {desc_prefix} {epoch + 1}/{NUM_EPOCHS} ===")
        random.shuffle(order)

        for start in tqdm(range(0, n_samples, BATCH_SIZE),
                          desc=f"{desc_prefix} {epoch + 1}"):
            batch_indices = order[start:start + BATCH_SIZE]
            images = []
            labels = []
            for idx in batch_indices:
                ex = dataset[idx]
                images.append(ex["image"].convert("RGB"))
                labels.append(raw_labels[idx])

            inputs = prepare_batch(images, labels)
            outputs = model(**inputs)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / n_batches
        print(f"  avg loss = {avg:.4f}")
        torch.cuda.empty_cache()


print(f"\nTraining with batch_size={BATCH_SIZE}")
run_training(train_raw_dataset, train_raw_labels, TRAIN_SIZE, desc_prefix="Epoch")

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


@torch.no_grad()
def generate_for_image(image):
    """Generate text for a single image using the chat template."""
    messages = build_inference_messages(image)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt",
    ).to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    input_len = inputs["input_ids"].shape[1]
    output_ids = generated_ids[0, input_len:]
    return processor.tokenizer.decode(output_ids, skip_special_tokens=True)


model.eval()
hallucination_cases = []

print(f"\nScanning {EVAL_SIZE} eval samples for hallucinations ...")
for i in tqdm(range(EVAL_SIZE), desc="Hallucination detection"):
    example = eval_raw_dataset[i]
    image = example["image"].convert("RGB")
    generated_text = generate_for_image(image)

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

print(f"Hallucination cases found: {len(hallucination_cases)} / {EVAL_SIZE}")

if not hallucination_cases:
    print("No hallucinations detected. Try more samples or fewer epochs.")
    exit(0)

# Save hallucination cases to JSON
halluc_json_path = os.path.join(OUTPUT_DIR, "hallucination_cases.json")
_serializable = [{**c, "hallucinated_objects": list(c["hallucinated_objects"])} for c in hallucination_cases]
with open(halluc_json_path, "w") as f:
    json.dump(_serializable, f)
print(f"Hallucination cases saved to {halluc_json_path}")

# Save hallucination eval sample images (non-fatal)
print(f"\nSaving hallucination eval images to {OUTPUT_DIR}/ ...")
_saved = 0
for case_item in hallucination_cases:
    eidx = case_item["eval_index"]
    try:
        eval_raw_dataset[eidx]["image"].save(
            os.path.join(OUTPUT_DIR, f"halluc_eval_{eidx}.png"))
        _saved += 1
    except OSError:
        if _saved == 0:
            print("  WARNING: Disk quota hit, skipping image saves.")
        break
print(f"  Saved {_saved}/{len(hallucination_cases)} images.")

# ──────────────────────────────────────────
# 5.  Helper functions for TRAK-CA
# ──────────────────────────────────────────

# TRAK hyperparameters
TRAK_PROJ_DIM = 2048        # projection dimension k
TRAK_LAMBDA = 1e-3          # ridge regularization parameter
TOP_K_PER_CASE = 50         # report top-k per case (full ranking is computed)
REMOVE_FRACTION = 0.1       # remove top 10% most harmful training samples


def hallucination_loss(eval_index: int, hallucinated_tokens: set):
    """Compute loss focused on hallucinated token positions."""
    example = eval_raw_dataset[eval_index]
    image = example["image"].convert("RGB")

    model.eval()
    generated_text = generate_for_image(image)
    model.train()

    # Forward pass with generated text as target
    messages = build_train_messages(image, generated_text)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]

    outputs = model(**inputs, labels=input_ids)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    token_ids = input_ids[0].tolist()
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


def get_merger_grad_projected(loss, proj_matrix_gpu):
    """Backprop, collect ONLY visual merger gradients, project on GPU.

    This is the core of TRAK-CA for decoder-only VLMs: we discard all LM
    decoder LoRA gradients (which carry language-bias signal) and keep only
    visual merger gradients (which carry visual-grounding signal).
    """
    model.zero_grad()
    loss.backward(retain_graph=False)

    grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if is_merger_param(name):
                grads.append(param.grad.detach().flatten())  # stay on GPU

    if not grads:
        return torch.zeros(proj_matrix_gpu.shape[1])

    g_merger = torch.cat(grads).float()     # (d_merger,) on GPU, float32
    phi = proj_matrix_gpu.T @ g_merger      # (k,) on GPU -- fast matmul
    return phi.cpu()                        # only k-dim transferred to CPU


# ──────────────────────────────────────────
# 6.  Full-dataset TRAK-CA
# ──────────────────────────────────────────
#
# v3: NO label-based pre-filtering.  We compute projected gradients for
# every single training sample, build the full kernel, and score against
# the entire training set for each hallucination case.
#
#   phi_i  = P^T nabla_{theta_merger} L(z_i)   for i = 1, ..., N  (ALL samples)
#   Phi    = [phi_1, ..., phi_N]^T              in R^{N x k}
#   G^{-1} = (Phi^T Phi + lambda I)^{-1}       in R^{k x k}
#   scores = phi_test^T  G^{-1}  Phi^T          in R^{N}
#
# The kernel G now reflects the TRUE training distribution, not a biased
# subset.  Attribution can discover unexpected culprits (e.g., a "mop"
# sample causing "dog" hallucinations via texture similarity).

# --- 6a. Initialize random projection matrix (d_merger x k) ---

d_merger = merger_param_count
k = min(TRAK_PROJ_DIM, d_merger)
print(f"\nTRAK-CA v3 (full-dataset): d_merger = {d_merger:,}, projection dim k = {k}")
print(f"Will compute gradients for ALL {TRAIN_SIZE} training samples.")

torch.manual_seed(42)
proj_matrix_gpu = (torch.randn(d_merger, k) / (k ** 0.5)).to(device)
print(f"Projection matrix: {proj_matrix_gpu.shape} "
      f"({proj_matrix_gpu.nbytes / 1e6:.1f} MB, on {device})")

# --- 6b. Compute projected merger gradients for ALL training samples ---
#
# This is the most expensive step.  For each training sample:
# forward + backward + project.  The merger-only filter keeps d_merger
# small, so the projection matmul is fast.  The bottleneck is the
# per-sample forward/backward.
#
# We accumulate Phi^T Phi incrementally to avoid storing the full Phi matrix
# when TRAIN_SIZE is very large.  For moderate sizes (<=50k), we also store
# Phi for per-sample scoring.

model.train()

# Strategy: store full Phi in CPU memory (N x k float32 ~ N * k * 4 bytes)
Phi = torch.zeros(TRAIN_SIZE, k)

# Also accumulate Phi^T Phi incrementally (k x k) for numerical stability
PhiTPhi_acc = torch.zeros(k, k)

print(f"\nComputing projected merger gradients for ALL {TRAIN_SIZE} training samples ...")
for idx in tqdm(range(TRAIN_SIZE), desc="Full-dataset train gradients (TRAK-CA)"):
    example = train_raw_dataset[idx]
    image = example["image"].convert("RGB")
    label = train_raw_labels[idx]
    inputs = prepare_single(image, label)
    outputs = model(**inputs)
    phi_i = get_merger_grad_projected(outputs.loss, proj_matrix_gpu)
    Phi[idx] = phi_i
    # Incremental rank-1 update: Phi^T Phi += phi_i phi_i^T
    PhiTPhi_acc += phi_i.unsqueeze(1) @ phi_i.unsqueeze(0)

    if (idx + 1) % 500 == 0 or idx + 1 == TRAIN_SIZE:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [{idx + 1}/{TRAIN_SIZE}] completed")

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
# This gives a score for EVERY training sample -- no pre-filtering.

train_influence_agg = defaultdict(float)
per_case_results = []

print(f"\nComputing TRAK-CA scores for {len(hallucination_cases)} hallucination cases "
      f"(scoring ALL {TRAIN_SIZE} training samples per case) ...")
for ci, case in enumerate(hallucination_cases):
    case_eval_idx = case["eval_index"]
    hall_tokens = case["hallucinated_objects"]

    # Test gradient (merger-only, projected on GPU)
    test_loss = hallucination_loss(case_eval_idx, hall_tokens)
    phi_test = get_merger_grad_projected(test_loss, proj_matrix_gpu)

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

print(f"\nSaving top-10 harmful training sample images to {OUTPUT_DIR}/ ...")
for rank, (idx, agg_score) in enumerate(harmful_sorted[:10], 1):
    try:
        train_raw_dataset[idx]["image"].save(
            os.path.join(OUTPUT_DIR, f"top{rank}_train_{idx}.png"))
    except OSError:
        print(f"  WARNING: Could not save image (disk quota?), skipping remaining.")
        break

# ──────────────────────────────────────────
# 8.  Write first-round summary
# ──────────────────────────────────────────

summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("Qwen3-VL Hallucination TDA (TRAK-CA v3 Full-Dataset) — Summary\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Model: {model_name}\n")
    f.write(f"Method: TRAK-CA v3 (Visual Merger gradients, FULL dataset)\n")
    f.write(f"  NO label-based pre-filtering — all {TRAIN_SIZE} samples scored\n")
    f.write(f"  Gradient scope: visual.merger (PatchMerger)\n")
    f.write(f"  Merger params: {merger_param_count:,} / {all_param_count:,} total "
            f"({merger_param_count / all_param_count * 100:.1f}%)\n\n")

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

# Reload fresh model + LoRA + unfreeze merger
del model
gc.collect()
torch.cuda.empty_cache()

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to(device)
for param in model.parameters():
    param.requires_grad = False
model = get_peft_model(model, lora_config)
# Re-unfreeze merger
for name, param in model.named_parameters():
    if "merger" in name:
        param.requires_grad = True
model.gradient_checkpointing_enable()

model.train()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-4,
)

CLEAN_SIZE = len(clean_indices)

print(f"\n{'=' * 60}")
print("RETRAINING on cleaned dataset ...")
print(f"{'=' * 60}")

run_training(clean_train_raw, clean_raw_labels, CLEAN_SIZE, desc_prefix="Retrain epoch")

print("\nRetraining done.")
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_after_cleaning"))
print(f"LoRA checkpoint saved to {OUTPUT_DIR}/lora_after_cleaning")

# ──────────────────────────────────────────
# 10. Re-evaluate hallucinations after cleaning
# ──────────────────────────────────────────

model.eval()
hallucination_cases_after = []

print(f"\nRe-scanning {EVAL_SIZE} eval samples for hallucinations ...")
for i in tqdm(range(EVAL_SIZE), desc="Re-eval hallucination detection"):
    example = eval_raw_dataset[i]
    image = example["image"].convert("RGB")
    generated_text = generate_for_image(image)

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
print(f"  Attribution method  : TRAK-CA v3 (visual merger, full-dataset, no pre-filter)")
print(f"  Merger params used  : {merger_param_count:,} / {all_param_count:,}")
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
    f.write(f"Attribution method  : TRAK-CA v3 (visual merger, full-dataset, no pre-filter)\n")
    f.write(f"Merger params used  : {merger_param_count:,} / {all_param_count:,}\n")
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
