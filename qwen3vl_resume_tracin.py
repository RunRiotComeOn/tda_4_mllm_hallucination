"""
Resume from saved LoRA checkpoint: re-detect hallucinations, then TracIn + retrain.
Skips the initial LoRA fine-tuning (already done).
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from peft import PeftModel, LoraConfig, get_peft_model
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

OUTPUT_DIR = "results_qwen3vl"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True

# ──────────────────────────────────────────
# 1.  Load Qwen3-VL-2B + saved LoRA
# ──────────────────────────────────────────

model_name = "Qwen/Qwen3-VL-2B-Instruct"
LORA_PATH = os.path.join(OUTPUT_DIR, "lora_before_cleaning")

processor = Qwen3VLProcessor.from_pretrained(model_name)
processor.tokenizer.padding_side = "left"

print(f"Loading base model + LoRA from {LORA_PATH} ...")
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to(device)

model = PeftModel.from_pretrained(base_model, LORA_PATH, is_trainable=True)
print("LoRA checkpoint loaded.")
model.print_trainable_parameters()

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
)

# ──────────────────────────────────────────
# 2.  Load MS COCO Detection
# ──────────────────────────────────────────

DEBUG_TRAIN_SIZE = int(os.environ.get("DEBUG_TRAIN_SIZE", 0))
DEBUG_EVAL_SIZE = int(os.environ.get("DEBUG_EVAL_SIZE", 0))

_full_train = load_dataset("detection-datasets/coco", split="train")
_full_eval = load_dataset("detection-datasets/coco", split="val")
category_names = _full_train.features["objects"]["category"].feature.names

train_raw_dataset = _full_train.select(range(DEBUG_TRAIN_SIZE)) if DEBUG_TRAIN_SIZE else _full_train
eval_raw_dataset = _full_eval.select(range(DEBUG_EVAL_SIZE)) if DEBUG_EVAL_SIZE else _full_eval

TRAIN_SIZE = len(train_raw_dataset)
EVAL_SIZE = len(eval_raw_dataset)

print(f"Train set: {TRAIN_SIZE} samples")
print(f"Eval set:  {EVAL_SIZE} samples")

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
# 2c. Chat-format helpers
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


def mask_labels(input_ids, pad_token_id):
    labels = input_ids.clone()
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    for i in range(labels.shape[0]):
        ids = labels[i]
        ids[ids == pad_token_id] = -100
        positions = (input_ids[i] == IM_START_ID).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            mask_end = min(positions[-1].item() + 2, ids.shape[0])
            ids[:mask_end] = -100
    return labels.squeeze(0) if input_ids.dim() == 1 else labels


def prepare_single(image, label_text):
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
    all_texts = []
    all_image_inputs = []
    for img, lab in zip(images, label_texts):
        msgs = build_train_messages(img, lab)
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        all_texts.append(text)
        img_inp, _ = process_vision_info(msgs)
        all_image_inputs.extend(img_inp)
    inputs = processor(
        text=all_texts, images=all_image_inputs, padding=True, return_tensors="pt",
    )
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    inputs["labels"] = mask_labels(inputs["input_ids"], pad_id)
    return {k: v.to(device) for k, v in inputs.items()}


# ──────────────────────────────────────────
# 3. Detect hallucinations (or load from cache)
# ──────────────────────────────────────────

def parse_gt_objects(text):
    return {obj.strip().lower() for obj in text.split(";") if obj.strip()}

_category_names_lower = [name.lower() for name in category_names]

def extract_generated_objects(text):
    text_lower = text.lower()
    found = set()
    for name_lower in _category_names_lower:
        if re.search(r'\b' + re.escape(name_lower) + r'\b', text_lower):
            found.add(name_lower)
    return found

@torch.no_grad()
def generate_for_image(image):
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


halluc_json_path = os.path.join(OUTPUT_DIR, "hallucination_cases.json")

if os.path.exists(halluc_json_path):
    print(f"\nLoading cached hallucination cases from {halluc_json_path} ...")
    with open(halluc_json_path) as f:
        _loaded = json.load(f)
    hallucination_cases = [{**c, "hallucinated_objects": set(c["hallucinated_objects"])} for c in _loaded]
    print(f"Loaded {len(hallucination_cases)} hallucination cases from cache.")
else:
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

    # Save to JSON
    _serializable = [{**c, "hallucinated_objects": list(c["hallucinated_objects"])} for c in hallucination_cases]
    with open(halluc_json_path, "w") as f:
        json.dump(_serializable, f)
    print(f"Saved to {halluc_json_path}")

if not hallucination_cases:
    print("No hallucinations detected.")
    exit(0)

# ──────────────────────────────────────────
# 4.  Helper functions for TracIn
# ──────────────────────────────────────────

def hallucination_loss(eval_index, hallucinated_tokens):
    example = eval_raw_dataset[eval_index]
    image = example["image"].convert("RGB")

    model.eval()
    generated_text = generate_for_image(image)
    model.train()

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


def get_lora_grad(loss):
    model.zero_grad()
    loss.backward(retain_graph=False)
    grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grads.append(param.grad.detach().cpu().float().flatten())
    torch.cuda.empty_cache()
    if not grads:
        return torch.zeros(1)
    return torch.cat(grads)


# ──────────────────────────────────────────
# 5.  TracIn
# ──────────────────────────────────────────

MAX_FILTERED = 500

train_labels_lower = [l.lower() for l in train_raw_labels]
object_to_train_indices = {}

all_halluc_objects = set()
for case in hallucination_cases:
    all_halluc_objects |= case["hallucinated_objects"]

print(f"\nUnique hallucinated objects across all cases: {len(all_halluc_objects)}")

for obj in sorted(all_halluc_objects):
    indices = [i for i, lab in enumerate(train_labels_lower) if obj in lab]
    object_to_train_indices[obj] = indices[:MAX_FILTERED]

all_train_indices = set()
for indices in object_to_train_indices.values():
    all_train_indices.update(indices)
all_train_indices = sorted(all_train_indices)
print(f"Unique training samples to compute gradients for: {len(all_train_indices)}")

if not all_train_indices:
    print("WARNING: No matching training samples. Skipping TracIn.")
    harmful_set = set()
    harmful_sorted = []
    per_case_results = []
    train_influence_agg = defaultdict(float)
else:
    model.train()
    train_grad_cache = {}

    print("\nPre-computing training gradients (one-time) ...")
    for idx in tqdm(all_train_indices, desc="Train gradients"):
        example = train_raw_dataset[idx]
        image = example["image"].convert("RGB")
        label = train_raw_labels[idx]
        inputs = prepare_single(image, label)
        outputs = model(**inputs)
        g_train = get_lora_grad(outputs.loss)
        train_grad_cache[idx] = g_train.half()

        if len(train_grad_cache) % 200 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Cached {len(train_grad_cache)} training gradients "
          f"(~{len(train_grad_cache) * train_grad_cache[all_train_indices[0]].nbytes / 1e9:.1f} GB)")

    train_influence_agg = defaultdict(float)
    per_case_results = []

    print(f"\nComputing TracIn for {len(hallucination_cases)} hallucination cases ...")
    for ci, case in enumerate(hallucination_cases):
        case_eval_idx = case["eval_index"]
        hall_tokens = case["hallucinated_objects"]

        test_loss = hallucination_loss(case_eval_idx, hall_tokens)
        g_test = get_lora_grad(test_loss)

        filtered_indices = set()
        for obj in hall_tokens:
            filtered_indices.update(object_to_train_indices.get(obj, []))
        filtered_indices = sorted(filtered_indices)

        case_influences = []
        for idx in filtered_indices:
            influence = torch.dot(g_test, train_grad_cache[idx].float()).item()
            case_influences.append((idx, influence))

        case_influences.sort(key=lambda x: -x[1])

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

    del train_grad_cache
    gc.collect()
    torch.cuda.empty_cache()

# ──────────────────────────────────────────
# 6.  Identify harmful training samples
# ──────────────────────────────────────────

harmful_set = set(train_influence_agg.keys())
harmful_sorted = sorted(train_influence_agg.items(), key=lambda x: -x[1])

print("\n" + "=" * 60)
print(f"HARMFUL TRAINING SAMPLES (positive influence): {len(harmful_set)} / {TRAIN_SIZE}")
print("=" * 60)

for rank, (idx, agg_score) in enumerate(harmful_sorted[:15], 1):
    print(f"  #{rank}  train_idx={idx}  agg_influence={agg_score:.4f}  "
          f"objects: {train_raw_labels[idx]}")

for rank, (idx, agg_score) in enumerate(harmful_sorted[:10], 1):
    try:
        train_raw_dataset[idx]["image"].save(
            os.path.join(OUTPUT_DIR, f"top{rank}_train_{idx}.png"))
    except OSError:
        break

# ──────────────────────────────────────────
# 7.  Write first-round summary
# ──────────────────────────────────────────

summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("Qwen3-VL Hallucination TDA — First Round Summary\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"Train size: {TRAIN_SIZE}   Eval size: {EVAL_SIZE}\n")
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
# 8.  Remove harmful samples & retrain (batched)
# ──────────────────────────────────────────

NUM_EPOCHS = 2
BATCH_SIZE = 16

clean_indices = [i for i in range(TRAIN_SIZE) if i not in harmful_set]
print(f"\nCleaned training set: {len(clean_indices)} / {TRAIN_SIZE} "
      f"(removed {len(harmful_set)})")

clean_train_raw = train_raw_dataset.select(clean_indices)
clean_raw_labels = [train_raw_labels[i] for i in clean_indices]

del model, base_model
gc.collect()
torch.cuda.empty_cache()

base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to(device)
for param in base_model.parameters():
    param.requires_grad = False
model = get_peft_model(base_model, lora_config)
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

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    n_batches = 0
    order = list(range(CLEAN_SIZE))
    random.shuffle(order)
    print(f"\n=== Retrain epoch {epoch + 1}/{NUM_EPOCHS} ===")

    for start in tqdm(range(0, CLEAN_SIZE, BATCH_SIZE), desc=f"Retrain epoch {epoch + 1}"):
        batch_indices = order[start:start + BATCH_SIZE]
        images = []
        labels = []
        for idx in batch_indices:
            ex = clean_train_raw[idx]
            images.append(ex["image"].convert("RGB"))
            labels.append(clean_raw_labels[idx])

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

print("\nRetraining done.")
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_after_cleaning"))

# ──────────────────────────────────────────
# 9. Re-evaluate hallucinations after cleaning
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
# 10. Before vs After comparison
# ──────────────────────────────────────────

n_before = len(hallucination_cases)
n_after = len(hallucination_cases_after)
rate_before = n_before / EVAL_SIZE * 100
rate_after = n_after / EVAL_SIZE * 100

print(f"\n{'=' * 60}")
print("BEFORE vs AFTER DATA CLEANING")
print(f"{'=' * 60}")
print(f"  Training samples    : {TRAIN_SIZE} -> {len(clean_indices)} "
      f"(removed {len(harmful_set)})")
print(f"  Hallucination cases : {n_before} -> {n_after}")
print(f"  Hallucination rate  : {rate_before:.1f}% -> {rate_after:.1f}%")
if n_before > 0:
    reduction = (n_before - n_after) / n_before * 100
    print(f"  Reduction           : {reduction:.1f}%")
print(f"{'=' * 60}")

with open(summary_path, "a", encoding="utf-8") as f:
    f.write("\n\n" + "=" * 60 + "\n")
    f.write("After Data Cleaning — Re-evaluation\n")
    f.write("=" * 60 + "\n\n")
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
