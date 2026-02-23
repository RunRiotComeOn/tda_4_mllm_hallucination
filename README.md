# Training Data Attribution for Multimodal LLM Hallucination

**Tracing Object Hallucinations in Vision-Language Models Back to Training Data via Gradient-Based Influence Functions**

## Abstract

Multimodal Large Language Models (MLLMs) frequently *hallucinate* — generating
text that describes objects not present in the input image. While prior work has
focused on detecting or mitigating hallucinations at inference time, the
fundamental question remains: **which training examples cause a model to
hallucinate a specific object?**

This project applies **Training Data Attribution (TDA)** — specifically
**TracIn** (Pruthi et al., 2020) — to trace object hallucinations in a
LoRA-fine-tuned BLIP captioning model back to individual training samples.
By computing the gradient inner product between a hallucination-targeted test
loss and each training sample's loss, we identify the training images whose
captions most strongly *promote* (or *suppress*) a given hallucination.

## Motivation

| Problem | Why It Matters |
|---|---|
| MLLMs hallucinate objects not in the image | Undermines trustworthiness in medical, autonomous driving, and accessibility applications |
| Existing fixes are inference-time patches | They treat symptoms, not root causes |
| Training data is the root cause | Noisy, mislabeled, or distributional biases in captions teach models to associate wrong objects |
| **TDA reveals the "why"** | Identifies which training samples are responsible, enabling targeted data curation |

## Method

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: LoRA Fine-tuning                                      │
│  BLIP-base + LoRA(r=8) on COCO captions (3k subset, FP16)     │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: Hallucination Detection                               │
│  Generate captions → spaCy noun extraction → set difference     │
│  hallucinated_nouns = gen_nouns − gt_nouns                      │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3: Hallucination-Targeted Loss                           │
│  L_hall = Σ log p(t_h | x) for each hallucinated token t_h     │
├─────────────────────────────────────────────────────────────────┤
│  Stage 4: TracIn Influence                                      │
│  I(z_train, z_test) = ∇_θ L_hall(z_test) · ∇_θ L(z_train)    │
│  Computed over LoRA parameters only (low-dimensional)           │
├─────────────────────────────────────────────────────────────────┤
│  Stage 5: Ranking & Analysis                                    │
│  Sort training samples by influence score                       │
│  Positive = promotes hallucination, Negative = suppresses       │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Choices

- **LoRA-only gradients**: Computing TracIn over full model parameters is
  intractable. LoRA reduces the parameter space from ~224M to ~0.3M, making
  gradient inner products feasible on consumer GPUs.
- **Hallucination-targeted loss**: Instead of using the standard captioning loss
  on the test sample, we define a loss that specifically measures the model's
  confidence on hallucinated tokens, yielding a gradient that points in the
  "hallucination direction."
- **Noun-level detection**: We use spaCy's POS tagger to extract nouns from
  both generated and ground-truth captions. The set difference identifies
  *object hallucinations* — the most semantically impactful type.
- **Pre-filtering**: To avoid computing gradients for all N training samples,
  we pre-filter to those whose captions contain the hallucinated noun, reducing
  computation by ~10×.

### Formal Definition

Given a test sample $z_{\text{test}}$ with hallucinated tokens $\mathcal{H}$,
we define:

$$L_{\text{hall}}(z_{\text{test}}) = \sum_{t \in \mathcal{H}} \log p_\theta(t \mid x_{\text{test}})$$

The TracIn influence of training sample $z_i$ is:

$$\mathcal{I}(z_i, z_{\text{test}}) = \nabla_\theta L_{\text{hall}}(z_{\text{test}}) \cdot \nabla_\theta L(z_i)$$

where $\theta$ denotes LoRA parameters only.

## Repository Structure

```
tda_4_mllm/
├── README.md                          # This file
├── blip_hallucination_tracin.py       # Main pipeline script
└── requirements.txt                   # Python dependencies
```

## Requirements

### Hardware

- GPU with >= 8 GB VRAM (tested on RTX 3060/4060/3070)
- ~16 GB system RAM

### Software

- Python >= 3.9
- CUDA >= 11.7

### Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

```bash
python blip_hallucination_tracin.py
```

The script will:

1. Download BLIP-base (~990 MB) and COCO captions (~3k subset)
2. LoRA fine-tune for 2 epochs (~5 min on RTX 3060)
3. Scan 500 samples for hallucinations
4. Compute TracIn influence for filtered training samples
5. Print top-10 most influential (hallucination-promoting) and bottom-5
   (hallucination-suppressing) training samples

### Memory Budget (8 GB VRAM)

| Component | Memory |
|---|---|
| BLIP-base FP16 | ~1.8 GB |
| LoRA parameters | ~2.4 MB |
| Batch (size=2, 384×384) | ~0.3 GB |
| Gradient checkpointing overhead | ~0.5 GB |
| Peak during backward | ~3.5 GB |
| **Total peak** | **~6.1 GB** |

## Methodology Details

### Why TracIn over Other TDA Methods?

| Method | Pros | Cons |
|---|---|---|
| **TracIn** (ours) | Simple, first-order, checkpoint-compatible | Approximation; requires training checkpoints |
| Influence Functions | Principled (second-order) | Hessian inversion intractable for large models |
| Datamodels | Model-agnostic | Requires thousands of retraining runs |
| TRAK | Efficient random projection | Less interpretable; projection variance |

TracIn is chosen for its simplicity and direct compatibility with LoRA: since
LoRA parameters are low-dimensional, the gradient inner product is both cheap
and relatively accurate.

### Hallucination Taxonomy

This project targets **object hallucinations** specifically:

- **Object hallucination**: Model mentions an object not in the image
  (e.g., "a cat on the table" when there is no cat)
- Attribute hallucination: Wrong attributes (color, size)
- Relation hallucination: Wrong spatial/action relations

Object hallucinations are detected via noun set difference, which is a
deliberately simple but effective heuristic aligned with the CHAIR metric
(Rohrbach et al., 2018).

## Expected Output

```
Hallucination cases found: 47 / 500

--- Analyzing case #23 ---
  Generated : a cat sitting on a bench in a park
  GT caption: a woman sitting on a bench in a park
  Halluc.   : {'cat'}

Test gradient dim: 294912

Filtered training samples containing hallucinated nouns: 89

============================================================
TOP-10 MOST INFLUENTIAL TRAINING SAMPLES
(positive = promotes hallucination)
============================================================

  #1  sample_idx=156  influence=0.034521
       caption: a cat lying on a park bench next to a tree

  #2  sample_idx=892  influence=0.028734
       caption: two cats sitting on a wooden bench
  ...
```

## Extending This Work

### Scaling Up

- **Larger models**: Apply to LLaVA, InstructBLIP, or Qwen-VL using QLoRA (4-bit)
- **Full COCO**: Scale to 118k training samples with pre-filtering + random sampling
- **Multi-checkpoint TracIn**: Average influence across multiple LoRA checkpoints for stability

### Data Curation Loop

```
Detect hallucination → TracIn attribution → Remove/relabel top-k samples → Retrain → Evaluate
```

This produces a *closed-loop* system where TDA directly improves model quality.

### Alternative Influence Targets

- **Attribute hallucinations**: Target adjective tokens instead of nouns
- **Relation hallucinations**: Target verb/preposition tokens
- **Faithfulness score**: Use CLIPScore or CHAIR as the test loss

## Related Work

- **TracIn**: Pruthi, Liu, Kale, Sundararajan. *Estimating Training Data Influence by Tracing Gradient Descent.* NeurIPS 2020.
- **BLIP**: Li, Li, Savarese, Hoi. *BLIP: Bootstrapping Language-Image Pre-training.* ICML 2022.
- **LoRA**: Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- **CHAIR**: Rohrbach et al. *Object Hallucination in Image Captioning.* EMNLP 2018.
- **Influence Functions**: Koh & Liang. *Understanding Black-box Predictions via Influence Functions.* ICML 2017.
- **TRAK**: Park, Georgiev, Ilyas, Leclerc, Madry. *TRAK: Attributing Model Behavior at Scale.* ICML 2023.
- **LLaVA**: Liu et al. *Visual Instruction Tuning.* NeurIPS 2023.
- **Hallucination Survey**: Bai et al. *Hallucination of Multimodal Large Language Models: A Survey.* 2024.

## License

MIT

## Citation

```bibtex
@misc{tda4mllm2025,
  title={Training Data Attribution for Multimodal LLM Hallucination},
  author={Anonymous},
  year={2025},
  note={https://github.com/anonymous/tda_4_mllm}
}
```
