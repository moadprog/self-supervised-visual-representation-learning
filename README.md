# self-supervised-visual-representation-learning with SimCLR

A PyTorch study of self-supervised visual representation learning with **SimCLR**, evaluated on **Food-101** using linear probing, supervised fine-tuning, and controlled ablation experiments.

The project investigates not only whether contrastive pretraining learns useful visual representations, but also **how representation quality evolves during training and which SimCLR design choices matter most**.

---

## Overview

Modern deep learning models usually rely on large labeled datasets. Self-supervised learning instead attempts to learn useful representations directly from unlabeled data.

This project implements the core SimCLR pipeline:

\[
x
\rightarrow
(t_1(x), t_2(x))
\rightarrow
f_\theta
\rightarrow
h
\rightarrow
g_\phi
\rightarrow
z
\rightarrow
\mathcal{L}_{\mathrm{NT-Xent}}
\]

where:

- \(t_1,t_2\) are stochastic image augmentations,
- \(f_\theta\) is a ResNet-50 encoder,
- \(h \in \mathbb{R}^{2048}\) is the learned representation,
- \(g_\phi\) is a nonlinear projection head,
- \(z \in \mathbb{R}^{128}\) is the contrastive embedding,
- \(\mathcal{L}_{\mathrm{NT-Xent}}\) is the normalized temperature-scaled cross-entropy loss.

The encoder is trained **without using Food-101 labels** during self-supervised pretraining.

The learned representation is then evaluated through:

1. **Linear probing**
2. **End-to-end supervised fine-tuning**
3. **Pretraining-duration analysis**
4. **Augmentation ablations**
5. **Projection-head ablation**

---

## Dataset

The main experiments use **Food-101**, containing 101 food categories.

| Split | Images |
|---|---:|
| Training | 75,750 |
| Test | 25,250 |
| Total | 101,000 |

During SimCLR pretraining, the class labels are ignored.

Each training image generates two independently augmented views:

\[
x \rightarrow x_1,\;x_2
\]

and the two views form a positive pair.

---

## Architecture

### Encoder

The encoder is a **ResNet-50** instantiated from `torchvision` with randomly initialized weights.

The original classification layer is removed:

\[
\text{ResNet-50}
\rightarrow
h \in \mathbb{R}^{2048}.
\]

No ImageNet-pretrained weights are used.

### Projection head

The SimCLR projection head is a two-layer MLP:

\[
2048
\rightarrow
2048
\rightarrow
\mathrm{ReLU}
\rightarrow
128.
\]

The contrastive objective is applied to the projected representation \(z\), while the encoder representation \(h\) is retained for downstream evaluation.

### Contrastive loss

For a batch of \(N\) images, two augmentations are generated for every image, producing \(2N\) representations.

For each anchor:

- one representation is its positive,
- the remaining \(2N-2\) representations act as negatives.

The model is trained with the NT-Xent loss:

\[
\ell_i
=
-\log
\frac{
\exp(\operatorname{sim}(z_i,z_j)/\tau)
}{
\sum_{k\neq i}
\exp(\operatorname{sim}(z_i,z_k)/\tau)
}.
\]

The main experiment uses:

\[
\tau = 0.5.
\]

---

## Main Pretraining Setup

The reference SimCLR model was trained with:

| Parameter | Value |
|---|---:|
| Encoder | ResNet-50 |
| Dataset | Food-101 |
| Input resolution | \(224\times224\) |
| Batch size | 128 |
| Epochs | 100 |
| Projection dimension | 128 |
| Temperature | 0.5 |
| Optimizer | AdamW |
| Initial learning rate | \(3\times10^{-4}\) |
| Weight decay | \(10^{-4}\) |
| LR schedule | Cosine annealing |
| Precision | Automatic mixed precision |

With batch size 128, every anchor is contrasted against:

\[
254
\]

negative samples within the batch.

Training was performed on an NVIDIA Tesla T4 GPU.

---

# Results

## 1. Linear Probe

The first evaluation asks:

> Are the representations learned by SimCLR already useful without modifying the encoder?

The encoder is frozen and only a linear classifier is trained:

\[
h_{2048}
\rightarrow
\mathrm{Linear}(2048,101).
\]

The same protocol is applied to a randomly initialized frozen ResNet-50.

| Frozen encoder | Top-1 accuracy |
|---|---:|
| Random ResNet-50 | **6.27%** |
| SimCLR ResNet-50 | **56.42%** |

\[
\boxed{+50.15\text{ percentage points}}
\]

![Linear probe comparison](figures/linear_probe_comparison.png)

This provides direct evidence that self-supervised contrastive pretraining transformed the encoder into a representation space where Food-101 categories are substantially more linearly separable.

Importantly, the classifier itself is only linear. The performance therefore reflects information already encoded in the learned representation.

---

## 2. Supervised Fine-Tuning

The second experiment asks:

> Does SimCLR also provide a better initialization for fully supervised training?

Two ResNet-50 models are compared:

\[
\text{Random initialization}
\rightarrow
\text{supervised training}
\]

versus

\[
\text{SimCLR initialization}
\rightarrow
\text{supervised fine-tuning}.
\]

All encoder parameters are trainable in both cases.

| Initialization | Test Top-1 | Test Top-5 |
|---|---:|---:|
| Supervised from scratch | **75.43%** | **93.13%** |
| SimCLR pretrained + fine-tuned | **78.65%** | **94.57%** |

SimCLR pretraining improves Top-1 accuracy by:

\[
\boxed{+3.21\text{ percentage points}}
\]

![Fine-tuning comparison](figures/finetuning_comparison.png)

The gain is naturally smaller than in linear probing because the supervised baseline is allowed to learn its entire representation from labels.

Nevertheless, the pretrained encoder reaches a better final downstream solution.

### Fine-tuning convergence

![Fine-tuning learning curves](figures/finetuning_learning_curves.png)

The learning curves also allow comparison of the optimization dynamics of random initialization and SimCLR initialization.

---

# Ablation Studies

## 3. Representation Quality vs Pretraining Duration

To measure how representation quality evolves during self-supervised training, intermediate SimCLR checkpoints were evaluated using the same linear-probe protocol.

| Pretraining epochs | Linear-probe Top-1 |
|---:|---:|
| 10 | **29.80%** |
| 20 | **39.62%** |
| 50 | **52.48%** |
| 100 | **56.40%** |

![Pretraining duration](figures/pretraining_duration.png)

Representation quality improves consistently with longer pretraining.

The gains are particularly large early in training:

\[
10\rightarrow20:
+9.82\text{ pp}
\]

\[
20\rightarrow50:
+12.86\text{ pp}
\]

while:

\[
50\rightarrow100:
+3.92\text{ pp}.
\]

This indicates clear diminishing returns after approximately 50 epochs, although additional pretraining still improves downstream representation quality.

---

## 4. Augmentation Ablation

SimCLR relies heavily on the construction of positive pairs.

The reference augmentation pipeline includes:

- random resized cropping,
- horizontal flipping,
- color jitter,
- grayscale conversion,
- Gaussian blur.

Three controlled modifications were evaluated after 30 epochs of SimCLR pretraining.

| Augmentation setup | Linear-probe Top-1 | Difference vs baseline |
|---|---:|---:|
| Weak crop | **36.67%** | -8.70 pp |
| No color jitter | **38.79%** | -6.58 pp |
| Full augmentation | **45.37%** | — |
| No Gaussian blur | **50.77%** | +5.40 pp |

![Augmentation ablation](figures/augmentation_ablation.png)

### Strong cropping matters

Weakening the random crop produces the largest degradation:

\[
45.37\%
\rightarrow
36.67\%.
\]

Strong spatial transformations therefore appear important for forcing the model to learn semantic invariance rather than relying on local visual correspondence.

### Color jitter matters

Removing color jitter reduces performance by:

\[
6.58\text{ pp}.
\]

Color perturbations prevent the model from solving the contrastive task primarily through low-level color statistics.

### Gaussian blur behaves differently on Food-101

Removing Gaussian blur improves performance:

\[
45.37\%
\rightarrow
50.77\%.
\]

In the reference implementation used here, Gaussian blur is applied to every augmented view.

Food categories often depend on fine texture, surface structure, toppings, or preparation details. Applying blur systematically may therefore remove information useful for learning Food-101 representations.

This result should not be interpreted as a general claim that Gaussian blur is harmful to SimCLR; it is specific to the dataset and augmentation configuration studied here.

---

## 5. Projection Head Ablation

SimCLR does not optimize the contrastive objective directly on the encoder representation \(h\).

Instead:

\[
h
\rightarrow
g_\phi(h)
\rightarrow
z
\rightarrow
\mathcal{L}_{\mathrm{NT-Xent}}.
\]

To test the importance of this design choice, a second model was trained without the projection head:

\[
h
\rightarrow
\mathcal{L}_{\mathrm{NT-Xent}}.
\]

Both models were evaluated after 30 epochs using linear probing.

| Architecture | Linear-probe Top-1 |
|---|---:|
| Without projection head | **31.21%** |
| With projection head | **45.37%** |

\[
\boxed{+14.16\text{ percentage points}}
\]

![Projection-head ablation](figures/projection_head_ablation.png)

The result supports the idea that the projection head provides a useful separation between:

\[
h:
\text{representation retained for downstream tasks}
\]

and

\[
z:
\text{representation specialized for contrastive optimization}.
\]

Optimizing NT-Xent directly on \(h\) produces substantially weaker downstream representations.

---

# Main Findings

The experiments lead to four main conclusions:

1. **SimCLR learns strong representations without class labels.**  
   Linear-probe accuracy increases from **6.27% with random frozen features to 56.42% with SimCLR features**.

2. **Self-supervised pretraining improves downstream supervised learning.**  
   Fine-tuning improves from **75.43% to 78.65% Top-1 accuracy** compared with supervised training from scratch.

3. **Representation quality strongly depends on augmentation design.**  
   Strong random cropping and color jitter substantially improve downstream representation quality, while always applying Gaussian blur hurts performance in this Food-101 setting.

4. **The projection head is a critical part of SimCLR.**  
   Removing it reduces 30-epoch linear-probe accuracy from **45.37% to 31.21%**.

Overall, the project demonstrates that contrastive pretraining is not merely optimizing an auxiliary loss: it progressively constructs a feature space that transfers effectively to semantic classification.

---

# Repository Structure

```text
self-supervised-visual-representation-learning/
│
├── src/
│   ├── data/
│   │   ├── augmentations.py
│   │   └── datasets.py
│   │
│   ├── models/
│   │   ├── encoder.py
│   │   ├── projection_head.py
│   │   └── simclr.py
│   │
│   ├── losses/
│   │   └── nt_xent.py
│   │
│   └── evaluate/
│       └── linear_probe_utils.py
│
├── scripts/
│   ├── train_stl10.py
│   ├── train_food101.py
│   ├── linear_probe_food101.py
│   ├── fine_tune_food101.py
│   ├── experiment_training_duration.py
│   ├── ablation_augmentations_food101.py
│   ├── ablation_projection_head_food101.py
│   ├── generate_figures.py
│   └── bootstrap_onyxia.sh
│
├── results/
│   ├── linear_probe_food101.json
│   ├── food101_finetuning_comparison.json
│   ├── supervised_scratch_history.csv
│   ├── simclr_finetuned_history.csv
│   ├── training_duration_study.json
│   ├── augmentation_ablation.json
│   └── projection_head_ablation.json
│
├── figures/
│   ├── linear_probe_comparison.png
│   ├── finetuning_comparison.png
│   ├── finetuning_learning_curves.png
│   ├── pretraining_duration.png
│   ├── augmentation_ablation.png
│   └── projection_head_ablation.png
│
├── requirements.txt
├── .gitignore
└── README.md