---
title: "RankMixer: Scaling Up Ranking Models in Industrial Recommenders"
date: 2026-06-29
description: "A walkthrough of the RankMixer architecture from ByteDance — a hardware-aware, attention-free ranking model that reaches 1B parameters with lower latency than its 8.7M parameter baseline."
tags: ["deep learning", "recommender systems", "mixture of experts", "mlp", "paper implementation"]
showToc: true
TocOpen: false
---

ByteDance's Douyin (TikTok) serves billions of users and over 100 million videos. The model deciding which video to show next — the ranking model — was running at roughly 4.5% GPU utilisation. That means for every 100 floating-point operations the hardware is capable of, the model was using fewer than 5.

The paper asks a direct question: *if you could design a hardware-aware, attention-free deep ranking model from scratch, what would it look like?*

The answer is RankMixer. By the end of this post you will understand the full architecture, why each design decision was made, and how the model reaches 1 billion parameters while running faster than the 8.7M parameter baseline it replaces.

A full PyTorch implementation is available [here](https://github.com/what-the-dev/RankMixer).

---

## The Problem with Scaling DLRMs

Deep Learning Recommendation Models (DLRMs) follow a common structure: multi-field feature data — user profiles, item attributes, interaction history — is converted into embeddings, passed through a dense interaction layer that models feature relationships, and then projected to a prediction.

The dense interaction layer is where the model's expressive capacity lives, and it is where most architectural effort has been spent. The problem is that most of these designs were built in the CPU era and inherited assumptions that do not hold on modern GPUs.

Specifically, the core operators in many DLRM interaction layers are **memory-bound** rather than **compute-bound**. A memory-bound operation is bottlenecked by how fast data can be transferred between memory and compute units. A compute-bound operation is bottlenecked by how fast the compute units can process data that is already available to them. Modern GPUs are extraordinarily fast at compute — large matrix multiplications can saturate thousands of CUDA cores simultaneously — but moving data in and out of memory is comparatively slow. When a model's critical path runs through memory-bound operations, the GPU sits mostly idle waiting for data. This is what a 4.5% MFU means in practice.

The naive response to this is to scale by depth and width — stack more layers, widen the hidden dimension. The gains are modest and occasionally negative. The Wukong model (Meta) shows the best scaling curve among prior work, but its architectural complexity is a major practical bottleneck: *"scaling up to high complexity presents notable challenges for real-time serving."*

The reason becomes clear when you decompose what latency actually depends on:

```
Latency ≈ (#Param × FLOPs/Param ratio) / (MFU × Hardware FLOPs)
```

This surfaces three independent levers:
1. **Lower the FLOPs/Param ratio** — add parameters without proportionally increasing compute.
2. **Raise MFU** — make better use of the hardware you already have.
3. **Raise hardware FLOPs** — quantisation (fp16/bf16) doubles theoretical peak throughput.

Most prior work only pulls the first lever, and only weakly. RankMixer is designed to pull all three simultaneously.

---

## Prior Approaches

It helps to understand what the alternatives look like before examining how RankMixer differs.

**Attention-based models** (AutoInt, WuKong, HiFormer, DHEN) learn feature interaction weights via self-attention. The mechanism works well in NLP and vision where all tokens share a unified semantic space — word embeddings or patch embeddings are meaningfully comparable. In recommendation systems, the feature space is inherently heterogeneous: a user ID embedding, an item category embedding, and a watch-duration embedding are not semantically comparable, so computing attention weights between them via inner product is on shaky ground. Beyond the modelling argument, self-attention is also memory-bound by construction — the full T×T attention weight matrix must be materialised in memory before the weighted sum can be computed.

**Non-attention models** (DCNv2, RDCN, DLRM-MLP) use explicit cross-product feature interactions or deep MLPs. These are more hardware-friendly but have limited capacity and do not scale well.

**MoE variants** scale by routing tokens to parallel expert sub-networks. The routing mechanism introduces problems of its own, which is covered in detail later.

Worth noting the lineage RankMixer draws from: Google's **MLP-Mixer** and **gMLP** demonstrated in computer vision that cross-token information mixing without attention can be surprisingly effective. RankMixer takes this idea and adapts it for recommendation — but the adaptation is non-trivial, precisely because recommendation feature tokens are not interchangeable the way image patches are.

---

## RankMixer: High-Level Architecture

The overall structure is straightforward:

```
Input (B, T, D)
  → RankMixerBlock × L
  → Flatten (B, T*D)
  → Linear head
  → Output (B, d_out)
```

Two properties define the design:

1. **Token mixing is parameter-free.** Cross-token information exchange happens through reshape and permute operations alone — no learned weights.
2. **Each token has its own independent FFN.** Rather than a shared transformation applied to all tokens, each token position has dedicated parameters.

The implementation reflects this directly:

```python
class RankMixer(nn.Module):
    def __init__(self, d_model, token_dim, num_heads, num_layers=2,
                 d_out=1, per_token_type='pffn', ffn_expansion_ratio=4,
                 num_experts=4, eps=1e-5):
        super().__init__()
        self.blocks = nn.ModuleList([
            RankMixerBlock(
                d_model=d_model, token_dim=token_dim, num_heads=num_heads,
                per_token_type=per_token_type, ffn_expansion_ratio=ffn_expansion_ratio,
                num_experts=num_experts, eps=eps,
            )
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model * token_dim, d_out)

    def forward(self, x):  # x: (B, T, D)
        for block in self.blocks:
            x = block(x)
        x = x.reshape(x.size(0), -1)  # (B, T*D)
        return self.head(x)
```

**A note on input tokenization.** The paper proposes a semantic clustering approach that groups the hundreds of raw feature embeddings into T dimension-aligned tokens (§3.2 of the paper). This is intentionally omitted from the implementation — how you tokenize inputs is problem-specific and data-specific. Semantic clustering simply means that input features are grouped into semantically similar feature vectors e.g user features, item features, user-item interaction features etc. Each group feature vector will have a different dimension. These need to be projected to a consistent input dimension `D`. They can then be batched and stacked. The model assumes inputs arrive as `(B, T, D)`.

**On scaling directions.** The paper identifies four orthogonal axes along which the model can be scaled: token count T, model dimension D, layer count L, and expert count E. Empirically, scaling D outperforms scaling L at the same parameter budget. The reason is hardware-shaped: larger D produces larger matrix multiplication shapes, which achieve better GPU utilisation. The production configurations reflect this — both the 100M and 1B models use only L=2 layers:

| Model | D | T | L |
|---|---|---|---|
| RankMixer-100M | 768 | 16 | 2 |
| RankMixer-1B | 1536 | 32 | 2 |

---

## The RankMixer Block

Each RankMixerBlock applies two sub-layers in a Pre-LayerNorm residual structure:

```
x → LN → Multi-Head Token Mixing → + x
  → LN → Per-Token Layer (PFFN or PReMoE) → + x
```

The forward pass is identical regardless of which per-token variant is used — the choice of PFFN or PReMoE is made at construction time and stored as `self.per_token_layer`. The forward method just runs the two sub-layers in sequence:

```python
class RankMixerBlock(nn.Module):
    def forward(self, x):
        residual = x
        x = self.ln_token(x)
        x = self.token_mixer(x)
        x = x + residual

        residual = x
        x = self.ln_per_token(x)
        x = self.per_token_layer(x)  # PFFN or PReMoE, set at init
        x = x + residual

        return x
```

{{< figure src="rankmixer_architecture.png" alt="RankMixer architecture diagram showing the RankMixerBlock with Token Mixing and Per-token FFN sub-layers, and the SMoE variant." caption="Figure 1: RankMixer block architecture. The token-mixing module divides each token's embedding into H heads, then recombines them across tokens. The per-token FFN sub-layer (and its SMoE variant) applies independent transformations per token position. *Image from Zhu et al., 2025 (arXiv:2507.15551).*" >}}

---

### Multi-Head Token Mixing

The first sub-layer is responsible for cross-token feature interaction — allowing information from one feature token to influence another. In a transformer, this is handled by self-attention. Here, it is handled by a parameter-free structured reshape and permute.

The intuition: each token's D-dimensional embedding is split into H heads of size D/H. These heads are then recombined across tokens, so each output "token" is composed of one head-slice from every input token.

The shape transformation:
```
(B, T, D)
  → reshape  → (B, T, H, D/H)
  → permute  → (B, H, T, D/H)
  → reshape  → (B, H, T*D/H)
```

Each of the H output tokens now contains one D/H-dimensional slice from every one of the T input tokens. Information has been mixed across the feature dimension without a single learned parameter.

```python
class MHTokenMixingBlock(nn.Module):
    def __init__(self, d_model, num_heads, token_dim):
        super().__init__()
        assert d_model % num_heads == 0
        assert token_dim == num_heads  # H == T required
        self.d_h = d_model // num_heads

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.shape
        x = x.reshape(B, T, self.num_heads, self.d_h)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, self.num_heads, T * self.d_h)
        return x
```

**Why this works better than attention for recommendation features.** The key constraint in recommendation is that the feature space is heterogeneous — user ID embeddings, item category embeddings, and watch-duration embeddings are not semantically comparable. Self-attention computes interaction weights via inner products between these heterogeneous vectors, which is not a meaningful operation. The paper quantifies this precisely in an ablation comparing routing strategies (Table 3 in the paper):

| Strategy | ΔAUC | ΔFLOPs |
|---|---|---|
| Multi-Head Token Mixing | baseline | baseline |
| Self-Attention | –0.03% | +71.8% |
| All-Share (no splitting) | –0.25% | 0% |
| All-Concat-MLP | –0.18% | 0% |

Self-attention achieves nearly identical accuracy — but at a 71.8% increase in FLOPs. Token mixing gets equivalent information exchange for free, and it does so without requiring the attention weight matrix to be materialised in memory. Reshape and permute are compute-bound rearrangements; there is no memory-bandwidth bottleneck. This is where the MFU improvement begins.

**The H == T constraint.** The number of heads must equal the number of tokens. This is required for the residual connection: the output `(B, H, T*D/H)` must have the same total size as the input `(B, T, D)`, which is satisfied since `H * (T * D/H) = T * D`. But it also means H and T are structurally coupled — you cannot change the number of tokens without changing the number of mixing heads.

---

### Per-Token FFN (PFFN) — Dense Variant

After token mixing, each token is transformed independently by a position-specific feed-forward network. The critical word is *position-specific*: token t always goes through FFN_t, and no parameters are shared across token positions.

This is different from a standard FFN, which applies one shared transformation to all tokens. It is also different from a Mixture-of-Experts (MMoE), where all experts see the same input and a routing mechanism decides which expert to use. In PFFN, the assignment is fixed — each FFN sees a structurally different input (a different feature token) and learns a dedicated transformation for that feature subspace.

The analogy is closer to a depthwise convolution with position-specific filters: the operation is local to each position, with no parameter sharing.

```python
class PFFN(nn.Module):
    def __init__(self, d_model, token_dim, expansion_ratio=4):
        super().__init__()
        self.ffns = nn.ModuleList([
            FFN(d_model, d_model * expansion_ratio)
            for _ in range(token_dim)
        ])

    def forward(self, x):  # x: (B, T, D)
        outputs = [self.ffns[t](x[:, t, :]) for t in range(self.token_dim)]
        return torch.stack(outputs, dim=1)  # (B, T, D)
```

The ablation confirms this matters: replacing per-token FFNs with a single shared FFN costs **–0.31% ΔAUC**. Shared FFNs collapse diverse feature subspaces into one transformation, causing high-frequency, dominant features to overshadow lower-frequency ones.

The parameter scaling behaviour is also notable: `#Param ≈ 4kLTD²`. Parameters grow linearly with T. This is the mechanism that allows the model to reach 1B parameters without proportionally increasing FLOPs — adding tokens adds parameters (more FFNs) but the compute per token stays constant.

---

## From Dense to Sparse: PReMoE

The PFFN variant is the dense baseline. To push toward 1B parameters and beyond, the per-token FFN is replaced with a per-token **Mixture-of-Experts** — each token position gets E small expert networks instead of one larger FFN.

### Why Vanilla Top-K MoE Fails Here

Standard sparse MoE routing uses Top-K selection: for each input, activate exactly k of the E experts. This has two specific problems in the per-token setting:

**Uniform token treatment.** Top-K forces every token to activate exactly k experts, regardless of how much information that token carries. High-information tokens (e.g., a rich user behaviour sequence) are under-served; low-information tokens over-served. The model cannot adapt capacity allocation to signal strength.

**Expert under-training.** Per-token FFNs already multiply the parameter count by T. Adding non-shared experts per token explodes this further. With Top-K routing, the already-large parameter space becomes unevenly trained: a few expert paths dominate, others rarely receive gradient, and you end up with dead or near-dead experts.

### ReLU Routing (ReMoE)

The solution is from Wang et al. (2024) — replace Top-K + softmax with a **ReLU gate**:

```
G = ReLU(Wx + b)    # shape: (B, E)
```

Each gate output is non-negative. Experts where the gate evaluates to zero are not activated. The number of active experts is *dynamic and data-driven* — no fixed k. High-information tokens produce larger activations and route to more experts; low-information tokens route to fewer, or potentially none.

Sparsity is regulated by an L1 penalty on the gate values:
```
L = L_task + λ · L_reg,    L_reg = Σ_i Σ_j G_{i,j}
```

The λ coefficient keeps the average number of active experts near a target budget without enforcing it rigidly per-sample.

```python
class ReLURouter(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.linear = nn.Linear(d_model, num_experts, bias=True)

    def forward(self, x):
        return F.relu(self.linear(x))  # (B, E), non-negative


class ReMoE(nn.Module):
    def __init__(self, d_model, num_experts, expansion_ratio=4):
        super().__init__()
        self.router = ReLURouter(d_model, num_experts)
        self.experts = nn.ModuleList([
            FFN(d_model, d_model * expansion_ratio)
            for _ in range(num_experts)
        ])

    def forward(self, x):  # x: (B, D)
        gates = self.router(x)                          # (B, E)
        expert_outputs = torch.stack(
            [e(x) for e in self.experts], dim=1
        )                                               # (B, E, D)
        return torch.sum(expert_outputs * gates.unsqueeze(-1), dim=1)
```

**What the routing learns.** Figure 4 in the paper shows the activated expert ratio for each token position across layers. The ratios vary meaningfully — they are not uniform, and they are not random. High-information token positions consistently activate a larger share of experts; lower-information positions activate fewer. This is the empirical confirmation that ReLU routing is learning something real about which feature tokens carry more information, and dynamically allocating model capacity accordingly.

### Dense-Training / Sparse-Inference (DTSI)

ReLU routing introduces a practical training challenge: because gate sparsity is learned rather than enforced, experts can still fall into under-training if routing collapses early. The paper addresses this with a dual-router training strategy.

During training, two routers are maintained simultaneously:
- **h_train** — a dense router (all experts receive input, no gating)
- **h_infer** — the sparse ReLU router

The L1 regularisation loss is applied to *both*. The dense router ensures every expert receives consistent gradient throughout training — no expert can starve because h_train always routes to all of them. The sparse router learns to route selectively under regularisation pressure from h_infer.

At inference, h_train is discarded and only h_infer is used.

The result: at 1/8 sparsity (only 12.5% of experts are active at inference), the model retains nearly all of the dense model's accuracy, with over 50% throughput improvement. Vanilla SMoE with a load-balancing loss degrades monotonically as sparsity increases. Dense-training is what makes the difference.

```python
class PReMoE(nn.Module):
    def __init__(self, d_model, token_dim, num_experts, expansion_ratio=4):
        super().__init__()
        self.remoes = nn.ModuleList([
            ReMoE(d_model, num_experts, expansion_ratio)
            for _ in range(token_dim)
        ])

    def forward(self, x):  # x: (B, T, D)
        outputs = [self.remoes[t](x[:, t, :]) for t in range(self.token_dim)]
        return torch.stack(outputs, dim=1)  # (B, T, D)
```

---

## Results

### Offline Comparison (~100M parameters)

Training data: Douyin recommendation logs, 300+ features, trillions of records per day over a two-week window. All models compared at roughly the same parameter count. Primary metric: AUC on Finish label (did the user complete watching the video). An AUC improvement of 0.0001 is considered confidently significant at this scale.

| Model | Finish AUC↑ | Finish UAUC↑ | Params | FLOPs/Batch |
|---|---|---|---|---|
| DLRM-MLP (base) | 0.8554 | 0.8270 | 8.7M | 52G |
| DCNv2 | +0.13% | +0.13% | 22M | 170G |
| DHEN | +0.18% | +0.26% | 22M | 158G |
| HiFormer | +0.48% | — | 116M | 326G |
| Wukong | +0.29% | +0.29% | 122M | 442G |
| **RankMixer-100M** | **+0.64%** | **+0.72%** | 107M | 233G |

RankMixer-100M leads on accuracy while using fewer FLOPs than HiFormer and less than half the FLOPs of Wukong.

### Scaling to 1B

| Model | #Param | FLOPs | MFU | Latency |
|---|---|---|---|---|
| Base-DLRM-8.7M | 8.7M | 52G | 4.51% | 16.12ms |
| Wukong (l=8, nL=32) | 122M | 442G | 18.51% | 33.7ms |
| **RankMixer-1B** | **1B** | **2,106G** | **44.57%** | **14.3ms** |

100× more parameters than the baseline. Lower latency. This is the headline result, and it is made possible by the latency decomposition introduced earlier:

```
Latency ≈ (#Param × FLOPs/Param ratio) / (MFU × Hardware FLOPs)
```

Each term is pulled in the right direction:

1. **FLOPs/Param ratio reduced 3.6×** (6.8 → 1.9 G/M): the per-token design adds parameters without proportionally increasing compute per forward pass.
2. **MFU raised ~10×** (4.47% → 44.57%): large D produces large GEMM shapes that saturate GPU compute; per-token FFNs are fused into a single 3D tensor kernel, eliminating per-kernel launch overhead (30% speedup).
3. **Hardware FLOPs doubled** via fp16 for matmul-heavy operations (45% throughput increase, 31.5% latency reduction). LayerNorm stays fp32.
4. **Sparse-GEMM for PReMoE** cuts end-to-end inference latency by over 40% at 1/8 sparsity.

### Online A/B Results

RankMixer-1B was deployed to full Douyin feed recommendation traffic, replacing a 16M-parameter baseline combining DLRM and DCN.

| Metric | Lift |
|---|---|
| Active Days | +0.2008% |
| Duration | +0.4996% |
| Finish rate | +1.6016% |
| Advertising AUC | +0.73% |
| Advertiser Value (ADVV) | +3.90% |

The paper notes that these gains had not converged at publication — long-term A/B experiments were still trending upward.

---

## Implementation Notes

The full PyTorch implementation is available [here](https://github.com/what-the-dev/RankMixer).

A few things worth noting for anyone working with or adapting the code:

**Two variants, one flag.** The `per_token_type` argument on `RankMixer` and `RankMixerBlock` selects between `"pffn"` (dense) and `"premoe"` (sparse MoE). The rest of the architecture is identical.

**The H == T constraint.** `MHTokenMixingBlock` asserts `num_heads == token_dim`. This is not obvious from the paper diagram but falls out of the residual connection requirement. If you change the number of tokens in your application, you must change the number of heads to match.

**DTSI is not implemented.** The implementation covers the inference-time architecture faithfully. Full training with dense-training / sparse-inference would require adding the h_train dense router alongside h_infer and applying the L1 regularisation loss to both during the training loop.

**No input layer.** The model takes `(B, T, D)` directly. How you get there — semantic tokenization, projection, grouping of raw features — is left to the user.

---

*Paper: Zhu et al., "RankMixer: Scaling Up Ranking Models in Industrial Recommenders," arXiv:2507.15551 (2025).*
