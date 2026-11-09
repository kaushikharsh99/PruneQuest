# Pruning Methodologies & Algorithms in PruneQuest

PruneQuest is built strictly for **physical structural compression**. It does not use unstructured weight masks or artificial zero-padding that leave parameter counts unchanged.

---

## 1. Block Influence (BI) Layer Scoring

PruneQuest implements layer importance estimation based on the **ShortGPT** Block Influence methodology.

### Mathematical Formulation
For layer $l \in [0, L-1]$ with input hidden states $X^{(l)} \in \mathbb{R}^{B \times T \times D}$ and output hidden states $X^{(l+1)} \in \mathbb{R}^{B \times T \times D}$:

$$\text{CosineSimilarity}(X^{(l)}_{b, t}, X^{(l+1)}_{b, t}) = \frac{X^{(l)}_{b, t} \cdot X^{(l+1)}_{b, t}}{\|X^{(l)}_{b, t}\|_2 \|X^{(l+1)}_{b, t}\|_2}$$

The Block Influence (BI) distance of layer $l$ across batch $B$ and token sequence $T$ is:

$$\text{BI}(l) = 1 - \frac{1}{|S|} \sum_{(b, t) \in S} \text{CosineSimilarity}(X^{(l)}_{b, t}, X^{(l+1)}_{b, t})$$

- **Low BI Score ($ \approx 0 $)**: The output representations of layer $l$ are almost identical to its inputs; the layer acts as a near-identity mapping and can be pruned with minimal behavioral change.
- **High BI Score**: The layer drastically transforms representations and is critical to model capability.

---

## 2. Physical Layer Pruning Procedure

When layer $k$ is selected for removal:
1. `adapter.model.model.layers.pop(k)` removes the module physically from `nn.ModuleList`.
2. `config.num_hidden_layers` is decremented by 1.
3. `config.layer_types.pop(k)` ensures strict adherence to Hugging Face `transformers` schema validation.
4. For all remaining layers $j \ge k$, `layer.self_attn.layer_idx = j` is updated to prevent key-value cache corruption in Rotary Position Embeddings (RoPE).

---

## 3. Layer Pruning vs. Attention Head Pruning

| Pruning Dimension | Supported in v0.1.0 | Technical Status / Reason |
| :--- | :--- | :--- |
| **Physical Layers** | **Yes** | Fully automated, validated, and verified on Hugging Face checkpoints. |
| **MLP Intermediate Slicing** | Experimental | Slices intermediate dimension across `gate_proj`, `up_proj`, and `down_proj`. |
| **Attention Head Pruning** | Planned for v0.2.0 | Blocked by upstream Hugging Face Qwen2 coupling: `head_dim = hidden_size // num_attention_heads`. Reducing heads modifies `head_dim`, breaking precomputed RoPE frequencies unless hidden dimension is concurrently modified. See `docs/compatibility.md`. |
