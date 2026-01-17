# FLUX.2-klein-4B Pure C Inference Engine - TODO

## Current State (2026-01-17)

### COMPLETED
- [x] VAE decoder implementation - fully working, matches Python output exactly
  - Fixed upsample weight order issue
  - Fixed post_quant_conv implementation
  - Verified with 64x64 test images
- [x] Weight loading from safetensors format
- [x] Sinusoidal timestep embedding - matches Python exactly
  - Fixed: divisor changed from `(half-1)` to `half`
- [x] RoPE positional encoding - matches Python exactly
  - Fixed axis mapping: y/x go in dims 64-127 (axes 2,3), not dims 0-63
  - Fixed repeat_interleave format (pairs of identical values)
  - Fixed complex rotation formula: out[i] = x[i]*cos - x[i+1]*sin
- [x] Python inference verification - confirmed produces coherent images
- [x] QKV projection bug fix - changed from fused QKV to separate Q, K, V projections
- [x] AdaLN normalization fix - changed from RMSNorm to LayerNorm
  - FLUX2 uses LayerNorm(elementwise_affine=False) before modulation, not RMSNorm!
  - QK normalization still uses RMSNorm with learnable weights (correct)

### COMPLETED (Single Block Fixes)
- [x] Fixed concatenation order: changed from [img, txt] to [txt, img] to match Python
- [x] Fixed RoPE application: apply image RoPE at offset (after text tokens)
- [x] Fixed fused QKV split: de-interleave from [seq, fused_dim] format
  - **Bug**: C was treating output as [Q_all, K_all, V_all, ...] but layout is [pos0_Q+K+V+gate+up, pos1_Q+K+V+gate+up, ...]
  - **Result**: Single block output now matches Python exactly

### COMPLETED (Double Block RoPE Fix)
- [x] Fixed text RoPE bug
  - **Root cause**: C was applying sequential position RoPE to text (positions 0, 1, 2, ...)
  - **Correct behavior**: FLUX.2 text tokens have txt_ids = [0, 0, 0, 0] (all zeros)
  - **Fix**: Skip text RoPE entirely (identity RoPE = no rotation)
  - **Result**: Output now matches Python exactly (mean=0.019203, std=1.052191 vs Python 0.019203, 1.052202)

### COMPLETED (Double Block Fixes)
- [x] Changed apply_adaln from RMSNorm to LayerNorm
- [x] Fixed sinusoidal timestep embedding to use [cos, sin] order (flip_sin_to_cos=True)
- [x] Fixed QKV to use separate projections instead of fused
- [x] Verified double block output matches Python with minor numerical precision differences

### COMPLETED (End-to-End Verification)
- [x] End-to-end image generation test
  - CLI (`./flux`) successfully runs and produces images
  - Tested with 64x64 output, 4 steps, seed 42
  - VAE decoding works - produces valid PPM output
  - Transformer forward pass runs through all 5 double + 20 single blocks
  - **Fixed**: Final norm scale/shift order was reversed (Python uses scale first, shift second)
  - **Result**: C velocity output matches Python within fp32 precision (mean diff: 0.000019, max diff: 0.000099)

### TODO
- [ ] Implement Qwen 4B text encoder (8GB weights in text_encoder/ directory)
- [ ] Performance optimization (OpenMP parallelization, SIMD)

### COMPLETED (Code Cleanup)
- [x] Removed debug printf statements from flux_transformer.c

## Development Good Practices

### Compilation
Always use optimization flags for faster development iterations:
```bash
gcc -O3 -march=native -ffast-math -o <output> <sources> -lm
```

### Testing
1. Run `make` with default flags for quick compilation
2. For isolated tests, compile directly with optimization flags -O3 -march=native -ffast-math
3. Use Python scripts for reference values
4. Compare intermediate values step-by-step when debugging

## Architecture Notes

### FLUX.2-klein-4B Model
- 5 double-stream transformer blocks
- 20 single-stream transformer blocks
- Hidden dim: 3072 (internal), 7680 (text encoder output)
- Attention heads: 24
- Head dim: 128
- MLP hidden: 9216
- RoPE: 4 axes with dims [32, 32, 32, 32] = 128 total
  - Axis 0-1: always 0 for images (cos=1, sin=0)
  - Axis 2: y/height position (dims 64-95)
  - Axis 3: x/width position (dims 96-127)
- Text encoder: Qwen 4B (separate from main transformer)

### Key Implementation Details

**Double Block Normalization:**
- `norm1` / `norm1_context`: LayerNorm(elementwise_affine=False) - mean subtraction
- `norm2` / `norm2_context`: LayerNorm(elementwise_affine=False) - mean subtraction
- `norm_q` / `norm_k`: RMSNorm with learnable weights [head_dim]

**Attention:**
- Joint attention: Q,K,V concatenated as [txt, img] before attention
- RoPE applied AFTER concatenation
- Output projection has no bias

**FFN:**
- SwiGLU activation: silu(gate) * up
- linear_in: [hidden, mlp_hidden*2] (gate+up fused)
- linear_out: [mlp_hidden, hidden]

## Debug Scripts
- `test_attention_detail.py` - Traces attention step by step, matches actual block exactly
- `test_joint_attention_v2.py` - Manual attention computation
- `test_transformer_simple_debug.py` - Step-by-step Python transformer debug
- `test_rope_compare.py` - RoPE comparison between Python and C

## Key Findings
1. guidance_embedder weights not in checkpoint - must zero them in Python for fair comparison
2. Model loading must use: load_config + from_config + manual safetensors load
3. Timestep scaled by 1000 before embedding (t=0.5 -> 500.0)
4. **FLUX2 uses LayerNorm, not RMSNorm for the main stream normalization**
5. QK normalization uses RMSNorm with learnable per-head weights
6. **Text tokens have txt_ids = [0,0,0,0] - text RoPE is always identity (no rotation)**
