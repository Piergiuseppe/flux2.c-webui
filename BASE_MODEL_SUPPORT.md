# Plan: FLUX.2 Klein Base 4B Support

## Overview

Add support for the undistilled FLUX.2-klein-base-4B model alongside the
existing distilled FLUX.2-klein-4B. The base model uses the **same architecture
and weights format** but requires **Classifier-Free Guidance (CFG)** at
inference time: two transformer forward passes per step (conditioned +
unconditioned), combined post-hoc.

Key differences from distilled:
- 50 steps default (vs 4), configurable
- guidance_scale = 4.0 default (vs 1.0)
- Two forward passes per step (empty-prompt + real-prompt)
- The model never receives a `guidance` parameter — CFG is external

## Autodetection

The HuggingFace model repos include a `model_index.json` at the root:

- **Distilled** has `"is_distilled": true`
- **Base** does NOT have `"is_distilled"` (field absent)

Plan:
1. The download scripts will download `model_index.json` into the model directory.
2. At load time (`flux_load_dir()`), parse `model_index.json` with a minimal
   JSON string search (no JSON library needed — just look for
   `"is_distilled"` and check if its value is `true`).
3. Store the result in `flux_ctx` as `int is_distilled` (1 = distilled, 0 = base).
4. Set defaults accordingly: steps=4/guidance=1.0 for distilled,
   steps=50/guidance=4.0 for base.
5. Provide `--base` CLI flag as manual override (forces base-model mode even
   if autodetection says otherwise, or if model_index.json is missing).

## Changes by File

### 1. `download_model.sh`

Add a `--base` option that switches the HuggingFace repo URL from
`FLUX.2-klein-4B` to `FLUX.2-klein-base-4B` and the output directory to
`./flux-klein-base-model`. Both variants download `model_index.json`.

**Important**: `model_index.json` is NOT currently downloaded by either
script. Both scripts must be updated to fetch it for both the distilled
and base model paths, since it is required for autodetection.

The text encoder, tokenizer, and VAE repos are the same for both models
(same Qwen3 4B, same VAE). Only the transformer weights differ. The script
structure stays the same, just with the repo URL parameterized.

### 2. `download_model.py`

Same changes: add `--base` flag that switches repo to
`black-forest-labs/FLUX.2-klein-base-4B` and default output dir to
`./flux-klein-base-model`. Add `model_index.json` to download patterns.

### 3. `flux.h`

Add to `flux_params`:
```c
float guidance;    /* CFG guidance scale (default: auto from model type) */
```

Add to `flux_ctx`:
```c
int is_distilled;  /* 1 = distilled (klein-4B), 0 = base (klein-base-4B) */
```

Add default: update `FLUX_PARAMS_DEFAULT` to include `guidance = 0.0`
(0 = auto, meaning: use model default — 1.0 for distilled, 4.0 for base).

### 4. `flux.c`

**Model detection** — in `flux_load_dir()`:
- After loading the model directory path, read `<model_dir>/model_index.json`.
- Search for `"is_distilled"` substring. If found and followed by `true`,
  set `ctx->is_distilled = 1`. Otherwise `ctx->is_distilled = 0`.
- Set `ctx->default_steps` to 4 or 50 accordingly.

**Empty-prompt encoding for CFG** — add a helper:
```c
float *flux_encode_empty_text(flux_ctx *ctx, int *out_seq_len);
```
This calls `flux_encode_text(ctx, "", out_seq_len)` — the empty string goes
through the same Qwen3 chat template, producing embeddings for the template
structure with no content, exactly as the Python reference does.

**Generation functions** — modify `flux_generate()`, `flux_img2img()`,
`flux_multiref()`:

For base model (CFG path):
1. Encode the real prompt: `text_emb_cond = flux_encode_text(ctx, prompt, ...)`
2. Encode the empty prompt: `text_emb_uncond = flux_encode_text(ctx, "", ...)`
3. Free the text encoder (as before, to save memory).
4. Load transformer.
5. Call `flux_sample_euler_cfg()` (new function, see below) passing both
   embeddings and the guidance scale.

For distilled model: no change — existing code path.

The branch is simply:
```c
if (ctx->is_distilled) {
    /* existing code path — single text_emb, flux_sample_euler() */
} else {
    /* CFG path — two text_embs, flux_sample_euler_cfg() */
}
```

### 5. `flux_sample.c`

**New function: `flux_sample_euler_cfg()`**

Matches the Python `denoise_cfg()` logic exactly:

```c
float *flux_sample_euler_cfg(void *transformer, void *text_encoder,
                              float *z, int batch, int channels, int h, int w,
                              const float *text_emb_cond, int text_seq_cond,
                              const float *text_emb_uncond, int text_seq_uncond,
                              float guidance_scale,
                              const float *schedule, int num_steps,
                              void (*progress_callback)(int step, int total));
```

Implementation per step (following Python reference):
1. Run transformer forward with `text_emb_uncond` → get `v_uncond`
2. Run transformer forward with `text_emb_cond` → get `v_cond`
3. Combine: `v = v_uncond + guidance_scale * (v_cond - v_uncond)`
4. Euler step: `z += dt * v`
5. Free `v_uncond`, `v_cond`

Note: the Python reference batches both into a single forward pass with
batch=2 for GPU efficiency. In our C implementation we run them sequentially
as two separate calls — this is simpler and equivalent in result. Batching
would require significant changes to the transformer for batch>1 support
with no benefit on MPS/CPU where we're memory-bound anyway.

Also add `flux_sample_euler_cfg_with_refs()` and
`flux_sample_euler_cfg_with_multi_refs()` variants for img2img with CFG,
following the same pattern (the Python reference doubles the reference
tokens along the batch dimension too).

### 6. `main.c`

- Add `--guidance` / `-g` option (float, default 0 = auto).
- Add `--base` flag (force base-model mode).
- Resolve guidance: if user specified guidance > 0, use it. Otherwise use
  model default (1.0 distilled, 4.0 base).
- Resolve steps: if user didn't specify steps, use model default (4 or 50).
- Pass `params.guidance` through to generation functions.

### 7. `flux_cli.c`

- Add `!guidance <value>` command to set/show guidance scale.
- When in base-model mode, default to 50 steps and guidance=4.0 for new
  generations.
- Display current model type (distilled/base) in `!help` or status output.

### 8. Targets (generic, openblas, mps)

No target-specific changes needed. The CFG logic is in `flux_sample.c` which
is pure C and shared across all targets. The transformer forward function
is already target-agnostic at the call site — Metal/BLAS/generic dispatch
happens inside `flux_transformer_forward()`. We just call it twice per step
instead of once.

The only concern is memory: running two forward passes means we temporarily
hold two velocity buffers (`v_cond` and `v_uncond`), each of size
`channels * h * w * sizeof(float)` = `128 * h * w * 4` bytes. For a 1024x1024
image this is `128 * 64 * 64 * 4 = 2MB` per buffer — negligible.

### 9. Makefile

No changes needed. No new source files, no new dependencies.

## What Does NOT Change

- Transformer architecture, kernels, Metal shaders — identical weights
- VAE encoder/decoder — same model
- Qwen3 text encoder — same model, just called twice for base
- Weight loading (`flux_safetensors.c`) — same format
- Timestep schedule (`flux_official_schedule()`) — already parameterized
  by num_steps, handles 50 steps correctly
- RoPE computation — identical
- Image I/O — identical

## Implementation Order

1. Download scripts (add `--base`, add `model_index.json`)
2. `flux.h` (add fields to structs)
3. `flux.c` (autodetection in `flux_load_dir()`, CFG generation paths)
4. `flux_sample.c` (add `flux_sample_euler_cfg()` + ref variants)
5. `main.c` (CLI flags)
6. `flux_cli.c` (interactive mode support)
7. Test with `make test` (distilled regression — must still pass)
8. Download base model, test with base model manually

## Testing

- `make test` must still pass (distilled model regression test).
- Manual test with base model:
  ```
  ./flux2 -d flux-klein-base-model -p "A fluffy orange cat" \
    --seed 42 --steps 50 -o /tmp/base_test.png -W 256 -H 256
  ```
- Verify autodetection works (no `--base` flag needed when model_index.json
  is present).
- Verify `--base` override works when model_index.json is missing.
- Verify distilled model still works identically (no regression).
