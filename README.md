# FLUX.2-klein-4B Pure C Implementation

Generate images from text prompts using FLUX.2-klein-4B, implemented entirely in C with zero external dependencies beyond the C standard library and BLAS.

## Quick Start

```bash
# Build (no dependencies needed)
make

# Download the model (~16GB)
pip install huggingface_hub
python download_model.py

# Generate an image
./flux -d flux-klein-model -p "A woman wearing sunglasses" -o output.png
```

That's it. No Python runtime, no PyTorch, no CUDA toolkit required at inference time.

## Example Output

![Woman with sunglasses](images/woman_with_sunglasses.png)

*Generated with: `./flux -d flux-klein-model -p "A picture of a woman in 1960 America. Sunglasses. ASA 400 film. Black and White." -W 250 -H 250 -o /tmp/woman.png`, and later processed with image to image generation via `./flux -d flux-klein-model -i /tmp/woman.png -o /tmp/woman2.png -p "oil painting of woman with sunglasses" -v -H 256 -W 256`*

## Features

- **Zero dependencies**: Pure C implementation, only needs BLAS (Apple Accelerate on macOS, OpenBLAS on Linux)
- **Metal GPU acceleration**: Automatic on Apple Silicon Macs
- **Text-to-image**: Generate images from text prompts
- **Image-to-image**: Transform existing images guided by prompts
- **Integrated text encoder**: Qwen3-4B encoder built-in, no external embedding computation needed
- **Memory efficient**: Automatic encoder release after encoding (~8GB freed)

## Usage

### Text-to-Image

```bash
./flux -d flux-klein-model -p "A fluffy orange cat sitting on a windowsill" -o cat.png
```

### Image-to-Image

Transform an existing image based on a prompt:

```bash
./flux -d flux-klein-model -p "oil painting style" -i photo.png -o painting.png -t 0.7
```

The `-t` (strength) parameter controls how much the image changes:
- `0.0` = no change (output equals input)
- `1.0` = full generation (input only provides composition hint)
- `0.7` = good balance for style transfer

### Command Line Options

**Required:**
```
-d, --dir PATH        Path to model directory
-p, --prompt TEXT     Text prompt for generation
-o, --output PATH     Output image path (.png or .ppm)
```

**Generation options:**
```
-W, --width N         Output width in pixels (default: 256)
-H, --height N        Output height in pixels (default: 256)
-s, --steps N         Sampling steps (default: 4)
-S, --seed N          Random seed for reproducibility
```

**Image-to-image options:**
```
-i, --input PATH      Input image for img2img
-t, --strength N      How much to change the image, 0.0-1.0 (default: 0.75)
```

**Other options:**
```
-e, --embeddings PATH Load pre-computed text embeddings (advanced)
-v, --verbose         Show detailed progress
-h, --help            Show help
```

### Reproducibility

The seed is always printed to stderr, even when random:
```
$ ./flux -d flux-klein-model -p "a landscape" -o out.png
Seed: 1705612345
out.png
```

To reproduce the same image, use the printed seed:
```
$ ./flux -d flux-klein-model -p "a landscape" -o out.png -S 1705612345
```

## Building

```bash
make
```

Build options:
```bash
make clean      # Clean build artifacts
make info       # Show build configuration
make test       # Run reference image test
```

On macOS, Metal GPU acceleration is automatically enabled. On Linux, ensure OpenBLAS is installed:
```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# Fedora
sudo dnf install openblas-devel
```

## Model Download

The model weights are downloaded from HuggingFace:

```bash
pip install huggingface_hub
python download_model.py
```

This downloads approximately 16GB to `./flux-klein-model`:
- VAE (~300MB)
- Transformer (~4GB)
- Qwen3-4B Text Encoder (~8GB)
- Tokenizer

## Technical Details

### Model Architecture

**FLUX.2-klein-4B** is a rectified flow transformer optimized for fast inference:

| Component | Architecture |
|-----------|-------------|
| Transformer | 5 double blocks + 20 single blocks, 3072 hidden dim, 24 attention heads |
| VAE | AutoencoderKL, 128 latent channels, 8x spatial compression |
| Text Encoder | Qwen3-4B, 36 layers, 2560 hidden dim |

**Inference steps**: This is a distilled model that produces good results with exactly 4 sampling steps.

### Memory Requirements

| Phase | Memory |
|-------|--------|
| Text encoding | ~8GB (encoder weights) |
| Diffusion | ~8GB (transformer ~4GB + VAE ~300MB + activations) |
| Peak | ~16GB (if encoder not released) |

The text encoder is automatically released after encoding, reducing peak memory during diffusion. If you generate multiple images with different prompts, the encoder reloads automatically.

### Resolution Limits

**Maximum resolution**: 1024x1024 pixels. Higher resolutions require prohibitive memory for the attention mechanisms.

**Minimum resolution**: 64x64 pixels.

Dimensions should be multiples of 16 (the VAE downsampling factor).

## C Library API

The implementation provides a clean C API for integration into other projects:

### Basic Usage

```c
#include "flux.h"

int main() {
    // Load model
    flux_ctx *ctx = flux_load_dir("flux-klein-model");
    if (!ctx) return 1;

    // Generate image
    flux_params params = FLUX_PARAMS_DEFAULT;
    params.width = 512;
    params.height = 512;

    flux_image *img = flux_generate(ctx, "a cat sitting on a rainbow", &params);

    // Save and cleanup
    flux_image_save(img, "output.png");
    flux_image_free(img);
    flux_free(ctx);

    return 0;
}
```

### Image-to-Image

```c
flux_image *input = flux_image_load("photo.png");

flux_params params = FLUX_PARAMS_DEFAULT;
params.strength = 0.7;

flux_image *output = flux_img2img(ctx, "oil painting style", input, &params);

flux_image_free(input);
flux_image_save(output, "painting.png");
flux_image_free(output);
```

### Memory Management

```c
// Generate first image (encoder loads, then auto-releases)
flux_image *img1 = flux_generate(ctx, "prompt 1", &params);

// Generate second image (encoder reloads automatically)
flux_image *img2 = flux_generate(ctx, "prompt 2", &params);

// For batch generation with same prompt, encode once:
float *emb = flux_encode_text(ctx, "shared prompt", &seq_len);
flux_release_text_encoder(ctx);  // Free ~8GB

// Generate multiple images with same embeddings
for (int i = 0; i < 10; i++) {
    flux_set_seed(i);
    flux_image *img = flux_generate_with_embeddings(ctx, emb, seq_len, &params);
    // ... save image
    flux_image_free(img);
}
free(emb);
```

### API Reference

**Core functions:**
```c
flux_ctx *flux_load_dir(const char *model_dir);
void flux_free(flux_ctx *ctx);
void flux_release_text_encoder(flux_ctx *ctx);

flux_image *flux_generate(flux_ctx *ctx, const char *prompt, const flux_params *params);
flux_image *flux_img2img(flux_ctx *ctx, const char *prompt, const flux_image *input, const flux_params *params);
```

**Image I/O:**
```c
flux_image *flux_image_load(const char *path);      // Load PNG or PPM
int flux_image_save(const flux_image *img, const char *path);  // Save PNG or PPM
flux_image *flux_image_create(int width, int height, int channels);
flux_image *flux_image_resize(const flux_image *img, int new_width, int new_height);
void flux_image_free(flux_image *img);
```

**Utilities:**
```c
void flux_set_seed(int64_t seed);
const char *flux_model_info(flux_ctx *ctx);
const char *flux_get_error(void);
```

**Low-level functions:**
```c
float *flux_encode_text(flux_ctx *ctx, const char *prompt, int *out_seq_len);
float *flux_encode_image(flux_ctx *ctx, const flux_image *img, int *out_h, int *out_w);
flux_image *flux_decode_latent(flux_ctx *ctx, const float *latent, int latent_h, int latent_w);
```

### Parameters Structure

```c
typedef struct {
    int width;              // Output width (default: 1024)
    int height;             // Output height (default: 1024)
    int num_steps;          // Inference steps (default: 4)
    float guidance_scale;   // CFG scale (default: 1.0)
    int64_t seed;           // Random seed (-1 for random)
    float strength;         // For img2img: 0.0-1.0 (default: 0.75)
} flux_params;

#define FLUX_PARAMS_DEFAULT { 1024, 1024, 4, 1.0f, -1, 0.75f }
```

## License

MIT
