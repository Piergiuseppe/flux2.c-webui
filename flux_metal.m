/*
 * FLUX Metal Acceleration - Optimized Implementation
 *
 * Uses Metal Performance Shaders (MPS) for GPU-accelerated matrix operations.
 * Optimizations:
 * - Weight buffer caching (weights stay on GPU)
 * - Shared memory buffers (zero-copy on Apple Silicon unified memory)
 * - Buffer pooling for activations
 * - Batched command buffer execution
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "flux_metal.h"
#include <stdio.h>
#include <string.h>
#include <pthread.h>

/* Global Metal state */
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static int g_initialized = 0;

/* ========================================================================
 * Batch Execution State
 * When in batch mode, operations are encoded but not executed until
 * flux_metal_end_batch() is called.
 * ======================================================================== */

#define MAX_BATCH_OUTPUTS 256

typedef struct {
    id<MTLBuffer> buffer;
    float *cpu_ptr;
    size_t size;
} pending_output_t;

static id<MTLCommandBuffer> g_batch_cmd = nil;
static int g_in_batch = 0;
static pending_output_t g_pending_outputs[MAX_BATCH_OUTPUTS];
static int g_pending_count = 0;

/* ========================================================================
 * Weight Buffer Cache
 * Cache GPU buffers for weight matrices to avoid repeated allocations.
 * Weights are identified by their CPU pointer address.
 * ======================================================================== */

#define WEIGHT_CACHE_SIZE 512

typedef struct {
    const void *cpu_ptr;      /* CPU pointer (key) */
    id<MTLBuffer> gpu_buffer; /* Cached GPU buffer */
    size_t size;              /* Buffer size */
} weight_cache_entry_t;

static weight_cache_entry_t g_weight_cache[WEIGHT_CACHE_SIZE];
static int g_weight_cache_count = 0;
static pthread_mutex_t g_cache_mutex = PTHREAD_MUTEX_INITIALIZER;

static id<MTLBuffer> get_cached_weight_buffer(const float *weights, size_t size) {
    pthread_mutex_lock(&g_cache_mutex);

    /* Look for existing entry */
    for (int i = 0; i < g_weight_cache_count; i++) {
        if (g_weight_cache[i].cpu_ptr == weights && g_weight_cache[i].size == size) {
            id<MTLBuffer> buf = g_weight_cache[i].gpu_buffer;
            pthread_mutex_unlock(&g_cache_mutex);
            return buf;
        }
    }

    /* Not found - create new buffer */
    if (g_weight_cache_count >= WEIGHT_CACHE_SIZE) {
        /* Cache full - just create without caching */
        pthread_mutex_unlock(&g_cache_mutex);
        return [g_device newBufferWithBytes:weights
                                     length:size
                                    options:MTLResourceStorageModeShared];
    }

    /* Create and cache */
    id<MTLBuffer> buf = [g_device newBufferWithBytes:weights
                                              length:size
                                             options:MTLResourceStorageModeShared];
    g_weight_cache[g_weight_cache_count].cpu_ptr = weights;
    g_weight_cache[g_weight_cache_count].gpu_buffer = buf;
    g_weight_cache[g_weight_cache_count].size = size;
    g_weight_cache_count++;

    pthread_mutex_unlock(&g_cache_mutex);
    return buf;
}

static void clear_weight_cache(void) {
    pthread_mutex_lock(&g_cache_mutex);
    for (int i = 0; i < g_weight_cache_count; i++) {
        g_weight_cache[i].gpu_buffer = nil;
        g_weight_cache[i].cpu_ptr = NULL;
    }
    g_weight_cache_count = 0;
    pthread_mutex_unlock(&g_cache_mutex);
}

/* ========================================================================
 * Activation Buffer Pool
 * Reusable GPU buffers for activation tensors to avoid per-operation allocation.
 * Buffers use shared memory mode for zero-copy CPU access on Apple Silicon.
 * ======================================================================== */

#define ACTIVATION_POOL_SIZE 64

typedef struct {
    id<MTLBuffer> buffer;
    size_t size;
    int in_use;
} pool_buffer_t;

static pool_buffer_t g_activation_pool[ACTIVATION_POOL_SIZE];
static int g_pool_count = 0;
static pthread_mutex_t g_pool_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Get a buffer from pool, or create new one if needed */
static id<MTLBuffer> pool_get_buffer(size_t size) {
    pthread_mutex_lock(&g_pool_mutex);

    /* Look for existing buffer of sufficient size */
    for (int i = 0; i < g_pool_count; i++) {
        if (!g_activation_pool[i].in_use && g_activation_pool[i].size >= size) {
            g_activation_pool[i].in_use = 1;
            id<MTLBuffer> buf = g_activation_pool[i].buffer;
            pthread_mutex_unlock(&g_pool_mutex);
            return buf;
        }
    }

    /* No suitable buffer found - create new one */
    if (g_pool_count < ACTIVATION_POOL_SIZE) {
        /* Round up size to reduce fragmentation */
        size_t alloc_size = size;
        if (alloc_size < 1024 * 1024) {
            alloc_size = ((alloc_size + 65535) / 65536) * 65536;  /* 64KB alignment */
        } else {
            alloc_size = ((alloc_size + 1048575) / 1048576) * 1048576;  /* 1MB alignment */
        }

        id<MTLBuffer> buf = [g_device newBufferWithLength:alloc_size
                                                  options:MTLResourceStorageModeShared];
        if (buf) {
            g_activation_pool[g_pool_count].buffer = buf;
            g_activation_pool[g_pool_count].size = alloc_size;
            g_activation_pool[g_pool_count].in_use = 1;
            g_pool_count++;
            pthread_mutex_unlock(&g_pool_mutex);
            return buf;
        }
    }

    pthread_mutex_unlock(&g_pool_mutex);

    /* Pool full or allocation failed - create temporary buffer */
    return [g_device newBufferWithLength:size options:MTLResourceStorageModeShared];
}

/* Return buffer to pool (mark as available) */
static void pool_release_buffer(id<MTLBuffer> buffer) {
    if (!buffer) return;

    pthread_mutex_lock(&g_pool_mutex);
    for (int i = 0; i < g_pool_count; i++) {
        if (g_activation_pool[i].buffer == buffer) {
            g_activation_pool[i].in_use = 0;
            break;
        }
    }
    pthread_mutex_unlock(&g_pool_mutex);
}

/* Release all buffers back to pool (call at end of batch) */
static void pool_release_all(void) {
    pthread_mutex_lock(&g_pool_mutex);
    for (int i = 0; i < g_pool_count; i++) {
        g_activation_pool[i].in_use = 0;
    }
    pthread_mutex_unlock(&g_pool_mutex);
}

/* Clear the entire pool */
static void clear_activation_pool(void) {
    pthread_mutex_lock(&g_pool_mutex);
    for (int i = 0; i < g_pool_count; i++) {
        g_activation_pool[i].buffer = nil;
        g_activation_pool[i].in_use = 0;
        g_activation_pool[i].size = 0;
    }
    g_pool_count = 0;
    pthread_mutex_unlock(&g_pool_mutex);
}


/* ========================================================================
 * Metal Initialization
 * ======================================================================== */

int flux_metal_init(void) {
    if (g_initialized) return 1;

    @autoreleasepool {
        /* Get default Metal device */
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            return 0;
        }

        /* Check if this is Apple Silicon */
        if (![g_device supportsFamily:MTLGPUFamilyApple7]) {
            if (![g_device supportsFamily:MTLGPUFamilyApple6]) {
                g_device = nil;
                return 0;
            }
        }

        /* Create command queue */
        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            g_device = nil;
            return 0;
        }

        /* Initialize weight cache */
        memset(g_weight_cache, 0, sizeof(g_weight_cache));

        g_initialized = 1;
        fprintf(stderr, "Metal: GPU acceleration enabled (%s)\n",
                [[g_device name] UTF8String]);
    }

    return 1;
}

int flux_metal_available(void) {
    return g_initialized;
}

void flux_metal_cleanup(void) {
    if (!g_initialized) return;

    @autoreleasepool {
        /* End any pending batch */
        if (g_in_batch) {
            flux_metal_end_batch();
        }
        clear_weight_cache();
        clear_activation_pool();
        g_queue = nil;
        g_device = nil;
        g_initialized = 0;
    }
}

/* ========================================================================
 * Batch Execution Functions
 * ======================================================================== */

void flux_metal_begin_batch(void) {
    if (!g_initialized || g_in_batch) return;

    @autoreleasepool {
        g_batch_cmd = [g_queue commandBuffer];
        g_in_batch = 1;
        g_pending_count = 0;
    }
}

void flux_metal_end_batch(void) {
    if (!g_initialized || !g_in_batch) return;

    @autoreleasepool {
        if (g_batch_cmd) {
            [g_batch_cmd commit];
            [g_batch_cmd waitUntilCompleted];

            /* Copy all pending outputs back to CPU */
            for (int i = 0; i < g_pending_count; i++) {
                memcpy(g_pending_outputs[i].cpu_ptr,
                       [g_pending_outputs[i].buffer contents],
                       g_pending_outputs[i].size);
                /* Don't nil the buffer - it's from the pool */
            }

            g_batch_cmd = nil;
        }
        g_in_batch = 0;
        g_pending_count = 0;

        /* Release all pooled buffers back to pool */
        pool_release_all();
    }
}

int flux_metal_in_batch(void) {
    return g_in_batch;
}

/* ========================================================================
 * Optimized Matrix Multiplication
 * ======================================================================== */

void flux_metal_sgemm(int transpose_a, int transpose_b,
                      int M, int N, int K,
                      float alpha,
                      const float *A, int lda,
                      const float *B, int ldb,
                      float beta,
                      float *C, int ldc) {
    if (!g_initialized) return;

    @autoreleasepool {
        /* Compute dimensions */
        int rowsA = transpose_a ? K : M;
        int colsA = transpose_a ? M : K;
        int rowsB = transpose_b ? N : K;
        int colsB = transpose_b ? K : N;

        size_t sizeA = (size_t)rowsA * lda * sizeof(float);
        size_t sizeB = (size_t)rowsB * ldb * sizeof(float);
        size_t sizeC = (size_t)M * ldc * sizeof(float);

        /* Get or create buffers
         * - B (weights) uses cache (likely reused across calls)
         * - A (input) and C (output) use pooled buffers to avoid allocation overhead
         */
        id<MTLBuffer> bufferB = get_cached_weight_buffer(B, sizeB);

        /* Use pooled buffers for activations */
        id<MTLBuffer> bufferA = pool_get_buffer(sizeA);
        id<MTLBuffer> bufferC = pool_get_buffer(sizeC);

        if (!bufferA || !bufferB || !bufferC) {
            /* Fallback if buffer creation fails */
            if (bufferA) pool_release_buffer(bufferA);
            if (bufferC) pool_release_buffer(bufferC);
            return;
        }

        /* Copy input A to GPU buffer */
        memcpy([bufferA contents], A, sizeA);

        /* Initialize C if beta != 0 */
        if (beta != 0.0f) {
            memcpy([bufferC contents], C, sizeC);
        }

        /* Create matrix descriptors */
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:rowsA
                             columns:colsA
                            rowBytes:lda * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:rowsB
                             columns:colsB
                            rowBytes:ldb * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M
                             columns:N
                            rowBytes:ldc * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        /* Create MPS matrices */
        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

        /* Create and configure matrix multiplication */
        MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_device
               transposeLeft:transpose_a ? YES : NO
              transposeRight:transpose_b ? YES : NO
                  resultRows:M
               resultColumns:N
             interiorColumns:K
                       alpha:alpha
                        beta:beta];

        /* Use batch command buffer if in batch mode, otherwise create new one */
        id<MTLCommandBuffer> cmdBuffer = g_in_batch ? g_batch_cmd : [g_queue commandBuffer];

        [matmul encodeToCommandBuffer:cmdBuffer
                           leftMatrix:matrixA
                          rightMatrix:matrixB
                         resultMatrix:matrixC];

        if (g_in_batch) {
            /* In batch mode: defer result copy until end_batch */
            if (g_pending_count < MAX_BATCH_OUTPUTS) {
                g_pending_outputs[g_pending_count].buffer = bufferC;
                g_pending_outputs[g_pending_count].cpu_ptr = C;
                g_pending_outputs[g_pending_count].size = sizeC;
                g_pending_count++;
                /* bufferA can be released immediately after encoding */
                pool_release_buffer(bufferA);
            } else {
                /* Too many pending outputs - fall back to immediate sync */
                [cmdBuffer commit];
                [cmdBuffer waitUntilCompleted];
                memcpy(C, [bufferC contents], sizeC);
                pool_release_buffer(bufferA);
                pool_release_buffer(bufferC);
            }
        } else {
            /* Not in batch mode: execute immediately */
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
            memcpy(C, [bufferC contents], sizeC);

            /* Release pooled buffers */
            pool_release_buffer(bufferA);
            pool_release_buffer(bufferC);
        }
    }
}

void flux_metal_sgemm_batch(int transpose_a, int transpose_b,
                            int M, int N, int K,
                            float alpha,
                            const float *A, int lda, int stride_a,
                            const float *B, int ldb, int stride_b,
                            float beta,
                            float *C, int ldc, int stride_c,
                            int batch_count) {
    if (!g_initialized || batch_count <= 0) return;

    @autoreleasepool {
        /* For batched ops, encode all into single command buffer */
        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

        int rowsA = transpose_a ? K : M;
        int colsA = transpose_a ? M : K;
        int rowsB = transpose_b ? N : K;
        int colsB = transpose_b ? K : N;

        /* Create descriptors once */
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:rowsA columns:colsA
                            rowBytes:lda * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:rowsB columns:colsB
                            rowBytes:ldb * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N
                            rowBytes:ldc * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        /* Create kernel once */
        MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_device
               transposeLeft:transpose_a ? YES : NO
              transposeRight:transpose_b ? YES : NO
                  resultRows:M resultColumns:N interiorColumns:K
                       alpha:alpha beta:beta];

        size_t sizeA_elem = (size_t)rowsA * lda * sizeof(float);
        size_t sizeB_elem = (size_t)rowsB * ldb * sizeof(float);
        size_t sizeC_elem = (size_t)M * ldc * sizeof(float);

        /* Store C buffers so we can copy results back after GPU completes */
        __strong id<MTLBuffer> *cBuffers = (__strong id<MTLBuffer> *)calloc(batch_count, sizeof(id<MTLBuffer>));
        float **cPtrs = (float **)malloc(batch_count * sizeof(float *));

        for (int i = 0; i < batch_count; i++) {
            const float *Ai = A + i * stride_a;
            const float *Bi = B + i * stride_b;
            float *Ci = C + i * stride_c;

            /* Use copy-based buffers to avoid alignment issues */
            id<MTLBuffer> bufA = [g_device newBufferWithBytes:Ai
                                                       length:sizeA_elem
                                                      options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufB = get_cached_weight_buffer(Bi, sizeB_elem);
            id<MTLBuffer> bufC = [g_device newBufferWithLength:sizeC_elem
                                                       options:MTLResourceStorageModeShared];

            cBuffers[i] = bufC;
            cPtrs[i] = Ci;

            MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
            MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
            MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

            [matmul encodeToCommandBuffer:cmdBuffer
                               leftMatrix:matA
                              rightMatrix:matB
                             resultMatrix:matC];
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        /* Copy results back and release buffers */
        for (int i = 0; i < batch_count; i++) {
            memcpy(cPtrs[i], [cBuffers[i] contents], sizeC_elem);
            cBuffers[i] = nil;  /* Release under ARC */
        }

        free(cBuffers);
        free(cPtrs);
    }
}

void flux_metal_sync(void) {
    /* All operations are currently synchronous */
}

size_t flux_metal_memory_used(void) {
    if (!g_initialized || !g_device) return 0;
    return [g_device currentAllocatedSize];
}

/* External softmax function from flux_kernels.c */
extern void flux_softmax(float *x, int rows, int cols);

/*
 * GPU-accelerated attention with batched heads.
 * Does: out = softmax(Q @ K^T * scale) @ V for all heads in parallel.
 *
 * Q: [heads, seq_q, head_dim]
 * K: [heads, seq_k, head_dim]
 * V: [heads, seq_k, head_dim]
 * scores_scratch: [heads * seq_q * seq_k]
 * out: [heads, seq_q, head_dim]
 */
void flux_metal_attention(float *out,
                          const float *Q, const float *K, const float *V,
                          float *scores_scratch,
                          int heads, int seq_q, int seq_k, int head_dim,
                          float scale) {
    if (!g_initialized || heads <= 0) return;

    @autoreleasepool {
        size_t q_stride = (size_t)seq_q * head_dim;
        size_t k_stride = (size_t)seq_k * head_dim;
        size_t v_stride = (size_t)seq_k * head_dim;
        size_t scores_stride = (size_t)seq_q * seq_k;
        size_t out_stride = (size_t)seq_q * head_dim;

        size_t sizeQ = heads * q_stride * sizeof(float);
        size_t sizeK = heads * k_stride * sizeof(float);
        size_t sizeV = heads * v_stride * sizeof(float);
        size_t sizeScores = heads * scores_stride * sizeof(float);
        size_t sizeOut = heads * out_stride * sizeof(float);

        /* Create GPU buffers using shared memory (zero-copy on Apple Silicon) */
        id<MTLBuffer> bufQ = [g_device newBufferWithBytes:Q length:sizeQ
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK = [g_device newBufferWithBytes:K length:sizeK
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufV = [g_device newBufferWithBytes:V length:sizeV
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufScores = [g_device newBufferWithLength:sizeScores
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut = [g_device newBufferWithLength:sizeOut
                                                     options:MTLResourceStorageModeShared];

        if (!bufQ || !bufK || !bufV || !bufScores || !bufOut) {
            return;
        }

        /* === Phase 1: Batched Q @ K^T === */
        {
            id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

            /* Descriptors for single head matrices */
            MPSMatrixDescriptor *descQ = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_q columns:head_dim
                                rowBytes:head_dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *descK = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_k columns:head_dim
                                rowBytes:head_dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *descScores = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_q columns:seq_k
                                rowBytes:seq_k * sizeof(float)
                                dataType:MPSDataTypeFloat32];

            /* Create matmul kernel: scores = scale * Q @ K^T */
            MPSMatrixMultiplication *matmul_qk = [[MPSMatrixMultiplication alloc]
                initWithDevice:g_device
                   transposeLeft:NO
                  transposeRight:YES
                      resultRows:seq_q
                   resultColumns:seq_k
                 interiorColumns:head_dim
                           alpha:scale
                            beta:0.0f];

            /* Encode all heads */
            for (int h = 0; h < heads; h++) {
                size_t offsetQ = h * q_stride * sizeof(float);
                size_t offsetK = h * k_stride * sizeof(float);
                size_t offsetS = h * scores_stride * sizeof(float);

                MPSMatrix *matQ = [[MPSMatrix alloc]
                    initWithBuffer:bufQ offset:offsetQ descriptor:descQ];
                MPSMatrix *matK = [[MPSMatrix alloc]
                    initWithBuffer:bufK offset:offsetK descriptor:descK];
                MPSMatrix *matScores = [[MPSMatrix alloc]
                    initWithBuffer:bufScores offset:offsetS descriptor:descScores];

                [matmul_qk encodeToCommandBuffer:cmdBuffer
                                      leftMatrix:matQ
                                     rightMatrix:matK
                                    resultMatrix:matScores];
            }

            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
        }

        /* Copy scores to CPU scratch buffer */
        memcpy(scores_scratch, [bufScores contents], sizeScores);

        /* === Phase 2: Softmax on CPU (per head, per row) === */
        for (int h = 0; h < heads; h++) {
            flux_softmax(scores_scratch + h * scores_stride, seq_q, seq_k);
        }

        /* Copy softmax results back to GPU */
        memcpy([bufScores contents], scores_scratch, sizeScores);

        /* === Phase 3: Batched scores @ V === */
        {
            id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

            MPSMatrixDescriptor *descScores = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_q columns:seq_k
                                rowBytes:seq_k * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *descV = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_k columns:head_dim
                                rowBytes:head_dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_q columns:head_dim
                                rowBytes:head_dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];

            /* Create matmul kernel: out = scores @ V */
            MPSMatrixMultiplication *matmul_sv = [[MPSMatrixMultiplication alloc]
                initWithDevice:g_device
                   transposeLeft:NO
                  transposeRight:NO
                      resultRows:seq_q
                   resultColumns:head_dim
                 interiorColumns:seq_k
                           alpha:1.0f
                            beta:0.0f];

            /* Encode all heads */
            for (int h = 0; h < heads; h++) {
                size_t offsetS = h * scores_stride * sizeof(float);
                size_t offsetV = h * v_stride * sizeof(float);
                size_t offsetO = h * out_stride * sizeof(float);

                MPSMatrix *matScores = [[MPSMatrix alloc]
                    initWithBuffer:bufScores offset:offsetS descriptor:descScores];
                MPSMatrix *matV = [[MPSMatrix alloc]
                    initWithBuffer:bufV offset:offsetV descriptor:descV];
                MPSMatrix *matOut = [[MPSMatrix alloc]
                    initWithBuffer:bufOut offset:offsetO descriptor:descOut];

                [matmul_sv encodeToCommandBuffer:cmdBuffer
                                      leftMatrix:matScores
                                     rightMatrix:matV
                                    resultMatrix:matOut];
            }

            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
        }

        /* Copy output back to CPU */
        memcpy(out, [bufOut contents], sizeOut);
    }
}
