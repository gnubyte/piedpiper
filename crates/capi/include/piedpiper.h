/*
 * piedpiper.h — C API of PiedPiper, a neural lossless compressor (the TufaZip method:
 * an online-learning Transformer driving an arithmetic coder). Output is bit-identical on
 * every platform. Link with libpiedpiper (shared) or piedpiper.lib / libpiedpiper.a (static;
 * on Windows also link ws2_32, userenv, ntdll and bcrypt; on Linux pthread, dl and m).
 *
 *   pp_buffer out = {0};
 *   pp_options opt; pp_options_default(&opt);  opt.level = 3;
 *   if (pp_compress(data, len, &opt, NULL, NULL, NULL, &out) != PP_OK) { puts(pp_last_error()); }
 *   ... use out.data / out.len ...
 *   pp_buffer_free(&out);
 */
#ifndef PIEDPIPER_H
#define PIEDPIPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes. */
#define PP_OK 0
#define PP_ERROR 1        /* see pp_last_error() */
#define PP_CANCELLED 2    /* the progress callback returned 0 */

#define PP_TOKENIZER_AUTO 0
#define PP_TOKENIZER_BYTES 1
#define PP_TOKENIZER_BPE 2

/* Output buffer owned by the library; release with pp_buffer_free. */
typedef struct pp_buffer {
    uint8_t *data;
    size_t len;
} pp_buffer;

/* Advanced network shape override (used when pp_options.has_model != 0). */
typedef struct pp_model {
    uint32_t d_model, n_layer, n_head, d_ff, window, seg_len, n_streams;
} pp_model;

/* Advanced training overrides (used when pp_options.has_train != 0; start from
 * pp_preset_train() to change single fields). */
typedef struct pp_train {
    float lr, lr_min;
    uint32_t warmup, decay;
    float beta1, beta2, eps, clip, init_scale;
    uint64_t seed;
} pp_train;

typedef struct pp_options {
    uint32_t level;        /* 1..9, default 2 */
    uint32_t tokenizer;    /* PP_TOKENIZER_* */
    uint32_t bpe_vocab;    /* for PP_TOKENIZER_BPE, default 4096 */
    uint32_t threads;      /* 0 = all cores; never changes the output */
    uint32_t allow_stored; /* store incompressible data verbatim (default 1) */
    uint32_t has_model;
    pp_model model;
    uint32_t has_train;
    pp_train train;
    uint32_t builtin_prior; /* start text from the built-in prior (levels 1-3), default 1 */
} pp_options;

/* Opaque pretrained prior. */
typedef struct pp_prior pp_prior;

/* Progress: (done input bytes, total input bytes, user) -> nonzero to continue, 0 to cancel.
 * May be called from a worker thread. */
typedef int (*pp_progress_fn)(uint64_t done, uint64_t total, void *user);

void pp_options_default(pp_options *opt);
/* Fills *train with the training preset of `level`. */
void pp_preset_train(uint32_t level, pp_train *train);

int pp_compress(const uint8_t *data, size_t len, const pp_options *opt, const pp_prior *prior,
                pp_progress_fn progress, void *user, pp_buffer *out);
int pp_decompress(const uint8_t *data, size_t len, uint32_t threads, const pp_prior *prior,
                  pp_progress_fn progress, void *user, pp_buffer *out);
/* Header as a JSON object. Every pp_buffer is NUL-terminated (len excludes the NUL), so
 * out->data can be used directly as a C string. */
int pp_inspect(const uint8_t *data, size_t len, pp_buffer *out);

/* Priors: returns NULL on error (see pp_last_error). */
pp_prior *pp_prior_load(const uint8_t *data, size_t len);
uint64_t pp_prior_id(const pp_prior *prior);
void pp_prior_free(pp_prior *prior);

void pp_buffer_free(pp_buffer *buf);
/* Message of the last PP_ERROR on this thread (valid until the next call). */
const char *pp_last_error(void);
const char *pp_version(void);

#ifdef __cplusplus
}
#endif
#endif /* PIEDPIPER_H */
