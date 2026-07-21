
// We need to define _GNU_SOURCE before
// _any_ headers files are imported to get
// the usage statistics of a thread (i.e. have RUSAGE_THREAD) on GNU/Linux
// https://manpages.courier-mta.org/htmlman2/getrusage.2.html
#ifndef _GNU_SOURCE // Avoid possible double-definition warning.
#define _GNU_SOURCE
#endif

#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-const-variable"
#pragma clang diagnostic ignored "-Wparentheses"
#pragma clang diagnostic ignored "-Wunused-label"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"
#elif __GNUC__
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-const-variable"
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif

// Headers
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialisation
struct futhark_context_config;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
int futhark_context_config_set_tuning_param(struct futhark_context_config *cfg, const char *param_name, size_t new_value);
struct futhark_context;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
void futhark_context_free(struct futhark_context *ctx);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg, int flag);
void futhark_context_config_set_profiling(struct futhark_context_config *cfg, int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg, int flag);
int futhark_get_tuning_param_count(void);
const char *futhark_get_tuning_param_name(int);
const char *futhark_get_tuning_param_class(int);

// Arrays
struct futhark_f64_1d;
struct futhark_f64_1d *futhark_new_f64_1d(struct futhark_context *ctx, const double *data, int64_t dim0);
struct futhark_f64_1d *futhark_new_raw_f64_1d(struct futhark_context *ctx, unsigned char *data, int64_t dim0);
int futhark_free_f64_1d(struct futhark_context *ctx, struct futhark_f64_1d *arr);
int futhark_values_f64_1d(struct futhark_context *ctx, struct futhark_f64_1d *arr, double *data);
int futhark_index_f64_1d(struct futhark_context *ctx, double *out, struct futhark_f64_1d *arr, int64_t i0);
unsigned char *futhark_values_raw_f64_1d(struct futhark_context *ctx, struct futhark_f64_1d *arr);
const int64_t *futhark_shape_f64_1d(struct futhark_context *ctx, struct futhark_f64_1d *arr);
struct futhark_f64_2d;
struct futhark_f64_2d *futhark_new_f64_2d(struct futhark_context *ctx, const double *data, int64_t dim0, int64_t dim1);
struct futhark_f64_2d *futhark_new_raw_f64_2d(struct futhark_context *ctx, unsigned char *data, int64_t dim0, int64_t dim1);
int futhark_free_f64_2d(struct futhark_context *ctx, struct futhark_f64_2d *arr);
int futhark_values_f64_2d(struct futhark_context *ctx, struct futhark_f64_2d *arr, double *data);
int futhark_index_f64_2d(struct futhark_context *ctx, double *out, struct futhark_f64_2d *arr, int64_t i0, int64_t i1);
unsigned char *futhark_values_raw_f64_2d(struct futhark_context *ctx, struct futhark_f64_2d *arr);
const int64_t *futhark_shape_f64_2d(struct futhark_context *ctx, struct futhark_f64_2d *arr);
struct futhark_i64_1d;
struct futhark_i64_1d *futhark_new_i64_1d(struct futhark_context *ctx, const int64_t *data, int64_t dim0);
struct futhark_i64_1d *futhark_new_raw_i64_1d(struct futhark_context *ctx, unsigned char *data, int64_t dim0);
int futhark_free_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr);
int futhark_values_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr, int64_t *data);
int futhark_index_i64_1d(struct futhark_context *ctx, int64_t *out, struct futhark_i64_1d *arr, int64_t i0);
unsigned char *futhark_values_raw_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr);
const int64_t *futhark_shape_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr);

// Opaque values
struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64;
struct futhark_opaque_tup2_f64_arr1d_f64;
struct futhark_opaque_params;
int futhark_free_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj);
int futhark_store_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64(struct futhark_context *ctx, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj, void **p, size_t *n);
struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *futhark_restore_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64(struct futhark_context *ctx, const void *p);
int futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_0(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj);
int futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_1(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj);
int futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_2(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj);
int futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_3(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj);
int futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_4(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj);
int futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_5(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj);
int futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_6(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj);
int futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_7(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj);
int futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_8(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj);
int futhark_new_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_f64_2d *f_0, const struct futhark_f64_2d *f_1, const struct futhark_f64_2d *f_2, const struct futhark_f64_2d *f_3, const struct futhark_f64_2d *f_4, const struct futhark_f64_2d *f_5, const struct futhark_f64_2d *f_6, const struct futhark_f64_2d *f_7, const struct futhark_f64_2d *f_8);
int futhark_free_opaque_tup2_f64_arr1d_f64(struct futhark_context *ctx, struct futhark_opaque_tup2_f64_arr1d_f64 *obj);
int futhark_store_opaque_tup2_f64_arr1d_f64(struct futhark_context *ctx, const struct futhark_opaque_tup2_f64_arr1d_f64 *obj, void **p, size_t *n);
struct futhark_opaque_tup2_f64_arr1d_f64 *futhark_restore_opaque_tup2_f64_arr1d_f64(struct futhark_context *ctx, const void *p);
int futhark_project_opaque_tup2_f64_arr1d_f64_0(struct futhark_context *ctx, double *out, const struct futhark_opaque_tup2_f64_arr1d_f64 *obj);
int futhark_project_opaque_tup2_f64_arr1d_f64_1(struct futhark_context *ctx, struct futhark_f64_1d **out, const struct futhark_opaque_tup2_f64_arr1d_f64 *obj);
int futhark_new_opaque_tup2_f64_arr1d_f64(struct futhark_context *ctx, struct futhark_opaque_tup2_f64_arr1d_f64 **out, const double f_0, const struct futhark_f64_1d *f_1);
int futhark_free_opaque_params(struct futhark_context *ctx, struct futhark_opaque_params *obj);
int futhark_store_opaque_params(struct futhark_context *ctx, const struct futhark_opaque_params *obj, void **p, size_t *n);
struct futhark_opaque_params *futhark_restore_opaque_params(struct futhark_context *ctx, const void *p);
int futhark_project_opaque_params_wdown(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *obj);
int futhark_project_opaque_params_wkey(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *obj);
int futhark_project_opaque_params_wout(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *obj);
int futhark_project_opaque_params_wpe(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *obj);
int futhark_project_opaque_params_wqry(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *obj);
int futhark_project_opaque_params_wte(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *obj);
int futhark_project_opaque_params_wup(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *obj);
int futhark_project_opaque_params_wval(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *obj);
int futhark_project_opaque_params_wvoc(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *obj);
int futhark_new_opaque_params(struct futhark_context *ctx, struct futhark_opaque_params **out, const struct futhark_f64_2d *f_wdown, const struct futhark_f64_2d *f_wkey, const struct futhark_f64_2d *f_wout, const struct futhark_f64_2d *f_wpe, const struct futhark_f64_2d *f_wqry, const struct futhark_f64_2d *f_wte, const struct futhark_f64_2d *f_wup, const struct futhark_f64_2d *f_wval, const struct futhark_f64_2d *f_wvoc);

// Entry points
int futhark_entry_cal_loss(struct futhark_context *ctx, struct futhark_opaque_tup2_f64_arr1d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3);
int futhark_entry_forward_seq(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2);
int futhark_entry_grad_loss(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3);
int futhark_entry_make_params(struct futhark_context *ctx, struct futhark_opaque_params **out, const struct futhark_f64_2d *in0, const struct futhark_f64_2d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3, const struct futhark_f64_2d *in4, const struct futhark_f64_2d *in5, const struct futhark_f64_2d *in6, const struct futhark_f64_2d *in7, const struct futhark_f64_2d *in8);

// Miscellaneous
int futhark_context_sync(struct futhark_context *ctx);
void futhark_context_config_set_cache_file(struct futhark_context_config *cfg, const char *f);
char *futhark_context_get_error(struct futhark_context *ctx);
void futhark_context_set_logging_file(struct futhark_context *ctx, FILE *f);
void futhark_context_pause_profiling(struct futhark_context *ctx);
void futhark_context_unpause_profiling(struct futhark_context *ctx);
char *futhark_context_report(struct futhark_context *ctx);
int futhark_context_clear_caches(struct futhark_context *ctx);
#define FUTHARK_BACKEND_c
#define FUTHARK_SUCCESS 0
#define FUTHARK_PROGRAM_ERROR 2
#define FUTHARK_OUT_OF_MEMORY 3

#ifdef __cplusplus
}
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
// If NDEBUG is set, the assert() macro will do nothing. Since Futhark
// (unfortunately) makes use of assert() for error detection (and even some
// side effects), we want to avoid that.
#undef NDEBUG
#include <assert.h>
#include <stdarg.h>
#define SCALAR_FUN_ATTR static inline
// Start of util.h.
//
// Various helper functions that are useful in all generated C code.

#include <errno.h>
#include <string.h>

static const char *fut_progname = "(embedded Futhark)";

static void futhark_panic(int eval, const char *fmt, ...) __attribute__((noreturn));
static char* msgprintf(const char *s, ...);
static void* slurp_file(const char *filename, size_t *size);
static int dump_file(const char *file, const void *buf, size_t n);
struct str_builder;
static void str_builder_init(struct str_builder *b);
static void str_builder(struct str_builder *b, const char *s, ...);
static char *strclone(const char *str);

static void futhark_panic(int eval, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "%s: ", fut_progname);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  exit(eval);
}

// For generating arbitrary-sized error messages.  It is the callers
// responsibility to free the buffer at some point.
static char* msgprintf(const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = 1 + (size_t)vsnprintf(NULL, 0, s, vl);
  char *buffer = (char*) malloc(needed);
  va_start(vl, s); // Must re-init.
  vsnprintf(buffer, needed, s, vl);
  return buffer;
}

static inline void check_err(int errval, int sets_errno, const char *fun, int line,
                             const char *msg, ...) {
  if (errval) {
    char errnum[10];

    va_list vl;
    va_start(vl, msg);

    fprintf(stderr, "ERROR: ");
    vfprintf(stderr, msg, vl);
    fprintf(stderr, " in %s() at line %d with error code %s\n",
            fun, line,
            sets_errno ? strerror(errno) : errnum);
    exit(errval);
  }
}

#define CHECK_ERR(err, ...) check_err(err, 0, __func__, __LINE__, __VA_ARGS__)
#define CHECK_ERRNO(err, ...) check_err(err, 1, __func__, __LINE__, __VA_ARGS__)

// Read the rest of an open file into a NUL-terminated string; returns
// NULL on error.
static void* fslurp_file(FILE *f, size_t *size) {
  long start = ftell(f);
  fseek(f, 0, SEEK_END);
  long src_size = ftell(f)-start;
  fseek(f, start, SEEK_SET);
  unsigned char *s = (unsigned char*) malloc((size_t)src_size + 1);
  if (fread(s, 1, (size_t)src_size, f) != (size_t)src_size) {
    free(s);
    s = NULL;
  } else {
    s[src_size] = '\0';
  }

  if (size) {
    *size = (size_t)src_size;
  }

  return s;
}

// Read a file into a NUL-terminated string; returns NULL on error.
static void* slurp_file(const char *filename, size_t *size) {
  FILE *f = fopen(filename, "rb"); // To avoid Windows messing with linebreaks.
  if (f == NULL) return NULL;
  unsigned char *s = fslurp_file(f, size);
  fclose(f);
  return s;
}

// Dump 'n' bytes from 'buf' into the file at the designated location.
// Returns 0 on success.
static int dump_file(const char *file, const void *buf, size_t n) {
  FILE *f = fopen(file, "w");

  if (f == NULL) {
    return 1;
  }

  if (fwrite(buf, sizeof(char), n, f) != n) {
    return 1;
  }

  if (fclose(f) != 0) {
    return 1;
  }

  return 0;
}

struct str_builder {
  char *str;
  size_t capacity; // Size of buffer.
  size_t used; // Bytes used, *not* including final zero.
};

static void str_builder_init(struct str_builder *b) {
  b->capacity = 10;
  b->used = 0;
  b->str = malloc(b->capacity);
  b->str[0] = 0;
}

static void str_builder(struct str_builder *b, const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = (size_t)vsnprintf(NULL, 0, s, vl);

  while (b->capacity < b->used + needed + 1) {
    b->capacity *= 2;
    b->str = realloc(b->str, b->capacity);
  }

  va_start(vl, s); // Must re-init.
  vsnprintf(b->str+b->used, b->capacity-b->used, s, vl);
  b->used += needed;
}

static void str_builder_str(struct str_builder *b, const char *s) {
  size_t needed = strlen(s);
  if (b->capacity < b->used + needed + 1) {
    b->capacity *= 2;
    b->str = realloc(b->str, b->capacity);
  }
  strcpy(b->str+b->used, s);
  b->used += needed;
}

static void str_builder_char(struct str_builder *b, char c) {
  size_t needed = 1;
  if (b->capacity < b->used + needed + 1) {
    b->capacity *= 2;
    b->str = realloc(b->str, b->capacity);
  }
  b->str[b->used] = c;
  b->str[b->used+1] = 0;
  b->used += needed;
}

static void str_builder_json_str(struct str_builder* sb, const char* s) {
  str_builder_char(sb, '"');
  for (int j = 0; s[j]; j++) {
    char c = s[j];
    switch (c) {
    case '\n':
      str_builder_str(sb, "\\n");
      break;
    case '"':
      str_builder_str(sb, "\\\"");
      break;
    default:
      str_builder_char(sb, c);
    }
  }
  str_builder_char(sb, '"');
}

static char *strclone(const char *str) {
  size_t size = strlen(str) + 1;
  char *copy = (char*) malloc(size);
  if (copy == NULL) {
    return NULL;
  }

  memcpy(copy, str, size);
  return copy;
}

// Assumes NULL-terminated.
static char *strconcat(const char *src_fragments[]) {
  size_t src_len = 0;
  const char **p;

  for (p = src_fragments; *p; p++) {
    src_len += strlen(*p);
  }

  char *src = (char*) malloc(src_len + 1);
  size_t n = 0;
  for (p = src_fragments; *p; p++) {
    strcpy(src + n, *p);
    n += strlen(*p);
  }

  return src;
}

// End of util.h.
// Start of cache.h

#define CACHE_HASH_SIZE 8 // In 32-bit words.

struct cache_hash {
  uint32_t hash[CACHE_HASH_SIZE];
};

// Initialise a blank cache.
static void cache_hash_init(struct cache_hash *c);

// Hash some bytes and add them to the accumulated hash.
static void cache_hash(struct cache_hash *out, const char *in, size_t n);

// Try to restore cache contents from a file with the given name.
// Assumes the cache is invalid if it contains the given hash.
// Allocates memory and reads the cache conents, which is returned in
// *buf with size *buflen.  If the cache is successfully loaded, this
// function returns 0.  Otherwise it returns nonzero.  Errno is set if
// the failure to load the cache is due to anything except invalid
// cache conents.  Note that failing to restore the cache is not
// necessarily a problem: it might just be invalid or not created yet.
static int cache_restore(const char *fname, const struct cache_hash *hash,
                         unsigned char **buf, size_t *buflen);

// Store cache contents in the given file, with the given hash.
static int cache_store(const char *fname, const struct cache_hash *hash,
                       const unsigned char *buf, size_t buflen);

// Now for the implementation.

static void cache_hash_init(struct cache_hash *c) {
  memset(c->hash, 0, CACHE_HASH_SIZE * sizeof(uint32_t));
}

static void cache_hash(struct cache_hash *out, const char *in, size_t n) {
  // Adaptation of djb2 for larger output size by storing intermediate
  // states.
  uint32_t hash = 5381;
  for (size_t i = 0; i < n; i++) {
    hash = ((hash << 5) + hash) + in[i];
    out->hash[i % CACHE_HASH_SIZE] ^= hash;
  }
}

#define CACHE_HEADER_SIZE 8
static const char cache_header[CACHE_HEADER_SIZE] = "FUTHARK\0";

static int cache_restore(const char *fname, const struct cache_hash *hash,
                         unsigned char **buf, size_t *buflen) {
  FILE *f = fopen(fname, "rb");

  if (f == NULL) {
    return 1;
  }

  char f_header[CACHE_HEADER_SIZE];

  if (fread(f_header, sizeof(char), CACHE_HEADER_SIZE, f) != CACHE_HEADER_SIZE) {
    goto error;
  }

  if (memcmp(f_header, cache_header, CACHE_HEADER_SIZE) != 0) {
    goto error;
  }

  if (fseek(f, 0, SEEK_END) != 0) {
    goto error;
  }
  int64_t f_size = (int64_t)ftell(f);
  if (fseek(f, CACHE_HEADER_SIZE, SEEK_SET) != 0) {
    goto error;
  }

  int64_t expected_size;

  if (fread(&expected_size, sizeof(int64_t), 1, f) != 1) {
    goto error;
  }

  if (f_size != expected_size) {
    errno = 0;
    goto error;
  }

  int32_t f_hash[CACHE_HASH_SIZE];

  if (fread(f_hash, sizeof(int32_t), CACHE_HASH_SIZE, f) != CACHE_HASH_SIZE) {
    goto error;
  }

  if (memcmp(f_hash, hash->hash, CACHE_HASH_SIZE) != 0) {
    errno = 0;
    goto error;
  }

  *buflen = f_size - CACHE_HEADER_SIZE - sizeof(int64_t) - CACHE_HASH_SIZE*sizeof(int32_t);
  *buf = malloc(*buflen);
  if (fread(*buf, sizeof(char), *buflen, f) != *buflen) {
    free(*buf);
    goto error;
  }

  fclose(f);

  return 0;

 error:
  fclose(f);
  return 1;
}

static int cache_store(const char *fname, const struct cache_hash *hash,
                       const unsigned char *buf, size_t buflen) {
  FILE *f = fopen(fname, "wb");

  if (f == NULL) {
    return 1;
  }

  if (fwrite(cache_header, CACHE_HEADER_SIZE, 1, f) != 1) {
    goto error;
  }

  int64_t size = CACHE_HEADER_SIZE + sizeof(int64_t) + CACHE_HASH_SIZE*sizeof(int32_t) + buflen;

  if (fwrite(&size, sizeof(size), 1, f) != 1) {
    goto error;
  }

  if (fwrite(hash->hash, sizeof(int32_t), CACHE_HASH_SIZE, f) != CACHE_HASH_SIZE) {
    goto error;
  }

  if (fwrite(buf, sizeof(unsigned char), buflen, f) != buflen) {
    goto error;
  }

  fclose(f);

  return 0;

 error:
  fclose(f);
  return 1;
}

// End of cache.h
// Start of half.h.

// Conversion functions are from http://half.sourceforge.net/, but
// translated to C.
//
// Copyright (c) 2012-2021 Christian Rau
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef __OPENCL_VERSION__
#define __constant
#endif

__constant static const uint16_t base_table[512] = {
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080, 0x0100,
  0x0200, 0x0400, 0x0800, 0x0C00, 0x1000, 0x1400, 0x1800, 0x1C00, 0x2000, 0x2400, 0x2800, 0x2C00, 0x3000, 0x3400, 0x3800, 0x3C00,
  0x4000, 0x4400, 0x4800, 0x4C00, 0x5000, 0x5400, 0x5800, 0x5C00, 0x6000, 0x6400, 0x6800, 0x6C00, 0x7000, 0x7400, 0x7800, 0x7C00,
  0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
  0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
  0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
  0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
  0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
  0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
  0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
  0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
  0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
  0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
  0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
  0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
  0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
  0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8001, 0x8002, 0x8004, 0x8008, 0x8010, 0x8020, 0x8040, 0x8080, 0x8100,
  0x8200, 0x8400, 0x8800, 0x8C00, 0x9000, 0x9400, 0x9800, 0x9C00, 0xA000, 0xA400, 0xA800, 0xAC00, 0xB000, 0xB400, 0xB800, 0xBC00,
  0xC000, 0xC400, 0xC800, 0xCC00, 0xD000, 0xD400, 0xD800, 0xDC00, 0xE000, 0xE400, 0xE800, 0xEC00, 0xF000, 0xF400, 0xF800, 0xFC00,
  0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00,
  0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00,
  0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00,
  0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00,
  0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00,
  0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00,
  0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00 };

__constant static const unsigned char shift_table[512] = {
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
  13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 13,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
  13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 13 };

__constant static const uint32_t mantissa_table[2048] = {
  0x00000000, 0x33800000, 0x34000000, 0x34400000, 0x34800000, 0x34A00000, 0x34C00000, 0x34E00000, 0x35000000, 0x35100000, 0x35200000, 0x35300000, 0x35400000, 0x35500000, 0x35600000, 0x35700000,
  0x35800000, 0x35880000, 0x35900000, 0x35980000, 0x35A00000, 0x35A80000, 0x35B00000, 0x35B80000, 0x35C00000, 0x35C80000, 0x35D00000, 0x35D80000, 0x35E00000, 0x35E80000, 0x35F00000, 0x35F80000,
  0x36000000, 0x36040000, 0x36080000, 0x360C0000, 0x36100000, 0x36140000, 0x36180000, 0x361C0000, 0x36200000, 0x36240000, 0x36280000, 0x362C0000, 0x36300000, 0x36340000, 0x36380000, 0x363C0000,
  0x36400000, 0x36440000, 0x36480000, 0x364C0000, 0x36500000, 0x36540000, 0x36580000, 0x365C0000, 0x36600000, 0x36640000, 0x36680000, 0x366C0000, 0x36700000, 0x36740000, 0x36780000, 0x367C0000,
  0x36800000, 0x36820000, 0x36840000, 0x36860000, 0x36880000, 0x368A0000, 0x368C0000, 0x368E0000, 0x36900000, 0x36920000, 0x36940000, 0x36960000, 0x36980000, 0x369A0000, 0x369C0000, 0x369E0000,
  0x36A00000, 0x36A20000, 0x36A40000, 0x36A60000, 0x36A80000, 0x36AA0000, 0x36AC0000, 0x36AE0000, 0x36B00000, 0x36B20000, 0x36B40000, 0x36B60000, 0x36B80000, 0x36BA0000, 0x36BC0000, 0x36BE0000,
  0x36C00000, 0x36C20000, 0x36C40000, 0x36C60000, 0x36C80000, 0x36CA0000, 0x36CC0000, 0x36CE0000, 0x36D00000, 0x36D20000, 0x36D40000, 0x36D60000, 0x36D80000, 0x36DA0000, 0x36DC0000, 0x36DE0000,
  0x36E00000, 0x36E20000, 0x36E40000, 0x36E60000, 0x36E80000, 0x36EA0000, 0x36EC0000, 0x36EE0000, 0x36F00000, 0x36F20000, 0x36F40000, 0x36F60000, 0x36F80000, 0x36FA0000, 0x36FC0000, 0x36FE0000,
  0x37000000, 0x37010000, 0x37020000, 0x37030000, 0x37040000, 0x37050000, 0x37060000, 0x37070000, 0x37080000, 0x37090000, 0x370A0000, 0x370B0000, 0x370C0000, 0x370D0000, 0x370E0000, 0x370F0000,
  0x37100000, 0x37110000, 0x37120000, 0x37130000, 0x37140000, 0x37150000, 0x37160000, 0x37170000, 0x37180000, 0x37190000, 0x371A0000, 0x371B0000, 0x371C0000, 0x371D0000, 0x371E0000, 0x371F0000,
  0x37200000, 0x37210000, 0x37220000, 0x37230000, 0x37240000, 0x37250000, 0x37260000, 0x37270000, 0x37280000, 0x37290000, 0x372A0000, 0x372B0000, 0x372C0000, 0x372D0000, 0x372E0000, 0x372F0000,
  0x37300000, 0x37310000, 0x37320000, 0x37330000, 0x37340000, 0x37350000, 0x37360000, 0x37370000, 0x37380000, 0x37390000, 0x373A0000, 0x373B0000, 0x373C0000, 0x373D0000, 0x373E0000, 0x373F0000,
  0x37400000, 0x37410000, 0x37420000, 0x37430000, 0x37440000, 0x37450000, 0x37460000, 0x37470000, 0x37480000, 0x37490000, 0x374A0000, 0x374B0000, 0x374C0000, 0x374D0000, 0x374E0000, 0x374F0000,
  0x37500000, 0x37510000, 0x37520000, 0x37530000, 0x37540000, 0x37550000, 0x37560000, 0x37570000, 0x37580000, 0x37590000, 0x375A0000, 0x375B0000, 0x375C0000, 0x375D0000, 0x375E0000, 0x375F0000,
  0x37600000, 0x37610000, 0x37620000, 0x37630000, 0x37640000, 0x37650000, 0x37660000, 0x37670000, 0x37680000, 0x37690000, 0x376A0000, 0x376B0000, 0x376C0000, 0x376D0000, 0x376E0000, 0x376F0000,
  0x37700000, 0x37710000, 0x37720000, 0x37730000, 0x37740000, 0x37750000, 0x37760000, 0x37770000, 0x37780000, 0x37790000, 0x377A0000, 0x377B0000, 0x377C0000, 0x377D0000, 0x377E0000, 0x377F0000,
  0x37800000, 0x37808000, 0x37810000, 0x37818000, 0x37820000, 0x37828000, 0x37830000, 0x37838000, 0x37840000, 0x37848000, 0x37850000, 0x37858000, 0x37860000, 0x37868000, 0x37870000, 0x37878000,
  0x37880000, 0x37888000, 0x37890000, 0x37898000, 0x378A0000, 0x378A8000, 0x378B0000, 0x378B8000, 0x378C0000, 0x378C8000, 0x378D0000, 0x378D8000, 0x378E0000, 0x378E8000, 0x378F0000, 0x378F8000,
  0x37900000, 0x37908000, 0x37910000, 0x37918000, 0x37920000, 0x37928000, 0x37930000, 0x37938000, 0x37940000, 0x37948000, 0x37950000, 0x37958000, 0x37960000, 0x37968000, 0x37970000, 0x37978000,
  0x37980000, 0x37988000, 0x37990000, 0x37998000, 0x379A0000, 0x379A8000, 0x379B0000, 0x379B8000, 0x379C0000, 0x379C8000, 0x379D0000, 0x379D8000, 0x379E0000, 0x379E8000, 0x379F0000, 0x379F8000,
  0x37A00000, 0x37A08000, 0x37A10000, 0x37A18000, 0x37A20000, 0x37A28000, 0x37A30000, 0x37A38000, 0x37A40000, 0x37A48000, 0x37A50000, 0x37A58000, 0x37A60000, 0x37A68000, 0x37A70000, 0x37A78000,
  0x37A80000, 0x37A88000, 0x37A90000, 0x37A98000, 0x37AA0000, 0x37AA8000, 0x37AB0000, 0x37AB8000, 0x37AC0000, 0x37AC8000, 0x37AD0000, 0x37AD8000, 0x37AE0000, 0x37AE8000, 0x37AF0000, 0x37AF8000,
  0x37B00000, 0x37B08000, 0x37B10000, 0x37B18000, 0x37B20000, 0x37B28000, 0x37B30000, 0x37B38000, 0x37B40000, 0x37B48000, 0x37B50000, 0x37B58000, 0x37B60000, 0x37B68000, 0x37B70000, 0x37B78000,
  0x37B80000, 0x37B88000, 0x37B90000, 0x37B98000, 0x37BA0000, 0x37BA8000, 0x37BB0000, 0x37BB8000, 0x37BC0000, 0x37BC8000, 0x37BD0000, 0x37BD8000, 0x37BE0000, 0x37BE8000, 0x37BF0000, 0x37BF8000,
  0x37C00000, 0x37C08000, 0x37C10000, 0x37C18000, 0x37C20000, 0x37C28000, 0x37C30000, 0x37C38000, 0x37C40000, 0x37C48000, 0x37C50000, 0x37C58000, 0x37C60000, 0x37C68000, 0x37C70000, 0x37C78000,
  0x37C80000, 0x37C88000, 0x37C90000, 0x37C98000, 0x37CA0000, 0x37CA8000, 0x37CB0000, 0x37CB8000, 0x37CC0000, 0x37CC8000, 0x37CD0000, 0x37CD8000, 0x37CE0000, 0x37CE8000, 0x37CF0000, 0x37CF8000,
  0x37D00000, 0x37D08000, 0x37D10000, 0x37D18000, 0x37D20000, 0x37D28000, 0x37D30000, 0x37D38000, 0x37D40000, 0x37D48000, 0x37D50000, 0x37D58000, 0x37D60000, 0x37D68000, 0x37D70000, 0x37D78000,
  0x37D80000, 0x37D88000, 0x37D90000, 0x37D98000, 0x37DA0000, 0x37DA8000, 0x37DB0000, 0x37DB8000, 0x37DC0000, 0x37DC8000, 0x37DD0000, 0x37DD8000, 0x37DE0000, 0x37DE8000, 0x37DF0000, 0x37DF8000,
  0x37E00000, 0x37E08000, 0x37E10000, 0x37E18000, 0x37E20000, 0x37E28000, 0x37E30000, 0x37E38000, 0x37E40000, 0x37E48000, 0x37E50000, 0x37E58000, 0x37E60000, 0x37E68000, 0x37E70000, 0x37E78000,
  0x37E80000, 0x37E88000, 0x37E90000, 0x37E98000, 0x37EA0000, 0x37EA8000, 0x37EB0000, 0x37EB8000, 0x37EC0000, 0x37EC8000, 0x37ED0000, 0x37ED8000, 0x37EE0000, 0x37EE8000, 0x37EF0000, 0x37EF8000,
  0x37F00000, 0x37F08000, 0x37F10000, 0x37F18000, 0x37F20000, 0x37F28000, 0x37F30000, 0x37F38000, 0x37F40000, 0x37F48000, 0x37F50000, 0x37F58000, 0x37F60000, 0x37F68000, 0x37F70000, 0x37F78000,
  0x37F80000, 0x37F88000, 0x37F90000, 0x37F98000, 0x37FA0000, 0x37FA8000, 0x37FB0000, 0x37FB8000, 0x37FC0000, 0x37FC8000, 0x37FD0000, 0x37FD8000, 0x37FE0000, 0x37FE8000, 0x37FF0000, 0x37FF8000,
  0x38000000, 0x38004000, 0x38008000, 0x3800C000, 0x38010000, 0x38014000, 0x38018000, 0x3801C000, 0x38020000, 0x38024000, 0x38028000, 0x3802C000, 0x38030000, 0x38034000, 0x38038000, 0x3803C000,
  0x38040000, 0x38044000, 0x38048000, 0x3804C000, 0x38050000, 0x38054000, 0x38058000, 0x3805C000, 0x38060000, 0x38064000, 0x38068000, 0x3806C000, 0x38070000, 0x38074000, 0x38078000, 0x3807C000,
  0x38080000, 0x38084000, 0x38088000, 0x3808C000, 0x38090000, 0x38094000, 0x38098000, 0x3809C000, 0x380A0000, 0x380A4000, 0x380A8000, 0x380AC000, 0x380B0000, 0x380B4000, 0x380B8000, 0x380BC000,
  0x380C0000, 0x380C4000, 0x380C8000, 0x380CC000, 0x380D0000, 0x380D4000, 0x380D8000, 0x380DC000, 0x380E0000, 0x380E4000, 0x380E8000, 0x380EC000, 0x380F0000, 0x380F4000, 0x380F8000, 0x380FC000,
  0x38100000, 0x38104000, 0x38108000, 0x3810C000, 0x38110000, 0x38114000, 0x38118000, 0x3811C000, 0x38120000, 0x38124000, 0x38128000, 0x3812C000, 0x38130000, 0x38134000, 0x38138000, 0x3813C000,
  0x38140000, 0x38144000, 0x38148000, 0x3814C000, 0x38150000, 0x38154000, 0x38158000, 0x3815C000, 0x38160000, 0x38164000, 0x38168000, 0x3816C000, 0x38170000, 0x38174000, 0x38178000, 0x3817C000,
  0x38180000, 0x38184000, 0x38188000, 0x3818C000, 0x38190000, 0x38194000, 0x38198000, 0x3819C000, 0x381A0000, 0x381A4000, 0x381A8000, 0x381AC000, 0x381B0000, 0x381B4000, 0x381B8000, 0x381BC000,
  0x381C0000, 0x381C4000, 0x381C8000, 0x381CC000, 0x381D0000, 0x381D4000, 0x381D8000, 0x381DC000, 0x381E0000, 0x381E4000, 0x381E8000, 0x381EC000, 0x381F0000, 0x381F4000, 0x381F8000, 0x381FC000,
  0x38200000, 0x38204000, 0x38208000, 0x3820C000, 0x38210000, 0x38214000, 0x38218000, 0x3821C000, 0x38220000, 0x38224000, 0x38228000, 0x3822C000, 0x38230000, 0x38234000, 0x38238000, 0x3823C000,
  0x38240000, 0x38244000, 0x38248000, 0x3824C000, 0x38250000, 0x38254000, 0x38258000, 0x3825C000, 0x38260000, 0x38264000, 0x38268000, 0x3826C000, 0x38270000, 0x38274000, 0x38278000, 0x3827C000,
  0x38280000, 0x38284000, 0x38288000, 0x3828C000, 0x38290000, 0x38294000, 0x38298000, 0x3829C000, 0x382A0000, 0x382A4000, 0x382A8000, 0x382AC000, 0x382B0000, 0x382B4000, 0x382B8000, 0x382BC000,
  0x382C0000, 0x382C4000, 0x382C8000, 0x382CC000, 0x382D0000, 0x382D4000, 0x382D8000, 0x382DC000, 0x382E0000, 0x382E4000, 0x382E8000, 0x382EC000, 0x382F0000, 0x382F4000, 0x382F8000, 0x382FC000,
  0x38300000, 0x38304000, 0x38308000, 0x3830C000, 0x38310000, 0x38314000, 0x38318000, 0x3831C000, 0x38320000, 0x38324000, 0x38328000, 0x3832C000, 0x38330000, 0x38334000, 0x38338000, 0x3833C000,
  0x38340000, 0x38344000, 0x38348000, 0x3834C000, 0x38350000, 0x38354000, 0x38358000, 0x3835C000, 0x38360000, 0x38364000, 0x38368000, 0x3836C000, 0x38370000, 0x38374000, 0x38378000, 0x3837C000,
  0x38380000, 0x38384000, 0x38388000, 0x3838C000, 0x38390000, 0x38394000, 0x38398000, 0x3839C000, 0x383A0000, 0x383A4000, 0x383A8000, 0x383AC000, 0x383B0000, 0x383B4000, 0x383B8000, 0x383BC000,
  0x383C0000, 0x383C4000, 0x383C8000, 0x383CC000, 0x383D0000, 0x383D4000, 0x383D8000, 0x383DC000, 0x383E0000, 0x383E4000, 0x383E8000, 0x383EC000, 0x383F0000, 0x383F4000, 0x383F8000, 0x383FC000,
  0x38400000, 0x38404000, 0x38408000, 0x3840C000, 0x38410000, 0x38414000, 0x38418000, 0x3841C000, 0x38420000, 0x38424000, 0x38428000, 0x3842C000, 0x38430000, 0x38434000, 0x38438000, 0x3843C000,
  0x38440000, 0x38444000, 0x38448000, 0x3844C000, 0x38450000, 0x38454000, 0x38458000, 0x3845C000, 0x38460000, 0x38464000, 0x38468000, 0x3846C000, 0x38470000, 0x38474000, 0x38478000, 0x3847C000,
  0x38480000, 0x38484000, 0x38488000, 0x3848C000, 0x38490000, 0x38494000, 0x38498000, 0x3849C000, 0x384A0000, 0x384A4000, 0x384A8000, 0x384AC000, 0x384B0000, 0x384B4000, 0x384B8000, 0x384BC000,
  0x384C0000, 0x384C4000, 0x384C8000, 0x384CC000, 0x384D0000, 0x384D4000, 0x384D8000, 0x384DC000, 0x384E0000, 0x384E4000, 0x384E8000, 0x384EC000, 0x384F0000, 0x384F4000, 0x384F8000, 0x384FC000,
  0x38500000, 0x38504000, 0x38508000, 0x3850C000, 0x38510000, 0x38514000, 0x38518000, 0x3851C000, 0x38520000, 0x38524000, 0x38528000, 0x3852C000, 0x38530000, 0x38534000, 0x38538000, 0x3853C000,
  0x38540000, 0x38544000, 0x38548000, 0x3854C000, 0x38550000, 0x38554000, 0x38558000, 0x3855C000, 0x38560000, 0x38564000, 0x38568000, 0x3856C000, 0x38570000, 0x38574000, 0x38578000, 0x3857C000,
  0x38580000, 0x38584000, 0x38588000, 0x3858C000, 0x38590000, 0x38594000, 0x38598000, 0x3859C000, 0x385A0000, 0x385A4000, 0x385A8000, 0x385AC000, 0x385B0000, 0x385B4000, 0x385B8000, 0x385BC000,
  0x385C0000, 0x385C4000, 0x385C8000, 0x385CC000, 0x385D0000, 0x385D4000, 0x385D8000, 0x385DC000, 0x385E0000, 0x385E4000, 0x385E8000, 0x385EC000, 0x385F0000, 0x385F4000, 0x385F8000, 0x385FC000,
  0x38600000, 0x38604000, 0x38608000, 0x3860C000, 0x38610000, 0x38614000, 0x38618000, 0x3861C000, 0x38620000, 0x38624000, 0x38628000, 0x3862C000, 0x38630000, 0x38634000, 0x38638000, 0x3863C000,
  0x38640000, 0x38644000, 0x38648000, 0x3864C000, 0x38650000, 0x38654000, 0x38658000, 0x3865C000, 0x38660000, 0x38664000, 0x38668000, 0x3866C000, 0x38670000, 0x38674000, 0x38678000, 0x3867C000,
  0x38680000, 0x38684000, 0x38688000, 0x3868C000, 0x38690000, 0x38694000, 0x38698000, 0x3869C000, 0x386A0000, 0x386A4000, 0x386A8000, 0x386AC000, 0x386B0000, 0x386B4000, 0x386B8000, 0x386BC000,
  0x386C0000, 0x386C4000, 0x386C8000, 0x386CC000, 0x386D0000, 0x386D4000, 0x386D8000, 0x386DC000, 0x386E0000, 0x386E4000, 0x386E8000, 0x386EC000, 0x386F0000, 0x386F4000, 0x386F8000, 0x386FC000,
  0x38700000, 0x38704000, 0x38708000, 0x3870C000, 0x38710000, 0x38714000, 0x38718000, 0x3871C000, 0x38720000, 0x38724000, 0x38728000, 0x3872C000, 0x38730000, 0x38734000, 0x38738000, 0x3873C000,
  0x38740000, 0x38744000, 0x38748000, 0x3874C000, 0x38750000, 0x38754000, 0x38758000, 0x3875C000, 0x38760000, 0x38764000, 0x38768000, 0x3876C000, 0x38770000, 0x38774000, 0x38778000, 0x3877C000,
  0x38780000, 0x38784000, 0x38788000, 0x3878C000, 0x38790000, 0x38794000, 0x38798000, 0x3879C000, 0x387A0000, 0x387A4000, 0x387A8000, 0x387AC000, 0x387B0000, 0x387B4000, 0x387B8000, 0x387BC000,
  0x387C0000, 0x387C4000, 0x387C8000, 0x387CC000, 0x387D0000, 0x387D4000, 0x387D8000, 0x387DC000, 0x387E0000, 0x387E4000, 0x387E8000, 0x387EC000, 0x387F0000, 0x387F4000, 0x387F8000, 0x387FC000,
  0x38000000, 0x38002000, 0x38004000, 0x38006000, 0x38008000, 0x3800A000, 0x3800C000, 0x3800E000, 0x38010000, 0x38012000, 0x38014000, 0x38016000, 0x38018000, 0x3801A000, 0x3801C000, 0x3801E000,
  0x38020000, 0x38022000, 0x38024000, 0x38026000, 0x38028000, 0x3802A000, 0x3802C000, 0x3802E000, 0x38030000, 0x38032000, 0x38034000, 0x38036000, 0x38038000, 0x3803A000, 0x3803C000, 0x3803E000,
  0x38040000, 0x38042000, 0x38044000, 0x38046000, 0x38048000, 0x3804A000, 0x3804C000, 0x3804E000, 0x38050000, 0x38052000, 0x38054000, 0x38056000, 0x38058000, 0x3805A000, 0x3805C000, 0x3805E000,
  0x38060000, 0x38062000, 0x38064000, 0x38066000, 0x38068000, 0x3806A000, 0x3806C000, 0x3806E000, 0x38070000, 0x38072000, 0x38074000, 0x38076000, 0x38078000, 0x3807A000, 0x3807C000, 0x3807E000,
  0x38080000, 0x38082000, 0x38084000, 0x38086000, 0x38088000, 0x3808A000, 0x3808C000, 0x3808E000, 0x38090000, 0x38092000, 0x38094000, 0x38096000, 0x38098000, 0x3809A000, 0x3809C000, 0x3809E000,
  0x380A0000, 0x380A2000, 0x380A4000, 0x380A6000, 0x380A8000, 0x380AA000, 0x380AC000, 0x380AE000, 0x380B0000, 0x380B2000, 0x380B4000, 0x380B6000, 0x380B8000, 0x380BA000, 0x380BC000, 0x380BE000,
  0x380C0000, 0x380C2000, 0x380C4000, 0x380C6000, 0x380C8000, 0x380CA000, 0x380CC000, 0x380CE000, 0x380D0000, 0x380D2000, 0x380D4000, 0x380D6000, 0x380D8000, 0x380DA000, 0x380DC000, 0x380DE000,
  0x380E0000, 0x380E2000, 0x380E4000, 0x380E6000, 0x380E8000, 0x380EA000, 0x380EC000, 0x380EE000, 0x380F0000, 0x380F2000, 0x380F4000, 0x380F6000, 0x380F8000, 0x380FA000, 0x380FC000, 0x380FE000,
  0x38100000, 0x38102000, 0x38104000, 0x38106000, 0x38108000, 0x3810A000, 0x3810C000, 0x3810E000, 0x38110000, 0x38112000, 0x38114000, 0x38116000, 0x38118000, 0x3811A000, 0x3811C000, 0x3811E000,
  0x38120000, 0x38122000, 0x38124000, 0x38126000, 0x38128000, 0x3812A000, 0x3812C000, 0x3812E000, 0x38130000, 0x38132000, 0x38134000, 0x38136000, 0x38138000, 0x3813A000, 0x3813C000, 0x3813E000,
  0x38140000, 0x38142000, 0x38144000, 0x38146000, 0x38148000, 0x3814A000, 0x3814C000, 0x3814E000, 0x38150000, 0x38152000, 0x38154000, 0x38156000, 0x38158000, 0x3815A000, 0x3815C000, 0x3815E000,
  0x38160000, 0x38162000, 0x38164000, 0x38166000, 0x38168000, 0x3816A000, 0x3816C000, 0x3816E000, 0x38170000, 0x38172000, 0x38174000, 0x38176000, 0x38178000, 0x3817A000, 0x3817C000, 0x3817E000,
  0x38180000, 0x38182000, 0x38184000, 0x38186000, 0x38188000, 0x3818A000, 0x3818C000, 0x3818E000, 0x38190000, 0x38192000, 0x38194000, 0x38196000, 0x38198000, 0x3819A000, 0x3819C000, 0x3819E000,
  0x381A0000, 0x381A2000, 0x381A4000, 0x381A6000, 0x381A8000, 0x381AA000, 0x381AC000, 0x381AE000, 0x381B0000, 0x381B2000, 0x381B4000, 0x381B6000, 0x381B8000, 0x381BA000, 0x381BC000, 0x381BE000,
  0x381C0000, 0x381C2000, 0x381C4000, 0x381C6000, 0x381C8000, 0x381CA000, 0x381CC000, 0x381CE000, 0x381D0000, 0x381D2000, 0x381D4000, 0x381D6000, 0x381D8000, 0x381DA000, 0x381DC000, 0x381DE000,
  0x381E0000, 0x381E2000, 0x381E4000, 0x381E6000, 0x381E8000, 0x381EA000, 0x381EC000, 0x381EE000, 0x381F0000, 0x381F2000, 0x381F4000, 0x381F6000, 0x381F8000, 0x381FA000, 0x381FC000, 0x381FE000,
  0x38200000, 0x38202000, 0x38204000, 0x38206000, 0x38208000, 0x3820A000, 0x3820C000, 0x3820E000, 0x38210000, 0x38212000, 0x38214000, 0x38216000, 0x38218000, 0x3821A000, 0x3821C000, 0x3821E000,
  0x38220000, 0x38222000, 0x38224000, 0x38226000, 0x38228000, 0x3822A000, 0x3822C000, 0x3822E000, 0x38230000, 0x38232000, 0x38234000, 0x38236000, 0x38238000, 0x3823A000, 0x3823C000, 0x3823E000,
  0x38240000, 0x38242000, 0x38244000, 0x38246000, 0x38248000, 0x3824A000, 0x3824C000, 0x3824E000, 0x38250000, 0x38252000, 0x38254000, 0x38256000, 0x38258000, 0x3825A000, 0x3825C000, 0x3825E000,
  0x38260000, 0x38262000, 0x38264000, 0x38266000, 0x38268000, 0x3826A000, 0x3826C000, 0x3826E000, 0x38270000, 0x38272000, 0x38274000, 0x38276000, 0x38278000, 0x3827A000, 0x3827C000, 0x3827E000,
  0x38280000, 0x38282000, 0x38284000, 0x38286000, 0x38288000, 0x3828A000, 0x3828C000, 0x3828E000, 0x38290000, 0x38292000, 0x38294000, 0x38296000, 0x38298000, 0x3829A000, 0x3829C000, 0x3829E000,
  0x382A0000, 0x382A2000, 0x382A4000, 0x382A6000, 0x382A8000, 0x382AA000, 0x382AC000, 0x382AE000, 0x382B0000, 0x382B2000, 0x382B4000, 0x382B6000, 0x382B8000, 0x382BA000, 0x382BC000, 0x382BE000,
  0x382C0000, 0x382C2000, 0x382C4000, 0x382C6000, 0x382C8000, 0x382CA000, 0x382CC000, 0x382CE000, 0x382D0000, 0x382D2000, 0x382D4000, 0x382D6000, 0x382D8000, 0x382DA000, 0x382DC000, 0x382DE000,
  0x382E0000, 0x382E2000, 0x382E4000, 0x382E6000, 0x382E8000, 0x382EA000, 0x382EC000, 0x382EE000, 0x382F0000, 0x382F2000, 0x382F4000, 0x382F6000, 0x382F8000, 0x382FA000, 0x382FC000, 0x382FE000,
  0x38300000, 0x38302000, 0x38304000, 0x38306000, 0x38308000, 0x3830A000, 0x3830C000, 0x3830E000, 0x38310000, 0x38312000, 0x38314000, 0x38316000, 0x38318000, 0x3831A000, 0x3831C000, 0x3831E000,
  0x38320000, 0x38322000, 0x38324000, 0x38326000, 0x38328000, 0x3832A000, 0x3832C000, 0x3832E000, 0x38330000, 0x38332000, 0x38334000, 0x38336000, 0x38338000, 0x3833A000, 0x3833C000, 0x3833E000,
  0x38340000, 0x38342000, 0x38344000, 0x38346000, 0x38348000, 0x3834A000, 0x3834C000, 0x3834E000, 0x38350000, 0x38352000, 0x38354000, 0x38356000, 0x38358000, 0x3835A000, 0x3835C000, 0x3835E000,
  0x38360000, 0x38362000, 0x38364000, 0x38366000, 0x38368000, 0x3836A000, 0x3836C000, 0x3836E000, 0x38370000, 0x38372000, 0x38374000, 0x38376000, 0x38378000, 0x3837A000, 0x3837C000, 0x3837E000,
  0x38380000, 0x38382000, 0x38384000, 0x38386000, 0x38388000, 0x3838A000, 0x3838C000, 0x3838E000, 0x38390000, 0x38392000, 0x38394000, 0x38396000, 0x38398000, 0x3839A000, 0x3839C000, 0x3839E000,
  0x383A0000, 0x383A2000, 0x383A4000, 0x383A6000, 0x383A8000, 0x383AA000, 0x383AC000, 0x383AE000, 0x383B0000, 0x383B2000, 0x383B4000, 0x383B6000, 0x383B8000, 0x383BA000, 0x383BC000, 0x383BE000,
  0x383C0000, 0x383C2000, 0x383C4000, 0x383C6000, 0x383C8000, 0x383CA000, 0x383CC000, 0x383CE000, 0x383D0000, 0x383D2000, 0x383D4000, 0x383D6000, 0x383D8000, 0x383DA000, 0x383DC000, 0x383DE000,
  0x383E0000, 0x383E2000, 0x383E4000, 0x383E6000, 0x383E8000, 0x383EA000, 0x383EC000, 0x383EE000, 0x383F0000, 0x383F2000, 0x383F4000, 0x383F6000, 0x383F8000, 0x383FA000, 0x383FC000, 0x383FE000,
  0x38400000, 0x38402000, 0x38404000, 0x38406000, 0x38408000, 0x3840A000, 0x3840C000, 0x3840E000, 0x38410000, 0x38412000, 0x38414000, 0x38416000, 0x38418000, 0x3841A000, 0x3841C000, 0x3841E000,
  0x38420000, 0x38422000, 0x38424000, 0x38426000, 0x38428000, 0x3842A000, 0x3842C000, 0x3842E000, 0x38430000, 0x38432000, 0x38434000, 0x38436000, 0x38438000, 0x3843A000, 0x3843C000, 0x3843E000,
  0x38440000, 0x38442000, 0x38444000, 0x38446000, 0x38448000, 0x3844A000, 0x3844C000, 0x3844E000, 0x38450000, 0x38452000, 0x38454000, 0x38456000, 0x38458000, 0x3845A000, 0x3845C000, 0x3845E000,
  0x38460000, 0x38462000, 0x38464000, 0x38466000, 0x38468000, 0x3846A000, 0x3846C000, 0x3846E000, 0x38470000, 0x38472000, 0x38474000, 0x38476000, 0x38478000, 0x3847A000, 0x3847C000, 0x3847E000,
  0x38480000, 0x38482000, 0x38484000, 0x38486000, 0x38488000, 0x3848A000, 0x3848C000, 0x3848E000, 0x38490000, 0x38492000, 0x38494000, 0x38496000, 0x38498000, 0x3849A000, 0x3849C000, 0x3849E000,
  0x384A0000, 0x384A2000, 0x384A4000, 0x384A6000, 0x384A8000, 0x384AA000, 0x384AC000, 0x384AE000, 0x384B0000, 0x384B2000, 0x384B4000, 0x384B6000, 0x384B8000, 0x384BA000, 0x384BC000, 0x384BE000,
  0x384C0000, 0x384C2000, 0x384C4000, 0x384C6000, 0x384C8000, 0x384CA000, 0x384CC000, 0x384CE000, 0x384D0000, 0x384D2000, 0x384D4000, 0x384D6000, 0x384D8000, 0x384DA000, 0x384DC000, 0x384DE000,
  0x384E0000, 0x384E2000, 0x384E4000, 0x384E6000, 0x384E8000, 0x384EA000, 0x384EC000, 0x384EE000, 0x384F0000, 0x384F2000, 0x384F4000, 0x384F6000, 0x384F8000, 0x384FA000, 0x384FC000, 0x384FE000,
  0x38500000, 0x38502000, 0x38504000, 0x38506000, 0x38508000, 0x3850A000, 0x3850C000, 0x3850E000, 0x38510000, 0x38512000, 0x38514000, 0x38516000, 0x38518000, 0x3851A000, 0x3851C000, 0x3851E000,
  0x38520000, 0x38522000, 0x38524000, 0x38526000, 0x38528000, 0x3852A000, 0x3852C000, 0x3852E000, 0x38530000, 0x38532000, 0x38534000, 0x38536000, 0x38538000, 0x3853A000, 0x3853C000, 0x3853E000,
  0x38540000, 0x38542000, 0x38544000, 0x38546000, 0x38548000, 0x3854A000, 0x3854C000, 0x3854E000, 0x38550000, 0x38552000, 0x38554000, 0x38556000, 0x38558000, 0x3855A000, 0x3855C000, 0x3855E000,
  0x38560000, 0x38562000, 0x38564000, 0x38566000, 0x38568000, 0x3856A000, 0x3856C000, 0x3856E000, 0x38570000, 0x38572000, 0x38574000, 0x38576000, 0x38578000, 0x3857A000, 0x3857C000, 0x3857E000,
  0x38580000, 0x38582000, 0x38584000, 0x38586000, 0x38588000, 0x3858A000, 0x3858C000, 0x3858E000, 0x38590000, 0x38592000, 0x38594000, 0x38596000, 0x38598000, 0x3859A000, 0x3859C000, 0x3859E000,
  0x385A0000, 0x385A2000, 0x385A4000, 0x385A6000, 0x385A8000, 0x385AA000, 0x385AC000, 0x385AE000, 0x385B0000, 0x385B2000, 0x385B4000, 0x385B6000, 0x385B8000, 0x385BA000, 0x385BC000, 0x385BE000,
  0x385C0000, 0x385C2000, 0x385C4000, 0x385C6000, 0x385C8000, 0x385CA000, 0x385CC000, 0x385CE000, 0x385D0000, 0x385D2000, 0x385D4000, 0x385D6000, 0x385D8000, 0x385DA000, 0x385DC000, 0x385DE000,
  0x385E0000, 0x385E2000, 0x385E4000, 0x385E6000, 0x385E8000, 0x385EA000, 0x385EC000, 0x385EE000, 0x385F0000, 0x385F2000, 0x385F4000, 0x385F6000, 0x385F8000, 0x385FA000, 0x385FC000, 0x385FE000,
  0x38600000, 0x38602000, 0x38604000, 0x38606000, 0x38608000, 0x3860A000, 0x3860C000, 0x3860E000, 0x38610000, 0x38612000, 0x38614000, 0x38616000, 0x38618000, 0x3861A000, 0x3861C000, 0x3861E000,
  0x38620000, 0x38622000, 0x38624000, 0x38626000, 0x38628000, 0x3862A000, 0x3862C000, 0x3862E000, 0x38630000, 0x38632000, 0x38634000, 0x38636000, 0x38638000, 0x3863A000, 0x3863C000, 0x3863E000,
  0x38640000, 0x38642000, 0x38644000, 0x38646000, 0x38648000, 0x3864A000, 0x3864C000, 0x3864E000, 0x38650000, 0x38652000, 0x38654000, 0x38656000, 0x38658000, 0x3865A000, 0x3865C000, 0x3865E000,
  0x38660000, 0x38662000, 0x38664000, 0x38666000, 0x38668000, 0x3866A000, 0x3866C000, 0x3866E000, 0x38670000, 0x38672000, 0x38674000, 0x38676000, 0x38678000, 0x3867A000, 0x3867C000, 0x3867E000,
  0x38680000, 0x38682000, 0x38684000, 0x38686000, 0x38688000, 0x3868A000, 0x3868C000, 0x3868E000, 0x38690000, 0x38692000, 0x38694000, 0x38696000, 0x38698000, 0x3869A000, 0x3869C000, 0x3869E000,
  0x386A0000, 0x386A2000, 0x386A4000, 0x386A6000, 0x386A8000, 0x386AA000, 0x386AC000, 0x386AE000, 0x386B0000, 0x386B2000, 0x386B4000, 0x386B6000, 0x386B8000, 0x386BA000, 0x386BC000, 0x386BE000,
  0x386C0000, 0x386C2000, 0x386C4000, 0x386C6000, 0x386C8000, 0x386CA000, 0x386CC000, 0x386CE000, 0x386D0000, 0x386D2000, 0x386D4000, 0x386D6000, 0x386D8000, 0x386DA000, 0x386DC000, 0x386DE000,
  0x386E0000, 0x386E2000, 0x386E4000, 0x386E6000, 0x386E8000, 0x386EA000, 0x386EC000, 0x386EE000, 0x386F0000, 0x386F2000, 0x386F4000, 0x386F6000, 0x386F8000, 0x386FA000, 0x386FC000, 0x386FE000,
  0x38700000, 0x38702000, 0x38704000, 0x38706000, 0x38708000, 0x3870A000, 0x3870C000, 0x3870E000, 0x38710000, 0x38712000, 0x38714000, 0x38716000, 0x38718000, 0x3871A000, 0x3871C000, 0x3871E000,
  0x38720000, 0x38722000, 0x38724000, 0x38726000, 0x38728000, 0x3872A000, 0x3872C000, 0x3872E000, 0x38730000, 0x38732000, 0x38734000, 0x38736000, 0x38738000, 0x3873A000, 0x3873C000, 0x3873E000,
  0x38740000, 0x38742000, 0x38744000, 0x38746000, 0x38748000, 0x3874A000, 0x3874C000, 0x3874E000, 0x38750000, 0x38752000, 0x38754000, 0x38756000, 0x38758000, 0x3875A000, 0x3875C000, 0x3875E000,
  0x38760000, 0x38762000, 0x38764000, 0x38766000, 0x38768000, 0x3876A000, 0x3876C000, 0x3876E000, 0x38770000, 0x38772000, 0x38774000, 0x38776000, 0x38778000, 0x3877A000, 0x3877C000, 0x3877E000,
  0x38780000, 0x38782000, 0x38784000, 0x38786000, 0x38788000, 0x3878A000, 0x3878C000, 0x3878E000, 0x38790000, 0x38792000, 0x38794000, 0x38796000, 0x38798000, 0x3879A000, 0x3879C000, 0x3879E000,
  0x387A0000, 0x387A2000, 0x387A4000, 0x387A6000, 0x387A8000, 0x387AA000, 0x387AC000, 0x387AE000, 0x387B0000, 0x387B2000, 0x387B4000, 0x387B6000, 0x387B8000, 0x387BA000, 0x387BC000, 0x387BE000,
  0x387C0000, 0x387C2000, 0x387C4000, 0x387C6000, 0x387C8000, 0x387CA000, 0x387CC000, 0x387CE000, 0x387D0000, 0x387D2000, 0x387D4000, 0x387D6000, 0x387D8000, 0x387DA000, 0x387DC000, 0x387DE000,
  0x387E0000, 0x387E2000, 0x387E4000, 0x387E6000, 0x387E8000, 0x387EA000, 0x387EC000, 0x387EE000, 0x387F0000, 0x387F2000, 0x387F4000, 0x387F6000, 0x387F8000, 0x387FA000, 0x387FC000, 0x387FE000 };
__constant static const uint32_t exponent_table[64] = {
  0x00000000, 0x00800000, 0x01000000, 0x01800000, 0x02000000, 0x02800000, 0x03000000, 0x03800000, 0x04000000, 0x04800000, 0x05000000, 0x05800000, 0x06000000, 0x06800000, 0x07000000, 0x07800000,
  0x08000000, 0x08800000, 0x09000000, 0x09800000, 0x0A000000, 0x0A800000, 0x0B000000, 0x0B800000, 0x0C000000, 0x0C800000, 0x0D000000, 0x0D800000, 0x0E000000, 0x0E800000, 0x0F000000, 0x47800000,
  0x80000000, 0x80800000, 0x81000000, 0x81800000, 0x82000000, 0x82800000, 0x83000000, 0x83800000, 0x84000000, 0x84800000, 0x85000000, 0x85800000, 0x86000000, 0x86800000, 0x87000000, 0x87800000,
  0x88000000, 0x88800000, 0x89000000, 0x89800000, 0x8A000000, 0x8A800000, 0x8B000000, 0x8B800000, 0x8C000000, 0x8C800000, 0x8D000000, 0x8D800000, 0x8E000000, 0x8E800000, 0x8F000000, 0xC7800000 };
__constant static const unsigned short offset_table[64] = {
  0, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
  0, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024 };

SCALAR_FUN_ATTR uint16_t float2halfbits(float value) {
  union { float x; uint32_t y; } u;
  u.x = value;
  uint32_t bits = u.y;

  uint16_t hbits = base_table[bits>>23] + (uint16_t)((bits&0x7FFFFF)>>shift_table[bits>>23]);;

  return hbits;
}

SCALAR_FUN_ATTR float halfbits2float(uint16_t value) {
  uint32_t bits = mantissa_table[offset_table[value>>10]+(value&0x3FF)] + exponent_table[value>>10];

  union { uint32_t x; float y; } u;
  u.x = bits;
  return u.y;
}

SCALAR_FUN_ATTR uint16_t halfbitsnextafter(uint16_t from, uint16_t to) {
  int fabs = from & 0x7FFF, tabs = to & 0x7FFF;
  if(fabs > 0x7C00 || tabs > 0x7C00) {
    return ((from&0x7FFF)>0x7C00) ? (from|0x200) : (to|0x200);
  }
  if(from == to || !(fabs|tabs)) {
    return to;
  }
  if(!fabs) {
    return (to&0x8000)+1;
  }
  unsigned int out =
    from +
    (((from>>15)^(unsigned int)((from^(0x8000|(0x8000-(from>>15))))<(to^(0x8000|(0x8000-(to>>15))))))<<1)
    - 1;
  return out;
}

// End of half.h.
// Start of timing.h.

// The function get_wall_time() returns the wall time in microseconds
// (with an unspecified offset).

#ifdef _WIN32

#define NOGDI
#include <windows.h>

static int64_t get_wall_time(void) {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

static int64_t get_wall_time_ns(void) {
  return get_wall_time() * 1000;
}

#else
// Assuming POSIX

#include <time.h>
#include <sys/time.h>

static int64_t get_wall_time_ns(void) {
  struct timespec time;
  assert(clock_gettime(CLOCK_MONOTONIC, &time) == 0);
  return time.tv_sec * 1000000000 + time.tv_nsec;
}

static int64_t get_wall_time(void) {
  return get_wall_time_ns() / 1000;
}


#endif

// End of timing.h.
// Start of lock.h.

// A very simple cross-platform implementation of locks.  Uses
// pthreads on Unix and some Windows thing there.  Futhark's
// host-level code is not multithreaded, but user code may be, so we
// need some mechanism for ensuring atomic access to API functions.
// This is that mechanism.  It is not exposed to user code at all, so
// we do not have to worry about name collisions.

#ifdef _WIN32

typedef HANDLE lock_t;

static void create_lock(lock_t *lock) {
  *lock = CreateMutex(NULL,  // Default security attributes.
                      FALSE, // Initially unlocked.
                      NULL); // Unnamed.
}

static void lock_lock(lock_t *lock) {
  assert(WaitForSingleObject(*lock, INFINITE) == WAIT_OBJECT_0);
}

static void lock_unlock(lock_t *lock) {
  assert(ReleaseMutex(*lock));
}

static void free_lock(lock_t *lock) {
  CloseHandle(*lock);
}

#else
// Assuming POSIX

#include <pthread.h>

typedef pthread_mutex_t lock_t;

static void create_lock(lock_t *lock) {
  int r = pthread_mutex_init(lock, NULL);
  assert(r == 0);
}

static void lock_lock(lock_t *lock) {
  int r = pthread_mutex_lock(lock);
  assert(r == 0);
}

static void lock_unlock(lock_t *lock) {
  int r = pthread_mutex_unlock(lock);
  assert(r == 0);
}

static void free_lock(lock_t *lock) {
  // Nothing to do for pthreads.
  (void)lock;
}

#endif

// End of lock.h.
// Start of free_list.h.

typedef uintptr_t fl_mem;

// An entry in the free list.  May be invalid, to avoid having to
// deallocate entries as soon as they are removed.  There is also a
// tag, to help with memory reuse.
struct free_list_entry {
  size_t size;
  fl_mem mem;
  const char *tag;
  unsigned char valid;
};

struct free_list {
  struct free_list_entry *entries; // Pointer to entries.
  int capacity;                    // Number of entries.
  int used;                        // Number of valid entries.
  lock_t lock;                     // Thread safety.
};

static void free_list_init(struct free_list *l) {
  l->capacity = 30; // Picked arbitrarily.
  l->used = 0;
  l->entries = (struct free_list_entry*) malloc(sizeof(struct free_list_entry) * l->capacity);
  for (int i = 0; i < l->capacity; i++) {
    l->entries[i].valid = 0;
  }
  create_lock(&l->lock);
}

// Remove invalid entries from the free list.
static void free_list_pack(struct free_list *l) {
  lock_lock(&l->lock);
  int p = 0;
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[p] = l->entries[i];
      if (i > p) {
        l->entries[i].valid = 0;
      }
      p++;
    }
  }

  // Now p is the number of used elements.  We don't want it to go
  // less than the default capacity (although in practice it's OK as
  // long as it doesn't become 1).
  if (p < 30) {
    p = 30;
  }
  l->entries = realloc(l->entries, p * sizeof(struct free_list_entry));
  l->capacity = p;
  lock_unlock(&l->lock);
}

static void free_list_destroy(struct free_list *l) {
  assert(l->used == 0);
  free(l->entries);
  free_lock(&l->lock);
}

// Not part of the interface, so no locking.
static int free_list_find_invalid(struct free_list *l) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (!l->entries[i].valid) {
      break;
    }
  }
  return i;
}

static void free_list_insert(struct free_list *l, size_t size, fl_mem mem, const char *tag) {
  lock_lock(&l->lock);
  int i = free_list_find_invalid(l);

  if (i == l->capacity) {
    // List is full; so we have to grow it.
    int new_capacity = l->capacity * 2 * sizeof(struct free_list_entry);
    l->entries = realloc(l->entries, new_capacity);
    for (int j = 0; j < l->capacity; j++) {
      l->entries[j+l->capacity].valid = 0;
    }
    l->capacity *= 2;
  }

  // Now 'i' points to the first invalid entry.
  l->entries[i].valid = 1;
  l->entries[i].size = size;
  l->entries[i].mem = mem;
  l->entries[i].tag = tag;

  l->used++;
  lock_unlock(&l->lock);
}

// Determine whether this entry in the free list is acceptable for
// satisfying the request.  Not public, so no locking.
static bool free_list_acceptable(size_t size, const char* tag, struct free_list_entry *entry) {
  // We check not just the hard requirement (is the entry acceptable
  // and big enough?) but also put a cap on how much wasted space
  // (internal fragmentation) we allow.  This is necessarily a
  // heuristic, and a crude one.

  if (!entry->valid) {
    return false;
  }

  if (size > entry->size) {
    return false;
  }

  // We know the block fits.  Now the question is whether it is too
  // big.  Our policy is as follows:
  //
  // 1) We don't care about wasted space below 4096 bytes (to avoid
  // churn in tiny allocations).
  //
  // 2) If the tag matches, we allow _any_ amount of wasted space.
  //
  // 3) Otherwise we allow up to 50% wasted space.

  if (entry->size < 4096) {
    return true;
  }

  if (entry->tag == tag) {
    return true;
  }

  if (entry->size < size * 2) {
    return true;
  }

  return false;
}

// Find and remove a memory block of the indicated tag, or if that
// does not exist, another memory block with exactly the desired size.
// Returns 0 on success.
static int free_list_find(struct free_list *l, size_t size, const char *tag,
                          size_t *size_out, fl_mem *mem_out) {
  lock_lock(&l->lock);
  int size_match = -1;
  int i;
  int ret = 1;
  for (i = 0; i < l->capacity; i++) {
    if (free_list_acceptable(size, tag, &l->entries[i]) &&
        (size_match < 0 || l->entries[i].size < l->entries[size_match].size)) {
      // If this entry is valid, has sufficient size, and is smaller than the
      // best entry found so far, use this entry.
      size_match = i;
    }
  }

  if (size_match >= 0) {
    l->entries[size_match].valid = 0;
    *size_out = l->entries[size_match].size;
    *mem_out = l->entries[size_match].mem;
    l->used--;
    ret = 0;
  }
  lock_unlock(&l->lock);
  return ret;
}

// Remove the first block in the free list.  Returns 0 if a block was
// removed, and nonzero if the free list was already empty.
static int free_list_first(struct free_list *l, fl_mem *mem_out) {
  lock_lock(&l->lock);
  int ret = 1;
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[i].valid = 0;
      *mem_out = l->entries[i].mem;
      l->used--;
      ret = 0;
      break;
    }
  }
  lock_unlock(&l->lock);
  return ret;
}

// End of free_list.h.
// Start of event_list.h

typedef int (*event_report_fn)(struct str_builder*, void*);

// A collection of key-value associations. Used to associate extra data with
// events.
struct kvs {
  // A buffer that contains all value data. Must be freed when the struct kvs is
  // no longer used.
  char *buf;

  // Size of buf in bytes.
  size_t buf_size;

  // Number of bytes used in buf.
  size_t buf_used;

  // Number of associations stored.
  size_t n;

  // Capacity of vals.
  size_t vals_capacity;

  // An array of keys.
  const char* *keys;

  // Indexes into 'buf' that contains the values as zero-terminated strings.
  size_t *vals;
};

static const size_t KVS_INIT_BUF_SIZE = 128;
static const size_t KVS_INIT_NUMKEYS = 8;

void kvs_init(struct kvs* kvs) {
  kvs->buf = malloc(KVS_INIT_BUF_SIZE);
  kvs->buf_size = KVS_INIT_BUF_SIZE;
  kvs->buf_used = 0;
  kvs->vals_capacity = KVS_INIT_NUMKEYS;
  kvs->keys = calloc(kvs->vals_capacity, sizeof(const char*));
  kvs->vals = calloc(kvs->vals_capacity, sizeof(size_t));
  kvs->n = 0;
}

struct kvs* kvs_new(void) {
  struct kvs *kvs = malloc(sizeof(struct kvs));
  kvs_init(kvs);
  return kvs;
}

void kvs_printf(struct kvs* kvs, const char* key, const char* fmt, ...) {
  va_list vl;
  va_start(vl, fmt);

  size_t needed = 1 + (size_t)vsnprintf(NULL, 0, fmt, vl);

  while (kvs->buf_used+needed > kvs->buf_size) {
    kvs->buf_size *= 2;
    kvs->buf = realloc(kvs->buf, kvs->buf_size * sizeof(const char*));
  }

  if (kvs->n == kvs->vals_capacity) {
    kvs->vals_capacity *= 2;
    kvs->vals = realloc(kvs->vals, kvs->vals_capacity * sizeof(size_t));
    kvs->keys = realloc(kvs->keys, kvs->vals_capacity * sizeof(char*));
  }

  kvs->keys[kvs->n] = key;
  kvs->vals[kvs->n] = kvs->buf_used;
  kvs->buf_used += needed;

  va_start(vl, fmt); // Must re-init.
  vsnprintf(&kvs->buf[kvs->vals[kvs->n]], needed, fmt, vl);

  kvs->n++;
}

void kvs_free(struct kvs* kvs) {
  free(kvs->vals);
  free(kvs->keys);
  free(kvs->buf);
}

// Assumes all of the values are valid JSON objects.
void kvs_json(const struct kvs* kvs, struct str_builder *sb) {
  str_builder_char(sb, '{');
  for (size_t i = 0; i < kvs->n; i++) {
    if (i != 0) {
      str_builder_str(sb, ",");
    }
    str_builder_json_str(sb, kvs->keys[i]);
    str_builder_str(sb, ":");
    str_builder_str(sb, &kvs->buf[kvs->vals[i]]);
  }
  str_builder_char(sb, '}');
}

void kvs_log(const struct kvs* kvs, const char* prefix, FILE* f) {
  for (size_t i = 0; i < kvs->n; i++) {
    fprintf(f, "%s%s: %s\n",
            prefix,
            kvs->keys[i],
            &kvs->buf[kvs->vals[i]]);
  }
}

struct event {
  void* data;
  event_report_fn f;
  const char* name;
  const char *provenance;
  // Key-value information that is also to be printed.
  struct kvs *kvs;
};

struct event_list {
  struct event *events;
  int num_events;
  int capacity;
};

static void event_list_init(struct event_list *l) {
  l->capacity = 100;
  l->num_events = 0;
  l->events = calloc(l->capacity, sizeof(struct event));
}

static void event_list_free(struct event_list *l) {
  free(l->events);
}

static void add_event_to_list(struct event_list *l,
                              const char* name,
                              const char* provenance,
                              struct kvs *kvs,
                              void* data,
                              event_report_fn f) {
  if (l->num_events == l->capacity) {
    l->capacity *= 2;
    l->events = realloc(l->events, l->capacity * sizeof(struct event));
  }
  l->events[l->num_events].name = name;
  l->events[l->num_events].provenance =
    provenance ? provenance : "unknown";
  l->events[l->num_events].kvs = kvs;
  l->events[l->num_events].data = data;
  l->events[l->num_events].f = f;
  l->num_events++;
}

static int report_events_in_list(struct event_list *l,
                                 struct str_builder* sb) {
  int ret = 0;
  for (int i = 0; i < l->num_events; i++) {
    if (i != 0) {
      str_builder_str(sb, ",");
    }
    str_builder_str(sb, "{\"name\":");
    str_builder_json_str(sb, l->events[i].name);
    str_builder_str(sb, ",\"provenance\":");
    str_builder_json_str(sb, l->events[i].provenance);
    if (l->events[i].f(sb, l->events[i].data) != 0) {
      ret = 1;
      break;
    }

    str_builder_str(sb, ",\"details\":");
    if (l->events[i].kvs) {
      kvs_json(l->events[i].kvs, sb);
      kvs_free(l->events[i].kvs);
    } else {
      str_builder_str(sb, "{}");
    }

    str_builder(sb, "}");
  }
  event_list_free(l);
  event_list_init(l);
  return ret;
}

// End of event_list.h
#include <getopt.h>
#include <ctype.h>
#include <inttypes.h>
static const char *entry_point = "main";
// Start of values.h.

//// Text I/O

typedef int (*writer)(FILE*, const void*);
typedef int (*bin_reader)(void*);
typedef int (*str_reader)(char *, void*);

struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  str_reader elem_reader;
};

static void skipspaces(FILE *f) {
  int c;
  do {
    c = getc(f);
  } while (isspace(c));

  if (c != EOF) {
    ungetc(c, f);
  }
}

static int constituent(char c) {
  return isalnum(c) || c == '.' || c == '-' || c == '+' || c == '_';
}

// Produces an empty token only on EOF.
static void next_token(FILE *f, char *buf, int bufsize) {
 start:
  skipspaces(f);

  int i = 0;
  while (i < bufsize) {
    int c = getc(f);
    buf[i] = (char)c;

    if (c == EOF) {
      buf[i] = 0;
      return;
    } else if (c == '-' && i == 1 && buf[0] == '-') {
      // Line comment, so skip to end of line and start over.
      for (; c != '\n' && c != EOF; c = getc(f));
      goto start;
    } else if (!constituent((char)c)) {
      if (i == 0) {
        // We permit single-character tokens that are not
        // constituents; this lets things like ']' and ',' be
        // tokens.
        buf[i+1] = 0;
        return;
      } else {
        ungetc(c, f);
        buf[i] = 0;
        return;
      }
    }

    i++;
  }

  buf[bufsize-1] = 0;
}

static int next_token_is(FILE *f, char *buf, int bufsize, const char* expected) {
  next_token(f, buf, bufsize);
  return strcmp(buf, expected) == 0;
}

static void remove_underscores(char *buf) {
  char *w = buf;

  for (char *r = buf; *r; r++) {
    if (*r != '_') {
      *w++ = *r;
    }
  }

  *w++ = 0;
}

static int read_str_elem(char *buf, struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    (size_t)(reader->n_elems_space * reader->elem_size));
  }

  ret = reader->elem_reader(buf, reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_str_array_elems(FILE *f,
                                char *buf, int bufsize,
                                struct array_reader *reader, int64_t dims) {
  int ret = 1;
  int expect_elem = 1;
  char *knows_dimsize = (char*) calloc((size_t)dims, sizeof(char));
  int cur_dim = (int)dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc((size_t)dims, sizeof(int64_t));

  while (1) {
    next_token(f, buf, bufsize);
    if (strcmp(buf, "]") == 0) {
      expect_elem = 0;
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (!expect_elem && strcmp(buf, ",") == 0) {
      expect_elem = 1;
    } else if (expect_elem) {
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        expect_elem = 0;
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_str_empty_array(FILE *f, char *buf, int bufsize,
                                const char *type_name, int64_t *shape, int64_t dims) {
  if (strlen(buf) == 0) {
    // EOF
    return 1;
  }

  if (strcmp(buf, "empty") != 0) {
    return 1;
  }

  if (!next_token_is(f, buf, bufsize, "(")) {
    return 1;
  }

  for (int i = 0; i < dims; i++) {
    if (!next_token_is(f, buf, bufsize, "[")) {
      return 1;
    }

    next_token(f, buf, bufsize);

    if (sscanf(buf, "%"SCNu64, (uint64_t*)&shape[i]) != 1) {
      return 1;
    }

    if (!next_token_is(f, buf, bufsize, "]")) {
      return 1;
    }
  }

  if (!next_token_is(f, buf, bufsize, type_name)) {
    return 1;
  }


  if (!next_token_is(f, buf, bufsize, ")")) {
    return 1;
  }

  // Check whether the array really is empty.
  for (int i = 0; i < dims; i++) {
    if (shape[i] == 0) {
      return 0;
    }
  }

  // Not an empty array!
  return 1;
}

static int read_str_array(FILE *f,
                          int64_t elem_size, str_reader elem_reader,
                          const char *type_name,
                          void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  char buf[100];

  int dims_seen;
  for (dims_seen = 0; dims_seen < dims; dims_seen++) {
    if (!next_token_is(f, buf, sizeof(buf), "[")) {
      break;
    }
  }

  if (dims_seen == 0) {
    return read_str_empty_array(f, buf, sizeof(buf), type_name, shape, dims);
  }

  if (dims_seen != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, (size_t)(elem_size*reader.n_elems_space));
  reader.elem_reader = elem_reader;

  ret = read_str_array_elems(f, buf, sizeof(buf), &reader, dims);

  *data = reader.elems;

  return ret;
}

#define READ_STR(MACRO, PTR, SUFFIX)                                   \
  remove_underscores(buf);                                              \
  int j;                                                                \
  if (sscanf(buf, "%"MACRO"%n", (PTR*)dest, &j) == 1) {                 \
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, SUFFIX) == 0);     \
  } else {                                                              \
    return 1;                                                           \
  }

static int read_str_i8(char *buf, void* dest) {
  // Some platforms (WINDOWS) does not support scanf %hhd or its
  // cousin, %SCNi8.  Read into int first to avoid corrupting
  // memory.
  //
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(int8_t*)dest = (int8_t)x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "i8") == 0);
  } else {
    return 1;
  }
}

static int read_str_u8(char *buf, void* dest) {
  // Some platforms (WINDOWS) does not support scanf %hhd or its
  // cousin, %SCNu8.  Read into int first to avoid corrupting
  // memory.
  //
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(uint8_t*)dest = (uint8_t)x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "u8") == 0);
  } else {
    return 1;
  }
}

static int read_str_i16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "i16");
}

static int read_str_u16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "u16");
}

static int read_str_i32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "i32");
}

static int read_str_u32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "u32");
}

static int read_str_i64(char *buf, void* dest) {
  READ_STR(SCNi64, int64_t, "i64");
}

static int read_str_u64(char *buf, void* dest) {
  // FIXME: This is not correct, as SCNu64 only permits decimal
  // literals.  However, SCNi64 does not handle very large numbers
  // correctly (it's really for signed numbers, so that's fair).
  READ_STR(SCNu64, uint64_t, "u64");
}

static int read_str_f16(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f16.nan") == 0) {
    *(uint16_t*)dest = float2halfbits(NAN);
    return 0;
  } else if (strcmp(buf, "f16.inf") == 0) {
    *(uint16_t*)dest = float2halfbits(INFINITY);
    return 0;
  } else if (strcmp(buf, "-f16.inf") == 0) {
    *(uint16_t*)dest = float2halfbits(-INFINITY);
    return 0;
  } else {
    int j;
    float x;
    if (sscanf(buf, "%f%n", &x, &j) == 1) {
      if (strcmp(buf+j, "") == 0 || strcmp(buf+j, "f16") == 0) {
        *(uint16_t*)dest = float2halfbits(x);
        return 0;
      }
    }
    return 1;
  }
}

static int read_str_f32(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f32.nan") == 0) {
    *(float*)dest = (float)NAN;
    return 0;
  } else if (strcmp(buf, "f32.inf") == 0) {
    *(float*)dest = (float)INFINITY;
    return 0;
  } else if (strcmp(buf, "-f32.inf") == 0) {
    *(float*)dest = (float)-INFINITY;
    return 0;
  } else {
    READ_STR("f", float, "f32");
  }
}

static int read_str_f64(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f64.nan") == 0) {
    *(double*)dest = (double)NAN;
    return 0;
  } else if (strcmp(buf, "f64.inf") == 0) {
    *(double*)dest = (double)INFINITY;
    return 0;
  } else if (strcmp(buf, "-f64.inf") == 0) {
    *(double*)dest = (double)-INFINITY;
    return 0;
  } else {
    READ_STR("lf", double, "f64");
  }
}

static int read_str_bool(char *buf, void* dest) {
  if (strcmp(buf, "true") == 0) {
    *(char*)dest = 1;
    return 0;
  } else if (strcmp(buf, "false") == 0) {
    *(char*)dest = 0;
    return 0;
  } else {
    return 1;
  }
}

static int read_str_unit(char *buf, void* dest) {
  (void)dest;
  if (strcmp(buf, "()") == 0) {
    return 0;
  } else {
    return 1;
  }
}

static int write_str_i8(FILE *out, const int8_t *src) {
  return fprintf(out, "%hhdi8", *src);
}

static int write_str_u8(FILE *out, const uint8_t *src) {
  return fprintf(out, "%hhuu8", *src);
}

static int write_str_i16(FILE *out, const int16_t *src) {
  return fprintf(out, "%hdi16", *src);
}

static int write_str_u16(FILE *out, const uint16_t *src) {
  return fprintf(out, "%huu16", *src);
}

static int write_str_i32(FILE *out, const int32_t *src) {
  return fprintf(out, "%di32", *src);
}

static int write_str_u32(FILE *out, const uint32_t *src) {
  return fprintf(out, "%uu32", *src);
}

static int write_str_i64(FILE *out, const int64_t *src) {
  return fprintf(out, "%"PRIi64"i64", *src);
}

static int write_str_u64(FILE *out, const uint64_t *src) {
  return fprintf(out, "%"PRIu64"u64", *src);
}

// FLT_DECIMAL_DIG and DBL_DECIMAL_DIG are defined in C11.
// If we want C99 compatibility, we must define them ourselves.
// We choose the standard values on platforms that use the IEEE754 defaults, with fallback to an overestimate.
#ifndef FLT_DECIMAL_DIG
  #if FLT_RADIX == 2 && FLT_MANT_DIG <= 24 && 9 < DECIMAL_DIG
    #define FLT_DECIMAL_DIG 9
  #else
    #define FLT_DECIMAL_DIG DECIMAL_DIG
  #endif
#endif
#ifndef DBL_DECIMAL_DIG
  #if FLT_RADIX == 2 && DBL_MANT_DIG <= 53 && 17 < DECIMAL_DIG
    #define DBL_DECIMAL_DIG 17
  #else
    #define DBL_DECIMAL_DIG DECIMAL_DIG
  #endif
#endif

static int write_str_f16(FILE *out, const uint16_t *src) {
  float x = halfbits2float(*src);
  if (isnan(x)) {
    return fprintf(out, "f16.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f16.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f16.inf");
  } else {
    return fprintf(out, "%.*gf16", FLT_DECIMAL_DIG, x);
  }
}

static int write_str_f32(FILE *out, const float *src) {
  float x = *src;
  if (isnan(x)) {
    return fprintf(out, "f32.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f32.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f32.inf");
  } else {
    return fprintf(out, "%.*gf32", FLT_DECIMAL_DIG, x);
  }
}

static int write_str_f64(FILE *out, const double *src) {
  double x = *src;
  if (isnan(x)) {
    return fprintf(out, "f64.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f64.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f64.inf");
  } else {
    return fprintf(out, "%.*gf64", DBL_DECIMAL_DIG, x);
  }
}

static int write_str_bool(FILE *out, const void *src) {
  return fprintf(out, *(char*)src ? "true" : "false");
}

static int write_str_unit(FILE *out, const void *src) {
  (void)src;
  return fprintf(out, "()");
}

//// Binary I/O

#define BINARY_FORMAT_VERSION 2
#define IS_BIG_ENDIAN (!*(unsigned char *)&(uint16_t){1})

static void flip_bytes(size_t elem_size, unsigned char *elem) {
  for (size_t j=0; j<elem_size/2; j++) {
    unsigned char head = elem[j];
    size_t tail_index = elem_size-1-j;
    elem[j] = elem[tail_index];
    elem[tail_index] = head;
  }
}

// On Windows we need to explicitly set the file mode to not mangle
// newline characters.  On *nix there is no difference.
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
static void set_binary_mode(FILE *f) {
  setmode(fileno(f), O_BINARY);
}
#else
static void set_binary_mode(FILE *f) {
  (void)f;
}
#endif

static int read_byte(FILE *f, void* dest) {
  size_t num_elems_read = fread(dest, 1, 1, f);
  return num_elems_read == 1 ? 0 : 1;
}

//// Types

struct primtype_info_t {
  const char binname[4]; // Used for parsing binary data.
  const char* type_name; // Same name as in Futhark.
  const int64_t size; // in bytes
  const writer write_str; // Write in text format.
  const str_reader read_str; // Read in text format.
};

static const struct primtype_info_t i8_info =
  {.binname = "  i8", .type_name = "i8",   .size = 1,
   .write_str = (writer)write_str_i8, .read_str = (str_reader)read_str_i8};
static const struct primtype_info_t i16_info =
  {.binname = " i16", .type_name = "i16",  .size = 2,
   .write_str = (writer)write_str_i16, .read_str = (str_reader)read_str_i16};
static const struct primtype_info_t i32_info =
  {.binname = " i32", .type_name = "i32",  .size = 4,
   .write_str = (writer)write_str_i32, .read_str = (str_reader)read_str_i32};
static const struct primtype_info_t i64_info =
  {.binname = " i64", .type_name = "i64",  .size = 8,
   .write_str = (writer)write_str_i64, .read_str = (str_reader)read_str_i64};
static const struct primtype_info_t u8_info =
  {.binname = "  u8", .type_name = "u8",   .size = 1,
   .write_str = (writer)write_str_u8, .read_str = (str_reader)read_str_u8};
static const struct primtype_info_t u16_info =
  {.binname = " u16", .type_name = "u16",  .size = 2,
   .write_str = (writer)write_str_u16, .read_str = (str_reader)read_str_u16};
static const struct primtype_info_t u32_info =
  {.binname = " u32", .type_name = "u32",  .size = 4,
   .write_str = (writer)write_str_u32, .read_str = (str_reader)read_str_u32};
static const struct primtype_info_t u64_info =
  {.binname = " u64", .type_name = "u64",  .size = 8,
   .write_str = (writer)write_str_u64, .read_str = (str_reader)read_str_u64};
static const struct primtype_info_t f16_info =
  {.binname = " f16", .type_name = "f16",  .size = 2,
   .write_str = (writer)write_str_f16, .read_str = (str_reader)read_str_f16};
static const struct primtype_info_t f32_info =
  {.binname = " f32", .type_name = "f32",  .size = 4,
   .write_str = (writer)write_str_f32, .read_str = (str_reader)read_str_f32};
static const struct primtype_info_t f64_info =
  {.binname = " f64", .type_name = "f64",  .size = 8,
   .write_str = (writer)write_str_f64, .read_str = (str_reader)read_str_f64};
static const struct primtype_info_t bool_info =
  {.binname = "bool", .type_name = "bool", .size = 1,
   .write_str = (writer)write_str_bool, .read_str = (str_reader)read_str_bool};
static const struct primtype_info_t unit_info =
  {.binname = "bool", .type_name = "unit",   .size = 1,
   .write_str = (writer)write_str_unit, .read_str = (str_reader)read_str_unit};

static const struct primtype_info_t* primtypes[] = {
  &i8_info, &i16_info, &i32_info, &i64_info,
  &u8_info, &u16_info, &u32_info, &u64_info,
  &f16_info, &f32_info, &f64_info,
  &bool_info,
  NULL // NULL-terminated
};

// General value interface.  All endian business taken care of at
// lower layers.

static int read_is_binary(FILE *f) {
  skipspaces(f);
  int c = getc(f);
  if (c == 'b') {
    int8_t bin_version;
    int ret = read_byte(f, &bin_version);

    if (ret != 0) { futhark_panic(1, "binary-input: could not read version.\n"); }

    if (bin_version != BINARY_FORMAT_VERSION) {
      futhark_panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
            bin_version, BINARY_FORMAT_VERSION);
    }

    return 1;
  }
  ungetc(c, f);
  return 0;
}

static const struct primtype_info_t* read_bin_read_type_enum(FILE *f) {
  char read_binname[4];

  int num_matched = fscanf(f, "%4c", read_binname);
  if (num_matched != 1) { futhark_panic(1, "binary-input: Couldn't read element type.\n"); }

  const struct primtype_info_t **type = primtypes;

  for (; *type != NULL; type++) {
    // I compare the 4 characters manually instead of using strncmp because
    // this allows any value to be used, also NULL bytes
    if (memcmp(read_binname, (*type)->binname, 4) == 0) {
      return *type;
    }
  }
  futhark_panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname);
  return NULL;
}

static void read_bin_ensure_scalar(FILE *f, const struct primtype_info_t *expected_type) {
  int8_t bin_dims;
  int ret = read_byte(f, &bin_dims);
  if (ret != 0) { futhark_panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != 0) {
    futhark_panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n",
          bin_dims);
  }

  const struct primtype_info_t *bin_type = read_bin_read_type_enum(f);
  if (bin_type != expected_type) {
    futhark_panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
          expected_type->type_name,
          bin_type->type_name);
  }
}

//// High-level interface

static int read_bin_array(FILE *f,
                          const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  int ret;

  int8_t bin_dims;
  ret = read_byte(f, &bin_dims);
  if (ret != 0) { futhark_panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != dims) {
    futhark_panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
          dims, bin_dims);
  }

  const struct primtype_info_t *bin_primtype = read_bin_read_type_enum(f);
  if (expected_type != bin_primtype) {
    futhark_panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
          dims, expected_type->type_name, dims, bin_primtype->type_name);
  }

  int64_t elem_count = 1;
  for (int i=0; i<dims; i++) {
    int64_t bin_shape;
    ret = (int)fread(&bin_shape, sizeof(bin_shape), 1, f);
    if (ret != 1) {
      futhark_panic(1, "binary-input: Couldn't read size for dimension %i of array.\n", i);
    }
    if (IS_BIG_ENDIAN) {
      flip_bytes(sizeof(bin_shape), (unsigned char*) &bin_shape);
    }
    elem_count *= bin_shape;
    shape[i] = bin_shape;
  }

  int64_t elem_size = expected_type->size;
  void* tmp = realloc(*data, (size_t)(elem_count * elem_size));
  if (tmp == NULL) {
    futhark_panic(1, "binary-input: Failed to allocate array of size %i.\n",
          elem_count * elem_size);
  }
  *data = tmp;

  int64_t num_elems_read = (int64_t)fread(*data, (size_t)elem_size, (size_t)elem_count, f);
  if (num_elems_read != elem_count) {
    futhark_panic(1, "binary-input: tried to read %i elements of an array, but only got %i elements.\n",
          elem_count, num_elems_read);
  }

  // If we're on big endian platform we must change all multibyte elements
  // from using little endian to big endian
  if (IS_BIG_ENDIAN && elem_size != 1) {
    flip_bytes((size_t)elem_size, (unsigned char*) *data);
  }

  return 0;
}

static int read_array(FILE *f, const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  if (!read_is_binary(f)) {
    return read_str_array(f, expected_type->size, (str_reader)expected_type->read_str, expected_type->type_name, data, shape, dims);
  } else {
    return read_bin_array(f, expected_type, data, shape, dims);
  }
}

static int end_of_input(FILE *f) {
  skipspaces(f);
  char token[2];
  next_token(f, token, sizeof(token));
  if (strcmp(token, "") == 0) {
    return 0;
  } else {
    return 1;
  }
}

static int write_str_array(FILE *out,
                           const struct primtype_info_t *elem_type,
                           const unsigned char *data,
                           const int64_t *shape,
                           int8_t rank) {
  if (rank==0) {
    elem_type->write_str(out, (const void*)data);
  } else {
    int64_t len = (int64_t)shape[0];
    int64_t slice_size = 1;

    int64_t elem_size = elem_type->size;
    for (int8_t i = 1; i < rank; i++) {
      slice_size *= shape[i];
    }

    if (len*slice_size == 0) {
      fprintf(out, "empty(");
      for (int64_t i = 0; i < rank; i++) {
        fprintf(out, "[%"PRIi64"]", shape[i]);
      }
      fprintf(out, "%s", elem_type->type_name);
      fprintf(out, ")");
    } else if (rank==1) {
      fputc('[', out);
      for (int64_t i = 0; i < len; i++) {
        elem_type->write_str(out, (const void*) (data + i * elem_size));
        if (i != len-1) {
          fprintf(out, ", ");
        }
      }
      fputc(']', out);
    } else {
      fputc('[', out);
      for (int64_t i = 0; i < len; i++) {
        write_str_array(out, elem_type, data + i * slice_size * elem_size, shape+1, rank-1);
        if (i != len-1) {
          fprintf(out, ", ");
        }
      }
      fputc(']', out);
    }
  }
  return 0;
}

static int write_bin_array(FILE *out,
                           const struct primtype_info_t *elem_type,
                           const unsigned char *data,
                           const int64_t *shape,
                           int8_t rank) {
  int64_t num_elems = 1;
  for (int64_t i = 0; i < rank; i++) {
    num_elems *= shape[i];
  }

  fputc('b', out);
  fputc((char)BINARY_FORMAT_VERSION, out);
  fwrite(&rank, sizeof(int8_t), 1, out);
  fwrite(elem_type->binname, 4, 1, out);
  if (shape != NULL) {
    fwrite(shape, sizeof(int64_t), (size_t)rank, out);
  }

  if (IS_BIG_ENDIAN) {
    for (int64_t i = 0; i < num_elems; i++) {
      const unsigned char *elem = data+i*elem_type->size;
      for (int64_t j = 0; j < elem_type->size; j++) {
        fwrite(&elem[elem_type->size-j], 1, 1, out);
      }
    }
  } else {
    fwrite(data, (size_t)elem_type->size, (size_t)num_elems, out);
  }

  return 0;
}

static int write_array(FILE *out, int write_binary,
                       const struct primtype_info_t *elem_type,
                       const void *data,
                       const int64_t *shape,
                       const int8_t rank) {
  if (write_binary) {
    return write_bin_array(out, elem_type, data, shape, rank);
  } else {
    return write_str_array(out, elem_type, data, shape, rank);
  }
}

static int read_scalar(FILE *f,
                       const struct primtype_info_t *expected_type, void *dest) {
  if (!read_is_binary(f)) {
    char buf[100];
    next_token(f, buf, sizeof(buf));
    return expected_type->read_str(buf, dest);
  } else {
    read_bin_ensure_scalar(f, expected_type);
    size_t elem_size = (size_t)expected_type->size;
    size_t num_elems_read = fread(dest, elem_size, 1, f);
    if (IS_BIG_ENDIAN) {
      flip_bytes(elem_size, (unsigned char*) dest);
    }
    return num_elems_read == 1 ? 0 : 1;
  }
}

static int write_scalar(FILE *out, int write_binary, const struct primtype_info_t *type, void *src) {
  if (write_binary) {
    return write_bin_array(out, type, src, NULL, 0);
  } else {
    return type->write_str(out, src);
  }
}

// End of values.h.

// Start of server.h.

// Forward declarations of things that we technically don't know until
// the application header file is included, but which we need.
struct futhark_context_config;
struct futhark_context;
char *futhark_context_get_error(struct futhark_context *ctx);
int futhark_context_sync(struct futhark_context *ctx);
int futhark_context_clear_caches(struct futhark_context *ctx);
int futhark_context_config_set_tuning_param(struct futhark_context_config *cfg,
                                            const char *param_name,
                                            size_t new_value);
int futhark_get_tuning_param_count(void);
const char* futhark_get_tuning_param_name(int i);
const char* futhark_get_tuning_param_class(int i);

typedef int (*restore_fn)(const void*, FILE*, struct futhark_context*, void*);
typedef void (*store_fn)(const void*, FILE*, struct futhark_context*, void*);
typedef int (*free_fn)(const void*, struct futhark_context*, void*);
typedef int (*array_new_fn)(struct futhark_context *, void**, const void*, const int64_t*);
typedef int (*array_set_fn)(struct futhark_context *, const void*, const void*, const int64_t*);
typedef const int64_t* (*array_shape_fn)(struct futhark_context*, const void*);
typedef int (*array_index_fn)(struct futhark_context*, void*, const void*, const int64_t*);
typedef int (*project_fn)(struct futhark_context*, void*, const void*);
typedef int (*variant_fn)(struct futhark_context*, const void*);
typedef int (*new_fn)(struct futhark_context*, void**, const void*[]);
typedef int (*destruct_fn)(struct futhark_context*, const void*[], const void*);

enum kind {
  PRIMITIVE,
  ARRAY,
  RECORD,
  SUM,
  OPAQUE
};

struct array {
  int rank;
  const struct type *element_type;
  array_new_fn new;
  array_set_fn set;
  array_shape_fn shape;
  array_index_fn index;
};

struct field {
  const char *name;
  const struct type *type;
  project_fn project;
};

struct record {
  int num_fields;
  const struct field* fields;
  new_fn new;
};

struct variant {
  const char *name;
  int num_types;
  const struct type **types;
  new_fn new;
  destruct_fn destruct;
};

struct sum {
  int num_variants;
  const struct variant *variants;
  variant_fn variant;
};

struct type {
  const char *name;
  restore_fn restore;
  store_fn store;
  free_fn free;
  const void *aux;
  const enum kind kind;
  const void *info;
};

int free_scalar(const void *aux, struct futhark_context *ctx, void *p) {
  (void)aux;
  (void)ctx;
  (void)p;
  // Nothing to do.
  return 0;
}

#define DEF_SCALAR_TYPE(T)                                      \
  int restore_##T(const void *aux, FILE *f,                     \
                  struct futhark_context *ctx, void *p) {       \
    (void)aux;                                                  \
    (void)ctx;                                                  \
    return read_scalar(f, &T##_info, p);                        \
  }                                                             \
                                                                \
  void store_##T(const void *aux, FILE *f,                      \
                 struct futhark_context *ctx, void *p) {        \
    (void)aux;                                                  \
    (void)ctx;                                                  \
    write_scalar(f, 1, &T##_info, p);                           \
  }                                                             \
                                                                \
  struct type type_##T =                                        \
    { .name = #T,                                               \
      .restore = restore_##T,                                   \
      .store = store_##T,                                       \
      .free = free_scalar                                       \
    }                                                           \

DEF_SCALAR_TYPE(i8);
DEF_SCALAR_TYPE(i16);
DEF_SCALAR_TYPE(i32);
DEF_SCALAR_TYPE(i64);
DEF_SCALAR_TYPE(u8);
DEF_SCALAR_TYPE(u16);
DEF_SCALAR_TYPE(u32);
DEF_SCALAR_TYPE(u64);
DEF_SCALAR_TYPE(f16);
DEF_SCALAR_TYPE(f32);
DEF_SCALAR_TYPE(f64);
DEF_SCALAR_TYPE(bool);

struct value {
  const struct type *type;
  union {
    void *v_ptr;
    int8_t  v_i8;
    int16_t v_i16;
    int32_t v_i32;
    int64_t v_i64;

    uint8_t  v_u8;
    uint16_t v_u16;
    uint32_t v_u32;
    uint64_t v_u64;

    uint16_t v_f16;
    float v_f32;
    double v_f64;

    bool v_bool;
  } value;
};

void* value_ptr(struct value *v) {
  if (v->type == &type_i8) {
    return &v->value.v_i8;
  }
  if (v->type == &type_i16) {
    return &v->value.v_i16;
  }
  if (v->type == &type_i32) {
    return &v->value.v_i32;
  }
  if (v->type == &type_i64) {
    return &v->value.v_i64;
  }
  if (v->type == &type_u8) {
    return &v->value.v_u8;
  }
  if (v->type == &type_u16) {
    return &v->value.v_u16;
  }
  if (v->type == &type_u32) {
    return &v->value.v_u32;
  }
  if (v->type == &type_u64) {
    return &v->value.v_u64;
  }
  if (v->type == &type_f16) {
    return &v->value.v_f16;
  }
  if (v->type == &type_f32) {
    return &v->value.v_f32;
  }
  if (v->type == &type_f64) {
    return &v->value.v_f64;
  }
  if (v->type == &type_bool) {
    return &v->value.v_bool;
  }
  return &v->value.v_ptr;
}

struct variable {
  // NULL name indicates free slot.  Name is owned by this struct.
  char *name;
  struct value value;
};

typedef int (*entry_point_fn)(struct futhark_context*, void*, void**);

struct entry_point {
  const char *name;
  entry_point_fn f;
  const char** tuning_params;
  const char** attrs;
  const struct type *out_type;
  bool out_unique;
  const struct type **in_types;
  bool *in_unique;
};

int entry_num_ins(struct entry_point *e) {
  int count = 0;
  while (e->in_types[count]) {
    count++;
  }
  return count;
}

struct futhark_prog {
  // Last entry point identified by NULL name.
  struct entry_point *entry_points;
  // Last type identified by NULL name.
  const struct type **types;
};

struct server_state {
  struct futhark_prog prog;
  struct futhark_context_config *cfg;
  struct futhark_context *ctx;
  int variables_capacity;
  struct variable *variables;
};

struct variable* get_variable(struct server_state *s,
                              const char *name) {
  for (int i = 0; i < s->variables_capacity; i++) {
    if (s->variables[i].name != NULL &&
        strcmp(s->variables[i].name, name) == 0) {
      return &s->variables[i];
    }
  }

  return NULL;
}

struct variable* create_variable(struct server_state *s,
                                 const char *name,
                                 const struct type *type) {
  int found = -1;
  for (int i = 0; i < s->variables_capacity; i++) {
    if (found == -1 && s->variables[i].name == NULL) {
      found = i;
    } else if (s->variables[i].name != NULL &&
               strcmp(s->variables[i].name, name) == 0) {
      return NULL;
    }
  }

  if (found != -1) {
    // Found a free spot.
    s->variables[found].name = strdup(name);
    s->variables[found].value.type = type;
    return &s->variables[found];
  }

  // Need to grow the buffer.
  found = s->variables_capacity;
  s->variables_capacity *= 2;
  s->variables = realloc(s->variables,
                         s->variables_capacity * sizeof(struct variable));

  s->variables[found].name = strdup(name);
  s->variables[found].value.type = type;

  for (int i = found+1; i < s->variables_capacity; i++) {
    s->variables[i].name = NULL;
  }

  return &s->variables[found];
}

void drop_variable(struct variable *v) {
  free(v->name);
  v->name = NULL;
}

int arg_exists(const char *args[], int i) {
  return args[i] != NULL;
}

const char* get_arg(const char *args[], int i) {
  if (!arg_exists(args, i)) {
    futhark_panic(1, "Insufficient command args.\n");
  }
  return args[i];
}

const struct type* get_type(struct server_state *s, const char *name) {
  for (int i = 0; s->prog.types[i]; i++) {
    if (strcmp(s->prog.types[i]->name, name) == 0) {
      return s->prog.types[i];
    }
  }

  futhark_panic(1, "Unknown type %s\n", name);
  return NULL;
}

struct entry_point* get_entry_point(struct server_state *s, const char *name) {
  for (int i = 0; s->prog.entry_points[i].name; i++) {
    if (strcmp(s->prog.entry_points[i].name, name) == 0) {
      return &s->prog.entry_points[i];
    }
  }

  return NULL;
}

// Print the command-done marker, indicating that we are ready for
// more input.
void ok(void) {
  printf("%%%%%% OK\n");
  fflush(stdout);
}

// Print the failure marker.  Output is now an error message until the
// next ok().
void failure(void) {
  printf("%%%%%% FAILURE\n");
}

void error_check(struct server_state *s, int err) {
  if (err != 0) {
    failure();
    char *error = futhark_context_get_error(s->ctx);
    if (error != NULL) {
      puts(error);
    }
    free(error);
  }
}

void cmd_call(struct server_state *s, const char *args[]) {
  const char *name = get_arg(args, 0);

  struct entry_point *e = get_entry_point(s, name);

  if (e == NULL) {
    failure();
    printf("Unknown entry point: %s\n", name);
    return;
  }

  int num_ins = entry_num_ins(e);
  // +1 to avoid zero-size arrays, which is UB.
  void* out;
  void* ins[num_ins+1];

  for (int i = 0; i < num_ins; i++) {
    const char *in_name = get_arg(args, 2+i);
    struct variable *v = get_variable(s, in_name);
    if (v == NULL) {
      failure();
      printf("Unknown variable: %s\n", in_name);
      return;
    }
    if (v->value.type != e->in_types[i]) {
      failure();
      printf("Wrong input type.  Expected %s, got %s.\n",
             e->in_types[i]->name, v->value.type->name);
      return;
    }
    ins[i] = value_ptr(&v->value);
  }

  const char *out_name = get_arg(args, 1);
  struct variable *v = create_variable(s, out_name, e->out_type);
  if (v == NULL) {
    failure();
    printf("Variable already exists: %s\n", out_name);
    return;
  }
  out = value_ptr(&v->value);

  int64_t t_start = get_wall_time();
  int err = e->f(s->ctx, out, ins);
  err |= futhark_context_sync(s->ctx);
  int64_t t_end = get_wall_time();
  long long int elapsed_usec = t_end - t_start;
  printf("runtime: %lld\n", elapsed_usec);

  error_check(s, err);
  if (err != 0) {
    // Need to uncreate the output variable, which would otherwise be left
    // in an uninitialised state.
    const char *out_name = get_arg(args, 1);
    struct variable *v = get_variable(s, out_name);
    if (v) {
      drop_variable(v);
    }
  }
}

void cmd_restore(struct server_state *s, const char *args[]) {
  const char *fname = get_arg(args, 0);

  FILE *f = fopen(fname, "rb");
  if (f == NULL) {
    failure();
    printf("Failed to open %s: %s\n", fname, strerror(errno));
    return;
  }

  int bad = 0;
  int values = 0;
  for (int i = 1; arg_exists(args, i); i+=2, values++) {
    const char *vname = get_arg(args, i);
    const char *type = get_arg(args, i+1);

    const struct type *t = get_type(s, type);
    struct variable *v = create_variable(s, vname, t);

    if (v == NULL) {
      bad = 1;
      failure();
      printf("Variable already exists: %s\n", vname);
      break;
    }

    errno = 0;
    if (t->restore(t->aux, f, s->ctx, value_ptr(&v->value)) != 0) {
      bad = 1;
      failure();
      printf("Failed to restore variable %s.\n"
             "Possibly malformed data in %s (errno: %s)\n",
             vname, fname, strerror(errno));
      drop_variable(v);
      break;
    }
  }

  if (!bad && end_of_input(f) != 0) {
    failure();
    printf("Expected EOF after reading %d values from %s\n",
           values, fname);
  }

  fclose(f);

  if (!bad) {
    int err = futhark_context_sync(s->ctx);
    error_check(s, err);
  }
}

void cmd_store(struct server_state *s, const char *args[]) {
  const char *fname = get_arg(args, 0);

  FILE *f = fopen(fname, "wb");
  if (f == NULL) {
    failure();
    printf("Failed to open %s: %s\n", fname, strerror(errno));
  } else {
    for (int i = 1; arg_exists(args, i); i++) {
      const char *vname = get_arg(args, i);
      struct variable *v = get_variable(s, vname);

      if (v == NULL) {
        failure();
        printf("Unknown variable: %s\n", vname);
        return;
      }

      const struct type *t = v->value.type;
      t->store(t->aux, f, s->ctx, value_ptr(&v->value));
    }
    fclose(f);
  }
}

void cmd_free(struct server_state *s, const char *args[]) {
  for (int i = 0; arg_exists(args, i); i++) {
    const char *name = get_arg(args, i);
    struct variable *v = get_variable(s, name);

    if (v == NULL) {
      failure();
      printf("Unknown variable: %s\n", name);
      return;
    }

    const struct type *t = v->value.type;

    int err = t->free(t->aux, s->ctx, value_ptr(&v->value));
    error_check(s, err);
    drop_variable(v);
  }
}

void cmd_rename(struct server_state *s, const char *args[]) {
  const char *oldname = get_arg(args, 0);
  const char *newname = get_arg(args, 1);
  struct variable *old = get_variable(s, oldname);
  struct variable *new = get_variable(s, newname);

  if (old == NULL) {
    failure();
    printf("Unknown variable: %s\n", oldname);
    return;
  }

  if (new != NULL) {
    failure();
    printf("Variable already exists: %s\n", newname);
    return;
  }

  free(old->name);
  old->name = strdup(newname);
}

void cmd_inputs(struct server_state *s, const char *args[]) {
  const char *name = get_arg(args, 0);
  struct entry_point *e = get_entry_point(s, name);

  if (e == NULL) {
    failure();
    printf("Unknown entry point: %s\n", name);
    return;
  }

  int num_ins = entry_num_ins(e);
  for (int i = 0; i < num_ins; i++) {
    if (e->in_unique[i]) {
      putchar('*');
    }
    puts(e->in_types[i]->name);
  }
}

void cmd_output(struct server_state *s, const char *args[]) {
  const char *name = get_arg(args, 0);
  struct entry_point *e = get_entry_point(s, name);

  if (e == NULL) {
    failure();
    printf("Unknown entry point: %s\n", name);
    return;
  }

  if (e->out_unique) {
    putchar('*');
  }
  puts(e->out_type->name);
}

void cmd_clear(struct server_state *s, const char *args[]) {
  (void)args;
  int err = 0;
  for (int i = 0; i < s->variables_capacity; i++) {
    struct variable *v = &s->variables[i];
    if (v->name != NULL) {
      err |= v->value.type->free(v->value.type->aux, s->ctx, value_ptr(&v->value));
      drop_variable(v);
    }
  }
  err |= futhark_context_clear_caches(s->ctx);
  error_check(s, err);
}

void cmd_pause_profiling(struct server_state *s, const char *args[]) {
  (void)args;
  futhark_context_pause_profiling(s->ctx);
}

void cmd_unpause_profiling(struct server_state *s, const char *args[]) {
  (void)args;
  futhark_context_unpause_profiling(s->ctx);
}

void cmd_report(struct server_state *s, const char *args[]) {
  (void)args;
  char *report = futhark_context_report(s->ctx);
  if (report) {
    puts(report);
  } else {
    failure();
    report = futhark_context_get_error(s->ctx);
    if (report) {
      puts(report);
    } else {
      puts("Failed to produce profiling report.\n");
    }
  }
  free(report);
}

void cmd_set_tuning_param(struct server_state *s, const char *args[]) {
  const char *param = get_arg(args, 0);
  const char *val_s = get_arg(args, 1);
  size_t val = atol(val_s);
  int err = futhark_context_config_set_tuning_param(s->cfg, param, val);

  error_check(s, err);

  if (err != 0) {
    printf("Failed to set tuning parameter %s to %ld\n", param, (long)val);
  }
}

void cmd_tuning_params(struct server_state *s, const char *args[]) {
  const char *name = get_arg(args, 0);
  struct entry_point *e = get_entry_point(s, name);

  if (e == NULL) {
    failure();
    printf("Unknown entry point: %s\n", name);
    return;
  }

  const char **params = e->tuning_params;
  for (int i = 0; params[i] != NULL; i++) {
    printf("%s\n", params[i]);
  }
}

void cmd_tuning_param_class(struct server_state *s, const char *args[]) {
  (void)s;
  const char *param = get_arg(args, 0);

  int n = futhark_get_tuning_param_count();

  for (int i = 0; i < n; i++) {
    if (strcmp(futhark_get_tuning_param_name(i), param) == 0) {
      printf("%s\n", futhark_get_tuning_param_class(i));
      return;
    }
  }

  failure();
  printf("Unknown tuning parameter: %s\n", param);
}

void cmd_attributes(struct server_state *s, const char *args[]) {
  const char *name = get_arg(args, 0);
  struct entry_point *e = get_entry_point(s, name);

  if (e == NULL) {
    failure();
    printf("Unknown entry point: %s\n", name);
    return;
  }

  const char **params = e->attrs;
  for (int i = 0; params[i] != NULL; i++) {
    printf("%s\n", params[i]);
  }
}

void cmd_kind(struct server_state *s, const char *args[]) {
  const char *type = get_arg(args, 0);
  const struct type *t = get_type(s, type);

  switch (t->kind) {
    case PRIMITIVE: printf("primitive\n"); return;
    case ARRAY:     printf("array\n");     return;
    case RECORD:    printf("record\n");    return;
    case SUM:       printf("sum\n");       return;
    case OPAQUE:    printf("opaque\n");    return;
  }
  futhark_panic(1, "Invalid kind detected on type \"%s\".\n", t->name);
}

void cmd_type(struct server_state *s, const char *args[]) {
  const char *from_name = get_arg(args, 0);
  struct variable *v = get_variable(s, from_name);

  if (v == NULL) {
    failure();
    printf("Unknown variable: %s\n", from_name);
    return;
  }

  printf("%s\n", v->value.type->name);
}

void cmd_shape(struct server_state *s, const char *args[]) {
  const char *name = get_arg(args, 0);
  struct variable* v = get_variable(s, name);

  if (v == NULL) {
    failure();
    printf("Unknown variable: %s\n", name);
    return;
  }

  if (v->value.type->kind != ARRAY) {
    failure();
    printf("Not an array type\n");
    return;
  }

  const struct array *a = v->value.type->info;

  const int64_t *shape = a->shape(s->ctx, v->value.value.v_ptr);
  for (int i = 0; i < a->rank; ++i) {
    printf("%lld\n", (long long)shape[i]);
  }
}

void cmd_elemtype(struct server_state *s, const char *args[]) {
  const char *type = get_arg(args, 0);
  const struct type *t = get_type(s, type);

  if (t->kind != ARRAY) {
    failure();
    printf("Not an array type\n");
    return;
  }

  const struct array *a = t->info;

  printf("%s\n", a->element_type->name);
}

void cmd_rank(struct server_state *s, const char *args[]) {
  const char *type = get_arg(args, 0);
  const struct type *t = get_type(s, type);

  if (t->kind != ARRAY) {
    failure();
    printf("Not an array type\n");
    return;
  }

  const struct array *a = t->info;
  printf("%d\n", a->rank);
}

void cmd_new_array(struct server_state *s, const char *args[]) {
  const char *to_name = get_arg(args, 0);
  const char *type_name = get_arg(args, 1);
  const struct type *type = get_type(s, type_name);
  struct variable *to = create_variable(s, to_name, type);

  if (to == NULL) {
    failure();
    printf("Variable already exists: %s\n", to_name);
    return;
  }

  if (type->kind != ARRAY) {
    failure();
    printf("Not an array type\n");
    return;
  }

  const struct array *a = type->info;

  int num_args = 0;
  for (int i = 2; arg_exists(args, i); i++) {
    num_args++;
  }

  if (num_args < a->rank) {
    failure();
    printf("Expected %d dimensions, but got %d.\n", a->rank, num_args);
    return;
  }

  int64_t* dims = alloca(a->rank * sizeof(int64_t));
  int64_t n_values = 1;

  for (int i = 0; i < a->rank; ++i) {
    const char *size_arg = get_arg(args, 2+i);
    char* end;
    errno = 0;
    int64_t size = strtoll(size_arg, &end, 10);

    if (errno == ERANGE || *end != '\0' || size < 0) {
      failure();
      printf("Invalid size `%s` of dimension %d.\n", size_arg, i+1);
      return;
    }

    dims[i] = size;
    n_values *= size;
  }

  if (num_args - a->rank != n_values) {
    failure();
    printf("Expected %d values, but got %d.\n", (int)n_values, num_args - a->rank);
    return;
  }

  const void** value_ptrs = alloca(n_values * sizeof(void*));

  for (int64_t i = 0; i < n_values; i++) {
    struct variable* v = get_variable(s, args[2+a->rank+i]);

    if (v == NULL) {
      failure();
      printf("Unknown variable: %s\n", args[2+a->rank+i]);
      return;
    }

    if (strcmp(v->value.type->name, a->element_type->name) != 0) {
      failure();
      printf("Value %d mismatch: expected type %s, got %s\n",
             (int)i, a->element_type->name, v->value.type->name);
      return;
    }

    value_ptrs[i] = value_ptr(&v->value);
  }

  a->new(s->ctx, value_ptr(&to->value), value_ptrs, dims);
}

void cmd_set(struct server_state *s, const char *args[]) {
  const char *arr_name = get_arg(args, 0);
  const char *val_name = get_arg(args, 1);
  struct variable* arr = get_variable(s, arr_name);
  struct variable* val = get_variable(s, val_name);

  if (arr == NULL) {
    failure();
    printf("Unknown variable: %s\n", arr_name);
    return;
  }
  if (val == NULL) {
    failure();
    printf("Unknown variable: %s\n", val_name);
    return;
  }

  if (arr->value.type->kind != ARRAY) {
    failure();
    printf("Not an array type\n");
    return;
  }

  const struct array *a = arr->value.type->info;

  if (strcmp(val->value.type->name, a->element_type->name) != 0) {
    failure();
    printf("Type mismatch: expected element of type %s, got %s\n",
            a->element_type->name, val->value.type->name);
    return;
  }

  for (int i = 0; ; ++i) {
    if (!arg_exists(args, 2+i)) {
      if (i != a->rank) {
        failure();
        printf("%d indices expected but %d values provided.\n", a->rank, i);
        return;
      }
      break;
    }
  }

  const int64_t *shape = a->shape(s->ctx, arr->value.value.v_ptr);
  int64_t* indices = alloca(a->rank * sizeof(int64_t));

  for (int i = 0; i < a->rank; ++i) {
    const char *idx_arg = get_arg(args, 2+i);
    char* end;
    errno = 0;
    int64_t idx = strtoll(idx_arg, &end, 10);

    if (errno == ERANGE || *end != '\0' || idx < 0 || idx >= shape[i]) {
      failure();
      printf("Invalid index `%s` on dimension %d.\n", idx_arg, i+1);
      return;
    }

    indices[i] = idx;
  }

  a->set(s->ctx, arr->value.value.v_ptr, value_ptr(&val->value), indices);
}

void cmd_index(struct server_state *s, const char *args[]) {
  const char *to_name = get_arg(args, 0);
  const char *from_name = get_arg(args, 1);
  struct variable* from = get_variable(s, from_name);

  if (from == NULL) {
    failure();
    printf("Unknown variable: %s\n", from_name);
    return;
  }

  if (from->value.type->kind != ARRAY) {
    failure();
    printf("Not an array type\n");
    return;
  }

  const struct array *a = from->value.type->info;

  for (int i = 0; ; ++i) {
    if (!arg_exists(args, 2+i)) {
      if (i != a->rank) {
        failure();
        printf("%d indices expected but %d values provided.\n", a->rank, i);
        return;
      }
      break;
    }
  }

  const int64_t *shape = a->shape(s->ctx, from->value.value.v_ptr);
  int64_t* indices = alloca(a->rank * sizeof(int64_t));

  for (int i = 0; i < a->rank; ++i) {
    const char *idx_arg = get_arg(args, 2+i);
    char* end;
    errno = 0;
    int64_t idx = strtoll(idx_arg, &end, 10);

    if (errno == ERANGE || *end != '\0' || idx < 0 || idx >= shape[i]) {
      failure();
      printf("Invalid index `%s` on dimension %d.\n", idx_arg, i+1);
      return;
    }

    indices[i] = idx;
  }

  struct variable* to = create_variable(s, to_name, a->element_type);

  if (to == NULL) {
    failure();
    printf("Variable already exists: %s\n", to_name);
    return;
  }

  a->index(s->ctx, value_ptr(&to->value), from->value.value.v_ptr, indices);
}

void cmd_fields(struct server_state *s, const char *args[]) {
  const char *type = get_arg(args, 0);
  const struct type *t = get_type(s, type);

  if (t->kind != RECORD) {
    failure();
    printf("Not a record type\n");
    return;
  }

  const struct record *r = t->info;

  for (int i = 0; i < r->num_fields; i++) {
    const struct field f = r->fields[i];
    printf("%s %s\n", f.name, f.type->name);
  }
}

void cmd_variants(struct server_state *s, const char *args[]) {
  const char *type = get_arg(args, 0);
  const struct type *t = get_type(s, type);

  if (t->kind != SUM) {
    failure();
    printf("Not a sum type\n");
    return;
  }

  const struct sum *st = t->info;

  for (int i = 0; i < st->num_variants; i++) {
    const struct variant *v = &st->variants[i];
    printf("%s\n", v->name);
    for (int i = 0; i < v->num_types; i++) {
      const struct type *f = v->types[i];
      printf("- %s\n", f->name);
    }
  }
}

void cmd_variant(struct server_state *s, const char *args[]) {
  const char *name = get_arg(args, 0);
  struct variable* v = get_variable(s, name);

  if (v == NULL) {
    failure();
    printf("Unknown variable: %s\n", name);
    return;
  }

  const struct type *t = get_type(s, v->value.type->name);

  if (t->kind != SUM) {
    failure();
    printf("Not a sum type\n");
    return;
  }

  const struct sum *st = t->info;

  int i = st->variant(s->ctx, v->value.value.v_ptr);
  const struct variant *var = &st->variants[i];
  printf("%s\n", var->name);
}

void cmd_project(struct server_state *s, const char *args[]) {
  const char *to_name = get_arg(args, 0);
  const char *from_name = get_arg(args, 1);
  const char *field_name = get_arg(args, 2);

  struct variable *from = get_variable(s, from_name);

  if (from == NULL) {
    failure();
    printf("Unknown variable: %s\n", from_name);
    return;
  }

  const struct type *from_type = from->value.type;

  if (from_type->kind != RECORD) {
    failure();
    printf("Not a record type\n");
    return;
  }

  const struct record *r = from_type->info;

  const struct field *field = NULL;
  for (int i = 0; i < r->num_fields; i++) {
    if (strcmp(r->fields[i].name, field_name) == 0) {
      field = &r->fields[i];
      break;
    }
  }

  if (field == NULL) {
    failure();
    printf("No such field\n");
  }

  struct variable *to = create_variable(s, to_name, field->type);

  if (to == NULL) {
    failure();
    printf("Variable already exists: %s\n", to_name);
    return;
  }

  field->project(s->ctx, value_ptr(&to->value), from->value.value.v_ptr);
}

void cmd_new(struct server_state *s, const char *args[]) {
  const char *to_name = get_arg(args, 0);
  const char *type_name = get_arg(args, 1);
  const struct type *type = get_type(s, type_name);
  struct variable *to = create_variable(s, to_name, type);

  if (to == NULL) {
    failure();
    printf("Variable already exists: %s\n", to_name);
    return;
  }

  if (type->kind != RECORD) {
    failure();
    printf("Not a record type\n");
    return;
  }

  const struct record *r = type->info;

  int num_args = 0;
  for (int i = 2; arg_exists(args, i); i++) {
    num_args++;
  }

  if (num_args != r->num_fields) {
    failure();
    printf("%d fields expected but %d values provided.\n", num_args, r->num_fields);
    return;
  }

  const void** value_ptrs = alloca(num_args * sizeof(void*));

  for (int i = 0; i < num_args; i++) {
    struct variable* v = get_variable(s, args[2+i]);

    if (v == NULL) {
      failure();
      printf("Unknown variable: %s\n", args[2+i]);
      return;
    }

    if (strcmp(v->value.type->name, r->fields[i].type->name) != 0) {
      failure();
      printf("Field %s mismatch: expected type %s, got %s\n",
             r->fields[i].name, r->fields[i].type->name, v->value.type->name);
      return;
    }

    value_ptrs[i] = value_ptr(&v->value);
  }

  r->new(s->ctx, value_ptr(&to->value), value_ptrs);
}

void cmd_construct(struct server_state *s, const char *args[]) {
  const char *to_name = get_arg(args, 0);
  const char *type_name = get_arg(args, 1);
  const char *variant_name = get_arg(args, 2);
  const struct type *type = get_type(s, type_name);
  struct variable *to = create_variable(s, to_name, type);

  if (to == NULL) {
    failure();
    printf("Variable already exists: %s\n", to_name);
    return;
  }

  if (type->kind != SUM) {
    failure();
    printf("Not a sum type\n");
    return;
  }

  const struct sum *st = type->info;

  for (int i = 0; i < st->num_variants; i++) {
    const struct variant *var = &st->variants[i];
    if (strcmp(var->name, variant_name) == 0) {
      int num_args = 0;
      for (int i = 3; arg_exists(args, i); i++) {
        num_args++;
      }

      if (num_args != var->num_types) {
        failure();
        printf("%d values expected but %d values provided.\n", var->num_types, num_args);
        return;
      }

      const void** value_ptrs = alloca(num_args * sizeof(void*));

      for (int i = 0; i < num_args; i++) {
        const char *vname = get_arg(args, 3+i);
        struct variable* v = get_variable(s, vname);

        if (v == NULL) {
          failure();
          printf("Unknown variable: %s\n", vname);
          return;
        }

        if (strcmp(v->value.type->name, var->types[i]->name) != 0) {
          failure();
          printf("Value %d mismatch: expected type %s, got %s\n",
                i, var->types[i]->name, v->value.type->name);
          return;
        }

        value_ptrs[i] = value_ptr(&v->value);
      }

      var->new(s->ctx, value_ptr(&to->value), value_ptrs);
      return;
    }
  }

  failure();
  printf("No such variant\n");
}

void cmd_destruct(struct server_state *s, const char *args[]) {
  const char *from_name = get_arg(args, 0);
  struct variable *v = get_variable(s, from_name);

  if (v == NULL) {
    failure();
    printf("Unknown variable: %s\n", from_name);
    return;
  }

  if (v->value.type->kind != SUM) {
    failure();
    printf("Not a sum type\n");
    return;
  }

  const struct sum *sum = v->value.type->info;
  const struct variant *var = &sum->variants[sum->variant(s->ctx, v->value.value.v_ptr)];

  int num_args = 0;
  for (int i = 1; arg_exists(args, i); i++) {
    num_args++;
  }

  if (num_args != var->num_types) {
    failure();
    printf("%d variables expected but %d variables provided.  %s\n", var->num_types, num_args, var->name);
    return;
  }

  const void **value_ptrs = alloca(num_args * sizeof(struct variable*));

  for (int i = 0; i < num_args; i++) {
    const char *vname = get_arg(args, i+1);
    struct variable *vn = create_variable(s, vname, var->types[i]);
    if (vn == NULL) {
      failure();
      printf("Variable already exists: %s\n", vname);
      return;
    }
    value_ptrs[i] = value_ptr(&vn->value);
  }

  var->destruct(s->ctx, value_ptrs, v->value.value.v_ptr);
  return;
}

void cmd_entry_points(struct server_state *s, const char *args[]) {
  (void)args;
  for (int i = 0; s->prog.entry_points[i].name; i++) {
    puts(s->prog.entry_points[i].name);
  }
}

void cmd_types(struct server_state *s, const char *args[]) {
  (void)args;
  for (int i = 0; s->prog.types[i] != NULL; i++) {
    puts(s->prog.types[i]->name);
  }
}

char *next_word(char **line) {
  char *p = *line;

  while (isspace(*p)) {
    p++;
  }

  if (*p == 0) {
    return NULL;
  }

  if (*p == '"') {
    char *save = p+1;
    // Skip ahead till closing quote.
    p++;

    while (*p && *p != '"') {
      p++;
    }

    if (*p == '"') {
      *p = 0;
      *line = p+1;
      return save;
    } else {
      return NULL;
    }
  } else {
    char *save = p;
    // Skip ahead till next whitespace.

    while (*p && !isspace(*p)) {
      p++;
    }

    if (*p) {
      *p = 0;
      *line = p+1;
    } else {
      *line = p;
    }
    return save;
  }
}

void process_line(struct server_state *s, char *line) {
  int max_num_tokens = 1000;
  const char* tokens[max_num_tokens];
  int num_tokens = 0;

  while ((tokens[num_tokens] = next_word(&line)) != NULL) {
    num_tokens++;
    if (num_tokens == max_num_tokens) {
      futhark_panic(1, "Line too long.\n");
    }
  }

  const char *command = tokens[0];

  if (command == NULL) {
    failure();
    printf("Empty line\n");
  } else if (strcmp(command, "call") == 0) {
    cmd_call(s, tokens+1);
  } else if (strcmp(command, "restore") == 0) {
    cmd_restore(s, tokens+1);
  } else if (strcmp(command, "store") == 0) {
    cmd_store(s, tokens+1);
  } else if (strcmp(command, "free") == 0) {
    cmd_free(s, tokens+1);
  } else if (strcmp(command, "rename") == 0) {
    cmd_rename(s, tokens+1);
  } else if (strcmp(command, "inputs") == 0) {
    cmd_inputs(s, tokens+1);
  } else if (strcmp(command, "output") == 0) {
    cmd_output(s, tokens+1);
  } else if (strcmp(command, "clear") == 0) {
    cmd_clear(s, tokens+1);
  } else if (strcmp(command, "pause_profiling") == 0) {
    cmd_pause_profiling(s, tokens+1);
  } else if (strcmp(command, "unpause_profiling") == 0) {
    cmd_unpause_profiling(s, tokens+1);
  } else if (strcmp(command, "report") == 0) {
    cmd_report(s, tokens+1);
  } else if (strcmp(command, "set_tuning_param") == 0) {
    cmd_set_tuning_param(s, tokens+1);
  } else if (strcmp(command, "tuning_params") == 0) {
    cmd_tuning_params(s, tokens+1);
  } else if (strcmp(command, "tuning_param_class") == 0) {
    cmd_tuning_param_class(s, tokens+1);
  } else if (strcmp(command, "kind") == 0) {
    cmd_kind(s, tokens+1);
  } else if (strcmp(command, "type") == 0) {
    cmd_type(s, tokens+1);
  } else if (strcmp(command, "shape") == 0) {
    cmd_shape(s, tokens+1);
  } else if (strcmp(command, "elemtype") == 0) {
    cmd_elemtype(s, tokens+1);
  } else if (strcmp(command, "rank") == 0) {
    cmd_rank(s, tokens+1);
  } else if (strcmp(command, "new_array") == 0) {
    cmd_new_array(s, tokens+1);
  } else if (strcmp(command, "set") == 0) {
    cmd_set(s, tokens+1);
  } else if (strcmp(command, "index") == 0) {
    cmd_index(s, tokens+1);
  } else if (strcmp(command, "fields") == 0) {
    cmd_fields(s, tokens+1);
  } else if (strcmp(command, "variants") == 0) {
    cmd_variants(s, tokens+1);
  } else if (strcmp(command, "variant") == 0) {
    cmd_variant(s, tokens+1);
  } else if (strcmp(command, "new") == 0) {
    cmd_new(s, tokens+1);
  } else if (strcmp(command, "construct") == 0) {
    cmd_construct(s, tokens+1);
  } else if (strcmp(command, "destruct") == 0) {
    cmd_destruct(s, tokens+1);
  } else if (strcmp(command, "project") == 0) {
    cmd_project(s, tokens+1);
  } else if (strcmp(command, "entry_points") == 0) {
    cmd_entry_points(s, tokens+1);
  } else if (strcmp(command, "attributes") == 0) {
    cmd_attributes(s, tokens+1);
  } else if (strcmp(command, "types") == 0) {
    cmd_types(s, tokens+1);
  } else {
    futhark_panic(1, "Unknown command: %s\n", command);
  }
}

void run_server(struct futhark_prog *prog,
                struct futhark_context_config *cfg,
                struct futhark_context *ctx) {
  char *line = NULL;
  size_t buflen = 0;
  ssize_t linelen;

  struct server_state s = {
    .cfg = cfg,
    .ctx = ctx,
    .variables_capacity = 100,
    .prog = *prog
  };

  s.variables = malloc(s.variables_capacity * sizeof(struct variable));

  for (int i = 0; i < s.variables_capacity; i++) {
    s.variables[i].name = NULL;
  }

  ok();
  while ((linelen = getline(&line, &buflen, stdin)) > 0) {
    process_line(&s, line);
    ok();
  }

  free(s.variables);
  free(line);
}

// The aux struct lets us write generic method implementations without
// code duplication.

typedef void* (*aux_array_new_fn)(struct futhark_context*, const void**, const int64_t*);
typedef const int64_t* (*aux_array_shape_fn)(struct futhark_context*, void*);
typedef int (*aux_array_index_fn)(struct futhark_context*, void*, const void*, const int64_t*);
typedef int (*aux_array_values_fn)(struct futhark_context*, void*, void*);
typedef int (*aux_array_free_fn)(struct futhark_context*, void*);

struct array_aux {
  int rank;
  const struct primtype_info_t* info;
  const char *name;
  aux_array_new_fn new;
  aux_array_shape_fn shape;
  aux_array_values_fn values;
  aux_array_free_fn free;
};

int restore_array(const struct array_aux *aux, FILE *f,
                  struct futhark_context *ctx, void *p) {
  void *data = NULL;
  int64_t shape[aux->rank];
  if (read_array(f, aux->info, &data, shape, aux->rank) != 0) {
    return 1;
  }

  void *arr = aux->new(ctx, data, shape);
  if (arr == NULL) {
    return 1;
  }
  int err = futhark_context_sync(ctx);
  *(void**)p = arr;
  free(data);
  return err;
}

void store_array(const struct array_aux *aux, FILE *f,
                 struct futhark_context *ctx, void *p) {
  void *arr = *(void**)p;
  const int64_t *shape = aux->shape(ctx, arr);
  int64_t size = sizeof(aux->info->size);
  for (int i = 0; i < aux->rank; i++) {
    size *= shape[i];
  }
  int32_t *data = malloc(size);
  assert(aux->values(ctx, arr, data) == 0);
  assert(futhark_context_sync(ctx) == 0);
  assert(write_array(f, 1, aux->info, data, shape, aux->rank) == 0);
  free(data);
}

int free_array(const struct array_aux *aux,
               struct futhark_context *ctx, void *p) {
  void *arr = *(void**)p;
  return aux->free(ctx, arr);
}

typedef void* (*opaque_restore_fn)(struct futhark_context*, void*);
typedef int (*opaque_store_fn)(struct futhark_context*, const void*, void **, size_t *);
typedef int (*opaque_free_fn)(struct futhark_context*, void*);

struct opaque_aux {
  opaque_restore_fn restore;
  opaque_store_fn store;
  opaque_free_fn free;
};

int restore_opaque(const struct opaque_aux *aux, FILE *f,
                   struct futhark_context *ctx, void *p) {
  // We have a problem: we need to load data from 'f', since the
  // restore function takes a pointer, but we don't know how much we
  // need (and cannot possibly).  So we do something hacky: we read
  // *all* of the file, pass all of the data to the restore function
  // (which doesn't care if there's extra at the end), then we compute
  // how much space the the object actually takes in serialised form
  // and rewind the file to that position.  The only downside is more IO.
  size_t start = ftell(f);
  size_t size;
  char *bytes = fslurp_file(f, &size);
  void *obj = aux->restore(ctx, bytes);
  free(bytes);
  if (obj != NULL) {
    *(void**)p = obj;
    size_t obj_size;
    (void)aux->store(ctx, obj, NULL, &obj_size);
    fseek(f, start+obj_size, SEEK_SET);
    return 0;
  } else {
    fseek(f, start, SEEK_SET);
    return 1;
  }
}

void store_opaque(const struct opaque_aux *aux, FILE *f,
                  struct futhark_context *ctx, void *p) {
  void *obj = *(void**)p;
  size_t obj_size;
  void *data = NULL;
  (void)aux->store(ctx, obj, &data, &obj_size);
  assert(futhark_context_sync(ctx) == 0);
  fwrite(data, sizeof(char), obj_size, f);
  free(data);
}

int free_opaque(const struct opaque_aux *aux,
                struct futhark_context *ctx, void *p) {
  void *obj = *(void**)p;
  return aux->free(ctx, obj);
}

// End of server.h.

// Start of tuning.h.


int is_blank_line_or_comment(const char *s) {
  size_t i = strspn(s, " \t\n");
  return s[i] == '\0' || // Line is blank.
         strncmp(s + i, "--", 2) == 0; // Line is comment.
}

static char* load_tuning_file(const char *fname,
                              void *cfg,
                              int (*set_tuning_param)(void*, const char*, size_t)) {
  const int max_line_len = 1024;
  char* line = (char*) malloc(max_line_len);

  FILE *f = fopen(fname, "r");

  if (f == NULL) {
    snprintf(line, max_line_len, "Cannot open file: %s", strerror(errno));
    return line;
  }

  int lineno = 0;
  while (fgets(line, max_line_len, f) != NULL) {
    lineno++;
    if (is_blank_line_or_comment(line)) {
      continue;
    }
    char *eql = strstr(line, "=");
    if (eql) {
      *eql = 0;
      char *endptr;
      int value = strtol(eql+1, &endptr, 10);
      if (*endptr && *endptr != '\n') {
        snprintf(line, max_line_len, "Invalid line %d (must be of form 'name=int').",
                 lineno);
        return line;
      }
      if (set_tuning_param(cfg, line, (size_t)value) != 0) {
        char* err = (char*) malloc(max_line_len + 50);
        snprintf(err, max_line_len + 50, "Unknown name '%s' on line %d.", line, lineno);
        free(line);
        return err;
      }
    } else {
      snprintf(line, max_line_len, "Invalid line %d (must be of form 'name=int').",
               lineno);
      return line;
    }
  }

  free(line);

  return NULL;
}

// End of tuning.h.

const struct type type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR;
const struct type type_ZLf64z2cUz20UZMZNf64ZR;
const struct type type_ZMZNZMZNf64;
const struct type type_ZMZNf64;
const struct type type_ZMZNi64;
const struct type type_params;
const struct field type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR_fields[] = {{.name ="0", .type =&type_ZMZNZMZNf64, .project =(project_fn) futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_0}, {.name ="1", .type =&type_ZMZNZMZNf64, .project =(project_fn) futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_1}, {.name ="2", .type =&type_ZMZNZMZNf64, .project =(project_fn) futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_2}, {.name ="3", .type =&type_ZMZNZMZNf64, .project =(project_fn) futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_3}, {.name ="4", .type =&type_ZMZNZMZNf64, .project =(project_fn) futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_4}, {.name ="5", .type =&type_ZMZNZMZNf64, .project =(project_fn) futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_5}, {.name ="6", .type =&type_ZMZNZMZNf64, .project =(project_fn) futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_6}, {.name ="7", .type =&type_ZMZNZMZNf64, .project =(project_fn) futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_7}, {.name ="8", .type =&type_ZMZNZMZNf64, .project =(project_fn) futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_8}};
int futhark_new_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_wrap(struct futhark_context *ctx, void **outp, const void *fields[])
{
    struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 * *out = (struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 * *) outp;
    const struct futhark_f64_2d * v0 = *(const struct futhark_f64_2d * *) fields[0];
    const struct futhark_f64_2d * v1 = *(const struct futhark_f64_2d * *) fields[1];
    const struct futhark_f64_2d * v2 = *(const struct futhark_f64_2d * *) fields[2];
    const struct futhark_f64_2d * v3 = *(const struct futhark_f64_2d * *) fields[3];
    const struct futhark_f64_2d * v4 = *(const struct futhark_f64_2d * *) fields[4];
    const struct futhark_f64_2d * v5 = *(const struct futhark_f64_2d * *) fields[5];
    const struct futhark_f64_2d * v6 = *(const struct futhark_f64_2d * *) fields[6];
    const struct futhark_f64_2d * v7 = *(const struct futhark_f64_2d * *) fields[7];
    const struct futhark_f64_2d * v8 = *(const struct futhark_f64_2d * *) fields[8];
    
    return futhark_new_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64(ctx, out, v0, v1, v2, v3, v4, v5, v6, v7, v8);
}
const struct record type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR_record = {.num_fields =9, .fields =type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR_fields, .new =futhark_new_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_wrap};
const struct opaque_aux type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR_aux = {.store =(opaque_store_fn) futhark_store_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64, .restore =(opaque_restore_fn) futhark_restore_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64, .free =(opaque_free_fn) futhark_free_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64};
const struct type type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR = {.name ="([][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64)", .restore =(restore_fn) restore_opaque, .store =(store_fn) store_opaque, .free =(free_fn) free_opaque, .aux =&type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR_aux, .kind =RECORD, .info =&type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR_record};
const struct field type_ZLf64z2cUz20UZMZNf64ZR_fields[] = {{.name ="0", .type =&type_f64, .project =(project_fn) futhark_project_opaque_tup2_f64_arr1d_f64_0}, {.name ="1", .type =&type_ZMZNf64, .project =(project_fn) futhark_project_opaque_tup2_f64_arr1d_f64_1}};
int futhark_new_opaque_tup2_f64_arr1d_f64_wrap(struct futhark_context *ctx, void **outp, const void *fields[])
{
    struct futhark_opaque_tup2_f64_arr1d_f64 * *out = (struct futhark_opaque_tup2_f64_arr1d_f64 * *) outp;
    const double v0 = *(const double *) fields[0];
    const struct futhark_f64_1d * v1 = *(const struct futhark_f64_1d * *) fields[1];
    
    return futhark_new_opaque_tup2_f64_arr1d_f64(ctx, out, v0, v1);
}
const struct record type_ZLf64z2cUz20UZMZNf64ZR_record = {.num_fields =2, .fields =type_ZLf64z2cUz20UZMZNf64ZR_fields, .new =futhark_new_opaque_tup2_f64_arr1d_f64_wrap};
const struct opaque_aux type_ZLf64z2cUz20UZMZNf64ZR_aux = {.store =(opaque_store_fn) futhark_store_opaque_tup2_f64_arr1d_f64, .restore =(opaque_restore_fn) futhark_restore_opaque_tup2_f64_arr1d_f64, .free =(opaque_free_fn) futhark_free_opaque_tup2_f64_arr1d_f64};
const struct type type_ZLf64z2cUz20UZMZNf64ZR = {.name ="(f64, []f64)", .restore =(restore_fn) restore_opaque, .store =(store_fn) store_opaque, .free =(free_fn) free_opaque, .aux =&type_ZLf64z2cUz20UZMZNf64ZR_aux, .kind =RECORD, .info =&type_ZLf64z2cUz20UZMZNf64ZR_record};
void *futhark_new_f64_2d_aux_wrap(struct futhark_context *ctx, const void *p, const int64_t *shape)
{
    return futhark_new_f64_2d(ctx, p, shape[0], shape[1]);
}
int futhark_new_f64_2d_wrap(struct futhark_context *ctx, struct futhark_f64_2d * *outp, double *ps[], const int64_t *shape)
{
    int64_t n_values = 1;
    
    for (int i = 0; i < 2; ++i)
        n_values *= shape[i];
    
    double *values = alloca(n_values * sizeof(double));
    
    for (int64_t i = 0; i < n_values; ++i)
        values[i] = *ps[i];
    *outp = futhark_new_f64_2d(ctx, values, shape[0], shape[1]);
    return 0;
}
int futhark_new_f64_2d_set(struct futhark_context *ctx, struct futhark_f64_2d * arr, double *val, const int64_t *is)
{
    const int64_t *shape = futhark_shape_f64_2d(ctx, arr);
    uint64_t idx = is[0];
    
    for (int i = 1; i < 2; ++i) {
        idx *= shape[i - 1];
        idx += is[i];
    }
    ((double *) futhark_values_raw_f64_2d(ctx, arr))[idx] = *val;
    return 0;
}
int futhark_index_f64_2d_wrap(struct futhark_context *ctx, void *dest, struct futhark_f64_2d * arr, const int64_t *is)
{
    return futhark_index_f64_2d(ctx, dest, arr, is[0], is[1]);
}
const struct array type_ZMZNZMZNf64_array = {.rank =2, .element_type =&type_f64, .new =(array_new_fn) futhark_new_f64_2d_wrap, .set =(array_set_fn) futhark_new_f64_2d_set, .shape =(array_shape_fn) futhark_shape_f64_2d, .index =(array_index_fn) futhark_index_f64_2d_wrap};
const struct array_aux type_ZMZNZMZNf64_aux = {.name ="[][]f64", .rank =2, .info =&f64_info, .new =(aux_array_new_fn) futhark_new_f64_2d_aux_wrap, .free =(aux_array_free_fn) futhark_free_f64_2d, .shape =(aux_array_shape_fn) futhark_shape_f64_2d, .values =(aux_array_values_fn) futhark_values_f64_2d};
const struct type type_ZMZNZMZNf64 = {.name ="[][]f64", .restore =(restore_fn) restore_array, .store =(store_fn) store_array, .free =(free_fn) free_array, .aux =&type_ZMZNZMZNf64_aux, .kind =ARRAY, .info =&type_ZMZNZMZNf64_array};
void *futhark_new_f64_1d_aux_wrap(struct futhark_context *ctx, const void *p, const int64_t *shape)
{
    return futhark_new_f64_1d(ctx, p, shape[0]);
}
int futhark_new_f64_1d_wrap(struct futhark_context *ctx, struct futhark_f64_1d * *outp, double *ps[], const int64_t *shape)
{
    int64_t n_values = 1;
    
    for (int i = 0; i < 1; ++i)
        n_values *= shape[i];
    
    double *values = alloca(n_values * sizeof(double));
    
    for (int64_t i = 0; i < n_values; ++i)
        values[i] = *ps[i];
    *outp = futhark_new_f64_1d(ctx, values, shape[0]);
    return 0;
}
int futhark_new_f64_1d_set(struct futhark_context *ctx, struct futhark_f64_1d * arr, double *val, const int64_t *is)
{
    const int64_t *shape = futhark_shape_f64_1d(ctx, arr);
    uint64_t idx = is[0];
    
    for (int i = 1; i < 1; ++i) {
        idx *= shape[i - 1];
        idx += is[i];
    }
    ((double *) futhark_values_raw_f64_1d(ctx, arr))[idx] = *val;
    return 0;
}
int futhark_index_f64_1d_wrap(struct futhark_context *ctx, void *dest, struct futhark_f64_1d * arr, const int64_t *is)
{
    return futhark_index_f64_1d(ctx, dest, arr, is[0]);
}
const struct array type_ZMZNf64_array = {.rank =1, .element_type =&type_f64, .new =(array_new_fn) futhark_new_f64_1d_wrap, .set =(array_set_fn) futhark_new_f64_1d_set, .shape =(array_shape_fn) futhark_shape_f64_1d, .index =(array_index_fn) futhark_index_f64_1d_wrap};
const struct array_aux type_ZMZNf64_aux = {.name ="[]f64", .rank =1, .info =&f64_info, .new =(aux_array_new_fn) futhark_new_f64_1d_aux_wrap, .free =(aux_array_free_fn) futhark_free_f64_1d, .shape =(aux_array_shape_fn) futhark_shape_f64_1d, .values =(aux_array_values_fn) futhark_values_f64_1d};
const struct type type_ZMZNf64 = {.name ="[]f64", .restore =(restore_fn) restore_array, .store =(store_fn) store_array, .free =(free_fn) free_array, .aux =&type_ZMZNf64_aux, .kind =ARRAY, .info =&type_ZMZNf64_array};
void *futhark_new_i64_1d_aux_wrap(struct futhark_context *ctx, const void *p, const int64_t *shape)
{
    return futhark_new_i64_1d(ctx, p, shape[0]);
}
int futhark_new_i64_1d_wrap(struct futhark_context *ctx, struct futhark_i64_1d * *outp, int64_t *ps[], const int64_t *shape)
{
    int64_t n_values = 1;
    
    for (int i = 0; i < 1; ++i)
        n_values *= shape[i];
    
    int64_t *values = alloca(n_values * sizeof(int64_t));
    
    for (int64_t i = 0; i < n_values; ++i)
        values[i] = *ps[i];
    *outp = futhark_new_i64_1d(ctx, values, shape[0]);
    return 0;
}
int futhark_new_i64_1d_set(struct futhark_context *ctx, struct futhark_i64_1d * arr, int64_t *val, const int64_t *is)
{
    const int64_t *shape = futhark_shape_i64_1d(ctx, arr);
    uint64_t idx = is[0];
    
    for (int i = 1; i < 1; ++i) {
        idx *= shape[i - 1];
        idx += is[i];
    }
    ((int64_t *) futhark_values_raw_i64_1d(ctx, arr))[idx] = *val;
    return 0;
}
int futhark_index_i64_1d_wrap(struct futhark_context *ctx, void *dest, struct futhark_i64_1d * arr, const int64_t *is)
{
    return futhark_index_i64_1d(ctx, dest, arr, is[0]);
}
const struct array type_ZMZNi64_array = {.rank =1, .element_type =&type_i64, .new =(array_new_fn) futhark_new_i64_1d_wrap, .set =(array_set_fn) futhark_new_i64_1d_set, .shape =(array_shape_fn) futhark_shape_i64_1d, .index =(array_index_fn) futhark_index_i64_1d_wrap};
const struct array_aux type_ZMZNi64_aux = {.name ="[]i64", .rank =1, .info =&i64_info, .new =(aux_array_new_fn) futhark_new_i64_1d_aux_wrap, .free =(aux_array_free_fn) futhark_free_i64_1d, .shape =(aux_array_shape_fn) futhark_shape_i64_1d, .values =(aux_array_values_fn) futhark_values_i64_1d};
const struct type type_ZMZNi64 = {.name ="[]i64", .restore =(restore_fn) restore_array, .store =(store_fn) store_array, .free =(free_fn) free_array, .aux =&type_ZMZNi64_aux, .kind =ARRAY, .info =&type_ZMZNi64_array};
const struct field type_params_fields[] = {{.name ="wdown", .type =&type_ZMZNZMZNf64, .project =(project_fn) futhark_project_opaque_params_wdown}, {.name ="wkey", .type =&type_ZMZNZMZNf64, .project =(project_fn) futhark_project_opaque_params_wkey}, {.name ="wout", .type =&type_ZMZNZMZNf64, .project =(project_fn) futhark_project_opaque_params_wout}, {.name ="wpe", .type =&type_ZMZNZMZNf64, .project =(project_fn) futhark_project_opaque_params_wpe}, {.name ="wqry", .type =&type_ZMZNZMZNf64, .project =(project_fn) futhark_project_opaque_params_wqry}, {.name ="wte", .type =&type_ZMZNZMZNf64, .project =(project_fn) futhark_project_opaque_params_wte}, {.name ="wup", .type =&type_ZMZNZMZNf64, .project =(project_fn) futhark_project_opaque_params_wup}, {.name ="wval", .type =&type_ZMZNZMZNf64, .project =(project_fn) futhark_project_opaque_params_wval}, {.name ="wvoc", .type =&type_ZMZNZMZNf64, .project =(project_fn) futhark_project_opaque_params_wvoc}};
int futhark_new_opaque_params_wrap(struct futhark_context *ctx, void **outp, const void *fields[])
{
    struct futhark_opaque_params * *out = (struct futhark_opaque_params * *) outp;
    const struct futhark_f64_2d * v0 = *(const struct futhark_f64_2d * *) fields[0];
    const struct futhark_f64_2d * v1 = *(const struct futhark_f64_2d * *) fields[1];
    const struct futhark_f64_2d * v2 = *(const struct futhark_f64_2d * *) fields[2];
    const struct futhark_f64_2d * v3 = *(const struct futhark_f64_2d * *) fields[3];
    const struct futhark_f64_2d * v4 = *(const struct futhark_f64_2d * *) fields[4];
    const struct futhark_f64_2d * v5 = *(const struct futhark_f64_2d * *) fields[5];
    const struct futhark_f64_2d * v6 = *(const struct futhark_f64_2d * *) fields[6];
    const struct futhark_f64_2d * v7 = *(const struct futhark_f64_2d * *) fields[7];
    const struct futhark_f64_2d * v8 = *(const struct futhark_f64_2d * *) fields[8];
    
    return futhark_new_opaque_params(ctx, out, v0, v1, v2, v3, v4, v5, v6, v7, v8);
}
const struct record type_params_record = {.num_fields =9, .fields =type_params_fields, .new =futhark_new_opaque_params_wrap};
const struct opaque_aux type_params_aux = {.store =(opaque_store_fn) futhark_store_opaque_params, .restore =(opaque_restore_fn) futhark_restore_opaque_params, .free =(opaque_free_fn) futhark_free_opaque_params};
const struct type type_params = {.name ="params", .restore =(restore_fn) restore_opaque, .store =(store_fn) store_opaque, .free =(free_fn) free_opaque, .aux =&type_params_aux, .kind =RECORD, .info =&type_params_record};
const struct type *cal_loss_in_types[] = {&type_params, &type_ZMZNi64, &type_ZMZNZMZNf64, &type_ZMZNZMZNf64, NULL};
bool cal_loss_in_unique[] = {false, false, false, false};
const char *cal_loss_tuning_params[] = {NULL};
const char *cal_loss_attrs[] = {NULL};
int call_cal_loss(struct futhark_context *ctx, void *out, void **ins)
{
    struct futhark_opaque_params * in0 = *(struct futhark_opaque_params * *) ins[0];
    struct futhark_i64_1d * in1 = *(struct futhark_i64_1d * *) ins[1];
    struct futhark_f64_2d * in2 = *(struct futhark_f64_2d * *) ins[2];
    struct futhark_f64_2d * in3 = *(struct futhark_f64_2d * *) ins[3];
    
    return futhark_entry_cal_loss(ctx, out, in0, in1, in2, in3);
}
const struct type *forward_seq_in_types[] = {&type_params, &type_ZMZNi64, &type_ZMZNZMZNf64, NULL};
bool forward_seq_in_unique[] = {false, false, false};
const char *forward_seq_tuning_params[] = {NULL};
const char *forward_seq_attrs[] = {NULL};
int call_forward_seq(struct futhark_context *ctx, void *out, void **ins)
{
    struct futhark_opaque_params * in0 = *(struct futhark_opaque_params * *) ins[0];
    struct futhark_i64_1d * in1 = *(struct futhark_i64_1d * *) ins[1];
    struct futhark_f64_2d * in2 = *(struct futhark_f64_2d * *) ins[2];
    
    return futhark_entry_forward_seq(ctx, out, in0, in1, in2);
}
const struct type *grad_loss_in_types[] = {&type_params, &type_ZMZNi64, &type_ZMZNZMZNf64, &type_ZMZNZMZNf64, NULL};
bool grad_loss_in_unique[] = {false, false, false, false};
const char *grad_loss_tuning_params[] = {NULL};
const char *grad_loss_attrs[] = {NULL};
int call_grad_loss(struct futhark_context *ctx, void *out, void **ins)
{
    struct futhark_opaque_params * in0 = *(struct futhark_opaque_params * *) ins[0];
    struct futhark_i64_1d * in1 = *(struct futhark_i64_1d * *) ins[1];
    struct futhark_f64_2d * in2 = *(struct futhark_f64_2d * *) ins[2];
    struct futhark_f64_2d * in3 = *(struct futhark_f64_2d * *) ins[3];
    
    return futhark_entry_grad_loss(ctx, out, in0, in1, in2, in3);
}
const struct type *make_params_in_types[] = {&type_ZMZNZMZNf64, &type_ZMZNZMZNf64, &type_ZMZNZMZNf64, &type_ZMZNZMZNf64, &type_ZMZNZMZNf64, &type_ZMZNZMZNf64, &type_ZMZNZMZNf64, &type_ZMZNZMZNf64, &type_ZMZNZMZNf64, NULL};
bool make_params_in_unique[] = {false, false, false, false, false, false, false, false, false};
const char *make_params_tuning_params[] = {NULL};
const char *make_params_attrs[] = {NULL};
int call_make_params(struct futhark_context *ctx, void *out, void **ins)
{
    struct futhark_f64_2d * in0 = *(struct futhark_f64_2d * *) ins[0];
    struct futhark_f64_2d * in1 = *(struct futhark_f64_2d * *) ins[1];
    struct futhark_f64_2d * in2 = *(struct futhark_f64_2d * *) ins[2];
    struct futhark_f64_2d * in3 = *(struct futhark_f64_2d * *) ins[3];
    struct futhark_f64_2d * in4 = *(struct futhark_f64_2d * *) ins[4];
    struct futhark_f64_2d * in5 = *(struct futhark_f64_2d * *) ins[5];
    struct futhark_f64_2d * in6 = *(struct futhark_f64_2d * *) ins[6];
    struct futhark_f64_2d * in7 = *(struct futhark_f64_2d * *) ins[7];
    struct futhark_f64_2d * in8 = *(struct futhark_f64_2d * *) ins[8];
    
    return futhark_entry_make_params(ctx, out, in0, in1, in2, in3, in4, in5, in6, in7, in8);
}
const struct type *types[] = {&type_i8, &type_i16, &type_i32, &type_i64, &type_u8, &type_u16, &type_u32, &type_u64, &type_f16, &type_f32, &type_f64, &type_bool, &type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR, &type_ZLf64z2cUz20UZMZNf64ZR, &type_ZMZNZMZNf64, &type_ZMZNf64, &type_ZMZNi64, &type_params, NULL};
struct entry_point entry_points[] = {{.name ="cal_loss", .f =call_cal_loss, .tuning_params =cal_loss_tuning_params, .in_types =cal_loss_in_types, .out_type =&type_ZLf64z2cUz20UZMZNf64ZR, .in_unique =cal_loss_in_unique, .out_unique =false, .attrs =cal_loss_attrs}, {.name ="forward_seq", .f =call_forward_seq, .tuning_params =forward_seq_tuning_params, .in_types =forward_seq_in_types, .out_type =&type_ZMZNZMZNf64, .in_unique =forward_seq_in_unique, .out_unique =false, .attrs =forward_seq_attrs}, {.name ="grad_loss", .f =call_grad_loss, .tuning_params =grad_loss_tuning_params, .in_types =grad_loss_in_types, .out_type =&type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR, .in_unique =grad_loss_in_unique, .out_unique =false, .attrs =grad_loss_attrs}, {.name ="make_params", .f =call_make_params, .tuning_params =make_params_tuning_params, .in_types =make_params_in_types, .out_type =&type_params, .in_unique =make_params_in_unique, .out_unique =false, .attrs =make_params_attrs}, {.name =NULL}};
struct futhark_prog prog = {.types =types, .entry_points =entry_points};
int parse_options(struct futhark_context_config *cfg, int argc, char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"debugging", no_argument, NULL, 1}, {"log", no_argument, NULL, 2}, {"profile", no_argument, NULL, 3}, {"help", no_argument, NULL, 4}, {"print-params", no_argument, NULL, 5}, {"param", required_argument, NULL, 6}, {"tuning", required_argument, NULL, 7}, {"cache-file", required_argument, NULL, 8}, {0, 0, 0, 0}};
    static char *option_descriptions = "  -D/--debugging     Perform possibly expensive internal correctness checks and verbose logging.\n  -L/--log           Print various low-overhead logging information while running.\n  -P/--profile       Enable the collection of profiling information.\n  -h/--help          Print help information and exit.\n  --print-params     Print all tuning parameters that can be set with --param or --tuning.\n  --param ASSIGNMENT Set a tuning parameter to the given value.\n  --tuning FILE      Read size=value assignments from the given file.\n  --cache-file FILE  Store program cache here.\n";
    
    while ((ch = getopt_long(argc, argv, ":DLPh", long_options, NULL)) != -1) {
        if (ch == 1 || ch == 'D')
            futhark_context_config_set_debugging(cfg, 1);
        if (ch == 2 || ch == 'L')
            futhark_context_config_set_logging(cfg, 1);
        if (ch == 3 || ch == 'P')
            futhark_context_config_set_profiling(cfg, 1);
        if (ch == 4 || ch == 'h') {
            printf("Usage: %s [OPTIONS]...\nOptions:\n\n%s\nFor more information, consult the Futhark User's Guide or the man pages.\n", fut_progname, option_descriptions);
            exit(0);
        }
        if (ch == 5) {
            int n = futhark_get_tuning_param_count();
            
            for (int i = 0; i < n; i++)
                printf("%s (%s)\n", futhark_get_tuning_param_name(i), futhark_get_tuning_param_class(i));
            exit(0);
        }
        if (ch == 6) {
            char *name = optarg;
            char *equals = strstr(optarg, "=");
            char *value_str = equals != NULL ? equals + 1 : optarg;
            int value = atoi(value_str);
            
            if (equals != NULL) {
                *equals = 0;
                if (futhark_context_config_set_tuning_param(cfg, name, value) != 0)
                    futhark_panic(1, "Unknown parameter: %s\n", name);
            } else
                futhark_panic(1, "Invalid argument for --parameter option: %s\n", optarg);
        }
        if (ch == 7) {
            char *ret = load_tuning_file(optarg, cfg, (int (*)(void *, const char *, size_t)) futhark_context_config_set_tuning_param);
            
            if (ret != NULL)
                futhark_panic(1, "When loading tuning file '%s': %s\n", optarg, ret);
        }
        if (ch == 8)
            futhark_context_config_set_cache_file(cfg, optarg);
        if (ch == ':')
            futhark_panic(-1, "Missing argument for option %s\n", argv[optind - 1]);
        if (ch == '?') {
            fprintf(stderr, "Usage: %s [OPTIONS]...\nOptions:\n\n%s\n", fut_progname, "  -D/--debugging     Perform possibly expensive internal correctness checks and verbose logging.\n  -L/--log           Print various low-overhead logging information while running.\n  -P/--profile       Enable the collection of profiling information.\n  -h/--help          Print help information and exit.\n  --print-params     Print all tuning parameters that can be set with --param or --tuning.\n  --param ASSIGNMENT Set a tuning parameter to the given value.\n  --tuning FILE      Read size=value assignments from the given file.\n  --cache-file FILE  Store program cache here.\n");
            futhark_panic(1, "Unknown option: %s\n", argv[optind - 1]);
        }
    }
    return optind;
}
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct futhark_context_config *cfg = futhark_context_config_new();
    
    assert(cfg != NULL);
    
    int parsed_options = parse_options(cfg, argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    if (argc != 0)
        futhark_panic(1, "Excess non-option: %s\n", argv[0]);
    
    struct futhark_context *ctx = futhark_context_new(cfg);
    
    assert(ctx != NULL);
    futhark_context_set_logging_file(ctx, stdout);
    
    char *error = futhark_context_get_error(ctx);
    
    if (error != NULL)
        futhark_panic(1, "Error during context initialisation:\n%s", error);
    if (entry_point != NULL)
        run_server(&prog, cfg, ctx);
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
}

#ifdef _MSC_VER
#define inline __inline
#endif
#include <string.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <ctype.h>



#define FUTHARK_F64_ENABLED

// Start of scalar.h.

// Implementation of the primitive scalar operations.  Very
// repetitive.  This code is inserted directly into both CUDA and
// OpenCL programs, as well as the CPU code, so it has some #ifdefs to
// work everywhere.  Some operations are defined as macros because
// this allows us to use them as constant expressions in things like
// array sizes and static initialisers.

// Some of the #ifdefs are because OpenCL uses type-generic functions
// for some operations (e.g. sqrt), while C and CUDA sensibly use
// distinct functions for different precisions (e.g. sqrtf() and
// sqrt()).  This is quite annoying.  Due to C's unfortunate casting
// rules, it is also really easy to accidentally implement
// floating-point functions in the wrong precision, so be careful.

// Double-precision definitions are only included if the preprocessor
// macro FUTHARK_F64_ENABLED is set.

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

SCALAR_FUN_ATTR int32_t fptobits_f32_i32(float x);
SCALAR_FUN_ATTR float bitstofp_i32_f32(int32_t x);

SCALAR_FUN_ATTR uint8_t   add8(uint8_t x, uint8_t y)   { return x + y; }
SCALAR_FUN_ATTR uint16_t add16(uint16_t x, uint16_t y) { return x + y; }
SCALAR_FUN_ATTR uint32_t add32(uint32_t x, uint32_t y) { return x + y; }
SCALAR_FUN_ATTR uint64_t add64(uint64_t x, uint64_t y) { return x + y; }

SCALAR_FUN_ATTR uint8_t   sub8(uint8_t x, uint8_t y)   { return x - y; }
SCALAR_FUN_ATTR uint16_t sub16(uint16_t x, uint16_t y) { return x - y; }
SCALAR_FUN_ATTR uint32_t sub32(uint32_t x, uint32_t y) { return x - y; }
SCALAR_FUN_ATTR uint64_t sub64(uint64_t x, uint64_t y) { return x - y; }

SCALAR_FUN_ATTR uint8_t   mul8(uint8_t x, uint8_t y)   { return x * y; }
SCALAR_FUN_ATTR uint16_t mul16(uint16_t x, uint16_t y) { return x * y; }
SCALAR_FUN_ATTR uint32_t mul32(uint32_t x, uint32_t y) { return x * y; }
SCALAR_FUN_ATTR uint64_t mul64(uint64_t x, uint64_t y) { return x * y; }

#if defined(ISPC)

SCALAR_FUN_ATTR uint8_t udiv8(uint8_t x, uint8_t y) {
  // This strange pattern is used to prevent the ISPC compiler from
  // causing SIGFPEs and bogus results on divisions where inactive lanes
  // have 0-valued divisors. It ensures that any inactive lane instead
  // has a divisor of 1. https://github.com/ispc/ispc/issues/2292
  uint8_t ys = 1;
  foreach_active(i) { ys = y; }
  return x / ys;
}

SCALAR_FUN_ATTR uint16_t udiv16(uint16_t x, uint16_t y) {
  uint16_t ys = 1;
  foreach_active(i) { ys = y; }
  return x / ys;
}

SCALAR_FUN_ATTR uint32_t udiv32(uint32_t x, uint32_t y) {
  uint32_t ys = 1;
  foreach_active(i) { ys = y; }
  return x / ys;
}

SCALAR_FUN_ATTR uint64_t udiv64(uint64_t x, uint64_t y) {
  uint64_t ys = 1;
  foreach_active(i) { ys = y; }
  return x / ys;
}

SCALAR_FUN_ATTR uint8_t udiv_up8(uint8_t x, uint8_t y) {
  uint8_t ys = 1;
  foreach_active(i) { ys = y; }
  return (x + y - 1) / ys;
}

SCALAR_FUN_ATTR uint16_t udiv_up16(uint16_t x, uint16_t y) {
  uint16_t ys = 1;
  foreach_active(i) { ys = y; }
  return (x + y - 1) / ys;
}

SCALAR_FUN_ATTR uint32_t udiv_up32(uint32_t x, uint32_t y) {
  uint32_t ys = 1;
  foreach_active(i) { ys = y; }
  return (x + y - 1) / ys;
}

SCALAR_FUN_ATTR uint64_t udiv_up64(uint64_t x, uint64_t y) {
  uint64_t ys = 1;
  foreach_active(i) { ys = y; }
  return (x + y - 1) / ys;
}

SCALAR_FUN_ATTR uint8_t umod8(uint8_t x, uint8_t y) {
  uint8_t ys = 1;
  foreach_active(i) { ys = y; }
  return x % ys;
}

SCALAR_FUN_ATTR uint16_t umod16(uint16_t x, uint16_t y) {
  uint16_t ys = 1;
  foreach_active(i) { ys = y; }
  return x % ys;
}

SCALAR_FUN_ATTR uint32_t umod32(uint32_t x, uint32_t y) {
  uint32_t ys = 1;
  foreach_active(i) { ys = y; }
  return x % ys;
}

SCALAR_FUN_ATTR uint64_t umod64(uint64_t x, uint64_t y) {
  uint64_t ys = 1;
  foreach_active(i) { ys = y; }
  return x % ys;
}

SCALAR_FUN_ATTR uint8_t udiv_safe8(uint8_t x, uint8_t y) {
  uint8_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x / ys;
}

SCALAR_FUN_ATTR uint16_t udiv_safe16(uint16_t x, uint16_t y) {
  uint16_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x / ys;
}

SCALAR_FUN_ATTR uint32_t udiv_safe32(uint32_t x, uint32_t y) {
  uint32_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x / ys;
}

SCALAR_FUN_ATTR uint64_t udiv_safe64(uint64_t x, uint64_t y) {
  uint64_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x / ys;
}

SCALAR_FUN_ATTR uint8_t udiv_up_safe8(uint8_t x, uint8_t y) {
  uint8_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : (x + y - 1) / ys;
}

SCALAR_FUN_ATTR uint16_t udiv_up_safe16(uint16_t x, uint16_t y) {
  uint16_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : (x + y - 1) / ys;
}

SCALAR_FUN_ATTR uint32_t udiv_up_safe32(uint32_t x, uint32_t y) {
  uint32_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : (x + y - 1) / ys;
}

SCALAR_FUN_ATTR uint64_t udiv_up_safe64(uint64_t x, uint64_t y) {
  uint64_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : (x + y - 1) / ys;
}

SCALAR_FUN_ATTR uint8_t umod_safe8(uint8_t x, uint8_t y) {
  uint8_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x % ys;
}

SCALAR_FUN_ATTR uint16_t umod_safe16(uint16_t x, uint16_t y) {
  uint16_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x % ys;
}

SCALAR_FUN_ATTR uint32_t umod_safe32(uint32_t x, uint32_t y) {
  uint32_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x % ys;
}

SCALAR_FUN_ATTR uint64_t umod_safe64(uint64_t x, uint64_t y) {
  uint64_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x % ys;
}

SCALAR_FUN_ATTR int8_t sdiv8(int8_t x, int8_t y) {
  int8_t ys = 1;
  foreach_active(i) { ys = y; }
  int8_t q = x / ys;
  int8_t r = x % ys;
  return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}

SCALAR_FUN_ATTR int16_t sdiv16(int16_t x, int16_t y) {
  int16_t ys = 1;
  foreach_active(i) { ys = y; }
  int16_t q = x / ys;
  int16_t r = x % ys;
  return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}

SCALAR_FUN_ATTR int32_t sdiv32(int32_t x, int32_t y) {
  int32_t ys = 1;
  foreach_active(i) { ys = y; }
  int32_t q = x / ys;
  int32_t r = x % ys;
  return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}

SCALAR_FUN_ATTR int64_t sdiv64(int64_t x, int64_t y) {
  int64_t ys = 1;
  foreach_active(i) { ys = y; }
  int64_t q = x / ys;
  int64_t r = x % ys;
  return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}

SCALAR_FUN_ATTR int8_t sdiv_up8(int8_t x, int8_t y) { return sdiv8(x + y - 1, y); }
SCALAR_FUN_ATTR int16_t sdiv_up16(int16_t x, int16_t y) { return sdiv16(x + y - 1, y); }
SCALAR_FUN_ATTR int32_t sdiv_up32(int32_t x, int32_t y) { return sdiv32(x + y - 1, y); }
SCALAR_FUN_ATTR int64_t sdiv_up64(int64_t x, int64_t y) { return sdiv64(x + y - 1, y); }

SCALAR_FUN_ATTR int8_t smod8(int8_t x, int8_t y) {
  int8_t ys = 1;
  foreach_active(i) { ys = y; }
  int8_t r = x % ys;
  return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}

SCALAR_FUN_ATTR int16_t smod16(int16_t x, int16_t y) {
  int16_t ys = 1;
  foreach_active(i) { ys = y; }
  int16_t r = x % ys;
  return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}

SCALAR_FUN_ATTR int32_t smod32(int32_t x, int32_t y) {
  int32_t ys = 1;
  foreach_active(i) { ys = y; }
  int32_t r = x % ys;
  return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}

SCALAR_FUN_ATTR int64_t smod64(int64_t x, int64_t y) {
  int64_t ys = 1;
  foreach_active(i) { ys = y; }
  int64_t r = x % ys;
  return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}

SCALAR_FUN_ATTR int8_t   sdiv_safe8(int8_t x, int8_t y)   { return y == 0 ? 0 : sdiv8(x, y); }
SCALAR_FUN_ATTR int16_t sdiv_safe16(int16_t x, int16_t y) { return y == 0 ? 0 : sdiv16(x, y); }
SCALAR_FUN_ATTR int32_t sdiv_safe32(int32_t x, int32_t y) { return y == 0 ? 0 : sdiv32(x, y); }
SCALAR_FUN_ATTR int64_t sdiv_safe64(int64_t x, int64_t y) { return y == 0 ? 0 : sdiv64(x, y); }

SCALAR_FUN_ATTR int8_t sdiv_up_safe8(int8_t x, int8_t y)     { return sdiv_safe8(x + y - 1, y); }
SCALAR_FUN_ATTR int16_t sdiv_up_safe16(int16_t x, int16_t y) { return sdiv_safe16(x + y - 1, y); }
SCALAR_FUN_ATTR int32_t sdiv_up_safe32(int32_t x, int32_t y) { return sdiv_safe32(x + y - 1, y); }
SCALAR_FUN_ATTR int64_t sdiv_up_safe64(int64_t x, int64_t y) { return sdiv_safe64(x + y - 1, y); }

SCALAR_FUN_ATTR int8_t   smod_safe8(int8_t x, int8_t y)   { return y == 0 ? 0 : smod8(x, y); }
SCALAR_FUN_ATTR int16_t smod_safe16(int16_t x, int16_t y) { return y == 0 ? 0 : smod16(x, y); }
SCALAR_FUN_ATTR int32_t smod_safe32(int32_t x, int32_t y) { return y == 0 ? 0 : smod32(x, y); }
SCALAR_FUN_ATTR int64_t smod_safe64(int64_t x, int64_t y) { return y == 0 ? 0 : smod64(x, y); }

SCALAR_FUN_ATTR int8_t squot8(int8_t x, int8_t y) {
  int8_t ys = 1;
  foreach_active(i) { ys = y; }
  return x / ys;
}

SCALAR_FUN_ATTR int16_t squot16(int16_t x, int16_t y) {
  int16_t ys = 1;
  foreach_active(i) { ys = y; }
  return x / ys;
}

SCALAR_FUN_ATTR int32_t squot32(int32_t x, int32_t y) {
  int32_t ys = 1;
  foreach_active(i) { ys = y; }
  return x / ys;
}

SCALAR_FUN_ATTR int64_t squot64(int64_t x, int64_t y) {
  int64_t ys = 1;
  foreach_active(i) { ys = y; }
  return x / ys;
}

SCALAR_FUN_ATTR int8_t srem8(int8_t x, int8_t y) {
  int8_t ys = 1;
  foreach_active(i) { ys = y; }
  return x % ys;
}

SCALAR_FUN_ATTR int16_t srem16(int16_t x, int16_t y) {
  int16_t ys = 1;
  foreach_active(i) { ys = y; }
  return x % ys;
}

SCALAR_FUN_ATTR int32_t srem32(int32_t x, int32_t y) {
  int32_t ys = 1;
  foreach_active(i) { ys = y; }
  return x % ys;
}

SCALAR_FUN_ATTR int64_t srem64(int64_t x, int64_t y) {
  int8_t ys = 1;
  foreach_active(i) { ys = y; }
  return x % ys;
}

SCALAR_FUN_ATTR int8_t squot_safe8(int8_t x, int8_t y) {
  int8_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x / ys;
}

SCALAR_FUN_ATTR int16_t squot_safe16(int16_t x, int16_t y) {
  int16_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x / ys;
}

SCALAR_FUN_ATTR int32_t squot_safe32(int32_t x, int32_t y) {
  int32_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x / ys;
}

SCALAR_FUN_ATTR int64_t squot_safe64(int64_t x, int64_t y) {
  int64_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x / ys;
}

SCALAR_FUN_ATTR int8_t srem_safe8(int8_t x, int8_t y) {
  int8_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x % ys;
}

SCALAR_FUN_ATTR int16_t srem_safe16(int16_t x, int16_t y) {
  int16_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x % ys;
}

SCALAR_FUN_ATTR int32_t srem_safe32(int32_t x, int32_t y) {
  int32_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x % ys;
}

SCALAR_FUN_ATTR int64_t srem_safe64(int64_t x, int64_t y) {
  int64_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x % ys;
}

#else

SCALAR_FUN_ATTR uint8_t   udiv8(uint8_t x, uint8_t y)   { return x / y; }
SCALAR_FUN_ATTR uint16_t udiv16(uint16_t x, uint16_t y) { return x / y; }
SCALAR_FUN_ATTR uint32_t udiv32(uint32_t x, uint32_t y) { return x / y; }
SCALAR_FUN_ATTR uint64_t udiv64(uint64_t x, uint64_t y) { return x / y; }

SCALAR_FUN_ATTR uint8_t   udiv_up8(uint8_t x, uint8_t y)   { return (x + y - 1) / y; }
SCALAR_FUN_ATTR uint16_t udiv_up16(uint16_t x, uint16_t y) { return (x + y - 1) / y; }
SCALAR_FUN_ATTR uint32_t udiv_up32(uint32_t x, uint32_t y) { return (x + y - 1) / y; }
SCALAR_FUN_ATTR uint64_t udiv_up64(uint64_t x, uint64_t y) { return (x + y - 1) / y; }

SCALAR_FUN_ATTR uint8_t   umod8(uint8_t x, uint8_t y)   { return x % y; }
SCALAR_FUN_ATTR uint16_t umod16(uint16_t x, uint16_t y) { return x % y; }
SCALAR_FUN_ATTR uint32_t umod32(uint32_t x, uint32_t y) { return x % y; }
SCALAR_FUN_ATTR uint64_t umod64(uint64_t x, uint64_t y) { return x % y; }

SCALAR_FUN_ATTR uint8_t   udiv_safe8(uint8_t x, uint8_t y)   { return y == 0 ? 0 : x / y; }
SCALAR_FUN_ATTR uint16_t udiv_safe16(uint16_t x, uint16_t y) { return y == 0 ? 0 : x / y; }
SCALAR_FUN_ATTR uint32_t udiv_safe32(uint32_t x, uint32_t y) { return y == 0 ? 0 : x / y; }
SCALAR_FUN_ATTR uint64_t udiv_safe64(uint64_t x, uint64_t y) { return y == 0 ? 0 : x / y; }

SCALAR_FUN_ATTR uint8_t   udiv_up_safe8(uint8_t x, uint8_t y)   { return y == 0 ? 0 : (x + y - 1) / y; }
SCALAR_FUN_ATTR uint16_t udiv_up_safe16(uint16_t x, uint16_t y) { return y == 0 ? 0 : (x + y - 1) / y; }
SCALAR_FUN_ATTR uint32_t udiv_up_safe32(uint32_t x, uint32_t y) { return y == 0 ? 0 : (x + y - 1) / y; }
SCALAR_FUN_ATTR uint64_t udiv_up_safe64(uint64_t x, uint64_t y) { return y == 0 ? 0 : (x + y - 1) / y; }

SCALAR_FUN_ATTR uint8_t   umod_safe8(uint8_t x, uint8_t y)   { return y == 0 ? 0 : x % y; }
SCALAR_FUN_ATTR uint16_t umod_safe16(uint16_t x, uint16_t y) { return y == 0 ? 0 : x % y; }
SCALAR_FUN_ATTR uint32_t umod_safe32(uint32_t x, uint32_t y) { return y == 0 ? 0 : x % y; }
SCALAR_FUN_ATTR uint64_t umod_safe64(uint64_t x, uint64_t y) { return y == 0 ? 0 : x % y; }

SCALAR_FUN_ATTR int8_t sdiv8(int8_t x, int8_t y) {
  int8_t q = x / y;
  int8_t r = x % y;
  return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}

SCALAR_FUN_ATTR int16_t sdiv16(int16_t x, int16_t y) {
  int16_t q = x / y;
  int16_t r = x % y;
  return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}

SCALAR_FUN_ATTR int32_t sdiv32(int32_t x, int32_t y) {
  int32_t q = x / y;
  int32_t r = x % y;
  return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}

SCALAR_FUN_ATTR int64_t sdiv64(int64_t x, int64_t y) {
  int64_t q = x / y;
  int64_t r = x % y;
  return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}

SCALAR_FUN_ATTR int8_t   sdiv_up8(int8_t x, int8_t y)   { return sdiv8(x + y - 1, y); }
SCALAR_FUN_ATTR int16_t sdiv_up16(int16_t x, int16_t y) { return sdiv16(x + y - 1, y); }
SCALAR_FUN_ATTR int32_t sdiv_up32(int32_t x, int32_t y) { return sdiv32(x + y - 1, y); }
SCALAR_FUN_ATTR int64_t sdiv_up64(int64_t x, int64_t y) { return sdiv64(x + y - 1, y); }

SCALAR_FUN_ATTR int8_t smod8(int8_t x, int8_t y) {
  int8_t r = x % y;
  return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}

SCALAR_FUN_ATTR int16_t smod16(int16_t x, int16_t y) {
  int16_t r = x % y;
  return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}

SCALAR_FUN_ATTR int32_t smod32(int32_t x, int32_t y) {
  int32_t r = x % y;
  return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}

SCALAR_FUN_ATTR int64_t smod64(int64_t x, int64_t y) {
  int64_t r = x % y;
  return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}

SCALAR_FUN_ATTR int8_t   sdiv_safe8(int8_t x, int8_t y)   { return y == 0 ? 0 : sdiv8(x, y); }
SCALAR_FUN_ATTR int16_t sdiv_safe16(int16_t x, int16_t y) { return y == 0 ? 0 : sdiv16(x, y); }
SCALAR_FUN_ATTR int32_t sdiv_safe32(int32_t x, int32_t y) { return y == 0 ? 0 : sdiv32(x, y); }
SCALAR_FUN_ATTR int64_t sdiv_safe64(int64_t x, int64_t y) { return y == 0 ? 0 : sdiv64(x, y); }

SCALAR_FUN_ATTR int8_t   sdiv_up_safe8(int8_t x, int8_t y)   { return sdiv_safe8(x + y - 1, y);}
SCALAR_FUN_ATTR int16_t sdiv_up_safe16(int16_t x, int16_t y) { return sdiv_safe16(x + y - 1, y); }
SCALAR_FUN_ATTR int32_t sdiv_up_safe32(int32_t x, int32_t y) { return sdiv_safe32(x + y - 1, y); }
SCALAR_FUN_ATTR int64_t sdiv_up_safe64(int64_t x, int64_t y) { return sdiv_safe64(x + y - 1, y); }

SCALAR_FUN_ATTR int8_t   smod_safe8(int8_t x, int8_t y)   { return y == 0 ? 0 : smod8(x, y); }
SCALAR_FUN_ATTR int16_t smod_safe16(int16_t x, int16_t y) { return y == 0 ? 0 : smod16(x, y); }
SCALAR_FUN_ATTR int32_t smod_safe32(int32_t x, int32_t y) { return y == 0 ? 0 : smod32(x, y); }
SCALAR_FUN_ATTR int64_t smod_safe64(int64_t x, int64_t y) { return y == 0 ? 0 : smod64(x, y); }

SCALAR_FUN_ATTR int8_t   squot8(int8_t x, int8_t y)   { return x / y; }
SCALAR_FUN_ATTR int16_t squot16(int16_t x, int16_t y) { return x / y; }
SCALAR_FUN_ATTR int32_t squot32(int32_t x, int32_t y) { return x / y; }
SCALAR_FUN_ATTR int64_t squot64(int64_t x, int64_t y) { return x / y; }

SCALAR_FUN_ATTR int8_t   srem8(int8_t x, int8_t y)   { return x % y; }
SCALAR_FUN_ATTR int16_t srem16(int16_t x, int16_t y) { return x % y; }
SCALAR_FUN_ATTR int32_t srem32(int32_t x, int32_t y) { return x % y; }
SCALAR_FUN_ATTR int64_t srem64(int64_t x, int64_t y) { return x % y; }

SCALAR_FUN_ATTR int8_t   squot_safe8(int8_t x, int8_t y)   { return y == 0 ? 0 : x / y; }
SCALAR_FUN_ATTR int16_t squot_safe16(int16_t x, int16_t y) { return y == 0 ? 0 : x / y; }
SCALAR_FUN_ATTR int32_t squot_safe32(int32_t x, int32_t y) { return y == 0 ? 0 : x / y; }
SCALAR_FUN_ATTR int64_t squot_safe64(int64_t x, int64_t y) { return y == 0 ? 0 : x / y; }

SCALAR_FUN_ATTR int8_t   srem_safe8(int8_t x, int8_t y)   { return y == 0 ? 0 : x % y; }
SCALAR_FUN_ATTR int16_t srem_safe16(int16_t x, int16_t y) { return y == 0 ? 0 : x % y; }
SCALAR_FUN_ATTR int32_t srem_safe32(int32_t x, int32_t y) { return y == 0 ? 0 : x % y; }
SCALAR_FUN_ATTR int64_t srem_safe64(int64_t x, int64_t y) { return y == 0 ? 0 : x % y; }

#endif

SCALAR_FUN_ATTR int8_t   smin8(int8_t x, int8_t y)   { return x < y ? x : y; }
SCALAR_FUN_ATTR int16_t smin16(int16_t x, int16_t y) { return x < y ? x : y; }
SCALAR_FUN_ATTR int32_t smin32(int32_t x, int32_t y) { return x < y ? x : y; }
SCALAR_FUN_ATTR int64_t smin64(int64_t x, int64_t y) { return x < y ? x : y; }

SCALAR_FUN_ATTR uint8_t   umin8(uint8_t x, uint8_t y)   { return x < y ? x : y; }
SCALAR_FUN_ATTR uint16_t umin16(uint16_t x, uint16_t y) { return x < y ? x : y; }
SCALAR_FUN_ATTR uint32_t umin32(uint32_t x, uint32_t y) { return x < y ? x : y; }
SCALAR_FUN_ATTR uint64_t umin64(uint64_t x, uint64_t y) { return x < y ? x : y; }

SCALAR_FUN_ATTR int8_t  smax8(int8_t x, int8_t y)    { return x < y ? y : x; }
SCALAR_FUN_ATTR int16_t smax16(int16_t x, int16_t y) { return x < y ? y : x; }
SCALAR_FUN_ATTR int32_t smax32(int32_t x, int32_t y) { return x < y ? y : x; }
SCALAR_FUN_ATTR int64_t smax64(int64_t x, int64_t y) { return x < y ? y : x; }

SCALAR_FUN_ATTR uint8_t   umax8(uint8_t x, uint8_t y)   { return x < y ? y : x; }
SCALAR_FUN_ATTR uint16_t umax16(uint16_t x, uint16_t y) { return x < y ? y : x; }
SCALAR_FUN_ATTR uint32_t umax32(uint32_t x, uint32_t y) { return x < y ? y : x; }
SCALAR_FUN_ATTR uint64_t umax64(uint64_t x, uint64_t y) { return x < y ? y : x; }

SCALAR_FUN_ATTR uint8_t   shl8(uint8_t x, uint8_t y)   { return (uint8_t)(x << y); }
SCALAR_FUN_ATTR uint16_t shl16(uint16_t x, uint16_t y) { return (uint16_t)(x << y); }
SCALAR_FUN_ATTR uint32_t shl32(uint32_t x, uint32_t y) { return x << y; }
SCALAR_FUN_ATTR uint64_t shl64(uint64_t x, uint64_t y) { return x << y; }

SCALAR_FUN_ATTR uint8_t   lshr8(uint8_t x, uint8_t y)   { return x >> y; }
SCALAR_FUN_ATTR uint16_t lshr16(uint16_t x, uint16_t y) { return x >> y; }
SCALAR_FUN_ATTR uint32_t lshr32(uint32_t x, uint32_t y) { return x >> y; }
SCALAR_FUN_ATTR uint64_t lshr64(uint64_t x, uint64_t y) { return x >> y; }

SCALAR_FUN_ATTR int8_t   ashr8(int8_t x, int8_t y)   { return x >> y; }
SCALAR_FUN_ATTR int16_t ashr16(int16_t x, int16_t y) { return x >> y; }
SCALAR_FUN_ATTR int32_t ashr32(int32_t x, int32_t y) { return x >> y; }
SCALAR_FUN_ATTR int64_t ashr64(int64_t x, int64_t y) { return x >> y; }

SCALAR_FUN_ATTR uint8_t   and8(uint8_t x, uint8_t y)   { return x & y; }
SCALAR_FUN_ATTR uint16_t and16(uint16_t x, uint16_t y) { return x & y; }
SCALAR_FUN_ATTR uint32_t and32(uint32_t x, uint32_t y) { return x & y; }
SCALAR_FUN_ATTR uint64_t and64(uint64_t x, uint64_t y) { return x & y; }

SCALAR_FUN_ATTR uint8_t    or8(uint8_t x, uint8_t y)  { return x | y; }
SCALAR_FUN_ATTR uint16_t or16(uint16_t x, uint16_t y) { return x | y; }
SCALAR_FUN_ATTR uint32_t or32(uint32_t x, uint32_t y) { return x | y; }
SCALAR_FUN_ATTR uint64_t or64(uint64_t x, uint64_t y) { return x | y; }

SCALAR_FUN_ATTR uint8_t   xor8(uint8_t x, uint8_t y)   { return x ^ y; }
SCALAR_FUN_ATTR uint16_t xor16(uint16_t x, uint16_t y) { return x ^ y; }
SCALAR_FUN_ATTR uint32_t xor32(uint32_t x, uint32_t y) { return x ^ y; }
SCALAR_FUN_ATTR uint64_t xor64(uint64_t x, uint64_t y) { return x ^ y; }

SCALAR_FUN_ATTR bool ult8(uint8_t x, uint8_t y)    { return x < y; }
SCALAR_FUN_ATTR bool ult16(uint16_t x, uint16_t y) { return x < y; }
SCALAR_FUN_ATTR bool ult32(uint32_t x, uint32_t y) { return x < y; }
SCALAR_FUN_ATTR bool ult64(uint64_t x, uint64_t y) { return x < y; }

SCALAR_FUN_ATTR bool ule8(uint8_t x, uint8_t y)    { return x <= y; }
SCALAR_FUN_ATTR bool ule16(uint16_t x, uint16_t y) { return x <= y; }
SCALAR_FUN_ATTR bool ule32(uint32_t x, uint32_t y) { return x <= y; }
SCALAR_FUN_ATTR bool ule64(uint64_t x, uint64_t y) { return x <= y; }

SCALAR_FUN_ATTR bool  slt8(int8_t x, int8_t y)   { return x < y; }
SCALAR_FUN_ATTR bool slt16(int16_t x, int16_t y) { return x < y; }
SCALAR_FUN_ATTR bool slt32(int32_t x, int32_t y) { return x < y; }
SCALAR_FUN_ATTR bool slt64(int64_t x, int64_t y) { return x < y; }

SCALAR_FUN_ATTR bool  sle8(int8_t x, int8_t y)   { return x <= y; }
SCALAR_FUN_ATTR bool sle16(int16_t x, int16_t y) { return x <= y; }
SCALAR_FUN_ATTR bool sle32(int32_t x, int32_t y) { return x <= y; }
SCALAR_FUN_ATTR bool sle64(int64_t x, int64_t y) { return x <= y; }

SCALAR_FUN_ATTR uint8_t pow8(uint8_t x, uint8_t y) {
  uint8_t res = 1, rem = y;
  while (rem != 0) {
    if (rem & 1)
      res *= x;
    rem >>= 1;
    x *= x;
  }
  return res;
}

SCALAR_FUN_ATTR uint16_t pow16(uint16_t x, uint16_t y) {
  uint16_t res = 1, rem = y;
  while (rem != 0) {
    if (rem & 1)
      res *= x;
    rem >>= 1;
    x *= x;
  }
  return res;
}

SCALAR_FUN_ATTR uint32_t pow32(uint32_t x, uint32_t y) {
  uint32_t res = 1, rem = y;
  while (rem != 0) {
    if (rem & 1)
      res *= x;
    rem >>= 1;
    x *= x;
  }
  return res;
}

SCALAR_FUN_ATTR uint64_t pow64(uint64_t x, uint64_t y) {
  uint64_t res = 1, rem = y;
  while (rem != 0) {
    if (rem & 1)
      res *= x;
    rem >>= 1;
    x *= x;
  }
  return res;
}

SCALAR_FUN_ATTR bool  itob_i8_bool(int8_t x)  { return x != 0; }
SCALAR_FUN_ATTR bool itob_i16_bool(int16_t x) { return x != 0; }
SCALAR_FUN_ATTR bool itob_i32_bool(int32_t x) { return x != 0; }
SCALAR_FUN_ATTR bool itob_i64_bool(int64_t x) { return x != 0; }

SCALAR_FUN_ATTR int8_t btoi_bool_i8(bool x)   { return x; }
SCALAR_FUN_ATTR int16_t btoi_bool_i16(bool x) { return x; }
SCALAR_FUN_ATTR int32_t btoi_bool_i32(bool x) { return x; }
SCALAR_FUN_ATTR int64_t btoi_bool_i64(bool x) { return x; }

#define sext_i8_i8(x) ((int8_t) (int8_t) (x))
#define sext_i8_i16(x) ((int16_t) (int8_t) (x))
#define sext_i8_i32(x) ((int32_t) (int8_t) (x))
#define sext_i8_i64(x) ((int64_t) (int8_t) (x))
#define sext_i16_i8(x) ((int8_t) (int16_t) (x))
#define sext_i16_i16(x) ((int16_t) (int16_t) (x))
#define sext_i16_i32(x) ((int32_t) (int16_t) (x))
#define sext_i16_i64(x) ((int64_t) (int16_t) (x))
#define sext_i32_i8(x) ((int8_t) (int32_t) (x))
#define sext_i32_i16(x) ((int16_t) (int32_t) (x))
#define sext_i32_i32(x) ((int32_t) (int32_t) (x))
#define sext_i32_i64(x) ((int64_t) (int32_t) (x))
#define sext_i64_i8(x) ((int8_t) (int64_t) (x))
#define sext_i64_i16(x) ((int16_t) (int64_t) (x))
#define sext_i64_i32(x) ((int32_t) (int64_t) (x))
#define sext_i64_i64(x) ((int64_t) (int64_t) (x))
#define zext_i8_i8(x) ((int8_t) (uint8_t) (x))
#define zext_i8_i16(x) ((int16_t) (uint8_t) (x))
#define zext_i8_i32(x) ((int32_t) (uint8_t) (x))
#define zext_i8_i64(x) ((int64_t) (uint8_t) (x))
#define zext_i16_i8(x) ((int8_t) (uint16_t) (x))
#define zext_i16_i16(x) ((int16_t) (uint16_t) (x))
#define zext_i16_i32(x) ((int32_t) (uint16_t) (x))
#define zext_i16_i64(x) ((int64_t) (uint16_t) (x))
#define zext_i32_i8(x) ((int8_t) (uint32_t) (x))
#define zext_i32_i16(x) ((int16_t) (uint32_t) (x))
#define zext_i32_i32(x) ((int32_t) (uint32_t) (x))
#define zext_i32_i64(x) ((int64_t) (uint32_t) (x))
#define zext_i64_i8(x) ((int8_t) (uint64_t) (x))
#define zext_i64_i16(x) ((int16_t) (uint64_t) (x))
#define zext_i64_i32(x) ((int32_t) (uint64_t) (x))
#define zext_i64_i64(x) ((int64_t) (uint64_t) (x))

SCALAR_FUN_ATTR int8_t   abs8(int8_t x)  { return (int8_t)abs(x); }
SCALAR_FUN_ATTR int16_t abs16(int16_t x) { return (int16_t)abs(x); }
SCALAR_FUN_ATTR int32_t abs32(int32_t x) { return abs(x); }
SCALAR_FUN_ATTR int64_t abs64(int64_t x) {
#if defined(__OPENCL_VERSION__) || defined(ISPC)
  return abs(x);
#else
  return llabs(x);
#endif
}

#if defined(__OPENCL_VERSION__)

SCALAR_FUN_ATTR int32_t  futrts_popc8(int8_t x)  { return popcount(x); }
SCALAR_FUN_ATTR int32_t futrts_popc16(int16_t x) { return popcount(x); }
SCALAR_FUN_ATTR int32_t futrts_popc32(int32_t x) { return popcount(x); }
SCALAR_FUN_ATTR int32_t futrts_popc64(int64_t x) { return popcount(x); }

#elif defined(__CUDA_ARCH__)

SCALAR_FUN_ATTR int32_t  futrts_popc8(int8_t x)  { return __popc(zext_i8_i32(x)); }
SCALAR_FUN_ATTR int32_t futrts_popc16(int16_t x) { return __popc(zext_i16_i32(x)); }
SCALAR_FUN_ATTR int32_t futrts_popc32(int32_t x) { return __popc(x); }
SCALAR_FUN_ATTR int32_t futrts_popc64(int64_t x) { return __popcll(x); }

#else // Not OpenCL or CUDA, but plain C.

SCALAR_FUN_ATTR int32_t futrts_popc8(uint8_t x) {
  int c = 0;
  for (; x; ++c) { x &= x - 1; }
  return c;
}

SCALAR_FUN_ATTR int32_t futrts_popc16(uint16_t x) {
  int c = 0;
  for (; x; ++c) { x &= x - 1; }
  return c;
}

SCALAR_FUN_ATTR int32_t futrts_popc32(uint32_t x) {
  int c = 0;
  for (; x; ++c) { x &= x - 1; }
  return c;
}

SCALAR_FUN_ATTR int32_t futrts_popc64(uint64_t x) {
  int c = 0;
  for (; x; ++c) { x &= x - 1; }
  return c;
}
#endif

#if defined(__OPENCL_VERSION__)
SCALAR_FUN_ATTR uint8_t  futrts_umul_hi8 ( uint8_t a,  uint8_t b) { return mul_hi(a, b); }
SCALAR_FUN_ATTR uint16_t futrts_umul_hi16(uint16_t a, uint16_t b) { return mul_hi(a, b); }
SCALAR_FUN_ATTR uint32_t futrts_umul_hi32(uint32_t a, uint32_t b) { return mul_hi(a, b); }
SCALAR_FUN_ATTR uint64_t futrts_umul_hi64(uint64_t a, uint64_t b) { return mul_hi(a, b); }
SCALAR_FUN_ATTR uint8_t  futrts_smul_hi8 ( int8_t a,  int8_t b) { return mul_hi(a, b); }
SCALAR_FUN_ATTR uint16_t futrts_smul_hi16(int16_t a, int16_t b) { return mul_hi(a, b); }
SCALAR_FUN_ATTR uint32_t futrts_smul_hi32(int32_t a, int32_t b) { return mul_hi(a, b); }
SCALAR_FUN_ATTR uint64_t futrts_smul_hi64(int64_t a, int64_t b) { return mul_hi(a, b); }
#elif defined(__CUDA_ARCH__)
SCALAR_FUN_ATTR  uint8_t futrts_umul_hi8(uint8_t a, uint8_t b) { return ((uint16_t)a) * ((uint16_t)b) >> 8; }
SCALAR_FUN_ATTR uint16_t futrts_umul_hi16(uint16_t a, uint16_t b) { return ((uint32_t)a) * ((uint32_t)b) >> 16; }
SCALAR_FUN_ATTR uint32_t futrts_umul_hi32(uint32_t a, uint32_t b) { return __umulhi(a, b); }
SCALAR_FUN_ATTR uint64_t futrts_umul_hi64(uint64_t a, uint64_t b) { return __umul64hi(a, b); }
SCALAR_FUN_ATTR  uint8_t futrts_smul_hi8 ( int8_t a, int8_t b) { return ((int16_t)a) * ((int16_t)b) >> 8; }
SCALAR_FUN_ATTR uint16_t futrts_smul_hi16(int16_t a, int16_t b) { return ((int32_t)a) * ((int32_t)b) >> 16; }
SCALAR_FUN_ATTR uint32_t futrts_smul_hi32(int32_t a, int32_t b) { return __mulhi(a, b); }
SCALAR_FUN_ATTR uint64_t futrts_smul_hi64(int64_t a, int64_t b) { return __mul64hi(a, b); }
#elif defined(ISPC)
SCALAR_FUN_ATTR uint8_t futrts_umul_hi8(uint8_t a, uint8_t b) { return ((uint16_t)a) * ((uint16_t)b) >> 8; }
SCALAR_FUN_ATTR uint16_t futrts_umul_hi16(uint16_t a, uint16_t b) { return ((uint32_t)a) * ((uint32_t)b) >> 16; }
SCALAR_FUN_ATTR uint32_t futrts_umul_hi32(uint32_t a, uint32_t b) { return ((uint64_t)a) * ((uint64_t)b) >> 32; }
SCALAR_FUN_ATTR uint64_t futrts_umul_hi64(uint64_t a, uint64_t b) {
  uint64_t ah = a >> 32;
  uint64_t al = a & 0xffffffff;
  uint64_t bh = b >> 32;
  uint64_t bl = b & 0xffffffff;

  uint64_t p1 = al * bl;
  uint64_t p2 = al * bh;
  uint64_t p3 = ah * bl;
  uint64_t p4 = ah * bh;

  uint64_t p1h = p1 >> 32;
  uint64_t p2h = p2 >> 32;
  uint64_t p3h = p3 >> 32;
  uint64_t p2l = p2 & 0xffffffff;
  uint64_t p3l = p3 & 0xffffffff;

  uint64_t l = p1h + p2l + p3l;
  uint64_t m = (p2 >> 32) + (p3 >> 32);
  uint64_t h = (l >> 32) + m + p4;

  return h;
}
SCALAR_FUN_ATTR  int8_t futrts_smul_hi8 ( int8_t a,  int8_t b) { return ((uint16_t)a) * ((uint16_t)b) >> 8; }
SCALAR_FUN_ATTR int16_t futrts_smul_hi16(int16_t a, int16_t b) { return ((uint32_t)a) * ((uint32_t)b) >> 16; }
SCALAR_FUN_ATTR int32_t futrts_smul_hi32(int32_t a, int32_t b) { return ((uint64_t)a) * ((uint64_t)b) >> 32; }
SCALAR_FUN_ATTR int64_t futrts_smul_hi64(int64_t a, int64_t b) {
  uint64_t ah = a >> 32;
  uint64_t al = a & 0xffffffff;
  uint64_t bh = b >> 32;
  uint64_t bl = b & 0xffffffff;

  uint64_t p1 =  al * bl;
  int64_t  p2 = al * bh;
  int64_t  p3 = ah * bl;
  uint64_t p4 =  ah * bh;

  uint64_t p1h = p1 >> 32;
  uint64_t p2h = p2 >> 32;
  uint64_t p3h = p3 >> 32;
  uint64_t p2l = p2 & 0xffffffff;
  uint64_t p3l = p3 & 0xffffffff;

  uint64_t l = p1h + p2l + p3l;
  uint64_t m = (p2 >> 32) + (p3 >> 32);
  uint64_t h = (l >> 32) + m + p4;

  return h;
}

#else // Not OpenCL, ISPC, or CUDA, but plain C.
SCALAR_FUN_ATTR uint8_t futrts_umul_hi8(uint8_t a, uint8_t b) { return ((uint16_t)a) * ((uint16_t)b) >> 8; }
SCALAR_FUN_ATTR uint16_t futrts_umul_hi16(uint16_t a, uint16_t b) { return ((uint32_t)a) * ((uint32_t)b) >> 16; }
SCALAR_FUN_ATTR uint32_t futrts_umul_hi32(uint32_t a, uint32_t b) { return ((uint64_t)a) * ((uint64_t)b) >> 32; }
SCALAR_FUN_ATTR uint64_t futrts_umul_hi64(uint64_t a, uint64_t b) { return ((__uint128_t)a) * ((__uint128_t)b) >> 64; }
SCALAR_FUN_ATTR int8_t futrts_smul_hi8(int8_t a, int8_t b) { return ((int16_t)a) * ((int16_t)b) >> 8; }
SCALAR_FUN_ATTR int16_t futrts_smul_hi16(int16_t a, int16_t b) { return ((int32_t)a) * ((int32_t)b) >> 16; }
SCALAR_FUN_ATTR int32_t futrts_smul_hi32(int32_t a, int32_t b) { return ((int64_t)a) * ((int64_t)b) >> 32; }
SCALAR_FUN_ATTR int64_t futrts_smul_hi64(int64_t a, int64_t b) { return ((__int128_t)a) * ((__int128_t)b) >> 64; }
#endif

#if defined(__OPENCL_VERSION__)
SCALAR_FUN_ATTR  uint8_t futrts_umad_hi8 ( uint8_t a,  uint8_t b,  uint8_t c) { return mad_hi(a, b, c); }
SCALAR_FUN_ATTR uint16_t futrts_umad_hi16(uint16_t a, uint16_t b, uint16_t c) { return mad_hi(a, b, c); }
SCALAR_FUN_ATTR uint32_t futrts_umad_hi32(uint32_t a, uint32_t b, uint32_t c) { return mad_hi(a, b, c); }
SCALAR_FUN_ATTR uint64_t futrts_umad_hi64(uint64_t a, uint64_t b, uint64_t c) { return mad_hi(a, b, c); }
SCALAR_FUN_ATTR  uint8_t futrts_smad_hi8( int8_t a,  int8_t b,   int8_t c) { return mad_hi(a, b, c); }
SCALAR_FUN_ATTR uint16_t futrts_smad_hi16(int16_t a, int16_t b, int16_t c) { return mad_hi(a, b, c); }
SCALAR_FUN_ATTR uint32_t futrts_smad_hi32(int32_t a, int32_t b, int32_t c) { return mad_hi(a, b, c); }
SCALAR_FUN_ATTR uint64_t futrts_smad_hi64(int64_t a, int64_t b, int64_t c) { return mad_hi(a, b, c); }
#else // Not OpenCL

SCALAR_FUN_ATTR  uint8_t futrts_umad_hi8( uint8_t a,  uint8_t b,  uint8_t c) { return futrts_umul_hi8(a, b) + c; }
SCALAR_FUN_ATTR uint16_t futrts_umad_hi16(uint16_t a, uint16_t b, uint16_t c) { return futrts_umul_hi16(a, b) + c; }
SCALAR_FUN_ATTR uint32_t futrts_umad_hi32(uint32_t a, uint32_t b, uint32_t c) { return futrts_umul_hi32(a, b) + c; }
SCALAR_FUN_ATTR uint64_t futrts_umad_hi64(uint64_t a, uint64_t b, uint64_t c) { return futrts_umul_hi64(a, b) + c; }
SCALAR_FUN_ATTR  uint8_t futrts_smad_hi8 ( int8_t a,  int8_t b,  int8_t c) { return futrts_smul_hi8(a, b) + c; }
SCALAR_FUN_ATTR uint16_t futrts_smad_hi16(int16_t a, int16_t b, int16_t c) { return futrts_smul_hi16(a, b) + c; }
SCALAR_FUN_ATTR uint32_t futrts_smad_hi32(int32_t a, int32_t b, int32_t c) { return futrts_smul_hi32(a, b) + c; }
SCALAR_FUN_ATTR uint64_t futrts_smad_hi64(int64_t a, int64_t b, int64_t c) { return futrts_smul_hi64(a, b) + c; }
#endif

#if defined(__OPENCL_VERSION__)
SCALAR_FUN_ATTR int32_t  futrts_clzz8(int8_t x)  { return clz(x); }
SCALAR_FUN_ATTR int32_t futrts_clzz16(int16_t x) { return clz(x); }
SCALAR_FUN_ATTR int32_t futrts_clzz32(int32_t x) { return clz(x); }
SCALAR_FUN_ATTR int32_t futrts_clzz64(int64_t x) { return clz(x); }

#elif defined(__CUDA_ARCH__)

SCALAR_FUN_ATTR int32_t  futrts_clzz8(int8_t x)  { return __clz(zext_i8_i32(x)) - 24; }
SCALAR_FUN_ATTR int32_t futrts_clzz16(int16_t x) { return __clz(zext_i16_i32(x)) - 16; }
SCALAR_FUN_ATTR int32_t futrts_clzz32(int32_t x) { return __clz(x); }
SCALAR_FUN_ATTR int32_t futrts_clzz64(int64_t x) { return __clzll(x); }

#elif defined(ISPC)

SCALAR_FUN_ATTR int32_t  futrts_clzz8(int8_t x)  { return count_leading_zeros((int32_t)(uint8_t)x)-24; }
SCALAR_FUN_ATTR int32_t futrts_clzz16(int16_t x) { return count_leading_zeros((int32_t)(uint16_t)x)-16; }
SCALAR_FUN_ATTR int32_t futrts_clzz32(int32_t x) { return count_leading_zeros(x); }
SCALAR_FUN_ATTR int32_t futrts_clzz64(int64_t x) { return count_leading_zeros(x); }

#else // Not OpenCL, ISPC or CUDA, but plain C.

SCALAR_FUN_ATTR int32_t futrts_clzz8(int8_t x)
{ return x == 0 ? 8 : __builtin_clz((uint32_t)zext_i8_i32(x)) - 24; }
SCALAR_FUN_ATTR int32_t futrts_clzz16(int16_t x)
{ return x == 0 ? 16 : __builtin_clz((uint32_t)zext_i16_i32(x)) - 16; }
SCALAR_FUN_ATTR int32_t futrts_clzz32(int32_t x)
{ return x == 0 ? 32 : __builtin_clz((uint32_t)x); }
SCALAR_FUN_ATTR int32_t futrts_clzz64(int64_t x)
{ return x == 0 ? 64 : __builtin_clzll((uint64_t)x); }
#endif

#if defined(__OPENCL_VERSION__)
SCALAR_FUN_ATTR int32_t futrts_ctzz8(int8_t x) {
  int i = 0;
  for (; i < 8 && (x & 1) == 0; i++, x >>= 1) ;
  return i;
}

SCALAR_FUN_ATTR int32_t futrts_ctzz16(int16_t x) {
  int i = 0;
  for (; i < 16 && (x & 1) == 0; i++, x >>= 1) ;
  return i;
}

SCALAR_FUN_ATTR int32_t futrts_ctzz32(int32_t x) {
  int i = 0;
  for (; i < 32 && (x & 1) == 0; i++, x >>= 1) ;
  return i;
}

SCALAR_FUN_ATTR int32_t futrts_ctzz64(int64_t x) {
  int i = 0;
  for (; i < 64 && (x & 1) == 0; i++, x >>= 1) ;
  return i;
}

#elif defined(__CUDA_ARCH__)

SCALAR_FUN_ATTR int32_t futrts_ctzz8(int8_t x) {
  int y = __ffs(x);
  return y == 0 ? 8 : y - 1;
}

SCALAR_FUN_ATTR int32_t futrts_ctzz16(int16_t x) {
  int y = __ffs(x);
  return y == 0 ? 16 : y - 1;
}

SCALAR_FUN_ATTR int32_t futrts_ctzz32(int32_t x) {
  int y = __ffs(x);
  return y == 0 ? 32 : y - 1;
}

SCALAR_FUN_ATTR int32_t futrts_ctzz64(int64_t x) {
  int y = __ffsll(x);
  return y == 0 ? 64 : y - 1;
}

#elif defined(ISPC)

SCALAR_FUN_ATTR int32_t futrts_ctzz8(int8_t x) { return x == 0 ? 8 : count_trailing_zeros((int32_t)x); }
SCALAR_FUN_ATTR int32_t futrts_ctzz16(int16_t x) { return x == 0 ? 16 : count_trailing_zeros((int32_t)x); }
SCALAR_FUN_ATTR int32_t futrts_ctzz32(int32_t x) { return count_trailing_zeros(x); }
SCALAR_FUN_ATTR int32_t futrts_ctzz64(int64_t x) { return count_trailing_zeros(x); }

#else // Not OpenCL or CUDA, but plain C.

SCALAR_FUN_ATTR int32_t  futrts_ctzz8(int8_t x)  { return x == 0 ? 8 : __builtin_ctz((uint32_t)x); }
SCALAR_FUN_ATTR int32_t futrts_ctzz16(int16_t x) { return x == 0 ? 16 : __builtin_ctz((uint32_t)x); }
SCALAR_FUN_ATTR int32_t futrts_ctzz32(int32_t x) { return x == 0 ? 32 : __builtin_ctz((uint32_t)x); }
SCALAR_FUN_ATTR int32_t futrts_ctzz64(int64_t x) { return x == 0 ? 64 : __builtin_ctzll((uint64_t)x); }
#endif

SCALAR_FUN_ATTR float fdiv32(float x, float y) { return x / y; }
SCALAR_FUN_ATTR float fadd32(float x, float y) { return x + y; }
SCALAR_FUN_ATTR float fsub32(float x, float y) { return x - y; }
SCALAR_FUN_ATTR float fmul32(float x, float y) { return x * y; }
SCALAR_FUN_ATTR bool cmplt32(float x, float y) { return x < y; }
SCALAR_FUN_ATTR bool cmple32(float x, float y) { return x <= y; }
SCALAR_FUN_ATTR float sitofp_i8_f32(int8_t x)  { return (float) x; }

SCALAR_FUN_ATTR float sitofp_i16_f32(int16_t x) { return (float) x; }
SCALAR_FUN_ATTR float sitofp_i32_f32(int32_t x) { return (float) x; }
SCALAR_FUN_ATTR float sitofp_i64_f32(int64_t x) { return (float) x; }
SCALAR_FUN_ATTR float  uitofp_i8_f32(uint8_t x)  { return (float) x; }
SCALAR_FUN_ATTR float uitofp_i16_f32(uint16_t x) { return (float) x; }
SCALAR_FUN_ATTR float uitofp_i32_f32(uint32_t x) { return (float) x; }
SCALAR_FUN_ATTR float uitofp_i64_f32(uint64_t x) { return (float) x; }

#ifdef __OPENCL_VERSION__
SCALAR_FUN_ATTR float fabs32(float x)          { return fabs(x); }
SCALAR_FUN_ATTR float fmax32(float x, float y) { return fmax(x, y); }
SCALAR_FUN_ATTR float fmin32(float x, float y) { return fmin(x, y); }
SCALAR_FUN_ATTR float fpow32(float x, float y) { return pow(x, y); }

#elif defined(ISPC)

SCALAR_FUN_ATTR float fabs32(float x) { return abs(x); }
SCALAR_FUN_ATTR float fmax32(float x, float y) { return isnan(x) ? y : isnan(y) ? x : max(x, y); }
SCALAR_FUN_ATTR float fmin32(float x, float y) { return isnan(x) ? y : isnan(y) ? x : min(x, y); }
SCALAR_FUN_ATTR float fpow32(float a, float b) {
  float ret;
  foreach_active (i) {
      uniform float r = pow(extract(a, i), extract(b, i));
      ret = insert(ret, i, r);
  }
  return ret;
}

#else // Not OpenCL, but CUDA or plain C.

SCALAR_FUN_ATTR float fabs32(float x)          { return fabsf(x); }
SCALAR_FUN_ATTR float fmax32(float x, float y) { return fmaxf(x, y); }
SCALAR_FUN_ATTR float fmin32(float x, float y) { return fminf(x, y); }
SCALAR_FUN_ATTR float fpow32(float x, float y) { return powf(x, y); }
#endif

SCALAR_FUN_ATTR bool futrts_isnan32(float x) { return isnan(x); }

#if defined(ISPC)

SCALAR_FUN_ATTR bool futrts_isinf32(float x) { return !isnan(x) && isnan(x - x); }

SCALAR_FUN_ATTR bool futrts_isfinite32(float x) { return !isnan(x) && !futrts_isinf32(x); }

#else

SCALAR_FUN_ATTR bool futrts_isinf32(float x) { return isinf(x); }

#endif

SCALAR_FUN_ATTR int8_t fptosi_f32_i8(float x) {
  if (futrts_isnan32(x) || futrts_isinf32(x)) {
    return 0;
  } else {
    return (int8_t) x;
  }
}

SCALAR_FUN_ATTR int16_t fptosi_f32_i16(float x) {
  if (futrts_isnan32(x) || futrts_isinf32(x)) {
    return 0;
  } else {
    return (int16_t) x;
  }
}

SCALAR_FUN_ATTR int32_t fptosi_f32_i32(float x) {
  if (futrts_isnan32(x) || futrts_isinf32(x)) {
    return 0;
  } else {
    return (int32_t) x;
  }
}

SCALAR_FUN_ATTR int64_t fptosi_f32_i64(float x) {
  if (futrts_isnan32(x) || futrts_isinf32(x)) {
    return 0;
  } else {
    return (int64_t) x;
  };
}

SCALAR_FUN_ATTR uint8_t fptoui_f32_i8(float x) {
  if (futrts_isnan32(x) || futrts_isinf32(x)) {
    return 0;
  } else {
    return (uint8_t) (int8_t) x;
  }
}

SCALAR_FUN_ATTR uint16_t fptoui_f32_i16(float x) {
  if (futrts_isnan32(x) || futrts_isinf32(x)) {
    return 0;
  } else {
    return (uint16_t) (int16_t) x;
  }
}

SCALAR_FUN_ATTR uint32_t fptoui_f32_i32(float x) {
  if (futrts_isnan32(x) || futrts_isinf32(x)) {
    return 0;
  } else {
    return (uint32_t) (int32_t) x;
  }
}

SCALAR_FUN_ATTR uint64_t fptoui_f32_i64(float x) {
  if (futrts_isnan32(x) || futrts_isinf32(x)) {
    return 0;
  } else {
    return (uint64_t) (int64_t) x;
  }
}

SCALAR_FUN_ATTR bool ftob_f32_bool(float x) { return x != 0; }
SCALAR_FUN_ATTR float btof_bool_f32(bool x) { return x ? 1 : 0; }

#ifdef __OPENCL_VERSION__
SCALAR_FUN_ATTR float futrts_log32(float x) { return log(x); }
SCALAR_FUN_ATTR float futrts_log2_32(float x) { return log2(x); }
SCALAR_FUN_ATTR float futrts_log10_32(float x) { return log10(x); }
SCALAR_FUN_ATTR float futrts_log1p_32(float x) { return log1p(x); }
SCALAR_FUN_ATTR float futrts_sqrt32(float x) { return sqrt(x); }
SCALAR_FUN_ATTR float futrts_rsqrt32(float x) { return rsqrt(x); }
SCALAR_FUN_ATTR float futrts_cbrt32(float x) { return cbrt(x); }
SCALAR_FUN_ATTR float futrts_exp32(float x) { return exp(x); }
SCALAR_FUN_ATTR float futrts_cos32(float x) { return cos(x); }
SCALAR_FUN_ATTR float futrts_cospi32(float x) { return cospi(x); }
SCALAR_FUN_ATTR float futrts_sin32(float x) { return sin(x); }
SCALAR_FUN_ATTR float futrts_sinpi32(float x) { return sinpi(x); }
SCALAR_FUN_ATTR float futrts_tan32(float x) { return tan(x); }
SCALAR_FUN_ATTR float futrts_tanpi32(float x) { return tanpi(x); }
SCALAR_FUN_ATTR float futrts_acos32(float x) { return acos(x); }
SCALAR_FUN_ATTR float futrts_acospi32(float x) { return acospi(x); }
SCALAR_FUN_ATTR float futrts_asin32(float x) { return asin(x); }
SCALAR_FUN_ATTR float futrts_asinpi32(float x) { return asinpi(x); }
SCALAR_FUN_ATTR float futrts_atan32(float x) { return atan(x); }
SCALAR_FUN_ATTR float futrts_atanpi32(float x) { return atanpi(x); }
SCALAR_FUN_ATTR float futrts_cosh32(float x) { return cosh(x); }
SCALAR_FUN_ATTR float futrts_sinh32(float x) { return sinh(x); }
SCALAR_FUN_ATTR float futrts_tanh32(float x) { return tanh(x); }
SCALAR_FUN_ATTR float futrts_acosh32(float x) { return acosh(x); }
SCALAR_FUN_ATTR float futrts_asinh32(float x) { return asinh(x); }
SCALAR_FUN_ATTR float futrts_atanh32(float x) { return atanh(x); }
SCALAR_FUN_ATTR float futrts_atan2_32(float x, float y) { return atan2(x, y); }
SCALAR_FUN_ATTR float futrts_atan2pi_32(float x, float y) { return atan2pi(x, y); }
SCALAR_FUN_ATTR float futrts_hypot32(float x, float y) { return hypot(x, y); }
SCALAR_FUN_ATTR float futrts_gamma32(float x) { return tgamma(x); }
SCALAR_FUN_ATTR float futrts_lgamma32(float x) { return lgamma(x); }
SCALAR_FUN_ATTR float futrts_erf32(float x) { return erf(x); }
SCALAR_FUN_ATTR float futrts_erfc32(float x) { return erfc(x); }
SCALAR_FUN_ATTR float fmod32(float x, float y) { return fmod(x, y); }
SCALAR_FUN_ATTR float futrts_round32(float x) { return rint(x); }
SCALAR_FUN_ATTR float futrts_floor32(float x) { return floor(x); }
SCALAR_FUN_ATTR float futrts_ceil32(float x) { return ceil(x); }
SCALAR_FUN_ATTR float futrts_nextafter32(float x, float y) { return nextafter(x, y); }
SCALAR_FUN_ATTR float futrts_lerp32(float v0, float v1, float t) { return mix(v0, v1, t); }
SCALAR_FUN_ATTR float futrts_ldexp32(float x, int32_t y) { return ldexp(x, y); }
SCALAR_FUN_ATTR float futrts_copysign32(float x, float y) { return copysign(x, y); }
SCALAR_FUN_ATTR float futrts_mad32(float a, float b, float c) { return mad(a, b, c); }
SCALAR_FUN_ATTR float futrts_fma32(float a, float b, float c) { return fma(a, b, c); }

#elif defined(ISPC)

SCALAR_FUN_ATTR float futrts_log32(float x) { return futrts_isfinite32(x) || (futrts_isinf32(x) && x < 0)? log(x) : x; }
SCALAR_FUN_ATTR float futrts_log2_32(float x) { return futrts_log32(x) / log(2.0f); }
SCALAR_FUN_ATTR float futrts_log10_32(float x) { return futrts_log32(x) / log(10.0f); }

SCALAR_FUN_ATTR float futrts_log1p_32(float x) {
  if(x == -1.0f || (futrts_isinf32(x) && x > 0.0f)) return x / 0.0f;
  float y = 1.0f + x;
  float z = y - 1.0f;
  return log(y) - (z-x)/y;
}

SCALAR_FUN_ATTR float futrts_sqrt32(float x) { return sqrt(x); }
SCALAR_FUN_ATTR float futrts_rsqrt32(float x) { return 1/sqrt(x); }

extern "C" unmasked uniform float cbrtf(uniform float);
SCALAR_FUN_ATTR float futrts_cbrt32(float x) {
  float res;
  foreach_active (i) {
    uniform float r = cbrtf(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}

SCALAR_FUN_ATTR float futrts_exp32(float x) { return exp(x); }
SCALAR_FUN_ATTR float futrts_cos32(float x) { return cos(x); }
SCALAR_FUN_ATTR float futrts_cospi32(float x) { return cos((float)M_PI*x); }
SCALAR_FUN_ATTR float futrts_sin32(float x) { return sin(x); }
SCALAR_FUN_ATTR float futrts_sinpi32(float x) { return sin(M_PI*x); }
SCALAR_FUN_ATTR float futrts_tan32(float x) { return tan(x); }
SCALAR_FUN_ATTR float futrts_tanpi32(float x) { return tan((float)M_PI*x); }
SCALAR_FUN_ATTR float futrts_acos32(float x) { return acos(x); }
SCALAR_FUN_ATTR float futrts_acospi32(float x) { return acos(x)/(float)M_PI; }
SCALAR_FUN_ATTR float futrts_asin32(float x) { return asin(x); }
SCALAR_FUN_ATTR float futrts_asinpi32(float x) { return asin(x)/(float)M_PI; }
SCALAR_FUN_ATTR float futrts_atan32(float x) { return atan(x); }
SCALAR_FUN_ATTR float futrts_atanpi32(float x) { return atan(x)/(float)M_PI; }
SCALAR_FUN_ATTR float futrts_cosh32(float x) { return (exp(x)+exp(-x)) / 2.0f; }
SCALAR_FUN_ATTR float futrts_sinh32(float x) { return (exp(x)-exp(-x)) / 2.0f; }
SCALAR_FUN_ATTR float futrts_tanh32(float x) { return futrts_sinh32(x)/futrts_cosh32(x); }

SCALAR_FUN_ATTR float futrts_acosh32(float x) {
  float f = x+sqrt(x*x-1);
  if (futrts_isfinite32(f)) return log(f);
  return f;
}

SCALAR_FUN_ATTR float futrts_asinh32(float x) {
  float f = x+sqrt(x*x+1);
  if (futrts_isfinite32(f)) return log(f);
  return f;
}

SCALAR_FUN_ATTR float futrts_atanh32(float x) {
  float f = (1+x)/(1-x);
  if (futrts_isfinite32(f)) return log(f)/2.0f;
  return f;
}

SCALAR_FUN_ATTR float futrts_atan2_32(float x, float y)
{ return (x == 0.0f && y == 0.0f) ? 0.0f : atan2(x, y); }
SCALAR_FUN_ATTR float futrts_atan2pi_32(float x, float y)
{ return (x == 0.0f && y == 0.0f) ? 0.0f : atan2(x, y) / (float)M_PI; }

SCALAR_FUN_ATTR float futrts_hypot32(float x, float y) {
  if (futrts_isfinite32(x) && futrts_isfinite32(y)) {
    x = abs(x);
    y = abs(y);
    float a;
    float b;
    if (x >= y){
        a = x;
        b = y;
    } else {
        a = y;
        b = x;
    }
    if(b == 0){
      return a;
    }

    int e;
    float an;
    float bn;
    an = frexp (a, &e);
    bn = ldexp (b, - e);
    float cn;
    cn = sqrt (an * an + bn * bn);
    return ldexp (cn, e);
  } else {
    if (futrts_isinf32(x) || futrts_isinf32(y)) return INFINITY;
    else return x + y;
  }

}

extern "C" unmasked uniform float tgammaf(uniform float x);
SCALAR_FUN_ATTR float futrts_gamma32(float x) {
  float res;
  foreach_active (i) {
    uniform float r = tgammaf(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}

extern "C" unmasked uniform float lgammaf(uniform float x);
SCALAR_FUN_ATTR float futrts_lgamma32(float x) {
  float res;
  foreach_active (i) {
    uniform float r = lgammaf(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}

extern "C" unmasked uniform float erff(uniform float x);
SCALAR_FUN_ATTR float futrts_erf32(float x) {
  float res;
  foreach_active (i) {
    uniform float r = erff(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}

extern "C" unmasked uniform float erfcf(uniform float x);
SCALAR_FUN_ATTR float futrts_erfc32(float x) {
  float res;
  foreach_active (i) {
    uniform float r = erfcf(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}

SCALAR_FUN_ATTR float fmod32(float x, float y) { return x - y * trunc(x/y); }
SCALAR_FUN_ATTR float futrts_round32(float x) { return round(x); }
SCALAR_FUN_ATTR float futrts_floor32(float x) { return floor(x); }
SCALAR_FUN_ATTR float futrts_ceil32(float x) { return ceil(x); }

extern "C" unmasked uniform float nextafterf(uniform float x, uniform float y);
SCALAR_FUN_ATTR float futrts_nextafter32(float x, float y) {
  float res;
  foreach_active (i) {
    uniform float r = nextafterf(extract(x, i), extract(y, i));
    res = insert(res, i, r);
  }
  return res;
}

SCALAR_FUN_ATTR float futrts_lerp32(float v0, float v1, float t) {
  return v0 + (v1 - v0) * t;
}

SCALAR_FUN_ATTR float futrts_ldexp32(float x, int32_t y) {
  return x * pow((uniform float)2.0, (float)y);
}

SCALAR_FUN_ATTR float futrts_copysign32(float x, float y) {
  int32_t xb = fptobits_f32_i32(x);
  int32_t yb = fptobits_f32_i32(y);
  return bitstofp_i32_f32((xb & ~(1<<31)) | (yb & (1<<31)));
}

SCALAR_FUN_ATTR float futrts_mad32(float a, float b, float c) {
  return a * b + c;
}

SCALAR_FUN_ATTR float futrts_fma32(float a, float b, float c) {
  return a * b + c;
}

#else // Not OpenCL or ISPC, but CUDA or plain C.

SCALAR_FUN_ATTR float futrts_log32(float x) { return logf(x); }
SCALAR_FUN_ATTR float futrts_log2_32(float x) { return log2f(x); }
SCALAR_FUN_ATTR float futrts_log10_32(float x) { return log10f(x); }
SCALAR_FUN_ATTR float futrts_log1p_32(float x) { return log1pf(x); }
SCALAR_FUN_ATTR float futrts_sqrt32(float x) { return sqrtf(x); }
SCALAR_FUN_ATTR float futrts_rsqrt32(float x) { return 1/sqrtf(x); }
SCALAR_FUN_ATTR float futrts_cbrt32(float x) { return cbrtf(x); }
SCALAR_FUN_ATTR float futrts_exp32(float x) { return expf(x); }
SCALAR_FUN_ATTR float futrts_cos32(float x) { return cosf(x); }

SCALAR_FUN_ATTR float futrts_cospi32(float x) {
#if defined(__CUDA_ARCH__)
  return cospif(x);
#else
  return cosf(((float)M_PI)*x);
#endif
}
SCALAR_FUN_ATTR float futrts_sin32(float x) { return sinf(x); }

SCALAR_FUN_ATTR float futrts_sinpi32(float x) {
#if defined(__CUDA_ARCH__)
  return sinpif(x);
#else
  return sinf((float)M_PI*x);
#endif
}

SCALAR_FUN_ATTR float futrts_tan32(float x) { return tanf(x); }
SCALAR_FUN_ATTR float futrts_tanpi32(float x) { return tanf((float)M_PI*x); }
SCALAR_FUN_ATTR float futrts_acos32(float x) { return acosf(x); }
SCALAR_FUN_ATTR float futrts_acospi32(float x) { return acosf(x)/(float)M_PI; }
SCALAR_FUN_ATTR float futrts_asin32(float x) { return asinf(x); }
SCALAR_FUN_ATTR float futrts_asinpi32(float x) { return asinf(x)/(float)M_PI; }
SCALAR_FUN_ATTR float futrts_atan32(float x) { return atanf(x); }
SCALAR_FUN_ATTR float futrts_atanpi32(float x) { return atanf(x)/(float)M_PI; }
SCALAR_FUN_ATTR float futrts_cosh32(float x) { return coshf(x); }
SCALAR_FUN_ATTR float futrts_sinh32(float x) { return sinhf(x); }
SCALAR_FUN_ATTR float futrts_tanh32(float x) { return tanhf(x); }
SCALAR_FUN_ATTR float futrts_acosh32(float x) { return acoshf(x); }
SCALAR_FUN_ATTR float futrts_asinh32(float x) { return asinhf(x); }
SCALAR_FUN_ATTR float futrts_atanh32(float x) { return atanhf(x); }
SCALAR_FUN_ATTR float futrts_atan2_32(float x, float y) { return atan2f(x, y); }
SCALAR_FUN_ATTR float futrts_atan2pi_32(float x, float y) { return atan2f(x, y) / (float)M_PI; }
SCALAR_FUN_ATTR float futrts_hypot32(float x, float y) { return hypotf(x, y); }
SCALAR_FUN_ATTR float futrts_gamma32(float x) { return tgammaf(x); }
SCALAR_FUN_ATTR float futrts_lgamma32(float x) { return lgammaf(x); }
SCALAR_FUN_ATTR float futrts_erf32(float x) { return erff(x); }
SCALAR_FUN_ATTR float futrts_erfc32(float x) { return erfcf(x); }
SCALAR_FUN_ATTR float fmod32(float x, float y) { return fmodf(x, y); }
SCALAR_FUN_ATTR float futrts_round32(float x) { return rintf(x); }
SCALAR_FUN_ATTR float futrts_floor32(float x) { return floorf(x); }
SCALAR_FUN_ATTR float futrts_ceil32(float x) { return ceilf(x); }
SCALAR_FUN_ATTR float futrts_nextafter32(float x, float y) { return nextafterf(x, y); }
SCALAR_FUN_ATTR float futrts_lerp32(float v0, float v1, float t) { return v0 + (v1 - v0) * t; }
SCALAR_FUN_ATTR float futrts_ldexp32(float x, int32_t y) { return ldexpf(x, y); }
SCALAR_FUN_ATTR float futrts_copysign32(float x, float y) { return copysignf(x, y); }
SCALAR_FUN_ATTR float futrts_mad32(float a, float b, float c) { return a * b + c; }
SCALAR_FUN_ATTR float futrts_fma32(float a, float b, float c) { return fmaf(a, b, c); }

#endif

#if defined(ISPC)

SCALAR_FUN_ATTR int32_t fptobits_f32_i32(float x) { return intbits(x); }
SCALAR_FUN_ATTR float bitstofp_i32_f32(int32_t x) { return floatbits(x); }
SCALAR_FUN_ATTR uniform int32_t fptobits_f32_i32(uniform float x) { return intbits(x); }
SCALAR_FUN_ATTR uniform float bitstofp_i32_f32(uniform int32_t x) { return floatbits(x); }

#else

SCALAR_FUN_ATTR int32_t fptobits_f32_i32(float x) {
  union {
    float f;
    int32_t t;
  } p;

  p.f = x;
  return p.t;
}

SCALAR_FUN_ATTR float bitstofp_i32_f32(int32_t x) {
  union {
    int32_t f;
    float t;
  } p;

  p.f = x;
  return p.t;
}
#endif

SCALAR_FUN_ATTR float fsignum32(float x) {
  return futrts_isnan32(x) ? x : (x > 0 ? 1 : 0) - (x < 0 ? 1 : 0);
}

#ifdef FUTHARK_F64_ENABLED

SCALAR_FUN_ATTR double bitstofp_i64_f64(int64_t x);
SCALAR_FUN_ATTR int64_t fptobits_f64_i64(double x);

#if defined(ISPC)

SCALAR_FUN_ATTR bool futrts_isinf64(double x) { return !isnan(x) && isnan(x - x); }
SCALAR_FUN_ATTR bool futrts_isfinite64(double x) { return !isnan(x) && !futrts_isinf64(x); }
SCALAR_FUN_ATTR double fdiv64(double x, double y) { return x / y; }
SCALAR_FUN_ATTR double fadd64(double x, double y) { return x + y; }
SCALAR_FUN_ATTR double fsub64(double x, double y) { return x - y; }
SCALAR_FUN_ATTR double fmul64(double x, double y) { return x * y; }
SCALAR_FUN_ATTR bool cmplt64(double x, double y) { return x < y; }
SCALAR_FUN_ATTR bool cmple64(double x, double y) { return x <= y; }
SCALAR_FUN_ATTR double sitofp_i8_f64(int8_t x) { return (double) x; }
SCALAR_FUN_ATTR double sitofp_i16_f64(int16_t x) { return (double) x; }
SCALAR_FUN_ATTR double sitofp_i32_f64(int32_t x) { return (double) x; }
SCALAR_FUN_ATTR double sitofp_i64_f64(int64_t x) { return (double) x; }
SCALAR_FUN_ATTR double uitofp_i8_f64(uint8_t x) { return (double) x; }
SCALAR_FUN_ATTR double uitofp_i16_f64(uint16_t x) { return (double) x; }
SCALAR_FUN_ATTR double uitofp_i32_f64(uint32_t x) { return (double) x; }
SCALAR_FUN_ATTR double uitofp_i64_f64(uint64_t x) { return (double) x; }
SCALAR_FUN_ATTR double fabs64(double x) { return abs(x); }
SCALAR_FUN_ATTR double fmax64(double x, double y) { return isnan(x) ? y : isnan(y) ? x : max(x, y); }
SCALAR_FUN_ATTR double fmin64(double x, double y) { return isnan(x) ? y : isnan(y) ? x : min(x, y); }

SCALAR_FUN_ATTR double fpow64(double a, double b) {
  float ret;
  foreach_active (i) {
      uniform float r = pow(extract(a, i), extract(b, i));
      ret = insert(ret, i, r);
  }
  return ret;
}
SCALAR_FUN_ATTR double futrts_log64(double x) { return futrts_isfinite64(x) || (futrts_isinf64(x) && x < 0)? log(x) : x; }
SCALAR_FUN_ATTR double futrts_log2_64(double x) { return futrts_log64(x)/log(2.0d); }
SCALAR_FUN_ATTR double futrts_log10_64(double x) { return futrts_log64(x)/log(10.0d); }

SCALAR_FUN_ATTR double futrts_log1p_64(double x) {
  if(x == -1.0d || (futrts_isinf64(x) && x > 0.0d)) return x / 0.0d;
  double y = 1.0d + x;
  double z = y - 1.0d;
  return log(y) - (z-x)/y;
}

SCALAR_FUN_ATTR double futrts_sqrt64(double x) { return sqrt(x); }
SCALAR_FUN_ATTR double futrts_rsqrt64(double x) { return 1/sqrt(x); }

SCALAR_FUN_ATTR double futrts_cbrt64(double x) {
  double res;
  foreach_active (i) {
    uniform double r = cbrtf(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}
SCALAR_FUN_ATTR double futrts_exp64(double x) { return exp(x); }
SCALAR_FUN_ATTR double futrts_cos64(double x) { return cos(x); }
SCALAR_FUN_ATTR double futrts_cospi64(double x) { return cos(M_PI*x); }
SCALAR_FUN_ATTR double futrts_sin64(double x) { return sin(x); }
SCALAR_FUN_ATTR double futrts_sinpi64(double x) { return sin(M_PI*x); }
SCALAR_FUN_ATTR double futrts_tan64(double x) { return tan(x); }
SCALAR_FUN_ATTR double futrts_tanpi64(double x) { return tan(M_PI*x); }
SCALAR_FUN_ATTR double futrts_acos64(double x) { return acos(x); }
SCALAR_FUN_ATTR double futrts_acospi64(double x) { return acos(x)/M_PI; }
SCALAR_FUN_ATTR double futrts_asin64(double x) { return asin(x); }
SCALAR_FUN_ATTR double futrts_asinpi64(double x) { return asin(x)/M_PI; }
SCALAR_FUN_ATTR double futrts_atan64(double x) { return atan(x); }
SCALAR_FUN_ATTR double futrts_atanpi64(double x) { return atan(x)/M_PI; }
SCALAR_FUN_ATTR double futrts_cosh64(double x) { return (exp(x)+exp(-x)) / 2.0d; }
SCALAR_FUN_ATTR double futrts_sinh64(double x) { return (exp(x)-exp(-x)) / 2.0d; }
SCALAR_FUN_ATTR double futrts_tanh64(double x) { return futrts_sinh64(x)/futrts_cosh64(x); }

SCALAR_FUN_ATTR double futrts_acosh64(double x) {
  double f = x+sqrt(x*x-1.0d);
  if(futrts_isfinite64(f)) return log(f);
  return f;
}

SCALAR_FUN_ATTR double futrts_asinh64(double x) {
  double f = x+sqrt(x*x+1.0d);
  if(futrts_isfinite64(f)) return log(f);
  return f;
}

SCALAR_FUN_ATTR double futrts_atanh64(double x) {
  double f = (1.0d+x)/(1.0d-x);
  if(futrts_isfinite64(f)) return log(f)/2.0d;
  return f;
}
SCALAR_FUN_ATTR double futrts_atan2_64(double x, double y) { return atan2(x, y); }

SCALAR_FUN_ATTR double futrts_atan2pi_64(double x, double y) { return atan2(x, y) / M_PI; }

extern "C" unmasked uniform double hypot(uniform double x, uniform double y);
SCALAR_FUN_ATTR double futrts_hypot64(double x, double y) {
  double res;
  foreach_active (i) {
    uniform double r = hypot(extract(x, i), extract(y, i));
    res = insert(res, i, r);
  }
  return res;
}

extern "C" unmasked uniform double tgamma(uniform double x);
SCALAR_FUN_ATTR double futrts_gamma64(double x) {
  double res;
  foreach_active (i) {
    uniform double r = tgamma(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}

extern "C" unmasked uniform double lgamma(uniform double x);
SCALAR_FUN_ATTR double futrts_lgamma64(double x) {
  double res;
  foreach_active (i) {
    uniform double r = lgamma(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}

extern "C" unmasked uniform double erf(uniform double x);
SCALAR_FUN_ATTR double futrts_erf64(double x) {
  double res;
  foreach_active (i) {
    uniform double r = erf(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}

extern "C" unmasked uniform double erfc(uniform double x);
SCALAR_FUN_ATTR double futrts_erfc64(double x) {
  double res;
  foreach_active (i) {
    uniform double r = erfc(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}

SCALAR_FUN_ATTR double futrts_fma64(double a, double b, double c) { return a * b + c; }
SCALAR_FUN_ATTR double futrts_round64(double x) { return round(x); }
SCALAR_FUN_ATTR double futrts_ceil64(double x) { return ceil(x); }

extern "C" unmasked uniform double nextafter(uniform float x, uniform double y);
SCALAR_FUN_ATTR double futrts_nextafter64(double x, double y) {
  double res;
  foreach_active (i) {
    uniform double r = nextafter(extract(x, i), extract(y, i));
    res = insert(res, i, r);
  }
  return res;
}

SCALAR_FUN_ATTR double futrts_floor64(double x) { return floor(x); }
SCALAR_FUN_ATTR bool futrts_isnan64(double x) { return isnan(x); }

SCALAR_FUN_ATTR int8_t fptosi_f64_i8(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (int8_t) x;
  }
}

SCALAR_FUN_ATTR int16_t fptosi_f64_i16(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (int16_t) x;
  }
}

SCALAR_FUN_ATTR int32_t fptosi_f64_i32(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (int32_t) x;
  }
}

SCALAR_FUN_ATTR int64_t fptosi_f64_i64(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (int64_t) x;
  }
}

SCALAR_FUN_ATTR uint8_t fptoui_f64_i8(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (uint8_t) (int8_t) x;
  }
}

SCALAR_FUN_ATTR uint16_t fptoui_f64_i16(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (uint16_t) (int16_t) x;
  }
}

SCALAR_FUN_ATTR uint32_t fptoui_f64_i32(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (uint32_t) (int32_t) x;
  }
}

SCALAR_FUN_ATTR uint64_t fptoui_f64_i64(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (uint64_t) (int64_t) x;
  }
}

SCALAR_FUN_ATTR bool ftob_f64_bool(double x) { return x != 0.0; }
SCALAR_FUN_ATTR double btof_bool_f64(bool x) { return x ? 1.0 : 0.0; }

SCALAR_FUN_ATTR int64_t fptobits_f64_i64(double x) {
  int64_t res;
  foreach_active (i) {
    uniform double tmp = extract(x, i);
    uniform int64_t r = *((uniform int64_t* uniform)&tmp);
    res = insert(res, i, r);
  }
  return res;
}

SCALAR_FUN_ATTR double bitstofp_i64_f64(int64_t x) {
  double res;
  foreach_active (i) {
    uniform int64_t tmp = extract(x, i);
    uniform double r = *((uniform double* uniform)&tmp);
    res = insert(res, i, r);
  }
  return res;
}

SCALAR_FUN_ATTR uniform int64_t fptobits_f64_i64(uniform double x) {
  return intbits(x);
}

SCALAR_FUN_ATTR uniform double bitstofp_i64_f64(uniform int64_t x) {
  return doublebits(x);
}

SCALAR_FUN_ATTR double fmod64(double x, double y) {
  return x - y * trunc(x/y);
}

SCALAR_FUN_ATTR double fsignum64(double x) {
  return futrts_isnan64(x) ? x : (x > 0 ? 1.0d : 0.0d) - (x < 0 ? 1.0d : 0.0d);
}

SCALAR_FUN_ATTR double futrts_lerp64(double v0, double v1, double t) {
  return v0 + (v1 - v0) * t;
}

SCALAR_FUN_ATTR double futrts_ldexp64(double x, int32_t y) {
  return x * pow((uniform double)2.0, (double)y);
}

SCALAR_FUN_ATTR double futrts_copysign64(double x, double y) {
  int64_t xb = fptobits_f64_i64(x);
  int64_t yb = fptobits_f64_i64(y);
  return bitstofp_i64_f64((xb & ~(((int64_t)1)<<63)) | (yb & (((int64_t)1)<<63)));
}

SCALAR_FUN_ATTR double futrts_mad64(double a, double b, double c) { return a * b + c; }
SCALAR_FUN_ATTR float fpconv_f32_f32(float x) { return (float) x; }
SCALAR_FUN_ATTR double fpconv_f32_f64(float x) { return (double) x; }
SCALAR_FUN_ATTR float fpconv_f64_f32(double x) { return (float) x; }
SCALAR_FUN_ATTR double fpconv_f64_f64(double x) { return (double) x; }

#else

SCALAR_FUN_ATTR double fdiv64(double x, double y) { return x / y; }
SCALAR_FUN_ATTR double fadd64(double x, double y) { return x + y; }
SCALAR_FUN_ATTR double fsub64(double x, double y) { return x - y; }
SCALAR_FUN_ATTR double fmul64(double x, double y) { return x * y; }
SCALAR_FUN_ATTR bool cmplt64(double x, double y) { return x < y; }
SCALAR_FUN_ATTR bool cmple64(double x, double y) { return x <= y; }
SCALAR_FUN_ATTR double sitofp_i8_f64(int8_t x) { return (double) x; }
SCALAR_FUN_ATTR double sitofp_i16_f64(int16_t x) { return (double) x; }
SCALAR_FUN_ATTR double sitofp_i32_f64(int32_t x) { return (double) x; }
SCALAR_FUN_ATTR double sitofp_i64_f64(int64_t x) { return (double) x; }
SCALAR_FUN_ATTR double uitofp_i8_f64(uint8_t x) { return (double) x; }
SCALAR_FUN_ATTR double uitofp_i16_f64(uint16_t x) { return (double) x; }
SCALAR_FUN_ATTR double uitofp_i32_f64(uint32_t x) { return (double) x; }
SCALAR_FUN_ATTR double uitofp_i64_f64(uint64_t x) { return (double) x; }
SCALAR_FUN_ATTR double fabs64(double x) { return fabs(x); }
SCALAR_FUN_ATTR double fmax64(double x, double y) { return fmax(x, y); }
SCALAR_FUN_ATTR double fmin64(double x, double y) { return fmin(x, y); }
SCALAR_FUN_ATTR double fpow64(double x, double y) { return pow(x, y); }
SCALAR_FUN_ATTR double futrts_log64(double x) { return log(x); }
SCALAR_FUN_ATTR double futrts_log2_64(double x) { return log2(x); }
SCALAR_FUN_ATTR double futrts_log10_64(double x) { return log10(x); }
SCALAR_FUN_ATTR double futrts_log1p_64(double x) { return log1p(x); }
SCALAR_FUN_ATTR double futrts_sqrt64(double x) { return sqrt(x); }
SCALAR_FUN_ATTR double futrts_rsqrt64(double x) { return 1/sqrt(x); }
SCALAR_FUN_ATTR double futrts_cbrt64(double x) { return cbrt(x); }
SCALAR_FUN_ATTR double futrts_exp64(double x) { return exp(x); }
SCALAR_FUN_ATTR double futrts_cos64(double x) { return cos(x); }

SCALAR_FUN_ATTR double futrts_cospi64(double x) {
#ifdef __OPENCL_VERSION__
  return cospi(x);
#elif defined(__CUDA_ARCH__)
  return cospi(x);
#else
  return cos(M_PI*x);
#endif
}

SCALAR_FUN_ATTR double futrts_sin64(double x) {
  return sin(x);
}

SCALAR_FUN_ATTR double futrts_sinpi64(double x) {
#ifdef __OPENCL_VERSION__
  return sinpi(x);
#elif defined(__CUDA_ARCH__)
  return sinpi(x);
#else
  return sin(M_PI*x);
#endif
}

SCALAR_FUN_ATTR double futrts_tan64(double x) {
  return tan(x);
}

SCALAR_FUN_ATTR double futrts_tanpi64(double x) {
#ifdef __OPENCL_VERSION__
  return tanpi(x);
#else
  return tan(M_PI*x);
#endif
}

SCALAR_FUN_ATTR double futrts_acos64(double x) {
  return acos(x);
}

SCALAR_FUN_ATTR double futrts_acospi64(double x) {
#ifdef __OPENCL_VERSION__
  return acospi(x);
#else
  return acos(x) / M_PI;
#endif
}

SCALAR_FUN_ATTR double futrts_asin64(double x) {
  return asin(x);
}

SCALAR_FUN_ATTR double futrts_asinpi64(double x) {
#ifdef __OPENCL_VERSION__
  return asinpi(x);
#else
  return asin(x) / M_PI;
#endif
}

SCALAR_FUN_ATTR double futrts_atan64(double x) {
  return atan(x);
}

SCALAR_FUN_ATTR double futrts_atanpi64(double x) {
#ifdef __OPENCL_VERSION__
  return atanpi(x);
#else
  return atan(x) / M_PI;
#endif
}

SCALAR_FUN_ATTR double futrts_cosh64(double x) { return cosh(x); }
SCALAR_FUN_ATTR double futrts_sinh64(double x) { return sinh(x); }
SCALAR_FUN_ATTR double futrts_tanh64(double x) { return tanh(x); }
SCALAR_FUN_ATTR double futrts_acosh64(double x) { return acosh(x); }
SCALAR_FUN_ATTR double futrts_asinh64(double x) { return asinh(x); }
SCALAR_FUN_ATTR double futrts_atanh64(double x) { return atanh(x); }
SCALAR_FUN_ATTR double futrts_atan2_64(double x, double y) { return atan2(x, y); }

SCALAR_FUN_ATTR double futrts_atan2pi_64(double x, double y) {
#ifdef __OPENCL_VERSION__
  return atan2pi(x, y);
#else
  return atan2(x, y) / M_PI;
#endif
}

SCALAR_FUN_ATTR double futrts_hypot64(double x, double y) { return hypot(x, y); }
SCALAR_FUN_ATTR double futrts_gamma64(double x) { return tgamma(x); }
SCALAR_FUN_ATTR double futrts_lgamma64(double x) { return lgamma(x); }
SCALAR_FUN_ATTR double futrts_erf64(double x) { return erf(x); }
SCALAR_FUN_ATTR double futrts_erfc64(double x) { return erfc(x); }
SCALAR_FUN_ATTR double futrts_fma64(double a, double b, double c) { return fma(a, b, c); }
SCALAR_FUN_ATTR double futrts_round64(double x) { return rint(x); }
SCALAR_FUN_ATTR double futrts_ceil64(double x) { return ceil(x); }
SCALAR_FUN_ATTR double futrts_nextafter64(double x, double y) { return nextafter(x, y); }
SCALAR_FUN_ATTR double futrts_floor64(double x) { return floor(x); }
SCALAR_FUN_ATTR bool futrts_isnan64(double x) { return isnan(x); }
SCALAR_FUN_ATTR bool futrts_isinf64(double x) { return isinf(x); }

SCALAR_FUN_ATTR int8_t fptosi_f64_i8(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (int8_t) x;
  }
}

SCALAR_FUN_ATTR int16_t fptosi_f64_i16(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (int16_t) x;
  }
}

SCALAR_FUN_ATTR int32_t fptosi_f64_i32(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (int32_t) x;
  }
}

SCALAR_FUN_ATTR int64_t fptosi_f64_i64(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (int64_t) x;
  }
}

SCALAR_FUN_ATTR uint8_t fptoui_f64_i8(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (uint8_t) (int8_t) x;
  }
}

SCALAR_FUN_ATTR uint16_t fptoui_f64_i16(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (uint16_t) (int16_t) x;
  }
}

SCALAR_FUN_ATTR uint32_t fptoui_f64_i32(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (uint32_t) (int32_t) x;
  }
}

SCALAR_FUN_ATTR uint64_t fptoui_f64_i64(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (uint64_t) (int64_t) x;
  }
}

SCALAR_FUN_ATTR bool ftob_f64_bool(double x) { return x != 0; }
SCALAR_FUN_ATTR double btof_bool_f64(bool x) { return x ? 1 : 0; }

SCALAR_FUN_ATTR int64_t fptobits_f64_i64(double x) {
  union {
    double f;
    int64_t t;
  } p;

  p.f = x;
  return p.t;
}

SCALAR_FUN_ATTR double bitstofp_i64_f64(int64_t x) {
  union {
    int64_t f;
    double t;
  } p;

  p.f = x;
  return p.t;
}

SCALAR_FUN_ATTR double fmod64(double x, double y) {
  return fmod(x, y);
}

SCALAR_FUN_ATTR double fsignum64(double x) {
  return futrts_isnan64(x) ? x : (x > 0) - (x < 0);
}

SCALAR_FUN_ATTR double futrts_lerp64(double v0, double v1, double t) {
#ifdef __OPENCL_VERSION__
  return mix(v0, v1, t);
#else
  return v0 + (v1 - v0) * t;
#endif
}

SCALAR_FUN_ATTR double futrts_ldexp64(double x, int32_t y) {
  return ldexp(x, y);
}

SCALAR_FUN_ATTR double futrts_copysign64(double x, double y) {
  return copysign(x, y);
}

SCALAR_FUN_ATTR double futrts_mad64(double a, double b, double c) {
#ifdef __OPENCL_VERSION__
  return mad(a, b, c);
#else
  return a * b + c;
#endif
}

SCALAR_FUN_ATTR float fpconv_f32_f32(float x) { return (float) x; }
SCALAR_FUN_ATTR double fpconv_f32_f64(float x) { return (double) x; }
SCALAR_FUN_ATTR float fpconv_f64_f32(double x) { return (float) x; }
SCALAR_FUN_ATTR double fpconv_f64_f64(double x) { return (double) x; }

#endif

#endif

#define futrts_cond_f16(x,y,z) ((x) ? (y) : (z))
#define futrts_cond_f32(x,y,z) ((x) ? (y) : (z))
#define futrts_cond_f64(x,y,z) ((x) ? (y) : (z))

#define futrts_cond_i8(x,y,z) ((x) ? (y) : (z))
#define futrts_cond_i16(x,y,z) ((x) ? (y) : (z))
#define futrts_cond_i32(x,y,z) ((x) ? (y) : (z))
#define futrts_cond_i64(x,y,z) ((x) ? (y) : (z))

#define futrts_cond_bool(x,y,z) ((x) ? (y) : (z))
#define futrts_cond_unit(x,y,z) ((x) ? (y) : (z))

// End of scalar.h.
// Start of scalar_f16.h.

// Half-precision is emulated if needed (e.g. in straight C) with the
// native type used if possible.  The emulation works by typedef'ing
// 'float' to 'f16', and then implementing all operations on single
// precision.  To cut down on duplication, we use the same code for
// those Futhark functions that require just operators or casts.  The
// in-memory representation for arrays will still be 16 bits even
// under emulation, so the compiler will have to be careful when
// generating reads or writes.

#if !defined(cl_khr_fp16) && !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600) && !(defined(ISPC))
#define EMULATE_F16
#endif

#if !defined(EMULATE_F16) && defined(__OPENCL_VERSION__)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#ifdef EMULATE_F16

// Note that the half-precision storage format is still 16 bits - the
// compiler will have to be real careful!
typedef float f16;

#elif defined(ISPC)
typedef float16 f16;

#else

#ifdef __CUDA_ARCH__
#include <cuda_fp16.h>
#endif

typedef half f16;

#endif

// Some of these functions convert to single precision because half
// precision versions are not available.
SCALAR_FUN_ATTR f16 fadd16(f16 x, f16 y) { return x + y; }
SCALAR_FUN_ATTR f16 fsub16(f16 x, f16 y) { return x - y; }
SCALAR_FUN_ATTR f16 fmul16(f16 x, f16 y) { return x * y; }
SCALAR_FUN_ATTR bool cmplt16(f16 x, f16 y) { return x < y; }
SCALAR_FUN_ATTR bool cmple16(f16 x, f16 y) { return x <= y; }
SCALAR_FUN_ATTR f16 sitofp_i8_f16(int8_t x) { return (f16) x; }
SCALAR_FUN_ATTR f16 sitofp_i16_f16(int16_t x) { return (f16) x; }
SCALAR_FUN_ATTR f16 sitofp_i32_f16(int32_t x) { return (f16) x; }
SCALAR_FUN_ATTR f16 sitofp_i64_f16(int64_t x) { return (f16) x; }
SCALAR_FUN_ATTR f16 uitofp_i8_f16(uint8_t x) { return (f16) x; }
SCALAR_FUN_ATTR f16 uitofp_i16_f16(uint16_t x) { return (f16) x; }
SCALAR_FUN_ATTR f16 uitofp_i32_f16(uint32_t x) { return (f16) x; }
SCALAR_FUN_ATTR f16 uitofp_i64_f16(uint64_t x) { return (f16) x; }
SCALAR_FUN_ATTR int8_t fptosi_f16_i8(f16 x) { return (int8_t) (float) x; }
SCALAR_FUN_ATTR int16_t fptosi_f16_i16(f16 x) { return (int16_t) x; }
SCALAR_FUN_ATTR int32_t fptosi_f16_i32(f16 x) { return (int32_t) x; }
SCALAR_FUN_ATTR int64_t fptosi_f16_i64(f16 x) { return (int64_t) x; }
SCALAR_FUN_ATTR uint8_t fptoui_f16_i8(f16 x) { return (uint8_t) (float) x; }
SCALAR_FUN_ATTR uint16_t fptoui_f16_i16(f16 x) { return (uint16_t) x; }
SCALAR_FUN_ATTR uint32_t fptoui_f16_i32(f16 x) { return (uint32_t) x; }
SCALAR_FUN_ATTR uint64_t fptoui_f16_i64(f16 x) { return (uint64_t) x; }
SCALAR_FUN_ATTR bool ftob_f16_bool(f16 x) { return x != (f16)0; }
SCALAR_FUN_ATTR f16 btof_bool_f16(bool x) { return x ? 1 : 0; }

#ifndef EMULATE_F16

SCALAR_FUN_ATTR bool futrts_isnan16(f16 x) { return isnan((float)x); }

#ifdef __OPENCL_VERSION__

SCALAR_FUN_ATTR f16 fabs16(f16 x) { return fabs(x); }
SCALAR_FUN_ATTR f16 fmax16(f16 x, f16 y) { return fmax(x, y); }
SCALAR_FUN_ATTR f16 fmin16(f16 x, f16 y) { return fmin(x, y); }
SCALAR_FUN_ATTR f16 fpow16(f16 x, f16 y) { return pow(x, y); }

#elif defined(ISPC)

SCALAR_FUN_ATTR f16 fabs16(f16 x) { return abs(x); }
SCALAR_FUN_ATTR f16 fmax16(f16 x, f16 y) { return futrts_isnan16(x) ? y : futrts_isnan16(y) ? x : max(x, y); }
SCALAR_FUN_ATTR f16 fmin16(f16 x, f16 y) { return futrts_isnan16(x) ? y : futrts_isnan16(y) ? x : min(x, y); }
SCALAR_FUN_ATTR f16 fpow16(f16 x, f16 y) { return pow(x, y); }

#else // Assuming CUDA.

SCALAR_FUN_ATTR f16 fabs16(f16 x) { return fabsf(x); }
SCALAR_FUN_ATTR f16 fmax16(f16 x, f16 y) { return fmaxf(x, y); }
SCALAR_FUN_ATTR f16 fmin16(f16 x, f16 y) { return fminf(x, y); }
SCALAR_FUN_ATTR f16 fpow16(f16 x, f16 y) { return powf(x, y); }

#endif

#if defined(ISPC)
SCALAR_FUN_ATTR bool futrts_isinf16(float x) { return !futrts_isnan16(x) && futrts_isnan16(x - x); }
SCALAR_FUN_ATTR bool futrts_isfinite16(float x) { return !futrts_isnan16(x) && !futrts_isinf16(x); }
#else
SCALAR_FUN_ATTR bool futrts_isinf16(f16 x) { return isinf((float)x); }
#endif

#ifdef __OPENCL_VERSION__
SCALAR_FUN_ATTR f16 futrts_log16(f16 x) { return log(x); }
SCALAR_FUN_ATTR f16 futrts_log2_16(f16 x) { return log2(x); }
SCALAR_FUN_ATTR f16 futrts_log10_16(f16 x) { return log10(x); }
SCALAR_FUN_ATTR f16 futrts_log1p_16(f16 x) { return log1p(x); }
SCALAR_FUN_ATTR f16 futrts_sqrt16(f16 x) { return sqrt(x); }
SCALAR_FUN_ATTR f16 futrts_rsqrt16(f16 x) { return rsqrt(x); }
SCALAR_FUN_ATTR f16 futrts_cbrt16(f16 x) { return cbrt(x); }
SCALAR_FUN_ATTR f16 futrts_exp16(f16 x) { return exp(x); }
SCALAR_FUN_ATTR f16 futrts_cos16(f16 x) { return cos(x); }
SCALAR_FUN_ATTR f16 futrts_cospi16(f16 x) { return cospi(x); }
SCALAR_FUN_ATTR f16 futrts_sin16(f16 x) { return sin(x); }
SCALAR_FUN_ATTR f16 futrts_sinpi16(f16 x) { return sinpi(x); }
SCALAR_FUN_ATTR f16 futrts_tan16(f16 x) { return tan(x); }
SCALAR_FUN_ATTR f16 futrts_tanpi16(f16 x) { return tanpi(x); }
SCALAR_FUN_ATTR f16 futrts_acos16(f16 x) { return acos(x); }
SCALAR_FUN_ATTR f16 futrts_acospi16(f16 x) { return acospi(x); }
SCALAR_FUN_ATTR f16 futrts_asin16(f16 x) { return asin(x); }
SCALAR_FUN_ATTR f16 futrts_asinpi16(f16 x) { return asinpi(x); }
SCALAR_FUN_ATTR f16 futrts_atan16(f16 x) { return atan(x); }
SCALAR_FUN_ATTR f16 futrts_atanpi16(f16 x) { return atanpi(x); }
SCALAR_FUN_ATTR f16 futrts_cosh16(f16 x) { return cosh(x); }
SCALAR_FUN_ATTR f16 futrts_sinh16(f16 x) { return sinh(x); }
SCALAR_FUN_ATTR f16 futrts_tanh16(f16 x) { return tanh(x); }
SCALAR_FUN_ATTR f16 futrts_acosh16(f16 x) { return acosh(x); }
SCALAR_FUN_ATTR f16 futrts_asinh16(f16 x) { return asinh(x); }
SCALAR_FUN_ATTR f16 futrts_atanh16(f16 x) { return atanh(x); }
SCALAR_FUN_ATTR f16 futrts_atan2_16(f16 x, f16 y) { return atan2(x, y); }
SCALAR_FUN_ATTR f16 futrts_atan2pi_16(f16 x, f16 y) { return atan2pi(x, y); }
SCALAR_FUN_ATTR f16 futrts_hypot16(f16 x, f16 y) { return hypot(x, y); }
SCALAR_FUN_ATTR f16 futrts_gamma16(f16 x) { return tgamma(x); }
SCALAR_FUN_ATTR f16 futrts_lgamma16(f16 x) { return lgamma(x); }
SCALAR_FUN_ATTR f16 futrts_erf16(f16 x) { return erf(x); }
SCALAR_FUN_ATTR f16 futrts_erfc16(f16 x) { return erfc(x); }
SCALAR_FUN_ATTR f16 fmod16(f16 x, f16 y) { return fmod(x, y); }
SCALAR_FUN_ATTR f16 futrts_round16(f16 x) { return rint(x); }
SCALAR_FUN_ATTR f16 futrts_floor16(f16 x) { return floor(x); }
SCALAR_FUN_ATTR f16 futrts_ceil16(f16 x) { return ceil(x); }
SCALAR_FUN_ATTR f16 futrts_nextafter16(f16 x, f16 y) { return nextafter(x, y); }
SCALAR_FUN_ATTR f16 futrts_lerp16(f16 v0, f16 v1, f16 t) { return mix(v0, v1, t); }
SCALAR_FUN_ATTR f16 futrts_ldexp16(f16 x, int32_t y) { return ldexp(x, y); }
SCALAR_FUN_ATTR f16 futrts_copysign16(f16 x, f16 y) { return copysign(x, y); }
SCALAR_FUN_ATTR f16 futrts_mad16(f16 a, f16 b, f16 c) { return mad(a, b, c); }
SCALAR_FUN_ATTR f16 futrts_fma16(f16 a, f16 b, f16 c) { return fma(a, b, c); }

#elif defined(ISPC)

SCALAR_FUN_ATTR f16 futrts_log16(f16 x) { return futrts_isfinite16(x) || (futrts_isinf16(x) && x < 0) ? log(x) : x; }
SCALAR_FUN_ATTR f16 futrts_log2_16(f16 x) { return futrts_log16(x) / log(2.0f16); }
SCALAR_FUN_ATTR f16 futrts_log10_16(f16 x) { return futrts_log16(x) / log(10.0f16); }
SCALAR_FUN_ATTR f16 futrts_log1p_16(f16 x) {
  if(x == -1.0f16 || (futrts_isinf16(x) && x > 0.0f16)) return x / 0.0f16;
  f16 y = 1.0f16 + x;
  f16 z = y - 1.0f16;
  return log(y) - (z-x)/y;
}
SCALAR_FUN_ATTR f16 futrts_sqrt16(f16 x) { return (float16)sqrt((float)x); }
SCALAR_FUN_ATTR f16 futrts_rsqrt16(f16 x) { return (float16)1/sqrt((float)x); }
SCALAR_FUN_ATTR f16 futrts_exp16(f16 x) { return exp(x); }
SCALAR_FUN_ATTR f16 futrts_cos16(f16 x) { return (float16)cos((float)x); }
SCALAR_FUN_ATTR f16 futrts_cospi16(f16 x) { return (float16)cos((float)M_PI*(float)x); }
SCALAR_FUN_ATTR f16 futrts_sin16(f16 x) { return (float16)sin((float)x); }
SCALAR_FUN_ATTR f16 futrts_sinpi16(f16 x) { return (float16)sin((float)M_PI*(float)x); }
SCALAR_FUN_ATTR f16 futrts_tan16(f16 x) { return (float16)tan((float)x); }
SCALAR_FUN_ATTR f16 futrts_tanpi16(f16 x) { return (float16)(tan((float)M_PI*(float)x)); }
SCALAR_FUN_ATTR f16 futrts_acos16(f16 x) { return (float16)acos((float)x); }
SCALAR_FUN_ATTR f16 futrts_acospi16(f16 x) { return (float16)(acos((float)x)/(float)M_PI); }
SCALAR_FUN_ATTR f16 futrts_asin16(f16 x) { return (float16)asin((float)x); }
SCALAR_FUN_ATTR f16 futrts_asinpi16(f16 x) { return (float16)(asin((float)x)/(float)M_PI); }
SCALAR_FUN_ATTR f16 futrts_atan16(f16 x) { return (float16)atan((float)x); }
SCALAR_FUN_ATTR f16 futrts_atanpi16(f16 x) { return (float16)(atan((float)x)/(float)M_PI); }
SCALAR_FUN_ATTR f16 futrts_cosh16(f16 x) { return (exp(x)+exp(-x)) / 2.0f16; }
SCALAR_FUN_ATTR f16 futrts_sinh16(f16 x) { return (exp(x)-exp(-x)) / 2.0f16; }
SCALAR_FUN_ATTR f16 futrts_tanh16(f16 x) { return futrts_sinh16(x)/futrts_cosh16(x); }
SCALAR_FUN_ATTR f16 futrts_acosh16(f16 x) {
  float16 f = x+(float16)sqrt((float)(x*x-1));
  if(futrts_isfinite16(f)) return log(f);
  return f;
}
SCALAR_FUN_ATTR f16 futrts_asinh16(f16 x) {
  float16 f = x+(float16)sqrt((float)(x*x+1));
  if(futrts_isfinite16(f)) return log(f);
  return f;
}
SCALAR_FUN_ATTR f16 futrts_atanh16(f16 x) {
  float16 f = (1+x)/(1-x);
  if(futrts_isfinite16(f)) return log(f)/2.0f16;
  return f;
}
SCALAR_FUN_ATTR f16 futrts_atan2_16(f16 x, f16 y) { return (float16)atan2((float)x, (float)y); }
SCALAR_FUN_ATTR f16 futrts_atan2pi_16(f16 x, f16 y) { return (float16)(atan2((float)x, (float)y)/(float)M_PI); }
SCALAR_FUN_ATTR f16 futrts_hypot16(f16 x, f16 y) { return (float16)futrts_hypot32((float)x, (float)y); }

extern "C" unmasked uniform float tgammaf(uniform float x);
SCALAR_FUN_ATTR f16 futrts_gamma16(f16 x) {
  f16 res;
  foreach_active (i) {
    uniform f16 r = (f16)tgammaf(extract((float)x, i));
    res = insert(res, i, r);
  }
  return res;
}

extern "C" unmasked uniform float lgammaf(uniform float x);
SCALAR_FUN_ATTR f16 futrts_lgamma16(f16 x) {
  f16 res;
  foreach_active (i) {
    uniform f16 r = (f16)lgammaf(extract((float)x, i));
    res = insert(res, i, r);
  }
  return res;
}
SCALAR_FUN_ATTR f16 futrts_cbrt16(f16 x) { return (f16)futrts_cbrt32((float)x); }
SCALAR_FUN_ATTR f16 futrts_erf16(f16 x) { return (f16)futrts_erf32((float)x); }
SCALAR_FUN_ATTR f16 futrts_erfc16(f16 x) { return (f16)futrts_erfc32((float)x); }
SCALAR_FUN_ATTR f16 fmod16(f16 x, f16 y) { return x - y * (float16)trunc((float) (x/y)); }
SCALAR_FUN_ATTR f16 futrts_round16(f16 x) { return (float16)round((float)x); }
SCALAR_FUN_ATTR f16 futrts_floor16(f16 x) { return (float16)floor((float)x); }
SCALAR_FUN_ATTR f16 futrts_ceil16(f16 x) { return (float16)ceil((float)x); }
SCALAR_FUN_ATTR f16 futrts_nextafter16(f16 x, f16 y) { return (float16)futrts_nextafter32((float)x, (float) y); }
SCALAR_FUN_ATTR f16 futrts_lerp16(f16 v0, f16 v1, f16 t) { return v0 + (v1 - v0) * t; }
SCALAR_FUN_ATTR f16 futrts_ldexp16(f16 x, int32_t y) { return futrts_ldexp32((float)x, y); }
SCALAR_FUN_ATTR f16 futrts_copysign16(f16 x, f16 y) { return futrts_copysign32((float)x, y); }
SCALAR_FUN_ATTR f16 futrts_mad16(f16 a, f16 b, f16 c) { return a * b + c; }
SCALAR_FUN_ATTR f16 futrts_fma16(f16 a, f16 b, f16 c) { return a * b + c; }

#else // Assume CUDA.

SCALAR_FUN_ATTR f16 futrts_log16(f16 x) { return hlog(x); }
SCALAR_FUN_ATTR f16 futrts_log2_16(f16 x) { return hlog2(x); }
SCALAR_FUN_ATTR f16 futrts_log10_16(f16 x) { return hlog10(x); }
SCALAR_FUN_ATTR f16 futrts_log1p_16(f16 x) { return (f16)log1pf((float)x); }
SCALAR_FUN_ATTR f16 futrts_sqrt16(f16 x) { return hsqrt(x); }
SCALAR_FUN_ATTR f16 futrts_rsqrt16(f16 x) { return hrsqrt(x); }
SCALAR_FUN_ATTR f16 futrts_cbrt16(f16 x) { return cbrtf(x); }
SCALAR_FUN_ATTR f16 futrts_exp16(f16 x) { return hexp(x); }
SCALAR_FUN_ATTR f16 futrts_cos16(f16 x) { return hcos(x); }
SCALAR_FUN_ATTR f16 futrts_cospi16(f16 x) { return hcos((f16)M_PI*x); }
SCALAR_FUN_ATTR f16 futrts_sin16(f16 x) { return hsin(x); }
SCALAR_FUN_ATTR f16 futrts_sinpi16(f16 x) { return hsin((f16)M_PI*x); }
SCALAR_FUN_ATTR f16 futrts_tan16(f16 x) { return tanf(x); }
SCALAR_FUN_ATTR f16 futrts_tanpi16(f16 x) { return tanf((f16)M_PI*x); }
SCALAR_FUN_ATTR f16 futrts_acos16(f16 x) { return acosf(x); }
SCALAR_FUN_ATTR f16 futrts_acospi16(f16 x) { return (f16)acosf(x)/(f16)M_PI; }
SCALAR_FUN_ATTR f16 futrts_asin16(f16 x) { return asinf(x); }
SCALAR_FUN_ATTR f16 futrts_asinpi16(f16 x) { return (f16)asinf(x)/(f16)M_PI; }
SCALAR_FUN_ATTR f16 futrts_atan16(f16 x) { return (f16)atanf(x); }
SCALAR_FUN_ATTR f16 futrts_atanpi16(f16 x) { return (f16)atanf(x)/(f16)M_PI; }
SCALAR_FUN_ATTR f16 futrts_cosh16(f16 x) { return coshf(x); }
SCALAR_FUN_ATTR f16 futrts_sinh16(f16 x) { return sinhf(x); }
SCALAR_FUN_ATTR f16 futrts_tanh16(f16 x) { return tanhf(x); }
SCALAR_FUN_ATTR f16 futrts_acosh16(f16 x) { return acoshf(x); }
SCALAR_FUN_ATTR f16 futrts_asinh16(f16 x) { return asinhf(x); }
SCALAR_FUN_ATTR f16 futrts_atanh16(f16 x) { return atanhf(x); }
SCALAR_FUN_ATTR f16 futrts_atan2_16(f16 x, f16 y) { return (f16)atan2f(x, y); }
SCALAR_FUN_ATTR f16 futrts_atan2pi_16(f16 x, f16 y) { return (f16)atan2f(x, y)/(f16)M_PI; }
SCALAR_FUN_ATTR f16 futrts_hypot16(f16 x, f16 y) { return hypotf(x, y); }
SCALAR_FUN_ATTR f16 futrts_gamma16(f16 x) { return tgammaf(x); }
SCALAR_FUN_ATTR f16 futrts_lgamma16(f16 x) { return lgammaf(x); }
SCALAR_FUN_ATTR f16 futrts_erf16(f16 x) { return erff(x); }
SCALAR_FUN_ATTR f16 futrts_erfc16(f16 x) { return erfcf(x); }
SCALAR_FUN_ATTR f16 fmod16(f16 x, f16 y) { return fmodf(x, y); }
SCALAR_FUN_ATTR f16 futrts_round16(f16 x) { return rintf(x); }
SCALAR_FUN_ATTR f16 futrts_floor16(f16 x) { return hfloor(x); }
SCALAR_FUN_ATTR f16 futrts_ceil16(f16 x) { return hceil(x); }
SCALAR_FUN_ATTR f16 futrts_nextafter16(f16 x, f16 y) { return __ushort_as_half(halfbitsnextafter(__half_as_ushort(x), __half_as_ushort(y))); }
SCALAR_FUN_ATTR f16 futrts_lerp16(f16 v0, f16 v1, f16 t) { return v0 + (v1 - v0) * t; }
SCALAR_FUN_ATTR f16 futrts_ldexp16(f16 x, int32_t y) { return futrts_ldexp32((float)x, y); }
SCALAR_FUN_ATTR f16 futrts_copysign16(f16 x, f16 y) { return futrts_copysign32((float)x, y); }
SCALAR_FUN_ATTR f16 futrts_mad16(f16 a, f16 b, f16 c) { return a * b + c; }
SCALAR_FUN_ATTR f16 futrts_fma16(f16 a, f16 b, f16 c) { return fmaf(a, b, c); }

#endif

// The CUDA __half type cannot be put in unions for some reason, so we
// use bespoke conversion functions instead.
#ifdef __CUDA_ARCH__
SCALAR_FUN_ATTR int16_t fptobits_f16_i16(f16 x) { return __half_as_ushort(x); }
SCALAR_FUN_ATTR f16 bitstofp_i16_f16(int16_t x) { return __ushort_as_half(x); }
#elif defined(ISPC)
SCALAR_FUN_ATTR int16_t fptobits_f16_i16(f16 x) { varying int16_t y = *((varying int16_t * uniform)&x); return y;
}
SCALAR_FUN_ATTR f16 bitstofp_i16_f16(int16_t x) { varying f16 y = *((varying f16 * uniform)&x); return y; }
#else
SCALAR_FUN_ATTR int16_t fptobits_f16_i16(f16 x) {
  union {
    f16 f;
    int16_t t;
  } p;

  p.f = x;
  return p.t;
}

SCALAR_FUN_ATTR f16 bitstofp_i16_f16(int16_t x) {
  union {
    int16_t f;
    f16 t;
  } p;

  p.f = x;
  return p.t;
}
#endif

#else // No native f16 - emulate.

SCALAR_FUN_ATTR f16 fabs16(f16 x) { return fabs32(x); }
SCALAR_FUN_ATTR f16 fmax16(f16 x, f16 y) { return fmax32(x, y); }
SCALAR_FUN_ATTR f16 fmin16(f16 x, f16 y) { return fmin32(x, y); }
SCALAR_FUN_ATTR f16 fpow16(f16 x, f16 y) { return fpow32(x, y); }
SCALAR_FUN_ATTR bool futrts_isnan16(f16 x) { return futrts_isnan32(x); }
SCALAR_FUN_ATTR bool futrts_isinf16(f16 x) { return futrts_isinf32(x); }
SCALAR_FUN_ATTR f16 futrts_log16(f16 x) { return futrts_log32(x); }
SCALAR_FUN_ATTR f16 futrts_log2_16(f16 x) { return futrts_log2_32(x); }
SCALAR_FUN_ATTR f16 futrts_log10_16(f16 x) { return futrts_log10_32(x); }
SCALAR_FUN_ATTR f16 futrts_log1p_16(f16 x) { return futrts_log1p_32(x); }
SCALAR_FUN_ATTR f16 futrts_sqrt16(f16 x) { return futrts_sqrt32(x); }
SCALAR_FUN_ATTR f16 futrts_rsqrt16(f16 x) { return futrts_rsqrt32(x); }
SCALAR_FUN_ATTR f16 futrts_cbrt16(f16 x) { return futrts_cbrt32(x); }
SCALAR_FUN_ATTR f16 futrts_exp16(f16 x) { return futrts_exp32(x); }
SCALAR_FUN_ATTR f16 futrts_cos16(f16 x) { return futrts_cos32(x); }
SCALAR_FUN_ATTR f16 futrts_cospi16(f16 x) { return futrts_cospi32(x); }
SCALAR_FUN_ATTR f16 futrts_sin16(f16 x) { return futrts_sin32(x); }
SCALAR_FUN_ATTR f16 futrts_sinpi16(f16 x) { return futrts_sinpi32(x); }
SCALAR_FUN_ATTR f16 futrts_tan16(f16 x) { return futrts_tan32(x); }
SCALAR_FUN_ATTR f16 futrts_tanpi16(f16 x) { return futrts_tanpi32(x); }
SCALAR_FUN_ATTR f16 futrts_acos16(f16 x) { return futrts_acos32(x); }
SCALAR_FUN_ATTR f16 futrts_acospi16(f16 x) { return futrts_acospi32(x); }
SCALAR_FUN_ATTR f16 futrts_asin16(f16 x) { return futrts_asin32(x); }
SCALAR_FUN_ATTR f16 futrts_asinpi16(f16 x) { return futrts_asinpi32(x); }
SCALAR_FUN_ATTR f16 futrts_atan16(f16 x) { return futrts_atan32(x); }
SCALAR_FUN_ATTR f16 futrts_atanpi16(f16 x) { return futrts_atanpi32(x); }
SCALAR_FUN_ATTR f16 futrts_cosh16(f16 x) { return futrts_cosh32(x); }
SCALAR_FUN_ATTR f16 futrts_sinh16(f16 x) { return futrts_sinh32(x); }
SCALAR_FUN_ATTR f16 futrts_tanh16(f16 x) { return futrts_tanh32(x); }
SCALAR_FUN_ATTR f16 futrts_acosh16(f16 x) { return futrts_acosh32(x); }
SCALAR_FUN_ATTR f16 futrts_asinh16(f16 x) { return futrts_asinh32(x); }
SCALAR_FUN_ATTR f16 futrts_atanh16(f16 x) { return futrts_atanh32(x); }
SCALAR_FUN_ATTR f16 futrts_atan2_16(f16 x, f16 y) { return futrts_atan2_32(x, y); }
SCALAR_FUN_ATTR f16 futrts_atan2pi_16(f16 x, f16 y) { return futrts_atan2pi_32(x, y); }
SCALAR_FUN_ATTR f16 futrts_hypot16(f16 x, f16 y) { return futrts_hypot32(x, y); }
SCALAR_FUN_ATTR f16 futrts_gamma16(f16 x) { return futrts_gamma32(x); }
SCALAR_FUN_ATTR f16 futrts_lgamma16(f16 x) { return futrts_lgamma32(x); }
SCALAR_FUN_ATTR f16 futrts_erf16(f16 x) { return futrts_erf32(x); }
SCALAR_FUN_ATTR f16 futrts_erfc16(f16 x) { return futrts_erfc32(x); }
SCALAR_FUN_ATTR f16 fmod16(f16 x, f16 y) { return fmod32(x, y); }
SCALAR_FUN_ATTR f16 futrts_round16(f16 x) { return futrts_round32(x); }
SCALAR_FUN_ATTR f16 futrts_floor16(f16 x) { return futrts_floor32(x); }
SCALAR_FUN_ATTR f16 futrts_ceil16(f16 x) { return futrts_ceil32(x); }
SCALAR_FUN_ATTR f16 futrts_nextafter16(f16 x, f16 y) { return halfbits2float(halfbitsnextafter(float2halfbits(x), float2halfbits(y))); }
SCALAR_FUN_ATTR f16 futrts_lerp16(f16 v0, f16 v1, f16 t) { return futrts_lerp32(v0, v1, t); }
SCALAR_FUN_ATTR f16 futrts_ldexp16(f16 x, int32_t y) { return futrts_ldexp32(x, y); }
SCALAR_FUN_ATTR f16 futrts_copysign16(f16 x, f16 y) { return futrts_copysign32((float)x, y); }
SCALAR_FUN_ATTR f16 futrts_mad16(f16 a, f16 b, f16 c) { return futrts_mad32(a, b, c); }
SCALAR_FUN_ATTR f16 futrts_fma16(f16 a, f16 b, f16 c) { return futrts_fma32(a, b, c); }

// Even when we are using an OpenCL that does not support cl_khr_fp16,
// it must still support vload_half for actually creating a
// half-precision number, which can then be efficiently converted to a
// float.  Similarly for vstore_half.
#ifdef __OPENCL_VERSION__

SCALAR_FUN_ATTR int16_t fptobits_f16_i16(f16 x) {
  int16_t y;
  // Violating strict aliasing here.
  vstore_half((float)x, 0, (half*)&y);
  return y;
}

SCALAR_FUN_ATTR f16 bitstofp_i16_f16(int16_t x) {
  return (f16)vload_half(0, (half*)&x);
}

#else
SCALAR_FUN_ATTR int16_t fptobits_f16_i16(f16 x) { return (int16_t)float2halfbits(x); }
SCALAR_FUN_ATTR f16 bitstofp_i16_f16(int16_t x) { return halfbits2float((uint16_t)x); }
SCALAR_FUN_ATTR f16 fsignum16(f16 x) { return futrts_isnan16(x) ? x : (x > 0 ? 1 : 0) - (x < 0 ? 1 : 0); }

#endif

#endif

SCALAR_FUN_ATTR float fpconv_f16_f16(f16 x) { return x; }
SCALAR_FUN_ATTR float fpconv_f16_f32(f16 x) { return x; }
SCALAR_FUN_ATTR f16 fpconv_f32_f16(float x) { return (f16) x; }

#ifdef FUTHARK_F64_ENABLED
SCALAR_FUN_ATTR double fpconv_f16_f64(f16 x) { return (double) x; }
#if defined(ISPC)
SCALAR_FUN_ATTR f16 fpconv_f64_f16(double x) { return (f16) ((float)x); }
#else
SCALAR_FUN_ATTR f16 fpconv_f64_f16(double x) { return (f16) x; }
#endif
#endif

// End of scalar_f16.h.

// Start of context_prototypes.h
//
// Prototypes for the functions in context.h, or that will be called
// from those functions, that need to be available very early.

struct futhark_context_config;
struct futhark_context;

struct tuning_param {
  const char *name;
  const char *var; // Z-encoded name.
  const char *class;
  bool set;
  int64_t val;
};

static void set_error(struct futhark_context* ctx, char *error);

// These are called in context/config new/free functions and contain
// shared setup.  They are generated by the compiler itself.
static int init_constants(struct futhark_context*);
static int free_constants(struct futhark_context*);
static void setup_program(struct futhark_context* ctx);
static void teardown_program(struct futhark_context *ctx);

// Allocate host memory.  Must be freed with host_free().
static void host_alloc(struct futhark_context* ctx, size_t size, const char* tag, size_t* size_out, void** mem_out);
// Allocate memory allocated with host_alloc().
static void host_free(struct futhark_context* ctx, size_t size, const char* tag, void* mem);

// Log that a copy has occurred. The provenance may be NULL, if we do not know
// where this came from.
static void log_copy(struct futhark_context* ctx,
                     const char *kind, const char *provenance,
                     int r,
                     int64_t dst_offset, int64_t dst_strides[r],
                     int64_t src_offset, int64_t src_strides[r],
                     int64_t shape[r]);

static void log_transpose(struct futhark_context* ctx,
                          int64_t k, int64_t m, int64_t n);

static bool lmad_map_tr(int64_t *num_arrays_out, int64_t *n_out, int64_t *m_out,
                        int r,
                        const int64_t dst_strides[r],
                        const int64_t src_strides[r],
                        const int64_t shape[r]);

static bool lmad_contiguous(int r, int64_t strides[r], int64_t shape[r]);

static bool lmad_memcpyable(int r,
                            int64_t dst_strides[r], int64_t src_strides[r], int64_t shape[r]);

static void add_event(struct futhark_context* ctx,
                      const char* name,
                      const char* provenance,
                      struct kvs *kvs,
                      void* data,
                      event_report_fn f);

// Functions that must be defined by the backend.
static void backend_context_config_setup(struct futhark_context_config* cfg);
static void backend_context_config_teardown(struct futhark_context_config* cfg);
static int backend_context_setup(struct futhark_context *ctx);
static void backend_context_teardown(struct futhark_context *ctx);

// End of of context_prototypes.h

struct memblock {
    int *references;
    unsigned char *mem;
    int64_t size;
    const char *desc;
};
struct constants {
    int dummy;
};
#define NUM_TUNING_PARAMS 0
static const char *tuning_param_names[] = {NULL};
static const char *tuning_param_vars[] = {NULL};
static const char *tuning_param_classes[] = {NULL};
static int64_t tuning_param_defaults[] = {0};
static const struct {
                 int dummy;
             } tuning_param_indexes = {0};
// Start of backends/c.h

struct futhark_context_config {
  int in_use;
  int debugging;
  int profiling;
  int logging;
  char *cache_fname;
  struct tuning_param tuning_params[NUM_TUNING_PARAMS];
};

static void backend_context_config_setup(struct futhark_context_config* cfg) {
  (void)cfg;
}

static void backend_context_config_teardown(struct futhark_context_config* cfg) {
  (void)cfg;
}

int futhark_context_config_set_tuning_param(struct futhark_context_config *cfg,
                                            const char *param_name,
                                            size_t new_value) {
  for (int i = 0; i < NUM_TUNING_PARAMS; i++) {
    if (strcmp(param_name, cfg->tuning_params[i].name) == 0) {
      cfg->tuning_params[i].val = new_value;
      cfg->tuning_params[i].set = true;
      return 0;
    }
  }

  return 1;
}

struct futhark_context {
  struct futhark_context_config* cfg;
  int detail_memory;
  int debugging;
  int profiling;
  int profiling_paused;
  int logging;
  lock_t lock;
  char *error;
  lock_t error_lock;
  FILE *log;
  struct constants *constants;
  struct free_list free_list;
  struct event_list event_list;
  int64_t peak_mem_usage_default;
  int64_t cur_mem_usage_default;
  struct program* program;
  bool program_initialised;
};

int backend_context_setup(struct futhark_context* ctx) {
  (void)ctx;
  return 0;
}

void backend_context_teardown(struct futhark_context* ctx) {
  (void)ctx;
}

int futhark_context_sync(struct futhark_context* ctx) {
  (void)ctx;
  return 0;
}

// End of backends/c.h

struct program {
    int dummy;
};
static void setup_program(struct futhark_context *ctx)
{
    (void) ctx;
    
    int error = 0;
    
    (void) error;
    ctx->program = malloc(sizeof(struct program));
}
static void teardown_program(struct futhark_context *ctx)
{
    (void) ctx;
    
    int error = 0;
    
    (void) error;
    free(ctx->program);
}
int memblock_unref(struct futhark_context *ctx, struct memblock *block, const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(ctx->log, "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n", desc, block->desc, "default space", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_default -= block->size;
            host_free(ctx, (size_t) block->size, desc, (void *) block->mem);
            free(block->references);
            if (ctx->detail_memory)
                fprintf(ctx->log, "%lld bytes freed (now allocated: %lld bytes)\n", (long long) block->size, (long long) ctx->cur_mem_usage_default);
        }
        block->references = NULL;
    }
    return 0;
}
int memblock_alloc(struct futhark_context *ctx, struct memblock *block, int64_t size, const char *desc)
{
    if (size < 0)
        futhark_panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n", (long long) size, desc, "default space", ctx->cur_mem_usage_default);
    
    int ret = memblock_unref(ctx, block, desc);
    
    if (ret != FUTHARK_SUCCESS)
        return ret;
    if (ctx->detail_memory)
        fprintf(ctx->log, "Allocating %lld bytes for %s in %s (currently allocated: %lld bytes).\n", (long long) size, desc, "default space", (long long) ctx->cur_mem_usage_default);
    host_alloc(ctx, (size_t) size, desc, (size_t *) &size, (void *) &block->mem);
    if (ctx->error == NULL) {
        block->references = (int *) malloc(sizeof(int));
        *block->references = 1;
        block->size = size;
        block->desc = desc;
        
        long long new_usage = ctx->cur_mem_usage_default + size;
        
        if (ctx->detail_memory)
            fprintf(ctx->log, "Received block of %lld bytes; now allocated: %lld bytes", (long long) block->size, new_usage);
        ctx->cur_mem_usage_default = new_usage;
        if (new_usage > ctx->peak_mem_usage_default) {
            ctx->peak_mem_usage_default = new_usage;
            if (ctx->detail_memory)
                fprintf(ctx->log, " (new peak).\n");
        } else if (ctx->detail_memory)
            fprintf(ctx->log, ".\n");
        return FUTHARK_SUCCESS;
    } else {
        // We are naively assuming that any memory allocation error is due to OOM.
        lock_lock(&ctx->error_lock);
        
        char *old_error = ctx->error;
        
        ctx->error = msgprintf("Failed to allocate memory in %s.\nAttempted allocation: %12lld bytes\nCurrently allocated:  %12lld bytes\n%s", "default space", (long long) size, (long long) ctx->cur_mem_usage_default, old_error);
        free(old_error);
        lock_unlock(&ctx->error_lock);
        return FUTHARK_OUT_OF_MEMORY;
    }
}
int memblock_set(struct futhark_context *ctx, struct memblock *lhs, struct memblock *rhs, const char *lhs_desc)
{
    int ret = memblock_unref(ctx, lhs, lhs_desc);
    
    if (rhs->references != NULL)
        (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
char *futhark_context_report(struct futhark_context *ctx)
{
    if (futhark_context_sync(ctx) != 0)
        return NULL;
    
    struct str_builder builder;
    
    str_builder_init(&builder);
    str_builder_char(&builder, '{');
    str_builder_str(&builder, "\"memory\":{");
    str_builder(&builder, "\"default space\": %lld", (long long) ctx->peak_mem_usage_default);
    str_builder_str(&builder, "},\"events\":[");
    if (report_events_in_list(&ctx->event_list, &builder) != 0) {
        free(builder.str);
        return NULL;
    } else {
        str_builder_str(&builder, "]}");
        return builder.str;
    }
}
int futhark_context_clear_caches(struct futhark_context *ctx)
{
    lock_lock(&ctx->lock);
    ctx->peak_mem_usage_default = 0;
    lock_unlock(&ctx->lock);
    return ctx->error != NULL;
}

// Start of context.h

// Internal functions.

static void set_error(struct futhark_context* ctx, char *error) {
  lock_lock(&ctx->error_lock);
  if (ctx->error == NULL) {
    ctx->error = error;
  } else {
    free(error);
  }
  lock_unlock(&ctx->error_lock);
}

// XXX: should be static, but used in ispc_util.h
void lexical_realloc_error(struct futhark_context* ctx, size_t new_size) {
  set_error(ctx,
            msgprintf("Failed to allocate memory.\nAttempted allocation: %12lld bytes\n",
                      (long long) new_size));
}

static int lexical_realloc(struct futhark_context *ctx,
                           unsigned char **ptr,
                           int64_t *old_size,
                           int64_t new_size) {
  unsigned char *new = realloc(*ptr, (size_t)new_size);
  if (new == NULL) {
    lexical_realloc_error(ctx, new_size);
    return FUTHARK_OUT_OF_MEMORY;
  } else {
    *ptr = new;
    *old_size = new_size;
    return FUTHARK_SUCCESS;
  }
}

static void free_all_in_free_list(struct futhark_context* ctx) {
  fl_mem mem;
  free_list_pack(&ctx->free_list);
  while (free_list_first(&ctx->free_list, (fl_mem*)&mem) == 0) {
    free((void*)mem);
  }
}

static int is_small_alloc(size_t size) {
  return size < 1024*1024;
}

static void host_alloc(struct futhark_context* ctx,
                       size_t size, const char* tag, size_t* size_out, void** mem_out) {
  if (is_small_alloc(size) || free_list_find(&ctx->free_list, size, tag, size_out, (fl_mem*)mem_out) != 0) {
    *size_out = size;
    *mem_out = malloc(size);
  }
}

static void host_free(struct futhark_context* ctx,
                      size_t size, const char* tag, void* mem) {
  // Small allocations are handled by malloc()s own free list.  The
  // threshold here is kind of arbitrary, but seems to work OK.
  // Larger allocations are mmap()ed/munmapped() every time, which is
  // very slow, and Futhark programs tend to use a few very large
  // allocations.
  if (is_small_alloc(size)) {
    free(mem);
  } else {
    free_list_insert(&ctx->free_list, size, (fl_mem)mem, tag);
  }
}

static void add_event(struct futhark_context* ctx,
                      const char* name,
                      const char* provenance,
                      struct kvs *kvs,
                      void* data,
                      event_report_fn f) {
  if (provenance == NULL) {
    provenance = "unknown";
  }
  if (ctx->logging) {
    fprintf(ctx->log, "Event: %s\n  at: %s\n", name, provenance);
    if (kvs) {
      kvs_log(kvs, "  ", ctx->log);
    }
  }
  add_event_to_list(&ctx->event_list, name, provenance, kvs, data, f);
}

char *futhark_context_get_error(struct futhark_context *ctx) {
  char *error = ctx->error;
  ctx->error = NULL;
  return error;
}

void futhark_context_config_set_debugging(struct futhark_context_config *cfg, int flag) {
    cfg->profiling = cfg->logging = cfg->debugging = flag;
}

void futhark_context_config_set_profiling(struct futhark_context_config *cfg, int flag) {
    cfg->profiling = flag;
}

void futhark_context_config_set_logging(struct futhark_context_config *cfg, int flag) {
    cfg->logging = flag;
}

void futhark_context_config_set_cache_file(struct futhark_context_config *cfg, const char *f) {
  cfg->cache_fname = strdup(f);
}

int futhark_get_tuning_param_count(void) {
  return NUM_TUNING_PARAMS;
}

const char *futhark_get_tuning_param_name(int i) {
  return tuning_param_names[i];
}

const char *futhark_get_tuning_param_class(int i) {
    return tuning_param_classes[i];
}

void futhark_context_set_logging_file(struct futhark_context *ctx, FILE *f){
  ctx->log = f;
}

void futhark_context_pause_profiling(struct futhark_context *ctx) {
  ctx->profiling_paused = 1;
}

void futhark_context_unpause_profiling(struct futhark_context *ctx) {
  ctx->profiling_paused = 0;
}

struct futhark_context_config* futhark_context_config_new(void) {
  struct futhark_context_config* cfg = malloc(sizeof(struct futhark_context_config));
  if (cfg == NULL) {
    return NULL;
  }
  cfg->in_use = 0;
  cfg->debugging = 0;
  cfg->profiling = 0;
  cfg->logging = 0;
  cfg->cache_fname = NULL;
  for (int i = 0; i < NUM_TUNING_PARAMS; i++) {
    cfg->tuning_params[i].set = false;
    cfg->tuning_params[i].val = tuning_param_defaults[i];
    cfg->tuning_params[i].name = tuning_param_names[i];
    cfg->tuning_params[i].var = tuning_param_vars[i];
    cfg->tuning_params[i].class = tuning_param_classes[i];
  }
  backend_context_config_setup(cfg);
  return cfg;
}

void futhark_context_config_free(struct futhark_context_config* cfg) {
  assert(!cfg->in_use);
  backend_context_config_teardown(cfg);
  free(cfg->cache_fname);
  free(cfg);
}

struct futhark_context* futhark_context_new(struct futhark_context_config* cfg) {
  struct futhark_context* ctx = malloc(sizeof(struct futhark_context));
  if (ctx == NULL) {
    return NULL;
  }
  assert(!cfg->in_use);
  ctx->cfg = cfg;
  ctx->cfg->in_use = 1;
  ctx->program_initialised = false;
  create_lock(&ctx->error_lock);
  create_lock(&ctx->lock);
  free_list_init(&ctx->free_list);
  event_list_init(&ctx->event_list);
  ctx->peak_mem_usage_default = 0;
  ctx->cur_mem_usage_default = 0;
  ctx->constants = malloc(sizeof(struct constants));
  ctx->debugging = cfg->debugging;
  ctx->logging = cfg->logging;
  ctx->detail_memory = cfg->logging;
  ctx->profiling = cfg->profiling;
  ctx->profiling_paused = 0;
  ctx->error = NULL;
  ctx->log = stderr;
  if (backend_context_setup(ctx) == 0) {
    setup_program(ctx);
    init_constants(ctx);
    ctx->program_initialised = true;
    (void)futhark_context_clear_caches(ctx);
    (void)futhark_context_sync(ctx);
  }
  return ctx;
}

void futhark_context_free(struct futhark_context* ctx) {
  if (ctx->program_initialised) {
    free_constants(ctx);
    teardown_program(ctx);
  }
  backend_context_teardown(ctx);
  free_all_in_free_list(ctx);
  free_list_destroy(&ctx->free_list);
  event_list_free(&ctx->event_list);
  free(ctx->constants);
  free(ctx->error);
  free_lock(&ctx->lock);
  free_lock(&ctx->error_lock);
  ctx->cfg->in_use = 0;
  free(ctx);
}

// End of context.h

// Start of copy.h

// Cache-oblivious map-transpose function.
#define GEN_MAP_TRANSPOSE(NAME, ELEM_TYPE)                              \
  static void map_transpose_##NAME                                      \
  (ELEM_TYPE* dst, ELEM_TYPE* src,                                      \
   int64_t k, int64_t m, int64_t n,                                     \
   int64_t cb, int64_t ce, int64_t rb, int64_t re)                      \
  {                                                                     \
  int32_t r = re - rb;                                                  \
  int32_t c = ce - cb;                                                  \
  if (k == 1) {                                                         \
    if (r <= 64 && c <= 64) {                                           \
      for (int64_t j = 0; j < c; j++) {                                 \
        for (int64_t i = 0; i < r; i++) {                               \
          dst[(j + cb) * n + (i + rb)] = src[(i + rb) * m + (j + cb)];  \
        }                                                               \
      }                                                                 \
    } else if (c <= r) {                                                \
      map_transpose_##NAME(dst, src, k, m, n, cb, ce, rb, rb + r/2);    \
      map_transpose_##NAME(dst, src, k, m, n, cb, ce, rb + r/2, re);    \
    } else {                                                            \
      map_transpose_##NAME(dst, src, k, m, n, cb, cb + c/2, rb, re);    \
      map_transpose_##NAME(dst, src, k, m, n, cb + c/2, ce, rb, re);    \
    }                                                                   \
  } else {                                                              \
  for (int64_t i = 0; i < k; i++) {                                     \
    map_transpose_##NAME(dst + i * m * n, src + i * m * n, 1, m, n, cb, ce, rb, re); \
  }\
} \
}

// Straightforward LMAD copy function.
#define GEN_LMAD_COPY_ELEMENTS(NAME, ELEM_TYPE)                         \
  static void lmad_copy_elements_##NAME(int r,                          \
                                        ELEM_TYPE* dst, int64_t dst_strides[r], \
                                        ELEM_TYPE *src, int64_t src_strides[r], \
                                        int64_t shape[r]) {             \
    if (r == 1) {                                                       \
      for (int i = 0; i < shape[0]; i++) {                              \
        dst[i*dst_strides[0]] = src[i*src_strides[0]];                  \
      }                                                                 \
    } else if (r > 1) {                                                 \
      for (int i = 0; i < shape[0]; i++) {                              \
        lmad_copy_elements_##NAME(r-1,                                  \
                                  dst+i*dst_strides[0], dst_strides+1,  \
                                  src+i*src_strides[0], src_strides+1,  \
                                  shape+1);                             \
      }                                                                 \
    }                                                                   \
  }                                                                     \

// Check whether this LMAD can be seen as a transposed 2D array.  This
// is done by checking every possible splitting point.
static bool lmad_is_tr(int64_t *n_out, int64_t *m_out,
                       int r,
                       const int64_t strides[r],
                       const int64_t shape[r]) {
  for (int i = 1; i < r; i++) {
    int n = 1, m = 1;
    bool ok = true;
    int64_t expected = 1;
    // Check strides before 'i'.
    for (int j = i-1; j >= 0; j--) {
      ok = ok && strides[j] == expected;
      expected *= shape[j];
      n *= shape[j];
    }
    // Check strides after 'i'.
    for (int j = r-1; j >= i; j--) {
      ok = ok && strides[j] == expected;
      expected *= shape[j];
      m *= shape[j];
    }
    if (ok) {
      *n_out = n;
      *m_out = m;
      return true;
    }
  }
  return false;
}

// This function determines whether the a 'dst' LMAD is row-major and
// 'src' LMAD is column-major.  Both LMADs are for arrays of the same
// shape.  Both LMADs are allowed to have additional dimensions "on
// top".  Essentially, this function determines whether a copy from
// 'src' to 'dst' is a "map(transpose)" that we know how to implement
// efficiently.  The LMADs can have arbitrary rank, and the main
// challenge here is checking whether the src LMAD actually
// corresponds to a 2D column-major layout by morally collapsing
// dimensions.  There is a lot of looping here, but the actual trip
// count is going to be very low in practice.
//
// Returns true if this is indeed a map(transpose), and writes the
// number of arrays, and moral array size to appropriate output
// parameters.
static bool lmad_map_tr(int64_t *num_arrays_out, int64_t *n_out, int64_t *m_out,
                        int r,
                        const int64_t dst_strides[r],
                        const int64_t src_strides[r],
                        const int64_t shape[r]) {
  int64_t rowmajor_strides[r];
  rowmajor_strides[r-1] = 1;

  for (int i = r-2; i >= 0; i--) {
    rowmajor_strides[i] = rowmajor_strides[i+1] * shape[i+1];
  }

  // map_r will be the number of mapped dimensions on top.
  int map_r = 0;
  int64_t num_arrays = 1;
  for (int i = 0; i < r; i++) {
    if (dst_strides[i] != rowmajor_strides[i] ||
        src_strides[i] != rowmajor_strides[i]) {
      break;
    } else {
      num_arrays *= shape[i];
      map_r++;
    }
  }

  *num_arrays_out = num_arrays;

  if (r==map_r) {
    return false;
  }

  if (memcmp(&rowmajor_strides[map_r],
             &dst_strides[map_r],
             sizeof(int64_t)*(r-map_r)) == 0) {
    return lmad_is_tr(n_out, m_out, r-map_r, src_strides+map_r, shape+map_r);
  } else if (memcmp(&rowmajor_strides[map_r],
                    &src_strides[map_r],
                    sizeof(int64_t)*(r-map_r)) == 0) {
    return lmad_is_tr(m_out, n_out, r-map_r, dst_strides+map_r, shape+map_r);
  }
  return false;
}

// Check if the strides correspond to row-major strides of *any*
// permutation of the shape.  This is done by recursive search with
// backtracking.  This is worst-case exponential, but hopefully the
// arrays we encounter do not have that many dimensions.
static bool lmad_contiguous_search(int checked, int64_t expected,
                                   int r,
                                   int64_t strides[r], int64_t shape[r], bool used[r]) {
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < r; j++) {
      if (!used[j] && strides[j] == expected && strides[j] >= 0) {
        used[j] = true;
        if (checked+1 == r ||
            lmad_contiguous_search(checked+1, expected * shape[j], r, strides, shape, used)) {
          return true;
        }
        used[j] = false;
      }
    }
  }
  return false;
}

// Does this LMAD correspond to an array with positive strides and no
// holes?
static bool lmad_contiguous(int r, int64_t strides[r], int64_t shape[r]) {
  bool used[r];
  for (int i = 0; i < r; i++) {
    used[i] = false;
  }
  return lmad_contiguous_search(0, 1, r, strides, shape, used);
}

// Does this copy correspond to something that could be done with a
// memcpy()-like operation?  I.e. do the LMADs actually represent the
// same in-memory layout and are they contiguous?
static bool lmad_memcpyable(int r,
                            int64_t dst_strides[r], int64_t src_strides[r], int64_t shape[r]) {
  if (!lmad_contiguous(r, dst_strides, shape)) {
    return false;
  }
  for (int i = 0; i < r; i++) {
    if (dst_strides[i] != src_strides[i] && shape[i] != 1) {
      return false;
    }
  }
  return true;
}


static void log_copy(struct futhark_context* ctx,
                     const char *kind, const char *provenance,
                     int r,
                     int64_t dst_offset, int64_t dst_strides[r],
                     int64_t src_offset, int64_t src_strides[r],
                     int64_t shape[r]) {
  if (ctx->logging) {
    fprintf(ctx->log, "\n# Copy %s\n", kind);
    if (provenance) { fprintf(ctx->log, "At: %s\n", provenance); }
    fprintf(ctx->log, "Shape: ");
    for (int i = 0; i < r; i++) { fprintf(ctx->log, "[%ld]", (long int)shape[i]); }
    fprintf(ctx->log, "\n");
    fprintf(ctx->log, "Dst offset: %ld\n", (long int)dst_offset);
    fprintf(ctx->log, "Dst strides:");
    for (int i = 0; i < r; i++) { fprintf(ctx->log, " %ld", (long int)dst_strides[i]); }
    fprintf(ctx->log, "\n");
    fprintf(ctx->log, "Src offset: %ld\n", (long int)src_offset);
    fprintf(ctx->log, "Src strides:");
    for (int i = 0; i < r; i++) { fprintf(ctx->log, " %ld", (long int)src_strides[i]); }
    fprintf(ctx->log, "\n");
  }
}

static void log_transpose(struct futhark_context* ctx,
                          int64_t k, int64_t n, int64_t m) {
  if (ctx->logging) {
    fprintf(ctx->log, "## Transpose\n");
    fprintf(ctx->log, "Arrays     : %ld\n", (long int)k);
    fprintf(ctx->log, "X elements : %ld\n", (long int)m);
    fprintf(ctx->log, "Y elements : %ld\n", (long int)n);
    fprintf(ctx->log, "\n");
  }
}

#define GEN_LMAD_COPY(NAME, ELEM_TYPE)                                  \
  static void lmad_copy_##NAME                                          \
  (struct futhark_context *ctx, int r,                                  \
   ELEM_TYPE* dst, int64_t dst_offset, int64_t dst_strides[r],          \
   ELEM_TYPE *src, int64_t src_offset, int64_t src_strides[r],          \
   int64_t shape[r]) {                                                  \
    log_copy(ctx, "CPU to CPU", NULL, r, dst_offset, dst_strides,       \
             src_offset, src_strides, shape);                           \
    int64_t size = 1;                                                   \
    for (int i = 0; i < r; i++) { size *= shape[i]; }                   \
    if (size == 0) { return; }                                          \
    int64_t k, n, m;                                                    \
    if (lmad_map_tr(&k, &n, &m,                                         \
                    r, dst_strides, src_strides, shape)) {              \
      log_transpose(ctx, k, n, m);                                      \
      map_transpose_##NAME                                              \
        (dst+dst_offset, src+src_offset, k, n, m, 0, n, 0, m);          \
    } else if (lmad_memcpyable(r, dst_strides, src_strides, shape)) {   \
      if (ctx->logging) {fprintf(ctx->log, "## Flat copy\n\n");}          \
      memcpy(dst+dst_offset, src+src_offset, size*sizeof(*dst));        \
    } else {                                                            \
      if (ctx->logging) {fprintf(ctx->log, "## General copy\n\n");}       \
      lmad_copy_elements_##NAME                                         \
        (r,                                                             \
         dst+dst_offset, dst_strides,                                   \
         src+src_offset, src_strides, shape);                           \
    }                                                                   \
  }

GEN_MAP_TRANSPOSE(1b, uint8_t)
GEN_MAP_TRANSPOSE(2b, uint16_t)
GEN_MAP_TRANSPOSE(4b, uint32_t)
GEN_MAP_TRANSPOSE(8b, uint64_t)

GEN_LMAD_COPY_ELEMENTS(1b, uint8_t)
GEN_LMAD_COPY_ELEMENTS(2b, uint16_t)
GEN_LMAD_COPY_ELEMENTS(4b, uint32_t)
GEN_LMAD_COPY_ELEMENTS(8b, uint64_t)

GEN_LMAD_COPY(1b, uint8_t)
GEN_LMAD_COPY(2b, uint16_t)
GEN_LMAD_COPY(4b, uint32_t)
GEN_LMAD_COPY(8b, uint64_t)

// End of copy.h

#define FUTHARK_FUN_ATTR static

FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_145558, double *out_prim_out_145559, struct memblock wdown_mem_143121, struct memblock wkey_mem_143122, struct memblock wout_mem_143123, struct memblock wpe_mem_143124, struct memblock wqry_mem_143125, struct memblock wte_mem_143126, struct memblock wup_mem_143127, struct memblock wval_mem_143128, struct memblock wvoc_mem_143129, struct memblock tokens_mem_143130, struct memblock target_mem_143131, struct memblock mask_mem_143132);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_145617, struct memblock wdown_mem_143121, struct memblock wkey_mem_143122, struct memblock wout_mem_143123, struct memblock wpe_mem_143124, struct memblock wqry_mem_143125, struct memblock wte_mem_143126, struct memblock wup_mem_143127, struct memblock wval_mem_143128, struct memblock wvoc_mem_143129, struct memblock tokens_mem_143130, struct memblock mask_mem_143131);
FUTHARK_FUN_ATTR int futrts_entry_grad_loss(struct futhark_context *ctx, struct memblock *mem_out_p_145674, struct memblock *mem_out_p_145675, struct memblock *mem_out_p_145676, struct memblock *mem_out_p_145677, struct memblock *mem_out_p_145678, struct memblock *mem_out_p_145679, struct memblock *mem_out_p_145680, struct memblock *mem_out_p_145681, struct memblock *mem_out_p_145682, struct memblock wdown_mem_143121, struct memblock wkey_mem_143122, struct memblock wout_mem_143123, struct memblock wpe_mem_143124, struct memblock wqry_mem_143125, struct memblock wte_mem_143126, struct memblock wup_mem_143127, struct memblock wval_mem_143128, struct memblock wvoc_mem_143129, struct memblock tokens_mem_143130, struct memblock target_mem_143131, struct memblock mask_mem_143132);
FUTHARK_FUN_ATTR int futrts_entry_make_params(struct futhark_context *ctx, struct memblock *mem_out_p_145919, struct memblock *mem_out_p_145920, struct memblock *mem_out_p_145921, struct memblock *mem_out_p_145922, struct memblock *mem_out_p_145923, struct memblock *mem_out_p_145924, struct memblock *mem_out_p_145925, struct memblock *mem_out_p_145926, struct memblock *mem_out_p_145927, struct memblock wte_mem_143121, struct memblock wpe_mem_143122, struct memblock wqry_mem_143123, struct memblock wkey_mem_143124, struct memblock wval_mem_143125, struct memblock wout_mem_143126, struct memblock wup_mem_143127, struct memblock wdown_mem_143128, struct memblock wvoc_mem_143129, int64_t sl_56320);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    return 0;
}
struct futhark_i64_1d {
    struct memblock mem;
    int64_t shape[1];
};
struct futhark_i64_1d *futhark_new_i64_1d(struct futhark_context *ctx, const int64_t *data, int64_t dim0)
{
    int err = 0;
    struct futhark_i64_1d *bad = NULL;
    struct futhark_i64_1d *arr = (struct futhark_i64_1d *) malloc(sizeof(struct futhark_i64_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    arr->shape[0] = dim0;
    if (memblock_alloc(ctx, &arr->mem, arr->shape[0] * 8, "arr->mem"))
        err = 1;
    if ((size_t) dim0 * 8 > 0)
        memmove(arr->mem.mem + 0, (const unsigned char *) data + 0, (size_t) dim0 * 8);
    lock_unlock(&ctx->lock);
    if (err != 0) {
        free(arr);
        return bad;
    }
    return arr;
}
struct futhark_i64_1d *futhark_new_raw_i64_1d(struct futhark_context *ctx, unsigned char *data, int64_t dim0)
{
    int err = 0;
    struct futhark_i64_1d *bad = NULL;
    struct futhark_i64_1d *arr = (struct futhark_i64_1d *) malloc(sizeof(struct futhark_i64_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    arr->mem.mem = data;
    arr->shape[0] = dim0;
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr, int64_t *data)
{
    int err = 0;
    
    lock_lock(&ctx->lock);
    if ((size_t) arr->shape[0] * 8 > 0)
        memmove((unsigned char *) data + 0, arr->mem.mem + 0, (size_t) arr->shape[0] * 8);
    lock_unlock(&ctx->lock);
    return err;
}
int futhark_index_i64_1d(struct futhark_context *ctx, int64_t *out, struct futhark_i64_1d *arr, int64_t i0)
{
    int err = 0;
    
    if (i0 >= 0 && i0 < arr->shape[0]) {
        lock_lock(&ctx->lock);
        if (8 > 0)
            memmove((unsigned char *) out + 0, arr->mem.mem + 8 * (i0 * 1), 8);
        lock_unlock(&ctx->lock);
    } else {
        err = 1;
        set_error(ctx, strdup("Index out of bounds."));
    }
    return err;
}
unsigned char *futhark_values_raw_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr)
{
    (void) ctx;
    return arr->shape;
}
struct futhark_f64_1d {
    struct memblock mem;
    int64_t shape[1];
};
struct futhark_f64_1d *futhark_new_f64_1d(struct futhark_context *ctx, const double *data, int64_t dim0)
{
    int err = 0;
    struct futhark_f64_1d *bad = NULL;
    struct futhark_f64_1d *arr = (struct futhark_f64_1d *) malloc(sizeof(struct futhark_f64_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    arr->shape[0] = dim0;
    if (memblock_alloc(ctx, &arr->mem, arr->shape[0] * 8, "arr->mem"))
        err = 1;
    if ((size_t) dim0 * 8 > 0)
        memmove(arr->mem.mem + 0, (const unsigned char *) data + 0, (size_t) dim0 * 8);
    lock_unlock(&ctx->lock);
    if (err != 0) {
        free(arr);
        return bad;
    }
    return arr;
}
struct futhark_f64_1d *futhark_new_raw_f64_1d(struct futhark_context *ctx, unsigned char *data, int64_t dim0)
{
    int err = 0;
    struct futhark_f64_1d *bad = NULL;
    struct futhark_f64_1d *arr = (struct futhark_f64_1d *) malloc(sizeof(struct futhark_f64_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    arr->mem.mem = data;
    arr->shape[0] = dim0;
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f64_1d(struct futhark_context *ctx, struct futhark_f64_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f64_1d(struct futhark_context *ctx, struct futhark_f64_1d *arr, double *data)
{
    int err = 0;
    
    lock_lock(&ctx->lock);
    if ((size_t) arr->shape[0] * 8 > 0)
        memmove((unsigned char *) data + 0, arr->mem.mem + 0, (size_t) arr->shape[0] * 8);
    lock_unlock(&ctx->lock);
    return err;
}
int futhark_index_f64_1d(struct futhark_context *ctx, double *out, struct futhark_f64_1d *arr, int64_t i0)
{
    int err = 0;
    
    if (i0 >= 0 && i0 < arr->shape[0]) {
        lock_lock(&ctx->lock);
        if (8 > 0)
            memmove((unsigned char *) out + 0, arr->mem.mem + 8 * (i0 * 1), 8);
        lock_unlock(&ctx->lock);
    } else {
        err = 1;
        set_error(ctx, strdup("Index out of bounds."));
    }
    return err;
}
unsigned char *futhark_values_raw_f64_1d(struct futhark_context *ctx, struct futhark_f64_1d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_f64_1d(struct futhark_context *ctx, struct futhark_f64_1d *arr)
{
    (void) ctx;
    return arr->shape;
}
struct futhark_f64_2d {
    struct memblock mem;
    int64_t shape[2];
};
struct futhark_f64_2d *futhark_new_f64_2d(struct futhark_context *ctx, const double *data, int64_t dim0, int64_t dim1)
{
    int err = 0;
    struct futhark_f64_2d *bad = NULL;
    struct futhark_f64_2d *arr = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    if (memblock_alloc(ctx, &arr->mem, arr->shape[0] * arr->shape[1] * 8, "arr->mem"))
        err = 1;
    if ((size_t) (dim0 * dim1) * 8 > 0)
        memmove(arr->mem.mem + 0, (const unsigned char *) data + 0, (size_t) (dim0 * dim1) * 8);
    lock_unlock(&ctx->lock);
    if (err != 0) {
        free(arr);
        return bad;
    }
    return arr;
}
struct futhark_f64_2d *futhark_new_raw_f64_2d(struct futhark_context *ctx, unsigned char *data, int64_t dim0, int64_t dim1)
{
    int err = 0;
    struct futhark_f64_2d *bad = NULL;
    struct futhark_f64_2d *arr = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    arr->mem.mem = data;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f64_2d(struct futhark_context *ctx, struct futhark_f64_2d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f64_2d(struct futhark_context *ctx, struct futhark_f64_2d *arr, double *data)
{
    int err = 0;
    
    lock_lock(&ctx->lock);
    if ((size_t) (arr->shape[0] * arr->shape[1]) * 8 > 0)
        memmove((unsigned char *) data + 0, arr->mem.mem + 0, (size_t) (arr->shape[0] * arr->shape[1]) * 8);
    lock_unlock(&ctx->lock);
    return err;
}
int futhark_index_f64_2d(struct futhark_context *ctx, double *out, struct futhark_f64_2d *arr, int64_t i0, int64_t i1)
{
    int err = 0;
    
    if ((i0 >= 0 && i0 < arr->shape[0]) && (i1 >= 0 && i1 < arr->shape[1])) {
        lock_lock(&ctx->lock);
        if (8 > 0)
            memmove((unsigned char *) out + 0, arr->mem.mem + 8 * (i0 * arr->shape[1] + i1 * 1), 8);
        lock_unlock(&ctx->lock);
    } else {
        err = 1;
        set_error(ctx, strdup("Index out of bounds."));
    }
    return err;
}
unsigned char *futhark_values_raw_f64_2d(struct futhark_context *ctx, struct futhark_f64_2d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_f64_2d(struct futhark_context *ctx, struct futhark_f64_2d *arr)
{
    (void) ctx;
    return arr->shape;
}
struct futhark_opaque_params {
    struct futhark_f64_2d *v0;
    struct futhark_f64_2d *v1;
    struct futhark_f64_2d *v2;
    struct futhark_f64_2d *v3;
    struct futhark_f64_2d *v4;
    struct futhark_f64_2d *v5;
    struct futhark_f64_2d *v6;
    struct futhark_f64_2d *v7;
    struct futhark_f64_2d *v8;
};
int futhark_project_opaque_params_wdown(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    struct futhark_f64_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_2d));
    memcpy(v, obj->v0, sizeof(struct futhark_f64_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_params_wkey(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    struct futhark_f64_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_2d));
    memcpy(v, obj->v1, sizeof(struct futhark_f64_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_params_wout(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    struct futhark_f64_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_2d));
    memcpy(v, obj->v2, sizeof(struct futhark_f64_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_params_wpe(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    struct futhark_f64_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_2d));
    memcpy(v, obj->v3, sizeof(struct futhark_f64_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_params_wqry(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    struct futhark_f64_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_2d));
    memcpy(v, obj->v4, sizeof(struct futhark_f64_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_params_wte(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    struct futhark_f64_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_2d));
    memcpy(v, obj->v5, sizeof(struct futhark_f64_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_params_wup(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    struct futhark_f64_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_2d));
    memcpy(v, obj->v6, sizeof(struct futhark_f64_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_params_wval(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    struct futhark_f64_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_2d));
    memcpy(v, obj->v7, sizeof(struct futhark_f64_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_params_wvoc(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    struct futhark_f64_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_2d));
    memcpy(v, obj->v8, sizeof(struct futhark_f64_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_new_opaque_params(struct futhark_context *ctx, struct futhark_opaque_params **out, const struct futhark_f64_2d *f_wdown, const struct futhark_f64_2d *f_wkey, const struct futhark_f64_2d *f_wout, const struct futhark_f64_2d *f_wpe, const struct futhark_f64_2d *f_wqry, const struct futhark_f64_2d *f_wte, const struct futhark_f64_2d *f_wup, const struct futhark_f64_2d *f_wval, const struct futhark_f64_2d *f_wvoc)
{
    struct futhark_opaque_params *v = malloc(sizeof(struct futhark_opaque_params));
    
    lock_lock(&ctx->lock);
    {
        v->v0 = malloc(sizeof(struct futhark_f64_2d));
        memcpy(v->v0, f_wdown, sizeof(struct futhark_f64_2d));
        (void) (*v->v0->mem.references)++;
    }
    {
        v->v1 = malloc(sizeof(struct futhark_f64_2d));
        memcpy(v->v1, f_wkey, sizeof(struct futhark_f64_2d));
        (void) (*v->v1->mem.references)++;
    }
    {
        v->v2 = malloc(sizeof(struct futhark_f64_2d));
        memcpy(v->v2, f_wout, sizeof(struct futhark_f64_2d));
        (void) (*v->v2->mem.references)++;
    }
    {
        v->v3 = malloc(sizeof(struct futhark_f64_2d));
        memcpy(v->v3, f_wpe, sizeof(struct futhark_f64_2d));
        (void) (*v->v3->mem.references)++;
    }
    {
        v->v4 = malloc(sizeof(struct futhark_f64_2d));
        memcpy(v->v4, f_wqry, sizeof(struct futhark_f64_2d));
        (void) (*v->v4->mem.references)++;
    }
    {
        v->v5 = malloc(sizeof(struct futhark_f64_2d));
        memcpy(v->v5, f_wte, sizeof(struct futhark_f64_2d));
        (void) (*v->v5->mem.references)++;
    }
    {
        v->v6 = malloc(sizeof(struct futhark_f64_2d));
        memcpy(v->v6, f_wup, sizeof(struct futhark_f64_2d));
        (void) (*v->v6->mem.references)++;
    }
    {
        v->v7 = malloc(sizeof(struct futhark_f64_2d));
        memcpy(v->v7, f_wval, sizeof(struct futhark_f64_2d));
        (void) (*v->v7->mem.references)++;
    }
    {
        v->v8 = malloc(sizeof(struct futhark_f64_2d));
        memcpy(v->v8, f_wvoc, sizeof(struct futhark_f64_2d));
        (void) (*v->v8->mem.references)++;
    }
    lock_unlock(&ctx->lock);
    *out = v;
    return FUTHARK_SUCCESS;
}
int futhark_free_opaque_params(struct futhark_context *ctx, struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    int ret = 0, tmp;
    
    if (obj->v0 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v0)) != 0)
        ret = tmp;
    if (obj->v1 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v1)) != 0)
        ret = tmp;
    if (obj->v2 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v2)) != 0)
        ret = tmp;
    if (obj->v3 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v3)) != 0)
        ret = tmp;
    if (obj->v4 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v4)) != 0)
        ret = tmp;
    if (obj->v5 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v5)) != 0)
        ret = tmp;
    if (obj->v6 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v6)) != 0)
        ret = tmp;
    if (obj->v7 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v7)) != 0)
        ret = tmp;
    if (obj->v8 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v8)) != 0)
        ret = tmp;
    free(obj);
    return ret;
}
int futhark_store_opaque_params(struct futhark_context *ctx, const struct futhark_opaque_params *obj, void **p, size_t *n)
{
    (void) ctx;
    
    int ret = 0;
    int64_t size_0 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v0)[0] * futhark_shape_f64_2d(ctx, obj->v0)[1] * sizeof(double);
    int64_t size_1 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v1)[0] * futhark_shape_f64_2d(ctx, obj->v1)[1] * sizeof(double);
    int64_t size_2 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v2)[0] * futhark_shape_f64_2d(ctx, obj->v2)[1] * sizeof(double);
    int64_t size_3 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v3)[0] * futhark_shape_f64_2d(ctx, obj->v3)[1] * sizeof(double);
    int64_t size_4 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v4)[0] * futhark_shape_f64_2d(ctx, obj->v4)[1] * sizeof(double);
    int64_t size_5 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v5)[0] * futhark_shape_f64_2d(ctx, obj->v5)[1] * sizeof(double);
    int64_t size_6 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v6)[0] * futhark_shape_f64_2d(ctx, obj->v6)[1] * sizeof(double);
    int64_t size_7 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v7)[0] * futhark_shape_f64_2d(ctx, obj->v7)[1] * sizeof(double);
    int64_t size_8 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v8)[0] * futhark_shape_f64_2d(ctx, obj->v8)[1] * sizeof(double);
    
    *n = size_0 + size_1 + size_2 + size_3 + size_4 + size_5 + size_6 + size_7 + size_8;
    if (p != NULL && *p == NULL)
        *p = malloc(*n);
    if (p != NULL) {
        unsigned char *out = *p;
        
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v0), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v0, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v0)[0] * futhark_shape_f64_2d(ctx, obj->v0)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v1), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v1, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v1)[0] * futhark_shape_f64_2d(ctx, obj->v1)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v2), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v2, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v2)[0] * futhark_shape_f64_2d(ctx, obj->v2)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v3), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v3, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v3)[0] * futhark_shape_f64_2d(ctx, obj->v3)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v4), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v4, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v4)[0] * futhark_shape_f64_2d(ctx, obj->v4)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v5), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v5, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v5)[0] * futhark_shape_f64_2d(ctx, obj->v5)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v6), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v6, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v6)[0] * futhark_shape_f64_2d(ctx, obj->v6)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v7), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v7, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v7)[0] * futhark_shape_f64_2d(ctx, obj->v7)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v8), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v8, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v8)[0] * futhark_shape_f64_2d(ctx, obj->v8)[1] * sizeof(double);
    }
    return ret;
}
struct futhark_opaque_params *futhark_restore_opaque_params(struct futhark_context *ctx, const void *p)
{
    (void) ctx;
    
    int err = 0;
    const unsigned char *src = p;
    struct futhark_opaque_params *obj = malloc(sizeof(struct futhark_opaque_params));
    int64_t shape_0[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_0, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_0 = src;
    
    obj->v0 = NULL;
    src += shape_0[0] * shape_0[1] * sizeof(double);
    
    int64_t shape_1[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_1, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_1 = src;
    
    obj->v1 = NULL;
    src += shape_1[0] * shape_1[1] * sizeof(double);
    
    int64_t shape_2[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_2, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_2 = src;
    
    obj->v2 = NULL;
    src += shape_2[0] * shape_2[1] * sizeof(double);
    
    int64_t shape_3[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_3, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_3 = src;
    
    obj->v3 = NULL;
    src += shape_3[0] * shape_3[1] * sizeof(double);
    
    int64_t shape_4[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_4, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_4 = src;
    
    obj->v4 = NULL;
    src += shape_4[0] * shape_4[1] * sizeof(double);
    
    int64_t shape_5[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_5, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_5 = src;
    
    obj->v5 = NULL;
    src += shape_5[0] * shape_5[1] * sizeof(double);
    
    int64_t shape_6[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_6, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_6 = src;
    
    obj->v6 = NULL;
    src += shape_6[0] * shape_6[1] * sizeof(double);
    
    int64_t shape_7[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_7, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_7 = src;
    
    obj->v7 = NULL;
    src += shape_7[0] * shape_7[1] * sizeof(double);
    
    int64_t shape_8[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_8, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_8 = src;
    
    obj->v8 = NULL;
    src += shape_8[0] * shape_8[1] * sizeof(double);
    if (err == 0) {
        obj->v0 = futhark_new_f64_2d(ctx, data_0, shape_0[0], shape_0[1]);
        if (obj->v0 == NULL)
            err = 1;
        obj->v1 = futhark_new_f64_2d(ctx, data_1, shape_1[0], shape_1[1]);
        if (obj->v1 == NULL)
            err = 1;
        obj->v2 = futhark_new_f64_2d(ctx, data_2, shape_2[0], shape_2[1]);
        if (obj->v2 == NULL)
            err = 1;
        obj->v3 = futhark_new_f64_2d(ctx, data_3, shape_3[0], shape_3[1]);
        if (obj->v3 == NULL)
            err = 1;
        obj->v4 = futhark_new_f64_2d(ctx, data_4, shape_4[0], shape_4[1]);
        if (obj->v4 == NULL)
            err = 1;
        obj->v5 = futhark_new_f64_2d(ctx, data_5, shape_5[0], shape_5[1]);
        if (obj->v5 == NULL)
            err = 1;
        obj->v6 = futhark_new_f64_2d(ctx, data_6, shape_6[0], shape_6[1]);
        if (obj->v6 == NULL)
            err = 1;
        obj->v7 = futhark_new_f64_2d(ctx, data_7, shape_7[0], shape_7[1]);
        if (obj->v7 == NULL)
            err = 1;
        obj->v8 = futhark_new_f64_2d(ctx, data_8, shape_8[0], shape_8[1]);
        if (obj->v8 == NULL)
            err = 1;
    }
    if (err != 0) {
        int ret = 0, tmp;
        
        if (obj->v0 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v0)) != 0)
            ret = tmp;
        if (obj->v1 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v1)) != 0)
            ret = tmp;
        if (obj->v2 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v2)) != 0)
            ret = tmp;
        if (obj->v3 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v3)) != 0)
            ret = tmp;
        if (obj->v4 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v4)) != 0)
            ret = tmp;
        if (obj->v5 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v5)) != 0)
            ret = tmp;
        if (obj->v6 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v6)) != 0)
            ret = tmp;
        if (obj->v7 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v7)) != 0)
            ret = tmp;
        if (obj->v8 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v8)) != 0)
            ret = tmp;
        free(obj);
        obj = NULL;
    }
    return obj;
}
struct futhark_opaque_tup2_f64_arr1d_f64 {
    double v0;
    struct futhark_f64_1d *v1;
};
int futhark_project_opaque_tup2_f64_arr1d_f64_0(struct futhark_context *ctx, double *out, const struct futhark_opaque_tup2_f64_arr1d_f64 *obj)
{
    (void) ctx;
    
    double v;
    
    v = obj->v0;
    *out = v;
    return 0;
}
int futhark_project_opaque_tup2_f64_arr1d_f64_1(struct futhark_context *ctx, struct futhark_f64_1d **out, const struct futhark_opaque_tup2_f64_arr1d_f64 *obj)
{
    (void) ctx;
    
    struct futhark_f64_1d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_1d));
    memcpy(v, obj->v1, sizeof(struct futhark_f64_1d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_new_opaque_tup2_f64_arr1d_f64(struct futhark_context *ctx, struct futhark_opaque_tup2_f64_arr1d_f64 **out, const double f_0, const struct futhark_f64_1d *f_1)
{
    struct futhark_opaque_tup2_f64_arr1d_f64 *v = malloc(sizeof(struct futhark_opaque_tup2_f64_arr1d_f64));
    
    lock_lock(&ctx->lock);
    v->v0 = f_0;
    {
        v->v1 = malloc(sizeof(struct futhark_f64_1d));
        memcpy(v->v1, f_1, sizeof(struct futhark_f64_1d));
        (void) (*v->v1->mem.references)++;
    }
    lock_unlock(&ctx->lock);
    *out = v;
    return FUTHARK_SUCCESS;
}
int futhark_free_opaque_tup2_f64_arr1d_f64(struct futhark_context *ctx, struct futhark_opaque_tup2_f64_arr1d_f64 *obj)
{
    (void) ctx;
    
    int ret = 0, tmp;
    
    if (obj->v1 != NULL && (tmp = futhark_free_f64_1d(ctx, obj->v1)) != 0)
        ret = tmp;
    free(obj);
    return ret;
}
int futhark_store_opaque_tup2_f64_arr1d_f64(struct futhark_context *ctx, const struct futhark_opaque_tup2_f64_arr1d_f64 *obj, void **p, size_t *n)
{
    (void) ctx;
    
    int ret = 0;
    int64_t size_0 = 7 + 0 * sizeof(int64_t) + 1 * sizeof(double);
    int64_t size_1 = 7 + 1 * sizeof(int64_t) + futhark_shape_f64_1d(ctx, obj->v1)[0] * sizeof(double);
    
    *n = size_0 + size_1;
    if (p != NULL && *p == NULL)
        *p = malloc(*n);
    if (p != NULL) {
        unsigned char *out = *p;
        
        *out++ = 'b';
        *out++ = 2;
        *out++ = 0;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, &obj->v0, sizeof(obj->v0));
        out += sizeof(obj->v0);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 1;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_1d(ctx, obj->v1), 1 * sizeof(int64_t));
        out += 1 * sizeof(int64_t);
        ret |= futhark_values_f64_1d(ctx, obj->v1, (void *) out);
        out += futhark_shape_f64_1d(ctx, obj->v1)[0] * sizeof(double);
    }
    return ret;
}
struct futhark_opaque_tup2_f64_arr1d_f64 *futhark_restore_opaque_tup2_f64_arr1d_f64(struct futhark_context *ctx, const void *p)
{
    (void) ctx;
    
    int err = 0;
    const unsigned char *src = p;
    struct futhark_opaque_tup2_f64_arr1d_f64 *obj = malloc(sizeof(struct futhark_opaque_tup2_f64_arr1d_f64));
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 0;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        src += 0 * sizeof(int64_t);
    }
    
    const void *data_0 = src;
    
    src += sizeof(obj->v0);
    
    int64_t shape_1[1] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 1;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_1, src, 1 * sizeof(int64_t));
        src += 1 * sizeof(int64_t);
    }
    
    const void *data_1 = src;
    
    obj->v1 = NULL;
    src += shape_1[0] * sizeof(double);
    if (err == 0) {
        memcpy(&obj->v0, data_0, sizeof(obj->v0));
        obj->v1 = futhark_new_f64_1d(ctx, data_1, shape_1[0]);
        if (obj->v1 == NULL)
            err = 1;
    }
    if (err != 0) {
        int ret = 0, tmp;
        
        if (obj->v1 != NULL && (tmp = futhark_free_f64_1d(ctx, obj->v1)) != 0)
            ret = tmp;
        free(obj);
        obj = NULL;
    }
    return obj;
}
struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 {
    struct futhark_f64_2d *v0;
    struct futhark_f64_2d *v1;
    struct futhark_f64_2d *v2;
    struct futhark_f64_2d *v3;
    struct futhark_f64_2d *v4;
    struct futhark_f64_2d *v5;
    struct futhark_f64_2d *v6;
    struct futhark_f64_2d *v7;
    struct futhark_f64_2d *v8;
};
int futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_0(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj)
{
    (void) ctx;
    
    struct futhark_f64_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_2d));
    memcpy(v, obj->v0, sizeof(struct futhark_f64_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_1(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj)
{
    (void) ctx;
    
    struct futhark_f64_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_2d));
    memcpy(v, obj->v1, sizeof(struct futhark_f64_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_2(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj)
{
    (void) ctx;
    
    struct futhark_f64_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_2d));
    memcpy(v, obj->v2, sizeof(struct futhark_f64_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_3(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj)
{
    (void) ctx;
    
    struct futhark_f64_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_2d));
    memcpy(v, obj->v3, sizeof(struct futhark_f64_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_4(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj)
{
    (void) ctx;
    
    struct futhark_f64_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_2d));
    memcpy(v, obj->v4, sizeof(struct futhark_f64_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_5(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj)
{
    (void) ctx;
    
    struct futhark_f64_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_2d));
    memcpy(v, obj->v5, sizeof(struct futhark_f64_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_6(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj)
{
    (void) ctx;
    
    struct futhark_f64_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_2d));
    memcpy(v, obj->v6, sizeof(struct futhark_f64_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_7(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj)
{
    (void) ctx;
    
    struct futhark_f64_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_2d));
    memcpy(v, obj->v7, sizeof(struct futhark_f64_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_8(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj)
{
    (void) ctx;
    
    struct futhark_f64_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_2d));
    memcpy(v, obj->v8, sizeof(struct futhark_f64_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_new_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_f64_2d *f_0, const struct futhark_f64_2d *f_1, const struct futhark_f64_2d *f_2, const struct futhark_f64_2d *f_3, const struct futhark_f64_2d *f_4, const struct futhark_f64_2d *f_5, const struct futhark_f64_2d *f_6, const struct futhark_f64_2d *f_7, const struct futhark_f64_2d *f_8)
{
    struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *v = malloc(sizeof(struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64));
    
    lock_lock(&ctx->lock);
    {
        v->v0 = malloc(sizeof(struct futhark_f64_2d));
        memcpy(v->v0, f_0, sizeof(struct futhark_f64_2d));
        (void) (*v->v0->mem.references)++;
    }
    {
        v->v1 = malloc(sizeof(struct futhark_f64_2d));
        memcpy(v->v1, f_1, sizeof(struct futhark_f64_2d));
        (void) (*v->v1->mem.references)++;
    }
    {
        v->v2 = malloc(sizeof(struct futhark_f64_2d));
        memcpy(v->v2, f_2, sizeof(struct futhark_f64_2d));
        (void) (*v->v2->mem.references)++;
    }
    {
        v->v3 = malloc(sizeof(struct futhark_f64_2d));
        memcpy(v->v3, f_3, sizeof(struct futhark_f64_2d));
        (void) (*v->v3->mem.references)++;
    }
    {
        v->v4 = malloc(sizeof(struct futhark_f64_2d));
        memcpy(v->v4, f_4, sizeof(struct futhark_f64_2d));
        (void) (*v->v4->mem.references)++;
    }
    {
        v->v5 = malloc(sizeof(struct futhark_f64_2d));
        memcpy(v->v5, f_5, sizeof(struct futhark_f64_2d));
        (void) (*v->v5->mem.references)++;
    }
    {
        v->v6 = malloc(sizeof(struct futhark_f64_2d));
        memcpy(v->v6, f_6, sizeof(struct futhark_f64_2d));
        (void) (*v->v6->mem.references)++;
    }
    {
        v->v7 = malloc(sizeof(struct futhark_f64_2d));
        memcpy(v->v7, f_7, sizeof(struct futhark_f64_2d));
        (void) (*v->v7->mem.references)++;
    }
    {
        v->v8 = malloc(sizeof(struct futhark_f64_2d));
        memcpy(v->v8, f_8, sizeof(struct futhark_f64_2d));
        (void) (*v->v8->mem.references)++;
    }
    lock_unlock(&ctx->lock);
    *out = v;
    return FUTHARK_SUCCESS;
}
int futhark_free_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj)
{
    (void) ctx;
    
    int ret = 0, tmp;
    
    if (obj->v0 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v0)) != 0)
        ret = tmp;
    if (obj->v1 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v1)) != 0)
        ret = tmp;
    if (obj->v2 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v2)) != 0)
        ret = tmp;
    if (obj->v3 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v3)) != 0)
        ret = tmp;
    if (obj->v4 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v4)) != 0)
        ret = tmp;
    if (obj->v5 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v5)) != 0)
        ret = tmp;
    if (obj->v6 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v6)) != 0)
        ret = tmp;
    if (obj->v7 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v7)) != 0)
        ret = tmp;
    if (obj->v8 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v8)) != 0)
        ret = tmp;
    free(obj);
    return ret;
}
int futhark_store_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64(struct futhark_context *ctx, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj, void **p, size_t *n)
{
    (void) ctx;
    
    int ret = 0;
    int64_t size_0 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v0)[0] * futhark_shape_f64_2d(ctx, obj->v0)[1] * sizeof(double);
    int64_t size_1 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v1)[0] * futhark_shape_f64_2d(ctx, obj->v1)[1] * sizeof(double);
    int64_t size_2 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v2)[0] * futhark_shape_f64_2d(ctx, obj->v2)[1] * sizeof(double);
    int64_t size_3 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v3)[0] * futhark_shape_f64_2d(ctx, obj->v3)[1] * sizeof(double);
    int64_t size_4 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v4)[0] * futhark_shape_f64_2d(ctx, obj->v4)[1] * sizeof(double);
    int64_t size_5 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v5)[0] * futhark_shape_f64_2d(ctx, obj->v5)[1] * sizeof(double);
    int64_t size_6 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v6)[0] * futhark_shape_f64_2d(ctx, obj->v6)[1] * sizeof(double);
    int64_t size_7 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v7)[0] * futhark_shape_f64_2d(ctx, obj->v7)[1] * sizeof(double);
    int64_t size_8 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v8)[0] * futhark_shape_f64_2d(ctx, obj->v8)[1] * sizeof(double);
    
    *n = size_0 + size_1 + size_2 + size_3 + size_4 + size_5 + size_6 + size_7 + size_8;
    if (p != NULL && *p == NULL)
        *p = malloc(*n);
    if (p != NULL) {
        unsigned char *out = *p;
        
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v0), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v0, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v0)[0] * futhark_shape_f64_2d(ctx, obj->v0)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v1), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v1, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v1)[0] * futhark_shape_f64_2d(ctx, obj->v1)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v2), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v2, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v2)[0] * futhark_shape_f64_2d(ctx, obj->v2)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v3), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v3, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v3)[0] * futhark_shape_f64_2d(ctx, obj->v3)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v4), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v4, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v4)[0] * futhark_shape_f64_2d(ctx, obj->v4)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v5), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v5, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v5)[0] * futhark_shape_f64_2d(ctx, obj->v5)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v6), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v6, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v6)[0] * futhark_shape_f64_2d(ctx, obj->v6)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v7), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v7, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v7)[0] * futhark_shape_f64_2d(ctx, obj->v7)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v8), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v8, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v8)[0] * futhark_shape_f64_2d(ctx, obj->v8)[1] * sizeof(double);
    }
    return ret;
}
struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *futhark_restore_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64(struct futhark_context *ctx, const void *p)
{
    (void) ctx;
    
    int err = 0;
    const unsigned char *src = p;
    struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj = malloc(sizeof(struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64));
    int64_t shape_0[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_0, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_0 = src;
    
    obj->v0 = NULL;
    src += shape_0[0] * shape_0[1] * sizeof(double);
    
    int64_t shape_1[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_1, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_1 = src;
    
    obj->v1 = NULL;
    src += shape_1[0] * shape_1[1] * sizeof(double);
    
    int64_t shape_2[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_2, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_2 = src;
    
    obj->v2 = NULL;
    src += shape_2[0] * shape_2[1] * sizeof(double);
    
    int64_t shape_3[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_3, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_3 = src;
    
    obj->v3 = NULL;
    src += shape_3[0] * shape_3[1] * sizeof(double);
    
    int64_t shape_4[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_4, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_4 = src;
    
    obj->v4 = NULL;
    src += shape_4[0] * shape_4[1] * sizeof(double);
    
    int64_t shape_5[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_5, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_5 = src;
    
    obj->v5 = NULL;
    src += shape_5[0] * shape_5[1] * sizeof(double);
    
    int64_t shape_6[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_6, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_6 = src;
    
    obj->v6 = NULL;
    src += shape_6[0] * shape_6[1] * sizeof(double);
    
    int64_t shape_7[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_7, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_7 = src;
    
    obj->v7 = NULL;
    src += shape_7[0] * shape_7[1] * sizeof(double);
    
    int64_t shape_8[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_8, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_8 = src;
    
    obj->v8 = NULL;
    src += shape_8[0] * shape_8[1] * sizeof(double);
    if (err == 0) {
        obj->v0 = futhark_new_f64_2d(ctx, data_0, shape_0[0], shape_0[1]);
        if (obj->v0 == NULL)
            err = 1;
        obj->v1 = futhark_new_f64_2d(ctx, data_1, shape_1[0], shape_1[1]);
        if (obj->v1 == NULL)
            err = 1;
        obj->v2 = futhark_new_f64_2d(ctx, data_2, shape_2[0], shape_2[1]);
        if (obj->v2 == NULL)
            err = 1;
        obj->v3 = futhark_new_f64_2d(ctx, data_3, shape_3[0], shape_3[1]);
        if (obj->v3 == NULL)
            err = 1;
        obj->v4 = futhark_new_f64_2d(ctx, data_4, shape_4[0], shape_4[1]);
        if (obj->v4 == NULL)
            err = 1;
        obj->v5 = futhark_new_f64_2d(ctx, data_5, shape_5[0], shape_5[1]);
        if (obj->v5 == NULL)
            err = 1;
        obj->v6 = futhark_new_f64_2d(ctx, data_6, shape_6[0], shape_6[1]);
        if (obj->v6 == NULL)
            err = 1;
        obj->v7 = futhark_new_f64_2d(ctx, data_7, shape_7[0], shape_7[1]);
        if (obj->v7 == NULL)
            err = 1;
        obj->v8 = futhark_new_f64_2d(ctx, data_8, shape_8[0], shape_8[1]);
        if (obj->v8 == NULL)
            err = 1;
    }
    if (err != 0) {
        int ret = 0, tmp;
        
        if (obj->v0 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v0)) != 0)
            ret = tmp;
        if (obj->v1 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v1)) != 0)
            ret = tmp;
        if (obj->v2 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v2)) != 0)
            ret = tmp;
        if (obj->v3 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v3)) != 0)
            ret = tmp;
        if (obj->v4 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v4)) != 0)
            ret = tmp;
        if (obj->v5 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v5)) != 0)
            ret = tmp;
        if (obj->v6 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v6)) != 0)
            ret = tmp;
        if (obj->v7 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v7)) != 0)
            ret = tmp;
        if (obj->v8 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v8)) != 0)
            ret = tmp;
        free(obj);
        obj = NULL;
    }
    return obj;
}

FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_145558, double *out_prim_out_145559, struct memblock wdown_mem_143121, struct memblock wkey_mem_143122, struct memblock wout_mem_143123, struct memblock wpe_mem_143124, struct memblock wqry_mem_143125, struct memblock wte_mem_143126, struct memblock wup_mem_143127, struct memblock wval_mem_143128, struct memblock wvoc_mem_143129, struct memblock tokens_mem_143130, struct memblock target_mem_143131, struct memblock mask_mem_143132)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_143133_cached_sizze_145560 = 0;
    unsigned char *mem_143133 = NULL;
    int64_t mem_143138_cached_sizze_145561 = 0;
    unsigned char *mem_143138 = NULL;
    int64_t mem_143149_cached_sizze_145562 = 0;
    unsigned char *mem_143149 = NULL;
    int64_t mem_143154_cached_sizze_145563 = 0;
    unsigned char *mem_143154 = NULL;
    int64_t mem_143161_cached_sizze_145564 = 0;
    unsigned char *mem_143161 = NULL;
    int64_t mem_143172_cached_sizze_145565 = 0;
    unsigned char *mem_143172 = NULL;
    int64_t mem_143177_cached_sizze_145566 = 0;
    unsigned char *mem_143177 = NULL;
    int64_t mem_143184_cached_sizze_145567 = 0;
    unsigned char *mem_143184 = NULL;
    int64_t mem_143195_cached_sizze_145568 = 0;
    unsigned char *mem_143195 = NULL;
    int64_t mem_143196_cached_sizze_145569 = 0;
    unsigned char *mem_143196 = NULL;
    int64_t mem_143197_cached_sizze_145570 = 0;
    unsigned char *mem_143197 = NULL;
    int64_t mem_143210_cached_sizze_145571 = 0;
    unsigned char *mem_143210 = NULL;
    int64_t mem_143211_cached_sizze_145572 = 0;
    unsigned char *mem_143211 = NULL;
    int64_t mem_143212_cached_sizze_145573 = 0;
    unsigned char *mem_143212 = NULL;
    int64_t mem_143243_cached_sizze_145574 = 0;
    unsigned char *mem_143243 = NULL;
    int64_t mem_143244_cached_sizze_145575 = 0;
    unsigned char *mem_143244 = NULL;
    int64_t mem_143245_cached_sizze_145576 = 0;
    unsigned char *mem_143245 = NULL;
    int64_t mem_143261_cached_sizze_145577 = 0;
    unsigned char *mem_143261 = NULL;
    int64_t mem_143262_cached_sizze_145578 = 0;
    unsigned char *mem_143262 = NULL;
    int64_t mem_143263_cached_sizze_145579 = 0;
    unsigned char *mem_143263 = NULL;
    int64_t mem_143276_cached_sizze_145580 = 0;
    unsigned char *mem_143276 = NULL;
    int64_t mem_143277_cached_sizze_145581 = 0;
    unsigned char *mem_143277 = NULL;
    int64_t mem_143278_cached_sizze_145582 = 0;
    unsigned char *mem_143278 = NULL;
    int64_t mem_143324_cached_sizze_145583 = 0;
    unsigned char *mem_143324 = NULL;
    int64_t mem_143330_cached_sizze_145584 = 0;
    unsigned char *mem_143330 = NULL;
    int64_t mem_143335_cached_sizze_145585 = 0;
    unsigned char *mem_143335 = NULL;
    int64_t mem_143346_cached_sizze_145586 = 0;
    unsigned char *mem_143346 = NULL;
    int64_t mem_143351_cached_sizze_145587 = 0;
    unsigned char *mem_143351 = NULL;
    int64_t mem_143362_cached_sizze_145588 = 0;
    unsigned char *mem_143362 = NULL;
    int64_t mem_143367_cached_sizze_145589 = 0;
    unsigned char *mem_143367 = NULL;
    int64_t mem_143374_cached_sizze_145590 = 0;
    unsigned char *mem_143374 = NULL;
    int64_t mem_143381_cached_sizze_145591 = 0;
    unsigned char *mem_143381 = NULL;
    int64_t mem_143392_cached_sizze_145592 = 0;
    unsigned char *mem_143392 = NULL;
    int64_t mem_143397_cached_sizze_145593 = 0;
    unsigned char *mem_143397 = NULL;
    int64_t mem_143408_cached_sizze_145594 = 0;
    unsigned char *mem_143408 = NULL;
    int64_t mem_143413_cached_sizze_145595 = 0;
    unsigned char *mem_143413 = NULL;
    int64_t mem_143429_cached_sizze_145596 = 0;
    unsigned char *mem_143429 = NULL;
    int64_t mem_143434_cached_sizze_145597 = 0;
    unsigned char *mem_143434 = NULL;
    int64_t mem_143445_cached_sizze_145598 = 0;
    unsigned char *mem_143445 = NULL;
    int64_t mem_143450_cached_sizze_145599 = 0;
    unsigned char *mem_143450 = NULL;
    int64_t mem_143461_cached_sizze_145600 = 0;
    unsigned char *mem_143461 = NULL;
    int64_t mem_143466_cached_sizze_145601 = 0;
    unsigned char *mem_143466 = NULL;
    int64_t mem_143477_cached_sizze_145602 = 0;
    unsigned char *mem_143477 = NULL;
    int64_t mem_143482_cached_sizze_145603 = 0;
    unsigned char *mem_143482 = NULL;
    int64_t mem_143489_cached_sizze_145604 = 0;
    unsigned char *mem_143489 = NULL;
    int64_t mem_143500_cached_sizze_145605 = 0;
    unsigned char *mem_143500 = NULL;
    int64_t mem_143505_cached_sizze_145606 = 0;
    unsigned char *mem_143505 = NULL;
    int64_t mem_143516_cached_sizze_145607 = 0;
    unsigned char *mem_143516 = NULL;
    int64_t mem_143521_cached_sizze_145608 = 0;
    unsigned char *mem_143521 = NULL;
    int64_t mem_143532_cached_sizze_145609 = 0;
    unsigned char *mem_143532 = NULL;
    int64_t mem_143537_cached_sizze_145610 = 0;
    unsigned char *mem_143537 = NULL;
    int64_t mem_143548_cached_sizze_145611 = 0;
    unsigned char *mem_143548 = NULL;
    int64_t mem_143553_cached_sizze_145612 = 0;
    unsigned char *mem_143553 = NULL;
    int64_t mem_143564_cached_sizze_145613 = 0;
    unsigned char *mem_143564 = NULL;
    int64_t mem_143569_cached_sizze_145614 = 0;
    unsigned char *mem_143569 = NULL;
    int64_t mem_143584_cached_sizze_145615 = 0;
    unsigned char *mem_143584 = NULL;
    int64_t mem_143591_cached_sizze_145616 = 0;
    unsigned char *mem_143591 = NULL;
    struct memblock mem_143580;
    
    mem_143580.references = NULL;
    
    struct memblock mem_out_145206;
    
    mem_out_145206.references = NULL;
    
    double prim_out_145207;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_143133_cached_sizze_145560 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143133, &mem_143133_cached_sizze_145560, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143138_cached_sizze_145561 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143138, &mem_143138_cached_sizze_145561, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_141993 = 0; i_141993 < (int64_t) 16; i_141993++) {
        // futhark/microgpt.fut:441:41-50
        
        int64_t tmp_128515 = ((int64_t *) tokens_mem_143130.mem)[i_141993];
        
        // futhark/microgpt.fut:441:37-51
        
        bool x_128516 = sle64((int64_t) 0, tmp_128515);
        
        // futhark/microgpt.fut:441:37-51
        
        bool y_128517 = slt64(tmp_128515, (int64_t) 27);
        
        // futhark/microgpt.fut:441:37-51
        
        bool bounds_check_128518 = x_128516 && y_128517;
        
        // futhark/microgpt.fut:441:37-51
        
        bool index_certs_128519;
        
        if (!bounds_check_128518) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_128515, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:441:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:441:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_141989 = 0; i_141989 < (int64_t) 16; i_141989++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_128526 = ((double *) wte_mem_143126.mem)[tmp_128515 * (int64_t) 16 + i_141989];
            
            ((double *) mem_143138)[i_141989] = lifted_lambda_res_128526;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143133, i_141993 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143138, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143149_cached_sizze_145562 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143149, &mem_143149_cached_sizze_145562, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143154_cached_sizze_145563 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143154, &mem_143154_cached_sizze_145563, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143161_cached_sizze_145564 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143161, &mem_143161_cached_sizze_145564, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142005 = 0; i_142005 < (int64_t) 16; i_142005++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_128552;
        double r_128554 = 0.0;
        
        for (int64_t i_128553 = 0; i_128553 < (int64_t) 16; i_128553++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_128555 = ((double *) wpe_mem_143124.mem)[i_142005 * (int64_t) 16 + i_128553];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_128556 = ((double *) mem_143133)[i_142005 * (int64_t) 16 + i_128553];
            
            // futhark/microgpt.fut:193:76-116
            
            double zp_res_128557 = zp_lhs_128555 + zp_rhs_128556;
            
            // futhark/microgpt.fut:193:94-163
            
            double zt_res_128558 = zp_res_128557 * zp_res_128557;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_128559 = r_128554 + zt_res_128558;
            double r_tmp_145211 = zp_res_128559;
            
            r_128554 = r_tmp_145211;
        }
        defunc_0_lifted_lambda_res_128552 = r_128554;
        // futhark/microgpt.fut:193:54-182
        
        double zs_res_128560 = defunc_0_lifted_lambda_res_128552 / 16.0;
        
        // futhark/microgpt.fut:194:24-55
        
        double zp_res_128561 = 1.0e-5 + zs_res_128560;
        
        // futhark/microgpt.fut:194:16-55
        
        double sqrt_res_128562 = futrts_sqrt64(zp_res_128561);
        
        // futhark/microgpt.fut:195:85-96
        
        double zs_res_128563 = 1.0 / sqrt_res_128562;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_141997 = 0; i_141997 < (int64_t) 16; i_141997++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_128570 = ((double *) wpe_mem_143124.mem)[i_142005 * (int64_t) 16 + i_141997];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_128571 = ((double *) mem_143133)[i_142005 * (int64_t) 16 + i_141997];
            
            // futhark/microgpt.fut:195:38-78
            
            double zp_res_128572 = zp_lhs_128570 + zp_rhs_128571;
            
            // futhark/microgpt.fut:195:56-96
            
            double zt_res_128573 = zs_res_128563 * zp_res_128572;
            
            ((double *) mem_143154)[i_141997] = zt_res_128573;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142001 = 0; i_142001 < (int64_t) 16; i_142001++) {
            // futhark/microgpt.fut:196:4-14
            
            double lifted_lambda_res_128581 = ((double *) mem_143154)[i_142001];
            
            ((double *) mem_143161)[i_142001] = lifted_lambda_res_128581;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143149, i_142005 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143161, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143172_cached_sizze_145565 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143172, &mem_143172_cached_sizze_145565, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143177_cached_sizze_145566 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143177, &mem_143177_cached_sizze_145566, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143184_cached_sizze_145567 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143184, &mem_143184_cached_sizze_145567, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142017 = 0; i_142017 < (int64_t) 16; i_142017++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_128590;
        double r_128592 = 0.0;
        
        for (int64_t i_128591 = 0; i_128591 < (int64_t) 16; i_128591++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_128593 = ((double *) mem_143149)[i_142017 * (int64_t) 16 + i_128591];
            
            // futhark/microgpt.fut:197:78-115
            
            double zt_res_128594 = zt_lhs_128593 * zt_lhs_128593;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_128595 = r_128592 + zt_res_128594;
            double r_tmp_145215 = zp_res_128595;
            
            r_128592 = r_tmp_145215;
        }
        defunc_0_lifted_lambda_res_128590 = r_128592;
        // futhark/microgpt.fut:197:57-133
        
        double zs_res_128596 = defunc_0_lifted_lambda_res_128590 / 16.0;
        
        // futhark/microgpt.fut:198:24-55
        
        double zp_res_128597 = 1.0e-5 + zs_res_128596;
        
        // futhark/microgpt.fut:198:16-55
        
        double sqrt_res_128598 = futrts_sqrt64(zp_res_128597);
        
        // futhark/microgpt.fut:199:59-70
        
        double zs_res_128599 = 1.0 / sqrt_res_128598;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142009 = 0; i_142009 < (int64_t) 16; i_142009++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_128606 = ((double *) mem_143149)[i_142017 * (int64_t) 16 + i_142009];
            
            // futhark/microgpt.fut:199:37-70
            
            double zt_res_128607 = zs_res_128599 * zt_lhs_128606;
            
            ((double *) mem_143177)[i_142009] = zt_res_128607;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142013 = 0; i_142013 < (int64_t) 16; i_142013++) {
            // futhark/microgpt.fut:200:4-14
            
            double lifted_lambda_res_128615 = ((double *) mem_143177)[i_142013];
            
            ((double *) mem_143184)[i_142013] = lifted_lambda_res_128615;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143172, i_142017 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143184, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143195_cached_sizze_145568 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143195, &mem_143195_cached_sizze_145568, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143196_cached_sizze_145569 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143196, &mem_143196_cached_sizze_145569, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143197_cached_sizze_145570 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143197, &mem_143197_cached_sizze_145570, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143210_cached_sizze_145571 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143210, &mem_143210_cached_sizze_145571, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143211_cached_sizze_145572 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143211, &mem_143211_cached_sizze_145572, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143212_cached_sizze_145573 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143212, &mem_143212_cached_sizze_145573, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142035 = 0; i_142035 < (int64_t) 16; i_142035++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142025 = 0; i_142025 < (int64_t) 16; i_142025++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_131696;
            double r_131698 = 0.0;
            
            for (int64_t i_131697 = 0; i_131697 < (int64_t) 16; i_131697++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_131699 = ((double *) wqry_mem_143125.mem)[i_142025 * (int64_t) 16 + i_131697];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_131700 = ((double *) mem_143172)[i_142035 * (int64_t) 16 + i_131697];
                
                // futhark/microgpt.fut:201:66-105
                
                double zt_res_131701 = zt_lhs_131699 * zt_rhs_131700;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_131702 = r_131698 + zt_res_131701;
                double r_tmp_145224 = zp_res_131702;
                
                r_131698 = r_tmp_145224;
            }
            defunc_0_lifted_lambda_res_131696 = r_131698;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_131709;
            double r_131711 = 0.0;
            
            for (int64_t i_131710 = 0; i_131710 < (int64_t) 16; i_131710++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_131712 = ((double *) wkey_mem_143122.mem)[i_142025 * (int64_t) 16 + i_131710];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_131713 = ((double *) mem_143172)[i_142035 * (int64_t) 16 + i_131710];
                
                // futhark/microgpt.fut:202:66-105
                
                double zt_res_131714 = zt_lhs_131712 * zt_rhs_131713;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_131715 = r_131711 + zt_res_131714;
                double r_tmp_145225 = zp_res_131715;
                
                r_131711 = r_tmp_145225;
            }
            defunc_0_lifted_lambda_res_131709 = r_131711;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_131725;
            double r_131727 = 0.0;
            
            for (int64_t i_131726 = 0; i_131726 < (int64_t) 16; i_131726++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_131728 = ((double *) wval_mem_143128.mem)[i_142025 * (int64_t) 16 + i_131726];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_131729 = ((double *) mem_143172)[i_142035 * (int64_t) 16 + i_131726];
                
                // futhark/microgpt.fut:203:66-105
                
                double zt_res_131730 = zt_lhs_131728 * zt_rhs_131729;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_131731 = r_131727 + zt_res_131730;
                double r_tmp_145226 = zp_res_131731;
                
                r_131727 = r_tmp_145226;
            }
            defunc_0_lifted_lambda_res_131725 = r_131727;
            ((double *) mem_143210)[i_142025] = defunc_0_lifted_lambda_res_131725;
            ((double *) mem_143211)[i_142025] = defunc_0_lifted_lambda_res_131709;
            ((double *) mem_143212)[i_142025] = defunc_0_lifted_lambda_res_131696;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143195, i_142035 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143210, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143196, i_142035 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143211, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143197, i_142035 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143212, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143243_cached_sizze_145574 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143243, &mem_143243_cached_sizze_145574, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143244_cached_sizze_145575 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143244, &mem_143244_cached_sizze_145575, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143245_cached_sizze_145576 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143245, &mem_143245_cached_sizze_145576, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143261_cached_sizze_145577 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143261, &mem_143261_cached_sizze_145577, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143262_cached_sizze_145578 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143262, &mem_143262_cached_sizze_145578, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143263_cached_sizze_145579 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143263, &mem_143263_cached_sizze_145579, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143276_cached_sizze_145580 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_143276, &mem_143276_cached_sizze_145580, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143277_cached_sizze_145581 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_143277, &mem_143277_cached_sizze_145581, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143278_cached_sizze_145582 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_143278, &mem_143278_cached_sizze_145582, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142065 = 0; i_142065 < (int64_t) 4; i_142065++) {
        // futhark/microgpt.fut:204:69-72
        
        int64_t zp_lhs_131572 = mul64((int64_t) 4, i_142065);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142055 = 0; i_142055 < (int64_t) 16; i_142055++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142045 = 0; i_142045 < (int64_t) 4; i_142045++) {
                // futhark/microgpt.fut:204:74-81
                
                int64_t tmp_131889 = add64(zp_lhs_131572, i_142045);
                
                // futhark/microgpt.fut:204:51-83
                
                bool x_131890 = sle64((int64_t) 0, tmp_131889);
                
                // futhark/microgpt.fut:204:51-83
                
                bool y_131891 = slt64(tmp_131889, (int64_t) 16);
                
                // futhark/microgpt.fut:204:51-83
                
                bool bounds_check_131892 = x_131890 && y_131891;
                
                // futhark/microgpt.fut:204:51-83
                
                bool index_certs_131893;
                
                if (!bounds_check_131892) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_131889, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:204:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:204:15-84\n   #9  futhark/microgpt.fut:442:7-76\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_131894 = ((double *) mem_143197)[i_142055 * (int64_t) 16 + tmp_131889];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_131902 = ((double *) mem_143196)[i_142055 * (int64_t) 16 + tmp_131889];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_131913 = ((double *) mem_143195)[i_142055 * (int64_t) 16 + tmp_131889];
                
                ((double *) mem_143276)[i_142045] = lifted_lambda_res_131913;
                ((double *) mem_143277)[i_142045] = lifted_lambda_res_131902;
                ((double *) mem_143278)[i_142045] = lifted_lambda_res_131894;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143261, i_142055 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143276, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143262, i_142055 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143277, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143263, i_142055 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143278, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143243, i_142065 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_143261, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143244, i_142065 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_143262, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143245, i_142065 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_143263, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143324_cached_sizze_145583 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143324, &mem_143324_cached_sizze_145583, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143330_cached_sizze_145584 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143330, &mem_143330_cached_sizze_145584, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143335_cached_sizze_145585 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143335, &mem_143335_cached_sizze_145585, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143346_cached_sizze_145586 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143346, &mem_143346_cached_sizze_145586, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143351_cached_sizze_145587 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143351, &mem_143351_cached_sizze_145587, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143362_cached_sizze_145588 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143362, &mem_143362_cached_sizze_145588, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143367_cached_sizze_145589 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143367, &mem_143367_cached_sizze_145589, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143374_cached_sizze_145590 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143374, &mem_143374_cached_sizze_145590, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143381_cached_sizze_145591 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143381, &mem_143381_cached_sizze_145591, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143392_cached_sizze_145592 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143392, &mem_143392_cached_sizze_145592, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143397_cached_sizze_145593 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_143397, &mem_143397_cached_sizze_145593, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143408_cached_sizze_145594 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143408, &mem_143408_cached_sizze_145594, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143413_cached_sizze_145595 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_143413, &mem_143413_cached_sizze_145595, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142121 = 0; i_142121 < (int64_t) 4; i_142121++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142075 = 0; i_142075 < (int64_t) 16; i_142075++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142071 = 0; i_142071 < (int64_t) 16; i_142071++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_128760;
                double r_128762 = 0.0;
                
                for (int64_t i_128761 = 0; i_128761 < (int64_t) 4; i_128761++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_128763 = ((double *) mem_143245)[i_142121 * (int64_t) 64 + i_142075 * (int64_t) 4 + i_128761];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_128764 = ((double *) mem_143244)[i_142121 * (int64_t) 64 + i_142071 * (int64_t) 4 + i_128761];
                    
                    // futhark/microgpt.fut:207:113-164
                    
                    double zt_res_128765 = zt_lhs_128763 * zt_rhs_128764;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_128766 = r_128762 + zt_res_128765;
                    double r_tmp_145239 = zp_res_128766;
                    
                    r_128762 = r_tmp_145239;
                }
                defunc_0_lifted_lambda_res_128760 = r_128762;
                ((double *) mem_143335)[i_142071] = defunc_0_lifted_lambda_res_128760;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143330, i_142075 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143335, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142083 = 0; i_142083 < (int64_t) 16; i_142083++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142079 = 0; i_142079 < (int64_t) 16; i_142079++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_128781 = ((double *) mem_143330)[i_142083 * (int64_t) 16 + i_142079];
                
                // futhark/microgpt.fut:208:47-78
                
                double zs_res_128782 = zs_lhs_128781 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_128783 = ((double *) mask_mem_143132.mem)[i_142083 * (int64_t) 16 + i_142079];
                
                // futhark/microgpt.fut:208:65-102
                
                double zp_res_128784 = zs_res_128782 + zp_rhs_128783;
                
                ((double *) mem_143351)[i_142079] = zp_res_128784;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143346, i_142083 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143351, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142101 = 0; i_142101 < (int64_t) 16; i_142101++) {
            // futhark/microgpt.fut:105:13-33
            
            double defunc_0_reduce_res_132016;
            double redout_142085 = -INFINITY;
            
            for (int64_t i_142086 = 0; i_142086 < (int64_t) 16; i_142086++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_131940 = ((double *) mem_143346)[i_142101 * (int64_t) 16 + i_142086];
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_128805 = fmax64(lifted_lambda_res_131940, redout_142085);
                double redout_tmp_145243 = max_res_128805;
                
                redout_142085 = redout_tmp_145243;
            }
            defunc_0_reduce_res_132016 = redout_142085;
            // futhark/microgpt.fut:210:67-76
            
            double neg_res_128806 = -defunc_0_reduce_res_132016;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142089 = 0; i_142089 < (int64_t) 16; i_142089++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_128813 = ((double *) mem_143346)[i_142101 * (int64_t) 16 + i_142089];
                
                // futhark/microgpt.fut:210:44-76
                
                double zp_res_128814 = neg_res_128806 + zp_lhs_128813;
                
                // futhark/microgpt.fut:210:37-76
                
                double exp_res_128815 = futrts_exp64(zp_res_128814);
                
                ((double *) mem_143367)[i_142089] = exp_res_128815;
            }
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128817;
            double r_128819 = 0.0;
            
            for (int64_t i_128818 = 0; i_128818 < (int64_t) 16; i_128818++) {
                // futhark/microgpt.fut:211:36-46
                
                double lifted_lambda_res_128820 = ((double *) mem_143367)[i_128818];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128821 = r_128819 + lifted_lambda_res_128820;
                double r_tmp_145245 = zp_res_128821;
                
                r_128819 = r_tmp_145245;
            }
            defunc_0_lifted_lambda_res_128817 = r_128819;
            // futhark/microgpt.fut:212:53-64
            
            double zs_res_128822 = 1.0 / defunc_0_lifted_lambda_res_128817;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142093 = 0; i_142093 < (int64_t) 16; i_142093++) {
                // futhark/microgpt.fut:212:37-47
                
                double zt_lhs_128829 = ((double *) mem_143367)[i_142093];
                
                // futhark/microgpt.fut:212:37-64
                
                double zt_res_128830 = zs_res_128822 * zt_lhs_128829;
                
                ((double *) mem_143374)[i_142093] = zt_res_128830;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142097 = 0; i_142097 < (int64_t) 16; i_142097++) {
                // futhark/microgpt.fut:213:4-14
                
                double lifted_lambda_res_128838 = ((double *) mem_143374)[i_142097];
                
                ((double *) mem_143381)[i_142097] = lifted_lambda_res_128838;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143362, i_142101 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143381, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142109 = 0; i_142109 < (int64_t) 16; i_142109++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142105 = 0; i_142105 < (int64_t) 4; i_142105++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_128853;
                double r_128855 = 0.0;
                
                for (int64_t i_128854 = 0; i_128854 < (int64_t) 16; i_128854++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_128856 = ((double *) mem_143362)[i_142109 * (int64_t) 16 + i_128854];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_128857 = ((double *) mem_143243)[i_142121 * (int64_t) 64 + i_128854 * (int64_t) 4 + i_142105];
                    
                    // futhark/microgpt.fut:214:66-111
                    
                    double zt_res_128858 = zt_lhs_128856 * zt_rhs_128857;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_128859 = r_128855 + zt_res_128858;
                    double r_tmp_145250 = zp_res_128859;
                    
                    r_128855 = r_tmp_145250;
                }
                defunc_0_lifted_lambda_res_128853 = r_128855;
                ((double *) mem_143397)[i_142105] = defunc_0_lifted_lambda_res_128853;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143392, i_142109 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143397, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142117 = 0; i_142117 < (int64_t) 16; i_142117++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142113 = 0; i_142113 < (int64_t) 4; i_142113++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_128874 = ((double *) mem_143392)[i_142117 * (int64_t) 4 + i_142113];
                
                ((double *) mem_143413)[i_142113] = lifted_lambda_res_128874;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143408, i_142117 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143413, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143324, i_142121 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_143408, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143429_cached_sizze_145596 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143429, &mem_143429_cached_sizze_145596, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143434_cached_sizze_145597 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143434, &mem_143434_cached_sizze_145597, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142129 = 0; i_142129 < (int64_t) 16; i_142129++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142125 = 0; i_142125 < (int64_t) 16; i_142125++) {
            // futhark/microgpt.fut:216:54-57
            
            int64_t tmp_128886 = sdiv64(i_142125, (int64_t) 4);
            
            // futhark/microgpt.fut:216:44-59
            
            bool x_128887 = sle64((int64_t) 0, tmp_128886);
            
            // futhark/microgpt.fut:216:44-59
            
            bool y_128888 = slt64(tmp_128886, (int64_t) 4);
            
            // futhark/microgpt.fut:216:44-59
            
            bool bounds_check_128889 = x_128887 && y_128888;
            
            // futhark/microgpt.fut:216:44-59
            
            bool index_certs_128890;
            
            if (!bounds_check_128889) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_128886, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:216:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:216:15-80\n   #6  futhark/microgpt.fut:442:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:216:74-77
            
            int64_t tmp_128891 = smod64(i_142125, (int64_t) 4);
            
            // futhark/microgpt.fut:216:44-79
            
            bool x_128892 = sle64((int64_t) 0, tmp_128891);
            
            // futhark/microgpt.fut:216:44-79
            
            bool y_128893 = slt64(tmp_128891, (int64_t) 4);
            
            // futhark/microgpt.fut:216:44-79
            
            bool bounds_check_128894 = x_128892 && y_128893;
            
            // futhark/microgpt.fut:216:44-79
            
            bool index_certs_128895;
            
            if (!bounds_check_128894) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_128891, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:216:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:216:15-80\n   #6  futhark/microgpt.fut:442:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_128896 = ((double *) mem_143324)[tmp_128886 * (int64_t) 64 + i_142129 * (int64_t) 4 + tmp_128891];
            
            ((double *) mem_143434)[i_142125] = lifted_lambda_res_128896;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143429, i_142129 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143434, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143445_cached_sizze_145598 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143445, &mem_143445_cached_sizze_145598, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143450_cached_sizze_145599 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143450, &mem_143450_cached_sizze_145599, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142137 = 0; i_142137 < (int64_t) 16; i_142137++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142133 = 0; i_142133 < (int64_t) 16; i_142133++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128911;
            double r_128913 = 0.0;
            
            for (int64_t i_128912 = 0; i_128912 < (int64_t) 16; i_128912++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128914 = ((double *) wout_mem_143123.mem)[i_142133 * (int64_t) 16 + i_128912];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128915 = ((double *) mem_143429)[i_142137 * (int64_t) 16 + i_128912];
                
                // futhark/microgpt.fut:217:67-106
                
                double zt_res_128916 = zt_lhs_128914 * zt_rhs_128915;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128917 = r_128913 + zt_res_128916;
                double r_tmp_145257 = zp_res_128917;
                
                r_128913 = r_tmp_145257;
            }
            defunc_0_lifted_lambda_res_128911 = r_128913;
            ((double *) mem_143450)[i_142133] = defunc_0_lifted_lambda_res_128911;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143445, i_142137 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143450, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143461_cached_sizze_145600 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143461, &mem_143461_cached_sizze_145600, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143466_cached_sizze_145601 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143466, &mem_143466_cached_sizze_145601, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142145 = 0; i_142145 < (int64_t) 16; i_142145++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142141 = 0; i_142141 < (int64_t) 16; i_142141++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_128932 = ((double *) mem_143445)[i_142145 * (int64_t) 16 + i_142141];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_128933 = ((double *) mem_143149)[i_142145 * (int64_t) 16 + i_142141];
            
            // futhark/microgpt.fut:218:46-84
            
            double zp_res_128934 = zp_lhs_128932 + zp_rhs_128933;
            
            ((double *) mem_143466)[i_142141] = zp_res_128934;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143461, i_142145 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143466, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143477_cached_sizze_145602 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143477, &mem_143477_cached_sizze_145602, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143482_cached_sizze_145603 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143482, &mem_143482_cached_sizze_145603, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143489_cached_sizze_145604 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143489, &mem_143489_cached_sizze_145604, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142157 = 0; i_142157 < (int64_t) 16; i_142157++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_128943;
        double r_128945 = 0.0;
        
        for (int64_t i_128944 = 0; i_128944 < (int64_t) 16; i_128944++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_128946 = ((double *) mem_143461)[i_142157 * (int64_t) 16 + i_128944];
            
            // futhark/microgpt.fut:219:79-118
            
            double zt_res_128947 = zt_lhs_128946 * zt_lhs_128946;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_128948 = r_128945 + zt_res_128947;
            double r_tmp_145261 = zp_res_128948;
            
            r_128945 = r_tmp_145261;
        }
        defunc_0_lifted_lambda_res_128943 = r_128945;
        // futhark/microgpt.fut:219:58-136
        
        double zs_res_128949 = defunc_0_lifted_lambda_res_128943 / 16.0;
        
        // futhark/microgpt.fut:220:24-55
        
        double zp_res_128950 = 1.0e-5 + zs_res_128949;
        
        // futhark/microgpt.fut:220:16-55
        
        double sqrt_res_128951 = futrts_sqrt64(zp_res_128950);
        
        // futhark/microgpt.fut:221:60-71
        
        double zs_res_128952 = 1.0 / sqrt_res_128951;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142149 = 0; i_142149 < (int64_t) 16; i_142149++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_128959 = ((double *) mem_143461)[i_142157 * (int64_t) 16 + i_142149];
            
            // futhark/microgpt.fut:221:37-71
            
            double zt_res_128960 = zs_res_128952 * zt_lhs_128959;
            
            ((double *) mem_143482)[i_142149] = zt_res_128960;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142153 = 0; i_142153 < (int64_t) 16; i_142153++) {
            // futhark/microgpt.fut:222:4-14
            
            double lifted_lambda_res_128968 = ((double *) mem_143482)[i_142153];
            
            ((double *) mem_143489)[i_142153] = lifted_lambda_res_128968;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143477, i_142157 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143489, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143500_cached_sizze_145605 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143500, &mem_143500_cached_sizze_145605, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143505_cached_sizze_145606 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143505, &mem_143505_cached_sizze_145606, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142165 = 0; i_142165 < (int64_t) 16; i_142165++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142161 = 0; i_142161 < (int64_t) 64; i_142161++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128984;
            double r_128986 = 0.0;
            
            for (int64_t i_128985 = 0; i_128985 < (int64_t) 16; i_128985++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128987 = ((double *) wup_mem_143127.mem)[i_142161 * (int64_t) 16 + i_128985];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128988 = ((double *) mem_143477)[i_142165 * (int64_t) 16 + i_128985];
                
                // futhark/microgpt.fut:223:67-106
                
                double zt_res_128989 = zt_lhs_128987 * zt_rhs_128988;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128990 = r_128986 + zt_res_128989;
                double r_tmp_145266 = zp_res_128990;
                
                r_128986 = r_tmp_145266;
            }
            defunc_0_lifted_lambda_res_128984 = r_128986;
            ((double *) mem_143505)[i_142161] = defunc_0_lifted_lambda_res_128984;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143500, i_142165 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143505, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143516_cached_sizze_145607 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143516, &mem_143516_cached_sizze_145607, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143521_cached_sizze_145608 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143521, &mem_143521_cached_sizze_145608, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142173 = 0; i_142173 < (int64_t) 16; i_142173++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142169 = 0; i_142169 < (int64_t) 64; i_142169++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_129005 = ((double *) mem_143500)[i_142173 * (int64_t) 64 + i_142169];
            
            // futhark/microgpt.fut:224:45-73
            
            double max_res_129006 = fmax64(0.0, max_arg0_129005);
            
            ((double *) mem_143521)[i_142169] = max_res_129006;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143516, i_142173 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143521, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143532_cached_sizze_145609 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143532, &mem_143532_cached_sizze_145609, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143537_cached_sizze_145610 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143537, &mem_143537_cached_sizze_145610, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142181 = 0; i_142181 < (int64_t) 16; i_142181++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142177 = 0; i_142177 < (int64_t) 16; i_142177++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129021;
            double r_129023 = 0.0;
            
            for (int64_t i_129022 = 0; i_129022 < (int64_t) 64; i_129022++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_129024 = ((double *) wdown_mem_143121.mem)[i_142177 * (int64_t) 64 + i_129022];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_129025 = ((double *) mem_143516)[i_142181 * (int64_t) 64 + i_129022];
                
                // futhark/microgpt.fut:225:67-108
                
                double zt_res_129026 = zt_lhs_129024 * zt_rhs_129025;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129027 = r_129023 + zt_res_129026;
                double r_tmp_145271 = zp_res_129027;
                
                r_129023 = r_tmp_145271;
            }
            defunc_0_lifted_lambda_res_129021 = r_129023;
            ((double *) mem_143537)[i_142177] = defunc_0_lifted_lambda_res_129021;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143532, i_142181 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143537, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143548_cached_sizze_145611 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143548, &mem_143548_cached_sizze_145611, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143553_cached_sizze_145612 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143553, &mem_143553_cached_sizze_145612, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142189 = 0; i_142189 < (int64_t) 16; i_142189++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142185 = 0; i_142185 < (int64_t) 16; i_142185++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_129042 = ((double *) mem_143532)[i_142189 * (int64_t) 16 + i_142185];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_129043 = ((double *) mem_143461)[i_142189 * (int64_t) 16 + i_142185];
            
            // futhark/microgpt.fut:226:46-85
            
            double zp_res_129044 = zp_lhs_129042 + zp_rhs_129043;
            
            ((double *) mem_143553)[i_142185] = zp_res_129044;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143548, i_142189 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143553, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143564_cached_sizze_145613 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_143564, &mem_143564_cached_sizze_145613, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143569_cached_sizze_145614 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_143569, &mem_143569_cached_sizze_145614, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142197 = 0; i_142197 < (int64_t) 16; i_142197++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142193 = 0; i_142193 < (int64_t) 27; i_142193++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129060;
            double r_129062 = 0.0;
            
            for (int64_t i_129061 = 0; i_129061 < (int64_t) 16; i_129061++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_129063 = ((double *) wvoc_mem_143129.mem)[i_142193 * (int64_t) 16 + i_129061];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_129064 = ((double *) mem_143548)[i_142197 * (int64_t) 16 + i_129061];
                
                // futhark/microgpt.fut:227:67-107
                
                double zt_res_129065 = zt_lhs_129063 * zt_rhs_129064;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129066 = r_129062 + zt_res_129065;
                double r_tmp_145276 = zp_res_129066;
                
                r_129062 = r_tmp_145276;
            }
            defunc_0_lifted_lambda_res_129060 = r_129062;
            ((double *) mem_143569)[i_142193] = defunc_0_lifted_lambda_res_129060;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143564, i_142197 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143569, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143580, (int64_t) 128, "mem_143580")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143584_cached_sizze_145615 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_143584, &mem_143584_cached_sizze_145615, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143591_cached_sizze_145616 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_143591, &mem_143591_cached_sizze_145616, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142211 = 0; i_142211 < (int64_t) 16; i_142211++) {
        double x_132039;
        double redout_142199 = -INFINITY;
        
        for (int64_t i_142200 = 0; i_142200 < (int64_t) 27; i_142200++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_131986 = ((double *) mem_143564)[i_142211 * (int64_t) 27 + i_142200];
            
            // futhark/microgpt.fut:105:13-33
            
            double max_res_129090 = fmax64(lifted_lambda_res_131986, redout_142199);
            double redout_tmp_145278 = max_res_129090;
            
            redout_142199 = redout_tmp_145278;
        }
        x_132039 = redout_142199;
        // futhark/microgpt.fut:229:67-76
        
        double neg_res_129091 = -x_132039;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_129075;
        double r_129077 = 0.0;
        
        for (int64_t i_129076 = 0; i_129076 < (int64_t) 27; i_129076++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142203 = 0; i_142203 < (int64_t) 27; i_142203++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_129098 = ((double *) mem_143564)[i_142211 * (int64_t) 27 + i_142203];
                
                // futhark/microgpt.fut:229:44-76
                
                double zp_res_129099 = neg_res_129091 + zp_lhs_129098;
                
                // futhark/microgpt.fut:229:37-76
                
                double exp_res_129100 = futrts_exp64(zp_res_129099);
                
                ((double *) mem_143584)[i_142203] = exp_res_129100;
            }
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129102;
            double r_129104 = 0.0;
            
            for (int64_t i_129103 = 0; i_129103 < (int64_t) 27; i_129103++) {
                // futhark/microgpt.fut:230:36-46
                
                double lifted_lambda_res_129105 = ((double *) mem_143584)[i_129103];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129106 = r_129104 + lifted_lambda_res_129105;
                double r_tmp_145281 = zp_res_129106;
                
                r_129104 = r_tmp_145281;
            }
            defunc_0_lifted_lambda_res_129102 = r_129104;
            // futhark/microgpt.fut:231:53-64
            
            double zs_res_129107 = 1.0 / defunc_0_lifted_lambda_res_129102;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142207 = 0; i_142207 < (int64_t) 27; i_142207++) {
                // futhark/microgpt.fut:231:37-47
                
                double zt_lhs_129114 = ((double *) mem_143584)[i_142207];
                
                // futhark/microgpt.fut:231:37-64
                
                double zt_res_129115 = zs_res_129107 * zt_lhs_129114;
                
                ((double *) mem_143591)[i_142207] = zt_res_129115;
            }
            // futhark/microgpt.fut:232:12-22
            
            double log_arg0_129117 = ((double *) mem_143591)[i_129076];
            
            // futhark/microgpt.fut:232:6-22
            
            double log_res_129118 = futrts_log64(log_arg0_129117);
            
            // futhark/microgpt.fut:61:46-49
            
            double zt_rhs_129119 = ((double *) target_mem_143131.mem)[i_142211 * (int64_t) 27 + i_129076];
            
            // futhark/microgpt.fut:232:6-48
            
            double zt_res_129120 = log_res_129118 * zt_rhs_129119;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_129121 = r_129077 + zt_res_129120;
            double r_tmp_145279 = zp_res_129121;
            
            r_129077 = r_tmp_145279;
        }
        defunc_0_lifted_lambda_res_129075 = r_129077;
        // futhark/microgpt.fut:228:37-232:54
        
        double neg_res_129122 = -defunc_0_lifted_lambda_res_129075;
        
        ((double *) mem_143580.mem)[i_142211] = neg_res_129122;
    }
    // futhark/microgpt.fut:61:13-49
    
    double defunc_0_lifted_lambda_res_129124;
    double r_129126 = 0.0;
    
    for (int64_t i_129125 = 0; i_129125 < (int64_t) 16; i_129125++) {
        // futhark/microgpt.fut:233:37-47
        
        double lifted_lambda_res_129127 = ((double *) mem_143580.mem)[i_129125];
        
        // futhark/microgpt.fut:61:40-49
        
        double zp_res_129128 = r_129126 + lifted_lambda_res_129127;
        double r_tmp_145283 = zp_res_129128;
        
        r_129126 = r_tmp_145283;
    }
    defunc_0_lifted_lambda_res_129124 = r_129126;
    // futhark/microgpt.fut:233:17-64
    
    double zs_res_129129 = defunc_0_lifted_lambda_res_129124 / 16.0;
    
    if (memblock_set(ctx, &mem_out_145206, &mem_143580, "mem_143580") != 0)
        return 1;
    prim_out_145207 = zs_res_129129;
    if (memblock_set(ctx, &*mem_out_p_145558, &mem_out_145206, "mem_out_145206") != 0)
        return 1;
    *out_prim_out_145559 = prim_out_145207;
    
  cleanup:
    {
        free(mem_143133);
        free(mem_143138);
        free(mem_143149);
        free(mem_143154);
        free(mem_143161);
        free(mem_143172);
        free(mem_143177);
        free(mem_143184);
        free(mem_143195);
        free(mem_143196);
        free(mem_143197);
        free(mem_143210);
        free(mem_143211);
        free(mem_143212);
        free(mem_143243);
        free(mem_143244);
        free(mem_143245);
        free(mem_143261);
        free(mem_143262);
        free(mem_143263);
        free(mem_143276);
        free(mem_143277);
        free(mem_143278);
        free(mem_143324);
        free(mem_143330);
        free(mem_143335);
        free(mem_143346);
        free(mem_143351);
        free(mem_143362);
        free(mem_143367);
        free(mem_143374);
        free(mem_143381);
        free(mem_143392);
        free(mem_143397);
        free(mem_143408);
        free(mem_143413);
        free(mem_143429);
        free(mem_143434);
        free(mem_143445);
        free(mem_143450);
        free(mem_143461);
        free(mem_143466);
        free(mem_143477);
        free(mem_143482);
        free(mem_143489);
        free(mem_143500);
        free(mem_143505);
        free(mem_143516);
        free(mem_143521);
        free(mem_143532);
        free(mem_143537);
        free(mem_143548);
        free(mem_143553);
        free(mem_143564);
        free(mem_143569);
        free(mem_143584);
        free(mem_143591);
        if (memblock_unref(ctx, &mem_143580, "mem_143580") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145206, "mem_out_145206") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_145617, struct memblock wdown_mem_143121, struct memblock wkey_mem_143122, struct memblock wout_mem_143123, struct memblock wpe_mem_143124, struct memblock wqry_mem_143125, struct memblock wte_mem_143126, struct memblock wup_mem_143127, struct memblock wval_mem_143128, struct memblock wvoc_mem_143129, struct memblock tokens_mem_143130, struct memblock mask_mem_143131)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_143132_cached_sizze_145618 = 0;
    unsigned char *mem_143132 = NULL;
    int64_t mem_143137_cached_sizze_145619 = 0;
    unsigned char *mem_143137 = NULL;
    int64_t mem_143148_cached_sizze_145620 = 0;
    unsigned char *mem_143148 = NULL;
    int64_t mem_143153_cached_sizze_145621 = 0;
    unsigned char *mem_143153 = NULL;
    int64_t mem_143160_cached_sizze_145622 = 0;
    unsigned char *mem_143160 = NULL;
    int64_t mem_143171_cached_sizze_145623 = 0;
    unsigned char *mem_143171 = NULL;
    int64_t mem_143176_cached_sizze_145624 = 0;
    unsigned char *mem_143176 = NULL;
    int64_t mem_143183_cached_sizze_145625 = 0;
    unsigned char *mem_143183 = NULL;
    int64_t mem_143194_cached_sizze_145626 = 0;
    unsigned char *mem_143194 = NULL;
    int64_t mem_143195_cached_sizze_145627 = 0;
    unsigned char *mem_143195 = NULL;
    int64_t mem_143196_cached_sizze_145628 = 0;
    unsigned char *mem_143196 = NULL;
    int64_t mem_143209_cached_sizze_145629 = 0;
    unsigned char *mem_143209 = NULL;
    int64_t mem_143210_cached_sizze_145630 = 0;
    unsigned char *mem_143210 = NULL;
    int64_t mem_143211_cached_sizze_145631 = 0;
    unsigned char *mem_143211 = NULL;
    int64_t mem_143242_cached_sizze_145632 = 0;
    unsigned char *mem_143242 = NULL;
    int64_t mem_143243_cached_sizze_145633 = 0;
    unsigned char *mem_143243 = NULL;
    int64_t mem_143244_cached_sizze_145634 = 0;
    unsigned char *mem_143244 = NULL;
    int64_t mem_143260_cached_sizze_145635 = 0;
    unsigned char *mem_143260 = NULL;
    int64_t mem_143261_cached_sizze_145636 = 0;
    unsigned char *mem_143261 = NULL;
    int64_t mem_143262_cached_sizze_145637 = 0;
    unsigned char *mem_143262 = NULL;
    int64_t mem_143275_cached_sizze_145638 = 0;
    unsigned char *mem_143275 = NULL;
    int64_t mem_143276_cached_sizze_145639 = 0;
    unsigned char *mem_143276 = NULL;
    int64_t mem_143277_cached_sizze_145640 = 0;
    unsigned char *mem_143277 = NULL;
    int64_t mem_143323_cached_sizze_145641 = 0;
    unsigned char *mem_143323 = NULL;
    int64_t mem_143329_cached_sizze_145642 = 0;
    unsigned char *mem_143329 = NULL;
    int64_t mem_143334_cached_sizze_145643 = 0;
    unsigned char *mem_143334 = NULL;
    int64_t mem_143345_cached_sizze_145644 = 0;
    unsigned char *mem_143345 = NULL;
    int64_t mem_143350_cached_sizze_145645 = 0;
    unsigned char *mem_143350 = NULL;
    int64_t mem_143361_cached_sizze_145646 = 0;
    unsigned char *mem_143361 = NULL;
    int64_t mem_143366_cached_sizze_145647 = 0;
    unsigned char *mem_143366 = NULL;
    int64_t mem_143373_cached_sizze_145648 = 0;
    unsigned char *mem_143373 = NULL;
    int64_t mem_143380_cached_sizze_145649 = 0;
    unsigned char *mem_143380 = NULL;
    int64_t mem_143391_cached_sizze_145650 = 0;
    unsigned char *mem_143391 = NULL;
    int64_t mem_143396_cached_sizze_145651 = 0;
    unsigned char *mem_143396 = NULL;
    int64_t mem_143407_cached_sizze_145652 = 0;
    unsigned char *mem_143407 = NULL;
    int64_t mem_143412_cached_sizze_145653 = 0;
    unsigned char *mem_143412 = NULL;
    int64_t mem_143428_cached_sizze_145654 = 0;
    unsigned char *mem_143428 = NULL;
    int64_t mem_143433_cached_sizze_145655 = 0;
    unsigned char *mem_143433 = NULL;
    int64_t mem_143444_cached_sizze_145656 = 0;
    unsigned char *mem_143444 = NULL;
    int64_t mem_143449_cached_sizze_145657 = 0;
    unsigned char *mem_143449 = NULL;
    int64_t mem_143460_cached_sizze_145658 = 0;
    unsigned char *mem_143460 = NULL;
    int64_t mem_143465_cached_sizze_145659 = 0;
    unsigned char *mem_143465 = NULL;
    int64_t mem_143476_cached_sizze_145660 = 0;
    unsigned char *mem_143476 = NULL;
    int64_t mem_143481_cached_sizze_145661 = 0;
    unsigned char *mem_143481 = NULL;
    int64_t mem_143488_cached_sizze_145662 = 0;
    unsigned char *mem_143488 = NULL;
    int64_t mem_143499_cached_sizze_145663 = 0;
    unsigned char *mem_143499 = NULL;
    int64_t mem_143504_cached_sizze_145664 = 0;
    unsigned char *mem_143504 = NULL;
    int64_t mem_143515_cached_sizze_145665 = 0;
    unsigned char *mem_143515 = NULL;
    int64_t mem_143520_cached_sizze_145666 = 0;
    unsigned char *mem_143520 = NULL;
    int64_t mem_143531_cached_sizze_145667 = 0;
    unsigned char *mem_143531 = NULL;
    int64_t mem_143536_cached_sizze_145668 = 0;
    unsigned char *mem_143536 = NULL;
    int64_t mem_143547_cached_sizze_145669 = 0;
    unsigned char *mem_143547 = NULL;
    int64_t mem_143552_cached_sizze_145670 = 0;
    unsigned char *mem_143552 = NULL;
    int64_t mem_143563_cached_sizze_145671 = 0;
    unsigned char *mem_143563 = NULL;
    int64_t mem_143568_cached_sizze_145672 = 0;
    unsigned char *mem_143568 = NULL;
    int64_t mem_143584_cached_sizze_145673 = 0;
    unsigned char *mem_143584 = NULL;
    struct memblock mem_143579;
    
    mem_143579.references = NULL;
    
    struct memblock mem_out_145206;
    
    mem_out_145206.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (mem_143132_cached_sizze_145618 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143132, &mem_143132_cached_sizze_145618, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143137_cached_sizze_145619 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143137, &mem_143137_cached_sizze_145619, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_141993 = 0; i_141993 < (int64_t) 16; i_141993++) {
        // futhark/microgpt.fut:436:41-50
        
        int64_t tmp_128514 = ((int64_t *) tokens_mem_143130.mem)[i_141993];
        
        // futhark/microgpt.fut:436:37-51
        
        bool x_128515 = sle64((int64_t) 0, tmp_128514);
        
        // futhark/microgpt.fut:436:37-51
        
        bool y_128516 = slt64(tmp_128514, (int64_t) 27);
        
        // futhark/microgpt.fut:436:37-51
        
        bool bounds_check_128517 = x_128515 && y_128516;
        
        // futhark/microgpt.fut:436:37-51
        
        bool index_certs_128518;
        
        if (!bounds_check_128517) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_128514, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:436:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:436:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_141989 = 0; i_141989 < (int64_t) 16; i_141989++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_128525 = ((double *) wte_mem_143126.mem)[tmp_128514 * (int64_t) 16 + i_141989];
            
            ((double *) mem_143137)[i_141989] = lifted_lambda_res_128525;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143132, i_141993 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143137, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143148_cached_sizze_145620 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143148, &mem_143148_cached_sizze_145620, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143153_cached_sizze_145621 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143153, &mem_143153_cached_sizze_145621, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143160_cached_sizze_145622 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143160, &mem_143160_cached_sizze_145622, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142005 = 0; i_142005 < (int64_t) 16; i_142005++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_128551;
        double r_128553 = 0.0;
        
        for (int64_t i_128552 = 0; i_128552 < (int64_t) 16; i_128552++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_128554 = ((double *) wpe_mem_143124.mem)[i_142005 * (int64_t) 16 + i_128552];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_128555 = ((double *) mem_143132)[i_142005 * (int64_t) 16 + i_128552];
            
            // futhark/microgpt.fut:138:76-116
            
            double zp_res_128556 = zp_lhs_128554 + zp_rhs_128555;
            
            // futhark/microgpt.fut:138:94-163
            
            double zt_res_128557 = zp_res_128556 * zp_res_128556;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_128558 = r_128553 + zt_res_128557;
            double r_tmp_145210 = zp_res_128558;
            
            r_128553 = r_tmp_145210;
        }
        defunc_0_lifted_lambda_res_128551 = r_128553;
        // futhark/microgpt.fut:138:54-182
        
        double zs_res_128559 = defunc_0_lifted_lambda_res_128551 / 16.0;
        
        // futhark/microgpt.fut:139:24-55
        
        double zp_res_128560 = 1.0e-5 + zs_res_128559;
        
        // futhark/microgpt.fut:139:16-55
        
        double sqrt_res_128561 = futrts_sqrt64(zp_res_128560);
        
        // futhark/microgpt.fut:140:85-96
        
        double zs_res_128562 = 1.0 / sqrt_res_128561;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_141997 = 0; i_141997 < (int64_t) 16; i_141997++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_128569 = ((double *) wpe_mem_143124.mem)[i_142005 * (int64_t) 16 + i_141997];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_128570 = ((double *) mem_143132)[i_142005 * (int64_t) 16 + i_141997];
            
            // futhark/microgpt.fut:140:38-78
            
            double zp_res_128571 = zp_lhs_128569 + zp_rhs_128570;
            
            // futhark/microgpt.fut:140:56-96
            
            double zt_res_128572 = zs_res_128562 * zp_res_128571;
            
            ((double *) mem_143153)[i_141997] = zt_res_128572;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142001 = 0; i_142001 < (int64_t) 16; i_142001++) {
            // futhark/microgpt.fut:141:4-14
            
            double lifted_lambda_res_128580 = ((double *) mem_143153)[i_142001];
            
            ((double *) mem_143160)[i_142001] = lifted_lambda_res_128580;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143148, i_142005 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143160, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143171_cached_sizze_145623 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143171, &mem_143171_cached_sizze_145623, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143176_cached_sizze_145624 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143176, &mem_143176_cached_sizze_145624, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143183_cached_sizze_145625 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143183, &mem_143183_cached_sizze_145625, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142017 = 0; i_142017 < (int64_t) 16; i_142017++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_128589;
        double r_128591 = 0.0;
        
        for (int64_t i_128590 = 0; i_128590 < (int64_t) 16; i_128590++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_128592 = ((double *) mem_143148)[i_142017 * (int64_t) 16 + i_128590];
            
            // futhark/microgpt.fut:142:78-115
            
            double zt_res_128593 = zt_lhs_128592 * zt_lhs_128592;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_128594 = r_128591 + zt_res_128593;
            double r_tmp_145214 = zp_res_128594;
            
            r_128591 = r_tmp_145214;
        }
        defunc_0_lifted_lambda_res_128589 = r_128591;
        // futhark/microgpt.fut:142:57-133
        
        double zs_res_128595 = defunc_0_lifted_lambda_res_128589 / 16.0;
        
        // futhark/microgpt.fut:143:24-55
        
        double zp_res_128596 = 1.0e-5 + zs_res_128595;
        
        // futhark/microgpt.fut:143:16-55
        
        double sqrt_res_128597 = futrts_sqrt64(zp_res_128596);
        
        // futhark/microgpt.fut:144:59-70
        
        double zs_res_128598 = 1.0 / sqrt_res_128597;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142009 = 0; i_142009 < (int64_t) 16; i_142009++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_128605 = ((double *) mem_143148)[i_142017 * (int64_t) 16 + i_142009];
            
            // futhark/microgpt.fut:144:37-70
            
            double zt_res_128606 = zs_res_128598 * zt_lhs_128605;
            
            ((double *) mem_143176)[i_142009] = zt_res_128606;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142013 = 0; i_142013 < (int64_t) 16; i_142013++) {
            // futhark/microgpt.fut:145:4-14
            
            double lifted_lambda_res_128614 = ((double *) mem_143176)[i_142013];
            
            ((double *) mem_143183)[i_142013] = lifted_lambda_res_128614;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143171, i_142017 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143183, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143194_cached_sizze_145626 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143194, &mem_143194_cached_sizze_145626, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143195_cached_sizze_145627 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143195, &mem_143195_cached_sizze_145627, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143196_cached_sizze_145628 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143196, &mem_143196_cached_sizze_145628, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143209_cached_sizze_145629 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143209, &mem_143209_cached_sizze_145629, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143210_cached_sizze_145630 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143210, &mem_143210_cached_sizze_145630, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143211_cached_sizze_145631 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143211, &mem_143211_cached_sizze_145631, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142035 = 0; i_142035 < (int64_t) 16; i_142035++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142025 = 0; i_142025 < (int64_t) 16; i_142025++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_131696;
            double r_131698 = 0.0;
            
            for (int64_t i_131697 = 0; i_131697 < (int64_t) 16; i_131697++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_131699 = ((double *) wqry_mem_143125.mem)[i_142025 * (int64_t) 16 + i_131697];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_131700 = ((double *) mem_143171)[i_142035 * (int64_t) 16 + i_131697];
                
                // futhark/microgpt.fut:146:66-105
                
                double zt_res_131701 = zt_lhs_131699 * zt_rhs_131700;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_131702 = r_131698 + zt_res_131701;
                double r_tmp_145223 = zp_res_131702;
                
                r_131698 = r_tmp_145223;
            }
            defunc_0_lifted_lambda_res_131696 = r_131698;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_131709;
            double r_131711 = 0.0;
            
            for (int64_t i_131710 = 0; i_131710 < (int64_t) 16; i_131710++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_131712 = ((double *) wkey_mem_143122.mem)[i_142025 * (int64_t) 16 + i_131710];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_131713 = ((double *) mem_143171)[i_142035 * (int64_t) 16 + i_131710];
                
                // futhark/microgpt.fut:147:66-105
                
                double zt_res_131714 = zt_lhs_131712 * zt_rhs_131713;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_131715 = r_131711 + zt_res_131714;
                double r_tmp_145224 = zp_res_131715;
                
                r_131711 = r_tmp_145224;
            }
            defunc_0_lifted_lambda_res_131709 = r_131711;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_131725;
            double r_131727 = 0.0;
            
            for (int64_t i_131726 = 0; i_131726 < (int64_t) 16; i_131726++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_131728 = ((double *) wval_mem_143128.mem)[i_142025 * (int64_t) 16 + i_131726];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_131729 = ((double *) mem_143171)[i_142035 * (int64_t) 16 + i_131726];
                
                // futhark/microgpt.fut:148:66-105
                
                double zt_res_131730 = zt_lhs_131728 * zt_rhs_131729;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_131731 = r_131727 + zt_res_131730;
                double r_tmp_145225 = zp_res_131731;
                
                r_131727 = r_tmp_145225;
            }
            defunc_0_lifted_lambda_res_131725 = r_131727;
            ((double *) mem_143209)[i_142025] = defunc_0_lifted_lambda_res_131725;
            ((double *) mem_143210)[i_142025] = defunc_0_lifted_lambda_res_131709;
            ((double *) mem_143211)[i_142025] = defunc_0_lifted_lambda_res_131696;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143194, i_142035 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143209, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143195, i_142035 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143210, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143196, i_142035 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143211, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143242_cached_sizze_145632 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143242, &mem_143242_cached_sizze_145632, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143243_cached_sizze_145633 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143243, &mem_143243_cached_sizze_145633, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143244_cached_sizze_145634 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143244, &mem_143244_cached_sizze_145634, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143260_cached_sizze_145635 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143260, &mem_143260_cached_sizze_145635, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143261_cached_sizze_145636 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143261, &mem_143261_cached_sizze_145636, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143262_cached_sizze_145637 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143262, &mem_143262_cached_sizze_145637, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143275_cached_sizze_145638 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_143275, &mem_143275_cached_sizze_145638, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143276_cached_sizze_145639 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_143276, &mem_143276_cached_sizze_145639, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143277_cached_sizze_145640 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_143277, &mem_143277_cached_sizze_145640, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142065 = 0; i_142065 < (int64_t) 4; i_142065++) {
        // futhark/microgpt.fut:149:69-72
        
        int64_t zp_lhs_131572 = mul64((int64_t) 4, i_142065);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142055 = 0; i_142055 < (int64_t) 16; i_142055++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142045 = 0; i_142045 < (int64_t) 4; i_142045++) {
                // futhark/microgpt.fut:149:74-81
                
                int64_t tmp_131889 = add64(zp_lhs_131572, i_142045);
                
                // futhark/microgpt.fut:149:51-83
                
                bool x_131890 = sle64((int64_t) 0, tmp_131889);
                
                // futhark/microgpt.fut:149:51-83
                
                bool y_131891 = slt64(tmp_131889, (int64_t) 16);
                
                // futhark/microgpt.fut:149:51-83
                
                bool bounds_check_131892 = x_131890 && y_131891;
                
                // futhark/microgpt.fut:149:51-83
                
                bool index_certs_131893;
                
                if (!bounds_check_131892) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_131889, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:149:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:149:15-84\n   #9  futhark/microgpt.fut:437:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_131894 = ((double *) mem_143196)[i_142055 * (int64_t) 16 + tmp_131889];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_131902 = ((double *) mem_143195)[i_142055 * (int64_t) 16 + tmp_131889];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_131913 = ((double *) mem_143194)[i_142055 * (int64_t) 16 + tmp_131889];
                
                ((double *) mem_143275)[i_142045] = lifted_lambda_res_131913;
                ((double *) mem_143276)[i_142045] = lifted_lambda_res_131902;
                ((double *) mem_143277)[i_142045] = lifted_lambda_res_131894;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143260, i_142055 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143275, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143261, i_142055 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143276, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143262, i_142055 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143277, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143242, i_142065 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_143260, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143243, i_142065 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_143261, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143244, i_142065 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_143262, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143323_cached_sizze_145641 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143323, &mem_143323_cached_sizze_145641, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143329_cached_sizze_145642 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143329, &mem_143329_cached_sizze_145642, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143334_cached_sizze_145643 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143334, &mem_143334_cached_sizze_145643, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143345_cached_sizze_145644 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143345, &mem_143345_cached_sizze_145644, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143350_cached_sizze_145645 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143350, &mem_143350_cached_sizze_145645, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143361_cached_sizze_145646 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143361, &mem_143361_cached_sizze_145646, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143366_cached_sizze_145647 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143366, &mem_143366_cached_sizze_145647, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143373_cached_sizze_145648 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143373, &mem_143373_cached_sizze_145648, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143380_cached_sizze_145649 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143380, &mem_143380_cached_sizze_145649, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143391_cached_sizze_145650 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143391, &mem_143391_cached_sizze_145650, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143396_cached_sizze_145651 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_143396, &mem_143396_cached_sizze_145651, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143407_cached_sizze_145652 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143407, &mem_143407_cached_sizze_145652, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143412_cached_sizze_145653 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_143412, &mem_143412_cached_sizze_145653, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142121 = 0; i_142121 < (int64_t) 4; i_142121++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142075 = 0; i_142075 < (int64_t) 16; i_142075++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142071 = 0; i_142071 < (int64_t) 16; i_142071++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_128759;
                double r_128761 = 0.0;
                
                for (int64_t i_128760 = 0; i_128760 < (int64_t) 4; i_128760++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_128762 = ((double *) mem_143244)[i_142121 * (int64_t) 64 + i_142075 * (int64_t) 4 + i_128760];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_128763 = ((double *) mem_143243)[i_142121 * (int64_t) 64 + i_142071 * (int64_t) 4 + i_128760];
                    
                    // futhark/microgpt.fut:152:113-164
                    
                    double zt_res_128764 = zt_lhs_128762 * zt_rhs_128763;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_128765 = r_128761 + zt_res_128764;
                    double r_tmp_145238 = zp_res_128765;
                    
                    r_128761 = r_tmp_145238;
                }
                defunc_0_lifted_lambda_res_128759 = r_128761;
                ((double *) mem_143334)[i_142071] = defunc_0_lifted_lambda_res_128759;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143329, i_142075 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143334, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142083 = 0; i_142083 < (int64_t) 16; i_142083++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142079 = 0; i_142079 < (int64_t) 16; i_142079++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_128780 = ((double *) mem_143329)[i_142083 * (int64_t) 16 + i_142079];
                
                // futhark/microgpt.fut:153:47-78
                
                double zs_res_128781 = zs_lhs_128780 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_128782 = ((double *) mask_mem_143131.mem)[i_142083 * (int64_t) 16 + i_142079];
                
                // futhark/microgpt.fut:153:65-102
                
                double zp_res_128783 = zs_res_128781 + zp_rhs_128782;
                
                ((double *) mem_143350)[i_142079] = zp_res_128783;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143345, i_142083 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143350, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142101 = 0; i_142101 < (int64_t) 16; i_142101++) {
            // futhark/microgpt.fut:105:13-33
            
            double defunc_0_reduce_res_131991;
            double redout_142085 = -INFINITY;
            
            for (int64_t i_142086 = 0; i_142086 < (int64_t) 16; i_142086++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_131940 = ((double *) mem_143345)[i_142101 * (int64_t) 16 + i_142086];
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_128804 = fmax64(lifted_lambda_res_131940, redout_142085);
                double redout_tmp_145242 = max_res_128804;
                
                redout_142085 = redout_tmp_145242;
            }
            defunc_0_reduce_res_131991 = redout_142085;
            // futhark/microgpt.fut:155:67-76
            
            double neg_res_128805 = -defunc_0_reduce_res_131991;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142089 = 0; i_142089 < (int64_t) 16; i_142089++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_128812 = ((double *) mem_143345)[i_142101 * (int64_t) 16 + i_142089];
                
                // futhark/microgpt.fut:155:44-76
                
                double zp_res_128813 = neg_res_128805 + zp_lhs_128812;
                
                // futhark/microgpt.fut:155:37-76
                
                double exp_res_128814 = futrts_exp64(zp_res_128813);
                
                ((double *) mem_143366)[i_142089] = exp_res_128814;
            }
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128816;
            double r_128818 = 0.0;
            
            for (int64_t i_128817 = 0; i_128817 < (int64_t) 16; i_128817++) {
                // futhark/microgpt.fut:156:36-46
                
                double lifted_lambda_res_128819 = ((double *) mem_143366)[i_128817];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128820 = r_128818 + lifted_lambda_res_128819;
                double r_tmp_145244 = zp_res_128820;
                
                r_128818 = r_tmp_145244;
            }
            defunc_0_lifted_lambda_res_128816 = r_128818;
            // futhark/microgpt.fut:157:53-64
            
            double zs_res_128821 = 1.0 / defunc_0_lifted_lambda_res_128816;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142093 = 0; i_142093 < (int64_t) 16; i_142093++) {
                // futhark/microgpt.fut:157:37-47
                
                double zt_lhs_128828 = ((double *) mem_143366)[i_142093];
                
                // futhark/microgpt.fut:157:37-64
                
                double zt_res_128829 = zs_res_128821 * zt_lhs_128828;
                
                ((double *) mem_143373)[i_142093] = zt_res_128829;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142097 = 0; i_142097 < (int64_t) 16; i_142097++) {
                // futhark/microgpt.fut:158:4-14
                
                double lifted_lambda_res_128837 = ((double *) mem_143373)[i_142097];
                
                ((double *) mem_143380)[i_142097] = lifted_lambda_res_128837;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143361, i_142101 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143380, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142109 = 0; i_142109 < (int64_t) 16; i_142109++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142105 = 0; i_142105 < (int64_t) 4; i_142105++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_128852;
                double r_128854 = 0.0;
                
                for (int64_t i_128853 = 0; i_128853 < (int64_t) 16; i_128853++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_128855 = ((double *) mem_143361)[i_142109 * (int64_t) 16 + i_128853];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_128856 = ((double *) mem_143242)[i_142121 * (int64_t) 64 + i_128853 * (int64_t) 4 + i_142105];
                    
                    // futhark/microgpt.fut:159:66-111
                    
                    double zt_res_128857 = zt_lhs_128855 * zt_rhs_128856;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_128858 = r_128854 + zt_res_128857;
                    double r_tmp_145249 = zp_res_128858;
                    
                    r_128854 = r_tmp_145249;
                }
                defunc_0_lifted_lambda_res_128852 = r_128854;
                ((double *) mem_143396)[i_142105] = defunc_0_lifted_lambda_res_128852;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143391, i_142109 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143396, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142117 = 0; i_142117 < (int64_t) 16; i_142117++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142113 = 0; i_142113 < (int64_t) 4; i_142113++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_128873 = ((double *) mem_143391)[i_142117 * (int64_t) 4 + i_142113];
                
                ((double *) mem_143412)[i_142113] = lifted_lambda_res_128873;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143407, i_142117 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143412, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143323, i_142121 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_143407, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143428_cached_sizze_145654 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143428, &mem_143428_cached_sizze_145654, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143433_cached_sizze_145655 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143433, &mem_143433_cached_sizze_145655, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142129 = 0; i_142129 < (int64_t) 16; i_142129++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142125 = 0; i_142125 < (int64_t) 16; i_142125++) {
            // futhark/microgpt.fut:161:54-57
            
            int64_t tmp_128885 = sdiv64(i_142125, (int64_t) 4);
            
            // futhark/microgpt.fut:161:44-59
            
            bool x_128886 = sle64((int64_t) 0, tmp_128885);
            
            // futhark/microgpt.fut:161:44-59
            
            bool y_128887 = slt64(tmp_128885, (int64_t) 4);
            
            // futhark/microgpt.fut:161:44-59
            
            bool bounds_check_128888 = x_128886 && y_128887;
            
            // futhark/microgpt.fut:161:44-59
            
            bool index_certs_128889;
            
            if (!bounds_check_128888) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_128885, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:161:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:161:15-80\n   #6  futhark/microgpt.fut:437:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:161:74-77
            
            int64_t tmp_128890 = smod64(i_142125, (int64_t) 4);
            
            // futhark/microgpt.fut:161:44-79
            
            bool x_128891 = sle64((int64_t) 0, tmp_128890);
            
            // futhark/microgpt.fut:161:44-79
            
            bool y_128892 = slt64(tmp_128890, (int64_t) 4);
            
            // futhark/microgpt.fut:161:44-79
            
            bool bounds_check_128893 = x_128891 && y_128892;
            
            // futhark/microgpt.fut:161:44-79
            
            bool index_certs_128894;
            
            if (!bounds_check_128893) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_128890, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:161:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:161:15-80\n   #6  futhark/microgpt.fut:437:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_128895 = ((double *) mem_143323)[tmp_128885 * (int64_t) 64 + i_142129 * (int64_t) 4 + tmp_128890];
            
            ((double *) mem_143433)[i_142125] = lifted_lambda_res_128895;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143428, i_142129 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143433, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143444_cached_sizze_145656 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143444, &mem_143444_cached_sizze_145656, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143449_cached_sizze_145657 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143449, &mem_143449_cached_sizze_145657, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142137 = 0; i_142137 < (int64_t) 16; i_142137++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142133 = 0; i_142133 < (int64_t) 16; i_142133++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128910;
            double r_128912 = 0.0;
            
            for (int64_t i_128911 = 0; i_128911 < (int64_t) 16; i_128911++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128913 = ((double *) wout_mem_143123.mem)[i_142133 * (int64_t) 16 + i_128911];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128914 = ((double *) mem_143428)[i_142137 * (int64_t) 16 + i_128911];
                
                // futhark/microgpt.fut:162:67-106
                
                double zt_res_128915 = zt_lhs_128913 * zt_rhs_128914;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128916 = r_128912 + zt_res_128915;
                double r_tmp_145256 = zp_res_128916;
                
                r_128912 = r_tmp_145256;
            }
            defunc_0_lifted_lambda_res_128910 = r_128912;
            ((double *) mem_143449)[i_142133] = defunc_0_lifted_lambda_res_128910;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143444, i_142137 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143449, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143460_cached_sizze_145658 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143460, &mem_143460_cached_sizze_145658, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143465_cached_sizze_145659 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143465, &mem_143465_cached_sizze_145659, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142145 = 0; i_142145 < (int64_t) 16; i_142145++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142141 = 0; i_142141 < (int64_t) 16; i_142141++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_128931 = ((double *) mem_143444)[i_142145 * (int64_t) 16 + i_142141];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_128932 = ((double *) mem_143148)[i_142145 * (int64_t) 16 + i_142141];
            
            // futhark/microgpt.fut:163:46-84
            
            double zp_res_128933 = zp_lhs_128931 + zp_rhs_128932;
            
            ((double *) mem_143465)[i_142141] = zp_res_128933;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143460, i_142145 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143465, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143476_cached_sizze_145660 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143476, &mem_143476_cached_sizze_145660, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143481_cached_sizze_145661 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143481, &mem_143481_cached_sizze_145661, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143488_cached_sizze_145662 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143488, &mem_143488_cached_sizze_145662, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142157 = 0; i_142157 < (int64_t) 16; i_142157++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_128942;
        double r_128944 = 0.0;
        
        for (int64_t i_128943 = 0; i_128943 < (int64_t) 16; i_128943++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_128945 = ((double *) mem_143460)[i_142157 * (int64_t) 16 + i_128943];
            
            // futhark/microgpt.fut:164:79-118
            
            double zt_res_128946 = zt_lhs_128945 * zt_lhs_128945;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_128947 = r_128944 + zt_res_128946;
            double r_tmp_145260 = zp_res_128947;
            
            r_128944 = r_tmp_145260;
        }
        defunc_0_lifted_lambda_res_128942 = r_128944;
        // futhark/microgpt.fut:164:58-136
        
        double zs_res_128948 = defunc_0_lifted_lambda_res_128942 / 16.0;
        
        // futhark/microgpt.fut:165:24-55
        
        double zp_res_128949 = 1.0e-5 + zs_res_128948;
        
        // futhark/microgpt.fut:165:16-55
        
        double sqrt_res_128950 = futrts_sqrt64(zp_res_128949);
        
        // futhark/microgpt.fut:166:60-71
        
        double zs_res_128951 = 1.0 / sqrt_res_128950;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142149 = 0; i_142149 < (int64_t) 16; i_142149++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_128958 = ((double *) mem_143460)[i_142157 * (int64_t) 16 + i_142149];
            
            // futhark/microgpt.fut:166:37-71
            
            double zt_res_128959 = zs_res_128951 * zt_lhs_128958;
            
            ((double *) mem_143481)[i_142149] = zt_res_128959;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142153 = 0; i_142153 < (int64_t) 16; i_142153++) {
            // futhark/microgpt.fut:167:4-14
            
            double lifted_lambda_res_128967 = ((double *) mem_143481)[i_142153];
            
            ((double *) mem_143488)[i_142153] = lifted_lambda_res_128967;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143476, i_142157 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143488, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143499_cached_sizze_145663 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143499, &mem_143499_cached_sizze_145663, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143504_cached_sizze_145664 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143504, &mem_143504_cached_sizze_145664, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142165 = 0; i_142165 < (int64_t) 16; i_142165++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142161 = 0; i_142161 < (int64_t) 64; i_142161++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128983;
            double r_128985 = 0.0;
            
            for (int64_t i_128984 = 0; i_128984 < (int64_t) 16; i_128984++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128986 = ((double *) wup_mem_143127.mem)[i_142161 * (int64_t) 16 + i_128984];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128987 = ((double *) mem_143476)[i_142165 * (int64_t) 16 + i_128984];
                
                // futhark/microgpt.fut:168:67-106
                
                double zt_res_128988 = zt_lhs_128986 * zt_rhs_128987;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128989 = r_128985 + zt_res_128988;
                double r_tmp_145265 = zp_res_128989;
                
                r_128985 = r_tmp_145265;
            }
            defunc_0_lifted_lambda_res_128983 = r_128985;
            ((double *) mem_143504)[i_142161] = defunc_0_lifted_lambda_res_128983;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143499, i_142165 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143504, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143515_cached_sizze_145665 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143515, &mem_143515_cached_sizze_145665, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143520_cached_sizze_145666 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143520, &mem_143520_cached_sizze_145666, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142173 = 0; i_142173 < (int64_t) 16; i_142173++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142169 = 0; i_142169 < (int64_t) 64; i_142169++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_129004 = ((double *) mem_143499)[i_142173 * (int64_t) 64 + i_142169];
            
            // futhark/microgpt.fut:169:45-73
            
            double max_res_129005 = fmax64(0.0, max_arg0_129004);
            
            ((double *) mem_143520)[i_142169] = max_res_129005;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143515, i_142173 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143520, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143531_cached_sizze_145667 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143531, &mem_143531_cached_sizze_145667, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143536_cached_sizze_145668 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143536, &mem_143536_cached_sizze_145668, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142181 = 0; i_142181 < (int64_t) 16; i_142181++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142177 = 0; i_142177 < (int64_t) 16; i_142177++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129020;
            double r_129022 = 0.0;
            
            for (int64_t i_129021 = 0; i_129021 < (int64_t) 64; i_129021++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_129023 = ((double *) wdown_mem_143121.mem)[i_142177 * (int64_t) 64 + i_129021];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_129024 = ((double *) mem_143515)[i_142181 * (int64_t) 64 + i_129021];
                
                // futhark/microgpt.fut:170:67-108
                
                double zt_res_129025 = zt_lhs_129023 * zt_rhs_129024;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129026 = r_129022 + zt_res_129025;
                double r_tmp_145270 = zp_res_129026;
                
                r_129022 = r_tmp_145270;
            }
            defunc_0_lifted_lambda_res_129020 = r_129022;
            ((double *) mem_143536)[i_142177] = defunc_0_lifted_lambda_res_129020;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143531, i_142181 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143536, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143547_cached_sizze_145669 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143547, &mem_143547_cached_sizze_145669, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143552_cached_sizze_145670 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143552, &mem_143552_cached_sizze_145670, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142189 = 0; i_142189 < (int64_t) 16; i_142189++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142185 = 0; i_142185 < (int64_t) 16; i_142185++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_129041 = ((double *) mem_143531)[i_142189 * (int64_t) 16 + i_142185];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_129042 = ((double *) mem_143460)[i_142189 * (int64_t) 16 + i_142185];
            
            // futhark/microgpt.fut:171:46-85
            
            double zp_res_129043 = zp_lhs_129041 + zp_rhs_129042;
            
            ((double *) mem_143552)[i_142185] = zp_res_129043;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143547, i_142189 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143552, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143563_cached_sizze_145671 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_143563, &mem_143563_cached_sizze_145671, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143568_cached_sizze_145672 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_143568, &mem_143568_cached_sizze_145672, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142197 = 0; i_142197 < (int64_t) 16; i_142197++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142193 = 0; i_142193 < (int64_t) 27; i_142193++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129059;
            double r_129061 = 0.0;
            
            for (int64_t i_129060 = 0; i_129060 < (int64_t) 16; i_129060++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_129062 = ((double *) wvoc_mem_143129.mem)[i_142193 * (int64_t) 16 + i_129060];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_129063 = ((double *) mem_143547)[i_142197 * (int64_t) 16 + i_129060];
                
                // futhark/microgpt.fut:172:67-107
                
                double zt_res_129064 = zt_lhs_129062 * zt_rhs_129063;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129065 = r_129061 + zt_res_129064;
                double r_tmp_145275 = zp_res_129065;
                
                r_129061 = r_tmp_145275;
            }
            defunc_0_lifted_lambda_res_129059 = r_129061;
            ((double *) mem_143568)[i_142193] = defunc_0_lifted_lambda_res_129059;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143563, i_142197 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143568, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143579, (int64_t) 3456, "mem_143579")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143584_cached_sizze_145673 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_143584, &mem_143584_cached_sizze_145673, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142205 = 0; i_142205 < (int64_t) 16; i_142205++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142201 = 0; i_142201 < (int64_t) 27; i_142201++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_129080 = ((double *) mem_143563)[i_142205 * (int64_t) 27 + i_142201];
            
            ((double *) mem_143584)[i_142201] = lifted_lambda_res_129080;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143579.mem, i_142205 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143584, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_145206, &mem_143579, "mem_143579") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145617, &mem_out_145206, "mem_out_145206") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_143132);
        free(mem_143137);
        free(mem_143148);
        free(mem_143153);
        free(mem_143160);
        free(mem_143171);
        free(mem_143176);
        free(mem_143183);
        free(mem_143194);
        free(mem_143195);
        free(mem_143196);
        free(mem_143209);
        free(mem_143210);
        free(mem_143211);
        free(mem_143242);
        free(mem_143243);
        free(mem_143244);
        free(mem_143260);
        free(mem_143261);
        free(mem_143262);
        free(mem_143275);
        free(mem_143276);
        free(mem_143277);
        free(mem_143323);
        free(mem_143329);
        free(mem_143334);
        free(mem_143345);
        free(mem_143350);
        free(mem_143361);
        free(mem_143366);
        free(mem_143373);
        free(mem_143380);
        free(mem_143391);
        free(mem_143396);
        free(mem_143407);
        free(mem_143412);
        free(mem_143428);
        free(mem_143433);
        free(mem_143444);
        free(mem_143449);
        free(mem_143460);
        free(mem_143465);
        free(mem_143476);
        free(mem_143481);
        free(mem_143488);
        free(mem_143499);
        free(mem_143504);
        free(mem_143515);
        free(mem_143520);
        free(mem_143531);
        free(mem_143536);
        free(mem_143547);
        free(mem_143552);
        free(mem_143563);
        free(mem_143568);
        free(mem_143584);
        if (memblock_unref(ctx, &mem_143579, "mem_143579") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145206, "mem_out_145206") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_grad_loss(struct futhark_context *ctx, struct memblock *mem_out_p_145674, struct memblock *mem_out_p_145675, struct memblock *mem_out_p_145676, struct memblock *mem_out_p_145677, struct memblock *mem_out_p_145678, struct memblock *mem_out_p_145679, struct memblock *mem_out_p_145680, struct memblock *mem_out_p_145681, struct memblock *mem_out_p_145682, struct memblock wdown_mem_143121, struct memblock wkey_mem_143122, struct memblock wout_mem_143123, struct memblock wpe_mem_143124, struct memblock wqry_mem_143125, struct memblock wte_mem_143126, struct memblock wup_mem_143127, struct memblock wval_mem_143128, struct memblock wvoc_mem_143129, struct memblock tokens_mem_143130, struct memblock target_mem_143131, struct memblock mask_mem_143132)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_143133_cached_sizze_145683 = 0;
    unsigned char *mem_143133 = NULL;
    int64_t mem_143138_cached_sizze_145684 = 0;
    unsigned char *mem_143138 = NULL;
    int64_t mem_143149_cached_sizze_145685 = 0;
    unsigned char *mem_143149 = NULL;
    int64_t mem_143150_cached_sizze_145686 = 0;
    unsigned char *mem_143150 = NULL;
    int64_t mem_143151_cached_sizze_145687 = 0;
    unsigned char *mem_143151 = NULL;
    int64_t mem_143170_cached_sizze_145688 = 0;
    unsigned char *mem_143170 = NULL;
    int64_t mem_143177_cached_sizze_145689 = 0;
    unsigned char *mem_143177 = NULL;
    int64_t mem_143182_cached_sizze_145690 = 0;
    unsigned char *mem_143182 = NULL;
    int64_t mem_143193_cached_sizze_145691 = 0;
    unsigned char *mem_143193 = NULL;
    int64_t mem_143198_cached_sizze_145692 = 0;
    unsigned char *mem_143198 = NULL;
    int64_t mem_143209_cached_sizze_145693 = 0;
    unsigned char *mem_143209 = NULL;
    int64_t mem_143210_cached_sizze_145694 = 0;
    unsigned char *mem_143210 = NULL;
    int64_t mem_143223_cached_sizze_145695 = 0;
    unsigned char *mem_143223 = NULL;
    int64_t mem_143230_cached_sizze_145696 = 0;
    unsigned char *mem_143230 = NULL;
    int64_t mem_143235_cached_sizze_145697 = 0;
    unsigned char *mem_143235 = NULL;
    int64_t mem_143246_cached_sizze_145698 = 0;
    unsigned char *mem_143246 = NULL;
    int64_t mem_143251_cached_sizze_145699 = 0;
    unsigned char *mem_143251 = NULL;
    int64_t mem_143262_cached_sizze_145700 = 0;
    unsigned char *mem_143262 = NULL;
    int64_t mem_143263_cached_sizze_145701 = 0;
    unsigned char *mem_143263 = NULL;
    int64_t mem_143264_cached_sizze_145702 = 0;
    unsigned char *mem_143264 = NULL;
    int64_t mem_143280_cached_sizze_145703 = 0;
    unsigned char *mem_143280 = NULL;
    int64_t mem_143281_cached_sizze_145704 = 0;
    unsigned char *mem_143281 = NULL;
    int64_t mem_143282_cached_sizze_145705 = 0;
    unsigned char *mem_143282 = NULL;
    int64_t mem_143295_cached_sizze_145706 = 0;
    unsigned char *mem_143295 = NULL;
    int64_t mem_143296_cached_sizze_145707 = 0;
    unsigned char *mem_143296 = NULL;
    int64_t mem_143297_cached_sizze_145708 = 0;
    unsigned char *mem_143297 = NULL;
    int64_t mem_143343_cached_sizze_145709 = 0;
    unsigned char *mem_143343 = NULL;
    int64_t mem_143344_cached_sizze_145710 = 0;
    unsigned char *mem_143344 = NULL;
    int64_t mem_143345_cached_sizze_145711 = 0;
    unsigned char *mem_143345 = NULL;
    int64_t mem_143346_cached_sizze_145712 = 0;
    unsigned char *mem_143346 = NULL;
    int64_t mem_143367_cached_sizze_145713 = 0;
    unsigned char *mem_143367 = NULL;
    int64_t mem_143368_cached_sizze_145714 = 0;
    unsigned char *mem_143368 = NULL;
    int64_t mem_143369_cached_sizze_145715 = 0;
    unsigned char *mem_143369 = NULL;
    int64_t mem_143370_cached_sizze_145716 = 0;
    unsigned char *mem_143370 = NULL;
    int64_t mem_143387_cached_sizze_145717 = 0;
    unsigned char *mem_143387 = NULL;
    int64_t mem_143388_cached_sizze_145718 = 0;
    unsigned char *mem_143388 = NULL;
    int64_t mem_143389_cached_sizze_145719 = 0;
    unsigned char *mem_143389 = NULL;
    int64_t mem_143390_cached_sizze_145720 = 0;
    unsigned char *mem_143390 = NULL;
    int64_t mem_143451_cached_sizze_145721 = 0;
    unsigned char *mem_143451 = NULL;
    int64_t mem_143452_cached_sizze_145722 = 0;
    unsigned char *mem_143452 = NULL;
    int64_t mem_143453_cached_sizze_145723 = 0;
    unsigned char *mem_143453 = NULL;
    int64_t mem_143454_cached_sizze_145724 = 0;
    unsigned char *mem_143454 = NULL;
    int64_t mem_143475_cached_sizze_145725 = 0;
    unsigned char *mem_143475 = NULL;
    int64_t mem_143476_cached_sizze_145726 = 0;
    unsigned char *mem_143476 = NULL;
    int64_t mem_143477_cached_sizze_145727 = 0;
    unsigned char *mem_143477 = NULL;
    int64_t mem_143478_cached_sizze_145728 = 0;
    unsigned char *mem_143478 = NULL;
    int64_t mem_143495_cached_sizze_145729 = 0;
    unsigned char *mem_143495 = NULL;
    int64_t mem_143496_cached_sizze_145730 = 0;
    unsigned char *mem_143496 = NULL;
    int64_t mem_143497_cached_sizze_145731 = 0;
    unsigned char *mem_143497 = NULL;
    int64_t mem_143498_cached_sizze_145732 = 0;
    unsigned char *mem_143498 = NULL;
    int64_t mem_143559_cached_sizze_145733 = 0;
    unsigned char *mem_143559 = NULL;
    int64_t mem_143560_cached_sizze_145734 = 0;
    unsigned char *mem_143560 = NULL;
    int64_t mem_143561_cached_sizze_145735 = 0;
    unsigned char *mem_143561 = NULL;
    int64_t mem_143562_cached_sizze_145736 = 0;
    unsigned char *mem_143562 = NULL;
    int64_t mem_143563_cached_sizze_145737 = 0;
    unsigned char *mem_143563 = NULL;
    int64_t mem_143564_cached_sizze_145738 = 0;
    unsigned char *mem_143564 = NULL;
    int64_t mem_143565_cached_sizze_145739 = 0;
    unsigned char *mem_143565 = NULL;
    int64_t mem_143566_cached_sizze_145740 = 0;
    unsigned char *mem_143566 = NULL;
    int64_t mem_143599_cached_sizze_145741 = 0;
    unsigned char *mem_143599 = NULL;
    int64_t mem_143600_cached_sizze_145742 = 0;
    unsigned char *mem_143600 = NULL;
    int64_t mem_143601_cached_sizze_145743 = 0;
    unsigned char *mem_143601 = NULL;
    int64_t mem_143602_cached_sizze_145744 = 0;
    unsigned char *mem_143602 = NULL;
    int64_t mem_143603_cached_sizze_145745 = 0;
    unsigned char *mem_143603 = NULL;
    int64_t mem_143604_cached_sizze_145746 = 0;
    unsigned char *mem_143604 = NULL;
    int64_t mem_143605_cached_sizze_145747 = 0;
    unsigned char *mem_143605 = NULL;
    int64_t mem_143606_cached_sizze_145748 = 0;
    unsigned char *mem_143606 = NULL;
    int64_t mem_143687_cached_sizze_145749 = 0;
    unsigned char *mem_143687 = NULL;
    int64_t mem_143688_cached_sizze_145750 = 0;
    unsigned char *mem_143688 = NULL;
    int64_t mem_143689_cached_sizze_145751 = 0;
    unsigned char *mem_143689 = NULL;
    int64_t mem_143690_cached_sizze_145752 = 0;
    unsigned char *mem_143690 = NULL;
    int64_t mem_143711_cached_sizze_145753 = 0;
    unsigned char *mem_143711 = NULL;
    int64_t mem_143712_cached_sizze_145754 = 0;
    unsigned char *mem_143712 = NULL;
    int64_t mem_143713_cached_sizze_145755 = 0;
    unsigned char *mem_143713 = NULL;
    int64_t mem_143714_cached_sizze_145756 = 0;
    unsigned char *mem_143714 = NULL;
    int64_t mem_143731_cached_sizze_145757 = 0;
    unsigned char *mem_143731 = NULL;
    int64_t mem_143732_cached_sizze_145758 = 0;
    unsigned char *mem_143732 = NULL;
    int64_t mem_143733_cached_sizze_145759 = 0;
    unsigned char *mem_143733 = NULL;
    int64_t mem_143734_cached_sizze_145760 = 0;
    unsigned char *mem_143734 = NULL;
    int64_t mem_143795_cached_sizze_145761 = 0;
    unsigned char *mem_143795 = NULL;
    int64_t mem_143796_cached_sizze_145762 = 0;
    unsigned char *mem_143796 = NULL;
    int64_t mem_143805_cached_sizze_145763 = 0;
    unsigned char *mem_143805 = NULL;
    int64_t mem_143806_cached_sizze_145764 = 0;
    unsigned char *mem_143806 = NULL;
    int64_t mem_143827_cached_sizze_145765 = 0;
    unsigned char *mem_143827 = NULL;
    int64_t mem_143828_cached_sizze_145766 = 0;
    unsigned char *mem_143828 = NULL;
    int64_t mem_143839_cached_sizze_145767 = 0;
    unsigned char *mem_143839 = NULL;
    int64_t mem_143840_cached_sizze_145768 = 0;
    unsigned char *mem_143840 = NULL;
    int64_t mem_143849_cached_sizze_145769 = 0;
    unsigned char *mem_143849 = NULL;
    int64_t mem_143850_cached_sizze_145770 = 0;
    unsigned char *mem_143850 = NULL;
    int64_t mem_143881_cached_sizze_145771 = 0;
    unsigned char *mem_143881 = NULL;
    int64_t mem_143882_cached_sizze_145772 = 0;
    unsigned char *mem_143882 = NULL;
    int64_t mem_143893_cached_sizze_145773 = 0;
    unsigned char *mem_143893 = NULL;
    int64_t mem_143894_cached_sizze_145774 = 0;
    unsigned char *mem_143894 = NULL;
    int64_t mem_143903_cached_sizze_145775 = 0;
    unsigned char *mem_143903 = NULL;
    int64_t mem_143904_cached_sizze_145776 = 0;
    unsigned char *mem_143904 = NULL;
    int64_t mem_143935_cached_sizze_145777 = 0;
    unsigned char *mem_143935 = NULL;
    int64_t mem_143941_cached_sizze_145778 = 0;
    unsigned char *mem_143941 = NULL;
    int64_t mem_143946_cached_sizze_145779 = 0;
    unsigned char *mem_143946 = NULL;
    int64_t mem_143962_cached_sizze_145780 = 0;
    unsigned char *mem_143962 = NULL;
    int64_t mem_143967_cached_sizze_145781 = 0;
    unsigned char *mem_143967 = NULL;
    int64_t mem_143978_cached_sizze_145782 = 0;
    unsigned char *mem_143978 = NULL;
    int64_t mem_143983_cached_sizze_145783 = 0;
    unsigned char *mem_143983 = NULL;
    int64_t mem_143994_cached_sizze_145784 = 0;
    unsigned char *mem_143994 = NULL;
    int64_t mem_143995_cached_sizze_145785 = 0;
    unsigned char *mem_143995 = NULL;
    int64_t mem_144008_cached_sizze_145786 = 0;
    unsigned char *mem_144008 = NULL;
    int64_t mem_144015_cached_sizze_145787 = 0;
    unsigned char *mem_144015 = NULL;
    int64_t mem_144020_cached_sizze_145788 = 0;
    unsigned char *mem_144020 = NULL;
    int64_t mem_144031_cached_sizze_145789 = 0;
    unsigned char *mem_144031 = NULL;
    int64_t mem_144036_cached_sizze_145790 = 0;
    unsigned char *mem_144036 = NULL;
    int64_t mem_144047_cached_sizze_145791 = 0;
    unsigned char *mem_144047 = NULL;
    int64_t mem_144052_cached_sizze_145792 = 0;
    unsigned char *mem_144052 = NULL;
    int64_t mem_144063_cached_sizze_145793 = 0;
    unsigned char *mem_144063 = NULL;
    int64_t mem_144068_cached_sizze_145794 = 0;
    unsigned char *mem_144068 = NULL;
    int64_t mem_144079_cached_sizze_145795 = 0;
    unsigned char *mem_144079 = NULL;
    int64_t mem_144084_cached_sizze_145796 = 0;
    unsigned char *mem_144084 = NULL;
    int64_t mem_144095_cached_sizze_145797 = 0;
    unsigned char *mem_144095 = NULL;
    int64_t mem_144100_cached_sizze_145798 = 0;
    unsigned char *mem_144100 = NULL;
    int64_t mem_144111_cached_sizze_145799 = 0;
    unsigned char *mem_144111 = NULL;
    int64_t mem_144112_cached_sizze_145800 = 0;
    unsigned char *mem_144112 = NULL;
    int64_t mem_144113_cached_sizze_145801 = 0;
    unsigned char *mem_144113 = NULL;
    int64_t mem_144114_cached_sizze_145802 = 0;
    unsigned char *mem_144114 = NULL;
    int64_t mem_144133_cached_sizze_145803 = 0;
    unsigned char *mem_144133 = NULL;
    int64_t mem_144140_cached_sizze_145804 = 0;
    unsigned char *mem_144140 = NULL;
    int64_t mem_144147_cached_sizze_145805 = 0;
    unsigned char *mem_144147 = NULL;
    int64_t mem_144152_cached_sizze_145806 = 0;
    unsigned char *mem_144152 = NULL;
    int64_t mem_144182_cached_sizze_145807 = 0;
    unsigned char *mem_144182 = NULL;
    int64_t mem_144188_cached_sizze_145808 = 0;
    unsigned char *mem_144188 = NULL;
    int64_t mem_144193_cached_sizze_145809 = 0;
    unsigned char *mem_144193 = NULL;
    int64_t mem_144209_cached_sizze_145810 = 0;
    unsigned char *mem_144209 = NULL;
    int64_t mem_144210_cached_sizze_145811 = 0;
    unsigned char *mem_144210 = NULL;
    int64_t mem_144219_cached_sizze_145812 = 0;
    unsigned char *mem_144219 = NULL;
    int64_t mem_144220_cached_sizze_145813 = 0;
    unsigned char *mem_144220 = NULL;
    int64_t mem_144241_cached_sizze_145814 = 0;
    unsigned char *mem_144241 = NULL;
    int64_t mem_144247_cached_sizze_145815 = 0;
    unsigned char *mem_144247 = NULL;
    int64_t mem_144252_cached_sizze_145816 = 0;
    unsigned char *mem_144252 = NULL;
    int64_t mem_144268_cached_sizze_145817 = 0;
    unsigned char *mem_144268 = NULL;
    int64_t mem_144273_cached_sizze_145818 = 0;
    unsigned char *mem_144273 = NULL;
    int64_t mem_144284_cached_sizze_145819 = 0;
    unsigned char *mem_144284 = NULL;
    int64_t mem_144289_cached_sizze_145820 = 0;
    unsigned char *mem_144289 = NULL;
    int64_t mem_144300_cached_sizze_145821 = 0;
    unsigned char *mem_144300 = NULL;
    int64_t mem_144305_cached_sizze_145822 = 0;
    unsigned char *mem_144305 = NULL;
    int64_t mem_144317_cached_sizze_145823 = 0;
    unsigned char *mem_144317 = NULL;
    int64_t mem_144326_cached_sizze_145824 = 0;
    unsigned char *mem_144326 = NULL;
    int64_t mem_144327_cached_sizze_145825 = 0;
    unsigned char *mem_144327 = NULL;
    int64_t mem_144348_cached_sizze_145826 = 0;
    unsigned char *mem_144348 = NULL;
    int64_t mem_144353_cached_sizze_145827 = 0;
    unsigned char *mem_144353 = NULL;
    int64_t mem_144364_cached_sizze_145828 = 0;
    unsigned char *mem_144364 = NULL;
    int64_t mem_144365_cached_sizze_145829 = 0;
    unsigned char *mem_144365 = NULL;
    int64_t mem_144378_cached_sizze_145830 = 0;
    unsigned char *mem_144378 = NULL;
    int64_t mem_144385_cached_sizze_145831 = 0;
    unsigned char *mem_144385 = NULL;
    int64_t mem_144390_cached_sizze_145832 = 0;
    unsigned char *mem_144390 = NULL;
    int64_t mem_144401_cached_sizze_145833 = 0;
    unsigned char *mem_144401 = NULL;
    int64_t mem_144407_cached_sizze_145834 = 0;
    unsigned char *mem_144407 = NULL;
    int64_t mem_144412_cached_sizze_145835 = 0;
    unsigned char *mem_144412 = NULL;
    int64_t mem_144428_cached_sizze_145836 = 0;
    unsigned char *mem_144428 = NULL;
    int64_t mem_144429_cached_sizze_145837 = 0;
    unsigned char *mem_144429 = NULL;
    int64_t mem_144430_cached_sizze_145838 = 0;
    unsigned char *mem_144430 = NULL;
    int64_t mem_144446_cached_sizze_145839 = 0;
    unsigned char *mem_144446 = NULL;
    int64_t mem_144447_cached_sizze_145840 = 0;
    unsigned char *mem_144447 = NULL;
    int64_t mem_144448_cached_sizze_145841 = 0;
    unsigned char *mem_144448 = NULL;
    int64_t mem_144461_cached_sizze_145842 = 0;
    unsigned char *mem_144461 = NULL;
    int64_t mem_144462_cached_sizze_145843 = 0;
    unsigned char *mem_144462 = NULL;
    int64_t mem_144503_cached_sizze_145844 = 0;
    unsigned char *mem_144503 = NULL;
    int64_t mem_144504_cached_sizze_145845 = 0;
    unsigned char *mem_144504 = NULL;
    int64_t mem_144515_cached_sizze_145846 = 0;
    unsigned char *mem_144515 = NULL;
    int64_t mem_144516_cached_sizze_145847 = 0;
    unsigned char *mem_144516 = NULL;
    int64_t mem_144525_cached_sizze_145848 = 0;
    unsigned char *mem_144525 = NULL;
    int64_t mem_144526_cached_sizze_145849 = 0;
    unsigned char *mem_144526 = NULL;
    int64_t mem_144557_cached_sizze_145850 = 0;
    unsigned char *mem_144557 = NULL;
    int64_t mem_144558_cached_sizze_145851 = 0;
    unsigned char *mem_144558 = NULL;
    int64_t mem_144569_cached_sizze_145852 = 0;
    unsigned char *mem_144569 = NULL;
    int64_t mem_144570_cached_sizze_145853 = 0;
    unsigned char *mem_144570 = NULL;
    int64_t mem_144579_cached_sizze_145854 = 0;
    unsigned char *mem_144579 = NULL;
    int64_t mem_144580_cached_sizze_145855 = 0;
    unsigned char *mem_144580 = NULL;
    int64_t mem_144611_cached_sizze_145856 = 0;
    unsigned char *mem_144611 = NULL;
    int64_t mem_144612_cached_sizze_145857 = 0;
    unsigned char *mem_144612 = NULL;
    int64_t mem_144613_cached_sizze_145858 = 0;
    unsigned char *mem_144613 = NULL;
    int64_t mem_144614_cached_sizze_145859 = 0;
    unsigned char *mem_144614 = NULL;
    int64_t mem_144631_cached_sizze_145860 = 0;
    unsigned char *mem_144631 = NULL;
    int64_t mem_144632_cached_sizze_145861 = 0;
    unsigned char *mem_144632 = NULL;
    int64_t mem_144633_cached_sizze_145862 = 0;
    unsigned char *mem_144633 = NULL;
    int64_t mem_144634_cached_sizze_145863 = 0;
    unsigned char *mem_144634 = NULL;
    int64_t mem_144675_cached_sizze_145864 = 0;
    unsigned char *mem_144675 = NULL;
    int64_t mem_144676_cached_sizze_145865 = 0;
    unsigned char *mem_144676 = NULL;
    int64_t mem_144687_cached_sizze_145866 = 0;
    unsigned char *mem_144687 = NULL;
    int64_t mem_144688_cached_sizze_145867 = 0;
    unsigned char *mem_144688 = NULL;
    int64_t mem_144697_cached_sizze_145868 = 0;
    unsigned char *mem_144697 = NULL;
    int64_t mem_144698_cached_sizze_145869 = 0;
    unsigned char *mem_144698 = NULL;
    int64_t mem_144729_cached_sizze_145870 = 0;
    unsigned char *mem_144729 = NULL;
    int64_t mem_144730_cached_sizze_145871 = 0;
    unsigned char *mem_144730 = NULL;
    int64_t mem_144739_cached_sizze_145872 = 0;
    unsigned char *mem_144739 = NULL;
    int64_t mem_144740_cached_sizze_145873 = 0;
    unsigned char *mem_144740 = NULL;
    int64_t mem_144761_cached_sizze_145874 = 0;
    unsigned char *mem_144761 = NULL;
    int64_t mem_144762_cached_sizze_145875 = 0;
    unsigned char *mem_144762 = NULL;
    int64_t mem_144773_cached_sizze_145876 = 0;
    unsigned char *mem_144773 = NULL;
    int64_t mem_144774_cached_sizze_145877 = 0;
    unsigned char *mem_144774 = NULL;
    int64_t mem_144783_cached_sizze_145878 = 0;
    unsigned char *mem_144783 = NULL;
    int64_t mem_144784_cached_sizze_145879 = 0;
    unsigned char *mem_144784 = NULL;
    int64_t mem_144815_cached_sizze_145880 = 0;
    unsigned char *mem_144815 = NULL;
    int64_t mem_144816_cached_sizze_145881 = 0;
    unsigned char *mem_144816 = NULL;
    int64_t mem_144827_cached_sizze_145882 = 0;
    unsigned char *mem_144827 = NULL;
    int64_t mem_144828_cached_sizze_145883 = 0;
    unsigned char *mem_144828 = NULL;
    int64_t mem_144837_cached_sizze_145884 = 0;
    unsigned char *mem_144837 = NULL;
    int64_t mem_144838_cached_sizze_145885 = 0;
    unsigned char *mem_144838 = NULL;
    int64_t mem_144870_cached_sizze_145886 = 0;
    unsigned char *mem_144870 = NULL;
    int64_t mem_144871_cached_sizze_145887 = 0;
    unsigned char *mem_144871 = NULL;
    int64_t mem_144872_cached_sizze_145888 = 0;
    unsigned char *mem_144872 = NULL;
    int64_t mem_144889_cached_sizze_145889 = 0;
    unsigned char *mem_144889 = NULL;
    int64_t mem_144890_cached_sizze_145890 = 0;
    unsigned char *mem_144890 = NULL;
    int64_t mem_144891_cached_sizze_145891 = 0;
    unsigned char *mem_144891 = NULL;
    int64_t mem_144892_cached_sizze_145892 = 0;
    unsigned char *mem_144892 = NULL;
    int64_t mem_144933_cached_sizze_145893 = 0;
    unsigned char *mem_144933 = NULL;
    int64_t mem_144938_cached_sizze_145894 = 0;
    unsigned char *mem_144938 = NULL;
    int64_t mem_144952_cached_sizze_145895 = 0;
    unsigned char *mem_144952 = NULL;
    int64_t mem_144953_cached_sizze_145896 = 0;
    unsigned char *mem_144953 = NULL;
    int64_t mem_144972_cached_sizze_145897 = 0;
    unsigned char *mem_144972 = NULL;
    int64_t mem_144973_cached_sizze_145898 = 0;
    unsigned char *mem_144973 = NULL;
    int64_t mem_144974_cached_sizze_145899 = 0;
    unsigned char *mem_144974 = NULL;
    int64_t mem_145011_cached_sizze_145900 = 0;
    unsigned char *mem_145011 = NULL;
    int64_t mem_145018_cached_sizze_145901 = 0;
    unsigned char *mem_145018 = NULL;
    int64_t mem_145023_cached_sizze_145902 = 0;
    unsigned char *mem_145023 = NULL;
    int64_t mem_145034_cached_sizze_145903 = 0;
    unsigned char *mem_145034 = NULL;
    int64_t mem_145035_cached_sizze_145904 = 0;
    unsigned char *mem_145035 = NULL;
    int64_t mem_145044_cached_sizze_145905 = 0;
    unsigned char *mem_145044 = NULL;
    int64_t mem_145045_cached_sizze_145906 = 0;
    unsigned char *mem_145045 = NULL;
    int64_t mem_145066_cached_sizze_145907 = 0;
    unsigned char *mem_145066 = NULL;
    int64_t mem_145067_cached_sizze_145908 = 0;
    unsigned char *mem_145067 = NULL;
    int64_t mem_145068_cached_sizze_145909 = 0;
    unsigned char *mem_145068 = NULL;
    int64_t mem_145069_cached_sizze_145910 = 0;
    unsigned char *mem_145069 = NULL;
    int64_t mem_145094_cached_sizze_145911 = 0;
    unsigned char *mem_145094 = NULL;
    int64_t mem_145095_cached_sizze_145912 = 0;
    unsigned char *mem_145095 = NULL;
    int64_t mem_145108_cached_sizze_145913 = 0;
    unsigned char *mem_145108 = NULL;
    int64_t mem_145118_cached_sizze_145914 = 0;
    unsigned char *mem_145118 = NULL;
    int64_t mem_145119_cached_sizze_145915 = 0;
    unsigned char *mem_145119 = NULL;
    int64_t mem_145145_cached_sizze_145916 = 0;
    unsigned char *mem_145145 = NULL;
    int64_t mem_145166_cached_sizze_145917 = 0;
    unsigned char *mem_145166 = NULL;
    int64_t mem_145167_cached_sizze_145918 = 0;
    unsigned char *mem_145167 = NULL;
    struct memblock mem_145157;
    
    mem_145157.references = NULL;
    
    struct memblock mem_145156;
    
    mem_145156.references = NULL;
    
    struct memblock mem_145140;
    
    mem_145140.references = NULL;
    
    struct memblock mem_145109;
    
    mem_145109.references = NULL;
    
    struct memblock mem_144951;
    
    mem_144951.references = NULL;
    
    struct memblock mem_144950;
    
    mem_144950.references = NULL;
    
    struct memblock mem_144949;
    
    mem_144949.references = NULL;
    
    struct memblock mem_144869;
    
    mem_144869.references = NULL;
    
    struct memblock mem_144316;
    
    mem_144316.references = NULL;
    
    struct memblock mem_out_145214;
    
    mem_out_145214.references = NULL;
    
    struct memblock mem_out_145213;
    
    mem_out_145213.references = NULL;
    
    struct memblock mem_out_145212;
    
    mem_out_145212.references = NULL;
    
    struct memblock mem_out_145211;
    
    mem_out_145211.references = NULL;
    
    struct memblock mem_out_145210;
    
    mem_out_145210.references = NULL;
    
    struct memblock mem_out_145209;
    
    mem_out_145209.references = NULL;
    
    struct memblock mem_out_145208;
    
    mem_out_145208.references = NULL;
    
    struct memblock mem_out_145207;
    
    mem_out_145207.references = NULL;
    
    struct memblock mem_out_145206;
    
    mem_out_145206.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (mem_143133_cached_sizze_145683 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143133, &mem_143133_cached_sizze_145683, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143138_cached_sizze_145684 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143138, &mem_143138_cached_sizze_145684, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_141993 = 0; i_141993 < (int64_t) 16; i_141993++) {
        // futhark/microgpt.fut:457:41-50
        
        int64_t tmp_128515 = ((int64_t *) tokens_mem_143130.mem)[i_141993];
        
        // futhark/microgpt.fut:457:37-51
        
        bool x_128516 = sle64((int64_t) 0, tmp_128515);
        
        // futhark/microgpt.fut:457:37-51
        
        bool y_128517 = slt64(tmp_128515, (int64_t) 27);
        
        // futhark/microgpt.fut:457:37-51
        
        bool bounds_check_128518 = x_128516 && y_128517;
        
        // futhark/microgpt.fut:457:37-51
        
        bool index_certs_128519;
        
        if (!bounds_check_128518) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_128515, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:457:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:457:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_141989 = 0; i_141989 < (int64_t) 16; i_141989++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_128526 = ((double *) wte_mem_143126.mem)[tmp_128515 * (int64_t) 16 + i_141989];
            
            ((double *) mem_143138)[i_141989] = lifted_lambda_res_128526;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143133, i_141993 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143138, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143149_cached_sizze_145685 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143149, &mem_143149_cached_sizze_145685, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143150_cached_sizze_145686 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143150, &mem_143150_cached_sizze_145686, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143151_cached_sizze_145687 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143151, &mem_143151_cached_sizze_145687, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142001 = 0; i_142001 < (int64_t) 16; i_142001++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_131941;
        double r_131943 = 0.0;
        
        for (int64_t i_131942 = 0; i_131942 < (int64_t) 16; i_131942++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_131944 = ((double *) wpe_mem_143124.mem)[i_142001 * (int64_t) 16 + i_131942];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_131945 = ((double *) mem_143133)[i_142001 * (int64_t) 16 + i_131942];
            
            // futhark/microgpt.fut:269:63-99
            
            double zp_res_131946 = zp_lhs_131944 + zp_rhs_131945;
            
            // futhark/microgpt.fut:269:79-142
            
            double zt_res_131947 = zp_res_131946 * zp_res_131946;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_131948 = r_131943 + zt_res_131947;
            double r_tmp_145220 = zp_res_131948;
            
            r_131943 = r_tmp_145220;
        }
        defunc_0_lifted_lambda_res_131941 = r_131943;
        // futhark/microgpt.fut:269:42-161
        
        double zs_res_131949 = defunc_0_lifted_lambda_res_131941 / 16.0;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_131956;
        double r_131958 = 0.0;
        
        for (int64_t i_131957 = 0; i_131957 < (int64_t) 16; i_131957++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_131959 = ((double *) wpe_mem_143124.mem)[i_142001 * (int64_t) 16 + i_131957];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_131960 = ((double *) mem_143133)[i_142001 * (int64_t) 16 + i_131957];
            
            // futhark/microgpt.fut:385:71-115
            
            double zp_res_131961 = zp_lhs_131959 + zp_rhs_131960;
            
            // futhark/microgpt.fut:385:91-166
            
            double zt_res_131962 = zp_res_131961 * zp_res_131961;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_131963 = r_131958 + zt_res_131962;
            double r_tmp_145221 = zp_res_131963;
            
            r_131958 = r_tmp_145221;
        }
        defunc_0_lifted_lambda_res_131956 = r_131958;
        // futhark/microgpt.fut:385:48-185
        
        double zs_res_131964 = defunc_0_lifted_lambda_res_131956 / 16.0;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_131974;
        double r_131976 = 0.0;
        
        for (int64_t i_131975 = 0; i_131975 < (int64_t) 16; i_131975++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_131977 = ((double *) wpe_mem_143124.mem)[i_142001 * (int64_t) 16 + i_131975];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_131978 = ((double *) mem_143133)[i_142001 * (int64_t) 16 + i_131975];
            
            // futhark/microgpt.fut:398:72-116
            
            double zp_res_131979 = zp_lhs_131977 + zp_rhs_131978;
            
            // futhark/microgpt.fut:398:92-167
            
            double zt_res_131980 = zp_res_131979 * zp_res_131979;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_131981 = r_131976 + zt_res_131980;
            double r_tmp_145222 = zp_res_131981;
            
            r_131976 = r_tmp_145222;
        }
        defunc_0_lifted_lambda_res_131974 = r_131976;
        // futhark/microgpt.fut:398:49-186
        
        double zs_res_131982 = defunc_0_lifted_lambda_res_131974 / 16.0;
        
        ((double *) mem_143149)[i_142001] = zs_res_131982;
        ((double *) mem_143150)[i_142001] = zs_res_131964;
        ((double *) mem_143151)[i_142001] = zs_res_131949;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143170_cached_sizze_145688 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143170, &mem_143170_cached_sizze_145688, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142007 = 0; i_142007 < (int64_t) 16; i_142007++) {
        // futhark/microgpt.fut:270:43-51
        
        double zp_lhs_128568 = ((double *) mem_143151)[i_142007];
        
        // futhark/microgpt.fut:270:43-79
        
        double zp_res_128569 = 1.0e-5 + zp_lhs_128568;
        
        // futhark/microgpt.fut:270:35-79
        
        double sqrt_res_128570 = futrts_sqrt64(zp_res_128569);
        
        ((double *) mem_143170)[i_142007] = sqrt_res_128570;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143177_cached_sizze_145689 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143177, &mem_143177_cached_sizze_145689, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143182_cached_sizze_145690 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143182, &mem_143182_cached_sizze_145690, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142015 = 0; i_142015 < (int64_t) 16; i_142015++) {
        // futhark/microgpt.fut:271:95-103
        
        double zs_rhs_128578 = ((double *) mem_143170)[i_142015];
        
        // futhark/microgpt.fut:271:87-103
        
        double zs_res_128579 = 1.0 / zs_rhs_128578;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142011 = 0; i_142011 < (int64_t) 16; i_142011++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_128586 = ((double *) wpe_mem_143124.mem)[i_142015 * (int64_t) 16 + i_142011];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_128587 = ((double *) mem_143133)[i_142015 * (int64_t) 16 + i_142011];
            
            // futhark/microgpt.fut:271:44-80
            
            double zp_res_128588 = zp_lhs_128586 + zp_rhs_128587;
            
            // futhark/microgpt.fut:271:60-103
            
            double zt_res_128589 = zs_res_128579 * zp_res_128588;
            
            ((double *) mem_143182)[i_142011] = zt_res_128589;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143177, i_142015 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143182, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143193_cached_sizze_145691 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143193, &mem_143193_cached_sizze_145691, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143198_cached_sizze_145692 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143198, &mem_143198_cached_sizze_145692, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142023 = 0; i_142023 < (int64_t) 16; i_142023++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142019 = 0; i_142019 < (int64_t) 16; i_142019++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_128604 = ((double *) mem_143177)[i_142023 * (int64_t) 16 + i_142019];
            
            ((double *) mem_143198)[i_142019] = lifted_lambda_res_128604;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143193, i_142023 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143198, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143209_cached_sizze_145693 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143209, &mem_143209_cached_sizze_145693, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143210_cached_sizze_145694 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143210, &mem_143210_cached_sizze_145694, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142029 = 0; i_142029 < (int64_t) 16; i_142029++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_132001;
        double r_132003 = 0.0;
        
        for (int64_t i_132002 = 0; i_132002 < (int64_t) 16; i_132002++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_132004 = ((double *) mem_143193)[i_142029 * (int64_t) 16 + i_132002];
            
            // futhark/microgpt.fut:273:65-102
            
            double zt_res_132005 = zt_lhs_132004 * zt_lhs_132004;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_132006 = r_132003 + zt_res_132005;
            double r_tmp_145230 = zp_res_132006;
            
            r_132003 = r_tmp_145230;
        }
        defunc_0_lifted_lambda_res_132001 = r_132003;
        // futhark/microgpt.fut:273:44-120
        
        double zs_res_132007 = defunc_0_lifted_lambda_res_132001 / 16.0;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_132014;
        double r_132016 = 0.0;
        
        for (int64_t i_132015 = 0; i_132015 < (int64_t) 16; i_132015++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_132017 = ((double *) mem_143193)[i_142029 * (int64_t) 16 + i_132015];
            
            // futhark/microgpt.fut:363:70-111
            
            double zt_res_132018 = zt_lhs_132017 * zt_lhs_132017;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_132019 = r_132016 + zt_res_132018;
            double r_tmp_145231 = zp_res_132019;
            
            r_132016 = r_tmp_145231;
        }
        defunc_0_lifted_lambda_res_132014 = r_132016;
        // futhark/microgpt.fut:363:48-129
        
        double zs_res_132020 = defunc_0_lifted_lambda_res_132014 / 16.0;
        
        ((double *) mem_143209)[i_142029] = zs_res_132020;
        ((double *) mem_143210)[i_142029] = zs_res_132007;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143223_cached_sizze_145695 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143223, &mem_143223_cached_sizze_145695, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142034 = 0; i_142034 < (int64_t) 16; i_142034++) {
        // futhark/microgpt.fut:274:45-55
        
        double zp_lhs_128627 = ((double *) mem_143210)[i_142034];
        
        // futhark/microgpt.fut:274:45-83
        
        double zp_res_128628 = 1.0e-5 + zp_lhs_128627;
        
        // futhark/microgpt.fut:274:37-83
        
        double sqrt_res_128629 = futrts_sqrt64(zp_res_128628);
        
        ((double *) mem_143223)[i_142034] = sqrt_res_128629;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143230_cached_sizze_145696 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143230, &mem_143230_cached_sizze_145696, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143235_cached_sizze_145697 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143235, &mem_143235_cached_sizze_145697, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142042 = 0; i_142042 < (int64_t) 16; i_142042++) {
        // futhark/microgpt.fut:275:76-86
        
        double zs_rhs_128637 = ((double *) mem_143223)[i_142042];
        
        // futhark/microgpt.fut:275:68-86
        
        double zs_res_128638 = 1.0 / zs_rhs_128637;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142038 = 0; i_142038 < (int64_t) 16; i_142038++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_128645 = ((double *) mem_143193)[i_142042 * (int64_t) 16 + i_142038];
            
            // futhark/microgpt.fut:275:46-86
            
            double zt_res_128646 = zs_res_128638 * zt_lhs_128645;
            
            ((double *) mem_143235)[i_142038] = zt_res_128646;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143230, i_142042 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143235, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143246_cached_sizze_145698 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143246, &mem_143246_cached_sizze_145698, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143251_cached_sizze_145699 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143251, &mem_143251_cached_sizze_145699, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142050 = 0; i_142050 < (int64_t) 16; i_142050++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142046 = 0; i_142046 < (int64_t) 16; i_142046++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_128661 = ((double *) mem_143230)[i_142050 * (int64_t) 16 + i_142046];
            
            ((double *) mem_143251)[i_142046] = lifted_lambda_res_128661;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143246, i_142050 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143251, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143262_cached_sizze_145700 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143262, &mem_143262_cached_sizze_145700, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143263_cached_sizze_145701 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143263, &mem_143263_cached_sizze_145701, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143264_cached_sizze_145702 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143264, &mem_143264_cached_sizze_145702, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143280_cached_sizze_145703 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143280, &mem_143280_cached_sizze_145703, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143281_cached_sizze_145704 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143281, &mem_143281_cached_sizze_145704, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143282_cached_sizze_145705 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143282, &mem_143282_cached_sizze_145705, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143295_cached_sizze_145706 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_143295, &mem_143295_cached_sizze_145706, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143296_cached_sizze_145707 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_143296, &mem_143296_cached_sizze_145707, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143297_cached_sizze_145708 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_143297, &mem_143297_cached_sizze_145708, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142078 = 0; i_142078 < (int64_t) 4; i_142078++) {
        // futhark/microgpt.fut:277:83-86
        
        int64_t zp_lhs_132101 = mul64((int64_t) 4, i_142078);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142068 = 0; i_142068 < (int64_t) 16; i_142068++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142058 = 0; i_142058 < (int64_t) 4; i_142058++) {
                // futhark/microgpt.fut:277:88-95
                
                int64_t zt_lhs_136230 = add64(zp_lhs_132101, i_142058);
                
                // futhark/microgpt.fut:277:70-97
                
                bool x_136231 = sle64((int64_t) 0, zt_lhs_136230);
                
                // futhark/microgpt.fut:277:70-97
                
                bool y_136232 = slt64(zt_lhs_136230, (int64_t) 16);
                
                // futhark/microgpt.fut:277:70-97
                
                bool bounds_check_136233 = x_136231 && y_136232;
                
                // futhark/microgpt.fut:277:70-97
                
                bool index_certs_136234;
                
                if (!bounds_check_136233) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_136230, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:277:70-97\n   #1  futhark/microgpt.fut:61:46-49\n   #2  futhark/microgpt.fut:277:49-127\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:15:29-44\n   #8  futhark/microgpt.fut:4:11-25\n   #9  futhark/microgpt.fut:15:15-45\n   #10 futhark/microgpt.fut:277:12-129\n   #11 futhark/microgpt.fut:459:5-75\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_136235;
                double r_136237 = 0.0;
                
                for (int64_t i_136236 = 0; i_136236 < (int64_t) 16; i_136236++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_136238 = ((double *) wqry_mem_143125.mem)[zt_lhs_136230 * (int64_t) 16 + i_136236];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_136239 = ((double *) mem_143246)[i_142068 * (int64_t) 16 + i_136236];
                    
                    // futhark/microgpt.fut:277:70-125
                    
                    double zt_res_136240 = zt_lhs_136238 * zt_rhs_136239;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_136241 = r_136237 + zt_res_136240;
                    double r_tmp_145246 = zp_res_136241;
                    
                    r_136237 = r_tmp_145246;
                }
                defunc_0_lifted_lambda_res_136235 = r_136237;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_136249;
                double r_136251 = 0.0;
                
                for (int64_t i_136250 = 0; i_136250 < (int64_t) 16; i_136250++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_136252 = ((double *) wkey_mem_143122.mem)[zt_lhs_136230 * (int64_t) 16 + i_136250];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_136253 = ((double *) mem_143246)[i_142068 * (int64_t) 16 + i_136250];
                    
                    // futhark/microgpt.fut:278:70-125
                    
                    double zt_res_136254 = zt_lhs_136252 * zt_rhs_136253;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_136255 = r_136251 + zt_res_136254;
                    double r_tmp_145247 = zp_res_136255;
                    
                    r_136251 = r_tmp_145247;
                }
                defunc_0_lifted_lambda_res_136249 = r_136251;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_136266;
                double r_136268 = 0.0;
                
                for (int64_t i_136267 = 0; i_136267 < (int64_t) 16; i_136267++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_136269 = ((double *) wval_mem_143128.mem)[zt_lhs_136230 * (int64_t) 16 + i_136267];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_136270 = ((double *) mem_143246)[i_142068 * (int64_t) 16 + i_136267];
                    
                    // futhark/microgpt.fut:279:70-125
                    
                    double zt_res_136271 = zt_lhs_136269 * zt_rhs_136270;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_136272 = r_136268 + zt_res_136271;
                    double r_tmp_145248 = zp_res_136272;
                    
                    r_136268 = r_tmp_145248;
                }
                defunc_0_lifted_lambda_res_136266 = r_136268;
                ((double *) mem_143295)[i_142058] = defunc_0_lifted_lambda_res_136266;
                ((double *) mem_143296)[i_142058] = defunc_0_lifted_lambda_res_136249;
                ((double *) mem_143297)[i_142058] = defunc_0_lifted_lambda_res_136235;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143280, i_142068 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143295, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143281, i_142068 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143296, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143282, i_142068 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143297, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143262, i_142078 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_143280, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143263, i_142078 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_143281, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143264, i_142078 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_143282, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143343_cached_sizze_145709 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143343, &mem_143343_cached_sizze_145709, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143344_cached_sizze_145710 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143344, &mem_143344_cached_sizze_145710, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143345_cached_sizze_145711 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143345, &mem_143345_cached_sizze_145711, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143346_cached_sizze_145712 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143346, &mem_143346_cached_sizze_145712, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143367_cached_sizze_145713 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143367, &mem_143367_cached_sizze_145713, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143368_cached_sizze_145714 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143368, &mem_143368_cached_sizze_145714, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143369_cached_sizze_145715 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143369, &mem_143369_cached_sizze_145715, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143370_cached_sizze_145716 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143370, &mem_143370_cached_sizze_145716, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143387_cached_sizze_145717 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143387, &mem_143387_cached_sizze_145717, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143388_cached_sizze_145718 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143388, &mem_143388_cached_sizze_145718, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143389_cached_sizze_145719 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143389, &mem_143389_cached_sizze_145719, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143390_cached_sizze_145720 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143390, &mem_143390_cached_sizze_145720, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142116 = 0; i_142116 < (int64_t) 4; i_142116++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142103 = 0; i_142103 < (int64_t) 16; i_142103++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142090 = 0; i_142090 < (int64_t) 16; i_142090++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_136654;
                double r_136656 = 0.0;
                
                for (int64_t i_136655 = 0; i_136655 < (int64_t) 4; i_136655++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_136657 = ((double *) mem_143264)[i_142116 * (int64_t) 64 + i_142103 * (int64_t) 4 + i_136655];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_136658 = ((double *) mem_143263)[i_142116 * (int64_t) 64 + i_142090 * (int64_t) 4 + i_136655];
                    
                    // futhark/microgpt.fut:280:111-164
                    
                    double zt_res_136659 = zt_lhs_136657 * zt_rhs_136658;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_136660 = r_136656 + zt_res_136659;
                    double r_tmp_145261 = zp_res_136660;
                    
                    r_136656 = r_tmp_145261;
                }
                defunc_0_lifted_lambda_res_136654 = r_136656;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_136667;
                double r_136669 = 0.0;
                
                for (int64_t i_136668 = 0; i_136668 < (int64_t) 4; i_136668++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_136670 = ((double *) mem_143264)[i_142116 * (int64_t) 64 + i_142103 * (int64_t) 4 + i_136668];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_136671 = ((double *) mem_143263)[i_142116 * (int64_t) 64 + i_142090 * (int64_t) 4 + i_136668];
                    
                    // futhark/microgpt.fut:322:119-178
                    
                    double zt_res_136672 = zt_lhs_136670 * zt_rhs_136671;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_136673 = r_136669 + zt_res_136672;
                    double r_tmp_145262 = zp_res_136673;
                    
                    r_136669 = r_tmp_145262;
                }
                defunc_0_lifted_lambda_res_136667 = r_136669;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_136683;
                double r_136685 = 0.0;
                
                for (int64_t i_136684 = 0; i_136684 < (int64_t) 4; i_136684++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_136686 = ((double *) mem_143264)[i_142116 * (int64_t) 64 + i_142103 * (int64_t) 4 + i_136684];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_136687 = ((double *) mem_143263)[i_142116 * (int64_t) 64 + i_142090 * (int64_t) 4 + i_136684];
                    
                    // futhark/microgpt.fut:331:119-178
                    
                    double zt_res_136688 = zt_lhs_136686 * zt_rhs_136687;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_136689 = r_136685 + zt_res_136688;
                    double r_tmp_145263 = zp_res_136689;
                    
                    r_136685 = r_tmp_145263;
                }
                defunc_0_lifted_lambda_res_136683 = r_136685;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_136701;
                double r_136703 = 0.0;
                
                for (int64_t i_136702 = 0; i_136702 < (int64_t) 4; i_136702++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_136704 = ((double *) mem_143264)[i_142116 * (int64_t) 64 + i_142103 * (int64_t) 4 + i_136702];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_136705 = ((double *) mem_143263)[i_142116 * (int64_t) 64 + i_142090 * (int64_t) 4 + i_136702];
                    
                    // futhark/microgpt.fut:347:119-178
                    
                    double zt_res_136706 = zt_lhs_136704 * zt_rhs_136705;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_136707 = r_136703 + zt_res_136706;
                    double r_tmp_145264 = zp_res_136707;
                    
                    r_136703 = r_tmp_145264;
                }
                defunc_0_lifted_lambda_res_136701 = r_136703;
                ((double *) mem_143387)[i_142090] = defunc_0_lifted_lambda_res_136701;
                ((double *) mem_143388)[i_142090] = defunc_0_lifted_lambda_res_136683;
                ((double *) mem_143389)[i_142090] = defunc_0_lifted_lambda_res_136667;
                ((double *) mem_143390)[i_142090] = defunc_0_lifted_lambda_res_136654;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143367, i_142103 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143387, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143368, i_142103 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143388, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143369, i_142103 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143389, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143370, i_142103 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143390, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143343, i_142116 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_143367, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143344, i_142116 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_143368, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143345, i_142116 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_143369, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143346, i_142116 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_143370, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143451_cached_sizze_145721 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143451, &mem_143451_cached_sizze_145721, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143452_cached_sizze_145722 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143452, &mem_143452_cached_sizze_145722, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143453_cached_sizze_145723 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143453, &mem_143453_cached_sizze_145723, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143454_cached_sizze_145724 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143454, &mem_143454_cached_sizze_145724, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143475_cached_sizze_145725 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143475, &mem_143475_cached_sizze_145725, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143476_cached_sizze_145726 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143476, &mem_143476_cached_sizze_145726, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143477_cached_sizze_145727 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143477, &mem_143477_cached_sizze_145727, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143478_cached_sizze_145728 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143478, &mem_143478_cached_sizze_145728, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143495_cached_sizze_145729 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143495, &mem_143495_cached_sizze_145729, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143496_cached_sizze_145730 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143496, &mem_143496_cached_sizze_145730, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143497_cached_sizze_145731 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143497, &mem_143497_cached_sizze_145731, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143498_cached_sizze_145732 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143498, &mem_143498_cached_sizze_145732, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142155 = 0; i_142155 < (int64_t) 4; i_142155++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142142 = 0; i_142142 < (int64_t) 16; i_142142++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142129 = 0; i_142129 < (int64_t) 16; i_142129++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_137051 = ((double *) mem_143346)[i_142155 * (int64_t) 256 + i_142142 * (int64_t) 16 + i_142129];
                
                // futhark/microgpt.fut:281:55-93
                
                double zs_res_137052 = zs_lhs_137051 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_137053 = ((double *) mask_mem_143132.mem)[i_142142 * (int64_t) 16 + i_142129];
                
                // futhark/microgpt.fut:281:80-117
                
                double zp_res_137054 = zs_res_137052 + zp_rhs_137053;
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_137061 = ((double *) mem_143345)[i_142155 * (int64_t) 256 + i_142142 * (int64_t) 16 + i_142129];
                
                // futhark/microgpt.fut:323:59-101
                
                double zs_res_137062 = zs_lhs_137061 / 2.0;
                
                // futhark/microgpt.fut:323:88-127
                
                double zp_res_137064 = zp_rhs_137053 + zs_res_137062;
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_137074 = ((double *) mem_143344)[i_142155 * (int64_t) 256 + i_142142 * (int64_t) 16 + i_142129];
                
                // futhark/microgpt.fut:332:59-101
                
                double zs_res_137075 = zs_lhs_137074 / 2.0;
                
                // futhark/microgpt.fut:332:88-127
                
                double zp_res_137077 = zp_rhs_137053 + zs_res_137075;
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_137089 = ((double *) mem_143343)[i_142155 * (int64_t) 256 + i_142142 * (int64_t) 16 + i_142129];
                
                // futhark/microgpt.fut:348:59-101
                
                double zs_res_137090 = zs_lhs_137089 / 2.0;
                
                // futhark/microgpt.fut:348:88-127
                
                double zp_res_137092 = zp_rhs_137053 + zs_res_137090;
                
                ((double *) mem_143495)[i_142129] = zp_res_137092;
                ((double *) mem_143496)[i_142129] = zp_res_137077;
                ((double *) mem_143497)[i_142129] = zp_res_137064;
                ((double *) mem_143498)[i_142129] = zp_res_137054;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143475, i_142142 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143495, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143476, i_142142 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143496, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143477, i_142142 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143497, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143478, i_142142 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143498, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143451, i_142155 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_143475, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143452, i_142155 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_143476, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143453, i_142155 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_143477, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143454, i_142155 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_143478, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143559_cached_sizze_145733 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143559, &mem_143559_cached_sizze_145733, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143560_cached_sizze_145734 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143560, &mem_143560_cached_sizze_145734, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143561_cached_sizze_145735 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143561, &mem_143561_cached_sizze_145735, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143562_cached_sizze_145736 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143562, &mem_143562_cached_sizze_145736, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143563_cached_sizze_145737 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143563, &mem_143563_cached_sizze_145737, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143564_cached_sizze_145738 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143564, &mem_143564_cached_sizze_145738, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143565_cached_sizze_145739 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143565, &mem_143565_cached_sizze_145739, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143566_cached_sizze_145740 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143566, &mem_143566_cached_sizze_145740, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143599_cached_sizze_145741 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143599, &mem_143599_cached_sizze_145741, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143600_cached_sizze_145742 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143600, &mem_143600_cached_sizze_145742, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143601_cached_sizze_145743 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143601, &mem_143601_cached_sizze_145743, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143602_cached_sizze_145744 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143602, &mem_143602_cached_sizze_145744, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143603_cached_sizze_145745 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143603, &mem_143603_cached_sizze_145745, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143604_cached_sizze_145746 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143604, &mem_143604_cached_sizze_145746, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143605_cached_sizze_145747 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143605, &mem_143605_cached_sizze_145747, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143606_cached_sizze_145748 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143606, &mem_143606_cached_sizze_145748, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142208 = 0; i_142208 < (int64_t) 4; i_142208++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142183 = 0; i_142183 < (int64_t) 16; i_142183++) {
            // futhark/microgpt.fut:105:13-33
            
            double defunc_0_reduce_res_141486;
            double defunc_0_reduce_res_141487;
            double defunc_0_reduce_res_141488;
            double defunc_0_reduce_res_141489;
            double defunc_0_reduce_res_141490;
            double defunc_0_reduce_res_141491;
            double redout_142160;
            double redout_142161;
            double redout_142162;
            double redout_142163;
            double redout_142164;
            double redout_142165;
            
            redout_142160 = -INFINITY;
            redout_142161 = -INFINITY;
            redout_142162 = -INFINITY;
            redout_142163 = -INFINITY;
            redout_142164 = -INFINITY;
            redout_142165 = -INFINITY;
            for (int64_t i_142166 = 0; i_142166 < (int64_t) 16; i_142166++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_138404 = ((double *) mem_143454)[i_142208 * (int64_t) 256 + i_142183 * (int64_t) 16 + i_142166];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_138414 = ((double *) mem_143453)[i_142208 * (int64_t) 256 + i_142183 * (int64_t) 16 + i_142166];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_138433 = ((double *) mem_143452)[i_142208 * (int64_t) 256 + i_142183 * (int64_t) 16 + i_142166];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_138477 = ((double *) mem_143451)[i_142208 * (int64_t) 256 + i_142183 * (int64_t) 16 + i_142166];
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_137704 = fmax64(lifted_lambda_res_138404, redout_142160);
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_137723 = fmax64(lifted_lambda_res_138414, redout_142161);
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_137745 = fmax64(lifted_lambda_res_138433, redout_142162);
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_137770 = fmax64(lifted_lambda_res_138433, redout_142163);
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_137820 = fmax64(lifted_lambda_res_138477, redout_142164);
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_137851 = fmax64(lifted_lambda_res_138477, redout_142165);
                double redout_tmp_145293 = max_res_137704;
                double redout_tmp_145294 = max_res_137723;
                double redout_tmp_145295 = max_res_137745;
                double redout_tmp_145296 = max_res_137770;
                double redout_tmp_145297 = max_res_137820;
                double redout_tmp_145298 = max_res_137851;
                
                redout_142160 = redout_tmp_145293;
                redout_142161 = redout_tmp_145294;
                redout_142162 = redout_tmp_145295;
                redout_142163 = redout_tmp_145296;
                redout_142164 = redout_tmp_145297;
                redout_142165 = redout_tmp_145298;
            }
            defunc_0_reduce_res_141486 = redout_142160;
            defunc_0_reduce_res_141487 = redout_142161;
            defunc_0_reduce_res_141488 = redout_142162;
            defunc_0_reduce_res_141489 = redout_142163;
            defunc_0_reduce_res_141490 = redout_142164;
            defunc_0_reduce_res_141491 = redout_142165;
            // futhark/microgpt.fut:343:148-174
            
            double neg_res_137778 = -defunc_0_reduce_res_141489;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137779;
            double r_137781 = 0.0;
            
            for (int64_t i_137780 = 0; i_137780 < (int64_t) 16; i_137780++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_137782 = ((double *) mem_143452)[i_142208 * (int64_t) 256 + i_142183 * (int64_t) 16 + i_137780];
                
                // futhark/microgpt.fut:343:114-174
                
                double zp_res_137783 = neg_res_137778 + zp_lhs_137782;
                
                // futhark/microgpt.fut:343:107-174
                
                double neg_res_137784 = -zp_res_137783;
                
                // futhark/microgpt.fut:100:42-54
                
                double max_res_137785 = fmax64(0.0, neg_res_137784);
                
                // futhark/microgpt.fut:100:35-54
                
                double sgn_res_137786 = fsignum64(max_res_137785);
                
                // futhark/microgpt.fut:343:88-177
                
                double neg_res_137787 = -sgn_res_137786;
                
                // futhark/microgpt.fut:343:79-178
                
                double zp_res_137788 = 1.0 + neg_res_137787;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137789 = r_137781 + zp_res_137788;
                double r_tmp_145299 = zp_res_137789;
                
                r_137781 = r_tmp_145299;
            }
            defunc_0_lifted_lambda_res_137779 = r_137781;
            // futhark/microgpt.fut:343:48-181
            
            double zs_res_137790 = 1.0 / defunc_0_lifted_lambda_res_137779;
            
            // futhark/microgpt.fut:359:148-174
            
            double neg_res_137859 = -defunc_0_reduce_res_141491;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137860;
            double r_137862 = 0.0;
            
            for (int64_t i_137861 = 0; i_137861 < (int64_t) 16; i_137861++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_137863 = ((double *) mem_143451)[i_142208 * (int64_t) 256 + i_142183 * (int64_t) 16 + i_137861];
                
                // futhark/microgpt.fut:359:114-174
                
                double zp_res_137864 = neg_res_137859 + zp_lhs_137863;
                
                // futhark/microgpt.fut:359:107-174
                
                double neg_res_137865 = -zp_res_137864;
                
                // futhark/microgpt.fut:100:42-54
                
                double max_res_137866 = fmax64(0.0, neg_res_137865);
                
                // futhark/microgpt.fut:100:35-54
                
                double sgn_res_137867 = fsignum64(max_res_137866);
                
                // futhark/microgpt.fut:359:88-177
                
                double neg_res_137868 = -sgn_res_137867;
                
                // futhark/microgpt.fut:359:79-178
                
                double zp_res_137869 = 1.0 + neg_res_137868;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137870 = r_137862 + zp_res_137869;
                double r_tmp_145300 = zp_res_137870;
                
                r_137862 = r_tmp_145300;
            }
            defunc_0_lifted_lambda_res_137860 = r_137862;
            // futhark/microgpt.fut:359:48-181
            
            double zs_res_137871 = 1.0 / defunc_0_lifted_lambda_res_137860;
            
            ((double *) mem_143599)[i_142183] = zs_res_137871;
            ((double *) mem_143600)[i_142183] = defunc_0_reduce_res_141491;
            ((double *) mem_143601)[i_142183] = defunc_0_reduce_res_141490;
            ((double *) mem_143602)[i_142183] = zs_res_137790;
            ((double *) mem_143603)[i_142183] = defunc_0_reduce_res_141489;
            ((double *) mem_143604)[i_142183] = defunc_0_reduce_res_141488;
            ((double *) mem_143605)[i_142183] = defunc_0_reduce_res_141487;
            ((double *) mem_143606)[i_142183] = defunc_0_reduce_res_141486;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143559, i_142208 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143599, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143560, i_142208 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143600, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143561, i_142208 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143601, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143562, i_142208 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143602, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143563, i_142208 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143603, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143564, i_142208 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143604, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143565, i_142208 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143605, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143566, i_142208 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143606, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143687_cached_sizze_145749 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143687, &mem_143687_cached_sizze_145749, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143688_cached_sizze_145750 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143688, &mem_143688_cached_sizze_145750, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143689_cached_sizze_145751 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143689, &mem_143689_cached_sizze_145751, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143690_cached_sizze_145752 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143690, &mem_143690_cached_sizze_145752, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143711_cached_sizze_145753 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143711, &mem_143711_cached_sizze_145753, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143712_cached_sizze_145754 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143712, &mem_143712_cached_sizze_145754, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143713_cached_sizze_145755 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143713, &mem_143713_cached_sizze_145755, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143714_cached_sizze_145756 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143714, &mem_143714_cached_sizze_145756, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143731_cached_sizze_145757 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143731, &mem_143731_cached_sizze_145757, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143732_cached_sizze_145758 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143732, &mem_143732_cached_sizze_145758, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143733_cached_sizze_145759 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143733, &mem_143733_cached_sizze_145759, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143734_cached_sizze_145760 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143734, &mem_143734_cached_sizze_145760, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142251 = 0; i_142251 < (int64_t) 4; i_142251++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142238 = 0; i_142238 < (int64_t) 16; i_142238++) {
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_138693 = ((double *) mem_143566)[i_142251 * (int64_t) 16 + i_142238];
            
            // futhark/microgpt.fut:283:91-114
            
            double neg_res_138694 = -neg_arg0_138693;
            
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_138755 = ((double *) mem_143561)[i_142251 * (int64_t) 16 + i_142238];
            
            // futhark/microgpt.fut:352:99-125
            
            double neg_res_138756 = -neg_arg0_138755;
            
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_138732 = ((double *) mem_143564)[i_142251 * (int64_t) 16 + i_142238];
            
            // futhark/microgpt.fut:336:99-125
            
            double neg_res_138733 = -neg_arg0_138732;
            
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_138711 = ((double *) mem_143565)[i_142251 * (int64_t) 16 + i_142238];
            
            // futhark/microgpt.fut:325:99-125
            
            double neg_res_138712 = -neg_arg0_138711;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142225 = 0; i_142225 < (int64_t) 16; i_142225++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_138875 = ((double *) mem_143454)[i_142251 * (int64_t) 256 + i_142238 * (int64_t) 16 + i_142225];
                
                // futhark/microgpt.fut:283:61-114
                
                double zp_res_138876 = neg_res_138694 + zp_lhs_138875;
                
                // futhark/microgpt.fut:283:54-114
                
                double exp_res_138877 = futrts_exp64(zp_res_138876);
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_138884 = ((double *) mem_143453)[i_142251 * (int64_t) 256 + i_142238 * (int64_t) 16 + i_142225];
                
                // futhark/microgpt.fut:325:65-125
                
                double zp_res_138885 = neg_res_138712 + zp_lhs_138884;
                
                // futhark/microgpt.fut:325:58-125
                
                double exp_res_138886 = futrts_exp64(zp_res_138885);
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_138896 = ((double *) mem_143452)[i_142251 * (int64_t) 256 + i_142238 * (int64_t) 16 + i_142225];
                
                // futhark/microgpt.fut:336:65-125
                
                double zp_res_138897 = neg_res_138733 + zp_lhs_138896;
                
                // futhark/microgpt.fut:336:58-125
                
                double exp_res_138898 = futrts_exp64(zp_res_138897);
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_138910 = ((double *) mem_143451)[i_142251 * (int64_t) 256 + i_142238 * (int64_t) 16 + i_142225];
                
                // futhark/microgpt.fut:352:65-125
                
                double zp_res_138911 = neg_res_138756 + zp_lhs_138910;
                
                // futhark/microgpt.fut:352:58-125
                
                double exp_res_138912 = futrts_exp64(zp_res_138911);
                
                ((double *) mem_143731)[i_142225] = exp_res_138912;
                ((double *) mem_143732)[i_142225] = exp_res_138898;
                ((double *) mem_143733)[i_142225] = exp_res_138886;
                ((double *) mem_143734)[i_142225] = exp_res_138877;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143711, i_142238 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143731, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143712, i_142238 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143732, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143713, i_142238 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143733, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143714, i_142238 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143734, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143687, i_142251 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_143711, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143688, i_142251 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_143712, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143689, i_142251 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_143713, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143690, i_142251 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_143714, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143795_cached_sizze_145761 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143795, &mem_143795_cached_sizze_145761, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143796_cached_sizze_145762 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143796, &mem_143796_cached_sizze_145762, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143805_cached_sizze_145763 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143805, &mem_143805_cached_sizze_145763, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143806_cached_sizze_145764 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143806, &mem_143806_cached_sizze_145764, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142267 = 0; i_142267 < (int64_t) 4; i_142267++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142260 = 0; i_142260 < (int64_t) 16; i_142260++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_138944;
            double r_138946 = 0.0;
            
            for (int64_t i_138945 = 0; i_138945 < (int64_t) 16; i_138945++) {
                // futhark/microgpt.fut:61:46-49
                
                double lifted_lambda_res_138947 = ((double *) mem_143690)[i_142267 * (int64_t) 256 + i_142260 * (int64_t) 16 + i_138945];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_138948 = r_138946 + lifted_lambda_res_138947;
                double r_tmp_145317 = zp_res_138948;
                
                r_138946 = r_tmp_145317;
            }
            defunc_0_lifted_lambda_res_138944 = r_138946;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_138955;
            double r_138957 = 0.0;
            
            for (int64_t i_138956 = 0; i_138956 < (int64_t) 16; i_138956++) {
                // futhark/microgpt.fut:61:46-49
                
                double lifted_lambda_res_138958 = ((double *) mem_143689)[i_142267 * (int64_t) 256 + i_142260 * (int64_t) 16 + i_138956];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_138959 = r_138957 + lifted_lambda_res_138958;
                double r_tmp_145318 = zp_res_138959;
                
                r_138957 = r_tmp_145318;
            }
            defunc_0_lifted_lambda_res_138955 = r_138957;
            ((double *) mem_143805)[i_142260] = defunc_0_lifted_lambda_res_138955;
            ((double *) mem_143806)[i_142260] = defunc_0_lifted_lambda_res_138944;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143795, i_142267 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143805, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143796, i_142267 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143806, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143827_cached_sizze_145765 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143827, &mem_143827_cached_sizze_145765, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143828_cached_sizze_145766 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143828, &mem_143828_cached_sizze_145766, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143839_cached_sizze_145767 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143839, &mem_143839_cached_sizze_145767, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143840_cached_sizze_145768 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143840, &mem_143840_cached_sizze_145768, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143849_cached_sizze_145769 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143849, &mem_143849_cached_sizze_145769, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143850_cached_sizze_145770 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143850, &mem_143850_cached_sizze_145770, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142288 = 0; i_142288 < (int64_t) 4; i_142288++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142281 = 0; i_142281 < (int64_t) 16; i_142281++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_rhs_138979 = ((double *) mem_143796)[i_142288 * (int64_t) 16 + i_142281];
            
            // futhark/microgpt.fut:285:84-109
            
            double zs_res_138980 = 1.0 / zs_rhs_138979;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_rhs_138996 = ((double *) mem_143795)[i_142288 * (int64_t) 16 + i_142281];
            
            // futhark/microgpt.fut:327:92-120
            
            double zs_res_138997 = 1.0 / zs_rhs_138996;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142274 = 0; i_142274 < (int64_t) 16; i_142274++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_139024 = ((double *) mem_143690)[i_142288 * (int64_t) 256 + i_142281 * (int64_t) 16 + i_142274];
                
                // futhark/microgpt.fut:285:54-109
                
                double zt_res_139025 = zs_res_138980 * zt_lhs_139024;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_139032 = ((double *) mem_143689)[i_142288 * (int64_t) 256 + i_142281 * (int64_t) 16 + i_142274];
                
                // futhark/microgpt.fut:327:58-120
                
                double zt_res_139033 = zs_res_138997 * zt_lhs_139032;
                
                ((double *) mem_143849)[i_142274] = zt_res_139033;
                ((double *) mem_143850)[i_142274] = zt_res_139025;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143839, i_142281 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143849, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143840, i_142281 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143850, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143827, i_142288 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_143839, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143828, i_142288 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_143840, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143881_cached_sizze_145771 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143881, &mem_143881_cached_sizze_145771, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143882_cached_sizze_145772 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_143882, &mem_143882_cached_sizze_145772, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143893_cached_sizze_145773 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143893, &mem_143893_cached_sizze_145773, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143894_cached_sizze_145774 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143894, &mem_143894_cached_sizze_145774, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143903_cached_sizze_145775 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143903, &mem_143903_cached_sizze_145775, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143904_cached_sizze_145776 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143904, &mem_143904_cached_sizze_145776, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142309 = 0; i_142309 < (int64_t) 4; i_142309++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142302 = 0; i_142302 < (int64_t) 16; i_142302++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142295 = 0; i_142295 < (int64_t) 16; i_142295++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_139096 = ((double *) mem_143828)[i_142309 * (int64_t) 256 + i_142302 * (int64_t) 16 + i_142295];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_139103 = ((double *) mem_143827)[i_142309 * (int64_t) 256 + i_142302 * (int64_t) 16 + i_142295];
                
                ((double *) mem_143903)[i_142295] = lifted_lambda_res_139103;
                ((double *) mem_143904)[i_142295] = lifted_lambda_res_139096;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143893, i_142302 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143903, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143894, i_142302 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143904, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143881, i_142309 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_143893, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143882, i_142309 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_143894, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143935_cached_sizze_145777 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143935, &mem_143935_cached_sizze_145777, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143941_cached_sizze_145778 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_143941, &mem_143941_cached_sizze_145778, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143946_cached_sizze_145779 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_143946, &mem_143946_cached_sizze_145779, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142322 = 0; i_142322 < (int64_t) 4; i_142322++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142318 = 0; i_142318 < (int64_t) 16; i_142318++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142314 = 0; i_142314 < (int64_t) 4; i_142314++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_128933;
                double r_128935 = 0.0;
                
                for (int64_t i_128934 = 0; i_128934 < (int64_t) 16; i_128934++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_128936 = ((double *) mem_143882)[i_142322 * (int64_t) 256 + i_142318 * (int64_t) 16 + i_128934];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_128937 = ((double *) mem_143262)[i_142322 * (int64_t) 64 + i_128934 * (int64_t) 4 + i_142314];
                    
                    // futhark/microgpt.fut:287:74-127
                    
                    double zt_res_128938 = zt_lhs_128936 * zt_rhs_128937;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_128939 = r_128935 + zt_res_128938;
                    double r_tmp_145334 = zp_res_128939;
                    
                    r_128935 = r_tmp_145334;
                }
                defunc_0_lifted_lambda_res_128933 = r_128935;
                ((double *) mem_143946)[i_142314] = defunc_0_lifted_lambda_res_128933;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_143941, i_142318 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143946, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_143935, i_142322 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_143941, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143962_cached_sizze_145780 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143962, &mem_143962_cached_sizze_145780, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143967_cached_sizze_145781 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143967, &mem_143967_cached_sizze_145781, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142330 = 0; i_142330 < (int64_t) 16; i_142330++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142326 = 0; i_142326 < (int64_t) 16; i_142326++) {
            // futhark/microgpt.fut:288:15-18
            
            int64_t tmp_128951 = sdiv64(i_142326, (int64_t) 4);
            
            // futhark/microgpt.fut:288:4-20
            
            bool x_128952 = sle64((int64_t) 0, tmp_128951);
            
            // futhark/microgpt.fut:288:4-20
            
            bool y_128953 = slt64(tmp_128951, (int64_t) 4);
            
            // futhark/microgpt.fut:288:4-20
            
            bool bounds_check_128954 = x_128952 && y_128953;
            
            // futhark/microgpt.fut:288:4-20
            
            bool index_certs_128955;
            
            if (!bounds_check_128954) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_128951, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:288:4-20\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:280:12-288:49\n   #6  futhark/microgpt.fut:459:5-75\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:288:35-38
            
            int64_t tmp_128956 = smod64(i_142326, (int64_t) 4);
            
            // futhark/microgpt.fut:288:4-40
            
            bool x_128957 = sle64((int64_t) 0, tmp_128956);
            
            // futhark/microgpt.fut:288:4-40
            
            bool y_128958 = slt64(tmp_128956, (int64_t) 4);
            
            // futhark/microgpt.fut:288:4-40
            
            bool bounds_check_128959 = x_128957 && y_128958;
            
            // futhark/microgpt.fut:288:4-40
            
            bool index_certs_128960;
            
            if (!bounds_check_128959) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_128956, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:288:4-40\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:280:12-288:49\n   #6  futhark/microgpt.fut:459:5-75\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_128961 = ((double *) mem_143935)[tmp_128951 * (int64_t) 64 + i_142330 * (int64_t) 4 + tmp_128956];
            
            ((double *) mem_143967)[i_142326] = lifted_lambda_res_128961;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143962, i_142330 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143967, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143978_cached_sizze_145782 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143978, &mem_143978_cached_sizze_145782, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143983_cached_sizze_145783 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143983, &mem_143983_cached_sizze_145783, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142338 = 0; i_142338 < (int64_t) 16; i_142338++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142334 = 0; i_142334 < (int64_t) 16; i_142334++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128976;
            double r_128978 = 0.0;
            
            for (int64_t i_128977 = 0; i_128977 < (int64_t) 16; i_128977++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128979 = ((double *) wout_mem_143123.mem)[i_142334 * (int64_t) 16 + i_128977];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128980 = ((double *) mem_143962)[i_142338 * (int64_t) 16 + i_128977];
                
                // futhark/microgpt.fut:289:64-104
                
                double zt_res_128981 = zt_lhs_128979 * zt_rhs_128980;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128982 = r_128978 + zt_res_128981;
                double r_tmp_145339 = zp_res_128982;
                
                r_128978 = r_tmp_145339;
            }
            defunc_0_lifted_lambda_res_128976 = r_128978;
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_128983 = ((double *) mem_143193)[i_142338 * (int64_t) 16 + i_142334];
            
            // futhark/microgpt.fut:289:43-128
            
            double zp_res_128984 = defunc_0_lifted_lambda_res_128976 + zp_rhs_128983;
            
            ((double *) mem_143983)[i_142334] = zp_res_128984;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143978, i_142338 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143983, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143994_cached_sizze_145784 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143994, &mem_143994_cached_sizze_145784, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143995_cached_sizze_145785 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143995, &mem_143995_cached_sizze_145785, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142344 = 0; i_142344 < (int64_t) 16; i_142344++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_134078;
        double r_134080 = 0.0;
        
        for (int64_t i_134079 = 0; i_134079 < (int64_t) 16; i_134079++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_134081 = ((double *) mem_143978)[i_142344 * (int64_t) 16 + i_134079];
            
            // futhark/microgpt.fut:290:66-105
            
            double zt_res_134082 = zt_lhs_134081 * zt_lhs_134081;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_134083 = r_134080 + zt_res_134082;
            double r_tmp_145342 = zp_res_134083;
            
            r_134080 = r_tmp_145342;
        }
        defunc_0_lifted_lambda_res_134078 = r_134080;
        // futhark/microgpt.fut:290:45-123
        
        double zs_res_134084 = defunc_0_lifted_lambda_res_134078 / 16.0;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_134091;
        double r_134093 = 0.0;
        
        for (int64_t i_134092 = 0; i_134092 < (int64_t) 16; i_134092++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_134094 = ((double *) mem_143978)[i_142344 * (int64_t) 16 + i_134092];
            
            // futhark/microgpt.fut:315:70-113
            
            double zt_res_134095 = zt_lhs_134094 * zt_lhs_134094;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_134096 = r_134093 + zt_res_134095;
            double r_tmp_145343 = zp_res_134096;
            
            r_134093 = r_tmp_145343;
        }
        defunc_0_lifted_lambda_res_134091 = r_134093;
        // futhark/microgpt.fut:315:48-131
        
        double zs_res_134097 = defunc_0_lifted_lambda_res_134091 / 16.0;
        
        ((double *) mem_143994)[i_142344] = zs_res_134097;
        ((double *) mem_143995)[i_142344] = zs_res_134084;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144008_cached_sizze_145786 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144008, &mem_144008_cached_sizze_145786, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142349 = 0; i_142349 < (int64_t) 16; i_142349++) {
        // futhark/microgpt.fut:291:45-55
        
        double zp_lhs_129007 = ((double *) mem_143995)[i_142349];
        
        // futhark/microgpt.fut:291:45-83
        
        double zp_res_129008 = 1.0e-5 + zp_lhs_129007;
        
        // futhark/microgpt.fut:291:37-83
        
        double sqrt_res_129009 = futrts_sqrt64(zp_res_129008);
        
        ((double *) mem_144008)[i_142349] = sqrt_res_129009;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144015_cached_sizze_145787 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144015, &mem_144015_cached_sizze_145787, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144020_cached_sizze_145788 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144020, &mem_144020_cached_sizze_145788, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142357 = 0; i_142357 < (int64_t) 16; i_142357++) {
        // futhark/microgpt.fut:292:77-87
        
        double zs_rhs_129017 = ((double *) mem_144008)[i_142357];
        
        // futhark/microgpt.fut:292:69-87
        
        double zs_res_129018 = 1.0 / zs_rhs_129017;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142353 = 0; i_142353 < (int64_t) 16; i_142353++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_129025 = ((double *) mem_143978)[i_142357 * (int64_t) 16 + i_142353];
            
            // futhark/microgpt.fut:292:46-87
            
            double zt_res_129026 = zs_res_129018 * zt_lhs_129025;
            
            ((double *) mem_144020)[i_142353] = zt_res_129026;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144015, i_142357 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144020, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144031_cached_sizze_145789 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144031, &mem_144031_cached_sizze_145789, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144036_cached_sizze_145790 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144036, &mem_144036_cached_sizze_145790, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142365 = 0; i_142365 < (int64_t) 16; i_142365++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142361 = 0; i_142361 < (int64_t) 16; i_142361++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_129041 = ((double *) mem_144015)[i_142365 * (int64_t) 16 + i_142361];
            
            ((double *) mem_144036)[i_142361] = lifted_lambda_res_129041;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144031, i_142365 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144036, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144047_cached_sizze_145791 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144047, &mem_144047_cached_sizze_145791, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144052_cached_sizze_145792 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144052, &mem_144052_cached_sizze_145792, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142373 = 0; i_142373 < (int64_t) 16; i_142373++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142369 = 0; i_142369 < (int64_t) 64; i_142369++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129057;
            double r_129059 = 0.0;
            
            for (int64_t i_129058 = 0; i_129058 < (int64_t) 16; i_129058++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_129060 = ((double *) wup_mem_143127.mem)[i_142369 * (int64_t) 16 + i_129058];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_129061 = ((double *) mem_144031)[i_142373 * (int64_t) 16 + i_129058];
                
                // futhark/microgpt.fut:294:63-102
                
                double zt_res_129062 = zt_lhs_129060 * zt_rhs_129061;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129063 = r_129059 + zt_res_129062;
                double r_tmp_145351 = zp_res_129063;
                
                r_129059 = r_tmp_145351;
            }
            defunc_0_lifted_lambda_res_129057 = r_129059;
            ((double *) mem_144052)[i_142369] = defunc_0_lifted_lambda_res_129057;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144047, i_142373 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144052, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144063_cached_sizze_145793 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144063, &mem_144063_cached_sizze_145793, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144068_cached_sizze_145794 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144068, &mem_144068_cached_sizze_145794, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142381 = 0; i_142381 < (int64_t) 16; i_142381++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142377 = 0; i_142377 < (int64_t) 64; i_142377++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_129078 = ((double *) mem_144047)[i_142381 * (int64_t) 64 + i_142377];
            
            // futhark/microgpt.fut:295:41-69
            
            double max_res_129079 = fmax64(0.0, max_arg0_129078);
            
            ((double *) mem_144068)[i_142377] = max_res_129079;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144063, i_142381 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144068, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144079_cached_sizze_145795 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144079, &mem_144079_cached_sizze_145795, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144084_cached_sizze_145796 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144084, &mem_144084_cached_sizze_145796, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142389 = 0; i_142389 < (int64_t) 16; i_142389++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142385 = 0; i_142385 < (int64_t) 16; i_142385++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129094;
            double r_129096 = 0.0;
            
            for (int64_t i_129095 = 0; i_129095 < (int64_t) 64; i_129095++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_129097 = ((double *) wdown_mem_143121.mem)[i_142385 * (int64_t) 64 + i_129095];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_129098 = ((double *) mem_144063)[i_142389 * (int64_t) 64 + i_129095];
                
                // futhark/microgpt.fut:296:64-105
                
                double zt_res_129099 = zt_lhs_129097 * zt_rhs_129098;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129100 = r_129096 + zt_res_129099;
                double r_tmp_145356 = zp_res_129100;
                
                r_129096 = r_tmp_145356;
            }
            defunc_0_lifted_lambda_res_129094 = r_129096;
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_129101 = ((double *) mem_143978)[i_142389 * (int64_t) 16 + i_142385];
            
            // futhark/microgpt.fut:296:43-130
            
            double zp_res_129102 = defunc_0_lifted_lambda_res_129094 + zp_rhs_129101;
            
            ((double *) mem_144084)[i_142385] = zp_res_129102;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144079, i_142389 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144084, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144095_cached_sizze_145797 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144095, &mem_144095_cached_sizze_145797, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144100_cached_sizze_145798 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144100, &mem_144100_cached_sizze_145798, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142397 = 0; i_142397 < (int64_t) 16; i_142397++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142393 = 0; i_142393 < (int64_t) 27; i_142393++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129118;
            double r_129120 = 0.0;
            
            for (int64_t i_129119 = 0; i_129119 < (int64_t) 16; i_129119++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_129121 = ((double *) wvoc_mem_143129.mem)[i_142393 * (int64_t) 16 + i_129119];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_129122 = ((double *) mem_144079)[i_142397 * (int64_t) 16 + i_129119];
                
                // futhark/microgpt.fut:297:63-103
                
                double zt_res_129123 = zt_lhs_129121 * zt_rhs_129122;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129124 = r_129120 + zt_res_129123;
                double r_tmp_145359 = zp_res_129124;
                
                r_129120 = r_tmp_145359;
            }
            defunc_0_lifted_lambda_res_129118 = r_129120;
            ((double *) mem_144100)[i_142393] = defunc_0_lifted_lambda_res_129118;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144095, i_142397 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144100, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144111_cached_sizze_145799 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144111, &mem_144111_cached_sizze_145799, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144112_cached_sizze_145800 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144112, &mem_144112_cached_sizze_145800, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144113_cached_sizze_145801 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_144113, &mem_144113_cached_sizze_145801, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144114_cached_sizze_145802 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144114, &mem_144114_cached_sizze_145802, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144133_cached_sizze_145803 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144133, &mem_144133_cached_sizze_145803, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144140_cached_sizze_145804 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144140, &mem_144140_cached_sizze_145804, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144147_cached_sizze_145805 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_144147, &mem_144147_cached_sizze_145805, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144152_cached_sizze_145806 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144152, &mem_144152_cached_sizze_145806, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142427 = 0; i_142427 < (int64_t) 16; i_142427++) {
        // futhark/microgpt.fut:105:13-33
        
        double defunc_0_reduce_res_141535;
        double defunc_0_reduce_res_141536;
        double defunc_0_reduce_res_141537;
        double redout_142399;
        double redout_142400;
        double redout_142401;
        
        redout_142399 = -INFINITY;
        redout_142400 = -INFINITY;
        redout_142401 = -INFINITY;
        for (int64_t i_142402 = 0; i_142402 < (int64_t) 27; i_142402++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_139253 = ((double *) mem_144095)[i_142427 * (int64_t) 27 + i_142402];
            
            // futhark/microgpt.fut:105:13-33
            
            double max_res_134233 = fmax64(lifted_lambda_res_139253, redout_142399);
            
            // futhark/microgpt.fut:105:13-33
            
            double max_res_134253 = fmax64(lifted_lambda_res_139253, redout_142400);
            
            // futhark/microgpt.fut:105:13-33
            
            double max_res_134323 = fmax64(lifted_lambda_res_139253, redout_142401);
            double redout_tmp_145364 = max_res_134233;
            double redout_tmp_145365 = max_res_134253;
            double redout_tmp_145366 = max_res_134323;
            
            redout_142399 = redout_tmp_145364;
            redout_142400 = redout_tmp_145365;
            redout_142401 = redout_tmp_145366;
        }
        defunc_0_reduce_res_141535 = redout_142399;
        defunc_0_reduce_res_141536 = redout_142400;
        defunc_0_reduce_res_141537 = redout_142401;
        // futhark/microgpt.fut:4:11-25
        for (int64_t nest_i_145367 = 0; nest_i_145367 < (int64_t) 27; nest_i_145367++) {
            ((double *) mem_144114)[i_142427 * (int64_t) 27 + nest_i_145367] = defunc_0_reduce_res_141535;
        }
        // futhark/microgpt.fut:303:67-76
        
        double neg_res_134254 = -defunc_0_reduce_res_141536;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142405 = 0; i_142405 < (int64_t) 27; i_142405++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_134261 = ((double *) mem_144095)[i_142427 * (int64_t) 27 + i_142405];
            
            // futhark/microgpt.fut:303:44-76
            
            double zp_res_134262 = neg_res_134254 + zp_lhs_134261;
            
            // futhark/microgpt.fut:303:37-76
            
            double exp_res_134263 = futrts_exp64(zp_res_134262);
            
            ((double *) mem_144133)[i_142405] = exp_res_134263;
        }
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_134265;
        double r_134267 = 0.0;
        
        for (int64_t i_134266 = 0; i_134266 < (int64_t) 27; i_134266++) {
            // futhark/microgpt.fut:304:36-46
            
            double lifted_lambda_res_134268 = ((double *) mem_144133)[i_134266];
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_134269 = r_134267 + lifted_lambda_res_134268;
            double r_tmp_145369 = zp_res_134269;
            
            r_134267 = r_tmp_145369;
        }
        defunc_0_lifted_lambda_res_134265 = r_134267;
        // futhark/microgpt.fut:305:55-66
        
        double zs_res_134270 = 1.0 / defunc_0_lifted_lambda_res_134265;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142409 = 0; i_142409 < (int64_t) 27; i_142409++) {
            // futhark/microgpt.fut:305:38-49
            
            double zt_lhs_134277 = ((double *) mem_144133)[i_142409];
            
            // futhark/microgpt.fut:305:38-66
            
            double zt_res_134278 = zs_res_134270 * zt_lhs_134277;
            
            ((double *) mem_144140)[i_142409] = zt_res_134278;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142417 = 0; i_142417 < (int64_t) 27; i_142417++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142413 = 0; i_142413 < (int64_t) 27; i_142413++) {
                // futhark/microgpt.fut:306:6-88
                
                bool cond_134284 = i_142413 == i_142417;
                
                // futhark/microgpt.fut:306:6-88
                
                double zt_lhs_134285;
                
                if (cond_134284) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_141532 = ((double *) target_mem_143131.mem)[i_142427 * (int64_t) 27 + i_142417];
                    
                    // futhark/microgpt.fut:306:35-77
                    
                    double zt_res_141533 = -6.25e-2 * zt_rhs_141532;
                    
                    zt_lhs_134285 = zt_res_141533;
                } else {
                    zt_lhs_134285 = 0.0;
                }
                // futhark/microgpt.fut:306:103-113
                
                double zs_rhs_134296 = ((double *) mem_144140)[i_142413];
                
                // futhark/microgpt.fut:306:95-113
                
                double zs_res_134297 = 1.0 / zs_rhs_134296;
                
                // futhark/microgpt.fut:306:6-113
                
                double zt_res_134298 = zt_lhs_134285 * zs_res_134297;
                
                ((double *) mem_144152)[i_142413] = zt_res_134298;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144147, i_142417 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144152, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t nest_i_145373 = 0; nest_i_145373 < (int64_t) 27; nest_i_145373++) {
            ((double *) mem_144112)[i_142427 * (int64_t) 27 + nest_i_145373] = defunc_0_reduce_res_141537;
        }
        // futhark/microgpt.fut:311:139-164
        
        double neg_res_134334 = -defunc_0_reduce_res_141537;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_134335;
        double r_134337 = 0.0;
        
        for (int64_t i_134336 = 0; i_134336 < (int64_t) 27; i_134336++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_134338 = ((double *) mem_144095)[i_142427 * (int64_t) 27 + i_134336];
            
            // futhark/microgpt.fut:311:114-164
            
            double zp_res_134339 = neg_res_134334 + zp_lhs_134338;
            
            // futhark/microgpt.fut:311:107-164
            
            double neg_res_134340 = -zp_res_134339;
            
            // futhark/microgpt.fut:100:42-54
            
            double max_res_134341 = fmax64(0.0, neg_res_134340);
            
            // futhark/microgpt.fut:100:35-54
            
            double sgn_res_134342 = fsignum64(max_res_134341);
            
            // futhark/microgpt.fut:311:88-167
            
            double neg_res_134343 = -sgn_res_134342;
            
            // futhark/microgpt.fut:311:79-168
            
            double zp_res_134344 = 1.0 + neg_res_134343;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_134345 = r_134337 + zp_res_134344;
            double r_tmp_145374 = zp_res_134345;
            
            r_134337 = r_tmp_145374;
        }
        defunc_0_lifted_lambda_res_134335 = r_134337;
        // futhark/microgpt.fut:311:48-171
        
        double zs_res_134346 = 1.0 / defunc_0_lifted_lambda_res_134335;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t nest_i_145375 = 0; nest_i_145375 < (int64_t) 27; nest_i_145375++) {
            ((double *) mem_144111)[i_142427 * (int64_t) 27 + nest_i_145375] = zs_res_134346;
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144113, i_142427 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_144147, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144182_cached_sizze_145807 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_144182, &mem_144182_cached_sizze_145807, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144188_cached_sizze_145808 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_144188, &mem_144188_cached_sizze_145808, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144193_cached_sizze_145809 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144193, &mem_144193_cached_sizze_145809, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142442 = 0; i_142442 < (int64_t) 16; i_142442++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142438 = 0; i_142438 < (int64_t) 27; i_142438++) {
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_129160 = ((double *) mem_144114)[i_142442 * (int64_t) 27 + i_142438];
            
            // futhark/microgpt.fut:300:85-108
            
            double neg_res_129161 = -neg_arg0_129160;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142434 = 0; i_142434 < (int64_t) 27; i_142434++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_129168 = ((double *) mem_144095)[i_142442 * (int64_t) 27 + i_142434];
                
                // futhark/microgpt.fut:300:62-108
                
                double zp_res_129169 = neg_res_129161 + zp_lhs_129168;
                
                // futhark/microgpt.fut:300:55-108
                
                double exp_res_129170 = futrts_exp64(zp_res_129169);
                
                ((double *) mem_144193)[i_142434] = exp_res_129170;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144188, i_142438 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144193, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144182, i_142442 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_144188, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144209_cached_sizze_145810 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144209, &mem_144209_cached_sizze_145810, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144210_cached_sizze_145811 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144210, &mem_144210_cached_sizze_145811, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144219_cached_sizze_145812 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144219, &mem_144219_cached_sizze_145812, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144220_cached_sizze_145813 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144220, &mem_144220_cached_sizze_145813, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142455 = 0; i_142455 < (int64_t) 16; i_142455++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142448 = 0; i_142448 < (int64_t) 27; i_142448++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_139335;
            double r_139337 = 0.0;
            
            for (int64_t i_139336 = 0; i_139336 < (int64_t) 27; i_139336++) {
                // futhark/microgpt.fut:61:46-49
                
                double lifted_lambda_res_139338 = ((double *) mem_144182)[i_142455 * (int64_t) 729 + i_142448 * (int64_t) 27 + i_139336];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_139339 = r_139337 + lifted_lambda_res_139338;
                double r_tmp_145383 = zp_res_139339;
                
                r_139337 = r_tmp_145383;
            }
            defunc_0_lifted_lambda_res_139335 = r_139337;
            // futhark/microgpt.fut:307:153-196
            
            double zt_res_139347 = defunc_0_lifted_lambda_res_139335 * defunc_0_lifted_lambda_res_139335;
            
            // futhark/microgpt.fut:307:144-196
            
            double zs_res_139348 = 1.0 / zt_res_139347;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_139349;
            double r_139351 = 0.0;
            
            for (int64_t i_139350 = 0; i_139350 < (int64_t) 27; i_139350++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_139352 = ((double *) mem_144113)[i_142455 * (int64_t) 729 + i_142448 * (int64_t) 27 + i_139350];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_139353 = ((double *) mem_144182)[i_142455 * (int64_t) 729 + i_142448 * (int64_t) 27 + i_139350];
                
                // futhark/microgpt.fut:307:78-137
                
                double zt_res_139354 = zt_lhs_139352 * zt_rhs_139353;
                
                // futhark/microgpt.fut:307:106-196
                
                double zt_res_139355 = zs_res_139348 * zt_res_139354;
                
                // futhark/microgpt.fut:307:70-196
                
                double neg_res_139356 = -zt_res_139355;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_139357 = r_139351 + neg_res_139356;
                double r_tmp_145384 = zp_res_139357;
                
                r_139351 = r_tmp_145384;
            }
            defunc_0_lifted_lambda_res_139349 = r_139351;
            ((double *) mem_144219)[i_142448] = defunc_0_lifted_lambda_res_139349;
            ((double *) mem_144220)[i_142448] = defunc_0_lifted_lambda_res_139335;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144209, i_142455 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144219, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144210, i_142455 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144220, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144241_cached_sizze_145814 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_144241, &mem_144241_cached_sizze_145814, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144247_cached_sizze_145815 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_144247, &mem_144247_cached_sizze_145815, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144252_cached_sizze_145816 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144252, &mem_144252_cached_sizze_145816, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142468 = 0; i_142468 < (int64_t) 16; i_142468++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142464 = 0; i_142464 < (int64_t) 27; i_142464++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_rhs_129299 = ((double *) mem_144210)[i_142468 * (int64_t) 27 + i_142464];
            
            // futhark/microgpt.fut:308:92-119
            
            double zs_res_129300 = 1.0 / zs_rhs_129299;
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_129301 = ((double *) mem_144209)[i_142468 * (int64_t) 27 + i_142464];
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142460 = 0; i_142460 < (int64_t) 27; i_142460++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_129308 = ((double *) mem_144113)[i_142468 * (int64_t) 729 + i_142464 * (int64_t) 27 + i_142460];
                
                // futhark/microgpt.fut:308:59-119
                
                double zt_res_129309 = zs_res_129300 * zt_lhs_129308;
                
                // futhark/microgpt.fut:308:87-145
                
                double zp_res_129310 = zp_rhs_129301 + zt_res_129309;
                
                ((double *) mem_144252)[i_142460] = zp_res_129310;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144247, i_142464 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144252, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144241, i_142468 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_144247, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144268_cached_sizze_145817 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144268, &mem_144268_cached_sizze_145817, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144273_cached_sizze_145818 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144273, &mem_144273_cached_sizze_145818, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142476 = 0; i_142476 < (int64_t) 16; i_142476++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142472 = 0; i_142472 < (int64_t) 27; i_142472++) {
            double f_elem_129323 = ((double *) mem_144114)[i_142476 * (int64_t) 27 + i_142472];
            
            // futhark/microgpt.fut:309:110-135
            
            double neg_res_129328 = -f_elem_129323;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129329;
            double r_129331 = 0.0;
            
            for (int64_t i_129330 = 0; i_129330 < (int64_t) 27; i_129330++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_129332 = ((double *) mem_144095)[i_142476 * (int64_t) 27 + i_129330];
                
                // futhark/microgpt.fut:309:85-135
                
                double zp_res_129333 = neg_res_129328 + zp_lhs_129332;
                
                // futhark/microgpt.fut:309:78-135
                
                double exp_res_129334 = futrts_exp64(zp_res_129333);
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_129335 = ((double *) mem_144241)[i_142476 * (int64_t) 729 + i_142472 * (int64_t) 27 + i_129330];
                
                // futhark/microgpt.fut:309:78-170
                
                double zt_res_129336 = exp_res_129334 * zt_rhs_129335;
                
                // futhark/microgpt.fut:309:70-170
                
                double neg_res_129337 = -zt_res_129336;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129338 = r_129331 + neg_res_129337;
                double r_tmp_145390 = zp_res_129338;
                
                r_129331 = r_tmp_145390;
            }
            defunc_0_lifted_lambda_res_129329 = r_129331;
            ((double *) mem_144273)[i_142472] = defunc_0_lifted_lambda_res_129329;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144268, i_142476 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144273, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144284_cached_sizze_145819 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144284, &mem_144284_cached_sizze_145819, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144289_cached_sizze_145820 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144289, &mem_144289_cached_sizze_145820, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142484 = 0; i_142484 < (int64_t) 16; i_142484++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142480 = 0; i_142480 < (int64_t) 27; i_142480++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129393;
            double r_129395 = 0.0;
            
            for (int64_t i_129394 = 0; i_129394 < (int64_t) 16; i_129394++) {
                // futhark/microgpt.fut:312:78-203
                
                bool cond_129396 = i_142484 == i_129394;
                
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_129400;
                
                if (cond_129396) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double x_141548 = ((double *) mem_144095)[i_129394 * (int64_t) 27 + i_142480];
                    
                    zp_lhs_129400 = x_141548;
                } else {
                    zp_lhs_129400 = 0.0;
                }
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_129402;
                
                if (cond_129396) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double x_141549 = ((double *) mem_144095)[i_129394 * (int64_t) 27 + i_142480];
                    
                    zp_lhs_129402 = x_141549;
                } else {
                    zp_lhs_129402 = 0.0;
                }
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_129404;
                double r_129406 = 0.0;
                
                for (int64_t i_129405 = 0; i_129405 < (int64_t) 27; i_129405++) {
                    // futhark/microgpt.fut:312:78-203
                    
                    double zp_lhs_129407;
                    
                    if (cond_129396) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double neg_arg0_141550 = ((double *) mem_144114)[i_129394 * (int64_t) 27 + i_129405];
                        
                        // futhark/microgpt.fut:312:137-160
                        
                        double neg_res_141551 = -neg_arg0_141550;
                        
                        // futhark/microgpt.fut:312:114-160
                        
                        double zp_res_141552 = zp_lhs_129400 + neg_res_141551;
                        
                        // futhark/microgpt.fut:312:107-160
                        
                        double exp_res_141553 = futrts_exp64(zp_res_141552);
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_141554 = ((double *) mem_144241)[i_129394 * (int64_t) 729 + i_129405 * (int64_t) 27 + i_142480];
                        
                        // futhark/microgpt.fut:312:107-192
                        
                        double zt_res_141555 = exp_res_141553 * zt_rhs_141554;
                        
                        zp_lhs_129407 = zt_res_141555;
                    } else {
                        zp_lhs_129407 = 0.0;
                    }
                    // futhark/microgpt.fut:312:210-383
                    
                    double zp_rhs_129414;
                    
                    if (cond_129396) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_141556 = ((double *) mem_144268)[i_129394 * (int64_t) 27 + i_129405];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double neg_arg0_141557 = ((double *) mem_144112)[i_129394 * (int64_t) 27 + i_129405];
                        
                        // futhark/microgpt.fut:312:320-343
                        
                        double neg_res_141558 = -neg_arg0_141557;
                        
                        // futhark/microgpt.fut:312:297-343
                        
                        double zp_res_141559 = zp_lhs_129402 + neg_res_141558;
                        
                        // futhark/microgpt.fut:312:290-343
                        
                        double neg_res_141560 = -zp_res_141559;
                        
                        // futhark/microgpt.fut:100:42-54
                        
                        double max_res_141561 = fmax64(0.0, neg_res_141560);
                        
                        // futhark/microgpt.fut:100:35-54
                        
                        double sgn_res_141562 = fsignum64(max_res_141561);
                        
                        // futhark/microgpt.fut:312:271-346
                        
                        double neg_res_141563 = -sgn_res_141562;
                        
                        // futhark/microgpt.fut:312:262-347
                        
                        double zp_res_141564 = 1.0 + neg_res_141563;
                        
                        // futhark/microgpt.fut:312:239-347
                        
                        double zt_res_141565 = zt_lhs_141556 * zp_res_141564;
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_141566 = ((double *) mem_144111)[i_129394 * (int64_t) 27 + i_129405];
                        
                        // futhark/microgpt.fut:312:257-372
                        
                        double zt_res_141567 = zt_res_141565 * zt_rhs_141566;
                        
                        zp_rhs_129414 = zt_res_141567;
                    } else {
                        zp_rhs_129414 = 0.0;
                    }
                    // futhark/microgpt.fut:312:78-383
                    
                    double zp_res_129427 = zp_lhs_129407 + zp_rhs_129414;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_129428 = r_129406 + zp_res_129427;
                    double r_tmp_145394 = zp_res_129428;
                    
                    r_129406 = r_tmp_145394;
                }
                defunc_0_lifted_lambda_res_129404 = r_129406;
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129429 = r_129395 + defunc_0_lifted_lambda_res_129404;
                double r_tmp_145393 = zp_res_129429;
                
                r_129395 = r_tmp_145393;
            }
            defunc_0_lifted_lambda_res_129393 = r_129395;
            ((double *) mem_144289)[i_142480] = defunc_0_lifted_lambda_res_129393;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144284, i_142484 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144289, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144300_cached_sizze_145821 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144300, &mem_144300_cached_sizze_145821, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144305_cached_sizze_145822 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144305, &mem_144305_cached_sizze_145822, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142492 = 0; i_142492 < (int64_t) 16; i_142492++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142488 = 0; i_142488 < (int64_t) 16; i_142488++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129436;
            double r_129438 = 0.0;
            
            for (int64_t i_129437 = 0; i_129437 < (int64_t) 27; i_129437++) {
                // futhark/microgpt.fut:313:67-176
                
                double x_129439 = ((double *) wvoc_mem_143129.mem)[i_129437 * (int64_t) 16 + i_142488];
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_129440;
                double r_129442 = 0.0;
                
                for (int64_t i_129441 = 0; i_129441 < (int64_t) 16; i_129441++) {
                    // futhark/microgpt.fut:313:90-148
                    
                    bool cond_129443 = i_142488 == i_129441;
                    
                    // futhark/microgpt.fut:313:90-148
                    
                    double zt_lhs_129444;
                    
                    if (cond_129443) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_t_res_141573 = ((double *) mem_144284)[i_142492 * (int64_t) 27 + i_129437];
                        
                        zt_lhs_129444 = zt_lhs_t_res_141573;
                    } else {
                        zt_lhs_129444 = 0.0;
                    }
                    // futhark/microgpt.fut:313:90-174
                    
                    double zt_res_129450 = x_129439 * zt_lhs_129444;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_129451 = r_129442 + zt_res_129450;
                    double r_tmp_145398 = zp_res_129451;
                    
                    r_129442 = r_tmp_145398;
                }
                defunc_0_lifted_lambda_res_129440 = r_129442;
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129452 = r_129438 + defunc_0_lifted_lambda_res_129440;
                double r_tmp_145397 = zp_res_129452;
                
                r_129438 = r_tmp_145397;
            }
            defunc_0_lifted_lambda_res_129436 = r_129438;
            ((double *) mem_144305)[i_142488] = defunc_0_lifted_lambda_res_129436;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144300, i_142492 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144305, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_144316, (int64_t) 8192, "mem_144316")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144317_cached_sizze_145823 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144317, &mem_144317_cached_sizze_145823, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144326_cached_sizze_145824 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144326, &mem_144326_cached_sizze_145824, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144327_cached_sizze_145825 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144327, &mem_144327_cached_sizze_145825, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142505 = 0; i_142505 < (int64_t) 16; i_142505++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142498 = 0; i_142498 < (int64_t) 64; i_142498++) {
            // futhark/microgpt.fut:4:11-25
            
            double indicatorp_arg0_139407 = ((double *) mem_144047)[i_142505 * (int64_t) 64 + i_142498];
            
            // futhark/microgpt.fut:100:42-54
            
            double max_res_139408 = fmax64(0.0, indicatorp_arg0_139407);
            
            // futhark/microgpt.fut:100:35-54
            
            double sgn_res_139409 = fsignum64(max_res_139408);
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_139410;
            double r_139412 = 0.0;
            
            for (int64_t i_139411 = 0; i_139411 < (int64_t) 16; i_139411++) {
                // futhark/microgpt.fut:314:105-216
                
                double x_139413 = ((double *) wdown_mem_143121.mem)[i_139411 * (int64_t) 64 + i_142498];
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_139414;
                double r_139416 = 0.0;
                
                for (int64_t i_139415 = 0; i_139415 < (int64_t) 64; i_139415++) {
                    // futhark/microgpt.fut:314:128-187
                    
                    bool cond_139417 = i_142498 == i_139415;
                    
                    // futhark/microgpt.fut:314:128-187
                    
                    double zt_lhs_139418;
                    
                    if (cond_139417) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_t_res_141579 = ((double *) mem_144300)[i_142505 * (int64_t) 16 + i_139411];
                        
                        zt_lhs_139418 = zt_lhs_t_res_141579;
                    } else {
                        zt_lhs_139418 = 0.0;
                    }
                    // futhark/microgpt.fut:314:128-214
                    
                    double zt_res_139424 = x_139413 * zt_lhs_139418;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_139425 = r_139416 + zt_res_139424;
                    double r_tmp_145404 = zp_res_139425;
                    
                    r_139416 = r_tmp_145404;
                }
                defunc_0_lifted_lambda_res_139414 = r_139416;
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_139426 = r_139412 + defunc_0_lifted_lambda_res_139414;
                double r_tmp_145403 = zp_res_139426;
                
                r_139412 = r_tmp_145403;
            }
            defunc_0_lifted_lambda_res_139410 = r_139412;
            // futhark/microgpt.fut:314:46-218
            
            double zt_res_139427 = sgn_res_139409 * defunc_0_lifted_lambda_res_139410;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_139430;
            double r_139432 = 0.0;
            
            for (int64_t i_139431 = 0; i_139431 < (int64_t) 16; i_139431++) {
                // futhark/microgpt.fut:396:69-178
                
                double x_139433 = ((double *) mem_144063)[i_139431 * (int64_t) 64 + i_142498];
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_139434;
                double r_139436 = 0.0;
                
                for (int64_t i_139435 = 0; i_139435 < (int64_t) 64; i_139435++) {
                    // futhark/microgpt.fut:396:92-151
                    
                    bool cond_139437 = i_142498 == i_139435;
                    
                    // futhark/microgpt.fut:396:92-151
                    
                    double zt_lhs_139438;
                    
                    if (cond_139437) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_t_res_141584 = ((double *) mem_144300)[i_139431 * (int64_t) 16 + i_142505];
                        
                        zt_lhs_139438 = zt_lhs_t_res_141584;
                    } else {
                        zt_lhs_139438 = 0.0;
                    }
                    // futhark/microgpt.fut:396:92-176
                    
                    double zt_res_139444 = x_139433 * zt_lhs_139438;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_139445 = r_139436 + zt_res_139444;
                    double r_tmp_145406 = zp_res_139445;
                    
                    r_139436 = r_tmp_145406;
                }
                defunc_0_lifted_lambda_res_139434 = r_139436;
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_139446 = r_139432 + defunc_0_lifted_lambda_res_139434;
                double r_tmp_145405 = zp_res_139446;
                
                r_139432 = r_tmp_145405;
            }
            defunc_0_lifted_lambda_res_139430 = r_139432;
            ((double *) mem_144326)[i_142498] = defunc_0_lifted_lambda_res_139430;
            ((double *) mem_144327)[i_142498] = zt_res_139427;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144316.mem, i_142505 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144326, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144317, i_142505 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144327, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144348_cached_sizze_145826 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144348, &mem_144348_cached_sizze_145826, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144353_cached_sizze_145827 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144353, &mem_144353_cached_sizze_145827, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142514 = 0; i_142514 < (int64_t) 16; i_142514++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142510 = 0; i_142510 < (int64_t) 16; i_142510++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129518;
            double r_129520 = 0.0;
            
            for (int64_t i_129519 = 0; i_129519 < (int64_t) 64; i_129519++) {
                // futhark/microgpt.fut:317:71-180
                
                double x_129521 = ((double *) wup_mem_143127.mem)[i_129519 * (int64_t) 16 + i_142510];
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_129522;
                double r_129524 = 0.0;
                
                for (int64_t i_129523 = 0; i_129523 < (int64_t) 16; i_129523++) {
                    // futhark/microgpt.fut:317:94-153
                    
                    bool cond_129525 = i_142510 == i_129523;
                    
                    // futhark/microgpt.fut:317:94-153
                    
                    double zt_lhs_129526;
                    
                    if (cond_129525) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_t_res_141591 = ((double *) mem_144317)[i_142514 * (int64_t) 64 + i_129519];
                        
                        zt_lhs_129526 = zt_lhs_t_res_141591;
                    } else {
                        zt_lhs_129526 = 0.0;
                    }
                    // futhark/microgpt.fut:317:94-178
                    
                    double zt_res_129532 = x_129521 * zt_lhs_129526;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_129533 = r_129524 + zt_res_129532;
                    double r_tmp_145410 = zp_res_129533;
                    
                    r_129524 = r_tmp_145410;
                }
                defunc_0_lifted_lambda_res_129522 = r_129524;
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129534 = r_129520 + defunc_0_lifted_lambda_res_129522;
                double r_tmp_145409 = zp_res_129534;
                
                r_129520 = r_tmp_145409;
            }
            defunc_0_lifted_lambda_res_129518 = r_129520;
            ((double *) mem_144353)[i_142510] = defunc_0_lifted_lambda_res_129518;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144348, i_142514 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144353, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144364_cached_sizze_145828 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144364, &mem_144364_cached_sizze_145828, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144365_cached_sizze_145829 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144365, &mem_144365_cached_sizze_145829, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142520 = 0; i_142520 < (int64_t) 16; i_142520++) {
        // futhark/microgpt.fut:316:47-59
        
        double zp_lhs_131747 = ((double *) mem_143994)[i_142520];
        
        // futhark/microgpt.fut:316:47-87
        
        double zp_res_131748 = 1.0e-5 + zp_lhs_131747;
        
        // futhark/microgpt.fut:316:39-87
        
        double sqrt_res_131749 = futrts_sqrt64(zp_res_131748);
        
        // futhark/microgpt.fut:318:129-158
        
        double zt_res_131757 = sqrt_res_131749 * sqrt_res_131749;
        
        // futhark/microgpt.fut:318:120-158
        
        double zs_res_131758 = 1.0 / zt_res_131757;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_131759;
        double r_131761 = 0.0;
        
        for (int64_t i_131760 = 0; i_131760 < (int64_t) 16; i_131760++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_131762 = ((double *) mem_144348)[i_142520 * (int64_t) 16 + i_131760];
            
            // futhark/microgpt.fut:61:46-49
            
            double zt_rhs_131763 = ((double *) mem_143978)[i_142520 * (int64_t) 16 + i_131760];
            
            // futhark/microgpt.fut:318:69-113
            
            double zt_res_131764 = zt_lhs_131762 * zt_rhs_131763;
            
            // futhark/microgpt.fut:318:90-158
            
            double zt_res_131765 = zs_res_131758 * zt_res_131764;
            
            // futhark/microgpt.fut:318:61-158
            
            double neg_res_131766 = -zt_res_131765;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_131767 = r_131761 + neg_res_131766;
            double r_tmp_145413 = zp_res_131767;
            
            r_131761 = r_tmp_145413;
        }
        defunc_0_lifted_lambda_res_131759 = r_131761;
        ((double *) mem_144364)[i_142520] = defunc_0_lifted_lambda_res_131759;
        ((double *) mem_144365)[i_142520] = sqrt_res_131749;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144378_cached_sizze_145830 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144378, &mem_144378_cached_sizze_145830, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142525 = 0; i_142525 < (int64_t) 16; i_142525++) {
        // futhark/microgpt.fut:319:39-51
        
        double zt_lhs_129562 = ((double *) mem_144364)[i_142525];
        
        // futhark/microgpt.fut:319:93-105
        
        double zp_lhs_129563 = ((double *) mem_143994)[i_142525];
        
        // futhark/microgpt.fut:319:93-133
        
        double zp_res_129564 = 1.0e-5 + zp_lhs_129563;
        
        // futhark/microgpt.fut:319:85-133
        
        double sqrt_res_129565 = futrts_sqrt64(zp_res_129564);
        
        // futhark/microgpt.fut:319:71-135
        
        double zt_res_129566 = 2.0 * sqrt_res_129565;
        
        // futhark/microgpt.fut:319:57-135
        
        double zs_res_129567 = 1.0 / zt_res_129566;
        
        // futhark/microgpt.fut:319:39-135
        
        double zt_res_129568 = zt_lhs_129562 * zs_res_129567;
        
        ((double *) mem_144378)[i_142525] = zt_res_129568;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144385_cached_sizze_145831 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144385, &mem_144385_cached_sizze_145831, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144390_cached_sizze_145832 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144390, &mem_144390_cached_sizze_145832, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142533 = 0; i_142533 < (int64_t) 16; i_142533++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142529 = 0; i_142529 < (int64_t) 16; i_142529++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_129582 = ((double *) mem_144300)[i_142533 * (int64_t) 16 + i_142529];
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129583;
            double r_129585 = 0.0;
            
            for (int64_t i_129584 = 0; i_129584 < (int64_t) 16; i_129584++) {
                // futhark/microgpt.fut:320:86-174
                
                bool cond_129586 = i_142533 == i_129584;
                
                // futhark/microgpt.fut:320:86-174
                
                double zp_lhs_129587;
                
                if (cond_129586) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_141593 = ((double *) mem_144348)[i_129584 * (int64_t) 16 + i_142529];
                    
                    // futhark/microgpt.fut:320:150-162
                    
                    double zs_rhs_141594 = ((double *) mem_144365)[i_129584];
                    
                    // futhark/microgpt.fut:320:142-162
                    
                    double zs_res_141595 = 1.0 / zs_rhs_141594;
                    
                    // futhark/microgpt.fut:320:116-162
                    
                    double zt_res_141596 = zt_lhs_141593 * zs_res_141595;
                    
                    zp_lhs_129587 = zt_res_141596;
                } else {
                    zp_lhs_129587 = 0.0;
                }
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_129595;
                
                if (cond_129586) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double x_141597 = ((double *) mem_143978)[i_129584 * (int64_t) 16 + i_142529];
                    
                    zt_rhs_129595 = x_141597;
                } else {
                    zt_rhs_129595 = 0.0;
                }
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_129597;
                
                if (cond_129586) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double x_141598 = ((double *) mem_143978)[i_129584 * (int64_t) 16 + i_142529];
                    
                    zt_rhs_129597 = x_141598;
                } else {
                    zt_rhs_129597 = 0.0;
                }
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_129599;
                double r_129601 = 0.0;
                
                for (int64_t i_129600 = 0; i_129600 < (int64_t) 16; i_129600++) {
                    // futhark/microgpt.fut:320:204-339
                    
                    double zp_lhs_129602;
                    
                    if (cond_129586) {
                        // futhark/microgpt.fut:320:235-303
                        
                        bool cond_141601 = i_142529 == i_129600;
                        
                        // futhark/microgpt.fut:320:235-303
                        
                        double zt_lhs_141602;
                        
                        if (cond_141601) {
                            // futhark/microgpt.fut:320:265-277
                            
                            double zs_lhs_141603 = ((double *) mem_144378)[i_129584];
                            
                            // futhark/microgpt.fut:320:265-292
                            
                            double zs_res_141604 = zs_lhs_141603 / 16.0;
                            
                            zt_lhs_141602 = zs_res_141604;
                        } else {
                            zt_lhs_141602 = 0.0;
                        }
                        // futhark/microgpt.fut:320:235-328
                        
                        double zt_res_141605 = zt_rhs_129595 * zt_lhs_141602;
                        
                        zp_lhs_129602 = zt_res_141605;
                    } else {
                        zp_lhs_129602 = 0.0;
                    }
                    // futhark/microgpt.fut:320:346-481
                    
                    double zp_rhs_129608;
                    
                    if (cond_129586) {
                        // futhark/microgpt.fut:320:377-445
                        
                        bool cond_141608 = i_142529 == i_129600;
                        
                        // futhark/microgpt.fut:320:377-445
                        
                        double zt_lhs_141609;
                        
                        if (cond_141608) {
                            // futhark/microgpt.fut:320:407-419
                            
                            double zs_lhs_141610 = ((double *) mem_144378)[i_129584];
                            
                            // futhark/microgpt.fut:320:407-434
                            
                            double zs_res_141611 = zs_lhs_141610 / 16.0;
                            
                            zt_lhs_141609 = zs_res_141611;
                        } else {
                            zt_lhs_141609 = 0.0;
                        }
                        // futhark/microgpt.fut:320:377-470
                        
                        double zt_res_141612 = zt_rhs_129597 * zt_lhs_141609;
                        
                        zp_rhs_129608 = zt_res_141612;
                    } else {
                        zp_rhs_129608 = 0.0;
                    }
                    // futhark/microgpt.fut:320:204-481
                    
                    double zp_res_129614 = zp_lhs_129602 + zp_rhs_129608;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_129615 = r_129601 + zp_res_129614;
                    double r_tmp_145418 = zp_res_129615;
                    
                    r_129601 = r_tmp_145418;
                }
                defunc_0_lifted_lambda_res_129599 = r_129601;
                // futhark/microgpt.fut:320:86-484
                
                double zp_res_129616 = zp_lhs_129587 + defunc_0_lifted_lambda_res_129599;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129617 = r_129585 + zp_res_129616;
                double r_tmp_145417 = zp_res_129617;
                
                r_129585 = r_tmp_145417;
            }
            defunc_0_lifted_lambda_res_129583 = r_129585;
            // futhark/microgpt.fut:320:37-487
            
            double zp_res_129618 = zp_lhs_129582 + defunc_0_lifted_lambda_res_129583;
            
            ((double *) mem_144390)[i_142529] = zp_res_129618;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144385, i_142533 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144390, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144401_cached_sizze_145833 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144401, &mem_144401_cached_sizze_145833, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144407_cached_sizze_145834 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144407, &mem_144407_cached_sizze_145834, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144412_cached_sizze_145835 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144412, &mem_144412_cached_sizze_145835, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142545 = 0; i_142545 < (int64_t) 4; i_142545++) {
        // futhark/microgpt.fut:321:112-115
        
        int64_t zp_lhs_129623 = mul64((int64_t) 4, i_142545);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142541 = 0; i_142541 < (int64_t) 16; i_142541++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142537 = 0; i_142537 < (int64_t) 4; i_142537++) {
                // futhark/microgpt.fut:321:117-125
                
                int64_t zeze_lhs_129628 = add64(zp_lhs_129623, i_142537);
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_129629;
                double r_129631 = 0.0;
                
                for (int64_t i_129630 = 0; i_129630 < (int64_t) 16; i_129630++) {
                    // futhark/microgpt.fut:321:75-219
                    
                    double x_129632 = ((double *) wout_mem_143123.mem)[i_129630 * (int64_t) 16 + zeze_lhs_129628];
                    
                    // futhark/microgpt.fut:61:13-49
                    
                    double defunc_0_lifted_lambda_res_129633;
                    double r_129635 = 0.0;
                    
                    for (int64_t i_129634 = 0; i_129634 < (int64_t) 16; i_129634++) {
                        // futhark/microgpt.fut:321:98-174
                        
                        bool cond_129636 = zeze_lhs_129628 == i_129634;
                        
                        // futhark/microgpt.fut:321:98-174
                        
                        double zt_lhs_129637;
                        
                        if (cond_129636) {
                            // futhark/microgpt.fut:61:46-49
                            
                            double zt_lhs_t_res_141618 = ((double *) mem_144385)[i_142541 * (int64_t) 16 + i_129630];
                            
                            zt_lhs_129637 = zt_lhs_t_res_141618;
                        } else {
                            zt_lhs_129637 = 0.0;
                        }
                        // futhark/microgpt.fut:321:98-217
                        
                        double zt_res_129643 = x_129632 * zt_lhs_129637;
                        
                        // futhark/microgpt.fut:61:40-49
                        
                        double zp_res_129644 = r_129635 + zt_res_129643;
                        double r_tmp_145423 = zp_res_129644;
                        
                        r_129635 = r_tmp_145423;
                    }
                    defunc_0_lifted_lambda_res_129633 = r_129635;
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_129645 = r_129631 + defunc_0_lifted_lambda_res_129633;
                    double r_tmp_145422 = zp_res_129645;
                    
                    r_129631 = r_tmp_145422;
                }
                defunc_0_lifted_lambda_res_129629 = r_129631;
                ((double *) mem_144412)[i_142537] = defunc_0_lifted_lambda_res_129629;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144407, i_142541 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144412, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144401, i_142545 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144407, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144428_cached_sizze_145836 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144428, &mem_144428_cached_sizze_145836, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144429_cached_sizze_145837 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144429, &mem_144429_cached_sizze_145837, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144430_cached_sizze_145838 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144430, &mem_144430_cached_sizze_145838, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144446_cached_sizze_145839 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144446, &mem_144446_cached_sizze_145839, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144447_cached_sizze_145840 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144447, &mem_144447_cached_sizze_145840, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144448_cached_sizze_145841 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144448, &mem_144448_cached_sizze_145841, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144461_cached_sizze_145842 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144461, &mem_144461_cached_sizze_145842, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144462_cached_sizze_145843 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144462, &mem_144462_cached_sizze_145843, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142570 = 0; i_142570 < (int64_t) 4; i_142570++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142560 = 0; i_142560 < (int64_t) 16; i_142560++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142551 = 0; i_142551 < (int64_t) 4; i_142551++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_139653 = ((double *) mem_144401)[i_142570 * (int64_t) 64 + i_142560 * (int64_t) 4 + i_142551];
                
                ((double *) mem_144461)[i_142551] = lifted_lambda_res_139653;
                ((double *) mem_144462)[i_142551] = lifted_lambda_res_139653;
            }
            // futhark/microgpt.fut:4:11-25
            // futhark/microgpt.fut:4:11-25
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144448, i_142560 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144462, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144446, i_142560 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144461, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144447, i_142560 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144462, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144428, i_142570 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144446, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144429, i_142570 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144447, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144430, i_142570 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144448, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144503_cached_sizze_145844 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144503, &mem_144503_cached_sizze_145844, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144504_cached_sizze_145845 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144504, &mem_144504_cached_sizze_145845, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144515_cached_sizze_145846 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144515, &mem_144515_cached_sizze_145846, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144516_cached_sizze_145847 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144516, &mem_144516_cached_sizze_145847, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144525_cached_sizze_145848 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144525, &mem_144525_cached_sizze_145848, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144526_cached_sizze_145849 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144526, &mem_144526_cached_sizze_145849, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142592 = 0; i_142592 < (int64_t) 4; i_142592++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142585 = 0; i_142585 < (int64_t) 16; i_142585++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142578 = 0; i_142578 < (int64_t) 16; i_142578++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_140053;
                double r_140055 = 0.0;
                
                for (int64_t i_140054 = 0; i_140054 < (int64_t) 4; i_140054++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_140056 = ((double *) mem_144429)[i_142592 * (int64_t) 64 + i_142585 * (int64_t) 4 + i_140054];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_140057 = ((double *) mem_143262)[i_142592 * (int64_t) 64 + i_142578 * (int64_t) 4 + i_140054];
                    
                    // futhark/microgpt.fut:334:79-139
                    
                    double zt_res_140058 = zt_lhs_140056 * zt_rhs_140057;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_140059 = r_140055 + zt_res_140058;
                    double r_tmp_145438 = zp_res_140059;
                    
                    r_140055 = r_tmp_145438;
                }
                defunc_0_lifted_lambda_res_140053 = r_140055;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_140066;
                double r_140068 = 0.0;
                
                for (int64_t i_140067 = 0; i_140067 < (int64_t) 4; i_140067++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_140069 = ((double *) mem_144428)[i_142592 * (int64_t) 64 + i_142585 * (int64_t) 4 + i_140067];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_140070 = ((double *) mem_143262)[i_142592 * (int64_t) 64 + i_142578 * (int64_t) 4 + i_140067];
                    
                    // futhark/microgpt.fut:350:79-139
                    
                    double zt_res_140071 = zt_lhs_140069 * zt_rhs_140070;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_140072 = r_140068 + zt_res_140071;
                    double r_tmp_145439 = zp_res_140072;
                    
                    r_140068 = r_tmp_145439;
                }
                defunc_0_lifted_lambda_res_140066 = r_140068;
                ((double *) mem_144525)[i_142578] = defunc_0_lifted_lambda_res_140066;
                ((double *) mem_144526)[i_142578] = defunc_0_lifted_lambda_res_140053;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144515, i_142585 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144525, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144516, i_142585 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144526, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144503, i_142592 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_144515, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144504, i_142592 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_144516, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144557_cached_sizze_145850 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144557, &mem_144557_cached_sizze_145850, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144558_cached_sizze_145851 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144558, &mem_144558_cached_sizze_145851, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144569_cached_sizze_145852 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144569, &mem_144569_cached_sizze_145852, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144570_cached_sizze_145853 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144570, &mem_144570_cached_sizze_145853, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144579_cached_sizze_145854 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144579, &mem_144579_cached_sizze_145854, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144580_cached_sizze_145855 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144580, &mem_144580_cached_sizze_145855, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142613 = 0; i_142613 < (int64_t) 4; i_142613++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142606 = 0; i_142606 < (int64_t) 16; i_142606++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142599 = 0; i_142599 < (int64_t) 16; i_142599++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_140305 = ((double *) mem_144504)[i_142613 * (int64_t) 256 + i_142606 * (int64_t) 16 + i_142599];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_140312 = ((double *) mem_144503)[i_142613 * (int64_t) 256 + i_142606 * (int64_t) 16 + i_142599];
                
                ((double *) mem_144579)[i_142599] = lifted_lambda_res_140312;
                ((double *) mem_144580)[i_142599] = lifted_lambda_res_140305;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144569, i_142606 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144579, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144570, i_142606 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144580, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144557, i_142613 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_144569, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144558, i_142613 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_144570, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144611_cached_sizze_145856 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144611, &mem_144611_cached_sizze_145856, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144612_cached_sizze_145857 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144612, &mem_144612_cached_sizze_145857, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144613_cached_sizze_145858 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144613, &mem_144613_cached_sizze_145858, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144614_cached_sizze_145859 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144614, &mem_144614_cached_sizze_145859, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144631_cached_sizze_145860 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144631, &mem_144631_cached_sizze_145860, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144632_cached_sizze_145861 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144632, &mem_144632_cached_sizze_145861, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144633_cached_sizze_145862 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144633, &mem_144633_cached_sizze_145862, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144634_cached_sizze_145863 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144634, &mem_144634_cached_sizze_145863, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142637 = 0; i_142637 < (int64_t) 4; i_142637++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142624 = 0; i_142624 < (int64_t) 16; i_142624++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_140185;
            double r_140187 = 0.0;
            
            for (int64_t i_140186 = 0; i_140186 < (int64_t) 16; i_140186++) {
                // futhark/microgpt.fut:61:46-49
                
                double lifted_lambda_res_140188 = ((double *) mem_143688)[i_142637 * (int64_t) 256 + i_142624 * (int64_t) 16 + i_140186];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_140189 = r_140187 + lifted_lambda_res_140188;
                double r_tmp_145454 = zp_res_140189;
                
                r_140187 = r_tmp_145454;
            }
            defunc_0_lifted_lambda_res_140185 = r_140187;
            // futhark/microgpt.fut:339:155-200
            
            double zt_res_140197 = defunc_0_lifted_lambda_res_140185 * defunc_0_lifted_lambda_res_140185;
            
            // futhark/microgpt.fut:339:146-200
            
            double zs_res_140198 = 1.0 / zt_res_140197;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_140199;
            double r_140201 = 0.0;
            
            for (int64_t i_140200 = 0; i_140200 < (int64_t) 16; i_140200++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_140202 = ((double *) mem_144558)[i_142637 * (int64_t) 256 + i_142624 * (int64_t) 16 + i_140200];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_140203 = ((double *) mem_143688)[i_142637 * (int64_t) 256 + i_142624 * (int64_t) 16 + i_140200];
                
                // futhark/microgpt.fut:339:78-139
                
                double zt_res_140204 = zt_lhs_140202 * zt_rhs_140203;
                
                // futhark/microgpt.fut:339:107-200
                
                double zt_res_140205 = zs_res_140198 * zt_res_140204;
                
                // futhark/microgpt.fut:339:70-200
                
                double neg_res_140206 = -zt_res_140205;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_140207 = r_140201 + neg_res_140206;
                double r_tmp_145455 = zp_res_140207;
                
                r_140201 = r_tmp_145455;
            }
            defunc_0_lifted_lambda_res_140199 = r_140201;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_140218;
            double r_140220 = 0.0;
            
            for (int64_t i_140219 = 0; i_140219 < (int64_t) 16; i_140219++) {
                // futhark/microgpt.fut:61:46-49
                
                double lifted_lambda_res_140221 = ((double *) mem_143687)[i_142637 * (int64_t) 256 + i_142624 * (int64_t) 16 + i_140219];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_140222 = r_140220 + lifted_lambda_res_140221;
                double r_tmp_145456 = zp_res_140222;
                
                r_140220 = r_tmp_145456;
            }
            defunc_0_lifted_lambda_res_140218 = r_140220;
            // futhark/microgpt.fut:355:155-200
            
            double zt_res_140230 = defunc_0_lifted_lambda_res_140218 * defunc_0_lifted_lambda_res_140218;
            
            // futhark/microgpt.fut:355:146-200
            
            double zs_res_140231 = 1.0 / zt_res_140230;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_140232;
            double r_140234 = 0.0;
            
            for (int64_t i_140233 = 0; i_140233 < (int64_t) 16; i_140233++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_140235 = ((double *) mem_144557)[i_142637 * (int64_t) 256 + i_142624 * (int64_t) 16 + i_140233];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_140236 = ((double *) mem_143687)[i_142637 * (int64_t) 256 + i_142624 * (int64_t) 16 + i_140233];
                
                // futhark/microgpt.fut:355:78-139
                
                double zt_res_140237 = zt_lhs_140235 * zt_rhs_140236;
                
                // futhark/microgpt.fut:355:107-200
                
                double zt_res_140238 = zs_res_140231 * zt_res_140237;
                
                // futhark/microgpt.fut:355:70-200
                
                double neg_res_140239 = -zt_res_140238;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_140240 = r_140234 + neg_res_140239;
                double r_tmp_145457 = zp_res_140240;
                
                r_140234 = r_tmp_145457;
            }
            defunc_0_lifted_lambda_res_140232 = r_140234;
            ((double *) mem_144631)[i_142624] = defunc_0_lifted_lambda_res_140232;
            ((double *) mem_144632)[i_142624] = defunc_0_lifted_lambda_res_140218;
            ((double *) mem_144633)[i_142624] = defunc_0_lifted_lambda_res_140199;
            ((double *) mem_144634)[i_142624] = defunc_0_lifted_lambda_res_140185;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144611, i_142637 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144631, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144612, i_142637 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144632, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144613, i_142637 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144633, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144614, i_142637 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144634, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144675_cached_sizze_145864 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144675, &mem_144675_cached_sizze_145864, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144676_cached_sizze_145865 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144676, &mem_144676_cached_sizze_145865, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144687_cached_sizze_145866 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144687, &mem_144687_cached_sizze_145866, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144688_cached_sizze_145867 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144688, &mem_144688_cached_sizze_145867, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144697_cached_sizze_145868 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144697, &mem_144697_cached_sizze_145868, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144698_cached_sizze_145869 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144698, &mem_144698_cached_sizze_145869, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142660 = 0; i_142660 < (int64_t) 4; i_142660++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142653 = 0; i_142653 < (int64_t) 16; i_142653++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_rhs_140336 = ((double *) mem_144614)[i_142660 * (int64_t) 16 + i_142653];
            
            // futhark/microgpt.fut:340:93-121
            
            double zs_res_140337 = 1.0 / zs_rhs_140336;
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_140338 = ((double *) mem_144613)[i_142660 * (int64_t) 16 + i_142653];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_140357 = ((double *) mem_144611)[i_142660 * (int64_t) 16 + i_142653];
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_rhs_140355 = ((double *) mem_144612)[i_142660 * (int64_t) 16 + i_142653];
            
            // futhark/microgpt.fut:356:93-121
            
            double zs_res_140356 = 1.0 / zs_rhs_140355;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142646 = 0; i_142646 < (int64_t) 16; i_142646++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_140385 = ((double *) mem_144558)[i_142660 * (int64_t) 256 + i_142653 * (int64_t) 16 + i_142646];
                
                // futhark/microgpt.fut:340:59-121
                
                double zt_res_140386 = zs_res_140337 * zt_lhs_140385;
                
                // futhark/microgpt.fut:340:88-148
                
                double zp_res_140387 = zp_rhs_140338 + zt_res_140386;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_140394 = ((double *) mem_144557)[i_142660 * (int64_t) 256 + i_142653 * (int64_t) 16 + i_142646];
                
                // futhark/microgpt.fut:356:59-121
                
                double zt_res_140395 = zs_res_140356 * zt_lhs_140394;
                
                // futhark/microgpt.fut:356:88-148
                
                double zp_res_140396 = zp_rhs_140357 + zt_res_140395;
                
                ((double *) mem_144697)[i_142646] = zp_res_140396;
                ((double *) mem_144698)[i_142646] = zp_res_140387;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144687, i_142653 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144697, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144688, i_142653 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144698, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144675, i_142660 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_144687, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144676, i_142660 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_144688, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144729_cached_sizze_145870 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144729, &mem_144729_cached_sizze_145870, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144730_cached_sizze_145871 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144730, &mem_144730_cached_sizze_145871, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144739_cached_sizze_145872 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144739, &mem_144739_cached_sizze_145872, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144740_cached_sizze_145873 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144740, &mem_144740_cached_sizze_145873, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142674 = 0; i_142674 < (int64_t) 4; i_142674++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142667 = 0; i_142667 < (int64_t) 16; i_142667++) {
            double f_elem_140416 = ((double *) mem_143564)[i_142674 * (int64_t) 16 + i_142667];
            double f_elem_140418 = ((double *) mem_143561)[i_142674 * (int64_t) 16 + i_142667];
            
            // futhark/microgpt.fut:341:119-145
            
            double neg_res_140423 = -f_elem_140416;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_140424;
            double r_140426 = 0.0;
            
            for (int64_t i_140425 = 0; i_140425 < (int64_t) 16; i_140425++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_140427 = ((double *) mem_143452)[i_142674 * (int64_t) 256 + i_142667 * (int64_t) 16 + i_140425];
                
                // futhark/microgpt.fut:341:85-145
                
                double zp_res_140428 = neg_res_140423 + zp_lhs_140427;
                
                // futhark/microgpt.fut:341:78-145
                
                double exp_res_140429 = futrts_exp64(zp_res_140428);
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_140430 = ((double *) mem_144676)[i_142674 * (int64_t) 256 + i_142667 * (int64_t) 16 + i_140425];
                
                // futhark/microgpt.fut:341:78-181
                
                double zt_res_140431 = exp_res_140429 * zt_rhs_140430;
                
                // futhark/microgpt.fut:341:70-181
                
                double neg_res_140432 = -zt_res_140431;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_140433 = r_140426 + neg_res_140432;
                double r_tmp_145468 = zp_res_140433;
                
                r_140426 = r_tmp_145468;
            }
            defunc_0_lifted_lambda_res_140424 = r_140426;
            // futhark/microgpt.fut:357:119-145
            
            double neg_res_140441 = -f_elem_140418;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_140442;
            double r_140444 = 0.0;
            
            for (int64_t i_140443 = 0; i_140443 < (int64_t) 16; i_140443++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_140445 = ((double *) mem_143451)[i_142674 * (int64_t) 256 + i_142667 * (int64_t) 16 + i_140443];
                
                // futhark/microgpt.fut:357:85-145
                
                double zp_res_140446 = neg_res_140441 + zp_lhs_140445;
                
                // futhark/microgpt.fut:357:78-145
                
                double exp_res_140447 = futrts_exp64(zp_res_140446);
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_140448 = ((double *) mem_144675)[i_142674 * (int64_t) 256 + i_142667 * (int64_t) 16 + i_140443];
                
                // futhark/microgpt.fut:357:78-181
                
                double zt_res_140449 = exp_res_140447 * zt_rhs_140448;
                
                // futhark/microgpt.fut:357:70-181
                
                double neg_res_140450 = -zt_res_140449;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_140451 = r_140444 + neg_res_140450;
                double r_tmp_145469 = zp_res_140451;
                
                r_140444 = r_tmp_145469;
            }
            defunc_0_lifted_lambda_res_140442 = r_140444;
            ((double *) mem_144739)[i_142667] = defunc_0_lifted_lambda_res_140442;
            ((double *) mem_144740)[i_142667] = defunc_0_lifted_lambda_res_140424;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144729, i_142674 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144739, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144730, i_142674 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144740, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144761_cached_sizze_145874 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144761, &mem_144761_cached_sizze_145874, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144762_cached_sizze_145875 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144762, &mem_144762_cached_sizze_145875, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144773_cached_sizze_145876 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144773, &mem_144773_cached_sizze_145876, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144774_cached_sizze_145877 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144774, &mem_144774_cached_sizze_145877, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144783_cached_sizze_145878 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144783, &mem_144783_cached_sizze_145878, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144784_cached_sizze_145879 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144784, &mem_144784_cached_sizze_145879, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142695 = 0; i_142695 < (int64_t) 4; i_142695++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142688 = 0; i_142688 < (int64_t) 16; i_142688++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142681 = 0; i_142681 < (int64_t) 16; i_142681++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_140576;
                double r_140578 = 0.0;
                
                for (int64_t i_140577 = 0; i_140577 < (int64_t) 16; i_140577++) {
                    // futhark/microgpt.fut:344:81-226
                    
                    bool cond_140579 = i_142688 == i_140577;
                    
                    // futhark/microgpt.fut:344:81-226
                    
                    double zp_lhs_140580;
                    
                    if (cond_140579) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zp_lhs_141674 = ((double *) mem_143452)[i_142695 * (int64_t) 256 + i_140577 * (int64_t) 16 + i_142681];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double neg_arg0_141675 = ((double *) mem_143564)[i_142695 * (int64_t) 16 + i_140577];
                        
                        // futhark/microgpt.fut:344:153-179
                        
                        double neg_res_141676 = -neg_arg0_141675;
                        
                        // futhark/microgpt.fut:344:119-179
                        
                        double zp_res_141677 = zp_lhs_141674 + neg_res_141676;
                        
                        // futhark/microgpt.fut:344:112-179
                        
                        double exp_res_141678 = futrts_exp64(zp_res_141677);
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_141679 = ((double *) mem_144676)[i_142695 * (int64_t) 256 + i_140577 * (int64_t) 16 + i_142681];
                        
                        // futhark/microgpt.fut:344:112-215
                        
                        double zt_res_141680 = exp_res_141678 * zt_rhs_141679;
                        
                        zp_lhs_140580 = zt_res_141680;
                    } else {
                        zp_lhs_140580 = 0.0;
                    }
                    // futhark/microgpt.fut:344:233-428
                    
                    double zp_rhs_140596;
                    
                    if (cond_140579) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_141685 = ((double *) mem_144730)[i_142695 * (int64_t) 16 + i_140577];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zp_lhs_141690 = ((double *) mem_143452)[i_142695 * (int64_t) 256 + i_140577 * (int64_t) 16 + i_142681];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double neg_arg0_141691 = ((double *) mem_143563)[i_142695 * (int64_t) 16 + i_140577];
                        
                        // futhark/microgpt.fut:344:359-385
                        
                        double neg_res_141692 = -neg_arg0_141691;
                        
                        // futhark/microgpt.fut:344:325-385
                        
                        double zp_res_141693 = zp_lhs_141690 + neg_res_141692;
                        
                        // futhark/microgpt.fut:344:318-385
                        
                        double neg_res_141694 = -zp_res_141693;
                        
                        // futhark/microgpt.fut:100:42-54
                        
                        double max_res_141695 = fmax64(0.0, neg_res_141694);
                        
                        // futhark/microgpt.fut:100:35-54
                        
                        double sgn_res_141696 = fsignum64(max_res_141695);
                        
                        // futhark/microgpt.fut:344:299-388
                        
                        double neg_res_141697 = -sgn_res_141696;
                        
                        // futhark/microgpt.fut:344:290-389
                        
                        double zp_res_141698 = 1.0 + neg_res_141697;
                        
                        // futhark/microgpt.fut:344:264-389
                        
                        double zt_res_141699 = zt_lhs_141685 * zp_res_141698;
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_141700 = ((double *) mem_143562)[i_142695 * (int64_t) 16 + i_140577];
                        
                        // futhark/microgpt.fut:344:285-417
                        
                        double zt_res_141701 = zt_res_141699 * zt_rhs_141700;
                        
                        zp_rhs_140596 = zt_res_141701;
                    } else {
                        zp_rhs_140596 = 0.0;
                    }
                    // futhark/microgpt.fut:344:81-428
                    
                    double zp_res_140618 = zp_lhs_140580 + zp_rhs_140596;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_140619 = r_140578 + zp_res_140618;
                    double r_tmp_145476 = zp_res_140619;
                    
                    r_140578 = r_tmp_145476;
                }
                defunc_0_lifted_lambda_res_140576 = r_140578;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_140622;
                double r_140624 = 0.0;
                
                for (int64_t i_140623 = 0; i_140623 < (int64_t) 16; i_140623++) {
                    // futhark/microgpt.fut:360:81-226
                    
                    bool cond_140625 = i_142688 == i_140623;
                    
                    // futhark/microgpt.fut:360:81-226
                    
                    double zp_lhs_140626;
                    
                    if (cond_140625) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zp_lhs_141710 = ((double *) mem_143451)[i_142695 * (int64_t) 256 + i_140623 * (int64_t) 16 + i_142681];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double neg_arg0_141711 = ((double *) mem_143561)[i_142695 * (int64_t) 16 + i_140623];
                        
                        // futhark/microgpt.fut:360:153-179
                        
                        double neg_res_141712 = -neg_arg0_141711;
                        
                        // futhark/microgpt.fut:360:119-179
                        
                        double zp_res_141713 = zp_lhs_141710 + neg_res_141712;
                        
                        // futhark/microgpt.fut:360:112-179
                        
                        double exp_res_141714 = futrts_exp64(zp_res_141713);
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_141715 = ((double *) mem_144675)[i_142695 * (int64_t) 256 + i_140623 * (int64_t) 16 + i_142681];
                        
                        // futhark/microgpt.fut:360:112-215
                        
                        double zt_res_141716 = exp_res_141714 * zt_rhs_141715;
                        
                        zp_lhs_140626 = zt_res_141716;
                    } else {
                        zp_lhs_140626 = 0.0;
                    }
                    // futhark/microgpt.fut:360:233-428
                    
                    double zp_rhs_140642;
                    
                    if (cond_140625) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_141721 = ((double *) mem_144729)[i_142695 * (int64_t) 16 + i_140623];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zp_lhs_141726 = ((double *) mem_143451)[i_142695 * (int64_t) 256 + i_140623 * (int64_t) 16 + i_142681];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double neg_arg0_141727 = ((double *) mem_143560)[i_142695 * (int64_t) 16 + i_140623];
                        
                        // futhark/microgpt.fut:360:359-385
                        
                        double neg_res_141728 = -neg_arg0_141727;
                        
                        // futhark/microgpt.fut:360:325-385
                        
                        double zp_res_141729 = zp_lhs_141726 + neg_res_141728;
                        
                        // futhark/microgpt.fut:360:318-385
                        
                        double neg_res_141730 = -zp_res_141729;
                        
                        // futhark/microgpt.fut:100:42-54
                        
                        double max_res_141731 = fmax64(0.0, neg_res_141730);
                        
                        // futhark/microgpt.fut:100:35-54
                        
                        double sgn_res_141732 = fsignum64(max_res_141731);
                        
                        // futhark/microgpt.fut:360:299-388
                        
                        double neg_res_141733 = -sgn_res_141732;
                        
                        // futhark/microgpt.fut:360:290-389
                        
                        double zp_res_141734 = 1.0 + neg_res_141733;
                        
                        // futhark/microgpt.fut:360:264-389
                        
                        double zt_res_141735 = zt_lhs_141721 * zp_res_141734;
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_141736 = ((double *) mem_143559)[i_142695 * (int64_t) 16 + i_140623];
                        
                        // futhark/microgpt.fut:360:285-417
                        
                        double zt_res_141737 = zt_res_141735 * zt_rhs_141736;
                        
                        zp_rhs_140642 = zt_res_141737;
                    } else {
                        zp_rhs_140642 = 0.0;
                    }
                    // futhark/microgpt.fut:360:81-428
                    
                    double zp_res_140664 = zp_lhs_140626 + zp_rhs_140642;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_140665 = r_140624 + zp_res_140664;
                    double r_tmp_145477 = zp_res_140665;
                    
                    r_140624 = r_tmp_145477;
                }
                defunc_0_lifted_lambda_res_140622 = r_140624;
                ((double *) mem_144783)[i_142681] = defunc_0_lifted_lambda_res_140622;
                ((double *) mem_144784)[i_142681] = defunc_0_lifted_lambda_res_140576;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144773, i_142688 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144783, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144774, i_142688 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144784, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144761, i_142695 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_144773, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144762, i_142695 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_144774, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144815_cached_sizze_145880 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144815, &mem_144815_cached_sizze_145880, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144816_cached_sizze_145881 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144816, &mem_144816_cached_sizze_145881, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144827_cached_sizze_145882 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144827, &mem_144827_cached_sizze_145882, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144828_cached_sizze_145883 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144828, &mem_144828_cached_sizze_145883, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144837_cached_sizze_145884 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144837, &mem_144837_cached_sizze_145884, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144838_cached_sizze_145885 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144838, &mem_144838_cached_sizze_145885, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142716 = 0; i_142716 < (int64_t) 4; i_142716++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142709 = 0; i_142709 < (int64_t) 16; i_142709++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142702 = 0; i_142702 < (int64_t) 16; i_142702++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_140946 = ((double *) mem_144762)[i_142716 * (int64_t) 256 + i_142709 * (int64_t) 16 + i_142702];
                
                // futhark/microgpt.fut:345:58-100
                
                double zs_res_140947 = zs_lhs_140946 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_140954 = ((double *) mem_144761)[i_142716 * (int64_t) 256 + i_142709 * (int64_t) 16 + i_142702];
                
                // futhark/microgpt.fut:361:58-100
                
                double zs_res_140955 = zs_lhs_140954 / 2.0;
                
                ((double *) mem_144837)[i_142702] = zs_res_140955;
                ((double *) mem_144838)[i_142702] = zs_res_140947;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144827, i_142709 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144837, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144828, i_142709 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144838, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144815, i_142716 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_144827, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144816, i_142716 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_144828, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_144869, (int64_t) 2048, "mem_144869")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144870_cached_sizze_145886 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144870, &mem_144870_cached_sizze_145886, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144871_cached_sizze_145887 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144871, &mem_144871_cached_sizze_145887, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144872_cached_sizze_145888 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144872, &mem_144872_cached_sizze_145888, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144889_cached_sizze_145889 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144889, &mem_144889_cached_sizze_145889, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144890_cached_sizze_145890 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144890, &mem_144890_cached_sizze_145890, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144891_cached_sizze_145891 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144891, &mem_144891_cached_sizze_145891, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144892_cached_sizze_145892 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144892, &mem_144892_cached_sizze_145892, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142740 = 0; i_142740 < (int64_t) 16; i_142740++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142727 = 0; i_142727 < (int64_t) 16; i_142727++) {
            // futhark/microgpt.fut:330:40-43
            
            int64_t zt_lhs_139851 = sdiv64(i_142727, (int64_t) 4);
            
            // futhark/microgpt.fut:330:27-45
            
            bool x_139852 = sle64((int64_t) 0, zt_lhs_139851);
            
            // futhark/microgpt.fut:330:27-45
            
            bool y_139853 = slt64(zt_lhs_139851, (int64_t) 4);
            
            // futhark/microgpt.fut:330:27-45
            
            bool bounds_check_139854 = x_139852 && y_139853;
            
            // futhark/microgpt.fut:330:27-45
            
            bool index_certs_139855;
            
            if (!bounds_check_139854) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_139851, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:330:27-45\n   #1  futhark/microgpt.fut:61:46-49\n   #2  futhark/microgpt.fut:330:5-108\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:322:13-330:118\n   #8  futhark/microgpt.fut:459:5-75\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:330:62-65
            
            int64_t zt_lhs_139856 = smod64(i_142727, (int64_t) 4);
            
            // futhark/microgpt.fut:330:27-67
            
            bool x_139857 = sle64((int64_t) 0, zt_lhs_139856);
            
            // futhark/microgpt.fut:330:27-67
            
            bool y_139858 = slt64(zt_lhs_139856, (int64_t) 4);
            
            // futhark/microgpt.fut:330:27-67
            
            bool bounds_check_139859 = x_139857 && y_139858;
            
            // futhark/microgpt.fut:330:27-67
            
            bool index_certs_139860;
            
            if (!bounds_check_139859) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_139856, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:330:27-67\n   #1  futhark/microgpt.fut:61:46-49\n   #2  futhark/microgpt.fut:330:5-108\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:322:13-330:118\n   #8  futhark/microgpt.fut:459:5-75\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_139861;
            double r_139863 = 0.0;
            
            for (int64_t i_139862 = 0; i_139862 < (int64_t) 16; i_139862++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_139864 = ((double *) mem_144430)[zt_lhs_139851 * (int64_t) 64 + i_139862 * (int64_t) 4 + zt_lhs_139856];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_139865 = ((double *) mem_143881)[zt_lhs_139851 * (int64_t) 256 + i_139862 * (int64_t) 16 + i_142740];
                
                // futhark/microgpt.fut:330:27-106
                
                double zt_res_139866 = zt_lhs_139864 * zt_rhs_139865;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_139867 = r_139863 + zt_res_139866;
                double r_tmp_145492 = zp_res_139867;
                
                r_139863 = r_tmp_145492;
            }
            defunc_0_lifted_lambda_res_139861 = r_139863;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_139875;
            double r_139877 = 0.0;
            
            for (int64_t i_139876 = 0; i_139876 < (int64_t) 16; i_139876++) {
                // futhark/microgpt.fut:346:27-175
                
                double x_139878 = ((double *) mem_143264)[zt_lhs_139851 * (int64_t) 64 + i_139876 * (int64_t) 4 + zt_lhs_139856];
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_139879;
                double r_139881 = 0.0;
                
                for (int64_t i_139880 = 0; i_139880 < (int64_t) 4; i_139880++) {
                    // futhark/microgpt.fut:346:49-128
                    
                    bool cond_139882 = zt_lhs_139856 == i_139880;
                    
                    // futhark/microgpt.fut:346:49-128
                    
                    double zt_lhs_139883;
                    
                    if (cond_139882) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_t_res_141632 = ((double *) mem_144816)[zt_lhs_139851 * (int64_t) 256 + i_139876 * (int64_t) 16 + i_142740];
                        
                        zt_lhs_139883 = zt_lhs_t_res_141632;
                    } else {
                        zt_lhs_139883 = 0.0;
                    }
                    // futhark/microgpt.fut:346:49-173
                    
                    double zt_res_139890 = x_139878 * zt_lhs_139883;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_139891 = r_139881 + zt_res_139890;
                    double r_tmp_145494 = zp_res_139891;
                    
                    r_139881 = r_tmp_145494;
                }
                defunc_0_lifted_lambda_res_139879 = r_139881;
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_139892 = r_139877 + defunc_0_lifted_lambda_res_139879;
                double r_tmp_145493 = zp_res_139892;
                
                r_139877 = r_tmp_145493;
            }
            defunc_0_lifted_lambda_res_139875 = r_139877;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_139903;
            double r_139905 = 0.0;
            
            for (int64_t i_139904 = 0; i_139904 < (int64_t) 16; i_139904++) {
                // futhark/microgpt.fut:362:27-175
                
                double x_139906 = ((double *) mem_143263)[zt_lhs_139851 * (int64_t) 64 + i_139904 * (int64_t) 4 + zt_lhs_139856];
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_139907;
                double r_139909 = 0.0;
                
                for (int64_t i_139908 = 0; i_139908 < (int64_t) 4; i_139908++) {
                    // futhark/microgpt.fut:362:49-128
                    
                    bool cond_139910 = zt_lhs_139856 == i_139908;
                    
                    // futhark/microgpt.fut:362:49-128
                    
                    double zt_lhs_139911;
                    
                    if (cond_139910) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_t_res_141638 = ((double *) mem_144815)[zt_lhs_139851 * (int64_t) 256 + i_142740 * (int64_t) 16 + i_139904];
                        
                        zt_lhs_139911 = zt_lhs_t_res_141638;
                    } else {
                        zt_lhs_139911 = 0.0;
                    }
                    // futhark/microgpt.fut:362:49-173
                    
                    double zt_res_139918 = x_139906 * zt_lhs_139911;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_139919 = r_139909 + zt_res_139918;
                    double r_tmp_145496 = zp_res_139919;
                    
                    r_139909 = r_tmp_145496;
                }
                defunc_0_lifted_lambda_res_139907 = r_139909;
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_139920 = r_139905 + defunc_0_lifted_lambda_res_139907;
                double r_tmp_145495 = zp_res_139920;
                
                r_139905 = r_tmp_145495;
            }
            defunc_0_lifted_lambda_res_139903 = r_139905;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_139928;
            double r_139930 = 0.0;
            
            for (int64_t i_139929 = 0; i_139929 < (int64_t) 16; i_139929++) {
                // futhark/microgpt.fut:394:68-177
                
                double x_139931 = ((double *) mem_143962)[i_139929 * (int64_t) 16 + i_142727];
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_139932;
                double r_139934 = 0.0;
                
                for (int64_t i_139933 = 0; i_139933 < (int64_t) 16; i_139933++) {
                    // futhark/microgpt.fut:394:91-150
                    
                    bool cond_139935 = i_142727 == i_139933;
                    
                    // futhark/microgpt.fut:394:91-150
                    
                    double zt_lhs_139936;
                    
                    if (cond_139935) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_t_res_141643 = ((double *) mem_144385)[i_139929 * (int64_t) 16 + i_142740];
                        
                        zt_lhs_139936 = zt_lhs_t_res_141643;
                    } else {
                        zt_lhs_139936 = 0.0;
                    }
                    // futhark/microgpt.fut:394:91-175
                    
                    double zt_res_139942 = x_139931 * zt_lhs_139936;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_139943 = r_139934 + zt_res_139942;
                    double r_tmp_145498 = zp_res_139943;
                    
                    r_139934 = r_tmp_145498;
                }
                defunc_0_lifted_lambda_res_139932 = r_139934;
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_139944 = r_139930 + defunc_0_lifted_lambda_res_139932;
                double r_tmp_145497 = zp_res_139944;
                
                r_139930 = r_tmp_145497;
            }
            defunc_0_lifted_lambda_res_139928 = r_139930;
            ((double *) mem_144889)[i_142727] = defunc_0_lifted_lambda_res_139928;
            ((double *) mem_144890)[i_142727] = defunc_0_lifted_lambda_res_139903;
            ((double *) mem_144891)[i_142727] = defunc_0_lifted_lambda_res_139875;
            ((double *) mem_144892)[i_142727] = defunc_0_lifted_lambda_res_139861;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144869.mem, i_142740 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144889, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144870, i_142740 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144890, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144871, i_142740 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144891, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144872, i_142740 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144892, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144933_cached_sizze_145893 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144933, &mem_144933_cached_sizze_145893, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144938_cached_sizze_145894 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144938, &mem_144938_cached_sizze_145894, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142751 = 0; i_142751 < (int64_t) 16; i_142751++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142747 = 0; i_142747 < (int64_t) 16; i_142747++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_130776;
            double r_130778 = 0.0;
            
            for (int64_t i_130777 = 0; i_130777 < (int64_t) 16; i_130777++) {
                // futhark/microgpt.fut:365:73-183
                
                double x_130779 = ((double *) wval_mem_143128.mem)[i_130777 * (int64_t) 16 + i_142747];
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_130780;
                double r_130782 = 0.0;
                
                for (int64_t i_130781 = 0; i_130781 < (int64_t) 16; i_130781++) {
                    // futhark/microgpt.fut:365:96-155
                    
                    bool cond_130783 = i_142747 == i_130781;
                    
                    // futhark/microgpt.fut:365:96-155
                    
                    double zt_lhs_130784;
                    
                    if (cond_130783) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_t_res_141768 = ((double *) mem_144872)[i_142751 * (int64_t) 16 + i_130777];
                        
                        zt_lhs_130784 = zt_lhs_t_res_141768;
                    } else {
                        zt_lhs_130784 = 0.0;
                    }
                    // futhark/microgpt.fut:365:96-181
                    
                    double zt_res_130790 = x_130779 * zt_lhs_130784;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_130791 = r_130782 + zt_res_130790;
                    double r_tmp_145502 = zp_res_130791;
                    
                    r_130782 = r_tmp_145502;
                }
                defunc_0_lifted_lambda_res_130780 = r_130782;
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_130792 = r_130778 + defunc_0_lifted_lambda_res_130780;
                double r_tmp_145501 = zp_res_130792;
                
                r_130778 = r_tmp_145501;
            }
            defunc_0_lifted_lambda_res_130776 = r_130778;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_130793;
            double r_130795 = 0.0;
            
            for (int64_t i_130794 = 0; i_130794 < (int64_t) 16; i_130794++) {
                // futhark/microgpt.fut:365:214-324
                
                double x_130796 = ((double *) wkey_mem_143122.mem)[i_130794 * (int64_t) 16 + i_142747];
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_130797;
                double r_130799 = 0.0;
                
                for (int64_t i_130798 = 0; i_130798 < (int64_t) 16; i_130798++) {
                    // futhark/microgpt.fut:365:237-296
                    
                    bool cond_130800 = i_142747 == i_130798;
                    
                    // futhark/microgpt.fut:365:237-296
                    
                    double zt_lhs_130801;
                    
                    if (cond_130800) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_t_res_141773 = ((double *) mem_144871)[i_142751 * (int64_t) 16 + i_130794];
                        
                        zt_lhs_130801 = zt_lhs_t_res_141773;
                    } else {
                        zt_lhs_130801 = 0.0;
                    }
                    // futhark/microgpt.fut:365:237-322
                    
                    double zt_res_130807 = x_130796 * zt_lhs_130801;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_130808 = r_130799 + zt_res_130807;
                    double r_tmp_145504 = zp_res_130808;
                    
                    r_130799 = r_tmp_145504;
                }
                defunc_0_lifted_lambda_res_130797 = r_130799;
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_130809 = r_130795 + defunc_0_lifted_lambda_res_130797;
                double r_tmp_145503 = zp_res_130809;
                
                r_130795 = r_tmp_145503;
            }
            defunc_0_lifted_lambda_res_130793 = r_130795;
            // futhark/microgpt.fut:365:51-326
            
            double zp_res_130810 = defunc_0_lifted_lambda_res_130776 + defunc_0_lifted_lambda_res_130793;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_130811;
            double r_130813 = 0.0;
            
            for (int64_t i_130812 = 0; i_130812 < (int64_t) 16; i_130812++) {
                // futhark/microgpt.fut:365:356-466
                
                double x_130814 = ((double *) wqry_mem_143125.mem)[i_130812 * (int64_t) 16 + i_142747];
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_130815;
                double r_130817 = 0.0;
                
                for (int64_t i_130816 = 0; i_130816 < (int64_t) 16; i_130816++) {
                    // futhark/microgpt.fut:365:379-438
                    
                    bool cond_130818 = i_142747 == i_130816;
                    
                    // futhark/microgpt.fut:365:379-438
                    
                    double zt_lhs_130819;
                    
                    if (cond_130818) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_t_res_141778 = ((double *) mem_144870)[i_142751 * (int64_t) 16 + i_130812];
                        
                        zt_lhs_130819 = zt_lhs_t_res_141778;
                    } else {
                        zt_lhs_130819 = 0.0;
                    }
                    // futhark/microgpt.fut:365:379-464
                    
                    double zt_res_130825 = x_130814 * zt_lhs_130819;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_130826 = r_130817 + zt_res_130825;
                    double r_tmp_145506 = zp_res_130826;
                    
                    r_130817 = r_tmp_145506;
                }
                defunc_0_lifted_lambda_res_130815 = r_130817;
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_130827 = r_130813 + defunc_0_lifted_lambda_res_130815;
                double r_tmp_145505 = zp_res_130827;
                
                r_130813 = r_tmp_145505;
            }
            defunc_0_lifted_lambda_res_130811 = r_130813;
            // futhark/microgpt.fut:365:187-468
            
            double zp_res_130828 = zp_res_130810 + defunc_0_lifted_lambda_res_130811;
            
            ((double *) mem_144938)[i_142747] = zp_res_130828;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144933, i_142751 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144938, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_144949, (int64_t) 2048, "mem_144949")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_144950, (int64_t) 2048, "mem_144950")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_144951, (int64_t) 2048, "mem_144951")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144952_cached_sizze_145895 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144952, &mem_144952_cached_sizze_145895, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144953_cached_sizze_145896 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144953, &mem_144953_cached_sizze_145896, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144972_cached_sizze_145897 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144972, &mem_144972_cached_sizze_145897, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144973_cached_sizze_145898 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144973, &mem_144973_cached_sizze_145898, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144974_cached_sizze_145899 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144974, &mem_144974_cached_sizze_145899, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142773 = 0; i_142773 < (int64_t) 16; i_142773++) {
        // futhark/microgpt.fut:364:47-59
        
        double zp_lhs_135602 = ((double *) mem_143209)[i_142773];
        
        // futhark/microgpt.fut:364:47-87
        
        double zp_res_135603 = 1.0e-5 + zp_lhs_135602;
        
        // futhark/microgpt.fut:364:39-87
        
        double sqrt_res_135604 = futrts_sqrt64(zp_res_135603);
        
        // futhark/microgpt.fut:366:128-157
        
        double zt_res_135612 = sqrt_res_135604 * sqrt_res_135604;
        
        // futhark/microgpt.fut:366:119-157
        
        double zs_res_135613 = 1.0 / zt_res_135612;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_135614;
        double r_135616 = 0.0;
        
        for (int64_t i_135615 = 0; i_135615 < (int64_t) 16; i_135615++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_135617 = ((double *) mem_144933)[i_142773 * (int64_t) 16 + i_135615];
            
            // futhark/microgpt.fut:61:46-49
            
            double zt_rhs_135618 = ((double *) mem_143193)[i_142773 * (int64_t) 16 + i_135615];
            
            // futhark/microgpt.fut:366:69-112
            
            double zt_res_135619 = zt_lhs_135617 * zt_rhs_135618;
            
            // futhark/microgpt.fut:366:90-157
            
            double zt_res_135620 = zs_res_135613 * zt_res_135619;
            
            // futhark/microgpt.fut:366:61-157
            
            double neg_res_135621 = -zt_res_135620;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_135622 = r_135616 + neg_res_135621;
            double r_tmp_145512 = zp_res_135622;
            
            r_135616 = r_tmp_145512;
        }
        defunc_0_lifted_lambda_res_135614 = r_135616;
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142759 = 0; i_142759 < (int64_t) 16; i_142759++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141030;
            double r_141032 = 0.0;
            
            for (int64_t i_141031 = 0; i_141031 < (int64_t) 16; i_141031++) {
                // futhark/microgpt.fut:391:68-176
                
                double x_141033 = ((double *) mem_143246)[i_141031 * (int64_t) 16 + i_142759];
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_141034;
                double r_141036 = 0.0;
                
                for (int64_t i_141035 = 0; i_141035 < (int64_t) 16; i_141035++) {
                    // futhark/microgpt.fut:391:91-150
                    
                    bool cond_141037 = i_142759 == i_141035;
                    
                    // futhark/microgpt.fut:391:91-150
                    
                    double zt_lhs_141038;
                    
                    if (cond_141037) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_t_res_141750 = ((double *) mem_144870)[i_141031 * (int64_t) 16 + i_142773];
                        
                        zt_lhs_141038 = zt_lhs_t_res_141750;
                    } else {
                        zt_lhs_141038 = 0.0;
                    }
                    // futhark/microgpt.fut:391:91-174
                    
                    double zt_res_141044 = x_141033 * zt_lhs_141038;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_141045 = r_141036 + zt_res_141044;
                    double r_tmp_145517 = zp_res_141045;
                    
                    r_141036 = r_tmp_145517;
                }
                defunc_0_lifted_lambda_res_141034 = r_141036;
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141046 = r_141032 + defunc_0_lifted_lambda_res_141034;
                double r_tmp_145516 = zp_res_141046;
                
                r_141032 = r_tmp_145516;
            }
            defunc_0_lifted_lambda_res_141030 = r_141032;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141049;
            double r_141051 = 0.0;
            
            for (int64_t i_141050 = 0; i_141050 < (int64_t) 16; i_141050++) {
                // futhark/microgpt.fut:392:68-176
                
                double x_141052 = ((double *) mem_143246)[i_141050 * (int64_t) 16 + i_142759];
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_141053;
                double r_141055 = 0.0;
                
                for (int64_t i_141054 = 0; i_141054 < (int64_t) 16; i_141054++) {
                    // futhark/microgpt.fut:392:91-150
                    
                    bool cond_141056 = i_142759 == i_141054;
                    
                    // futhark/microgpt.fut:392:91-150
                    
                    double zt_lhs_141057;
                    
                    if (cond_141056) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_t_res_141755 = ((double *) mem_144871)[i_141050 * (int64_t) 16 + i_142773];
                        
                        zt_lhs_141057 = zt_lhs_t_res_141755;
                    } else {
                        zt_lhs_141057 = 0.0;
                    }
                    // futhark/microgpt.fut:392:91-174
                    
                    double zt_res_141063 = x_141052 * zt_lhs_141057;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_141064 = r_141055 + zt_res_141063;
                    double r_tmp_145519 = zp_res_141064;
                    
                    r_141055 = r_tmp_145519;
                }
                defunc_0_lifted_lambda_res_141053 = r_141055;
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141065 = r_141051 + defunc_0_lifted_lambda_res_141053;
                double r_tmp_145518 = zp_res_141065;
                
                r_141051 = r_tmp_145518;
            }
            defunc_0_lifted_lambda_res_141049 = r_141051;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141071;
            double r_141073 = 0.0;
            
            for (int64_t i_141072 = 0; i_141072 < (int64_t) 16; i_141072++) {
                // futhark/microgpt.fut:393:68-176
                
                double x_141074 = ((double *) mem_143246)[i_141072 * (int64_t) 16 + i_142759];
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_141075;
                double r_141077 = 0.0;
                
                for (int64_t i_141076 = 0; i_141076 < (int64_t) 16; i_141076++) {
                    // futhark/microgpt.fut:393:91-150
                    
                    bool cond_141078 = i_142759 == i_141076;
                    
                    // futhark/microgpt.fut:393:91-150
                    
                    double zt_lhs_141079;
                    
                    if (cond_141078) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_t_res_141760 = ((double *) mem_144872)[i_141072 * (int64_t) 16 + i_142773];
                        
                        zt_lhs_141079 = zt_lhs_t_res_141760;
                    } else {
                        zt_lhs_141079 = 0.0;
                    }
                    // futhark/microgpt.fut:393:91-174
                    
                    double zt_res_141085 = x_141074 * zt_lhs_141079;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_141086 = r_141077 + zt_res_141085;
                    double r_tmp_145521 = zp_res_141086;
                    
                    r_141077 = r_tmp_145521;
                }
                defunc_0_lifted_lambda_res_141075 = r_141077;
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141087 = r_141073 + defunc_0_lifted_lambda_res_141075;
                double r_tmp_145520 = zp_res_141087;
                
                r_141073 = r_tmp_145520;
            }
            defunc_0_lifted_lambda_res_141071 = r_141073;
            ((double *) mem_144972)[i_142759] = defunc_0_lifted_lambda_res_141071;
            ((double *) mem_144973)[i_142759] = defunc_0_lifted_lambda_res_141049;
            ((double *) mem_144974)[i_142759] = defunc_0_lifted_lambda_res_141030;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144949.mem, i_142773 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144972, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144950.mem, i_142773 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144973, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144951.mem, i_142773 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144974, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        ((double *) mem_144952)[i_142773] = defunc_0_lifted_lambda_res_135614;
        ((double *) mem_144953)[i_142773] = sqrt_res_135604;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145011_cached_sizze_145900 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145011, &mem_145011_cached_sizze_145900, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142781 = 0; i_142781 < (int64_t) 16; i_142781++) {
        // futhark/microgpt.fut:367:39-51
        
        double zt_lhs_130856 = ((double *) mem_144952)[i_142781];
        
        // futhark/microgpt.fut:367:93-105
        
        double zp_lhs_130857 = ((double *) mem_143209)[i_142781];
        
        // futhark/microgpt.fut:367:93-133
        
        double zp_res_130858 = 1.0e-5 + zp_lhs_130857;
        
        // futhark/microgpt.fut:367:85-133
        
        double sqrt_res_130859 = futrts_sqrt64(zp_res_130858);
        
        // futhark/microgpt.fut:367:71-135
        
        double zt_res_130860 = 2.0 * sqrt_res_130859;
        
        // futhark/microgpt.fut:367:57-135
        
        double zs_res_130861 = 1.0 / zt_res_130860;
        
        // futhark/microgpt.fut:367:39-135
        
        double zt_res_130862 = zt_lhs_130856 * zs_res_130861;
        
        ((double *) mem_145011)[i_142781] = zt_res_130862;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145018_cached_sizze_145901 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145018, &mem_145018_cached_sizze_145901, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145023_cached_sizze_145902 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145023, &mem_145023_cached_sizze_145902, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142789 = 0; i_142789 < (int64_t) 16; i_142789++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142785 = 0; i_142785 < (int64_t) 16; i_142785++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_130876 = ((double *) mem_144385)[i_142789 * (int64_t) 16 + i_142785];
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_130877;
            double r_130879 = 0.0;
            
            for (int64_t i_130878 = 0; i_130878 < (int64_t) 16; i_130878++) {
                // futhark/microgpt.fut:368:86-174
                
                bool cond_130880 = i_142789 == i_130878;
                
                // futhark/microgpt.fut:368:86-174
                
                double zp_lhs_130881;
                
                if (cond_130880) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_141780 = ((double *) mem_144933)[i_130878 * (int64_t) 16 + i_142785];
                    
                    // futhark/microgpt.fut:368:150-162
                    
                    double zs_rhs_141781 = ((double *) mem_144953)[i_130878];
                    
                    // futhark/microgpt.fut:368:142-162
                    
                    double zs_res_141782 = 1.0 / zs_rhs_141781;
                    
                    // futhark/microgpt.fut:368:116-162
                    
                    double zt_res_141783 = zt_lhs_141780 * zs_res_141782;
                    
                    zp_lhs_130881 = zt_res_141783;
                } else {
                    zp_lhs_130881 = 0.0;
                }
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_130889;
                
                if (cond_130880) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double x_141784 = ((double *) mem_143193)[i_130878 * (int64_t) 16 + i_142785];
                    
                    zt_rhs_130889 = x_141784;
                } else {
                    zt_rhs_130889 = 0.0;
                }
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_130891;
                
                if (cond_130880) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double x_141785 = ((double *) mem_143193)[i_130878 * (int64_t) 16 + i_142785];
                    
                    zt_rhs_130891 = x_141785;
                } else {
                    zt_rhs_130891 = 0.0;
                }
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_130893;
                double r_130895 = 0.0;
                
                for (int64_t i_130894 = 0; i_130894 < (int64_t) 16; i_130894++) {
                    // futhark/microgpt.fut:368:204-338
                    
                    double zp_lhs_130896;
                    
                    if (cond_130880) {
                        // futhark/microgpt.fut:368:235-303
                        
                        bool cond_141788 = i_142785 == i_130894;
                        
                        // futhark/microgpt.fut:368:235-303
                        
                        double zt_lhs_141789;
                        
                        if (cond_141788) {
                            // futhark/microgpt.fut:368:265-277
                            
                            double zs_lhs_141790 = ((double *) mem_145011)[i_130878];
                            
                            // futhark/microgpt.fut:368:265-292
                            
                            double zs_res_141791 = zs_lhs_141790 / 16.0;
                            
                            zt_lhs_141789 = zs_res_141791;
                        } else {
                            zt_lhs_141789 = 0.0;
                        }
                        // futhark/microgpt.fut:368:235-327
                        
                        double zt_res_141792 = zt_rhs_130889 * zt_lhs_141789;
                        
                        zp_lhs_130896 = zt_res_141792;
                    } else {
                        zp_lhs_130896 = 0.0;
                    }
                    // futhark/microgpt.fut:368:345-479
                    
                    double zp_rhs_130902;
                    
                    if (cond_130880) {
                        // futhark/microgpt.fut:368:376-444
                        
                        bool cond_141795 = i_142785 == i_130894;
                        
                        // futhark/microgpt.fut:368:376-444
                        
                        double zt_lhs_141796;
                        
                        if (cond_141795) {
                            // futhark/microgpt.fut:368:406-418
                            
                            double zs_lhs_141797 = ((double *) mem_145011)[i_130878];
                            
                            // futhark/microgpt.fut:368:406-433
                            
                            double zs_res_141798 = zs_lhs_141797 / 16.0;
                            
                            zt_lhs_141796 = zs_res_141798;
                        } else {
                            zt_lhs_141796 = 0.0;
                        }
                        // futhark/microgpt.fut:368:376-468
                        
                        double zt_res_141799 = zt_rhs_130891 * zt_lhs_141796;
                        
                        zp_rhs_130902 = zt_res_141799;
                    } else {
                        zp_rhs_130902 = 0.0;
                    }
                    // futhark/microgpt.fut:368:204-479
                    
                    double zp_res_130908 = zp_lhs_130896 + zp_rhs_130902;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_130909 = r_130895 + zp_res_130908;
                    double r_tmp_145526 = zp_res_130909;
                    
                    r_130895 = r_tmp_145526;
                }
                defunc_0_lifted_lambda_res_130893 = r_130895;
                // futhark/microgpt.fut:368:86-482
                
                double zp_res_130910 = zp_lhs_130881 + defunc_0_lifted_lambda_res_130893;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_130911 = r_130879 + zp_res_130910;
                double r_tmp_145525 = zp_res_130911;
                
                r_130879 = r_tmp_145525;
            }
            defunc_0_lifted_lambda_res_130877 = r_130879;
            // futhark/microgpt.fut:368:37-485
            
            double zp_res_130912 = zp_lhs_130876 + defunc_0_lifted_lambda_res_130877;
            
            ((double *) mem_145023)[i_142785] = zp_res_130912;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_145018, i_142789 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145023, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145034_cached_sizze_145903 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145034, &mem_145034_cached_sizze_145903, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145035_cached_sizze_145904 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145035, &mem_145035_cached_sizze_145904, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145044_cached_sizze_145905 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145044, &mem_145044_cached_sizze_145905, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145045_cached_sizze_145906 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145045, &mem_145045_cached_sizze_145906, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142802 = 0; i_142802 < (int64_t) 16; i_142802++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142795 = 0; i_142795 < (int64_t) 16; i_142795++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_141176 = ((double *) mem_145018)[i_142802 * (int64_t) 16 + i_142795];
            
            ((double *) mem_145044)[i_142795] = lifted_lambda_res_141176;
            ((double *) mem_145045)[i_142795] = lifted_lambda_res_141176;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_145034, i_142802 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145044, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_145035, i_142802 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145045, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145066_cached_sizze_145907 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145066, &mem_145066_cached_sizze_145907, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145067_cached_sizze_145908 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145067, &mem_145067_cached_sizze_145908, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145068_cached_sizze_145909 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145068, &mem_145068_cached_sizze_145909, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145069_cached_sizze_145910 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145069, &mem_145069_cached_sizze_145910, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142813 = 0; i_142813 < (int64_t) 16; i_142813++) {
        // futhark/microgpt.fut:386:47-59
        
        double zp_lhs_135733 = ((double *) mem_143150)[i_142813];
        
        // futhark/microgpt.fut:386:47-87
        
        double zp_res_135734 = 1.0e-5 + zp_lhs_135733;
        
        // futhark/microgpt.fut:386:39-87
        
        double sqrt_res_135735 = futrts_sqrt64(zp_res_135734);
        
        // futhark/microgpt.fut:388:156-185
        
        double zt_res_135743 = sqrt_res_135735 * sqrt_res_135735;
        
        // futhark/microgpt.fut:388:147-185
        
        double zs_res_135744 = 1.0 / zt_res_135743;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_135745;
        double r_135747 = 0.0;
        
        for (int64_t i_135746 = 0; i_135746 < (int64_t) 16; i_135746++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_135748 = ((double *) mem_145035)[i_142813 * (int64_t) 16 + i_135746];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_135749 = ((double *) wpe_mem_143124.mem)[i_142813 * (int64_t) 16 + i_135746];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_135750 = ((double *) mem_143133)[i_142813 * (int64_t) 16 + i_135746];
            
            // futhark/microgpt.fut:388:95-139
            
            double zp_res_135751 = zp_lhs_135749 + zp_rhs_135750;
            
            // futhark/microgpt.fut:388:69-139
            
            double zt_res_135752 = zt_lhs_135748 * zp_res_135751;
            
            // futhark/microgpt.fut:388:90-185
            
            double zt_res_135753 = zs_res_135744 * zt_res_135752;
            
            // futhark/microgpt.fut:388:61-185
            
            double neg_res_135754 = -zt_res_135753;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_135755 = r_135747 + neg_res_135754;
            double r_tmp_145535 = zp_res_135755;
            
            r_135747 = r_tmp_145535;
        }
        defunc_0_lifted_lambda_res_135745 = r_135747;
        // futhark/microgpt.fut:399:47-59
        
        double zp_lhs_135766 = ((double *) mem_143149)[i_142813];
        
        // futhark/microgpt.fut:399:47-87
        
        double zp_res_135767 = 1.0e-5 + zp_lhs_135766;
        
        // futhark/microgpt.fut:399:39-87
        
        double sqrt_res_135768 = futrts_sqrt64(zp_res_135767);
        
        // futhark/microgpt.fut:401:156-185
        
        double zt_res_135776 = sqrt_res_135768 * sqrt_res_135768;
        
        // futhark/microgpt.fut:401:147-185
        
        double zs_res_135777 = 1.0 / zt_res_135776;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_135778;
        double r_135780 = 0.0;
        
        for (int64_t i_135779 = 0; i_135779 < (int64_t) 16; i_135779++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_135781 = ((double *) mem_145034)[i_142813 * (int64_t) 16 + i_135779];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_135782 = ((double *) wpe_mem_143124.mem)[i_142813 * (int64_t) 16 + i_135779];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_135783 = ((double *) mem_143133)[i_142813 * (int64_t) 16 + i_135779];
            
            // futhark/microgpt.fut:401:95-139
            
            double zp_res_135784 = zp_lhs_135782 + zp_rhs_135783;
            
            // futhark/microgpt.fut:401:69-139
            
            double zt_res_135785 = zt_lhs_135781 * zp_res_135784;
            
            // futhark/microgpt.fut:401:90-185
            
            double zt_res_135786 = zs_res_135777 * zt_res_135785;
            
            // futhark/microgpt.fut:401:61-185
            
            double neg_res_135787 = -zt_res_135786;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_135788 = r_135780 + neg_res_135787;
            double r_tmp_145536 = zp_res_135788;
            
            r_135780 = r_tmp_145536;
        }
        defunc_0_lifted_lambda_res_135778 = r_135780;
        ((double *) mem_145066)[i_142813] = defunc_0_lifted_lambda_res_135778;
        ((double *) mem_145067)[i_142813] = sqrt_res_135768;
        ((double *) mem_145068)[i_142813] = defunc_0_lifted_lambda_res_135745;
        ((double *) mem_145069)[i_142813] = sqrt_res_135735;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145094_cached_sizze_145911 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145094, &mem_145094_cached_sizze_145911, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145095_cached_sizze_145912 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145095, &mem_145095_cached_sizze_145912, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142822 = 0; i_142822 < (int64_t) 16; i_142822++) {
        // futhark/microgpt.fut:389:39-51
        
        double zt_lhs_135849 = ((double *) mem_145068)[i_142822];
        
        // futhark/microgpt.fut:389:93-105
        
        double zp_lhs_135850 = ((double *) mem_143150)[i_142822];
        
        // futhark/microgpt.fut:389:93-133
        
        double zp_res_135851 = 1.0e-5 + zp_lhs_135850;
        
        // futhark/microgpt.fut:389:85-133
        
        double sqrt_res_135852 = futrts_sqrt64(zp_res_135851);
        
        // futhark/microgpt.fut:389:71-135
        
        double zt_res_135853 = 2.0 * sqrt_res_135852;
        
        // futhark/microgpt.fut:389:57-135
        
        double zs_res_135854 = 1.0 / zt_res_135853;
        
        // futhark/microgpt.fut:389:39-135
        
        double zt_res_135855 = zt_lhs_135849 * zs_res_135854;
        
        // futhark/microgpt.fut:402:39-51
        
        double zt_lhs_135862 = ((double *) mem_145066)[i_142822];
        
        // futhark/microgpt.fut:402:93-105
        
        double zp_lhs_135863 = ((double *) mem_143149)[i_142822];
        
        // futhark/microgpt.fut:402:93-133
        
        double zp_res_135864 = 1.0e-5 + zp_lhs_135863;
        
        // futhark/microgpt.fut:402:85-133
        
        double sqrt_res_135865 = futrts_sqrt64(zp_res_135864);
        
        // futhark/microgpt.fut:402:71-135
        
        double zt_res_135866 = 2.0 * sqrt_res_135865;
        
        // futhark/microgpt.fut:402:57-135
        
        double zs_res_135867 = 1.0 / zt_res_135866;
        
        // futhark/microgpt.fut:402:39-135
        
        double zt_res_135868 = zt_lhs_135862 * zs_res_135867;
        
        ((double *) mem_145094)[i_142822] = zt_res_135868;
        ((double *) mem_145095)[i_142822] = zt_res_135855;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145108_cached_sizze_145913 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145108, &mem_145108_cached_sizze_145913, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145109, (int64_t) 2048, "mem_145109")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145118_cached_sizze_145914 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145118, &mem_145118_cached_sizze_145914, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145119_cached_sizze_145915 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145119, &mem_145119_cached_sizze_145915, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142836 = 0; i_142836 < (int64_t) 16; i_142836++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142829 = 0; i_142829 < (int64_t) 16; i_142829++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141202;
            double r_141204 = 0.0;
            
            for (int64_t i_141203 = 0; i_141203 < (int64_t) 16; i_141203++) {
                // futhark/microgpt.fut:390:60-148
                
                bool cond_141205 = i_142836 == i_141203;
                
                // futhark/microgpt.fut:390:60-148
                
                double zp_lhs_141206;
                
                if (cond_141205) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_141804 = ((double *) mem_145035)[i_141203 * (int64_t) 16 + i_142829];
                    
                    // futhark/microgpt.fut:390:124-136
                    
                    double zs_rhs_141805 = ((double *) mem_145069)[i_141203];
                    
                    // futhark/microgpt.fut:390:116-136
                    
                    double zs_res_141806 = 1.0 / zs_rhs_141805;
                    
                    // futhark/microgpt.fut:390:90-136
                    
                    double zt_res_141807 = zt_lhs_141804 * zs_res_141806;
                    
                    zp_lhs_141206 = zt_res_141807;
                } else {
                    zp_lhs_141206 = 0.0;
                }
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_141215;
                
                if (cond_141205) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double x_141808 = ((double *) wpe_mem_143124.mem)[i_141203 * (int64_t) 16 + i_142829];
                    
                    zp_lhs_141215 = x_141808;
                } else {
                    zp_lhs_141215 = 0.0;
                }
                // futhark/microgpt.fut:61:46-49
                
                double zp_rhs_141217;
                
                if (cond_141205) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double x_141809 = ((double *) mem_143133)[i_141203 * (int64_t) 16 + i_142829];
                    
                    zp_rhs_141217 = x_141809;
                } else {
                    zp_rhs_141217 = 0.0;
                }
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_141219;
                
                if (cond_141205) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double x_141810 = ((double *) wpe_mem_143124.mem)[i_141203 * (int64_t) 16 + i_142829];
                    
                    zp_lhs_141219 = x_141810;
                } else {
                    zp_lhs_141219 = 0.0;
                }
                // futhark/microgpt.fut:61:46-49
                
                double zp_rhs_141221;
                
                if (cond_141205) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double x_141811 = ((double *) mem_143133)[i_141203 * (int64_t) 16 + i_142829];
                    
                    zp_rhs_141221 = x_141811;
                } else {
                    zp_rhs_141221 = 0.0;
                }
                // futhark/microgpt.fut:390:284-328
                
                double zp_res_141223 = zp_lhs_141215 + zp_rhs_141217;
                
                // futhark/microgpt.fut:390:453-497
                
                double zp_res_141224 = zp_lhs_141219 + zp_rhs_141221;
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_141225;
                double r_141227 = 0.0;
                
                for (int64_t i_141226 = 0; i_141226 < (int64_t) 16; i_141226++) {
                    // futhark/microgpt.fut:390:178-340
                    
                    double zp_lhs_141228;
                    
                    if (cond_141205) {
                        // futhark/microgpt.fut:390:209-277
                        
                        bool cond_141814 = i_142829 == i_141226;
                        
                        // futhark/microgpt.fut:390:209-277
                        
                        double zt_lhs_141815;
                        
                        if (cond_141814) {
                            // futhark/microgpt.fut:390:239-251
                            
                            double zs_lhs_141816 = ((double *) mem_145095)[i_141203];
                            
                            // futhark/microgpt.fut:390:239-266
                            
                            double zs_res_141817 = zs_lhs_141816 / 16.0;
                            
                            zt_lhs_141815 = zs_res_141817;
                        } else {
                            zt_lhs_141815 = 0.0;
                        }
                        // futhark/microgpt.fut:390:209-328
                        
                        double zt_res_141818 = zp_res_141223 * zt_lhs_141815;
                        
                        zp_lhs_141228 = zt_res_141818;
                    } else {
                        zp_lhs_141228 = 0.0;
                    }
                    // futhark/microgpt.fut:390:347-509
                    
                    double zp_rhs_141234;
                    
                    if (cond_141205) {
                        // futhark/microgpt.fut:390:378-446
                        
                        bool cond_141821 = i_142829 == i_141226;
                        
                        // futhark/microgpt.fut:390:378-446
                        
                        double zt_lhs_141822;
                        
                        if (cond_141821) {
                            // futhark/microgpt.fut:390:408-420
                            
                            double zs_lhs_141823 = ((double *) mem_145095)[i_141203];
                            
                            // futhark/microgpt.fut:390:408-435
                            
                            double zs_res_141824 = zs_lhs_141823 / 16.0;
                            
                            zt_lhs_141822 = zs_res_141824;
                        } else {
                            zt_lhs_141822 = 0.0;
                        }
                        // futhark/microgpt.fut:390:378-497
                        
                        double zt_res_141825 = zp_res_141224 * zt_lhs_141822;
                        
                        zp_rhs_141234 = zt_res_141825;
                    } else {
                        zp_rhs_141234 = 0.0;
                    }
                    // futhark/microgpt.fut:390:178-509
                    
                    double zp_res_141240 = zp_lhs_141228 + zp_rhs_141234;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_141241 = r_141227 + zp_res_141240;
                    double r_tmp_145544 = zp_res_141241;
                    
                    r_141227 = r_tmp_145544;
                }
                defunc_0_lifted_lambda_res_141225 = r_141227;
                // futhark/microgpt.fut:390:60-512
                
                double zp_res_141242 = zp_lhs_141206 + defunc_0_lifted_lambda_res_141225;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141243 = r_141204 + zp_res_141242;
                double r_tmp_145543 = zp_res_141243;
                
                r_141204 = r_tmp_145543;
            }
            defunc_0_lifted_lambda_res_141202 = r_141204;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141249;
            double r_141251 = 0.0;
            
            for (int64_t i_141250 = 0; i_141250 < (int64_t) 16; i_141250++) {
                // futhark/microgpt.fut:403:60-148
                
                bool cond_141252 = i_142836 == i_141250;
                
                // futhark/microgpt.fut:403:60-148
                
                double zp_lhs_141253;
                
                if (cond_141252) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_141827 = ((double *) mem_145034)[i_141250 * (int64_t) 16 + i_142829];
                    
                    // futhark/microgpt.fut:403:124-136
                    
                    double zs_rhs_141828 = ((double *) mem_145067)[i_141250];
                    
                    // futhark/microgpt.fut:403:116-136
                    
                    double zs_res_141829 = 1.0 / zs_rhs_141828;
                    
                    // futhark/microgpt.fut:403:90-136
                    
                    double zt_res_141830 = zt_lhs_141827 * zs_res_141829;
                    
                    zp_lhs_141253 = zt_res_141830;
                } else {
                    zp_lhs_141253 = 0.0;
                }
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_141262;
                
                if (cond_141252) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double x_141831 = ((double *) wpe_mem_143124.mem)[i_141250 * (int64_t) 16 + i_142829];
                    
                    zp_lhs_141262 = x_141831;
                } else {
                    zp_lhs_141262 = 0.0;
                }
                // futhark/microgpt.fut:61:46-49
                
                double zp_rhs_141264;
                
                if (cond_141252) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double x_141832 = ((double *) mem_143133)[i_141250 * (int64_t) 16 + i_142829];
                    
                    zp_rhs_141264 = x_141832;
                } else {
                    zp_rhs_141264 = 0.0;
                }
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_141266;
                
                if (cond_141252) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double x_141833 = ((double *) wpe_mem_143124.mem)[i_141250 * (int64_t) 16 + i_142829];
                    
                    zp_lhs_141266 = x_141833;
                } else {
                    zp_lhs_141266 = 0.0;
                }
                // futhark/microgpt.fut:61:46-49
                
                double zp_rhs_141268;
                
                if (cond_141252) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double x_141834 = ((double *) mem_143133)[i_141250 * (int64_t) 16 + i_142829];
                    
                    zp_rhs_141268 = x_141834;
                } else {
                    zp_rhs_141268 = 0.0;
                }
                // futhark/microgpt.fut:403:284-328
                
                double zp_res_141270 = zp_lhs_141262 + zp_rhs_141264;
                
                // futhark/microgpt.fut:403:453-497
                
                double zp_res_141271 = zp_lhs_141266 + zp_rhs_141268;
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_141272;
                double r_141274 = 0.0;
                
                for (int64_t i_141273 = 0; i_141273 < (int64_t) 16; i_141273++) {
                    // futhark/microgpt.fut:403:178-340
                    
                    double zp_lhs_141275;
                    
                    if (cond_141252) {
                        // futhark/microgpt.fut:403:209-277
                        
                        bool cond_141837 = i_142829 == i_141273;
                        
                        // futhark/microgpt.fut:403:209-277
                        
                        double zt_lhs_141838;
                        
                        if (cond_141837) {
                            // futhark/microgpt.fut:403:239-251
                            
                            double zs_lhs_141839 = ((double *) mem_145094)[i_141250];
                            
                            // futhark/microgpt.fut:403:239-266
                            
                            double zs_res_141840 = zs_lhs_141839 / 16.0;
                            
                            zt_lhs_141838 = zs_res_141840;
                        } else {
                            zt_lhs_141838 = 0.0;
                        }
                        // futhark/microgpt.fut:403:209-328
                        
                        double zt_res_141841 = zp_res_141270 * zt_lhs_141838;
                        
                        zp_lhs_141275 = zt_res_141841;
                    } else {
                        zp_lhs_141275 = 0.0;
                    }
                    // futhark/microgpt.fut:403:347-509
                    
                    double zp_rhs_141281;
                    
                    if (cond_141252) {
                        // futhark/microgpt.fut:403:378-446
                        
                        bool cond_141844 = i_142829 == i_141273;
                        
                        // futhark/microgpt.fut:403:378-446
                        
                        double zt_lhs_141845;
                        
                        if (cond_141844) {
                            // futhark/microgpt.fut:403:408-420
                            
                            double zs_lhs_141846 = ((double *) mem_145094)[i_141250];
                            
                            // futhark/microgpt.fut:403:408-435
                            
                            double zs_res_141847 = zs_lhs_141846 / 16.0;
                            
                            zt_lhs_141845 = zs_res_141847;
                        } else {
                            zt_lhs_141845 = 0.0;
                        }
                        // futhark/microgpt.fut:403:378-497
                        
                        double zt_res_141848 = zp_res_141271 * zt_lhs_141845;
                        
                        zp_rhs_141281 = zt_res_141848;
                    } else {
                        zp_rhs_141281 = 0.0;
                    }
                    // futhark/microgpt.fut:403:178-509
                    
                    double zp_res_141287 = zp_lhs_141275 + zp_rhs_141281;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_141288 = r_141274 + zp_res_141287;
                    double r_tmp_145546 = zp_res_141288;
                    
                    r_141274 = r_tmp_145546;
                }
                defunc_0_lifted_lambda_res_141272 = r_141274;
                // futhark/microgpt.fut:403:60-512
                
                double zp_res_141289 = zp_lhs_141253 + defunc_0_lifted_lambda_res_141272;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141290 = r_141251 + zp_res_141289;
                double r_tmp_145545 = zp_res_141290;
                
                r_141251 = r_tmp_145545;
            }
            defunc_0_lifted_lambda_res_141249 = r_141251;
            ((double *) mem_145118)[i_142829] = defunc_0_lifted_lambda_res_141249;
            ((double *) mem_145119)[i_142829] = defunc_0_lifted_lambda_res_141202;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_145108, i_142836 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145118, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_145109.mem, i_142836 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145119, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145140, (int64_t) 8192, "mem_145140")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145145_cached_sizze_145916 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145145, &mem_145145_cached_sizze_145916, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142845 = 0; i_142845 < (int64_t) 64; i_142845++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142841 = 0; i_142841 < (int64_t) 16; i_142841++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_131138;
            double r_131140 = 0.0;
            
            for (int64_t i_131139 = 0; i_131139 < (int64_t) 16; i_131139++) {
                // futhark/microgpt.fut:395:67-176
                
                double x_131141 = ((double *) mem_144031)[i_131139 * (int64_t) 16 + i_142841];
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_131142;
                double r_131144 = 0.0;
                
                for (int64_t i_131143 = 0; i_131143 < (int64_t) 16; i_131143++) {
                    // futhark/microgpt.fut:395:90-149
                    
                    bool cond_131145 = i_142841 == i_131143;
                    
                    // futhark/microgpt.fut:395:90-149
                    
                    double zt_lhs_131146;
                    
                    if (cond_131145) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_t_res_141855 = ((double *) mem_144317)[i_131139 * (int64_t) 64 + i_142845];
                        
                        zt_lhs_131146 = zt_lhs_t_res_141855;
                    } else {
                        zt_lhs_131146 = 0.0;
                    }
                    // futhark/microgpt.fut:395:90-174
                    
                    double zt_res_131152 = x_131141 * zt_lhs_131146;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_131153 = r_131144 + zt_res_131152;
                    double r_tmp_145550 = zp_res_131153;
                    
                    r_131144 = r_tmp_145550;
                }
                defunc_0_lifted_lambda_res_131142 = r_131144;
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_131154 = r_131140 + defunc_0_lifted_lambda_res_131142;
                double r_tmp_145549 = zp_res_131154;
                
                r_131140 = r_tmp_145549;
            }
            defunc_0_lifted_lambda_res_131138 = r_131140;
            ((double *) mem_145145)[i_142841] = defunc_0_lifted_lambda_res_131138;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_145140.mem, i_142845 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145145, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145156, (int64_t) 3456, "mem_145156")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145157, (int64_t) 3456, "mem_145157")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145166_cached_sizze_145917 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145166, &mem_145166_cached_sizze_145917, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145167_cached_sizze_145918 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145167, &mem_145167_cached_sizze_145918, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142858 = 0; i_142858 < (int64_t) 27; i_142858++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142851 = 0; i_142851 < (int64_t) 16; i_142851++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141404;
            double r_141406 = 0.0;
            
            for (int64_t i_141405 = 0; i_141405 < (int64_t) 16; i_141405++) {
                // futhark/microgpt.fut:397:68-176
                
                double x_141407 = ((double *) mem_144079)[i_141405 * (int64_t) 16 + i_142851];
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_141408;
                double r_141410 = 0.0;
                
                for (int64_t i_141409 = 0; i_141409 < (int64_t) 16; i_141409++) {
                    // futhark/microgpt.fut:397:91-149
                    
                    bool cond_141411 = i_142851 == i_141409;
                    
                    // futhark/microgpt.fut:397:91-149
                    
                    double zt_lhs_141412;
                    
                    if (cond_141411) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_t_res_141861 = ((double *) mem_144284)[i_141405 * (int64_t) 27 + i_142858];
                        
                        zt_lhs_141412 = zt_lhs_t_res_141861;
                    } else {
                        zt_lhs_141412 = 0.0;
                    }
                    // futhark/microgpt.fut:397:91-174
                    
                    double zt_res_141418 = x_141407 * zt_lhs_141412;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_141419 = r_141410 + zt_res_141418;
                    double r_tmp_145556 = zp_res_141419;
                    
                    r_141410 = r_tmp_145556;
                }
                defunc_0_lifted_lambda_res_141408 = r_141410;
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141420 = r_141406 + defunc_0_lifted_lambda_res_141408;
                double r_tmp_145555 = zp_res_141420;
                
                r_141406 = r_tmp_145555;
            }
            defunc_0_lifted_lambda_res_141404 = r_141406;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141423;
            double r_141425 = 0.0;
            
            for (int64_t i_141424 = 0; i_141424 < (int64_t) 16; i_141424++) {
                // futhark/microgpt.fut:460:62-71
                
                int64_t zeze_lhs_141426 = ((int64_t *) tokens_mem_143130.mem)[i_141424];
                
                // futhark/microgpt.fut:460:58-109
                
                bool cond_141427 = zeze_lhs_141426 == i_142858;
                
                // futhark/microgpt.fut:460:58-109
                
                double lifted_lambda_res_141428;
                
                if (cond_141427) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double lifted_lambda_res_t_res_141866 = ((double *) mem_145108)[i_141424 * (int64_t) 16 + i_142851];
                    
                    lifted_lambda_res_141428 = lifted_lambda_res_t_res_141866;
                } else {
                    lifted_lambda_res_141428 = 0.0;
                }
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141434 = r_141425 + lifted_lambda_res_141428;
                double r_tmp_145557 = zp_res_141434;
                
                r_141425 = r_tmp_145557;
            }
            defunc_0_lifted_lambda_res_141423 = r_141425;
            ((double *) mem_145166)[i_142851] = defunc_0_lifted_lambda_res_141423;
            ((double *) mem_145167)[i_142851] = defunc_0_lifted_lambda_res_141404;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_145156.mem, i_142858 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145166, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_145157.mem, i_142858 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145167, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    if (memblock_set(ctx, &mem_out_145206, &mem_145156, "mem_145156") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_145207, &mem_145109, "mem_145109") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_145208, &mem_144951, "mem_144951") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_145209, &mem_144950, "mem_144950") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_145210, &mem_144949, "mem_144949") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_145211, &mem_144869, "mem_144869") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_145212, &mem_145140, "mem_145140") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_145213, &mem_144316, "mem_144316") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_145214, &mem_145157, "mem_145157") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145674, &mem_out_145206, "mem_out_145206") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145675, &mem_out_145207, "mem_out_145207") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145676, &mem_out_145208, "mem_out_145208") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145677, &mem_out_145209, "mem_out_145209") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145678, &mem_out_145210, "mem_out_145210") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145679, &mem_out_145211, "mem_out_145211") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145680, &mem_out_145212, "mem_out_145212") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145681, &mem_out_145213, "mem_out_145213") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145682, &mem_out_145214, "mem_out_145214") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_143133);
        free(mem_143138);
        free(mem_143149);
        free(mem_143150);
        free(mem_143151);
        free(mem_143170);
        free(mem_143177);
        free(mem_143182);
        free(mem_143193);
        free(mem_143198);
        free(mem_143209);
        free(mem_143210);
        free(mem_143223);
        free(mem_143230);
        free(mem_143235);
        free(mem_143246);
        free(mem_143251);
        free(mem_143262);
        free(mem_143263);
        free(mem_143264);
        free(mem_143280);
        free(mem_143281);
        free(mem_143282);
        free(mem_143295);
        free(mem_143296);
        free(mem_143297);
        free(mem_143343);
        free(mem_143344);
        free(mem_143345);
        free(mem_143346);
        free(mem_143367);
        free(mem_143368);
        free(mem_143369);
        free(mem_143370);
        free(mem_143387);
        free(mem_143388);
        free(mem_143389);
        free(mem_143390);
        free(mem_143451);
        free(mem_143452);
        free(mem_143453);
        free(mem_143454);
        free(mem_143475);
        free(mem_143476);
        free(mem_143477);
        free(mem_143478);
        free(mem_143495);
        free(mem_143496);
        free(mem_143497);
        free(mem_143498);
        free(mem_143559);
        free(mem_143560);
        free(mem_143561);
        free(mem_143562);
        free(mem_143563);
        free(mem_143564);
        free(mem_143565);
        free(mem_143566);
        free(mem_143599);
        free(mem_143600);
        free(mem_143601);
        free(mem_143602);
        free(mem_143603);
        free(mem_143604);
        free(mem_143605);
        free(mem_143606);
        free(mem_143687);
        free(mem_143688);
        free(mem_143689);
        free(mem_143690);
        free(mem_143711);
        free(mem_143712);
        free(mem_143713);
        free(mem_143714);
        free(mem_143731);
        free(mem_143732);
        free(mem_143733);
        free(mem_143734);
        free(mem_143795);
        free(mem_143796);
        free(mem_143805);
        free(mem_143806);
        free(mem_143827);
        free(mem_143828);
        free(mem_143839);
        free(mem_143840);
        free(mem_143849);
        free(mem_143850);
        free(mem_143881);
        free(mem_143882);
        free(mem_143893);
        free(mem_143894);
        free(mem_143903);
        free(mem_143904);
        free(mem_143935);
        free(mem_143941);
        free(mem_143946);
        free(mem_143962);
        free(mem_143967);
        free(mem_143978);
        free(mem_143983);
        free(mem_143994);
        free(mem_143995);
        free(mem_144008);
        free(mem_144015);
        free(mem_144020);
        free(mem_144031);
        free(mem_144036);
        free(mem_144047);
        free(mem_144052);
        free(mem_144063);
        free(mem_144068);
        free(mem_144079);
        free(mem_144084);
        free(mem_144095);
        free(mem_144100);
        free(mem_144111);
        free(mem_144112);
        free(mem_144113);
        free(mem_144114);
        free(mem_144133);
        free(mem_144140);
        free(mem_144147);
        free(mem_144152);
        free(mem_144182);
        free(mem_144188);
        free(mem_144193);
        free(mem_144209);
        free(mem_144210);
        free(mem_144219);
        free(mem_144220);
        free(mem_144241);
        free(mem_144247);
        free(mem_144252);
        free(mem_144268);
        free(mem_144273);
        free(mem_144284);
        free(mem_144289);
        free(mem_144300);
        free(mem_144305);
        free(mem_144317);
        free(mem_144326);
        free(mem_144327);
        free(mem_144348);
        free(mem_144353);
        free(mem_144364);
        free(mem_144365);
        free(mem_144378);
        free(mem_144385);
        free(mem_144390);
        free(mem_144401);
        free(mem_144407);
        free(mem_144412);
        free(mem_144428);
        free(mem_144429);
        free(mem_144430);
        free(mem_144446);
        free(mem_144447);
        free(mem_144448);
        free(mem_144461);
        free(mem_144462);
        free(mem_144503);
        free(mem_144504);
        free(mem_144515);
        free(mem_144516);
        free(mem_144525);
        free(mem_144526);
        free(mem_144557);
        free(mem_144558);
        free(mem_144569);
        free(mem_144570);
        free(mem_144579);
        free(mem_144580);
        free(mem_144611);
        free(mem_144612);
        free(mem_144613);
        free(mem_144614);
        free(mem_144631);
        free(mem_144632);
        free(mem_144633);
        free(mem_144634);
        free(mem_144675);
        free(mem_144676);
        free(mem_144687);
        free(mem_144688);
        free(mem_144697);
        free(mem_144698);
        free(mem_144729);
        free(mem_144730);
        free(mem_144739);
        free(mem_144740);
        free(mem_144761);
        free(mem_144762);
        free(mem_144773);
        free(mem_144774);
        free(mem_144783);
        free(mem_144784);
        free(mem_144815);
        free(mem_144816);
        free(mem_144827);
        free(mem_144828);
        free(mem_144837);
        free(mem_144838);
        free(mem_144870);
        free(mem_144871);
        free(mem_144872);
        free(mem_144889);
        free(mem_144890);
        free(mem_144891);
        free(mem_144892);
        free(mem_144933);
        free(mem_144938);
        free(mem_144952);
        free(mem_144953);
        free(mem_144972);
        free(mem_144973);
        free(mem_144974);
        free(mem_145011);
        free(mem_145018);
        free(mem_145023);
        free(mem_145034);
        free(mem_145035);
        free(mem_145044);
        free(mem_145045);
        free(mem_145066);
        free(mem_145067);
        free(mem_145068);
        free(mem_145069);
        free(mem_145094);
        free(mem_145095);
        free(mem_145108);
        free(mem_145118);
        free(mem_145119);
        free(mem_145145);
        free(mem_145166);
        free(mem_145167);
        if (memblock_unref(ctx, &mem_145157, "mem_145157") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145156, "mem_145156") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145140, "mem_145140") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145109, "mem_145109") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_144951, "mem_144951") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_144950, "mem_144950") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_144949, "mem_144949") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_144869, "mem_144869") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_144316, "mem_144316") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145214, "mem_out_145214") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145213, "mem_out_145213") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145212, "mem_out_145212") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145211, "mem_out_145211") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145210, "mem_out_145210") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145209, "mem_out_145209") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145208, "mem_out_145208") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145207, "mem_out_145207") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145206, "mem_out_145206") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_make_params(struct futhark_context *ctx, struct memblock *mem_out_p_145919, struct memblock *mem_out_p_145920, struct memblock *mem_out_p_145921, struct memblock *mem_out_p_145922, struct memblock *mem_out_p_145923, struct memblock *mem_out_p_145924, struct memblock *mem_out_p_145925, struct memblock *mem_out_p_145926, struct memblock *mem_out_p_145927, struct memblock wte_mem_143121, struct memblock wpe_mem_143122, struct memblock wqry_mem_143123, struct memblock wkey_mem_143124, struct memblock wval_mem_143125, struct memblock wout_mem_143126, struct memblock wup_mem_143127, struct memblock wdown_mem_143128, struct memblock wvoc_mem_143129, int64_t sl_56320)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_145214;
    
    mem_out_145214.references = NULL;
    
    struct memblock mem_out_145213;
    
    mem_out_145213.references = NULL;
    
    struct memblock mem_out_145212;
    
    mem_out_145212.references = NULL;
    
    struct memblock mem_out_145211;
    
    mem_out_145211.references = NULL;
    
    struct memblock mem_out_145210;
    
    mem_out_145210.references = NULL;
    
    struct memblock mem_out_145209;
    
    mem_out_145209.references = NULL;
    
    struct memblock mem_out_145208;
    
    mem_out_145208.references = NULL;
    
    struct memblock mem_out_145207;
    
    mem_out_145207.references = NULL;
    
    struct memblock mem_out_145206;
    
    mem_out_145206.references = NULL;
    if (memblock_set(ctx, &mem_out_145206, &wdown_mem_143128, "wdown_mem_143128") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_145207, &wkey_mem_143124, "wkey_mem_143124") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_145208, &wout_mem_143126, "wout_mem_143126") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_145209, &wpe_mem_143122, "wpe_mem_143122") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_145210, &wqry_mem_143123, "wqry_mem_143123") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_145211, &wte_mem_143121, "wte_mem_143121") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_145212, &wup_mem_143127, "wup_mem_143127") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_145213, &wval_mem_143125, "wval_mem_143125") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_145214, &wvoc_mem_143129, "wvoc_mem_143129") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145919, &mem_out_145206, "mem_out_145206") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145920, &mem_out_145207, "mem_out_145207") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145921, &mem_out_145208, "mem_out_145208") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145922, &mem_out_145209, "mem_out_145209") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145923, &mem_out_145210, "mem_out_145210") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145924, &mem_out_145211, "mem_out_145211") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145925, &mem_out_145212, "mem_out_145212") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145926, &mem_out_145213, "mem_out_145213") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_145927, &mem_out_145214, "mem_out_145214") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_145214, "mem_out_145214") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145213, "mem_out_145213") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145212, "mem_out_145212") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145211, "mem_out_145211") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145210, "mem_out_145210") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145209, "mem_out_145209") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145208, "mem_out_145208") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145207, "mem_out_145207") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_145206, "mem_out_145206") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_cal_loss(struct futhark_context *ctx, struct futhark_opaque_tup2_f64_arr1d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3)
{
    double prim_out_145207 = 0.0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_145206;
    
    mem_out_145206.references = NULL;
    
    struct memblock mask_mem_143132;
    
    mask_mem_143132.references = NULL;
    
    struct memblock target_mem_143131;
    
    target_mem_143131.references = NULL;
    
    struct memblock tokens_mem_143130;
    
    tokens_mem_143130.references = NULL;
    
    struct memblock wvoc_mem_143129;
    
    wvoc_mem_143129.references = NULL;
    
    struct memblock wval_mem_143128;
    
    wval_mem_143128.references = NULL;
    
    struct memblock wup_mem_143127;
    
    wup_mem_143127.references = NULL;
    
    struct memblock wte_mem_143126;
    
    wte_mem_143126.references = NULL;
    
    struct memblock wqry_mem_143125;
    
    wqry_mem_143125.references = NULL;
    
    struct memblock wpe_mem_143124;
    
    wpe_mem_143124.references = NULL;
    
    struct memblock wout_mem_143123;
    
    wout_mem_143123.references = NULL;
    
    struct memblock wkey_mem_143122;
    
    wkey_mem_143122.references = NULL;
    
    struct memblock wdown_mem_143121;
    
    wdown_mem_143121.references = NULL;
    wdown_mem_143121 = in0->v0->mem;
    wkey_mem_143122 = in0->v1->mem;
    wout_mem_143123 = in0->v2->mem;
    wpe_mem_143124 = in0->v3->mem;
    wqry_mem_143125 = in0->v4->mem;
    wte_mem_143126 = in0->v5->mem;
    wup_mem_143127 = in0->v6->mem;
    wval_mem_143128 = in0->v7->mem;
    wvoc_mem_143129 = in0->v8->mem;
    tokens_mem_143130 = in1->mem;
    target_mem_143131 = in2->mem;
    mask_mem_143132 = in3->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && (((int64_t) 16 == in2->shape[0] && (int64_t) 27 == in2->shape[1]) && ((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_cal_loss(ctx, &mem_out_145206, &prim_out_145207, wdown_mem_143121, wkey_mem_143122, wout_mem_143123, wpe_mem_143124, wqry_mem_143125, wte_mem_143126, wup_mem_143127, wval_mem_143128, wvoc_mem_143129, tokens_mem_143130, target_mem_143131, mask_mem_143132);
        if (ret == 0) {
            assert((*out = (struct futhark_opaque_tup2_f64_arr1d_f64 *) malloc(sizeof(struct futhark_opaque_tup2_f64_arr1d_f64))) != NULL);
            (*out)->v0 = prim_out_145207;
            assert(((*out)->v1 = (struct futhark_f64_1d *) malloc(sizeof(struct futhark_f64_1d))) != NULL);
            (*out)->v1->mem = mem_out_145206;
            (*out)->v1->shape[0] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_forward_seq(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_145206;
    
    mem_out_145206.references = NULL;
    
    struct memblock mask_mem_143131;
    
    mask_mem_143131.references = NULL;
    
    struct memblock tokens_mem_143130;
    
    tokens_mem_143130.references = NULL;
    
    struct memblock wvoc_mem_143129;
    
    wvoc_mem_143129.references = NULL;
    
    struct memblock wval_mem_143128;
    
    wval_mem_143128.references = NULL;
    
    struct memblock wup_mem_143127;
    
    wup_mem_143127.references = NULL;
    
    struct memblock wte_mem_143126;
    
    wte_mem_143126.references = NULL;
    
    struct memblock wqry_mem_143125;
    
    wqry_mem_143125.references = NULL;
    
    struct memblock wpe_mem_143124;
    
    wpe_mem_143124.references = NULL;
    
    struct memblock wout_mem_143123;
    
    wout_mem_143123.references = NULL;
    
    struct memblock wkey_mem_143122;
    
    wkey_mem_143122.references = NULL;
    
    struct memblock wdown_mem_143121;
    
    wdown_mem_143121.references = NULL;
    wdown_mem_143121 = in0->v0->mem;
    wkey_mem_143122 = in0->v1->mem;
    wout_mem_143123 = in0->v2->mem;
    wpe_mem_143124 = in0->v3->mem;
    wqry_mem_143125 = in0->v4->mem;
    wte_mem_143126 = in0->v5->mem;
    wup_mem_143127 = in0->v6->mem;
    wval_mem_143128 = in0->v7->mem;
    wvoc_mem_143129 = in0->v8->mem;
    tokens_mem_143130 = in1->mem;
    mask_mem_143131 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_145206, wdown_mem_143121, wkey_mem_143122, wout_mem_143123, wpe_mem_143124, wqry_mem_143125, wte_mem_143126, wup_mem_143127, wval_mem_143128, wvoc_mem_143129, tokens_mem_143130, mask_mem_143131);
        if (ret == 0) {
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_145206;
            (*out)->shape[0] = (int64_t) 16;
            (*out)->shape[1] = (int64_t) 27;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_grad_loss(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_145214;
    
    mem_out_145214.references = NULL;
    
    struct memblock mem_out_145213;
    
    mem_out_145213.references = NULL;
    
    struct memblock mem_out_145212;
    
    mem_out_145212.references = NULL;
    
    struct memblock mem_out_145211;
    
    mem_out_145211.references = NULL;
    
    struct memblock mem_out_145210;
    
    mem_out_145210.references = NULL;
    
    struct memblock mem_out_145209;
    
    mem_out_145209.references = NULL;
    
    struct memblock mem_out_145208;
    
    mem_out_145208.references = NULL;
    
    struct memblock mem_out_145207;
    
    mem_out_145207.references = NULL;
    
    struct memblock mem_out_145206;
    
    mem_out_145206.references = NULL;
    
    struct memblock mask_mem_143132;
    
    mask_mem_143132.references = NULL;
    
    struct memblock target_mem_143131;
    
    target_mem_143131.references = NULL;
    
    struct memblock tokens_mem_143130;
    
    tokens_mem_143130.references = NULL;
    
    struct memblock wvoc_mem_143129;
    
    wvoc_mem_143129.references = NULL;
    
    struct memblock wval_mem_143128;
    
    wval_mem_143128.references = NULL;
    
    struct memblock wup_mem_143127;
    
    wup_mem_143127.references = NULL;
    
    struct memblock wte_mem_143126;
    
    wte_mem_143126.references = NULL;
    
    struct memblock wqry_mem_143125;
    
    wqry_mem_143125.references = NULL;
    
    struct memblock wpe_mem_143124;
    
    wpe_mem_143124.references = NULL;
    
    struct memblock wout_mem_143123;
    
    wout_mem_143123.references = NULL;
    
    struct memblock wkey_mem_143122;
    
    wkey_mem_143122.references = NULL;
    
    struct memblock wdown_mem_143121;
    
    wdown_mem_143121.references = NULL;
    wdown_mem_143121 = in0->v0->mem;
    wkey_mem_143122 = in0->v1->mem;
    wout_mem_143123 = in0->v2->mem;
    wpe_mem_143124 = in0->v3->mem;
    wqry_mem_143125 = in0->v4->mem;
    wte_mem_143126 = in0->v5->mem;
    wup_mem_143127 = in0->v6->mem;
    wval_mem_143128 = in0->v7->mem;
    wvoc_mem_143129 = in0->v8->mem;
    tokens_mem_143130 = in1->mem;
    target_mem_143131 = in2->mem;
    mask_mem_143132 = in3->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && (((int64_t) 16 == in2->shape[0] && (int64_t) 27 == in2->shape[1]) && ((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_grad_loss(ctx, &mem_out_145206, &mem_out_145207, &mem_out_145208, &mem_out_145209, &mem_out_145210, &mem_out_145211, &mem_out_145212, &mem_out_145213, &mem_out_145214, wdown_mem_143121, wkey_mem_143122, wout_mem_143123, wpe_mem_143124, wqry_mem_143125, wte_mem_143126, wup_mem_143127, wval_mem_143128, wvoc_mem_143129, tokens_mem_143130, target_mem_143131, mask_mem_143132);
        if (ret == 0) {
            assert((*out = (struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_145206;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_145207;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_145208;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_145209;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_145210;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_145211;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_145212;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_145213;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_145214;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_make_params(struct futhark_context *ctx, struct futhark_opaque_params **out, const struct futhark_f64_2d *in0, const struct futhark_f64_2d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3, const struct futhark_f64_2d *in4, const struct futhark_f64_2d *in5, const struct futhark_f64_2d *in6, const struct futhark_f64_2d *in7, const struct futhark_f64_2d *in8)
{
    int64_t sl_56320 = (int64_t) 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_145214;
    
    mem_out_145214.references = NULL;
    
    struct memblock mem_out_145213;
    
    mem_out_145213.references = NULL;
    
    struct memblock mem_out_145212;
    
    mem_out_145212.references = NULL;
    
    struct memblock mem_out_145211;
    
    mem_out_145211.references = NULL;
    
    struct memblock mem_out_145210;
    
    mem_out_145210.references = NULL;
    
    struct memblock mem_out_145209;
    
    mem_out_145209.references = NULL;
    
    struct memblock mem_out_145208;
    
    mem_out_145208.references = NULL;
    
    struct memblock mem_out_145207;
    
    mem_out_145207.references = NULL;
    
    struct memblock mem_out_145206;
    
    mem_out_145206.references = NULL;
    
    struct memblock wvoc_mem_143129;
    
    wvoc_mem_143129.references = NULL;
    
    struct memblock wdown_mem_143128;
    
    wdown_mem_143128.references = NULL;
    
    struct memblock wup_mem_143127;
    
    wup_mem_143127.references = NULL;
    
    struct memblock wout_mem_143126;
    
    wout_mem_143126.references = NULL;
    
    struct memblock wval_mem_143125;
    
    wval_mem_143125.references = NULL;
    
    struct memblock wkey_mem_143124;
    
    wkey_mem_143124.references = NULL;
    
    struct memblock wqry_mem_143123;
    
    wqry_mem_143123.references = NULL;
    
    struct memblock wpe_mem_143122;
    
    wpe_mem_143122.references = NULL;
    
    struct memblock wte_mem_143121;
    
    wte_mem_143121.references = NULL;
    wte_mem_143121 = in0->mem;
    sl_56320 = in0->shape[1];
    wpe_mem_143122 = in1->mem;
    sl_56320 = in1->shape[0];
    wqry_mem_143123 = in2->mem;
    wkey_mem_143124 = in3->mem;
    wval_mem_143125 = in4->mem;
    wout_mem_143126 = in5->mem;
    wup_mem_143127 = in6->mem;
    wdown_mem_143128 = in7->mem;
    wvoc_mem_143129 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && sl_56320 == in0->shape[1]) && ((sl_56320 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_make_params(ctx, &mem_out_145206, &mem_out_145207, &mem_out_145208, &mem_out_145209, &mem_out_145210, &mem_out_145211, &mem_out_145212, &mem_out_145213, &mem_out_145214, wte_mem_143121, wpe_mem_143122, wqry_mem_143123, wkey_mem_143124, wval_mem_143125, wout_mem_143126, wup_mem_143127, wdown_mem_143128, wvoc_mem_143129, sl_56320);
        if (ret == 0) {
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_145206;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_145207;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_145208;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_145209;
            (*out)->v3->shape[0] = sl_56320;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_145210;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_145211;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = sl_56320;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_145212;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_145213;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_145214;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
