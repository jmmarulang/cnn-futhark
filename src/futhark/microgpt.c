
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

FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_141905, double *out_prim_out_141906, struct memblock wdown_mem_139491, struct memblock wkey_mem_139492, struct memblock wout_mem_139493, struct memblock wpe_mem_139494, struct memblock wqry_mem_139495, struct memblock wte_mem_139496, struct memblock wup_mem_139497, struct memblock wval_mem_139498, struct memblock wvoc_mem_139499, struct memblock tokens_mem_139500, struct memblock target_mem_139501, struct memblock mask_mem_139502);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_141964, struct memblock wdown_mem_139491, struct memblock wkey_mem_139492, struct memblock wout_mem_139493, struct memblock wpe_mem_139494, struct memblock wqry_mem_139495, struct memblock wte_mem_139496, struct memblock wup_mem_139497, struct memblock wval_mem_139498, struct memblock wvoc_mem_139499, struct memblock tokens_mem_139500, struct memblock mask_mem_139501);
FUTHARK_FUN_ATTR int futrts_entry_grad_loss(struct futhark_context *ctx, struct memblock *mem_out_p_142021, struct memblock *mem_out_p_142022, struct memblock *mem_out_p_142023, struct memblock *mem_out_p_142024, struct memblock *mem_out_p_142025, struct memblock *mem_out_p_142026, struct memblock *mem_out_p_142027, struct memblock *mem_out_p_142028, struct memblock *mem_out_p_142029, struct memblock wdown_mem_139491, struct memblock wkey_mem_139492, struct memblock wout_mem_139493, struct memblock wpe_mem_139494, struct memblock wqry_mem_139495, struct memblock wte_mem_139496, struct memblock wup_mem_139497, struct memblock wval_mem_139498, struct memblock wvoc_mem_139499, struct memblock tokens_mem_139500, struct memblock target_mem_139501, struct memblock mask_mem_139502);
FUTHARK_FUN_ATTR int futrts_entry_make_params(struct futhark_context *ctx, struct memblock *mem_out_p_142266, struct memblock *mem_out_p_142267, struct memblock *mem_out_p_142268, struct memblock *mem_out_p_142269, struct memblock *mem_out_p_142270, struct memblock *mem_out_p_142271, struct memblock *mem_out_p_142272, struct memblock *mem_out_p_142273, struct memblock *mem_out_p_142274, struct memblock wte_mem_139491, struct memblock wpe_mem_139492, struct memblock wqry_mem_139493, struct memblock wkey_mem_139494, struct memblock wval_mem_139495, struct memblock wout_mem_139496, struct memblock wup_mem_139497, struct memblock wdown_mem_139498, struct memblock wvoc_mem_139499, int64_t sl_54536);

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

FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_141905, double *out_prim_out_141906, struct memblock wdown_mem_139491, struct memblock wkey_mem_139492, struct memblock wout_mem_139493, struct memblock wpe_mem_139494, struct memblock wqry_mem_139495, struct memblock wte_mem_139496, struct memblock wup_mem_139497, struct memblock wval_mem_139498, struct memblock wvoc_mem_139499, struct memblock tokens_mem_139500, struct memblock target_mem_139501, struct memblock mask_mem_139502)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_139503_cached_sizze_141907 = 0;
    unsigned char *mem_139503 = NULL;
    int64_t mem_139508_cached_sizze_141908 = 0;
    unsigned char *mem_139508 = NULL;
    int64_t mem_139519_cached_sizze_141909 = 0;
    unsigned char *mem_139519 = NULL;
    int64_t mem_139524_cached_sizze_141910 = 0;
    unsigned char *mem_139524 = NULL;
    int64_t mem_139531_cached_sizze_141911 = 0;
    unsigned char *mem_139531 = NULL;
    int64_t mem_139542_cached_sizze_141912 = 0;
    unsigned char *mem_139542 = NULL;
    int64_t mem_139547_cached_sizze_141913 = 0;
    unsigned char *mem_139547 = NULL;
    int64_t mem_139554_cached_sizze_141914 = 0;
    unsigned char *mem_139554 = NULL;
    int64_t mem_139565_cached_sizze_141915 = 0;
    unsigned char *mem_139565 = NULL;
    int64_t mem_139566_cached_sizze_141916 = 0;
    unsigned char *mem_139566 = NULL;
    int64_t mem_139567_cached_sizze_141917 = 0;
    unsigned char *mem_139567 = NULL;
    int64_t mem_139580_cached_sizze_141918 = 0;
    unsigned char *mem_139580 = NULL;
    int64_t mem_139581_cached_sizze_141919 = 0;
    unsigned char *mem_139581 = NULL;
    int64_t mem_139582_cached_sizze_141920 = 0;
    unsigned char *mem_139582 = NULL;
    int64_t mem_139613_cached_sizze_141921 = 0;
    unsigned char *mem_139613 = NULL;
    int64_t mem_139614_cached_sizze_141922 = 0;
    unsigned char *mem_139614 = NULL;
    int64_t mem_139615_cached_sizze_141923 = 0;
    unsigned char *mem_139615 = NULL;
    int64_t mem_139631_cached_sizze_141924 = 0;
    unsigned char *mem_139631 = NULL;
    int64_t mem_139632_cached_sizze_141925 = 0;
    unsigned char *mem_139632 = NULL;
    int64_t mem_139633_cached_sizze_141926 = 0;
    unsigned char *mem_139633 = NULL;
    int64_t mem_139646_cached_sizze_141927 = 0;
    unsigned char *mem_139646 = NULL;
    int64_t mem_139647_cached_sizze_141928 = 0;
    unsigned char *mem_139647 = NULL;
    int64_t mem_139648_cached_sizze_141929 = 0;
    unsigned char *mem_139648 = NULL;
    int64_t mem_139694_cached_sizze_141930 = 0;
    unsigned char *mem_139694 = NULL;
    int64_t mem_139700_cached_sizze_141931 = 0;
    unsigned char *mem_139700 = NULL;
    int64_t mem_139705_cached_sizze_141932 = 0;
    unsigned char *mem_139705 = NULL;
    int64_t mem_139716_cached_sizze_141933 = 0;
    unsigned char *mem_139716 = NULL;
    int64_t mem_139721_cached_sizze_141934 = 0;
    unsigned char *mem_139721 = NULL;
    int64_t mem_139732_cached_sizze_141935 = 0;
    unsigned char *mem_139732 = NULL;
    int64_t mem_139737_cached_sizze_141936 = 0;
    unsigned char *mem_139737 = NULL;
    int64_t mem_139744_cached_sizze_141937 = 0;
    unsigned char *mem_139744 = NULL;
    int64_t mem_139751_cached_sizze_141938 = 0;
    unsigned char *mem_139751 = NULL;
    int64_t mem_139762_cached_sizze_141939 = 0;
    unsigned char *mem_139762 = NULL;
    int64_t mem_139767_cached_sizze_141940 = 0;
    unsigned char *mem_139767 = NULL;
    int64_t mem_139778_cached_sizze_141941 = 0;
    unsigned char *mem_139778 = NULL;
    int64_t mem_139783_cached_sizze_141942 = 0;
    unsigned char *mem_139783 = NULL;
    int64_t mem_139799_cached_sizze_141943 = 0;
    unsigned char *mem_139799 = NULL;
    int64_t mem_139804_cached_sizze_141944 = 0;
    unsigned char *mem_139804 = NULL;
    int64_t mem_139815_cached_sizze_141945 = 0;
    unsigned char *mem_139815 = NULL;
    int64_t mem_139820_cached_sizze_141946 = 0;
    unsigned char *mem_139820 = NULL;
    int64_t mem_139831_cached_sizze_141947 = 0;
    unsigned char *mem_139831 = NULL;
    int64_t mem_139836_cached_sizze_141948 = 0;
    unsigned char *mem_139836 = NULL;
    int64_t mem_139847_cached_sizze_141949 = 0;
    unsigned char *mem_139847 = NULL;
    int64_t mem_139852_cached_sizze_141950 = 0;
    unsigned char *mem_139852 = NULL;
    int64_t mem_139859_cached_sizze_141951 = 0;
    unsigned char *mem_139859 = NULL;
    int64_t mem_139870_cached_sizze_141952 = 0;
    unsigned char *mem_139870 = NULL;
    int64_t mem_139875_cached_sizze_141953 = 0;
    unsigned char *mem_139875 = NULL;
    int64_t mem_139886_cached_sizze_141954 = 0;
    unsigned char *mem_139886 = NULL;
    int64_t mem_139891_cached_sizze_141955 = 0;
    unsigned char *mem_139891 = NULL;
    int64_t mem_139902_cached_sizze_141956 = 0;
    unsigned char *mem_139902 = NULL;
    int64_t mem_139907_cached_sizze_141957 = 0;
    unsigned char *mem_139907 = NULL;
    int64_t mem_139918_cached_sizze_141958 = 0;
    unsigned char *mem_139918 = NULL;
    int64_t mem_139923_cached_sizze_141959 = 0;
    unsigned char *mem_139923 = NULL;
    int64_t mem_139934_cached_sizze_141960 = 0;
    unsigned char *mem_139934 = NULL;
    int64_t mem_139939_cached_sizze_141961 = 0;
    unsigned char *mem_139939 = NULL;
    int64_t mem_139954_cached_sizze_141962 = 0;
    unsigned char *mem_139954 = NULL;
    int64_t mem_139961_cached_sizze_141963 = 0;
    unsigned char *mem_139961 = NULL;
    struct memblock mem_139950;
    
    mem_139950.references = NULL;
    
    struct memblock mem_out_141576;
    
    mem_out_141576.references = NULL;
    
    double prim_out_141577;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_139503_cached_sizze_141907 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139503, &mem_139503_cached_sizze_141907, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139508_cached_sizze_141908 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139508, &mem_139508_cached_sizze_141908, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138363 = 0; i_138363 < (int64_t) 16; i_138363++) {
        // futhark/microgpt.fut:441:41-50
        
        int64_t tmp_127870 = ((int64_t *) tokens_mem_139500.mem)[i_138363];
        
        // futhark/microgpt.fut:441:37-51
        
        bool x_127871 = sle64((int64_t) 0, tmp_127870);
        
        // futhark/microgpt.fut:441:37-51
        
        bool y_127872 = slt64(tmp_127870, (int64_t) 27);
        
        // futhark/microgpt.fut:441:37-51
        
        bool bounds_check_127873 = x_127871 && y_127872;
        
        // futhark/microgpt.fut:441:37-51
        
        bool index_certs_127874;
        
        if (!bounds_check_127873) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_127870, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:441:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:441:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138359 = 0; i_138359 < (int64_t) 16; i_138359++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_127881 = ((double *) wte_mem_139496.mem)[tmp_127870 * (int64_t) 16 + i_138359];
            
            ((double *) mem_139508)[i_138359] = lifted_lambda_res_127881;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139503, i_138363 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139508, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139519_cached_sizze_141909 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139519, &mem_139519_cached_sizze_141909, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139524_cached_sizze_141910 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139524, &mem_139524_cached_sizze_141910, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139531_cached_sizze_141911 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139531, &mem_139531_cached_sizze_141911, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138375 = 0; i_138375 < (int64_t) 16; i_138375++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_127907;
        double r_127909 = 0.0;
        
        for (int64_t i_127908 = 0; i_127908 < (int64_t) 16; i_127908++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_127910 = ((double *) wpe_mem_139494.mem)[i_138375 * (int64_t) 16 + i_127908];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_127911 = ((double *) mem_139503)[i_138375 * (int64_t) 16 + i_127908];
            
            // futhark/microgpt.fut:193:76-116
            
            double zp_res_127912 = zp_lhs_127910 + zp_rhs_127911;
            
            // futhark/microgpt.fut:193:94-163
            
            double zt_res_127913 = zp_res_127912 * zp_res_127912;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_127914 = r_127909 + zt_res_127913;
            double r_tmp_141581 = zp_res_127914;
            
            r_127909 = r_tmp_141581;
        }
        defunc_0_lifted_lambda_res_127907 = r_127909;
        // futhark/microgpt.fut:193:54-182
        
        double zs_res_127915 = defunc_0_lifted_lambda_res_127907 / 16.0;
        
        // futhark/microgpt.fut:194:24-55
        
        double zp_res_127916 = 1.0e-5 + zs_res_127915;
        
        // futhark/microgpt.fut:194:16-55
        
        double sqrt_res_127917 = futrts_sqrt64(zp_res_127916);
        
        // futhark/microgpt.fut:195:85-96
        
        double zs_res_127918 = 1.0 / sqrt_res_127917;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138367 = 0; i_138367 < (int64_t) 16; i_138367++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_127925 = ((double *) wpe_mem_139494.mem)[i_138375 * (int64_t) 16 + i_138367];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_127926 = ((double *) mem_139503)[i_138375 * (int64_t) 16 + i_138367];
            
            // futhark/microgpt.fut:195:38-78
            
            double zp_res_127927 = zp_lhs_127925 + zp_rhs_127926;
            
            // futhark/microgpt.fut:195:56-96
            
            double zt_res_127928 = zs_res_127918 * zp_res_127927;
            
            ((double *) mem_139524)[i_138367] = zt_res_127928;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138371 = 0; i_138371 < (int64_t) 16; i_138371++) {
            // futhark/microgpt.fut:196:4-14
            
            double lifted_lambda_res_127936 = ((double *) mem_139524)[i_138371];
            
            ((double *) mem_139531)[i_138371] = lifted_lambda_res_127936;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139519, i_138375 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139531, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139542_cached_sizze_141912 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139542, &mem_139542_cached_sizze_141912, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139547_cached_sizze_141913 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139547, &mem_139547_cached_sizze_141913, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139554_cached_sizze_141914 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139554, &mem_139554_cached_sizze_141914, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138387 = 0; i_138387 < (int64_t) 16; i_138387++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_127945;
        double r_127947 = 0.0;
        
        for (int64_t i_127946 = 0; i_127946 < (int64_t) 16; i_127946++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_127948 = ((double *) mem_139519)[i_138387 * (int64_t) 16 + i_127946];
            
            // futhark/microgpt.fut:197:78-115
            
            double zt_res_127949 = zt_lhs_127948 * zt_lhs_127948;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_127950 = r_127947 + zt_res_127949;
            double r_tmp_141585 = zp_res_127950;
            
            r_127947 = r_tmp_141585;
        }
        defunc_0_lifted_lambda_res_127945 = r_127947;
        // futhark/microgpt.fut:197:57-133
        
        double zs_res_127951 = defunc_0_lifted_lambda_res_127945 / 16.0;
        
        // futhark/microgpt.fut:198:24-55
        
        double zp_res_127952 = 1.0e-5 + zs_res_127951;
        
        // futhark/microgpt.fut:198:16-55
        
        double sqrt_res_127953 = futrts_sqrt64(zp_res_127952);
        
        // futhark/microgpt.fut:199:59-70
        
        double zs_res_127954 = 1.0 / sqrt_res_127953;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138379 = 0; i_138379 < (int64_t) 16; i_138379++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_127961 = ((double *) mem_139519)[i_138387 * (int64_t) 16 + i_138379];
            
            // futhark/microgpt.fut:199:37-70
            
            double zt_res_127962 = zs_res_127954 * zt_lhs_127961;
            
            ((double *) mem_139547)[i_138379] = zt_res_127962;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138383 = 0; i_138383 < (int64_t) 16; i_138383++) {
            // futhark/microgpt.fut:200:4-14
            
            double lifted_lambda_res_127970 = ((double *) mem_139547)[i_138383];
            
            ((double *) mem_139554)[i_138383] = lifted_lambda_res_127970;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139542, i_138387 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139554, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139565_cached_sizze_141915 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139565, &mem_139565_cached_sizze_141915, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139566_cached_sizze_141916 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139566, &mem_139566_cached_sizze_141916, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139567_cached_sizze_141917 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139567, &mem_139567_cached_sizze_141917, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139580_cached_sizze_141918 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139580, &mem_139580_cached_sizze_141918, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139581_cached_sizze_141919 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139581, &mem_139581_cached_sizze_141919, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139582_cached_sizze_141920 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139582, &mem_139582_cached_sizze_141920, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138405 = 0; i_138405 < (int64_t) 16; i_138405++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138395 = 0; i_138395 < (int64_t) 16; i_138395++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128807;
            double r_128809 = 0.0;
            
            for (int64_t i_128808 = 0; i_128808 < (int64_t) 16; i_128808++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128810 = ((double *) wqry_mem_139495.mem)[i_138395 * (int64_t) 16 + i_128808];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128811 = ((double *) mem_139542)[i_138405 * (int64_t) 16 + i_128808];
                
                // futhark/microgpt.fut:201:66-105
                
                double zt_res_128812 = zt_lhs_128810 * zt_rhs_128811;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128813 = r_128809 + zt_res_128812;
                double r_tmp_141594 = zp_res_128813;
                
                r_128809 = r_tmp_141594;
            }
            defunc_0_lifted_lambda_res_128807 = r_128809;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128820;
            double r_128822 = 0.0;
            
            for (int64_t i_128821 = 0; i_128821 < (int64_t) 16; i_128821++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128823 = ((double *) wkey_mem_139492.mem)[i_138395 * (int64_t) 16 + i_128821];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128824 = ((double *) mem_139542)[i_138405 * (int64_t) 16 + i_128821];
                
                // futhark/microgpt.fut:202:66-105
                
                double zt_res_128825 = zt_lhs_128823 * zt_rhs_128824;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128826 = r_128822 + zt_res_128825;
                double r_tmp_141595 = zp_res_128826;
                
                r_128822 = r_tmp_141595;
            }
            defunc_0_lifted_lambda_res_128820 = r_128822;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128836;
            double r_128838 = 0.0;
            
            for (int64_t i_128837 = 0; i_128837 < (int64_t) 16; i_128837++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128839 = ((double *) wval_mem_139498.mem)[i_138395 * (int64_t) 16 + i_128837];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128840 = ((double *) mem_139542)[i_138405 * (int64_t) 16 + i_128837];
                
                // futhark/microgpt.fut:203:66-105
                
                double zt_res_128841 = zt_lhs_128839 * zt_rhs_128840;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128842 = r_128838 + zt_res_128841;
                double r_tmp_141596 = zp_res_128842;
                
                r_128838 = r_tmp_141596;
            }
            defunc_0_lifted_lambda_res_128836 = r_128838;
            ((double *) mem_139580)[i_138395] = defunc_0_lifted_lambda_res_128836;
            ((double *) mem_139581)[i_138395] = defunc_0_lifted_lambda_res_128820;
            ((double *) mem_139582)[i_138395] = defunc_0_lifted_lambda_res_128807;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139565, i_138405 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139580, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139566, i_138405 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139581, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139567, i_138405 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139582, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139613_cached_sizze_141921 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139613, &mem_139613_cached_sizze_141921, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139614_cached_sizze_141922 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139614, &mem_139614_cached_sizze_141922, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139615_cached_sizze_141923 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139615, &mem_139615_cached_sizze_141923, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139631_cached_sizze_141924 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139631, &mem_139631_cached_sizze_141924, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139632_cached_sizze_141925 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139632, &mem_139632_cached_sizze_141925, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139633_cached_sizze_141926 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139633, &mem_139633_cached_sizze_141926, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139646_cached_sizze_141927 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_139646, &mem_139646_cached_sizze_141927, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139647_cached_sizze_141928 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_139647, &mem_139647_cached_sizze_141928, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139648_cached_sizze_141929 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_139648, &mem_139648_cached_sizze_141929, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138435 = 0; i_138435 < (int64_t) 4; i_138435++) {
        // futhark/microgpt.fut:204:69-72
        
        int64_t zp_lhs_128683 = mul64((int64_t) 4, i_138435);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138425 = 0; i_138425 < (int64_t) 16; i_138425++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138415 = 0; i_138415 < (int64_t) 4; i_138415++) {
                // futhark/microgpt.fut:204:74-81
                
                int64_t tmp_129000 = add64(zp_lhs_128683, i_138415);
                
                // futhark/microgpt.fut:204:51-83
                
                bool x_129001 = sle64((int64_t) 0, tmp_129000);
                
                // futhark/microgpt.fut:204:51-83
                
                bool y_129002 = slt64(tmp_129000, (int64_t) 16);
                
                // futhark/microgpt.fut:204:51-83
                
                bool bounds_check_129003 = x_129001 && y_129002;
                
                // futhark/microgpt.fut:204:51-83
                
                bool index_certs_129004;
                
                if (!bounds_check_129003) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_129000, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:204:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:204:15-84\n   #9  futhark/microgpt.fut:442:7-76\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129005 = ((double *) mem_139567)[i_138425 * (int64_t) 16 + tmp_129000];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129013 = ((double *) mem_139566)[i_138425 * (int64_t) 16 + tmp_129000];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129024 = ((double *) mem_139565)[i_138425 * (int64_t) 16 + tmp_129000];
                
                ((double *) mem_139646)[i_138415] = lifted_lambda_res_129024;
                ((double *) mem_139647)[i_138415] = lifted_lambda_res_129013;
                ((double *) mem_139648)[i_138415] = lifted_lambda_res_129005;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139631, i_138425 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139646, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139632, i_138425 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139647, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139633, i_138425 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139648, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139613, i_138435 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_139631, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139614, i_138435 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_139632, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139615, i_138435 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_139633, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139694_cached_sizze_141930 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139694, &mem_139694_cached_sizze_141930, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139700_cached_sizze_141931 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139700, &mem_139700_cached_sizze_141931, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139705_cached_sizze_141932 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139705, &mem_139705_cached_sizze_141932, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139716_cached_sizze_141933 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139716, &mem_139716_cached_sizze_141933, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139721_cached_sizze_141934 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139721, &mem_139721_cached_sizze_141934, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139732_cached_sizze_141935 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139732, &mem_139732_cached_sizze_141935, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139737_cached_sizze_141936 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139737, &mem_139737_cached_sizze_141936, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139744_cached_sizze_141937 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139744, &mem_139744_cached_sizze_141937, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139751_cached_sizze_141938 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139751, &mem_139751_cached_sizze_141938, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139762_cached_sizze_141939 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139762, &mem_139762_cached_sizze_141939, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139767_cached_sizze_141940 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_139767, &mem_139767_cached_sizze_141940, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139778_cached_sizze_141941 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139778, &mem_139778_cached_sizze_141941, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139783_cached_sizze_141942 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_139783, &mem_139783_cached_sizze_141942, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138491 = 0; i_138491 < (int64_t) 4; i_138491++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138445 = 0; i_138445 < (int64_t) 16; i_138445++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138441 = 0; i_138441 < (int64_t) 16; i_138441++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_128115;
                double r_128117 = 0.0;
                
                for (int64_t i_128116 = 0; i_128116 < (int64_t) 4; i_128116++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_128118 = ((double *) mem_139615)[i_138491 * (int64_t) 64 + i_138445 * (int64_t) 4 + i_128116];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_128119 = ((double *) mem_139614)[i_138491 * (int64_t) 64 + i_138441 * (int64_t) 4 + i_128116];
                    
                    // futhark/microgpt.fut:207:113-164
                    
                    double zt_res_128120 = zt_lhs_128118 * zt_rhs_128119;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_128121 = r_128117 + zt_res_128120;
                    double r_tmp_141609 = zp_res_128121;
                    
                    r_128117 = r_tmp_141609;
                }
                defunc_0_lifted_lambda_res_128115 = r_128117;
                ((double *) mem_139705)[i_138441] = defunc_0_lifted_lambda_res_128115;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139700, i_138445 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139705, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138453 = 0; i_138453 < (int64_t) 16; i_138453++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138449 = 0; i_138449 < (int64_t) 16; i_138449++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_128136 = ((double *) mem_139700)[i_138453 * (int64_t) 16 + i_138449];
                
                // futhark/microgpt.fut:208:47-78
                
                double zs_res_128137 = zs_lhs_128136 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_128138 = ((double *) mask_mem_139502.mem)[i_138453 * (int64_t) 16 + i_138449];
                
                // futhark/microgpt.fut:208:65-102
                
                double zp_res_128139 = zs_res_128137 + zp_rhs_128138;
                
                ((double *) mem_139721)[i_138449] = zp_res_128139;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139716, i_138453 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139721, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138471 = 0; i_138471 < (int64_t) 16; i_138471++) {
            // futhark/microgpt.fut:105:13-33
            
            double defunc_0_reduce_res_129127;
            double redout_138455 = -INFINITY;
            
            for (int64_t i_138456 = 0; i_138456 < (int64_t) 16; i_138456++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129051 = ((double *) mem_139716)[i_138471 * (int64_t) 16 + i_138456];
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_128160 = fmax64(lifted_lambda_res_129051, redout_138455);
                double redout_tmp_141613 = max_res_128160;
                
                redout_138455 = redout_tmp_141613;
            }
            defunc_0_reduce_res_129127 = redout_138455;
            // futhark/microgpt.fut:210:67-76
            
            double neg_res_128161 = -defunc_0_reduce_res_129127;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138459 = 0; i_138459 < (int64_t) 16; i_138459++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_128168 = ((double *) mem_139716)[i_138471 * (int64_t) 16 + i_138459];
                
                // futhark/microgpt.fut:210:44-76
                
                double zp_res_128169 = neg_res_128161 + zp_lhs_128168;
                
                // futhark/microgpt.fut:210:37-76
                
                double exp_res_128170 = futrts_exp64(zp_res_128169);
                
                ((double *) mem_139737)[i_138459] = exp_res_128170;
            }
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128172;
            double r_128174 = 0.0;
            
            for (int64_t i_128173 = 0; i_128173 < (int64_t) 16; i_128173++) {
                // futhark/microgpt.fut:211:36-46
                
                double lifted_lambda_res_128175 = ((double *) mem_139737)[i_128173];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128176 = r_128174 + lifted_lambda_res_128175;
                double r_tmp_141615 = zp_res_128176;
                
                r_128174 = r_tmp_141615;
            }
            defunc_0_lifted_lambda_res_128172 = r_128174;
            // futhark/microgpt.fut:212:53-64
            
            double zs_res_128177 = 1.0 / defunc_0_lifted_lambda_res_128172;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138463 = 0; i_138463 < (int64_t) 16; i_138463++) {
                // futhark/microgpt.fut:212:37-47
                
                double zt_lhs_128184 = ((double *) mem_139737)[i_138463];
                
                // futhark/microgpt.fut:212:37-64
                
                double zt_res_128185 = zs_res_128177 * zt_lhs_128184;
                
                ((double *) mem_139744)[i_138463] = zt_res_128185;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138467 = 0; i_138467 < (int64_t) 16; i_138467++) {
                // futhark/microgpt.fut:213:4-14
                
                double lifted_lambda_res_128193 = ((double *) mem_139744)[i_138467];
                
                ((double *) mem_139751)[i_138467] = lifted_lambda_res_128193;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139732, i_138471 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139751, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138479 = 0; i_138479 < (int64_t) 16; i_138479++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138475 = 0; i_138475 < (int64_t) 4; i_138475++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_128208;
                double r_128210 = 0.0;
                
                for (int64_t i_128209 = 0; i_128209 < (int64_t) 16; i_128209++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_128211 = ((double *) mem_139732)[i_138479 * (int64_t) 16 + i_128209];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_128212 = ((double *) mem_139613)[i_138491 * (int64_t) 64 + i_128209 * (int64_t) 4 + i_138475];
                    
                    // futhark/microgpt.fut:214:66-111
                    
                    double zt_res_128213 = zt_lhs_128211 * zt_rhs_128212;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_128214 = r_128210 + zt_res_128213;
                    double r_tmp_141620 = zp_res_128214;
                    
                    r_128210 = r_tmp_141620;
                }
                defunc_0_lifted_lambda_res_128208 = r_128210;
                ((double *) mem_139767)[i_138475] = defunc_0_lifted_lambda_res_128208;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139762, i_138479 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139767, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138487 = 0; i_138487 < (int64_t) 16; i_138487++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138483 = 0; i_138483 < (int64_t) 4; i_138483++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_128229 = ((double *) mem_139762)[i_138487 * (int64_t) 4 + i_138483];
                
                ((double *) mem_139783)[i_138483] = lifted_lambda_res_128229;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139778, i_138487 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139783, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139694, i_138491 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_139778, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139799_cached_sizze_141943 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139799, &mem_139799_cached_sizze_141943, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139804_cached_sizze_141944 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139804, &mem_139804_cached_sizze_141944, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138499 = 0; i_138499 < (int64_t) 16; i_138499++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138495 = 0; i_138495 < (int64_t) 16; i_138495++) {
            // futhark/microgpt.fut:216:54-57
            
            int64_t tmp_128241 = sdiv64(i_138495, (int64_t) 4);
            
            // futhark/microgpt.fut:216:44-59
            
            bool x_128242 = sle64((int64_t) 0, tmp_128241);
            
            // futhark/microgpt.fut:216:44-59
            
            bool y_128243 = slt64(tmp_128241, (int64_t) 4);
            
            // futhark/microgpt.fut:216:44-59
            
            bool bounds_check_128244 = x_128242 && y_128243;
            
            // futhark/microgpt.fut:216:44-59
            
            bool index_certs_128245;
            
            if (!bounds_check_128244) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_128241, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:216:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:216:15-80\n   #6  futhark/microgpt.fut:442:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:216:74-77
            
            int64_t tmp_128246 = smod64(i_138495, (int64_t) 4);
            
            // futhark/microgpt.fut:216:44-79
            
            bool x_128247 = sle64((int64_t) 0, tmp_128246);
            
            // futhark/microgpt.fut:216:44-79
            
            bool y_128248 = slt64(tmp_128246, (int64_t) 4);
            
            // futhark/microgpt.fut:216:44-79
            
            bool bounds_check_128249 = x_128247 && y_128248;
            
            // futhark/microgpt.fut:216:44-79
            
            bool index_certs_128250;
            
            if (!bounds_check_128249) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_128246, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:216:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:216:15-80\n   #6  futhark/microgpt.fut:442:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_128251 = ((double *) mem_139694)[tmp_128241 * (int64_t) 64 + i_138499 * (int64_t) 4 + tmp_128246];
            
            ((double *) mem_139804)[i_138495] = lifted_lambda_res_128251;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139799, i_138499 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139804, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139815_cached_sizze_141945 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139815, &mem_139815_cached_sizze_141945, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139820_cached_sizze_141946 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139820, &mem_139820_cached_sizze_141946, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138507 = 0; i_138507 < (int64_t) 16; i_138507++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138503 = 0; i_138503 < (int64_t) 16; i_138503++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128266;
            double r_128268 = 0.0;
            
            for (int64_t i_128267 = 0; i_128267 < (int64_t) 16; i_128267++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128269 = ((double *) wout_mem_139493.mem)[i_138503 * (int64_t) 16 + i_128267];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128270 = ((double *) mem_139799)[i_138507 * (int64_t) 16 + i_128267];
                
                // futhark/microgpt.fut:217:67-106
                
                double zt_res_128271 = zt_lhs_128269 * zt_rhs_128270;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128272 = r_128268 + zt_res_128271;
                double r_tmp_141627 = zp_res_128272;
                
                r_128268 = r_tmp_141627;
            }
            defunc_0_lifted_lambda_res_128266 = r_128268;
            ((double *) mem_139820)[i_138503] = defunc_0_lifted_lambda_res_128266;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139815, i_138507 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139820, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139831_cached_sizze_141947 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139831, &mem_139831_cached_sizze_141947, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139836_cached_sizze_141948 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139836, &mem_139836_cached_sizze_141948, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138515 = 0; i_138515 < (int64_t) 16; i_138515++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138511 = 0; i_138511 < (int64_t) 16; i_138511++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_128287 = ((double *) mem_139815)[i_138515 * (int64_t) 16 + i_138511];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_128288 = ((double *) mem_139519)[i_138515 * (int64_t) 16 + i_138511];
            
            // futhark/microgpt.fut:218:46-84
            
            double zp_res_128289 = zp_lhs_128287 + zp_rhs_128288;
            
            ((double *) mem_139836)[i_138511] = zp_res_128289;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139831, i_138515 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139836, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139847_cached_sizze_141949 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139847, &mem_139847_cached_sizze_141949, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139852_cached_sizze_141950 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139852, &mem_139852_cached_sizze_141950, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139859_cached_sizze_141951 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139859, &mem_139859_cached_sizze_141951, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138527 = 0; i_138527 < (int64_t) 16; i_138527++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_128298;
        double r_128300 = 0.0;
        
        for (int64_t i_128299 = 0; i_128299 < (int64_t) 16; i_128299++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_128301 = ((double *) mem_139831)[i_138527 * (int64_t) 16 + i_128299];
            
            // futhark/microgpt.fut:219:79-118
            
            double zt_res_128302 = zt_lhs_128301 * zt_lhs_128301;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_128303 = r_128300 + zt_res_128302;
            double r_tmp_141631 = zp_res_128303;
            
            r_128300 = r_tmp_141631;
        }
        defunc_0_lifted_lambda_res_128298 = r_128300;
        // futhark/microgpt.fut:219:58-136
        
        double zs_res_128304 = defunc_0_lifted_lambda_res_128298 / 16.0;
        
        // futhark/microgpt.fut:220:24-55
        
        double zp_res_128305 = 1.0e-5 + zs_res_128304;
        
        // futhark/microgpt.fut:220:16-55
        
        double sqrt_res_128306 = futrts_sqrt64(zp_res_128305);
        
        // futhark/microgpt.fut:221:60-71
        
        double zs_res_128307 = 1.0 / sqrt_res_128306;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138519 = 0; i_138519 < (int64_t) 16; i_138519++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_128314 = ((double *) mem_139831)[i_138527 * (int64_t) 16 + i_138519];
            
            // futhark/microgpt.fut:221:37-71
            
            double zt_res_128315 = zs_res_128307 * zt_lhs_128314;
            
            ((double *) mem_139852)[i_138519] = zt_res_128315;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138523 = 0; i_138523 < (int64_t) 16; i_138523++) {
            // futhark/microgpt.fut:222:4-14
            
            double lifted_lambda_res_128323 = ((double *) mem_139852)[i_138523];
            
            ((double *) mem_139859)[i_138523] = lifted_lambda_res_128323;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139847, i_138527 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139859, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139870_cached_sizze_141952 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139870, &mem_139870_cached_sizze_141952, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139875_cached_sizze_141953 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139875, &mem_139875_cached_sizze_141953, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138535 = 0; i_138535 < (int64_t) 16; i_138535++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138531 = 0; i_138531 < (int64_t) 64; i_138531++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128339;
            double r_128341 = 0.0;
            
            for (int64_t i_128340 = 0; i_128340 < (int64_t) 16; i_128340++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128342 = ((double *) wup_mem_139497.mem)[i_138531 * (int64_t) 16 + i_128340];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128343 = ((double *) mem_139847)[i_138535 * (int64_t) 16 + i_128340];
                
                // futhark/microgpt.fut:223:67-106
                
                double zt_res_128344 = zt_lhs_128342 * zt_rhs_128343;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128345 = r_128341 + zt_res_128344;
                double r_tmp_141636 = zp_res_128345;
                
                r_128341 = r_tmp_141636;
            }
            defunc_0_lifted_lambda_res_128339 = r_128341;
            ((double *) mem_139875)[i_138531] = defunc_0_lifted_lambda_res_128339;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139870, i_138535 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139875, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139886_cached_sizze_141954 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139886, &mem_139886_cached_sizze_141954, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139891_cached_sizze_141955 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139891, &mem_139891_cached_sizze_141955, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138543 = 0; i_138543 < (int64_t) 16; i_138543++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138539 = 0; i_138539 < (int64_t) 64; i_138539++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_128360 = ((double *) mem_139870)[i_138543 * (int64_t) 64 + i_138539];
            
            // futhark/microgpt.fut:224:45-73
            
            double max_res_128361 = fmax64(0.0, max_arg0_128360);
            
            ((double *) mem_139891)[i_138539] = max_res_128361;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139886, i_138543 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139891, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139902_cached_sizze_141956 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139902, &mem_139902_cached_sizze_141956, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139907_cached_sizze_141957 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139907, &mem_139907_cached_sizze_141957, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138551 = 0; i_138551 < (int64_t) 16; i_138551++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138547 = 0; i_138547 < (int64_t) 16; i_138547++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128376;
            double r_128378 = 0.0;
            
            for (int64_t i_128377 = 0; i_128377 < (int64_t) 64; i_128377++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128379 = ((double *) wdown_mem_139491.mem)[i_138547 * (int64_t) 64 + i_128377];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128380 = ((double *) mem_139886)[i_138551 * (int64_t) 64 + i_128377];
                
                // futhark/microgpt.fut:225:67-108
                
                double zt_res_128381 = zt_lhs_128379 * zt_rhs_128380;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128382 = r_128378 + zt_res_128381;
                double r_tmp_141641 = zp_res_128382;
                
                r_128378 = r_tmp_141641;
            }
            defunc_0_lifted_lambda_res_128376 = r_128378;
            ((double *) mem_139907)[i_138547] = defunc_0_lifted_lambda_res_128376;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139902, i_138551 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139907, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139918_cached_sizze_141958 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139918, &mem_139918_cached_sizze_141958, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139923_cached_sizze_141959 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139923, &mem_139923_cached_sizze_141959, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138559 = 0; i_138559 < (int64_t) 16; i_138559++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138555 = 0; i_138555 < (int64_t) 16; i_138555++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_128397 = ((double *) mem_139902)[i_138559 * (int64_t) 16 + i_138555];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_128398 = ((double *) mem_139831)[i_138559 * (int64_t) 16 + i_138555];
            
            // futhark/microgpt.fut:226:46-85
            
            double zp_res_128399 = zp_lhs_128397 + zp_rhs_128398;
            
            ((double *) mem_139923)[i_138555] = zp_res_128399;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139918, i_138559 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139923, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139934_cached_sizze_141960 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_139934, &mem_139934_cached_sizze_141960, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139939_cached_sizze_141961 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_139939, &mem_139939_cached_sizze_141961, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138567 = 0; i_138567 < (int64_t) 16; i_138567++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138563 = 0; i_138563 < (int64_t) 27; i_138563++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128415;
            double r_128417 = 0.0;
            
            for (int64_t i_128416 = 0; i_128416 < (int64_t) 16; i_128416++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128418 = ((double *) wvoc_mem_139499.mem)[i_138563 * (int64_t) 16 + i_128416];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128419 = ((double *) mem_139918)[i_138567 * (int64_t) 16 + i_128416];
                
                // futhark/microgpt.fut:227:67-107
                
                double zt_res_128420 = zt_lhs_128418 * zt_rhs_128419;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128421 = r_128417 + zt_res_128420;
                double r_tmp_141646 = zp_res_128421;
                
                r_128417 = r_tmp_141646;
            }
            defunc_0_lifted_lambda_res_128415 = r_128417;
            ((double *) mem_139939)[i_138563] = defunc_0_lifted_lambda_res_128415;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139934, i_138567 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139939, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_139950, (int64_t) 128, "mem_139950")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139954_cached_sizze_141962 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_139954, &mem_139954_cached_sizze_141962, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139961_cached_sizze_141963 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_139961, &mem_139961_cached_sizze_141963, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138581 = 0; i_138581 < (int64_t) 16; i_138581++) {
        double x_129150;
        double redout_138569 = -INFINITY;
        
        for (int64_t i_138570 = 0; i_138570 < (int64_t) 27; i_138570++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_129097 = ((double *) mem_139934)[i_138581 * (int64_t) 27 + i_138570];
            
            // futhark/microgpt.fut:105:13-33
            
            double max_res_128445 = fmax64(lifted_lambda_res_129097, redout_138569);
            double redout_tmp_141648 = max_res_128445;
            
            redout_138569 = redout_tmp_141648;
        }
        x_129150 = redout_138569;
        // futhark/microgpt.fut:229:67-76
        
        double neg_res_128446 = -x_129150;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_128430;
        double r_128432 = 0.0;
        
        for (int64_t i_128431 = 0; i_128431 < (int64_t) 27; i_128431++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138573 = 0; i_138573 < (int64_t) 27; i_138573++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_128453 = ((double *) mem_139934)[i_138581 * (int64_t) 27 + i_138573];
                
                // futhark/microgpt.fut:229:44-76
                
                double zp_res_128454 = neg_res_128446 + zp_lhs_128453;
                
                // futhark/microgpt.fut:229:37-76
                
                double exp_res_128455 = futrts_exp64(zp_res_128454);
                
                ((double *) mem_139954)[i_138573] = exp_res_128455;
            }
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128457;
            double r_128459 = 0.0;
            
            for (int64_t i_128458 = 0; i_128458 < (int64_t) 27; i_128458++) {
                // futhark/microgpt.fut:230:36-46
                
                double lifted_lambda_res_128460 = ((double *) mem_139954)[i_128458];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128461 = r_128459 + lifted_lambda_res_128460;
                double r_tmp_141651 = zp_res_128461;
                
                r_128459 = r_tmp_141651;
            }
            defunc_0_lifted_lambda_res_128457 = r_128459;
            // futhark/microgpt.fut:231:53-64
            
            double zs_res_128462 = 1.0 / defunc_0_lifted_lambda_res_128457;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138577 = 0; i_138577 < (int64_t) 27; i_138577++) {
                // futhark/microgpt.fut:231:37-47
                
                double zt_lhs_128469 = ((double *) mem_139954)[i_138577];
                
                // futhark/microgpt.fut:231:37-64
                
                double zt_res_128470 = zs_res_128462 * zt_lhs_128469;
                
                ((double *) mem_139961)[i_138577] = zt_res_128470;
            }
            // futhark/microgpt.fut:232:12-22
            
            double log_arg0_128472 = ((double *) mem_139961)[i_128431];
            
            // futhark/microgpt.fut:232:6-22
            
            double log_res_128473 = futrts_log64(log_arg0_128472);
            
            // futhark/microgpt.fut:61:46-49
            
            double zt_rhs_128474 = ((double *) target_mem_139501.mem)[i_138581 * (int64_t) 27 + i_128431];
            
            // futhark/microgpt.fut:232:6-48
            
            double zt_res_128475 = log_res_128473 * zt_rhs_128474;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_128476 = r_128432 + zt_res_128475;
            double r_tmp_141649 = zp_res_128476;
            
            r_128432 = r_tmp_141649;
        }
        defunc_0_lifted_lambda_res_128430 = r_128432;
        // futhark/microgpt.fut:228:37-232:54
        
        double neg_res_128477 = -defunc_0_lifted_lambda_res_128430;
        
        ((double *) mem_139950.mem)[i_138581] = neg_res_128477;
    }
    // futhark/microgpt.fut:61:13-49
    
    double defunc_0_lifted_lambda_res_128479;
    double r_128481 = 0.0;
    
    for (int64_t i_128480 = 0; i_128480 < (int64_t) 16; i_128480++) {
        // futhark/microgpt.fut:233:37-47
        
        double lifted_lambda_res_128482 = ((double *) mem_139950.mem)[i_128480];
        
        // futhark/microgpt.fut:61:40-49
        
        double zp_res_128483 = r_128481 + lifted_lambda_res_128482;
        double r_tmp_141653 = zp_res_128483;
        
        r_128481 = r_tmp_141653;
    }
    defunc_0_lifted_lambda_res_128479 = r_128481;
    // futhark/microgpt.fut:233:17-64
    
    double zs_res_128484 = defunc_0_lifted_lambda_res_128479 / 16.0;
    
    if (memblock_set(ctx, &mem_out_141576, &mem_139950, "mem_139950") != 0)
        return 1;
    prim_out_141577 = zs_res_128484;
    if (memblock_set(ctx, &*mem_out_p_141905, &mem_out_141576, "mem_out_141576") != 0)
        return 1;
    *out_prim_out_141906 = prim_out_141577;
    
  cleanup:
    {
        free(mem_139503);
        free(mem_139508);
        free(mem_139519);
        free(mem_139524);
        free(mem_139531);
        free(mem_139542);
        free(mem_139547);
        free(mem_139554);
        free(mem_139565);
        free(mem_139566);
        free(mem_139567);
        free(mem_139580);
        free(mem_139581);
        free(mem_139582);
        free(mem_139613);
        free(mem_139614);
        free(mem_139615);
        free(mem_139631);
        free(mem_139632);
        free(mem_139633);
        free(mem_139646);
        free(mem_139647);
        free(mem_139648);
        free(mem_139694);
        free(mem_139700);
        free(mem_139705);
        free(mem_139716);
        free(mem_139721);
        free(mem_139732);
        free(mem_139737);
        free(mem_139744);
        free(mem_139751);
        free(mem_139762);
        free(mem_139767);
        free(mem_139778);
        free(mem_139783);
        free(mem_139799);
        free(mem_139804);
        free(mem_139815);
        free(mem_139820);
        free(mem_139831);
        free(mem_139836);
        free(mem_139847);
        free(mem_139852);
        free(mem_139859);
        free(mem_139870);
        free(mem_139875);
        free(mem_139886);
        free(mem_139891);
        free(mem_139902);
        free(mem_139907);
        free(mem_139918);
        free(mem_139923);
        free(mem_139934);
        free(mem_139939);
        free(mem_139954);
        free(mem_139961);
        if (memblock_unref(ctx, &mem_139950, "mem_139950") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141576, "mem_out_141576") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_141964, struct memblock wdown_mem_139491, struct memblock wkey_mem_139492, struct memblock wout_mem_139493, struct memblock wpe_mem_139494, struct memblock wqry_mem_139495, struct memblock wte_mem_139496, struct memblock wup_mem_139497, struct memblock wval_mem_139498, struct memblock wvoc_mem_139499, struct memblock tokens_mem_139500, struct memblock mask_mem_139501)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_139502_cached_sizze_141965 = 0;
    unsigned char *mem_139502 = NULL;
    int64_t mem_139507_cached_sizze_141966 = 0;
    unsigned char *mem_139507 = NULL;
    int64_t mem_139518_cached_sizze_141967 = 0;
    unsigned char *mem_139518 = NULL;
    int64_t mem_139523_cached_sizze_141968 = 0;
    unsigned char *mem_139523 = NULL;
    int64_t mem_139530_cached_sizze_141969 = 0;
    unsigned char *mem_139530 = NULL;
    int64_t mem_139541_cached_sizze_141970 = 0;
    unsigned char *mem_139541 = NULL;
    int64_t mem_139546_cached_sizze_141971 = 0;
    unsigned char *mem_139546 = NULL;
    int64_t mem_139553_cached_sizze_141972 = 0;
    unsigned char *mem_139553 = NULL;
    int64_t mem_139564_cached_sizze_141973 = 0;
    unsigned char *mem_139564 = NULL;
    int64_t mem_139565_cached_sizze_141974 = 0;
    unsigned char *mem_139565 = NULL;
    int64_t mem_139566_cached_sizze_141975 = 0;
    unsigned char *mem_139566 = NULL;
    int64_t mem_139579_cached_sizze_141976 = 0;
    unsigned char *mem_139579 = NULL;
    int64_t mem_139580_cached_sizze_141977 = 0;
    unsigned char *mem_139580 = NULL;
    int64_t mem_139581_cached_sizze_141978 = 0;
    unsigned char *mem_139581 = NULL;
    int64_t mem_139612_cached_sizze_141979 = 0;
    unsigned char *mem_139612 = NULL;
    int64_t mem_139613_cached_sizze_141980 = 0;
    unsigned char *mem_139613 = NULL;
    int64_t mem_139614_cached_sizze_141981 = 0;
    unsigned char *mem_139614 = NULL;
    int64_t mem_139630_cached_sizze_141982 = 0;
    unsigned char *mem_139630 = NULL;
    int64_t mem_139631_cached_sizze_141983 = 0;
    unsigned char *mem_139631 = NULL;
    int64_t mem_139632_cached_sizze_141984 = 0;
    unsigned char *mem_139632 = NULL;
    int64_t mem_139645_cached_sizze_141985 = 0;
    unsigned char *mem_139645 = NULL;
    int64_t mem_139646_cached_sizze_141986 = 0;
    unsigned char *mem_139646 = NULL;
    int64_t mem_139647_cached_sizze_141987 = 0;
    unsigned char *mem_139647 = NULL;
    int64_t mem_139693_cached_sizze_141988 = 0;
    unsigned char *mem_139693 = NULL;
    int64_t mem_139699_cached_sizze_141989 = 0;
    unsigned char *mem_139699 = NULL;
    int64_t mem_139704_cached_sizze_141990 = 0;
    unsigned char *mem_139704 = NULL;
    int64_t mem_139715_cached_sizze_141991 = 0;
    unsigned char *mem_139715 = NULL;
    int64_t mem_139720_cached_sizze_141992 = 0;
    unsigned char *mem_139720 = NULL;
    int64_t mem_139731_cached_sizze_141993 = 0;
    unsigned char *mem_139731 = NULL;
    int64_t mem_139736_cached_sizze_141994 = 0;
    unsigned char *mem_139736 = NULL;
    int64_t mem_139743_cached_sizze_141995 = 0;
    unsigned char *mem_139743 = NULL;
    int64_t mem_139750_cached_sizze_141996 = 0;
    unsigned char *mem_139750 = NULL;
    int64_t mem_139761_cached_sizze_141997 = 0;
    unsigned char *mem_139761 = NULL;
    int64_t mem_139766_cached_sizze_141998 = 0;
    unsigned char *mem_139766 = NULL;
    int64_t mem_139777_cached_sizze_141999 = 0;
    unsigned char *mem_139777 = NULL;
    int64_t mem_139782_cached_sizze_142000 = 0;
    unsigned char *mem_139782 = NULL;
    int64_t mem_139798_cached_sizze_142001 = 0;
    unsigned char *mem_139798 = NULL;
    int64_t mem_139803_cached_sizze_142002 = 0;
    unsigned char *mem_139803 = NULL;
    int64_t mem_139814_cached_sizze_142003 = 0;
    unsigned char *mem_139814 = NULL;
    int64_t mem_139819_cached_sizze_142004 = 0;
    unsigned char *mem_139819 = NULL;
    int64_t mem_139830_cached_sizze_142005 = 0;
    unsigned char *mem_139830 = NULL;
    int64_t mem_139835_cached_sizze_142006 = 0;
    unsigned char *mem_139835 = NULL;
    int64_t mem_139846_cached_sizze_142007 = 0;
    unsigned char *mem_139846 = NULL;
    int64_t mem_139851_cached_sizze_142008 = 0;
    unsigned char *mem_139851 = NULL;
    int64_t mem_139858_cached_sizze_142009 = 0;
    unsigned char *mem_139858 = NULL;
    int64_t mem_139869_cached_sizze_142010 = 0;
    unsigned char *mem_139869 = NULL;
    int64_t mem_139874_cached_sizze_142011 = 0;
    unsigned char *mem_139874 = NULL;
    int64_t mem_139885_cached_sizze_142012 = 0;
    unsigned char *mem_139885 = NULL;
    int64_t mem_139890_cached_sizze_142013 = 0;
    unsigned char *mem_139890 = NULL;
    int64_t mem_139901_cached_sizze_142014 = 0;
    unsigned char *mem_139901 = NULL;
    int64_t mem_139906_cached_sizze_142015 = 0;
    unsigned char *mem_139906 = NULL;
    int64_t mem_139917_cached_sizze_142016 = 0;
    unsigned char *mem_139917 = NULL;
    int64_t mem_139922_cached_sizze_142017 = 0;
    unsigned char *mem_139922 = NULL;
    int64_t mem_139933_cached_sizze_142018 = 0;
    unsigned char *mem_139933 = NULL;
    int64_t mem_139938_cached_sizze_142019 = 0;
    unsigned char *mem_139938 = NULL;
    int64_t mem_139954_cached_sizze_142020 = 0;
    unsigned char *mem_139954 = NULL;
    struct memblock mem_139949;
    
    mem_139949.references = NULL;
    
    struct memblock mem_out_141576;
    
    mem_out_141576.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (mem_139502_cached_sizze_141965 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139502, &mem_139502_cached_sizze_141965, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139507_cached_sizze_141966 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139507, &mem_139507_cached_sizze_141966, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138363 = 0; i_138363 < (int64_t) 16; i_138363++) {
        // futhark/microgpt.fut:436:41-50
        
        int64_t tmp_127869 = ((int64_t *) tokens_mem_139500.mem)[i_138363];
        
        // futhark/microgpt.fut:436:37-51
        
        bool x_127870 = sle64((int64_t) 0, tmp_127869);
        
        // futhark/microgpt.fut:436:37-51
        
        bool y_127871 = slt64(tmp_127869, (int64_t) 27);
        
        // futhark/microgpt.fut:436:37-51
        
        bool bounds_check_127872 = x_127870 && y_127871;
        
        // futhark/microgpt.fut:436:37-51
        
        bool index_certs_127873;
        
        if (!bounds_check_127872) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_127869, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:436:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:436:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138359 = 0; i_138359 < (int64_t) 16; i_138359++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_127880 = ((double *) wte_mem_139496.mem)[tmp_127869 * (int64_t) 16 + i_138359];
            
            ((double *) mem_139507)[i_138359] = lifted_lambda_res_127880;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139502, i_138363 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139507, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139518_cached_sizze_141967 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139518, &mem_139518_cached_sizze_141967, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139523_cached_sizze_141968 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139523, &mem_139523_cached_sizze_141968, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139530_cached_sizze_141969 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139530, &mem_139530_cached_sizze_141969, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138375 = 0; i_138375 < (int64_t) 16; i_138375++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_127906;
        double r_127908 = 0.0;
        
        for (int64_t i_127907 = 0; i_127907 < (int64_t) 16; i_127907++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_127909 = ((double *) wpe_mem_139494.mem)[i_138375 * (int64_t) 16 + i_127907];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_127910 = ((double *) mem_139502)[i_138375 * (int64_t) 16 + i_127907];
            
            // futhark/microgpt.fut:138:76-116
            
            double zp_res_127911 = zp_lhs_127909 + zp_rhs_127910;
            
            // futhark/microgpt.fut:138:94-163
            
            double zt_res_127912 = zp_res_127911 * zp_res_127911;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_127913 = r_127908 + zt_res_127912;
            double r_tmp_141580 = zp_res_127913;
            
            r_127908 = r_tmp_141580;
        }
        defunc_0_lifted_lambda_res_127906 = r_127908;
        // futhark/microgpt.fut:138:54-182
        
        double zs_res_127914 = defunc_0_lifted_lambda_res_127906 / 16.0;
        
        // futhark/microgpt.fut:139:24-55
        
        double zp_res_127915 = 1.0e-5 + zs_res_127914;
        
        // futhark/microgpt.fut:139:16-55
        
        double sqrt_res_127916 = futrts_sqrt64(zp_res_127915);
        
        // futhark/microgpt.fut:140:85-96
        
        double zs_res_127917 = 1.0 / sqrt_res_127916;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138367 = 0; i_138367 < (int64_t) 16; i_138367++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_127924 = ((double *) wpe_mem_139494.mem)[i_138375 * (int64_t) 16 + i_138367];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_127925 = ((double *) mem_139502)[i_138375 * (int64_t) 16 + i_138367];
            
            // futhark/microgpt.fut:140:38-78
            
            double zp_res_127926 = zp_lhs_127924 + zp_rhs_127925;
            
            // futhark/microgpt.fut:140:56-96
            
            double zt_res_127927 = zs_res_127917 * zp_res_127926;
            
            ((double *) mem_139523)[i_138367] = zt_res_127927;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138371 = 0; i_138371 < (int64_t) 16; i_138371++) {
            // futhark/microgpt.fut:141:4-14
            
            double lifted_lambda_res_127935 = ((double *) mem_139523)[i_138371];
            
            ((double *) mem_139530)[i_138371] = lifted_lambda_res_127935;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139518, i_138375 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139530, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139541_cached_sizze_141970 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139541, &mem_139541_cached_sizze_141970, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139546_cached_sizze_141971 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139546, &mem_139546_cached_sizze_141971, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139553_cached_sizze_141972 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139553, &mem_139553_cached_sizze_141972, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138387 = 0; i_138387 < (int64_t) 16; i_138387++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_127944;
        double r_127946 = 0.0;
        
        for (int64_t i_127945 = 0; i_127945 < (int64_t) 16; i_127945++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_127947 = ((double *) mem_139518)[i_138387 * (int64_t) 16 + i_127945];
            
            // futhark/microgpt.fut:142:78-115
            
            double zt_res_127948 = zt_lhs_127947 * zt_lhs_127947;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_127949 = r_127946 + zt_res_127948;
            double r_tmp_141584 = zp_res_127949;
            
            r_127946 = r_tmp_141584;
        }
        defunc_0_lifted_lambda_res_127944 = r_127946;
        // futhark/microgpt.fut:142:57-133
        
        double zs_res_127950 = defunc_0_lifted_lambda_res_127944 / 16.0;
        
        // futhark/microgpt.fut:143:24-55
        
        double zp_res_127951 = 1.0e-5 + zs_res_127950;
        
        // futhark/microgpt.fut:143:16-55
        
        double sqrt_res_127952 = futrts_sqrt64(zp_res_127951);
        
        // futhark/microgpt.fut:144:59-70
        
        double zs_res_127953 = 1.0 / sqrt_res_127952;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138379 = 0; i_138379 < (int64_t) 16; i_138379++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_127960 = ((double *) mem_139518)[i_138387 * (int64_t) 16 + i_138379];
            
            // futhark/microgpt.fut:144:37-70
            
            double zt_res_127961 = zs_res_127953 * zt_lhs_127960;
            
            ((double *) mem_139546)[i_138379] = zt_res_127961;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138383 = 0; i_138383 < (int64_t) 16; i_138383++) {
            // futhark/microgpt.fut:145:4-14
            
            double lifted_lambda_res_127969 = ((double *) mem_139546)[i_138383];
            
            ((double *) mem_139553)[i_138383] = lifted_lambda_res_127969;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139541, i_138387 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139553, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139564_cached_sizze_141973 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139564, &mem_139564_cached_sizze_141973, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139565_cached_sizze_141974 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139565, &mem_139565_cached_sizze_141974, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139566_cached_sizze_141975 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139566, &mem_139566_cached_sizze_141975, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139579_cached_sizze_141976 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139579, &mem_139579_cached_sizze_141976, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139580_cached_sizze_141977 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139580, &mem_139580_cached_sizze_141977, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139581_cached_sizze_141978 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139581, &mem_139581_cached_sizze_141978, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138405 = 0; i_138405 < (int64_t) 16; i_138405++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138395 = 0; i_138395 < (int64_t) 16; i_138395++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128807;
            double r_128809 = 0.0;
            
            for (int64_t i_128808 = 0; i_128808 < (int64_t) 16; i_128808++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128810 = ((double *) wqry_mem_139495.mem)[i_138395 * (int64_t) 16 + i_128808];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128811 = ((double *) mem_139541)[i_138405 * (int64_t) 16 + i_128808];
                
                // futhark/microgpt.fut:146:66-105
                
                double zt_res_128812 = zt_lhs_128810 * zt_rhs_128811;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128813 = r_128809 + zt_res_128812;
                double r_tmp_141593 = zp_res_128813;
                
                r_128809 = r_tmp_141593;
            }
            defunc_0_lifted_lambda_res_128807 = r_128809;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128820;
            double r_128822 = 0.0;
            
            for (int64_t i_128821 = 0; i_128821 < (int64_t) 16; i_128821++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128823 = ((double *) wkey_mem_139492.mem)[i_138395 * (int64_t) 16 + i_128821];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128824 = ((double *) mem_139541)[i_138405 * (int64_t) 16 + i_128821];
                
                // futhark/microgpt.fut:147:66-105
                
                double zt_res_128825 = zt_lhs_128823 * zt_rhs_128824;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128826 = r_128822 + zt_res_128825;
                double r_tmp_141594 = zp_res_128826;
                
                r_128822 = r_tmp_141594;
            }
            defunc_0_lifted_lambda_res_128820 = r_128822;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128836;
            double r_128838 = 0.0;
            
            for (int64_t i_128837 = 0; i_128837 < (int64_t) 16; i_128837++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128839 = ((double *) wval_mem_139498.mem)[i_138395 * (int64_t) 16 + i_128837];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128840 = ((double *) mem_139541)[i_138405 * (int64_t) 16 + i_128837];
                
                // futhark/microgpt.fut:148:66-105
                
                double zt_res_128841 = zt_lhs_128839 * zt_rhs_128840;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128842 = r_128838 + zt_res_128841;
                double r_tmp_141595 = zp_res_128842;
                
                r_128838 = r_tmp_141595;
            }
            defunc_0_lifted_lambda_res_128836 = r_128838;
            ((double *) mem_139579)[i_138395] = defunc_0_lifted_lambda_res_128836;
            ((double *) mem_139580)[i_138395] = defunc_0_lifted_lambda_res_128820;
            ((double *) mem_139581)[i_138395] = defunc_0_lifted_lambda_res_128807;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139564, i_138405 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139579, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139565, i_138405 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139580, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139566, i_138405 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139581, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139612_cached_sizze_141979 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139612, &mem_139612_cached_sizze_141979, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139613_cached_sizze_141980 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139613, &mem_139613_cached_sizze_141980, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139614_cached_sizze_141981 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139614, &mem_139614_cached_sizze_141981, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139630_cached_sizze_141982 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139630, &mem_139630_cached_sizze_141982, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139631_cached_sizze_141983 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139631, &mem_139631_cached_sizze_141983, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139632_cached_sizze_141984 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139632, &mem_139632_cached_sizze_141984, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139645_cached_sizze_141985 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_139645, &mem_139645_cached_sizze_141985, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139646_cached_sizze_141986 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_139646, &mem_139646_cached_sizze_141986, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139647_cached_sizze_141987 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_139647, &mem_139647_cached_sizze_141987, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138435 = 0; i_138435 < (int64_t) 4; i_138435++) {
        // futhark/microgpt.fut:149:69-72
        
        int64_t zp_lhs_128683 = mul64((int64_t) 4, i_138435);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138425 = 0; i_138425 < (int64_t) 16; i_138425++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138415 = 0; i_138415 < (int64_t) 4; i_138415++) {
                // futhark/microgpt.fut:149:74-81
                
                int64_t tmp_129000 = add64(zp_lhs_128683, i_138415);
                
                // futhark/microgpt.fut:149:51-83
                
                bool x_129001 = sle64((int64_t) 0, tmp_129000);
                
                // futhark/microgpt.fut:149:51-83
                
                bool y_129002 = slt64(tmp_129000, (int64_t) 16);
                
                // futhark/microgpt.fut:149:51-83
                
                bool bounds_check_129003 = x_129001 && y_129002;
                
                // futhark/microgpt.fut:149:51-83
                
                bool index_certs_129004;
                
                if (!bounds_check_129003) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_129000, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:149:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:149:15-84\n   #9  futhark/microgpt.fut:437:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129005 = ((double *) mem_139566)[i_138425 * (int64_t) 16 + tmp_129000];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129013 = ((double *) mem_139565)[i_138425 * (int64_t) 16 + tmp_129000];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129024 = ((double *) mem_139564)[i_138425 * (int64_t) 16 + tmp_129000];
                
                ((double *) mem_139645)[i_138415] = lifted_lambda_res_129024;
                ((double *) mem_139646)[i_138415] = lifted_lambda_res_129013;
                ((double *) mem_139647)[i_138415] = lifted_lambda_res_129005;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139630, i_138425 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139645, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139631, i_138425 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139646, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139632, i_138425 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139647, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139612, i_138435 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_139630, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139613, i_138435 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_139631, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139614, i_138435 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_139632, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139693_cached_sizze_141988 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139693, &mem_139693_cached_sizze_141988, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139699_cached_sizze_141989 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139699, &mem_139699_cached_sizze_141989, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139704_cached_sizze_141990 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139704, &mem_139704_cached_sizze_141990, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139715_cached_sizze_141991 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139715, &mem_139715_cached_sizze_141991, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139720_cached_sizze_141992 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139720, &mem_139720_cached_sizze_141992, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139731_cached_sizze_141993 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139731, &mem_139731_cached_sizze_141993, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139736_cached_sizze_141994 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139736, &mem_139736_cached_sizze_141994, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139743_cached_sizze_141995 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139743, &mem_139743_cached_sizze_141995, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139750_cached_sizze_141996 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139750, &mem_139750_cached_sizze_141996, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139761_cached_sizze_141997 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139761, &mem_139761_cached_sizze_141997, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139766_cached_sizze_141998 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_139766, &mem_139766_cached_sizze_141998, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139777_cached_sizze_141999 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139777, &mem_139777_cached_sizze_141999, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139782_cached_sizze_142000 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_139782, &mem_139782_cached_sizze_142000, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138491 = 0; i_138491 < (int64_t) 4; i_138491++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138445 = 0; i_138445 < (int64_t) 16; i_138445++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138441 = 0; i_138441 < (int64_t) 16; i_138441++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_128114;
                double r_128116 = 0.0;
                
                for (int64_t i_128115 = 0; i_128115 < (int64_t) 4; i_128115++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_128117 = ((double *) mem_139614)[i_138491 * (int64_t) 64 + i_138445 * (int64_t) 4 + i_128115];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_128118 = ((double *) mem_139613)[i_138491 * (int64_t) 64 + i_138441 * (int64_t) 4 + i_128115];
                    
                    // futhark/microgpt.fut:152:113-164
                    
                    double zt_res_128119 = zt_lhs_128117 * zt_rhs_128118;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_128120 = r_128116 + zt_res_128119;
                    double r_tmp_141608 = zp_res_128120;
                    
                    r_128116 = r_tmp_141608;
                }
                defunc_0_lifted_lambda_res_128114 = r_128116;
                ((double *) mem_139704)[i_138441] = defunc_0_lifted_lambda_res_128114;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139699, i_138445 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139704, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138453 = 0; i_138453 < (int64_t) 16; i_138453++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138449 = 0; i_138449 < (int64_t) 16; i_138449++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_128135 = ((double *) mem_139699)[i_138453 * (int64_t) 16 + i_138449];
                
                // futhark/microgpt.fut:153:47-78
                
                double zs_res_128136 = zs_lhs_128135 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_128137 = ((double *) mask_mem_139501.mem)[i_138453 * (int64_t) 16 + i_138449];
                
                // futhark/microgpt.fut:153:65-102
                
                double zp_res_128138 = zs_res_128136 + zp_rhs_128137;
                
                ((double *) mem_139720)[i_138449] = zp_res_128138;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139715, i_138453 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139720, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138471 = 0; i_138471 < (int64_t) 16; i_138471++) {
            // futhark/microgpt.fut:105:13-33
            
            double defunc_0_reduce_res_129102;
            double redout_138455 = -INFINITY;
            
            for (int64_t i_138456 = 0; i_138456 < (int64_t) 16; i_138456++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129051 = ((double *) mem_139715)[i_138471 * (int64_t) 16 + i_138456];
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_128159 = fmax64(lifted_lambda_res_129051, redout_138455);
                double redout_tmp_141612 = max_res_128159;
                
                redout_138455 = redout_tmp_141612;
            }
            defunc_0_reduce_res_129102 = redout_138455;
            // futhark/microgpt.fut:155:67-76
            
            double neg_res_128160 = -defunc_0_reduce_res_129102;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138459 = 0; i_138459 < (int64_t) 16; i_138459++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_128167 = ((double *) mem_139715)[i_138471 * (int64_t) 16 + i_138459];
                
                // futhark/microgpt.fut:155:44-76
                
                double zp_res_128168 = neg_res_128160 + zp_lhs_128167;
                
                // futhark/microgpt.fut:155:37-76
                
                double exp_res_128169 = futrts_exp64(zp_res_128168);
                
                ((double *) mem_139736)[i_138459] = exp_res_128169;
            }
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128171;
            double r_128173 = 0.0;
            
            for (int64_t i_128172 = 0; i_128172 < (int64_t) 16; i_128172++) {
                // futhark/microgpt.fut:156:36-46
                
                double lifted_lambda_res_128174 = ((double *) mem_139736)[i_128172];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128175 = r_128173 + lifted_lambda_res_128174;
                double r_tmp_141614 = zp_res_128175;
                
                r_128173 = r_tmp_141614;
            }
            defunc_0_lifted_lambda_res_128171 = r_128173;
            // futhark/microgpt.fut:157:53-64
            
            double zs_res_128176 = 1.0 / defunc_0_lifted_lambda_res_128171;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138463 = 0; i_138463 < (int64_t) 16; i_138463++) {
                // futhark/microgpt.fut:157:37-47
                
                double zt_lhs_128183 = ((double *) mem_139736)[i_138463];
                
                // futhark/microgpt.fut:157:37-64
                
                double zt_res_128184 = zs_res_128176 * zt_lhs_128183;
                
                ((double *) mem_139743)[i_138463] = zt_res_128184;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138467 = 0; i_138467 < (int64_t) 16; i_138467++) {
                // futhark/microgpt.fut:158:4-14
                
                double lifted_lambda_res_128192 = ((double *) mem_139743)[i_138467];
                
                ((double *) mem_139750)[i_138467] = lifted_lambda_res_128192;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139731, i_138471 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139750, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138479 = 0; i_138479 < (int64_t) 16; i_138479++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138475 = 0; i_138475 < (int64_t) 4; i_138475++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_128207;
                double r_128209 = 0.0;
                
                for (int64_t i_128208 = 0; i_128208 < (int64_t) 16; i_128208++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_128210 = ((double *) mem_139731)[i_138479 * (int64_t) 16 + i_128208];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_128211 = ((double *) mem_139612)[i_138491 * (int64_t) 64 + i_128208 * (int64_t) 4 + i_138475];
                    
                    // futhark/microgpt.fut:159:66-111
                    
                    double zt_res_128212 = zt_lhs_128210 * zt_rhs_128211;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_128213 = r_128209 + zt_res_128212;
                    double r_tmp_141619 = zp_res_128213;
                    
                    r_128209 = r_tmp_141619;
                }
                defunc_0_lifted_lambda_res_128207 = r_128209;
                ((double *) mem_139766)[i_138475] = defunc_0_lifted_lambda_res_128207;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139761, i_138479 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139766, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138487 = 0; i_138487 < (int64_t) 16; i_138487++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138483 = 0; i_138483 < (int64_t) 4; i_138483++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_128228 = ((double *) mem_139761)[i_138487 * (int64_t) 4 + i_138483];
                
                ((double *) mem_139782)[i_138483] = lifted_lambda_res_128228;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139777, i_138487 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139782, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139693, i_138491 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_139777, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139798_cached_sizze_142001 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139798, &mem_139798_cached_sizze_142001, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139803_cached_sizze_142002 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139803, &mem_139803_cached_sizze_142002, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138499 = 0; i_138499 < (int64_t) 16; i_138499++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138495 = 0; i_138495 < (int64_t) 16; i_138495++) {
            // futhark/microgpt.fut:161:54-57
            
            int64_t tmp_128240 = sdiv64(i_138495, (int64_t) 4);
            
            // futhark/microgpt.fut:161:44-59
            
            bool x_128241 = sle64((int64_t) 0, tmp_128240);
            
            // futhark/microgpt.fut:161:44-59
            
            bool y_128242 = slt64(tmp_128240, (int64_t) 4);
            
            // futhark/microgpt.fut:161:44-59
            
            bool bounds_check_128243 = x_128241 && y_128242;
            
            // futhark/microgpt.fut:161:44-59
            
            bool index_certs_128244;
            
            if (!bounds_check_128243) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_128240, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:161:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:161:15-80\n   #6  futhark/microgpt.fut:437:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:161:74-77
            
            int64_t tmp_128245 = smod64(i_138495, (int64_t) 4);
            
            // futhark/microgpt.fut:161:44-79
            
            bool x_128246 = sle64((int64_t) 0, tmp_128245);
            
            // futhark/microgpt.fut:161:44-79
            
            bool y_128247 = slt64(tmp_128245, (int64_t) 4);
            
            // futhark/microgpt.fut:161:44-79
            
            bool bounds_check_128248 = x_128246 && y_128247;
            
            // futhark/microgpt.fut:161:44-79
            
            bool index_certs_128249;
            
            if (!bounds_check_128248) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_128245, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:161:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:161:15-80\n   #6  futhark/microgpt.fut:437:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_128250 = ((double *) mem_139693)[tmp_128240 * (int64_t) 64 + i_138499 * (int64_t) 4 + tmp_128245];
            
            ((double *) mem_139803)[i_138495] = lifted_lambda_res_128250;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139798, i_138499 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139803, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139814_cached_sizze_142003 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139814, &mem_139814_cached_sizze_142003, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139819_cached_sizze_142004 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139819, &mem_139819_cached_sizze_142004, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138507 = 0; i_138507 < (int64_t) 16; i_138507++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138503 = 0; i_138503 < (int64_t) 16; i_138503++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128265;
            double r_128267 = 0.0;
            
            for (int64_t i_128266 = 0; i_128266 < (int64_t) 16; i_128266++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128268 = ((double *) wout_mem_139493.mem)[i_138503 * (int64_t) 16 + i_128266];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128269 = ((double *) mem_139798)[i_138507 * (int64_t) 16 + i_128266];
                
                // futhark/microgpt.fut:162:67-106
                
                double zt_res_128270 = zt_lhs_128268 * zt_rhs_128269;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128271 = r_128267 + zt_res_128270;
                double r_tmp_141626 = zp_res_128271;
                
                r_128267 = r_tmp_141626;
            }
            defunc_0_lifted_lambda_res_128265 = r_128267;
            ((double *) mem_139819)[i_138503] = defunc_0_lifted_lambda_res_128265;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139814, i_138507 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139819, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139830_cached_sizze_142005 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139830, &mem_139830_cached_sizze_142005, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139835_cached_sizze_142006 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139835, &mem_139835_cached_sizze_142006, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138515 = 0; i_138515 < (int64_t) 16; i_138515++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138511 = 0; i_138511 < (int64_t) 16; i_138511++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_128286 = ((double *) mem_139814)[i_138515 * (int64_t) 16 + i_138511];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_128287 = ((double *) mem_139518)[i_138515 * (int64_t) 16 + i_138511];
            
            // futhark/microgpt.fut:163:46-84
            
            double zp_res_128288 = zp_lhs_128286 + zp_rhs_128287;
            
            ((double *) mem_139835)[i_138511] = zp_res_128288;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139830, i_138515 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139835, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139846_cached_sizze_142007 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139846, &mem_139846_cached_sizze_142007, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139851_cached_sizze_142008 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139851, &mem_139851_cached_sizze_142008, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139858_cached_sizze_142009 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139858, &mem_139858_cached_sizze_142009, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138527 = 0; i_138527 < (int64_t) 16; i_138527++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_128297;
        double r_128299 = 0.0;
        
        for (int64_t i_128298 = 0; i_128298 < (int64_t) 16; i_128298++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_128300 = ((double *) mem_139830)[i_138527 * (int64_t) 16 + i_128298];
            
            // futhark/microgpt.fut:164:79-118
            
            double zt_res_128301 = zt_lhs_128300 * zt_lhs_128300;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_128302 = r_128299 + zt_res_128301;
            double r_tmp_141630 = zp_res_128302;
            
            r_128299 = r_tmp_141630;
        }
        defunc_0_lifted_lambda_res_128297 = r_128299;
        // futhark/microgpt.fut:164:58-136
        
        double zs_res_128303 = defunc_0_lifted_lambda_res_128297 / 16.0;
        
        // futhark/microgpt.fut:165:24-55
        
        double zp_res_128304 = 1.0e-5 + zs_res_128303;
        
        // futhark/microgpt.fut:165:16-55
        
        double sqrt_res_128305 = futrts_sqrt64(zp_res_128304);
        
        // futhark/microgpt.fut:166:60-71
        
        double zs_res_128306 = 1.0 / sqrt_res_128305;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138519 = 0; i_138519 < (int64_t) 16; i_138519++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_128313 = ((double *) mem_139830)[i_138527 * (int64_t) 16 + i_138519];
            
            // futhark/microgpt.fut:166:37-71
            
            double zt_res_128314 = zs_res_128306 * zt_lhs_128313;
            
            ((double *) mem_139851)[i_138519] = zt_res_128314;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138523 = 0; i_138523 < (int64_t) 16; i_138523++) {
            // futhark/microgpt.fut:167:4-14
            
            double lifted_lambda_res_128322 = ((double *) mem_139851)[i_138523];
            
            ((double *) mem_139858)[i_138523] = lifted_lambda_res_128322;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139846, i_138527 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139858, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139869_cached_sizze_142010 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139869, &mem_139869_cached_sizze_142010, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139874_cached_sizze_142011 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139874, &mem_139874_cached_sizze_142011, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138535 = 0; i_138535 < (int64_t) 16; i_138535++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138531 = 0; i_138531 < (int64_t) 64; i_138531++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128338;
            double r_128340 = 0.0;
            
            for (int64_t i_128339 = 0; i_128339 < (int64_t) 16; i_128339++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128341 = ((double *) wup_mem_139497.mem)[i_138531 * (int64_t) 16 + i_128339];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128342 = ((double *) mem_139846)[i_138535 * (int64_t) 16 + i_128339];
                
                // futhark/microgpt.fut:168:67-106
                
                double zt_res_128343 = zt_lhs_128341 * zt_rhs_128342;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128344 = r_128340 + zt_res_128343;
                double r_tmp_141635 = zp_res_128344;
                
                r_128340 = r_tmp_141635;
            }
            defunc_0_lifted_lambda_res_128338 = r_128340;
            ((double *) mem_139874)[i_138531] = defunc_0_lifted_lambda_res_128338;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139869, i_138535 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139874, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139885_cached_sizze_142012 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139885, &mem_139885_cached_sizze_142012, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139890_cached_sizze_142013 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139890, &mem_139890_cached_sizze_142013, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138543 = 0; i_138543 < (int64_t) 16; i_138543++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138539 = 0; i_138539 < (int64_t) 64; i_138539++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_128359 = ((double *) mem_139869)[i_138543 * (int64_t) 64 + i_138539];
            
            // futhark/microgpt.fut:169:45-73
            
            double max_res_128360 = fmax64(0.0, max_arg0_128359);
            
            ((double *) mem_139890)[i_138539] = max_res_128360;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139885, i_138543 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139890, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139901_cached_sizze_142014 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139901, &mem_139901_cached_sizze_142014, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139906_cached_sizze_142015 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139906, &mem_139906_cached_sizze_142015, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138551 = 0; i_138551 < (int64_t) 16; i_138551++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138547 = 0; i_138547 < (int64_t) 16; i_138547++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128375;
            double r_128377 = 0.0;
            
            for (int64_t i_128376 = 0; i_128376 < (int64_t) 64; i_128376++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128378 = ((double *) wdown_mem_139491.mem)[i_138547 * (int64_t) 64 + i_128376];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128379 = ((double *) mem_139885)[i_138551 * (int64_t) 64 + i_128376];
                
                // futhark/microgpt.fut:170:67-108
                
                double zt_res_128380 = zt_lhs_128378 * zt_rhs_128379;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128381 = r_128377 + zt_res_128380;
                double r_tmp_141640 = zp_res_128381;
                
                r_128377 = r_tmp_141640;
            }
            defunc_0_lifted_lambda_res_128375 = r_128377;
            ((double *) mem_139906)[i_138547] = defunc_0_lifted_lambda_res_128375;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139901, i_138551 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139906, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139917_cached_sizze_142016 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139917, &mem_139917_cached_sizze_142016, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139922_cached_sizze_142017 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139922, &mem_139922_cached_sizze_142017, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138559 = 0; i_138559 < (int64_t) 16; i_138559++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138555 = 0; i_138555 < (int64_t) 16; i_138555++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_128396 = ((double *) mem_139901)[i_138559 * (int64_t) 16 + i_138555];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_128397 = ((double *) mem_139830)[i_138559 * (int64_t) 16 + i_138555];
            
            // futhark/microgpt.fut:171:46-85
            
            double zp_res_128398 = zp_lhs_128396 + zp_rhs_128397;
            
            ((double *) mem_139922)[i_138555] = zp_res_128398;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139917, i_138559 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139922, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139933_cached_sizze_142018 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_139933, &mem_139933_cached_sizze_142018, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139938_cached_sizze_142019 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_139938, &mem_139938_cached_sizze_142019, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138567 = 0; i_138567 < (int64_t) 16; i_138567++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138563 = 0; i_138563 < (int64_t) 27; i_138563++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128414;
            double r_128416 = 0.0;
            
            for (int64_t i_128415 = 0; i_128415 < (int64_t) 16; i_128415++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128417 = ((double *) wvoc_mem_139499.mem)[i_138563 * (int64_t) 16 + i_128415];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128418 = ((double *) mem_139917)[i_138567 * (int64_t) 16 + i_128415];
                
                // futhark/microgpt.fut:172:67-107
                
                double zt_res_128419 = zt_lhs_128417 * zt_rhs_128418;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128420 = r_128416 + zt_res_128419;
                double r_tmp_141645 = zp_res_128420;
                
                r_128416 = r_tmp_141645;
            }
            defunc_0_lifted_lambda_res_128414 = r_128416;
            ((double *) mem_139938)[i_138563] = defunc_0_lifted_lambda_res_128414;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139933, i_138567 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139938, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_139949, (int64_t) 3456, "mem_139949")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139954_cached_sizze_142020 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_139954, &mem_139954_cached_sizze_142020, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138575 = 0; i_138575 < (int64_t) 16; i_138575++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138571 = 0; i_138571 < (int64_t) 27; i_138571++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_128435 = ((double *) mem_139933)[i_138575 * (int64_t) 27 + i_138571];
            
            ((double *) mem_139954)[i_138571] = lifted_lambda_res_128435;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139949.mem, i_138575 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139954, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_141576, &mem_139949, "mem_139949") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141964, &mem_out_141576, "mem_out_141576") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_139502);
        free(mem_139507);
        free(mem_139518);
        free(mem_139523);
        free(mem_139530);
        free(mem_139541);
        free(mem_139546);
        free(mem_139553);
        free(mem_139564);
        free(mem_139565);
        free(mem_139566);
        free(mem_139579);
        free(mem_139580);
        free(mem_139581);
        free(mem_139612);
        free(mem_139613);
        free(mem_139614);
        free(mem_139630);
        free(mem_139631);
        free(mem_139632);
        free(mem_139645);
        free(mem_139646);
        free(mem_139647);
        free(mem_139693);
        free(mem_139699);
        free(mem_139704);
        free(mem_139715);
        free(mem_139720);
        free(mem_139731);
        free(mem_139736);
        free(mem_139743);
        free(mem_139750);
        free(mem_139761);
        free(mem_139766);
        free(mem_139777);
        free(mem_139782);
        free(mem_139798);
        free(mem_139803);
        free(mem_139814);
        free(mem_139819);
        free(mem_139830);
        free(mem_139835);
        free(mem_139846);
        free(mem_139851);
        free(mem_139858);
        free(mem_139869);
        free(mem_139874);
        free(mem_139885);
        free(mem_139890);
        free(mem_139901);
        free(mem_139906);
        free(mem_139917);
        free(mem_139922);
        free(mem_139933);
        free(mem_139938);
        free(mem_139954);
        if (memblock_unref(ctx, &mem_139949, "mem_139949") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141576, "mem_out_141576") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_grad_loss(struct futhark_context *ctx, struct memblock *mem_out_p_142021, struct memblock *mem_out_p_142022, struct memblock *mem_out_p_142023, struct memblock *mem_out_p_142024, struct memblock *mem_out_p_142025, struct memblock *mem_out_p_142026, struct memblock *mem_out_p_142027, struct memblock *mem_out_p_142028, struct memblock *mem_out_p_142029, struct memblock wdown_mem_139491, struct memblock wkey_mem_139492, struct memblock wout_mem_139493, struct memblock wpe_mem_139494, struct memblock wqry_mem_139495, struct memblock wte_mem_139496, struct memblock wup_mem_139497, struct memblock wval_mem_139498, struct memblock wvoc_mem_139499, struct memblock tokens_mem_139500, struct memblock target_mem_139501, struct memblock mask_mem_139502)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_139503_cached_sizze_142030 = 0;
    unsigned char *mem_139503 = NULL;
    int64_t mem_139508_cached_sizze_142031 = 0;
    unsigned char *mem_139508 = NULL;
    int64_t mem_139519_cached_sizze_142032 = 0;
    unsigned char *mem_139519 = NULL;
    int64_t mem_139520_cached_sizze_142033 = 0;
    unsigned char *mem_139520 = NULL;
    int64_t mem_139521_cached_sizze_142034 = 0;
    unsigned char *mem_139521 = NULL;
    int64_t mem_139540_cached_sizze_142035 = 0;
    unsigned char *mem_139540 = NULL;
    int64_t mem_139547_cached_sizze_142036 = 0;
    unsigned char *mem_139547 = NULL;
    int64_t mem_139552_cached_sizze_142037 = 0;
    unsigned char *mem_139552 = NULL;
    int64_t mem_139563_cached_sizze_142038 = 0;
    unsigned char *mem_139563 = NULL;
    int64_t mem_139568_cached_sizze_142039 = 0;
    unsigned char *mem_139568 = NULL;
    int64_t mem_139579_cached_sizze_142040 = 0;
    unsigned char *mem_139579 = NULL;
    int64_t mem_139580_cached_sizze_142041 = 0;
    unsigned char *mem_139580 = NULL;
    int64_t mem_139593_cached_sizze_142042 = 0;
    unsigned char *mem_139593 = NULL;
    int64_t mem_139600_cached_sizze_142043 = 0;
    unsigned char *mem_139600 = NULL;
    int64_t mem_139605_cached_sizze_142044 = 0;
    unsigned char *mem_139605 = NULL;
    int64_t mem_139616_cached_sizze_142045 = 0;
    unsigned char *mem_139616 = NULL;
    int64_t mem_139621_cached_sizze_142046 = 0;
    unsigned char *mem_139621 = NULL;
    int64_t mem_139632_cached_sizze_142047 = 0;
    unsigned char *mem_139632 = NULL;
    int64_t mem_139633_cached_sizze_142048 = 0;
    unsigned char *mem_139633 = NULL;
    int64_t mem_139634_cached_sizze_142049 = 0;
    unsigned char *mem_139634 = NULL;
    int64_t mem_139650_cached_sizze_142050 = 0;
    unsigned char *mem_139650 = NULL;
    int64_t mem_139651_cached_sizze_142051 = 0;
    unsigned char *mem_139651 = NULL;
    int64_t mem_139652_cached_sizze_142052 = 0;
    unsigned char *mem_139652 = NULL;
    int64_t mem_139665_cached_sizze_142053 = 0;
    unsigned char *mem_139665 = NULL;
    int64_t mem_139666_cached_sizze_142054 = 0;
    unsigned char *mem_139666 = NULL;
    int64_t mem_139667_cached_sizze_142055 = 0;
    unsigned char *mem_139667 = NULL;
    int64_t mem_139713_cached_sizze_142056 = 0;
    unsigned char *mem_139713 = NULL;
    int64_t mem_139714_cached_sizze_142057 = 0;
    unsigned char *mem_139714 = NULL;
    int64_t mem_139715_cached_sizze_142058 = 0;
    unsigned char *mem_139715 = NULL;
    int64_t mem_139716_cached_sizze_142059 = 0;
    unsigned char *mem_139716 = NULL;
    int64_t mem_139737_cached_sizze_142060 = 0;
    unsigned char *mem_139737 = NULL;
    int64_t mem_139738_cached_sizze_142061 = 0;
    unsigned char *mem_139738 = NULL;
    int64_t mem_139739_cached_sizze_142062 = 0;
    unsigned char *mem_139739 = NULL;
    int64_t mem_139740_cached_sizze_142063 = 0;
    unsigned char *mem_139740 = NULL;
    int64_t mem_139757_cached_sizze_142064 = 0;
    unsigned char *mem_139757 = NULL;
    int64_t mem_139758_cached_sizze_142065 = 0;
    unsigned char *mem_139758 = NULL;
    int64_t mem_139759_cached_sizze_142066 = 0;
    unsigned char *mem_139759 = NULL;
    int64_t mem_139760_cached_sizze_142067 = 0;
    unsigned char *mem_139760 = NULL;
    int64_t mem_139821_cached_sizze_142068 = 0;
    unsigned char *mem_139821 = NULL;
    int64_t mem_139822_cached_sizze_142069 = 0;
    unsigned char *mem_139822 = NULL;
    int64_t mem_139823_cached_sizze_142070 = 0;
    unsigned char *mem_139823 = NULL;
    int64_t mem_139824_cached_sizze_142071 = 0;
    unsigned char *mem_139824 = NULL;
    int64_t mem_139845_cached_sizze_142072 = 0;
    unsigned char *mem_139845 = NULL;
    int64_t mem_139846_cached_sizze_142073 = 0;
    unsigned char *mem_139846 = NULL;
    int64_t mem_139847_cached_sizze_142074 = 0;
    unsigned char *mem_139847 = NULL;
    int64_t mem_139848_cached_sizze_142075 = 0;
    unsigned char *mem_139848 = NULL;
    int64_t mem_139865_cached_sizze_142076 = 0;
    unsigned char *mem_139865 = NULL;
    int64_t mem_139866_cached_sizze_142077 = 0;
    unsigned char *mem_139866 = NULL;
    int64_t mem_139867_cached_sizze_142078 = 0;
    unsigned char *mem_139867 = NULL;
    int64_t mem_139868_cached_sizze_142079 = 0;
    unsigned char *mem_139868 = NULL;
    int64_t mem_139929_cached_sizze_142080 = 0;
    unsigned char *mem_139929 = NULL;
    int64_t mem_139930_cached_sizze_142081 = 0;
    unsigned char *mem_139930 = NULL;
    int64_t mem_139931_cached_sizze_142082 = 0;
    unsigned char *mem_139931 = NULL;
    int64_t mem_139932_cached_sizze_142083 = 0;
    unsigned char *mem_139932 = NULL;
    int64_t mem_139933_cached_sizze_142084 = 0;
    unsigned char *mem_139933 = NULL;
    int64_t mem_139934_cached_sizze_142085 = 0;
    unsigned char *mem_139934 = NULL;
    int64_t mem_139935_cached_sizze_142086 = 0;
    unsigned char *mem_139935 = NULL;
    int64_t mem_139936_cached_sizze_142087 = 0;
    unsigned char *mem_139936 = NULL;
    int64_t mem_139969_cached_sizze_142088 = 0;
    unsigned char *mem_139969 = NULL;
    int64_t mem_139970_cached_sizze_142089 = 0;
    unsigned char *mem_139970 = NULL;
    int64_t mem_139971_cached_sizze_142090 = 0;
    unsigned char *mem_139971 = NULL;
    int64_t mem_139972_cached_sizze_142091 = 0;
    unsigned char *mem_139972 = NULL;
    int64_t mem_139973_cached_sizze_142092 = 0;
    unsigned char *mem_139973 = NULL;
    int64_t mem_139974_cached_sizze_142093 = 0;
    unsigned char *mem_139974 = NULL;
    int64_t mem_139975_cached_sizze_142094 = 0;
    unsigned char *mem_139975 = NULL;
    int64_t mem_139976_cached_sizze_142095 = 0;
    unsigned char *mem_139976 = NULL;
    int64_t mem_140057_cached_sizze_142096 = 0;
    unsigned char *mem_140057 = NULL;
    int64_t mem_140058_cached_sizze_142097 = 0;
    unsigned char *mem_140058 = NULL;
    int64_t mem_140059_cached_sizze_142098 = 0;
    unsigned char *mem_140059 = NULL;
    int64_t mem_140060_cached_sizze_142099 = 0;
    unsigned char *mem_140060 = NULL;
    int64_t mem_140081_cached_sizze_142100 = 0;
    unsigned char *mem_140081 = NULL;
    int64_t mem_140082_cached_sizze_142101 = 0;
    unsigned char *mem_140082 = NULL;
    int64_t mem_140083_cached_sizze_142102 = 0;
    unsigned char *mem_140083 = NULL;
    int64_t mem_140084_cached_sizze_142103 = 0;
    unsigned char *mem_140084 = NULL;
    int64_t mem_140101_cached_sizze_142104 = 0;
    unsigned char *mem_140101 = NULL;
    int64_t mem_140102_cached_sizze_142105 = 0;
    unsigned char *mem_140102 = NULL;
    int64_t mem_140103_cached_sizze_142106 = 0;
    unsigned char *mem_140103 = NULL;
    int64_t mem_140104_cached_sizze_142107 = 0;
    unsigned char *mem_140104 = NULL;
    int64_t mem_140165_cached_sizze_142108 = 0;
    unsigned char *mem_140165 = NULL;
    int64_t mem_140166_cached_sizze_142109 = 0;
    unsigned char *mem_140166 = NULL;
    int64_t mem_140175_cached_sizze_142110 = 0;
    unsigned char *mem_140175 = NULL;
    int64_t mem_140176_cached_sizze_142111 = 0;
    unsigned char *mem_140176 = NULL;
    int64_t mem_140197_cached_sizze_142112 = 0;
    unsigned char *mem_140197 = NULL;
    int64_t mem_140198_cached_sizze_142113 = 0;
    unsigned char *mem_140198 = NULL;
    int64_t mem_140209_cached_sizze_142114 = 0;
    unsigned char *mem_140209 = NULL;
    int64_t mem_140210_cached_sizze_142115 = 0;
    unsigned char *mem_140210 = NULL;
    int64_t mem_140219_cached_sizze_142116 = 0;
    unsigned char *mem_140219 = NULL;
    int64_t mem_140220_cached_sizze_142117 = 0;
    unsigned char *mem_140220 = NULL;
    int64_t mem_140251_cached_sizze_142118 = 0;
    unsigned char *mem_140251 = NULL;
    int64_t mem_140252_cached_sizze_142119 = 0;
    unsigned char *mem_140252 = NULL;
    int64_t mem_140263_cached_sizze_142120 = 0;
    unsigned char *mem_140263 = NULL;
    int64_t mem_140264_cached_sizze_142121 = 0;
    unsigned char *mem_140264 = NULL;
    int64_t mem_140273_cached_sizze_142122 = 0;
    unsigned char *mem_140273 = NULL;
    int64_t mem_140274_cached_sizze_142123 = 0;
    unsigned char *mem_140274 = NULL;
    int64_t mem_140305_cached_sizze_142124 = 0;
    unsigned char *mem_140305 = NULL;
    int64_t mem_140311_cached_sizze_142125 = 0;
    unsigned char *mem_140311 = NULL;
    int64_t mem_140316_cached_sizze_142126 = 0;
    unsigned char *mem_140316 = NULL;
    int64_t mem_140332_cached_sizze_142127 = 0;
    unsigned char *mem_140332 = NULL;
    int64_t mem_140337_cached_sizze_142128 = 0;
    unsigned char *mem_140337 = NULL;
    int64_t mem_140348_cached_sizze_142129 = 0;
    unsigned char *mem_140348 = NULL;
    int64_t mem_140353_cached_sizze_142130 = 0;
    unsigned char *mem_140353 = NULL;
    int64_t mem_140364_cached_sizze_142131 = 0;
    unsigned char *mem_140364 = NULL;
    int64_t mem_140365_cached_sizze_142132 = 0;
    unsigned char *mem_140365 = NULL;
    int64_t mem_140378_cached_sizze_142133 = 0;
    unsigned char *mem_140378 = NULL;
    int64_t mem_140385_cached_sizze_142134 = 0;
    unsigned char *mem_140385 = NULL;
    int64_t mem_140390_cached_sizze_142135 = 0;
    unsigned char *mem_140390 = NULL;
    int64_t mem_140401_cached_sizze_142136 = 0;
    unsigned char *mem_140401 = NULL;
    int64_t mem_140406_cached_sizze_142137 = 0;
    unsigned char *mem_140406 = NULL;
    int64_t mem_140417_cached_sizze_142138 = 0;
    unsigned char *mem_140417 = NULL;
    int64_t mem_140422_cached_sizze_142139 = 0;
    unsigned char *mem_140422 = NULL;
    int64_t mem_140433_cached_sizze_142140 = 0;
    unsigned char *mem_140433 = NULL;
    int64_t mem_140438_cached_sizze_142141 = 0;
    unsigned char *mem_140438 = NULL;
    int64_t mem_140449_cached_sizze_142142 = 0;
    unsigned char *mem_140449 = NULL;
    int64_t mem_140454_cached_sizze_142143 = 0;
    unsigned char *mem_140454 = NULL;
    int64_t mem_140465_cached_sizze_142144 = 0;
    unsigned char *mem_140465 = NULL;
    int64_t mem_140470_cached_sizze_142145 = 0;
    unsigned char *mem_140470 = NULL;
    int64_t mem_140481_cached_sizze_142146 = 0;
    unsigned char *mem_140481 = NULL;
    int64_t mem_140482_cached_sizze_142147 = 0;
    unsigned char *mem_140482 = NULL;
    int64_t mem_140483_cached_sizze_142148 = 0;
    unsigned char *mem_140483 = NULL;
    int64_t mem_140484_cached_sizze_142149 = 0;
    unsigned char *mem_140484 = NULL;
    int64_t mem_140502_cached_sizze_142150 = 0;
    unsigned char *mem_140502 = NULL;
    int64_t mem_140507_cached_sizze_142151 = 0;
    unsigned char *mem_140507 = NULL;
    int64_t mem_140511_cached_sizze_142152 = 0;
    unsigned char *mem_140511 = NULL;
    int64_t mem_140518_cached_sizze_142153 = 0;
    unsigned char *mem_140518 = NULL;
    int64_t mem_140552_cached_sizze_142154 = 0;
    unsigned char *mem_140552 = NULL;
    int64_t mem_140558_cached_sizze_142155 = 0;
    unsigned char *mem_140558 = NULL;
    int64_t mem_140563_cached_sizze_142156 = 0;
    unsigned char *mem_140563 = NULL;
    int64_t mem_140579_cached_sizze_142157 = 0;
    unsigned char *mem_140579 = NULL;
    int64_t mem_140580_cached_sizze_142158 = 0;
    unsigned char *mem_140580 = NULL;
    int64_t mem_140589_cached_sizze_142159 = 0;
    unsigned char *mem_140589 = NULL;
    int64_t mem_140590_cached_sizze_142160 = 0;
    unsigned char *mem_140590 = NULL;
    int64_t mem_140611_cached_sizze_142161 = 0;
    unsigned char *mem_140611 = NULL;
    int64_t mem_140617_cached_sizze_142162 = 0;
    unsigned char *mem_140617 = NULL;
    int64_t mem_140622_cached_sizze_142163 = 0;
    unsigned char *mem_140622 = NULL;
    int64_t mem_140638_cached_sizze_142164 = 0;
    unsigned char *mem_140638 = NULL;
    int64_t mem_140643_cached_sizze_142165 = 0;
    unsigned char *mem_140643 = NULL;
    int64_t mem_140654_cached_sizze_142166 = 0;
    unsigned char *mem_140654 = NULL;
    int64_t mem_140659_cached_sizze_142167 = 0;
    unsigned char *mem_140659 = NULL;
    int64_t mem_140670_cached_sizze_142168 = 0;
    unsigned char *mem_140670 = NULL;
    int64_t mem_140675_cached_sizze_142169 = 0;
    unsigned char *mem_140675 = NULL;
    int64_t mem_140687_cached_sizze_142170 = 0;
    unsigned char *mem_140687 = NULL;
    int64_t mem_140696_cached_sizze_142171 = 0;
    unsigned char *mem_140696 = NULL;
    int64_t mem_140697_cached_sizze_142172 = 0;
    unsigned char *mem_140697 = NULL;
    int64_t mem_140718_cached_sizze_142173 = 0;
    unsigned char *mem_140718 = NULL;
    int64_t mem_140723_cached_sizze_142174 = 0;
    unsigned char *mem_140723 = NULL;
    int64_t mem_140734_cached_sizze_142175 = 0;
    unsigned char *mem_140734 = NULL;
    int64_t mem_140735_cached_sizze_142176 = 0;
    unsigned char *mem_140735 = NULL;
    int64_t mem_140748_cached_sizze_142177 = 0;
    unsigned char *mem_140748 = NULL;
    int64_t mem_140755_cached_sizze_142178 = 0;
    unsigned char *mem_140755 = NULL;
    int64_t mem_140760_cached_sizze_142179 = 0;
    unsigned char *mem_140760 = NULL;
    int64_t mem_140771_cached_sizze_142180 = 0;
    unsigned char *mem_140771 = NULL;
    int64_t mem_140777_cached_sizze_142181 = 0;
    unsigned char *mem_140777 = NULL;
    int64_t mem_140782_cached_sizze_142182 = 0;
    unsigned char *mem_140782 = NULL;
    int64_t mem_140798_cached_sizze_142183 = 0;
    unsigned char *mem_140798 = NULL;
    int64_t mem_140799_cached_sizze_142184 = 0;
    unsigned char *mem_140799 = NULL;
    int64_t mem_140800_cached_sizze_142185 = 0;
    unsigned char *mem_140800 = NULL;
    int64_t mem_140816_cached_sizze_142186 = 0;
    unsigned char *mem_140816 = NULL;
    int64_t mem_140817_cached_sizze_142187 = 0;
    unsigned char *mem_140817 = NULL;
    int64_t mem_140818_cached_sizze_142188 = 0;
    unsigned char *mem_140818 = NULL;
    int64_t mem_140831_cached_sizze_142189 = 0;
    unsigned char *mem_140831 = NULL;
    int64_t mem_140832_cached_sizze_142190 = 0;
    unsigned char *mem_140832 = NULL;
    int64_t mem_140873_cached_sizze_142191 = 0;
    unsigned char *mem_140873 = NULL;
    int64_t mem_140874_cached_sizze_142192 = 0;
    unsigned char *mem_140874 = NULL;
    int64_t mem_140885_cached_sizze_142193 = 0;
    unsigned char *mem_140885 = NULL;
    int64_t mem_140886_cached_sizze_142194 = 0;
    unsigned char *mem_140886 = NULL;
    int64_t mem_140895_cached_sizze_142195 = 0;
    unsigned char *mem_140895 = NULL;
    int64_t mem_140896_cached_sizze_142196 = 0;
    unsigned char *mem_140896 = NULL;
    int64_t mem_140927_cached_sizze_142197 = 0;
    unsigned char *mem_140927 = NULL;
    int64_t mem_140928_cached_sizze_142198 = 0;
    unsigned char *mem_140928 = NULL;
    int64_t mem_140939_cached_sizze_142199 = 0;
    unsigned char *mem_140939 = NULL;
    int64_t mem_140940_cached_sizze_142200 = 0;
    unsigned char *mem_140940 = NULL;
    int64_t mem_140949_cached_sizze_142201 = 0;
    unsigned char *mem_140949 = NULL;
    int64_t mem_140950_cached_sizze_142202 = 0;
    unsigned char *mem_140950 = NULL;
    int64_t mem_140981_cached_sizze_142203 = 0;
    unsigned char *mem_140981 = NULL;
    int64_t mem_140982_cached_sizze_142204 = 0;
    unsigned char *mem_140982 = NULL;
    int64_t mem_140983_cached_sizze_142205 = 0;
    unsigned char *mem_140983 = NULL;
    int64_t mem_140984_cached_sizze_142206 = 0;
    unsigned char *mem_140984 = NULL;
    int64_t mem_141001_cached_sizze_142207 = 0;
    unsigned char *mem_141001 = NULL;
    int64_t mem_141002_cached_sizze_142208 = 0;
    unsigned char *mem_141002 = NULL;
    int64_t mem_141003_cached_sizze_142209 = 0;
    unsigned char *mem_141003 = NULL;
    int64_t mem_141004_cached_sizze_142210 = 0;
    unsigned char *mem_141004 = NULL;
    int64_t mem_141045_cached_sizze_142211 = 0;
    unsigned char *mem_141045 = NULL;
    int64_t mem_141046_cached_sizze_142212 = 0;
    unsigned char *mem_141046 = NULL;
    int64_t mem_141057_cached_sizze_142213 = 0;
    unsigned char *mem_141057 = NULL;
    int64_t mem_141058_cached_sizze_142214 = 0;
    unsigned char *mem_141058 = NULL;
    int64_t mem_141067_cached_sizze_142215 = 0;
    unsigned char *mem_141067 = NULL;
    int64_t mem_141068_cached_sizze_142216 = 0;
    unsigned char *mem_141068 = NULL;
    int64_t mem_141099_cached_sizze_142217 = 0;
    unsigned char *mem_141099 = NULL;
    int64_t mem_141100_cached_sizze_142218 = 0;
    unsigned char *mem_141100 = NULL;
    int64_t mem_141109_cached_sizze_142219 = 0;
    unsigned char *mem_141109 = NULL;
    int64_t mem_141110_cached_sizze_142220 = 0;
    unsigned char *mem_141110 = NULL;
    int64_t mem_141131_cached_sizze_142221 = 0;
    unsigned char *mem_141131 = NULL;
    int64_t mem_141132_cached_sizze_142222 = 0;
    unsigned char *mem_141132 = NULL;
    int64_t mem_141143_cached_sizze_142223 = 0;
    unsigned char *mem_141143 = NULL;
    int64_t mem_141144_cached_sizze_142224 = 0;
    unsigned char *mem_141144 = NULL;
    int64_t mem_141153_cached_sizze_142225 = 0;
    unsigned char *mem_141153 = NULL;
    int64_t mem_141154_cached_sizze_142226 = 0;
    unsigned char *mem_141154 = NULL;
    int64_t mem_141185_cached_sizze_142227 = 0;
    unsigned char *mem_141185 = NULL;
    int64_t mem_141186_cached_sizze_142228 = 0;
    unsigned char *mem_141186 = NULL;
    int64_t mem_141197_cached_sizze_142229 = 0;
    unsigned char *mem_141197 = NULL;
    int64_t mem_141198_cached_sizze_142230 = 0;
    unsigned char *mem_141198 = NULL;
    int64_t mem_141207_cached_sizze_142231 = 0;
    unsigned char *mem_141207 = NULL;
    int64_t mem_141208_cached_sizze_142232 = 0;
    unsigned char *mem_141208 = NULL;
    int64_t mem_141240_cached_sizze_142233 = 0;
    unsigned char *mem_141240 = NULL;
    int64_t mem_141241_cached_sizze_142234 = 0;
    unsigned char *mem_141241 = NULL;
    int64_t mem_141242_cached_sizze_142235 = 0;
    unsigned char *mem_141242 = NULL;
    int64_t mem_141259_cached_sizze_142236 = 0;
    unsigned char *mem_141259 = NULL;
    int64_t mem_141260_cached_sizze_142237 = 0;
    unsigned char *mem_141260 = NULL;
    int64_t mem_141261_cached_sizze_142238 = 0;
    unsigned char *mem_141261 = NULL;
    int64_t mem_141262_cached_sizze_142239 = 0;
    unsigned char *mem_141262 = NULL;
    int64_t mem_141303_cached_sizze_142240 = 0;
    unsigned char *mem_141303 = NULL;
    int64_t mem_141308_cached_sizze_142241 = 0;
    unsigned char *mem_141308 = NULL;
    int64_t mem_141322_cached_sizze_142242 = 0;
    unsigned char *mem_141322 = NULL;
    int64_t mem_141323_cached_sizze_142243 = 0;
    unsigned char *mem_141323 = NULL;
    int64_t mem_141342_cached_sizze_142244 = 0;
    unsigned char *mem_141342 = NULL;
    int64_t mem_141343_cached_sizze_142245 = 0;
    unsigned char *mem_141343 = NULL;
    int64_t mem_141344_cached_sizze_142246 = 0;
    unsigned char *mem_141344 = NULL;
    int64_t mem_141381_cached_sizze_142247 = 0;
    unsigned char *mem_141381 = NULL;
    int64_t mem_141388_cached_sizze_142248 = 0;
    unsigned char *mem_141388 = NULL;
    int64_t mem_141393_cached_sizze_142249 = 0;
    unsigned char *mem_141393 = NULL;
    int64_t mem_141404_cached_sizze_142250 = 0;
    unsigned char *mem_141404 = NULL;
    int64_t mem_141405_cached_sizze_142251 = 0;
    unsigned char *mem_141405 = NULL;
    int64_t mem_141414_cached_sizze_142252 = 0;
    unsigned char *mem_141414 = NULL;
    int64_t mem_141415_cached_sizze_142253 = 0;
    unsigned char *mem_141415 = NULL;
    int64_t mem_141436_cached_sizze_142254 = 0;
    unsigned char *mem_141436 = NULL;
    int64_t mem_141437_cached_sizze_142255 = 0;
    unsigned char *mem_141437 = NULL;
    int64_t mem_141438_cached_sizze_142256 = 0;
    unsigned char *mem_141438 = NULL;
    int64_t mem_141439_cached_sizze_142257 = 0;
    unsigned char *mem_141439 = NULL;
    int64_t mem_141464_cached_sizze_142258 = 0;
    unsigned char *mem_141464 = NULL;
    int64_t mem_141465_cached_sizze_142259 = 0;
    unsigned char *mem_141465 = NULL;
    int64_t mem_141478_cached_sizze_142260 = 0;
    unsigned char *mem_141478 = NULL;
    int64_t mem_141488_cached_sizze_142261 = 0;
    unsigned char *mem_141488 = NULL;
    int64_t mem_141489_cached_sizze_142262 = 0;
    unsigned char *mem_141489 = NULL;
    int64_t mem_141515_cached_sizze_142263 = 0;
    unsigned char *mem_141515 = NULL;
    int64_t mem_141536_cached_sizze_142264 = 0;
    unsigned char *mem_141536 = NULL;
    int64_t mem_141537_cached_sizze_142265 = 0;
    unsigned char *mem_141537 = NULL;
    struct memblock mem_141527;
    
    mem_141527.references = NULL;
    
    struct memblock mem_141526;
    
    mem_141526.references = NULL;
    
    struct memblock mem_141510;
    
    mem_141510.references = NULL;
    
    struct memblock mem_141479;
    
    mem_141479.references = NULL;
    
    struct memblock mem_141321;
    
    mem_141321.references = NULL;
    
    struct memblock mem_141320;
    
    mem_141320.references = NULL;
    
    struct memblock mem_141319;
    
    mem_141319.references = NULL;
    
    struct memblock mem_141239;
    
    mem_141239.references = NULL;
    
    struct memblock mem_140686;
    
    mem_140686.references = NULL;
    
    struct memblock mem_out_141584;
    
    mem_out_141584.references = NULL;
    
    struct memblock mem_out_141583;
    
    mem_out_141583.references = NULL;
    
    struct memblock mem_out_141582;
    
    mem_out_141582.references = NULL;
    
    struct memblock mem_out_141581;
    
    mem_out_141581.references = NULL;
    
    struct memblock mem_out_141580;
    
    mem_out_141580.references = NULL;
    
    struct memblock mem_out_141579;
    
    mem_out_141579.references = NULL;
    
    struct memblock mem_out_141578;
    
    mem_out_141578.references = NULL;
    
    struct memblock mem_out_141577;
    
    mem_out_141577.references = NULL;
    
    struct memblock mem_out_141576;
    
    mem_out_141576.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (mem_139503_cached_sizze_142030 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139503, &mem_139503_cached_sizze_142030, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139508_cached_sizze_142031 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139508, &mem_139508_cached_sizze_142031, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138363 = 0; i_138363 < (int64_t) 16; i_138363++) {
        // futhark/microgpt.fut:457:41-50
        
        int64_t tmp_124004 = ((int64_t *) tokens_mem_139500.mem)[i_138363];
        
        // futhark/microgpt.fut:457:37-51
        
        bool x_124005 = sle64((int64_t) 0, tmp_124004);
        
        // futhark/microgpt.fut:457:37-51
        
        bool y_124006 = slt64(tmp_124004, (int64_t) 27);
        
        // futhark/microgpt.fut:457:37-51
        
        bool bounds_check_124007 = x_124005 && y_124006;
        
        // futhark/microgpt.fut:457:37-51
        
        bool index_certs_124008;
        
        if (!bounds_check_124007) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_124004, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:457:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:457:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138359 = 0; i_138359 < (int64_t) 16; i_138359++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_124015 = ((double *) wte_mem_139496.mem)[tmp_124004 * (int64_t) 16 + i_138359];
            
            ((double *) mem_139508)[i_138359] = lifted_lambda_res_124015;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139503, i_138363 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139508, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139519_cached_sizze_142032 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139519, &mem_139519_cached_sizze_142032, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139520_cached_sizze_142033 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139520, &mem_139520_cached_sizze_142033, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139521_cached_sizze_142034 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139521, &mem_139521_cached_sizze_142034, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138371 = 0; i_138371 < (int64_t) 16; i_138371++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_129052;
        double r_129054 = 0.0;
        
        for (int64_t i_129053 = 0; i_129053 < (int64_t) 16; i_129053++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_129055 = ((double *) wpe_mem_139494.mem)[i_138371 * (int64_t) 16 + i_129053];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_129056 = ((double *) mem_139503)[i_138371 * (int64_t) 16 + i_129053];
            
            // futhark/microgpt.fut:269:63-99
            
            double zp_res_129057 = zp_lhs_129055 + zp_rhs_129056;
            
            // futhark/microgpt.fut:269:79-142
            
            double zt_res_129058 = zp_res_129057 * zp_res_129057;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_129059 = r_129054 + zt_res_129058;
            double r_tmp_141590 = zp_res_129059;
            
            r_129054 = r_tmp_141590;
        }
        defunc_0_lifted_lambda_res_129052 = r_129054;
        // futhark/microgpt.fut:269:42-161
        
        double zs_res_129060 = defunc_0_lifted_lambda_res_129052 / 16.0;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_129067;
        double r_129069 = 0.0;
        
        for (int64_t i_129068 = 0; i_129068 < (int64_t) 16; i_129068++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_129070 = ((double *) wpe_mem_139494.mem)[i_138371 * (int64_t) 16 + i_129068];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_129071 = ((double *) mem_139503)[i_138371 * (int64_t) 16 + i_129068];
            
            // futhark/microgpt.fut:385:71-115
            
            double zp_res_129072 = zp_lhs_129070 + zp_rhs_129071;
            
            // futhark/microgpt.fut:385:91-166
            
            double zt_res_129073 = zp_res_129072 * zp_res_129072;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_129074 = r_129069 + zt_res_129073;
            double r_tmp_141591 = zp_res_129074;
            
            r_129069 = r_tmp_141591;
        }
        defunc_0_lifted_lambda_res_129067 = r_129069;
        // futhark/microgpt.fut:385:48-185
        
        double zs_res_129075 = defunc_0_lifted_lambda_res_129067 / 16.0;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_129085;
        double r_129087 = 0.0;
        
        for (int64_t i_129086 = 0; i_129086 < (int64_t) 16; i_129086++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_129088 = ((double *) wpe_mem_139494.mem)[i_138371 * (int64_t) 16 + i_129086];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_129089 = ((double *) mem_139503)[i_138371 * (int64_t) 16 + i_129086];
            
            // futhark/microgpt.fut:398:72-116
            
            double zp_res_129090 = zp_lhs_129088 + zp_rhs_129089;
            
            // futhark/microgpt.fut:398:92-167
            
            double zt_res_129091 = zp_res_129090 * zp_res_129090;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_129092 = r_129087 + zt_res_129091;
            double r_tmp_141592 = zp_res_129092;
            
            r_129087 = r_tmp_141592;
        }
        defunc_0_lifted_lambda_res_129085 = r_129087;
        // futhark/microgpt.fut:398:49-186
        
        double zs_res_129093 = defunc_0_lifted_lambda_res_129085 / 16.0;
        
        ((double *) mem_139519)[i_138371] = zs_res_129093;
        ((double *) mem_139520)[i_138371] = zs_res_129075;
        ((double *) mem_139521)[i_138371] = zs_res_129060;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139540_cached_sizze_142035 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139540, &mem_139540_cached_sizze_142035, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138377 = 0; i_138377 < (int64_t) 16; i_138377++) {
        // futhark/microgpt.fut:270:43-51
        
        double zp_lhs_124040 = ((double *) mem_139521)[i_138377];
        
        // futhark/microgpt.fut:270:43-79
        
        double zp_res_124041 = 1.0e-5 + zp_lhs_124040;
        
        // futhark/microgpt.fut:270:35-79
        
        double sqrt_res_124042 = futrts_sqrt64(zp_res_124041);
        
        ((double *) mem_139540)[i_138377] = sqrt_res_124042;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139547_cached_sizze_142036 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139547, &mem_139547_cached_sizze_142036, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139552_cached_sizze_142037 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139552, &mem_139552_cached_sizze_142037, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138385 = 0; i_138385 < (int64_t) 16; i_138385++) {
        // futhark/microgpt.fut:271:95-103
        
        double zs_rhs_124050 = ((double *) mem_139540)[i_138385];
        
        // futhark/microgpt.fut:271:87-103
        
        double zs_res_124051 = 1.0 / zs_rhs_124050;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138381 = 0; i_138381 < (int64_t) 16; i_138381++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_124058 = ((double *) wpe_mem_139494.mem)[i_138385 * (int64_t) 16 + i_138381];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_124059 = ((double *) mem_139503)[i_138385 * (int64_t) 16 + i_138381];
            
            // futhark/microgpt.fut:271:44-80
            
            double zp_res_124060 = zp_lhs_124058 + zp_rhs_124059;
            
            // futhark/microgpt.fut:271:60-103
            
            double zt_res_124061 = zs_res_124051 * zp_res_124060;
            
            ((double *) mem_139552)[i_138381] = zt_res_124061;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139547, i_138385 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139552, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139563_cached_sizze_142038 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139563, &mem_139563_cached_sizze_142038, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139568_cached_sizze_142039 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139568, &mem_139568_cached_sizze_142039, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138393 = 0; i_138393 < (int64_t) 16; i_138393++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138389 = 0; i_138389 < (int64_t) 16; i_138389++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_124076 = ((double *) mem_139547)[i_138393 * (int64_t) 16 + i_138389];
            
            ((double *) mem_139568)[i_138389] = lifted_lambda_res_124076;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139563, i_138393 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139568, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139579_cached_sizze_142040 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139579, &mem_139579_cached_sizze_142040, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139580_cached_sizze_142041 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139580, &mem_139580_cached_sizze_142041, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138399 = 0; i_138399 < (int64_t) 16; i_138399++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_129112;
        double r_129114 = 0.0;
        
        for (int64_t i_129113 = 0; i_129113 < (int64_t) 16; i_129113++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_129115 = ((double *) mem_139563)[i_138399 * (int64_t) 16 + i_129113];
            
            // futhark/microgpt.fut:273:65-102
            
            double zt_res_129116 = zt_lhs_129115 * zt_lhs_129115;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_129117 = r_129114 + zt_res_129116;
            double r_tmp_141600 = zp_res_129117;
            
            r_129114 = r_tmp_141600;
        }
        defunc_0_lifted_lambda_res_129112 = r_129114;
        // futhark/microgpt.fut:273:44-120
        
        double zs_res_129118 = defunc_0_lifted_lambda_res_129112 / 16.0;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_129125;
        double r_129127 = 0.0;
        
        for (int64_t i_129126 = 0; i_129126 < (int64_t) 16; i_129126++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_129128 = ((double *) mem_139563)[i_138399 * (int64_t) 16 + i_129126];
            
            // futhark/microgpt.fut:363:70-111
            
            double zt_res_129129 = zt_lhs_129128 * zt_lhs_129128;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_129130 = r_129127 + zt_res_129129;
            double r_tmp_141601 = zp_res_129130;
            
            r_129127 = r_tmp_141601;
        }
        defunc_0_lifted_lambda_res_129125 = r_129127;
        // futhark/microgpt.fut:363:48-129
        
        double zs_res_129131 = defunc_0_lifted_lambda_res_129125 / 16.0;
        
        ((double *) mem_139579)[i_138399] = zs_res_129131;
        ((double *) mem_139580)[i_138399] = zs_res_129118;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139593_cached_sizze_142042 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139593, &mem_139593_cached_sizze_142042, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138404 = 0; i_138404 < (int64_t) 16; i_138404++) {
        // futhark/microgpt.fut:274:45-55
        
        double zp_lhs_124099 = ((double *) mem_139580)[i_138404];
        
        // futhark/microgpt.fut:274:45-83
        
        double zp_res_124100 = 1.0e-5 + zp_lhs_124099;
        
        // futhark/microgpt.fut:274:37-83
        
        double sqrt_res_124101 = futrts_sqrt64(zp_res_124100);
        
        ((double *) mem_139593)[i_138404] = sqrt_res_124101;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139600_cached_sizze_142043 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139600, &mem_139600_cached_sizze_142043, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139605_cached_sizze_142044 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139605, &mem_139605_cached_sizze_142044, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138412 = 0; i_138412 < (int64_t) 16; i_138412++) {
        // futhark/microgpt.fut:275:76-86
        
        double zs_rhs_124109 = ((double *) mem_139593)[i_138412];
        
        // futhark/microgpt.fut:275:68-86
        
        double zs_res_124110 = 1.0 / zs_rhs_124109;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138408 = 0; i_138408 < (int64_t) 16; i_138408++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_124117 = ((double *) mem_139563)[i_138412 * (int64_t) 16 + i_138408];
            
            // futhark/microgpt.fut:275:46-86
            
            double zt_res_124118 = zs_res_124110 * zt_lhs_124117;
            
            ((double *) mem_139605)[i_138408] = zt_res_124118;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139600, i_138412 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139605, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139616_cached_sizze_142045 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139616, &mem_139616_cached_sizze_142045, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139621_cached_sizze_142046 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139621, &mem_139621_cached_sizze_142046, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138420 = 0; i_138420 < (int64_t) 16; i_138420++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138416 = 0; i_138416 < (int64_t) 16; i_138416++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_124133 = ((double *) mem_139600)[i_138420 * (int64_t) 16 + i_138416];
            
            ((double *) mem_139621)[i_138416] = lifted_lambda_res_124133;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139616, i_138420 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139621, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139632_cached_sizze_142047 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139632, &mem_139632_cached_sizze_142047, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139633_cached_sizze_142048 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139633, &mem_139633_cached_sizze_142048, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139634_cached_sizze_142049 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139634, &mem_139634_cached_sizze_142049, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139650_cached_sizze_142050 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139650, &mem_139650_cached_sizze_142050, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139651_cached_sizze_142051 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139651, &mem_139651_cached_sizze_142051, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139652_cached_sizze_142052 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139652, &mem_139652_cached_sizze_142052, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139665_cached_sizze_142053 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_139665, &mem_139665_cached_sizze_142053, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139666_cached_sizze_142054 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_139666, &mem_139666_cached_sizze_142054, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139667_cached_sizze_142055 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_139667, &mem_139667_cached_sizze_142055, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138448 = 0; i_138448 < (int64_t) 4; i_138448++) {
        // futhark/microgpt.fut:277:83-86
        
        int64_t zp_lhs_129212 = mul64((int64_t) 4, i_138448);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138438 = 0; i_138438 < (int64_t) 16; i_138438++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138428 = 0; i_138428 < (int64_t) 4; i_138428++) {
                // futhark/microgpt.fut:277:88-95
                
                int64_t zt_lhs_133231 = add64(zp_lhs_129212, i_138428);
                
                // futhark/microgpt.fut:277:70-97
                
                bool x_133232 = sle64((int64_t) 0, zt_lhs_133231);
                
                // futhark/microgpt.fut:277:70-97
                
                bool y_133233 = slt64(zt_lhs_133231, (int64_t) 16);
                
                // futhark/microgpt.fut:277:70-97
                
                bool bounds_check_133234 = x_133232 && y_133233;
                
                // futhark/microgpt.fut:277:70-97
                
                bool index_certs_133235;
                
                if (!bounds_check_133234) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_133231, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:277:70-97\n   #1  futhark/microgpt.fut:61:46-49\n   #2  futhark/microgpt.fut:277:49-127\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:15:29-44\n   #8  futhark/microgpt.fut:4:11-25\n   #9  futhark/microgpt.fut:15:15-45\n   #10 futhark/microgpt.fut:277:12-129\n   #11 futhark/microgpt.fut:459:5-75\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_133236;
                double r_133238 = 0.0;
                
                for (int64_t i_133237 = 0; i_133237 < (int64_t) 16; i_133237++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_133239 = ((double *) wqry_mem_139495.mem)[zt_lhs_133231 * (int64_t) 16 + i_133237];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_133240 = ((double *) mem_139616)[i_138438 * (int64_t) 16 + i_133237];
                    
                    // futhark/microgpt.fut:277:70-125
                    
                    double zt_res_133241 = zt_lhs_133239 * zt_rhs_133240;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_133242 = r_133238 + zt_res_133241;
                    double r_tmp_141616 = zp_res_133242;
                    
                    r_133238 = r_tmp_141616;
                }
                defunc_0_lifted_lambda_res_133236 = r_133238;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_133250;
                double r_133252 = 0.0;
                
                for (int64_t i_133251 = 0; i_133251 < (int64_t) 16; i_133251++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_133253 = ((double *) wkey_mem_139492.mem)[zt_lhs_133231 * (int64_t) 16 + i_133251];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_133254 = ((double *) mem_139616)[i_138438 * (int64_t) 16 + i_133251];
                    
                    // futhark/microgpt.fut:278:70-125
                    
                    double zt_res_133255 = zt_lhs_133253 * zt_rhs_133254;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_133256 = r_133252 + zt_res_133255;
                    double r_tmp_141617 = zp_res_133256;
                    
                    r_133252 = r_tmp_141617;
                }
                defunc_0_lifted_lambda_res_133250 = r_133252;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_133267;
                double r_133269 = 0.0;
                
                for (int64_t i_133268 = 0; i_133268 < (int64_t) 16; i_133268++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_133270 = ((double *) wval_mem_139498.mem)[zt_lhs_133231 * (int64_t) 16 + i_133268];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_133271 = ((double *) mem_139616)[i_138438 * (int64_t) 16 + i_133268];
                    
                    // futhark/microgpt.fut:279:70-125
                    
                    double zt_res_133272 = zt_lhs_133270 * zt_rhs_133271;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_133273 = r_133269 + zt_res_133272;
                    double r_tmp_141618 = zp_res_133273;
                    
                    r_133269 = r_tmp_141618;
                }
                defunc_0_lifted_lambda_res_133267 = r_133269;
                ((double *) mem_139665)[i_138428] = defunc_0_lifted_lambda_res_133267;
                ((double *) mem_139666)[i_138428] = defunc_0_lifted_lambda_res_133250;
                ((double *) mem_139667)[i_138428] = defunc_0_lifted_lambda_res_133236;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139650, i_138438 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139665, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139651, i_138438 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139666, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139652, i_138438 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139667, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139632, i_138448 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_139650, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139633, i_138448 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_139651, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139634, i_138448 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_139652, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139713_cached_sizze_142056 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139713, &mem_139713_cached_sizze_142056, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139714_cached_sizze_142057 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139714, &mem_139714_cached_sizze_142057, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139715_cached_sizze_142058 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139715, &mem_139715_cached_sizze_142058, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139716_cached_sizze_142059 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139716, &mem_139716_cached_sizze_142059, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139737_cached_sizze_142060 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139737, &mem_139737_cached_sizze_142060, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139738_cached_sizze_142061 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139738, &mem_139738_cached_sizze_142061, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139739_cached_sizze_142062 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139739, &mem_139739_cached_sizze_142062, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139740_cached_sizze_142063 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139740, &mem_139740_cached_sizze_142063, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139757_cached_sizze_142064 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139757, &mem_139757_cached_sizze_142064, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139758_cached_sizze_142065 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139758, &mem_139758_cached_sizze_142065, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139759_cached_sizze_142066 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139759, &mem_139759_cached_sizze_142066, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139760_cached_sizze_142067 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139760, &mem_139760_cached_sizze_142067, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138486 = 0; i_138486 < (int64_t) 4; i_138486++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138473 = 0; i_138473 < (int64_t) 16; i_138473++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138460 = 0; i_138460 < (int64_t) 16; i_138460++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_133655;
                double r_133657 = 0.0;
                
                for (int64_t i_133656 = 0; i_133656 < (int64_t) 4; i_133656++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_133658 = ((double *) mem_139634)[i_138486 * (int64_t) 64 + i_138473 * (int64_t) 4 + i_133656];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_133659 = ((double *) mem_139633)[i_138486 * (int64_t) 64 + i_138460 * (int64_t) 4 + i_133656];
                    
                    // futhark/microgpt.fut:280:111-164
                    
                    double zt_res_133660 = zt_lhs_133658 * zt_rhs_133659;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_133661 = r_133657 + zt_res_133660;
                    double r_tmp_141631 = zp_res_133661;
                    
                    r_133657 = r_tmp_141631;
                }
                defunc_0_lifted_lambda_res_133655 = r_133657;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_133668;
                double r_133670 = 0.0;
                
                for (int64_t i_133669 = 0; i_133669 < (int64_t) 4; i_133669++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_133671 = ((double *) mem_139634)[i_138486 * (int64_t) 64 + i_138473 * (int64_t) 4 + i_133669];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_133672 = ((double *) mem_139633)[i_138486 * (int64_t) 64 + i_138460 * (int64_t) 4 + i_133669];
                    
                    // futhark/microgpt.fut:322:119-178
                    
                    double zt_res_133673 = zt_lhs_133671 * zt_rhs_133672;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_133674 = r_133670 + zt_res_133673;
                    double r_tmp_141632 = zp_res_133674;
                    
                    r_133670 = r_tmp_141632;
                }
                defunc_0_lifted_lambda_res_133668 = r_133670;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_133684;
                double r_133686 = 0.0;
                
                for (int64_t i_133685 = 0; i_133685 < (int64_t) 4; i_133685++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_133687 = ((double *) mem_139634)[i_138486 * (int64_t) 64 + i_138473 * (int64_t) 4 + i_133685];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_133688 = ((double *) mem_139633)[i_138486 * (int64_t) 64 + i_138460 * (int64_t) 4 + i_133685];
                    
                    // futhark/microgpt.fut:331:119-178
                    
                    double zt_res_133689 = zt_lhs_133687 * zt_rhs_133688;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_133690 = r_133686 + zt_res_133689;
                    double r_tmp_141633 = zp_res_133690;
                    
                    r_133686 = r_tmp_141633;
                }
                defunc_0_lifted_lambda_res_133684 = r_133686;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_133702;
                double r_133704 = 0.0;
                
                for (int64_t i_133703 = 0; i_133703 < (int64_t) 4; i_133703++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_133705 = ((double *) mem_139634)[i_138486 * (int64_t) 64 + i_138473 * (int64_t) 4 + i_133703];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_133706 = ((double *) mem_139633)[i_138486 * (int64_t) 64 + i_138460 * (int64_t) 4 + i_133703];
                    
                    // futhark/microgpt.fut:347:119-178
                    
                    double zt_res_133707 = zt_lhs_133705 * zt_rhs_133706;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_133708 = r_133704 + zt_res_133707;
                    double r_tmp_141634 = zp_res_133708;
                    
                    r_133704 = r_tmp_141634;
                }
                defunc_0_lifted_lambda_res_133702 = r_133704;
                ((double *) mem_139757)[i_138460] = defunc_0_lifted_lambda_res_133702;
                ((double *) mem_139758)[i_138460] = defunc_0_lifted_lambda_res_133684;
                ((double *) mem_139759)[i_138460] = defunc_0_lifted_lambda_res_133668;
                ((double *) mem_139760)[i_138460] = defunc_0_lifted_lambda_res_133655;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139737, i_138473 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139757, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139738, i_138473 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139758, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139739, i_138473 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139759, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139740, i_138473 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139760, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139713, i_138486 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139737, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139714, i_138486 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139738, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139715, i_138486 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139739, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139716, i_138486 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139740, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139821_cached_sizze_142068 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139821, &mem_139821_cached_sizze_142068, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139822_cached_sizze_142069 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139822, &mem_139822_cached_sizze_142069, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139823_cached_sizze_142070 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139823, &mem_139823_cached_sizze_142070, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139824_cached_sizze_142071 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139824, &mem_139824_cached_sizze_142071, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139845_cached_sizze_142072 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139845, &mem_139845_cached_sizze_142072, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139846_cached_sizze_142073 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139846, &mem_139846_cached_sizze_142073, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139847_cached_sizze_142074 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139847, &mem_139847_cached_sizze_142074, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139848_cached_sizze_142075 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139848, &mem_139848_cached_sizze_142075, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139865_cached_sizze_142076 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139865, &mem_139865_cached_sizze_142076, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139866_cached_sizze_142077 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139866, &mem_139866_cached_sizze_142077, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139867_cached_sizze_142078 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139867, &mem_139867_cached_sizze_142078, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139868_cached_sizze_142079 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139868, &mem_139868_cached_sizze_142079, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138525 = 0; i_138525 < (int64_t) 4; i_138525++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138512 = 0; i_138512 < (int64_t) 16; i_138512++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138499 = 0; i_138499 < (int64_t) 16; i_138499++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_134052 = ((double *) mem_139716)[i_138525 * (int64_t) 256 + i_138512 * (int64_t) 16 + i_138499];
                
                // futhark/microgpt.fut:281:55-93
                
                double zs_res_134053 = zs_lhs_134052 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_134054 = ((double *) mask_mem_139502.mem)[i_138512 * (int64_t) 16 + i_138499];
                
                // futhark/microgpt.fut:281:80-117
                
                double zp_res_134055 = zs_res_134053 + zp_rhs_134054;
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_134062 = ((double *) mem_139715)[i_138525 * (int64_t) 256 + i_138512 * (int64_t) 16 + i_138499];
                
                // futhark/microgpt.fut:323:59-101
                
                double zs_res_134063 = zs_lhs_134062 / 2.0;
                
                // futhark/microgpt.fut:323:88-127
                
                double zp_res_134065 = zp_rhs_134054 + zs_res_134063;
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_134075 = ((double *) mem_139714)[i_138525 * (int64_t) 256 + i_138512 * (int64_t) 16 + i_138499];
                
                // futhark/microgpt.fut:332:59-101
                
                double zs_res_134076 = zs_lhs_134075 / 2.0;
                
                // futhark/microgpt.fut:332:88-127
                
                double zp_res_134078 = zp_rhs_134054 + zs_res_134076;
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_134090 = ((double *) mem_139713)[i_138525 * (int64_t) 256 + i_138512 * (int64_t) 16 + i_138499];
                
                // futhark/microgpt.fut:348:59-101
                
                double zs_res_134091 = zs_lhs_134090 / 2.0;
                
                // futhark/microgpt.fut:348:88-127
                
                double zp_res_134093 = zp_rhs_134054 + zs_res_134091;
                
                ((double *) mem_139865)[i_138499] = zp_res_134093;
                ((double *) mem_139866)[i_138499] = zp_res_134078;
                ((double *) mem_139867)[i_138499] = zp_res_134065;
                ((double *) mem_139868)[i_138499] = zp_res_134055;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139845, i_138512 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139865, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139846, i_138512 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139866, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139847, i_138512 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139867, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139848, i_138512 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139868, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139821, i_138525 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139845, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139822, i_138525 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139846, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139823, i_138525 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139847, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_139824, i_138525 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139848, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:61:13-49
    
    double defunc_0_lifted_lambda_res_126037;
    double r_126039 = 0.0;
    
    for (int64_t i_126038 = 0; i_126038 < (int64_t) 16; i_126038++) {
        // futhark/microgpt.fut:61:40-49
        
        double zp_res_126040 = 1.0 + r_126039;
        double r_tmp_141647 = zp_res_126040;
        
        r_126039 = r_tmp_141647;
    }
    defunc_0_lifted_lambda_res_126037 = r_126039;
    // futhark/microgpt.fut:61:13-49
    
    double defunc_0_lifted_lambda_res_125608;
    double r_125610 = 0.0;
    
    for (int64_t i_125609 = 0; i_125609 < (int64_t) 16; i_125609++) {
        // futhark/microgpt.fut:61:40-49
        
        double zp_res_125611 = 1.0 + r_125610;
        double r_tmp_141648 = zp_res_125611;
        
        r_125610 = r_tmp_141648;
    }
    defunc_0_lifted_lambda_res_125608 = r_125610;
    // futhark/microgpt.fut:4:11-25
    if (mem_139929_cached_sizze_142080 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139929, &mem_139929_cached_sizze_142080, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139930_cached_sizze_142081 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139930, &mem_139930_cached_sizze_142081, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139931_cached_sizze_142082 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139931, &mem_139931_cached_sizze_142082, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139932_cached_sizze_142083 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139932, &mem_139932_cached_sizze_142083, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139933_cached_sizze_142084 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139933, &mem_139933_cached_sizze_142084, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139934_cached_sizze_142085 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139934, &mem_139934_cached_sizze_142085, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139935_cached_sizze_142086 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139935, &mem_139935_cached_sizze_142086, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139936_cached_sizze_142087 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139936, &mem_139936_cached_sizze_142087, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139969_cached_sizze_142088 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139969, &mem_139969_cached_sizze_142088, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139970_cached_sizze_142089 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139970, &mem_139970_cached_sizze_142089, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139971_cached_sizze_142090 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139971, &mem_139971_cached_sizze_142090, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139972_cached_sizze_142091 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139972, &mem_139972_cached_sizze_142091, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139973_cached_sizze_142092 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139973, &mem_139973_cached_sizze_142092, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139974_cached_sizze_142093 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139974, &mem_139974_cached_sizze_142093, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139975_cached_sizze_142094 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139975, &mem_139975_cached_sizze_142094, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139976_cached_sizze_142095 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139976, &mem_139976_cached_sizze_142095, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138578 = 0; i_138578 < (int64_t) 4; i_138578++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138553 = 0; i_138553 < (int64_t) 16; i_138553++) {
            // futhark/microgpt.fut:105:13-33
            
            double defunc_0_reduce_res_138065;
            double defunc_0_reduce_res_138066;
            double defunc_0_reduce_res_138067;
            double defunc_0_reduce_res_138068;
            double defunc_0_reduce_res_138069;
            double defunc_0_reduce_res_138070;
            double redout_138530;
            double redout_138531;
            double redout_138532;
            double redout_138533;
            double redout_138534;
            double redout_138535;
            
            redout_138530 = -INFINITY;
            redout_138531 = -INFINITY;
            redout_138532 = -INFINITY;
            redout_138533 = -INFINITY;
            redout_138534 = -INFINITY;
            redout_138535 = -INFINITY;
            for (int64_t i_138536 = 0; i_138536 < (int64_t) 16; i_138536++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_135405 = ((double *) mem_139824)[i_138578 * (int64_t) 256 + i_138553 * (int64_t) 16 + i_138536];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_135415 = ((double *) mem_139823)[i_138578 * (int64_t) 256 + i_138553 * (int64_t) 16 + i_138536];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_135434 = ((double *) mem_139822)[i_138578 * (int64_t) 256 + i_138553 * (int64_t) 16 + i_138536];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_135478 = ((double *) mem_139821)[i_138578 * (int64_t) 256 + i_138553 * (int64_t) 16 + i_138536];
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_134705 = fmax64(lifted_lambda_res_135405, redout_138530);
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_134724 = fmax64(lifted_lambda_res_135415, redout_138531);
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_134746 = fmax64(lifted_lambda_res_135434, redout_138532);
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_134771 = fmax64(lifted_lambda_res_135434, redout_138533);
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_134821 = fmax64(lifted_lambda_res_135478, redout_138534);
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_134852 = fmax64(lifted_lambda_res_135478, redout_138535);
                double redout_tmp_141665 = max_res_134705;
                double redout_tmp_141666 = max_res_134724;
                double redout_tmp_141667 = max_res_134746;
                double redout_tmp_141668 = max_res_134771;
                double redout_tmp_141669 = max_res_134821;
                double redout_tmp_141670 = max_res_134852;
                
                redout_138530 = redout_tmp_141665;
                redout_138531 = redout_tmp_141666;
                redout_138532 = redout_tmp_141667;
                redout_138533 = redout_tmp_141668;
                redout_138534 = redout_tmp_141669;
                redout_138535 = redout_tmp_141670;
            }
            defunc_0_reduce_res_138065 = redout_138530;
            defunc_0_reduce_res_138066 = redout_138531;
            defunc_0_reduce_res_138067 = redout_138532;
            defunc_0_reduce_res_138068 = redout_138533;
            defunc_0_reduce_res_138069 = redout_138534;
            defunc_0_reduce_res_138070 = redout_138535;
            // futhark/microgpt.fut:343:172-198
            
            double neg_res_134779 = -defunc_0_reduce_res_138068;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_134780;
            double r_134782 = 0.0;
            
            for (int64_t i_134781 = 0; i_134781 < (int64_t) 16; i_134781++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_134783 = ((double *) mem_139822)[i_138578 * (int64_t) 256 + i_138553 * (int64_t) 16 + i_134781];
                
                // futhark/microgpt.fut:343:138-198
                
                double zp_res_134784 = neg_res_134779 + zp_lhs_134783;
                
                // futhark/microgpt.fut:343:131-198
                
                double neg_res_134785 = -zp_res_134784;
                
                // futhark/microgpt.fut:100:42-54
                
                double max_res_134786 = fmax64(0.0, neg_res_134785);
                
                // futhark/microgpt.fut:100:35-54
                
                double sgn_res_134787 = fsignum64(max_res_134786);
                
                // futhark/microgpt.fut:343:112-201
                
                double neg_res_134788 = -sgn_res_134787;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_134789 = r_134782 + neg_res_134788;
                double r_tmp_141671 = zp_res_134789;
                
                r_134782 = r_tmp_141671;
            }
            defunc_0_lifted_lambda_res_134780 = r_134782;
            // futhark/microgpt.fut:343:58-204
            
            double zp_res_134790 = defunc_0_lifted_lambda_res_125608 + defunc_0_lifted_lambda_res_134780;
            
            // futhark/microgpt.fut:343:48-204
            
            double zs_res_134791 = 1.0 / zp_res_134790;
            
            // futhark/microgpt.fut:359:172-198
            
            double neg_res_134860 = -defunc_0_reduce_res_138070;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_134861;
            double r_134863 = 0.0;
            
            for (int64_t i_134862 = 0; i_134862 < (int64_t) 16; i_134862++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_134864 = ((double *) mem_139821)[i_138578 * (int64_t) 256 + i_138553 * (int64_t) 16 + i_134862];
                
                // futhark/microgpt.fut:359:138-198
                
                double zp_res_134865 = neg_res_134860 + zp_lhs_134864;
                
                // futhark/microgpt.fut:359:131-198
                
                double neg_res_134866 = -zp_res_134865;
                
                // futhark/microgpt.fut:100:42-54
                
                double max_res_134867 = fmax64(0.0, neg_res_134866);
                
                // futhark/microgpt.fut:100:35-54
                
                double sgn_res_134868 = fsignum64(max_res_134867);
                
                // futhark/microgpt.fut:359:112-201
                
                double neg_res_134869 = -sgn_res_134868;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_134870 = r_134863 + neg_res_134869;
                double r_tmp_141672 = zp_res_134870;
                
                r_134863 = r_tmp_141672;
            }
            defunc_0_lifted_lambda_res_134861 = r_134863;
            // futhark/microgpt.fut:359:58-204
            
            double zp_res_134871 = defunc_0_lifted_lambda_res_126037 + defunc_0_lifted_lambda_res_134861;
            
            // futhark/microgpt.fut:359:48-204
            
            double zs_res_134872 = 1.0 / zp_res_134871;
            
            ((double *) mem_139969)[i_138553] = zs_res_134872;
            ((double *) mem_139970)[i_138553] = defunc_0_reduce_res_138070;
            ((double *) mem_139971)[i_138553] = defunc_0_reduce_res_138069;
            ((double *) mem_139972)[i_138553] = zs_res_134791;
            ((double *) mem_139973)[i_138553] = defunc_0_reduce_res_138068;
            ((double *) mem_139974)[i_138553] = defunc_0_reduce_res_138067;
            ((double *) mem_139975)[i_138553] = defunc_0_reduce_res_138066;
            ((double *) mem_139976)[i_138553] = defunc_0_reduce_res_138065;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139929, i_138578 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139969, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139930, i_138578 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139970, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139931, i_138578 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139971, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139932, i_138578 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139972, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139933, i_138578 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139973, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139934, i_138578 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139974, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139935, i_138578 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139975, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_139936, i_138578 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139976, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140057_cached_sizze_142096 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_140057, &mem_140057_cached_sizze_142096, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140058_cached_sizze_142097 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_140058, &mem_140058_cached_sizze_142097, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140059_cached_sizze_142098 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_140059, &mem_140059_cached_sizze_142098, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140060_cached_sizze_142099 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_140060, &mem_140060_cached_sizze_142099, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140081_cached_sizze_142100 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140081, &mem_140081_cached_sizze_142100, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140082_cached_sizze_142101 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140082, &mem_140082_cached_sizze_142101, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140083_cached_sizze_142102 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140083, &mem_140083_cached_sizze_142102, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140084_cached_sizze_142103 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140084, &mem_140084_cached_sizze_142103, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140101_cached_sizze_142104 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140101, &mem_140101_cached_sizze_142104, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140102_cached_sizze_142105 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140102, &mem_140102_cached_sizze_142105, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140103_cached_sizze_142106 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140103, &mem_140103_cached_sizze_142106, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140104_cached_sizze_142107 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140104, &mem_140104_cached_sizze_142107, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138621 = 0; i_138621 < (int64_t) 4; i_138621++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138608 = 0; i_138608 < (int64_t) 16; i_138608++) {
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_135694 = ((double *) mem_139936)[i_138621 * (int64_t) 16 + i_138608];
            
            // futhark/microgpt.fut:283:91-114
            
            double neg_res_135695 = -neg_arg0_135694;
            
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_135756 = ((double *) mem_139931)[i_138621 * (int64_t) 16 + i_138608];
            
            // futhark/microgpt.fut:352:99-125
            
            double neg_res_135757 = -neg_arg0_135756;
            
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_135733 = ((double *) mem_139934)[i_138621 * (int64_t) 16 + i_138608];
            
            // futhark/microgpt.fut:336:99-125
            
            double neg_res_135734 = -neg_arg0_135733;
            
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_135712 = ((double *) mem_139935)[i_138621 * (int64_t) 16 + i_138608];
            
            // futhark/microgpt.fut:325:99-125
            
            double neg_res_135713 = -neg_arg0_135712;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138595 = 0; i_138595 < (int64_t) 16; i_138595++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_135876 = ((double *) mem_139824)[i_138621 * (int64_t) 256 + i_138608 * (int64_t) 16 + i_138595];
                
                // futhark/microgpt.fut:283:61-114
                
                double zp_res_135877 = neg_res_135695 + zp_lhs_135876;
                
                // futhark/microgpt.fut:283:54-114
                
                double exp_res_135878 = futrts_exp64(zp_res_135877);
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_135885 = ((double *) mem_139823)[i_138621 * (int64_t) 256 + i_138608 * (int64_t) 16 + i_138595];
                
                // futhark/microgpt.fut:325:65-125
                
                double zp_res_135886 = neg_res_135713 + zp_lhs_135885;
                
                // futhark/microgpt.fut:325:58-125
                
                double exp_res_135887 = futrts_exp64(zp_res_135886);
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_135897 = ((double *) mem_139822)[i_138621 * (int64_t) 256 + i_138608 * (int64_t) 16 + i_138595];
                
                // futhark/microgpt.fut:336:65-125
                
                double zp_res_135898 = neg_res_135734 + zp_lhs_135897;
                
                // futhark/microgpt.fut:336:58-125
                
                double exp_res_135899 = futrts_exp64(zp_res_135898);
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_135911 = ((double *) mem_139821)[i_138621 * (int64_t) 256 + i_138608 * (int64_t) 16 + i_138595];
                
                // futhark/microgpt.fut:352:65-125
                
                double zp_res_135912 = neg_res_135757 + zp_lhs_135911;
                
                // futhark/microgpt.fut:352:58-125
                
                double exp_res_135913 = futrts_exp64(zp_res_135912);
                
                ((double *) mem_140101)[i_138595] = exp_res_135913;
                ((double *) mem_140102)[i_138595] = exp_res_135899;
                ((double *) mem_140103)[i_138595] = exp_res_135887;
                ((double *) mem_140104)[i_138595] = exp_res_135878;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140081, i_138608 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140101, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140082, i_138608 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140102, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140083, i_138608 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140103, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140084, i_138608 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140104, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140057, i_138621 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_140081, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140058, i_138621 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_140082, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140059, i_138621 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_140083, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140060, i_138621 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_140084, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140165_cached_sizze_142108 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_140165, &mem_140165_cached_sizze_142108, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140166_cached_sizze_142109 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_140166, &mem_140166_cached_sizze_142109, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140175_cached_sizze_142110 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140175, &mem_140175_cached_sizze_142110, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140176_cached_sizze_142111 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140176, &mem_140176_cached_sizze_142111, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138637 = 0; i_138637 < (int64_t) 4; i_138637++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138630 = 0; i_138630 < (int64_t) 16; i_138630++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_135945;
            double r_135947 = 0.0;
            
            for (int64_t i_135946 = 0; i_135946 < (int64_t) 16; i_135946++) {
                // futhark/microgpt.fut:61:46-49
                
                double lifted_lambda_res_135948 = ((double *) mem_140060)[i_138637 * (int64_t) 256 + i_138630 * (int64_t) 16 + i_135946];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_135949 = r_135947 + lifted_lambda_res_135948;
                double r_tmp_141689 = zp_res_135949;
                
                r_135947 = r_tmp_141689;
            }
            defunc_0_lifted_lambda_res_135945 = r_135947;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_135956;
            double r_135958 = 0.0;
            
            for (int64_t i_135957 = 0; i_135957 < (int64_t) 16; i_135957++) {
                // futhark/microgpt.fut:61:46-49
                
                double lifted_lambda_res_135959 = ((double *) mem_140059)[i_138637 * (int64_t) 256 + i_138630 * (int64_t) 16 + i_135957];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_135960 = r_135958 + lifted_lambda_res_135959;
                double r_tmp_141690 = zp_res_135960;
                
                r_135958 = r_tmp_141690;
            }
            defunc_0_lifted_lambda_res_135956 = r_135958;
            ((double *) mem_140175)[i_138630] = defunc_0_lifted_lambda_res_135956;
            ((double *) mem_140176)[i_138630] = defunc_0_lifted_lambda_res_135945;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140165, i_138637 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140175, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140166, i_138637 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140176, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140197_cached_sizze_142112 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_140197, &mem_140197_cached_sizze_142112, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140198_cached_sizze_142113 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_140198, &mem_140198_cached_sizze_142113, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140209_cached_sizze_142114 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140209, &mem_140209_cached_sizze_142114, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140210_cached_sizze_142115 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140210, &mem_140210_cached_sizze_142115, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140219_cached_sizze_142116 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140219, &mem_140219_cached_sizze_142116, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140220_cached_sizze_142117 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140220, &mem_140220_cached_sizze_142117, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138658 = 0; i_138658 < (int64_t) 4; i_138658++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138651 = 0; i_138651 < (int64_t) 16; i_138651++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_rhs_135980 = ((double *) mem_140166)[i_138658 * (int64_t) 16 + i_138651];
            
            // futhark/microgpt.fut:285:84-109
            
            double zs_res_135981 = 1.0 / zs_rhs_135980;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_rhs_135997 = ((double *) mem_140165)[i_138658 * (int64_t) 16 + i_138651];
            
            // futhark/microgpt.fut:327:92-120
            
            double zs_res_135998 = 1.0 / zs_rhs_135997;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138644 = 0; i_138644 < (int64_t) 16; i_138644++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_136025 = ((double *) mem_140060)[i_138658 * (int64_t) 256 + i_138651 * (int64_t) 16 + i_138644];
                
                // futhark/microgpt.fut:285:54-109
                
                double zt_res_136026 = zs_res_135981 * zt_lhs_136025;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_136033 = ((double *) mem_140059)[i_138658 * (int64_t) 256 + i_138651 * (int64_t) 16 + i_138644];
                
                // futhark/microgpt.fut:327:58-120
                
                double zt_res_136034 = zs_res_135998 * zt_lhs_136033;
                
                ((double *) mem_140219)[i_138644] = zt_res_136034;
                ((double *) mem_140220)[i_138644] = zt_res_136026;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140209, i_138651 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140219, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140210, i_138651 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140220, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140197, i_138658 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_140209, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140198, i_138658 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_140210, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140251_cached_sizze_142118 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_140251, &mem_140251_cached_sizze_142118, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140252_cached_sizze_142119 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_140252, &mem_140252_cached_sizze_142119, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140263_cached_sizze_142120 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140263, &mem_140263_cached_sizze_142120, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140264_cached_sizze_142121 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140264, &mem_140264_cached_sizze_142121, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140273_cached_sizze_142122 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140273, &mem_140273_cached_sizze_142122, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140274_cached_sizze_142123 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140274, &mem_140274_cached_sizze_142123, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138679 = 0; i_138679 < (int64_t) 4; i_138679++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138672 = 0; i_138672 < (int64_t) 16; i_138672++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138665 = 0; i_138665 < (int64_t) 16; i_138665++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_136097 = ((double *) mem_140198)[i_138679 * (int64_t) 256 + i_138672 * (int64_t) 16 + i_138665];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_136104 = ((double *) mem_140197)[i_138679 * (int64_t) 256 + i_138672 * (int64_t) 16 + i_138665];
                
                ((double *) mem_140273)[i_138665] = lifted_lambda_res_136104;
                ((double *) mem_140274)[i_138665] = lifted_lambda_res_136097;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140263, i_138672 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140273, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140264, i_138672 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140274, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140251, i_138679 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_140263, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140252, i_138679 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_140264, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140305_cached_sizze_142124 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140305, &mem_140305_cached_sizze_142124, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140311_cached_sizze_142125 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_140311, &mem_140311_cached_sizze_142125, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140316_cached_sizze_142126 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_140316, &mem_140316_cached_sizze_142126, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138692 = 0; i_138692 < (int64_t) 4; i_138692++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138688 = 0; i_138688 < (int64_t) 16; i_138688++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138684 = 0; i_138684 < (int64_t) 4; i_138684++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_124405;
                double r_124407 = 0.0;
                
                for (int64_t i_124406 = 0; i_124406 < (int64_t) 16; i_124406++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_124408 = ((double *) mem_140252)[i_138692 * (int64_t) 256 + i_138688 * (int64_t) 16 + i_124406];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_124409 = ((double *) mem_139632)[i_138692 * (int64_t) 64 + i_124406 * (int64_t) 4 + i_138684];
                    
                    // futhark/microgpt.fut:287:74-127
                    
                    double zt_res_124410 = zt_lhs_124408 * zt_rhs_124409;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_124411 = r_124407 + zt_res_124410;
                    double r_tmp_141706 = zp_res_124411;
                    
                    r_124407 = r_tmp_141706;
                }
                defunc_0_lifted_lambda_res_124405 = r_124407;
                ((double *) mem_140316)[i_138684] = defunc_0_lifted_lambda_res_124405;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140311, i_138688 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140316, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140305, i_138692 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_140311, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140332_cached_sizze_142127 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140332, &mem_140332_cached_sizze_142127, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140337_cached_sizze_142128 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140337, &mem_140337_cached_sizze_142128, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138700 = 0; i_138700 < (int64_t) 16; i_138700++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138696 = 0; i_138696 < (int64_t) 16; i_138696++) {
            // futhark/microgpt.fut:288:15-18
            
            int64_t tmp_124423 = sdiv64(i_138696, (int64_t) 4);
            
            // futhark/microgpt.fut:288:4-20
            
            bool x_124424 = sle64((int64_t) 0, tmp_124423);
            
            // futhark/microgpt.fut:288:4-20
            
            bool y_124425 = slt64(tmp_124423, (int64_t) 4);
            
            // futhark/microgpt.fut:288:4-20
            
            bool bounds_check_124426 = x_124424 && y_124425;
            
            // futhark/microgpt.fut:288:4-20
            
            bool index_certs_124427;
            
            if (!bounds_check_124426) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_124423, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:288:4-20\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:280:12-288:49\n   #6  futhark/microgpt.fut:459:5-75\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:288:35-38
            
            int64_t tmp_124428 = smod64(i_138696, (int64_t) 4);
            
            // futhark/microgpt.fut:288:4-40
            
            bool x_124429 = sle64((int64_t) 0, tmp_124428);
            
            // futhark/microgpt.fut:288:4-40
            
            bool y_124430 = slt64(tmp_124428, (int64_t) 4);
            
            // futhark/microgpt.fut:288:4-40
            
            bool bounds_check_124431 = x_124429 && y_124430;
            
            // futhark/microgpt.fut:288:4-40
            
            bool index_certs_124432;
            
            if (!bounds_check_124431) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_124428, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:288:4-40\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:280:12-288:49\n   #6  futhark/microgpt.fut:459:5-75\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_124433 = ((double *) mem_140305)[tmp_124423 * (int64_t) 64 + i_138700 * (int64_t) 4 + tmp_124428];
            
            ((double *) mem_140337)[i_138696] = lifted_lambda_res_124433;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140332, i_138700 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140337, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140348_cached_sizze_142129 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140348, &mem_140348_cached_sizze_142129, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140353_cached_sizze_142130 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140353, &mem_140353_cached_sizze_142130, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138708 = 0; i_138708 < (int64_t) 16; i_138708++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138704 = 0; i_138704 < (int64_t) 16; i_138704++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_124448;
            double r_124450 = 0.0;
            
            for (int64_t i_124449 = 0; i_124449 < (int64_t) 16; i_124449++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_124451 = ((double *) wout_mem_139493.mem)[i_138704 * (int64_t) 16 + i_124449];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_124452 = ((double *) mem_140332)[i_138708 * (int64_t) 16 + i_124449];
                
                // futhark/microgpt.fut:289:64-104
                
                double zt_res_124453 = zt_lhs_124451 * zt_rhs_124452;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_124454 = r_124450 + zt_res_124453;
                double r_tmp_141711 = zp_res_124454;
                
                r_124450 = r_tmp_141711;
            }
            defunc_0_lifted_lambda_res_124448 = r_124450;
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_124455 = ((double *) mem_139563)[i_138708 * (int64_t) 16 + i_138704];
            
            // futhark/microgpt.fut:289:43-128
            
            double zp_res_124456 = defunc_0_lifted_lambda_res_124448 + zp_rhs_124455;
            
            ((double *) mem_140353)[i_138704] = zp_res_124456;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140348, i_138708 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140353, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140364_cached_sizze_142131 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140364, &mem_140364_cached_sizze_142131, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140365_cached_sizze_142132 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140365, &mem_140365_cached_sizze_142132, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138714 = 0; i_138714 < (int64_t) 16; i_138714++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_131189;
        double r_131191 = 0.0;
        
        for (int64_t i_131190 = 0; i_131190 < (int64_t) 16; i_131190++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_131192 = ((double *) mem_140348)[i_138714 * (int64_t) 16 + i_131190];
            
            // futhark/microgpt.fut:290:66-105
            
            double zt_res_131193 = zt_lhs_131192 * zt_lhs_131192;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_131194 = r_131191 + zt_res_131193;
            double r_tmp_141714 = zp_res_131194;
            
            r_131191 = r_tmp_141714;
        }
        defunc_0_lifted_lambda_res_131189 = r_131191;
        // futhark/microgpt.fut:290:45-123
        
        double zs_res_131195 = defunc_0_lifted_lambda_res_131189 / 16.0;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_131202;
        double r_131204 = 0.0;
        
        for (int64_t i_131203 = 0; i_131203 < (int64_t) 16; i_131203++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_131205 = ((double *) mem_140348)[i_138714 * (int64_t) 16 + i_131203];
            
            // futhark/microgpt.fut:315:70-113
            
            double zt_res_131206 = zt_lhs_131205 * zt_lhs_131205;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_131207 = r_131204 + zt_res_131206;
            double r_tmp_141715 = zp_res_131207;
            
            r_131204 = r_tmp_141715;
        }
        defunc_0_lifted_lambda_res_131202 = r_131204;
        // futhark/microgpt.fut:315:48-131
        
        double zs_res_131208 = defunc_0_lifted_lambda_res_131202 / 16.0;
        
        ((double *) mem_140364)[i_138714] = zs_res_131208;
        ((double *) mem_140365)[i_138714] = zs_res_131195;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140378_cached_sizze_142133 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140378, &mem_140378_cached_sizze_142133, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138719 = 0; i_138719 < (int64_t) 16; i_138719++) {
        // futhark/microgpt.fut:291:45-55
        
        double zp_lhs_124479 = ((double *) mem_140365)[i_138719];
        
        // futhark/microgpt.fut:291:45-83
        
        double zp_res_124480 = 1.0e-5 + zp_lhs_124479;
        
        // futhark/microgpt.fut:291:37-83
        
        double sqrt_res_124481 = futrts_sqrt64(zp_res_124480);
        
        ((double *) mem_140378)[i_138719] = sqrt_res_124481;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140385_cached_sizze_142134 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140385, &mem_140385_cached_sizze_142134, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140390_cached_sizze_142135 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140390, &mem_140390_cached_sizze_142135, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138727 = 0; i_138727 < (int64_t) 16; i_138727++) {
        // futhark/microgpt.fut:292:77-87
        
        double zs_rhs_124489 = ((double *) mem_140378)[i_138727];
        
        // futhark/microgpt.fut:292:69-87
        
        double zs_res_124490 = 1.0 / zs_rhs_124489;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138723 = 0; i_138723 < (int64_t) 16; i_138723++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_124497 = ((double *) mem_140348)[i_138727 * (int64_t) 16 + i_138723];
            
            // futhark/microgpt.fut:292:46-87
            
            double zt_res_124498 = zs_res_124490 * zt_lhs_124497;
            
            ((double *) mem_140390)[i_138723] = zt_res_124498;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140385, i_138727 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140390, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140401_cached_sizze_142136 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140401, &mem_140401_cached_sizze_142136, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140406_cached_sizze_142137 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140406, &mem_140406_cached_sizze_142137, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138735 = 0; i_138735 < (int64_t) 16; i_138735++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138731 = 0; i_138731 < (int64_t) 16; i_138731++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_124513 = ((double *) mem_140385)[i_138735 * (int64_t) 16 + i_138731];
            
            ((double *) mem_140406)[i_138731] = lifted_lambda_res_124513;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140401, i_138735 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140406, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140417_cached_sizze_142138 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_140417, &mem_140417_cached_sizze_142138, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140422_cached_sizze_142139 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_140422, &mem_140422_cached_sizze_142139, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138743 = 0; i_138743 < (int64_t) 16; i_138743++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138739 = 0; i_138739 < (int64_t) 64; i_138739++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_124529;
            double r_124531 = 0.0;
            
            for (int64_t i_124530 = 0; i_124530 < (int64_t) 16; i_124530++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_124532 = ((double *) wup_mem_139497.mem)[i_138739 * (int64_t) 16 + i_124530];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_124533 = ((double *) mem_140401)[i_138743 * (int64_t) 16 + i_124530];
                
                // futhark/microgpt.fut:294:63-102
                
                double zt_res_124534 = zt_lhs_124532 * zt_rhs_124533;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_124535 = r_124531 + zt_res_124534;
                double r_tmp_141723 = zp_res_124535;
                
                r_124531 = r_tmp_141723;
            }
            defunc_0_lifted_lambda_res_124529 = r_124531;
            ((double *) mem_140422)[i_138739] = defunc_0_lifted_lambda_res_124529;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140417, i_138743 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140422, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140433_cached_sizze_142140 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_140433, &mem_140433_cached_sizze_142140, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140438_cached_sizze_142141 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_140438, &mem_140438_cached_sizze_142141, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138751 = 0; i_138751 < (int64_t) 16; i_138751++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138747 = 0; i_138747 < (int64_t) 64; i_138747++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_124550 = ((double *) mem_140417)[i_138751 * (int64_t) 64 + i_138747];
            
            // futhark/microgpt.fut:295:41-69
            
            double max_res_124551 = fmax64(0.0, max_arg0_124550);
            
            ((double *) mem_140438)[i_138747] = max_res_124551;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140433, i_138751 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140438, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140449_cached_sizze_142142 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140449, &mem_140449_cached_sizze_142142, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140454_cached_sizze_142143 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140454, &mem_140454_cached_sizze_142143, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138759 = 0; i_138759 < (int64_t) 16; i_138759++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138755 = 0; i_138755 < (int64_t) 16; i_138755++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_124566;
            double r_124568 = 0.0;
            
            for (int64_t i_124567 = 0; i_124567 < (int64_t) 64; i_124567++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_124569 = ((double *) wdown_mem_139491.mem)[i_138755 * (int64_t) 64 + i_124567];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_124570 = ((double *) mem_140433)[i_138759 * (int64_t) 64 + i_124567];
                
                // futhark/microgpt.fut:296:64-105
                
                double zt_res_124571 = zt_lhs_124569 * zt_rhs_124570;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_124572 = r_124568 + zt_res_124571;
                double r_tmp_141728 = zp_res_124572;
                
                r_124568 = r_tmp_141728;
            }
            defunc_0_lifted_lambda_res_124566 = r_124568;
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_124573 = ((double *) mem_140348)[i_138759 * (int64_t) 16 + i_138755];
            
            // futhark/microgpt.fut:296:43-130
            
            double zp_res_124574 = defunc_0_lifted_lambda_res_124566 + zp_rhs_124573;
            
            ((double *) mem_140454)[i_138755] = zp_res_124574;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140449, i_138759 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140454, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140465_cached_sizze_142144 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_140465, &mem_140465_cached_sizze_142144, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140470_cached_sizze_142145 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_140470, &mem_140470_cached_sizze_142145, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138767 = 0; i_138767 < (int64_t) 16; i_138767++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138763 = 0; i_138763 < (int64_t) 27; i_138763++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_124590;
            double r_124592 = 0.0;
            
            for (int64_t i_124591 = 0; i_124591 < (int64_t) 16; i_124591++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_124593 = ((double *) wvoc_mem_139499.mem)[i_138763 * (int64_t) 16 + i_124591];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_124594 = ((double *) mem_140449)[i_138767 * (int64_t) 16 + i_124591];
                
                // futhark/microgpt.fut:297:63-103
                
                double zt_res_124595 = zt_lhs_124593 * zt_rhs_124594;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_124596 = r_124592 + zt_res_124595;
                double r_tmp_141731 = zp_res_124596;
                
                r_124592 = r_tmp_141731;
            }
            defunc_0_lifted_lambda_res_124590 = r_124592;
            ((double *) mem_140470)[i_138763] = defunc_0_lifted_lambda_res_124590;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140465, i_138767 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140470, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:61:13-49
    
    double defunc_0_lifted_lambda_res_124835;
    double r_124837 = 0.0;
    
    for (int64_t i_124836 = 0; i_124836 < (int64_t) 27; i_124836++) {
        // futhark/microgpt.fut:61:40-49
        
        double zp_res_124838 = 1.0 + r_124837;
        double r_tmp_141732 = zp_res_124838;
        
        r_124837 = r_tmp_141732;
    }
    defunc_0_lifted_lambda_res_124835 = r_124837;
    // futhark/microgpt.fut:4:11-25
    if (mem_140481_cached_sizze_142146 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_140481, &mem_140481_cached_sizze_142146, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140482_cached_sizze_142147 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_140482, &mem_140482_cached_sizze_142147, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140483_cached_sizze_142148 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_140483, &mem_140483_cached_sizze_142148, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140484_cached_sizze_142149 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_140484, &mem_140484_cached_sizze_142149, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:105:13-33
    if (mem_140502_cached_sizze_142150 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_140502, &mem_140502_cached_sizze_142150, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140507_cached_sizze_142151 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_140507, &mem_140507_cached_sizze_142151, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138797 = 0; i_138797 < (int64_t) 16; i_138797++) {
        // futhark/microgpt.fut:105:13-33
        
        double defunc_0_reduce_res_138168;
        double defunc_0_reduce_res_138169;
        double redout_138784;
        double redout_138785;
        
        redout_138784 = -INFINITY;
        redout_138785 = -INFINITY;
        for (int64_t i_138787 = 0; i_138787 < (int64_t) 27; i_138787++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_136275 = ((double *) mem_140465)[i_138797 * (int64_t) 27 + i_138787];
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138781 = 0; i_138781 < (int64_t) 27; i_138781++) {
                // futhark/microgpt.fut:302:55-306:90
                
                bool cond_136284 = i_138781 == i_138787;
                
                // futhark/microgpt.fut:302:55-306:90
                
                double lifted_lambda_res_136285;
                
                if (cond_136284) {
                    // futhark/microgpt.fut:105:13-33
                    
                    double defunc_0_reduce_res_138115;
                    double redout_138769 = -INFINITY;
                    
                    for (int64_t i_138770 = 0; i_138770 < (int64_t) 27; i_138770++) {
                        // futhark/microgpt.fut:4:11-25
                        
                        double lifted_lambda_res_138121 = ((double *) mem_140465)[i_138797 * (int64_t) 27 + i_138770];
                        
                        // futhark/microgpt.fut:105:13-33
                        
                        double max_res_138124 = fmax64(lifted_lambda_res_138121, redout_138769);
                        double redout_tmp_141741 = max_res_138124;
                        
                        redout_138769 = redout_tmp_141741;
                    }
                    defunc_0_reduce_res_138115 = redout_138769;
                    // futhark/microgpt.fut:303:67-76
                    
                    double neg_res_138126 = -defunc_0_reduce_res_138115;
                    
                    // futhark/microgpt.fut:4:11-25
                    if (mem_140511_cached_sizze_142152 < (int64_t) 216) {
                        err = lexical_realloc(ctx, &mem_140511, &mem_140511_cached_sizze_142152, (int64_t) 216);
                        if (err != FUTHARK_SUCCESS)
                            goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    for (int64_t i_138773 = 0; i_138773 < (int64_t) 27; i_138773++) {
                        // futhark/microgpt.fut:4:11-25
                        
                        double zp_lhs_138133 = ((double *) mem_140465)[i_138797 * (int64_t) 27 + i_138773];
                        
                        // futhark/microgpt.fut:303:44-76
                        
                        double zp_res_138134 = neg_res_138126 + zp_lhs_138133;
                        
                        // futhark/microgpt.fut:303:37-76
                        
                        double exp_res_138135 = futrts_exp64(zp_res_138134);
                        
                        ((double *) mem_140511)[i_138773] = exp_res_138135;
                    }
                    // futhark/microgpt.fut:61:13-49
                    
                    double defunc_0_lifted_lambda_res_138138;
                    double r_138140 = 0.0;
                    
                    for (int64_t i_138139 = 0; i_138139 < (int64_t) 27; i_138139++) {
                        // futhark/microgpt.fut:304:36-46
                        
                        double lifted_lambda_res_138141 = ((double *) mem_140511)[i_138139];
                        
                        // futhark/microgpt.fut:61:40-49
                        
                        double zp_res_138142 = r_138140 + lifted_lambda_res_138141;
                        double r_tmp_141743 = zp_res_138142;
                        
                        r_138140 = r_tmp_141743;
                    }
                    defunc_0_lifted_lambda_res_138138 = r_138140;
                    // futhark/microgpt.fut:305:53-64
                    
                    double zs_res_138143 = 1.0 / defunc_0_lifted_lambda_res_138138;
                    
                    // futhark/microgpt.fut:4:11-25
                    if (mem_140518_cached_sizze_142153 < (int64_t) 216) {
                        err = lexical_realloc(ctx, &mem_140518, &mem_140518_cached_sizze_142153, (int64_t) 216);
                        if (err != FUTHARK_SUCCESS)
                            goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    for (int64_t i_138777 = 0; i_138777 < (int64_t) 27; i_138777++) {
                        // futhark/microgpt.fut:305:37-47
                        
                        double zt_lhs_138150 = ((double *) mem_140511)[i_138777];
                        
                        // futhark/microgpt.fut:305:37-64
                        
                        double zt_res_138151 = zs_res_138143 * zt_lhs_138150;
                        
                        ((double *) mem_140518)[i_138777] = zt_res_138151;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_138158 = ((double *) target_mem_139501.mem)[i_138797 * (int64_t) 27 + i_138787];
                    
                    // futhark/microgpt.fut:306:7-49
                    
                    double zt_res_138159 = -6.25e-2 * zt_rhs_138158;
                    
                    // futhark/microgpt.fut:306:64-74
                    
                    double zs_rhs_138164 = ((double *) mem_140518)[i_138781];
                    
                    // futhark/microgpt.fut:306:56-74
                    
                    double zs_res_138165 = 1.0 / zs_rhs_138164;
                    
                    // futhark/microgpt.fut:306:25-74
                    
                    double zt_res_138166 = zt_res_138159 * zs_res_138165;
                    
                    lifted_lambda_res_136285 = zt_res_138166;
                } else {
                    lifted_lambda_res_136285 = 0.0;
                }
                ((double *) mem_140507)[i_138781] = lifted_lambda_res_136285;
            }
            // futhark/microgpt.fut:105:13-33
            
            double max_res_131345 = fmax64(lifted_lambda_res_136275, redout_138784);
            
            // futhark/microgpt.fut:105:13-33
            
            double max_res_131436 = fmax64(lifted_lambda_res_136275, redout_138785);
            
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140502, i_138787 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140507, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            
            double redout_tmp_141737 = max_res_131345;
            double redout_tmp_141738 = max_res_131436;
            
            redout_138784 = redout_tmp_141737;
            redout_138785 = redout_tmp_141738;
        }
        defunc_0_reduce_res_138168 = redout_138784;
        defunc_0_reduce_res_138169 = redout_138785;
        // futhark/microgpt.fut:4:11-25
        for (int64_t nest_i_141745 = 0; nest_i_141745 < (int64_t) 27; nest_i_141745++) {
            ((double *) mem_140484)[i_138797 * (int64_t) 27 + nest_i_141745] = defunc_0_reduce_res_138168;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t nest_i_141746 = 0; nest_i_141746 < (int64_t) 27; nest_i_141746++) {
            ((double *) mem_140482)[i_138797 * (int64_t) 27 + nest_i_141746] = defunc_0_reduce_res_138169;
        }
        // futhark/microgpt.fut:311:163-188
        
        double neg_res_131447 = -defunc_0_reduce_res_138169;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_131448;
        double r_131450 = 0.0;
        
        for (int64_t i_131449 = 0; i_131449 < (int64_t) 27; i_131449++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_131451 = ((double *) mem_140465)[i_138797 * (int64_t) 27 + i_131449];
            
            // futhark/microgpt.fut:311:138-188
            
            double zp_res_131452 = neg_res_131447 + zp_lhs_131451;
            
            // futhark/microgpt.fut:311:131-188
            
            double neg_res_131453 = -zp_res_131452;
            
            // futhark/microgpt.fut:100:42-54
            
            double max_res_131454 = fmax64(0.0, neg_res_131453);
            
            // futhark/microgpt.fut:100:35-54
            
            double sgn_res_131455 = fsignum64(max_res_131454);
            
            // futhark/microgpt.fut:311:112-191
            
            double neg_res_131456 = -sgn_res_131455;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_131457 = r_131450 + neg_res_131456;
            double r_tmp_141747 = zp_res_131457;
            
            r_131450 = r_tmp_141747;
        }
        defunc_0_lifted_lambda_res_131448 = r_131450;
        // futhark/microgpt.fut:311:58-194
        
        double zp_res_131458 = defunc_0_lifted_lambda_res_124835 + defunc_0_lifted_lambda_res_131448;
        
        // futhark/microgpt.fut:311:48-194
        
        double zs_res_131459 = 1.0 / zp_res_131458;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t nest_i_141748 = 0; nest_i_141748 < (int64_t) 27; nest_i_141748++) {
            ((double *) mem_140481)[i_138797 * (int64_t) 27 + nest_i_141748] = zs_res_131459;
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140483, i_138797 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_140502, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140552_cached_sizze_142154 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_140552, &mem_140552_cached_sizze_142154, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140558_cached_sizze_142155 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_140558, &mem_140558_cached_sizze_142155, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140563_cached_sizze_142156 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_140563, &mem_140563_cached_sizze_142156, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138812 = 0; i_138812 < (int64_t) 16; i_138812++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138808 = 0; i_138808 < (int64_t) 27; i_138808++) {
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_124632 = ((double *) mem_140484)[i_138812 * (int64_t) 27 + i_138808];
            
            // futhark/microgpt.fut:300:85-108
            
            double neg_res_124633 = -neg_arg0_124632;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138804 = 0; i_138804 < (int64_t) 27; i_138804++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_124640 = ((double *) mem_140465)[i_138812 * (int64_t) 27 + i_138804];
                
                // futhark/microgpt.fut:300:62-108
                
                double zp_res_124641 = neg_res_124633 + zp_lhs_124640;
                
                // futhark/microgpt.fut:300:55-108
                
                double exp_res_124642 = futrts_exp64(zp_res_124641);
                
                ((double *) mem_140563)[i_138804] = exp_res_124642;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140558, i_138808 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140563, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140552, i_138812 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_140558, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140579_cached_sizze_142157 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_140579, &mem_140579_cached_sizze_142157, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140580_cached_sizze_142158 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_140580, &mem_140580_cached_sizze_142158, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140589_cached_sizze_142159 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_140589, &mem_140589_cached_sizze_142159, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140590_cached_sizze_142160 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_140590, &mem_140590_cached_sizze_142160, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138825 = 0; i_138825 < (int64_t) 16; i_138825++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138818 = 0; i_138818 < (int64_t) 27; i_138818++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_136649;
            double r_136651 = 0.0;
            
            for (int64_t i_136650 = 0; i_136650 < (int64_t) 27; i_136650++) {
                // futhark/microgpt.fut:61:46-49
                
                double lifted_lambda_res_136652 = ((double *) mem_140552)[i_138825 * (int64_t) 729 + i_138818 * (int64_t) 27 + i_136650];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_136653 = r_136651 + lifted_lambda_res_136652;
                double r_tmp_141756 = zp_res_136653;
                
                r_136651 = r_tmp_141756;
            }
            defunc_0_lifted_lambda_res_136649 = r_136651;
            // futhark/microgpt.fut:307:147-186
            
            double zt_res_136661 = defunc_0_lifted_lambda_res_136649 * defunc_0_lifted_lambda_res_136649;
            
            // futhark/microgpt.fut:307:138-186
            
            double zs_res_136662 = 1.0 / zt_res_136661;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_136663;
            double r_136665 = 0.0;
            
            for (int64_t i_136664 = 0; i_136664 < (int64_t) 27; i_136664++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_136666 = ((double *) mem_140483)[i_138825 * (int64_t) 729 + i_138818 * (int64_t) 27 + i_136664];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_136667 = ((double *) mem_140552)[i_138825 * (int64_t) 729 + i_138818 * (int64_t) 27 + i_136664];
                
                // futhark/microgpt.fut:307:76-131
                
                double zt_res_136668 = zt_lhs_136666 * zt_rhs_136667;
                
                // futhark/microgpt.fut:307:102-186
                
                double zt_res_136669 = zs_res_136662 * zt_res_136668;
                
                // futhark/microgpt.fut:307:68-186
                
                double neg_res_136670 = -zt_res_136669;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_136671 = r_136665 + neg_res_136670;
                double r_tmp_141757 = zp_res_136671;
                
                r_136665 = r_tmp_141757;
            }
            defunc_0_lifted_lambda_res_136663 = r_136665;
            ((double *) mem_140589)[i_138818] = defunc_0_lifted_lambda_res_136663;
            ((double *) mem_140590)[i_138818] = defunc_0_lifted_lambda_res_136649;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140579, i_138825 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140589, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140580, i_138825 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140590, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140611_cached_sizze_142161 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_140611, &mem_140611_cached_sizze_142161, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140617_cached_sizze_142162 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_140617, &mem_140617_cached_sizze_142162, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140622_cached_sizze_142163 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_140622, &mem_140622_cached_sizze_142163, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138838 = 0; i_138838 < (int64_t) 16; i_138838++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138834 = 0; i_138834 < (int64_t) 27; i_138834++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_rhs_124772 = ((double *) mem_140580)[i_138838 * (int64_t) 27 + i_138834];
            
            // futhark/microgpt.fut:308:92-119
            
            double zs_res_124773 = 1.0 / zs_rhs_124772;
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_124774 = ((double *) mem_140579)[i_138838 * (int64_t) 27 + i_138834];
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138830 = 0; i_138830 < (int64_t) 27; i_138830++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_124781 = ((double *) mem_140483)[i_138838 * (int64_t) 729 + i_138834 * (int64_t) 27 + i_138830];
                
                // futhark/microgpt.fut:308:59-119
                
                double zt_res_124782 = zs_res_124773 * zt_lhs_124781;
                
                // futhark/microgpt.fut:308:87-145
                
                double zp_res_124783 = zp_rhs_124774 + zt_res_124782;
                
                ((double *) mem_140622)[i_138830] = zp_res_124783;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140617, i_138834 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140622, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140611, i_138838 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_140617, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140638_cached_sizze_142164 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_140638, &mem_140638_cached_sizze_142164, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140643_cached_sizze_142165 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_140643, &mem_140643_cached_sizze_142165, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138846 = 0; i_138846 < (int64_t) 16; i_138846++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138842 = 0; i_138842 < (int64_t) 27; i_138842++) {
            double f_elem_124796 = ((double *) mem_140484)[i_138846 * (int64_t) 27 + i_138842];
            
            // futhark/microgpt.fut:309:110-135
            
            double neg_res_124801 = -f_elem_124796;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_124802;
            double r_124804 = 0.0;
            
            for (int64_t i_124803 = 0; i_124803 < (int64_t) 27; i_124803++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_124805 = ((double *) mem_140465)[i_138846 * (int64_t) 27 + i_124803];
                
                // futhark/microgpt.fut:309:85-135
                
                double zp_res_124806 = neg_res_124801 + zp_lhs_124805;
                
                // futhark/microgpt.fut:309:78-135
                
                double exp_res_124807 = futrts_exp64(zp_res_124806);
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_124808 = ((double *) mem_140611)[i_138846 * (int64_t) 729 + i_138842 * (int64_t) 27 + i_124803];
                
                // futhark/microgpt.fut:309:78-170
                
                double zt_res_124809 = exp_res_124807 * zt_rhs_124808;
                
                // futhark/microgpt.fut:309:70-170
                
                double neg_res_124810 = -zt_res_124809;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_124811 = r_124804 + neg_res_124810;
                double r_tmp_141763 = zp_res_124811;
                
                r_124804 = r_tmp_141763;
            }
            defunc_0_lifted_lambda_res_124802 = r_124804;
            ((double *) mem_140643)[i_138842] = defunc_0_lifted_lambda_res_124802;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140638, i_138846 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140643, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140654_cached_sizze_142166 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_140654, &mem_140654_cached_sizze_142166, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140659_cached_sizze_142167 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_140659, &mem_140659_cached_sizze_142167, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138854 = 0; i_138854 < (int64_t) 16; i_138854++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138850 = 0; i_138850 < (int64_t) 27; i_138850++) {
            double f_elem_124872 = ((double *) mem_140465)[i_138854 * (int64_t) 27 + i_138850];
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_124877;
            double r_124879 = 0.0;
            
            for (int64_t i_124878 = 0; i_124878 < (int64_t) 27; i_124878++) {
                // futhark/microgpt.fut:61:46-49
                
                double neg_arg0_124880 = ((double *) mem_140484)[i_138854 * (int64_t) 27 + i_124878];
                
                // futhark/microgpt.fut:312:89-113
                
                double neg_res_124881 = -neg_arg0_124880;
                
                // futhark/microgpt.fut:312:66-113
                
                double zp_res_124882 = f_elem_124872 + neg_res_124881;
                
                // futhark/microgpt.fut:312:59-113
                
                double exp_res_124883 = futrts_exp64(zp_res_124882);
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_124884 = ((double *) mem_140611)[i_138854 * (int64_t) 729 + i_124878 * (int64_t) 27 + i_138850];
                
                // futhark/microgpt.fut:312:59-146
                
                double zt_res_124885 = exp_res_124883 * zt_rhs_124884;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_124886 = r_124879 + zt_res_124885;
                double r_tmp_141766 = zp_res_124886;
                
                r_124879 = r_tmp_141766;
            }
            defunc_0_lifted_lambda_res_124877 = r_124879;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_124887;
            double r_124889 = 0.0;
            
            for (int64_t i_124888 = 0; i_124888 < (int64_t) 27; i_124888++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_124890 = ((double *) mem_140638)[i_138854 * (int64_t) 27 + i_124888];
                
                // futhark/microgpt.fut:61:46-49
                
                double neg_arg0_124891 = ((double *) mem_140482)[i_138854 * (int64_t) 27 + i_124888];
                
                // futhark/microgpt.fut:312:260-284
                
                double neg_res_124892 = -neg_arg0_124891;
                
                // futhark/microgpt.fut:312:237-284
                
                double zp_res_124893 = f_elem_124872 + neg_res_124892;
                
                // futhark/microgpt.fut:312:230-284
                
                double neg_res_124894 = -zp_res_124893;
                
                // futhark/microgpt.fut:100:42-54
                
                double max_res_124895 = fmax64(0.0, neg_res_124894);
                
                // futhark/microgpt.fut:100:35-54
                
                double sgn_res_124896 = fsignum64(max_res_124895);
                
                // futhark/microgpt.fut:312:211-287
                
                double neg_res_124897 = -sgn_res_124896;
                
                // futhark/microgpt.fut:312:202-288
                
                double zp_res_124898 = 1.0 + neg_res_124897;
                
                // futhark/microgpt.fut:312:178-288
                
                double zt_res_124899 = zt_lhs_124890 * zp_res_124898;
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_124900 = ((double *) mem_140481)[i_138854 * (int64_t) 27 + i_124888];
                
                // futhark/microgpt.fut:312:197-314
                
                double zt_res_124901 = zt_res_124899 * zt_rhs_124900;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_124902 = r_124889 + zt_res_124901;
                double r_tmp_141767 = zp_res_124902;
                
                r_124889 = r_tmp_141767;
            }
            defunc_0_lifted_lambda_res_124887 = r_124889;
            // futhark/microgpt.fut:312:36-316
            
            double zp_res_124903 = defunc_0_lifted_lambda_res_124877 + defunc_0_lifted_lambda_res_124887;
            
            ((double *) mem_140659)[i_138850] = zp_res_124903;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140654, i_138854 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140659, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140670_cached_sizze_142168 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140670, &mem_140670_cached_sizze_142168, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140675_cached_sizze_142169 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140675, &mem_140675_cached_sizze_142169, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138862 = 0; i_138862 < (int64_t) 16; i_138862++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138858 = 0; i_138858 < (int64_t) 16; i_138858++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_124918;
            double r_124920 = 0.0;
            
            for (int64_t i_124919 = 0; i_124919 < (int64_t) 27; i_124919++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_124921 = ((double *) mem_140654)[i_138862 * (int64_t) 27 + i_124919];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_124922 = ((double *) wvoc_mem_139499.mem)[i_124919 * (int64_t) 16 + i_138858];
                
                // futhark/microgpt.fut:313:67-111
                
                double zt_res_124923 = zt_lhs_124921 * zt_rhs_124922;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_124924 = r_124920 + zt_res_124923;
                double r_tmp_141770 = zp_res_124924;
                
                r_124920 = r_tmp_141770;
            }
            defunc_0_lifted_lambda_res_124918 = r_124920;
            ((double *) mem_140675)[i_138858] = defunc_0_lifted_lambda_res_124918;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140670, i_138862 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140675, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_140686, (int64_t) 8192, "mem_140686")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140687_cached_sizze_142170 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_140687, &mem_140687_cached_sizze_142170, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140696_cached_sizze_142171 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_140696, &mem_140696_cached_sizze_142171, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140697_cached_sizze_142172 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_140697, &mem_140697_cached_sizze_142172, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138875 = 0; i_138875 < (int64_t) 16; i_138875++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138868 = 0; i_138868 < (int64_t) 64; i_138868++) {
            // futhark/microgpt.fut:4:11-25
            
            double indicatorp_arg0_136696 = ((double *) mem_140417)[i_138875 * (int64_t) 64 + i_138868];
            
            // futhark/microgpt.fut:100:42-54
            
            double max_res_136697 = fmax64(0.0, indicatorp_arg0_136696);
            
            // futhark/microgpt.fut:100:35-54
            
            double sgn_res_136698 = fsignum64(max_res_136697);
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_136699;
            double r_136701 = 0.0;
            
            for (int64_t i_136700 = 0; i_136700 < (int64_t) 16; i_136700++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_136702 = ((double *) mem_140670)[i_138875 * (int64_t) 16 + i_136700];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_136703 = ((double *) wdown_mem_139491.mem)[i_136700 * (int64_t) 64 + i_138868];
                
                // futhark/microgpt.fut:314:105-151
                
                double zt_res_136704 = zt_lhs_136702 * zt_rhs_136703;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_136705 = r_136701 + zt_res_136704;
                double r_tmp_141775 = zp_res_136705;
                
                r_136701 = r_tmp_141775;
            }
            defunc_0_lifted_lambda_res_136699 = r_136701;
            // futhark/microgpt.fut:314:46-153
            
            double zt_res_136706 = sgn_res_136698 * defunc_0_lifted_lambda_res_136699;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_136713;
            double r_136715 = 0.0;
            
            for (int64_t i_136714 = 0; i_136714 < (int64_t) 16; i_136714++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_136716 = ((double *) mem_140670)[i_136714 * (int64_t) 16 + i_138875];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_136717 = ((double *) mem_140433)[i_136714 * (int64_t) 64 + i_138868];
                
                // futhark/microgpt.fut:396:69-113
                
                double zt_res_136718 = zt_lhs_136716 * zt_rhs_136717;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_136719 = r_136715 + zt_res_136718;
                double r_tmp_141776 = zp_res_136719;
                
                r_136715 = r_tmp_141776;
            }
            defunc_0_lifted_lambda_res_136713 = r_136715;
            ((double *) mem_140696)[i_138868] = defunc_0_lifted_lambda_res_136713;
            ((double *) mem_140697)[i_138868] = zt_res_136706;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140686.mem, i_138875 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140696, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140687, i_138875 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140697, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140718_cached_sizze_142173 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140718, &mem_140718_cached_sizze_142173, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140723_cached_sizze_142174 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140723, &mem_140723_cached_sizze_142174, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138884 = 0; i_138884 < (int64_t) 16; i_138884++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138880 = 0; i_138880 < (int64_t) 16; i_138880++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_124988;
            double r_124990 = 0.0;
            
            for (int64_t i_124989 = 0; i_124989 < (int64_t) 64; i_124989++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_124991 = ((double *) mem_140687)[i_138884 * (int64_t) 64 + i_124989];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_124992 = ((double *) wup_mem_139497.mem)[i_124989 * (int64_t) 16 + i_138880];
                
                // futhark/microgpt.fut:317:71-115
                
                double zt_res_124993 = zt_lhs_124991 * zt_rhs_124992;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_124994 = r_124990 + zt_res_124993;
                double r_tmp_141779 = zp_res_124994;
                
                r_124990 = r_tmp_141779;
            }
            defunc_0_lifted_lambda_res_124988 = r_124990;
            ((double *) mem_140723)[i_138880] = defunc_0_lifted_lambda_res_124988;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140718, i_138884 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140723, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140734_cached_sizze_142175 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140734, &mem_140734_cached_sizze_142175, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140735_cached_sizze_142176 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140735, &mem_140735_cached_sizze_142176, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138890 = 0; i_138890 < (int64_t) 16; i_138890++) {
        // futhark/microgpt.fut:316:47-59
        
        double zp_lhs_128858 = ((double *) mem_140364)[i_138890];
        
        // futhark/microgpt.fut:316:47-87
        
        double zp_res_128859 = 1.0e-5 + zp_lhs_128858;
        
        // futhark/microgpt.fut:316:39-87
        
        double sqrt_res_128860 = futrts_sqrt64(zp_res_128859);
        
        // futhark/microgpt.fut:318:129-158
        
        double zt_res_128868 = sqrt_res_128860 * sqrt_res_128860;
        
        // futhark/microgpt.fut:318:120-158
        
        double zs_res_128869 = 1.0 / zt_res_128868;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_128870;
        double r_128872 = 0.0;
        
        for (int64_t i_128871 = 0; i_128871 < (int64_t) 16; i_128871++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_128873 = ((double *) mem_140718)[i_138890 * (int64_t) 16 + i_128871];
            
            // futhark/microgpt.fut:61:46-49
            
            double zt_rhs_128874 = ((double *) mem_140348)[i_138890 * (int64_t) 16 + i_128871];
            
            // futhark/microgpt.fut:318:69-113
            
            double zt_res_128875 = zt_lhs_128873 * zt_rhs_128874;
            
            // futhark/microgpt.fut:318:90-158
            
            double zt_res_128876 = zs_res_128869 * zt_res_128875;
            
            // futhark/microgpt.fut:318:61-158
            
            double neg_res_128877 = -zt_res_128876;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_128878 = r_128872 + neg_res_128877;
            double r_tmp_141782 = zp_res_128878;
            
            r_128872 = r_tmp_141782;
        }
        defunc_0_lifted_lambda_res_128870 = r_128872;
        ((double *) mem_140734)[i_138890] = defunc_0_lifted_lambda_res_128870;
        ((double *) mem_140735)[i_138890] = sqrt_res_128860;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140748_cached_sizze_142177 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140748, &mem_140748_cached_sizze_142177, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138895 = 0; i_138895 < (int64_t) 16; i_138895++) {
        // futhark/microgpt.fut:319:39-51
        
        double zt_lhs_125022 = ((double *) mem_140734)[i_138895];
        
        // futhark/microgpt.fut:319:93-105
        
        double zp_lhs_125023 = ((double *) mem_140364)[i_138895];
        
        // futhark/microgpt.fut:319:93-133
        
        double zp_res_125024 = 1.0e-5 + zp_lhs_125023;
        
        // futhark/microgpt.fut:319:85-133
        
        double sqrt_res_125025 = futrts_sqrt64(zp_res_125024);
        
        // futhark/microgpt.fut:319:71-135
        
        double zt_res_125026 = 2.0 * sqrt_res_125025;
        
        // futhark/microgpt.fut:319:57-135
        
        double zs_res_125027 = 1.0 / zt_res_125026;
        
        // futhark/microgpt.fut:319:39-135
        
        double zt_res_125028 = zt_lhs_125022 * zs_res_125027;
        
        ((double *) mem_140748)[i_138895] = zt_res_125028;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140755_cached_sizze_142178 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140755, &mem_140755_cached_sizze_142178, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140760_cached_sizze_142179 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140760, &mem_140760_cached_sizze_142179, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138903 = 0; i_138903 < (int64_t) 16; i_138903++) {
        // futhark/microgpt.fut:320:98-110
        
        double zs_rhs_125036 = ((double *) mem_140735)[i_138903];
        
        // futhark/microgpt.fut:320:90-110
        
        double zs_res_125037 = 1.0 / zs_rhs_125036;
        
        // futhark/microgpt.fut:320:120-132
        
        double zs_lhs_125038 = ((double *) mem_140748)[i_138903];
        
        // futhark/microgpt.fut:320:120-147
        
        double zs_res_125039 = zs_lhs_125038 / 16.0;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138899 = 0; i_138899 < (int64_t) 16; i_138899++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_125046 = ((double *) mem_140670)[i_138903 * (int64_t) 16 + i_138899];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_125047 = ((double *) mem_140718)[i_138903 * (int64_t) 16 + i_138899];
            
            // futhark/microgpt.fut:320:64-110
            
            double zt_res_125048 = zs_res_125037 * zt_lhs_125047;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_125049 = ((double *) mem_140348)[i_138903 * (int64_t) 16 + i_138899];
            
            // futhark/microgpt.fut:320:133-172
            
            double zt_res_125050 = zs_res_125039 * zt_rhs_125049;
            
            // futhark/microgpt.fut:320:149-232
            
            double zp_res_125051 = zt_res_125050 + zt_res_125050;
            
            // futhark/microgpt.fut:320:85-232
            
            double zp_res_125052 = zt_res_125048 + zp_res_125051;
            
            // futhark/microgpt.fut:320:37-232
            
            double zp_res_125053 = zp_lhs_125046 + zp_res_125052;
            
            ((double *) mem_140760)[i_138899] = zp_res_125053;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140755, i_138903 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140760, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140771_cached_sizze_142180 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140771, &mem_140771_cached_sizze_142180, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140777_cached_sizze_142181 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_140777, &mem_140777_cached_sizze_142181, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140782_cached_sizze_142182 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_140782, &mem_140782_cached_sizze_142182, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138915 = 0; i_138915 < (int64_t) 4; i_138915++) {
        // futhark/microgpt.fut:321:122-125
        
        int64_t zp_lhs_125058 = mul64((int64_t) 4, i_138915);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138911 = 0; i_138911 < (int64_t) 16; i_138911++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138907 = 0; i_138907 < (int64_t) 4; i_138907++) {
                // futhark/microgpt.fut:321:127-135
                
                int64_t zt_rhs_125067 = add64(zp_lhs_125058, i_138907);
                
                // futhark/microgpt.fut:321:100-137
                
                bool x_125068 = sle64((int64_t) 0, zt_rhs_125067);
                
                // futhark/microgpt.fut:321:100-137
                
                bool y_125069 = slt64(zt_rhs_125067, (int64_t) 16);
                
                // futhark/microgpt.fut:321:100-137
                
                bool bounds_check_125070 = x_125068 && y_125069;
                
                // futhark/microgpt.fut:321:100-137
                
                bool index_certs_125071;
                
                if (!bounds_check_125070) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_rhs_125067, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:321:100-137\n   #1  futhark/microgpt.fut:61:46-49\n   #2  futhark/microgpt.fut:321:53-139\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:15:29-44\n   #8  futhark/microgpt.fut:4:11-25\n   #9  futhark/microgpt.fut:15:15-45\n   #10 futhark/microgpt.fut:321:13-141\n   #11 futhark/microgpt.fut:459:5-75\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_125072;
                double r_125074 = 0.0;
                
                for (int64_t i_125073 = 0; i_125073 < (int64_t) 16; i_125073++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_125075 = ((double *) mem_140755)[i_138911 * (int64_t) 16 + i_125073];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_125076 = ((double *) wout_mem_139493.mem)[i_125073 * (int64_t) 16 + zt_rhs_125067];
                    
                    // futhark/microgpt.fut:321:75-137
                    
                    double zt_res_125077 = zt_lhs_125075 * zt_rhs_125076;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_125078 = r_125074 + zt_res_125077;
                    double r_tmp_141789 = zp_res_125078;
                    
                    r_125074 = r_tmp_141789;
                }
                defunc_0_lifted_lambda_res_125072 = r_125074;
                ((double *) mem_140782)[i_138907] = defunc_0_lifted_lambda_res_125072;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140777, i_138911 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140782, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140771, i_138915 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_140777, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140798_cached_sizze_142183 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140798, &mem_140798_cached_sizze_142183, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140799_cached_sizze_142184 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140799, &mem_140799_cached_sizze_142184, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140800_cached_sizze_142185 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140800, &mem_140800_cached_sizze_142185, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140816_cached_sizze_142186 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_140816, &mem_140816_cached_sizze_142186, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140817_cached_sizze_142187 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_140817, &mem_140817_cached_sizze_142187, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140818_cached_sizze_142188 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_140818, &mem_140818_cached_sizze_142188, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140831_cached_sizze_142189 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_140831, &mem_140831_cached_sizze_142189, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140832_cached_sizze_142190 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_140832, &mem_140832_cached_sizze_142190, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138940 = 0; i_138940 < (int64_t) 4; i_138940++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138930 = 0; i_138930 < (int64_t) 16; i_138930++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138921 = 0; i_138921 < (int64_t) 4; i_138921++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_136876 = ((double *) mem_140771)[i_138940 * (int64_t) 64 + i_138930 * (int64_t) 4 + i_138921];
                
                ((double *) mem_140831)[i_138921] = lifted_lambda_res_136876;
                ((double *) mem_140832)[i_138921] = lifted_lambda_res_136876;
            }
            // futhark/microgpt.fut:4:11-25
            // futhark/microgpt.fut:4:11-25
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140818, i_138930 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140832, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140816, i_138930 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140831, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140817, i_138930 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140832, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140798, i_138940 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_140816, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140799, i_138940 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_140817, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140800, i_138940 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_140818, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140873_cached_sizze_142191 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_140873, &mem_140873_cached_sizze_142191, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140874_cached_sizze_142192 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_140874, &mem_140874_cached_sizze_142192, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140885_cached_sizze_142193 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140885, &mem_140885_cached_sizze_142193, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140886_cached_sizze_142194 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140886, &mem_140886_cached_sizze_142194, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140895_cached_sizze_142195 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140895, &mem_140895_cached_sizze_142195, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140896_cached_sizze_142196 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140896, &mem_140896_cached_sizze_142196, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138962 = 0; i_138962 < (int64_t) 4; i_138962++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138955 = 0; i_138955 < (int64_t) 16; i_138955++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138948 = 0; i_138948 < (int64_t) 16; i_138948++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_137206;
                double r_137208 = 0.0;
                
                for (int64_t i_137207 = 0; i_137207 < (int64_t) 4; i_137207++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_137209 = ((double *) mem_140799)[i_138962 * (int64_t) 64 + i_138955 * (int64_t) 4 + i_137207];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_137210 = ((double *) mem_139632)[i_138962 * (int64_t) 64 + i_138948 * (int64_t) 4 + i_137207];
                    
                    // futhark/microgpt.fut:334:79-139
                    
                    double zt_res_137211 = zt_lhs_137209 * zt_rhs_137210;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_137212 = r_137208 + zt_res_137211;
                    double r_tmp_141804 = zp_res_137212;
                    
                    r_137208 = r_tmp_141804;
                }
                defunc_0_lifted_lambda_res_137206 = r_137208;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_137219;
                double r_137221 = 0.0;
                
                for (int64_t i_137220 = 0; i_137220 < (int64_t) 4; i_137220++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_137222 = ((double *) mem_140798)[i_138962 * (int64_t) 64 + i_138955 * (int64_t) 4 + i_137220];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_137223 = ((double *) mem_139632)[i_138962 * (int64_t) 64 + i_138948 * (int64_t) 4 + i_137220];
                    
                    // futhark/microgpt.fut:350:79-139
                    
                    double zt_res_137224 = zt_lhs_137222 * zt_rhs_137223;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_137225 = r_137221 + zt_res_137224;
                    double r_tmp_141805 = zp_res_137225;
                    
                    r_137221 = r_tmp_141805;
                }
                defunc_0_lifted_lambda_res_137219 = r_137221;
                ((double *) mem_140895)[i_138948] = defunc_0_lifted_lambda_res_137219;
                ((double *) mem_140896)[i_138948] = defunc_0_lifted_lambda_res_137206;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140885, i_138955 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140895, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140886, i_138955 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140896, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140873, i_138962 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_140885, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140874, i_138962 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_140886, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140927_cached_sizze_142197 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_140927, &mem_140927_cached_sizze_142197, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140928_cached_sizze_142198 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_140928, &mem_140928_cached_sizze_142198, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140939_cached_sizze_142199 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140939, &mem_140939_cached_sizze_142199, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140940_cached_sizze_142200 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140940, &mem_140940_cached_sizze_142200, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140949_cached_sizze_142201 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140949, &mem_140949_cached_sizze_142201, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140950_cached_sizze_142202 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140950, &mem_140950_cached_sizze_142202, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_138983 = 0; i_138983 < (int64_t) 4; i_138983++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138976 = 0; i_138976 < (int64_t) 16; i_138976++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_138969 = 0; i_138969 < (int64_t) 16; i_138969++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_137458 = ((double *) mem_140874)[i_138983 * (int64_t) 256 + i_138976 * (int64_t) 16 + i_138969];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_137465 = ((double *) mem_140873)[i_138983 * (int64_t) 256 + i_138976 * (int64_t) 16 + i_138969];
                
                ((double *) mem_140949)[i_138969] = lifted_lambda_res_137465;
                ((double *) mem_140950)[i_138969] = lifted_lambda_res_137458;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140939, i_138976 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140949, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140940, i_138976 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140950, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140927, i_138983 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_140939, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140928, i_138983 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_140940, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140981_cached_sizze_142203 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_140981, &mem_140981_cached_sizze_142203, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140982_cached_sizze_142204 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_140982, &mem_140982_cached_sizze_142204, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140983_cached_sizze_142205 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_140983, &mem_140983_cached_sizze_142205, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140984_cached_sizze_142206 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_140984, &mem_140984_cached_sizze_142206, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141001_cached_sizze_142207 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141001, &mem_141001_cached_sizze_142207, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141002_cached_sizze_142208 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141002, &mem_141002_cached_sizze_142208, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141003_cached_sizze_142209 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141003, &mem_141003_cached_sizze_142209, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141004_cached_sizze_142210 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141004, &mem_141004_cached_sizze_142210, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139007 = 0; i_139007 < (int64_t) 4; i_139007++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_138994 = 0; i_138994 < (int64_t) 16; i_138994++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137338;
            double r_137340 = 0.0;
            
            for (int64_t i_137339 = 0; i_137339 < (int64_t) 16; i_137339++) {
                // futhark/microgpt.fut:61:46-49
                
                double lifted_lambda_res_137341 = ((double *) mem_140058)[i_139007 * (int64_t) 256 + i_138994 * (int64_t) 16 + i_137339];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137342 = r_137340 + lifted_lambda_res_137341;
                double r_tmp_141820 = zp_res_137342;
                
                r_137340 = r_tmp_141820;
            }
            defunc_0_lifted_lambda_res_137338 = r_137340;
            // futhark/microgpt.fut:339:155-200
            
            double zt_res_137350 = defunc_0_lifted_lambda_res_137338 * defunc_0_lifted_lambda_res_137338;
            
            // futhark/microgpt.fut:339:146-200
            
            double zs_res_137351 = 1.0 / zt_res_137350;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137352;
            double r_137354 = 0.0;
            
            for (int64_t i_137353 = 0; i_137353 < (int64_t) 16; i_137353++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_137355 = ((double *) mem_140928)[i_139007 * (int64_t) 256 + i_138994 * (int64_t) 16 + i_137353];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_137356 = ((double *) mem_140058)[i_139007 * (int64_t) 256 + i_138994 * (int64_t) 16 + i_137353];
                
                // futhark/microgpt.fut:339:78-139
                
                double zt_res_137357 = zt_lhs_137355 * zt_rhs_137356;
                
                // futhark/microgpt.fut:339:107-200
                
                double zt_res_137358 = zs_res_137351 * zt_res_137357;
                
                // futhark/microgpt.fut:339:70-200
                
                double neg_res_137359 = -zt_res_137358;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137360 = r_137354 + neg_res_137359;
                double r_tmp_141821 = zp_res_137360;
                
                r_137354 = r_tmp_141821;
            }
            defunc_0_lifted_lambda_res_137352 = r_137354;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137371;
            double r_137373 = 0.0;
            
            for (int64_t i_137372 = 0; i_137372 < (int64_t) 16; i_137372++) {
                // futhark/microgpt.fut:61:46-49
                
                double lifted_lambda_res_137374 = ((double *) mem_140057)[i_139007 * (int64_t) 256 + i_138994 * (int64_t) 16 + i_137372];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137375 = r_137373 + lifted_lambda_res_137374;
                double r_tmp_141822 = zp_res_137375;
                
                r_137373 = r_tmp_141822;
            }
            defunc_0_lifted_lambda_res_137371 = r_137373;
            // futhark/microgpt.fut:355:155-200
            
            double zt_res_137383 = defunc_0_lifted_lambda_res_137371 * defunc_0_lifted_lambda_res_137371;
            
            // futhark/microgpt.fut:355:146-200
            
            double zs_res_137384 = 1.0 / zt_res_137383;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137385;
            double r_137387 = 0.0;
            
            for (int64_t i_137386 = 0; i_137386 < (int64_t) 16; i_137386++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_137388 = ((double *) mem_140927)[i_139007 * (int64_t) 256 + i_138994 * (int64_t) 16 + i_137386];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_137389 = ((double *) mem_140057)[i_139007 * (int64_t) 256 + i_138994 * (int64_t) 16 + i_137386];
                
                // futhark/microgpt.fut:355:78-139
                
                double zt_res_137390 = zt_lhs_137388 * zt_rhs_137389;
                
                // futhark/microgpt.fut:355:107-200
                
                double zt_res_137391 = zs_res_137384 * zt_res_137390;
                
                // futhark/microgpt.fut:355:70-200
                
                double neg_res_137392 = -zt_res_137391;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137393 = r_137387 + neg_res_137392;
                double r_tmp_141823 = zp_res_137393;
                
                r_137387 = r_tmp_141823;
            }
            defunc_0_lifted_lambda_res_137385 = r_137387;
            ((double *) mem_141001)[i_138994] = defunc_0_lifted_lambda_res_137385;
            ((double *) mem_141002)[i_138994] = defunc_0_lifted_lambda_res_137371;
            ((double *) mem_141003)[i_138994] = defunc_0_lifted_lambda_res_137352;
            ((double *) mem_141004)[i_138994] = defunc_0_lifted_lambda_res_137338;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140981, i_139007 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141001, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140982, i_139007 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141002, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140983, i_139007 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141003, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_140984, i_139007 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141004, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141045_cached_sizze_142211 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141045, &mem_141045_cached_sizze_142211, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141046_cached_sizze_142212 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141046, &mem_141046_cached_sizze_142212, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141057_cached_sizze_142213 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141057, &mem_141057_cached_sizze_142213, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141058_cached_sizze_142214 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141058, &mem_141058_cached_sizze_142214, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141067_cached_sizze_142215 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141067, &mem_141067_cached_sizze_142215, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141068_cached_sizze_142216 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141068, &mem_141068_cached_sizze_142216, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139030 = 0; i_139030 < (int64_t) 4; i_139030++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139023 = 0; i_139023 < (int64_t) 16; i_139023++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_rhs_137489 = ((double *) mem_140984)[i_139030 * (int64_t) 16 + i_139023];
            
            // futhark/microgpt.fut:340:93-121
            
            double zs_res_137490 = 1.0 / zs_rhs_137489;
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_137491 = ((double *) mem_140983)[i_139030 * (int64_t) 16 + i_139023];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_137510 = ((double *) mem_140981)[i_139030 * (int64_t) 16 + i_139023];
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_rhs_137508 = ((double *) mem_140982)[i_139030 * (int64_t) 16 + i_139023];
            
            // futhark/microgpt.fut:356:93-121
            
            double zs_res_137509 = 1.0 / zs_rhs_137508;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_139016 = 0; i_139016 < (int64_t) 16; i_139016++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_137538 = ((double *) mem_140928)[i_139030 * (int64_t) 256 + i_139023 * (int64_t) 16 + i_139016];
                
                // futhark/microgpt.fut:340:59-121
                
                double zt_res_137539 = zs_res_137490 * zt_lhs_137538;
                
                // futhark/microgpt.fut:340:88-148
                
                double zp_res_137540 = zp_rhs_137491 + zt_res_137539;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_137547 = ((double *) mem_140927)[i_139030 * (int64_t) 256 + i_139023 * (int64_t) 16 + i_139016];
                
                // futhark/microgpt.fut:356:59-121
                
                double zt_res_137548 = zs_res_137509 * zt_lhs_137547;
                
                // futhark/microgpt.fut:356:88-148
                
                double zp_res_137549 = zp_rhs_137510 + zt_res_137548;
                
                ((double *) mem_141067)[i_139016] = zp_res_137549;
                ((double *) mem_141068)[i_139016] = zp_res_137540;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141057, i_139023 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141067, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141058, i_139023 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141068, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141045, i_139030 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141057, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141046, i_139030 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141058, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141099_cached_sizze_142217 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141099, &mem_141099_cached_sizze_142217, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141100_cached_sizze_142218 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141100, &mem_141100_cached_sizze_142218, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141109_cached_sizze_142219 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141109, &mem_141109_cached_sizze_142219, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141110_cached_sizze_142220 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141110, &mem_141110_cached_sizze_142220, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139044 = 0; i_139044 < (int64_t) 4; i_139044++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139037 = 0; i_139037 < (int64_t) 16; i_139037++) {
            double f_elem_137569 = ((double *) mem_139934)[i_139044 * (int64_t) 16 + i_139037];
            double f_elem_137571 = ((double *) mem_139931)[i_139044 * (int64_t) 16 + i_139037];
            
            // futhark/microgpt.fut:341:119-145
            
            double neg_res_137576 = -f_elem_137569;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137577;
            double r_137579 = 0.0;
            
            for (int64_t i_137578 = 0; i_137578 < (int64_t) 16; i_137578++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_137580 = ((double *) mem_139822)[i_139044 * (int64_t) 256 + i_139037 * (int64_t) 16 + i_137578];
                
                // futhark/microgpt.fut:341:85-145
                
                double zp_res_137581 = neg_res_137576 + zp_lhs_137580;
                
                // futhark/microgpt.fut:341:78-145
                
                double exp_res_137582 = futrts_exp64(zp_res_137581);
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_137583 = ((double *) mem_141046)[i_139044 * (int64_t) 256 + i_139037 * (int64_t) 16 + i_137578];
                
                // futhark/microgpt.fut:341:78-181
                
                double zt_res_137584 = exp_res_137582 * zt_rhs_137583;
                
                // futhark/microgpt.fut:341:70-181
                
                double neg_res_137585 = -zt_res_137584;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137586 = r_137579 + neg_res_137585;
                double r_tmp_141834 = zp_res_137586;
                
                r_137579 = r_tmp_141834;
            }
            defunc_0_lifted_lambda_res_137577 = r_137579;
            // futhark/microgpt.fut:357:119-145
            
            double neg_res_137594 = -f_elem_137571;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137595;
            double r_137597 = 0.0;
            
            for (int64_t i_137596 = 0; i_137596 < (int64_t) 16; i_137596++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_137598 = ((double *) mem_139821)[i_139044 * (int64_t) 256 + i_139037 * (int64_t) 16 + i_137596];
                
                // futhark/microgpt.fut:357:85-145
                
                double zp_res_137599 = neg_res_137594 + zp_lhs_137598;
                
                // futhark/microgpt.fut:357:78-145
                
                double exp_res_137600 = futrts_exp64(zp_res_137599);
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_137601 = ((double *) mem_141045)[i_139044 * (int64_t) 256 + i_139037 * (int64_t) 16 + i_137596];
                
                // futhark/microgpt.fut:357:78-181
                
                double zt_res_137602 = exp_res_137600 * zt_rhs_137601;
                
                // futhark/microgpt.fut:357:70-181
                
                double neg_res_137603 = -zt_res_137602;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137604 = r_137597 + neg_res_137603;
                double r_tmp_141835 = zp_res_137604;
                
                r_137597 = r_tmp_141835;
            }
            defunc_0_lifted_lambda_res_137595 = r_137597;
            ((double *) mem_141109)[i_139037] = defunc_0_lifted_lambda_res_137595;
            ((double *) mem_141110)[i_139037] = defunc_0_lifted_lambda_res_137577;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141099, i_139044 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141109, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141100, i_139044 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141110, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141131_cached_sizze_142221 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141131, &mem_141131_cached_sizze_142221, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141132_cached_sizze_142222 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141132, &mem_141132_cached_sizze_142222, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141143_cached_sizze_142223 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141143, &mem_141143_cached_sizze_142223, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141144_cached_sizze_142224 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141144, &mem_141144_cached_sizze_142224, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141153_cached_sizze_142225 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141153, &mem_141153_cached_sizze_142225, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141154_cached_sizze_142226 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141154, &mem_141154_cached_sizze_142226, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139065 = 0; i_139065 < (int64_t) 4; i_139065++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139058 = 0; i_139058 < (int64_t) 16; i_139058++) {
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_137624 = ((double *) mem_139934)[i_139065 * (int64_t) 16 + i_139058];
            
            // futhark/microgpt.fut:344:101-127
            
            double neg_res_137625 = -neg_arg0_137624;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_137626 = ((double *) mem_141100)[i_139065 * (int64_t) 16 + i_139058];
            
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_137627 = ((double *) mem_139933)[i_139065 * (int64_t) 16 + i_139058];
            
            // futhark/microgpt.fut:344:266-292
            
            double neg_res_137628 = -neg_arg0_137627;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_137629 = ((double *) mem_139932)[i_139065 * (int64_t) 16 + i_139058];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_137662 = ((double *) mem_139929)[i_139065 * (int64_t) 16 + i_139058];
            
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_137660 = ((double *) mem_139930)[i_139065 * (int64_t) 16 + i_139058];
            
            // futhark/microgpt.fut:360:266-292
            
            double neg_res_137661 = -neg_arg0_137660;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_137659 = ((double *) mem_141099)[i_139065 * (int64_t) 16 + i_139058];
            
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_137657 = ((double *) mem_139931)[i_139065 * (int64_t) 16 + i_139058];
            
            // futhark/microgpt.fut:360:101-127
            
            double neg_res_137658 = -neg_arg0_137657;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_139051 = 0; i_139051 < (int64_t) 16; i_139051++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_137701 = ((double *) mem_139822)[i_139065 * (int64_t) 256 + i_139058 * (int64_t) 16 + i_139051];
                
                // futhark/microgpt.fut:344:67-127
                
                double zp_res_137702 = neg_res_137625 + zp_lhs_137701;
                
                // futhark/microgpt.fut:344:60-127
                
                double exp_res_137703 = futrts_exp64(zp_res_137702);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_137704 = ((double *) mem_141046)[i_139065 * (int64_t) 256 + i_139058 * (int64_t) 16 + i_139051];
                
                // futhark/microgpt.fut:344:60-163
                
                double zt_res_137705 = exp_res_137703 * zt_rhs_137704;
                
                // futhark/microgpt.fut:344:232-292
                
                double zp_res_137706 = neg_res_137628 + zp_lhs_137701;
                
                // futhark/microgpt.fut:344:225-292
                
                double neg_res_137707 = -zp_res_137706;
                
                // futhark/microgpt.fut:100:42-54
                
                double max_res_137708 = fmax64(0.0, neg_res_137707);
                
                // futhark/microgpt.fut:100:35-54
                
                double sgn_res_137709 = fsignum64(max_res_137708);
                
                // futhark/microgpt.fut:344:206-295
                
                double neg_res_137710 = -sgn_res_137709;
                
                // futhark/microgpt.fut:344:197-296
                
                double zp_res_137711 = 1.0 + neg_res_137710;
                
                // futhark/microgpt.fut:344:171-296
                
                double zt_res_137712 = zt_lhs_137626 * zp_res_137711;
                
                // futhark/microgpt.fut:344:192-324
                
                double zt_res_137713 = zt_rhs_137629 * zt_res_137712;
                
                // futhark/microgpt.fut:344:131-324
                
                double zp_res_137714 = zt_res_137705 + zt_res_137713;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_137721 = ((double *) mem_139821)[i_139065 * (int64_t) 256 + i_139058 * (int64_t) 16 + i_139051];
                
                // futhark/microgpt.fut:360:67-127
                
                double zp_res_137722 = neg_res_137658 + zp_lhs_137721;
                
                // futhark/microgpt.fut:360:60-127
                
                double exp_res_137723 = futrts_exp64(zp_res_137722);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_137724 = ((double *) mem_141045)[i_139065 * (int64_t) 256 + i_139058 * (int64_t) 16 + i_139051];
                
                // futhark/microgpt.fut:360:60-163
                
                double zt_res_137725 = exp_res_137723 * zt_rhs_137724;
                
                // futhark/microgpt.fut:360:232-292
                
                double zp_res_137726 = neg_res_137661 + zp_lhs_137721;
                
                // futhark/microgpt.fut:360:225-292
                
                double neg_res_137727 = -zp_res_137726;
                
                // futhark/microgpt.fut:100:42-54
                
                double max_res_137728 = fmax64(0.0, neg_res_137727);
                
                // futhark/microgpt.fut:100:35-54
                
                double sgn_res_137729 = fsignum64(max_res_137728);
                
                // futhark/microgpt.fut:360:206-295
                
                double neg_res_137730 = -sgn_res_137729;
                
                // futhark/microgpt.fut:360:197-296
                
                double zp_res_137731 = 1.0 + neg_res_137730;
                
                // futhark/microgpt.fut:360:171-296
                
                double zt_res_137732 = zt_lhs_137659 * zp_res_137731;
                
                // futhark/microgpt.fut:360:192-324
                
                double zt_res_137733 = zt_rhs_137662 * zt_res_137732;
                
                // futhark/microgpt.fut:360:131-324
                
                double zp_res_137734 = zt_res_137725 + zt_res_137733;
                
                ((double *) mem_141153)[i_139051] = zp_res_137734;
                ((double *) mem_141154)[i_139051] = zp_res_137714;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141143, i_139058 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141153, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141144, i_139058 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141154, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141131, i_139065 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141143, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141132, i_139065 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141144, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141185_cached_sizze_142227 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141185, &mem_141185_cached_sizze_142227, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141186_cached_sizze_142228 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141186, &mem_141186_cached_sizze_142228, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141197_cached_sizze_142229 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141197, &mem_141197_cached_sizze_142229, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141198_cached_sizze_142230 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141198, &mem_141198_cached_sizze_142230, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141207_cached_sizze_142231 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141207, &mem_141207_cached_sizze_142231, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141208_cached_sizze_142232 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141208, &mem_141208_cached_sizze_142232, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139086 = 0; i_139086 < (int64_t) 4; i_139086++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139079 = 0; i_139079 < (int64_t) 16; i_139079++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_139072 = 0; i_139072 < (int64_t) 16; i_139072++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_137799 = ((double *) mem_141132)[i_139086 * (int64_t) 256 + i_139079 * (int64_t) 16 + i_139072];
                
                // futhark/microgpt.fut:345:58-100
                
                double zs_res_137800 = zs_lhs_137799 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_137807 = ((double *) mem_141131)[i_139086 * (int64_t) 256 + i_139079 * (int64_t) 16 + i_139072];
                
                // futhark/microgpt.fut:361:58-100
                
                double zs_res_137808 = zs_lhs_137807 / 2.0;
                
                ((double *) mem_141207)[i_139072] = zs_res_137808;
                ((double *) mem_141208)[i_139072] = zs_res_137800;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141197, i_139079 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141207, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141198, i_139079 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141208, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141185, i_139086 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141197, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141186, i_139086 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141198, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_141239, (int64_t) 2048, "mem_141239")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141240_cached_sizze_142233 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141240, &mem_141240_cached_sizze_142233, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141241_cached_sizze_142234 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141241, &mem_141241_cached_sizze_142234, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141242_cached_sizze_142235 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141242, &mem_141242_cached_sizze_142235, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141259_cached_sizze_142236 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141259, &mem_141259_cached_sizze_142236, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141260_cached_sizze_142237 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141260, &mem_141260_cached_sizze_142237, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141261_cached_sizze_142238 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141261, &mem_141261_cached_sizze_142238, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141262_cached_sizze_142239 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141262, &mem_141262_cached_sizze_142239, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139110 = 0; i_139110 < (int64_t) 16; i_139110++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139097 = 0; i_139097 < (int64_t) 16; i_139097++) {
            // futhark/microgpt.fut:330:40-43
            
            int64_t zt_lhs_137056 = sdiv64(i_139097, (int64_t) 4);
            
            // futhark/microgpt.fut:330:27-45
            
            bool x_137057 = sle64((int64_t) 0, zt_lhs_137056);
            
            // futhark/microgpt.fut:330:27-45
            
            bool y_137058 = slt64(zt_lhs_137056, (int64_t) 4);
            
            // futhark/microgpt.fut:330:27-45
            
            bool bounds_check_137059 = x_137057 && y_137058;
            
            // futhark/microgpt.fut:330:27-45
            
            bool index_certs_137060;
            
            if (!bounds_check_137059) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_137056, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:330:27-45\n   #1  futhark/microgpt.fut:61:46-49\n   #2  futhark/microgpt.fut:330:5-108\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:322:13-330:118\n   #8  futhark/microgpt.fut:459:5-75\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:330:62-65
            
            int64_t zt_lhs_137061 = smod64(i_139097, (int64_t) 4);
            
            // futhark/microgpt.fut:330:27-67
            
            bool x_137062 = sle64((int64_t) 0, zt_lhs_137061);
            
            // futhark/microgpt.fut:330:27-67
            
            bool y_137063 = slt64(zt_lhs_137061, (int64_t) 4);
            
            // futhark/microgpt.fut:330:27-67
            
            bool bounds_check_137064 = x_137062 && y_137063;
            
            // futhark/microgpt.fut:330:27-67
            
            bool index_certs_137065;
            
            if (!bounds_check_137064) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_137061, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:330:27-67\n   #1  futhark/microgpt.fut:61:46-49\n   #2  futhark/microgpt.fut:330:5-108\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:322:13-330:118\n   #8  futhark/microgpt.fut:459:5-75\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137066;
            double r_137068 = 0.0;
            
            for (int64_t i_137067 = 0; i_137067 < (int64_t) 16; i_137067++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_137069 = ((double *) mem_140800)[zt_lhs_137056 * (int64_t) 64 + i_137067 * (int64_t) 4 + zt_lhs_137061];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_137070 = ((double *) mem_140251)[zt_lhs_137056 * (int64_t) 256 + i_137067 * (int64_t) 16 + i_139110];
                
                // futhark/microgpt.fut:330:27-106
                
                double zt_res_137071 = zt_lhs_137069 * zt_rhs_137070;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137072 = r_137068 + zt_res_137071;
                double r_tmp_141856 = zp_res_137072;
                
                r_137068 = r_tmp_141856;
            }
            defunc_0_lifted_lambda_res_137066 = r_137068;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137085;
            double r_137087 = 0.0;
            
            for (int64_t i_137086 = 0; i_137086 < (int64_t) 16; i_137086++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_137088 = ((double *) mem_141186)[zt_lhs_137056 * (int64_t) 256 + i_137086 * (int64_t) 16 + i_139110];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_137089 = ((double *) mem_139634)[zt_lhs_137056 * (int64_t) 64 + i_137086 * (int64_t) 4 + zt_lhs_137061];
                
                // futhark/microgpt.fut:346:27-105
                
                double zt_res_137090 = zt_lhs_137088 * zt_rhs_137089;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137091 = r_137087 + zt_res_137090;
                double r_tmp_141857 = zp_res_137091;
                
                r_137087 = r_tmp_141857;
            }
            defunc_0_lifted_lambda_res_137085 = r_137087;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137107;
            double r_137109 = 0.0;
            
            for (int64_t i_137108 = 0; i_137108 < (int64_t) 16; i_137108++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_137110 = ((double *) mem_141185)[zt_lhs_137056 * (int64_t) 256 + i_139110 * (int64_t) 16 + i_137108];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_137111 = ((double *) mem_139633)[zt_lhs_137056 * (int64_t) 64 + i_137108 * (int64_t) 4 + zt_lhs_137061];
                
                // futhark/microgpt.fut:362:27-105
                
                double zt_res_137112 = zt_lhs_137110 * zt_rhs_137111;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137113 = r_137109 + zt_res_137112;
                double r_tmp_141858 = zp_res_137113;
                
                r_137109 = r_tmp_141858;
            }
            defunc_0_lifted_lambda_res_137107 = r_137109;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137125;
            double r_137127 = 0.0;
            
            for (int64_t i_137126 = 0; i_137126 < (int64_t) 16; i_137126++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_137128 = ((double *) mem_140755)[i_137126 * (int64_t) 16 + i_139110];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_137129 = ((double *) mem_140332)[i_137126 * (int64_t) 16 + i_139097];
                
                // futhark/microgpt.fut:394:68-112
                
                double zt_res_137130 = zt_lhs_137128 * zt_rhs_137129;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137131 = r_137127 + zt_res_137130;
                double r_tmp_141859 = zp_res_137131;
                
                r_137127 = r_tmp_141859;
            }
            defunc_0_lifted_lambda_res_137125 = r_137127;
            ((double *) mem_141259)[i_139097] = defunc_0_lifted_lambda_res_137125;
            ((double *) mem_141260)[i_139097] = defunc_0_lifted_lambda_res_137107;
            ((double *) mem_141261)[i_139097] = defunc_0_lifted_lambda_res_137085;
            ((double *) mem_141262)[i_139097] = defunc_0_lifted_lambda_res_137066;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141239.mem, i_139110 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141259, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141240, i_139110 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141260, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141241, i_139110 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141261, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141242, i_139110 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141262, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141303_cached_sizze_142240 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141303, &mem_141303_cached_sizze_142240, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141308_cached_sizze_142241 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141308, &mem_141308_cached_sizze_142241, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139121 = 0; i_139121 < (int64_t) 16; i_139121++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139117 = 0; i_139117 < (int64_t) 16; i_139117++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126197;
            double r_126199 = 0.0;
            
            for (int64_t i_126198 = 0; i_126198 < (int64_t) 16; i_126198++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_126200 = ((double *) mem_141242)[i_139121 * (int64_t) 16 + i_126198];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_126201 = ((double *) wval_mem_139498.mem)[i_126198 * (int64_t) 16 + i_139117];
                
                // futhark/microgpt.fut:365:73-118
                
                double zt_res_126202 = zt_lhs_126200 * zt_rhs_126201;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126203 = r_126199 + zt_res_126202;
                double r_tmp_141862 = zp_res_126203;
                
                r_126199 = r_tmp_141862;
            }
            defunc_0_lifted_lambda_res_126197 = r_126199;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126204;
            double r_126206 = 0.0;
            
            for (int64_t i_126205 = 0; i_126205 < (int64_t) 16; i_126205++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_126207 = ((double *) mem_141241)[i_139121 * (int64_t) 16 + i_126205];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_126208 = ((double *) wkey_mem_139492.mem)[i_126205 * (int64_t) 16 + i_139117];
                
                // futhark/microgpt.fut:365:149-194
                
                double zt_res_126209 = zt_lhs_126207 * zt_rhs_126208;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126210 = r_126206 + zt_res_126209;
                double r_tmp_141863 = zp_res_126210;
                
                r_126206 = r_tmp_141863;
            }
            defunc_0_lifted_lambda_res_126204 = r_126206;
            // futhark/microgpt.fut:365:51-196
            
            double zp_res_126211 = defunc_0_lifted_lambda_res_126197 + defunc_0_lifted_lambda_res_126204;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126212;
            double r_126214 = 0.0;
            
            for (int64_t i_126213 = 0; i_126213 < (int64_t) 16; i_126213++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_126215 = ((double *) mem_141240)[i_139121 * (int64_t) 16 + i_126213];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_126216 = ((double *) wqry_mem_139495.mem)[i_126213 * (int64_t) 16 + i_139117];
                
                // futhark/microgpt.fut:365:226-271
                
                double zt_res_126217 = zt_lhs_126215 * zt_rhs_126216;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126218 = r_126214 + zt_res_126217;
                double r_tmp_141864 = zp_res_126218;
                
                r_126214 = r_tmp_141864;
            }
            defunc_0_lifted_lambda_res_126212 = r_126214;
            // futhark/microgpt.fut:365:122-273
            
            double zp_res_126219 = zp_res_126211 + defunc_0_lifted_lambda_res_126212;
            
            ((double *) mem_141308)[i_139117] = zp_res_126219;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141303, i_139121 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141308, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_141319, (int64_t) 2048, "mem_141319")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_141320, (int64_t) 2048, "mem_141320")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_141321, (int64_t) 2048, "mem_141321")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141322_cached_sizze_142242 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141322, &mem_141322_cached_sizze_142242, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141323_cached_sizze_142243 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141323, &mem_141323_cached_sizze_142243, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141342_cached_sizze_142244 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141342, &mem_141342_cached_sizze_142244, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141343_cached_sizze_142245 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141343, &mem_141343_cached_sizze_142245, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141344_cached_sizze_142246 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141344, &mem_141344_cached_sizze_142246, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139143 = 0; i_139143 < (int64_t) 16; i_139143++) {
        // futhark/microgpt.fut:364:47-59
        
        double zp_lhs_132661 = ((double *) mem_139579)[i_139143];
        
        // futhark/microgpt.fut:364:47-87
        
        double zp_res_132662 = 1.0e-5 + zp_lhs_132661;
        
        // futhark/microgpt.fut:364:39-87
        
        double sqrt_res_132663 = futrts_sqrt64(zp_res_132662);
        
        // futhark/microgpt.fut:366:128-157
        
        double zt_res_132671 = sqrt_res_132663 * sqrt_res_132663;
        
        // futhark/microgpt.fut:366:119-157
        
        double zs_res_132672 = 1.0 / zt_res_132671;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_132673;
        double r_132675 = 0.0;
        
        for (int64_t i_132674 = 0; i_132674 < (int64_t) 16; i_132674++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_132676 = ((double *) mem_141303)[i_139143 * (int64_t) 16 + i_132674];
            
            // futhark/microgpt.fut:61:46-49
            
            double zt_rhs_132677 = ((double *) mem_139563)[i_139143 * (int64_t) 16 + i_132674];
            
            // futhark/microgpt.fut:366:69-112
            
            double zt_res_132678 = zt_lhs_132676 * zt_rhs_132677;
            
            // futhark/microgpt.fut:366:90-157
            
            double zt_res_132679 = zs_res_132672 * zt_res_132678;
            
            // futhark/microgpt.fut:366:61-157
            
            double neg_res_132680 = -zt_res_132679;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_132681 = r_132675 + neg_res_132680;
            double r_tmp_141870 = zp_res_132681;
            
            r_132675 = r_tmp_141870;
        }
        defunc_0_lifted_lambda_res_132673 = r_132675;
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139129 = 0; i_139129 < (int64_t) 16; i_139129++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137875;
            double r_137877 = 0.0;
            
            for (int64_t i_137876 = 0; i_137876 < (int64_t) 16; i_137876++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_137878 = ((double *) mem_141240)[i_137876 * (int64_t) 16 + i_139143];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_137879 = ((double *) mem_139616)[i_137876 * (int64_t) 16 + i_139129];
                
                // futhark/microgpt.fut:391:68-111
                
                double zt_res_137880 = zt_lhs_137878 * zt_rhs_137879;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137881 = r_137877 + zt_res_137880;
                double r_tmp_141874 = zp_res_137881;
                
                r_137877 = r_tmp_141874;
            }
            defunc_0_lifted_lambda_res_137875 = r_137877;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137888;
            double r_137890 = 0.0;
            
            for (int64_t i_137889 = 0; i_137889 < (int64_t) 16; i_137889++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_137891 = ((double *) mem_141241)[i_137889 * (int64_t) 16 + i_139143];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_137892 = ((double *) mem_139616)[i_137889 * (int64_t) 16 + i_139129];
                
                // futhark/microgpt.fut:392:68-111
                
                double zt_res_137893 = zt_lhs_137891 * zt_rhs_137892;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137894 = r_137890 + zt_res_137893;
                double r_tmp_141875 = zp_res_137894;
                
                r_137890 = r_tmp_141875;
            }
            defunc_0_lifted_lambda_res_137888 = r_137890;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137904;
            double r_137906 = 0.0;
            
            for (int64_t i_137905 = 0; i_137905 < (int64_t) 16; i_137905++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_137907 = ((double *) mem_141242)[i_137905 * (int64_t) 16 + i_139143];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_137908 = ((double *) mem_139616)[i_137905 * (int64_t) 16 + i_139129];
                
                // futhark/microgpt.fut:393:68-111
                
                double zt_res_137909 = zt_lhs_137907 * zt_rhs_137908;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137910 = r_137906 + zt_res_137909;
                double r_tmp_141876 = zp_res_137910;
                
                r_137906 = r_tmp_141876;
            }
            defunc_0_lifted_lambda_res_137904 = r_137906;
            ((double *) mem_141342)[i_139129] = defunc_0_lifted_lambda_res_137904;
            ((double *) mem_141343)[i_139129] = defunc_0_lifted_lambda_res_137888;
            ((double *) mem_141344)[i_139129] = defunc_0_lifted_lambda_res_137875;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141319.mem, i_139143 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141342, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141320.mem, i_139143 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141343, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141321.mem, i_139143 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141344, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        ((double *) mem_141322)[i_139143] = defunc_0_lifted_lambda_res_132673;
        ((double *) mem_141323)[i_139143] = sqrt_res_132663;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141381_cached_sizze_142247 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141381, &mem_141381_cached_sizze_142247, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139151 = 0; i_139151 < (int64_t) 16; i_139151++) {
        // futhark/microgpt.fut:367:39-51
        
        double zt_lhs_126247 = ((double *) mem_141322)[i_139151];
        
        // futhark/microgpt.fut:367:93-105
        
        double zp_lhs_126248 = ((double *) mem_139579)[i_139151];
        
        // futhark/microgpt.fut:367:93-133
        
        double zp_res_126249 = 1.0e-5 + zp_lhs_126248;
        
        // futhark/microgpt.fut:367:85-133
        
        double sqrt_res_126250 = futrts_sqrt64(zp_res_126249);
        
        // futhark/microgpt.fut:367:71-135
        
        double zt_res_126251 = 2.0 * sqrt_res_126250;
        
        // futhark/microgpt.fut:367:57-135
        
        double zs_res_126252 = 1.0 / zt_res_126251;
        
        // futhark/microgpt.fut:367:39-135
        
        double zt_res_126253 = zt_lhs_126247 * zs_res_126252;
        
        ((double *) mem_141381)[i_139151] = zt_res_126253;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141388_cached_sizze_142248 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141388, &mem_141388_cached_sizze_142248, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141393_cached_sizze_142249 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141393, &mem_141393_cached_sizze_142249, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139159 = 0; i_139159 < (int64_t) 16; i_139159++) {
        // futhark/microgpt.fut:368:98-110
        
        double zs_rhs_126261 = ((double *) mem_141323)[i_139159];
        
        // futhark/microgpt.fut:368:90-110
        
        double zs_res_126262 = 1.0 / zs_rhs_126261;
        
        // futhark/microgpt.fut:368:120-132
        
        double zs_lhs_126263 = ((double *) mem_141381)[i_139159];
        
        // futhark/microgpt.fut:368:120-147
        
        double zs_res_126264 = zs_lhs_126263 / 16.0;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139155 = 0; i_139155 < (int64_t) 16; i_139155++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_126271 = ((double *) mem_140755)[i_139159 * (int64_t) 16 + i_139155];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_126272 = ((double *) mem_141303)[i_139159 * (int64_t) 16 + i_139155];
            
            // futhark/microgpt.fut:368:64-110
            
            double zt_res_126273 = zs_res_126262 * zt_lhs_126272;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_126274 = ((double *) mem_139563)[i_139159 * (int64_t) 16 + i_139155];
            
            // futhark/microgpt.fut:368:133-171
            
            double zt_res_126275 = zs_res_126264 * zt_rhs_126274;
            
            // futhark/microgpt.fut:368:149-230
            
            double zp_res_126276 = zt_res_126275 + zt_res_126275;
            
            // futhark/microgpt.fut:368:85-230
            
            double zp_res_126277 = zt_res_126273 + zp_res_126276;
            
            // futhark/microgpt.fut:368:37-230
            
            double zp_res_126278 = zp_lhs_126271 + zp_res_126277;
            
            ((double *) mem_141393)[i_139155] = zp_res_126278;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141388, i_139159 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141393, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141404_cached_sizze_142250 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141404, &mem_141404_cached_sizze_142250, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141405_cached_sizze_142251 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141405, &mem_141405_cached_sizze_142251, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141414_cached_sizze_142252 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141414, &mem_141414_cached_sizze_142252, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141415_cached_sizze_142253 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141415, &mem_141415_cached_sizze_142253, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139172 = 0; i_139172 < (int64_t) 16; i_139172++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139165 = 0; i_139165 < (int64_t) 16; i_139165++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_137934 = ((double *) mem_141388)[i_139172 * (int64_t) 16 + i_139165];
            
            ((double *) mem_141414)[i_139165] = lifted_lambda_res_137934;
            ((double *) mem_141415)[i_139165] = lifted_lambda_res_137934;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141404, i_139172 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141414, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141405, i_139172 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141415, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141436_cached_sizze_142254 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141436, &mem_141436_cached_sizze_142254, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141437_cached_sizze_142255 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141437, &mem_141437_cached_sizze_142255, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141438_cached_sizze_142256 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141438, &mem_141438_cached_sizze_142256, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141439_cached_sizze_142257 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141439, &mem_141439_cached_sizze_142257, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139183 = 0; i_139183 < (int64_t) 16; i_139183++) {
        // futhark/microgpt.fut:386:47-59
        
        double zp_lhs_132786 = ((double *) mem_139520)[i_139183];
        
        // futhark/microgpt.fut:386:47-87
        
        double zp_res_132787 = 1.0e-5 + zp_lhs_132786;
        
        // futhark/microgpt.fut:386:39-87
        
        double sqrt_res_132788 = futrts_sqrt64(zp_res_132787);
        
        // futhark/microgpt.fut:388:156-185
        
        double zt_res_132796 = sqrt_res_132788 * sqrt_res_132788;
        
        // futhark/microgpt.fut:388:147-185
        
        double zs_res_132797 = 1.0 / zt_res_132796;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_132798;
        double r_132800 = 0.0;
        
        for (int64_t i_132799 = 0; i_132799 < (int64_t) 16; i_132799++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_132801 = ((double *) mem_141405)[i_139183 * (int64_t) 16 + i_132799];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_132802 = ((double *) wpe_mem_139494.mem)[i_139183 * (int64_t) 16 + i_132799];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_132803 = ((double *) mem_139503)[i_139183 * (int64_t) 16 + i_132799];
            
            // futhark/microgpt.fut:388:95-139
            
            double zp_res_132804 = zp_lhs_132802 + zp_rhs_132803;
            
            // futhark/microgpt.fut:388:69-139
            
            double zt_res_132805 = zt_lhs_132801 * zp_res_132804;
            
            // futhark/microgpt.fut:388:90-185
            
            double zt_res_132806 = zs_res_132797 * zt_res_132805;
            
            // futhark/microgpt.fut:388:61-185
            
            double neg_res_132807 = -zt_res_132806;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_132808 = r_132800 + neg_res_132807;
            double r_tmp_141888 = zp_res_132808;
            
            r_132800 = r_tmp_141888;
        }
        defunc_0_lifted_lambda_res_132798 = r_132800;
        // futhark/microgpt.fut:399:47-59
        
        double zp_lhs_132819 = ((double *) mem_139519)[i_139183];
        
        // futhark/microgpt.fut:399:47-87
        
        double zp_res_132820 = 1.0e-5 + zp_lhs_132819;
        
        // futhark/microgpt.fut:399:39-87
        
        double sqrt_res_132821 = futrts_sqrt64(zp_res_132820);
        
        // futhark/microgpt.fut:401:156-185
        
        double zt_res_132829 = sqrt_res_132821 * sqrt_res_132821;
        
        // futhark/microgpt.fut:401:147-185
        
        double zs_res_132830 = 1.0 / zt_res_132829;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_132831;
        double r_132833 = 0.0;
        
        for (int64_t i_132832 = 0; i_132832 < (int64_t) 16; i_132832++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_132834 = ((double *) mem_141404)[i_139183 * (int64_t) 16 + i_132832];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_132835 = ((double *) wpe_mem_139494.mem)[i_139183 * (int64_t) 16 + i_132832];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_132836 = ((double *) mem_139503)[i_139183 * (int64_t) 16 + i_132832];
            
            // futhark/microgpt.fut:401:95-139
            
            double zp_res_132837 = zp_lhs_132835 + zp_rhs_132836;
            
            // futhark/microgpt.fut:401:69-139
            
            double zt_res_132838 = zt_lhs_132834 * zp_res_132837;
            
            // futhark/microgpt.fut:401:90-185
            
            double zt_res_132839 = zs_res_132830 * zt_res_132838;
            
            // futhark/microgpt.fut:401:61-185
            
            double neg_res_132840 = -zt_res_132839;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_132841 = r_132833 + neg_res_132840;
            double r_tmp_141889 = zp_res_132841;
            
            r_132833 = r_tmp_141889;
        }
        defunc_0_lifted_lambda_res_132831 = r_132833;
        ((double *) mem_141436)[i_139183] = defunc_0_lifted_lambda_res_132831;
        ((double *) mem_141437)[i_139183] = sqrt_res_132821;
        ((double *) mem_141438)[i_139183] = defunc_0_lifted_lambda_res_132798;
        ((double *) mem_141439)[i_139183] = sqrt_res_132788;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141464_cached_sizze_142258 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141464, &mem_141464_cached_sizze_142258, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141465_cached_sizze_142259 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141465, &mem_141465_cached_sizze_142259, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139192 = 0; i_139192 < (int64_t) 16; i_139192++) {
        // futhark/microgpt.fut:389:39-51
        
        double zt_lhs_132902 = ((double *) mem_141438)[i_139192];
        
        // futhark/microgpt.fut:389:93-105
        
        double zp_lhs_132903 = ((double *) mem_139520)[i_139192];
        
        // futhark/microgpt.fut:389:93-133
        
        double zp_res_132904 = 1.0e-5 + zp_lhs_132903;
        
        // futhark/microgpt.fut:389:85-133
        
        double sqrt_res_132905 = futrts_sqrt64(zp_res_132904);
        
        // futhark/microgpt.fut:389:71-135
        
        double zt_res_132906 = 2.0 * sqrt_res_132905;
        
        // futhark/microgpt.fut:389:57-135
        
        double zs_res_132907 = 1.0 / zt_res_132906;
        
        // futhark/microgpt.fut:389:39-135
        
        double zt_res_132908 = zt_lhs_132902 * zs_res_132907;
        
        // futhark/microgpt.fut:402:39-51
        
        double zt_lhs_132915 = ((double *) mem_141436)[i_139192];
        
        // futhark/microgpt.fut:402:93-105
        
        double zp_lhs_132916 = ((double *) mem_139519)[i_139192];
        
        // futhark/microgpt.fut:402:93-133
        
        double zp_res_132917 = 1.0e-5 + zp_lhs_132916;
        
        // futhark/microgpt.fut:402:85-133
        
        double sqrt_res_132918 = futrts_sqrt64(zp_res_132917);
        
        // futhark/microgpt.fut:402:71-135
        
        double zt_res_132919 = 2.0 * sqrt_res_132918;
        
        // futhark/microgpt.fut:402:57-135
        
        double zs_res_132920 = 1.0 / zt_res_132919;
        
        // futhark/microgpt.fut:402:39-135
        
        double zt_res_132921 = zt_lhs_132915 * zs_res_132920;
        
        ((double *) mem_141464)[i_139192] = zt_res_132921;
        ((double *) mem_141465)[i_139192] = zt_res_132908;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141478_cached_sizze_142260 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141478, &mem_141478_cached_sizze_142260, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_141479, (int64_t) 2048, "mem_141479")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141488_cached_sizze_142261 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141488, &mem_141488_cached_sizze_142261, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141489_cached_sizze_142262 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141489, &mem_141489_cached_sizze_142262, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139206 = 0; i_139206 < (int64_t) 16; i_139206++) {
        // futhark/microgpt.fut:390:72-84
        
        double zs_rhs_132939 = ((double *) mem_141439)[i_139206];
        
        // futhark/microgpt.fut:390:64-84
        
        double zs_res_132940 = 1.0 / zs_rhs_132939;
        
        // futhark/microgpt.fut:390:94-106
        
        double zs_lhs_132941 = ((double *) mem_141465)[i_139206];
        
        // futhark/microgpt.fut:390:94-121
        
        double zs_res_132942 = zs_lhs_132941 / 16.0;
        
        // futhark/microgpt.fut:403:94-106
        
        double zs_lhs_132966 = ((double *) mem_141464)[i_139206];
        
        // futhark/microgpt.fut:403:94-121
        
        double zs_res_132967 = zs_lhs_132966 / 16.0;
        
        // futhark/microgpt.fut:403:72-84
        
        double zs_rhs_132964 = ((double *) mem_141437)[i_139206];
        
        // futhark/microgpt.fut:403:64-84
        
        double zs_res_132965 = 1.0 / zs_rhs_132964;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139199 = 0; i_139199 < (int64_t) 16; i_139199++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_137961 = ((double *) mem_141405)[i_139206 * (int64_t) 16 + i_139199];
            
            // futhark/microgpt.fut:390:38-84
            
            double zt_res_137962 = zs_res_132940 * zt_lhs_137961;
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_137963 = ((double *) wpe_mem_139494.mem)[i_139206 * (int64_t) 16 + i_139199];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_137964 = ((double *) mem_139503)[i_139206 * (int64_t) 16 + i_139199];
            
            // futhark/microgpt.fut:390:128-172
            
            double zp_res_137965 = zp_lhs_137963 + zp_rhs_137964;
            
            // futhark/microgpt.fut:390:107-172
            
            double zt_res_137966 = zs_res_132942 * zp_res_137965;
            
            // futhark/microgpt.fut:390:123-259
            
            double zp_res_137967 = zt_res_137966 + zt_res_137966;
            
            // futhark/microgpt.fut:390:59-259
            
            double zp_res_137968 = zt_res_137962 + zp_res_137967;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_137975 = ((double *) mem_141404)[i_139206 * (int64_t) 16 + i_139199];
            
            // futhark/microgpt.fut:403:38-84
            
            double zt_res_137976 = zs_res_132965 * zt_lhs_137975;
            
            // futhark/microgpt.fut:403:107-172
            
            double zt_res_137980 = zs_res_132967 * zp_res_137965;
            
            // futhark/microgpt.fut:403:123-259
            
            double zp_res_137981 = zt_res_137980 + zt_res_137980;
            
            // futhark/microgpt.fut:403:59-259
            
            double zp_res_137982 = zt_res_137976 + zp_res_137981;
            
            ((double *) mem_141488)[i_139199] = zp_res_137982;
            ((double *) mem_141489)[i_139199] = zp_res_137968;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141478, i_139206 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141488, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141479.mem, i_139206 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141489, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_141510, (int64_t) 8192, "mem_141510")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141515_cached_sizze_142263 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141515, &mem_141515_cached_sizze_142263, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139215 = 0; i_139215 < (int64_t) 64; i_139215++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139211 = 0; i_139211 < (int64_t) 16; i_139211++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126479;
            double r_126481 = 0.0;
            
            for (int64_t i_126480 = 0; i_126480 < (int64_t) 16; i_126480++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_126482 = ((double *) mem_140687)[i_126480 * (int64_t) 64 + i_139215];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_126483 = ((double *) mem_140401)[i_126480 * (int64_t) 16 + i_139211];
                
                // futhark/microgpt.fut:395:67-111
                
                double zt_res_126484 = zt_lhs_126482 * zt_rhs_126483;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126485 = r_126481 + zt_res_126484;
                double r_tmp_141898 = zp_res_126485;
                
                r_126481 = r_tmp_141898;
            }
            defunc_0_lifted_lambda_res_126479 = r_126481;
            ((double *) mem_141515)[i_139211] = defunc_0_lifted_lambda_res_126479;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141510.mem, i_139215 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141515, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_141526, (int64_t) 3456, "mem_141526")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_141527, (int64_t) 3456, "mem_141527")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141536_cached_sizze_142264 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141536, &mem_141536_cached_sizze_142264, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141537_cached_sizze_142265 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141537, &mem_141537_cached_sizze_142265, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139228 = 0; i_139228 < (int64_t) 27; i_139228++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139221 = 0; i_139221 < (int64_t) 16; i_139221++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_138003;
            double r_138005 = 0.0;
            
            for (int64_t i_138004 = 0; i_138004 < (int64_t) 16; i_138004++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_138006 = ((double *) mem_140654)[i_138004 * (int64_t) 27 + i_139228];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_138007 = ((double *) mem_140449)[i_138004 * (int64_t) 16 + i_139221];
                
                // futhark/microgpt.fut:397:68-111
                
                double zt_res_138008 = zt_lhs_138006 * zt_rhs_138007;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_138009 = r_138005 + zt_res_138008;
                double r_tmp_141903 = zp_res_138009;
                
                r_138005 = r_tmp_141903;
            }
            defunc_0_lifted_lambda_res_138003 = r_138005;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_138012;
            double r_138014 = 0.0;
            
            for (int64_t i_138013 = 0; i_138013 < (int64_t) 16; i_138013++) {
                // futhark/microgpt.fut:460:62-71
                
                int64_t zeze_lhs_138015 = ((int64_t *) tokens_mem_139500.mem)[i_138013];
                
                // futhark/microgpt.fut:460:58-109
                
                bool cond_138016 = zeze_lhs_138015 == i_139228;
                
                // futhark/microgpt.fut:460:58-109
                
                double lifted_lambda_res_138017;
                
                if (cond_138016) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double lifted_lambda_res_t_res_138236 = ((double *) mem_141478)[i_138013 * (int64_t) 16 + i_139221];
                    
                    lifted_lambda_res_138017 = lifted_lambda_res_t_res_138236;
                } else {
                    lifted_lambda_res_138017 = 0.0;
                }
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_138023 = r_138014 + lifted_lambda_res_138017;
                double r_tmp_141904 = zp_res_138023;
                
                r_138014 = r_tmp_141904;
            }
            defunc_0_lifted_lambda_res_138012 = r_138014;
            ((double *) mem_141536)[i_139221] = defunc_0_lifted_lambda_res_138012;
            ((double *) mem_141537)[i_139221] = defunc_0_lifted_lambda_res_138003;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141526.mem, i_139228 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141536, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141527.mem, i_139228 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141537, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    if (memblock_set(ctx, &mem_out_141576, &mem_141526, "mem_141526") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_141577, &mem_141479, "mem_141479") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_141578, &mem_141321, "mem_141321") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_141579, &mem_141320, "mem_141320") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_141580, &mem_141319, "mem_141319") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_141581, &mem_141239, "mem_141239") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_141582, &mem_141510, "mem_141510") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_141583, &mem_140686, "mem_140686") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_141584, &mem_141527, "mem_141527") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_142021, &mem_out_141576, "mem_out_141576") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_142022, &mem_out_141577, "mem_out_141577") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_142023, &mem_out_141578, "mem_out_141578") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_142024, &mem_out_141579, "mem_out_141579") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_142025, &mem_out_141580, "mem_out_141580") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_142026, &mem_out_141581, "mem_out_141581") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_142027, &mem_out_141582, "mem_out_141582") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_142028, &mem_out_141583, "mem_out_141583") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_142029, &mem_out_141584, "mem_out_141584") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_139503);
        free(mem_139508);
        free(mem_139519);
        free(mem_139520);
        free(mem_139521);
        free(mem_139540);
        free(mem_139547);
        free(mem_139552);
        free(mem_139563);
        free(mem_139568);
        free(mem_139579);
        free(mem_139580);
        free(mem_139593);
        free(mem_139600);
        free(mem_139605);
        free(mem_139616);
        free(mem_139621);
        free(mem_139632);
        free(mem_139633);
        free(mem_139634);
        free(mem_139650);
        free(mem_139651);
        free(mem_139652);
        free(mem_139665);
        free(mem_139666);
        free(mem_139667);
        free(mem_139713);
        free(mem_139714);
        free(mem_139715);
        free(mem_139716);
        free(mem_139737);
        free(mem_139738);
        free(mem_139739);
        free(mem_139740);
        free(mem_139757);
        free(mem_139758);
        free(mem_139759);
        free(mem_139760);
        free(mem_139821);
        free(mem_139822);
        free(mem_139823);
        free(mem_139824);
        free(mem_139845);
        free(mem_139846);
        free(mem_139847);
        free(mem_139848);
        free(mem_139865);
        free(mem_139866);
        free(mem_139867);
        free(mem_139868);
        free(mem_139929);
        free(mem_139930);
        free(mem_139931);
        free(mem_139932);
        free(mem_139933);
        free(mem_139934);
        free(mem_139935);
        free(mem_139936);
        free(mem_139969);
        free(mem_139970);
        free(mem_139971);
        free(mem_139972);
        free(mem_139973);
        free(mem_139974);
        free(mem_139975);
        free(mem_139976);
        free(mem_140057);
        free(mem_140058);
        free(mem_140059);
        free(mem_140060);
        free(mem_140081);
        free(mem_140082);
        free(mem_140083);
        free(mem_140084);
        free(mem_140101);
        free(mem_140102);
        free(mem_140103);
        free(mem_140104);
        free(mem_140165);
        free(mem_140166);
        free(mem_140175);
        free(mem_140176);
        free(mem_140197);
        free(mem_140198);
        free(mem_140209);
        free(mem_140210);
        free(mem_140219);
        free(mem_140220);
        free(mem_140251);
        free(mem_140252);
        free(mem_140263);
        free(mem_140264);
        free(mem_140273);
        free(mem_140274);
        free(mem_140305);
        free(mem_140311);
        free(mem_140316);
        free(mem_140332);
        free(mem_140337);
        free(mem_140348);
        free(mem_140353);
        free(mem_140364);
        free(mem_140365);
        free(mem_140378);
        free(mem_140385);
        free(mem_140390);
        free(mem_140401);
        free(mem_140406);
        free(mem_140417);
        free(mem_140422);
        free(mem_140433);
        free(mem_140438);
        free(mem_140449);
        free(mem_140454);
        free(mem_140465);
        free(mem_140470);
        free(mem_140481);
        free(mem_140482);
        free(mem_140483);
        free(mem_140484);
        free(mem_140502);
        free(mem_140507);
        free(mem_140511);
        free(mem_140518);
        free(mem_140552);
        free(mem_140558);
        free(mem_140563);
        free(mem_140579);
        free(mem_140580);
        free(mem_140589);
        free(mem_140590);
        free(mem_140611);
        free(mem_140617);
        free(mem_140622);
        free(mem_140638);
        free(mem_140643);
        free(mem_140654);
        free(mem_140659);
        free(mem_140670);
        free(mem_140675);
        free(mem_140687);
        free(mem_140696);
        free(mem_140697);
        free(mem_140718);
        free(mem_140723);
        free(mem_140734);
        free(mem_140735);
        free(mem_140748);
        free(mem_140755);
        free(mem_140760);
        free(mem_140771);
        free(mem_140777);
        free(mem_140782);
        free(mem_140798);
        free(mem_140799);
        free(mem_140800);
        free(mem_140816);
        free(mem_140817);
        free(mem_140818);
        free(mem_140831);
        free(mem_140832);
        free(mem_140873);
        free(mem_140874);
        free(mem_140885);
        free(mem_140886);
        free(mem_140895);
        free(mem_140896);
        free(mem_140927);
        free(mem_140928);
        free(mem_140939);
        free(mem_140940);
        free(mem_140949);
        free(mem_140950);
        free(mem_140981);
        free(mem_140982);
        free(mem_140983);
        free(mem_140984);
        free(mem_141001);
        free(mem_141002);
        free(mem_141003);
        free(mem_141004);
        free(mem_141045);
        free(mem_141046);
        free(mem_141057);
        free(mem_141058);
        free(mem_141067);
        free(mem_141068);
        free(mem_141099);
        free(mem_141100);
        free(mem_141109);
        free(mem_141110);
        free(mem_141131);
        free(mem_141132);
        free(mem_141143);
        free(mem_141144);
        free(mem_141153);
        free(mem_141154);
        free(mem_141185);
        free(mem_141186);
        free(mem_141197);
        free(mem_141198);
        free(mem_141207);
        free(mem_141208);
        free(mem_141240);
        free(mem_141241);
        free(mem_141242);
        free(mem_141259);
        free(mem_141260);
        free(mem_141261);
        free(mem_141262);
        free(mem_141303);
        free(mem_141308);
        free(mem_141322);
        free(mem_141323);
        free(mem_141342);
        free(mem_141343);
        free(mem_141344);
        free(mem_141381);
        free(mem_141388);
        free(mem_141393);
        free(mem_141404);
        free(mem_141405);
        free(mem_141414);
        free(mem_141415);
        free(mem_141436);
        free(mem_141437);
        free(mem_141438);
        free(mem_141439);
        free(mem_141464);
        free(mem_141465);
        free(mem_141478);
        free(mem_141488);
        free(mem_141489);
        free(mem_141515);
        free(mem_141536);
        free(mem_141537);
        if (memblock_unref(ctx, &mem_141527, "mem_141527") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_141526, "mem_141526") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_141510, "mem_141510") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_141479, "mem_141479") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_141321, "mem_141321") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_141320, "mem_141320") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_141319, "mem_141319") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_141239, "mem_141239") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140686, "mem_140686") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141584, "mem_out_141584") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141583, "mem_out_141583") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141582, "mem_out_141582") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141581, "mem_out_141581") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141580, "mem_out_141580") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141579, "mem_out_141579") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141578, "mem_out_141578") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141577, "mem_out_141577") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141576, "mem_out_141576") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_make_params(struct futhark_context *ctx, struct memblock *mem_out_p_142266, struct memblock *mem_out_p_142267, struct memblock *mem_out_p_142268, struct memblock *mem_out_p_142269, struct memblock *mem_out_p_142270, struct memblock *mem_out_p_142271, struct memblock *mem_out_p_142272, struct memblock *mem_out_p_142273, struct memblock *mem_out_p_142274, struct memblock wte_mem_139491, struct memblock wpe_mem_139492, struct memblock wqry_mem_139493, struct memblock wkey_mem_139494, struct memblock wval_mem_139495, struct memblock wout_mem_139496, struct memblock wup_mem_139497, struct memblock wdown_mem_139498, struct memblock wvoc_mem_139499, int64_t sl_54536)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_141584;
    
    mem_out_141584.references = NULL;
    
    struct memblock mem_out_141583;
    
    mem_out_141583.references = NULL;
    
    struct memblock mem_out_141582;
    
    mem_out_141582.references = NULL;
    
    struct memblock mem_out_141581;
    
    mem_out_141581.references = NULL;
    
    struct memblock mem_out_141580;
    
    mem_out_141580.references = NULL;
    
    struct memblock mem_out_141579;
    
    mem_out_141579.references = NULL;
    
    struct memblock mem_out_141578;
    
    mem_out_141578.references = NULL;
    
    struct memblock mem_out_141577;
    
    mem_out_141577.references = NULL;
    
    struct memblock mem_out_141576;
    
    mem_out_141576.references = NULL;
    if (memblock_set(ctx, &mem_out_141576, &wdown_mem_139498, "wdown_mem_139498") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_141577, &wkey_mem_139494, "wkey_mem_139494") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_141578, &wout_mem_139496, "wout_mem_139496") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_141579, &wpe_mem_139492, "wpe_mem_139492") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_141580, &wqry_mem_139493, "wqry_mem_139493") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_141581, &wte_mem_139491, "wte_mem_139491") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_141582, &wup_mem_139497, "wup_mem_139497") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_141583, &wval_mem_139495, "wval_mem_139495") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_141584, &wvoc_mem_139499, "wvoc_mem_139499") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_142266, &mem_out_141576, "mem_out_141576") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_142267, &mem_out_141577, "mem_out_141577") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_142268, &mem_out_141578, "mem_out_141578") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_142269, &mem_out_141579, "mem_out_141579") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_142270, &mem_out_141580, "mem_out_141580") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_142271, &mem_out_141581, "mem_out_141581") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_142272, &mem_out_141582, "mem_out_141582") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_142273, &mem_out_141583, "mem_out_141583") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_142274, &mem_out_141584, "mem_out_141584") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_141584, "mem_out_141584") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141583, "mem_out_141583") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141582, "mem_out_141582") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141581, "mem_out_141581") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141580, "mem_out_141580") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141579, "mem_out_141579") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141578, "mem_out_141578") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141577, "mem_out_141577") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_141576, "mem_out_141576") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_cal_loss(struct futhark_context *ctx, struct futhark_opaque_tup2_f64_arr1d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3)
{
    double prim_out_141577 = 0.0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_141576;
    
    mem_out_141576.references = NULL;
    
    struct memblock mask_mem_139502;
    
    mask_mem_139502.references = NULL;
    
    struct memblock target_mem_139501;
    
    target_mem_139501.references = NULL;
    
    struct memblock tokens_mem_139500;
    
    tokens_mem_139500.references = NULL;
    
    struct memblock wvoc_mem_139499;
    
    wvoc_mem_139499.references = NULL;
    
    struct memblock wval_mem_139498;
    
    wval_mem_139498.references = NULL;
    
    struct memblock wup_mem_139497;
    
    wup_mem_139497.references = NULL;
    
    struct memblock wte_mem_139496;
    
    wte_mem_139496.references = NULL;
    
    struct memblock wqry_mem_139495;
    
    wqry_mem_139495.references = NULL;
    
    struct memblock wpe_mem_139494;
    
    wpe_mem_139494.references = NULL;
    
    struct memblock wout_mem_139493;
    
    wout_mem_139493.references = NULL;
    
    struct memblock wkey_mem_139492;
    
    wkey_mem_139492.references = NULL;
    
    struct memblock wdown_mem_139491;
    
    wdown_mem_139491.references = NULL;
    wdown_mem_139491 = in0->v0->mem;
    wkey_mem_139492 = in0->v1->mem;
    wout_mem_139493 = in0->v2->mem;
    wpe_mem_139494 = in0->v3->mem;
    wqry_mem_139495 = in0->v4->mem;
    wte_mem_139496 = in0->v5->mem;
    wup_mem_139497 = in0->v6->mem;
    wval_mem_139498 = in0->v7->mem;
    wvoc_mem_139499 = in0->v8->mem;
    tokens_mem_139500 = in1->mem;
    target_mem_139501 = in2->mem;
    mask_mem_139502 = in3->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && (((int64_t) 16 == in2->shape[0] && (int64_t) 27 == in2->shape[1]) && ((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_cal_loss(ctx, &mem_out_141576, &prim_out_141577, wdown_mem_139491, wkey_mem_139492, wout_mem_139493, wpe_mem_139494, wqry_mem_139495, wte_mem_139496, wup_mem_139497, wval_mem_139498, wvoc_mem_139499, tokens_mem_139500, target_mem_139501, mask_mem_139502);
        if (ret == 0) {
            assert((*out = (struct futhark_opaque_tup2_f64_arr1d_f64 *) malloc(sizeof(struct futhark_opaque_tup2_f64_arr1d_f64))) != NULL);
            (*out)->v0 = prim_out_141577;
            assert(((*out)->v1 = (struct futhark_f64_1d *) malloc(sizeof(struct futhark_f64_1d))) != NULL);
            (*out)->v1->mem = mem_out_141576;
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
    
    struct memblock mem_out_141576;
    
    mem_out_141576.references = NULL;
    
    struct memblock mask_mem_139501;
    
    mask_mem_139501.references = NULL;
    
    struct memblock tokens_mem_139500;
    
    tokens_mem_139500.references = NULL;
    
    struct memblock wvoc_mem_139499;
    
    wvoc_mem_139499.references = NULL;
    
    struct memblock wval_mem_139498;
    
    wval_mem_139498.references = NULL;
    
    struct memblock wup_mem_139497;
    
    wup_mem_139497.references = NULL;
    
    struct memblock wte_mem_139496;
    
    wte_mem_139496.references = NULL;
    
    struct memblock wqry_mem_139495;
    
    wqry_mem_139495.references = NULL;
    
    struct memblock wpe_mem_139494;
    
    wpe_mem_139494.references = NULL;
    
    struct memblock wout_mem_139493;
    
    wout_mem_139493.references = NULL;
    
    struct memblock wkey_mem_139492;
    
    wkey_mem_139492.references = NULL;
    
    struct memblock wdown_mem_139491;
    
    wdown_mem_139491.references = NULL;
    wdown_mem_139491 = in0->v0->mem;
    wkey_mem_139492 = in0->v1->mem;
    wout_mem_139493 = in0->v2->mem;
    wpe_mem_139494 = in0->v3->mem;
    wqry_mem_139495 = in0->v4->mem;
    wte_mem_139496 = in0->v5->mem;
    wup_mem_139497 = in0->v6->mem;
    wval_mem_139498 = in0->v7->mem;
    wvoc_mem_139499 = in0->v8->mem;
    tokens_mem_139500 = in1->mem;
    mask_mem_139501 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_141576, wdown_mem_139491, wkey_mem_139492, wout_mem_139493, wpe_mem_139494, wqry_mem_139495, wte_mem_139496, wup_mem_139497, wval_mem_139498, wvoc_mem_139499, tokens_mem_139500, mask_mem_139501);
        if (ret == 0) {
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_141576;
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
    
    struct memblock mem_out_141584;
    
    mem_out_141584.references = NULL;
    
    struct memblock mem_out_141583;
    
    mem_out_141583.references = NULL;
    
    struct memblock mem_out_141582;
    
    mem_out_141582.references = NULL;
    
    struct memblock mem_out_141581;
    
    mem_out_141581.references = NULL;
    
    struct memblock mem_out_141580;
    
    mem_out_141580.references = NULL;
    
    struct memblock mem_out_141579;
    
    mem_out_141579.references = NULL;
    
    struct memblock mem_out_141578;
    
    mem_out_141578.references = NULL;
    
    struct memblock mem_out_141577;
    
    mem_out_141577.references = NULL;
    
    struct memblock mem_out_141576;
    
    mem_out_141576.references = NULL;
    
    struct memblock mask_mem_139502;
    
    mask_mem_139502.references = NULL;
    
    struct memblock target_mem_139501;
    
    target_mem_139501.references = NULL;
    
    struct memblock tokens_mem_139500;
    
    tokens_mem_139500.references = NULL;
    
    struct memblock wvoc_mem_139499;
    
    wvoc_mem_139499.references = NULL;
    
    struct memblock wval_mem_139498;
    
    wval_mem_139498.references = NULL;
    
    struct memblock wup_mem_139497;
    
    wup_mem_139497.references = NULL;
    
    struct memblock wte_mem_139496;
    
    wte_mem_139496.references = NULL;
    
    struct memblock wqry_mem_139495;
    
    wqry_mem_139495.references = NULL;
    
    struct memblock wpe_mem_139494;
    
    wpe_mem_139494.references = NULL;
    
    struct memblock wout_mem_139493;
    
    wout_mem_139493.references = NULL;
    
    struct memblock wkey_mem_139492;
    
    wkey_mem_139492.references = NULL;
    
    struct memblock wdown_mem_139491;
    
    wdown_mem_139491.references = NULL;
    wdown_mem_139491 = in0->v0->mem;
    wkey_mem_139492 = in0->v1->mem;
    wout_mem_139493 = in0->v2->mem;
    wpe_mem_139494 = in0->v3->mem;
    wqry_mem_139495 = in0->v4->mem;
    wte_mem_139496 = in0->v5->mem;
    wup_mem_139497 = in0->v6->mem;
    wval_mem_139498 = in0->v7->mem;
    wvoc_mem_139499 = in0->v8->mem;
    tokens_mem_139500 = in1->mem;
    target_mem_139501 = in2->mem;
    mask_mem_139502 = in3->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && (((int64_t) 16 == in2->shape[0] && (int64_t) 27 == in2->shape[1]) && ((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_grad_loss(ctx, &mem_out_141576, &mem_out_141577, &mem_out_141578, &mem_out_141579, &mem_out_141580, &mem_out_141581, &mem_out_141582, &mem_out_141583, &mem_out_141584, wdown_mem_139491, wkey_mem_139492, wout_mem_139493, wpe_mem_139494, wqry_mem_139495, wte_mem_139496, wup_mem_139497, wval_mem_139498, wvoc_mem_139499, tokens_mem_139500, target_mem_139501, mask_mem_139502);
        if (ret == 0) {
            assert((*out = (struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_141576;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_141577;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_141578;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_141579;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_141580;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_141581;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_141582;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_141583;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_141584;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_make_params(struct futhark_context *ctx, struct futhark_opaque_params **out, const struct futhark_f64_2d *in0, const struct futhark_f64_2d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3, const struct futhark_f64_2d *in4, const struct futhark_f64_2d *in5, const struct futhark_f64_2d *in6, const struct futhark_f64_2d *in7, const struct futhark_f64_2d *in8)
{
    int64_t sl_54536 = (int64_t) 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_141584;
    
    mem_out_141584.references = NULL;
    
    struct memblock mem_out_141583;
    
    mem_out_141583.references = NULL;
    
    struct memblock mem_out_141582;
    
    mem_out_141582.references = NULL;
    
    struct memblock mem_out_141581;
    
    mem_out_141581.references = NULL;
    
    struct memblock mem_out_141580;
    
    mem_out_141580.references = NULL;
    
    struct memblock mem_out_141579;
    
    mem_out_141579.references = NULL;
    
    struct memblock mem_out_141578;
    
    mem_out_141578.references = NULL;
    
    struct memblock mem_out_141577;
    
    mem_out_141577.references = NULL;
    
    struct memblock mem_out_141576;
    
    mem_out_141576.references = NULL;
    
    struct memblock wvoc_mem_139499;
    
    wvoc_mem_139499.references = NULL;
    
    struct memblock wdown_mem_139498;
    
    wdown_mem_139498.references = NULL;
    
    struct memblock wup_mem_139497;
    
    wup_mem_139497.references = NULL;
    
    struct memblock wout_mem_139496;
    
    wout_mem_139496.references = NULL;
    
    struct memblock wval_mem_139495;
    
    wval_mem_139495.references = NULL;
    
    struct memblock wkey_mem_139494;
    
    wkey_mem_139494.references = NULL;
    
    struct memblock wqry_mem_139493;
    
    wqry_mem_139493.references = NULL;
    
    struct memblock wpe_mem_139492;
    
    wpe_mem_139492.references = NULL;
    
    struct memblock wte_mem_139491;
    
    wte_mem_139491.references = NULL;
    wte_mem_139491 = in0->mem;
    sl_54536 = in0->shape[1];
    wpe_mem_139492 = in1->mem;
    sl_54536 = in1->shape[0];
    wqry_mem_139493 = in2->mem;
    wkey_mem_139494 = in3->mem;
    wval_mem_139495 = in4->mem;
    wout_mem_139496 = in5->mem;
    wup_mem_139497 = in6->mem;
    wdown_mem_139498 = in7->mem;
    wvoc_mem_139499 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && sl_54536 == in0->shape[1]) && ((sl_54536 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_make_params(ctx, &mem_out_141576, &mem_out_141577, &mem_out_141578, &mem_out_141579, &mem_out_141580, &mem_out_141581, &mem_out_141582, &mem_out_141583, &mem_out_141584, wte_mem_139491, wpe_mem_139492, wqry_mem_139493, wkey_mem_139494, wval_mem_139495, wout_mem_139496, wup_mem_139497, wdown_mem_139498, wvoc_mem_139499, sl_54536);
        if (ret == 0) {
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_141576;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_141577;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_141578;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_141579;
            (*out)->v3->shape[0] = sl_54536;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_141580;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_141581;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = sl_54536;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_141582;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_141583;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_141584;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
