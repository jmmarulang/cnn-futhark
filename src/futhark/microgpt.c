
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

FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_143474, double *out_prim_out_143475, struct memblock wdown_mem_141053, struct memblock wkey_mem_141054, struct memblock wout_mem_141055, struct memblock wpe_mem_141056, struct memblock wqry_mem_141057, struct memblock wte_mem_141058, struct memblock wup_mem_141059, struct memblock wval_mem_141060, struct memblock wvoc_mem_141061, struct memblock tokens_mem_141062, struct memblock target_mem_141063, struct memblock mask_mem_141064);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_143533, struct memblock wdown_mem_141053, struct memblock wkey_mem_141054, struct memblock wout_mem_141055, struct memblock wpe_mem_141056, struct memblock wqry_mem_141057, struct memblock wte_mem_141058, struct memblock wup_mem_141059, struct memblock wval_mem_141060, struct memblock wvoc_mem_141061, struct memblock tokens_mem_141062, struct memblock mask_mem_141063);
FUTHARK_FUN_ATTR int futrts_entry_grad_loss(struct futhark_context *ctx, struct memblock *mem_out_p_143590, struct memblock *mem_out_p_143591, struct memblock *mem_out_p_143592, struct memblock *mem_out_p_143593, struct memblock *mem_out_p_143594, struct memblock *mem_out_p_143595, struct memblock *mem_out_p_143596, struct memblock *mem_out_p_143597, struct memblock *mem_out_p_143598, struct memblock wdown_mem_141053, struct memblock wkey_mem_141054, struct memblock wout_mem_141055, struct memblock wpe_mem_141056, struct memblock wqry_mem_141057, struct memblock wte_mem_141058, struct memblock wup_mem_141059, struct memblock wval_mem_141060, struct memblock wvoc_mem_141061, struct memblock tokens_mem_141062, struct memblock target_mem_141063, struct memblock mask_mem_141064);
FUTHARK_FUN_ATTR int futrts_entry_make_params(struct futhark_context *ctx, struct memblock *mem_out_p_143835, struct memblock *mem_out_p_143836, struct memblock *mem_out_p_143837, struct memblock *mem_out_p_143838, struct memblock *mem_out_p_143839, struct memblock *mem_out_p_143840, struct memblock *mem_out_p_143841, struct memblock *mem_out_p_143842, struct memblock *mem_out_p_143843, struct memblock wte_mem_141053, struct memblock wpe_mem_141054, struct memblock wqry_mem_141055, struct memblock wkey_mem_141056, struct memblock wval_mem_141057, struct memblock wout_mem_141058, struct memblock wup_mem_141059, struct memblock wdown_mem_141060, struct memblock wvoc_mem_141061, int64_t sl_55409);

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

FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_143474, double *out_prim_out_143475, struct memblock wdown_mem_141053, struct memblock wkey_mem_141054, struct memblock wout_mem_141055, struct memblock wpe_mem_141056, struct memblock wqry_mem_141057, struct memblock wte_mem_141058, struct memblock wup_mem_141059, struct memblock wval_mem_141060, struct memblock wvoc_mem_141061, struct memblock tokens_mem_141062, struct memblock target_mem_141063, struct memblock mask_mem_141064)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_141065_cached_sizze_143476 = 0;
    unsigned char *mem_141065 = NULL;
    int64_t mem_141070_cached_sizze_143477 = 0;
    unsigned char *mem_141070 = NULL;
    int64_t mem_141081_cached_sizze_143478 = 0;
    unsigned char *mem_141081 = NULL;
    int64_t mem_141086_cached_sizze_143479 = 0;
    unsigned char *mem_141086 = NULL;
    int64_t mem_141093_cached_sizze_143480 = 0;
    unsigned char *mem_141093 = NULL;
    int64_t mem_141104_cached_sizze_143481 = 0;
    unsigned char *mem_141104 = NULL;
    int64_t mem_141109_cached_sizze_143482 = 0;
    unsigned char *mem_141109 = NULL;
    int64_t mem_141116_cached_sizze_143483 = 0;
    unsigned char *mem_141116 = NULL;
    int64_t mem_141127_cached_sizze_143484 = 0;
    unsigned char *mem_141127 = NULL;
    int64_t mem_141128_cached_sizze_143485 = 0;
    unsigned char *mem_141128 = NULL;
    int64_t mem_141129_cached_sizze_143486 = 0;
    unsigned char *mem_141129 = NULL;
    int64_t mem_141142_cached_sizze_143487 = 0;
    unsigned char *mem_141142 = NULL;
    int64_t mem_141143_cached_sizze_143488 = 0;
    unsigned char *mem_141143 = NULL;
    int64_t mem_141144_cached_sizze_143489 = 0;
    unsigned char *mem_141144 = NULL;
    int64_t mem_141175_cached_sizze_143490 = 0;
    unsigned char *mem_141175 = NULL;
    int64_t mem_141176_cached_sizze_143491 = 0;
    unsigned char *mem_141176 = NULL;
    int64_t mem_141177_cached_sizze_143492 = 0;
    unsigned char *mem_141177 = NULL;
    int64_t mem_141193_cached_sizze_143493 = 0;
    unsigned char *mem_141193 = NULL;
    int64_t mem_141194_cached_sizze_143494 = 0;
    unsigned char *mem_141194 = NULL;
    int64_t mem_141195_cached_sizze_143495 = 0;
    unsigned char *mem_141195 = NULL;
    int64_t mem_141208_cached_sizze_143496 = 0;
    unsigned char *mem_141208 = NULL;
    int64_t mem_141209_cached_sizze_143497 = 0;
    unsigned char *mem_141209 = NULL;
    int64_t mem_141210_cached_sizze_143498 = 0;
    unsigned char *mem_141210 = NULL;
    int64_t mem_141256_cached_sizze_143499 = 0;
    unsigned char *mem_141256 = NULL;
    int64_t mem_141262_cached_sizze_143500 = 0;
    unsigned char *mem_141262 = NULL;
    int64_t mem_141267_cached_sizze_143501 = 0;
    unsigned char *mem_141267 = NULL;
    int64_t mem_141278_cached_sizze_143502 = 0;
    unsigned char *mem_141278 = NULL;
    int64_t mem_141283_cached_sizze_143503 = 0;
    unsigned char *mem_141283 = NULL;
    int64_t mem_141294_cached_sizze_143504 = 0;
    unsigned char *mem_141294 = NULL;
    int64_t mem_141299_cached_sizze_143505 = 0;
    unsigned char *mem_141299 = NULL;
    int64_t mem_141306_cached_sizze_143506 = 0;
    unsigned char *mem_141306 = NULL;
    int64_t mem_141313_cached_sizze_143507 = 0;
    unsigned char *mem_141313 = NULL;
    int64_t mem_141324_cached_sizze_143508 = 0;
    unsigned char *mem_141324 = NULL;
    int64_t mem_141329_cached_sizze_143509 = 0;
    unsigned char *mem_141329 = NULL;
    int64_t mem_141340_cached_sizze_143510 = 0;
    unsigned char *mem_141340 = NULL;
    int64_t mem_141345_cached_sizze_143511 = 0;
    unsigned char *mem_141345 = NULL;
    int64_t mem_141361_cached_sizze_143512 = 0;
    unsigned char *mem_141361 = NULL;
    int64_t mem_141366_cached_sizze_143513 = 0;
    unsigned char *mem_141366 = NULL;
    int64_t mem_141377_cached_sizze_143514 = 0;
    unsigned char *mem_141377 = NULL;
    int64_t mem_141382_cached_sizze_143515 = 0;
    unsigned char *mem_141382 = NULL;
    int64_t mem_141393_cached_sizze_143516 = 0;
    unsigned char *mem_141393 = NULL;
    int64_t mem_141398_cached_sizze_143517 = 0;
    unsigned char *mem_141398 = NULL;
    int64_t mem_141409_cached_sizze_143518 = 0;
    unsigned char *mem_141409 = NULL;
    int64_t mem_141414_cached_sizze_143519 = 0;
    unsigned char *mem_141414 = NULL;
    int64_t mem_141421_cached_sizze_143520 = 0;
    unsigned char *mem_141421 = NULL;
    int64_t mem_141432_cached_sizze_143521 = 0;
    unsigned char *mem_141432 = NULL;
    int64_t mem_141437_cached_sizze_143522 = 0;
    unsigned char *mem_141437 = NULL;
    int64_t mem_141448_cached_sizze_143523 = 0;
    unsigned char *mem_141448 = NULL;
    int64_t mem_141453_cached_sizze_143524 = 0;
    unsigned char *mem_141453 = NULL;
    int64_t mem_141464_cached_sizze_143525 = 0;
    unsigned char *mem_141464 = NULL;
    int64_t mem_141469_cached_sizze_143526 = 0;
    unsigned char *mem_141469 = NULL;
    int64_t mem_141480_cached_sizze_143527 = 0;
    unsigned char *mem_141480 = NULL;
    int64_t mem_141485_cached_sizze_143528 = 0;
    unsigned char *mem_141485 = NULL;
    int64_t mem_141496_cached_sizze_143529 = 0;
    unsigned char *mem_141496 = NULL;
    int64_t mem_141501_cached_sizze_143530 = 0;
    unsigned char *mem_141501 = NULL;
    int64_t mem_141516_cached_sizze_143531 = 0;
    unsigned char *mem_141516 = NULL;
    int64_t mem_141523_cached_sizze_143532 = 0;
    unsigned char *mem_141523 = NULL;
    struct memblock mem_141512;
    
    mem_141512.references = NULL;
    
    struct memblock mem_out_143138;
    
    mem_out_143138.references = NULL;
    
    double prim_out_143139;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_141065_cached_sizze_143476 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141065, &mem_141065_cached_sizze_143476, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141070_cached_sizze_143477 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141070, &mem_141070_cached_sizze_143477, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139925 = 0; i_139925 < (int64_t) 16; i_139925++) {
        // futhark/microgpt.fut:441:41-50
        
        int64_t tmp_126287 = ((int64_t *) tokens_mem_141062.mem)[i_139925];
        
        // futhark/microgpt.fut:441:37-51
        
        bool x_126288 = sle64((int64_t) 0, tmp_126287);
        
        // futhark/microgpt.fut:441:37-51
        
        bool y_126289 = slt64(tmp_126287, (int64_t) 27);
        
        // futhark/microgpt.fut:441:37-51
        
        bool bounds_check_126290 = x_126288 && y_126289;
        
        // futhark/microgpt.fut:441:37-51
        
        bool index_certs_126291;
        
        if (!bounds_check_126290) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_126287, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:441:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:441:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139921 = 0; i_139921 < (int64_t) 16; i_139921++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_126298 = ((double *) wte_mem_141058.mem)[tmp_126287 * (int64_t) 16 + i_139921];
            
            ((double *) mem_141070)[i_139921] = lifted_lambda_res_126298;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141065, i_139925 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141070, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141081_cached_sizze_143478 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141081, &mem_141081_cached_sizze_143478, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141086_cached_sizze_143479 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141086, &mem_141086_cached_sizze_143479, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141093_cached_sizze_143480 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141093, &mem_141093_cached_sizze_143480, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139937 = 0; i_139937 < (int64_t) 16; i_139937++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_126324;
        double r_126326 = 0.0;
        
        for (int64_t i_126325 = 0; i_126325 < (int64_t) 16; i_126325++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_126327 = ((double *) wpe_mem_141056.mem)[i_139937 * (int64_t) 16 + i_126325];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_126328 = ((double *) mem_141065)[i_139937 * (int64_t) 16 + i_126325];
            
            // futhark/microgpt.fut:193:76-116
            
            double zp_res_126329 = zp_lhs_126327 + zp_rhs_126328;
            
            // futhark/microgpt.fut:193:94-163
            
            double zt_res_126330 = zp_res_126329 * zp_res_126329;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_126331 = r_126326 + zt_res_126330;
            double r_tmp_143143 = zp_res_126331;
            
            r_126326 = r_tmp_143143;
        }
        defunc_0_lifted_lambda_res_126324 = r_126326;
        // futhark/microgpt.fut:193:54-182
        
        double zs_res_126332 = defunc_0_lifted_lambda_res_126324 / 16.0;
        
        // futhark/microgpt.fut:194:24-55
        
        double zp_res_126333 = 1.0e-5 + zs_res_126332;
        
        // futhark/microgpt.fut:194:16-55
        
        double sqrt_res_126334 = futrts_sqrt64(zp_res_126333);
        
        // futhark/microgpt.fut:195:85-96
        
        double zs_res_126335 = 1.0 / sqrt_res_126334;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139929 = 0; i_139929 < (int64_t) 16; i_139929++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_126342 = ((double *) wpe_mem_141056.mem)[i_139937 * (int64_t) 16 + i_139929];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_126343 = ((double *) mem_141065)[i_139937 * (int64_t) 16 + i_139929];
            
            // futhark/microgpt.fut:195:38-78
            
            double zp_res_126344 = zp_lhs_126342 + zp_rhs_126343;
            
            // futhark/microgpt.fut:195:56-96
            
            double zt_res_126345 = zs_res_126335 * zp_res_126344;
            
            ((double *) mem_141086)[i_139929] = zt_res_126345;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139933 = 0; i_139933 < (int64_t) 16; i_139933++) {
            // futhark/microgpt.fut:196:4-14
            
            double lifted_lambda_res_126353 = ((double *) mem_141086)[i_139933];
            
            ((double *) mem_141093)[i_139933] = lifted_lambda_res_126353;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141081, i_139937 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141093, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141104_cached_sizze_143481 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141104, &mem_141104_cached_sizze_143481, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141109_cached_sizze_143482 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141109, &mem_141109_cached_sizze_143482, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141116_cached_sizze_143483 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141116, &mem_141116_cached_sizze_143483, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139949 = 0; i_139949 < (int64_t) 16; i_139949++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_126362;
        double r_126364 = 0.0;
        
        for (int64_t i_126363 = 0; i_126363 < (int64_t) 16; i_126363++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_126365 = ((double *) mem_141081)[i_139949 * (int64_t) 16 + i_126363];
            
            // futhark/microgpt.fut:197:78-115
            
            double zt_res_126366 = zt_lhs_126365 * zt_lhs_126365;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_126367 = r_126364 + zt_res_126366;
            double r_tmp_143147 = zp_res_126367;
            
            r_126364 = r_tmp_143147;
        }
        defunc_0_lifted_lambda_res_126362 = r_126364;
        // futhark/microgpt.fut:197:57-133
        
        double zs_res_126368 = defunc_0_lifted_lambda_res_126362 / 16.0;
        
        // futhark/microgpt.fut:198:24-55
        
        double zp_res_126369 = 1.0e-5 + zs_res_126368;
        
        // futhark/microgpt.fut:198:16-55
        
        double sqrt_res_126370 = futrts_sqrt64(zp_res_126369);
        
        // futhark/microgpt.fut:199:59-70
        
        double zs_res_126371 = 1.0 / sqrt_res_126370;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139941 = 0; i_139941 < (int64_t) 16; i_139941++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_126378 = ((double *) mem_141081)[i_139949 * (int64_t) 16 + i_139941];
            
            // futhark/microgpt.fut:199:37-70
            
            double zt_res_126379 = zs_res_126371 * zt_lhs_126378;
            
            ((double *) mem_141109)[i_139941] = zt_res_126379;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139945 = 0; i_139945 < (int64_t) 16; i_139945++) {
            // futhark/microgpt.fut:200:4-14
            
            double lifted_lambda_res_126387 = ((double *) mem_141109)[i_139945];
            
            ((double *) mem_141116)[i_139945] = lifted_lambda_res_126387;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141104, i_139949 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141116, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141127_cached_sizze_143484 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141127, &mem_141127_cached_sizze_143484, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141128_cached_sizze_143485 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141128, &mem_141128_cached_sizze_143485, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141129_cached_sizze_143486 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141129, &mem_141129_cached_sizze_143486, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141142_cached_sizze_143487 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141142, &mem_141142_cached_sizze_143487, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141143_cached_sizze_143488 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141143, &mem_141143_cached_sizze_143488, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141144_cached_sizze_143489 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141144, &mem_141144_cached_sizze_143489, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139967 = 0; i_139967 < (int64_t) 16; i_139967++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139957 = 0; i_139957 < (int64_t) 16; i_139957++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129405;
            double r_129407 = 0.0;
            
            for (int64_t i_129406 = 0; i_129406 < (int64_t) 16; i_129406++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_129408 = ((double *) wqry_mem_141057.mem)[i_139957 * (int64_t) 16 + i_129406];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_129409 = ((double *) mem_141104)[i_139967 * (int64_t) 16 + i_129406];
                
                // futhark/microgpt.fut:201:66-105
                
                double zt_res_129410 = zt_lhs_129408 * zt_rhs_129409;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129411 = r_129407 + zt_res_129410;
                double r_tmp_143156 = zp_res_129411;
                
                r_129407 = r_tmp_143156;
            }
            defunc_0_lifted_lambda_res_129405 = r_129407;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129418;
            double r_129420 = 0.0;
            
            for (int64_t i_129419 = 0; i_129419 < (int64_t) 16; i_129419++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_129421 = ((double *) wkey_mem_141054.mem)[i_139957 * (int64_t) 16 + i_129419];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_129422 = ((double *) mem_141104)[i_139967 * (int64_t) 16 + i_129419];
                
                // futhark/microgpt.fut:202:66-105
                
                double zt_res_129423 = zt_lhs_129421 * zt_rhs_129422;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129424 = r_129420 + zt_res_129423;
                double r_tmp_143157 = zp_res_129424;
                
                r_129420 = r_tmp_143157;
            }
            defunc_0_lifted_lambda_res_129418 = r_129420;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129434;
            double r_129436 = 0.0;
            
            for (int64_t i_129435 = 0; i_129435 < (int64_t) 16; i_129435++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_129437 = ((double *) wval_mem_141060.mem)[i_139957 * (int64_t) 16 + i_129435];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_129438 = ((double *) mem_141104)[i_139967 * (int64_t) 16 + i_129435];
                
                // futhark/microgpt.fut:203:66-105
                
                double zt_res_129439 = zt_lhs_129437 * zt_rhs_129438;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129440 = r_129436 + zt_res_129439;
                double r_tmp_143158 = zp_res_129440;
                
                r_129436 = r_tmp_143158;
            }
            defunc_0_lifted_lambda_res_129434 = r_129436;
            ((double *) mem_141142)[i_139957] = defunc_0_lifted_lambda_res_129434;
            ((double *) mem_141143)[i_139957] = defunc_0_lifted_lambda_res_129418;
            ((double *) mem_141144)[i_139957] = defunc_0_lifted_lambda_res_129405;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141127, i_139967 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141142, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141128, i_139967 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141143, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141129, i_139967 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141144, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141175_cached_sizze_143490 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141175, &mem_141175_cached_sizze_143490, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141176_cached_sizze_143491 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141176, &mem_141176_cached_sizze_143491, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141177_cached_sizze_143492 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141177, &mem_141177_cached_sizze_143492, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141193_cached_sizze_143493 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141193, &mem_141193_cached_sizze_143493, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141194_cached_sizze_143494 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141194, &mem_141194_cached_sizze_143494, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141195_cached_sizze_143495 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141195, &mem_141195_cached_sizze_143495, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141208_cached_sizze_143496 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_141208, &mem_141208_cached_sizze_143496, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141209_cached_sizze_143497 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_141209, &mem_141209_cached_sizze_143497, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141210_cached_sizze_143498 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_141210, &mem_141210_cached_sizze_143498, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139997 = 0; i_139997 < (int64_t) 4; i_139997++) {
        // futhark/microgpt.fut:204:69-72
        
        int64_t zp_lhs_129281 = mul64((int64_t) 4, i_139997);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139987 = 0; i_139987 < (int64_t) 16; i_139987++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_139977 = 0; i_139977 < (int64_t) 4; i_139977++) {
                // futhark/microgpt.fut:204:74-81
                
                int64_t tmp_129598 = add64(zp_lhs_129281, i_139977);
                
                // futhark/microgpt.fut:204:51-83
                
                bool x_129599 = sle64((int64_t) 0, tmp_129598);
                
                // futhark/microgpt.fut:204:51-83
                
                bool y_129600 = slt64(tmp_129598, (int64_t) 16);
                
                // futhark/microgpt.fut:204:51-83
                
                bool bounds_check_129601 = x_129599 && y_129600;
                
                // futhark/microgpt.fut:204:51-83
                
                bool index_certs_129602;
                
                if (!bounds_check_129601) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_129598, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:204:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:204:15-84\n   #9  futhark/microgpt.fut:442:7-76\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129603 = ((double *) mem_141129)[i_139987 * (int64_t) 16 + tmp_129598];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129611 = ((double *) mem_141128)[i_139987 * (int64_t) 16 + tmp_129598];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129622 = ((double *) mem_141127)[i_139987 * (int64_t) 16 + tmp_129598];
                
                ((double *) mem_141208)[i_139977] = lifted_lambda_res_129622;
                ((double *) mem_141209)[i_139977] = lifted_lambda_res_129611;
                ((double *) mem_141210)[i_139977] = lifted_lambda_res_129603;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141193, i_139987 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141208, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141194, i_139987 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141209, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141195, i_139987 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141210, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141175, i_139997 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_141193, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141176, i_139997 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_141194, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141177, i_139997 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_141195, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141256_cached_sizze_143499 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141256, &mem_141256_cached_sizze_143499, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141262_cached_sizze_143500 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141262, &mem_141262_cached_sizze_143500, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141267_cached_sizze_143501 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141267, &mem_141267_cached_sizze_143501, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141278_cached_sizze_143502 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141278, &mem_141278_cached_sizze_143502, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141283_cached_sizze_143503 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141283, &mem_141283_cached_sizze_143503, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141294_cached_sizze_143504 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141294, &mem_141294_cached_sizze_143504, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141299_cached_sizze_143505 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141299, &mem_141299_cached_sizze_143505, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141306_cached_sizze_143506 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141306, &mem_141306_cached_sizze_143506, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141313_cached_sizze_143507 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141313, &mem_141313_cached_sizze_143507, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141324_cached_sizze_143508 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141324, &mem_141324_cached_sizze_143508, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141329_cached_sizze_143509 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_141329, &mem_141329_cached_sizze_143509, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141340_cached_sizze_143510 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141340, &mem_141340_cached_sizze_143510, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141345_cached_sizze_143511 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_141345, &mem_141345_cached_sizze_143511, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140053 = 0; i_140053 < (int64_t) 4; i_140053++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140007 = 0; i_140007 < (int64_t) 16; i_140007++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140003 = 0; i_140003 < (int64_t) 16; i_140003++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_126532;
                double r_126534 = 0.0;
                
                for (int64_t i_126533 = 0; i_126533 < (int64_t) 4; i_126533++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_126535 = ((double *) mem_141177)[i_140053 * (int64_t) 64 + i_140007 * (int64_t) 4 + i_126533];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_126536 = ((double *) mem_141176)[i_140053 * (int64_t) 64 + i_140003 * (int64_t) 4 + i_126533];
                    
                    // futhark/microgpt.fut:207:113-164
                    
                    double zt_res_126537 = zt_lhs_126535 * zt_rhs_126536;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_126538 = r_126534 + zt_res_126537;
                    double r_tmp_143171 = zp_res_126538;
                    
                    r_126534 = r_tmp_143171;
                }
                defunc_0_lifted_lambda_res_126532 = r_126534;
                ((double *) mem_141267)[i_140003] = defunc_0_lifted_lambda_res_126532;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141262, i_140007 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141267, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140015 = 0; i_140015 < (int64_t) 16; i_140015++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140011 = 0; i_140011 < (int64_t) 16; i_140011++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_126553 = ((double *) mem_141262)[i_140015 * (int64_t) 16 + i_140011];
                
                // futhark/microgpt.fut:208:47-78
                
                double zs_res_126554 = zs_lhs_126553 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_126555 = ((double *) mask_mem_141064.mem)[i_140015 * (int64_t) 16 + i_140011];
                
                // futhark/microgpt.fut:208:65-102
                
                double zp_res_126556 = zs_res_126554 + zp_rhs_126555;
                
                ((double *) mem_141283)[i_140011] = zp_res_126556;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141278, i_140015 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141283, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140033 = 0; i_140033 < (int64_t) 16; i_140033++) {
            // futhark/microgpt.fut:105:13-33
            
            double defunc_0_reduce_res_129725;
            double redout_140017 = -INFINITY;
            
            for (int64_t i_140018 = 0; i_140018 < (int64_t) 16; i_140018++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129649 = ((double *) mem_141278)[i_140033 * (int64_t) 16 + i_140018];
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_126577 = fmax64(lifted_lambda_res_129649, redout_140017);
                double redout_tmp_143175 = max_res_126577;
                
                redout_140017 = redout_tmp_143175;
            }
            defunc_0_reduce_res_129725 = redout_140017;
            // futhark/microgpt.fut:210:67-76
            
            double neg_res_126578 = -defunc_0_reduce_res_129725;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140021 = 0; i_140021 < (int64_t) 16; i_140021++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_126585 = ((double *) mem_141278)[i_140033 * (int64_t) 16 + i_140021];
                
                // futhark/microgpt.fut:210:44-76
                
                double zp_res_126586 = neg_res_126578 + zp_lhs_126585;
                
                // futhark/microgpt.fut:210:37-76
                
                double exp_res_126587 = futrts_exp64(zp_res_126586);
                
                ((double *) mem_141299)[i_140021] = exp_res_126587;
            }
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126589;
            double r_126591 = 0.0;
            
            for (int64_t i_126590 = 0; i_126590 < (int64_t) 16; i_126590++) {
                // futhark/microgpt.fut:211:36-46
                
                double lifted_lambda_res_126592 = ((double *) mem_141299)[i_126590];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126593 = r_126591 + lifted_lambda_res_126592;
                double r_tmp_143177 = zp_res_126593;
                
                r_126591 = r_tmp_143177;
            }
            defunc_0_lifted_lambda_res_126589 = r_126591;
            // futhark/microgpt.fut:212:53-64
            
            double zs_res_126594 = 1.0 / defunc_0_lifted_lambda_res_126589;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140025 = 0; i_140025 < (int64_t) 16; i_140025++) {
                // futhark/microgpt.fut:212:37-47
                
                double zt_lhs_126601 = ((double *) mem_141299)[i_140025];
                
                // futhark/microgpt.fut:212:37-64
                
                double zt_res_126602 = zs_res_126594 * zt_lhs_126601;
                
                ((double *) mem_141306)[i_140025] = zt_res_126602;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140029 = 0; i_140029 < (int64_t) 16; i_140029++) {
                // futhark/microgpt.fut:213:4-14
                
                double lifted_lambda_res_126610 = ((double *) mem_141306)[i_140029];
                
                ((double *) mem_141313)[i_140029] = lifted_lambda_res_126610;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141294, i_140033 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141313, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140041 = 0; i_140041 < (int64_t) 16; i_140041++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140037 = 0; i_140037 < (int64_t) 4; i_140037++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_126625;
                double r_126627 = 0.0;
                
                for (int64_t i_126626 = 0; i_126626 < (int64_t) 16; i_126626++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_126628 = ((double *) mem_141294)[i_140041 * (int64_t) 16 + i_126626];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_126629 = ((double *) mem_141175)[i_140053 * (int64_t) 64 + i_126626 * (int64_t) 4 + i_140037];
                    
                    // futhark/microgpt.fut:214:66-111
                    
                    double zt_res_126630 = zt_lhs_126628 * zt_rhs_126629;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_126631 = r_126627 + zt_res_126630;
                    double r_tmp_143182 = zp_res_126631;
                    
                    r_126627 = r_tmp_143182;
                }
                defunc_0_lifted_lambda_res_126625 = r_126627;
                ((double *) mem_141329)[i_140037] = defunc_0_lifted_lambda_res_126625;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141324, i_140041 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141329, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140049 = 0; i_140049 < (int64_t) 16; i_140049++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140045 = 0; i_140045 < (int64_t) 4; i_140045++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_126646 = ((double *) mem_141324)[i_140049 * (int64_t) 4 + i_140045];
                
                ((double *) mem_141345)[i_140045] = lifted_lambda_res_126646;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141340, i_140049 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141345, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141256, i_140053 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_141340, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141361_cached_sizze_143512 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141361, &mem_141361_cached_sizze_143512, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141366_cached_sizze_143513 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141366, &mem_141366_cached_sizze_143513, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140061 = 0; i_140061 < (int64_t) 16; i_140061++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140057 = 0; i_140057 < (int64_t) 16; i_140057++) {
            // futhark/microgpt.fut:216:54-57
            
            int64_t tmp_126658 = sdiv64(i_140057, (int64_t) 4);
            
            // futhark/microgpt.fut:216:44-59
            
            bool x_126659 = sle64((int64_t) 0, tmp_126658);
            
            // futhark/microgpt.fut:216:44-59
            
            bool y_126660 = slt64(tmp_126658, (int64_t) 4);
            
            // futhark/microgpt.fut:216:44-59
            
            bool bounds_check_126661 = x_126659 && y_126660;
            
            // futhark/microgpt.fut:216:44-59
            
            bool index_certs_126662;
            
            if (!bounds_check_126661) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_126658, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:216:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:216:15-80\n   #6  futhark/microgpt.fut:442:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:216:74-77
            
            int64_t tmp_126663 = smod64(i_140057, (int64_t) 4);
            
            // futhark/microgpt.fut:216:44-79
            
            bool x_126664 = sle64((int64_t) 0, tmp_126663);
            
            // futhark/microgpt.fut:216:44-79
            
            bool y_126665 = slt64(tmp_126663, (int64_t) 4);
            
            // futhark/microgpt.fut:216:44-79
            
            bool bounds_check_126666 = x_126664 && y_126665;
            
            // futhark/microgpt.fut:216:44-79
            
            bool index_certs_126667;
            
            if (!bounds_check_126666) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_126663, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:216:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:216:15-80\n   #6  futhark/microgpt.fut:442:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_126668 = ((double *) mem_141256)[tmp_126658 * (int64_t) 64 + i_140061 * (int64_t) 4 + tmp_126663];
            
            ((double *) mem_141366)[i_140057] = lifted_lambda_res_126668;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141361, i_140061 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141366, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141377_cached_sizze_143514 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141377, &mem_141377_cached_sizze_143514, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141382_cached_sizze_143515 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141382, &mem_141382_cached_sizze_143515, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140069 = 0; i_140069 < (int64_t) 16; i_140069++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140065 = 0; i_140065 < (int64_t) 16; i_140065++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126683;
            double r_126685 = 0.0;
            
            for (int64_t i_126684 = 0; i_126684 < (int64_t) 16; i_126684++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_126686 = ((double *) wout_mem_141055.mem)[i_140065 * (int64_t) 16 + i_126684];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_126687 = ((double *) mem_141361)[i_140069 * (int64_t) 16 + i_126684];
                
                // futhark/microgpt.fut:217:67-106
                
                double zt_res_126688 = zt_lhs_126686 * zt_rhs_126687;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126689 = r_126685 + zt_res_126688;
                double r_tmp_143189 = zp_res_126689;
                
                r_126685 = r_tmp_143189;
            }
            defunc_0_lifted_lambda_res_126683 = r_126685;
            ((double *) mem_141382)[i_140065] = defunc_0_lifted_lambda_res_126683;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141377, i_140069 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141382, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141393_cached_sizze_143516 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141393, &mem_141393_cached_sizze_143516, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141398_cached_sizze_143517 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141398, &mem_141398_cached_sizze_143517, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140077 = 0; i_140077 < (int64_t) 16; i_140077++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140073 = 0; i_140073 < (int64_t) 16; i_140073++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_126704 = ((double *) mem_141377)[i_140077 * (int64_t) 16 + i_140073];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_126705 = ((double *) mem_141081)[i_140077 * (int64_t) 16 + i_140073];
            
            // futhark/microgpt.fut:218:46-84
            
            double zp_res_126706 = zp_lhs_126704 + zp_rhs_126705;
            
            ((double *) mem_141398)[i_140073] = zp_res_126706;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141393, i_140077 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141398, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141409_cached_sizze_143518 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141409, &mem_141409_cached_sizze_143518, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141414_cached_sizze_143519 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141414, &mem_141414_cached_sizze_143519, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141421_cached_sizze_143520 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141421, &mem_141421_cached_sizze_143520, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140089 = 0; i_140089 < (int64_t) 16; i_140089++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_126715;
        double r_126717 = 0.0;
        
        for (int64_t i_126716 = 0; i_126716 < (int64_t) 16; i_126716++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_126718 = ((double *) mem_141393)[i_140089 * (int64_t) 16 + i_126716];
            
            // futhark/microgpt.fut:219:79-118
            
            double zt_res_126719 = zt_lhs_126718 * zt_lhs_126718;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_126720 = r_126717 + zt_res_126719;
            double r_tmp_143193 = zp_res_126720;
            
            r_126717 = r_tmp_143193;
        }
        defunc_0_lifted_lambda_res_126715 = r_126717;
        // futhark/microgpt.fut:219:58-136
        
        double zs_res_126721 = defunc_0_lifted_lambda_res_126715 / 16.0;
        
        // futhark/microgpt.fut:220:24-55
        
        double zp_res_126722 = 1.0e-5 + zs_res_126721;
        
        // futhark/microgpt.fut:220:16-55
        
        double sqrt_res_126723 = futrts_sqrt64(zp_res_126722);
        
        // futhark/microgpt.fut:221:60-71
        
        double zs_res_126724 = 1.0 / sqrt_res_126723;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140081 = 0; i_140081 < (int64_t) 16; i_140081++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_126731 = ((double *) mem_141393)[i_140089 * (int64_t) 16 + i_140081];
            
            // futhark/microgpt.fut:221:37-71
            
            double zt_res_126732 = zs_res_126724 * zt_lhs_126731;
            
            ((double *) mem_141414)[i_140081] = zt_res_126732;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140085 = 0; i_140085 < (int64_t) 16; i_140085++) {
            // futhark/microgpt.fut:222:4-14
            
            double lifted_lambda_res_126740 = ((double *) mem_141414)[i_140085];
            
            ((double *) mem_141421)[i_140085] = lifted_lambda_res_126740;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141409, i_140089 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141421, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141432_cached_sizze_143521 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141432, &mem_141432_cached_sizze_143521, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141437_cached_sizze_143522 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141437, &mem_141437_cached_sizze_143522, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140097 = 0; i_140097 < (int64_t) 16; i_140097++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140093 = 0; i_140093 < (int64_t) 64; i_140093++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126756;
            double r_126758 = 0.0;
            
            for (int64_t i_126757 = 0; i_126757 < (int64_t) 16; i_126757++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_126759 = ((double *) wup_mem_141059.mem)[i_140093 * (int64_t) 16 + i_126757];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_126760 = ((double *) mem_141409)[i_140097 * (int64_t) 16 + i_126757];
                
                // futhark/microgpt.fut:223:67-106
                
                double zt_res_126761 = zt_lhs_126759 * zt_rhs_126760;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126762 = r_126758 + zt_res_126761;
                double r_tmp_143198 = zp_res_126762;
                
                r_126758 = r_tmp_143198;
            }
            defunc_0_lifted_lambda_res_126756 = r_126758;
            ((double *) mem_141437)[i_140093] = defunc_0_lifted_lambda_res_126756;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141432, i_140097 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141437, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141448_cached_sizze_143523 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141448, &mem_141448_cached_sizze_143523, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141453_cached_sizze_143524 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141453, &mem_141453_cached_sizze_143524, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140105 = 0; i_140105 < (int64_t) 16; i_140105++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140101 = 0; i_140101 < (int64_t) 64; i_140101++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_126777 = ((double *) mem_141432)[i_140105 * (int64_t) 64 + i_140101];
            
            // futhark/microgpt.fut:224:45-73
            
            double max_res_126778 = fmax64(0.0, max_arg0_126777);
            
            ((double *) mem_141453)[i_140101] = max_res_126778;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141448, i_140105 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141453, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141464_cached_sizze_143525 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141464, &mem_141464_cached_sizze_143525, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141469_cached_sizze_143526 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141469, &mem_141469_cached_sizze_143526, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140113 = 0; i_140113 < (int64_t) 16; i_140113++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140109 = 0; i_140109 < (int64_t) 16; i_140109++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126793;
            double r_126795 = 0.0;
            
            for (int64_t i_126794 = 0; i_126794 < (int64_t) 64; i_126794++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_126796 = ((double *) wdown_mem_141053.mem)[i_140109 * (int64_t) 64 + i_126794];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_126797 = ((double *) mem_141448)[i_140113 * (int64_t) 64 + i_126794];
                
                // futhark/microgpt.fut:225:67-108
                
                double zt_res_126798 = zt_lhs_126796 * zt_rhs_126797;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126799 = r_126795 + zt_res_126798;
                double r_tmp_143203 = zp_res_126799;
                
                r_126795 = r_tmp_143203;
            }
            defunc_0_lifted_lambda_res_126793 = r_126795;
            ((double *) mem_141469)[i_140109] = defunc_0_lifted_lambda_res_126793;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141464, i_140113 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141469, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141480_cached_sizze_143527 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141480, &mem_141480_cached_sizze_143527, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141485_cached_sizze_143528 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141485, &mem_141485_cached_sizze_143528, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140121 = 0; i_140121 < (int64_t) 16; i_140121++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140117 = 0; i_140117 < (int64_t) 16; i_140117++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_126814 = ((double *) mem_141464)[i_140121 * (int64_t) 16 + i_140117];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_126815 = ((double *) mem_141393)[i_140121 * (int64_t) 16 + i_140117];
            
            // futhark/microgpt.fut:226:46-85
            
            double zp_res_126816 = zp_lhs_126814 + zp_rhs_126815;
            
            ((double *) mem_141485)[i_140117] = zp_res_126816;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141480, i_140121 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141485, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141496_cached_sizze_143529 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_141496, &mem_141496_cached_sizze_143529, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141501_cached_sizze_143530 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_141501, &mem_141501_cached_sizze_143530, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140129 = 0; i_140129 < (int64_t) 16; i_140129++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140125 = 0; i_140125 < (int64_t) 27; i_140125++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126832;
            double r_126834 = 0.0;
            
            for (int64_t i_126833 = 0; i_126833 < (int64_t) 16; i_126833++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_126835 = ((double *) wvoc_mem_141061.mem)[i_140125 * (int64_t) 16 + i_126833];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_126836 = ((double *) mem_141480)[i_140129 * (int64_t) 16 + i_126833];
                
                // futhark/microgpt.fut:227:67-107
                
                double zt_res_126837 = zt_lhs_126835 * zt_rhs_126836;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126838 = r_126834 + zt_res_126837;
                double r_tmp_143208 = zp_res_126838;
                
                r_126834 = r_tmp_143208;
            }
            defunc_0_lifted_lambda_res_126832 = r_126834;
            ((double *) mem_141501)[i_140125] = defunc_0_lifted_lambda_res_126832;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141496, i_140129 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141501, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_141512, (int64_t) 128, "mem_141512")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141516_cached_sizze_143531 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_141516, &mem_141516_cached_sizze_143531, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141523_cached_sizze_143532 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_141523, &mem_141523_cached_sizze_143532, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140143 = 0; i_140143 < (int64_t) 16; i_140143++) {
        double x_129748;
        double redout_140131 = -INFINITY;
        
        for (int64_t i_140132 = 0; i_140132 < (int64_t) 27; i_140132++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_129695 = ((double *) mem_141496)[i_140143 * (int64_t) 27 + i_140132];
            
            // futhark/microgpt.fut:105:13-33
            
            double max_res_126862 = fmax64(lifted_lambda_res_129695, redout_140131);
            double redout_tmp_143210 = max_res_126862;
            
            redout_140131 = redout_tmp_143210;
        }
        x_129748 = redout_140131;
        // futhark/microgpt.fut:229:67-76
        
        double neg_res_126863 = -x_129748;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_126847;
        double r_126849 = 0.0;
        
        for (int64_t i_126848 = 0; i_126848 < (int64_t) 27; i_126848++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140135 = 0; i_140135 < (int64_t) 27; i_140135++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_126870 = ((double *) mem_141496)[i_140143 * (int64_t) 27 + i_140135];
                
                // futhark/microgpt.fut:229:44-76
                
                double zp_res_126871 = neg_res_126863 + zp_lhs_126870;
                
                // futhark/microgpt.fut:229:37-76
                
                double exp_res_126872 = futrts_exp64(zp_res_126871);
                
                ((double *) mem_141516)[i_140135] = exp_res_126872;
            }
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126874;
            double r_126876 = 0.0;
            
            for (int64_t i_126875 = 0; i_126875 < (int64_t) 27; i_126875++) {
                // futhark/microgpt.fut:230:36-46
                
                double lifted_lambda_res_126877 = ((double *) mem_141516)[i_126875];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126878 = r_126876 + lifted_lambda_res_126877;
                double r_tmp_143213 = zp_res_126878;
                
                r_126876 = r_tmp_143213;
            }
            defunc_0_lifted_lambda_res_126874 = r_126876;
            // futhark/microgpt.fut:231:53-64
            
            double zs_res_126879 = 1.0 / defunc_0_lifted_lambda_res_126874;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140139 = 0; i_140139 < (int64_t) 27; i_140139++) {
                // futhark/microgpt.fut:231:37-47
                
                double zt_lhs_126886 = ((double *) mem_141516)[i_140139];
                
                // futhark/microgpt.fut:231:37-64
                
                double zt_res_126887 = zs_res_126879 * zt_lhs_126886;
                
                ((double *) mem_141523)[i_140139] = zt_res_126887;
            }
            // futhark/microgpt.fut:232:12-22
            
            double log_arg0_126889 = ((double *) mem_141523)[i_126848];
            
            // futhark/microgpt.fut:232:6-22
            
            double log_res_126890 = futrts_log64(log_arg0_126889);
            
            // futhark/microgpt.fut:61:46-49
            
            double zt_rhs_126891 = ((double *) target_mem_141063.mem)[i_140143 * (int64_t) 27 + i_126848];
            
            // futhark/microgpt.fut:232:6-48
            
            double zt_res_126892 = log_res_126890 * zt_rhs_126891;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_126893 = r_126849 + zt_res_126892;
            double r_tmp_143211 = zp_res_126893;
            
            r_126849 = r_tmp_143211;
        }
        defunc_0_lifted_lambda_res_126847 = r_126849;
        // futhark/microgpt.fut:228:37-232:54
        
        double neg_res_126894 = -defunc_0_lifted_lambda_res_126847;
        
        ((double *) mem_141512.mem)[i_140143] = neg_res_126894;
    }
    // futhark/microgpt.fut:61:13-49
    
    double defunc_0_lifted_lambda_res_126896;
    double r_126898 = 0.0;
    
    for (int64_t i_126897 = 0; i_126897 < (int64_t) 16; i_126897++) {
        // futhark/microgpt.fut:233:37-47
        
        double lifted_lambda_res_126899 = ((double *) mem_141512.mem)[i_126897];
        
        // futhark/microgpt.fut:61:40-49
        
        double zp_res_126900 = r_126898 + lifted_lambda_res_126899;
        double r_tmp_143215 = zp_res_126900;
        
        r_126898 = r_tmp_143215;
    }
    defunc_0_lifted_lambda_res_126896 = r_126898;
    // futhark/microgpt.fut:233:17-64
    
    double zs_res_126901 = defunc_0_lifted_lambda_res_126896 / 16.0;
    
    if (memblock_set(ctx, &mem_out_143138, &mem_141512, "mem_141512") != 0)
        return 1;
    prim_out_143139 = zs_res_126901;
    if (memblock_set(ctx, &*mem_out_p_143474, &mem_out_143138, "mem_out_143138") != 0)
        return 1;
    *out_prim_out_143475 = prim_out_143139;
    
  cleanup:
    {
        free(mem_141065);
        free(mem_141070);
        free(mem_141081);
        free(mem_141086);
        free(mem_141093);
        free(mem_141104);
        free(mem_141109);
        free(mem_141116);
        free(mem_141127);
        free(mem_141128);
        free(mem_141129);
        free(mem_141142);
        free(mem_141143);
        free(mem_141144);
        free(mem_141175);
        free(mem_141176);
        free(mem_141177);
        free(mem_141193);
        free(mem_141194);
        free(mem_141195);
        free(mem_141208);
        free(mem_141209);
        free(mem_141210);
        free(mem_141256);
        free(mem_141262);
        free(mem_141267);
        free(mem_141278);
        free(mem_141283);
        free(mem_141294);
        free(mem_141299);
        free(mem_141306);
        free(mem_141313);
        free(mem_141324);
        free(mem_141329);
        free(mem_141340);
        free(mem_141345);
        free(mem_141361);
        free(mem_141366);
        free(mem_141377);
        free(mem_141382);
        free(mem_141393);
        free(mem_141398);
        free(mem_141409);
        free(mem_141414);
        free(mem_141421);
        free(mem_141432);
        free(mem_141437);
        free(mem_141448);
        free(mem_141453);
        free(mem_141464);
        free(mem_141469);
        free(mem_141480);
        free(mem_141485);
        free(mem_141496);
        free(mem_141501);
        free(mem_141516);
        free(mem_141523);
        if (memblock_unref(ctx, &mem_141512, "mem_141512") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143138, "mem_out_143138") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_143533, struct memblock wdown_mem_141053, struct memblock wkey_mem_141054, struct memblock wout_mem_141055, struct memblock wpe_mem_141056, struct memblock wqry_mem_141057, struct memblock wte_mem_141058, struct memblock wup_mem_141059, struct memblock wval_mem_141060, struct memblock wvoc_mem_141061, struct memblock tokens_mem_141062, struct memblock mask_mem_141063)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_141064_cached_sizze_143534 = 0;
    unsigned char *mem_141064 = NULL;
    int64_t mem_141069_cached_sizze_143535 = 0;
    unsigned char *mem_141069 = NULL;
    int64_t mem_141080_cached_sizze_143536 = 0;
    unsigned char *mem_141080 = NULL;
    int64_t mem_141085_cached_sizze_143537 = 0;
    unsigned char *mem_141085 = NULL;
    int64_t mem_141092_cached_sizze_143538 = 0;
    unsigned char *mem_141092 = NULL;
    int64_t mem_141103_cached_sizze_143539 = 0;
    unsigned char *mem_141103 = NULL;
    int64_t mem_141108_cached_sizze_143540 = 0;
    unsigned char *mem_141108 = NULL;
    int64_t mem_141115_cached_sizze_143541 = 0;
    unsigned char *mem_141115 = NULL;
    int64_t mem_141126_cached_sizze_143542 = 0;
    unsigned char *mem_141126 = NULL;
    int64_t mem_141127_cached_sizze_143543 = 0;
    unsigned char *mem_141127 = NULL;
    int64_t mem_141128_cached_sizze_143544 = 0;
    unsigned char *mem_141128 = NULL;
    int64_t mem_141141_cached_sizze_143545 = 0;
    unsigned char *mem_141141 = NULL;
    int64_t mem_141142_cached_sizze_143546 = 0;
    unsigned char *mem_141142 = NULL;
    int64_t mem_141143_cached_sizze_143547 = 0;
    unsigned char *mem_141143 = NULL;
    int64_t mem_141174_cached_sizze_143548 = 0;
    unsigned char *mem_141174 = NULL;
    int64_t mem_141175_cached_sizze_143549 = 0;
    unsigned char *mem_141175 = NULL;
    int64_t mem_141176_cached_sizze_143550 = 0;
    unsigned char *mem_141176 = NULL;
    int64_t mem_141192_cached_sizze_143551 = 0;
    unsigned char *mem_141192 = NULL;
    int64_t mem_141193_cached_sizze_143552 = 0;
    unsigned char *mem_141193 = NULL;
    int64_t mem_141194_cached_sizze_143553 = 0;
    unsigned char *mem_141194 = NULL;
    int64_t mem_141207_cached_sizze_143554 = 0;
    unsigned char *mem_141207 = NULL;
    int64_t mem_141208_cached_sizze_143555 = 0;
    unsigned char *mem_141208 = NULL;
    int64_t mem_141209_cached_sizze_143556 = 0;
    unsigned char *mem_141209 = NULL;
    int64_t mem_141255_cached_sizze_143557 = 0;
    unsigned char *mem_141255 = NULL;
    int64_t mem_141261_cached_sizze_143558 = 0;
    unsigned char *mem_141261 = NULL;
    int64_t mem_141266_cached_sizze_143559 = 0;
    unsigned char *mem_141266 = NULL;
    int64_t mem_141277_cached_sizze_143560 = 0;
    unsigned char *mem_141277 = NULL;
    int64_t mem_141282_cached_sizze_143561 = 0;
    unsigned char *mem_141282 = NULL;
    int64_t mem_141293_cached_sizze_143562 = 0;
    unsigned char *mem_141293 = NULL;
    int64_t mem_141298_cached_sizze_143563 = 0;
    unsigned char *mem_141298 = NULL;
    int64_t mem_141305_cached_sizze_143564 = 0;
    unsigned char *mem_141305 = NULL;
    int64_t mem_141312_cached_sizze_143565 = 0;
    unsigned char *mem_141312 = NULL;
    int64_t mem_141323_cached_sizze_143566 = 0;
    unsigned char *mem_141323 = NULL;
    int64_t mem_141328_cached_sizze_143567 = 0;
    unsigned char *mem_141328 = NULL;
    int64_t mem_141339_cached_sizze_143568 = 0;
    unsigned char *mem_141339 = NULL;
    int64_t mem_141344_cached_sizze_143569 = 0;
    unsigned char *mem_141344 = NULL;
    int64_t mem_141360_cached_sizze_143570 = 0;
    unsigned char *mem_141360 = NULL;
    int64_t mem_141365_cached_sizze_143571 = 0;
    unsigned char *mem_141365 = NULL;
    int64_t mem_141376_cached_sizze_143572 = 0;
    unsigned char *mem_141376 = NULL;
    int64_t mem_141381_cached_sizze_143573 = 0;
    unsigned char *mem_141381 = NULL;
    int64_t mem_141392_cached_sizze_143574 = 0;
    unsigned char *mem_141392 = NULL;
    int64_t mem_141397_cached_sizze_143575 = 0;
    unsigned char *mem_141397 = NULL;
    int64_t mem_141408_cached_sizze_143576 = 0;
    unsigned char *mem_141408 = NULL;
    int64_t mem_141413_cached_sizze_143577 = 0;
    unsigned char *mem_141413 = NULL;
    int64_t mem_141420_cached_sizze_143578 = 0;
    unsigned char *mem_141420 = NULL;
    int64_t mem_141431_cached_sizze_143579 = 0;
    unsigned char *mem_141431 = NULL;
    int64_t mem_141436_cached_sizze_143580 = 0;
    unsigned char *mem_141436 = NULL;
    int64_t mem_141447_cached_sizze_143581 = 0;
    unsigned char *mem_141447 = NULL;
    int64_t mem_141452_cached_sizze_143582 = 0;
    unsigned char *mem_141452 = NULL;
    int64_t mem_141463_cached_sizze_143583 = 0;
    unsigned char *mem_141463 = NULL;
    int64_t mem_141468_cached_sizze_143584 = 0;
    unsigned char *mem_141468 = NULL;
    int64_t mem_141479_cached_sizze_143585 = 0;
    unsigned char *mem_141479 = NULL;
    int64_t mem_141484_cached_sizze_143586 = 0;
    unsigned char *mem_141484 = NULL;
    int64_t mem_141495_cached_sizze_143587 = 0;
    unsigned char *mem_141495 = NULL;
    int64_t mem_141500_cached_sizze_143588 = 0;
    unsigned char *mem_141500 = NULL;
    int64_t mem_141516_cached_sizze_143589 = 0;
    unsigned char *mem_141516 = NULL;
    struct memblock mem_141511;
    
    mem_141511.references = NULL;
    
    struct memblock mem_out_143138;
    
    mem_out_143138.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (mem_141064_cached_sizze_143534 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141064, &mem_141064_cached_sizze_143534, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141069_cached_sizze_143535 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141069, &mem_141069_cached_sizze_143535, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139925 = 0; i_139925 < (int64_t) 16; i_139925++) {
        // futhark/microgpt.fut:436:41-50
        
        int64_t tmp_126286 = ((int64_t *) tokens_mem_141062.mem)[i_139925];
        
        // futhark/microgpt.fut:436:37-51
        
        bool x_126287 = sle64((int64_t) 0, tmp_126286);
        
        // futhark/microgpt.fut:436:37-51
        
        bool y_126288 = slt64(tmp_126286, (int64_t) 27);
        
        // futhark/microgpt.fut:436:37-51
        
        bool bounds_check_126289 = x_126287 && y_126288;
        
        // futhark/microgpt.fut:436:37-51
        
        bool index_certs_126290;
        
        if (!bounds_check_126289) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_126286, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:436:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:436:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139921 = 0; i_139921 < (int64_t) 16; i_139921++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_126297 = ((double *) wte_mem_141058.mem)[tmp_126286 * (int64_t) 16 + i_139921];
            
            ((double *) mem_141069)[i_139921] = lifted_lambda_res_126297;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141064, i_139925 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141069, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141080_cached_sizze_143536 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141080, &mem_141080_cached_sizze_143536, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141085_cached_sizze_143537 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141085, &mem_141085_cached_sizze_143537, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141092_cached_sizze_143538 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141092, &mem_141092_cached_sizze_143538, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139937 = 0; i_139937 < (int64_t) 16; i_139937++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_126323;
        double r_126325 = 0.0;
        
        for (int64_t i_126324 = 0; i_126324 < (int64_t) 16; i_126324++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_126326 = ((double *) wpe_mem_141056.mem)[i_139937 * (int64_t) 16 + i_126324];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_126327 = ((double *) mem_141064)[i_139937 * (int64_t) 16 + i_126324];
            
            // futhark/microgpt.fut:138:76-116
            
            double zp_res_126328 = zp_lhs_126326 + zp_rhs_126327;
            
            // futhark/microgpt.fut:138:94-163
            
            double zt_res_126329 = zp_res_126328 * zp_res_126328;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_126330 = r_126325 + zt_res_126329;
            double r_tmp_143142 = zp_res_126330;
            
            r_126325 = r_tmp_143142;
        }
        defunc_0_lifted_lambda_res_126323 = r_126325;
        // futhark/microgpt.fut:138:54-182
        
        double zs_res_126331 = defunc_0_lifted_lambda_res_126323 / 16.0;
        
        // futhark/microgpt.fut:139:24-55
        
        double zp_res_126332 = 1.0e-5 + zs_res_126331;
        
        // futhark/microgpt.fut:139:16-55
        
        double sqrt_res_126333 = futrts_sqrt64(zp_res_126332);
        
        // futhark/microgpt.fut:140:85-96
        
        double zs_res_126334 = 1.0 / sqrt_res_126333;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139929 = 0; i_139929 < (int64_t) 16; i_139929++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_126341 = ((double *) wpe_mem_141056.mem)[i_139937 * (int64_t) 16 + i_139929];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_126342 = ((double *) mem_141064)[i_139937 * (int64_t) 16 + i_139929];
            
            // futhark/microgpt.fut:140:38-78
            
            double zp_res_126343 = zp_lhs_126341 + zp_rhs_126342;
            
            // futhark/microgpt.fut:140:56-96
            
            double zt_res_126344 = zs_res_126334 * zp_res_126343;
            
            ((double *) mem_141085)[i_139929] = zt_res_126344;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139933 = 0; i_139933 < (int64_t) 16; i_139933++) {
            // futhark/microgpt.fut:141:4-14
            
            double lifted_lambda_res_126352 = ((double *) mem_141085)[i_139933];
            
            ((double *) mem_141092)[i_139933] = lifted_lambda_res_126352;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141080, i_139937 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141092, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141103_cached_sizze_143539 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141103, &mem_141103_cached_sizze_143539, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141108_cached_sizze_143540 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141108, &mem_141108_cached_sizze_143540, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141115_cached_sizze_143541 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141115, &mem_141115_cached_sizze_143541, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139949 = 0; i_139949 < (int64_t) 16; i_139949++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_126361;
        double r_126363 = 0.0;
        
        for (int64_t i_126362 = 0; i_126362 < (int64_t) 16; i_126362++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_126364 = ((double *) mem_141080)[i_139949 * (int64_t) 16 + i_126362];
            
            // futhark/microgpt.fut:142:78-115
            
            double zt_res_126365 = zt_lhs_126364 * zt_lhs_126364;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_126366 = r_126363 + zt_res_126365;
            double r_tmp_143146 = zp_res_126366;
            
            r_126363 = r_tmp_143146;
        }
        defunc_0_lifted_lambda_res_126361 = r_126363;
        // futhark/microgpt.fut:142:57-133
        
        double zs_res_126367 = defunc_0_lifted_lambda_res_126361 / 16.0;
        
        // futhark/microgpt.fut:143:24-55
        
        double zp_res_126368 = 1.0e-5 + zs_res_126367;
        
        // futhark/microgpt.fut:143:16-55
        
        double sqrt_res_126369 = futrts_sqrt64(zp_res_126368);
        
        // futhark/microgpt.fut:144:59-70
        
        double zs_res_126370 = 1.0 / sqrt_res_126369;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139941 = 0; i_139941 < (int64_t) 16; i_139941++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_126377 = ((double *) mem_141080)[i_139949 * (int64_t) 16 + i_139941];
            
            // futhark/microgpt.fut:144:37-70
            
            double zt_res_126378 = zs_res_126370 * zt_lhs_126377;
            
            ((double *) mem_141108)[i_139941] = zt_res_126378;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139945 = 0; i_139945 < (int64_t) 16; i_139945++) {
            // futhark/microgpt.fut:145:4-14
            
            double lifted_lambda_res_126386 = ((double *) mem_141108)[i_139945];
            
            ((double *) mem_141115)[i_139945] = lifted_lambda_res_126386;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141103, i_139949 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141115, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141126_cached_sizze_143542 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141126, &mem_141126_cached_sizze_143542, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141127_cached_sizze_143543 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141127, &mem_141127_cached_sizze_143543, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141128_cached_sizze_143544 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141128, &mem_141128_cached_sizze_143544, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141141_cached_sizze_143545 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141141, &mem_141141_cached_sizze_143545, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141142_cached_sizze_143546 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141142, &mem_141142_cached_sizze_143546, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141143_cached_sizze_143547 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141143, &mem_141143_cached_sizze_143547, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139967 = 0; i_139967 < (int64_t) 16; i_139967++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139957 = 0; i_139957 < (int64_t) 16; i_139957++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129405;
            double r_129407 = 0.0;
            
            for (int64_t i_129406 = 0; i_129406 < (int64_t) 16; i_129406++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_129408 = ((double *) wqry_mem_141057.mem)[i_139957 * (int64_t) 16 + i_129406];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_129409 = ((double *) mem_141103)[i_139967 * (int64_t) 16 + i_129406];
                
                // futhark/microgpt.fut:146:66-105
                
                double zt_res_129410 = zt_lhs_129408 * zt_rhs_129409;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129411 = r_129407 + zt_res_129410;
                double r_tmp_143155 = zp_res_129411;
                
                r_129407 = r_tmp_143155;
            }
            defunc_0_lifted_lambda_res_129405 = r_129407;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129418;
            double r_129420 = 0.0;
            
            for (int64_t i_129419 = 0; i_129419 < (int64_t) 16; i_129419++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_129421 = ((double *) wkey_mem_141054.mem)[i_139957 * (int64_t) 16 + i_129419];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_129422 = ((double *) mem_141103)[i_139967 * (int64_t) 16 + i_129419];
                
                // futhark/microgpt.fut:147:66-105
                
                double zt_res_129423 = zt_lhs_129421 * zt_rhs_129422;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129424 = r_129420 + zt_res_129423;
                double r_tmp_143156 = zp_res_129424;
                
                r_129420 = r_tmp_143156;
            }
            defunc_0_lifted_lambda_res_129418 = r_129420;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_129434;
            double r_129436 = 0.0;
            
            for (int64_t i_129435 = 0; i_129435 < (int64_t) 16; i_129435++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_129437 = ((double *) wval_mem_141060.mem)[i_139957 * (int64_t) 16 + i_129435];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_129438 = ((double *) mem_141103)[i_139967 * (int64_t) 16 + i_129435];
                
                // futhark/microgpt.fut:148:66-105
                
                double zt_res_129439 = zt_lhs_129437 * zt_rhs_129438;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_129440 = r_129436 + zt_res_129439;
                double r_tmp_143157 = zp_res_129440;
                
                r_129436 = r_tmp_143157;
            }
            defunc_0_lifted_lambda_res_129434 = r_129436;
            ((double *) mem_141141)[i_139957] = defunc_0_lifted_lambda_res_129434;
            ((double *) mem_141142)[i_139957] = defunc_0_lifted_lambda_res_129418;
            ((double *) mem_141143)[i_139957] = defunc_0_lifted_lambda_res_129405;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141126, i_139967 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141141, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141127, i_139967 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141142, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141128, i_139967 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141143, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141174_cached_sizze_143548 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141174, &mem_141174_cached_sizze_143548, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141175_cached_sizze_143549 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141175, &mem_141175_cached_sizze_143549, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141176_cached_sizze_143550 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141176, &mem_141176_cached_sizze_143550, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141192_cached_sizze_143551 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141192, &mem_141192_cached_sizze_143551, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141193_cached_sizze_143552 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141193, &mem_141193_cached_sizze_143552, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141194_cached_sizze_143553 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141194, &mem_141194_cached_sizze_143553, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141207_cached_sizze_143554 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_141207, &mem_141207_cached_sizze_143554, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141208_cached_sizze_143555 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_141208, &mem_141208_cached_sizze_143555, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141209_cached_sizze_143556 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_141209, &mem_141209_cached_sizze_143556, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139997 = 0; i_139997 < (int64_t) 4; i_139997++) {
        // futhark/microgpt.fut:149:69-72
        
        int64_t zp_lhs_129281 = mul64((int64_t) 4, i_139997);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139987 = 0; i_139987 < (int64_t) 16; i_139987++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_139977 = 0; i_139977 < (int64_t) 4; i_139977++) {
                // futhark/microgpt.fut:149:74-81
                
                int64_t tmp_129598 = add64(zp_lhs_129281, i_139977);
                
                // futhark/microgpt.fut:149:51-83
                
                bool x_129599 = sle64((int64_t) 0, tmp_129598);
                
                // futhark/microgpt.fut:149:51-83
                
                bool y_129600 = slt64(tmp_129598, (int64_t) 16);
                
                // futhark/microgpt.fut:149:51-83
                
                bool bounds_check_129601 = x_129599 && y_129600;
                
                // futhark/microgpt.fut:149:51-83
                
                bool index_certs_129602;
                
                if (!bounds_check_129601) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_129598, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:149:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:149:15-84\n   #9  futhark/microgpt.fut:437:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129603 = ((double *) mem_141128)[i_139987 * (int64_t) 16 + tmp_129598];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129611 = ((double *) mem_141127)[i_139987 * (int64_t) 16 + tmp_129598];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129622 = ((double *) mem_141126)[i_139987 * (int64_t) 16 + tmp_129598];
                
                ((double *) mem_141207)[i_139977] = lifted_lambda_res_129622;
                ((double *) mem_141208)[i_139977] = lifted_lambda_res_129611;
                ((double *) mem_141209)[i_139977] = lifted_lambda_res_129603;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141192, i_139987 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141207, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141193, i_139987 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141208, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141194, i_139987 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141209, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141174, i_139997 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_141192, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141175, i_139997 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_141193, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141176, i_139997 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_141194, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141255_cached_sizze_143557 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141255, &mem_141255_cached_sizze_143557, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141261_cached_sizze_143558 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141261, &mem_141261_cached_sizze_143558, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141266_cached_sizze_143559 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141266, &mem_141266_cached_sizze_143559, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141277_cached_sizze_143560 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141277, &mem_141277_cached_sizze_143560, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141282_cached_sizze_143561 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141282, &mem_141282_cached_sizze_143561, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141293_cached_sizze_143562 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141293, &mem_141293_cached_sizze_143562, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141298_cached_sizze_143563 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141298, &mem_141298_cached_sizze_143563, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141305_cached_sizze_143564 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141305, &mem_141305_cached_sizze_143564, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141312_cached_sizze_143565 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141312, &mem_141312_cached_sizze_143565, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141323_cached_sizze_143566 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141323, &mem_141323_cached_sizze_143566, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141328_cached_sizze_143567 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_141328, &mem_141328_cached_sizze_143567, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141339_cached_sizze_143568 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141339, &mem_141339_cached_sizze_143568, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141344_cached_sizze_143569 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_141344, &mem_141344_cached_sizze_143569, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140053 = 0; i_140053 < (int64_t) 4; i_140053++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140007 = 0; i_140007 < (int64_t) 16; i_140007++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140003 = 0; i_140003 < (int64_t) 16; i_140003++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_126531;
                double r_126533 = 0.0;
                
                for (int64_t i_126532 = 0; i_126532 < (int64_t) 4; i_126532++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_126534 = ((double *) mem_141176)[i_140053 * (int64_t) 64 + i_140007 * (int64_t) 4 + i_126532];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_126535 = ((double *) mem_141175)[i_140053 * (int64_t) 64 + i_140003 * (int64_t) 4 + i_126532];
                    
                    // futhark/microgpt.fut:152:113-164
                    
                    double zt_res_126536 = zt_lhs_126534 * zt_rhs_126535;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_126537 = r_126533 + zt_res_126536;
                    double r_tmp_143170 = zp_res_126537;
                    
                    r_126533 = r_tmp_143170;
                }
                defunc_0_lifted_lambda_res_126531 = r_126533;
                ((double *) mem_141266)[i_140003] = defunc_0_lifted_lambda_res_126531;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141261, i_140007 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141266, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140015 = 0; i_140015 < (int64_t) 16; i_140015++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140011 = 0; i_140011 < (int64_t) 16; i_140011++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_126552 = ((double *) mem_141261)[i_140015 * (int64_t) 16 + i_140011];
                
                // futhark/microgpt.fut:153:47-78
                
                double zs_res_126553 = zs_lhs_126552 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_126554 = ((double *) mask_mem_141063.mem)[i_140015 * (int64_t) 16 + i_140011];
                
                // futhark/microgpt.fut:153:65-102
                
                double zp_res_126555 = zs_res_126553 + zp_rhs_126554;
                
                ((double *) mem_141282)[i_140011] = zp_res_126555;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141277, i_140015 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141282, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140033 = 0; i_140033 < (int64_t) 16; i_140033++) {
            // futhark/microgpt.fut:105:13-33
            
            double defunc_0_reduce_res_129700;
            double redout_140017 = -INFINITY;
            
            for (int64_t i_140018 = 0; i_140018 < (int64_t) 16; i_140018++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129649 = ((double *) mem_141277)[i_140033 * (int64_t) 16 + i_140018];
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_126576 = fmax64(lifted_lambda_res_129649, redout_140017);
                double redout_tmp_143174 = max_res_126576;
                
                redout_140017 = redout_tmp_143174;
            }
            defunc_0_reduce_res_129700 = redout_140017;
            // futhark/microgpt.fut:155:67-76
            
            double neg_res_126577 = -defunc_0_reduce_res_129700;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140021 = 0; i_140021 < (int64_t) 16; i_140021++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_126584 = ((double *) mem_141277)[i_140033 * (int64_t) 16 + i_140021];
                
                // futhark/microgpt.fut:155:44-76
                
                double zp_res_126585 = neg_res_126577 + zp_lhs_126584;
                
                // futhark/microgpt.fut:155:37-76
                
                double exp_res_126586 = futrts_exp64(zp_res_126585);
                
                ((double *) mem_141298)[i_140021] = exp_res_126586;
            }
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126588;
            double r_126590 = 0.0;
            
            for (int64_t i_126589 = 0; i_126589 < (int64_t) 16; i_126589++) {
                // futhark/microgpt.fut:156:36-46
                
                double lifted_lambda_res_126591 = ((double *) mem_141298)[i_126589];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126592 = r_126590 + lifted_lambda_res_126591;
                double r_tmp_143176 = zp_res_126592;
                
                r_126590 = r_tmp_143176;
            }
            defunc_0_lifted_lambda_res_126588 = r_126590;
            // futhark/microgpt.fut:157:53-64
            
            double zs_res_126593 = 1.0 / defunc_0_lifted_lambda_res_126588;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140025 = 0; i_140025 < (int64_t) 16; i_140025++) {
                // futhark/microgpt.fut:157:37-47
                
                double zt_lhs_126600 = ((double *) mem_141298)[i_140025];
                
                // futhark/microgpt.fut:157:37-64
                
                double zt_res_126601 = zs_res_126593 * zt_lhs_126600;
                
                ((double *) mem_141305)[i_140025] = zt_res_126601;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140029 = 0; i_140029 < (int64_t) 16; i_140029++) {
                // futhark/microgpt.fut:158:4-14
                
                double lifted_lambda_res_126609 = ((double *) mem_141305)[i_140029];
                
                ((double *) mem_141312)[i_140029] = lifted_lambda_res_126609;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141293, i_140033 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141312, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140041 = 0; i_140041 < (int64_t) 16; i_140041++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140037 = 0; i_140037 < (int64_t) 4; i_140037++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_126624;
                double r_126626 = 0.0;
                
                for (int64_t i_126625 = 0; i_126625 < (int64_t) 16; i_126625++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_126627 = ((double *) mem_141293)[i_140041 * (int64_t) 16 + i_126625];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_126628 = ((double *) mem_141174)[i_140053 * (int64_t) 64 + i_126625 * (int64_t) 4 + i_140037];
                    
                    // futhark/microgpt.fut:159:66-111
                    
                    double zt_res_126629 = zt_lhs_126627 * zt_rhs_126628;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_126630 = r_126626 + zt_res_126629;
                    double r_tmp_143181 = zp_res_126630;
                    
                    r_126626 = r_tmp_143181;
                }
                defunc_0_lifted_lambda_res_126624 = r_126626;
                ((double *) mem_141328)[i_140037] = defunc_0_lifted_lambda_res_126624;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141323, i_140041 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141328, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140049 = 0; i_140049 < (int64_t) 16; i_140049++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140045 = 0; i_140045 < (int64_t) 4; i_140045++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_126645 = ((double *) mem_141323)[i_140049 * (int64_t) 4 + i_140045];
                
                ((double *) mem_141344)[i_140045] = lifted_lambda_res_126645;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141339, i_140049 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141344, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141255, i_140053 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_141339, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141360_cached_sizze_143570 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141360, &mem_141360_cached_sizze_143570, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141365_cached_sizze_143571 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141365, &mem_141365_cached_sizze_143571, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140061 = 0; i_140061 < (int64_t) 16; i_140061++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140057 = 0; i_140057 < (int64_t) 16; i_140057++) {
            // futhark/microgpt.fut:161:54-57
            
            int64_t tmp_126657 = sdiv64(i_140057, (int64_t) 4);
            
            // futhark/microgpt.fut:161:44-59
            
            bool x_126658 = sle64((int64_t) 0, tmp_126657);
            
            // futhark/microgpt.fut:161:44-59
            
            bool y_126659 = slt64(tmp_126657, (int64_t) 4);
            
            // futhark/microgpt.fut:161:44-59
            
            bool bounds_check_126660 = x_126658 && y_126659;
            
            // futhark/microgpt.fut:161:44-59
            
            bool index_certs_126661;
            
            if (!bounds_check_126660) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_126657, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:161:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:161:15-80\n   #6  futhark/microgpt.fut:437:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:161:74-77
            
            int64_t tmp_126662 = smod64(i_140057, (int64_t) 4);
            
            // futhark/microgpt.fut:161:44-79
            
            bool x_126663 = sle64((int64_t) 0, tmp_126662);
            
            // futhark/microgpt.fut:161:44-79
            
            bool y_126664 = slt64(tmp_126662, (int64_t) 4);
            
            // futhark/microgpt.fut:161:44-79
            
            bool bounds_check_126665 = x_126663 && y_126664;
            
            // futhark/microgpt.fut:161:44-79
            
            bool index_certs_126666;
            
            if (!bounds_check_126665) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_126662, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:161:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:161:15-80\n   #6  futhark/microgpt.fut:437:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_126667 = ((double *) mem_141255)[tmp_126657 * (int64_t) 64 + i_140061 * (int64_t) 4 + tmp_126662];
            
            ((double *) mem_141365)[i_140057] = lifted_lambda_res_126667;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141360, i_140061 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141365, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141376_cached_sizze_143572 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141376, &mem_141376_cached_sizze_143572, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141381_cached_sizze_143573 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141381, &mem_141381_cached_sizze_143573, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140069 = 0; i_140069 < (int64_t) 16; i_140069++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140065 = 0; i_140065 < (int64_t) 16; i_140065++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126682;
            double r_126684 = 0.0;
            
            for (int64_t i_126683 = 0; i_126683 < (int64_t) 16; i_126683++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_126685 = ((double *) wout_mem_141055.mem)[i_140065 * (int64_t) 16 + i_126683];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_126686 = ((double *) mem_141360)[i_140069 * (int64_t) 16 + i_126683];
                
                // futhark/microgpt.fut:162:67-106
                
                double zt_res_126687 = zt_lhs_126685 * zt_rhs_126686;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126688 = r_126684 + zt_res_126687;
                double r_tmp_143188 = zp_res_126688;
                
                r_126684 = r_tmp_143188;
            }
            defunc_0_lifted_lambda_res_126682 = r_126684;
            ((double *) mem_141381)[i_140065] = defunc_0_lifted_lambda_res_126682;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141376, i_140069 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141381, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141392_cached_sizze_143574 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141392, &mem_141392_cached_sizze_143574, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141397_cached_sizze_143575 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141397, &mem_141397_cached_sizze_143575, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140077 = 0; i_140077 < (int64_t) 16; i_140077++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140073 = 0; i_140073 < (int64_t) 16; i_140073++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_126703 = ((double *) mem_141376)[i_140077 * (int64_t) 16 + i_140073];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_126704 = ((double *) mem_141080)[i_140077 * (int64_t) 16 + i_140073];
            
            // futhark/microgpt.fut:163:46-84
            
            double zp_res_126705 = zp_lhs_126703 + zp_rhs_126704;
            
            ((double *) mem_141397)[i_140073] = zp_res_126705;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141392, i_140077 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141397, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141408_cached_sizze_143576 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141408, &mem_141408_cached_sizze_143576, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141413_cached_sizze_143577 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141413, &mem_141413_cached_sizze_143577, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141420_cached_sizze_143578 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141420, &mem_141420_cached_sizze_143578, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140089 = 0; i_140089 < (int64_t) 16; i_140089++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_126714;
        double r_126716 = 0.0;
        
        for (int64_t i_126715 = 0; i_126715 < (int64_t) 16; i_126715++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_126717 = ((double *) mem_141392)[i_140089 * (int64_t) 16 + i_126715];
            
            // futhark/microgpt.fut:164:79-118
            
            double zt_res_126718 = zt_lhs_126717 * zt_lhs_126717;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_126719 = r_126716 + zt_res_126718;
            double r_tmp_143192 = zp_res_126719;
            
            r_126716 = r_tmp_143192;
        }
        defunc_0_lifted_lambda_res_126714 = r_126716;
        // futhark/microgpt.fut:164:58-136
        
        double zs_res_126720 = defunc_0_lifted_lambda_res_126714 / 16.0;
        
        // futhark/microgpt.fut:165:24-55
        
        double zp_res_126721 = 1.0e-5 + zs_res_126720;
        
        // futhark/microgpt.fut:165:16-55
        
        double sqrt_res_126722 = futrts_sqrt64(zp_res_126721);
        
        // futhark/microgpt.fut:166:60-71
        
        double zs_res_126723 = 1.0 / sqrt_res_126722;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140081 = 0; i_140081 < (int64_t) 16; i_140081++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_126730 = ((double *) mem_141392)[i_140089 * (int64_t) 16 + i_140081];
            
            // futhark/microgpt.fut:166:37-71
            
            double zt_res_126731 = zs_res_126723 * zt_lhs_126730;
            
            ((double *) mem_141413)[i_140081] = zt_res_126731;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140085 = 0; i_140085 < (int64_t) 16; i_140085++) {
            // futhark/microgpt.fut:167:4-14
            
            double lifted_lambda_res_126739 = ((double *) mem_141413)[i_140085];
            
            ((double *) mem_141420)[i_140085] = lifted_lambda_res_126739;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141408, i_140089 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141420, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141431_cached_sizze_143579 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141431, &mem_141431_cached_sizze_143579, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141436_cached_sizze_143580 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141436, &mem_141436_cached_sizze_143580, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140097 = 0; i_140097 < (int64_t) 16; i_140097++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140093 = 0; i_140093 < (int64_t) 64; i_140093++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126755;
            double r_126757 = 0.0;
            
            for (int64_t i_126756 = 0; i_126756 < (int64_t) 16; i_126756++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_126758 = ((double *) wup_mem_141059.mem)[i_140093 * (int64_t) 16 + i_126756];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_126759 = ((double *) mem_141408)[i_140097 * (int64_t) 16 + i_126756];
                
                // futhark/microgpt.fut:168:67-106
                
                double zt_res_126760 = zt_lhs_126758 * zt_rhs_126759;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126761 = r_126757 + zt_res_126760;
                double r_tmp_143197 = zp_res_126761;
                
                r_126757 = r_tmp_143197;
            }
            defunc_0_lifted_lambda_res_126755 = r_126757;
            ((double *) mem_141436)[i_140093] = defunc_0_lifted_lambda_res_126755;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141431, i_140097 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141436, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141447_cached_sizze_143581 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141447, &mem_141447_cached_sizze_143581, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141452_cached_sizze_143582 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141452, &mem_141452_cached_sizze_143582, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140105 = 0; i_140105 < (int64_t) 16; i_140105++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140101 = 0; i_140101 < (int64_t) 64; i_140101++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_126776 = ((double *) mem_141431)[i_140105 * (int64_t) 64 + i_140101];
            
            // futhark/microgpt.fut:169:45-73
            
            double max_res_126777 = fmax64(0.0, max_arg0_126776);
            
            ((double *) mem_141452)[i_140101] = max_res_126777;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141447, i_140105 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141452, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141463_cached_sizze_143583 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141463, &mem_141463_cached_sizze_143583, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141468_cached_sizze_143584 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141468, &mem_141468_cached_sizze_143584, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140113 = 0; i_140113 < (int64_t) 16; i_140113++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140109 = 0; i_140109 < (int64_t) 16; i_140109++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126792;
            double r_126794 = 0.0;
            
            for (int64_t i_126793 = 0; i_126793 < (int64_t) 64; i_126793++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_126795 = ((double *) wdown_mem_141053.mem)[i_140109 * (int64_t) 64 + i_126793];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_126796 = ((double *) mem_141447)[i_140113 * (int64_t) 64 + i_126793];
                
                // futhark/microgpt.fut:170:67-108
                
                double zt_res_126797 = zt_lhs_126795 * zt_rhs_126796;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126798 = r_126794 + zt_res_126797;
                double r_tmp_143202 = zp_res_126798;
                
                r_126794 = r_tmp_143202;
            }
            defunc_0_lifted_lambda_res_126792 = r_126794;
            ((double *) mem_141468)[i_140109] = defunc_0_lifted_lambda_res_126792;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141463, i_140113 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141468, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141479_cached_sizze_143585 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141479, &mem_141479_cached_sizze_143585, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141484_cached_sizze_143586 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141484, &mem_141484_cached_sizze_143586, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140121 = 0; i_140121 < (int64_t) 16; i_140121++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140117 = 0; i_140117 < (int64_t) 16; i_140117++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_126813 = ((double *) mem_141463)[i_140121 * (int64_t) 16 + i_140117];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_126814 = ((double *) mem_141392)[i_140121 * (int64_t) 16 + i_140117];
            
            // futhark/microgpt.fut:171:46-85
            
            double zp_res_126815 = zp_lhs_126813 + zp_rhs_126814;
            
            ((double *) mem_141484)[i_140117] = zp_res_126815;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141479, i_140121 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141484, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141495_cached_sizze_143587 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_141495, &mem_141495_cached_sizze_143587, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141500_cached_sizze_143588 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_141500, &mem_141500_cached_sizze_143588, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140129 = 0; i_140129 < (int64_t) 16; i_140129++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140125 = 0; i_140125 < (int64_t) 27; i_140125++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126831;
            double r_126833 = 0.0;
            
            for (int64_t i_126832 = 0; i_126832 < (int64_t) 16; i_126832++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_126834 = ((double *) wvoc_mem_141061.mem)[i_140125 * (int64_t) 16 + i_126832];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_126835 = ((double *) mem_141479)[i_140129 * (int64_t) 16 + i_126832];
                
                // futhark/microgpt.fut:172:67-107
                
                double zt_res_126836 = zt_lhs_126834 * zt_rhs_126835;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126837 = r_126833 + zt_res_126836;
                double r_tmp_143207 = zp_res_126837;
                
                r_126833 = r_tmp_143207;
            }
            defunc_0_lifted_lambda_res_126831 = r_126833;
            ((double *) mem_141500)[i_140125] = defunc_0_lifted_lambda_res_126831;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141495, i_140129 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141500, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_141511, (int64_t) 3456, "mem_141511")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141516_cached_sizze_143589 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_141516, &mem_141516_cached_sizze_143589, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140137 = 0; i_140137 < (int64_t) 16; i_140137++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140133 = 0; i_140133 < (int64_t) 27; i_140133++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_126852 = ((double *) mem_141495)[i_140137 * (int64_t) 27 + i_140133];
            
            ((double *) mem_141516)[i_140133] = lifted_lambda_res_126852;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141511.mem, i_140137 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141516, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_143138, &mem_141511, "mem_141511") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143533, &mem_out_143138, "mem_out_143138") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_141064);
        free(mem_141069);
        free(mem_141080);
        free(mem_141085);
        free(mem_141092);
        free(mem_141103);
        free(mem_141108);
        free(mem_141115);
        free(mem_141126);
        free(mem_141127);
        free(mem_141128);
        free(mem_141141);
        free(mem_141142);
        free(mem_141143);
        free(mem_141174);
        free(mem_141175);
        free(mem_141176);
        free(mem_141192);
        free(mem_141193);
        free(mem_141194);
        free(mem_141207);
        free(mem_141208);
        free(mem_141209);
        free(mem_141255);
        free(mem_141261);
        free(mem_141266);
        free(mem_141277);
        free(mem_141282);
        free(mem_141293);
        free(mem_141298);
        free(mem_141305);
        free(mem_141312);
        free(mem_141323);
        free(mem_141328);
        free(mem_141339);
        free(mem_141344);
        free(mem_141360);
        free(mem_141365);
        free(mem_141376);
        free(mem_141381);
        free(mem_141392);
        free(mem_141397);
        free(mem_141408);
        free(mem_141413);
        free(mem_141420);
        free(mem_141431);
        free(mem_141436);
        free(mem_141447);
        free(mem_141452);
        free(mem_141463);
        free(mem_141468);
        free(mem_141479);
        free(mem_141484);
        free(mem_141495);
        free(mem_141500);
        free(mem_141516);
        if (memblock_unref(ctx, &mem_141511, "mem_141511") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143138, "mem_out_143138") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_grad_loss(struct futhark_context *ctx, struct memblock *mem_out_p_143590, struct memblock *mem_out_p_143591, struct memblock *mem_out_p_143592, struct memblock *mem_out_p_143593, struct memblock *mem_out_p_143594, struct memblock *mem_out_p_143595, struct memblock *mem_out_p_143596, struct memblock *mem_out_p_143597, struct memblock *mem_out_p_143598, struct memblock wdown_mem_141053, struct memblock wkey_mem_141054, struct memblock wout_mem_141055, struct memblock wpe_mem_141056, struct memblock wqry_mem_141057, struct memblock wte_mem_141058, struct memblock wup_mem_141059, struct memblock wval_mem_141060, struct memblock wvoc_mem_141061, struct memblock tokens_mem_141062, struct memblock target_mem_141063, struct memblock mask_mem_141064)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_141065_cached_sizze_143599 = 0;
    unsigned char *mem_141065 = NULL;
    int64_t mem_141070_cached_sizze_143600 = 0;
    unsigned char *mem_141070 = NULL;
    int64_t mem_141081_cached_sizze_143601 = 0;
    unsigned char *mem_141081 = NULL;
    int64_t mem_141082_cached_sizze_143602 = 0;
    unsigned char *mem_141082 = NULL;
    int64_t mem_141083_cached_sizze_143603 = 0;
    unsigned char *mem_141083 = NULL;
    int64_t mem_141102_cached_sizze_143604 = 0;
    unsigned char *mem_141102 = NULL;
    int64_t mem_141109_cached_sizze_143605 = 0;
    unsigned char *mem_141109 = NULL;
    int64_t mem_141114_cached_sizze_143606 = 0;
    unsigned char *mem_141114 = NULL;
    int64_t mem_141125_cached_sizze_143607 = 0;
    unsigned char *mem_141125 = NULL;
    int64_t mem_141130_cached_sizze_143608 = 0;
    unsigned char *mem_141130 = NULL;
    int64_t mem_141141_cached_sizze_143609 = 0;
    unsigned char *mem_141141 = NULL;
    int64_t mem_141142_cached_sizze_143610 = 0;
    unsigned char *mem_141142 = NULL;
    int64_t mem_141155_cached_sizze_143611 = 0;
    unsigned char *mem_141155 = NULL;
    int64_t mem_141162_cached_sizze_143612 = 0;
    unsigned char *mem_141162 = NULL;
    int64_t mem_141167_cached_sizze_143613 = 0;
    unsigned char *mem_141167 = NULL;
    int64_t mem_141178_cached_sizze_143614 = 0;
    unsigned char *mem_141178 = NULL;
    int64_t mem_141183_cached_sizze_143615 = 0;
    unsigned char *mem_141183 = NULL;
    int64_t mem_141194_cached_sizze_143616 = 0;
    unsigned char *mem_141194 = NULL;
    int64_t mem_141195_cached_sizze_143617 = 0;
    unsigned char *mem_141195 = NULL;
    int64_t mem_141196_cached_sizze_143618 = 0;
    unsigned char *mem_141196 = NULL;
    int64_t mem_141212_cached_sizze_143619 = 0;
    unsigned char *mem_141212 = NULL;
    int64_t mem_141213_cached_sizze_143620 = 0;
    unsigned char *mem_141213 = NULL;
    int64_t mem_141214_cached_sizze_143621 = 0;
    unsigned char *mem_141214 = NULL;
    int64_t mem_141227_cached_sizze_143622 = 0;
    unsigned char *mem_141227 = NULL;
    int64_t mem_141228_cached_sizze_143623 = 0;
    unsigned char *mem_141228 = NULL;
    int64_t mem_141229_cached_sizze_143624 = 0;
    unsigned char *mem_141229 = NULL;
    int64_t mem_141275_cached_sizze_143625 = 0;
    unsigned char *mem_141275 = NULL;
    int64_t mem_141276_cached_sizze_143626 = 0;
    unsigned char *mem_141276 = NULL;
    int64_t mem_141277_cached_sizze_143627 = 0;
    unsigned char *mem_141277 = NULL;
    int64_t mem_141278_cached_sizze_143628 = 0;
    unsigned char *mem_141278 = NULL;
    int64_t mem_141299_cached_sizze_143629 = 0;
    unsigned char *mem_141299 = NULL;
    int64_t mem_141300_cached_sizze_143630 = 0;
    unsigned char *mem_141300 = NULL;
    int64_t mem_141301_cached_sizze_143631 = 0;
    unsigned char *mem_141301 = NULL;
    int64_t mem_141302_cached_sizze_143632 = 0;
    unsigned char *mem_141302 = NULL;
    int64_t mem_141319_cached_sizze_143633 = 0;
    unsigned char *mem_141319 = NULL;
    int64_t mem_141320_cached_sizze_143634 = 0;
    unsigned char *mem_141320 = NULL;
    int64_t mem_141321_cached_sizze_143635 = 0;
    unsigned char *mem_141321 = NULL;
    int64_t mem_141322_cached_sizze_143636 = 0;
    unsigned char *mem_141322 = NULL;
    int64_t mem_141383_cached_sizze_143637 = 0;
    unsigned char *mem_141383 = NULL;
    int64_t mem_141384_cached_sizze_143638 = 0;
    unsigned char *mem_141384 = NULL;
    int64_t mem_141385_cached_sizze_143639 = 0;
    unsigned char *mem_141385 = NULL;
    int64_t mem_141386_cached_sizze_143640 = 0;
    unsigned char *mem_141386 = NULL;
    int64_t mem_141407_cached_sizze_143641 = 0;
    unsigned char *mem_141407 = NULL;
    int64_t mem_141408_cached_sizze_143642 = 0;
    unsigned char *mem_141408 = NULL;
    int64_t mem_141409_cached_sizze_143643 = 0;
    unsigned char *mem_141409 = NULL;
    int64_t mem_141410_cached_sizze_143644 = 0;
    unsigned char *mem_141410 = NULL;
    int64_t mem_141427_cached_sizze_143645 = 0;
    unsigned char *mem_141427 = NULL;
    int64_t mem_141428_cached_sizze_143646 = 0;
    unsigned char *mem_141428 = NULL;
    int64_t mem_141429_cached_sizze_143647 = 0;
    unsigned char *mem_141429 = NULL;
    int64_t mem_141430_cached_sizze_143648 = 0;
    unsigned char *mem_141430 = NULL;
    int64_t mem_141491_cached_sizze_143649 = 0;
    unsigned char *mem_141491 = NULL;
    int64_t mem_141492_cached_sizze_143650 = 0;
    unsigned char *mem_141492 = NULL;
    int64_t mem_141493_cached_sizze_143651 = 0;
    unsigned char *mem_141493 = NULL;
    int64_t mem_141494_cached_sizze_143652 = 0;
    unsigned char *mem_141494 = NULL;
    int64_t mem_141495_cached_sizze_143653 = 0;
    unsigned char *mem_141495 = NULL;
    int64_t mem_141496_cached_sizze_143654 = 0;
    unsigned char *mem_141496 = NULL;
    int64_t mem_141497_cached_sizze_143655 = 0;
    unsigned char *mem_141497 = NULL;
    int64_t mem_141498_cached_sizze_143656 = 0;
    unsigned char *mem_141498 = NULL;
    int64_t mem_141531_cached_sizze_143657 = 0;
    unsigned char *mem_141531 = NULL;
    int64_t mem_141532_cached_sizze_143658 = 0;
    unsigned char *mem_141532 = NULL;
    int64_t mem_141533_cached_sizze_143659 = 0;
    unsigned char *mem_141533 = NULL;
    int64_t mem_141534_cached_sizze_143660 = 0;
    unsigned char *mem_141534 = NULL;
    int64_t mem_141535_cached_sizze_143661 = 0;
    unsigned char *mem_141535 = NULL;
    int64_t mem_141536_cached_sizze_143662 = 0;
    unsigned char *mem_141536 = NULL;
    int64_t mem_141537_cached_sizze_143663 = 0;
    unsigned char *mem_141537 = NULL;
    int64_t mem_141538_cached_sizze_143664 = 0;
    unsigned char *mem_141538 = NULL;
    int64_t mem_141619_cached_sizze_143665 = 0;
    unsigned char *mem_141619 = NULL;
    int64_t mem_141620_cached_sizze_143666 = 0;
    unsigned char *mem_141620 = NULL;
    int64_t mem_141621_cached_sizze_143667 = 0;
    unsigned char *mem_141621 = NULL;
    int64_t mem_141622_cached_sizze_143668 = 0;
    unsigned char *mem_141622 = NULL;
    int64_t mem_141643_cached_sizze_143669 = 0;
    unsigned char *mem_141643 = NULL;
    int64_t mem_141644_cached_sizze_143670 = 0;
    unsigned char *mem_141644 = NULL;
    int64_t mem_141645_cached_sizze_143671 = 0;
    unsigned char *mem_141645 = NULL;
    int64_t mem_141646_cached_sizze_143672 = 0;
    unsigned char *mem_141646 = NULL;
    int64_t mem_141663_cached_sizze_143673 = 0;
    unsigned char *mem_141663 = NULL;
    int64_t mem_141664_cached_sizze_143674 = 0;
    unsigned char *mem_141664 = NULL;
    int64_t mem_141665_cached_sizze_143675 = 0;
    unsigned char *mem_141665 = NULL;
    int64_t mem_141666_cached_sizze_143676 = 0;
    unsigned char *mem_141666 = NULL;
    int64_t mem_141727_cached_sizze_143677 = 0;
    unsigned char *mem_141727 = NULL;
    int64_t mem_141728_cached_sizze_143678 = 0;
    unsigned char *mem_141728 = NULL;
    int64_t mem_141737_cached_sizze_143679 = 0;
    unsigned char *mem_141737 = NULL;
    int64_t mem_141738_cached_sizze_143680 = 0;
    unsigned char *mem_141738 = NULL;
    int64_t mem_141759_cached_sizze_143681 = 0;
    unsigned char *mem_141759 = NULL;
    int64_t mem_141760_cached_sizze_143682 = 0;
    unsigned char *mem_141760 = NULL;
    int64_t mem_141771_cached_sizze_143683 = 0;
    unsigned char *mem_141771 = NULL;
    int64_t mem_141772_cached_sizze_143684 = 0;
    unsigned char *mem_141772 = NULL;
    int64_t mem_141781_cached_sizze_143685 = 0;
    unsigned char *mem_141781 = NULL;
    int64_t mem_141782_cached_sizze_143686 = 0;
    unsigned char *mem_141782 = NULL;
    int64_t mem_141813_cached_sizze_143687 = 0;
    unsigned char *mem_141813 = NULL;
    int64_t mem_141814_cached_sizze_143688 = 0;
    unsigned char *mem_141814 = NULL;
    int64_t mem_141825_cached_sizze_143689 = 0;
    unsigned char *mem_141825 = NULL;
    int64_t mem_141826_cached_sizze_143690 = 0;
    unsigned char *mem_141826 = NULL;
    int64_t mem_141835_cached_sizze_143691 = 0;
    unsigned char *mem_141835 = NULL;
    int64_t mem_141836_cached_sizze_143692 = 0;
    unsigned char *mem_141836 = NULL;
    int64_t mem_141867_cached_sizze_143693 = 0;
    unsigned char *mem_141867 = NULL;
    int64_t mem_141873_cached_sizze_143694 = 0;
    unsigned char *mem_141873 = NULL;
    int64_t mem_141878_cached_sizze_143695 = 0;
    unsigned char *mem_141878 = NULL;
    int64_t mem_141894_cached_sizze_143696 = 0;
    unsigned char *mem_141894 = NULL;
    int64_t mem_141899_cached_sizze_143697 = 0;
    unsigned char *mem_141899 = NULL;
    int64_t mem_141910_cached_sizze_143698 = 0;
    unsigned char *mem_141910 = NULL;
    int64_t mem_141915_cached_sizze_143699 = 0;
    unsigned char *mem_141915 = NULL;
    int64_t mem_141926_cached_sizze_143700 = 0;
    unsigned char *mem_141926 = NULL;
    int64_t mem_141927_cached_sizze_143701 = 0;
    unsigned char *mem_141927 = NULL;
    int64_t mem_141940_cached_sizze_143702 = 0;
    unsigned char *mem_141940 = NULL;
    int64_t mem_141947_cached_sizze_143703 = 0;
    unsigned char *mem_141947 = NULL;
    int64_t mem_141952_cached_sizze_143704 = 0;
    unsigned char *mem_141952 = NULL;
    int64_t mem_141963_cached_sizze_143705 = 0;
    unsigned char *mem_141963 = NULL;
    int64_t mem_141968_cached_sizze_143706 = 0;
    unsigned char *mem_141968 = NULL;
    int64_t mem_141979_cached_sizze_143707 = 0;
    unsigned char *mem_141979 = NULL;
    int64_t mem_141984_cached_sizze_143708 = 0;
    unsigned char *mem_141984 = NULL;
    int64_t mem_141995_cached_sizze_143709 = 0;
    unsigned char *mem_141995 = NULL;
    int64_t mem_142000_cached_sizze_143710 = 0;
    unsigned char *mem_142000 = NULL;
    int64_t mem_142011_cached_sizze_143711 = 0;
    unsigned char *mem_142011 = NULL;
    int64_t mem_142016_cached_sizze_143712 = 0;
    unsigned char *mem_142016 = NULL;
    int64_t mem_142027_cached_sizze_143713 = 0;
    unsigned char *mem_142027 = NULL;
    int64_t mem_142032_cached_sizze_143714 = 0;
    unsigned char *mem_142032 = NULL;
    int64_t mem_142043_cached_sizze_143715 = 0;
    unsigned char *mem_142043 = NULL;
    int64_t mem_142044_cached_sizze_143716 = 0;
    unsigned char *mem_142044 = NULL;
    int64_t mem_142045_cached_sizze_143717 = 0;
    unsigned char *mem_142045 = NULL;
    int64_t mem_142046_cached_sizze_143718 = 0;
    unsigned char *mem_142046 = NULL;
    int64_t mem_142064_cached_sizze_143719 = 0;
    unsigned char *mem_142064 = NULL;
    int64_t mem_142069_cached_sizze_143720 = 0;
    unsigned char *mem_142069 = NULL;
    int64_t mem_142073_cached_sizze_143721 = 0;
    unsigned char *mem_142073 = NULL;
    int64_t mem_142080_cached_sizze_143722 = 0;
    unsigned char *mem_142080 = NULL;
    int64_t mem_142114_cached_sizze_143723 = 0;
    unsigned char *mem_142114 = NULL;
    int64_t mem_142120_cached_sizze_143724 = 0;
    unsigned char *mem_142120 = NULL;
    int64_t mem_142125_cached_sizze_143725 = 0;
    unsigned char *mem_142125 = NULL;
    int64_t mem_142141_cached_sizze_143726 = 0;
    unsigned char *mem_142141 = NULL;
    int64_t mem_142142_cached_sizze_143727 = 0;
    unsigned char *mem_142142 = NULL;
    int64_t mem_142151_cached_sizze_143728 = 0;
    unsigned char *mem_142151 = NULL;
    int64_t mem_142152_cached_sizze_143729 = 0;
    unsigned char *mem_142152 = NULL;
    int64_t mem_142173_cached_sizze_143730 = 0;
    unsigned char *mem_142173 = NULL;
    int64_t mem_142179_cached_sizze_143731 = 0;
    unsigned char *mem_142179 = NULL;
    int64_t mem_142184_cached_sizze_143732 = 0;
    unsigned char *mem_142184 = NULL;
    int64_t mem_142200_cached_sizze_143733 = 0;
    unsigned char *mem_142200 = NULL;
    int64_t mem_142205_cached_sizze_143734 = 0;
    unsigned char *mem_142205 = NULL;
    int64_t mem_142216_cached_sizze_143735 = 0;
    unsigned char *mem_142216 = NULL;
    int64_t mem_142221_cached_sizze_143736 = 0;
    unsigned char *mem_142221 = NULL;
    int64_t mem_142232_cached_sizze_143737 = 0;
    unsigned char *mem_142232 = NULL;
    int64_t mem_142237_cached_sizze_143738 = 0;
    unsigned char *mem_142237 = NULL;
    int64_t mem_142249_cached_sizze_143739 = 0;
    unsigned char *mem_142249 = NULL;
    int64_t mem_142258_cached_sizze_143740 = 0;
    unsigned char *mem_142258 = NULL;
    int64_t mem_142259_cached_sizze_143741 = 0;
    unsigned char *mem_142259 = NULL;
    int64_t mem_142280_cached_sizze_143742 = 0;
    unsigned char *mem_142280 = NULL;
    int64_t mem_142285_cached_sizze_143743 = 0;
    unsigned char *mem_142285 = NULL;
    int64_t mem_142296_cached_sizze_143744 = 0;
    unsigned char *mem_142296 = NULL;
    int64_t mem_142297_cached_sizze_143745 = 0;
    unsigned char *mem_142297 = NULL;
    int64_t mem_142310_cached_sizze_143746 = 0;
    unsigned char *mem_142310 = NULL;
    int64_t mem_142317_cached_sizze_143747 = 0;
    unsigned char *mem_142317 = NULL;
    int64_t mem_142322_cached_sizze_143748 = 0;
    unsigned char *mem_142322 = NULL;
    int64_t mem_142333_cached_sizze_143749 = 0;
    unsigned char *mem_142333 = NULL;
    int64_t mem_142339_cached_sizze_143750 = 0;
    unsigned char *mem_142339 = NULL;
    int64_t mem_142344_cached_sizze_143751 = 0;
    unsigned char *mem_142344 = NULL;
    int64_t mem_142360_cached_sizze_143752 = 0;
    unsigned char *mem_142360 = NULL;
    int64_t mem_142361_cached_sizze_143753 = 0;
    unsigned char *mem_142361 = NULL;
    int64_t mem_142362_cached_sizze_143754 = 0;
    unsigned char *mem_142362 = NULL;
    int64_t mem_142378_cached_sizze_143755 = 0;
    unsigned char *mem_142378 = NULL;
    int64_t mem_142379_cached_sizze_143756 = 0;
    unsigned char *mem_142379 = NULL;
    int64_t mem_142380_cached_sizze_143757 = 0;
    unsigned char *mem_142380 = NULL;
    int64_t mem_142393_cached_sizze_143758 = 0;
    unsigned char *mem_142393 = NULL;
    int64_t mem_142394_cached_sizze_143759 = 0;
    unsigned char *mem_142394 = NULL;
    int64_t mem_142435_cached_sizze_143760 = 0;
    unsigned char *mem_142435 = NULL;
    int64_t mem_142436_cached_sizze_143761 = 0;
    unsigned char *mem_142436 = NULL;
    int64_t mem_142447_cached_sizze_143762 = 0;
    unsigned char *mem_142447 = NULL;
    int64_t mem_142448_cached_sizze_143763 = 0;
    unsigned char *mem_142448 = NULL;
    int64_t mem_142457_cached_sizze_143764 = 0;
    unsigned char *mem_142457 = NULL;
    int64_t mem_142458_cached_sizze_143765 = 0;
    unsigned char *mem_142458 = NULL;
    int64_t mem_142489_cached_sizze_143766 = 0;
    unsigned char *mem_142489 = NULL;
    int64_t mem_142490_cached_sizze_143767 = 0;
    unsigned char *mem_142490 = NULL;
    int64_t mem_142501_cached_sizze_143768 = 0;
    unsigned char *mem_142501 = NULL;
    int64_t mem_142502_cached_sizze_143769 = 0;
    unsigned char *mem_142502 = NULL;
    int64_t mem_142511_cached_sizze_143770 = 0;
    unsigned char *mem_142511 = NULL;
    int64_t mem_142512_cached_sizze_143771 = 0;
    unsigned char *mem_142512 = NULL;
    int64_t mem_142543_cached_sizze_143772 = 0;
    unsigned char *mem_142543 = NULL;
    int64_t mem_142544_cached_sizze_143773 = 0;
    unsigned char *mem_142544 = NULL;
    int64_t mem_142545_cached_sizze_143774 = 0;
    unsigned char *mem_142545 = NULL;
    int64_t mem_142546_cached_sizze_143775 = 0;
    unsigned char *mem_142546 = NULL;
    int64_t mem_142563_cached_sizze_143776 = 0;
    unsigned char *mem_142563 = NULL;
    int64_t mem_142564_cached_sizze_143777 = 0;
    unsigned char *mem_142564 = NULL;
    int64_t mem_142565_cached_sizze_143778 = 0;
    unsigned char *mem_142565 = NULL;
    int64_t mem_142566_cached_sizze_143779 = 0;
    unsigned char *mem_142566 = NULL;
    int64_t mem_142607_cached_sizze_143780 = 0;
    unsigned char *mem_142607 = NULL;
    int64_t mem_142608_cached_sizze_143781 = 0;
    unsigned char *mem_142608 = NULL;
    int64_t mem_142619_cached_sizze_143782 = 0;
    unsigned char *mem_142619 = NULL;
    int64_t mem_142620_cached_sizze_143783 = 0;
    unsigned char *mem_142620 = NULL;
    int64_t mem_142629_cached_sizze_143784 = 0;
    unsigned char *mem_142629 = NULL;
    int64_t mem_142630_cached_sizze_143785 = 0;
    unsigned char *mem_142630 = NULL;
    int64_t mem_142661_cached_sizze_143786 = 0;
    unsigned char *mem_142661 = NULL;
    int64_t mem_142662_cached_sizze_143787 = 0;
    unsigned char *mem_142662 = NULL;
    int64_t mem_142671_cached_sizze_143788 = 0;
    unsigned char *mem_142671 = NULL;
    int64_t mem_142672_cached_sizze_143789 = 0;
    unsigned char *mem_142672 = NULL;
    int64_t mem_142693_cached_sizze_143790 = 0;
    unsigned char *mem_142693 = NULL;
    int64_t mem_142694_cached_sizze_143791 = 0;
    unsigned char *mem_142694 = NULL;
    int64_t mem_142705_cached_sizze_143792 = 0;
    unsigned char *mem_142705 = NULL;
    int64_t mem_142706_cached_sizze_143793 = 0;
    unsigned char *mem_142706 = NULL;
    int64_t mem_142715_cached_sizze_143794 = 0;
    unsigned char *mem_142715 = NULL;
    int64_t mem_142716_cached_sizze_143795 = 0;
    unsigned char *mem_142716 = NULL;
    int64_t mem_142747_cached_sizze_143796 = 0;
    unsigned char *mem_142747 = NULL;
    int64_t mem_142748_cached_sizze_143797 = 0;
    unsigned char *mem_142748 = NULL;
    int64_t mem_142759_cached_sizze_143798 = 0;
    unsigned char *mem_142759 = NULL;
    int64_t mem_142760_cached_sizze_143799 = 0;
    unsigned char *mem_142760 = NULL;
    int64_t mem_142769_cached_sizze_143800 = 0;
    unsigned char *mem_142769 = NULL;
    int64_t mem_142770_cached_sizze_143801 = 0;
    unsigned char *mem_142770 = NULL;
    int64_t mem_142802_cached_sizze_143802 = 0;
    unsigned char *mem_142802 = NULL;
    int64_t mem_142803_cached_sizze_143803 = 0;
    unsigned char *mem_142803 = NULL;
    int64_t mem_142804_cached_sizze_143804 = 0;
    unsigned char *mem_142804 = NULL;
    int64_t mem_142821_cached_sizze_143805 = 0;
    unsigned char *mem_142821 = NULL;
    int64_t mem_142822_cached_sizze_143806 = 0;
    unsigned char *mem_142822 = NULL;
    int64_t mem_142823_cached_sizze_143807 = 0;
    unsigned char *mem_142823 = NULL;
    int64_t mem_142824_cached_sizze_143808 = 0;
    unsigned char *mem_142824 = NULL;
    int64_t mem_142865_cached_sizze_143809 = 0;
    unsigned char *mem_142865 = NULL;
    int64_t mem_142870_cached_sizze_143810 = 0;
    unsigned char *mem_142870 = NULL;
    int64_t mem_142884_cached_sizze_143811 = 0;
    unsigned char *mem_142884 = NULL;
    int64_t mem_142885_cached_sizze_143812 = 0;
    unsigned char *mem_142885 = NULL;
    int64_t mem_142904_cached_sizze_143813 = 0;
    unsigned char *mem_142904 = NULL;
    int64_t mem_142905_cached_sizze_143814 = 0;
    unsigned char *mem_142905 = NULL;
    int64_t mem_142906_cached_sizze_143815 = 0;
    unsigned char *mem_142906 = NULL;
    int64_t mem_142943_cached_sizze_143816 = 0;
    unsigned char *mem_142943 = NULL;
    int64_t mem_142950_cached_sizze_143817 = 0;
    unsigned char *mem_142950 = NULL;
    int64_t mem_142955_cached_sizze_143818 = 0;
    unsigned char *mem_142955 = NULL;
    int64_t mem_142966_cached_sizze_143819 = 0;
    unsigned char *mem_142966 = NULL;
    int64_t mem_142967_cached_sizze_143820 = 0;
    unsigned char *mem_142967 = NULL;
    int64_t mem_142976_cached_sizze_143821 = 0;
    unsigned char *mem_142976 = NULL;
    int64_t mem_142977_cached_sizze_143822 = 0;
    unsigned char *mem_142977 = NULL;
    int64_t mem_142998_cached_sizze_143823 = 0;
    unsigned char *mem_142998 = NULL;
    int64_t mem_142999_cached_sizze_143824 = 0;
    unsigned char *mem_142999 = NULL;
    int64_t mem_143000_cached_sizze_143825 = 0;
    unsigned char *mem_143000 = NULL;
    int64_t mem_143001_cached_sizze_143826 = 0;
    unsigned char *mem_143001 = NULL;
    int64_t mem_143026_cached_sizze_143827 = 0;
    unsigned char *mem_143026 = NULL;
    int64_t mem_143027_cached_sizze_143828 = 0;
    unsigned char *mem_143027 = NULL;
    int64_t mem_143040_cached_sizze_143829 = 0;
    unsigned char *mem_143040 = NULL;
    int64_t mem_143050_cached_sizze_143830 = 0;
    unsigned char *mem_143050 = NULL;
    int64_t mem_143051_cached_sizze_143831 = 0;
    unsigned char *mem_143051 = NULL;
    int64_t mem_143077_cached_sizze_143832 = 0;
    unsigned char *mem_143077 = NULL;
    int64_t mem_143098_cached_sizze_143833 = 0;
    unsigned char *mem_143098 = NULL;
    int64_t mem_143099_cached_sizze_143834 = 0;
    unsigned char *mem_143099 = NULL;
    struct memblock mem_143089;
    
    mem_143089.references = NULL;
    
    struct memblock mem_143088;
    
    mem_143088.references = NULL;
    
    struct memblock mem_143072;
    
    mem_143072.references = NULL;
    
    struct memblock mem_143041;
    
    mem_143041.references = NULL;
    
    struct memblock mem_142883;
    
    mem_142883.references = NULL;
    
    struct memblock mem_142882;
    
    mem_142882.references = NULL;
    
    struct memblock mem_142881;
    
    mem_142881.references = NULL;
    
    struct memblock mem_142801;
    
    mem_142801.references = NULL;
    
    struct memblock mem_142248;
    
    mem_142248.references = NULL;
    
    struct memblock mem_out_143146;
    
    mem_out_143146.references = NULL;
    
    struct memblock mem_out_143145;
    
    mem_out_143145.references = NULL;
    
    struct memblock mem_out_143144;
    
    mem_out_143144.references = NULL;
    
    struct memblock mem_out_143143;
    
    mem_out_143143.references = NULL;
    
    struct memblock mem_out_143142;
    
    mem_out_143142.references = NULL;
    
    struct memblock mem_out_143141;
    
    mem_out_143141.references = NULL;
    
    struct memblock mem_out_143140;
    
    mem_out_143140.references = NULL;
    
    struct memblock mem_out_143139;
    
    mem_out_143139.references = NULL;
    
    struct memblock mem_out_143138;
    
    mem_out_143138.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (mem_141065_cached_sizze_143599 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141065, &mem_141065_cached_sizze_143599, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141070_cached_sizze_143600 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141070, &mem_141070_cached_sizze_143600, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139925 = 0; i_139925 < (int64_t) 16; i_139925++) {
        // futhark/microgpt.fut:457:41-50
        
        int64_t tmp_126287 = ((int64_t *) tokens_mem_141062.mem)[i_139925];
        
        // futhark/microgpt.fut:457:37-51
        
        bool x_126288 = sle64((int64_t) 0, tmp_126287);
        
        // futhark/microgpt.fut:457:37-51
        
        bool y_126289 = slt64(tmp_126287, (int64_t) 27);
        
        // futhark/microgpt.fut:457:37-51
        
        bool bounds_check_126290 = x_126288 && y_126289;
        
        // futhark/microgpt.fut:457:37-51
        
        bool index_certs_126291;
        
        if (!bounds_check_126290) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_126287, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:457:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:457:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139921 = 0; i_139921 < (int64_t) 16; i_139921++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_126298 = ((double *) wte_mem_141058.mem)[tmp_126287 * (int64_t) 16 + i_139921];
            
            ((double *) mem_141070)[i_139921] = lifted_lambda_res_126298;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141065, i_139925 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141070, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141081_cached_sizze_143601 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141081, &mem_141081_cached_sizze_143601, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141082_cached_sizze_143602 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141082, &mem_141082_cached_sizze_143602, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141083_cached_sizze_143603 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141083, &mem_141083_cached_sizze_143603, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139933 = 0; i_139933 < (int64_t) 16; i_139933++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_129650;
        double r_129652 = 0.0;
        
        for (int64_t i_129651 = 0; i_129651 < (int64_t) 16; i_129651++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_129653 = ((double *) wpe_mem_141056.mem)[i_139933 * (int64_t) 16 + i_129651];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_129654 = ((double *) mem_141065)[i_139933 * (int64_t) 16 + i_129651];
            
            // futhark/microgpt.fut:269:63-99
            
            double zp_res_129655 = zp_lhs_129653 + zp_rhs_129654;
            
            // futhark/microgpt.fut:269:79-142
            
            double zt_res_129656 = zp_res_129655 * zp_res_129655;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_129657 = r_129652 + zt_res_129656;
            double r_tmp_143152 = zp_res_129657;
            
            r_129652 = r_tmp_143152;
        }
        defunc_0_lifted_lambda_res_129650 = r_129652;
        // futhark/microgpt.fut:269:42-161
        
        double zs_res_129658 = defunc_0_lifted_lambda_res_129650 / 16.0;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_129665;
        double r_129667 = 0.0;
        
        for (int64_t i_129666 = 0; i_129666 < (int64_t) 16; i_129666++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_129668 = ((double *) wpe_mem_141056.mem)[i_139933 * (int64_t) 16 + i_129666];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_129669 = ((double *) mem_141065)[i_139933 * (int64_t) 16 + i_129666];
            
            // futhark/microgpt.fut:385:71-115
            
            double zp_res_129670 = zp_lhs_129668 + zp_rhs_129669;
            
            // futhark/microgpt.fut:385:91-166
            
            double zt_res_129671 = zp_res_129670 * zp_res_129670;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_129672 = r_129667 + zt_res_129671;
            double r_tmp_143153 = zp_res_129672;
            
            r_129667 = r_tmp_143153;
        }
        defunc_0_lifted_lambda_res_129665 = r_129667;
        // futhark/microgpt.fut:385:48-185
        
        double zs_res_129673 = defunc_0_lifted_lambda_res_129665 / 16.0;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_129683;
        double r_129685 = 0.0;
        
        for (int64_t i_129684 = 0; i_129684 < (int64_t) 16; i_129684++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_129686 = ((double *) wpe_mem_141056.mem)[i_139933 * (int64_t) 16 + i_129684];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_129687 = ((double *) mem_141065)[i_139933 * (int64_t) 16 + i_129684];
            
            // futhark/microgpt.fut:398:72-116
            
            double zp_res_129688 = zp_lhs_129686 + zp_rhs_129687;
            
            // futhark/microgpt.fut:398:92-167
            
            double zt_res_129689 = zp_res_129688 * zp_res_129688;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_129690 = r_129685 + zt_res_129689;
            double r_tmp_143154 = zp_res_129690;
            
            r_129685 = r_tmp_143154;
        }
        defunc_0_lifted_lambda_res_129683 = r_129685;
        // futhark/microgpt.fut:398:49-186
        
        double zs_res_129691 = defunc_0_lifted_lambda_res_129683 / 16.0;
        
        ((double *) mem_141081)[i_139933] = zs_res_129691;
        ((double *) mem_141082)[i_139933] = zs_res_129673;
        ((double *) mem_141083)[i_139933] = zs_res_129658;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141102_cached_sizze_143604 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141102, &mem_141102_cached_sizze_143604, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139939 = 0; i_139939 < (int64_t) 16; i_139939++) {
        // futhark/microgpt.fut:270:43-51
        
        double zp_lhs_126340 = ((double *) mem_141083)[i_139939];
        
        // futhark/microgpt.fut:270:43-79
        
        double zp_res_126341 = 1.0e-5 + zp_lhs_126340;
        
        // futhark/microgpt.fut:270:35-79
        
        double sqrt_res_126342 = futrts_sqrt64(zp_res_126341);
        
        ((double *) mem_141102)[i_139939] = sqrt_res_126342;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141109_cached_sizze_143605 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141109, &mem_141109_cached_sizze_143605, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141114_cached_sizze_143606 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141114, &mem_141114_cached_sizze_143606, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139947 = 0; i_139947 < (int64_t) 16; i_139947++) {
        // futhark/microgpt.fut:271:95-103
        
        double zs_rhs_126350 = ((double *) mem_141102)[i_139947];
        
        // futhark/microgpt.fut:271:87-103
        
        double zs_res_126351 = 1.0 / zs_rhs_126350;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139943 = 0; i_139943 < (int64_t) 16; i_139943++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_126358 = ((double *) wpe_mem_141056.mem)[i_139947 * (int64_t) 16 + i_139943];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_126359 = ((double *) mem_141065)[i_139947 * (int64_t) 16 + i_139943];
            
            // futhark/microgpt.fut:271:44-80
            
            double zp_res_126360 = zp_lhs_126358 + zp_rhs_126359;
            
            // futhark/microgpt.fut:271:60-103
            
            double zt_res_126361 = zs_res_126351 * zp_res_126360;
            
            ((double *) mem_141114)[i_139943] = zt_res_126361;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141109, i_139947 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141114, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141125_cached_sizze_143607 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141125, &mem_141125_cached_sizze_143607, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141130_cached_sizze_143608 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141130, &mem_141130_cached_sizze_143608, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139955 = 0; i_139955 < (int64_t) 16; i_139955++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139951 = 0; i_139951 < (int64_t) 16; i_139951++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_126376 = ((double *) mem_141109)[i_139955 * (int64_t) 16 + i_139951];
            
            ((double *) mem_141130)[i_139951] = lifted_lambda_res_126376;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141125, i_139955 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141130, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141141_cached_sizze_143609 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141141, &mem_141141_cached_sizze_143609, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141142_cached_sizze_143610 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141142, &mem_141142_cached_sizze_143610, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139961 = 0; i_139961 < (int64_t) 16; i_139961++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_129710;
        double r_129712 = 0.0;
        
        for (int64_t i_129711 = 0; i_129711 < (int64_t) 16; i_129711++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_129713 = ((double *) mem_141125)[i_139961 * (int64_t) 16 + i_129711];
            
            // futhark/microgpt.fut:273:65-102
            
            double zt_res_129714 = zt_lhs_129713 * zt_lhs_129713;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_129715 = r_129712 + zt_res_129714;
            double r_tmp_143162 = zp_res_129715;
            
            r_129712 = r_tmp_143162;
        }
        defunc_0_lifted_lambda_res_129710 = r_129712;
        // futhark/microgpt.fut:273:44-120
        
        double zs_res_129716 = defunc_0_lifted_lambda_res_129710 / 16.0;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_129723;
        double r_129725 = 0.0;
        
        for (int64_t i_129724 = 0; i_129724 < (int64_t) 16; i_129724++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_129726 = ((double *) mem_141125)[i_139961 * (int64_t) 16 + i_129724];
            
            // futhark/microgpt.fut:363:70-111
            
            double zt_res_129727 = zt_lhs_129726 * zt_lhs_129726;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_129728 = r_129725 + zt_res_129727;
            double r_tmp_143163 = zp_res_129728;
            
            r_129725 = r_tmp_143163;
        }
        defunc_0_lifted_lambda_res_129723 = r_129725;
        // futhark/microgpt.fut:363:48-129
        
        double zs_res_129729 = defunc_0_lifted_lambda_res_129723 / 16.0;
        
        ((double *) mem_141141)[i_139961] = zs_res_129729;
        ((double *) mem_141142)[i_139961] = zs_res_129716;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141155_cached_sizze_143611 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141155, &mem_141155_cached_sizze_143611, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139966 = 0; i_139966 < (int64_t) 16; i_139966++) {
        // futhark/microgpt.fut:274:45-55
        
        double zp_lhs_126399 = ((double *) mem_141142)[i_139966];
        
        // futhark/microgpt.fut:274:45-83
        
        double zp_res_126400 = 1.0e-5 + zp_lhs_126399;
        
        // futhark/microgpt.fut:274:37-83
        
        double sqrt_res_126401 = futrts_sqrt64(zp_res_126400);
        
        ((double *) mem_141155)[i_139966] = sqrt_res_126401;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141162_cached_sizze_143612 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141162, &mem_141162_cached_sizze_143612, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141167_cached_sizze_143613 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141167, &mem_141167_cached_sizze_143613, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139974 = 0; i_139974 < (int64_t) 16; i_139974++) {
        // futhark/microgpt.fut:275:76-86
        
        double zs_rhs_126409 = ((double *) mem_141155)[i_139974];
        
        // futhark/microgpt.fut:275:68-86
        
        double zs_res_126410 = 1.0 / zs_rhs_126409;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139970 = 0; i_139970 < (int64_t) 16; i_139970++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_126417 = ((double *) mem_141125)[i_139974 * (int64_t) 16 + i_139970];
            
            // futhark/microgpt.fut:275:46-86
            
            double zt_res_126418 = zs_res_126410 * zt_lhs_126417;
            
            ((double *) mem_141167)[i_139970] = zt_res_126418;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141162, i_139974 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141167, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141178_cached_sizze_143614 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141178, &mem_141178_cached_sizze_143614, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141183_cached_sizze_143615 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141183, &mem_141183_cached_sizze_143615, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_139982 = 0; i_139982 < (int64_t) 16; i_139982++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_139978 = 0; i_139978 < (int64_t) 16; i_139978++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_126433 = ((double *) mem_141162)[i_139982 * (int64_t) 16 + i_139978];
            
            ((double *) mem_141183)[i_139978] = lifted_lambda_res_126433;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141178, i_139982 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141183, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141194_cached_sizze_143616 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141194, &mem_141194_cached_sizze_143616, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141195_cached_sizze_143617 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141195, &mem_141195_cached_sizze_143617, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141196_cached_sizze_143618 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141196, &mem_141196_cached_sizze_143618, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141212_cached_sizze_143619 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141212, &mem_141212_cached_sizze_143619, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141213_cached_sizze_143620 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141213, &mem_141213_cached_sizze_143620, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141214_cached_sizze_143621 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141214, &mem_141214_cached_sizze_143621, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141227_cached_sizze_143622 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_141227, &mem_141227_cached_sizze_143622, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141228_cached_sizze_143623 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_141228, &mem_141228_cached_sizze_143623, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141229_cached_sizze_143624 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_141229, &mem_141229_cached_sizze_143624, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140010 = 0; i_140010 < (int64_t) 4; i_140010++) {
        // futhark/microgpt.fut:277:83-86
        
        int64_t zp_lhs_129810 = mul64((int64_t) 4, i_140010);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140000 = 0; i_140000 < (int64_t) 16; i_140000++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_139990 = 0; i_139990 < (int64_t) 4; i_139990++) {
                // futhark/microgpt.fut:277:88-95
                
                int64_t zt_lhs_133905 = add64(zp_lhs_129810, i_139990);
                
                // futhark/microgpt.fut:277:70-97
                
                bool x_133906 = sle64((int64_t) 0, zt_lhs_133905);
                
                // futhark/microgpt.fut:277:70-97
                
                bool y_133907 = slt64(zt_lhs_133905, (int64_t) 16);
                
                // futhark/microgpt.fut:277:70-97
                
                bool bounds_check_133908 = x_133906 && y_133907;
                
                // futhark/microgpt.fut:277:70-97
                
                bool index_certs_133909;
                
                if (!bounds_check_133908) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_133905, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:277:70-97\n   #1  futhark/microgpt.fut:61:46-49\n   #2  futhark/microgpt.fut:277:49-127\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:15:29-44\n   #8  futhark/microgpt.fut:4:11-25\n   #9  futhark/microgpt.fut:15:15-45\n   #10 futhark/microgpt.fut:277:12-129\n   #11 futhark/microgpt.fut:459:5-75\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_133910;
                double r_133912 = 0.0;
                
                for (int64_t i_133911 = 0; i_133911 < (int64_t) 16; i_133911++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_133913 = ((double *) wqry_mem_141057.mem)[zt_lhs_133905 * (int64_t) 16 + i_133911];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_133914 = ((double *) mem_141178)[i_140000 * (int64_t) 16 + i_133911];
                    
                    // futhark/microgpt.fut:277:70-125
                    
                    double zt_res_133915 = zt_lhs_133913 * zt_rhs_133914;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_133916 = r_133912 + zt_res_133915;
                    double r_tmp_143178 = zp_res_133916;
                    
                    r_133912 = r_tmp_143178;
                }
                defunc_0_lifted_lambda_res_133910 = r_133912;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_133924;
                double r_133926 = 0.0;
                
                for (int64_t i_133925 = 0; i_133925 < (int64_t) 16; i_133925++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_133927 = ((double *) wkey_mem_141054.mem)[zt_lhs_133905 * (int64_t) 16 + i_133925];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_133928 = ((double *) mem_141178)[i_140000 * (int64_t) 16 + i_133925];
                    
                    // futhark/microgpt.fut:278:70-125
                    
                    double zt_res_133929 = zt_lhs_133927 * zt_rhs_133928;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_133930 = r_133926 + zt_res_133929;
                    double r_tmp_143179 = zp_res_133930;
                    
                    r_133926 = r_tmp_143179;
                }
                defunc_0_lifted_lambda_res_133924 = r_133926;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_133941;
                double r_133943 = 0.0;
                
                for (int64_t i_133942 = 0; i_133942 < (int64_t) 16; i_133942++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_133944 = ((double *) wval_mem_141060.mem)[zt_lhs_133905 * (int64_t) 16 + i_133942];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_133945 = ((double *) mem_141178)[i_140000 * (int64_t) 16 + i_133942];
                    
                    // futhark/microgpt.fut:279:70-125
                    
                    double zt_res_133946 = zt_lhs_133944 * zt_rhs_133945;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_133947 = r_133943 + zt_res_133946;
                    double r_tmp_143180 = zp_res_133947;
                    
                    r_133943 = r_tmp_143180;
                }
                defunc_0_lifted_lambda_res_133941 = r_133943;
                ((double *) mem_141227)[i_139990] = defunc_0_lifted_lambda_res_133941;
                ((double *) mem_141228)[i_139990] = defunc_0_lifted_lambda_res_133924;
                ((double *) mem_141229)[i_139990] = defunc_0_lifted_lambda_res_133910;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141212, i_140000 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141227, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141213, i_140000 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141228, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141214, i_140000 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141229, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141194, i_140010 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_141212, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141195, i_140010 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_141213, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141196, i_140010 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_141214, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141275_cached_sizze_143625 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141275, &mem_141275_cached_sizze_143625, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141276_cached_sizze_143626 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141276, &mem_141276_cached_sizze_143626, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141277_cached_sizze_143627 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141277, &mem_141277_cached_sizze_143627, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141278_cached_sizze_143628 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141278, &mem_141278_cached_sizze_143628, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141299_cached_sizze_143629 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141299, &mem_141299_cached_sizze_143629, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141300_cached_sizze_143630 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141300, &mem_141300_cached_sizze_143630, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141301_cached_sizze_143631 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141301, &mem_141301_cached_sizze_143631, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141302_cached_sizze_143632 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141302, &mem_141302_cached_sizze_143632, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141319_cached_sizze_143633 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141319, &mem_141319_cached_sizze_143633, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141320_cached_sizze_143634 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141320, &mem_141320_cached_sizze_143634, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141321_cached_sizze_143635 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141321, &mem_141321_cached_sizze_143635, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141322_cached_sizze_143636 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141322, &mem_141322_cached_sizze_143636, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140048 = 0; i_140048 < (int64_t) 4; i_140048++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140035 = 0; i_140035 < (int64_t) 16; i_140035++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140022 = 0; i_140022 < (int64_t) 16; i_140022++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_134329;
                double r_134331 = 0.0;
                
                for (int64_t i_134330 = 0; i_134330 < (int64_t) 4; i_134330++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_134332 = ((double *) mem_141196)[i_140048 * (int64_t) 64 + i_140035 * (int64_t) 4 + i_134330];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_134333 = ((double *) mem_141195)[i_140048 * (int64_t) 64 + i_140022 * (int64_t) 4 + i_134330];
                    
                    // futhark/microgpt.fut:280:111-164
                    
                    double zt_res_134334 = zt_lhs_134332 * zt_rhs_134333;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_134335 = r_134331 + zt_res_134334;
                    double r_tmp_143193 = zp_res_134335;
                    
                    r_134331 = r_tmp_143193;
                }
                defunc_0_lifted_lambda_res_134329 = r_134331;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_134342;
                double r_134344 = 0.0;
                
                for (int64_t i_134343 = 0; i_134343 < (int64_t) 4; i_134343++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_134345 = ((double *) mem_141196)[i_140048 * (int64_t) 64 + i_140035 * (int64_t) 4 + i_134343];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_134346 = ((double *) mem_141195)[i_140048 * (int64_t) 64 + i_140022 * (int64_t) 4 + i_134343];
                    
                    // futhark/microgpt.fut:322:119-178
                    
                    double zt_res_134347 = zt_lhs_134345 * zt_rhs_134346;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_134348 = r_134344 + zt_res_134347;
                    double r_tmp_143194 = zp_res_134348;
                    
                    r_134344 = r_tmp_143194;
                }
                defunc_0_lifted_lambda_res_134342 = r_134344;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_134358;
                double r_134360 = 0.0;
                
                for (int64_t i_134359 = 0; i_134359 < (int64_t) 4; i_134359++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_134361 = ((double *) mem_141196)[i_140048 * (int64_t) 64 + i_140035 * (int64_t) 4 + i_134359];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_134362 = ((double *) mem_141195)[i_140048 * (int64_t) 64 + i_140022 * (int64_t) 4 + i_134359];
                    
                    // futhark/microgpt.fut:331:119-178
                    
                    double zt_res_134363 = zt_lhs_134361 * zt_rhs_134362;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_134364 = r_134360 + zt_res_134363;
                    double r_tmp_143195 = zp_res_134364;
                    
                    r_134360 = r_tmp_143195;
                }
                defunc_0_lifted_lambda_res_134358 = r_134360;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_134376;
                double r_134378 = 0.0;
                
                for (int64_t i_134377 = 0; i_134377 < (int64_t) 4; i_134377++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_134379 = ((double *) mem_141196)[i_140048 * (int64_t) 64 + i_140035 * (int64_t) 4 + i_134377];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_134380 = ((double *) mem_141195)[i_140048 * (int64_t) 64 + i_140022 * (int64_t) 4 + i_134377];
                    
                    // futhark/microgpt.fut:347:119-178
                    
                    double zt_res_134381 = zt_lhs_134379 * zt_rhs_134380;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_134382 = r_134378 + zt_res_134381;
                    double r_tmp_143196 = zp_res_134382;
                    
                    r_134378 = r_tmp_143196;
                }
                defunc_0_lifted_lambda_res_134376 = r_134378;
                ((double *) mem_141319)[i_140022] = defunc_0_lifted_lambda_res_134376;
                ((double *) mem_141320)[i_140022] = defunc_0_lifted_lambda_res_134358;
                ((double *) mem_141321)[i_140022] = defunc_0_lifted_lambda_res_134342;
                ((double *) mem_141322)[i_140022] = defunc_0_lifted_lambda_res_134329;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141299, i_140035 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141319, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141300, i_140035 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141320, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141301, i_140035 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141321, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141302, i_140035 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141322, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141275, i_140048 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141299, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141276, i_140048 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141300, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141277, i_140048 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141301, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141278, i_140048 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141302, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141383_cached_sizze_143637 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141383, &mem_141383_cached_sizze_143637, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141384_cached_sizze_143638 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141384, &mem_141384_cached_sizze_143638, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141385_cached_sizze_143639 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141385, &mem_141385_cached_sizze_143639, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141386_cached_sizze_143640 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141386, &mem_141386_cached_sizze_143640, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141407_cached_sizze_143641 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141407, &mem_141407_cached_sizze_143641, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141408_cached_sizze_143642 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141408, &mem_141408_cached_sizze_143642, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141409_cached_sizze_143643 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141409, &mem_141409_cached_sizze_143643, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141410_cached_sizze_143644 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141410, &mem_141410_cached_sizze_143644, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141427_cached_sizze_143645 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141427, &mem_141427_cached_sizze_143645, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141428_cached_sizze_143646 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141428, &mem_141428_cached_sizze_143646, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141429_cached_sizze_143647 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141429, &mem_141429_cached_sizze_143647, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141430_cached_sizze_143648 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141430, &mem_141430_cached_sizze_143648, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140087 = 0; i_140087 < (int64_t) 4; i_140087++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140074 = 0; i_140074 < (int64_t) 16; i_140074++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140061 = 0; i_140061 < (int64_t) 16; i_140061++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_134726 = ((double *) mem_141278)[i_140087 * (int64_t) 256 + i_140074 * (int64_t) 16 + i_140061];
                
                // futhark/microgpt.fut:281:55-93
                
                double zs_res_134727 = zs_lhs_134726 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_134728 = ((double *) mask_mem_141064.mem)[i_140074 * (int64_t) 16 + i_140061];
                
                // futhark/microgpt.fut:281:80-117
                
                double zp_res_134729 = zs_res_134727 + zp_rhs_134728;
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_134736 = ((double *) mem_141277)[i_140087 * (int64_t) 256 + i_140074 * (int64_t) 16 + i_140061];
                
                // futhark/microgpt.fut:323:59-101
                
                double zs_res_134737 = zs_lhs_134736 / 2.0;
                
                // futhark/microgpt.fut:323:88-127
                
                double zp_res_134739 = zp_rhs_134728 + zs_res_134737;
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_134749 = ((double *) mem_141276)[i_140087 * (int64_t) 256 + i_140074 * (int64_t) 16 + i_140061];
                
                // futhark/microgpt.fut:332:59-101
                
                double zs_res_134750 = zs_lhs_134749 / 2.0;
                
                // futhark/microgpt.fut:332:88-127
                
                double zp_res_134752 = zp_rhs_134728 + zs_res_134750;
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_134764 = ((double *) mem_141275)[i_140087 * (int64_t) 256 + i_140074 * (int64_t) 16 + i_140061];
                
                // futhark/microgpt.fut:348:59-101
                
                double zs_res_134765 = zs_lhs_134764 / 2.0;
                
                // futhark/microgpt.fut:348:88-127
                
                double zp_res_134767 = zp_rhs_134728 + zs_res_134765;
                
                ((double *) mem_141427)[i_140061] = zp_res_134767;
                ((double *) mem_141428)[i_140061] = zp_res_134752;
                ((double *) mem_141429)[i_140061] = zp_res_134739;
                ((double *) mem_141430)[i_140061] = zp_res_134729;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141407, i_140074 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141427, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141408, i_140074 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141428, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141409, i_140074 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141429, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141410, i_140074 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141430, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141383, i_140087 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141407, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141384, i_140087 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141408, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141385, i_140087 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141409, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141386, i_140087 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141410, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141491_cached_sizze_143649 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141491, &mem_141491_cached_sizze_143649, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141492_cached_sizze_143650 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141492, &mem_141492_cached_sizze_143650, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141493_cached_sizze_143651 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141493, &mem_141493_cached_sizze_143651, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141494_cached_sizze_143652 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141494, &mem_141494_cached_sizze_143652, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141495_cached_sizze_143653 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141495, &mem_141495_cached_sizze_143653, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141496_cached_sizze_143654 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141496, &mem_141496_cached_sizze_143654, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141497_cached_sizze_143655 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141497, &mem_141497_cached_sizze_143655, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141498_cached_sizze_143656 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141498, &mem_141498_cached_sizze_143656, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141531_cached_sizze_143657 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141531, &mem_141531_cached_sizze_143657, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141532_cached_sizze_143658 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141532, &mem_141532_cached_sizze_143658, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141533_cached_sizze_143659 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141533, &mem_141533_cached_sizze_143659, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141534_cached_sizze_143660 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141534, &mem_141534_cached_sizze_143660, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141535_cached_sizze_143661 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141535, &mem_141535_cached_sizze_143661, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141536_cached_sizze_143662 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141536, &mem_141536_cached_sizze_143662, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141537_cached_sizze_143663 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141537, &mem_141537_cached_sizze_143663, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141538_cached_sizze_143664 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141538, &mem_141538_cached_sizze_143664, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140140 = 0; i_140140 < (int64_t) 4; i_140140++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140115 = 0; i_140115 < (int64_t) 16; i_140115++) {
            // futhark/microgpt.fut:105:13-33
            
            double defunc_0_reduce_res_139383;
            double defunc_0_reduce_res_139384;
            double defunc_0_reduce_res_139385;
            double defunc_0_reduce_res_139386;
            double defunc_0_reduce_res_139387;
            double defunc_0_reduce_res_139388;
            double redout_140092;
            double redout_140093;
            double redout_140094;
            double redout_140095;
            double redout_140096;
            double redout_140097;
            
            redout_140092 = -INFINITY;
            redout_140093 = -INFINITY;
            redout_140094 = -INFINITY;
            redout_140095 = -INFINITY;
            redout_140096 = -INFINITY;
            redout_140097 = -INFINITY;
            for (int64_t i_140098 = 0; i_140098 < (int64_t) 16; i_140098++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_136079 = ((double *) mem_141386)[i_140140 * (int64_t) 256 + i_140115 * (int64_t) 16 + i_140098];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_136089 = ((double *) mem_141385)[i_140140 * (int64_t) 256 + i_140115 * (int64_t) 16 + i_140098];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_136108 = ((double *) mem_141384)[i_140140 * (int64_t) 256 + i_140115 * (int64_t) 16 + i_140098];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_136152 = ((double *) mem_141383)[i_140140 * (int64_t) 256 + i_140115 * (int64_t) 16 + i_140098];
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_135379 = fmax64(lifted_lambda_res_136079, redout_140092);
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_135398 = fmax64(lifted_lambda_res_136089, redout_140093);
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_135420 = fmax64(lifted_lambda_res_136108, redout_140094);
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_135445 = fmax64(lifted_lambda_res_136108, redout_140095);
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_135495 = fmax64(lifted_lambda_res_136152, redout_140096);
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_135526 = fmax64(lifted_lambda_res_136152, redout_140097);
                double redout_tmp_143225 = max_res_135379;
                double redout_tmp_143226 = max_res_135398;
                double redout_tmp_143227 = max_res_135420;
                double redout_tmp_143228 = max_res_135445;
                double redout_tmp_143229 = max_res_135495;
                double redout_tmp_143230 = max_res_135526;
                
                redout_140092 = redout_tmp_143225;
                redout_140093 = redout_tmp_143226;
                redout_140094 = redout_tmp_143227;
                redout_140095 = redout_tmp_143228;
                redout_140096 = redout_tmp_143229;
                redout_140097 = redout_tmp_143230;
            }
            defunc_0_reduce_res_139383 = redout_140092;
            defunc_0_reduce_res_139384 = redout_140093;
            defunc_0_reduce_res_139385 = redout_140094;
            defunc_0_reduce_res_139386 = redout_140095;
            defunc_0_reduce_res_139387 = redout_140096;
            defunc_0_reduce_res_139388 = redout_140097;
            // futhark/microgpt.fut:343:148-174
            
            double neg_res_135453 = -defunc_0_reduce_res_139386;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_135454;
            double r_135456 = 0.0;
            
            for (int64_t i_135455 = 0; i_135455 < (int64_t) 16; i_135455++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_135457 = ((double *) mem_141384)[i_140140 * (int64_t) 256 + i_140115 * (int64_t) 16 + i_135455];
                
                // futhark/microgpt.fut:343:114-174
                
                double zp_res_135458 = neg_res_135453 + zp_lhs_135457;
                
                // futhark/microgpt.fut:343:107-174
                
                double neg_res_135459 = -zp_res_135458;
                
                // futhark/microgpt.fut:100:42-54
                
                double max_res_135460 = fmax64(0.0, neg_res_135459);
                
                // futhark/microgpt.fut:100:35-54
                
                double sgn_res_135461 = fsignum64(max_res_135460);
                
                // futhark/microgpt.fut:343:88-177
                
                double neg_res_135462 = -sgn_res_135461;
                
                // futhark/microgpt.fut:343:79-178
                
                double zp_res_135463 = 1.0 + neg_res_135462;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_135464 = r_135456 + zp_res_135463;
                double r_tmp_143231 = zp_res_135464;
                
                r_135456 = r_tmp_143231;
            }
            defunc_0_lifted_lambda_res_135454 = r_135456;
            // futhark/microgpt.fut:343:48-181
            
            double zs_res_135465 = 1.0 / defunc_0_lifted_lambda_res_135454;
            
            // futhark/microgpt.fut:359:148-174
            
            double neg_res_135534 = -defunc_0_reduce_res_139388;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_135535;
            double r_135537 = 0.0;
            
            for (int64_t i_135536 = 0; i_135536 < (int64_t) 16; i_135536++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_135538 = ((double *) mem_141383)[i_140140 * (int64_t) 256 + i_140115 * (int64_t) 16 + i_135536];
                
                // futhark/microgpt.fut:359:114-174
                
                double zp_res_135539 = neg_res_135534 + zp_lhs_135538;
                
                // futhark/microgpt.fut:359:107-174
                
                double neg_res_135540 = -zp_res_135539;
                
                // futhark/microgpt.fut:100:42-54
                
                double max_res_135541 = fmax64(0.0, neg_res_135540);
                
                // futhark/microgpt.fut:100:35-54
                
                double sgn_res_135542 = fsignum64(max_res_135541);
                
                // futhark/microgpt.fut:359:88-177
                
                double neg_res_135543 = -sgn_res_135542;
                
                // futhark/microgpt.fut:359:79-178
                
                double zp_res_135544 = 1.0 + neg_res_135543;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_135545 = r_135537 + zp_res_135544;
                double r_tmp_143232 = zp_res_135545;
                
                r_135537 = r_tmp_143232;
            }
            defunc_0_lifted_lambda_res_135535 = r_135537;
            // futhark/microgpt.fut:359:48-181
            
            double zs_res_135546 = 1.0 / defunc_0_lifted_lambda_res_135535;
            
            ((double *) mem_141531)[i_140115] = zs_res_135546;
            ((double *) mem_141532)[i_140115] = defunc_0_reduce_res_139388;
            ((double *) mem_141533)[i_140115] = defunc_0_reduce_res_139387;
            ((double *) mem_141534)[i_140115] = zs_res_135465;
            ((double *) mem_141535)[i_140115] = defunc_0_reduce_res_139386;
            ((double *) mem_141536)[i_140115] = defunc_0_reduce_res_139385;
            ((double *) mem_141537)[i_140115] = defunc_0_reduce_res_139384;
            ((double *) mem_141538)[i_140115] = defunc_0_reduce_res_139383;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141491, i_140140 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141531, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141492, i_140140 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141532, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141493, i_140140 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141533, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141494, i_140140 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141534, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141495, i_140140 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141535, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141496, i_140140 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141536, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141497, i_140140 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141537, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141498, i_140140 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141538, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141619_cached_sizze_143665 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141619, &mem_141619_cached_sizze_143665, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141620_cached_sizze_143666 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141620, &mem_141620_cached_sizze_143666, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141621_cached_sizze_143667 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141621, &mem_141621_cached_sizze_143667, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141622_cached_sizze_143668 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141622, &mem_141622_cached_sizze_143668, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141643_cached_sizze_143669 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141643, &mem_141643_cached_sizze_143669, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141644_cached_sizze_143670 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141644, &mem_141644_cached_sizze_143670, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141645_cached_sizze_143671 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141645, &mem_141645_cached_sizze_143671, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141646_cached_sizze_143672 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141646, &mem_141646_cached_sizze_143672, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141663_cached_sizze_143673 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141663, &mem_141663_cached_sizze_143673, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141664_cached_sizze_143674 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141664, &mem_141664_cached_sizze_143674, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141665_cached_sizze_143675 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141665, &mem_141665_cached_sizze_143675, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141666_cached_sizze_143676 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141666, &mem_141666_cached_sizze_143676, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140183 = 0; i_140183 < (int64_t) 4; i_140183++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140170 = 0; i_140170 < (int64_t) 16; i_140170++) {
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_136368 = ((double *) mem_141498)[i_140183 * (int64_t) 16 + i_140170];
            
            // futhark/microgpt.fut:283:91-114
            
            double neg_res_136369 = -neg_arg0_136368;
            
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_136430 = ((double *) mem_141493)[i_140183 * (int64_t) 16 + i_140170];
            
            // futhark/microgpt.fut:352:99-125
            
            double neg_res_136431 = -neg_arg0_136430;
            
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_136407 = ((double *) mem_141496)[i_140183 * (int64_t) 16 + i_140170];
            
            // futhark/microgpt.fut:336:99-125
            
            double neg_res_136408 = -neg_arg0_136407;
            
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_136386 = ((double *) mem_141497)[i_140183 * (int64_t) 16 + i_140170];
            
            // futhark/microgpt.fut:325:99-125
            
            double neg_res_136387 = -neg_arg0_136386;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140157 = 0; i_140157 < (int64_t) 16; i_140157++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_136550 = ((double *) mem_141386)[i_140183 * (int64_t) 256 + i_140170 * (int64_t) 16 + i_140157];
                
                // futhark/microgpt.fut:283:61-114
                
                double zp_res_136551 = neg_res_136369 + zp_lhs_136550;
                
                // futhark/microgpt.fut:283:54-114
                
                double exp_res_136552 = futrts_exp64(zp_res_136551);
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_136559 = ((double *) mem_141385)[i_140183 * (int64_t) 256 + i_140170 * (int64_t) 16 + i_140157];
                
                // futhark/microgpt.fut:325:65-125
                
                double zp_res_136560 = neg_res_136387 + zp_lhs_136559;
                
                // futhark/microgpt.fut:325:58-125
                
                double exp_res_136561 = futrts_exp64(zp_res_136560);
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_136571 = ((double *) mem_141384)[i_140183 * (int64_t) 256 + i_140170 * (int64_t) 16 + i_140157];
                
                // futhark/microgpt.fut:336:65-125
                
                double zp_res_136572 = neg_res_136408 + zp_lhs_136571;
                
                // futhark/microgpt.fut:336:58-125
                
                double exp_res_136573 = futrts_exp64(zp_res_136572);
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_136585 = ((double *) mem_141383)[i_140183 * (int64_t) 256 + i_140170 * (int64_t) 16 + i_140157];
                
                // futhark/microgpt.fut:352:65-125
                
                double zp_res_136586 = neg_res_136431 + zp_lhs_136585;
                
                // futhark/microgpt.fut:352:58-125
                
                double exp_res_136587 = futrts_exp64(zp_res_136586);
                
                ((double *) mem_141663)[i_140157] = exp_res_136587;
                ((double *) mem_141664)[i_140157] = exp_res_136573;
                ((double *) mem_141665)[i_140157] = exp_res_136561;
                ((double *) mem_141666)[i_140157] = exp_res_136552;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141643, i_140170 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141663, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141644, i_140170 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141664, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141645, i_140170 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141665, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141646, i_140170 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141666, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141619, i_140183 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141643, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141620, i_140183 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141644, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141621, i_140183 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141645, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141622, i_140183 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141646, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141727_cached_sizze_143677 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141727, &mem_141727_cached_sizze_143677, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141728_cached_sizze_143678 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141728, &mem_141728_cached_sizze_143678, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141737_cached_sizze_143679 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141737, &mem_141737_cached_sizze_143679, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141738_cached_sizze_143680 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141738, &mem_141738_cached_sizze_143680, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140199 = 0; i_140199 < (int64_t) 4; i_140199++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140192 = 0; i_140192 < (int64_t) 16; i_140192++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_136619;
            double r_136621 = 0.0;
            
            for (int64_t i_136620 = 0; i_136620 < (int64_t) 16; i_136620++) {
                // futhark/microgpt.fut:61:46-49
                
                double lifted_lambda_res_136622 = ((double *) mem_141622)[i_140199 * (int64_t) 256 + i_140192 * (int64_t) 16 + i_136620];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_136623 = r_136621 + lifted_lambda_res_136622;
                double r_tmp_143249 = zp_res_136623;
                
                r_136621 = r_tmp_143249;
            }
            defunc_0_lifted_lambda_res_136619 = r_136621;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_136630;
            double r_136632 = 0.0;
            
            for (int64_t i_136631 = 0; i_136631 < (int64_t) 16; i_136631++) {
                // futhark/microgpt.fut:61:46-49
                
                double lifted_lambda_res_136633 = ((double *) mem_141621)[i_140199 * (int64_t) 256 + i_140192 * (int64_t) 16 + i_136631];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_136634 = r_136632 + lifted_lambda_res_136633;
                double r_tmp_143250 = zp_res_136634;
                
                r_136632 = r_tmp_143250;
            }
            defunc_0_lifted_lambda_res_136630 = r_136632;
            ((double *) mem_141737)[i_140192] = defunc_0_lifted_lambda_res_136630;
            ((double *) mem_141738)[i_140192] = defunc_0_lifted_lambda_res_136619;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141727, i_140199 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141737, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141728, i_140199 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141738, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141759_cached_sizze_143681 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141759, &mem_141759_cached_sizze_143681, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141760_cached_sizze_143682 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141760, &mem_141760_cached_sizze_143682, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141771_cached_sizze_143683 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141771, &mem_141771_cached_sizze_143683, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141772_cached_sizze_143684 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141772, &mem_141772_cached_sizze_143684, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141781_cached_sizze_143685 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141781, &mem_141781_cached_sizze_143685, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141782_cached_sizze_143686 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141782, &mem_141782_cached_sizze_143686, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140220 = 0; i_140220 < (int64_t) 4; i_140220++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140213 = 0; i_140213 < (int64_t) 16; i_140213++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_rhs_136654 = ((double *) mem_141728)[i_140220 * (int64_t) 16 + i_140213];
            
            // futhark/microgpt.fut:285:84-109
            
            double zs_res_136655 = 1.0 / zs_rhs_136654;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_rhs_136671 = ((double *) mem_141727)[i_140220 * (int64_t) 16 + i_140213];
            
            // futhark/microgpt.fut:327:92-120
            
            double zs_res_136672 = 1.0 / zs_rhs_136671;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140206 = 0; i_140206 < (int64_t) 16; i_140206++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_136699 = ((double *) mem_141622)[i_140220 * (int64_t) 256 + i_140213 * (int64_t) 16 + i_140206];
                
                // futhark/microgpt.fut:285:54-109
                
                double zt_res_136700 = zs_res_136655 * zt_lhs_136699;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_136707 = ((double *) mem_141621)[i_140220 * (int64_t) 256 + i_140213 * (int64_t) 16 + i_140206];
                
                // futhark/microgpt.fut:327:58-120
                
                double zt_res_136708 = zs_res_136672 * zt_lhs_136707;
                
                ((double *) mem_141781)[i_140206] = zt_res_136708;
                ((double *) mem_141782)[i_140206] = zt_res_136700;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141771, i_140213 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141781, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141772, i_140213 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141782, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141759, i_140220 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141771, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141760, i_140220 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141772, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141813_cached_sizze_143687 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141813, &mem_141813_cached_sizze_143687, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141814_cached_sizze_143688 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141814, &mem_141814_cached_sizze_143688, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141825_cached_sizze_143689 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141825, &mem_141825_cached_sizze_143689, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141826_cached_sizze_143690 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141826, &mem_141826_cached_sizze_143690, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141835_cached_sizze_143691 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141835, &mem_141835_cached_sizze_143691, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141836_cached_sizze_143692 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141836, &mem_141836_cached_sizze_143692, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140241 = 0; i_140241 < (int64_t) 4; i_140241++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140234 = 0; i_140234 < (int64_t) 16; i_140234++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140227 = 0; i_140227 < (int64_t) 16; i_140227++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_136771 = ((double *) mem_141760)[i_140241 * (int64_t) 256 + i_140234 * (int64_t) 16 + i_140227];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_136778 = ((double *) mem_141759)[i_140241 * (int64_t) 256 + i_140234 * (int64_t) 16 + i_140227];
                
                ((double *) mem_141835)[i_140227] = lifted_lambda_res_136778;
                ((double *) mem_141836)[i_140227] = lifted_lambda_res_136771;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141825, i_140234 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141835, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141826, i_140234 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141836, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141813, i_140241 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141825, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141814, i_140241 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_141826, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141867_cached_sizze_143693 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141867, &mem_141867_cached_sizze_143693, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141873_cached_sizze_143694 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141873, &mem_141873_cached_sizze_143694, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141878_cached_sizze_143695 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_141878, &mem_141878_cached_sizze_143695, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140254 = 0; i_140254 < (int64_t) 4; i_140254++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140250 = 0; i_140250 < (int64_t) 16; i_140250++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140246 = 0; i_140246 < (int64_t) 4; i_140246++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_126705;
                double r_126707 = 0.0;
                
                for (int64_t i_126706 = 0; i_126706 < (int64_t) 16; i_126706++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_126708 = ((double *) mem_141814)[i_140254 * (int64_t) 256 + i_140250 * (int64_t) 16 + i_126706];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_126709 = ((double *) mem_141194)[i_140254 * (int64_t) 64 + i_126706 * (int64_t) 4 + i_140246];
                    
                    // futhark/microgpt.fut:287:74-127
                    
                    double zt_res_126710 = zt_lhs_126708 * zt_rhs_126709;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_126711 = r_126707 + zt_res_126710;
                    double r_tmp_143266 = zp_res_126711;
                    
                    r_126707 = r_tmp_143266;
                }
                defunc_0_lifted_lambda_res_126705 = r_126707;
                ((double *) mem_141878)[i_140246] = defunc_0_lifted_lambda_res_126705;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_141873, i_140250 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141878, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_141867, i_140254 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_141873, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141894_cached_sizze_143696 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141894, &mem_141894_cached_sizze_143696, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141899_cached_sizze_143697 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141899, &mem_141899_cached_sizze_143697, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140262 = 0; i_140262 < (int64_t) 16; i_140262++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140258 = 0; i_140258 < (int64_t) 16; i_140258++) {
            // futhark/microgpt.fut:288:15-18
            
            int64_t tmp_126723 = sdiv64(i_140258, (int64_t) 4);
            
            // futhark/microgpt.fut:288:4-20
            
            bool x_126724 = sle64((int64_t) 0, tmp_126723);
            
            // futhark/microgpt.fut:288:4-20
            
            bool y_126725 = slt64(tmp_126723, (int64_t) 4);
            
            // futhark/microgpt.fut:288:4-20
            
            bool bounds_check_126726 = x_126724 && y_126725;
            
            // futhark/microgpt.fut:288:4-20
            
            bool index_certs_126727;
            
            if (!bounds_check_126726) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_126723, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:288:4-20\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:280:12-288:49\n   #6  futhark/microgpt.fut:459:5-75\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:288:35-38
            
            int64_t tmp_126728 = smod64(i_140258, (int64_t) 4);
            
            // futhark/microgpt.fut:288:4-40
            
            bool x_126729 = sle64((int64_t) 0, tmp_126728);
            
            // futhark/microgpt.fut:288:4-40
            
            bool y_126730 = slt64(tmp_126728, (int64_t) 4);
            
            // futhark/microgpt.fut:288:4-40
            
            bool bounds_check_126731 = x_126729 && y_126730;
            
            // futhark/microgpt.fut:288:4-40
            
            bool index_certs_126732;
            
            if (!bounds_check_126731) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_126728, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:288:4-40\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:280:12-288:49\n   #6  futhark/microgpt.fut:459:5-75\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_126733 = ((double *) mem_141867)[tmp_126723 * (int64_t) 64 + i_140262 * (int64_t) 4 + tmp_126728];
            
            ((double *) mem_141899)[i_140258] = lifted_lambda_res_126733;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141894, i_140262 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141899, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141910_cached_sizze_143698 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141910, &mem_141910_cached_sizze_143698, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141915_cached_sizze_143699 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141915, &mem_141915_cached_sizze_143699, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140270 = 0; i_140270 < (int64_t) 16; i_140270++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140266 = 0; i_140266 < (int64_t) 16; i_140266++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126748;
            double r_126750 = 0.0;
            
            for (int64_t i_126749 = 0; i_126749 < (int64_t) 16; i_126749++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_126751 = ((double *) wout_mem_141055.mem)[i_140266 * (int64_t) 16 + i_126749];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_126752 = ((double *) mem_141894)[i_140270 * (int64_t) 16 + i_126749];
                
                // futhark/microgpt.fut:289:64-104
                
                double zt_res_126753 = zt_lhs_126751 * zt_rhs_126752;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126754 = r_126750 + zt_res_126753;
                double r_tmp_143271 = zp_res_126754;
                
                r_126750 = r_tmp_143271;
            }
            defunc_0_lifted_lambda_res_126748 = r_126750;
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_126755 = ((double *) mem_141125)[i_140270 * (int64_t) 16 + i_140266];
            
            // futhark/microgpt.fut:289:43-128
            
            double zp_res_126756 = defunc_0_lifted_lambda_res_126748 + zp_rhs_126755;
            
            ((double *) mem_141915)[i_140266] = zp_res_126756;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141910, i_140270 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141915, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141926_cached_sizze_143700 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141926, &mem_141926_cached_sizze_143700, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141927_cached_sizze_143701 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141927, &mem_141927_cached_sizze_143701, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140276 = 0; i_140276 < (int64_t) 16; i_140276++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_131787;
        double r_131789 = 0.0;
        
        for (int64_t i_131788 = 0; i_131788 < (int64_t) 16; i_131788++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_131790 = ((double *) mem_141910)[i_140276 * (int64_t) 16 + i_131788];
            
            // futhark/microgpt.fut:290:66-105
            
            double zt_res_131791 = zt_lhs_131790 * zt_lhs_131790;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_131792 = r_131789 + zt_res_131791;
            double r_tmp_143274 = zp_res_131792;
            
            r_131789 = r_tmp_143274;
        }
        defunc_0_lifted_lambda_res_131787 = r_131789;
        // futhark/microgpt.fut:290:45-123
        
        double zs_res_131793 = defunc_0_lifted_lambda_res_131787 / 16.0;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_131800;
        double r_131802 = 0.0;
        
        for (int64_t i_131801 = 0; i_131801 < (int64_t) 16; i_131801++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_131803 = ((double *) mem_141910)[i_140276 * (int64_t) 16 + i_131801];
            
            // futhark/microgpt.fut:315:70-113
            
            double zt_res_131804 = zt_lhs_131803 * zt_lhs_131803;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_131805 = r_131802 + zt_res_131804;
            double r_tmp_143275 = zp_res_131805;
            
            r_131802 = r_tmp_143275;
        }
        defunc_0_lifted_lambda_res_131800 = r_131802;
        // futhark/microgpt.fut:315:48-131
        
        double zs_res_131806 = defunc_0_lifted_lambda_res_131800 / 16.0;
        
        ((double *) mem_141926)[i_140276] = zs_res_131806;
        ((double *) mem_141927)[i_140276] = zs_res_131793;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141940_cached_sizze_143702 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141940, &mem_141940_cached_sizze_143702, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140281 = 0; i_140281 < (int64_t) 16; i_140281++) {
        // futhark/microgpt.fut:291:45-55
        
        double zp_lhs_126779 = ((double *) mem_141927)[i_140281];
        
        // futhark/microgpt.fut:291:45-83
        
        double zp_res_126780 = 1.0e-5 + zp_lhs_126779;
        
        // futhark/microgpt.fut:291:37-83
        
        double sqrt_res_126781 = futrts_sqrt64(zp_res_126780);
        
        ((double *) mem_141940)[i_140281] = sqrt_res_126781;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141947_cached_sizze_143703 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141947, &mem_141947_cached_sizze_143703, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141952_cached_sizze_143704 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141952, &mem_141952_cached_sizze_143704, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140289 = 0; i_140289 < (int64_t) 16; i_140289++) {
        // futhark/microgpt.fut:292:77-87
        
        double zs_rhs_126789 = ((double *) mem_141940)[i_140289];
        
        // futhark/microgpt.fut:292:69-87
        
        double zs_res_126790 = 1.0 / zs_rhs_126789;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140285 = 0; i_140285 < (int64_t) 16; i_140285++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_126797 = ((double *) mem_141910)[i_140289 * (int64_t) 16 + i_140285];
            
            // futhark/microgpt.fut:292:46-87
            
            double zt_res_126798 = zs_res_126790 * zt_lhs_126797;
            
            ((double *) mem_141952)[i_140285] = zt_res_126798;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141947, i_140289 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141952, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141963_cached_sizze_143705 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_141963, &mem_141963_cached_sizze_143705, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141968_cached_sizze_143706 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_141968, &mem_141968_cached_sizze_143706, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140297 = 0; i_140297 < (int64_t) 16; i_140297++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140293 = 0; i_140293 < (int64_t) 16; i_140293++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_126813 = ((double *) mem_141947)[i_140297 * (int64_t) 16 + i_140293];
            
            ((double *) mem_141968)[i_140293] = lifted_lambda_res_126813;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141963, i_140297 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141968, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141979_cached_sizze_143707 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141979, &mem_141979_cached_sizze_143707, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141984_cached_sizze_143708 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_141984, &mem_141984_cached_sizze_143708, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140305 = 0; i_140305 < (int64_t) 16; i_140305++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140301 = 0; i_140301 < (int64_t) 64; i_140301++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126829;
            double r_126831 = 0.0;
            
            for (int64_t i_126830 = 0; i_126830 < (int64_t) 16; i_126830++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_126832 = ((double *) wup_mem_141059.mem)[i_140301 * (int64_t) 16 + i_126830];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_126833 = ((double *) mem_141963)[i_140305 * (int64_t) 16 + i_126830];
                
                // futhark/microgpt.fut:294:63-102
                
                double zt_res_126834 = zt_lhs_126832 * zt_rhs_126833;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126835 = r_126831 + zt_res_126834;
                double r_tmp_143283 = zp_res_126835;
                
                r_126831 = r_tmp_143283;
            }
            defunc_0_lifted_lambda_res_126829 = r_126831;
            ((double *) mem_141984)[i_140301] = defunc_0_lifted_lambda_res_126829;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141979, i_140305 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_141984, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_141995_cached_sizze_143709 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_141995, &mem_141995_cached_sizze_143709, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142000_cached_sizze_143710 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_142000, &mem_142000_cached_sizze_143710, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140313 = 0; i_140313 < (int64_t) 16; i_140313++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140309 = 0; i_140309 < (int64_t) 64; i_140309++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_126850 = ((double *) mem_141979)[i_140313 * (int64_t) 64 + i_140309];
            
            // futhark/microgpt.fut:295:41-69
            
            double max_res_126851 = fmax64(0.0, max_arg0_126850);
            
            ((double *) mem_142000)[i_140309] = max_res_126851;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_141995, i_140313 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142000, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142011_cached_sizze_143711 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142011, &mem_142011_cached_sizze_143711, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142016_cached_sizze_143712 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142016, &mem_142016_cached_sizze_143712, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140321 = 0; i_140321 < (int64_t) 16; i_140321++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140317 = 0; i_140317 < (int64_t) 16; i_140317++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126866;
            double r_126868 = 0.0;
            
            for (int64_t i_126867 = 0; i_126867 < (int64_t) 64; i_126867++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_126869 = ((double *) wdown_mem_141053.mem)[i_140317 * (int64_t) 64 + i_126867];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_126870 = ((double *) mem_141995)[i_140321 * (int64_t) 64 + i_126867];
                
                // futhark/microgpt.fut:296:64-105
                
                double zt_res_126871 = zt_lhs_126869 * zt_rhs_126870;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126872 = r_126868 + zt_res_126871;
                double r_tmp_143288 = zp_res_126872;
                
                r_126868 = r_tmp_143288;
            }
            defunc_0_lifted_lambda_res_126866 = r_126868;
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_126873 = ((double *) mem_141910)[i_140321 * (int64_t) 16 + i_140317];
            
            // futhark/microgpt.fut:296:43-130
            
            double zp_res_126874 = defunc_0_lifted_lambda_res_126866 + zp_rhs_126873;
            
            ((double *) mem_142016)[i_140317] = zp_res_126874;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142011, i_140321 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142016, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142027_cached_sizze_143713 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_142027, &mem_142027_cached_sizze_143713, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142032_cached_sizze_143714 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_142032, &mem_142032_cached_sizze_143714, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140329 = 0; i_140329 < (int64_t) 16; i_140329++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140325 = 0; i_140325 < (int64_t) 27; i_140325++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_126890;
            double r_126892 = 0.0;
            
            for (int64_t i_126891 = 0; i_126891 < (int64_t) 16; i_126891++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_126893 = ((double *) wvoc_mem_141061.mem)[i_140325 * (int64_t) 16 + i_126891];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_126894 = ((double *) mem_142011)[i_140329 * (int64_t) 16 + i_126891];
                
                // futhark/microgpt.fut:297:63-103
                
                double zt_res_126895 = zt_lhs_126893 * zt_rhs_126894;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_126896 = r_126892 + zt_res_126895;
                double r_tmp_143291 = zp_res_126896;
                
                r_126892 = r_tmp_143291;
            }
            defunc_0_lifted_lambda_res_126890 = r_126892;
            ((double *) mem_142032)[i_140325] = defunc_0_lifted_lambda_res_126890;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142027, i_140329 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142032, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142043_cached_sizze_143715 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_142043, &mem_142043_cached_sizze_143715, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142044_cached_sizze_143716 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_142044, &mem_142044_cached_sizze_143716, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142045_cached_sizze_143717 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_142045, &mem_142045_cached_sizze_143717, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142046_cached_sizze_143718 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_142046, &mem_142046_cached_sizze_143718, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:105:13-33
    if (mem_142064_cached_sizze_143719 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_142064, &mem_142064_cached_sizze_143719, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142069_cached_sizze_143720 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_142069, &mem_142069_cached_sizze_143720, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140359 = 0; i_140359 < (int64_t) 16; i_140359++) {
        // futhark/microgpt.fut:105:13-33
        
        double defunc_0_reduce_res_139486;
        double defunc_0_reduce_res_139487;
        double redout_140346;
        double redout_140347;
        
        redout_140346 = -INFINITY;
        redout_140347 = -INFINITY;
        for (int64_t i_140349 = 0; i_140349 < (int64_t) 27; i_140349++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_136949 = ((double *) mem_142027)[i_140359 * (int64_t) 27 + i_140349];
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140343 = 0; i_140343 < (int64_t) 27; i_140343++) {
                // futhark/microgpt.fut:302:55-306:90
                
                bool cond_136958 = i_140343 == i_140349;
                
                // futhark/microgpt.fut:302:55-306:90
                
                double lifted_lambda_res_136959;
                
                if (cond_136958) {
                    // futhark/microgpt.fut:105:13-33
                    
                    double defunc_0_reduce_res_139433;
                    double redout_140331 = -INFINITY;
                    
                    for (int64_t i_140332 = 0; i_140332 < (int64_t) 27; i_140332++) {
                        // futhark/microgpt.fut:4:11-25
                        
                        double lifted_lambda_res_139439 = ((double *) mem_142027)[i_140359 * (int64_t) 27 + i_140332];
                        
                        // futhark/microgpt.fut:105:13-33
                        
                        double max_res_139442 = fmax64(lifted_lambda_res_139439, redout_140331);
                        double redout_tmp_143300 = max_res_139442;
                        
                        redout_140331 = redout_tmp_143300;
                    }
                    defunc_0_reduce_res_139433 = redout_140331;
                    // futhark/microgpt.fut:303:67-76
                    
                    double neg_res_139444 = -defunc_0_reduce_res_139433;
                    
                    // futhark/microgpt.fut:4:11-25
                    if (mem_142073_cached_sizze_143721 < (int64_t) 216) {
                        err = lexical_realloc(ctx, &mem_142073, &mem_142073_cached_sizze_143721, (int64_t) 216);
                        if (err != FUTHARK_SUCCESS)
                            goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    for (int64_t i_140335 = 0; i_140335 < (int64_t) 27; i_140335++) {
                        // futhark/microgpt.fut:4:11-25
                        
                        double zp_lhs_139451 = ((double *) mem_142027)[i_140359 * (int64_t) 27 + i_140335];
                        
                        // futhark/microgpt.fut:303:44-76
                        
                        double zp_res_139452 = neg_res_139444 + zp_lhs_139451;
                        
                        // futhark/microgpt.fut:303:37-76
                        
                        double exp_res_139453 = futrts_exp64(zp_res_139452);
                        
                        ((double *) mem_142073)[i_140335] = exp_res_139453;
                    }
                    // futhark/microgpt.fut:61:13-49
                    
                    double defunc_0_lifted_lambda_res_139456;
                    double r_139458 = 0.0;
                    
                    for (int64_t i_139457 = 0; i_139457 < (int64_t) 27; i_139457++) {
                        // futhark/microgpt.fut:304:36-46
                        
                        double lifted_lambda_res_139459 = ((double *) mem_142073)[i_139457];
                        
                        // futhark/microgpt.fut:61:40-49
                        
                        double zp_res_139460 = r_139458 + lifted_lambda_res_139459;
                        double r_tmp_143302 = zp_res_139460;
                        
                        r_139458 = r_tmp_143302;
                    }
                    defunc_0_lifted_lambda_res_139456 = r_139458;
                    // futhark/microgpt.fut:305:55-66
                    
                    double zs_res_139461 = 1.0 / defunc_0_lifted_lambda_res_139456;
                    
                    // futhark/microgpt.fut:4:11-25
                    if (mem_142080_cached_sizze_143722 < (int64_t) 216) {
                        err = lexical_realloc(ctx, &mem_142080, &mem_142080_cached_sizze_143722, (int64_t) 216);
                        if (err != FUTHARK_SUCCESS)
                            goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    for (int64_t i_140339 = 0; i_140339 < (int64_t) 27; i_140339++) {
                        // futhark/microgpt.fut:305:38-49
                        
                        double zt_lhs_139468 = ((double *) mem_142073)[i_140339];
                        
                        // futhark/microgpt.fut:305:38-66
                        
                        double zt_res_139469 = zs_res_139461 * zt_lhs_139468;
                        
                        ((double *) mem_142080)[i_140339] = zt_res_139469;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_139476 = ((double *) target_mem_141063.mem)[i_140359 * (int64_t) 27 + i_140349];
                    
                    // futhark/microgpt.fut:306:7-49
                    
                    double zt_res_139477 = -6.25e-2 * zt_rhs_139476;
                    
                    // futhark/microgpt.fut:306:64-74
                    
                    double zs_rhs_139482 = ((double *) mem_142080)[i_140343];
                    
                    // futhark/microgpt.fut:306:56-74
                    
                    double zs_res_139483 = 1.0 / zs_rhs_139482;
                    
                    // futhark/microgpt.fut:306:25-74
                    
                    double zt_res_139484 = zt_res_139477 * zs_res_139483;
                    
                    lifted_lambda_res_136959 = zt_res_139484;
                } else {
                    lifted_lambda_res_136959 = 0.0;
                }
                ((double *) mem_142069)[i_140343] = lifted_lambda_res_136959;
            }
            // futhark/microgpt.fut:105:13-33
            
            double max_res_131943 = fmax64(lifted_lambda_res_136949, redout_140346);
            
            // futhark/microgpt.fut:105:13-33
            
            double max_res_132034 = fmax64(lifted_lambda_res_136949, redout_140347);
            
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_142064, i_140349 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142069, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            
            double redout_tmp_143296 = max_res_131943;
            double redout_tmp_143297 = max_res_132034;
            
            redout_140346 = redout_tmp_143296;
            redout_140347 = redout_tmp_143297;
        }
        defunc_0_reduce_res_139486 = redout_140346;
        defunc_0_reduce_res_139487 = redout_140347;
        // futhark/microgpt.fut:4:11-25
        for (int64_t nest_i_143304 = 0; nest_i_143304 < (int64_t) 27; nest_i_143304++) {
            ((double *) mem_142046)[i_140359 * (int64_t) 27 + nest_i_143304] = defunc_0_reduce_res_139486;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t nest_i_143305 = 0; nest_i_143305 < (int64_t) 27; nest_i_143305++) {
            ((double *) mem_142044)[i_140359 * (int64_t) 27 + nest_i_143305] = defunc_0_reduce_res_139487;
        }
        // futhark/microgpt.fut:311:139-164
        
        double neg_res_132045 = -defunc_0_reduce_res_139487;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_132046;
        double r_132048 = 0.0;
        
        for (int64_t i_132047 = 0; i_132047 < (int64_t) 27; i_132047++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_132049 = ((double *) mem_142027)[i_140359 * (int64_t) 27 + i_132047];
            
            // futhark/microgpt.fut:311:114-164
            
            double zp_res_132050 = neg_res_132045 + zp_lhs_132049;
            
            // futhark/microgpt.fut:311:107-164
            
            double neg_res_132051 = -zp_res_132050;
            
            // futhark/microgpt.fut:100:42-54
            
            double max_res_132052 = fmax64(0.0, neg_res_132051);
            
            // futhark/microgpt.fut:100:35-54
            
            double sgn_res_132053 = fsignum64(max_res_132052);
            
            // futhark/microgpt.fut:311:88-167
            
            double neg_res_132054 = -sgn_res_132053;
            
            // futhark/microgpt.fut:311:79-168
            
            double zp_res_132055 = 1.0 + neg_res_132054;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_132056 = r_132048 + zp_res_132055;
            double r_tmp_143306 = zp_res_132056;
            
            r_132048 = r_tmp_143306;
        }
        defunc_0_lifted_lambda_res_132046 = r_132048;
        // futhark/microgpt.fut:311:48-171
        
        double zs_res_132057 = 1.0 / defunc_0_lifted_lambda_res_132046;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t nest_i_143307 = 0; nest_i_143307 < (int64_t) 27; nest_i_143307++) {
            ((double *) mem_142043)[i_140359 * (int64_t) 27 + nest_i_143307] = zs_res_132057;
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_142045, i_140359 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_142064, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142114_cached_sizze_143723 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_142114, &mem_142114_cached_sizze_143723, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142120_cached_sizze_143724 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_142120, &mem_142120_cached_sizze_143724, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142125_cached_sizze_143725 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_142125, &mem_142125_cached_sizze_143725, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140374 = 0; i_140374 < (int64_t) 16; i_140374++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140370 = 0; i_140370 < (int64_t) 27; i_140370++) {
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_126932 = ((double *) mem_142046)[i_140374 * (int64_t) 27 + i_140370];
            
            // futhark/microgpt.fut:300:85-108
            
            double neg_res_126933 = -neg_arg0_126932;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140366 = 0; i_140366 < (int64_t) 27; i_140366++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_126940 = ((double *) mem_142027)[i_140374 * (int64_t) 27 + i_140366];
                
                // futhark/microgpt.fut:300:62-108
                
                double zp_res_126941 = neg_res_126933 + zp_lhs_126940;
                
                // futhark/microgpt.fut:300:55-108
                
                double exp_res_126942 = futrts_exp64(zp_res_126941);
                
                ((double *) mem_142125)[i_140366] = exp_res_126942;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_142120, i_140370 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142125, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_142114, i_140374 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_142120, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142141_cached_sizze_143726 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_142141, &mem_142141_cached_sizze_143726, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142142_cached_sizze_143727 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_142142, &mem_142142_cached_sizze_143727, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142151_cached_sizze_143728 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_142151, &mem_142151_cached_sizze_143728, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142152_cached_sizze_143729 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_142152, &mem_142152_cached_sizze_143729, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140387 = 0; i_140387 < (int64_t) 16; i_140387++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140380 = 0; i_140380 < (int64_t) 27; i_140380++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137323;
            double r_137325 = 0.0;
            
            for (int64_t i_137324 = 0; i_137324 < (int64_t) 27; i_137324++) {
                // futhark/microgpt.fut:61:46-49
                
                double lifted_lambda_res_137326 = ((double *) mem_142114)[i_140387 * (int64_t) 729 + i_140380 * (int64_t) 27 + i_137324];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137327 = r_137325 + lifted_lambda_res_137326;
                double r_tmp_143315 = zp_res_137327;
                
                r_137325 = r_tmp_143315;
            }
            defunc_0_lifted_lambda_res_137323 = r_137325;
            // futhark/microgpt.fut:307:153-196
            
            double zt_res_137335 = defunc_0_lifted_lambda_res_137323 * defunc_0_lifted_lambda_res_137323;
            
            // futhark/microgpt.fut:307:144-196
            
            double zs_res_137336 = 1.0 / zt_res_137335;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137337;
            double r_137339 = 0.0;
            
            for (int64_t i_137338 = 0; i_137338 < (int64_t) 27; i_137338++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_137340 = ((double *) mem_142045)[i_140387 * (int64_t) 729 + i_140380 * (int64_t) 27 + i_137338];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_137341 = ((double *) mem_142114)[i_140387 * (int64_t) 729 + i_140380 * (int64_t) 27 + i_137338];
                
                // futhark/microgpt.fut:307:78-137
                
                double zt_res_137342 = zt_lhs_137340 * zt_rhs_137341;
                
                // futhark/microgpt.fut:307:106-196
                
                double zt_res_137343 = zs_res_137336 * zt_res_137342;
                
                // futhark/microgpt.fut:307:70-196
                
                double neg_res_137344 = -zt_res_137343;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137345 = r_137339 + neg_res_137344;
                double r_tmp_143316 = zp_res_137345;
                
                r_137339 = r_tmp_143316;
            }
            defunc_0_lifted_lambda_res_137337 = r_137339;
            ((double *) mem_142151)[i_140380] = defunc_0_lifted_lambda_res_137337;
            ((double *) mem_142152)[i_140380] = defunc_0_lifted_lambda_res_137323;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142141, i_140387 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142151, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142142, i_140387 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142152, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142173_cached_sizze_143730 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_142173, &mem_142173_cached_sizze_143730, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142179_cached_sizze_143731 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_142179, &mem_142179_cached_sizze_143731, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142184_cached_sizze_143732 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_142184, &mem_142184_cached_sizze_143732, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140400 = 0; i_140400 < (int64_t) 16; i_140400++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140396 = 0; i_140396 < (int64_t) 27; i_140396++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_rhs_127072 = ((double *) mem_142142)[i_140400 * (int64_t) 27 + i_140396];
            
            // futhark/microgpt.fut:308:92-119
            
            double zs_res_127073 = 1.0 / zs_rhs_127072;
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_127074 = ((double *) mem_142141)[i_140400 * (int64_t) 27 + i_140396];
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140392 = 0; i_140392 < (int64_t) 27; i_140392++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_127081 = ((double *) mem_142045)[i_140400 * (int64_t) 729 + i_140396 * (int64_t) 27 + i_140392];
                
                // futhark/microgpt.fut:308:59-119
                
                double zt_res_127082 = zs_res_127073 * zt_lhs_127081;
                
                // futhark/microgpt.fut:308:87-145
                
                double zp_res_127083 = zp_rhs_127074 + zt_res_127082;
                
                ((double *) mem_142184)[i_140392] = zp_res_127083;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_142179, i_140396 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142184, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_142173, i_140400 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_142179, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142200_cached_sizze_143733 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_142200, &mem_142200_cached_sizze_143733, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142205_cached_sizze_143734 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_142205, &mem_142205_cached_sizze_143734, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140408 = 0; i_140408 < (int64_t) 16; i_140408++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140404 = 0; i_140404 < (int64_t) 27; i_140404++) {
            double f_elem_127096 = ((double *) mem_142046)[i_140408 * (int64_t) 27 + i_140404];
            
            // futhark/microgpt.fut:309:110-135
            
            double neg_res_127101 = -f_elem_127096;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_127102;
            double r_127104 = 0.0;
            
            for (int64_t i_127103 = 0; i_127103 < (int64_t) 27; i_127103++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_127105 = ((double *) mem_142027)[i_140408 * (int64_t) 27 + i_127103];
                
                // futhark/microgpt.fut:309:85-135
                
                double zp_res_127106 = neg_res_127101 + zp_lhs_127105;
                
                // futhark/microgpt.fut:309:78-135
                
                double exp_res_127107 = futrts_exp64(zp_res_127106);
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_127108 = ((double *) mem_142173)[i_140408 * (int64_t) 729 + i_140404 * (int64_t) 27 + i_127103];
                
                // futhark/microgpt.fut:309:78-170
                
                double zt_res_127109 = exp_res_127107 * zt_rhs_127108;
                
                // futhark/microgpt.fut:309:70-170
                
                double neg_res_127110 = -zt_res_127109;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_127111 = r_127104 + neg_res_127110;
                double r_tmp_143322 = zp_res_127111;
                
                r_127104 = r_tmp_143322;
            }
            defunc_0_lifted_lambda_res_127102 = r_127104;
            ((double *) mem_142205)[i_140404] = defunc_0_lifted_lambda_res_127102;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142200, i_140408 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142205, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142216_cached_sizze_143735 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_142216, &mem_142216_cached_sizze_143735, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142221_cached_sizze_143736 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_142221, &mem_142221_cached_sizze_143736, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140416 = 0; i_140416 < (int64_t) 16; i_140416++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140412 = 0; i_140412 < (int64_t) 27; i_140412++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_127166;
            double r_127168 = 0.0;
            
            for (int64_t i_127167 = 0; i_127167 < (int64_t) 16; i_127167++) {
                // futhark/microgpt.fut:312:78-203
                
                bool cond_127169 = i_140416 == i_127167;
                
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_127173;
                
                if (cond_127169) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double x_139496 = ((double *) mem_142027)[i_127167 * (int64_t) 27 + i_140412];
                    
                    zp_lhs_127173 = x_139496;
                } else {
                    zp_lhs_127173 = 0.0;
                }
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_127175;
                
                if (cond_127169) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double x_139497 = ((double *) mem_142027)[i_127167 * (int64_t) 27 + i_140412];
                    
                    zp_lhs_127175 = x_139497;
                } else {
                    zp_lhs_127175 = 0.0;
                }
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_127177;
                double r_127179 = 0.0;
                
                for (int64_t i_127178 = 0; i_127178 < (int64_t) 27; i_127178++) {
                    // futhark/microgpt.fut:312:78-203
                    
                    double zp_lhs_127180;
                    
                    if (cond_127169) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double neg_arg0_139498 = ((double *) mem_142046)[i_127167 * (int64_t) 27 + i_127178];
                        
                        // futhark/microgpt.fut:312:137-160
                        
                        double neg_res_139499 = -neg_arg0_139498;
                        
                        // futhark/microgpt.fut:312:114-160
                        
                        double zp_res_139500 = zp_lhs_127173 + neg_res_139499;
                        
                        // futhark/microgpt.fut:312:107-160
                        
                        double exp_res_139501 = futrts_exp64(zp_res_139500);
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_139502 = ((double *) mem_142173)[i_127167 * (int64_t) 729 + i_127178 * (int64_t) 27 + i_140412];
                        
                        // futhark/microgpt.fut:312:107-192
                        
                        double zt_res_139503 = exp_res_139501 * zt_rhs_139502;
                        
                        zp_lhs_127180 = zt_res_139503;
                    } else {
                        zp_lhs_127180 = 0.0;
                    }
                    // futhark/microgpt.fut:312:210-383
                    
                    double zp_rhs_127187;
                    
                    if (cond_127169) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_139504 = ((double *) mem_142200)[i_127167 * (int64_t) 27 + i_127178];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double neg_arg0_139505 = ((double *) mem_142044)[i_127167 * (int64_t) 27 + i_127178];
                        
                        // futhark/microgpt.fut:312:320-343
                        
                        double neg_res_139506 = -neg_arg0_139505;
                        
                        // futhark/microgpt.fut:312:297-343
                        
                        double zp_res_139507 = zp_lhs_127175 + neg_res_139506;
                        
                        // futhark/microgpt.fut:312:290-343
                        
                        double neg_res_139508 = -zp_res_139507;
                        
                        // futhark/microgpt.fut:100:42-54
                        
                        double max_res_139509 = fmax64(0.0, neg_res_139508);
                        
                        // futhark/microgpt.fut:100:35-54
                        
                        double sgn_res_139510 = fsignum64(max_res_139509);
                        
                        // futhark/microgpt.fut:312:271-346
                        
                        double neg_res_139511 = -sgn_res_139510;
                        
                        // futhark/microgpt.fut:312:262-347
                        
                        double zp_res_139512 = 1.0 + neg_res_139511;
                        
                        // futhark/microgpt.fut:312:239-347
                        
                        double zt_res_139513 = zt_lhs_139504 * zp_res_139512;
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_139514 = ((double *) mem_142043)[i_127167 * (int64_t) 27 + i_127178];
                        
                        // futhark/microgpt.fut:312:257-372
                        
                        double zt_res_139515 = zt_res_139513 * zt_rhs_139514;
                        
                        zp_rhs_127187 = zt_res_139515;
                    } else {
                        zp_rhs_127187 = 0.0;
                    }
                    // futhark/microgpt.fut:312:78-383
                    
                    double zp_res_127200 = zp_lhs_127180 + zp_rhs_127187;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_127201 = r_127179 + zp_res_127200;
                    double r_tmp_143326 = zp_res_127201;
                    
                    r_127179 = r_tmp_143326;
                }
                defunc_0_lifted_lambda_res_127177 = r_127179;
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_127202 = r_127168 + defunc_0_lifted_lambda_res_127177;
                double r_tmp_143325 = zp_res_127202;
                
                r_127168 = r_tmp_143325;
            }
            defunc_0_lifted_lambda_res_127166 = r_127168;
            ((double *) mem_142221)[i_140412] = defunc_0_lifted_lambda_res_127166;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142216, i_140416 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142221, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142232_cached_sizze_143737 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142232, &mem_142232_cached_sizze_143737, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142237_cached_sizze_143738 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142237, &mem_142237_cached_sizze_143738, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140424 = 0; i_140424 < (int64_t) 16; i_140424++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140420 = 0; i_140420 < (int64_t) 16; i_140420++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_127217;
            double r_127219 = 0.0;
            
            for (int64_t i_127218 = 0; i_127218 < (int64_t) 27; i_127218++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_127220 = ((double *) mem_142216)[i_140424 * (int64_t) 27 + i_127218];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_127221 = ((double *) wvoc_mem_141061.mem)[i_127218 * (int64_t) 16 + i_140420];
                
                // futhark/microgpt.fut:313:67-111
                
                double zt_res_127222 = zt_lhs_127220 * zt_rhs_127221;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_127223 = r_127219 + zt_res_127222;
                double r_tmp_143329 = zp_res_127223;
                
                r_127219 = r_tmp_143329;
            }
            defunc_0_lifted_lambda_res_127217 = r_127219;
            ((double *) mem_142237)[i_140420] = defunc_0_lifted_lambda_res_127217;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142232, i_140424 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142237, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_142248, (int64_t) 8192, "mem_142248")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142249_cached_sizze_143739 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_142249, &mem_142249_cached_sizze_143739, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142258_cached_sizze_143740 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_142258, &mem_142258_cached_sizze_143740, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142259_cached_sizze_143741 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_142259, &mem_142259_cached_sizze_143741, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140437 = 0; i_140437 < (int64_t) 16; i_140437++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140430 = 0; i_140430 < (int64_t) 64; i_140430++) {
            // futhark/microgpt.fut:4:11-25
            
            double indicatorp_arg0_137390 = ((double *) mem_141979)[i_140437 * (int64_t) 64 + i_140430];
            
            // futhark/microgpt.fut:100:42-54
            
            double max_res_137391 = fmax64(0.0, indicatorp_arg0_137390);
            
            // futhark/microgpt.fut:100:35-54
            
            double sgn_res_137392 = fsignum64(max_res_137391);
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137393;
            double r_137395 = 0.0;
            
            for (int64_t i_137394 = 0; i_137394 < (int64_t) 16; i_137394++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_137396 = ((double *) mem_142232)[i_140437 * (int64_t) 16 + i_137394];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_137397 = ((double *) wdown_mem_141053.mem)[i_137394 * (int64_t) 64 + i_140430];
                
                // futhark/microgpt.fut:314:105-151
                
                double zt_res_137398 = zt_lhs_137396 * zt_rhs_137397;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137399 = r_137395 + zt_res_137398;
                double r_tmp_143334 = zp_res_137399;
                
                r_137395 = r_tmp_143334;
            }
            defunc_0_lifted_lambda_res_137393 = r_137395;
            // futhark/microgpt.fut:314:46-153
            
            double zt_res_137400 = sgn_res_137392 * defunc_0_lifted_lambda_res_137393;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137407;
            double r_137409 = 0.0;
            
            for (int64_t i_137408 = 0; i_137408 < (int64_t) 16; i_137408++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_137410 = ((double *) mem_142232)[i_137408 * (int64_t) 16 + i_140437];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_137411 = ((double *) mem_141995)[i_137408 * (int64_t) 64 + i_140430];
                
                // futhark/microgpt.fut:396:69-113
                
                double zt_res_137412 = zt_lhs_137410 * zt_rhs_137411;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137413 = r_137409 + zt_res_137412;
                double r_tmp_143335 = zp_res_137413;
                
                r_137409 = r_tmp_143335;
            }
            defunc_0_lifted_lambda_res_137407 = r_137409;
            ((double *) mem_142258)[i_140430] = defunc_0_lifted_lambda_res_137407;
            ((double *) mem_142259)[i_140430] = zt_res_137400;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142248.mem, i_140437 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142258, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142249, i_140437 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142259, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142280_cached_sizze_143742 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142280, &mem_142280_cached_sizze_143742, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142285_cached_sizze_143743 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142285, &mem_142285_cached_sizze_143743, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140446 = 0; i_140446 < (int64_t) 16; i_140446++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140442 = 0; i_140442 < (int64_t) 16; i_140442++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_127287;
            double r_127289 = 0.0;
            
            for (int64_t i_127288 = 0; i_127288 < (int64_t) 64; i_127288++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_127290 = ((double *) mem_142249)[i_140446 * (int64_t) 64 + i_127288];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_127291 = ((double *) wup_mem_141059.mem)[i_127288 * (int64_t) 16 + i_140442];
                
                // futhark/microgpt.fut:317:71-115
                
                double zt_res_127292 = zt_lhs_127290 * zt_rhs_127291;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_127293 = r_127289 + zt_res_127292;
                double r_tmp_143338 = zp_res_127293;
                
                r_127289 = r_tmp_143338;
            }
            defunc_0_lifted_lambda_res_127287 = r_127289;
            ((double *) mem_142285)[i_140442] = defunc_0_lifted_lambda_res_127287;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142280, i_140446 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142285, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142296_cached_sizze_143744 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142296, &mem_142296_cached_sizze_143744, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142297_cached_sizze_143745 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142297, &mem_142297_cached_sizze_143745, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140452 = 0; i_140452 < (int64_t) 16; i_140452++) {
        // futhark/microgpt.fut:316:47-59
        
        double zp_lhs_129456 = ((double *) mem_141926)[i_140452];
        
        // futhark/microgpt.fut:316:47-87
        
        double zp_res_129457 = 1.0e-5 + zp_lhs_129456;
        
        // futhark/microgpt.fut:316:39-87
        
        double sqrt_res_129458 = futrts_sqrt64(zp_res_129457);
        
        // futhark/microgpt.fut:318:129-158
        
        double zt_res_129466 = sqrt_res_129458 * sqrt_res_129458;
        
        // futhark/microgpt.fut:318:120-158
        
        double zs_res_129467 = 1.0 / zt_res_129466;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_129468;
        double r_129470 = 0.0;
        
        for (int64_t i_129469 = 0; i_129469 < (int64_t) 16; i_129469++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_129471 = ((double *) mem_142280)[i_140452 * (int64_t) 16 + i_129469];
            
            // futhark/microgpt.fut:61:46-49
            
            double zt_rhs_129472 = ((double *) mem_141910)[i_140452 * (int64_t) 16 + i_129469];
            
            // futhark/microgpt.fut:318:69-113
            
            double zt_res_129473 = zt_lhs_129471 * zt_rhs_129472;
            
            // futhark/microgpt.fut:318:90-158
            
            double zt_res_129474 = zs_res_129467 * zt_res_129473;
            
            // futhark/microgpt.fut:318:61-158
            
            double neg_res_129475 = -zt_res_129474;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_129476 = r_129470 + neg_res_129475;
            double r_tmp_143341 = zp_res_129476;
            
            r_129470 = r_tmp_143341;
        }
        defunc_0_lifted_lambda_res_129468 = r_129470;
        ((double *) mem_142296)[i_140452] = defunc_0_lifted_lambda_res_129468;
        ((double *) mem_142297)[i_140452] = sqrt_res_129458;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142310_cached_sizze_143746 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142310, &mem_142310_cached_sizze_143746, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140457 = 0; i_140457 < (int64_t) 16; i_140457++) {
        // futhark/microgpt.fut:319:39-51
        
        double zt_lhs_127321 = ((double *) mem_142296)[i_140457];
        
        // futhark/microgpt.fut:319:93-105
        
        double zp_lhs_127322 = ((double *) mem_141926)[i_140457];
        
        // futhark/microgpt.fut:319:93-133
        
        double zp_res_127323 = 1.0e-5 + zp_lhs_127322;
        
        // futhark/microgpt.fut:319:85-133
        
        double sqrt_res_127324 = futrts_sqrt64(zp_res_127323);
        
        // futhark/microgpt.fut:319:71-135
        
        double zt_res_127325 = 2.0 * sqrt_res_127324;
        
        // futhark/microgpt.fut:319:57-135
        
        double zs_res_127326 = 1.0 / zt_res_127325;
        
        // futhark/microgpt.fut:319:39-135
        
        double zt_res_127327 = zt_lhs_127321 * zs_res_127326;
        
        ((double *) mem_142310)[i_140457] = zt_res_127327;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142317_cached_sizze_143747 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142317, &mem_142317_cached_sizze_143747, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142322_cached_sizze_143748 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142322, &mem_142322_cached_sizze_143748, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140465 = 0; i_140465 < (int64_t) 16; i_140465++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140461 = 0; i_140461 < (int64_t) 16; i_140461++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_127341 = ((double *) mem_142232)[i_140465 * (int64_t) 16 + i_140461];
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_127342;
            double r_127344 = 0.0;
            
            for (int64_t i_127343 = 0; i_127343 < (int64_t) 16; i_127343++) {
                // futhark/microgpt.fut:320:86-174
                
                bool cond_127345 = i_140465 == i_127343;
                
                // futhark/microgpt.fut:320:86-174
                
                double zp_lhs_127346;
                
                if (cond_127345) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_139521 = ((double *) mem_142280)[i_127343 * (int64_t) 16 + i_140461];
                    
                    // futhark/microgpt.fut:320:150-162
                    
                    double zs_rhs_139522 = ((double *) mem_142297)[i_127343];
                    
                    // futhark/microgpt.fut:320:142-162
                    
                    double zs_res_139523 = 1.0 / zs_rhs_139522;
                    
                    // futhark/microgpt.fut:320:116-162
                    
                    double zt_res_139524 = zt_lhs_139521 * zs_res_139523;
                    
                    zp_lhs_127346 = zt_res_139524;
                } else {
                    zp_lhs_127346 = 0.0;
                }
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_127351;
                double r_127353 = 0.0;
                
                for (int64_t i_127352 = 0; i_127352 < (int64_t) 16; i_127352++) {
                    // futhark/microgpt.fut:320:204-339
                    
                    double zp_lhs_127354;
                    
                    if (cond_127345) {
                        // futhark/microgpt.fut:320:234-328
                        
                        bool cond_139529 = i_140461 == i_127352;
                        
                        // futhark/microgpt.fut:320:234-328
                        
                        double zp_lhs_t_res_139530;
                        
                        if (cond_139529) {
                            // futhark/microgpt.fut:320:265-277
                            
                            double zs_lhs_139531 = ((double *) mem_142310)[i_127343];
                            
                            // futhark/microgpt.fut:320:265-292
                            
                            double zs_res_139532 = zs_lhs_139531 / 16.0;
                            
                            // futhark/microgpt.fut:61:46-49
                            
                            double zt_rhs_139533 = ((double *) mem_141910)[i_127343 * (int64_t) 16 + i_140461];
                            
                            // futhark/microgpt.fut:320:278-317
                            
                            double zt_res_139534 = zs_res_139532 * zt_rhs_139533;
                            
                            zp_lhs_t_res_139530 = zt_res_139534;
                        } else {
                            zp_lhs_t_res_139530 = 0.0;
                        }
                        zp_lhs_127354 = zp_lhs_t_res_139530;
                    } else {
                        zp_lhs_127354 = 0.0;
                    }
                    // futhark/microgpt.fut:320:346-481
                    
                    double zp_rhs_127361;
                    
                    if (cond_127345) {
                        // futhark/microgpt.fut:320:376-470
                        
                        bool cond_139539 = i_140461 == i_127352;
                        
                        // futhark/microgpt.fut:320:376-470
                        
                        double zp_rhs_t_res_139540;
                        
                        if (cond_139539) {
                            // futhark/microgpt.fut:320:407-419
                            
                            double zs_lhs_139541 = ((double *) mem_142310)[i_127343];
                            
                            // futhark/microgpt.fut:320:407-434
                            
                            double zs_res_139542 = zs_lhs_139541 / 16.0;
                            
                            // futhark/microgpt.fut:61:46-49
                            
                            double zt_rhs_139543 = ((double *) mem_141910)[i_127343 * (int64_t) 16 + i_140461];
                            
                            // futhark/microgpt.fut:320:420-459
                            
                            double zt_res_139544 = zs_res_139542 * zt_rhs_139543;
                            
                            zp_rhs_t_res_139540 = zt_res_139544;
                        } else {
                            zp_rhs_t_res_139540 = 0.0;
                        }
                        zp_rhs_127361 = zp_rhs_t_res_139540;
                    } else {
                        zp_rhs_127361 = 0.0;
                    }
                    // futhark/microgpt.fut:320:204-481
                    
                    double zp_res_127368 = zp_lhs_127354 + zp_rhs_127361;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_127369 = r_127353 + zp_res_127368;
                    double r_tmp_143346 = zp_res_127369;
                    
                    r_127353 = r_tmp_143346;
                }
                defunc_0_lifted_lambda_res_127351 = r_127353;
                // futhark/microgpt.fut:320:86-484
                
                double zp_res_127370 = zp_lhs_127346 + defunc_0_lifted_lambda_res_127351;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_127371 = r_127344 + zp_res_127370;
                double r_tmp_143345 = zp_res_127371;
                
                r_127344 = r_tmp_143345;
            }
            defunc_0_lifted_lambda_res_127342 = r_127344;
            // futhark/microgpt.fut:320:37-487
            
            double zp_res_127372 = zp_lhs_127341 + defunc_0_lifted_lambda_res_127342;
            
            ((double *) mem_142322)[i_140461] = zp_res_127372;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142317, i_140465 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142322, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142333_cached_sizze_143749 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142333, &mem_142333_cached_sizze_143749, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142339_cached_sizze_143750 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_142339, &mem_142339_cached_sizze_143750, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142344_cached_sizze_143751 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_142344, &mem_142344_cached_sizze_143751, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140477 = 0; i_140477 < (int64_t) 4; i_140477++) {
        // futhark/microgpt.fut:321:122-125
        
        int64_t zp_lhs_127377 = mul64((int64_t) 4, i_140477);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140473 = 0; i_140473 < (int64_t) 16; i_140473++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140469 = 0; i_140469 < (int64_t) 4; i_140469++) {
                // futhark/microgpt.fut:321:127-135
                
                int64_t zt_rhs_127386 = add64(zp_lhs_127377, i_140469);
                
                // futhark/microgpt.fut:321:100-137
                
                bool x_127387 = sle64((int64_t) 0, zt_rhs_127386);
                
                // futhark/microgpt.fut:321:100-137
                
                bool y_127388 = slt64(zt_rhs_127386, (int64_t) 16);
                
                // futhark/microgpt.fut:321:100-137
                
                bool bounds_check_127389 = x_127387 && y_127388;
                
                // futhark/microgpt.fut:321:100-137
                
                bool index_certs_127390;
                
                if (!bounds_check_127389) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_rhs_127386, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:321:100-137\n   #1  futhark/microgpt.fut:61:46-49\n   #2  futhark/microgpt.fut:321:53-139\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:15:29-44\n   #8  futhark/microgpt.fut:4:11-25\n   #9  futhark/microgpt.fut:15:15-45\n   #10 futhark/microgpt.fut:321:13-141\n   #11 futhark/microgpt.fut:459:5-75\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_127391;
                double r_127393 = 0.0;
                
                for (int64_t i_127392 = 0; i_127392 < (int64_t) 16; i_127392++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_127394 = ((double *) mem_142317)[i_140473 * (int64_t) 16 + i_127392];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_127395 = ((double *) wout_mem_141055.mem)[i_127392 * (int64_t) 16 + zt_rhs_127386];
                    
                    // futhark/microgpt.fut:321:75-137
                    
                    double zt_res_127396 = zt_lhs_127394 * zt_rhs_127395;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_127397 = r_127393 + zt_res_127396;
                    double r_tmp_143350 = zp_res_127397;
                    
                    r_127393 = r_tmp_143350;
                }
                defunc_0_lifted_lambda_res_127391 = r_127393;
                ((double *) mem_142344)[i_140469] = defunc_0_lifted_lambda_res_127391;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_142339, i_140473 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142344, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_142333, i_140477 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_142339, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142360_cached_sizze_143752 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142360, &mem_142360_cached_sizze_143752, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142361_cached_sizze_143753 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142361, &mem_142361_cached_sizze_143753, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142362_cached_sizze_143754 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142362, &mem_142362_cached_sizze_143754, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142378_cached_sizze_143755 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_142378, &mem_142378_cached_sizze_143755, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142379_cached_sizze_143756 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_142379, &mem_142379_cached_sizze_143756, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142380_cached_sizze_143757 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_142380, &mem_142380_cached_sizze_143757, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142393_cached_sizze_143758 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_142393, &mem_142393_cached_sizze_143758, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142394_cached_sizze_143759 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_142394, &mem_142394_cached_sizze_143759, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140502 = 0; i_140502 < (int64_t) 4; i_140502++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140492 = 0; i_140492 < (int64_t) 16; i_140492++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140483 = 0; i_140483 < (int64_t) 4; i_140483++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_137594 = ((double *) mem_142333)[i_140502 * (int64_t) 64 + i_140492 * (int64_t) 4 + i_140483];
                
                ((double *) mem_142393)[i_140483] = lifted_lambda_res_137594;
                ((double *) mem_142394)[i_140483] = lifted_lambda_res_137594;
            }
            // futhark/microgpt.fut:4:11-25
            // futhark/microgpt.fut:4:11-25
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_142380, i_140492 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142394, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_142378, i_140492 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142393, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_142379, i_140492 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142394, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_142360, i_140502 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_142378, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_142361, i_140502 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_142379, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_142362, i_140502 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_142380, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142435_cached_sizze_143760 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_142435, &mem_142435_cached_sizze_143760, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142436_cached_sizze_143761 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_142436, &mem_142436_cached_sizze_143761, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142447_cached_sizze_143762 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142447, &mem_142447_cached_sizze_143762, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142448_cached_sizze_143763 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142448, &mem_142448_cached_sizze_143763, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142457_cached_sizze_143764 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142457, &mem_142457_cached_sizze_143764, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142458_cached_sizze_143765 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142458, &mem_142458_cached_sizze_143765, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140524 = 0; i_140524 < (int64_t) 4; i_140524++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140517 = 0; i_140517 < (int64_t) 16; i_140517++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140510 = 0; i_140510 < (int64_t) 16; i_140510++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_137924;
                double r_137926 = 0.0;
                
                for (int64_t i_137925 = 0; i_137925 < (int64_t) 4; i_137925++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_137927 = ((double *) mem_142361)[i_140524 * (int64_t) 64 + i_140517 * (int64_t) 4 + i_137925];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_137928 = ((double *) mem_141194)[i_140524 * (int64_t) 64 + i_140510 * (int64_t) 4 + i_137925];
                    
                    // futhark/microgpt.fut:334:79-139
                    
                    double zt_res_137929 = zt_lhs_137927 * zt_rhs_137928;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_137930 = r_137926 + zt_res_137929;
                    double r_tmp_143365 = zp_res_137930;
                    
                    r_137926 = r_tmp_143365;
                }
                defunc_0_lifted_lambda_res_137924 = r_137926;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_137937;
                double r_137939 = 0.0;
                
                for (int64_t i_137938 = 0; i_137938 < (int64_t) 4; i_137938++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_137940 = ((double *) mem_142360)[i_140524 * (int64_t) 64 + i_140517 * (int64_t) 4 + i_137938];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_137941 = ((double *) mem_141194)[i_140524 * (int64_t) 64 + i_140510 * (int64_t) 4 + i_137938];
                    
                    // futhark/microgpt.fut:350:79-139
                    
                    double zt_res_137942 = zt_lhs_137940 * zt_rhs_137941;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_137943 = r_137939 + zt_res_137942;
                    double r_tmp_143366 = zp_res_137943;
                    
                    r_137939 = r_tmp_143366;
                }
                defunc_0_lifted_lambda_res_137937 = r_137939;
                ((double *) mem_142457)[i_140510] = defunc_0_lifted_lambda_res_137937;
                ((double *) mem_142458)[i_140510] = defunc_0_lifted_lambda_res_137924;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_142447, i_140517 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142457, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_142448, i_140517 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142458, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_142435, i_140524 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_142447, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_142436, i_140524 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_142448, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142489_cached_sizze_143766 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_142489, &mem_142489_cached_sizze_143766, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142490_cached_sizze_143767 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_142490, &mem_142490_cached_sizze_143767, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142501_cached_sizze_143768 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142501, &mem_142501_cached_sizze_143768, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142502_cached_sizze_143769 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142502, &mem_142502_cached_sizze_143769, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142511_cached_sizze_143770 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142511, &mem_142511_cached_sizze_143770, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142512_cached_sizze_143771 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142512, &mem_142512_cached_sizze_143771, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140545 = 0; i_140545 < (int64_t) 4; i_140545++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140538 = 0; i_140538 < (int64_t) 16; i_140538++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140531 = 0; i_140531 < (int64_t) 16; i_140531++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_138176 = ((double *) mem_142436)[i_140545 * (int64_t) 256 + i_140538 * (int64_t) 16 + i_140531];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_138183 = ((double *) mem_142435)[i_140545 * (int64_t) 256 + i_140538 * (int64_t) 16 + i_140531];
                
                ((double *) mem_142511)[i_140531] = lifted_lambda_res_138183;
                ((double *) mem_142512)[i_140531] = lifted_lambda_res_138176;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_142501, i_140538 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142511, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_142502, i_140538 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142512, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_142489, i_140545 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_142501, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_142490, i_140545 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_142502, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142543_cached_sizze_143772 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_142543, &mem_142543_cached_sizze_143772, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142544_cached_sizze_143773 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_142544, &mem_142544_cached_sizze_143773, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142545_cached_sizze_143774 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_142545, &mem_142545_cached_sizze_143774, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142546_cached_sizze_143775 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_142546, &mem_142546_cached_sizze_143775, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142563_cached_sizze_143776 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142563, &mem_142563_cached_sizze_143776, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142564_cached_sizze_143777 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142564, &mem_142564_cached_sizze_143777, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142565_cached_sizze_143778 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142565, &mem_142565_cached_sizze_143778, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142566_cached_sizze_143779 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142566, &mem_142566_cached_sizze_143779, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140569 = 0; i_140569 < (int64_t) 4; i_140569++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140556 = 0; i_140556 < (int64_t) 16; i_140556++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_138056;
            double r_138058 = 0.0;
            
            for (int64_t i_138057 = 0; i_138057 < (int64_t) 16; i_138057++) {
                // futhark/microgpt.fut:61:46-49
                
                double lifted_lambda_res_138059 = ((double *) mem_141620)[i_140569 * (int64_t) 256 + i_140556 * (int64_t) 16 + i_138057];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_138060 = r_138058 + lifted_lambda_res_138059;
                double r_tmp_143381 = zp_res_138060;
                
                r_138058 = r_tmp_143381;
            }
            defunc_0_lifted_lambda_res_138056 = r_138058;
            // futhark/microgpt.fut:339:155-200
            
            double zt_res_138068 = defunc_0_lifted_lambda_res_138056 * defunc_0_lifted_lambda_res_138056;
            
            // futhark/microgpt.fut:339:146-200
            
            double zs_res_138069 = 1.0 / zt_res_138068;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_138070;
            double r_138072 = 0.0;
            
            for (int64_t i_138071 = 0; i_138071 < (int64_t) 16; i_138071++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_138073 = ((double *) mem_142490)[i_140569 * (int64_t) 256 + i_140556 * (int64_t) 16 + i_138071];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_138074 = ((double *) mem_141620)[i_140569 * (int64_t) 256 + i_140556 * (int64_t) 16 + i_138071];
                
                // futhark/microgpt.fut:339:78-139
                
                double zt_res_138075 = zt_lhs_138073 * zt_rhs_138074;
                
                // futhark/microgpt.fut:339:107-200
                
                double zt_res_138076 = zs_res_138069 * zt_res_138075;
                
                // futhark/microgpt.fut:339:70-200
                
                double neg_res_138077 = -zt_res_138076;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_138078 = r_138072 + neg_res_138077;
                double r_tmp_143382 = zp_res_138078;
                
                r_138072 = r_tmp_143382;
            }
            defunc_0_lifted_lambda_res_138070 = r_138072;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_138089;
            double r_138091 = 0.0;
            
            for (int64_t i_138090 = 0; i_138090 < (int64_t) 16; i_138090++) {
                // futhark/microgpt.fut:61:46-49
                
                double lifted_lambda_res_138092 = ((double *) mem_141619)[i_140569 * (int64_t) 256 + i_140556 * (int64_t) 16 + i_138090];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_138093 = r_138091 + lifted_lambda_res_138092;
                double r_tmp_143383 = zp_res_138093;
                
                r_138091 = r_tmp_143383;
            }
            defunc_0_lifted_lambda_res_138089 = r_138091;
            // futhark/microgpt.fut:355:155-200
            
            double zt_res_138101 = defunc_0_lifted_lambda_res_138089 * defunc_0_lifted_lambda_res_138089;
            
            // futhark/microgpt.fut:355:146-200
            
            double zs_res_138102 = 1.0 / zt_res_138101;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_138103;
            double r_138105 = 0.0;
            
            for (int64_t i_138104 = 0; i_138104 < (int64_t) 16; i_138104++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_138106 = ((double *) mem_142489)[i_140569 * (int64_t) 256 + i_140556 * (int64_t) 16 + i_138104];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_138107 = ((double *) mem_141619)[i_140569 * (int64_t) 256 + i_140556 * (int64_t) 16 + i_138104];
                
                // futhark/microgpt.fut:355:78-139
                
                double zt_res_138108 = zt_lhs_138106 * zt_rhs_138107;
                
                // futhark/microgpt.fut:355:107-200
                
                double zt_res_138109 = zs_res_138102 * zt_res_138108;
                
                // futhark/microgpt.fut:355:70-200
                
                double neg_res_138110 = -zt_res_138109;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_138111 = r_138105 + neg_res_138110;
                double r_tmp_143384 = zp_res_138111;
                
                r_138105 = r_tmp_143384;
            }
            defunc_0_lifted_lambda_res_138103 = r_138105;
            ((double *) mem_142563)[i_140556] = defunc_0_lifted_lambda_res_138103;
            ((double *) mem_142564)[i_140556] = defunc_0_lifted_lambda_res_138089;
            ((double *) mem_142565)[i_140556] = defunc_0_lifted_lambda_res_138070;
            ((double *) mem_142566)[i_140556] = defunc_0_lifted_lambda_res_138056;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142543, i_140569 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142563, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142544, i_140569 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142564, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142545, i_140569 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142565, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142546, i_140569 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142566, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142607_cached_sizze_143780 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_142607, &mem_142607_cached_sizze_143780, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142608_cached_sizze_143781 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_142608, &mem_142608_cached_sizze_143781, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142619_cached_sizze_143782 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142619, &mem_142619_cached_sizze_143782, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142620_cached_sizze_143783 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142620, &mem_142620_cached_sizze_143783, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142629_cached_sizze_143784 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142629, &mem_142629_cached_sizze_143784, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142630_cached_sizze_143785 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142630, &mem_142630_cached_sizze_143785, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140592 = 0; i_140592 < (int64_t) 4; i_140592++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140585 = 0; i_140585 < (int64_t) 16; i_140585++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_rhs_138207 = ((double *) mem_142546)[i_140592 * (int64_t) 16 + i_140585];
            
            // futhark/microgpt.fut:340:93-121
            
            double zs_res_138208 = 1.0 / zs_rhs_138207;
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_138209 = ((double *) mem_142545)[i_140592 * (int64_t) 16 + i_140585];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_138228 = ((double *) mem_142543)[i_140592 * (int64_t) 16 + i_140585];
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_rhs_138226 = ((double *) mem_142544)[i_140592 * (int64_t) 16 + i_140585];
            
            // futhark/microgpt.fut:356:93-121
            
            double zs_res_138227 = 1.0 / zs_rhs_138226;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140578 = 0; i_140578 < (int64_t) 16; i_140578++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_138256 = ((double *) mem_142490)[i_140592 * (int64_t) 256 + i_140585 * (int64_t) 16 + i_140578];
                
                // futhark/microgpt.fut:340:59-121
                
                double zt_res_138257 = zs_res_138208 * zt_lhs_138256;
                
                // futhark/microgpt.fut:340:88-148
                
                double zp_res_138258 = zp_rhs_138209 + zt_res_138257;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_138265 = ((double *) mem_142489)[i_140592 * (int64_t) 256 + i_140585 * (int64_t) 16 + i_140578];
                
                // futhark/microgpt.fut:356:59-121
                
                double zt_res_138266 = zs_res_138227 * zt_lhs_138265;
                
                // futhark/microgpt.fut:356:88-148
                
                double zp_res_138267 = zp_rhs_138228 + zt_res_138266;
                
                ((double *) mem_142629)[i_140578] = zp_res_138267;
                ((double *) mem_142630)[i_140578] = zp_res_138258;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_142619, i_140585 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142629, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_142620, i_140585 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142630, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_142607, i_140592 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_142619, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_142608, i_140592 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_142620, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142661_cached_sizze_143786 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_142661, &mem_142661_cached_sizze_143786, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142662_cached_sizze_143787 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_142662, &mem_142662_cached_sizze_143787, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142671_cached_sizze_143788 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142671, &mem_142671_cached_sizze_143788, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142672_cached_sizze_143789 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142672, &mem_142672_cached_sizze_143789, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140606 = 0; i_140606 < (int64_t) 4; i_140606++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140599 = 0; i_140599 < (int64_t) 16; i_140599++) {
            double f_elem_138287 = ((double *) mem_141496)[i_140606 * (int64_t) 16 + i_140599];
            double f_elem_138289 = ((double *) mem_141493)[i_140606 * (int64_t) 16 + i_140599];
            
            // futhark/microgpt.fut:341:119-145
            
            double neg_res_138294 = -f_elem_138287;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_138295;
            double r_138297 = 0.0;
            
            for (int64_t i_138296 = 0; i_138296 < (int64_t) 16; i_138296++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_138298 = ((double *) mem_141384)[i_140606 * (int64_t) 256 + i_140599 * (int64_t) 16 + i_138296];
                
                // futhark/microgpt.fut:341:85-145
                
                double zp_res_138299 = neg_res_138294 + zp_lhs_138298;
                
                // futhark/microgpt.fut:341:78-145
                
                double exp_res_138300 = futrts_exp64(zp_res_138299);
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_138301 = ((double *) mem_142608)[i_140606 * (int64_t) 256 + i_140599 * (int64_t) 16 + i_138296];
                
                // futhark/microgpt.fut:341:78-181
                
                double zt_res_138302 = exp_res_138300 * zt_rhs_138301;
                
                // futhark/microgpt.fut:341:70-181
                
                double neg_res_138303 = -zt_res_138302;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_138304 = r_138297 + neg_res_138303;
                double r_tmp_143395 = zp_res_138304;
                
                r_138297 = r_tmp_143395;
            }
            defunc_0_lifted_lambda_res_138295 = r_138297;
            // futhark/microgpt.fut:357:119-145
            
            double neg_res_138312 = -f_elem_138289;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_138313;
            double r_138315 = 0.0;
            
            for (int64_t i_138314 = 0; i_138314 < (int64_t) 16; i_138314++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_138316 = ((double *) mem_141383)[i_140606 * (int64_t) 256 + i_140599 * (int64_t) 16 + i_138314];
                
                // futhark/microgpt.fut:357:85-145
                
                double zp_res_138317 = neg_res_138312 + zp_lhs_138316;
                
                // futhark/microgpt.fut:357:78-145
                
                double exp_res_138318 = futrts_exp64(zp_res_138317);
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_138319 = ((double *) mem_142607)[i_140606 * (int64_t) 256 + i_140599 * (int64_t) 16 + i_138314];
                
                // futhark/microgpt.fut:357:78-181
                
                double zt_res_138320 = exp_res_138318 * zt_rhs_138319;
                
                // futhark/microgpt.fut:357:70-181
                
                double neg_res_138321 = -zt_res_138320;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_138322 = r_138315 + neg_res_138321;
                double r_tmp_143396 = zp_res_138322;
                
                r_138315 = r_tmp_143396;
            }
            defunc_0_lifted_lambda_res_138313 = r_138315;
            ((double *) mem_142671)[i_140599] = defunc_0_lifted_lambda_res_138313;
            ((double *) mem_142672)[i_140599] = defunc_0_lifted_lambda_res_138295;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142661, i_140606 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142671, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142662, i_140606 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142672, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142693_cached_sizze_143790 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_142693, &mem_142693_cached_sizze_143790, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142694_cached_sizze_143791 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_142694, &mem_142694_cached_sizze_143791, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142705_cached_sizze_143792 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142705, &mem_142705_cached_sizze_143792, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142706_cached_sizze_143793 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142706, &mem_142706_cached_sizze_143793, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142715_cached_sizze_143794 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142715, &mem_142715_cached_sizze_143794, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142716_cached_sizze_143795 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142716, &mem_142716_cached_sizze_143795, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140627 = 0; i_140627 < (int64_t) 4; i_140627++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140620 = 0; i_140620 < (int64_t) 16; i_140620++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140613 = 0; i_140613 < (int64_t) 16; i_140613++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_138447;
                double r_138449 = 0.0;
                
                for (int64_t i_138448 = 0; i_138448 < (int64_t) 16; i_138448++) {
                    // futhark/microgpt.fut:344:81-226
                    
                    bool cond_138450 = i_140620 == i_138448;
                    
                    // futhark/microgpt.fut:344:81-226
                    
                    double zp_lhs_138451;
                    
                    if (cond_138450) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zp_lhs_139584 = ((double *) mem_141384)[i_140627 * (int64_t) 256 + i_138448 * (int64_t) 16 + i_140613];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double neg_arg0_139585 = ((double *) mem_141496)[i_140627 * (int64_t) 16 + i_138448];
                        
                        // futhark/microgpt.fut:344:153-179
                        
                        double neg_res_139586 = -neg_arg0_139585;
                        
                        // futhark/microgpt.fut:344:119-179
                        
                        double zp_res_139587 = zp_lhs_139584 + neg_res_139586;
                        
                        // futhark/microgpt.fut:344:112-179
                        
                        double exp_res_139588 = futrts_exp64(zp_res_139587);
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_139589 = ((double *) mem_142608)[i_140627 * (int64_t) 256 + i_138448 * (int64_t) 16 + i_140613];
                        
                        // futhark/microgpt.fut:344:112-215
                        
                        double zt_res_139590 = exp_res_139588 * zt_rhs_139589;
                        
                        zp_lhs_138451 = zt_res_139590;
                    } else {
                        zp_lhs_138451 = 0.0;
                    }
                    // futhark/microgpt.fut:344:233-428
                    
                    double zp_rhs_138467;
                    
                    if (cond_138450) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_139595 = ((double *) mem_142662)[i_140627 * (int64_t) 16 + i_138448];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zp_lhs_139600 = ((double *) mem_141384)[i_140627 * (int64_t) 256 + i_138448 * (int64_t) 16 + i_140613];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double neg_arg0_139601 = ((double *) mem_141495)[i_140627 * (int64_t) 16 + i_138448];
                        
                        // futhark/microgpt.fut:344:359-385
                        
                        double neg_res_139602 = -neg_arg0_139601;
                        
                        // futhark/microgpt.fut:344:325-385
                        
                        double zp_res_139603 = zp_lhs_139600 + neg_res_139602;
                        
                        // futhark/microgpt.fut:344:318-385
                        
                        double neg_res_139604 = -zp_res_139603;
                        
                        // futhark/microgpt.fut:100:42-54
                        
                        double max_res_139605 = fmax64(0.0, neg_res_139604);
                        
                        // futhark/microgpt.fut:100:35-54
                        
                        double sgn_res_139606 = fsignum64(max_res_139605);
                        
                        // futhark/microgpt.fut:344:299-388
                        
                        double neg_res_139607 = -sgn_res_139606;
                        
                        // futhark/microgpt.fut:344:290-389
                        
                        double zp_res_139608 = 1.0 + neg_res_139607;
                        
                        // futhark/microgpt.fut:344:264-389
                        
                        double zt_res_139609 = zt_lhs_139595 * zp_res_139608;
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_139610 = ((double *) mem_141494)[i_140627 * (int64_t) 16 + i_138448];
                        
                        // futhark/microgpt.fut:344:285-417
                        
                        double zt_res_139611 = zt_res_139609 * zt_rhs_139610;
                        
                        zp_rhs_138467 = zt_res_139611;
                    } else {
                        zp_rhs_138467 = 0.0;
                    }
                    // futhark/microgpt.fut:344:81-428
                    
                    double zp_res_138489 = zp_lhs_138451 + zp_rhs_138467;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_138490 = r_138449 + zp_res_138489;
                    double r_tmp_143403 = zp_res_138490;
                    
                    r_138449 = r_tmp_143403;
                }
                defunc_0_lifted_lambda_res_138447 = r_138449;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_138493;
                double r_138495 = 0.0;
                
                for (int64_t i_138494 = 0; i_138494 < (int64_t) 16; i_138494++) {
                    // futhark/microgpt.fut:360:81-226
                    
                    bool cond_138496 = i_140620 == i_138494;
                    
                    // futhark/microgpt.fut:360:81-226
                    
                    double zp_lhs_138497;
                    
                    if (cond_138496) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zp_lhs_139620 = ((double *) mem_141383)[i_140627 * (int64_t) 256 + i_138494 * (int64_t) 16 + i_140613];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double neg_arg0_139621 = ((double *) mem_141493)[i_140627 * (int64_t) 16 + i_138494];
                        
                        // futhark/microgpt.fut:360:153-179
                        
                        double neg_res_139622 = -neg_arg0_139621;
                        
                        // futhark/microgpt.fut:360:119-179
                        
                        double zp_res_139623 = zp_lhs_139620 + neg_res_139622;
                        
                        // futhark/microgpt.fut:360:112-179
                        
                        double exp_res_139624 = futrts_exp64(zp_res_139623);
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_139625 = ((double *) mem_142607)[i_140627 * (int64_t) 256 + i_138494 * (int64_t) 16 + i_140613];
                        
                        // futhark/microgpt.fut:360:112-215
                        
                        double zt_res_139626 = exp_res_139624 * zt_rhs_139625;
                        
                        zp_lhs_138497 = zt_res_139626;
                    } else {
                        zp_lhs_138497 = 0.0;
                    }
                    // futhark/microgpt.fut:360:233-428
                    
                    double zp_rhs_138513;
                    
                    if (cond_138496) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_139631 = ((double *) mem_142661)[i_140627 * (int64_t) 16 + i_138494];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zp_lhs_139636 = ((double *) mem_141383)[i_140627 * (int64_t) 256 + i_138494 * (int64_t) 16 + i_140613];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double neg_arg0_139637 = ((double *) mem_141492)[i_140627 * (int64_t) 16 + i_138494];
                        
                        // futhark/microgpt.fut:360:359-385
                        
                        double neg_res_139638 = -neg_arg0_139637;
                        
                        // futhark/microgpt.fut:360:325-385
                        
                        double zp_res_139639 = zp_lhs_139636 + neg_res_139638;
                        
                        // futhark/microgpt.fut:360:318-385
                        
                        double neg_res_139640 = -zp_res_139639;
                        
                        // futhark/microgpt.fut:100:42-54
                        
                        double max_res_139641 = fmax64(0.0, neg_res_139640);
                        
                        // futhark/microgpt.fut:100:35-54
                        
                        double sgn_res_139642 = fsignum64(max_res_139641);
                        
                        // futhark/microgpt.fut:360:299-388
                        
                        double neg_res_139643 = -sgn_res_139642;
                        
                        // futhark/microgpt.fut:360:290-389
                        
                        double zp_res_139644 = 1.0 + neg_res_139643;
                        
                        // futhark/microgpt.fut:360:264-389
                        
                        double zt_res_139645 = zt_lhs_139631 * zp_res_139644;
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_139646 = ((double *) mem_141491)[i_140627 * (int64_t) 16 + i_138494];
                        
                        // futhark/microgpt.fut:360:285-417
                        
                        double zt_res_139647 = zt_res_139645 * zt_rhs_139646;
                        
                        zp_rhs_138513 = zt_res_139647;
                    } else {
                        zp_rhs_138513 = 0.0;
                    }
                    // futhark/microgpt.fut:360:81-428
                    
                    double zp_res_138535 = zp_lhs_138497 + zp_rhs_138513;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_138536 = r_138495 + zp_res_138535;
                    double r_tmp_143404 = zp_res_138536;
                    
                    r_138495 = r_tmp_143404;
                }
                defunc_0_lifted_lambda_res_138493 = r_138495;
                ((double *) mem_142715)[i_140613] = defunc_0_lifted_lambda_res_138493;
                ((double *) mem_142716)[i_140613] = defunc_0_lifted_lambda_res_138447;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_142705, i_140620 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142715, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_142706, i_140620 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142716, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_142693, i_140627 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_142705, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_142694, i_140627 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_142706, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142747_cached_sizze_143796 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_142747, &mem_142747_cached_sizze_143796, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142748_cached_sizze_143797 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_142748, &mem_142748_cached_sizze_143797, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142759_cached_sizze_143798 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142759, &mem_142759_cached_sizze_143798, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142760_cached_sizze_143799 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142760, &mem_142760_cached_sizze_143799, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142769_cached_sizze_143800 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142769, &mem_142769_cached_sizze_143800, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142770_cached_sizze_143801 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142770, &mem_142770_cached_sizze_143801, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140648 = 0; i_140648 < (int64_t) 4; i_140648++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140641 = 0; i_140641 < (int64_t) 16; i_140641++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_140634 = 0; i_140634 < (int64_t) 16; i_140634++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_138817 = ((double *) mem_142694)[i_140648 * (int64_t) 256 + i_140641 * (int64_t) 16 + i_140634];
                
                // futhark/microgpt.fut:345:58-100
                
                double zs_res_138818 = zs_lhs_138817 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_138825 = ((double *) mem_142693)[i_140648 * (int64_t) 256 + i_140641 * (int64_t) 16 + i_140634];
                
                // futhark/microgpt.fut:361:58-100
                
                double zs_res_138826 = zs_lhs_138825 / 2.0;
                
                ((double *) mem_142769)[i_140634] = zs_res_138826;
                ((double *) mem_142770)[i_140634] = zs_res_138818;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_142759, i_140641 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142769, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_142760, i_140641 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142770, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_142747, i_140648 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_142759, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_142748, i_140648 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_142760, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_142801, (int64_t) 2048, "mem_142801")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142802_cached_sizze_143802 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142802, &mem_142802_cached_sizze_143802, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142803_cached_sizze_143803 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142803, &mem_142803_cached_sizze_143803, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142804_cached_sizze_143804 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142804, &mem_142804_cached_sizze_143804, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142821_cached_sizze_143805 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142821, &mem_142821_cached_sizze_143805, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142822_cached_sizze_143806 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142822, &mem_142822_cached_sizze_143806, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142823_cached_sizze_143807 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142823, &mem_142823_cached_sizze_143807, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142824_cached_sizze_143808 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142824, &mem_142824_cached_sizze_143808, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140672 = 0; i_140672 < (int64_t) 16; i_140672++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140659 = 0; i_140659 < (int64_t) 16; i_140659++) {
            // futhark/microgpt.fut:330:40-43
            
            int64_t zt_lhs_137774 = sdiv64(i_140659, (int64_t) 4);
            
            // futhark/microgpt.fut:330:27-45
            
            bool x_137775 = sle64((int64_t) 0, zt_lhs_137774);
            
            // futhark/microgpt.fut:330:27-45
            
            bool y_137776 = slt64(zt_lhs_137774, (int64_t) 4);
            
            // futhark/microgpt.fut:330:27-45
            
            bool bounds_check_137777 = x_137775 && y_137776;
            
            // futhark/microgpt.fut:330:27-45
            
            bool index_certs_137778;
            
            if (!bounds_check_137777) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_137774, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:330:27-45\n   #1  futhark/microgpt.fut:61:46-49\n   #2  futhark/microgpt.fut:330:5-108\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:322:13-330:118\n   #8  futhark/microgpt.fut:459:5-75\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:330:62-65
            
            int64_t zt_lhs_137779 = smod64(i_140659, (int64_t) 4);
            
            // futhark/microgpt.fut:330:27-67
            
            bool x_137780 = sle64((int64_t) 0, zt_lhs_137779);
            
            // futhark/microgpt.fut:330:27-67
            
            bool y_137781 = slt64(zt_lhs_137779, (int64_t) 4);
            
            // futhark/microgpt.fut:330:27-67
            
            bool bounds_check_137782 = x_137780 && y_137781;
            
            // futhark/microgpt.fut:330:27-67
            
            bool index_certs_137783;
            
            if (!bounds_check_137782) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_137779, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:330:27-67\n   #1  futhark/microgpt.fut:61:46-49\n   #2  futhark/microgpt.fut:330:5-108\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:322:13-330:118\n   #8  futhark/microgpt.fut:459:5-75\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137784;
            double r_137786 = 0.0;
            
            for (int64_t i_137785 = 0; i_137785 < (int64_t) 16; i_137785++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_137787 = ((double *) mem_142362)[zt_lhs_137774 * (int64_t) 64 + i_137785 * (int64_t) 4 + zt_lhs_137779];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_137788 = ((double *) mem_141813)[zt_lhs_137774 * (int64_t) 256 + i_137785 * (int64_t) 16 + i_140672];
                
                // futhark/microgpt.fut:330:27-106
                
                double zt_res_137789 = zt_lhs_137787 * zt_rhs_137788;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137790 = r_137786 + zt_res_137789;
                double r_tmp_143419 = zp_res_137790;
                
                r_137786 = r_tmp_143419;
            }
            defunc_0_lifted_lambda_res_137784 = r_137786;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137803;
            double r_137805 = 0.0;
            
            for (int64_t i_137804 = 0; i_137804 < (int64_t) 16; i_137804++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_137806 = ((double *) mem_142748)[zt_lhs_137774 * (int64_t) 256 + i_137804 * (int64_t) 16 + i_140672];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_137807 = ((double *) mem_141196)[zt_lhs_137774 * (int64_t) 64 + i_137804 * (int64_t) 4 + zt_lhs_137779];
                
                // futhark/microgpt.fut:346:27-105
                
                double zt_res_137808 = zt_lhs_137806 * zt_rhs_137807;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137809 = r_137805 + zt_res_137808;
                double r_tmp_143420 = zp_res_137809;
                
                r_137805 = r_tmp_143420;
            }
            defunc_0_lifted_lambda_res_137803 = r_137805;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137825;
            double r_137827 = 0.0;
            
            for (int64_t i_137826 = 0; i_137826 < (int64_t) 16; i_137826++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_137828 = ((double *) mem_142747)[zt_lhs_137774 * (int64_t) 256 + i_140672 * (int64_t) 16 + i_137826];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_137829 = ((double *) mem_141195)[zt_lhs_137774 * (int64_t) 64 + i_137826 * (int64_t) 4 + zt_lhs_137779];
                
                // futhark/microgpt.fut:362:27-105
                
                double zt_res_137830 = zt_lhs_137828 * zt_rhs_137829;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137831 = r_137827 + zt_res_137830;
                double r_tmp_143421 = zp_res_137831;
                
                r_137827 = r_tmp_143421;
            }
            defunc_0_lifted_lambda_res_137825 = r_137827;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_137843;
            double r_137845 = 0.0;
            
            for (int64_t i_137844 = 0; i_137844 < (int64_t) 16; i_137844++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_137846 = ((double *) mem_142317)[i_137844 * (int64_t) 16 + i_140672];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_137847 = ((double *) mem_141894)[i_137844 * (int64_t) 16 + i_140659];
                
                // futhark/microgpt.fut:394:68-112
                
                double zt_res_137848 = zt_lhs_137846 * zt_rhs_137847;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_137849 = r_137845 + zt_res_137848;
                double r_tmp_143422 = zp_res_137849;
                
                r_137845 = r_tmp_143422;
            }
            defunc_0_lifted_lambda_res_137843 = r_137845;
            ((double *) mem_142821)[i_140659] = defunc_0_lifted_lambda_res_137843;
            ((double *) mem_142822)[i_140659] = defunc_0_lifted_lambda_res_137825;
            ((double *) mem_142823)[i_140659] = defunc_0_lifted_lambda_res_137803;
            ((double *) mem_142824)[i_140659] = defunc_0_lifted_lambda_res_137784;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142801.mem, i_140672 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142821, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142802, i_140672 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142822, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142803, i_140672 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142823, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142804, i_140672 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142824, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142865_cached_sizze_143809 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142865, &mem_142865_cached_sizze_143809, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142870_cached_sizze_143810 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142870, &mem_142870_cached_sizze_143810, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140683 = 0; i_140683 < (int64_t) 16; i_140683++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140679 = 0; i_140679 < (int64_t) 16; i_140679++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128532;
            double r_128534 = 0.0;
            
            for (int64_t i_128533 = 0; i_128533 < (int64_t) 16; i_128533++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128535 = ((double *) mem_142804)[i_140683 * (int64_t) 16 + i_128533];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128536 = ((double *) wval_mem_141060.mem)[i_128533 * (int64_t) 16 + i_140679];
                
                // futhark/microgpt.fut:365:73-118
                
                double zt_res_128537 = zt_lhs_128535 * zt_rhs_128536;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128538 = r_128534 + zt_res_128537;
                double r_tmp_143425 = zp_res_128538;
                
                r_128534 = r_tmp_143425;
            }
            defunc_0_lifted_lambda_res_128532 = r_128534;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128539;
            double r_128541 = 0.0;
            
            for (int64_t i_128540 = 0; i_128540 < (int64_t) 16; i_128540++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128542 = ((double *) mem_142803)[i_140683 * (int64_t) 16 + i_128540];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128543 = ((double *) wkey_mem_141054.mem)[i_128540 * (int64_t) 16 + i_140679];
                
                // futhark/microgpt.fut:365:149-194
                
                double zt_res_128544 = zt_lhs_128542 * zt_rhs_128543;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128545 = r_128541 + zt_res_128544;
                double r_tmp_143426 = zp_res_128545;
                
                r_128541 = r_tmp_143426;
            }
            defunc_0_lifted_lambda_res_128539 = r_128541;
            // futhark/microgpt.fut:365:51-196
            
            double zp_res_128546 = defunc_0_lifted_lambda_res_128532 + defunc_0_lifted_lambda_res_128539;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128547;
            double r_128549 = 0.0;
            
            for (int64_t i_128548 = 0; i_128548 < (int64_t) 16; i_128548++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128550 = ((double *) mem_142802)[i_140683 * (int64_t) 16 + i_128548];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128551 = ((double *) wqry_mem_141057.mem)[i_128548 * (int64_t) 16 + i_140679];
                
                // futhark/microgpt.fut:365:226-271
                
                double zt_res_128552 = zt_lhs_128550 * zt_rhs_128551;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128553 = r_128549 + zt_res_128552;
                double r_tmp_143427 = zp_res_128553;
                
                r_128549 = r_tmp_143427;
            }
            defunc_0_lifted_lambda_res_128547 = r_128549;
            // futhark/microgpt.fut:365:122-273
            
            double zp_res_128554 = zp_res_128546 + defunc_0_lifted_lambda_res_128547;
            
            ((double *) mem_142870)[i_140679] = zp_res_128554;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142865, i_140683 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142870, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_142881, (int64_t) 2048, "mem_142881")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_142882, (int64_t) 2048, "mem_142882")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_142883, (int64_t) 2048, "mem_142883")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142884_cached_sizze_143811 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142884, &mem_142884_cached_sizze_143811, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142885_cached_sizze_143812 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142885, &mem_142885_cached_sizze_143812, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142904_cached_sizze_143813 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142904, &mem_142904_cached_sizze_143813, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142905_cached_sizze_143814 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142905, &mem_142905_cached_sizze_143814, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142906_cached_sizze_143815 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142906, &mem_142906_cached_sizze_143815, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140705 = 0; i_140705 < (int64_t) 16; i_140705++) {
        // futhark/microgpt.fut:364:47-59
        
        double zp_lhs_133283 = ((double *) mem_141141)[i_140705];
        
        // futhark/microgpt.fut:364:47-87
        
        double zp_res_133284 = 1.0e-5 + zp_lhs_133283;
        
        // futhark/microgpt.fut:364:39-87
        
        double sqrt_res_133285 = futrts_sqrt64(zp_res_133284);
        
        // futhark/microgpt.fut:366:128-157
        
        double zt_res_133293 = sqrt_res_133285 * sqrt_res_133285;
        
        // futhark/microgpt.fut:366:119-157
        
        double zs_res_133294 = 1.0 / zt_res_133293;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_133295;
        double r_133297 = 0.0;
        
        for (int64_t i_133296 = 0; i_133296 < (int64_t) 16; i_133296++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_133298 = ((double *) mem_142865)[i_140705 * (int64_t) 16 + i_133296];
            
            // futhark/microgpt.fut:61:46-49
            
            double zt_rhs_133299 = ((double *) mem_141125)[i_140705 * (int64_t) 16 + i_133296];
            
            // futhark/microgpt.fut:366:69-112
            
            double zt_res_133300 = zt_lhs_133298 * zt_rhs_133299;
            
            // futhark/microgpt.fut:366:90-157
            
            double zt_res_133301 = zs_res_133294 * zt_res_133300;
            
            // futhark/microgpt.fut:366:61-157
            
            double neg_res_133302 = -zt_res_133301;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_133303 = r_133297 + neg_res_133302;
            double r_tmp_143433 = zp_res_133303;
            
            r_133297 = r_tmp_143433;
        }
        defunc_0_lifted_lambda_res_133295 = r_133297;
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140691 = 0; i_140691 < (int64_t) 16; i_140691++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_138893;
            double r_138895 = 0.0;
            
            for (int64_t i_138894 = 0; i_138894 < (int64_t) 16; i_138894++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_138896 = ((double *) mem_142802)[i_138894 * (int64_t) 16 + i_140705];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_138897 = ((double *) mem_141178)[i_138894 * (int64_t) 16 + i_140691];
                
                // futhark/microgpt.fut:391:68-111
                
                double zt_res_138898 = zt_lhs_138896 * zt_rhs_138897;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_138899 = r_138895 + zt_res_138898;
                double r_tmp_143437 = zp_res_138899;
                
                r_138895 = r_tmp_143437;
            }
            defunc_0_lifted_lambda_res_138893 = r_138895;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_138906;
            double r_138908 = 0.0;
            
            for (int64_t i_138907 = 0; i_138907 < (int64_t) 16; i_138907++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_138909 = ((double *) mem_142803)[i_138907 * (int64_t) 16 + i_140705];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_138910 = ((double *) mem_141178)[i_138907 * (int64_t) 16 + i_140691];
                
                // futhark/microgpt.fut:392:68-111
                
                double zt_res_138911 = zt_lhs_138909 * zt_rhs_138910;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_138912 = r_138908 + zt_res_138911;
                double r_tmp_143438 = zp_res_138912;
                
                r_138908 = r_tmp_143438;
            }
            defunc_0_lifted_lambda_res_138906 = r_138908;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_138922;
            double r_138924 = 0.0;
            
            for (int64_t i_138923 = 0; i_138923 < (int64_t) 16; i_138923++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_138925 = ((double *) mem_142804)[i_138923 * (int64_t) 16 + i_140705];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_138926 = ((double *) mem_141178)[i_138923 * (int64_t) 16 + i_140691];
                
                // futhark/microgpt.fut:393:68-111
                
                double zt_res_138927 = zt_lhs_138925 * zt_rhs_138926;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_138928 = r_138924 + zt_res_138927;
                double r_tmp_143439 = zp_res_138928;
                
                r_138924 = r_tmp_143439;
            }
            defunc_0_lifted_lambda_res_138922 = r_138924;
            ((double *) mem_142904)[i_140691] = defunc_0_lifted_lambda_res_138922;
            ((double *) mem_142905)[i_140691] = defunc_0_lifted_lambda_res_138906;
            ((double *) mem_142906)[i_140691] = defunc_0_lifted_lambda_res_138893;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142881.mem, i_140705 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142904, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142882.mem, i_140705 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142905, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142883.mem, i_140705 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142906, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        ((double *) mem_142884)[i_140705] = defunc_0_lifted_lambda_res_133295;
        ((double *) mem_142885)[i_140705] = sqrt_res_133285;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142943_cached_sizze_143816 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142943, &mem_142943_cached_sizze_143816, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140713 = 0; i_140713 < (int64_t) 16; i_140713++) {
        // futhark/microgpt.fut:367:39-51
        
        double zt_lhs_128582 = ((double *) mem_142884)[i_140713];
        
        // futhark/microgpt.fut:367:93-105
        
        double zp_lhs_128583 = ((double *) mem_141141)[i_140713];
        
        // futhark/microgpt.fut:367:93-133
        
        double zp_res_128584 = 1.0e-5 + zp_lhs_128583;
        
        // futhark/microgpt.fut:367:85-133
        
        double sqrt_res_128585 = futrts_sqrt64(zp_res_128584);
        
        // futhark/microgpt.fut:367:71-135
        
        double zt_res_128586 = 2.0 * sqrt_res_128585;
        
        // futhark/microgpt.fut:367:57-135
        
        double zs_res_128587 = 1.0 / zt_res_128586;
        
        // futhark/microgpt.fut:367:39-135
        
        double zt_res_128588 = zt_lhs_128582 * zs_res_128587;
        
        ((double *) mem_142943)[i_140713] = zt_res_128588;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142950_cached_sizze_143817 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142950, &mem_142950_cached_sizze_143817, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142955_cached_sizze_143818 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142955, &mem_142955_cached_sizze_143818, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140721 = 0; i_140721 < (int64_t) 16; i_140721++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140717 = 0; i_140717 < (int64_t) 16; i_140717++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_128602 = ((double *) mem_142317)[i_140721 * (int64_t) 16 + i_140717];
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128603;
            double r_128605 = 0.0;
            
            for (int64_t i_128604 = 0; i_128604 < (int64_t) 16; i_128604++) {
                // futhark/microgpt.fut:368:86-174
                
                bool cond_128606 = i_140721 == i_128604;
                
                // futhark/microgpt.fut:368:86-174
                
                double zp_lhs_128607;
                
                if (cond_128606) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_139660 = ((double *) mem_142865)[i_128604 * (int64_t) 16 + i_140717];
                    
                    // futhark/microgpt.fut:368:150-162
                    
                    double zs_rhs_139661 = ((double *) mem_142885)[i_128604];
                    
                    // futhark/microgpt.fut:368:142-162
                    
                    double zs_res_139662 = 1.0 / zs_rhs_139661;
                    
                    // futhark/microgpt.fut:368:116-162
                    
                    double zt_res_139663 = zt_lhs_139660 * zs_res_139662;
                    
                    zp_lhs_128607 = zt_res_139663;
                } else {
                    zp_lhs_128607 = 0.0;
                }
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_128612;
                double r_128614 = 0.0;
                
                for (int64_t i_128613 = 0; i_128613 < (int64_t) 16; i_128613++) {
                    // futhark/microgpt.fut:368:204-338
                    
                    double zp_lhs_128615;
                    
                    if (cond_128606) {
                        // futhark/microgpt.fut:368:234-327
                        
                        bool cond_139668 = i_140717 == i_128613;
                        
                        // futhark/microgpt.fut:368:234-327
                        
                        double zp_lhs_t_res_139669;
                        
                        if (cond_139668) {
                            // futhark/microgpt.fut:368:265-277
                            
                            double zs_lhs_139670 = ((double *) mem_142943)[i_128604];
                            
                            // futhark/microgpt.fut:368:265-292
                            
                            double zs_res_139671 = zs_lhs_139670 / 16.0;
                            
                            // futhark/microgpt.fut:61:46-49
                            
                            double zt_rhs_139672 = ((double *) mem_141125)[i_128604 * (int64_t) 16 + i_140717];
                            
                            // futhark/microgpt.fut:368:278-316
                            
                            double zt_res_139673 = zs_res_139671 * zt_rhs_139672;
                            
                            zp_lhs_t_res_139669 = zt_res_139673;
                        } else {
                            zp_lhs_t_res_139669 = 0.0;
                        }
                        zp_lhs_128615 = zp_lhs_t_res_139669;
                    } else {
                        zp_lhs_128615 = 0.0;
                    }
                    // futhark/microgpt.fut:368:345-479
                    
                    double zp_rhs_128622;
                    
                    if (cond_128606) {
                        // futhark/microgpt.fut:368:375-468
                        
                        bool cond_139678 = i_140717 == i_128613;
                        
                        // futhark/microgpt.fut:368:375-468
                        
                        double zp_rhs_t_res_139679;
                        
                        if (cond_139678) {
                            // futhark/microgpt.fut:368:406-418
                            
                            double zs_lhs_139680 = ((double *) mem_142943)[i_128604];
                            
                            // futhark/microgpt.fut:368:406-433
                            
                            double zs_res_139681 = zs_lhs_139680 / 16.0;
                            
                            // futhark/microgpt.fut:61:46-49
                            
                            double zt_rhs_139682 = ((double *) mem_141125)[i_128604 * (int64_t) 16 + i_140717];
                            
                            // futhark/microgpt.fut:368:419-457
                            
                            double zt_res_139683 = zs_res_139681 * zt_rhs_139682;
                            
                            zp_rhs_t_res_139679 = zt_res_139683;
                        } else {
                            zp_rhs_t_res_139679 = 0.0;
                        }
                        zp_rhs_128622 = zp_rhs_t_res_139679;
                    } else {
                        zp_rhs_128622 = 0.0;
                    }
                    // futhark/microgpt.fut:368:204-479
                    
                    double zp_res_128629 = zp_lhs_128615 + zp_rhs_128622;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_128630 = r_128614 + zp_res_128629;
                    double r_tmp_143444 = zp_res_128630;
                    
                    r_128614 = r_tmp_143444;
                }
                defunc_0_lifted_lambda_res_128612 = r_128614;
                // futhark/microgpt.fut:368:86-482
                
                double zp_res_128631 = zp_lhs_128607 + defunc_0_lifted_lambda_res_128612;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128632 = r_128605 + zp_res_128631;
                double r_tmp_143443 = zp_res_128632;
                
                r_128605 = r_tmp_143443;
            }
            defunc_0_lifted_lambda_res_128603 = r_128605;
            // futhark/microgpt.fut:368:37-485
            
            double zp_res_128633 = zp_lhs_128602 + defunc_0_lifted_lambda_res_128603;
            
            ((double *) mem_142955)[i_140717] = zp_res_128633;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142950, i_140721 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142955, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142966_cached_sizze_143819 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142966, &mem_142966_cached_sizze_143819, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142967_cached_sizze_143820 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_142967, &mem_142967_cached_sizze_143820, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142976_cached_sizze_143821 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142976, &mem_142976_cached_sizze_143821, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142977_cached_sizze_143822 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142977, &mem_142977_cached_sizze_143822, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140734 = 0; i_140734 < (int64_t) 16; i_140734++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140727 = 0; i_140727 < (int64_t) 16; i_140727++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_138976 = ((double *) mem_142950)[i_140734 * (int64_t) 16 + i_140727];
            
            ((double *) mem_142976)[i_140727] = lifted_lambda_res_138976;
            ((double *) mem_142977)[i_140727] = lifted_lambda_res_138976;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142966, i_140734 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142976, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_142967, i_140734 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_142977, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142998_cached_sizze_143823 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142998, &mem_142998_cached_sizze_143823, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_142999_cached_sizze_143824 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_142999, &mem_142999_cached_sizze_143824, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143000_cached_sizze_143825 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143000, &mem_143000_cached_sizze_143825, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143001_cached_sizze_143826 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143001, &mem_143001_cached_sizze_143826, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140745 = 0; i_140745 < (int64_t) 16; i_140745++) {
        // futhark/microgpt.fut:386:47-59
        
        double zp_lhs_133408 = ((double *) mem_141082)[i_140745];
        
        // futhark/microgpt.fut:386:47-87
        
        double zp_res_133409 = 1.0e-5 + zp_lhs_133408;
        
        // futhark/microgpt.fut:386:39-87
        
        double sqrt_res_133410 = futrts_sqrt64(zp_res_133409);
        
        // futhark/microgpt.fut:388:156-185
        
        double zt_res_133418 = sqrt_res_133410 * sqrt_res_133410;
        
        // futhark/microgpt.fut:388:147-185
        
        double zs_res_133419 = 1.0 / zt_res_133418;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_133420;
        double r_133422 = 0.0;
        
        for (int64_t i_133421 = 0; i_133421 < (int64_t) 16; i_133421++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_133423 = ((double *) mem_142967)[i_140745 * (int64_t) 16 + i_133421];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_133424 = ((double *) wpe_mem_141056.mem)[i_140745 * (int64_t) 16 + i_133421];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_133425 = ((double *) mem_141065)[i_140745 * (int64_t) 16 + i_133421];
            
            // futhark/microgpt.fut:388:95-139
            
            double zp_res_133426 = zp_lhs_133424 + zp_rhs_133425;
            
            // futhark/microgpt.fut:388:69-139
            
            double zt_res_133427 = zt_lhs_133423 * zp_res_133426;
            
            // futhark/microgpt.fut:388:90-185
            
            double zt_res_133428 = zs_res_133419 * zt_res_133427;
            
            // futhark/microgpt.fut:388:61-185
            
            double neg_res_133429 = -zt_res_133428;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_133430 = r_133422 + neg_res_133429;
            double r_tmp_143453 = zp_res_133430;
            
            r_133422 = r_tmp_143453;
        }
        defunc_0_lifted_lambda_res_133420 = r_133422;
        // futhark/microgpt.fut:399:47-59
        
        double zp_lhs_133441 = ((double *) mem_141081)[i_140745];
        
        // futhark/microgpt.fut:399:47-87
        
        double zp_res_133442 = 1.0e-5 + zp_lhs_133441;
        
        // futhark/microgpt.fut:399:39-87
        
        double sqrt_res_133443 = futrts_sqrt64(zp_res_133442);
        
        // futhark/microgpt.fut:401:156-185
        
        double zt_res_133451 = sqrt_res_133443 * sqrt_res_133443;
        
        // futhark/microgpt.fut:401:147-185
        
        double zs_res_133452 = 1.0 / zt_res_133451;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_133453;
        double r_133455 = 0.0;
        
        for (int64_t i_133454 = 0; i_133454 < (int64_t) 16; i_133454++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_133456 = ((double *) mem_142966)[i_140745 * (int64_t) 16 + i_133454];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_133457 = ((double *) wpe_mem_141056.mem)[i_140745 * (int64_t) 16 + i_133454];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_133458 = ((double *) mem_141065)[i_140745 * (int64_t) 16 + i_133454];
            
            // futhark/microgpt.fut:401:95-139
            
            double zp_res_133459 = zp_lhs_133457 + zp_rhs_133458;
            
            // futhark/microgpt.fut:401:69-139
            
            double zt_res_133460 = zt_lhs_133456 * zp_res_133459;
            
            // futhark/microgpt.fut:401:90-185
            
            double zt_res_133461 = zs_res_133452 * zt_res_133460;
            
            // futhark/microgpt.fut:401:61-185
            
            double neg_res_133462 = -zt_res_133461;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_133463 = r_133455 + neg_res_133462;
            double r_tmp_143454 = zp_res_133463;
            
            r_133455 = r_tmp_143454;
        }
        defunc_0_lifted_lambda_res_133453 = r_133455;
        ((double *) mem_142998)[i_140745] = defunc_0_lifted_lambda_res_133453;
        ((double *) mem_142999)[i_140745] = sqrt_res_133443;
        ((double *) mem_143000)[i_140745] = defunc_0_lifted_lambda_res_133420;
        ((double *) mem_143001)[i_140745] = sqrt_res_133410;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143026_cached_sizze_143827 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143026, &mem_143026_cached_sizze_143827, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143027_cached_sizze_143828 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143027, &mem_143027_cached_sizze_143828, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140754 = 0; i_140754 < (int64_t) 16; i_140754++) {
        // futhark/microgpt.fut:389:39-51
        
        double zt_lhs_133524 = ((double *) mem_143000)[i_140754];
        
        // futhark/microgpt.fut:389:93-105
        
        double zp_lhs_133525 = ((double *) mem_141082)[i_140754];
        
        // futhark/microgpt.fut:389:93-133
        
        double zp_res_133526 = 1.0e-5 + zp_lhs_133525;
        
        // futhark/microgpt.fut:389:85-133
        
        double sqrt_res_133527 = futrts_sqrt64(zp_res_133526);
        
        // futhark/microgpt.fut:389:71-135
        
        double zt_res_133528 = 2.0 * sqrt_res_133527;
        
        // futhark/microgpt.fut:389:57-135
        
        double zs_res_133529 = 1.0 / zt_res_133528;
        
        // futhark/microgpt.fut:389:39-135
        
        double zt_res_133530 = zt_lhs_133524 * zs_res_133529;
        
        // futhark/microgpt.fut:402:39-51
        
        double zt_lhs_133537 = ((double *) mem_142998)[i_140754];
        
        // futhark/microgpt.fut:402:93-105
        
        double zp_lhs_133538 = ((double *) mem_141081)[i_140754];
        
        // futhark/microgpt.fut:402:93-133
        
        double zp_res_133539 = 1.0e-5 + zp_lhs_133538;
        
        // futhark/microgpt.fut:402:85-133
        
        double sqrt_res_133540 = futrts_sqrt64(zp_res_133539);
        
        // futhark/microgpt.fut:402:71-135
        
        double zt_res_133541 = 2.0 * sqrt_res_133540;
        
        // futhark/microgpt.fut:402:57-135
        
        double zs_res_133542 = 1.0 / zt_res_133541;
        
        // futhark/microgpt.fut:402:39-135
        
        double zt_res_133543 = zt_lhs_133537 * zs_res_133542;
        
        ((double *) mem_143026)[i_140754] = zt_res_133543;
        ((double *) mem_143027)[i_140754] = zt_res_133530;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143040_cached_sizze_143829 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143040, &mem_143040_cached_sizze_143829, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143041, (int64_t) 2048, "mem_143041")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143050_cached_sizze_143830 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143050, &mem_143050_cached_sizze_143830, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143051_cached_sizze_143831 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143051, &mem_143051_cached_sizze_143831, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140768 = 0; i_140768 < (int64_t) 16; i_140768++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140761 = 0; i_140761 < (int64_t) 16; i_140761++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_138999;
            double r_139001 = 0.0;
            
            for (int64_t i_139000 = 0; i_139000 < (int64_t) 16; i_139000++) {
                // futhark/microgpt.fut:390:60-148
                
                bool cond_139002 = i_140768 == i_139000;
                
                // futhark/microgpt.fut:390:60-148
                
                double zp_lhs_139003;
                
                if (cond_139002) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_139691 = ((double *) mem_142967)[i_139000 * (int64_t) 16 + i_140761];
                    
                    // futhark/microgpt.fut:390:124-136
                    
                    double zs_rhs_139692 = ((double *) mem_143001)[i_139000];
                    
                    // futhark/microgpt.fut:390:116-136
                    
                    double zs_res_139693 = 1.0 / zs_rhs_139692;
                    
                    // futhark/microgpt.fut:390:90-136
                    
                    double zt_res_139694 = zt_lhs_139691 * zs_res_139693;
                    
                    zp_lhs_139003 = zt_res_139694;
                } else {
                    zp_lhs_139003 = 0.0;
                }
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_139012;
                double r_139014 = 0.0;
                
                for (int64_t i_139013 = 0; i_139013 < (int64_t) 16; i_139013++) {
                    // futhark/microgpt.fut:390:178-340
                    
                    double zp_lhs_139015;
                    
                    if (cond_139002) {
                        // futhark/microgpt.fut:390:208-329
                        
                        bool cond_139705 = i_140761 == i_139013;
                        
                        // futhark/microgpt.fut:390:208-329
                        
                        double zp_lhs_t_res_139706;
                        
                        if (cond_139705) {
                            // futhark/microgpt.fut:390:239-251
                            
                            double zs_lhs_139707 = ((double *) mem_143027)[i_139000];
                            
                            // futhark/microgpt.fut:390:239-266
                            
                            double zs_res_139708 = zs_lhs_139707 / 16.0;
                            
                            // futhark/microgpt.fut:61:46-49
                            
                            double zp_lhs_139713 = ((double *) wpe_mem_141056.mem)[i_139000 * (int64_t) 16 + i_140761];
                            
                            // futhark/microgpt.fut:61:46-49
                            
                            double zp_rhs_139714 = ((double *) mem_141065)[i_139000 * (int64_t) 16 + i_140761];
                            
                            // futhark/microgpt.fut:390:273-317
                            
                            double zp_res_139715 = zp_lhs_139713 + zp_rhs_139714;
                            
                            // futhark/microgpt.fut:390:252-317
                            
                            double zt_res_139716 = zs_res_139708 * zp_res_139715;
                            
                            zp_lhs_t_res_139706 = zt_res_139716;
                        } else {
                            zp_lhs_t_res_139706 = 0.0;
                        }
                        zp_lhs_139015 = zp_lhs_t_res_139706;
                    } else {
                        zp_lhs_139015 = 0.0;
                    }
                    // futhark/microgpt.fut:390:347-509
                    
                    double zp_rhs_139028;
                    
                    if (cond_139002) {
                        // futhark/microgpt.fut:390:377-498
                        
                        bool cond_139727 = i_140761 == i_139013;
                        
                        // futhark/microgpt.fut:390:377-498
                        
                        double zp_rhs_t_res_139728;
                        
                        if (cond_139727) {
                            // futhark/microgpt.fut:390:408-420
                            
                            double zs_lhs_139729 = ((double *) mem_143027)[i_139000];
                            
                            // futhark/microgpt.fut:390:408-435
                            
                            double zs_res_139730 = zs_lhs_139729 / 16.0;
                            
                            // futhark/microgpt.fut:61:46-49
                            
                            double zp_lhs_139735 = ((double *) wpe_mem_141056.mem)[i_139000 * (int64_t) 16 + i_140761];
                            
                            // futhark/microgpt.fut:61:46-49
                            
                            double zp_rhs_139736 = ((double *) mem_141065)[i_139000 * (int64_t) 16 + i_140761];
                            
                            // futhark/microgpt.fut:390:442-486
                            
                            double zp_res_139737 = zp_lhs_139735 + zp_rhs_139736;
                            
                            // futhark/microgpt.fut:390:421-486
                            
                            double zt_res_139738 = zs_res_139730 * zp_res_139737;
                            
                            zp_rhs_t_res_139728 = zt_res_139738;
                        } else {
                            zp_rhs_t_res_139728 = 0.0;
                        }
                        zp_rhs_139028 = zp_rhs_t_res_139728;
                    } else {
                        zp_rhs_139028 = 0.0;
                    }
                    // futhark/microgpt.fut:390:178-509
                    
                    double zp_res_139041 = zp_lhs_139015 + zp_rhs_139028;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_139042 = r_139014 + zp_res_139041;
                    double r_tmp_143462 = zp_res_139042;
                    
                    r_139014 = r_tmp_143462;
                }
                defunc_0_lifted_lambda_res_139012 = r_139014;
                // futhark/microgpt.fut:390:60-512
                
                double zp_res_139043 = zp_lhs_139003 + defunc_0_lifted_lambda_res_139012;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_139044 = r_139001 + zp_res_139043;
                double r_tmp_143461 = zp_res_139044;
                
                r_139001 = r_tmp_143461;
            }
            defunc_0_lifted_lambda_res_138999 = r_139001;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_139047;
            double r_139049 = 0.0;
            
            for (int64_t i_139048 = 0; i_139048 < (int64_t) 16; i_139048++) {
                // futhark/microgpt.fut:403:60-148
                
                bool cond_139050 = i_140768 == i_139048;
                
                // futhark/microgpt.fut:403:60-148
                
                double zp_lhs_139051;
                
                if (cond_139050) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_139743 = ((double *) mem_142966)[i_139048 * (int64_t) 16 + i_140761];
                    
                    // futhark/microgpt.fut:403:124-136
                    
                    double zs_rhs_139744 = ((double *) mem_142999)[i_139048];
                    
                    // futhark/microgpt.fut:403:116-136
                    
                    double zs_res_139745 = 1.0 / zs_rhs_139744;
                    
                    // futhark/microgpt.fut:403:90-136
                    
                    double zt_res_139746 = zt_lhs_139743 * zs_res_139745;
                    
                    zp_lhs_139051 = zt_res_139746;
                } else {
                    zp_lhs_139051 = 0.0;
                }
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_139060;
                double r_139062 = 0.0;
                
                for (int64_t i_139061 = 0; i_139061 < (int64_t) 16; i_139061++) {
                    // futhark/microgpt.fut:403:178-340
                    
                    double zp_lhs_139063;
                    
                    if (cond_139050) {
                        // futhark/microgpt.fut:403:208-329
                        
                        bool cond_139757 = i_140761 == i_139061;
                        
                        // futhark/microgpt.fut:403:208-329
                        
                        double zp_lhs_t_res_139758;
                        
                        if (cond_139757) {
                            // futhark/microgpt.fut:403:239-251
                            
                            double zs_lhs_139759 = ((double *) mem_143026)[i_139048];
                            
                            // futhark/microgpt.fut:403:239-266
                            
                            double zs_res_139760 = zs_lhs_139759 / 16.0;
                            
                            // futhark/microgpt.fut:61:46-49
                            
                            double zp_lhs_139765 = ((double *) wpe_mem_141056.mem)[i_139048 * (int64_t) 16 + i_140761];
                            
                            // futhark/microgpt.fut:61:46-49
                            
                            double zp_rhs_139766 = ((double *) mem_141065)[i_139048 * (int64_t) 16 + i_140761];
                            
                            // futhark/microgpt.fut:403:273-317
                            
                            double zp_res_139767 = zp_lhs_139765 + zp_rhs_139766;
                            
                            // futhark/microgpt.fut:403:252-317
                            
                            double zt_res_139768 = zs_res_139760 * zp_res_139767;
                            
                            zp_lhs_t_res_139758 = zt_res_139768;
                        } else {
                            zp_lhs_t_res_139758 = 0.0;
                        }
                        zp_lhs_139063 = zp_lhs_t_res_139758;
                    } else {
                        zp_lhs_139063 = 0.0;
                    }
                    // futhark/microgpt.fut:403:347-509
                    
                    double zp_rhs_139076;
                    
                    if (cond_139050) {
                        // futhark/microgpt.fut:403:377-498
                        
                        bool cond_139779 = i_140761 == i_139061;
                        
                        // futhark/microgpt.fut:403:377-498
                        
                        double zp_rhs_t_res_139780;
                        
                        if (cond_139779) {
                            // futhark/microgpt.fut:403:408-420
                            
                            double zs_lhs_139781 = ((double *) mem_143026)[i_139048];
                            
                            // futhark/microgpt.fut:403:408-435
                            
                            double zs_res_139782 = zs_lhs_139781 / 16.0;
                            
                            // futhark/microgpt.fut:61:46-49
                            
                            double zp_lhs_139787 = ((double *) wpe_mem_141056.mem)[i_139048 * (int64_t) 16 + i_140761];
                            
                            // futhark/microgpt.fut:61:46-49
                            
                            double zp_rhs_139788 = ((double *) mem_141065)[i_139048 * (int64_t) 16 + i_140761];
                            
                            // futhark/microgpt.fut:403:442-486
                            
                            double zp_res_139789 = zp_lhs_139787 + zp_rhs_139788;
                            
                            // futhark/microgpt.fut:403:421-486
                            
                            double zt_res_139790 = zs_res_139782 * zp_res_139789;
                            
                            zp_rhs_t_res_139780 = zt_res_139790;
                        } else {
                            zp_rhs_t_res_139780 = 0.0;
                        }
                        zp_rhs_139076 = zp_rhs_t_res_139780;
                    } else {
                        zp_rhs_139076 = 0.0;
                    }
                    // futhark/microgpt.fut:403:178-509
                    
                    double zp_res_139089 = zp_lhs_139063 + zp_rhs_139076;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_139090 = r_139062 + zp_res_139089;
                    double r_tmp_143464 = zp_res_139090;
                    
                    r_139062 = r_tmp_143464;
                }
                defunc_0_lifted_lambda_res_139060 = r_139062;
                // futhark/microgpt.fut:403:60-512
                
                double zp_res_139091 = zp_lhs_139051 + defunc_0_lifted_lambda_res_139060;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_139092 = r_139049 + zp_res_139091;
                double r_tmp_143463 = zp_res_139092;
                
                r_139049 = r_tmp_143463;
            }
            defunc_0_lifted_lambda_res_139047 = r_139049;
            ((double *) mem_143050)[i_140761] = defunc_0_lifted_lambda_res_139047;
            ((double *) mem_143051)[i_140761] = defunc_0_lifted_lambda_res_138999;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143040, i_140768 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143050, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143041.mem, i_140768 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143051, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143072, (int64_t) 8192, "mem_143072")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143077_cached_sizze_143832 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143077, &mem_143077_cached_sizze_143832, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140777 = 0; i_140777 < (int64_t) 64; i_140777++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140773 = 0; i_140773 < (int64_t) 16; i_140773++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_128860;
            double r_128862 = 0.0;
            
            for (int64_t i_128861 = 0; i_128861 < (int64_t) 16; i_128861++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_128863 = ((double *) mem_142249)[i_128861 * (int64_t) 64 + i_140777];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_128864 = ((double *) mem_141963)[i_128861 * (int64_t) 16 + i_140773];
                
                // futhark/microgpt.fut:395:67-111
                
                double zt_res_128865 = zt_lhs_128863 * zt_rhs_128864;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_128866 = r_128862 + zt_res_128865;
                double r_tmp_143467 = zp_res_128866;
                
                r_128862 = r_tmp_143467;
            }
            defunc_0_lifted_lambda_res_128860 = r_128862;
            ((double *) mem_143077)[i_140773] = defunc_0_lifted_lambda_res_128860;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143072.mem, i_140777 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143077, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143088, (int64_t) 3456, "mem_143088")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143089, (int64_t) 3456, "mem_143089")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143098_cached_sizze_143833 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143098, &mem_143098_cached_sizze_143833, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143099_cached_sizze_143834 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143099, &mem_143099_cached_sizze_143834, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_140790 = 0; i_140790 < (int64_t) 27; i_140790++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_140783 = 0; i_140783 < (int64_t) 16; i_140783++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_139321;
            double r_139323 = 0.0;
            
            for (int64_t i_139322 = 0; i_139322 < (int64_t) 16; i_139322++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_139324 = ((double *) mem_142216)[i_139322 * (int64_t) 27 + i_140790];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_139325 = ((double *) mem_142011)[i_139322 * (int64_t) 16 + i_140783];
                
                // futhark/microgpt.fut:397:68-111
                
                double zt_res_139326 = zt_lhs_139324 * zt_rhs_139325;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_139327 = r_139323 + zt_res_139326;
                double r_tmp_143472 = zp_res_139327;
                
                r_139323 = r_tmp_143472;
            }
            defunc_0_lifted_lambda_res_139321 = r_139323;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_139330;
            double r_139332 = 0.0;
            
            for (int64_t i_139331 = 0; i_139331 < (int64_t) 16; i_139331++) {
                // futhark/microgpt.fut:460:62-71
                
                int64_t zeze_lhs_139333 = ((int64_t *) tokens_mem_141062.mem)[i_139331];
                
                // futhark/microgpt.fut:460:58-109
                
                bool cond_139334 = zeze_lhs_139333 == i_140790;
                
                // futhark/microgpt.fut:460:58-109
                
                double lifted_lambda_res_139335;
                
                if (cond_139334) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double lifted_lambda_res_t_res_139798 = ((double *) mem_143040)[i_139331 * (int64_t) 16 + i_140783];
                    
                    lifted_lambda_res_139335 = lifted_lambda_res_t_res_139798;
                } else {
                    lifted_lambda_res_139335 = 0.0;
                }
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_139341 = r_139332 + lifted_lambda_res_139335;
                double r_tmp_143473 = zp_res_139341;
                
                r_139332 = r_tmp_143473;
            }
            defunc_0_lifted_lambda_res_139330 = r_139332;
            ((double *) mem_143098)[i_140783] = defunc_0_lifted_lambda_res_139330;
            ((double *) mem_143099)[i_140783] = defunc_0_lifted_lambda_res_139321;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143088.mem, i_140790 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143098, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143089.mem, i_140790 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143099, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    if (memblock_set(ctx, &mem_out_143138, &mem_143088, "mem_143088") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_143139, &mem_143041, "mem_143041") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_143140, &mem_142883, "mem_142883") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_143141, &mem_142882, "mem_142882") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_143142, &mem_142881, "mem_142881") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_143143, &mem_142801, "mem_142801") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_143144, &mem_143072, "mem_143072") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_143145, &mem_142248, "mem_142248") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_143146, &mem_143089, "mem_143089") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143590, &mem_out_143138, "mem_out_143138") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143591, &mem_out_143139, "mem_out_143139") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143592, &mem_out_143140, "mem_out_143140") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143593, &mem_out_143141, "mem_out_143141") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143594, &mem_out_143142, "mem_out_143142") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143595, &mem_out_143143, "mem_out_143143") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143596, &mem_out_143144, "mem_out_143144") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143597, &mem_out_143145, "mem_out_143145") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143598, &mem_out_143146, "mem_out_143146") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_141065);
        free(mem_141070);
        free(mem_141081);
        free(mem_141082);
        free(mem_141083);
        free(mem_141102);
        free(mem_141109);
        free(mem_141114);
        free(mem_141125);
        free(mem_141130);
        free(mem_141141);
        free(mem_141142);
        free(mem_141155);
        free(mem_141162);
        free(mem_141167);
        free(mem_141178);
        free(mem_141183);
        free(mem_141194);
        free(mem_141195);
        free(mem_141196);
        free(mem_141212);
        free(mem_141213);
        free(mem_141214);
        free(mem_141227);
        free(mem_141228);
        free(mem_141229);
        free(mem_141275);
        free(mem_141276);
        free(mem_141277);
        free(mem_141278);
        free(mem_141299);
        free(mem_141300);
        free(mem_141301);
        free(mem_141302);
        free(mem_141319);
        free(mem_141320);
        free(mem_141321);
        free(mem_141322);
        free(mem_141383);
        free(mem_141384);
        free(mem_141385);
        free(mem_141386);
        free(mem_141407);
        free(mem_141408);
        free(mem_141409);
        free(mem_141410);
        free(mem_141427);
        free(mem_141428);
        free(mem_141429);
        free(mem_141430);
        free(mem_141491);
        free(mem_141492);
        free(mem_141493);
        free(mem_141494);
        free(mem_141495);
        free(mem_141496);
        free(mem_141497);
        free(mem_141498);
        free(mem_141531);
        free(mem_141532);
        free(mem_141533);
        free(mem_141534);
        free(mem_141535);
        free(mem_141536);
        free(mem_141537);
        free(mem_141538);
        free(mem_141619);
        free(mem_141620);
        free(mem_141621);
        free(mem_141622);
        free(mem_141643);
        free(mem_141644);
        free(mem_141645);
        free(mem_141646);
        free(mem_141663);
        free(mem_141664);
        free(mem_141665);
        free(mem_141666);
        free(mem_141727);
        free(mem_141728);
        free(mem_141737);
        free(mem_141738);
        free(mem_141759);
        free(mem_141760);
        free(mem_141771);
        free(mem_141772);
        free(mem_141781);
        free(mem_141782);
        free(mem_141813);
        free(mem_141814);
        free(mem_141825);
        free(mem_141826);
        free(mem_141835);
        free(mem_141836);
        free(mem_141867);
        free(mem_141873);
        free(mem_141878);
        free(mem_141894);
        free(mem_141899);
        free(mem_141910);
        free(mem_141915);
        free(mem_141926);
        free(mem_141927);
        free(mem_141940);
        free(mem_141947);
        free(mem_141952);
        free(mem_141963);
        free(mem_141968);
        free(mem_141979);
        free(mem_141984);
        free(mem_141995);
        free(mem_142000);
        free(mem_142011);
        free(mem_142016);
        free(mem_142027);
        free(mem_142032);
        free(mem_142043);
        free(mem_142044);
        free(mem_142045);
        free(mem_142046);
        free(mem_142064);
        free(mem_142069);
        free(mem_142073);
        free(mem_142080);
        free(mem_142114);
        free(mem_142120);
        free(mem_142125);
        free(mem_142141);
        free(mem_142142);
        free(mem_142151);
        free(mem_142152);
        free(mem_142173);
        free(mem_142179);
        free(mem_142184);
        free(mem_142200);
        free(mem_142205);
        free(mem_142216);
        free(mem_142221);
        free(mem_142232);
        free(mem_142237);
        free(mem_142249);
        free(mem_142258);
        free(mem_142259);
        free(mem_142280);
        free(mem_142285);
        free(mem_142296);
        free(mem_142297);
        free(mem_142310);
        free(mem_142317);
        free(mem_142322);
        free(mem_142333);
        free(mem_142339);
        free(mem_142344);
        free(mem_142360);
        free(mem_142361);
        free(mem_142362);
        free(mem_142378);
        free(mem_142379);
        free(mem_142380);
        free(mem_142393);
        free(mem_142394);
        free(mem_142435);
        free(mem_142436);
        free(mem_142447);
        free(mem_142448);
        free(mem_142457);
        free(mem_142458);
        free(mem_142489);
        free(mem_142490);
        free(mem_142501);
        free(mem_142502);
        free(mem_142511);
        free(mem_142512);
        free(mem_142543);
        free(mem_142544);
        free(mem_142545);
        free(mem_142546);
        free(mem_142563);
        free(mem_142564);
        free(mem_142565);
        free(mem_142566);
        free(mem_142607);
        free(mem_142608);
        free(mem_142619);
        free(mem_142620);
        free(mem_142629);
        free(mem_142630);
        free(mem_142661);
        free(mem_142662);
        free(mem_142671);
        free(mem_142672);
        free(mem_142693);
        free(mem_142694);
        free(mem_142705);
        free(mem_142706);
        free(mem_142715);
        free(mem_142716);
        free(mem_142747);
        free(mem_142748);
        free(mem_142759);
        free(mem_142760);
        free(mem_142769);
        free(mem_142770);
        free(mem_142802);
        free(mem_142803);
        free(mem_142804);
        free(mem_142821);
        free(mem_142822);
        free(mem_142823);
        free(mem_142824);
        free(mem_142865);
        free(mem_142870);
        free(mem_142884);
        free(mem_142885);
        free(mem_142904);
        free(mem_142905);
        free(mem_142906);
        free(mem_142943);
        free(mem_142950);
        free(mem_142955);
        free(mem_142966);
        free(mem_142967);
        free(mem_142976);
        free(mem_142977);
        free(mem_142998);
        free(mem_142999);
        free(mem_143000);
        free(mem_143001);
        free(mem_143026);
        free(mem_143027);
        free(mem_143040);
        free(mem_143050);
        free(mem_143051);
        free(mem_143077);
        free(mem_143098);
        free(mem_143099);
        if (memblock_unref(ctx, &mem_143089, "mem_143089") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_143088, "mem_143088") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_143072, "mem_143072") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_143041, "mem_143041") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_142883, "mem_142883") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_142882, "mem_142882") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_142881, "mem_142881") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_142801, "mem_142801") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_142248, "mem_142248") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143146, "mem_out_143146") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143145, "mem_out_143145") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143144, "mem_out_143144") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143143, "mem_out_143143") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143142, "mem_out_143142") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143141, "mem_out_143141") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143140, "mem_out_143140") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143139, "mem_out_143139") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143138, "mem_out_143138") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_make_params(struct futhark_context *ctx, struct memblock *mem_out_p_143835, struct memblock *mem_out_p_143836, struct memblock *mem_out_p_143837, struct memblock *mem_out_p_143838, struct memblock *mem_out_p_143839, struct memblock *mem_out_p_143840, struct memblock *mem_out_p_143841, struct memblock *mem_out_p_143842, struct memblock *mem_out_p_143843, struct memblock wte_mem_141053, struct memblock wpe_mem_141054, struct memblock wqry_mem_141055, struct memblock wkey_mem_141056, struct memblock wval_mem_141057, struct memblock wout_mem_141058, struct memblock wup_mem_141059, struct memblock wdown_mem_141060, struct memblock wvoc_mem_141061, int64_t sl_55409)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_143146;
    
    mem_out_143146.references = NULL;
    
    struct memblock mem_out_143145;
    
    mem_out_143145.references = NULL;
    
    struct memblock mem_out_143144;
    
    mem_out_143144.references = NULL;
    
    struct memblock mem_out_143143;
    
    mem_out_143143.references = NULL;
    
    struct memblock mem_out_143142;
    
    mem_out_143142.references = NULL;
    
    struct memblock mem_out_143141;
    
    mem_out_143141.references = NULL;
    
    struct memblock mem_out_143140;
    
    mem_out_143140.references = NULL;
    
    struct memblock mem_out_143139;
    
    mem_out_143139.references = NULL;
    
    struct memblock mem_out_143138;
    
    mem_out_143138.references = NULL;
    if (memblock_set(ctx, &mem_out_143138, &wdown_mem_141060, "wdown_mem_141060") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_143139, &wkey_mem_141056, "wkey_mem_141056") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_143140, &wout_mem_141058, "wout_mem_141058") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_143141, &wpe_mem_141054, "wpe_mem_141054") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_143142, &wqry_mem_141055, "wqry_mem_141055") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_143143, &wte_mem_141053, "wte_mem_141053") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_143144, &wup_mem_141059, "wup_mem_141059") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_143145, &wval_mem_141057, "wval_mem_141057") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_143146, &wvoc_mem_141061, "wvoc_mem_141061") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143835, &mem_out_143138, "mem_out_143138") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143836, &mem_out_143139, "mem_out_143139") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143837, &mem_out_143140, "mem_out_143140") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143838, &mem_out_143141, "mem_out_143141") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143839, &mem_out_143142, "mem_out_143142") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143840, &mem_out_143143, "mem_out_143143") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143841, &mem_out_143144, "mem_out_143144") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143842, &mem_out_143145, "mem_out_143145") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_143843, &mem_out_143146, "mem_out_143146") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_143146, "mem_out_143146") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143145, "mem_out_143145") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143144, "mem_out_143144") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143143, "mem_out_143143") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143142, "mem_out_143142") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143141, "mem_out_143141") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143140, "mem_out_143140") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143139, "mem_out_143139") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_143138, "mem_out_143138") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_cal_loss(struct futhark_context *ctx, struct futhark_opaque_tup2_f64_arr1d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3)
{
    double prim_out_143139 = 0.0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_143138;
    
    mem_out_143138.references = NULL;
    
    struct memblock mask_mem_141064;
    
    mask_mem_141064.references = NULL;
    
    struct memblock target_mem_141063;
    
    target_mem_141063.references = NULL;
    
    struct memblock tokens_mem_141062;
    
    tokens_mem_141062.references = NULL;
    
    struct memblock wvoc_mem_141061;
    
    wvoc_mem_141061.references = NULL;
    
    struct memblock wval_mem_141060;
    
    wval_mem_141060.references = NULL;
    
    struct memblock wup_mem_141059;
    
    wup_mem_141059.references = NULL;
    
    struct memblock wte_mem_141058;
    
    wte_mem_141058.references = NULL;
    
    struct memblock wqry_mem_141057;
    
    wqry_mem_141057.references = NULL;
    
    struct memblock wpe_mem_141056;
    
    wpe_mem_141056.references = NULL;
    
    struct memblock wout_mem_141055;
    
    wout_mem_141055.references = NULL;
    
    struct memblock wkey_mem_141054;
    
    wkey_mem_141054.references = NULL;
    
    struct memblock wdown_mem_141053;
    
    wdown_mem_141053.references = NULL;
    wdown_mem_141053 = in0->v0->mem;
    wkey_mem_141054 = in0->v1->mem;
    wout_mem_141055 = in0->v2->mem;
    wpe_mem_141056 = in0->v3->mem;
    wqry_mem_141057 = in0->v4->mem;
    wte_mem_141058 = in0->v5->mem;
    wup_mem_141059 = in0->v6->mem;
    wval_mem_141060 = in0->v7->mem;
    wvoc_mem_141061 = in0->v8->mem;
    tokens_mem_141062 = in1->mem;
    target_mem_141063 = in2->mem;
    mask_mem_141064 = in3->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && (((int64_t) 16 == in2->shape[0] && (int64_t) 27 == in2->shape[1]) && ((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_cal_loss(ctx, &mem_out_143138, &prim_out_143139, wdown_mem_141053, wkey_mem_141054, wout_mem_141055, wpe_mem_141056, wqry_mem_141057, wte_mem_141058, wup_mem_141059, wval_mem_141060, wvoc_mem_141061, tokens_mem_141062, target_mem_141063, mask_mem_141064);
        if (ret == 0) {
            assert((*out = (struct futhark_opaque_tup2_f64_arr1d_f64 *) malloc(sizeof(struct futhark_opaque_tup2_f64_arr1d_f64))) != NULL);
            (*out)->v0 = prim_out_143139;
            assert(((*out)->v1 = (struct futhark_f64_1d *) malloc(sizeof(struct futhark_f64_1d))) != NULL);
            (*out)->v1->mem = mem_out_143138;
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
    
    struct memblock mem_out_143138;
    
    mem_out_143138.references = NULL;
    
    struct memblock mask_mem_141063;
    
    mask_mem_141063.references = NULL;
    
    struct memblock tokens_mem_141062;
    
    tokens_mem_141062.references = NULL;
    
    struct memblock wvoc_mem_141061;
    
    wvoc_mem_141061.references = NULL;
    
    struct memblock wval_mem_141060;
    
    wval_mem_141060.references = NULL;
    
    struct memblock wup_mem_141059;
    
    wup_mem_141059.references = NULL;
    
    struct memblock wte_mem_141058;
    
    wte_mem_141058.references = NULL;
    
    struct memblock wqry_mem_141057;
    
    wqry_mem_141057.references = NULL;
    
    struct memblock wpe_mem_141056;
    
    wpe_mem_141056.references = NULL;
    
    struct memblock wout_mem_141055;
    
    wout_mem_141055.references = NULL;
    
    struct memblock wkey_mem_141054;
    
    wkey_mem_141054.references = NULL;
    
    struct memblock wdown_mem_141053;
    
    wdown_mem_141053.references = NULL;
    wdown_mem_141053 = in0->v0->mem;
    wkey_mem_141054 = in0->v1->mem;
    wout_mem_141055 = in0->v2->mem;
    wpe_mem_141056 = in0->v3->mem;
    wqry_mem_141057 = in0->v4->mem;
    wte_mem_141058 = in0->v5->mem;
    wup_mem_141059 = in0->v6->mem;
    wval_mem_141060 = in0->v7->mem;
    wvoc_mem_141061 = in0->v8->mem;
    tokens_mem_141062 = in1->mem;
    mask_mem_141063 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_143138, wdown_mem_141053, wkey_mem_141054, wout_mem_141055, wpe_mem_141056, wqry_mem_141057, wte_mem_141058, wup_mem_141059, wval_mem_141060, wvoc_mem_141061, tokens_mem_141062, mask_mem_141063);
        if (ret == 0) {
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_143138;
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
    
    struct memblock mem_out_143146;
    
    mem_out_143146.references = NULL;
    
    struct memblock mem_out_143145;
    
    mem_out_143145.references = NULL;
    
    struct memblock mem_out_143144;
    
    mem_out_143144.references = NULL;
    
    struct memblock mem_out_143143;
    
    mem_out_143143.references = NULL;
    
    struct memblock mem_out_143142;
    
    mem_out_143142.references = NULL;
    
    struct memblock mem_out_143141;
    
    mem_out_143141.references = NULL;
    
    struct memblock mem_out_143140;
    
    mem_out_143140.references = NULL;
    
    struct memblock mem_out_143139;
    
    mem_out_143139.references = NULL;
    
    struct memblock mem_out_143138;
    
    mem_out_143138.references = NULL;
    
    struct memblock mask_mem_141064;
    
    mask_mem_141064.references = NULL;
    
    struct memblock target_mem_141063;
    
    target_mem_141063.references = NULL;
    
    struct memblock tokens_mem_141062;
    
    tokens_mem_141062.references = NULL;
    
    struct memblock wvoc_mem_141061;
    
    wvoc_mem_141061.references = NULL;
    
    struct memblock wval_mem_141060;
    
    wval_mem_141060.references = NULL;
    
    struct memblock wup_mem_141059;
    
    wup_mem_141059.references = NULL;
    
    struct memblock wte_mem_141058;
    
    wte_mem_141058.references = NULL;
    
    struct memblock wqry_mem_141057;
    
    wqry_mem_141057.references = NULL;
    
    struct memblock wpe_mem_141056;
    
    wpe_mem_141056.references = NULL;
    
    struct memblock wout_mem_141055;
    
    wout_mem_141055.references = NULL;
    
    struct memblock wkey_mem_141054;
    
    wkey_mem_141054.references = NULL;
    
    struct memblock wdown_mem_141053;
    
    wdown_mem_141053.references = NULL;
    wdown_mem_141053 = in0->v0->mem;
    wkey_mem_141054 = in0->v1->mem;
    wout_mem_141055 = in0->v2->mem;
    wpe_mem_141056 = in0->v3->mem;
    wqry_mem_141057 = in0->v4->mem;
    wte_mem_141058 = in0->v5->mem;
    wup_mem_141059 = in0->v6->mem;
    wval_mem_141060 = in0->v7->mem;
    wvoc_mem_141061 = in0->v8->mem;
    tokens_mem_141062 = in1->mem;
    target_mem_141063 = in2->mem;
    mask_mem_141064 = in3->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && (((int64_t) 16 == in2->shape[0] && (int64_t) 27 == in2->shape[1]) && ((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_grad_loss(ctx, &mem_out_143138, &mem_out_143139, &mem_out_143140, &mem_out_143141, &mem_out_143142, &mem_out_143143, &mem_out_143144, &mem_out_143145, &mem_out_143146, wdown_mem_141053, wkey_mem_141054, wout_mem_141055, wpe_mem_141056, wqry_mem_141057, wte_mem_141058, wup_mem_141059, wval_mem_141060, wvoc_mem_141061, tokens_mem_141062, target_mem_141063, mask_mem_141064);
        if (ret == 0) {
            assert((*out = (struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_143138;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_143139;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_143140;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_143141;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_143142;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_143143;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_143144;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_143145;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_143146;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_make_params(struct futhark_context *ctx, struct futhark_opaque_params **out, const struct futhark_f64_2d *in0, const struct futhark_f64_2d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3, const struct futhark_f64_2d *in4, const struct futhark_f64_2d *in5, const struct futhark_f64_2d *in6, const struct futhark_f64_2d *in7, const struct futhark_f64_2d *in8)
{
    int64_t sl_55409 = (int64_t) 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_143146;
    
    mem_out_143146.references = NULL;
    
    struct memblock mem_out_143145;
    
    mem_out_143145.references = NULL;
    
    struct memblock mem_out_143144;
    
    mem_out_143144.references = NULL;
    
    struct memblock mem_out_143143;
    
    mem_out_143143.references = NULL;
    
    struct memblock mem_out_143142;
    
    mem_out_143142.references = NULL;
    
    struct memblock mem_out_143141;
    
    mem_out_143141.references = NULL;
    
    struct memblock mem_out_143140;
    
    mem_out_143140.references = NULL;
    
    struct memblock mem_out_143139;
    
    mem_out_143139.references = NULL;
    
    struct memblock mem_out_143138;
    
    mem_out_143138.references = NULL;
    
    struct memblock wvoc_mem_141061;
    
    wvoc_mem_141061.references = NULL;
    
    struct memblock wdown_mem_141060;
    
    wdown_mem_141060.references = NULL;
    
    struct memblock wup_mem_141059;
    
    wup_mem_141059.references = NULL;
    
    struct memblock wout_mem_141058;
    
    wout_mem_141058.references = NULL;
    
    struct memblock wval_mem_141057;
    
    wval_mem_141057.references = NULL;
    
    struct memblock wkey_mem_141056;
    
    wkey_mem_141056.references = NULL;
    
    struct memblock wqry_mem_141055;
    
    wqry_mem_141055.references = NULL;
    
    struct memblock wpe_mem_141054;
    
    wpe_mem_141054.references = NULL;
    
    struct memblock wte_mem_141053;
    
    wte_mem_141053.references = NULL;
    wte_mem_141053 = in0->mem;
    sl_55409 = in0->shape[1];
    wpe_mem_141054 = in1->mem;
    sl_55409 = in1->shape[0];
    wqry_mem_141055 = in2->mem;
    wkey_mem_141056 = in3->mem;
    wval_mem_141057 = in4->mem;
    wout_mem_141058 = in5->mem;
    wup_mem_141059 = in6->mem;
    wdown_mem_141060 = in7->mem;
    wvoc_mem_141061 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && sl_55409 == in0->shape[1]) && ((sl_55409 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_make_params(ctx, &mem_out_143138, &mem_out_143139, &mem_out_143140, &mem_out_143141, &mem_out_143142, &mem_out_143143, &mem_out_143144, &mem_out_143145, &mem_out_143146, wte_mem_141053, wpe_mem_141054, wqry_mem_141055, wkey_mem_141056, wval_mem_141057, wout_mem_141058, wup_mem_141059, wdown_mem_141060, wvoc_mem_141061, sl_55409);
        if (ret == 0) {
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_143138;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_143139;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_143140;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_143141;
            (*out)->v3->shape[0] = sl_55409;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_143142;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_143143;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = sl_55409;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_143144;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_143145;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_143146;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
