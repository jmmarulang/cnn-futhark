
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
struct futhark_f64_2d;
struct futhark_f64_2d *futhark_new_f64_2d(struct futhark_context *ctx, const double *data, int64_t dim0, int64_t dim1);
struct futhark_f64_2d *futhark_new_raw_f64_2d(struct futhark_context *ctx, unsigned char *data, int64_t dim0, int64_t dim1);
int futhark_free_f64_2d(struct futhark_context *ctx, struct futhark_f64_2d *arr);
int futhark_values_f64_2d(struct futhark_context *ctx, struct futhark_f64_2d *arr, double *data);
int futhark_index_f64_2d(struct futhark_context *ctx, double *out, struct futhark_f64_2d *arr, int64_t i0, int64_t i1);
unsigned char *futhark_values_raw_f64_2d(struct futhark_context *ctx, struct futhark_f64_2d *arr);
const int64_t *futhark_shape_f64_2d(struct futhark_context *ctx, struct futhark_f64_2d *arr);
struct futhark_f64_3d;
struct futhark_f64_3d *futhark_new_f64_3d(struct futhark_context *ctx, const double *data, int64_t dim0, int64_t dim1, int64_t dim2);
struct futhark_f64_3d *futhark_new_raw_f64_3d(struct futhark_context *ctx, unsigned char *data, int64_t dim0, int64_t dim1, int64_t dim2);
int futhark_free_f64_3d(struct futhark_context *ctx, struct futhark_f64_3d *arr);
int futhark_values_f64_3d(struct futhark_context *ctx, struct futhark_f64_3d *arr, double *data);
int futhark_index_f64_3d(struct futhark_context *ctx, double *out, struct futhark_f64_3d *arr, int64_t i0, int64_t i1, int64_t i2);
unsigned char *futhark_values_raw_f64_3d(struct futhark_context *ctx, struct futhark_f64_3d *arr);
const int64_t *futhark_shape_f64_3d(struct futhark_context *ctx, struct futhark_f64_3d *arr);
struct futhark_i64_1d;
struct futhark_i64_1d *futhark_new_i64_1d(struct futhark_context *ctx, const int64_t *data, int64_t dim0);
struct futhark_i64_1d *futhark_new_raw_i64_1d(struct futhark_context *ctx, unsigned char *data, int64_t dim0);
int futhark_free_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr);
int futhark_values_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr, int64_t *data);
int futhark_index_i64_1d(struct futhark_context *ctx, int64_t *out, struct futhark_i64_1d *arr, int64_t i0);
unsigned char *futhark_values_raw_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr);
const int64_t *futhark_shape_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr);
struct futhark_i64_2d;
struct futhark_i64_2d *futhark_new_i64_2d(struct futhark_context *ctx, const int64_t *data, int64_t dim0, int64_t dim1);
struct futhark_i64_2d *futhark_new_raw_i64_2d(struct futhark_context *ctx, unsigned char *data, int64_t dim0, int64_t dim1);
int futhark_free_i64_2d(struct futhark_context *ctx, struct futhark_i64_2d *arr);
int futhark_values_i64_2d(struct futhark_context *ctx, struct futhark_i64_2d *arr, int64_t *data);
int futhark_index_i64_2d(struct futhark_context *ctx, int64_t *out, struct futhark_i64_2d *arr, int64_t i0, int64_t i1);
unsigned char *futhark_values_raw_i64_2d(struct futhark_context *ctx, struct futhark_i64_2d *arr);
const int64_t *futhark_shape_i64_2d(struct futhark_context *ctx, struct futhark_i64_2d *arr);

// Opaque values
struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64;
struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64;
struct futhark_opaque_params;
int futhark_free_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64(struct futhark_context *ctx, struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj);
int futhark_store_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64(struct futhark_context *ctx, const struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj, void **p, size_t *n);
struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *futhark_restore_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64(struct futhark_context *ctx, const void *p);
int futhark_project_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_0(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj);
int futhark_project_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_1(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj);
int futhark_project_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_2(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj);
int futhark_new_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64(struct futhark_context *ctx, struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *f_0, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *f_1, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *f_2);
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
int futhark_entry_forward_seq(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2);
int futhark_entry_to_params(struct futhark_context *ctx, struct futhark_opaque_params **out, const struct futhark_f64_2d *in0, const struct futhark_f64_2d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3, const struct futhark_f64_2d *in4, const struct futhark_f64_2d *in5, const struct futhark_f64_2d *in6, const struct futhark_f64_2d *in7, const struct futhark_f64_2d *in8);
int futhark_entry_train(struct futhark_context *ctx, struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_opaque_params *in1, const struct futhark_opaque_params *in2, const struct futhark_f64_3d *in3, const struct futhark_i64_1d *in4, const struct futhark_i64_2d *in5);
int futhark_entry_zero_params(struct futhark_context *ctx, struct futhark_opaque_params **out);

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

const struct type type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRZR;
const struct type type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR;
const struct type type_ZMZNZMZNZMZNf64;
const struct type type_ZMZNZMZNf64;
const struct type type_ZMZNZMZNi64;
const struct type type_ZMZNi64;
const struct type type_params;
const struct field type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRZR_fields[] = {{.name ="0", .type =&type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR, .project =(project_fn) futhark_project_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_0}, {.name ="1", .type =&type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR, .project =(project_fn) futhark_project_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_1}, {.name ="2", .type =&type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR, .project =(project_fn) futhark_project_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_2}};
int futhark_new_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_wrap(struct futhark_context *ctx, void **outp, const void *fields[])
{
    struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 * *out = (struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 * *) outp;
    const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 * v0 = *(const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 * *) fields[0];
    const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 * v1 = *(const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 * *) fields[1];
    const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 * v2 = *(const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 * *) fields[2];
    
    return futhark_new_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64(ctx, out, v0, v1, v2);
}
const struct record type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRZR_record = {.num_fields =3, .fields =type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRZR_fields, .new =futhark_new_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_wrap};
const struct opaque_aux type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRZR_aux = {.store =(opaque_store_fn) futhark_store_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64, .restore =(opaque_restore_fn) futhark_restore_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64, .free =(opaque_free_fn) futhark_free_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64};
const struct type type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRZR = {.name ="(([][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64), ([][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64), ([][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64))", .restore =(restore_fn) restore_opaque, .store =(store_fn) store_opaque, .free =(free_fn) free_opaque, .aux =&type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRZR_aux, .kind =RECORD, .info =&type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRZR_record};
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
void *futhark_new_f64_3d_aux_wrap(struct futhark_context *ctx, const void *p, const int64_t *shape)
{
    return futhark_new_f64_3d(ctx, p, shape[0], shape[1], shape[2]);
}
int futhark_new_f64_3d_wrap(struct futhark_context *ctx, struct futhark_f64_3d * *outp, double *ps[], const int64_t *shape)
{
    int64_t n_values = 1;
    
    for (int i = 0; i < 3; ++i)
        n_values *= shape[i];
    
    double *values = alloca(n_values * sizeof(double));
    
    for (int64_t i = 0; i < n_values; ++i)
        values[i] = *ps[i];
    *outp = futhark_new_f64_3d(ctx, values, shape[0], shape[1], shape[2]);
    return 0;
}
int futhark_new_f64_3d_set(struct futhark_context *ctx, struct futhark_f64_3d * arr, double *val, const int64_t *is)
{
    const int64_t *shape = futhark_shape_f64_3d(ctx, arr);
    uint64_t idx = is[0];
    
    for (int i = 1; i < 3; ++i) {
        idx *= shape[i - 1];
        idx += is[i];
    }
    ((double *) futhark_values_raw_f64_3d(ctx, arr))[idx] = *val;
    return 0;
}
int futhark_index_f64_3d_wrap(struct futhark_context *ctx, void *dest, struct futhark_f64_3d * arr, const int64_t *is)
{
    return futhark_index_f64_3d(ctx, dest, arr, is[0], is[1], is[2]);
}
const struct array type_ZMZNZMZNZMZNf64_array = {.rank =3, .element_type =&type_f64, .new =(array_new_fn) futhark_new_f64_3d_wrap, .set =(array_set_fn) futhark_new_f64_3d_set, .shape =(array_shape_fn) futhark_shape_f64_3d, .index =(array_index_fn) futhark_index_f64_3d_wrap};
const struct array_aux type_ZMZNZMZNZMZNf64_aux = {.name ="[][][]f64", .rank =3, .info =&f64_info, .new =(aux_array_new_fn) futhark_new_f64_3d_aux_wrap, .free =(aux_array_free_fn) futhark_free_f64_3d, .shape =(aux_array_shape_fn) futhark_shape_f64_3d, .values =(aux_array_values_fn) futhark_values_f64_3d};
const struct type type_ZMZNZMZNZMZNf64 = {.name ="[][][]f64", .restore =(restore_fn) restore_array, .store =(store_fn) store_array, .free =(free_fn) free_array, .aux =&type_ZMZNZMZNZMZNf64_aux, .kind =ARRAY, .info =&type_ZMZNZMZNZMZNf64_array};
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
void *futhark_new_i64_2d_aux_wrap(struct futhark_context *ctx, const void *p, const int64_t *shape)
{
    return futhark_new_i64_2d(ctx, p, shape[0], shape[1]);
}
int futhark_new_i64_2d_wrap(struct futhark_context *ctx, struct futhark_i64_2d * *outp, int64_t *ps[], const int64_t *shape)
{
    int64_t n_values = 1;
    
    for (int i = 0; i < 2; ++i)
        n_values *= shape[i];
    
    int64_t *values = alloca(n_values * sizeof(int64_t));
    
    for (int64_t i = 0; i < n_values; ++i)
        values[i] = *ps[i];
    *outp = futhark_new_i64_2d(ctx, values, shape[0], shape[1]);
    return 0;
}
int futhark_new_i64_2d_set(struct futhark_context *ctx, struct futhark_i64_2d * arr, int64_t *val, const int64_t *is)
{
    const int64_t *shape = futhark_shape_i64_2d(ctx, arr);
    uint64_t idx = is[0];
    
    for (int i = 1; i < 2; ++i) {
        idx *= shape[i - 1];
        idx += is[i];
    }
    ((int64_t *) futhark_values_raw_i64_2d(ctx, arr))[idx] = *val;
    return 0;
}
int futhark_index_i64_2d_wrap(struct futhark_context *ctx, void *dest, struct futhark_i64_2d * arr, const int64_t *is)
{
    return futhark_index_i64_2d(ctx, dest, arr, is[0], is[1]);
}
const struct array type_ZMZNZMZNi64_array = {.rank =2, .element_type =&type_i64, .new =(array_new_fn) futhark_new_i64_2d_wrap, .set =(array_set_fn) futhark_new_i64_2d_set, .shape =(array_shape_fn) futhark_shape_i64_2d, .index =(array_index_fn) futhark_index_i64_2d_wrap};
const struct array_aux type_ZMZNZMZNi64_aux = {.name ="[][]i64", .rank =2, .info =&i64_info, .new =(aux_array_new_fn) futhark_new_i64_2d_aux_wrap, .free =(aux_array_free_fn) futhark_free_i64_2d, .shape =(aux_array_shape_fn) futhark_shape_i64_2d, .values =(aux_array_values_fn) futhark_values_i64_2d};
const struct type type_ZMZNZMZNi64 = {.name ="[][]i64", .restore =(restore_fn) restore_array, .store =(store_fn) store_array, .free =(free_fn) free_array, .aux =&type_ZMZNZMZNi64_aux, .kind =ARRAY, .info =&type_ZMZNZMZNi64_array};
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
const struct type *to_params_in_types[] = {&type_ZMZNZMZNf64, &type_ZMZNZMZNf64, &type_ZMZNZMZNf64, &type_ZMZNZMZNf64, &type_ZMZNZMZNf64, &type_ZMZNZMZNf64, &type_ZMZNZMZNf64, &type_ZMZNZMZNf64, &type_ZMZNZMZNf64, NULL};
bool to_params_in_unique[] = {false, false, false, false, false, false, false, false, false};
const char *to_params_tuning_params[] = {NULL};
const char *to_params_attrs[] = {NULL};
int call_to_params(struct futhark_context *ctx, void *out, void **ins)
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
    
    return futhark_entry_to_params(ctx, out, in0, in1, in2, in3, in4, in5, in6, in7, in8);
}
const struct type *train_in_types[] = {&type_params, &type_params, &type_params, &type_ZMZNZMZNZMZNf64, &type_ZMZNi64, &type_ZMZNZMZNi64, NULL};
bool train_in_unique[] = {false, false, false, false, false, false};
const char *train_tuning_params[] = {NULL};
const char *train_attrs[] = {NULL};
int call_train(struct futhark_context *ctx, void *out, void **ins)
{
    struct futhark_opaque_params * in0 = *(struct futhark_opaque_params * *) ins[0];
    struct futhark_opaque_params * in1 = *(struct futhark_opaque_params * *) ins[1];
    struct futhark_opaque_params * in2 = *(struct futhark_opaque_params * *) ins[2];
    struct futhark_f64_3d * in3 = *(struct futhark_f64_3d * *) ins[3];
    struct futhark_i64_1d * in4 = *(struct futhark_i64_1d * *) ins[4];
    struct futhark_i64_2d * in5 = *(struct futhark_i64_2d * *) ins[5];
    
    return futhark_entry_train(ctx, out, in0, in1, in2, in3, in4, in5);
}
const struct type *zzero_params_in_types[] = {NULL};
bool zzero_params_in_unique[] = {};
const char *zzero_params_tuning_params[] = {NULL};
const char *zzero_params_attrs[] = {NULL};
int call_zzero_params(struct futhark_context *ctx, void *out, void **ins)
{
    (void) ins;
    return futhark_entry_zero_params(ctx, out);
}
const struct type *types[] = {&type_i8, &type_i16, &type_i32, &type_i64, &type_u8, &type_u16, &type_u32, &type_u64, &type_f16, &type_f32, &type_f64, &type_bool, &type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRZR, &type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR, &type_ZMZNZMZNZMZNf64, &type_ZMZNZMZNf64, &type_ZMZNZMZNi64, &type_ZMZNi64, &type_params, NULL};
struct entry_point entry_points[] = {{.name ="forward_seq", .f =call_forward_seq, .tuning_params =forward_seq_tuning_params, .in_types =forward_seq_in_types, .out_type =&type_ZMZNZMZNf64, .in_unique =forward_seq_in_unique, .out_unique =false, .attrs =forward_seq_attrs}, {.name ="to_params", .f =call_to_params, .tuning_params =to_params_tuning_params, .in_types =to_params_in_types, .out_type =&type_params, .in_unique =to_params_in_unique, .out_unique =false, .attrs =to_params_attrs}, {.name ="train", .f =call_train, .tuning_params =train_tuning_params, .in_types =train_in_types, .out_type =&type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRZR, .in_unique =train_in_unique, .out_unique =false, .attrs =train_attrs}, {.name ="zero_params", .f =call_zzero_params, .tuning_params =zzero_params_tuning_params, .in_types =zzero_params_in_types, .out_type =&type_params, .in_unique =zzero_params_in_unique, .out_unique =false, .attrs =zzero_params_attrs}, {.name =NULL}};
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
    struct memblock mem_86828;
    struct memblock mem_86829;
    struct memblock mem_86830;
    struct memblock mem_86831;
    struct memblock mem_86832;
    struct memblock mem_86833;
    struct memblock mem_86834;
    struct memblock mem_86835;
    struct memblock mem_86836;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_10203(struct futhark_context *ctx, struct memblock *mem_out_p_88923, struct memblock *mem_out_p_88924, struct memblock *mem_out_p_88925, struct memblock w_mem_86837, struct memblock mw_mem_86838, struct memblock vw_mem_86839, struct memblock dw_mem_86840, int64_t n_62521, int64_t m_62522, int64_t step_62527, double lt_r_62528);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_10204(struct futhark_context *ctx, struct memblock *mem_out_p_88928, struct memblock *mem_out_p_88929, struct memblock *mem_out_p_88930, struct memblock w_mem_86837, struct memblock mw_mem_86838, struct memblock vw_mem_86839, struct memblock dw_mem_86840, int64_t n_63554, int64_t m_63555, int64_t step_63560, double lt_r_63561);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_88933, struct memblock wdown_mem_86837, struct memblock wkey_mem_86838, struct memblock wout_mem_86839, struct memblock wpe_mem_86840, struct memblock wqry_mem_86841, struct memblock wte_mem_86842, struct memblock wup_mem_86843, struct memblock wval_mem_86844, struct memblock wvoc_mem_86845, struct memblock tokens_mem_86846, struct memblock mask_mem_86847);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_88987, struct memblock *mem_out_p_88988, struct memblock *mem_out_p_88989, struct memblock *mem_out_p_88990, struct memblock *mem_out_p_88991, struct memblock *mem_out_p_88992, struct memblock *mem_out_p_88993, struct memblock *mem_out_p_88994, struct memblock *mem_out_p_88995, struct memblock wte_mem_86837, struct memblock wpe_mem_86838, struct memblock wqry_mem_86839, struct memblock wkey_mem_86840, struct memblock wval_mem_86841, struct memblock wout_mem_86842, struct memblock wup_mem_86843, struct memblock wdown_mem_86844, struct memblock wvoc_mem_86845);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_88996, struct memblock *mem_out_p_88997, struct memblock *mem_out_p_88998, struct memblock *mem_out_p_88999, struct memblock *mem_out_p_89000, struct memblock *mem_out_p_89001, struct memblock *mem_out_p_89002, struct memblock *mem_out_p_89003, struct memblock *mem_out_p_89004, struct memblock *mem_out_p_89005, struct memblock *mem_out_p_89006, struct memblock *mem_out_p_89007, struct memblock *mem_out_p_89008, struct memblock *mem_out_p_89009, struct memblock *mem_out_p_89010, struct memblock *mem_out_p_89011, struct memblock *mem_out_p_89012, struct memblock *mem_out_p_89013, struct memblock *mem_out_p_89014, struct memblock *mem_out_p_89015, struct memblock *mem_out_p_89016, struct memblock *mem_out_p_89017, struct memblock *mem_out_p_89018, struct memblock *mem_out_p_89019, struct memblock *mem_out_p_89020, struct memblock *mem_out_p_89021, struct memblock *mem_out_p_89022, struct memblock wdown_mem_86837, struct memblock wkey_mem_86838, struct memblock wout_mem_86839, struct memblock wpe_mem_86840, struct memblock wqry_mem_86841, struct memblock wte_mem_86842, struct memblock wup_mem_86843, struct memblock wval_mem_86844, struct memblock wvoc_mem_86845, struct memblock wdown_mem_86846, struct memblock wkey_mem_86847, struct memblock wout_mem_86848, struct memblock wpe_mem_86849, struct memblock wqry_mem_86850, struct memblock wte_mem_86851, struct memblock wup_mem_86852, struct memblock wval_mem_86853, struct memblock wvoc_mem_86854, struct memblock wdown_mem_86855, struct memblock wkey_mem_86856, struct memblock wout_mem_86857, struct memblock wpe_mem_86858, struct memblock wqry_mem_86859, struct memblock wte_mem_86860, struct memblock wup_mem_86861, struct memblock wval_mem_86862, struct memblock wvoc_mem_86863, struct memblock masks_mem_86864, struct memblock dls_mem_86865, struct memblock seqs_mem_86866);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_89188, struct memblock *mem_out_p_89189, struct memblock *mem_out_p_89190, struct memblock *mem_out_p_89191, struct memblock *mem_out_p_89192, struct memblock *mem_out_p_89193, struct memblock *mem_out_p_89194, struct memblock *mem_out_p_89195, struct memblock *mem_out_p_89196);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_86828 (ctx->constants->mem_86828)
    #define mem_86829 (ctx->constants->mem_86829)
    #define mem_86830 (ctx->constants->mem_86830)
    #define mem_86831 (ctx->constants->mem_86831)
    #define mem_86832 (ctx->constants->mem_86832)
    #define mem_86833 (ctx->constants->mem_86833)
    #define mem_86834 (ctx->constants->mem_86834)
    #define mem_86835 (ctx->constants->mem_86835)
    #define mem_86836 (ctx->constants->mem_86836)
    mem_86828.references = NULL;
    mem_86829.references = NULL;
    mem_86830.references = NULL;
    mem_86831.references = NULL;
    mem_86832.references = NULL;
    mem_86833.references = NULL;
    mem_86834.references = NULL;
    mem_86835.references = NULL;
    mem_86836.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_86828, (int64_t) 3456, "mem_86828")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_88905 = 0; nest_i_88905 < (int64_t) 27; nest_i_88905++) {
        for (int64_t nest_i_88906 = 0; nest_i_88906 < (int64_t) 16; nest_i_88906++) {
            ((double *) mem_86828.mem)[nest_i_88905 * (int64_t) 16 + nest_i_88906] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_86829, (int64_t) 2048, "mem_86829")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_88907 = 0; nest_i_88907 < (int64_t) 16; nest_i_88907++) {
        for (int64_t nest_i_88908 = 0; nest_i_88908 < (int64_t) 16; nest_i_88908++) {
            ((double *) mem_86829.mem)[nest_i_88907 * (int64_t) 16 + nest_i_88908] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_86830, (int64_t) 2048, "mem_86830")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_88909 = 0; nest_i_88909 < (int64_t) 16; nest_i_88909++) {
        for (int64_t nest_i_88910 = 0; nest_i_88910 < (int64_t) 16; nest_i_88910++) {
            ((double *) mem_86830.mem)[nest_i_88909 * (int64_t) 16 + nest_i_88910] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_86831, (int64_t) 2048, "mem_86831")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_88911 = 0; nest_i_88911 < (int64_t) 16; nest_i_88911++) {
        for (int64_t nest_i_88912 = 0; nest_i_88912 < (int64_t) 16; nest_i_88912++) {
            ((double *) mem_86831.mem)[nest_i_88911 * (int64_t) 16 + nest_i_88912] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_86832, (int64_t) 2048, "mem_86832")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_88913 = 0; nest_i_88913 < (int64_t) 16; nest_i_88913++) {
        for (int64_t nest_i_88914 = 0; nest_i_88914 < (int64_t) 16; nest_i_88914++) {
            ((double *) mem_86832.mem)[nest_i_88913 * (int64_t) 16 + nest_i_88914] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_86833, (int64_t) 2048, "mem_86833")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_88915 = 0; nest_i_88915 < (int64_t) 16; nest_i_88915++) {
        for (int64_t nest_i_88916 = 0; nest_i_88916 < (int64_t) 16; nest_i_88916++) {
            ((double *) mem_86833.mem)[nest_i_88915 * (int64_t) 16 + nest_i_88916] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_86834, (int64_t) 8192, "mem_86834")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_88917 = 0; nest_i_88917 < (int64_t) 64; nest_i_88917++) {
        for (int64_t nest_i_88918 = 0; nest_i_88918 < (int64_t) 16; nest_i_88918++) {
            ((double *) mem_86834.mem)[nest_i_88917 * (int64_t) 16 + nest_i_88918] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_86835, (int64_t) 8192, "mem_86835")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_88919 = 0; nest_i_88919 < (int64_t) 16; nest_i_88919++) {
        for (int64_t nest_i_88920 = 0; nest_i_88920 < (int64_t) 64; nest_i_88920++) {
            ((double *) mem_86835.mem)[nest_i_88919 * (int64_t) 64 + nest_i_88920] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_86836, (int64_t) 3456, "mem_86836")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_88921 = 0; nest_i_88921 < (int64_t) 27; nest_i_88921++) {
        for (int64_t nest_i_88922 = 0; nest_i_88922 < (int64_t) 16; nest_i_88922++) {
            ((double *) mem_86836.mem)[nest_i_88921 * (int64_t) 16 + nest_i_88922] = 0.0;
        }
    }
    #undef mem_86828
    #undef mem_86829
    #undef mem_86830
    #undef mem_86831
    #undef mem_86832
    #undef mem_86833
    #undef mem_86834
    #undef mem_86835
    #undef mem_86836
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_86828, "ctx->constants->mem_86828") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_86829, "ctx->constants->mem_86829") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_86830, "ctx->constants->mem_86830") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_86831, "ctx->constants->mem_86831") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_86832, "ctx->constants->mem_86832") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_86833, "ctx->constants->mem_86833") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_86834, "ctx->constants->mem_86834") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_86835, "ctx->constants->mem_86835") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_86836, "ctx->constants->mem_86836") != 0)
        return 1;
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
struct futhark_i64_2d {
    struct memblock mem;
    int64_t shape[2];
};
struct futhark_i64_2d *futhark_new_i64_2d(struct futhark_context *ctx, const int64_t *data, int64_t dim0, int64_t dim1)
{
    int err = 0;
    struct futhark_i64_2d *bad = NULL;
    struct futhark_i64_2d *arr = (struct futhark_i64_2d *) malloc(sizeof(struct futhark_i64_2d));
    
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
struct futhark_i64_2d *futhark_new_raw_i64_2d(struct futhark_context *ctx, unsigned char *data, int64_t dim0, int64_t dim1)
{
    int err = 0;
    struct futhark_i64_2d *bad = NULL;
    struct futhark_i64_2d *arr = (struct futhark_i64_2d *) malloc(sizeof(struct futhark_i64_2d));
    
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
int futhark_free_i64_2d(struct futhark_context *ctx, struct futhark_i64_2d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_i64_2d(struct futhark_context *ctx, struct futhark_i64_2d *arr, int64_t *data)
{
    int err = 0;
    
    lock_lock(&ctx->lock);
    if ((size_t) (arr->shape[0] * arr->shape[1]) * 8 > 0)
        memmove((unsigned char *) data + 0, arr->mem.mem + 0, (size_t) (arr->shape[0] * arr->shape[1]) * 8);
    lock_unlock(&ctx->lock);
    return err;
}
int futhark_index_i64_2d(struct futhark_context *ctx, int64_t *out, struct futhark_i64_2d *arr, int64_t i0, int64_t i1)
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
unsigned char *futhark_values_raw_i64_2d(struct futhark_context *ctx, struct futhark_i64_2d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_i64_2d(struct futhark_context *ctx, struct futhark_i64_2d *arr)
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
struct futhark_f64_3d {
    struct memblock mem;
    int64_t shape[3];
};
struct futhark_f64_3d *futhark_new_f64_3d(struct futhark_context *ctx, const double *data, int64_t dim0, int64_t dim1, int64_t dim2)
{
    int err = 0;
    struct futhark_f64_3d *bad = NULL;
    struct futhark_f64_3d *arr = (struct futhark_f64_3d *) malloc(sizeof(struct futhark_f64_3d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    arr->shape[2] = dim2;
    if (memblock_alloc(ctx, &arr->mem, arr->shape[0] * arr->shape[1] * arr->shape[2] * 8, "arr->mem"))
        err = 1;
    if ((size_t) (dim0 * dim1 * dim2) * 8 > 0)
        memmove(arr->mem.mem + 0, (const unsigned char *) data + 0, (size_t) (dim0 * dim1 * dim2) * 8);
    lock_unlock(&ctx->lock);
    if (err != 0) {
        free(arr);
        return bad;
    }
    return arr;
}
struct futhark_f64_3d *futhark_new_raw_f64_3d(struct futhark_context *ctx, unsigned char *data, int64_t dim0, int64_t dim1, int64_t dim2)
{
    int err = 0;
    struct futhark_f64_3d *bad = NULL;
    struct futhark_f64_3d *arr = (struct futhark_f64_3d *) malloc(sizeof(struct futhark_f64_3d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    arr->mem.mem = data;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    arr->shape[2] = dim2;
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f64_3d(struct futhark_context *ctx, struct futhark_f64_3d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f64_3d(struct futhark_context *ctx, struct futhark_f64_3d *arr, double *data)
{
    int err = 0;
    
    lock_lock(&ctx->lock);
    if ((size_t) (arr->shape[0] * arr->shape[1] * arr->shape[2]) * 8 > 0)
        memmove((unsigned char *) data + 0, arr->mem.mem + 0, (size_t) (arr->shape[0] * arr->shape[1] * arr->shape[2]) * 8);
    lock_unlock(&ctx->lock);
    return err;
}
int futhark_index_f64_3d(struct futhark_context *ctx, double *out, struct futhark_f64_3d *arr, int64_t i0, int64_t i1, int64_t i2)
{
    int err = 0;
    
    if ((i0 >= 0 && i0 < arr->shape[0]) && ((i1 >= 0 && i1 < arr->shape[1]) && (i2 >= 0 && i2 < arr->shape[2]))) {
        lock_lock(&ctx->lock);
        if (8 > 0)
            memmove((unsigned char *) out + 0, arr->mem.mem + 8 * (i0 * (arr->shape[1] * arr->shape[2]) + i1 * arr->shape[2] + i2 * 1), 8);
        lock_unlock(&ctx->lock);
    } else {
        err = 1;
        set_error(ctx, strdup("Index out of bounds."));
    }
    return err;
}
unsigned char *futhark_values_raw_f64_3d(struct futhark_context *ctx, struct futhark_f64_3d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_f64_3d(struct futhark_context *ctx, struct futhark_f64_3d *arr)
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
struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 {
    struct futhark_f64_2d *v0;
    struct futhark_f64_2d *v1;
    struct futhark_f64_2d *v2;
    struct futhark_f64_2d *v3;
    struct futhark_f64_2d *v4;
    struct futhark_f64_2d *v5;
    struct futhark_f64_2d *v6;
    struct futhark_f64_2d *v7;
    struct futhark_f64_2d *v8;
    struct futhark_f64_2d *v9;
    struct futhark_f64_2d *v10;
    struct futhark_f64_2d *v11;
    struct futhark_f64_2d *v12;
    struct futhark_f64_2d *v13;
    struct futhark_f64_2d *v14;
    struct futhark_f64_2d *v15;
    struct futhark_f64_2d *v16;
    struct futhark_f64_2d *v17;
    struct futhark_f64_2d *v18;
    struct futhark_f64_2d *v19;
    struct futhark_f64_2d *v20;
    struct futhark_f64_2d *v21;
    struct futhark_f64_2d *v22;
    struct futhark_f64_2d *v23;
    struct futhark_f64_2d *v24;
    struct futhark_f64_2d *v25;
    struct futhark_f64_2d *v26;
};
int futhark_project_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_0(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj)
{
    (void) ctx;
    
    struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64));
    v->v0 = malloc(sizeof(*v->v0));
    memcpy(v->v0, obj->v0, sizeof(*obj->v0));
    (void) (*v->v0->mem.references)++;
    v->v1 = malloc(sizeof(*v->v1));
    memcpy(v->v1, obj->v1, sizeof(*obj->v1));
    (void) (*v->v1->mem.references)++;
    v->v2 = malloc(sizeof(*v->v2));
    memcpy(v->v2, obj->v2, sizeof(*obj->v2));
    (void) (*v->v2->mem.references)++;
    v->v3 = malloc(sizeof(*v->v3));
    memcpy(v->v3, obj->v3, sizeof(*obj->v3));
    (void) (*v->v3->mem.references)++;
    v->v4 = malloc(sizeof(*v->v4));
    memcpy(v->v4, obj->v4, sizeof(*obj->v4));
    (void) (*v->v4->mem.references)++;
    v->v5 = malloc(sizeof(*v->v5));
    memcpy(v->v5, obj->v5, sizeof(*obj->v5));
    (void) (*v->v5->mem.references)++;
    v->v6 = malloc(sizeof(*v->v6));
    memcpy(v->v6, obj->v6, sizeof(*obj->v6));
    (void) (*v->v6->mem.references)++;
    v->v7 = malloc(sizeof(*v->v7));
    memcpy(v->v7, obj->v7, sizeof(*obj->v7));
    (void) (*v->v7->mem.references)++;
    v->v8 = malloc(sizeof(*v->v8));
    memcpy(v->v8, obj->v8, sizeof(*obj->v8));
    (void) (*v->v8->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_1(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj)
{
    (void) ctx;
    
    struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64));
    v->v0 = malloc(sizeof(*v->v0));
    memcpy(v->v0, obj->v9, sizeof(*obj->v9));
    (void) (*v->v0->mem.references)++;
    v->v1 = malloc(sizeof(*v->v1));
    memcpy(v->v1, obj->v10, sizeof(*obj->v10));
    (void) (*v->v1->mem.references)++;
    v->v2 = malloc(sizeof(*v->v2));
    memcpy(v->v2, obj->v11, sizeof(*obj->v11));
    (void) (*v->v2->mem.references)++;
    v->v3 = malloc(sizeof(*v->v3));
    memcpy(v->v3, obj->v12, sizeof(*obj->v12));
    (void) (*v->v3->mem.references)++;
    v->v4 = malloc(sizeof(*v->v4));
    memcpy(v->v4, obj->v13, sizeof(*obj->v13));
    (void) (*v->v4->mem.references)++;
    v->v5 = malloc(sizeof(*v->v5));
    memcpy(v->v5, obj->v14, sizeof(*obj->v14));
    (void) (*v->v5->mem.references)++;
    v->v6 = malloc(sizeof(*v->v6));
    memcpy(v->v6, obj->v15, sizeof(*obj->v15));
    (void) (*v->v6->mem.references)++;
    v->v7 = malloc(sizeof(*v->v7));
    memcpy(v->v7, obj->v16, sizeof(*obj->v16));
    (void) (*v->v7->mem.references)++;
    v->v8 = malloc(sizeof(*v->v8));
    memcpy(v->v8, obj->v17, sizeof(*obj->v17));
    (void) (*v->v8->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_2(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj)
{
    (void) ctx;
    
    struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64));
    v->v0 = malloc(sizeof(*v->v0));
    memcpy(v->v0, obj->v18, sizeof(*obj->v18));
    (void) (*v->v0->mem.references)++;
    v->v1 = malloc(sizeof(*v->v1));
    memcpy(v->v1, obj->v19, sizeof(*obj->v19));
    (void) (*v->v1->mem.references)++;
    v->v2 = malloc(sizeof(*v->v2));
    memcpy(v->v2, obj->v20, sizeof(*obj->v20));
    (void) (*v->v2->mem.references)++;
    v->v3 = malloc(sizeof(*v->v3));
    memcpy(v->v3, obj->v21, sizeof(*obj->v21));
    (void) (*v->v3->mem.references)++;
    v->v4 = malloc(sizeof(*v->v4));
    memcpy(v->v4, obj->v22, sizeof(*obj->v22));
    (void) (*v->v4->mem.references)++;
    v->v5 = malloc(sizeof(*v->v5));
    memcpy(v->v5, obj->v23, sizeof(*obj->v23));
    (void) (*v->v5->mem.references)++;
    v->v6 = malloc(sizeof(*v->v6));
    memcpy(v->v6, obj->v24, sizeof(*obj->v24));
    (void) (*v->v6->mem.references)++;
    v->v7 = malloc(sizeof(*v->v7));
    memcpy(v->v7, obj->v25, sizeof(*obj->v25));
    (void) (*v->v7->mem.references)++;
    v->v8 = malloc(sizeof(*v->v8));
    memcpy(v->v8, obj->v26, sizeof(*obj->v26));
    (void) (*v->v8->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_new_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64(struct futhark_context *ctx, struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *f_0, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *f_1, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *f_2)
{
    struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *v = malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64));
    
    lock_lock(&ctx->lock);
    {
        {
            v->v0 = malloc(sizeof(*f_0->v0));
            memcpy(v->v0, f_0->v0, sizeof(*f_0->v0));
            (void) (*v->v0->mem.references)++;
        }
        {
            v->v1 = malloc(sizeof(*f_0->v1));
            memcpy(v->v1, f_0->v1, sizeof(*f_0->v1));
            (void) (*v->v1->mem.references)++;
        }
        {
            v->v2 = malloc(sizeof(*f_0->v2));
            memcpy(v->v2, f_0->v2, sizeof(*f_0->v2));
            (void) (*v->v2->mem.references)++;
        }
        {
            v->v3 = malloc(sizeof(*f_0->v3));
            memcpy(v->v3, f_0->v3, sizeof(*f_0->v3));
            (void) (*v->v3->mem.references)++;
        }
        {
            v->v4 = malloc(sizeof(*f_0->v4));
            memcpy(v->v4, f_0->v4, sizeof(*f_0->v4));
            (void) (*v->v4->mem.references)++;
        }
        {
            v->v5 = malloc(sizeof(*f_0->v5));
            memcpy(v->v5, f_0->v5, sizeof(*f_0->v5));
            (void) (*v->v5->mem.references)++;
        }
        {
            v->v6 = malloc(sizeof(*f_0->v6));
            memcpy(v->v6, f_0->v6, sizeof(*f_0->v6));
            (void) (*v->v6->mem.references)++;
        }
        {
            v->v7 = malloc(sizeof(*f_0->v7));
            memcpy(v->v7, f_0->v7, sizeof(*f_0->v7));
            (void) (*v->v7->mem.references)++;
        }
        {
            v->v8 = malloc(sizeof(*f_0->v8));
            memcpy(v->v8, f_0->v8, sizeof(*f_0->v8));
            (void) (*v->v8->mem.references)++;
        }
    }
    {
        {
            v->v9 = malloc(sizeof(*f_1->v0));
            memcpy(v->v9, f_1->v0, sizeof(*f_1->v0));
            (void) (*v->v9->mem.references)++;
        }
        {
            v->v10 = malloc(sizeof(*f_1->v1));
            memcpy(v->v10, f_1->v1, sizeof(*f_1->v1));
            (void) (*v->v10->mem.references)++;
        }
        {
            v->v11 = malloc(sizeof(*f_1->v2));
            memcpy(v->v11, f_1->v2, sizeof(*f_1->v2));
            (void) (*v->v11->mem.references)++;
        }
        {
            v->v12 = malloc(sizeof(*f_1->v3));
            memcpy(v->v12, f_1->v3, sizeof(*f_1->v3));
            (void) (*v->v12->mem.references)++;
        }
        {
            v->v13 = malloc(sizeof(*f_1->v4));
            memcpy(v->v13, f_1->v4, sizeof(*f_1->v4));
            (void) (*v->v13->mem.references)++;
        }
        {
            v->v14 = malloc(sizeof(*f_1->v5));
            memcpy(v->v14, f_1->v5, sizeof(*f_1->v5));
            (void) (*v->v14->mem.references)++;
        }
        {
            v->v15 = malloc(sizeof(*f_1->v6));
            memcpy(v->v15, f_1->v6, sizeof(*f_1->v6));
            (void) (*v->v15->mem.references)++;
        }
        {
            v->v16 = malloc(sizeof(*f_1->v7));
            memcpy(v->v16, f_1->v7, sizeof(*f_1->v7));
            (void) (*v->v16->mem.references)++;
        }
        {
            v->v17 = malloc(sizeof(*f_1->v8));
            memcpy(v->v17, f_1->v8, sizeof(*f_1->v8));
            (void) (*v->v17->mem.references)++;
        }
    }
    {
        {
            v->v18 = malloc(sizeof(*f_2->v0));
            memcpy(v->v18, f_2->v0, sizeof(*f_2->v0));
            (void) (*v->v18->mem.references)++;
        }
        {
            v->v19 = malloc(sizeof(*f_2->v1));
            memcpy(v->v19, f_2->v1, sizeof(*f_2->v1));
            (void) (*v->v19->mem.references)++;
        }
        {
            v->v20 = malloc(sizeof(*f_2->v2));
            memcpy(v->v20, f_2->v2, sizeof(*f_2->v2));
            (void) (*v->v20->mem.references)++;
        }
        {
            v->v21 = malloc(sizeof(*f_2->v3));
            memcpy(v->v21, f_2->v3, sizeof(*f_2->v3));
            (void) (*v->v21->mem.references)++;
        }
        {
            v->v22 = malloc(sizeof(*f_2->v4));
            memcpy(v->v22, f_2->v4, sizeof(*f_2->v4));
            (void) (*v->v22->mem.references)++;
        }
        {
            v->v23 = malloc(sizeof(*f_2->v5));
            memcpy(v->v23, f_2->v5, sizeof(*f_2->v5));
            (void) (*v->v23->mem.references)++;
        }
        {
            v->v24 = malloc(sizeof(*f_2->v6));
            memcpy(v->v24, f_2->v6, sizeof(*f_2->v6));
            (void) (*v->v24->mem.references)++;
        }
        {
            v->v25 = malloc(sizeof(*f_2->v7));
            memcpy(v->v25, f_2->v7, sizeof(*f_2->v7));
            (void) (*v->v25->mem.references)++;
        }
        {
            v->v26 = malloc(sizeof(*f_2->v8));
            memcpy(v->v26, f_2->v8, sizeof(*f_2->v8));
            (void) (*v->v26->mem.references)++;
        }
    }
    lock_unlock(&ctx->lock);
    *out = v;
    return FUTHARK_SUCCESS;
}
int futhark_free_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64(struct futhark_context *ctx, struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj)
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
    if (obj->v9 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v9)) != 0)
        ret = tmp;
    if (obj->v10 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v10)) != 0)
        ret = tmp;
    if (obj->v11 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v11)) != 0)
        ret = tmp;
    if (obj->v12 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v12)) != 0)
        ret = tmp;
    if (obj->v13 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v13)) != 0)
        ret = tmp;
    if (obj->v14 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v14)) != 0)
        ret = tmp;
    if (obj->v15 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v15)) != 0)
        ret = tmp;
    if (obj->v16 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v16)) != 0)
        ret = tmp;
    if (obj->v17 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v17)) != 0)
        ret = tmp;
    if (obj->v18 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v18)) != 0)
        ret = tmp;
    if (obj->v19 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v19)) != 0)
        ret = tmp;
    if (obj->v20 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v20)) != 0)
        ret = tmp;
    if (obj->v21 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v21)) != 0)
        ret = tmp;
    if (obj->v22 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v22)) != 0)
        ret = tmp;
    if (obj->v23 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v23)) != 0)
        ret = tmp;
    if (obj->v24 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v24)) != 0)
        ret = tmp;
    if (obj->v25 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v25)) != 0)
        ret = tmp;
    if (obj->v26 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v26)) != 0)
        ret = tmp;
    free(obj);
    return ret;
}
int futhark_store_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64(struct futhark_context *ctx, const struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj, void **p, size_t *n)
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
    int64_t size_9 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v9)[0] * futhark_shape_f64_2d(ctx, obj->v9)[1] * sizeof(double);
    int64_t size_10 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v10)[0] * futhark_shape_f64_2d(ctx, obj->v10)[1] * sizeof(double);
    int64_t size_11 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v11)[0] * futhark_shape_f64_2d(ctx, obj->v11)[1] * sizeof(double);
    int64_t size_12 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v12)[0] * futhark_shape_f64_2d(ctx, obj->v12)[1] * sizeof(double);
    int64_t size_13 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v13)[0] * futhark_shape_f64_2d(ctx, obj->v13)[1] * sizeof(double);
    int64_t size_14 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v14)[0] * futhark_shape_f64_2d(ctx, obj->v14)[1] * sizeof(double);
    int64_t size_15 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v15)[0] * futhark_shape_f64_2d(ctx, obj->v15)[1] * sizeof(double);
    int64_t size_16 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v16)[0] * futhark_shape_f64_2d(ctx, obj->v16)[1] * sizeof(double);
    int64_t size_17 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v17)[0] * futhark_shape_f64_2d(ctx, obj->v17)[1] * sizeof(double);
    int64_t size_18 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v18)[0] * futhark_shape_f64_2d(ctx, obj->v18)[1] * sizeof(double);
    int64_t size_19 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v19)[0] * futhark_shape_f64_2d(ctx, obj->v19)[1] * sizeof(double);
    int64_t size_20 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v20)[0] * futhark_shape_f64_2d(ctx, obj->v20)[1] * sizeof(double);
    int64_t size_21 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v21)[0] * futhark_shape_f64_2d(ctx, obj->v21)[1] * sizeof(double);
    int64_t size_22 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v22)[0] * futhark_shape_f64_2d(ctx, obj->v22)[1] * sizeof(double);
    int64_t size_23 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v23)[0] * futhark_shape_f64_2d(ctx, obj->v23)[1] * sizeof(double);
    int64_t size_24 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v24)[0] * futhark_shape_f64_2d(ctx, obj->v24)[1] * sizeof(double);
    int64_t size_25 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v25)[0] * futhark_shape_f64_2d(ctx, obj->v25)[1] * sizeof(double);
    int64_t size_26 = 7 + 2 * sizeof(int64_t) + futhark_shape_f64_2d(ctx, obj->v26)[0] * futhark_shape_f64_2d(ctx, obj->v26)[1] * sizeof(double);
    
    *n = size_0 + size_1 + size_2 + size_3 + size_4 + size_5 + size_6 + size_7 + size_8 + size_9 + size_10 + size_11 + size_12 + size_13 + size_14 + size_15 + size_16 + size_17 + size_18 + size_19 + size_20 + size_21 + size_22 + size_23 + size_24 + size_25 + size_26;
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
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v9), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v9, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v9)[0] * futhark_shape_f64_2d(ctx, obj->v9)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v10), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v10, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v10)[0] * futhark_shape_f64_2d(ctx, obj->v10)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v11), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v11, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v11)[0] * futhark_shape_f64_2d(ctx, obj->v11)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v12), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v12, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v12)[0] * futhark_shape_f64_2d(ctx, obj->v12)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v13), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v13, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v13)[0] * futhark_shape_f64_2d(ctx, obj->v13)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v14), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v14, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v14)[0] * futhark_shape_f64_2d(ctx, obj->v14)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v15), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v15, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v15)[0] * futhark_shape_f64_2d(ctx, obj->v15)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v16), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v16, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v16)[0] * futhark_shape_f64_2d(ctx, obj->v16)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v17), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v17, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v17)[0] * futhark_shape_f64_2d(ctx, obj->v17)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v18), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v18, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v18)[0] * futhark_shape_f64_2d(ctx, obj->v18)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v19), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v19, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v19)[0] * futhark_shape_f64_2d(ctx, obj->v19)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v20), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v20, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v20)[0] * futhark_shape_f64_2d(ctx, obj->v20)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v21), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v21, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v21)[0] * futhark_shape_f64_2d(ctx, obj->v21)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v22), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v22, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v22)[0] * futhark_shape_f64_2d(ctx, obj->v22)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v23), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v23, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v23)[0] * futhark_shape_f64_2d(ctx, obj->v23)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v24), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v24, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v24)[0] * futhark_shape_f64_2d(ctx, obj->v24)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v25), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v25, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v25)[0] * futhark_shape_f64_2d(ctx, obj->v25)[1] * sizeof(double);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_2d(ctx, obj->v26), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f64_2d(ctx, obj->v26, (void *) out);
        out += futhark_shape_f64_2d(ctx, obj->v26)[0] * futhark_shape_f64_2d(ctx, obj->v26)[1] * sizeof(double);
    }
    return ret;
}
struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *futhark_restore_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64(struct futhark_context *ctx, const void *p)
{
    (void) ctx;
    
    int err = 0;
    const unsigned char *src = p;
    struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *obj = malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64));
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
    
    int64_t shape_9[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_9, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_9 = src;
    
    obj->v9 = NULL;
    src += shape_9[0] * shape_9[1] * sizeof(double);
    
    int64_t shape_10[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_10, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_10 = src;
    
    obj->v10 = NULL;
    src += shape_10[0] * shape_10[1] * sizeof(double);
    
    int64_t shape_11[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_11, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_11 = src;
    
    obj->v11 = NULL;
    src += shape_11[0] * shape_11[1] * sizeof(double);
    
    int64_t shape_12[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_12, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_12 = src;
    
    obj->v12 = NULL;
    src += shape_12[0] * shape_12[1] * sizeof(double);
    
    int64_t shape_13[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_13, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_13 = src;
    
    obj->v13 = NULL;
    src += shape_13[0] * shape_13[1] * sizeof(double);
    
    int64_t shape_14[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_14, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_14 = src;
    
    obj->v14 = NULL;
    src += shape_14[0] * shape_14[1] * sizeof(double);
    
    int64_t shape_15[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_15, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_15 = src;
    
    obj->v15 = NULL;
    src += shape_15[0] * shape_15[1] * sizeof(double);
    
    int64_t shape_16[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_16, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_16 = src;
    
    obj->v16 = NULL;
    src += shape_16[0] * shape_16[1] * sizeof(double);
    
    int64_t shape_17[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_17, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_17 = src;
    
    obj->v17 = NULL;
    src += shape_17[0] * shape_17[1] * sizeof(double);
    
    int64_t shape_18[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_18, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_18 = src;
    
    obj->v18 = NULL;
    src += shape_18[0] * shape_18[1] * sizeof(double);
    
    int64_t shape_19[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_19, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_19 = src;
    
    obj->v19 = NULL;
    src += shape_19[0] * shape_19[1] * sizeof(double);
    
    int64_t shape_20[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_20, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_20 = src;
    
    obj->v20 = NULL;
    src += shape_20[0] * shape_20[1] * sizeof(double);
    
    int64_t shape_21[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_21, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_21 = src;
    
    obj->v21 = NULL;
    src += shape_21[0] * shape_21[1] * sizeof(double);
    
    int64_t shape_22[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_22, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_22 = src;
    
    obj->v22 = NULL;
    src += shape_22[0] * shape_22[1] * sizeof(double);
    
    int64_t shape_23[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_23, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_23 = src;
    
    obj->v23 = NULL;
    src += shape_23[0] * shape_23[1] * sizeof(double);
    
    int64_t shape_24[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_24, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_24 = src;
    
    obj->v24 = NULL;
    src += shape_24[0] * shape_24[1] * sizeof(double);
    
    int64_t shape_25[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_25, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_25 = src;
    
    obj->v25 = NULL;
    src += shape_25[0] * shape_25[1] * sizeof(double);
    
    int64_t shape_26[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_26, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_26 = src;
    
    obj->v26 = NULL;
    src += shape_26[0] * shape_26[1] * sizeof(double);
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
        obj->v9 = futhark_new_f64_2d(ctx, data_9, shape_9[0], shape_9[1]);
        if (obj->v9 == NULL)
            err = 1;
        obj->v10 = futhark_new_f64_2d(ctx, data_10, shape_10[0], shape_10[1]);
        if (obj->v10 == NULL)
            err = 1;
        obj->v11 = futhark_new_f64_2d(ctx, data_11, shape_11[0], shape_11[1]);
        if (obj->v11 == NULL)
            err = 1;
        obj->v12 = futhark_new_f64_2d(ctx, data_12, shape_12[0], shape_12[1]);
        if (obj->v12 == NULL)
            err = 1;
        obj->v13 = futhark_new_f64_2d(ctx, data_13, shape_13[0], shape_13[1]);
        if (obj->v13 == NULL)
            err = 1;
        obj->v14 = futhark_new_f64_2d(ctx, data_14, shape_14[0], shape_14[1]);
        if (obj->v14 == NULL)
            err = 1;
        obj->v15 = futhark_new_f64_2d(ctx, data_15, shape_15[0], shape_15[1]);
        if (obj->v15 == NULL)
            err = 1;
        obj->v16 = futhark_new_f64_2d(ctx, data_16, shape_16[0], shape_16[1]);
        if (obj->v16 == NULL)
            err = 1;
        obj->v17 = futhark_new_f64_2d(ctx, data_17, shape_17[0], shape_17[1]);
        if (obj->v17 == NULL)
            err = 1;
        obj->v18 = futhark_new_f64_2d(ctx, data_18, shape_18[0], shape_18[1]);
        if (obj->v18 == NULL)
            err = 1;
        obj->v19 = futhark_new_f64_2d(ctx, data_19, shape_19[0], shape_19[1]);
        if (obj->v19 == NULL)
            err = 1;
        obj->v20 = futhark_new_f64_2d(ctx, data_20, shape_20[0], shape_20[1]);
        if (obj->v20 == NULL)
            err = 1;
        obj->v21 = futhark_new_f64_2d(ctx, data_21, shape_21[0], shape_21[1]);
        if (obj->v21 == NULL)
            err = 1;
        obj->v22 = futhark_new_f64_2d(ctx, data_22, shape_22[0], shape_22[1]);
        if (obj->v22 == NULL)
            err = 1;
        obj->v23 = futhark_new_f64_2d(ctx, data_23, shape_23[0], shape_23[1]);
        if (obj->v23 == NULL)
            err = 1;
        obj->v24 = futhark_new_f64_2d(ctx, data_24, shape_24[0], shape_24[1]);
        if (obj->v24 == NULL)
            err = 1;
        obj->v25 = futhark_new_f64_2d(ctx, data_25, shape_25[0], shape_25[1]);
        if (obj->v25 == NULL)
            err = 1;
        obj->v26 = futhark_new_f64_2d(ctx, data_26, shape_26[0], shape_26[1]);
        if (obj->v26 == NULL)
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
        if (obj->v9 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v9)) != 0)
            ret = tmp;
        if (obj->v10 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v10)) != 0)
            ret = tmp;
        if (obj->v11 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v11)) != 0)
            ret = tmp;
        if (obj->v12 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v12)) != 0)
            ret = tmp;
        if (obj->v13 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v13)) != 0)
            ret = tmp;
        if (obj->v14 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v14)) != 0)
            ret = tmp;
        if (obj->v15 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v15)) != 0)
            ret = tmp;
        if (obj->v16 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v16)) != 0)
            ret = tmp;
        if (obj->v17 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v17)) != 0)
            ret = tmp;
        if (obj->v18 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v18)) != 0)
            ret = tmp;
        if (obj->v19 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v19)) != 0)
            ret = tmp;
        if (obj->v20 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v20)) != 0)
            ret = tmp;
        if (obj->v21 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v21)) != 0)
            ret = tmp;
        if (obj->v22 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v22)) != 0)
            ret = tmp;
        if (obj->v23 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v23)) != 0)
            ret = tmp;
        if (obj->v24 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v24)) != 0)
            ret = tmp;
        if (obj->v25 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v25)) != 0)
            ret = tmp;
        if (obj->v26 != NULL && (tmp = futhark_free_f64_2d(ctx, obj->v26)) != 0)
            ret = tmp;
        free(obj);
        obj = NULL;
    }
    return obj;
}

FUTHARK_FUN_ATTR int futrts_adam_opt_w_10203(struct futhark_context *ctx, struct memblock *mem_out_p_88923, struct memblock *mem_out_p_88924, struct memblock *mem_out_p_88925, struct memblock w_mem_86837, struct memblock mw_mem_86838, struct memblock vw_mem_86839, struct memblock dw_mem_86840, int64_t n_62521, int64_t m_62522, int64_t step_62527, double lt_r_62528)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_86881_cached_sizze_88926 = 0;
    unsigned char *mem_86881 = NULL;
    int64_t mem_86884_cached_sizze_88927 = 0;
    unsigned char *mem_86884 = NULL;
    struct memblock mem_86919;
    
    mem_86919.references = NULL;
    
    struct memblock mem_86846;
    
    mem_86846.references = NULL;
    
    struct memblock mem_86843;
    
    mem_86843.references = NULL;
    
    struct memblock mem_out_88610;
    
    mem_out_88610.references = NULL;
    
    struct memblock mem_out_88609;
    
    mem_out_88609.references = NULL;
    
    struct memblock mem_out_88608;
    
    mem_out_88608.references = NULL;
    
    struct memblock mem_86828 = ctx->constants->mem_86828;
    struct memblock mem_86829 = ctx->constants->mem_86829;
    struct memblock mem_86830 = ctx->constants->mem_86830;
    struct memblock mem_86831 = ctx->constants->mem_86831;
    struct memblock mem_86832 = ctx->constants->mem_86832;
    struct memblock mem_86833 = ctx->constants->mem_86833;
    struct memblock mem_86834 = ctx->constants->mem_86834;
    struct memblock mem_86835 = ctx->constants->mem_86835;
    struct memblock mem_86836 = ctx->constants->mem_86836;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_86841 = (int64_t) 8 * n_62521;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_86842 = m_62522 * binop_x_86841;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_86843, bytes_86842, "mem_86843")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_86846, bytes_86842, "mem_86846")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86010 = 0; i_86010 < n_62521; i_86010++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86003 = 0; i_86003 < m_62522; i_86003++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_82291 = ((double *) mw_mem_86838.mem)[i_86010 * m_62522 + i_86003];
            
            // futhark/microgpt.fut:392:10-20
            
            double zp_lhs_82292 = 0.85 * zt_rhs_82291;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_82293 = ((double *) dw_mem_86840.mem)[i_86010 * m_62522 + i_86003];
            
            // futhark/microgpt.fut:392:35-45
            
            double zp_rhs_82294 = 0.15000000000000002 * zt_rhs_82293;
            
            // futhark/microgpt.fut:392:21-45
            
            double lifted_lambda_res_82295 = zp_lhs_82292 + zp_rhs_82294;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_82302 = ((double *) vw_mem_86839.mem)[i_86010 * m_62522 + i_86003];
            
            // futhark/microgpt.fut:394:10-20
            
            double zp_lhs_82303 = 0.99 * zt_rhs_82302;
            
            // futhark/microgpt.fut:394:35-45
            
            double zt_lhs_82305 = 1.0000000000000009e-2 * zt_rhs_82293;
            
            // futhark/microgpt.fut:394:46-56
            
            double zp_rhs_82306 = zt_rhs_82293 * zt_lhs_82305;
            
            // futhark/microgpt.fut:394:21-56
            
            double lifted_lambda_res_82307 = zp_lhs_82303 + zp_rhs_82306;
            
            ((double *) mem_86843.mem)[i_86010 * m_62522 + i_86003] = lifted_lambda_res_82307;
            ((double *) mem_86846.mem)[i_86010 * m_62522 + i_86003] = lifted_lambda_res_82295;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_67594 = sitofp_i64_f64(step_62527);
    
    // futhark/microgpt.fut:396:54-57
    
    double ztzt_rhs_67595 = 1.0 + i64_res_67594;
    
    // futhark/microgpt.fut:396:30-57
    
    double zm_rhs_67596 = fpow64(0.85, ztzt_rhs_67595);
    
    // futhark/microgpt.fut:396:23-57
    
    double zs_rhs_67597 = 1.0 - zm_rhs_67596;
    
    // futhark/microgpt.fut:398:31-58
    
    double zm_rhs_67635 = fpow64(0.99, ztzt_rhs_67595);
    
    // futhark/microgpt.fut:398:23-58
    
    double zs_rhs_67636 = 1.0 - zm_rhs_67635;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_86881_cached_sizze_88926 < bytes_86842) {
        err = lexical_realloc(ctx, &mem_86881, &mem_86881_cached_sizze_88926, bytes_86842);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86884_cached_sizze_88927 < bytes_86842) {
        err = lexical_realloc(ctx, &mem_86884, &mem_86884_cached_sizze_88927, bytes_86842);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86024 = 0; i_86024 < n_62521; i_86024++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86017 = 0; i_86017 < m_62522; i_86017++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_82327 = ((double *) mem_86846.mem)[i_86024 * m_62522 + i_86017];
            
            // futhark/microgpt.fut:396:18-57
            
            double lifted_lambda_res_82328 = zs_lhs_82327 / zs_rhs_67597;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_82335 = ((double *) mem_86843.mem)[i_86024 * m_62522 + i_86017];
            
            // futhark/microgpt.fut:398:18-58
            
            double lifted_lambda_res_82336 = zs_lhs_82335 / zs_rhs_67636;
            
            ((double *) mem_86881)[i_86024 * m_62522 + i_86017] = lifted_lambda_res_82336;
            ((double *) mem_86884)[i_86024 * m_62522 + i_86017] = lifted_lambda_res_82328;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_86919, bytes_86842, "mem_86919")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86033 = 0; i_86033 < n_62521; i_86033++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86029 = 0; i_86029 < m_62522; i_86029++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_66662 = ((double *) w_mem_86837.mem)[i_86033 * m_62522 + i_86029];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_66663 = ((double *) mem_86884)[i_86033 * m_62522 + i_86029];
            
            // futhark/microgpt.fut:400:21-34
            
            double zs_lhs_66664 = lt_r_62528 * zt_rhs_66663;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_66665 = ((double *) mem_86881)[i_86033 * m_62522 + i_86029];
            
            // futhark/microgpt.fut:400:51-57
            
            double zp_lhs_66666 = fpow64(ztzt_lhs_66665, 0.5);
            
            // futhark/microgpt.fut:400:59-71
            
            double zs_rhs_66667 = 1.0e-8 + zp_lhs_66666;
            
            // futhark/microgpt.fut:400:35-71
            
            double zm_rhs_66668 = zs_lhs_66664 / zs_rhs_66667;
            
            // futhark/microgpt.fut:400:13-71
            
            double lifted_lambda_res_66669 = zm_lhs_66662 - zm_rhs_66668;
            
            ((double *) mem_86919.mem)[i_86033 * m_62522 + i_86029] = lifted_lambda_res_66669;
        }
    }
    if (memblock_set(ctx, &mem_out_88608, &mem_86919, "mem_86919") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88609, &mem_86846, "mem_86846") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88610, &mem_86843, "mem_86843") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88923, &mem_out_88608, "mem_out_88608") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88924, &mem_out_88609, "mem_out_88609") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88925, &mem_out_88610, "mem_out_88610") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_86881);
        free(mem_86884);
        if (memblock_unref(ctx, &mem_86919, "mem_86919") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_86846, "mem_86846") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_86843, "mem_86843") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88610, "mem_out_88610") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88609, "mem_out_88609") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88608, "mem_out_88608") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_10204(struct futhark_context *ctx, struct memblock *mem_out_p_88928, struct memblock *mem_out_p_88929, struct memblock *mem_out_p_88930, struct memblock w_mem_86837, struct memblock mw_mem_86838, struct memblock vw_mem_86839, struct memblock dw_mem_86840, int64_t n_63554, int64_t m_63555, int64_t step_63560, double lt_r_63561)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_86881_cached_sizze_88931 = 0;
    unsigned char *mem_86881 = NULL;
    int64_t mem_86884_cached_sizze_88932 = 0;
    unsigned char *mem_86884 = NULL;
    struct memblock mem_86919;
    
    mem_86919.references = NULL;
    
    struct memblock mem_86846;
    
    mem_86846.references = NULL;
    
    struct memblock mem_86843;
    
    mem_86843.references = NULL;
    
    struct memblock mem_out_88610;
    
    mem_out_88610.references = NULL;
    
    struct memblock mem_out_88609;
    
    mem_out_88609.references = NULL;
    
    struct memblock mem_out_88608;
    
    mem_out_88608.references = NULL;
    
    struct memblock mem_86828 = ctx->constants->mem_86828;
    struct memblock mem_86829 = ctx->constants->mem_86829;
    struct memblock mem_86830 = ctx->constants->mem_86830;
    struct memblock mem_86831 = ctx->constants->mem_86831;
    struct memblock mem_86832 = ctx->constants->mem_86832;
    struct memblock mem_86833 = ctx->constants->mem_86833;
    struct memblock mem_86834 = ctx->constants->mem_86834;
    struct memblock mem_86835 = ctx->constants->mem_86835;
    struct memblock mem_86836 = ctx->constants->mem_86836;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_86841 = (int64_t) 8 * n_63554;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_86842 = m_63555 * binop_x_86841;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_86843, bytes_86842, "mem_86843")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_86846, bytes_86842, "mem_86846")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86010 = 0; i_86010 < n_63554; i_86010++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86003 = 0; i_86003 < m_63555; i_86003++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_82291 = ((double *) mw_mem_86838.mem)[i_86010 * m_63555 + i_86003];
            
            // futhark/microgpt.fut:392:10-20
            
            double zp_lhs_82292 = 0.85 * zt_rhs_82291;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_82293 = ((double *) dw_mem_86840.mem)[i_86010 * m_63555 + i_86003];
            
            // futhark/microgpt.fut:392:35-45
            
            double zp_rhs_82294 = 0.15000000000000002 * zt_rhs_82293;
            
            // futhark/microgpt.fut:392:21-45
            
            double lifted_lambda_res_82295 = zp_lhs_82292 + zp_rhs_82294;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_82302 = ((double *) vw_mem_86839.mem)[i_86010 * m_63555 + i_86003];
            
            // futhark/microgpt.fut:394:10-20
            
            double zp_lhs_82303 = 0.99 * zt_rhs_82302;
            
            // futhark/microgpt.fut:394:35-45
            
            double zt_lhs_82305 = 1.0000000000000009e-2 * zt_rhs_82293;
            
            // futhark/microgpt.fut:394:46-56
            
            double zp_rhs_82306 = zt_rhs_82293 * zt_lhs_82305;
            
            // futhark/microgpt.fut:394:21-56
            
            double lifted_lambda_res_82307 = zp_lhs_82303 + zp_rhs_82306;
            
            ((double *) mem_86843.mem)[i_86010 * m_63555 + i_86003] = lifted_lambda_res_82307;
            ((double *) mem_86846.mem)[i_86010 * m_63555 + i_86003] = lifted_lambda_res_82295;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_67594 = sitofp_i64_f64(step_63560);
    
    // futhark/microgpt.fut:396:54-57
    
    double ztzt_rhs_67595 = 1.0 + i64_res_67594;
    
    // futhark/microgpt.fut:396:30-57
    
    double zm_rhs_67596 = fpow64(0.85, ztzt_rhs_67595);
    
    // futhark/microgpt.fut:396:23-57
    
    double zs_rhs_67597 = 1.0 - zm_rhs_67596;
    
    // futhark/microgpt.fut:398:31-58
    
    double zm_rhs_67635 = fpow64(0.99, ztzt_rhs_67595);
    
    // futhark/microgpt.fut:398:23-58
    
    double zs_rhs_67636 = 1.0 - zm_rhs_67635;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_86881_cached_sizze_88931 < bytes_86842) {
        err = lexical_realloc(ctx, &mem_86881, &mem_86881_cached_sizze_88931, bytes_86842);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86884_cached_sizze_88932 < bytes_86842) {
        err = lexical_realloc(ctx, &mem_86884, &mem_86884_cached_sizze_88932, bytes_86842);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86024 = 0; i_86024 < n_63554; i_86024++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86017 = 0; i_86017 < m_63555; i_86017++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_82327 = ((double *) mem_86846.mem)[i_86024 * m_63555 + i_86017];
            
            // futhark/microgpt.fut:396:18-57
            
            double lifted_lambda_res_82328 = zs_lhs_82327 / zs_rhs_67597;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_82335 = ((double *) mem_86843.mem)[i_86024 * m_63555 + i_86017];
            
            // futhark/microgpt.fut:398:18-58
            
            double lifted_lambda_res_82336 = zs_lhs_82335 / zs_rhs_67636;
            
            ((double *) mem_86881)[i_86024 * m_63555 + i_86017] = lifted_lambda_res_82336;
            ((double *) mem_86884)[i_86024 * m_63555 + i_86017] = lifted_lambda_res_82328;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_86919, bytes_86842, "mem_86919")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86033 = 0; i_86033 < n_63554; i_86033++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86029 = 0; i_86029 < m_63555; i_86029++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_66662 = ((double *) w_mem_86837.mem)[i_86033 * m_63555 + i_86029];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_66663 = ((double *) mem_86884)[i_86033 * m_63555 + i_86029];
            
            // futhark/microgpt.fut:400:21-34
            
            double zs_lhs_66664 = lt_r_63561 * zt_rhs_66663;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_66665 = ((double *) mem_86881)[i_86033 * m_63555 + i_86029];
            
            // futhark/microgpt.fut:400:51-57
            
            double zp_lhs_66666 = fpow64(ztzt_lhs_66665, 0.5);
            
            // futhark/microgpt.fut:400:59-71
            
            double zs_rhs_66667 = 1.0e-8 + zp_lhs_66666;
            
            // futhark/microgpt.fut:400:35-71
            
            double zm_rhs_66668 = zs_lhs_66664 / zs_rhs_66667;
            
            // futhark/microgpt.fut:400:13-71
            
            double lifted_lambda_res_66669 = zm_lhs_66662 - zm_rhs_66668;
            
            ((double *) mem_86919.mem)[i_86033 * m_63555 + i_86029] = lifted_lambda_res_66669;
        }
    }
    if (memblock_set(ctx, &mem_out_88608, &mem_86919, "mem_86919") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88609, &mem_86846, "mem_86846") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88610, &mem_86843, "mem_86843") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88928, &mem_out_88608, "mem_out_88608") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88929, &mem_out_88609, "mem_out_88609") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88930, &mem_out_88610, "mem_out_88610") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_86881);
        free(mem_86884);
        if (memblock_unref(ctx, &mem_86919, "mem_86919") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_86846, "mem_86846") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_86843, "mem_86843") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88610, "mem_out_88610") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88609, "mem_out_88609") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88608, "mem_out_88608") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_88933, struct memblock wdown_mem_86837, struct memblock wkey_mem_86838, struct memblock wout_mem_86839, struct memblock wpe_mem_86840, struct memblock wqry_mem_86841, struct memblock wte_mem_86842, struct memblock wup_mem_86843, struct memblock wval_mem_86844, struct memblock wvoc_mem_86845, struct memblock tokens_mem_86846, struct memblock mask_mem_86847)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_86848_cached_sizze_88934 = 0;
    unsigned char *mem_86848 = NULL;
    int64_t mem_86853_cached_sizze_88935 = 0;
    unsigned char *mem_86853 = NULL;
    int64_t mem_86864_cached_sizze_88936 = 0;
    unsigned char *mem_86864 = NULL;
    int64_t mem_86869_cached_sizze_88937 = 0;
    unsigned char *mem_86869 = NULL;
    int64_t mem_86880_cached_sizze_88938 = 0;
    unsigned char *mem_86880 = NULL;
    int64_t mem_86885_cached_sizze_88939 = 0;
    unsigned char *mem_86885 = NULL;
    int64_t mem_86892_cached_sizze_88940 = 0;
    unsigned char *mem_86892 = NULL;
    int64_t mem_86903_cached_sizze_88941 = 0;
    unsigned char *mem_86903 = NULL;
    int64_t mem_86908_cached_sizze_88942 = 0;
    unsigned char *mem_86908 = NULL;
    int64_t mem_86915_cached_sizze_88943 = 0;
    unsigned char *mem_86915 = NULL;
    int64_t mem_86926_cached_sizze_88944 = 0;
    unsigned char *mem_86926 = NULL;
    int64_t mem_86927_cached_sizze_88945 = 0;
    unsigned char *mem_86927 = NULL;
    int64_t mem_86928_cached_sizze_88946 = 0;
    unsigned char *mem_86928 = NULL;
    int64_t mem_86941_cached_sizze_88947 = 0;
    unsigned char *mem_86941 = NULL;
    int64_t mem_86942_cached_sizze_88948 = 0;
    unsigned char *mem_86942 = NULL;
    int64_t mem_86943_cached_sizze_88949 = 0;
    unsigned char *mem_86943 = NULL;
    int64_t mem_86974_cached_sizze_88950 = 0;
    unsigned char *mem_86974 = NULL;
    int64_t mem_86975_cached_sizze_88951 = 0;
    unsigned char *mem_86975 = NULL;
    int64_t mem_86976_cached_sizze_88952 = 0;
    unsigned char *mem_86976 = NULL;
    int64_t mem_86992_cached_sizze_88953 = 0;
    unsigned char *mem_86992 = NULL;
    int64_t mem_86993_cached_sizze_88954 = 0;
    unsigned char *mem_86993 = NULL;
    int64_t mem_86994_cached_sizze_88955 = 0;
    unsigned char *mem_86994 = NULL;
    int64_t mem_87007_cached_sizze_88956 = 0;
    unsigned char *mem_87007 = NULL;
    int64_t mem_87008_cached_sizze_88957 = 0;
    unsigned char *mem_87008 = NULL;
    int64_t mem_87009_cached_sizze_88958 = 0;
    unsigned char *mem_87009 = NULL;
    int64_t mem_87055_cached_sizze_88959 = 0;
    unsigned char *mem_87055 = NULL;
    int64_t mem_87061_cached_sizze_88960 = 0;
    unsigned char *mem_87061 = NULL;
    int64_t mem_87066_cached_sizze_88961 = 0;
    unsigned char *mem_87066 = NULL;
    int64_t mem_87077_cached_sizze_88962 = 0;
    unsigned char *mem_87077 = NULL;
    int64_t mem_87082_cached_sizze_88963 = 0;
    unsigned char *mem_87082 = NULL;
    int64_t mem_87093_cached_sizze_88964 = 0;
    unsigned char *mem_87093 = NULL;
    int64_t mem_87098_cached_sizze_88965 = 0;
    unsigned char *mem_87098 = NULL;
    int64_t mem_87105_cached_sizze_88966 = 0;
    unsigned char *mem_87105 = NULL;
    int64_t mem_87116_cached_sizze_88967 = 0;
    unsigned char *mem_87116 = NULL;
    int64_t mem_87121_cached_sizze_88968 = 0;
    unsigned char *mem_87121 = NULL;
    int64_t mem_87137_cached_sizze_88969 = 0;
    unsigned char *mem_87137 = NULL;
    int64_t mem_87142_cached_sizze_88970 = 0;
    unsigned char *mem_87142 = NULL;
    int64_t mem_87153_cached_sizze_88971 = 0;
    unsigned char *mem_87153 = NULL;
    int64_t mem_87158_cached_sizze_88972 = 0;
    unsigned char *mem_87158 = NULL;
    int64_t mem_87169_cached_sizze_88973 = 0;
    unsigned char *mem_87169 = NULL;
    int64_t mem_87174_cached_sizze_88974 = 0;
    unsigned char *mem_87174 = NULL;
    int64_t mem_87185_cached_sizze_88975 = 0;
    unsigned char *mem_87185 = NULL;
    int64_t mem_87190_cached_sizze_88976 = 0;
    unsigned char *mem_87190 = NULL;
    int64_t mem_87197_cached_sizze_88977 = 0;
    unsigned char *mem_87197 = NULL;
    int64_t mem_87208_cached_sizze_88978 = 0;
    unsigned char *mem_87208 = NULL;
    int64_t mem_87213_cached_sizze_88979 = 0;
    unsigned char *mem_87213 = NULL;
    int64_t mem_87224_cached_sizze_88980 = 0;
    unsigned char *mem_87224 = NULL;
    int64_t mem_87229_cached_sizze_88981 = 0;
    unsigned char *mem_87229 = NULL;
    int64_t mem_87240_cached_sizze_88982 = 0;
    unsigned char *mem_87240 = NULL;
    int64_t mem_87245_cached_sizze_88983 = 0;
    unsigned char *mem_87245 = NULL;
    int64_t mem_87256_cached_sizze_88984 = 0;
    unsigned char *mem_87256 = NULL;
    int64_t mem_87261_cached_sizze_88985 = 0;
    unsigned char *mem_87261 = NULL;
    int64_t mem_87277_cached_sizze_88986 = 0;
    unsigned char *mem_87277 = NULL;
    struct memblock mem_87272;
    
    mem_87272.references = NULL;
    
    struct memblock mem_out_88608;
    
    mem_out_88608.references = NULL;
    
    struct memblock mem_86828 = ctx->constants->mem_86828;
    struct memblock mem_86829 = ctx->constants->mem_86829;
    struct memblock mem_86830 = ctx->constants->mem_86830;
    struct memblock mem_86831 = ctx->constants->mem_86831;
    struct memblock mem_86832 = ctx->constants->mem_86832;
    struct memblock mem_86833 = ctx->constants->mem_86833;
    struct memblock mem_86834 = ctx->constants->mem_86834;
    struct memblock mem_86835 = ctx->constants->mem_86835;
    struct memblock mem_86836 = ctx->constants->mem_86836;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_86848_cached_sizze_88934 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_86848, &mem_86848_cached_sizze_88934, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86853_cached_sizze_88935 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_86853, &mem_86853_cached_sizze_88935, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86005 = 0; i_86005 < (int64_t) 16; i_86005++) {
        // futhark/microgpt.fut:377:41-50
        
        int64_t tmp_76841 = ((int64_t *) tokens_mem_86846.mem)[i_86005];
        
        // futhark/microgpt.fut:377:37-51
        
        bool x_76842 = sle64((int64_t) 0, tmp_76841);
        
        // futhark/microgpt.fut:377:37-51
        
        bool y_76843 = slt64(tmp_76841, (int64_t) 27);
        
        // futhark/microgpt.fut:377:37-51
        
        bool bounds_check_76844 = x_76842 && y_76843;
        
        // futhark/microgpt.fut:377:37-51
        
        bool index_certs_76845;
        
        if (!bounds_check_76844) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_76841, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:377:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:377:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86001 = 0; i_86001 < (int64_t) 16; i_86001++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_76852 = ((double *) wte_mem_86842.mem)[tmp_76841 * (int64_t) 16 + i_86001];
            
            ((double *) mem_86853)[i_86001] = lifted_lambda_res_76852;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_86848, i_86005 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_86853, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86864_cached_sizze_88936 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_86864, &mem_86864_cached_sizze_88936, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86869_cached_sizze_88937 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_86869, &mem_86869_cached_sizze_88937, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86013 = 0; i_86013 < (int64_t) 16; i_86013++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86009 = 0; i_86009 < (int64_t) 16; i_86009++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_76884 = ((double *) wpe_mem_86840.mem)[i_86013 * (int64_t) 16 + i_86009];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_76885 = ((double *) mem_86848)[i_86013 * (int64_t) 16 + i_86009];
            
            // futhark/microgpt.fut:148:42-82
            
            double zp_res_76886 = zp_lhs_76884 + zp_rhs_76885;
            
            ((double *) mem_86869)[i_86009] = zp_res_76886;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_86864, i_86013 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_86869, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86880_cached_sizze_88938 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_86880, &mem_86880_cached_sizze_88938, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86885_cached_sizze_88939 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_86885, &mem_86885_cached_sizze_88939, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86892_cached_sizze_88940 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_86892, &mem_86892_cached_sizze_88940, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86025 = 0; i_86025 < (int64_t) 16; i_86025++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86017 = 0; i_86017 < (int64_t) 16; i_86017++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_76901 = ((double *) mem_86864)[i_86025 * (int64_t) 16 + i_86017];
            
            // futhark/microgpt.fut:149:77-114
            
            double zt_res_76902 = zt_lhs_76901 * zt_lhs_76901;
            
            ((double *) mem_86885)[i_86017] = zt_res_76902;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_76904;
        double r_76906 = 0.0;
        
        for (int64_t i_76905 = 0; i_76905 < (int64_t) 16; i_76905++) {
            // futhark/microgpt.fut:150:37-47
            
            double lifted_lambda_res_76907 = ((double *) mem_86885)[i_76905];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_76908 = r_76906 + lifted_lambda_res_76907;
            double r_tmp_88615 = zp_res_76908;
            
            r_76906 = r_tmp_88615;
        }
        defunc_0_lifted_lambda_res_76904 = r_76906;
        // futhark/microgpt.fut:150:17-64
        
        double zs_res_76909 = defunc_0_lifted_lambda_res_76904 / 16.0;
        
        // futhark/microgpt.fut:151:24-55
        
        double zp_res_76910 = 1.0e-5 + zs_res_76909;
        
        // futhark/microgpt.fut:151:16-55
        
        double sqrt_res_76911 = futrts_sqrt64(zp_res_76910);
        
        // futhark/microgpt.fut:152:27-38
        
        double zs_res_76912 = 1.0 / sqrt_res_76911;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86021 = 0; i_86021 < (int64_t) 16; i_86021++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_76919 = ((double *) mem_86864)[i_86025 * (int64_t) 16 + i_86021];
            
            // futhark/microgpt.fut:152:5-38
            
            double zt_res_76920 = zs_res_76912 * zt_lhs_76919;
            
            ((double *) mem_86892)[i_86021] = zt_res_76920;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_86880, i_86025 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_86892, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86903_cached_sizze_88941 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_86903, &mem_86903_cached_sizze_88941, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86908_cached_sizze_88942 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_86908, &mem_86908_cached_sizze_88942, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86915_cached_sizze_88943 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_86915, &mem_86915_cached_sizze_88943, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86037 = 0; i_86037 < (int64_t) 16; i_86037++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86029 = 0; i_86029 < (int64_t) 16; i_86029++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_76935 = ((double *) mem_86880)[i_86037 * (int64_t) 16 + i_86029];
            
            // futhark/microgpt.fut:153:77-114
            
            double zt_res_76936 = zt_lhs_76935 * zt_lhs_76935;
            
            ((double *) mem_86908)[i_86029] = zt_res_76936;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_76938;
        double r_76940 = 0.0;
        
        for (int64_t i_76939 = 0; i_76939 < (int64_t) 16; i_76939++) {
            // futhark/microgpt.fut:154:37-47
            
            double lifted_lambda_res_76941 = ((double *) mem_86908)[i_76939];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_76942 = r_76940 + lifted_lambda_res_76941;
            double r_tmp_88619 = zp_res_76942;
            
            r_76940 = r_tmp_88619;
        }
        defunc_0_lifted_lambda_res_76938 = r_76940;
        // futhark/microgpt.fut:154:17-64
        
        double zs_res_76943 = defunc_0_lifted_lambda_res_76938 / 16.0;
        
        // futhark/microgpt.fut:155:24-55
        
        double zp_res_76944 = 1.0e-5 + zs_res_76943;
        
        // futhark/microgpt.fut:155:16-55
        
        double sqrt_res_76945 = futrts_sqrt64(zp_res_76944);
        
        // futhark/microgpt.fut:156:27-38
        
        double zs_res_76946 = 1.0 / sqrt_res_76945;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86033 = 0; i_86033 < (int64_t) 16; i_86033++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_76953 = ((double *) mem_86880)[i_86037 * (int64_t) 16 + i_86033];
            
            // futhark/microgpt.fut:156:5-38
            
            double zt_res_76954 = zs_res_76946 * zt_lhs_76953;
            
            ((double *) mem_86915)[i_86033] = zt_res_76954;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_86903, i_86037 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_86915, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86926_cached_sizze_88944 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_86926, &mem_86926_cached_sizze_88944, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86927_cached_sizze_88945 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_86927, &mem_86927_cached_sizze_88945, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86928_cached_sizze_88946 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_86928, &mem_86928_cached_sizze_88946, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86941_cached_sizze_88947 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_86941, &mem_86941_cached_sizze_88947, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86942_cached_sizze_88948 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_86942, &mem_86942_cached_sizze_88948, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86943_cached_sizze_88949 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_86943, &mem_86943_cached_sizze_88949, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86055 = 0; i_86055 < (int64_t) 16; i_86055++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86045 = 0; i_86045 < (int64_t) 16; i_86045++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_82510;
            double r_82512 = 0.0;
            
            for (int64_t i_82511 = 0; i_82511 < (int64_t) 16; i_82511++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_82513 = ((double *) wqry_mem_86841.mem)[i_86045 * (int64_t) 16 + i_82511];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_82514 = ((double *) mem_86903)[i_86055 * (int64_t) 16 + i_82511];
                
                // futhark/microgpt.fut:157:66-105
                
                double zt_res_82515 = zt_lhs_82513 * zt_rhs_82514;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_82516 = r_82512 + zt_res_82515;
                double r_tmp_88627 = zp_res_82516;
                
                r_82512 = r_tmp_88627;
            }
            defunc_0_lifted_lambda_res_82510 = r_82512;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_82523;
            double r_82525 = 0.0;
            
            for (int64_t i_82524 = 0; i_82524 < (int64_t) 16; i_82524++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_82526 = ((double *) wkey_mem_86838.mem)[i_86045 * (int64_t) 16 + i_82524];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_82527 = ((double *) mem_86903)[i_86055 * (int64_t) 16 + i_82524];
                
                // futhark/microgpt.fut:158:66-105
                
                double zt_res_82528 = zt_lhs_82526 * zt_rhs_82527;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_82529 = r_82525 + zt_res_82528;
                double r_tmp_88628 = zp_res_82529;
                
                r_82525 = r_tmp_88628;
            }
            defunc_0_lifted_lambda_res_82523 = r_82525;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_82539;
            double r_82541 = 0.0;
            
            for (int64_t i_82540 = 0; i_82540 < (int64_t) 16; i_82540++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_82542 = ((double *) wval_mem_86844.mem)[i_86045 * (int64_t) 16 + i_82540];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_82543 = ((double *) mem_86903)[i_86055 * (int64_t) 16 + i_82540];
                
                // futhark/microgpt.fut:159:66-105
                
                double zt_res_82544 = zt_lhs_82542 * zt_rhs_82543;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_82545 = r_82541 + zt_res_82544;
                double r_tmp_88629 = zp_res_82545;
                
                r_82541 = r_tmp_88629;
            }
            defunc_0_lifted_lambda_res_82539 = r_82541;
            ((double *) mem_86941)[i_86045] = defunc_0_lifted_lambda_res_82539;
            ((double *) mem_86942)[i_86045] = defunc_0_lifted_lambda_res_82523;
            ((double *) mem_86943)[i_86045] = defunc_0_lifted_lambda_res_82510;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_86926, i_86055 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_86941, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_86927, i_86055 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_86942, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_86928, i_86055 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_86943, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86974_cached_sizze_88950 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_86974, &mem_86974_cached_sizze_88950, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86975_cached_sizze_88951 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_86975, &mem_86975_cached_sizze_88951, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86976_cached_sizze_88952 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_86976, &mem_86976_cached_sizze_88952, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86992_cached_sizze_88953 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_86992, &mem_86992_cached_sizze_88953, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86993_cached_sizze_88954 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_86993, &mem_86993_cached_sizze_88954, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86994_cached_sizze_88955 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_86994, &mem_86994_cached_sizze_88955, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87007_cached_sizze_88956 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_87007, &mem_87007_cached_sizze_88956, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87008_cached_sizze_88957 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_87008, &mem_87008_cached_sizze_88957, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87009_cached_sizze_88958 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_87009, &mem_87009_cached_sizze_88958, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86085 = 0; i_86085 < (int64_t) 4; i_86085++) {
        // futhark/microgpt.fut:160:69-72
        
        int64_t zp_lhs_82385 = mul64((int64_t) 4, i_86085);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86075 = 0; i_86075 < (int64_t) 16; i_86075++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86065 = 0; i_86065 < (int64_t) 4; i_86065++) {
                // futhark/microgpt.fut:160:74-81
                
                int64_t tmp_82703 = add64(zp_lhs_82385, i_86065);
                
                // futhark/microgpt.fut:160:51-83
                
                bool x_82704 = sle64((int64_t) 0, tmp_82703);
                
                // futhark/microgpt.fut:160:51-83
                
                bool y_82705 = slt64(tmp_82703, (int64_t) 16);
                
                // futhark/microgpt.fut:160:51-83
                
                bool bounds_check_82706 = x_82704 && y_82705;
                
                // futhark/microgpt.fut:160:51-83
                
                bool index_certs_82707;
                
                if (!bounds_check_82706) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_82703, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:160:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:160:15-84\n   #9  futhark/microgpt.fut:378:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_82708 = ((double *) mem_86928)[i_86075 * (int64_t) 16 + tmp_82703];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_82716 = ((double *) mem_86927)[i_86075 * (int64_t) 16 + tmp_82703];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_82727 = ((double *) mem_86926)[i_86075 * (int64_t) 16 + tmp_82703];
                
                ((double *) mem_87007)[i_86065] = lifted_lambda_res_82727;
                ((double *) mem_87008)[i_86065] = lifted_lambda_res_82716;
                ((double *) mem_87009)[i_86065] = lifted_lambda_res_82708;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_86992, i_86075 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87007, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_86993, i_86075 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87008, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_86994, i_86075 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87009, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_86974, i_86085 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_86992, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_86975, i_86085 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_86993, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_86976, i_86085 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_86994, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87055_cached_sizze_88959 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87055, &mem_87055_cached_sizze_88959, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87061_cached_sizze_88960 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87061, &mem_87061_cached_sizze_88960, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87066_cached_sizze_88961 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87066, &mem_87066_cached_sizze_88961, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87077_cached_sizze_88962 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87077, &mem_87077_cached_sizze_88962, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87082_cached_sizze_88963 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87082, &mem_87082_cached_sizze_88963, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87093_cached_sizze_88964 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87093, &mem_87093_cached_sizze_88964, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87098_cached_sizze_88965 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87098, &mem_87098_cached_sizze_88965, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87105_cached_sizze_88966 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87105, &mem_87105_cached_sizze_88966, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87116_cached_sizze_88967 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_87116, &mem_87116_cached_sizze_88967, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87121_cached_sizze_88968 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_87121, &mem_87121_cached_sizze_88968, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86129 = 0; i_86129 < (int64_t) 4; i_86129++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86095 = 0; i_86095 < (int64_t) 16; i_86095++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86091 = 0; i_86091 < (int64_t) 16; i_86091++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77099;
                double r_77101 = 0.0;
                
                for (int64_t i_77100 = 0; i_77100 < (int64_t) 4; i_77100++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77102 = ((double *) mem_86976)[i_86129 * (int64_t) 64 + i_86095 * (int64_t) 4 + i_77100];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77103 = ((double *) mem_86975)[i_86129 * (int64_t) 64 + i_86091 * (int64_t) 4 + i_77100];
                    
                    // futhark/microgpt.fut:163:113-164
                    
                    double zt_res_77104 = zt_lhs_77102 * zt_rhs_77103;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77105 = r_77101 + zt_res_77104;
                    double r_tmp_88642 = zp_res_77105;
                    
                    r_77101 = r_tmp_88642;
                }
                defunc_0_lifted_lambda_res_77099 = r_77101;
                ((double *) mem_87066)[i_86091] = defunc_0_lifted_lambda_res_77099;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87061, i_86095 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87066, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86103 = 0; i_86103 < (int64_t) 16; i_86103++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86099 = 0; i_86099 < (int64_t) 16; i_86099++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_77120 = ((double *) mem_87061)[i_86103 * (int64_t) 16 + i_86099];
                
                // futhark/microgpt.fut:164:47-78
                
                double zs_res_77121 = zs_lhs_77120 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_77122 = ((double *) mask_mem_86847.mem)[i_86103 * (int64_t) 16 + i_86099];
                
                // futhark/microgpt.fut:164:65-102
                
                double zp_res_77123 = zs_res_77121 + zp_rhs_77122;
                
                ((double *) mem_87082)[i_86099] = zp_res_77123;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87077, i_86103 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87082, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86117 = 0; i_86117 < (int64_t) 16; i_86117++) {
            // futhark/microgpt.fut:103:13-33
            
            double defunc_0_reduce_res_82800;
            double redout_86105 = -INFINITY;
            
            for (int64_t i_86106 = 0; i_86106 < (int64_t) 16; i_86106++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_82754 = ((double *) mem_87077)[i_86117 * (int64_t) 16 + i_86106];
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_77144 = fmax64(lifted_lambda_res_82754, redout_86105);
                double redout_tmp_88646 = max_res_77144;
                
                redout_86105 = redout_tmp_88646;
            }
            defunc_0_reduce_res_82800 = redout_86105;
            // futhark/microgpt.fut:166:65-74
            
            double neg_res_77145 = -defunc_0_reduce_res_82800;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86109 = 0; i_86109 < (int64_t) 16; i_86109++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_77152 = ((double *) mem_87077)[i_86117 * (int64_t) 16 + i_86109];
                
                // futhark/microgpt.fut:166:43-74
                
                double zp_res_77153 = neg_res_77145 + zp_lhs_77152;
                
                // futhark/microgpt.fut:166:36-74
                
                double exp_res_77154 = futrts_exp64(zp_res_77153);
                
                ((double *) mem_87098)[i_86109] = exp_res_77154;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77156;
            double r_77158 = 0.0;
            
            for (int64_t i_77157 = 0; i_77157 < (int64_t) 16; i_77157++) {
                // futhark/microgpt.fut:167:36-46
                
                double lifted_lambda_res_77159 = ((double *) mem_87098)[i_77157];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77160 = r_77158 + lifted_lambda_res_77159;
                double r_tmp_88648 = zp_res_77160;
                
                r_77158 = r_tmp_88648;
            }
            defunc_0_lifted_lambda_res_77156 = r_77158;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86113 = 0; i_86113 < (int64_t) 16; i_86113++) {
                // futhark/microgpt.fut:168:5-15
                
                double zs_lhs_77167 = ((double *) mem_87098)[i_86113];
                
                // futhark/microgpt.fut:168:5-23
                
                double zs_res_77168 = zs_lhs_77167 / defunc_0_lifted_lambda_res_77156;
                
                ((double *) mem_87105)[i_86113] = zs_res_77168;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87093, i_86117 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87105, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86125 = 0; i_86125 < (int64_t) 16; i_86125++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86121 = 0; i_86121 < (int64_t) 4; i_86121++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77183;
                double r_77185 = 0.0;
                
                for (int64_t i_77184 = 0; i_77184 < (int64_t) 16; i_77184++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77186 = ((double *) mem_87093)[i_86125 * (int64_t) 16 + i_77184];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77187 = ((double *) mem_86974)[i_86129 * (int64_t) 64 + i_77184 * (int64_t) 4 + i_86121];
                    
                    // futhark/microgpt.fut:169:26-71
                    
                    double zt_res_77188 = zt_lhs_77186 * zt_rhs_77187;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77189 = r_77185 + zt_res_77188;
                    double r_tmp_88652 = zp_res_77189;
                    
                    r_77185 = r_tmp_88652;
                }
                defunc_0_lifted_lambda_res_77183 = r_77185;
                ((double *) mem_87121)[i_86121] = defunc_0_lifted_lambda_res_77183;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87116, i_86125 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87121, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_87055, i_86129 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_87116, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87137_cached_sizze_88969 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87137, &mem_87137_cached_sizze_88969, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87142_cached_sizze_88970 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87142, &mem_87142_cached_sizze_88970, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86137 = 0; i_86137 < (int64_t) 16; i_86137++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86133 = 0; i_86133 < (int64_t) 16; i_86133++) {
            // futhark/microgpt.fut:170:55-58
            
            int64_t tmp_77201 = sdiv64(i_86133, (int64_t) 4);
            
            // futhark/microgpt.fut:170:45-60
            
            bool x_77202 = sle64((int64_t) 0, tmp_77201);
            
            // futhark/microgpt.fut:170:45-60
            
            bool y_77203 = slt64(tmp_77201, (int64_t) 4);
            
            // futhark/microgpt.fut:170:45-60
            
            bool bounds_check_77204 = x_77202 && y_77203;
            
            // futhark/microgpt.fut:170:45-60
            
            bool index_certs_77205;
            
            if (!bounds_check_77204) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_77201, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:170:45-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:170:16-81\n   #6  futhark/microgpt.fut:378:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:170:75-78
            
            int64_t tmp_77206 = smod64(i_86133, (int64_t) 4);
            
            // futhark/microgpt.fut:170:45-80
            
            bool x_77207 = sle64((int64_t) 0, tmp_77206);
            
            // futhark/microgpt.fut:170:45-80
            
            bool y_77208 = slt64(tmp_77206, (int64_t) 4);
            
            // futhark/microgpt.fut:170:45-80
            
            bool bounds_check_77209 = x_77207 && y_77208;
            
            // futhark/microgpt.fut:170:45-80
            
            bool index_certs_77210;
            
            if (!bounds_check_77209) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_77206, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:170:45-80\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:170:16-81\n   #6  futhark/microgpt.fut:378:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_77211 = ((double *) mem_87055)[tmp_77201 * (int64_t) 64 + i_86137 * (int64_t) 4 + tmp_77206];
            
            ((double *) mem_87142)[i_86133] = lifted_lambda_res_77211;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_87137, i_86137 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87142, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87153_cached_sizze_88971 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87153, &mem_87153_cached_sizze_88971, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87158_cached_sizze_88972 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87158, &mem_87158_cached_sizze_88972, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86145 = 0; i_86145 < (int64_t) 16; i_86145++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86141 = 0; i_86141 < (int64_t) 16; i_86141++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77226;
            double r_77228 = 0.0;
            
            for (int64_t i_77227 = 0; i_77227 < (int64_t) 16; i_77227++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_77229 = ((double *) wout_mem_86839.mem)[i_86141 * (int64_t) 16 + i_77227];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_77230 = ((double *) mem_87137)[i_86145 * (int64_t) 16 + i_77227];
                
                // futhark/microgpt.fut:171:67-107
                
                double zt_res_77231 = zt_lhs_77229 * zt_rhs_77230;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77232 = r_77228 + zt_res_77231;
                double r_tmp_88657 = zp_res_77232;
                
                r_77228 = r_tmp_88657;
            }
            defunc_0_lifted_lambda_res_77226 = r_77228;
            ((double *) mem_87158)[i_86141] = defunc_0_lifted_lambda_res_77226;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_87153, i_86145 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87158, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87169_cached_sizze_88973 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87169, &mem_87169_cached_sizze_88973, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87174_cached_sizze_88974 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87174, &mem_87174_cached_sizze_88974, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86153 = 0; i_86153 < (int64_t) 16; i_86153++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86149 = 0; i_86149 < (int64_t) 16; i_86149++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_77247 = ((double *) mem_87153)[i_86153 * (int64_t) 16 + i_86149];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_77248 = ((double *) mem_86880)[i_86153 * (int64_t) 16 + i_86149];
            
            // futhark/microgpt.fut:172:46-84
            
            double zp_res_77249 = zp_lhs_77247 + zp_rhs_77248;
            
            ((double *) mem_87174)[i_86149] = zp_res_77249;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_87169, i_86153 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87174, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87185_cached_sizze_88975 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87185, &mem_87185_cached_sizze_88975, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87190_cached_sizze_88976 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87190, &mem_87190_cached_sizze_88976, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87197_cached_sizze_88977 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87197, &mem_87197_cached_sizze_88977, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86165 = 0; i_86165 < (int64_t) 16; i_86165++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86157 = 0; i_86157 < (int64_t) 16; i_86157++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_77264 = ((double *) mem_87169)[i_86165 * (int64_t) 16 + i_86157];
            
            // futhark/microgpt.fut:173:78-117
            
            double zt_res_77265 = zt_lhs_77264 * zt_lhs_77264;
            
            ((double *) mem_87190)[i_86157] = zt_res_77265;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_77267;
        double r_77269 = 0.0;
        
        for (int64_t i_77268 = 0; i_77268 < (int64_t) 16; i_77268++) {
            // futhark/microgpt.fut:174:37-47
            
            double lifted_lambda_res_77270 = ((double *) mem_87190)[i_77268];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_77271 = r_77269 + lifted_lambda_res_77270;
            double r_tmp_88662 = zp_res_77271;
            
            r_77269 = r_tmp_88662;
        }
        defunc_0_lifted_lambda_res_77267 = r_77269;
        // futhark/microgpt.fut:174:17-64
        
        double zs_res_77272 = defunc_0_lifted_lambda_res_77267 / 16.0;
        
        // futhark/microgpt.fut:175:24-55
        
        double zp_res_77273 = 1.0e-5 + zs_res_77272;
        
        // futhark/microgpt.fut:175:16-55
        
        double sqrt_res_77274 = futrts_sqrt64(zp_res_77273);
        
        // futhark/microgpt.fut:176:28-39
        
        double zs_res_77275 = 1.0 / sqrt_res_77274;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86161 = 0; i_86161 < (int64_t) 16; i_86161++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_77282 = ((double *) mem_87169)[i_86165 * (int64_t) 16 + i_86161];
            
            // futhark/microgpt.fut:176:5-39
            
            double zt_res_77283 = zs_res_77275 * zt_lhs_77282;
            
            ((double *) mem_87197)[i_86161] = zt_res_77283;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_87185, i_86165 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87197, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87208_cached_sizze_88978 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_87208, &mem_87208_cached_sizze_88978, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87213_cached_sizze_88979 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_87213, &mem_87213_cached_sizze_88979, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86173 = 0; i_86173 < (int64_t) 16; i_86173++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86169 = 0; i_86169 < (int64_t) 64; i_86169++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77299;
            double r_77301 = 0.0;
            
            for (int64_t i_77300 = 0; i_77300 < (int64_t) 16; i_77300++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_77302 = ((double *) wup_mem_86843.mem)[i_86169 * (int64_t) 16 + i_77300];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_77303 = ((double *) mem_87185)[i_86173 * (int64_t) 16 + i_77300];
                
                // futhark/microgpt.fut:177:67-106
                
                double zt_res_77304 = zt_lhs_77302 * zt_rhs_77303;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77305 = r_77301 + zt_res_77304;
                double r_tmp_88666 = zp_res_77305;
                
                r_77301 = r_tmp_88666;
            }
            defunc_0_lifted_lambda_res_77299 = r_77301;
            ((double *) mem_87213)[i_86169] = defunc_0_lifted_lambda_res_77299;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_87208, i_86173 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87213, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87224_cached_sizze_88980 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_87224, &mem_87224_cached_sizze_88980, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87229_cached_sizze_88981 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_87229, &mem_87229_cached_sizze_88981, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86181 = 0; i_86181 < (int64_t) 16; i_86181++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86177 = 0; i_86177 < (int64_t) 64; i_86177++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_77320 = ((double *) mem_87208)[i_86181 * (int64_t) 64 + i_86177];
            
            // futhark/microgpt.fut:178:45-73
            
            double max_res_77321 = fmax64(0.0, max_arg0_77320);
            
            ((double *) mem_87229)[i_86177] = max_res_77321;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_87224, i_86181 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87229, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87240_cached_sizze_88982 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87240, &mem_87240_cached_sizze_88982, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87245_cached_sizze_88983 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87245, &mem_87245_cached_sizze_88983, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86189 = 0; i_86189 < (int64_t) 16; i_86189++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86185 = 0; i_86185 < (int64_t) 16; i_86185++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77336;
            double r_77338 = 0.0;
            
            for (int64_t i_77337 = 0; i_77337 < (int64_t) 64; i_77337++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_77339 = ((double *) wdown_mem_86837.mem)[i_86185 * (int64_t) 64 + i_77337];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_77340 = ((double *) mem_87224)[i_86189 * (int64_t) 64 + i_77337];
                
                // futhark/microgpt.fut:179:67-108
                
                double zt_res_77341 = zt_lhs_77339 * zt_rhs_77340;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77342 = r_77338 + zt_res_77341;
                double r_tmp_88671 = zp_res_77342;
                
                r_77338 = r_tmp_88671;
            }
            defunc_0_lifted_lambda_res_77336 = r_77338;
            ((double *) mem_87245)[i_86185] = defunc_0_lifted_lambda_res_77336;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_87240, i_86189 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87245, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87256_cached_sizze_88984 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87256, &mem_87256_cached_sizze_88984, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87261_cached_sizze_88985 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87261, &mem_87261_cached_sizze_88985, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86197 = 0; i_86197 < (int64_t) 16; i_86197++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86193 = 0; i_86193 < (int64_t) 16; i_86193++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_77357 = ((double *) mem_87240)[i_86197 * (int64_t) 16 + i_86193];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_77358 = ((double *) mem_87169)[i_86197 * (int64_t) 16 + i_86193];
            
            // futhark/microgpt.fut:180:46-85
            
            double zp_res_77359 = zp_lhs_77357 + zp_rhs_77358;
            
            ((double *) mem_87261)[i_86193] = zp_res_77359;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_87256, i_86197 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87261, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_87272, (int64_t) 3456, "mem_87272")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87277_cached_sizze_88986 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_87277, &mem_87277_cached_sizze_88986, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86205 = 0; i_86205 < (int64_t) 16; i_86205++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86201 = 0; i_86201 < (int64_t) 27; i_86201++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77375;
            double r_77377 = 0.0;
            
            for (int64_t i_77376 = 0; i_77376 < (int64_t) 16; i_77376++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_77378 = ((double *) wvoc_mem_86845.mem)[i_86201 * (int64_t) 16 + i_77376];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_77379 = ((double *) mem_87256)[i_86205 * (int64_t) 16 + i_77376];
                
                // futhark/microgpt.fut:181:56-96
                
                double zt_res_77380 = zt_lhs_77378 * zt_rhs_77379;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77381 = r_77377 + zt_res_77380;
                double r_tmp_88676 = zp_res_77381;
                
                r_77377 = r_tmp_88676;
            }
            defunc_0_lifted_lambda_res_77375 = r_77377;
            ((double *) mem_87277)[i_86201] = defunc_0_lifted_lambda_res_77375;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_87272.mem, i_86205 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87277, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_88608, &mem_87272, "mem_87272") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88933, &mem_out_88608, "mem_out_88608") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_86848);
        free(mem_86853);
        free(mem_86864);
        free(mem_86869);
        free(mem_86880);
        free(mem_86885);
        free(mem_86892);
        free(mem_86903);
        free(mem_86908);
        free(mem_86915);
        free(mem_86926);
        free(mem_86927);
        free(mem_86928);
        free(mem_86941);
        free(mem_86942);
        free(mem_86943);
        free(mem_86974);
        free(mem_86975);
        free(mem_86976);
        free(mem_86992);
        free(mem_86993);
        free(mem_86994);
        free(mem_87007);
        free(mem_87008);
        free(mem_87009);
        free(mem_87055);
        free(mem_87061);
        free(mem_87066);
        free(mem_87077);
        free(mem_87082);
        free(mem_87093);
        free(mem_87098);
        free(mem_87105);
        free(mem_87116);
        free(mem_87121);
        free(mem_87137);
        free(mem_87142);
        free(mem_87153);
        free(mem_87158);
        free(mem_87169);
        free(mem_87174);
        free(mem_87185);
        free(mem_87190);
        free(mem_87197);
        free(mem_87208);
        free(mem_87213);
        free(mem_87224);
        free(mem_87229);
        free(mem_87240);
        free(mem_87245);
        free(mem_87256);
        free(mem_87261);
        free(mem_87277);
        if (memblock_unref(ctx, &mem_87272, "mem_87272") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88608, "mem_out_88608") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_88987, struct memblock *mem_out_p_88988, struct memblock *mem_out_p_88989, struct memblock *mem_out_p_88990, struct memblock *mem_out_p_88991, struct memblock *mem_out_p_88992, struct memblock *mem_out_p_88993, struct memblock *mem_out_p_88994, struct memblock *mem_out_p_88995, struct memblock wte_mem_86837, struct memblock wpe_mem_86838, struct memblock wqry_mem_86839, struct memblock wkey_mem_86840, struct memblock wval_mem_86841, struct memblock wout_mem_86842, struct memblock wup_mem_86843, struct memblock wdown_mem_86844, struct memblock wvoc_mem_86845)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_88616;
    
    mem_out_88616.references = NULL;
    
    struct memblock mem_out_88615;
    
    mem_out_88615.references = NULL;
    
    struct memblock mem_out_88614;
    
    mem_out_88614.references = NULL;
    
    struct memblock mem_out_88613;
    
    mem_out_88613.references = NULL;
    
    struct memblock mem_out_88612;
    
    mem_out_88612.references = NULL;
    
    struct memblock mem_out_88611;
    
    mem_out_88611.references = NULL;
    
    struct memblock mem_out_88610;
    
    mem_out_88610.references = NULL;
    
    struct memblock mem_out_88609;
    
    mem_out_88609.references = NULL;
    
    struct memblock mem_out_88608;
    
    mem_out_88608.references = NULL;
    
    struct memblock mem_86828 = ctx->constants->mem_86828;
    struct memblock mem_86829 = ctx->constants->mem_86829;
    struct memblock mem_86830 = ctx->constants->mem_86830;
    struct memblock mem_86831 = ctx->constants->mem_86831;
    struct memblock mem_86832 = ctx->constants->mem_86832;
    struct memblock mem_86833 = ctx->constants->mem_86833;
    struct memblock mem_86834 = ctx->constants->mem_86834;
    struct memblock mem_86835 = ctx->constants->mem_86835;
    struct memblock mem_86836 = ctx->constants->mem_86836;
    
    if (memblock_set(ctx, &mem_out_88608, &wdown_mem_86844, "wdown_mem_86844") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88609, &wkey_mem_86840, "wkey_mem_86840") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88610, &wout_mem_86842, "wout_mem_86842") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88611, &wpe_mem_86838, "wpe_mem_86838") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88612, &wqry_mem_86839, "wqry_mem_86839") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88613, &wte_mem_86837, "wte_mem_86837") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88614, &wup_mem_86843, "wup_mem_86843") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88615, &wval_mem_86841, "wval_mem_86841") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88616, &wvoc_mem_86845, "wvoc_mem_86845") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88987, &mem_out_88608, "mem_out_88608") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88988, &mem_out_88609, "mem_out_88609") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88989, &mem_out_88610, "mem_out_88610") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88990, &mem_out_88611, "mem_out_88611") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88991, &mem_out_88612, "mem_out_88612") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88992, &mem_out_88613, "mem_out_88613") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88993, &mem_out_88614, "mem_out_88614") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88994, &mem_out_88615, "mem_out_88615") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88995, &mem_out_88616, "mem_out_88616") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_88616, "mem_out_88616") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88615, "mem_out_88615") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88614, "mem_out_88614") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88613, "mem_out_88613") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88612, "mem_out_88612") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88611, "mem_out_88611") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88610, "mem_out_88610") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88609, "mem_out_88609") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88608, "mem_out_88608") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_88996, struct memblock *mem_out_p_88997, struct memblock *mem_out_p_88998, struct memblock *mem_out_p_88999, struct memblock *mem_out_p_89000, struct memblock *mem_out_p_89001, struct memblock *mem_out_p_89002, struct memblock *mem_out_p_89003, struct memblock *mem_out_p_89004, struct memblock *mem_out_p_89005, struct memblock *mem_out_p_89006, struct memblock *mem_out_p_89007, struct memblock *mem_out_p_89008, struct memblock *mem_out_p_89009, struct memblock *mem_out_p_89010, struct memblock *mem_out_p_89011, struct memblock *mem_out_p_89012, struct memblock *mem_out_p_89013, struct memblock *mem_out_p_89014, struct memblock *mem_out_p_89015, struct memblock *mem_out_p_89016, struct memblock *mem_out_p_89017, struct memblock *mem_out_p_89018, struct memblock *mem_out_p_89019, struct memblock *mem_out_p_89020, struct memblock *mem_out_p_89021, struct memblock *mem_out_p_89022, struct memblock wdown_mem_86837, struct memblock wkey_mem_86838, struct memblock wout_mem_86839, struct memblock wpe_mem_86840, struct memblock wqry_mem_86841, struct memblock wte_mem_86842, struct memblock wup_mem_86843, struct memblock wval_mem_86844, struct memblock wvoc_mem_86845, struct memblock wdown_mem_86846, struct memblock wkey_mem_86847, struct memblock wout_mem_86848, struct memblock wpe_mem_86849, struct memblock wqry_mem_86850, struct memblock wte_mem_86851, struct memblock wup_mem_86852, struct memblock wval_mem_86853, struct memblock wvoc_mem_86854, struct memblock wdown_mem_86855, struct memblock wkey_mem_86856, struct memblock wout_mem_86857, struct memblock wpe_mem_86858, struct memblock wqry_mem_86859, struct memblock wte_mem_86860, struct memblock wup_mem_86861, struct memblock wval_mem_86862, struct memblock wvoc_mem_86863, struct memblock masks_mem_86864, struct memblock dls_mem_86865, struct memblock seqs_mem_86866)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_86975_cached_sizze_89023 = 0;
    unsigned char *mem_86975 = NULL;
    int64_t mem_86976_cached_sizze_89024 = 0;
    unsigned char *mem_86976 = NULL;
    int64_t mem_86985_cached_sizze_89025 = 0;
    unsigned char *mem_86985 = NULL;
    int64_t mem_86992_cached_sizze_89026 = 0;
    unsigned char *mem_86992 = NULL;
    int64_t mem_87007_cached_sizze_89027 = 0;
    unsigned char *mem_87007 = NULL;
    int64_t mem_87008_cached_sizze_89028 = 0;
    unsigned char *mem_87008 = NULL;
    int64_t mem_87017_cached_sizze_89029 = 0;
    unsigned char *mem_87017 = NULL;
    int64_t mem_87024_cached_sizze_89030 = 0;
    unsigned char *mem_87024 = NULL;
    int64_t mem_87039_cached_sizze_89031 = 0;
    unsigned char *mem_87039 = NULL;
    int64_t mem_87040_cached_sizze_89032 = 0;
    unsigned char *mem_87040 = NULL;
    int64_t mem_87049_cached_sizze_89033 = 0;
    unsigned char *mem_87049 = NULL;
    int64_t mem_87050_cached_sizze_89034 = 0;
    unsigned char *mem_87050 = NULL;
    int64_t mem_87071_cached_sizze_89035 = 0;
    unsigned char *mem_87071 = NULL;
    int64_t mem_87072_cached_sizze_89036 = 0;
    unsigned char *mem_87072 = NULL;
    int64_t mem_87073_cached_sizze_89037 = 0;
    unsigned char *mem_87073 = NULL;
    int64_t mem_87085_cached_sizze_89038 = 0;
    unsigned char *mem_87085 = NULL;
    int64_t mem_87086_cached_sizze_89039 = 0;
    unsigned char *mem_87086 = NULL;
    int64_t mem_87110_cached_sizze_89040 = 0;
    unsigned char *mem_87110 = NULL;
    int64_t mem_87111_cached_sizze_89041 = 0;
    unsigned char *mem_87111 = NULL;
    int64_t mem_87112_cached_sizze_89042 = 0;
    unsigned char *mem_87112 = NULL;
    int64_t mem_87113_cached_sizze_89043 = 0;
    unsigned char *mem_87113 = NULL;
    int64_t mem_87129_cached_sizze_89044 = 0;
    unsigned char *mem_87129 = NULL;
    int64_t mem_87130_cached_sizze_89045 = 0;
    unsigned char *mem_87130 = NULL;
    int64_t mem_87131_cached_sizze_89046 = 0;
    unsigned char *mem_87131 = NULL;
    int64_t mem_87165_cached_sizze_89047 = 0;
    unsigned char *mem_87165 = NULL;
    int64_t mem_87166_cached_sizze_89048 = 0;
    unsigned char *mem_87166 = NULL;
    int64_t mem_87167_cached_sizze_89049 = 0;
    unsigned char *mem_87167 = NULL;
    int64_t mem_87183_cached_sizze_89050 = 0;
    unsigned char *mem_87183 = NULL;
    int64_t mem_87184_cached_sizze_89051 = 0;
    unsigned char *mem_87184 = NULL;
    int64_t mem_87185_cached_sizze_89052 = 0;
    unsigned char *mem_87185 = NULL;
    int64_t mem_87198_cached_sizze_89053 = 0;
    unsigned char *mem_87198 = NULL;
    int64_t mem_87199_cached_sizze_89054 = 0;
    unsigned char *mem_87199 = NULL;
    int64_t mem_87200_cached_sizze_89055 = 0;
    unsigned char *mem_87200 = NULL;
    int64_t mem_87246_cached_sizze_89056 = 0;
    unsigned char *mem_87246 = NULL;
    int64_t mem_87247_cached_sizze_89057 = 0;
    unsigned char *mem_87247 = NULL;
    int64_t mem_87258_cached_sizze_89058 = 0;
    unsigned char *mem_87258 = NULL;
    int64_t mem_87259_cached_sizze_89059 = 0;
    unsigned char *mem_87259 = NULL;
    int64_t mem_87268_cached_sizze_89060 = 0;
    unsigned char *mem_87268 = NULL;
    int64_t mem_87269_cached_sizze_89061 = 0;
    unsigned char *mem_87269 = NULL;
    int64_t mem_87290_cached_sizze_89062 = 0;
    unsigned char *mem_87290 = NULL;
    int64_t mem_87295_cached_sizze_89063 = 0;
    unsigned char *mem_87295 = NULL;
    int64_t mem_87306_cached_sizze_89064 = 0;
    unsigned char *mem_87306 = NULL;
    int64_t mem_87311_cached_sizze_89065 = 0;
    unsigned char *mem_87311 = NULL;
    int64_t mem_87318_cached_sizze_89066 = 0;
    unsigned char *mem_87318 = NULL;
    int64_t mem_87329_cached_sizze_89067 = 0;
    unsigned char *mem_87329 = NULL;
    int64_t mem_87334_cached_sizze_89068 = 0;
    unsigned char *mem_87334 = NULL;
    int64_t mem_87355_cached_sizze_89069 = 0;
    unsigned char *mem_87355 = NULL;
    int64_t mem_87360_cached_sizze_89070 = 0;
    unsigned char *mem_87360 = NULL;
    int64_t mem_87371_cached_sizze_89071 = 0;
    unsigned char *mem_87371 = NULL;
    int64_t mem_87376_cached_sizze_89072 = 0;
    unsigned char *mem_87376 = NULL;
    int64_t mem_87387_cached_sizze_89073 = 0;
    unsigned char *mem_87387 = NULL;
    int64_t mem_87392_cached_sizze_89074 = 0;
    unsigned char *mem_87392 = NULL;
    int64_t mem_87403_cached_sizze_89075 = 0;
    unsigned char *mem_87403 = NULL;
    int64_t mem_87404_cached_sizze_89076 = 0;
    unsigned char *mem_87404 = NULL;
    int64_t mem_87413_cached_sizze_89077 = 0;
    unsigned char *mem_87413 = NULL;
    int64_t mem_87414_cached_sizze_89078 = 0;
    unsigned char *mem_87414 = NULL;
    int64_t mem_87435_cached_sizze_89079 = 0;
    unsigned char *mem_87435 = NULL;
    int64_t mem_87436_cached_sizze_89080 = 0;
    unsigned char *mem_87436 = NULL;
    int64_t mem_87444_cached_sizze_89081 = 0;
    unsigned char *mem_87444 = NULL;
    int64_t mem_87458_cached_sizze_89082 = 0;
    unsigned char *mem_87458 = NULL;
    int64_t mem_87463_cached_sizze_89083 = 0;
    unsigned char *mem_87463 = NULL;
    int64_t mem_87474_cached_sizze_89084 = 0;
    unsigned char *mem_87474 = NULL;
    int64_t mem_87479_cached_sizze_89085 = 0;
    unsigned char *mem_87479 = NULL;
    int64_t mem_87490_cached_sizze_89086 = 0;
    unsigned char *mem_87490 = NULL;
    int64_t mem_87495_cached_sizze_89087 = 0;
    unsigned char *mem_87495 = NULL;
    int64_t mem_87506_cached_sizze_89088 = 0;
    unsigned char *mem_87506 = NULL;
    int64_t mem_87511_cached_sizze_89089 = 0;
    unsigned char *mem_87511 = NULL;
    int64_t mem_87522_cached_sizze_89090 = 0;
    unsigned char *mem_87522 = NULL;
    int64_t mem_87523_cached_sizze_89091 = 0;
    unsigned char *mem_87523 = NULL;
    int64_t mem_87532_cached_sizze_89092 = 0;
    unsigned char *mem_87532 = NULL;
    int64_t mem_87533_cached_sizze_89093 = 0;
    unsigned char *mem_87533 = NULL;
    int64_t mem_87546_cached_sizze_89094 = 0;
    unsigned char *mem_87546 = NULL;
    int64_t mem_87547_cached_sizze_89095 = 0;
    unsigned char *mem_87547 = NULL;
    int64_t mem_87568_cached_sizze_89096 = 0;
    unsigned char *mem_87568 = NULL;
    int64_t mem_87575_cached_sizze_89097 = 0;
    unsigned char *mem_87575 = NULL;
    int64_t mem_87580_cached_sizze_89098 = 0;
    unsigned char *mem_87580 = NULL;
    int64_t mem_87591_cached_sizze_89099 = 0;
    unsigned char *mem_87591 = NULL;
    int64_t mem_87596_cached_sizze_89100 = 0;
    unsigned char *mem_87596 = NULL;
    int64_t mem_87607_cached_sizze_89101 = 0;
    unsigned char *mem_87607 = NULL;
    int64_t mem_87608_cached_sizze_89102 = 0;
    unsigned char *mem_87608 = NULL;
    int64_t mem_87617_cached_sizze_89103 = 0;
    unsigned char *mem_87617 = NULL;
    int64_t mem_87618_cached_sizze_89104 = 0;
    unsigned char *mem_87618 = NULL;
    int64_t mem_87639_cached_sizze_89105 = 0;
    unsigned char *mem_87639 = NULL;
    int64_t mem_87644_cached_sizze_89106 = 0;
    unsigned char *mem_87644 = NULL;
    int64_t mem_87655_cached_sizze_89107 = 0;
    unsigned char *mem_87655 = NULL;
    int64_t mem_87660_cached_sizze_89108 = 0;
    unsigned char *mem_87660 = NULL;
    int64_t mem_87671_cached_sizze_89109 = 0;
    unsigned char *mem_87671 = NULL;
    int64_t mem_87672_cached_sizze_89110 = 0;
    unsigned char *mem_87672 = NULL;
    int64_t mem_87685_cached_sizze_89111 = 0;
    unsigned char *mem_87685 = NULL;
    int64_t mem_87692_cached_sizze_89112 = 0;
    unsigned char *mem_87692 = NULL;
    int64_t mem_87702_cached_sizze_89113 = 0;
    unsigned char *mem_87702 = NULL;
    int64_t mem_87707_cached_sizze_89114 = 0;
    unsigned char *mem_87707 = NULL;
    int64_t mem_87718_cached_sizze_89115 = 0;
    unsigned char *mem_87718 = NULL;
    int64_t mem_87719_cached_sizze_89116 = 0;
    unsigned char *mem_87719 = NULL;
    int64_t mem_87728_cached_sizze_89117 = 0;
    unsigned char *mem_87728 = NULL;
    int64_t mem_87729_cached_sizze_89118 = 0;
    unsigned char *mem_87729 = NULL;
    int64_t mem_87750_cached_sizze_89119 = 0;
    unsigned char *mem_87750 = NULL;
    int64_t mem_87751_cached_sizze_89120 = 0;
    unsigned char *mem_87751 = NULL;
    int64_t mem_87762_cached_sizze_89121 = 0;
    unsigned char *mem_87762 = NULL;
    int64_t mem_87763_cached_sizze_89122 = 0;
    unsigned char *mem_87763 = NULL;
    int64_t mem_87772_cached_sizze_89123 = 0;
    unsigned char *mem_87772 = NULL;
    int64_t mem_87779_cached_sizze_89124 = 0;
    unsigned char *mem_87779 = NULL;
    int64_t mem_87804_cached_sizze_89125 = 0;
    unsigned char *mem_87804 = NULL;
    int64_t mem_87805_cached_sizze_89126 = 0;
    unsigned char *mem_87805 = NULL;
    int64_t mem_87816_cached_sizze_89127 = 0;
    unsigned char *mem_87816 = NULL;
    int64_t mem_87817_cached_sizze_89128 = 0;
    unsigned char *mem_87817 = NULL;
    int64_t mem_87826_cached_sizze_89129 = 0;
    unsigned char *mem_87826 = NULL;
    int64_t mem_87833_cached_sizze_89130 = 0;
    unsigned char *mem_87833 = NULL;
    int64_t mem_87840_cached_sizze_89131 = 0;
    unsigned char *mem_87840 = NULL;
    int64_t mem_87865_cached_sizze_89132 = 0;
    unsigned char *mem_87865 = NULL;
    int64_t mem_87866_cached_sizze_89133 = 0;
    unsigned char *mem_87866 = NULL;
    int64_t mem_87877_cached_sizze_89134 = 0;
    unsigned char *mem_87877 = NULL;
    int64_t mem_87878_cached_sizze_89135 = 0;
    unsigned char *mem_87878 = NULL;
    int64_t mem_87887_cached_sizze_89136 = 0;
    unsigned char *mem_87887 = NULL;
    int64_t mem_87894_cached_sizze_89137 = 0;
    unsigned char *mem_87894 = NULL;
    int64_t mem_87919_cached_sizze_89138 = 0;
    unsigned char *mem_87919 = NULL;
    int64_t mem_87924_cached_sizze_89139 = 0;
    unsigned char *mem_87924 = NULL;
    int64_t mem_87935_cached_sizze_89140 = 0;
    unsigned char *mem_87935 = NULL;
    int64_t mem_87941_cached_sizze_89141 = 0;
    unsigned char *mem_87941 = NULL;
    int64_t mem_87946_cached_sizze_89142 = 0;
    unsigned char *mem_87946 = NULL;
    int64_t mem_87962_cached_sizze_89143 = 0;
    unsigned char *mem_87962 = NULL;
    int64_t mem_87968_cached_sizze_89144 = 0;
    unsigned char *mem_87968 = NULL;
    int64_t mem_87973_cached_sizze_89145 = 0;
    unsigned char *mem_87973 = NULL;
    int64_t mem_87989_cached_sizze_89146 = 0;
    unsigned char *mem_87989 = NULL;
    int64_t mem_87990_cached_sizze_89147 = 0;
    unsigned char *mem_87990 = NULL;
    int64_t mem_88001_cached_sizze_89148 = 0;
    unsigned char *mem_88001 = NULL;
    int64_t mem_88002_cached_sizze_89149 = 0;
    unsigned char *mem_88002 = NULL;
    int64_t mem_88011_cached_sizze_89150 = 0;
    unsigned char *mem_88011 = NULL;
    int64_t mem_88012_cached_sizze_89151 = 0;
    unsigned char *mem_88012 = NULL;
    int64_t mem_88043_cached_sizze_89152 = 0;
    unsigned char *mem_88043 = NULL;
    int64_t mem_88044_cached_sizze_89153 = 0;
    unsigned char *mem_88044 = NULL;
    int64_t mem_88045_cached_sizze_89154 = 0;
    unsigned char *mem_88045 = NULL;
    int64_t mem_88058_cached_sizze_89155 = 0;
    unsigned char *mem_88058 = NULL;
    int64_t mem_88059_cached_sizze_89156 = 0;
    unsigned char *mem_88059 = NULL;
    int64_t mem_88060_cached_sizze_89157 = 0;
    unsigned char *mem_88060 = NULL;
    int64_t mem_88091_cached_sizze_89158 = 0;
    unsigned char *mem_88091 = NULL;
    int64_t mem_88092_cached_sizze_89159 = 0;
    unsigned char *mem_88092 = NULL;
    int64_t mem_88093_cached_sizze_89160 = 0;
    unsigned char *mem_88093 = NULL;
    int64_t mem_88094_cached_sizze_89161 = 0;
    unsigned char *mem_88094 = NULL;
    int64_t mem_88111_cached_sizze_89162 = 0;
    unsigned char *mem_88111 = NULL;
    int64_t mem_88112_cached_sizze_89163 = 0;
    unsigned char *mem_88112 = NULL;
    int64_t mem_88113_cached_sizze_89164 = 0;
    unsigned char *mem_88113 = NULL;
    int64_t mem_88114_cached_sizze_89165 = 0;
    unsigned char *mem_88114 = NULL;
    int64_t mem_88155_cached_sizze_89166 = 0;
    unsigned char *mem_88155 = NULL;
    int64_t mem_88156_cached_sizze_89167 = 0;
    unsigned char *mem_88156 = NULL;
    int64_t mem_88169_cached_sizze_89168 = 0;
    unsigned char *mem_88169 = NULL;
    int64_t mem_88176_cached_sizze_89169 = 0;
    unsigned char *mem_88176 = NULL;
    int64_t mem_88186_cached_sizze_89170 = 0;
    unsigned char *mem_88186 = NULL;
    int64_t mem_88191_cached_sizze_89171 = 0;
    unsigned char *mem_88191 = NULL;
    int64_t mem_88202_cached_sizze_89172 = 0;
    unsigned char *mem_88202 = NULL;
    int64_t mem_88203_cached_sizze_89173 = 0;
    unsigned char *mem_88203 = NULL;
    int64_t mem_88216_cached_sizze_89174 = 0;
    unsigned char *mem_88216 = NULL;
    int64_t mem_88223_cached_sizze_89175 = 0;
    unsigned char *mem_88223 = NULL;
    int64_t mem_88233_cached_sizze_89176 = 0;
    unsigned char *mem_88233 = NULL;
    int64_t mem_88238_cached_sizze_89177 = 0;
    unsigned char *mem_88238 = NULL;
    int64_t mem_88249_cached_sizze_89178 = 0;
    unsigned char *mem_88249 = NULL;
    int64_t mem_88250_cached_sizze_89179 = 0;
    unsigned char *mem_88250 = NULL;
    int64_t mem_88259_cached_sizze_89180 = 0;
    unsigned char *mem_88259 = NULL;
    int64_t mem_88260_cached_sizze_89181 = 0;
    unsigned char *mem_88260 = NULL;
    int64_t mem_88281_cached_sizze_89182 = 0;
    unsigned char *mem_88281 = NULL;
    int64_t mem_88286_cached_sizze_89183 = 0;
    unsigned char *mem_88286 = NULL;
    int64_t mem_88297_cached_sizze_89184 = 0;
    unsigned char *mem_88297 = NULL;
    int64_t mem_88298_cached_sizze_89185 = 0;
    unsigned char *mem_88298 = NULL;
    int64_t mem_88307_cached_sizze_89186 = 0;
    unsigned char *mem_88307 = NULL;
    int64_t mem_88308_cached_sizze_89187 = 0;
    unsigned char *mem_88308 = NULL;
    struct memblock mem_param_tmp_88661;
    
    mem_param_tmp_88661.references = NULL;
    
    struct memblock mem_param_tmp_88660;
    
    mem_param_tmp_88660.references = NULL;
    
    struct memblock mem_param_tmp_88659;
    
    mem_param_tmp_88659.references = NULL;
    
    struct memblock mem_param_tmp_88658;
    
    mem_param_tmp_88658.references = NULL;
    
    struct memblock mem_param_tmp_88657;
    
    mem_param_tmp_88657.references = NULL;
    
    struct memblock mem_param_tmp_88656;
    
    mem_param_tmp_88656.references = NULL;
    
    struct memblock mem_param_tmp_88655;
    
    mem_param_tmp_88655.references = NULL;
    
    struct memblock mem_param_tmp_88654;
    
    mem_param_tmp_88654.references = NULL;
    
    struct memblock mem_param_tmp_88653;
    
    mem_param_tmp_88653.references = NULL;
    
    struct memblock mem_param_tmp_88652;
    
    mem_param_tmp_88652.references = NULL;
    
    struct memblock mem_param_tmp_88651;
    
    mem_param_tmp_88651.references = NULL;
    
    struct memblock mem_param_tmp_88650;
    
    mem_param_tmp_88650.references = NULL;
    
    struct memblock mem_param_tmp_88649;
    
    mem_param_tmp_88649.references = NULL;
    
    struct memblock mem_param_tmp_88648;
    
    mem_param_tmp_88648.references = NULL;
    
    struct memblock mem_param_tmp_88647;
    
    mem_param_tmp_88647.references = NULL;
    
    struct memblock mem_param_tmp_88646;
    
    mem_param_tmp_88646.references = NULL;
    
    struct memblock mem_param_tmp_88645;
    
    mem_param_tmp_88645.references = NULL;
    
    struct memblock mem_param_tmp_88644;
    
    mem_param_tmp_88644.references = NULL;
    
    struct memblock mem_param_tmp_88643;
    
    mem_param_tmp_88643.references = NULL;
    
    struct memblock mem_param_tmp_88642;
    
    mem_param_tmp_88642.references = NULL;
    
    struct memblock mem_param_tmp_88641;
    
    mem_param_tmp_88641.references = NULL;
    
    struct memblock mem_param_tmp_88640;
    
    mem_param_tmp_88640.references = NULL;
    
    struct memblock mem_param_tmp_88639;
    
    mem_param_tmp_88639.references = NULL;
    
    struct memblock mem_param_tmp_88638;
    
    mem_param_tmp_88638.references = NULL;
    
    struct memblock mem_param_tmp_88637;
    
    mem_param_tmp_88637.references = NULL;
    
    struct memblock mem_param_tmp_88636;
    
    mem_param_tmp_88636.references = NULL;
    
    struct memblock mem_param_tmp_88635;
    
    mem_param_tmp_88635.references = NULL;
    
    struct memblock ext_mem_88425;
    
    ext_mem_88425.references = NULL;
    
    struct memblock ext_mem_88426;
    
    ext_mem_88426.references = NULL;
    
    struct memblock ext_mem_88427;
    
    ext_mem_88427.references = NULL;
    
    struct memblock mem_88423;
    
    mem_88423.references = NULL;
    
    struct memblock mem_88421;
    
    mem_88421.references = NULL;
    
    struct memblock mem_88419;
    
    mem_88419.references = NULL;
    
    struct memblock mem_88417;
    
    mem_88417.references = NULL;
    
    struct memblock ext_mem_88414;
    
    ext_mem_88414.references = NULL;
    
    struct memblock ext_mem_88415;
    
    ext_mem_88415.references = NULL;
    
    struct memblock ext_mem_88416;
    
    ext_mem_88416.references = NULL;
    
    struct memblock mem_88412;
    
    mem_88412.references = NULL;
    
    struct memblock mem_88410;
    
    mem_88410.references = NULL;
    
    struct memblock mem_88408;
    
    mem_88408.references = NULL;
    
    struct memblock mem_88406;
    
    mem_88406.references = NULL;
    
    struct memblock ext_mem_88403;
    
    ext_mem_88403.references = NULL;
    
    struct memblock ext_mem_88404;
    
    ext_mem_88404.references = NULL;
    
    struct memblock ext_mem_88405;
    
    ext_mem_88405.references = NULL;
    
    struct memblock mem_88401;
    
    mem_88401.references = NULL;
    
    struct memblock mem_88399;
    
    mem_88399.references = NULL;
    
    struct memblock mem_88397;
    
    mem_88397.references = NULL;
    
    struct memblock mem_88395;
    
    mem_88395.references = NULL;
    
    struct memblock ext_mem_88392;
    
    ext_mem_88392.references = NULL;
    
    struct memblock ext_mem_88393;
    
    ext_mem_88393.references = NULL;
    
    struct memblock ext_mem_88394;
    
    ext_mem_88394.references = NULL;
    
    struct memblock mem_88390;
    
    mem_88390.references = NULL;
    
    struct memblock mem_88388;
    
    mem_88388.references = NULL;
    
    struct memblock mem_88386;
    
    mem_88386.references = NULL;
    
    struct memblock mem_88384;
    
    mem_88384.references = NULL;
    
    struct memblock ext_mem_88381;
    
    ext_mem_88381.references = NULL;
    
    struct memblock ext_mem_88382;
    
    ext_mem_88382.references = NULL;
    
    struct memblock ext_mem_88383;
    
    ext_mem_88383.references = NULL;
    
    struct memblock mem_88379;
    
    mem_88379.references = NULL;
    
    struct memblock mem_88377;
    
    mem_88377.references = NULL;
    
    struct memblock mem_88375;
    
    mem_88375.references = NULL;
    
    struct memblock mem_88373;
    
    mem_88373.references = NULL;
    
    struct memblock ext_mem_88370;
    
    ext_mem_88370.references = NULL;
    
    struct memblock ext_mem_88371;
    
    ext_mem_88371.references = NULL;
    
    struct memblock ext_mem_88372;
    
    ext_mem_88372.references = NULL;
    
    struct memblock mem_88368;
    
    mem_88368.references = NULL;
    
    struct memblock mem_88366;
    
    mem_88366.references = NULL;
    
    struct memblock mem_88364;
    
    mem_88364.references = NULL;
    
    struct memblock mem_88362;
    
    mem_88362.references = NULL;
    
    struct memblock ext_mem_88359;
    
    ext_mem_88359.references = NULL;
    
    struct memblock ext_mem_88360;
    
    ext_mem_88360.references = NULL;
    
    struct memblock ext_mem_88361;
    
    ext_mem_88361.references = NULL;
    
    struct memblock mem_88357;
    
    mem_88357.references = NULL;
    
    struct memblock mem_88355;
    
    mem_88355.references = NULL;
    
    struct memblock mem_88353;
    
    mem_88353.references = NULL;
    
    struct memblock mem_88351;
    
    mem_88351.references = NULL;
    
    struct memblock ext_mem_88348;
    
    ext_mem_88348.references = NULL;
    
    struct memblock ext_mem_88349;
    
    ext_mem_88349.references = NULL;
    
    struct memblock ext_mem_88350;
    
    ext_mem_88350.references = NULL;
    
    struct memblock mem_88346;
    
    mem_88346.references = NULL;
    
    struct memblock mem_88344;
    
    mem_88344.references = NULL;
    
    struct memblock mem_88342;
    
    mem_88342.references = NULL;
    
    struct memblock mem_88340;
    
    mem_88340.references = NULL;
    
    struct memblock ext_mem_88337;
    
    ext_mem_88337.references = NULL;
    
    struct memblock ext_mem_88338;
    
    ext_mem_88338.references = NULL;
    
    struct memblock ext_mem_88339;
    
    ext_mem_88339.references = NULL;
    
    struct memblock mem_88335;
    
    mem_88335.references = NULL;
    
    struct memblock mem_88333;
    
    mem_88333.references = NULL;
    
    struct memblock mem_88331;
    
    mem_88331.references = NULL;
    
    struct memblock mem_88329;
    
    mem_88329.references = NULL;
    
    struct memblock mem_param_86974;
    
    mem_param_86974.references = NULL;
    
    struct memblock mem_param_86970;
    
    mem_param_86970.references = NULL;
    
    struct memblock mem_param_86966;
    
    mem_param_86966.references = NULL;
    
    struct memblock mem_param_86962;
    
    mem_param_86962.references = NULL;
    
    struct memblock mem_param_86958;
    
    mem_param_86958.references = NULL;
    
    struct memblock mem_param_86954;
    
    mem_param_86954.references = NULL;
    
    struct memblock mem_param_86950;
    
    mem_param_86950.references = NULL;
    
    struct memblock mem_param_86946;
    
    mem_param_86946.references = NULL;
    
    struct memblock mem_param_86942;
    
    mem_param_86942.references = NULL;
    
    struct memblock mem_param_86938;
    
    mem_param_86938.references = NULL;
    
    struct memblock mem_param_86934;
    
    mem_param_86934.references = NULL;
    
    struct memblock mem_param_86930;
    
    mem_param_86930.references = NULL;
    
    struct memblock mem_param_86926;
    
    mem_param_86926.references = NULL;
    
    struct memblock mem_param_86922;
    
    mem_param_86922.references = NULL;
    
    struct memblock mem_param_86918;
    
    mem_param_86918.references = NULL;
    
    struct memblock mem_param_86914;
    
    mem_param_86914.references = NULL;
    
    struct memblock mem_param_86910;
    
    mem_param_86910.references = NULL;
    
    struct memblock mem_param_86906;
    
    mem_param_86906.references = NULL;
    
    struct memblock mem_param_86902;
    
    mem_param_86902.references = NULL;
    
    struct memblock mem_param_86898;
    
    mem_param_86898.references = NULL;
    
    struct memblock mem_param_86894;
    
    mem_param_86894.references = NULL;
    
    struct memblock mem_param_86890;
    
    mem_param_86890.references = NULL;
    
    struct memblock mem_param_86886;
    
    mem_param_86886.references = NULL;
    
    struct memblock mem_param_86882;
    
    mem_param_86882.references = NULL;
    
    struct memblock mem_param_86878;
    
    mem_param_86878.references = NULL;
    
    struct memblock mem_param_86874;
    
    mem_param_86874.references = NULL;
    
    struct memblock mem_param_86870;
    
    mem_param_86870.references = NULL;
    
    struct memblock ext_mem_88509;
    
    ext_mem_88509.references = NULL;
    
    struct memblock ext_mem_88510;
    
    ext_mem_88510.references = NULL;
    
    struct memblock ext_mem_88511;
    
    ext_mem_88511.references = NULL;
    
    struct memblock ext_mem_88512;
    
    ext_mem_88512.references = NULL;
    
    struct memblock ext_mem_88513;
    
    ext_mem_88513.references = NULL;
    
    struct memblock ext_mem_88514;
    
    ext_mem_88514.references = NULL;
    
    struct memblock ext_mem_88515;
    
    ext_mem_88515.references = NULL;
    
    struct memblock ext_mem_88516;
    
    ext_mem_88516.references = NULL;
    
    struct memblock ext_mem_88517;
    
    ext_mem_88517.references = NULL;
    
    struct memblock ext_mem_88518;
    
    ext_mem_88518.references = NULL;
    
    struct memblock ext_mem_88519;
    
    ext_mem_88519.references = NULL;
    
    struct memblock ext_mem_88520;
    
    ext_mem_88520.references = NULL;
    
    struct memblock ext_mem_88521;
    
    ext_mem_88521.references = NULL;
    
    struct memblock ext_mem_88522;
    
    ext_mem_88522.references = NULL;
    
    struct memblock ext_mem_88523;
    
    ext_mem_88523.references = NULL;
    
    struct memblock ext_mem_88524;
    
    ext_mem_88524.references = NULL;
    
    struct memblock ext_mem_88525;
    
    ext_mem_88525.references = NULL;
    
    struct memblock ext_mem_88526;
    
    ext_mem_88526.references = NULL;
    
    struct memblock ext_mem_88527;
    
    ext_mem_88527.references = NULL;
    
    struct memblock ext_mem_88528;
    
    ext_mem_88528.references = NULL;
    
    struct memblock ext_mem_88529;
    
    ext_mem_88529.references = NULL;
    
    struct memblock ext_mem_88530;
    
    ext_mem_88530.references = NULL;
    
    struct memblock ext_mem_88531;
    
    ext_mem_88531.references = NULL;
    
    struct memblock ext_mem_88532;
    
    ext_mem_88532.references = NULL;
    
    struct memblock ext_mem_88533;
    
    ext_mem_88533.references = NULL;
    
    struct memblock ext_mem_88534;
    
    ext_mem_88534.references = NULL;
    
    struct memblock ext_mem_88535;
    
    ext_mem_88535.references = NULL;
    
    struct memblock mem_out_88634;
    
    mem_out_88634.references = NULL;
    
    struct memblock mem_out_88633;
    
    mem_out_88633.references = NULL;
    
    struct memblock mem_out_88632;
    
    mem_out_88632.references = NULL;
    
    struct memblock mem_out_88631;
    
    mem_out_88631.references = NULL;
    
    struct memblock mem_out_88630;
    
    mem_out_88630.references = NULL;
    
    struct memblock mem_out_88629;
    
    mem_out_88629.references = NULL;
    
    struct memblock mem_out_88628;
    
    mem_out_88628.references = NULL;
    
    struct memblock mem_out_88627;
    
    mem_out_88627.references = NULL;
    
    struct memblock mem_out_88626;
    
    mem_out_88626.references = NULL;
    
    struct memblock mem_out_88625;
    
    mem_out_88625.references = NULL;
    
    struct memblock mem_out_88624;
    
    mem_out_88624.references = NULL;
    
    struct memblock mem_out_88623;
    
    mem_out_88623.references = NULL;
    
    struct memblock mem_out_88622;
    
    mem_out_88622.references = NULL;
    
    struct memblock mem_out_88621;
    
    mem_out_88621.references = NULL;
    
    struct memblock mem_out_88620;
    
    mem_out_88620.references = NULL;
    
    struct memblock mem_out_88619;
    
    mem_out_88619.references = NULL;
    
    struct memblock mem_out_88618;
    
    mem_out_88618.references = NULL;
    
    struct memblock mem_out_88617;
    
    mem_out_88617.references = NULL;
    
    struct memblock mem_out_88616;
    
    mem_out_88616.references = NULL;
    
    struct memblock mem_out_88615;
    
    mem_out_88615.references = NULL;
    
    struct memblock mem_out_88614;
    
    mem_out_88614.references = NULL;
    
    struct memblock mem_out_88613;
    
    mem_out_88613.references = NULL;
    
    struct memblock mem_out_88612;
    
    mem_out_88612.references = NULL;
    
    struct memblock mem_out_88611;
    
    mem_out_88611.references = NULL;
    
    struct memblock mem_out_88610;
    
    mem_out_88610.references = NULL;
    
    struct memblock mem_out_88609;
    
    mem_out_88609.references = NULL;
    
    struct memblock mem_out_88608;
    
    mem_out_88608.references = NULL;
    
    struct memblock mem_86828 = ctx->constants->mem_86828;
    struct memblock mem_86829 = ctx->constants->mem_86829;
    struct memblock mem_86830 = ctx->constants->mem_86830;
    struct memblock mem_86831 = ctx->constants->mem_86831;
    struct memblock mem_86832 = ctx->constants->mem_86832;
    struct memblock mem_86833 = ctx->constants->mem_86833;
    struct memblock mem_86834 = ctx->constants->mem_86834;
    struct memblock mem_86835 = ctx->constants->mem_86835;
    struct memblock mem_86836 = ctx->constants->mem_86836;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_86975_cached_sizze_89023 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_86975, &mem_86975_cached_sizze_89023, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86976_cached_sizze_89024 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_86976, &mem_86976_cached_sizze_89024, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86985_cached_sizze_89025 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_86985, &mem_86985_cached_sizze_89025, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_86992_cached_sizze_89026 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_86992, &mem_86992_cached_sizze_89026, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87007_cached_sizze_89027 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_87007, &mem_87007_cached_sizze_89027, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87008_cached_sizze_89028 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87008, &mem_87008_cached_sizze_89028, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87017_cached_sizze_89029 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87017, &mem_87017_cached_sizze_89029, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87024_cached_sizze_89030 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_87024, &mem_87024_cached_sizze_89030, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87039_cached_sizze_89031 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87039, &mem_87039_cached_sizze_89031, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87040_cached_sizze_89032 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87040, &mem_87040_cached_sizze_89032, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87049_cached_sizze_89033 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87049, &mem_87049_cached_sizze_89033, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87050_cached_sizze_89034 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87050, &mem_87050_cached_sizze_89034, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87071_cached_sizze_89035 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87071, &mem_87071_cached_sizze_89035, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87072_cached_sizze_89036 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87072, &mem_87072_cached_sizze_89036, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87073_cached_sizze_89037 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87073, &mem_87073_cached_sizze_89037, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87085_cached_sizze_89038 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87085, &mem_87085_cached_sizze_89038, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87086_cached_sizze_89039 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87086, &mem_87086_cached_sizze_89039, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87110_cached_sizze_89040 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87110, &mem_87110_cached_sizze_89040, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87111_cached_sizze_89041 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87111, &mem_87111_cached_sizze_89041, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87112_cached_sizze_89042 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87112, &mem_87112_cached_sizze_89042, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87113_cached_sizze_89043 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87113, &mem_87113_cached_sizze_89043, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87129_cached_sizze_89044 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87129, &mem_87129_cached_sizze_89044, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87130_cached_sizze_89045 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87130, &mem_87130_cached_sizze_89045, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87131_cached_sizze_89046 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87131, &mem_87131_cached_sizze_89046, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87165_cached_sizze_89047 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87165, &mem_87165_cached_sizze_89047, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87166_cached_sizze_89048 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87166, &mem_87166_cached_sizze_89048, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87167_cached_sizze_89049 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87167, &mem_87167_cached_sizze_89049, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87183_cached_sizze_89050 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_87183, &mem_87183_cached_sizze_89050, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87184_cached_sizze_89051 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_87184, &mem_87184_cached_sizze_89051, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87185_cached_sizze_89052 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_87185, &mem_87185_cached_sizze_89052, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87198_cached_sizze_89053 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_87198, &mem_87198_cached_sizze_89053, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87199_cached_sizze_89054 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_87199, &mem_87199_cached_sizze_89054, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87200_cached_sizze_89055 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_87200, &mem_87200_cached_sizze_89055, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87246_cached_sizze_89056 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_87246, &mem_87246_cached_sizze_89056, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87247_cached_sizze_89057 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87247, &mem_87247_cached_sizze_89057, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87258_cached_sizze_89058 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87258, &mem_87258_cached_sizze_89058, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87259_cached_sizze_89059 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87259, &mem_87259_cached_sizze_89059, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87268_cached_sizze_89060 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87268, &mem_87268_cached_sizze_89060, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87269_cached_sizze_89061 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87269, &mem_87269_cached_sizze_89061, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87290_cached_sizze_89062 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87290, &mem_87290_cached_sizze_89062, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87295_cached_sizze_89063 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87295, &mem_87295_cached_sizze_89063, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87306_cached_sizze_89064 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87306, &mem_87306_cached_sizze_89064, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87311_cached_sizze_89065 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87311, &mem_87311_cached_sizze_89065, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87318_cached_sizze_89066 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87318, &mem_87318_cached_sizze_89066, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87329_cached_sizze_89067 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_87329, &mem_87329_cached_sizze_89067, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87334_cached_sizze_89068 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_87334, &mem_87334_cached_sizze_89068, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87355_cached_sizze_89069 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87355, &mem_87355_cached_sizze_89069, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87360_cached_sizze_89070 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87360, &mem_87360_cached_sizze_89070, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87371_cached_sizze_89071 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87371, &mem_87371_cached_sizze_89071, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87376_cached_sizze_89072 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87376, &mem_87376_cached_sizze_89072, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87387_cached_sizze_89073 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87387, &mem_87387_cached_sizze_89073, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87392_cached_sizze_89074 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87392, &mem_87392_cached_sizze_89074, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87403_cached_sizze_89075 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87403, &mem_87403_cached_sizze_89075, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87404_cached_sizze_89076 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87404, &mem_87404_cached_sizze_89076, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87413_cached_sizze_89077 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87413, &mem_87413_cached_sizze_89077, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87414_cached_sizze_89078 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87414, &mem_87414_cached_sizze_89078, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87435_cached_sizze_89079 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87435, &mem_87435_cached_sizze_89079, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87436_cached_sizze_89080 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_87436, &mem_87436_cached_sizze_89080, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87444_cached_sizze_89081 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_87444, &mem_87444_cached_sizze_89081, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87458_cached_sizze_89082 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_87458, &mem_87458_cached_sizze_89082, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87463_cached_sizze_89083 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_87463, &mem_87463_cached_sizze_89083, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87474_cached_sizze_89084 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87474, &mem_87474_cached_sizze_89084, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87479_cached_sizze_89085 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87479, &mem_87479_cached_sizze_89085, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87490_cached_sizze_89086 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87490, &mem_87490_cached_sizze_89086, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87495_cached_sizze_89087 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87495, &mem_87495_cached_sizze_89087, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87506_cached_sizze_89088 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_87506, &mem_87506_cached_sizze_89088, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87511_cached_sizze_89089 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_87511, &mem_87511_cached_sizze_89089, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87522_cached_sizze_89090 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_87522, &mem_87522_cached_sizze_89090, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87523_cached_sizze_89091 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_87523, &mem_87523_cached_sizze_89091, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87532_cached_sizze_89092 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_87532, &mem_87532_cached_sizze_89092, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87533_cached_sizze_89093 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_87533, &mem_87533_cached_sizze_89093, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87546_cached_sizze_89094 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_87546, &mem_87546_cached_sizze_89094, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87547_cached_sizze_89095 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_87547, &mem_87547_cached_sizze_89095, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87568_cached_sizze_89096 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87568, &mem_87568_cached_sizze_89096, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87575_cached_sizze_89097 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_87575, &mem_87575_cached_sizze_89097, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87580_cached_sizze_89098 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_87580, &mem_87580_cached_sizze_89098, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87591_cached_sizze_89099 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87591, &mem_87591_cached_sizze_89099, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87596_cached_sizze_89100 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87596, &mem_87596_cached_sizze_89100, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87607_cached_sizze_89101 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_87607, &mem_87607_cached_sizze_89101, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87608_cached_sizze_89102 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_87608, &mem_87608_cached_sizze_89102, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87617_cached_sizze_89103 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_87617, &mem_87617_cached_sizze_89103, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87618_cached_sizze_89104 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_87618, &mem_87618_cached_sizze_89104, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87639_cached_sizze_89105 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_87639, &mem_87639_cached_sizze_89105, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87644_cached_sizze_89106 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_87644, &mem_87644_cached_sizze_89106, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87655_cached_sizze_89107 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87655, &mem_87655_cached_sizze_89107, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87660_cached_sizze_89108 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87660, &mem_87660_cached_sizze_89108, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87671_cached_sizze_89109 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87671, &mem_87671_cached_sizze_89109, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87672_cached_sizze_89110 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87672, &mem_87672_cached_sizze_89110, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87685_cached_sizze_89111 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87685, &mem_87685_cached_sizze_89111, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87692_cached_sizze_89112 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87692, &mem_87692_cached_sizze_89112, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87702_cached_sizze_89113 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87702, &mem_87702_cached_sizze_89113, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87707_cached_sizze_89114 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87707, &mem_87707_cached_sizze_89114, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87718_cached_sizze_89115 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87718, &mem_87718_cached_sizze_89115, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87719_cached_sizze_89116 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87719, &mem_87719_cached_sizze_89116, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87728_cached_sizze_89117 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87728, &mem_87728_cached_sizze_89117, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87729_cached_sizze_89118 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87729, &mem_87729_cached_sizze_89118, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87750_cached_sizze_89119 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_87750, &mem_87750_cached_sizze_89119, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87751_cached_sizze_89120 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87751, &mem_87751_cached_sizze_89120, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87762_cached_sizze_89121 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87762, &mem_87762_cached_sizze_89121, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87763_cached_sizze_89122 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_87763, &mem_87763_cached_sizze_89122, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87772_cached_sizze_89123 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_87772, &mem_87772_cached_sizze_89123, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87779_cached_sizze_89124 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87779, &mem_87779_cached_sizze_89124, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87804_cached_sizze_89125 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_87804, &mem_87804_cached_sizze_89125, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87805_cached_sizze_89126 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_87805, &mem_87805_cached_sizze_89126, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87816_cached_sizze_89127 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87816, &mem_87816_cached_sizze_89127, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87817_cached_sizze_89128 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87817, &mem_87817_cached_sizze_89128, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87826_cached_sizze_89129 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87826, &mem_87826_cached_sizze_89129, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87833_cached_sizze_89130 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87833, &mem_87833_cached_sizze_89130, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87840_cached_sizze_89131 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87840, &mem_87840_cached_sizze_89131, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87865_cached_sizze_89132 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87865, &mem_87865_cached_sizze_89132, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87866_cached_sizze_89133 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_87866, &mem_87866_cached_sizze_89133, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87877_cached_sizze_89134 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_87877, &mem_87877_cached_sizze_89134, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87878_cached_sizze_89135 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87878, &mem_87878_cached_sizze_89135, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87887_cached_sizze_89136 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87887, &mem_87887_cached_sizze_89136, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87894_cached_sizze_89137 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_87894, &mem_87894_cached_sizze_89137, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87919_cached_sizze_89138 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_87919, &mem_87919_cached_sizze_89138, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87924_cached_sizze_89139 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87924, &mem_87924_cached_sizze_89139, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87935_cached_sizze_89140 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_87935, &mem_87935_cached_sizze_89140, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87941_cached_sizze_89141 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87941, &mem_87941_cached_sizze_89141, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87946_cached_sizze_89142 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87946, &mem_87946_cached_sizze_89142, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87962_cached_sizze_89143 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_87962, &mem_87962_cached_sizze_89143, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87968_cached_sizze_89144 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87968, &mem_87968_cached_sizze_89144, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87973_cached_sizze_89145 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87973, &mem_87973_cached_sizze_89145, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87989_cached_sizze_89146 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87989, &mem_87989_cached_sizze_89146, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87990_cached_sizze_89147 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87990, &mem_87990_cached_sizze_89147, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88001_cached_sizze_89148 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88001, &mem_88001_cached_sizze_89148, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88002_cached_sizze_89149 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88002, &mem_88002_cached_sizze_89149, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88011_cached_sizze_89150 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_88011, &mem_88011_cached_sizze_89150, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88012_cached_sizze_89151 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_88012, &mem_88012_cached_sizze_89151, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88043_cached_sizze_89152 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88043, &mem_88043_cached_sizze_89152, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88044_cached_sizze_89153 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88044, &mem_88044_cached_sizze_89153, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88045_cached_sizze_89154 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88045, &mem_88045_cached_sizze_89154, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88058_cached_sizze_89155 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88058, &mem_88058_cached_sizze_89155, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88059_cached_sizze_89156 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88059, &mem_88059_cached_sizze_89156, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88060_cached_sizze_89157 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88060, &mem_88060_cached_sizze_89157, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88091_cached_sizze_89158 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88091, &mem_88091_cached_sizze_89158, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88092_cached_sizze_89159 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88092, &mem_88092_cached_sizze_89159, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88093_cached_sizze_89160 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88093, &mem_88093_cached_sizze_89160, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88094_cached_sizze_89161 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88094, &mem_88094_cached_sizze_89161, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88111_cached_sizze_89162 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88111, &mem_88111_cached_sizze_89162, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88112_cached_sizze_89163 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88112, &mem_88112_cached_sizze_89163, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88113_cached_sizze_89164 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88113, &mem_88113_cached_sizze_89164, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88114_cached_sizze_89165 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88114, &mem_88114_cached_sizze_89165, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88155_cached_sizze_89166 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88155, &mem_88155_cached_sizze_89166, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88156_cached_sizze_89167 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88156, &mem_88156_cached_sizze_89167, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88169_cached_sizze_89168 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88169, &mem_88169_cached_sizze_89168, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88176_cached_sizze_89169 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88176, &mem_88176_cached_sizze_89169, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88186_cached_sizze_89170 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88186, &mem_88186_cached_sizze_89170, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88191_cached_sizze_89171 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88191, &mem_88191_cached_sizze_89171, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88202_cached_sizze_89172 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88202, &mem_88202_cached_sizze_89172, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88203_cached_sizze_89173 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88203, &mem_88203_cached_sizze_89173, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88216_cached_sizze_89174 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88216, &mem_88216_cached_sizze_89174, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88223_cached_sizze_89175 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88223, &mem_88223_cached_sizze_89175, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88233_cached_sizze_89176 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88233, &mem_88233_cached_sizze_89176, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88238_cached_sizze_89177 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88238, &mem_88238_cached_sizze_89177, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88249_cached_sizze_89178 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88249, &mem_88249_cached_sizze_89178, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88250_cached_sizze_89179 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88250, &mem_88250_cached_sizze_89179, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88259_cached_sizze_89180 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88259, &mem_88259_cached_sizze_89180, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88260_cached_sizze_89181 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88260, &mem_88260_cached_sizze_89181, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88281_cached_sizze_89182 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_88281, &mem_88281_cached_sizze_89182, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88286_cached_sizze_89183 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88286, &mem_88286_cached_sizze_89183, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88297_cached_sizze_89184 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_88297, &mem_88297_cached_sizze_89184, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88298_cached_sizze_89185 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_88298, &mem_88298_cached_sizze_89185, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88307_cached_sizze_89186 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88307, &mem_88307_cached_sizze_89186, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88308_cached_sizze_89187 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88308, &mem_88308_cached_sizze_89187, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:471:5-476:51
    if (memblock_set(ctx, &mem_param_86870, &wdown_mem_86837, "wdown_mem_86837") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86874, &wkey_mem_86838, "wkey_mem_86838") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86878, &wout_mem_86839, "wout_mem_86839") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86882, &wpe_mem_86840, "wpe_mem_86840") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86886, &wqry_mem_86841, "wqry_mem_86841") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86890, &wte_mem_86842, "wte_mem_86842") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86894, &wup_mem_86843, "wup_mem_86843") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86898, &wval_mem_86844, "wval_mem_86844") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86902, &wvoc_mem_86845, "wvoc_mem_86845") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86906, &wdown_mem_86846, "wdown_mem_86846") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86910, &wkey_mem_86847, "wkey_mem_86847") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86914, &wout_mem_86848, "wout_mem_86848") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86918, &wpe_mem_86849, "wpe_mem_86849") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86922, &wqry_mem_86850, "wqry_mem_86850") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86926, &wte_mem_86851, "wte_mem_86851") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86930, &wup_mem_86852, "wup_mem_86852") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86934, &wval_mem_86853, "wval_mem_86853") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86938, &wvoc_mem_86854, "wvoc_mem_86854") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86942, &wdown_mem_86855, "wdown_mem_86855") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86946, &wkey_mem_86856, "wkey_mem_86856") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86950, &wout_mem_86857, "wout_mem_86857") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86954, &wpe_mem_86858, "wpe_mem_86858") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86958, &wqry_mem_86859, "wqry_mem_86859") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86962, &wte_mem_86860, "wte_mem_86860") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86966, &wup_mem_86861, "wup_mem_86861") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86970, &wval_mem_86862, "wval_mem_86862") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_86974, &wvoc_mem_86863, "wvoc_mem_86863") != 0)
        return 1;
    for (int64_t step_80406 = 0; step_80406 < (int64_t) 500; step_80406++) {
        // futhark/microgpt.fut:473:16-25
        
        int64_t dl_80434 = ((int64_t *) dls_mem_86865.mem)[step_80406];
        
        // futhark/microgpt.fut:386:37-40
        
        int64_t zl_rhs_80439 = sub64(dl_80434, (int64_t) 1);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86011 = 0; i_86011 < (int64_t) 16; i_86011++) {
            // futhark/microgpt.fut:386:25-81
            
            bool cond_82312 = slt64(i_86011, zl_rhs_80439);
            
            // futhark/microgpt.fut:386:56-59
            
            int64_t zeze_lhs_82313 = add64((int64_t) 1, i_86011);
            
            // futhark/microgpt.fut:386:47-60
            
            bool x_82314 = sle64((int64_t) 0, zeze_lhs_82313);
            
            // futhark/microgpt.fut:386:47-60
            
            bool y_82315 = slt64(zeze_lhs_82313, (int64_t) 16);
            
            // futhark/microgpt.fut:386:47-60
            
            bool bounds_check_82316 = x_82314 && y_82315;
            
            // futhark/microgpt.fut:9:27-39
            
            bool loop_not_taken_82317 = !cond_82312;
            
            // futhark/microgpt.fut:9:27-39
            
            bool protect_assert_disj_82318 = bounds_check_82316 || loop_not_taken_82317;
            
            // futhark/microgpt.fut:386:47-60
            
            bool index_certs_82319;
            
            if (!protect_assert_disj_82318) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_82313, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:386:47-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:386:3-83\n   #6  futhark/microgpt.fut:444:18-38\n   #7  futhark/microgpt.fut:454:26-460:31\n   #8  futhark/microgpt.fut:476:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            int64_t tmp_82334 = ((int64_t *) seqs_mem_86866.mem)[step_80406 * (int64_t) 16 + i_86011];
            
            // futhark/microgpt.fut:446:37-51
            
            bool x_82335 = sle64((int64_t) 0, tmp_82334);
            
            // futhark/microgpt.fut:446:37-51
            
            bool y_82336 = slt64(tmp_82334, (int64_t) 27);
            
            // futhark/microgpt.fut:446:37-51
            
            bool bounds_check_82337 = x_82335 && y_82336;
            
            // futhark/microgpt.fut:446:37-51
            
            bool index_certs_82338;
            
            if (!bounds_check_82337) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_82334, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:446:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:446:16-55\n   #6  futhark/microgpt.fut:454:26-460:31\n   #7  futhark/microgpt.fut:476:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:386:47-60
            
            int64_t zeze_lhs_82320;
            
            if (cond_82312) {
                int64_t x_85824 = ((int64_t *) seqs_mem_86866.mem)[step_80406 * (int64_t) 16 + zeze_lhs_82313];
                
                zeze_lhs_82320 = x_85824;
            } else {
                zeze_lhs_82320 = (int64_t) 0;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86001 = 0; i_86001 < (int64_t) 27; i_86001++) {
                // futhark/microgpt.fut:386:61-65
                
                bool cond_t_res_82324 = zeze_lhs_82320 == i_86001;
                
                // futhark/microgpt.fut:9:27-39
                
                bool x_82325 = cond_82312 && cond_t_res_82324;
                
                // futhark/microgpt.fut:386:25-81
                
                double lifted_lambda_res_82326;
                
                if (x_82325) {
                    lifted_lambda_res_82326 = 1.0;
                } else {
                    lifted_lambda_res_82326 = 0.0;
                }
                ((double *) mem_86985)[i_86001] = lifted_lambda_res_82326;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86005 = 0; i_86005 < (int64_t) 16; i_86005++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_82345 = ((double *) mem_param_86890.mem)[tmp_82334 * (int64_t) 16 + i_86005];
                
                ((double *) mem_86992)[i_86005] = lifted_lambda_res_82345;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_86975, i_86011 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_86992, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_86976, i_86011 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_86985, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86026 = 0; i_86026 < (int64_t) 16; i_86026++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86016 = 0; i_86016 < (int64_t) 16; i_86016++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_82370 = ((double *) mem_param_86882.mem)[i_86026 * (int64_t) 16 + i_86016];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_82371 = ((double *) mem_86975)[i_86026 * (int64_t) 16 + i_86016];
                
                // futhark/microgpt.fut:231:39-75
                
                double zp_res_82372 = zp_lhs_82370 + zp_rhs_82371;
                
                ((double *) mem_87017)[i_86016] = zp_res_82372;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86020 = 0; i_86020 < (int64_t) 27; i_86020++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_82386 = ((double *) mem_86976)[i_86026 * (int64_t) 27 + i_86020];
                
                // futhark/microgpt.fut:267:43-85
                
                double zt_res_82387 = -6.25e-2 * zt_rhs_82386;
                
                ((double *) mem_87024)[i_86020] = zt_res_82387;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87007, i_86026 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87024, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87008, i_86026 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87017, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86040 = 0; i_86040 < (int64_t) 16; i_86040++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_82406;
            double r_82408 = 0.0;
            
            for (int64_t i_82407 = 0; i_82407 < (int64_t) 16; i_82407++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_82409 = ((double *) mem_87008)[i_86040 * (int64_t) 16 + i_82407];
                
                // futhark/microgpt.fut:232:70-103
                
                double zt_res_82410 = zt_lhs_82409 * zt_lhs_82409;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_82411 = r_82408 + zt_res_82410;
                double r_tmp_88699 = zp_res_82411;
                
                r_82408 = r_tmp_88699;
            }
            defunc_0_lifted_lambda_res_82406 = r_82408;
            // futhark/microgpt.fut:232:50-121
            
            double zs_res_82412 = defunc_0_lifted_lambda_res_82406 / 16.0;
            
            // futhark/microgpt.fut:233:23-53
            
            double zp_res_82413 = 1.0e-5 + zs_res_82412;
            
            // futhark/microgpt.fut:233:15-53
            
            double sqrt_res_82414 = futrts_sqrt64(zp_res_82413);
            
            // futhark/microgpt.fut:234:25-35
            
            double zs_res_82415 = 1.0 / sqrt_res_82414;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86033 = 0; i_86033 < (int64_t) 16; i_86033++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_84176 = ((double *) mem_87008)[i_86040 * (int64_t) 16 + i_86033];
                
                // futhark/microgpt.fut:234:5-35
                
                double zt_res_84177 = zs_res_82415 * zt_lhs_84176;
                
                // futhark/microgpt.fut:316:45-86
                
                double zt_res_84185 = zt_lhs_84176 * zt_lhs_84176;
                
                ((double *) mem_87049)[i_86033] = zt_res_84185;
                ((double *) mem_87050)[i_86033] = zt_res_84177;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87039, i_86040 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87049, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87040, i_86040 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87050, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86056 = 0; i_86056 < (int64_t) 16; i_86056++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_82514;
            double r_82516 = 0.0;
            
            for (int64_t i_82515 = 0; i_82515 < (int64_t) 16; i_82515++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_82517 = ((double *) mem_87040)[i_86056 * (int64_t) 16 + i_82515];
                
                // futhark/microgpt.fut:235:71-106
                
                double zt_res_82518 = zt_lhs_82517 * zt_lhs_82517;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_82519 = r_82516 + zt_res_82518;
                double r_tmp_88705 = zp_res_82519;
                
                r_82516 = r_tmp_88705;
            }
            defunc_0_lifted_lambda_res_82514 = r_82516;
            // futhark/microgpt.fut:235:50-124
            
            double zs_res_82520 = defunc_0_lifted_lambda_res_82514 / 16.0;
            
            // futhark/microgpt.fut:236:24-54
            
            double zp_res_82521 = 1.0e-5 + zs_res_82520;
            
            // futhark/microgpt.fut:236:16-54
            
            double sqrt_res_82522 = futrts_sqrt64(zp_res_82521);
            
            // futhark/microgpt.fut:237:25-36
            
            double zs_res_82523 = 1.0 / sqrt_res_82522;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86047 = 0; i_86047 < (int64_t) 16; i_86047++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_84205 = ((double *) mem_87040)[i_86056 * (int64_t) 16 + i_86047];
                
                // futhark/microgpt.fut:237:5-36
                
                double zt_res_84206 = zs_res_82523 * zt_lhs_84205;
                
                // futhark/microgpt.fut:309:45-86
                
                double zt_res_84214 = zt_lhs_84205 * zt_lhs_84205;
                
                ((double *) mem_87085)[i_86047] = zt_res_84214;
                ((double *) mem_87086)[i_86047] = zt_res_84206;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_82557;
            double r_82559 = 0.0;
            
            for (int64_t i_82558 = 0; i_82558 < (int64_t) 16; i_82558++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_82560 = ((double *) mem_87039)[i_86056 * (int64_t) 16 + i_82558];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_82561 = r_82559 + lifted_lambda_res_82560;
                double r_tmp_88708 = zp_res_82561;
                
                r_82559 = r_tmp_88708;
            }
            defunc_0_lifted_lambda_res_82557 = r_82559;
            // futhark/microgpt.fut:317:36-94
            
            double zs_res_82562 = defunc_0_lifted_lambda_res_82557 / 16.0;
            
            ((double *) mem_87071)[i_86056] = zs_res_82562;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87072, i_86056 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87085, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87073, i_86056 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87086, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86078 = 0; i_86078 < (int64_t) 16; i_86078++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86066 = 0; i_86066 < (int64_t) 16; i_86066++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_84277;
                double r_84279 = 0.0;
                
                for (int64_t i_84278 = 0; i_84278 < (int64_t) 16; i_84278++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_84280 = ((double *) mem_param_86886.mem)[i_86066 * (int64_t) 16 + i_84278];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_84281 = ((double *) mem_87073)[i_86078 * (int64_t) 16 + i_84278];
                    
                    // futhark/microgpt.fut:238:63-102
                    
                    double zt_res_84282 = zt_lhs_84280 * zt_rhs_84281;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_84283 = r_84279 + zt_res_84282;
                    double r_tmp_88716 = zp_res_84283;
                    
                    r_84279 = r_tmp_88716;
                }
                defunc_0_lifted_lambda_res_84277 = r_84279;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_84290;
                double r_84292 = 0.0;
                
                for (int64_t i_84291 = 0; i_84291 < (int64_t) 16; i_84291++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_84293 = ((double *) mem_param_86874.mem)[i_86066 * (int64_t) 16 + i_84291];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_84294 = ((double *) mem_87073)[i_86078 * (int64_t) 16 + i_84291];
                    
                    // futhark/microgpt.fut:239:63-102
                    
                    double zt_res_84295 = zt_lhs_84293 * zt_rhs_84294;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_84296 = r_84292 + zt_res_84295;
                    double r_tmp_88717 = zp_res_84296;
                    
                    r_84292 = r_tmp_88717;
                }
                defunc_0_lifted_lambda_res_84290 = r_84292;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_84306;
                double r_84308 = 0.0;
                
                for (int64_t i_84307 = 0; i_84307 < (int64_t) 16; i_84307++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_84309 = ((double *) mem_param_86898.mem)[i_86066 * (int64_t) 16 + i_84307];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_84310 = ((double *) mem_87073)[i_86078 * (int64_t) 16 + i_84307];
                    
                    // futhark/microgpt.fut:240:63-102
                    
                    double zt_res_84311 = zt_lhs_84309 * zt_rhs_84310;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_84312 = r_84308 + zt_res_84311;
                    double r_tmp_88718 = zp_res_84312;
                    
                    r_84308 = r_tmp_88718;
                }
                defunc_0_lifted_lambda_res_84306 = r_84308;
                ((double *) mem_87129)[i_86066] = defunc_0_lifted_lambda_res_84306;
                ((double *) mem_87130)[i_86066] = defunc_0_lifted_lambda_res_84290;
                ((double *) mem_87131)[i_86066] = defunc_0_lifted_lambda_res_84277;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_82793;
            double r_82795 = 0.0;
            
            for (int64_t i_82794 = 0; i_82794 < (int64_t) 16; i_82794++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_82796 = ((double *) mem_87072)[i_86078 * (int64_t) 16 + i_82794];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_82797 = r_82795 + lifted_lambda_res_82796;
                double r_tmp_88719 = zp_res_82797;
                
                r_82795 = r_tmp_88719;
            }
            defunc_0_lifted_lambda_res_82793 = r_82795;
            // futhark/microgpt.fut:310:36-94
            
            double zs_res_82798 = defunc_0_lifted_lambda_res_82793 / 16.0;
            
            ((double *) mem_87110)[i_86078] = zs_res_82798;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87111, i_86078 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87129, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87112, i_86078 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87130, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87113, i_86078 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87131, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86109 = 0; i_86109 < (int64_t) 4; i_86109++) {
            // futhark/microgpt.fut:241:67-70
            
            int64_t zp_lhs_82869 = mul64((int64_t) 4, i_86109);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86099 = 0; i_86099 < (int64_t) 16; i_86099++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_86089 = 0; i_86089 < (int64_t) 4; i_86089++) {
                    // futhark/microgpt.fut:241:72-79
                    
                    int64_t tmp_84470 = add64(zp_lhs_82869, i_86089);
                    
                    // futhark/microgpt.fut:241:48-81
                    
                    bool x_84471 = sle64((int64_t) 0, tmp_84470);
                    
                    // futhark/microgpt.fut:241:48-81
                    
                    bool y_84472 = slt64(tmp_84470, (int64_t) 16);
                    
                    // futhark/microgpt.fut:241:48-81
                    
                    bool bounds_check_84473 = x_84471 && y_84472;
                    
                    // futhark/microgpt.fut:241:48-81
                    
                    bool index_certs_84474;
                    
                    if (!bounds_check_84473) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_84470, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:241:48-81\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:241:12-82\n   #9  futhark/microgpt.fut:449:5-76\n   #10 futhark/microgpt.fut:454:26-460:31\n   #11 futhark/microgpt.fut:476:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_84475 = ((double *) mem_87113)[i_86099 * (int64_t) 16 + tmp_84470];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_84483 = ((double *) mem_87112)[i_86099 * (int64_t) 16 + tmp_84470];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_84494 = ((double *) mem_87111)[i_86099 * (int64_t) 16 + tmp_84470];
                    
                    ((double *) mem_87198)[i_86089] = lifted_lambda_res_84494;
                    ((double *) mem_87199)[i_86089] = lifted_lambda_res_84483;
                    ((double *) mem_87200)[i_86089] = lifted_lambda_res_84475;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_87183, i_86099 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87198, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_87184, i_86099 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87199, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_87185, i_86099 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87200, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_87165, i_86109 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_87183, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_87166, i_86109 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_87184, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_87167, i_86109 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_87185, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86161 = 0; i_86161 < (int64_t) 4; i_86161++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86124 = 0; i_86124 < (int64_t) 16; i_86124++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_86117 = 0; i_86117 < (int64_t) 16; i_86117++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_84573;
                    double r_84575 = 0.0;
                    
                    for (int64_t i_84574 = 0; i_84574 < (int64_t) 4; i_84574++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_84576 = ((double *) mem_87167)[i_86161 * (int64_t) 64 + i_86124 * (int64_t) 4 + i_84574];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_84577 = ((double *) mem_87166)[i_86161 * (int64_t) 64 + i_86117 * (int64_t) 4 + i_84574];
                        
                        // futhark/microgpt.fut:244:110-163
                        
                        double zt_res_84578 = zt_lhs_84576 * zt_rhs_84577;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_84579 = r_84575 + zt_res_84578;
                        double r_tmp_88735 = zp_res_84579;
                        
                        r_84575 = r_tmp_88735;
                    }
                    defunc_0_lifted_lambda_res_84573 = r_84575;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_84586;
                    double r_84588 = 0.0;
                    
                    for (int64_t i_84587 = 0; i_84587 < (int64_t) 4; i_84587++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_84589 = ((double *) mem_87167)[i_86161 * (int64_t) 64 + i_86124 * (int64_t) 4 + i_84587];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_84590 = ((double *) mem_87166)[i_86161 * (int64_t) 64 + i_86117 * (int64_t) 4 + i_84587];
                        
                        // futhark/microgpt.fut:291:75-134
                        
                        double zt_res_84591 = zt_lhs_84589 * zt_rhs_84590;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_84592 = r_84588 + zt_res_84591;
                        double r_tmp_88736 = zp_res_84592;
                        
                        r_84588 = r_tmp_88736;
                    }
                    defunc_0_lifted_lambda_res_84586 = r_84588;
                    ((double *) mem_87268)[i_86117] = defunc_0_lifted_lambda_res_84586;
                    ((double *) mem_87269)[i_86117] = defunc_0_lifted_lambda_res_84573;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_87258, i_86124 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87268, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_87259, i_86124 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87269, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86133 = 0; i_86133 < (int64_t) 16; i_86133++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_86129 = 0; i_86129 < (int64_t) 16; i_86129++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_82978 = ((double *) mem_87259)[i_86133 * (int64_t) 16 + i_86129];
                    
                    // futhark/microgpt.fut:245:47-78
                    
                    double zs_res_82979 = zs_lhs_82978 / 2.0;
                    double zp_rhs_82980 = ((double *) masks_mem_86864.mem)[step_80406 * (int64_t) 256 + i_86133 * (int64_t) 16 + i_86129];
                    
                    // futhark/microgpt.fut:245:65-102
                    
                    double zp_res_82981 = zs_res_82979 + zp_rhs_82980;
                    
                    ((double *) mem_87295)[i_86129] = zp_res_82981;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_87290, i_86133 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87295, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86147 = 0; i_86147 < (int64_t) 16; i_86147++) {
                // futhark/microgpt.fut:103:13-33
                
                double defunc_0_reduce_res_85845;
                double redout_86135 = -INFINITY;
                
                for (int64_t i_86136 = 0; i_86136 < (int64_t) 16; i_86136++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_84610 = ((double *) mem_87290)[i_86147 * (int64_t) 16 + i_86136];
                    
                    // futhark/microgpt.fut:103:13-33
                    
                    double max_res_83002 = fmax64(lifted_lambda_res_84610, redout_86135);
                    double redout_tmp_88740 = max_res_83002;
                    
                    redout_86135 = redout_tmp_88740;
                }
                defunc_0_reduce_res_85845 = redout_86135;
                // futhark/microgpt.fut:247:65-74
                
                double neg_res_83003 = -defunc_0_reduce_res_85845;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_86139 = 0; i_86139 < (int64_t) 16; i_86139++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_83010 = ((double *) mem_87290)[i_86147 * (int64_t) 16 + i_86139];
                    
                    // futhark/microgpt.fut:247:43-74
                    
                    double zp_res_83011 = neg_res_83003 + zp_lhs_83010;
                    
                    // futhark/microgpt.fut:247:36-74
                    
                    double exp_res_83012 = futrts_exp64(zp_res_83011);
                    
                    ((double *) mem_87311)[i_86139] = exp_res_83012;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_83014;
                double r_83016 = 0.0;
                
                for (int64_t i_83015 = 0; i_83015 < (int64_t) 16; i_83015++) {
                    // futhark/microgpt.fut:248:36-46
                    
                    double lifted_lambda_res_83017 = ((double *) mem_87311)[i_83015];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_83018 = r_83016 + lifted_lambda_res_83017;
                    double r_tmp_88742 = zp_res_83018;
                    
                    r_83016 = r_tmp_88742;
                }
                defunc_0_lifted_lambda_res_83014 = r_83016;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_86143 = 0; i_86143 < (int64_t) 16; i_86143++) {
                    // futhark/microgpt.fut:249:5-15
                    
                    double zs_lhs_83025 = ((double *) mem_87311)[i_86143];
                    
                    // futhark/microgpt.fut:249:5-23
                    
                    double zs_res_83026 = zs_lhs_83025 / defunc_0_lifted_lambda_res_83014;
                    
                    ((double *) mem_87318)[i_86143] = zs_res_83026;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_87306, i_86147 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87318, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86155 = 0; i_86155 < (int64_t) 16; i_86155++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_86151 = 0; i_86151 < (int64_t) 4; i_86151++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_83041;
                    double r_83043 = 0.0;
                    
                    for (int64_t i_83042 = 0; i_83042 < (int64_t) 16; i_83042++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_83044 = ((double *) mem_87306)[i_86155 * (int64_t) 16 + i_83042];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_83045 = ((double *) mem_87165)[i_86161 * (int64_t) 64 + i_83042 * (int64_t) 4 + i_86151];
                        
                        // futhark/microgpt.fut:250:26-72
                        
                        double zt_res_83046 = zt_lhs_83044 * zt_rhs_83045;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_83047 = r_83043 + zt_res_83046;
                        double r_tmp_88746 = zp_res_83047;
                        
                        r_83043 = r_tmp_88746;
                    }
                    defunc_0_lifted_lambda_res_83041 = r_83043;
                    ((double *) mem_87334)[i_86151] = defunc_0_lifted_lambda_res_83041;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_87329, i_86155 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87334, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_87246, i_86161 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_87258, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_87247, i_86161 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_87329, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86170 = 0; i_86170 < (int64_t) 16; i_86170++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86166 = 0; i_86166 < (int64_t) 16; i_86166++) {
                // futhark/microgpt.fut:251:52-55
                
                int64_t tmp_80792 = sdiv64(i_86166, (int64_t) 4);
                
                // futhark/microgpt.fut:251:41-57
                
                bool x_80793 = sle64((int64_t) 0, tmp_80792);
                
                // futhark/microgpt.fut:251:41-57
                
                bool y_80794 = slt64(tmp_80792, (int64_t) 4);
                
                // futhark/microgpt.fut:251:41-57
                
                bool bounds_check_80795 = x_80793 && y_80794;
                
                // futhark/microgpt.fut:251:41-57
                
                bool index_certs_80796;
                
                if (!bounds_check_80795) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_80792, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:251:41-57\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:251:12-78\n   #6  futhark/microgpt.fut:449:5-76\n   #7  futhark/microgpt.fut:454:26-460:31\n   #8  futhark/microgpt.fut:476:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:251:72-75
                
                int64_t tmp_80797 = smod64(i_86166, (int64_t) 4);
                
                // futhark/microgpt.fut:251:41-77
                
                bool x_80798 = sle64((int64_t) 0, tmp_80797);
                
                // futhark/microgpt.fut:251:41-77
                
                bool y_80799 = slt64(tmp_80797, (int64_t) 4);
                
                // futhark/microgpt.fut:251:41-77
                
                bool bounds_check_80800 = x_80798 && y_80799;
                
                // futhark/microgpt.fut:251:41-77
                
                bool index_certs_80801;
                
                if (!bounds_check_80800) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_80797, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:251:41-77\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:251:12-78\n   #6  futhark/microgpt.fut:449:5-76\n   #7  futhark/microgpt.fut:454:26-460:31\n   #8  futhark/microgpt.fut:476:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_80802 = ((double *) mem_87247)[tmp_80792 * (int64_t) 64 + i_86170 * (int64_t) 4 + tmp_80797];
                
                ((double *) mem_87360)[i_86166] = lifted_lambda_res_80802;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87355, i_86170 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87360, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86178 = 0; i_86178 < (int64_t) 16; i_86178++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86174 = 0; i_86174 < (int64_t) 16; i_86174++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_80817;
                double r_80819 = 0.0;
                
                for (int64_t i_80818 = 0; i_80818 < (int64_t) 16; i_80818++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_80820 = ((double *) mem_param_86878.mem)[i_86174 * (int64_t) 16 + i_80818];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_80821 = ((double *) mem_87355)[i_86178 * (int64_t) 16 + i_80818];
                    
                    // futhark/microgpt.fut:252:63-103
                    
                    double zt_res_80822 = zt_lhs_80820 * zt_rhs_80821;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_80823 = r_80819 + zt_res_80822;
                    double r_tmp_88751 = zp_res_80823;
                    
                    r_80819 = r_tmp_88751;
                }
                defunc_0_lifted_lambda_res_80817 = r_80819;
                ((double *) mem_87376)[i_86174] = defunc_0_lifted_lambda_res_80817;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87371, i_86178 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87376, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86186 = 0; i_86186 < (int64_t) 16; i_86186++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86182 = 0; i_86182 < (int64_t) 16; i_86182++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_80838 = ((double *) mem_87371)[i_86186 * (int64_t) 16 + i_86182];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_80839 = ((double *) mem_87040)[i_86186 * (int64_t) 16 + i_86182];
                
                // futhark/microgpt.fut:253:42-80
                
                double zp_res_80840 = zp_lhs_80838 + zp_rhs_80839;
                
                ((double *) mem_87392)[i_86182] = zp_res_80840;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87387, i_86186 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87392, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86199 = 0; i_86199 < (int64_t) 16; i_86199++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_83094;
            double r_83096 = 0.0;
            
            for (int64_t i_83095 = 0; i_83095 < (int64_t) 16; i_83095++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_83097 = ((double *) mem_87387)[i_86199 * (int64_t) 16 + i_83095];
                
                // futhark/microgpt.fut:254:75-114
                
                double zt_res_83098 = zt_lhs_83097 * zt_lhs_83097;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_83099 = r_83096 + zt_res_83098;
                double r_tmp_88756 = zp_res_83099;
                
                r_83096 = r_tmp_88756;
            }
            defunc_0_lifted_lambda_res_83094 = r_83096;
            // futhark/microgpt.fut:254:54-132
            
            double zs_res_83100 = defunc_0_lifted_lambda_res_83094 / 16.0;
            
            // futhark/microgpt.fut:255:24-55
            
            double zp_res_83101 = 1.0e-5 + zs_res_83100;
            
            // futhark/microgpt.fut:255:16-55
            
            double sqrt_res_83102 = futrts_sqrt64(zp_res_83101);
            
            // futhark/microgpt.fut:256:28-39
            
            double zs_res_83103 = 1.0 / sqrt_res_83102;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86192 = 0; i_86192 < (int64_t) 16; i_86192++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_84649 = ((double *) mem_87387)[i_86199 * (int64_t) 16 + i_86192];
                
                // futhark/microgpt.fut:256:5-39
                
                double zt_res_84650 = zs_res_83103 * zt_lhs_84649;
                
                // futhark/microgpt.fut:282:45-88
                
                double zt_res_84658 = zt_lhs_84649 * zt_lhs_84649;
                
                ((double *) mem_87413)[i_86192] = zt_res_84658;
                ((double *) mem_87414)[i_86192] = zt_res_84650;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87403, i_86199 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87413, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87404, i_86199 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87414, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86210 = 0; i_86210 < (int64_t) 16; i_86210++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86204 = 0; i_86204 < (int64_t) 64; i_86204++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_83151;
                double r_83153 = 0.0;
                
                for (int64_t i_83152 = 0; i_83152 < (int64_t) 16; i_83152++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_83154 = ((double *) mem_param_86894.mem)[i_86204 * (int64_t) 16 + i_83152];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_83155 = ((double *) mem_87404)[i_86210 * (int64_t) 16 + i_83152];
                    
                    // futhark/microgpt.fut:257:63-102
                    
                    double zt_res_83156 = zt_lhs_83154 * zt_rhs_83155;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_83157 = r_83153 + zt_res_83156;
                    double r_tmp_88762 = zp_res_83157;
                    
                    r_83153 = r_tmp_88762;
                }
                defunc_0_lifted_lambda_res_83151 = r_83153;
                ((double *) mem_87444)[i_86204] = defunc_0_lifted_lambda_res_83151;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_83165;
            double r_83167 = 0.0;
            
            for (int64_t i_83166 = 0; i_83166 < (int64_t) 16; i_83166++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_83168 = ((double *) mem_87403)[i_86210 * (int64_t) 16 + i_83166];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_83169 = r_83167 + lifted_lambda_res_83168;
                double r_tmp_88763 = zp_res_83169;
                
                r_83167 = r_tmp_88763;
            }
            defunc_0_lifted_lambda_res_83165 = r_83167;
            // futhark/microgpt.fut:283:36-94
            
            double zs_res_83170 = defunc_0_lifted_lambda_res_83165 / 16.0;
            
            ((double *) mem_87435)[i_86210] = zs_res_83170;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87436, i_86210 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87444, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86219 = 0; i_86219 < (int64_t) 16; i_86219++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86215 = 0; i_86215 < (int64_t) 64; i_86215++) {
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_80903 = ((double *) mem_87436)[i_86219 * (int64_t) 64 + i_86215];
                
                // futhark/microgpt.fut:258:41-69
                
                double max_res_80904 = fmax64(0.0, max_arg0_80903);
                
                ((double *) mem_87463)[i_86215] = max_res_80904;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87458, i_86219 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87463, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86227 = 0; i_86227 < (int64_t) 16; i_86227++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86223 = 0; i_86223 < (int64_t) 16; i_86223++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_80919;
                double r_80921 = 0.0;
                
                for (int64_t i_80920 = 0; i_80920 < (int64_t) 64; i_80920++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_80922 = ((double *) mem_param_86870.mem)[i_86223 * (int64_t) 64 + i_80920];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_80923 = ((double *) mem_87458)[i_86227 * (int64_t) 64 + i_80920];
                    
                    // futhark/microgpt.fut:259:63-104
                    
                    double zt_res_80924 = zt_lhs_80922 * zt_rhs_80923;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_80925 = r_80921 + zt_res_80924;
                    double r_tmp_88768 = zp_res_80925;
                    
                    r_80921 = r_tmp_88768;
                }
                defunc_0_lifted_lambda_res_80919 = r_80921;
                ((double *) mem_87479)[i_86223] = defunc_0_lifted_lambda_res_80919;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87474, i_86227 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87479, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86235 = 0; i_86235 < (int64_t) 16; i_86235++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86231 = 0; i_86231 < (int64_t) 16; i_86231++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_80940 = ((double *) mem_87474)[i_86235 * (int64_t) 16 + i_86231];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_80941 = ((double *) mem_87387)[i_86235 * (int64_t) 16 + i_86231];
                
                // futhark/microgpt.fut:260:42-81
                
                double zp_res_80942 = zp_lhs_80940 + zp_rhs_80941;
                
                ((double *) mem_87495)[i_86231] = zp_res_80942;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87490, i_86235 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87495, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86243 = 0; i_86243 < (int64_t) 16; i_86243++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86239 = 0; i_86239 < (int64_t) 27; i_86239++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_80957;
                double r_80959 = 0.0;
                
                for (int64_t i_80958 = 0; i_80958 < (int64_t) 16; i_80958++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_80960 = ((double *) mem_param_86902.mem)[i_86239 * (int64_t) 16 + i_80958];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_80961 = ((double *) mem_87490)[i_86243 * (int64_t) 16 + i_80958];
                    
                    // futhark/microgpt.fut:261:63-103
                    
                    double zt_res_80962 = zt_lhs_80960 * zt_rhs_80961;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_80963 = r_80959 + zt_res_80962;
                    double r_tmp_88773 = zp_res_80963;
                    
                    r_80959 = r_tmp_88773;
                }
                defunc_0_lifted_lambda_res_80957 = r_80959;
                ((double *) mem_87511)[i_86239] = defunc_0_lifted_lambda_res_80957;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87506, i_86243 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87511, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86266 = 0; i_86266 < (int64_t) 16; i_86266++) {
            // futhark/microgpt.fut:103:13-33
            
            double defunc_0_reduce_res_85864;
            double defunc_0_reduce_res_85865;
            double redout_86245;
            double redout_86246;
            
            redout_86245 = -INFINITY;
            redout_86246 = -INFINITY;
            for (int64_t i_86247 = 0; i_86247 < (int64_t) 27; i_86247++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_84726 = ((double *) mem_87506)[i_86266 * (int64_t) 27 + i_86247];
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_83200 = fmax64(lifted_lambda_res_84726, redout_86245);
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_83244 = fmax64(lifted_lambda_res_84726, redout_86246);
                double redout_tmp_88776 = max_res_83200;
                double redout_tmp_88777 = max_res_83244;
                
                redout_86245 = redout_tmp_88776;
                redout_86246 = redout_tmp_88777;
            }
            defunc_0_reduce_res_85864 = redout_86245;
            defunc_0_reduce_res_85865 = redout_86246;
            // futhark/microgpt.fut:269:65-74
            
            double neg_res_83201 = -defunc_0_reduce_res_85864;
            
            // futhark/microgpt.fut:273:65-74
            
            double neg_res_83245 = -defunc_0_reduce_res_85865;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86252 = 0; i_86252 < (int64_t) 27; i_86252++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_84765 = ((double *) mem_87506)[i_86266 * (int64_t) 27 + i_86252];
                
                // futhark/microgpt.fut:269:43-74
                
                double zp_res_84766 = neg_res_83201 + zp_lhs_84765;
                
                // futhark/microgpt.fut:269:36-74
                
                double exp_res_84767 = futrts_exp64(zp_res_84766);
                
                // futhark/microgpt.fut:273:43-74
                
                double zp_res_84775 = neg_res_83245 + zp_lhs_84765;
                
                // futhark/microgpt.fut:273:36-74
                
                double exp_res_84776 = futrts_exp64(zp_res_84775);
                
                ((double *) mem_87532)[i_86252] = exp_res_84776;
                ((double *) mem_87533)[i_86252] = exp_res_84767;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_83212;
            double r_83214 = 0.0;
            
            for (int64_t i_83213 = 0; i_83213 < (int64_t) 27; i_83213++) {
                // futhark/microgpt.fut:270:36-46
                
                double lifted_lambda_res_83215 = ((double *) mem_87533)[i_83213];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_83216 = r_83214 + lifted_lambda_res_83215;
                double r_tmp_88780 = zp_res_83216;
                
                r_83214 = r_tmp_88780;
            }
            defunc_0_lifted_lambda_res_83212 = r_83214;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_83256;
            double r_83258 = 0.0;
            
            for (int64_t i_83257 = 0; i_83257 < (int64_t) 27; i_83257++) {
                // futhark/microgpt.fut:274:36-46
                
                double lifted_lambda_res_83259 = ((double *) mem_87532)[i_83257];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_83260 = r_83258 + lifted_lambda_res_83259;
                double r_tmp_88781 = zp_res_83260;
                
                r_83258 = r_tmp_88781;
            }
            defunc_0_lifted_lambda_res_83256 = r_83258;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86259 = 0; i_86259 < (int64_t) 27; i_86259++) {
                // futhark/microgpt.fut:271:5-15
                
                double zs_lhs_84794 = ((double *) mem_87533)[i_86259];
                
                // futhark/microgpt.fut:271:5-23
                
                double zs_res_84795 = zs_lhs_84794 / defunc_0_lifted_lambda_res_83212;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_84802 = ((double *) mem_87007)[i_86266 * (int64_t) 27 + i_86259];
                
                // futhark/microgpt.fut:275:37-47
                
                double zs_lhs_84803 = ((double *) mem_87532)[i_86259];
                
                // futhark/microgpt.fut:275:37-55
                
                double zs_res_84804 = zs_lhs_84803 / defunc_0_lifted_lambda_res_83256;
                
                // futhark/microgpt.fut:275:28-55
                
                double zs_res_84805 = 1.0 / zs_res_84804;
                
                // futhark/microgpt.fut:275:5-55
                
                double zt_res_84806 = zt_lhs_84802 * zs_res_84805;
                
                ((double *) mem_87546)[i_86259] = zt_res_84806;
                ((double *) mem_87547)[i_86259] = zs_res_84795;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87522, i_86266 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87546, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87523, i_86266 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87547, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86271 = 0; i_86271 < (int64_t) 16; i_86271++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_81081;
            double r_81083 = 0.0;
            
            for (int64_t i_81082 = 0; i_81082 < (int64_t) 27; i_81082++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_81084 = ((double *) mem_87522)[i_86271 * (int64_t) 27 + i_81082];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_81085 = ((double *) mem_87523)[i_86271 * (int64_t) 27 + i_81082];
                
                // futhark/microgpt.fut:276:54-93
                
                double zt_res_81086 = zt_lhs_81084 * zt_rhs_81085;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_81087 = r_81083 + zt_res_81086;
                double r_tmp_88785 = zp_res_81087;
                
                r_81083 = r_tmp_88785;
            }
            defunc_0_lifted_lambda_res_81081 = r_81083;
            ((double *) mem_87568)[i_86271] = defunc_0_lifted_lambda_res_81081;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86279 = 0; i_86279 < (int64_t) 16; i_86279++) {
            // futhark/microgpt.fut:277:94-104
            
            double neg_arg0_81095 = ((double *) mem_87568)[i_86279];
            
            // futhark/microgpt.fut:277:88-104
            
            double neg_res_81096 = -neg_arg0_81095;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86275 = 0; i_86275 < (int64_t) 27; i_86275++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_81103 = ((double *) mem_87523)[i_86279 * (int64_t) 27 + i_86275];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_81104 = ((double *) mem_87522)[i_86279 * (int64_t) 27 + i_86275];
                
                // futhark/microgpt.fut:277:65-104
                
                double zp_res_81105 = neg_res_81096 + zp_lhs_81104;
                
                // futhark/microgpt.fut:277:42-104
                
                double zt_res_81106 = zt_lhs_81103 * zp_res_81105;
                
                ((double *) mem_87580)[i_86275] = zt_res_81106;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87575, i_86279 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87580, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86287 = 0; i_86287 < (int64_t) 16; i_86287++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86283 = 0; i_86283 < (int64_t) 16; i_86283++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81121;
                double r_81123 = 0.0;
                
                for (int64_t i_81122 = 0; i_81122 < (int64_t) 27; i_81122++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81124 = ((double *) mem_param_86902.mem)[i_81122 * (int64_t) 16 + i_86283];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81125 = ((double *) mem_87575)[i_86287 * (int64_t) 27 + i_81122];
                    
                    // futhark/microgpt.fut:278:63-103
                    
                    double zt_res_81126 = zt_lhs_81124 * zt_rhs_81125;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81127 = r_81123 + zt_res_81126;
                    double r_tmp_88790 = zp_res_81127;
                    
                    r_81123 = r_tmp_88790;
                }
                defunc_0_lifted_lambda_res_81121 = r_81123;
                ((double *) mem_87596)[i_86283] = defunc_0_lifted_lambda_res_81121;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87591, i_86287 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87596, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86300 = 0; i_86300 < (int64_t) 16; i_86300++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86293 = 0; i_86293 < (int64_t) 64; i_86293++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_84832;
                double r_84834 = 0.0;
                
                for (int64_t i_84833 = 0; i_84833 < (int64_t) 16; i_84833++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_84835 = ((double *) mem_param_86870.mem)[i_84833 * (int64_t) 64 + i_86293];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_84836 = ((double *) mem_87591)[i_86300 * (int64_t) 16 + i_84833];
                    
                    // futhark/microgpt.fut:279:63-104
                    
                    double zt_res_84837 = zt_lhs_84835 * zt_rhs_84836;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_84838 = r_84834 + zt_res_84837;
                    double r_tmp_88795 = zp_res_84838;
                    
                    r_84834 = r_tmp_88795;
                }
                defunc_0_lifted_lambda_res_84832 = r_84834;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_84845;
                double r_84847 = 0.0;
                
                for (int64_t i_84846 = 0; i_84846 < (int64_t) 16; i_84846++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_84848 = ((double *) mem_87591)[i_84846 * (int64_t) 16 + i_86300];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_84849 = ((double *) mem_87458)[i_84846 * (int64_t) 64 + i_86293];
                    
                    // futhark/microgpt.fut:331:69-112
                    
                    double zt_res_84850 = zt_lhs_84848 * zt_rhs_84849;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_84851 = r_84847 + zt_res_84850;
                    double r_tmp_88796 = zp_res_84851;
                    
                    r_84847 = r_tmp_88796;
                }
                defunc_0_lifted_lambda_res_84845 = r_84847;
                ((double *) mem_87617)[i_86293] = defunc_0_lifted_lambda_res_84845;
                ((double *) mem_87618)[i_86293] = defunc_0_lifted_lambda_res_84832;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87607, i_86300 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87617, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87608, i_86300 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87618, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86309 = 0; i_86309 < (int64_t) 16; i_86309++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86305 = 0; i_86305 < (int64_t) 64; i_86305++) {
                // futhark/microgpt.fut:4:11-25
                
                double indicatorp_arg0_81163 = ((double *) mem_87436)[i_86309 * (int64_t) 64 + i_86305];
                
                // futhark/microgpt.fut:124:42-54
                
                double max_res_81164 = fmax64(0.0, indicatorp_arg0_81163);
                
                // futhark/microgpt.fut:124:35-54
                
                double sgn_res_81165 = fsignum64(max_res_81164);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_81166 = ((double *) mem_87608)[i_86309 * (int64_t) 64 + i_86305];
                
                // futhark/microgpt.fut:280:43-94
                
                double zt_res_81167 = sgn_res_81165 * zt_rhs_81166;
                
                ((double *) mem_87644)[i_86305] = zt_res_81167;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87639, i_86309 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87644, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86317 = 0; i_86317 < (int64_t) 16; i_86317++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86313 = 0; i_86313 < (int64_t) 16; i_86313++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81182;
                double r_81184 = 0.0;
                
                for (int64_t i_81183 = 0; i_81183 < (int64_t) 64; i_81183++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81185 = ((double *) mem_param_86894.mem)[i_81183 * (int64_t) 16 + i_86313];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81186 = ((double *) mem_87639)[i_86317 * (int64_t) 64 + i_81183];
                    
                    // futhark/microgpt.fut:281:63-102
                    
                    double zt_res_81187 = zt_lhs_81185 * zt_rhs_81186;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81188 = r_81184 + zt_res_81187;
                    double r_tmp_88801 = zp_res_81188;
                    
                    r_81184 = r_tmp_88801;
                }
                defunc_0_lifted_lambda_res_81182 = r_81184;
                ((double *) mem_87660)[i_86313] = defunc_0_lifted_lambda_res_81182;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87655, i_86317 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87660, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86323 = 0; i_86323 < (int64_t) 16; i_86323++) {
            // futhark/microgpt.fut:284:43-55
            
            double zp_lhs_82278 = ((double *) mem_87435)[i_86323];
            
            // futhark/microgpt.fut:284:43-83
            
            double zp_res_82279 = 1.0e-5 + zp_lhs_82278;
            
            // futhark/microgpt.fut:284:35-83
            
            double sqrt_res_82280 = futrts_sqrt64(zp_res_82279);
            
            // futhark/microgpt.fut:285:65-85
            
            double zs_res_82288 = 1.0 / sqrt_res_82280;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_82289;
            double r_82291 = 0.0;
            
            for (int64_t i_82290 = 0; i_82290 < (int64_t) 16; i_82290++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_82292 = ((double *) mem_87387)[i_86323 * (int64_t) 16 + i_82290];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_82293 = ((double *) mem_87655)[i_86323 * (int64_t) 16 + i_82290];
                
                // futhark/microgpt.fut:285:93-136
                
                double zt_res_82294 = zt_lhs_82292 * zt_rhs_82293;
                
                // futhark/microgpt.fut:285:113-163
                
                double zt_res_82295 = zs_res_82288 * zt_res_82294;
                
                // futhark/microgpt.fut:285:69-163
                
                double zt_res_82296 = zs_res_82288 * zt_res_82295;
                
                // futhark/microgpt.fut:285:57-163
                
                double neg_res_82297 = -zt_res_82296;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_82298 = r_82291 + neg_res_82297;
                double r_tmp_88804 = zp_res_82298;
                
                r_82291 = r_tmp_88804;
            }
            defunc_0_lifted_lambda_res_82289 = r_82291;
            ((double *) mem_87671)[i_86323] = defunc_0_lifted_lambda_res_82289;
            ((double *) mem_87672)[i_86323] = sqrt_res_82280;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86328 = 0; i_86328 < (int64_t) 16; i_86328++) {
            // futhark/microgpt.fut:286:35-47
            
            double zt_lhs_81255 = ((double *) mem_87671)[i_86328];
            
            // futhark/microgpt.fut:286:89-101
            
            double zp_lhs_81256 = ((double *) mem_87435)[i_86328];
            
            // futhark/microgpt.fut:286:89-129
            
            double zp_res_81257 = 1.0e-5 + zp_lhs_81256;
            
            // futhark/microgpt.fut:286:81-129
            
            double sqrt_res_81258 = futrts_sqrt64(zp_res_81257);
            
            // futhark/microgpt.fut:286:67-131
            
            double zt_res_81259 = 2.0 * sqrt_res_81258;
            
            // futhark/microgpt.fut:286:53-131
            
            double zs_res_81260 = 1.0 / zt_res_81259;
            
            // futhark/microgpt.fut:286:35-131
            
            double zt_res_81261 = zt_lhs_81255 * zs_res_81260;
            
            ((double *) mem_87685)[i_86328] = zt_res_81261;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86332 = 0; i_86332 < (int64_t) 16; i_86332++) {
            // futhark/microgpt.fut:287:45-57
            
            double zs_lhs_81269 = ((double *) mem_87685)[i_86332];
            
            // futhark/microgpt.fut:287:45-72
            
            double zs_res_81270 = zs_lhs_81269 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_88807 = 0; nest_i_88807 < (int64_t) 16; nest_i_88807++) {
                ((double *) mem_87692)[i_86332 * (int64_t) 16 + nest_i_88807] = zs_res_81270;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86340 = 0; i_86340 < (int64_t) 16; i_86340++) {
            // futhark/microgpt.fut:288:105-117
            
            double zs_rhs_81279 = ((double *) mem_87672)[i_86340];
            
            // futhark/microgpt.fut:288:97-117
            
            double zs_res_81280 = 1.0 / zs_rhs_81279;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86336 = 0; i_86336 < (int64_t) 16; i_86336++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_81287 = ((double *) mem_87591)[i_86340 * (int64_t) 16 + i_86336];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_81288 = ((double *) mem_87655)[i_86340 * (int64_t) 16 + i_86336];
                
                // futhark/microgpt.fut:288:72-117
                
                double zt_res_81289 = zs_res_81280 * zt_lhs_81288;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_81290 = ((double *) mem_87387)[i_86340 * (int64_t) 16 + i_86336];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_81291 = ((double *) mem_87692)[i_86340 * (int64_t) 16 + i_86336];
                
                // futhark/microgpt.fut:288:125-169
                
                double zt_res_81292 = zt_lhs_81290 * zt_rhs_81291;
                
                // futhark/microgpt.fut:288:92-169
                
                double zp_res_81293 = zt_res_81289 + zt_res_81292;
                
                // futhark/microgpt.fut:288:120-221
                
                double zp_res_81294 = zt_res_81292 + zp_res_81293;
                
                // futhark/microgpt.fut:288:45-221
                
                double zp_res_81295 = zp_lhs_81287 + zp_res_81294;
                
                ((double *) mem_87707)[i_86336] = zp_res_81295;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87702, i_86340 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87707, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86353 = 0; i_86353 < (int64_t) 16; i_86353++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86346 = 0; i_86346 < (int64_t) 16; i_86346++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_84874;
                double r_84876 = 0.0;
                
                for (int64_t i_84875 = 0; i_84875 < (int64_t) 16; i_84875++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_84877 = ((double *) mem_param_86878.mem)[i_84875 * (int64_t) 16 + i_86346];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_84878 = ((double *) mem_87702)[i_86353 * (int64_t) 16 + i_84875];
                    
                    // futhark/microgpt.fut:289:67-112
                    
                    double zt_res_84879 = zt_lhs_84877 * zt_rhs_84878;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_84880 = r_84876 + zt_res_84879;
                    double r_tmp_88814 = zp_res_84880;
                    
                    r_84876 = r_tmp_88814;
                }
                defunc_0_lifted_lambda_res_84874 = r_84876;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_84887;
                double r_84889 = 0.0;
                
                for (int64_t i_84888 = 0; i_84888 < (int64_t) 16; i_84888++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_84890 = ((double *) mem_87702)[i_84888 * (int64_t) 16 + i_86353];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_84891 = ((double *) mem_87355)[i_84888 * (int64_t) 16 + i_86346];
                    
                    // futhark/microgpt.fut:329:68-112
                    
                    double zt_res_84892 = zt_lhs_84890 * zt_rhs_84891;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_84893 = r_84889 + zt_res_84892;
                    double r_tmp_88815 = zp_res_84893;
                    
                    r_84889 = r_tmp_88815;
                }
                defunc_0_lifted_lambda_res_84887 = r_84889;
                ((double *) mem_87728)[i_86346] = defunc_0_lifted_lambda_res_84887;
                ((double *) mem_87729)[i_86346] = defunc_0_lifted_lambda_res_84874;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87718, i_86353 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87728, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87719, i_86353 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87729, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86375 = 0; i_86375 < (int64_t) 4; i_86375++) {
            // futhark/microgpt.fut:290:74-77
            
            int64_t zp_lhs_83388 = mul64((int64_t) 4, i_86375);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86368 = 0; i_86368 < (int64_t) 16; i_86368++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_86358 = 0; i_86358 < (int64_t) 4; i_86358++) {
                    // futhark/microgpt.fut:290:79-87
                    
                    int64_t tmp_84915 = add64(zp_lhs_83388, i_86358);
                    
                    // futhark/microgpt.fut:290:52-89
                    
                    bool x_84916 = sle64((int64_t) 0, tmp_84915);
                    
                    // futhark/microgpt.fut:290:52-89
                    
                    bool y_84917 = slt64(tmp_84915, (int64_t) 16);
                    
                    // futhark/microgpt.fut:290:52-89
                    
                    bool bounds_check_84918 = x_84916 && y_84917;
                    
                    // futhark/microgpt.fut:290:52-89
                    
                    bool index_certs_84919;
                    
                    if (!bounds_check_84918) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_84915, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:290:52-89\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:290:13-90\n   #9  futhark/microgpt.fut:449:5-76\n   #10 futhark/microgpt.fut:454:26-460:31\n   #11 futhark/microgpt.fut:476:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_84920 = ((double *) mem_87719)[i_86368 * (int64_t) 16 + tmp_84915];
                    
                    ((double *) mem_87772)[i_86358] = lifted_lambda_res_84920;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_86362 = 0; i_86362 < (int64_t) 16; i_86362++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_84934 = ((double *) mem_87246)[i_86375 * (int64_t) 256 + i_86368 * (int64_t) 16 + i_86362];
                    
                    // futhark/microgpt.fut:292:55-97
                    
                    double zs_res_84935 = zs_lhs_84934 / 2.0;
                    double zp_rhs_84936 = ((double *) masks_mem_86864.mem)[step_80406 * (int64_t) 256 + i_86368 * (int64_t) 16 + i_86362];
                    
                    // futhark/microgpt.fut:292:84-123
                    
                    double zp_res_84937 = zs_res_84935 + zp_rhs_84936;
                    
                    ((double *) mem_87779)[i_86362] = zp_res_84937;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_87762, i_86368 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87779, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_87763, i_86368 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87772, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_87750, i_86375 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_87762, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_87751, i_86375 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_87763, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86402 = 0; i_86402 < (int64_t) 4; i_86402++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86395 = 0; i_86395 < (int64_t) 16; i_86395++) {
                // futhark/microgpt.fut:4:11-25
                
                double defunc_0_reduce_res_85883;
                double redout_86379 = -INFINITY;
                
                for (int64_t i_86381 = 0; i_86381 < (int64_t) 16; i_86381++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_85055 = ((double *) mem_87750)[i_86402 * (int64_t) 256 + i_86395 * (int64_t) 16 + i_86381];
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_85066;
                    double r_85068 = 0.0;
                    
                    for (int64_t i_85067 = 0; i_85067 < (int64_t) 4; i_85067++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_85069 = ((double *) mem_87751)[i_86402 * (int64_t) 64 + i_86395 * (int64_t) 4 + i_85067];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_85070 = ((double *) mem_87165)[i_86402 * (int64_t) 64 + i_86381 * (int64_t) 4 + i_85067];
                        
                        // futhark/microgpt.fut:297:75-135
                        
                        double zt_res_85071 = zt_lhs_85069 * zt_rhs_85070;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_85072 = r_85068 + zt_res_85071;
                        double r_tmp_88828 = zp_res_85072;
                        
                        r_85068 = r_tmp_88828;
                    }
                    defunc_0_lifted_lambda_res_85066 = r_85068;
                    // futhark/microgpt.fut:103:13-33
                    
                    double max_res_84974 = fmax64(lifted_lambda_res_85055, redout_86379);
                    
                    ((double *) mem_87826)[i_86381] = defunc_0_lifted_lambda_res_85066;
                    
                    double redout_tmp_88826 = max_res_84974;
                    
                    redout_86379 = redout_tmp_88826;
                }
                defunc_0_reduce_res_85883 = redout_86379;
                // futhark/microgpt.fut:294:78-88
                
                double neg_res_84975 = -defunc_0_reduce_res_85883;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_86385 = 0; i_86385 < (int64_t) 16; i_86385++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_84982 = ((double *) mem_87750)[i_86402 * (int64_t) 256 + i_86395 * (int64_t) 16 + i_86385];
                    
                    // futhark/microgpt.fut:294:45-88
                    
                    double zp_res_84983 = neg_res_84975 + zp_lhs_84982;
                    
                    // futhark/microgpt.fut:294:38-88
                    
                    double exp_res_84984 = futrts_exp64(zp_res_84983);
                    
                    ((double *) mem_87833)[i_86385] = exp_res_84984;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_84986;
                double r_84988 = 0.0;
                
                for (int64_t i_84987 = 0; i_84987 < (int64_t) 16; i_84987++) {
                    // futhark/microgpt.fut:295:38-50
                    
                    double lifted_lambda_res_84989 = ((double *) mem_87833)[i_84987];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_84990 = r_84988 + lifted_lambda_res_84989;
                    double r_tmp_88830 = zp_res_84990;
                    
                    r_84988 = r_tmp_88830;
                }
                defunc_0_lifted_lambda_res_84986 = r_84988;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_86389 = 0; i_86389 < (int64_t) 16; i_86389++) {
                    // futhark/microgpt.fut:296:5-17
                    
                    double zs_lhs_84997 = ((double *) mem_87833)[i_86389];
                    
                    // futhark/microgpt.fut:296:5-26
                    
                    double zs_res_84998 = zs_lhs_84997 / defunc_0_lifted_lambda_res_84986;
                    
                    ((double *) mem_87840)[i_86389] = zs_res_84998;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_87816, i_86395 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87826, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_87817, i_86395 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87840, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_87804, i_86402 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_87816, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_87805, i_86402 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_87817, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86424 = 0; i_86424 < (int64_t) 4; i_86424++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86417 = 0; i_86417 < (int64_t) 16; i_86417++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_86407 = 0; i_86407 < (int64_t) 16; i_86407++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_85106 = ((double *) mem_87804)[i_86424 * (int64_t) 256 + i_86417 * (int64_t) 16 + i_86407];
                    
                    ((double *) mem_87887)[i_86407] = lifted_lambda_res_85106;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_86411 = 0; i_86411 < (int64_t) 4; i_86411++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_85120;
                    double r_85122 = 0.0;
                    
                    for (int64_t i_85121 = 0; i_85121 < (int64_t) 16; i_85121++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_85123 = ((double *) mem_87805)[i_86424 * (int64_t) 256 + i_85121 * (int64_t) 16 + i_86417];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_85124 = ((double *) mem_87751)[i_86424 * (int64_t) 64 + i_85121 * (int64_t) 4 + i_86411];
                        
                        // futhark/microgpt.fut:302:75-136
                        
                        double zt_res_85125 = zt_lhs_85123 * zt_rhs_85124;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_85126 = r_85122 + zt_res_85125;
                        double r_tmp_88838 = zp_res_85126;
                        
                        r_85122 = r_tmp_88838;
                    }
                    defunc_0_lifted_lambda_res_85120 = r_85122;
                    ((double *) mem_87894)[i_86411] = defunc_0_lifted_lambda_res_85120;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_87877, i_86417 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87894, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_87878, i_86417 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87887, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_87865, i_86424 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_87877, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_87866, i_86424 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_87878, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86433 = 0; i_86433 < (int64_t) 4; i_86433++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86429 = 0; i_86429 < (int64_t) 16; i_86429++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81506;
                double r_81508 = 0.0;
                
                for (int64_t i_81507 = 0; i_81507 < (int64_t) 16; i_81507++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81509 = ((double *) mem_87866)[i_86433 * (int64_t) 256 + i_86429 * (int64_t) 16 + i_81507];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81510 = ((double *) mem_87805)[i_86433 * (int64_t) 256 + i_86429 * (int64_t) 16 + i_81507];
                    
                    // futhark/microgpt.fut:299:66-127
                    
                    double zt_res_81511 = zt_lhs_81509 * zt_rhs_81510;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81512 = r_81508 + zt_res_81511;
                    double r_tmp_88841 = zp_res_81512;
                    
                    r_81508 = r_tmp_88841;
                }
                defunc_0_lifted_lambda_res_81506 = r_81508;
                ((double *) mem_87924)[i_86429] = defunc_0_lifted_lambda_res_81506;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87919, i_86433 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87924, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86445 = 0; i_86445 < (int64_t) 4; i_86445++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86441 = 0; i_86441 < (int64_t) 16; i_86441++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_81527 = ((double *) mem_87919)[i_86445 * (int64_t) 16 + i_86441];
                
                // futhark/microgpt.fut:300:122-148
                
                double neg_res_81528 = -neg_arg0_81527;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_86437 = 0; i_86437 < (int64_t) 16; i_86437++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_81535 = ((double *) mem_87805)[i_86445 * (int64_t) 256 + i_86441 * (int64_t) 16 + i_86437];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_81536 = ((double *) mem_87866)[i_86445 * (int64_t) 256 + i_86441 * (int64_t) 16 + i_86437];
                    
                    // futhark/microgpt.fut:300:88-148
                    
                    double zp_res_81537 = neg_res_81528 + zp_lhs_81536;
                    
                    // futhark/microgpt.fut:300:54-148
                    
                    double zt_res_81538 = zt_lhs_81535 * zp_res_81537;
                    
                    ((double *) mem_87946)[i_86437] = zt_res_81538;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_87941, i_86441 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87946, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_87935, i_86445 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_87941, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86457 = 0; i_86457 < (int64_t) 4; i_86457++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86453 = 0; i_86453 < (int64_t) 16; i_86453++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_86449 = 0; i_86449 < (int64_t) 16; i_86449++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_81560 = ((double *) mem_87935)[i_86457 * (int64_t) 256 + i_86453 * (int64_t) 16 + i_86449];
                    
                    // futhark/microgpt.fut:301:54-96
                    
                    double zs_res_81561 = zs_lhs_81560 / 2.0;
                    
                    ((double *) mem_87973)[i_86449] = zs_res_81561;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_87968, i_86453 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87973, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_87962, i_86457 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_87968, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86477 = 0; i_86477 < (int64_t) 4; i_86477++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86470 = 0; i_86470 < (int64_t) 16; i_86470++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_86463 = 0; i_86463 < (int64_t) 4; i_86463++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_85207;
                    double r_85209 = 0.0;
                    
                    for (int64_t i_85208 = 0; i_85208 < (int64_t) 16; i_85208++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_85210 = ((double *) mem_87167)[i_86477 * (int64_t) 64 + i_85208 * (int64_t) 4 + i_86463];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_85211 = ((double *) mem_87962)[i_86477 * (int64_t) 256 + i_85208 * (int64_t) 16 + i_86470];
                        
                        // futhark/microgpt.fut:303:75-135
                        
                        double zt_res_85212 = zt_lhs_85210 * zt_rhs_85211;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_85213 = r_85209 + zt_res_85212;
                        double r_tmp_88854 = zp_res_85213;
                        
                        r_85209 = r_tmp_88854;
                    }
                    defunc_0_lifted_lambda_res_85207 = r_85209;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_85220;
                    double r_85222 = 0.0;
                    
                    for (int64_t i_85221 = 0; i_85221 < (int64_t) 16; i_85221++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_85223 = ((double *) mem_87962)[i_86477 * (int64_t) 256 + i_86470 * (int64_t) 16 + i_85221];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_85224 = ((double *) mem_87166)[i_86477 * (int64_t) 64 + i_85221 * (int64_t) 4 + i_86463];
                        
                        // futhark/microgpt.fut:304:75-135
                        
                        double zt_res_85225 = zt_lhs_85223 * zt_rhs_85224;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_85226 = r_85222 + zt_res_85225;
                        double r_tmp_88855 = zp_res_85226;
                        
                        r_85222 = r_tmp_88855;
                    }
                    defunc_0_lifted_lambda_res_85220 = r_85222;
                    ((double *) mem_88011)[i_86463] = defunc_0_lifted_lambda_res_85220;
                    ((double *) mem_88012)[i_86463] = defunc_0_lifted_lambda_res_85207;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88001, i_86470 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88011, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88002, i_86470 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88012, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_87989, i_86477 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_88001, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_87990, i_86477 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_88002, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86496 = 0; i_86496 < (int64_t) 16; i_86496++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86486 = 0; i_86486 < (int64_t) 16; i_86486++) {
                // futhark/microgpt.fut:305:57-60
                
                int64_t tmp_85289 = sdiv64(i_86486, (int64_t) 4);
                
                // futhark/microgpt.fut:305:44-62
                
                bool x_85290 = sle64((int64_t) 0, tmp_85289);
                
                // futhark/microgpt.fut:305:44-62
                
                bool y_85291 = slt64(tmp_85289, (int64_t) 4);
                
                // futhark/microgpt.fut:305:44-62
                
                bool bounds_check_85292 = x_85290 && y_85291;
                
                // futhark/microgpt.fut:305:44-62
                
                bool index_certs_85293;
                
                if (!bounds_check_85292) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_85289, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:305:44-62\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:305:13-85\n   #6  futhark/microgpt.fut:449:5-76\n   #7  futhark/microgpt.fut:454:26-460:31\n   #8  futhark/microgpt.fut:476:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:305:79-82
                
                int64_t tmp_85294 = smod64(i_86486, (int64_t) 4);
                
                // futhark/microgpt.fut:305:44-84
                
                bool x_85295 = sle64((int64_t) 0, tmp_85294);
                
                // futhark/microgpt.fut:305:44-84
                
                bool y_85296 = slt64(tmp_85294, (int64_t) 4);
                
                // futhark/microgpt.fut:305:44-84
                
                bool bounds_check_85297 = x_85295 && y_85296;
                
                // futhark/microgpt.fut:305:44-84
                
                bool index_certs_85298;
                
                if (!bounds_check_85297) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_85294, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:305:44-84\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:305:13-85\n   #6  futhark/microgpt.fut:449:5-76\n   #7  futhark/microgpt.fut:454:26-460:31\n   #8  futhark/microgpt.fut:476:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_85299 = ((double *) mem_87865)[tmp_85289 * (int64_t) 64 + i_86496 * (int64_t) 4 + tmp_85294];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_85312 = ((double *) mem_87990)[tmp_85289 * (int64_t) 64 + i_86496 * (int64_t) 4 + tmp_85294];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_85328 = ((double *) mem_87989)[tmp_85289 * (int64_t) 64 + i_86496 * (int64_t) 4 + tmp_85294];
                
                ((double *) mem_88058)[i_86486] = lifted_lambda_res_85328;
                ((double *) mem_88059)[i_86486] = lifted_lambda_res_85312;
                ((double *) mem_88060)[i_86486] = lifted_lambda_res_85299;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88043, i_86496 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88058, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88044, i_86496 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88059, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88045, i_86496 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88060, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86521 = 0; i_86521 < (int64_t) 16; i_86521++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86508 = 0; i_86508 < (int64_t) 16; i_86508++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85491;
                double r_85493 = 0.0;
                
                for (int64_t i_85492 = 0; i_85492 < (int64_t) 16; i_85492++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_85494 = ((double *) mem_param_86898.mem)[i_85492 * (int64_t) 16 + i_86508];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_85495 = ((double *) mem_88045)[i_86521 * (int64_t) 16 + i_85492];
                    
                    // futhark/microgpt.fut:308:69-114
                    
                    double zt_res_85496 = zt_lhs_85494 * zt_rhs_85495;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85497 = r_85493 + zt_res_85496;
                    double r_tmp_88870 = zp_res_85497;
                    
                    r_85493 = r_tmp_88870;
                }
                defunc_0_lifted_lambda_res_85491 = r_85493;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85498;
                double r_85500 = 0.0;
                
                for (int64_t i_85499 = 0; i_85499 < (int64_t) 16; i_85499++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_85501 = ((double *) mem_param_86874.mem)[i_85499 * (int64_t) 16 + i_86508];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_85502 = ((double *) mem_88044)[i_86521 * (int64_t) 16 + i_85499];
                    
                    // futhark/microgpt.fut:308:145-190
                    
                    double zt_res_85503 = zt_lhs_85501 * zt_rhs_85502;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85504 = r_85500 + zt_res_85503;
                    double r_tmp_88871 = zp_res_85504;
                    
                    r_85500 = r_tmp_88871;
                }
                defunc_0_lifted_lambda_res_85498 = r_85500;
                // futhark/microgpt.fut:308:47-192
                
                double zp_res_85505 = defunc_0_lifted_lambda_res_85491 + defunc_0_lifted_lambda_res_85498;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85506;
                double r_85508 = 0.0;
                
                for (int64_t i_85507 = 0; i_85507 < (int64_t) 16; i_85507++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_85509 = ((double *) mem_param_86886.mem)[i_85507 * (int64_t) 16 + i_86508];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_85510 = ((double *) mem_88043)[i_86521 * (int64_t) 16 + i_85507];
                    
                    // futhark/microgpt.fut:308:222-267
                    
                    double zt_res_85511 = zt_lhs_85509 * zt_rhs_85510;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85512 = r_85508 + zt_res_85511;
                    double r_tmp_88872 = zp_res_85512;
                    
                    r_85508 = r_tmp_88872;
                }
                defunc_0_lifted_lambda_res_85506 = r_85508;
                // futhark/microgpt.fut:308:118-269
                
                double zp_res_85513 = zp_res_85505 + defunc_0_lifted_lambda_res_85506;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85520;
                double r_85522 = 0.0;
                
                for (int64_t i_85521 = 0; i_85521 < (int64_t) 16; i_85521++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_85523 = ((double *) mem_88043)[i_85521 * (int64_t) 16 + i_86521];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_85524 = ((double *) mem_87073)[i_85521 * (int64_t) 16 + i_86508];
                    
                    // futhark/microgpt.fut:326:68-111
                    
                    double zt_res_85525 = zt_lhs_85523 * zt_rhs_85524;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85526 = r_85522 + zt_res_85525;
                    double r_tmp_88873 = zp_res_85526;
                    
                    r_85522 = r_tmp_88873;
                }
                defunc_0_lifted_lambda_res_85520 = r_85522;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85536;
                double r_85538 = 0.0;
                
                for (int64_t i_85537 = 0; i_85537 < (int64_t) 16; i_85537++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_85539 = ((double *) mem_88044)[i_85537 * (int64_t) 16 + i_86521];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_85540 = ((double *) mem_87073)[i_85537 * (int64_t) 16 + i_86508];
                    
                    // futhark/microgpt.fut:327:68-111
                    
                    double zt_res_85541 = zt_lhs_85539 * zt_rhs_85540;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85542 = r_85538 + zt_res_85541;
                    double r_tmp_88874 = zp_res_85542;
                    
                    r_85538 = r_tmp_88874;
                }
                defunc_0_lifted_lambda_res_85536 = r_85538;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85554;
                double r_85556 = 0.0;
                
                for (int64_t i_85555 = 0; i_85555 < (int64_t) 16; i_85555++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_85557 = ((double *) mem_88045)[i_85555 * (int64_t) 16 + i_86521];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_85558 = ((double *) mem_87073)[i_85555 * (int64_t) 16 + i_86508];
                    
                    // futhark/microgpt.fut:328:68-111
                    
                    double zt_res_85559 = zt_lhs_85557 * zt_rhs_85558;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85560 = r_85556 + zt_res_85559;
                    double r_tmp_88875 = zp_res_85560;
                    
                    r_85556 = r_tmp_88875;
                }
                defunc_0_lifted_lambda_res_85554 = r_85556;
                ((double *) mem_88111)[i_86508] = defunc_0_lifted_lambda_res_85554;
                ((double *) mem_88112)[i_86508] = defunc_0_lifted_lambda_res_85536;
                ((double *) mem_88113)[i_86508] = defunc_0_lifted_lambda_res_85520;
                ((double *) mem_88114)[i_86508] = zp_res_85513;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88091, i_86521 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88111, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88092, i_86521 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88112, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88093, i_86521 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88113, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88094, i_86521 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88114, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86530 = 0; i_86530 < (int64_t) 16; i_86530++) {
            // futhark/microgpt.fut:311:43-55
            
            double zp_lhs_82240 = ((double *) mem_87110)[i_86530];
            
            // futhark/microgpt.fut:311:43-83
            
            double zp_res_82241 = 1.0e-5 + zp_lhs_82240;
            
            // futhark/microgpt.fut:311:35-83
            
            double sqrt_res_82242 = futrts_sqrt64(zp_res_82241);
            
            // futhark/microgpt.fut:312:65-85
            
            double zs_res_82250 = 1.0 / sqrt_res_82242;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_82251;
            double r_82253 = 0.0;
            
            for (int64_t i_82252 = 0; i_82252 < (int64_t) 16; i_82252++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_82254 = ((double *) mem_87040)[i_86530 * (int64_t) 16 + i_82252];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_82255 = ((double *) mem_88094)[i_86530 * (int64_t) 16 + i_82252];
                
                // futhark/microgpt.fut:312:93-136
                
                double zt_res_82256 = zt_lhs_82254 * zt_rhs_82255;
                
                // futhark/microgpt.fut:312:112-163
                
                double zt_res_82257 = zs_res_82250 * zt_res_82256;
                
                // futhark/microgpt.fut:312:69-163
                
                double zt_res_82258 = zs_res_82250 * zt_res_82257;
                
                // futhark/microgpt.fut:312:57-163
                
                double neg_res_82259 = -zt_res_82258;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_82260 = r_82253 + neg_res_82259;
                double r_tmp_88878 = zp_res_82260;
                
                r_82253 = r_tmp_88878;
            }
            defunc_0_lifted_lambda_res_82251 = r_82253;
            ((double *) mem_88155)[i_86530] = defunc_0_lifted_lambda_res_82251;
            ((double *) mem_88156)[i_86530] = sqrt_res_82242;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86535 = 0; i_86535 < (int64_t) 16; i_86535++) {
            // futhark/microgpt.fut:313:35-47
            
            double zt_lhs_81813 = ((double *) mem_88155)[i_86535];
            
            // futhark/microgpt.fut:313:89-101
            
            double zp_lhs_81814 = ((double *) mem_87110)[i_86535];
            
            // futhark/microgpt.fut:313:89-129
            
            double zp_res_81815 = 1.0e-5 + zp_lhs_81814;
            
            // futhark/microgpt.fut:313:81-129
            
            double sqrt_res_81816 = futrts_sqrt64(zp_res_81815);
            
            // futhark/microgpt.fut:313:67-131
            
            double zt_res_81817 = 2.0 * sqrt_res_81816;
            
            // futhark/microgpt.fut:313:53-131
            
            double zs_res_81818 = 1.0 / zt_res_81817;
            
            // futhark/microgpt.fut:313:35-131
            
            double zt_res_81819 = zt_lhs_81813 * zs_res_81818;
            
            ((double *) mem_88169)[i_86535] = zt_res_81819;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86539 = 0; i_86539 < (int64_t) 16; i_86539++) {
            // futhark/microgpt.fut:314:45-57
            
            double zs_lhs_81827 = ((double *) mem_88169)[i_86539];
            
            // futhark/microgpt.fut:314:45-72
            
            double zs_res_81828 = zs_lhs_81827 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_88881 = 0; nest_i_88881 < (int64_t) 16; nest_i_88881++) {
                ((double *) mem_88176)[i_86539 * (int64_t) 16 + nest_i_88881] = zs_res_81828;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86547 = 0; i_86547 < (int64_t) 16; i_86547++) {
            // futhark/microgpt.fut:315:107-119
            
            double zs_rhs_81837 = ((double *) mem_88156)[i_86547];
            
            // futhark/microgpt.fut:315:99-119
            
            double zs_res_81838 = 1.0 / zs_rhs_81837;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86543 = 0; i_86543 < (int64_t) 16; i_86543++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_81845 = ((double *) mem_87702)[i_86547 * (int64_t) 16 + i_86543];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_81846 = ((double *) mem_88094)[i_86547 * (int64_t) 16 + i_86543];
                
                // futhark/microgpt.fut:315:73-119
                
                double zt_res_81847 = zs_res_81838 * zt_lhs_81846;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_81848 = ((double *) mem_87040)[i_86547 * (int64_t) 16 + i_86543];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_81849 = ((double *) mem_88176)[i_86547 * (int64_t) 16 + i_86543];
                
                // futhark/microgpt.fut:315:127-170
                
                double zt_res_81850 = zt_lhs_81848 * zt_rhs_81849;
                
                // futhark/microgpt.fut:315:94-170
                
                double zp_res_81851 = zt_res_81847 + zt_res_81850;
                
                // futhark/microgpt.fut:315:122-221
                
                double zp_res_81852 = zt_res_81850 + zp_res_81851;
                
                // futhark/microgpt.fut:315:45-221
                
                double zp_res_81853 = zp_lhs_81845 + zp_res_81852;
                
                ((double *) mem_88191)[i_86543] = zp_res_81853;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88186, i_86547 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88191, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86553 = 0; i_86553 < (int64_t) 16; i_86553++) {
            // futhark/microgpt.fut:318:43-55
            
            double zp_lhs_82202 = ((double *) mem_87071)[i_86553];
            
            // futhark/microgpt.fut:318:43-83
            
            double zp_res_82203 = 1.0e-5 + zp_lhs_82202;
            
            // futhark/microgpt.fut:318:35-83
            
            double sqrt_res_82204 = futrts_sqrt64(zp_res_82203);
            
            // futhark/microgpt.fut:319:65-85
            
            double zs_res_82212 = 1.0 / sqrt_res_82204;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_82213;
            double r_82215 = 0.0;
            
            for (int64_t i_82214 = 0; i_82214 < (int64_t) 16; i_82214++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_82216 = ((double *) mem_87008)[i_86553 * (int64_t) 16 + i_82214];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_82217 = ((double *) mem_88186)[i_86553 * (int64_t) 16 + i_82214];
                
                // futhark/microgpt.fut:319:93-136
                
                double zt_res_82218 = zt_lhs_82216 * zt_rhs_82217;
                
                // futhark/microgpt.fut:319:112-163
                
                double zt_res_82219 = zs_res_82212 * zt_res_82218;
                
                // futhark/microgpt.fut:319:69-163
                
                double zt_res_82220 = zs_res_82212 * zt_res_82219;
                
                // futhark/microgpt.fut:319:57-163
                
                double neg_res_82221 = -zt_res_82220;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_82222 = r_82215 + neg_res_82221;
                double r_tmp_88886 = zp_res_82222;
                
                r_82215 = r_tmp_88886;
            }
            defunc_0_lifted_lambda_res_82213 = r_82215;
            ((double *) mem_88202)[i_86553] = defunc_0_lifted_lambda_res_82213;
            ((double *) mem_88203)[i_86553] = sqrt_res_82204;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86558 = 0; i_86558 < (int64_t) 16; i_86558++) {
            // futhark/microgpt.fut:320:35-47
            
            double zt_lhs_81920 = ((double *) mem_88202)[i_86558];
            
            // futhark/microgpt.fut:320:89-101
            
            double zp_lhs_81921 = ((double *) mem_87071)[i_86558];
            
            // futhark/microgpt.fut:320:89-129
            
            double zp_res_81922 = 1.0e-5 + zp_lhs_81921;
            
            // futhark/microgpt.fut:320:81-129
            
            double sqrt_res_81923 = futrts_sqrt64(zp_res_81922);
            
            // futhark/microgpt.fut:320:67-131
            
            double zt_res_81924 = 2.0 * sqrt_res_81923;
            
            // futhark/microgpt.fut:320:53-131
            
            double zs_res_81925 = 1.0 / zt_res_81924;
            
            // futhark/microgpt.fut:320:35-131
            
            double zt_res_81926 = zt_lhs_81920 * zs_res_81925;
            
            ((double *) mem_88216)[i_86558] = zt_res_81926;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86562 = 0; i_86562 < (int64_t) 16; i_86562++) {
            // futhark/microgpt.fut:321:45-57
            
            double zs_lhs_81934 = ((double *) mem_88216)[i_86562];
            
            // futhark/microgpt.fut:321:45-72
            
            double zs_res_81935 = zs_lhs_81934 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_88889 = 0; nest_i_88889 < (int64_t) 16; nest_i_88889++) {
                ((double *) mem_88223)[i_86562 * (int64_t) 16 + nest_i_88889] = zs_res_81935;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86570 = 0; i_86570 < (int64_t) 16; i_86570++) {
            // futhark/microgpt.fut:322:81-93
            
            double zs_rhs_81944 = ((double *) mem_88203)[i_86570];
            
            // futhark/microgpt.fut:322:73-93
            
            double zs_res_81945 = 1.0 / zs_rhs_81944;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86566 = 0; i_86566 < (int64_t) 16; i_86566++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_81952 = ((double *) mem_88186)[i_86570 * (int64_t) 16 + i_86566];
                
                // futhark/microgpt.fut:322:47-93
                
                double zt_res_81953 = zs_res_81945 * zt_lhs_81952;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_81954 = ((double *) mem_87008)[i_86570 * (int64_t) 16 + i_86566];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_81955 = ((double *) mem_88223)[i_86570 * (int64_t) 16 + i_86566];
                
                // futhark/microgpt.fut:322:101-144
                
                double zt_res_81956 = zt_lhs_81954 * zt_rhs_81955;
                
                // futhark/microgpt.fut:322:68-144
                
                double zp_res_81957 = zt_res_81953 + zt_res_81956;
                
                // futhark/microgpt.fut:322:96-195
                
                double zp_res_81958 = zt_res_81956 + zp_res_81957;
                
                ((double *) mem_88238)[i_86566] = zp_res_81958;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88233, i_86570 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88238, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86583 = 0; i_86583 < (int64_t) 16; i_86583++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86576 = 0; i_86576 < (int64_t) 16; i_86576++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_85586 = ((double *) mem_88233)[i_86583 * (int64_t) 16 + i_86576];
                
                ((double *) mem_88259)[i_86576] = lifted_lambda_res_85586;
                ((double *) mem_88260)[i_86576] = lifted_lambda_res_85586;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88249, i_86583 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88259, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88250, i_86583 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88260, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86592 = 0; i_86592 < (int64_t) 64; i_86592++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86588 = 0; i_86588 < (int64_t) 16; i_86588++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_82072;
                double r_82074 = 0.0;
                
                for (int64_t i_82073 = 0; i_82073 < (int64_t) 16; i_82073++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_82075 = ((double *) mem_87639)[i_82073 * (int64_t) 64 + i_86592];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_82076 = ((double *) mem_87404)[i_82073 * (int64_t) 16 + i_86588];
                    
                    // futhark/microgpt.fut:330:67-110
                    
                    double zt_res_82077 = zt_lhs_82075 * zt_rhs_82076;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82078 = r_82074 + zt_res_82077;
                    double r_tmp_88898 = zp_res_82078;
                    
                    r_82074 = r_tmp_88898;
                }
                defunc_0_lifted_lambda_res_82072 = r_82074;
                ((double *) mem_88286)[i_86588] = defunc_0_lifted_lambda_res_82072;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88281, i_86592 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88286, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86605 = 0; i_86605 < (int64_t) 27; i_86605++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86598 = 0; i_86598 < (int64_t) 16; i_86598++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85614;
                double r_85616 = 0.0;
                
                for (int64_t i_85615 = 0; i_85615 < (int64_t) 16; i_85615++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_85617 = ((double *) mem_87575)[i_85615 * (int64_t) 27 + i_86605];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_85618 = ((double *) mem_87490)[i_85615 * (int64_t) 16 + i_86598];
                    
                    // futhark/microgpt.fut:332:68-111
                    
                    double zt_res_85619 = zt_lhs_85617 * zt_rhs_85618;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85620 = r_85616 + zt_res_85619;
                    double r_tmp_88903 = zp_res_85620;
                    
                    r_85616 = r_tmp_88903;
                }
                defunc_0_lifted_lambda_res_85614 = r_85616;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85623;
                double r_85625 = 0.0;
                
                for (int64_t i_85624 = 0; i_85624 < (int64_t) 16; i_85624++) {
                    int64_t zeze_lhs_85626 = ((int64_t *) seqs_mem_86866.mem)[step_80406 * (int64_t) 16 + i_85624];
                    
                    // futhark/microgpt.fut:450:58-109
                    
                    bool cond_85627 = zeze_lhs_85626 == i_86605;
                    
                    // futhark/microgpt.fut:450:58-109
                    
                    double lifted_lambda_res_85628;
                    
                    if (cond_85627) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double lifted_lambda_res_t_res_85918 = ((double *) mem_88249)[i_85624 * (int64_t) 16 + i_86598];
                        
                        lifted_lambda_res_85628 = lifted_lambda_res_t_res_85918;
                    } else {
                        lifted_lambda_res_85628 = 0.0;
                    }
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85634 = r_85625 + lifted_lambda_res_85628;
                    double r_tmp_88904 = zp_res_85634;
                    
                    r_85625 = r_tmp_88904;
                }
                defunc_0_lifted_lambda_res_85623 = r_85625;
                ((double *) mem_88307)[i_86598] = defunc_0_lifted_lambda_res_85623;
                ((double *) mem_88308)[i_86598] = defunc_0_lifted_lambda_res_85614;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88297, i_86605 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88307, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88298, i_86605 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88308, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:66:26-45
        
        double i64_res_82156 = sitofp_i64_f64(step_80406);
        
        // futhark/microgpt.fut:406:46-65
        
        double zm_rhs_82157 = i64_res_82156 / 500.0;
        
        // futhark/microgpt.fut:406:24-65
        
        double zt_rhs_82158 = 1.0 - zm_rhs_82157;
        
        // futhark/microgpt.fut:406:19-65
        
        double lt_r_82159 = 1.0e-2 * zt_rhs_82158;
        
        // futhark/microgpt.fut:408:5-52
        if (memblock_alloc(ctx, &mem_88329, (int64_t) 3456, "mem_88329")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:408:5-52
        // futhark/microgpt.fut:408:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88329.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86890.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:408:5-52
        if (memblock_alloc(ctx, &mem_88331, (int64_t) 3456, "mem_88331")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:408:5-52
        // futhark/microgpt.fut:408:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88331.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86926.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:408:5-52
        if (memblock_alloc(ctx, &mem_88333, (int64_t) 3456, "mem_88333")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:408:5-52
        // futhark/microgpt.fut:408:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88333.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86962.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:408:5-52
        if (memblock_alloc(ctx, &mem_88335, (int64_t) 3456, "mem_88335")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:408:5-52
        // futhark/microgpt.fut:408:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88335.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_88297, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:408:5-52
        if (futrts_adam_opt_w_10203(ctx, &ext_mem_88339, &ext_mem_88338, &ext_mem_88337, mem_88329, mem_88331, mem_88333, mem_88335, (int64_t) 27, (int64_t) 16, step_80406, lt_r_82159) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_88329, "mem_88329") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88331, "mem_88331") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88333, "mem_88333") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88335, "mem_88335") != 0)
            return 1;
        // futhark/microgpt.fut:410:5-52
        if (memblock_alloc(ctx, &mem_88340, (int64_t) 2048, "mem_88340")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:410:5-52
        // futhark/microgpt.fut:410:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88340.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86882.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:410:5-52
        if (memblock_alloc(ctx, &mem_88342, (int64_t) 2048, "mem_88342")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:410:5-52
        // futhark/microgpt.fut:410:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88342.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86918.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:410:5-52
        if (memblock_alloc(ctx, &mem_88344, (int64_t) 2048, "mem_88344")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:410:5-52
        // futhark/microgpt.fut:410:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88344.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86954.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:410:5-52
        if (memblock_alloc(ctx, &mem_88346, (int64_t) 2048, "mem_88346")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:410:5-52
        // futhark/microgpt.fut:410:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88346.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_88250, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:410:5-52
        if (futrts_adam_opt_w_10204(ctx, &ext_mem_88350, &ext_mem_88349, &ext_mem_88348, mem_88340, mem_88342, mem_88344, mem_88346, (int64_t) 16, (int64_t) 16, step_80406, lt_r_82159) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_88340, "mem_88340") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88342, "mem_88342") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88344, "mem_88344") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88346, "mem_88346") != 0)
            return 1;
        // futhark/microgpt.fut:412:5-56
        if (memblock_alloc(ctx, &mem_88351, (int64_t) 2048, "mem_88351")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:412:5-56
        // futhark/microgpt.fut:412:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88351.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86886.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:412:5-56
        if (memblock_alloc(ctx, &mem_88353, (int64_t) 2048, "mem_88353")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:412:5-56
        // futhark/microgpt.fut:412:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88353.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86922.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:412:5-56
        if (memblock_alloc(ctx, &mem_88355, (int64_t) 2048, "mem_88355")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:412:5-56
        // futhark/microgpt.fut:412:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88355.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86958.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:412:5-56
        if (memblock_alloc(ctx, &mem_88357, (int64_t) 2048, "mem_88357")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:412:5-56
        // futhark/microgpt.fut:412:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88357.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_88093, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:412:5-56
        if (futrts_adam_opt_w_10204(ctx, &ext_mem_88361, &ext_mem_88360, &ext_mem_88359, mem_88351, mem_88353, mem_88355, mem_88357, (int64_t) 16, (int64_t) 16, step_80406, lt_r_82159) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_88351, "mem_88351") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88353, "mem_88353") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88355, "mem_88355") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88357, "mem_88357") != 0)
            return 1;
        // futhark/microgpt.fut:414:5-56
        if (memblock_alloc(ctx, &mem_88362, (int64_t) 2048, "mem_88362")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:414:5-56
        // futhark/microgpt.fut:414:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88362.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86874.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:414:5-56
        if (memblock_alloc(ctx, &mem_88364, (int64_t) 2048, "mem_88364")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:414:5-56
        // futhark/microgpt.fut:414:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88364.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86910.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:414:5-56
        if (memblock_alloc(ctx, &mem_88366, (int64_t) 2048, "mem_88366")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:414:5-56
        // futhark/microgpt.fut:414:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88366.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86946.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:414:5-56
        if (memblock_alloc(ctx, &mem_88368, (int64_t) 2048, "mem_88368")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:414:5-56
        // futhark/microgpt.fut:414:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88368.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_88092, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:414:5-56
        if (futrts_adam_opt_w_10204(ctx, &ext_mem_88372, &ext_mem_88371, &ext_mem_88370, mem_88362, mem_88364, mem_88366, mem_88368, (int64_t) 16, (int64_t) 16, step_80406, lt_r_82159) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_88362, "mem_88362") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88364, "mem_88364") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88366, "mem_88366") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88368, "mem_88368") != 0)
            return 1;
        // futhark/microgpt.fut:416:5-56
        if (memblock_alloc(ctx, &mem_88373, (int64_t) 2048, "mem_88373")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:416:5-56
        // futhark/microgpt.fut:416:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88373.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86898.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:416:5-56
        if (memblock_alloc(ctx, &mem_88375, (int64_t) 2048, "mem_88375")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:416:5-56
        // futhark/microgpt.fut:416:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88375.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86934.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:416:5-56
        if (memblock_alloc(ctx, &mem_88377, (int64_t) 2048, "mem_88377")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:416:5-56
        // futhark/microgpt.fut:416:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88377.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86970.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:416:5-56
        if (memblock_alloc(ctx, &mem_88379, (int64_t) 2048, "mem_88379")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:416:5-56
        // futhark/microgpt.fut:416:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88379.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_88091, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:416:5-56
        if (futrts_adam_opt_w_10204(ctx, &ext_mem_88383, &ext_mem_88382, &ext_mem_88381, mem_88373, mem_88375, mem_88377, mem_88379, (int64_t) 16, (int64_t) 16, step_80406, lt_r_82159) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_88373, "mem_88373") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88375, "mem_88375") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88377, "mem_88377") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88379, "mem_88379") != 0)
            return 1;
        // futhark/microgpt.fut:418:5-56
        if (memblock_alloc(ctx, &mem_88384, (int64_t) 2048, "mem_88384")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:418:5-56
        // futhark/microgpt.fut:418:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88384.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86878.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:418:5-56
        if (memblock_alloc(ctx, &mem_88386, (int64_t) 2048, "mem_88386")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:418:5-56
        // futhark/microgpt.fut:418:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88386.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86914.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:418:5-56
        if (memblock_alloc(ctx, &mem_88388, (int64_t) 2048, "mem_88388")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:418:5-56
        // futhark/microgpt.fut:418:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88388.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86950.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:418:5-56
        if (memblock_alloc(ctx, &mem_88390, (int64_t) 2048, "mem_88390")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:418:5-56
        // futhark/microgpt.fut:418:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88390.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_87718, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:418:5-56
        if (futrts_adam_opt_w_10204(ctx, &ext_mem_88394, &ext_mem_88393, &ext_mem_88392, mem_88384, mem_88386, mem_88388, mem_88390, (int64_t) 16, (int64_t) 16, step_80406, lt_r_82159) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_88384, "mem_88384") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88386, "mem_88386") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88388, "mem_88388") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88390, "mem_88390") != 0)
            return 1;
        // futhark/microgpt.fut:420:5-52
        if (memblock_alloc(ctx, &mem_88395, (int64_t) 8192, "mem_88395")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:420:5-52
        // futhark/microgpt.fut:420:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88395.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86894.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:420:5-52
        if (memblock_alloc(ctx, &mem_88397, (int64_t) 8192, "mem_88397")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:420:5-52
        // futhark/microgpt.fut:420:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88397.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86930.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:420:5-52
        if (memblock_alloc(ctx, &mem_88399, (int64_t) 8192, "mem_88399")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:420:5-52
        // futhark/microgpt.fut:420:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88399.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86966.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:420:5-52
        if (memblock_alloc(ctx, &mem_88401, (int64_t) 8192, "mem_88401")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:420:5-52
        // futhark/microgpt.fut:420:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88401.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_88281, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:420:5-52
        if (futrts_adam_opt_w_10203(ctx, &ext_mem_88405, &ext_mem_88404, &ext_mem_88403, mem_88395, mem_88397, mem_88399, mem_88401, (int64_t) 64, (int64_t) 16, step_80406, lt_r_82159) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_88395, "mem_88395") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88397, "mem_88397") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88399, "mem_88399") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88401, "mem_88401") != 0)
            return 1;
        // futhark/microgpt.fut:422:5-60
        if (memblock_alloc(ctx, &mem_88406, (int64_t) 8192, "mem_88406")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:422:5-60
        // futhark/microgpt.fut:422:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88406.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_86870.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:422:5-60
        if (memblock_alloc(ctx, &mem_88408, (int64_t) 8192, "mem_88408")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:422:5-60
        // futhark/microgpt.fut:422:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88408.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_86906.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:422:5-60
        if (memblock_alloc(ctx, &mem_88410, (int64_t) 8192, "mem_88410")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:422:5-60
        // futhark/microgpt.fut:422:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88410.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_86942.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:422:5-60
        if (memblock_alloc(ctx, &mem_88412, (int64_t) 8192, "mem_88412")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:422:5-60
        // futhark/microgpt.fut:422:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88412.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_87607, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:422:5-60
        if (futrts_adam_opt_w_10203(ctx, &ext_mem_88416, &ext_mem_88415, &ext_mem_88414, mem_88406, mem_88408, mem_88410, mem_88412, (int64_t) 16, (int64_t) 64, step_80406, lt_r_82159) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_88406, "mem_88406") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88408, "mem_88408") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88410, "mem_88410") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88412, "mem_88412") != 0)
            return 1;
        // futhark/microgpt.fut:424:5-56
        if (memblock_alloc(ctx, &mem_88417, (int64_t) 3456, "mem_88417")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:424:5-56
        // futhark/microgpt.fut:424:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88417.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86902.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:424:5-56
        if (memblock_alloc(ctx, &mem_88419, (int64_t) 3456, "mem_88419")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:424:5-56
        // futhark/microgpt.fut:424:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88419.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86938.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:424:5-56
        if (memblock_alloc(ctx, &mem_88421, (int64_t) 3456, "mem_88421")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:424:5-56
        // futhark/microgpt.fut:424:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88421.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_86974.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:424:5-56
        if (memblock_alloc(ctx, &mem_88423, (int64_t) 3456, "mem_88423")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:424:5-56
        // futhark/microgpt.fut:424:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88423.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_88298, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:424:5-56
        if (futrts_adam_opt_w_10203(ctx, &ext_mem_88427, &ext_mem_88426, &ext_mem_88425, mem_88417, mem_88419, mem_88421, mem_88423, (int64_t) 27, (int64_t) 16, step_80406, lt_r_82159) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_88417, "mem_88417") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88419, "mem_88419") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88421, "mem_88421") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88423, "mem_88423") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88635, &ext_mem_88416, "ext_mem_88416") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88636, &ext_mem_88372, "ext_mem_88372") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88637, &ext_mem_88394, "ext_mem_88394") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88638, &ext_mem_88350, "ext_mem_88350") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88639, &ext_mem_88361, "ext_mem_88361") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88640, &ext_mem_88339, "ext_mem_88339") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88641, &ext_mem_88405, "ext_mem_88405") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88642, &ext_mem_88383, "ext_mem_88383") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88643, &ext_mem_88427, "ext_mem_88427") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88644, &ext_mem_88415, "ext_mem_88415") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88645, &ext_mem_88371, "ext_mem_88371") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88646, &ext_mem_88393, "ext_mem_88393") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88647, &ext_mem_88349, "ext_mem_88349") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88648, &ext_mem_88360, "ext_mem_88360") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88649, &ext_mem_88338, "ext_mem_88338") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88650, &ext_mem_88404, "ext_mem_88404") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88651, &ext_mem_88382, "ext_mem_88382") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88652, &ext_mem_88426, "ext_mem_88426") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88653, &ext_mem_88414, "ext_mem_88414") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88654, &ext_mem_88370, "ext_mem_88370") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88655, &ext_mem_88392, "ext_mem_88392") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88656, &ext_mem_88348, "ext_mem_88348") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88657, &ext_mem_88359, "ext_mem_88359") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88658, &ext_mem_88337, "ext_mem_88337") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88659, &ext_mem_88403, "ext_mem_88403") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88660, &ext_mem_88381, "ext_mem_88381") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_88661, &ext_mem_88425, "ext_mem_88425") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86870, &mem_param_tmp_88635, "mem_param_tmp_88635") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86874, &mem_param_tmp_88636, "mem_param_tmp_88636") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86878, &mem_param_tmp_88637, "mem_param_tmp_88637") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86882, &mem_param_tmp_88638, "mem_param_tmp_88638") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86886, &mem_param_tmp_88639, "mem_param_tmp_88639") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86890, &mem_param_tmp_88640, "mem_param_tmp_88640") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86894, &mem_param_tmp_88641, "mem_param_tmp_88641") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86898, &mem_param_tmp_88642, "mem_param_tmp_88642") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86902, &mem_param_tmp_88643, "mem_param_tmp_88643") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86906, &mem_param_tmp_88644, "mem_param_tmp_88644") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86910, &mem_param_tmp_88645, "mem_param_tmp_88645") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86914, &mem_param_tmp_88646, "mem_param_tmp_88646") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86918, &mem_param_tmp_88647, "mem_param_tmp_88647") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86922, &mem_param_tmp_88648, "mem_param_tmp_88648") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86926, &mem_param_tmp_88649, "mem_param_tmp_88649") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86930, &mem_param_tmp_88650, "mem_param_tmp_88650") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86934, &mem_param_tmp_88651, "mem_param_tmp_88651") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86938, &mem_param_tmp_88652, "mem_param_tmp_88652") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86942, &mem_param_tmp_88653, "mem_param_tmp_88653") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86946, &mem_param_tmp_88654, "mem_param_tmp_88654") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86950, &mem_param_tmp_88655, "mem_param_tmp_88655") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86954, &mem_param_tmp_88656, "mem_param_tmp_88656") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86958, &mem_param_tmp_88657, "mem_param_tmp_88657") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86962, &mem_param_tmp_88658, "mem_param_tmp_88658") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86966, &mem_param_tmp_88659, "mem_param_tmp_88659") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86970, &mem_param_tmp_88660, "mem_param_tmp_88660") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_86974, &mem_param_tmp_88661, "mem_param_tmp_88661") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_88535, &mem_param_86870, "mem_param_86870") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88534, &mem_param_86874, "mem_param_86874") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88533, &mem_param_86878, "mem_param_86878") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88532, &mem_param_86882, "mem_param_86882") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88531, &mem_param_86886, "mem_param_86886") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88530, &mem_param_86890, "mem_param_86890") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88529, &mem_param_86894, "mem_param_86894") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88528, &mem_param_86898, "mem_param_86898") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88527, &mem_param_86902, "mem_param_86902") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88526, &mem_param_86906, "mem_param_86906") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88525, &mem_param_86910, "mem_param_86910") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88524, &mem_param_86914, "mem_param_86914") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88523, &mem_param_86918, "mem_param_86918") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88522, &mem_param_86922, "mem_param_86922") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88521, &mem_param_86926, "mem_param_86926") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88520, &mem_param_86930, "mem_param_86930") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88519, &mem_param_86934, "mem_param_86934") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88518, &mem_param_86938, "mem_param_86938") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88517, &mem_param_86942, "mem_param_86942") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88516, &mem_param_86946, "mem_param_86946") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88515, &mem_param_86950, "mem_param_86950") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88514, &mem_param_86954, "mem_param_86954") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88513, &mem_param_86958, "mem_param_86958") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88512, &mem_param_86962, "mem_param_86962") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88511, &mem_param_86966, "mem_param_86966") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88510, &mem_param_86970, "mem_param_86970") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_88509, &mem_param_86974, "mem_param_86974") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88608, &ext_mem_88530, "ext_mem_88530") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88609, &ext_mem_88532, "ext_mem_88532") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88610, &ext_mem_88531, "ext_mem_88531") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88611, &ext_mem_88534, "ext_mem_88534") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88612, &ext_mem_88528, "ext_mem_88528") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88613, &ext_mem_88533, "ext_mem_88533") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88614, &ext_mem_88529, "ext_mem_88529") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88615, &ext_mem_88535, "ext_mem_88535") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88616, &ext_mem_88527, "ext_mem_88527") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88617, &ext_mem_88521, "ext_mem_88521") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88618, &ext_mem_88523, "ext_mem_88523") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88619, &ext_mem_88522, "ext_mem_88522") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88620, &ext_mem_88525, "ext_mem_88525") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88621, &ext_mem_88519, "ext_mem_88519") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88622, &ext_mem_88524, "ext_mem_88524") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88623, &ext_mem_88520, "ext_mem_88520") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88624, &ext_mem_88526, "ext_mem_88526") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88625, &ext_mem_88518, "ext_mem_88518") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88626, &ext_mem_88512, "ext_mem_88512") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88627, &ext_mem_88514, "ext_mem_88514") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88628, &ext_mem_88513, "ext_mem_88513") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88629, &ext_mem_88516, "ext_mem_88516") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88630, &ext_mem_88510, "ext_mem_88510") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88631, &ext_mem_88515, "ext_mem_88515") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88632, &ext_mem_88511, "ext_mem_88511") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88633, &ext_mem_88517, "ext_mem_88517") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88634, &ext_mem_88509, "ext_mem_88509") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88996, &mem_out_88608, "mem_out_88608") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88997, &mem_out_88609, "mem_out_88609") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88998, &mem_out_88610, "mem_out_88610") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_88999, &mem_out_88611, "mem_out_88611") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89000, &mem_out_88612, "mem_out_88612") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89001, &mem_out_88613, "mem_out_88613") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89002, &mem_out_88614, "mem_out_88614") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89003, &mem_out_88615, "mem_out_88615") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89004, &mem_out_88616, "mem_out_88616") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89005, &mem_out_88617, "mem_out_88617") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89006, &mem_out_88618, "mem_out_88618") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89007, &mem_out_88619, "mem_out_88619") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89008, &mem_out_88620, "mem_out_88620") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89009, &mem_out_88621, "mem_out_88621") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89010, &mem_out_88622, "mem_out_88622") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89011, &mem_out_88623, "mem_out_88623") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89012, &mem_out_88624, "mem_out_88624") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89013, &mem_out_88625, "mem_out_88625") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89014, &mem_out_88626, "mem_out_88626") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89015, &mem_out_88627, "mem_out_88627") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89016, &mem_out_88628, "mem_out_88628") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89017, &mem_out_88629, "mem_out_88629") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89018, &mem_out_88630, "mem_out_88630") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89019, &mem_out_88631, "mem_out_88631") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89020, &mem_out_88632, "mem_out_88632") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89021, &mem_out_88633, "mem_out_88633") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89022, &mem_out_88634, "mem_out_88634") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_86975);
        free(mem_86976);
        free(mem_86985);
        free(mem_86992);
        free(mem_87007);
        free(mem_87008);
        free(mem_87017);
        free(mem_87024);
        free(mem_87039);
        free(mem_87040);
        free(mem_87049);
        free(mem_87050);
        free(mem_87071);
        free(mem_87072);
        free(mem_87073);
        free(mem_87085);
        free(mem_87086);
        free(mem_87110);
        free(mem_87111);
        free(mem_87112);
        free(mem_87113);
        free(mem_87129);
        free(mem_87130);
        free(mem_87131);
        free(mem_87165);
        free(mem_87166);
        free(mem_87167);
        free(mem_87183);
        free(mem_87184);
        free(mem_87185);
        free(mem_87198);
        free(mem_87199);
        free(mem_87200);
        free(mem_87246);
        free(mem_87247);
        free(mem_87258);
        free(mem_87259);
        free(mem_87268);
        free(mem_87269);
        free(mem_87290);
        free(mem_87295);
        free(mem_87306);
        free(mem_87311);
        free(mem_87318);
        free(mem_87329);
        free(mem_87334);
        free(mem_87355);
        free(mem_87360);
        free(mem_87371);
        free(mem_87376);
        free(mem_87387);
        free(mem_87392);
        free(mem_87403);
        free(mem_87404);
        free(mem_87413);
        free(mem_87414);
        free(mem_87435);
        free(mem_87436);
        free(mem_87444);
        free(mem_87458);
        free(mem_87463);
        free(mem_87474);
        free(mem_87479);
        free(mem_87490);
        free(mem_87495);
        free(mem_87506);
        free(mem_87511);
        free(mem_87522);
        free(mem_87523);
        free(mem_87532);
        free(mem_87533);
        free(mem_87546);
        free(mem_87547);
        free(mem_87568);
        free(mem_87575);
        free(mem_87580);
        free(mem_87591);
        free(mem_87596);
        free(mem_87607);
        free(mem_87608);
        free(mem_87617);
        free(mem_87618);
        free(mem_87639);
        free(mem_87644);
        free(mem_87655);
        free(mem_87660);
        free(mem_87671);
        free(mem_87672);
        free(mem_87685);
        free(mem_87692);
        free(mem_87702);
        free(mem_87707);
        free(mem_87718);
        free(mem_87719);
        free(mem_87728);
        free(mem_87729);
        free(mem_87750);
        free(mem_87751);
        free(mem_87762);
        free(mem_87763);
        free(mem_87772);
        free(mem_87779);
        free(mem_87804);
        free(mem_87805);
        free(mem_87816);
        free(mem_87817);
        free(mem_87826);
        free(mem_87833);
        free(mem_87840);
        free(mem_87865);
        free(mem_87866);
        free(mem_87877);
        free(mem_87878);
        free(mem_87887);
        free(mem_87894);
        free(mem_87919);
        free(mem_87924);
        free(mem_87935);
        free(mem_87941);
        free(mem_87946);
        free(mem_87962);
        free(mem_87968);
        free(mem_87973);
        free(mem_87989);
        free(mem_87990);
        free(mem_88001);
        free(mem_88002);
        free(mem_88011);
        free(mem_88012);
        free(mem_88043);
        free(mem_88044);
        free(mem_88045);
        free(mem_88058);
        free(mem_88059);
        free(mem_88060);
        free(mem_88091);
        free(mem_88092);
        free(mem_88093);
        free(mem_88094);
        free(mem_88111);
        free(mem_88112);
        free(mem_88113);
        free(mem_88114);
        free(mem_88155);
        free(mem_88156);
        free(mem_88169);
        free(mem_88176);
        free(mem_88186);
        free(mem_88191);
        free(mem_88202);
        free(mem_88203);
        free(mem_88216);
        free(mem_88223);
        free(mem_88233);
        free(mem_88238);
        free(mem_88249);
        free(mem_88250);
        free(mem_88259);
        free(mem_88260);
        free(mem_88281);
        free(mem_88286);
        free(mem_88297);
        free(mem_88298);
        free(mem_88307);
        free(mem_88308);
        if (memblock_unref(ctx, &mem_param_tmp_88661, "mem_param_tmp_88661") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88660, "mem_param_tmp_88660") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88659, "mem_param_tmp_88659") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88658, "mem_param_tmp_88658") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88657, "mem_param_tmp_88657") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88656, "mem_param_tmp_88656") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88655, "mem_param_tmp_88655") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88654, "mem_param_tmp_88654") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88653, "mem_param_tmp_88653") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88652, "mem_param_tmp_88652") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88651, "mem_param_tmp_88651") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88650, "mem_param_tmp_88650") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88649, "mem_param_tmp_88649") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88648, "mem_param_tmp_88648") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88647, "mem_param_tmp_88647") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88646, "mem_param_tmp_88646") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88645, "mem_param_tmp_88645") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88644, "mem_param_tmp_88644") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88643, "mem_param_tmp_88643") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88642, "mem_param_tmp_88642") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88641, "mem_param_tmp_88641") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88640, "mem_param_tmp_88640") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88639, "mem_param_tmp_88639") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88638, "mem_param_tmp_88638") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88637, "mem_param_tmp_88637") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88636, "mem_param_tmp_88636") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_88635, "mem_param_tmp_88635") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88425, "ext_mem_88425") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88426, "ext_mem_88426") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88427, "ext_mem_88427") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88423, "mem_88423") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88421, "mem_88421") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88419, "mem_88419") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88417, "mem_88417") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88414, "ext_mem_88414") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88415, "ext_mem_88415") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88416, "ext_mem_88416") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88412, "mem_88412") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88410, "mem_88410") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88408, "mem_88408") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88406, "mem_88406") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88403, "ext_mem_88403") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88404, "ext_mem_88404") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88405, "ext_mem_88405") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88401, "mem_88401") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88399, "mem_88399") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88397, "mem_88397") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88395, "mem_88395") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88392, "ext_mem_88392") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88393, "ext_mem_88393") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88394, "ext_mem_88394") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88390, "mem_88390") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88388, "mem_88388") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88386, "mem_88386") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88384, "mem_88384") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88381, "ext_mem_88381") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88382, "ext_mem_88382") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88383, "ext_mem_88383") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88379, "mem_88379") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88377, "mem_88377") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88375, "mem_88375") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88373, "mem_88373") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88370, "ext_mem_88370") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88371, "ext_mem_88371") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88372, "ext_mem_88372") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88368, "mem_88368") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88366, "mem_88366") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88364, "mem_88364") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88362, "mem_88362") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88359, "ext_mem_88359") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88360, "ext_mem_88360") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88361, "ext_mem_88361") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88357, "mem_88357") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88355, "mem_88355") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88353, "mem_88353") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88351, "mem_88351") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88348, "ext_mem_88348") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88349, "ext_mem_88349") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88350, "ext_mem_88350") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88346, "mem_88346") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88344, "mem_88344") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88342, "mem_88342") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88340, "mem_88340") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88337, "ext_mem_88337") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88338, "ext_mem_88338") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88339, "ext_mem_88339") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88335, "mem_88335") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88333, "mem_88333") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88331, "mem_88331") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88329, "mem_88329") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86974, "mem_param_86974") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86970, "mem_param_86970") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86966, "mem_param_86966") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86962, "mem_param_86962") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86958, "mem_param_86958") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86954, "mem_param_86954") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86950, "mem_param_86950") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86946, "mem_param_86946") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86942, "mem_param_86942") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86938, "mem_param_86938") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86934, "mem_param_86934") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86930, "mem_param_86930") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86926, "mem_param_86926") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86922, "mem_param_86922") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86918, "mem_param_86918") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86914, "mem_param_86914") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86910, "mem_param_86910") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86906, "mem_param_86906") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86902, "mem_param_86902") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86898, "mem_param_86898") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86894, "mem_param_86894") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86890, "mem_param_86890") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86886, "mem_param_86886") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86882, "mem_param_86882") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86878, "mem_param_86878") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86874, "mem_param_86874") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_86870, "mem_param_86870") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88509, "ext_mem_88509") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88510, "ext_mem_88510") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88511, "ext_mem_88511") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88512, "ext_mem_88512") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88513, "ext_mem_88513") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88514, "ext_mem_88514") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88515, "ext_mem_88515") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88516, "ext_mem_88516") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88517, "ext_mem_88517") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88518, "ext_mem_88518") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88519, "ext_mem_88519") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88520, "ext_mem_88520") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88521, "ext_mem_88521") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88522, "ext_mem_88522") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88523, "ext_mem_88523") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88524, "ext_mem_88524") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88525, "ext_mem_88525") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88526, "ext_mem_88526") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88527, "ext_mem_88527") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88528, "ext_mem_88528") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88529, "ext_mem_88529") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88530, "ext_mem_88530") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88531, "ext_mem_88531") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88532, "ext_mem_88532") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88533, "ext_mem_88533") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88534, "ext_mem_88534") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_88535, "ext_mem_88535") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88634, "mem_out_88634") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88633, "mem_out_88633") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88632, "mem_out_88632") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88631, "mem_out_88631") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88630, "mem_out_88630") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88629, "mem_out_88629") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88628, "mem_out_88628") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88627, "mem_out_88627") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88626, "mem_out_88626") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88625, "mem_out_88625") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88624, "mem_out_88624") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88623, "mem_out_88623") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88622, "mem_out_88622") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88621, "mem_out_88621") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88620, "mem_out_88620") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88619, "mem_out_88619") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88618, "mem_out_88618") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88617, "mem_out_88617") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88616, "mem_out_88616") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88615, "mem_out_88615") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88614, "mem_out_88614") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88613, "mem_out_88613") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88612, "mem_out_88612") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88611, "mem_out_88611") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88610, "mem_out_88610") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88609, "mem_out_88609") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88608, "mem_out_88608") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_89188, struct memblock *mem_out_p_89189, struct memblock *mem_out_p_89190, struct memblock *mem_out_p_89191, struct memblock *mem_out_p_89192, struct memblock *mem_out_p_89193, struct memblock *mem_out_p_89194, struct memblock *mem_out_p_89195, struct memblock *mem_out_p_89196)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_88616;
    
    mem_out_88616.references = NULL;
    
    struct memblock mem_out_88615;
    
    mem_out_88615.references = NULL;
    
    struct memblock mem_out_88614;
    
    mem_out_88614.references = NULL;
    
    struct memblock mem_out_88613;
    
    mem_out_88613.references = NULL;
    
    struct memblock mem_out_88612;
    
    mem_out_88612.references = NULL;
    
    struct memblock mem_out_88611;
    
    mem_out_88611.references = NULL;
    
    struct memblock mem_out_88610;
    
    mem_out_88610.references = NULL;
    
    struct memblock mem_out_88609;
    
    mem_out_88609.references = NULL;
    
    struct memblock mem_out_88608;
    
    mem_out_88608.references = NULL;
    
    struct memblock mem_86828 = ctx->constants->mem_86828;
    struct memblock mem_86829 = ctx->constants->mem_86829;
    struct memblock mem_86830 = ctx->constants->mem_86830;
    struct memblock mem_86831 = ctx->constants->mem_86831;
    struct memblock mem_86832 = ctx->constants->mem_86832;
    struct memblock mem_86833 = ctx->constants->mem_86833;
    struct memblock mem_86834 = ctx->constants->mem_86834;
    struct memblock mem_86835 = ctx->constants->mem_86835;
    struct memblock mem_86836 = ctx->constants->mem_86836;
    
    if (memblock_set(ctx, &mem_out_88608, &mem_86835, "mem_86835") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88609, &mem_86831, "mem_86831") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88610, &mem_86833, "mem_86833") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88611, &mem_86829, "mem_86829") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88612, &mem_86830, "mem_86830") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88613, &mem_86828, "mem_86828") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88614, &mem_86834, "mem_86834") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88615, &mem_86832, "mem_86832") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_88616, &mem_86836, "mem_86836") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89188, &mem_out_88608, "mem_out_88608") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89189, &mem_out_88609, "mem_out_88609") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89190, &mem_out_88610, "mem_out_88610") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89191, &mem_out_88611, "mem_out_88611") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89192, &mem_out_88612, "mem_out_88612") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89193, &mem_out_88613, "mem_out_88613") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89194, &mem_out_88614, "mem_out_88614") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89195, &mem_out_88615, "mem_out_88615") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89196, &mem_out_88616, "mem_out_88616") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_88616, "mem_out_88616") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88615, "mem_out_88615") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88614, "mem_out_88614") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88613, "mem_out_88613") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88612, "mem_out_88612") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88611, "mem_out_88611") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88610, "mem_out_88610") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88609, "mem_out_88609") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_88608, "mem_out_88608") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_forward_seq(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_88608;
    
    mem_out_88608.references = NULL;
    
    struct memblock mask_mem_86847;
    
    mask_mem_86847.references = NULL;
    
    struct memblock tokens_mem_86846;
    
    tokens_mem_86846.references = NULL;
    
    struct memblock wvoc_mem_86845;
    
    wvoc_mem_86845.references = NULL;
    
    struct memblock wval_mem_86844;
    
    wval_mem_86844.references = NULL;
    
    struct memblock wup_mem_86843;
    
    wup_mem_86843.references = NULL;
    
    struct memblock wte_mem_86842;
    
    wte_mem_86842.references = NULL;
    
    struct memblock wqry_mem_86841;
    
    wqry_mem_86841.references = NULL;
    
    struct memblock wpe_mem_86840;
    
    wpe_mem_86840.references = NULL;
    
    struct memblock wout_mem_86839;
    
    wout_mem_86839.references = NULL;
    
    struct memblock wkey_mem_86838;
    
    wkey_mem_86838.references = NULL;
    
    struct memblock wdown_mem_86837;
    
    wdown_mem_86837.references = NULL;
    wdown_mem_86837 = in0->v0->mem;
    wkey_mem_86838 = in0->v1->mem;
    wout_mem_86839 = in0->v2->mem;
    wpe_mem_86840 = in0->v3->mem;
    wqry_mem_86841 = in0->v4->mem;
    wte_mem_86842 = in0->v5->mem;
    wup_mem_86843 = in0->v6->mem;
    wval_mem_86844 = in0->v7->mem;
    wvoc_mem_86845 = in0->v8->mem;
    tokens_mem_86846 = in1->mem;
    mask_mem_86847 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_88608, wdown_mem_86837, wkey_mem_86838, wout_mem_86839, wpe_mem_86840, wqry_mem_86841, wte_mem_86842, wup_mem_86843, wval_mem_86844, wvoc_mem_86845, tokens_mem_86846, mask_mem_86847);
        if (ret == 0) {
            struct memblock mem_86828 = ctx->constants->mem_86828;
            struct memblock mem_86829 = ctx->constants->mem_86829;
            struct memblock mem_86830 = ctx->constants->mem_86830;
            struct memblock mem_86831 = ctx->constants->mem_86831;
            struct memblock mem_86832 = ctx->constants->mem_86832;
            struct memblock mem_86833 = ctx->constants->mem_86833;
            struct memblock mem_86834 = ctx->constants->mem_86834;
            struct memblock mem_86835 = ctx->constants->mem_86835;
            struct memblock mem_86836 = ctx->constants->mem_86836;
            
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_88608;
            (*out)->shape[0] = (int64_t) 16;
            (*out)->shape[1] = (int64_t) 27;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_to_params(struct futhark_context *ctx, struct futhark_opaque_params **out, const struct futhark_f64_2d *in0, const struct futhark_f64_2d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3, const struct futhark_f64_2d *in4, const struct futhark_f64_2d *in5, const struct futhark_f64_2d *in6, const struct futhark_f64_2d *in7, const struct futhark_f64_2d *in8)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_88616;
    
    mem_out_88616.references = NULL;
    
    struct memblock mem_out_88615;
    
    mem_out_88615.references = NULL;
    
    struct memblock mem_out_88614;
    
    mem_out_88614.references = NULL;
    
    struct memblock mem_out_88613;
    
    mem_out_88613.references = NULL;
    
    struct memblock mem_out_88612;
    
    mem_out_88612.references = NULL;
    
    struct memblock mem_out_88611;
    
    mem_out_88611.references = NULL;
    
    struct memblock mem_out_88610;
    
    mem_out_88610.references = NULL;
    
    struct memblock mem_out_88609;
    
    mem_out_88609.references = NULL;
    
    struct memblock mem_out_88608;
    
    mem_out_88608.references = NULL;
    
    struct memblock wvoc_mem_86845;
    
    wvoc_mem_86845.references = NULL;
    
    struct memblock wdown_mem_86844;
    
    wdown_mem_86844.references = NULL;
    
    struct memblock wup_mem_86843;
    
    wup_mem_86843.references = NULL;
    
    struct memblock wout_mem_86842;
    
    wout_mem_86842.references = NULL;
    
    struct memblock wval_mem_86841;
    
    wval_mem_86841.references = NULL;
    
    struct memblock wkey_mem_86840;
    
    wkey_mem_86840.references = NULL;
    
    struct memblock wqry_mem_86839;
    
    wqry_mem_86839.references = NULL;
    
    struct memblock wpe_mem_86838;
    
    wpe_mem_86838.references = NULL;
    
    struct memblock wte_mem_86837;
    
    wte_mem_86837.references = NULL;
    wte_mem_86837 = in0->mem;
    wpe_mem_86838 = in1->mem;
    wqry_mem_86839 = in2->mem;
    wkey_mem_86840 = in3->mem;
    wval_mem_86841 = in4->mem;
    wout_mem_86842 = in5->mem;
    wup_mem_86843 = in6->mem;
    wdown_mem_86844 = in7->mem;
    wvoc_mem_86845 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_88608, &mem_out_88609, &mem_out_88610, &mem_out_88611, &mem_out_88612, &mem_out_88613, &mem_out_88614, &mem_out_88615, &mem_out_88616, wte_mem_86837, wpe_mem_86838, wqry_mem_86839, wkey_mem_86840, wval_mem_86841, wout_mem_86842, wup_mem_86843, wdown_mem_86844, wvoc_mem_86845);
        if (ret == 0) {
            struct memblock mem_86828 = ctx->constants->mem_86828;
            struct memblock mem_86829 = ctx->constants->mem_86829;
            struct memblock mem_86830 = ctx->constants->mem_86830;
            struct memblock mem_86831 = ctx->constants->mem_86831;
            struct memblock mem_86832 = ctx->constants->mem_86832;
            struct memblock mem_86833 = ctx->constants->mem_86833;
            struct memblock mem_86834 = ctx->constants->mem_86834;
            struct memblock mem_86835 = ctx->constants->mem_86835;
            struct memblock mem_86836 = ctx->constants->mem_86836;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_88608;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_88609;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_88610;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_88611;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_88612;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_88613;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_88614;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_88615;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_88616;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_train(struct futhark_context *ctx, struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_opaque_params *in1, const struct futhark_opaque_params *in2, const struct futhark_f64_3d *in3, const struct futhark_i64_1d *in4, const struct futhark_i64_2d *in5)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_88634;
    
    mem_out_88634.references = NULL;
    
    struct memblock mem_out_88633;
    
    mem_out_88633.references = NULL;
    
    struct memblock mem_out_88632;
    
    mem_out_88632.references = NULL;
    
    struct memblock mem_out_88631;
    
    mem_out_88631.references = NULL;
    
    struct memblock mem_out_88630;
    
    mem_out_88630.references = NULL;
    
    struct memblock mem_out_88629;
    
    mem_out_88629.references = NULL;
    
    struct memblock mem_out_88628;
    
    mem_out_88628.references = NULL;
    
    struct memblock mem_out_88627;
    
    mem_out_88627.references = NULL;
    
    struct memblock mem_out_88626;
    
    mem_out_88626.references = NULL;
    
    struct memblock mem_out_88625;
    
    mem_out_88625.references = NULL;
    
    struct memblock mem_out_88624;
    
    mem_out_88624.references = NULL;
    
    struct memblock mem_out_88623;
    
    mem_out_88623.references = NULL;
    
    struct memblock mem_out_88622;
    
    mem_out_88622.references = NULL;
    
    struct memblock mem_out_88621;
    
    mem_out_88621.references = NULL;
    
    struct memblock mem_out_88620;
    
    mem_out_88620.references = NULL;
    
    struct memblock mem_out_88619;
    
    mem_out_88619.references = NULL;
    
    struct memblock mem_out_88618;
    
    mem_out_88618.references = NULL;
    
    struct memblock mem_out_88617;
    
    mem_out_88617.references = NULL;
    
    struct memblock mem_out_88616;
    
    mem_out_88616.references = NULL;
    
    struct memblock mem_out_88615;
    
    mem_out_88615.references = NULL;
    
    struct memblock mem_out_88614;
    
    mem_out_88614.references = NULL;
    
    struct memblock mem_out_88613;
    
    mem_out_88613.references = NULL;
    
    struct memblock mem_out_88612;
    
    mem_out_88612.references = NULL;
    
    struct memblock mem_out_88611;
    
    mem_out_88611.references = NULL;
    
    struct memblock mem_out_88610;
    
    mem_out_88610.references = NULL;
    
    struct memblock mem_out_88609;
    
    mem_out_88609.references = NULL;
    
    struct memblock mem_out_88608;
    
    mem_out_88608.references = NULL;
    
    struct memblock seqs_mem_86866;
    
    seqs_mem_86866.references = NULL;
    
    struct memblock dls_mem_86865;
    
    dls_mem_86865.references = NULL;
    
    struct memblock masks_mem_86864;
    
    masks_mem_86864.references = NULL;
    
    struct memblock wvoc_mem_86863;
    
    wvoc_mem_86863.references = NULL;
    
    struct memblock wval_mem_86862;
    
    wval_mem_86862.references = NULL;
    
    struct memblock wup_mem_86861;
    
    wup_mem_86861.references = NULL;
    
    struct memblock wte_mem_86860;
    
    wte_mem_86860.references = NULL;
    
    struct memblock wqry_mem_86859;
    
    wqry_mem_86859.references = NULL;
    
    struct memblock wpe_mem_86858;
    
    wpe_mem_86858.references = NULL;
    
    struct memblock wout_mem_86857;
    
    wout_mem_86857.references = NULL;
    
    struct memblock wkey_mem_86856;
    
    wkey_mem_86856.references = NULL;
    
    struct memblock wdown_mem_86855;
    
    wdown_mem_86855.references = NULL;
    
    struct memblock wvoc_mem_86854;
    
    wvoc_mem_86854.references = NULL;
    
    struct memblock wval_mem_86853;
    
    wval_mem_86853.references = NULL;
    
    struct memblock wup_mem_86852;
    
    wup_mem_86852.references = NULL;
    
    struct memblock wte_mem_86851;
    
    wte_mem_86851.references = NULL;
    
    struct memblock wqry_mem_86850;
    
    wqry_mem_86850.references = NULL;
    
    struct memblock wpe_mem_86849;
    
    wpe_mem_86849.references = NULL;
    
    struct memblock wout_mem_86848;
    
    wout_mem_86848.references = NULL;
    
    struct memblock wkey_mem_86847;
    
    wkey_mem_86847.references = NULL;
    
    struct memblock wdown_mem_86846;
    
    wdown_mem_86846.references = NULL;
    
    struct memblock wvoc_mem_86845;
    
    wvoc_mem_86845.references = NULL;
    
    struct memblock wval_mem_86844;
    
    wval_mem_86844.references = NULL;
    
    struct memblock wup_mem_86843;
    
    wup_mem_86843.references = NULL;
    
    struct memblock wte_mem_86842;
    
    wte_mem_86842.references = NULL;
    
    struct memblock wqry_mem_86841;
    
    wqry_mem_86841.references = NULL;
    
    struct memblock wpe_mem_86840;
    
    wpe_mem_86840.references = NULL;
    
    struct memblock wout_mem_86839;
    
    wout_mem_86839.references = NULL;
    
    struct memblock wkey_mem_86838;
    
    wkey_mem_86838.references = NULL;
    
    struct memblock wdown_mem_86837;
    
    wdown_mem_86837.references = NULL;
    wdown_mem_86837 = in0->v0->mem;
    wkey_mem_86838 = in0->v1->mem;
    wout_mem_86839 = in0->v2->mem;
    wpe_mem_86840 = in0->v3->mem;
    wqry_mem_86841 = in0->v4->mem;
    wte_mem_86842 = in0->v5->mem;
    wup_mem_86843 = in0->v6->mem;
    wval_mem_86844 = in0->v7->mem;
    wvoc_mem_86845 = in0->v8->mem;
    wdown_mem_86846 = in1->v0->mem;
    wkey_mem_86847 = in1->v1->mem;
    wout_mem_86848 = in1->v2->mem;
    wpe_mem_86849 = in1->v3->mem;
    wqry_mem_86850 = in1->v4->mem;
    wte_mem_86851 = in1->v5->mem;
    wup_mem_86852 = in1->v6->mem;
    wval_mem_86853 = in1->v7->mem;
    wvoc_mem_86854 = in1->v8->mem;
    wdown_mem_86855 = in2->v0->mem;
    wkey_mem_86856 = in2->v1->mem;
    wout_mem_86857 = in2->v2->mem;
    wpe_mem_86858 = in2->v3->mem;
    wqry_mem_86859 = in2->v4->mem;
    wte_mem_86860 = in2->v5->mem;
    wup_mem_86861 = in2->v6->mem;
    wval_mem_86862 = in2->v7->mem;
    wvoc_mem_86863 = in2->v8->mem;
    masks_mem_86864 = in3->mem;
    dls_mem_86865 = in4->mem;
    seqs_mem_86866 = in5->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && (((int64_t) 500 == in3->shape[0] && ((int64_t) 16 == in3->shape[1] && (int64_t) 16 == in3->shape[2])) && ((int64_t) 500 == in4->shape[0] && ((int64_t) 500 == in5->shape[0] && (int64_t) 16 == in5->shape[1]))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_88608, &mem_out_88609, &mem_out_88610, &mem_out_88611, &mem_out_88612, &mem_out_88613, &mem_out_88614, &mem_out_88615, &mem_out_88616, &mem_out_88617, &mem_out_88618, &mem_out_88619, &mem_out_88620, &mem_out_88621, &mem_out_88622, &mem_out_88623, &mem_out_88624, &mem_out_88625, &mem_out_88626, &mem_out_88627, &mem_out_88628, &mem_out_88629, &mem_out_88630, &mem_out_88631, &mem_out_88632, &mem_out_88633, &mem_out_88634, wdown_mem_86837, wkey_mem_86838, wout_mem_86839, wpe_mem_86840, wqry_mem_86841, wte_mem_86842, wup_mem_86843, wval_mem_86844, wvoc_mem_86845, wdown_mem_86846, wkey_mem_86847, wout_mem_86848, wpe_mem_86849, wqry_mem_86850, wte_mem_86851, wup_mem_86852, wval_mem_86853, wvoc_mem_86854, wdown_mem_86855, wkey_mem_86856, wout_mem_86857, wpe_mem_86858, wqry_mem_86859, wte_mem_86860, wup_mem_86861, wval_mem_86862, wvoc_mem_86863, masks_mem_86864, dls_mem_86865, seqs_mem_86866);
        if (ret == 0) {
            struct memblock mem_86828 = ctx->constants->mem_86828;
            struct memblock mem_86829 = ctx->constants->mem_86829;
            struct memblock mem_86830 = ctx->constants->mem_86830;
            struct memblock mem_86831 = ctx->constants->mem_86831;
            struct memblock mem_86832 = ctx->constants->mem_86832;
            struct memblock mem_86833 = ctx->constants->mem_86833;
            struct memblock mem_86834 = ctx->constants->mem_86834;
            struct memblock mem_86835 = ctx->constants->mem_86835;
            struct memblock mem_86836 = ctx->constants->mem_86836;
            
            assert((*out = (struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_88608;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_88609;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_88610;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_88611;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_88612;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_88613;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_88614;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_88615;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_88616;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v9->mem = mem_out_88617;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v10->mem = mem_out_88618;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v11->mem = mem_out_88619;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v12->mem = mem_out_88620;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v13->mem = mem_out_88621;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v14->mem = mem_out_88622;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v15->mem = mem_out_88623;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v16->mem = mem_out_88624;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v17->mem = mem_out_88625;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v18->mem = mem_out_88626;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v19->mem = mem_out_88627;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v20->mem = mem_out_88628;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v21->mem = mem_out_88629;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v22->mem = mem_out_88630;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v23->mem = mem_out_88631;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v24->mem = mem_out_88632;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v25->mem = mem_out_88633;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v26->mem = mem_out_88634;
            (*out)->v26->shape[0] = (int64_t) 27;
            (*out)->v26->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_zero_params(struct futhark_context *ctx, struct futhark_opaque_params **out)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_88616;
    
    mem_out_88616.references = NULL;
    
    struct memblock mem_out_88615;
    
    mem_out_88615.references = NULL;
    
    struct memblock mem_out_88614;
    
    mem_out_88614.references = NULL;
    
    struct memblock mem_out_88613;
    
    mem_out_88613.references = NULL;
    
    struct memblock mem_out_88612;
    
    mem_out_88612.references = NULL;
    
    struct memblock mem_out_88611;
    
    mem_out_88611.references = NULL;
    
    struct memblock mem_out_88610;
    
    mem_out_88610.references = NULL;
    
    struct memblock mem_out_88609;
    
    mem_out_88609.references = NULL;
    
    struct memblock mem_out_88608;
    
    mem_out_88608.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_88608, &mem_out_88609, &mem_out_88610, &mem_out_88611, &mem_out_88612, &mem_out_88613, &mem_out_88614, &mem_out_88615, &mem_out_88616);
        if (ret == 0) {
            struct memblock mem_86828 = ctx->constants->mem_86828;
            struct memblock mem_86829 = ctx->constants->mem_86829;
            struct memblock mem_86830 = ctx->constants->mem_86830;
            struct memblock mem_86831 = ctx->constants->mem_86831;
            struct memblock mem_86832 = ctx->constants->mem_86832;
            struct memblock mem_86833 = ctx->constants->mem_86833;
            struct memblock mem_86834 = ctx->constants->mem_86834;
            struct memblock mem_86835 = ctx->constants->mem_86835;
            struct memblock mem_86836 = ctx->constants->mem_86836;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_88608;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_88609;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_88610;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_88611;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_88612;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_88613;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_88614;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_88615;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_88616;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
