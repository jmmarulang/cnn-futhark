
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
    struct memblock mem_88640;
    struct memblock mem_88641;
    struct memblock mem_88642;
    struct memblock mem_88643;
    struct memblock mem_88644;
    struct memblock mem_88645;
    struct memblock mem_88646;
    struct memblock mem_88647;
    struct memblock mem_88648;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_10080(struct futhark_context *ctx, struct memblock *mem_out_p_90767, struct memblock *mem_out_p_90768, struct memblock *mem_out_p_90769, struct memblock w_mem_88649, struct memblock mw_mem_88650, struct memblock vw_mem_88651, struct memblock dw_mem_88652, int64_t n_62854, int64_t m_62855, int64_t step_62860, double lt_r_62861);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_10081(struct futhark_context *ctx, struct memblock *mem_out_p_90772, struct memblock *mem_out_p_90773, struct memblock *mem_out_p_90774, struct memblock w_mem_88649, struct memblock mw_mem_88650, struct memblock vw_mem_88651, struct memblock dw_mem_88652, int64_t n_63887, int64_t m_63888, int64_t step_63893, double lt_r_63894);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_90777, struct memblock wdown_mem_88649, struct memblock wkey_mem_88650, struct memblock wout_mem_88651, struct memblock wpe_mem_88652, struct memblock wqry_mem_88653, struct memblock wte_mem_88654, struct memblock wup_mem_88655, struct memblock wval_mem_88656, struct memblock wvoc_mem_88657, struct memblock tokens_mem_88658, struct memblock mask_mem_88659);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_90831, struct memblock *mem_out_p_90832, struct memblock *mem_out_p_90833, struct memblock *mem_out_p_90834, struct memblock *mem_out_p_90835, struct memblock *mem_out_p_90836, struct memblock *mem_out_p_90837, struct memblock *mem_out_p_90838, struct memblock *mem_out_p_90839, struct memblock wte_mem_88649, struct memblock wpe_mem_88650, struct memblock wqry_mem_88651, struct memblock wkey_mem_88652, struct memblock wval_mem_88653, struct memblock wout_mem_88654, struct memblock wup_mem_88655, struct memblock wdown_mem_88656, struct memblock wvoc_mem_88657);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_90840, struct memblock *mem_out_p_90841, struct memblock *mem_out_p_90842, struct memblock *mem_out_p_90843, struct memblock *mem_out_p_90844, struct memblock *mem_out_p_90845, struct memblock *mem_out_p_90846, struct memblock *mem_out_p_90847, struct memblock *mem_out_p_90848, struct memblock *mem_out_p_90849, struct memblock *mem_out_p_90850, struct memblock *mem_out_p_90851, struct memblock *mem_out_p_90852, struct memblock *mem_out_p_90853, struct memblock *mem_out_p_90854, struct memblock *mem_out_p_90855, struct memblock *mem_out_p_90856, struct memblock *mem_out_p_90857, struct memblock *mem_out_p_90858, struct memblock *mem_out_p_90859, struct memblock *mem_out_p_90860, struct memblock *mem_out_p_90861, struct memblock *mem_out_p_90862, struct memblock *mem_out_p_90863, struct memblock *mem_out_p_90864, struct memblock *mem_out_p_90865, struct memblock *mem_out_p_90866, struct memblock wdown_mem_88649, struct memblock wkey_mem_88650, struct memblock wout_mem_88651, struct memblock wpe_mem_88652, struct memblock wqry_mem_88653, struct memblock wte_mem_88654, struct memblock wup_mem_88655, struct memblock wval_mem_88656, struct memblock wvoc_mem_88657, struct memblock wdown_mem_88658, struct memblock wkey_mem_88659, struct memblock wout_mem_88660, struct memblock wpe_mem_88661, struct memblock wqry_mem_88662, struct memblock wte_mem_88663, struct memblock wup_mem_88664, struct memblock wval_mem_88665, struct memblock wvoc_mem_88666, struct memblock wdown_mem_88667, struct memblock wkey_mem_88668, struct memblock wout_mem_88669, struct memblock wpe_mem_88670, struct memblock wqry_mem_88671, struct memblock wte_mem_88672, struct memblock wup_mem_88673, struct memblock wval_mem_88674, struct memblock wvoc_mem_88675, struct memblock masks_mem_88676, struct memblock dls_mem_88677, struct memblock seqs_mem_88678);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_91036, struct memblock *mem_out_p_91037, struct memblock *mem_out_p_91038, struct memblock *mem_out_p_91039, struct memblock *mem_out_p_91040, struct memblock *mem_out_p_91041, struct memblock *mem_out_p_91042, struct memblock *mem_out_p_91043, struct memblock *mem_out_p_91044);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_88640 (ctx->constants->mem_88640)
    #define mem_88641 (ctx->constants->mem_88641)
    #define mem_88642 (ctx->constants->mem_88642)
    #define mem_88643 (ctx->constants->mem_88643)
    #define mem_88644 (ctx->constants->mem_88644)
    #define mem_88645 (ctx->constants->mem_88645)
    #define mem_88646 (ctx->constants->mem_88646)
    #define mem_88647 (ctx->constants->mem_88647)
    #define mem_88648 (ctx->constants->mem_88648)
    mem_88640.references = NULL;
    mem_88641.references = NULL;
    mem_88642.references = NULL;
    mem_88643.references = NULL;
    mem_88644.references = NULL;
    mem_88645.references = NULL;
    mem_88646.references = NULL;
    mem_88647.references = NULL;
    mem_88648.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_88640, (int64_t) 3456, "mem_88640")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_90749 = 0; nest_i_90749 < (int64_t) 27; nest_i_90749++) {
        for (int64_t nest_i_90750 = 0; nest_i_90750 < (int64_t) 16; nest_i_90750++) {
            ((double *) mem_88640.mem)[nest_i_90749 * (int64_t) 16 + nest_i_90750] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_88641, (int64_t) 2048, "mem_88641")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_90751 = 0; nest_i_90751 < (int64_t) 16; nest_i_90751++) {
        for (int64_t nest_i_90752 = 0; nest_i_90752 < (int64_t) 16; nest_i_90752++) {
            ((double *) mem_88641.mem)[nest_i_90751 * (int64_t) 16 + nest_i_90752] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_88642, (int64_t) 2048, "mem_88642")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_90753 = 0; nest_i_90753 < (int64_t) 16; nest_i_90753++) {
        for (int64_t nest_i_90754 = 0; nest_i_90754 < (int64_t) 16; nest_i_90754++) {
            ((double *) mem_88642.mem)[nest_i_90753 * (int64_t) 16 + nest_i_90754] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_88643, (int64_t) 2048, "mem_88643")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_90755 = 0; nest_i_90755 < (int64_t) 16; nest_i_90755++) {
        for (int64_t nest_i_90756 = 0; nest_i_90756 < (int64_t) 16; nest_i_90756++) {
            ((double *) mem_88643.mem)[nest_i_90755 * (int64_t) 16 + nest_i_90756] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_88644, (int64_t) 2048, "mem_88644")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_90757 = 0; nest_i_90757 < (int64_t) 16; nest_i_90757++) {
        for (int64_t nest_i_90758 = 0; nest_i_90758 < (int64_t) 16; nest_i_90758++) {
            ((double *) mem_88644.mem)[nest_i_90757 * (int64_t) 16 + nest_i_90758] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_88645, (int64_t) 2048, "mem_88645")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_90759 = 0; nest_i_90759 < (int64_t) 16; nest_i_90759++) {
        for (int64_t nest_i_90760 = 0; nest_i_90760 < (int64_t) 16; nest_i_90760++) {
            ((double *) mem_88645.mem)[nest_i_90759 * (int64_t) 16 + nest_i_90760] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_88646, (int64_t) 8192, "mem_88646")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_90761 = 0; nest_i_90761 < (int64_t) 64; nest_i_90761++) {
        for (int64_t nest_i_90762 = 0; nest_i_90762 < (int64_t) 16; nest_i_90762++) {
            ((double *) mem_88646.mem)[nest_i_90761 * (int64_t) 16 + nest_i_90762] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_88647, (int64_t) 8192, "mem_88647")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_90763 = 0; nest_i_90763 < (int64_t) 16; nest_i_90763++) {
        for (int64_t nest_i_90764 = 0; nest_i_90764 < (int64_t) 64; nest_i_90764++) {
            ((double *) mem_88647.mem)[nest_i_90763 * (int64_t) 64 + nest_i_90764] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_88648, (int64_t) 3456, "mem_88648")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_90765 = 0; nest_i_90765 < (int64_t) 27; nest_i_90765++) {
        for (int64_t nest_i_90766 = 0; nest_i_90766 < (int64_t) 16; nest_i_90766++) {
            ((double *) mem_88648.mem)[nest_i_90765 * (int64_t) 16 + nest_i_90766] = 0.0;
        }
    }
    #undef mem_88640
    #undef mem_88641
    #undef mem_88642
    #undef mem_88643
    #undef mem_88644
    #undef mem_88645
    #undef mem_88646
    #undef mem_88647
    #undef mem_88648
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_88640, "ctx->constants->mem_88640") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_88641, "ctx->constants->mem_88641") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_88642, "ctx->constants->mem_88642") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_88643, "ctx->constants->mem_88643") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_88644, "ctx->constants->mem_88644") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_88645, "ctx->constants->mem_88645") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_88646, "ctx->constants->mem_88646") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_88647, "ctx->constants->mem_88647") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_88648, "ctx->constants->mem_88648") != 0)
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_10080(struct futhark_context *ctx, struct memblock *mem_out_p_90767, struct memblock *mem_out_p_90768, struct memblock *mem_out_p_90769, struct memblock w_mem_88649, struct memblock mw_mem_88650, struct memblock vw_mem_88651, struct memblock dw_mem_88652, int64_t n_62854, int64_t m_62855, int64_t step_62860, double lt_r_62861)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_88693_cached_sizze_90770 = 0;
    unsigned char *mem_88693 = NULL;
    int64_t mem_88696_cached_sizze_90771 = 0;
    unsigned char *mem_88696 = NULL;
    struct memblock mem_88731;
    
    mem_88731.references = NULL;
    
    struct memblock mem_88658;
    
    mem_88658.references = NULL;
    
    struct memblock mem_88655;
    
    mem_88655.references = NULL;
    
    struct memblock mem_out_90450;
    
    mem_out_90450.references = NULL;
    
    struct memblock mem_out_90449;
    
    mem_out_90449.references = NULL;
    
    struct memblock mem_out_90448;
    
    mem_out_90448.references = NULL;
    
    struct memblock mem_88640 = ctx->constants->mem_88640;
    struct memblock mem_88641 = ctx->constants->mem_88641;
    struct memblock mem_88642 = ctx->constants->mem_88642;
    struct memblock mem_88643 = ctx->constants->mem_88643;
    struct memblock mem_88644 = ctx->constants->mem_88644;
    struct memblock mem_88645 = ctx->constants->mem_88645;
    struct memblock mem_88646 = ctx->constants->mem_88646;
    struct memblock mem_88647 = ctx->constants->mem_88647;
    struct memblock mem_88648 = ctx->constants->mem_88648;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_88653 = (int64_t) 8 * n_62854;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_88654 = m_62855 * binop_x_88653;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_88655, bytes_88654, "mem_88655")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_88658, bytes_88654, "mem_88658")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87801 = 0; i_87801 < n_62854; i_87801++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87794 = 0; i_87794 < m_62855; i_87794++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_83914 = ((double *) mw_mem_88650.mem)[i_87801 * m_62855 + i_87794];
            
            // futhark/microgpt.fut:383:10-20
            
            double zp_lhs_83915 = 0.85 * zt_rhs_83914;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_83916 = ((double *) dw_mem_88652.mem)[i_87801 * m_62855 + i_87794];
            
            // futhark/microgpt.fut:383:35-45
            
            double zp_rhs_83917 = 0.15000000000000002 * zt_rhs_83916;
            
            // futhark/microgpt.fut:383:21-45
            
            double lifted_lambda_res_83918 = zp_lhs_83915 + zp_rhs_83917;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_83925 = ((double *) vw_mem_88651.mem)[i_87801 * m_62855 + i_87794];
            
            // futhark/microgpt.fut:385:10-20
            
            double zp_lhs_83926 = 0.99 * zt_rhs_83925;
            
            // futhark/microgpt.fut:385:35-45
            
            double zt_lhs_83928 = 1.0000000000000009e-2 * zt_rhs_83916;
            
            // futhark/microgpt.fut:385:46-56
            
            double zp_rhs_83929 = zt_rhs_83916 * zt_lhs_83928;
            
            // futhark/microgpt.fut:385:21-56
            
            double lifted_lambda_res_83930 = zp_lhs_83926 + zp_rhs_83929;
            
            ((double *) mem_88655.mem)[i_87801 * m_62855 + i_87794] = lifted_lambda_res_83930;
            ((double *) mem_88658.mem)[i_87801 * m_62855 + i_87794] = lifted_lambda_res_83918;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_67927 = sitofp_i64_f64(step_62860);
    
    // futhark/microgpt.fut:387:54-57
    
    double ztzt_rhs_67928 = 1.0 + i64_res_67927;
    
    // futhark/microgpt.fut:387:30-57
    
    double zm_rhs_67929 = fpow64(0.85, ztzt_rhs_67928);
    
    // futhark/microgpt.fut:387:23-57
    
    double zs_rhs_67930 = 1.0 - zm_rhs_67929;
    
    // futhark/microgpt.fut:389:31-58
    
    double zm_rhs_67968 = fpow64(0.99, ztzt_rhs_67928);
    
    // futhark/microgpt.fut:389:23-58
    
    double zs_rhs_67969 = 1.0 - zm_rhs_67968;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_88693_cached_sizze_90770 < bytes_88654) {
        err = lexical_realloc(ctx, &mem_88693, &mem_88693_cached_sizze_90770, bytes_88654);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88696_cached_sizze_90771 < bytes_88654) {
        err = lexical_realloc(ctx, &mem_88696, &mem_88696_cached_sizze_90771, bytes_88654);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87815 = 0; i_87815 < n_62854; i_87815++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87808 = 0; i_87808 < m_62855; i_87808++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_83950 = ((double *) mem_88658.mem)[i_87815 * m_62855 + i_87808];
            
            // futhark/microgpt.fut:387:18-57
            
            double lifted_lambda_res_83951 = zs_lhs_83950 / zs_rhs_67930;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_83958 = ((double *) mem_88655.mem)[i_87815 * m_62855 + i_87808];
            
            // futhark/microgpt.fut:389:18-58
            
            double lifted_lambda_res_83959 = zs_lhs_83958 / zs_rhs_67969;
            
            ((double *) mem_88693)[i_87815 * m_62855 + i_87808] = lifted_lambda_res_83959;
            ((double *) mem_88696)[i_87815 * m_62855 + i_87808] = lifted_lambda_res_83951;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_88731, bytes_88654, "mem_88731")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87824 = 0; i_87824 < n_62854; i_87824++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87820 = 0; i_87820 < m_62855; i_87820++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_66995 = ((double *) w_mem_88649.mem)[i_87824 * m_62855 + i_87820];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_66996 = ((double *) mem_88696)[i_87824 * m_62855 + i_87820];
            
            // futhark/microgpt.fut:391:21-34
            
            double zs_lhs_66997 = lt_r_62861 * zt_rhs_66996;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_66998 = ((double *) mem_88693)[i_87824 * m_62855 + i_87820];
            
            // futhark/microgpt.fut:391:51-57
            
            double zp_lhs_66999 = fpow64(ztzt_lhs_66998, 0.5);
            
            // futhark/microgpt.fut:391:59-71
            
            double zs_rhs_67000 = 1.0e-8 + zp_lhs_66999;
            
            // futhark/microgpt.fut:391:35-71
            
            double zm_rhs_67001 = zs_lhs_66997 / zs_rhs_67000;
            
            // futhark/microgpt.fut:391:13-71
            
            double lifted_lambda_res_67002 = zm_lhs_66995 - zm_rhs_67001;
            
            ((double *) mem_88731.mem)[i_87824 * m_62855 + i_87820] = lifted_lambda_res_67002;
        }
    }
    if (memblock_set(ctx, &mem_out_90448, &mem_88731, "mem_88731") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90449, &mem_88658, "mem_88658") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90450, &mem_88655, "mem_88655") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90767, &mem_out_90448, "mem_out_90448") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90768, &mem_out_90449, "mem_out_90449") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90769, &mem_out_90450, "mem_out_90450") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_88693);
        free(mem_88696);
        if (memblock_unref(ctx, &mem_88731, "mem_88731") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88658, "mem_88658") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88655, "mem_88655") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90450, "mem_out_90450") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90449, "mem_out_90449") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90448, "mem_out_90448") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_10081(struct futhark_context *ctx, struct memblock *mem_out_p_90772, struct memblock *mem_out_p_90773, struct memblock *mem_out_p_90774, struct memblock w_mem_88649, struct memblock mw_mem_88650, struct memblock vw_mem_88651, struct memblock dw_mem_88652, int64_t n_63887, int64_t m_63888, int64_t step_63893, double lt_r_63894)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_88693_cached_sizze_90775 = 0;
    unsigned char *mem_88693 = NULL;
    int64_t mem_88696_cached_sizze_90776 = 0;
    unsigned char *mem_88696 = NULL;
    struct memblock mem_88731;
    
    mem_88731.references = NULL;
    
    struct memblock mem_88658;
    
    mem_88658.references = NULL;
    
    struct memblock mem_88655;
    
    mem_88655.references = NULL;
    
    struct memblock mem_out_90450;
    
    mem_out_90450.references = NULL;
    
    struct memblock mem_out_90449;
    
    mem_out_90449.references = NULL;
    
    struct memblock mem_out_90448;
    
    mem_out_90448.references = NULL;
    
    struct memblock mem_88640 = ctx->constants->mem_88640;
    struct memblock mem_88641 = ctx->constants->mem_88641;
    struct memblock mem_88642 = ctx->constants->mem_88642;
    struct memblock mem_88643 = ctx->constants->mem_88643;
    struct memblock mem_88644 = ctx->constants->mem_88644;
    struct memblock mem_88645 = ctx->constants->mem_88645;
    struct memblock mem_88646 = ctx->constants->mem_88646;
    struct memblock mem_88647 = ctx->constants->mem_88647;
    struct memblock mem_88648 = ctx->constants->mem_88648;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_88653 = (int64_t) 8 * n_63887;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_88654 = m_63888 * binop_x_88653;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_88655, bytes_88654, "mem_88655")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_88658, bytes_88654, "mem_88658")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87801 = 0; i_87801 < n_63887; i_87801++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87794 = 0; i_87794 < m_63888; i_87794++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_83914 = ((double *) mw_mem_88650.mem)[i_87801 * m_63888 + i_87794];
            
            // futhark/microgpt.fut:383:10-20
            
            double zp_lhs_83915 = 0.85 * zt_rhs_83914;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_83916 = ((double *) dw_mem_88652.mem)[i_87801 * m_63888 + i_87794];
            
            // futhark/microgpt.fut:383:35-45
            
            double zp_rhs_83917 = 0.15000000000000002 * zt_rhs_83916;
            
            // futhark/microgpt.fut:383:21-45
            
            double lifted_lambda_res_83918 = zp_lhs_83915 + zp_rhs_83917;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_83925 = ((double *) vw_mem_88651.mem)[i_87801 * m_63888 + i_87794];
            
            // futhark/microgpt.fut:385:10-20
            
            double zp_lhs_83926 = 0.99 * zt_rhs_83925;
            
            // futhark/microgpt.fut:385:35-45
            
            double zt_lhs_83928 = 1.0000000000000009e-2 * zt_rhs_83916;
            
            // futhark/microgpt.fut:385:46-56
            
            double zp_rhs_83929 = zt_rhs_83916 * zt_lhs_83928;
            
            // futhark/microgpt.fut:385:21-56
            
            double lifted_lambda_res_83930 = zp_lhs_83926 + zp_rhs_83929;
            
            ((double *) mem_88655.mem)[i_87801 * m_63888 + i_87794] = lifted_lambda_res_83930;
            ((double *) mem_88658.mem)[i_87801 * m_63888 + i_87794] = lifted_lambda_res_83918;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_67927 = sitofp_i64_f64(step_63893);
    
    // futhark/microgpt.fut:387:54-57
    
    double ztzt_rhs_67928 = 1.0 + i64_res_67927;
    
    // futhark/microgpt.fut:387:30-57
    
    double zm_rhs_67929 = fpow64(0.85, ztzt_rhs_67928);
    
    // futhark/microgpt.fut:387:23-57
    
    double zs_rhs_67930 = 1.0 - zm_rhs_67929;
    
    // futhark/microgpt.fut:389:31-58
    
    double zm_rhs_67968 = fpow64(0.99, ztzt_rhs_67928);
    
    // futhark/microgpt.fut:389:23-58
    
    double zs_rhs_67969 = 1.0 - zm_rhs_67968;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_88693_cached_sizze_90775 < bytes_88654) {
        err = lexical_realloc(ctx, &mem_88693, &mem_88693_cached_sizze_90775, bytes_88654);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88696_cached_sizze_90776 < bytes_88654) {
        err = lexical_realloc(ctx, &mem_88696, &mem_88696_cached_sizze_90776, bytes_88654);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87815 = 0; i_87815 < n_63887; i_87815++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87808 = 0; i_87808 < m_63888; i_87808++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_83950 = ((double *) mem_88658.mem)[i_87815 * m_63888 + i_87808];
            
            // futhark/microgpt.fut:387:18-57
            
            double lifted_lambda_res_83951 = zs_lhs_83950 / zs_rhs_67930;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_83958 = ((double *) mem_88655.mem)[i_87815 * m_63888 + i_87808];
            
            // futhark/microgpt.fut:389:18-58
            
            double lifted_lambda_res_83959 = zs_lhs_83958 / zs_rhs_67969;
            
            ((double *) mem_88693)[i_87815 * m_63888 + i_87808] = lifted_lambda_res_83959;
            ((double *) mem_88696)[i_87815 * m_63888 + i_87808] = lifted_lambda_res_83951;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_88731, bytes_88654, "mem_88731")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87824 = 0; i_87824 < n_63887; i_87824++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87820 = 0; i_87820 < m_63888; i_87820++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_66995 = ((double *) w_mem_88649.mem)[i_87824 * m_63888 + i_87820];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_66996 = ((double *) mem_88696)[i_87824 * m_63888 + i_87820];
            
            // futhark/microgpt.fut:391:21-34
            
            double zs_lhs_66997 = lt_r_63894 * zt_rhs_66996;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_66998 = ((double *) mem_88693)[i_87824 * m_63888 + i_87820];
            
            // futhark/microgpt.fut:391:51-57
            
            double zp_lhs_66999 = fpow64(ztzt_lhs_66998, 0.5);
            
            // futhark/microgpt.fut:391:59-71
            
            double zs_rhs_67000 = 1.0e-8 + zp_lhs_66999;
            
            // futhark/microgpt.fut:391:35-71
            
            double zm_rhs_67001 = zs_lhs_66997 / zs_rhs_67000;
            
            // futhark/microgpt.fut:391:13-71
            
            double lifted_lambda_res_67002 = zm_lhs_66995 - zm_rhs_67001;
            
            ((double *) mem_88731.mem)[i_87824 * m_63888 + i_87820] = lifted_lambda_res_67002;
        }
    }
    if (memblock_set(ctx, &mem_out_90448, &mem_88731, "mem_88731") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90449, &mem_88658, "mem_88658") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90450, &mem_88655, "mem_88655") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90772, &mem_out_90448, "mem_out_90448") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90773, &mem_out_90449, "mem_out_90449") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90774, &mem_out_90450, "mem_out_90450") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_88693);
        free(mem_88696);
        if (memblock_unref(ctx, &mem_88731, "mem_88731") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88658, "mem_88658") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_88655, "mem_88655") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90450, "mem_out_90450") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90449, "mem_out_90449") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90448, "mem_out_90448") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_90777, struct memblock wdown_mem_88649, struct memblock wkey_mem_88650, struct memblock wout_mem_88651, struct memblock wpe_mem_88652, struct memblock wqry_mem_88653, struct memblock wte_mem_88654, struct memblock wup_mem_88655, struct memblock wval_mem_88656, struct memblock wvoc_mem_88657, struct memblock tokens_mem_88658, struct memblock mask_mem_88659)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_88660_cached_sizze_90778 = 0;
    unsigned char *mem_88660 = NULL;
    int64_t mem_88665_cached_sizze_90779 = 0;
    unsigned char *mem_88665 = NULL;
    int64_t mem_88676_cached_sizze_90780 = 0;
    unsigned char *mem_88676 = NULL;
    int64_t mem_88681_cached_sizze_90781 = 0;
    unsigned char *mem_88681 = NULL;
    int64_t mem_88692_cached_sizze_90782 = 0;
    unsigned char *mem_88692 = NULL;
    int64_t mem_88697_cached_sizze_90783 = 0;
    unsigned char *mem_88697 = NULL;
    int64_t mem_88704_cached_sizze_90784 = 0;
    unsigned char *mem_88704 = NULL;
    int64_t mem_88715_cached_sizze_90785 = 0;
    unsigned char *mem_88715 = NULL;
    int64_t mem_88720_cached_sizze_90786 = 0;
    unsigned char *mem_88720 = NULL;
    int64_t mem_88727_cached_sizze_90787 = 0;
    unsigned char *mem_88727 = NULL;
    int64_t mem_88738_cached_sizze_90788 = 0;
    unsigned char *mem_88738 = NULL;
    int64_t mem_88739_cached_sizze_90789 = 0;
    unsigned char *mem_88739 = NULL;
    int64_t mem_88740_cached_sizze_90790 = 0;
    unsigned char *mem_88740 = NULL;
    int64_t mem_88753_cached_sizze_90791 = 0;
    unsigned char *mem_88753 = NULL;
    int64_t mem_88754_cached_sizze_90792 = 0;
    unsigned char *mem_88754 = NULL;
    int64_t mem_88755_cached_sizze_90793 = 0;
    unsigned char *mem_88755 = NULL;
    int64_t mem_88786_cached_sizze_90794 = 0;
    unsigned char *mem_88786 = NULL;
    int64_t mem_88787_cached_sizze_90795 = 0;
    unsigned char *mem_88787 = NULL;
    int64_t mem_88788_cached_sizze_90796 = 0;
    unsigned char *mem_88788 = NULL;
    int64_t mem_88804_cached_sizze_90797 = 0;
    unsigned char *mem_88804 = NULL;
    int64_t mem_88805_cached_sizze_90798 = 0;
    unsigned char *mem_88805 = NULL;
    int64_t mem_88806_cached_sizze_90799 = 0;
    unsigned char *mem_88806 = NULL;
    int64_t mem_88819_cached_sizze_90800 = 0;
    unsigned char *mem_88819 = NULL;
    int64_t mem_88820_cached_sizze_90801 = 0;
    unsigned char *mem_88820 = NULL;
    int64_t mem_88821_cached_sizze_90802 = 0;
    unsigned char *mem_88821 = NULL;
    int64_t mem_88867_cached_sizze_90803 = 0;
    unsigned char *mem_88867 = NULL;
    int64_t mem_88873_cached_sizze_90804 = 0;
    unsigned char *mem_88873 = NULL;
    int64_t mem_88878_cached_sizze_90805 = 0;
    unsigned char *mem_88878 = NULL;
    int64_t mem_88889_cached_sizze_90806 = 0;
    unsigned char *mem_88889 = NULL;
    int64_t mem_88894_cached_sizze_90807 = 0;
    unsigned char *mem_88894 = NULL;
    int64_t mem_88905_cached_sizze_90808 = 0;
    unsigned char *mem_88905 = NULL;
    int64_t mem_88910_cached_sizze_90809 = 0;
    unsigned char *mem_88910 = NULL;
    int64_t mem_88917_cached_sizze_90810 = 0;
    unsigned char *mem_88917 = NULL;
    int64_t mem_88928_cached_sizze_90811 = 0;
    unsigned char *mem_88928 = NULL;
    int64_t mem_88933_cached_sizze_90812 = 0;
    unsigned char *mem_88933 = NULL;
    int64_t mem_88949_cached_sizze_90813 = 0;
    unsigned char *mem_88949 = NULL;
    int64_t mem_88954_cached_sizze_90814 = 0;
    unsigned char *mem_88954 = NULL;
    int64_t mem_88965_cached_sizze_90815 = 0;
    unsigned char *mem_88965 = NULL;
    int64_t mem_88970_cached_sizze_90816 = 0;
    unsigned char *mem_88970 = NULL;
    int64_t mem_88981_cached_sizze_90817 = 0;
    unsigned char *mem_88981 = NULL;
    int64_t mem_88986_cached_sizze_90818 = 0;
    unsigned char *mem_88986 = NULL;
    int64_t mem_88997_cached_sizze_90819 = 0;
    unsigned char *mem_88997 = NULL;
    int64_t mem_89002_cached_sizze_90820 = 0;
    unsigned char *mem_89002 = NULL;
    int64_t mem_89009_cached_sizze_90821 = 0;
    unsigned char *mem_89009 = NULL;
    int64_t mem_89020_cached_sizze_90822 = 0;
    unsigned char *mem_89020 = NULL;
    int64_t mem_89025_cached_sizze_90823 = 0;
    unsigned char *mem_89025 = NULL;
    int64_t mem_89036_cached_sizze_90824 = 0;
    unsigned char *mem_89036 = NULL;
    int64_t mem_89041_cached_sizze_90825 = 0;
    unsigned char *mem_89041 = NULL;
    int64_t mem_89052_cached_sizze_90826 = 0;
    unsigned char *mem_89052 = NULL;
    int64_t mem_89057_cached_sizze_90827 = 0;
    unsigned char *mem_89057 = NULL;
    int64_t mem_89068_cached_sizze_90828 = 0;
    unsigned char *mem_89068 = NULL;
    int64_t mem_89073_cached_sizze_90829 = 0;
    unsigned char *mem_89073 = NULL;
    int64_t mem_89089_cached_sizze_90830 = 0;
    unsigned char *mem_89089 = NULL;
    struct memblock mem_89084;
    
    mem_89084.references = NULL;
    
    struct memblock mem_out_90448;
    
    mem_out_90448.references = NULL;
    
    struct memblock mem_88640 = ctx->constants->mem_88640;
    struct memblock mem_88641 = ctx->constants->mem_88641;
    struct memblock mem_88642 = ctx->constants->mem_88642;
    struct memblock mem_88643 = ctx->constants->mem_88643;
    struct memblock mem_88644 = ctx->constants->mem_88644;
    struct memblock mem_88645 = ctx->constants->mem_88645;
    struct memblock mem_88646 = ctx->constants->mem_88646;
    struct memblock mem_88647 = ctx->constants->mem_88647;
    struct memblock mem_88648 = ctx->constants->mem_88648;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_88660_cached_sizze_90778 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88660, &mem_88660_cached_sizze_90778, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88665_cached_sizze_90779 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88665, &mem_88665_cached_sizze_90779, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87796 = 0; i_87796 < (int64_t) 16; i_87796++) {
        // futhark/microgpt.fut:368:41-50
        
        int64_t tmp_76559 = ((int64_t *) tokens_mem_88658.mem)[i_87796];
        
        // futhark/microgpt.fut:368:37-51
        
        bool x_76560 = sle64((int64_t) 0, tmp_76559);
        
        // futhark/microgpt.fut:368:37-51
        
        bool y_76561 = slt64(tmp_76559, (int64_t) 27);
        
        // futhark/microgpt.fut:368:37-51
        
        bool bounds_check_76562 = x_76560 && y_76561;
        
        // futhark/microgpt.fut:368:37-51
        
        bool index_certs_76563;
        
        if (!bounds_check_76562) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_76559, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:368:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:368:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87792 = 0; i_87792 < (int64_t) 16; i_87792++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_76570 = ((double *) wte_mem_88654.mem)[tmp_76559 * (int64_t) 16 + i_87792];
            
            ((double *) mem_88665)[i_87792] = lifted_lambda_res_76570;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88660, i_87796 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88665, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88676_cached_sizze_90780 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88676, &mem_88676_cached_sizze_90780, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88681_cached_sizze_90781 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88681, &mem_88681_cached_sizze_90781, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87804 = 0; i_87804 < (int64_t) 16; i_87804++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87800 = 0; i_87800 < (int64_t) 16; i_87800++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_76602 = ((double *) wpe_mem_88652.mem)[i_87804 * (int64_t) 16 + i_87800];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_76603 = ((double *) mem_88660)[i_87804 * (int64_t) 16 + i_87800];
            
            // futhark/microgpt.fut:149:42-82
            
            double zp_res_76604 = zp_lhs_76602 + zp_rhs_76603;
            
            ((double *) mem_88681)[i_87800] = zp_res_76604;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88676, i_87804 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88681, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88692_cached_sizze_90782 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88692, &mem_88692_cached_sizze_90782, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88697_cached_sizze_90783 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88697, &mem_88697_cached_sizze_90783, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88704_cached_sizze_90784 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88704, &mem_88704_cached_sizze_90784, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87816 = 0; i_87816 < (int64_t) 16; i_87816++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87808 = 0; i_87808 < (int64_t) 16; i_87808++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_76619 = ((double *) mem_88676)[i_87816 * (int64_t) 16 + i_87808];
            
            // futhark/microgpt.fut:150:77-114
            
            double zt_res_76620 = zt_lhs_76619 * zt_lhs_76619;
            
            ((double *) mem_88697)[i_87808] = zt_res_76620;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_76622;
        double r_76624 = 0.0;
        
        for (int64_t i_76623 = 0; i_76623 < (int64_t) 16; i_76623++) {
            // futhark/microgpt.fut:151:37-47
            
            double lifted_lambda_res_76625 = ((double *) mem_88697)[i_76623];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_76626 = r_76624 + lifted_lambda_res_76625;
            double r_tmp_90455 = zp_res_76626;
            
            r_76624 = r_tmp_90455;
        }
        defunc_0_lifted_lambda_res_76622 = r_76624;
        // futhark/microgpt.fut:151:17-64
        
        double zs_res_76627 = defunc_0_lifted_lambda_res_76622 / 16.0;
        
        // futhark/microgpt.fut:152:24-55
        
        double zp_res_76628 = 1.0e-5 + zs_res_76627;
        
        // futhark/microgpt.fut:152:16-55
        
        double sqrt_res_76629 = futrts_sqrt64(zp_res_76628);
        
        // futhark/microgpt.fut:153:27-38
        
        double zs_res_76630 = 1.0 / sqrt_res_76629;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87812 = 0; i_87812 < (int64_t) 16; i_87812++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_76637 = ((double *) mem_88676)[i_87816 * (int64_t) 16 + i_87812];
            
            // futhark/microgpt.fut:153:5-38
            
            double zt_res_76638 = zs_res_76630 * zt_lhs_76637;
            
            ((double *) mem_88704)[i_87812] = zt_res_76638;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88692, i_87816 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88704, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88715_cached_sizze_90785 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88715, &mem_88715_cached_sizze_90785, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88720_cached_sizze_90786 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88720, &mem_88720_cached_sizze_90786, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88727_cached_sizze_90787 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88727, &mem_88727_cached_sizze_90787, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87828 = 0; i_87828 < (int64_t) 16; i_87828++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87820 = 0; i_87820 < (int64_t) 16; i_87820++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_76653 = ((double *) mem_88692)[i_87828 * (int64_t) 16 + i_87820];
            
            // futhark/microgpt.fut:154:77-114
            
            double zt_res_76654 = zt_lhs_76653 * zt_lhs_76653;
            
            ((double *) mem_88720)[i_87820] = zt_res_76654;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_76656;
        double r_76658 = 0.0;
        
        for (int64_t i_76657 = 0; i_76657 < (int64_t) 16; i_76657++) {
            // futhark/microgpt.fut:155:37-47
            
            double lifted_lambda_res_76659 = ((double *) mem_88720)[i_76657];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_76660 = r_76658 + lifted_lambda_res_76659;
            double r_tmp_90459 = zp_res_76660;
            
            r_76658 = r_tmp_90459;
        }
        defunc_0_lifted_lambda_res_76656 = r_76658;
        // futhark/microgpt.fut:155:17-64
        
        double zs_res_76661 = defunc_0_lifted_lambda_res_76656 / 16.0;
        
        // futhark/microgpt.fut:156:24-55
        
        double zp_res_76662 = 1.0e-5 + zs_res_76661;
        
        // futhark/microgpt.fut:156:16-55
        
        double sqrt_res_76663 = futrts_sqrt64(zp_res_76662);
        
        // futhark/microgpt.fut:157:27-38
        
        double zs_res_76664 = 1.0 / sqrt_res_76663;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87824 = 0; i_87824 < (int64_t) 16; i_87824++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_76671 = ((double *) mem_88692)[i_87828 * (int64_t) 16 + i_87824];
            
            // futhark/microgpt.fut:157:5-38
            
            double zt_res_76672 = zs_res_76664 * zt_lhs_76671;
            
            ((double *) mem_88727)[i_87824] = zt_res_76672;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88715, i_87828 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88727, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88738_cached_sizze_90788 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88738, &mem_88738_cached_sizze_90788, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88739_cached_sizze_90789 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88739, &mem_88739_cached_sizze_90789, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88740_cached_sizze_90790 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88740, &mem_88740_cached_sizze_90790, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88753_cached_sizze_90791 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88753, &mem_88753_cached_sizze_90791, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88754_cached_sizze_90792 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88754, &mem_88754_cached_sizze_90792, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88755_cached_sizze_90793 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88755, &mem_88755_cached_sizze_90793, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87846 = 0; i_87846 < (int64_t) 16; i_87846++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87836 = 0; i_87836 < (int64_t) 16; i_87836++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_84133;
            double r_84135 = 0.0;
            
            for (int64_t i_84134 = 0; i_84134 < (int64_t) 16; i_84134++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_84136 = ((double *) wqry_mem_88653.mem)[i_87836 * (int64_t) 16 + i_84134];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_84137 = ((double *) mem_88715)[i_87846 * (int64_t) 16 + i_84134];
                
                // futhark/microgpt.fut:158:66-105
                
                double zt_res_84138 = zt_lhs_84136 * zt_rhs_84137;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_84139 = r_84135 + zt_res_84138;
                double r_tmp_90467 = zp_res_84139;
                
                r_84135 = r_tmp_90467;
            }
            defunc_0_lifted_lambda_res_84133 = r_84135;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_84146;
            double r_84148 = 0.0;
            
            for (int64_t i_84147 = 0; i_84147 < (int64_t) 16; i_84147++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_84149 = ((double *) wkey_mem_88650.mem)[i_87836 * (int64_t) 16 + i_84147];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_84150 = ((double *) mem_88715)[i_87846 * (int64_t) 16 + i_84147];
                
                // futhark/microgpt.fut:159:66-105
                
                double zt_res_84151 = zt_lhs_84149 * zt_rhs_84150;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_84152 = r_84148 + zt_res_84151;
                double r_tmp_90468 = zp_res_84152;
                
                r_84148 = r_tmp_90468;
            }
            defunc_0_lifted_lambda_res_84146 = r_84148;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_84162;
            double r_84164 = 0.0;
            
            for (int64_t i_84163 = 0; i_84163 < (int64_t) 16; i_84163++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_84165 = ((double *) wval_mem_88656.mem)[i_87836 * (int64_t) 16 + i_84163];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_84166 = ((double *) mem_88715)[i_87846 * (int64_t) 16 + i_84163];
                
                // futhark/microgpt.fut:160:66-105
                
                double zt_res_84167 = zt_lhs_84165 * zt_rhs_84166;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_84168 = r_84164 + zt_res_84167;
                double r_tmp_90469 = zp_res_84168;
                
                r_84164 = r_tmp_90469;
            }
            defunc_0_lifted_lambda_res_84162 = r_84164;
            ((double *) mem_88753)[i_87836] = defunc_0_lifted_lambda_res_84162;
            ((double *) mem_88754)[i_87836] = defunc_0_lifted_lambda_res_84146;
            ((double *) mem_88755)[i_87836] = defunc_0_lifted_lambda_res_84133;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88738, i_87846 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88753, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88739, i_87846 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88754, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88740, i_87846 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88755, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88786_cached_sizze_90794 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88786, &mem_88786_cached_sizze_90794, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88787_cached_sizze_90795 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88787, &mem_88787_cached_sizze_90795, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88788_cached_sizze_90796 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88788, &mem_88788_cached_sizze_90796, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88804_cached_sizze_90797 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88804, &mem_88804_cached_sizze_90797, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88805_cached_sizze_90798 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88805, &mem_88805_cached_sizze_90798, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88806_cached_sizze_90799 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88806, &mem_88806_cached_sizze_90799, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88819_cached_sizze_90800 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_88819, &mem_88819_cached_sizze_90800, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88820_cached_sizze_90801 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_88820, &mem_88820_cached_sizze_90801, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88821_cached_sizze_90802 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_88821, &mem_88821_cached_sizze_90802, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87876 = 0; i_87876 < (int64_t) 4; i_87876++) {
        // futhark/microgpt.fut:161:69-72
        
        int64_t zp_lhs_84008 = mul64((int64_t) 4, i_87876);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87866 = 0; i_87866 < (int64_t) 16; i_87866++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87856 = 0; i_87856 < (int64_t) 4; i_87856++) {
                // futhark/microgpt.fut:161:74-81
                
                int64_t tmp_84326 = add64(zp_lhs_84008, i_87856);
                
                // futhark/microgpt.fut:161:51-83
                
                bool x_84327 = sle64((int64_t) 0, tmp_84326);
                
                // futhark/microgpt.fut:161:51-83
                
                bool y_84328 = slt64(tmp_84326, (int64_t) 16);
                
                // futhark/microgpt.fut:161:51-83
                
                bool bounds_check_84329 = x_84327 && y_84328;
                
                // futhark/microgpt.fut:161:51-83
                
                bool index_certs_84330;
                
                if (!bounds_check_84329) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_84326, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:161:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:161:15-84\n   #9  futhark/microgpt.fut:369:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_84331 = ((double *) mem_88740)[i_87866 * (int64_t) 16 + tmp_84326];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_84339 = ((double *) mem_88739)[i_87866 * (int64_t) 16 + tmp_84326];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_84350 = ((double *) mem_88738)[i_87866 * (int64_t) 16 + tmp_84326];
                
                ((double *) mem_88819)[i_87856] = lifted_lambda_res_84350;
                ((double *) mem_88820)[i_87856] = lifted_lambda_res_84339;
                ((double *) mem_88821)[i_87856] = lifted_lambda_res_84331;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88804, i_87866 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88819, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88805, i_87866 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88820, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88806, i_87866 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88821, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88786, i_87876 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_88804, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88787, i_87876 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_88805, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88788, i_87876 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_88806, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88867_cached_sizze_90803 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88867, &mem_88867_cached_sizze_90803, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88873_cached_sizze_90804 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88873, &mem_88873_cached_sizze_90804, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88878_cached_sizze_90805 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88878, &mem_88878_cached_sizze_90805, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88889_cached_sizze_90806 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88889, &mem_88889_cached_sizze_90806, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88894_cached_sizze_90807 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88894, &mem_88894_cached_sizze_90807, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88905_cached_sizze_90808 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88905, &mem_88905_cached_sizze_90808, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88910_cached_sizze_90809 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88910, &mem_88910_cached_sizze_90809, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88917_cached_sizze_90810 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88917, &mem_88917_cached_sizze_90810, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88928_cached_sizze_90811 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88928, &mem_88928_cached_sizze_90811, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88933_cached_sizze_90812 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_88933, &mem_88933_cached_sizze_90812, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87920 = 0; i_87920 < (int64_t) 4; i_87920++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87886 = 0; i_87886 < (int64_t) 16; i_87886++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87882 = 0; i_87882 < (int64_t) 16; i_87882++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_76817;
                double r_76819 = 0.0;
                
                for (int64_t i_76818 = 0; i_76818 < (int64_t) 4; i_76818++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_76820 = ((double *) mem_88788)[i_87920 * (int64_t) 64 + i_87886 * (int64_t) 4 + i_76818];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_76821 = ((double *) mem_88787)[i_87920 * (int64_t) 64 + i_87882 * (int64_t) 4 + i_76818];
                    
                    // futhark/microgpt.fut:164:113-164
                    
                    double zt_res_76822 = zt_lhs_76820 * zt_rhs_76821;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_76823 = r_76819 + zt_res_76822;
                    double r_tmp_90482 = zp_res_76823;
                    
                    r_76819 = r_tmp_90482;
                }
                defunc_0_lifted_lambda_res_76817 = r_76819;
                ((double *) mem_88878)[i_87882] = defunc_0_lifted_lambda_res_76817;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88873, i_87886 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88878, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87894 = 0; i_87894 < (int64_t) 16; i_87894++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87890 = 0; i_87890 < (int64_t) 16; i_87890++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_76838 = ((double *) mem_88873)[i_87894 * (int64_t) 16 + i_87890];
                
                // futhark/microgpt.fut:165:47-78
                
                double zs_res_76839 = zs_lhs_76838 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_76840 = ((double *) mask_mem_88659.mem)[i_87894 * (int64_t) 16 + i_87890];
                
                // futhark/microgpt.fut:165:65-102
                
                double zp_res_76841 = zs_res_76839 + zp_rhs_76840;
                
                ((double *) mem_88894)[i_87890] = zp_res_76841;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88889, i_87894 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88894, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87908 = 0; i_87908 < (int64_t) 16; i_87908++) {
            // futhark/microgpt.fut:103:13-33
            
            double defunc_0_reduce_res_84423;
            double redout_87896 = -INFINITY;
            
            for (int64_t i_87897 = 0; i_87897 < (int64_t) 16; i_87897++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_84377 = ((double *) mem_88889)[i_87908 * (int64_t) 16 + i_87897];
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_76862 = fmax64(lifted_lambda_res_84377, redout_87896);
                double redout_tmp_90486 = max_res_76862;
                
                redout_87896 = redout_tmp_90486;
            }
            defunc_0_reduce_res_84423 = redout_87896;
            // futhark/microgpt.fut:167:65-74
            
            double neg_res_76863 = -defunc_0_reduce_res_84423;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87900 = 0; i_87900 < (int64_t) 16; i_87900++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_76870 = ((double *) mem_88889)[i_87908 * (int64_t) 16 + i_87900];
                
                // futhark/microgpt.fut:167:43-74
                
                double zp_res_76871 = neg_res_76863 + zp_lhs_76870;
                
                // futhark/microgpt.fut:167:36-74
                
                double exp_res_76872 = futrts_exp64(zp_res_76871);
                
                ((double *) mem_88910)[i_87900] = exp_res_76872;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_76874;
            double r_76876 = 0.0;
            
            for (int64_t i_76875 = 0; i_76875 < (int64_t) 16; i_76875++) {
                // futhark/microgpt.fut:168:36-46
                
                double lifted_lambda_res_76877 = ((double *) mem_88910)[i_76875];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_76878 = r_76876 + lifted_lambda_res_76877;
                double r_tmp_90488 = zp_res_76878;
                
                r_76876 = r_tmp_90488;
            }
            defunc_0_lifted_lambda_res_76874 = r_76876;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87904 = 0; i_87904 < (int64_t) 16; i_87904++) {
                // futhark/microgpt.fut:169:5-15
                
                double zs_lhs_76885 = ((double *) mem_88910)[i_87904];
                
                // futhark/microgpt.fut:169:5-23
                
                double zs_res_76886 = zs_lhs_76885 / defunc_0_lifted_lambda_res_76874;
                
                ((double *) mem_88917)[i_87904] = zs_res_76886;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88905, i_87908 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88917, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87916 = 0; i_87916 < (int64_t) 16; i_87916++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87912 = 0; i_87912 < (int64_t) 4; i_87912++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_76901;
                double r_76903 = 0.0;
                
                for (int64_t i_76902 = 0; i_76902 < (int64_t) 16; i_76902++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_76904 = ((double *) mem_88905)[i_87916 * (int64_t) 16 + i_76902];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_76905 = ((double *) mem_88786)[i_87920 * (int64_t) 64 + i_76902 * (int64_t) 4 + i_87912];
                    
                    // futhark/microgpt.fut:170:26-71
                    
                    double zt_res_76906 = zt_lhs_76904 * zt_rhs_76905;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_76907 = r_76903 + zt_res_76906;
                    double r_tmp_90492 = zp_res_76907;
                    
                    r_76903 = r_tmp_90492;
                }
                defunc_0_lifted_lambda_res_76901 = r_76903;
                ((double *) mem_88933)[i_87912] = defunc_0_lifted_lambda_res_76901;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88928, i_87916 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88933, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88867, i_87920 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_88928, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88949_cached_sizze_90813 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88949, &mem_88949_cached_sizze_90813, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88954_cached_sizze_90814 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88954, &mem_88954_cached_sizze_90814, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87928 = 0; i_87928 < (int64_t) 16; i_87928++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87924 = 0; i_87924 < (int64_t) 16; i_87924++) {
            // futhark/microgpt.fut:171:55-58
            
            int64_t tmp_76919 = sdiv64(i_87924, (int64_t) 4);
            
            // futhark/microgpt.fut:171:45-60
            
            bool x_76920 = sle64((int64_t) 0, tmp_76919);
            
            // futhark/microgpt.fut:171:45-60
            
            bool y_76921 = slt64(tmp_76919, (int64_t) 4);
            
            // futhark/microgpt.fut:171:45-60
            
            bool bounds_check_76922 = x_76920 && y_76921;
            
            // futhark/microgpt.fut:171:45-60
            
            bool index_certs_76923;
            
            if (!bounds_check_76922) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_76919, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:171:45-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:171:16-81\n   #6  futhark/microgpt.fut:369:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:171:75-78
            
            int64_t tmp_76924 = smod64(i_87924, (int64_t) 4);
            
            // futhark/microgpt.fut:171:45-80
            
            bool x_76925 = sle64((int64_t) 0, tmp_76924);
            
            // futhark/microgpt.fut:171:45-80
            
            bool y_76926 = slt64(tmp_76924, (int64_t) 4);
            
            // futhark/microgpt.fut:171:45-80
            
            bool bounds_check_76927 = x_76925 && y_76926;
            
            // futhark/microgpt.fut:171:45-80
            
            bool index_certs_76928;
            
            if (!bounds_check_76927) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_76924, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:171:45-80\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:171:16-81\n   #6  futhark/microgpt.fut:369:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_76929 = ((double *) mem_88867)[tmp_76919 * (int64_t) 64 + i_87928 * (int64_t) 4 + tmp_76924];
            
            ((double *) mem_88954)[i_87924] = lifted_lambda_res_76929;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88949, i_87928 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88954, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88965_cached_sizze_90815 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88965, &mem_88965_cached_sizze_90815, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88970_cached_sizze_90816 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88970, &mem_88970_cached_sizze_90816, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87936 = 0; i_87936 < (int64_t) 16; i_87936++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87932 = 0; i_87932 < (int64_t) 16; i_87932++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_76944;
            double r_76946 = 0.0;
            
            for (int64_t i_76945 = 0; i_76945 < (int64_t) 16; i_76945++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_76947 = ((double *) wout_mem_88651.mem)[i_87932 * (int64_t) 16 + i_76945];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_76948 = ((double *) mem_88949)[i_87936 * (int64_t) 16 + i_76945];
                
                // futhark/microgpt.fut:172:67-107
                
                double zt_res_76949 = zt_lhs_76947 * zt_rhs_76948;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_76950 = r_76946 + zt_res_76949;
                double r_tmp_90497 = zp_res_76950;
                
                r_76946 = r_tmp_90497;
            }
            defunc_0_lifted_lambda_res_76944 = r_76946;
            ((double *) mem_88970)[i_87932] = defunc_0_lifted_lambda_res_76944;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88965, i_87936 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88970, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88981_cached_sizze_90817 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88981, &mem_88981_cached_sizze_90817, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88986_cached_sizze_90818 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88986, &mem_88986_cached_sizze_90818, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87944 = 0; i_87944 < (int64_t) 16; i_87944++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87940 = 0; i_87940 < (int64_t) 16; i_87940++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_76965 = ((double *) mem_88965)[i_87944 * (int64_t) 16 + i_87940];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_76966 = ((double *) mem_88692)[i_87944 * (int64_t) 16 + i_87940];
            
            // futhark/microgpt.fut:173:46-84
            
            double zp_res_76967 = zp_lhs_76965 + zp_rhs_76966;
            
            ((double *) mem_88986)[i_87940] = zp_res_76967;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88981, i_87944 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88986, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88997_cached_sizze_90819 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88997, &mem_88997_cached_sizze_90819, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89002_cached_sizze_90820 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89002, &mem_89002_cached_sizze_90820, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89009_cached_sizze_90821 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89009, &mem_89009_cached_sizze_90821, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87956 = 0; i_87956 < (int64_t) 16; i_87956++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87948 = 0; i_87948 < (int64_t) 16; i_87948++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_76982 = ((double *) mem_88981)[i_87956 * (int64_t) 16 + i_87948];
            
            // futhark/microgpt.fut:174:78-117
            
            double zt_res_76983 = zt_lhs_76982 * zt_lhs_76982;
            
            ((double *) mem_89002)[i_87948] = zt_res_76983;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_76985;
        double r_76987 = 0.0;
        
        for (int64_t i_76986 = 0; i_76986 < (int64_t) 16; i_76986++) {
            // futhark/microgpt.fut:175:37-47
            
            double lifted_lambda_res_76988 = ((double *) mem_89002)[i_76986];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_76989 = r_76987 + lifted_lambda_res_76988;
            double r_tmp_90502 = zp_res_76989;
            
            r_76987 = r_tmp_90502;
        }
        defunc_0_lifted_lambda_res_76985 = r_76987;
        // futhark/microgpt.fut:175:17-64
        
        double zs_res_76990 = defunc_0_lifted_lambda_res_76985 / 16.0;
        
        // futhark/microgpt.fut:176:24-55
        
        double zp_res_76991 = 1.0e-5 + zs_res_76990;
        
        // futhark/microgpt.fut:176:16-55
        
        double sqrt_res_76992 = futrts_sqrt64(zp_res_76991);
        
        // futhark/microgpt.fut:177:28-39
        
        double zs_res_76993 = 1.0 / sqrt_res_76992;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87952 = 0; i_87952 < (int64_t) 16; i_87952++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_77000 = ((double *) mem_88981)[i_87956 * (int64_t) 16 + i_87952];
            
            // futhark/microgpt.fut:177:5-39
            
            double zt_res_77001 = zs_res_76993 * zt_lhs_77000;
            
            ((double *) mem_89009)[i_87952] = zt_res_77001;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88997, i_87956 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89009, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89020_cached_sizze_90822 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_89020, &mem_89020_cached_sizze_90822, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89025_cached_sizze_90823 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_89025, &mem_89025_cached_sizze_90823, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87964 = 0; i_87964 < (int64_t) 16; i_87964++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87960 = 0; i_87960 < (int64_t) 64; i_87960++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77017;
            double r_77019 = 0.0;
            
            for (int64_t i_77018 = 0; i_77018 < (int64_t) 16; i_77018++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_77020 = ((double *) wup_mem_88655.mem)[i_87960 * (int64_t) 16 + i_77018];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_77021 = ((double *) mem_88997)[i_87964 * (int64_t) 16 + i_77018];
                
                // futhark/microgpt.fut:178:67-106
                
                double zt_res_77022 = zt_lhs_77020 * zt_rhs_77021;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77023 = r_77019 + zt_res_77022;
                double r_tmp_90506 = zp_res_77023;
                
                r_77019 = r_tmp_90506;
            }
            defunc_0_lifted_lambda_res_77017 = r_77019;
            ((double *) mem_89025)[i_87960] = defunc_0_lifted_lambda_res_77017;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_89020, i_87964 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89025, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89036_cached_sizze_90824 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_89036, &mem_89036_cached_sizze_90824, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89041_cached_sizze_90825 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_89041, &mem_89041_cached_sizze_90825, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87972 = 0; i_87972 < (int64_t) 16; i_87972++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87968 = 0; i_87968 < (int64_t) 64; i_87968++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_77038 = ((double *) mem_89020)[i_87972 * (int64_t) 64 + i_87968];
            
            // futhark/microgpt.fut:179:45-73
            
            double max_res_77039 = fmax64(0.0, max_arg0_77038);
            
            ((double *) mem_89041)[i_87968] = max_res_77039;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_89036, i_87972 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89041, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89052_cached_sizze_90826 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89052, &mem_89052_cached_sizze_90826, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89057_cached_sizze_90827 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89057, &mem_89057_cached_sizze_90827, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87980 = 0; i_87980 < (int64_t) 16; i_87980++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87976 = 0; i_87976 < (int64_t) 16; i_87976++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77054;
            double r_77056 = 0.0;
            
            for (int64_t i_77055 = 0; i_77055 < (int64_t) 64; i_77055++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_77057 = ((double *) wdown_mem_88649.mem)[i_87976 * (int64_t) 64 + i_77055];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_77058 = ((double *) mem_89036)[i_87980 * (int64_t) 64 + i_77055];
                
                // futhark/microgpt.fut:180:67-108
                
                double zt_res_77059 = zt_lhs_77057 * zt_rhs_77058;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77060 = r_77056 + zt_res_77059;
                double r_tmp_90511 = zp_res_77060;
                
                r_77056 = r_tmp_90511;
            }
            defunc_0_lifted_lambda_res_77054 = r_77056;
            ((double *) mem_89057)[i_87976] = defunc_0_lifted_lambda_res_77054;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_89052, i_87980 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89057, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89068_cached_sizze_90828 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89068, &mem_89068_cached_sizze_90828, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89073_cached_sizze_90829 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89073, &mem_89073_cached_sizze_90829, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87988 = 0; i_87988 < (int64_t) 16; i_87988++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87984 = 0; i_87984 < (int64_t) 16; i_87984++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_77075 = ((double *) mem_89052)[i_87988 * (int64_t) 16 + i_87984];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_77076 = ((double *) mem_88981)[i_87988 * (int64_t) 16 + i_87984];
            
            // futhark/microgpt.fut:181:46-85
            
            double zp_res_77077 = zp_lhs_77075 + zp_rhs_77076;
            
            ((double *) mem_89073)[i_87984] = zp_res_77077;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_89068, i_87988 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89073, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_89084, (int64_t) 3456, "mem_89084")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89089_cached_sizze_90830 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_89089, &mem_89089_cached_sizze_90830, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87996 = 0; i_87996 < (int64_t) 16; i_87996++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87992 = 0; i_87992 < (int64_t) 27; i_87992++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77093;
            double r_77095 = 0.0;
            
            for (int64_t i_77094 = 0; i_77094 < (int64_t) 16; i_77094++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_77096 = ((double *) wvoc_mem_88657.mem)[i_87992 * (int64_t) 16 + i_77094];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_77097 = ((double *) mem_89068)[i_87996 * (int64_t) 16 + i_77094];
                
                // futhark/microgpt.fut:182:56-96
                
                double zt_res_77098 = zt_lhs_77096 * zt_rhs_77097;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77099 = r_77095 + zt_res_77098;
                double r_tmp_90516 = zp_res_77099;
                
                r_77095 = r_tmp_90516;
            }
            defunc_0_lifted_lambda_res_77093 = r_77095;
            ((double *) mem_89089)[i_87992] = defunc_0_lifted_lambda_res_77093;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_89084.mem, i_87996 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89089, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_90448, &mem_89084, "mem_89084") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90777, &mem_out_90448, "mem_out_90448") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_88660);
        free(mem_88665);
        free(mem_88676);
        free(mem_88681);
        free(mem_88692);
        free(mem_88697);
        free(mem_88704);
        free(mem_88715);
        free(mem_88720);
        free(mem_88727);
        free(mem_88738);
        free(mem_88739);
        free(mem_88740);
        free(mem_88753);
        free(mem_88754);
        free(mem_88755);
        free(mem_88786);
        free(mem_88787);
        free(mem_88788);
        free(mem_88804);
        free(mem_88805);
        free(mem_88806);
        free(mem_88819);
        free(mem_88820);
        free(mem_88821);
        free(mem_88867);
        free(mem_88873);
        free(mem_88878);
        free(mem_88889);
        free(mem_88894);
        free(mem_88905);
        free(mem_88910);
        free(mem_88917);
        free(mem_88928);
        free(mem_88933);
        free(mem_88949);
        free(mem_88954);
        free(mem_88965);
        free(mem_88970);
        free(mem_88981);
        free(mem_88986);
        free(mem_88997);
        free(mem_89002);
        free(mem_89009);
        free(mem_89020);
        free(mem_89025);
        free(mem_89036);
        free(mem_89041);
        free(mem_89052);
        free(mem_89057);
        free(mem_89068);
        free(mem_89073);
        free(mem_89089);
        if (memblock_unref(ctx, &mem_89084, "mem_89084") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90448, "mem_out_90448") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_90831, struct memblock *mem_out_p_90832, struct memblock *mem_out_p_90833, struct memblock *mem_out_p_90834, struct memblock *mem_out_p_90835, struct memblock *mem_out_p_90836, struct memblock *mem_out_p_90837, struct memblock *mem_out_p_90838, struct memblock *mem_out_p_90839, struct memblock wte_mem_88649, struct memblock wpe_mem_88650, struct memblock wqry_mem_88651, struct memblock wkey_mem_88652, struct memblock wval_mem_88653, struct memblock wout_mem_88654, struct memblock wup_mem_88655, struct memblock wdown_mem_88656, struct memblock wvoc_mem_88657)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_90456;
    
    mem_out_90456.references = NULL;
    
    struct memblock mem_out_90455;
    
    mem_out_90455.references = NULL;
    
    struct memblock mem_out_90454;
    
    mem_out_90454.references = NULL;
    
    struct memblock mem_out_90453;
    
    mem_out_90453.references = NULL;
    
    struct memblock mem_out_90452;
    
    mem_out_90452.references = NULL;
    
    struct memblock mem_out_90451;
    
    mem_out_90451.references = NULL;
    
    struct memblock mem_out_90450;
    
    mem_out_90450.references = NULL;
    
    struct memblock mem_out_90449;
    
    mem_out_90449.references = NULL;
    
    struct memblock mem_out_90448;
    
    mem_out_90448.references = NULL;
    
    struct memblock mem_88640 = ctx->constants->mem_88640;
    struct memblock mem_88641 = ctx->constants->mem_88641;
    struct memblock mem_88642 = ctx->constants->mem_88642;
    struct memblock mem_88643 = ctx->constants->mem_88643;
    struct memblock mem_88644 = ctx->constants->mem_88644;
    struct memblock mem_88645 = ctx->constants->mem_88645;
    struct memblock mem_88646 = ctx->constants->mem_88646;
    struct memblock mem_88647 = ctx->constants->mem_88647;
    struct memblock mem_88648 = ctx->constants->mem_88648;
    
    if (memblock_set(ctx, &mem_out_90448, &wdown_mem_88656, "wdown_mem_88656") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90449, &wkey_mem_88652, "wkey_mem_88652") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90450, &wout_mem_88654, "wout_mem_88654") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90451, &wpe_mem_88650, "wpe_mem_88650") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90452, &wqry_mem_88651, "wqry_mem_88651") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90453, &wte_mem_88649, "wte_mem_88649") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90454, &wup_mem_88655, "wup_mem_88655") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90455, &wval_mem_88653, "wval_mem_88653") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90456, &wvoc_mem_88657, "wvoc_mem_88657") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90831, &mem_out_90448, "mem_out_90448") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90832, &mem_out_90449, "mem_out_90449") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90833, &mem_out_90450, "mem_out_90450") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90834, &mem_out_90451, "mem_out_90451") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90835, &mem_out_90452, "mem_out_90452") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90836, &mem_out_90453, "mem_out_90453") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90837, &mem_out_90454, "mem_out_90454") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90838, &mem_out_90455, "mem_out_90455") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90839, &mem_out_90456, "mem_out_90456") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_90456, "mem_out_90456") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90455, "mem_out_90455") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90454, "mem_out_90454") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90453, "mem_out_90453") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90452, "mem_out_90452") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90451, "mem_out_90451") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90450, "mem_out_90450") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90449, "mem_out_90449") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90448, "mem_out_90448") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_90840, struct memblock *mem_out_p_90841, struct memblock *mem_out_p_90842, struct memblock *mem_out_p_90843, struct memblock *mem_out_p_90844, struct memblock *mem_out_p_90845, struct memblock *mem_out_p_90846, struct memblock *mem_out_p_90847, struct memblock *mem_out_p_90848, struct memblock *mem_out_p_90849, struct memblock *mem_out_p_90850, struct memblock *mem_out_p_90851, struct memblock *mem_out_p_90852, struct memblock *mem_out_p_90853, struct memblock *mem_out_p_90854, struct memblock *mem_out_p_90855, struct memblock *mem_out_p_90856, struct memblock *mem_out_p_90857, struct memblock *mem_out_p_90858, struct memblock *mem_out_p_90859, struct memblock *mem_out_p_90860, struct memblock *mem_out_p_90861, struct memblock *mem_out_p_90862, struct memblock *mem_out_p_90863, struct memblock *mem_out_p_90864, struct memblock *mem_out_p_90865, struct memblock *mem_out_p_90866, struct memblock wdown_mem_88649, struct memblock wkey_mem_88650, struct memblock wout_mem_88651, struct memblock wpe_mem_88652, struct memblock wqry_mem_88653, struct memblock wte_mem_88654, struct memblock wup_mem_88655, struct memblock wval_mem_88656, struct memblock wvoc_mem_88657, struct memblock wdown_mem_88658, struct memblock wkey_mem_88659, struct memblock wout_mem_88660, struct memblock wpe_mem_88661, struct memblock wqry_mem_88662, struct memblock wte_mem_88663, struct memblock wup_mem_88664, struct memblock wval_mem_88665, struct memblock wvoc_mem_88666, struct memblock wdown_mem_88667, struct memblock wkey_mem_88668, struct memblock wout_mem_88669, struct memblock wpe_mem_88670, struct memblock wqry_mem_88671, struct memblock wte_mem_88672, struct memblock wup_mem_88673, struct memblock wval_mem_88674, struct memblock wvoc_mem_88675, struct memblock masks_mem_88676, struct memblock dls_mem_88677, struct memblock seqs_mem_88678)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_88787_cached_sizze_90867 = 0;
    unsigned char *mem_88787 = NULL;
    int64_t mem_88788_cached_sizze_90868 = 0;
    unsigned char *mem_88788 = NULL;
    int64_t mem_88797_cached_sizze_90869 = 0;
    unsigned char *mem_88797 = NULL;
    int64_t mem_88804_cached_sizze_90870 = 0;
    unsigned char *mem_88804 = NULL;
    int64_t mem_88819_cached_sizze_90871 = 0;
    unsigned char *mem_88819 = NULL;
    int64_t mem_88820_cached_sizze_90872 = 0;
    unsigned char *mem_88820 = NULL;
    int64_t mem_88829_cached_sizze_90873 = 0;
    unsigned char *mem_88829 = NULL;
    int64_t mem_88836_cached_sizze_90874 = 0;
    unsigned char *mem_88836 = NULL;
    int64_t mem_88851_cached_sizze_90875 = 0;
    unsigned char *mem_88851 = NULL;
    int64_t mem_88852_cached_sizze_90876 = 0;
    unsigned char *mem_88852 = NULL;
    int64_t mem_88861_cached_sizze_90877 = 0;
    unsigned char *mem_88861 = NULL;
    int64_t mem_88862_cached_sizze_90878 = 0;
    unsigned char *mem_88862 = NULL;
    int64_t mem_88883_cached_sizze_90879 = 0;
    unsigned char *mem_88883 = NULL;
    int64_t mem_88884_cached_sizze_90880 = 0;
    unsigned char *mem_88884 = NULL;
    int64_t mem_88885_cached_sizze_90881 = 0;
    unsigned char *mem_88885 = NULL;
    int64_t mem_88897_cached_sizze_90882 = 0;
    unsigned char *mem_88897 = NULL;
    int64_t mem_88898_cached_sizze_90883 = 0;
    unsigned char *mem_88898 = NULL;
    int64_t mem_88922_cached_sizze_90884 = 0;
    unsigned char *mem_88922 = NULL;
    int64_t mem_88923_cached_sizze_90885 = 0;
    unsigned char *mem_88923 = NULL;
    int64_t mem_88924_cached_sizze_90886 = 0;
    unsigned char *mem_88924 = NULL;
    int64_t mem_88925_cached_sizze_90887 = 0;
    unsigned char *mem_88925 = NULL;
    int64_t mem_88926_cached_sizze_90888 = 0;
    unsigned char *mem_88926 = NULL;
    int64_t mem_88945_cached_sizze_90889 = 0;
    unsigned char *mem_88945 = NULL;
    int64_t mem_88946_cached_sizze_90890 = 0;
    unsigned char *mem_88946 = NULL;
    int64_t mem_88947_cached_sizze_90891 = 0;
    unsigned char *mem_88947 = NULL;
    int64_t mem_88984_cached_sizze_90892 = 0;
    unsigned char *mem_88984 = NULL;
    int64_t mem_88985_cached_sizze_90893 = 0;
    unsigned char *mem_88985 = NULL;
    int64_t mem_88986_cached_sizze_90894 = 0;
    unsigned char *mem_88986 = NULL;
    int64_t mem_89002_cached_sizze_90895 = 0;
    unsigned char *mem_89002 = NULL;
    int64_t mem_89003_cached_sizze_90896 = 0;
    unsigned char *mem_89003 = NULL;
    int64_t mem_89004_cached_sizze_90897 = 0;
    unsigned char *mem_89004 = NULL;
    int64_t mem_89017_cached_sizze_90898 = 0;
    unsigned char *mem_89017 = NULL;
    int64_t mem_89018_cached_sizze_90899 = 0;
    unsigned char *mem_89018 = NULL;
    int64_t mem_89019_cached_sizze_90900 = 0;
    unsigned char *mem_89019 = NULL;
    int64_t mem_89065_cached_sizze_90901 = 0;
    unsigned char *mem_89065 = NULL;
    int64_t mem_89066_cached_sizze_90902 = 0;
    unsigned char *mem_89066 = NULL;
    int64_t mem_89077_cached_sizze_90903 = 0;
    unsigned char *mem_89077 = NULL;
    int64_t mem_89078_cached_sizze_90904 = 0;
    unsigned char *mem_89078 = NULL;
    int64_t mem_89087_cached_sizze_90905 = 0;
    unsigned char *mem_89087 = NULL;
    int64_t mem_89088_cached_sizze_90906 = 0;
    unsigned char *mem_89088 = NULL;
    int64_t mem_89109_cached_sizze_90907 = 0;
    unsigned char *mem_89109 = NULL;
    int64_t mem_89114_cached_sizze_90908 = 0;
    unsigned char *mem_89114 = NULL;
    int64_t mem_89125_cached_sizze_90909 = 0;
    unsigned char *mem_89125 = NULL;
    int64_t mem_89130_cached_sizze_90910 = 0;
    unsigned char *mem_89130 = NULL;
    int64_t mem_89137_cached_sizze_90911 = 0;
    unsigned char *mem_89137 = NULL;
    int64_t mem_89144_cached_sizze_90912 = 0;
    unsigned char *mem_89144 = NULL;
    int64_t mem_89155_cached_sizze_90913 = 0;
    unsigned char *mem_89155 = NULL;
    int64_t mem_89160_cached_sizze_90914 = 0;
    unsigned char *mem_89160 = NULL;
    int64_t mem_89181_cached_sizze_90915 = 0;
    unsigned char *mem_89181 = NULL;
    int64_t mem_89182_cached_sizze_90916 = 0;
    unsigned char *mem_89182 = NULL;
    int64_t mem_89190_cached_sizze_90917 = 0;
    unsigned char *mem_89190 = NULL;
    int64_t mem_89204_cached_sizze_90918 = 0;
    unsigned char *mem_89204 = NULL;
    int64_t mem_89209_cached_sizze_90919 = 0;
    unsigned char *mem_89209 = NULL;
    int64_t mem_89220_cached_sizze_90920 = 0;
    unsigned char *mem_89220 = NULL;
    int64_t mem_89225_cached_sizze_90921 = 0;
    unsigned char *mem_89225 = NULL;
    int64_t mem_89236_cached_sizze_90922 = 0;
    unsigned char *mem_89236 = NULL;
    int64_t mem_89237_cached_sizze_90923 = 0;
    unsigned char *mem_89237 = NULL;
    int64_t mem_89246_cached_sizze_90924 = 0;
    unsigned char *mem_89246 = NULL;
    int64_t mem_89247_cached_sizze_90925 = 0;
    unsigned char *mem_89247 = NULL;
    int64_t mem_89268_cached_sizze_90926 = 0;
    unsigned char *mem_89268 = NULL;
    int64_t mem_89269_cached_sizze_90927 = 0;
    unsigned char *mem_89269 = NULL;
    int64_t mem_89277_cached_sizze_90928 = 0;
    unsigned char *mem_89277 = NULL;
    int64_t mem_89291_cached_sizze_90929 = 0;
    unsigned char *mem_89291 = NULL;
    int64_t mem_89292_cached_sizze_90930 = 0;
    unsigned char *mem_89292 = NULL;
    int64_t mem_89300_cached_sizze_90931 = 0;
    unsigned char *mem_89300 = NULL;
    int64_t mem_89314_cached_sizze_90932 = 0;
    unsigned char *mem_89314 = NULL;
    int64_t mem_89319_cached_sizze_90933 = 0;
    unsigned char *mem_89319 = NULL;
    int64_t mem_89330_cached_sizze_90934 = 0;
    unsigned char *mem_89330 = NULL;
    int64_t mem_89335_cached_sizze_90935 = 0;
    unsigned char *mem_89335 = NULL;
    int64_t mem_89346_cached_sizze_90936 = 0;
    unsigned char *mem_89346 = NULL;
    int64_t mem_89351_cached_sizze_90937 = 0;
    unsigned char *mem_89351 = NULL;
    int64_t mem_89362_cached_sizze_90938 = 0;
    unsigned char *mem_89362 = NULL;
    int64_t mem_89363_cached_sizze_90939 = 0;
    unsigned char *mem_89363 = NULL;
    int64_t mem_89372_cached_sizze_90940 = 0;
    unsigned char *mem_89372 = NULL;
    int64_t mem_89373_cached_sizze_90941 = 0;
    unsigned char *mem_89373 = NULL;
    int64_t mem_89386_cached_sizze_90942 = 0;
    unsigned char *mem_89386 = NULL;
    int64_t mem_89387_cached_sizze_90943 = 0;
    unsigned char *mem_89387 = NULL;
    int64_t mem_89400_cached_sizze_90944 = 0;
    unsigned char *mem_89400 = NULL;
    int64_t mem_89401_cached_sizze_90945 = 0;
    unsigned char *mem_89401 = NULL;
    int64_t mem_89422_cached_sizze_90946 = 0;
    unsigned char *mem_89422 = NULL;
    int64_t mem_89429_cached_sizze_90947 = 0;
    unsigned char *mem_89429 = NULL;
    int64_t mem_89434_cached_sizze_90948 = 0;
    unsigned char *mem_89434 = NULL;
    int64_t mem_89445_cached_sizze_90949 = 0;
    unsigned char *mem_89445 = NULL;
    int64_t mem_89450_cached_sizze_90950 = 0;
    unsigned char *mem_89450 = NULL;
    int64_t mem_89461_cached_sizze_90951 = 0;
    unsigned char *mem_89461 = NULL;
    int64_t mem_89462_cached_sizze_90952 = 0;
    unsigned char *mem_89462 = NULL;
    int64_t mem_89471_cached_sizze_90953 = 0;
    unsigned char *mem_89471 = NULL;
    int64_t mem_89472_cached_sizze_90954 = 0;
    unsigned char *mem_89472 = NULL;
    int64_t mem_89493_cached_sizze_90955 = 0;
    unsigned char *mem_89493 = NULL;
    int64_t mem_89498_cached_sizze_90956 = 0;
    unsigned char *mem_89498 = NULL;
    int64_t mem_89509_cached_sizze_90957 = 0;
    unsigned char *mem_89509 = NULL;
    int64_t mem_89514_cached_sizze_90958 = 0;
    unsigned char *mem_89514 = NULL;
    int64_t mem_89525_cached_sizze_90959 = 0;
    unsigned char *mem_89525 = NULL;
    int64_t mem_89532_cached_sizze_90960 = 0;
    unsigned char *mem_89532 = NULL;
    int64_t mem_89539_cached_sizze_90961 = 0;
    unsigned char *mem_89539 = NULL;
    int64_t mem_89549_cached_sizze_90962 = 0;
    unsigned char *mem_89549 = NULL;
    int64_t mem_89554_cached_sizze_90963 = 0;
    unsigned char *mem_89554 = NULL;
    int64_t mem_89565_cached_sizze_90964 = 0;
    unsigned char *mem_89565 = NULL;
    int64_t mem_89566_cached_sizze_90965 = 0;
    unsigned char *mem_89566 = NULL;
    int64_t mem_89575_cached_sizze_90966 = 0;
    unsigned char *mem_89575 = NULL;
    int64_t mem_89576_cached_sizze_90967 = 0;
    unsigned char *mem_89576 = NULL;
    int64_t mem_89597_cached_sizze_90968 = 0;
    unsigned char *mem_89597 = NULL;
    int64_t mem_89598_cached_sizze_90969 = 0;
    unsigned char *mem_89598 = NULL;
    int64_t mem_89609_cached_sizze_90970 = 0;
    unsigned char *mem_89609 = NULL;
    int64_t mem_89610_cached_sizze_90971 = 0;
    unsigned char *mem_89610 = NULL;
    int64_t mem_89619_cached_sizze_90972 = 0;
    unsigned char *mem_89619 = NULL;
    int64_t mem_89626_cached_sizze_90973 = 0;
    unsigned char *mem_89626 = NULL;
    int64_t mem_89651_cached_sizze_90974 = 0;
    unsigned char *mem_89651 = NULL;
    int64_t mem_89652_cached_sizze_90975 = 0;
    unsigned char *mem_89652 = NULL;
    int64_t mem_89663_cached_sizze_90976 = 0;
    unsigned char *mem_89663 = NULL;
    int64_t mem_89664_cached_sizze_90977 = 0;
    unsigned char *mem_89664 = NULL;
    int64_t mem_89673_cached_sizze_90978 = 0;
    unsigned char *mem_89673 = NULL;
    int64_t mem_89680_cached_sizze_90979 = 0;
    unsigned char *mem_89680 = NULL;
    int64_t mem_89687_cached_sizze_90980 = 0;
    unsigned char *mem_89687 = NULL;
    int64_t mem_89694_cached_sizze_90981 = 0;
    unsigned char *mem_89694 = NULL;
    int64_t mem_89719_cached_sizze_90982 = 0;
    unsigned char *mem_89719 = NULL;
    int64_t mem_89720_cached_sizze_90983 = 0;
    unsigned char *mem_89720 = NULL;
    int64_t mem_89731_cached_sizze_90984 = 0;
    unsigned char *mem_89731 = NULL;
    int64_t mem_89732_cached_sizze_90985 = 0;
    unsigned char *mem_89732 = NULL;
    int64_t mem_89741_cached_sizze_90986 = 0;
    unsigned char *mem_89741 = NULL;
    int64_t mem_89748_cached_sizze_90987 = 0;
    unsigned char *mem_89748 = NULL;
    int64_t mem_89773_cached_sizze_90988 = 0;
    unsigned char *mem_89773 = NULL;
    int64_t mem_89778_cached_sizze_90989 = 0;
    unsigned char *mem_89778 = NULL;
    int64_t mem_89789_cached_sizze_90990 = 0;
    unsigned char *mem_89789 = NULL;
    int64_t mem_89795_cached_sizze_90991 = 0;
    unsigned char *mem_89795 = NULL;
    int64_t mem_89800_cached_sizze_90992 = 0;
    unsigned char *mem_89800 = NULL;
    int64_t mem_89816_cached_sizze_90993 = 0;
    unsigned char *mem_89816 = NULL;
    int64_t mem_89822_cached_sizze_90994 = 0;
    unsigned char *mem_89822 = NULL;
    int64_t mem_89827_cached_sizze_90995 = 0;
    unsigned char *mem_89827 = NULL;
    int64_t mem_89843_cached_sizze_90996 = 0;
    unsigned char *mem_89843 = NULL;
    int64_t mem_89844_cached_sizze_90997 = 0;
    unsigned char *mem_89844 = NULL;
    int64_t mem_89855_cached_sizze_90998 = 0;
    unsigned char *mem_89855 = NULL;
    int64_t mem_89856_cached_sizze_90999 = 0;
    unsigned char *mem_89856 = NULL;
    int64_t mem_89865_cached_sizze_91000 = 0;
    unsigned char *mem_89865 = NULL;
    int64_t mem_89866_cached_sizze_91001 = 0;
    unsigned char *mem_89866 = NULL;
    int64_t mem_89897_cached_sizze_91002 = 0;
    unsigned char *mem_89897 = NULL;
    int64_t mem_89898_cached_sizze_91003 = 0;
    unsigned char *mem_89898 = NULL;
    int64_t mem_89899_cached_sizze_91004 = 0;
    unsigned char *mem_89899 = NULL;
    int64_t mem_89912_cached_sizze_91005 = 0;
    unsigned char *mem_89912 = NULL;
    int64_t mem_89913_cached_sizze_91006 = 0;
    unsigned char *mem_89913 = NULL;
    int64_t mem_89914_cached_sizze_91007 = 0;
    unsigned char *mem_89914 = NULL;
    int64_t mem_89945_cached_sizze_91008 = 0;
    unsigned char *mem_89945 = NULL;
    int64_t mem_89946_cached_sizze_91009 = 0;
    unsigned char *mem_89946 = NULL;
    int64_t mem_89947_cached_sizze_91010 = 0;
    unsigned char *mem_89947 = NULL;
    int64_t mem_89948_cached_sizze_91011 = 0;
    unsigned char *mem_89948 = NULL;
    int64_t mem_89965_cached_sizze_91012 = 0;
    unsigned char *mem_89965 = NULL;
    int64_t mem_89966_cached_sizze_91013 = 0;
    unsigned char *mem_89966 = NULL;
    int64_t mem_89967_cached_sizze_91014 = 0;
    unsigned char *mem_89967 = NULL;
    int64_t mem_89968_cached_sizze_91015 = 0;
    unsigned char *mem_89968 = NULL;
    int64_t mem_90009_cached_sizze_91016 = 0;
    unsigned char *mem_90009 = NULL;
    int64_t mem_90016_cached_sizze_91017 = 0;
    unsigned char *mem_90016 = NULL;
    int64_t mem_90023_cached_sizze_91018 = 0;
    unsigned char *mem_90023 = NULL;
    int64_t mem_90033_cached_sizze_91019 = 0;
    unsigned char *mem_90033 = NULL;
    int64_t mem_90038_cached_sizze_91020 = 0;
    unsigned char *mem_90038 = NULL;
    int64_t mem_90049_cached_sizze_91021 = 0;
    unsigned char *mem_90049 = NULL;
    int64_t mem_90056_cached_sizze_91022 = 0;
    unsigned char *mem_90056 = NULL;
    int64_t mem_90063_cached_sizze_91023 = 0;
    unsigned char *mem_90063 = NULL;
    int64_t mem_90073_cached_sizze_91024 = 0;
    unsigned char *mem_90073 = NULL;
    int64_t mem_90078_cached_sizze_91025 = 0;
    unsigned char *mem_90078 = NULL;
    int64_t mem_90089_cached_sizze_91026 = 0;
    unsigned char *mem_90089 = NULL;
    int64_t mem_90090_cached_sizze_91027 = 0;
    unsigned char *mem_90090 = NULL;
    int64_t mem_90099_cached_sizze_91028 = 0;
    unsigned char *mem_90099 = NULL;
    int64_t mem_90100_cached_sizze_91029 = 0;
    unsigned char *mem_90100 = NULL;
    int64_t mem_90121_cached_sizze_91030 = 0;
    unsigned char *mem_90121 = NULL;
    int64_t mem_90126_cached_sizze_91031 = 0;
    unsigned char *mem_90126 = NULL;
    int64_t mem_90137_cached_sizze_91032 = 0;
    unsigned char *mem_90137 = NULL;
    int64_t mem_90138_cached_sizze_91033 = 0;
    unsigned char *mem_90138 = NULL;
    int64_t mem_90147_cached_sizze_91034 = 0;
    unsigned char *mem_90147 = NULL;
    int64_t mem_90148_cached_sizze_91035 = 0;
    unsigned char *mem_90148 = NULL;
    struct memblock mem_param_tmp_90501;
    
    mem_param_tmp_90501.references = NULL;
    
    struct memblock mem_param_tmp_90500;
    
    mem_param_tmp_90500.references = NULL;
    
    struct memblock mem_param_tmp_90499;
    
    mem_param_tmp_90499.references = NULL;
    
    struct memblock mem_param_tmp_90498;
    
    mem_param_tmp_90498.references = NULL;
    
    struct memblock mem_param_tmp_90497;
    
    mem_param_tmp_90497.references = NULL;
    
    struct memblock mem_param_tmp_90496;
    
    mem_param_tmp_90496.references = NULL;
    
    struct memblock mem_param_tmp_90495;
    
    mem_param_tmp_90495.references = NULL;
    
    struct memblock mem_param_tmp_90494;
    
    mem_param_tmp_90494.references = NULL;
    
    struct memblock mem_param_tmp_90493;
    
    mem_param_tmp_90493.references = NULL;
    
    struct memblock mem_param_tmp_90492;
    
    mem_param_tmp_90492.references = NULL;
    
    struct memblock mem_param_tmp_90491;
    
    mem_param_tmp_90491.references = NULL;
    
    struct memblock mem_param_tmp_90490;
    
    mem_param_tmp_90490.references = NULL;
    
    struct memblock mem_param_tmp_90489;
    
    mem_param_tmp_90489.references = NULL;
    
    struct memblock mem_param_tmp_90488;
    
    mem_param_tmp_90488.references = NULL;
    
    struct memblock mem_param_tmp_90487;
    
    mem_param_tmp_90487.references = NULL;
    
    struct memblock mem_param_tmp_90486;
    
    mem_param_tmp_90486.references = NULL;
    
    struct memblock mem_param_tmp_90485;
    
    mem_param_tmp_90485.references = NULL;
    
    struct memblock mem_param_tmp_90484;
    
    mem_param_tmp_90484.references = NULL;
    
    struct memblock mem_param_tmp_90483;
    
    mem_param_tmp_90483.references = NULL;
    
    struct memblock mem_param_tmp_90482;
    
    mem_param_tmp_90482.references = NULL;
    
    struct memblock mem_param_tmp_90481;
    
    mem_param_tmp_90481.references = NULL;
    
    struct memblock mem_param_tmp_90480;
    
    mem_param_tmp_90480.references = NULL;
    
    struct memblock mem_param_tmp_90479;
    
    mem_param_tmp_90479.references = NULL;
    
    struct memblock mem_param_tmp_90478;
    
    mem_param_tmp_90478.references = NULL;
    
    struct memblock mem_param_tmp_90477;
    
    mem_param_tmp_90477.references = NULL;
    
    struct memblock mem_param_tmp_90476;
    
    mem_param_tmp_90476.references = NULL;
    
    struct memblock mem_param_tmp_90475;
    
    mem_param_tmp_90475.references = NULL;
    
    struct memblock ext_mem_90265;
    
    ext_mem_90265.references = NULL;
    
    struct memblock ext_mem_90266;
    
    ext_mem_90266.references = NULL;
    
    struct memblock ext_mem_90267;
    
    ext_mem_90267.references = NULL;
    
    struct memblock mem_90263;
    
    mem_90263.references = NULL;
    
    struct memblock mem_90261;
    
    mem_90261.references = NULL;
    
    struct memblock mem_90259;
    
    mem_90259.references = NULL;
    
    struct memblock mem_90257;
    
    mem_90257.references = NULL;
    
    struct memblock ext_mem_90254;
    
    ext_mem_90254.references = NULL;
    
    struct memblock ext_mem_90255;
    
    ext_mem_90255.references = NULL;
    
    struct memblock ext_mem_90256;
    
    ext_mem_90256.references = NULL;
    
    struct memblock mem_90252;
    
    mem_90252.references = NULL;
    
    struct memblock mem_90250;
    
    mem_90250.references = NULL;
    
    struct memblock mem_90248;
    
    mem_90248.references = NULL;
    
    struct memblock mem_90246;
    
    mem_90246.references = NULL;
    
    struct memblock ext_mem_90243;
    
    ext_mem_90243.references = NULL;
    
    struct memblock ext_mem_90244;
    
    ext_mem_90244.references = NULL;
    
    struct memblock ext_mem_90245;
    
    ext_mem_90245.references = NULL;
    
    struct memblock mem_90241;
    
    mem_90241.references = NULL;
    
    struct memblock mem_90239;
    
    mem_90239.references = NULL;
    
    struct memblock mem_90237;
    
    mem_90237.references = NULL;
    
    struct memblock mem_90235;
    
    mem_90235.references = NULL;
    
    struct memblock ext_mem_90232;
    
    ext_mem_90232.references = NULL;
    
    struct memblock ext_mem_90233;
    
    ext_mem_90233.references = NULL;
    
    struct memblock ext_mem_90234;
    
    ext_mem_90234.references = NULL;
    
    struct memblock mem_90230;
    
    mem_90230.references = NULL;
    
    struct memblock mem_90228;
    
    mem_90228.references = NULL;
    
    struct memblock mem_90226;
    
    mem_90226.references = NULL;
    
    struct memblock mem_90224;
    
    mem_90224.references = NULL;
    
    struct memblock ext_mem_90221;
    
    ext_mem_90221.references = NULL;
    
    struct memblock ext_mem_90222;
    
    ext_mem_90222.references = NULL;
    
    struct memblock ext_mem_90223;
    
    ext_mem_90223.references = NULL;
    
    struct memblock mem_90219;
    
    mem_90219.references = NULL;
    
    struct memblock mem_90217;
    
    mem_90217.references = NULL;
    
    struct memblock mem_90215;
    
    mem_90215.references = NULL;
    
    struct memblock mem_90213;
    
    mem_90213.references = NULL;
    
    struct memblock ext_mem_90210;
    
    ext_mem_90210.references = NULL;
    
    struct memblock ext_mem_90211;
    
    ext_mem_90211.references = NULL;
    
    struct memblock ext_mem_90212;
    
    ext_mem_90212.references = NULL;
    
    struct memblock mem_90208;
    
    mem_90208.references = NULL;
    
    struct memblock mem_90206;
    
    mem_90206.references = NULL;
    
    struct memblock mem_90204;
    
    mem_90204.references = NULL;
    
    struct memblock mem_90202;
    
    mem_90202.references = NULL;
    
    struct memblock ext_mem_90199;
    
    ext_mem_90199.references = NULL;
    
    struct memblock ext_mem_90200;
    
    ext_mem_90200.references = NULL;
    
    struct memblock ext_mem_90201;
    
    ext_mem_90201.references = NULL;
    
    struct memblock mem_90197;
    
    mem_90197.references = NULL;
    
    struct memblock mem_90195;
    
    mem_90195.references = NULL;
    
    struct memblock mem_90193;
    
    mem_90193.references = NULL;
    
    struct memblock mem_90191;
    
    mem_90191.references = NULL;
    
    struct memblock ext_mem_90188;
    
    ext_mem_90188.references = NULL;
    
    struct memblock ext_mem_90189;
    
    ext_mem_90189.references = NULL;
    
    struct memblock ext_mem_90190;
    
    ext_mem_90190.references = NULL;
    
    struct memblock mem_90186;
    
    mem_90186.references = NULL;
    
    struct memblock mem_90184;
    
    mem_90184.references = NULL;
    
    struct memblock mem_90182;
    
    mem_90182.references = NULL;
    
    struct memblock mem_90180;
    
    mem_90180.references = NULL;
    
    struct memblock ext_mem_90177;
    
    ext_mem_90177.references = NULL;
    
    struct memblock ext_mem_90178;
    
    ext_mem_90178.references = NULL;
    
    struct memblock ext_mem_90179;
    
    ext_mem_90179.references = NULL;
    
    struct memblock mem_90175;
    
    mem_90175.references = NULL;
    
    struct memblock mem_90173;
    
    mem_90173.references = NULL;
    
    struct memblock mem_90171;
    
    mem_90171.references = NULL;
    
    struct memblock mem_90169;
    
    mem_90169.references = NULL;
    
    struct memblock mem_param_88786;
    
    mem_param_88786.references = NULL;
    
    struct memblock mem_param_88782;
    
    mem_param_88782.references = NULL;
    
    struct memblock mem_param_88778;
    
    mem_param_88778.references = NULL;
    
    struct memblock mem_param_88774;
    
    mem_param_88774.references = NULL;
    
    struct memblock mem_param_88770;
    
    mem_param_88770.references = NULL;
    
    struct memblock mem_param_88766;
    
    mem_param_88766.references = NULL;
    
    struct memblock mem_param_88762;
    
    mem_param_88762.references = NULL;
    
    struct memblock mem_param_88758;
    
    mem_param_88758.references = NULL;
    
    struct memblock mem_param_88754;
    
    mem_param_88754.references = NULL;
    
    struct memblock mem_param_88750;
    
    mem_param_88750.references = NULL;
    
    struct memblock mem_param_88746;
    
    mem_param_88746.references = NULL;
    
    struct memblock mem_param_88742;
    
    mem_param_88742.references = NULL;
    
    struct memblock mem_param_88738;
    
    mem_param_88738.references = NULL;
    
    struct memblock mem_param_88734;
    
    mem_param_88734.references = NULL;
    
    struct memblock mem_param_88730;
    
    mem_param_88730.references = NULL;
    
    struct memblock mem_param_88726;
    
    mem_param_88726.references = NULL;
    
    struct memblock mem_param_88722;
    
    mem_param_88722.references = NULL;
    
    struct memblock mem_param_88718;
    
    mem_param_88718.references = NULL;
    
    struct memblock mem_param_88714;
    
    mem_param_88714.references = NULL;
    
    struct memblock mem_param_88710;
    
    mem_param_88710.references = NULL;
    
    struct memblock mem_param_88706;
    
    mem_param_88706.references = NULL;
    
    struct memblock mem_param_88702;
    
    mem_param_88702.references = NULL;
    
    struct memblock mem_param_88698;
    
    mem_param_88698.references = NULL;
    
    struct memblock mem_param_88694;
    
    mem_param_88694.references = NULL;
    
    struct memblock mem_param_88690;
    
    mem_param_88690.references = NULL;
    
    struct memblock mem_param_88686;
    
    mem_param_88686.references = NULL;
    
    struct memblock mem_param_88682;
    
    mem_param_88682.references = NULL;
    
    struct memblock ext_mem_90349;
    
    ext_mem_90349.references = NULL;
    
    struct memblock ext_mem_90350;
    
    ext_mem_90350.references = NULL;
    
    struct memblock ext_mem_90351;
    
    ext_mem_90351.references = NULL;
    
    struct memblock ext_mem_90352;
    
    ext_mem_90352.references = NULL;
    
    struct memblock ext_mem_90353;
    
    ext_mem_90353.references = NULL;
    
    struct memblock ext_mem_90354;
    
    ext_mem_90354.references = NULL;
    
    struct memblock ext_mem_90355;
    
    ext_mem_90355.references = NULL;
    
    struct memblock ext_mem_90356;
    
    ext_mem_90356.references = NULL;
    
    struct memblock ext_mem_90357;
    
    ext_mem_90357.references = NULL;
    
    struct memblock ext_mem_90358;
    
    ext_mem_90358.references = NULL;
    
    struct memblock ext_mem_90359;
    
    ext_mem_90359.references = NULL;
    
    struct memblock ext_mem_90360;
    
    ext_mem_90360.references = NULL;
    
    struct memblock ext_mem_90361;
    
    ext_mem_90361.references = NULL;
    
    struct memblock ext_mem_90362;
    
    ext_mem_90362.references = NULL;
    
    struct memblock ext_mem_90363;
    
    ext_mem_90363.references = NULL;
    
    struct memblock ext_mem_90364;
    
    ext_mem_90364.references = NULL;
    
    struct memblock ext_mem_90365;
    
    ext_mem_90365.references = NULL;
    
    struct memblock ext_mem_90366;
    
    ext_mem_90366.references = NULL;
    
    struct memblock ext_mem_90367;
    
    ext_mem_90367.references = NULL;
    
    struct memblock ext_mem_90368;
    
    ext_mem_90368.references = NULL;
    
    struct memblock ext_mem_90369;
    
    ext_mem_90369.references = NULL;
    
    struct memblock ext_mem_90370;
    
    ext_mem_90370.references = NULL;
    
    struct memblock ext_mem_90371;
    
    ext_mem_90371.references = NULL;
    
    struct memblock ext_mem_90372;
    
    ext_mem_90372.references = NULL;
    
    struct memblock ext_mem_90373;
    
    ext_mem_90373.references = NULL;
    
    struct memblock ext_mem_90374;
    
    ext_mem_90374.references = NULL;
    
    struct memblock ext_mem_90375;
    
    ext_mem_90375.references = NULL;
    
    struct memblock mem_out_90474;
    
    mem_out_90474.references = NULL;
    
    struct memblock mem_out_90473;
    
    mem_out_90473.references = NULL;
    
    struct memblock mem_out_90472;
    
    mem_out_90472.references = NULL;
    
    struct memblock mem_out_90471;
    
    mem_out_90471.references = NULL;
    
    struct memblock mem_out_90470;
    
    mem_out_90470.references = NULL;
    
    struct memblock mem_out_90469;
    
    mem_out_90469.references = NULL;
    
    struct memblock mem_out_90468;
    
    mem_out_90468.references = NULL;
    
    struct memblock mem_out_90467;
    
    mem_out_90467.references = NULL;
    
    struct memblock mem_out_90466;
    
    mem_out_90466.references = NULL;
    
    struct memblock mem_out_90465;
    
    mem_out_90465.references = NULL;
    
    struct memblock mem_out_90464;
    
    mem_out_90464.references = NULL;
    
    struct memblock mem_out_90463;
    
    mem_out_90463.references = NULL;
    
    struct memblock mem_out_90462;
    
    mem_out_90462.references = NULL;
    
    struct memblock mem_out_90461;
    
    mem_out_90461.references = NULL;
    
    struct memblock mem_out_90460;
    
    mem_out_90460.references = NULL;
    
    struct memblock mem_out_90459;
    
    mem_out_90459.references = NULL;
    
    struct memblock mem_out_90458;
    
    mem_out_90458.references = NULL;
    
    struct memblock mem_out_90457;
    
    mem_out_90457.references = NULL;
    
    struct memblock mem_out_90456;
    
    mem_out_90456.references = NULL;
    
    struct memblock mem_out_90455;
    
    mem_out_90455.references = NULL;
    
    struct memblock mem_out_90454;
    
    mem_out_90454.references = NULL;
    
    struct memblock mem_out_90453;
    
    mem_out_90453.references = NULL;
    
    struct memblock mem_out_90452;
    
    mem_out_90452.references = NULL;
    
    struct memblock mem_out_90451;
    
    mem_out_90451.references = NULL;
    
    struct memblock mem_out_90450;
    
    mem_out_90450.references = NULL;
    
    struct memblock mem_out_90449;
    
    mem_out_90449.references = NULL;
    
    struct memblock mem_out_90448;
    
    mem_out_90448.references = NULL;
    
    struct memblock mem_88640 = ctx->constants->mem_88640;
    struct memblock mem_88641 = ctx->constants->mem_88641;
    struct memblock mem_88642 = ctx->constants->mem_88642;
    struct memblock mem_88643 = ctx->constants->mem_88643;
    struct memblock mem_88644 = ctx->constants->mem_88644;
    struct memblock mem_88645 = ctx->constants->mem_88645;
    struct memblock mem_88646 = ctx->constants->mem_88646;
    struct memblock mem_88647 = ctx->constants->mem_88647;
    struct memblock mem_88648 = ctx->constants->mem_88648;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_88787_cached_sizze_90867 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88787, &mem_88787_cached_sizze_90867, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88788_cached_sizze_90868 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_88788, &mem_88788_cached_sizze_90868, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88797_cached_sizze_90869 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_88797, &mem_88797_cached_sizze_90869, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88804_cached_sizze_90870 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88804, &mem_88804_cached_sizze_90870, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88819_cached_sizze_90871 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_88819, &mem_88819_cached_sizze_90871, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88820_cached_sizze_90872 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88820, &mem_88820_cached_sizze_90872, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88829_cached_sizze_90873 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88829, &mem_88829_cached_sizze_90873, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88836_cached_sizze_90874 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_88836, &mem_88836_cached_sizze_90874, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88851_cached_sizze_90875 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88851, &mem_88851_cached_sizze_90875, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88852_cached_sizze_90876 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88852, &mem_88852_cached_sizze_90876, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88861_cached_sizze_90877 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88861, &mem_88861_cached_sizze_90877, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88862_cached_sizze_90878 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88862, &mem_88862_cached_sizze_90878, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88883_cached_sizze_90879 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88883, &mem_88883_cached_sizze_90879, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88884_cached_sizze_90880 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88884, &mem_88884_cached_sizze_90880, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88885_cached_sizze_90881 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88885, &mem_88885_cached_sizze_90881, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88897_cached_sizze_90882 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88897, &mem_88897_cached_sizze_90882, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88898_cached_sizze_90883 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88898, &mem_88898_cached_sizze_90883, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88922_cached_sizze_90884 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88922, &mem_88922_cached_sizze_90884, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88923_cached_sizze_90885 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88923, &mem_88923_cached_sizze_90885, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88924_cached_sizze_90886 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88924, &mem_88924_cached_sizze_90886, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88925_cached_sizze_90887 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88925, &mem_88925_cached_sizze_90887, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88926_cached_sizze_90888 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88926, &mem_88926_cached_sizze_90888, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88945_cached_sizze_90889 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88945, &mem_88945_cached_sizze_90889, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88946_cached_sizze_90890 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88946, &mem_88946_cached_sizze_90890, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88947_cached_sizze_90891 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88947, &mem_88947_cached_sizze_90891, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88984_cached_sizze_90892 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88984, &mem_88984_cached_sizze_90892, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88985_cached_sizze_90893 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88985, &mem_88985_cached_sizze_90893, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88986_cached_sizze_90894 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88986, &mem_88986_cached_sizze_90894, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89002_cached_sizze_90895 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_89002, &mem_89002_cached_sizze_90895, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89003_cached_sizze_90896 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_89003, &mem_89003_cached_sizze_90896, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89004_cached_sizze_90897 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_89004, &mem_89004_cached_sizze_90897, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89017_cached_sizze_90898 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_89017, &mem_89017_cached_sizze_90898, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89018_cached_sizze_90899 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_89018, &mem_89018_cached_sizze_90899, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89019_cached_sizze_90900 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_89019, &mem_89019_cached_sizze_90900, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89065_cached_sizze_90901 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_89065, &mem_89065_cached_sizze_90901, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89066_cached_sizze_90902 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89066, &mem_89066_cached_sizze_90902, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89077_cached_sizze_90903 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89077, &mem_89077_cached_sizze_90903, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89078_cached_sizze_90904 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89078, &mem_89078_cached_sizze_90904, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89087_cached_sizze_90905 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89087, &mem_89087_cached_sizze_90905, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89088_cached_sizze_90906 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89088, &mem_89088_cached_sizze_90906, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89109_cached_sizze_90907 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89109, &mem_89109_cached_sizze_90907, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89114_cached_sizze_90908 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89114, &mem_89114_cached_sizze_90908, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89125_cached_sizze_90909 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89125, &mem_89125_cached_sizze_90909, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89130_cached_sizze_90910 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89130, &mem_89130_cached_sizze_90910, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89137_cached_sizze_90911 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89137, &mem_89137_cached_sizze_90911, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89144_cached_sizze_90912 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89144, &mem_89144_cached_sizze_90912, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89155_cached_sizze_90913 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_89155, &mem_89155_cached_sizze_90913, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89160_cached_sizze_90914 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_89160, &mem_89160_cached_sizze_90914, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89181_cached_sizze_90915 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89181, &mem_89181_cached_sizze_90915, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89182_cached_sizze_90916 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89182, &mem_89182_cached_sizze_90916, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89190_cached_sizze_90917 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89190, &mem_89190_cached_sizze_90917, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89204_cached_sizze_90918 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89204, &mem_89204_cached_sizze_90918, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89209_cached_sizze_90919 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89209, &mem_89209_cached_sizze_90919, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89220_cached_sizze_90920 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89220, &mem_89220_cached_sizze_90920, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89225_cached_sizze_90921 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89225, &mem_89225_cached_sizze_90921, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89236_cached_sizze_90922 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89236, &mem_89236_cached_sizze_90922, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89237_cached_sizze_90923 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89237, &mem_89237_cached_sizze_90923, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89246_cached_sizze_90924 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89246, &mem_89246_cached_sizze_90924, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89247_cached_sizze_90925 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89247, &mem_89247_cached_sizze_90925, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89268_cached_sizze_90926 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89268, &mem_89268_cached_sizze_90926, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89269_cached_sizze_90927 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_89269, &mem_89269_cached_sizze_90927, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89277_cached_sizze_90928 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_89277, &mem_89277_cached_sizze_90928, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89291_cached_sizze_90929 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89291, &mem_89291_cached_sizze_90929, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89292_cached_sizze_90930 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_89292, &mem_89292_cached_sizze_90930, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89300_cached_sizze_90931 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_89300, &mem_89300_cached_sizze_90931, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89314_cached_sizze_90932 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89314, &mem_89314_cached_sizze_90932, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89319_cached_sizze_90933 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89319, &mem_89319_cached_sizze_90933, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89330_cached_sizze_90934 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89330, &mem_89330_cached_sizze_90934, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89335_cached_sizze_90935 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89335, &mem_89335_cached_sizze_90935, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89346_cached_sizze_90936 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_89346, &mem_89346_cached_sizze_90936, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89351_cached_sizze_90937 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_89351, &mem_89351_cached_sizze_90937, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89362_cached_sizze_90938 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_89362, &mem_89362_cached_sizze_90938, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89363_cached_sizze_90939 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_89363, &mem_89363_cached_sizze_90939, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89372_cached_sizze_90940 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_89372, &mem_89372_cached_sizze_90940, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89373_cached_sizze_90941 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_89373, &mem_89373_cached_sizze_90941, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89386_cached_sizze_90942 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_89386, &mem_89386_cached_sizze_90942, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89387_cached_sizze_90943 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_89387, &mem_89387_cached_sizze_90943, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89400_cached_sizze_90944 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_89400, &mem_89400_cached_sizze_90944, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89401_cached_sizze_90945 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_89401, &mem_89401_cached_sizze_90945, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89422_cached_sizze_90946 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89422, &mem_89422_cached_sizze_90946, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89429_cached_sizze_90947 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_89429, &mem_89429_cached_sizze_90947, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89434_cached_sizze_90948 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_89434, &mem_89434_cached_sizze_90948, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89445_cached_sizze_90949 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89445, &mem_89445_cached_sizze_90949, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89450_cached_sizze_90950 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89450, &mem_89450_cached_sizze_90950, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89461_cached_sizze_90951 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_89461, &mem_89461_cached_sizze_90951, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89462_cached_sizze_90952 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_89462, &mem_89462_cached_sizze_90952, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89471_cached_sizze_90953 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_89471, &mem_89471_cached_sizze_90953, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89472_cached_sizze_90954 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_89472, &mem_89472_cached_sizze_90954, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89493_cached_sizze_90955 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_89493, &mem_89493_cached_sizze_90955, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89498_cached_sizze_90956 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_89498, &mem_89498_cached_sizze_90956, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89509_cached_sizze_90957 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89509, &mem_89509_cached_sizze_90957, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89514_cached_sizze_90958 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89514, &mem_89514_cached_sizze_90958, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89525_cached_sizze_90959 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89525, &mem_89525_cached_sizze_90959, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89532_cached_sizze_90960 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89532, &mem_89532_cached_sizze_90960, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89539_cached_sizze_90961 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89539, &mem_89539_cached_sizze_90961, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89549_cached_sizze_90962 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89549, &mem_89549_cached_sizze_90962, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89554_cached_sizze_90963 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89554, &mem_89554_cached_sizze_90963, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89565_cached_sizze_90964 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89565, &mem_89565_cached_sizze_90964, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89566_cached_sizze_90965 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89566, &mem_89566_cached_sizze_90965, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89575_cached_sizze_90966 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89575, &mem_89575_cached_sizze_90966, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89576_cached_sizze_90967 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89576, &mem_89576_cached_sizze_90967, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89597_cached_sizze_90968 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_89597, &mem_89597_cached_sizze_90968, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89598_cached_sizze_90969 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89598, &mem_89598_cached_sizze_90969, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89609_cached_sizze_90970 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89609, &mem_89609_cached_sizze_90970, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89610_cached_sizze_90971 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_89610, &mem_89610_cached_sizze_90971, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89619_cached_sizze_90972 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_89619, &mem_89619_cached_sizze_90972, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89626_cached_sizze_90973 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89626, &mem_89626_cached_sizze_90973, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89651_cached_sizze_90974 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_89651, &mem_89651_cached_sizze_90974, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89652_cached_sizze_90975 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_89652, &mem_89652_cached_sizze_90975, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89663_cached_sizze_90976 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89663, &mem_89663_cached_sizze_90976, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89664_cached_sizze_90977 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89664, &mem_89664_cached_sizze_90977, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89673_cached_sizze_90978 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89673, &mem_89673_cached_sizze_90978, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89680_cached_sizze_90979 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89680, &mem_89680_cached_sizze_90979, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89687_cached_sizze_90980 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89687, &mem_89687_cached_sizze_90980, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89694_cached_sizze_90981 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89694, &mem_89694_cached_sizze_90981, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89719_cached_sizze_90982 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89719, &mem_89719_cached_sizze_90982, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89720_cached_sizze_90983 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_89720, &mem_89720_cached_sizze_90983, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89731_cached_sizze_90984 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_89731, &mem_89731_cached_sizze_90984, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89732_cached_sizze_90985 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89732, &mem_89732_cached_sizze_90985, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89741_cached_sizze_90986 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89741, &mem_89741_cached_sizze_90986, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89748_cached_sizze_90987 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_89748, &mem_89748_cached_sizze_90987, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89773_cached_sizze_90988 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_89773, &mem_89773_cached_sizze_90988, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89778_cached_sizze_90989 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89778, &mem_89778_cached_sizze_90989, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89789_cached_sizze_90990 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_89789, &mem_89789_cached_sizze_90990, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89795_cached_sizze_90991 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89795, &mem_89795_cached_sizze_90991, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89800_cached_sizze_90992 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89800, &mem_89800_cached_sizze_90992, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89816_cached_sizze_90993 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_89816, &mem_89816_cached_sizze_90993, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89822_cached_sizze_90994 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89822, &mem_89822_cached_sizze_90994, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89827_cached_sizze_90995 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89827, &mem_89827_cached_sizze_90995, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89843_cached_sizze_90996 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89843, &mem_89843_cached_sizze_90996, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89844_cached_sizze_90997 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89844, &mem_89844_cached_sizze_90997, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89855_cached_sizze_90998 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_89855, &mem_89855_cached_sizze_90998, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89856_cached_sizze_90999 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_89856, &mem_89856_cached_sizze_90999, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89865_cached_sizze_91000 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_89865, &mem_89865_cached_sizze_91000, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89866_cached_sizze_91001 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_89866, &mem_89866_cached_sizze_91001, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89897_cached_sizze_91002 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89897, &mem_89897_cached_sizze_91002, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89898_cached_sizze_91003 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89898, &mem_89898_cached_sizze_91003, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89899_cached_sizze_91004 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89899, &mem_89899_cached_sizze_91004, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89912_cached_sizze_91005 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89912, &mem_89912_cached_sizze_91005, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89913_cached_sizze_91006 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89913, &mem_89913_cached_sizze_91006, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89914_cached_sizze_91007 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89914, &mem_89914_cached_sizze_91007, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89945_cached_sizze_91008 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89945, &mem_89945_cached_sizze_91008, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89946_cached_sizze_91009 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89946, &mem_89946_cached_sizze_91009, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89947_cached_sizze_91010 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89947, &mem_89947_cached_sizze_91010, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89948_cached_sizze_91011 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89948, &mem_89948_cached_sizze_91011, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89965_cached_sizze_91012 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89965, &mem_89965_cached_sizze_91012, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89966_cached_sizze_91013 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89966, &mem_89966_cached_sizze_91013, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89967_cached_sizze_91014 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89967, &mem_89967_cached_sizze_91014, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89968_cached_sizze_91015 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89968, &mem_89968_cached_sizze_91015, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90009_cached_sizze_91016 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_90009, &mem_90009_cached_sizze_91016, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90016_cached_sizze_91017 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_90016, &mem_90016_cached_sizze_91017, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90023_cached_sizze_91018 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_90023, &mem_90023_cached_sizze_91018, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90033_cached_sizze_91019 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_90033, &mem_90033_cached_sizze_91019, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90038_cached_sizze_91020 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_90038, &mem_90038_cached_sizze_91020, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90049_cached_sizze_91021 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_90049, &mem_90049_cached_sizze_91021, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90056_cached_sizze_91022 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_90056, &mem_90056_cached_sizze_91022, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90063_cached_sizze_91023 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_90063, &mem_90063_cached_sizze_91023, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90073_cached_sizze_91024 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_90073, &mem_90073_cached_sizze_91024, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90078_cached_sizze_91025 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_90078, &mem_90078_cached_sizze_91025, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90089_cached_sizze_91026 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_90089, &mem_90089_cached_sizze_91026, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90090_cached_sizze_91027 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_90090, &mem_90090_cached_sizze_91027, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90099_cached_sizze_91028 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_90099, &mem_90099_cached_sizze_91028, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90100_cached_sizze_91029 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_90100, &mem_90100_cached_sizze_91029, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90121_cached_sizze_91030 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_90121, &mem_90121_cached_sizze_91030, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90126_cached_sizze_91031 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_90126, &mem_90126_cached_sizze_91031, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90137_cached_sizze_91032 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_90137, &mem_90137_cached_sizze_91032, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90138_cached_sizze_91033 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_90138, &mem_90138_cached_sizze_91033, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90147_cached_sizze_91034 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_90147, &mem_90147_cached_sizze_91034, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_90148_cached_sizze_91035 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_90148, &mem_90148_cached_sizze_91035, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:462:5-467:51
    if (memblock_set(ctx, &mem_param_88682, &wdown_mem_88649, "wdown_mem_88649") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88686, &wkey_mem_88650, "wkey_mem_88650") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88690, &wout_mem_88651, "wout_mem_88651") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88694, &wpe_mem_88652, "wpe_mem_88652") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88698, &wqry_mem_88653, "wqry_mem_88653") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88702, &wte_mem_88654, "wte_mem_88654") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88706, &wup_mem_88655, "wup_mem_88655") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88710, &wval_mem_88656, "wval_mem_88656") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88714, &wvoc_mem_88657, "wvoc_mem_88657") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88718, &wdown_mem_88658, "wdown_mem_88658") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88722, &wkey_mem_88659, "wkey_mem_88659") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88726, &wout_mem_88660, "wout_mem_88660") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88730, &wpe_mem_88661, "wpe_mem_88661") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88734, &wqry_mem_88662, "wqry_mem_88662") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88738, &wte_mem_88663, "wte_mem_88663") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88742, &wup_mem_88664, "wup_mem_88664") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88746, &wval_mem_88665, "wval_mem_88665") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88750, &wvoc_mem_88666, "wvoc_mem_88666") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88754, &wdown_mem_88667, "wdown_mem_88667") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88758, &wkey_mem_88668, "wkey_mem_88668") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88762, &wout_mem_88669, "wout_mem_88669") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88766, &wpe_mem_88670, "wpe_mem_88670") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88770, &wqry_mem_88671, "wqry_mem_88671") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88774, &wte_mem_88672, "wte_mem_88672") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88778, &wup_mem_88673, "wup_mem_88673") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88782, &wval_mem_88674, "wval_mem_88674") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_88786, &wvoc_mem_88675, "wvoc_mem_88675") != 0)
        return 1;
    for (int64_t step_81997 = 0; step_81997 < (int64_t) 500; step_81997++) {
        // futhark/microgpt.fut:464:16-25
        
        int64_t dl_82025 = ((int64_t *) dls_mem_88677.mem)[step_81997];
        
        // futhark/microgpt.fut:377:37-40
        
        int64_t zl_rhs_82030 = sub64(dl_82025, (int64_t) 1);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87802 = 0; i_87802 < (int64_t) 16; i_87802++) {
            // futhark/microgpt.fut:377:25-81
            
            bool cond_83821 = slt64(i_87802, zl_rhs_82030);
            
            // futhark/microgpt.fut:377:56-59
            
            int64_t zeze_lhs_83822 = add64((int64_t) 1, i_87802);
            
            // futhark/microgpt.fut:377:47-60
            
            bool x_83823 = sle64((int64_t) 0, zeze_lhs_83822);
            
            // futhark/microgpt.fut:377:47-60
            
            bool y_83824 = slt64(zeze_lhs_83822, (int64_t) 16);
            
            // futhark/microgpt.fut:377:47-60
            
            bool bounds_check_83825 = x_83823 && y_83824;
            
            // futhark/microgpt.fut:9:27-39
            
            bool loop_not_taken_83826 = !cond_83821;
            
            // futhark/microgpt.fut:9:27-39
            
            bool protect_assert_disj_83827 = bounds_check_83825 || loop_not_taken_83826;
            
            // futhark/microgpt.fut:377:47-60
            
            bool index_certs_83828;
            
            if (!protect_assert_disj_83827) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_83822, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:377:47-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:377:3-83\n   #6  futhark/microgpt.fut:435:18-38\n   #7  futhark/microgpt.fut:445:26-451:31\n   #8  futhark/microgpt.fut:467:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            int64_t tmp_83843 = ((int64_t *) seqs_mem_88678.mem)[step_81997 * (int64_t) 16 + i_87802];
            
            // futhark/microgpt.fut:437:37-51
            
            bool x_83844 = sle64((int64_t) 0, tmp_83843);
            
            // futhark/microgpt.fut:437:37-51
            
            bool y_83845 = slt64(tmp_83843, (int64_t) 27);
            
            // futhark/microgpt.fut:437:37-51
            
            bool bounds_check_83846 = x_83844 && y_83845;
            
            // futhark/microgpt.fut:437:37-51
            
            bool index_certs_83847;
            
            if (!bounds_check_83846) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_83843, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:437:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:437:16-55\n   #6  futhark/microgpt.fut:445:26-451:31\n   #7  futhark/microgpt.fut:467:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:377:47-60
            
            int64_t zeze_lhs_83829;
            
            if (cond_83821) {
                int64_t x_87611 = ((int64_t *) seqs_mem_88678.mem)[step_81997 * (int64_t) 16 + zeze_lhs_83822];
                
                zeze_lhs_83829 = x_87611;
            } else {
                zeze_lhs_83829 = (int64_t) 0;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87792 = 0; i_87792 < (int64_t) 27; i_87792++) {
                // futhark/microgpt.fut:377:61-65
                
                bool cond_t_res_83833 = zeze_lhs_83829 == i_87792;
                
                // futhark/microgpt.fut:9:27-39
                
                bool x_83834 = cond_83821 && cond_t_res_83833;
                
                // futhark/microgpt.fut:377:25-81
                
                double lifted_lambda_res_83835;
                
                if (x_83834) {
                    lifted_lambda_res_83835 = 1.0;
                } else {
                    lifted_lambda_res_83835 = 0.0;
                }
                ((double *) mem_88797)[i_87792] = lifted_lambda_res_83835;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87796 = 0; i_87796 < (int64_t) 16; i_87796++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_83854 = ((double *) mem_param_88702.mem)[tmp_83843 * (int64_t) 16 + i_87796];
                
                ((double *) mem_88804)[i_87796] = lifted_lambda_res_83854;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88787, i_87802 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88804, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88788, i_87802 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88797, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87817 = 0; i_87817 < (int64_t) 16; i_87817++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87807 = 0; i_87807 < (int64_t) 16; i_87807++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_83879 = ((double *) mem_param_88694.mem)[i_87817 * (int64_t) 16 + i_87807];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_83880 = ((double *) mem_88787)[i_87817 * (int64_t) 16 + i_87807];
                
                // futhark/microgpt.fut:232:39-75
                
                double zp_res_83881 = zp_lhs_83879 + zp_rhs_83880;
                
                ((double *) mem_88829)[i_87807] = zp_res_83881;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87811 = 0; i_87811 < (int64_t) 27; i_87811++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_83895 = ((double *) mem_88788)[i_87817 * (int64_t) 27 + i_87811];
                
                // futhark/microgpt.fut:264:43-85
                
                double zt_res_83896 = -6.25e-2 * zt_rhs_83895;
                
                ((double *) mem_88836)[i_87811] = zt_res_83896;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88819, i_87817 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88836, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88820, i_87817 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88829, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87831 = 0; i_87831 < (int64_t) 16; i_87831++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_83915;
            double r_83917 = 0.0;
            
            for (int64_t i_83916 = 0; i_83916 < (int64_t) 16; i_83916++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_83918 = ((double *) mem_88820)[i_87831 * (int64_t) 16 + i_83916];
                
                // futhark/microgpt.fut:233:70-103
                
                double zt_res_83919 = zt_lhs_83918 * zt_lhs_83918;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_83920 = r_83917 + zt_res_83919;
                double r_tmp_90539 = zp_res_83920;
                
                r_83917 = r_tmp_90539;
            }
            defunc_0_lifted_lambda_res_83915 = r_83917;
            // futhark/microgpt.fut:233:50-121
            
            double zs_res_83921 = defunc_0_lifted_lambda_res_83915 / 16.0;
            
            // futhark/microgpt.fut:234:23-53
            
            double zp_res_83922 = 1.0e-5 + zs_res_83921;
            
            // futhark/microgpt.fut:234:15-53
            
            double sqrt_res_83923 = futrts_sqrt64(zp_res_83922);
            
            // futhark/microgpt.fut:235:25-35
            
            double zs_res_83924 = 1.0 / sqrt_res_83923;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87824 = 0; i_87824 < (int64_t) 16; i_87824++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_85920 = ((double *) mem_88820)[i_87831 * (int64_t) 16 + i_87824];
                
                // futhark/microgpt.fut:235:5-35
                
                double zt_res_85921 = zs_res_83924 * zt_lhs_85920;
                
                // futhark/microgpt.fut:307:45-86
                
                double zt_res_85929 = zt_lhs_85920 * zt_lhs_85920;
                
                ((double *) mem_88861)[i_87824] = zt_res_85929;
                ((double *) mem_88862)[i_87824] = zt_res_85921;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88851, i_87831 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88861, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88852, i_87831 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88862, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87847 = 0; i_87847 < (int64_t) 16; i_87847++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_84023;
            double r_84025 = 0.0;
            
            for (int64_t i_84024 = 0; i_84024 < (int64_t) 16; i_84024++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_84026 = ((double *) mem_88852)[i_87847 * (int64_t) 16 + i_84024];
                
                // futhark/microgpt.fut:236:71-106
                
                double zt_res_84027 = zt_lhs_84026 * zt_lhs_84026;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_84028 = r_84025 + zt_res_84027;
                double r_tmp_90545 = zp_res_84028;
                
                r_84025 = r_tmp_90545;
            }
            defunc_0_lifted_lambda_res_84023 = r_84025;
            // futhark/microgpt.fut:236:50-124
            
            double zs_res_84029 = defunc_0_lifted_lambda_res_84023 / 16.0;
            
            // futhark/microgpt.fut:237:24-54
            
            double zp_res_84030 = 1.0e-5 + zs_res_84029;
            
            // futhark/microgpt.fut:237:16-54
            
            double sqrt_res_84031 = futrts_sqrt64(zp_res_84030);
            
            // futhark/microgpt.fut:238:25-36
            
            double zs_res_84032 = 1.0 / sqrt_res_84031;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87838 = 0; i_87838 < (int64_t) 16; i_87838++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_85949 = ((double *) mem_88852)[i_87847 * (int64_t) 16 + i_87838];
                
                // futhark/microgpt.fut:238:5-36
                
                double zt_res_85950 = zs_res_84032 * zt_lhs_85949;
                
                // futhark/microgpt.fut:300:45-86
                
                double zt_res_85958 = zt_lhs_85949 * zt_lhs_85949;
                
                ((double *) mem_88897)[i_87838] = zt_res_85958;
                ((double *) mem_88898)[i_87838] = zt_res_85950;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_84066;
            double r_84068 = 0.0;
            
            for (int64_t i_84067 = 0; i_84067 < (int64_t) 16; i_84067++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_84069 = ((double *) mem_88851)[i_87847 * (int64_t) 16 + i_84067];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_84070 = r_84068 + lifted_lambda_res_84069;
                double r_tmp_90548 = zp_res_84070;
                
                r_84068 = r_tmp_90548;
            }
            defunc_0_lifted_lambda_res_84066 = r_84068;
            // futhark/microgpt.fut:308:36-94
            
            double zs_res_84071 = defunc_0_lifted_lambda_res_84066 / 16.0;
            
            ((double *) mem_88883)[i_87847] = zs_res_84071;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88884, i_87847 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88897, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88885, i_87847 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88898, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87871 = 0; i_87871 < (int64_t) 16; i_87871++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87857 = 0; i_87857 < (int64_t) 16; i_87857++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_86021;
                double r_86023 = 0.0;
                
                for (int64_t i_86022 = 0; i_86022 < (int64_t) 16; i_86022++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_86024 = ((double *) mem_param_88698.mem)[i_87857 * (int64_t) 16 + i_86022];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_86025 = ((double *) mem_88885)[i_87871 * (int64_t) 16 + i_86022];
                    
                    // futhark/microgpt.fut:239:63-102
                    
                    double zt_res_86026 = zt_lhs_86024 * zt_rhs_86025;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_86027 = r_86023 + zt_res_86026;
                    double r_tmp_90557 = zp_res_86027;
                    
                    r_86023 = r_tmp_90557;
                }
                defunc_0_lifted_lambda_res_86021 = r_86023;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_86034;
                double r_86036 = 0.0;
                
                for (int64_t i_86035 = 0; i_86035 < (int64_t) 16; i_86035++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_86037 = ((double *) mem_param_88686.mem)[i_87857 * (int64_t) 16 + i_86035];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_86038 = ((double *) mem_88885)[i_87871 * (int64_t) 16 + i_86035];
                    
                    // futhark/microgpt.fut:240:63-102
                    
                    double zt_res_86039 = zt_lhs_86037 * zt_rhs_86038;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_86040 = r_86036 + zt_res_86039;
                    double r_tmp_90558 = zp_res_86040;
                    
                    r_86036 = r_tmp_90558;
                }
                defunc_0_lifted_lambda_res_86034 = r_86036;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_86050;
                double r_86052 = 0.0;
                
                for (int64_t i_86051 = 0; i_86051 < (int64_t) 16; i_86051++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_86053 = ((double *) mem_param_88710.mem)[i_87857 * (int64_t) 16 + i_86051];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_86054 = ((double *) mem_88885)[i_87871 * (int64_t) 16 + i_86051];
                    
                    // futhark/microgpt.fut:241:63-102
                    
                    double zt_res_86055 = zt_lhs_86053 * zt_rhs_86054;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_86056 = r_86052 + zt_res_86055;
                    double r_tmp_90559 = zp_res_86056;
                    
                    r_86052 = r_tmp_90559;
                }
                defunc_0_lifted_lambda_res_86050 = r_86052;
                ((double *) mem_88945)[i_87857] = defunc_0_lifted_lambda_res_86050;
                ((double *) mem_88946)[i_87857] = defunc_0_lifted_lambda_res_86034;
                ((double *) mem_88947)[i_87857] = defunc_0_lifted_lambda_res_86021;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_84413;
            double r_84415 = 0.0;
            
            for (int64_t i_84414 = 0; i_84414 < (int64_t) 16; i_84414++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_84416 = ((double *) mem_88884)[i_87871 * (int64_t) 16 + i_84414];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_84417 = r_84415 + lifted_lambda_res_84416;
                double r_tmp_90560 = zp_res_84417;
                
                r_84415 = r_tmp_90560;
            }
            defunc_0_lifted_lambda_res_84413 = r_84415;
            // futhark/microgpt.fut:301:36-94
            
            double zs_res_84418 = defunc_0_lifted_lambda_res_84413 / 16.0;
            
            // futhark/microgpt.fut:309:43-55
            
            double zp_lhs_84432 = ((double *) mem_88883)[i_87871];
            
            // futhark/microgpt.fut:309:43-83
            
            double zp_res_84433 = 1.0e-5 + zp_lhs_84432;
            
            // futhark/microgpt.fut:309:35-83
            
            double sqrt_res_84434 = futrts_sqrt64(zp_res_84433);
            
            ((double *) mem_88922)[i_87871] = sqrt_res_84434;
            ((double *) mem_88923)[i_87871] = zs_res_84418;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88924, i_87871 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88945, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88925, i_87871 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88946, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88926, i_87871 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88947, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87903 = 0; i_87903 < (int64_t) 4; i_87903++) {
            // futhark/microgpt.fut:242:67-70
            
            int64_t zp_lhs_84506 = mul64((int64_t) 4, i_87903);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87893 = 0; i_87893 < (int64_t) 16; i_87893++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87883 = 0; i_87883 < (int64_t) 4; i_87883++) {
                    // futhark/microgpt.fut:242:72-79
                    
                    int64_t tmp_86214 = add64(zp_lhs_84506, i_87883);
                    
                    // futhark/microgpt.fut:242:48-81
                    
                    bool x_86215 = sle64((int64_t) 0, tmp_86214);
                    
                    // futhark/microgpt.fut:242:48-81
                    
                    bool y_86216 = slt64(tmp_86214, (int64_t) 16);
                    
                    // futhark/microgpt.fut:242:48-81
                    
                    bool bounds_check_86217 = x_86215 && y_86216;
                    
                    // futhark/microgpt.fut:242:48-81
                    
                    bool index_certs_86218;
                    
                    if (!bounds_check_86217) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_86214, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:242:48-81\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:242:12-82\n   #9  futhark/microgpt.fut:440:5-76\n   #10 futhark/microgpt.fut:445:26-451:31\n   #11 futhark/microgpt.fut:467:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_86219 = ((double *) mem_88926)[i_87893 * (int64_t) 16 + tmp_86214];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_86227 = ((double *) mem_88925)[i_87893 * (int64_t) 16 + tmp_86214];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_86238 = ((double *) mem_88924)[i_87893 * (int64_t) 16 + tmp_86214];
                    
                    ((double *) mem_89017)[i_87883] = lifted_lambda_res_86238;
                    ((double *) mem_89018)[i_87883] = lifted_lambda_res_86227;
                    ((double *) mem_89019)[i_87883] = lifted_lambda_res_86219;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_89002, i_87893 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89017, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_89003, i_87893 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89018, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_89004, i_87893 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89019, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_88984, i_87903 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_89002, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_88985, i_87903 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_89003, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_88986, i_87903 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_89004, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87959 = 0; i_87959 < (int64_t) 4; i_87959++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87918 = 0; i_87918 < (int64_t) 16; i_87918++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87911 = 0; i_87911 < (int64_t) 16; i_87911++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_86317;
                    double r_86319 = 0.0;
                    
                    for (int64_t i_86318 = 0; i_86318 < (int64_t) 4; i_86318++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_86320 = ((double *) mem_88986)[i_87959 * (int64_t) 64 + i_87918 * (int64_t) 4 + i_86318];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_86321 = ((double *) mem_88985)[i_87959 * (int64_t) 64 + i_87911 * (int64_t) 4 + i_86318];
                        
                        // futhark/microgpt.fut:245:110-163
                        
                        double zt_res_86322 = zt_lhs_86320 * zt_rhs_86321;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_86323 = r_86319 + zt_res_86322;
                        double r_tmp_90576 = zp_res_86323;
                        
                        r_86319 = r_tmp_90576;
                    }
                    defunc_0_lifted_lambda_res_86317 = r_86319;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_86330;
                    double r_86332 = 0.0;
                    
                    for (int64_t i_86331 = 0; i_86331 < (int64_t) 4; i_86331++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_86333 = ((double *) mem_88986)[i_87959 * (int64_t) 64 + i_87918 * (int64_t) 4 + i_86331];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_86334 = ((double *) mem_88985)[i_87959 * (int64_t) 64 + i_87911 * (int64_t) 4 + i_86331];
                        
                        // futhark/microgpt.fut:284:75-134
                        
                        double zt_res_86335 = zt_lhs_86333 * zt_rhs_86334;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_86336 = r_86332 + zt_res_86335;
                        double r_tmp_90577 = zp_res_86336;
                        
                        r_86332 = r_tmp_90577;
                    }
                    defunc_0_lifted_lambda_res_86330 = r_86332;
                    ((double *) mem_89087)[i_87911] = defunc_0_lifted_lambda_res_86330;
                    ((double *) mem_89088)[i_87911] = defunc_0_lifted_lambda_res_86317;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_89077, i_87918 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89087, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_89078, i_87918 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89088, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87927 = 0; i_87927 < (int64_t) 16; i_87927++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87923 = 0; i_87923 < (int64_t) 16; i_87923++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_84615 = ((double *) mem_89078)[i_87927 * (int64_t) 16 + i_87923];
                    
                    // futhark/microgpt.fut:246:47-78
                    
                    double zs_res_84616 = zs_lhs_84615 / 2.0;
                    double zp_rhs_84617 = ((double *) masks_mem_88676.mem)[step_81997 * (int64_t) 256 + i_87927 * (int64_t) 16 + i_87923];
                    
                    // futhark/microgpt.fut:246:65-102
                    
                    double zp_res_84618 = zs_res_84616 + zp_rhs_84617;
                    
                    ((double *) mem_89114)[i_87923] = zp_res_84618;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_89109, i_87927 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89114, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87945 = 0; i_87945 < (int64_t) 16; i_87945++) {
                // futhark/microgpt.fut:103:13-33
                
                double defunc_0_reduce_res_87632;
                double redout_87929 = -INFINITY;
                
                for (int64_t i_87930 = 0; i_87930 < (int64_t) 16; i_87930++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_86354 = ((double *) mem_89109)[i_87945 * (int64_t) 16 + i_87930];
                    
                    // futhark/microgpt.fut:103:13-33
                    
                    double max_res_84639 = fmax64(lifted_lambda_res_86354, redout_87929);
                    double redout_tmp_90581 = max_res_84639;
                    
                    redout_87929 = redout_tmp_90581;
                }
                defunc_0_reduce_res_87632 = redout_87929;
                // futhark/microgpt.fut:113:47-56
                
                double neg_res_84640 = -defunc_0_reduce_res_87632;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87933 = 0; i_87933 < (int64_t) 16; i_87933++) {
                    // futhark/microgpt.fut:113:38-41
                    
                    double lifted_lambda_res_84647 = ((double *) mem_89109)[i_87945 * (int64_t) 16 + i_87933];
                    
                    // futhark/microgpt.fut:113:38-56
                    
                    double zp_res_84648 = neg_res_84640 + lifted_lambda_res_84647;
                    
                    // futhark/microgpt.fut:113:31-56
                    
                    double exp_res_84649 = futrts_exp64(zp_res_84648);
                    
                    ((double *) mem_89130)[i_87933] = exp_res_84649;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_84651;
                double r_84653 = 0.0;
                
                for (int64_t i_84652 = 0; i_84652 < (int64_t) 16; i_84652++) {
                    // futhark/microgpt.fut:114:32-39
                    
                    double lifted_lambda_res_84654 = ((double *) mem_89130)[i_84652];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_84655 = r_84653 + lifted_lambda_res_84654;
                    double r_tmp_90583 = zp_res_84655;
                    
                    r_84653 = r_tmp_90583;
                }
                defunc_0_lifted_lambda_res_84651 = r_84653;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87937 = 0; i_87937 < (int64_t) 16; i_87937++) {
                    // futhark/microgpt.fut:115:23-30
                    
                    double zs_lhs_84662 = ((double *) mem_89130)[i_87937];
                    
                    // futhark/microgpt.fut:115:23-40
                    
                    double zs_res_84663 = zs_lhs_84662 / defunc_0_lifted_lambda_res_84651;
                    
                    ((double *) mem_89137)[i_87937] = zs_res_84663;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87941 = 0; i_87941 < (int64_t) 16; i_87941++) {
                    // futhark/microgpt.fut:248:4-14
                    
                    double lifted_lambda_res_84671 = ((double *) mem_89137)[i_87941];
                    
                    ((double *) mem_89144)[i_87941] = lifted_lambda_res_84671;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_89125, i_87945 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89144, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87953 = 0; i_87953 < (int64_t) 16; i_87953++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87949 = 0; i_87949 < (int64_t) 4; i_87949++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_84686;
                    double r_84688 = 0.0;
                    
                    for (int64_t i_84687 = 0; i_84687 < (int64_t) 16; i_84687++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_84689 = ((double *) mem_89125)[i_87953 * (int64_t) 16 + i_84687];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_84690 = ((double *) mem_88984)[i_87959 * (int64_t) 64 + i_84687 * (int64_t) 4 + i_87949];
                        
                        // futhark/microgpt.fut:249:26-72
                        
                        double zt_res_84691 = zt_lhs_84689 * zt_rhs_84690;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_84692 = r_84688 + zt_res_84691;
                        double r_tmp_90588 = zp_res_84692;
                        
                        r_84688 = r_tmp_90588;
                    }
                    defunc_0_lifted_lambda_res_84686 = r_84688;
                    ((double *) mem_89160)[i_87949] = defunc_0_lifted_lambda_res_84686;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_89155, i_87953 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89160, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_89065, i_87959 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_89077, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_89066, i_87959 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_89155, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87970 = 0; i_87970 < (int64_t) 16; i_87970++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87964 = 0; i_87964 < (int64_t) 16; i_87964++) {
                // futhark/microgpt.fut:250:52-55
                
                int64_t tmp_84741 = sdiv64(i_87964, (int64_t) 4);
                
                // futhark/microgpt.fut:250:41-57
                
                bool x_84742 = sle64((int64_t) 0, tmp_84741);
                
                // futhark/microgpt.fut:250:41-57
                
                bool y_84743 = slt64(tmp_84741, (int64_t) 4);
                
                // futhark/microgpt.fut:250:41-57
                
                bool bounds_check_84744 = x_84742 && y_84743;
                
                // futhark/microgpt.fut:250:41-57
                
                bool index_certs_84745;
                
                if (!bounds_check_84744) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_84741, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:250:41-57\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:250:12-78\n   #6  futhark/microgpt.fut:440:5-76\n   #7  futhark/microgpt.fut:445:26-451:31\n   #8  futhark/microgpt.fut:467:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:250:72-75
                
                int64_t tmp_84746 = smod64(i_87964, (int64_t) 4);
                
                // futhark/microgpt.fut:250:41-77
                
                bool x_84747 = sle64((int64_t) 0, tmp_84746);
                
                // futhark/microgpt.fut:250:41-77
                
                bool y_84748 = slt64(tmp_84746, (int64_t) 4);
                
                // futhark/microgpt.fut:250:41-77
                
                bool bounds_check_84749 = x_84747 && y_84748;
                
                // futhark/microgpt.fut:250:41-77
                
                bool index_certs_84750;
                
                if (!bounds_check_84749) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_84746, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:250:41-77\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:250:12-78\n   #6  futhark/microgpt.fut:440:5-76\n   #7  futhark/microgpt.fut:445:26-451:31\n   #8  futhark/microgpt.fut:467:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_84751 = ((double *) mem_89066)[tmp_84741 * (int64_t) 64 + i_87970 * (int64_t) 4 + tmp_84746];
                
                ((double *) mem_89190)[i_87964] = lifted_lambda_res_84751;
            }
            // futhark/microgpt.fut:302:43-55
            
            double zp_lhs_84759 = ((double *) mem_88923)[i_87970];
            
            // futhark/microgpt.fut:302:43-83
            
            double zp_res_84760 = 1.0e-5 + zp_lhs_84759;
            
            // futhark/microgpt.fut:302:35-83
            
            double sqrt_res_84761 = futrts_sqrt64(zp_res_84760);
            
            ((double *) mem_89181)[i_87970] = sqrt_res_84761;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89182, i_87970 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89190, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87979 = 0; i_87979 < (int64_t) 16; i_87979++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87975 = 0; i_87975 < (int64_t) 16; i_87975++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_82416;
                double r_82418 = 0.0;
                
                for (int64_t i_82417 = 0; i_82417 < (int64_t) 16; i_82417++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_82419 = ((double *) mem_param_88690.mem)[i_87975 * (int64_t) 16 + i_82417];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_82420 = ((double *) mem_89182)[i_87979 * (int64_t) 16 + i_82417];
                    
                    // futhark/microgpt.fut:251:63-103
                    
                    double zt_res_82421 = zt_lhs_82419 * zt_rhs_82420;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82422 = r_82418 + zt_res_82421;
                    double r_tmp_90594 = zp_res_82422;
                    
                    r_82418 = r_tmp_90594;
                }
                defunc_0_lifted_lambda_res_82416 = r_82418;
                ((double *) mem_89209)[i_87975] = defunc_0_lifted_lambda_res_82416;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89204, i_87979 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89209, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87987 = 0; i_87987 < (int64_t) 16; i_87987++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87983 = 0; i_87983 < (int64_t) 16; i_87983++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_82437 = ((double *) mem_89204)[i_87987 * (int64_t) 16 + i_87983];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_82438 = ((double *) mem_88852)[i_87987 * (int64_t) 16 + i_87983];
                
                // futhark/microgpt.fut:252:42-80
                
                double zp_res_82439 = zp_lhs_82437 + zp_rhs_82438;
                
                ((double *) mem_89225)[i_87983] = zp_res_82439;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89220, i_87987 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89225, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88000 = 0; i_88000 < (int64_t) 16; i_88000++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_84779;
            double r_84781 = 0.0;
            
            for (int64_t i_84780 = 0; i_84780 < (int64_t) 16; i_84780++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_84782 = ((double *) mem_89220)[i_88000 * (int64_t) 16 + i_84780];
                
                // futhark/microgpt.fut:253:75-114
                
                double zt_res_84783 = zt_lhs_84782 * zt_lhs_84782;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_84784 = r_84781 + zt_res_84783;
                double r_tmp_90599 = zp_res_84784;
                
                r_84781 = r_tmp_90599;
            }
            defunc_0_lifted_lambda_res_84779 = r_84781;
            // futhark/microgpt.fut:253:54-132
            
            double zs_res_84785 = defunc_0_lifted_lambda_res_84779 / 16.0;
            
            // futhark/microgpt.fut:254:24-55
            
            double zp_res_84786 = 1.0e-5 + zs_res_84785;
            
            // futhark/microgpt.fut:254:16-55
            
            double sqrt_res_84787 = futrts_sqrt64(zp_res_84786);
            
            // futhark/microgpt.fut:255:28-39
            
            double zs_res_84788 = 1.0 / sqrt_res_84787;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87993 = 0; i_87993 < (int64_t) 16; i_87993++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_86395 = ((double *) mem_89220)[i_88000 * (int64_t) 16 + i_87993];
                
                // futhark/microgpt.fut:255:5-39
                
                double zt_res_86396 = zs_res_84788 * zt_lhs_86395;
                
                // futhark/microgpt.fut:275:42-81
                
                double zt_res_86404 = zt_lhs_86395 * zt_lhs_86395;
                
                ((double *) mem_89246)[i_87993] = zt_res_86404;
                ((double *) mem_89247)[i_87993] = zt_res_86396;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89236, i_88000 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89246, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89237, i_88000 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89247, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88011 = 0; i_88011 < (int64_t) 16; i_88011++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88005 = 0; i_88005 < (int64_t) 64; i_88005++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_84836;
                double r_84838 = 0.0;
                
                for (int64_t i_84837 = 0; i_84837 < (int64_t) 16; i_84837++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_84839 = ((double *) mem_param_88706.mem)[i_88005 * (int64_t) 16 + i_84837];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_84840 = ((double *) mem_89237)[i_88011 * (int64_t) 16 + i_84837];
                    
                    // futhark/microgpt.fut:256:63-102
                    
                    double zt_res_84841 = zt_lhs_84839 * zt_rhs_84840;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_84842 = r_84838 + zt_res_84841;
                    double r_tmp_90605 = zp_res_84842;
                    
                    r_84838 = r_tmp_90605;
                }
                defunc_0_lifted_lambda_res_84836 = r_84838;
                ((double *) mem_89277)[i_88005] = defunc_0_lifted_lambda_res_84836;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_84850;
            double r_84852 = 0.0;
            
            for (int64_t i_84851 = 0; i_84851 < (int64_t) 16; i_84851++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_84853 = ((double *) mem_89236)[i_88011 * (int64_t) 16 + i_84851];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_84854 = r_84852 + lifted_lambda_res_84853;
                double r_tmp_90606 = zp_res_84854;
                
                r_84852 = r_tmp_90606;
            }
            defunc_0_lifted_lambda_res_84850 = r_84852;
            // futhark/microgpt.fut:276:34-88
            
            double zs_res_84855 = defunc_0_lifted_lambda_res_84850 / 16.0;
            
            ((double *) mem_89268)[i_88011] = zs_res_84855;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89269, i_88011 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89277, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88022 = 0; i_88022 < (int64_t) 16; i_88022++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88016 = 0; i_88016 < (int64_t) 64; i_88016++) {
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_84879 = ((double *) mem_89269)[i_88022 * (int64_t) 64 + i_88016];
                
                // futhark/microgpt.fut:257:41-69
                
                double max_res_84880 = fmax64(0.0, max_arg0_84879);
                
                ((double *) mem_89300)[i_88016] = max_res_84880;
            }
            // futhark/microgpt.fut:277:41-51
            
            double zp_lhs_84888 = ((double *) mem_89268)[i_88022];
            
            // futhark/microgpt.fut:277:41-79
            
            double zp_res_84889 = 1.0e-5 + zp_lhs_84888;
            
            // futhark/microgpt.fut:277:33-79
            
            double sqrt_res_84890 = futrts_sqrt64(zp_res_84889);
            
            ((double *) mem_89291)[i_88022] = sqrt_res_84890;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89292, i_88022 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89300, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88031 = 0; i_88031 < (int64_t) 16; i_88031++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88027 = 0; i_88027 < (int64_t) 16; i_88027++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_82518;
                double r_82520 = 0.0;
                
                for (int64_t i_82519 = 0; i_82519 < (int64_t) 64; i_82519++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_82521 = ((double *) mem_param_88682.mem)[i_88027 * (int64_t) 64 + i_82519];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_82522 = ((double *) mem_89292)[i_88031 * (int64_t) 64 + i_82519];
                    
                    // futhark/microgpt.fut:258:63-104
                    
                    double zt_res_82523 = zt_lhs_82521 * zt_rhs_82522;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82524 = r_82520 + zt_res_82523;
                    double r_tmp_90612 = zp_res_82524;
                    
                    r_82520 = r_tmp_90612;
                }
                defunc_0_lifted_lambda_res_82518 = r_82520;
                ((double *) mem_89319)[i_88027] = defunc_0_lifted_lambda_res_82518;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89314, i_88031 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89319, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88039 = 0; i_88039 < (int64_t) 16; i_88039++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88035 = 0; i_88035 < (int64_t) 16; i_88035++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_82539 = ((double *) mem_89314)[i_88039 * (int64_t) 16 + i_88035];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_82540 = ((double *) mem_89220)[i_88039 * (int64_t) 16 + i_88035];
                
                // futhark/microgpt.fut:259:42-81
                
                double zp_res_82541 = zp_lhs_82539 + zp_rhs_82540;
                
                ((double *) mem_89335)[i_88035] = zp_res_82541;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89330, i_88039 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89335, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88047 = 0; i_88047 < (int64_t) 16; i_88047++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88043 = 0; i_88043 < (int64_t) 27; i_88043++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_82556;
                double r_82558 = 0.0;
                
                for (int64_t i_82557 = 0; i_82557 < (int64_t) 16; i_82557++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_82559 = ((double *) mem_param_88714.mem)[i_88043 * (int64_t) 16 + i_82557];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_82560 = ((double *) mem_89330)[i_88047 * (int64_t) 16 + i_82557];
                    
                    // futhark/microgpt.fut:260:63-103
                    
                    double zt_res_82561 = zt_lhs_82559 * zt_rhs_82560;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82562 = r_82558 + zt_res_82561;
                    double r_tmp_90617 = zp_res_82562;
                    
                    r_82558 = r_tmp_90617;
                }
                defunc_0_lifted_lambda_res_82556 = r_82558;
                ((double *) mem_89351)[i_88043] = defunc_0_lifted_lambda_res_82556;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89346, i_88047 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89351, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88077 = 0; i_88077 < (int64_t) 16; i_88077++) {
            // futhark/microgpt.fut:103:13-33
            
            double defunc_0_reduce_res_87652;
            double defunc_0_reduce_res_87653;
            double redout_88049;
            double redout_88050;
            
            redout_88049 = -INFINITY;
            redout_88050 = -INFINITY;
            for (int64_t i_88051 = 0; i_88051 < (int64_t) 27; i_88051++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_86472 = ((double *) mem_89346)[i_88077 * (int64_t) 27 + i_88051];
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_84920 = fmax64(lifted_lambda_res_86472, redout_88049);
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_84972 = fmax64(lifted_lambda_res_86472, redout_88050);
                double redout_tmp_90620 = max_res_84920;
                double redout_tmp_90621 = max_res_84972;
                
                redout_88049 = redout_tmp_90620;
                redout_88050 = redout_tmp_90621;
            }
            defunc_0_reduce_res_87652 = redout_88049;
            defunc_0_reduce_res_87653 = redout_88050;
            // futhark/microgpt.fut:113:47-56
            
            double neg_res_84921 = -defunc_0_reduce_res_87652;
            
            // futhark/microgpt.fut:113:47-56
            
            double neg_res_84973 = -defunc_0_reduce_res_87653;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88056 = 0; i_88056 < (int64_t) 27; i_88056++) {
                // futhark/microgpt.fut:113:38-41
                
                double lifted_lambda_res_86511 = ((double *) mem_89346)[i_88077 * (int64_t) 27 + i_88056];
                
                // futhark/microgpt.fut:113:38-56
                
                double zp_res_86512 = neg_res_84921 + lifted_lambda_res_86511;
                
                // futhark/microgpt.fut:113:31-56
                
                double exp_res_86513 = futrts_exp64(zp_res_86512);
                
                // futhark/microgpt.fut:113:38-56
                
                double zp_res_86521 = neg_res_84973 + lifted_lambda_res_86511;
                
                // futhark/microgpt.fut:113:31-56
                
                double exp_res_86522 = futrts_exp64(zp_res_86521);
                
                ((double *) mem_89372)[i_88056] = exp_res_86522;
                ((double *) mem_89373)[i_88056] = exp_res_86513;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_84932;
            double r_84934 = 0.0;
            
            for (int64_t i_84933 = 0; i_84933 < (int64_t) 27; i_84933++) {
                // futhark/microgpt.fut:114:32-39
                
                double lifted_lambda_res_84935 = ((double *) mem_89373)[i_84933];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_84936 = r_84934 + lifted_lambda_res_84935;
                double r_tmp_90624 = zp_res_84936;
                
                r_84934 = r_tmp_90624;
            }
            defunc_0_lifted_lambda_res_84932 = r_84934;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_84984;
            double r_84986 = 0.0;
            
            for (int64_t i_84985 = 0; i_84985 < (int64_t) 27; i_84985++) {
                // futhark/microgpt.fut:114:32-39
                
                double lifted_lambda_res_84987 = ((double *) mem_89372)[i_84985];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_84988 = r_84986 + lifted_lambda_res_84987;
                double r_tmp_90625 = zp_res_84988;
                
                r_84986 = r_tmp_90625;
            }
            defunc_0_lifted_lambda_res_84984 = r_84986;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88063 = 0; i_88063 < (int64_t) 27; i_88063++) {
                // futhark/microgpt.fut:115:23-30
                
                double zs_lhs_86540 = ((double *) mem_89373)[i_88063];
                
                // futhark/microgpt.fut:115:23-40
                
                double zs_res_86541 = zs_lhs_86540 / defunc_0_lifted_lambda_res_84932;
                
                // futhark/microgpt.fut:115:23-30
                
                double zs_lhs_86548 = ((double *) mem_89372)[i_88063];
                
                // futhark/microgpt.fut:115:23-40
                
                double zs_res_86549 = zs_lhs_86548 / defunc_0_lifted_lambda_res_84984;
                
                ((double *) mem_89386)[i_88063] = zs_res_86549;
                ((double *) mem_89387)[i_88063] = zs_res_86541;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88070 = 0; i_88070 < (int64_t) 27; i_88070++) {
                // futhark/microgpt.fut:266:4-14
                
                double lifted_lambda_res_86567 = ((double *) mem_89387)[i_88070];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_86574 = ((double *) mem_88819)[i_88077 * (int64_t) 27 + i_88070];
                
                // futhark/microgpt.fut:268:36-46
                
                double zs_rhs_86575 = ((double *) mem_89386)[i_88070];
                
                // futhark/microgpt.fut:268:28-46
                
                double zs_res_86576 = 1.0 / zs_rhs_86575;
                
                // futhark/microgpt.fut:268:5-46
                
                double zt_res_86577 = zt_lhs_86574 * zs_res_86576;
                
                ((double *) mem_89400)[i_88070] = zt_res_86577;
                ((double *) mem_89401)[i_88070] = lifted_lambda_res_86567;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89362, i_88077 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89400, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89363, i_88077 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89401, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88082 = 0; i_88082 < (int64_t) 16; i_88082++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_82696;
            double r_82698 = 0.0;
            
            for (int64_t i_82697 = 0; i_82697 < (int64_t) 27; i_82697++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_82699 = ((double *) mem_89362)[i_88082 * (int64_t) 27 + i_82697];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_82700 = ((double *) mem_89363)[i_88082 * (int64_t) 27 + i_82697];
                
                // futhark/microgpt.fut:269:54-93
                
                double zt_res_82701 = zt_lhs_82699 * zt_rhs_82700;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_82702 = r_82698 + zt_res_82701;
                double r_tmp_90631 = zp_res_82702;
                
                r_82698 = r_tmp_90631;
            }
            defunc_0_lifted_lambda_res_82696 = r_82698;
            ((double *) mem_89422)[i_88082] = defunc_0_lifted_lambda_res_82696;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88090 = 0; i_88090 < (int64_t) 16; i_88090++) {
            // futhark/microgpt.fut:270:94-104
            
            double neg_arg0_82710 = ((double *) mem_89422)[i_88090];
            
            // futhark/microgpt.fut:270:88-104
            
            double neg_res_82711 = -neg_arg0_82710;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88086 = 0; i_88086 < (int64_t) 27; i_88086++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_82718 = ((double *) mem_89363)[i_88090 * (int64_t) 27 + i_88086];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_82719 = ((double *) mem_89362)[i_88090 * (int64_t) 27 + i_88086];
                
                // futhark/microgpt.fut:270:65-104
                
                double zp_res_82720 = neg_res_82711 + zp_lhs_82719;
                
                // futhark/microgpt.fut:270:42-104
                
                double zt_res_82721 = zt_lhs_82718 * zp_res_82720;
                
                ((double *) mem_89434)[i_88086] = zt_res_82721;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89429, i_88090 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89434, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88098 = 0; i_88098 < (int64_t) 16; i_88098++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88094 = 0; i_88094 < (int64_t) 16; i_88094++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_82736;
                double r_82738 = 0.0;
                
                for (int64_t i_82737 = 0; i_82737 < (int64_t) 27; i_82737++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_82739 = ((double *) mem_param_88714.mem)[i_82737 * (int64_t) 16 + i_88094];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_82740 = ((double *) mem_89429)[i_88098 * (int64_t) 27 + i_82737];
                    
                    // futhark/microgpt.fut:271:63-103
                    
                    double zt_res_82741 = zt_lhs_82739 * zt_rhs_82740;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82742 = r_82738 + zt_res_82741;
                    double r_tmp_90636 = zp_res_82742;
                    
                    r_82738 = r_tmp_90636;
                }
                defunc_0_lifted_lambda_res_82736 = r_82738;
                ((double *) mem_89450)[i_88094] = defunc_0_lifted_lambda_res_82736;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89445, i_88098 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89450, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88111 = 0; i_88111 < (int64_t) 16; i_88111++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88104 = 0; i_88104 < (int64_t) 64; i_88104++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_86605;
                double r_86607 = 0.0;
                
                for (int64_t i_86606 = 0; i_86606 < (int64_t) 16; i_86606++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_86608 = ((double *) mem_param_88682.mem)[i_86606 * (int64_t) 64 + i_88104];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_86609 = ((double *) mem_89445)[i_88111 * (int64_t) 16 + i_86606];
                    
                    // futhark/microgpt.fut:272:63-104
                    
                    double zt_res_86610 = zt_lhs_86608 * zt_rhs_86609;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_86611 = r_86607 + zt_res_86610;
                    double r_tmp_90641 = zp_res_86611;
                    
                    r_86607 = r_tmp_90641;
                }
                defunc_0_lifted_lambda_res_86605 = r_86607;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_86618;
                double r_86620 = 0.0;
                
                for (int64_t i_86619 = 0; i_86619 < (int64_t) 16; i_86619++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_86621 = ((double *) mem_89445)[i_86619 * (int64_t) 16 + i_88111];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_86622 = ((double *) mem_89292)[i_86619 * (int64_t) 64 + i_88104];
                    
                    // futhark/microgpt.fut:322:69-112
                    
                    double zt_res_86623 = zt_lhs_86621 * zt_rhs_86622;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_86624 = r_86620 + zt_res_86623;
                    double r_tmp_90642 = zp_res_86624;
                    
                    r_86620 = r_tmp_90642;
                }
                defunc_0_lifted_lambda_res_86618 = r_86620;
                ((double *) mem_89471)[i_88104] = defunc_0_lifted_lambda_res_86618;
                ((double *) mem_89472)[i_88104] = defunc_0_lifted_lambda_res_86605;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89461, i_88111 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89471, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89462, i_88111 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89472, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88120 = 0; i_88120 < (int64_t) 16; i_88120++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88116 = 0; i_88116 < (int64_t) 64; i_88116++) {
                // futhark/microgpt.fut:4:11-25
                
                double indicatorp_arg0_82778 = ((double *) mem_89269)[i_88120 * (int64_t) 64 + i_88116];
                
                // futhark/microgpt.fut:125:42-54
                
                double max_res_82779 = fmax64(0.0, indicatorp_arg0_82778);
                
                // futhark/microgpt.fut:125:35-54
                
                double sgn_res_82780 = fsignum64(max_res_82779);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_82781 = ((double *) mem_89462)[i_88120 * (int64_t) 64 + i_88116];
                
                // futhark/microgpt.fut:273:43-94
                
                double zt_res_82782 = sgn_res_82780 * zt_rhs_82781;
                
                ((double *) mem_89498)[i_88116] = zt_res_82782;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89493, i_88120 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89498, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88128 = 0; i_88128 < (int64_t) 16; i_88128++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88124 = 0; i_88124 < (int64_t) 16; i_88124++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_82797;
                double r_82799 = 0.0;
                
                for (int64_t i_82798 = 0; i_82798 < (int64_t) 64; i_82798++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_82800 = ((double *) mem_param_88706.mem)[i_82798 * (int64_t) 16 + i_88124];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_82801 = ((double *) mem_89493)[i_88128 * (int64_t) 64 + i_82798];
                    
                    // futhark/microgpt.fut:274:63-102
                    
                    double zt_res_82802 = zt_lhs_82800 * zt_rhs_82801;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82803 = r_82799 + zt_res_82802;
                    double r_tmp_90647 = zp_res_82803;
                    
                    r_82799 = r_tmp_90647;
                }
                defunc_0_lifted_lambda_res_82797 = r_82799;
                ((double *) mem_89514)[i_88124] = defunc_0_lifted_lambda_res_82797;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89509, i_88128 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89514, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88132 = 0; i_88132 < (int64_t) 16; i_88132++) {
            // futhark/microgpt.fut:278:50-61
            
            double zs_rhs_82851 = ((double *) mem_89291)[i_88132];
            
            // futhark/microgpt.fut:278:42-61
            
            double zs_res_82852 = 1.0 / zs_rhs_82851;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_82853;
            double r_82855 = 0.0;
            
            for (int64_t i_82854 = 0; i_82854 < (int64_t) 16; i_82854++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_82856 = ((double *) mem_89220)[i_88132 * (int64_t) 16 + i_82854];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_82857 = ((double *) mem_89509)[i_88132 * (int64_t) 16 + i_82854];
                
                // futhark/microgpt.fut:278:91-134
                
                double zt_res_82858 = zt_lhs_82856 * zt_rhs_82857;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_82859 = r_82855 + zt_res_82858;
                double r_tmp_90649 = zp_res_82859;
                
                r_82855 = r_tmp_90649;
            }
            defunc_0_lifted_lambda_res_82853 = r_82855;
            // futhark/microgpt.fut:278:69-162
            
            double zt_res_82860 = zs_res_82852 * defunc_0_lifted_lambda_res_82853;
            
            // futhark/microgpt.fut:278:46-162
            
            double zt_res_82861 = zs_res_82852 * zt_res_82860;
            
            // futhark/microgpt.fut:278:34-162
            
            double neg_res_82862 = -zt_res_82861;
            
            ((double *) mem_89525)[i_88132] = neg_res_82862;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88136 = 0; i_88136 < (int64_t) 16; i_88136++) {
            // futhark/microgpt.fut:279:35-46
            
            double zt_lhs_82870 = ((double *) mem_89525)[i_88136];
            
            // futhark/microgpt.fut:279:88-99
            
            double zp_lhs_82871 = ((double *) mem_89268)[i_88136];
            
            // futhark/microgpt.fut:279:88-127
            
            double zp_res_82872 = 1.0e-5 + zp_lhs_82871;
            
            // futhark/microgpt.fut:279:80-127
            
            double sqrt_res_82873 = futrts_sqrt64(zp_res_82872);
            
            // futhark/microgpt.fut:279:66-129
            
            double zt_res_82874 = 2.0 * sqrt_res_82873;
            
            // futhark/microgpt.fut:279:52-129
            
            double zs_res_82875 = 1.0 / zt_res_82874;
            
            // futhark/microgpt.fut:279:35-129
            
            double zt_res_82876 = zt_lhs_82870 * zs_res_82875;
            
            ((double *) mem_89532)[i_88136] = zt_res_82876;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88140 = 0; i_88140 < (int64_t) 16; i_88140++) {
            // futhark/microgpt.fut:280:45-57
            
            double zs_lhs_82884 = ((double *) mem_89532)[i_88140];
            
            // futhark/microgpt.fut:280:45-72
            
            double zs_res_82885 = zs_lhs_82884 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_90652 = 0; nest_i_90652 < (int64_t) 16; nest_i_90652++) {
                ((double *) mem_89539)[i_88140 * (int64_t) 16 + nest_i_90652] = zs_res_82885;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88148 = 0; i_88148 < (int64_t) 16; i_88148++) {
            // futhark/microgpt.fut:281:105-116
            
            double zs_rhs_82894 = ((double *) mem_89291)[i_88148];
            
            // futhark/microgpt.fut:281:97-116
            
            double zs_res_82895 = 1.0 / zs_rhs_82894;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88144 = 0; i_88144 < (int64_t) 16; i_88144++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_82902 = ((double *) mem_89445)[i_88148 * (int64_t) 16 + i_88144];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_82903 = ((double *) mem_89509)[i_88148 * (int64_t) 16 + i_88144];
                
                // futhark/microgpt.fut:281:72-116
                
                double zt_res_82904 = zs_res_82895 * zt_lhs_82903;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_82905 = ((double *) mem_89220)[i_88148 * (int64_t) 16 + i_88144];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_82906 = ((double *) mem_89539)[i_88148 * (int64_t) 16 + i_88144];
                
                // futhark/microgpt.fut:281:124-168
                
                double zt_res_82907 = zt_lhs_82905 * zt_rhs_82906;
                
                // futhark/microgpt.fut:281:92-168
                
                double zp_res_82908 = zt_res_82904 + zt_res_82907;
                
                // futhark/microgpt.fut:281:119-220
                
                double zp_res_82909 = zt_res_82907 + zp_res_82908;
                
                // futhark/microgpt.fut:281:45-220
                
                double zp_res_82910 = zp_lhs_82902 + zp_res_82909;
                
                ((double *) mem_89554)[i_88144] = zp_res_82910;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89549, i_88148 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89554, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88161 = 0; i_88161 < (int64_t) 16; i_88161++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88154 = 0; i_88154 < (int64_t) 16; i_88154++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_86647;
                double r_86649 = 0.0;
                
                for (int64_t i_86648 = 0; i_86648 < (int64_t) 16; i_86648++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_86650 = ((double *) mem_param_88690.mem)[i_86648 * (int64_t) 16 + i_88154];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_86651 = ((double *) mem_89549)[i_88161 * (int64_t) 16 + i_86648];
                    
                    // futhark/microgpt.fut:282:67-112
                    
                    double zt_res_86652 = zt_lhs_86650 * zt_rhs_86651;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_86653 = r_86649 + zt_res_86652;
                    double r_tmp_90659 = zp_res_86653;
                    
                    r_86649 = r_tmp_90659;
                }
                defunc_0_lifted_lambda_res_86647 = r_86649;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_86660;
                double r_86662 = 0.0;
                
                for (int64_t i_86661 = 0; i_86661 < (int64_t) 16; i_86661++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_86663 = ((double *) mem_89549)[i_86661 * (int64_t) 16 + i_88161];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_86664 = ((double *) mem_89182)[i_86661 * (int64_t) 16 + i_88154];
                    
                    // futhark/microgpt.fut:320:68-112
                    
                    double zt_res_86665 = zt_lhs_86663 * zt_rhs_86664;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_86666 = r_86662 + zt_res_86665;
                    double r_tmp_90660 = zp_res_86666;
                    
                    r_86662 = r_tmp_90660;
                }
                defunc_0_lifted_lambda_res_86660 = r_86662;
                ((double *) mem_89575)[i_88154] = defunc_0_lifted_lambda_res_86660;
                ((double *) mem_89576)[i_88154] = defunc_0_lifted_lambda_res_86647;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89565, i_88161 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89575, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89566, i_88161 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89576, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88183 = 0; i_88183 < (int64_t) 4; i_88183++) {
            // futhark/microgpt.fut:283:74-77
            
            int64_t zp_lhs_85124 = mul64((int64_t) 4, i_88183);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88176 = 0; i_88176 < (int64_t) 16; i_88176++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_88166 = 0; i_88166 < (int64_t) 4; i_88166++) {
                    // futhark/microgpt.fut:283:79-87
                    
                    int64_t tmp_86688 = add64(zp_lhs_85124, i_88166);
                    
                    // futhark/microgpt.fut:283:52-89
                    
                    bool x_86689 = sle64((int64_t) 0, tmp_86688);
                    
                    // futhark/microgpt.fut:283:52-89
                    
                    bool y_86690 = slt64(tmp_86688, (int64_t) 16);
                    
                    // futhark/microgpt.fut:283:52-89
                    
                    bool bounds_check_86691 = x_86689 && y_86690;
                    
                    // futhark/microgpt.fut:283:52-89
                    
                    bool index_certs_86692;
                    
                    if (!bounds_check_86691) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_86688, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:283:52-89\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:283:13-90\n   #9  futhark/microgpt.fut:440:5-76\n   #10 futhark/microgpt.fut:445:26-451:31\n   #11 futhark/microgpt.fut:467:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_86693 = ((double *) mem_89566)[i_88176 * (int64_t) 16 + tmp_86688];
                    
                    ((double *) mem_89619)[i_88166] = lifted_lambda_res_86693;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_88170 = 0; i_88170 < (int64_t) 16; i_88170++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_86707 = ((double *) mem_89065)[i_88183 * (int64_t) 256 + i_88176 * (int64_t) 16 + i_88170];
                    
                    // futhark/microgpt.fut:285:55-97
                    
                    double zs_res_86708 = zs_lhs_86707 / 2.0;
                    double zp_rhs_86709 = ((double *) masks_mem_88676.mem)[step_81997 * (int64_t) 256 + i_88176 * (int64_t) 16 + i_88170];
                    
                    // futhark/microgpt.fut:285:84-123
                    
                    double zp_res_86710 = zs_res_86708 + zp_rhs_86709;
                    
                    ((double *) mem_89626)[i_88170] = zp_res_86710;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_89609, i_88176 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89626, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_89610, i_88176 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89619, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_89597, i_88183 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_89609, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_89598, i_88183 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_89610, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88214 = 0; i_88214 < (int64_t) 4; i_88214++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88207 = 0; i_88207 < (int64_t) 16; i_88207++) {
                // futhark/microgpt.fut:4:11-25
                
                double defunc_0_reduce_res_87673;
                double redout_88187 = -INFINITY;
                
                for (int64_t i_88189 = 0; i_88189 < (int64_t) 16; i_88189++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_86836 = ((double *) mem_89597)[i_88214 * (int64_t) 256 + i_88207 * (int64_t) 16 + i_88189];
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_86847;
                    double r_86849 = 0.0;
                    
                    for (int64_t i_86848 = 0; i_86848 < (int64_t) 4; i_86848++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_86850 = ((double *) mem_89598)[i_88214 * (int64_t) 64 + i_88207 * (int64_t) 4 + i_86848];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_86851 = ((double *) mem_88984)[i_88214 * (int64_t) 64 + i_88189 * (int64_t) 4 + i_86848];
                        
                        // futhark/microgpt.fut:288:75-135
                        
                        double zt_res_86852 = zt_lhs_86850 * zt_rhs_86851;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_86853 = r_86849 + zt_res_86852;
                        double r_tmp_90673 = zp_res_86853;
                        
                        r_86849 = r_tmp_90673;
                    }
                    defunc_0_lifted_lambda_res_86847 = r_86849;
                    // futhark/microgpt.fut:103:13-33
                    
                    double max_res_86747 = fmax64(lifted_lambda_res_86836, redout_88187);
                    
                    ((double *) mem_89673)[i_88189] = defunc_0_lifted_lambda_res_86847;
                    
                    double redout_tmp_90671 = max_res_86747;
                    
                    redout_88187 = redout_tmp_90671;
                }
                defunc_0_reduce_res_87673 = redout_88187;
                // futhark/microgpt.fut:113:47-56
                
                double neg_res_86748 = -defunc_0_reduce_res_87673;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_88193 = 0; i_88193 < (int64_t) 16; i_88193++) {
                    // futhark/microgpt.fut:113:38-41
                    
                    double lifted_lambda_res_86755 = ((double *) mem_89597)[i_88214 * (int64_t) 256 + i_88207 * (int64_t) 16 + i_88193];
                    
                    // futhark/microgpt.fut:113:38-56
                    
                    double zp_res_86756 = neg_res_86748 + lifted_lambda_res_86755;
                    
                    // futhark/microgpt.fut:113:31-56
                    
                    double exp_res_86757 = futrts_exp64(zp_res_86756);
                    
                    ((double *) mem_89680)[i_88193] = exp_res_86757;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_86759;
                double r_86761 = 0.0;
                
                for (int64_t i_86760 = 0; i_86760 < (int64_t) 16; i_86760++) {
                    // futhark/microgpt.fut:114:32-39
                    
                    double lifted_lambda_res_86762 = ((double *) mem_89680)[i_86760];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_86763 = r_86761 + lifted_lambda_res_86762;
                    double r_tmp_90675 = zp_res_86763;
                    
                    r_86761 = r_tmp_90675;
                }
                defunc_0_lifted_lambda_res_86759 = r_86761;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_88197 = 0; i_88197 < (int64_t) 16; i_88197++) {
                    // futhark/microgpt.fut:115:23-30
                    
                    double zs_lhs_86770 = ((double *) mem_89680)[i_88197];
                    
                    // futhark/microgpt.fut:115:23-40
                    
                    double zs_res_86771 = zs_lhs_86770 / defunc_0_lifted_lambda_res_86759;
                    
                    ((double *) mem_89687)[i_88197] = zs_res_86771;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_88201 = 0; i_88201 < (int64_t) 16; i_88201++) {
                    // futhark/microgpt.fut:287:4-16
                    
                    double lifted_lambda_res_86779 = ((double *) mem_89687)[i_88201];
                    
                    ((double *) mem_89694)[i_88201] = lifted_lambda_res_86779;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_89663, i_88207 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89673, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_89664, i_88207 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89694, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_89651, i_88214 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_89663, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_89652, i_88214 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_89664, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88236 = 0; i_88236 < (int64_t) 4; i_88236++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88229 = 0; i_88229 < (int64_t) 16; i_88229++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_88219 = 0; i_88219 < (int64_t) 16; i_88219++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_86889 = ((double *) mem_89651)[i_88236 * (int64_t) 256 + i_88229 * (int64_t) 16 + i_88219];
                    
                    ((double *) mem_89741)[i_88219] = lifted_lambda_res_86889;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_88223 = 0; i_88223 < (int64_t) 4; i_88223++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_86903;
                    double r_86905 = 0.0;
                    
                    for (int64_t i_86904 = 0; i_86904 < (int64_t) 16; i_86904++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_86906 = ((double *) mem_89652)[i_88236 * (int64_t) 256 + i_86904 * (int64_t) 16 + i_88229];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_86907 = ((double *) mem_89598)[i_88236 * (int64_t) 64 + i_86904 * (int64_t) 4 + i_88223];
                        
                        // futhark/microgpt.fut:293:75-136
                        
                        double zt_res_86908 = zt_lhs_86906 * zt_rhs_86907;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_86909 = r_86905 + zt_res_86908;
                        double r_tmp_90684 = zp_res_86909;
                        
                        r_86905 = r_tmp_90684;
                    }
                    defunc_0_lifted_lambda_res_86903 = r_86905;
                    ((double *) mem_89748)[i_88223] = defunc_0_lifted_lambda_res_86903;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_89731, i_88229 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89748, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_89732, i_88229 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89741, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_89719, i_88236 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_89731, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_89720, i_88236 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_89732, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88245 = 0; i_88245 < (int64_t) 4; i_88245++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88241 = 0; i_88241 < (int64_t) 16; i_88241++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_83129;
                double r_83131 = 0.0;
                
                for (int64_t i_83130 = 0; i_83130 < (int64_t) 16; i_83130++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_83132 = ((double *) mem_89720)[i_88245 * (int64_t) 256 + i_88241 * (int64_t) 16 + i_83130];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_83133 = ((double *) mem_89652)[i_88245 * (int64_t) 256 + i_88241 * (int64_t) 16 + i_83130];
                    
                    // futhark/microgpt.fut:290:66-127
                    
                    double zt_res_83134 = zt_lhs_83132 * zt_rhs_83133;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_83135 = r_83131 + zt_res_83134;
                    double r_tmp_90687 = zp_res_83135;
                    
                    r_83131 = r_tmp_90687;
                }
                defunc_0_lifted_lambda_res_83129 = r_83131;
                ((double *) mem_89778)[i_88241] = defunc_0_lifted_lambda_res_83129;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89773, i_88245 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89778, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88257 = 0; i_88257 < (int64_t) 4; i_88257++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88253 = 0; i_88253 < (int64_t) 16; i_88253++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_83150 = ((double *) mem_89773)[i_88257 * (int64_t) 16 + i_88253];
                
                // futhark/microgpt.fut:291:122-148
                
                double neg_res_83151 = -neg_arg0_83150;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_88249 = 0; i_88249 < (int64_t) 16; i_88249++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_83158 = ((double *) mem_89652)[i_88257 * (int64_t) 256 + i_88253 * (int64_t) 16 + i_88249];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_83159 = ((double *) mem_89720)[i_88257 * (int64_t) 256 + i_88253 * (int64_t) 16 + i_88249];
                    
                    // futhark/microgpt.fut:291:88-148
                    
                    double zp_res_83160 = neg_res_83151 + zp_lhs_83159;
                    
                    // futhark/microgpt.fut:291:54-148
                    
                    double zt_res_83161 = zt_lhs_83158 * zp_res_83160;
                    
                    ((double *) mem_89800)[i_88249] = zt_res_83161;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_89795, i_88253 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89800, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_89789, i_88257 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_89795, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88269 = 0; i_88269 < (int64_t) 4; i_88269++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88265 = 0; i_88265 < (int64_t) 16; i_88265++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_88261 = 0; i_88261 < (int64_t) 16; i_88261++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_83183 = ((double *) mem_89789)[i_88269 * (int64_t) 256 + i_88265 * (int64_t) 16 + i_88261];
                    
                    // futhark/microgpt.fut:292:54-96
                    
                    double zs_res_83184 = zs_lhs_83183 / 2.0;
                    
                    ((double *) mem_89827)[i_88261] = zs_res_83184;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_89822, i_88265 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89827, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_89816, i_88269 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_89822, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88289 = 0; i_88289 < (int64_t) 4; i_88289++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88282 = 0; i_88282 < (int64_t) 16; i_88282++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_88275 = 0; i_88275 < (int64_t) 4; i_88275++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_86990;
                    double r_86992 = 0.0;
                    
                    for (int64_t i_86991 = 0; i_86991 < (int64_t) 16; i_86991++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_86993 = ((double *) mem_88986)[i_88289 * (int64_t) 64 + i_86991 * (int64_t) 4 + i_88275];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_86994 = ((double *) mem_89816)[i_88289 * (int64_t) 256 + i_86991 * (int64_t) 16 + i_88282];
                        
                        // futhark/microgpt.fut:294:75-135
                        
                        double zt_res_86995 = zt_lhs_86993 * zt_rhs_86994;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_86996 = r_86992 + zt_res_86995;
                        double r_tmp_90700 = zp_res_86996;
                        
                        r_86992 = r_tmp_90700;
                    }
                    defunc_0_lifted_lambda_res_86990 = r_86992;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_87003;
                    double r_87005 = 0.0;
                    
                    for (int64_t i_87004 = 0; i_87004 < (int64_t) 16; i_87004++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_87006 = ((double *) mem_89816)[i_88289 * (int64_t) 256 + i_88282 * (int64_t) 16 + i_87004];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_87007 = ((double *) mem_88985)[i_88289 * (int64_t) 64 + i_87004 * (int64_t) 4 + i_88275];
                        
                        // futhark/microgpt.fut:295:75-135
                        
                        double zt_res_87008 = zt_lhs_87006 * zt_rhs_87007;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_87009 = r_87005 + zt_res_87008;
                        double r_tmp_90701 = zp_res_87009;
                        
                        r_87005 = r_tmp_90701;
                    }
                    defunc_0_lifted_lambda_res_87003 = r_87005;
                    ((double *) mem_89865)[i_88275] = defunc_0_lifted_lambda_res_87003;
                    ((double *) mem_89866)[i_88275] = defunc_0_lifted_lambda_res_86990;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_89855, i_88282 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89865, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_89856, i_88282 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89866, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_89843, i_88289 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_89855, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_89844, i_88289 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_89856, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88308 = 0; i_88308 < (int64_t) 16; i_88308++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88298 = 0; i_88298 < (int64_t) 16; i_88298++) {
                // futhark/microgpt.fut:296:57-60
                
                int64_t tmp_87072 = sdiv64(i_88298, (int64_t) 4);
                
                // futhark/microgpt.fut:296:44-62
                
                bool x_87073 = sle64((int64_t) 0, tmp_87072);
                
                // futhark/microgpt.fut:296:44-62
                
                bool y_87074 = slt64(tmp_87072, (int64_t) 4);
                
                // futhark/microgpt.fut:296:44-62
                
                bool bounds_check_87075 = x_87073 && y_87074;
                
                // futhark/microgpt.fut:296:44-62
                
                bool index_certs_87076;
                
                if (!bounds_check_87075) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_87072, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:296:44-62\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:296:13-85\n   #6  futhark/microgpt.fut:440:5-76\n   #7  futhark/microgpt.fut:445:26-451:31\n   #8  futhark/microgpt.fut:467:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:296:79-82
                
                int64_t tmp_87077 = smod64(i_88298, (int64_t) 4);
                
                // futhark/microgpt.fut:296:44-84
                
                bool x_87078 = sle64((int64_t) 0, tmp_87077);
                
                // futhark/microgpt.fut:296:44-84
                
                bool y_87079 = slt64(tmp_87077, (int64_t) 4);
                
                // futhark/microgpt.fut:296:44-84
                
                bool bounds_check_87080 = x_87078 && y_87079;
                
                // futhark/microgpt.fut:296:44-84
                
                bool index_certs_87081;
                
                if (!bounds_check_87080) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_87077, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:296:44-84\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:296:13-85\n   #6  futhark/microgpt.fut:440:5-76\n   #7  futhark/microgpt.fut:445:26-451:31\n   #8  futhark/microgpt.fut:467:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_87082 = ((double *) mem_89719)[tmp_87072 * (int64_t) 64 + i_88308 * (int64_t) 4 + tmp_87077];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_87095 = ((double *) mem_89844)[tmp_87072 * (int64_t) 64 + i_88308 * (int64_t) 4 + tmp_87077];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_87111 = ((double *) mem_89843)[tmp_87072 * (int64_t) 64 + i_88308 * (int64_t) 4 + tmp_87077];
                
                ((double *) mem_89912)[i_88298] = lifted_lambda_res_87111;
                ((double *) mem_89913)[i_88298] = lifted_lambda_res_87095;
                ((double *) mem_89914)[i_88298] = lifted_lambda_res_87082;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89897, i_88308 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89912, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89898, i_88308 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89913, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89899, i_88308 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89914, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88333 = 0; i_88333 < (int64_t) 16; i_88333++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88320 = 0; i_88320 < (int64_t) 16; i_88320++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_87274;
                double r_87276 = 0.0;
                
                for (int64_t i_87275 = 0; i_87275 < (int64_t) 16; i_87275++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_87277 = ((double *) mem_param_88710.mem)[i_87275 * (int64_t) 16 + i_88320];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_87278 = ((double *) mem_89899)[i_88333 * (int64_t) 16 + i_87275];
                    
                    // futhark/microgpt.fut:299:69-114
                    
                    double zt_res_87279 = zt_lhs_87277 * zt_rhs_87278;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_87280 = r_87276 + zt_res_87279;
                    double r_tmp_90716 = zp_res_87280;
                    
                    r_87276 = r_tmp_90716;
                }
                defunc_0_lifted_lambda_res_87274 = r_87276;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_87281;
                double r_87283 = 0.0;
                
                for (int64_t i_87282 = 0; i_87282 < (int64_t) 16; i_87282++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_87284 = ((double *) mem_param_88686.mem)[i_87282 * (int64_t) 16 + i_88320];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_87285 = ((double *) mem_89898)[i_88333 * (int64_t) 16 + i_87282];
                    
                    // futhark/microgpt.fut:299:145-190
                    
                    double zt_res_87286 = zt_lhs_87284 * zt_rhs_87285;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_87287 = r_87283 + zt_res_87286;
                    double r_tmp_90717 = zp_res_87287;
                    
                    r_87283 = r_tmp_90717;
                }
                defunc_0_lifted_lambda_res_87281 = r_87283;
                // futhark/microgpt.fut:299:47-192
                
                double zp_res_87288 = defunc_0_lifted_lambda_res_87274 + defunc_0_lifted_lambda_res_87281;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_87289;
                double r_87291 = 0.0;
                
                for (int64_t i_87290 = 0; i_87290 < (int64_t) 16; i_87290++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_87292 = ((double *) mem_param_88698.mem)[i_87290 * (int64_t) 16 + i_88320];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_87293 = ((double *) mem_89897)[i_88333 * (int64_t) 16 + i_87290];
                    
                    // futhark/microgpt.fut:299:222-267
                    
                    double zt_res_87294 = zt_lhs_87292 * zt_rhs_87293;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_87295 = r_87291 + zt_res_87294;
                    double r_tmp_90718 = zp_res_87295;
                    
                    r_87291 = r_tmp_90718;
                }
                defunc_0_lifted_lambda_res_87289 = r_87291;
                // futhark/microgpt.fut:299:118-269
                
                double zp_res_87296 = zp_res_87288 + defunc_0_lifted_lambda_res_87289;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_87303;
                double r_87305 = 0.0;
                
                for (int64_t i_87304 = 0; i_87304 < (int64_t) 16; i_87304++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_87306 = ((double *) mem_89897)[i_87304 * (int64_t) 16 + i_88333];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_87307 = ((double *) mem_88885)[i_87304 * (int64_t) 16 + i_88320];
                    
                    // futhark/microgpt.fut:317:68-111
                    
                    double zt_res_87308 = zt_lhs_87306 * zt_rhs_87307;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_87309 = r_87305 + zt_res_87308;
                    double r_tmp_90719 = zp_res_87309;
                    
                    r_87305 = r_tmp_90719;
                }
                defunc_0_lifted_lambda_res_87303 = r_87305;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_87319;
                double r_87321 = 0.0;
                
                for (int64_t i_87320 = 0; i_87320 < (int64_t) 16; i_87320++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_87322 = ((double *) mem_89898)[i_87320 * (int64_t) 16 + i_88333];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_87323 = ((double *) mem_88885)[i_87320 * (int64_t) 16 + i_88320];
                    
                    // futhark/microgpt.fut:318:68-111
                    
                    double zt_res_87324 = zt_lhs_87322 * zt_rhs_87323;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_87325 = r_87321 + zt_res_87324;
                    double r_tmp_90720 = zp_res_87325;
                    
                    r_87321 = r_tmp_90720;
                }
                defunc_0_lifted_lambda_res_87319 = r_87321;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_87337;
                double r_87339 = 0.0;
                
                for (int64_t i_87338 = 0; i_87338 < (int64_t) 16; i_87338++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_87340 = ((double *) mem_89899)[i_87338 * (int64_t) 16 + i_88333];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_87341 = ((double *) mem_88885)[i_87338 * (int64_t) 16 + i_88320];
                    
                    // futhark/microgpt.fut:319:68-111
                    
                    double zt_res_87342 = zt_lhs_87340 * zt_rhs_87341;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_87343 = r_87339 + zt_res_87342;
                    double r_tmp_90721 = zp_res_87343;
                    
                    r_87339 = r_tmp_90721;
                }
                defunc_0_lifted_lambda_res_87337 = r_87339;
                ((double *) mem_89965)[i_88320] = defunc_0_lifted_lambda_res_87337;
                ((double *) mem_89966)[i_88320] = defunc_0_lifted_lambda_res_87319;
                ((double *) mem_89967)[i_88320] = defunc_0_lifted_lambda_res_87303;
                ((double *) mem_89968)[i_88320] = zp_res_87296;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89945, i_88333 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89965, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89946, i_88333 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89966, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89947, i_88333 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89967, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89948, i_88333 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89968, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88340 = 0; i_88340 < (int64_t) 16; i_88340++) {
            // futhark/microgpt.fut:303:51-63
            
            double zs_rhs_83417 = ((double *) mem_89181)[i_88340];
            
            // futhark/microgpt.fut:303:43-63
            
            double zs_res_83418 = 1.0 / zs_rhs_83417;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_83419;
            double r_83421 = 0.0;
            
            for (int64_t i_83420 = 0; i_83420 < (int64_t) 16; i_83420++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_83422 = ((double *) mem_88852)[i_88340 * (int64_t) 16 + i_83420];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_83423 = ((double *) mem_89948)[i_88340 * (int64_t) 16 + i_83420];
                
                // futhark/microgpt.fut:303:93-136
                
                double zt_res_83424 = zt_lhs_83422 * zt_rhs_83423;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_83425 = r_83421 + zt_res_83424;
                double r_tmp_90723 = zp_res_83425;
                
                r_83421 = r_tmp_90723;
            }
            defunc_0_lifted_lambda_res_83419 = r_83421;
            // futhark/microgpt.fut:303:71-165
            
            double zt_res_83426 = zs_res_83418 * defunc_0_lifted_lambda_res_83419;
            
            // futhark/microgpt.fut:303:47-165
            
            double zt_res_83427 = zs_res_83418 * zt_res_83426;
            
            // futhark/microgpt.fut:303:35-165
            
            double neg_res_83428 = -zt_res_83427;
            
            ((double *) mem_90009)[i_88340] = neg_res_83428;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88344 = 0; i_88344 < (int64_t) 16; i_88344++) {
            // futhark/microgpt.fut:304:35-47
            
            double zt_lhs_83436 = ((double *) mem_90009)[i_88344];
            
            // futhark/microgpt.fut:304:89-101
            
            double zp_lhs_83437 = ((double *) mem_88923)[i_88344];
            
            // futhark/microgpt.fut:304:89-129
            
            double zp_res_83438 = 1.0e-5 + zp_lhs_83437;
            
            // futhark/microgpt.fut:304:81-129
            
            double sqrt_res_83439 = futrts_sqrt64(zp_res_83438);
            
            // futhark/microgpt.fut:304:67-131
            
            double zt_res_83440 = 2.0 * sqrt_res_83439;
            
            // futhark/microgpt.fut:304:53-131
            
            double zs_res_83441 = 1.0 / zt_res_83440;
            
            // futhark/microgpt.fut:304:35-131
            
            double zt_res_83442 = zt_lhs_83436 * zs_res_83441;
            
            ((double *) mem_90016)[i_88344] = zt_res_83442;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88348 = 0; i_88348 < (int64_t) 16; i_88348++) {
            // futhark/microgpt.fut:305:45-57
            
            double zs_lhs_83450 = ((double *) mem_90016)[i_88348];
            
            // futhark/microgpt.fut:305:45-72
            
            double zs_res_83451 = zs_lhs_83450 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_90726 = 0; nest_i_90726 < (int64_t) 16; nest_i_90726++) {
                ((double *) mem_90023)[i_88348 * (int64_t) 16 + nest_i_90726] = zs_res_83451;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88356 = 0; i_88356 < (int64_t) 16; i_88356++) {
            // futhark/microgpt.fut:306:107-119
            
            double zs_rhs_83460 = ((double *) mem_89181)[i_88356];
            
            // futhark/microgpt.fut:306:99-119
            
            double zs_res_83461 = 1.0 / zs_rhs_83460;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88352 = 0; i_88352 < (int64_t) 16; i_88352++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_83468 = ((double *) mem_89549)[i_88356 * (int64_t) 16 + i_88352];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_83469 = ((double *) mem_89948)[i_88356 * (int64_t) 16 + i_88352];
                
                // futhark/microgpt.fut:306:73-119
                
                double zt_res_83470 = zs_res_83461 * zt_lhs_83469;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_83471 = ((double *) mem_88852)[i_88356 * (int64_t) 16 + i_88352];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_83472 = ((double *) mem_90023)[i_88356 * (int64_t) 16 + i_88352];
                
                // futhark/microgpt.fut:306:127-170
                
                double zt_res_83473 = zt_lhs_83471 * zt_rhs_83472;
                
                // futhark/microgpt.fut:306:94-170
                
                double zp_res_83474 = zt_res_83470 + zt_res_83473;
                
                // futhark/microgpt.fut:306:122-221
                
                double zp_res_83475 = zt_res_83473 + zp_res_83474;
                
                // futhark/microgpt.fut:306:45-221
                
                double zp_res_83476 = zp_lhs_83468 + zp_res_83475;
                
                ((double *) mem_90038)[i_88352] = zp_res_83476;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_90033, i_88356 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_90038, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88360 = 0; i_88360 < (int64_t) 16; i_88360++) {
            // futhark/microgpt.fut:310:51-63
            
            double zs_rhs_83524 = ((double *) mem_88922)[i_88360];
            
            // futhark/microgpt.fut:310:43-63
            
            double zs_res_83525 = 1.0 / zs_rhs_83524;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_83526;
            double r_83528 = 0.0;
            
            for (int64_t i_83527 = 0; i_83527 < (int64_t) 16; i_83527++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_83529 = ((double *) mem_88820)[i_88360 * (int64_t) 16 + i_83527];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_83530 = ((double *) mem_90033)[i_88360 * (int64_t) 16 + i_83527];
                
                // futhark/microgpt.fut:310:93-136
                
                double zt_res_83531 = zt_lhs_83529 * zt_rhs_83530;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_83532 = r_83528 + zt_res_83531;
                double r_tmp_90730 = zp_res_83532;
                
                r_83528 = r_tmp_90730;
            }
            defunc_0_lifted_lambda_res_83526 = r_83528;
            // futhark/microgpt.fut:310:71-165
            
            double zt_res_83533 = zs_res_83525 * defunc_0_lifted_lambda_res_83526;
            
            // futhark/microgpt.fut:310:47-165
            
            double zt_res_83534 = zs_res_83525 * zt_res_83533;
            
            // futhark/microgpt.fut:310:35-165
            
            double neg_res_83535 = -zt_res_83534;
            
            ((double *) mem_90049)[i_88360] = neg_res_83535;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88364 = 0; i_88364 < (int64_t) 16; i_88364++) {
            // futhark/microgpt.fut:311:35-47
            
            double zt_lhs_83543 = ((double *) mem_90049)[i_88364];
            
            // futhark/microgpt.fut:311:89-101
            
            double zp_lhs_83544 = ((double *) mem_88883)[i_88364];
            
            // futhark/microgpt.fut:311:89-129
            
            double zp_res_83545 = 1.0e-5 + zp_lhs_83544;
            
            // futhark/microgpt.fut:311:81-129
            
            double sqrt_res_83546 = futrts_sqrt64(zp_res_83545);
            
            // futhark/microgpt.fut:311:67-131
            
            double zt_res_83547 = 2.0 * sqrt_res_83546;
            
            // futhark/microgpt.fut:311:53-131
            
            double zs_res_83548 = 1.0 / zt_res_83547;
            
            // futhark/microgpt.fut:311:35-131
            
            double zt_res_83549 = zt_lhs_83543 * zs_res_83548;
            
            ((double *) mem_90056)[i_88364] = zt_res_83549;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88368 = 0; i_88368 < (int64_t) 16; i_88368++) {
            // futhark/microgpt.fut:312:45-57
            
            double zs_lhs_83557 = ((double *) mem_90056)[i_88368];
            
            // futhark/microgpt.fut:312:45-72
            
            double zs_res_83558 = zs_lhs_83557 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_90733 = 0; nest_i_90733 < (int64_t) 16; nest_i_90733++) {
                ((double *) mem_90063)[i_88368 * (int64_t) 16 + nest_i_90733] = zs_res_83558;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88376 = 0; i_88376 < (int64_t) 16; i_88376++) {
            // futhark/microgpt.fut:313:81-93
            
            double zs_rhs_83567 = ((double *) mem_88922)[i_88376];
            
            // futhark/microgpt.fut:313:73-93
            
            double zs_res_83568 = 1.0 / zs_rhs_83567;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88372 = 0; i_88372 < (int64_t) 16; i_88372++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_83575 = ((double *) mem_90033)[i_88376 * (int64_t) 16 + i_88372];
                
                // futhark/microgpt.fut:313:47-93
                
                double zt_res_83576 = zs_res_83568 * zt_lhs_83575;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_83577 = ((double *) mem_88820)[i_88376 * (int64_t) 16 + i_88372];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_83578 = ((double *) mem_90063)[i_88376 * (int64_t) 16 + i_88372];
                
                // futhark/microgpt.fut:313:101-144
                
                double zt_res_83579 = zt_lhs_83577 * zt_rhs_83578;
                
                // futhark/microgpt.fut:313:68-144
                
                double zp_res_83580 = zt_res_83576 + zt_res_83579;
                
                // futhark/microgpt.fut:313:96-195
                
                double zp_res_83581 = zt_res_83579 + zp_res_83580;
                
                ((double *) mem_90078)[i_88372] = zp_res_83581;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_90073, i_88376 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_90078, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88389 = 0; i_88389 < (int64_t) 16; i_88389++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88382 = 0; i_88382 < (int64_t) 16; i_88382++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_87369 = ((double *) mem_90073)[i_88389 * (int64_t) 16 + i_88382];
                
                ((double *) mem_90099)[i_88382] = lifted_lambda_res_87369;
                ((double *) mem_90100)[i_88382] = lifted_lambda_res_87369;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_90089, i_88389 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_90099, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_90090, i_88389 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_90100, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88398 = 0; i_88398 < (int64_t) 64; i_88398++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88394 = 0; i_88394 < (int64_t) 16; i_88394++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_83695;
                double r_83697 = 0.0;
                
                for (int64_t i_83696 = 0; i_83696 < (int64_t) 16; i_83696++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_83698 = ((double *) mem_89493)[i_83696 * (int64_t) 64 + i_88398];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_83699 = ((double *) mem_89237)[i_83696 * (int64_t) 16 + i_88394];
                    
                    // futhark/microgpt.fut:321:67-110
                    
                    double zt_res_83700 = zt_lhs_83698 * zt_rhs_83699;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_83701 = r_83697 + zt_res_83700;
                    double r_tmp_90742 = zp_res_83701;
                    
                    r_83697 = r_tmp_90742;
                }
                defunc_0_lifted_lambda_res_83695 = r_83697;
                ((double *) mem_90126)[i_88394] = defunc_0_lifted_lambda_res_83695;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_90121, i_88398 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_90126, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_88411 = 0; i_88411 < (int64_t) 27; i_88411++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_88404 = 0; i_88404 < (int64_t) 16; i_88404++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_87397;
                double r_87399 = 0.0;
                
                for (int64_t i_87398 = 0; i_87398 < (int64_t) 16; i_87398++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_87400 = ((double *) mem_89429)[i_87398 * (int64_t) 27 + i_88411];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_87401 = ((double *) mem_89330)[i_87398 * (int64_t) 16 + i_88404];
                    
                    // futhark/microgpt.fut:323:68-111
                    
                    double zt_res_87402 = zt_lhs_87400 * zt_rhs_87401;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_87403 = r_87399 + zt_res_87402;
                    double r_tmp_90747 = zp_res_87403;
                    
                    r_87399 = r_tmp_90747;
                }
                defunc_0_lifted_lambda_res_87397 = r_87399;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_87406;
                double r_87408 = 0.0;
                
                for (int64_t i_87407 = 0; i_87407 < (int64_t) 16; i_87407++) {
                    int64_t zeze_lhs_87409 = ((int64_t *) seqs_mem_88678.mem)[step_81997 * (int64_t) 16 + i_87407];
                    
                    // futhark/microgpt.fut:441:58-109
                    
                    bool cond_87410 = zeze_lhs_87409 == i_88411;
                    
                    // futhark/microgpt.fut:441:58-109
                    
                    double lifted_lambda_res_87411;
                    
                    if (cond_87410) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double lifted_lambda_res_t_res_87709 = ((double *) mem_90089)[i_87407 * (int64_t) 16 + i_88404];
                        
                        lifted_lambda_res_87411 = lifted_lambda_res_t_res_87709;
                    } else {
                        lifted_lambda_res_87411 = 0.0;
                    }
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_87417 = r_87408 + lifted_lambda_res_87411;
                    double r_tmp_90748 = zp_res_87417;
                    
                    r_87408 = r_tmp_90748;
                }
                defunc_0_lifted_lambda_res_87406 = r_87408;
                ((double *) mem_90147)[i_88404] = defunc_0_lifted_lambda_res_87406;
                ((double *) mem_90148)[i_88404] = defunc_0_lifted_lambda_res_87397;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_90137, i_88411 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_90147, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_90138, i_88411 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_90148, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:66:26-45
        
        double i64_res_83779 = sitofp_i64_f64(step_81997);
        
        // futhark/microgpt.fut:397:46-65
        
        double zm_rhs_83780 = i64_res_83779 / 500.0;
        
        // futhark/microgpt.fut:397:24-65
        
        double zt_rhs_83781 = 1.0 - zm_rhs_83780;
        
        // futhark/microgpt.fut:397:19-65
        
        double lt_r_83782 = 1.0e-2 * zt_rhs_83781;
        
        // futhark/microgpt.fut:399:5-52
        if (memblock_alloc(ctx, &mem_90169, (int64_t) 3456, "mem_90169")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:399:5-52
        // futhark/microgpt.fut:399:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90169.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88702.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:399:5-52
        if (memblock_alloc(ctx, &mem_90171, (int64_t) 3456, "mem_90171")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:399:5-52
        // futhark/microgpt.fut:399:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90171.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88738.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:399:5-52
        if (memblock_alloc(ctx, &mem_90173, (int64_t) 3456, "mem_90173")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:399:5-52
        // futhark/microgpt.fut:399:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90173.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88774.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:399:5-52
        if (memblock_alloc(ctx, &mem_90175, (int64_t) 3456, "mem_90175")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:399:5-52
        // futhark/microgpt.fut:399:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90175.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_90137, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:399:5-52
        if (futrts_adam_opt_w_10080(ctx, &ext_mem_90179, &ext_mem_90178, &ext_mem_90177, mem_90169, mem_90171, mem_90173, mem_90175, (int64_t) 27, (int64_t) 16, step_81997, lt_r_83782) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_90169, "mem_90169") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90171, "mem_90171") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90173, "mem_90173") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90175, "mem_90175") != 0)
            return 1;
        // futhark/microgpt.fut:401:5-52
        if (memblock_alloc(ctx, &mem_90180, (int64_t) 2048, "mem_90180")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:401:5-52
        // futhark/microgpt.fut:401:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90180.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88694.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:401:5-52
        if (memblock_alloc(ctx, &mem_90182, (int64_t) 2048, "mem_90182")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:401:5-52
        // futhark/microgpt.fut:401:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90182.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88730.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:401:5-52
        if (memblock_alloc(ctx, &mem_90184, (int64_t) 2048, "mem_90184")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:401:5-52
        // futhark/microgpt.fut:401:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90184.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88766.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:401:5-52
        if (memblock_alloc(ctx, &mem_90186, (int64_t) 2048, "mem_90186")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:401:5-52
        // futhark/microgpt.fut:401:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90186.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_90090, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:401:5-52
        if (futrts_adam_opt_w_10081(ctx, &ext_mem_90190, &ext_mem_90189, &ext_mem_90188, mem_90180, mem_90182, mem_90184, mem_90186, (int64_t) 16, (int64_t) 16, step_81997, lt_r_83782) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_90180, "mem_90180") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90182, "mem_90182") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90184, "mem_90184") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90186, "mem_90186") != 0)
            return 1;
        // futhark/microgpt.fut:403:5-56
        if (memblock_alloc(ctx, &mem_90191, (int64_t) 2048, "mem_90191")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:403:5-56
        // futhark/microgpt.fut:403:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90191.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88698.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:403:5-56
        if (memblock_alloc(ctx, &mem_90193, (int64_t) 2048, "mem_90193")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:403:5-56
        // futhark/microgpt.fut:403:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90193.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88734.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:403:5-56
        if (memblock_alloc(ctx, &mem_90195, (int64_t) 2048, "mem_90195")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:403:5-56
        // futhark/microgpt.fut:403:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90195.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88770.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:403:5-56
        if (memblock_alloc(ctx, &mem_90197, (int64_t) 2048, "mem_90197")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:403:5-56
        // futhark/microgpt.fut:403:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90197.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_89947, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:403:5-56
        if (futrts_adam_opt_w_10081(ctx, &ext_mem_90201, &ext_mem_90200, &ext_mem_90199, mem_90191, mem_90193, mem_90195, mem_90197, (int64_t) 16, (int64_t) 16, step_81997, lt_r_83782) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_90191, "mem_90191") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90193, "mem_90193") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90195, "mem_90195") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90197, "mem_90197") != 0)
            return 1;
        // futhark/microgpt.fut:405:5-56
        if (memblock_alloc(ctx, &mem_90202, (int64_t) 2048, "mem_90202")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:405:5-56
        // futhark/microgpt.fut:405:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90202.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88686.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:405:5-56
        if (memblock_alloc(ctx, &mem_90204, (int64_t) 2048, "mem_90204")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:405:5-56
        // futhark/microgpt.fut:405:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90204.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88722.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:405:5-56
        if (memblock_alloc(ctx, &mem_90206, (int64_t) 2048, "mem_90206")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:405:5-56
        // futhark/microgpt.fut:405:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90206.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88758.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:405:5-56
        if (memblock_alloc(ctx, &mem_90208, (int64_t) 2048, "mem_90208")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:405:5-56
        // futhark/microgpt.fut:405:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90208.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_89946, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:405:5-56
        if (futrts_adam_opt_w_10081(ctx, &ext_mem_90212, &ext_mem_90211, &ext_mem_90210, mem_90202, mem_90204, mem_90206, mem_90208, (int64_t) 16, (int64_t) 16, step_81997, lt_r_83782) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_90202, "mem_90202") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90204, "mem_90204") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90206, "mem_90206") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90208, "mem_90208") != 0)
            return 1;
        // futhark/microgpt.fut:407:5-56
        if (memblock_alloc(ctx, &mem_90213, (int64_t) 2048, "mem_90213")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:407:5-56
        // futhark/microgpt.fut:407:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90213.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88710.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:407:5-56
        if (memblock_alloc(ctx, &mem_90215, (int64_t) 2048, "mem_90215")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:407:5-56
        // futhark/microgpt.fut:407:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90215.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88746.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:407:5-56
        if (memblock_alloc(ctx, &mem_90217, (int64_t) 2048, "mem_90217")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:407:5-56
        // futhark/microgpt.fut:407:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90217.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88782.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:407:5-56
        if (memblock_alloc(ctx, &mem_90219, (int64_t) 2048, "mem_90219")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:407:5-56
        // futhark/microgpt.fut:407:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90219.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_89945, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:407:5-56
        if (futrts_adam_opt_w_10081(ctx, &ext_mem_90223, &ext_mem_90222, &ext_mem_90221, mem_90213, mem_90215, mem_90217, mem_90219, (int64_t) 16, (int64_t) 16, step_81997, lt_r_83782) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_90213, "mem_90213") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90215, "mem_90215") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90217, "mem_90217") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90219, "mem_90219") != 0)
            return 1;
        // futhark/microgpt.fut:409:5-56
        if (memblock_alloc(ctx, &mem_90224, (int64_t) 2048, "mem_90224")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:409:5-56
        // futhark/microgpt.fut:409:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90224.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88690.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:409:5-56
        if (memblock_alloc(ctx, &mem_90226, (int64_t) 2048, "mem_90226")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:409:5-56
        // futhark/microgpt.fut:409:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90226.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88726.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:409:5-56
        if (memblock_alloc(ctx, &mem_90228, (int64_t) 2048, "mem_90228")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:409:5-56
        // futhark/microgpt.fut:409:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90228.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88762.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:409:5-56
        if (memblock_alloc(ctx, &mem_90230, (int64_t) 2048, "mem_90230")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:409:5-56
        // futhark/microgpt.fut:409:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90230.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_89565, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:409:5-56
        if (futrts_adam_opt_w_10081(ctx, &ext_mem_90234, &ext_mem_90233, &ext_mem_90232, mem_90224, mem_90226, mem_90228, mem_90230, (int64_t) 16, (int64_t) 16, step_81997, lt_r_83782) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_90224, "mem_90224") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90226, "mem_90226") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90228, "mem_90228") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90230, "mem_90230") != 0)
            return 1;
        // futhark/microgpt.fut:411:5-52
        if (memblock_alloc(ctx, &mem_90235, (int64_t) 8192, "mem_90235")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:411:5-52
        // futhark/microgpt.fut:411:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90235.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88706.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:411:5-52
        if (memblock_alloc(ctx, &mem_90237, (int64_t) 8192, "mem_90237")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:411:5-52
        // futhark/microgpt.fut:411:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90237.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88742.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:411:5-52
        if (memblock_alloc(ctx, &mem_90239, (int64_t) 8192, "mem_90239")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:411:5-52
        // futhark/microgpt.fut:411:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90239.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88778.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:411:5-52
        if (memblock_alloc(ctx, &mem_90241, (int64_t) 8192, "mem_90241")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:411:5-52
        // futhark/microgpt.fut:411:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90241.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_90121, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:411:5-52
        if (futrts_adam_opt_w_10080(ctx, &ext_mem_90245, &ext_mem_90244, &ext_mem_90243, mem_90235, mem_90237, mem_90239, mem_90241, (int64_t) 64, (int64_t) 16, step_81997, lt_r_83782) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_90235, "mem_90235") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90237, "mem_90237") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90239, "mem_90239") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90241, "mem_90241") != 0)
            return 1;
        // futhark/microgpt.fut:413:5-60
        if (memblock_alloc(ctx, &mem_90246, (int64_t) 8192, "mem_90246")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:413:5-60
        // futhark/microgpt.fut:413:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90246.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_88682.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:413:5-60
        if (memblock_alloc(ctx, &mem_90248, (int64_t) 8192, "mem_90248")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:413:5-60
        // futhark/microgpt.fut:413:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90248.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_88718.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:413:5-60
        if (memblock_alloc(ctx, &mem_90250, (int64_t) 8192, "mem_90250")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:413:5-60
        // futhark/microgpt.fut:413:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90250.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_88754.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:413:5-60
        if (memblock_alloc(ctx, &mem_90252, (int64_t) 8192, "mem_90252")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:413:5-60
        // futhark/microgpt.fut:413:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90252.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_89461, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:413:5-60
        if (futrts_adam_opt_w_10080(ctx, &ext_mem_90256, &ext_mem_90255, &ext_mem_90254, mem_90246, mem_90248, mem_90250, mem_90252, (int64_t) 16, (int64_t) 64, step_81997, lt_r_83782) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_90246, "mem_90246") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90248, "mem_90248") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90250, "mem_90250") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90252, "mem_90252") != 0)
            return 1;
        // futhark/microgpt.fut:415:5-56
        if (memblock_alloc(ctx, &mem_90257, (int64_t) 3456, "mem_90257")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:415:5-56
        // futhark/microgpt.fut:415:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90257.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88714.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:415:5-56
        if (memblock_alloc(ctx, &mem_90259, (int64_t) 3456, "mem_90259")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:415:5-56
        // futhark/microgpt.fut:415:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90259.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88750.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:415:5-56
        if (memblock_alloc(ctx, &mem_90261, (int64_t) 3456, "mem_90261")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:415:5-56
        // futhark/microgpt.fut:415:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90261.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_88786.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:415:5-56
        if (memblock_alloc(ctx, &mem_90263, (int64_t) 3456, "mem_90263")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:415:5-56
        // futhark/microgpt.fut:415:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_90263.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_90138, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:415:5-56
        if (futrts_adam_opt_w_10080(ctx, &ext_mem_90267, &ext_mem_90266, &ext_mem_90265, mem_90257, mem_90259, mem_90261, mem_90263, (int64_t) 27, (int64_t) 16, step_81997, lt_r_83782) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_90257, "mem_90257") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90259, "mem_90259") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90261, "mem_90261") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90263, "mem_90263") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90475, &ext_mem_90256, "ext_mem_90256") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90476, &ext_mem_90212, "ext_mem_90212") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90477, &ext_mem_90234, "ext_mem_90234") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90478, &ext_mem_90190, "ext_mem_90190") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90479, &ext_mem_90201, "ext_mem_90201") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90480, &ext_mem_90179, "ext_mem_90179") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90481, &ext_mem_90245, "ext_mem_90245") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90482, &ext_mem_90223, "ext_mem_90223") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90483, &ext_mem_90267, "ext_mem_90267") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90484, &ext_mem_90255, "ext_mem_90255") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90485, &ext_mem_90211, "ext_mem_90211") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90486, &ext_mem_90233, "ext_mem_90233") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90487, &ext_mem_90189, "ext_mem_90189") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90488, &ext_mem_90200, "ext_mem_90200") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90489, &ext_mem_90178, "ext_mem_90178") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90490, &ext_mem_90244, "ext_mem_90244") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90491, &ext_mem_90222, "ext_mem_90222") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90492, &ext_mem_90266, "ext_mem_90266") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90493, &ext_mem_90254, "ext_mem_90254") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90494, &ext_mem_90210, "ext_mem_90210") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90495, &ext_mem_90232, "ext_mem_90232") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90496, &ext_mem_90188, "ext_mem_90188") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90497, &ext_mem_90199, "ext_mem_90199") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90498, &ext_mem_90177, "ext_mem_90177") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90499, &ext_mem_90243, "ext_mem_90243") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90500, &ext_mem_90221, "ext_mem_90221") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_90501, &ext_mem_90265, "ext_mem_90265") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88682, &mem_param_tmp_90475, "mem_param_tmp_90475") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88686, &mem_param_tmp_90476, "mem_param_tmp_90476") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88690, &mem_param_tmp_90477, "mem_param_tmp_90477") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88694, &mem_param_tmp_90478, "mem_param_tmp_90478") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88698, &mem_param_tmp_90479, "mem_param_tmp_90479") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88702, &mem_param_tmp_90480, "mem_param_tmp_90480") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88706, &mem_param_tmp_90481, "mem_param_tmp_90481") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88710, &mem_param_tmp_90482, "mem_param_tmp_90482") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88714, &mem_param_tmp_90483, "mem_param_tmp_90483") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88718, &mem_param_tmp_90484, "mem_param_tmp_90484") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88722, &mem_param_tmp_90485, "mem_param_tmp_90485") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88726, &mem_param_tmp_90486, "mem_param_tmp_90486") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88730, &mem_param_tmp_90487, "mem_param_tmp_90487") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88734, &mem_param_tmp_90488, "mem_param_tmp_90488") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88738, &mem_param_tmp_90489, "mem_param_tmp_90489") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88742, &mem_param_tmp_90490, "mem_param_tmp_90490") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88746, &mem_param_tmp_90491, "mem_param_tmp_90491") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88750, &mem_param_tmp_90492, "mem_param_tmp_90492") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88754, &mem_param_tmp_90493, "mem_param_tmp_90493") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88758, &mem_param_tmp_90494, "mem_param_tmp_90494") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88762, &mem_param_tmp_90495, "mem_param_tmp_90495") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88766, &mem_param_tmp_90496, "mem_param_tmp_90496") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88770, &mem_param_tmp_90497, "mem_param_tmp_90497") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88774, &mem_param_tmp_90498, "mem_param_tmp_90498") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88778, &mem_param_tmp_90499, "mem_param_tmp_90499") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88782, &mem_param_tmp_90500, "mem_param_tmp_90500") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_88786, &mem_param_tmp_90501, "mem_param_tmp_90501") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_90375, &mem_param_88682, "mem_param_88682") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90374, &mem_param_88686, "mem_param_88686") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90373, &mem_param_88690, "mem_param_88690") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90372, &mem_param_88694, "mem_param_88694") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90371, &mem_param_88698, "mem_param_88698") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90370, &mem_param_88702, "mem_param_88702") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90369, &mem_param_88706, "mem_param_88706") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90368, &mem_param_88710, "mem_param_88710") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90367, &mem_param_88714, "mem_param_88714") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90366, &mem_param_88718, "mem_param_88718") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90365, &mem_param_88722, "mem_param_88722") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90364, &mem_param_88726, "mem_param_88726") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90363, &mem_param_88730, "mem_param_88730") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90362, &mem_param_88734, "mem_param_88734") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90361, &mem_param_88738, "mem_param_88738") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90360, &mem_param_88742, "mem_param_88742") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90359, &mem_param_88746, "mem_param_88746") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90358, &mem_param_88750, "mem_param_88750") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90357, &mem_param_88754, "mem_param_88754") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90356, &mem_param_88758, "mem_param_88758") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90355, &mem_param_88762, "mem_param_88762") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90354, &mem_param_88766, "mem_param_88766") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90353, &mem_param_88770, "mem_param_88770") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90352, &mem_param_88774, "mem_param_88774") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90351, &mem_param_88778, "mem_param_88778") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90350, &mem_param_88782, "mem_param_88782") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_90349, &mem_param_88786, "mem_param_88786") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90448, &ext_mem_90370, "ext_mem_90370") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90449, &ext_mem_90372, "ext_mem_90372") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90450, &ext_mem_90371, "ext_mem_90371") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90451, &ext_mem_90374, "ext_mem_90374") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90452, &ext_mem_90368, "ext_mem_90368") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90453, &ext_mem_90373, "ext_mem_90373") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90454, &ext_mem_90369, "ext_mem_90369") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90455, &ext_mem_90375, "ext_mem_90375") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90456, &ext_mem_90367, "ext_mem_90367") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90457, &ext_mem_90361, "ext_mem_90361") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90458, &ext_mem_90363, "ext_mem_90363") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90459, &ext_mem_90362, "ext_mem_90362") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90460, &ext_mem_90365, "ext_mem_90365") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90461, &ext_mem_90359, "ext_mem_90359") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90462, &ext_mem_90364, "ext_mem_90364") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90463, &ext_mem_90360, "ext_mem_90360") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90464, &ext_mem_90366, "ext_mem_90366") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90465, &ext_mem_90358, "ext_mem_90358") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90466, &ext_mem_90352, "ext_mem_90352") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90467, &ext_mem_90354, "ext_mem_90354") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90468, &ext_mem_90353, "ext_mem_90353") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90469, &ext_mem_90356, "ext_mem_90356") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90470, &ext_mem_90350, "ext_mem_90350") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90471, &ext_mem_90355, "ext_mem_90355") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90472, &ext_mem_90351, "ext_mem_90351") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90473, &ext_mem_90357, "ext_mem_90357") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90474, &ext_mem_90349, "ext_mem_90349") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90840, &mem_out_90448, "mem_out_90448") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90841, &mem_out_90449, "mem_out_90449") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90842, &mem_out_90450, "mem_out_90450") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90843, &mem_out_90451, "mem_out_90451") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90844, &mem_out_90452, "mem_out_90452") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90845, &mem_out_90453, "mem_out_90453") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90846, &mem_out_90454, "mem_out_90454") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90847, &mem_out_90455, "mem_out_90455") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90848, &mem_out_90456, "mem_out_90456") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90849, &mem_out_90457, "mem_out_90457") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90850, &mem_out_90458, "mem_out_90458") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90851, &mem_out_90459, "mem_out_90459") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90852, &mem_out_90460, "mem_out_90460") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90853, &mem_out_90461, "mem_out_90461") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90854, &mem_out_90462, "mem_out_90462") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90855, &mem_out_90463, "mem_out_90463") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90856, &mem_out_90464, "mem_out_90464") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90857, &mem_out_90465, "mem_out_90465") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90858, &mem_out_90466, "mem_out_90466") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90859, &mem_out_90467, "mem_out_90467") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90860, &mem_out_90468, "mem_out_90468") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90861, &mem_out_90469, "mem_out_90469") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90862, &mem_out_90470, "mem_out_90470") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90863, &mem_out_90471, "mem_out_90471") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90864, &mem_out_90472, "mem_out_90472") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90865, &mem_out_90473, "mem_out_90473") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90866, &mem_out_90474, "mem_out_90474") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_88787);
        free(mem_88788);
        free(mem_88797);
        free(mem_88804);
        free(mem_88819);
        free(mem_88820);
        free(mem_88829);
        free(mem_88836);
        free(mem_88851);
        free(mem_88852);
        free(mem_88861);
        free(mem_88862);
        free(mem_88883);
        free(mem_88884);
        free(mem_88885);
        free(mem_88897);
        free(mem_88898);
        free(mem_88922);
        free(mem_88923);
        free(mem_88924);
        free(mem_88925);
        free(mem_88926);
        free(mem_88945);
        free(mem_88946);
        free(mem_88947);
        free(mem_88984);
        free(mem_88985);
        free(mem_88986);
        free(mem_89002);
        free(mem_89003);
        free(mem_89004);
        free(mem_89017);
        free(mem_89018);
        free(mem_89019);
        free(mem_89065);
        free(mem_89066);
        free(mem_89077);
        free(mem_89078);
        free(mem_89087);
        free(mem_89088);
        free(mem_89109);
        free(mem_89114);
        free(mem_89125);
        free(mem_89130);
        free(mem_89137);
        free(mem_89144);
        free(mem_89155);
        free(mem_89160);
        free(mem_89181);
        free(mem_89182);
        free(mem_89190);
        free(mem_89204);
        free(mem_89209);
        free(mem_89220);
        free(mem_89225);
        free(mem_89236);
        free(mem_89237);
        free(mem_89246);
        free(mem_89247);
        free(mem_89268);
        free(mem_89269);
        free(mem_89277);
        free(mem_89291);
        free(mem_89292);
        free(mem_89300);
        free(mem_89314);
        free(mem_89319);
        free(mem_89330);
        free(mem_89335);
        free(mem_89346);
        free(mem_89351);
        free(mem_89362);
        free(mem_89363);
        free(mem_89372);
        free(mem_89373);
        free(mem_89386);
        free(mem_89387);
        free(mem_89400);
        free(mem_89401);
        free(mem_89422);
        free(mem_89429);
        free(mem_89434);
        free(mem_89445);
        free(mem_89450);
        free(mem_89461);
        free(mem_89462);
        free(mem_89471);
        free(mem_89472);
        free(mem_89493);
        free(mem_89498);
        free(mem_89509);
        free(mem_89514);
        free(mem_89525);
        free(mem_89532);
        free(mem_89539);
        free(mem_89549);
        free(mem_89554);
        free(mem_89565);
        free(mem_89566);
        free(mem_89575);
        free(mem_89576);
        free(mem_89597);
        free(mem_89598);
        free(mem_89609);
        free(mem_89610);
        free(mem_89619);
        free(mem_89626);
        free(mem_89651);
        free(mem_89652);
        free(mem_89663);
        free(mem_89664);
        free(mem_89673);
        free(mem_89680);
        free(mem_89687);
        free(mem_89694);
        free(mem_89719);
        free(mem_89720);
        free(mem_89731);
        free(mem_89732);
        free(mem_89741);
        free(mem_89748);
        free(mem_89773);
        free(mem_89778);
        free(mem_89789);
        free(mem_89795);
        free(mem_89800);
        free(mem_89816);
        free(mem_89822);
        free(mem_89827);
        free(mem_89843);
        free(mem_89844);
        free(mem_89855);
        free(mem_89856);
        free(mem_89865);
        free(mem_89866);
        free(mem_89897);
        free(mem_89898);
        free(mem_89899);
        free(mem_89912);
        free(mem_89913);
        free(mem_89914);
        free(mem_89945);
        free(mem_89946);
        free(mem_89947);
        free(mem_89948);
        free(mem_89965);
        free(mem_89966);
        free(mem_89967);
        free(mem_89968);
        free(mem_90009);
        free(mem_90016);
        free(mem_90023);
        free(mem_90033);
        free(mem_90038);
        free(mem_90049);
        free(mem_90056);
        free(mem_90063);
        free(mem_90073);
        free(mem_90078);
        free(mem_90089);
        free(mem_90090);
        free(mem_90099);
        free(mem_90100);
        free(mem_90121);
        free(mem_90126);
        free(mem_90137);
        free(mem_90138);
        free(mem_90147);
        free(mem_90148);
        if (memblock_unref(ctx, &mem_param_tmp_90501, "mem_param_tmp_90501") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90500, "mem_param_tmp_90500") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90499, "mem_param_tmp_90499") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90498, "mem_param_tmp_90498") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90497, "mem_param_tmp_90497") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90496, "mem_param_tmp_90496") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90495, "mem_param_tmp_90495") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90494, "mem_param_tmp_90494") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90493, "mem_param_tmp_90493") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90492, "mem_param_tmp_90492") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90491, "mem_param_tmp_90491") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90490, "mem_param_tmp_90490") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90489, "mem_param_tmp_90489") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90488, "mem_param_tmp_90488") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90487, "mem_param_tmp_90487") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90486, "mem_param_tmp_90486") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90485, "mem_param_tmp_90485") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90484, "mem_param_tmp_90484") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90483, "mem_param_tmp_90483") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90482, "mem_param_tmp_90482") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90481, "mem_param_tmp_90481") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90480, "mem_param_tmp_90480") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90479, "mem_param_tmp_90479") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90478, "mem_param_tmp_90478") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90477, "mem_param_tmp_90477") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90476, "mem_param_tmp_90476") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_90475, "mem_param_tmp_90475") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90265, "ext_mem_90265") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90266, "ext_mem_90266") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90267, "ext_mem_90267") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90263, "mem_90263") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90261, "mem_90261") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90259, "mem_90259") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90257, "mem_90257") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90254, "ext_mem_90254") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90255, "ext_mem_90255") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90256, "ext_mem_90256") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90252, "mem_90252") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90250, "mem_90250") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90248, "mem_90248") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90246, "mem_90246") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90243, "ext_mem_90243") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90244, "ext_mem_90244") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90245, "ext_mem_90245") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90241, "mem_90241") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90239, "mem_90239") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90237, "mem_90237") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90235, "mem_90235") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90232, "ext_mem_90232") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90233, "ext_mem_90233") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90234, "ext_mem_90234") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90230, "mem_90230") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90228, "mem_90228") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90226, "mem_90226") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90224, "mem_90224") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90221, "ext_mem_90221") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90222, "ext_mem_90222") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90223, "ext_mem_90223") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90219, "mem_90219") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90217, "mem_90217") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90215, "mem_90215") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90213, "mem_90213") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90210, "ext_mem_90210") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90211, "ext_mem_90211") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90212, "ext_mem_90212") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90208, "mem_90208") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90206, "mem_90206") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90204, "mem_90204") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90202, "mem_90202") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90199, "ext_mem_90199") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90200, "ext_mem_90200") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90201, "ext_mem_90201") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90197, "mem_90197") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90195, "mem_90195") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90193, "mem_90193") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90191, "mem_90191") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90188, "ext_mem_90188") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90189, "ext_mem_90189") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90190, "ext_mem_90190") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90186, "mem_90186") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90184, "mem_90184") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90182, "mem_90182") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90180, "mem_90180") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90177, "ext_mem_90177") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90178, "ext_mem_90178") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90179, "ext_mem_90179") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90175, "mem_90175") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90173, "mem_90173") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90171, "mem_90171") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_90169, "mem_90169") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88786, "mem_param_88786") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88782, "mem_param_88782") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88778, "mem_param_88778") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88774, "mem_param_88774") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88770, "mem_param_88770") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88766, "mem_param_88766") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88762, "mem_param_88762") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88758, "mem_param_88758") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88754, "mem_param_88754") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88750, "mem_param_88750") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88746, "mem_param_88746") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88742, "mem_param_88742") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88738, "mem_param_88738") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88734, "mem_param_88734") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88730, "mem_param_88730") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88726, "mem_param_88726") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88722, "mem_param_88722") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88718, "mem_param_88718") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88714, "mem_param_88714") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88710, "mem_param_88710") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88706, "mem_param_88706") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88702, "mem_param_88702") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88698, "mem_param_88698") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88694, "mem_param_88694") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88690, "mem_param_88690") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88686, "mem_param_88686") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_88682, "mem_param_88682") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90349, "ext_mem_90349") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90350, "ext_mem_90350") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90351, "ext_mem_90351") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90352, "ext_mem_90352") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90353, "ext_mem_90353") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90354, "ext_mem_90354") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90355, "ext_mem_90355") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90356, "ext_mem_90356") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90357, "ext_mem_90357") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90358, "ext_mem_90358") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90359, "ext_mem_90359") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90360, "ext_mem_90360") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90361, "ext_mem_90361") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90362, "ext_mem_90362") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90363, "ext_mem_90363") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90364, "ext_mem_90364") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90365, "ext_mem_90365") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90366, "ext_mem_90366") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90367, "ext_mem_90367") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90368, "ext_mem_90368") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90369, "ext_mem_90369") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90370, "ext_mem_90370") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90371, "ext_mem_90371") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90372, "ext_mem_90372") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90373, "ext_mem_90373") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90374, "ext_mem_90374") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_90375, "ext_mem_90375") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90474, "mem_out_90474") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90473, "mem_out_90473") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90472, "mem_out_90472") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90471, "mem_out_90471") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90470, "mem_out_90470") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90469, "mem_out_90469") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90468, "mem_out_90468") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90467, "mem_out_90467") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90466, "mem_out_90466") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90465, "mem_out_90465") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90464, "mem_out_90464") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90463, "mem_out_90463") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90462, "mem_out_90462") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90461, "mem_out_90461") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90460, "mem_out_90460") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90459, "mem_out_90459") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90458, "mem_out_90458") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90457, "mem_out_90457") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90456, "mem_out_90456") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90455, "mem_out_90455") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90454, "mem_out_90454") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90453, "mem_out_90453") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90452, "mem_out_90452") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90451, "mem_out_90451") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90450, "mem_out_90450") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90449, "mem_out_90449") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90448, "mem_out_90448") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_91036, struct memblock *mem_out_p_91037, struct memblock *mem_out_p_91038, struct memblock *mem_out_p_91039, struct memblock *mem_out_p_91040, struct memblock *mem_out_p_91041, struct memblock *mem_out_p_91042, struct memblock *mem_out_p_91043, struct memblock *mem_out_p_91044)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_90456;
    
    mem_out_90456.references = NULL;
    
    struct memblock mem_out_90455;
    
    mem_out_90455.references = NULL;
    
    struct memblock mem_out_90454;
    
    mem_out_90454.references = NULL;
    
    struct memblock mem_out_90453;
    
    mem_out_90453.references = NULL;
    
    struct memblock mem_out_90452;
    
    mem_out_90452.references = NULL;
    
    struct memblock mem_out_90451;
    
    mem_out_90451.references = NULL;
    
    struct memblock mem_out_90450;
    
    mem_out_90450.references = NULL;
    
    struct memblock mem_out_90449;
    
    mem_out_90449.references = NULL;
    
    struct memblock mem_out_90448;
    
    mem_out_90448.references = NULL;
    
    struct memblock mem_88640 = ctx->constants->mem_88640;
    struct memblock mem_88641 = ctx->constants->mem_88641;
    struct memblock mem_88642 = ctx->constants->mem_88642;
    struct memblock mem_88643 = ctx->constants->mem_88643;
    struct memblock mem_88644 = ctx->constants->mem_88644;
    struct memblock mem_88645 = ctx->constants->mem_88645;
    struct memblock mem_88646 = ctx->constants->mem_88646;
    struct memblock mem_88647 = ctx->constants->mem_88647;
    struct memblock mem_88648 = ctx->constants->mem_88648;
    
    if (memblock_set(ctx, &mem_out_90448, &mem_88647, "mem_88647") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90449, &mem_88643, "mem_88643") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90450, &mem_88645, "mem_88645") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90451, &mem_88641, "mem_88641") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90452, &mem_88642, "mem_88642") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90453, &mem_88640, "mem_88640") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90454, &mem_88646, "mem_88646") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90455, &mem_88644, "mem_88644") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_90456, &mem_88648, "mem_88648") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_91036, &mem_out_90448, "mem_out_90448") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_91037, &mem_out_90449, "mem_out_90449") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_91038, &mem_out_90450, "mem_out_90450") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_91039, &mem_out_90451, "mem_out_90451") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_91040, &mem_out_90452, "mem_out_90452") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_91041, &mem_out_90453, "mem_out_90453") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_91042, &mem_out_90454, "mem_out_90454") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_91043, &mem_out_90455, "mem_out_90455") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_91044, &mem_out_90456, "mem_out_90456") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_90456, "mem_out_90456") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90455, "mem_out_90455") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90454, "mem_out_90454") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90453, "mem_out_90453") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90452, "mem_out_90452") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90451, "mem_out_90451") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90450, "mem_out_90450") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90449, "mem_out_90449") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_90448, "mem_out_90448") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_forward_seq(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_90448;
    
    mem_out_90448.references = NULL;
    
    struct memblock mask_mem_88659;
    
    mask_mem_88659.references = NULL;
    
    struct memblock tokens_mem_88658;
    
    tokens_mem_88658.references = NULL;
    
    struct memblock wvoc_mem_88657;
    
    wvoc_mem_88657.references = NULL;
    
    struct memblock wval_mem_88656;
    
    wval_mem_88656.references = NULL;
    
    struct memblock wup_mem_88655;
    
    wup_mem_88655.references = NULL;
    
    struct memblock wte_mem_88654;
    
    wte_mem_88654.references = NULL;
    
    struct memblock wqry_mem_88653;
    
    wqry_mem_88653.references = NULL;
    
    struct memblock wpe_mem_88652;
    
    wpe_mem_88652.references = NULL;
    
    struct memblock wout_mem_88651;
    
    wout_mem_88651.references = NULL;
    
    struct memblock wkey_mem_88650;
    
    wkey_mem_88650.references = NULL;
    
    struct memblock wdown_mem_88649;
    
    wdown_mem_88649.references = NULL;
    wdown_mem_88649 = in0->v0->mem;
    wkey_mem_88650 = in0->v1->mem;
    wout_mem_88651 = in0->v2->mem;
    wpe_mem_88652 = in0->v3->mem;
    wqry_mem_88653 = in0->v4->mem;
    wte_mem_88654 = in0->v5->mem;
    wup_mem_88655 = in0->v6->mem;
    wval_mem_88656 = in0->v7->mem;
    wvoc_mem_88657 = in0->v8->mem;
    tokens_mem_88658 = in1->mem;
    mask_mem_88659 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_90448, wdown_mem_88649, wkey_mem_88650, wout_mem_88651, wpe_mem_88652, wqry_mem_88653, wte_mem_88654, wup_mem_88655, wval_mem_88656, wvoc_mem_88657, tokens_mem_88658, mask_mem_88659);
        if (ret == 0) {
            struct memblock mem_88640 = ctx->constants->mem_88640;
            struct memblock mem_88641 = ctx->constants->mem_88641;
            struct memblock mem_88642 = ctx->constants->mem_88642;
            struct memblock mem_88643 = ctx->constants->mem_88643;
            struct memblock mem_88644 = ctx->constants->mem_88644;
            struct memblock mem_88645 = ctx->constants->mem_88645;
            struct memblock mem_88646 = ctx->constants->mem_88646;
            struct memblock mem_88647 = ctx->constants->mem_88647;
            struct memblock mem_88648 = ctx->constants->mem_88648;
            
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_90448;
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
    
    struct memblock mem_out_90456;
    
    mem_out_90456.references = NULL;
    
    struct memblock mem_out_90455;
    
    mem_out_90455.references = NULL;
    
    struct memblock mem_out_90454;
    
    mem_out_90454.references = NULL;
    
    struct memblock mem_out_90453;
    
    mem_out_90453.references = NULL;
    
    struct memblock mem_out_90452;
    
    mem_out_90452.references = NULL;
    
    struct memblock mem_out_90451;
    
    mem_out_90451.references = NULL;
    
    struct memblock mem_out_90450;
    
    mem_out_90450.references = NULL;
    
    struct memblock mem_out_90449;
    
    mem_out_90449.references = NULL;
    
    struct memblock mem_out_90448;
    
    mem_out_90448.references = NULL;
    
    struct memblock wvoc_mem_88657;
    
    wvoc_mem_88657.references = NULL;
    
    struct memblock wdown_mem_88656;
    
    wdown_mem_88656.references = NULL;
    
    struct memblock wup_mem_88655;
    
    wup_mem_88655.references = NULL;
    
    struct memblock wout_mem_88654;
    
    wout_mem_88654.references = NULL;
    
    struct memblock wval_mem_88653;
    
    wval_mem_88653.references = NULL;
    
    struct memblock wkey_mem_88652;
    
    wkey_mem_88652.references = NULL;
    
    struct memblock wqry_mem_88651;
    
    wqry_mem_88651.references = NULL;
    
    struct memblock wpe_mem_88650;
    
    wpe_mem_88650.references = NULL;
    
    struct memblock wte_mem_88649;
    
    wte_mem_88649.references = NULL;
    wte_mem_88649 = in0->mem;
    wpe_mem_88650 = in1->mem;
    wqry_mem_88651 = in2->mem;
    wkey_mem_88652 = in3->mem;
    wval_mem_88653 = in4->mem;
    wout_mem_88654 = in5->mem;
    wup_mem_88655 = in6->mem;
    wdown_mem_88656 = in7->mem;
    wvoc_mem_88657 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_90448, &mem_out_90449, &mem_out_90450, &mem_out_90451, &mem_out_90452, &mem_out_90453, &mem_out_90454, &mem_out_90455, &mem_out_90456, wte_mem_88649, wpe_mem_88650, wqry_mem_88651, wkey_mem_88652, wval_mem_88653, wout_mem_88654, wup_mem_88655, wdown_mem_88656, wvoc_mem_88657);
        if (ret == 0) {
            struct memblock mem_88640 = ctx->constants->mem_88640;
            struct memblock mem_88641 = ctx->constants->mem_88641;
            struct memblock mem_88642 = ctx->constants->mem_88642;
            struct memblock mem_88643 = ctx->constants->mem_88643;
            struct memblock mem_88644 = ctx->constants->mem_88644;
            struct memblock mem_88645 = ctx->constants->mem_88645;
            struct memblock mem_88646 = ctx->constants->mem_88646;
            struct memblock mem_88647 = ctx->constants->mem_88647;
            struct memblock mem_88648 = ctx->constants->mem_88648;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_90448;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_90449;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_90450;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_90451;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_90452;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_90453;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_90454;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_90455;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_90456;
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
    
    struct memblock mem_out_90474;
    
    mem_out_90474.references = NULL;
    
    struct memblock mem_out_90473;
    
    mem_out_90473.references = NULL;
    
    struct memblock mem_out_90472;
    
    mem_out_90472.references = NULL;
    
    struct memblock mem_out_90471;
    
    mem_out_90471.references = NULL;
    
    struct memblock mem_out_90470;
    
    mem_out_90470.references = NULL;
    
    struct memblock mem_out_90469;
    
    mem_out_90469.references = NULL;
    
    struct memblock mem_out_90468;
    
    mem_out_90468.references = NULL;
    
    struct memblock mem_out_90467;
    
    mem_out_90467.references = NULL;
    
    struct memblock mem_out_90466;
    
    mem_out_90466.references = NULL;
    
    struct memblock mem_out_90465;
    
    mem_out_90465.references = NULL;
    
    struct memblock mem_out_90464;
    
    mem_out_90464.references = NULL;
    
    struct memblock mem_out_90463;
    
    mem_out_90463.references = NULL;
    
    struct memblock mem_out_90462;
    
    mem_out_90462.references = NULL;
    
    struct memblock mem_out_90461;
    
    mem_out_90461.references = NULL;
    
    struct memblock mem_out_90460;
    
    mem_out_90460.references = NULL;
    
    struct memblock mem_out_90459;
    
    mem_out_90459.references = NULL;
    
    struct memblock mem_out_90458;
    
    mem_out_90458.references = NULL;
    
    struct memblock mem_out_90457;
    
    mem_out_90457.references = NULL;
    
    struct memblock mem_out_90456;
    
    mem_out_90456.references = NULL;
    
    struct memblock mem_out_90455;
    
    mem_out_90455.references = NULL;
    
    struct memblock mem_out_90454;
    
    mem_out_90454.references = NULL;
    
    struct memblock mem_out_90453;
    
    mem_out_90453.references = NULL;
    
    struct memblock mem_out_90452;
    
    mem_out_90452.references = NULL;
    
    struct memblock mem_out_90451;
    
    mem_out_90451.references = NULL;
    
    struct memblock mem_out_90450;
    
    mem_out_90450.references = NULL;
    
    struct memblock mem_out_90449;
    
    mem_out_90449.references = NULL;
    
    struct memblock mem_out_90448;
    
    mem_out_90448.references = NULL;
    
    struct memblock seqs_mem_88678;
    
    seqs_mem_88678.references = NULL;
    
    struct memblock dls_mem_88677;
    
    dls_mem_88677.references = NULL;
    
    struct memblock masks_mem_88676;
    
    masks_mem_88676.references = NULL;
    
    struct memblock wvoc_mem_88675;
    
    wvoc_mem_88675.references = NULL;
    
    struct memblock wval_mem_88674;
    
    wval_mem_88674.references = NULL;
    
    struct memblock wup_mem_88673;
    
    wup_mem_88673.references = NULL;
    
    struct memblock wte_mem_88672;
    
    wte_mem_88672.references = NULL;
    
    struct memblock wqry_mem_88671;
    
    wqry_mem_88671.references = NULL;
    
    struct memblock wpe_mem_88670;
    
    wpe_mem_88670.references = NULL;
    
    struct memblock wout_mem_88669;
    
    wout_mem_88669.references = NULL;
    
    struct memblock wkey_mem_88668;
    
    wkey_mem_88668.references = NULL;
    
    struct memblock wdown_mem_88667;
    
    wdown_mem_88667.references = NULL;
    
    struct memblock wvoc_mem_88666;
    
    wvoc_mem_88666.references = NULL;
    
    struct memblock wval_mem_88665;
    
    wval_mem_88665.references = NULL;
    
    struct memblock wup_mem_88664;
    
    wup_mem_88664.references = NULL;
    
    struct memblock wte_mem_88663;
    
    wte_mem_88663.references = NULL;
    
    struct memblock wqry_mem_88662;
    
    wqry_mem_88662.references = NULL;
    
    struct memblock wpe_mem_88661;
    
    wpe_mem_88661.references = NULL;
    
    struct memblock wout_mem_88660;
    
    wout_mem_88660.references = NULL;
    
    struct memblock wkey_mem_88659;
    
    wkey_mem_88659.references = NULL;
    
    struct memblock wdown_mem_88658;
    
    wdown_mem_88658.references = NULL;
    
    struct memblock wvoc_mem_88657;
    
    wvoc_mem_88657.references = NULL;
    
    struct memblock wval_mem_88656;
    
    wval_mem_88656.references = NULL;
    
    struct memblock wup_mem_88655;
    
    wup_mem_88655.references = NULL;
    
    struct memblock wte_mem_88654;
    
    wte_mem_88654.references = NULL;
    
    struct memblock wqry_mem_88653;
    
    wqry_mem_88653.references = NULL;
    
    struct memblock wpe_mem_88652;
    
    wpe_mem_88652.references = NULL;
    
    struct memblock wout_mem_88651;
    
    wout_mem_88651.references = NULL;
    
    struct memblock wkey_mem_88650;
    
    wkey_mem_88650.references = NULL;
    
    struct memblock wdown_mem_88649;
    
    wdown_mem_88649.references = NULL;
    wdown_mem_88649 = in0->v0->mem;
    wkey_mem_88650 = in0->v1->mem;
    wout_mem_88651 = in0->v2->mem;
    wpe_mem_88652 = in0->v3->mem;
    wqry_mem_88653 = in0->v4->mem;
    wte_mem_88654 = in0->v5->mem;
    wup_mem_88655 = in0->v6->mem;
    wval_mem_88656 = in0->v7->mem;
    wvoc_mem_88657 = in0->v8->mem;
    wdown_mem_88658 = in1->v0->mem;
    wkey_mem_88659 = in1->v1->mem;
    wout_mem_88660 = in1->v2->mem;
    wpe_mem_88661 = in1->v3->mem;
    wqry_mem_88662 = in1->v4->mem;
    wte_mem_88663 = in1->v5->mem;
    wup_mem_88664 = in1->v6->mem;
    wval_mem_88665 = in1->v7->mem;
    wvoc_mem_88666 = in1->v8->mem;
    wdown_mem_88667 = in2->v0->mem;
    wkey_mem_88668 = in2->v1->mem;
    wout_mem_88669 = in2->v2->mem;
    wpe_mem_88670 = in2->v3->mem;
    wqry_mem_88671 = in2->v4->mem;
    wte_mem_88672 = in2->v5->mem;
    wup_mem_88673 = in2->v6->mem;
    wval_mem_88674 = in2->v7->mem;
    wvoc_mem_88675 = in2->v8->mem;
    masks_mem_88676 = in3->mem;
    dls_mem_88677 = in4->mem;
    seqs_mem_88678 = in5->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && (((int64_t) 500 == in3->shape[0] && ((int64_t) 16 == in3->shape[1] && (int64_t) 16 == in3->shape[2])) && ((int64_t) 500 == in4->shape[0] && ((int64_t) 500 == in5->shape[0] && (int64_t) 16 == in5->shape[1]))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_90448, &mem_out_90449, &mem_out_90450, &mem_out_90451, &mem_out_90452, &mem_out_90453, &mem_out_90454, &mem_out_90455, &mem_out_90456, &mem_out_90457, &mem_out_90458, &mem_out_90459, &mem_out_90460, &mem_out_90461, &mem_out_90462, &mem_out_90463, &mem_out_90464, &mem_out_90465, &mem_out_90466, &mem_out_90467, &mem_out_90468, &mem_out_90469, &mem_out_90470, &mem_out_90471, &mem_out_90472, &mem_out_90473, &mem_out_90474, wdown_mem_88649, wkey_mem_88650, wout_mem_88651, wpe_mem_88652, wqry_mem_88653, wte_mem_88654, wup_mem_88655, wval_mem_88656, wvoc_mem_88657, wdown_mem_88658, wkey_mem_88659, wout_mem_88660, wpe_mem_88661, wqry_mem_88662, wte_mem_88663, wup_mem_88664, wval_mem_88665, wvoc_mem_88666, wdown_mem_88667, wkey_mem_88668, wout_mem_88669, wpe_mem_88670, wqry_mem_88671, wte_mem_88672, wup_mem_88673, wval_mem_88674, wvoc_mem_88675, masks_mem_88676, dls_mem_88677, seqs_mem_88678);
        if (ret == 0) {
            struct memblock mem_88640 = ctx->constants->mem_88640;
            struct memblock mem_88641 = ctx->constants->mem_88641;
            struct memblock mem_88642 = ctx->constants->mem_88642;
            struct memblock mem_88643 = ctx->constants->mem_88643;
            struct memblock mem_88644 = ctx->constants->mem_88644;
            struct memblock mem_88645 = ctx->constants->mem_88645;
            struct memblock mem_88646 = ctx->constants->mem_88646;
            struct memblock mem_88647 = ctx->constants->mem_88647;
            struct memblock mem_88648 = ctx->constants->mem_88648;
            
            assert((*out = (struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_90448;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_90449;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_90450;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_90451;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_90452;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_90453;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_90454;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_90455;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_90456;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v9->mem = mem_out_90457;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v10->mem = mem_out_90458;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v11->mem = mem_out_90459;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v12->mem = mem_out_90460;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v13->mem = mem_out_90461;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v14->mem = mem_out_90462;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v15->mem = mem_out_90463;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v16->mem = mem_out_90464;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v17->mem = mem_out_90465;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v18->mem = mem_out_90466;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v19->mem = mem_out_90467;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v20->mem = mem_out_90468;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v21->mem = mem_out_90469;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v22->mem = mem_out_90470;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v23->mem = mem_out_90471;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v24->mem = mem_out_90472;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v25->mem = mem_out_90473;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v26->mem = mem_out_90474;
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
    
    struct memblock mem_out_90456;
    
    mem_out_90456.references = NULL;
    
    struct memblock mem_out_90455;
    
    mem_out_90455.references = NULL;
    
    struct memblock mem_out_90454;
    
    mem_out_90454.references = NULL;
    
    struct memblock mem_out_90453;
    
    mem_out_90453.references = NULL;
    
    struct memblock mem_out_90452;
    
    mem_out_90452.references = NULL;
    
    struct memblock mem_out_90451;
    
    mem_out_90451.references = NULL;
    
    struct memblock mem_out_90450;
    
    mem_out_90450.references = NULL;
    
    struct memblock mem_out_90449;
    
    mem_out_90449.references = NULL;
    
    struct memblock mem_out_90448;
    
    mem_out_90448.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_90448, &mem_out_90449, &mem_out_90450, &mem_out_90451, &mem_out_90452, &mem_out_90453, &mem_out_90454, &mem_out_90455, &mem_out_90456);
        if (ret == 0) {
            struct memblock mem_88640 = ctx->constants->mem_88640;
            struct memblock mem_88641 = ctx->constants->mem_88641;
            struct memblock mem_88642 = ctx->constants->mem_88642;
            struct memblock mem_88643 = ctx->constants->mem_88643;
            struct memblock mem_88644 = ctx->constants->mem_88644;
            struct memblock mem_88645 = ctx->constants->mem_88645;
            struct memblock mem_88646 = ctx->constants->mem_88646;
            struct memblock mem_88647 = ctx->constants->mem_88647;
            struct memblock mem_88648 = ctx->constants->mem_88648;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_90448;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_90449;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_90450;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_90451;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_90452;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_90453;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_90454;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_90455;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_90456;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
