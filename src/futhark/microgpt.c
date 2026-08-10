
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
struct futhark_opaque_tup2_f64_arr1d_f64;
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
const struct type type_ZLf64z2cUz20UZMZNf64ZR;
const struct type type_ZMZNZMZNZMZNf64;
const struct type type_ZMZNZMZNf64;
const struct type type_ZMZNZMZNi64;
const struct type type_ZMZNf64;
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
const struct type *types[] = {&type_i8, &type_i16, &type_i32, &type_i64, &type_u8, &type_u16, &type_u32, &type_u64, &type_f16, &type_f32, &type_f64, &type_bool, &type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRZR, &type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR, &type_ZLf64z2cUz20UZMZNf64ZR, &type_ZMZNZMZNZMZNf64, &type_ZMZNZMZNf64, &type_ZMZNZMZNi64, &type_ZMZNf64, &type_ZMZNi64, &type_params, NULL};
struct entry_point entry_points[] = {{.name ="cal_loss", .f =call_cal_loss, .tuning_params =cal_loss_tuning_params, .in_types =cal_loss_in_types, .out_type =&type_ZLf64z2cUz20UZMZNf64ZR, .in_unique =cal_loss_in_unique, .out_unique =false, .attrs =cal_loss_attrs}, {.name ="forward_seq", .f =call_forward_seq, .tuning_params =forward_seq_tuning_params, .in_types =forward_seq_in_types, .out_type =&type_ZMZNZMZNf64, .in_unique =forward_seq_in_unique, .out_unique =false, .attrs =forward_seq_attrs}, {.name ="to_params", .f =call_to_params, .tuning_params =to_params_tuning_params, .in_types =to_params_in_types, .out_type =&type_params, .in_unique =to_params_in_unique, .out_unique =false, .attrs =to_params_attrs}, {.name ="train", .f =call_train, .tuning_params =train_tuning_params, .in_types =train_in_types, .out_type =&type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRZR, .in_unique =train_in_unique, .out_unique =false, .attrs =train_attrs}, {.name ="zero_params", .f =call_zzero_params, .tuning_params =zzero_params_tuning_params, .in_types =zzero_params_in_types, .out_type =&type_params, .in_unique =zzero_params_in_unique, .out_unique =false, .attrs =zzero_params_attrs}, {.name =NULL}};
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
    struct memblock mem_126137;
    struct memblock mem_126138;
    struct memblock mem_126139;
    struct memblock mem_126140;
    struct memblock mem_126141;
    struct memblock mem_126142;
    struct memblock mem_126143;
    struct memblock mem_126144;
    struct memblock mem_126145;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_12300(struct futhark_context *ctx, struct memblock *mem_out_p_128294, struct memblock *mem_out_p_128295, struct memblock *mem_out_p_128296, struct memblock w_mem_126146, struct memblock mw_mem_126147, struct memblock vw_mem_126148, struct memblock dw_mem_126149, int64_t n_93423, int64_t m_93424, int64_t step_93429, double lt_r_93430);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_12301(struct futhark_context *ctx, struct memblock *mem_out_p_128299, struct memblock *mem_out_p_128300, struct memblock *mem_out_p_128301, struct memblock w_mem_126146, struct memblock mw_mem_126147, struct memblock vw_mem_126148, struct memblock dw_mem_126149, int64_t n_94456, int64_t m_94457, int64_t step_94462, double lt_r_94463);
FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_128304, double *out_prim_out_128305, struct memblock wdown_mem_126146, struct memblock wkey_mem_126147, struct memblock wout_mem_126148, struct memblock wpe_mem_126149, struct memblock wqry_mem_126150, struct memblock wte_mem_126151, struct memblock wup_mem_126152, struct memblock wval_mem_126153, struct memblock wvoc_mem_126154, struct memblock tokens_mem_126155, struct memblock target_mem_126156, struct memblock mask_mem_126157);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_128363, struct memblock wdown_mem_126146, struct memblock wkey_mem_126147, struct memblock wout_mem_126148, struct memblock wpe_mem_126149, struct memblock wqry_mem_126150, struct memblock wte_mem_126151, struct memblock wup_mem_126152, struct memblock wval_mem_126153, struct memblock wvoc_mem_126154, struct memblock tokens_mem_126155, struct memblock mask_mem_126156);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_128420, struct memblock *mem_out_p_128421, struct memblock *mem_out_p_128422, struct memblock *mem_out_p_128423, struct memblock *mem_out_p_128424, struct memblock *mem_out_p_128425, struct memblock *mem_out_p_128426, struct memblock *mem_out_p_128427, struct memblock *mem_out_p_128428, struct memblock wte_mem_126146, struct memblock wpe_mem_126147, struct memblock wqry_mem_126148, struct memblock wkey_mem_126149, struct memblock wval_mem_126150, struct memblock wout_mem_126151, struct memblock wup_mem_126152, struct memblock wdown_mem_126153, struct memblock wvoc_mem_126154);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_128429, struct memblock *mem_out_p_128430, struct memblock *mem_out_p_128431, struct memblock *mem_out_p_128432, struct memblock *mem_out_p_128433, struct memblock *mem_out_p_128434, struct memblock *mem_out_p_128435, struct memblock *mem_out_p_128436, struct memblock *mem_out_p_128437, struct memblock *mem_out_p_128438, struct memblock *mem_out_p_128439, struct memblock *mem_out_p_128440, struct memblock *mem_out_p_128441, struct memblock *mem_out_p_128442, struct memblock *mem_out_p_128443, struct memblock *mem_out_p_128444, struct memblock *mem_out_p_128445, struct memblock *mem_out_p_128446, struct memblock *mem_out_p_128447, struct memblock *mem_out_p_128448, struct memblock *mem_out_p_128449, struct memblock *mem_out_p_128450, struct memblock *mem_out_p_128451, struct memblock *mem_out_p_128452, struct memblock *mem_out_p_128453, struct memblock *mem_out_p_128454, struct memblock *mem_out_p_128455, struct memblock wdown_mem_126146, struct memblock wkey_mem_126147, struct memblock wout_mem_126148, struct memblock wpe_mem_126149, struct memblock wqry_mem_126150, struct memblock wte_mem_126151, struct memblock wup_mem_126152, struct memblock wval_mem_126153, struct memblock wvoc_mem_126154, struct memblock wdown_mem_126155, struct memblock wkey_mem_126156, struct memblock wout_mem_126157, struct memblock wpe_mem_126158, struct memblock wqry_mem_126159, struct memblock wte_mem_126160, struct memblock wup_mem_126161, struct memblock wval_mem_126162, struct memblock wvoc_mem_126163, struct memblock wdown_mem_126164, struct memblock wkey_mem_126165, struct memblock wout_mem_126166, struct memblock wpe_mem_126167, struct memblock wqry_mem_126168, struct memblock wte_mem_126169, struct memblock wup_mem_126170, struct memblock wval_mem_126171, struct memblock wvoc_mem_126172, struct memblock masks_mem_126173, struct memblock dls_mem_126174, struct memblock seqs_mem_126175);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_128627, struct memblock *mem_out_p_128628, struct memblock *mem_out_p_128629, struct memblock *mem_out_p_128630, struct memblock *mem_out_p_128631, struct memblock *mem_out_p_128632, struct memblock *mem_out_p_128633, struct memblock *mem_out_p_128634, struct memblock *mem_out_p_128635);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_126137 (ctx->constants->mem_126137)
    #define mem_126138 (ctx->constants->mem_126138)
    #define mem_126139 (ctx->constants->mem_126139)
    #define mem_126140 (ctx->constants->mem_126140)
    #define mem_126141 (ctx->constants->mem_126141)
    #define mem_126142 (ctx->constants->mem_126142)
    #define mem_126143 (ctx->constants->mem_126143)
    #define mem_126144 (ctx->constants->mem_126144)
    #define mem_126145 (ctx->constants->mem_126145)
    mem_126137.references = NULL;
    mem_126138.references = NULL;
    mem_126139.references = NULL;
    mem_126140.references = NULL;
    mem_126141.references = NULL;
    mem_126142.references = NULL;
    mem_126143.references = NULL;
    mem_126144.references = NULL;
    mem_126145.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_126137, (int64_t) 3456, "mem_126137")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_128276 = 0; nest_i_128276 < (int64_t) 27; nest_i_128276++) {
        for (int64_t nest_i_128277 = 0; nest_i_128277 < (int64_t) 16; nest_i_128277++) {
            ((double *) mem_126137.mem)[nest_i_128276 * (int64_t) 16 + nest_i_128277] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_126138, (int64_t) 2048, "mem_126138")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_128278 = 0; nest_i_128278 < (int64_t) 16; nest_i_128278++) {
        for (int64_t nest_i_128279 = 0; nest_i_128279 < (int64_t) 16; nest_i_128279++) {
            ((double *) mem_126138.mem)[nest_i_128278 * (int64_t) 16 + nest_i_128279] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_126139, (int64_t) 2048, "mem_126139")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_128280 = 0; nest_i_128280 < (int64_t) 16; nest_i_128280++) {
        for (int64_t nest_i_128281 = 0; nest_i_128281 < (int64_t) 16; nest_i_128281++) {
            ((double *) mem_126139.mem)[nest_i_128280 * (int64_t) 16 + nest_i_128281] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_126140, (int64_t) 2048, "mem_126140")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_128282 = 0; nest_i_128282 < (int64_t) 16; nest_i_128282++) {
        for (int64_t nest_i_128283 = 0; nest_i_128283 < (int64_t) 16; nest_i_128283++) {
            ((double *) mem_126140.mem)[nest_i_128282 * (int64_t) 16 + nest_i_128283] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_126141, (int64_t) 2048, "mem_126141")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_128284 = 0; nest_i_128284 < (int64_t) 16; nest_i_128284++) {
        for (int64_t nest_i_128285 = 0; nest_i_128285 < (int64_t) 16; nest_i_128285++) {
            ((double *) mem_126141.mem)[nest_i_128284 * (int64_t) 16 + nest_i_128285] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_126142, (int64_t) 2048, "mem_126142")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_128286 = 0; nest_i_128286 < (int64_t) 16; nest_i_128286++) {
        for (int64_t nest_i_128287 = 0; nest_i_128287 < (int64_t) 16; nest_i_128287++) {
            ((double *) mem_126142.mem)[nest_i_128286 * (int64_t) 16 + nest_i_128287] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_126143, (int64_t) 8192, "mem_126143")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_128288 = 0; nest_i_128288 < (int64_t) 64; nest_i_128288++) {
        for (int64_t nest_i_128289 = 0; nest_i_128289 < (int64_t) 16; nest_i_128289++) {
            ((double *) mem_126143.mem)[nest_i_128288 * (int64_t) 16 + nest_i_128289] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_126144, (int64_t) 8192, "mem_126144")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_128290 = 0; nest_i_128290 < (int64_t) 16; nest_i_128290++) {
        for (int64_t nest_i_128291 = 0; nest_i_128291 < (int64_t) 64; nest_i_128291++) {
            ((double *) mem_126144.mem)[nest_i_128290 * (int64_t) 64 + nest_i_128291] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_126145, (int64_t) 3456, "mem_126145")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_128292 = 0; nest_i_128292 < (int64_t) 27; nest_i_128292++) {
        for (int64_t nest_i_128293 = 0; nest_i_128293 < (int64_t) 16; nest_i_128293++) {
            ((double *) mem_126145.mem)[nest_i_128292 * (int64_t) 16 + nest_i_128293] = 0.0;
        }
    }
    #undef mem_126137
    #undef mem_126138
    #undef mem_126139
    #undef mem_126140
    #undef mem_126141
    #undef mem_126142
    #undef mem_126143
    #undef mem_126144
    #undef mem_126145
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_126137, "ctx->constants->mem_126137") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_126138, "ctx->constants->mem_126138") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_126139, "ctx->constants->mem_126139") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_126140, "ctx->constants->mem_126140") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_126141, "ctx->constants->mem_126141") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_126142, "ctx->constants->mem_126142") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_126143, "ctx->constants->mem_126143") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_126144, "ctx->constants->mem_126144") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_126145, "ctx->constants->mem_126145") != 0)
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_12300(struct futhark_context *ctx, struct memblock *mem_out_p_128294, struct memblock *mem_out_p_128295, struct memblock *mem_out_p_128296, struct memblock w_mem_126146, struct memblock mw_mem_126147, struct memblock vw_mem_126148, struct memblock dw_mem_126149, int64_t n_93423, int64_t m_93424, int64_t step_93429, double lt_r_93430)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_126190_cached_sizze_128297 = 0;
    unsigned char *mem_126190 = NULL;
    int64_t mem_126193_cached_sizze_128298 = 0;
    unsigned char *mem_126193 = NULL;
    struct memblock mem_126228;
    
    mem_126228.references = NULL;
    
    struct memblock mem_126155;
    
    mem_126155.references = NULL;
    
    struct memblock mem_126152;
    
    mem_126152.references = NULL;
    
    struct memblock mem_out_127956;
    
    mem_out_127956.references = NULL;
    
    struct memblock mem_out_127955;
    
    mem_out_127955.references = NULL;
    
    struct memblock mem_out_127954;
    
    mem_out_127954.references = NULL;
    
    struct memblock mem_126137 = ctx->constants->mem_126137;
    struct memblock mem_126138 = ctx->constants->mem_126138;
    struct memblock mem_126139 = ctx->constants->mem_126139;
    struct memblock mem_126140 = ctx->constants->mem_126140;
    struct memblock mem_126141 = ctx->constants->mem_126141;
    struct memblock mem_126142 = ctx->constants->mem_126142;
    struct memblock mem_126143 = ctx->constants->mem_126143;
    struct memblock mem_126144 = ctx->constants->mem_126144;
    struct memblock mem_126145 = ctx->constants->mem_126145;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_126150 = (int64_t) 8 * n_93423;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_126151 = m_93424 * binop_x_126150;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_126152, bytes_126151, "mem_126152")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_126155, bytes_126151, "mem_126155")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125290 = 0; i_125290 < n_93423; i_125290++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125283 = 0; i_125283 < m_93424; i_125283++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_118171 = ((double *) mw_mem_126147.mem)[i_125290 * m_93424 + i_125283];
            
            // futhark/microgpt.fut:447:10-20
            
            double zp_lhs_118172 = 0.85 * zt_rhs_118171;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_118173 = ((double *) dw_mem_126149.mem)[i_125290 * m_93424 + i_125283];
            
            // futhark/microgpt.fut:447:35-45
            
            double zp_rhs_118174 = 0.15000000000000002 * zt_rhs_118173;
            
            // futhark/microgpt.fut:447:21-45
            
            double lifted_lambda_res_118175 = zp_lhs_118172 + zp_rhs_118174;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_118182 = ((double *) vw_mem_126148.mem)[i_125290 * m_93424 + i_125283];
            
            // futhark/microgpt.fut:449:10-20
            
            double zp_lhs_118183 = 0.99 * zt_rhs_118182;
            
            // futhark/microgpt.fut:449:35-45
            
            double zt_lhs_118185 = 1.0000000000000009e-2 * zt_rhs_118173;
            
            // futhark/microgpt.fut:449:46-56
            
            double zp_rhs_118186 = zt_rhs_118173 * zt_lhs_118185;
            
            // futhark/microgpt.fut:449:21-56
            
            double lifted_lambda_res_118187 = zp_lhs_118183 + zp_rhs_118186;
            
            ((double *) mem_126152.mem)[i_125290 * m_93424 + i_125283] = lifted_lambda_res_118187;
            ((double *) mem_126155.mem)[i_125290 * m_93424 + i_125283] = lifted_lambda_res_118175;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_98414 = sitofp_i64_f64(step_93429);
    
    // futhark/microgpt.fut:451:54-57
    
    double ztzt_rhs_98415 = 1.0 + i64_res_98414;
    
    // futhark/microgpt.fut:451:30-57
    
    double zm_rhs_98416 = fpow64(0.85, ztzt_rhs_98415);
    
    // futhark/microgpt.fut:451:23-57
    
    double zs_rhs_98417 = 1.0 - zm_rhs_98416;
    
    // futhark/microgpt.fut:453:31-58
    
    double zm_rhs_98455 = fpow64(0.99, ztzt_rhs_98415);
    
    // futhark/microgpt.fut:453:23-58
    
    double zs_rhs_98456 = 1.0 - zm_rhs_98455;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_126190_cached_sizze_128297 < bytes_126151) {
        err = lexical_realloc(ctx, &mem_126190, &mem_126190_cached_sizze_128297, bytes_126151);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126193_cached_sizze_128298 < bytes_126151) {
        err = lexical_realloc(ctx, &mem_126193, &mem_126193_cached_sizze_128298, bytes_126151);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125304 = 0; i_125304 < n_93423; i_125304++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125297 = 0; i_125297 < m_93424; i_125297++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_118207 = ((double *) mem_126155.mem)[i_125304 * m_93424 + i_125297];
            
            // futhark/microgpt.fut:451:18-57
            
            double lifted_lambda_res_118208 = zs_lhs_118207 / zs_rhs_98417;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_118215 = ((double *) mem_126152.mem)[i_125304 * m_93424 + i_125297];
            
            // futhark/microgpt.fut:453:18-58
            
            double lifted_lambda_res_118216 = zs_lhs_118215 / zs_rhs_98456;
            
            ((double *) mem_126190)[i_125304 * m_93424 + i_125297] = lifted_lambda_res_118216;
            ((double *) mem_126193)[i_125304 * m_93424 + i_125297] = lifted_lambda_res_118208;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_126228, bytes_126151, "mem_126228")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125313 = 0; i_125313 < n_93423; i_125313++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125309 = 0; i_125309 < m_93424; i_125309++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_97635 = ((double *) w_mem_126146.mem)[i_125313 * m_93424 + i_125309];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_97636 = ((double *) mem_126193)[i_125313 * m_93424 + i_125309];
            
            // futhark/microgpt.fut:455:21-34
            
            double zs_lhs_97637 = lt_r_93430 * zt_rhs_97636;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_97638 = ((double *) mem_126190)[i_125313 * m_93424 + i_125309];
            
            // futhark/microgpt.fut:455:51-57
            
            double zp_lhs_97639 = fpow64(ztzt_lhs_97638, 0.5);
            
            // futhark/microgpt.fut:455:59-71
            
            double zs_rhs_97640 = 1.0e-8 + zp_lhs_97639;
            
            // futhark/microgpt.fut:455:35-71
            
            double zm_rhs_97641 = zs_lhs_97637 / zs_rhs_97640;
            
            // futhark/microgpt.fut:455:13-71
            
            double lifted_lambda_res_97642 = zm_lhs_97635 - zm_rhs_97641;
            
            ((double *) mem_126228.mem)[i_125313 * m_93424 + i_125309] = lifted_lambda_res_97642;
        }
    }
    if (memblock_set(ctx, &mem_out_127954, &mem_126228, "mem_126228") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127955, &mem_126155, "mem_126155") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127956, &mem_126152, "mem_126152") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128294, &mem_out_127954, "mem_out_127954") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128295, &mem_out_127955, "mem_out_127955") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128296, &mem_out_127956, "mem_out_127956") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_126190);
        free(mem_126193);
        if (memblock_unref(ctx, &mem_126228, "mem_126228") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_126155, "mem_126155") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_126152, "mem_126152") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127956, "mem_out_127956") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127955, "mem_out_127955") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127954, "mem_out_127954") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_12301(struct futhark_context *ctx, struct memblock *mem_out_p_128299, struct memblock *mem_out_p_128300, struct memblock *mem_out_p_128301, struct memblock w_mem_126146, struct memblock mw_mem_126147, struct memblock vw_mem_126148, struct memblock dw_mem_126149, int64_t n_94456, int64_t m_94457, int64_t step_94462, double lt_r_94463)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_126190_cached_sizze_128302 = 0;
    unsigned char *mem_126190 = NULL;
    int64_t mem_126193_cached_sizze_128303 = 0;
    unsigned char *mem_126193 = NULL;
    struct memblock mem_126228;
    
    mem_126228.references = NULL;
    
    struct memblock mem_126155;
    
    mem_126155.references = NULL;
    
    struct memblock mem_126152;
    
    mem_126152.references = NULL;
    
    struct memblock mem_out_127956;
    
    mem_out_127956.references = NULL;
    
    struct memblock mem_out_127955;
    
    mem_out_127955.references = NULL;
    
    struct memblock mem_out_127954;
    
    mem_out_127954.references = NULL;
    
    struct memblock mem_126137 = ctx->constants->mem_126137;
    struct memblock mem_126138 = ctx->constants->mem_126138;
    struct memblock mem_126139 = ctx->constants->mem_126139;
    struct memblock mem_126140 = ctx->constants->mem_126140;
    struct memblock mem_126141 = ctx->constants->mem_126141;
    struct memblock mem_126142 = ctx->constants->mem_126142;
    struct memblock mem_126143 = ctx->constants->mem_126143;
    struct memblock mem_126144 = ctx->constants->mem_126144;
    struct memblock mem_126145 = ctx->constants->mem_126145;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_126150 = (int64_t) 8 * n_94456;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_126151 = m_94457 * binop_x_126150;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_126152, bytes_126151, "mem_126152")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_126155, bytes_126151, "mem_126155")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125290 = 0; i_125290 < n_94456; i_125290++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125283 = 0; i_125283 < m_94457; i_125283++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_118171 = ((double *) mw_mem_126147.mem)[i_125290 * m_94457 + i_125283];
            
            // futhark/microgpt.fut:447:10-20
            
            double zp_lhs_118172 = 0.85 * zt_rhs_118171;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_118173 = ((double *) dw_mem_126149.mem)[i_125290 * m_94457 + i_125283];
            
            // futhark/microgpt.fut:447:35-45
            
            double zp_rhs_118174 = 0.15000000000000002 * zt_rhs_118173;
            
            // futhark/microgpt.fut:447:21-45
            
            double lifted_lambda_res_118175 = zp_lhs_118172 + zp_rhs_118174;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_118182 = ((double *) vw_mem_126148.mem)[i_125290 * m_94457 + i_125283];
            
            // futhark/microgpt.fut:449:10-20
            
            double zp_lhs_118183 = 0.99 * zt_rhs_118182;
            
            // futhark/microgpt.fut:449:35-45
            
            double zt_lhs_118185 = 1.0000000000000009e-2 * zt_rhs_118173;
            
            // futhark/microgpt.fut:449:46-56
            
            double zp_rhs_118186 = zt_rhs_118173 * zt_lhs_118185;
            
            // futhark/microgpt.fut:449:21-56
            
            double lifted_lambda_res_118187 = zp_lhs_118183 + zp_rhs_118186;
            
            ((double *) mem_126152.mem)[i_125290 * m_94457 + i_125283] = lifted_lambda_res_118187;
            ((double *) mem_126155.mem)[i_125290 * m_94457 + i_125283] = lifted_lambda_res_118175;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_98414 = sitofp_i64_f64(step_94462);
    
    // futhark/microgpt.fut:451:54-57
    
    double ztzt_rhs_98415 = 1.0 + i64_res_98414;
    
    // futhark/microgpt.fut:451:30-57
    
    double zm_rhs_98416 = fpow64(0.85, ztzt_rhs_98415);
    
    // futhark/microgpt.fut:451:23-57
    
    double zs_rhs_98417 = 1.0 - zm_rhs_98416;
    
    // futhark/microgpt.fut:453:31-58
    
    double zm_rhs_98455 = fpow64(0.99, ztzt_rhs_98415);
    
    // futhark/microgpt.fut:453:23-58
    
    double zs_rhs_98456 = 1.0 - zm_rhs_98455;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_126190_cached_sizze_128302 < bytes_126151) {
        err = lexical_realloc(ctx, &mem_126190, &mem_126190_cached_sizze_128302, bytes_126151);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126193_cached_sizze_128303 < bytes_126151) {
        err = lexical_realloc(ctx, &mem_126193, &mem_126193_cached_sizze_128303, bytes_126151);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125304 = 0; i_125304 < n_94456; i_125304++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125297 = 0; i_125297 < m_94457; i_125297++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_118207 = ((double *) mem_126155.mem)[i_125304 * m_94457 + i_125297];
            
            // futhark/microgpt.fut:451:18-57
            
            double lifted_lambda_res_118208 = zs_lhs_118207 / zs_rhs_98417;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_118215 = ((double *) mem_126152.mem)[i_125304 * m_94457 + i_125297];
            
            // futhark/microgpt.fut:453:18-58
            
            double lifted_lambda_res_118216 = zs_lhs_118215 / zs_rhs_98456;
            
            ((double *) mem_126190)[i_125304 * m_94457 + i_125297] = lifted_lambda_res_118216;
            ((double *) mem_126193)[i_125304 * m_94457 + i_125297] = lifted_lambda_res_118208;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_126228, bytes_126151, "mem_126228")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125313 = 0; i_125313 < n_94456; i_125313++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125309 = 0; i_125309 < m_94457; i_125309++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_97635 = ((double *) w_mem_126146.mem)[i_125313 * m_94457 + i_125309];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_97636 = ((double *) mem_126193)[i_125313 * m_94457 + i_125309];
            
            // futhark/microgpt.fut:455:21-34
            
            double zs_lhs_97637 = lt_r_94463 * zt_rhs_97636;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_97638 = ((double *) mem_126190)[i_125313 * m_94457 + i_125309];
            
            // futhark/microgpt.fut:455:51-57
            
            double zp_lhs_97639 = fpow64(ztzt_lhs_97638, 0.5);
            
            // futhark/microgpt.fut:455:59-71
            
            double zs_rhs_97640 = 1.0e-8 + zp_lhs_97639;
            
            // futhark/microgpt.fut:455:35-71
            
            double zm_rhs_97641 = zs_lhs_97637 / zs_rhs_97640;
            
            // futhark/microgpt.fut:455:13-71
            
            double lifted_lambda_res_97642 = zm_lhs_97635 - zm_rhs_97641;
            
            ((double *) mem_126228.mem)[i_125313 * m_94457 + i_125309] = lifted_lambda_res_97642;
        }
    }
    if (memblock_set(ctx, &mem_out_127954, &mem_126228, "mem_126228") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127955, &mem_126155, "mem_126155") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127956, &mem_126152, "mem_126152") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128299, &mem_out_127954, "mem_out_127954") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128300, &mem_out_127955, "mem_out_127955") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128301, &mem_out_127956, "mem_out_127956") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_126190);
        free(mem_126193);
        if (memblock_unref(ctx, &mem_126228, "mem_126228") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_126155, "mem_126155") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_126152, "mem_126152") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127956, "mem_out_127956") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127955, "mem_out_127955") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127954, "mem_out_127954") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_128304, double *out_prim_out_128305, struct memblock wdown_mem_126146, struct memblock wkey_mem_126147, struct memblock wout_mem_126148, struct memblock wpe_mem_126149, struct memblock wqry_mem_126150, struct memblock wte_mem_126151, struct memblock wup_mem_126152, struct memblock wval_mem_126153, struct memblock wvoc_mem_126154, struct memblock tokens_mem_126155, struct memblock target_mem_126156, struct memblock mask_mem_126157)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_126158_cached_sizze_128306 = 0;
    unsigned char *mem_126158 = NULL;
    int64_t mem_126163_cached_sizze_128307 = 0;
    unsigned char *mem_126163 = NULL;
    int64_t mem_126174_cached_sizze_128308 = 0;
    unsigned char *mem_126174 = NULL;
    int64_t mem_126179_cached_sizze_128309 = 0;
    unsigned char *mem_126179 = NULL;
    int64_t mem_126186_cached_sizze_128310 = 0;
    unsigned char *mem_126186 = NULL;
    int64_t mem_126197_cached_sizze_128311 = 0;
    unsigned char *mem_126197 = NULL;
    int64_t mem_126202_cached_sizze_128312 = 0;
    unsigned char *mem_126202 = NULL;
    int64_t mem_126209_cached_sizze_128313 = 0;
    unsigned char *mem_126209 = NULL;
    int64_t mem_126220_cached_sizze_128314 = 0;
    unsigned char *mem_126220 = NULL;
    int64_t mem_126221_cached_sizze_128315 = 0;
    unsigned char *mem_126221 = NULL;
    int64_t mem_126222_cached_sizze_128316 = 0;
    unsigned char *mem_126222 = NULL;
    int64_t mem_126235_cached_sizze_128317 = 0;
    unsigned char *mem_126235 = NULL;
    int64_t mem_126236_cached_sizze_128318 = 0;
    unsigned char *mem_126236 = NULL;
    int64_t mem_126237_cached_sizze_128319 = 0;
    unsigned char *mem_126237 = NULL;
    int64_t mem_126268_cached_sizze_128320 = 0;
    unsigned char *mem_126268 = NULL;
    int64_t mem_126269_cached_sizze_128321 = 0;
    unsigned char *mem_126269 = NULL;
    int64_t mem_126270_cached_sizze_128322 = 0;
    unsigned char *mem_126270 = NULL;
    int64_t mem_126286_cached_sizze_128323 = 0;
    unsigned char *mem_126286 = NULL;
    int64_t mem_126287_cached_sizze_128324 = 0;
    unsigned char *mem_126287 = NULL;
    int64_t mem_126288_cached_sizze_128325 = 0;
    unsigned char *mem_126288 = NULL;
    int64_t mem_126301_cached_sizze_128326 = 0;
    unsigned char *mem_126301 = NULL;
    int64_t mem_126302_cached_sizze_128327 = 0;
    unsigned char *mem_126302 = NULL;
    int64_t mem_126303_cached_sizze_128328 = 0;
    unsigned char *mem_126303 = NULL;
    int64_t mem_126349_cached_sizze_128329 = 0;
    unsigned char *mem_126349 = NULL;
    int64_t mem_126355_cached_sizze_128330 = 0;
    unsigned char *mem_126355 = NULL;
    int64_t mem_126360_cached_sizze_128331 = 0;
    unsigned char *mem_126360 = NULL;
    int64_t mem_126371_cached_sizze_128332 = 0;
    unsigned char *mem_126371 = NULL;
    int64_t mem_126376_cached_sizze_128333 = 0;
    unsigned char *mem_126376 = NULL;
    int64_t mem_126387_cached_sizze_128334 = 0;
    unsigned char *mem_126387 = NULL;
    int64_t mem_126392_cached_sizze_128335 = 0;
    unsigned char *mem_126392 = NULL;
    int64_t mem_126399_cached_sizze_128336 = 0;
    unsigned char *mem_126399 = NULL;
    int64_t mem_126406_cached_sizze_128337 = 0;
    unsigned char *mem_126406 = NULL;
    int64_t mem_126417_cached_sizze_128338 = 0;
    unsigned char *mem_126417 = NULL;
    int64_t mem_126422_cached_sizze_128339 = 0;
    unsigned char *mem_126422 = NULL;
    int64_t mem_126433_cached_sizze_128340 = 0;
    unsigned char *mem_126433 = NULL;
    int64_t mem_126438_cached_sizze_128341 = 0;
    unsigned char *mem_126438 = NULL;
    int64_t mem_126454_cached_sizze_128342 = 0;
    unsigned char *mem_126454 = NULL;
    int64_t mem_126459_cached_sizze_128343 = 0;
    unsigned char *mem_126459 = NULL;
    int64_t mem_126470_cached_sizze_128344 = 0;
    unsigned char *mem_126470 = NULL;
    int64_t mem_126475_cached_sizze_128345 = 0;
    unsigned char *mem_126475 = NULL;
    int64_t mem_126486_cached_sizze_128346 = 0;
    unsigned char *mem_126486 = NULL;
    int64_t mem_126491_cached_sizze_128347 = 0;
    unsigned char *mem_126491 = NULL;
    int64_t mem_126502_cached_sizze_128348 = 0;
    unsigned char *mem_126502 = NULL;
    int64_t mem_126507_cached_sizze_128349 = 0;
    unsigned char *mem_126507 = NULL;
    int64_t mem_126514_cached_sizze_128350 = 0;
    unsigned char *mem_126514 = NULL;
    int64_t mem_126525_cached_sizze_128351 = 0;
    unsigned char *mem_126525 = NULL;
    int64_t mem_126530_cached_sizze_128352 = 0;
    unsigned char *mem_126530 = NULL;
    int64_t mem_126541_cached_sizze_128353 = 0;
    unsigned char *mem_126541 = NULL;
    int64_t mem_126546_cached_sizze_128354 = 0;
    unsigned char *mem_126546 = NULL;
    int64_t mem_126557_cached_sizze_128355 = 0;
    unsigned char *mem_126557 = NULL;
    int64_t mem_126562_cached_sizze_128356 = 0;
    unsigned char *mem_126562 = NULL;
    int64_t mem_126573_cached_sizze_128357 = 0;
    unsigned char *mem_126573 = NULL;
    int64_t mem_126578_cached_sizze_128358 = 0;
    unsigned char *mem_126578 = NULL;
    int64_t mem_126589_cached_sizze_128359 = 0;
    unsigned char *mem_126589 = NULL;
    int64_t mem_126594_cached_sizze_128360 = 0;
    unsigned char *mem_126594 = NULL;
    int64_t mem_126609_cached_sizze_128361 = 0;
    unsigned char *mem_126609 = NULL;
    int64_t mem_126616_cached_sizze_128362 = 0;
    unsigned char *mem_126616 = NULL;
    struct memblock mem_126605;
    
    mem_126605.references = NULL;
    
    struct memblock mem_out_127954;
    
    mem_out_127954.references = NULL;
    
    struct memblock mem_126137 = ctx->constants->mem_126137;
    struct memblock mem_126138 = ctx->constants->mem_126138;
    struct memblock mem_126139 = ctx->constants->mem_126139;
    struct memblock mem_126140 = ctx->constants->mem_126140;
    struct memblock mem_126141 = ctx->constants->mem_126141;
    struct memblock mem_126142 = ctx->constants->mem_126142;
    struct memblock mem_126143 = ctx->constants->mem_126143;
    struct memblock mem_126144 = ctx->constants->mem_126144;
    struct memblock mem_126145 = ctx->constants->mem_126145;
    double prim_out_127955;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_126158_cached_sizze_128306 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126158, &mem_126158_cached_sizze_128306, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126163_cached_sizze_128307 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126163, &mem_126163_cached_sizze_128307, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125285 = 0; i_125285 < (int64_t) 16; i_125285++) {
        // futhark/microgpt.fut:437:41-50
        
        int64_t tmp_109360 = ((int64_t *) tokens_mem_126155.mem)[i_125285];
        
        // futhark/microgpt.fut:437:37-51
        
        bool x_109361 = sle64((int64_t) 0, tmp_109360);
        
        // futhark/microgpt.fut:437:37-51
        
        bool y_109362 = slt64(tmp_109360, (int64_t) 27);
        
        // futhark/microgpt.fut:437:37-51
        
        bool bounds_check_109363 = x_109361 && y_109362;
        
        // futhark/microgpt.fut:437:37-51
        
        bool index_certs_109364;
        
        if (!bounds_check_109363) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_109360, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:437:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:437:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125281 = 0; i_125281 < (int64_t) 16; i_125281++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_109371 = ((double *) wte_mem_126151.mem)[tmp_109360 * (int64_t) 16 + i_125281];
            
            ((double *) mem_126163)[i_125281] = lifted_lambda_res_109371;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126158, i_125285 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126163, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126174_cached_sizze_128308 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126174, &mem_126174_cached_sizze_128308, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126179_cached_sizze_128309 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126179, &mem_126179_cached_sizze_128309, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126186_cached_sizze_128310 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126186, &mem_126186_cached_sizze_128310, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125297 = 0; i_125297 < (int64_t) 16; i_125297++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_109397;
        double r_109399 = 0.0;
        
        for (int64_t i_109398 = 0; i_109398 < (int64_t) 16; i_109398++) {
            // futhark/microgpt.fut:71:46-49
            
            double zp_lhs_109400 = ((double *) wpe_mem_126149.mem)[i_125297 * (int64_t) 16 + i_109398];
            
            // futhark/microgpt.fut:71:46-49
            
            double zp_rhs_109401 = ((double *) mem_126158)[i_125297 * (int64_t) 16 + i_109398];
            
            // futhark/microgpt.fut:203:76-116
            
            double zp_res_109402 = zp_lhs_109400 + zp_rhs_109401;
            
            // futhark/microgpt.fut:203:94-163
            
            double zt_res_109403 = zp_res_109402 * zp_res_109402;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_109404 = r_109399 + zt_res_109403;
            double r_tmp_127959 = zp_res_109404;
            
            r_109399 = r_tmp_127959;
        }
        defunc_0_lifted_lambda_res_109397 = r_109399;
        // futhark/microgpt.fut:203:54-182
        
        double zs_res_109405 = defunc_0_lifted_lambda_res_109397 / 16.0;
        
        // futhark/microgpt.fut:204:24-55
        
        double zp_res_109406 = 1.0e-5 + zs_res_109405;
        
        // futhark/microgpt.fut:204:16-55
        
        double sqrt_res_109407 = futrts_sqrt64(zp_res_109406);
        
        // futhark/microgpt.fut:205:85-96
        
        double zs_res_109408 = 1.0 / sqrt_res_109407;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125289 = 0; i_125289 < (int64_t) 16; i_125289++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_109415 = ((double *) wpe_mem_126149.mem)[i_125297 * (int64_t) 16 + i_125289];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_109416 = ((double *) mem_126158)[i_125297 * (int64_t) 16 + i_125289];
            
            // futhark/microgpt.fut:205:38-78
            
            double zp_res_109417 = zp_lhs_109415 + zp_rhs_109416;
            
            // futhark/microgpt.fut:205:56-96
            
            double zt_res_109418 = zs_res_109408 * zp_res_109417;
            
            ((double *) mem_126179)[i_125289] = zt_res_109418;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125293 = 0; i_125293 < (int64_t) 16; i_125293++) {
            // futhark/microgpt.fut:206:4-14
            
            double lifted_lambda_res_109426 = ((double *) mem_126179)[i_125293];
            
            ((double *) mem_126186)[i_125293] = lifted_lambda_res_109426;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126174, i_125297 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126186, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126197_cached_sizze_128311 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126197, &mem_126197_cached_sizze_128311, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126202_cached_sizze_128312 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126202, &mem_126202_cached_sizze_128312, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126209_cached_sizze_128313 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126209, &mem_126209_cached_sizze_128313, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125309 = 0; i_125309 < (int64_t) 16; i_125309++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_109435;
        double r_109437 = 0.0;
        
        for (int64_t i_109436 = 0; i_109436 < (int64_t) 16; i_109436++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_109438 = ((double *) mem_126174)[i_125309 * (int64_t) 16 + i_109436];
            
            // futhark/microgpt.fut:207:78-115
            
            double zt_res_109439 = zt_lhs_109438 * zt_lhs_109438;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_109440 = r_109437 + zt_res_109439;
            double r_tmp_127963 = zp_res_109440;
            
            r_109437 = r_tmp_127963;
        }
        defunc_0_lifted_lambda_res_109435 = r_109437;
        // futhark/microgpt.fut:207:57-133
        
        double zs_res_109441 = defunc_0_lifted_lambda_res_109435 / 16.0;
        
        // futhark/microgpt.fut:208:24-55
        
        double zp_res_109442 = 1.0e-5 + zs_res_109441;
        
        // futhark/microgpt.fut:208:16-55
        
        double sqrt_res_109443 = futrts_sqrt64(zp_res_109442);
        
        // futhark/microgpt.fut:209:59-70
        
        double zs_res_109444 = 1.0 / sqrt_res_109443;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125301 = 0; i_125301 < (int64_t) 16; i_125301++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_109451 = ((double *) mem_126174)[i_125309 * (int64_t) 16 + i_125301];
            
            // futhark/microgpt.fut:209:37-70
            
            double zt_res_109452 = zs_res_109444 * zt_lhs_109451;
            
            ((double *) mem_126202)[i_125301] = zt_res_109452;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125305 = 0; i_125305 < (int64_t) 16; i_125305++) {
            // futhark/microgpt.fut:210:4-14
            
            double lifted_lambda_res_109460 = ((double *) mem_126202)[i_125305];
            
            ((double *) mem_126209)[i_125305] = lifted_lambda_res_109460;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126197, i_125309 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126209, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126220_cached_sizze_128314 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126220, &mem_126220_cached_sizze_128314, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126221_cached_sizze_128315 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126221, &mem_126221_cached_sizze_128315, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126222_cached_sizze_128316 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126222, &mem_126222_cached_sizze_128316, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126235_cached_sizze_128317 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126235, &mem_126235_cached_sizze_128317, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126236_cached_sizze_128318 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126236, &mem_126236_cached_sizze_128318, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126237_cached_sizze_128319 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126237, &mem_126237_cached_sizze_128319, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125327 = 0; i_125327 < (int64_t) 16; i_125327++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125317 = 0; i_125317 < (int64_t) 16; i_125317++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_118389;
            double r_118391 = 0.0;
            
            for (int64_t i_118390 = 0; i_118390 < (int64_t) 16; i_118390++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_118392 = ((double *) wqry_mem_126150.mem)[i_125317 * (int64_t) 16 + i_118390];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_118393 = ((double *) mem_126197)[i_125327 * (int64_t) 16 + i_118390];
                
                // futhark/microgpt.fut:211:66-105
                
                double zt_res_118394 = zt_lhs_118392 * zt_rhs_118393;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_118395 = r_118391 + zt_res_118394;
                double r_tmp_127972 = zp_res_118395;
                
                r_118391 = r_tmp_127972;
            }
            defunc_0_lifted_lambda_res_118389 = r_118391;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_118402;
            double r_118404 = 0.0;
            
            for (int64_t i_118403 = 0; i_118403 < (int64_t) 16; i_118403++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_118405 = ((double *) wkey_mem_126147.mem)[i_125317 * (int64_t) 16 + i_118403];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_118406 = ((double *) mem_126197)[i_125327 * (int64_t) 16 + i_118403];
                
                // futhark/microgpt.fut:212:66-105
                
                double zt_res_118407 = zt_lhs_118405 * zt_rhs_118406;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_118408 = r_118404 + zt_res_118407;
                double r_tmp_127973 = zp_res_118408;
                
                r_118404 = r_tmp_127973;
            }
            defunc_0_lifted_lambda_res_118402 = r_118404;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_118418;
            double r_118420 = 0.0;
            
            for (int64_t i_118419 = 0; i_118419 < (int64_t) 16; i_118419++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_118421 = ((double *) wval_mem_126153.mem)[i_125317 * (int64_t) 16 + i_118419];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_118422 = ((double *) mem_126197)[i_125327 * (int64_t) 16 + i_118419];
                
                // futhark/microgpt.fut:213:66-105
                
                double zt_res_118423 = zt_lhs_118421 * zt_rhs_118422;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_118424 = r_118420 + zt_res_118423;
                double r_tmp_127974 = zp_res_118424;
                
                r_118420 = r_tmp_127974;
            }
            defunc_0_lifted_lambda_res_118418 = r_118420;
            ((double *) mem_126235)[i_125317] = defunc_0_lifted_lambda_res_118418;
            ((double *) mem_126236)[i_125317] = defunc_0_lifted_lambda_res_118402;
            ((double *) mem_126237)[i_125317] = defunc_0_lifted_lambda_res_118389;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126220, i_125327 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126235, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126221, i_125327 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126236, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126222, i_125327 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126237, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126268_cached_sizze_128320 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126268, &mem_126268_cached_sizze_128320, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126269_cached_sizze_128321 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126269, &mem_126269_cached_sizze_128321, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126270_cached_sizze_128322 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126270, &mem_126270_cached_sizze_128322, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126286_cached_sizze_128323 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126286, &mem_126286_cached_sizze_128323, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126287_cached_sizze_128324 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126287, &mem_126287_cached_sizze_128324, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126288_cached_sizze_128325 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126288, &mem_126288_cached_sizze_128325, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126301_cached_sizze_128326 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_126301, &mem_126301_cached_sizze_128326, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126302_cached_sizze_128327 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_126302, &mem_126302_cached_sizze_128327, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126303_cached_sizze_128328 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_126303, &mem_126303_cached_sizze_128328, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125357 = 0; i_125357 < (int64_t) 4; i_125357++) {
        // futhark/microgpt.fut:214:69-72
        
        int64_t zp_lhs_118265 = mul64((int64_t) 4, i_125357);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125347 = 0; i_125347 < (int64_t) 16; i_125347++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125337 = 0; i_125337 < (int64_t) 4; i_125337++) {
                // futhark/microgpt.fut:214:74-81
                
                int64_t tmp_118582 = add64(zp_lhs_118265, i_125337);
                
                // futhark/microgpt.fut:214:51-83
                
                bool x_118583 = sle64((int64_t) 0, tmp_118582);
                
                // futhark/microgpt.fut:214:51-83
                
                bool y_118584 = slt64(tmp_118582, (int64_t) 16);
                
                // futhark/microgpt.fut:214:51-83
                
                bool bounds_check_118585 = x_118583 && y_118584;
                
                // futhark/microgpt.fut:214:51-83
                
                bool index_certs_118586;
                
                if (!bounds_check_118585) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_118582, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:214:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:214:15-84\n   #9  futhark/microgpt.fut:438:7-76\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_118587 = ((double *) mem_126222)[i_125347 * (int64_t) 16 + tmp_118582];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_118595 = ((double *) mem_126221)[i_125347 * (int64_t) 16 + tmp_118582];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_118606 = ((double *) mem_126220)[i_125347 * (int64_t) 16 + tmp_118582];
                
                ((double *) mem_126301)[i_125337] = lifted_lambda_res_118606;
                ((double *) mem_126302)[i_125337] = lifted_lambda_res_118595;
                ((double *) mem_126303)[i_125337] = lifted_lambda_res_118587;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126286, i_125347 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126301, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126287, i_125347 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126302, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126288, i_125347 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126303, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_126268, i_125357 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_126286, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_126269, i_125357 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_126287, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_126270, i_125357 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_126288, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126349_cached_sizze_128329 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126349, &mem_126349_cached_sizze_128329, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126355_cached_sizze_128330 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126355, &mem_126355_cached_sizze_128330, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126360_cached_sizze_128331 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126360, &mem_126360_cached_sizze_128331, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126371_cached_sizze_128332 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126371, &mem_126371_cached_sizze_128332, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126376_cached_sizze_128333 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126376, &mem_126376_cached_sizze_128333, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126387_cached_sizze_128334 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126387, &mem_126387_cached_sizze_128334, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126392_cached_sizze_128335 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126392, &mem_126392_cached_sizze_128335, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126399_cached_sizze_128336 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126399, &mem_126399_cached_sizze_128336, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126406_cached_sizze_128337 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126406, &mem_126406_cached_sizze_128337, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126417_cached_sizze_128338 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126417, &mem_126417_cached_sizze_128338, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126422_cached_sizze_128339 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_126422, &mem_126422_cached_sizze_128339, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126433_cached_sizze_128340 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126433, &mem_126433_cached_sizze_128340, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126438_cached_sizze_128341 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_126438, &mem_126438_cached_sizze_128341, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125413 = 0; i_125413 < (int64_t) 4; i_125413++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125367 = 0; i_125367 < (int64_t) 16; i_125367++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125363 = 0; i_125363 < (int64_t) 16; i_125363++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_109605;
                double r_109607 = 0.0;
                
                for (int64_t i_109606 = 0; i_109606 < (int64_t) 4; i_109606++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_109608 = ((double *) mem_126270)[i_125413 * (int64_t) 64 + i_125367 * (int64_t) 4 + i_109606];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_109609 = ((double *) mem_126269)[i_125413 * (int64_t) 64 + i_125363 * (int64_t) 4 + i_109606];
                    
                    // futhark/microgpt.fut:217:113-164
                    
                    double zt_res_109610 = zt_lhs_109608 * zt_rhs_109609;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_109611 = r_109607 + zt_res_109610;
                    double r_tmp_127987 = zp_res_109611;
                    
                    r_109607 = r_tmp_127987;
                }
                defunc_0_lifted_lambda_res_109605 = r_109607;
                ((double *) mem_126360)[i_125363] = defunc_0_lifted_lambda_res_109605;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126355, i_125367 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126360, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125375 = 0; i_125375 < (int64_t) 16; i_125375++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125371 = 0; i_125371 < (int64_t) 16; i_125371++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_109626 = ((double *) mem_126355)[i_125375 * (int64_t) 16 + i_125371];
                
                // futhark/microgpt.fut:218:47-78
                
                double zs_res_109627 = zs_lhs_109626 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_109628 = ((double *) mask_mem_126157.mem)[i_125375 * (int64_t) 16 + i_125371];
                
                // futhark/microgpt.fut:218:65-102
                
                double zp_res_109629 = zs_res_109627 + zp_rhs_109628;
                
                ((double *) mem_126376)[i_125371] = zp_res_109629;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126371, i_125375 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126376, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125393 = 0; i_125393 < (int64_t) 16; i_125393++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_118709;
            double redout_125377 = -INFINITY;
            
            for (int64_t i_125378 = 0; i_125378 < (int64_t) 16; i_125378++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_118633 = ((double *) mem_126371)[i_125393 * (int64_t) 16 + i_125378];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_109650 = fmax64(lifted_lambda_res_118633, redout_125377);
                double redout_tmp_127991 = max_res_109650;
                
                redout_125377 = redout_tmp_127991;
            }
            defunc_0_reduce_res_118709 = redout_125377;
            // futhark/microgpt.fut:220:67-76
            
            double neg_res_109651 = -defunc_0_reduce_res_118709;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125381 = 0; i_125381 < (int64_t) 16; i_125381++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_109658 = ((double *) mem_126371)[i_125393 * (int64_t) 16 + i_125381];
                
                // futhark/microgpt.fut:220:44-76
                
                double zp_res_109659 = neg_res_109651 + zp_lhs_109658;
                
                // futhark/microgpt.fut:220:37-76
                
                double exp_res_109660 = futrts_exp64(zp_res_109659);
                
                ((double *) mem_126392)[i_125381] = exp_res_109660;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_109662;
            double r_109664 = 0.0;
            
            for (int64_t i_109663 = 0; i_109663 < (int64_t) 16; i_109663++) {
                // futhark/microgpt.fut:221:36-46
                
                double lifted_lambda_res_109665 = ((double *) mem_126392)[i_109663];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_109666 = r_109664 + lifted_lambda_res_109665;
                double r_tmp_127993 = zp_res_109666;
                
                r_109664 = r_tmp_127993;
            }
            defunc_0_lifted_lambda_res_109662 = r_109664;
            // futhark/microgpt.fut:222:53-64
            
            double zs_res_109667 = 1.0 / defunc_0_lifted_lambda_res_109662;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125385 = 0; i_125385 < (int64_t) 16; i_125385++) {
                // futhark/microgpt.fut:222:37-47
                
                double zt_lhs_109674 = ((double *) mem_126392)[i_125385];
                
                // futhark/microgpt.fut:222:37-64
                
                double zt_res_109675 = zs_res_109667 * zt_lhs_109674;
                
                ((double *) mem_126399)[i_125385] = zt_res_109675;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125389 = 0; i_125389 < (int64_t) 16; i_125389++) {
                // futhark/microgpt.fut:223:4-14
                
                double lifted_lambda_res_109683 = ((double *) mem_126399)[i_125389];
                
                ((double *) mem_126406)[i_125389] = lifted_lambda_res_109683;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126387, i_125393 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126406, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125401 = 0; i_125401 < (int64_t) 16; i_125401++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125397 = 0; i_125397 < (int64_t) 4; i_125397++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_109698;
                double r_109700 = 0.0;
                
                for (int64_t i_109699 = 0; i_109699 < (int64_t) 16; i_109699++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_109701 = ((double *) mem_126387)[i_125401 * (int64_t) 16 + i_109699];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_109702 = ((double *) mem_126268)[i_125413 * (int64_t) 64 + i_109699 * (int64_t) 4 + i_125397];
                    
                    // futhark/microgpt.fut:224:66-111
                    
                    double zt_res_109703 = zt_lhs_109701 * zt_rhs_109702;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_109704 = r_109700 + zt_res_109703;
                    double r_tmp_127998 = zp_res_109704;
                    
                    r_109700 = r_tmp_127998;
                }
                defunc_0_lifted_lambda_res_109698 = r_109700;
                ((double *) mem_126422)[i_125397] = defunc_0_lifted_lambda_res_109698;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126417, i_125401 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126422, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125409 = 0; i_125409 < (int64_t) 16; i_125409++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125405 = 0; i_125405 < (int64_t) 4; i_125405++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_109719 = ((double *) mem_126417)[i_125409 * (int64_t) 4 + i_125405];
                
                ((double *) mem_126438)[i_125405] = lifted_lambda_res_109719;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126433, i_125409 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126438, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_126349, i_125413 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_126433, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126454_cached_sizze_128342 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126454, &mem_126454_cached_sizze_128342, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126459_cached_sizze_128343 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126459, &mem_126459_cached_sizze_128343, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125421 = 0; i_125421 < (int64_t) 16; i_125421++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125417 = 0; i_125417 < (int64_t) 16; i_125417++) {
            // futhark/microgpt.fut:226:54-57
            
            int64_t tmp_109731 = sdiv64(i_125417, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-59
            
            bool x_109732 = sle64((int64_t) 0, tmp_109731);
            
            // futhark/microgpt.fut:226:44-59
            
            bool y_109733 = slt64(tmp_109731, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-59
            
            bool bounds_check_109734 = x_109732 && y_109733;
            
            // futhark/microgpt.fut:226:44-59
            
            bool index_certs_109735;
            
            if (!bounds_check_109734) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_109731, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:226:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:226:15-80\n   #6  futhark/microgpt.fut:438:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:226:74-77
            
            int64_t tmp_109736 = smod64(i_125417, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-79
            
            bool x_109737 = sle64((int64_t) 0, tmp_109736);
            
            // futhark/microgpt.fut:226:44-79
            
            bool y_109738 = slt64(tmp_109736, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-79
            
            bool bounds_check_109739 = x_109737 && y_109738;
            
            // futhark/microgpt.fut:226:44-79
            
            bool index_certs_109740;
            
            if (!bounds_check_109739) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_109736, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:226:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:226:15-80\n   #6  futhark/microgpt.fut:438:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_109741 = ((double *) mem_126349)[tmp_109731 * (int64_t) 64 + i_125421 * (int64_t) 4 + tmp_109736];
            
            ((double *) mem_126459)[i_125417] = lifted_lambda_res_109741;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126454, i_125421 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126459, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126470_cached_sizze_128344 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126470, &mem_126470_cached_sizze_128344, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126475_cached_sizze_128345 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126475, &mem_126475_cached_sizze_128345, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125429 = 0; i_125429 < (int64_t) 16; i_125429++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125425 = 0; i_125425 < (int64_t) 16; i_125425++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_109756;
            double r_109758 = 0.0;
            
            for (int64_t i_109757 = 0; i_109757 < (int64_t) 16; i_109757++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_109759 = ((double *) wout_mem_126148.mem)[i_125425 * (int64_t) 16 + i_109757];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_109760 = ((double *) mem_126454)[i_125429 * (int64_t) 16 + i_109757];
                
                // futhark/microgpt.fut:227:67-106
                
                double zt_res_109761 = zt_lhs_109759 * zt_rhs_109760;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_109762 = r_109758 + zt_res_109761;
                double r_tmp_128005 = zp_res_109762;
                
                r_109758 = r_tmp_128005;
            }
            defunc_0_lifted_lambda_res_109756 = r_109758;
            ((double *) mem_126475)[i_125425] = defunc_0_lifted_lambda_res_109756;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126470, i_125429 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126475, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126486_cached_sizze_128346 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126486, &mem_126486_cached_sizze_128346, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126491_cached_sizze_128347 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126491, &mem_126491_cached_sizze_128347, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125437 = 0; i_125437 < (int64_t) 16; i_125437++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125433 = 0; i_125433 < (int64_t) 16; i_125433++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_109777 = ((double *) mem_126470)[i_125437 * (int64_t) 16 + i_125433];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_109778 = ((double *) mem_126174)[i_125437 * (int64_t) 16 + i_125433];
            
            // futhark/microgpt.fut:228:46-84
            
            double zp_res_109779 = zp_lhs_109777 + zp_rhs_109778;
            
            ((double *) mem_126491)[i_125433] = zp_res_109779;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126486, i_125437 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126491, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126502_cached_sizze_128348 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126502, &mem_126502_cached_sizze_128348, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126507_cached_sizze_128349 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126507, &mem_126507_cached_sizze_128349, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126514_cached_sizze_128350 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126514, &mem_126514_cached_sizze_128350, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125449 = 0; i_125449 < (int64_t) 16; i_125449++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_109788;
        double r_109790 = 0.0;
        
        for (int64_t i_109789 = 0; i_109789 < (int64_t) 16; i_109789++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_109791 = ((double *) mem_126486)[i_125449 * (int64_t) 16 + i_109789];
            
            // futhark/microgpt.fut:229:79-118
            
            double zt_res_109792 = zt_lhs_109791 * zt_lhs_109791;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_109793 = r_109790 + zt_res_109792;
            double r_tmp_128009 = zp_res_109793;
            
            r_109790 = r_tmp_128009;
        }
        defunc_0_lifted_lambda_res_109788 = r_109790;
        // futhark/microgpt.fut:229:58-136
        
        double zs_res_109794 = defunc_0_lifted_lambda_res_109788 / 16.0;
        
        // futhark/microgpt.fut:230:24-55
        
        double zp_res_109795 = 1.0e-5 + zs_res_109794;
        
        // futhark/microgpt.fut:230:16-55
        
        double sqrt_res_109796 = futrts_sqrt64(zp_res_109795);
        
        // futhark/microgpt.fut:231:60-71
        
        double zs_res_109797 = 1.0 / sqrt_res_109796;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125441 = 0; i_125441 < (int64_t) 16; i_125441++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_109804 = ((double *) mem_126486)[i_125449 * (int64_t) 16 + i_125441];
            
            // futhark/microgpt.fut:231:37-71
            
            double zt_res_109805 = zs_res_109797 * zt_lhs_109804;
            
            ((double *) mem_126507)[i_125441] = zt_res_109805;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125445 = 0; i_125445 < (int64_t) 16; i_125445++) {
            // futhark/microgpt.fut:232:4-14
            
            double lifted_lambda_res_109813 = ((double *) mem_126507)[i_125445];
            
            ((double *) mem_126514)[i_125445] = lifted_lambda_res_109813;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126502, i_125449 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126514, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126525_cached_sizze_128351 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_126525, &mem_126525_cached_sizze_128351, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126530_cached_sizze_128352 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126530, &mem_126530_cached_sizze_128352, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125457 = 0; i_125457 < (int64_t) 16; i_125457++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125453 = 0; i_125453 < (int64_t) 64; i_125453++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_109829;
            double r_109831 = 0.0;
            
            for (int64_t i_109830 = 0; i_109830 < (int64_t) 16; i_109830++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_109832 = ((double *) wup_mem_126152.mem)[i_125453 * (int64_t) 16 + i_109830];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_109833 = ((double *) mem_126502)[i_125457 * (int64_t) 16 + i_109830];
                
                // futhark/microgpt.fut:233:67-106
                
                double zt_res_109834 = zt_lhs_109832 * zt_rhs_109833;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_109835 = r_109831 + zt_res_109834;
                double r_tmp_128014 = zp_res_109835;
                
                r_109831 = r_tmp_128014;
            }
            defunc_0_lifted_lambda_res_109829 = r_109831;
            ((double *) mem_126530)[i_125453] = defunc_0_lifted_lambda_res_109829;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126525, i_125457 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126530, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126541_cached_sizze_128353 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_126541, &mem_126541_cached_sizze_128353, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126546_cached_sizze_128354 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126546, &mem_126546_cached_sizze_128354, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125465 = 0; i_125465 < (int64_t) 16; i_125465++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125461 = 0; i_125461 < (int64_t) 64; i_125461++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_109850 = ((double *) mem_126525)[i_125465 * (int64_t) 64 + i_125461];
            
            // futhark/microgpt.fut:234:45-73
            
            double max_res_109851 = fmax64(0.0, max_arg0_109850);
            
            ((double *) mem_126546)[i_125461] = max_res_109851;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126541, i_125465 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126546, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126557_cached_sizze_128355 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126557, &mem_126557_cached_sizze_128355, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126562_cached_sizze_128356 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126562, &mem_126562_cached_sizze_128356, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125473 = 0; i_125473 < (int64_t) 16; i_125473++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125469 = 0; i_125469 < (int64_t) 16; i_125469++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_109866;
            double r_109868 = 0.0;
            
            for (int64_t i_109867 = 0; i_109867 < (int64_t) 64; i_109867++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_109869 = ((double *) wdown_mem_126146.mem)[i_125469 * (int64_t) 64 + i_109867];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_109870 = ((double *) mem_126541)[i_125473 * (int64_t) 64 + i_109867];
                
                // futhark/microgpt.fut:235:67-108
                
                double zt_res_109871 = zt_lhs_109869 * zt_rhs_109870;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_109872 = r_109868 + zt_res_109871;
                double r_tmp_128019 = zp_res_109872;
                
                r_109868 = r_tmp_128019;
            }
            defunc_0_lifted_lambda_res_109866 = r_109868;
            ((double *) mem_126562)[i_125469] = defunc_0_lifted_lambda_res_109866;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126557, i_125473 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126562, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126573_cached_sizze_128357 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126573, &mem_126573_cached_sizze_128357, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126578_cached_sizze_128358 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126578, &mem_126578_cached_sizze_128358, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125481 = 0; i_125481 < (int64_t) 16; i_125481++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125477 = 0; i_125477 < (int64_t) 16; i_125477++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_109887 = ((double *) mem_126557)[i_125481 * (int64_t) 16 + i_125477];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_109888 = ((double *) mem_126486)[i_125481 * (int64_t) 16 + i_125477];
            
            // futhark/microgpt.fut:236:46-85
            
            double zp_res_109889 = zp_lhs_109887 + zp_rhs_109888;
            
            ((double *) mem_126578)[i_125477] = zp_res_109889;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126573, i_125481 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126578, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126589_cached_sizze_128359 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_126589, &mem_126589_cached_sizze_128359, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126594_cached_sizze_128360 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_126594, &mem_126594_cached_sizze_128360, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125489 = 0; i_125489 < (int64_t) 16; i_125489++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125485 = 0; i_125485 < (int64_t) 27; i_125485++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_109905;
            double r_109907 = 0.0;
            
            for (int64_t i_109906 = 0; i_109906 < (int64_t) 16; i_109906++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_109908 = ((double *) wvoc_mem_126154.mem)[i_125485 * (int64_t) 16 + i_109906];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_109909 = ((double *) mem_126573)[i_125489 * (int64_t) 16 + i_109906];
                
                // futhark/microgpt.fut:237:67-107
                
                double zt_res_109910 = zt_lhs_109908 * zt_rhs_109909;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_109911 = r_109907 + zt_res_109910;
                double r_tmp_128024 = zp_res_109911;
                
                r_109907 = r_tmp_128024;
            }
            defunc_0_lifted_lambda_res_109905 = r_109907;
            ((double *) mem_126594)[i_125485] = defunc_0_lifted_lambda_res_109905;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126589, i_125489 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126594, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_126605, (int64_t) 128, "mem_126605")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126609_cached_sizze_128361 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_126609, &mem_126609_cached_sizze_128361, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126616_cached_sizze_128362 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_126616, &mem_126616_cached_sizze_128362, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125503 = 0; i_125503 < (int64_t) 16; i_125503++) {
        double x_118732;
        double redout_125491 = -INFINITY;
        
        for (int64_t i_125492 = 0; i_125492 < (int64_t) 27; i_125492++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_118679 = ((double *) mem_126589)[i_125503 * (int64_t) 27 + i_125492];
            
            // futhark/microgpt.fut:115:13-33
            
            double max_res_109935 = fmax64(lifted_lambda_res_118679, redout_125491);
            double redout_tmp_128026 = max_res_109935;
            
            redout_125491 = redout_tmp_128026;
        }
        x_118732 = redout_125491;
        // futhark/microgpt.fut:239:67-76
        
        double neg_res_109936 = -x_118732;
        
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_109920;
        double r_109922 = 0.0;
        
        for (int64_t i_109921 = 0; i_109921 < (int64_t) 27; i_109921++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125495 = 0; i_125495 < (int64_t) 27; i_125495++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_109943 = ((double *) mem_126589)[i_125503 * (int64_t) 27 + i_125495];
                
                // futhark/microgpt.fut:239:44-76
                
                double zp_res_109944 = neg_res_109936 + zp_lhs_109943;
                
                // futhark/microgpt.fut:239:37-76
                
                double exp_res_109945 = futrts_exp64(zp_res_109944);
                
                ((double *) mem_126609)[i_125495] = exp_res_109945;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_109947;
            double r_109949 = 0.0;
            
            for (int64_t i_109948 = 0; i_109948 < (int64_t) 27; i_109948++) {
                // futhark/microgpt.fut:240:36-46
                
                double lifted_lambda_res_109950 = ((double *) mem_126609)[i_109948];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_109951 = r_109949 + lifted_lambda_res_109950;
                double r_tmp_128029 = zp_res_109951;
                
                r_109949 = r_tmp_128029;
            }
            defunc_0_lifted_lambda_res_109947 = r_109949;
            // futhark/microgpt.fut:241:53-64
            
            double zs_res_109952 = 1.0 / defunc_0_lifted_lambda_res_109947;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125499 = 0; i_125499 < (int64_t) 27; i_125499++) {
                // futhark/microgpt.fut:241:37-47
                
                double zt_lhs_109959 = ((double *) mem_126609)[i_125499];
                
                // futhark/microgpt.fut:241:37-64
                
                double zt_res_109960 = zs_res_109952 * zt_lhs_109959;
                
                ((double *) mem_126616)[i_125499] = zt_res_109960;
            }
            // futhark/microgpt.fut:242:12-22
            
            double log_arg0_109962 = ((double *) mem_126616)[i_109921];
            
            // futhark/microgpt.fut:242:6-22
            
            double log_res_109963 = futrts_log64(log_arg0_109962);
            
            // futhark/microgpt.fut:71:46-49
            
            double zt_rhs_109964 = ((double *) target_mem_126156.mem)[i_125503 * (int64_t) 27 + i_109921];
            
            // futhark/microgpt.fut:242:6-48
            
            double zt_res_109965 = log_res_109963 * zt_rhs_109964;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_109966 = r_109922 + zt_res_109965;
            double r_tmp_128027 = zp_res_109966;
            
            r_109922 = r_tmp_128027;
        }
        defunc_0_lifted_lambda_res_109920 = r_109922;
        // futhark/microgpt.fut:238:37-242:54
        
        double neg_res_109967 = -defunc_0_lifted_lambda_res_109920;
        
        ((double *) mem_126605.mem)[i_125503] = neg_res_109967;
    }
    // futhark/microgpt.fut:71:13-49
    
    double defunc_0_lifted_lambda_res_109969;
    double r_109971 = 0.0;
    
    for (int64_t i_109970 = 0; i_109970 < (int64_t) 16; i_109970++) {
        // futhark/microgpt.fut:243:37-47
        
        double lifted_lambda_res_109972 = ((double *) mem_126605.mem)[i_109970];
        
        // futhark/microgpt.fut:71:40-49
        
        double zp_res_109973 = r_109971 + lifted_lambda_res_109972;
        double r_tmp_128031 = zp_res_109973;
        
        r_109971 = r_tmp_128031;
    }
    defunc_0_lifted_lambda_res_109969 = r_109971;
    // futhark/microgpt.fut:243:17-64
    
    double zs_res_109974 = defunc_0_lifted_lambda_res_109969 / 16.0;
    
    if (memblock_set(ctx, &mem_out_127954, &mem_126605, "mem_126605") != 0)
        return 1;
    prim_out_127955 = zs_res_109974;
    if (memblock_set(ctx, &*mem_out_p_128304, &mem_out_127954, "mem_out_127954") != 0)
        return 1;
    *out_prim_out_128305 = prim_out_127955;
    
  cleanup:
    {
        free(mem_126158);
        free(mem_126163);
        free(mem_126174);
        free(mem_126179);
        free(mem_126186);
        free(mem_126197);
        free(mem_126202);
        free(mem_126209);
        free(mem_126220);
        free(mem_126221);
        free(mem_126222);
        free(mem_126235);
        free(mem_126236);
        free(mem_126237);
        free(mem_126268);
        free(mem_126269);
        free(mem_126270);
        free(mem_126286);
        free(mem_126287);
        free(mem_126288);
        free(mem_126301);
        free(mem_126302);
        free(mem_126303);
        free(mem_126349);
        free(mem_126355);
        free(mem_126360);
        free(mem_126371);
        free(mem_126376);
        free(mem_126387);
        free(mem_126392);
        free(mem_126399);
        free(mem_126406);
        free(mem_126417);
        free(mem_126422);
        free(mem_126433);
        free(mem_126438);
        free(mem_126454);
        free(mem_126459);
        free(mem_126470);
        free(mem_126475);
        free(mem_126486);
        free(mem_126491);
        free(mem_126502);
        free(mem_126507);
        free(mem_126514);
        free(mem_126525);
        free(mem_126530);
        free(mem_126541);
        free(mem_126546);
        free(mem_126557);
        free(mem_126562);
        free(mem_126573);
        free(mem_126578);
        free(mem_126589);
        free(mem_126594);
        free(mem_126609);
        free(mem_126616);
        if (memblock_unref(ctx, &mem_126605, "mem_126605") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127954, "mem_out_127954") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_128363, struct memblock wdown_mem_126146, struct memblock wkey_mem_126147, struct memblock wout_mem_126148, struct memblock wpe_mem_126149, struct memblock wqry_mem_126150, struct memblock wte_mem_126151, struct memblock wup_mem_126152, struct memblock wval_mem_126153, struct memblock wvoc_mem_126154, struct memblock tokens_mem_126155, struct memblock mask_mem_126156)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_126157_cached_sizze_128364 = 0;
    unsigned char *mem_126157 = NULL;
    int64_t mem_126162_cached_sizze_128365 = 0;
    unsigned char *mem_126162 = NULL;
    int64_t mem_126173_cached_sizze_128366 = 0;
    unsigned char *mem_126173 = NULL;
    int64_t mem_126178_cached_sizze_128367 = 0;
    unsigned char *mem_126178 = NULL;
    int64_t mem_126185_cached_sizze_128368 = 0;
    unsigned char *mem_126185 = NULL;
    int64_t mem_126196_cached_sizze_128369 = 0;
    unsigned char *mem_126196 = NULL;
    int64_t mem_126201_cached_sizze_128370 = 0;
    unsigned char *mem_126201 = NULL;
    int64_t mem_126208_cached_sizze_128371 = 0;
    unsigned char *mem_126208 = NULL;
    int64_t mem_126219_cached_sizze_128372 = 0;
    unsigned char *mem_126219 = NULL;
    int64_t mem_126220_cached_sizze_128373 = 0;
    unsigned char *mem_126220 = NULL;
    int64_t mem_126221_cached_sizze_128374 = 0;
    unsigned char *mem_126221 = NULL;
    int64_t mem_126234_cached_sizze_128375 = 0;
    unsigned char *mem_126234 = NULL;
    int64_t mem_126235_cached_sizze_128376 = 0;
    unsigned char *mem_126235 = NULL;
    int64_t mem_126236_cached_sizze_128377 = 0;
    unsigned char *mem_126236 = NULL;
    int64_t mem_126267_cached_sizze_128378 = 0;
    unsigned char *mem_126267 = NULL;
    int64_t mem_126268_cached_sizze_128379 = 0;
    unsigned char *mem_126268 = NULL;
    int64_t mem_126269_cached_sizze_128380 = 0;
    unsigned char *mem_126269 = NULL;
    int64_t mem_126285_cached_sizze_128381 = 0;
    unsigned char *mem_126285 = NULL;
    int64_t mem_126286_cached_sizze_128382 = 0;
    unsigned char *mem_126286 = NULL;
    int64_t mem_126287_cached_sizze_128383 = 0;
    unsigned char *mem_126287 = NULL;
    int64_t mem_126300_cached_sizze_128384 = 0;
    unsigned char *mem_126300 = NULL;
    int64_t mem_126301_cached_sizze_128385 = 0;
    unsigned char *mem_126301 = NULL;
    int64_t mem_126302_cached_sizze_128386 = 0;
    unsigned char *mem_126302 = NULL;
    int64_t mem_126348_cached_sizze_128387 = 0;
    unsigned char *mem_126348 = NULL;
    int64_t mem_126354_cached_sizze_128388 = 0;
    unsigned char *mem_126354 = NULL;
    int64_t mem_126359_cached_sizze_128389 = 0;
    unsigned char *mem_126359 = NULL;
    int64_t mem_126370_cached_sizze_128390 = 0;
    unsigned char *mem_126370 = NULL;
    int64_t mem_126375_cached_sizze_128391 = 0;
    unsigned char *mem_126375 = NULL;
    int64_t mem_126386_cached_sizze_128392 = 0;
    unsigned char *mem_126386 = NULL;
    int64_t mem_126391_cached_sizze_128393 = 0;
    unsigned char *mem_126391 = NULL;
    int64_t mem_126398_cached_sizze_128394 = 0;
    unsigned char *mem_126398 = NULL;
    int64_t mem_126405_cached_sizze_128395 = 0;
    unsigned char *mem_126405 = NULL;
    int64_t mem_126416_cached_sizze_128396 = 0;
    unsigned char *mem_126416 = NULL;
    int64_t mem_126421_cached_sizze_128397 = 0;
    unsigned char *mem_126421 = NULL;
    int64_t mem_126432_cached_sizze_128398 = 0;
    unsigned char *mem_126432 = NULL;
    int64_t mem_126437_cached_sizze_128399 = 0;
    unsigned char *mem_126437 = NULL;
    int64_t mem_126453_cached_sizze_128400 = 0;
    unsigned char *mem_126453 = NULL;
    int64_t mem_126458_cached_sizze_128401 = 0;
    unsigned char *mem_126458 = NULL;
    int64_t mem_126469_cached_sizze_128402 = 0;
    unsigned char *mem_126469 = NULL;
    int64_t mem_126474_cached_sizze_128403 = 0;
    unsigned char *mem_126474 = NULL;
    int64_t mem_126485_cached_sizze_128404 = 0;
    unsigned char *mem_126485 = NULL;
    int64_t mem_126490_cached_sizze_128405 = 0;
    unsigned char *mem_126490 = NULL;
    int64_t mem_126501_cached_sizze_128406 = 0;
    unsigned char *mem_126501 = NULL;
    int64_t mem_126506_cached_sizze_128407 = 0;
    unsigned char *mem_126506 = NULL;
    int64_t mem_126513_cached_sizze_128408 = 0;
    unsigned char *mem_126513 = NULL;
    int64_t mem_126524_cached_sizze_128409 = 0;
    unsigned char *mem_126524 = NULL;
    int64_t mem_126529_cached_sizze_128410 = 0;
    unsigned char *mem_126529 = NULL;
    int64_t mem_126540_cached_sizze_128411 = 0;
    unsigned char *mem_126540 = NULL;
    int64_t mem_126545_cached_sizze_128412 = 0;
    unsigned char *mem_126545 = NULL;
    int64_t mem_126556_cached_sizze_128413 = 0;
    unsigned char *mem_126556 = NULL;
    int64_t mem_126561_cached_sizze_128414 = 0;
    unsigned char *mem_126561 = NULL;
    int64_t mem_126572_cached_sizze_128415 = 0;
    unsigned char *mem_126572 = NULL;
    int64_t mem_126577_cached_sizze_128416 = 0;
    unsigned char *mem_126577 = NULL;
    int64_t mem_126588_cached_sizze_128417 = 0;
    unsigned char *mem_126588 = NULL;
    int64_t mem_126593_cached_sizze_128418 = 0;
    unsigned char *mem_126593 = NULL;
    int64_t mem_126609_cached_sizze_128419 = 0;
    unsigned char *mem_126609 = NULL;
    struct memblock mem_126604;
    
    mem_126604.references = NULL;
    
    struct memblock mem_out_127954;
    
    mem_out_127954.references = NULL;
    
    struct memblock mem_126137 = ctx->constants->mem_126137;
    struct memblock mem_126138 = ctx->constants->mem_126138;
    struct memblock mem_126139 = ctx->constants->mem_126139;
    struct memblock mem_126140 = ctx->constants->mem_126140;
    struct memblock mem_126141 = ctx->constants->mem_126141;
    struct memblock mem_126142 = ctx->constants->mem_126142;
    struct memblock mem_126143 = ctx->constants->mem_126143;
    struct memblock mem_126144 = ctx->constants->mem_126144;
    struct memblock mem_126145 = ctx->constants->mem_126145;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_126157_cached_sizze_128364 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126157, &mem_126157_cached_sizze_128364, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126162_cached_sizze_128365 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126162, &mem_126162_cached_sizze_128365, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125285 = 0; i_125285 < (int64_t) 16; i_125285++) {
        // futhark/microgpt.fut:432:41-50
        
        int64_t tmp_109359 = ((int64_t *) tokens_mem_126155.mem)[i_125285];
        
        // futhark/microgpt.fut:432:37-51
        
        bool x_109360 = sle64((int64_t) 0, tmp_109359);
        
        // futhark/microgpt.fut:432:37-51
        
        bool y_109361 = slt64(tmp_109359, (int64_t) 27);
        
        // futhark/microgpt.fut:432:37-51
        
        bool bounds_check_109362 = x_109360 && y_109361;
        
        // futhark/microgpt.fut:432:37-51
        
        bool index_certs_109363;
        
        if (!bounds_check_109362) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_109359, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:432:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:432:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125281 = 0; i_125281 < (int64_t) 16; i_125281++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_109370 = ((double *) wte_mem_126151.mem)[tmp_109359 * (int64_t) 16 + i_125281];
            
            ((double *) mem_126162)[i_125281] = lifted_lambda_res_109370;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126157, i_125285 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126162, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126173_cached_sizze_128366 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126173, &mem_126173_cached_sizze_128366, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126178_cached_sizze_128367 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126178, &mem_126178_cached_sizze_128367, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126185_cached_sizze_128368 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126185, &mem_126185_cached_sizze_128368, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125297 = 0; i_125297 < (int64_t) 16; i_125297++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_109396;
        double r_109398 = 0.0;
        
        for (int64_t i_109397 = 0; i_109397 < (int64_t) 16; i_109397++) {
            // futhark/microgpt.fut:71:46-49
            
            double zp_lhs_109399 = ((double *) wpe_mem_126149.mem)[i_125297 * (int64_t) 16 + i_109397];
            
            // futhark/microgpt.fut:71:46-49
            
            double zp_rhs_109400 = ((double *) mem_126157)[i_125297 * (int64_t) 16 + i_109397];
            
            // futhark/microgpt.fut:148:76-116
            
            double zp_res_109401 = zp_lhs_109399 + zp_rhs_109400;
            
            // futhark/microgpt.fut:148:94-163
            
            double zt_res_109402 = zp_res_109401 * zp_res_109401;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_109403 = r_109398 + zt_res_109402;
            double r_tmp_127958 = zp_res_109403;
            
            r_109398 = r_tmp_127958;
        }
        defunc_0_lifted_lambda_res_109396 = r_109398;
        // futhark/microgpt.fut:148:54-182
        
        double zs_res_109404 = defunc_0_lifted_lambda_res_109396 / 16.0;
        
        // futhark/microgpt.fut:149:24-55
        
        double zp_res_109405 = 1.0e-5 + zs_res_109404;
        
        // futhark/microgpt.fut:149:16-55
        
        double sqrt_res_109406 = futrts_sqrt64(zp_res_109405);
        
        // futhark/microgpt.fut:150:85-96
        
        double zs_res_109407 = 1.0 / sqrt_res_109406;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125289 = 0; i_125289 < (int64_t) 16; i_125289++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_109414 = ((double *) wpe_mem_126149.mem)[i_125297 * (int64_t) 16 + i_125289];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_109415 = ((double *) mem_126157)[i_125297 * (int64_t) 16 + i_125289];
            
            // futhark/microgpt.fut:150:38-78
            
            double zp_res_109416 = zp_lhs_109414 + zp_rhs_109415;
            
            // futhark/microgpt.fut:150:56-96
            
            double zt_res_109417 = zs_res_109407 * zp_res_109416;
            
            ((double *) mem_126178)[i_125289] = zt_res_109417;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125293 = 0; i_125293 < (int64_t) 16; i_125293++) {
            // futhark/microgpt.fut:151:4-14
            
            double lifted_lambda_res_109425 = ((double *) mem_126178)[i_125293];
            
            ((double *) mem_126185)[i_125293] = lifted_lambda_res_109425;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126173, i_125297 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126185, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126196_cached_sizze_128369 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126196, &mem_126196_cached_sizze_128369, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126201_cached_sizze_128370 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126201, &mem_126201_cached_sizze_128370, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126208_cached_sizze_128371 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126208, &mem_126208_cached_sizze_128371, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125309 = 0; i_125309 < (int64_t) 16; i_125309++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_109434;
        double r_109436 = 0.0;
        
        for (int64_t i_109435 = 0; i_109435 < (int64_t) 16; i_109435++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_109437 = ((double *) mem_126173)[i_125309 * (int64_t) 16 + i_109435];
            
            // futhark/microgpt.fut:152:78-115
            
            double zt_res_109438 = zt_lhs_109437 * zt_lhs_109437;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_109439 = r_109436 + zt_res_109438;
            double r_tmp_127962 = zp_res_109439;
            
            r_109436 = r_tmp_127962;
        }
        defunc_0_lifted_lambda_res_109434 = r_109436;
        // futhark/microgpt.fut:152:57-133
        
        double zs_res_109440 = defunc_0_lifted_lambda_res_109434 / 16.0;
        
        // futhark/microgpt.fut:153:24-55
        
        double zp_res_109441 = 1.0e-5 + zs_res_109440;
        
        // futhark/microgpt.fut:153:16-55
        
        double sqrt_res_109442 = futrts_sqrt64(zp_res_109441);
        
        // futhark/microgpt.fut:154:59-70
        
        double zs_res_109443 = 1.0 / sqrt_res_109442;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125301 = 0; i_125301 < (int64_t) 16; i_125301++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_109450 = ((double *) mem_126173)[i_125309 * (int64_t) 16 + i_125301];
            
            // futhark/microgpt.fut:154:37-70
            
            double zt_res_109451 = zs_res_109443 * zt_lhs_109450;
            
            ((double *) mem_126201)[i_125301] = zt_res_109451;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125305 = 0; i_125305 < (int64_t) 16; i_125305++) {
            // futhark/microgpt.fut:155:4-14
            
            double lifted_lambda_res_109459 = ((double *) mem_126201)[i_125305];
            
            ((double *) mem_126208)[i_125305] = lifted_lambda_res_109459;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126196, i_125309 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126208, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126219_cached_sizze_128372 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126219, &mem_126219_cached_sizze_128372, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126220_cached_sizze_128373 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126220, &mem_126220_cached_sizze_128373, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126221_cached_sizze_128374 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126221, &mem_126221_cached_sizze_128374, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126234_cached_sizze_128375 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126234, &mem_126234_cached_sizze_128375, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126235_cached_sizze_128376 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126235, &mem_126235_cached_sizze_128376, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126236_cached_sizze_128377 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126236, &mem_126236_cached_sizze_128377, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125327 = 0; i_125327 < (int64_t) 16; i_125327++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125317 = 0; i_125317 < (int64_t) 16; i_125317++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_118389;
            double r_118391 = 0.0;
            
            for (int64_t i_118390 = 0; i_118390 < (int64_t) 16; i_118390++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_118392 = ((double *) wqry_mem_126150.mem)[i_125317 * (int64_t) 16 + i_118390];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_118393 = ((double *) mem_126196)[i_125327 * (int64_t) 16 + i_118390];
                
                // futhark/microgpt.fut:156:66-105
                
                double zt_res_118394 = zt_lhs_118392 * zt_rhs_118393;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_118395 = r_118391 + zt_res_118394;
                double r_tmp_127971 = zp_res_118395;
                
                r_118391 = r_tmp_127971;
            }
            defunc_0_lifted_lambda_res_118389 = r_118391;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_118402;
            double r_118404 = 0.0;
            
            for (int64_t i_118403 = 0; i_118403 < (int64_t) 16; i_118403++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_118405 = ((double *) wkey_mem_126147.mem)[i_125317 * (int64_t) 16 + i_118403];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_118406 = ((double *) mem_126196)[i_125327 * (int64_t) 16 + i_118403];
                
                // futhark/microgpt.fut:157:66-105
                
                double zt_res_118407 = zt_lhs_118405 * zt_rhs_118406;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_118408 = r_118404 + zt_res_118407;
                double r_tmp_127972 = zp_res_118408;
                
                r_118404 = r_tmp_127972;
            }
            defunc_0_lifted_lambda_res_118402 = r_118404;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_118418;
            double r_118420 = 0.0;
            
            for (int64_t i_118419 = 0; i_118419 < (int64_t) 16; i_118419++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_118421 = ((double *) wval_mem_126153.mem)[i_125317 * (int64_t) 16 + i_118419];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_118422 = ((double *) mem_126196)[i_125327 * (int64_t) 16 + i_118419];
                
                // futhark/microgpt.fut:158:66-105
                
                double zt_res_118423 = zt_lhs_118421 * zt_rhs_118422;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_118424 = r_118420 + zt_res_118423;
                double r_tmp_127973 = zp_res_118424;
                
                r_118420 = r_tmp_127973;
            }
            defunc_0_lifted_lambda_res_118418 = r_118420;
            ((double *) mem_126234)[i_125317] = defunc_0_lifted_lambda_res_118418;
            ((double *) mem_126235)[i_125317] = defunc_0_lifted_lambda_res_118402;
            ((double *) mem_126236)[i_125317] = defunc_0_lifted_lambda_res_118389;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126219, i_125327 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126234, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126220, i_125327 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126235, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126221, i_125327 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126236, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126267_cached_sizze_128378 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126267, &mem_126267_cached_sizze_128378, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126268_cached_sizze_128379 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126268, &mem_126268_cached_sizze_128379, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126269_cached_sizze_128380 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126269, &mem_126269_cached_sizze_128380, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126285_cached_sizze_128381 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126285, &mem_126285_cached_sizze_128381, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126286_cached_sizze_128382 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126286, &mem_126286_cached_sizze_128382, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126287_cached_sizze_128383 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126287, &mem_126287_cached_sizze_128383, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126300_cached_sizze_128384 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_126300, &mem_126300_cached_sizze_128384, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126301_cached_sizze_128385 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_126301, &mem_126301_cached_sizze_128385, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126302_cached_sizze_128386 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_126302, &mem_126302_cached_sizze_128386, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125357 = 0; i_125357 < (int64_t) 4; i_125357++) {
        // futhark/microgpt.fut:159:69-72
        
        int64_t zp_lhs_118265 = mul64((int64_t) 4, i_125357);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125347 = 0; i_125347 < (int64_t) 16; i_125347++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125337 = 0; i_125337 < (int64_t) 4; i_125337++) {
                // futhark/microgpt.fut:159:74-81
                
                int64_t tmp_118582 = add64(zp_lhs_118265, i_125337);
                
                // futhark/microgpt.fut:159:51-83
                
                bool x_118583 = sle64((int64_t) 0, tmp_118582);
                
                // futhark/microgpt.fut:159:51-83
                
                bool y_118584 = slt64(tmp_118582, (int64_t) 16);
                
                // futhark/microgpt.fut:159:51-83
                
                bool bounds_check_118585 = x_118583 && y_118584;
                
                // futhark/microgpt.fut:159:51-83
                
                bool index_certs_118586;
                
                if (!bounds_check_118585) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_118582, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:159:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:159:15-84\n   #9  futhark/microgpt.fut:433:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_118587 = ((double *) mem_126221)[i_125347 * (int64_t) 16 + tmp_118582];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_118595 = ((double *) mem_126220)[i_125347 * (int64_t) 16 + tmp_118582];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_118606 = ((double *) mem_126219)[i_125347 * (int64_t) 16 + tmp_118582];
                
                ((double *) mem_126300)[i_125337] = lifted_lambda_res_118606;
                ((double *) mem_126301)[i_125337] = lifted_lambda_res_118595;
                ((double *) mem_126302)[i_125337] = lifted_lambda_res_118587;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126285, i_125347 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126300, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126286, i_125347 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126301, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126287, i_125347 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126302, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_126267, i_125357 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_126285, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_126268, i_125357 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_126286, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_126269, i_125357 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_126287, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126348_cached_sizze_128387 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126348, &mem_126348_cached_sizze_128387, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126354_cached_sizze_128388 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126354, &mem_126354_cached_sizze_128388, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126359_cached_sizze_128389 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126359, &mem_126359_cached_sizze_128389, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126370_cached_sizze_128390 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126370, &mem_126370_cached_sizze_128390, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126375_cached_sizze_128391 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126375, &mem_126375_cached_sizze_128391, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126386_cached_sizze_128392 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126386, &mem_126386_cached_sizze_128392, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126391_cached_sizze_128393 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126391, &mem_126391_cached_sizze_128393, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126398_cached_sizze_128394 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126398, &mem_126398_cached_sizze_128394, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126405_cached_sizze_128395 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126405, &mem_126405_cached_sizze_128395, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126416_cached_sizze_128396 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126416, &mem_126416_cached_sizze_128396, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126421_cached_sizze_128397 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_126421, &mem_126421_cached_sizze_128397, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126432_cached_sizze_128398 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126432, &mem_126432_cached_sizze_128398, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126437_cached_sizze_128399 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_126437, &mem_126437_cached_sizze_128399, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125413 = 0; i_125413 < (int64_t) 4; i_125413++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125367 = 0; i_125367 < (int64_t) 16; i_125367++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125363 = 0; i_125363 < (int64_t) 16; i_125363++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_109604;
                double r_109606 = 0.0;
                
                for (int64_t i_109605 = 0; i_109605 < (int64_t) 4; i_109605++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_109607 = ((double *) mem_126269)[i_125413 * (int64_t) 64 + i_125367 * (int64_t) 4 + i_109605];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_109608 = ((double *) mem_126268)[i_125413 * (int64_t) 64 + i_125363 * (int64_t) 4 + i_109605];
                    
                    // futhark/microgpt.fut:162:113-164
                    
                    double zt_res_109609 = zt_lhs_109607 * zt_rhs_109608;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_109610 = r_109606 + zt_res_109609;
                    double r_tmp_127986 = zp_res_109610;
                    
                    r_109606 = r_tmp_127986;
                }
                defunc_0_lifted_lambda_res_109604 = r_109606;
                ((double *) mem_126359)[i_125363] = defunc_0_lifted_lambda_res_109604;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126354, i_125367 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126359, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125375 = 0; i_125375 < (int64_t) 16; i_125375++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125371 = 0; i_125371 < (int64_t) 16; i_125371++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_109625 = ((double *) mem_126354)[i_125375 * (int64_t) 16 + i_125371];
                
                // futhark/microgpt.fut:163:47-78
                
                double zs_res_109626 = zs_lhs_109625 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_109627 = ((double *) mask_mem_126156.mem)[i_125375 * (int64_t) 16 + i_125371];
                
                // futhark/microgpt.fut:163:65-102
                
                double zp_res_109628 = zs_res_109626 + zp_rhs_109627;
                
                ((double *) mem_126375)[i_125371] = zp_res_109628;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126370, i_125375 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126375, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125393 = 0; i_125393 < (int64_t) 16; i_125393++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_118684;
            double redout_125377 = -INFINITY;
            
            for (int64_t i_125378 = 0; i_125378 < (int64_t) 16; i_125378++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_118633 = ((double *) mem_126370)[i_125393 * (int64_t) 16 + i_125378];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_109649 = fmax64(lifted_lambda_res_118633, redout_125377);
                double redout_tmp_127990 = max_res_109649;
                
                redout_125377 = redout_tmp_127990;
            }
            defunc_0_reduce_res_118684 = redout_125377;
            // futhark/microgpt.fut:165:67-76
            
            double neg_res_109650 = -defunc_0_reduce_res_118684;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125381 = 0; i_125381 < (int64_t) 16; i_125381++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_109657 = ((double *) mem_126370)[i_125393 * (int64_t) 16 + i_125381];
                
                // futhark/microgpt.fut:165:44-76
                
                double zp_res_109658 = neg_res_109650 + zp_lhs_109657;
                
                // futhark/microgpt.fut:165:37-76
                
                double exp_res_109659 = futrts_exp64(zp_res_109658);
                
                ((double *) mem_126391)[i_125381] = exp_res_109659;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_109661;
            double r_109663 = 0.0;
            
            for (int64_t i_109662 = 0; i_109662 < (int64_t) 16; i_109662++) {
                // futhark/microgpt.fut:166:36-46
                
                double lifted_lambda_res_109664 = ((double *) mem_126391)[i_109662];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_109665 = r_109663 + lifted_lambda_res_109664;
                double r_tmp_127992 = zp_res_109665;
                
                r_109663 = r_tmp_127992;
            }
            defunc_0_lifted_lambda_res_109661 = r_109663;
            // futhark/microgpt.fut:167:53-64
            
            double zs_res_109666 = 1.0 / defunc_0_lifted_lambda_res_109661;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125385 = 0; i_125385 < (int64_t) 16; i_125385++) {
                // futhark/microgpt.fut:167:37-47
                
                double zt_lhs_109673 = ((double *) mem_126391)[i_125385];
                
                // futhark/microgpt.fut:167:37-64
                
                double zt_res_109674 = zs_res_109666 * zt_lhs_109673;
                
                ((double *) mem_126398)[i_125385] = zt_res_109674;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125389 = 0; i_125389 < (int64_t) 16; i_125389++) {
                // futhark/microgpt.fut:168:4-14
                
                double lifted_lambda_res_109682 = ((double *) mem_126398)[i_125389];
                
                ((double *) mem_126405)[i_125389] = lifted_lambda_res_109682;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126386, i_125393 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126405, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125401 = 0; i_125401 < (int64_t) 16; i_125401++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125397 = 0; i_125397 < (int64_t) 4; i_125397++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_109697;
                double r_109699 = 0.0;
                
                for (int64_t i_109698 = 0; i_109698 < (int64_t) 16; i_109698++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_109700 = ((double *) mem_126386)[i_125401 * (int64_t) 16 + i_109698];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_109701 = ((double *) mem_126267)[i_125413 * (int64_t) 64 + i_109698 * (int64_t) 4 + i_125397];
                    
                    // futhark/microgpt.fut:169:66-111
                    
                    double zt_res_109702 = zt_lhs_109700 * zt_rhs_109701;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_109703 = r_109699 + zt_res_109702;
                    double r_tmp_127997 = zp_res_109703;
                    
                    r_109699 = r_tmp_127997;
                }
                defunc_0_lifted_lambda_res_109697 = r_109699;
                ((double *) mem_126421)[i_125397] = defunc_0_lifted_lambda_res_109697;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126416, i_125401 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126421, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125409 = 0; i_125409 < (int64_t) 16; i_125409++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125405 = 0; i_125405 < (int64_t) 4; i_125405++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_109718 = ((double *) mem_126416)[i_125409 * (int64_t) 4 + i_125405];
                
                ((double *) mem_126437)[i_125405] = lifted_lambda_res_109718;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126432, i_125409 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126437, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_126348, i_125413 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_126432, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126453_cached_sizze_128400 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126453, &mem_126453_cached_sizze_128400, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126458_cached_sizze_128401 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126458, &mem_126458_cached_sizze_128401, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125421 = 0; i_125421 < (int64_t) 16; i_125421++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125417 = 0; i_125417 < (int64_t) 16; i_125417++) {
            // futhark/microgpt.fut:171:54-57
            
            int64_t tmp_109730 = sdiv64(i_125417, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-59
            
            bool x_109731 = sle64((int64_t) 0, tmp_109730);
            
            // futhark/microgpt.fut:171:44-59
            
            bool y_109732 = slt64(tmp_109730, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-59
            
            bool bounds_check_109733 = x_109731 && y_109732;
            
            // futhark/microgpt.fut:171:44-59
            
            bool index_certs_109734;
            
            if (!bounds_check_109733) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_109730, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:171:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:171:15-80\n   #6  futhark/microgpt.fut:433:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:171:74-77
            
            int64_t tmp_109735 = smod64(i_125417, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-79
            
            bool x_109736 = sle64((int64_t) 0, tmp_109735);
            
            // futhark/microgpt.fut:171:44-79
            
            bool y_109737 = slt64(tmp_109735, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-79
            
            bool bounds_check_109738 = x_109736 && y_109737;
            
            // futhark/microgpt.fut:171:44-79
            
            bool index_certs_109739;
            
            if (!bounds_check_109738) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_109735, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:171:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:171:15-80\n   #6  futhark/microgpt.fut:433:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_109740 = ((double *) mem_126348)[tmp_109730 * (int64_t) 64 + i_125421 * (int64_t) 4 + tmp_109735];
            
            ((double *) mem_126458)[i_125417] = lifted_lambda_res_109740;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126453, i_125421 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126458, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126469_cached_sizze_128402 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126469, &mem_126469_cached_sizze_128402, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126474_cached_sizze_128403 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126474, &mem_126474_cached_sizze_128403, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125429 = 0; i_125429 < (int64_t) 16; i_125429++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125425 = 0; i_125425 < (int64_t) 16; i_125425++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_109755;
            double r_109757 = 0.0;
            
            for (int64_t i_109756 = 0; i_109756 < (int64_t) 16; i_109756++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_109758 = ((double *) wout_mem_126148.mem)[i_125425 * (int64_t) 16 + i_109756];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_109759 = ((double *) mem_126453)[i_125429 * (int64_t) 16 + i_109756];
                
                // futhark/microgpt.fut:172:67-106
                
                double zt_res_109760 = zt_lhs_109758 * zt_rhs_109759;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_109761 = r_109757 + zt_res_109760;
                double r_tmp_128004 = zp_res_109761;
                
                r_109757 = r_tmp_128004;
            }
            defunc_0_lifted_lambda_res_109755 = r_109757;
            ((double *) mem_126474)[i_125425] = defunc_0_lifted_lambda_res_109755;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126469, i_125429 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126474, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126485_cached_sizze_128404 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126485, &mem_126485_cached_sizze_128404, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126490_cached_sizze_128405 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126490, &mem_126490_cached_sizze_128405, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125437 = 0; i_125437 < (int64_t) 16; i_125437++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125433 = 0; i_125433 < (int64_t) 16; i_125433++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_109776 = ((double *) mem_126469)[i_125437 * (int64_t) 16 + i_125433];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_109777 = ((double *) mem_126173)[i_125437 * (int64_t) 16 + i_125433];
            
            // futhark/microgpt.fut:173:46-84
            
            double zp_res_109778 = zp_lhs_109776 + zp_rhs_109777;
            
            ((double *) mem_126490)[i_125433] = zp_res_109778;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126485, i_125437 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126490, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126501_cached_sizze_128406 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126501, &mem_126501_cached_sizze_128406, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126506_cached_sizze_128407 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126506, &mem_126506_cached_sizze_128407, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126513_cached_sizze_128408 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126513, &mem_126513_cached_sizze_128408, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125449 = 0; i_125449 < (int64_t) 16; i_125449++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_109787;
        double r_109789 = 0.0;
        
        for (int64_t i_109788 = 0; i_109788 < (int64_t) 16; i_109788++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_109790 = ((double *) mem_126485)[i_125449 * (int64_t) 16 + i_109788];
            
            // futhark/microgpt.fut:174:79-118
            
            double zt_res_109791 = zt_lhs_109790 * zt_lhs_109790;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_109792 = r_109789 + zt_res_109791;
            double r_tmp_128008 = zp_res_109792;
            
            r_109789 = r_tmp_128008;
        }
        defunc_0_lifted_lambda_res_109787 = r_109789;
        // futhark/microgpt.fut:174:58-136
        
        double zs_res_109793 = defunc_0_lifted_lambda_res_109787 / 16.0;
        
        // futhark/microgpt.fut:175:24-55
        
        double zp_res_109794 = 1.0e-5 + zs_res_109793;
        
        // futhark/microgpt.fut:175:16-55
        
        double sqrt_res_109795 = futrts_sqrt64(zp_res_109794);
        
        // futhark/microgpt.fut:176:60-71
        
        double zs_res_109796 = 1.0 / sqrt_res_109795;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125441 = 0; i_125441 < (int64_t) 16; i_125441++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_109803 = ((double *) mem_126485)[i_125449 * (int64_t) 16 + i_125441];
            
            // futhark/microgpt.fut:176:37-71
            
            double zt_res_109804 = zs_res_109796 * zt_lhs_109803;
            
            ((double *) mem_126506)[i_125441] = zt_res_109804;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125445 = 0; i_125445 < (int64_t) 16; i_125445++) {
            // futhark/microgpt.fut:177:4-14
            
            double lifted_lambda_res_109812 = ((double *) mem_126506)[i_125445];
            
            ((double *) mem_126513)[i_125445] = lifted_lambda_res_109812;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126501, i_125449 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126513, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126524_cached_sizze_128409 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_126524, &mem_126524_cached_sizze_128409, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126529_cached_sizze_128410 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126529, &mem_126529_cached_sizze_128410, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125457 = 0; i_125457 < (int64_t) 16; i_125457++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125453 = 0; i_125453 < (int64_t) 64; i_125453++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_109828;
            double r_109830 = 0.0;
            
            for (int64_t i_109829 = 0; i_109829 < (int64_t) 16; i_109829++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_109831 = ((double *) wup_mem_126152.mem)[i_125453 * (int64_t) 16 + i_109829];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_109832 = ((double *) mem_126501)[i_125457 * (int64_t) 16 + i_109829];
                
                // futhark/microgpt.fut:178:67-106
                
                double zt_res_109833 = zt_lhs_109831 * zt_rhs_109832;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_109834 = r_109830 + zt_res_109833;
                double r_tmp_128013 = zp_res_109834;
                
                r_109830 = r_tmp_128013;
            }
            defunc_0_lifted_lambda_res_109828 = r_109830;
            ((double *) mem_126529)[i_125453] = defunc_0_lifted_lambda_res_109828;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126524, i_125457 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126529, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126540_cached_sizze_128411 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_126540, &mem_126540_cached_sizze_128411, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126545_cached_sizze_128412 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126545, &mem_126545_cached_sizze_128412, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125465 = 0; i_125465 < (int64_t) 16; i_125465++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125461 = 0; i_125461 < (int64_t) 64; i_125461++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_109849 = ((double *) mem_126524)[i_125465 * (int64_t) 64 + i_125461];
            
            // futhark/microgpt.fut:179:45-73
            
            double max_res_109850 = fmax64(0.0, max_arg0_109849);
            
            ((double *) mem_126545)[i_125461] = max_res_109850;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126540, i_125465 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126545, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126556_cached_sizze_128413 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126556, &mem_126556_cached_sizze_128413, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126561_cached_sizze_128414 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126561, &mem_126561_cached_sizze_128414, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125473 = 0; i_125473 < (int64_t) 16; i_125473++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125469 = 0; i_125469 < (int64_t) 16; i_125469++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_109865;
            double r_109867 = 0.0;
            
            for (int64_t i_109866 = 0; i_109866 < (int64_t) 64; i_109866++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_109868 = ((double *) wdown_mem_126146.mem)[i_125469 * (int64_t) 64 + i_109866];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_109869 = ((double *) mem_126540)[i_125473 * (int64_t) 64 + i_109866];
                
                // futhark/microgpt.fut:180:67-108
                
                double zt_res_109870 = zt_lhs_109868 * zt_rhs_109869;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_109871 = r_109867 + zt_res_109870;
                double r_tmp_128018 = zp_res_109871;
                
                r_109867 = r_tmp_128018;
            }
            defunc_0_lifted_lambda_res_109865 = r_109867;
            ((double *) mem_126561)[i_125469] = defunc_0_lifted_lambda_res_109865;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126556, i_125473 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126561, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126572_cached_sizze_128415 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126572, &mem_126572_cached_sizze_128415, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126577_cached_sizze_128416 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126577, &mem_126577_cached_sizze_128416, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125481 = 0; i_125481 < (int64_t) 16; i_125481++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125477 = 0; i_125477 < (int64_t) 16; i_125477++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_109886 = ((double *) mem_126556)[i_125481 * (int64_t) 16 + i_125477];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_109887 = ((double *) mem_126485)[i_125481 * (int64_t) 16 + i_125477];
            
            // futhark/microgpt.fut:181:46-85
            
            double zp_res_109888 = zp_lhs_109886 + zp_rhs_109887;
            
            ((double *) mem_126577)[i_125477] = zp_res_109888;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126572, i_125481 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126577, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126588_cached_sizze_128417 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_126588, &mem_126588_cached_sizze_128417, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126593_cached_sizze_128418 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_126593, &mem_126593_cached_sizze_128418, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125489 = 0; i_125489 < (int64_t) 16; i_125489++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125485 = 0; i_125485 < (int64_t) 27; i_125485++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_109904;
            double r_109906 = 0.0;
            
            for (int64_t i_109905 = 0; i_109905 < (int64_t) 16; i_109905++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_109907 = ((double *) wvoc_mem_126154.mem)[i_125485 * (int64_t) 16 + i_109905];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_109908 = ((double *) mem_126572)[i_125489 * (int64_t) 16 + i_109905];
                
                // futhark/microgpt.fut:182:67-107
                
                double zt_res_109909 = zt_lhs_109907 * zt_rhs_109908;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_109910 = r_109906 + zt_res_109909;
                double r_tmp_128023 = zp_res_109910;
                
                r_109906 = r_tmp_128023;
            }
            defunc_0_lifted_lambda_res_109904 = r_109906;
            ((double *) mem_126593)[i_125485] = defunc_0_lifted_lambda_res_109904;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126588, i_125489 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126593, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_126604, (int64_t) 3456, "mem_126604")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126609_cached_sizze_128419 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_126609, &mem_126609_cached_sizze_128419, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_125497 = 0; i_125497 < (int64_t) 16; i_125497++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125493 = 0; i_125493 < (int64_t) 27; i_125493++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_109925 = ((double *) mem_126588)[i_125497 * (int64_t) 27 + i_125493];
            
            ((double *) mem_126609)[i_125493] = lifted_lambda_res_109925;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_126604.mem, i_125497 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126609, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_127954, &mem_126604, "mem_126604") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128363, &mem_out_127954, "mem_out_127954") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_126157);
        free(mem_126162);
        free(mem_126173);
        free(mem_126178);
        free(mem_126185);
        free(mem_126196);
        free(mem_126201);
        free(mem_126208);
        free(mem_126219);
        free(mem_126220);
        free(mem_126221);
        free(mem_126234);
        free(mem_126235);
        free(mem_126236);
        free(mem_126267);
        free(mem_126268);
        free(mem_126269);
        free(mem_126285);
        free(mem_126286);
        free(mem_126287);
        free(mem_126300);
        free(mem_126301);
        free(mem_126302);
        free(mem_126348);
        free(mem_126354);
        free(mem_126359);
        free(mem_126370);
        free(mem_126375);
        free(mem_126386);
        free(mem_126391);
        free(mem_126398);
        free(mem_126405);
        free(mem_126416);
        free(mem_126421);
        free(mem_126432);
        free(mem_126437);
        free(mem_126453);
        free(mem_126458);
        free(mem_126469);
        free(mem_126474);
        free(mem_126485);
        free(mem_126490);
        free(mem_126501);
        free(mem_126506);
        free(mem_126513);
        free(mem_126524);
        free(mem_126529);
        free(mem_126540);
        free(mem_126545);
        free(mem_126556);
        free(mem_126561);
        free(mem_126572);
        free(mem_126577);
        free(mem_126588);
        free(mem_126593);
        free(mem_126609);
        if (memblock_unref(ctx, &mem_126604, "mem_126604") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127954, "mem_out_127954") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_128420, struct memblock *mem_out_p_128421, struct memblock *mem_out_p_128422, struct memblock *mem_out_p_128423, struct memblock *mem_out_p_128424, struct memblock *mem_out_p_128425, struct memblock *mem_out_p_128426, struct memblock *mem_out_p_128427, struct memblock *mem_out_p_128428, struct memblock wte_mem_126146, struct memblock wpe_mem_126147, struct memblock wqry_mem_126148, struct memblock wkey_mem_126149, struct memblock wval_mem_126150, struct memblock wout_mem_126151, struct memblock wup_mem_126152, struct memblock wdown_mem_126153, struct memblock wvoc_mem_126154)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_127962;
    
    mem_out_127962.references = NULL;
    
    struct memblock mem_out_127961;
    
    mem_out_127961.references = NULL;
    
    struct memblock mem_out_127960;
    
    mem_out_127960.references = NULL;
    
    struct memblock mem_out_127959;
    
    mem_out_127959.references = NULL;
    
    struct memblock mem_out_127958;
    
    mem_out_127958.references = NULL;
    
    struct memblock mem_out_127957;
    
    mem_out_127957.references = NULL;
    
    struct memblock mem_out_127956;
    
    mem_out_127956.references = NULL;
    
    struct memblock mem_out_127955;
    
    mem_out_127955.references = NULL;
    
    struct memblock mem_out_127954;
    
    mem_out_127954.references = NULL;
    
    struct memblock mem_126137 = ctx->constants->mem_126137;
    struct memblock mem_126138 = ctx->constants->mem_126138;
    struct memblock mem_126139 = ctx->constants->mem_126139;
    struct memblock mem_126140 = ctx->constants->mem_126140;
    struct memblock mem_126141 = ctx->constants->mem_126141;
    struct memblock mem_126142 = ctx->constants->mem_126142;
    struct memblock mem_126143 = ctx->constants->mem_126143;
    struct memblock mem_126144 = ctx->constants->mem_126144;
    struct memblock mem_126145 = ctx->constants->mem_126145;
    
    if (memblock_set(ctx, &mem_out_127954, &wdown_mem_126153, "wdown_mem_126153") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127955, &wkey_mem_126149, "wkey_mem_126149") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127956, &wout_mem_126151, "wout_mem_126151") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127957, &wpe_mem_126147, "wpe_mem_126147") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127958, &wqry_mem_126148, "wqry_mem_126148") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127959, &wte_mem_126146, "wte_mem_126146") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127960, &wup_mem_126152, "wup_mem_126152") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127961, &wval_mem_126150, "wval_mem_126150") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127962, &wvoc_mem_126154, "wvoc_mem_126154") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128420, &mem_out_127954, "mem_out_127954") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128421, &mem_out_127955, "mem_out_127955") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128422, &mem_out_127956, "mem_out_127956") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128423, &mem_out_127957, "mem_out_127957") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128424, &mem_out_127958, "mem_out_127958") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128425, &mem_out_127959, "mem_out_127959") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128426, &mem_out_127960, "mem_out_127960") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128427, &mem_out_127961, "mem_out_127961") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128428, &mem_out_127962, "mem_out_127962") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_127962, "mem_out_127962") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127961, "mem_out_127961") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127960, "mem_out_127960") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127959, "mem_out_127959") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127958, "mem_out_127958") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127957, "mem_out_127957") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127956, "mem_out_127956") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127955, "mem_out_127955") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127954, "mem_out_127954") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_128429, struct memblock *mem_out_p_128430, struct memblock *mem_out_p_128431, struct memblock *mem_out_p_128432, struct memblock *mem_out_p_128433, struct memblock *mem_out_p_128434, struct memblock *mem_out_p_128435, struct memblock *mem_out_p_128436, struct memblock *mem_out_p_128437, struct memblock *mem_out_p_128438, struct memblock *mem_out_p_128439, struct memblock *mem_out_p_128440, struct memblock *mem_out_p_128441, struct memblock *mem_out_p_128442, struct memblock *mem_out_p_128443, struct memblock *mem_out_p_128444, struct memblock *mem_out_p_128445, struct memblock *mem_out_p_128446, struct memblock *mem_out_p_128447, struct memblock *mem_out_p_128448, struct memblock *mem_out_p_128449, struct memblock *mem_out_p_128450, struct memblock *mem_out_p_128451, struct memblock *mem_out_p_128452, struct memblock *mem_out_p_128453, struct memblock *mem_out_p_128454, struct memblock *mem_out_p_128455, struct memblock wdown_mem_126146, struct memblock wkey_mem_126147, struct memblock wout_mem_126148, struct memblock wpe_mem_126149, struct memblock wqry_mem_126150, struct memblock wte_mem_126151, struct memblock wup_mem_126152, struct memblock wval_mem_126153, struct memblock wvoc_mem_126154, struct memblock wdown_mem_126155, struct memblock wkey_mem_126156, struct memblock wout_mem_126157, struct memblock wpe_mem_126158, struct memblock wqry_mem_126159, struct memblock wte_mem_126160, struct memblock wup_mem_126161, struct memblock wval_mem_126162, struct memblock wvoc_mem_126163, struct memblock wdown_mem_126164, struct memblock wkey_mem_126165, struct memblock wout_mem_126166, struct memblock wpe_mem_126167, struct memblock wqry_mem_126168, struct memblock wte_mem_126169, struct memblock wup_mem_126170, struct memblock wval_mem_126171, struct memblock wvoc_mem_126172, struct memblock masks_mem_126173, struct memblock dls_mem_126174, struct memblock seqs_mem_126175)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_126284_cached_sizze_128456 = 0;
    unsigned char *mem_126284 = NULL;
    int64_t mem_126285_cached_sizze_128457 = 0;
    unsigned char *mem_126285 = NULL;
    int64_t mem_126294_cached_sizze_128458 = 0;
    unsigned char *mem_126294 = NULL;
    int64_t mem_126301_cached_sizze_128459 = 0;
    unsigned char *mem_126301 = NULL;
    int64_t mem_126316_cached_sizze_128460 = 0;
    unsigned char *mem_126316 = NULL;
    int64_t mem_126317_cached_sizze_128461 = 0;
    unsigned char *mem_126317 = NULL;
    int64_t mem_126318_cached_sizze_128462 = 0;
    unsigned char *mem_126318 = NULL;
    int64_t mem_126329_cached_sizze_128463 = 0;
    unsigned char *mem_126329 = NULL;
    int64_t mem_126346_cached_sizze_128464 = 0;
    unsigned char *mem_126346 = NULL;
    int64_t mem_126347_cached_sizze_128465 = 0;
    unsigned char *mem_126347 = NULL;
    int64_t mem_126355_cached_sizze_128466 = 0;
    unsigned char *mem_126355 = NULL;
    int64_t mem_126369_cached_sizze_128467 = 0;
    unsigned char *mem_126369 = NULL;
    int64_t mem_126370_cached_sizze_128468 = 0;
    unsigned char *mem_126370 = NULL;
    int64_t mem_126371_cached_sizze_128469 = 0;
    unsigned char *mem_126371 = NULL;
    int64_t mem_126384_cached_sizze_128470 = 0;
    unsigned char *mem_126384 = NULL;
    int64_t mem_126385_cached_sizze_128471 = 0;
    unsigned char *mem_126385 = NULL;
    int64_t mem_126386_cached_sizze_128472 = 0;
    unsigned char *mem_126386 = NULL;
    int64_t mem_126417_cached_sizze_128473 = 0;
    unsigned char *mem_126417 = NULL;
    int64_t mem_126418_cached_sizze_128474 = 0;
    unsigned char *mem_126418 = NULL;
    int64_t mem_126419_cached_sizze_128475 = 0;
    unsigned char *mem_126419 = NULL;
    int64_t mem_126435_cached_sizze_128476 = 0;
    unsigned char *mem_126435 = NULL;
    int64_t mem_126436_cached_sizze_128477 = 0;
    unsigned char *mem_126436 = NULL;
    int64_t mem_126437_cached_sizze_128478 = 0;
    unsigned char *mem_126437 = NULL;
    int64_t mem_126450_cached_sizze_128479 = 0;
    unsigned char *mem_126450 = NULL;
    int64_t mem_126451_cached_sizze_128480 = 0;
    unsigned char *mem_126451 = NULL;
    int64_t mem_126452_cached_sizze_128481 = 0;
    unsigned char *mem_126452 = NULL;
    int64_t mem_126498_cached_sizze_128482 = 0;
    unsigned char *mem_126498 = NULL;
    int64_t mem_126499_cached_sizze_128483 = 0;
    unsigned char *mem_126499 = NULL;
    int64_t mem_126500_cached_sizze_128484 = 0;
    unsigned char *mem_126500 = NULL;
    int64_t mem_126501_cached_sizze_128485 = 0;
    unsigned char *mem_126501 = NULL;
    int64_t mem_126522_cached_sizze_128486 = 0;
    unsigned char *mem_126522 = NULL;
    int64_t mem_126523_cached_sizze_128487 = 0;
    unsigned char *mem_126523 = NULL;
    int64_t mem_126524_cached_sizze_128488 = 0;
    unsigned char *mem_126524 = NULL;
    int64_t mem_126525_cached_sizze_128489 = 0;
    unsigned char *mem_126525 = NULL;
    int64_t mem_126542_cached_sizze_128490 = 0;
    unsigned char *mem_126542 = NULL;
    int64_t mem_126543_cached_sizze_128491 = 0;
    unsigned char *mem_126543 = NULL;
    int64_t mem_126544_cached_sizze_128492 = 0;
    unsigned char *mem_126544 = NULL;
    int64_t mem_126545_cached_sizze_128493 = 0;
    unsigned char *mem_126545 = NULL;
    int64_t mem_126586_cached_sizze_128494 = 0;
    unsigned char *mem_126586 = NULL;
    int64_t mem_126591_cached_sizze_128495 = 0;
    unsigned char *mem_126591 = NULL;
    int64_t mem_126595_cached_sizze_128496 = 0;
    unsigned char *mem_126595 = NULL;
    int64_t mem_126629_cached_sizze_128497 = 0;
    unsigned char *mem_126629 = NULL;
    int64_t mem_126634_cached_sizze_128498 = 0;
    unsigned char *mem_126634 = NULL;
    int64_t mem_126645_cached_sizze_128499 = 0;
    unsigned char *mem_126645 = NULL;
    int64_t mem_126650_cached_sizze_128500 = 0;
    unsigned char *mem_126650 = NULL;
    int64_t mem_126661_cached_sizze_128501 = 0;
    unsigned char *mem_126661 = NULL;
    int64_t mem_126666_cached_sizze_128502 = 0;
    unsigned char *mem_126666 = NULL;
    int64_t mem_126677_cached_sizze_128503 = 0;
    unsigned char *mem_126677 = NULL;
    int64_t mem_126678_cached_sizze_128504 = 0;
    unsigned char *mem_126678 = NULL;
    int64_t mem_126686_cached_sizze_128505 = 0;
    unsigned char *mem_126686 = NULL;
    int64_t mem_126700_cached_sizze_128506 = 0;
    unsigned char *mem_126700 = NULL;
    int64_t mem_126705_cached_sizze_128507 = 0;
    unsigned char *mem_126705 = NULL;
    int64_t mem_126716_cached_sizze_128508 = 0;
    unsigned char *mem_126716 = NULL;
    int64_t mem_126721_cached_sizze_128509 = 0;
    unsigned char *mem_126721 = NULL;
    int64_t mem_126732_cached_sizze_128510 = 0;
    unsigned char *mem_126732 = NULL;
    int64_t mem_126737_cached_sizze_128511 = 0;
    unsigned char *mem_126737 = NULL;
    int64_t mem_126748_cached_sizze_128512 = 0;
    unsigned char *mem_126748 = NULL;
    int64_t mem_126753_cached_sizze_128513 = 0;
    unsigned char *mem_126753 = NULL;
    int64_t mem_126764_cached_sizze_128514 = 0;
    unsigned char *mem_126764 = NULL;
    int64_t mem_126769_cached_sizze_128515 = 0;
    unsigned char *mem_126769 = NULL;
    int64_t mem_126780_cached_sizze_128516 = 0;
    unsigned char *mem_126780 = NULL;
    int64_t mem_126781_cached_sizze_128517 = 0;
    unsigned char *mem_126781 = NULL;
    int64_t mem_126782_cached_sizze_128518 = 0;
    unsigned char *mem_126782 = NULL;
    int64_t mem_126796_cached_sizze_128519 = 0;
    unsigned char *mem_126796 = NULL;
    int64_t mem_126801_cached_sizze_128520 = 0;
    unsigned char *mem_126801 = NULL;
    int64_t mem_126805_cached_sizze_128521 = 0;
    unsigned char *mem_126805 = NULL;
    int64_t mem_126834_cached_sizze_128522 = 0;
    unsigned char *mem_126834 = NULL;
    int64_t mem_126840_cached_sizze_128523 = 0;
    unsigned char *mem_126840 = NULL;
    int64_t mem_126845_cached_sizze_128524 = 0;
    unsigned char *mem_126845 = NULL;
    int64_t mem_126861_cached_sizze_128525 = 0;
    unsigned char *mem_126861 = NULL;
    int64_t mem_126866_cached_sizze_128526 = 0;
    unsigned char *mem_126866 = NULL;
    int64_t mem_126877_cached_sizze_128527 = 0;
    unsigned char *mem_126877 = NULL;
    int64_t mem_126883_cached_sizze_128528 = 0;
    unsigned char *mem_126883 = NULL;
    int64_t mem_126888_cached_sizze_128529 = 0;
    unsigned char *mem_126888 = NULL;
    int64_t mem_126904_cached_sizze_128530 = 0;
    unsigned char *mem_126904 = NULL;
    int64_t mem_126909_cached_sizze_128531 = 0;
    unsigned char *mem_126909 = NULL;
    int64_t mem_126920_cached_sizze_128532 = 0;
    unsigned char *mem_126920 = NULL;
    int64_t mem_126925_cached_sizze_128533 = 0;
    unsigned char *mem_126925 = NULL;
    int64_t mem_126936_cached_sizze_128534 = 0;
    unsigned char *mem_126936 = NULL;
    int64_t mem_126941_cached_sizze_128535 = 0;
    unsigned char *mem_126941 = NULL;
    int64_t mem_126952_cached_sizze_128536 = 0;
    unsigned char *mem_126952 = NULL;
    int64_t mem_126953_cached_sizze_128537 = 0;
    unsigned char *mem_126953 = NULL;
    int64_t mem_126962_cached_sizze_128538 = 0;
    unsigned char *mem_126962 = NULL;
    int64_t mem_126963_cached_sizze_128539 = 0;
    unsigned char *mem_126963 = NULL;
    int64_t mem_126984_cached_sizze_128540 = 0;
    unsigned char *mem_126984 = NULL;
    int64_t mem_126989_cached_sizze_128541 = 0;
    unsigned char *mem_126989 = NULL;
    int64_t mem_127000_cached_sizze_128542 = 0;
    unsigned char *mem_127000 = NULL;
    int64_t mem_127005_cached_sizze_128543 = 0;
    unsigned char *mem_127005 = NULL;
    int64_t mem_127016_cached_sizze_128544 = 0;
    unsigned char *mem_127016 = NULL;
    int64_t mem_127021_cached_sizze_128545 = 0;
    unsigned char *mem_127021 = NULL;
    int64_t mem_127032_cached_sizze_128546 = 0;
    unsigned char *mem_127032 = NULL;
    int64_t mem_127033_cached_sizze_128547 = 0;
    unsigned char *mem_127033 = NULL;
    int64_t mem_127046_cached_sizze_128548 = 0;
    unsigned char *mem_127046 = NULL;
    int64_t mem_127051_cached_sizze_128549 = 0;
    unsigned char *mem_127051 = NULL;
    int64_t mem_127062_cached_sizze_128550 = 0;
    unsigned char *mem_127062 = NULL;
    int64_t mem_127067_cached_sizze_128551 = 0;
    unsigned char *mem_127067 = NULL;
    int64_t mem_127078_cached_sizze_128552 = 0;
    unsigned char *mem_127078 = NULL;
    int64_t mem_127079_cached_sizze_128553 = 0;
    unsigned char *mem_127079 = NULL;
    int64_t mem_127088_cached_sizze_128554 = 0;
    unsigned char *mem_127088 = NULL;
    int64_t mem_127089_cached_sizze_128555 = 0;
    unsigned char *mem_127089 = NULL;
    int64_t mem_127110_cached_sizze_128556 = 0;
    unsigned char *mem_127110 = NULL;
    int64_t mem_127116_cached_sizze_128557 = 0;
    unsigned char *mem_127116 = NULL;
    int64_t mem_127121_cached_sizze_128558 = 0;
    unsigned char *mem_127121 = NULL;
    int64_t mem_127137_cached_sizze_128559 = 0;
    unsigned char *mem_127137 = NULL;
    int64_t mem_127138_cached_sizze_128560 = 0;
    unsigned char *mem_127138 = NULL;
    int64_t mem_127139_cached_sizze_128561 = 0;
    unsigned char *mem_127139 = NULL;
    int64_t mem_127155_cached_sizze_128562 = 0;
    unsigned char *mem_127155 = NULL;
    int64_t mem_127156_cached_sizze_128563 = 0;
    unsigned char *mem_127156 = NULL;
    int64_t mem_127157_cached_sizze_128564 = 0;
    unsigned char *mem_127157 = NULL;
    int64_t mem_127170_cached_sizze_128565 = 0;
    unsigned char *mem_127170 = NULL;
    int64_t mem_127171_cached_sizze_128566 = 0;
    unsigned char *mem_127171 = NULL;
    int64_t mem_127172_cached_sizze_128567 = 0;
    unsigned char *mem_127172 = NULL;
    int64_t mem_127182_cached_sizze_128568 = 0;
    unsigned char *mem_127182 = NULL;
    int64_t mem_127189_cached_sizze_128569 = 0;
    unsigned char *mem_127189 = NULL;
    int64_t mem_127190_cached_sizze_128570 = 0;
    unsigned char *mem_127190 = NULL;
    int64_t mem_127191_cached_sizze_128571 = 0;
    unsigned char *mem_127191 = NULL;
    int64_t mem_127202_cached_sizze_128572 = 0;
    unsigned char *mem_127202 = NULL;
    int64_t mem_127219_cached_sizze_128573 = 0;
    unsigned char *mem_127219 = NULL;
    int64_t mem_127224_cached_sizze_128574 = 0;
    unsigned char *mem_127224 = NULL;
    int64_t mem_127235_cached_sizze_128575 = 0;
    unsigned char *mem_127235 = NULL;
    int64_t mem_127242_cached_sizze_128576 = 0;
    unsigned char *mem_127242 = NULL;
    int64_t mem_127247_cached_sizze_128577 = 0;
    unsigned char *mem_127247 = NULL;
    int64_t mem_127258_cached_sizze_128578 = 0;
    unsigned char *mem_127258 = NULL;
    int64_t mem_127259_cached_sizze_128579 = 0;
    unsigned char *mem_127259 = NULL;
    int64_t mem_127260_cached_sizze_128580 = 0;
    unsigned char *mem_127260 = NULL;
    int64_t mem_127271_cached_sizze_128581 = 0;
    unsigned char *mem_127271 = NULL;
    int64_t mem_127288_cached_sizze_128582 = 0;
    unsigned char *mem_127288 = NULL;
    int64_t mem_127293_cached_sizze_128583 = 0;
    unsigned char *mem_127293 = NULL;
    int64_t mem_127304_cached_sizze_128584 = 0;
    unsigned char *mem_127304 = NULL;
    int64_t mem_127311_cached_sizze_128585 = 0;
    unsigned char *mem_127311 = NULL;
    int64_t mem_127316_cached_sizze_128586 = 0;
    unsigned char *mem_127316 = NULL;
    int64_t mem_127363_cached_sizze_128587 = 0;
    unsigned char *mem_127363 = NULL;
    int64_t mem_127364_cached_sizze_128588 = 0;
    unsigned char *mem_127364 = NULL;
    int64_t mem_127365_cached_sizze_128589 = 0;
    unsigned char *mem_127365 = NULL;
    int64_t mem_127378_cached_sizze_128590 = 0;
    unsigned char *mem_127378 = NULL;
    int64_t mem_127379_cached_sizze_128591 = 0;
    unsigned char *mem_127379 = NULL;
    int64_t mem_127380_cached_sizze_128592 = 0;
    unsigned char *mem_127380 = NULL;
    int64_t mem_127411_cached_sizze_128593 = 0;
    unsigned char *mem_127411 = NULL;
    int64_t mem_127412_cached_sizze_128594 = 0;
    unsigned char *mem_127412 = NULL;
    int64_t mem_127413_cached_sizze_128595 = 0;
    unsigned char *mem_127413 = NULL;
    int64_t mem_127414_cached_sizze_128596 = 0;
    unsigned char *mem_127414 = NULL;
    int64_t mem_127431_cached_sizze_128597 = 0;
    unsigned char *mem_127431 = NULL;
    int64_t mem_127432_cached_sizze_128598 = 0;
    unsigned char *mem_127432 = NULL;
    int64_t mem_127433_cached_sizze_128599 = 0;
    unsigned char *mem_127433 = NULL;
    int64_t mem_127434_cached_sizze_128600 = 0;
    unsigned char *mem_127434 = NULL;
    int64_t mem_127475_cached_sizze_128601 = 0;
    unsigned char *mem_127475 = NULL;
    int64_t mem_127480_cached_sizze_128602 = 0;
    unsigned char *mem_127480 = NULL;
    int64_t mem_127491_cached_sizze_128603 = 0;
    unsigned char *mem_127491 = NULL;
    int64_t mem_127492_cached_sizze_128604 = 0;
    unsigned char *mem_127492 = NULL;
    int64_t mem_127505_cached_sizze_128605 = 0;
    unsigned char *mem_127505 = NULL;
    int64_t mem_127510_cached_sizze_128606 = 0;
    unsigned char *mem_127510 = NULL;
    int64_t mem_127521_cached_sizze_128607 = 0;
    unsigned char *mem_127521 = NULL;
    int64_t mem_127522_cached_sizze_128608 = 0;
    unsigned char *mem_127522 = NULL;
    int64_t mem_127531_cached_sizze_128609 = 0;
    unsigned char *mem_127531 = NULL;
    int64_t mem_127532_cached_sizze_128610 = 0;
    unsigned char *mem_127532 = NULL;
    int64_t mem_127553_cached_sizze_128611 = 0;
    unsigned char *mem_127553 = NULL;
    int64_t mem_127554_cached_sizze_128612 = 0;
    unsigned char *mem_127554 = NULL;
    int64_t mem_127555_cached_sizze_128613 = 0;
    unsigned char *mem_127555 = NULL;
    int64_t mem_127556_cached_sizze_128614 = 0;
    unsigned char *mem_127556 = NULL;
    int64_t mem_127581_cached_sizze_128615 = 0;
    unsigned char *mem_127581 = NULL;
    int64_t mem_127582_cached_sizze_128616 = 0;
    unsigned char *mem_127582 = NULL;
    int64_t mem_127595_cached_sizze_128617 = 0;
    unsigned char *mem_127595 = NULL;
    int64_t mem_127596_cached_sizze_128618 = 0;
    unsigned char *mem_127596 = NULL;
    int64_t mem_127605_cached_sizze_128619 = 0;
    unsigned char *mem_127605 = NULL;
    int64_t mem_127606_cached_sizze_128620 = 0;
    unsigned char *mem_127606 = NULL;
    int64_t mem_127627_cached_sizze_128621 = 0;
    unsigned char *mem_127627 = NULL;
    int64_t mem_127632_cached_sizze_128622 = 0;
    unsigned char *mem_127632 = NULL;
    int64_t mem_127643_cached_sizze_128623 = 0;
    unsigned char *mem_127643 = NULL;
    int64_t mem_127644_cached_sizze_128624 = 0;
    unsigned char *mem_127644 = NULL;
    int64_t mem_127653_cached_sizze_128625 = 0;
    unsigned char *mem_127653 = NULL;
    int64_t mem_127654_cached_sizze_128626 = 0;
    unsigned char *mem_127654 = NULL;
    struct memblock mem_param_tmp_128007;
    
    mem_param_tmp_128007.references = NULL;
    
    struct memblock mem_param_tmp_128006;
    
    mem_param_tmp_128006.references = NULL;
    
    struct memblock mem_param_tmp_128005;
    
    mem_param_tmp_128005.references = NULL;
    
    struct memblock mem_param_tmp_128004;
    
    mem_param_tmp_128004.references = NULL;
    
    struct memblock mem_param_tmp_128003;
    
    mem_param_tmp_128003.references = NULL;
    
    struct memblock mem_param_tmp_128002;
    
    mem_param_tmp_128002.references = NULL;
    
    struct memblock mem_param_tmp_128001;
    
    mem_param_tmp_128001.references = NULL;
    
    struct memblock mem_param_tmp_128000;
    
    mem_param_tmp_128000.references = NULL;
    
    struct memblock mem_param_tmp_127999;
    
    mem_param_tmp_127999.references = NULL;
    
    struct memblock mem_param_tmp_127998;
    
    mem_param_tmp_127998.references = NULL;
    
    struct memblock mem_param_tmp_127997;
    
    mem_param_tmp_127997.references = NULL;
    
    struct memblock mem_param_tmp_127996;
    
    mem_param_tmp_127996.references = NULL;
    
    struct memblock mem_param_tmp_127995;
    
    mem_param_tmp_127995.references = NULL;
    
    struct memblock mem_param_tmp_127994;
    
    mem_param_tmp_127994.references = NULL;
    
    struct memblock mem_param_tmp_127993;
    
    mem_param_tmp_127993.references = NULL;
    
    struct memblock mem_param_tmp_127992;
    
    mem_param_tmp_127992.references = NULL;
    
    struct memblock mem_param_tmp_127991;
    
    mem_param_tmp_127991.references = NULL;
    
    struct memblock mem_param_tmp_127990;
    
    mem_param_tmp_127990.references = NULL;
    
    struct memblock mem_param_tmp_127989;
    
    mem_param_tmp_127989.references = NULL;
    
    struct memblock mem_param_tmp_127988;
    
    mem_param_tmp_127988.references = NULL;
    
    struct memblock mem_param_tmp_127987;
    
    mem_param_tmp_127987.references = NULL;
    
    struct memblock mem_param_tmp_127986;
    
    mem_param_tmp_127986.references = NULL;
    
    struct memblock mem_param_tmp_127985;
    
    mem_param_tmp_127985.references = NULL;
    
    struct memblock mem_param_tmp_127984;
    
    mem_param_tmp_127984.references = NULL;
    
    struct memblock mem_param_tmp_127983;
    
    mem_param_tmp_127983.references = NULL;
    
    struct memblock mem_param_tmp_127982;
    
    mem_param_tmp_127982.references = NULL;
    
    struct memblock mem_param_tmp_127981;
    
    mem_param_tmp_127981.references = NULL;
    
    struct memblock ext_mem_127771;
    
    ext_mem_127771.references = NULL;
    
    struct memblock ext_mem_127772;
    
    ext_mem_127772.references = NULL;
    
    struct memblock ext_mem_127773;
    
    ext_mem_127773.references = NULL;
    
    struct memblock mem_127769;
    
    mem_127769.references = NULL;
    
    struct memblock mem_127767;
    
    mem_127767.references = NULL;
    
    struct memblock mem_127765;
    
    mem_127765.references = NULL;
    
    struct memblock mem_127763;
    
    mem_127763.references = NULL;
    
    struct memblock ext_mem_127760;
    
    ext_mem_127760.references = NULL;
    
    struct memblock ext_mem_127761;
    
    ext_mem_127761.references = NULL;
    
    struct memblock ext_mem_127762;
    
    ext_mem_127762.references = NULL;
    
    struct memblock mem_127758;
    
    mem_127758.references = NULL;
    
    struct memblock mem_127756;
    
    mem_127756.references = NULL;
    
    struct memblock mem_127754;
    
    mem_127754.references = NULL;
    
    struct memblock mem_127752;
    
    mem_127752.references = NULL;
    
    struct memblock ext_mem_127749;
    
    ext_mem_127749.references = NULL;
    
    struct memblock ext_mem_127750;
    
    ext_mem_127750.references = NULL;
    
    struct memblock ext_mem_127751;
    
    ext_mem_127751.references = NULL;
    
    struct memblock mem_127747;
    
    mem_127747.references = NULL;
    
    struct memblock mem_127745;
    
    mem_127745.references = NULL;
    
    struct memblock mem_127743;
    
    mem_127743.references = NULL;
    
    struct memblock mem_127741;
    
    mem_127741.references = NULL;
    
    struct memblock ext_mem_127738;
    
    ext_mem_127738.references = NULL;
    
    struct memblock ext_mem_127739;
    
    ext_mem_127739.references = NULL;
    
    struct memblock ext_mem_127740;
    
    ext_mem_127740.references = NULL;
    
    struct memblock mem_127736;
    
    mem_127736.references = NULL;
    
    struct memblock mem_127734;
    
    mem_127734.references = NULL;
    
    struct memblock mem_127732;
    
    mem_127732.references = NULL;
    
    struct memblock mem_127730;
    
    mem_127730.references = NULL;
    
    struct memblock ext_mem_127727;
    
    ext_mem_127727.references = NULL;
    
    struct memblock ext_mem_127728;
    
    ext_mem_127728.references = NULL;
    
    struct memblock ext_mem_127729;
    
    ext_mem_127729.references = NULL;
    
    struct memblock mem_127725;
    
    mem_127725.references = NULL;
    
    struct memblock mem_127723;
    
    mem_127723.references = NULL;
    
    struct memblock mem_127721;
    
    mem_127721.references = NULL;
    
    struct memblock mem_127719;
    
    mem_127719.references = NULL;
    
    struct memblock ext_mem_127716;
    
    ext_mem_127716.references = NULL;
    
    struct memblock ext_mem_127717;
    
    ext_mem_127717.references = NULL;
    
    struct memblock ext_mem_127718;
    
    ext_mem_127718.references = NULL;
    
    struct memblock mem_127714;
    
    mem_127714.references = NULL;
    
    struct memblock mem_127712;
    
    mem_127712.references = NULL;
    
    struct memblock mem_127710;
    
    mem_127710.references = NULL;
    
    struct memblock mem_127708;
    
    mem_127708.references = NULL;
    
    struct memblock ext_mem_127705;
    
    ext_mem_127705.references = NULL;
    
    struct memblock ext_mem_127706;
    
    ext_mem_127706.references = NULL;
    
    struct memblock ext_mem_127707;
    
    ext_mem_127707.references = NULL;
    
    struct memblock mem_127703;
    
    mem_127703.references = NULL;
    
    struct memblock mem_127701;
    
    mem_127701.references = NULL;
    
    struct memblock mem_127699;
    
    mem_127699.references = NULL;
    
    struct memblock mem_127697;
    
    mem_127697.references = NULL;
    
    struct memblock ext_mem_127694;
    
    ext_mem_127694.references = NULL;
    
    struct memblock ext_mem_127695;
    
    ext_mem_127695.references = NULL;
    
    struct memblock ext_mem_127696;
    
    ext_mem_127696.references = NULL;
    
    struct memblock mem_127692;
    
    mem_127692.references = NULL;
    
    struct memblock mem_127690;
    
    mem_127690.references = NULL;
    
    struct memblock mem_127688;
    
    mem_127688.references = NULL;
    
    struct memblock mem_127686;
    
    mem_127686.references = NULL;
    
    struct memblock ext_mem_127683;
    
    ext_mem_127683.references = NULL;
    
    struct memblock ext_mem_127684;
    
    ext_mem_127684.references = NULL;
    
    struct memblock ext_mem_127685;
    
    ext_mem_127685.references = NULL;
    
    struct memblock mem_127681;
    
    mem_127681.references = NULL;
    
    struct memblock mem_127679;
    
    mem_127679.references = NULL;
    
    struct memblock mem_127677;
    
    mem_127677.references = NULL;
    
    struct memblock mem_127675;
    
    mem_127675.references = NULL;
    
    struct memblock mem_param_126283;
    
    mem_param_126283.references = NULL;
    
    struct memblock mem_param_126279;
    
    mem_param_126279.references = NULL;
    
    struct memblock mem_param_126275;
    
    mem_param_126275.references = NULL;
    
    struct memblock mem_param_126271;
    
    mem_param_126271.references = NULL;
    
    struct memblock mem_param_126267;
    
    mem_param_126267.references = NULL;
    
    struct memblock mem_param_126263;
    
    mem_param_126263.references = NULL;
    
    struct memblock mem_param_126259;
    
    mem_param_126259.references = NULL;
    
    struct memblock mem_param_126255;
    
    mem_param_126255.references = NULL;
    
    struct memblock mem_param_126251;
    
    mem_param_126251.references = NULL;
    
    struct memblock mem_param_126247;
    
    mem_param_126247.references = NULL;
    
    struct memblock mem_param_126243;
    
    mem_param_126243.references = NULL;
    
    struct memblock mem_param_126239;
    
    mem_param_126239.references = NULL;
    
    struct memblock mem_param_126235;
    
    mem_param_126235.references = NULL;
    
    struct memblock mem_param_126231;
    
    mem_param_126231.references = NULL;
    
    struct memblock mem_param_126227;
    
    mem_param_126227.references = NULL;
    
    struct memblock mem_param_126223;
    
    mem_param_126223.references = NULL;
    
    struct memblock mem_param_126219;
    
    mem_param_126219.references = NULL;
    
    struct memblock mem_param_126215;
    
    mem_param_126215.references = NULL;
    
    struct memblock mem_param_126211;
    
    mem_param_126211.references = NULL;
    
    struct memblock mem_param_126207;
    
    mem_param_126207.references = NULL;
    
    struct memblock mem_param_126203;
    
    mem_param_126203.references = NULL;
    
    struct memblock mem_param_126199;
    
    mem_param_126199.references = NULL;
    
    struct memblock mem_param_126195;
    
    mem_param_126195.references = NULL;
    
    struct memblock mem_param_126191;
    
    mem_param_126191.references = NULL;
    
    struct memblock mem_param_126187;
    
    mem_param_126187.references = NULL;
    
    struct memblock mem_param_126183;
    
    mem_param_126183.references = NULL;
    
    struct memblock mem_param_126179;
    
    mem_param_126179.references = NULL;
    
    struct memblock ext_mem_127855;
    
    ext_mem_127855.references = NULL;
    
    struct memblock ext_mem_127856;
    
    ext_mem_127856.references = NULL;
    
    struct memblock ext_mem_127857;
    
    ext_mem_127857.references = NULL;
    
    struct memblock ext_mem_127858;
    
    ext_mem_127858.references = NULL;
    
    struct memblock ext_mem_127859;
    
    ext_mem_127859.references = NULL;
    
    struct memblock ext_mem_127860;
    
    ext_mem_127860.references = NULL;
    
    struct memblock ext_mem_127861;
    
    ext_mem_127861.references = NULL;
    
    struct memblock ext_mem_127862;
    
    ext_mem_127862.references = NULL;
    
    struct memblock ext_mem_127863;
    
    ext_mem_127863.references = NULL;
    
    struct memblock ext_mem_127864;
    
    ext_mem_127864.references = NULL;
    
    struct memblock ext_mem_127865;
    
    ext_mem_127865.references = NULL;
    
    struct memblock ext_mem_127866;
    
    ext_mem_127866.references = NULL;
    
    struct memblock ext_mem_127867;
    
    ext_mem_127867.references = NULL;
    
    struct memblock ext_mem_127868;
    
    ext_mem_127868.references = NULL;
    
    struct memblock ext_mem_127869;
    
    ext_mem_127869.references = NULL;
    
    struct memblock ext_mem_127870;
    
    ext_mem_127870.references = NULL;
    
    struct memblock ext_mem_127871;
    
    ext_mem_127871.references = NULL;
    
    struct memblock ext_mem_127872;
    
    ext_mem_127872.references = NULL;
    
    struct memblock ext_mem_127873;
    
    ext_mem_127873.references = NULL;
    
    struct memblock ext_mem_127874;
    
    ext_mem_127874.references = NULL;
    
    struct memblock ext_mem_127875;
    
    ext_mem_127875.references = NULL;
    
    struct memblock ext_mem_127876;
    
    ext_mem_127876.references = NULL;
    
    struct memblock ext_mem_127877;
    
    ext_mem_127877.references = NULL;
    
    struct memblock ext_mem_127878;
    
    ext_mem_127878.references = NULL;
    
    struct memblock ext_mem_127879;
    
    ext_mem_127879.references = NULL;
    
    struct memblock ext_mem_127880;
    
    ext_mem_127880.references = NULL;
    
    struct memblock ext_mem_127881;
    
    ext_mem_127881.references = NULL;
    
    struct memblock mem_out_127980;
    
    mem_out_127980.references = NULL;
    
    struct memblock mem_out_127979;
    
    mem_out_127979.references = NULL;
    
    struct memblock mem_out_127978;
    
    mem_out_127978.references = NULL;
    
    struct memblock mem_out_127977;
    
    mem_out_127977.references = NULL;
    
    struct memblock mem_out_127976;
    
    mem_out_127976.references = NULL;
    
    struct memblock mem_out_127975;
    
    mem_out_127975.references = NULL;
    
    struct memblock mem_out_127974;
    
    mem_out_127974.references = NULL;
    
    struct memblock mem_out_127973;
    
    mem_out_127973.references = NULL;
    
    struct memblock mem_out_127972;
    
    mem_out_127972.references = NULL;
    
    struct memblock mem_out_127971;
    
    mem_out_127971.references = NULL;
    
    struct memblock mem_out_127970;
    
    mem_out_127970.references = NULL;
    
    struct memblock mem_out_127969;
    
    mem_out_127969.references = NULL;
    
    struct memblock mem_out_127968;
    
    mem_out_127968.references = NULL;
    
    struct memblock mem_out_127967;
    
    mem_out_127967.references = NULL;
    
    struct memblock mem_out_127966;
    
    mem_out_127966.references = NULL;
    
    struct memblock mem_out_127965;
    
    mem_out_127965.references = NULL;
    
    struct memblock mem_out_127964;
    
    mem_out_127964.references = NULL;
    
    struct memblock mem_out_127963;
    
    mem_out_127963.references = NULL;
    
    struct memblock mem_out_127962;
    
    mem_out_127962.references = NULL;
    
    struct memblock mem_out_127961;
    
    mem_out_127961.references = NULL;
    
    struct memblock mem_out_127960;
    
    mem_out_127960.references = NULL;
    
    struct memblock mem_out_127959;
    
    mem_out_127959.references = NULL;
    
    struct memblock mem_out_127958;
    
    mem_out_127958.references = NULL;
    
    struct memblock mem_out_127957;
    
    mem_out_127957.references = NULL;
    
    struct memblock mem_out_127956;
    
    mem_out_127956.references = NULL;
    
    struct memblock mem_out_127955;
    
    mem_out_127955.references = NULL;
    
    struct memblock mem_out_127954;
    
    mem_out_127954.references = NULL;
    
    struct memblock mem_126137 = ctx->constants->mem_126137;
    struct memblock mem_126138 = ctx->constants->mem_126138;
    struct memblock mem_126139 = ctx->constants->mem_126139;
    struct memblock mem_126140 = ctx->constants->mem_126140;
    struct memblock mem_126141 = ctx->constants->mem_126141;
    struct memblock mem_126142 = ctx->constants->mem_126142;
    struct memblock mem_126143 = ctx->constants->mem_126143;
    struct memblock mem_126144 = ctx->constants->mem_126144;
    struct memblock mem_126145 = ctx->constants->mem_126145;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_126284_cached_sizze_128456 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126284, &mem_126284_cached_sizze_128456, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126285_cached_sizze_128457 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_126285, &mem_126285_cached_sizze_128457, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126294_cached_sizze_128458 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_126294, &mem_126294_cached_sizze_128458, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126301_cached_sizze_128459 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126301, &mem_126301_cached_sizze_128459, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126316_cached_sizze_128460 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126316, &mem_126316_cached_sizze_128460, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126317_cached_sizze_128461 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126317, &mem_126317_cached_sizze_128461, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126318_cached_sizze_128462 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126318, &mem_126318_cached_sizze_128462, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126329_cached_sizze_128463 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126329, &mem_126329_cached_sizze_128463, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126346_cached_sizze_128464 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126346, &mem_126346_cached_sizze_128464, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126347_cached_sizze_128465 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126347, &mem_126347_cached_sizze_128465, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126355_cached_sizze_128466 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126355, &mem_126355_cached_sizze_128466, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126369_cached_sizze_128467 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126369, &mem_126369_cached_sizze_128467, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126370_cached_sizze_128468 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126370, &mem_126370_cached_sizze_128468, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126371_cached_sizze_128469 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126371, &mem_126371_cached_sizze_128469, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126384_cached_sizze_128470 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126384, &mem_126384_cached_sizze_128470, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126385_cached_sizze_128471 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126385, &mem_126385_cached_sizze_128471, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126386_cached_sizze_128472 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126386, &mem_126386_cached_sizze_128472, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126417_cached_sizze_128473 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126417, &mem_126417_cached_sizze_128473, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126418_cached_sizze_128474 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126418, &mem_126418_cached_sizze_128474, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126419_cached_sizze_128475 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126419, &mem_126419_cached_sizze_128475, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126435_cached_sizze_128476 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126435, &mem_126435_cached_sizze_128476, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126436_cached_sizze_128477 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126436, &mem_126436_cached_sizze_128477, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126437_cached_sizze_128478 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126437, &mem_126437_cached_sizze_128478, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126450_cached_sizze_128479 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_126450, &mem_126450_cached_sizze_128479, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126451_cached_sizze_128480 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_126451, &mem_126451_cached_sizze_128480, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126452_cached_sizze_128481 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_126452, &mem_126452_cached_sizze_128481, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126498_cached_sizze_128482 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_126498, &mem_126498_cached_sizze_128482, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126499_cached_sizze_128483 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_126499, &mem_126499_cached_sizze_128483, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126500_cached_sizze_128484 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_126500, &mem_126500_cached_sizze_128484, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126501_cached_sizze_128485 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126501, &mem_126501_cached_sizze_128485, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126522_cached_sizze_128486 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126522, &mem_126522_cached_sizze_128486, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126523_cached_sizze_128487 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126523, &mem_126523_cached_sizze_128487, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126524_cached_sizze_128488 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126524, &mem_126524_cached_sizze_128488, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126525_cached_sizze_128489 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126525, &mem_126525_cached_sizze_128489, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126542_cached_sizze_128490 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126542, &mem_126542_cached_sizze_128490, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126543_cached_sizze_128491 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126543, &mem_126543_cached_sizze_128491, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126544_cached_sizze_128492 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126544, &mem_126544_cached_sizze_128492, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126545_cached_sizze_128493 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126545, &mem_126545_cached_sizze_128493, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126586_cached_sizze_128494 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126586, &mem_126586_cached_sizze_128494, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126591_cached_sizze_128495 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_126591, &mem_126591_cached_sizze_128495, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126595_cached_sizze_128496 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126595, &mem_126595_cached_sizze_128496, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126629_cached_sizze_128497 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126629, &mem_126629_cached_sizze_128497, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126634_cached_sizze_128498 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126634, &mem_126634_cached_sizze_128498, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126645_cached_sizze_128499 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126645, &mem_126645_cached_sizze_128499, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126650_cached_sizze_128500 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126650, &mem_126650_cached_sizze_128500, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126661_cached_sizze_128501 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126661, &mem_126661_cached_sizze_128501, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126666_cached_sizze_128502 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126666, &mem_126666_cached_sizze_128502, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126677_cached_sizze_128503 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126677, &mem_126677_cached_sizze_128503, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126678_cached_sizze_128504 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126678, &mem_126678_cached_sizze_128504, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126686_cached_sizze_128505 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126686, &mem_126686_cached_sizze_128505, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126700_cached_sizze_128506 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_126700, &mem_126700_cached_sizze_128506, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126705_cached_sizze_128507 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126705, &mem_126705_cached_sizze_128507, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126716_cached_sizze_128508 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_126716, &mem_126716_cached_sizze_128508, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126721_cached_sizze_128509 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126721, &mem_126721_cached_sizze_128509, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126732_cached_sizze_128510 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126732, &mem_126732_cached_sizze_128510, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126737_cached_sizze_128511 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126737, &mem_126737_cached_sizze_128511, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126748_cached_sizze_128512 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126748, &mem_126748_cached_sizze_128512, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126753_cached_sizze_128513 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126753, &mem_126753_cached_sizze_128513, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126764_cached_sizze_128514 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_126764, &mem_126764_cached_sizze_128514, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126769_cached_sizze_128515 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_126769, &mem_126769_cached_sizze_128515, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126780_cached_sizze_128516 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_126780, &mem_126780_cached_sizze_128516, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126781_cached_sizze_128517 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_126781, &mem_126781_cached_sizze_128517, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126782_cached_sizze_128518 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_126782, &mem_126782_cached_sizze_128518, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:115:13-33
    if (mem_126796_cached_sizze_128519 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_126796, &mem_126796_cached_sizze_128519, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126801_cached_sizze_128520 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_126801, &mem_126801_cached_sizze_128520, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126834_cached_sizze_128522 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_126834, &mem_126834_cached_sizze_128522, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126840_cached_sizze_128523 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_126840, &mem_126840_cached_sizze_128523, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126845_cached_sizze_128524 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_126845, &mem_126845_cached_sizze_128524, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126861_cached_sizze_128525 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_126861, &mem_126861_cached_sizze_128525, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126866_cached_sizze_128526 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_126866, &mem_126866_cached_sizze_128526, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126877_cached_sizze_128527 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_126877, &mem_126877_cached_sizze_128527, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126883_cached_sizze_128528 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_126883, &mem_126883_cached_sizze_128528, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126888_cached_sizze_128529 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_126888, &mem_126888_cached_sizze_128529, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126904_cached_sizze_128530 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_126904, &mem_126904_cached_sizze_128530, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126909_cached_sizze_128531 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_126909, &mem_126909_cached_sizze_128531, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126920_cached_sizze_128532 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126920, &mem_126920_cached_sizze_128532, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126925_cached_sizze_128533 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126925, &mem_126925_cached_sizze_128533, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126936_cached_sizze_128534 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_126936, &mem_126936_cached_sizze_128534, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126941_cached_sizze_128535 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_126941, &mem_126941_cached_sizze_128535, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126952_cached_sizze_128536 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_126952, &mem_126952_cached_sizze_128536, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126953_cached_sizze_128537 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_126953, &mem_126953_cached_sizze_128537, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126962_cached_sizze_128538 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126962, &mem_126962_cached_sizze_128538, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126963_cached_sizze_128539 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126963, &mem_126963_cached_sizze_128539, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126984_cached_sizze_128540 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_126984, &mem_126984_cached_sizze_128540, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_126989_cached_sizze_128541 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_126989, &mem_126989_cached_sizze_128541, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127000_cached_sizze_128542 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127000, &mem_127000_cached_sizze_128542, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127005_cached_sizze_128543 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127005, &mem_127005_cached_sizze_128543, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127016_cached_sizze_128544 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127016, &mem_127016_cached_sizze_128544, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127021_cached_sizze_128545 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127021, &mem_127021_cached_sizze_128545, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127032_cached_sizze_128546 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127032, &mem_127032_cached_sizze_128546, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127033_cached_sizze_128547 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127033, &mem_127033_cached_sizze_128547, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127046_cached_sizze_128548 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127046, &mem_127046_cached_sizze_128548, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127051_cached_sizze_128549 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127051, &mem_127051_cached_sizze_128549, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127062_cached_sizze_128550 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127062, &mem_127062_cached_sizze_128550, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127067_cached_sizze_128551 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127067, &mem_127067_cached_sizze_128551, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127078_cached_sizze_128552 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127078, &mem_127078_cached_sizze_128552, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127079_cached_sizze_128553 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127079, &mem_127079_cached_sizze_128553, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127088_cached_sizze_128554 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127088, &mem_127088_cached_sizze_128554, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127089_cached_sizze_128555 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127089, &mem_127089_cached_sizze_128555, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127110_cached_sizze_128556 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127110, &mem_127110_cached_sizze_128556, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127116_cached_sizze_128557 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_127116, &mem_127116_cached_sizze_128557, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127121_cached_sizze_128558 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_127121, &mem_127121_cached_sizze_128558, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127137_cached_sizze_128559 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127137, &mem_127137_cached_sizze_128559, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127138_cached_sizze_128560 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127138, &mem_127138_cached_sizze_128560, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127139_cached_sizze_128561 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127139, &mem_127139_cached_sizze_128561, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127155_cached_sizze_128562 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_127155, &mem_127155_cached_sizze_128562, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127156_cached_sizze_128563 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_127156, &mem_127156_cached_sizze_128563, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127157_cached_sizze_128564 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_127157, &mem_127157_cached_sizze_128564, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127170_cached_sizze_128565 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_127170, &mem_127170_cached_sizze_128565, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127171_cached_sizze_128566 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_127171, &mem_127171_cached_sizze_128566, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127172_cached_sizze_128567 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_127172, &mem_127172_cached_sizze_128567, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127182_cached_sizze_128568 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127182, &mem_127182_cached_sizze_128568, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127189_cached_sizze_128569 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127189, &mem_127189_cached_sizze_128569, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127190_cached_sizze_128570 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127190, &mem_127190_cached_sizze_128570, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127191_cached_sizze_128571 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127191, &mem_127191_cached_sizze_128571, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:115:13-33
    if (mem_127202_cached_sizze_128572 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127202, &mem_127202_cached_sizze_128572, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127219_cached_sizze_128573 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127219, &mem_127219_cached_sizze_128573, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127224_cached_sizze_128574 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127224, &mem_127224_cached_sizze_128574, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127235_cached_sizze_128575 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127235, &mem_127235_cached_sizze_128575, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127242_cached_sizze_128576 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127242, &mem_127242_cached_sizze_128576, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127247_cached_sizze_128577 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127247, &mem_127247_cached_sizze_128577, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127258_cached_sizze_128578 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127258, &mem_127258_cached_sizze_128578, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127259_cached_sizze_128579 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127259, &mem_127259_cached_sizze_128579, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127260_cached_sizze_128580 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127260, &mem_127260_cached_sizze_128580, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:115:13-33
    if (mem_127271_cached_sizze_128581 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127271, &mem_127271_cached_sizze_128581, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127288_cached_sizze_128582 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127288, &mem_127288_cached_sizze_128582, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127293_cached_sizze_128583 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127293, &mem_127293_cached_sizze_128583, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127304_cached_sizze_128584 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127304, &mem_127304_cached_sizze_128584, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127311_cached_sizze_128585 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127311, &mem_127311_cached_sizze_128585, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127316_cached_sizze_128586 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127316, &mem_127316_cached_sizze_128586, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127363_cached_sizze_128587 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127363, &mem_127363_cached_sizze_128587, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127364_cached_sizze_128588 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127364, &mem_127364_cached_sizze_128588, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127365_cached_sizze_128589 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127365, &mem_127365_cached_sizze_128589, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127378_cached_sizze_128590 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127378, &mem_127378_cached_sizze_128590, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127379_cached_sizze_128591 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127379, &mem_127379_cached_sizze_128591, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127380_cached_sizze_128592 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127380, &mem_127380_cached_sizze_128592, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127411_cached_sizze_128593 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127411, &mem_127411_cached_sizze_128593, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127412_cached_sizze_128594 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127412, &mem_127412_cached_sizze_128594, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127413_cached_sizze_128595 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127413, &mem_127413_cached_sizze_128595, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127414_cached_sizze_128596 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127414, &mem_127414_cached_sizze_128596, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127431_cached_sizze_128597 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127431, &mem_127431_cached_sizze_128597, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127432_cached_sizze_128598 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127432, &mem_127432_cached_sizze_128598, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127433_cached_sizze_128599 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127433, &mem_127433_cached_sizze_128599, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127434_cached_sizze_128600 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127434, &mem_127434_cached_sizze_128600, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127475_cached_sizze_128601 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127475, &mem_127475_cached_sizze_128601, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127480_cached_sizze_128602 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127480, &mem_127480_cached_sizze_128602, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127491_cached_sizze_128603 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127491, &mem_127491_cached_sizze_128603, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127492_cached_sizze_128604 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127492, &mem_127492_cached_sizze_128604, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127505_cached_sizze_128605 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127505, &mem_127505_cached_sizze_128605, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127510_cached_sizze_128606 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127510, &mem_127510_cached_sizze_128606, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127521_cached_sizze_128607 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127521, &mem_127521_cached_sizze_128607, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127522_cached_sizze_128608 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127522, &mem_127522_cached_sizze_128608, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127531_cached_sizze_128609 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127531, &mem_127531_cached_sizze_128609, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127532_cached_sizze_128610 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127532, &mem_127532_cached_sizze_128610, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127553_cached_sizze_128611 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127553, &mem_127553_cached_sizze_128611, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127554_cached_sizze_128612 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127554, &mem_127554_cached_sizze_128612, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127555_cached_sizze_128613 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127555, &mem_127555_cached_sizze_128613, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127556_cached_sizze_128614 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127556, &mem_127556_cached_sizze_128614, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127581_cached_sizze_128615 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127581, &mem_127581_cached_sizze_128615, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127582_cached_sizze_128616 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127582, &mem_127582_cached_sizze_128616, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127595_cached_sizze_128617 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127595, &mem_127595_cached_sizze_128617, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127596_cached_sizze_128618 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_127596, &mem_127596_cached_sizze_128618, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127605_cached_sizze_128619 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127605, &mem_127605_cached_sizze_128619, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127606_cached_sizze_128620 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127606, &mem_127606_cached_sizze_128620, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127627_cached_sizze_128621 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_127627, &mem_127627_cached_sizze_128621, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127632_cached_sizze_128622 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127632, &mem_127632_cached_sizze_128622, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127643_cached_sizze_128623 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_127643, &mem_127643_cached_sizze_128623, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127644_cached_sizze_128624 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_127644, &mem_127644_cached_sizze_128624, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127653_cached_sizze_128625 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127653, &mem_127653_cached_sizze_128625, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_127654_cached_sizze_128626 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_127654, &mem_127654_cached_sizze_128626, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:599:5-604:51
    if (memblock_set(ctx, &mem_param_126179, &wdown_mem_126146, "wdown_mem_126146") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126183, &wkey_mem_126147, "wkey_mem_126147") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126187, &wout_mem_126148, "wout_mem_126148") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126191, &wpe_mem_126149, "wpe_mem_126149") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126195, &wqry_mem_126150, "wqry_mem_126150") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126199, &wte_mem_126151, "wte_mem_126151") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126203, &wup_mem_126152, "wup_mem_126152") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126207, &wval_mem_126153, "wval_mem_126153") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126211, &wvoc_mem_126154, "wvoc_mem_126154") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126215, &wdown_mem_126155, "wdown_mem_126155") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126219, &wkey_mem_126156, "wkey_mem_126156") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126223, &wout_mem_126157, "wout_mem_126157") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126227, &wpe_mem_126158, "wpe_mem_126158") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126231, &wqry_mem_126159, "wqry_mem_126159") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126235, &wte_mem_126160, "wte_mem_126160") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126239, &wup_mem_126161, "wup_mem_126161") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126243, &wval_mem_126162, "wval_mem_126162") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126247, &wvoc_mem_126163, "wvoc_mem_126163") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126251, &wdown_mem_126164, "wdown_mem_126164") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126255, &wkey_mem_126165, "wkey_mem_126165") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126259, &wout_mem_126166, "wout_mem_126166") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126263, &wpe_mem_126167, "wpe_mem_126167") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126267, &wqry_mem_126168, "wqry_mem_126168") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126271, &wte_mem_126169, "wte_mem_126169") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126275, &wup_mem_126170, "wup_mem_126170") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126279, &wval_mem_126171, "wval_mem_126171") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_126283, &wvoc_mem_126172, "wvoc_mem_126172") != 0)
        return 1;
    for (int64_t step_115941 = 0; step_115941 < (int64_t) 500; step_115941++) {
        // futhark/microgpt.fut:601:16-25
        
        int64_t dl_115969 = ((int64_t *) dls_mem_126174.mem)[step_115941];
        
        // futhark/microgpt.fut:441:37-40
        
        int64_t zl_rhs_115974 = sub64(dl_115969, (int64_t) 1);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125291 = 0; i_125291 < (int64_t) 16; i_125291++) {
            // futhark/microgpt.fut:441:25-81
            
            bool cond_118246 = slt64(i_125291, zl_rhs_115974);
            
            // futhark/microgpt.fut:441:56-59
            
            int64_t zeze_lhs_118247 = add64((int64_t) 1, i_125291);
            
            // futhark/microgpt.fut:441:47-60
            
            bool x_118248 = sle64((int64_t) 0, zeze_lhs_118247);
            
            // futhark/microgpt.fut:441:47-60
            
            bool y_118249 = slt64(zeze_lhs_118247, (int64_t) 16);
            
            // futhark/microgpt.fut:441:47-60
            
            bool bounds_check_118250 = x_118248 && y_118249;
            
            // futhark/microgpt.fut:9:27-39
            
            bool loop_not_taken_118251 = !cond_118246;
            
            // futhark/microgpt.fut:9:27-39
            
            bool protect_assert_disj_118252 = bounds_check_118250 || loop_not_taken_118251;
            
            // futhark/microgpt.fut:441:47-60
            
            bool index_certs_118253;
            
            if (!protect_assert_disj_118252) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_118247, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:441:47-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:441:3-83\n   #6  futhark/microgpt.fut:548:18-38\n   #7  futhark/microgpt.fut:570:26-576:31\n   #8  futhark/microgpt.fut:604:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            int64_t tmp_118268 = ((int64_t *) seqs_mem_126175.mem)[step_115941 * (int64_t) 16 + i_125291];
            
            // futhark/microgpt.fut:550:37-51
            
            bool x_118269 = sle64((int64_t) 0, tmp_118268);
            
            // futhark/microgpt.fut:550:37-51
            
            bool y_118270 = slt64(tmp_118268, (int64_t) 27);
            
            // futhark/microgpt.fut:550:37-51
            
            bool bounds_check_118271 = x_118269 && y_118270;
            
            // futhark/microgpt.fut:550:37-51
            
            bool index_certs_118272;
            
            if (!bounds_check_118271) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_118268, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:550:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:550:16-55\n   #6  futhark/microgpt.fut:570:26-576:31\n   #7  futhark/microgpt.fut:604:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:441:47-60
            
            int64_t zeze_lhs_118254;
            
            if (cond_118246) {
                int64_t x_125045 = ((int64_t *) seqs_mem_126175.mem)[step_115941 * (int64_t) 16 + zeze_lhs_118247];
                
                zeze_lhs_118254 = x_125045;
            } else {
                zeze_lhs_118254 = (int64_t) 0;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125281 = 0; i_125281 < (int64_t) 27; i_125281++) {
                // futhark/microgpt.fut:441:61-65
                
                bool cond_t_res_118258 = zeze_lhs_118254 == i_125281;
                
                // futhark/microgpt.fut:9:27-39
                
                bool x_118259 = cond_118246 && cond_t_res_118258;
                
                // futhark/microgpt.fut:441:25-81
                
                double lifted_lambda_res_118260;
                
                if (x_118259) {
                    lifted_lambda_res_118260 = 1.0;
                } else {
                    lifted_lambda_res_118260 = 0.0;
                }
                ((double *) mem_126294)[i_125281] = lifted_lambda_res_118260;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125285 = 0; i_125285 < (int64_t) 16; i_125285++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_118279 = ((double *) mem_param_126199.mem)[tmp_118268 * (int64_t) 16 + i_125285];
                
                ((double *) mem_126301)[i_125285] = lifted_lambda_res_118279;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126284, i_125291 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126301, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126285, i_125291 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126294, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125304 = 0; i_125304 < (int64_t) 16; i_125304++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_118359;
            double r_118361 = 0.0;
            
            for (int64_t i_118360 = 0; i_118360 < (int64_t) 16; i_118360++) {
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_118362 = ((double *) mem_param_126191.mem)[i_125304 * (int64_t) 16 + i_118360];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_rhs_118363 = ((double *) mem_126284)[i_125304 * (int64_t) 16 + i_118360];
                
                // futhark/microgpt.fut:279:123-159
                
                double zp_res_118364 = zp_lhs_118362 + zp_rhs_118363;
                
                // futhark/microgpt.fut:279:139-202
                
                double zt_res_118365 = zp_res_118364 * zp_res_118364;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_118366 = r_118361 + zt_res_118365;
                double r_tmp_128042 = zp_res_118366;
                
                r_118361 = r_tmp_128042;
            }
            defunc_0_lifted_lambda_res_118359 = r_118361;
            // futhark/microgpt.fut:279:102-221
            
            double zs_res_118367 = defunc_0_lifted_lambda_res_118359 / 16.0;
            
            // futhark/microgpt.fut:279:207-250
            
            double zp_res_118368 = 1.0e-5 + zs_res_118367;
            
            // futhark/microgpt.fut:279:92-250
            
            double sqrt_res_118369 = futrts_sqrt64(zp_res_118368);
            
            // futhark/microgpt.fut:279:83-252
            
            double zs_res_118370 = 1.0 / sqrt_res_118369;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125296 = 0; i_125296 < (int64_t) 16; i_125296++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_118377 = ((double *) mem_param_126191.mem)[i_125304 * (int64_t) 16 + i_125296];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_118378 = ((double *) mem_126284)[i_125304 * (int64_t) 16 + i_125296];
                
                // futhark/microgpt.fut:279:40-76
                
                double zp_res_118379 = zp_lhs_118377 + zp_rhs_118378;
                
                // futhark/microgpt.fut:279:56-252
                
                double zt_res_118380 = zs_res_118370 * zp_res_118379;
                
                ((double *) mem_126329)[i_125296] = zt_res_118380;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_118388;
            double r_118390 = 0.0;
            
            for (int64_t i_118389 = 0; i_118389 < (int64_t) 16; i_118389++) {
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_118391 = ((double *) mem_param_126191.mem)[i_125304 * (int64_t) 16 + i_118389];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_rhs_118392 = ((double *) mem_126284)[i_125304 * (int64_t) 16 + i_118389];
                
                // futhark/microgpt.fut:366:71-115
                
                double zp_res_118393 = zp_lhs_118391 + zp_rhs_118392;
                
                // futhark/microgpt.fut:366:91-166
                
                double zt_res_118394 = zp_res_118393 * zp_res_118393;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_118395 = r_118390 + zt_res_118394;
                double r_tmp_128044 = zp_res_118395;
                
                r_118390 = r_tmp_128044;
            }
            defunc_0_lifted_lambda_res_118388 = r_118390;
            // futhark/microgpt.fut:366:48-185
            
            double zs_res_118396 = defunc_0_lifted_lambda_res_118388 / 16.0;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_118406;
            double r_118408 = 0.0;
            
            for (int64_t i_118407 = 0; i_118407 < (int64_t) 16; i_118407++) {
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_118409 = ((double *) mem_param_126191.mem)[i_125304 * (int64_t) 16 + i_118407];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_rhs_118410 = ((double *) mem_126284)[i_125304 * (int64_t) 16 + i_118407];
                
                // futhark/microgpt.fut:379:72-116
                
                double zp_res_118411 = zp_lhs_118409 + zp_rhs_118410;
                
                // futhark/microgpt.fut:379:92-167
                
                double zt_res_118412 = zp_res_118411 * zp_res_118411;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_118413 = r_118408 + zt_res_118412;
                double r_tmp_128045 = zp_res_118413;
                
                r_118408 = r_tmp_128045;
            }
            defunc_0_lifted_lambda_res_118406 = r_118408;
            // futhark/microgpt.fut:379:49-186
            
            double zs_res_118414 = defunc_0_lifted_lambda_res_118406 / 16.0;
            
            ((double *) mem_126316)[i_125304] = zs_res_118414;
            ((double *) mem_126317)[i_125304] = zs_res_118396;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126318, i_125304 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126329, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125316 = 0; i_125316 < (int64_t) 16; i_125316++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_118433;
            double r_118435 = 0.0;
            
            for (int64_t i_118434 = 0; i_118434 < (int64_t) 16; i_118434++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_118436 = ((double *) mem_126318)[i_125316 * (int64_t) 16 + i_118434];
                
                // futhark/microgpt.fut:280:98-131
                
                double zt_res_118437 = zt_lhs_118436 * zt_lhs_118436;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_118438 = r_118435 + zt_res_118437;
                double r_tmp_128048 = zp_res_118438;
                
                r_118435 = r_tmp_128048;
            }
            defunc_0_lifted_lambda_res_118433 = r_118435;
            // futhark/microgpt.fut:280:78-149
            
            double zs_res_118439 = defunc_0_lifted_lambda_res_118433 / 16.0;
            
            // futhark/microgpt.fut:280:135-178
            
            double zp_res_118440 = 1.0e-5 + zs_res_118439;
            
            // futhark/microgpt.fut:280:68-178
            
            double sqrt_res_118441 = futrts_sqrt64(zp_res_118440);
            
            // futhark/microgpt.fut:280:59-180
            
            double zs_res_118442 = 1.0 / sqrt_res_118441;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125310 = 0; i_125310 < (int64_t) 16; i_125310++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_118449 = ((double *) mem_126318)[i_125316 * (int64_t) 16 + i_125310];
                
                // futhark/microgpt.fut:280:39-180
                
                double zt_res_118450 = zs_res_118442 * zt_lhs_118449;
                
                ((double *) mem_126355)[i_125310] = zt_res_118450;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_118458;
            double r_118460 = 0.0;
            
            for (int64_t i_118459 = 0; i_118459 < (int64_t) 16; i_118459++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_118461 = ((double *) mem_126318)[i_125316 * (int64_t) 16 + i_118459];
                
                // futhark/microgpt.fut:345:70-111
                
                double zt_res_118462 = zt_lhs_118461 * zt_lhs_118461;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_118463 = r_118460 + zt_res_118462;
                double r_tmp_128050 = zp_res_118463;
                
                r_118460 = r_tmp_128050;
            }
            defunc_0_lifted_lambda_res_118458 = r_118460;
            // futhark/microgpt.fut:345:48-129
            
            double zs_res_118464 = defunc_0_lifted_lambda_res_118458 / 16.0;
            
            ((double *) mem_126346)[i_125316] = zs_res_118464;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126347, i_125316 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126355, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125335 = 0; i_125335 < (int64_t) 16; i_125335++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125325 = 0; i_125325 < (int64_t) 16; i_125325++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_121052;
                double r_121054 = 0.0;
                
                for (int64_t i_121053 = 0; i_121053 < (int64_t) 16; i_121053++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_121055 = ((double *) mem_param_126195.mem)[i_125325 * (int64_t) 16 + i_121053];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_121056 = ((double *) mem_126347)[i_125335 * (int64_t) 16 + i_121053];
                    
                    // futhark/microgpt.fut:281:59-94
                    
                    double zt_res_121057 = zt_lhs_121055 * zt_rhs_121056;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_121058 = r_121054 + zt_res_121057;
                    double r_tmp_128057 = zp_res_121058;
                    
                    r_121054 = r_tmp_128057;
                }
                defunc_0_lifted_lambda_res_121052 = r_121054;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_121065;
                double r_121067 = 0.0;
                
                for (int64_t i_121066 = 0; i_121066 < (int64_t) 16; i_121066++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_121068 = ((double *) mem_param_126183.mem)[i_125325 * (int64_t) 16 + i_121066];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_121069 = ((double *) mem_126347)[i_125335 * (int64_t) 16 + i_121066];
                    
                    // futhark/microgpt.fut:282:62-101
                    
                    double zt_res_121070 = zt_lhs_121068 * zt_rhs_121069;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_121071 = r_121067 + zt_res_121070;
                    double r_tmp_128058 = zp_res_121071;
                    
                    r_121067 = r_tmp_128058;
                }
                defunc_0_lifted_lambda_res_121065 = r_121067;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_121081;
                double r_121083 = 0.0;
                
                for (int64_t i_121082 = 0; i_121082 < (int64_t) 16; i_121082++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_121084 = ((double *) mem_param_126207.mem)[i_125325 * (int64_t) 16 + i_121082];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_121085 = ((double *) mem_126347)[i_125335 * (int64_t) 16 + i_121082];
                    
                    // futhark/microgpt.fut:283:63-102
                    
                    double zt_res_121086 = zt_lhs_121084 * zt_rhs_121085;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_121087 = r_121083 + zt_res_121086;
                    double r_tmp_128059 = zp_res_121087;
                    
                    r_121083 = r_tmp_128059;
                }
                defunc_0_lifted_lambda_res_121081 = r_121083;
                ((double *) mem_126384)[i_125325] = defunc_0_lifted_lambda_res_121081;
                ((double *) mem_126385)[i_125325] = defunc_0_lifted_lambda_res_121065;
                ((double *) mem_126386)[i_125325] = defunc_0_lifted_lambda_res_121052;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126369, i_125335 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126384, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126370, i_125335 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126385, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126371, i_125335 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126386, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125365 = 0; i_125365 < (int64_t) 4; i_125365++) {
            // futhark/microgpt.fut:284:66-69
            
            int64_t zp_lhs_118665 = mul64((int64_t) 4, i_125365);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125355 = 0; i_125355 < (int64_t) 16; i_125355++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_125345 = 0; i_125345 < (int64_t) 4; i_125345++) {
                    // futhark/microgpt.fut:284:71-78
                    
                    int64_t tmp_121245 = add64(zp_lhs_118665, i_125345);
                    
                    // futhark/microgpt.fut:284:48-80
                    
                    bool x_121246 = sle64((int64_t) 0, tmp_121245);
                    
                    // futhark/microgpt.fut:284:48-80
                    
                    bool y_121247 = slt64(tmp_121245, (int64_t) 16);
                    
                    // futhark/microgpt.fut:284:48-80
                    
                    bool bounds_check_121248 = x_121246 && y_121247;
                    
                    // futhark/microgpt.fut:284:48-80
                    
                    bool index_certs_121249;
                    
                    if (!bounds_check_121248) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_121245, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:284:48-80\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:284:12-81\n   #9  futhark/microgpt.fut:553:5-76\n   #10 futhark/microgpt.fut:570:26-576:31\n   #11 futhark/microgpt.fut:604:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_121250 = ((double *) mem_126371)[i_125355 * (int64_t) 16 + tmp_121245];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_121258 = ((double *) mem_126370)[i_125355 * (int64_t) 16 + tmp_121245];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_121269 = ((double *) mem_126369)[i_125355 * (int64_t) 16 + tmp_121245];
                    
                    ((double *) mem_126450)[i_125345] = lifted_lambda_res_121269;
                    ((double *) mem_126451)[i_125345] = lifted_lambda_res_121258;
                    ((double *) mem_126452)[i_125345] = lifted_lambda_res_121250;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_126435, i_125355 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126450, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_126436, i_125355 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126451, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_126437, i_125355 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126452, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_126417, i_125365 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_126435, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_126418, i_125365 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_126436, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_126419, i_125365 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_126437, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125417 = 0; i_125417 < (int64_t) 4; i_125417++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125390 = 0; i_125390 < (int64_t) 16; i_125390++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_125377 = 0; i_125377 < (int64_t) 16; i_125377++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_121693;
                    double r_121695 = 0.0;
                    
                    for (int64_t i_121694 = 0; i_121694 < (int64_t) 4; i_121694++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_121696 = ((double *) mem_126419)[i_125417 * (int64_t) 64 + i_125390 * (int64_t) 4 + i_121694];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_121697 = ((double *) mem_126418)[i_125417 * (int64_t) 64 + i_125377 * (int64_t) 4 + i_121694];
                        
                        // futhark/microgpt.fut:287:112-165
                        
                        double zt_res_121698 = zt_lhs_121696 * zt_rhs_121697;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_121699 = r_121695 + zt_res_121698;
                        double r_tmp_128081 = zp_res_121699;
                        
                        r_121695 = r_tmp_128081;
                    }
                    defunc_0_lifted_lambda_res_121693 = r_121695;
                    // futhark/microgpt.fut:287:92-182
                    
                    double zs_res_121700 = defunc_0_lifted_lambda_res_121693 / 2.0;
                    double zp_rhs_121701 = ((double *) masks_mem_126173.mem)[step_115941 * (int64_t) 256 + i_125390 * (int64_t) 16 + i_125377];
                    
                    // futhark/microgpt.fut:287:169-206
                    
                    double zp_res_121702 = zs_res_121700 + zp_rhs_121701;
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_121709;
                    double r_121711 = 0.0;
                    
                    for (int64_t i_121710 = 0; i_121710 < (int64_t) 4; i_121710++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_121712 = ((double *) mem_126419)[i_125417 * (int64_t) 64 + i_125390 * (int64_t) 4 + i_121710];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_121713 = ((double *) mem_126418)[i_125417 * (int64_t) 64 + i_125377 * (int64_t) 4 + i_121710];
                        
                        // futhark/microgpt.fut:322:89-148
                        
                        double zt_res_121714 = zt_lhs_121712 * zt_rhs_121713;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_121715 = r_121711 + zt_res_121714;
                        double r_tmp_128082 = zp_res_121715;
                        
                        r_121711 = r_tmp_128082;
                    }
                    defunc_0_lifted_lambda_res_121709 = r_121711;
                    // futhark/microgpt.fut:322:68-165
                    
                    double zs_res_121716 = defunc_0_lifted_lambda_res_121709 / 2.0;
                    
                    // futhark/microgpt.fut:322:152-191
                    
                    double zp_res_121718 = zp_rhs_121701 + zs_res_121716;
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_121728;
                    double r_121730 = 0.0;
                    
                    for (int64_t i_121729 = 0; i_121729 < (int64_t) 4; i_121729++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_121731 = ((double *) mem_126419)[i_125417 * (int64_t) 64 + i_125390 * (int64_t) 4 + i_121729];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_121732 = ((double *) mem_126418)[i_125417 * (int64_t) 64 + i_125377 * (int64_t) 4 + i_121729];
                        
                        // futhark/microgpt.fut:325:89-148
                        
                        double zt_res_121733 = zt_lhs_121731 * zt_rhs_121732;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_121734 = r_121730 + zt_res_121733;
                        double r_tmp_128083 = zp_res_121734;
                        
                        r_121730 = r_tmp_128083;
                    }
                    defunc_0_lifted_lambda_res_121728 = r_121730;
                    // futhark/microgpt.fut:325:68-165
                    
                    double zs_res_121735 = defunc_0_lifted_lambda_res_121728 / 2.0;
                    
                    // futhark/microgpt.fut:325:152-191
                    
                    double zp_res_121737 = zp_rhs_121701 + zs_res_121735;
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_121749;
                    double r_121751 = 0.0;
                    
                    for (int64_t i_121750 = 0; i_121750 < (int64_t) 4; i_121750++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_121752 = ((double *) mem_126419)[i_125417 * (int64_t) 64 + i_125390 * (int64_t) 4 + i_121750];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_121753 = ((double *) mem_126418)[i_125417 * (int64_t) 64 + i_125377 * (int64_t) 4 + i_121750];
                        
                        // futhark/microgpt.fut:333:89-148
                        
                        double zt_res_121754 = zt_lhs_121752 * zt_rhs_121753;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_121755 = r_121751 + zt_res_121754;
                        double r_tmp_128084 = zp_res_121755;
                        
                        r_121751 = r_tmp_128084;
                    }
                    defunc_0_lifted_lambda_res_121749 = r_121751;
                    // futhark/microgpt.fut:333:68-165
                    
                    double zs_res_121756 = defunc_0_lifted_lambda_res_121749 / 2.0;
                    
                    // futhark/microgpt.fut:333:152-191
                    
                    double zp_res_121758 = zp_rhs_121701 + zs_res_121756;
                    
                    ((double *) mem_126542)[i_125377] = zp_res_121758;
                    ((double *) mem_126543)[i_125377] = zp_res_121737;
                    ((double *) mem_126544)[i_125377] = zp_res_121718;
                    ((double *) mem_126545)[i_125377] = zp_res_121702;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_126522, i_125390 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126542, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_126523, i_125390 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126543, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_126524, i_125390 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126544, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_126525, i_125390 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126545, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125407 = 0; i_125407 < (int64_t) 16; i_125407++) {
                double x_125064;
                double redout_125395 = -INFINITY;
                
                for (int64_t i_125396 = 0; i_125396 < (int64_t) 16; i_125396++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_121779 = ((double *) mem_126525)[i_125407 * (int64_t) 16 + i_125396];
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_119090 = fmax64(lifted_lambda_res_121779, redout_125395);
                    double redout_tmp_128086 = max_res_119090;
                    
                    redout_125395 = redout_tmp_128086;
                }
                x_125064 = redout_125395;
                // futhark/microgpt.fut:288:88-137
                
                double neg_res_119091 = -x_125064;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_125403 = 0; i_125403 < (int64_t) 4; i_125403++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_119075;
                    double r_119077 = 0.0;
                    
                    for (int64_t i_119076 = 0; i_119076 < (int64_t) 16; i_119076++) {
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_125399 = 0; i_125399 < (int64_t) 16; i_125399++) {
                            // futhark/microgpt.fut:4:11-25
                            
                            double zp_lhs_119098 = ((double *) mem_126525)[i_125407 * (int64_t) 16 + i_125399];
                            
                            // futhark/microgpt.fut:288:65-138
                            
                            double zp_res_119099 = neg_res_119091 + zp_lhs_119098;
                            
                            // futhark/microgpt.fut:288:58-138
                            
                            double exp_res_119100 = futrts_exp64(zp_res_119099);
                            
                            ((double *) mem_126595)[i_125399] = exp_res_119100;
                        }
                        // futhark/microgpt.fut:289:6-16
                        
                        double zt_lhs_119102 = ((double *) mem_126595)[i_119076];
                        
                        // futhark/microgpt.fut:71:13-49
                        
                        double defunc_0_lifted_lambda_res_119103;
                        double r_119105 = 0.0;
                        
                        for (int64_t i_119104 = 0; i_119104 < (int64_t) 16; i_119104++) {
                            // futhark/microgpt.fut:289:51-61
                            
                            double lifted_lambda_res_119106 = ((double *) mem_126595)[i_119104];
                            
                            // futhark/microgpt.fut:71:40-49
                            
                            double zp_res_119107 = r_119105 + lifted_lambda_res_119106;
                            double r_tmp_128090 = zp_res_119107;
                            
                            r_119105 = r_tmp_128090;
                        }
                        defunc_0_lifted_lambda_res_119103 = r_119105;
                        // futhark/microgpt.fut:289:22-62
                        
                        double zs_res_119108 = 1.0 / defunc_0_lifted_lambda_res_119103;
                        
                        // futhark/microgpt.fut:289:6-62
                        
                        double zt_res_119109 = zt_lhs_119102 * zs_res_119108;
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_119110 = ((double *) mem_126417)[i_125417 * (int64_t) 64 + i_119076 * (int64_t) 4 + i_125403];
                        
                        // futhark/microgpt.fut:289:17-94
                        
                        double zt_res_119111 = zt_res_119109 * zt_rhs_119110;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_119112 = r_119077 + zt_res_119111;
                        double r_tmp_128088 = zp_res_119112;
                        
                        r_119077 = r_tmp_128088;
                    }
                    defunc_0_lifted_lambda_res_119075 = r_119077;
                    ((double *) mem_126591)[i_125403] = defunc_0_lifted_lambda_res_119075;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_126586, i_125407 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126591, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_126498, i_125417 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_126522, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_126499, i_125417 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_126523, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_126500, i_125417 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_126524, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_126501, i_125417 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_126586, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125428 = 0; i_125428 < (int64_t) 16; i_125428++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125424 = 0; i_125424 < (int64_t) 16; i_125424++) {
                // futhark/microgpt.fut:290:52-55
                
                int64_t tmp_116285 = sdiv64(i_125424, (int64_t) 4);
                
                // futhark/microgpt.fut:290:41-57
                
                bool x_116286 = sle64((int64_t) 0, tmp_116285);
                
                // futhark/microgpt.fut:290:41-57
                
                bool y_116287 = slt64(tmp_116285, (int64_t) 4);
                
                // futhark/microgpt.fut:290:41-57
                
                bool bounds_check_116288 = x_116286 && y_116287;
                
                // futhark/microgpt.fut:290:41-57
                
                bool index_certs_116289;
                
                if (!bounds_check_116288) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_116285, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:290:41-57\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:290:12-78\n   #6  futhark/microgpt.fut:553:5-76\n   #7  futhark/microgpt.fut:570:26-576:31\n   #8  futhark/microgpt.fut:604:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:290:72-75
                
                int64_t tmp_116290 = smod64(i_125424, (int64_t) 4);
                
                // futhark/microgpt.fut:290:41-77
                
                bool x_116291 = sle64((int64_t) 0, tmp_116290);
                
                // futhark/microgpt.fut:290:41-77
                
                bool y_116292 = slt64(tmp_116290, (int64_t) 4);
                
                // futhark/microgpt.fut:290:41-77
                
                bool bounds_check_116293 = x_116291 && y_116292;
                
                // futhark/microgpt.fut:290:41-77
                
                bool index_certs_116294;
                
                if (!bounds_check_116293) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_116290, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:290:41-77\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:290:12-78\n   #6  futhark/microgpt.fut:553:5-76\n   #7  futhark/microgpt.fut:570:26-576:31\n   #8  futhark/microgpt.fut:604:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_116295 = ((double *) mem_126501)[tmp_116285 * (int64_t) 64 + i_125428 * (int64_t) 4 + tmp_116290];
                
                ((double *) mem_126634)[i_125424] = lifted_lambda_res_116295;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126629, i_125428 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126634, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125436 = 0; i_125436 < (int64_t) 16; i_125436++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125432 = 0; i_125432 < (int64_t) 16; i_125432++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_116310;
                double r_116312 = 0.0;
                
                for (int64_t i_116311 = 0; i_116311 < (int64_t) 16; i_116311++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_116313 = ((double *) mem_param_126187.mem)[i_125432 * (int64_t) 16 + i_116311];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_116314 = ((double *) mem_126629)[i_125436 * (int64_t) 16 + i_116311];
                    
                    // futhark/microgpt.fut:291:63-103
                    
                    double zt_res_116315 = zt_lhs_116313 * zt_rhs_116314;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_116316 = r_116312 + zt_res_116315;
                    double r_tmp_128095 = zp_res_116316;
                    
                    r_116312 = r_tmp_128095;
                }
                defunc_0_lifted_lambda_res_116310 = r_116312;
                ((double *) mem_126650)[i_125432] = defunc_0_lifted_lambda_res_116310;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126645, i_125436 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126650, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125444 = 0; i_125444 < (int64_t) 16; i_125444++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125440 = 0; i_125440 < (int64_t) 16; i_125440++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_116331 = ((double *) mem_126645)[i_125444 * (int64_t) 16 + i_125440];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_116332 = ((double *) mem_126318)[i_125444 * (int64_t) 16 + i_125440];
                
                // futhark/microgpt.fut:292:42-80
                
                double zp_res_116333 = zp_lhs_116331 + zp_rhs_116332;
                
                ((double *) mem_126666)[i_125440] = zp_res_116333;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126661, i_125444 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126666, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125454 = 0; i_125454 < (int64_t) 16; i_125454++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_119232;
            double r_119234 = 0.0;
            
            for (int64_t i_119233 = 0; i_119233 < (int64_t) 16; i_119233++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_119235 = ((double *) mem_126661)[i_125454 * (int64_t) 16 + i_119233];
                
                // futhark/microgpt.fut:293:105-144
                
                double zt_res_119236 = zt_lhs_119235 * zt_lhs_119235;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_119237 = r_119234 + zt_res_119236;
                double r_tmp_128100 = zp_res_119237;
                
                r_119234 = r_tmp_128100;
            }
            defunc_0_lifted_lambda_res_119232 = r_119234;
            // futhark/microgpt.fut:293:84-162
            
            double zs_res_119238 = defunc_0_lifted_lambda_res_119232 / 16.0;
            
            // futhark/microgpt.fut:293:148-191
            
            double zp_res_119239 = 1.0e-5 + zs_res_119238;
            
            // futhark/microgpt.fut:293:74-191
            
            double sqrt_res_119240 = futrts_sqrt64(zp_res_119239);
            
            // futhark/microgpt.fut:293:65-193
            
            double zs_res_119241 = 1.0 / sqrt_res_119240;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125448 = 0; i_125448 < (int64_t) 16; i_125448++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_119248 = ((double *) mem_126661)[i_125454 * (int64_t) 16 + i_125448];
                
                // futhark/microgpt.fut:293:42-193
                
                double zt_res_119249 = zs_res_119241 * zt_lhs_119248;
                
                ((double *) mem_126686)[i_125448] = zt_res_119249;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_119257;
            double r_119259 = 0.0;
            
            for (int64_t i_119258 = 0; i_119258 < (int64_t) 16; i_119258++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_119260 = ((double *) mem_126661)[i_125454 * (int64_t) 16 + i_119258];
                
                // futhark/microgpt.fut:314:68-111
                
                double zt_res_119261 = zt_lhs_119260 * zt_lhs_119260;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_119262 = r_119259 + zt_res_119261;
                double r_tmp_128102 = zp_res_119262;
                
                r_119259 = r_tmp_128102;
            }
            defunc_0_lifted_lambda_res_119257 = r_119259;
            // futhark/microgpt.fut:314:46-129
            
            double zs_res_119263 = defunc_0_lifted_lambda_res_119257 / 16.0;
            
            ((double *) mem_126677)[i_125454] = zs_res_119263;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126678, i_125454 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126686, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125463 = 0; i_125463 < (int64_t) 16; i_125463++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125459 = 0; i_125459 < (int64_t) 64; i_125459++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_116375;
                double r_116377 = 0.0;
                
                for (int64_t i_116376 = 0; i_116376 < (int64_t) 16; i_116376++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_116378 = ((double *) mem_param_126203.mem)[i_125459 * (int64_t) 16 + i_116376];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_116379 = ((double *) mem_126678)[i_125463 * (int64_t) 16 + i_116376];
                    
                    // futhark/microgpt.fut:294:63-102
                    
                    double zt_res_116380 = zt_lhs_116378 * zt_rhs_116379;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_116381 = r_116377 + zt_res_116380;
                    double r_tmp_128105 = zp_res_116381;
                    
                    r_116377 = r_tmp_128105;
                }
                defunc_0_lifted_lambda_res_116375 = r_116377;
                ((double *) mem_126705)[i_125459] = defunc_0_lifted_lambda_res_116375;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126700, i_125463 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126705, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125471 = 0; i_125471 < (int64_t) 16; i_125471++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125467 = 0; i_125467 < (int64_t) 64; i_125467++) {
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_116396 = ((double *) mem_126700)[i_125471 * (int64_t) 64 + i_125467];
                
                // futhark/microgpt.fut:295:41-69
                
                double max_res_116397 = fmax64(0.0, max_arg0_116396);
                
                ((double *) mem_126721)[i_125467] = max_res_116397;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126716, i_125471 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126721, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125479 = 0; i_125479 < (int64_t) 16; i_125479++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125475 = 0; i_125475 < (int64_t) 16; i_125475++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_116412;
                double r_116414 = 0.0;
                
                for (int64_t i_116413 = 0; i_116413 < (int64_t) 64; i_116413++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_116415 = ((double *) mem_param_126179.mem)[i_125475 * (int64_t) 64 + i_116413];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_116416 = ((double *) mem_126716)[i_125479 * (int64_t) 64 + i_116413];
                    
                    // futhark/microgpt.fut:296:63-104
                    
                    double zt_res_116417 = zt_lhs_116415 * zt_rhs_116416;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_116418 = r_116414 + zt_res_116417;
                    double r_tmp_128110 = zp_res_116418;
                    
                    r_116414 = r_tmp_128110;
                }
                defunc_0_lifted_lambda_res_116412 = r_116414;
                ((double *) mem_126737)[i_125475] = defunc_0_lifted_lambda_res_116412;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126732, i_125479 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126737, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125487 = 0; i_125487 < (int64_t) 16; i_125487++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125483 = 0; i_125483 < (int64_t) 16; i_125483++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_116433 = ((double *) mem_126732)[i_125487 * (int64_t) 16 + i_125483];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_116434 = ((double *) mem_126661)[i_125487 * (int64_t) 16 + i_125483];
                
                // futhark/microgpt.fut:297:42-81
                
                double zp_res_116435 = zp_lhs_116433 + zp_rhs_116434;
                
                ((double *) mem_126753)[i_125483] = zp_res_116435;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126748, i_125487 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126753, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125495 = 0; i_125495 < (int64_t) 16; i_125495++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125491 = 0; i_125491 < (int64_t) 27; i_125491++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_116450;
                double r_116452 = 0.0;
                
                for (int64_t i_116451 = 0; i_116451 < (int64_t) 16; i_116451++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_116453 = ((double *) mem_param_126211.mem)[i_125491 * (int64_t) 16 + i_116451];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_116454 = ((double *) mem_126748)[i_125495 * (int64_t) 16 + i_116451];
                    
                    // futhark/microgpt.fut:298:63-103
                    
                    double zt_res_116455 = zt_lhs_116453 * zt_rhs_116454;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_116456 = r_116452 + zt_res_116455;
                    double r_tmp_128115 = zp_res_116456;
                    
                    r_116452 = r_tmp_128115;
                }
                defunc_0_lifted_lambda_res_116450 = r_116452;
                ((double *) mem_126769)[i_125491] = defunc_0_lifted_lambda_res_116450;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126764, i_125495 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126769, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125519 = 0; i_125519 < (int64_t) 16; i_125519++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_125131;
            double defunc_0_reduce_res_125132;
            double redout_125508;
            double redout_125509;
            
            redout_125508 = -INFINITY;
            redout_125509 = -INFINITY;
            for (int64_t i_125511 = 0; i_125511 < (int64_t) 27; i_125511++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_121958 = ((double *) mem_126764)[i_125519 * (int64_t) 27 + i_125511];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_125505 = 0; i_125505 < (int64_t) 27; i_125505++) {
                    // futhark/microgpt.fut:304:55-305:137
                    
                    bool cond_121967 = i_125505 == i_125511;
                    
                    // futhark/microgpt.fut:304:55-305:137
                    
                    double lifted_lambda_res_121968;
                    
                    if (cond_121967) {
                        // futhark/microgpt.fut:115:13-33
                        
                        double defunc_0_reduce_res_125087;
                        double redout_125497 = -INFINITY;
                        
                        for (int64_t i_125498 = 0; i_125498 < (int64_t) 27; i_125498++) {
                            // futhark/microgpt.fut:4:11-25
                            
                            double lifted_lambda_res_125093 = ((double *) mem_126764)[i_125519 * (int64_t) 27 + i_125498];
                            
                            // futhark/microgpt.fut:115:13-33
                            
                            double max_res_125096 = fmax64(lifted_lambda_res_125093, redout_125497);
                            double redout_tmp_128123 = max_res_125096;
                            
                            redout_125497 = redout_tmp_128123;
                        }
                        defunc_0_reduce_res_125087 = redout_125497;
                        // futhark/microgpt.fut:304:145-194
                        
                        double neg_res_125098 = -defunc_0_reduce_res_125087;
                        
                        // futhark/microgpt.fut:4:11-25
                        if (mem_126805_cached_sizze_128521 < (int64_t) 216) {
                            err = lexical_realloc(ctx, &mem_126805, &mem_126805_cached_sizze_128521, (int64_t) 216);
                            if (err != FUTHARK_SUCCESS)
                                goto cleanup;
                        }
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_125501 = 0; i_125501 < (int64_t) 27; i_125501++) {
                            // futhark/microgpt.fut:4:11-25
                            
                            double zp_lhs_125105 = ((double *) mem_126764)[i_125519 * (int64_t) 27 + i_125501];
                            
                            // futhark/microgpt.fut:304:122-195
                            
                            double zp_res_125106 = neg_res_125098 + zp_lhs_125105;
                            
                            // futhark/microgpt.fut:304:115-195
                            
                            double exp_res_125107 = futrts_exp64(zp_res_125106);
                            
                            ((double *) mem_126805)[i_125501] = exp_res_125107;
                        }
                        // futhark/microgpt.fut:4:11-25
                        
                        double zt_rhs_125114 = ((double *) mem_126285)[i_125519 * (int64_t) 27 + i_125511];
                        
                        // futhark/microgpt.fut:305:7-49
                        
                        double zt_res_125115 = -6.25e-2 * zt_rhs_125114;
                        
                        // futhark/microgpt.fut:305:65-75
                        
                        double zt_lhs_125120 = ((double *) mem_126805)[i_125505];
                        
                        // futhark/microgpt.fut:71:13-49
                        
                        double defunc_0_lifted_lambda_res_125121;
                        double r_125123 = 0.0;
                        
                        for (int64_t i_125122 = 0; i_125122 < (int64_t) 27; i_125122++) {
                            // futhark/microgpt.fut:305:110-120
                            
                            double lifted_lambda_res_125124 = ((double *) mem_126805)[i_125122];
                            
                            // futhark/microgpt.fut:71:40-49
                            
                            double zp_res_125125 = r_125123 + lifted_lambda_res_125124;
                            double r_tmp_128125 = zp_res_125125;
                            
                            r_125123 = r_tmp_128125;
                        }
                        defunc_0_lifted_lambda_res_125121 = r_125123;
                        // futhark/microgpt.fut:305:81-121
                        
                        double zs_res_125126 = 1.0 / defunc_0_lifted_lambda_res_125121;
                        
                        // futhark/microgpt.fut:305:65-121
                        
                        double zt_res_125127 = zt_lhs_125120 * zs_res_125126;
                        
                        // futhark/microgpt.fut:305:56-121
                        
                        double zs_res_125128 = 1.0 / zt_res_125127;
                        
                        // futhark/microgpt.fut:305:25-121
                        
                        double zt_res_125129 = zt_res_125115 * zs_res_125128;
                        
                        lifted_lambda_res_121968 = zt_res_125129;
                    } else {
                        lifted_lambda_res_121968 = 0.0;
                    }
                    ((double *) mem_126801)[i_125505] = lifted_lambda_res_121968;
                }
                // futhark/microgpt.fut:115:13-33
                
                double max_res_119389 = fmax64(lifted_lambda_res_121958, redout_125508);
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_119471 = fmax64(lifted_lambda_res_121958, redout_125509);
                
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_126796, i_125511 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126801, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
                
                double redout_tmp_128119 = max_res_119389;
                double redout_tmp_128120 = max_res_119471;
                
                redout_125508 = redout_tmp_128119;
                redout_125509 = redout_tmp_128120;
            }
            defunc_0_reduce_res_125131 = redout_125508;
            defunc_0_reduce_res_125132 = redout_125509;
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_128126 = 0; nest_i_128126 < (int64_t) 27; nest_i_128126++) {
                ((double *) mem_126782)[i_125519 * (int64_t) 27 + nest_i_128126] = defunc_0_reduce_res_125131;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_128127 = 0; nest_i_128127 < (int64_t) 27; nest_i_128127++) {
                ((double *) mem_126780)[i_125519 * (int64_t) 27 + nest_i_128127] = defunc_0_reduce_res_125132;
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_126781, i_125519 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_126796, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125533 = 0; i_125533 < (int64_t) 16; i_125533++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125529 = 0; i_125529 < (int64_t) 27; i_125529++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_116492 = ((double *) mem_126782)[i_125533 * (int64_t) 27 + i_125529];
                
                // futhark/microgpt.fut:302:85-108
                
                double neg_res_116493 = -neg_arg0_116492;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_125525 = 0; i_125525 < (int64_t) 27; i_125525++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_116500 = ((double *) mem_126764)[i_125533 * (int64_t) 27 + i_125525];
                    
                    // futhark/microgpt.fut:302:62-108
                    
                    double zp_res_116501 = neg_res_116493 + zp_lhs_116500;
                    
                    // futhark/microgpt.fut:302:55-108
                    
                    double exp_res_116502 = futrts_exp64(zp_res_116501);
                    
                    ((double *) mem_126845)[i_125525] = exp_res_116502;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_126840, i_125529 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126845, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_126834, i_125533 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_126840, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125541 = 0; i_125541 < (int64_t) 16; i_125541++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125537 = 0; i_125537 < (int64_t) 27; i_125537++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_116518;
                double r_116520 = 0.0;
                
                for (int64_t i_116519 = 0; i_116519 < (int64_t) 27; i_116519++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_116521 = ((double *) mem_126834)[i_125541 * (int64_t) 729 + i_125537 * (int64_t) 27 + i_116519];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_116522 = r_116520 + lifted_lambda_res_116521;
                    double r_tmp_128133 = zp_res_116522;
                    
                    r_116520 = r_tmp_128133;
                }
                defunc_0_lifted_lambda_res_116518 = r_116520;
                ((double *) mem_126866)[i_125537] = defunc_0_lifted_lambda_res_116518;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126861, i_125541 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126866, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125553 = 0; i_125553 < (int64_t) 16; i_125553++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125549 = 0; i_125549 < (int64_t) 27; i_125549++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_116597 = ((double *) mem_126861)[i_125553 * (int64_t) 27 + i_125549];
                
                // futhark/microgpt.fut:306:86-111
                
                double zs_res_116598 = 1.0 / zs_rhs_116597;
                
                // futhark/microgpt.fut:306:217-256
                
                double zt_res_116599 = zs_rhs_116597 * zs_rhs_116597;
                
                // futhark/microgpt.fut:306:208-256
                
                double zs_res_116600 = 1.0 / zt_res_116599;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_116601;
                double r_116603 = 0.0;
                
                for (int64_t i_116602 = 0; i_116602 < (int64_t) 27; i_116602++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_116604 = ((double *) mem_126781)[i_125553 * (int64_t) 729 + i_125549 * (int64_t) 27 + i_116602];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_116605 = ((double *) mem_126834)[i_125553 * (int64_t) 729 + i_125549 * (int64_t) 27 + i_116602];
                    
                    // futhark/microgpt.fut:306:148-201
                    
                    double zt_res_116606 = zt_lhs_116604 * zt_rhs_116605;
                    
                    // futhark/microgpt.fut:306:173-256
                    
                    double zt_res_116607 = zs_res_116600 * zt_res_116606;
                    
                    // futhark/microgpt.fut:306:140-256
                    
                    double neg_res_116608 = -zt_res_116607;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_116609 = r_116603 + neg_res_116608;
                    double r_tmp_128136 = zp_res_116609;
                    
                    r_116603 = r_tmp_128136;
                }
                defunc_0_lifted_lambda_res_116601 = r_116603;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_125545 = 0; i_125545 < (int64_t) 27; i_125545++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_116616 = ((double *) mem_126781)[i_125553 * (int64_t) 729 + i_125549 * (int64_t) 27 + i_125545];
                    
                    // futhark/microgpt.fut:306:56-111
                    
                    double zt_res_116617 = zs_res_116598 * zt_lhs_116616;
                    
                    // futhark/microgpt.fut:306:81-261
                    
                    double zp_res_116618 = defunc_0_lifted_lambda_res_116601 + zt_res_116617;
                    
                    ((double *) mem_126888)[i_125545] = zp_res_116618;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_126883, i_125549 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126888, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_126877, i_125553 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_126883, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125561 = 0; i_125561 < (int64_t) 16; i_125561++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125557 = 0; i_125557 < (int64_t) 27; i_125557++) {
                double f_elem_116652 = ((double *) mem_126764)[i_125561 * (int64_t) 27 + i_125557];
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_116657;
                double r_116659 = 0.0;
                
                for (int64_t i_116658 = 0; i_116658 < (int64_t) 27; i_116658++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double neg_arg0_116660 = ((double *) mem_126782)[i_125561 * (int64_t) 27 + i_116658];
                    
                    // futhark/microgpt.fut:308:88-111
                    
                    double neg_res_116661 = -neg_arg0_116660;
                    
                    // futhark/microgpt.fut:308:65-111
                    
                    double zp_res_116662 = f_elem_116652 + neg_res_116661;
                    
                    // futhark/microgpt.fut:308:58-111
                    
                    double exp_res_116663 = futrts_exp64(zp_res_116662);
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_116664 = ((double *) mem_126877)[i_125561 * (int64_t) 729 + i_116658 * (int64_t) 27 + i_125557];
                    
                    // futhark/microgpt.fut:308:58-143
                    
                    double zt_res_116665 = exp_res_116663 * zt_rhs_116664;
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_116666;
                    double r_116668 = 0.0;
                    
                    for (int64_t i_116667 = 0; i_116667 < (int64_t) 27; i_116667++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zp_lhs_116669 = ((double *) mem_126764)[i_125561 * (int64_t) 27 + i_116667];
                        
                        // futhark/microgpt.fut:308:188-234
                        
                        double zp_res_116670 = neg_res_116661 + zp_lhs_116669;
                        
                        // futhark/microgpt.fut:308:181-234
                        
                        double exp_res_116671 = futrts_exp64(zp_res_116670);
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_116672 = ((double *) mem_126877)[i_125561 * (int64_t) 729 + i_116658 * (int64_t) 27 + i_116667];
                        
                        // futhark/microgpt.fut:308:181-266
                        
                        double zt_res_116673 = exp_res_116671 * zt_rhs_116672;
                        
                        // futhark/microgpt.fut:308:173-266
                        
                        double neg_res_116674 = -zt_res_116673;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_116675 = r_116668 + neg_res_116674;
                        double r_tmp_128141 = zp_res_116675;
                        
                        r_116668 = r_tmp_128141;
                    }
                    defunc_0_lifted_lambda_res_116666 = r_116668;
                    // futhark/microgpt.fut:71:46-49
                    
                    double neg_arg0_116676 = ((double *) mem_126780)[i_125561 * (int64_t) 27 + i_116658];
                    
                    // futhark/microgpt.fut:308:334-357
                    
                    double neg_res_116677 = -neg_arg0_116676;
                    
                    // futhark/microgpt.fut:308:311-357
                    
                    double zp_res_116678 = f_elem_116652 + neg_res_116677;
                    
                    // futhark/microgpt.fut:308:304-357
                    
                    double neg_res_116679 = -zp_res_116678;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_116680 = fmax64(0.0, neg_res_116679);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_116681 = fsignum64(max_res_116680);
                    
                    // futhark/microgpt.fut:308:285-360
                    
                    double neg_res_116682 = -sgn_res_116681;
                    
                    // futhark/microgpt.fut:308:276-361
                    
                    double zp_res_116683 = 1.0 + neg_res_116682;
                    
                    // futhark/microgpt.fut:308:152-361
                    
                    double zt_res_116684 = defunc_0_lifted_lambda_res_116666 * zp_res_116683;
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_116685;
                    double r_116687 = 0.0;
                    
                    for (int64_t i_116686 = 0; i_116686 < (int64_t) 27; i_116686++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zp_lhs_116688 = ((double *) mem_126764)[i_125561 * (int64_t) 27 + i_116686];
                        
                        // futhark/microgpt.fut:308:435-481
                        
                        double zp_res_116689 = neg_res_116677 + zp_lhs_116688;
                        
                        // futhark/microgpt.fut:308:428-481
                        
                        double neg_res_116690 = -zp_res_116689;
                        
                        // futhark/microgpt.fut:110:42-54
                        
                        double max_res_116691 = fmax64(0.0, neg_res_116690);
                        
                        // futhark/microgpt.fut:110:35-54
                        
                        double sgn_res_116692 = fsignum64(max_res_116691);
                        
                        // futhark/microgpt.fut:308:409-484
                        
                        double neg_res_116693 = -sgn_res_116692;
                        
                        // futhark/microgpt.fut:308:400-485
                        
                        double zp_res_116694 = 1.0 + neg_res_116693;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_116695 = r_116687 + zp_res_116694;
                        double r_tmp_128142 = zp_res_116695;
                        
                        r_116687 = r_tmp_128142;
                    }
                    defunc_0_lifted_lambda_res_116685 = r_116687;
                    // futhark/microgpt.fut:308:370-488
                    
                    double zs_res_116696 = 1.0 / defunc_0_lifted_lambda_res_116685;
                    
                    // futhark/microgpt.fut:308:271-488
                    
                    double zt_res_116697 = zt_res_116684 * zs_res_116696;
                    
                    // futhark/microgpt.fut:308:115-488
                    
                    double zp_res_116698 = zt_res_116665 + zt_res_116697;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_116699 = r_116659 + zp_res_116698;
                    double r_tmp_128140 = zp_res_116699;
                    
                    r_116659 = r_tmp_128140;
                }
                defunc_0_lifted_lambda_res_116657 = r_116659;
                ((double *) mem_126909)[i_125557] = defunc_0_lifted_lambda_res_116657;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126904, i_125561 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126909, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125569 = 0; i_125569 < (int64_t) 16; i_125569++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125565 = 0; i_125565 < (int64_t) 16; i_125565++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_116714;
                double r_116716 = 0.0;
                
                for (int64_t i_116715 = 0; i_116715 < (int64_t) 27; i_116715++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_116717 = ((double *) mem_126904)[i_125569 * (int64_t) 27 + i_116715];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_116718 = ((double *) mem_param_126211.mem)[i_116715 * (int64_t) 16 + i_125565];
                    
                    // futhark/microgpt.fut:309:63-103
                    
                    double zt_res_116719 = zt_lhs_116717 * zt_rhs_116718;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_116720 = r_116716 + zt_res_116719;
                    double r_tmp_128145 = zp_res_116720;
                    
                    r_116716 = r_tmp_128145;
                }
                defunc_0_lifted_lambda_res_116714 = r_116716;
                ((double *) mem_126925)[i_125565] = defunc_0_lifted_lambda_res_116714;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126920, i_125569 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126925, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125577 = 0; i_125577 < (int64_t) 16; i_125577++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125573 = 0; i_125573 < (int64_t) 16; i_125573++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_116735 = ((double *) mem_126920)[i_125577 * (int64_t) 16 + i_125573];
                
                ((double *) mem_126941)[i_125573] = lifted_lambda_res_116735;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126936, i_125577 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126941, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125590 = 0; i_125590 < (int64_t) 16; i_125590++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125583 = 0; i_125583 < (int64_t) 64; i_125583++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_122289;
                double r_122291 = 0.0;
                
                for (int64_t i_122290 = 0; i_122290 < (int64_t) 16; i_122290++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_122292 = ((double *) mem_126936)[i_125590 * (int64_t) 16 + i_122290];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_122293 = ((double *) mem_param_126179.mem)[i_122290 * (int64_t) 64 + i_125583];
                    
                    // futhark/microgpt.fut:311:63-104
                    
                    double zt_res_122294 = zt_lhs_122292 * zt_rhs_122293;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_122295 = r_122291 + zt_res_122294;
                    double r_tmp_128152 = zp_res_122295;
                    
                    r_122291 = r_tmp_128152;
                }
                defunc_0_lifted_lambda_res_122289 = r_122291;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_122302;
                double r_122304 = 0.0;
                
                for (int64_t i_122303 = 0; i_122303 < (int64_t) 16; i_122303++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_122305 = ((double *) mem_126936)[i_122303 * (int64_t) 16 + i_125590];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_122306 = ((double *) mem_126716)[i_122303 * (int64_t) 64 + i_125583];
                    
                    // futhark/microgpt.fut:377:69-112
                    
                    double zt_res_122307 = zt_lhs_122305 * zt_rhs_122306;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_122308 = r_122304 + zt_res_122307;
                    double r_tmp_128153 = zp_res_122308;
                    
                    r_122304 = r_tmp_128153;
                }
                defunc_0_lifted_lambda_res_122302 = r_122304;
                ((double *) mem_126962)[i_125583] = defunc_0_lifted_lambda_res_122302;
                ((double *) mem_126963)[i_125583] = defunc_0_lifted_lambda_res_122289;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126952, i_125590 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126962, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126953, i_125590 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126963, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125599 = 0; i_125599 < (int64_t) 16; i_125599++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125595 = 0; i_125595 < (int64_t) 64; i_125595++) {
                // futhark/microgpt.fut:4:11-25
                
                double indicatorp_arg0_116771 = ((double *) mem_126700)[i_125599 * (int64_t) 64 + i_125595];
                
                // futhark/microgpt.fut:110:42-54
                
                double max_res_116772 = fmax64(0.0, indicatorp_arg0_116771);
                
                // futhark/microgpt.fut:110:35-54
                
                double sgn_res_116773 = fsignum64(max_res_116772);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_116774 = ((double *) mem_126953)[i_125599 * (int64_t) 64 + i_125595];
                
                // futhark/microgpt.fut:312:43-94
                
                double zt_res_116775 = sgn_res_116773 * zt_rhs_116774;
                
                ((double *) mem_126989)[i_125595] = zt_res_116775;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_126984, i_125599 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_126989, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125607 = 0; i_125607 < (int64_t) 16; i_125607++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125603 = 0; i_125603 < (int64_t) 16; i_125603++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_116790;
                double r_116792 = 0.0;
                
                for (int64_t i_116791 = 0; i_116791 < (int64_t) 64; i_116791++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_116793 = ((double *) mem_126984)[i_125607 * (int64_t) 64 + i_116791];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_116794 = ((double *) mem_param_126203.mem)[i_116791 * (int64_t) 16 + i_125603];
                    
                    // futhark/microgpt.fut:313:63-102
                    
                    double zt_res_116795 = zt_lhs_116793 * zt_rhs_116794;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_116796 = r_116792 + zt_res_116795;
                    double r_tmp_128158 = zp_res_116796;
                    
                    r_116792 = r_tmp_128158;
                }
                defunc_0_lifted_lambda_res_116790 = r_116792;
                ((double *) mem_127005)[i_125603] = defunc_0_lifted_lambda_res_116790;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127000, i_125607 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127005, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125615 = 0; i_125615 < (int64_t) 16; i_125615++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125611 = 0; i_125611 < (int64_t) 16; i_125611++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_116835 = ((double *) mem_127000)[i_125615 * (int64_t) 16 + i_125611];
                
                ((double *) mem_127021)[i_125611] = lifted_lambda_res_116835;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127016, i_125615 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127021, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125621 = 0; i_125621 < (int64_t) 16; i_125621++) {
            // futhark/microgpt.fut:315:46-57
            
            double zp_lhs_118206 = ((double *) mem_126677)[i_125621];
            
            // futhark/microgpt.fut:315:46-85
            
            double zp_res_118207 = 1.0e-5 + zp_lhs_118206;
            
            // futhark/microgpt.fut:315:38-85
            
            double sqrt_res_118208 = futrts_sqrt64(zp_res_118207);
            
            // futhark/microgpt.fut:317:128-155
            
            double zt_res_118216 = sqrt_res_118208 * sqrt_res_118208;
            
            // futhark/microgpt.fut:317:119-155
            
            double zs_res_118217 = 1.0 / zt_res_118216;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_118218;
            double r_118220 = 0.0;
            
            for (int64_t i_118219 = 0; i_118219 < (int64_t) 16; i_118219++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_118221 = ((double *) mem_127016)[i_125621 * (int64_t) 16 + i_118219];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_118222 = ((double *) mem_126661)[i_125621 * (int64_t) 16 + i_118219];
                
                // futhark/microgpt.fut:317:69-112
                
                double zt_res_118223 = zt_lhs_118221 * zt_rhs_118222;
                
                // futhark/microgpt.fut:317:89-155
                
                double zt_res_118224 = zs_res_118217 * zt_res_118223;
                
                // futhark/microgpt.fut:317:61-155
                
                double neg_res_118225 = -zt_res_118224;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_118226 = r_118220 + neg_res_118225;
                double r_tmp_128163 = zp_res_118226;
                
                r_118220 = r_tmp_128163;
            }
            defunc_0_lifted_lambda_res_118218 = r_118220;
            // futhark/microgpt.fut:317:181-244
            
            double zt_res_118230 = 2.0 * sqrt_res_118208;
            
            // futhark/microgpt.fut:317:167-244
            
            double zs_res_118231 = 1.0 / zt_res_118230;
            
            // futhark/microgpt.fut:317:39-244
            
            double zt_res_118232 = defunc_0_lifted_lambda_res_118218 * zs_res_118231;
            
            ((double *) mem_127032)[i_125621] = zt_res_118232;
            ((double *) mem_127033)[i_125621] = sqrt_res_118208;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125630 = 0; i_125630 < (int64_t) 16; i_125630++) {
            // futhark/microgpt.fut:318:96-107
            
            double zs_rhs_116869 = ((double *) mem_127033)[i_125630];
            
            // futhark/microgpt.fut:318:88-107
            
            double zs_res_116870 = 1.0 / zs_rhs_116869;
            
            // futhark/microgpt.fut:318:117-128
            
            double zs_lhs_116871 = ((double *) mem_127032)[i_125630];
            
            // futhark/microgpt.fut:318:117-143
            
            double zs_res_116872 = zs_lhs_116871 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125626 = 0; i_125626 < (int64_t) 16; i_125626++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_116879 = ((double *) mem_126936)[i_125630 * (int64_t) 16 + i_125626];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_116880 = ((double *) mem_127016)[i_125630 * (int64_t) 16 + i_125626];
                
                // futhark/microgpt.fut:318:63-107
                
                double zt_res_116881 = zs_res_116870 * zt_lhs_116880;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_116882 = ((double *) mem_126661)[i_125630 * (int64_t) 16 + i_125626];
                
                // futhark/microgpt.fut:318:129-168
                
                double zt_res_116883 = zs_res_116872 * zt_rhs_116882;
                
                // futhark/microgpt.fut:318:145-227
                
                double zp_res_116884 = zt_res_116883 + zt_res_116883;
                
                // futhark/microgpt.fut:318:83-227
                
                double zp_res_116885 = zt_res_116881 + zp_res_116884;
                
                // futhark/microgpt.fut:318:37-227
                
                double zp_res_116886 = zp_lhs_116879 + zp_res_116885;
                
                ((double *) mem_127051)[i_125626] = zp_res_116886;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127046, i_125630 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127051, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125638 = 0; i_125638 < (int64_t) 16; i_125638++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125634 = 0; i_125634 < (int64_t) 16; i_125634++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_116901 = ((double *) mem_127046)[i_125638 * (int64_t) 16 + i_125634];
                
                ((double *) mem_127067)[i_125634] = lifted_lambda_res_116901;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127062, i_125638 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127067, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125651 = 0; i_125651 < (int64_t) 16; i_125651++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125644 = 0; i_125644 < (int64_t) 16; i_125644++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_122333;
                double r_122335 = 0.0;
                
                for (int64_t i_122334 = 0; i_122334 < (int64_t) 16; i_122334++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_122336 = ((double *) mem_127062)[i_125651 * (int64_t) 16 + i_122334];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_122337 = ((double *) mem_param_126187.mem)[i_122334 * (int64_t) 16 + i_125644];
                    
                    // futhark/microgpt.fut:320:67-112
                    
                    double zt_res_122338 = zt_lhs_122336 * zt_rhs_122337;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_122339 = r_122335 + zt_res_122338;
                    double r_tmp_128172 = zp_res_122339;
                    
                    r_122335 = r_tmp_128172;
                }
                defunc_0_lifted_lambda_res_122333 = r_122335;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_122346;
                double r_122348 = 0.0;
                
                for (int64_t i_122347 = 0; i_122347 < (int64_t) 16; i_122347++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_122349 = ((double *) mem_127062)[i_122347 * (int64_t) 16 + i_125651];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_122350 = ((double *) mem_126629)[i_122347 * (int64_t) 16 + i_125644];
                    
                    // futhark/microgpt.fut:375:68-112
                    
                    double zt_res_122351 = zt_lhs_122349 * zt_rhs_122350;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_122352 = r_122348 + zt_res_122351;
                    double r_tmp_128173 = zp_res_122352;
                    
                    r_122348 = r_tmp_128173;
                }
                defunc_0_lifted_lambda_res_122346 = r_122348;
                ((double *) mem_127088)[i_125644] = defunc_0_lifted_lambda_res_122346;
                ((double *) mem_127089)[i_125644] = defunc_0_lifted_lambda_res_122333;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127078, i_125651 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127088, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127079, i_125651 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127089, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125664 = 0; i_125664 < (int64_t) 4; i_125664++) {
            // futhark/microgpt.fut:321:74-77
            
            int64_t zp_lhs_116927 = mul64((int64_t) 4, i_125664);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125660 = 0; i_125660 < (int64_t) 16; i_125660++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_125656 = 0; i_125656 < (int64_t) 4; i_125656++) {
                    // futhark/microgpt.fut:321:79-87
                    
                    int64_t tmp_116936 = add64(zp_lhs_116927, i_125656);
                    
                    // futhark/microgpt.fut:321:52-89
                    
                    bool x_116937 = sle64((int64_t) 0, tmp_116936);
                    
                    // futhark/microgpt.fut:321:52-89
                    
                    bool y_116938 = slt64(tmp_116936, (int64_t) 16);
                    
                    // futhark/microgpt.fut:321:52-89
                    
                    bool bounds_check_116939 = x_116937 && y_116938;
                    
                    // futhark/microgpt.fut:321:52-89
                    
                    bool index_certs_116940;
                    
                    if (!bounds_check_116939) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_116936, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:321:52-89\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:321:13-90\n   #9  futhark/microgpt.fut:553:5-76\n   #10 futhark/microgpt.fut:570:26-576:31\n   #11 futhark/microgpt.fut:604:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_116941 = ((double *) mem_127079)[i_125660 * (int64_t) 16 + tmp_116936];
                    
                    ((double *) mem_127121)[i_125656] = lifted_lambda_res_116941;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_127116, i_125660 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127121, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_127110, i_125664 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_127116, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125770 = 0; i_125770 < (int64_t) 4; i_125770++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125760 = 0; i_125760 < (int64_t) 16; i_125760++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_125750 = 0; i_125750 < (int64_t) 4; i_125750++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_123290;
                    double r_123292 = 0.0;
                    
                    for (int64_t i_123291 = 0; i_123291 < (int64_t) 16; i_123291++) {
                        // futhark/microgpt.fut:115:13-33
                        
                        double defunc_0_reduce_res_125153;
                        double redout_125666 = -INFINITY;
                        
                        for (int64_t i_125667 = 0; i_125667 < (int64_t) 16; i_125667++) {
                            // futhark/microgpt.fut:4:11-25
                            
                            double lifted_lambda_res_123696 = ((double *) mem_126500)[i_125770 * (int64_t) 256 + i_123291 * (int64_t) 16 + i_125667];
                            
                            // futhark/microgpt.fut:115:13-33
                            
                            double max_res_123305 = fmax64(lifted_lambda_res_123696, redout_125666);
                            double redout_tmp_128187 = max_res_123305;
                            
                            redout_125666 = redout_tmp_128187;
                        }
                        defunc_0_reduce_res_125153 = redout_125666;
                        // futhark/microgpt.fut:323:142-203
                        
                        double neg_res_123306 = -defunc_0_reduce_res_125153;
                        
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_125670 = 0; i_125670 < (int64_t) 16; i_125670++) {
                            // futhark/microgpt.fut:4:11-25
                            
                            double zp_lhs_123313 = ((double *) mem_126500)[i_125770 * (int64_t) 256 + i_123291 * (int64_t) 16 + i_125670];
                            
                            // futhark/microgpt.fut:323:108-204
                            
                            double zp_res_123314 = neg_res_123306 + zp_lhs_123313;
                            
                            // futhark/microgpt.fut:323:101-204
                            
                            double exp_res_123315 = futrts_exp64(zp_res_123314);
                            
                            ((double *) mem_127182)[i_125670] = exp_res_123315;
                        }
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_123317 = ((double *) mem_127110)[i_125770 * (int64_t) 64 + i_123291 * (int64_t) 4 + i_125750];
                        
                        // futhark/microgpt.fut:324:39-51
                        
                        double zt_lhs_123318 = ((double *) mem_127182)[i_125760];
                        
                        // futhark/microgpt.fut:71:13-49
                        
                        double defunc_0_lifted_lambda_res_123319;
                        double r_123321 = 0.0;
                        
                        for (int64_t i_123320 = 0; i_123320 < (int64_t) 16; i_123320++) {
                            // futhark/microgpt.fut:324:87-99
                            
                            double lifted_lambda_res_123322 = ((double *) mem_127182)[i_123320];
                            
                            // futhark/microgpt.fut:71:40-49
                            
                            double zp_res_123323 = r_123321 + lifted_lambda_res_123322;
                            double r_tmp_128189 = zp_res_123323;
                            
                            r_123321 = r_tmp_128189;
                        }
                        defunc_0_lifted_lambda_res_123319 = r_123321;
                        // futhark/microgpt.fut:324:57-100
                        
                        double zs_res_123324 = 1.0 / defunc_0_lifted_lambda_res_123319;
                        
                        // futhark/microgpt.fut:324:39-100
                        
                        double zt_res_123325 = zt_lhs_123318 * zs_res_123324;
                        
                        // futhark/microgpt.fut:324:5-100
                        
                        double zt_res_123326 = zt_lhs_123317 * zt_res_123325;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_123327 = r_123292 + zt_res_123326;
                        double r_tmp_128186 = zp_res_123327;
                        
                        r_123292 = r_tmp_128186;
                    }
                    defunc_0_lifted_lambda_res_123290 = r_123292;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_123334;
                    double r_123336 = 0.0;
                    
                    for (int64_t i_123335 = 0; i_123335 < (int64_t) 16; i_123335++) {
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_125684 = 0; i_125684 < (int64_t) 16; i_125684++) {
                            // futhark/microgpt.fut:115:13-33
                            
                            double defunc_0_reduce_res_125155;
                            double defunc_0_reduce_res_125156;
                            double redout_125673;
                            double redout_125674;
                            
                            redout_125673 = -INFINITY;
                            redout_125674 = -INFINITY;
                            for (int64_t i_125676 = 0; i_125676 < (int64_t) 16; i_125676++) {
                                // futhark/microgpt.fut:4:11-25
                                
                                double lifted_lambda_res_123932 = ((double *) mem_126499)[i_125770 * (int64_t) 256 + i_125684 * (int64_t) 16 + i_125676];
                                
                                // futhark/microgpt.fut:71:13-49
                                
                                double defunc_0_lifted_lambda_res_123943;
                                double r_123945 = 0.0;
                                
                                for (int64_t i_123944 = 0; i_123944 < (int64_t) 4; i_123944++) {
                                    // futhark/microgpt.fut:71:46-49
                                    
                                    double zt_lhs_123946 = ((double *) mem_127110)[i_125770 * (int64_t) 64 + i_125684 * (int64_t) 4 + i_123944];
                                    
                                    // futhark/microgpt.fut:71:46-49
                                    
                                    double zt_rhs_123947 = ((double *) mem_126417)[i_125770 * (int64_t) 64 + i_125676 * (int64_t) 4 + i_123944];
                                    
                                    // futhark/microgpt.fut:329:70-130
                                    
                                    double zt_res_123948 = zt_lhs_123946 * zt_rhs_123947;
                                    
                                    // futhark/microgpt.fut:71:40-49
                                    
                                    double zp_res_123949 = r_123945 + zt_res_123948;
                                    double r_tmp_128197 = zp_res_123949;
                                    
                                    r_123945 = r_tmp_128197;
                                }
                                defunc_0_lifted_lambda_res_123943 = r_123945;
                                // futhark/microgpt.fut:115:13-33
                                
                                double max_res_123786 = fmax64(lifted_lambda_res_123932, redout_125673);
                                
                                // futhark/microgpt.fut:115:13-33
                                
                                double max_res_123828 = fmax64(lifted_lambda_res_123932, redout_125674);
                                
                                ((double *) mem_127202)[i_125676] = defunc_0_lifted_lambda_res_123943;
                                
                                double redout_tmp_128194 = max_res_123786;
                                double redout_tmp_128195 = max_res_123828;
                                
                                redout_125673 = redout_tmp_128194;
                                redout_125674 = redout_tmp_128195;
                            }
                            defunc_0_reduce_res_125155 = redout_125673;
                            defunc_0_reduce_res_125156 = redout_125674;
                            ((double *) mem_127189)[i_125684] = defunc_0_reduce_res_125156;
                            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127190, i_125684 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127202, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                            ((double *) mem_127191)[i_125684] = defunc_0_reduce_res_125155;
                        }
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_125694 = 0; i_125694 < (int64_t) 16; i_125694++) {
                            // futhark/microgpt.fut:327:96-108
                            
                            double neg_arg0_123363 = ((double *) mem_127191)[i_125694];
                            
                            // futhark/microgpt.fut:327:90-108
                            
                            double neg_res_123364 = -neg_arg0_123363;
                            
                            // futhark/microgpt.fut:4:11-25
                            for (int64_t i_125690 = 0; i_125690 < (int64_t) 16; i_125690++) {
                                // futhark/microgpt.fut:4:11-25
                                
                                double zp_lhs_123371 = ((double *) mem_126499)[i_125770 * (int64_t) 256 + i_125694 * (int64_t) 16 + i_125690];
                                
                                // futhark/microgpt.fut:327:56-108
                                
                                double zp_res_123372 = neg_res_123364 + zp_lhs_123371;
                                
                                // futhark/microgpt.fut:327:49-108
                                
                                double exp_res_123373 = futrts_exp64(zp_res_123372);
                                
                                ((double *) mem_127224)[i_125690] = exp_res_123373;
                            }
                            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127219, i_125694 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127224, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                        }
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_125698 = 0; i_125698 < (int64_t) 16; i_125698++) {
                            // futhark/microgpt.fut:71:13-49
                            
                            double defunc_0_lifted_lambda_res_123382;
                            double r_123384 = 0.0;
                            
                            for (int64_t i_123383 = 0; i_123383 < (int64_t) 16; i_123383++) {
                                // futhark/microgpt.fut:71:46-49
                                
                                double lifted_lambda_res_123385 = ((double *) mem_127219)[i_125698 * (int64_t) 16 + i_123383];
                                
                                // futhark/microgpt.fut:71:40-49
                                
                                double zp_res_123386 = r_123384 + lifted_lambda_res_123385;
                                double r_tmp_128201 = zp_res_123386;
                                
                                r_123384 = r_tmp_128201;
                            }
                            defunc_0_lifted_lambda_res_123382 = r_123384;
                            ((double *) mem_127235)[i_125698] = defunc_0_lifted_lambda_res_123382;
                        }
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_125706 = 0; i_125706 < (int64_t) 16; i_125706++) {
                            // futhark/microgpt.fut:330:84-96
                            
                            double zs_rhs_123415 = ((double *) mem_127235)[i_125706];
                            
                            // futhark/microgpt.fut:330:76-96
                            
                            double zs_res_123416 = 1.0 / zs_rhs_123415;
                            
                            // futhark/microgpt.fut:330:195-224
                            
                            double zt_res_123417 = zs_rhs_123415 * zs_rhs_123415;
                            
                            // futhark/microgpt.fut:330:186-224
                            
                            double zs_res_123418 = 1.0 / zt_res_123417;
                            
                            // futhark/microgpt.fut:71:13-49
                            
                            double defunc_0_lifted_lambda_res_123419;
                            double r_123421 = 0.0;
                            
                            for (int64_t i_123420 = 0; i_123420 < (int64_t) 16; i_123420++) {
                                // futhark/microgpt.fut:71:46-49
                                
                                double zt_lhs_123422 = ((double *) mem_127190)[i_125706 * (int64_t) 16 + i_123420];
                                
                                // futhark/microgpt.fut:71:46-49
                                
                                double zt_rhs_123423 = ((double *) mem_127219)[i_125706 * (int64_t) 16 + i_123420];
                                
                                // futhark/microgpt.fut:330:134-179
                                
                                double zt_res_123424 = zt_lhs_123422 * zt_rhs_123423;
                                
                                // futhark/microgpt.fut:330:155-224
                                
                                double zt_res_123425 = zs_res_123418 * zt_res_123424;
                                
                                // futhark/microgpt.fut:330:126-224
                                
                                double neg_res_123426 = -zt_res_123425;
                                
                                // futhark/microgpt.fut:71:40-49
                                
                                double zp_res_123427 = r_123421 + neg_res_123426;
                                double r_tmp_128203 = zp_res_123427;
                                
                                r_123421 = r_tmp_128203;
                            }
                            defunc_0_lifted_lambda_res_123419 = r_123421;
                            // futhark/microgpt.fut:4:11-25
                            for (int64_t i_125702 = 0; i_125702 < (int64_t) 16; i_125702++) {
                                // futhark/microgpt.fut:4:11-25
                                
                                double zt_lhs_123434 = ((double *) mem_127190)[i_125706 * (int64_t) 16 + i_125702];
                                
                                // futhark/microgpt.fut:330:50-96
                                
                                double zt_res_123435 = zs_res_123416 * zt_lhs_123434;
                                
                                // futhark/microgpt.fut:330:71-229
                                
                                double zp_res_123436 = defunc_0_lifted_lambda_res_123419 + zt_res_123435;
                                
                                ((double *) mem_127247)[i_125702] = zp_res_123436;
                            }
                            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127242, i_125706 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127247, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                        }
                        // futhark/microgpt.fut:71:46-49
                        
                        double zp_lhs_123459 = ((double *) mem_126499)[i_125770 * (int64_t) 256 + i_123335 * (int64_t) 16 + i_125760];
                        
                        // futhark/microgpt.fut:332:56-68
                        
                        double neg_arg0_123460 = ((double *) mem_127191)[i_123335];
                        
                        // futhark/microgpt.fut:332:50-68
                        
                        double neg_res_123461 = -neg_arg0_123460;
                        
                        // futhark/microgpt.fut:332:16-68
                        
                        double zp_res_123462 = zp_lhs_123459 + neg_res_123461;
                        
                        // futhark/microgpt.fut:332:9-68
                        
                        double exp_res_123463 = futrts_exp64(zp_res_123462);
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_123464 = ((double *) mem_127242)[i_123335 * (int64_t) 16 + i_125760];
                        
                        // futhark/microgpt.fut:332:9-96
                        
                        double zt_res_123465 = exp_res_123463 * zt_rhs_123464;
                        
                        // futhark/microgpt.fut:71:13-49
                        
                        double defunc_0_lifted_lambda_res_123466;
                        double r_123468 = 0.0;
                        
                        for (int64_t i_123467 = 0; i_123467 < (int64_t) 16; i_123467++) {
                            // futhark/microgpt.fut:71:46-49
                            
                            double zp_lhs_123469 = ((double *) mem_126499)[i_125770 * (int64_t) 256 + i_123335 * (int64_t) 16 + i_123467];
                            
                            // futhark/microgpt.fut:332:142-194
                            
                            double zp_res_123470 = neg_res_123461 + zp_lhs_123469;
                            
                            // futhark/microgpt.fut:332:135-194
                            
                            double exp_res_123471 = futrts_exp64(zp_res_123470);
                            
                            // futhark/microgpt.fut:71:46-49
                            
                            double zt_rhs_123472 = ((double *) mem_127242)[i_123335 * (int64_t) 16 + i_123467];
                            
                            // futhark/microgpt.fut:332:135-222
                            
                            double zt_res_123473 = exp_res_123471 * zt_rhs_123472;
                            
                            // futhark/microgpt.fut:332:127-222
                            
                            double neg_res_123474 = -zt_res_123473;
                            
                            // futhark/microgpt.fut:71:40-49
                            
                            double zp_res_123475 = r_123468 + neg_res_123474;
                            double r_tmp_128205 = zp_res_123475;
                            
                            r_123468 = r_tmp_128205;
                        }
                        defunc_0_lifted_lambda_res_123466 = r_123468;
                        // futhark/microgpt.fut:332:307-319
                        
                        double neg_arg0_123476 = ((double *) mem_127189)[i_123335];
                        
                        // futhark/microgpt.fut:332:301-319
                        
                        double neg_res_123477 = -neg_arg0_123476;
                        
                        // futhark/microgpt.fut:332:267-319
                        
                        double zp_res_123478 = zp_lhs_123459 + neg_res_123477;
                        
                        // futhark/microgpt.fut:332:260-319
                        
                        double neg_res_123479 = -zp_res_123478;
                        
                        // futhark/microgpt.fut:110:42-54
                        
                        double max_res_123480 = fmax64(0.0, neg_res_123479);
                        
                        // futhark/microgpt.fut:110:35-54
                        
                        double sgn_res_123481 = fsignum64(max_res_123480);
                        
                        // futhark/microgpt.fut:332:241-322
                        
                        double neg_res_123482 = -sgn_res_123481;
                        
                        // futhark/microgpt.fut:332:232-323
                        
                        double zp_res_123483 = 1.0 + neg_res_123482;
                        
                        // futhark/microgpt.fut:332:105-323
                        
                        double zt_res_123484 = defunc_0_lifted_lambda_res_123466 * zp_res_123483;
                        
                        // futhark/microgpt.fut:71:13-49
                        
                        double defunc_0_lifted_lambda_res_123485;
                        double r_123487 = 0.0;
                        
                        for (int64_t i_123486 = 0; i_123486 < (int64_t) 16; i_123486++) {
                            // futhark/microgpt.fut:71:46-49
                            
                            double zp_lhs_123488 = ((double *) mem_126499)[i_125770 * (int64_t) 256 + i_123335 * (int64_t) 16 + i_123486];
                            
                            // futhark/microgpt.fut:332:398-450
                            
                            double zp_res_123489 = neg_res_123477 + zp_lhs_123488;
                            
                            // futhark/microgpt.fut:332:391-450
                            
                            double neg_res_123490 = -zp_res_123489;
                            
                            // futhark/microgpt.fut:110:42-54
                            
                            double max_res_123491 = fmax64(0.0, neg_res_123490);
                            
                            // futhark/microgpt.fut:110:35-54
                            
                            double sgn_res_123492 = fsignum64(max_res_123491);
                            
                            // futhark/microgpt.fut:332:372-453
                            
                            double neg_res_123493 = -sgn_res_123492;
                            
                            // futhark/microgpt.fut:332:363-454
                            
                            double zp_res_123494 = 1.0 + neg_res_123493;
                            
                            // futhark/microgpt.fut:71:40-49
                            
                            double zp_res_123495 = r_123487 + zp_res_123494;
                            double r_tmp_128206 = zp_res_123495;
                            
                            r_123487 = r_tmp_128206;
                        }
                        defunc_0_lifted_lambda_res_123485 = r_123487;
                        // futhark/microgpt.fut:332:332-457
                        
                        double zs_res_123496 = 1.0 / defunc_0_lifted_lambda_res_123485;
                        
                        // futhark/microgpt.fut:332:227-457
                        
                        double zt_res_123497 = zt_res_123484 * zs_res_123496;
                        
                        // futhark/microgpt.fut:332:72-457
                        
                        double zp_res_123498 = zt_res_123465 + zt_res_123497;
                        
                        // futhark/microgpt.fut:332:98-475
                        
                        double zs_res_123499 = zp_res_123498 / 2.0;
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_123500 = ((double *) mem_126419)[i_125770 * (int64_t) 64 + i_123335 * (int64_t) 4 + i_125750];
                        
                        // futhark/microgpt.fut:332:462-508
                        
                        double zt_res_123501 = zs_res_123499 * zt_rhs_123500;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_123502 = r_123336 + zt_res_123501;
                        double r_tmp_128190 = zp_res_123502;
                        
                        r_123336 = r_tmp_128190;
                    }
                    defunc_0_lifted_lambda_res_123334 = r_123336;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_123512;
                    double r_123514 = 0.0;
                    
                    for (int64_t i_123513 = 0; i_123513 < (int64_t) 16; i_123513++) {
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_125720 = 0; i_125720 < (int64_t) 16; i_125720++) {
                            // futhark/microgpt.fut:115:13-33
                            
                            double defunc_0_reduce_res_125166;
                            double defunc_0_reduce_res_125167;
                            double redout_125709;
                            double redout_125710;
                            
                            redout_125709 = -INFINITY;
                            redout_125710 = -INFINITY;
                            for (int64_t i_125712 = 0; i_125712 < (int64_t) 16; i_125712++) {
                                // futhark/microgpt.fut:4:11-25
                                
                                double lifted_lambda_res_124218 = ((double *) mem_126498)[i_125770 * (int64_t) 256 + i_125720 * (int64_t) 16 + i_125712];
                                
                                // futhark/microgpt.fut:71:13-49
                                
                                double defunc_0_lifted_lambda_res_124229;
                                double r_124231 = 0.0;
                                
                                for (int64_t i_124230 = 0; i_124230 < (int64_t) 4; i_124230++) {
                                    // futhark/microgpt.fut:71:46-49
                                    
                                    double zt_lhs_124232 = ((double *) mem_127110)[i_125770 * (int64_t) 64 + i_125720 * (int64_t) 4 + i_124230];
                                    
                                    // futhark/microgpt.fut:71:46-49
                                    
                                    double zt_rhs_124233 = ((double *) mem_126417)[i_125770 * (int64_t) 64 + i_125712 * (int64_t) 4 + i_124230];
                                    
                                    // futhark/microgpt.fut:337:70-130
                                    
                                    double zt_res_124234 = zt_lhs_124232 * zt_rhs_124233;
                                    
                                    // futhark/microgpt.fut:71:40-49
                                    
                                    double zp_res_124235 = r_124231 + zt_res_124234;
                                    double r_tmp_128214 = zp_res_124235;
                                    
                                    r_124231 = r_tmp_128214;
                                }
                                defunc_0_lifted_lambda_res_124229 = r_124231;
                                // futhark/microgpt.fut:115:13-33
                                
                                double max_res_124072 = fmax64(lifted_lambda_res_124218, redout_125709);
                                
                                // futhark/microgpt.fut:115:13-33
                                
                                double max_res_124114 = fmax64(lifted_lambda_res_124218, redout_125710);
                                
                                ((double *) mem_127271)[i_125712] = defunc_0_lifted_lambda_res_124229;
                                
                                double redout_tmp_128211 = max_res_124072;
                                double redout_tmp_128212 = max_res_124114;
                                
                                redout_125709 = redout_tmp_128211;
                                redout_125710 = redout_tmp_128212;
                            }
                            defunc_0_reduce_res_125166 = redout_125709;
                            defunc_0_reduce_res_125167 = redout_125710;
                            ((double *) mem_127258)[i_125720] = defunc_0_reduce_res_125167;
                            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127259, i_125720 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127271, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                            ((double *) mem_127260)[i_125720] = defunc_0_reduce_res_125166;
                        }
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_125730 = 0; i_125730 < (int64_t) 16; i_125730++) {
                            // futhark/microgpt.fut:335:96-108
                            
                            double neg_arg0_123541 = ((double *) mem_127260)[i_125730];
                            
                            // futhark/microgpt.fut:335:90-108
                            
                            double neg_res_123542 = -neg_arg0_123541;
                            
                            // futhark/microgpt.fut:4:11-25
                            for (int64_t i_125726 = 0; i_125726 < (int64_t) 16; i_125726++) {
                                // futhark/microgpt.fut:4:11-25
                                
                                double zp_lhs_123549 = ((double *) mem_126498)[i_125770 * (int64_t) 256 + i_125730 * (int64_t) 16 + i_125726];
                                
                                // futhark/microgpt.fut:335:56-108
                                
                                double zp_res_123550 = neg_res_123542 + zp_lhs_123549;
                                
                                // futhark/microgpt.fut:335:49-108
                                
                                double exp_res_123551 = futrts_exp64(zp_res_123550);
                                
                                ((double *) mem_127293)[i_125726] = exp_res_123551;
                            }
                            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127288, i_125730 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127293, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                        }
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_125734 = 0; i_125734 < (int64_t) 16; i_125734++) {
                            // futhark/microgpt.fut:71:13-49
                            
                            double defunc_0_lifted_lambda_res_123560;
                            double r_123562 = 0.0;
                            
                            for (int64_t i_123561 = 0; i_123561 < (int64_t) 16; i_123561++) {
                                // futhark/microgpt.fut:71:46-49
                                
                                double lifted_lambda_res_123563 = ((double *) mem_127288)[i_125734 * (int64_t) 16 + i_123561];
                                
                                // futhark/microgpt.fut:71:40-49
                                
                                double zp_res_123564 = r_123562 + lifted_lambda_res_123563;
                                double r_tmp_128218 = zp_res_123564;
                                
                                r_123562 = r_tmp_128218;
                            }
                            defunc_0_lifted_lambda_res_123560 = r_123562;
                            ((double *) mem_127304)[i_125734] = defunc_0_lifted_lambda_res_123560;
                        }
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_125742 = 0; i_125742 < (int64_t) 16; i_125742++) {
                            // futhark/microgpt.fut:338:84-96
                            
                            double zs_rhs_123593 = ((double *) mem_127304)[i_125742];
                            
                            // futhark/microgpt.fut:338:76-96
                            
                            double zs_res_123594 = 1.0 / zs_rhs_123593;
                            
                            // futhark/microgpt.fut:338:195-224
                            
                            double zt_res_123595 = zs_rhs_123593 * zs_rhs_123593;
                            
                            // futhark/microgpt.fut:338:186-224
                            
                            double zs_res_123596 = 1.0 / zt_res_123595;
                            
                            // futhark/microgpt.fut:71:13-49
                            
                            double defunc_0_lifted_lambda_res_123597;
                            double r_123599 = 0.0;
                            
                            for (int64_t i_123598 = 0; i_123598 < (int64_t) 16; i_123598++) {
                                // futhark/microgpt.fut:71:46-49
                                
                                double zt_lhs_123600 = ((double *) mem_127259)[i_125742 * (int64_t) 16 + i_123598];
                                
                                // futhark/microgpt.fut:71:46-49
                                
                                double zt_rhs_123601 = ((double *) mem_127288)[i_125742 * (int64_t) 16 + i_123598];
                                
                                // futhark/microgpt.fut:338:134-179
                                
                                double zt_res_123602 = zt_lhs_123600 * zt_rhs_123601;
                                
                                // futhark/microgpt.fut:338:155-224
                                
                                double zt_res_123603 = zs_res_123596 * zt_res_123602;
                                
                                // futhark/microgpt.fut:338:126-224
                                
                                double neg_res_123604 = -zt_res_123603;
                                
                                // futhark/microgpt.fut:71:40-49
                                
                                double zp_res_123605 = r_123599 + neg_res_123604;
                                double r_tmp_128220 = zp_res_123605;
                                
                                r_123599 = r_tmp_128220;
                            }
                            defunc_0_lifted_lambda_res_123597 = r_123599;
                            // futhark/microgpt.fut:4:11-25
                            for (int64_t i_125738 = 0; i_125738 < (int64_t) 16; i_125738++) {
                                // futhark/microgpt.fut:4:11-25
                                
                                double zt_lhs_123612 = ((double *) mem_127259)[i_125742 * (int64_t) 16 + i_125738];
                                
                                // futhark/microgpt.fut:338:50-96
                                
                                double zt_res_123613 = zs_res_123594 * zt_lhs_123612;
                                
                                // futhark/microgpt.fut:338:71-229
                                
                                double zp_res_123614 = defunc_0_lifted_lambda_res_123597 + zt_res_123613;
                                
                                ((double *) mem_127316)[i_125738] = zp_res_123614;
                            }
                            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127311, i_125742 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127316, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                        }
                        // futhark/microgpt.fut:71:46-49
                        
                        double zp_lhs_123637 = ((double *) mem_126498)[i_125770 * (int64_t) 256 + i_125760 * (int64_t) 16 + i_123513];
                        
                        // futhark/microgpt.fut:340:56-68
                        
                        double neg_arg0_123638 = ((double *) mem_127260)[i_125760];
                        
                        // futhark/microgpt.fut:340:50-68
                        
                        double neg_res_123639 = -neg_arg0_123638;
                        
                        // futhark/microgpt.fut:340:16-68
                        
                        double zp_res_123640 = zp_lhs_123637 + neg_res_123639;
                        
                        // futhark/microgpt.fut:340:9-68
                        
                        double exp_res_123641 = futrts_exp64(zp_res_123640);
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_123642 = ((double *) mem_127311)[i_125760 * (int64_t) 16 + i_123513];
                        
                        // futhark/microgpt.fut:340:9-96
                        
                        double zt_res_123643 = exp_res_123641 * zt_rhs_123642;
                        
                        // futhark/microgpt.fut:71:13-49
                        
                        double defunc_0_lifted_lambda_res_123644;
                        double r_123646 = 0.0;
                        
                        for (int64_t i_123645 = 0; i_123645 < (int64_t) 16; i_123645++) {
                            // futhark/microgpt.fut:71:46-49
                            
                            double zp_lhs_123647 = ((double *) mem_126498)[i_125770 * (int64_t) 256 + i_125760 * (int64_t) 16 + i_123645];
                            
                            // futhark/microgpt.fut:340:142-194
                            
                            double zp_res_123648 = neg_res_123639 + zp_lhs_123647;
                            
                            // futhark/microgpt.fut:340:135-194
                            
                            double exp_res_123649 = futrts_exp64(zp_res_123648);
                            
                            // futhark/microgpt.fut:71:46-49
                            
                            double zt_rhs_123650 = ((double *) mem_127311)[i_125760 * (int64_t) 16 + i_123645];
                            
                            // futhark/microgpt.fut:340:135-222
                            
                            double zt_res_123651 = exp_res_123649 * zt_rhs_123650;
                            
                            // futhark/microgpt.fut:340:127-222
                            
                            double neg_res_123652 = -zt_res_123651;
                            
                            // futhark/microgpt.fut:71:40-49
                            
                            double zp_res_123653 = r_123646 + neg_res_123652;
                            double r_tmp_128222 = zp_res_123653;
                            
                            r_123646 = r_tmp_128222;
                        }
                        defunc_0_lifted_lambda_res_123644 = r_123646;
                        // futhark/microgpt.fut:340:307-319
                        
                        double neg_arg0_123654 = ((double *) mem_127258)[i_125760];
                        
                        // futhark/microgpt.fut:340:301-319
                        
                        double neg_res_123655 = -neg_arg0_123654;
                        
                        // futhark/microgpt.fut:340:267-319
                        
                        double zp_res_123656 = zp_lhs_123637 + neg_res_123655;
                        
                        // futhark/microgpt.fut:340:260-319
                        
                        double neg_res_123657 = -zp_res_123656;
                        
                        // futhark/microgpt.fut:110:42-54
                        
                        double max_res_123658 = fmax64(0.0, neg_res_123657);
                        
                        // futhark/microgpt.fut:110:35-54
                        
                        double sgn_res_123659 = fsignum64(max_res_123658);
                        
                        // futhark/microgpt.fut:340:241-322
                        
                        double neg_res_123660 = -sgn_res_123659;
                        
                        // futhark/microgpt.fut:340:232-323
                        
                        double zp_res_123661 = 1.0 + neg_res_123660;
                        
                        // futhark/microgpt.fut:340:105-323
                        
                        double zt_res_123662 = defunc_0_lifted_lambda_res_123644 * zp_res_123661;
                        
                        // futhark/microgpt.fut:71:13-49
                        
                        double defunc_0_lifted_lambda_res_123663;
                        double r_123665 = 0.0;
                        
                        for (int64_t i_123664 = 0; i_123664 < (int64_t) 16; i_123664++) {
                            // futhark/microgpt.fut:71:46-49
                            
                            double zp_lhs_123666 = ((double *) mem_126498)[i_125770 * (int64_t) 256 + i_125760 * (int64_t) 16 + i_123664];
                            
                            // futhark/microgpt.fut:340:398-450
                            
                            double zp_res_123667 = neg_res_123655 + zp_lhs_123666;
                            
                            // futhark/microgpt.fut:340:391-450
                            
                            double neg_res_123668 = -zp_res_123667;
                            
                            // futhark/microgpt.fut:110:42-54
                            
                            double max_res_123669 = fmax64(0.0, neg_res_123668);
                            
                            // futhark/microgpt.fut:110:35-54
                            
                            double sgn_res_123670 = fsignum64(max_res_123669);
                            
                            // futhark/microgpt.fut:340:372-453
                            
                            double neg_res_123671 = -sgn_res_123670;
                            
                            // futhark/microgpt.fut:340:363-454
                            
                            double zp_res_123672 = 1.0 + neg_res_123671;
                            
                            // futhark/microgpt.fut:71:40-49
                            
                            double zp_res_123673 = r_123665 + zp_res_123672;
                            double r_tmp_128223 = zp_res_123673;
                            
                            r_123665 = r_tmp_128223;
                        }
                        defunc_0_lifted_lambda_res_123663 = r_123665;
                        // futhark/microgpt.fut:340:332-457
                        
                        double zs_res_123674 = 1.0 / defunc_0_lifted_lambda_res_123663;
                        
                        // futhark/microgpt.fut:340:227-457
                        
                        double zt_res_123675 = zt_res_123662 * zs_res_123674;
                        
                        // futhark/microgpt.fut:340:72-457
                        
                        double zp_res_123676 = zt_res_123643 + zt_res_123675;
                        
                        // futhark/microgpt.fut:340:98-475
                        
                        double zs_res_123677 = zp_res_123676 / 2.0;
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_123678 = ((double *) mem_126418)[i_125770 * (int64_t) 64 + i_123513 * (int64_t) 4 + i_125750];
                        
                        // futhark/microgpt.fut:340:462-508
                        
                        double zt_res_123679 = zs_res_123677 * zt_rhs_123678;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_123680 = r_123514 + zt_res_123679;
                        double r_tmp_128207 = zp_res_123680;
                        
                        r_123514 = r_tmp_128207;
                    }
                    defunc_0_lifted_lambda_res_123512 = r_123514;
                    ((double *) mem_127170)[i_125750] = defunc_0_lifted_lambda_res_123512;
                    ((double *) mem_127171)[i_125750] = defunc_0_lifted_lambda_res_123334;
                    ((double *) mem_127172)[i_125750] = defunc_0_lifted_lambda_res_123290;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_127155, i_125760 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127170, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_127156, i_125760 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127171, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_127157, i_125760 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127172, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_127137, i_125770 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_127155, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_127138, i_125770 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_127156, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_127139, i_125770 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_127157, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125790 = 0; i_125790 < (int64_t) 16; i_125790++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125780 = 0; i_125780 < (int64_t) 16; i_125780++) {
                // futhark/microgpt.fut:341:57-60
                
                int64_t tmp_124410 = sdiv64(i_125780, (int64_t) 4);
                
                // futhark/microgpt.fut:341:44-62
                
                bool x_124411 = sle64((int64_t) 0, tmp_124410);
                
                // futhark/microgpt.fut:341:44-62
                
                bool y_124412 = slt64(tmp_124410, (int64_t) 4);
                
                // futhark/microgpt.fut:341:44-62
                
                bool bounds_check_124413 = x_124411 && y_124412;
                
                // futhark/microgpt.fut:341:44-62
                
                bool index_certs_124414;
                
                if (!bounds_check_124413) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_124410, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:341:44-62\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:341:13-85\n   #6  futhark/microgpt.fut:553:5-76\n   #7  futhark/microgpt.fut:570:26-576:31\n   #8  futhark/microgpt.fut:604:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:341:79-82
                
                int64_t tmp_124415 = smod64(i_125780, (int64_t) 4);
                
                // futhark/microgpt.fut:341:44-84
                
                bool x_124416 = sle64((int64_t) 0, tmp_124415);
                
                // futhark/microgpt.fut:341:44-84
                
                bool y_124417 = slt64(tmp_124415, (int64_t) 4);
                
                // futhark/microgpt.fut:341:44-84
                
                bool bounds_check_124418 = x_124416 && y_124417;
                
                // futhark/microgpt.fut:341:44-84
                
                bool index_certs_124419;
                
                if (!bounds_check_124418) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_124415, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:341:44-84\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:341:13-85\n   #6  futhark/microgpt.fut:553:5-76\n   #7  futhark/microgpt.fut:570:26-576:31\n   #8  futhark/microgpt.fut:604:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_124420 = ((double *) mem_127139)[tmp_124410 * (int64_t) 64 + i_125790 * (int64_t) 4 + tmp_124415];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_124433 = ((double *) mem_127138)[tmp_124410 * (int64_t) 64 + i_125790 * (int64_t) 4 + tmp_124415];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_124449 = ((double *) mem_127137)[tmp_124410 * (int64_t) 64 + i_125790 * (int64_t) 4 + tmp_124415];
                
                ((double *) mem_127378)[i_125780] = lifted_lambda_res_124449;
                ((double *) mem_127379)[i_125780] = lifted_lambda_res_124433;
                ((double *) mem_127380)[i_125780] = lifted_lambda_res_124420;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127363, i_125790 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127378, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127364, i_125790 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127379, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127365, i_125790 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127380, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125815 = 0; i_125815 < (int64_t) 16; i_125815++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125802 = 0; i_125802 < (int64_t) 16; i_125802++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_124612;
                double r_124614 = 0.0;
                
                for (int64_t i_124613 = 0; i_124613 < (int64_t) 16; i_124613++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_124615 = ((double *) mem_127365)[i_125815 * (int64_t) 16 + i_124613];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_124616 = ((double *) mem_param_126207.mem)[i_124613 * (int64_t) 16 + i_125802];
                    
                    // futhark/microgpt.fut:344:69-114
                    
                    double zt_res_124617 = zt_lhs_124615 * zt_rhs_124616;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_124618 = r_124614 + zt_res_124617;
                    double r_tmp_128238 = zp_res_124618;
                    
                    r_124614 = r_tmp_128238;
                }
                defunc_0_lifted_lambda_res_124612 = r_124614;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_124619;
                double r_124621 = 0.0;
                
                for (int64_t i_124620 = 0; i_124620 < (int64_t) 16; i_124620++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_124622 = ((double *) mem_127364)[i_125815 * (int64_t) 16 + i_124620];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_124623 = ((double *) mem_param_126183.mem)[i_124620 * (int64_t) 16 + i_125802];
                    
                    // futhark/microgpt.fut:344:145-190
                    
                    double zt_res_124624 = zt_lhs_124622 * zt_rhs_124623;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_124625 = r_124621 + zt_res_124624;
                    double r_tmp_128239 = zp_res_124625;
                    
                    r_124621 = r_tmp_128239;
                }
                defunc_0_lifted_lambda_res_124619 = r_124621;
                // futhark/microgpt.fut:344:47-192
                
                double zp_res_124626 = defunc_0_lifted_lambda_res_124612 + defunc_0_lifted_lambda_res_124619;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_124627;
                double r_124629 = 0.0;
                
                for (int64_t i_124628 = 0; i_124628 < (int64_t) 16; i_124628++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_124630 = ((double *) mem_127363)[i_125815 * (int64_t) 16 + i_124628];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_124631 = ((double *) mem_param_126195.mem)[i_124628 * (int64_t) 16 + i_125802];
                    
                    // futhark/microgpt.fut:344:222-267
                    
                    double zt_res_124632 = zt_lhs_124630 * zt_rhs_124631;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_124633 = r_124629 + zt_res_124632;
                    double r_tmp_128240 = zp_res_124633;
                    
                    r_124629 = r_tmp_128240;
                }
                defunc_0_lifted_lambda_res_124627 = r_124629;
                // futhark/microgpt.fut:344:118-269
                
                double zp_res_124634 = zp_res_124626 + defunc_0_lifted_lambda_res_124627;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_124641;
                double r_124643 = 0.0;
                
                for (int64_t i_124642 = 0; i_124642 < (int64_t) 16; i_124642++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_124644 = ((double *) mem_127363)[i_124642 * (int64_t) 16 + i_125815];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_124645 = ((double *) mem_126347)[i_124642 * (int64_t) 16 + i_125802];
                    
                    // futhark/microgpt.fut:372:68-111
                    
                    double zt_res_124646 = zt_lhs_124644 * zt_rhs_124645;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_124647 = r_124643 + zt_res_124646;
                    double r_tmp_128241 = zp_res_124647;
                    
                    r_124643 = r_tmp_128241;
                }
                defunc_0_lifted_lambda_res_124641 = r_124643;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_124657;
                double r_124659 = 0.0;
                
                for (int64_t i_124658 = 0; i_124658 < (int64_t) 16; i_124658++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_124660 = ((double *) mem_127364)[i_124658 * (int64_t) 16 + i_125815];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_124661 = ((double *) mem_126347)[i_124658 * (int64_t) 16 + i_125802];
                    
                    // futhark/microgpt.fut:373:68-111
                    
                    double zt_res_124662 = zt_lhs_124660 * zt_rhs_124661;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_124663 = r_124659 + zt_res_124662;
                    double r_tmp_128242 = zp_res_124663;
                    
                    r_124659 = r_tmp_128242;
                }
                defunc_0_lifted_lambda_res_124657 = r_124659;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_124675;
                double r_124677 = 0.0;
                
                for (int64_t i_124676 = 0; i_124676 < (int64_t) 16; i_124676++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_124678 = ((double *) mem_127365)[i_124676 * (int64_t) 16 + i_125815];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_124679 = ((double *) mem_126347)[i_124676 * (int64_t) 16 + i_125802];
                    
                    // futhark/microgpt.fut:374:68-111
                    
                    double zt_res_124680 = zt_lhs_124678 * zt_rhs_124679;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_124681 = r_124677 + zt_res_124680;
                    double r_tmp_128243 = zp_res_124681;
                    
                    r_124677 = r_tmp_128243;
                }
                defunc_0_lifted_lambda_res_124675 = r_124677;
                ((double *) mem_127431)[i_125802] = defunc_0_lifted_lambda_res_124675;
                ((double *) mem_127432)[i_125802] = defunc_0_lifted_lambda_res_124657;
                ((double *) mem_127433)[i_125802] = defunc_0_lifted_lambda_res_124641;
                ((double *) mem_127434)[i_125802] = zp_res_124634;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127411, i_125815 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127431, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127412, i_125815 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127432, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127413, i_125815 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127433, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127414, i_125815 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127434, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125826 = 0; i_125826 < (int64_t) 16; i_125826++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125822 = 0; i_125822 < (int64_t) 16; i_125822++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_117613 = ((double *) mem_127414)[i_125826 * (int64_t) 16 + i_125822];
                
                ((double *) mem_127480)[i_125822] = lifted_lambda_res_117613;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127475, i_125826 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127480, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125832 = 0; i_125832 < (int64_t) 16; i_125832++) {
            // futhark/microgpt.fut:346:47-59
            
            double zp_lhs_118162 = ((double *) mem_126346)[i_125832];
            
            // futhark/microgpt.fut:346:47-87
            
            double zp_res_118163 = 1.0e-5 + zp_lhs_118162;
            
            // futhark/microgpt.fut:346:39-87
            
            double sqrt_res_118164 = futrts_sqrt64(zp_res_118163);
            
            // futhark/microgpt.fut:348:129-158
            
            double zt_res_118172 = sqrt_res_118164 * sqrt_res_118164;
            
            // futhark/microgpt.fut:348:120-158
            
            double zs_res_118173 = 1.0 / zt_res_118172;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_118174;
            double r_118176 = 0.0;
            
            for (int64_t i_118175 = 0; i_118175 < (int64_t) 16; i_118175++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_118177 = ((double *) mem_127475)[i_125832 * (int64_t) 16 + i_118175];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_118178 = ((double *) mem_126318)[i_125832 * (int64_t) 16 + i_118175];
                
                // futhark/microgpt.fut:348:70-113
                
                double zt_res_118179 = zt_lhs_118177 * zt_rhs_118178;
                
                // futhark/microgpt.fut:348:91-158
                
                double zt_res_118180 = zs_res_118173 * zt_res_118179;
                
                // futhark/microgpt.fut:348:62-158
                
                double neg_res_118181 = -zt_res_118180;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_118182 = r_118176 + neg_res_118181;
                double r_tmp_128248 = zp_res_118182;
                
                r_118176 = r_tmp_128248;
            }
            defunc_0_lifted_lambda_res_118174 = r_118176;
            // futhark/microgpt.fut:348:184-248
            
            double zt_res_118186 = 2.0 * sqrt_res_118164;
            
            // futhark/microgpt.fut:348:170-248
            
            double zs_res_118187 = 1.0 / zt_res_118186;
            
            // futhark/microgpt.fut:348:40-248
            
            double zt_res_118188 = defunc_0_lifted_lambda_res_118174 * zs_res_118187;
            
            ((double *) mem_127491)[i_125832] = zt_res_118188;
            ((double *) mem_127492)[i_125832] = sqrt_res_118164;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125841 = 0; i_125841 < (int64_t) 16; i_125841++) {
            // futhark/microgpt.fut:349:98-110
            
            double zs_rhs_117647 = ((double *) mem_127492)[i_125841];
            
            // futhark/microgpt.fut:349:90-110
            
            double zs_res_117648 = 1.0 / zs_rhs_117647;
            
            // futhark/microgpt.fut:349:120-132
            
            double zs_lhs_117649 = ((double *) mem_127491)[i_125841];
            
            // futhark/microgpt.fut:349:120-147
            
            double zs_res_117650 = zs_lhs_117649 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125837 = 0; i_125837 < (int64_t) 16; i_125837++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_117657 = ((double *) mem_127062)[i_125841 * (int64_t) 16 + i_125837];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_117658 = ((double *) mem_127475)[i_125841 * (int64_t) 16 + i_125837];
                
                // futhark/microgpt.fut:349:64-110
                
                double zt_res_117659 = zs_res_117648 * zt_lhs_117658;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_117660 = ((double *) mem_126318)[i_125841 * (int64_t) 16 + i_125837];
                
                // futhark/microgpt.fut:349:133-171
                
                double zt_res_117661 = zs_res_117650 * zt_rhs_117660;
                
                // futhark/microgpt.fut:349:149-230
                
                double zp_res_117662 = zt_res_117661 + zt_res_117661;
                
                // futhark/microgpt.fut:349:85-230
                
                double zp_res_117663 = zt_res_117659 + zp_res_117662;
                
                // futhark/microgpt.fut:349:37-230
                
                double zp_res_117664 = zp_lhs_117657 + zp_res_117663;
                
                ((double *) mem_127510)[i_125837] = zp_res_117664;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127505, i_125841 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127510, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125854 = 0; i_125854 < (int64_t) 16; i_125854++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125847 = 0; i_125847 < (int64_t) 16; i_125847++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_124707 = ((double *) mem_127505)[i_125854 * (int64_t) 16 + i_125847];
                
                ((double *) mem_127531)[i_125847] = lifted_lambda_res_124707;
                ((double *) mem_127532)[i_125847] = lifted_lambda_res_124707;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127521, i_125854 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127531, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127522, i_125854 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127532, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125865 = 0; i_125865 < (int64_t) 16; i_125865++) {
            // futhark/microgpt.fut:367:47-59
            
            double zp_lhs_120742 = ((double *) mem_126317)[i_125865];
            
            // futhark/microgpt.fut:367:47-87
            
            double zp_res_120743 = 1.0e-5 + zp_lhs_120742;
            
            // futhark/microgpt.fut:367:39-87
            
            double sqrt_res_120744 = futrts_sqrt64(zp_res_120743);
            
            // futhark/microgpt.fut:369:156-185
            
            double zt_res_120752 = sqrt_res_120744 * sqrt_res_120744;
            
            // futhark/microgpt.fut:369:147-185
            
            double zs_res_120753 = 1.0 / zt_res_120752;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_120754;
            double r_120756 = 0.0;
            
            for (int64_t i_120755 = 0; i_120755 < (int64_t) 16; i_120755++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_120757 = ((double *) mem_127522)[i_125865 * (int64_t) 16 + i_120755];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_120758 = ((double *) mem_param_126191.mem)[i_125865 * (int64_t) 16 + i_120755];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_rhs_120759 = ((double *) mem_126284)[i_125865 * (int64_t) 16 + i_120755];
                
                // futhark/microgpt.fut:369:95-139
                
                double zp_res_120760 = zp_lhs_120758 + zp_rhs_120759;
                
                // futhark/microgpt.fut:369:69-139
                
                double zt_res_120761 = zt_lhs_120757 * zp_res_120760;
                
                // futhark/microgpt.fut:369:90-185
                
                double zt_res_120762 = zs_res_120753 * zt_res_120761;
                
                // futhark/microgpt.fut:369:61-185
                
                double neg_res_120763 = -zt_res_120762;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_120764 = r_120756 + neg_res_120763;
                double r_tmp_128259 = zp_res_120764;
                
                r_120756 = r_tmp_128259;
            }
            defunc_0_lifted_lambda_res_120754 = r_120756;
            // futhark/microgpt.fut:380:47-59
            
            double zp_lhs_120775 = ((double *) mem_126316)[i_125865];
            
            // futhark/microgpt.fut:380:47-87
            
            double zp_res_120776 = 1.0e-5 + zp_lhs_120775;
            
            // futhark/microgpt.fut:380:39-87
            
            double sqrt_res_120777 = futrts_sqrt64(zp_res_120776);
            
            // futhark/microgpt.fut:382:156-185
            
            double zt_res_120785 = sqrt_res_120777 * sqrt_res_120777;
            
            // futhark/microgpt.fut:382:147-185
            
            double zs_res_120786 = 1.0 / zt_res_120785;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_120787;
            double r_120789 = 0.0;
            
            for (int64_t i_120788 = 0; i_120788 < (int64_t) 16; i_120788++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_120790 = ((double *) mem_127521)[i_125865 * (int64_t) 16 + i_120788];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_120791 = ((double *) mem_param_126191.mem)[i_125865 * (int64_t) 16 + i_120788];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_rhs_120792 = ((double *) mem_126284)[i_125865 * (int64_t) 16 + i_120788];
                
                // futhark/microgpt.fut:382:95-139
                
                double zp_res_120793 = zp_lhs_120791 + zp_rhs_120792;
                
                // futhark/microgpt.fut:382:69-139
                
                double zt_res_120794 = zt_lhs_120790 * zp_res_120793;
                
                // futhark/microgpt.fut:382:90-185
                
                double zt_res_120795 = zs_res_120786 * zt_res_120794;
                
                // futhark/microgpt.fut:382:61-185
                
                double neg_res_120796 = -zt_res_120795;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_120797 = r_120789 + neg_res_120796;
                double r_tmp_128260 = zp_res_120797;
                
                r_120789 = r_tmp_128260;
            }
            defunc_0_lifted_lambda_res_120787 = r_120789;
            ((double *) mem_127553)[i_125865] = defunc_0_lifted_lambda_res_120787;
            ((double *) mem_127554)[i_125865] = sqrt_res_120777;
            ((double *) mem_127555)[i_125865] = defunc_0_lifted_lambda_res_120754;
            ((double *) mem_127556)[i_125865] = sqrt_res_120744;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125874 = 0; i_125874 < (int64_t) 16; i_125874++) {
            // futhark/microgpt.fut:370:39-51
            
            double zt_lhs_120858 = ((double *) mem_127555)[i_125874];
            
            // futhark/microgpt.fut:370:93-105
            
            double zp_lhs_120859 = ((double *) mem_126317)[i_125874];
            
            // futhark/microgpt.fut:370:93-133
            
            double zp_res_120860 = 1.0e-5 + zp_lhs_120859;
            
            // futhark/microgpt.fut:370:85-133
            
            double sqrt_res_120861 = futrts_sqrt64(zp_res_120860);
            
            // futhark/microgpt.fut:370:71-135
            
            double zt_res_120862 = 2.0 * sqrt_res_120861;
            
            // futhark/microgpt.fut:370:57-135
            
            double zs_res_120863 = 1.0 / zt_res_120862;
            
            // futhark/microgpt.fut:370:39-135
            
            double zt_res_120864 = zt_lhs_120858 * zs_res_120863;
            
            // futhark/microgpt.fut:383:39-51
            
            double zt_lhs_120871 = ((double *) mem_127553)[i_125874];
            
            // futhark/microgpt.fut:383:93-105
            
            double zp_lhs_120872 = ((double *) mem_126316)[i_125874];
            
            // futhark/microgpt.fut:383:93-133
            
            double zp_res_120873 = 1.0e-5 + zp_lhs_120872;
            
            // futhark/microgpt.fut:383:85-133
            
            double sqrt_res_120874 = futrts_sqrt64(zp_res_120873);
            
            // futhark/microgpt.fut:383:71-135
            
            double zt_res_120875 = 2.0 * sqrt_res_120874;
            
            // futhark/microgpt.fut:383:57-135
            
            double zs_res_120876 = 1.0 / zt_res_120875;
            
            // futhark/microgpt.fut:383:39-135
            
            double zt_res_120877 = zt_lhs_120871 * zs_res_120876;
            
            ((double *) mem_127581)[i_125874] = zt_res_120877;
            ((double *) mem_127582)[i_125874] = zt_res_120864;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125888 = 0; i_125888 < (int64_t) 16; i_125888++) {
            // futhark/microgpt.fut:371:72-84
            
            double zs_rhs_120895 = ((double *) mem_127556)[i_125888];
            
            // futhark/microgpt.fut:371:64-84
            
            double zs_res_120896 = 1.0 / zs_rhs_120895;
            
            // futhark/microgpt.fut:371:94-106
            
            double zs_lhs_120897 = ((double *) mem_127582)[i_125888];
            
            // futhark/microgpt.fut:371:94-121
            
            double zs_res_120898 = zs_lhs_120897 / 16.0;
            
            // futhark/microgpt.fut:384:94-106
            
            double zs_lhs_120922 = ((double *) mem_127581)[i_125888];
            
            // futhark/microgpt.fut:384:94-121
            
            double zs_res_120923 = zs_lhs_120922 / 16.0;
            
            // futhark/microgpt.fut:384:72-84
            
            double zs_rhs_120920 = ((double *) mem_127554)[i_125888];
            
            // futhark/microgpt.fut:384:64-84
            
            double zs_res_120921 = 1.0 / zs_rhs_120920;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125881 = 0; i_125881 < (int64_t) 16; i_125881++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_124734 = ((double *) mem_127522)[i_125888 * (int64_t) 16 + i_125881];
                
                // futhark/microgpt.fut:371:38-84
                
                double zt_res_124735 = zs_res_120896 * zt_lhs_124734;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_124736 = ((double *) mem_param_126191.mem)[i_125888 * (int64_t) 16 + i_125881];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_124737 = ((double *) mem_126284)[i_125888 * (int64_t) 16 + i_125881];
                
                // futhark/microgpt.fut:371:128-172
                
                double zp_res_124738 = zp_lhs_124736 + zp_rhs_124737;
                
                // futhark/microgpt.fut:371:107-172
                
                double zt_res_124739 = zs_res_120898 * zp_res_124738;
                
                // futhark/microgpt.fut:371:123-259
                
                double zp_res_124740 = zt_res_124739 + zt_res_124739;
                
                // futhark/microgpt.fut:371:59-259
                
                double zp_res_124741 = zt_res_124735 + zp_res_124740;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_124748 = ((double *) mem_127521)[i_125888 * (int64_t) 16 + i_125881];
                
                // futhark/microgpt.fut:384:38-84
                
                double zt_res_124749 = zs_res_120921 * zt_lhs_124748;
                
                // futhark/microgpt.fut:384:107-172
                
                double zt_res_124753 = zs_res_120923 * zp_res_124738;
                
                // futhark/microgpt.fut:384:123-259
                
                double zp_res_124754 = zt_res_124753 + zt_res_124753;
                
                // futhark/microgpt.fut:384:59-259
                
                double zp_res_124755 = zt_res_124749 + zp_res_124754;
                
                ((double *) mem_127605)[i_125881] = zp_res_124755;
                ((double *) mem_127606)[i_125881] = zp_res_124741;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127595, i_125888 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127605, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127596, i_125888 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127606, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125897 = 0; i_125897 < (int64_t) 64; i_125897++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125893 = 0; i_125893 < (int64_t) 16; i_125893++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_117865;
                double r_117867 = 0.0;
                
                for (int64_t i_117866 = 0; i_117866 < (int64_t) 16; i_117866++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_117868 = ((double *) mem_126984)[i_117866 * (int64_t) 64 + i_125897];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_117869 = ((double *) mem_126678)[i_117866 * (int64_t) 16 + i_125893];
                    
                    // futhark/microgpt.fut:376:67-110
                    
                    double zt_res_117870 = zt_lhs_117868 * zt_rhs_117869;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_117871 = r_117867 + zt_res_117870;
                    double r_tmp_128269 = zp_res_117871;
                    
                    r_117867 = r_tmp_128269;
                }
                defunc_0_lifted_lambda_res_117865 = r_117867;
                ((double *) mem_127632)[i_125893] = defunc_0_lifted_lambda_res_117865;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127627, i_125897 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127632, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_125910 = 0; i_125910 < (int64_t) 27; i_125910++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_125903 = 0; i_125903 < (int64_t) 16; i_125903++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_124776;
                double r_124778 = 0.0;
                
                for (int64_t i_124777 = 0; i_124777 < (int64_t) 16; i_124777++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_124779 = ((double *) mem_126904)[i_124777 * (int64_t) 27 + i_125910];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_124780 = ((double *) mem_126748)[i_124777 * (int64_t) 16 + i_125903];
                    
                    // futhark/microgpt.fut:378:68-111
                    
                    double zt_res_124781 = zt_lhs_124779 * zt_rhs_124780;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_124782 = r_124778 + zt_res_124781;
                    double r_tmp_128274 = zp_res_124782;
                    
                    r_124778 = r_tmp_128274;
                }
                defunc_0_lifted_lambda_res_124776 = r_124778;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_124785;
                double r_124787 = 0.0;
                
                for (int64_t i_124786 = 0; i_124786 < (int64_t) 16; i_124786++) {
                    int64_t zeze_lhs_124788 = ((int64_t *) seqs_mem_126175.mem)[step_115941 * (int64_t) 16 + i_124786];
                    
                    // futhark/microgpt.fut:554:58-109
                    
                    bool cond_124789 = zeze_lhs_124788 == i_125910;
                    
                    // futhark/microgpt.fut:554:58-109
                    
                    double lifted_lambda_res_124790;
                    
                    if (cond_124789) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double lifted_lambda_res_t_res_125201 = ((double *) mem_127595)[i_124786 * (int64_t) 16 + i_125903];
                        
                        lifted_lambda_res_124790 = lifted_lambda_res_t_res_125201;
                    } else {
                        lifted_lambda_res_124790 = 0.0;
                    }
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_124796 = r_124787 + lifted_lambda_res_124790;
                    double r_tmp_128275 = zp_res_124796;
                    
                    r_124787 = r_tmp_128275;
                }
                defunc_0_lifted_lambda_res_124785 = r_124787;
                ((double *) mem_127653)[i_125903] = defunc_0_lifted_lambda_res_124785;
                ((double *) mem_127654)[i_125903] = defunc_0_lifted_lambda_res_124776;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127643, i_125910 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127653, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_127644, i_125910 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_127654, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:66:26-45
        
        double i64_res_118036 = sitofp_i64_f64(step_115941);
        
        // futhark/microgpt.fut:489:46-65
        
        double zm_rhs_118037 = i64_res_118036 / 500.0;
        
        // futhark/microgpt.fut:489:24-65
        
        double zt_rhs_118038 = 1.0 - zm_rhs_118037;
        
        // futhark/microgpt.fut:489:19-65
        
        double lt_r_118039 = 1.0e-2 * zt_rhs_118038;
        
        // futhark/microgpt.fut:491:5-52
        if (memblock_alloc(ctx, &mem_127675, (int64_t) 3456, "mem_127675")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:491:5-52
        // futhark/microgpt.fut:491:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127675.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126199.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:491:5-52
        if (memblock_alloc(ctx, &mem_127677, (int64_t) 3456, "mem_127677")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:491:5-52
        // futhark/microgpt.fut:491:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127677.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126235.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:491:5-52
        if (memblock_alloc(ctx, &mem_127679, (int64_t) 3456, "mem_127679")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:491:5-52
        // futhark/microgpt.fut:491:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127679.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126271.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:491:5-52
        if (memblock_alloc(ctx, &mem_127681, (int64_t) 3456, "mem_127681")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:491:5-52
        // futhark/microgpt.fut:491:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127681.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_127643, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:491:5-52
        if (futrts_adam_opt_w_12300(ctx, &ext_mem_127685, &ext_mem_127684, &ext_mem_127683, mem_127675, mem_127677, mem_127679, mem_127681, (int64_t) 27, (int64_t) 16, step_115941, lt_r_118039) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_127675, "mem_127675") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127677, "mem_127677") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127679, "mem_127679") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127681, "mem_127681") != 0)
            return 1;
        // futhark/microgpt.fut:493:5-52
        if (memblock_alloc(ctx, &mem_127686, (int64_t) 2048, "mem_127686")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:493:5-52
        // futhark/microgpt.fut:493:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127686.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126191.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:493:5-52
        if (memblock_alloc(ctx, &mem_127688, (int64_t) 2048, "mem_127688")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:493:5-52
        // futhark/microgpt.fut:493:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127688.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126227.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:493:5-52
        if (memblock_alloc(ctx, &mem_127690, (int64_t) 2048, "mem_127690")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:493:5-52
        // futhark/microgpt.fut:493:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127690.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126263.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:493:5-52
        if (memblock_alloc(ctx, &mem_127692, (int64_t) 2048, "mem_127692")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:493:5-52
        // futhark/microgpt.fut:493:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127692.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_127596, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:493:5-52
        if (futrts_adam_opt_w_12301(ctx, &ext_mem_127696, &ext_mem_127695, &ext_mem_127694, mem_127686, mem_127688, mem_127690, mem_127692, (int64_t) 16, (int64_t) 16, step_115941, lt_r_118039) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_127686, "mem_127686") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127688, "mem_127688") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127690, "mem_127690") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127692, "mem_127692") != 0)
            return 1;
        // futhark/microgpt.fut:495:5-56
        if (memblock_alloc(ctx, &mem_127697, (int64_t) 2048, "mem_127697")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:495:5-56
        // futhark/microgpt.fut:495:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127697.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126195.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:495:5-56
        if (memblock_alloc(ctx, &mem_127699, (int64_t) 2048, "mem_127699")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:495:5-56
        // futhark/microgpt.fut:495:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127699.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126231.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:495:5-56
        if (memblock_alloc(ctx, &mem_127701, (int64_t) 2048, "mem_127701")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:495:5-56
        // futhark/microgpt.fut:495:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127701.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126267.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:495:5-56
        if (memblock_alloc(ctx, &mem_127703, (int64_t) 2048, "mem_127703")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:495:5-56
        // futhark/microgpt.fut:495:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127703.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_127413, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:495:5-56
        if (futrts_adam_opt_w_12301(ctx, &ext_mem_127707, &ext_mem_127706, &ext_mem_127705, mem_127697, mem_127699, mem_127701, mem_127703, (int64_t) 16, (int64_t) 16, step_115941, lt_r_118039) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_127697, "mem_127697") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127699, "mem_127699") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127701, "mem_127701") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127703, "mem_127703") != 0)
            return 1;
        // futhark/microgpt.fut:497:5-56
        if (memblock_alloc(ctx, &mem_127708, (int64_t) 2048, "mem_127708")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:497:5-56
        // futhark/microgpt.fut:497:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127708.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126183.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:497:5-56
        if (memblock_alloc(ctx, &mem_127710, (int64_t) 2048, "mem_127710")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:497:5-56
        // futhark/microgpt.fut:497:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127710.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126219.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:497:5-56
        if (memblock_alloc(ctx, &mem_127712, (int64_t) 2048, "mem_127712")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:497:5-56
        // futhark/microgpt.fut:497:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127712.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126255.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:497:5-56
        if (memblock_alloc(ctx, &mem_127714, (int64_t) 2048, "mem_127714")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:497:5-56
        // futhark/microgpt.fut:497:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127714.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_127412, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:497:5-56
        if (futrts_adam_opt_w_12301(ctx, &ext_mem_127718, &ext_mem_127717, &ext_mem_127716, mem_127708, mem_127710, mem_127712, mem_127714, (int64_t) 16, (int64_t) 16, step_115941, lt_r_118039) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_127708, "mem_127708") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127710, "mem_127710") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127712, "mem_127712") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127714, "mem_127714") != 0)
            return 1;
        // futhark/microgpt.fut:499:5-56
        if (memblock_alloc(ctx, &mem_127719, (int64_t) 2048, "mem_127719")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:499:5-56
        // futhark/microgpt.fut:499:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127719.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126207.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:499:5-56
        if (memblock_alloc(ctx, &mem_127721, (int64_t) 2048, "mem_127721")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:499:5-56
        // futhark/microgpt.fut:499:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127721.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126243.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:499:5-56
        if (memblock_alloc(ctx, &mem_127723, (int64_t) 2048, "mem_127723")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:499:5-56
        // futhark/microgpt.fut:499:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127723.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126279.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:499:5-56
        if (memblock_alloc(ctx, &mem_127725, (int64_t) 2048, "mem_127725")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:499:5-56
        // futhark/microgpt.fut:499:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127725.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_127411, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:499:5-56
        if (futrts_adam_opt_w_12301(ctx, &ext_mem_127729, &ext_mem_127728, &ext_mem_127727, mem_127719, mem_127721, mem_127723, mem_127725, (int64_t) 16, (int64_t) 16, step_115941, lt_r_118039) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_127719, "mem_127719") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127721, "mem_127721") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127723, "mem_127723") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127725, "mem_127725") != 0)
            return 1;
        // futhark/microgpt.fut:501:5-56
        if (memblock_alloc(ctx, &mem_127730, (int64_t) 2048, "mem_127730")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:501:5-56
        // futhark/microgpt.fut:501:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127730.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126187.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:501:5-56
        if (memblock_alloc(ctx, &mem_127732, (int64_t) 2048, "mem_127732")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:501:5-56
        // futhark/microgpt.fut:501:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127732.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126223.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:501:5-56
        if (memblock_alloc(ctx, &mem_127734, (int64_t) 2048, "mem_127734")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:501:5-56
        // futhark/microgpt.fut:501:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127734.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126259.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:501:5-56
        if (memblock_alloc(ctx, &mem_127736, (int64_t) 2048, "mem_127736")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:501:5-56
        // futhark/microgpt.fut:501:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127736.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_127078, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:501:5-56
        if (futrts_adam_opt_w_12301(ctx, &ext_mem_127740, &ext_mem_127739, &ext_mem_127738, mem_127730, mem_127732, mem_127734, mem_127736, (int64_t) 16, (int64_t) 16, step_115941, lt_r_118039) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_127730, "mem_127730") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127732, "mem_127732") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127734, "mem_127734") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127736, "mem_127736") != 0)
            return 1;
        // futhark/microgpt.fut:503:5-52
        if (memblock_alloc(ctx, &mem_127741, (int64_t) 8192, "mem_127741")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:503:5-52
        // futhark/microgpt.fut:503:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127741.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126203.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:503:5-52
        if (memblock_alloc(ctx, &mem_127743, (int64_t) 8192, "mem_127743")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:503:5-52
        // futhark/microgpt.fut:503:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127743.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126239.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:503:5-52
        if (memblock_alloc(ctx, &mem_127745, (int64_t) 8192, "mem_127745")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:503:5-52
        // futhark/microgpt.fut:503:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127745.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126275.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:503:5-52
        if (memblock_alloc(ctx, &mem_127747, (int64_t) 8192, "mem_127747")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:503:5-52
        // futhark/microgpt.fut:503:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127747.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_127627, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:503:5-52
        if (futrts_adam_opt_w_12300(ctx, &ext_mem_127751, &ext_mem_127750, &ext_mem_127749, mem_127741, mem_127743, mem_127745, mem_127747, (int64_t) 64, (int64_t) 16, step_115941, lt_r_118039) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_127741, "mem_127741") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127743, "mem_127743") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127745, "mem_127745") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127747, "mem_127747") != 0)
            return 1;
        // futhark/microgpt.fut:505:5-60
        if (memblock_alloc(ctx, &mem_127752, (int64_t) 8192, "mem_127752")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:505:5-60
        // futhark/microgpt.fut:505:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127752.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_126179.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:505:5-60
        if (memblock_alloc(ctx, &mem_127754, (int64_t) 8192, "mem_127754")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:505:5-60
        // futhark/microgpt.fut:505:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127754.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_126215.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:505:5-60
        if (memblock_alloc(ctx, &mem_127756, (int64_t) 8192, "mem_127756")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:505:5-60
        // futhark/microgpt.fut:505:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127756.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_126251.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:505:5-60
        if (memblock_alloc(ctx, &mem_127758, (int64_t) 8192, "mem_127758")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:505:5-60
        // futhark/microgpt.fut:505:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127758.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_126952, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:505:5-60
        if (futrts_adam_opt_w_12300(ctx, &ext_mem_127762, &ext_mem_127761, &ext_mem_127760, mem_127752, mem_127754, mem_127756, mem_127758, (int64_t) 16, (int64_t) 64, step_115941, lt_r_118039) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_127752, "mem_127752") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127754, "mem_127754") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127756, "mem_127756") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127758, "mem_127758") != 0)
            return 1;
        // futhark/microgpt.fut:507:5-56
        if (memblock_alloc(ctx, &mem_127763, (int64_t) 3456, "mem_127763")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:507:5-56
        // futhark/microgpt.fut:507:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127763.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126211.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:507:5-56
        if (memblock_alloc(ctx, &mem_127765, (int64_t) 3456, "mem_127765")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:507:5-56
        // futhark/microgpt.fut:507:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127765.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126247.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:507:5-56
        if (memblock_alloc(ctx, &mem_127767, (int64_t) 3456, "mem_127767")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:507:5-56
        // futhark/microgpt.fut:507:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127767.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_126283.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:507:5-56
        if (memblock_alloc(ctx, &mem_127769, (int64_t) 3456, "mem_127769")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:507:5-56
        // futhark/microgpt.fut:507:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_127769.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_127644, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:507:5-56
        if (futrts_adam_opt_w_12300(ctx, &ext_mem_127773, &ext_mem_127772, &ext_mem_127771, mem_127763, mem_127765, mem_127767, mem_127769, (int64_t) 27, (int64_t) 16, step_115941, lt_r_118039) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_127763, "mem_127763") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127765, "mem_127765") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127767, "mem_127767") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127769, "mem_127769") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127981, &ext_mem_127762, "ext_mem_127762") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127982, &ext_mem_127718, "ext_mem_127718") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127983, &ext_mem_127740, "ext_mem_127740") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127984, &ext_mem_127696, "ext_mem_127696") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127985, &ext_mem_127707, "ext_mem_127707") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127986, &ext_mem_127685, "ext_mem_127685") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127987, &ext_mem_127751, "ext_mem_127751") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127988, &ext_mem_127729, "ext_mem_127729") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127989, &ext_mem_127773, "ext_mem_127773") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127990, &ext_mem_127761, "ext_mem_127761") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127991, &ext_mem_127717, "ext_mem_127717") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127992, &ext_mem_127739, "ext_mem_127739") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127993, &ext_mem_127695, "ext_mem_127695") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127994, &ext_mem_127706, "ext_mem_127706") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127995, &ext_mem_127684, "ext_mem_127684") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127996, &ext_mem_127750, "ext_mem_127750") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127997, &ext_mem_127728, "ext_mem_127728") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127998, &ext_mem_127772, "ext_mem_127772") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_127999, &ext_mem_127760, "ext_mem_127760") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_128000, &ext_mem_127716, "ext_mem_127716") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_128001, &ext_mem_127738, "ext_mem_127738") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_128002, &ext_mem_127694, "ext_mem_127694") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_128003, &ext_mem_127705, "ext_mem_127705") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_128004, &ext_mem_127683, "ext_mem_127683") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_128005, &ext_mem_127749, "ext_mem_127749") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_128006, &ext_mem_127727, "ext_mem_127727") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_128007, &ext_mem_127771, "ext_mem_127771") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126179, &mem_param_tmp_127981, "mem_param_tmp_127981") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126183, &mem_param_tmp_127982, "mem_param_tmp_127982") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126187, &mem_param_tmp_127983, "mem_param_tmp_127983") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126191, &mem_param_tmp_127984, "mem_param_tmp_127984") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126195, &mem_param_tmp_127985, "mem_param_tmp_127985") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126199, &mem_param_tmp_127986, "mem_param_tmp_127986") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126203, &mem_param_tmp_127987, "mem_param_tmp_127987") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126207, &mem_param_tmp_127988, "mem_param_tmp_127988") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126211, &mem_param_tmp_127989, "mem_param_tmp_127989") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126215, &mem_param_tmp_127990, "mem_param_tmp_127990") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126219, &mem_param_tmp_127991, "mem_param_tmp_127991") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126223, &mem_param_tmp_127992, "mem_param_tmp_127992") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126227, &mem_param_tmp_127993, "mem_param_tmp_127993") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126231, &mem_param_tmp_127994, "mem_param_tmp_127994") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126235, &mem_param_tmp_127995, "mem_param_tmp_127995") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126239, &mem_param_tmp_127996, "mem_param_tmp_127996") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126243, &mem_param_tmp_127997, "mem_param_tmp_127997") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126247, &mem_param_tmp_127998, "mem_param_tmp_127998") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126251, &mem_param_tmp_127999, "mem_param_tmp_127999") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126255, &mem_param_tmp_128000, "mem_param_tmp_128000") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126259, &mem_param_tmp_128001, "mem_param_tmp_128001") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126263, &mem_param_tmp_128002, "mem_param_tmp_128002") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126267, &mem_param_tmp_128003, "mem_param_tmp_128003") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126271, &mem_param_tmp_128004, "mem_param_tmp_128004") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126275, &mem_param_tmp_128005, "mem_param_tmp_128005") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126279, &mem_param_tmp_128006, "mem_param_tmp_128006") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_126283, &mem_param_tmp_128007, "mem_param_tmp_128007") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_127881, &mem_param_126179, "mem_param_126179") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127880, &mem_param_126183, "mem_param_126183") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127879, &mem_param_126187, "mem_param_126187") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127878, &mem_param_126191, "mem_param_126191") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127877, &mem_param_126195, "mem_param_126195") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127876, &mem_param_126199, "mem_param_126199") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127875, &mem_param_126203, "mem_param_126203") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127874, &mem_param_126207, "mem_param_126207") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127873, &mem_param_126211, "mem_param_126211") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127872, &mem_param_126215, "mem_param_126215") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127871, &mem_param_126219, "mem_param_126219") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127870, &mem_param_126223, "mem_param_126223") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127869, &mem_param_126227, "mem_param_126227") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127868, &mem_param_126231, "mem_param_126231") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127867, &mem_param_126235, "mem_param_126235") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127866, &mem_param_126239, "mem_param_126239") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127865, &mem_param_126243, "mem_param_126243") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127864, &mem_param_126247, "mem_param_126247") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127863, &mem_param_126251, "mem_param_126251") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127862, &mem_param_126255, "mem_param_126255") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127861, &mem_param_126259, "mem_param_126259") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127860, &mem_param_126263, "mem_param_126263") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127859, &mem_param_126267, "mem_param_126267") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127858, &mem_param_126271, "mem_param_126271") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127857, &mem_param_126275, "mem_param_126275") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127856, &mem_param_126279, "mem_param_126279") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_127855, &mem_param_126283, "mem_param_126283") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127954, &ext_mem_127876, "ext_mem_127876") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127955, &ext_mem_127878, "ext_mem_127878") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127956, &ext_mem_127877, "ext_mem_127877") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127957, &ext_mem_127880, "ext_mem_127880") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127958, &ext_mem_127874, "ext_mem_127874") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127959, &ext_mem_127879, "ext_mem_127879") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127960, &ext_mem_127875, "ext_mem_127875") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127961, &ext_mem_127881, "ext_mem_127881") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127962, &ext_mem_127873, "ext_mem_127873") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127963, &ext_mem_127867, "ext_mem_127867") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127964, &ext_mem_127869, "ext_mem_127869") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127965, &ext_mem_127868, "ext_mem_127868") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127966, &ext_mem_127871, "ext_mem_127871") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127967, &ext_mem_127865, "ext_mem_127865") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127968, &ext_mem_127870, "ext_mem_127870") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127969, &ext_mem_127866, "ext_mem_127866") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127970, &ext_mem_127872, "ext_mem_127872") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127971, &ext_mem_127864, "ext_mem_127864") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127972, &ext_mem_127858, "ext_mem_127858") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127973, &ext_mem_127860, "ext_mem_127860") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127974, &ext_mem_127859, "ext_mem_127859") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127975, &ext_mem_127862, "ext_mem_127862") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127976, &ext_mem_127856, "ext_mem_127856") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127977, &ext_mem_127861, "ext_mem_127861") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127978, &ext_mem_127857, "ext_mem_127857") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127979, &ext_mem_127863, "ext_mem_127863") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127980, &ext_mem_127855, "ext_mem_127855") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128429, &mem_out_127954, "mem_out_127954") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128430, &mem_out_127955, "mem_out_127955") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128431, &mem_out_127956, "mem_out_127956") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128432, &mem_out_127957, "mem_out_127957") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128433, &mem_out_127958, "mem_out_127958") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128434, &mem_out_127959, "mem_out_127959") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128435, &mem_out_127960, "mem_out_127960") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128436, &mem_out_127961, "mem_out_127961") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128437, &mem_out_127962, "mem_out_127962") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128438, &mem_out_127963, "mem_out_127963") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128439, &mem_out_127964, "mem_out_127964") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128440, &mem_out_127965, "mem_out_127965") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128441, &mem_out_127966, "mem_out_127966") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128442, &mem_out_127967, "mem_out_127967") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128443, &mem_out_127968, "mem_out_127968") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128444, &mem_out_127969, "mem_out_127969") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128445, &mem_out_127970, "mem_out_127970") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128446, &mem_out_127971, "mem_out_127971") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128447, &mem_out_127972, "mem_out_127972") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128448, &mem_out_127973, "mem_out_127973") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128449, &mem_out_127974, "mem_out_127974") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128450, &mem_out_127975, "mem_out_127975") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128451, &mem_out_127976, "mem_out_127976") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128452, &mem_out_127977, "mem_out_127977") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128453, &mem_out_127978, "mem_out_127978") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128454, &mem_out_127979, "mem_out_127979") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128455, &mem_out_127980, "mem_out_127980") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_126284);
        free(mem_126285);
        free(mem_126294);
        free(mem_126301);
        free(mem_126316);
        free(mem_126317);
        free(mem_126318);
        free(mem_126329);
        free(mem_126346);
        free(mem_126347);
        free(mem_126355);
        free(mem_126369);
        free(mem_126370);
        free(mem_126371);
        free(mem_126384);
        free(mem_126385);
        free(mem_126386);
        free(mem_126417);
        free(mem_126418);
        free(mem_126419);
        free(mem_126435);
        free(mem_126436);
        free(mem_126437);
        free(mem_126450);
        free(mem_126451);
        free(mem_126452);
        free(mem_126498);
        free(mem_126499);
        free(mem_126500);
        free(mem_126501);
        free(mem_126522);
        free(mem_126523);
        free(mem_126524);
        free(mem_126525);
        free(mem_126542);
        free(mem_126543);
        free(mem_126544);
        free(mem_126545);
        free(mem_126586);
        free(mem_126591);
        free(mem_126595);
        free(mem_126629);
        free(mem_126634);
        free(mem_126645);
        free(mem_126650);
        free(mem_126661);
        free(mem_126666);
        free(mem_126677);
        free(mem_126678);
        free(mem_126686);
        free(mem_126700);
        free(mem_126705);
        free(mem_126716);
        free(mem_126721);
        free(mem_126732);
        free(mem_126737);
        free(mem_126748);
        free(mem_126753);
        free(mem_126764);
        free(mem_126769);
        free(mem_126780);
        free(mem_126781);
        free(mem_126782);
        free(mem_126796);
        free(mem_126801);
        free(mem_126805);
        free(mem_126834);
        free(mem_126840);
        free(mem_126845);
        free(mem_126861);
        free(mem_126866);
        free(mem_126877);
        free(mem_126883);
        free(mem_126888);
        free(mem_126904);
        free(mem_126909);
        free(mem_126920);
        free(mem_126925);
        free(mem_126936);
        free(mem_126941);
        free(mem_126952);
        free(mem_126953);
        free(mem_126962);
        free(mem_126963);
        free(mem_126984);
        free(mem_126989);
        free(mem_127000);
        free(mem_127005);
        free(mem_127016);
        free(mem_127021);
        free(mem_127032);
        free(mem_127033);
        free(mem_127046);
        free(mem_127051);
        free(mem_127062);
        free(mem_127067);
        free(mem_127078);
        free(mem_127079);
        free(mem_127088);
        free(mem_127089);
        free(mem_127110);
        free(mem_127116);
        free(mem_127121);
        free(mem_127137);
        free(mem_127138);
        free(mem_127139);
        free(mem_127155);
        free(mem_127156);
        free(mem_127157);
        free(mem_127170);
        free(mem_127171);
        free(mem_127172);
        free(mem_127182);
        free(mem_127189);
        free(mem_127190);
        free(mem_127191);
        free(mem_127202);
        free(mem_127219);
        free(mem_127224);
        free(mem_127235);
        free(mem_127242);
        free(mem_127247);
        free(mem_127258);
        free(mem_127259);
        free(mem_127260);
        free(mem_127271);
        free(mem_127288);
        free(mem_127293);
        free(mem_127304);
        free(mem_127311);
        free(mem_127316);
        free(mem_127363);
        free(mem_127364);
        free(mem_127365);
        free(mem_127378);
        free(mem_127379);
        free(mem_127380);
        free(mem_127411);
        free(mem_127412);
        free(mem_127413);
        free(mem_127414);
        free(mem_127431);
        free(mem_127432);
        free(mem_127433);
        free(mem_127434);
        free(mem_127475);
        free(mem_127480);
        free(mem_127491);
        free(mem_127492);
        free(mem_127505);
        free(mem_127510);
        free(mem_127521);
        free(mem_127522);
        free(mem_127531);
        free(mem_127532);
        free(mem_127553);
        free(mem_127554);
        free(mem_127555);
        free(mem_127556);
        free(mem_127581);
        free(mem_127582);
        free(mem_127595);
        free(mem_127596);
        free(mem_127605);
        free(mem_127606);
        free(mem_127627);
        free(mem_127632);
        free(mem_127643);
        free(mem_127644);
        free(mem_127653);
        free(mem_127654);
        if (memblock_unref(ctx, &mem_param_tmp_128007, "mem_param_tmp_128007") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_128006, "mem_param_tmp_128006") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_128005, "mem_param_tmp_128005") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_128004, "mem_param_tmp_128004") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_128003, "mem_param_tmp_128003") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_128002, "mem_param_tmp_128002") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_128001, "mem_param_tmp_128001") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_128000, "mem_param_tmp_128000") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127999, "mem_param_tmp_127999") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127998, "mem_param_tmp_127998") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127997, "mem_param_tmp_127997") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127996, "mem_param_tmp_127996") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127995, "mem_param_tmp_127995") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127994, "mem_param_tmp_127994") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127993, "mem_param_tmp_127993") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127992, "mem_param_tmp_127992") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127991, "mem_param_tmp_127991") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127990, "mem_param_tmp_127990") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127989, "mem_param_tmp_127989") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127988, "mem_param_tmp_127988") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127987, "mem_param_tmp_127987") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127986, "mem_param_tmp_127986") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127985, "mem_param_tmp_127985") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127984, "mem_param_tmp_127984") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127983, "mem_param_tmp_127983") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127982, "mem_param_tmp_127982") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_127981, "mem_param_tmp_127981") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127771, "ext_mem_127771") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127772, "ext_mem_127772") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127773, "ext_mem_127773") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127769, "mem_127769") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127767, "mem_127767") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127765, "mem_127765") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127763, "mem_127763") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127760, "ext_mem_127760") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127761, "ext_mem_127761") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127762, "ext_mem_127762") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127758, "mem_127758") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127756, "mem_127756") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127754, "mem_127754") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127752, "mem_127752") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127749, "ext_mem_127749") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127750, "ext_mem_127750") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127751, "ext_mem_127751") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127747, "mem_127747") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127745, "mem_127745") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127743, "mem_127743") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127741, "mem_127741") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127738, "ext_mem_127738") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127739, "ext_mem_127739") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127740, "ext_mem_127740") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127736, "mem_127736") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127734, "mem_127734") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127732, "mem_127732") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127730, "mem_127730") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127727, "ext_mem_127727") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127728, "ext_mem_127728") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127729, "ext_mem_127729") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127725, "mem_127725") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127723, "mem_127723") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127721, "mem_127721") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127719, "mem_127719") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127716, "ext_mem_127716") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127717, "ext_mem_127717") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127718, "ext_mem_127718") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127714, "mem_127714") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127712, "mem_127712") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127710, "mem_127710") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127708, "mem_127708") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127705, "ext_mem_127705") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127706, "ext_mem_127706") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127707, "ext_mem_127707") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127703, "mem_127703") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127701, "mem_127701") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127699, "mem_127699") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127697, "mem_127697") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127694, "ext_mem_127694") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127695, "ext_mem_127695") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127696, "ext_mem_127696") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127692, "mem_127692") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127690, "mem_127690") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127688, "mem_127688") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127686, "mem_127686") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127683, "ext_mem_127683") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127684, "ext_mem_127684") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127685, "ext_mem_127685") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127681, "mem_127681") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127679, "mem_127679") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127677, "mem_127677") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_127675, "mem_127675") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126283, "mem_param_126283") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126279, "mem_param_126279") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126275, "mem_param_126275") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126271, "mem_param_126271") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126267, "mem_param_126267") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126263, "mem_param_126263") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126259, "mem_param_126259") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126255, "mem_param_126255") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126251, "mem_param_126251") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126247, "mem_param_126247") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126243, "mem_param_126243") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126239, "mem_param_126239") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126235, "mem_param_126235") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126231, "mem_param_126231") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126227, "mem_param_126227") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126223, "mem_param_126223") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126219, "mem_param_126219") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126215, "mem_param_126215") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126211, "mem_param_126211") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126207, "mem_param_126207") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126203, "mem_param_126203") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126199, "mem_param_126199") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126195, "mem_param_126195") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126191, "mem_param_126191") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126187, "mem_param_126187") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126183, "mem_param_126183") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_126179, "mem_param_126179") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127855, "ext_mem_127855") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127856, "ext_mem_127856") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127857, "ext_mem_127857") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127858, "ext_mem_127858") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127859, "ext_mem_127859") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127860, "ext_mem_127860") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127861, "ext_mem_127861") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127862, "ext_mem_127862") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127863, "ext_mem_127863") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127864, "ext_mem_127864") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127865, "ext_mem_127865") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127866, "ext_mem_127866") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127867, "ext_mem_127867") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127868, "ext_mem_127868") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127869, "ext_mem_127869") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127870, "ext_mem_127870") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127871, "ext_mem_127871") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127872, "ext_mem_127872") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127873, "ext_mem_127873") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127874, "ext_mem_127874") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127875, "ext_mem_127875") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127876, "ext_mem_127876") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127877, "ext_mem_127877") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127878, "ext_mem_127878") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127879, "ext_mem_127879") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127880, "ext_mem_127880") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_127881, "ext_mem_127881") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127980, "mem_out_127980") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127979, "mem_out_127979") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127978, "mem_out_127978") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127977, "mem_out_127977") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127976, "mem_out_127976") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127975, "mem_out_127975") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127974, "mem_out_127974") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127973, "mem_out_127973") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127972, "mem_out_127972") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127971, "mem_out_127971") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127970, "mem_out_127970") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127969, "mem_out_127969") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127968, "mem_out_127968") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127967, "mem_out_127967") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127966, "mem_out_127966") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127965, "mem_out_127965") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127964, "mem_out_127964") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127963, "mem_out_127963") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127962, "mem_out_127962") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127961, "mem_out_127961") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127960, "mem_out_127960") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127959, "mem_out_127959") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127958, "mem_out_127958") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127957, "mem_out_127957") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127956, "mem_out_127956") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127955, "mem_out_127955") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127954, "mem_out_127954") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_128627, struct memblock *mem_out_p_128628, struct memblock *mem_out_p_128629, struct memblock *mem_out_p_128630, struct memblock *mem_out_p_128631, struct memblock *mem_out_p_128632, struct memblock *mem_out_p_128633, struct memblock *mem_out_p_128634, struct memblock *mem_out_p_128635)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_127962;
    
    mem_out_127962.references = NULL;
    
    struct memblock mem_out_127961;
    
    mem_out_127961.references = NULL;
    
    struct memblock mem_out_127960;
    
    mem_out_127960.references = NULL;
    
    struct memblock mem_out_127959;
    
    mem_out_127959.references = NULL;
    
    struct memblock mem_out_127958;
    
    mem_out_127958.references = NULL;
    
    struct memblock mem_out_127957;
    
    mem_out_127957.references = NULL;
    
    struct memblock mem_out_127956;
    
    mem_out_127956.references = NULL;
    
    struct memblock mem_out_127955;
    
    mem_out_127955.references = NULL;
    
    struct memblock mem_out_127954;
    
    mem_out_127954.references = NULL;
    
    struct memblock mem_126137 = ctx->constants->mem_126137;
    struct memblock mem_126138 = ctx->constants->mem_126138;
    struct memblock mem_126139 = ctx->constants->mem_126139;
    struct memblock mem_126140 = ctx->constants->mem_126140;
    struct memblock mem_126141 = ctx->constants->mem_126141;
    struct memblock mem_126142 = ctx->constants->mem_126142;
    struct memblock mem_126143 = ctx->constants->mem_126143;
    struct memblock mem_126144 = ctx->constants->mem_126144;
    struct memblock mem_126145 = ctx->constants->mem_126145;
    
    if (memblock_set(ctx, &mem_out_127954, &mem_126144, "mem_126144") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127955, &mem_126140, "mem_126140") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127956, &mem_126142, "mem_126142") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127957, &mem_126138, "mem_126138") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127958, &mem_126139, "mem_126139") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127959, &mem_126137, "mem_126137") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127960, &mem_126143, "mem_126143") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127961, &mem_126141, "mem_126141") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_127962, &mem_126145, "mem_126145") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128627, &mem_out_127954, "mem_out_127954") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128628, &mem_out_127955, "mem_out_127955") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128629, &mem_out_127956, "mem_out_127956") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128630, &mem_out_127957, "mem_out_127957") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128631, &mem_out_127958, "mem_out_127958") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128632, &mem_out_127959, "mem_out_127959") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128633, &mem_out_127960, "mem_out_127960") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128634, &mem_out_127961, "mem_out_127961") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_128635, &mem_out_127962, "mem_out_127962") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_127962, "mem_out_127962") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127961, "mem_out_127961") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127960, "mem_out_127960") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127959, "mem_out_127959") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127958, "mem_out_127958") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127957, "mem_out_127957") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127956, "mem_out_127956") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127955, "mem_out_127955") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_127954, "mem_out_127954") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_cal_loss(struct futhark_context *ctx, struct futhark_opaque_tup2_f64_arr1d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3)
{
    double prim_out_127955 = 0.0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_127954;
    
    mem_out_127954.references = NULL;
    
    struct memblock mask_mem_126157;
    
    mask_mem_126157.references = NULL;
    
    struct memblock target_mem_126156;
    
    target_mem_126156.references = NULL;
    
    struct memblock tokens_mem_126155;
    
    tokens_mem_126155.references = NULL;
    
    struct memblock wvoc_mem_126154;
    
    wvoc_mem_126154.references = NULL;
    
    struct memblock wval_mem_126153;
    
    wval_mem_126153.references = NULL;
    
    struct memblock wup_mem_126152;
    
    wup_mem_126152.references = NULL;
    
    struct memblock wte_mem_126151;
    
    wte_mem_126151.references = NULL;
    
    struct memblock wqry_mem_126150;
    
    wqry_mem_126150.references = NULL;
    
    struct memblock wpe_mem_126149;
    
    wpe_mem_126149.references = NULL;
    
    struct memblock wout_mem_126148;
    
    wout_mem_126148.references = NULL;
    
    struct memblock wkey_mem_126147;
    
    wkey_mem_126147.references = NULL;
    
    struct memblock wdown_mem_126146;
    
    wdown_mem_126146.references = NULL;
    wdown_mem_126146 = in0->v0->mem;
    wkey_mem_126147 = in0->v1->mem;
    wout_mem_126148 = in0->v2->mem;
    wpe_mem_126149 = in0->v3->mem;
    wqry_mem_126150 = in0->v4->mem;
    wte_mem_126151 = in0->v5->mem;
    wup_mem_126152 = in0->v6->mem;
    wval_mem_126153 = in0->v7->mem;
    wvoc_mem_126154 = in0->v8->mem;
    tokens_mem_126155 = in1->mem;
    target_mem_126156 = in2->mem;
    mask_mem_126157 = in3->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && (((int64_t) 16 == in2->shape[0] && (int64_t) 27 == in2->shape[1]) && ((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_cal_loss(ctx, &mem_out_127954, &prim_out_127955, wdown_mem_126146, wkey_mem_126147, wout_mem_126148, wpe_mem_126149, wqry_mem_126150, wte_mem_126151, wup_mem_126152, wval_mem_126153, wvoc_mem_126154, tokens_mem_126155, target_mem_126156, mask_mem_126157);
        if (ret == 0) {
            struct memblock mem_126137 = ctx->constants->mem_126137;
            struct memblock mem_126138 = ctx->constants->mem_126138;
            struct memblock mem_126139 = ctx->constants->mem_126139;
            struct memblock mem_126140 = ctx->constants->mem_126140;
            struct memblock mem_126141 = ctx->constants->mem_126141;
            struct memblock mem_126142 = ctx->constants->mem_126142;
            struct memblock mem_126143 = ctx->constants->mem_126143;
            struct memblock mem_126144 = ctx->constants->mem_126144;
            struct memblock mem_126145 = ctx->constants->mem_126145;
            
            assert((*out = (struct futhark_opaque_tup2_f64_arr1d_f64 *) malloc(sizeof(struct futhark_opaque_tup2_f64_arr1d_f64))) != NULL);
            (*out)->v0 = prim_out_127955;
            assert(((*out)->v1 = (struct futhark_f64_1d *) malloc(sizeof(struct futhark_f64_1d))) != NULL);
            (*out)->v1->mem = mem_out_127954;
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
    
    struct memblock mem_out_127954;
    
    mem_out_127954.references = NULL;
    
    struct memblock mask_mem_126156;
    
    mask_mem_126156.references = NULL;
    
    struct memblock tokens_mem_126155;
    
    tokens_mem_126155.references = NULL;
    
    struct memblock wvoc_mem_126154;
    
    wvoc_mem_126154.references = NULL;
    
    struct memblock wval_mem_126153;
    
    wval_mem_126153.references = NULL;
    
    struct memblock wup_mem_126152;
    
    wup_mem_126152.references = NULL;
    
    struct memblock wte_mem_126151;
    
    wte_mem_126151.references = NULL;
    
    struct memblock wqry_mem_126150;
    
    wqry_mem_126150.references = NULL;
    
    struct memblock wpe_mem_126149;
    
    wpe_mem_126149.references = NULL;
    
    struct memblock wout_mem_126148;
    
    wout_mem_126148.references = NULL;
    
    struct memblock wkey_mem_126147;
    
    wkey_mem_126147.references = NULL;
    
    struct memblock wdown_mem_126146;
    
    wdown_mem_126146.references = NULL;
    wdown_mem_126146 = in0->v0->mem;
    wkey_mem_126147 = in0->v1->mem;
    wout_mem_126148 = in0->v2->mem;
    wpe_mem_126149 = in0->v3->mem;
    wqry_mem_126150 = in0->v4->mem;
    wte_mem_126151 = in0->v5->mem;
    wup_mem_126152 = in0->v6->mem;
    wval_mem_126153 = in0->v7->mem;
    wvoc_mem_126154 = in0->v8->mem;
    tokens_mem_126155 = in1->mem;
    mask_mem_126156 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_127954, wdown_mem_126146, wkey_mem_126147, wout_mem_126148, wpe_mem_126149, wqry_mem_126150, wte_mem_126151, wup_mem_126152, wval_mem_126153, wvoc_mem_126154, tokens_mem_126155, mask_mem_126156);
        if (ret == 0) {
            struct memblock mem_126137 = ctx->constants->mem_126137;
            struct memblock mem_126138 = ctx->constants->mem_126138;
            struct memblock mem_126139 = ctx->constants->mem_126139;
            struct memblock mem_126140 = ctx->constants->mem_126140;
            struct memblock mem_126141 = ctx->constants->mem_126141;
            struct memblock mem_126142 = ctx->constants->mem_126142;
            struct memblock mem_126143 = ctx->constants->mem_126143;
            struct memblock mem_126144 = ctx->constants->mem_126144;
            struct memblock mem_126145 = ctx->constants->mem_126145;
            
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_127954;
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
    
    struct memblock mem_out_127962;
    
    mem_out_127962.references = NULL;
    
    struct memblock mem_out_127961;
    
    mem_out_127961.references = NULL;
    
    struct memblock mem_out_127960;
    
    mem_out_127960.references = NULL;
    
    struct memblock mem_out_127959;
    
    mem_out_127959.references = NULL;
    
    struct memblock mem_out_127958;
    
    mem_out_127958.references = NULL;
    
    struct memblock mem_out_127957;
    
    mem_out_127957.references = NULL;
    
    struct memblock mem_out_127956;
    
    mem_out_127956.references = NULL;
    
    struct memblock mem_out_127955;
    
    mem_out_127955.references = NULL;
    
    struct memblock mem_out_127954;
    
    mem_out_127954.references = NULL;
    
    struct memblock wvoc_mem_126154;
    
    wvoc_mem_126154.references = NULL;
    
    struct memblock wdown_mem_126153;
    
    wdown_mem_126153.references = NULL;
    
    struct memblock wup_mem_126152;
    
    wup_mem_126152.references = NULL;
    
    struct memblock wout_mem_126151;
    
    wout_mem_126151.references = NULL;
    
    struct memblock wval_mem_126150;
    
    wval_mem_126150.references = NULL;
    
    struct memblock wkey_mem_126149;
    
    wkey_mem_126149.references = NULL;
    
    struct memblock wqry_mem_126148;
    
    wqry_mem_126148.references = NULL;
    
    struct memblock wpe_mem_126147;
    
    wpe_mem_126147.references = NULL;
    
    struct memblock wte_mem_126146;
    
    wte_mem_126146.references = NULL;
    wte_mem_126146 = in0->mem;
    wpe_mem_126147 = in1->mem;
    wqry_mem_126148 = in2->mem;
    wkey_mem_126149 = in3->mem;
    wval_mem_126150 = in4->mem;
    wout_mem_126151 = in5->mem;
    wup_mem_126152 = in6->mem;
    wdown_mem_126153 = in7->mem;
    wvoc_mem_126154 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_127954, &mem_out_127955, &mem_out_127956, &mem_out_127957, &mem_out_127958, &mem_out_127959, &mem_out_127960, &mem_out_127961, &mem_out_127962, wte_mem_126146, wpe_mem_126147, wqry_mem_126148, wkey_mem_126149, wval_mem_126150, wout_mem_126151, wup_mem_126152, wdown_mem_126153, wvoc_mem_126154);
        if (ret == 0) {
            struct memblock mem_126137 = ctx->constants->mem_126137;
            struct memblock mem_126138 = ctx->constants->mem_126138;
            struct memblock mem_126139 = ctx->constants->mem_126139;
            struct memblock mem_126140 = ctx->constants->mem_126140;
            struct memblock mem_126141 = ctx->constants->mem_126141;
            struct memblock mem_126142 = ctx->constants->mem_126142;
            struct memblock mem_126143 = ctx->constants->mem_126143;
            struct memblock mem_126144 = ctx->constants->mem_126144;
            struct memblock mem_126145 = ctx->constants->mem_126145;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_127954;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_127955;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_127956;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_127957;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_127958;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_127959;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_127960;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_127961;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_127962;
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
    
    struct memblock mem_out_127980;
    
    mem_out_127980.references = NULL;
    
    struct memblock mem_out_127979;
    
    mem_out_127979.references = NULL;
    
    struct memblock mem_out_127978;
    
    mem_out_127978.references = NULL;
    
    struct memblock mem_out_127977;
    
    mem_out_127977.references = NULL;
    
    struct memblock mem_out_127976;
    
    mem_out_127976.references = NULL;
    
    struct memblock mem_out_127975;
    
    mem_out_127975.references = NULL;
    
    struct memblock mem_out_127974;
    
    mem_out_127974.references = NULL;
    
    struct memblock mem_out_127973;
    
    mem_out_127973.references = NULL;
    
    struct memblock mem_out_127972;
    
    mem_out_127972.references = NULL;
    
    struct memblock mem_out_127971;
    
    mem_out_127971.references = NULL;
    
    struct memblock mem_out_127970;
    
    mem_out_127970.references = NULL;
    
    struct memblock mem_out_127969;
    
    mem_out_127969.references = NULL;
    
    struct memblock mem_out_127968;
    
    mem_out_127968.references = NULL;
    
    struct memblock mem_out_127967;
    
    mem_out_127967.references = NULL;
    
    struct memblock mem_out_127966;
    
    mem_out_127966.references = NULL;
    
    struct memblock mem_out_127965;
    
    mem_out_127965.references = NULL;
    
    struct memblock mem_out_127964;
    
    mem_out_127964.references = NULL;
    
    struct memblock mem_out_127963;
    
    mem_out_127963.references = NULL;
    
    struct memblock mem_out_127962;
    
    mem_out_127962.references = NULL;
    
    struct memblock mem_out_127961;
    
    mem_out_127961.references = NULL;
    
    struct memblock mem_out_127960;
    
    mem_out_127960.references = NULL;
    
    struct memblock mem_out_127959;
    
    mem_out_127959.references = NULL;
    
    struct memblock mem_out_127958;
    
    mem_out_127958.references = NULL;
    
    struct memblock mem_out_127957;
    
    mem_out_127957.references = NULL;
    
    struct memblock mem_out_127956;
    
    mem_out_127956.references = NULL;
    
    struct memblock mem_out_127955;
    
    mem_out_127955.references = NULL;
    
    struct memblock mem_out_127954;
    
    mem_out_127954.references = NULL;
    
    struct memblock seqs_mem_126175;
    
    seqs_mem_126175.references = NULL;
    
    struct memblock dls_mem_126174;
    
    dls_mem_126174.references = NULL;
    
    struct memblock masks_mem_126173;
    
    masks_mem_126173.references = NULL;
    
    struct memblock wvoc_mem_126172;
    
    wvoc_mem_126172.references = NULL;
    
    struct memblock wval_mem_126171;
    
    wval_mem_126171.references = NULL;
    
    struct memblock wup_mem_126170;
    
    wup_mem_126170.references = NULL;
    
    struct memblock wte_mem_126169;
    
    wte_mem_126169.references = NULL;
    
    struct memblock wqry_mem_126168;
    
    wqry_mem_126168.references = NULL;
    
    struct memblock wpe_mem_126167;
    
    wpe_mem_126167.references = NULL;
    
    struct memblock wout_mem_126166;
    
    wout_mem_126166.references = NULL;
    
    struct memblock wkey_mem_126165;
    
    wkey_mem_126165.references = NULL;
    
    struct memblock wdown_mem_126164;
    
    wdown_mem_126164.references = NULL;
    
    struct memblock wvoc_mem_126163;
    
    wvoc_mem_126163.references = NULL;
    
    struct memblock wval_mem_126162;
    
    wval_mem_126162.references = NULL;
    
    struct memblock wup_mem_126161;
    
    wup_mem_126161.references = NULL;
    
    struct memblock wte_mem_126160;
    
    wte_mem_126160.references = NULL;
    
    struct memblock wqry_mem_126159;
    
    wqry_mem_126159.references = NULL;
    
    struct memblock wpe_mem_126158;
    
    wpe_mem_126158.references = NULL;
    
    struct memblock wout_mem_126157;
    
    wout_mem_126157.references = NULL;
    
    struct memblock wkey_mem_126156;
    
    wkey_mem_126156.references = NULL;
    
    struct memblock wdown_mem_126155;
    
    wdown_mem_126155.references = NULL;
    
    struct memblock wvoc_mem_126154;
    
    wvoc_mem_126154.references = NULL;
    
    struct memblock wval_mem_126153;
    
    wval_mem_126153.references = NULL;
    
    struct memblock wup_mem_126152;
    
    wup_mem_126152.references = NULL;
    
    struct memblock wte_mem_126151;
    
    wte_mem_126151.references = NULL;
    
    struct memblock wqry_mem_126150;
    
    wqry_mem_126150.references = NULL;
    
    struct memblock wpe_mem_126149;
    
    wpe_mem_126149.references = NULL;
    
    struct memblock wout_mem_126148;
    
    wout_mem_126148.references = NULL;
    
    struct memblock wkey_mem_126147;
    
    wkey_mem_126147.references = NULL;
    
    struct memblock wdown_mem_126146;
    
    wdown_mem_126146.references = NULL;
    wdown_mem_126146 = in0->v0->mem;
    wkey_mem_126147 = in0->v1->mem;
    wout_mem_126148 = in0->v2->mem;
    wpe_mem_126149 = in0->v3->mem;
    wqry_mem_126150 = in0->v4->mem;
    wte_mem_126151 = in0->v5->mem;
    wup_mem_126152 = in0->v6->mem;
    wval_mem_126153 = in0->v7->mem;
    wvoc_mem_126154 = in0->v8->mem;
    wdown_mem_126155 = in1->v0->mem;
    wkey_mem_126156 = in1->v1->mem;
    wout_mem_126157 = in1->v2->mem;
    wpe_mem_126158 = in1->v3->mem;
    wqry_mem_126159 = in1->v4->mem;
    wte_mem_126160 = in1->v5->mem;
    wup_mem_126161 = in1->v6->mem;
    wval_mem_126162 = in1->v7->mem;
    wvoc_mem_126163 = in1->v8->mem;
    wdown_mem_126164 = in2->v0->mem;
    wkey_mem_126165 = in2->v1->mem;
    wout_mem_126166 = in2->v2->mem;
    wpe_mem_126167 = in2->v3->mem;
    wqry_mem_126168 = in2->v4->mem;
    wte_mem_126169 = in2->v5->mem;
    wup_mem_126170 = in2->v6->mem;
    wval_mem_126171 = in2->v7->mem;
    wvoc_mem_126172 = in2->v8->mem;
    masks_mem_126173 = in3->mem;
    dls_mem_126174 = in4->mem;
    seqs_mem_126175 = in5->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && (((int64_t) 500 == in3->shape[0] && ((int64_t) 16 == in3->shape[1] && (int64_t) 16 == in3->shape[2])) && ((int64_t) 500 == in4->shape[0] && ((int64_t) 500 == in5->shape[0] && (int64_t) 16 == in5->shape[1]))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_127954, &mem_out_127955, &mem_out_127956, &mem_out_127957, &mem_out_127958, &mem_out_127959, &mem_out_127960, &mem_out_127961, &mem_out_127962, &mem_out_127963, &mem_out_127964, &mem_out_127965, &mem_out_127966, &mem_out_127967, &mem_out_127968, &mem_out_127969, &mem_out_127970, &mem_out_127971, &mem_out_127972, &mem_out_127973, &mem_out_127974, &mem_out_127975, &mem_out_127976, &mem_out_127977, &mem_out_127978, &mem_out_127979, &mem_out_127980, wdown_mem_126146, wkey_mem_126147, wout_mem_126148, wpe_mem_126149, wqry_mem_126150, wte_mem_126151, wup_mem_126152, wval_mem_126153, wvoc_mem_126154, wdown_mem_126155, wkey_mem_126156, wout_mem_126157, wpe_mem_126158, wqry_mem_126159, wte_mem_126160, wup_mem_126161, wval_mem_126162, wvoc_mem_126163, wdown_mem_126164, wkey_mem_126165, wout_mem_126166, wpe_mem_126167, wqry_mem_126168, wte_mem_126169, wup_mem_126170, wval_mem_126171, wvoc_mem_126172, masks_mem_126173, dls_mem_126174, seqs_mem_126175);
        if (ret == 0) {
            struct memblock mem_126137 = ctx->constants->mem_126137;
            struct memblock mem_126138 = ctx->constants->mem_126138;
            struct memblock mem_126139 = ctx->constants->mem_126139;
            struct memblock mem_126140 = ctx->constants->mem_126140;
            struct memblock mem_126141 = ctx->constants->mem_126141;
            struct memblock mem_126142 = ctx->constants->mem_126142;
            struct memblock mem_126143 = ctx->constants->mem_126143;
            struct memblock mem_126144 = ctx->constants->mem_126144;
            struct memblock mem_126145 = ctx->constants->mem_126145;
            
            assert((*out = (struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_127954;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_127955;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_127956;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_127957;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_127958;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_127959;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_127960;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_127961;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_127962;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v9->mem = mem_out_127963;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v10->mem = mem_out_127964;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v11->mem = mem_out_127965;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v12->mem = mem_out_127966;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v13->mem = mem_out_127967;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v14->mem = mem_out_127968;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v15->mem = mem_out_127969;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v16->mem = mem_out_127970;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v17->mem = mem_out_127971;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v18->mem = mem_out_127972;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v19->mem = mem_out_127973;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v20->mem = mem_out_127974;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v21->mem = mem_out_127975;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v22->mem = mem_out_127976;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v23->mem = mem_out_127977;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v24->mem = mem_out_127978;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v25->mem = mem_out_127979;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v26->mem = mem_out_127980;
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
    
    struct memblock mem_out_127962;
    
    mem_out_127962.references = NULL;
    
    struct memblock mem_out_127961;
    
    mem_out_127961.references = NULL;
    
    struct memblock mem_out_127960;
    
    mem_out_127960.references = NULL;
    
    struct memblock mem_out_127959;
    
    mem_out_127959.references = NULL;
    
    struct memblock mem_out_127958;
    
    mem_out_127958.references = NULL;
    
    struct memblock mem_out_127957;
    
    mem_out_127957.references = NULL;
    
    struct memblock mem_out_127956;
    
    mem_out_127956.references = NULL;
    
    struct memblock mem_out_127955;
    
    mem_out_127955.references = NULL;
    
    struct memblock mem_out_127954;
    
    mem_out_127954.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_127954, &mem_out_127955, &mem_out_127956, &mem_out_127957, &mem_out_127958, &mem_out_127959, &mem_out_127960, &mem_out_127961, &mem_out_127962);
        if (ret == 0) {
            struct memblock mem_126137 = ctx->constants->mem_126137;
            struct memblock mem_126138 = ctx->constants->mem_126138;
            struct memblock mem_126139 = ctx->constants->mem_126139;
            struct memblock mem_126140 = ctx->constants->mem_126140;
            struct memblock mem_126141 = ctx->constants->mem_126141;
            struct memblock mem_126142 = ctx->constants->mem_126142;
            struct memblock mem_126143 = ctx->constants->mem_126143;
            struct memblock mem_126144 = ctx->constants->mem_126144;
            struct memblock mem_126145 = ctx->constants->mem_126145;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_127954;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_127955;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_127956;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_127957;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_127958;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_127959;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_127960;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_127961;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_127962;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
