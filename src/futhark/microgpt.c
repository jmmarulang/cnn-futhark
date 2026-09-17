
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
struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64;
struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64;
struct futhark_opaque_params;
int futhark_free_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64(struct futhark_context *ctx, struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 *obj);
int futhark_store_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64(struct futhark_context *ctx, const struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 *obj, void **p, size_t *n);
struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 *futhark_restore_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64(struct futhark_context *ctx, const void *p);
int futhark_project_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64_0(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 *obj);
int futhark_project_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64_1(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 *obj);
int futhark_project_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64_2(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 *obj);
int futhark_project_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64_3(struct futhark_context *ctx, struct futhark_f64_1d **out, const struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 *obj);
int futhark_new_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64(struct futhark_context *ctx, struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *f_0, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *f_1, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *f_2, const struct futhark_f64_1d *f_3);
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
int futhark_entry_cal_loss(struct futhark_context *ctx, double *out, const int64_t in0, const struct futhark_opaque_params *in1, const struct futhark_i64_1d *in2, const struct futhark_f64_2d *in3);
int futhark_entry_forward_seq(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2);
int futhark_entry_to_params(struct futhark_context *ctx, struct futhark_opaque_params **out, const struct futhark_f64_2d *in0, const struct futhark_f64_2d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3, const struct futhark_f64_2d *in4, const struct futhark_f64_2d *in5, const struct futhark_f64_2d *in6, const struct futhark_f64_2d *in7, const struct futhark_f64_2d *in8);
int futhark_entry_train(struct futhark_context *ctx, struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_opaque_params *in1, const struct futhark_opaque_params *in2, const struct futhark_f64_3d *in3, const struct futhark_i64_1d *in4, const struct futhark_i64_2d *in5);
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

const struct type type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZMZNf64ZR;
const struct type type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR;
const struct type type_ZMZNZMZNZMZNf64;
const struct type type_ZMZNZMZNf64;
const struct type type_ZMZNZMZNi64;
const struct type type_ZMZNf64;
const struct type type_ZMZNi64;
const struct type type_params;
const struct field type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZMZNf64ZR_fields[] = {{.name ="0", .type =&type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR, .project =(project_fn) futhark_project_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64_0}, {.name ="1", .type =&type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR, .project =(project_fn) futhark_project_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64_1}, {.name ="2", .type =&type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR, .project =(project_fn) futhark_project_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64_2}, {.name ="3", .type =&type_ZMZNf64, .project =(project_fn) futhark_project_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64_3}};
int futhark_new_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64_wrap(struct futhark_context *ctx, void **outp, const void *fields[])
{
    struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 * *out = (struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 * *) outp;
    const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 * v0 = *(const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 * *) fields[0];
    const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 * v1 = *(const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 * *) fields[1];
    const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 * v2 = *(const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 * *) fields[2];
    const struct futhark_f64_1d * v3 = *(const struct futhark_f64_1d * *) fields[3];
    
    return futhark_new_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64(ctx, out, v0, v1, v2, v3);
}
const struct record type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZMZNf64ZR_record = {.num_fields =4, .fields =type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZMZNf64ZR_fields, .new =futhark_new_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64_wrap};
const struct opaque_aux type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZMZNf64ZR_aux = {.store =(opaque_store_fn) futhark_store_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64, .restore =(opaque_restore_fn) futhark_restore_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64, .free =(opaque_free_fn) futhark_free_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64};
const struct type type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZMZNf64ZR = {.name ="(([][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64), ([][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64), ([][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64, [][]f64), []f64)", .restore =(restore_fn) restore_opaque, .store =(store_fn) store_opaque, .free =(free_fn) free_opaque, .aux =&type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZMZNf64ZR_aux, .kind =RECORD, .info =&type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZMZNf64ZR_record};
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
const struct type *cal_loss_in_types[] = {&type_i64, &type_params, &type_ZMZNi64, &type_ZMZNZMZNf64, NULL};
bool cal_loss_in_unique[] = {false, false, false, false};
const char *cal_loss_tuning_params[] = {NULL};
const char *cal_loss_attrs[] = {NULL};
int call_cal_loss(struct futhark_context *ctx, void *out, void **ins)
{
    int64_t in0 = *(int64_t *) ins[0];
    struct futhark_opaque_params * in1 = *(struct futhark_opaque_params * *) ins[1];
    struct futhark_i64_1d * in2 = *(struct futhark_i64_1d * *) ins[2];
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
const struct type *types[] = {&type_i8, &type_i16, &type_i32, &type_i64, &type_u8, &type_u16, &type_u32, &type_u64, &type_f16, &type_f32, &type_f64, &type_bool, &type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZMZNf64ZR, &type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR, &type_ZMZNZMZNZMZNf64, &type_ZMZNZMZNf64, &type_ZMZNZMZNi64, &type_ZMZNf64, &type_ZMZNi64, &type_params, NULL};
struct entry_point entry_points[] = {{.name ="cal_loss", .f =call_cal_loss, .tuning_params =cal_loss_tuning_params, .in_types =cal_loss_in_types, .out_type =&type_f64, .in_unique =cal_loss_in_unique, .out_unique =false, .attrs =cal_loss_attrs}, {.name ="forward_seq", .f =call_forward_seq, .tuning_params =forward_seq_tuning_params, .in_types =forward_seq_in_types, .out_type =&type_ZMZNZMZNf64, .in_unique =forward_seq_in_unique, .out_unique =false, .attrs =forward_seq_attrs}, {.name ="to_params", .f =call_to_params, .tuning_params =to_params_tuning_params, .in_types =to_params_in_types, .out_type =&type_params, .in_unique =to_params_in_unique, .out_unique =false, .attrs =to_params_attrs}, {.name ="train", .f =call_train, .tuning_params =train_tuning_params, .in_types =train_in_types, .out_type =&type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZMZNf64ZR, .in_unique =train_in_unique, .out_unique =false, .attrs =train_attrs}, {.name ="zero_params", .f =call_zzero_params, .tuning_params =zzero_params_tuning_params, .in_types =zzero_params_in_types, .out_type =&type_params, .in_unique =zzero_params_in_unique, .out_unique =false, .attrs =zzero_params_attrs}, {.name =NULL}};
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
    struct memblock mem_102600;
    struct memblock mem_102601;
    struct memblock mem_102602;
    struct memblock mem_102603;
    struct memblock mem_102604;
    struct memblock mem_102605;
    struct memblock mem_102606;
    struct memblock mem_102607;
    struct memblock mem_102608;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_11260(struct futhark_context *ctx, struct memblock *mem_out_p_105287, struct memblock *mem_out_p_105288, struct memblock *mem_out_p_105289, struct memblock w_mem_102609, struct memblock mw_mem_102610, struct memblock vw_mem_102611, struct memblock dw_mem_102612, int64_t n_72507, int64_t m_72508, int64_t step_72513, double lt_r_72514);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_11261(struct futhark_context *ctx, struct memblock *mem_out_p_105293, struct memblock *mem_out_p_105294, struct memblock *mem_out_p_105295, struct memblock w_mem_102609, struct memblock mw_mem_102610, struct memblock vw_mem_102611, struct memblock dw_mem_102612, int64_t n_73540, int64_t m_73541, int64_t step_73546, double lt_r_73547);
FUTHARK_FUN_ATTR int futrts_cal_target_8179(struct futhark_context *ctx, struct memblock *mem_out_p_105299, struct memblock tokens_mem_102609, int64_t n_45888);
FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, double *out_prim_out_105301, struct memblock wdown_mem_102609, struct memblock wkey_mem_102610, struct memblock wout_mem_102611, struct memblock wpe_mem_102612, struct memblock wqry_mem_102613, struct memblock wte_mem_102614, struct memblock wup_mem_102615, struct memblock wval_mem_102616, struct memblock wvoc_mem_102617, struct memblock tokens_mem_102618, struct memblock mask_mem_102619, int64_t dl_52055);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_105361, struct memblock wdown_mem_102609, struct memblock wkey_mem_102610, struct memblock wout_mem_102611, struct memblock wpe_mem_102612, struct memblock wqry_mem_102613, struct memblock wte_mem_102614, struct memblock wup_mem_102615, struct memblock wval_mem_102616, struct memblock wvoc_mem_102617, struct memblock tokens_mem_102618, struct memblock mask_mem_102619);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_105416, struct memblock *mem_out_p_105417, struct memblock *mem_out_p_105418, struct memblock *mem_out_p_105419, struct memblock *mem_out_p_105420, struct memblock *mem_out_p_105421, struct memblock *mem_out_p_105422, struct memblock *mem_out_p_105423, struct memblock *mem_out_p_105424, struct memblock wte_mem_102609, struct memblock wpe_mem_102610, struct memblock wqry_mem_102611, struct memblock wkey_mem_102612, struct memblock wval_mem_102613, struct memblock wout_mem_102614, struct memblock wup_mem_102615, struct memblock wdown_mem_102616, struct memblock wvoc_mem_102617);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_105425, struct memblock *mem_out_p_105426, struct memblock *mem_out_p_105427, struct memblock *mem_out_p_105428, struct memblock *mem_out_p_105429, struct memblock *mem_out_p_105430, struct memblock *mem_out_p_105431, struct memblock *mem_out_p_105432, struct memblock *mem_out_p_105433, struct memblock *mem_out_p_105434, struct memblock *mem_out_p_105435, struct memblock *mem_out_p_105436, struct memblock *mem_out_p_105437, struct memblock *mem_out_p_105438, struct memblock *mem_out_p_105439, struct memblock *mem_out_p_105440, struct memblock *mem_out_p_105441, struct memblock *mem_out_p_105442, struct memblock *mem_out_p_105443, struct memblock *mem_out_p_105444, struct memblock *mem_out_p_105445, struct memblock *mem_out_p_105446, struct memblock *mem_out_p_105447, struct memblock *mem_out_p_105448, struct memblock *mem_out_p_105449, struct memblock *mem_out_p_105450, struct memblock *mem_out_p_105451, struct memblock *mem_out_p_105452, struct memblock wdown_mem_102609, struct memblock wkey_mem_102610, struct memblock wout_mem_102611, struct memblock wpe_mem_102612, struct memblock wqry_mem_102613, struct memblock wte_mem_102614, struct memblock wup_mem_102615, struct memblock wval_mem_102616, struct memblock wvoc_mem_102617, struct memblock wdown_mem_102618, struct memblock wkey_mem_102619, struct memblock wout_mem_102620, struct memblock wpe_mem_102621, struct memblock wqry_mem_102622, struct memblock wte_mem_102623, struct memblock wup_mem_102624, struct memblock wval_mem_102625, struct memblock wvoc_mem_102626, struct memblock wdown_mem_102627, struct memblock wkey_mem_102628, struct memblock wout_mem_102629, struct memblock wpe_mem_102630, struct memblock wqry_mem_102631, struct memblock wte_mem_102632, struct memblock wup_mem_102633, struct memblock wval_mem_102634, struct memblock wvoc_mem_102635, struct memblock masks_mem_102636, struct memblock dls_mem_102637, struct memblock seqs_mem_102638, int64_t n_74237);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_105679, struct memblock *mem_out_p_105680, struct memblock *mem_out_p_105681, struct memblock *mem_out_p_105682, struct memblock *mem_out_p_105683, struct memblock *mem_out_p_105684, struct memblock *mem_out_p_105685, struct memblock *mem_out_p_105686, struct memblock *mem_out_p_105687);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_102600 (ctx->constants->mem_102600)
    #define mem_102601 (ctx->constants->mem_102601)
    #define mem_102602 (ctx->constants->mem_102602)
    #define mem_102603 (ctx->constants->mem_102603)
    #define mem_102604 (ctx->constants->mem_102604)
    #define mem_102605 (ctx->constants->mem_102605)
    #define mem_102606 (ctx->constants->mem_102606)
    #define mem_102607 (ctx->constants->mem_102607)
    #define mem_102608 (ctx->constants->mem_102608)
    mem_102600.references = NULL;
    mem_102601.references = NULL;
    mem_102602.references = NULL;
    mem_102603.references = NULL;
    mem_102604.references = NULL;
    mem_102605.references = NULL;
    mem_102606.references = NULL;
    mem_102607.references = NULL;
    mem_102608.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_102600, (int64_t) 3456, "mem_102600")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_105269 = 0; nest_i_105269 < (int64_t) 27; nest_i_105269++) {
        for (int64_t nest_i_105270 = 0; nest_i_105270 < (int64_t) 16; nest_i_105270++) {
            ((double *) mem_102600.mem)[nest_i_105269 * (int64_t) 16 + nest_i_105270] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_102601, (int64_t) 2048, "mem_102601")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_105271 = 0; nest_i_105271 < (int64_t) 16; nest_i_105271++) {
        for (int64_t nest_i_105272 = 0; nest_i_105272 < (int64_t) 16; nest_i_105272++) {
            ((double *) mem_102601.mem)[nest_i_105271 * (int64_t) 16 + nest_i_105272] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_102602, (int64_t) 2048, "mem_102602")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_105273 = 0; nest_i_105273 < (int64_t) 16; nest_i_105273++) {
        for (int64_t nest_i_105274 = 0; nest_i_105274 < (int64_t) 16; nest_i_105274++) {
            ((double *) mem_102602.mem)[nest_i_105273 * (int64_t) 16 + nest_i_105274] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_102603, (int64_t) 2048, "mem_102603")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_105275 = 0; nest_i_105275 < (int64_t) 16; nest_i_105275++) {
        for (int64_t nest_i_105276 = 0; nest_i_105276 < (int64_t) 16; nest_i_105276++) {
            ((double *) mem_102603.mem)[nest_i_105275 * (int64_t) 16 + nest_i_105276] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_102604, (int64_t) 2048, "mem_102604")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_105277 = 0; nest_i_105277 < (int64_t) 16; nest_i_105277++) {
        for (int64_t nest_i_105278 = 0; nest_i_105278 < (int64_t) 16; nest_i_105278++) {
            ((double *) mem_102604.mem)[nest_i_105277 * (int64_t) 16 + nest_i_105278] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_102605, (int64_t) 2048, "mem_102605")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_105279 = 0; nest_i_105279 < (int64_t) 16; nest_i_105279++) {
        for (int64_t nest_i_105280 = 0; nest_i_105280 < (int64_t) 16; nest_i_105280++) {
            ((double *) mem_102605.mem)[nest_i_105279 * (int64_t) 16 + nest_i_105280] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_102606, (int64_t) 8192, "mem_102606")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_105281 = 0; nest_i_105281 < (int64_t) 64; nest_i_105281++) {
        for (int64_t nest_i_105282 = 0; nest_i_105282 < (int64_t) 16; nest_i_105282++) {
            ((double *) mem_102606.mem)[nest_i_105281 * (int64_t) 16 + nest_i_105282] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_102607, (int64_t) 8192, "mem_102607")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_105283 = 0; nest_i_105283 < (int64_t) 16; nest_i_105283++) {
        for (int64_t nest_i_105284 = 0; nest_i_105284 < (int64_t) 64; nest_i_105284++) {
            ((double *) mem_102607.mem)[nest_i_105283 * (int64_t) 64 + nest_i_105284] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_102608, (int64_t) 3456, "mem_102608")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_105285 = 0; nest_i_105285 < (int64_t) 27; nest_i_105285++) {
        for (int64_t nest_i_105286 = 0; nest_i_105286 < (int64_t) 16; nest_i_105286++) {
            ((double *) mem_102608.mem)[nest_i_105285 * (int64_t) 16 + nest_i_105286] = 0.0;
        }
    }
    #undef mem_102600
    #undef mem_102601
    #undef mem_102602
    #undef mem_102603
    #undef mem_102604
    #undef mem_102605
    #undef mem_102606
    #undef mem_102607
    #undef mem_102608
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_102600, "ctx->constants->mem_102600") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_102601, "ctx->constants->mem_102601") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_102602, "ctx->constants->mem_102602") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_102603, "ctx->constants->mem_102603") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_102604, "ctx->constants->mem_102604") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_102605, "ctx->constants->mem_102605") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_102606, "ctx->constants->mem_102606") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_102607, "ctx->constants->mem_102607") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_102608, "ctx->constants->mem_102608") != 0)
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
struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 {
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
    struct futhark_f64_1d *v27;
};
int futhark_project_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64_0(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 *obj)
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
int futhark_project_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64_1(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 *obj)
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
int futhark_project_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64_2(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 *obj)
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
int futhark_project_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64_3(struct futhark_context *ctx, struct futhark_f64_1d **out, const struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 *obj)
{
    (void) ctx;
    
    struct futhark_f64_1d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f64_1d));
    memcpy(v, obj->v27, sizeof(struct futhark_f64_1d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_new_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64(struct futhark_context *ctx, struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 **out, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *f_0, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *f_1, const struct futhark_opaque_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *f_2, const struct futhark_f64_1d *f_3)
{
    struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 *v = malloc(sizeof(struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64));
    
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
    {
        v->v27 = malloc(sizeof(struct futhark_f64_1d));
        memcpy(v->v27, f_3, sizeof(struct futhark_f64_1d));
        (void) (*v->v27->mem.references)++;
    }
    lock_unlock(&ctx->lock);
    *out = v;
    return FUTHARK_SUCCESS;
}
int futhark_free_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64(struct futhark_context *ctx, struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 *obj)
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
    if (obj->v27 != NULL && (tmp = futhark_free_f64_1d(ctx, obj->v27)) != 0)
        ret = tmp;
    free(obj);
    return ret;
}
int futhark_store_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64(struct futhark_context *ctx, const struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 *obj, void **p, size_t *n)
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
    int64_t size_27 = 7 + 1 * sizeof(int64_t) + futhark_shape_f64_1d(ctx, obj->v27)[0] * sizeof(double);
    
    *n = size_0 + size_1 + size_2 + size_3 + size_4 + size_5 + size_6 + size_7 + size_8 + size_9 + size_10 + size_11 + size_12 + size_13 + size_14 + size_15 + size_16 + size_17 + size_18 + size_19 + size_20 + size_21 + size_22 + size_23 + size_24 + size_25 + size_26 + size_27;
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
        *out++ = 'b';
        *out++ = 2;
        *out++ = 1;
        memcpy(out, " f64", 4);
        out += 4;
        memcpy(out, futhark_shape_f64_1d(ctx, obj->v27), 1 * sizeof(int64_t));
        out += 1 * sizeof(int64_t);
        ret |= futhark_values_f64_1d(ctx, obj->v27, (void *) out);
        out += futhark_shape_f64_1d(ctx, obj->v27)[0] * sizeof(double);
    }
    return ret;
}
struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 *futhark_restore_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64(struct futhark_context *ctx, const void *p)
{
    (void) ctx;
    
    int err = 0;
    const unsigned char *src = p;
    struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 *obj = malloc(sizeof(struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64));
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
    
    int64_t shape_27[1] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 1;
    err |= memcmp(src, " f64", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_27, src, 1 * sizeof(int64_t));
        src += 1 * sizeof(int64_t);
    }
    
    const void *data_27 = src;
    
    obj->v27 = NULL;
    src += shape_27[0] * sizeof(double);
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
        obj->v27 = futhark_new_f64_1d(ctx, data_27, shape_27[0]);
        if (obj->v27 == NULL)
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
        if (obj->v27 != NULL && (tmp = futhark_free_f64_1d(ctx, obj->v27)) != 0)
            ret = tmp;
        free(obj);
        obj = NULL;
    }
    return obj;
}

FUTHARK_FUN_ATTR int futrts_adam_opt_w_11260(struct futhark_context *ctx, struct memblock *mem_out_p_105287, struct memblock *mem_out_p_105288, struct memblock *mem_out_p_105289, struct memblock w_mem_102609, struct memblock mw_mem_102610, struct memblock vw_mem_102611, struct memblock dw_mem_102612, int64_t n_72507, int64_t m_72508, int64_t step_72513, double lt_r_72514)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_102630_cached_sizze_105290 = 0;
    unsigned char *mem_102630 = NULL;
    int64_t mem_102653_cached_sizze_105291 = 0;
    unsigned char *mem_102653 = NULL;
    int64_t mem_102656_cached_sizze_105292 = 0;
    unsigned char *mem_102656 = NULL;
    struct memblock mem_102691;
    
    mem_102691.references = NULL;
    
    struct memblock mem_102618;
    
    mem_102618.references = NULL;
    
    struct memblock mem_102615;
    
    mem_102615.references = NULL;
    
    struct memblock mem_out_104891;
    
    mem_out_104891.references = NULL;
    
    struct memblock mem_out_104890;
    
    mem_out_104890.references = NULL;
    
    struct memblock mem_out_104889;
    
    mem_out_104889.references = NULL;
    
    struct memblock mem_102600 = ctx->constants->mem_102600;
    struct memblock mem_102601 = ctx->constants->mem_102601;
    struct memblock mem_102602 = ctx->constants->mem_102602;
    struct memblock mem_102603 = ctx->constants->mem_102603;
    struct memblock mem_102604 = ctx->constants->mem_102604;
    struct memblock mem_102605 = ctx->constants->mem_102605;
    struct memblock mem_102606 = ctx->constants->mem_102606;
    struct memblock mem_102607 = ctx->constants->mem_102607;
    struct memblock mem_102608 = ctx->constants->mem_102608;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_102613 = (int64_t) 8 * n_72507;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_102614 = m_72508 * binop_x_102613;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_102627 = (int64_t) 8 * m_72508;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_102615, bytes_102614, "mem_102615")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_102618, bytes_102614, "mem_102618")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102630_cached_sizze_105290 < bytes_102627) {
        err = lexical_realloc(ctx, &mem_102630, &mem_102630_cached_sizze_105290, bytes_102627);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101582 = 0; i_101582 < n_72507; i_101582++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101575 = 0; i_101575 < m_72508; i_101575++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_92717 = ((double *) mw_mem_102610.mem)[i_101582 * m_72508 + i_101575];
            
            // futhark/microgpt.fut:417:10-20
            
            double zp_lhs_92718 = 0.85 * zt_rhs_92717;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_92719 = ((double *) dw_mem_102612.mem)[i_101582 * m_72508 + i_101575];
            
            // futhark/microgpt.fut:417:35-45
            
            double zp_rhs_92720 = 0.15000000000000002 * zt_rhs_92719;
            
            // futhark/microgpt.fut:417:21-45
            
            double lifted_lambda_res_92721 = zp_lhs_92718 + zp_rhs_92720;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_92728 = ((double *) vw_mem_102611.mem)[i_101582 * m_72508 + i_101575];
            
            // futhark/microgpt.fut:419:10-20
            
            double zp_lhs_92729 = 0.99 * zt_rhs_92728;
            
            // futhark/microgpt.fut:419:35-45
            
            double zt_lhs_92731 = 1.0000000000000009e-2 * zt_rhs_92719;
            
            // futhark/microgpt.fut:419:46-56
            
            double zp_rhs_92732 = zt_rhs_92719 * zt_lhs_92731;
            
            // futhark/microgpt.fut:419:21-56
            
            double lifted_lambda_res_92733 = zp_lhs_92729 + zp_rhs_92732;
            
            ((double *) mem_102615.mem)[i_101582 * m_72508 + i_101575] = lifted_lambda_res_92733;
            ((double *) mem_102630)[i_101575] = lifted_lambda_res_92721;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102618.mem, i_101582 * m_72508, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102630, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {m_72508});
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_77580 = sitofp_i64_f64(step_72513);
    
    // futhark/microgpt.fut:421:54-57
    
    double ztzt_rhs_77581 = 1.0 + i64_res_77580;
    
    // futhark/microgpt.fut:421:30-57
    
    double zm_rhs_77582 = fpow64(0.85, ztzt_rhs_77581);
    
    // futhark/microgpt.fut:421:23-57
    
    double zs_rhs_77583 = 1.0 - zm_rhs_77582;
    
    // futhark/microgpt.fut:423:31-58
    
    double zm_rhs_77621 = fpow64(0.99, ztzt_rhs_77581);
    
    // futhark/microgpt.fut:423:23-58
    
    double zs_rhs_77622 = 1.0 - zm_rhs_77621;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_102653_cached_sizze_105291 < bytes_102614) {
        err = lexical_realloc(ctx, &mem_102653, &mem_102653_cached_sizze_105291, bytes_102614);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102656_cached_sizze_105292 < bytes_102614) {
        err = lexical_realloc(ctx, &mem_102656, &mem_102656_cached_sizze_105292, bytes_102614);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101596 = 0; i_101596 < n_72507; i_101596++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101589 = 0; i_101589 < m_72508; i_101589++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_92753 = ((double *) mem_102618.mem)[i_101596 * m_72508 + i_101589];
            
            // futhark/microgpt.fut:421:18-57
            
            double lifted_lambda_res_92754 = zs_lhs_92753 / zs_rhs_77583;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_92761 = ((double *) mem_102615.mem)[i_101596 * m_72508 + i_101589];
            
            // futhark/microgpt.fut:423:18-58
            
            double lifted_lambda_res_92762 = zs_lhs_92761 / zs_rhs_77622;
            
            ((double *) mem_102653)[i_101596 * m_72508 + i_101589] = lifted_lambda_res_92762;
            ((double *) mem_102656)[i_101596 * m_72508 + i_101589] = lifted_lambda_res_92754;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_102691, bytes_102614, "mem_102691")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101605 = 0; i_101605 < n_72507; i_101605++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101601 = 0; i_101601 < m_72508; i_101601++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_76744 = ((double *) w_mem_102609.mem)[i_101605 * m_72508 + i_101601];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_76745 = ((double *) mem_102656)[i_101605 * m_72508 + i_101601];
            
            // futhark/microgpt.fut:425:21-34
            
            double zs_lhs_76746 = lt_r_72514 * zt_rhs_76745;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_76747 = ((double *) mem_102653)[i_101605 * m_72508 + i_101601];
            
            // futhark/microgpt.fut:425:51-57
            
            double zp_lhs_76748 = fpow64(ztzt_lhs_76747, 0.5);
            
            // futhark/microgpt.fut:425:59-71
            
            double zs_rhs_76749 = 1.0e-8 + zp_lhs_76748;
            
            // futhark/microgpt.fut:425:35-71
            
            double zm_rhs_76750 = zs_lhs_76746 / zs_rhs_76749;
            
            // futhark/microgpt.fut:425:13-71
            
            double lifted_lambda_res_76751 = zm_lhs_76744 - zm_rhs_76750;
            
            ((double *) mem_102691.mem)[i_101605 * m_72508 + i_101601] = lifted_lambda_res_76751;
        }
    }
    if (memblock_set(ctx, &mem_out_104889, &mem_102691, "mem_102691") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104890, &mem_102618, "mem_102618") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104891, &mem_102615, "mem_102615") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105287, &mem_out_104889, "mem_out_104889") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105288, &mem_out_104890, "mem_out_104890") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105289, &mem_out_104891, "mem_out_104891") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_102630);
        free(mem_102653);
        free(mem_102656);
        if (memblock_unref(ctx, &mem_102691, "mem_102691") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_102618, "mem_102618") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_102615, "mem_102615") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104891, "mem_out_104891") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104890, "mem_out_104890") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104889, "mem_out_104889") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_11261(struct futhark_context *ctx, struct memblock *mem_out_p_105293, struct memblock *mem_out_p_105294, struct memblock *mem_out_p_105295, struct memblock w_mem_102609, struct memblock mw_mem_102610, struct memblock vw_mem_102611, struct memblock dw_mem_102612, int64_t n_73540, int64_t m_73541, int64_t step_73546, double lt_r_73547)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_102630_cached_sizze_105296 = 0;
    unsigned char *mem_102630 = NULL;
    int64_t mem_102653_cached_sizze_105297 = 0;
    unsigned char *mem_102653 = NULL;
    int64_t mem_102656_cached_sizze_105298 = 0;
    unsigned char *mem_102656 = NULL;
    struct memblock mem_102691;
    
    mem_102691.references = NULL;
    
    struct memblock mem_102618;
    
    mem_102618.references = NULL;
    
    struct memblock mem_102615;
    
    mem_102615.references = NULL;
    
    struct memblock mem_out_104891;
    
    mem_out_104891.references = NULL;
    
    struct memblock mem_out_104890;
    
    mem_out_104890.references = NULL;
    
    struct memblock mem_out_104889;
    
    mem_out_104889.references = NULL;
    
    struct memblock mem_102600 = ctx->constants->mem_102600;
    struct memblock mem_102601 = ctx->constants->mem_102601;
    struct memblock mem_102602 = ctx->constants->mem_102602;
    struct memblock mem_102603 = ctx->constants->mem_102603;
    struct memblock mem_102604 = ctx->constants->mem_102604;
    struct memblock mem_102605 = ctx->constants->mem_102605;
    struct memblock mem_102606 = ctx->constants->mem_102606;
    struct memblock mem_102607 = ctx->constants->mem_102607;
    struct memblock mem_102608 = ctx->constants->mem_102608;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_102613 = (int64_t) 8 * n_73540;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_102614 = m_73541 * binop_x_102613;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_102627 = (int64_t) 8 * m_73541;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_102615, bytes_102614, "mem_102615")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_102618, bytes_102614, "mem_102618")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102630_cached_sizze_105296 < bytes_102627) {
        err = lexical_realloc(ctx, &mem_102630, &mem_102630_cached_sizze_105296, bytes_102627);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101582 = 0; i_101582 < n_73540; i_101582++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101575 = 0; i_101575 < m_73541; i_101575++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_92717 = ((double *) mw_mem_102610.mem)[i_101582 * m_73541 + i_101575];
            
            // futhark/microgpt.fut:417:10-20
            
            double zp_lhs_92718 = 0.85 * zt_rhs_92717;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_92719 = ((double *) dw_mem_102612.mem)[i_101582 * m_73541 + i_101575];
            
            // futhark/microgpt.fut:417:35-45
            
            double zp_rhs_92720 = 0.15000000000000002 * zt_rhs_92719;
            
            // futhark/microgpt.fut:417:21-45
            
            double lifted_lambda_res_92721 = zp_lhs_92718 + zp_rhs_92720;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_92728 = ((double *) vw_mem_102611.mem)[i_101582 * m_73541 + i_101575];
            
            // futhark/microgpt.fut:419:10-20
            
            double zp_lhs_92729 = 0.99 * zt_rhs_92728;
            
            // futhark/microgpt.fut:419:35-45
            
            double zt_lhs_92731 = 1.0000000000000009e-2 * zt_rhs_92719;
            
            // futhark/microgpt.fut:419:46-56
            
            double zp_rhs_92732 = zt_rhs_92719 * zt_lhs_92731;
            
            // futhark/microgpt.fut:419:21-56
            
            double lifted_lambda_res_92733 = zp_lhs_92729 + zp_rhs_92732;
            
            ((double *) mem_102615.mem)[i_101582 * m_73541 + i_101575] = lifted_lambda_res_92733;
            ((double *) mem_102630)[i_101575] = lifted_lambda_res_92721;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102618.mem, i_101582 * m_73541, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102630, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {m_73541});
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_77580 = sitofp_i64_f64(step_73546);
    
    // futhark/microgpt.fut:421:54-57
    
    double ztzt_rhs_77581 = 1.0 + i64_res_77580;
    
    // futhark/microgpt.fut:421:30-57
    
    double zm_rhs_77582 = fpow64(0.85, ztzt_rhs_77581);
    
    // futhark/microgpt.fut:421:23-57
    
    double zs_rhs_77583 = 1.0 - zm_rhs_77582;
    
    // futhark/microgpt.fut:423:31-58
    
    double zm_rhs_77621 = fpow64(0.99, ztzt_rhs_77581);
    
    // futhark/microgpt.fut:423:23-58
    
    double zs_rhs_77622 = 1.0 - zm_rhs_77621;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_102653_cached_sizze_105297 < bytes_102614) {
        err = lexical_realloc(ctx, &mem_102653, &mem_102653_cached_sizze_105297, bytes_102614);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102656_cached_sizze_105298 < bytes_102614) {
        err = lexical_realloc(ctx, &mem_102656, &mem_102656_cached_sizze_105298, bytes_102614);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101596 = 0; i_101596 < n_73540; i_101596++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101589 = 0; i_101589 < m_73541; i_101589++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_92753 = ((double *) mem_102618.mem)[i_101596 * m_73541 + i_101589];
            
            // futhark/microgpt.fut:421:18-57
            
            double lifted_lambda_res_92754 = zs_lhs_92753 / zs_rhs_77583;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_92761 = ((double *) mem_102615.mem)[i_101596 * m_73541 + i_101589];
            
            // futhark/microgpt.fut:423:18-58
            
            double lifted_lambda_res_92762 = zs_lhs_92761 / zs_rhs_77622;
            
            ((double *) mem_102653)[i_101596 * m_73541 + i_101589] = lifted_lambda_res_92762;
            ((double *) mem_102656)[i_101596 * m_73541 + i_101589] = lifted_lambda_res_92754;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_102691, bytes_102614, "mem_102691")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101605 = 0; i_101605 < n_73540; i_101605++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101601 = 0; i_101601 < m_73541; i_101601++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_76744 = ((double *) w_mem_102609.mem)[i_101605 * m_73541 + i_101601];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_76745 = ((double *) mem_102656)[i_101605 * m_73541 + i_101601];
            
            // futhark/microgpt.fut:425:21-34
            
            double zs_lhs_76746 = lt_r_73547 * zt_rhs_76745;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_76747 = ((double *) mem_102653)[i_101605 * m_73541 + i_101601];
            
            // futhark/microgpt.fut:425:51-57
            
            double zp_lhs_76748 = fpow64(ztzt_lhs_76747, 0.5);
            
            // futhark/microgpt.fut:425:59-71
            
            double zs_rhs_76749 = 1.0e-8 + zp_lhs_76748;
            
            // futhark/microgpt.fut:425:35-71
            
            double zm_rhs_76750 = zs_lhs_76746 / zs_rhs_76749;
            
            // futhark/microgpt.fut:425:13-71
            
            double lifted_lambda_res_76751 = zm_lhs_76744 - zm_rhs_76750;
            
            ((double *) mem_102691.mem)[i_101605 * m_73541 + i_101601] = lifted_lambda_res_76751;
        }
    }
    if (memblock_set(ctx, &mem_out_104889, &mem_102691, "mem_102691") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104890, &mem_102618, "mem_102618") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104891, &mem_102615, "mem_102615") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105293, &mem_out_104889, "mem_out_104889") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105294, &mem_out_104890, "mem_out_104890") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105295, &mem_out_104891, "mem_out_104891") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_102630);
        free(mem_102653);
        free(mem_102656);
        if (memblock_unref(ctx, &mem_102691, "mem_102691") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_102618, "mem_102618") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_102615, "mem_102615") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104891, "mem_out_104891") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104890, "mem_out_104890") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104889, "mem_out_104889") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_cal_target_8179(struct futhark_context *ctx, struct memblock *mem_out_p_105299, struct memblock tokens_mem_102609, int64_t n_45888)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_102615_cached_sizze_105300 = 0;
    unsigned char *mem_102615 = NULL;
    struct memblock mem_102610;
    
    mem_102610.references = NULL;
    
    struct memblock mem_out_104889;
    
    mem_out_104889.references = NULL;
    
    struct memblock mem_102600 = ctx->constants->mem_102600;
    struct memblock mem_102601 = ctx->constants->mem_102601;
    struct memblock mem_102602 = ctx->constants->mem_102602;
    struct memblock mem_102603 = ctx->constants->mem_102603;
    struct memblock mem_102604 = ctx->constants->mem_102604;
    struct memblock mem_102605 = ctx->constants->mem_102605;
    struct memblock mem_102606 = ctx->constants->mem_102606;
    struct memblock mem_102607 = ctx->constants->mem_102607;
    struct memblock mem_102608 = ctx->constants->mem_102608;
    
    // futhark/microgpt.fut:405:37-40
    
    int64_t zl_rhs_76635 = sub64(n_45888, (int64_t) 1);
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_102610, (int64_t) 3456, "mem_102610")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102615_cached_sizze_105300 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_102615, &mem_102615_cached_sizze_105300, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101577 = 0; i_101577 < (int64_t) 16; i_101577++) {
        // futhark/microgpt.fut:405:25-81
        
        bool cond_76638 = slt64(i_101577, zl_rhs_76635);
        
        // futhark/microgpt.fut:405:56-59
        
        int64_t zeze_lhs_76639 = add64((int64_t) 1, i_101577);
        
        // futhark/microgpt.fut:405:47-60
        
        bool x_76640 = sle64((int64_t) 0, zeze_lhs_76639);
        
        // futhark/microgpt.fut:405:47-60
        
        bool y_76641 = slt64(zeze_lhs_76639, (int64_t) 16);
        
        // futhark/microgpt.fut:405:47-60
        
        bool bounds_check_76642 = x_76640 && y_76641;
        
        // futhark/microgpt.fut:9:27-39
        
        bool loop_not_taken_76643 = !cond_76638;
        
        // futhark/microgpt.fut:9:27-39
        
        bool protect_assert_disj_76644 = bounds_check_76642 || loop_not_taken_76643;
        
        // futhark/microgpt.fut:405:47-60
        
        bool index_certs_76645;
        
        if (!protect_assert_disj_76644) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_76639, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:405:47-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:405:3-83\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:405:47-60
        
        int64_t zeze_lhs_76646;
        
        if (cond_76638) {
            // futhark/microgpt.fut:9:27-39
            
            int64_t x_92613 = ((int64_t *) tokens_mem_102609.mem)[zeze_lhs_76639];
            
            zeze_lhs_76646 = x_92613;
        } else {
            zeze_lhs_76646 = (int64_t) 0;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101573 = 0; i_101573 < (int64_t) 27; i_101573++) {
            // futhark/microgpt.fut:405:61-65
            
            bool cond_t_res_76650 = zeze_lhs_76646 == i_101573;
            
            // futhark/microgpt.fut:9:27-39
            
            bool x_76651 = cond_76638 && cond_t_res_76650;
            
            // futhark/microgpt.fut:405:25-81
            
            double lifted_lambda_res_76652;
            
            if (x_76651) {
                lifted_lambda_res_76652 = 1.0;
            } else {
                lifted_lambda_res_76652 = 0.0;
            }
            ((double *) mem_102615)[i_101573] = lifted_lambda_res_76652;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102610.mem, i_101577 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102615, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_104889, &mem_102610, "mem_102610") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105299, &mem_out_104889, "mem_out_104889") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_102615);
        if (memblock_unref(ctx, &mem_102610, "mem_102610") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104889, "mem_out_104889") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, double *out_prim_out_105301, struct memblock wdown_mem_102609, struct memblock wkey_mem_102610, struct memblock wout_mem_102611, struct memblock wpe_mem_102612, struct memblock wqry_mem_102613, struct memblock wte_mem_102614, struct memblock wup_mem_102615, struct memblock wval_mem_102616, struct memblock wvoc_mem_102617, struct memblock tokens_mem_102618, struct memblock mask_mem_102619, int64_t dl_52055)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_102621_cached_sizze_105302 = 0;
    unsigned char *mem_102621 = NULL;
    int64_t mem_102626_cached_sizze_105303 = 0;
    unsigned char *mem_102626 = NULL;
    int64_t mem_102637_cached_sizze_105304 = 0;
    unsigned char *mem_102637 = NULL;
    int64_t mem_102642_cached_sizze_105305 = 0;
    unsigned char *mem_102642 = NULL;
    int64_t mem_102653_cached_sizze_105306 = 0;
    unsigned char *mem_102653 = NULL;
    int64_t mem_102658_cached_sizze_105307 = 0;
    unsigned char *mem_102658 = NULL;
    int64_t mem_102665_cached_sizze_105308 = 0;
    unsigned char *mem_102665 = NULL;
    int64_t mem_102676_cached_sizze_105309 = 0;
    unsigned char *mem_102676 = NULL;
    int64_t mem_102681_cached_sizze_105310 = 0;
    unsigned char *mem_102681 = NULL;
    int64_t mem_102688_cached_sizze_105311 = 0;
    unsigned char *mem_102688 = NULL;
    int64_t mem_102699_cached_sizze_105312 = 0;
    unsigned char *mem_102699 = NULL;
    int64_t mem_102700_cached_sizze_105313 = 0;
    unsigned char *mem_102700 = NULL;
    int64_t mem_102701_cached_sizze_105314 = 0;
    unsigned char *mem_102701 = NULL;
    int64_t mem_102714_cached_sizze_105315 = 0;
    unsigned char *mem_102714 = NULL;
    int64_t mem_102715_cached_sizze_105316 = 0;
    unsigned char *mem_102715 = NULL;
    int64_t mem_102716_cached_sizze_105317 = 0;
    unsigned char *mem_102716 = NULL;
    int64_t mem_102747_cached_sizze_105318 = 0;
    unsigned char *mem_102747 = NULL;
    int64_t mem_102748_cached_sizze_105319 = 0;
    unsigned char *mem_102748 = NULL;
    int64_t mem_102749_cached_sizze_105320 = 0;
    unsigned char *mem_102749 = NULL;
    int64_t mem_102765_cached_sizze_105321 = 0;
    unsigned char *mem_102765 = NULL;
    int64_t mem_102766_cached_sizze_105322 = 0;
    unsigned char *mem_102766 = NULL;
    int64_t mem_102767_cached_sizze_105323 = 0;
    unsigned char *mem_102767 = NULL;
    int64_t mem_102780_cached_sizze_105324 = 0;
    unsigned char *mem_102780 = NULL;
    int64_t mem_102781_cached_sizze_105325 = 0;
    unsigned char *mem_102781 = NULL;
    int64_t mem_102782_cached_sizze_105326 = 0;
    unsigned char *mem_102782 = NULL;
    int64_t mem_102828_cached_sizze_105327 = 0;
    unsigned char *mem_102828 = NULL;
    int64_t mem_102834_cached_sizze_105328 = 0;
    unsigned char *mem_102834 = NULL;
    int64_t mem_102839_cached_sizze_105329 = 0;
    unsigned char *mem_102839 = NULL;
    int64_t mem_102850_cached_sizze_105330 = 0;
    unsigned char *mem_102850 = NULL;
    int64_t mem_102855_cached_sizze_105331 = 0;
    unsigned char *mem_102855 = NULL;
    int64_t mem_102866_cached_sizze_105332 = 0;
    unsigned char *mem_102866 = NULL;
    int64_t mem_102871_cached_sizze_105333 = 0;
    unsigned char *mem_102871 = NULL;
    int64_t mem_102878_cached_sizze_105334 = 0;
    unsigned char *mem_102878 = NULL;
    int64_t mem_102885_cached_sizze_105335 = 0;
    unsigned char *mem_102885 = NULL;
    int64_t mem_102896_cached_sizze_105336 = 0;
    unsigned char *mem_102896 = NULL;
    int64_t mem_102901_cached_sizze_105337 = 0;
    unsigned char *mem_102901 = NULL;
    int64_t mem_102917_cached_sizze_105338 = 0;
    unsigned char *mem_102917 = NULL;
    int64_t mem_102922_cached_sizze_105339 = 0;
    unsigned char *mem_102922 = NULL;
    int64_t mem_102933_cached_sizze_105340 = 0;
    unsigned char *mem_102933 = NULL;
    int64_t mem_102938_cached_sizze_105341 = 0;
    unsigned char *mem_102938 = NULL;
    int64_t mem_102949_cached_sizze_105342 = 0;
    unsigned char *mem_102949 = NULL;
    int64_t mem_102954_cached_sizze_105343 = 0;
    unsigned char *mem_102954 = NULL;
    int64_t mem_102965_cached_sizze_105344 = 0;
    unsigned char *mem_102965 = NULL;
    int64_t mem_102970_cached_sizze_105345 = 0;
    unsigned char *mem_102970 = NULL;
    int64_t mem_102977_cached_sizze_105346 = 0;
    unsigned char *mem_102977 = NULL;
    int64_t mem_102988_cached_sizze_105347 = 0;
    unsigned char *mem_102988 = NULL;
    int64_t mem_102993_cached_sizze_105348 = 0;
    unsigned char *mem_102993 = NULL;
    int64_t mem_103004_cached_sizze_105349 = 0;
    unsigned char *mem_103004 = NULL;
    int64_t mem_103009_cached_sizze_105350 = 0;
    unsigned char *mem_103009 = NULL;
    int64_t mem_103020_cached_sizze_105351 = 0;
    unsigned char *mem_103020 = NULL;
    int64_t mem_103025_cached_sizze_105352 = 0;
    unsigned char *mem_103025 = NULL;
    int64_t mem_103036_cached_sizze_105353 = 0;
    unsigned char *mem_103036 = NULL;
    int64_t mem_103041_cached_sizze_105354 = 0;
    unsigned char *mem_103041 = NULL;
    int64_t mem_103052_cached_sizze_105355 = 0;
    unsigned char *mem_103052 = NULL;
    int64_t mem_103057_cached_sizze_105356 = 0;
    unsigned char *mem_103057 = NULL;
    int64_t mem_103068_cached_sizze_105357 = 0;
    unsigned char *mem_103068 = NULL;
    int64_t mem_103072_cached_sizze_105358 = 0;
    unsigned char *mem_103072 = NULL;
    int64_t mem_103079_cached_sizze_105359 = 0;
    unsigned char *mem_103079 = NULL;
    int64_t mem_103086_cached_sizze_105360 = 0;
    unsigned char *mem_103086 = NULL;
    struct memblock ext_mem_102620;
    
    ext_mem_102620.references = NULL;
    
    struct memblock mem_102600 = ctx->constants->mem_102600;
    struct memblock mem_102601 = ctx->constants->mem_102601;
    struct memblock mem_102602 = ctx->constants->mem_102602;
    struct memblock mem_102603 = ctx->constants->mem_102603;
    struct memblock mem_102604 = ctx->constants->mem_102604;
    struct memblock mem_102605 = ctx->constants->mem_102605;
    struct memblock mem_102606 = ctx->constants->mem_102606;
    struct memblock mem_102607 = ctx->constants->mem_102607;
    struct memblock mem_102608 = ctx->constants->mem_102608;
    double prim_out_104889;
    
    // futhark/microgpt.fut:409:17-37
    if (futrts_cal_target_8179(ctx, &ext_mem_102620, tokens_mem_102618, dl_52055) != 0) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102621_cached_sizze_105302 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102621, &mem_102621_cached_sizze_105302, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102626_cached_sizze_105303 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102626, &mem_102626_cached_sizze_105303, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101577 = 0; i_101577 < (int64_t) 16; i_101577++) {
        // futhark/microgpt.fut:410:41-50
        
        int64_t tmp_85252 = ((int64_t *) tokens_mem_102618.mem)[i_101577];
        
        // futhark/microgpt.fut:410:37-51
        
        bool x_85253 = sle64((int64_t) 0, tmp_85252);
        
        // futhark/microgpt.fut:410:37-51
        
        bool y_85254 = slt64(tmp_85252, (int64_t) 27);
        
        // futhark/microgpt.fut:410:37-51
        
        bool bounds_check_85255 = x_85253 && y_85254;
        
        // futhark/microgpt.fut:410:37-51
        
        bool index_certs_85256;
        
        if (!bounds_check_85255) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_85252, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:410:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:410:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101573 = 0; i_101573 < (int64_t) 16; i_101573++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_85263 = ((double *) wte_mem_102614.mem)[tmp_85252 * (int64_t) 16 + i_101573];
            
            ((double *) mem_102626)[i_101573] = lifted_lambda_res_85263;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102621, i_101577 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102626, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102637_cached_sizze_105304 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102637, &mem_102637_cached_sizze_105304, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102642_cached_sizze_105305 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102642, &mem_102642_cached_sizze_105305, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101585 = 0; i_101585 < (int64_t) 16; i_101585++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101581 = 0; i_101581 < (int64_t) 16; i_101581++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_85295 = ((double *) wpe_mem_102612.mem)[i_101585 * (int64_t) 16 + i_101581];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_85296 = ((double *) mem_102621)[i_101585 * (int64_t) 16 + i_101581];
            
            // futhark/microgpt.fut:200:40-72
            
            double zp_res_85297 = zp_lhs_85295 + zp_rhs_85296;
            
            ((double *) mem_102642)[i_101581] = zp_res_85297;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102637, i_101585 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102642, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102653_cached_sizze_105306 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102653, &mem_102653_cached_sizze_105306, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102658_cached_sizze_105307 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102658, &mem_102658_cached_sizze_105307, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102665_cached_sizze_105308 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102665, &mem_102665_cached_sizze_105308, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101597 = 0; i_101597 < (int64_t) 16; i_101597++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101589 = 0; i_101589 < (int64_t) 16; i_101589++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_85312 = ((double *) mem_102637)[i_101597 * (int64_t) 16 + i_101589];
            
            // futhark/microgpt.fut:201:64-93
            
            double zt_res_85313 = zt_lhs_85312 * zt_lhs_85312;
            
            ((double *) mem_102658)[i_101589] = zt_res_85313;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_85315;
        double r_85317 = 0.0;
        
        for (int64_t i_85316 = 0; i_85316 < (int64_t) 16; i_85316++) {
            // futhark/microgpt.fut:202:35-43
            
            double lifted_lambda_res_85318 = ((double *) mem_102658)[i_85316];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_85319 = r_85317 + lifted_lambda_res_85318;
            double r_tmp_104896 = zp_res_85319;
            
            r_85317 = r_tmp_104896;
        }
        defunc_0_lifted_lambda_res_85315 = r_85317;
        // futhark/microgpt.fut:202:17-60
        
        double zs_res_85320 = defunc_0_lifted_lambda_res_85315 / 16.0;
        
        // futhark/microgpt.fut:203:24-55
        
        double zp_res_85321 = 1.0e-5 + zs_res_85320;
        
        // futhark/microgpt.fut:203:16-55
        
        double sqrt_res_85322 = futrts_sqrt64(zp_res_85321);
        
        // futhark/microgpt.fut:204:42-53
        
        double zs_res_85323 = 1.0 / sqrt_res_85322;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101593 = 0; i_101593 < (int64_t) 16; i_101593++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_85330 = ((double *) mem_102637)[i_101597 * (int64_t) 16 + i_101593];
            
            // futhark/microgpt.fut:204:24-53
            
            double zt_res_85331 = zs_res_85323 * zt_lhs_85330;
            
            ((double *) mem_102665)[i_101593] = zt_res_85331;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102653, i_101597 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102665, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102676_cached_sizze_105309 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102676, &mem_102676_cached_sizze_105309, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102681_cached_sizze_105310 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102681, &mem_102681_cached_sizze_105310, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102688_cached_sizze_105311 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102688, &mem_102688_cached_sizze_105311, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101609 = 0; i_101609 < (int64_t) 16; i_101609++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101601 = 0; i_101601 < (int64_t) 16; i_101601++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_85346 = ((double *) mem_102653)[i_101609 * (int64_t) 16 + i_101601];
            
            // futhark/microgpt.fut:205:64-93
            
            double zt_res_85347 = zt_lhs_85346 * zt_lhs_85346;
            
            ((double *) mem_102681)[i_101601] = zt_res_85347;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_85349;
        double r_85351 = 0.0;
        
        for (int64_t i_85350 = 0; i_85350 < (int64_t) 16; i_85350++) {
            // futhark/microgpt.fut:206:35-43
            
            double lifted_lambda_res_85352 = ((double *) mem_102681)[i_85350];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_85353 = r_85351 + lifted_lambda_res_85352;
            double r_tmp_104900 = zp_res_85353;
            
            r_85351 = r_tmp_104900;
        }
        defunc_0_lifted_lambda_res_85349 = r_85351;
        // futhark/microgpt.fut:206:17-60
        
        double zs_res_85354 = defunc_0_lifted_lambda_res_85349 / 16.0;
        
        // futhark/microgpt.fut:207:24-55
        
        double zp_res_85355 = 1.0e-5 + zs_res_85354;
        
        // futhark/microgpt.fut:207:16-55
        
        double sqrt_res_85356 = futrts_sqrt64(zp_res_85355);
        
        // futhark/microgpt.fut:208:42-53
        
        double zs_res_85357 = 1.0 / sqrt_res_85356;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101605 = 0; i_101605 < (int64_t) 16; i_101605++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_85364 = ((double *) mem_102653)[i_101609 * (int64_t) 16 + i_101605];
            
            // futhark/microgpt.fut:208:24-53
            
            double zt_res_85365 = zs_res_85357 * zt_lhs_85364;
            
            ((double *) mem_102688)[i_101605] = zt_res_85365;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102676, i_101609 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102688, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102699_cached_sizze_105312 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102699, &mem_102699_cached_sizze_105312, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102700_cached_sizze_105313 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102700, &mem_102700_cached_sizze_105313, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102701_cached_sizze_105314 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102701, &mem_102701_cached_sizze_105314, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102714_cached_sizze_105315 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102714, &mem_102714_cached_sizze_105315, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102715_cached_sizze_105316 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102715, &mem_102715_cached_sizze_105316, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102716_cached_sizze_105317 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102716, &mem_102716_cached_sizze_105317, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101627 = 0; i_101627 < (int64_t) 16; i_101627++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101617 = 0; i_101617 < (int64_t) 16; i_101617++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_92936;
            double r_92938 = 0.0;
            
            for (int64_t i_92937 = 0; i_92937 < (int64_t) 16; i_92937++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_92939 = ((double *) wqry_mem_102613.mem)[i_101617 * (int64_t) 16 + i_92937];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_92940 = ((double *) mem_102676)[i_101627 * (int64_t) 16 + i_92937];
                
                // futhark/microgpt.fut:209:72-103
                
                double zt_res_92941 = zt_lhs_92939 * zt_rhs_92940;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_92942 = r_92938 + zt_res_92941;
                double r_tmp_104908 = zp_res_92942;
                
                r_92938 = r_tmp_104908;
            }
            defunc_0_lifted_lambda_res_92936 = r_92938;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_92949;
            double r_92951 = 0.0;
            
            for (int64_t i_92950 = 0; i_92950 < (int64_t) 16; i_92950++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_92952 = ((double *) wkey_mem_102610.mem)[i_101617 * (int64_t) 16 + i_92950];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_92953 = ((double *) mem_102676)[i_101627 * (int64_t) 16 + i_92950];
                
                // futhark/microgpt.fut:210:72-103
                
                double zt_res_92954 = zt_lhs_92952 * zt_rhs_92953;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_92955 = r_92951 + zt_res_92954;
                double r_tmp_104909 = zp_res_92955;
                
                r_92951 = r_tmp_104909;
            }
            defunc_0_lifted_lambda_res_92949 = r_92951;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_92965;
            double r_92967 = 0.0;
            
            for (int64_t i_92966 = 0; i_92966 < (int64_t) 16; i_92966++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_92968 = ((double *) wval_mem_102616.mem)[i_101617 * (int64_t) 16 + i_92966];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_92969 = ((double *) mem_102676)[i_101627 * (int64_t) 16 + i_92966];
                
                // futhark/microgpt.fut:211:72-103
                
                double zt_res_92970 = zt_lhs_92968 * zt_rhs_92969;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_92971 = r_92967 + zt_res_92970;
                double r_tmp_104910 = zp_res_92971;
                
                r_92967 = r_tmp_104910;
            }
            defunc_0_lifted_lambda_res_92965 = r_92967;
            ((double *) mem_102714)[i_101617] = defunc_0_lifted_lambda_res_92965;
            ((double *) mem_102715)[i_101617] = defunc_0_lifted_lambda_res_92949;
            ((double *) mem_102716)[i_101617] = defunc_0_lifted_lambda_res_92936;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102699, i_101627 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102714, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102700, i_101627 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102715, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102701, i_101627 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102716, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102747_cached_sizze_105318 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102747, &mem_102747_cached_sizze_105318, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102748_cached_sizze_105319 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102748, &mem_102748_cached_sizze_105319, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102749_cached_sizze_105320 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102749, &mem_102749_cached_sizze_105320, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102765_cached_sizze_105321 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_102765, &mem_102765_cached_sizze_105321, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102766_cached_sizze_105322 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_102766, &mem_102766_cached_sizze_105322, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102767_cached_sizze_105323 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_102767, &mem_102767_cached_sizze_105323, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102780_cached_sizze_105324 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_102780, &mem_102780_cached_sizze_105324, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102781_cached_sizze_105325 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_102781, &mem_102781_cached_sizze_105325, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102782_cached_sizze_105326 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_102782, &mem_102782_cached_sizze_105326, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101657 = 0; i_101657 < (int64_t) 4; i_101657++) {
        // futhark/microgpt.fut:212:83-86
        
        int64_t zp_lhs_92811 = mul64((int64_t) 4, i_101657);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101647 = 0; i_101647 < (int64_t) 16; i_101647++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101637 = 0; i_101637 < (int64_t) 4; i_101637++) {
                // futhark/microgpt.fut:212:88-93
                
                int64_t tmp_93129 = add64(zp_lhs_92811, i_101637);
                
                // futhark/microgpt.fut:212:69-95
                
                bool x_93130 = sle64((int64_t) 0, tmp_93129);
                
                // futhark/microgpt.fut:212:69-95
                
                bool y_93131 = slt64(tmp_93129, (int64_t) 16);
                
                // futhark/microgpt.fut:212:69-95
                
                bool bounds_check_93132 = x_93130 && y_93131;
                
                // futhark/microgpt.fut:212:69-95
                
                bool index_certs_93133;
                
                if (!bounds_check_93132) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_93129, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:212:69-95\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:212:52-96\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:212:33-98\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:6:13-17\n   #9  futhark/microgpt.fut:212:15-100\n   #10 futhark/microgpt.fut:411:7-76\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_93134 = ((double *) mem_102701)[i_101647 * (int64_t) 16 + tmp_93129];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_93142 = ((double *) mem_102700)[i_101647 * (int64_t) 16 + tmp_93129];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_93153 = ((double *) mem_102699)[i_101647 * (int64_t) 16 + tmp_93129];
                
                ((double *) mem_102780)[i_101637] = lifted_lambda_res_93153;
                ((double *) mem_102781)[i_101637] = lifted_lambda_res_93142;
                ((double *) mem_102782)[i_101637] = lifted_lambda_res_93134;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102765, i_101647 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102780, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102766, i_101647 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102781, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102767, i_101647 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102782, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_102747, i_101657 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_102765, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_102748, i_101657 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_102766, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_102749, i_101657 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_102767, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102828_cached_sizze_105327 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102828, &mem_102828_cached_sizze_105327, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102834_cached_sizze_105328 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102834, &mem_102834_cached_sizze_105328, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102839_cached_sizze_105329 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102839, &mem_102839_cached_sizze_105329, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102850_cached_sizze_105330 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102850, &mem_102850_cached_sizze_105330, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102855_cached_sizze_105331 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102855, &mem_102855_cached_sizze_105331, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102866_cached_sizze_105332 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102866, &mem_102866_cached_sizze_105332, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102871_cached_sizze_105333 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102871, &mem_102871_cached_sizze_105333, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102878_cached_sizze_105334 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102878, &mem_102878_cached_sizze_105334, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102885_cached_sizze_105335 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102885, &mem_102885_cached_sizze_105335, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102896_cached_sizze_105336 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_102896, &mem_102896_cached_sizze_105336, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102901_cached_sizze_105337 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_102901, &mem_102901_cached_sizze_105337, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101705 = 0; i_101705 < (int64_t) 4; i_101705++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101667 = 0; i_101667 < (int64_t) 16; i_101667++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101663 = 0; i_101663 < (int64_t) 16; i_101663++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85510;
                double r_85512 = 0.0;
                
                for (int64_t i_85511 = 0; i_85511 < (int64_t) 4; i_85511++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_85513 = ((double *) mem_102749)[i_101705 * (int64_t) 64 + i_101667 * (int64_t) 4 + i_85511];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_85514 = ((double *) mem_102748)[i_101705 * (int64_t) 64 + i_101663 * (int64_t) 4 + i_85511];
                    
                    // futhark/microgpt.fut:215:100-139
                    
                    double zt_res_85515 = zt_lhs_85513 * zt_rhs_85514;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85516 = r_85512 + zt_res_85515;
                    double r_tmp_104923 = zp_res_85516;
                    
                    r_85512 = r_tmp_104923;
                }
                defunc_0_lifted_lambda_res_85510 = r_85512;
                ((double *) mem_102839)[i_101663] = defunc_0_lifted_lambda_res_85510;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102834, i_101667 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102839, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101675 = 0; i_101675 < (int64_t) 16; i_101675++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101671 = 0; i_101671 < (int64_t) 16; i_101671++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_85531 = ((double *) mem_102834)[i_101675 * (int64_t) 16 + i_101671];
                
                // futhark/microgpt.fut:216:43-70
                
                double zs_res_85532 = zs_lhs_85531 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_85533 = ((double *) mask_mem_102619.mem)[i_101675 * (int64_t) 16 + i_101671];
                
                // futhark/microgpt.fut:216:57-90
                
                double zp_res_85534 = zs_res_85532 + zp_rhs_85533;
                
                ((double *) mem_102855)[i_101671] = zp_res_85534;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102850, i_101675 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102855, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101693 = 0; i_101693 < (int64_t) 16; i_101693++) {
            // futhark/microgpt.fut:103:13-33
            
            double defunc_0_reduce_res_93249;
            double redout_101677 = -INFINITY;
            
            for (int64_t i_101678 = 0; i_101678 < (int64_t) 16; i_101678++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_93180 = ((double *) mem_102850)[i_101693 * (int64_t) 16 + i_101678];
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_85555 = fmax64(lifted_lambda_res_93180, redout_101677);
                double redout_tmp_104927 = max_res_85555;
                
                redout_101677 = redout_tmp_104927;
            }
            defunc_0_reduce_res_93249 = redout_101677;
            // futhark/microgpt.fut:113:47-56
            
            double neg_res_85556 = -defunc_0_reduce_res_93249;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101681 = 0; i_101681 < (int64_t) 16; i_101681++) {
                // futhark/microgpt.fut:113:38-41
                
                double lifted_lambda_res_85563 = ((double *) mem_102850)[i_101693 * (int64_t) 16 + i_101681];
                
                // futhark/microgpt.fut:113:38-56
                
                double zp_res_85564 = neg_res_85556 + lifted_lambda_res_85563;
                
                // futhark/microgpt.fut:113:31-56
                
                double exp_res_85565 = futrts_exp64(zp_res_85564);
                
                ((double *) mem_102871)[i_101681] = exp_res_85565;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_85567;
            double r_85569 = 0.0;
            
            for (int64_t i_85568 = 0; i_85568 < (int64_t) 16; i_85568++) {
                // futhark/microgpt.fut:114:32-39
                
                double lifted_lambda_res_85570 = ((double *) mem_102871)[i_85568];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_85571 = r_85569 + lifted_lambda_res_85570;
                double r_tmp_104929 = zp_res_85571;
                
                r_85569 = r_tmp_104929;
            }
            defunc_0_lifted_lambda_res_85567 = r_85569;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101685 = 0; i_101685 < (int64_t) 16; i_101685++) {
                // futhark/microgpt.fut:115:23-30
                
                double zs_lhs_85578 = ((double *) mem_102871)[i_101685];
                
                // futhark/microgpt.fut:115:23-40
                
                double zs_res_85579 = zs_lhs_85578 / defunc_0_lifted_lambda_res_85567;
                
                ((double *) mem_102878)[i_101685] = zs_res_85579;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101689 = 0; i_101689 < (int64_t) 16; i_101689++) {
                // futhark/microgpt.fut:218:23-31
                
                double lifted_lambda_res_85587 = ((double *) mem_102878)[i_101689];
                
                ((double *) mem_102885)[i_101689] = lifted_lambda_res_85587;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102866, i_101693 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102885, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101701 = 0; i_101701 < (int64_t) 16; i_101701++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101697 = 0; i_101697 < (int64_t) 4; i_101697++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85602;
                double r_85604 = 0.0;
                
                for (int64_t i_85603 = 0; i_85603 < (int64_t) 16; i_85603++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_85605 = ((double *) mem_102866)[i_101701 * (int64_t) 16 + i_85603];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_85606 = ((double *) mem_102747)[i_101705 * (int64_t) 64 + i_85603 * (int64_t) 4 + i_101697];
                    
                    // futhark/microgpt.fut:219:61-96
                    
                    double zt_res_85607 = zt_lhs_85605 * zt_rhs_85606;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85608 = r_85604 + zt_res_85607;
                    double r_tmp_104934 = zp_res_85608;
                    
                    r_85604 = r_tmp_104934;
                }
                defunc_0_lifted_lambda_res_85602 = r_85604;
                ((double *) mem_102901)[i_101697] = defunc_0_lifted_lambda_res_85602;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102896, i_101701 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102901, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_102828, i_101705 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_102896, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102917_cached_sizze_105338 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102917, &mem_102917_cached_sizze_105338, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102922_cached_sizze_105339 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102922, &mem_102922_cached_sizze_105339, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101713 = 0; i_101713 < (int64_t) 16; i_101713++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101709 = 0; i_101709 < (int64_t) 16; i_101709++) {
            // futhark/microgpt.fut:220:61-64
            
            int64_t tmp_85620 = sdiv64(i_101709, (int64_t) 4);
            
            // futhark/microgpt.fut:220:53-66
            
            bool x_85621 = sle64((int64_t) 0, tmp_85620);
            
            // futhark/microgpt.fut:220:53-66
            
            bool y_85622 = slt64(tmp_85620, (int64_t) 4);
            
            // futhark/microgpt.fut:220:53-66
            
            bool bounds_check_85623 = x_85621 && y_85622;
            
            // futhark/microgpt.fut:220:53-66
            
            bool index_certs_85624;
            
            if (!bounds_check_85623) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_85620, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:220:53-66\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:220:35-83\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:220:16-85\n   #7  futhark/microgpt.fut:411:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:220:77-80
            
            int64_t tmp_85625 = smod64(i_101709, (int64_t) 4);
            
            // futhark/microgpt.fut:220:53-82
            
            bool x_85626 = sle64((int64_t) 0, tmp_85625);
            
            // futhark/microgpt.fut:220:53-82
            
            bool y_85627 = slt64(tmp_85625, (int64_t) 4);
            
            // futhark/microgpt.fut:220:53-82
            
            bool bounds_check_85628 = x_85626 && y_85627;
            
            // futhark/microgpt.fut:220:53-82
            
            bool index_certs_85629;
            
            if (!bounds_check_85628) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_85625, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:220:53-82\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:220:35-83\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:220:16-85\n   #7  futhark/microgpt.fut:411:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_85630 = ((double *) mem_102828)[tmp_85620 * (int64_t) 64 + i_101713 * (int64_t) 4 + tmp_85625];
            
            ((double *) mem_102922)[i_101709] = lifted_lambda_res_85630;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102917, i_101713 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102922, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102933_cached_sizze_105340 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102933, &mem_102933_cached_sizze_105340, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102938_cached_sizze_105341 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102938, &mem_102938_cached_sizze_105341, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101721 = 0; i_101721 < (int64_t) 16; i_101721++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101717 = 0; i_101717 < (int64_t) 16; i_101717++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_85645;
            double r_85647 = 0.0;
            
            for (int64_t i_85646 = 0; i_85646 < (int64_t) 16; i_85646++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_85648 = ((double *) wout_mem_102611.mem)[i_101717 * (int64_t) 16 + i_85646];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_85649 = ((double *) mem_102917)[i_101721 * (int64_t) 16 + i_85646];
                
                // futhark/microgpt.fut:221:73-105
                
                double zt_res_85650 = zt_lhs_85648 * zt_rhs_85649;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_85651 = r_85647 + zt_res_85650;
                double r_tmp_104939 = zp_res_85651;
                
                r_85647 = r_tmp_104939;
            }
            defunc_0_lifted_lambda_res_85645 = r_85647;
            ((double *) mem_102938)[i_101717] = defunc_0_lifted_lambda_res_85645;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102933, i_101721 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102938, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102949_cached_sizze_105342 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102949, &mem_102949_cached_sizze_105342, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102954_cached_sizze_105343 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102954, &mem_102954_cached_sizze_105343, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101729 = 0; i_101729 < (int64_t) 16; i_101729++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101725 = 0; i_101725 < (int64_t) 16; i_101725++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_85666 = ((double *) mem_102933)[i_101729 * (int64_t) 16 + i_101725];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_85667 = ((double *) mem_102653)[i_101729 * (int64_t) 16 + i_101725];
            
            // futhark/microgpt.fut:222:42-72
            
            double zp_res_85668 = zp_lhs_85666 + zp_rhs_85667;
            
            ((double *) mem_102954)[i_101725] = zp_res_85668;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102949, i_101729 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102954, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102965_cached_sizze_105344 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102965, &mem_102965_cached_sizze_105344, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102970_cached_sizze_105345 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102970, &mem_102970_cached_sizze_105345, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102977_cached_sizze_105346 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102977, &mem_102977_cached_sizze_105346, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101741 = 0; i_101741 < (int64_t) 16; i_101741++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101733 = 0; i_101733 < (int64_t) 16; i_101733++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_85683 = ((double *) mem_102949)[i_101741 * (int64_t) 16 + i_101733];
            
            // futhark/microgpt.fut:223:65-96
            
            double zt_res_85684 = zt_lhs_85683 * zt_lhs_85683;
            
            ((double *) mem_102970)[i_101733] = zt_res_85684;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_85686;
        double r_85688 = 0.0;
        
        for (int64_t i_85687 = 0; i_85687 < (int64_t) 16; i_85687++) {
            // futhark/microgpt.fut:224:35-43
            
            double lifted_lambda_res_85689 = ((double *) mem_102970)[i_85687];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_85690 = r_85688 + lifted_lambda_res_85689;
            double r_tmp_104944 = zp_res_85690;
            
            r_85688 = r_tmp_104944;
        }
        defunc_0_lifted_lambda_res_85686 = r_85688;
        // futhark/microgpt.fut:224:17-60
        
        double zs_res_85691 = defunc_0_lifted_lambda_res_85686 / 16.0;
        
        // futhark/microgpt.fut:225:24-55
        
        double zp_res_85692 = 1.0e-5 + zs_res_85691;
        
        // futhark/microgpt.fut:225:16-55
        
        double sqrt_res_85693 = futrts_sqrt64(zp_res_85692);
        
        // futhark/microgpt.fut:226:43-54
        
        double zs_res_85694 = 1.0 / sqrt_res_85693;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101737 = 0; i_101737 < (int64_t) 16; i_101737++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_85701 = ((double *) mem_102949)[i_101741 * (int64_t) 16 + i_101737];
            
            // futhark/microgpt.fut:226:24-54
            
            double zt_res_85702 = zs_res_85694 * zt_lhs_85701;
            
            ((double *) mem_102977)[i_101737] = zt_res_85702;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102965, i_101741 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102977, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102988_cached_sizze_105347 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_102988, &mem_102988_cached_sizze_105347, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102993_cached_sizze_105348 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_102993, &mem_102993_cached_sizze_105348, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101749 = 0; i_101749 < (int64_t) 16; i_101749++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101745 = 0; i_101745 < (int64_t) 64; i_101745++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_85718;
            double r_85720 = 0.0;
            
            for (int64_t i_85719 = 0; i_85719 < (int64_t) 16; i_85719++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_85721 = ((double *) wup_mem_102615.mem)[i_101745 * (int64_t) 16 + i_85719];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_85722 = ((double *) mem_102965)[i_101749 * (int64_t) 16 + i_85719];
                
                // futhark/microgpt.fut:227:73-104
                
                double zt_res_85723 = zt_lhs_85721 * zt_rhs_85722;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_85724 = r_85720 + zt_res_85723;
                double r_tmp_104948 = zp_res_85724;
                
                r_85720 = r_tmp_104948;
            }
            defunc_0_lifted_lambda_res_85718 = r_85720;
            ((double *) mem_102993)[i_101745] = defunc_0_lifted_lambda_res_85718;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102988, i_101749 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102993, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103004_cached_sizze_105349 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_103004, &mem_103004_cached_sizze_105349, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103009_cached_sizze_105350 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_103009, &mem_103009_cached_sizze_105350, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101757 = 0; i_101757 < (int64_t) 16; i_101757++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101753 = 0; i_101753 < (int64_t) 64; i_101753++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_85739 = ((double *) mem_102988)[i_101757 * (int64_t) 64 + i_101753];
            
            // futhark/microgpt.fut:228:42-66
            
            double max_res_85740 = fmax64(0.0, max_arg0_85739);
            
            ((double *) mem_103009)[i_101753] = max_res_85740;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_103004, i_101757 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103009, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103020_cached_sizze_105351 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103020, &mem_103020_cached_sizze_105351, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103025_cached_sizze_105352 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103025, &mem_103025_cached_sizze_105352, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101765 = 0; i_101765 < (int64_t) 16; i_101765++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101761 = 0; i_101761 < (int64_t) 16; i_101761++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_85755;
            double r_85757 = 0.0;
            
            for (int64_t i_85756 = 0; i_85756 < (int64_t) 64; i_85756++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_85758 = ((double *) wdown_mem_102609.mem)[i_101761 * (int64_t) 64 + i_85756];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_85759 = ((double *) mem_103004)[i_101765 * (int64_t) 64 + i_85756];
                
                // futhark/microgpt.fut:229:73-106
                
                double zt_res_85760 = zt_lhs_85758 * zt_rhs_85759;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_85761 = r_85757 + zt_res_85760;
                double r_tmp_104953 = zp_res_85761;
                
                r_85757 = r_tmp_104953;
            }
            defunc_0_lifted_lambda_res_85755 = r_85757;
            ((double *) mem_103025)[i_101761] = defunc_0_lifted_lambda_res_85755;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_103020, i_101765 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103025, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103036_cached_sizze_105353 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103036, &mem_103036_cached_sizze_105353, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103041_cached_sizze_105354 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103041, &mem_103041_cached_sizze_105354, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101773 = 0; i_101773 < (int64_t) 16; i_101773++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101769 = 0; i_101769 < (int64_t) 16; i_101769++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_85776 = ((double *) mem_103020)[i_101773 * (int64_t) 16 + i_101769];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_85777 = ((double *) mem_102949)[i_101773 * (int64_t) 16 + i_101769];
            
            // futhark/microgpt.fut:230:42-73
            
            double zp_res_85778 = zp_lhs_85776 + zp_rhs_85777;
            
            ((double *) mem_103041)[i_101769] = zp_res_85778;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_103036, i_101773 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103041, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103052_cached_sizze_105355 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_103052, &mem_103052_cached_sizze_105355, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103057_cached_sizze_105356 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_103057, &mem_103057_cached_sizze_105356, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101781 = 0; i_101781 < (int64_t) 16; i_101781++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101777 = 0; i_101777 < (int64_t) 27; i_101777++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_85794;
            double r_85796 = 0.0;
            
            for (int64_t i_85795 = 0; i_85795 < (int64_t) 16; i_85795++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_85797 = ((double *) wvoc_mem_102617.mem)[i_101777 * (int64_t) 16 + i_85795];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_85798 = ((double *) mem_103036)[i_101781 * (int64_t) 16 + i_85795];
                
                // futhark/microgpt.fut:231:73-105
                
                double zt_res_85799 = zt_lhs_85797 * zt_rhs_85798;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_85800 = r_85796 + zt_res_85799;
                double r_tmp_104958 = zp_res_85800;
                
                r_85796 = r_tmp_104958;
            }
            defunc_0_lifted_lambda_res_85794 = r_85796;
            ((double *) mem_103057)[i_101777] = defunc_0_lifted_lambda_res_85794;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_103052, i_101781 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103057, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103068_cached_sizze_105357 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103068, &mem_103068_cached_sizze_105357, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103072_cached_sizze_105358 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_103072, &mem_103072_cached_sizze_105358, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103079_cached_sizze_105359 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_103079, &mem_103079_cached_sizze_105359, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103086_cached_sizze_105360 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_103086, &mem_103086_cached_sizze_105360, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101799 = 0; i_101799 < (int64_t) 16; i_101799++) {
        // futhark/microgpt.fut:103:13-33
        
        double defunc_0_reduce_res_93268;
        double redout_101783 = -INFINITY;
        
        for (int64_t i_101784 = 0; i_101784 < (int64_t) 27; i_101784++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_93223 = ((double *) mem_103052)[i_101799 * (int64_t) 27 + i_101784];
            
            // futhark/microgpt.fut:103:13-33
            
            double max_res_85821 = fmax64(lifted_lambda_res_93223, redout_101783);
            double redout_tmp_104960 = max_res_85821;
            
            redout_101783 = redout_tmp_104960;
        }
        defunc_0_reduce_res_93268 = redout_101783;
        // futhark/microgpt.fut:113:47-56
        
        double neg_res_85822 = -defunc_0_reduce_res_93268;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101787 = 0; i_101787 < (int64_t) 27; i_101787++) {
            // futhark/microgpt.fut:113:38-41
            
            double lifted_lambda_res_85829 = ((double *) mem_103052)[i_101799 * (int64_t) 27 + i_101787];
            
            // futhark/microgpt.fut:113:38-56
            
            double zp_res_85830 = neg_res_85822 + lifted_lambda_res_85829;
            
            // futhark/microgpt.fut:113:31-56
            
            double exp_res_85831 = futrts_exp64(zp_res_85830);
            
            ((double *) mem_103072)[i_101787] = exp_res_85831;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_85833;
        double r_85835 = 0.0;
        
        for (int64_t i_85834 = 0; i_85834 < (int64_t) 27; i_85834++) {
            // futhark/microgpt.fut:114:32-39
            
            double lifted_lambda_res_85836 = ((double *) mem_103072)[i_85834];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_85837 = r_85835 + lifted_lambda_res_85836;
            double r_tmp_104962 = zp_res_85837;
            
            r_85835 = r_tmp_104962;
        }
        defunc_0_lifted_lambda_res_85833 = r_85835;
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101791 = 0; i_101791 < (int64_t) 27; i_101791++) {
            // futhark/microgpt.fut:115:23-30
            
            double zs_lhs_85844 = ((double *) mem_103072)[i_101791];
            
            // futhark/microgpt.fut:115:23-40
            
            double zs_res_85845 = zs_lhs_85844 / defunc_0_lifted_lambda_res_85833;
            
            ((double *) mem_103079)[i_101791] = zs_res_85845;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101795 = 0; i_101795 < (int64_t) 27; i_101795++) {
            // futhark/microgpt.fut:233:4-14
            
            double log_arg0_85853 = ((double *) mem_103079)[i_101795];
            
            // futhark/microgpt.fut:232:66-233:14
            
            double log_res_85854 = futrts_log64(log_arg0_85853);
            
            ((double *) mem_103086)[i_101795] = log_res_85854;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_85856;
        double r_85858 = 0.0;
        
        for (int64_t i_85857 = 0; i_85857 < (int64_t) 27; i_85857++) {
            // futhark/microgpt.fut:234:32-41
            
            double zt_lhs_85859 = ((double *) mem_103086)[i_85857];
            
            // futhark/microgpt.fut:71:46-49
            
            double zt_rhs_85860 = ((double *) ext_mem_102620.mem)[i_101799 * (int64_t) 27 + i_85857];
            
            // futhark/microgpt.fut:234:32-63
            
            double zt_res_85861 = zt_lhs_85859 * zt_rhs_85860;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_85862 = r_85858 + zt_res_85861;
            double r_tmp_104965 = zp_res_85862;
            
            r_85858 = r_tmp_104965;
        }
        defunc_0_lifted_lambda_res_85856 = r_85858;
        // futhark/microgpt.fut:234:5-65
        
        double neg_res_85863 = -defunc_0_lifted_lambda_res_85856;
        
        ((double *) mem_103068)[i_101799] = neg_res_85863;
    }
    if (memblock_unref(ctx, &ext_mem_102620, "ext_mem_102620") != 0)
        return 1;
    // futhark/microgpt.fut:71:13-49
    
    double defunc_0_lifted_lambda_res_85865;
    double r_85867 = 0.0;
    
    for (int64_t i_85866 = 0; i_85866 < (int64_t) 16; i_85866++) {
        // futhark/microgpt.fut:235:24-32
        
        double lifted_lambda_res_85868 = ((double *) mem_103068)[i_85866];
        
        // futhark/microgpt.fut:71:40-49
        
        double zp_res_85869 = r_85867 + lifted_lambda_res_85868;
        double r_tmp_104966 = zp_res_85869;
        
        r_85867 = r_tmp_104966;
    }
    defunc_0_lifted_lambda_res_85865 = r_85867;
    // futhark/microgpt.fut:235:6-49
    
    double zs_res_85870 = defunc_0_lifted_lambda_res_85865 / 16.0;
    
    prim_out_104889 = zs_res_85870;
    *out_prim_out_105301 = prim_out_104889;
    
  cleanup:
    {
        free(mem_102621);
        free(mem_102626);
        free(mem_102637);
        free(mem_102642);
        free(mem_102653);
        free(mem_102658);
        free(mem_102665);
        free(mem_102676);
        free(mem_102681);
        free(mem_102688);
        free(mem_102699);
        free(mem_102700);
        free(mem_102701);
        free(mem_102714);
        free(mem_102715);
        free(mem_102716);
        free(mem_102747);
        free(mem_102748);
        free(mem_102749);
        free(mem_102765);
        free(mem_102766);
        free(mem_102767);
        free(mem_102780);
        free(mem_102781);
        free(mem_102782);
        free(mem_102828);
        free(mem_102834);
        free(mem_102839);
        free(mem_102850);
        free(mem_102855);
        free(mem_102866);
        free(mem_102871);
        free(mem_102878);
        free(mem_102885);
        free(mem_102896);
        free(mem_102901);
        free(mem_102917);
        free(mem_102922);
        free(mem_102933);
        free(mem_102938);
        free(mem_102949);
        free(mem_102954);
        free(mem_102965);
        free(mem_102970);
        free(mem_102977);
        free(mem_102988);
        free(mem_102993);
        free(mem_103004);
        free(mem_103009);
        free(mem_103020);
        free(mem_103025);
        free(mem_103036);
        free(mem_103041);
        free(mem_103052);
        free(mem_103057);
        free(mem_103068);
        free(mem_103072);
        free(mem_103079);
        free(mem_103086);
        if (memblock_unref(ctx, &ext_mem_102620, "ext_mem_102620") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_105361, struct memblock wdown_mem_102609, struct memblock wkey_mem_102610, struct memblock wout_mem_102611, struct memblock wpe_mem_102612, struct memblock wqry_mem_102613, struct memblock wte_mem_102614, struct memblock wup_mem_102615, struct memblock wval_mem_102616, struct memblock wvoc_mem_102617, struct memblock tokens_mem_102618, struct memblock mask_mem_102619)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_102620_cached_sizze_105362 = 0;
    unsigned char *mem_102620 = NULL;
    int64_t mem_102625_cached_sizze_105363 = 0;
    unsigned char *mem_102625 = NULL;
    int64_t mem_102636_cached_sizze_105364 = 0;
    unsigned char *mem_102636 = NULL;
    int64_t mem_102641_cached_sizze_105365 = 0;
    unsigned char *mem_102641 = NULL;
    int64_t mem_102652_cached_sizze_105366 = 0;
    unsigned char *mem_102652 = NULL;
    int64_t mem_102657_cached_sizze_105367 = 0;
    unsigned char *mem_102657 = NULL;
    int64_t mem_102664_cached_sizze_105368 = 0;
    unsigned char *mem_102664 = NULL;
    int64_t mem_102675_cached_sizze_105369 = 0;
    unsigned char *mem_102675 = NULL;
    int64_t mem_102680_cached_sizze_105370 = 0;
    unsigned char *mem_102680 = NULL;
    int64_t mem_102687_cached_sizze_105371 = 0;
    unsigned char *mem_102687 = NULL;
    int64_t mem_102698_cached_sizze_105372 = 0;
    unsigned char *mem_102698 = NULL;
    int64_t mem_102699_cached_sizze_105373 = 0;
    unsigned char *mem_102699 = NULL;
    int64_t mem_102700_cached_sizze_105374 = 0;
    unsigned char *mem_102700 = NULL;
    int64_t mem_102713_cached_sizze_105375 = 0;
    unsigned char *mem_102713 = NULL;
    int64_t mem_102714_cached_sizze_105376 = 0;
    unsigned char *mem_102714 = NULL;
    int64_t mem_102715_cached_sizze_105377 = 0;
    unsigned char *mem_102715 = NULL;
    int64_t mem_102746_cached_sizze_105378 = 0;
    unsigned char *mem_102746 = NULL;
    int64_t mem_102747_cached_sizze_105379 = 0;
    unsigned char *mem_102747 = NULL;
    int64_t mem_102748_cached_sizze_105380 = 0;
    unsigned char *mem_102748 = NULL;
    int64_t mem_102764_cached_sizze_105381 = 0;
    unsigned char *mem_102764 = NULL;
    int64_t mem_102765_cached_sizze_105382 = 0;
    unsigned char *mem_102765 = NULL;
    int64_t mem_102766_cached_sizze_105383 = 0;
    unsigned char *mem_102766 = NULL;
    int64_t mem_102779_cached_sizze_105384 = 0;
    unsigned char *mem_102779 = NULL;
    int64_t mem_102780_cached_sizze_105385 = 0;
    unsigned char *mem_102780 = NULL;
    int64_t mem_102781_cached_sizze_105386 = 0;
    unsigned char *mem_102781 = NULL;
    int64_t mem_102827_cached_sizze_105387 = 0;
    unsigned char *mem_102827 = NULL;
    int64_t mem_102833_cached_sizze_105388 = 0;
    unsigned char *mem_102833 = NULL;
    int64_t mem_102838_cached_sizze_105389 = 0;
    unsigned char *mem_102838 = NULL;
    int64_t mem_102849_cached_sizze_105390 = 0;
    unsigned char *mem_102849 = NULL;
    int64_t mem_102854_cached_sizze_105391 = 0;
    unsigned char *mem_102854 = NULL;
    int64_t mem_102865_cached_sizze_105392 = 0;
    unsigned char *mem_102865 = NULL;
    int64_t mem_102870_cached_sizze_105393 = 0;
    unsigned char *mem_102870 = NULL;
    int64_t mem_102877_cached_sizze_105394 = 0;
    unsigned char *mem_102877 = NULL;
    int64_t mem_102884_cached_sizze_105395 = 0;
    unsigned char *mem_102884 = NULL;
    int64_t mem_102895_cached_sizze_105396 = 0;
    unsigned char *mem_102895 = NULL;
    int64_t mem_102900_cached_sizze_105397 = 0;
    unsigned char *mem_102900 = NULL;
    int64_t mem_102916_cached_sizze_105398 = 0;
    unsigned char *mem_102916 = NULL;
    int64_t mem_102921_cached_sizze_105399 = 0;
    unsigned char *mem_102921 = NULL;
    int64_t mem_102932_cached_sizze_105400 = 0;
    unsigned char *mem_102932 = NULL;
    int64_t mem_102937_cached_sizze_105401 = 0;
    unsigned char *mem_102937 = NULL;
    int64_t mem_102948_cached_sizze_105402 = 0;
    unsigned char *mem_102948 = NULL;
    int64_t mem_102953_cached_sizze_105403 = 0;
    unsigned char *mem_102953 = NULL;
    int64_t mem_102964_cached_sizze_105404 = 0;
    unsigned char *mem_102964 = NULL;
    int64_t mem_102969_cached_sizze_105405 = 0;
    unsigned char *mem_102969 = NULL;
    int64_t mem_102976_cached_sizze_105406 = 0;
    unsigned char *mem_102976 = NULL;
    int64_t mem_102987_cached_sizze_105407 = 0;
    unsigned char *mem_102987 = NULL;
    int64_t mem_102992_cached_sizze_105408 = 0;
    unsigned char *mem_102992 = NULL;
    int64_t mem_103003_cached_sizze_105409 = 0;
    unsigned char *mem_103003 = NULL;
    int64_t mem_103008_cached_sizze_105410 = 0;
    unsigned char *mem_103008 = NULL;
    int64_t mem_103019_cached_sizze_105411 = 0;
    unsigned char *mem_103019 = NULL;
    int64_t mem_103024_cached_sizze_105412 = 0;
    unsigned char *mem_103024 = NULL;
    int64_t mem_103035_cached_sizze_105413 = 0;
    unsigned char *mem_103035 = NULL;
    int64_t mem_103040_cached_sizze_105414 = 0;
    unsigned char *mem_103040 = NULL;
    int64_t mem_103056_cached_sizze_105415 = 0;
    unsigned char *mem_103056 = NULL;
    struct memblock mem_103051;
    
    mem_103051.references = NULL;
    
    struct memblock mem_out_104889;
    
    mem_out_104889.references = NULL;
    
    struct memblock mem_102600 = ctx->constants->mem_102600;
    struct memblock mem_102601 = ctx->constants->mem_102601;
    struct memblock mem_102602 = ctx->constants->mem_102602;
    struct memblock mem_102603 = ctx->constants->mem_102603;
    struct memblock mem_102604 = ctx->constants->mem_102604;
    struct memblock mem_102605 = ctx->constants->mem_102605;
    struct memblock mem_102606 = ctx->constants->mem_102606;
    struct memblock mem_102607 = ctx->constants->mem_102607;
    struct memblock mem_102608 = ctx->constants->mem_102608;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_102620_cached_sizze_105362 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102620, &mem_102620_cached_sizze_105362, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102625_cached_sizze_105363 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102625, &mem_102625_cached_sizze_105363, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101577 = 0; i_101577 < (int64_t) 16; i_101577++) {
        // futhark/microgpt.fut:401:41-50
        
        int64_t tmp_85250 = ((int64_t *) tokens_mem_102618.mem)[i_101577];
        
        // futhark/microgpt.fut:401:37-51
        
        bool x_85251 = sle64((int64_t) 0, tmp_85250);
        
        // futhark/microgpt.fut:401:37-51
        
        bool y_85252 = slt64(tmp_85250, (int64_t) 27);
        
        // futhark/microgpt.fut:401:37-51
        
        bool bounds_check_85253 = x_85251 && y_85252;
        
        // futhark/microgpt.fut:401:37-51
        
        bool index_certs_85254;
        
        if (!bounds_check_85253) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_85250, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:401:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:401:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101573 = 0; i_101573 < (int64_t) 16; i_101573++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_85261 = ((double *) wte_mem_102614.mem)[tmp_85250 * (int64_t) 16 + i_101573];
            
            ((double *) mem_102625)[i_101573] = lifted_lambda_res_85261;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102620, i_101577 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102625, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102636_cached_sizze_105364 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102636, &mem_102636_cached_sizze_105364, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102641_cached_sizze_105365 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102641, &mem_102641_cached_sizze_105365, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101585 = 0; i_101585 < (int64_t) 16; i_101585++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101581 = 0; i_101581 < (int64_t) 16; i_101581++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_85293 = ((double *) wpe_mem_102612.mem)[i_101585 * (int64_t) 16 + i_101581];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_85294 = ((double *) mem_102620)[i_101585 * (int64_t) 16 + i_101581];
            
            // futhark/microgpt.fut:149:38-70
            
            double zp_res_85295 = zp_lhs_85293 + zp_rhs_85294;
            
            ((double *) mem_102641)[i_101581] = zp_res_85295;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102636, i_101585 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102641, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102652_cached_sizze_105366 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102652, &mem_102652_cached_sizze_105366, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102657_cached_sizze_105367 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102657, &mem_102657_cached_sizze_105367, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102664_cached_sizze_105368 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102664, &mem_102664_cached_sizze_105368, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101597 = 0; i_101597 < (int64_t) 16; i_101597++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101589 = 0; i_101589 < (int64_t) 16; i_101589++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_85310 = ((double *) mem_102636)[i_101597 * (int64_t) 16 + i_101589];
            
            // futhark/microgpt.fut:150:64-93
            
            double zt_res_85311 = zt_lhs_85310 * zt_lhs_85310;
            
            ((double *) mem_102657)[i_101589] = zt_res_85311;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_85313;
        double r_85315 = 0.0;
        
        for (int64_t i_85314 = 0; i_85314 < (int64_t) 16; i_85314++) {
            // futhark/microgpt.fut:151:35-43
            
            double lifted_lambda_res_85316 = ((double *) mem_102657)[i_85314];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_85317 = r_85315 + lifted_lambda_res_85316;
            double r_tmp_104896 = zp_res_85317;
            
            r_85315 = r_tmp_104896;
        }
        defunc_0_lifted_lambda_res_85313 = r_85315;
        // futhark/microgpt.fut:151:17-60
        
        double zs_res_85318 = defunc_0_lifted_lambda_res_85313 / 16.0;
        
        // futhark/microgpt.fut:152:24-55
        
        double zp_res_85319 = 1.0e-5 + zs_res_85318;
        
        // futhark/microgpt.fut:152:16-55
        
        double sqrt_res_85320 = futrts_sqrt64(zp_res_85319);
        
        // futhark/microgpt.fut:153:42-53
        
        double zs_res_85321 = 1.0 / sqrt_res_85320;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101593 = 0; i_101593 < (int64_t) 16; i_101593++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_85328 = ((double *) mem_102636)[i_101597 * (int64_t) 16 + i_101593];
            
            // futhark/microgpt.fut:153:24-53
            
            double zt_res_85329 = zs_res_85321 * zt_lhs_85328;
            
            ((double *) mem_102664)[i_101593] = zt_res_85329;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102652, i_101597 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102664, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102675_cached_sizze_105369 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102675, &mem_102675_cached_sizze_105369, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102680_cached_sizze_105370 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102680, &mem_102680_cached_sizze_105370, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102687_cached_sizze_105371 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102687, &mem_102687_cached_sizze_105371, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101609 = 0; i_101609 < (int64_t) 16; i_101609++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101601 = 0; i_101601 < (int64_t) 16; i_101601++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_85344 = ((double *) mem_102652)[i_101609 * (int64_t) 16 + i_101601];
            
            // futhark/microgpt.fut:154:64-93
            
            double zt_res_85345 = zt_lhs_85344 * zt_lhs_85344;
            
            ((double *) mem_102680)[i_101601] = zt_res_85345;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_85347;
        double r_85349 = 0.0;
        
        for (int64_t i_85348 = 0; i_85348 < (int64_t) 16; i_85348++) {
            // futhark/microgpt.fut:155:35-43
            
            double lifted_lambda_res_85350 = ((double *) mem_102680)[i_85348];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_85351 = r_85349 + lifted_lambda_res_85350;
            double r_tmp_104900 = zp_res_85351;
            
            r_85349 = r_tmp_104900;
        }
        defunc_0_lifted_lambda_res_85347 = r_85349;
        // futhark/microgpt.fut:155:17-60
        
        double zs_res_85352 = defunc_0_lifted_lambda_res_85347 / 16.0;
        
        // futhark/microgpt.fut:156:24-55
        
        double zp_res_85353 = 1.0e-5 + zs_res_85352;
        
        // futhark/microgpt.fut:156:16-55
        
        double sqrt_res_85354 = futrts_sqrt64(zp_res_85353);
        
        // futhark/microgpt.fut:157:42-53
        
        double zs_res_85355 = 1.0 / sqrt_res_85354;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101605 = 0; i_101605 < (int64_t) 16; i_101605++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_85362 = ((double *) mem_102652)[i_101609 * (int64_t) 16 + i_101605];
            
            // futhark/microgpt.fut:157:24-53
            
            double zt_res_85363 = zs_res_85355 * zt_lhs_85362;
            
            ((double *) mem_102687)[i_101605] = zt_res_85363;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102675, i_101609 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102687, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102698_cached_sizze_105372 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102698, &mem_102698_cached_sizze_105372, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102699_cached_sizze_105373 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102699, &mem_102699_cached_sizze_105373, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102700_cached_sizze_105374 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102700, &mem_102700_cached_sizze_105374, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102713_cached_sizze_105375 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102713, &mem_102713_cached_sizze_105375, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102714_cached_sizze_105376 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102714, &mem_102714_cached_sizze_105376, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102715_cached_sizze_105377 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102715, &mem_102715_cached_sizze_105377, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101627 = 0; i_101627 < (int64_t) 16; i_101627++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101617 = 0; i_101617 < (int64_t) 16; i_101617++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_92936;
            double r_92938 = 0.0;
            
            for (int64_t i_92937 = 0; i_92937 < (int64_t) 16; i_92937++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_92939 = ((double *) wqry_mem_102613.mem)[i_101617 * (int64_t) 16 + i_92937];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_92940 = ((double *) mem_102675)[i_101627 * (int64_t) 16 + i_92937];
                
                // futhark/microgpt.fut:158:72-103
                
                double zt_res_92941 = zt_lhs_92939 * zt_rhs_92940;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_92942 = r_92938 + zt_res_92941;
                double r_tmp_104908 = zp_res_92942;
                
                r_92938 = r_tmp_104908;
            }
            defunc_0_lifted_lambda_res_92936 = r_92938;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_92949;
            double r_92951 = 0.0;
            
            for (int64_t i_92950 = 0; i_92950 < (int64_t) 16; i_92950++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_92952 = ((double *) wkey_mem_102610.mem)[i_101617 * (int64_t) 16 + i_92950];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_92953 = ((double *) mem_102675)[i_101627 * (int64_t) 16 + i_92950];
                
                // futhark/microgpt.fut:159:72-103
                
                double zt_res_92954 = zt_lhs_92952 * zt_rhs_92953;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_92955 = r_92951 + zt_res_92954;
                double r_tmp_104909 = zp_res_92955;
                
                r_92951 = r_tmp_104909;
            }
            defunc_0_lifted_lambda_res_92949 = r_92951;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_92965;
            double r_92967 = 0.0;
            
            for (int64_t i_92966 = 0; i_92966 < (int64_t) 16; i_92966++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_92968 = ((double *) wval_mem_102616.mem)[i_101617 * (int64_t) 16 + i_92966];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_92969 = ((double *) mem_102675)[i_101627 * (int64_t) 16 + i_92966];
                
                // futhark/microgpt.fut:160:72-103
                
                double zt_res_92970 = zt_lhs_92968 * zt_rhs_92969;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_92971 = r_92967 + zt_res_92970;
                double r_tmp_104910 = zp_res_92971;
                
                r_92967 = r_tmp_104910;
            }
            defunc_0_lifted_lambda_res_92965 = r_92967;
            ((double *) mem_102713)[i_101617] = defunc_0_lifted_lambda_res_92965;
            ((double *) mem_102714)[i_101617] = defunc_0_lifted_lambda_res_92949;
            ((double *) mem_102715)[i_101617] = defunc_0_lifted_lambda_res_92936;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102698, i_101627 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102713, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102699, i_101627 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102714, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102700, i_101627 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102715, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102746_cached_sizze_105378 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102746, &mem_102746_cached_sizze_105378, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102747_cached_sizze_105379 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102747, &mem_102747_cached_sizze_105379, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102748_cached_sizze_105380 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102748, &mem_102748_cached_sizze_105380, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102764_cached_sizze_105381 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_102764, &mem_102764_cached_sizze_105381, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102765_cached_sizze_105382 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_102765, &mem_102765_cached_sizze_105382, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102766_cached_sizze_105383 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_102766, &mem_102766_cached_sizze_105383, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102779_cached_sizze_105384 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_102779, &mem_102779_cached_sizze_105384, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102780_cached_sizze_105385 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_102780, &mem_102780_cached_sizze_105385, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102781_cached_sizze_105386 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_102781, &mem_102781_cached_sizze_105386, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101657 = 0; i_101657 < (int64_t) 4; i_101657++) {
        // futhark/microgpt.fut:161:83-86
        
        int64_t zp_lhs_92811 = mul64((int64_t) 4, i_101657);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101647 = 0; i_101647 < (int64_t) 16; i_101647++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101637 = 0; i_101637 < (int64_t) 4; i_101637++) {
                // futhark/microgpt.fut:161:88-93
                
                int64_t tmp_93129 = add64(zp_lhs_92811, i_101637);
                
                // futhark/microgpt.fut:161:69-95
                
                bool x_93130 = sle64((int64_t) 0, tmp_93129);
                
                // futhark/microgpt.fut:161:69-95
                
                bool y_93131 = slt64(tmp_93129, (int64_t) 16);
                
                // futhark/microgpt.fut:161:69-95
                
                bool bounds_check_93132 = x_93130 && y_93131;
                
                // futhark/microgpt.fut:161:69-95
                
                bool index_certs_93133;
                
                if (!bounds_check_93132) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_93129, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:161:69-95\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:161:52-96\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:161:33-98\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:6:13-17\n   #9  futhark/microgpt.fut:161:15-100\n   #10 futhark/microgpt.fut:402:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_93134 = ((double *) mem_102700)[i_101647 * (int64_t) 16 + tmp_93129];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_93142 = ((double *) mem_102699)[i_101647 * (int64_t) 16 + tmp_93129];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_93153 = ((double *) mem_102698)[i_101647 * (int64_t) 16 + tmp_93129];
                
                ((double *) mem_102779)[i_101637] = lifted_lambda_res_93153;
                ((double *) mem_102780)[i_101637] = lifted_lambda_res_93142;
                ((double *) mem_102781)[i_101637] = lifted_lambda_res_93134;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102764, i_101647 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102779, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102765, i_101647 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102780, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102766, i_101647 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102781, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_102746, i_101657 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_102764, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_102747, i_101657 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_102765, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_102748, i_101657 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_102766, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102827_cached_sizze_105387 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102827, &mem_102827_cached_sizze_105387, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102833_cached_sizze_105388 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102833, &mem_102833_cached_sizze_105388, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102838_cached_sizze_105389 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102838, &mem_102838_cached_sizze_105389, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102849_cached_sizze_105390 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102849, &mem_102849_cached_sizze_105390, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102854_cached_sizze_105391 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102854, &mem_102854_cached_sizze_105391, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102865_cached_sizze_105392 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102865, &mem_102865_cached_sizze_105392, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102870_cached_sizze_105393 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102870, &mem_102870_cached_sizze_105393, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102877_cached_sizze_105394 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102877, &mem_102877_cached_sizze_105394, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102884_cached_sizze_105395 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102884, &mem_102884_cached_sizze_105395, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102895_cached_sizze_105396 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_102895, &mem_102895_cached_sizze_105396, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102900_cached_sizze_105397 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_102900, &mem_102900_cached_sizze_105397, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101705 = 0; i_101705 < (int64_t) 4; i_101705++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101667 = 0; i_101667 < (int64_t) 16; i_101667++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101663 = 0; i_101663 < (int64_t) 16; i_101663++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85508;
                double r_85510 = 0.0;
                
                for (int64_t i_85509 = 0; i_85509 < (int64_t) 4; i_85509++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_85511 = ((double *) mem_102748)[i_101705 * (int64_t) 64 + i_101667 * (int64_t) 4 + i_85509];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_85512 = ((double *) mem_102747)[i_101705 * (int64_t) 64 + i_101663 * (int64_t) 4 + i_85509];
                    
                    // futhark/microgpt.fut:164:100-139
                    
                    double zt_res_85513 = zt_lhs_85511 * zt_rhs_85512;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85514 = r_85510 + zt_res_85513;
                    double r_tmp_104923 = zp_res_85514;
                    
                    r_85510 = r_tmp_104923;
                }
                defunc_0_lifted_lambda_res_85508 = r_85510;
                ((double *) mem_102838)[i_101663] = defunc_0_lifted_lambda_res_85508;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102833, i_101667 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102838, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101675 = 0; i_101675 < (int64_t) 16; i_101675++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101671 = 0; i_101671 < (int64_t) 16; i_101671++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_85529 = ((double *) mem_102833)[i_101675 * (int64_t) 16 + i_101671];
                
                // futhark/microgpt.fut:165:43-70
                
                double zs_res_85530 = zs_lhs_85529 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_85531 = ((double *) mask_mem_102619.mem)[i_101675 * (int64_t) 16 + i_101671];
                
                // futhark/microgpt.fut:165:57-90
                
                double zp_res_85532 = zs_res_85530 + zp_rhs_85531;
                
                ((double *) mem_102854)[i_101671] = zp_res_85532;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102849, i_101675 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102854, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101693 = 0; i_101693 < (int64_t) 16; i_101693++) {
            // futhark/microgpt.fut:103:13-33
            
            double defunc_0_reduce_res_93228;
            double redout_101677 = -INFINITY;
            
            for (int64_t i_101678 = 0; i_101678 < (int64_t) 16; i_101678++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_93180 = ((double *) mem_102849)[i_101693 * (int64_t) 16 + i_101678];
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_85553 = fmax64(lifted_lambda_res_93180, redout_101677);
                double redout_tmp_104927 = max_res_85553;
                
                redout_101677 = redout_tmp_104927;
            }
            defunc_0_reduce_res_93228 = redout_101677;
            // futhark/microgpt.fut:113:47-56
            
            double neg_res_85554 = -defunc_0_reduce_res_93228;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101681 = 0; i_101681 < (int64_t) 16; i_101681++) {
                // futhark/microgpt.fut:113:38-41
                
                double lifted_lambda_res_85561 = ((double *) mem_102849)[i_101693 * (int64_t) 16 + i_101681];
                
                // futhark/microgpt.fut:113:38-56
                
                double zp_res_85562 = neg_res_85554 + lifted_lambda_res_85561;
                
                // futhark/microgpt.fut:113:31-56
                
                double exp_res_85563 = futrts_exp64(zp_res_85562);
                
                ((double *) mem_102870)[i_101681] = exp_res_85563;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_85565;
            double r_85567 = 0.0;
            
            for (int64_t i_85566 = 0; i_85566 < (int64_t) 16; i_85566++) {
                // futhark/microgpt.fut:114:32-39
                
                double lifted_lambda_res_85568 = ((double *) mem_102870)[i_85566];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_85569 = r_85567 + lifted_lambda_res_85568;
                double r_tmp_104929 = zp_res_85569;
                
                r_85567 = r_tmp_104929;
            }
            defunc_0_lifted_lambda_res_85565 = r_85567;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101685 = 0; i_101685 < (int64_t) 16; i_101685++) {
                // futhark/microgpt.fut:115:23-30
                
                double zs_lhs_85576 = ((double *) mem_102870)[i_101685];
                
                // futhark/microgpt.fut:115:23-40
                
                double zs_res_85577 = zs_lhs_85576 / defunc_0_lifted_lambda_res_85565;
                
                ((double *) mem_102877)[i_101685] = zs_res_85577;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101689 = 0; i_101689 < (int64_t) 16; i_101689++) {
                // futhark/microgpt.fut:167:23-31
                
                double lifted_lambda_res_85585 = ((double *) mem_102877)[i_101689];
                
                ((double *) mem_102884)[i_101689] = lifted_lambda_res_85585;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102865, i_101693 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102884, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101701 = 0; i_101701 < (int64_t) 16; i_101701++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101697 = 0; i_101697 < (int64_t) 4; i_101697++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85600;
                double r_85602 = 0.0;
                
                for (int64_t i_85601 = 0; i_85601 < (int64_t) 16; i_85601++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_85603 = ((double *) mem_102865)[i_101701 * (int64_t) 16 + i_85601];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_85604 = ((double *) mem_102746)[i_101705 * (int64_t) 64 + i_85601 * (int64_t) 4 + i_101697];
                    
                    // futhark/microgpt.fut:168:61-96
                    
                    double zt_res_85605 = zt_lhs_85603 * zt_rhs_85604;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85606 = r_85602 + zt_res_85605;
                    double r_tmp_104934 = zp_res_85606;
                    
                    r_85602 = r_tmp_104934;
                }
                defunc_0_lifted_lambda_res_85600 = r_85602;
                ((double *) mem_102900)[i_101697] = defunc_0_lifted_lambda_res_85600;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102895, i_101701 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102900, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_102827, i_101705 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_102895, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102916_cached_sizze_105398 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102916, &mem_102916_cached_sizze_105398, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102921_cached_sizze_105399 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102921, &mem_102921_cached_sizze_105399, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101713 = 0; i_101713 < (int64_t) 16; i_101713++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101709 = 0; i_101709 < (int64_t) 16; i_101709++) {
            // futhark/microgpt.fut:169:61-64
            
            int64_t tmp_85618 = sdiv64(i_101709, (int64_t) 4);
            
            // futhark/microgpt.fut:169:53-66
            
            bool x_85619 = sle64((int64_t) 0, tmp_85618);
            
            // futhark/microgpt.fut:169:53-66
            
            bool y_85620 = slt64(tmp_85618, (int64_t) 4);
            
            // futhark/microgpt.fut:169:53-66
            
            bool bounds_check_85621 = x_85619 && y_85620;
            
            // futhark/microgpt.fut:169:53-66
            
            bool index_certs_85622;
            
            if (!bounds_check_85621) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_85618, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:169:53-66\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:169:35-83\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:169:16-85\n   #7  futhark/microgpt.fut:402:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:169:77-80
            
            int64_t tmp_85623 = smod64(i_101709, (int64_t) 4);
            
            // futhark/microgpt.fut:169:53-82
            
            bool x_85624 = sle64((int64_t) 0, tmp_85623);
            
            // futhark/microgpt.fut:169:53-82
            
            bool y_85625 = slt64(tmp_85623, (int64_t) 4);
            
            // futhark/microgpt.fut:169:53-82
            
            bool bounds_check_85626 = x_85624 && y_85625;
            
            // futhark/microgpt.fut:169:53-82
            
            bool index_certs_85627;
            
            if (!bounds_check_85626) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_85623, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:169:53-82\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:169:35-83\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:169:16-85\n   #7  futhark/microgpt.fut:402:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_85628 = ((double *) mem_102827)[tmp_85618 * (int64_t) 64 + i_101713 * (int64_t) 4 + tmp_85623];
            
            ((double *) mem_102921)[i_101709] = lifted_lambda_res_85628;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102916, i_101713 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102921, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102932_cached_sizze_105400 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102932, &mem_102932_cached_sizze_105400, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102937_cached_sizze_105401 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102937, &mem_102937_cached_sizze_105401, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101721 = 0; i_101721 < (int64_t) 16; i_101721++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101717 = 0; i_101717 < (int64_t) 16; i_101717++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_85643;
            double r_85645 = 0.0;
            
            for (int64_t i_85644 = 0; i_85644 < (int64_t) 16; i_85644++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_85646 = ((double *) wout_mem_102611.mem)[i_101717 * (int64_t) 16 + i_85644];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_85647 = ((double *) mem_102916)[i_101721 * (int64_t) 16 + i_85644];
                
                // futhark/microgpt.fut:170:73-105
                
                double zt_res_85648 = zt_lhs_85646 * zt_rhs_85647;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_85649 = r_85645 + zt_res_85648;
                double r_tmp_104939 = zp_res_85649;
                
                r_85645 = r_tmp_104939;
            }
            defunc_0_lifted_lambda_res_85643 = r_85645;
            ((double *) mem_102937)[i_101717] = defunc_0_lifted_lambda_res_85643;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102932, i_101721 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102937, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102948_cached_sizze_105402 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102948, &mem_102948_cached_sizze_105402, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102953_cached_sizze_105403 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102953, &mem_102953_cached_sizze_105403, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101729 = 0; i_101729 < (int64_t) 16; i_101729++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101725 = 0; i_101725 < (int64_t) 16; i_101725++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_85664 = ((double *) mem_102932)[i_101729 * (int64_t) 16 + i_101725];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_85665 = ((double *) mem_102652)[i_101729 * (int64_t) 16 + i_101725];
            
            // futhark/microgpt.fut:171:42-72
            
            double zp_res_85666 = zp_lhs_85664 + zp_rhs_85665;
            
            ((double *) mem_102953)[i_101725] = zp_res_85666;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102948, i_101729 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102953, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102964_cached_sizze_105404 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102964, &mem_102964_cached_sizze_105404, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102969_cached_sizze_105405 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102969, &mem_102969_cached_sizze_105405, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102976_cached_sizze_105406 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102976, &mem_102976_cached_sizze_105406, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101741 = 0; i_101741 < (int64_t) 16; i_101741++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101733 = 0; i_101733 < (int64_t) 16; i_101733++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_85681 = ((double *) mem_102948)[i_101741 * (int64_t) 16 + i_101733];
            
            // futhark/microgpt.fut:172:65-96
            
            double zt_res_85682 = zt_lhs_85681 * zt_lhs_85681;
            
            ((double *) mem_102969)[i_101733] = zt_res_85682;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_85684;
        double r_85686 = 0.0;
        
        for (int64_t i_85685 = 0; i_85685 < (int64_t) 16; i_85685++) {
            // futhark/microgpt.fut:173:35-43
            
            double lifted_lambda_res_85687 = ((double *) mem_102969)[i_85685];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_85688 = r_85686 + lifted_lambda_res_85687;
            double r_tmp_104944 = zp_res_85688;
            
            r_85686 = r_tmp_104944;
        }
        defunc_0_lifted_lambda_res_85684 = r_85686;
        // futhark/microgpt.fut:173:17-60
        
        double zs_res_85689 = defunc_0_lifted_lambda_res_85684 / 16.0;
        
        // futhark/microgpt.fut:174:24-55
        
        double zp_res_85690 = 1.0e-5 + zs_res_85689;
        
        // futhark/microgpt.fut:174:16-55
        
        double sqrt_res_85691 = futrts_sqrt64(zp_res_85690);
        
        // futhark/microgpt.fut:175:43-54
        
        double zs_res_85692 = 1.0 / sqrt_res_85691;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101737 = 0; i_101737 < (int64_t) 16; i_101737++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_85699 = ((double *) mem_102948)[i_101741 * (int64_t) 16 + i_101737];
            
            // futhark/microgpt.fut:175:24-54
            
            double zt_res_85700 = zs_res_85692 * zt_lhs_85699;
            
            ((double *) mem_102976)[i_101737] = zt_res_85700;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102964, i_101741 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102976, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102987_cached_sizze_105407 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_102987, &mem_102987_cached_sizze_105407, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102992_cached_sizze_105408 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_102992, &mem_102992_cached_sizze_105408, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101749 = 0; i_101749 < (int64_t) 16; i_101749++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101745 = 0; i_101745 < (int64_t) 64; i_101745++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_85716;
            double r_85718 = 0.0;
            
            for (int64_t i_85717 = 0; i_85717 < (int64_t) 16; i_85717++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_85719 = ((double *) wup_mem_102615.mem)[i_101745 * (int64_t) 16 + i_85717];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_85720 = ((double *) mem_102964)[i_101749 * (int64_t) 16 + i_85717];
                
                // futhark/microgpt.fut:176:73-104
                
                double zt_res_85721 = zt_lhs_85719 * zt_rhs_85720;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_85722 = r_85718 + zt_res_85721;
                double r_tmp_104948 = zp_res_85722;
                
                r_85718 = r_tmp_104948;
            }
            defunc_0_lifted_lambda_res_85716 = r_85718;
            ((double *) mem_102992)[i_101745] = defunc_0_lifted_lambda_res_85716;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102987, i_101749 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102992, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103003_cached_sizze_105409 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_103003, &mem_103003_cached_sizze_105409, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103008_cached_sizze_105410 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_103008, &mem_103008_cached_sizze_105410, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101757 = 0; i_101757 < (int64_t) 16; i_101757++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101753 = 0; i_101753 < (int64_t) 64; i_101753++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_85737 = ((double *) mem_102987)[i_101757 * (int64_t) 64 + i_101753];
            
            // futhark/microgpt.fut:177:42-66
            
            double max_res_85738 = fmax64(0.0, max_arg0_85737);
            
            ((double *) mem_103008)[i_101753] = max_res_85738;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_103003, i_101757 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103008, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103019_cached_sizze_105411 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103019, &mem_103019_cached_sizze_105411, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103024_cached_sizze_105412 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103024, &mem_103024_cached_sizze_105412, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101765 = 0; i_101765 < (int64_t) 16; i_101765++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101761 = 0; i_101761 < (int64_t) 16; i_101761++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_85753;
            double r_85755 = 0.0;
            
            for (int64_t i_85754 = 0; i_85754 < (int64_t) 64; i_85754++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_85756 = ((double *) wdown_mem_102609.mem)[i_101761 * (int64_t) 64 + i_85754];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_85757 = ((double *) mem_103003)[i_101765 * (int64_t) 64 + i_85754];
                
                // futhark/microgpt.fut:178:73-106
                
                double zt_res_85758 = zt_lhs_85756 * zt_rhs_85757;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_85759 = r_85755 + zt_res_85758;
                double r_tmp_104953 = zp_res_85759;
                
                r_85755 = r_tmp_104953;
            }
            defunc_0_lifted_lambda_res_85753 = r_85755;
            ((double *) mem_103024)[i_101761] = defunc_0_lifted_lambda_res_85753;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_103019, i_101765 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103024, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103035_cached_sizze_105413 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103035, &mem_103035_cached_sizze_105413, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103040_cached_sizze_105414 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103040, &mem_103040_cached_sizze_105414, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101773 = 0; i_101773 < (int64_t) 16; i_101773++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101769 = 0; i_101769 < (int64_t) 16; i_101769++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_85774 = ((double *) mem_103019)[i_101773 * (int64_t) 16 + i_101769];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_85775 = ((double *) mem_102948)[i_101773 * (int64_t) 16 + i_101769];
            
            // futhark/microgpt.fut:179:42-73
            
            double zp_res_85776 = zp_lhs_85774 + zp_rhs_85775;
            
            ((double *) mem_103040)[i_101769] = zp_res_85776;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_103035, i_101773 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103040, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_103051, (int64_t) 3456, "mem_103051")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103056_cached_sizze_105415 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_103056, &mem_103056_cached_sizze_105415, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_101781 = 0; i_101781 < (int64_t) 16; i_101781++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101777 = 0; i_101777 < (int64_t) 27; i_101777++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_85792;
            double r_85794 = 0.0;
            
            for (int64_t i_85793 = 0; i_85793 < (int64_t) 16; i_85793++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_85795 = ((double *) wvoc_mem_102617.mem)[i_101777 * (int64_t) 16 + i_85793];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_85796 = ((double *) mem_103035)[i_101781 * (int64_t) 16 + i_85793];
                
                // futhark/microgpt.fut:180:62-94
                
                double zt_res_85797 = zt_lhs_85795 * zt_rhs_85796;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_85798 = r_85794 + zt_res_85797;
                double r_tmp_104958 = zp_res_85798;
                
                r_85794 = r_tmp_104958;
            }
            defunc_0_lifted_lambda_res_85792 = r_85794;
            ((double *) mem_103056)[i_101777] = defunc_0_lifted_lambda_res_85792;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_103051.mem, i_101781 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103056, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_104889, &mem_103051, "mem_103051") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105361, &mem_out_104889, "mem_out_104889") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_102620);
        free(mem_102625);
        free(mem_102636);
        free(mem_102641);
        free(mem_102652);
        free(mem_102657);
        free(mem_102664);
        free(mem_102675);
        free(mem_102680);
        free(mem_102687);
        free(mem_102698);
        free(mem_102699);
        free(mem_102700);
        free(mem_102713);
        free(mem_102714);
        free(mem_102715);
        free(mem_102746);
        free(mem_102747);
        free(mem_102748);
        free(mem_102764);
        free(mem_102765);
        free(mem_102766);
        free(mem_102779);
        free(mem_102780);
        free(mem_102781);
        free(mem_102827);
        free(mem_102833);
        free(mem_102838);
        free(mem_102849);
        free(mem_102854);
        free(mem_102865);
        free(mem_102870);
        free(mem_102877);
        free(mem_102884);
        free(mem_102895);
        free(mem_102900);
        free(mem_102916);
        free(mem_102921);
        free(mem_102932);
        free(mem_102937);
        free(mem_102948);
        free(mem_102953);
        free(mem_102964);
        free(mem_102969);
        free(mem_102976);
        free(mem_102987);
        free(mem_102992);
        free(mem_103003);
        free(mem_103008);
        free(mem_103019);
        free(mem_103024);
        free(mem_103035);
        free(mem_103040);
        free(mem_103056);
        if (memblock_unref(ctx, &mem_103051, "mem_103051") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104889, "mem_out_104889") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_105416, struct memblock *mem_out_p_105417, struct memblock *mem_out_p_105418, struct memblock *mem_out_p_105419, struct memblock *mem_out_p_105420, struct memblock *mem_out_p_105421, struct memblock *mem_out_p_105422, struct memblock *mem_out_p_105423, struct memblock *mem_out_p_105424, struct memblock wte_mem_102609, struct memblock wpe_mem_102610, struct memblock wqry_mem_102611, struct memblock wkey_mem_102612, struct memblock wval_mem_102613, struct memblock wout_mem_102614, struct memblock wup_mem_102615, struct memblock wdown_mem_102616, struct memblock wvoc_mem_102617)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_104897;
    
    mem_out_104897.references = NULL;
    
    struct memblock mem_out_104896;
    
    mem_out_104896.references = NULL;
    
    struct memblock mem_out_104895;
    
    mem_out_104895.references = NULL;
    
    struct memblock mem_out_104894;
    
    mem_out_104894.references = NULL;
    
    struct memblock mem_out_104893;
    
    mem_out_104893.references = NULL;
    
    struct memblock mem_out_104892;
    
    mem_out_104892.references = NULL;
    
    struct memblock mem_out_104891;
    
    mem_out_104891.references = NULL;
    
    struct memblock mem_out_104890;
    
    mem_out_104890.references = NULL;
    
    struct memblock mem_out_104889;
    
    mem_out_104889.references = NULL;
    
    struct memblock mem_102600 = ctx->constants->mem_102600;
    struct memblock mem_102601 = ctx->constants->mem_102601;
    struct memblock mem_102602 = ctx->constants->mem_102602;
    struct memblock mem_102603 = ctx->constants->mem_102603;
    struct memblock mem_102604 = ctx->constants->mem_102604;
    struct memblock mem_102605 = ctx->constants->mem_102605;
    struct memblock mem_102606 = ctx->constants->mem_102606;
    struct memblock mem_102607 = ctx->constants->mem_102607;
    struct memblock mem_102608 = ctx->constants->mem_102608;
    
    if (memblock_set(ctx, &mem_out_104889, &wdown_mem_102616, "wdown_mem_102616") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104890, &wkey_mem_102612, "wkey_mem_102612") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104891, &wout_mem_102614, "wout_mem_102614") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104892, &wpe_mem_102610, "wpe_mem_102610") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104893, &wqry_mem_102611, "wqry_mem_102611") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104894, &wte_mem_102609, "wte_mem_102609") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104895, &wup_mem_102615, "wup_mem_102615") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104896, &wval_mem_102613, "wval_mem_102613") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104897, &wvoc_mem_102617, "wvoc_mem_102617") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105416, &mem_out_104889, "mem_out_104889") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105417, &mem_out_104890, "mem_out_104890") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105418, &mem_out_104891, "mem_out_104891") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105419, &mem_out_104892, "mem_out_104892") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105420, &mem_out_104893, "mem_out_104893") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105421, &mem_out_104894, "mem_out_104894") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105422, &mem_out_104895, "mem_out_104895") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105423, &mem_out_104896, "mem_out_104896") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105424, &mem_out_104897, "mem_out_104897") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_104897, "mem_out_104897") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104896, "mem_out_104896") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104895, "mem_out_104895") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104894, "mem_out_104894") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104893, "mem_out_104893") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104892, "mem_out_104892") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104891, "mem_out_104891") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104890, "mem_out_104890") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104889, "mem_out_104889") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_105425, struct memblock *mem_out_p_105426, struct memblock *mem_out_p_105427, struct memblock *mem_out_p_105428, struct memblock *mem_out_p_105429, struct memblock *mem_out_p_105430, struct memblock *mem_out_p_105431, struct memblock *mem_out_p_105432, struct memblock *mem_out_p_105433, struct memblock *mem_out_p_105434, struct memblock *mem_out_p_105435, struct memblock *mem_out_p_105436, struct memblock *mem_out_p_105437, struct memblock *mem_out_p_105438, struct memblock *mem_out_p_105439, struct memblock *mem_out_p_105440, struct memblock *mem_out_p_105441, struct memblock *mem_out_p_105442, struct memblock *mem_out_p_105443, struct memblock *mem_out_p_105444, struct memblock *mem_out_p_105445, struct memblock *mem_out_p_105446, struct memblock *mem_out_p_105447, struct memblock *mem_out_p_105448, struct memblock *mem_out_p_105449, struct memblock *mem_out_p_105450, struct memblock *mem_out_p_105451, struct memblock *mem_out_p_105452, struct memblock wdown_mem_102609, struct memblock wkey_mem_102610, struct memblock wout_mem_102611, struct memblock wpe_mem_102612, struct memblock wqry_mem_102613, struct memblock wte_mem_102614, struct memblock wup_mem_102615, struct memblock wval_mem_102616, struct memblock wvoc_mem_102617, struct memblock wdown_mem_102618, struct memblock wkey_mem_102619, struct memblock wout_mem_102620, struct memblock wpe_mem_102621, struct memblock wqry_mem_102622, struct memblock wte_mem_102623, struct memblock wup_mem_102624, struct memblock wval_mem_102625, struct memblock wvoc_mem_102626, struct memblock wdown_mem_102627, struct memblock wkey_mem_102628, struct memblock wout_mem_102629, struct memblock wpe_mem_102630, struct memblock wqry_mem_102631, struct memblock wte_mem_102632, struct memblock wup_mem_102633, struct memblock wval_mem_102634, struct memblock wvoc_mem_102635, struct memblock masks_mem_102636, struct memblock dls_mem_102637, struct memblock seqs_mem_102638, int64_t n_74237)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_102755_cached_sizze_105453 = 0;
    unsigned char *mem_102755 = NULL;
    int64_t mem_102756_cached_sizze_105454 = 0;
    unsigned char *mem_102756 = NULL;
    int64_t mem_102757_cached_sizze_105455 = 0;
    unsigned char *mem_102757 = NULL;
    int64_t mem_102770_cached_sizze_105456 = 0;
    unsigned char *mem_102770 = NULL;
    int64_t mem_102771_cached_sizze_105457 = 0;
    unsigned char *mem_102771 = NULL;
    int64_t mem_102784_cached_sizze_105458 = 0;
    unsigned char *mem_102784 = NULL;
    int64_t mem_102803_cached_sizze_105459 = 0;
    unsigned char *mem_102803 = NULL;
    int64_t mem_102804_cached_sizze_105460 = 0;
    unsigned char *mem_102804 = NULL;
    int64_t mem_102813_cached_sizze_105461 = 0;
    unsigned char *mem_102813 = NULL;
    int64_t mem_102814_cached_sizze_105462 = 0;
    unsigned char *mem_102814 = NULL;
    int64_t mem_102835_cached_sizze_105463 = 0;
    unsigned char *mem_102835 = NULL;
    int64_t mem_102836_cached_sizze_105464 = 0;
    unsigned char *mem_102836 = NULL;
    int64_t mem_102837_cached_sizze_105465 = 0;
    unsigned char *mem_102837 = NULL;
    int64_t mem_102850_cached_sizze_105466 = 0;
    unsigned char *mem_102850 = NULL;
    int64_t mem_102851_cached_sizze_105467 = 0;
    unsigned char *mem_102851 = NULL;
    int64_t mem_102852_cached_sizze_105468 = 0;
    unsigned char *mem_102852 = NULL;
    int64_t mem_102871_cached_sizze_105469 = 0;
    unsigned char *mem_102871 = NULL;
    int64_t mem_102890_cached_sizze_105470 = 0;
    unsigned char *mem_102890 = NULL;
    int64_t mem_102891_cached_sizze_105471 = 0;
    unsigned char *mem_102891 = NULL;
    int64_t mem_102892_cached_sizze_105472 = 0;
    unsigned char *mem_102892 = NULL;
    int64_t mem_102893_cached_sizze_105473 = 0;
    unsigned char *mem_102893 = NULL;
    int64_t mem_102909_cached_sizze_105474 = 0;
    unsigned char *mem_102909 = NULL;
    int64_t mem_102910_cached_sizze_105475 = 0;
    unsigned char *mem_102910 = NULL;
    int64_t mem_102911_cached_sizze_105476 = 0;
    unsigned char *mem_102911 = NULL;
    int64_t mem_102930_cached_sizze_105477 = 0;
    unsigned char *mem_102930 = NULL;
    int64_t mem_102952_cached_sizze_105478 = 0;
    unsigned char *mem_102952 = NULL;
    int64_t mem_102953_cached_sizze_105479 = 0;
    unsigned char *mem_102953 = NULL;
    int64_t mem_102954_cached_sizze_105480 = 0;
    unsigned char *mem_102954 = NULL;
    int64_t mem_102955_cached_sizze_105481 = 0;
    unsigned char *mem_102955 = NULL;
    int64_t mem_102956_cached_sizze_105482 = 0;
    unsigned char *mem_102956 = NULL;
    int64_t mem_102957_cached_sizze_105483 = 0;
    unsigned char *mem_102957 = NULL;
    int64_t mem_102958_cached_sizze_105484 = 0;
    unsigned char *mem_102958 = NULL;
    int64_t mem_102959_cached_sizze_105485 = 0;
    unsigned char *mem_102959 = NULL;
    int64_t mem_102990_cached_sizze_105486 = 0;
    unsigned char *mem_102990 = NULL;
    int64_t mem_102991_cached_sizze_105487 = 0;
    unsigned char *mem_102991 = NULL;
    int64_t mem_102992_cached_sizze_105488 = 0;
    unsigned char *mem_102992 = NULL;
    int64_t mem_102993_cached_sizze_105489 = 0;
    unsigned char *mem_102993 = NULL;
    int64_t mem_102994_cached_sizze_105490 = 0;
    unsigned char *mem_102994 = NULL;
    int64_t mem_102995_cached_sizze_105491 = 0;
    unsigned char *mem_102995 = NULL;
    int64_t mem_103062_cached_sizze_105492 = 0;
    unsigned char *mem_103062 = NULL;
    int64_t mem_103063_cached_sizze_105493 = 0;
    unsigned char *mem_103063 = NULL;
    int64_t mem_103064_cached_sizze_105494 = 0;
    unsigned char *mem_103064 = NULL;
    int64_t mem_103065_cached_sizze_105495 = 0;
    unsigned char *mem_103065 = NULL;
    int64_t mem_103066_cached_sizze_105496 = 0;
    unsigned char *mem_103066 = NULL;
    int64_t mem_103067_cached_sizze_105497 = 0;
    unsigned char *mem_103067 = NULL;
    int64_t mem_103098_cached_sizze_105498 = 0;
    unsigned char *mem_103098 = NULL;
    int64_t mem_103099_cached_sizze_105499 = 0;
    unsigned char *mem_103099 = NULL;
    int64_t mem_103100_cached_sizze_105500 = 0;
    unsigned char *mem_103100 = NULL;
    int64_t mem_103101_cached_sizze_105501 = 0;
    unsigned char *mem_103101 = NULL;
    int64_t mem_103102_cached_sizze_105502 = 0;
    unsigned char *mem_103102 = NULL;
    int64_t mem_103103_cached_sizze_105503 = 0;
    unsigned char *mem_103103 = NULL;
    int64_t mem_103128_cached_sizze_105504 = 0;
    unsigned char *mem_103128 = NULL;
    int64_t mem_103129_cached_sizze_105505 = 0;
    unsigned char *mem_103129 = NULL;
    int64_t mem_103130_cached_sizze_105506 = 0;
    unsigned char *mem_103130 = NULL;
    int64_t mem_103131_cached_sizze_105507 = 0;
    unsigned char *mem_103131 = NULL;
    int64_t mem_103132_cached_sizze_105508 = 0;
    unsigned char *mem_103132 = NULL;
    int64_t mem_103133_cached_sizze_105509 = 0;
    unsigned char *mem_103133 = NULL;
    int64_t mem_103224_cached_sizze_105510 = 0;
    unsigned char *mem_103224 = NULL;
    int64_t mem_103225_cached_sizze_105511 = 0;
    unsigned char *mem_103225 = NULL;
    int64_t mem_103226_cached_sizze_105512 = 0;
    unsigned char *mem_103226 = NULL;
    int64_t mem_103242_cached_sizze_105513 = 0;
    unsigned char *mem_103242 = NULL;
    int64_t mem_103243_cached_sizze_105514 = 0;
    unsigned char *mem_103243 = NULL;
    int64_t mem_103244_cached_sizze_105515 = 0;
    unsigned char *mem_103244 = NULL;
    int64_t mem_103257_cached_sizze_105516 = 0;
    unsigned char *mem_103257 = NULL;
    int64_t mem_103258_cached_sizze_105517 = 0;
    unsigned char *mem_103258 = NULL;
    int64_t mem_103259_cached_sizze_105518 = 0;
    unsigned char *mem_103259 = NULL;
    int64_t mem_103290_cached_sizze_105519 = 0;
    unsigned char *mem_103290 = NULL;
    int64_t mem_103291_cached_sizze_105520 = 0;
    unsigned char *mem_103291 = NULL;
    int64_t mem_103300_cached_sizze_105521 = 0;
    unsigned char *mem_103300 = NULL;
    int64_t mem_103301_cached_sizze_105522 = 0;
    unsigned char *mem_103301 = NULL;
    int64_t mem_103322_cached_sizze_105523 = 0;
    unsigned char *mem_103322 = NULL;
    int64_t mem_103323_cached_sizze_105524 = 0;
    unsigned char *mem_103323 = NULL;
    int64_t mem_103332_cached_sizze_105525 = 0;
    unsigned char *mem_103332 = NULL;
    int64_t mem_103333_cached_sizze_105526 = 0;
    unsigned char *mem_103333 = NULL;
    int64_t mem_103346_cached_sizze_105527 = 0;
    unsigned char *mem_103346 = NULL;
    int64_t mem_103347_cached_sizze_105528 = 0;
    unsigned char *mem_103347 = NULL;
    int64_t mem_103360_cached_sizze_105529 = 0;
    unsigned char *mem_103360 = NULL;
    int64_t mem_103361_cached_sizze_105530 = 0;
    unsigned char *mem_103361 = NULL;
    int64_t mem_103382_cached_sizze_105531 = 0;
    unsigned char *mem_103382 = NULL;
    int64_t mem_103383_cached_sizze_105532 = 0;
    unsigned char *mem_103383 = NULL;
    int64_t mem_103392_cached_sizze_105533 = 0;
    unsigned char *mem_103392 = NULL;
    int64_t mem_103393_cached_sizze_105534 = 0;
    unsigned char *mem_103393 = NULL;
    int64_t mem_103429_cached_sizze_105535 = 0;
    unsigned char *mem_103429 = NULL;
    int64_t mem_103430_cached_sizze_105536 = 0;
    unsigned char *mem_103430 = NULL;
    int64_t mem_103431_cached_sizze_105537 = 0;
    unsigned char *mem_103431 = NULL;
    int64_t mem_103443_cached_sizze_105538 = 0;
    unsigned char *mem_103443 = NULL;
    int64_t mem_103444_cached_sizze_105539 = 0;
    unsigned char *mem_103444 = NULL;
    int64_t mem_103468_cached_sizze_105540 = 0;
    unsigned char *mem_103468 = NULL;
    int64_t mem_103469_cached_sizze_105541 = 0;
    unsigned char *mem_103469 = NULL;
    int64_t mem_103478_cached_sizze_105542 = 0;
    unsigned char *mem_103478 = NULL;
    int64_t mem_103479_cached_sizze_105543 = 0;
    unsigned char *mem_103479 = NULL;
    int64_t mem_103500_cached_sizze_105544 = 0;
    unsigned char *mem_103500 = NULL;
    int64_t mem_103501_cached_sizze_105545 = 0;
    unsigned char *mem_103501 = NULL;
    int64_t mem_103510_cached_sizze_105546 = 0;
    unsigned char *mem_103510 = NULL;
    int64_t mem_103511_cached_sizze_105547 = 0;
    unsigned char *mem_103511 = NULL;
    int64_t mem_103532_cached_sizze_105548 = 0;
    unsigned char *mem_103532 = NULL;
    int64_t mem_103533_cached_sizze_105549 = 0;
    unsigned char *mem_103533 = NULL;
    int64_t mem_103534_cached_sizze_105550 = 0;
    unsigned char *mem_103534 = NULL;
    int64_t mem_103547_cached_sizze_105551 = 0;
    unsigned char *mem_103547 = NULL;
    int64_t mem_103548_cached_sizze_105552 = 0;
    unsigned char *mem_103548 = NULL;
    int64_t mem_103549_cached_sizze_105553 = 0;
    unsigned char *mem_103549 = NULL;
    int64_t mem_103568_cached_sizze_105554 = 0;
    unsigned char *mem_103568 = NULL;
    int64_t mem_103587_cached_sizze_105555 = 0;
    unsigned char *mem_103587 = NULL;
    int64_t mem_103588_cached_sizze_105556 = 0;
    unsigned char *mem_103588 = NULL;
    int64_t mem_103589_cached_sizze_105557 = 0;
    unsigned char *mem_103589 = NULL;
    int64_t mem_103601_cached_sizze_105558 = 0;
    unsigned char *mem_103601 = NULL;
    int64_t mem_103602_cached_sizze_105559 = 0;
    unsigned char *mem_103602 = NULL;
    int64_t mem_103626_cached_sizze_105560 = 0;
    unsigned char *mem_103626 = NULL;
    int64_t mem_103627_cached_sizze_105561 = 0;
    unsigned char *mem_103627 = NULL;
    int64_t mem_103628_cached_sizze_105562 = 0;
    unsigned char *mem_103628 = NULL;
    int64_t mem_103640_cached_sizze_105563 = 0;
    unsigned char *mem_103640 = NULL;
    int64_t mem_103641_cached_sizze_105564 = 0;
    unsigned char *mem_103641 = NULL;
    int64_t mem_103665_cached_sizze_105565 = 0;
    unsigned char *mem_103665 = NULL;
    int64_t mem_103666_cached_sizze_105566 = 0;
    unsigned char *mem_103666 = NULL;
    int64_t mem_103675_cached_sizze_105567 = 0;
    unsigned char *mem_103675 = NULL;
    int64_t mem_103676_cached_sizze_105568 = 0;
    unsigned char *mem_103676 = NULL;
    int64_t mem_103697_cached_sizze_105569 = 0;
    unsigned char *mem_103697 = NULL;
    int64_t mem_103698_cached_sizze_105570 = 0;
    unsigned char *mem_103698 = NULL;
    int64_t mem_103707_cached_sizze_105571 = 0;
    unsigned char *mem_103707 = NULL;
    int64_t mem_103708_cached_sizze_105572 = 0;
    unsigned char *mem_103708 = NULL;
    int64_t mem_103729_cached_sizze_105573 = 0;
    unsigned char *mem_103729 = NULL;
    int64_t mem_103730_cached_sizze_105574 = 0;
    unsigned char *mem_103730 = NULL;
    int64_t mem_103739_cached_sizze_105575 = 0;
    unsigned char *mem_103739 = NULL;
    int64_t mem_103740_cached_sizze_105576 = 0;
    unsigned char *mem_103740 = NULL;
    int64_t mem_103761_cached_sizze_105577 = 0;
    unsigned char *mem_103761 = NULL;
    int64_t mem_103762_cached_sizze_105578 = 0;
    unsigned char *mem_103762 = NULL;
    int64_t mem_103763_cached_sizze_105579 = 0;
    unsigned char *mem_103763 = NULL;
    int64_t mem_103775_cached_sizze_105580 = 0;
    unsigned char *mem_103775 = NULL;
    int64_t mem_103776_cached_sizze_105581 = 0;
    unsigned char *mem_103776 = NULL;
    int64_t mem_103777_cached_sizze_105582 = 0;
    unsigned char *mem_103777 = NULL;
    int64_t mem_103796_cached_sizze_105583 = 0;
    unsigned char *mem_103796 = NULL;
    int64_t mem_103797_cached_sizze_105584 = 0;
    unsigned char *mem_103797 = NULL;
    int64_t mem_103798_cached_sizze_105585 = 0;
    unsigned char *mem_103798 = NULL;
    int64_t mem_103817_cached_sizze_105586 = 0;
    unsigned char *mem_103817 = NULL;
    int64_t mem_103818_cached_sizze_105587 = 0;
    unsigned char *mem_103818 = NULL;
    int64_t mem_103819_cached_sizze_105588 = 0;
    unsigned char *mem_103819 = NULL;
    int64_t mem_103849_cached_sizze_105589 = 0;
    unsigned char *mem_103849 = NULL;
    int64_t mem_103856_cached_sizze_105590 = 0;
    unsigned char *mem_103856 = NULL;
    int64_t mem_103861_cached_sizze_105591 = 0;
    unsigned char *mem_103861 = NULL;
    int64_t mem_103872_cached_sizze_105592 = 0;
    unsigned char *mem_103872 = NULL;
    int64_t mem_103877_cached_sizze_105593 = 0;
    unsigned char *mem_103877 = NULL;
    int64_t mem_103888_cached_sizze_105594 = 0;
    unsigned char *mem_103888 = NULL;
    int64_t mem_103889_cached_sizze_105595 = 0;
    unsigned char *mem_103889 = NULL;
    int64_t mem_103898_cached_sizze_105596 = 0;
    unsigned char *mem_103898 = NULL;
    int64_t mem_103899_cached_sizze_105597 = 0;
    unsigned char *mem_103899 = NULL;
    int64_t mem_103920_cached_sizze_105598 = 0;
    unsigned char *mem_103920 = NULL;
    int64_t mem_103925_cached_sizze_105599 = 0;
    unsigned char *mem_103925 = NULL;
    int64_t mem_103936_cached_sizze_105600 = 0;
    unsigned char *mem_103936 = NULL;
    int64_t mem_103941_cached_sizze_105601 = 0;
    unsigned char *mem_103941 = NULL;
    int64_t mem_103952_cached_sizze_105602 = 0;
    unsigned char *mem_103952 = NULL;
    int64_t mem_103959_cached_sizze_105603 = 0;
    unsigned char *mem_103959 = NULL;
    int64_t mem_103966_cached_sizze_105604 = 0;
    unsigned char *mem_103966 = NULL;
    int64_t mem_103976_cached_sizze_105605 = 0;
    unsigned char *mem_103976 = NULL;
    int64_t mem_103981_cached_sizze_105606 = 0;
    unsigned char *mem_103981 = NULL;
    int64_t mem_103992_cached_sizze_105607 = 0;
    unsigned char *mem_103992 = NULL;
    int64_t mem_103993_cached_sizze_105608 = 0;
    unsigned char *mem_103993 = NULL;
    int64_t mem_104002_cached_sizze_105609 = 0;
    unsigned char *mem_104002 = NULL;
    int64_t mem_104003_cached_sizze_105610 = 0;
    unsigned char *mem_104003 = NULL;
    int64_t mem_104024_cached_sizze_105611 = 0;
    unsigned char *mem_104024 = NULL;
    int64_t mem_104025_cached_sizze_105612 = 0;
    unsigned char *mem_104025 = NULL;
    int64_t mem_104036_cached_sizze_105613 = 0;
    unsigned char *mem_104036 = NULL;
    int64_t mem_104037_cached_sizze_105614 = 0;
    unsigned char *mem_104037 = NULL;
    int64_t mem_104046_cached_sizze_105615 = 0;
    unsigned char *mem_104046 = NULL;
    int64_t mem_104053_cached_sizze_105616 = 0;
    unsigned char *mem_104053 = NULL;
    int64_t mem_104078_cached_sizze_105617 = 0;
    unsigned char *mem_104078 = NULL;
    int64_t mem_104079_cached_sizze_105618 = 0;
    unsigned char *mem_104079 = NULL;
    int64_t mem_104090_cached_sizze_105619 = 0;
    unsigned char *mem_104090 = NULL;
    int64_t mem_104091_cached_sizze_105620 = 0;
    unsigned char *mem_104091 = NULL;
    int64_t mem_104100_cached_sizze_105621 = 0;
    unsigned char *mem_104100 = NULL;
    int64_t mem_104107_cached_sizze_105622 = 0;
    unsigned char *mem_104107 = NULL;
    int64_t mem_104114_cached_sizze_105623 = 0;
    unsigned char *mem_104114 = NULL;
    int64_t mem_104121_cached_sizze_105624 = 0;
    unsigned char *mem_104121 = NULL;
    int64_t mem_104146_cached_sizze_105625 = 0;
    unsigned char *mem_104146 = NULL;
    int64_t mem_104147_cached_sizze_105626 = 0;
    unsigned char *mem_104147 = NULL;
    int64_t mem_104158_cached_sizze_105627 = 0;
    unsigned char *mem_104158 = NULL;
    int64_t mem_104159_cached_sizze_105628 = 0;
    unsigned char *mem_104159 = NULL;
    int64_t mem_104168_cached_sizze_105629 = 0;
    unsigned char *mem_104168 = NULL;
    int64_t mem_104175_cached_sizze_105630 = 0;
    unsigned char *mem_104175 = NULL;
    int64_t mem_104200_cached_sizze_105631 = 0;
    unsigned char *mem_104200 = NULL;
    int64_t mem_104205_cached_sizze_105632 = 0;
    unsigned char *mem_104205 = NULL;
    int64_t mem_104216_cached_sizze_105633 = 0;
    unsigned char *mem_104216 = NULL;
    int64_t mem_104222_cached_sizze_105634 = 0;
    unsigned char *mem_104222 = NULL;
    int64_t mem_104227_cached_sizze_105635 = 0;
    unsigned char *mem_104227 = NULL;
    int64_t mem_104243_cached_sizze_105636 = 0;
    unsigned char *mem_104243 = NULL;
    int64_t mem_104249_cached_sizze_105637 = 0;
    unsigned char *mem_104249 = NULL;
    int64_t mem_104254_cached_sizze_105638 = 0;
    unsigned char *mem_104254 = NULL;
    int64_t mem_104270_cached_sizze_105639 = 0;
    unsigned char *mem_104270 = NULL;
    int64_t mem_104271_cached_sizze_105640 = 0;
    unsigned char *mem_104271 = NULL;
    int64_t mem_104282_cached_sizze_105641 = 0;
    unsigned char *mem_104282 = NULL;
    int64_t mem_104283_cached_sizze_105642 = 0;
    unsigned char *mem_104283 = NULL;
    int64_t mem_104292_cached_sizze_105643 = 0;
    unsigned char *mem_104292 = NULL;
    int64_t mem_104293_cached_sizze_105644 = 0;
    unsigned char *mem_104293 = NULL;
    int64_t mem_104324_cached_sizze_105645 = 0;
    unsigned char *mem_104324 = NULL;
    int64_t mem_104325_cached_sizze_105646 = 0;
    unsigned char *mem_104325 = NULL;
    int64_t mem_104326_cached_sizze_105647 = 0;
    unsigned char *mem_104326 = NULL;
    int64_t mem_104339_cached_sizze_105648 = 0;
    unsigned char *mem_104339 = NULL;
    int64_t mem_104340_cached_sizze_105649 = 0;
    unsigned char *mem_104340 = NULL;
    int64_t mem_104341_cached_sizze_105650 = 0;
    unsigned char *mem_104341 = NULL;
    int64_t mem_104372_cached_sizze_105651 = 0;
    unsigned char *mem_104372 = NULL;
    int64_t mem_104373_cached_sizze_105652 = 0;
    unsigned char *mem_104373 = NULL;
    int64_t mem_104374_cached_sizze_105653 = 0;
    unsigned char *mem_104374 = NULL;
    int64_t mem_104375_cached_sizze_105654 = 0;
    unsigned char *mem_104375 = NULL;
    int64_t mem_104392_cached_sizze_105655 = 0;
    unsigned char *mem_104392 = NULL;
    int64_t mem_104393_cached_sizze_105656 = 0;
    unsigned char *mem_104393 = NULL;
    int64_t mem_104394_cached_sizze_105657 = 0;
    unsigned char *mem_104394 = NULL;
    int64_t mem_104395_cached_sizze_105658 = 0;
    unsigned char *mem_104395 = NULL;
    int64_t mem_104436_cached_sizze_105659 = 0;
    unsigned char *mem_104436 = NULL;
    int64_t mem_104443_cached_sizze_105660 = 0;
    unsigned char *mem_104443 = NULL;
    int64_t mem_104450_cached_sizze_105661 = 0;
    unsigned char *mem_104450 = NULL;
    int64_t mem_104460_cached_sizze_105662 = 0;
    unsigned char *mem_104460 = NULL;
    int64_t mem_104465_cached_sizze_105663 = 0;
    unsigned char *mem_104465 = NULL;
    int64_t mem_104476_cached_sizze_105664 = 0;
    unsigned char *mem_104476 = NULL;
    int64_t mem_104483_cached_sizze_105665 = 0;
    unsigned char *mem_104483 = NULL;
    int64_t mem_104490_cached_sizze_105666 = 0;
    unsigned char *mem_104490 = NULL;
    int64_t mem_104500_cached_sizze_105667 = 0;
    unsigned char *mem_104500 = NULL;
    int64_t mem_104505_cached_sizze_105668 = 0;
    unsigned char *mem_104505 = NULL;
    int64_t mem_104516_cached_sizze_105669 = 0;
    unsigned char *mem_104516 = NULL;
    int64_t mem_104517_cached_sizze_105670 = 0;
    unsigned char *mem_104517 = NULL;
    int64_t mem_104526_cached_sizze_105671 = 0;
    unsigned char *mem_104526 = NULL;
    int64_t mem_104527_cached_sizze_105672 = 0;
    unsigned char *mem_104527 = NULL;
    int64_t mem_104548_cached_sizze_105673 = 0;
    unsigned char *mem_104548 = NULL;
    int64_t mem_104553_cached_sizze_105674 = 0;
    unsigned char *mem_104553 = NULL;
    int64_t mem_104564_cached_sizze_105675 = 0;
    unsigned char *mem_104564 = NULL;
    int64_t mem_104565_cached_sizze_105676 = 0;
    unsigned char *mem_104565 = NULL;
    int64_t mem_104574_cached_sizze_105677 = 0;
    unsigned char *mem_104574 = NULL;
    int64_t mem_104575_cached_sizze_105678 = 0;
    unsigned char *mem_104575 = NULL;
    struct memblock mem_param_tmp_104945;
    
    mem_param_tmp_104945.references = NULL;
    
    struct memblock mem_param_tmp_104944;
    
    mem_param_tmp_104944.references = NULL;
    
    struct memblock mem_param_tmp_104943;
    
    mem_param_tmp_104943.references = NULL;
    
    struct memblock mem_param_tmp_104942;
    
    mem_param_tmp_104942.references = NULL;
    
    struct memblock mem_param_tmp_104941;
    
    mem_param_tmp_104941.references = NULL;
    
    struct memblock mem_param_tmp_104940;
    
    mem_param_tmp_104940.references = NULL;
    
    struct memblock mem_param_tmp_104939;
    
    mem_param_tmp_104939.references = NULL;
    
    struct memblock mem_param_tmp_104938;
    
    mem_param_tmp_104938.references = NULL;
    
    struct memblock mem_param_tmp_104937;
    
    mem_param_tmp_104937.references = NULL;
    
    struct memblock mem_param_tmp_104936;
    
    mem_param_tmp_104936.references = NULL;
    
    struct memblock mem_param_tmp_104935;
    
    mem_param_tmp_104935.references = NULL;
    
    struct memblock mem_param_tmp_104934;
    
    mem_param_tmp_104934.references = NULL;
    
    struct memblock mem_param_tmp_104933;
    
    mem_param_tmp_104933.references = NULL;
    
    struct memblock mem_param_tmp_104932;
    
    mem_param_tmp_104932.references = NULL;
    
    struct memblock mem_param_tmp_104931;
    
    mem_param_tmp_104931.references = NULL;
    
    struct memblock mem_param_tmp_104930;
    
    mem_param_tmp_104930.references = NULL;
    
    struct memblock mem_param_tmp_104929;
    
    mem_param_tmp_104929.references = NULL;
    
    struct memblock mem_param_tmp_104928;
    
    mem_param_tmp_104928.references = NULL;
    
    struct memblock mem_param_tmp_104927;
    
    mem_param_tmp_104927.references = NULL;
    
    struct memblock mem_param_tmp_104926;
    
    mem_param_tmp_104926.references = NULL;
    
    struct memblock mem_param_tmp_104925;
    
    mem_param_tmp_104925.references = NULL;
    
    struct memblock mem_param_tmp_104924;
    
    mem_param_tmp_104924.references = NULL;
    
    struct memblock mem_param_tmp_104923;
    
    mem_param_tmp_104923.references = NULL;
    
    struct memblock mem_param_tmp_104922;
    
    mem_param_tmp_104922.references = NULL;
    
    struct memblock mem_param_tmp_104921;
    
    mem_param_tmp_104921.references = NULL;
    
    struct memblock mem_param_tmp_104920;
    
    mem_param_tmp_104920.references = NULL;
    
    struct memblock mem_param_tmp_104919;
    
    mem_param_tmp_104919.references = NULL;
    
    struct memblock mem_param_tmp_104918;
    
    mem_param_tmp_104918.references = NULL;
    
    struct memblock mem_104696;
    
    mem_104696.references = NULL;
    
    struct memblock ext_mem_104692;
    
    ext_mem_104692.references = NULL;
    
    struct memblock ext_mem_104693;
    
    ext_mem_104693.references = NULL;
    
    struct memblock ext_mem_104694;
    
    ext_mem_104694.references = NULL;
    
    struct memblock mem_104690;
    
    mem_104690.references = NULL;
    
    struct memblock mem_104688;
    
    mem_104688.references = NULL;
    
    struct memblock mem_104686;
    
    mem_104686.references = NULL;
    
    struct memblock mem_104684;
    
    mem_104684.references = NULL;
    
    struct memblock ext_mem_104681;
    
    ext_mem_104681.references = NULL;
    
    struct memblock ext_mem_104682;
    
    ext_mem_104682.references = NULL;
    
    struct memblock ext_mem_104683;
    
    ext_mem_104683.references = NULL;
    
    struct memblock mem_104679;
    
    mem_104679.references = NULL;
    
    struct memblock mem_104677;
    
    mem_104677.references = NULL;
    
    struct memblock mem_104675;
    
    mem_104675.references = NULL;
    
    struct memblock mem_104673;
    
    mem_104673.references = NULL;
    
    struct memblock ext_mem_104670;
    
    ext_mem_104670.references = NULL;
    
    struct memblock ext_mem_104671;
    
    ext_mem_104671.references = NULL;
    
    struct memblock ext_mem_104672;
    
    ext_mem_104672.references = NULL;
    
    struct memblock mem_104668;
    
    mem_104668.references = NULL;
    
    struct memblock mem_104666;
    
    mem_104666.references = NULL;
    
    struct memblock mem_104664;
    
    mem_104664.references = NULL;
    
    struct memblock mem_104662;
    
    mem_104662.references = NULL;
    
    struct memblock ext_mem_104659;
    
    ext_mem_104659.references = NULL;
    
    struct memblock ext_mem_104660;
    
    ext_mem_104660.references = NULL;
    
    struct memblock ext_mem_104661;
    
    ext_mem_104661.references = NULL;
    
    struct memblock mem_104657;
    
    mem_104657.references = NULL;
    
    struct memblock mem_104655;
    
    mem_104655.references = NULL;
    
    struct memblock mem_104653;
    
    mem_104653.references = NULL;
    
    struct memblock mem_104651;
    
    mem_104651.references = NULL;
    
    struct memblock ext_mem_104648;
    
    ext_mem_104648.references = NULL;
    
    struct memblock ext_mem_104649;
    
    ext_mem_104649.references = NULL;
    
    struct memblock ext_mem_104650;
    
    ext_mem_104650.references = NULL;
    
    struct memblock mem_104646;
    
    mem_104646.references = NULL;
    
    struct memblock mem_104644;
    
    mem_104644.references = NULL;
    
    struct memblock mem_104642;
    
    mem_104642.references = NULL;
    
    struct memblock mem_104640;
    
    mem_104640.references = NULL;
    
    struct memblock ext_mem_104637;
    
    ext_mem_104637.references = NULL;
    
    struct memblock ext_mem_104638;
    
    ext_mem_104638.references = NULL;
    
    struct memblock ext_mem_104639;
    
    ext_mem_104639.references = NULL;
    
    struct memblock mem_104635;
    
    mem_104635.references = NULL;
    
    struct memblock mem_104633;
    
    mem_104633.references = NULL;
    
    struct memblock mem_104631;
    
    mem_104631.references = NULL;
    
    struct memblock mem_104629;
    
    mem_104629.references = NULL;
    
    struct memblock ext_mem_104626;
    
    ext_mem_104626.references = NULL;
    
    struct memblock ext_mem_104627;
    
    ext_mem_104627.references = NULL;
    
    struct memblock ext_mem_104628;
    
    ext_mem_104628.references = NULL;
    
    struct memblock mem_104624;
    
    mem_104624.references = NULL;
    
    struct memblock mem_104622;
    
    mem_104622.references = NULL;
    
    struct memblock mem_104620;
    
    mem_104620.references = NULL;
    
    struct memblock mem_104618;
    
    mem_104618.references = NULL;
    
    struct memblock ext_mem_104615;
    
    ext_mem_104615.references = NULL;
    
    struct memblock ext_mem_104616;
    
    ext_mem_104616.references = NULL;
    
    struct memblock ext_mem_104617;
    
    ext_mem_104617.references = NULL;
    
    struct memblock mem_104613;
    
    mem_104613.references = NULL;
    
    struct memblock mem_104611;
    
    mem_104611.references = NULL;
    
    struct memblock mem_104609;
    
    mem_104609.references = NULL;
    
    struct memblock mem_104607;
    
    mem_104607.references = NULL;
    
    struct memblock ext_mem_104604;
    
    ext_mem_104604.references = NULL;
    
    struct memblock ext_mem_104605;
    
    ext_mem_104605.references = NULL;
    
    struct memblock ext_mem_104606;
    
    ext_mem_104606.references = NULL;
    
    struct memblock mem_104602;
    
    mem_104602.references = NULL;
    
    struct memblock mem_104600;
    
    mem_104600.references = NULL;
    
    struct memblock mem_104598;
    
    mem_104598.references = NULL;
    
    struct memblock mem_104596;
    
    mem_104596.references = NULL;
    
    struct memblock ext_mem_102754;
    
    ext_mem_102754.references = NULL;
    
    struct memblock mem_param_102751;
    
    mem_param_102751.references = NULL;
    
    struct memblock mem_param_102748;
    
    mem_param_102748.references = NULL;
    
    struct memblock mem_param_102744;
    
    mem_param_102744.references = NULL;
    
    struct memblock mem_param_102740;
    
    mem_param_102740.references = NULL;
    
    struct memblock mem_param_102736;
    
    mem_param_102736.references = NULL;
    
    struct memblock mem_param_102732;
    
    mem_param_102732.references = NULL;
    
    struct memblock mem_param_102728;
    
    mem_param_102728.references = NULL;
    
    struct memblock mem_param_102724;
    
    mem_param_102724.references = NULL;
    
    struct memblock mem_param_102720;
    
    mem_param_102720.references = NULL;
    
    struct memblock mem_param_102716;
    
    mem_param_102716.references = NULL;
    
    struct memblock mem_param_102712;
    
    mem_param_102712.references = NULL;
    
    struct memblock mem_param_102708;
    
    mem_param_102708.references = NULL;
    
    struct memblock mem_param_102704;
    
    mem_param_102704.references = NULL;
    
    struct memblock mem_param_102700;
    
    mem_param_102700.references = NULL;
    
    struct memblock mem_param_102696;
    
    mem_param_102696.references = NULL;
    
    struct memblock mem_param_102692;
    
    mem_param_102692.references = NULL;
    
    struct memblock mem_param_102688;
    
    mem_param_102688.references = NULL;
    
    struct memblock mem_param_102684;
    
    mem_param_102684.references = NULL;
    
    struct memblock mem_param_102680;
    
    mem_param_102680.references = NULL;
    
    struct memblock mem_param_102676;
    
    mem_param_102676.references = NULL;
    
    struct memblock mem_param_102672;
    
    mem_param_102672.references = NULL;
    
    struct memblock mem_param_102668;
    
    mem_param_102668.references = NULL;
    
    struct memblock mem_param_102664;
    
    mem_param_102664.references = NULL;
    
    struct memblock mem_param_102660;
    
    mem_param_102660.references = NULL;
    
    struct memblock mem_param_102656;
    
    mem_param_102656.references = NULL;
    
    struct memblock mem_param_102652;
    
    mem_param_102652.references = NULL;
    
    struct memblock mem_param_102648;
    
    mem_param_102648.references = NULL;
    
    struct memblock mem_param_102644;
    
    mem_param_102644.references = NULL;
    
    struct memblock ext_mem_104780;
    
    ext_mem_104780.references = NULL;
    
    struct memblock ext_mem_104781;
    
    ext_mem_104781.references = NULL;
    
    struct memblock ext_mem_104782;
    
    ext_mem_104782.references = NULL;
    
    struct memblock ext_mem_104783;
    
    ext_mem_104783.references = NULL;
    
    struct memblock ext_mem_104784;
    
    ext_mem_104784.references = NULL;
    
    struct memblock ext_mem_104785;
    
    ext_mem_104785.references = NULL;
    
    struct memblock ext_mem_104786;
    
    ext_mem_104786.references = NULL;
    
    struct memblock ext_mem_104787;
    
    ext_mem_104787.references = NULL;
    
    struct memblock ext_mem_104788;
    
    ext_mem_104788.references = NULL;
    
    struct memblock ext_mem_104789;
    
    ext_mem_104789.references = NULL;
    
    struct memblock ext_mem_104790;
    
    ext_mem_104790.references = NULL;
    
    struct memblock ext_mem_104791;
    
    ext_mem_104791.references = NULL;
    
    struct memblock ext_mem_104792;
    
    ext_mem_104792.references = NULL;
    
    struct memblock ext_mem_104793;
    
    ext_mem_104793.references = NULL;
    
    struct memblock ext_mem_104794;
    
    ext_mem_104794.references = NULL;
    
    struct memblock ext_mem_104795;
    
    ext_mem_104795.references = NULL;
    
    struct memblock ext_mem_104796;
    
    ext_mem_104796.references = NULL;
    
    struct memblock ext_mem_104797;
    
    ext_mem_104797.references = NULL;
    
    struct memblock ext_mem_104798;
    
    ext_mem_104798.references = NULL;
    
    struct memblock ext_mem_104799;
    
    ext_mem_104799.references = NULL;
    
    struct memblock ext_mem_104800;
    
    ext_mem_104800.references = NULL;
    
    struct memblock ext_mem_104801;
    
    ext_mem_104801.references = NULL;
    
    struct memblock ext_mem_104802;
    
    ext_mem_104802.references = NULL;
    
    struct memblock ext_mem_104803;
    
    ext_mem_104803.references = NULL;
    
    struct memblock ext_mem_104804;
    
    ext_mem_104804.references = NULL;
    
    struct memblock ext_mem_104805;
    
    ext_mem_104805.references = NULL;
    
    struct memblock ext_mem_104806;
    
    ext_mem_104806.references = NULL;
    
    struct memblock ext_mem_104807;
    
    ext_mem_104807.references = NULL;
    
    struct memblock mem_102752;
    
    mem_102752.references = NULL;
    
    struct memblock mem_102640;
    
    mem_102640.references = NULL;
    
    struct memblock mem_out_104916;
    
    mem_out_104916.references = NULL;
    
    struct memblock mem_out_104915;
    
    mem_out_104915.references = NULL;
    
    struct memblock mem_out_104914;
    
    mem_out_104914.references = NULL;
    
    struct memblock mem_out_104913;
    
    mem_out_104913.references = NULL;
    
    struct memblock mem_out_104912;
    
    mem_out_104912.references = NULL;
    
    struct memblock mem_out_104911;
    
    mem_out_104911.references = NULL;
    
    struct memblock mem_out_104910;
    
    mem_out_104910.references = NULL;
    
    struct memblock mem_out_104909;
    
    mem_out_104909.references = NULL;
    
    struct memblock mem_out_104908;
    
    mem_out_104908.references = NULL;
    
    struct memblock mem_out_104907;
    
    mem_out_104907.references = NULL;
    
    struct memblock mem_out_104906;
    
    mem_out_104906.references = NULL;
    
    struct memblock mem_out_104905;
    
    mem_out_104905.references = NULL;
    
    struct memblock mem_out_104904;
    
    mem_out_104904.references = NULL;
    
    struct memblock mem_out_104903;
    
    mem_out_104903.references = NULL;
    
    struct memblock mem_out_104902;
    
    mem_out_104902.references = NULL;
    
    struct memblock mem_out_104901;
    
    mem_out_104901.references = NULL;
    
    struct memblock mem_out_104900;
    
    mem_out_104900.references = NULL;
    
    struct memblock mem_out_104899;
    
    mem_out_104899.references = NULL;
    
    struct memblock mem_out_104898;
    
    mem_out_104898.references = NULL;
    
    struct memblock mem_out_104897;
    
    mem_out_104897.references = NULL;
    
    struct memblock mem_out_104896;
    
    mem_out_104896.references = NULL;
    
    struct memblock mem_out_104895;
    
    mem_out_104895.references = NULL;
    
    struct memblock mem_out_104894;
    
    mem_out_104894.references = NULL;
    
    struct memblock mem_out_104893;
    
    mem_out_104893.references = NULL;
    
    struct memblock mem_out_104892;
    
    mem_out_104892.references = NULL;
    
    struct memblock mem_out_104891;
    
    mem_out_104891.references = NULL;
    
    struct memblock mem_out_104890;
    
    mem_out_104890.references = NULL;
    
    struct memblock mem_out_104889;
    
    mem_out_104889.references = NULL;
    
    struct memblock mem_102600 = ctx->constants->mem_102600;
    struct memblock mem_102601 = ctx->constants->mem_102601;
    struct memblock mem_102602 = ctx->constants->mem_102602;
    struct memblock mem_102603 = ctx->constants->mem_102603;
    struct memblock mem_102604 = ctx->constants->mem_102604;
    struct memblock mem_102605 = ctx->constants->mem_102605;
    struct memblock mem_102606 = ctx->constants->mem_102606;
    struct memblock mem_102607 = ctx->constants->mem_102607;
    struct memblock mem_102608 = ctx->constants->mem_102608;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_102639 = (int64_t) 8 * n_74237;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_102640, bytes_102639, "mem_102640")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_104917 = 0; nest_i_104917 < n_74237; nest_i_104917++) {
        ((double *) mem_102640.mem)[nest_i_104917] = 0.0;
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_90187 = sitofp_i64_f64(n_74237);
    
    // futhark/microgpt.fut:409:17-37
    if (memblock_alloc(ctx, &mem_102752, (int64_t) 128, "mem_102752")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102755_cached_sizze_105453 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_102755, &mem_102755_cached_sizze_105453, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102756_cached_sizze_105454 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102756, &mem_102756_cached_sizze_105454, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102757_cached_sizze_105455 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102757, &mem_102757_cached_sizze_105455, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102770_cached_sizze_105456 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102770, &mem_102770_cached_sizze_105456, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102771_cached_sizze_105457 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102771, &mem_102771_cached_sizze_105457, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102784_cached_sizze_105458 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_102784, &mem_102784_cached_sizze_105458, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102803_cached_sizze_105459 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102803, &mem_102803_cached_sizze_105459, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102804_cached_sizze_105460 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102804, &mem_102804_cached_sizze_105460, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102813_cached_sizze_105461 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102813, &mem_102813_cached_sizze_105461, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102814_cached_sizze_105462 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102814, &mem_102814_cached_sizze_105462, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102835_cached_sizze_105463 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102835, &mem_102835_cached_sizze_105463, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102836_cached_sizze_105464 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102836, &mem_102836_cached_sizze_105464, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102837_cached_sizze_105465 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102837, &mem_102837_cached_sizze_105465, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102850_cached_sizze_105466 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102850, &mem_102850_cached_sizze_105466, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102851_cached_sizze_105467 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102851, &mem_102851_cached_sizze_105467, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102852_cached_sizze_105468 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102852, &mem_102852_cached_sizze_105468, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102871_cached_sizze_105469 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102871, &mem_102871_cached_sizze_105469, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102890_cached_sizze_105470 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102890, &mem_102890_cached_sizze_105470, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102891_cached_sizze_105471 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102891, &mem_102891_cached_sizze_105471, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102892_cached_sizze_105472 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102892, &mem_102892_cached_sizze_105472, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102893_cached_sizze_105473 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102893, &mem_102893_cached_sizze_105473, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102909_cached_sizze_105474 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102909, &mem_102909_cached_sizze_105474, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102910_cached_sizze_105475 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102910, &mem_102910_cached_sizze_105475, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102911_cached_sizze_105476 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102911, &mem_102911_cached_sizze_105476, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102930_cached_sizze_105477 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102930, &mem_102930_cached_sizze_105477, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102952_cached_sizze_105478 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102952, &mem_102952_cached_sizze_105478, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102953_cached_sizze_105479 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102953, &mem_102953_cached_sizze_105479, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102954_cached_sizze_105480 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102954, &mem_102954_cached_sizze_105480, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102955_cached_sizze_105481 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102955, &mem_102955_cached_sizze_105481, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102956_cached_sizze_105482 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102956, &mem_102956_cached_sizze_105482, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102957_cached_sizze_105483 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102957, &mem_102957_cached_sizze_105483, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102958_cached_sizze_105484 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102958, &mem_102958_cached_sizze_105484, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102959_cached_sizze_105485 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_102959, &mem_102959_cached_sizze_105485, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102990_cached_sizze_105486 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102990, &mem_102990_cached_sizze_105486, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102991_cached_sizze_105487 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102991, &mem_102991_cached_sizze_105487, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102992_cached_sizze_105488 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102992, &mem_102992_cached_sizze_105488, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102993_cached_sizze_105489 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102993, &mem_102993_cached_sizze_105489, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102994_cached_sizze_105490 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102994, &mem_102994_cached_sizze_105490, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_102995_cached_sizze_105491 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_102995, &mem_102995_cached_sizze_105491, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103062_cached_sizze_105492 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103062, &mem_103062_cached_sizze_105492, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103063_cached_sizze_105493 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103063, &mem_103063_cached_sizze_105493, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103064_cached_sizze_105494 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103064, &mem_103064_cached_sizze_105494, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103065_cached_sizze_105495 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103065, &mem_103065_cached_sizze_105495, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103066_cached_sizze_105496 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103066, &mem_103066_cached_sizze_105496, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103067_cached_sizze_105497 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103067, &mem_103067_cached_sizze_105497, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103098_cached_sizze_105498 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_103098, &mem_103098_cached_sizze_105498, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103099_cached_sizze_105499 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_103099, &mem_103099_cached_sizze_105499, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103100_cached_sizze_105500 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_103100, &mem_103100_cached_sizze_105500, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103101_cached_sizze_105501 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_103101, &mem_103101_cached_sizze_105501, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103102_cached_sizze_105502 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_103102, &mem_103102_cached_sizze_105502, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103103_cached_sizze_105503 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_103103, &mem_103103_cached_sizze_105503, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103128_cached_sizze_105504 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_103128, &mem_103128_cached_sizze_105504, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103129_cached_sizze_105505 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_103129, &mem_103129_cached_sizze_105505, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103130_cached_sizze_105506 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_103130, &mem_103130_cached_sizze_105506, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103131_cached_sizze_105507 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_103131, &mem_103131_cached_sizze_105507, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103132_cached_sizze_105508 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_103132, &mem_103132_cached_sizze_105508, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103133_cached_sizze_105509 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_103133, &mem_103133_cached_sizze_105509, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103224_cached_sizze_105510 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_103224, &mem_103224_cached_sizze_105510, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103225_cached_sizze_105511 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103225, &mem_103225_cached_sizze_105511, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103226_cached_sizze_105512 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103226, &mem_103226_cached_sizze_105512, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103242_cached_sizze_105513 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103242, &mem_103242_cached_sizze_105513, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103243_cached_sizze_105514 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103243, &mem_103243_cached_sizze_105514, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103244_cached_sizze_105515 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103244, &mem_103244_cached_sizze_105515, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103257_cached_sizze_105516 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103257, &mem_103257_cached_sizze_105516, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103258_cached_sizze_105517 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103258, &mem_103258_cached_sizze_105517, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103259_cached_sizze_105518 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103259, &mem_103259_cached_sizze_105518, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103290_cached_sizze_105519 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103290, &mem_103290_cached_sizze_105519, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103291_cached_sizze_105520 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103291, &mem_103291_cached_sizze_105520, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103300_cached_sizze_105521 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103300, &mem_103300_cached_sizze_105521, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103301_cached_sizze_105522 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103301, &mem_103301_cached_sizze_105522, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103322_cached_sizze_105523 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103322, &mem_103322_cached_sizze_105523, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103323_cached_sizze_105524 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103323, &mem_103323_cached_sizze_105524, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103332_cached_sizze_105525 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103332, &mem_103332_cached_sizze_105525, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103333_cached_sizze_105526 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103333, &mem_103333_cached_sizze_105526, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103346_cached_sizze_105527 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103346, &mem_103346_cached_sizze_105527, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103347_cached_sizze_105528 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103347, &mem_103347_cached_sizze_105528, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103360_cached_sizze_105529 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103360, &mem_103360_cached_sizze_105529, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103361_cached_sizze_105530 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103361, &mem_103361_cached_sizze_105530, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103382_cached_sizze_105531 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_103382, &mem_103382_cached_sizze_105531, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103383_cached_sizze_105532 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_103383, &mem_103383_cached_sizze_105532, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103392_cached_sizze_105533 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_103392, &mem_103392_cached_sizze_105533, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103393_cached_sizze_105534 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_103393, &mem_103393_cached_sizze_105534, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103429_cached_sizze_105535 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103429, &mem_103429_cached_sizze_105535, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103430_cached_sizze_105536 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103430, &mem_103430_cached_sizze_105536, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103431_cached_sizze_105537 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103431, &mem_103431_cached_sizze_105537, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103443_cached_sizze_105538 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103443, &mem_103443_cached_sizze_105538, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103444_cached_sizze_105539 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103444, &mem_103444_cached_sizze_105539, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103468_cached_sizze_105540 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103468, &mem_103468_cached_sizze_105540, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103469_cached_sizze_105541 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103469, &mem_103469_cached_sizze_105541, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103478_cached_sizze_105542 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103478, &mem_103478_cached_sizze_105542, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103479_cached_sizze_105543 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103479, &mem_103479_cached_sizze_105543, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103500_cached_sizze_105544 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103500, &mem_103500_cached_sizze_105544, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103501_cached_sizze_105545 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103501, &mem_103501_cached_sizze_105545, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103510_cached_sizze_105546 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103510, &mem_103510_cached_sizze_105546, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103511_cached_sizze_105547 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103511, &mem_103511_cached_sizze_105547, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103532_cached_sizze_105548 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103532, &mem_103532_cached_sizze_105548, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103533_cached_sizze_105549 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103533, &mem_103533_cached_sizze_105549, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103534_cached_sizze_105550 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103534, &mem_103534_cached_sizze_105550, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103547_cached_sizze_105551 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103547, &mem_103547_cached_sizze_105551, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103548_cached_sizze_105552 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103548, &mem_103548_cached_sizze_105552, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103549_cached_sizze_105553 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103549, &mem_103549_cached_sizze_105553, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103568_cached_sizze_105554 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103568, &mem_103568_cached_sizze_105554, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103587_cached_sizze_105555 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103587, &mem_103587_cached_sizze_105555, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103588_cached_sizze_105556 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_103588, &mem_103588_cached_sizze_105556, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103589_cached_sizze_105557 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_103589, &mem_103589_cached_sizze_105557, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103601_cached_sizze_105558 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_103601, &mem_103601_cached_sizze_105558, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103602_cached_sizze_105559 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_103602, &mem_103602_cached_sizze_105559, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103626_cached_sizze_105560 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103626, &mem_103626_cached_sizze_105560, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103627_cached_sizze_105561 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_103627, &mem_103627_cached_sizze_105561, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103628_cached_sizze_105562 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_103628, &mem_103628_cached_sizze_105562, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103640_cached_sizze_105563 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_103640, &mem_103640_cached_sizze_105563, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103641_cached_sizze_105564 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_103641, &mem_103641_cached_sizze_105564, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103665_cached_sizze_105565 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103665, &mem_103665_cached_sizze_105565, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103666_cached_sizze_105566 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103666, &mem_103666_cached_sizze_105566, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103675_cached_sizze_105567 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103675, &mem_103675_cached_sizze_105567, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103676_cached_sizze_105568 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103676, &mem_103676_cached_sizze_105568, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103697_cached_sizze_105569 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103697, &mem_103697_cached_sizze_105569, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103698_cached_sizze_105570 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103698, &mem_103698_cached_sizze_105570, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103707_cached_sizze_105571 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103707, &mem_103707_cached_sizze_105571, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103708_cached_sizze_105572 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103708, &mem_103708_cached_sizze_105572, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103729_cached_sizze_105573 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_103729, &mem_103729_cached_sizze_105573, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103730_cached_sizze_105574 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_103730, &mem_103730_cached_sizze_105574, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103739_cached_sizze_105575 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_103739, &mem_103739_cached_sizze_105575, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103740_cached_sizze_105576 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_103740, &mem_103740_cached_sizze_105576, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103761_cached_sizze_105577 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_103761, &mem_103761_cached_sizze_105577, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103762_cached_sizze_105578 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_103762, &mem_103762_cached_sizze_105578, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103763_cached_sizze_105579 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103763, &mem_103763_cached_sizze_105579, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103775_cached_sizze_105580 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_103775, &mem_103775_cached_sizze_105580, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103776_cached_sizze_105581 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_103776, &mem_103776_cached_sizze_105581, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103777_cached_sizze_105582 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_103777, &mem_103777_cached_sizze_105582, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103796_cached_sizze_105583 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_103796, &mem_103796_cached_sizze_105583, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103797_cached_sizze_105584 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_103797, &mem_103797_cached_sizze_105584, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103798_cached_sizze_105585 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_103798, &mem_103798_cached_sizze_105585, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103817_cached_sizze_105586 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_103817, &mem_103817_cached_sizze_105586, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103818_cached_sizze_105587 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_103818, &mem_103818_cached_sizze_105587, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103819_cached_sizze_105588 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_103819, &mem_103819_cached_sizze_105588, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103849_cached_sizze_105589 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103849, &mem_103849_cached_sizze_105589, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103856_cached_sizze_105590 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_103856, &mem_103856_cached_sizze_105590, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103861_cached_sizze_105591 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_103861, &mem_103861_cached_sizze_105591, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103872_cached_sizze_105592 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103872, &mem_103872_cached_sizze_105592, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103877_cached_sizze_105593 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103877, &mem_103877_cached_sizze_105593, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103888_cached_sizze_105594 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_103888, &mem_103888_cached_sizze_105594, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103889_cached_sizze_105595 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_103889, &mem_103889_cached_sizze_105595, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103898_cached_sizze_105596 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_103898, &mem_103898_cached_sizze_105596, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103899_cached_sizze_105597 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_103899, &mem_103899_cached_sizze_105597, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103920_cached_sizze_105598 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_103920, &mem_103920_cached_sizze_105598, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103925_cached_sizze_105599 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_103925, &mem_103925_cached_sizze_105599, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103936_cached_sizze_105600 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103936, &mem_103936_cached_sizze_105600, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103941_cached_sizze_105601 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103941, &mem_103941_cached_sizze_105601, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103952_cached_sizze_105602 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103952, &mem_103952_cached_sizze_105602, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103959_cached_sizze_105603 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103959, &mem_103959_cached_sizze_105603, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103966_cached_sizze_105604 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103966, &mem_103966_cached_sizze_105604, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103976_cached_sizze_105605 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103976, &mem_103976_cached_sizze_105605, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103981_cached_sizze_105606 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_103981, &mem_103981_cached_sizze_105606, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103992_cached_sizze_105607 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103992, &mem_103992_cached_sizze_105607, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_103993_cached_sizze_105608 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_103993, &mem_103993_cached_sizze_105608, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104002_cached_sizze_105609 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104002, &mem_104002_cached_sizze_105609, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104003_cached_sizze_105610 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104003, &mem_104003_cached_sizze_105610, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104024_cached_sizze_105611 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_104024, &mem_104024_cached_sizze_105611, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104025_cached_sizze_105612 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104025, &mem_104025_cached_sizze_105612, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104036_cached_sizze_105613 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104036, &mem_104036_cached_sizze_105613, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104037_cached_sizze_105614 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_104037, &mem_104037_cached_sizze_105614, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104046_cached_sizze_105615 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_104046, &mem_104046_cached_sizze_105615, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104053_cached_sizze_105616 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104053, &mem_104053_cached_sizze_105616, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104078_cached_sizze_105617 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_104078, &mem_104078_cached_sizze_105617, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104079_cached_sizze_105618 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_104079, &mem_104079_cached_sizze_105618, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104090_cached_sizze_105619 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104090, &mem_104090_cached_sizze_105619, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104091_cached_sizze_105620 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104091, &mem_104091_cached_sizze_105620, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104100_cached_sizze_105621 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104100, &mem_104100_cached_sizze_105621, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104107_cached_sizze_105622 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104107, &mem_104107_cached_sizze_105622, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104114_cached_sizze_105623 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104114, &mem_104114_cached_sizze_105623, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104121_cached_sizze_105624 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104121, &mem_104121_cached_sizze_105624, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104146_cached_sizze_105625 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104146, &mem_104146_cached_sizze_105625, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104147_cached_sizze_105626 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_104147, &mem_104147_cached_sizze_105626, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104158_cached_sizze_105627 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_104158, &mem_104158_cached_sizze_105627, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104159_cached_sizze_105628 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104159, &mem_104159_cached_sizze_105628, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104168_cached_sizze_105629 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104168, &mem_104168_cached_sizze_105629, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104175_cached_sizze_105630 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_104175, &mem_104175_cached_sizze_105630, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104200_cached_sizze_105631 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_104200, &mem_104200_cached_sizze_105631, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104205_cached_sizze_105632 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104205, &mem_104205_cached_sizze_105632, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104216_cached_sizze_105633 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_104216, &mem_104216_cached_sizze_105633, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104222_cached_sizze_105634 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104222, &mem_104222_cached_sizze_105634, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104227_cached_sizze_105635 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104227, &mem_104227_cached_sizze_105635, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104243_cached_sizze_105636 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_104243, &mem_104243_cached_sizze_105636, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104249_cached_sizze_105637 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104249, &mem_104249_cached_sizze_105637, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104254_cached_sizze_105638 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104254, &mem_104254_cached_sizze_105638, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104270_cached_sizze_105639 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104270, &mem_104270_cached_sizze_105639, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104271_cached_sizze_105640 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104271, &mem_104271_cached_sizze_105640, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104282_cached_sizze_105641 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_104282, &mem_104282_cached_sizze_105641, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104283_cached_sizze_105642 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_104283, &mem_104283_cached_sizze_105642, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104292_cached_sizze_105643 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_104292, &mem_104292_cached_sizze_105643, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104293_cached_sizze_105644 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_104293, &mem_104293_cached_sizze_105644, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104324_cached_sizze_105645 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104324, &mem_104324_cached_sizze_105645, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104325_cached_sizze_105646 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104325, &mem_104325_cached_sizze_105646, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104326_cached_sizze_105647 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104326, &mem_104326_cached_sizze_105647, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104339_cached_sizze_105648 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104339, &mem_104339_cached_sizze_105648, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104340_cached_sizze_105649 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104340, &mem_104340_cached_sizze_105649, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104341_cached_sizze_105650 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104341, &mem_104341_cached_sizze_105650, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104372_cached_sizze_105651 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104372, &mem_104372_cached_sizze_105651, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104373_cached_sizze_105652 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104373, &mem_104373_cached_sizze_105652, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104374_cached_sizze_105653 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104374, &mem_104374_cached_sizze_105653, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104375_cached_sizze_105654 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104375, &mem_104375_cached_sizze_105654, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104392_cached_sizze_105655 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104392, &mem_104392_cached_sizze_105655, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104393_cached_sizze_105656 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104393, &mem_104393_cached_sizze_105656, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104394_cached_sizze_105657 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104394, &mem_104394_cached_sizze_105657, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104395_cached_sizze_105658 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104395, &mem_104395_cached_sizze_105658, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104436_cached_sizze_105659 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104436, &mem_104436_cached_sizze_105659, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104443_cached_sizze_105660 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104443, &mem_104443_cached_sizze_105660, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104450_cached_sizze_105661 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104450, &mem_104450_cached_sizze_105661, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104460_cached_sizze_105662 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104460, &mem_104460_cached_sizze_105662, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104465_cached_sizze_105663 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104465, &mem_104465_cached_sizze_105663, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104476_cached_sizze_105664 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104476, &mem_104476_cached_sizze_105664, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104483_cached_sizze_105665 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104483, &mem_104483_cached_sizze_105665, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104490_cached_sizze_105666 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104490, &mem_104490_cached_sizze_105666, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104500_cached_sizze_105667 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104500, &mem_104500_cached_sizze_105667, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104505_cached_sizze_105668 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104505, &mem_104505_cached_sizze_105668, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104516_cached_sizze_105669 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104516, &mem_104516_cached_sizze_105669, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104517_cached_sizze_105670 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_104517, &mem_104517_cached_sizze_105670, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104526_cached_sizze_105671 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104526, &mem_104526_cached_sizze_105671, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104527_cached_sizze_105672 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104527, &mem_104527_cached_sizze_105672, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104548_cached_sizze_105673 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_104548, &mem_104548_cached_sizze_105673, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104553_cached_sizze_105674 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104553, &mem_104553_cached_sizze_105674, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104564_cached_sizze_105675 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_104564, &mem_104564_cached_sizze_105675, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104565_cached_sizze_105676 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_104565, &mem_104565_cached_sizze_105676, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104574_cached_sizze_105677 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104574, &mem_104574_cached_sizze_105677, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_104575_cached_sizze_105678 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_104575, &mem_104575_cached_sizze_105678, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:498:5-505:33
    if (memblock_set(ctx, &mem_param_102644, &wdown_mem_102609, "wdown_mem_102609") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102648, &wkey_mem_102610, "wkey_mem_102610") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102652, &wout_mem_102611, "wout_mem_102611") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102656, &wpe_mem_102612, "wpe_mem_102612") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102660, &wqry_mem_102613, "wqry_mem_102613") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102664, &wte_mem_102614, "wte_mem_102614") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102668, &wup_mem_102615, "wup_mem_102615") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102672, &wval_mem_102616, "wval_mem_102616") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102676, &wvoc_mem_102617, "wvoc_mem_102617") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102680, &wdown_mem_102618, "wdown_mem_102618") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102684, &wkey_mem_102619, "wkey_mem_102619") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102688, &wout_mem_102620, "wout_mem_102620") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102692, &wpe_mem_102621, "wpe_mem_102621") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102696, &wqry_mem_102622, "wqry_mem_102622") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102700, &wte_mem_102623, "wte_mem_102623") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102704, &wup_mem_102624, "wup_mem_102624") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102708, &wval_mem_102625, "wval_mem_102625") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102712, &wvoc_mem_102626, "wvoc_mem_102626") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102716, &wdown_mem_102627, "wdown_mem_102627") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102720, &wkey_mem_102628, "wkey_mem_102628") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102724, &wout_mem_102629, "wout_mem_102629") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102728, &wpe_mem_102630, "wpe_mem_102630") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102732, &wqry_mem_102631, "wqry_mem_102631") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102736, &wte_mem_102632, "wte_mem_102632") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102740, &wup_mem_102633, "wup_mem_102633") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102744, &wval_mem_102634, "wval_mem_102634") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102748, &wvoc_mem_102635, "wvoc_mem_102635") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_102751, &mem_102640, "mem_102640") != 0)
        return 1;
    for (int64_t step_90216 = 0; step_90216 < n_74237; step_90216++) {
        // futhark/microgpt.fut:500:16-25
        
        int64_t dl_90245 = ((int64_t *) dls_mem_102637.mem)[step_90216];
        
        // futhark/microgpt.fut:409:17-37
        // futhark/microgpt.fut:409:17-37
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_102752.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint64_t *) seqs_mem_102638.mem, step_90216 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        // futhark/microgpt.fut:409:17-37
        if (futrts_cal_target_8179(ctx, &ext_mem_102754, mem_102752, dl_90245) != 0) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101588 = 0; i_101588 < (int64_t) 16; i_101588++) {
            int64_t tmp_92683 = ((int64_t *) seqs_mem_102638.mem)[step_90216 * (int64_t) 16 + i_101588];
            
            // futhark/microgpt.fut:410:37-51
            
            bool x_92684 = sle64((int64_t) 0, tmp_92683);
            
            // futhark/microgpt.fut:410:37-51
            
            bool y_92685 = slt64(tmp_92683, (int64_t) 27);
            
            // futhark/microgpt.fut:410:37-51
            
            bool bounds_check_92686 = x_92684 && y_92685;
            
            // futhark/microgpt.fut:410:37-51
            
            bool index_certs_92687;
            
            if (!bounds_check_92686) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_92683, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:410:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:410:16-55\n   #6  futhark/microgpt.fut:479:26-483:39\n   #7  futhark/microgpt.fut:503:35-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101575 = 0; i_101575 < (int64_t) 16; i_101575++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_97308 = ((double *) mem_param_102664.mem)[tmp_92683 * (int64_t) 16 + i_101575];
                
                ((double *) mem_102770)[i_101575] = lifted_lambda_res_97308;
                ((double *) mem_102771)[i_101575] = lifted_lambda_res_97308;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101580 = 0; i_101580 < (int64_t) 27; i_101580++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_92730 = ((double *) ext_mem_102754.mem)[i_101588 * (int64_t) 27 + i_101580];
                
                // futhark/microgpt.fut:297:51-87
                
                double zt_res_92731 = -6.25e-2 * zt_rhs_92730;
                
                ((double *) mem_102784)[i_101580] = zt_res_92731;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102755, i_101588 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102784, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102756, i_101588 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102770, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102757, i_101588 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102771, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101603 = 0; i_101603 < (int64_t) 16; i_101603++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101596 = 0; i_101596 < (int64_t) 16; i_101596++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_97336 = ((double *) mem_param_102656.mem)[i_101603 * (int64_t) 16 + i_101596];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_97337 = ((double *) mem_102757)[i_101603 * (int64_t) 16 + i_101596];
                
                // futhark/microgpt.fut:200:40-72
                
                double zp_res_97338 = zp_lhs_97336 + zp_rhs_97337;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_97346 = ((double *) mem_102756)[i_101603 * (int64_t) 16 + i_101596];
                
                // futhark/microgpt.fut:265:35-63
                
                double zp_res_97347 = zp_lhs_97336 + zp_rhs_97346;
                
                ((double *) mem_102813)[i_101596] = zp_res_97347;
                ((double *) mem_102814)[i_101596] = zp_res_97338;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102803, i_101603 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102813, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102804, i_101603 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102814, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101626 = 0; i_101626 < (int64_t) 16; i_101626++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_92902;
            double r_92904 = 0.0;
            
            for (int64_t i_92903 = 0; i_92903 < (int64_t) 16; i_92903++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_92905 = ((double *) mem_102803)[i_101626 * (int64_t) 16 + i_92903];
                
                // futhark/microgpt.fut:266:58-83
                
                double zt_res_92906 = zt_lhs_92905 * zt_lhs_92905;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_92907 = r_92904 + zt_res_92906;
                double r_tmp_104987 = zp_res_92907;
                
                r_92904 = r_tmp_104987;
            }
            defunc_0_lifted_lambda_res_92902 = r_92904;
            // futhark/microgpt.fut:266:40-101
            
            double zs_res_92908 = defunc_0_lifted_lambda_res_92902 / 16.0;
            
            // futhark/microgpt.fut:267:23-53
            
            double zp_res_92909 = 1.0e-5 + zs_res_92908;
            
            // futhark/microgpt.fut:267:15-53
            
            double sqrt_res_92910 = futrts_sqrt64(zp_res_92909);
            
            // futhark/microgpt.fut:268:39-49
            
            double zs_res_92911 = 1.0 / sqrt_res_92910;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101612 = 0; i_101612 < (int64_t) 16; i_101612++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_97400 = ((double *) mem_102804)[i_101626 * (int64_t) 16 + i_101612];
                
                // futhark/microgpt.fut:201:64-93
                
                double zt_res_97401 = zt_lhs_97400 * zt_lhs_97400;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_97408 = ((double *) mem_102803)[i_101626 * (int64_t) 16 + i_101612];
                
                // futhark/microgpt.fut:268:23-49
                
                double zt_res_97409 = zs_res_92911 * zt_lhs_97408;
                
                // futhark/microgpt.fut:340:53-86
                
                double zt_res_97420 = zt_lhs_97408 * zt_lhs_97408;
                
                ((double *) mem_102850)[i_101612] = zt_res_97420;
                ((double *) mem_102851)[i_101612] = zt_res_97409;
                ((double *) mem_102852)[i_101612] = zt_res_97401;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_92878;
            double r_92880 = 0.0;
            
            for (int64_t i_92879 = 0; i_92879 < (int64_t) 16; i_92879++) {
                // futhark/microgpt.fut:202:35-43
                
                double lifted_lambda_res_92881 = ((double *) mem_102852)[i_92879];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_92882 = r_92880 + lifted_lambda_res_92881;
                double r_tmp_104991 = zp_res_92882;
                
                r_92880 = r_tmp_104991;
            }
            defunc_0_lifted_lambda_res_92878 = r_92880;
            // futhark/microgpt.fut:202:17-60
            
            double zs_res_92883 = defunc_0_lifted_lambda_res_92878 / 16.0;
            
            // futhark/microgpt.fut:203:24-55
            
            double zp_res_92884 = 1.0e-5 + zs_res_92883;
            
            // futhark/microgpt.fut:203:16-55
            
            double sqrt_res_92885 = futrts_sqrt64(zp_res_92884);
            
            // futhark/microgpt.fut:204:42-53
            
            double zs_res_92886 = 1.0 / sqrt_res_92885;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101618 = 0; i_101618 < (int64_t) 16; i_101618++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_92893 = ((double *) mem_102804)[i_101626 * (int64_t) 16 + i_101618];
                
                // futhark/microgpt.fut:204:24-53
                
                double zt_res_92894 = zs_res_92886 * zt_lhs_92893;
                
                ((double *) mem_102871)[i_101618] = zt_res_92894;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102835, i_101626 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102850, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102836, i_101626 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102851, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102837, i_101626 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102871, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101652 = 0; i_101652 < (int64_t) 16; i_101652++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_93165;
            double r_93167 = 0.0;
            
            for (int64_t i_93166 = 0; i_93166 < (int64_t) 16; i_93166++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_93168 = ((double *) mem_102836)[i_101652 * (int64_t) 16 + i_93166];
                
                // futhark/microgpt.fut:269:61-90
                
                double zt_res_93169 = zt_lhs_93168 * zt_lhs_93168;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_93170 = r_93167 + zt_res_93169;
                double r_tmp_104997 = zp_res_93170;
                
                r_93167 = r_tmp_104997;
            }
            defunc_0_lifted_lambda_res_93165 = r_93167;
            // futhark/microgpt.fut:269:42-108
            
            double zs_res_93171 = defunc_0_lifted_lambda_res_93165 / 16.0;
            
            // futhark/microgpt.fut:270:24-55
            
            double zp_res_93172 = 1.0e-5 + zs_res_93171;
            
            // futhark/microgpt.fut:270:16-55
            
            double sqrt_res_93173 = futrts_sqrt64(zp_res_93172);
            
            // futhark/microgpt.fut:271:42-53
            
            double zs_res_93174 = 1.0 / sqrt_res_93173;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101636 = 0; i_101636 < (int64_t) 16; i_101636++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_97476 = ((double *) mem_102837)[i_101652 * (int64_t) 16 + i_101636];
                
                // futhark/microgpt.fut:205:64-93
                
                double zt_res_97477 = zt_lhs_97476 * zt_lhs_97476;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_97484 = ((double *) mem_102836)[i_101652 * (int64_t) 16 + i_101636];
                
                // futhark/microgpt.fut:271:24-53
                
                double zt_res_97485 = zs_res_93174 * zt_lhs_97484;
                
                // futhark/microgpt.fut:333:53-86
                
                double zt_res_97496 = zt_lhs_97484 * zt_lhs_97484;
                
                ((double *) mem_102909)[i_101636] = zt_res_97496;
                ((double *) mem_102910)[i_101636] = zt_res_97485;
                ((double *) mem_102911)[i_101636] = zt_res_97477;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_93141;
            double r_93143 = 0.0;
            
            for (int64_t i_93142 = 0; i_93142 < (int64_t) 16; i_93142++) {
                // futhark/microgpt.fut:206:35-43
                
                double lifted_lambda_res_93144 = ((double *) mem_102911)[i_93142];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_93145 = r_93143 + lifted_lambda_res_93144;
                double r_tmp_105001 = zp_res_93145;
                
                r_93143 = r_tmp_105001;
            }
            defunc_0_lifted_lambda_res_93141 = r_93143;
            // futhark/microgpt.fut:206:17-60
            
            double zs_res_93146 = defunc_0_lifted_lambda_res_93141 / 16.0;
            
            // futhark/microgpt.fut:207:24-55
            
            double zp_res_93147 = 1.0e-5 + zs_res_93146;
            
            // futhark/microgpt.fut:207:16-55
            
            double sqrt_res_93148 = futrts_sqrt64(zp_res_93147);
            
            // futhark/microgpt.fut:208:42-53
            
            double zs_res_93149 = 1.0 / sqrt_res_93148;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101642 = 0; i_101642 < (int64_t) 16; i_101642++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_93156 = ((double *) mem_102837)[i_101652 * (int64_t) 16 + i_101642];
                
                // futhark/microgpt.fut:208:24-53
                
                double zt_res_93157 = zs_res_93149 * zt_lhs_93156;
                
                ((double *) mem_102930)[i_101642] = zt_res_93157;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_93213;
            double r_93215 = 0.0;
            
            for (int64_t i_93214 = 0; i_93214 < (int64_t) 16; i_93214++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_93216 = ((double *) mem_102835)[i_101652 * (int64_t) 16 + i_93214];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_93217 = r_93215 + lifted_lambda_res_93216;
                double r_tmp_105003 = zp_res_93217;
                
                r_93215 = r_tmp_105003;
            }
            defunc_0_lifted_lambda_res_93213 = r_93215;
            // futhark/microgpt.fut:341:34-86
            
            double zs_res_93218 = defunc_0_lifted_lambda_res_93213 / 16.0;
            
            ((double *) mem_102890)[i_101652] = zs_res_93218;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102891, i_101652 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102909, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102892, i_101652 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102910, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102893, i_101652 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102930, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101692 = 0; i_101692 < (int64_t) 16; i_101692++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101669 = 0; i_101669 < (int64_t) 16; i_101669++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_97837;
                double r_97839 = 0.0;
                
                for (int64_t i_97838 = 0; i_97838 < (int64_t) 16; i_97838++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_97840 = ((double *) mem_param_102660.mem)[i_101669 * (int64_t) 16 + i_97838];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_97841 = ((double *) mem_102893)[i_101692 * (int64_t) 16 + i_97838];
                    
                    // futhark/microgpt.fut:209:72-103
                    
                    double zt_res_97842 = zt_lhs_97840 * zt_rhs_97841;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_97843 = r_97839 + zt_res_97842;
                    double r_tmp_105018 = zp_res_97843;
                    
                    r_97839 = r_tmp_105018;
                }
                defunc_0_lifted_lambda_res_97837 = r_97839;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_97850;
                double r_97852 = 0.0;
                
                for (int64_t i_97851 = 0; i_97851 < (int64_t) 16; i_97851++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_97853 = ((double *) mem_param_102648.mem)[i_101669 * (int64_t) 16 + i_97851];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_97854 = ((double *) mem_102893)[i_101692 * (int64_t) 16 + i_97851];
                    
                    // futhark/microgpt.fut:210:72-103
                    
                    double zt_res_97855 = zt_lhs_97853 * zt_rhs_97854;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_97856 = r_97852 + zt_res_97855;
                    double r_tmp_105019 = zp_res_97856;
                    
                    r_97852 = r_tmp_105019;
                }
                defunc_0_lifted_lambda_res_97850 = r_97852;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_97866;
                double r_97868 = 0.0;
                
                for (int64_t i_97867 = 0; i_97867 < (int64_t) 16; i_97867++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_97869 = ((double *) mem_param_102672.mem)[i_101669 * (int64_t) 16 + i_97867];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_97870 = ((double *) mem_102893)[i_101692 * (int64_t) 16 + i_97867];
                    
                    // futhark/microgpt.fut:211:72-103
                    
                    double zt_res_97871 = zt_lhs_97869 * zt_rhs_97870;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_97872 = r_97868 + zt_res_97871;
                    double r_tmp_105020 = zp_res_97872;
                    
                    r_97868 = r_tmp_105020;
                }
                defunc_0_lifted_lambda_res_97866 = r_97868;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_97884;
                double r_97886 = 0.0;
                
                for (int64_t i_97885 = 0; i_97885 < (int64_t) 16; i_97885++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_97887 = ((double *) mem_param_102660.mem)[i_101669 * (int64_t) 16 + i_97885];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_97888 = ((double *) mem_102892)[i_101692 * (int64_t) 16 + i_97885];
                    
                    // futhark/microgpt.fut:272:69-100
                    
                    double zt_res_97889 = zt_lhs_97887 * zt_rhs_97888;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_97890 = r_97886 + zt_res_97889;
                    double r_tmp_105021 = zp_res_97890;
                    
                    r_97886 = r_tmp_105021;
                }
                defunc_0_lifted_lambda_res_97884 = r_97886;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_97904;
                double r_97906 = 0.0;
                
                for (int64_t i_97905 = 0; i_97905 < (int64_t) 16; i_97905++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_97907 = ((double *) mem_param_102648.mem)[i_101669 * (int64_t) 16 + i_97905];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_97908 = ((double *) mem_102892)[i_101692 * (int64_t) 16 + i_97905];
                    
                    // futhark/microgpt.fut:273:69-100
                    
                    double zt_res_97909 = zt_lhs_97907 * zt_rhs_97908;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_97910 = r_97906 + zt_res_97909;
                    double r_tmp_105022 = zp_res_97910;
                    
                    r_97906 = r_tmp_105022;
                }
                defunc_0_lifted_lambda_res_97904 = r_97906;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_97926;
                double r_97928 = 0.0;
                
                for (int64_t i_97927 = 0; i_97927 < (int64_t) 16; i_97927++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_97929 = ((double *) mem_param_102672.mem)[i_101669 * (int64_t) 16 + i_97927];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_97930 = ((double *) mem_102892)[i_101692 * (int64_t) 16 + i_97927];
                    
                    // futhark/microgpt.fut:274:69-100
                    
                    double zt_res_97931 = zt_lhs_97929 * zt_rhs_97930;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_97932 = r_97928 + zt_res_97931;
                    double r_tmp_105023 = zp_res_97932;
                    
                    r_97928 = r_tmp_105023;
                }
                defunc_0_lifted_lambda_res_97926 = r_97928;
                ((double *) mem_102990)[i_101669] = defunc_0_lifted_lambda_res_97926;
                ((double *) mem_102991)[i_101669] = defunc_0_lifted_lambda_res_97904;
                ((double *) mem_102992)[i_101669] = defunc_0_lifted_lambda_res_97884;
                ((double *) mem_102993)[i_101669] = defunc_0_lifted_lambda_res_97866;
                ((double *) mem_102994)[i_101669] = defunc_0_lifted_lambda_res_97850;
                ((double *) mem_102995)[i_101669] = defunc_0_lifted_lambda_res_97837;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_94217;
            double r_94219 = 0.0;
            
            for (int64_t i_94218 = 0; i_94218 < (int64_t) 16; i_94218++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_94220 = ((double *) mem_102891)[i_101692 * (int64_t) 16 + i_94218];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_94221 = r_94219 + lifted_lambda_res_94220;
                double r_tmp_105024 = zp_res_94221;
                
                r_94219 = r_tmp_105024;
            }
            defunc_0_lifted_lambda_res_94217 = r_94219;
            // futhark/microgpt.fut:334:34-86
            
            double zs_res_94222 = defunc_0_lifted_lambda_res_94217 / 16.0;
            
            // futhark/microgpt.fut:342:41-51
            
            double zp_lhs_94242 = ((double *) mem_102890)[i_101692];
            
            // futhark/microgpt.fut:342:41-79
            
            double zp_res_94243 = 1.0e-5 + zp_lhs_94242;
            
            // futhark/microgpt.fut:342:33-79
            
            double sqrt_res_94244 = futrts_sqrt64(zp_res_94243);
            
            ((double *) mem_102952)[i_101692] = sqrt_res_94244;
            ((double *) mem_102953)[i_101692] = zs_res_94222;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102954, i_101692 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102990, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102955, i_101692 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102991, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102956, i_101692 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102992, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102957, i_101692 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102993, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102958, i_101692 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102994, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_102959, i_101692 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_102995, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101751 = 0; i_101751 < (int64_t) 4; i_101751++) {
            // futhark/microgpt.fut:212:83-86
            
            int64_t zp_lhs_94666 = mul64((int64_t) 4, i_101751);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101732 = 0; i_101732 < (int64_t) 16; i_101732++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_101713 = 0; i_101713 < (int64_t) 4; i_101713++) {
                    // futhark/microgpt.fut:212:88-93
                    
                    int64_t tmp_98679 = add64(zp_lhs_94666, i_101713);
                    
                    // futhark/microgpt.fut:212:69-95
                    
                    bool x_98680 = sle64((int64_t) 0, tmp_98679);
                    
                    // futhark/microgpt.fut:212:69-95
                    
                    bool y_98681 = slt64(tmp_98679, (int64_t) 16);
                    
                    // futhark/microgpt.fut:212:69-95
                    
                    bool bounds_check_98682 = x_98680 && y_98681;
                    
                    // futhark/microgpt.fut:212:69-95
                    
                    bool index_certs_98683;
                    
                    if (!bounds_check_98682) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_98679, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:212:69-95\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:212:52-96\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:212:33-98\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:6:13-17\n   #9  futhark/microgpt.fut:212:15-100\n   #10 futhark/microgpt.fut:411:7-76\n   #11 futhark/microgpt.fut:479:26-483:39\n   #12 futhark/microgpt.fut:503:35-76\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_98684 = ((double *) mem_102959)[i_101732 * (int64_t) 16 + tmp_98679];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_98692 = ((double *) mem_102958)[i_101732 * (int64_t) 16 + tmp_98679];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_98703 = ((double *) mem_102957)[i_101732 * (int64_t) 16 + tmp_98679];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_98716 = ((double *) mem_102956)[i_101732 * (int64_t) 16 + tmp_98679];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_98731 = ((double *) mem_102955)[i_101732 * (int64_t) 16 + tmp_98679];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_98748 = ((double *) mem_102954)[i_101732 * (int64_t) 16 + tmp_98679];
                    
                    ((double *) mem_103128)[i_101713] = lifted_lambda_res_98748;
                    ((double *) mem_103129)[i_101713] = lifted_lambda_res_98731;
                    ((double *) mem_103130)[i_101713] = lifted_lambda_res_98716;
                    ((double *) mem_103131)[i_101713] = lifted_lambda_res_98703;
                    ((double *) mem_103132)[i_101713] = lifted_lambda_res_98692;
                    ((double *) mem_103133)[i_101713] = lifted_lambda_res_98684;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_103098, i_101732 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103128, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_103099, i_101732 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103129, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_103100, i_101732 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103130, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_103101, i_101732 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103131, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_103102, i_101732 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103132, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_103103, i_101732 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103133, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_103062, i_101751 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_103098, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_103063, i_101751 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_103099, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_103064, i_101751 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_103100, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_103065, i_101751 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_103101, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_103066, i_101751 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_103102, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_103067, i_101751 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_103103, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101843 = 0; i_101843 < (int64_t) 4; i_101843++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101774 = 0; i_101774 < (int64_t) 16; i_101774++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_101764 = 0; i_101764 < (int64_t) 16; i_101764++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_99174;
                    double r_99176 = 0.0;
                    
                    for (int64_t i_99175 = 0; i_99175 < (int64_t) 4; i_99175++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_99177 = ((double *) mem_103067)[i_101843 * (int64_t) 64 + i_101774 * (int64_t) 4 + i_99175];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_99178 = ((double *) mem_103066)[i_101843 * (int64_t) 64 + i_101764 * (int64_t) 4 + i_99175];
                        
                        // futhark/microgpt.fut:215:100-139
                        
                        double zt_res_99179 = zt_lhs_99177 * zt_rhs_99178;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_99180 = r_99176 + zt_res_99179;
                        double r_tmp_105052 = zp_res_99180;
                        
                        r_99176 = r_tmp_105052;
                    }
                    defunc_0_lifted_lambda_res_99174 = r_99176;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_99187;
                    double r_99189 = 0.0;
                    
                    for (int64_t i_99188 = 0; i_99188 < (int64_t) 4; i_99188++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_99190 = ((double *) mem_103064)[i_101843 * (int64_t) 64 + i_101774 * (int64_t) 4 + i_99188];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_99191 = ((double *) mem_103063)[i_101843 * (int64_t) 64 + i_101764 * (int64_t) 4 + i_99188];
                        
                        // futhark/microgpt.fut:278:97-138
                        
                        double zt_res_99192 = zt_lhs_99190 * zt_rhs_99191;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_99193 = r_99189 + zt_res_99192;
                        double r_tmp_105053 = zp_res_99193;
                        
                        r_99189 = r_tmp_105053;
                    }
                    defunc_0_lifted_lambda_res_99187 = r_99189;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_99203;
                    double r_99205 = 0.0;
                    
                    for (int64_t i_99204 = 0; i_99204 < (int64_t) 4; i_99204++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_99206 = ((double *) mem_103064)[i_101843 * (int64_t) 64 + i_101774 * (int64_t) 4 + i_99204];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_99207 = ((double *) mem_103063)[i_101843 * (int64_t) 64 + i_101764 * (int64_t) 4 + i_99204];
                        
                        // futhark/microgpt.fut:317:91-138
                        
                        double zt_res_99208 = zt_lhs_99206 * zt_rhs_99207;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_99209 = r_99205 + zt_res_99208;
                        double r_tmp_105054 = zp_res_99209;
                        
                        r_99205 = r_tmp_105054;
                    }
                    defunc_0_lifted_lambda_res_99203 = r_99205;
                    ((double *) mem_103257)[i_101764] = defunc_0_lifted_lambda_res_99203;
                    ((double *) mem_103258)[i_101764] = defunc_0_lifted_lambda_res_99187;
                    ((double *) mem_103259)[i_101764] = defunc_0_lifted_lambda_res_99174;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_103242, i_101774 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103257, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_103243, i_101774 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103258, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_103244, i_101774 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103259, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101789 = 0; i_101789 < (int64_t) 16; i_101789++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_101782 = 0; i_101782 < (int64_t) 16; i_101782++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_99231 = ((double *) mem_103244)[i_101789 * (int64_t) 16 + i_101782];
                    
                    // futhark/microgpt.fut:216:43-70
                    
                    double zs_res_99232 = zs_lhs_99231 / 2.0;
                    double zp_rhs_99233 = ((double *) masks_mem_102636.mem)[step_90216 * (int64_t) 256 + i_101789 * (int64_t) 16 + i_101782];
                    
                    // futhark/microgpt.fut:216:57-90
                    
                    double zp_res_99234 = zs_res_99232 + zp_rhs_99233;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_99241 = ((double *) mem_103243)[i_101789 * (int64_t) 16 + i_101782];
                    
                    // futhark/microgpt.fut:279:43-70
                    
                    double zs_res_99242 = zs_lhs_99241 / 2.0;
                    
                    // futhark/microgpt.fut:279:57-90
                    
                    double zp_res_99244 = zp_rhs_99233 + zs_res_99242;
                    
                    ((double *) mem_103300)[i_101782] = zp_res_99244;
                    ((double *) mem_103301)[i_101782] = zp_res_99234;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_103290, i_101789 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103300, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_103291, i_101789 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103301, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101820 = 0; i_101820 < (int64_t) 16; i_101820++) {
                // futhark/microgpt.fut:103:13-33
                
                double defunc_0_reduce_res_101370;
                double defunc_0_reduce_res_101371;
                double redout_101792;
                double redout_101793;
                
                redout_101792 = -INFINITY;
                redout_101793 = -INFINITY;
                for (int64_t i_101794 = 0; i_101794 < (int64_t) 16; i_101794++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_99307 = ((double *) mem_103291)[i_101820 * (int64_t) 16 + i_101794];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_99317 = ((double *) mem_103290)[i_101820 * (int64_t) 16 + i_101794];
                    
                    // futhark/microgpt.fut:103:13-33
                    
                    double max_res_98977 = fmax64(lifted_lambda_res_99307, redout_101792);
                    
                    // futhark/microgpt.fut:103:13-33
                    
                    double max_res_99029 = fmax64(lifted_lambda_res_99317, redout_101793);
                    double redout_tmp_105061 = max_res_98977;
                    double redout_tmp_105062 = max_res_99029;
                    
                    redout_101792 = redout_tmp_105061;
                    redout_101793 = redout_tmp_105062;
                }
                defunc_0_reduce_res_101370 = redout_101792;
                defunc_0_reduce_res_101371 = redout_101793;
                // futhark/microgpt.fut:113:47-56
                
                double neg_res_98978 = -defunc_0_reduce_res_101370;
                
                // futhark/microgpt.fut:113:47-56
                
                double neg_res_99030 = -defunc_0_reduce_res_101371;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_101799 = 0; i_101799 < (int64_t) 16; i_101799++) {
                    // futhark/microgpt.fut:113:38-41
                    
                    double lifted_lambda_res_99346 = ((double *) mem_103291)[i_101820 * (int64_t) 16 + i_101799];
                    
                    // futhark/microgpt.fut:113:38-56
                    
                    double zp_res_99347 = neg_res_98978 + lifted_lambda_res_99346;
                    
                    // futhark/microgpt.fut:113:31-56
                    
                    double exp_res_99348 = futrts_exp64(zp_res_99347);
                    
                    // futhark/microgpt.fut:113:38-41
                    
                    double lifted_lambda_res_99355 = ((double *) mem_103290)[i_101820 * (int64_t) 16 + i_101799];
                    
                    // futhark/microgpt.fut:113:38-56
                    
                    double zp_res_99356 = neg_res_99030 + lifted_lambda_res_99355;
                    
                    // futhark/microgpt.fut:113:31-56
                    
                    double exp_res_99357 = futrts_exp64(zp_res_99356);
                    
                    ((double *) mem_103332)[i_101799] = exp_res_99357;
                    ((double *) mem_103333)[i_101799] = exp_res_99348;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_98989;
                double r_98991 = 0.0;
                
                for (int64_t i_98990 = 0; i_98990 < (int64_t) 16; i_98990++) {
                    // futhark/microgpt.fut:114:32-39
                    
                    double lifted_lambda_res_98992 = ((double *) mem_103333)[i_98990];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_98993 = r_98991 + lifted_lambda_res_98992;
                    double r_tmp_105065 = zp_res_98993;
                    
                    r_98991 = r_tmp_105065;
                }
                defunc_0_lifted_lambda_res_98989 = r_98991;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_99041;
                double r_99043 = 0.0;
                
                for (int64_t i_99042 = 0; i_99042 < (int64_t) 16; i_99042++) {
                    // futhark/microgpt.fut:114:32-39
                    
                    double lifted_lambda_res_99044 = ((double *) mem_103332)[i_99042];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_99045 = r_99043 + lifted_lambda_res_99044;
                    double r_tmp_105066 = zp_res_99045;
                    
                    r_99043 = r_tmp_105066;
                }
                defunc_0_lifted_lambda_res_99041 = r_99043;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_101806 = 0; i_101806 < (int64_t) 16; i_101806++) {
                    // futhark/microgpt.fut:115:23-30
                    
                    double zs_lhs_99375 = ((double *) mem_103333)[i_101806];
                    
                    // futhark/microgpt.fut:115:23-40
                    
                    double zs_res_99376 = zs_lhs_99375 / defunc_0_lifted_lambda_res_98989;
                    
                    // futhark/microgpt.fut:115:23-30
                    
                    double zs_lhs_99383 = ((double *) mem_103332)[i_101806];
                    
                    // futhark/microgpt.fut:115:23-40
                    
                    double zs_res_99384 = zs_lhs_99383 / defunc_0_lifted_lambda_res_99041;
                    
                    ((double *) mem_103346)[i_101806] = zs_res_99384;
                    ((double *) mem_103347)[i_101806] = zs_res_99376;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_101813 = 0; i_101813 < (int64_t) 16; i_101813++) {
                    // futhark/microgpt.fut:218:23-31
                    
                    double lifted_lambda_res_99402 = ((double *) mem_103347)[i_101813];
                    
                    // futhark/microgpt.fut:281:23-31
                    
                    double lifted_lambda_res_99409 = ((double *) mem_103346)[i_101813];
                    
                    ((double *) mem_103360)[i_101813] = lifted_lambda_res_99409;
                    ((double *) mem_103361)[i_101813] = lifted_lambda_res_99402;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_103322, i_101820 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103360, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_103323, i_101820 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103361, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101834 = 0; i_101834 < (int64_t) 16; i_101834++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_101827 = 0; i_101827 < (int64_t) 4; i_101827++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_99435;
                    double r_99437 = 0.0;
                    
                    for (int64_t i_99436 = 0; i_99436 < (int64_t) 16; i_99436++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_99438 = ((double *) mem_103323)[i_101834 * (int64_t) 16 + i_99436];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_99439 = ((double *) mem_103065)[i_101843 * (int64_t) 64 + i_99436 * (int64_t) 4 + i_101827];
                        
                        // futhark/microgpt.fut:219:61-96
                        
                        double zt_res_99440 = zt_lhs_99438 * zt_rhs_99439;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_99441 = r_99437 + zt_res_99440;
                        double r_tmp_105075 = zp_res_99441;
                        
                        r_99437 = r_tmp_105075;
                    }
                    defunc_0_lifted_lambda_res_99435 = r_99437;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_99448;
                    double r_99450 = 0.0;
                    
                    for (int64_t i_99449 = 0; i_99449 < (int64_t) 16; i_99449++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_99451 = ((double *) mem_103322)[i_101834 * (int64_t) 16 + i_99449];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_99452 = ((double *) mem_103062)[i_101843 * (int64_t) 64 + i_99449 * (int64_t) 4 + i_101827];
                        
                        // futhark/microgpt.fut:282:61-97
                        
                        double zt_res_99453 = zt_lhs_99451 * zt_rhs_99452;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_99454 = r_99450 + zt_res_99453;
                        double r_tmp_105076 = zp_res_99454;
                        
                        r_99450 = r_tmp_105076;
                    }
                    defunc_0_lifted_lambda_res_99448 = r_99450;
                    ((double *) mem_103392)[i_101827] = defunc_0_lifted_lambda_res_99448;
                    ((double *) mem_103393)[i_101827] = defunc_0_lifted_lambda_res_99435;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_103382, i_101834 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103392, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_103383, i_101834 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103393, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_103224, i_101843 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_103242, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_103225, i_101843 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_103382, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_103226, i_101843 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_103383, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101860 = 0; i_101860 < (int64_t) 16; i_101860++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101851 = 0; i_101851 < (int64_t) 16; i_101851++) {
                // futhark/microgpt.fut:220:61-64
                
                int64_t tmp_99494 = sdiv64(i_101851, (int64_t) 4);
                
                // futhark/microgpt.fut:220:53-66
                
                bool x_99495 = sle64((int64_t) 0, tmp_99494);
                
                // futhark/microgpt.fut:220:53-66
                
                bool y_99496 = slt64(tmp_99494, (int64_t) 4);
                
                // futhark/microgpt.fut:220:53-66
                
                bool bounds_check_99497 = x_99495 && y_99496;
                
                // futhark/microgpt.fut:220:53-66
                
                bool index_certs_99498;
                
                if (!bounds_check_99497) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_99494, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:220:53-66\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:220:35-83\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:220:16-85\n   #7  futhark/microgpt.fut:411:7-76\n   #8  futhark/microgpt.fut:479:26-483:39\n   #9  futhark/microgpt.fut:503:35-76\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:220:77-80
                
                int64_t tmp_99499 = smod64(i_101851, (int64_t) 4);
                
                // futhark/microgpt.fut:220:53-82
                
                bool x_99500 = sle64((int64_t) 0, tmp_99499);
                
                // futhark/microgpt.fut:220:53-82
                
                bool y_99501 = slt64(tmp_99499, (int64_t) 4);
                
                // futhark/microgpt.fut:220:53-82
                
                bool bounds_check_99502 = x_99500 && y_99501;
                
                // futhark/microgpt.fut:220:53-82
                
                bool index_certs_99503;
                
                if (!bounds_check_99502) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_99499, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:220:53-82\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:220:35-83\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:220:16-85\n   #7  futhark/microgpt.fut:411:7-76\n   #8  futhark/microgpt.fut:479:26-483:39\n   #9  futhark/microgpt.fut:503:35-76\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_99504 = ((double *) mem_103226)[tmp_99494 * (int64_t) 64 + i_101860 * (int64_t) 4 + tmp_99499];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_99517 = ((double *) mem_103225)[tmp_99494 * (int64_t) 64 + i_101860 * (int64_t) 4 + tmp_99499];
                
                ((double *) mem_103443)[i_101851] = lifted_lambda_res_99517;
                ((double *) mem_103444)[i_101851] = lifted_lambda_res_99504;
            }
            // futhark/microgpt.fut:335:41-51
            
            double zp_lhs_95458 = ((double *) mem_102953)[i_101860];
            
            // futhark/microgpt.fut:335:41-79
            
            double zp_res_95459 = 1.0e-5 + zp_lhs_95458;
            
            // futhark/microgpt.fut:335:33-79
            
            double sqrt_res_95460 = futrts_sqrt64(zp_res_95459);
            
            ((double *) mem_103429)[i_101860] = sqrt_res_95460;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103430, i_101860 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103443, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103431, i_101860 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103444, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101875 = 0; i_101875 < (int64_t) 16; i_101875++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101868 = 0; i_101868 < (int64_t) 16; i_101868++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_99537;
                double r_99539 = 0.0;
                
                for (int64_t i_99538 = 0; i_99538 < (int64_t) 16; i_99538++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_99540 = ((double *) mem_param_102652.mem)[i_101868 * (int64_t) 16 + i_99538];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_99541 = ((double *) mem_103431)[i_101875 * (int64_t) 16 + i_99538];
                    
                    // futhark/microgpt.fut:221:73-105
                    
                    double zt_res_99542 = zt_lhs_99540 * zt_rhs_99541;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_99543 = r_99539 + zt_res_99542;
                    double r_tmp_105086 = zp_res_99543;
                    
                    r_99539 = r_tmp_105086;
                }
                defunc_0_lifted_lambda_res_99537 = r_99539;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_99550;
                double r_99552 = 0.0;
                
                for (int64_t i_99551 = 0; i_99551 < (int64_t) 16; i_99551++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_99553 = ((double *) mem_param_102652.mem)[i_101868 * (int64_t) 16 + i_99551];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_99554 = ((double *) mem_103430)[i_101875 * (int64_t) 16 + i_99551];
                    
                    // futhark/microgpt.fut:284:69-101
                    
                    double zt_res_99555 = zt_lhs_99553 * zt_rhs_99554;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_99556 = r_99552 + zt_res_99555;
                    double r_tmp_105087 = zp_res_99556;
                    
                    r_99552 = r_tmp_105087;
                }
                defunc_0_lifted_lambda_res_99550 = r_99552;
                ((double *) mem_103478)[i_101868] = defunc_0_lifted_lambda_res_99550;
                ((double *) mem_103479)[i_101868] = defunc_0_lifted_lambda_res_99537;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103468, i_101875 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103478, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103469, i_101875 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103479, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101889 = 0; i_101889 < (int64_t) 16; i_101889++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101882 = 0; i_101882 < (int64_t) 16; i_101882++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_99576 = ((double *) mem_103469)[i_101889 * (int64_t) 16 + i_101882];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_99577 = ((double *) mem_102837)[i_101889 * (int64_t) 16 + i_101882];
                
                // futhark/microgpt.fut:222:42-72
                
                double zp_res_99578 = zp_lhs_99576 + zp_rhs_99577;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_99585 = ((double *) mem_103468)[i_101889 * (int64_t) 16 + i_101882];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_99586 = ((double *) mem_102836)[i_101889 * (int64_t) 16 + i_101882];
                
                // futhark/microgpt.fut:285:38-68
                
                double zp_res_99587 = zp_lhs_99585 + zp_rhs_99586;
                
                ((double *) mem_103510)[i_101882] = zp_res_99587;
                ((double *) mem_103511)[i_101882] = zp_res_99578;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103500, i_101889 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103510, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103501, i_101889 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103511, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101912 = 0; i_101912 < (int64_t) 16; i_101912++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_95681;
            double r_95683 = 0.0;
            
            for (int64_t i_95682 = 0; i_95682 < (int64_t) 16; i_95682++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_95684 = ((double *) mem_103500)[i_101912 * (int64_t) 16 + i_95682];
                
                // futhark/microgpt.fut:286:62-93
                
                double zt_res_95685 = zt_lhs_95684 * zt_lhs_95684;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_95686 = r_95683 + zt_res_95685;
                double r_tmp_105095 = zp_res_95686;
                
                r_95683 = r_tmp_105095;
            }
            defunc_0_lifted_lambda_res_95681 = r_95683;
            // futhark/microgpt.fut:286:43-111
            
            double zs_res_95687 = defunc_0_lifted_lambda_res_95681 / 16.0;
            
            // futhark/microgpt.fut:287:24-55
            
            double zp_res_95688 = 1.0e-5 + zs_res_95687;
            
            // futhark/microgpt.fut:287:16-55
            
            double sqrt_res_95689 = futrts_sqrt64(zp_res_95688);
            
            // futhark/microgpt.fut:288:43-54
            
            double zs_res_95690 = 1.0 / sqrt_res_95689;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101898 = 0; i_101898 < (int64_t) 16; i_101898++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_99640 = ((double *) mem_103501)[i_101912 * (int64_t) 16 + i_101898];
                
                // futhark/microgpt.fut:223:65-96
                
                double zt_res_99641 = zt_lhs_99640 * zt_lhs_99640;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_99648 = ((double *) mem_103500)[i_101912 * (int64_t) 16 + i_101898];
                
                // futhark/microgpt.fut:288:24-54
                
                double zt_res_99649 = zs_res_95690 * zt_lhs_99648;
                
                // futhark/microgpt.fut:308:53-88
                
                double zt_res_99660 = zt_lhs_99648 * zt_lhs_99648;
                
                ((double *) mem_103547)[i_101898] = zt_res_99660;
                ((double *) mem_103548)[i_101898] = zt_res_99649;
                ((double *) mem_103549)[i_101898] = zt_res_99641;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_95657;
            double r_95659 = 0.0;
            
            for (int64_t i_95658 = 0; i_95658 < (int64_t) 16; i_95658++) {
                // futhark/microgpt.fut:224:35-43
                
                double lifted_lambda_res_95660 = ((double *) mem_103549)[i_95658];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_95661 = r_95659 + lifted_lambda_res_95660;
                double r_tmp_105099 = zp_res_95661;
                
                r_95659 = r_tmp_105099;
            }
            defunc_0_lifted_lambda_res_95657 = r_95659;
            // futhark/microgpt.fut:224:17-60
            
            double zs_res_95662 = defunc_0_lifted_lambda_res_95657 / 16.0;
            
            // futhark/microgpt.fut:225:24-55
            
            double zp_res_95663 = 1.0e-5 + zs_res_95662;
            
            // futhark/microgpt.fut:225:16-55
            
            double sqrt_res_95664 = futrts_sqrt64(zp_res_95663);
            
            // futhark/microgpt.fut:226:43-54
            
            double zs_res_95665 = 1.0 / sqrt_res_95664;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101904 = 0; i_101904 < (int64_t) 16; i_101904++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_95672 = ((double *) mem_103501)[i_101912 * (int64_t) 16 + i_101904];
                
                // futhark/microgpt.fut:226:24-54
                
                double zt_res_95673 = zs_res_95665 * zt_lhs_95672;
                
                ((double *) mem_103568)[i_101904] = zt_res_95673;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103532, i_101912 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103547, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103533, i_101912 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103548, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103534, i_101912 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103568, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101929 = 0; i_101929 < (int64_t) 16; i_101929++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101920 = 0; i_101920 < (int64_t) 64; i_101920++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_99683;
                double r_99685 = 0.0;
                
                for (int64_t i_99684 = 0; i_99684 < (int64_t) 16; i_99684++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_99686 = ((double *) mem_param_102668.mem)[i_101920 * (int64_t) 16 + i_99684];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_99687 = ((double *) mem_103534)[i_101929 * (int64_t) 16 + i_99684];
                    
                    // futhark/microgpt.fut:227:73-104
                    
                    double zt_res_99688 = zt_lhs_99686 * zt_rhs_99687;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_99689 = r_99685 + zt_res_99688;
                    double r_tmp_105106 = zp_res_99689;
                    
                    r_99685 = r_tmp_105106;
                }
                defunc_0_lifted_lambda_res_99683 = r_99685;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_99696;
                double r_99698 = 0.0;
                
                for (int64_t i_99697 = 0; i_99697 < (int64_t) 16; i_99697++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_99699 = ((double *) mem_param_102668.mem)[i_101920 * (int64_t) 16 + i_99697];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_99700 = ((double *) mem_103533)[i_101929 * (int64_t) 16 + i_99697];
                    
                    // futhark/microgpt.fut:289:69-100
                    
                    double zt_res_99701 = zt_lhs_99699 * zt_rhs_99700;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_99702 = r_99698 + zt_res_99701;
                    double r_tmp_105107 = zp_res_99702;
                    
                    r_99698 = r_tmp_105107;
                }
                defunc_0_lifted_lambda_res_99696 = r_99698;
                ((double *) mem_103601)[i_101920] = defunc_0_lifted_lambda_res_99696;
                ((double *) mem_103602)[i_101920] = defunc_0_lifted_lambda_res_99683;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_95836;
            double r_95838 = 0.0;
            
            for (int64_t i_95837 = 0; i_95837 < (int64_t) 16; i_95837++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_95839 = ((double *) mem_103532)[i_101929 * (int64_t) 16 + i_95837];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_95840 = r_95838 + lifted_lambda_res_95839;
                double r_tmp_105108 = zp_res_95840;
                
                r_95838 = r_tmp_105108;
            }
            defunc_0_lifted_lambda_res_95836 = r_95838;
            // futhark/microgpt.fut:309:34-86
            
            double zs_res_95841 = defunc_0_lifted_lambda_res_95836 / 16.0;
            
            ((double *) mem_103587)[i_101929] = zs_res_95841;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103588, i_101929 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103601, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103589, i_101929 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103602, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101946 = 0; i_101946 < (int64_t) 16; i_101946++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101937 = 0; i_101937 < (int64_t) 64; i_101937++) {
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_99722 = ((double *) mem_103589)[i_101946 * (int64_t) 64 + i_101937];
                
                // futhark/microgpt.fut:228:42-66
                
                double max_res_99723 = fmax64(0.0, max_arg0_99722);
                
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_99730 = ((double *) mem_103588)[i_101946 * (int64_t) 64 + i_101937];
                
                // futhark/microgpt.fut:290:38-62
                
                double max_res_99731 = fmax64(0.0, max_arg0_99730);
                
                ((double *) mem_103640)[i_101937] = max_res_99731;
                ((double *) mem_103641)[i_101937] = max_res_99723;
            }
            // futhark/microgpt.fut:310:41-51
            
            double zp_lhs_95940 = ((double *) mem_103587)[i_101946];
            
            // futhark/microgpt.fut:310:41-79
            
            double zp_res_95941 = 1.0e-5 + zp_lhs_95940;
            
            // futhark/microgpt.fut:310:33-79
            
            double sqrt_res_95942 = futrts_sqrt64(zp_res_95941);
            
            ((double *) mem_103626)[i_101946] = sqrt_res_95942;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103627, i_101946 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103640, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103628, i_101946 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103641, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101961 = 0; i_101961 < (int64_t) 16; i_101961++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101954 = 0; i_101954 < (int64_t) 16; i_101954++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_99751;
                double r_99753 = 0.0;
                
                for (int64_t i_99752 = 0; i_99752 < (int64_t) 64; i_99752++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_99754 = ((double *) mem_param_102644.mem)[i_101954 * (int64_t) 64 + i_99752];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_99755 = ((double *) mem_103628)[i_101961 * (int64_t) 64 + i_99752];
                    
                    // futhark/microgpt.fut:229:73-106
                    
                    double zt_res_99756 = zt_lhs_99754 * zt_rhs_99755;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_99757 = r_99753 + zt_res_99756;
                    double r_tmp_105118 = zp_res_99757;
                    
                    r_99753 = r_tmp_105118;
                }
                defunc_0_lifted_lambda_res_99751 = r_99753;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_99764;
                double r_99766 = 0.0;
                
                for (int64_t i_99765 = 0; i_99765 < (int64_t) 64; i_99765++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_99767 = ((double *) mem_param_102644.mem)[i_101954 * (int64_t) 64 + i_99765];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_99768 = ((double *) mem_103627)[i_101961 * (int64_t) 64 + i_99765];
                    
                    // futhark/microgpt.fut:291:69-102
                    
                    double zt_res_99769 = zt_lhs_99767 * zt_rhs_99768;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_99770 = r_99766 + zt_res_99769;
                    double r_tmp_105119 = zp_res_99770;
                    
                    r_99766 = r_tmp_105119;
                }
                defunc_0_lifted_lambda_res_99764 = r_99766;
                ((double *) mem_103675)[i_101954] = defunc_0_lifted_lambda_res_99764;
                ((double *) mem_103676)[i_101954] = defunc_0_lifted_lambda_res_99751;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103665, i_101961 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103675, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103666, i_101961 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103676, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101975 = 0; i_101975 < (int64_t) 16; i_101975++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101968 = 0; i_101968 < (int64_t) 16; i_101968++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_99790 = ((double *) mem_103666)[i_101975 * (int64_t) 16 + i_101968];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_99791 = ((double *) mem_103501)[i_101975 * (int64_t) 16 + i_101968];
                
                // futhark/microgpt.fut:230:42-73
                
                double zp_res_99792 = zp_lhs_99790 + zp_rhs_99791;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_99799 = ((double *) mem_103665)[i_101975 * (int64_t) 16 + i_101968];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_99800 = ((double *) mem_103500)[i_101975 * (int64_t) 16 + i_101968];
                
                // futhark/microgpt.fut:292:38-69
                
                double zp_res_99801 = zp_lhs_99799 + zp_rhs_99800;
                
                ((double *) mem_103707)[i_101968] = zp_res_99801;
                ((double *) mem_103708)[i_101968] = zp_res_99792;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103697, i_101975 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103707, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103698, i_101975 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103708, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_101989 = 0; i_101989 < (int64_t) 16; i_101989++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_101982 = 0; i_101982 < (int64_t) 27; i_101982++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_99821;
                double r_99823 = 0.0;
                
                for (int64_t i_99822 = 0; i_99822 < (int64_t) 16; i_99822++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_99824 = ((double *) mem_param_102676.mem)[i_101982 * (int64_t) 16 + i_99822];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_99825 = ((double *) mem_103698)[i_101989 * (int64_t) 16 + i_99822];
                    
                    // futhark/microgpt.fut:231:73-105
                    
                    double zt_res_99826 = zt_lhs_99824 * zt_rhs_99825;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_99827 = r_99823 + zt_res_99826;
                    double r_tmp_105128 = zp_res_99827;
                    
                    r_99823 = r_tmp_105128;
                }
                defunc_0_lifted_lambda_res_99821 = r_99823;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_99834;
                double r_99836 = 0.0;
                
                for (int64_t i_99835 = 0; i_99835 < (int64_t) 16; i_99835++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_99837 = ((double *) mem_param_102676.mem)[i_101982 * (int64_t) 16 + i_99835];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_99838 = ((double *) mem_103697)[i_101989 * (int64_t) 16 + i_99835];
                    
                    // futhark/microgpt.fut:293:69-101
                    
                    double zt_res_99839 = zt_lhs_99837 * zt_rhs_99838;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_99840 = r_99836 + zt_res_99839;
                    double r_tmp_105129 = zp_res_99840;
                    
                    r_99836 = r_tmp_105129;
                }
                defunc_0_lifted_lambda_res_99834 = r_99836;
                ((double *) mem_103739)[i_101982] = defunc_0_lifted_lambda_res_99834;
                ((double *) mem_103740)[i_101982] = defunc_0_lifted_lambda_res_99821;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103729, i_101989 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103739, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103730, i_101989 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103740, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102032 = 0; i_102032 < (int64_t) 16; i_102032++) {
            // futhark/microgpt.fut:103:13-33
            
            double defunc_0_reduce_res_101409;
            double defunc_0_reduce_res_101410;
            double defunc_0_reduce_res_101411;
            double redout_101992;
            double redout_101993;
            double redout_101994;
            
            redout_101992 = -INFINITY;
            redout_101993 = -INFINITY;
            redout_101994 = -INFINITY;
            for (int64_t i_101995 = 0; i_101995 < (int64_t) 27; i_101995++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_99975 = ((double *) mem_103730)[i_102032 * (int64_t) 27 + i_101995];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_99985 = ((double *) mem_103729)[i_102032 * (int64_t) 27 + i_101995];
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_96248 = fmax64(lifted_lambda_res_99975, redout_101992);
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_96309 = fmax64(lifted_lambda_res_99985, redout_101993);
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_96364 = fmax64(lifted_lambda_res_99985, redout_101994);
                double redout_tmp_105133 = max_res_96248;
                double redout_tmp_105134 = max_res_96309;
                double redout_tmp_105135 = max_res_96364;
                
                redout_101992 = redout_tmp_105133;
                redout_101993 = redout_tmp_105134;
                redout_101994 = redout_tmp_105135;
            }
            defunc_0_reduce_res_101409 = redout_101992;
            defunc_0_reduce_res_101410 = redout_101993;
            defunc_0_reduce_res_101411 = redout_101994;
            // futhark/microgpt.fut:113:47-56
            
            double neg_res_96249 = -defunc_0_reduce_res_101409;
            
            // futhark/microgpt.fut:113:47-56
            
            double neg_res_96365 = -defunc_0_reduce_res_101411;
            
            // futhark/microgpt.fut:113:47-56
            
            double neg_res_96310 = -defunc_0_reduce_res_101410;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102002 = 0; i_102002 < (int64_t) 27; i_102002++) {
                // futhark/microgpt.fut:113:38-41
                
                double lifted_lambda_res_100070 = ((double *) mem_103730)[i_102032 * (int64_t) 27 + i_102002];
                
                // futhark/microgpt.fut:113:38-56
                
                double zp_res_100071 = neg_res_96249 + lifted_lambda_res_100070;
                
                // futhark/microgpt.fut:113:31-56
                
                double exp_res_100072 = futrts_exp64(zp_res_100071);
                
                // futhark/microgpt.fut:113:38-41
                
                double lifted_lambda_res_100079 = ((double *) mem_103729)[i_102032 * (int64_t) 27 + i_102002];
                
                // futhark/microgpt.fut:113:38-56
                
                double zp_res_100080 = neg_res_96310 + lifted_lambda_res_100079;
                
                // futhark/microgpt.fut:113:31-56
                
                double exp_res_100081 = futrts_exp64(zp_res_100080);
                
                // futhark/microgpt.fut:113:38-56
                
                double zp_res_100092 = neg_res_96365 + lifted_lambda_res_100079;
                
                // futhark/microgpt.fut:113:31-56
                
                double exp_res_100093 = futrts_exp64(zp_res_100092);
                
                ((double *) mem_103775)[i_102002] = exp_res_100093;
                ((double *) mem_103776)[i_102002] = exp_res_100081;
                ((double *) mem_103777)[i_102002] = exp_res_100072;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_96260;
            double r_96262 = 0.0;
            
            for (int64_t i_96261 = 0; i_96261 < (int64_t) 27; i_96261++) {
                // futhark/microgpt.fut:114:32-39
                
                double lifted_lambda_res_96263 = ((double *) mem_103777)[i_96261];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_96264 = r_96262 + lifted_lambda_res_96263;
                double r_tmp_105139 = zp_res_96264;
                
                r_96262 = r_tmp_105139;
            }
            defunc_0_lifted_lambda_res_96260 = r_96262;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_96376;
            double r_96378 = 0.0;
            
            for (int64_t i_96377 = 0; i_96377 < (int64_t) 27; i_96377++) {
                // futhark/microgpt.fut:114:32-39
                
                double lifted_lambda_res_96379 = ((double *) mem_103775)[i_96377];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_96380 = r_96378 + lifted_lambda_res_96379;
                double r_tmp_105140 = zp_res_96380;
                
                r_96378 = r_tmp_105140;
            }
            defunc_0_lifted_lambda_res_96376 = r_96378;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_96321;
            double r_96323 = 0.0;
            
            for (int64_t i_96322 = 0; i_96322 < (int64_t) 27; i_96322++) {
                // futhark/microgpt.fut:114:32-39
                
                double lifted_lambda_res_96324 = ((double *) mem_103776)[i_96322];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_96325 = r_96323 + lifted_lambda_res_96324;
                double r_tmp_105141 = zp_res_96325;
                
                r_96323 = r_tmp_105141;
            }
            defunc_0_lifted_lambda_res_96321 = r_96323;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102012 = 0; i_102012 < (int64_t) 27; i_102012++) {
                // futhark/microgpt.fut:115:23-30
                
                double zs_lhs_100145 = ((double *) mem_103777)[i_102012];
                
                // futhark/microgpt.fut:115:23-40
                
                double zs_res_100146 = zs_lhs_100145 / defunc_0_lifted_lambda_res_96260;
                
                // futhark/microgpt.fut:115:23-30
                
                double zs_lhs_100153 = ((double *) mem_103776)[i_102012];
                
                // futhark/microgpt.fut:115:23-40
                
                double zs_res_100154 = zs_lhs_100153 / defunc_0_lifted_lambda_res_96321;
                
                // futhark/microgpt.fut:115:23-30
                
                double zs_lhs_100164 = ((double *) mem_103775)[i_102012];
                
                // futhark/microgpt.fut:115:23-40
                
                double zs_res_100165 = zs_lhs_100164 / defunc_0_lifted_lambda_res_96376;
                
                ((double *) mem_103796)[i_102012] = zs_res_100165;
                ((double *) mem_103797)[i_102012] = zs_res_100154;
                ((double *) mem_103798)[i_102012] = zs_res_100146;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102022 = 0; i_102022 < (int64_t) 27; i_102022++) {
                // futhark/microgpt.fut:233:4-14
                
                double log_arg0_100216 = ((double *) mem_103798)[i_102022];
                
                // futhark/microgpt.fut:232:66-233:14
                
                double log_res_100217 = futrts_log64(log_arg0_100216);
                
                // futhark/microgpt.fut:299:24-34
                
                double lifted_lambda_res_100224 = ((double *) mem_103797)[i_102022];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_100234 = ((double *) mem_102755)[i_102032 * (int64_t) 27 + i_102022];
                
                // futhark/microgpt.fut:301:4-14
                
                double zs_rhs_100235 = ((double *) mem_103796)[i_102022];
                
                // futhark/microgpt.fut:300:74-301:14
                
                double zs_res_100236 = 1.0 / zs_rhs_100235;
                
                // futhark/microgpt.fut:300:53-301:14
                
                double zt_res_100237 = zt_lhs_100234 * zs_res_100236;
                
                ((double *) mem_103817)[i_102022] = zt_res_100237;
                ((double *) mem_103818)[i_102022] = lifted_lambda_res_100224;
                ((double *) mem_103819)[i_102022] = log_res_100217;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_96283;
            double r_96285 = 0.0;
            
            for (int64_t i_96284 = 0; i_96284 < (int64_t) 27; i_96284++) {
                // futhark/microgpt.fut:234:32-41
                
                double zt_lhs_96286 = ((double *) mem_103819)[i_96284];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_96287 = ((double *) ext_mem_102754.mem)[i_102032 * (int64_t) 27 + i_96284];
                
                // futhark/microgpt.fut:234:32-63
                
                double zt_res_96288 = zt_lhs_96286 * zt_rhs_96287;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_96289 = r_96285 + zt_res_96288;
                double r_tmp_105148 = zp_res_96289;
                
                r_96285 = r_tmp_105148;
            }
            defunc_0_lifted_lambda_res_96283 = r_96285;
            // futhark/microgpt.fut:234:5-65
            
            double neg_res_96290 = -defunc_0_lifted_lambda_res_96283;
            
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103761, i_102032 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103817, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103762, i_102032 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103818, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            ((double *) mem_103763)[i_102032] = neg_res_96290;
        }
        if (memblock_unref(ctx, &ext_mem_102754, "ext_mem_102754") != 0)
            return 1;
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_90847;
        double r_90849 = 0.0;
        
        for (int64_t i_90848 = 0; i_90848 < (int64_t) 16; i_90848++) {
            // futhark/microgpt.fut:235:24-32
            
            double lifted_lambda_res_90850 = ((double *) mem_103763)[i_90848];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_90851 = r_90849 + lifted_lambda_res_90850;
            double r_tmp_105149 = zp_res_90851;
            
            r_90849 = r_tmp_105149;
        }
        defunc_0_lifted_lambda_res_90847 = r_90849;
        // futhark/microgpt.fut:235:6-49
        
        double zs_res_90852 = defunc_0_lifted_lambda_res_90847 / 16.0;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102038 = 0; i_102038 < (int64_t) 16; i_102038++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_91497;
            double r_91499 = 0.0;
            
            for (int64_t i_91498 = 0; i_91498 < (int64_t) 27; i_91498++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_91500 = ((double *) mem_103761)[i_102038 * (int64_t) 27 + i_91498];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_91501 = ((double *) mem_103762)[i_102038 * (int64_t) 27 + i_91498];
                
                // futhark/microgpt.fut:302:53-90
                
                double zt_res_91502 = zt_lhs_91500 * zt_rhs_91501;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_91503 = r_91499 + zt_res_91502;
                double r_tmp_105151 = zp_res_91503;
                
                r_91499 = r_tmp_105151;
            }
            defunc_0_lifted_lambda_res_91497 = r_91499;
            ((double *) mem_103849)[i_102038] = defunc_0_lifted_lambda_res_91497;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102046 = 0; i_102046 < (int64_t) 16; i_102046++) {
            // futhark/microgpt.fut:303:103-113
            
            double neg_arg0_91511 = ((double *) mem_103849)[i_102046];
            
            // futhark/microgpt.fut:303:97-113
            
            double neg_res_91512 = -neg_arg0_91511;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102042 = 0; i_102042 < (int64_t) 27; i_102042++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_91519 = ((double *) mem_103762)[i_102046 * (int64_t) 27 + i_102042];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_91520 = ((double *) mem_103761)[i_102046 * (int64_t) 27 + i_102042];
                
                // futhark/microgpt.fut:303:75-113
                
                double zp_res_91521 = neg_res_91512 + zp_lhs_91520;
                
                // futhark/microgpt.fut:303:53-113
                
                double zt_res_91522 = zt_lhs_91519 * zp_res_91521;
                
                ((double *) mem_103861)[i_102042] = zt_res_91522;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103856, i_102046 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103861, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102054 = 0; i_102054 < (int64_t) 16; i_102054++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102050 = 0; i_102050 < (int64_t) 16; i_102050++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_91537;
                double r_91539 = 0.0;
                
                for (int64_t i_91538 = 0; i_91538 < (int64_t) 27; i_91538++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_91540 = ((double *) mem_param_102676.mem)[i_91538 * (int64_t) 16 + i_102050];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_91541 = ((double *) mem_103856)[i_102054 * (int64_t) 27 + i_91538];
                    
                    // futhark/microgpt.fut:304:73-110
                    
                    double zt_res_91542 = zt_lhs_91540 * zt_rhs_91541;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_91543 = r_91539 + zt_res_91542;
                    double r_tmp_105156 = zp_res_91543;
                    
                    r_91539 = r_tmp_105156;
                }
                defunc_0_lifted_lambda_res_91537 = r_91539;
                ((double *) mem_103877)[i_102050] = defunc_0_lifted_lambda_res_91537;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103872, i_102054 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103877, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102067 = 0; i_102067 < (int64_t) 16; i_102067++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102060 = 0; i_102060 < (int64_t) 64; i_102060++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_100270;
                double r_100272 = 0.0;
                
                for (int64_t i_100271 = 0; i_100271 < (int64_t) 16; i_100271++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_100273 = ((double *) mem_param_102644.mem)[i_100271 * (int64_t) 64 + i_102060];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_100274 = ((double *) mem_103872)[i_102067 * (int64_t) 16 + i_100271];
                    
                    // futhark/microgpt.fut:305:73-111
                    
                    double zt_res_100275 = zt_lhs_100273 * zt_rhs_100274;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_100276 = r_100272 + zt_res_100275;
                    double r_tmp_105161 = zp_res_100276;
                    
                    r_100272 = r_tmp_105161;
                }
                defunc_0_lifted_lambda_res_100270 = r_100272;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_100283;
                double r_100285 = 0.0;
                
                for (int64_t i_100284 = 0; i_100284 < (int64_t) 16; i_100284++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_100286 = ((double *) mem_103872)[i_100284 * (int64_t) 16 + i_102067];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_100287 = ((double *) mem_103627)[i_100284 * (int64_t) 64 + i_102060];
                    
                    // futhark/microgpt.fut:355:75-111
                    
                    double zt_res_100288 = zt_lhs_100286 * zt_rhs_100287;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_100289 = r_100285 + zt_res_100288;
                    double r_tmp_105162 = zp_res_100289;
                    
                    r_100285 = r_tmp_105162;
                }
                defunc_0_lifted_lambda_res_100283 = r_100285;
                ((double *) mem_103898)[i_102060] = defunc_0_lifted_lambda_res_100283;
                ((double *) mem_103899)[i_102060] = defunc_0_lifted_lambda_res_100270;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103888, i_102067 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103898, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103889, i_102067 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103899, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102076 = 0; i_102076 < (int64_t) 16; i_102076++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102072 = 0; i_102072 < (int64_t) 64; i_102072++) {
                // futhark/microgpt.fut:4:11-25
                
                double indicatorp_arg0_91579 = ((double *) mem_103588)[i_102076 * (int64_t) 64 + i_102072];
                
                // futhark/microgpt.fut:125:42-54
                
                double max_res_91580 = fmax64(0.0, indicatorp_arg0_91579);
                
                // futhark/microgpt.fut:125:35-54
                
                double sgn_res_91581 = fsignum64(max_res_91580);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_91582 = ((double *) mem_103889)[i_102076 * (int64_t) 64 + i_102072];
                
                // futhark/microgpt.fut:306:42-90
                
                double zt_res_91583 = sgn_res_91581 * zt_rhs_91582;
                
                ((double *) mem_103925)[i_102072] = zt_res_91583;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103920, i_102076 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103925, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102084 = 0; i_102084 < (int64_t) 16; i_102084++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102080 = 0; i_102080 < (int64_t) 16; i_102080++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_91598;
                double r_91600 = 0.0;
                
                for (int64_t i_91599 = 0; i_91599 < (int64_t) 64; i_91599++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_91601 = ((double *) mem_param_102668.mem)[i_91599 * (int64_t) 16 + i_102080];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_91602 = ((double *) mem_103920)[i_102084 * (int64_t) 64 + i_91599];
                    
                    // futhark/microgpt.fut:307:73-109
                    
                    double zt_res_91603 = zt_lhs_91601 * zt_rhs_91602;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_91604 = r_91600 + zt_res_91603;
                    double r_tmp_105167 = zp_res_91604;
                    
                    r_91600 = r_tmp_105167;
                }
                defunc_0_lifted_lambda_res_91598 = r_91600;
                ((double *) mem_103941)[i_102080] = defunc_0_lifted_lambda_res_91598;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103936, i_102084 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103941, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102088 = 0; i_102088 < (int64_t) 16; i_102088++) {
            // futhark/microgpt.fut:311:49-59
            
            double zs_rhs_91652 = ((double *) mem_103626)[i_102088];
            
            // futhark/microgpt.fut:311:41-59
            
            double zs_res_91653 = 1.0 / zs_rhs_91652;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_91654;
            double r_91656 = 0.0;
            
            for (int64_t i_91655 = 0; i_91655 < (int64_t) 16; i_91655++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_91657 = ((double *) mem_103500)[i_102088 * (int64_t) 16 + i_91655];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_91658 = ((double *) mem_103936)[i_102088 * (int64_t) 16 + i_91655];
                
                // futhark/microgpt.fut:311:87-123
                
                double zt_res_91659 = zt_lhs_91657 * zt_rhs_91658;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_91660 = r_91656 + zt_res_91659;
                double r_tmp_105169 = zp_res_91660;
                
                r_91656 = r_tmp_105169;
            }
            defunc_0_lifted_lambda_res_91654 = r_91656;
            // futhark/microgpt.fut:311:67-150
            
            double zt_res_91661 = zs_res_91653 * defunc_0_lifted_lambda_res_91654;
            
            // futhark/microgpt.fut:311:45-150
            
            double zt_res_91662 = zs_res_91653 * zt_res_91661;
            
            // futhark/microgpt.fut:311:33-150
            
            double neg_res_91663 = -zt_res_91662;
            
            ((double *) mem_103952)[i_102088] = neg_res_91663;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102092 = 0; i_102092 < (int64_t) 16; i_102092++) {
            // futhark/microgpt.fut:312:33-43
            
            double zt_lhs_91671 = ((double *) mem_103952)[i_102092];
            
            // futhark/microgpt.fut:312:85-95
            
            double zp_lhs_91672 = ((double *) mem_103587)[i_102092];
            
            // futhark/microgpt.fut:312:85-123
            
            double zp_res_91673 = 1.0e-5 + zp_lhs_91672;
            
            // futhark/microgpt.fut:312:77-123
            
            double sqrt_res_91674 = futrts_sqrt64(zp_res_91673);
            
            // futhark/microgpt.fut:312:63-125
            
            double zt_res_91675 = 2.0 * sqrt_res_91674;
            
            // futhark/microgpt.fut:312:49-125
            
            double zs_res_91676 = 1.0 / zt_res_91675;
            
            // futhark/microgpt.fut:312:33-125
            
            double zt_res_91677 = zt_lhs_91671 * zs_res_91676;
            
            ((double *) mem_103959)[i_102092] = zt_res_91677;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102096 = 0; i_102096 < (int64_t) 16; i_102096++) {
            // futhark/microgpt.fut:313:53-63
            
            double zs_lhs_91685 = ((double *) mem_103959)[i_102096];
            
            // futhark/microgpt.fut:313:53-78
            
            double zs_res_91686 = zs_lhs_91685 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_105172 = 0; nest_i_105172 < (int64_t) 16; nest_i_105172++) {
                ((double *) mem_103966)[i_102096 * (int64_t) 16 + nest_i_105172] = zs_res_91686;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102104 = 0; i_102104 < (int64_t) 16; i_102104++) {
            // futhark/microgpt.fut:314:107-117
            
            double zs_rhs_91695 = ((double *) mem_103626)[i_102104];
            
            // futhark/microgpt.fut:314:99-117
            
            double zs_res_91696 = 1.0 / zs_rhs_91695;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102100 = 0; i_102100 < (int64_t) 16; i_102100++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_91703 = ((double *) mem_103872)[i_102104 * (int64_t) 16 + i_102100];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_91704 = ((double *) mem_103936)[i_102104 * (int64_t) 16 + i_102100];
                
                // futhark/microgpt.fut:314:77-117
                
                double zt_res_91705 = zs_res_91696 * zt_lhs_91704;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_91706 = ((double *) mem_103500)[i_102104 * (int64_t) 16 + i_102100];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_91707 = ((double *) mem_103966)[i_102104 * (int64_t) 16 + i_102100];
                
                // futhark/microgpt.fut:314:125-161
                
                double zt_res_91708 = zt_lhs_91706 * zt_rhs_91707;
                
                // futhark/microgpt.fut:314:94-161
                
                double zp_res_91709 = zt_res_91705 + zt_res_91708;
                
                // futhark/microgpt.fut:314:120-205
                
                double zp_res_91710 = zt_res_91708 + zp_res_91709;
                
                // futhark/microgpt.fut:314:53-205
                
                double zp_res_91711 = zp_lhs_91703 + zp_res_91710;
                
                ((double *) mem_103981)[i_102100] = zp_res_91711;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103976, i_102104 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_103981, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102117 = 0; i_102117 < (int64_t) 16; i_102117++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102110 = 0; i_102110 < (int64_t) 16; i_102110++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_100312;
                double r_100314 = 0.0;
                
                for (int64_t i_100313 = 0; i_100313 < (int64_t) 16; i_100313++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_100315 = ((double *) mem_param_102652.mem)[i_100313 * (int64_t) 16 + i_102110];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_100316 = ((double *) mem_103976)[i_102117 * (int64_t) 16 + i_100313];
                    
                    // futhark/microgpt.fut:315:73-110
                    
                    double zt_res_100317 = zt_lhs_100315 * zt_rhs_100316;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_100318 = r_100314 + zt_res_100317;
                    double r_tmp_105179 = zp_res_100318;
                    
                    r_100314 = r_tmp_105179;
                }
                defunc_0_lifted_lambda_res_100312 = r_100314;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_100325;
                double r_100327 = 0.0;
                
                for (int64_t i_100326 = 0; i_100326 < (int64_t) 16; i_100326++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_100328 = ((double *) mem_103976)[i_100326 * (int64_t) 16 + i_102117];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_100329 = ((double *) mem_103430)[i_100326 * (int64_t) 16 + i_102110];
                    
                    // futhark/microgpt.fut:353:74-110
                    
                    double zt_res_100330 = zt_lhs_100328 * zt_rhs_100329;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_100331 = r_100327 + zt_res_100330;
                    double r_tmp_105180 = zp_res_100331;
                    
                    r_100327 = r_tmp_105180;
                }
                defunc_0_lifted_lambda_res_100325 = r_100327;
                ((double *) mem_104002)[i_102110] = defunc_0_lifted_lambda_res_100325;
                ((double *) mem_104003)[i_102110] = defunc_0_lifted_lambda_res_100312;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103992, i_102117 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104002, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_103993, i_102117 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104003, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102139 = 0; i_102139 < (int64_t) 4; i_102139++) {
            // futhark/microgpt.fut:316:88-91
            
            int64_t zp_lhs_96517 = mul64((int64_t) 4, i_102139);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102132 = 0; i_102132 < (int64_t) 16; i_102132++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_102122 = 0; i_102122 < (int64_t) 4; i_102122++) {
                    // futhark/microgpt.fut:316:93-99
                    
                    int64_t tmp_100353 = add64(zp_lhs_96517, i_102122);
                    
                    // futhark/microgpt.fut:316:70-101
                    
                    bool x_100354 = sle64((int64_t) 0, tmp_100353);
                    
                    // futhark/microgpt.fut:316:70-101
                    
                    bool y_100355 = slt64(tmp_100353, (int64_t) 16);
                    
                    // futhark/microgpt.fut:316:70-101
                    
                    bool bounds_check_100356 = x_100354 && y_100355;
                    
                    // futhark/microgpt.fut:316:70-101
                    
                    bool index_certs_100357;
                    
                    if (!bounds_check_100356) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_100353, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:316:70-101\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:316:52-102\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:316:32-104\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:6:13-17\n   #9  futhark/microgpt.fut:316:13-106\n   #10 futhark/microgpt.fut:474:5-76\n   #11 futhark/microgpt.fut:479:26-486:31\n   #12 futhark/microgpt.fut:503:35-76\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_100358 = ((double *) mem_103993)[i_102132 * (int64_t) 16 + tmp_100353];
                    
                    ((double *) mem_104046)[i_102122] = lifted_lambda_res_100358;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_102126 = 0; i_102126 < (int64_t) 16; i_102126++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_100372 = ((double *) mem_103224)[i_102139 * (int64_t) 256 + i_102132 * (int64_t) 16 + i_102126];
                    
                    // futhark/microgpt.fut:318:61-97
                    
                    double zs_res_100373 = zs_lhs_100372 / 2.0;
                    double zp_rhs_100374 = ((double *) masks_mem_102636.mem)[step_90216 * (int64_t) 256 + i_102132 * (int64_t) 16 + i_102126];
                    
                    // futhark/microgpt.fut:318:84-119
                    
                    double zp_res_100375 = zs_res_100373 + zp_rhs_100374;
                    
                    ((double *) mem_104053)[i_102126] = zp_res_100375;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_104036, i_102132 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104053, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_104037, i_102132 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104046, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_104024, i_102139 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_104036, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_104025, i_102139 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_104037, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102170 = 0; i_102170 < (int64_t) 4; i_102170++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102163 = 0; i_102163 < (int64_t) 16; i_102163++) {
                // futhark/microgpt.fut:4:11-25
                
                double defunc_0_reduce_res_101434;
                double redout_102143 = -INFINITY;
                
                for (int64_t i_102145 = 0; i_102145 < (int64_t) 16; i_102145++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_100501 = ((double *) mem_104024)[i_102170 * (int64_t) 256 + i_102163 * (int64_t) 16 + i_102145];
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_100512;
                    double r_100514 = 0.0;
                    
                    for (int64_t i_100513 = 0; i_100513 < (int64_t) 4; i_100513++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_100515 = ((double *) mem_104025)[i_102170 * (int64_t) 64 + i_102163 * (int64_t) 4 + i_100513];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_100516 = ((double *) mem_103062)[i_102170 * (int64_t) 64 + i_102145 * (int64_t) 4 + i_100513];
                        
                        // futhark/microgpt.fut:321:91-139
                        
                        double zt_res_100517 = zt_lhs_100515 * zt_rhs_100516;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_100518 = r_100514 + zt_res_100517;
                        double r_tmp_105193 = zp_res_100518;
                        
                        r_100514 = r_tmp_105193;
                    }
                    defunc_0_lifted_lambda_res_100512 = r_100514;
                    // futhark/microgpt.fut:103:13-33
                    
                    double max_res_100412 = fmax64(lifted_lambda_res_100501, redout_102143);
                    
                    ((double *) mem_104100)[i_102145] = defunc_0_lifted_lambda_res_100512;
                    
                    double redout_tmp_105191 = max_res_100412;
                    
                    redout_102143 = redout_tmp_105191;
                }
                defunc_0_reduce_res_101434 = redout_102143;
                // futhark/microgpt.fut:113:47-56
                
                double neg_res_100413 = -defunc_0_reduce_res_101434;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_102149 = 0; i_102149 < (int64_t) 16; i_102149++) {
                    // futhark/microgpt.fut:113:38-41
                    
                    double lifted_lambda_res_100420 = ((double *) mem_104024)[i_102170 * (int64_t) 256 + i_102163 * (int64_t) 16 + i_102149];
                    
                    // futhark/microgpt.fut:113:38-56
                    
                    double zp_res_100421 = neg_res_100413 + lifted_lambda_res_100420;
                    
                    // futhark/microgpt.fut:113:31-56
                    
                    double exp_res_100422 = futrts_exp64(zp_res_100421);
                    
                    ((double *) mem_104107)[i_102149] = exp_res_100422;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_100424;
                double r_100426 = 0.0;
                
                for (int64_t i_100425 = 0; i_100425 < (int64_t) 16; i_100425++) {
                    // futhark/microgpt.fut:114:32-39
                    
                    double lifted_lambda_res_100427 = ((double *) mem_104107)[i_100425];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_100428 = r_100426 + lifted_lambda_res_100427;
                    double r_tmp_105195 = zp_res_100428;
                    
                    r_100426 = r_tmp_105195;
                }
                defunc_0_lifted_lambda_res_100424 = r_100426;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_102153 = 0; i_102153 < (int64_t) 16; i_102153++) {
                    // futhark/microgpt.fut:115:23-30
                    
                    double zs_lhs_100435 = ((double *) mem_104107)[i_102153];
                    
                    // futhark/microgpt.fut:115:23-40
                    
                    double zs_res_100436 = zs_lhs_100435 / defunc_0_lifted_lambda_res_100424;
                    
                    ((double *) mem_104114)[i_102153] = zs_res_100436;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_102157 = 0; i_102157 < (int64_t) 16; i_102157++) {
                    // futhark/microgpt.fut:320:24-34
                    
                    double lifted_lambda_res_100444 = ((double *) mem_104114)[i_102157];
                    
                    ((double *) mem_104121)[i_102157] = lifted_lambda_res_100444;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_104090, i_102163 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104100, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_104091, i_102163 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104121, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_104078, i_102170 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_104090, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_104079, i_102170 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_104091, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102192 = 0; i_102192 < (int64_t) 4; i_102192++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102185 = 0; i_102185 < (int64_t) 16; i_102185++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_102175 = 0; i_102175 < (int64_t) 16; i_102175++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_100554 = ((double *) mem_104078)[i_102192 * (int64_t) 256 + i_102185 * (int64_t) 16 + i_102175];
                    
                    ((double *) mem_104168)[i_102175] = lifted_lambda_res_100554;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_102179 = 0; i_102179 < (int64_t) 4; i_102179++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_100568;
                    double r_100570 = 0.0;
                    
                    for (int64_t i_100569 = 0; i_100569 < (int64_t) 16; i_100569++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_100571 = ((double *) mem_104079)[i_102192 * (int64_t) 256 + i_100569 * (int64_t) 16 + i_102185];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_100572 = ((double *) mem_104025)[i_102192 * (int64_t) 64 + i_100569 * (int64_t) 4 + i_102179];
                        
                        // futhark/microgpt.fut:326:91-140
                        
                        double zt_res_100573 = zt_lhs_100571 * zt_rhs_100572;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_100574 = r_100570 + zt_res_100573;
                        double r_tmp_105204 = zp_res_100574;
                        
                        r_100570 = r_tmp_105204;
                    }
                    defunc_0_lifted_lambda_res_100568 = r_100570;
                    ((double *) mem_104175)[i_102179] = defunc_0_lifted_lambda_res_100568;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_104158, i_102185 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104175, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_104159, i_102185 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104168, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_104146, i_102192 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_104158, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_104147, i_102192 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_104159, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102201 = 0; i_102201 < (int64_t) 4; i_102201++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102197 = 0; i_102197 < (int64_t) 16; i_102197++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_91930;
                double r_91932 = 0.0;
                
                for (int64_t i_91931 = 0; i_91931 < (int64_t) 16; i_91931++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_91933 = ((double *) mem_104147)[i_102201 * (int64_t) 256 + i_102197 * (int64_t) 16 + i_91931];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_91934 = ((double *) mem_104079)[i_102201 * (int64_t) 256 + i_102197 * (int64_t) 16 + i_91931];
                    
                    // futhark/microgpt.fut:323:72-121
                    
                    double zt_res_91935 = zt_lhs_91933 * zt_rhs_91934;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_91936 = r_91932 + zt_res_91935;
                    double r_tmp_105207 = zp_res_91936;
                    
                    r_91932 = r_tmp_105207;
                }
                defunc_0_lifted_lambda_res_91930 = r_91932;
                ((double *) mem_104205)[i_102197] = defunc_0_lifted_lambda_res_91930;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_104200, i_102201 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104205, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102213 = 0; i_102213 < (int64_t) 4; i_102213++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102209 = 0; i_102209 < (int64_t) 16; i_102209++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_91951 = ((double *) mem_104200)[i_102213 * (int64_t) 16 + i_102209];
                
                // futhark/microgpt.fut:324:128-150
                
                double neg_res_91952 = -neg_arg0_91951;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_102205 = 0; i_102205 < (int64_t) 16; i_102205++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_91959 = ((double *) mem_104079)[i_102213 * (int64_t) 256 + i_102209 * (int64_t) 16 + i_102205];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_91960 = ((double *) mem_104147)[i_102213 * (int64_t) 256 + i_102209 * (int64_t) 16 + i_102205];
                    
                    // futhark/microgpt.fut:324:100-150
                    
                    double zp_res_91961 = neg_res_91952 + zp_lhs_91960;
                    
                    // futhark/microgpt.fut:324:72-150
                    
                    double zt_res_91962 = zt_lhs_91959 * zp_res_91961;
                    
                    ((double *) mem_104227)[i_102205] = zt_res_91962;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_104222, i_102209 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104227, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_104216, i_102213 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_104222, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102225 = 0; i_102225 < (int64_t) 4; i_102225++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102221 = 0; i_102221 < (int64_t) 16; i_102221++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_102217 = 0; i_102217 < (int64_t) 16; i_102217++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_91984 = ((double *) mem_104216)[i_102225 * (int64_t) 256 + i_102221 * (int64_t) 16 + i_102217];
                    
                    // futhark/microgpt.fut:325:60-96
                    
                    double zs_res_91985 = zs_lhs_91984 / 2.0;
                    
                    ((double *) mem_104254)[i_102217] = zs_res_91985;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_104249, i_102221 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104254, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_104243, i_102225 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_104249, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102245 = 0; i_102245 < (int64_t) 4; i_102245++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102238 = 0; i_102238 < (int64_t) 16; i_102238++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_102231 = 0; i_102231 < (int64_t) 4; i_102231++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_100655;
                    double r_100657 = 0.0;
                    
                    for (int64_t i_100656 = 0; i_100656 < (int64_t) 16; i_100656++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_100658 = ((double *) mem_103064)[i_102245 * (int64_t) 64 + i_100656 * (int64_t) 4 + i_102231];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_100659 = ((double *) mem_104243)[i_102245 * (int64_t) 256 + i_100656 * (int64_t) 16 + i_102238];
                        
                        // futhark/microgpt.fut:327:91-139
                        
                        double zt_res_100660 = zt_lhs_100658 * zt_rhs_100659;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_100661 = r_100657 + zt_res_100660;
                        double r_tmp_105220 = zp_res_100661;
                        
                        r_100657 = r_tmp_105220;
                    }
                    defunc_0_lifted_lambda_res_100655 = r_100657;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_100668;
                    double r_100670 = 0.0;
                    
                    for (int64_t i_100669 = 0; i_100669 < (int64_t) 16; i_100669++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_100671 = ((double *) mem_104243)[i_102245 * (int64_t) 256 + i_102238 * (int64_t) 16 + i_100669];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_100672 = ((double *) mem_103063)[i_102245 * (int64_t) 64 + i_100669 * (int64_t) 4 + i_102231];
                        
                        // futhark/microgpt.fut:328:91-139
                        
                        double zt_res_100673 = zt_lhs_100671 * zt_rhs_100672;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_100674 = r_100670 + zt_res_100673;
                        double r_tmp_105221 = zp_res_100674;
                        
                        r_100670 = r_tmp_105221;
                    }
                    defunc_0_lifted_lambda_res_100668 = r_100670;
                    ((double *) mem_104292)[i_102231] = defunc_0_lifted_lambda_res_100668;
                    ((double *) mem_104293)[i_102231] = defunc_0_lifted_lambda_res_100655;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_104282, i_102238 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104292, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_104283, i_102238 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104293, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_104270, i_102245 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_104282, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_104271, i_102245 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_104283, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102264 = 0; i_102264 < (int64_t) 16; i_102264++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102254 = 0; i_102254 < (int64_t) 16; i_102254++) {
                // futhark/microgpt.fut:329:63-66
                
                int64_t tmp_100737 = sdiv64(i_102254, (int64_t) 4);
                
                // futhark/microgpt.fut:329:52-68
                
                bool x_100738 = sle64((int64_t) 0, tmp_100737);
                
                // futhark/microgpt.fut:329:52-68
                
                bool y_100739 = slt64(tmp_100737, (int64_t) 4);
                
                // futhark/microgpt.fut:329:52-68
                
                bool bounds_check_100740 = x_100738 && y_100739;
                
                // futhark/microgpt.fut:329:52-68
                
                bool index_certs_100741;
                
                if (!bounds_check_100740) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_100737, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:329:52-68\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:329:33-87\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:329:13-89\n   #7  futhark/microgpt.fut:474:5-76\n   #8  futhark/microgpt.fut:479:26-486:31\n   #9  futhark/microgpt.fut:503:35-76\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:329:81-84
                
                int64_t tmp_100742 = smod64(i_102254, (int64_t) 4);
                
                // futhark/microgpt.fut:329:52-86
                
                bool x_100743 = sle64((int64_t) 0, tmp_100742);
                
                // futhark/microgpt.fut:329:52-86
                
                bool y_100744 = slt64(tmp_100742, (int64_t) 4);
                
                // futhark/microgpt.fut:329:52-86
                
                bool bounds_check_100745 = x_100743 && y_100744;
                
                // futhark/microgpt.fut:329:52-86
                
                bool index_certs_100746;
                
                if (!bounds_check_100745) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_100742, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:329:52-86\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:329:33-87\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:329:13-89\n   #7  futhark/microgpt.fut:474:5-76\n   #8  futhark/microgpt.fut:479:26-486:31\n   #9  futhark/microgpt.fut:503:35-76\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_100747 = ((double *) mem_104146)[tmp_100737 * (int64_t) 64 + i_102264 * (int64_t) 4 + tmp_100742];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_100760 = ((double *) mem_104271)[tmp_100737 * (int64_t) 64 + i_102264 * (int64_t) 4 + tmp_100742];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_100776 = ((double *) mem_104270)[tmp_100737 * (int64_t) 64 + i_102264 * (int64_t) 4 + tmp_100742];
                
                ((double *) mem_104339)[i_102254] = lifted_lambda_res_100776;
                ((double *) mem_104340)[i_102254] = lifted_lambda_res_100760;
                ((double *) mem_104341)[i_102254] = lifted_lambda_res_100747;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_104324, i_102264 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104339, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_104325, i_102264 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104340, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_104326, i_102264 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104341, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102289 = 0; i_102289 < (int64_t) 16; i_102289++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102276 = 0; i_102276 < (int64_t) 16; i_102276++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_100939;
                double r_100941 = 0.0;
                
                for (int64_t i_100940 = 0; i_100940 < (int64_t) 16; i_100940++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_100942 = ((double *) mem_param_102672.mem)[i_100940 * (int64_t) 16 + i_102276];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_100943 = ((double *) mem_104326)[i_102289 * (int64_t) 16 + i_100940];
                    
                    // futhark/microgpt.fut:332:75-112
                    
                    double zt_res_100944 = zt_lhs_100942 * zt_rhs_100943;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_100945 = r_100941 + zt_res_100944;
                    double r_tmp_105236 = zp_res_100945;
                    
                    r_100941 = r_tmp_105236;
                }
                defunc_0_lifted_lambda_res_100939 = r_100941;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_100946;
                double r_100948 = 0.0;
                
                for (int64_t i_100947 = 0; i_100947 < (int64_t) 16; i_100947++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_100949 = ((double *) mem_param_102648.mem)[i_100947 * (int64_t) 16 + i_102276];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_100950 = ((double *) mem_104325)[i_102289 * (int64_t) 16 + i_100947];
                    
                    // futhark/microgpt.fut:332:141-178
                    
                    double zt_res_100951 = zt_lhs_100949 * zt_rhs_100950;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_100952 = r_100948 + zt_res_100951;
                    double r_tmp_105237 = zp_res_100952;
                    
                    r_100948 = r_tmp_105237;
                }
                defunc_0_lifted_lambda_res_100946 = r_100948;
                // futhark/microgpt.fut:332:55-180
                
                double zp_res_100953 = defunc_0_lifted_lambda_res_100939 + defunc_0_lifted_lambda_res_100946;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_100954;
                double r_100956 = 0.0;
                
                for (int64_t i_100955 = 0; i_100955 < (int64_t) 16; i_100955++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_100957 = ((double *) mem_param_102660.mem)[i_100955 * (int64_t) 16 + i_102276];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_100958 = ((double *) mem_104324)[i_102289 * (int64_t) 16 + i_100955];
                    
                    // futhark/microgpt.fut:332:208-245
                    
                    double zt_res_100959 = zt_lhs_100957 * zt_rhs_100958;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_100960 = r_100956 + zt_res_100959;
                    double r_tmp_105238 = zp_res_100960;
                    
                    r_100956 = r_tmp_105238;
                }
                defunc_0_lifted_lambda_res_100954 = r_100956;
                // futhark/microgpt.fut:332:116-247
                
                double zp_res_100961 = zp_res_100953 + defunc_0_lifted_lambda_res_100954;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_100968;
                double r_100970 = 0.0;
                
                for (int64_t i_100969 = 0; i_100969 < (int64_t) 16; i_100969++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_100971 = ((double *) mem_104324)[i_100969 * (int64_t) 16 + i_102289];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_100972 = ((double *) mem_102892)[i_100969 * (int64_t) 16 + i_102276];
                    
                    // futhark/microgpt.fut:350:74-109
                    
                    double zt_res_100973 = zt_lhs_100971 * zt_rhs_100972;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_100974 = r_100970 + zt_res_100973;
                    double r_tmp_105239 = zp_res_100974;
                    
                    r_100970 = r_tmp_105239;
                }
                defunc_0_lifted_lambda_res_100968 = r_100970;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_100984;
                double r_100986 = 0.0;
                
                for (int64_t i_100985 = 0; i_100985 < (int64_t) 16; i_100985++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_100987 = ((double *) mem_104325)[i_100985 * (int64_t) 16 + i_102289];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_100988 = ((double *) mem_102892)[i_100985 * (int64_t) 16 + i_102276];
                    
                    // futhark/microgpt.fut:351:74-109
                    
                    double zt_res_100989 = zt_lhs_100987 * zt_rhs_100988;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_100990 = r_100986 + zt_res_100989;
                    double r_tmp_105240 = zp_res_100990;
                    
                    r_100986 = r_tmp_105240;
                }
                defunc_0_lifted_lambda_res_100984 = r_100986;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_101002;
                double r_101004 = 0.0;
                
                for (int64_t i_101003 = 0; i_101003 < (int64_t) 16; i_101003++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_101005 = ((double *) mem_104326)[i_101003 * (int64_t) 16 + i_102289];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_101006 = ((double *) mem_102892)[i_101003 * (int64_t) 16 + i_102276];
                    
                    // futhark/microgpt.fut:352:74-109
                    
                    double zt_res_101007 = zt_lhs_101005 * zt_rhs_101006;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_101008 = r_101004 + zt_res_101007;
                    double r_tmp_105241 = zp_res_101008;
                    
                    r_101004 = r_tmp_105241;
                }
                defunc_0_lifted_lambda_res_101002 = r_101004;
                ((double *) mem_104392)[i_102276] = defunc_0_lifted_lambda_res_101002;
                ((double *) mem_104393)[i_102276] = defunc_0_lifted_lambda_res_100984;
                ((double *) mem_104394)[i_102276] = defunc_0_lifted_lambda_res_100968;
                ((double *) mem_104395)[i_102276] = zp_res_100961;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_104372, i_102289 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104392, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_104373, i_102289 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104393, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_104374, i_102289 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104394, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_104375, i_102289 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104395, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102296 = 0; i_102296 < (int64_t) 16; i_102296++) {
            // futhark/microgpt.fut:336:49-59
            
            double zs_rhs_92218 = ((double *) mem_103429)[i_102296];
            
            // futhark/microgpt.fut:336:41-59
            
            double zs_res_92219 = 1.0 / zs_rhs_92218;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_92220;
            double r_92222 = 0.0;
            
            for (int64_t i_92221 = 0; i_92221 < (int64_t) 16; i_92221++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_92223 = ((double *) mem_102836)[i_102296 * (int64_t) 16 + i_92221];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_92224 = ((double *) mem_104375)[i_102296 * (int64_t) 16 + i_92221];
                
                // futhark/microgpt.fut:336:87-122
                
                double zt_res_92225 = zt_lhs_92223 * zt_rhs_92224;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_92226 = r_92222 + zt_res_92225;
                double r_tmp_105243 = zp_res_92226;
                
                r_92222 = r_tmp_105243;
            }
            defunc_0_lifted_lambda_res_92220 = r_92222;
            // futhark/microgpt.fut:336:67-149
            
            double zt_res_92227 = zs_res_92219 * defunc_0_lifted_lambda_res_92220;
            
            // futhark/microgpt.fut:336:45-149
            
            double zt_res_92228 = zs_res_92219 * zt_res_92227;
            
            // futhark/microgpt.fut:336:33-149
            
            double neg_res_92229 = -zt_res_92228;
            
            ((double *) mem_104436)[i_102296] = neg_res_92229;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102300 = 0; i_102300 < (int64_t) 16; i_102300++) {
            // futhark/microgpt.fut:337:33-43
            
            double zt_lhs_92237 = ((double *) mem_104436)[i_102300];
            
            // futhark/microgpt.fut:337:85-95
            
            double zp_lhs_92238 = ((double *) mem_102953)[i_102300];
            
            // futhark/microgpt.fut:337:85-123
            
            double zp_res_92239 = 1.0e-5 + zp_lhs_92238;
            
            // futhark/microgpt.fut:337:77-123
            
            double sqrt_res_92240 = futrts_sqrt64(zp_res_92239);
            
            // futhark/microgpt.fut:337:63-125
            
            double zt_res_92241 = 2.0 * sqrt_res_92240;
            
            // futhark/microgpt.fut:337:49-125
            
            double zs_res_92242 = 1.0 / zt_res_92241;
            
            // futhark/microgpt.fut:337:33-125
            
            double zt_res_92243 = zt_lhs_92237 * zs_res_92242;
            
            ((double *) mem_104443)[i_102300] = zt_res_92243;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102304 = 0; i_102304 < (int64_t) 16; i_102304++) {
            // futhark/microgpt.fut:338:53-63
            
            double zs_lhs_92251 = ((double *) mem_104443)[i_102304];
            
            // futhark/microgpt.fut:338:53-78
            
            double zs_res_92252 = zs_lhs_92251 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_105246 = 0; nest_i_105246 < (int64_t) 16; nest_i_105246++) {
                ((double *) mem_104450)[i_102304 * (int64_t) 16 + nest_i_105246] = zs_res_92252;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102312 = 0; i_102312 < (int64_t) 16; i_102312++) {
            // futhark/microgpt.fut:339:107-117
            
            double zs_rhs_92261 = ((double *) mem_103429)[i_102312];
            
            // futhark/microgpt.fut:339:99-117
            
            double zs_res_92262 = 1.0 / zs_rhs_92261;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102308 = 0; i_102308 < (int64_t) 16; i_102308++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_92269 = ((double *) mem_103976)[i_102312 * (int64_t) 16 + i_102308];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_92270 = ((double *) mem_104375)[i_102312 * (int64_t) 16 + i_102308];
                
                // futhark/microgpt.fut:339:77-117
                
                double zt_res_92271 = zs_res_92262 * zt_lhs_92270;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_92272 = ((double *) mem_102836)[i_102312 * (int64_t) 16 + i_102308];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_92273 = ((double *) mem_104450)[i_102312 * (int64_t) 16 + i_102308];
                
                // futhark/microgpt.fut:339:125-160
                
                double zt_res_92274 = zt_lhs_92272 * zt_rhs_92273;
                
                // futhark/microgpt.fut:339:94-160
                
                double zp_res_92275 = zt_res_92271 + zt_res_92274;
                
                // futhark/microgpt.fut:339:120-203
                
                double zp_res_92276 = zt_res_92274 + zp_res_92275;
                
                // futhark/microgpt.fut:339:53-203
                
                double zp_res_92277 = zp_lhs_92269 + zp_res_92276;
                
                ((double *) mem_104465)[i_102308] = zp_res_92277;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_104460, i_102312 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104465, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102316 = 0; i_102316 < (int64_t) 16; i_102316++) {
            // futhark/microgpt.fut:343:49-59
            
            double zs_rhs_92325 = ((double *) mem_102952)[i_102316];
            
            // futhark/microgpt.fut:343:41-59
            
            double zs_res_92326 = 1.0 / zs_rhs_92325;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_92327;
            double r_92329 = 0.0;
            
            for (int64_t i_92328 = 0; i_92328 < (int64_t) 16; i_92328++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_92330 = ((double *) mem_102803)[i_102316 * (int64_t) 16 + i_92328];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_92331 = ((double *) mem_104460)[i_102316 * (int64_t) 16 + i_92328];
                
                // futhark/microgpt.fut:343:87-122
                
                double zt_res_92332 = zt_lhs_92330 * zt_rhs_92331;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_92333 = r_92329 + zt_res_92332;
                double r_tmp_105250 = zp_res_92333;
                
                r_92329 = r_tmp_105250;
            }
            defunc_0_lifted_lambda_res_92327 = r_92329;
            // futhark/microgpt.fut:343:67-149
            
            double zt_res_92334 = zs_res_92326 * defunc_0_lifted_lambda_res_92327;
            
            // futhark/microgpt.fut:343:45-149
            
            double zt_res_92335 = zs_res_92326 * zt_res_92334;
            
            // futhark/microgpt.fut:343:33-149
            
            double neg_res_92336 = -zt_res_92335;
            
            ((double *) mem_104476)[i_102316] = neg_res_92336;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102320 = 0; i_102320 < (int64_t) 16; i_102320++) {
            // futhark/microgpt.fut:344:33-43
            
            double zt_lhs_92344 = ((double *) mem_104476)[i_102320];
            
            // futhark/microgpt.fut:344:85-95
            
            double zp_lhs_92345 = ((double *) mem_102890)[i_102320];
            
            // futhark/microgpt.fut:344:85-123
            
            double zp_res_92346 = 1.0e-5 + zp_lhs_92345;
            
            // futhark/microgpt.fut:344:77-123
            
            double sqrt_res_92347 = futrts_sqrt64(zp_res_92346);
            
            // futhark/microgpt.fut:344:63-125
            
            double zt_res_92348 = 2.0 * sqrt_res_92347;
            
            // futhark/microgpt.fut:344:49-125
            
            double zs_res_92349 = 1.0 / zt_res_92348;
            
            // futhark/microgpt.fut:344:33-125
            
            double zt_res_92350 = zt_lhs_92344 * zs_res_92349;
            
            ((double *) mem_104483)[i_102320] = zt_res_92350;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102324 = 0; i_102324 < (int64_t) 16; i_102324++) {
            // futhark/microgpt.fut:345:53-63
            
            double zs_lhs_92358 = ((double *) mem_104483)[i_102324];
            
            // futhark/microgpt.fut:345:53-78
            
            double zs_res_92359 = zs_lhs_92358 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_105253 = 0; nest_i_105253 < (int64_t) 16; nest_i_105253++) {
                ((double *) mem_104490)[i_102324 * (int64_t) 16 + nest_i_105253] = zs_res_92359;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102332 = 0; i_102332 < (int64_t) 16; i_102332++) {
            // futhark/microgpt.fut:346:85-95
            
            double zs_rhs_92368 = ((double *) mem_102952)[i_102332];
            
            // futhark/microgpt.fut:346:77-95
            
            double zs_res_92369 = 1.0 / zs_rhs_92368;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102328 = 0; i_102328 < (int64_t) 16; i_102328++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_92376 = ((double *) mem_104460)[i_102332 * (int64_t) 16 + i_102328];
                
                // futhark/microgpt.fut:346:55-95
                
                double zt_res_92377 = zs_res_92369 * zt_lhs_92376;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_92378 = ((double *) mem_102803)[i_102332 * (int64_t) 16 + i_102328];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_92379 = ((double *) mem_104490)[i_102332 * (int64_t) 16 + i_102328];
                
                // futhark/microgpt.fut:346:103-138
                
                double zt_res_92380 = zt_lhs_92378 * zt_rhs_92379;
                
                // futhark/microgpt.fut:346:72-138
                
                double zp_res_92381 = zt_res_92377 + zt_res_92380;
                
                // futhark/microgpt.fut:346:98-181
                
                double zp_res_92382 = zt_res_92380 + zp_res_92381;
                
                ((double *) mem_104505)[i_102328] = zp_res_92382;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_104500, i_102332 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104505, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102345 = 0; i_102345 < (int64_t) 16; i_102345++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102338 = 0; i_102338 < (int64_t) 16; i_102338++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_101034 = ((double *) mem_104500)[i_102345 * (int64_t) 16 + i_102338];
                
                ((double *) mem_104526)[i_102338] = lifted_lambda_res_101034;
                ((double *) mem_104527)[i_102338] = lifted_lambda_res_101034;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_104516, i_102345 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104526, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_104517, i_102345 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104527, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102354 = 0; i_102354 < (int64_t) 64; i_102354++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102350 = 0; i_102350 < (int64_t) 16; i_102350++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_92496;
                double r_92498 = 0.0;
                
                for (int64_t i_92497 = 0; i_92497 < (int64_t) 16; i_92497++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_92499 = ((double *) mem_103920)[i_92497 * (int64_t) 64 + i_102354];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_92500 = ((double *) mem_103533)[i_92497 * (int64_t) 16 + i_102350];
                    
                    // futhark/microgpt.fut:354:73-109
                    
                    double zt_res_92501 = zt_lhs_92499 * zt_rhs_92500;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_92502 = r_92498 + zt_res_92501;
                    double r_tmp_105262 = zp_res_92502;
                    
                    r_92498 = r_tmp_105262;
                }
                defunc_0_lifted_lambda_res_92496 = r_92498;
                ((double *) mem_104553)[i_102350] = defunc_0_lifted_lambda_res_92496;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_104548, i_102354 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104553, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_102367 = 0; i_102367 < (int64_t) 27; i_102367++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_102360 = 0; i_102360 < (int64_t) 16; i_102360++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_101062;
                double r_101064 = 0.0;
                
                for (int64_t i_101063 = 0; i_101063 < (int64_t) 16; i_101063++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_101065 = ((double *) mem_103856)[i_101063 * (int64_t) 27 + i_102367];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_101066 = ((double *) mem_103697)[i_101063 * (int64_t) 16 + i_102360];
                    
                    // futhark/microgpt.fut:356:74-110
                    
                    double zt_res_101067 = zt_lhs_101065 * zt_rhs_101066;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_101068 = r_101064 + zt_res_101067;
                    double r_tmp_105267 = zp_res_101068;
                    
                    r_101064 = r_tmp_105267;
                }
                defunc_0_lifted_lambda_res_101062 = r_101064;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_101071;
                double r_101073 = 0.0;
                
                for (int64_t i_101072 = 0; i_101072 < (int64_t) 16; i_101072++) {
                    int64_t zeze_lhs_101074 = ((int64_t *) seqs_mem_102638.mem)[step_90216 * (int64_t) 16 + i_101072];
                    
                    // futhark/microgpt.fut:475:58-109
                    
                    bool cond_101075 = zeze_lhs_101074 == i_102367;
                    
                    // futhark/microgpt.fut:475:58-109
                    
                    double lifted_lambda_res_101076;
                    
                    if (cond_101075) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double lifted_lambda_res_t_res_101470 = ((double *) mem_104516)[i_101072 * (int64_t) 16 + i_102360];
                        
                        lifted_lambda_res_101076 = lifted_lambda_res_t_res_101470;
                    } else {
                        lifted_lambda_res_101076 = 0.0;
                    }
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_101082 = r_101073 + lifted_lambda_res_101076;
                    double r_tmp_105268 = zp_res_101082;
                    
                    r_101073 = r_tmp_105268;
                }
                defunc_0_lifted_lambda_res_101071 = r_101073;
                ((double *) mem_104574)[i_102360] = defunc_0_lifted_lambda_res_101071;
                ((double *) mem_104575)[i_102360] = defunc_0_lifted_lambda_res_101062;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_104564, i_102367 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104574, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_104565, i_102367 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_104575, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:66:26-45
        
        double i64_res_92580 = sitofp_i64_f64(step_90216);
        
        // futhark/microgpt.fut:431:46-63
        
        double zm_rhs_92581 = i64_res_92580 / i64_res_90187;
        
        // futhark/microgpt.fut:431:24-63
        
        double zt_rhs_92582 = 1.0 - zm_rhs_92581;
        
        // futhark/microgpt.fut:431:19-63
        
        double lt_r_92583 = 1.0e-2 * zt_rhs_92582;
        
        // futhark/microgpt.fut:433:5-52
        if (memblock_alloc(ctx, &mem_104596, (int64_t) 3456, "mem_104596")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:433:5-52
        // futhark/microgpt.fut:433:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104596.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102664.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:433:5-52
        if (memblock_alloc(ctx, &mem_104598, (int64_t) 3456, "mem_104598")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:433:5-52
        // futhark/microgpt.fut:433:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104598.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102700.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:433:5-52
        if (memblock_alloc(ctx, &mem_104600, (int64_t) 3456, "mem_104600")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:433:5-52
        // futhark/microgpt.fut:433:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104600.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102736.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:433:5-52
        if (memblock_alloc(ctx, &mem_104602, (int64_t) 3456, "mem_104602")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:433:5-52
        // futhark/microgpt.fut:433:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104602.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_104564, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:433:5-52
        if (futrts_adam_opt_w_11260(ctx, &ext_mem_104606, &ext_mem_104605, &ext_mem_104604, mem_104596, mem_104598, mem_104600, mem_104602, (int64_t) 27, (int64_t) 16, step_90216, lt_r_92583) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_104596, "mem_104596") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104598, "mem_104598") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104600, "mem_104600") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104602, "mem_104602") != 0)
            return 1;
        // futhark/microgpt.fut:435:5-52
        if (memblock_alloc(ctx, &mem_104607, (int64_t) 2048, "mem_104607")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:435:5-52
        // futhark/microgpt.fut:435:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104607.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102656.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:435:5-52
        if (memblock_alloc(ctx, &mem_104609, (int64_t) 2048, "mem_104609")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:435:5-52
        // futhark/microgpt.fut:435:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104609.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102692.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:435:5-52
        if (memblock_alloc(ctx, &mem_104611, (int64_t) 2048, "mem_104611")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:435:5-52
        // futhark/microgpt.fut:435:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104611.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102728.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:435:5-52
        if (memblock_alloc(ctx, &mem_104613, (int64_t) 2048, "mem_104613")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:435:5-52
        // futhark/microgpt.fut:435:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104613.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_104517, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:435:5-52
        if (futrts_adam_opt_w_11261(ctx, &ext_mem_104617, &ext_mem_104616, &ext_mem_104615, mem_104607, mem_104609, mem_104611, mem_104613, (int64_t) 16, (int64_t) 16, step_90216, lt_r_92583) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_104607, "mem_104607") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104609, "mem_104609") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104611, "mem_104611") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104613, "mem_104613") != 0)
            return 1;
        // futhark/microgpt.fut:437:5-56
        if (memblock_alloc(ctx, &mem_104618, (int64_t) 2048, "mem_104618")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:437:5-56
        // futhark/microgpt.fut:437:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104618.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102660.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:437:5-56
        if (memblock_alloc(ctx, &mem_104620, (int64_t) 2048, "mem_104620")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:437:5-56
        // futhark/microgpt.fut:437:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104620.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102696.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:437:5-56
        if (memblock_alloc(ctx, &mem_104622, (int64_t) 2048, "mem_104622")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:437:5-56
        // futhark/microgpt.fut:437:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104622.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102732.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:437:5-56
        if (memblock_alloc(ctx, &mem_104624, (int64_t) 2048, "mem_104624")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:437:5-56
        // futhark/microgpt.fut:437:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104624.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_104374, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:437:5-56
        if (futrts_adam_opt_w_11261(ctx, &ext_mem_104628, &ext_mem_104627, &ext_mem_104626, mem_104618, mem_104620, mem_104622, mem_104624, (int64_t) 16, (int64_t) 16, step_90216, lt_r_92583) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_104618, "mem_104618") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104620, "mem_104620") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104622, "mem_104622") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104624, "mem_104624") != 0)
            return 1;
        // futhark/microgpt.fut:439:5-56
        if (memblock_alloc(ctx, &mem_104629, (int64_t) 2048, "mem_104629")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:439:5-56
        // futhark/microgpt.fut:439:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104629.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102648.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:439:5-56
        if (memblock_alloc(ctx, &mem_104631, (int64_t) 2048, "mem_104631")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:439:5-56
        // futhark/microgpt.fut:439:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104631.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102684.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:439:5-56
        if (memblock_alloc(ctx, &mem_104633, (int64_t) 2048, "mem_104633")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:439:5-56
        // futhark/microgpt.fut:439:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104633.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102720.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:439:5-56
        if (memblock_alloc(ctx, &mem_104635, (int64_t) 2048, "mem_104635")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:439:5-56
        // futhark/microgpt.fut:439:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104635.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_104373, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:439:5-56
        if (futrts_adam_opt_w_11261(ctx, &ext_mem_104639, &ext_mem_104638, &ext_mem_104637, mem_104629, mem_104631, mem_104633, mem_104635, (int64_t) 16, (int64_t) 16, step_90216, lt_r_92583) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_104629, "mem_104629") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104631, "mem_104631") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104633, "mem_104633") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104635, "mem_104635") != 0)
            return 1;
        // futhark/microgpt.fut:441:5-56
        if (memblock_alloc(ctx, &mem_104640, (int64_t) 2048, "mem_104640")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:441:5-56
        // futhark/microgpt.fut:441:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104640.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102672.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:441:5-56
        if (memblock_alloc(ctx, &mem_104642, (int64_t) 2048, "mem_104642")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:441:5-56
        // futhark/microgpt.fut:441:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104642.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102708.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:441:5-56
        if (memblock_alloc(ctx, &mem_104644, (int64_t) 2048, "mem_104644")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:441:5-56
        // futhark/microgpt.fut:441:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104644.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102744.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:441:5-56
        if (memblock_alloc(ctx, &mem_104646, (int64_t) 2048, "mem_104646")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:441:5-56
        // futhark/microgpt.fut:441:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104646.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_104372, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:441:5-56
        if (futrts_adam_opt_w_11261(ctx, &ext_mem_104650, &ext_mem_104649, &ext_mem_104648, mem_104640, mem_104642, mem_104644, mem_104646, (int64_t) 16, (int64_t) 16, step_90216, lt_r_92583) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_104640, "mem_104640") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104642, "mem_104642") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104644, "mem_104644") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104646, "mem_104646") != 0)
            return 1;
        // futhark/microgpt.fut:443:5-56
        if (memblock_alloc(ctx, &mem_104651, (int64_t) 2048, "mem_104651")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:443:5-56
        // futhark/microgpt.fut:443:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104651.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102652.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:443:5-56
        if (memblock_alloc(ctx, &mem_104653, (int64_t) 2048, "mem_104653")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:443:5-56
        // futhark/microgpt.fut:443:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104653.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102688.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:443:5-56
        if (memblock_alloc(ctx, &mem_104655, (int64_t) 2048, "mem_104655")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:443:5-56
        // futhark/microgpt.fut:443:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104655.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102724.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:443:5-56
        if (memblock_alloc(ctx, &mem_104657, (int64_t) 2048, "mem_104657")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:443:5-56
        // futhark/microgpt.fut:443:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104657.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_103992, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:443:5-56
        if (futrts_adam_opt_w_11261(ctx, &ext_mem_104661, &ext_mem_104660, &ext_mem_104659, mem_104651, mem_104653, mem_104655, mem_104657, (int64_t) 16, (int64_t) 16, step_90216, lt_r_92583) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_104651, "mem_104651") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104653, "mem_104653") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104655, "mem_104655") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104657, "mem_104657") != 0)
            return 1;
        // futhark/microgpt.fut:445:5-52
        if (memblock_alloc(ctx, &mem_104662, (int64_t) 8192, "mem_104662")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:445:5-52
        // futhark/microgpt.fut:445:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104662.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102668.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:445:5-52
        if (memblock_alloc(ctx, &mem_104664, (int64_t) 8192, "mem_104664")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:445:5-52
        // futhark/microgpt.fut:445:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104664.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102704.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:445:5-52
        if (memblock_alloc(ctx, &mem_104666, (int64_t) 8192, "mem_104666")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:445:5-52
        // futhark/microgpt.fut:445:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104666.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102740.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:445:5-52
        if (memblock_alloc(ctx, &mem_104668, (int64_t) 8192, "mem_104668")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:445:5-52
        // futhark/microgpt.fut:445:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104668.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_104548, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:445:5-52
        if (futrts_adam_opt_w_11260(ctx, &ext_mem_104672, &ext_mem_104671, &ext_mem_104670, mem_104662, mem_104664, mem_104666, mem_104668, (int64_t) 64, (int64_t) 16, step_90216, lt_r_92583) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_104662, "mem_104662") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104664, "mem_104664") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104666, "mem_104666") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104668, "mem_104668") != 0)
            return 1;
        // futhark/microgpt.fut:447:5-60
        if (memblock_alloc(ctx, &mem_104673, (int64_t) 8192, "mem_104673")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:447:5-60
        // futhark/microgpt.fut:447:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104673.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_102644.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:447:5-60
        if (memblock_alloc(ctx, &mem_104675, (int64_t) 8192, "mem_104675")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:447:5-60
        // futhark/microgpt.fut:447:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104675.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_102680.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:447:5-60
        if (memblock_alloc(ctx, &mem_104677, (int64_t) 8192, "mem_104677")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:447:5-60
        // futhark/microgpt.fut:447:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104677.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_102716.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:447:5-60
        if (memblock_alloc(ctx, &mem_104679, (int64_t) 8192, "mem_104679")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:447:5-60
        // futhark/microgpt.fut:447:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104679.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_103888, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:447:5-60
        if (futrts_adam_opt_w_11260(ctx, &ext_mem_104683, &ext_mem_104682, &ext_mem_104681, mem_104673, mem_104675, mem_104677, mem_104679, (int64_t) 16, (int64_t) 64, step_90216, lt_r_92583) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_104673, "mem_104673") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104675, "mem_104675") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104677, "mem_104677") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104679, "mem_104679") != 0)
            return 1;
        // futhark/microgpt.fut:449:5-56
        if (memblock_alloc(ctx, &mem_104684, (int64_t) 3456, "mem_104684")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:449:5-56
        // futhark/microgpt.fut:449:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104684.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102676.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:449:5-56
        if (memblock_alloc(ctx, &mem_104686, (int64_t) 3456, "mem_104686")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:449:5-56
        // futhark/microgpt.fut:449:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104686.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102712.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:449:5-56
        if (memblock_alloc(ctx, &mem_104688, (int64_t) 3456, "mem_104688")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:449:5-56
        // futhark/microgpt.fut:449:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104688.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_102748.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:449:5-56
        if (memblock_alloc(ctx, &mem_104690, (int64_t) 3456, "mem_104690")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:449:5-56
        // futhark/microgpt.fut:449:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_104690.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_104565, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:449:5-56
        if (futrts_adam_opt_w_11260(ctx, &ext_mem_104694, &ext_mem_104693, &ext_mem_104692, mem_104684, mem_104686, mem_104688, mem_104690, (int64_t) 27, (int64_t) 16, step_90216, lt_r_92583) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_104684, "mem_104684") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104686, "mem_104686") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104688, "mem_104688") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104690, "mem_104690") != 0)
            return 1;
        // futhark/microgpt.fut:504:21-33
        if (memblock_alloc(ctx, &mem_104696, bytes_102639, "mem_104696")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:504:21-33
        // futhark/microgpt.fut:504:21-33
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_104696.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_param_102751.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {n_74237});
        // futhark/microgpt.fut:504:21-52
        ((double *) mem_104696.mem)[step_90216] = zs_res_90852;
        if (memblock_set(ctx, &mem_param_tmp_104918, &ext_mem_104683, "ext_mem_104683") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104919, &ext_mem_104639, "ext_mem_104639") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104920, &ext_mem_104661, "ext_mem_104661") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104921, &ext_mem_104617, "ext_mem_104617") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104922, &ext_mem_104628, "ext_mem_104628") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104923, &ext_mem_104606, "ext_mem_104606") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104924, &ext_mem_104672, "ext_mem_104672") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104925, &ext_mem_104650, "ext_mem_104650") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104926, &ext_mem_104694, "ext_mem_104694") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104927, &ext_mem_104682, "ext_mem_104682") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104928, &ext_mem_104638, "ext_mem_104638") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104929, &ext_mem_104660, "ext_mem_104660") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104930, &ext_mem_104616, "ext_mem_104616") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104931, &ext_mem_104627, "ext_mem_104627") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104932, &ext_mem_104605, "ext_mem_104605") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104933, &ext_mem_104671, "ext_mem_104671") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104934, &ext_mem_104649, "ext_mem_104649") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104935, &ext_mem_104693, "ext_mem_104693") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104936, &ext_mem_104681, "ext_mem_104681") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104937, &ext_mem_104637, "ext_mem_104637") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104938, &ext_mem_104659, "ext_mem_104659") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104939, &ext_mem_104615, "ext_mem_104615") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104940, &ext_mem_104626, "ext_mem_104626") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104941, &ext_mem_104604, "ext_mem_104604") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104942, &ext_mem_104670, "ext_mem_104670") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104943, &ext_mem_104648, "ext_mem_104648") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104944, &ext_mem_104692, "ext_mem_104692") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_104945, &mem_104696, "mem_104696") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102644, &mem_param_tmp_104918, "mem_param_tmp_104918") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102648, &mem_param_tmp_104919, "mem_param_tmp_104919") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102652, &mem_param_tmp_104920, "mem_param_tmp_104920") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102656, &mem_param_tmp_104921, "mem_param_tmp_104921") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102660, &mem_param_tmp_104922, "mem_param_tmp_104922") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102664, &mem_param_tmp_104923, "mem_param_tmp_104923") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102668, &mem_param_tmp_104924, "mem_param_tmp_104924") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102672, &mem_param_tmp_104925, "mem_param_tmp_104925") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102676, &mem_param_tmp_104926, "mem_param_tmp_104926") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102680, &mem_param_tmp_104927, "mem_param_tmp_104927") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102684, &mem_param_tmp_104928, "mem_param_tmp_104928") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102688, &mem_param_tmp_104929, "mem_param_tmp_104929") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102692, &mem_param_tmp_104930, "mem_param_tmp_104930") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102696, &mem_param_tmp_104931, "mem_param_tmp_104931") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102700, &mem_param_tmp_104932, "mem_param_tmp_104932") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102704, &mem_param_tmp_104933, "mem_param_tmp_104933") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102708, &mem_param_tmp_104934, "mem_param_tmp_104934") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102712, &mem_param_tmp_104935, "mem_param_tmp_104935") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102716, &mem_param_tmp_104936, "mem_param_tmp_104936") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102720, &mem_param_tmp_104937, "mem_param_tmp_104937") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102724, &mem_param_tmp_104938, "mem_param_tmp_104938") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102728, &mem_param_tmp_104939, "mem_param_tmp_104939") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102732, &mem_param_tmp_104940, "mem_param_tmp_104940") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102736, &mem_param_tmp_104941, "mem_param_tmp_104941") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102740, &mem_param_tmp_104942, "mem_param_tmp_104942") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102744, &mem_param_tmp_104943, "mem_param_tmp_104943") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102748, &mem_param_tmp_104944, "mem_param_tmp_104944") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_102751, &mem_param_tmp_104945, "mem_param_tmp_104945") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_104807, &mem_param_102644, "mem_param_102644") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104806, &mem_param_102648, "mem_param_102648") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104805, &mem_param_102652, "mem_param_102652") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104804, &mem_param_102656, "mem_param_102656") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104803, &mem_param_102660, "mem_param_102660") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104802, &mem_param_102664, "mem_param_102664") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104801, &mem_param_102668, "mem_param_102668") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104800, &mem_param_102672, "mem_param_102672") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104799, &mem_param_102676, "mem_param_102676") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104798, &mem_param_102680, "mem_param_102680") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104797, &mem_param_102684, "mem_param_102684") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104796, &mem_param_102688, "mem_param_102688") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104795, &mem_param_102692, "mem_param_102692") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104794, &mem_param_102696, "mem_param_102696") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104793, &mem_param_102700, "mem_param_102700") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104792, &mem_param_102704, "mem_param_102704") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104791, &mem_param_102708, "mem_param_102708") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104790, &mem_param_102712, "mem_param_102712") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104789, &mem_param_102716, "mem_param_102716") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104788, &mem_param_102720, "mem_param_102720") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104787, &mem_param_102724, "mem_param_102724") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104786, &mem_param_102728, "mem_param_102728") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104785, &mem_param_102732, "mem_param_102732") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104784, &mem_param_102736, "mem_param_102736") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104783, &mem_param_102740, "mem_param_102740") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104782, &mem_param_102744, "mem_param_102744") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104781, &mem_param_102748, "mem_param_102748") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_104780, &mem_param_102751, "mem_param_102751") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_102640, "mem_102640") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_102752, "mem_102752") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104889, &ext_mem_104802, "ext_mem_104802") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104890, &ext_mem_104804, "ext_mem_104804") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104891, &ext_mem_104803, "ext_mem_104803") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104892, &ext_mem_104806, "ext_mem_104806") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104893, &ext_mem_104800, "ext_mem_104800") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104894, &ext_mem_104805, "ext_mem_104805") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104895, &ext_mem_104801, "ext_mem_104801") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104896, &ext_mem_104807, "ext_mem_104807") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104897, &ext_mem_104799, "ext_mem_104799") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104898, &ext_mem_104793, "ext_mem_104793") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104899, &ext_mem_104795, "ext_mem_104795") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104900, &ext_mem_104794, "ext_mem_104794") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104901, &ext_mem_104797, "ext_mem_104797") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104902, &ext_mem_104791, "ext_mem_104791") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104903, &ext_mem_104796, "ext_mem_104796") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104904, &ext_mem_104792, "ext_mem_104792") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104905, &ext_mem_104798, "ext_mem_104798") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104906, &ext_mem_104790, "ext_mem_104790") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104907, &ext_mem_104784, "ext_mem_104784") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104908, &ext_mem_104786, "ext_mem_104786") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104909, &ext_mem_104785, "ext_mem_104785") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104910, &ext_mem_104788, "ext_mem_104788") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104911, &ext_mem_104782, "ext_mem_104782") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104912, &ext_mem_104787, "ext_mem_104787") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104913, &ext_mem_104783, "ext_mem_104783") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104914, &ext_mem_104789, "ext_mem_104789") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104915, &ext_mem_104781, "ext_mem_104781") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104916, &ext_mem_104780, "ext_mem_104780") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105425, &mem_out_104889, "mem_out_104889") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105426, &mem_out_104890, "mem_out_104890") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105427, &mem_out_104891, "mem_out_104891") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105428, &mem_out_104892, "mem_out_104892") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105429, &mem_out_104893, "mem_out_104893") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105430, &mem_out_104894, "mem_out_104894") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105431, &mem_out_104895, "mem_out_104895") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105432, &mem_out_104896, "mem_out_104896") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105433, &mem_out_104897, "mem_out_104897") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105434, &mem_out_104898, "mem_out_104898") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105435, &mem_out_104899, "mem_out_104899") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105436, &mem_out_104900, "mem_out_104900") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105437, &mem_out_104901, "mem_out_104901") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105438, &mem_out_104902, "mem_out_104902") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105439, &mem_out_104903, "mem_out_104903") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105440, &mem_out_104904, "mem_out_104904") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105441, &mem_out_104905, "mem_out_104905") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105442, &mem_out_104906, "mem_out_104906") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105443, &mem_out_104907, "mem_out_104907") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105444, &mem_out_104908, "mem_out_104908") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105445, &mem_out_104909, "mem_out_104909") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105446, &mem_out_104910, "mem_out_104910") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105447, &mem_out_104911, "mem_out_104911") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105448, &mem_out_104912, "mem_out_104912") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105449, &mem_out_104913, "mem_out_104913") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105450, &mem_out_104914, "mem_out_104914") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105451, &mem_out_104915, "mem_out_104915") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105452, &mem_out_104916, "mem_out_104916") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_102755);
        free(mem_102756);
        free(mem_102757);
        free(mem_102770);
        free(mem_102771);
        free(mem_102784);
        free(mem_102803);
        free(mem_102804);
        free(mem_102813);
        free(mem_102814);
        free(mem_102835);
        free(mem_102836);
        free(mem_102837);
        free(mem_102850);
        free(mem_102851);
        free(mem_102852);
        free(mem_102871);
        free(mem_102890);
        free(mem_102891);
        free(mem_102892);
        free(mem_102893);
        free(mem_102909);
        free(mem_102910);
        free(mem_102911);
        free(mem_102930);
        free(mem_102952);
        free(mem_102953);
        free(mem_102954);
        free(mem_102955);
        free(mem_102956);
        free(mem_102957);
        free(mem_102958);
        free(mem_102959);
        free(mem_102990);
        free(mem_102991);
        free(mem_102992);
        free(mem_102993);
        free(mem_102994);
        free(mem_102995);
        free(mem_103062);
        free(mem_103063);
        free(mem_103064);
        free(mem_103065);
        free(mem_103066);
        free(mem_103067);
        free(mem_103098);
        free(mem_103099);
        free(mem_103100);
        free(mem_103101);
        free(mem_103102);
        free(mem_103103);
        free(mem_103128);
        free(mem_103129);
        free(mem_103130);
        free(mem_103131);
        free(mem_103132);
        free(mem_103133);
        free(mem_103224);
        free(mem_103225);
        free(mem_103226);
        free(mem_103242);
        free(mem_103243);
        free(mem_103244);
        free(mem_103257);
        free(mem_103258);
        free(mem_103259);
        free(mem_103290);
        free(mem_103291);
        free(mem_103300);
        free(mem_103301);
        free(mem_103322);
        free(mem_103323);
        free(mem_103332);
        free(mem_103333);
        free(mem_103346);
        free(mem_103347);
        free(mem_103360);
        free(mem_103361);
        free(mem_103382);
        free(mem_103383);
        free(mem_103392);
        free(mem_103393);
        free(mem_103429);
        free(mem_103430);
        free(mem_103431);
        free(mem_103443);
        free(mem_103444);
        free(mem_103468);
        free(mem_103469);
        free(mem_103478);
        free(mem_103479);
        free(mem_103500);
        free(mem_103501);
        free(mem_103510);
        free(mem_103511);
        free(mem_103532);
        free(mem_103533);
        free(mem_103534);
        free(mem_103547);
        free(mem_103548);
        free(mem_103549);
        free(mem_103568);
        free(mem_103587);
        free(mem_103588);
        free(mem_103589);
        free(mem_103601);
        free(mem_103602);
        free(mem_103626);
        free(mem_103627);
        free(mem_103628);
        free(mem_103640);
        free(mem_103641);
        free(mem_103665);
        free(mem_103666);
        free(mem_103675);
        free(mem_103676);
        free(mem_103697);
        free(mem_103698);
        free(mem_103707);
        free(mem_103708);
        free(mem_103729);
        free(mem_103730);
        free(mem_103739);
        free(mem_103740);
        free(mem_103761);
        free(mem_103762);
        free(mem_103763);
        free(mem_103775);
        free(mem_103776);
        free(mem_103777);
        free(mem_103796);
        free(mem_103797);
        free(mem_103798);
        free(mem_103817);
        free(mem_103818);
        free(mem_103819);
        free(mem_103849);
        free(mem_103856);
        free(mem_103861);
        free(mem_103872);
        free(mem_103877);
        free(mem_103888);
        free(mem_103889);
        free(mem_103898);
        free(mem_103899);
        free(mem_103920);
        free(mem_103925);
        free(mem_103936);
        free(mem_103941);
        free(mem_103952);
        free(mem_103959);
        free(mem_103966);
        free(mem_103976);
        free(mem_103981);
        free(mem_103992);
        free(mem_103993);
        free(mem_104002);
        free(mem_104003);
        free(mem_104024);
        free(mem_104025);
        free(mem_104036);
        free(mem_104037);
        free(mem_104046);
        free(mem_104053);
        free(mem_104078);
        free(mem_104079);
        free(mem_104090);
        free(mem_104091);
        free(mem_104100);
        free(mem_104107);
        free(mem_104114);
        free(mem_104121);
        free(mem_104146);
        free(mem_104147);
        free(mem_104158);
        free(mem_104159);
        free(mem_104168);
        free(mem_104175);
        free(mem_104200);
        free(mem_104205);
        free(mem_104216);
        free(mem_104222);
        free(mem_104227);
        free(mem_104243);
        free(mem_104249);
        free(mem_104254);
        free(mem_104270);
        free(mem_104271);
        free(mem_104282);
        free(mem_104283);
        free(mem_104292);
        free(mem_104293);
        free(mem_104324);
        free(mem_104325);
        free(mem_104326);
        free(mem_104339);
        free(mem_104340);
        free(mem_104341);
        free(mem_104372);
        free(mem_104373);
        free(mem_104374);
        free(mem_104375);
        free(mem_104392);
        free(mem_104393);
        free(mem_104394);
        free(mem_104395);
        free(mem_104436);
        free(mem_104443);
        free(mem_104450);
        free(mem_104460);
        free(mem_104465);
        free(mem_104476);
        free(mem_104483);
        free(mem_104490);
        free(mem_104500);
        free(mem_104505);
        free(mem_104516);
        free(mem_104517);
        free(mem_104526);
        free(mem_104527);
        free(mem_104548);
        free(mem_104553);
        free(mem_104564);
        free(mem_104565);
        free(mem_104574);
        free(mem_104575);
        if (memblock_unref(ctx, &mem_param_tmp_104945, "mem_param_tmp_104945") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104944, "mem_param_tmp_104944") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104943, "mem_param_tmp_104943") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104942, "mem_param_tmp_104942") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104941, "mem_param_tmp_104941") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104940, "mem_param_tmp_104940") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104939, "mem_param_tmp_104939") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104938, "mem_param_tmp_104938") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104937, "mem_param_tmp_104937") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104936, "mem_param_tmp_104936") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104935, "mem_param_tmp_104935") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104934, "mem_param_tmp_104934") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104933, "mem_param_tmp_104933") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104932, "mem_param_tmp_104932") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104931, "mem_param_tmp_104931") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104930, "mem_param_tmp_104930") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104929, "mem_param_tmp_104929") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104928, "mem_param_tmp_104928") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104927, "mem_param_tmp_104927") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104926, "mem_param_tmp_104926") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104925, "mem_param_tmp_104925") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104924, "mem_param_tmp_104924") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104923, "mem_param_tmp_104923") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104922, "mem_param_tmp_104922") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104921, "mem_param_tmp_104921") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104920, "mem_param_tmp_104920") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104919, "mem_param_tmp_104919") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_104918, "mem_param_tmp_104918") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104696, "mem_104696") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104692, "ext_mem_104692") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104693, "ext_mem_104693") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104694, "ext_mem_104694") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104690, "mem_104690") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104688, "mem_104688") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104686, "mem_104686") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104684, "mem_104684") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104681, "ext_mem_104681") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104682, "ext_mem_104682") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104683, "ext_mem_104683") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104679, "mem_104679") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104677, "mem_104677") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104675, "mem_104675") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104673, "mem_104673") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104670, "ext_mem_104670") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104671, "ext_mem_104671") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104672, "ext_mem_104672") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104668, "mem_104668") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104666, "mem_104666") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104664, "mem_104664") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104662, "mem_104662") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104659, "ext_mem_104659") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104660, "ext_mem_104660") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104661, "ext_mem_104661") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104657, "mem_104657") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104655, "mem_104655") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104653, "mem_104653") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104651, "mem_104651") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104648, "ext_mem_104648") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104649, "ext_mem_104649") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104650, "ext_mem_104650") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104646, "mem_104646") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104644, "mem_104644") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104642, "mem_104642") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104640, "mem_104640") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104637, "ext_mem_104637") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104638, "ext_mem_104638") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104639, "ext_mem_104639") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104635, "mem_104635") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104633, "mem_104633") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104631, "mem_104631") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104629, "mem_104629") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104626, "ext_mem_104626") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104627, "ext_mem_104627") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104628, "ext_mem_104628") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104624, "mem_104624") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104622, "mem_104622") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104620, "mem_104620") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104618, "mem_104618") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104615, "ext_mem_104615") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104616, "ext_mem_104616") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104617, "ext_mem_104617") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104613, "mem_104613") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104611, "mem_104611") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104609, "mem_104609") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104607, "mem_104607") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104604, "ext_mem_104604") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104605, "ext_mem_104605") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104606, "ext_mem_104606") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104602, "mem_104602") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104600, "mem_104600") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104598, "mem_104598") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_104596, "mem_104596") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_102754, "ext_mem_102754") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102751, "mem_param_102751") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102748, "mem_param_102748") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102744, "mem_param_102744") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102740, "mem_param_102740") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102736, "mem_param_102736") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102732, "mem_param_102732") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102728, "mem_param_102728") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102724, "mem_param_102724") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102720, "mem_param_102720") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102716, "mem_param_102716") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102712, "mem_param_102712") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102708, "mem_param_102708") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102704, "mem_param_102704") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102700, "mem_param_102700") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102696, "mem_param_102696") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102692, "mem_param_102692") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102688, "mem_param_102688") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102684, "mem_param_102684") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102680, "mem_param_102680") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102676, "mem_param_102676") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102672, "mem_param_102672") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102668, "mem_param_102668") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102664, "mem_param_102664") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102660, "mem_param_102660") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102656, "mem_param_102656") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102652, "mem_param_102652") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102648, "mem_param_102648") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_102644, "mem_param_102644") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104780, "ext_mem_104780") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104781, "ext_mem_104781") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104782, "ext_mem_104782") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104783, "ext_mem_104783") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104784, "ext_mem_104784") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104785, "ext_mem_104785") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104786, "ext_mem_104786") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104787, "ext_mem_104787") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104788, "ext_mem_104788") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104789, "ext_mem_104789") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104790, "ext_mem_104790") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104791, "ext_mem_104791") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104792, "ext_mem_104792") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104793, "ext_mem_104793") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104794, "ext_mem_104794") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104795, "ext_mem_104795") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104796, "ext_mem_104796") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104797, "ext_mem_104797") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104798, "ext_mem_104798") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104799, "ext_mem_104799") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104800, "ext_mem_104800") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104801, "ext_mem_104801") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104802, "ext_mem_104802") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104803, "ext_mem_104803") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104804, "ext_mem_104804") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104805, "ext_mem_104805") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104806, "ext_mem_104806") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_104807, "ext_mem_104807") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_102752, "mem_102752") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_102640, "mem_102640") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104916, "mem_out_104916") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104915, "mem_out_104915") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104914, "mem_out_104914") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104913, "mem_out_104913") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104912, "mem_out_104912") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104911, "mem_out_104911") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104910, "mem_out_104910") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104909, "mem_out_104909") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104908, "mem_out_104908") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104907, "mem_out_104907") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104906, "mem_out_104906") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104905, "mem_out_104905") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104904, "mem_out_104904") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104903, "mem_out_104903") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104902, "mem_out_104902") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104901, "mem_out_104901") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104900, "mem_out_104900") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104899, "mem_out_104899") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104898, "mem_out_104898") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104897, "mem_out_104897") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104896, "mem_out_104896") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104895, "mem_out_104895") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104894, "mem_out_104894") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104893, "mem_out_104893") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104892, "mem_out_104892") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104891, "mem_out_104891") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104890, "mem_out_104890") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104889, "mem_out_104889") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_105679, struct memblock *mem_out_p_105680, struct memblock *mem_out_p_105681, struct memblock *mem_out_p_105682, struct memblock *mem_out_p_105683, struct memblock *mem_out_p_105684, struct memblock *mem_out_p_105685, struct memblock *mem_out_p_105686, struct memblock *mem_out_p_105687)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_104897;
    
    mem_out_104897.references = NULL;
    
    struct memblock mem_out_104896;
    
    mem_out_104896.references = NULL;
    
    struct memblock mem_out_104895;
    
    mem_out_104895.references = NULL;
    
    struct memblock mem_out_104894;
    
    mem_out_104894.references = NULL;
    
    struct memblock mem_out_104893;
    
    mem_out_104893.references = NULL;
    
    struct memblock mem_out_104892;
    
    mem_out_104892.references = NULL;
    
    struct memblock mem_out_104891;
    
    mem_out_104891.references = NULL;
    
    struct memblock mem_out_104890;
    
    mem_out_104890.references = NULL;
    
    struct memblock mem_out_104889;
    
    mem_out_104889.references = NULL;
    
    struct memblock mem_102600 = ctx->constants->mem_102600;
    struct memblock mem_102601 = ctx->constants->mem_102601;
    struct memblock mem_102602 = ctx->constants->mem_102602;
    struct memblock mem_102603 = ctx->constants->mem_102603;
    struct memblock mem_102604 = ctx->constants->mem_102604;
    struct memblock mem_102605 = ctx->constants->mem_102605;
    struct memblock mem_102606 = ctx->constants->mem_102606;
    struct memblock mem_102607 = ctx->constants->mem_102607;
    struct memblock mem_102608 = ctx->constants->mem_102608;
    
    if (memblock_set(ctx, &mem_out_104889, &mem_102607, "mem_102607") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104890, &mem_102603, "mem_102603") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104891, &mem_102605, "mem_102605") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104892, &mem_102601, "mem_102601") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104893, &mem_102602, "mem_102602") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104894, &mem_102600, "mem_102600") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104895, &mem_102606, "mem_102606") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104896, &mem_102604, "mem_102604") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_104897, &mem_102608, "mem_102608") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105679, &mem_out_104889, "mem_out_104889") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105680, &mem_out_104890, "mem_out_104890") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105681, &mem_out_104891, "mem_out_104891") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105682, &mem_out_104892, "mem_out_104892") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105683, &mem_out_104893, "mem_out_104893") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105684, &mem_out_104894, "mem_out_104894") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105685, &mem_out_104895, "mem_out_104895") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105686, &mem_out_104896, "mem_out_104896") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_105687, &mem_out_104897, "mem_out_104897") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_104897, "mem_out_104897") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104896, "mem_out_104896") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104895, "mem_out_104895") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104894, "mem_out_104894") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104893, "mem_out_104893") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104892, "mem_out_104892") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104891, "mem_out_104891") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104890, "mem_out_104890") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_104889, "mem_out_104889") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_cal_loss(struct futhark_context *ctx, double *out, const int64_t in0, const struct futhark_opaque_params *in1, const struct futhark_i64_1d *in2, const struct futhark_f64_2d *in3)
{
    int64_t dl_52055 = (int64_t) 0;
    double prim_out_104889 = 0.0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mask_mem_102619;
    
    mask_mem_102619.references = NULL;
    
    struct memblock tokens_mem_102618;
    
    tokens_mem_102618.references = NULL;
    
    struct memblock wvoc_mem_102617;
    
    wvoc_mem_102617.references = NULL;
    
    struct memblock wval_mem_102616;
    
    wval_mem_102616.references = NULL;
    
    struct memblock wup_mem_102615;
    
    wup_mem_102615.references = NULL;
    
    struct memblock wte_mem_102614;
    
    wte_mem_102614.references = NULL;
    
    struct memblock wqry_mem_102613;
    
    wqry_mem_102613.references = NULL;
    
    struct memblock wpe_mem_102612;
    
    wpe_mem_102612.references = NULL;
    
    struct memblock wout_mem_102611;
    
    wout_mem_102611.references = NULL;
    
    struct memblock wkey_mem_102610;
    
    wkey_mem_102610.references = NULL;
    
    struct memblock wdown_mem_102609;
    
    wdown_mem_102609.references = NULL;
    dl_52055 = in0;
    wdown_mem_102609 = in1->v0->mem;
    wkey_mem_102610 = in1->v1->mem;
    wout_mem_102611 = in1->v2->mem;
    wpe_mem_102612 = in1->v3->mem;
    wqry_mem_102613 = in1->v4->mem;
    wte_mem_102614 = in1->v5->mem;
    wup_mem_102615 = in1->v6->mem;
    wval_mem_102616 = in1->v7->mem;
    wvoc_mem_102617 = in1->v8->mem;
    tokens_mem_102618 = in2->mem;
    mask_mem_102619 = in3->mem;
    if (!(((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in2->shape[0] && ((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_cal_loss(ctx, &prim_out_104889, wdown_mem_102609, wkey_mem_102610, wout_mem_102611, wpe_mem_102612, wqry_mem_102613, wte_mem_102614, wup_mem_102615, wval_mem_102616, wvoc_mem_102617, tokens_mem_102618, mask_mem_102619, dl_52055);
        if (ret == 0) {
            struct memblock mem_102600 = ctx->constants->mem_102600;
            struct memblock mem_102601 = ctx->constants->mem_102601;
            struct memblock mem_102602 = ctx->constants->mem_102602;
            struct memblock mem_102603 = ctx->constants->mem_102603;
            struct memblock mem_102604 = ctx->constants->mem_102604;
            struct memblock mem_102605 = ctx->constants->mem_102605;
            struct memblock mem_102606 = ctx->constants->mem_102606;
            struct memblock mem_102607 = ctx->constants->mem_102607;
            struct memblock mem_102608 = ctx->constants->mem_102608;
            
            *out = prim_out_104889;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_forward_seq(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_104889;
    
    mem_out_104889.references = NULL;
    
    struct memblock mask_mem_102619;
    
    mask_mem_102619.references = NULL;
    
    struct memblock tokens_mem_102618;
    
    tokens_mem_102618.references = NULL;
    
    struct memblock wvoc_mem_102617;
    
    wvoc_mem_102617.references = NULL;
    
    struct memblock wval_mem_102616;
    
    wval_mem_102616.references = NULL;
    
    struct memblock wup_mem_102615;
    
    wup_mem_102615.references = NULL;
    
    struct memblock wte_mem_102614;
    
    wte_mem_102614.references = NULL;
    
    struct memblock wqry_mem_102613;
    
    wqry_mem_102613.references = NULL;
    
    struct memblock wpe_mem_102612;
    
    wpe_mem_102612.references = NULL;
    
    struct memblock wout_mem_102611;
    
    wout_mem_102611.references = NULL;
    
    struct memblock wkey_mem_102610;
    
    wkey_mem_102610.references = NULL;
    
    struct memblock wdown_mem_102609;
    
    wdown_mem_102609.references = NULL;
    wdown_mem_102609 = in0->v0->mem;
    wkey_mem_102610 = in0->v1->mem;
    wout_mem_102611 = in0->v2->mem;
    wpe_mem_102612 = in0->v3->mem;
    wqry_mem_102613 = in0->v4->mem;
    wte_mem_102614 = in0->v5->mem;
    wup_mem_102615 = in0->v6->mem;
    wval_mem_102616 = in0->v7->mem;
    wvoc_mem_102617 = in0->v8->mem;
    tokens_mem_102618 = in1->mem;
    mask_mem_102619 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_104889, wdown_mem_102609, wkey_mem_102610, wout_mem_102611, wpe_mem_102612, wqry_mem_102613, wte_mem_102614, wup_mem_102615, wval_mem_102616, wvoc_mem_102617, tokens_mem_102618, mask_mem_102619);
        if (ret == 0) {
            struct memblock mem_102600 = ctx->constants->mem_102600;
            struct memblock mem_102601 = ctx->constants->mem_102601;
            struct memblock mem_102602 = ctx->constants->mem_102602;
            struct memblock mem_102603 = ctx->constants->mem_102603;
            struct memblock mem_102604 = ctx->constants->mem_102604;
            struct memblock mem_102605 = ctx->constants->mem_102605;
            struct memblock mem_102606 = ctx->constants->mem_102606;
            struct memblock mem_102607 = ctx->constants->mem_102607;
            struct memblock mem_102608 = ctx->constants->mem_102608;
            
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_104889;
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
    
    struct memblock mem_out_104897;
    
    mem_out_104897.references = NULL;
    
    struct memblock mem_out_104896;
    
    mem_out_104896.references = NULL;
    
    struct memblock mem_out_104895;
    
    mem_out_104895.references = NULL;
    
    struct memblock mem_out_104894;
    
    mem_out_104894.references = NULL;
    
    struct memblock mem_out_104893;
    
    mem_out_104893.references = NULL;
    
    struct memblock mem_out_104892;
    
    mem_out_104892.references = NULL;
    
    struct memblock mem_out_104891;
    
    mem_out_104891.references = NULL;
    
    struct memblock mem_out_104890;
    
    mem_out_104890.references = NULL;
    
    struct memblock mem_out_104889;
    
    mem_out_104889.references = NULL;
    
    struct memblock wvoc_mem_102617;
    
    wvoc_mem_102617.references = NULL;
    
    struct memblock wdown_mem_102616;
    
    wdown_mem_102616.references = NULL;
    
    struct memblock wup_mem_102615;
    
    wup_mem_102615.references = NULL;
    
    struct memblock wout_mem_102614;
    
    wout_mem_102614.references = NULL;
    
    struct memblock wval_mem_102613;
    
    wval_mem_102613.references = NULL;
    
    struct memblock wkey_mem_102612;
    
    wkey_mem_102612.references = NULL;
    
    struct memblock wqry_mem_102611;
    
    wqry_mem_102611.references = NULL;
    
    struct memblock wpe_mem_102610;
    
    wpe_mem_102610.references = NULL;
    
    struct memblock wte_mem_102609;
    
    wte_mem_102609.references = NULL;
    wte_mem_102609 = in0->mem;
    wpe_mem_102610 = in1->mem;
    wqry_mem_102611 = in2->mem;
    wkey_mem_102612 = in3->mem;
    wval_mem_102613 = in4->mem;
    wout_mem_102614 = in5->mem;
    wup_mem_102615 = in6->mem;
    wdown_mem_102616 = in7->mem;
    wvoc_mem_102617 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_104889, &mem_out_104890, &mem_out_104891, &mem_out_104892, &mem_out_104893, &mem_out_104894, &mem_out_104895, &mem_out_104896, &mem_out_104897, wte_mem_102609, wpe_mem_102610, wqry_mem_102611, wkey_mem_102612, wval_mem_102613, wout_mem_102614, wup_mem_102615, wdown_mem_102616, wvoc_mem_102617);
        if (ret == 0) {
            struct memblock mem_102600 = ctx->constants->mem_102600;
            struct memblock mem_102601 = ctx->constants->mem_102601;
            struct memblock mem_102602 = ctx->constants->mem_102602;
            struct memblock mem_102603 = ctx->constants->mem_102603;
            struct memblock mem_102604 = ctx->constants->mem_102604;
            struct memblock mem_102605 = ctx->constants->mem_102605;
            struct memblock mem_102606 = ctx->constants->mem_102606;
            struct memblock mem_102607 = ctx->constants->mem_102607;
            struct memblock mem_102608 = ctx->constants->mem_102608;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_104889;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_104890;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_104891;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_104892;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_104893;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_104894;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_104895;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_104896;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_104897;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_train(struct futhark_context *ctx, struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_opaque_params *in1, const struct futhark_opaque_params *in2, const struct futhark_f64_3d *in3, const struct futhark_i64_1d *in4, const struct futhark_i64_2d *in5)
{
    int64_t n_74237 = (int64_t) 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_104916;
    
    mem_out_104916.references = NULL;
    
    struct memblock mem_out_104915;
    
    mem_out_104915.references = NULL;
    
    struct memblock mem_out_104914;
    
    mem_out_104914.references = NULL;
    
    struct memblock mem_out_104913;
    
    mem_out_104913.references = NULL;
    
    struct memblock mem_out_104912;
    
    mem_out_104912.references = NULL;
    
    struct memblock mem_out_104911;
    
    mem_out_104911.references = NULL;
    
    struct memblock mem_out_104910;
    
    mem_out_104910.references = NULL;
    
    struct memblock mem_out_104909;
    
    mem_out_104909.references = NULL;
    
    struct memblock mem_out_104908;
    
    mem_out_104908.references = NULL;
    
    struct memblock mem_out_104907;
    
    mem_out_104907.references = NULL;
    
    struct memblock mem_out_104906;
    
    mem_out_104906.references = NULL;
    
    struct memblock mem_out_104905;
    
    mem_out_104905.references = NULL;
    
    struct memblock mem_out_104904;
    
    mem_out_104904.references = NULL;
    
    struct memblock mem_out_104903;
    
    mem_out_104903.references = NULL;
    
    struct memblock mem_out_104902;
    
    mem_out_104902.references = NULL;
    
    struct memblock mem_out_104901;
    
    mem_out_104901.references = NULL;
    
    struct memblock mem_out_104900;
    
    mem_out_104900.references = NULL;
    
    struct memblock mem_out_104899;
    
    mem_out_104899.references = NULL;
    
    struct memblock mem_out_104898;
    
    mem_out_104898.references = NULL;
    
    struct memblock mem_out_104897;
    
    mem_out_104897.references = NULL;
    
    struct memblock mem_out_104896;
    
    mem_out_104896.references = NULL;
    
    struct memblock mem_out_104895;
    
    mem_out_104895.references = NULL;
    
    struct memblock mem_out_104894;
    
    mem_out_104894.references = NULL;
    
    struct memblock mem_out_104893;
    
    mem_out_104893.references = NULL;
    
    struct memblock mem_out_104892;
    
    mem_out_104892.references = NULL;
    
    struct memblock mem_out_104891;
    
    mem_out_104891.references = NULL;
    
    struct memblock mem_out_104890;
    
    mem_out_104890.references = NULL;
    
    struct memblock mem_out_104889;
    
    mem_out_104889.references = NULL;
    
    struct memblock seqs_mem_102638;
    
    seqs_mem_102638.references = NULL;
    
    struct memblock dls_mem_102637;
    
    dls_mem_102637.references = NULL;
    
    struct memblock masks_mem_102636;
    
    masks_mem_102636.references = NULL;
    
    struct memblock wvoc_mem_102635;
    
    wvoc_mem_102635.references = NULL;
    
    struct memblock wval_mem_102634;
    
    wval_mem_102634.references = NULL;
    
    struct memblock wup_mem_102633;
    
    wup_mem_102633.references = NULL;
    
    struct memblock wte_mem_102632;
    
    wte_mem_102632.references = NULL;
    
    struct memblock wqry_mem_102631;
    
    wqry_mem_102631.references = NULL;
    
    struct memblock wpe_mem_102630;
    
    wpe_mem_102630.references = NULL;
    
    struct memblock wout_mem_102629;
    
    wout_mem_102629.references = NULL;
    
    struct memblock wkey_mem_102628;
    
    wkey_mem_102628.references = NULL;
    
    struct memblock wdown_mem_102627;
    
    wdown_mem_102627.references = NULL;
    
    struct memblock wvoc_mem_102626;
    
    wvoc_mem_102626.references = NULL;
    
    struct memblock wval_mem_102625;
    
    wval_mem_102625.references = NULL;
    
    struct memblock wup_mem_102624;
    
    wup_mem_102624.references = NULL;
    
    struct memblock wte_mem_102623;
    
    wte_mem_102623.references = NULL;
    
    struct memblock wqry_mem_102622;
    
    wqry_mem_102622.references = NULL;
    
    struct memblock wpe_mem_102621;
    
    wpe_mem_102621.references = NULL;
    
    struct memblock wout_mem_102620;
    
    wout_mem_102620.references = NULL;
    
    struct memblock wkey_mem_102619;
    
    wkey_mem_102619.references = NULL;
    
    struct memblock wdown_mem_102618;
    
    wdown_mem_102618.references = NULL;
    
    struct memblock wvoc_mem_102617;
    
    wvoc_mem_102617.references = NULL;
    
    struct memblock wval_mem_102616;
    
    wval_mem_102616.references = NULL;
    
    struct memblock wup_mem_102615;
    
    wup_mem_102615.references = NULL;
    
    struct memblock wte_mem_102614;
    
    wte_mem_102614.references = NULL;
    
    struct memblock wqry_mem_102613;
    
    wqry_mem_102613.references = NULL;
    
    struct memblock wpe_mem_102612;
    
    wpe_mem_102612.references = NULL;
    
    struct memblock wout_mem_102611;
    
    wout_mem_102611.references = NULL;
    
    struct memblock wkey_mem_102610;
    
    wkey_mem_102610.references = NULL;
    
    struct memblock wdown_mem_102609;
    
    wdown_mem_102609.references = NULL;
    wdown_mem_102609 = in0->v0->mem;
    wkey_mem_102610 = in0->v1->mem;
    wout_mem_102611 = in0->v2->mem;
    wpe_mem_102612 = in0->v3->mem;
    wqry_mem_102613 = in0->v4->mem;
    wte_mem_102614 = in0->v5->mem;
    wup_mem_102615 = in0->v6->mem;
    wval_mem_102616 = in0->v7->mem;
    wvoc_mem_102617 = in0->v8->mem;
    wdown_mem_102618 = in1->v0->mem;
    wkey_mem_102619 = in1->v1->mem;
    wout_mem_102620 = in1->v2->mem;
    wpe_mem_102621 = in1->v3->mem;
    wqry_mem_102622 = in1->v4->mem;
    wte_mem_102623 = in1->v5->mem;
    wup_mem_102624 = in1->v6->mem;
    wval_mem_102625 = in1->v7->mem;
    wvoc_mem_102626 = in1->v8->mem;
    wdown_mem_102627 = in2->v0->mem;
    wkey_mem_102628 = in2->v1->mem;
    wout_mem_102629 = in2->v2->mem;
    wpe_mem_102630 = in2->v3->mem;
    wqry_mem_102631 = in2->v4->mem;
    wte_mem_102632 = in2->v5->mem;
    wup_mem_102633 = in2->v6->mem;
    wval_mem_102634 = in2->v7->mem;
    wvoc_mem_102635 = in2->v8->mem;
    masks_mem_102636 = in3->mem;
    n_74237 = in3->shape[0];
    dls_mem_102637 = in4->mem;
    n_74237 = in4->shape[0];
    seqs_mem_102638 = in5->mem;
    n_74237 = in5->shape[0];
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && ((n_74237 == in3->shape[0] && ((int64_t) 16 == in3->shape[1] && (int64_t) 16 == in3->shape[2])) && (n_74237 == in4->shape[0] && (n_74237 == in5->shape[0] && (int64_t) 16 == in5->shape[1]))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_104889, &mem_out_104890, &mem_out_104891, &mem_out_104892, &mem_out_104893, &mem_out_104894, &mem_out_104895, &mem_out_104896, &mem_out_104897, &mem_out_104898, &mem_out_104899, &mem_out_104900, &mem_out_104901, &mem_out_104902, &mem_out_104903, &mem_out_104904, &mem_out_104905, &mem_out_104906, &mem_out_104907, &mem_out_104908, &mem_out_104909, &mem_out_104910, &mem_out_104911, &mem_out_104912, &mem_out_104913, &mem_out_104914, &mem_out_104915, &mem_out_104916, wdown_mem_102609, wkey_mem_102610, wout_mem_102611, wpe_mem_102612, wqry_mem_102613, wte_mem_102614, wup_mem_102615, wval_mem_102616, wvoc_mem_102617, wdown_mem_102618, wkey_mem_102619, wout_mem_102620, wpe_mem_102621, wqry_mem_102622, wte_mem_102623, wup_mem_102624, wval_mem_102625, wvoc_mem_102626, wdown_mem_102627, wkey_mem_102628, wout_mem_102629, wpe_mem_102630, wqry_mem_102631, wte_mem_102632, wup_mem_102633, wval_mem_102634, wvoc_mem_102635, masks_mem_102636, dls_mem_102637, seqs_mem_102638, n_74237);
        if (ret == 0) {
            struct memblock mem_102600 = ctx->constants->mem_102600;
            struct memblock mem_102601 = ctx->constants->mem_102601;
            struct memblock mem_102602 = ctx->constants->mem_102602;
            struct memblock mem_102603 = ctx->constants->mem_102603;
            struct memblock mem_102604 = ctx->constants->mem_102604;
            struct memblock mem_102605 = ctx->constants->mem_102605;
            struct memblock mem_102606 = ctx->constants->mem_102606;
            struct memblock mem_102607 = ctx->constants->mem_102607;
            struct memblock mem_102608 = ctx->constants->mem_102608;
            
            assert((*out = (struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64 *) malloc(sizeof(struct futhark_opaque_tup4_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr1d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_104889;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_104890;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_104891;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_104892;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_104893;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_104894;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_104895;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_104896;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_104897;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v9->mem = mem_out_104898;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v10->mem = mem_out_104899;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v11->mem = mem_out_104900;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v12->mem = mem_out_104901;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v13->mem = mem_out_104902;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v14->mem = mem_out_104903;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v15->mem = mem_out_104904;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v16->mem = mem_out_104905;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v17->mem = mem_out_104906;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v18->mem = mem_out_104907;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v19->mem = mem_out_104908;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v20->mem = mem_out_104909;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v21->mem = mem_out_104910;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v22->mem = mem_out_104911;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v23->mem = mem_out_104912;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v24->mem = mem_out_104913;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v25->mem = mem_out_104914;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v26->mem = mem_out_104915;
            (*out)->v26->shape[0] = (int64_t) 27;
            (*out)->v26->shape[1] = (int64_t) 16;
            assert(((*out)->v27 = (struct futhark_f64_1d *) malloc(sizeof(struct futhark_f64_1d))) != NULL);
            (*out)->v27->mem = mem_out_104916;
            (*out)->v27->shape[0] = n_74237;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_zero_params(struct futhark_context *ctx, struct futhark_opaque_params **out)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_104897;
    
    mem_out_104897.references = NULL;
    
    struct memblock mem_out_104896;
    
    mem_out_104896.references = NULL;
    
    struct memblock mem_out_104895;
    
    mem_out_104895.references = NULL;
    
    struct memblock mem_out_104894;
    
    mem_out_104894.references = NULL;
    
    struct memblock mem_out_104893;
    
    mem_out_104893.references = NULL;
    
    struct memblock mem_out_104892;
    
    mem_out_104892.references = NULL;
    
    struct memblock mem_out_104891;
    
    mem_out_104891.references = NULL;
    
    struct memblock mem_out_104890;
    
    mem_out_104890.references = NULL;
    
    struct memblock mem_out_104889;
    
    mem_out_104889.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_104889, &mem_out_104890, &mem_out_104891, &mem_out_104892, &mem_out_104893, &mem_out_104894, &mem_out_104895, &mem_out_104896, &mem_out_104897);
        if (ret == 0) {
            struct memblock mem_102600 = ctx->constants->mem_102600;
            struct memblock mem_102601 = ctx->constants->mem_102601;
            struct memblock mem_102602 = ctx->constants->mem_102602;
            struct memblock mem_102603 = ctx->constants->mem_102603;
            struct memblock mem_102604 = ctx->constants->mem_102604;
            struct memblock mem_102605 = ctx->constants->mem_102605;
            struct memblock mem_102606 = ctx->constants->mem_102606;
            struct memblock mem_102607 = ctx->constants->mem_102607;
            struct memblock mem_102608 = ctx->constants->mem_102608;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_104889;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_104890;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_104891;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_104892;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_104893;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_104894;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_104895;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_104896;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_104897;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
