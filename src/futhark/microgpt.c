
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
    struct memblock mem_132596;
    struct memblock mem_132597;
    struct memblock mem_132598;
    struct memblock mem_132599;
    struct memblock mem_132600;
    struct memblock mem_132601;
    struct memblock mem_132602;
    struct memblock mem_132603;
    struct memblock mem_132604;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_12605(struct futhark_context *ctx, struct memblock *mem_out_p_134866, struct memblock *mem_out_p_134867, struct memblock *mem_out_p_134868, struct memblock w_mem_132605, struct memblock mw_mem_132606, struct memblock vw_mem_132607, struct memblock dw_mem_132608, int64_t n_97788, int64_t m_97789, int64_t step_97794, double lt_r_97795);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_12606(struct futhark_context *ctx, struct memblock *mem_out_p_134871, struct memblock *mem_out_p_134872, struct memblock *mem_out_p_134873, struct memblock w_mem_132605, struct memblock mw_mem_132606, struct memblock vw_mem_132607, struct memblock dw_mem_132608, int64_t n_98821, int64_t m_98822, int64_t step_98827, double lt_r_98828);
FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_134876, double *out_prim_out_134877, struct memblock wdown_mem_132605, struct memblock wkey_mem_132606, struct memblock wout_mem_132607, struct memblock wpe_mem_132608, struct memblock wqry_mem_132609, struct memblock wte_mem_132610, struct memblock wup_mem_132611, struct memblock wval_mem_132612, struct memblock wvoc_mem_132613, struct memblock tokens_mem_132614, struct memblock target_mem_132615, struct memblock mask_mem_132616);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_134935, struct memblock wdown_mem_132605, struct memblock wkey_mem_132606, struct memblock wout_mem_132607, struct memblock wpe_mem_132608, struct memblock wqry_mem_132609, struct memblock wte_mem_132610, struct memblock wup_mem_132611, struct memblock wval_mem_132612, struct memblock wvoc_mem_132613, struct memblock tokens_mem_132614, struct memblock mask_mem_132615);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_134992, struct memblock *mem_out_p_134993, struct memblock *mem_out_p_134994, struct memblock *mem_out_p_134995, struct memblock *mem_out_p_134996, struct memblock *mem_out_p_134997, struct memblock *mem_out_p_134998, struct memblock *mem_out_p_134999, struct memblock *mem_out_p_135000, struct memblock wte_mem_132605, struct memblock wpe_mem_132606, struct memblock wqry_mem_132607, struct memblock wkey_mem_132608, struct memblock wval_mem_132609, struct memblock wout_mem_132610, struct memblock wup_mem_132611, struct memblock wdown_mem_132612, struct memblock wvoc_mem_132613);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_135001, struct memblock *mem_out_p_135002, struct memblock *mem_out_p_135003, struct memblock *mem_out_p_135004, struct memblock *mem_out_p_135005, struct memblock *mem_out_p_135006, struct memblock *mem_out_p_135007, struct memblock *mem_out_p_135008, struct memblock *mem_out_p_135009, struct memblock *mem_out_p_135010, struct memblock *mem_out_p_135011, struct memblock *mem_out_p_135012, struct memblock *mem_out_p_135013, struct memblock *mem_out_p_135014, struct memblock *mem_out_p_135015, struct memblock *mem_out_p_135016, struct memblock *mem_out_p_135017, struct memblock *mem_out_p_135018, struct memblock *mem_out_p_135019, struct memblock *mem_out_p_135020, struct memblock *mem_out_p_135021, struct memblock *mem_out_p_135022, struct memblock *mem_out_p_135023, struct memblock *mem_out_p_135024, struct memblock *mem_out_p_135025, struct memblock *mem_out_p_135026, struct memblock *mem_out_p_135027, struct memblock wdown_mem_132605, struct memblock wkey_mem_132606, struct memblock wout_mem_132607, struct memblock wpe_mem_132608, struct memblock wqry_mem_132609, struct memblock wte_mem_132610, struct memblock wup_mem_132611, struct memblock wval_mem_132612, struct memblock wvoc_mem_132613, struct memblock wdown_mem_132614, struct memblock wkey_mem_132615, struct memblock wout_mem_132616, struct memblock wpe_mem_132617, struct memblock wqry_mem_132618, struct memblock wte_mem_132619, struct memblock wup_mem_132620, struct memblock wval_mem_132621, struct memblock wvoc_mem_132622, struct memblock wdown_mem_132623, struct memblock wkey_mem_132624, struct memblock wout_mem_132625, struct memblock wpe_mem_132626, struct memblock wqry_mem_132627, struct memblock wte_mem_132628, struct memblock wup_mem_132629, struct memblock wval_mem_132630, struct memblock wvoc_mem_132631, struct memblock masks_mem_132632, struct memblock dls_mem_132633, struct memblock seqs_mem_132634);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_135210, struct memblock *mem_out_p_135211, struct memblock *mem_out_p_135212, struct memblock *mem_out_p_135213, struct memblock *mem_out_p_135214, struct memblock *mem_out_p_135215, struct memblock *mem_out_p_135216, struct memblock *mem_out_p_135217, struct memblock *mem_out_p_135218);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_132596 (ctx->constants->mem_132596)
    #define mem_132597 (ctx->constants->mem_132597)
    #define mem_132598 (ctx->constants->mem_132598)
    #define mem_132599 (ctx->constants->mem_132599)
    #define mem_132600 (ctx->constants->mem_132600)
    #define mem_132601 (ctx->constants->mem_132601)
    #define mem_132602 (ctx->constants->mem_132602)
    #define mem_132603 (ctx->constants->mem_132603)
    #define mem_132604 (ctx->constants->mem_132604)
    mem_132596.references = NULL;
    mem_132597.references = NULL;
    mem_132598.references = NULL;
    mem_132599.references = NULL;
    mem_132600.references = NULL;
    mem_132601.references = NULL;
    mem_132602.references = NULL;
    mem_132603.references = NULL;
    mem_132604.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_132596, (int64_t) 3456, "mem_132596")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_134848 = 0; nest_i_134848 < (int64_t) 27; nest_i_134848++) {
        for (int64_t nest_i_134849 = 0; nest_i_134849 < (int64_t) 16; nest_i_134849++) {
            ((double *) mem_132596.mem)[nest_i_134848 * (int64_t) 16 + nest_i_134849] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_132597, (int64_t) 2048, "mem_132597")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_134850 = 0; nest_i_134850 < (int64_t) 16; nest_i_134850++) {
        for (int64_t nest_i_134851 = 0; nest_i_134851 < (int64_t) 16; nest_i_134851++) {
            ((double *) mem_132597.mem)[nest_i_134850 * (int64_t) 16 + nest_i_134851] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_132598, (int64_t) 2048, "mem_132598")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_134852 = 0; nest_i_134852 < (int64_t) 16; nest_i_134852++) {
        for (int64_t nest_i_134853 = 0; nest_i_134853 < (int64_t) 16; nest_i_134853++) {
            ((double *) mem_132598.mem)[nest_i_134852 * (int64_t) 16 + nest_i_134853] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_132599, (int64_t) 2048, "mem_132599")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_134854 = 0; nest_i_134854 < (int64_t) 16; nest_i_134854++) {
        for (int64_t nest_i_134855 = 0; nest_i_134855 < (int64_t) 16; nest_i_134855++) {
            ((double *) mem_132599.mem)[nest_i_134854 * (int64_t) 16 + nest_i_134855] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_132600, (int64_t) 2048, "mem_132600")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_134856 = 0; nest_i_134856 < (int64_t) 16; nest_i_134856++) {
        for (int64_t nest_i_134857 = 0; nest_i_134857 < (int64_t) 16; nest_i_134857++) {
            ((double *) mem_132600.mem)[nest_i_134856 * (int64_t) 16 + nest_i_134857] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_132601, (int64_t) 2048, "mem_132601")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_134858 = 0; nest_i_134858 < (int64_t) 16; nest_i_134858++) {
        for (int64_t nest_i_134859 = 0; nest_i_134859 < (int64_t) 16; nest_i_134859++) {
            ((double *) mem_132601.mem)[nest_i_134858 * (int64_t) 16 + nest_i_134859] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_132602, (int64_t) 8192, "mem_132602")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_134860 = 0; nest_i_134860 < (int64_t) 64; nest_i_134860++) {
        for (int64_t nest_i_134861 = 0; nest_i_134861 < (int64_t) 16; nest_i_134861++) {
            ((double *) mem_132602.mem)[nest_i_134860 * (int64_t) 16 + nest_i_134861] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_132603, (int64_t) 8192, "mem_132603")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_134862 = 0; nest_i_134862 < (int64_t) 16; nest_i_134862++) {
        for (int64_t nest_i_134863 = 0; nest_i_134863 < (int64_t) 64; nest_i_134863++) {
            ((double *) mem_132603.mem)[nest_i_134862 * (int64_t) 64 + nest_i_134863] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_132604, (int64_t) 3456, "mem_132604")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_134864 = 0; nest_i_134864 < (int64_t) 27; nest_i_134864++) {
        for (int64_t nest_i_134865 = 0; nest_i_134865 < (int64_t) 16; nest_i_134865++) {
            ((double *) mem_132604.mem)[nest_i_134864 * (int64_t) 16 + nest_i_134865] = 0.0;
        }
    }
    #undef mem_132596
    #undef mem_132597
    #undef mem_132598
    #undef mem_132599
    #undef mem_132600
    #undef mem_132601
    #undef mem_132602
    #undef mem_132603
    #undef mem_132604
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_132596, "ctx->constants->mem_132596") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_132597, "ctx->constants->mem_132597") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_132598, "ctx->constants->mem_132598") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_132599, "ctx->constants->mem_132599") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_132600, "ctx->constants->mem_132600") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_132601, "ctx->constants->mem_132601") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_132602, "ctx->constants->mem_132602") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_132603, "ctx->constants->mem_132603") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_132604, "ctx->constants->mem_132604") != 0)
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_12605(struct futhark_context *ctx, struct memblock *mem_out_p_134866, struct memblock *mem_out_p_134867, struct memblock *mem_out_p_134868, struct memblock w_mem_132605, struct memblock mw_mem_132606, struct memblock vw_mem_132607, struct memblock dw_mem_132608, int64_t n_97788, int64_t m_97789, int64_t step_97794, double lt_r_97795)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_132649_cached_sizze_134869 = 0;
    unsigned char *mem_132649 = NULL;
    int64_t mem_132652_cached_sizze_134870 = 0;
    unsigned char *mem_132652 = NULL;
    struct memblock mem_132687;
    
    mem_132687.references = NULL;
    
    struct memblock mem_132614;
    
    mem_132614.references = NULL;
    
    struct memblock mem_132611;
    
    mem_132611.references = NULL;
    
    struct memblock mem_out_134507;
    
    mem_out_134507.references = NULL;
    
    struct memblock mem_out_134506;
    
    mem_out_134506.references = NULL;
    
    struct memblock mem_out_134505;
    
    mem_out_134505.references = NULL;
    
    struct memblock mem_132596 = ctx->constants->mem_132596;
    struct memblock mem_132597 = ctx->constants->mem_132597;
    struct memblock mem_132598 = ctx->constants->mem_132598;
    struct memblock mem_132599 = ctx->constants->mem_132599;
    struct memblock mem_132600 = ctx->constants->mem_132600;
    struct memblock mem_132601 = ctx->constants->mem_132601;
    struct memblock mem_132602 = ctx->constants->mem_132602;
    struct memblock mem_132603 = ctx->constants->mem_132603;
    struct memblock mem_132604 = ctx->constants->mem_132604;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_132609 = (int64_t) 8 * n_97788;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_132610 = m_97789 * binop_x_132609;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_132611, bytes_132610, "mem_132611")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_132614, bytes_132610, "mem_132614")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131721 = 0; i_131721 < n_97788; i_131721++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131714 = 0; i_131714 < m_97789; i_131714++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_124572 = ((double *) mw_mem_132606.mem)[i_131721 * m_97789 + i_131714];
            
            // futhark/microgpt.fut:462:10-20
            
            double zp_lhs_124573 = 0.85 * zt_rhs_124572;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_124574 = ((double *) dw_mem_132608.mem)[i_131721 * m_97789 + i_131714];
            
            // futhark/microgpt.fut:462:35-45
            
            double zp_rhs_124575 = 0.15000000000000002 * zt_rhs_124574;
            
            // futhark/microgpt.fut:462:21-45
            
            double lifted_lambda_res_124576 = zp_lhs_124573 + zp_rhs_124575;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_124583 = ((double *) vw_mem_132607.mem)[i_131721 * m_97789 + i_131714];
            
            // futhark/microgpt.fut:464:10-20
            
            double zp_lhs_124584 = 0.99 * zt_rhs_124583;
            
            // futhark/microgpt.fut:464:35-45
            
            double zt_lhs_124586 = 1.0000000000000009e-2 * zt_rhs_124574;
            
            // futhark/microgpt.fut:464:46-56
            
            double zp_rhs_124587 = zt_rhs_124574 * zt_lhs_124586;
            
            // futhark/microgpt.fut:464:21-56
            
            double lifted_lambda_res_124588 = zp_lhs_124584 + zp_rhs_124587;
            
            ((double *) mem_132611.mem)[i_131721 * m_97789 + i_131714] = lifted_lambda_res_124588;
            ((double *) mem_132614.mem)[i_131721 * m_97789 + i_131714] = lifted_lambda_res_124576;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_103217 = sitofp_i64_f64(step_97794);
    
    // futhark/microgpt.fut:466:54-57
    
    double ztzt_rhs_103218 = 1.0 + i64_res_103217;
    
    // futhark/microgpt.fut:466:30-57
    
    double zm_rhs_103219 = fpow64(0.85, ztzt_rhs_103218);
    
    // futhark/microgpt.fut:466:23-57
    
    double zs_rhs_103220 = 1.0 - zm_rhs_103219;
    
    // futhark/microgpt.fut:468:31-58
    
    double zm_rhs_103258 = fpow64(0.99, ztzt_rhs_103218);
    
    // futhark/microgpt.fut:468:23-58
    
    double zs_rhs_103259 = 1.0 - zm_rhs_103258;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_132649_cached_sizze_134869 < bytes_132610) {
        err = lexical_realloc(ctx, &mem_132649, &mem_132649_cached_sizze_134869, bytes_132610);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132652_cached_sizze_134870 < bytes_132610) {
        err = lexical_realloc(ctx, &mem_132652, &mem_132652_cached_sizze_134870, bytes_132610);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131735 = 0; i_131735 < n_97788; i_131735++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131728 = 0; i_131728 < m_97789; i_131728++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_124608 = ((double *) mem_132614.mem)[i_131735 * m_97789 + i_131728];
            
            // futhark/microgpt.fut:466:18-57
            
            double lifted_lambda_res_124609 = zs_lhs_124608 / zs_rhs_103220;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_124616 = ((double *) mem_132611.mem)[i_131735 * m_97789 + i_131728];
            
            // futhark/microgpt.fut:468:18-58
            
            double lifted_lambda_res_124617 = zs_lhs_124616 / zs_rhs_103259;
            
            ((double *) mem_132649)[i_131735 * m_97789 + i_131728] = lifted_lambda_res_124617;
            ((double *) mem_132652)[i_131735 * m_97789 + i_131728] = lifted_lambda_res_124609;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_132687, bytes_132610, "mem_132687")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131744 = 0; i_131744 < n_97788; i_131744++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131740 = 0; i_131740 < m_97789; i_131740++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_102308 = ((double *) w_mem_132605.mem)[i_131744 * m_97789 + i_131740];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_102309 = ((double *) mem_132652)[i_131744 * m_97789 + i_131740];
            
            // futhark/microgpt.fut:470:21-34
            
            double zs_lhs_102310 = lt_r_97795 * zt_rhs_102309;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_102311 = ((double *) mem_132649)[i_131744 * m_97789 + i_131740];
            
            // futhark/microgpt.fut:470:51-57
            
            double zp_lhs_102312 = fpow64(ztzt_lhs_102311, 0.5);
            
            // futhark/microgpt.fut:470:59-71
            
            double zs_rhs_102313 = 1.0e-8 + zp_lhs_102312;
            
            // futhark/microgpt.fut:470:35-71
            
            double zm_rhs_102314 = zs_lhs_102310 / zs_rhs_102313;
            
            // futhark/microgpt.fut:470:13-71
            
            double lifted_lambda_res_102315 = zm_lhs_102308 - zm_rhs_102314;
            
            ((double *) mem_132687.mem)[i_131744 * m_97789 + i_131740] = lifted_lambda_res_102315;
        }
    }
    if (memblock_set(ctx, &mem_out_134505, &mem_132687, "mem_132687") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134506, &mem_132614, "mem_132614") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134507, &mem_132611, "mem_132611") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_134866, &mem_out_134505, "mem_out_134505") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_134867, &mem_out_134506, "mem_out_134506") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_134868, &mem_out_134507, "mem_out_134507") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_132649);
        free(mem_132652);
        if (memblock_unref(ctx, &mem_132687, "mem_132687") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_132614, "mem_132614") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_132611, "mem_132611") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134507, "mem_out_134507") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134506, "mem_out_134506") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134505, "mem_out_134505") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_12606(struct futhark_context *ctx, struct memblock *mem_out_p_134871, struct memblock *mem_out_p_134872, struct memblock *mem_out_p_134873, struct memblock w_mem_132605, struct memblock mw_mem_132606, struct memblock vw_mem_132607, struct memblock dw_mem_132608, int64_t n_98821, int64_t m_98822, int64_t step_98827, double lt_r_98828)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_132649_cached_sizze_134874 = 0;
    unsigned char *mem_132649 = NULL;
    int64_t mem_132652_cached_sizze_134875 = 0;
    unsigned char *mem_132652 = NULL;
    struct memblock mem_132687;
    
    mem_132687.references = NULL;
    
    struct memblock mem_132614;
    
    mem_132614.references = NULL;
    
    struct memblock mem_132611;
    
    mem_132611.references = NULL;
    
    struct memblock mem_out_134507;
    
    mem_out_134507.references = NULL;
    
    struct memblock mem_out_134506;
    
    mem_out_134506.references = NULL;
    
    struct memblock mem_out_134505;
    
    mem_out_134505.references = NULL;
    
    struct memblock mem_132596 = ctx->constants->mem_132596;
    struct memblock mem_132597 = ctx->constants->mem_132597;
    struct memblock mem_132598 = ctx->constants->mem_132598;
    struct memblock mem_132599 = ctx->constants->mem_132599;
    struct memblock mem_132600 = ctx->constants->mem_132600;
    struct memblock mem_132601 = ctx->constants->mem_132601;
    struct memblock mem_132602 = ctx->constants->mem_132602;
    struct memblock mem_132603 = ctx->constants->mem_132603;
    struct memblock mem_132604 = ctx->constants->mem_132604;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_132609 = (int64_t) 8 * n_98821;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_132610 = m_98822 * binop_x_132609;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_132611, bytes_132610, "mem_132611")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_132614, bytes_132610, "mem_132614")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131721 = 0; i_131721 < n_98821; i_131721++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131714 = 0; i_131714 < m_98822; i_131714++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_124572 = ((double *) mw_mem_132606.mem)[i_131721 * m_98822 + i_131714];
            
            // futhark/microgpt.fut:462:10-20
            
            double zp_lhs_124573 = 0.85 * zt_rhs_124572;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_124574 = ((double *) dw_mem_132608.mem)[i_131721 * m_98822 + i_131714];
            
            // futhark/microgpt.fut:462:35-45
            
            double zp_rhs_124575 = 0.15000000000000002 * zt_rhs_124574;
            
            // futhark/microgpt.fut:462:21-45
            
            double lifted_lambda_res_124576 = zp_lhs_124573 + zp_rhs_124575;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_124583 = ((double *) vw_mem_132607.mem)[i_131721 * m_98822 + i_131714];
            
            // futhark/microgpt.fut:464:10-20
            
            double zp_lhs_124584 = 0.99 * zt_rhs_124583;
            
            // futhark/microgpt.fut:464:35-45
            
            double zt_lhs_124586 = 1.0000000000000009e-2 * zt_rhs_124574;
            
            // futhark/microgpt.fut:464:46-56
            
            double zp_rhs_124587 = zt_rhs_124574 * zt_lhs_124586;
            
            // futhark/microgpt.fut:464:21-56
            
            double lifted_lambda_res_124588 = zp_lhs_124584 + zp_rhs_124587;
            
            ((double *) mem_132611.mem)[i_131721 * m_98822 + i_131714] = lifted_lambda_res_124588;
            ((double *) mem_132614.mem)[i_131721 * m_98822 + i_131714] = lifted_lambda_res_124576;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_103217 = sitofp_i64_f64(step_98827);
    
    // futhark/microgpt.fut:466:54-57
    
    double ztzt_rhs_103218 = 1.0 + i64_res_103217;
    
    // futhark/microgpt.fut:466:30-57
    
    double zm_rhs_103219 = fpow64(0.85, ztzt_rhs_103218);
    
    // futhark/microgpt.fut:466:23-57
    
    double zs_rhs_103220 = 1.0 - zm_rhs_103219;
    
    // futhark/microgpt.fut:468:31-58
    
    double zm_rhs_103258 = fpow64(0.99, ztzt_rhs_103218);
    
    // futhark/microgpt.fut:468:23-58
    
    double zs_rhs_103259 = 1.0 - zm_rhs_103258;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_132649_cached_sizze_134874 < bytes_132610) {
        err = lexical_realloc(ctx, &mem_132649, &mem_132649_cached_sizze_134874, bytes_132610);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132652_cached_sizze_134875 < bytes_132610) {
        err = lexical_realloc(ctx, &mem_132652, &mem_132652_cached_sizze_134875, bytes_132610);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131735 = 0; i_131735 < n_98821; i_131735++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131728 = 0; i_131728 < m_98822; i_131728++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_124608 = ((double *) mem_132614.mem)[i_131735 * m_98822 + i_131728];
            
            // futhark/microgpt.fut:466:18-57
            
            double lifted_lambda_res_124609 = zs_lhs_124608 / zs_rhs_103220;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_124616 = ((double *) mem_132611.mem)[i_131735 * m_98822 + i_131728];
            
            // futhark/microgpt.fut:468:18-58
            
            double lifted_lambda_res_124617 = zs_lhs_124616 / zs_rhs_103259;
            
            ((double *) mem_132649)[i_131735 * m_98822 + i_131728] = lifted_lambda_res_124617;
            ((double *) mem_132652)[i_131735 * m_98822 + i_131728] = lifted_lambda_res_124609;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_132687, bytes_132610, "mem_132687")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131744 = 0; i_131744 < n_98821; i_131744++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131740 = 0; i_131740 < m_98822; i_131740++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_102308 = ((double *) w_mem_132605.mem)[i_131744 * m_98822 + i_131740];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_102309 = ((double *) mem_132652)[i_131744 * m_98822 + i_131740];
            
            // futhark/microgpt.fut:470:21-34
            
            double zs_lhs_102310 = lt_r_98828 * zt_rhs_102309;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_102311 = ((double *) mem_132649)[i_131744 * m_98822 + i_131740];
            
            // futhark/microgpt.fut:470:51-57
            
            double zp_lhs_102312 = fpow64(ztzt_lhs_102311, 0.5);
            
            // futhark/microgpt.fut:470:59-71
            
            double zs_rhs_102313 = 1.0e-8 + zp_lhs_102312;
            
            // futhark/microgpt.fut:470:35-71
            
            double zm_rhs_102314 = zs_lhs_102310 / zs_rhs_102313;
            
            // futhark/microgpt.fut:470:13-71
            
            double lifted_lambda_res_102315 = zm_lhs_102308 - zm_rhs_102314;
            
            ((double *) mem_132687.mem)[i_131744 * m_98822 + i_131740] = lifted_lambda_res_102315;
        }
    }
    if (memblock_set(ctx, &mem_out_134505, &mem_132687, "mem_132687") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134506, &mem_132614, "mem_132614") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134507, &mem_132611, "mem_132611") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_134871, &mem_out_134505, "mem_out_134505") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_134872, &mem_out_134506, "mem_out_134506") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_134873, &mem_out_134507, "mem_out_134507") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_132649);
        free(mem_132652);
        if (memblock_unref(ctx, &mem_132687, "mem_132687") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_132614, "mem_132614") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_132611, "mem_132611") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134507, "mem_out_134507") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134506, "mem_out_134506") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134505, "mem_out_134505") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_134876, double *out_prim_out_134877, struct memblock wdown_mem_132605, struct memblock wkey_mem_132606, struct memblock wout_mem_132607, struct memblock wpe_mem_132608, struct memblock wqry_mem_132609, struct memblock wte_mem_132610, struct memblock wup_mem_132611, struct memblock wval_mem_132612, struct memblock wvoc_mem_132613, struct memblock tokens_mem_132614, struct memblock target_mem_132615, struct memblock mask_mem_132616)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_132617_cached_sizze_134878 = 0;
    unsigned char *mem_132617 = NULL;
    int64_t mem_132622_cached_sizze_134879 = 0;
    unsigned char *mem_132622 = NULL;
    int64_t mem_132633_cached_sizze_134880 = 0;
    unsigned char *mem_132633 = NULL;
    int64_t mem_132638_cached_sizze_134881 = 0;
    unsigned char *mem_132638 = NULL;
    int64_t mem_132645_cached_sizze_134882 = 0;
    unsigned char *mem_132645 = NULL;
    int64_t mem_132656_cached_sizze_134883 = 0;
    unsigned char *mem_132656 = NULL;
    int64_t mem_132661_cached_sizze_134884 = 0;
    unsigned char *mem_132661 = NULL;
    int64_t mem_132668_cached_sizze_134885 = 0;
    unsigned char *mem_132668 = NULL;
    int64_t mem_132679_cached_sizze_134886 = 0;
    unsigned char *mem_132679 = NULL;
    int64_t mem_132680_cached_sizze_134887 = 0;
    unsigned char *mem_132680 = NULL;
    int64_t mem_132681_cached_sizze_134888 = 0;
    unsigned char *mem_132681 = NULL;
    int64_t mem_132694_cached_sizze_134889 = 0;
    unsigned char *mem_132694 = NULL;
    int64_t mem_132695_cached_sizze_134890 = 0;
    unsigned char *mem_132695 = NULL;
    int64_t mem_132696_cached_sizze_134891 = 0;
    unsigned char *mem_132696 = NULL;
    int64_t mem_132727_cached_sizze_134892 = 0;
    unsigned char *mem_132727 = NULL;
    int64_t mem_132728_cached_sizze_134893 = 0;
    unsigned char *mem_132728 = NULL;
    int64_t mem_132729_cached_sizze_134894 = 0;
    unsigned char *mem_132729 = NULL;
    int64_t mem_132745_cached_sizze_134895 = 0;
    unsigned char *mem_132745 = NULL;
    int64_t mem_132746_cached_sizze_134896 = 0;
    unsigned char *mem_132746 = NULL;
    int64_t mem_132747_cached_sizze_134897 = 0;
    unsigned char *mem_132747 = NULL;
    int64_t mem_132760_cached_sizze_134898 = 0;
    unsigned char *mem_132760 = NULL;
    int64_t mem_132761_cached_sizze_134899 = 0;
    unsigned char *mem_132761 = NULL;
    int64_t mem_132762_cached_sizze_134900 = 0;
    unsigned char *mem_132762 = NULL;
    int64_t mem_132808_cached_sizze_134901 = 0;
    unsigned char *mem_132808 = NULL;
    int64_t mem_132814_cached_sizze_134902 = 0;
    unsigned char *mem_132814 = NULL;
    int64_t mem_132819_cached_sizze_134903 = 0;
    unsigned char *mem_132819 = NULL;
    int64_t mem_132830_cached_sizze_134904 = 0;
    unsigned char *mem_132830 = NULL;
    int64_t mem_132835_cached_sizze_134905 = 0;
    unsigned char *mem_132835 = NULL;
    int64_t mem_132846_cached_sizze_134906 = 0;
    unsigned char *mem_132846 = NULL;
    int64_t mem_132851_cached_sizze_134907 = 0;
    unsigned char *mem_132851 = NULL;
    int64_t mem_132858_cached_sizze_134908 = 0;
    unsigned char *mem_132858 = NULL;
    int64_t mem_132865_cached_sizze_134909 = 0;
    unsigned char *mem_132865 = NULL;
    int64_t mem_132876_cached_sizze_134910 = 0;
    unsigned char *mem_132876 = NULL;
    int64_t mem_132881_cached_sizze_134911 = 0;
    unsigned char *mem_132881 = NULL;
    int64_t mem_132892_cached_sizze_134912 = 0;
    unsigned char *mem_132892 = NULL;
    int64_t mem_132897_cached_sizze_134913 = 0;
    unsigned char *mem_132897 = NULL;
    int64_t mem_132913_cached_sizze_134914 = 0;
    unsigned char *mem_132913 = NULL;
    int64_t mem_132918_cached_sizze_134915 = 0;
    unsigned char *mem_132918 = NULL;
    int64_t mem_132929_cached_sizze_134916 = 0;
    unsigned char *mem_132929 = NULL;
    int64_t mem_132934_cached_sizze_134917 = 0;
    unsigned char *mem_132934 = NULL;
    int64_t mem_132945_cached_sizze_134918 = 0;
    unsigned char *mem_132945 = NULL;
    int64_t mem_132950_cached_sizze_134919 = 0;
    unsigned char *mem_132950 = NULL;
    int64_t mem_132961_cached_sizze_134920 = 0;
    unsigned char *mem_132961 = NULL;
    int64_t mem_132966_cached_sizze_134921 = 0;
    unsigned char *mem_132966 = NULL;
    int64_t mem_132973_cached_sizze_134922 = 0;
    unsigned char *mem_132973 = NULL;
    int64_t mem_132984_cached_sizze_134923 = 0;
    unsigned char *mem_132984 = NULL;
    int64_t mem_132989_cached_sizze_134924 = 0;
    unsigned char *mem_132989 = NULL;
    int64_t mem_133000_cached_sizze_134925 = 0;
    unsigned char *mem_133000 = NULL;
    int64_t mem_133005_cached_sizze_134926 = 0;
    unsigned char *mem_133005 = NULL;
    int64_t mem_133016_cached_sizze_134927 = 0;
    unsigned char *mem_133016 = NULL;
    int64_t mem_133021_cached_sizze_134928 = 0;
    unsigned char *mem_133021 = NULL;
    int64_t mem_133032_cached_sizze_134929 = 0;
    unsigned char *mem_133032 = NULL;
    int64_t mem_133037_cached_sizze_134930 = 0;
    unsigned char *mem_133037 = NULL;
    int64_t mem_133048_cached_sizze_134931 = 0;
    unsigned char *mem_133048 = NULL;
    int64_t mem_133053_cached_sizze_134932 = 0;
    unsigned char *mem_133053 = NULL;
    int64_t mem_133068_cached_sizze_134933 = 0;
    unsigned char *mem_133068 = NULL;
    int64_t mem_133075_cached_sizze_134934 = 0;
    unsigned char *mem_133075 = NULL;
    struct memblock mem_133064;
    
    mem_133064.references = NULL;
    
    struct memblock mem_out_134505;
    
    mem_out_134505.references = NULL;
    
    struct memblock mem_132596 = ctx->constants->mem_132596;
    struct memblock mem_132597 = ctx->constants->mem_132597;
    struct memblock mem_132598 = ctx->constants->mem_132598;
    struct memblock mem_132599 = ctx->constants->mem_132599;
    struct memblock mem_132600 = ctx->constants->mem_132600;
    struct memblock mem_132601 = ctx->constants->mem_132601;
    struct memblock mem_132602 = ctx->constants->mem_132602;
    struct memblock mem_132603 = ctx->constants->mem_132603;
    struct memblock mem_132604 = ctx->constants->mem_132604;
    double prim_out_134506;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_132617_cached_sizze_134878 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132617, &mem_132617_cached_sizze_134878, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132622_cached_sizze_134879 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132622, &mem_132622_cached_sizze_134879, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131716 = 0; i_131716 < (int64_t) 16; i_131716++) {
        // futhark/microgpt.fut:452:41-50
        
        int64_t tmp_109994 = ((int64_t *) tokens_mem_132614.mem)[i_131716];
        
        // futhark/microgpt.fut:452:37-51
        
        bool x_109995 = sle64((int64_t) 0, tmp_109994);
        
        // futhark/microgpt.fut:452:37-51
        
        bool y_109996 = slt64(tmp_109994, (int64_t) 27);
        
        // futhark/microgpt.fut:452:37-51
        
        bool bounds_check_109997 = x_109995 && y_109996;
        
        // futhark/microgpt.fut:452:37-51
        
        bool index_certs_109998;
        
        if (!bounds_check_109997) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_109994, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:452:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:452:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131712 = 0; i_131712 < (int64_t) 16; i_131712++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_110005 = ((double *) wte_mem_132610.mem)[tmp_109994 * (int64_t) 16 + i_131712];
            
            ((double *) mem_132622)[i_131712] = lifted_lambda_res_110005;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132617, i_131716 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132622, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132633_cached_sizze_134880 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132633, &mem_132633_cached_sizze_134880, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132638_cached_sizze_134881 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132638, &mem_132638_cached_sizze_134881, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132645_cached_sizze_134882 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132645, &mem_132645_cached_sizze_134882, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131728 = 0; i_131728 < (int64_t) 16; i_131728++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_110031;
        double r_110033 = 0.0;
        
        for (int64_t i_110032 = 0; i_110032 < (int64_t) 16; i_110032++) {
            // futhark/microgpt.fut:71:46-49
            
            double zp_lhs_110034 = ((double *) wpe_mem_132608.mem)[i_131728 * (int64_t) 16 + i_110032];
            
            // futhark/microgpt.fut:71:46-49
            
            double zp_rhs_110035 = ((double *) mem_132617)[i_131728 * (int64_t) 16 + i_110032];
            
            // futhark/microgpt.fut:203:76-116
            
            double zp_res_110036 = zp_lhs_110034 + zp_rhs_110035;
            
            // futhark/microgpt.fut:203:94-163
            
            double zt_res_110037 = zp_res_110036 * zp_res_110036;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_110038 = r_110033 + zt_res_110037;
            double r_tmp_134510 = zp_res_110038;
            
            r_110033 = r_tmp_134510;
        }
        defunc_0_lifted_lambda_res_110031 = r_110033;
        // futhark/microgpt.fut:203:54-182
        
        double zs_res_110039 = defunc_0_lifted_lambda_res_110031 / 16.0;
        
        // futhark/microgpt.fut:204:24-55
        
        double zp_res_110040 = 1.0e-5 + zs_res_110039;
        
        // futhark/microgpt.fut:204:16-55
        
        double sqrt_res_110041 = futrts_sqrt64(zp_res_110040);
        
        // futhark/microgpt.fut:205:85-96
        
        double zs_res_110042 = 1.0 / sqrt_res_110041;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131720 = 0; i_131720 < (int64_t) 16; i_131720++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_110049 = ((double *) wpe_mem_132608.mem)[i_131728 * (int64_t) 16 + i_131720];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_110050 = ((double *) mem_132617)[i_131728 * (int64_t) 16 + i_131720];
            
            // futhark/microgpt.fut:205:38-78
            
            double zp_res_110051 = zp_lhs_110049 + zp_rhs_110050;
            
            // futhark/microgpt.fut:205:56-96
            
            double zt_res_110052 = zs_res_110042 * zp_res_110051;
            
            ((double *) mem_132638)[i_131720] = zt_res_110052;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131724 = 0; i_131724 < (int64_t) 16; i_131724++) {
            // futhark/microgpt.fut:206:4-14
            
            double lifted_lambda_res_110060 = ((double *) mem_132638)[i_131724];
            
            ((double *) mem_132645)[i_131724] = lifted_lambda_res_110060;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132633, i_131728 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132645, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132656_cached_sizze_134883 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132656, &mem_132656_cached_sizze_134883, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132661_cached_sizze_134884 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132661, &mem_132661_cached_sizze_134884, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132668_cached_sizze_134885 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132668, &mem_132668_cached_sizze_134885, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131740 = 0; i_131740 < (int64_t) 16; i_131740++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_110069;
        double r_110071 = 0.0;
        
        for (int64_t i_110070 = 0; i_110070 < (int64_t) 16; i_110070++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_110072 = ((double *) mem_132633)[i_131740 * (int64_t) 16 + i_110070];
            
            // futhark/microgpt.fut:207:78-115
            
            double zt_res_110073 = zt_lhs_110072 * zt_lhs_110072;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_110074 = r_110071 + zt_res_110073;
            double r_tmp_134514 = zp_res_110074;
            
            r_110071 = r_tmp_134514;
        }
        defunc_0_lifted_lambda_res_110069 = r_110071;
        // futhark/microgpt.fut:207:57-133
        
        double zs_res_110075 = defunc_0_lifted_lambda_res_110069 / 16.0;
        
        // futhark/microgpt.fut:208:24-55
        
        double zp_res_110076 = 1.0e-5 + zs_res_110075;
        
        // futhark/microgpt.fut:208:16-55
        
        double sqrt_res_110077 = futrts_sqrt64(zp_res_110076);
        
        // futhark/microgpt.fut:209:59-70
        
        double zs_res_110078 = 1.0 / sqrt_res_110077;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131732 = 0; i_131732 < (int64_t) 16; i_131732++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_110085 = ((double *) mem_132633)[i_131740 * (int64_t) 16 + i_131732];
            
            // futhark/microgpt.fut:209:37-70
            
            double zt_res_110086 = zs_res_110078 * zt_lhs_110085;
            
            ((double *) mem_132661)[i_131732] = zt_res_110086;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131736 = 0; i_131736 < (int64_t) 16; i_131736++) {
            // futhark/microgpt.fut:210:4-14
            
            double lifted_lambda_res_110094 = ((double *) mem_132661)[i_131736];
            
            ((double *) mem_132668)[i_131736] = lifted_lambda_res_110094;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132656, i_131740 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132668, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132679_cached_sizze_134886 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132679, &mem_132679_cached_sizze_134886, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132680_cached_sizze_134887 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132680, &mem_132680_cached_sizze_134887, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132681_cached_sizze_134888 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132681, &mem_132681_cached_sizze_134888, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132694_cached_sizze_134889 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132694, &mem_132694_cached_sizze_134889, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132695_cached_sizze_134890 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132695, &mem_132695_cached_sizze_134890, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132696_cached_sizze_134891 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132696, &mem_132696_cached_sizze_134891, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131758 = 0; i_131758 < (int64_t) 16; i_131758++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131748 = 0; i_131748 < (int64_t) 16; i_131748++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124790;
            double r_124792 = 0.0;
            
            for (int64_t i_124791 = 0; i_124791 < (int64_t) 16; i_124791++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_124793 = ((double *) wqry_mem_132609.mem)[i_131748 * (int64_t) 16 + i_124791];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_124794 = ((double *) mem_132656)[i_131758 * (int64_t) 16 + i_124791];
                
                // futhark/microgpt.fut:211:66-105
                
                double zt_res_124795 = zt_lhs_124793 * zt_rhs_124794;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_124796 = r_124792 + zt_res_124795;
                double r_tmp_134523 = zp_res_124796;
                
                r_124792 = r_tmp_134523;
            }
            defunc_0_lifted_lambda_res_124790 = r_124792;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124803;
            double r_124805 = 0.0;
            
            for (int64_t i_124804 = 0; i_124804 < (int64_t) 16; i_124804++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_124806 = ((double *) wkey_mem_132606.mem)[i_131748 * (int64_t) 16 + i_124804];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_124807 = ((double *) mem_132656)[i_131758 * (int64_t) 16 + i_124804];
                
                // futhark/microgpt.fut:212:66-105
                
                double zt_res_124808 = zt_lhs_124806 * zt_rhs_124807;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_124809 = r_124805 + zt_res_124808;
                double r_tmp_134524 = zp_res_124809;
                
                r_124805 = r_tmp_134524;
            }
            defunc_0_lifted_lambda_res_124803 = r_124805;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124819;
            double r_124821 = 0.0;
            
            for (int64_t i_124820 = 0; i_124820 < (int64_t) 16; i_124820++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_124822 = ((double *) wval_mem_132612.mem)[i_131748 * (int64_t) 16 + i_124820];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_124823 = ((double *) mem_132656)[i_131758 * (int64_t) 16 + i_124820];
                
                // futhark/microgpt.fut:213:66-105
                
                double zt_res_124824 = zt_lhs_124822 * zt_rhs_124823;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_124825 = r_124821 + zt_res_124824;
                double r_tmp_134525 = zp_res_124825;
                
                r_124821 = r_tmp_134525;
            }
            defunc_0_lifted_lambda_res_124819 = r_124821;
            ((double *) mem_132694)[i_131748] = defunc_0_lifted_lambda_res_124819;
            ((double *) mem_132695)[i_131748] = defunc_0_lifted_lambda_res_124803;
            ((double *) mem_132696)[i_131748] = defunc_0_lifted_lambda_res_124790;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132679, i_131758 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132694, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132680, i_131758 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132695, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132681, i_131758 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132696, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132727_cached_sizze_134892 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132727, &mem_132727_cached_sizze_134892, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132728_cached_sizze_134893 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132728, &mem_132728_cached_sizze_134893, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132729_cached_sizze_134894 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132729, &mem_132729_cached_sizze_134894, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132745_cached_sizze_134895 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_132745, &mem_132745_cached_sizze_134895, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132746_cached_sizze_134896 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_132746, &mem_132746_cached_sizze_134896, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132747_cached_sizze_134897 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_132747, &mem_132747_cached_sizze_134897, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132760_cached_sizze_134898 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_132760, &mem_132760_cached_sizze_134898, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132761_cached_sizze_134899 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_132761, &mem_132761_cached_sizze_134899, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132762_cached_sizze_134900 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_132762, &mem_132762_cached_sizze_134900, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131788 = 0; i_131788 < (int64_t) 4; i_131788++) {
        // futhark/microgpt.fut:214:69-72
        
        int64_t zp_lhs_124666 = mul64((int64_t) 4, i_131788);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131778 = 0; i_131778 < (int64_t) 16; i_131778++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131768 = 0; i_131768 < (int64_t) 4; i_131768++) {
                // futhark/microgpt.fut:214:74-81
                
                int64_t tmp_124983 = add64(zp_lhs_124666, i_131768);
                
                // futhark/microgpt.fut:214:51-83
                
                bool x_124984 = sle64((int64_t) 0, tmp_124983);
                
                // futhark/microgpt.fut:214:51-83
                
                bool y_124985 = slt64(tmp_124983, (int64_t) 16);
                
                // futhark/microgpt.fut:214:51-83
                
                bool bounds_check_124986 = x_124984 && y_124985;
                
                // futhark/microgpt.fut:214:51-83
                
                bool index_certs_124987;
                
                if (!bounds_check_124986) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_124983, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:214:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:214:15-84\n   #9  futhark/microgpt.fut:453:7-76\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_124988 = ((double *) mem_132681)[i_131778 * (int64_t) 16 + tmp_124983];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_124996 = ((double *) mem_132680)[i_131778 * (int64_t) 16 + tmp_124983];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_125007 = ((double *) mem_132679)[i_131778 * (int64_t) 16 + tmp_124983];
                
                ((double *) mem_132760)[i_131768] = lifted_lambda_res_125007;
                ((double *) mem_132761)[i_131768] = lifted_lambda_res_124996;
                ((double *) mem_132762)[i_131768] = lifted_lambda_res_124988;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132745, i_131778 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132760, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132746, i_131778 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132761, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132747, i_131778 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132762, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_132727, i_131788 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_132745, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_132728, i_131788 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_132746, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_132729, i_131788 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_132747, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132808_cached_sizze_134901 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132808, &mem_132808_cached_sizze_134901, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132814_cached_sizze_134902 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132814, &mem_132814_cached_sizze_134902, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132819_cached_sizze_134903 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132819, &mem_132819_cached_sizze_134903, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132830_cached_sizze_134904 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132830, &mem_132830_cached_sizze_134904, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132835_cached_sizze_134905 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132835, &mem_132835_cached_sizze_134905, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132846_cached_sizze_134906 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132846, &mem_132846_cached_sizze_134906, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132851_cached_sizze_134907 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132851, &mem_132851_cached_sizze_134907, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132858_cached_sizze_134908 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132858, &mem_132858_cached_sizze_134908, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132865_cached_sizze_134909 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132865, &mem_132865_cached_sizze_134909, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132876_cached_sizze_134910 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_132876, &mem_132876_cached_sizze_134910, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132881_cached_sizze_134911 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_132881, &mem_132881_cached_sizze_134911, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132892_cached_sizze_134912 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_132892, &mem_132892_cached_sizze_134912, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132897_cached_sizze_134913 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_132897, &mem_132897_cached_sizze_134913, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131844 = 0; i_131844 < (int64_t) 4; i_131844++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131798 = 0; i_131798 < (int64_t) 16; i_131798++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131794 = 0; i_131794 < (int64_t) 16; i_131794++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_110239;
                double r_110241 = 0.0;
                
                for (int64_t i_110240 = 0; i_110240 < (int64_t) 4; i_110240++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_110242 = ((double *) mem_132729)[i_131844 * (int64_t) 64 + i_131798 * (int64_t) 4 + i_110240];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_110243 = ((double *) mem_132728)[i_131844 * (int64_t) 64 + i_131794 * (int64_t) 4 + i_110240];
                    
                    // futhark/microgpt.fut:217:113-164
                    
                    double zt_res_110244 = zt_lhs_110242 * zt_rhs_110243;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_110245 = r_110241 + zt_res_110244;
                    double r_tmp_134538 = zp_res_110245;
                    
                    r_110241 = r_tmp_134538;
                }
                defunc_0_lifted_lambda_res_110239 = r_110241;
                ((double *) mem_132819)[i_131794] = defunc_0_lifted_lambda_res_110239;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132814, i_131798 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132819, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131806 = 0; i_131806 < (int64_t) 16; i_131806++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131802 = 0; i_131802 < (int64_t) 16; i_131802++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_110260 = ((double *) mem_132814)[i_131806 * (int64_t) 16 + i_131802];
                
                // futhark/microgpt.fut:218:47-78
                
                double zs_res_110261 = zs_lhs_110260 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_110262 = ((double *) mask_mem_132616.mem)[i_131806 * (int64_t) 16 + i_131802];
                
                // futhark/microgpt.fut:218:65-102
                
                double zp_res_110263 = zs_res_110261 + zp_rhs_110262;
                
                ((double *) mem_132835)[i_131802] = zp_res_110263;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132830, i_131806 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132835, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131824 = 0; i_131824 < (int64_t) 16; i_131824++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_125110;
            double redout_131808 = -INFINITY;
            
            for (int64_t i_131809 = 0; i_131809 < (int64_t) 16; i_131809++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_125034 = ((double *) mem_132830)[i_131824 * (int64_t) 16 + i_131809];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_110284 = fmax64(lifted_lambda_res_125034, redout_131808);
                double redout_tmp_134542 = max_res_110284;
                
                redout_131808 = redout_tmp_134542;
            }
            defunc_0_reduce_res_125110 = redout_131808;
            // futhark/microgpt.fut:220:67-76
            
            double neg_res_110285 = -defunc_0_reduce_res_125110;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131812 = 0; i_131812 < (int64_t) 16; i_131812++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_110292 = ((double *) mem_132830)[i_131824 * (int64_t) 16 + i_131812];
                
                // futhark/microgpt.fut:220:44-76
                
                double zp_res_110293 = neg_res_110285 + zp_lhs_110292;
                
                // futhark/microgpt.fut:220:37-76
                
                double exp_res_110294 = futrts_exp64(zp_res_110293);
                
                ((double *) mem_132851)[i_131812] = exp_res_110294;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_110296;
            double r_110298 = 0.0;
            
            for (int64_t i_110297 = 0; i_110297 < (int64_t) 16; i_110297++) {
                // futhark/microgpt.fut:221:36-46
                
                double lifted_lambda_res_110299 = ((double *) mem_132851)[i_110297];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_110300 = r_110298 + lifted_lambda_res_110299;
                double r_tmp_134544 = zp_res_110300;
                
                r_110298 = r_tmp_134544;
            }
            defunc_0_lifted_lambda_res_110296 = r_110298;
            // futhark/microgpt.fut:222:53-64
            
            double zs_res_110301 = 1.0 / defunc_0_lifted_lambda_res_110296;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131816 = 0; i_131816 < (int64_t) 16; i_131816++) {
                // futhark/microgpt.fut:222:37-47
                
                double zt_lhs_110308 = ((double *) mem_132851)[i_131816];
                
                // futhark/microgpt.fut:222:37-64
                
                double zt_res_110309 = zs_res_110301 * zt_lhs_110308;
                
                ((double *) mem_132858)[i_131816] = zt_res_110309;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131820 = 0; i_131820 < (int64_t) 16; i_131820++) {
                // futhark/microgpt.fut:223:4-14
                
                double lifted_lambda_res_110317 = ((double *) mem_132858)[i_131820];
                
                ((double *) mem_132865)[i_131820] = lifted_lambda_res_110317;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132846, i_131824 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132865, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131832 = 0; i_131832 < (int64_t) 16; i_131832++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131828 = 0; i_131828 < (int64_t) 4; i_131828++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_110332;
                double r_110334 = 0.0;
                
                for (int64_t i_110333 = 0; i_110333 < (int64_t) 16; i_110333++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_110335 = ((double *) mem_132846)[i_131832 * (int64_t) 16 + i_110333];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_110336 = ((double *) mem_132727)[i_131844 * (int64_t) 64 + i_110333 * (int64_t) 4 + i_131828];
                    
                    // futhark/microgpt.fut:224:66-111
                    
                    double zt_res_110337 = zt_lhs_110335 * zt_rhs_110336;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_110338 = r_110334 + zt_res_110337;
                    double r_tmp_134549 = zp_res_110338;
                    
                    r_110334 = r_tmp_134549;
                }
                defunc_0_lifted_lambda_res_110332 = r_110334;
                ((double *) mem_132881)[i_131828] = defunc_0_lifted_lambda_res_110332;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132876, i_131832 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132881, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131840 = 0; i_131840 < (int64_t) 16; i_131840++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131836 = 0; i_131836 < (int64_t) 4; i_131836++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_110353 = ((double *) mem_132876)[i_131840 * (int64_t) 4 + i_131836];
                
                ((double *) mem_132897)[i_131836] = lifted_lambda_res_110353;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132892, i_131840 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132897, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_132808, i_131844 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_132892, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132913_cached_sizze_134914 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132913, &mem_132913_cached_sizze_134914, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132918_cached_sizze_134915 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132918, &mem_132918_cached_sizze_134915, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131852 = 0; i_131852 < (int64_t) 16; i_131852++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131848 = 0; i_131848 < (int64_t) 16; i_131848++) {
            // futhark/microgpt.fut:226:54-57
            
            int64_t tmp_110365 = sdiv64(i_131848, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-59
            
            bool x_110366 = sle64((int64_t) 0, tmp_110365);
            
            // futhark/microgpt.fut:226:44-59
            
            bool y_110367 = slt64(tmp_110365, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-59
            
            bool bounds_check_110368 = x_110366 && y_110367;
            
            // futhark/microgpt.fut:226:44-59
            
            bool index_certs_110369;
            
            if (!bounds_check_110368) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_110365, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:226:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:226:15-80\n   #6  futhark/microgpt.fut:453:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:226:74-77
            
            int64_t tmp_110370 = smod64(i_131848, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-79
            
            bool x_110371 = sle64((int64_t) 0, tmp_110370);
            
            // futhark/microgpt.fut:226:44-79
            
            bool y_110372 = slt64(tmp_110370, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-79
            
            bool bounds_check_110373 = x_110371 && y_110372;
            
            // futhark/microgpt.fut:226:44-79
            
            bool index_certs_110374;
            
            if (!bounds_check_110373) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_110370, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:226:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:226:15-80\n   #6  futhark/microgpt.fut:453:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_110375 = ((double *) mem_132808)[tmp_110365 * (int64_t) 64 + i_131852 * (int64_t) 4 + tmp_110370];
            
            ((double *) mem_132918)[i_131848] = lifted_lambda_res_110375;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132913, i_131852 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132918, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132929_cached_sizze_134916 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132929, &mem_132929_cached_sizze_134916, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132934_cached_sizze_134917 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132934, &mem_132934_cached_sizze_134917, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131860 = 0; i_131860 < (int64_t) 16; i_131860++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131856 = 0; i_131856 < (int64_t) 16; i_131856++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_110390;
            double r_110392 = 0.0;
            
            for (int64_t i_110391 = 0; i_110391 < (int64_t) 16; i_110391++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_110393 = ((double *) wout_mem_132607.mem)[i_131856 * (int64_t) 16 + i_110391];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_110394 = ((double *) mem_132913)[i_131860 * (int64_t) 16 + i_110391];
                
                // futhark/microgpt.fut:227:67-106
                
                double zt_res_110395 = zt_lhs_110393 * zt_rhs_110394;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_110396 = r_110392 + zt_res_110395;
                double r_tmp_134556 = zp_res_110396;
                
                r_110392 = r_tmp_134556;
            }
            defunc_0_lifted_lambda_res_110390 = r_110392;
            ((double *) mem_132934)[i_131856] = defunc_0_lifted_lambda_res_110390;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132929, i_131860 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132934, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132945_cached_sizze_134918 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132945, &mem_132945_cached_sizze_134918, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132950_cached_sizze_134919 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132950, &mem_132950_cached_sizze_134919, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131868 = 0; i_131868 < (int64_t) 16; i_131868++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131864 = 0; i_131864 < (int64_t) 16; i_131864++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_110411 = ((double *) mem_132929)[i_131868 * (int64_t) 16 + i_131864];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_110412 = ((double *) mem_132633)[i_131868 * (int64_t) 16 + i_131864];
            
            // futhark/microgpt.fut:228:46-84
            
            double zp_res_110413 = zp_lhs_110411 + zp_rhs_110412;
            
            ((double *) mem_132950)[i_131864] = zp_res_110413;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132945, i_131868 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132950, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132961_cached_sizze_134920 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132961, &mem_132961_cached_sizze_134920, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132966_cached_sizze_134921 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132966, &mem_132966_cached_sizze_134921, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132973_cached_sizze_134922 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132973, &mem_132973_cached_sizze_134922, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131880 = 0; i_131880 < (int64_t) 16; i_131880++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_110422;
        double r_110424 = 0.0;
        
        for (int64_t i_110423 = 0; i_110423 < (int64_t) 16; i_110423++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_110425 = ((double *) mem_132945)[i_131880 * (int64_t) 16 + i_110423];
            
            // futhark/microgpt.fut:229:79-118
            
            double zt_res_110426 = zt_lhs_110425 * zt_lhs_110425;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_110427 = r_110424 + zt_res_110426;
            double r_tmp_134560 = zp_res_110427;
            
            r_110424 = r_tmp_134560;
        }
        defunc_0_lifted_lambda_res_110422 = r_110424;
        // futhark/microgpt.fut:229:58-136
        
        double zs_res_110428 = defunc_0_lifted_lambda_res_110422 / 16.0;
        
        // futhark/microgpt.fut:230:24-55
        
        double zp_res_110429 = 1.0e-5 + zs_res_110428;
        
        // futhark/microgpt.fut:230:16-55
        
        double sqrt_res_110430 = futrts_sqrt64(zp_res_110429);
        
        // futhark/microgpt.fut:231:60-71
        
        double zs_res_110431 = 1.0 / sqrt_res_110430;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131872 = 0; i_131872 < (int64_t) 16; i_131872++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_110438 = ((double *) mem_132945)[i_131880 * (int64_t) 16 + i_131872];
            
            // futhark/microgpt.fut:231:37-71
            
            double zt_res_110439 = zs_res_110431 * zt_lhs_110438;
            
            ((double *) mem_132966)[i_131872] = zt_res_110439;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131876 = 0; i_131876 < (int64_t) 16; i_131876++) {
            // futhark/microgpt.fut:232:4-14
            
            double lifted_lambda_res_110447 = ((double *) mem_132966)[i_131876];
            
            ((double *) mem_132973)[i_131876] = lifted_lambda_res_110447;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132961, i_131880 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132973, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132984_cached_sizze_134923 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_132984, &mem_132984_cached_sizze_134923, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132989_cached_sizze_134924 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_132989, &mem_132989_cached_sizze_134924, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131888 = 0; i_131888 < (int64_t) 16; i_131888++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131884 = 0; i_131884 < (int64_t) 64; i_131884++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_110463;
            double r_110465 = 0.0;
            
            for (int64_t i_110464 = 0; i_110464 < (int64_t) 16; i_110464++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_110466 = ((double *) wup_mem_132611.mem)[i_131884 * (int64_t) 16 + i_110464];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_110467 = ((double *) mem_132961)[i_131888 * (int64_t) 16 + i_110464];
                
                // futhark/microgpt.fut:233:67-106
                
                double zt_res_110468 = zt_lhs_110466 * zt_rhs_110467;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_110469 = r_110465 + zt_res_110468;
                double r_tmp_134565 = zp_res_110469;
                
                r_110465 = r_tmp_134565;
            }
            defunc_0_lifted_lambda_res_110463 = r_110465;
            ((double *) mem_132989)[i_131884] = defunc_0_lifted_lambda_res_110463;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132984, i_131888 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132989, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133000_cached_sizze_134925 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_133000, &mem_133000_cached_sizze_134925, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133005_cached_sizze_134926 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133005, &mem_133005_cached_sizze_134926, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131896 = 0; i_131896 < (int64_t) 16; i_131896++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131892 = 0; i_131892 < (int64_t) 64; i_131892++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_110484 = ((double *) mem_132984)[i_131896 * (int64_t) 64 + i_131892];
            
            // futhark/microgpt.fut:234:45-73
            
            double max_res_110485 = fmax64(0.0, max_arg0_110484);
            
            ((double *) mem_133005)[i_131892] = max_res_110485;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_133000, i_131896 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133005, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133016_cached_sizze_134927 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133016, &mem_133016_cached_sizze_134927, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133021_cached_sizze_134928 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133021, &mem_133021_cached_sizze_134928, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131904 = 0; i_131904 < (int64_t) 16; i_131904++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131900 = 0; i_131900 < (int64_t) 16; i_131900++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_110500;
            double r_110502 = 0.0;
            
            for (int64_t i_110501 = 0; i_110501 < (int64_t) 64; i_110501++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_110503 = ((double *) wdown_mem_132605.mem)[i_131900 * (int64_t) 64 + i_110501];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_110504 = ((double *) mem_133000)[i_131904 * (int64_t) 64 + i_110501];
                
                // futhark/microgpt.fut:235:67-108
                
                double zt_res_110505 = zt_lhs_110503 * zt_rhs_110504;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_110506 = r_110502 + zt_res_110505;
                double r_tmp_134570 = zp_res_110506;
                
                r_110502 = r_tmp_134570;
            }
            defunc_0_lifted_lambda_res_110500 = r_110502;
            ((double *) mem_133021)[i_131900] = defunc_0_lifted_lambda_res_110500;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_133016, i_131904 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133021, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133032_cached_sizze_134929 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133032, &mem_133032_cached_sizze_134929, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133037_cached_sizze_134930 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133037, &mem_133037_cached_sizze_134930, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131912 = 0; i_131912 < (int64_t) 16; i_131912++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131908 = 0; i_131908 < (int64_t) 16; i_131908++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_110521 = ((double *) mem_133016)[i_131912 * (int64_t) 16 + i_131908];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_110522 = ((double *) mem_132945)[i_131912 * (int64_t) 16 + i_131908];
            
            // futhark/microgpt.fut:236:46-85
            
            double zp_res_110523 = zp_lhs_110521 + zp_rhs_110522;
            
            ((double *) mem_133037)[i_131908] = zp_res_110523;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_133032, i_131912 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133037, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133048_cached_sizze_134931 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_133048, &mem_133048_cached_sizze_134931, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133053_cached_sizze_134932 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_133053, &mem_133053_cached_sizze_134932, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131920 = 0; i_131920 < (int64_t) 16; i_131920++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131916 = 0; i_131916 < (int64_t) 27; i_131916++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_110539;
            double r_110541 = 0.0;
            
            for (int64_t i_110540 = 0; i_110540 < (int64_t) 16; i_110540++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_110542 = ((double *) wvoc_mem_132613.mem)[i_131916 * (int64_t) 16 + i_110540];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_110543 = ((double *) mem_133032)[i_131920 * (int64_t) 16 + i_110540];
                
                // futhark/microgpt.fut:237:67-107
                
                double zt_res_110544 = zt_lhs_110542 * zt_rhs_110543;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_110545 = r_110541 + zt_res_110544;
                double r_tmp_134575 = zp_res_110545;
                
                r_110541 = r_tmp_134575;
            }
            defunc_0_lifted_lambda_res_110539 = r_110541;
            ((double *) mem_133053)[i_131916] = defunc_0_lifted_lambda_res_110539;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_133048, i_131920 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133053, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_133064, (int64_t) 128, "mem_133064")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133068_cached_sizze_134933 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_133068, &mem_133068_cached_sizze_134933, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133075_cached_sizze_134934 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_133075, &mem_133075_cached_sizze_134934, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131934 = 0; i_131934 < (int64_t) 16; i_131934++) {
        double x_125133;
        double redout_131922 = -INFINITY;
        
        for (int64_t i_131923 = 0; i_131923 < (int64_t) 27; i_131923++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_125080 = ((double *) mem_133048)[i_131934 * (int64_t) 27 + i_131923];
            
            // futhark/microgpt.fut:115:13-33
            
            double max_res_110569 = fmax64(lifted_lambda_res_125080, redout_131922);
            double redout_tmp_134577 = max_res_110569;
            
            redout_131922 = redout_tmp_134577;
        }
        x_125133 = redout_131922;
        // futhark/microgpt.fut:239:67-76
        
        double neg_res_110570 = -x_125133;
        
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_110554;
        double r_110556 = 0.0;
        
        for (int64_t i_110555 = 0; i_110555 < (int64_t) 27; i_110555++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131926 = 0; i_131926 < (int64_t) 27; i_131926++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_110577 = ((double *) mem_133048)[i_131934 * (int64_t) 27 + i_131926];
                
                // futhark/microgpt.fut:239:44-76
                
                double zp_res_110578 = neg_res_110570 + zp_lhs_110577;
                
                // futhark/microgpt.fut:239:37-76
                
                double exp_res_110579 = futrts_exp64(zp_res_110578);
                
                ((double *) mem_133068)[i_131926] = exp_res_110579;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_110581;
            double r_110583 = 0.0;
            
            for (int64_t i_110582 = 0; i_110582 < (int64_t) 27; i_110582++) {
                // futhark/microgpt.fut:240:36-46
                
                double lifted_lambda_res_110584 = ((double *) mem_133068)[i_110582];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_110585 = r_110583 + lifted_lambda_res_110584;
                double r_tmp_134580 = zp_res_110585;
                
                r_110583 = r_tmp_134580;
            }
            defunc_0_lifted_lambda_res_110581 = r_110583;
            // futhark/microgpt.fut:241:53-64
            
            double zs_res_110586 = 1.0 / defunc_0_lifted_lambda_res_110581;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131930 = 0; i_131930 < (int64_t) 27; i_131930++) {
                // futhark/microgpt.fut:241:37-47
                
                double zt_lhs_110593 = ((double *) mem_133068)[i_131930];
                
                // futhark/microgpt.fut:241:37-64
                
                double zt_res_110594 = zs_res_110586 * zt_lhs_110593;
                
                ((double *) mem_133075)[i_131930] = zt_res_110594;
            }
            // futhark/microgpt.fut:242:12-22
            
            double log_arg0_110596 = ((double *) mem_133075)[i_110555];
            
            // futhark/microgpt.fut:242:6-22
            
            double log_res_110597 = futrts_log64(log_arg0_110596);
            
            // futhark/microgpt.fut:71:46-49
            
            double zt_rhs_110598 = ((double *) target_mem_132615.mem)[i_131934 * (int64_t) 27 + i_110555];
            
            // futhark/microgpt.fut:242:6-48
            
            double zt_res_110599 = log_res_110597 * zt_rhs_110598;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_110600 = r_110556 + zt_res_110599;
            double r_tmp_134578 = zp_res_110600;
            
            r_110556 = r_tmp_134578;
        }
        defunc_0_lifted_lambda_res_110554 = r_110556;
        // futhark/microgpt.fut:238:37-242:54
        
        double neg_res_110601 = -defunc_0_lifted_lambda_res_110554;
        
        ((double *) mem_133064.mem)[i_131934] = neg_res_110601;
    }
    // futhark/microgpt.fut:71:13-49
    
    double defunc_0_lifted_lambda_res_110603;
    double r_110605 = 0.0;
    
    for (int64_t i_110604 = 0; i_110604 < (int64_t) 16; i_110604++) {
        // futhark/microgpt.fut:243:37-47
        
        double lifted_lambda_res_110606 = ((double *) mem_133064.mem)[i_110604];
        
        // futhark/microgpt.fut:71:40-49
        
        double zp_res_110607 = r_110605 + lifted_lambda_res_110606;
        double r_tmp_134582 = zp_res_110607;
        
        r_110605 = r_tmp_134582;
    }
    defunc_0_lifted_lambda_res_110603 = r_110605;
    // futhark/microgpt.fut:243:17-64
    
    double zs_res_110608 = defunc_0_lifted_lambda_res_110603 / 16.0;
    
    if (memblock_set(ctx, &mem_out_134505, &mem_133064, "mem_133064") != 0)
        return 1;
    prim_out_134506 = zs_res_110608;
    if (memblock_set(ctx, &*mem_out_p_134876, &mem_out_134505, "mem_out_134505") != 0)
        return 1;
    *out_prim_out_134877 = prim_out_134506;
    
  cleanup:
    {
        free(mem_132617);
        free(mem_132622);
        free(mem_132633);
        free(mem_132638);
        free(mem_132645);
        free(mem_132656);
        free(mem_132661);
        free(mem_132668);
        free(mem_132679);
        free(mem_132680);
        free(mem_132681);
        free(mem_132694);
        free(mem_132695);
        free(mem_132696);
        free(mem_132727);
        free(mem_132728);
        free(mem_132729);
        free(mem_132745);
        free(mem_132746);
        free(mem_132747);
        free(mem_132760);
        free(mem_132761);
        free(mem_132762);
        free(mem_132808);
        free(mem_132814);
        free(mem_132819);
        free(mem_132830);
        free(mem_132835);
        free(mem_132846);
        free(mem_132851);
        free(mem_132858);
        free(mem_132865);
        free(mem_132876);
        free(mem_132881);
        free(mem_132892);
        free(mem_132897);
        free(mem_132913);
        free(mem_132918);
        free(mem_132929);
        free(mem_132934);
        free(mem_132945);
        free(mem_132950);
        free(mem_132961);
        free(mem_132966);
        free(mem_132973);
        free(mem_132984);
        free(mem_132989);
        free(mem_133000);
        free(mem_133005);
        free(mem_133016);
        free(mem_133021);
        free(mem_133032);
        free(mem_133037);
        free(mem_133048);
        free(mem_133053);
        free(mem_133068);
        free(mem_133075);
        if (memblock_unref(ctx, &mem_133064, "mem_133064") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134505, "mem_out_134505") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_134935, struct memblock wdown_mem_132605, struct memblock wkey_mem_132606, struct memblock wout_mem_132607, struct memblock wpe_mem_132608, struct memblock wqry_mem_132609, struct memblock wte_mem_132610, struct memblock wup_mem_132611, struct memblock wval_mem_132612, struct memblock wvoc_mem_132613, struct memblock tokens_mem_132614, struct memblock mask_mem_132615)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_132616_cached_sizze_134936 = 0;
    unsigned char *mem_132616 = NULL;
    int64_t mem_132621_cached_sizze_134937 = 0;
    unsigned char *mem_132621 = NULL;
    int64_t mem_132632_cached_sizze_134938 = 0;
    unsigned char *mem_132632 = NULL;
    int64_t mem_132637_cached_sizze_134939 = 0;
    unsigned char *mem_132637 = NULL;
    int64_t mem_132644_cached_sizze_134940 = 0;
    unsigned char *mem_132644 = NULL;
    int64_t mem_132655_cached_sizze_134941 = 0;
    unsigned char *mem_132655 = NULL;
    int64_t mem_132660_cached_sizze_134942 = 0;
    unsigned char *mem_132660 = NULL;
    int64_t mem_132667_cached_sizze_134943 = 0;
    unsigned char *mem_132667 = NULL;
    int64_t mem_132678_cached_sizze_134944 = 0;
    unsigned char *mem_132678 = NULL;
    int64_t mem_132679_cached_sizze_134945 = 0;
    unsigned char *mem_132679 = NULL;
    int64_t mem_132680_cached_sizze_134946 = 0;
    unsigned char *mem_132680 = NULL;
    int64_t mem_132693_cached_sizze_134947 = 0;
    unsigned char *mem_132693 = NULL;
    int64_t mem_132694_cached_sizze_134948 = 0;
    unsigned char *mem_132694 = NULL;
    int64_t mem_132695_cached_sizze_134949 = 0;
    unsigned char *mem_132695 = NULL;
    int64_t mem_132726_cached_sizze_134950 = 0;
    unsigned char *mem_132726 = NULL;
    int64_t mem_132727_cached_sizze_134951 = 0;
    unsigned char *mem_132727 = NULL;
    int64_t mem_132728_cached_sizze_134952 = 0;
    unsigned char *mem_132728 = NULL;
    int64_t mem_132744_cached_sizze_134953 = 0;
    unsigned char *mem_132744 = NULL;
    int64_t mem_132745_cached_sizze_134954 = 0;
    unsigned char *mem_132745 = NULL;
    int64_t mem_132746_cached_sizze_134955 = 0;
    unsigned char *mem_132746 = NULL;
    int64_t mem_132759_cached_sizze_134956 = 0;
    unsigned char *mem_132759 = NULL;
    int64_t mem_132760_cached_sizze_134957 = 0;
    unsigned char *mem_132760 = NULL;
    int64_t mem_132761_cached_sizze_134958 = 0;
    unsigned char *mem_132761 = NULL;
    int64_t mem_132807_cached_sizze_134959 = 0;
    unsigned char *mem_132807 = NULL;
    int64_t mem_132813_cached_sizze_134960 = 0;
    unsigned char *mem_132813 = NULL;
    int64_t mem_132818_cached_sizze_134961 = 0;
    unsigned char *mem_132818 = NULL;
    int64_t mem_132829_cached_sizze_134962 = 0;
    unsigned char *mem_132829 = NULL;
    int64_t mem_132834_cached_sizze_134963 = 0;
    unsigned char *mem_132834 = NULL;
    int64_t mem_132845_cached_sizze_134964 = 0;
    unsigned char *mem_132845 = NULL;
    int64_t mem_132850_cached_sizze_134965 = 0;
    unsigned char *mem_132850 = NULL;
    int64_t mem_132857_cached_sizze_134966 = 0;
    unsigned char *mem_132857 = NULL;
    int64_t mem_132864_cached_sizze_134967 = 0;
    unsigned char *mem_132864 = NULL;
    int64_t mem_132875_cached_sizze_134968 = 0;
    unsigned char *mem_132875 = NULL;
    int64_t mem_132880_cached_sizze_134969 = 0;
    unsigned char *mem_132880 = NULL;
    int64_t mem_132891_cached_sizze_134970 = 0;
    unsigned char *mem_132891 = NULL;
    int64_t mem_132896_cached_sizze_134971 = 0;
    unsigned char *mem_132896 = NULL;
    int64_t mem_132912_cached_sizze_134972 = 0;
    unsigned char *mem_132912 = NULL;
    int64_t mem_132917_cached_sizze_134973 = 0;
    unsigned char *mem_132917 = NULL;
    int64_t mem_132928_cached_sizze_134974 = 0;
    unsigned char *mem_132928 = NULL;
    int64_t mem_132933_cached_sizze_134975 = 0;
    unsigned char *mem_132933 = NULL;
    int64_t mem_132944_cached_sizze_134976 = 0;
    unsigned char *mem_132944 = NULL;
    int64_t mem_132949_cached_sizze_134977 = 0;
    unsigned char *mem_132949 = NULL;
    int64_t mem_132960_cached_sizze_134978 = 0;
    unsigned char *mem_132960 = NULL;
    int64_t mem_132965_cached_sizze_134979 = 0;
    unsigned char *mem_132965 = NULL;
    int64_t mem_132972_cached_sizze_134980 = 0;
    unsigned char *mem_132972 = NULL;
    int64_t mem_132983_cached_sizze_134981 = 0;
    unsigned char *mem_132983 = NULL;
    int64_t mem_132988_cached_sizze_134982 = 0;
    unsigned char *mem_132988 = NULL;
    int64_t mem_132999_cached_sizze_134983 = 0;
    unsigned char *mem_132999 = NULL;
    int64_t mem_133004_cached_sizze_134984 = 0;
    unsigned char *mem_133004 = NULL;
    int64_t mem_133015_cached_sizze_134985 = 0;
    unsigned char *mem_133015 = NULL;
    int64_t mem_133020_cached_sizze_134986 = 0;
    unsigned char *mem_133020 = NULL;
    int64_t mem_133031_cached_sizze_134987 = 0;
    unsigned char *mem_133031 = NULL;
    int64_t mem_133036_cached_sizze_134988 = 0;
    unsigned char *mem_133036 = NULL;
    int64_t mem_133047_cached_sizze_134989 = 0;
    unsigned char *mem_133047 = NULL;
    int64_t mem_133052_cached_sizze_134990 = 0;
    unsigned char *mem_133052 = NULL;
    int64_t mem_133068_cached_sizze_134991 = 0;
    unsigned char *mem_133068 = NULL;
    struct memblock mem_133063;
    
    mem_133063.references = NULL;
    
    struct memblock mem_out_134505;
    
    mem_out_134505.references = NULL;
    
    struct memblock mem_132596 = ctx->constants->mem_132596;
    struct memblock mem_132597 = ctx->constants->mem_132597;
    struct memblock mem_132598 = ctx->constants->mem_132598;
    struct memblock mem_132599 = ctx->constants->mem_132599;
    struct memblock mem_132600 = ctx->constants->mem_132600;
    struct memblock mem_132601 = ctx->constants->mem_132601;
    struct memblock mem_132602 = ctx->constants->mem_132602;
    struct memblock mem_132603 = ctx->constants->mem_132603;
    struct memblock mem_132604 = ctx->constants->mem_132604;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_132616_cached_sizze_134936 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132616, &mem_132616_cached_sizze_134936, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132621_cached_sizze_134937 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132621, &mem_132621_cached_sizze_134937, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131716 = 0; i_131716 < (int64_t) 16; i_131716++) {
        // futhark/microgpt.fut:447:41-50
        
        int64_t tmp_109993 = ((int64_t *) tokens_mem_132614.mem)[i_131716];
        
        // futhark/microgpt.fut:447:37-51
        
        bool x_109994 = sle64((int64_t) 0, tmp_109993);
        
        // futhark/microgpt.fut:447:37-51
        
        bool y_109995 = slt64(tmp_109993, (int64_t) 27);
        
        // futhark/microgpt.fut:447:37-51
        
        bool bounds_check_109996 = x_109994 && y_109995;
        
        // futhark/microgpt.fut:447:37-51
        
        bool index_certs_109997;
        
        if (!bounds_check_109996) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_109993, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:447:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:447:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131712 = 0; i_131712 < (int64_t) 16; i_131712++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_110004 = ((double *) wte_mem_132610.mem)[tmp_109993 * (int64_t) 16 + i_131712];
            
            ((double *) mem_132621)[i_131712] = lifted_lambda_res_110004;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132616, i_131716 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132621, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132632_cached_sizze_134938 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132632, &mem_132632_cached_sizze_134938, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132637_cached_sizze_134939 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132637, &mem_132637_cached_sizze_134939, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132644_cached_sizze_134940 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132644, &mem_132644_cached_sizze_134940, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131728 = 0; i_131728 < (int64_t) 16; i_131728++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_110030;
        double r_110032 = 0.0;
        
        for (int64_t i_110031 = 0; i_110031 < (int64_t) 16; i_110031++) {
            // futhark/microgpt.fut:71:46-49
            
            double zp_lhs_110033 = ((double *) wpe_mem_132608.mem)[i_131728 * (int64_t) 16 + i_110031];
            
            // futhark/microgpt.fut:71:46-49
            
            double zp_rhs_110034 = ((double *) mem_132616)[i_131728 * (int64_t) 16 + i_110031];
            
            // futhark/microgpt.fut:148:76-116
            
            double zp_res_110035 = zp_lhs_110033 + zp_rhs_110034;
            
            // futhark/microgpt.fut:148:94-163
            
            double zt_res_110036 = zp_res_110035 * zp_res_110035;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_110037 = r_110032 + zt_res_110036;
            double r_tmp_134509 = zp_res_110037;
            
            r_110032 = r_tmp_134509;
        }
        defunc_0_lifted_lambda_res_110030 = r_110032;
        // futhark/microgpt.fut:148:54-182
        
        double zs_res_110038 = defunc_0_lifted_lambda_res_110030 / 16.0;
        
        // futhark/microgpt.fut:149:24-55
        
        double zp_res_110039 = 1.0e-5 + zs_res_110038;
        
        // futhark/microgpt.fut:149:16-55
        
        double sqrt_res_110040 = futrts_sqrt64(zp_res_110039);
        
        // futhark/microgpt.fut:150:85-96
        
        double zs_res_110041 = 1.0 / sqrt_res_110040;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131720 = 0; i_131720 < (int64_t) 16; i_131720++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_110048 = ((double *) wpe_mem_132608.mem)[i_131728 * (int64_t) 16 + i_131720];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_110049 = ((double *) mem_132616)[i_131728 * (int64_t) 16 + i_131720];
            
            // futhark/microgpt.fut:150:38-78
            
            double zp_res_110050 = zp_lhs_110048 + zp_rhs_110049;
            
            // futhark/microgpt.fut:150:56-96
            
            double zt_res_110051 = zs_res_110041 * zp_res_110050;
            
            ((double *) mem_132637)[i_131720] = zt_res_110051;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131724 = 0; i_131724 < (int64_t) 16; i_131724++) {
            // futhark/microgpt.fut:151:4-14
            
            double lifted_lambda_res_110059 = ((double *) mem_132637)[i_131724];
            
            ((double *) mem_132644)[i_131724] = lifted_lambda_res_110059;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132632, i_131728 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132644, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132655_cached_sizze_134941 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132655, &mem_132655_cached_sizze_134941, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132660_cached_sizze_134942 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132660, &mem_132660_cached_sizze_134942, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132667_cached_sizze_134943 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132667, &mem_132667_cached_sizze_134943, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131740 = 0; i_131740 < (int64_t) 16; i_131740++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_110068;
        double r_110070 = 0.0;
        
        for (int64_t i_110069 = 0; i_110069 < (int64_t) 16; i_110069++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_110071 = ((double *) mem_132632)[i_131740 * (int64_t) 16 + i_110069];
            
            // futhark/microgpt.fut:152:78-115
            
            double zt_res_110072 = zt_lhs_110071 * zt_lhs_110071;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_110073 = r_110070 + zt_res_110072;
            double r_tmp_134513 = zp_res_110073;
            
            r_110070 = r_tmp_134513;
        }
        defunc_0_lifted_lambda_res_110068 = r_110070;
        // futhark/microgpt.fut:152:57-133
        
        double zs_res_110074 = defunc_0_lifted_lambda_res_110068 / 16.0;
        
        // futhark/microgpt.fut:153:24-55
        
        double zp_res_110075 = 1.0e-5 + zs_res_110074;
        
        // futhark/microgpt.fut:153:16-55
        
        double sqrt_res_110076 = futrts_sqrt64(zp_res_110075);
        
        // futhark/microgpt.fut:154:59-70
        
        double zs_res_110077 = 1.0 / sqrt_res_110076;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131732 = 0; i_131732 < (int64_t) 16; i_131732++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_110084 = ((double *) mem_132632)[i_131740 * (int64_t) 16 + i_131732];
            
            // futhark/microgpt.fut:154:37-70
            
            double zt_res_110085 = zs_res_110077 * zt_lhs_110084;
            
            ((double *) mem_132660)[i_131732] = zt_res_110085;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131736 = 0; i_131736 < (int64_t) 16; i_131736++) {
            // futhark/microgpt.fut:155:4-14
            
            double lifted_lambda_res_110093 = ((double *) mem_132660)[i_131736];
            
            ((double *) mem_132667)[i_131736] = lifted_lambda_res_110093;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132655, i_131740 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132667, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132678_cached_sizze_134944 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132678, &mem_132678_cached_sizze_134944, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132679_cached_sizze_134945 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132679, &mem_132679_cached_sizze_134945, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132680_cached_sizze_134946 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132680, &mem_132680_cached_sizze_134946, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132693_cached_sizze_134947 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132693, &mem_132693_cached_sizze_134947, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132694_cached_sizze_134948 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132694, &mem_132694_cached_sizze_134948, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132695_cached_sizze_134949 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132695, &mem_132695_cached_sizze_134949, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131758 = 0; i_131758 < (int64_t) 16; i_131758++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131748 = 0; i_131748 < (int64_t) 16; i_131748++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124790;
            double r_124792 = 0.0;
            
            for (int64_t i_124791 = 0; i_124791 < (int64_t) 16; i_124791++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_124793 = ((double *) wqry_mem_132609.mem)[i_131748 * (int64_t) 16 + i_124791];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_124794 = ((double *) mem_132655)[i_131758 * (int64_t) 16 + i_124791];
                
                // futhark/microgpt.fut:156:66-105
                
                double zt_res_124795 = zt_lhs_124793 * zt_rhs_124794;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_124796 = r_124792 + zt_res_124795;
                double r_tmp_134522 = zp_res_124796;
                
                r_124792 = r_tmp_134522;
            }
            defunc_0_lifted_lambda_res_124790 = r_124792;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124803;
            double r_124805 = 0.0;
            
            for (int64_t i_124804 = 0; i_124804 < (int64_t) 16; i_124804++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_124806 = ((double *) wkey_mem_132606.mem)[i_131748 * (int64_t) 16 + i_124804];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_124807 = ((double *) mem_132655)[i_131758 * (int64_t) 16 + i_124804];
                
                // futhark/microgpt.fut:157:66-105
                
                double zt_res_124808 = zt_lhs_124806 * zt_rhs_124807;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_124809 = r_124805 + zt_res_124808;
                double r_tmp_134523 = zp_res_124809;
                
                r_124805 = r_tmp_134523;
            }
            defunc_0_lifted_lambda_res_124803 = r_124805;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124819;
            double r_124821 = 0.0;
            
            for (int64_t i_124820 = 0; i_124820 < (int64_t) 16; i_124820++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_124822 = ((double *) wval_mem_132612.mem)[i_131748 * (int64_t) 16 + i_124820];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_124823 = ((double *) mem_132655)[i_131758 * (int64_t) 16 + i_124820];
                
                // futhark/microgpt.fut:158:66-105
                
                double zt_res_124824 = zt_lhs_124822 * zt_rhs_124823;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_124825 = r_124821 + zt_res_124824;
                double r_tmp_134524 = zp_res_124825;
                
                r_124821 = r_tmp_134524;
            }
            defunc_0_lifted_lambda_res_124819 = r_124821;
            ((double *) mem_132693)[i_131748] = defunc_0_lifted_lambda_res_124819;
            ((double *) mem_132694)[i_131748] = defunc_0_lifted_lambda_res_124803;
            ((double *) mem_132695)[i_131748] = defunc_0_lifted_lambda_res_124790;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132678, i_131758 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132693, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132679, i_131758 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132694, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132680, i_131758 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132695, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132726_cached_sizze_134950 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132726, &mem_132726_cached_sizze_134950, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132727_cached_sizze_134951 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132727, &mem_132727_cached_sizze_134951, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132728_cached_sizze_134952 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132728, &mem_132728_cached_sizze_134952, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132744_cached_sizze_134953 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_132744, &mem_132744_cached_sizze_134953, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132745_cached_sizze_134954 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_132745, &mem_132745_cached_sizze_134954, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132746_cached_sizze_134955 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_132746, &mem_132746_cached_sizze_134955, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132759_cached_sizze_134956 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_132759, &mem_132759_cached_sizze_134956, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132760_cached_sizze_134957 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_132760, &mem_132760_cached_sizze_134957, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132761_cached_sizze_134958 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_132761, &mem_132761_cached_sizze_134958, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131788 = 0; i_131788 < (int64_t) 4; i_131788++) {
        // futhark/microgpt.fut:159:69-72
        
        int64_t zp_lhs_124666 = mul64((int64_t) 4, i_131788);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131778 = 0; i_131778 < (int64_t) 16; i_131778++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131768 = 0; i_131768 < (int64_t) 4; i_131768++) {
                // futhark/microgpt.fut:159:74-81
                
                int64_t tmp_124983 = add64(zp_lhs_124666, i_131768);
                
                // futhark/microgpt.fut:159:51-83
                
                bool x_124984 = sle64((int64_t) 0, tmp_124983);
                
                // futhark/microgpt.fut:159:51-83
                
                bool y_124985 = slt64(tmp_124983, (int64_t) 16);
                
                // futhark/microgpt.fut:159:51-83
                
                bool bounds_check_124986 = x_124984 && y_124985;
                
                // futhark/microgpt.fut:159:51-83
                
                bool index_certs_124987;
                
                if (!bounds_check_124986) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_124983, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:159:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:159:15-84\n   #9  futhark/microgpt.fut:448:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_124988 = ((double *) mem_132680)[i_131778 * (int64_t) 16 + tmp_124983];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_124996 = ((double *) mem_132679)[i_131778 * (int64_t) 16 + tmp_124983];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_125007 = ((double *) mem_132678)[i_131778 * (int64_t) 16 + tmp_124983];
                
                ((double *) mem_132759)[i_131768] = lifted_lambda_res_125007;
                ((double *) mem_132760)[i_131768] = lifted_lambda_res_124996;
                ((double *) mem_132761)[i_131768] = lifted_lambda_res_124988;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132744, i_131778 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132759, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132745, i_131778 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132760, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132746, i_131778 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132761, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_132726, i_131788 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_132744, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_132727, i_131788 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_132745, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_132728, i_131788 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_132746, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132807_cached_sizze_134959 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132807, &mem_132807_cached_sizze_134959, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132813_cached_sizze_134960 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132813, &mem_132813_cached_sizze_134960, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132818_cached_sizze_134961 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132818, &mem_132818_cached_sizze_134961, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132829_cached_sizze_134962 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132829, &mem_132829_cached_sizze_134962, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132834_cached_sizze_134963 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132834, &mem_132834_cached_sizze_134963, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132845_cached_sizze_134964 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132845, &mem_132845_cached_sizze_134964, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132850_cached_sizze_134965 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132850, &mem_132850_cached_sizze_134965, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132857_cached_sizze_134966 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132857, &mem_132857_cached_sizze_134966, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132864_cached_sizze_134967 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132864, &mem_132864_cached_sizze_134967, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132875_cached_sizze_134968 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_132875, &mem_132875_cached_sizze_134968, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132880_cached_sizze_134969 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_132880, &mem_132880_cached_sizze_134969, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132891_cached_sizze_134970 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_132891, &mem_132891_cached_sizze_134970, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132896_cached_sizze_134971 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_132896, &mem_132896_cached_sizze_134971, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131844 = 0; i_131844 < (int64_t) 4; i_131844++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131798 = 0; i_131798 < (int64_t) 16; i_131798++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131794 = 0; i_131794 < (int64_t) 16; i_131794++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_110238;
                double r_110240 = 0.0;
                
                for (int64_t i_110239 = 0; i_110239 < (int64_t) 4; i_110239++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_110241 = ((double *) mem_132728)[i_131844 * (int64_t) 64 + i_131798 * (int64_t) 4 + i_110239];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_110242 = ((double *) mem_132727)[i_131844 * (int64_t) 64 + i_131794 * (int64_t) 4 + i_110239];
                    
                    // futhark/microgpt.fut:162:113-164
                    
                    double zt_res_110243 = zt_lhs_110241 * zt_rhs_110242;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_110244 = r_110240 + zt_res_110243;
                    double r_tmp_134537 = zp_res_110244;
                    
                    r_110240 = r_tmp_134537;
                }
                defunc_0_lifted_lambda_res_110238 = r_110240;
                ((double *) mem_132818)[i_131794] = defunc_0_lifted_lambda_res_110238;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132813, i_131798 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132818, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131806 = 0; i_131806 < (int64_t) 16; i_131806++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131802 = 0; i_131802 < (int64_t) 16; i_131802++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_110259 = ((double *) mem_132813)[i_131806 * (int64_t) 16 + i_131802];
                
                // futhark/microgpt.fut:163:47-78
                
                double zs_res_110260 = zs_lhs_110259 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_110261 = ((double *) mask_mem_132615.mem)[i_131806 * (int64_t) 16 + i_131802];
                
                // futhark/microgpt.fut:163:65-102
                
                double zp_res_110262 = zs_res_110260 + zp_rhs_110261;
                
                ((double *) mem_132834)[i_131802] = zp_res_110262;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132829, i_131806 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132834, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131824 = 0; i_131824 < (int64_t) 16; i_131824++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_125085;
            double redout_131808 = -INFINITY;
            
            for (int64_t i_131809 = 0; i_131809 < (int64_t) 16; i_131809++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_125034 = ((double *) mem_132829)[i_131824 * (int64_t) 16 + i_131809];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_110283 = fmax64(lifted_lambda_res_125034, redout_131808);
                double redout_tmp_134541 = max_res_110283;
                
                redout_131808 = redout_tmp_134541;
            }
            defunc_0_reduce_res_125085 = redout_131808;
            // futhark/microgpt.fut:165:67-76
            
            double neg_res_110284 = -defunc_0_reduce_res_125085;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131812 = 0; i_131812 < (int64_t) 16; i_131812++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_110291 = ((double *) mem_132829)[i_131824 * (int64_t) 16 + i_131812];
                
                // futhark/microgpt.fut:165:44-76
                
                double zp_res_110292 = neg_res_110284 + zp_lhs_110291;
                
                // futhark/microgpt.fut:165:37-76
                
                double exp_res_110293 = futrts_exp64(zp_res_110292);
                
                ((double *) mem_132850)[i_131812] = exp_res_110293;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_110295;
            double r_110297 = 0.0;
            
            for (int64_t i_110296 = 0; i_110296 < (int64_t) 16; i_110296++) {
                // futhark/microgpt.fut:166:36-46
                
                double lifted_lambda_res_110298 = ((double *) mem_132850)[i_110296];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_110299 = r_110297 + lifted_lambda_res_110298;
                double r_tmp_134543 = zp_res_110299;
                
                r_110297 = r_tmp_134543;
            }
            defunc_0_lifted_lambda_res_110295 = r_110297;
            // futhark/microgpt.fut:167:53-64
            
            double zs_res_110300 = 1.0 / defunc_0_lifted_lambda_res_110295;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131816 = 0; i_131816 < (int64_t) 16; i_131816++) {
                // futhark/microgpt.fut:167:37-47
                
                double zt_lhs_110307 = ((double *) mem_132850)[i_131816];
                
                // futhark/microgpt.fut:167:37-64
                
                double zt_res_110308 = zs_res_110300 * zt_lhs_110307;
                
                ((double *) mem_132857)[i_131816] = zt_res_110308;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131820 = 0; i_131820 < (int64_t) 16; i_131820++) {
                // futhark/microgpt.fut:168:4-14
                
                double lifted_lambda_res_110316 = ((double *) mem_132857)[i_131820];
                
                ((double *) mem_132864)[i_131820] = lifted_lambda_res_110316;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132845, i_131824 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132864, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131832 = 0; i_131832 < (int64_t) 16; i_131832++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131828 = 0; i_131828 < (int64_t) 4; i_131828++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_110331;
                double r_110333 = 0.0;
                
                for (int64_t i_110332 = 0; i_110332 < (int64_t) 16; i_110332++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_110334 = ((double *) mem_132845)[i_131832 * (int64_t) 16 + i_110332];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_110335 = ((double *) mem_132726)[i_131844 * (int64_t) 64 + i_110332 * (int64_t) 4 + i_131828];
                    
                    // futhark/microgpt.fut:169:66-111
                    
                    double zt_res_110336 = zt_lhs_110334 * zt_rhs_110335;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_110337 = r_110333 + zt_res_110336;
                    double r_tmp_134548 = zp_res_110337;
                    
                    r_110333 = r_tmp_134548;
                }
                defunc_0_lifted_lambda_res_110331 = r_110333;
                ((double *) mem_132880)[i_131828] = defunc_0_lifted_lambda_res_110331;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132875, i_131832 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132880, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131840 = 0; i_131840 < (int64_t) 16; i_131840++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131836 = 0; i_131836 < (int64_t) 4; i_131836++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_110352 = ((double *) mem_132875)[i_131840 * (int64_t) 4 + i_131836];
                
                ((double *) mem_132896)[i_131836] = lifted_lambda_res_110352;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132891, i_131840 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132896, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_132807, i_131844 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_132891, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132912_cached_sizze_134972 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132912, &mem_132912_cached_sizze_134972, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132917_cached_sizze_134973 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132917, &mem_132917_cached_sizze_134973, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131852 = 0; i_131852 < (int64_t) 16; i_131852++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131848 = 0; i_131848 < (int64_t) 16; i_131848++) {
            // futhark/microgpt.fut:171:54-57
            
            int64_t tmp_110364 = sdiv64(i_131848, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-59
            
            bool x_110365 = sle64((int64_t) 0, tmp_110364);
            
            // futhark/microgpt.fut:171:44-59
            
            bool y_110366 = slt64(tmp_110364, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-59
            
            bool bounds_check_110367 = x_110365 && y_110366;
            
            // futhark/microgpt.fut:171:44-59
            
            bool index_certs_110368;
            
            if (!bounds_check_110367) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_110364, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:171:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:171:15-80\n   #6  futhark/microgpt.fut:448:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:171:74-77
            
            int64_t tmp_110369 = smod64(i_131848, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-79
            
            bool x_110370 = sle64((int64_t) 0, tmp_110369);
            
            // futhark/microgpt.fut:171:44-79
            
            bool y_110371 = slt64(tmp_110369, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-79
            
            bool bounds_check_110372 = x_110370 && y_110371;
            
            // futhark/microgpt.fut:171:44-79
            
            bool index_certs_110373;
            
            if (!bounds_check_110372) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_110369, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:171:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:171:15-80\n   #6  futhark/microgpt.fut:448:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_110374 = ((double *) mem_132807)[tmp_110364 * (int64_t) 64 + i_131852 * (int64_t) 4 + tmp_110369];
            
            ((double *) mem_132917)[i_131848] = lifted_lambda_res_110374;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132912, i_131852 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132917, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132928_cached_sizze_134974 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132928, &mem_132928_cached_sizze_134974, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132933_cached_sizze_134975 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132933, &mem_132933_cached_sizze_134975, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131860 = 0; i_131860 < (int64_t) 16; i_131860++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131856 = 0; i_131856 < (int64_t) 16; i_131856++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_110389;
            double r_110391 = 0.0;
            
            for (int64_t i_110390 = 0; i_110390 < (int64_t) 16; i_110390++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_110392 = ((double *) wout_mem_132607.mem)[i_131856 * (int64_t) 16 + i_110390];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_110393 = ((double *) mem_132912)[i_131860 * (int64_t) 16 + i_110390];
                
                // futhark/microgpt.fut:172:67-106
                
                double zt_res_110394 = zt_lhs_110392 * zt_rhs_110393;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_110395 = r_110391 + zt_res_110394;
                double r_tmp_134555 = zp_res_110395;
                
                r_110391 = r_tmp_134555;
            }
            defunc_0_lifted_lambda_res_110389 = r_110391;
            ((double *) mem_132933)[i_131856] = defunc_0_lifted_lambda_res_110389;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132928, i_131860 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132933, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132944_cached_sizze_134976 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132944, &mem_132944_cached_sizze_134976, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132949_cached_sizze_134977 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132949, &mem_132949_cached_sizze_134977, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131868 = 0; i_131868 < (int64_t) 16; i_131868++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131864 = 0; i_131864 < (int64_t) 16; i_131864++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_110410 = ((double *) mem_132928)[i_131868 * (int64_t) 16 + i_131864];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_110411 = ((double *) mem_132632)[i_131868 * (int64_t) 16 + i_131864];
            
            // futhark/microgpt.fut:173:46-84
            
            double zp_res_110412 = zp_lhs_110410 + zp_rhs_110411;
            
            ((double *) mem_132949)[i_131864] = zp_res_110412;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132944, i_131868 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132949, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132960_cached_sizze_134978 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132960, &mem_132960_cached_sizze_134978, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132965_cached_sizze_134979 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132965, &mem_132965_cached_sizze_134979, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132972_cached_sizze_134980 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132972, &mem_132972_cached_sizze_134980, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131880 = 0; i_131880 < (int64_t) 16; i_131880++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_110421;
        double r_110423 = 0.0;
        
        for (int64_t i_110422 = 0; i_110422 < (int64_t) 16; i_110422++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_110424 = ((double *) mem_132944)[i_131880 * (int64_t) 16 + i_110422];
            
            // futhark/microgpt.fut:174:79-118
            
            double zt_res_110425 = zt_lhs_110424 * zt_lhs_110424;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_110426 = r_110423 + zt_res_110425;
            double r_tmp_134559 = zp_res_110426;
            
            r_110423 = r_tmp_134559;
        }
        defunc_0_lifted_lambda_res_110421 = r_110423;
        // futhark/microgpt.fut:174:58-136
        
        double zs_res_110427 = defunc_0_lifted_lambda_res_110421 / 16.0;
        
        // futhark/microgpt.fut:175:24-55
        
        double zp_res_110428 = 1.0e-5 + zs_res_110427;
        
        // futhark/microgpt.fut:175:16-55
        
        double sqrt_res_110429 = futrts_sqrt64(zp_res_110428);
        
        // futhark/microgpt.fut:176:60-71
        
        double zs_res_110430 = 1.0 / sqrt_res_110429;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131872 = 0; i_131872 < (int64_t) 16; i_131872++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_110437 = ((double *) mem_132944)[i_131880 * (int64_t) 16 + i_131872];
            
            // futhark/microgpt.fut:176:37-71
            
            double zt_res_110438 = zs_res_110430 * zt_lhs_110437;
            
            ((double *) mem_132965)[i_131872] = zt_res_110438;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131876 = 0; i_131876 < (int64_t) 16; i_131876++) {
            // futhark/microgpt.fut:177:4-14
            
            double lifted_lambda_res_110446 = ((double *) mem_132965)[i_131876];
            
            ((double *) mem_132972)[i_131876] = lifted_lambda_res_110446;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132960, i_131880 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132972, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132983_cached_sizze_134981 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_132983, &mem_132983_cached_sizze_134981, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132988_cached_sizze_134982 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_132988, &mem_132988_cached_sizze_134982, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131888 = 0; i_131888 < (int64_t) 16; i_131888++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131884 = 0; i_131884 < (int64_t) 64; i_131884++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_110462;
            double r_110464 = 0.0;
            
            for (int64_t i_110463 = 0; i_110463 < (int64_t) 16; i_110463++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_110465 = ((double *) wup_mem_132611.mem)[i_131884 * (int64_t) 16 + i_110463];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_110466 = ((double *) mem_132960)[i_131888 * (int64_t) 16 + i_110463];
                
                // futhark/microgpt.fut:178:67-106
                
                double zt_res_110467 = zt_lhs_110465 * zt_rhs_110466;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_110468 = r_110464 + zt_res_110467;
                double r_tmp_134564 = zp_res_110468;
                
                r_110464 = r_tmp_134564;
            }
            defunc_0_lifted_lambda_res_110462 = r_110464;
            ((double *) mem_132988)[i_131884] = defunc_0_lifted_lambda_res_110462;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132983, i_131888 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132988, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132999_cached_sizze_134983 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_132999, &mem_132999_cached_sizze_134983, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133004_cached_sizze_134984 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133004, &mem_133004_cached_sizze_134984, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131896 = 0; i_131896 < (int64_t) 16; i_131896++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131892 = 0; i_131892 < (int64_t) 64; i_131892++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_110483 = ((double *) mem_132983)[i_131896 * (int64_t) 64 + i_131892];
            
            // futhark/microgpt.fut:179:45-73
            
            double max_res_110484 = fmax64(0.0, max_arg0_110483);
            
            ((double *) mem_133004)[i_131892] = max_res_110484;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_132999, i_131896 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133004, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133015_cached_sizze_134985 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133015, &mem_133015_cached_sizze_134985, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133020_cached_sizze_134986 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133020, &mem_133020_cached_sizze_134986, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131904 = 0; i_131904 < (int64_t) 16; i_131904++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131900 = 0; i_131900 < (int64_t) 16; i_131900++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_110499;
            double r_110501 = 0.0;
            
            for (int64_t i_110500 = 0; i_110500 < (int64_t) 64; i_110500++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_110502 = ((double *) wdown_mem_132605.mem)[i_131900 * (int64_t) 64 + i_110500];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_110503 = ((double *) mem_132999)[i_131904 * (int64_t) 64 + i_110500];
                
                // futhark/microgpt.fut:180:67-108
                
                double zt_res_110504 = zt_lhs_110502 * zt_rhs_110503;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_110505 = r_110501 + zt_res_110504;
                double r_tmp_134569 = zp_res_110505;
                
                r_110501 = r_tmp_134569;
            }
            defunc_0_lifted_lambda_res_110499 = r_110501;
            ((double *) mem_133020)[i_131900] = defunc_0_lifted_lambda_res_110499;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_133015, i_131904 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133020, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133031_cached_sizze_134987 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133031, &mem_133031_cached_sizze_134987, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133036_cached_sizze_134988 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133036, &mem_133036_cached_sizze_134988, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131912 = 0; i_131912 < (int64_t) 16; i_131912++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131908 = 0; i_131908 < (int64_t) 16; i_131908++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_110520 = ((double *) mem_133015)[i_131912 * (int64_t) 16 + i_131908];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_110521 = ((double *) mem_132944)[i_131912 * (int64_t) 16 + i_131908];
            
            // futhark/microgpt.fut:181:46-85
            
            double zp_res_110522 = zp_lhs_110520 + zp_rhs_110521;
            
            ((double *) mem_133036)[i_131908] = zp_res_110522;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_133031, i_131912 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133036, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133047_cached_sizze_134989 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_133047, &mem_133047_cached_sizze_134989, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133052_cached_sizze_134990 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_133052, &mem_133052_cached_sizze_134990, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131920 = 0; i_131920 < (int64_t) 16; i_131920++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131916 = 0; i_131916 < (int64_t) 27; i_131916++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_110538;
            double r_110540 = 0.0;
            
            for (int64_t i_110539 = 0; i_110539 < (int64_t) 16; i_110539++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_110541 = ((double *) wvoc_mem_132613.mem)[i_131916 * (int64_t) 16 + i_110539];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_110542 = ((double *) mem_133031)[i_131920 * (int64_t) 16 + i_110539];
                
                // futhark/microgpt.fut:182:67-107
                
                double zt_res_110543 = zt_lhs_110541 * zt_rhs_110542;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_110544 = r_110540 + zt_res_110543;
                double r_tmp_134574 = zp_res_110544;
                
                r_110540 = r_tmp_134574;
            }
            defunc_0_lifted_lambda_res_110538 = r_110540;
            ((double *) mem_133052)[i_131916] = defunc_0_lifted_lambda_res_110538;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_133047, i_131920 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133052, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_133063, (int64_t) 3456, "mem_133063")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133068_cached_sizze_134991 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_133068, &mem_133068_cached_sizze_134991, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_131928 = 0; i_131928 < (int64_t) 16; i_131928++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131924 = 0; i_131924 < (int64_t) 27; i_131924++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_110559 = ((double *) mem_133047)[i_131928 * (int64_t) 27 + i_131924];
            
            ((double *) mem_133068)[i_131924] = lifted_lambda_res_110559;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_133063.mem, i_131928 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133068, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_134505, &mem_133063, "mem_133063") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_134935, &mem_out_134505, "mem_out_134505") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_132616);
        free(mem_132621);
        free(mem_132632);
        free(mem_132637);
        free(mem_132644);
        free(mem_132655);
        free(mem_132660);
        free(mem_132667);
        free(mem_132678);
        free(mem_132679);
        free(mem_132680);
        free(mem_132693);
        free(mem_132694);
        free(mem_132695);
        free(mem_132726);
        free(mem_132727);
        free(mem_132728);
        free(mem_132744);
        free(mem_132745);
        free(mem_132746);
        free(mem_132759);
        free(mem_132760);
        free(mem_132761);
        free(mem_132807);
        free(mem_132813);
        free(mem_132818);
        free(mem_132829);
        free(mem_132834);
        free(mem_132845);
        free(mem_132850);
        free(mem_132857);
        free(mem_132864);
        free(mem_132875);
        free(mem_132880);
        free(mem_132891);
        free(mem_132896);
        free(mem_132912);
        free(mem_132917);
        free(mem_132928);
        free(mem_132933);
        free(mem_132944);
        free(mem_132949);
        free(mem_132960);
        free(mem_132965);
        free(mem_132972);
        free(mem_132983);
        free(mem_132988);
        free(mem_132999);
        free(mem_133004);
        free(mem_133015);
        free(mem_133020);
        free(mem_133031);
        free(mem_133036);
        free(mem_133047);
        free(mem_133052);
        free(mem_133068);
        if (memblock_unref(ctx, &mem_133063, "mem_133063") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134505, "mem_out_134505") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_134992, struct memblock *mem_out_p_134993, struct memblock *mem_out_p_134994, struct memblock *mem_out_p_134995, struct memblock *mem_out_p_134996, struct memblock *mem_out_p_134997, struct memblock *mem_out_p_134998, struct memblock *mem_out_p_134999, struct memblock *mem_out_p_135000, struct memblock wte_mem_132605, struct memblock wpe_mem_132606, struct memblock wqry_mem_132607, struct memblock wkey_mem_132608, struct memblock wval_mem_132609, struct memblock wout_mem_132610, struct memblock wup_mem_132611, struct memblock wdown_mem_132612, struct memblock wvoc_mem_132613)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_134513;
    
    mem_out_134513.references = NULL;
    
    struct memblock mem_out_134512;
    
    mem_out_134512.references = NULL;
    
    struct memblock mem_out_134511;
    
    mem_out_134511.references = NULL;
    
    struct memblock mem_out_134510;
    
    mem_out_134510.references = NULL;
    
    struct memblock mem_out_134509;
    
    mem_out_134509.references = NULL;
    
    struct memblock mem_out_134508;
    
    mem_out_134508.references = NULL;
    
    struct memblock mem_out_134507;
    
    mem_out_134507.references = NULL;
    
    struct memblock mem_out_134506;
    
    mem_out_134506.references = NULL;
    
    struct memblock mem_out_134505;
    
    mem_out_134505.references = NULL;
    
    struct memblock mem_132596 = ctx->constants->mem_132596;
    struct memblock mem_132597 = ctx->constants->mem_132597;
    struct memblock mem_132598 = ctx->constants->mem_132598;
    struct memblock mem_132599 = ctx->constants->mem_132599;
    struct memblock mem_132600 = ctx->constants->mem_132600;
    struct memblock mem_132601 = ctx->constants->mem_132601;
    struct memblock mem_132602 = ctx->constants->mem_132602;
    struct memblock mem_132603 = ctx->constants->mem_132603;
    struct memblock mem_132604 = ctx->constants->mem_132604;
    
    if (memblock_set(ctx, &mem_out_134505, &wdown_mem_132612, "wdown_mem_132612") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134506, &wkey_mem_132608, "wkey_mem_132608") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134507, &wout_mem_132610, "wout_mem_132610") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134508, &wpe_mem_132606, "wpe_mem_132606") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134509, &wqry_mem_132607, "wqry_mem_132607") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134510, &wte_mem_132605, "wte_mem_132605") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134511, &wup_mem_132611, "wup_mem_132611") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134512, &wval_mem_132609, "wval_mem_132609") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134513, &wvoc_mem_132613, "wvoc_mem_132613") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_134992, &mem_out_134505, "mem_out_134505") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_134993, &mem_out_134506, "mem_out_134506") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_134994, &mem_out_134507, "mem_out_134507") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_134995, &mem_out_134508, "mem_out_134508") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_134996, &mem_out_134509, "mem_out_134509") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_134997, &mem_out_134510, "mem_out_134510") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_134998, &mem_out_134511, "mem_out_134511") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_134999, &mem_out_134512, "mem_out_134512") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135000, &mem_out_134513, "mem_out_134513") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_134513, "mem_out_134513") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134512, "mem_out_134512") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134511, "mem_out_134511") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134510, "mem_out_134510") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134509, "mem_out_134509") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134508, "mem_out_134508") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134507, "mem_out_134507") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134506, "mem_out_134506") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134505, "mem_out_134505") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_135001, struct memblock *mem_out_p_135002, struct memblock *mem_out_p_135003, struct memblock *mem_out_p_135004, struct memblock *mem_out_p_135005, struct memblock *mem_out_p_135006, struct memblock *mem_out_p_135007, struct memblock *mem_out_p_135008, struct memblock *mem_out_p_135009, struct memblock *mem_out_p_135010, struct memblock *mem_out_p_135011, struct memblock *mem_out_p_135012, struct memblock *mem_out_p_135013, struct memblock *mem_out_p_135014, struct memblock *mem_out_p_135015, struct memblock *mem_out_p_135016, struct memblock *mem_out_p_135017, struct memblock *mem_out_p_135018, struct memblock *mem_out_p_135019, struct memblock *mem_out_p_135020, struct memblock *mem_out_p_135021, struct memblock *mem_out_p_135022, struct memblock *mem_out_p_135023, struct memblock *mem_out_p_135024, struct memblock *mem_out_p_135025, struct memblock *mem_out_p_135026, struct memblock *mem_out_p_135027, struct memblock wdown_mem_132605, struct memblock wkey_mem_132606, struct memblock wout_mem_132607, struct memblock wpe_mem_132608, struct memblock wqry_mem_132609, struct memblock wte_mem_132610, struct memblock wup_mem_132611, struct memblock wval_mem_132612, struct memblock wvoc_mem_132613, struct memblock wdown_mem_132614, struct memblock wkey_mem_132615, struct memblock wout_mem_132616, struct memblock wpe_mem_132617, struct memblock wqry_mem_132618, struct memblock wte_mem_132619, struct memblock wup_mem_132620, struct memblock wval_mem_132621, struct memblock wvoc_mem_132622, struct memblock wdown_mem_132623, struct memblock wkey_mem_132624, struct memblock wout_mem_132625, struct memblock wpe_mem_132626, struct memblock wqry_mem_132627, struct memblock wte_mem_132628, struct memblock wup_mem_132629, struct memblock wval_mem_132630, struct memblock wvoc_mem_132631, struct memblock masks_mem_132632, struct memblock dls_mem_132633, struct memblock seqs_mem_132634)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_132743_cached_sizze_135028 = 0;
    unsigned char *mem_132743 = NULL;
    int64_t mem_132744_cached_sizze_135029 = 0;
    unsigned char *mem_132744 = NULL;
    int64_t mem_132753_cached_sizze_135030 = 0;
    unsigned char *mem_132753 = NULL;
    int64_t mem_132760_cached_sizze_135031 = 0;
    unsigned char *mem_132760 = NULL;
    int64_t mem_132775_cached_sizze_135032 = 0;
    unsigned char *mem_132775 = NULL;
    int64_t mem_132776_cached_sizze_135033 = 0;
    unsigned char *mem_132776 = NULL;
    int64_t mem_132785_cached_sizze_135034 = 0;
    unsigned char *mem_132785 = NULL;
    int64_t mem_132792_cached_sizze_135035 = 0;
    unsigned char *mem_132792 = NULL;
    int64_t mem_132807_cached_sizze_135036 = 0;
    unsigned char *mem_132807 = NULL;
    int64_t mem_132808_cached_sizze_135037 = 0;
    unsigned char *mem_132808 = NULL;
    int64_t mem_132817_cached_sizze_135038 = 0;
    unsigned char *mem_132817 = NULL;
    int64_t mem_132818_cached_sizze_135039 = 0;
    unsigned char *mem_132818 = NULL;
    int64_t mem_132831_cached_sizze_135040 = 0;
    unsigned char *mem_132831 = NULL;
    int64_t mem_132846_cached_sizze_135041 = 0;
    unsigned char *mem_132846 = NULL;
    int64_t mem_132847_cached_sizze_135042 = 0;
    unsigned char *mem_132847 = NULL;
    int64_t mem_132848_cached_sizze_135043 = 0;
    unsigned char *mem_132848 = NULL;
    int64_t mem_132860_cached_sizze_135044 = 0;
    unsigned char *mem_132860 = NULL;
    int64_t mem_132861_cached_sizze_135045 = 0;
    unsigned char *mem_132861 = NULL;
    int64_t mem_132874_cached_sizze_135046 = 0;
    unsigned char *mem_132874 = NULL;
    int64_t mem_132892_cached_sizze_135047 = 0;
    unsigned char *mem_132892 = NULL;
    int64_t mem_132893_cached_sizze_135048 = 0;
    unsigned char *mem_132893 = NULL;
    int64_t mem_132894_cached_sizze_135049 = 0;
    unsigned char *mem_132894 = NULL;
    int64_t mem_132895_cached_sizze_135050 = 0;
    unsigned char *mem_132895 = NULL;
    int64_t mem_132911_cached_sizze_135051 = 0;
    unsigned char *mem_132911 = NULL;
    int64_t mem_132912_cached_sizze_135052 = 0;
    unsigned char *mem_132912 = NULL;
    int64_t mem_132913_cached_sizze_135053 = 0;
    unsigned char *mem_132913 = NULL;
    int64_t mem_132947_cached_sizze_135054 = 0;
    unsigned char *mem_132947 = NULL;
    int64_t mem_132948_cached_sizze_135055 = 0;
    unsigned char *mem_132948 = NULL;
    int64_t mem_132949_cached_sizze_135056 = 0;
    unsigned char *mem_132949 = NULL;
    int64_t mem_132965_cached_sizze_135057 = 0;
    unsigned char *mem_132965 = NULL;
    int64_t mem_132966_cached_sizze_135058 = 0;
    unsigned char *mem_132966 = NULL;
    int64_t mem_132967_cached_sizze_135059 = 0;
    unsigned char *mem_132967 = NULL;
    int64_t mem_132980_cached_sizze_135060 = 0;
    unsigned char *mem_132980 = NULL;
    int64_t mem_132981_cached_sizze_135061 = 0;
    unsigned char *mem_132981 = NULL;
    int64_t mem_132982_cached_sizze_135062 = 0;
    unsigned char *mem_132982 = NULL;
    int64_t mem_133028_cached_sizze_135063 = 0;
    unsigned char *mem_133028 = NULL;
    int64_t mem_133029_cached_sizze_135064 = 0;
    unsigned char *mem_133029 = NULL;
    int64_t mem_133030_cached_sizze_135065 = 0;
    unsigned char *mem_133030 = NULL;
    int64_t mem_133031_cached_sizze_135066 = 0;
    unsigned char *mem_133031 = NULL;
    int64_t mem_133032_cached_sizze_135067 = 0;
    unsigned char *mem_133032 = NULL;
    int64_t mem_133055_cached_sizze_135068 = 0;
    unsigned char *mem_133055 = NULL;
    int64_t mem_133056_cached_sizze_135069 = 0;
    unsigned char *mem_133056 = NULL;
    int64_t mem_133057_cached_sizze_135070 = 0;
    unsigned char *mem_133057 = NULL;
    int64_t mem_133058_cached_sizze_135071 = 0;
    unsigned char *mem_133058 = NULL;
    int64_t mem_133059_cached_sizze_135072 = 0;
    unsigned char *mem_133059 = NULL;
    int64_t mem_133077_cached_sizze_135073 = 0;
    unsigned char *mem_133077 = NULL;
    int64_t mem_133078_cached_sizze_135074 = 0;
    unsigned char *mem_133078 = NULL;
    int64_t mem_133091_cached_sizze_135075 = 0;
    unsigned char *mem_133091 = NULL;
    int64_t mem_133092_cached_sizze_135076 = 0;
    unsigned char *mem_133092 = NULL;
    int64_t mem_133122_cached_sizze_135077 = 0;
    unsigned char *mem_133122 = NULL;
    int64_t mem_133127_cached_sizze_135078 = 0;
    unsigned char *mem_133127 = NULL;
    int64_t mem_133160_cached_sizze_135079 = 0;
    unsigned char *mem_133160 = NULL;
    int64_t mem_133165_cached_sizze_135080 = 0;
    unsigned char *mem_133165 = NULL;
    int64_t mem_133176_cached_sizze_135081 = 0;
    unsigned char *mem_133176 = NULL;
    int64_t mem_133181_cached_sizze_135082 = 0;
    unsigned char *mem_133181 = NULL;
    int64_t mem_133192_cached_sizze_135083 = 0;
    unsigned char *mem_133192 = NULL;
    int64_t mem_133197_cached_sizze_135084 = 0;
    unsigned char *mem_133197 = NULL;
    int64_t mem_133208_cached_sizze_135085 = 0;
    unsigned char *mem_133208 = NULL;
    int64_t mem_133209_cached_sizze_135086 = 0;
    unsigned char *mem_133209 = NULL;
    int64_t mem_133218_cached_sizze_135087 = 0;
    unsigned char *mem_133218 = NULL;
    int64_t mem_133219_cached_sizze_135088 = 0;
    unsigned char *mem_133219 = NULL;
    int64_t mem_133232_cached_sizze_135089 = 0;
    unsigned char *mem_133232 = NULL;
    int64_t mem_133247_cached_sizze_135090 = 0;
    unsigned char *mem_133247 = NULL;
    int64_t mem_133248_cached_sizze_135091 = 0;
    unsigned char *mem_133248 = NULL;
    int64_t mem_133256_cached_sizze_135092 = 0;
    unsigned char *mem_133256 = NULL;
    int64_t mem_133270_cached_sizze_135093 = 0;
    unsigned char *mem_133270 = NULL;
    int64_t mem_133275_cached_sizze_135094 = 0;
    unsigned char *mem_133275 = NULL;
    int64_t mem_133286_cached_sizze_135095 = 0;
    unsigned char *mem_133286 = NULL;
    int64_t mem_133291_cached_sizze_135096 = 0;
    unsigned char *mem_133291 = NULL;
    int64_t mem_133302_cached_sizze_135097 = 0;
    unsigned char *mem_133302 = NULL;
    int64_t mem_133307_cached_sizze_135098 = 0;
    unsigned char *mem_133307 = NULL;
    int64_t mem_133318_cached_sizze_135099 = 0;
    unsigned char *mem_133318 = NULL;
    int64_t mem_133323_cached_sizze_135100 = 0;
    unsigned char *mem_133323 = NULL;
    int64_t mem_133334_cached_sizze_135101 = 0;
    unsigned char *mem_133334 = NULL;
    int64_t mem_133335_cached_sizze_135102 = 0;
    unsigned char *mem_133335 = NULL;
    int64_t mem_133348_cached_sizze_135103 = 0;
    unsigned char *mem_133348 = NULL;
    int64_t mem_133353_cached_sizze_135104 = 0;
    unsigned char *mem_133353 = NULL;
    int64_t mem_133364_cached_sizze_135105 = 0;
    unsigned char *mem_133364 = NULL;
    int64_t mem_133365_cached_sizze_135106 = 0;
    unsigned char *mem_133365 = NULL;
    int64_t mem_133372_cached_sizze_135107 = 0;
    unsigned char *mem_133372 = NULL;
    int64_t mem_133385_cached_sizze_135108 = 0;
    unsigned char *mem_133385 = NULL;
    int64_t mem_133390_cached_sizze_135109 = 0;
    unsigned char *mem_133390 = NULL;
    int64_t mem_133397_cached_sizze_135110 = 0;
    unsigned char *mem_133397 = NULL;
    int64_t mem_133408_cached_sizze_135111 = 0;
    unsigned char *mem_133408 = NULL;
    int64_t mem_133415_cached_sizze_135112 = 0;
    unsigned char *mem_133415 = NULL;
    int64_t mem_133420_cached_sizze_135113 = 0;
    unsigned char *mem_133420 = NULL;
    int64_t mem_133431_cached_sizze_135114 = 0;
    unsigned char *mem_133431 = NULL;
    int64_t mem_133436_cached_sizze_135115 = 0;
    unsigned char *mem_133436 = NULL;
    int64_t mem_133447_cached_sizze_135116 = 0;
    unsigned char *mem_133447 = NULL;
    int64_t mem_133448_cached_sizze_135117 = 0;
    unsigned char *mem_133448 = NULL;
    int64_t mem_133457_cached_sizze_135118 = 0;
    unsigned char *mem_133457 = NULL;
    int64_t mem_133458_cached_sizze_135119 = 0;
    unsigned char *mem_133458 = NULL;
    int64_t mem_133479_cached_sizze_135120 = 0;
    unsigned char *mem_133479 = NULL;
    int64_t mem_133484_cached_sizze_135121 = 0;
    unsigned char *mem_133484 = NULL;
    int64_t mem_133495_cached_sizze_135122 = 0;
    unsigned char *mem_133495 = NULL;
    int64_t mem_133500_cached_sizze_135123 = 0;
    unsigned char *mem_133500 = NULL;
    int64_t mem_133511_cached_sizze_135124 = 0;
    unsigned char *mem_133511 = NULL;
    int64_t mem_133512_cached_sizze_135125 = 0;
    unsigned char *mem_133512 = NULL;
    int64_t mem_133525_cached_sizze_135126 = 0;
    unsigned char *mem_133525 = NULL;
    int64_t mem_133532_cached_sizze_135127 = 0;
    unsigned char *mem_133532 = NULL;
    int64_t mem_133542_cached_sizze_135128 = 0;
    unsigned char *mem_133542 = NULL;
    int64_t mem_133547_cached_sizze_135129 = 0;
    unsigned char *mem_133547 = NULL;
    int64_t mem_133558_cached_sizze_135130 = 0;
    unsigned char *mem_133558 = NULL;
    int64_t mem_133559_cached_sizze_135131 = 0;
    unsigned char *mem_133559 = NULL;
    int64_t mem_133568_cached_sizze_135132 = 0;
    unsigned char *mem_133568 = NULL;
    int64_t mem_133569_cached_sizze_135133 = 0;
    unsigned char *mem_133569 = NULL;
    int64_t mem_133590_cached_sizze_135134 = 0;
    unsigned char *mem_133590 = NULL;
    int64_t mem_133591_cached_sizze_135135 = 0;
    unsigned char *mem_133591 = NULL;
    int64_t mem_133592_cached_sizze_135136 = 0;
    unsigned char *mem_133592 = NULL;
    int64_t mem_133608_cached_sizze_135137 = 0;
    unsigned char *mem_133608 = NULL;
    int64_t mem_133609_cached_sizze_135138 = 0;
    unsigned char *mem_133609 = NULL;
    int64_t mem_133610_cached_sizze_135139 = 0;
    unsigned char *mem_133610 = NULL;
    int64_t mem_133623_cached_sizze_135140 = 0;
    unsigned char *mem_133623 = NULL;
    int64_t mem_133630_cached_sizze_135141 = 0;
    unsigned char *mem_133630 = NULL;
    int64_t mem_133631_cached_sizze_135142 = 0;
    unsigned char *mem_133631 = NULL;
    int64_t mem_133671_cached_sizze_135143 = 0;
    unsigned char *mem_133671 = NULL;
    int64_t mem_133672_cached_sizze_135144 = 0;
    unsigned char *mem_133672 = NULL;
    int64_t mem_133673_cached_sizze_135145 = 0;
    unsigned char *mem_133673 = NULL;
    int64_t mem_133689_cached_sizze_135146 = 0;
    unsigned char *mem_133689 = NULL;
    int64_t mem_133690_cached_sizze_135147 = 0;
    unsigned char *mem_133690 = NULL;
    int64_t mem_133691_cached_sizze_135148 = 0;
    unsigned char *mem_133691 = NULL;
    int64_t mem_133704_cached_sizze_135149 = 0;
    unsigned char *mem_133704 = NULL;
    int64_t mem_133711_cached_sizze_135150 = 0;
    unsigned char *mem_133711 = NULL;
    int64_t mem_133712_cached_sizze_135151 = 0;
    unsigned char *mem_133712 = NULL;
    int64_t mem_133752_cached_sizze_135152 = 0;
    unsigned char *mem_133752 = NULL;
    int64_t mem_133753_cached_sizze_135153 = 0;
    unsigned char *mem_133753 = NULL;
    int64_t mem_133754_cached_sizze_135154 = 0;
    unsigned char *mem_133754 = NULL;
    int64_t mem_133755_cached_sizze_135155 = 0;
    unsigned char *mem_133755 = NULL;
    int64_t mem_133772_cached_sizze_135156 = 0;
    unsigned char *mem_133772 = NULL;
    int64_t mem_133773_cached_sizze_135157 = 0;
    unsigned char *mem_133773 = NULL;
    int64_t mem_133774_cached_sizze_135158 = 0;
    unsigned char *mem_133774 = NULL;
    int64_t mem_133775_cached_sizze_135159 = 0;
    unsigned char *mem_133775 = NULL;
    int64_t mem_133816_cached_sizze_135160 = 0;
    unsigned char *mem_133816 = NULL;
    int64_t mem_133817_cached_sizze_135161 = 0;
    unsigned char *mem_133817 = NULL;
    int64_t mem_133828_cached_sizze_135162 = 0;
    unsigned char *mem_133828 = NULL;
    int64_t mem_133829_cached_sizze_135163 = 0;
    unsigned char *mem_133829 = NULL;
    int64_t mem_133838_cached_sizze_135164 = 0;
    unsigned char *mem_133838 = NULL;
    int64_t mem_133839_cached_sizze_135165 = 0;
    unsigned char *mem_133839 = NULL;
    int64_t mem_133870_cached_sizze_135166 = 0;
    unsigned char *mem_133870 = NULL;
    int64_t mem_133871_cached_sizze_135167 = 0;
    unsigned char *mem_133871 = NULL;
    int64_t mem_133881_cached_sizze_135168 = 0;
    unsigned char *mem_133881 = NULL;
    int64_t mem_133882_cached_sizze_135169 = 0;
    unsigned char *mem_133882 = NULL;
    int64_t mem_133890_cached_sizze_135170 = 0;
    unsigned char *mem_133890 = NULL;
    int64_t mem_133913_cached_sizze_135171 = 0;
    unsigned char *mem_133913 = NULL;
    int64_t mem_133919_cached_sizze_135172 = 0;
    unsigned char *mem_133919 = NULL;
    int64_t mem_133924_cached_sizze_135173 = 0;
    unsigned char *mem_133924 = NULL;
    int64_t mem_133940_cached_sizze_135174 = 0;
    unsigned char *mem_133940 = NULL;
    int64_t mem_133941_cached_sizze_135175 = 0;
    unsigned char *mem_133941 = NULL;
    int64_t mem_133942_cached_sizze_135176 = 0;
    unsigned char *mem_133942 = NULL;
    int64_t mem_133955_cached_sizze_135177 = 0;
    unsigned char *mem_133955 = NULL;
    int64_t mem_133956_cached_sizze_135178 = 0;
    unsigned char *mem_133956 = NULL;
    int64_t mem_133957_cached_sizze_135179 = 0;
    unsigned char *mem_133957 = NULL;
    int64_t mem_133988_cached_sizze_135180 = 0;
    unsigned char *mem_133988 = NULL;
    int64_t mem_133989_cached_sizze_135181 = 0;
    unsigned char *mem_133989 = NULL;
    int64_t mem_133990_cached_sizze_135182 = 0;
    unsigned char *mem_133990 = NULL;
    int64_t mem_133991_cached_sizze_135183 = 0;
    unsigned char *mem_133991 = NULL;
    int64_t mem_134008_cached_sizze_135184 = 0;
    unsigned char *mem_134008 = NULL;
    int64_t mem_134009_cached_sizze_135185 = 0;
    unsigned char *mem_134009 = NULL;
    int64_t mem_134010_cached_sizze_135186 = 0;
    unsigned char *mem_134010 = NULL;
    int64_t mem_134011_cached_sizze_135187 = 0;
    unsigned char *mem_134011 = NULL;
    int64_t mem_134052_cached_sizze_135188 = 0;
    unsigned char *mem_134052 = NULL;
    int64_t mem_134053_cached_sizze_135189 = 0;
    unsigned char *mem_134053 = NULL;
    int64_t mem_134066_cached_sizze_135190 = 0;
    unsigned char *mem_134066 = NULL;
    int64_t mem_134073_cached_sizze_135191 = 0;
    unsigned char *mem_134073 = NULL;
    int64_t mem_134083_cached_sizze_135192 = 0;
    unsigned char *mem_134083 = NULL;
    int64_t mem_134088_cached_sizze_135193 = 0;
    unsigned char *mem_134088 = NULL;
    int64_t mem_134099_cached_sizze_135194 = 0;
    unsigned char *mem_134099 = NULL;
    int64_t mem_134100_cached_sizze_135195 = 0;
    unsigned char *mem_134100 = NULL;
    int64_t mem_134113_cached_sizze_135196 = 0;
    unsigned char *mem_134113 = NULL;
    int64_t mem_134120_cached_sizze_135197 = 0;
    unsigned char *mem_134120 = NULL;
    int64_t mem_134130_cached_sizze_135198 = 0;
    unsigned char *mem_134130 = NULL;
    int64_t mem_134135_cached_sizze_135199 = 0;
    unsigned char *mem_134135 = NULL;
    int64_t mem_134146_cached_sizze_135200 = 0;
    unsigned char *mem_134146 = NULL;
    int64_t mem_134147_cached_sizze_135201 = 0;
    unsigned char *mem_134147 = NULL;
    int64_t mem_134156_cached_sizze_135202 = 0;
    unsigned char *mem_134156 = NULL;
    int64_t mem_134157_cached_sizze_135203 = 0;
    unsigned char *mem_134157 = NULL;
    int64_t mem_134178_cached_sizze_135204 = 0;
    unsigned char *mem_134178 = NULL;
    int64_t mem_134183_cached_sizze_135205 = 0;
    unsigned char *mem_134183 = NULL;
    int64_t mem_134194_cached_sizze_135206 = 0;
    unsigned char *mem_134194 = NULL;
    int64_t mem_134195_cached_sizze_135207 = 0;
    unsigned char *mem_134195 = NULL;
    int64_t mem_134204_cached_sizze_135208 = 0;
    unsigned char *mem_134204 = NULL;
    int64_t mem_134205_cached_sizze_135209 = 0;
    unsigned char *mem_134205 = NULL;
    struct memblock mem_param_tmp_134558;
    
    mem_param_tmp_134558.references = NULL;
    
    struct memblock mem_param_tmp_134557;
    
    mem_param_tmp_134557.references = NULL;
    
    struct memblock mem_param_tmp_134556;
    
    mem_param_tmp_134556.references = NULL;
    
    struct memblock mem_param_tmp_134555;
    
    mem_param_tmp_134555.references = NULL;
    
    struct memblock mem_param_tmp_134554;
    
    mem_param_tmp_134554.references = NULL;
    
    struct memblock mem_param_tmp_134553;
    
    mem_param_tmp_134553.references = NULL;
    
    struct memblock mem_param_tmp_134552;
    
    mem_param_tmp_134552.references = NULL;
    
    struct memblock mem_param_tmp_134551;
    
    mem_param_tmp_134551.references = NULL;
    
    struct memblock mem_param_tmp_134550;
    
    mem_param_tmp_134550.references = NULL;
    
    struct memblock mem_param_tmp_134549;
    
    mem_param_tmp_134549.references = NULL;
    
    struct memblock mem_param_tmp_134548;
    
    mem_param_tmp_134548.references = NULL;
    
    struct memblock mem_param_tmp_134547;
    
    mem_param_tmp_134547.references = NULL;
    
    struct memblock mem_param_tmp_134546;
    
    mem_param_tmp_134546.references = NULL;
    
    struct memblock mem_param_tmp_134545;
    
    mem_param_tmp_134545.references = NULL;
    
    struct memblock mem_param_tmp_134544;
    
    mem_param_tmp_134544.references = NULL;
    
    struct memblock mem_param_tmp_134543;
    
    mem_param_tmp_134543.references = NULL;
    
    struct memblock mem_param_tmp_134542;
    
    mem_param_tmp_134542.references = NULL;
    
    struct memblock mem_param_tmp_134541;
    
    mem_param_tmp_134541.references = NULL;
    
    struct memblock mem_param_tmp_134540;
    
    mem_param_tmp_134540.references = NULL;
    
    struct memblock mem_param_tmp_134539;
    
    mem_param_tmp_134539.references = NULL;
    
    struct memblock mem_param_tmp_134538;
    
    mem_param_tmp_134538.references = NULL;
    
    struct memblock mem_param_tmp_134537;
    
    mem_param_tmp_134537.references = NULL;
    
    struct memblock mem_param_tmp_134536;
    
    mem_param_tmp_134536.references = NULL;
    
    struct memblock mem_param_tmp_134535;
    
    mem_param_tmp_134535.references = NULL;
    
    struct memblock mem_param_tmp_134534;
    
    mem_param_tmp_134534.references = NULL;
    
    struct memblock mem_param_tmp_134533;
    
    mem_param_tmp_134533.references = NULL;
    
    struct memblock mem_param_tmp_134532;
    
    mem_param_tmp_134532.references = NULL;
    
    struct memblock ext_mem_134322;
    
    ext_mem_134322.references = NULL;
    
    struct memblock ext_mem_134323;
    
    ext_mem_134323.references = NULL;
    
    struct memblock ext_mem_134324;
    
    ext_mem_134324.references = NULL;
    
    struct memblock mem_134320;
    
    mem_134320.references = NULL;
    
    struct memblock mem_134318;
    
    mem_134318.references = NULL;
    
    struct memblock mem_134316;
    
    mem_134316.references = NULL;
    
    struct memblock mem_134314;
    
    mem_134314.references = NULL;
    
    struct memblock ext_mem_134311;
    
    ext_mem_134311.references = NULL;
    
    struct memblock ext_mem_134312;
    
    ext_mem_134312.references = NULL;
    
    struct memblock ext_mem_134313;
    
    ext_mem_134313.references = NULL;
    
    struct memblock mem_134309;
    
    mem_134309.references = NULL;
    
    struct memblock mem_134307;
    
    mem_134307.references = NULL;
    
    struct memblock mem_134305;
    
    mem_134305.references = NULL;
    
    struct memblock mem_134303;
    
    mem_134303.references = NULL;
    
    struct memblock ext_mem_134300;
    
    ext_mem_134300.references = NULL;
    
    struct memblock ext_mem_134301;
    
    ext_mem_134301.references = NULL;
    
    struct memblock ext_mem_134302;
    
    ext_mem_134302.references = NULL;
    
    struct memblock mem_134298;
    
    mem_134298.references = NULL;
    
    struct memblock mem_134296;
    
    mem_134296.references = NULL;
    
    struct memblock mem_134294;
    
    mem_134294.references = NULL;
    
    struct memblock mem_134292;
    
    mem_134292.references = NULL;
    
    struct memblock ext_mem_134289;
    
    ext_mem_134289.references = NULL;
    
    struct memblock ext_mem_134290;
    
    ext_mem_134290.references = NULL;
    
    struct memblock ext_mem_134291;
    
    ext_mem_134291.references = NULL;
    
    struct memblock mem_134287;
    
    mem_134287.references = NULL;
    
    struct memblock mem_134285;
    
    mem_134285.references = NULL;
    
    struct memblock mem_134283;
    
    mem_134283.references = NULL;
    
    struct memblock mem_134281;
    
    mem_134281.references = NULL;
    
    struct memblock ext_mem_134278;
    
    ext_mem_134278.references = NULL;
    
    struct memblock ext_mem_134279;
    
    ext_mem_134279.references = NULL;
    
    struct memblock ext_mem_134280;
    
    ext_mem_134280.references = NULL;
    
    struct memblock mem_134276;
    
    mem_134276.references = NULL;
    
    struct memblock mem_134274;
    
    mem_134274.references = NULL;
    
    struct memblock mem_134272;
    
    mem_134272.references = NULL;
    
    struct memblock mem_134270;
    
    mem_134270.references = NULL;
    
    struct memblock ext_mem_134267;
    
    ext_mem_134267.references = NULL;
    
    struct memblock ext_mem_134268;
    
    ext_mem_134268.references = NULL;
    
    struct memblock ext_mem_134269;
    
    ext_mem_134269.references = NULL;
    
    struct memblock mem_134265;
    
    mem_134265.references = NULL;
    
    struct memblock mem_134263;
    
    mem_134263.references = NULL;
    
    struct memblock mem_134261;
    
    mem_134261.references = NULL;
    
    struct memblock mem_134259;
    
    mem_134259.references = NULL;
    
    struct memblock ext_mem_134256;
    
    ext_mem_134256.references = NULL;
    
    struct memblock ext_mem_134257;
    
    ext_mem_134257.references = NULL;
    
    struct memblock ext_mem_134258;
    
    ext_mem_134258.references = NULL;
    
    struct memblock mem_134254;
    
    mem_134254.references = NULL;
    
    struct memblock mem_134252;
    
    mem_134252.references = NULL;
    
    struct memblock mem_134250;
    
    mem_134250.references = NULL;
    
    struct memblock mem_134248;
    
    mem_134248.references = NULL;
    
    struct memblock ext_mem_134245;
    
    ext_mem_134245.references = NULL;
    
    struct memblock ext_mem_134246;
    
    ext_mem_134246.references = NULL;
    
    struct memblock ext_mem_134247;
    
    ext_mem_134247.references = NULL;
    
    struct memblock mem_134243;
    
    mem_134243.references = NULL;
    
    struct memblock mem_134241;
    
    mem_134241.references = NULL;
    
    struct memblock mem_134239;
    
    mem_134239.references = NULL;
    
    struct memblock mem_134237;
    
    mem_134237.references = NULL;
    
    struct memblock ext_mem_134234;
    
    ext_mem_134234.references = NULL;
    
    struct memblock ext_mem_134235;
    
    ext_mem_134235.references = NULL;
    
    struct memblock ext_mem_134236;
    
    ext_mem_134236.references = NULL;
    
    struct memblock mem_134232;
    
    mem_134232.references = NULL;
    
    struct memblock mem_134230;
    
    mem_134230.references = NULL;
    
    struct memblock mem_134228;
    
    mem_134228.references = NULL;
    
    struct memblock mem_134226;
    
    mem_134226.references = NULL;
    
    struct memblock mem_param_132742;
    
    mem_param_132742.references = NULL;
    
    struct memblock mem_param_132738;
    
    mem_param_132738.references = NULL;
    
    struct memblock mem_param_132734;
    
    mem_param_132734.references = NULL;
    
    struct memblock mem_param_132730;
    
    mem_param_132730.references = NULL;
    
    struct memblock mem_param_132726;
    
    mem_param_132726.references = NULL;
    
    struct memblock mem_param_132722;
    
    mem_param_132722.references = NULL;
    
    struct memblock mem_param_132718;
    
    mem_param_132718.references = NULL;
    
    struct memblock mem_param_132714;
    
    mem_param_132714.references = NULL;
    
    struct memblock mem_param_132710;
    
    mem_param_132710.references = NULL;
    
    struct memblock mem_param_132706;
    
    mem_param_132706.references = NULL;
    
    struct memblock mem_param_132702;
    
    mem_param_132702.references = NULL;
    
    struct memblock mem_param_132698;
    
    mem_param_132698.references = NULL;
    
    struct memblock mem_param_132694;
    
    mem_param_132694.references = NULL;
    
    struct memblock mem_param_132690;
    
    mem_param_132690.references = NULL;
    
    struct memblock mem_param_132686;
    
    mem_param_132686.references = NULL;
    
    struct memblock mem_param_132682;
    
    mem_param_132682.references = NULL;
    
    struct memblock mem_param_132678;
    
    mem_param_132678.references = NULL;
    
    struct memblock mem_param_132674;
    
    mem_param_132674.references = NULL;
    
    struct memblock mem_param_132670;
    
    mem_param_132670.references = NULL;
    
    struct memblock mem_param_132666;
    
    mem_param_132666.references = NULL;
    
    struct memblock mem_param_132662;
    
    mem_param_132662.references = NULL;
    
    struct memblock mem_param_132658;
    
    mem_param_132658.references = NULL;
    
    struct memblock mem_param_132654;
    
    mem_param_132654.references = NULL;
    
    struct memblock mem_param_132650;
    
    mem_param_132650.references = NULL;
    
    struct memblock mem_param_132646;
    
    mem_param_132646.references = NULL;
    
    struct memblock mem_param_132642;
    
    mem_param_132642.references = NULL;
    
    struct memblock mem_param_132638;
    
    mem_param_132638.references = NULL;
    
    struct memblock ext_mem_134406;
    
    ext_mem_134406.references = NULL;
    
    struct memblock ext_mem_134407;
    
    ext_mem_134407.references = NULL;
    
    struct memblock ext_mem_134408;
    
    ext_mem_134408.references = NULL;
    
    struct memblock ext_mem_134409;
    
    ext_mem_134409.references = NULL;
    
    struct memblock ext_mem_134410;
    
    ext_mem_134410.references = NULL;
    
    struct memblock ext_mem_134411;
    
    ext_mem_134411.references = NULL;
    
    struct memblock ext_mem_134412;
    
    ext_mem_134412.references = NULL;
    
    struct memblock ext_mem_134413;
    
    ext_mem_134413.references = NULL;
    
    struct memblock ext_mem_134414;
    
    ext_mem_134414.references = NULL;
    
    struct memblock ext_mem_134415;
    
    ext_mem_134415.references = NULL;
    
    struct memblock ext_mem_134416;
    
    ext_mem_134416.references = NULL;
    
    struct memblock ext_mem_134417;
    
    ext_mem_134417.references = NULL;
    
    struct memblock ext_mem_134418;
    
    ext_mem_134418.references = NULL;
    
    struct memblock ext_mem_134419;
    
    ext_mem_134419.references = NULL;
    
    struct memblock ext_mem_134420;
    
    ext_mem_134420.references = NULL;
    
    struct memblock ext_mem_134421;
    
    ext_mem_134421.references = NULL;
    
    struct memblock ext_mem_134422;
    
    ext_mem_134422.references = NULL;
    
    struct memblock ext_mem_134423;
    
    ext_mem_134423.references = NULL;
    
    struct memblock ext_mem_134424;
    
    ext_mem_134424.references = NULL;
    
    struct memblock ext_mem_134425;
    
    ext_mem_134425.references = NULL;
    
    struct memblock ext_mem_134426;
    
    ext_mem_134426.references = NULL;
    
    struct memblock ext_mem_134427;
    
    ext_mem_134427.references = NULL;
    
    struct memblock ext_mem_134428;
    
    ext_mem_134428.references = NULL;
    
    struct memblock ext_mem_134429;
    
    ext_mem_134429.references = NULL;
    
    struct memblock ext_mem_134430;
    
    ext_mem_134430.references = NULL;
    
    struct memblock ext_mem_134431;
    
    ext_mem_134431.references = NULL;
    
    struct memblock ext_mem_134432;
    
    ext_mem_134432.references = NULL;
    
    struct memblock mem_out_134531;
    
    mem_out_134531.references = NULL;
    
    struct memblock mem_out_134530;
    
    mem_out_134530.references = NULL;
    
    struct memblock mem_out_134529;
    
    mem_out_134529.references = NULL;
    
    struct memblock mem_out_134528;
    
    mem_out_134528.references = NULL;
    
    struct memblock mem_out_134527;
    
    mem_out_134527.references = NULL;
    
    struct memblock mem_out_134526;
    
    mem_out_134526.references = NULL;
    
    struct memblock mem_out_134525;
    
    mem_out_134525.references = NULL;
    
    struct memblock mem_out_134524;
    
    mem_out_134524.references = NULL;
    
    struct memblock mem_out_134523;
    
    mem_out_134523.references = NULL;
    
    struct memblock mem_out_134522;
    
    mem_out_134522.references = NULL;
    
    struct memblock mem_out_134521;
    
    mem_out_134521.references = NULL;
    
    struct memblock mem_out_134520;
    
    mem_out_134520.references = NULL;
    
    struct memblock mem_out_134519;
    
    mem_out_134519.references = NULL;
    
    struct memblock mem_out_134518;
    
    mem_out_134518.references = NULL;
    
    struct memblock mem_out_134517;
    
    mem_out_134517.references = NULL;
    
    struct memblock mem_out_134516;
    
    mem_out_134516.references = NULL;
    
    struct memblock mem_out_134515;
    
    mem_out_134515.references = NULL;
    
    struct memblock mem_out_134514;
    
    mem_out_134514.references = NULL;
    
    struct memblock mem_out_134513;
    
    mem_out_134513.references = NULL;
    
    struct memblock mem_out_134512;
    
    mem_out_134512.references = NULL;
    
    struct memblock mem_out_134511;
    
    mem_out_134511.references = NULL;
    
    struct memblock mem_out_134510;
    
    mem_out_134510.references = NULL;
    
    struct memblock mem_out_134509;
    
    mem_out_134509.references = NULL;
    
    struct memblock mem_out_134508;
    
    mem_out_134508.references = NULL;
    
    struct memblock mem_out_134507;
    
    mem_out_134507.references = NULL;
    
    struct memblock mem_out_134506;
    
    mem_out_134506.references = NULL;
    
    struct memblock mem_out_134505;
    
    mem_out_134505.references = NULL;
    
    struct memblock mem_132596 = ctx->constants->mem_132596;
    struct memblock mem_132597 = ctx->constants->mem_132597;
    struct memblock mem_132598 = ctx->constants->mem_132598;
    struct memblock mem_132599 = ctx->constants->mem_132599;
    struct memblock mem_132600 = ctx->constants->mem_132600;
    struct memblock mem_132601 = ctx->constants->mem_132601;
    struct memblock mem_132602 = ctx->constants->mem_132602;
    struct memblock mem_132603 = ctx->constants->mem_132603;
    struct memblock mem_132604 = ctx->constants->mem_132604;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_132743_cached_sizze_135028 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132743, &mem_132743_cached_sizze_135028, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132744_cached_sizze_135029 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_132744, &mem_132744_cached_sizze_135029, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132753_cached_sizze_135030 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_132753, &mem_132753_cached_sizze_135030, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132760_cached_sizze_135031 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132760, &mem_132760_cached_sizze_135031, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132775_cached_sizze_135032 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_132775, &mem_132775_cached_sizze_135032, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132776_cached_sizze_135033 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132776, &mem_132776_cached_sizze_135033, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132785_cached_sizze_135034 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132785, &mem_132785_cached_sizze_135034, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132792_cached_sizze_135035 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_132792, &mem_132792_cached_sizze_135035, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132807_cached_sizze_135036 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132807, &mem_132807_cached_sizze_135036, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132808_cached_sizze_135037 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132808, &mem_132808_cached_sizze_135037, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132817_cached_sizze_135038 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132817, &mem_132817_cached_sizze_135038, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132818_cached_sizze_135039 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132818, &mem_132818_cached_sizze_135039, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132831_cached_sizze_135040 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132831, &mem_132831_cached_sizze_135040, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132846_cached_sizze_135041 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132846, &mem_132846_cached_sizze_135041, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132847_cached_sizze_135042 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132847, &mem_132847_cached_sizze_135042, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132848_cached_sizze_135043 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132848, &mem_132848_cached_sizze_135043, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132860_cached_sizze_135044 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132860, &mem_132860_cached_sizze_135044, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132861_cached_sizze_135045 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132861, &mem_132861_cached_sizze_135045, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132874_cached_sizze_135046 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132874, &mem_132874_cached_sizze_135046, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132892_cached_sizze_135047 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132892, &mem_132892_cached_sizze_135047, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132893_cached_sizze_135048 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132893, &mem_132893_cached_sizze_135048, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132894_cached_sizze_135049 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132894, &mem_132894_cached_sizze_135049, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132895_cached_sizze_135050 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132895, &mem_132895_cached_sizze_135050, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132911_cached_sizze_135051 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132911, &mem_132911_cached_sizze_135051, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132912_cached_sizze_135052 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132912, &mem_132912_cached_sizze_135052, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132913_cached_sizze_135053 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_132913, &mem_132913_cached_sizze_135053, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132947_cached_sizze_135054 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132947, &mem_132947_cached_sizze_135054, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132948_cached_sizze_135055 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132948, &mem_132948_cached_sizze_135055, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132949_cached_sizze_135056 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_132949, &mem_132949_cached_sizze_135056, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132965_cached_sizze_135057 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_132965, &mem_132965_cached_sizze_135057, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132966_cached_sizze_135058 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_132966, &mem_132966_cached_sizze_135058, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132967_cached_sizze_135059 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_132967, &mem_132967_cached_sizze_135059, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132980_cached_sizze_135060 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_132980, &mem_132980_cached_sizze_135060, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132981_cached_sizze_135061 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_132981, &mem_132981_cached_sizze_135061, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_132982_cached_sizze_135062 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_132982, &mem_132982_cached_sizze_135062, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133028_cached_sizze_135063 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133028, &mem_133028_cached_sizze_135063, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133029_cached_sizze_135064 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133029, &mem_133029_cached_sizze_135064, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133030_cached_sizze_135065 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133030, &mem_133030_cached_sizze_135065, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133031_cached_sizze_135066 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_133031, &mem_133031_cached_sizze_135066, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133032_cached_sizze_135067 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133032, &mem_133032_cached_sizze_135067, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133055_cached_sizze_135068 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133055, &mem_133055_cached_sizze_135068, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133056_cached_sizze_135069 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133056, &mem_133056_cached_sizze_135069, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133057_cached_sizze_135070 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133057, &mem_133057_cached_sizze_135070, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133058_cached_sizze_135071 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133058, &mem_133058_cached_sizze_135071, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133059_cached_sizze_135072 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133059, &mem_133059_cached_sizze_135072, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133077_cached_sizze_135073 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133077, &mem_133077_cached_sizze_135073, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133078_cached_sizze_135074 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133078, &mem_133078_cached_sizze_135074, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133091_cached_sizze_135075 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133091, &mem_133091_cached_sizze_135075, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133092_cached_sizze_135076 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133092, &mem_133092_cached_sizze_135076, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133122_cached_sizze_135077 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133122, &mem_133122_cached_sizze_135077, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133127_cached_sizze_135078 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_133127, &mem_133127_cached_sizze_135078, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133160_cached_sizze_135079 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133160, &mem_133160_cached_sizze_135079, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133165_cached_sizze_135080 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133165, &mem_133165_cached_sizze_135080, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133176_cached_sizze_135081 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133176, &mem_133176_cached_sizze_135081, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133181_cached_sizze_135082 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133181, &mem_133181_cached_sizze_135082, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133192_cached_sizze_135083 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133192, &mem_133192_cached_sizze_135083, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133197_cached_sizze_135084 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133197, &mem_133197_cached_sizze_135084, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133208_cached_sizze_135085 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133208, &mem_133208_cached_sizze_135085, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133209_cached_sizze_135086 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133209, &mem_133209_cached_sizze_135086, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133218_cached_sizze_135087 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133218, &mem_133218_cached_sizze_135087, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133219_cached_sizze_135088 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133219, &mem_133219_cached_sizze_135088, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133232_cached_sizze_135089 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133232, &mem_133232_cached_sizze_135089, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133247_cached_sizze_135090 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133247, &mem_133247_cached_sizze_135090, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133248_cached_sizze_135091 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_133248, &mem_133248_cached_sizze_135091, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133256_cached_sizze_135092 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133256, &mem_133256_cached_sizze_135092, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133270_cached_sizze_135093 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_133270, &mem_133270_cached_sizze_135093, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133275_cached_sizze_135094 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133275, &mem_133275_cached_sizze_135094, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133286_cached_sizze_135095 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133286, &mem_133286_cached_sizze_135095, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133291_cached_sizze_135096 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133291, &mem_133291_cached_sizze_135096, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133302_cached_sizze_135097 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133302, &mem_133302_cached_sizze_135097, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133307_cached_sizze_135098 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133307, &mem_133307_cached_sizze_135098, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133318_cached_sizze_135099 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_133318, &mem_133318_cached_sizze_135099, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133323_cached_sizze_135100 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_133323, &mem_133323_cached_sizze_135100, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133334_cached_sizze_135101 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133334, &mem_133334_cached_sizze_135101, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133335_cached_sizze_135102 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133335, &mem_133335_cached_sizze_135102, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133348_cached_sizze_135103 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_133348, &mem_133348_cached_sizze_135103, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133353_cached_sizze_135104 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_133353, &mem_133353_cached_sizze_135104, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133364_cached_sizze_135105 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133364, &mem_133364_cached_sizze_135105, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133365_cached_sizze_135106 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133365, &mem_133365_cached_sizze_135106, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133372_cached_sizze_135107 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_133372, &mem_133372_cached_sizze_135107, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133385_cached_sizze_135108 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_133385, &mem_133385_cached_sizze_135108, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133390_cached_sizze_135109 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_133390, &mem_133390_cached_sizze_135109, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133397_cached_sizze_135110 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_133397, &mem_133397_cached_sizze_135110, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133408_cached_sizze_135111 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133408, &mem_133408_cached_sizze_135111, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133415_cached_sizze_135112 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_133415, &mem_133415_cached_sizze_135112, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133420_cached_sizze_135113 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_133420, &mem_133420_cached_sizze_135113, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133431_cached_sizze_135114 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133431, &mem_133431_cached_sizze_135114, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133436_cached_sizze_135115 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133436, &mem_133436_cached_sizze_135115, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133447_cached_sizze_135116 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_133447, &mem_133447_cached_sizze_135116, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133448_cached_sizze_135117 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_133448, &mem_133448_cached_sizze_135117, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133457_cached_sizze_135118 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133457, &mem_133457_cached_sizze_135118, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133458_cached_sizze_135119 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133458, &mem_133458_cached_sizze_135119, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133479_cached_sizze_135120 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_133479, &mem_133479_cached_sizze_135120, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133484_cached_sizze_135121 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133484, &mem_133484_cached_sizze_135121, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133495_cached_sizze_135122 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133495, &mem_133495_cached_sizze_135122, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133500_cached_sizze_135123 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133500, &mem_133500_cached_sizze_135123, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133511_cached_sizze_135124 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133511, &mem_133511_cached_sizze_135124, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133512_cached_sizze_135125 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133512, &mem_133512_cached_sizze_135125, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133525_cached_sizze_135126 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133525, &mem_133525_cached_sizze_135126, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133532_cached_sizze_135127 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133532, &mem_133532_cached_sizze_135127, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133542_cached_sizze_135128 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133542, &mem_133542_cached_sizze_135128, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133547_cached_sizze_135129 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133547, &mem_133547_cached_sizze_135129, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133558_cached_sizze_135130 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133558, &mem_133558_cached_sizze_135130, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133559_cached_sizze_135131 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133559, &mem_133559_cached_sizze_135131, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133568_cached_sizze_135132 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133568, &mem_133568_cached_sizze_135132, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133569_cached_sizze_135133 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133569, &mem_133569_cached_sizze_135133, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133590_cached_sizze_135134 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_133590, &mem_133590_cached_sizze_135134, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133591_cached_sizze_135135 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_133591, &mem_133591_cached_sizze_135135, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133592_cached_sizze_135136 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133592, &mem_133592_cached_sizze_135136, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133608_cached_sizze_135137 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133608, &mem_133608_cached_sizze_135137, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133609_cached_sizze_135138 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133609, &mem_133609_cached_sizze_135138, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133610_cached_sizze_135139 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133610, &mem_133610_cached_sizze_135139, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133623_cached_sizze_135140 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_133623, &mem_133623_cached_sizze_135140, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133630_cached_sizze_135141 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133630, &mem_133630_cached_sizze_135141, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133631_cached_sizze_135142 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133631, &mem_133631_cached_sizze_135142, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133671_cached_sizze_135143 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_133671, &mem_133671_cached_sizze_135143, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133672_cached_sizze_135144 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_133672, &mem_133672_cached_sizze_135144, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133673_cached_sizze_135145 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133673, &mem_133673_cached_sizze_135145, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133689_cached_sizze_135146 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133689, &mem_133689_cached_sizze_135146, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133690_cached_sizze_135147 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133690, &mem_133690_cached_sizze_135147, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133691_cached_sizze_135148 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133691, &mem_133691_cached_sizze_135148, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133704_cached_sizze_135149 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_133704, &mem_133704_cached_sizze_135149, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133711_cached_sizze_135150 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133711, &mem_133711_cached_sizze_135150, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133712_cached_sizze_135151 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133712, &mem_133712_cached_sizze_135151, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133752_cached_sizze_135152 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133752, &mem_133752_cached_sizze_135152, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133753_cached_sizze_135153 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133753, &mem_133753_cached_sizze_135153, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133754_cached_sizze_135154 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133754, &mem_133754_cached_sizze_135154, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133755_cached_sizze_135155 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133755, &mem_133755_cached_sizze_135155, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133772_cached_sizze_135156 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133772, &mem_133772_cached_sizze_135156, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133773_cached_sizze_135157 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133773, &mem_133773_cached_sizze_135157, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133774_cached_sizze_135158 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133774, &mem_133774_cached_sizze_135158, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133775_cached_sizze_135159 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133775, &mem_133775_cached_sizze_135159, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133816_cached_sizze_135160 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_133816, &mem_133816_cached_sizze_135160, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133817_cached_sizze_135161 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_133817, &mem_133817_cached_sizze_135161, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133828_cached_sizze_135162 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133828, &mem_133828_cached_sizze_135162, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133829_cached_sizze_135163 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133829, &mem_133829_cached_sizze_135163, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133838_cached_sizze_135164 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133838, &mem_133838_cached_sizze_135164, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133839_cached_sizze_135165 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133839, &mem_133839_cached_sizze_135165, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133870_cached_sizze_135166 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133870, &mem_133870_cached_sizze_135166, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133871_cached_sizze_135167 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133871, &mem_133871_cached_sizze_135167, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133881_cached_sizze_135168 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133881, &mem_133881_cached_sizze_135168, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133882_cached_sizze_135169 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133882, &mem_133882_cached_sizze_135169, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133890_cached_sizze_135170 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_133890, &mem_133890_cached_sizze_135170, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133913_cached_sizze_135171 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133913, &mem_133913_cached_sizze_135171, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133919_cached_sizze_135172 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_133919, &mem_133919_cached_sizze_135172, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133924_cached_sizze_135173 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_133924, &mem_133924_cached_sizze_135173, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133940_cached_sizze_135174 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133940, &mem_133940_cached_sizze_135174, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133941_cached_sizze_135175 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133941, &mem_133941_cached_sizze_135175, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133942_cached_sizze_135176 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133942, &mem_133942_cached_sizze_135176, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133955_cached_sizze_135177 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133955, &mem_133955_cached_sizze_135177, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133956_cached_sizze_135178 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133956, &mem_133956_cached_sizze_135178, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133957_cached_sizze_135179 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_133957, &mem_133957_cached_sizze_135179, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133988_cached_sizze_135180 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133988, &mem_133988_cached_sizze_135180, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133989_cached_sizze_135181 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133989, &mem_133989_cached_sizze_135181, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133990_cached_sizze_135182 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133990, &mem_133990_cached_sizze_135182, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_133991_cached_sizze_135183 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_133991, &mem_133991_cached_sizze_135183, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134008_cached_sizze_135184 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_134008, &mem_134008_cached_sizze_135184, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134009_cached_sizze_135185 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_134009, &mem_134009_cached_sizze_135185, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134010_cached_sizze_135186 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_134010, &mem_134010_cached_sizze_135186, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134011_cached_sizze_135187 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_134011, &mem_134011_cached_sizze_135187, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134052_cached_sizze_135188 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_134052, &mem_134052_cached_sizze_135188, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134053_cached_sizze_135189 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_134053, &mem_134053_cached_sizze_135189, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134066_cached_sizze_135190 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_134066, &mem_134066_cached_sizze_135190, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134073_cached_sizze_135191 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_134073, &mem_134073_cached_sizze_135191, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134083_cached_sizze_135192 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_134083, &mem_134083_cached_sizze_135192, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134088_cached_sizze_135193 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_134088, &mem_134088_cached_sizze_135193, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134099_cached_sizze_135194 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_134099, &mem_134099_cached_sizze_135194, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134100_cached_sizze_135195 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_134100, &mem_134100_cached_sizze_135195, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134113_cached_sizze_135196 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_134113, &mem_134113_cached_sizze_135196, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134120_cached_sizze_135197 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_134120, &mem_134120_cached_sizze_135197, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134130_cached_sizze_135198 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_134130, &mem_134130_cached_sizze_135198, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134135_cached_sizze_135199 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_134135, &mem_134135_cached_sizze_135199, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134146_cached_sizze_135200 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_134146, &mem_134146_cached_sizze_135200, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134147_cached_sizze_135201 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_134147, &mem_134147_cached_sizze_135201, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134156_cached_sizze_135202 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_134156, &mem_134156_cached_sizze_135202, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134157_cached_sizze_135203 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_134157, &mem_134157_cached_sizze_135203, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134178_cached_sizze_135204 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_134178, &mem_134178_cached_sizze_135204, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134183_cached_sizze_135205 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_134183, &mem_134183_cached_sizze_135205, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134194_cached_sizze_135206 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_134194, &mem_134194_cached_sizze_135206, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134195_cached_sizze_135207 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_134195, &mem_134195_cached_sizze_135207, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134204_cached_sizze_135208 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_134204, &mem_134204_cached_sizze_135208, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_134205_cached_sizze_135209 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_134205, &mem_134205_cached_sizze_135209, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:614:5-619:51
    if (memblock_set(ctx, &mem_param_132638, &wdown_mem_132605, "wdown_mem_132605") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132642, &wkey_mem_132606, "wkey_mem_132606") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132646, &wout_mem_132607, "wout_mem_132607") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132650, &wpe_mem_132608, "wpe_mem_132608") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132654, &wqry_mem_132609, "wqry_mem_132609") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132658, &wte_mem_132610, "wte_mem_132610") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132662, &wup_mem_132611, "wup_mem_132611") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132666, &wval_mem_132612, "wval_mem_132612") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132670, &wvoc_mem_132613, "wvoc_mem_132613") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132674, &wdown_mem_132614, "wdown_mem_132614") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132678, &wkey_mem_132615, "wkey_mem_132615") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132682, &wout_mem_132616, "wout_mem_132616") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132686, &wpe_mem_132617, "wpe_mem_132617") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132690, &wqry_mem_132618, "wqry_mem_132618") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132694, &wte_mem_132619, "wte_mem_132619") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132698, &wup_mem_132620, "wup_mem_132620") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132702, &wval_mem_132621, "wval_mem_132621") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132706, &wvoc_mem_132622, "wvoc_mem_132622") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132710, &wdown_mem_132623, "wdown_mem_132623") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132714, &wkey_mem_132624, "wkey_mem_132624") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132718, &wout_mem_132625, "wout_mem_132625") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132722, &wpe_mem_132626, "wpe_mem_132626") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132726, &wqry_mem_132627, "wqry_mem_132627") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132730, &wte_mem_132628, "wte_mem_132628") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132734, &wup_mem_132629, "wup_mem_132629") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132738, &wval_mem_132630, "wval_mem_132630") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_132742, &wvoc_mem_132631, "wvoc_mem_132631") != 0)
        return 1;
    for (int64_t step_122159 = 0; step_122159 < (int64_t) 500; step_122159++) {
        // futhark/microgpt.fut:616:16-25
        
        int64_t dl_122187 = ((int64_t *) dls_mem_132633.mem)[step_122159];
        
        // futhark/microgpt.fut:456:37-40
        
        int64_t zl_rhs_122192 = sub64(dl_122187, (int64_t) 1);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131722 = 0; i_131722 < (int64_t) 16; i_131722++) {
            // futhark/microgpt.fut:456:25-81
            
            bool cond_125204 = slt64(i_131722, zl_rhs_122192);
            
            // futhark/microgpt.fut:456:56-59
            
            int64_t zeze_lhs_125205 = add64((int64_t) 1, i_131722);
            
            // futhark/microgpt.fut:456:47-60
            
            bool x_125206 = sle64((int64_t) 0, zeze_lhs_125205);
            
            // futhark/microgpt.fut:456:47-60
            
            bool y_125207 = slt64(zeze_lhs_125205, (int64_t) 16);
            
            // futhark/microgpt.fut:456:47-60
            
            bool bounds_check_125208 = x_125206 && y_125207;
            
            // futhark/microgpt.fut:9:27-39
            
            bool loop_not_taken_125209 = !cond_125204;
            
            // futhark/microgpt.fut:9:27-39
            
            bool protect_assert_disj_125210 = bounds_check_125208 || loop_not_taken_125209;
            
            // futhark/microgpt.fut:456:47-60
            
            bool index_certs_125211;
            
            if (!protect_assert_disj_125210) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_125205, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:456:47-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:456:3-83\n   #6  futhark/microgpt.fut:563:18-38\n   #7  futhark/microgpt.fut:585:26-591:31\n   #8  futhark/microgpt.fut:619:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            int64_t tmp_125226 = ((int64_t *) seqs_mem_132634.mem)[step_122159 * (int64_t) 16 + i_131722];
            
            // futhark/microgpt.fut:565:37-51
            
            bool x_125227 = sle64((int64_t) 0, tmp_125226);
            
            // futhark/microgpt.fut:565:37-51
            
            bool y_125228 = slt64(tmp_125226, (int64_t) 27);
            
            // futhark/microgpt.fut:565:37-51
            
            bool bounds_check_125229 = x_125227 && y_125228;
            
            // futhark/microgpt.fut:565:37-51
            
            bool index_certs_125230;
            
            if (!bounds_check_125229) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_125226, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:565:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:565:16-55\n   #6  futhark/microgpt.fut:585:26-591:31\n   #7  futhark/microgpt.fut:619:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:456:47-60
            
            int64_t zeze_lhs_125212;
            
            if (cond_125204) {
                int64_t x_131515 = ((int64_t *) seqs_mem_132634.mem)[step_122159 * (int64_t) 16 + zeze_lhs_125205];
                
                zeze_lhs_125212 = x_131515;
            } else {
                zeze_lhs_125212 = (int64_t) 0;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131712 = 0; i_131712 < (int64_t) 27; i_131712++) {
                // futhark/microgpt.fut:456:61-65
                
                bool cond_t_res_125216 = zeze_lhs_125212 == i_131712;
                
                // futhark/microgpt.fut:9:27-39
                
                bool x_125217 = cond_125204 && cond_t_res_125216;
                
                // futhark/microgpt.fut:456:25-81
                
                double lifted_lambda_res_125218;
                
                if (x_125217) {
                    lifted_lambda_res_125218 = 1.0;
                } else {
                    lifted_lambda_res_125218 = 0.0;
                }
                ((double *) mem_132753)[i_131712] = lifted_lambda_res_125218;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131716 = 0; i_131716 < (int64_t) 16; i_131716++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_125237 = ((double *) mem_param_132658.mem)[tmp_125226 * (int64_t) 16 + i_131716];
                
                ((double *) mem_132760)[i_131716] = lifted_lambda_res_125237;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132743, i_131722 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132760, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132744, i_131722 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132753, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131737 = 0; i_131737 < (int64_t) 16; i_131737++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131727 = 0; i_131727 < (int64_t) 16; i_131727++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_125262 = ((double *) mem_param_132650.mem)[i_131737 * (int64_t) 16 + i_131727];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_125263 = ((double *) mem_132743)[i_131737 * (int64_t) 16 + i_131727];
                
                // futhark/microgpt.fut:279:39-75
                
                double zp_res_125264 = zp_lhs_125262 + zp_rhs_125263;
                
                ((double *) mem_132785)[i_131727] = zp_res_125264;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131731 = 0; i_131731 < (int64_t) 27; i_131731++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_125278 = ((double *) mem_132744)[i_131737 * (int64_t) 27 + i_131731];
                
                // futhark/microgpt.fut:312:54-96
                
                double zt_res_125279 = -6.25e-2 * zt_rhs_125278;
                
                ((double *) mem_132792)[i_131731] = zt_res_125279;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132775, i_131737 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132792, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132776, i_131737 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132785, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131755 = 0; i_131755 < (int64_t) 16; i_131755++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131744 = 0; i_131744 < (int64_t) 16; i_131744++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_128057 = ((double *) mem_132776)[i_131755 * (int64_t) 16 + i_131744];
                
                // futhark/microgpt.fut:280:69-102
                
                double zt_res_128058 = zt_lhs_128057 * zt_lhs_128057;
                
                ((double *) mem_132817)[i_131744] = zt_res_128058;
                ((double *) mem_132818)[i_131744] = zt_res_128058;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_125307;
            double r_125309 = 0.0;
            
            for (int64_t i_125308 = 0; i_125308 < (int64_t) 16; i_125308++) {
                // futhark/microgpt.fut:281:35-43
                
                double lifted_lambda_res_125310 = ((double *) mem_132818)[i_125308];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_125311 = r_125309 + lifted_lambda_res_125310;
                double r_tmp_134598 = zp_res_125311;
                
                r_125309 = r_tmp_134598;
            }
            defunc_0_lifted_lambda_res_125307 = r_125309;
            // futhark/microgpt.fut:281:16-60
            
            double zs_res_125312 = defunc_0_lifted_lambda_res_125307 / 16.0;
            
            // futhark/microgpt.fut:282:23-53
            
            double zp_res_125313 = 1.0e-5 + zs_res_125312;
            
            // futhark/microgpt.fut:282:15-53
            
            double sqrt_res_125314 = futrts_sqrt64(zp_res_125313);
            
            // futhark/microgpt.fut:283:25-35
            
            double zs_res_125315 = 1.0 / sqrt_res_125314;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131749 = 0; i_131749 < (int64_t) 16; i_131749++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_125322 = ((double *) mem_132776)[i_131755 * (int64_t) 16 + i_131749];
                
                // futhark/microgpt.fut:283:5-35
                
                double zt_res_125323 = zs_res_125315 * zt_lhs_125322;
                
                ((double *) mem_132831)[i_131749] = zt_res_125323;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132807, i_131755 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132817, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132808, i_131755 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132831, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131775 = 0; i_131775 < (int64_t) 16; i_131775++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131762 = 0; i_131762 < (int64_t) 16; i_131762++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_128087 = ((double *) mem_132808)[i_131775 * (int64_t) 16 + i_131762];
                
                // futhark/microgpt.fut:284:73-110
                
                double zt_res_128088 = zt_lhs_128087 * zt_lhs_128087;
                
                ((double *) mem_132860)[i_131762] = zt_res_128088;
                ((double *) mem_132861)[i_131762] = zt_res_128088;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_125431;
            double r_125433 = 0.0;
            
            for (int64_t i_125432 = 0; i_125432 < (int64_t) 16; i_125432++) {
                // futhark/microgpt.fut:285:37-47
                
                double lifted_lambda_res_125434 = ((double *) mem_132861)[i_125432];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_125435 = r_125433 + lifted_lambda_res_125434;
                double r_tmp_134605 = zp_res_125435;
                
                r_125433 = r_tmp_134605;
            }
            defunc_0_lifted_lambda_res_125431 = r_125433;
            // futhark/microgpt.fut:285:17-64
            
            double zs_res_125436 = defunc_0_lifted_lambda_res_125431 / 16.0;
            
            // futhark/microgpt.fut:286:24-55
            
            double zp_res_125437 = 1.0e-5 + zs_res_125436;
            
            // futhark/microgpt.fut:286:16-55
            
            double sqrt_res_125438 = futrts_sqrt64(zp_res_125437);
            
            // futhark/microgpt.fut:287:27-38
            
            double zs_res_125439 = 1.0 / sqrt_res_125438;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131767 = 0; i_131767 < (int64_t) 16; i_131767++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_125446 = ((double *) mem_132808)[i_131775 * (int64_t) 16 + i_131767];
                
                // futhark/microgpt.fut:287:5-38
                
                double zt_res_125447 = zs_res_125439 * zt_lhs_125446;
                
                ((double *) mem_132874)[i_131767] = zt_res_125447;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_125473;
            double r_125475 = 0.0;
            
            for (int64_t i_125474 = 0; i_125474 < (int64_t) 16; i_125474++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_125476 = ((double *) mem_132807)[i_131775 * (int64_t) 16 + i_125474];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_125477 = r_125475 + lifted_lambda_res_125476;
                double r_tmp_134607 = zp_res_125477;
                
                r_125475 = r_tmp_134607;
            }
            defunc_0_lifted_lambda_res_125473 = r_125475;
            // futhark/microgpt.fut:375:40-98
            
            double zs_res_125478 = defunc_0_lifted_lambda_res_125473 / 16.0;
            
            ((double *) mem_132846)[i_131775] = zs_res_125478;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132847, i_131775 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132860, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132848, i_131775 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132874, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131797 = 0; i_131797 < (int64_t) 16; i_131797++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131785 = 0; i_131785 < (int64_t) 16; i_131785++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_128160;
                double r_128162 = 0.0;
                
                for (int64_t i_128161 = 0; i_128161 < (int64_t) 16; i_128161++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_128163 = ((double *) mem_param_132654.mem)[i_131785 * (int64_t) 16 + i_128161];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_128164 = ((double *) mem_132848)[i_131797 * (int64_t) 16 + i_128161];
                    
                    // futhark/microgpt.fut:288:63-102
                    
                    double zt_res_128165 = zt_lhs_128163 * zt_rhs_128164;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_128166 = r_128162 + zt_res_128165;
                    double r_tmp_134615 = zp_res_128166;
                    
                    r_128162 = r_tmp_134615;
                }
                defunc_0_lifted_lambda_res_128160 = r_128162;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_128173;
                double r_128175 = 0.0;
                
                for (int64_t i_128174 = 0; i_128174 < (int64_t) 16; i_128174++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_128176 = ((double *) mem_param_132642.mem)[i_131785 * (int64_t) 16 + i_128174];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_128177 = ((double *) mem_132848)[i_131797 * (int64_t) 16 + i_128174];
                    
                    // futhark/microgpt.fut:289:63-102
                    
                    double zt_res_128178 = zt_lhs_128176 * zt_rhs_128177;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_128179 = r_128175 + zt_res_128178;
                    double r_tmp_134616 = zp_res_128179;
                    
                    r_128175 = r_tmp_134616;
                }
                defunc_0_lifted_lambda_res_128173 = r_128175;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_128189;
                double r_128191 = 0.0;
                
                for (int64_t i_128190 = 0; i_128190 < (int64_t) 16; i_128190++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_128192 = ((double *) mem_param_132666.mem)[i_131785 * (int64_t) 16 + i_128190];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_128193 = ((double *) mem_132848)[i_131797 * (int64_t) 16 + i_128190];
                    
                    // futhark/microgpt.fut:290:63-102
                    
                    double zt_res_128194 = zt_lhs_128192 * zt_rhs_128193;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_128195 = r_128191 + zt_res_128194;
                    double r_tmp_134617 = zp_res_128195;
                    
                    r_128191 = r_tmp_134617;
                }
                defunc_0_lifted_lambda_res_128189 = r_128191;
                ((double *) mem_132911)[i_131785] = defunc_0_lifted_lambda_res_128189;
                ((double *) mem_132912)[i_131785] = defunc_0_lifted_lambda_res_128173;
                ((double *) mem_132913)[i_131785] = defunc_0_lifted_lambda_res_128160;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_125709;
            double r_125711 = 0.0;
            
            for (int64_t i_125710 = 0; i_125710 < (int64_t) 16; i_125710++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_125712 = ((double *) mem_132847)[i_131797 * (int64_t) 16 + i_125710];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_125713 = r_125711 + lifted_lambda_res_125712;
                double r_tmp_134618 = zp_res_125713;
                
                r_125711 = r_tmp_134618;
            }
            defunc_0_lifted_lambda_res_125709 = r_125711;
            // futhark/microgpt.fut:368:40-98
            
            double zs_res_125714 = defunc_0_lifted_lambda_res_125709 / 16.0;
            
            ((double *) mem_132892)[i_131797] = zs_res_125714;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132893, i_131797 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132911, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132894, i_131797 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132912, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_132895, i_131797 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132913, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131828 = 0; i_131828 < (int64_t) 4; i_131828++) {
            // futhark/microgpt.fut:291:67-70
            
            int64_t zp_lhs_125785 = mul64((int64_t) 4, i_131828);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131818 = 0; i_131818 < (int64_t) 16; i_131818++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_131808 = 0; i_131808 < (int64_t) 4; i_131808++) {
                    // futhark/microgpt.fut:291:72-79
                    
                    int64_t tmp_128353 = add64(zp_lhs_125785, i_131808);
                    
                    // futhark/microgpt.fut:291:48-81
                    
                    bool x_128354 = sle64((int64_t) 0, tmp_128353);
                    
                    // futhark/microgpt.fut:291:48-81
                    
                    bool y_128355 = slt64(tmp_128353, (int64_t) 16);
                    
                    // futhark/microgpt.fut:291:48-81
                    
                    bool bounds_check_128356 = x_128354 && y_128355;
                    
                    // futhark/microgpt.fut:291:48-81
                    
                    bool index_certs_128357;
                    
                    if (!bounds_check_128356) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_128353, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:291:48-81\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:291:12-82\n   #9  futhark/microgpt.fut:568:5-76\n   #10 futhark/microgpt.fut:585:26-591:31\n   #11 futhark/microgpt.fut:619:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_128358 = ((double *) mem_132895)[i_131818 * (int64_t) 16 + tmp_128353];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_128366 = ((double *) mem_132894)[i_131818 * (int64_t) 16 + tmp_128353];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_128377 = ((double *) mem_132893)[i_131818 * (int64_t) 16 + tmp_128353];
                    
                    ((double *) mem_132980)[i_131808] = lifted_lambda_res_128377;
                    ((double *) mem_132981)[i_131808] = lifted_lambda_res_128366;
                    ((double *) mem_132982)[i_131808] = lifted_lambda_res_128358;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_132965, i_131818 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132980, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_132966, i_131818 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132981, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_132967, i_131818 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_132982, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_132947, i_131828 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_132965, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_132948, i_131828 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_132966, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_132949, i_131828 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_132967, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131885 = 0; i_131885 < (int64_t) 4; i_131885++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131861 = 0; i_131861 < (int64_t) 16; i_131861++) {
                // futhark/microgpt.fut:115:13-33
                
                double defunc_0_reduce_res_131535;
                double defunc_0_reduce_res_131536;
                double defunc_0_reduce_res_131537;
                double defunc_0_reduce_res_131538;
                double redout_131832;
                double redout_131833;
                double redout_131834;
                double redout_131835;
                
                redout_131832 = -INFINITY;
                redout_131833 = -INFINITY;
                redout_131834 = -INFINITY;
                redout_131835 = -INFINITY;
                for (int64_t i_131836 = 0; i_131836 < (int64_t) 16; i_131836++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_129426;
                    double r_129428 = 0.0;
                    
                    for (int64_t i_129427 = 0; i_129427 < (int64_t) 4; i_129427++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_129429 = ((double *) mem_132949)[i_131885 * (int64_t) 64 + i_131861 * (int64_t) 4 + i_129427];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_129430 = ((double *) mem_132948)[i_131885 * (int64_t) 64 + i_131836 * (int64_t) 4 + i_129427];
                        
                        // futhark/microgpt.fut:294:148-201
                        
                        double zt_res_129431 = zt_lhs_129429 * zt_rhs_129430;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_129432 = r_129428 + zt_res_129431;
                        double r_tmp_134642 = zp_res_129432;
                        
                        r_129428 = r_tmp_134642;
                    }
                    defunc_0_lifted_lambda_res_129426 = r_129428;
                    // futhark/microgpt.fut:294:128-218
                    
                    double zs_res_129433 = defunc_0_lifted_lambda_res_129426 / 2.0;
                    double zp_rhs_129434 = ((double *) masks_mem_132632.mem)[step_122159 * (int64_t) 256 + i_131861 * (int64_t) 16 + i_131836];
                    
                    // futhark/microgpt.fut:294:205-242
                    
                    double zp_res_129435 = zs_res_129433 + zp_rhs_129434;
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_129445;
                    double r_129447 = 0.0;
                    
                    for (int64_t i_129446 = 0; i_129446 < (int64_t) 4; i_129446++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_129448 = ((double *) mem_132949)[i_131885 * (int64_t) 64 + i_131861 * (int64_t) 4 + i_129446];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_129449 = ((double *) mem_132948)[i_131885 * (int64_t) 64 + i_131836 * (int64_t) 4 + i_129446];
                        
                        // futhark/microgpt.fut:340:127-186
                        
                        double zt_res_129450 = zt_lhs_129448 * zt_rhs_129449;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_129451 = r_129447 + zt_res_129450;
                        double r_tmp_134643 = zp_res_129451;
                        
                        r_129447 = r_tmp_134643;
                    }
                    defunc_0_lifted_lambda_res_129445 = r_129447;
                    // futhark/microgpt.fut:340:106-203
                    
                    double zs_res_129452 = defunc_0_lifted_lambda_res_129445 / 2.0;
                    
                    // futhark/microgpt.fut:340:190-229
                    
                    double zp_res_129454 = zp_rhs_129434 + zs_res_129452;
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_129482;
                    double r_129484 = 0.0;
                    
                    for (int64_t i_129483 = 0; i_129483 < (int64_t) 4; i_129483++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_129485 = ((double *) mem_132949)[i_131885 * (int64_t) 64 + i_131861 * (int64_t) 4 + i_129483];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_129486 = ((double *) mem_132948)[i_131885 * (int64_t) 64 + i_131836 * (int64_t) 4 + i_129483];
                        
                        // futhark/microgpt.fut:346:97-156
                        
                        double zt_res_129487 = zt_lhs_129485 * zt_rhs_129486;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_129488 = r_129484 + zt_res_129487;
                        double r_tmp_134644 = zp_res_129488;
                        
                        r_129484 = r_tmp_134644;
                    }
                    defunc_0_lifted_lambda_res_129482 = r_129484;
                    // futhark/microgpt.fut:346:76-173
                    
                    double zs_res_129489 = defunc_0_lifted_lambda_res_129482 / 2.0;
                    
                    // futhark/microgpt.fut:346:160-199
                    
                    double zp_res_129491 = zp_rhs_129434 + zs_res_129489;
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_129521;
                    double r_129523 = 0.0;
                    
                    for (int64_t i_129522 = 0; i_129522 < (int64_t) 4; i_129522++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_129524 = ((double *) mem_132949)[i_131885 * (int64_t) 64 + i_131861 * (int64_t) 4 + i_129522];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_129525 = ((double *) mem_132948)[i_131885 * (int64_t) 64 + i_131836 * (int64_t) 4 + i_129522];
                        
                        // futhark/microgpt.fut:355:97-156
                        
                        double zt_res_129526 = zt_lhs_129524 * zt_rhs_129525;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_129527 = r_129523 + zt_res_129526;
                        double r_tmp_134645 = zp_res_129527;
                        
                        r_129523 = r_tmp_134645;
                    }
                    defunc_0_lifted_lambda_res_129521 = r_129523;
                    // futhark/microgpt.fut:355:76-173
                    
                    double zs_res_129528 = defunc_0_lifted_lambda_res_129521 / 2.0;
                    
                    // futhark/microgpt.fut:355:160-199
                    
                    double zp_res_129530 = zp_rhs_129434 + zs_res_129528;
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_128859 = fmax64(zp_res_129435, redout_131832);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_128922 = fmax64(zp_res_129454, redout_131833);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_128989 = fmax64(zp_res_129491, redout_131834);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_129055 = fmax64(zp_res_129530, redout_131835);
                    double redout_tmp_134638 = max_res_128859;
                    double redout_tmp_134639 = max_res_128922;
                    double redout_tmp_134640 = max_res_128989;
                    double redout_tmp_134641 = max_res_129055;
                    
                    redout_131832 = redout_tmp_134638;
                    redout_131833 = redout_tmp_134639;
                    redout_131834 = redout_tmp_134640;
                    redout_131835 = redout_tmp_134641;
                }
                defunc_0_reduce_res_131535 = redout_131832;
                defunc_0_reduce_res_131536 = redout_131833;
                defunc_0_reduce_res_131537 = redout_131834;
                defunc_0_reduce_res_131538 = redout_131835;
                // futhark/microgpt.fut:295:168-177
                
                double neg_res_128860 = -defunc_0_reduce_res_131535;
                
                // futhark/microgpt.fut:341:179-189
                
                double neg_res_128923 = -defunc_0_reduce_res_131536;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_131841 = 0; i_131841 < (int64_t) 16; i_131841++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_129572;
                    double r_129574 = 0.0;
                    
                    for (int64_t i_129573 = 0; i_129573 < (int64_t) 4; i_129573++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_129575 = ((double *) mem_132949)[i_131885 * (int64_t) 64 + i_131861 * (int64_t) 4 + i_129573];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_129576 = ((double *) mem_132948)[i_131885 * (int64_t) 64 + i_131841 * (int64_t) 4 + i_129573];
                        
                        // futhark/microgpt.fut:295:67-120
                        
                        double zt_res_129577 = zt_lhs_129575 * zt_rhs_129576;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_129578 = r_129574 + zt_res_129577;
                        double r_tmp_134648 = zp_res_129578;
                        
                        r_129574 = r_tmp_134648;
                    }
                    defunc_0_lifted_lambda_res_129572 = r_129574;
                    // futhark/microgpt.fut:295:47-137
                    
                    double zs_res_129579 = defunc_0_lifted_lambda_res_129572 / 2.0;
                    double zp_rhs_129580 = ((double *) masks_mem_132632.mem)[step_122159 * (int64_t) 256 + i_131861 * (int64_t) 16 + i_131841];
                    
                    // futhark/microgpt.fut:295:124-161
                    
                    double zp_res_129581 = zs_res_129579 + zp_rhs_129580;
                    
                    // futhark/microgpt.fut:295:139-177
                    
                    double zp_res_129582 = neg_res_128860 + zp_res_129581;
                    
                    // futhark/microgpt.fut:295:37-177
                    
                    double exp_res_129583 = futrts_exp64(zp_res_129582);
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_129590;
                    double r_129592 = 0.0;
                    
                    for (int64_t i_129591 = 0; i_129591 < (int64_t) 4; i_129591++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_129593 = ((double *) mem_132949)[i_131885 * (int64_t) 64 + i_131861 * (int64_t) 4 + i_129591];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_129594 = ((double *) mem_132948)[i_131885 * (int64_t) 64 + i_131841 * (int64_t) 4 + i_129591];
                        
                        // futhark/microgpt.fut:341:70-129
                        
                        double zt_res_129595 = zt_lhs_129593 * zt_rhs_129594;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_129596 = r_129592 + zt_res_129595;
                        double r_tmp_134649 = zp_res_129596;
                        
                        r_129592 = r_tmp_134649;
                    }
                    defunc_0_lifted_lambda_res_129590 = r_129592;
                    // futhark/microgpt.fut:341:49-146
                    
                    double zs_res_129597 = defunc_0_lifted_lambda_res_129590 / 2.0;
                    
                    // futhark/microgpt.fut:341:133-172
                    
                    double zp_res_129599 = zp_rhs_129580 + zs_res_129597;
                    
                    // futhark/microgpt.fut:341:148-189
                    
                    double zp_res_129600 = neg_res_128923 + zp_res_129599;
                    
                    // futhark/microgpt.fut:341:39-189
                    
                    double exp_res_129601 = futrts_exp64(zp_res_129600);
                    
                    ((double *) mem_133077)[i_131841] = exp_res_129601;
                    ((double *) mem_133078)[i_131841] = exp_res_129583;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_128880;
                double r_128882 = 0.0;
                
                for (int64_t i_128881 = 0; i_128881 < (int64_t) 16; i_128881++) {
                    // futhark/microgpt.fut:296:36-46
                    
                    double lifted_lambda_res_128883 = ((double *) mem_133078)[i_128881];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_128884 = r_128882 + lifted_lambda_res_128883;
                    double r_tmp_134650 = zp_res_128884;
                    
                    r_128882 = r_tmp_134650;
                }
                defunc_0_lifted_lambda_res_128880 = r_128882;
                // futhark/microgpt.fut:297:21-32
                
                double zs_res_128885 = 1.0 / defunc_0_lifted_lambda_res_128880;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_128943;
                double r_128945 = 0.0;
                
                for (int64_t i_128944 = 0; i_128944 < (int64_t) 16; i_128944++) {
                    // futhark/microgpt.fut:342:38-50
                    
                    double lifted_lambda_res_128946 = ((double *) mem_133077)[i_128944];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_128947 = r_128945 + lifted_lambda_res_128946;
                    double r_tmp_134651 = zp_res_128947;
                    
                    r_128945 = r_tmp_134651;
                }
                defunc_0_lifted_lambda_res_128943 = r_128945;
                // futhark/microgpt.fut:343:23-35
                
                double zs_res_128948 = 1.0 / defunc_0_lifted_lambda_res_128943;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_131848 = 0; i_131848 < (int64_t) 16; i_131848++) {
                    // futhark/microgpt.fut:297:5-15
                    
                    double zt_lhs_129619 = ((double *) mem_133078)[i_131848];
                    
                    // futhark/microgpt.fut:297:5-32
                    
                    double zt_res_129620 = zs_res_128885 * zt_lhs_129619;
                    
                    // futhark/microgpt.fut:343:5-17
                    
                    double zt_lhs_129627 = ((double *) mem_133077)[i_131848];
                    
                    // futhark/microgpt.fut:343:5-35
                    
                    double zt_res_129628 = zs_res_128948 * zt_lhs_129627;
                    
                    ((double *) mem_133091)[i_131848] = zt_res_129628;
                    ((double *) mem_133092)[i_131848] = zt_res_129620;
                }
                // futhark/microgpt.fut:352:247-273
                
                double neg_res_128997 = -defunc_0_reduce_res_131537;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_128998;
                double r_129000 = 0.0;
                
                for (int64_t i_128999 = 0; i_128999 < (int64_t) 16; i_128999++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_129001;
                    double r_129003 = 0.0;
                    
                    for (int64_t i_129002 = 0; i_129002 < (int64_t) 4; i_129002++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_129004 = ((double *) mem_132949)[i_131885 * (int64_t) 64 + i_131861 * (int64_t) 4 + i_129002];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_129005 = ((double *) mem_132948)[i_131885 * (int64_t) 64 + i_128999 * (int64_t) 4 + i_129002];
                        
                        // futhark/microgpt.fut:352:138-197
                        
                        double zt_res_129006 = zt_lhs_129004 * zt_rhs_129005;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_129007 = r_129003 + zt_res_129006;
                        double r_tmp_134655 = zp_res_129007;
                        
                        r_129003 = r_tmp_134655;
                    }
                    defunc_0_lifted_lambda_res_129001 = r_129003;
                    // futhark/microgpt.fut:352:117-214
                    
                    double zs_res_129008 = defunc_0_lifted_lambda_res_129001 / 2.0;
                    double zp_rhs_129009 = ((double *) masks_mem_132632.mem)[step_122159 * (int64_t) 256 + i_131861 * (int64_t) 16 + i_128999];
                    
                    // futhark/microgpt.fut:352:201-240
                    
                    double zp_res_129010 = zs_res_129008 + zp_rhs_129009;
                    
                    // futhark/microgpt.fut:352:216-273
                    
                    double zp_res_129011 = neg_res_128997 + zp_res_129010;
                    
                    // futhark/microgpt.fut:352:107-273
                    
                    double neg_res_129012 = -zp_res_129011;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_129013 = fmax64(0.0, neg_res_129012);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_129014 = fsignum64(max_res_129013);
                    
                    // futhark/microgpt.fut:352:88-276
                    
                    double neg_res_129015 = -sgn_res_129014;
                    
                    // futhark/microgpt.fut:352:79-277
                    
                    double zp_res_129016 = 1.0 + neg_res_129015;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_129017 = r_129000 + zp_res_129016;
                    double r_tmp_134654 = zp_res_129017;
                    
                    r_129000 = r_tmp_134654;
                }
                defunc_0_lifted_lambda_res_128998 = r_129000;
                // futhark/microgpt.fut:352:48-280
                
                double zs_res_129018 = 1.0 / defunc_0_lifted_lambda_res_128998;
                
                ((double *) mem_133055)[i_131861] = defunc_0_reduce_res_131538;
                ((double *) mem_133056)[i_131861] = zs_res_129018;
                ((double *) mem_133057)[i_131861] = defunc_0_reduce_res_131537;
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_133058, i_131861 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133091, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_133059, i_131861 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133092, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131873 = 0; i_131873 < (int64_t) 16; i_131873++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_131869 = 0; i_131869 < (int64_t) 4; i_131869++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_126387;
                    double r_126389 = 0.0;
                    
                    for (int64_t i_126388 = 0; i_126388 < (int64_t) 16; i_126388++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_126390 = ((double *) mem_133059)[i_131873 * (int64_t) 16 + i_126388];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_126391 = ((double *) mem_132947)[i_131885 * (int64_t) 64 + i_126388 * (int64_t) 4 + i_131869];
                        
                        // futhark/microgpt.fut:298:26-72
                        
                        double zt_res_126392 = zt_lhs_126390 * zt_rhs_126391;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_126393 = r_126389 + zt_res_126392;
                        double r_tmp_134658 = zp_res_126393;
                        
                        r_126389 = r_tmp_134658;
                    }
                    defunc_0_lifted_lambda_res_126387 = r_126389;
                    ((double *) mem_133127)[i_131869] = defunc_0_lifted_lambda_res_126387;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_133122, i_131873 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133127, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133028, i_131885 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133055, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133029, i_131885 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133056, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133030, i_131885 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133057, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_133031, i_131885 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_133058, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_133032, i_131885 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_133122, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131897 = 0; i_131897 < (int64_t) 16; i_131897++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131893 = 0; i_131893 < (int64_t) 16; i_131893++) {
                // futhark/microgpt.fut:299:52-55
                
                int64_t tmp_122541 = sdiv64(i_131893, (int64_t) 4);
                
                // futhark/microgpt.fut:299:41-57
                
                bool x_122542 = sle64((int64_t) 0, tmp_122541);
                
                // futhark/microgpt.fut:299:41-57
                
                bool y_122543 = slt64(tmp_122541, (int64_t) 4);
                
                // futhark/microgpt.fut:299:41-57
                
                bool bounds_check_122544 = x_122542 && y_122543;
                
                // futhark/microgpt.fut:299:41-57
                
                bool index_certs_122545;
                
                if (!bounds_check_122544) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_122541, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:299:41-57\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:299:12-78\n   #6  futhark/microgpt.fut:568:5-76\n   #7  futhark/microgpt.fut:585:26-591:31\n   #8  futhark/microgpt.fut:619:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:299:72-75
                
                int64_t tmp_122546 = smod64(i_131893, (int64_t) 4);
                
                // futhark/microgpt.fut:299:41-77
                
                bool x_122547 = sle64((int64_t) 0, tmp_122546);
                
                // futhark/microgpt.fut:299:41-77
                
                bool y_122548 = slt64(tmp_122546, (int64_t) 4);
                
                // futhark/microgpt.fut:299:41-77
                
                bool bounds_check_122549 = x_122547 && y_122548;
                
                // futhark/microgpt.fut:299:41-77
                
                bool index_certs_122550;
                
                if (!bounds_check_122549) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_122546, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:299:41-77\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:299:12-78\n   #6  futhark/microgpt.fut:568:5-76\n   #7  futhark/microgpt.fut:585:26-591:31\n   #8  futhark/microgpt.fut:619:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_122551 = ((double *) mem_133032)[tmp_122541 * (int64_t) 64 + i_131897 * (int64_t) 4 + tmp_122546];
                
                ((double *) mem_133165)[i_131893] = lifted_lambda_res_122551;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133160, i_131897 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133165, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131905 = 0; i_131905 < (int64_t) 16; i_131905++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131901 = 0; i_131901 < (int64_t) 16; i_131901++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_122566;
                double r_122568 = 0.0;
                
                for (int64_t i_122567 = 0; i_122567 < (int64_t) 16; i_122567++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_122569 = ((double *) mem_param_132646.mem)[i_131901 * (int64_t) 16 + i_122567];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_122570 = ((double *) mem_133160)[i_131905 * (int64_t) 16 + i_122567];
                    
                    // futhark/microgpt.fut:300:63-103
                    
                    double zt_res_122571 = zt_lhs_122569 * zt_rhs_122570;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_122572 = r_122568 + zt_res_122571;
                    double r_tmp_134663 = zp_res_122572;
                    
                    r_122568 = r_tmp_134663;
                }
                defunc_0_lifted_lambda_res_122566 = r_122568;
                ((double *) mem_133181)[i_131901] = defunc_0_lifted_lambda_res_122566;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133176, i_131905 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133181, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131913 = 0; i_131913 < (int64_t) 16; i_131913++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131909 = 0; i_131909 < (int64_t) 16; i_131909++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_122587 = ((double *) mem_133176)[i_131913 * (int64_t) 16 + i_131909];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_122588 = ((double *) mem_132808)[i_131913 * (int64_t) 16 + i_131909];
                
                // futhark/microgpt.fut:301:42-80
                
                double zp_res_122589 = zp_lhs_122587 + zp_rhs_122588;
                
                ((double *) mem_133197)[i_131909] = zp_res_122589;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133192, i_131913 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133197, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131930 = 0; i_131930 < (int64_t) 16; i_131930++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131919 = 0; i_131919 < (int64_t) 16; i_131919++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_129673 = ((double *) mem_133192)[i_131930 * (int64_t) 16 + i_131919];
                
                // futhark/microgpt.fut:302:74-113
                
                double zt_res_129674 = zt_lhs_129673 * zt_lhs_129673;
                
                ((double *) mem_133218)[i_131919] = zt_res_129674;
                ((double *) mem_133219)[i_131919] = zt_res_129674;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_126615;
            double r_126617 = 0.0;
            
            for (int64_t i_126616 = 0; i_126616 < (int64_t) 16; i_126616++) {
                // futhark/microgpt.fut:303:37-47
                
                double lifted_lambda_res_126618 = ((double *) mem_133219)[i_126616];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_126619 = r_126617 + lifted_lambda_res_126618;
                double r_tmp_134670 = zp_res_126619;
                
                r_126617 = r_tmp_134670;
            }
            defunc_0_lifted_lambda_res_126615 = r_126617;
            // futhark/microgpt.fut:303:17-64
            
            double zs_res_126620 = defunc_0_lifted_lambda_res_126615 / 16.0;
            
            // futhark/microgpt.fut:304:24-55
            
            double zp_res_126621 = 1.0e-5 + zs_res_126620;
            
            // futhark/microgpt.fut:304:16-55
            
            double sqrt_res_126622 = futrts_sqrt64(zp_res_126621);
            
            // futhark/microgpt.fut:305:28-39
            
            double zs_res_126623 = 1.0 / sqrt_res_126622;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131924 = 0; i_131924 < (int64_t) 16; i_131924++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_126630 = ((double *) mem_133192)[i_131930 * (int64_t) 16 + i_131924];
                
                // futhark/microgpt.fut:305:5-39
                
                double zt_res_126631 = zs_res_126623 * zt_lhs_126630;
                
                ((double *) mem_133232)[i_131924] = zt_res_126631;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133208, i_131930 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133218, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133209, i_131930 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133232, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131941 = 0; i_131941 < (int64_t) 16; i_131941++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131935 = 0; i_131935 < (int64_t) 64; i_131935++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_126671;
                double r_126673 = 0.0;
                
                for (int64_t i_126672 = 0; i_126672 < (int64_t) 16; i_126672++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_126674 = ((double *) mem_param_132662.mem)[i_131935 * (int64_t) 16 + i_126672];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_126675 = ((double *) mem_133209)[i_131941 * (int64_t) 16 + i_126672];
                    
                    // futhark/microgpt.fut:306:63-102
                    
                    double zt_res_126676 = zt_lhs_126674 * zt_rhs_126675;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_126677 = r_126673 + zt_res_126676;
                    double r_tmp_134675 = zp_res_126677;
                    
                    r_126673 = r_tmp_134675;
                }
                defunc_0_lifted_lambda_res_126671 = r_126673;
                ((double *) mem_133256)[i_131935] = defunc_0_lifted_lambda_res_126671;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_126685;
            double r_126687 = 0.0;
            
            for (int64_t i_126686 = 0; i_126686 < (int64_t) 16; i_126686++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_126688 = ((double *) mem_133208)[i_131941 * (int64_t) 16 + i_126686];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_126689 = r_126687 + lifted_lambda_res_126688;
                double r_tmp_134676 = zp_res_126689;
                
                r_126687 = r_tmp_134676;
            }
            defunc_0_lifted_lambda_res_126685 = r_126687;
            // futhark/microgpt.fut:332:40-98
            
            double zs_res_126690 = defunc_0_lifted_lambda_res_126685 / 16.0;
            
            ((double *) mem_133247)[i_131941] = zs_res_126690;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133248, i_131941 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133256, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131950 = 0; i_131950 < (int64_t) 16; i_131950++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131946 = 0; i_131946 < (int64_t) 64; i_131946++) {
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_122660 = ((double *) mem_133248)[i_131950 * (int64_t) 64 + i_131946];
                
                // futhark/microgpt.fut:307:41-69
                
                double max_res_122661 = fmax64(0.0, max_arg0_122660);
                
                ((double *) mem_133275)[i_131946] = max_res_122661;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133270, i_131950 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133275, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131958 = 0; i_131958 < (int64_t) 16; i_131958++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131954 = 0; i_131954 < (int64_t) 16; i_131954++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_122676;
                double r_122678 = 0.0;
                
                for (int64_t i_122677 = 0; i_122677 < (int64_t) 64; i_122677++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_122679 = ((double *) mem_param_132638.mem)[i_131954 * (int64_t) 64 + i_122677];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_122680 = ((double *) mem_133270)[i_131958 * (int64_t) 64 + i_122677];
                    
                    // futhark/microgpt.fut:308:63-104
                    
                    double zt_res_122681 = zt_lhs_122679 * zt_rhs_122680;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_122682 = r_122678 + zt_res_122681;
                    double r_tmp_134681 = zp_res_122682;
                    
                    r_122678 = r_tmp_134681;
                }
                defunc_0_lifted_lambda_res_122676 = r_122678;
                ((double *) mem_133291)[i_131954] = defunc_0_lifted_lambda_res_122676;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133286, i_131958 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133291, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131966 = 0; i_131966 < (int64_t) 16; i_131966++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131962 = 0; i_131962 < (int64_t) 16; i_131962++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_122697 = ((double *) mem_133286)[i_131966 * (int64_t) 16 + i_131962];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_122698 = ((double *) mem_133192)[i_131966 * (int64_t) 16 + i_131962];
                
                // futhark/microgpt.fut:309:42-81
                
                double zp_res_122699 = zp_lhs_122697 + zp_rhs_122698;
                
                ((double *) mem_133307)[i_131962] = zp_res_122699;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133302, i_131966 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133307, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131974 = 0; i_131974 < (int64_t) 16; i_131974++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131970 = 0; i_131970 < (int64_t) 27; i_131970++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_122714;
                double r_122716 = 0.0;
                
                for (int64_t i_122715 = 0; i_122715 < (int64_t) 16; i_122715++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_122717 = ((double *) mem_param_132670.mem)[i_131970 * (int64_t) 16 + i_122715];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_122718 = ((double *) mem_133302)[i_131974 * (int64_t) 16 + i_122715];
                    
                    // futhark/microgpt.fut:310:63-103
                    
                    double zt_res_122719 = zt_lhs_122717 * zt_rhs_122718;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_122720 = r_122716 + zt_res_122719;
                    double r_tmp_134686 = zp_res_122720;
                    
                    r_122716 = r_tmp_134686;
                }
                defunc_0_lifted_lambda_res_122714 = r_122716;
                ((double *) mem_133323)[i_131970] = defunc_0_lifted_lambda_res_122714;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133318, i_131974 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133323, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131982 = 0; i_131982 < (int64_t) 16; i_131982++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_131561;
            double redout_131976 = -INFINITY;
            
            for (int64_t i_131977 = 0; i_131977 < (int64_t) 27; i_131977++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129705 = ((double *) mem_133318)[i_131982 * (int64_t) 27 + i_131977];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_125170 = fmax64(lifted_lambda_res_129705, redout_131976);
                double redout_tmp_134689 = max_res_125170;
                
                redout_131976 = redout_tmp_134689;
            }
            defunc_0_reduce_res_131561 = redout_131976;
            // futhark/microgpt.fut:325:129-146
            
            double neg_res_125178 = -defunc_0_reduce_res_131561;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_125179;
            double r_125181 = 0.0;
            
            for (int64_t i_125180 = 0; i_125180 < (int64_t) 27; i_125180++) {
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_125182 = ((double *) mem_133318)[i_131982 * (int64_t) 27 + i_125180];
                
                // futhark/microgpt.fut:325:104-146
                
                double zp_res_125183 = neg_res_125178 + zp_lhs_125182;
                
                // futhark/microgpt.fut:325:97-146
                
                double neg_res_125184 = -zp_res_125183;
                
                // futhark/microgpt.fut:110:42-54
                
                double max_res_125185 = fmax64(0.0, neg_res_125184);
                
                // futhark/microgpt.fut:110:35-54
                
                double sgn_res_125186 = fsignum64(max_res_125185);
                
                // futhark/microgpt.fut:325:78-149
                
                double neg_res_125187 = -sgn_res_125186;
                
                // futhark/microgpt.fut:325:69-150
                
                double zp_res_125188 = 1.0 + neg_res_125187;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_125189 = r_125181 + zp_res_125188;
                double r_tmp_134690 = zp_res_125189;
                
                r_125181 = r_tmp_134690;
            }
            defunc_0_lifted_lambda_res_125179 = r_125181;
            // futhark/microgpt.fut:325:38-153
            
            double zs_res_125190 = 1.0 / defunc_0_lifted_lambda_res_125179;
            
            ((double *) mem_133334)[i_131982] = zs_res_125190;
            ((double *) mem_133335)[i_131982] = defunc_0_reduce_res_131561;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_131991 = 0; i_131991 < (int64_t) 16; i_131991++) {
            // futhark/microgpt.fut:314:82-92
            
            double neg_arg0_122765 = ((double *) mem_133335)[i_131991];
            
            // futhark/microgpt.fut:314:76-92
            
            double neg_res_122766 = -neg_arg0_122765;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_131987 = 0; i_131987 < (int64_t) 27; i_131987++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_122773 = ((double *) mem_133318)[i_131991 * (int64_t) 27 + i_131987];
                
                // futhark/microgpt.fut:314:53-92
                
                double zp_res_122774 = neg_res_122766 + zp_lhs_122773;
                
                // futhark/microgpt.fut:314:46-92
                
                double exp_res_122775 = futrts_exp64(zp_res_122774);
                
                ((double *) mem_133353)[i_131987] = exp_res_122775;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133348, i_131991 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133353, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132003 = 0; i_132003 < (int64_t) 16; i_132003++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_125084;
            double r_125086 = 0.0;
            
            for (int64_t i_125085 = 0; i_125085 < (int64_t) 27; i_125085++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_125087 = ((double *) mem_133348)[i_132003 * (int64_t) 27 + i_125085];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_125088 = r_125086 + lifted_lambda_res_125087;
                double r_tmp_134695 = zp_res_125088;
                
                r_125086 = r_tmp_134695;
            }
            defunc_0_lifted_lambda_res_125084 = r_125086;
            // futhark/microgpt.fut:319:115-140
            
            double zt_res_125096 = defunc_0_lifted_lambda_res_125084 * defunc_0_lifted_lambda_res_125084;
            
            // futhark/microgpt.fut:319:106-140
            
            double zs_res_125097 = 1.0 / zt_res_125096;
            double x_131564;
            double redout_131993 = -INFINITY;
            
            for (int64_t i_131994 = 0; i_131994 < (int64_t) 27; i_131994++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129724 = ((double *) mem_133318)[i_132003 * (int64_t) 27 + i_131994];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_125113 = fmax64(lifted_lambda_res_129724, redout_131993);
                double redout_tmp_134696 = max_res_125113;
                
                redout_131993 = redout_tmp_134696;
            }
            x_131564 = redout_131993;
            // futhark/microgpt.fut:317:67-76
            
            double neg_res_125114 = -x_131564;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_125098;
            double r_125100 = 0.0;
            
            for (int64_t i_125099 = 0; i_125099 < (int64_t) 27; i_125099++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_131997 = 0; i_131997 < (int64_t) 27; i_131997++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_125121 = ((double *) mem_133318)[i_132003 * (int64_t) 27 + i_131997];
                    
                    // futhark/microgpt.fut:317:44-76
                    
                    double zp_res_125122 = neg_res_125114 + zp_lhs_125121;
                    
                    // futhark/microgpt.fut:317:37-76
                    
                    double exp_res_125123 = futrts_exp64(zp_res_125122);
                    
                    ((double *) mem_133372)[i_131997] = exp_res_125123;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_125125;
                double r_125127 = 0.0;
                
                for (int64_t i_125126 = 0; i_125126 < (int64_t) 27; i_125126++) {
                    // futhark/microgpt.fut:318:36-46
                    
                    double lifted_lambda_res_125128 = ((double *) mem_133372)[i_125126];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_125129 = r_125127 + lifted_lambda_res_125128;
                    double r_tmp_134699 = zp_res_125129;
                    
                    r_125127 = r_tmp_134699;
                }
                defunc_0_lifted_lambda_res_125125 = r_125127;
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_125130 = ((double *) mem_132775)[i_132003 * (int64_t) 27 + i_125099];
                
                // futhark/microgpt.fut:319:46-56
                
                double zt_lhs_125131 = ((double *) mem_133372)[i_125099];
                
                // futhark/microgpt.fut:319:62-73
                
                double zs_res_125132 = 1.0 / defunc_0_lifted_lambda_res_125125;
                
                // futhark/microgpt.fut:319:46-73
                
                double zt_res_125133 = zt_lhs_125131 * zs_res_125132;
                
                // futhark/microgpt.fut:319:37-73
                
                double zs_res_125134 = 1.0 / zt_res_125133;
                
                // futhark/microgpt.fut:319:14-73
                
                double zt_res_125135 = zt_lhs_125130 * zs_res_125134;
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_125136 = ((double *) mem_133348)[i_132003 * (int64_t) 27 + i_125099];
                
                // futhark/microgpt.fut:319:32-99
                
                double zt_res_125137 = zt_res_125135 * zt_rhs_125136;
                
                // futhark/microgpt.fut:319:78-140
                
                double zt_res_125138 = zs_res_125097 * zt_res_125137;
                
                // futhark/microgpt.fut:319:5-140
                
                double neg_res_125139 = -zt_res_125138;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_125140 = r_125100 + neg_res_125139;
                double r_tmp_134697 = zp_res_125140;
                
                r_125100 = r_tmp_134697;
            }
            defunc_0_lifted_lambda_res_125098 = r_125100;
            ((double *) mem_133364)[i_132003] = defunc_0_lifted_lambda_res_125098;
            ((double *) mem_133365)[i_132003] = defunc_0_lifted_lambda_res_125084;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132018 = 0; i_132018 < (int64_t) 16; i_132018++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_131565;
            double redout_132006 = -INFINITY;
            
            for (int64_t i_132007 = 0; i_132007 < (int64_t) 27; i_132007++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129748 = ((double *) mem_133318)[i_132018 * (int64_t) 27 + i_132007];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_122861 = fmax64(lifted_lambda_res_129748, redout_132006);
                double redout_tmp_134701 = max_res_122861;
                
                redout_132006 = redout_tmp_134701;
            }
            defunc_0_reduce_res_131565 = redout_132006;
            // futhark/microgpt.fut:321:69-78
            
            double neg_res_122862 = -defunc_0_reduce_res_131565;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132010 = 0; i_132010 < (int64_t) 27; i_132010++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_122869 = ((double *) mem_133318)[i_132018 * (int64_t) 27 + i_132010];
                
                // futhark/microgpt.fut:321:45-78
                
                double zp_res_122870 = neg_res_122862 + zp_lhs_122869;
                
                // futhark/microgpt.fut:321:38-78
                
                double exp_res_122871 = futrts_exp64(zp_res_122870);
                
                ((double *) mem_133390)[i_132010] = exp_res_122871;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_122873;
            double r_122875 = 0.0;
            
            for (int64_t i_122874 = 0; i_122874 < (int64_t) 27; i_122874++) {
                // futhark/microgpt.fut:322:38-49
                
                double lifted_lambda_res_122876 = ((double *) mem_133390)[i_122874];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_122877 = r_122875 + lifted_lambda_res_122876;
                double r_tmp_134703 = zp_res_122877;
                
                r_122875 = r_tmp_134703;
            }
            defunc_0_lifted_lambda_res_122873 = r_122875;
            // futhark/microgpt.fut:323:55-67
            
            double zs_res_122878 = 1.0 / defunc_0_lifted_lambda_res_122873;
            
            // futhark/microgpt.fut:323:85-95
            
            double zs_rhs_122879 = ((double *) mem_133365)[i_132018];
            
            // futhark/microgpt.fut:323:77-95
            
            double zs_res_122880 = 1.0 / zs_rhs_122879;
            
            // futhark/microgpt.fut:323:102-112
            
            double zp_rhs_122881 = ((double *) mem_133364)[i_132018];
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132014 = 0; i_132014 < (int64_t) 27; i_132014++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_122888 = ((double *) mem_132775)[i_132018 * (int64_t) 27 + i_132014];
                
                // futhark/microgpt.fut:323:39-49
                
                double zt_lhs_122889 = ((double *) mem_133390)[i_132014];
                
                // futhark/microgpt.fut:323:39-67
                
                double zt_res_122890 = zs_res_122878 * zt_lhs_122889;
                
                // futhark/microgpt.fut:323:30-67
                
                double zs_res_122891 = 1.0 / zt_res_122890;
                
                // futhark/microgpt.fut:323:7-67
                
                double zt_res_122892 = zt_lhs_122888 * zs_res_122891;
                
                // futhark/microgpt.fut:323:25-95
                
                double zt_res_122893 = zs_res_122880 * zt_res_122892;
                
                // futhark/microgpt.fut:323:72-112
                
                double zp_res_122894 = zp_rhs_122881 + zt_res_122893;
                
                ((double *) mem_133397)[i_132014] = zp_res_122894;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133385, i_132018 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133397, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132022 = 0; i_132022 < (int64_t) 16; i_132022++) {
            double eta_p_elem_122899 = ((double *) mem_133335)[i_132022];
            
            // futhark/microgpt.fut:324:100-117
            
            double neg_res_122904 = -eta_p_elem_122899;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_122905;
            double r_122907 = 0.0;
            
            for (int64_t i_122906 = 0; i_122906 < (int64_t) 27; i_122906++) {
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_122908 = ((double *) mem_133318)[i_132022 * (int64_t) 27 + i_122906];
                
                // futhark/microgpt.fut:324:75-117
                
                double zp_res_122909 = neg_res_122904 + zp_lhs_122908;
                
                // futhark/microgpt.fut:324:68-117
                
                double exp_res_122910 = futrts_exp64(zp_res_122909);
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_122911 = ((double *) mem_133385)[i_132022 * (int64_t) 27 + i_122906];
                
                // futhark/microgpt.fut:324:68-144
                
                double zt_res_122912 = exp_res_122910 * zt_rhs_122911;
                
                // futhark/microgpt.fut:324:60-144
                
                double neg_res_122913 = -zt_res_122912;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_122914 = r_122907 + neg_res_122913;
                double r_tmp_134706 = zp_res_122914;
                
                r_122907 = r_tmp_134706;
            }
            defunc_0_lifted_lambda_res_122905 = r_122907;
            ((double *) mem_133408)[i_132022] = defunc_0_lifted_lambda_res_122905;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132030 = 0; i_132030 < (int64_t) 16; i_132030++) {
            // futhark/microgpt.fut:326:73-83
            
            double neg_arg0_122943 = ((double *) mem_133335)[i_132030];
            
            // futhark/microgpt.fut:326:67-83
            
            double neg_res_122944 = -neg_arg0_122943;
            
            // futhark/microgpt.fut:326:116-126
            
            double zt_lhs_122945 = ((double *) mem_133408)[i_132030];
            
            // futhark/microgpt.fut:326:218-228
            
            double zt_rhs_122946 = ((double *) mem_133334)[i_132030];
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132026 = 0; i_132026 < (int64_t) 27; i_132026++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_122953 = ((double *) mem_133318)[i_132030 * (int64_t) 27 + i_132026];
                
                // futhark/microgpt.fut:326:44-83
                
                double zp_res_122954 = neg_res_122944 + zp_lhs_122953;
                
                // futhark/microgpt.fut:326:37-83
                
                double exp_res_122955 = futrts_exp64(zp_res_122954);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_122956 = ((double *) mem_133385)[i_132030 * (int64_t) 27 + i_132026];
                
                // futhark/microgpt.fut:326:37-108
                
                double zt_res_122957 = exp_res_122955 * zt_rhs_122956;
                
                // futhark/microgpt.fut:326:160-206
                
                double neg_res_122958 = -zp_res_122954;
                
                // futhark/microgpt.fut:110:42-54
                
                double max_res_122959 = fmax64(0.0, neg_res_122958);
                
                // futhark/microgpt.fut:110:35-54
                
                double sgn_res_122960 = fsignum64(max_res_122959);
                
                // futhark/microgpt.fut:326:141-209
                
                double neg_res_122961 = -sgn_res_122960;
                
                // futhark/microgpt.fut:326:132-210
                
                double zp_res_122962 = 1.0 + neg_res_122961;
                
                // futhark/microgpt.fut:326:116-210
                
                double zt_res_122963 = zt_lhs_122945 * zp_res_122962;
                
                // futhark/microgpt.fut:326:127-228
                
                double zt_res_122964 = zt_rhs_122946 * zt_res_122963;
                
                // futhark/microgpt.fut:326:87-228
                
                double zp_res_122965 = zt_res_122957 + zt_res_122964;
                
                ((double *) mem_133420)[i_132026] = zp_res_122965;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133415, i_132030 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133420, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132038 = 0; i_132038 < (int64_t) 16; i_132038++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132034 = 0; i_132034 < (int64_t) 16; i_132034++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_122980;
                double r_122982 = 0.0;
                
                for (int64_t i_122981 = 0; i_122981 < (int64_t) 27; i_122981++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_122983 = ((double *) mem_133415)[i_132038 * (int64_t) 27 + i_122981];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_122984 = ((double *) mem_param_132670.mem)[i_122981 * (int64_t) 16 + i_132034];
                    
                    // futhark/microgpt.fut:327:67-111
                    
                    double zt_res_122985 = zt_lhs_122983 * zt_rhs_122984;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_122986 = r_122982 + zt_res_122985;
                    double r_tmp_134711 = zp_res_122986;
                    
                    r_122982 = r_tmp_134711;
                }
                defunc_0_lifted_lambda_res_122980 = r_122982;
                ((double *) mem_133436)[i_132034] = defunc_0_lifted_lambda_res_122980;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133431, i_132038 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133436, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132051 = 0; i_132051 < (int64_t) 16; i_132051++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132044 = 0; i_132044 < (int64_t) 64; i_132044++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_129773;
                double r_129775 = 0.0;
                
                for (int64_t i_129774 = 0; i_129774 < (int64_t) 16; i_129774++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_129776 = ((double *) mem_133431)[i_132051 * (int64_t) 16 + i_129774];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_129777 = ((double *) mem_param_132638.mem)[i_129774 * (int64_t) 64 + i_132044];
                    
                    // futhark/microgpt.fut:328:67-113
                    
                    double zt_res_129778 = zt_lhs_129776 * zt_rhs_129777;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_129779 = r_129775 + zt_res_129778;
                    double r_tmp_134716 = zp_res_129779;
                    
                    r_129775 = r_tmp_134716;
                }
                defunc_0_lifted_lambda_res_129773 = r_129775;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_129786;
                double r_129788 = 0.0;
                
                for (int64_t i_129787 = 0; i_129787 < (int64_t) 16; i_129787++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_129789 = ((double *) mem_133431)[i_129787 * (int64_t) 16 + i_132051];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_129790 = ((double *) mem_133270)[i_129787 * (int64_t) 64 + i_132044];
                    
                    // futhark/microgpt.fut:397:69-113
                    
                    double zt_res_129791 = zt_lhs_129789 * zt_rhs_129790;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_129792 = r_129788 + zt_res_129791;
                    double r_tmp_134717 = zp_res_129792;
                    
                    r_129788 = r_tmp_134717;
                }
                defunc_0_lifted_lambda_res_129786 = r_129788;
                ((double *) mem_133457)[i_132044] = defunc_0_lifted_lambda_res_129786;
                ((double *) mem_133458)[i_132044] = defunc_0_lifted_lambda_res_129773;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133447, i_132051 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133457, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133448, i_132051 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133458, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132060 = 0; i_132060 < (int64_t) 16; i_132060++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132056 = 0; i_132056 < (int64_t) 64; i_132056++) {
                // futhark/microgpt.fut:4:11-25
                
                double indicatorp_arg0_123022 = ((double *) mem_133248)[i_132060 * (int64_t) 64 + i_132056];
                
                // futhark/microgpt.fut:110:42-54
                
                double max_res_123023 = fmax64(0.0, indicatorp_arg0_123022);
                
                // futhark/microgpt.fut:110:35-54
                
                double sgn_res_123024 = fsignum64(max_res_123023);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_123025 = ((double *) mem_133448)[i_132060 * (int64_t) 64 + i_132056];
                
                // futhark/microgpt.fut:329:46-102
                
                double zt_res_123026 = sgn_res_123024 * zt_rhs_123025;
                
                ((double *) mem_133484)[i_132056] = zt_res_123026;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133479, i_132060 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133484, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132068 = 0; i_132068 < (int64_t) 16; i_132068++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132064 = 0; i_132064 < (int64_t) 16; i_132064++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_123041;
                double r_123043 = 0.0;
                
                for (int64_t i_123042 = 0; i_123042 < (int64_t) 64; i_123042++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_123044 = ((double *) mem_133479)[i_132068 * (int64_t) 64 + i_123042];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_123045 = ((double *) mem_param_132662.mem)[i_123042 * (int64_t) 16 + i_132064];
                    
                    // futhark/microgpt.fut:330:67-111
                    
                    double zt_res_123046 = zt_lhs_123044 * zt_rhs_123045;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_123047 = r_123043 + zt_res_123046;
                    double r_tmp_134722 = zp_res_123047;
                    
                    r_123043 = r_tmp_134722;
                }
                defunc_0_lifted_lambda_res_123041 = r_123043;
                ((double *) mem_133500)[i_132064] = defunc_0_lifted_lambda_res_123041;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133495, i_132068 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133500, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132074 = 0; i_132074 < (int64_t) 16; i_132074++) {
            // futhark/microgpt.fut:333:47-59
            
            double zp_lhs_125046 = ((double *) mem_133247)[i_132074];
            
            // futhark/microgpt.fut:333:47-87
            
            double zp_res_125047 = 1.0e-5 + zp_lhs_125046;
            
            // futhark/microgpt.fut:333:39-87
            
            double sqrt_res_125048 = futrts_sqrt64(zp_res_125047);
            
            // futhark/microgpt.fut:334:129-158
            
            double zt_res_125056 = sqrt_res_125048 * sqrt_res_125048;
            
            // futhark/microgpt.fut:334:120-158
            
            double zs_res_125057 = 1.0 / zt_res_125056;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_125058;
            double r_125060 = 0.0;
            
            for (int64_t i_125059 = 0; i_125059 < (int64_t) 16; i_125059++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_125061 = ((double *) mem_133495)[i_132074 * (int64_t) 16 + i_125059];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_125062 = ((double *) mem_133192)[i_132074 * (int64_t) 16 + i_125059];
                
                // futhark/microgpt.fut:334:69-113
                
                double zt_res_125063 = zt_lhs_125061 * zt_rhs_125062;
                
                // futhark/microgpt.fut:334:90-158
                
                double zt_res_125064 = zs_res_125057 * zt_res_125063;
                
                // futhark/microgpt.fut:334:61-158
                
                double neg_res_125065 = -zt_res_125064;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_125066 = r_125060 + neg_res_125065;
                double r_tmp_134725 = zp_res_125066;
                
                r_125060 = r_tmp_134725;
            }
            defunc_0_lifted_lambda_res_125058 = r_125060;
            ((double *) mem_133511)[i_132074] = defunc_0_lifted_lambda_res_125058;
            ((double *) mem_133512)[i_132074] = sqrt_res_125048;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132079 = 0; i_132079 < (int64_t) 16; i_132079++) {
            // futhark/microgpt.fut:335:39-51
            
            double zt_lhs_123114 = ((double *) mem_133511)[i_132079];
            
            // futhark/microgpt.fut:335:93-105
            
            double zp_lhs_123115 = ((double *) mem_133247)[i_132079];
            
            // futhark/microgpt.fut:335:93-133
            
            double zp_res_123116 = 1.0e-5 + zp_lhs_123115;
            
            // futhark/microgpt.fut:335:85-133
            
            double sqrt_res_123117 = futrts_sqrt64(zp_res_123116);
            
            // futhark/microgpt.fut:335:71-135
            
            double zt_res_123118 = 2.0 * sqrt_res_123117;
            
            // futhark/microgpt.fut:335:57-135
            
            double zs_res_123119 = 1.0 / zt_res_123118;
            
            // futhark/microgpt.fut:335:39-135
            
            double zt_res_123120 = zt_lhs_123114 * zs_res_123119;
            
            ((double *) mem_133525)[i_132079] = zt_res_123120;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132083 = 0; i_132083 < (int64_t) 16; i_132083++) {
            // futhark/microgpt.fut:336:49-61
            
            double zs_lhs_123128 = ((double *) mem_133525)[i_132083];
            
            // futhark/microgpt.fut:336:49-76
            
            double zs_res_123129 = zs_lhs_123128 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_134728 = 0; nest_i_134728 < (int64_t) 16; nest_i_134728++) {
                ((double *) mem_133532)[i_132083 * (int64_t) 16 + nest_i_134728] = zs_res_123129;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132091 = 0; i_132091 < (int64_t) 16; i_132091++) {
            // futhark/microgpt.fut:337:99-111
            
            double zs_rhs_123138 = ((double *) mem_133512)[i_132091];
            
            // futhark/microgpt.fut:337:91-111
            
            double zs_res_123139 = 1.0 / zs_rhs_123138;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132087 = 0; i_132087 < (int64_t) 16; i_132087++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_123146 = ((double *) mem_133431)[i_132091 * (int64_t) 16 + i_132087];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_123147 = ((double *) mem_133495)[i_132091 * (int64_t) 16 + i_132087];
                
                // futhark/microgpt.fut:337:65-111
                
                double zt_res_123148 = zs_res_123139 * zt_lhs_123147;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_123149 = ((double *) mem_133532)[i_132091 * (int64_t) 16 + i_132087];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_123150 = ((double *) mem_133192)[i_132091 * (int64_t) 16 + i_132087];
                
                // futhark/microgpt.fut:337:119-163
                
                double zt_res_123151 = zt_lhs_123149 * zt_rhs_123150;
                
                // futhark/microgpt.fut:337:86-163
                
                double zp_res_123152 = zt_res_123148 + zt_res_123151;
                
                // futhark/microgpt.fut:337:114-215
                
                double zp_res_123153 = zt_res_123151 + zp_res_123152;
                
                // futhark/microgpt.fut:337:37-215
                
                double zp_res_123154 = zp_lhs_123146 + zp_res_123153;
                
                ((double *) mem_133547)[i_132087] = zp_res_123154;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133542, i_132091 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133547, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132104 = 0; i_132104 < (int64_t) 16; i_132104++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132097 = 0; i_132097 < (int64_t) 16; i_132097++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_129815;
                double r_129817 = 0.0;
                
                for (int64_t i_129816 = 0; i_129816 < (int64_t) 16; i_129816++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_129818 = ((double *) mem_133542)[i_132104 * (int64_t) 16 + i_129816];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_129819 = ((double *) mem_param_132646.mem)[i_129816 * (int64_t) 16 + i_132097];
                    
                    // futhark/microgpt.fut:338:67-112
                    
                    double zt_res_129820 = zt_lhs_129818 * zt_rhs_129819;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_129821 = r_129817 + zt_res_129820;
                    double r_tmp_134735 = zp_res_129821;
                    
                    r_129817 = r_tmp_134735;
                }
                defunc_0_lifted_lambda_res_129815 = r_129817;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_129828;
                double r_129830 = 0.0;
                
                for (int64_t i_129829 = 0; i_129829 < (int64_t) 16; i_129829++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_129831 = ((double *) mem_133542)[i_129829 * (int64_t) 16 + i_132104];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_129832 = ((double *) mem_133160)[i_129829 * (int64_t) 16 + i_132097];
                    
                    // futhark/microgpt.fut:395:68-112
                    
                    double zt_res_129833 = zt_lhs_129831 * zt_rhs_129832;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_129834 = r_129830 + zt_res_129833;
                    double r_tmp_134736 = zp_res_129834;
                    
                    r_129830 = r_tmp_134736;
                }
                defunc_0_lifted_lambda_res_129828 = r_129830;
                ((double *) mem_133568)[i_132097] = defunc_0_lifted_lambda_res_129828;
                ((double *) mem_133569)[i_132097] = defunc_0_lifted_lambda_res_129815;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133558, i_132104 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133568, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133559, i_132104 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133569, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132134 = 0; i_132134 < (int64_t) 4; i_132134++) {
            // futhark/microgpt.fut:339:74-77
            
            int64_t zp_lhs_126876 = mul64((int64_t) 4, i_132134);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132124 = 0; i_132124 < (int64_t) 16; i_132124++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_132109 = 0; i_132109 < (int64_t) 4; i_132109++) {
                    // futhark/microgpt.fut:339:79-87
                    
                    int64_t tmp_129915 = add64(zp_lhs_126876, i_132109);
                    
                    // futhark/microgpt.fut:339:52-89
                    
                    bool x_129916 = sle64((int64_t) 0, tmp_129915);
                    
                    // futhark/microgpt.fut:339:52-89
                    
                    bool y_129917 = slt64(tmp_129915, (int64_t) 16);
                    
                    // futhark/microgpt.fut:339:52-89
                    
                    bool bounds_check_129918 = x_129916 && y_129917;
                    
                    // futhark/microgpt.fut:339:52-89
                    
                    bool index_certs_129919;
                    
                    if (!bounds_check_129918) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_129915, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:339:52-89\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:339:13-90\n   #9  futhark/microgpt.fut:568:5-76\n   #10 futhark/microgpt.fut:585:26-591:31\n   #11 futhark/microgpt.fut:619:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_129920 = ((double *) mem_133559)[i_132124 * (int64_t) 16 + tmp_129915];
                    
                    ((double *) mem_133623)[i_132109] = lifted_lambda_res_129920;
                }
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_129928 = ((double *) mem_133030)[i_132134 * (int64_t) 16 + i_132124];
                
                // futhark/microgpt.fut:347:198-224
                
                double neg_res_129929 = -neg_arg0_129928;
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_129958 = ((double *) mem_133028)[i_132134 * (int64_t) 16 + i_132124];
                
                // futhark/microgpt.fut:356:198-224
                
                double neg_res_129959 = -neg_arg0_129958;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_132115 = 0; i_132115 < (int64_t) 16; i_132115++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_129997;
                    double r_129999 = 0.0;
                    
                    for (int64_t i_129998 = 0; i_129998 < (int64_t) 4; i_129998++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_130000 = ((double *) mem_132949)[i_132134 * (int64_t) 64 + i_132124 * (int64_t) 4 + i_129998];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_130001 = ((double *) mem_132948)[i_132134 * (int64_t) 64 + i_132115 * (int64_t) 4 + i_129998];
                        
                        // futhark/microgpt.fut:347:89-148
                        
                        double zt_res_130002 = zt_lhs_130000 * zt_rhs_130001;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_130003 = r_129999 + zt_res_130002;
                        double r_tmp_134746 = zp_res_130003;
                        
                        r_129999 = r_tmp_134746;
                    }
                    defunc_0_lifted_lambda_res_129997 = r_129999;
                    // futhark/microgpt.fut:347:68-165
                    
                    double zs_res_130004 = defunc_0_lifted_lambda_res_129997 / 2.0;
                    double zp_rhs_130005 = ((double *) masks_mem_132632.mem)[step_122159 * (int64_t) 256 + i_132124 * (int64_t) 16 + i_132115];
                    
                    // futhark/microgpt.fut:347:152-191
                    
                    double zp_res_130006 = zs_res_130004 + zp_rhs_130005;
                    
                    // futhark/microgpt.fut:347:167-224
                    
                    double zp_res_130007 = neg_res_129929 + zp_res_130006;
                    
                    // futhark/microgpt.fut:347:58-224
                    
                    double exp_res_130008 = futrts_exp64(zp_res_130007);
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_130015;
                    double r_130017 = 0.0;
                    
                    for (int64_t i_130016 = 0; i_130016 < (int64_t) 4; i_130016++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_130018 = ((double *) mem_132949)[i_132134 * (int64_t) 64 + i_132124 * (int64_t) 4 + i_130016];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_130019 = ((double *) mem_132948)[i_132134 * (int64_t) 64 + i_132115 * (int64_t) 4 + i_130016];
                        
                        // futhark/microgpt.fut:356:89-148
                        
                        double zt_res_130020 = zt_lhs_130018 * zt_rhs_130019;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_130021 = r_130017 + zt_res_130020;
                        double r_tmp_134747 = zp_res_130021;
                        
                        r_130017 = r_tmp_134747;
                    }
                    defunc_0_lifted_lambda_res_130015 = r_130017;
                    // futhark/microgpt.fut:356:68-165
                    
                    double zs_res_130022 = defunc_0_lifted_lambda_res_130015 / 2.0;
                    
                    // futhark/microgpt.fut:356:152-191
                    
                    double zp_res_130024 = zp_rhs_130005 + zs_res_130022;
                    
                    // futhark/microgpt.fut:356:167-224
                    
                    double zp_res_130025 = neg_res_129959 + zp_res_130024;
                    
                    // futhark/microgpt.fut:356:58-224
                    
                    double exp_res_130026 = futrts_exp64(zp_res_130025);
                    
                    ((double *) mem_133630)[i_132115] = exp_res_130026;
                    ((double *) mem_133631)[i_132115] = exp_res_130008;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_133608, i_132124 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133630, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_133609, i_132124 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133631, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_133610, i_132124 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133623, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_133590, i_132134 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_133608, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_133591, i_132134 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_133609, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_133592, i_132134 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_133610, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132165 = 0; i_132165 < (int64_t) 4; i_132165++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132155 = 0; i_132155 < (int64_t) 16; i_132155++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_132140 = 0; i_132140 < (int64_t) 4; i_132140++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_130116;
                    double r_130118 = 0.0;
                    
                    for (int64_t i_130117 = 0; i_130117 < (int64_t) 16; i_130117++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_130119 = ((double *) mem_133592)[i_132165 * (int64_t) 64 + i_130117 * (int64_t) 4 + i_132140];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_130120 = ((double *) mem_133031)[i_132165 * (int64_t) 256 + i_130117 * (int64_t) 16 + i_132155];
                        
                        // futhark/microgpt.fut:344:67-128
                        
                        double zt_res_130121 = zt_lhs_130119 * zt_rhs_130120;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_130122 = r_130118 + zt_res_130121;
                        double r_tmp_134755 = zp_res_130122;
                        
                        r_130118 = r_tmp_134755;
                    }
                    defunc_0_lifted_lambda_res_130116 = r_130118;
                    ((double *) mem_133704)[i_132140] = defunc_0_lifted_lambda_res_130116;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_132146 = 0; i_132146 < (int64_t) 16; i_132146++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_130185;
                    double r_130187 = 0.0;
                    
                    for (int64_t i_130186 = 0; i_130186 < (int64_t) 4; i_130186++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_130188 = ((double *) mem_133592)[i_132165 * (int64_t) 64 + i_132155 * (int64_t) 4 + i_130186];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_130189 = ((double *) mem_132947)[i_132165 * (int64_t) 64 + i_132146 * (int64_t) 4 + i_130186];
                        
                        // futhark/microgpt.fut:345:87-147
                        
                        double zt_res_130190 = zt_lhs_130188 * zt_rhs_130189;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_130191 = r_130187 + zt_res_130190;
                        double r_tmp_134758 = zp_res_130191;
                        
                        r_130187 = r_tmp_134758;
                    }
                    defunc_0_lifted_lambda_res_130185 = r_130187;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_130198;
                    double r_130200 = 0.0;
                    
                    for (int64_t i_130199 = 0; i_130199 < (int64_t) 4; i_130199++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_130201 = ((double *) mem_133592)[i_132165 * (int64_t) 64 + i_132155 * (int64_t) 4 + i_130199];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_130202 = ((double *) mem_132947)[i_132165 * (int64_t) 64 + i_132146 * (int64_t) 4 + i_130199];
                        
                        // futhark/microgpt.fut:354:87-147
                        
                        double zt_res_130203 = zt_lhs_130201 * zt_rhs_130202;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_130204 = r_130200 + zt_res_130203;
                        double r_tmp_134759 = zp_res_130204;
                        
                        r_130200 = r_tmp_134759;
                    }
                    defunc_0_lifted_lambda_res_130198 = r_130200;
                    ((double *) mem_133711)[i_132146] = defunc_0_lifted_lambda_res_130198;
                    ((double *) mem_133712)[i_132146] = defunc_0_lifted_lambda_res_130185;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_133689, i_132155 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133711, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_133690, i_132155 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133712, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_133691, i_132155 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133704, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_133671, i_132165 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_133689, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_133672, i_132165 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_133690, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_133673, i_132165 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_133691, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132190 = 0; i_132190 < (int64_t) 4; i_132190++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132177 = 0; i_132177 < (int64_t) 16; i_132177++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_130320;
                double r_130322 = 0.0;
                
                for (int64_t i_130321 = 0; i_130321 < (int64_t) 16; i_130321++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_130323 = ((double *) mem_133591)[i_132190 * (int64_t) 256 + i_132177 * (int64_t) 16 + i_130321];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_130324 = r_130322 + lifted_lambda_res_130323;
                    double r_tmp_134768 = zp_res_130324;
                    
                    r_130322 = r_tmp_134768;
                }
                defunc_0_lifted_lambda_res_130320 = r_130322;
                // futhark/microgpt.fut:349:155-200
                
                double zt_res_130332 = defunc_0_lifted_lambda_res_130320 * defunc_0_lifted_lambda_res_130320;
                
                // futhark/microgpt.fut:349:146-200
                
                double zs_res_130333 = 1.0 / zt_res_130332;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_130334;
                double r_130336 = 0.0;
                
                for (int64_t i_130335 = 0; i_130335 < (int64_t) 16; i_130335++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_130337 = ((double *) mem_133672)[i_132190 * (int64_t) 256 + i_132177 * (int64_t) 16 + i_130335];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_130338 = ((double *) mem_133591)[i_132190 * (int64_t) 256 + i_132177 * (int64_t) 16 + i_130335];
                    
                    // futhark/microgpt.fut:349:78-139
                    
                    double zt_res_130339 = zt_lhs_130337 * zt_rhs_130338;
                    
                    // futhark/microgpt.fut:349:107-200
                    
                    double zt_res_130340 = zs_res_130333 * zt_res_130339;
                    
                    // futhark/microgpt.fut:349:70-200
                    
                    double neg_res_130341 = -zt_res_130340;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_130342 = r_130336 + neg_res_130341;
                    double r_tmp_134769 = zp_res_130342;
                    
                    r_130336 = r_tmp_134769;
                }
                defunc_0_lifted_lambda_res_130334 = r_130336;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_130353;
                double r_130355 = 0.0;
                
                for (int64_t i_130354 = 0; i_130354 < (int64_t) 16; i_130354++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_130356 = ((double *) mem_133590)[i_132190 * (int64_t) 256 + i_132177 * (int64_t) 16 + i_130354];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_130357 = r_130355 + lifted_lambda_res_130356;
                    double r_tmp_134770 = zp_res_130357;
                    
                    r_130355 = r_tmp_134770;
                }
                defunc_0_lifted_lambda_res_130353 = r_130355;
                // futhark/microgpt.fut:358:155-200
                
                double zt_res_130365 = defunc_0_lifted_lambda_res_130353 * defunc_0_lifted_lambda_res_130353;
                
                // futhark/microgpt.fut:358:146-200
                
                double zs_res_130366 = 1.0 / zt_res_130365;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_130367;
                double r_130369 = 0.0;
                
                for (int64_t i_130368 = 0; i_130368 < (int64_t) 16; i_130368++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_130370 = ((double *) mem_133671)[i_132190 * (int64_t) 256 + i_132177 * (int64_t) 16 + i_130368];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_130371 = ((double *) mem_133590)[i_132190 * (int64_t) 256 + i_132177 * (int64_t) 16 + i_130368];
                    
                    // futhark/microgpt.fut:358:78-139
                    
                    double zt_res_130372 = zt_lhs_130370 * zt_rhs_130371;
                    
                    // futhark/microgpt.fut:358:107-200
                    
                    double zt_res_130373 = zs_res_130366 * zt_res_130372;
                    
                    // futhark/microgpt.fut:358:70-200
                    
                    double neg_res_130374 = -zt_res_130373;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_130375 = r_130369 + neg_res_130374;
                    double r_tmp_134771 = zp_res_130375;
                    
                    r_130369 = r_tmp_134771;
                }
                defunc_0_lifted_lambda_res_130367 = r_130369;
                ((double *) mem_133772)[i_132177] = defunc_0_lifted_lambda_res_130367;
                ((double *) mem_133773)[i_132177] = defunc_0_lifted_lambda_res_130353;
                ((double *) mem_133774)[i_132177] = defunc_0_lifted_lambda_res_130334;
                ((double *) mem_133775)[i_132177] = defunc_0_lifted_lambda_res_130320;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133752, i_132190 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133772, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133753, i_132190 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133773, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133754, i_132190 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133774, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133755, i_132190 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133775, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132213 = 0; i_132213 < (int64_t) 4; i_132213++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132206 = 0; i_132206 < (int64_t) 16; i_132206++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_130401 = ((double *) mem_133755)[i_132213 * (int64_t) 16 + i_132206];
                
                // futhark/microgpt.fut:350:93-121
                
                double zs_res_130402 = 1.0 / zs_rhs_130401;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_130403 = ((double *) mem_133754)[i_132213 * (int64_t) 16 + i_132206];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_130422 = ((double *) mem_133752)[i_132213 * (int64_t) 16 + i_132206];
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_130420 = ((double *) mem_133753)[i_132213 * (int64_t) 16 + i_132206];
                
                // futhark/microgpt.fut:359:93-121
                
                double zs_res_130421 = 1.0 / zs_rhs_130420;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_132199 = 0; i_132199 < (int64_t) 16; i_132199++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_130450 = ((double *) mem_133672)[i_132213 * (int64_t) 256 + i_132206 * (int64_t) 16 + i_132199];
                    
                    // futhark/microgpt.fut:350:59-121
                    
                    double zt_res_130451 = zs_res_130402 * zt_lhs_130450;
                    
                    // futhark/microgpt.fut:350:88-148
                    
                    double zp_res_130452 = zp_rhs_130403 + zt_res_130451;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_130459 = ((double *) mem_133671)[i_132213 * (int64_t) 256 + i_132206 * (int64_t) 16 + i_132199];
                    
                    // futhark/microgpt.fut:359:59-121
                    
                    double zt_res_130460 = zs_res_130421 * zt_lhs_130459;
                    
                    // futhark/microgpt.fut:359:88-148
                    
                    double zp_res_130461 = zp_rhs_130422 + zt_res_130460;
                    
                    ((double *) mem_133838)[i_132199] = zp_res_130461;
                    ((double *) mem_133839)[i_132199] = zp_res_130452;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_133828, i_132206 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133838, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_133829, i_132206 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133839, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_133816, i_132213 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_133828, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_133817, i_132213 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_133829, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132231 = 0; i_132231 < (int64_t) 4; i_132231++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132224 = 0; i_132224 < (int64_t) 16; i_132224++) {
                double f_elem_130737 = ((double *) mem_133030)[i_132231 * (int64_t) 16 + i_132224];
                double f_elem_130739 = ((double *) mem_133028)[i_132231 * (int64_t) 16 + i_132224];
                
                // futhark/microgpt.fut:351:218-244
                
                double neg_res_130744 = -f_elem_130737;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_130745;
                double r_130747 = 0.0;
                
                for (int64_t i_130746 = 0; i_130746 < (int64_t) 16; i_130746++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_130748;
                    double r_130750 = 0.0;
                    
                    for (int64_t i_130749 = 0; i_130749 < (int64_t) 4; i_130749++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_130751 = ((double *) mem_132949)[i_132231 * (int64_t) 64 + i_132224 * (int64_t) 4 + i_130749];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_130752 = ((double *) mem_132948)[i_132231 * (int64_t) 64 + i_130746 * (int64_t) 4 + i_130749];
                        
                        // futhark/microgpt.fut:351:109-168
                        
                        double zt_res_130753 = zt_lhs_130751 * zt_rhs_130752;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_130754 = r_130750 + zt_res_130753;
                        double r_tmp_134783 = zp_res_130754;
                        
                        r_130750 = r_tmp_134783;
                    }
                    defunc_0_lifted_lambda_res_130748 = r_130750;
                    // futhark/microgpt.fut:351:88-185
                    
                    double zs_res_130755 = defunc_0_lifted_lambda_res_130748 / 2.0;
                    double zp_rhs_130756 = ((double *) masks_mem_132632.mem)[step_122159 * (int64_t) 256 + i_132224 * (int64_t) 16 + i_130746];
                    
                    // futhark/microgpt.fut:351:172-211
                    
                    double zp_res_130757 = zs_res_130755 + zp_rhs_130756;
                    
                    // futhark/microgpt.fut:351:187-244
                    
                    double zp_res_130758 = neg_res_130744 + zp_res_130757;
                    
                    // futhark/microgpt.fut:351:78-244
                    
                    double exp_res_130759 = futrts_exp64(zp_res_130758);
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_130760 = ((double *) mem_133817)[i_132231 * (int64_t) 256 + i_132224 * (int64_t) 16 + i_130746];
                    
                    // futhark/microgpt.fut:351:78-280
                    
                    double zt_res_130761 = exp_res_130759 * zt_rhs_130760;
                    
                    // futhark/microgpt.fut:351:70-280
                    
                    double neg_res_130762 = -zt_res_130761;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_130763 = r_130747 + neg_res_130762;
                    double r_tmp_134782 = zp_res_130763;
                    
                    r_130747 = r_tmp_134782;
                }
                defunc_0_lifted_lambda_res_130745 = r_130747;
                // futhark/microgpt.fut:360:218-244
                
                double neg_res_130773 = -f_elem_130739;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_130774;
                double r_130776 = 0.0;
                
                for (int64_t i_130775 = 0; i_130775 < (int64_t) 16; i_130775++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_130777;
                    double r_130779 = 0.0;
                    
                    for (int64_t i_130778 = 0; i_130778 < (int64_t) 4; i_130778++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_130780 = ((double *) mem_132949)[i_132231 * (int64_t) 64 + i_132224 * (int64_t) 4 + i_130778];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_130781 = ((double *) mem_132948)[i_132231 * (int64_t) 64 + i_130775 * (int64_t) 4 + i_130778];
                        
                        // futhark/microgpt.fut:360:109-168
                        
                        double zt_res_130782 = zt_lhs_130780 * zt_rhs_130781;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_130783 = r_130779 + zt_res_130782;
                        double r_tmp_134785 = zp_res_130783;
                        
                        r_130779 = r_tmp_134785;
                    }
                    defunc_0_lifted_lambda_res_130777 = r_130779;
                    // futhark/microgpt.fut:360:88-185
                    
                    double zs_res_130784 = defunc_0_lifted_lambda_res_130777 / 2.0;
                    double zp_rhs_130785 = ((double *) masks_mem_132632.mem)[step_122159 * (int64_t) 256 + i_132224 * (int64_t) 16 + i_130775];
                    
                    // futhark/microgpt.fut:360:172-211
                    
                    double zp_res_130786 = zs_res_130784 + zp_rhs_130785;
                    
                    // futhark/microgpt.fut:360:187-244
                    
                    double zp_res_130787 = neg_res_130773 + zp_res_130786;
                    
                    // futhark/microgpt.fut:360:78-244
                    
                    double exp_res_130788 = futrts_exp64(zp_res_130787);
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_130789 = ((double *) mem_133816)[i_132231 * (int64_t) 256 + i_132224 * (int64_t) 16 + i_130775];
                    
                    // futhark/microgpt.fut:360:78-280
                    
                    double zt_res_130790 = exp_res_130788 * zt_rhs_130789;
                    
                    // futhark/microgpt.fut:360:70-280
                    
                    double neg_res_130791 = -zt_res_130790;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_130792 = r_130776 + neg_res_130791;
                    double r_tmp_134784 = zp_res_130792;
                    
                    r_130776 = r_tmp_134784;
                }
                defunc_0_lifted_lambda_res_130774 = r_130776;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_130804;
                double r_130806 = 0.0;
                
                for (int64_t i_130805 = 0; i_130805 < (int64_t) 16; i_130805++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_130807;
                    double r_130809 = 0.0;
                    
                    for (int64_t i_130808 = 0; i_130808 < (int64_t) 4; i_130808++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_130810 = ((double *) mem_132949)[i_132231 * (int64_t) 64 + i_132224 * (int64_t) 4 + i_130808];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_130811 = ((double *) mem_132948)[i_132231 * (int64_t) 64 + i_130805 * (int64_t) 4 + i_130808];
                        
                        // futhark/microgpt.fut:361:138-197
                        
                        double zt_res_130812 = zt_lhs_130810 * zt_rhs_130811;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_130813 = r_130809 + zt_res_130812;
                        double r_tmp_134787 = zp_res_130813;
                        
                        r_130809 = r_tmp_134787;
                    }
                    defunc_0_lifted_lambda_res_130807 = r_130809;
                    // futhark/microgpt.fut:361:117-214
                    
                    double zs_res_130814 = defunc_0_lifted_lambda_res_130807 / 2.0;
                    double zp_rhs_130815 = ((double *) masks_mem_132632.mem)[step_122159 * (int64_t) 256 + i_132224 * (int64_t) 16 + i_130805];
                    
                    // futhark/microgpt.fut:361:201-240
                    
                    double zp_res_130816 = zs_res_130814 + zp_rhs_130815;
                    
                    // futhark/microgpt.fut:361:216-273
                    
                    double zp_res_130817 = neg_res_130773 + zp_res_130816;
                    
                    // futhark/microgpt.fut:361:107-273
                    
                    double neg_res_130818 = -zp_res_130817;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_130819 = fmax64(0.0, neg_res_130818);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_130820 = fsignum64(max_res_130819);
                    
                    // futhark/microgpt.fut:361:88-276
                    
                    double neg_res_130821 = -sgn_res_130820;
                    
                    // futhark/microgpt.fut:361:79-277
                    
                    double zp_res_130822 = 1.0 + neg_res_130821;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_130823 = r_130806 + zp_res_130822;
                    double r_tmp_134786 = zp_res_130823;
                    
                    r_130806 = r_tmp_134786;
                }
                defunc_0_lifted_lambda_res_130804 = r_130806;
                // futhark/microgpt.fut:361:48-280
                
                double zs_res_130824 = 1.0 / defunc_0_lifted_lambda_res_130804;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_132218 = 0; i_132218 < (int64_t) 4; i_132218++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_130841;
                    double r_130843 = 0.0;
                    
                    for (int64_t i_130842 = 0; i_130842 < (int64_t) 16; i_130842++) {
                        // futhark/microgpt.fut:71:13-49
                        
                        double defunc_0_lifted_lambda_res_130844;
                        double r_130846 = 0.0;
                        
                        for (int64_t i_130845 = 0; i_130845 < (int64_t) 4; i_130845++) {
                            // futhark/microgpt.fut:71:46-49
                            
                            double zt_lhs_130847 = ((double *) mem_132949)[i_132231 * (int64_t) 64 + i_132224 * (int64_t) 4 + i_130845];
                            
                            // futhark/microgpt.fut:71:46-49
                            
                            double zt_rhs_130848 = ((double *) mem_132948)[i_132231 * (int64_t) 64 + i_130842 * (int64_t) 4 + i_130845];
                            
                            // futhark/microgpt.fut:362:102-161
                            
                            double zt_res_130849 = zt_lhs_130847 * zt_rhs_130848;
                            
                            // futhark/microgpt.fut:71:40-49
                            
                            double zp_res_130850 = r_130846 + zt_res_130849;
                            double r_tmp_134790 = zp_res_130850;
                            
                            r_130846 = r_tmp_134790;
                        }
                        defunc_0_lifted_lambda_res_130844 = r_130846;
                        // futhark/microgpt.fut:362:81-178
                        
                        double zs_res_130851 = defunc_0_lifted_lambda_res_130844 / 2.0;
                        double zp_rhs_130852 = ((double *) masks_mem_132632.mem)[step_122159 * (int64_t) 256 + i_132224 * (int64_t) 16 + i_130842];
                        
                        // futhark/microgpt.fut:362:165-204
                        
                        double zp_res_130853 = zs_res_130851 + zp_rhs_130852;
                        
                        // futhark/microgpt.fut:362:180-237
                        
                        double zp_res_130854 = neg_res_130773 + zp_res_130853;
                        
                        // futhark/microgpt.fut:362:71-237
                        
                        double exp_res_130855 = futrts_exp64(zp_res_130854);
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_130856 = ((double *) mem_133816)[i_132231 * (int64_t) 256 + i_132224 * (int64_t) 16 + i_130842];
                        
                        // futhark/microgpt.fut:362:71-273
                        
                        double zt_res_130857 = exp_res_130855 * zt_rhs_130856;
                        
                        // futhark/microgpt.fut:362:241-288
                        
                        double zs_res_130858 = zt_res_130857 / 2.0;
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_130859 = ((double *) mem_132948)[i_132231 * (int64_t) 64 + i_130842 * (int64_t) 4 + i_132218];
                        
                        // futhark/microgpt.fut:362:275-321
                        
                        double zt_res_130860 = zs_res_130858 * zt_rhs_130859;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_130861 = r_130843 + zt_res_130860;
                        double r_tmp_134789 = zp_res_130861;
                        
                        r_130843 = r_tmp_134789;
                    }
                    defunc_0_lifted_lambda_res_130841 = r_130843;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_130862;
                    double r_130864 = 0.0;
                    
                    for (int64_t i_130863 = 0; i_130863 < (int64_t) 16; i_130863++) {
                        // futhark/microgpt.fut:71:13-49
                        
                        double defunc_0_lifted_lambda_res_130865;
                        double r_130867 = 0.0;
                        
                        for (int64_t i_130866 = 0; i_130866 < (int64_t) 4; i_130866++) {
                            // futhark/microgpt.fut:71:46-49
                            
                            double zt_lhs_130868 = ((double *) mem_132949)[i_132231 * (int64_t) 64 + i_132224 * (int64_t) 4 + i_130866];
                            
                            // futhark/microgpt.fut:71:46-49
                            
                            double zt_rhs_130869 = ((double *) mem_132948)[i_132231 * (int64_t) 64 + i_130863 * (int64_t) 4 + i_130866];
                            
                            // futhark/microgpt.fut:362:440-499
                            
                            double zt_res_130870 = zt_lhs_130868 * zt_rhs_130869;
                            
                            // futhark/microgpt.fut:71:40-49
                            
                            double zp_res_130871 = r_130867 + zt_res_130870;
                            double r_tmp_134792 = zp_res_130871;
                            
                            r_130867 = r_tmp_134792;
                        }
                        defunc_0_lifted_lambda_res_130865 = r_130867;
                        // futhark/microgpt.fut:362:419-516
                        
                        double zs_res_130872 = defunc_0_lifted_lambda_res_130865 / 2.0;
                        double zp_rhs_130873 = ((double *) masks_mem_132632.mem)[step_122159 * (int64_t) 256 + i_132224 * (int64_t) 16 + i_130863];
                        
                        // futhark/microgpt.fut:362:503-542
                        
                        double zp_res_130874 = zs_res_130872 + zp_rhs_130873;
                        
                        // futhark/microgpt.fut:362:518-575
                        
                        double zp_res_130875 = neg_res_130773 + zp_res_130874;
                        
                        // futhark/microgpt.fut:362:409-575
                        
                        double neg_res_130876 = -zp_res_130875;
                        
                        // futhark/microgpt.fut:110:42-54
                        
                        double max_res_130877 = fmax64(0.0, neg_res_130876);
                        
                        // futhark/microgpt.fut:110:35-54
                        
                        double sgn_res_130878 = fsignum64(max_res_130877);
                        
                        // futhark/microgpt.fut:362:390-578
                        
                        double neg_res_130879 = -sgn_res_130878;
                        
                        // futhark/microgpt.fut:362:381-579
                        
                        double zp_res_130880 = 1.0 + neg_res_130879;
                        
                        // futhark/microgpt.fut:362:355-579
                        
                        double zt_res_130881 = defunc_0_lifted_lambda_res_130774 * zp_res_130880;
                        
                        // futhark/microgpt.fut:362:376-607
                        
                        double zt_res_130882 = zs_res_130824 * zt_res_130881;
                        
                        // futhark/microgpt.fut:362:583-622
                        
                        double zs_res_130883 = zt_res_130882 / 2.0;
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_130884 = ((double *) mem_132948)[i_132231 * (int64_t) 64 + i_130863 * (int64_t) 4 + i_132218];
                        
                        // futhark/microgpt.fut:362:609-655
                        
                        double zt_res_130885 = zs_res_130883 * zt_rhs_130884;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_130886 = r_130864 + zt_res_130885;
                        double r_tmp_134791 = zp_res_130886;
                        
                        r_130864 = r_tmp_134791;
                    }
                    defunc_0_lifted_lambda_res_130862 = r_130864;
                    // futhark/microgpt.fut:362:46-657
                    
                    double zp_res_130887 = defunc_0_lifted_lambda_res_130841 + defunc_0_lifted_lambda_res_130862;
                    
                    ((double *) mem_133890)[i_132218] = zp_res_130887;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_133881, i_132224 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133890, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                ((double *) mem_133882)[i_132224] = defunc_0_lifted_lambda_res_130745;
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_133870, i_132231 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_133881, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133871, i_132231 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133882, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132244 = 0; i_132244 < (int64_t) 4; i_132244++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132240 = 0; i_132240 < (int64_t) 16; i_132240++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_132236 = 0; i_132236 < (int64_t) 4; i_132236++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_123561;
                    double r_123563 = 0.0;
                    
                    for (int64_t i_123562 = 0; i_123562 < (int64_t) 16; i_123562++) {
                        // futhark/microgpt.fut:71:13-49
                        
                        double defunc_0_lifted_lambda_res_123564;
                        double r_123566 = 0.0;
                        
                        for (int64_t i_123565 = 0; i_123565 < (int64_t) 4; i_123565++) {
                            // futhark/microgpt.fut:71:46-49
                            
                            double zt_lhs_123567 = ((double *) mem_132949)[i_132244 * (int64_t) 64 + i_123562 * (int64_t) 4 + i_123565];
                            
                            // futhark/microgpt.fut:71:46-49
                            
                            double zt_rhs_123568 = ((double *) mem_132948)[i_132244 * (int64_t) 64 + i_132240 * (int64_t) 4 + i_123565];
                            
                            // futhark/microgpt.fut:353:102-161
                            
                            double zt_res_123569 = zt_lhs_123567 * zt_rhs_123568;
                            
                            // futhark/microgpt.fut:71:40-49
                            
                            double zp_res_123570 = r_123566 + zt_res_123569;
                            double r_tmp_134797 = zp_res_123570;
                            
                            r_123566 = r_tmp_134797;
                        }
                        defunc_0_lifted_lambda_res_123564 = r_123566;
                        // futhark/microgpt.fut:353:81-178
                        
                        double zs_res_123571 = defunc_0_lifted_lambda_res_123564 / 2.0;
                        double zp_rhs_123572 = ((double *) masks_mem_132632.mem)[step_122159 * (int64_t) 256 + i_123562 * (int64_t) 16 + i_132240];
                        
                        // futhark/microgpt.fut:353:165-204
                        
                        double zp_res_123573 = zs_res_123571 + zp_rhs_123572;
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double neg_arg0_123574 = ((double *) mem_133030)[i_132244 * (int64_t) 16 + i_123562];
                        
                        // futhark/microgpt.fut:353:211-237
                        
                        double neg_res_123575 = -neg_arg0_123574;
                        
                        // futhark/microgpt.fut:353:180-237
                        
                        double zp_res_123576 = zp_res_123573 + neg_res_123575;
                        
                        // futhark/microgpt.fut:353:71-237
                        
                        double exp_res_123577 = futrts_exp64(zp_res_123576);
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_123578 = ((double *) mem_133817)[i_132244 * (int64_t) 256 + i_123562 * (int64_t) 16 + i_132240];
                        
                        // futhark/microgpt.fut:353:71-273
                        
                        double zt_res_123579 = exp_res_123577 * zt_rhs_123578;
                        
                        // futhark/microgpt.fut:353:241-288
                        
                        double zs_res_123580 = zt_res_123579 / 2.0;
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_123581 = ((double *) mem_132949)[i_132244 * (int64_t) 64 + i_123562 * (int64_t) 4 + i_132236];
                        
                        // futhark/microgpt.fut:353:275-321
                        
                        double zt_res_123582 = zs_res_123580 * zt_rhs_123581;
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_123583 = ((double *) mem_133871)[i_132244 * (int64_t) 16 + i_123562];
                        
                        // futhark/microgpt.fut:71:13-49
                        
                        double defunc_0_lifted_lambda_res_123584;
                        double r_123586 = 0.0;
                        
                        for (int64_t i_123585 = 0; i_123585 < (int64_t) 4; i_123585++) {
                            // futhark/microgpt.fut:71:46-49
                            
                            double zt_lhs_123587 = ((double *) mem_132949)[i_132244 * (int64_t) 64 + i_123562 * (int64_t) 4 + i_123585];
                            
                            // futhark/microgpt.fut:71:46-49
                            
                            double zt_rhs_123588 = ((double *) mem_132948)[i_132244 * (int64_t) 64 + i_132240 * (int64_t) 4 + i_123585];
                            
                            // futhark/microgpt.fut:353:416-475
                            
                            double zt_res_123589 = zt_lhs_123587 * zt_rhs_123588;
                            
                            // futhark/microgpt.fut:71:40-49
                            
                            double zp_res_123590 = r_123586 + zt_res_123589;
                            double r_tmp_134798 = zp_res_123590;
                            
                            r_123586 = r_tmp_134798;
                        }
                        defunc_0_lifted_lambda_res_123584 = r_123586;
                        // futhark/microgpt.fut:353:395-492
                        
                        double zs_res_123591 = defunc_0_lifted_lambda_res_123584 / 2.0;
                        
                        // futhark/microgpt.fut:353:479-518
                        
                        double zp_res_123592 = zp_rhs_123572 + zs_res_123591;
                        
                        // futhark/microgpt.fut:353:494-551
                        
                        double zp_res_123593 = neg_res_123575 + zp_res_123592;
                        
                        // futhark/microgpt.fut:353:385-551
                        
                        double neg_res_123594 = -zp_res_123593;
                        
                        // futhark/microgpt.fut:110:42-54
                        
                        double max_res_123595 = fmax64(0.0, neg_res_123594);
                        
                        // futhark/microgpt.fut:110:35-54
                        
                        double sgn_res_123596 = fsignum64(max_res_123595);
                        
                        // futhark/microgpt.fut:353:366-554
                        
                        double neg_res_123597 = -sgn_res_123596;
                        
                        // futhark/microgpt.fut:353:357-555
                        
                        double zp_res_123598 = 1.0 + neg_res_123597;
                        
                        // futhark/microgpt.fut:353:331-555
                        
                        double zt_res_123599 = zt_lhs_123583 * zp_res_123598;
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_123600 = ((double *) mem_133029)[i_132244 * (int64_t) 16 + i_123562];
                        
                        // futhark/microgpt.fut:353:352-583
                        
                        double zt_res_123601 = zt_res_123599 * zt_rhs_123600;
                        
                        // futhark/microgpt.fut:353:559-598
                        
                        double zs_res_123602 = zt_res_123601 / 2.0;
                        
                        // futhark/microgpt.fut:353:585-631
                        
                        double zt_res_123603 = zt_rhs_123581 * zs_res_123602;
                        
                        // futhark/microgpt.fut:353:290-631
                        
                        double zp_res_123604 = zt_res_123582 + zt_res_123603;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_123605 = r_123563 + zp_res_123604;
                        double r_tmp_134796 = zp_res_123605;
                        
                        r_123563 = r_tmp_134796;
                    }
                    defunc_0_lifted_lambda_res_123561 = r_123563;
                    ((double *) mem_133924)[i_132236] = defunc_0_lifted_lambda_res_123561;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_133919, i_132240 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133924, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_133913, i_132244 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_133919, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132262 = 0; i_132262 < (int64_t) 16; i_132262++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132252 = 0; i_132252 < (int64_t) 16; i_132252++) {
                // futhark/microgpt.fut:363:57-60
                
                int64_t tmp_130960 = sdiv64(i_132252, (int64_t) 4);
                
                // futhark/microgpt.fut:363:44-62
                
                bool x_130961 = sle64((int64_t) 0, tmp_130960);
                
                // futhark/microgpt.fut:363:44-62
                
                bool y_130962 = slt64(tmp_130960, (int64_t) 4);
                
                // futhark/microgpt.fut:363:44-62
                
                bool bounds_check_130963 = x_130961 && y_130962;
                
                // futhark/microgpt.fut:363:44-62
                
                bool index_certs_130964;
                
                if (!bounds_check_130963) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_130960, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:363:44-62\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:363:13-85\n   #6  futhark/microgpt.fut:568:5-76\n   #7  futhark/microgpt.fut:585:26-591:31\n   #8  futhark/microgpt.fut:619:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:363:79-82
                
                int64_t tmp_130965 = smod64(i_132252, (int64_t) 4);
                
                // futhark/microgpt.fut:363:44-84
                
                bool x_130966 = sle64((int64_t) 0, tmp_130965);
                
                // futhark/microgpt.fut:363:44-84
                
                bool y_130967 = slt64(tmp_130965, (int64_t) 4);
                
                // futhark/microgpt.fut:363:44-84
                
                bool bounds_check_130968 = x_130966 && y_130967;
                
                // futhark/microgpt.fut:363:44-84
                
                bool index_certs_130969;
                
                if (!bounds_check_130968) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_130965, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:363:44-84\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:363:13-85\n   #6  futhark/microgpt.fut:568:5-76\n   #7  futhark/microgpt.fut:585:26-591:31\n   #8  futhark/microgpt.fut:619:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_130970 = ((double *) mem_133673)[tmp_130960 * (int64_t) 64 + i_132262 * (int64_t) 4 + tmp_130965];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_130983 = ((double *) mem_133913)[tmp_130960 * (int64_t) 64 + i_132262 * (int64_t) 4 + tmp_130965];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_130999 = ((double *) mem_133870)[tmp_130960 * (int64_t) 64 + i_132262 * (int64_t) 4 + tmp_130965];
                
                ((double *) mem_133955)[i_132252] = lifted_lambda_res_130999;
                ((double *) mem_133956)[i_132252] = lifted_lambda_res_130983;
                ((double *) mem_133957)[i_132252] = lifted_lambda_res_130970;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133940, i_132262 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133955, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133941, i_132262 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133956, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133942, i_132262 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_133957, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132287 = 0; i_132287 < (int64_t) 16; i_132287++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132274 = 0; i_132274 < (int64_t) 16; i_132274++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_131162;
                double r_131164 = 0.0;
                
                for (int64_t i_131163 = 0; i_131163 < (int64_t) 16; i_131163++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_131165 = ((double *) mem_133942)[i_132287 * (int64_t) 16 + i_131163];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_131166 = ((double *) mem_param_132666.mem)[i_131163 * (int64_t) 16 + i_132274];
                    
                    // futhark/microgpt.fut:366:69-114
                    
                    double zt_res_131167 = zt_lhs_131165 * zt_rhs_131166;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_131168 = r_131164 + zt_res_131167;
                    double r_tmp_134813 = zp_res_131168;
                    
                    r_131164 = r_tmp_134813;
                }
                defunc_0_lifted_lambda_res_131162 = r_131164;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_131169;
                double r_131171 = 0.0;
                
                for (int64_t i_131170 = 0; i_131170 < (int64_t) 16; i_131170++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_131172 = ((double *) mem_133941)[i_132287 * (int64_t) 16 + i_131170];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_131173 = ((double *) mem_param_132642.mem)[i_131170 * (int64_t) 16 + i_132274];
                    
                    // futhark/microgpt.fut:366:145-190
                    
                    double zt_res_131174 = zt_lhs_131172 * zt_rhs_131173;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_131175 = r_131171 + zt_res_131174;
                    double r_tmp_134814 = zp_res_131175;
                    
                    r_131171 = r_tmp_134814;
                }
                defunc_0_lifted_lambda_res_131169 = r_131171;
                // futhark/microgpt.fut:366:47-192
                
                double zp_res_131176 = defunc_0_lifted_lambda_res_131162 + defunc_0_lifted_lambda_res_131169;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_131177;
                double r_131179 = 0.0;
                
                for (int64_t i_131178 = 0; i_131178 < (int64_t) 16; i_131178++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_131180 = ((double *) mem_133940)[i_132287 * (int64_t) 16 + i_131178];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_131181 = ((double *) mem_param_132654.mem)[i_131178 * (int64_t) 16 + i_132274];
                    
                    // futhark/microgpt.fut:366:222-267
                    
                    double zt_res_131182 = zt_lhs_131180 * zt_rhs_131181;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_131183 = r_131179 + zt_res_131182;
                    double r_tmp_134815 = zp_res_131183;
                    
                    r_131179 = r_tmp_134815;
                }
                defunc_0_lifted_lambda_res_131177 = r_131179;
                // futhark/microgpt.fut:366:118-269
                
                double zp_res_131184 = zp_res_131176 + defunc_0_lifted_lambda_res_131177;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_131191;
                double r_131193 = 0.0;
                
                for (int64_t i_131192 = 0; i_131192 < (int64_t) 16; i_131192++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_131194 = ((double *) mem_133940)[i_131192 * (int64_t) 16 + i_132287];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_131195 = ((double *) mem_132848)[i_131192 * (int64_t) 16 + i_132274];
                    
                    // futhark/microgpt.fut:392:68-111
                    
                    double zt_res_131196 = zt_lhs_131194 * zt_rhs_131195;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_131197 = r_131193 + zt_res_131196;
                    double r_tmp_134816 = zp_res_131197;
                    
                    r_131193 = r_tmp_134816;
                }
                defunc_0_lifted_lambda_res_131191 = r_131193;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_131207;
                double r_131209 = 0.0;
                
                for (int64_t i_131208 = 0; i_131208 < (int64_t) 16; i_131208++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_131210 = ((double *) mem_133941)[i_131208 * (int64_t) 16 + i_132287];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_131211 = ((double *) mem_132848)[i_131208 * (int64_t) 16 + i_132274];
                    
                    // futhark/microgpt.fut:393:68-111
                    
                    double zt_res_131212 = zt_lhs_131210 * zt_rhs_131211;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_131213 = r_131209 + zt_res_131212;
                    double r_tmp_134817 = zp_res_131213;
                    
                    r_131209 = r_tmp_134817;
                }
                defunc_0_lifted_lambda_res_131207 = r_131209;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_131225;
                double r_131227 = 0.0;
                
                for (int64_t i_131226 = 0; i_131226 < (int64_t) 16; i_131226++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_131228 = ((double *) mem_133942)[i_131226 * (int64_t) 16 + i_132287];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_131229 = ((double *) mem_132848)[i_131226 * (int64_t) 16 + i_132274];
                    
                    // futhark/microgpt.fut:394:68-111
                    
                    double zt_res_131230 = zt_lhs_131228 * zt_rhs_131229;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_131231 = r_131227 + zt_res_131230;
                    double r_tmp_134818 = zp_res_131231;
                    
                    r_131227 = r_tmp_134818;
                }
                defunc_0_lifted_lambda_res_131225 = r_131227;
                ((double *) mem_134008)[i_132274] = defunc_0_lifted_lambda_res_131225;
                ((double *) mem_134009)[i_132274] = defunc_0_lifted_lambda_res_131207;
                ((double *) mem_134010)[i_132274] = defunc_0_lifted_lambda_res_131191;
                ((double *) mem_134011)[i_132274] = zp_res_131184;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133988, i_132287 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_134008, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133989, i_132287 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_134009, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133990, i_132287 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_134010, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_133991, i_132287 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_134011, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132296 = 0; i_132296 < (int64_t) 16; i_132296++) {
            // futhark/microgpt.fut:369:47-59
            
            double zp_lhs_124521 = ((double *) mem_132892)[i_132296];
            
            // futhark/microgpt.fut:369:47-87
            
            double zp_res_124522 = 1.0e-5 + zp_lhs_124521;
            
            // futhark/microgpt.fut:369:39-87
            
            double sqrt_res_124523 = futrts_sqrt64(zp_res_124522);
            
            // futhark/microgpt.fut:370:128-157
            
            double zt_res_124531 = sqrt_res_124523 * sqrt_res_124523;
            
            // futhark/microgpt.fut:370:119-157
            
            double zs_res_124532 = 1.0 / zt_res_124531;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124533;
            double r_124535 = 0.0;
            
            for (int64_t i_124534 = 0; i_124534 < (int64_t) 16; i_124534++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_124536 = ((double *) mem_133991)[i_132296 * (int64_t) 16 + i_124534];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_124537 = ((double *) mem_132808)[i_132296 * (int64_t) 16 + i_124534];
                
                // futhark/microgpt.fut:370:69-112
                
                double zt_res_124538 = zt_lhs_124536 * zt_rhs_124537;
                
                // futhark/microgpt.fut:370:90-157
                
                double zt_res_124539 = zs_res_124532 * zt_res_124538;
                
                // futhark/microgpt.fut:370:61-157
                
                double neg_res_124540 = -zt_res_124539;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_124541 = r_124535 + neg_res_124540;
                double r_tmp_134821 = zp_res_124541;
                
                r_124535 = r_tmp_134821;
            }
            defunc_0_lifted_lambda_res_124533 = r_124535;
            ((double *) mem_134052)[i_132296] = defunc_0_lifted_lambda_res_124533;
            ((double *) mem_134053)[i_132296] = sqrt_res_124523;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132301 = 0; i_132301 < (int64_t) 16; i_132301++) {
            // futhark/microgpt.fut:371:39-51
            
            double zt_lhs_124094 = ((double *) mem_134052)[i_132301];
            
            // futhark/microgpt.fut:371:93-105
            
            double zp_lhs_124095 = ((double *) mem_132892)[i_132301];
            
            // futhark/microgpt.fut:371:93-133
            
            double zp_res_124096 = 1.0e-5 + zp_lhs_124095;
            
            // futhark/microgpt.fut:371:85-133
            
            double sqrt_res_124097 = futrts_sqrt64(zp_res_124096);
            
            // futhark/microgpt.fut:371:71-135
            
            double zt_res_124098 = 2.0 * sqrt_res_124097;
            
            // futhark/microgpt.fut:371:57-135
            
            double zs_res_124099 = 1.0 / zt_res_124098;
            
            // futhark/microgpt.fut:371:39-135
            
            double zt_res_124100 = zt_lhs_124094 * zs_res_124099;
            
            ((double *) mem_134066)[i_132301] = zt_res_124100;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132305 = 0; i_132305 < (int64_t) 16; i_132305++) {
            // futhark/microgpt.fut:372:49-61
            
            double zs_lhs_124108 = ((double *) mem_134066)[i_132305];
            
            // futhark/microgpt.fut:372:49-76
            
            double zs_res_124109 = zs_lhs_124108 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_134824 = 0; nest_i_134824 < (int64_t) 16; nest_i_134824++) {
                ((double *) mem_134073)[i_132305 * (int64_t) 16 + nest_i_134824] = zs_res_124109;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132313 = 0; i_132313 < (int64_t) 16; i_132313++) {
            // futhark/microgpt.fut:373:99-111
            
            double zs_rhs_124118 = ((double *) mem_134053)[i_132313];
            
            // futhark/microgpt.fut:373:91-111
            
            double zs_res_124119 = 1.0 / zs_rhs_124118;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132309 = 0; i_132309 < (int64_t) 16; i_132309++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_124126 = ((double *) mem_133542)[i_132313 * (int64_t) 16 + i_132309];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_124127 = ((double *) mem_133991)[i_132313 * (int64_t) 16 + i_132309];
                
                // futhark/microgpt.fut:373:65-111
                
                double zt_res_124128 = zs_res_124119 * zt_lhs_124127;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_124129 = ((double *) mem_134073)[i_132313 * (int64_t) 16 + i_132309];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_124130 = ((double *) mem_132808)[i_132313 * (int64_t) 16 + i_132309];
                
                // futhark/microgpt.fut:373:119-162
                
                double zt_res_124131 = zt_lhs_124129 * zt_rhs_124130;
                
                // futhark/microgpt.fut:373:86-162
                
                double zp_res_124132 = zt_res_124128 + zt_res_124131;
                
                // futhark/microgpt.fut:373:114-213
                
                double zp_res_124133 = zt_res_124131 + zp_res_124132;
                
                // futhark/microgpt.fut:373:37-213
                
                double zp_res_124134 = zp_lhs_124126 + zp_res_124133;
                
                ((double *) mem_134088)[i_132309] = zp_res_124134;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_134083, i_132313 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_134088, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132319 = 0; i_132319 < (int64_t) 16; i_132319++) {
            // futhark/microgpt.fut:376:47-59
            
            double zp_lhs_124483 = ((double *) mem_132846)[i_132319];
            
            // futhark/microgpt.fut:376:47-87
            
            double zp_res_124484 = 1.0e-5 + zp_lhs_124483;
            
            // futhark/microgpt.fut:376:39-87
            
            double sqrt_res_124485 = futrts_sqrt64(zp_res_124484);
            
            // futhark/microgpt.fut:377:128-157
            
            double zt_res_124493 = sqrt_res_124485 * sqrt_res_124485;
            
            // futhark/microgpt.fut:377:119-157
            
            double zs_res_124494 = 1.0 / zt_res_124493;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124495;
            double r_124497 = 0.0;
            
            for (int64_t i_124496 = 0; i_124496 < (int64_t) 16; i_124496++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_124498 = ((double *) mem_134083)[i_132319 * (int64_t) 16 + i_124496];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_124499 = ((double *) mem_132776)[i_132319 * (int64_t) 16 + i_124496];
                
                // futhark/microgpt.fut:377:69-112
                
                double zt_res_124500 = zt_lhs_124498 * zt_rhs_124499;
                
                // futhark/microgpt.fut:377:90-157
                
                double zt_res_124501 = zs_res_124494 * zt_res_124500;
                
                // futhark/microgpt.fut:377:61-157
                
                double neg_res_124502 = -zt_res_124501;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_124503 = r_124497 + neg_res_124502;
                double r_tmp_134829 = zp_res_124503;
                
                r_124497 = r_tmp_134829;
            }
            defunc_0_lifted_lambda_res_124495 = r_124497;
            ((double *) mem_134099)[i_132319] = defunc_0_lifted_lambda_res_124495;
            ((double *) mem_134100)[i_132319] = sqrt_res_124485;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132324 = 0; i_132324 < (int64_t) 16; i_132324++) {
            // futhark/microgpt.fut:378:39-51
            
            double zt_lhs_124201 = ((double *) mem_134099)[i_132324];
            
            // futhark/microgpt.fut:378:93-105
            
            double zp_lhs_124202 = ((double *) mem_132846)[i_132324];
            
            // futhark/microgpt.fut:378:93-133
            
            double zp_res_124203 = 1.0e-5 + zp_lhs_124202;
            
            // futhark/microgpt.fut:378:85-133
            
            double sqrt_res_124204 = futrts_sqrt64(zp_res_124203);
            
            // futhark/microgpt.fut:378:71-135
            
            double zt_res_124205 = 2.0 * sqrt_res_124204;
            
            // futhark/microgpt.fut:378:57-135
            
            double zs_res_124206 = 1.0 / zt_res_124205;
            
            // futhark/microgpt.fut:378:39-135
            
            double zt_res_124207 = zt_lhs_124201 * zs_res_124206;
            
            ((double *) mem_134113)[i_132324] = zt_res_124207;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132328 = 0; i_132328 < (int64_t) 16; i_132328++) {
            // futhark/microgpt.fut:379:49-61
            
            double zs_lhs_124215 = ((double *) mem_134113)[i_132328];
            
            // futhark/microgpt.fut:379:49-76
            
            double zs_res_124216 = zs_lhs_124215 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_134832 = 0; nest_i_134832 < (int64_t) 16; nest_i_134832++) {
                ((double *) mem_134120)[i_132328 * (int64_t) 16 + nest_i_134832] = zs_res_124216;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132336 = 0; i_132336 < (int64_t) 16; i_132336++) {
            // futhark/microgpt.fut:380:73-85
            
            double zs_rhs_124225 = ((double *) mem_134100)[i_132336];
            
            // futhark/microgpt.fut:380:65-85
            
            double zs_res_124226 = 1.0 / zs_rhs_124225;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132332 = 0; i_132332 < (int64_t) 16; i_132332++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_124233 = ((double *) mem_134083)[i_132336 * (int64_t) 16 + i_132332];
                
                // futhark/microgpt.fut:380:39-85
                
                double zt_res_124234 = zs_res_124226 * zt_lhs_124233;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_124235 = ((double *) mem_134120)[i_132336 * (int64_t) 16 + i_132332];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_124236 = ((double *) mem_132776)[i_132336 * (int64_t) 16 + i_132332];
                
                // futhark/microgpt.fut:380:93-136
                
                double zt_res_124237 = zt_lhs_124235 * zt_rhs_124236;
                
                // futhark/microgpt.fut:380:60-136
                
                double zp_res_124238 = zt_res_124234 + zt_res_124237;
                
                // futhark/microgpt.fut:380:88-187
                
                double zp_res_124239 = zt_res_124237 + zp_res_124238;
                
                ((double *) mem_134135)[i_132332] = zp_res_124239;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_134130, i_132336 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_134135, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132349 = 0; i_132349 < (int64_t) 16; i_132349++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132342 = 0; i_132342 < (int64_t) 16; i_132342++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_131257 = ((double *) mem_134130)[i_132349 * (int64_t) 16 + i_132342];
                
                ((double *) mem_134156)[i_132342] = lifted_lambda_res_131257;
                ((double *) mem_134157)[i_132342] = lifted_lambda_res_131257;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_134146, i_132349 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_134156, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_134147, i_132349 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_134157, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132358 = 0; i_132358 < (int64_t) 64; i_132358++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132354 = 0; i_132354 < (int64_t) 16; i_132354++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_124353;
                double r_124355 = 0.0;
                
                for (int64_t i_124354 = 0; i_124354 < (int64_t) 16; i_124354++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_124356 = ((double *) mem_133479)[i_124354 * (int64_t) 64 + i_132358];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_124357 = ((double *) mem_133209)[i_124354 * (int64_t) 16 + i_132354];
                    
                    // futhark/microgpt.fut:396:67-111
                    
                    double zt_res_124358 = zt_lhs_124356 * zt_rhs_124357;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_124359 = r_124355 + zt_res_124358;
                    double r_tmp_134841 = zp_res_124359;
                    
                    r_124355 = r_tmp_134841;
                }
                defunc_0_lifted_lambda_res_124353 = r_124355;
                ((double *) mem_134183)[i_132354] = defunc_0_lifted_lambda_res_124353;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_134178, i_132358 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_134183, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_132371 = 0; i_132371 < (int64_t) 27; i_132371++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_132364 = 0; i_132364 < (int64_t) 16; i_132364++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_131285;
                double r_131287 = 0.0;
                
                for (int64_t i_131286 = 0; i_131286 < (int64_t) 16; i_131286++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_131288 = ((double *) mem_133415)[i_131286 * (int64_t) 27 + i_132371];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_131289 = ((double *) mem_133302)[i_131286 * (int64_t) 16 + i_132364];
                    
                    // futhark/microgpt.fut:398:68-111
                    
                    double zt_res_131290 = zt_lhs_131288 * zt_rhs_131289;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_131291 = r_131287 + zt_res_131290;
                    double r_tmp_134846 = zp_res_131291;
                    
                    r_131287 = r_tmp_134846;
                }
                defunc_0_lifted_lambda_res_131285 = r_131287;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_131294;
                double r_131296 = 0.0;
                
                for (int64_t i_131295 = 0; i_131295 < (int64_t) 16; i_131295++) {
                    int64_t zeze_lhs_131297 = ((int64_t *) seqs_mem_132634.mem)[step_122159 * (int64_t) 16 + i_131295];
                    
                    // futhark/microgpt.fut:569:58-109
                    
                    bool cond_131298 = zeze_lhs_131297 == i_132371;
                    
                    // futhark/microgpt.fut:569:58-109
                    
                    double lifted_lambda_res_131299;
                    
                    if (cond_131298) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double lifted_lambda_res_t_res_131618 = ((double *) mem_134146)[i_131295 * (int64_t) 16 + i_132364];
                        
                        lifted_lambda_res_131299 = lifted_lambda_res_t_res_131618;
                    } else {
                        lifted_lambda_res_131299 = 0.0;
                    }
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_131305 = r_131296 + lifted_lambda_res_131299;
                    double r_tmp_134847 = zp_res_131305;
                    
                    r_131296 = r_tmp_134847;
                }
                defunc_0_lifted_lambda_res_131294 = r_131296;
                ((double *) mem_134204)[i_132364] = defunc_0_lifted_lambda_res_131294;
                ((double *) mem_134205)[i_132364] = defunc_0_lifted_lambda_res_131285;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_134194, i_132371 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_134204, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_134195, i_132371 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_134205, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:66:26-45
        
        double i64_res_124437 = sitofp_i64_f64(step_122159);
        
        // futhark/microgpt.fut:504:46-65
        
        double zm_rhs_124438 = i64_res_124437 / 500.0;
        
        // futhark/microgpt.fut:504:24-65
        
        double zt_rhs_124439 = 1.0 - zm_rhs_124438;
        
        // futhark/microgpt.fut:504:19-65
        
        double lt_r_124440 = 1.0e-2 * zt_rhs_124439;
        
        // futhark/microgpt.fut:506:5-52
        if (memblock_alloc(ctx, &mem_134226, (int64_t) 3456, "mem_134226")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:506:5-52
        // futhark/microgpt.fut:506:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134226.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132658.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:506:5-52
        if (memblock_alloc(ctx, &mem_134228, (int64_t) 3456, "mem_134228")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:506:5-52
        // futhark/microgpt.fut:506:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134228.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132694.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:506:5-52
        if (memblock_alloc(ctx, &mem_134230, (int64_t) 3456, "mem_134230")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:506:5-52
        // futhark/microgpt.fut:506:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134230.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132730.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:506:5-52
        if (memblock_alloc(ctx, &mem_134232, (int64_t) 3456, "mem_134232")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:506:5-52
        // futhark/microgpt.fut:506:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134232.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_134194, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:506:5-52
        if (futrts_adam_opt_w_12605(ctx, &ext_mem_134236, &ext_mem_134235, &ext_mem_134234, mem_134226, mem_134228, mem_134230, mem_134232, (int64_t) 27, (int64_t) 16, step_122159, lt_r_124440) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_134226, "mem_134226") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134228, "mem_134228") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134230, "mem_134230") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134232, "mem_134232") != 0)
            return 1;
        // futhark/microgpt.fut:508:5-52
        if (memblock_alloc(ctx, &mem_134237, (int64_t) 2048, "mem_134237")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:508:5-52
        // futhark/microgpt.fut:508:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134237.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132650.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:508:5-52
        if (memblock_alloc(ctx, &mem_134239, (int64_t) 2048, "mem_134239")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:508:5-52
        // futhark/microgpt.fut:508:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134239.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132686.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:508:5-52
        if (memblock_alloc(ctx, &mem_134241, (int64_t) 2048, "mem_134241")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:508:5-52
        // futhark/microgpt.fut:508:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134241.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132722.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:508:5-52
        if (memblock_alloc(ctx, &mem_134243, (int64_t) 2048, "mem_134243")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:508:5-52
        // futhark/microgpt.fut:508:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134243.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_134147, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:508:5-52
        if (futrts_adam_opt_w_12606(ctx, &ext_mem_134247, &ext_mem_134246, &ext_mem_134245, mem_134237, mem_134239, mem_134241, mem_134243, (int64_t) 16, (int64_t) 16, step_122159, lt_r_124440) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_134237, "mem_134237") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134239, "mem_134239") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134241, "mem_134241") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134243, "mem_134243") != 0)
            return 1;
        // futhark/microgpt.fut:510:5-56
        if (memblock_alloc(ctx, &mem_134248, (int64_t) 2048, "mem_134248")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:510:5-56
        // futhark/microgpt.fut:510:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134248.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132654.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:510:5-56
        if (memblock_alloc(ctx, &mem_134250, (int64_t) 2048, "mem_134250")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:510:5-56
        // futhark/microgpt.fut:510:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134250.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132690.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:510:5-56
        if (memblock_alloc(ctx, &mem_134252, (int64_t) 2048, "mem_134252")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:510:5-56
        // futhark/microgpt.fut:510:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134252.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132726.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:510:5-56
        if (memblock_alloc(ctx, &mem_134254, (int64_t) 2048, "mem_134254")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:510:5-56
        // futhark/microgpt.fut:510:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134254.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_133990, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:510:5-56
        if (futrts_adam_opt_w_12606(ctx, &ext_mem_134258, &ext_mem_134257, &ext_mem_134256, mem_134248, mem_134250, mem_134252, mem_134254, (int64_t) 16, (int64_t) 16, step_122159, lt_r_124440) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_134248, "mem_134248") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134250, "mem_134250") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134252, "mem_134252") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134254, "mem_134254") != 0)
            return 1;
        // futhark/microgpt.fut:512:5-56
        if (memblock_alloc(ctx, &mem_134259, (int64_t) 2048, "mem_134259")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:512:5-56
        // futhark/microgpt.fut:512:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134259.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132642.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:512:5-56
        if (memblock_alloc(ctx, &mem_134261, (int64_t) 2048, "mem_134261")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:512:5-56
        // futhark/microgpt.fut:512:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134261.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132678.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:512:5-56
        if (memblock_alloc(ctx, &mem_134263, (int64_t) 2048, "mem_134263")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:512:5-56
        // futhark/microgpt.fut:512:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134263.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132714.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:512:5-56
        if (memblock_alloc(ctx, &mem_134265, (int64_t) 2048, "mem_134265")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:512:5-56
        // futhark/microgpt.fut:512:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134265.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_133989, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:512:5-56
        if (futrts_adam_opt_w_12606(ctx, &ext_mem_134269, &ext_mem_134268, &ext_mem_134267, mem_134259, mem_134261, mem_134263, mem_134265, (int64_t) 16, (int64_t) 16, step_122159, lt_r_124440) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_134259, "mem_134259") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134261, "mem_134261") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134263, "mem_134263") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134265, "mem_134265") != 0)
            return 1;
        // futhark/microgpt.fut:514:5-56
        if (memblock_alloc(ctx, &mem_134270, (int64_t) 2048, "mem_134270")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:514:5-56
        // futhark/microgpt.fut:514:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134270.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132666.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:514:5-56
        if (memblock_alloc(ctx, &mem_134272, (int64_t) 2048, "mem_134272")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:514:5-56
        // futhark/microgpt.fut:514:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134272.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132702.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:514:5-56
        if (memblock_alloc(ctx, &mem_134274, (int64_t) 2048, "mem_134274")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:514:5-56
        // futhark/microgpt.fut:514:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134274.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132738.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:514:5-56
        if (memblock_alloc(ctx, &mem_134276, (int64_t) 2048, "mem_134276")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:514:5-56
        // futhark/microgpt.fut:514:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134276.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_133988, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:514:5-56
        if (futrts_adam_opt_w_12606(ctx, &ext_mem_134280, &ext_mem_134279, &ext_mem_134278, mem_134270, mem_134272, mem_134274, mem_134276, (int64_t) 16, (int64_t) 16, step_122159, lt_r_124440) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_134270, "mem_134270") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134272, "mem_134272") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134274, "mem_134274") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134276, "mem_134276") != 0)
            return 1;
        // futhark/microgpt.fut:516:5-56
        if (memblock_alloc(ctx, &mem_134281, (int64_t) 2048, "mem_134281")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:516:5-56
        // futhark/microgpt.fut:516:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134281.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132646.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:516:5-56
        if (memblock_alloc(ctx, &mem_134283, (int64_t) 2048, "mem_134283")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:516:5-56
        // futhark/microgpt.fut:516:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134283.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132682.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:516:5-56
        if (memblock_alloc(ctx, &mem_134285, (int64_t) 2048, "mem_134285")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:516:5-56
        // futhark/microgpt.fut:516:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134285.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132718.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:516:5-56
        if (memblock_alloc(ctx, &mem_134287, (int64_t) 2048, "mem_134287")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:516:5-56
        // futhark/microgpt.fut:516:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134287.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_133558, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:516:5-56
        if (futrts_adam_opt_w_12606(ctx, &ext_mem_134291, &ext_mem_134290, &ext_mem_134289, mem_134281, mem_134283, mem_134285, mem_134287, (int64_t) 16, (int64_t) 16, step_122159, lt_r_124440) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_134281, "mem_134281") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134283, "mem_134283") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134285, "mem_134285") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134287, "mem_134287") != 0)
            return 1;
        // futhark/microgpt.fut:518:5-52
        if (memblock_alloc(ctx, &mem_134292, (int64_t) 8192, "mem_134292")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:518:5-52
        // futhark/microgpt.fut:518:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134292.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132662.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:518:5-52
        if (memblock_alloc(ctx, &mem_134294, (int64_t) 8192, "mem_134294")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:518:5-52
        // futhark/microgpt.fut:518:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134294.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132698.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:518:5-52
        if (memblock_alloc(ctx, &mem_134296, (int64_t) 8192, "mem_134296")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:518:5-52
        // futhark/microgpt.fut:518:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134296.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132734.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:518:5-52
        if (memblock_alloc(ctx, &mem_134298, (int64_t) 8192, "mem_134298")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:518:5-52
        // futhark/microgpt.fut:518:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134298.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_134178, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:518:5-52
        if (futrts_adam_opt_w_12605(ctx, &ext_mem_134302, &ext_mem_134301, &ext_mem_134300, mem_134292, mem_134294, mem_134296, mem_134298, (int64_t) 64, (int64_t) 16, step_122159, lt_r_124440) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_134292, "mem_134292") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134294, "mem_134294") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134296, "mem_134296") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134298, "mem_134298") != 0)
            return 1;
        // futhark/microgpt.fut:520:5-60
        if (memblock_alloc(ctx, &mem_134303, (int64_t) 8192, "mem_134303")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:520:5-60
        // futhark/microgpt.fut:520:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134303.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_132638.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:520:5-60
        if (memblock_alloc(ctx, &mem_134305, (int64_t) 8192, "mem_134305")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:520:5-60
        // futhark/microgpt.fut:520:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134305.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_132674.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:520:5-60
        if (memblock_alloc(ctx, &mem_134307, (int64_t) 8192, "mem_134307")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:520:5-60
        // futhark/microgpt.fut:520:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134307.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_132710.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:520:5-60
        if (memblock_alloc(ctx, &mem_134309, (int64_t) 8192, "mem_134309")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:520:5-60
        // futhark/microgpt.fut:520:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134309.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_133447, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:520:5-60
        if (futrts_adam_opt_w_12605(ctx, &ext_mem_134313, &ext_mem_134312, &ext_mem_134311, mem_134303, mem_134305, mem_134307, mem_134309, (int64_t) 16, (int64_t) 64, step_122159, lt_r_124440) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_134303, "mem_134303") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134305, "mem_134305") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134307, "mem_134307") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134309, "mem_134309") != 0)
            return 1;
        // futhark/microgpt.fut:522:5-56
        if (memblock_alloc(ctx, &mem_134314, (int64_t) 3456, "mem_134314")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:522:5-56
        // futhark/microgpt.fut:522:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134314.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132670.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:522:5-56
        if (memblock_alloc(ctx, &mem_134316, (int64_t) 3456, "mem_134316")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:522:5-56
        // futhark/microgpt.fut:522:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134316.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132706.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:522:5-56
        if (memblock_alloc(ctx, &mem_134318, (int64_t) 3456, "mem_134318")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:522:5-56
        // futhark/microgpt.fut:522:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134318.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_132742.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:522:5-56
        if (memblock_alloc(ctx, &mem_134320, (int64_t) 3456, "mem_134320")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:522:5-56
        // futhark/microgpt.fut:522:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_134320.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_134195, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:522:5-56
        if (futrts_adam_opt_w_12605(ctx, &ext_mem_134324, &ext_mem_134323, &ext_mem_134322, mem_134314, mem_134316, mem_134318, mem_134320, (int64_t) 27, (int64_t) 16, step_122159, lt_r_124440) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_134314, "mem_134314") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134316, "mem_134316") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134318, "mem_134318") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134320, "mem_134320") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134532, &ext_mem_134313, "ext_mem_134313") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134533, &ext_mem_134269, "ext_mem_134269") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134534, &ext_mem_134291, "ext_mem_134291") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134535, &ext_mem_134247, "ext_mem_134247") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134536, &ext_mem_134258, "ext_mem_134258") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134537, &ext_mem_134236, "ext_mem_134236") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134538, &ext_mem_134302, "ext_mem_134302") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134539, &ext_mem_134280, "ext_mem_134280") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134540, &ext_mem_134324, "ext_mem_134324") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134541, &ext_mem_134312, "ext_mem_134312") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134542, &ext_mem_134268, "ext_mem_134268") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134543, &ext_mem_134290, "ext_mem_134290") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134544, &ext_mem_134246, "ext_mem_134246") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134545, &ext_mem_134257, "ext_mem_134257") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134546, &ext_mem_134235, "ext_mem_134235") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134547, &ext_mem_134301, "ext_mem_134301") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134548, &ext_mem_134279, "ext_mem_134279") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134549, &ext_mem_134323, "ext_mem_134323") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134550, &ext_mem_134311, "ext_mem_134311") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134551, &ext_mem_134267, "ext_mem_134267") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134552, &ext_mem_134289, "ext_mem_134289") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134553, &ext_mem_134245, "ext_mem_134245") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134554, &ext_mem_134256, "ext_mem_134256") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134555, &ext_mem_134234, "ext_mem_134234") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134556, &ext_mem_134300, "ext_mem_134300") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134557, &ext_mem_134278, "ext_mem_134278") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_134558, &ext_mem_134322, "ext_mem_134322") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132638, &mem_param_tmp_134532, "mem_param_tmp_134532") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132642, &mem_param_tmp_134533, "mem_param_tmp_134533") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132646, &mem_param_tmp_134534, "mem_param_tmp_134534") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132650, &mem_param_tmp_134535, "mem_param_tmp_134535") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132654, &mem_param_tmp_134536, "mem_param_tmp_134536") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132658, &mem_param_tmp_134537, "mem_param_tmp_134537") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132662, &mem_param_tmp_134538, "mem_param_tmp_134538") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132666, &mem_param_tmp_134539, "mem_param_tmp_134539") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132670, &mem_param_tmp_134540, "mem_param_tmp_134540") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132674, &mem_param_tmp_134541, "mem_param_tmp_134541") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132678, &mem_param_tmp_134542, "mem_param_tmp_134542") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132682, &mem_param_tmp_134543, "mem_param_tmp_134543") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132686, &mem_param_tmp_134544, "mem_param_tmp_134544") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132690, &mem_param_tmp_134545, "mem_param_tmp_134545") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132694, &mem_param_tmp_134546, "mem_param_tmp_134546") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132698, &mem_param_tmp_134547, "mem_param_tmp_134547") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132702, &mem_param_tmp_134548, "mem_param_tmp_134548") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132706, &mem_param_tmp_134549, "mem_param_tmp_134549") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132710, &mem_param_tmp_134550, "mem_param_tmp_134550") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132714, &mem_param_tmp_134551, "mem_param_tmp_134551") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132718, &mem_param_tmp_134552, "mem_param_tmp_134552") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132722, &mem_param_tmp_134553, "mem_param_tmp_134553") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132726, &mem_param_tmp_134554, "mem_param_tmp_134554") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132730, &mem_param_tmp_134555, "mem_param_tmp_134555") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132734, &mem_param_tmp_134556, "mem_param_tmp_134556") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132738, &mem_param_tmp_134557, "mem_param_tmp_134557") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_132742, &mem_param_tmp_134558, "mem_param_tmp_134558") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_134432, &mem_param_132638, "mem_param_132638") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134431, &mem_param_132642, "mem_param_132642") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134430, &mem_param_132646, "mem_param_132646") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134429, &mem_param_132650, "mem_param_132650") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134428, &mem_param_132654, "mem_param_132654") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134427, &mem_param_132658, "mem_param_132658") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134426, &mem_param_132662, "mem_param_132662") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134425, &mem_param_132666, "mem_param_132666") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134424, &mem_param_132670, "mem_param_132670") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134423, &mem_param_132674, "mem_param_132674") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134422, &mem_param_132678, "mem_param_132678") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134421, &mem_param_132682, "mem_param_132682") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134420, &mem_param_132686, "mem_param_132686") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134419, &mem_param_132690, "mem_param_132690") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134418, &mem_param_132694, "mem_param_132694") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134417, &mem_param_132698, "mem_param_132698") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134416, &mem_param_132702, "mem_param_132702") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134415, &mem_param_132706, "mem_param_132706") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134414, &mem_param_132710, "mem_param_132710") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134413, &mem_param_132714, "mem_param_132714") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134412, &mem_param_132718, "mem_param_132718") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134411, &mem_param_132722, "mem_param_132722") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134410, &mem_param_132726, "mem_param_132726") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134409, &mem_param_132730, "mem_param_132730") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134408, &mem_param_132734, "mem_param_132734") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134407, &mem_param_132738, "mem_param_132738") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_134406, &mem_param_132742, "mem_param_132742") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134505, &ext_mem_134427, "ext_mem_134427") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134506, &ext_mem_134429, "ext_mem_134429") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134507, &ext_mem_134428, "ext_mem_134428") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134508, &ext_mem_134431, "ext_mem_134431") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134509, &ext_mem_134425, "ext_mem_134425") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134510, &ext_mem_134430, "ext_mem_134430") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134511, &ext_mem_134426, "ext_mem_134426") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134512, &ext_mem_134432, "ext_mem_134432") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134513, &ext_mem_134424, "ext_mem_134424") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134514, &ext_mem_134418, "ext_mem_134418") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134515, &ext_mem_134420, "ext_mem_134420") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134516, &ext_mem_134419, "ext_mem_134419") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134517, &ext_mem_134422, "ext_mem_134422") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134518, &ext_mem_134416, "ext_mem_134416") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134519, &ext_mem_134421, "ext_mem_134421") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134520, &ext_mem_134417, "ext_mem_134417") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134521, &ext_mem_134423, "ext_mem_134423") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134522, &ext_mem_134415, "ext_mem_134415") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134523, &ext_mem_134409, "ext_mem_134409") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134524, &ext_mem_134411, "ext_mem_134411") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134525, &ext_mem_134410, "ext_mem_134410") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134526, &ext_mem_134413, "ext_mem_134413") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134527, &ext_mem_134407, "ext_mem_134407") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134528, &ext_mem_134412, "ext_mem_134412") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134529, &ext_mem_134408, "ext_mem_134408") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134530, &ext_mem_134414, "ext_mem_134414") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134531, &ext_mem_134406, "ext_mem_134406") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135001, &mem_out_134505, "mem_out_134505") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135002, &mem_out_134506, "mem_out_134506") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135003, &mem_out_134507, "mem_out_134507") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135004, &mem_out_134508, "mem_out_134508") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135005, &mem_out_134509, "mem_out_134509") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135006, &mem_out_134510, "mem_out_134510") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135007, &mem_out_134511, "mem_out_134511") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135008, &mem_out_134512, "mem_out_134512") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135009, &mem_out_134513, "mem_out_134513") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135010, &mem_out_134514, "mem_out_134514") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135011, &mem_out_134515, "mem_out_134515") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135012, &mem_out_134516, "mem_out_134516") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135013, &mem_out_134517, "mem_out_134517") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135014, &mem_out_134518, "mem_out_134518") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135015, &mem_out_134519, "mem_out_134519") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135016, &mem_out_134520, "mem_out_134520") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135017, &mem_out_134521, "mem_out_134521") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135018, &mem_out_134522, "mem_out_134522") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135019, &mem_out_134523, "mem_out_134523") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135020, &mem_out_134524, "mem_out_134524") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135021, &mem_out_134525, "mem_out_134525") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135022, &mem_out_134526, "mem_out_134526") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135023, &mem_out_134527, "mem_out_134527") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135024, &mem_out_134528, "mem_out_134528") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135025, &mem_out_134529, "mem_out_134529") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135026, &mem_out_134530, "mem_out_134530") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135027, &mem_out_134531, "mem_out_134531") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_132743);
        free(mem_132744);
        free(mem_132753);
        free(mem_132760);
        free(mem_132775);
        free(mem_132776);
        free(mem_132785);
        free(mem_132792);
        free(mem_132807);
        free(mem_132808);
        free(mem_132817);
        free(mem_132818);
        free(mem_132831);
        free(mem_132846);
        free(mem_132847);
        free(mem_132848);
        free(mem_132860);
        free(mem_132861);
        free(mem_132874);
        free(mem_132892);
        free(mem_132893);
        free(mem_132894);
        free(mem_132895);
        free(mem_132911);
        free(mem_132912);
        free(mem_132913);
        free(mem_132947);
        free(mem_132948);
        free(mem_132949);
        free(mem_132965);
        free(mem_132966);
        free(mem_132967);
        free(mem_132980);
        free(mem_132981);
        free(mem_132982);
        free(mem_133028);
        free(mem_133029);
        free(mem_133030);
        free(mem_133031);
        free(mem_133032);
        free(mem_133055);
        free(mem_133056);
        free(mem_133057);
        free(mem_133058);
        free(mem_133059);
        free(mem_133077);
        free(mem_133078);
        free(mem_133091);
        free(mem_133092);
        free(mem_133122);
        free(mem_133127);
        free(mem_133160);
        free(mem_133165);
        free(mem_133176);
        free(mem_133181);
        free(mem_133192);
        free(mem_133197);
        free(mem_133208);
        free(mem_133209);
        free(mem_133218);
        free(mem_133219);
        free(mem_133232);
        free(mem_133247);
        free(mem_133248);
        free(mem_133256);
        free(mem_133270);
        free(mem_133275);
        free(mem_133286);
        free(mem_133291);
        free(mem_133302);
        free(mem_133307);
        free(mem_133318);
        free(mem_133323);
        free(mem_133334);
        free(mem_133335);
        free(mem_133348);
        free(mem_133353);
        free(mem_133364);
        free(mem_133365);
        free(mem_133372);
        free(mem_133385);
        free(mem_133390);
        free(mem_133397);
        free(mem_133408);
        free(mem_133415);
        free(mem_133420);
        free(mem_133431);
        free(mem_133436);
        free(mem_133447);
        free(mem_133448);
        free(mem_133457);
        free(mem_133458);
        free(mem_133479);
        free(mem_133484);
        free(mem_133495);
        free(mem_133500);
        free(mem_133511);
        free(mem_133512);
        free(mem_133525);
        free(mem_133532);
        free(mem_133542);
        free(mem_133547);
        free(mem_133558);
        free(mem_133559);
        free(mem_133568);
        free(mem_133569);
        free(mem_133590);
        free(mem_133591);
        free(mem_133592);
        free(mem_133608);
        free(mem_133609);
        free(mem_133610);
        free(mem_133623);
        free(mem_133630);
        free(mem_133631);
        free(mem_133671);
        free(mem_133672);
        free(mem_133673);
        free(mem_133689);
        free(mem_133690);
        free(mem_133691);
        free(mem_133704);
        free(mem_133711);
        free(mem_133712);
        free(mem_133752);
        free(mem_133753);
        free(mem_133754);
        free(mem_133755);
        free(mem_133772);
        free(mem_133773);
        free(mem_133774);
        free(mem_133775);
        free(mem_133816);
        free(mem_133817);
        free(mem_133828);
        free(mem_133829);
        free(mem_133838);
        free(mem_133839);
        free(mem_133870);
        free(mem_133871);
        free(mem_133881);
        free(mem_133882);
        free(mem_133890);
        free(mem_133913);
        free(mem_133919);
        free(mem_133924);
        free(mem_133940);
        free(mem_133941);
        free(mem_133942);
        free(mem_133955);
        free(mem_133956);
        free(mem_133957);
        free(mem_133988);
        free(mem_133989);
        free(mem_133990);
        free(mem_133991);
        free(mem_134008);
        free(mem_134009);
        free(mem_134010);
        free(mem_134011);
        free(mem_134052);
        free(mem_134053);
        free(mem_134066);
        free(mem_134073);
        free(mem_134083);
        free(mem_134088);
        free(mem_134099);
        free(mem_134100);
        free(mem_134113);
        free(mem_134120);
        free(mem_134130);
        free(mem_134135);
        free(mem_134146);
        free(mem_134147);
        free(mem_134156);
        free(mem_134157);
        free(mem_134178);
        free(mem_134183);
        free(mem_134194);
        free(mem_134195);
        free(mem_134204);
        free(mem_134205);
        if (memblock_unref(ctx, &mem_param_tmp_134558, "mem_param_tmp_134558") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134557, "mem_param_tmp_134557") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134556, "mem_param_tmp_134556") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134555, "mem_param_tmp_134555") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134554, "mem_param_tmp_134554") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134553, "mem_param_tmp_134553") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134552, "mem_param_tmp_134552") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134551, "mem_param_tmp_134551") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134550, "mem_param_tmp_134550") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134549, "mem_param_tmp_134549") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134548, "mem_param_tmp_134548") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134547, "mem_param_tmp_134547") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134546, "mem_param_tmp_134546") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134545, "mem_param_tmp_134545") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134544, "mem_param_tmp_134544") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134543, "mem_param_tmp_134543") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134542, "mem_param_tmp_134542") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134541, "mem_param_tmp_134541") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134540, "mem_param_tmp_134540") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134539, "mem_param_tmp_134539") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134538, "mem_param_tmp_134538") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134537, "mem_param_tmp_134537") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134536, "mem_param_tmp_134536") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134535, "mem_param_tmp_134535") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134534, "mem_param_tmp_134534") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134533, "mem_param_tmp_134533") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_134532, "mem_param_tmp_134532") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134322, "ext_mem_134322") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134323, "ext_mem_134323") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134324, "ext_mem_134324") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134320, "mem_134320") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134318, "mem_134318") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134316, "mem_134316") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134314, "mem_134314") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134311, "ext_mem_134311") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134312, "ext_mem_134312") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134313, "ext_mem_134313") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134309, "mem_134309") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134307, "mem_134307") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134305, "mem_134305") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134303, "mem_134303") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134300, "ext_mem_134300") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134301, "ext_mem_134301") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134302, "ext_mem_134302") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134298, "mem_134298") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134296, "mem_134296") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134294, "mem_134294") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134292, "mem_134292") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134289, "ext_mem_134289") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134290, "ext_mem_134290") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134291, "ext_mem_134291") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134287, "mem_134287") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134285, "mem_134285") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134283, "mem_134283") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134281, "mem_134281") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134278, "ext_mem_134278") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134279, "ext_mem_134279") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134280, "ext_mem_134280") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134276, "mem_134276") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134274, "mem_134274") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134272, "mem_134272") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134270, "mem_134270") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134267, "ext_mem_134267") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134268, "ext_mem_134268") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134269, "ext_mem_134269") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134265, "mem_134265") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134263, "mem_134263") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134261, "mem_134261") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134259, "mem_134259") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134256, "ext_mem_134256") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134257, "ext_mem_134257") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134258, "ext_mem_134258") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134254, "mem_134254") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134252, "mem_134252") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134250, "mem_134250") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134248, "mem_134248") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134245, "ext_mem_134245") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134246, "ext_mem_134246") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134247, "ext_mem_134247") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134243, "mem_134243") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134241, "mem_134241") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134239, "mem_134239") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134237, "mem_134237") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134234, "ext_mem_134234") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134235, "ext_mem_134235") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134236, "ext_mem_134236") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134232, "mem_134232") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134230, "mem_134230") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134228, "mem_134228") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_134226, "mem_134226") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132742, "mem_param_132742") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132738, "mem_param_132738") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132734, "mem_param_132734") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132730, "mem_param_132730") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132726, "mem_param_132726") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132722, "mem_param_132722") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132718, "mem_param_132718") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132714, "mem_param_132714") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132710, "mem_param_132710") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132706, "mem_param_132706") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132702, "mem_param_132702") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132698, "mem_param_132698") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132694, "mem_param_132694") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132690, "mem_param_132690") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132686, "mem_param_132686") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132682, "mem_param_132682") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132678, "mem_param_132678") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132674, "mem_param_132674") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132670, "mem_param_132670") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132666, "mem_param_132666") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132662, "mem_param_132662") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132658, "mem_param_132658") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132654, "mem_param_132654") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132650, "mem_param_132650") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132646, "mem_param_132646") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132642, "mem_param_132642") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_132638, "mem_param_132638") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134406, "ext_mem_134406") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134407, "ext_mem_134407") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134408, "ext_mem_134408") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134409, "ext_mem_134409") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134410, "ext_mem_134410") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134411, "ext_mem_134411") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134412, "ext_mem_134412") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134413, "ext_mem_134413") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134414, "ext_mem_134414") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134415, "ext_mem_134415") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134416, "ext_mem_134416") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134417, "ext_mem_134417") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134418, "ext_mem_134418") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134419, "ext_mem_134419") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134420, "ext_mem_134420") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134421, "ext_mem_134421") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134422, "ext_mem_134422") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134423, "ext_mem_134423") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134424, "ext_mem_134424") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134425, "ext_mem_134425") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134426, "ext_mem_134426") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134427, "ext_mem_134427") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134428, "ext_mem_134428") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134429, "ext_mem_134429") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134430, "ext_mem_134430") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134431, "ext_mem_134431") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_134432, "ext_mem_134432") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134531, "mem_out_134531") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134530, "mem_out_134530") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134529, "mem_out_134529") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134528, "mem_out_134528") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134527, "mem_out_134527") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134526, "mem_out_134526") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134525, "mem_out_134525") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134524, "mem_out_134524") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134523, "mem_out_134523") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134522, "mem_out_134522") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134521, "mem_out_134521") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134520, "mem_out_134520") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134519, "mem_out_134519") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134518, "mem_out_134518") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134517, "mem_out_134517") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134516, "mem_out_134516") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134515, "mem_out_134515") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134514, "mem_out_134514") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134513, "mem_out_134513") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134512, "mem_out_134512") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134511, "mem_out_134511") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134510, "mem_out_134510") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134509, "mem_out_134509") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134508, "mem_out_134508") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134507, "mem_out_134507") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134506, "mem_out_134506") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134505, "mem_out_134505") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_135210, struct memblock *mem_out_p_135211, struct memblock *mem_out_p_135212, struct memblock *mem_out_p_135213, struct memblock *mem_out_p_135214, struct memblock *mem_out_p_135215, struct memblock *mem_out_p_135216, struct memblock *mem_out_p_135217, struct memblock *mem_out_p_135218)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_134513;
    
    mem_out_134513.references = NULL;
    
    struct memblock mem_out_134512;
    
    mem_out_134512.references = NULL;
    
    struct memblock mem_out_134511;
    
    mem_out_134511.references = NULL;
    
    struct memblock mem_out_134510;
    
    mem_out_134510.references = NULL;
    
    struct memblock mem_out_134509;
    
    mem_out_134509.references = NULL;
    
    struct memblock mem_out_134508;
    
    mem_out_134508.references = NULL;
    
    struct memblock mem_out_134507;
    
    mem_out_134507.references = NULL;
    
    struct memblock mem_out_134506;
    
    mem_out_134506.references = NULL;
    
    struct memblock mem_out_134505;
    
    mem_out_134505.references = NULL;
    
    struct memblock mem_132596 = ctx->constants->mem_132596;
    struct memblock mem_132597 = ctx->constants->mem_132597;
    struct memblock mem_132598 = ctx->constants->mem_132598;
    struct memblock mem_132599 = ctx->constants->mem_132599;
    struct memblock mem_132600 = ctx->constants->mem_132600;
    struct memblock mem_132601 = ctx->constants->mem_132601;
    struct memblock mem_132602 = ctx->constants->mem_132602;
    struct memblock mem_132603 = ctx->constants->mem_132603;
    struct memblock mem_132604 = ctx->constants->mem_132604;
    
    if (memblock_set(ctx, &mem_out_134505, &mem_132603, "mem_132603") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134506, &mem_132599, "mem_132599") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134507, &mem_132601, "mem_132601") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134508, &mem_132597, "mem_132597") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134509, &mem_132598, "mem_132598") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134510, &mem_132596, "mem_132596") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134511, &mem_132602, "mem_132602") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134512, &mem_132600, "mem_132600") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_134513, &mem_132604, "mem_132604") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135210, &mem_out_134505, "mem_out_134505") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135211, &mem_out_134506, "mem_out_134506") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135212, &mem_out_134507, "mem_out_134507") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135213, &mem_out_134508, "mem_out_134508") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135214, &mem_out_134509, "mem_out_134509") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135215, &mem_out_134510, "mem_out_134510") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135216, &mem_out_134511, "mem_out_134511") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135217, &mem_out_134512, "mem_out_134512") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_135218, &mem_out_134513, "mem_out_134513") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_134513, "mem_out_134513") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134512, "mem_out_134512") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134511, "mem_out_134511") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134510, "mem_out_134510") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134509, "mem_out_134509") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134508, "mem_out_134508") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134507, "mem_out_134507") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134506, "mem_out_134506") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_134505, "mem_out_134505") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_cal_loss(struct futhark_context *ctx, struct futhark_opaque_tup2_f64_arr1d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3)
{
    double prim_out_134506 = 0.0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_134505;
    
    mem_out_134505.references = NULL;
    
    struct memblock mask_mem_132616;
    
    mask_mem_132616.references = NULL;
    
    struct memblock target_mem_132615;
    
    target_mem_132615.references = NULL;
    
    struct memblock tokens_mem_132614;
    
    tokens_mem_132614.references = NULL;
    
    struct memblock wvoc_mem_132613;
    
    wvoc_mem_132613.references = NULL;
    
    struct memblock wval_mem_132612;
    
    wval_mem_132612.references = NULL;
    
    struct memblock wup_mem_132611;
    
    wup_mem_132611.references = NULL;
    
    struct memblock wte_mem_132610;
    
    wte_mem_132610.references = NULL;
    
    struct memblock wqry_mem_132609;
    
    wqry_mem_132609.references = NULL;
    
    struct memblock wpe_mem_132608;
    
    wpe_mem_132608.references = NULL;
    
    struct memblock wout_mem_132607;
    
    wout_mem_132607.references = NULL;
    
    struct memblock wkey_mem_132606;
    
    wkey_mem_132606.references = NULL;
    
    struct memblock wdown_mem_132605;
    
    wdown_mem_132605.references = NULL;
    wdown_mem_132605 = in0->v0->mem;
    wkey_mem_132606 = in0->v1->mem;
    wout_mem_132607 = in0->v2->mem;
    wpe_mem_132608 = in0->v3->mem;
    wqry_mem_132609 = in0->v4->mem;
    wte_mem_132610 = in0->v5->mem;
    wup_mem_132611 = in0->v6->mem;
    wval_mem_132612 = in0->v7->mem;
    wvoc_mem_132613 = in0->v8->mem;
    tokens_mem_132614 = in1->mem;
    target_mem_132615 = in2->mem;
    mask_mem_132616 = in3->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && (((int64_t) 16 == in2->shape[0] && (int64_t) 27 == in2->shape[1]) && ((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_cal_loss(ctx, &mem_out_134505, &prim_out_134506, wdown_mem_132605, wkey_mem_132606, wout_mem_132607, wpe_mem_132608, wqry_mem_132609, wte_mem_132610, wup_mem_132611, wval_mem_132612, wvoc_mem_132613, tokens_mem_132614, target_mem_132615, mask_mem_132616);
        if (ret == 0) {
            struct memblock mem_132596 = ctx->constants->mem_132596;
            struct memblock mem_132597 = ctx->constants->mem_132597;
            struct memblock mem_132598 = ctx->constants->mem_132598;
            struct memblock mem_132599 = ctx->constants->mem_132599;
            struct memblock mem_132600 = ctx->constants->mem_132600;
            struct memblock mem_132601 = ctx->constants->mem_132601;
            struct memblock mem_132602 = ctx->constants->mem_132602;
            struct memblock mem_132603 = ctx->constants->mem_132603;
            struct memblock mem_132604 = ctx->constants->mem_132604;
            
            assert((*out = (struct futhark_opaque_tup2_f64_arr1d_f64 *) malloc(sizeof(struct futhark_opaque_tup2_f64_arr1d_f64))) != NULL);
            (*out)->v0 = prim_out_134506;
            assert(((*out)->v1 = (struct futhark_f64_1d *) malloc(sizeof(struct futhark_f64_1d))) != NULL);
            (*out)->v1->mem = mem_out_134505;
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
    
    struct memblock mem_out_134505;
    
    mem_out_134505.references = NULL;
    
    struct memblock mask_mem_132615;
    
    mask_mem_132615.references = NULL;
    
    struct memblock tokens_mem_132614;
    
    tokens_mem_132614.references = NULL;
    
    struct memblock wvoc_mem_132613;
    
    wvoc_mem_132613.references = NULL;
    
    struct memblock wval_mem_132612;
    
    wval_mem_132612.references = NULL;
    
    struct memblock wup_mem_132611;
    
    wup_mem_132611.references = NULL;
    
    struct memblock wte_mem_132610;
    
    wte_mem_132610.references = NULL;
    
    struct memblock wqry_mem_132609;
    
    wqry_mem_132609.references = NULL;
    
    struct memblock wpe_mem_132608;
    
    wpe_mem_132608.references = NULL;
    
    struct memblock wout_mem_132607;
    
    wout_mem_132607.references = NULL;
    
    struct memblock wkey_mem_132606;
    
    wkey_mem_132606.references = NULL;
    
    struct memblock wdown_mem_132605;
    
    wdown_mem_132605.references = NULL;
    wdown_mem_132605 = in0->v0->mem;
    wkey_mem_132606 = in0->v1->mem;
    wout_mem_132607 = in0->v2->mem;
    wpe_mem_132608 = in0->v3->mem;
    wqry_mem_132609 = in0->v4->mem;
    wte_mem_132610 = in0->v5->mem;
    wup_mem_132611 = in0->v6->mem;
    wval_mem_132612 = in0->v7->mem;
    wvoc_mem_132613 = in0->v8->mem;
    tokens_mem_132614 = in1->mem;
    mask_mem_132615 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_134505, wdown_mem_132605, wkey_mem_132606, wout_mem_132607, wpe_mem_132608, wqry_mem_132609, wte_mem_132610, wup_mem_132611, wval_mem_132612, wvoc_mem_132613, tokens_mem_132614, mask_mem_132615);
        if (ret == 0) {
            struct memblock mem_132596 = ctx->constants->mem_132596;
            struct memblock mem_132597 = ctx->constants->mem_132597;
            struct memblock mem_132598 = ctx->constants->mem_132598;
            struct memblock mem_132599 = ctx->constants->mem_132599;
            struct memblock mem_132600 = ctx->constants->mem_132600;
            struct memblock mem_132601 = ctx->constants->mem_132601;
            struct memblock mem_132602 = ctx->constants->mem_132602;
            struct memblock mem_132603 = ctx->constants->mem_132603;
            struct memblock mem_132604 = ctx->constants->mem_132604;
            
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_134505;
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
    
    struct memblock mem_out_134513;
    
    mem_out_134513.references = NULL;
    
    struct memblock mem_out_134512;
    
    mem_out_134512.references = NULL;
    
    struct memblock mem_out_134511;
    
    mem_out_134511.references = NULL;
    
    struct memblock mem_out_134510;
    
    mem_out_134510.references = NULL;
    
    struct memblock mem_out_134509;
    
    mem_out_134509.references = NULL;
    
    struct memblock mem_out_134508;
    
    mem_out_134508.references = NULL;
    
    struct memblock mem_out_134507;
    
    mem_out_134507.references = NULL;
    
    struct memblock mem_out_134506;
    
    mem_out_134506.references = NULL;
    
    struct memblock mem_out_134505;
    
    mem_out_134505.references = NULL;
    
    struct memblock wvoc_mem_132613;
    
    wvoc_mem_132613.references = NULL;
    
    struct memblock wdown_mem_132612;
    
    wdown_mem_132612.references = NULL;
    
    struct memblock wup_mem_132611;
    
    wup_mem_132611.references = NULL;
    
    struct memblock wout_mem_132610;
    
    wout_mem_132610.references = NULL;
    
    struct memblock wval_mem_132609;
    
    wval_mem_132609.references = NULL;
    
    struct memblock wkey_mem_132608;
    
    wkey_mem_132608.references = NULL;
    
    struct memblock wqry_mem_132607;
    
    wqry_mem_132607.references = NULL;
    
    struct memblock wpe_mem_132606;
    
    wpe_mem_132606.references = NULL;
    
    struct memblock wte_mem_132605;
    
    wte_mem_132605.references = NULL;
    wte_mem_132605 = in0->mem;
    wpe_mem_132606 = in1->mem;
    wqry_mem_132607 = in2->mem;
    wkey_mem_132608 = in3->mem;
    wval_mem_132609 = in4->mem;
    wout_mem_132610 = in5->mem;
    wup_mem_132611 = in6->mem;
    wdown_mem_132612 = in7->mem;
    wvoc_mem_132613 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_134505, &mem_out_134506, &mem_out_134507, &mem_out_134508, &mem_out_134509, &mem_out_134510, &mem_out_134511, &mem_out_134512, &mem_out_134513, wte_mem_132605, wpe_mem_132606, wqry_mem_132607, wkey_mem_132608, wval_mem_132609, wout_mem_132610, wup_mem_132611, wdown_mem_132612, wvoc_mem_132613);
        if (ret == 0) {
            struct memblock mem_132596 = ctx->constants->mem_132596;
            struct memblock mem_132597 = ctx->constants->mem_132597;
            struct memblock mem_132598 = ctx->constants->mem_132598;
            struct memblock mem_132599 = ctx->constants->mem_132599;
            struct memblock mem_132600 = ctx->constants->mem_132600;
            struct memblock mem_132601 = ctx->constants->mem_132601;
            struct memblock mem_132602 = ctx->constants->mem_132602;
            struct memblock mem_132603 = ctx->constants->mem_132603;
            struct memblock mem_132604 = ctx->constants->mem_132604;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_134505;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_134506;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_134507;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_134508;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_134509;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_134510;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_134511;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_134512;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_134513;
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
    
    struct memblock mem_out_134531;
    
    mem_out_134531.references = NULL;
    
    struct memblock mem_out_134530;
    
    mem_out_134530.references = NULL;
    
    struct memblock mem_out_134529;
    
    mem_out_134529.references = NULL;
    
    struct memblock mem_out_134528;
    
    mem_out_134528.references = NULL;
    
    struct memblock mem_out_134527;
    
    mem_out_134527.references = NULL;
    
    struct memblock mem_out_134526;
    
    mem_out_134526.references = NULL;
    
    struct memblock mem_out_134525;
    
    mem_out_134525.references = NULL;
    
    struct memblock mem_out_134524;
    
    mem_out_134524.references = NULL;
    
    struct memblock mem_out_134523;
    
    mem_out_134523.references = NULL;
    
    struct memblock mem_out_134522;
    
    mem_out_134522.references = NULL;
    
    struct memblock mem_out_134521;
    
    mem_out_134521.references = NULL;
    
    struct memblock mem_out_134520;
    
    mem_out_134520.references = NULL;
    
    struct memblock mem_out_134519;
    
    mem_out_134519.references = NULL;
    
    struct memblock mem_out_134518;
    
    mem_out_134518.references = NULL;
    
    struct memblock mem_out_134517;
    
    mem_out_134517.references = NULL;
    
    struct memblock mem_out_134516;
    
    mem_out_134516.references = NULL;
    
    struct memblock mem_out_134515;
    
    mem_out_134515.references = NULL;
    
    struct memblock mem_out_134514;
    
    mem_out_134514.references = NULL;
    
    struct memblock mem_out_134513;
    
    mem_out_134513.references = NULL;
    
    struct memblock mem_out_134512;
    
    mem_out_134512.references = NULL;
    
    struct memblock mem_out_134511;
    
    mem_out_134511.references = NULL;
    
    struct memblock mem_out_134510;
    
    mem_out_134510.references = NULL;
    
    struct memblock mem_out_134509;
    
    mem_out_134509.references = NULL;
    
    struct memblock mem_out_134508;
    
    mem_out_134508.references = NULL;
    
    struct memblock mem_out_134507;
    
    mem_out_134507.references = NULL;
    
    struct memblock mem_out_134506;
    
    mem_out_134506.references = NULL;
    
    struct memblock mem_out_134505;
    
    mem_out_134505.references = NULL;
    
    struct memblock seqs_mem_132634;
    
    seqs_mem_132634.references = NULL;
    
    struct memblock dls_mem_132633;
    
    dls_mem_132633.references = NULL;
    
    struct memblock masks_mem_132632;
    
    masks_mem_132632.references = NULL;
    
    struct memblock wvoc_mem_132631;
    
    wvoc_mem_132631.references = NULL;
    
    struct memblock wval_mem_132630;
    
    wval_mem_132630.references = NULL;
    
    struct memblock wup_mem_132629;
    
    wup_mem_132629.references = NULL;
    
    struct memblock wte_mem_132628;
    
    wte_mem_132628.references = NULL;
    
    struct memblock wqry_mem_132627;
    
    wqry_mem_132627.references = NULL;
    
    struct memblock wpe_mem_132626;
    
    wpe_mem_132626.references = NULL;
    
    struct memblock wout_mem_132625;
    
    wout_mem_132625.references = NULL;
    
    struct memblock wkey_mem_132624;
    
    wkey_mem_132624.references = NULL;
    
    struct memblock wdown_mem_132623;
    
    wdown_mem_132623.references = NULL;
    
    struct memblock wvoc_mem_132622;
    
    wvoc_mem_132622.references = NULL;
    
    struct memblock wval_mem_132621;
    
    wval_mem_132621.references = NULL;
    
    struct memblock wup_mem_132620;
    
    wup_mem_132620.references = NULL;
    
    struct memblock wte_mem_132619;
    
    wte_mem_132619.references = NULL;
    
    struct memblock wqry_mem_132618;
    
    wqry_mem_132618.references = NULL;
    
    struct memblock wpe_mem_132617;
    
    wpe_mem_132617.references = NULL;
    
    struct memblock wout_mem_132616;
    
    wout_mem_132616.references = NULL;
    
    struct memblock wkey_mem_132615;
    
    wkey_mem_132615.references = NULL;
    
    struct memblock wdown_mem_132614;
    
    wdown_mem_132614.references = NULL;
    
    struct memblock wvoc_mem_132613;
    
    wvoc_mem_132613.references = NULL;
    
    struct memblock wval_mem_132612;
    
    wval_mem_132612.references = NULL;
    
    struct memblock wup_mem_132611;
    
    wup_mem_132611.references = NULL;
    
    struct memblock wte_mem_132610;
    
    wte_mem_132610.references = NULL;
    
    struct memblock wqry_mem_132609;
    
    wqry_mem_132609.references = NULL;
    
    struct memblock wpe_mem_132608;
    
    wpe_mem_132608.references = NULL;
    
    struct memblock wout_mem_132607;
    
    wout_mem_132607.references = NULL;
    
    struct memblock wkey_mem_132606;
    
    wkey_mem_132606.references = NULL;
    
    struct memblock wdown_mem_132605;
    
    wdown_mem_132605.references = NULL;
    wdown_mem_132605 = in0->v0->mem;
    wkey_mem_132606 = in0->v1->mem;
    wout_mem_132607 = in0->v2->mem;
    wpe_mem_132608 = in0->v3->mem;
    wqry_mem_132609 = in0->v4->mem;
    wte_mem_132610 = in0->v5->mem;
    wup_mem_132611 = in0->v6->mem;
    wval_mem_132612 = in0->v7->mem;
    wvoc_mem_132613 = in0->v8->mem;
    wdown_mem_132614 = in1->v0->mem;
    wkey_mem_132615 = in1->v1->mem;
    wout_mem_132616 = in1->v2->mem;
    wpe_mem_132617 = in1->v3->mem;
    wqry_mem_132618 = in1->v4->mem;
    wte_mem_132619 = in1->v5->mem;
    wup_mem_132620 = in1->v6->mem;
    wval_mem_132621 = in1->v7->mem;
    wvoc_mem_132622 = in1->v8->mem;
    wdown_mem_132623 = in2->v0->mem;
    wkey_mem_132624 = in2->v1->mem;
    wout_mem_132625 = in2->v2->mem;
    wpe_mem_132626 = in2->v3->mem;
    wqry_mem_132627 = in2->v4->mem;
    wte_mem_132628 = in2->v5->mem;
    wup_mem_132629 = in2->v6->mem;
    wval_mem_132630 = in2->v7->mem;
    wvoc_mem_132631 = in2->v8->mem;
    masks_mem_132632 = in3->mem;
    dls_mem_132633 = in4->mem;
    seqs_mem_132634 = in5->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && (((int64_t) 500 == in3->shape[0] && ((int64_t) 16 == in3->shape[1] && (int64_t) 16 == in3->shape[2])) && ((int64_t) 500 == in4->shape[0] && ((int64_t) 500 == in5->shape[0] && (int64_t) 16 == in5->shape[1]))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_134505, &mem_out_134506, &mem_out_134507, &mem_out_134508, &mem_out_134509, &mem_out_134510, &mem_out_134511, &mem_out_134512, &mem_out_134513, &mem_out_134514, &mem_out_134515, &mem_out_134516, &mem_out_134517, &mem_out_134518, &mem_out_134519, &mem_out_134520, &mem_out_134521, &mem_out_134522, &mem_out_134523, &mem_out_134524, &mem_out_134525, &mem_out_134526, &mem_out_134527, &mem_out_134528, &mem_out_134529, &mem_out_134530, &mem_out_134531, wdown_mem_132605, wkey_mem_132606, wout_mem_132607, wpe_mem_132608, wqry_mem_132609, wte_mem_132610, wup_mem_132611, wval_mem_132612, wvoc_mem_132613, wdown_mem_132614, wkey_mem_132615, wout_mem_132616, wpe_mem_132617, wqry_mem_132618, wte_mem_132619, wup_mem_132620, wval_mem_132621, wvoc_mem_132622, wdown_mem_132623, wkey_mem_132624, wout_mem_132625, wpe_mem_132626, wqry_mem_132627, wte_mem_132628, wup_mem_132629, wval_mem_132630, wvoc_mem_132631, masks_mem_132632, dls_mem_132633, seqs_mem_132634);
        if (ret == 0) {
            struct memblock mem_132596 = ctx->constants->mem_132596;
            struct memblock mem_132597 = ctx->constants->mem_132597;
            struct memblock mem_132598 = ctx->constants->mem_132598;
            struct memblock mem_132599 = ctx->constants->mem_132599;
            struct memblock mem_132600 = ctx->constants->mem_132600;
            struct memblock mem_132601 = ctx->constants->mem_132601;
            struct memblock mem_132602 = ctx->constants->mem_132602;
            struct memblock mem_132603 = ctx->constants->mem_132603;
            struct memblock mem_132604 = ctx->constants->mem_132604;
            
            assert((*out = (struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_134505;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_134506;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_134507;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_134508;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_134509;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_134510;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_134511;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_134512;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_134513;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v9->mem = mem_out_134514;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v10->mem = mem_out_134515;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v11->mem = mem_out_134516;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v12->mem = mem_out_134517;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v13->mem = mem_out_134518;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v14->mem = mem_out_134519;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v15->mem = mem_out_134520;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v16->mem = mem_out_134521;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v17->mem = mem_out_134522;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v18->mem = mem_out_134523;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v19->mem = mem_out_134524;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v20->mem = mem_out_134525;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v21->mem = mem_out_134526;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v22->mem = mem_out_134527;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v23->mem = mem_out_134528;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v24->mem = mem_out_134529;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v25->mem = mem_out_134530;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v26->mem = mem_out_134531;
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
    
    struct memblock mem_out_134513;
    
    mem_out_134513.references = NULL;
    
    struct memblock mem_out_134512;
    
    mem_out_134512.references = NULL;
    
    struct memblock mem_out_134511;
    
    mem_out_134511.references = NULL;
    
    struct memblock mem_out_134510;
    
    mem_out_134510.references = NULL;
    
    struct memblock mem_out_134509;
    
    mem_out_134509.references = NULL;
    
    struct memblock mem_out_134508;
    
    mem_out_134508.references = NULL;
    
    struct memblock mem_out_134507;
    
    mem_out_134507.references = NULL;
    
    struct memblock mem_out_134506;
    
    mem_out_134506.references = NULL;
    
    struct memblock mem_out_134505;
    
    mem_out_134505.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_134505, &mem_out_134506, &mem_out_134507, &mem_out_134508, &mem_out_134509, &mem_out_134510, &mem_out_134511, &mem_out_134512, &mem_out_134513);
        if (ret == 0) {
            struct memblock mem_132596 = ctx->constants->mem_132596;
            struct memblock mem_132597 = ctx->constants->mem_132597;
            struct memblock mem_132598 = ctx->constants->mem_132598;
            struct memblock mem_132599 = ctx->constants->mem_132599;
            struct memblock mem_132600 = ctx->constants->mem_132600;
            struct memblock mem_132601 = ctx->constants->mem_132601;
            struct memblock mem_132602 = ctx->constants->mem_132602;
            struct memblock mem_132603 = ctx->constants->mem_132603;
            struct memblock mem_132604 = ctx->constants->mem_132604;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_134505;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_134506;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_134507;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_134508;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_134509;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_134510;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_134511;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_134512;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_134513;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
