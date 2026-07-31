
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
int futhark_entry_train(struct futhark_context *ctx, struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const int64_t in0, const struct futhark_opaque_params *in1, const struct futhark_opaque_params *in2, const struct futhark_opaque_params *in3, const struct futhark_f64_3d *in4, const struct futhark_i64_1d *in5, const struct futhark_i64_2d *in6);
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
const struct type *train_in_types[] = {&type_i64, &type_params, &type_params, &type_params, &type_ZMZNZMZNZMZNf64, &type_ZMZNi64, &type_ZMZNZMZNi64, NULL};
bool train_in_unique[] = {false, false, false, false, false, false, false};
const char *train_tuning_params[] = {NULL};
const char *train_attrs[] = {NULL};
int call_train(struct futhark_context *ctx, void *out, void **ins)
{
    int64_t in0 = *(int64_t *) ins[0];
    struct futhark_opaque_params * in1 = *(struct futhark_opaque_params * *) ins[1];
    struct futhark_opaque_params * in2 = *(struct futhark_opaque_params * *) ins[2];
    struct futhark_opaque_params * in3 = *(struct futhark_opaque_params * *) ins[3];
    struct futhark_f64_3d * in4 = *(struct futhark_f64_3d * *) ins[4];
    struct futhark_i64_1d * in5 = *(struct futhark_i64_1d * *) ins[5];
    struct futhark_i64_2d * in6 = *(struct futhark_i64_2d * *) ins[6];
    
    return futhark_entry_train(ctx, out, in0, in1, in2, in3, in4, in5, in6);
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
    struct memblock mem_152772;
    struct memblock mem_152773;
    struct memblock mem_152774;
    struct memblock mem_152775;
    struct memblock mem_152776;
    struct memblock mem_152777;
    struct memblock mem_152778;
    struct memblock mem_152779;
    struct memblock mem_152780;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_12919(struct futhark_context *ctx, struct memblock *mem_out_p_155690, struct memblock *mem_out_p_155691, struct memblock *mem_out_p_155692, struct memblock w_mem_152781, struct memblock mw_mem_152782, struct memblock vw_mem_152783, struct memblock dw_mem_152784, int64_t n_110599, int64_t m_110600, int64_t step_110605, double lt_r_110606, double beta1_110607, double beta2_110608, double eps_adam_110609);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_12920(struct futhark_context *ctx, struct memblock *mem_out_p_155695, struct memblock *mem_out_p_155696, struct memblock *mem_out_p_155697, struct memblock w_mem_152781, struct memblock mw_mem_152782, struct memblock vw_mem_152783, struct memblock dw_mem_152784, int64_t n_111788, int64_t m_111789, int64_t step_111794, double lt_r_111795, double beta1_111796, double beta2_111797, double eps_adam_111798);
FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_155700, double *out_prim_out_155701, struct memblock wdown_mem_152781, struct memblock wkey_mem_152782, struct memblock wout_mem_152783, struct memblock wpe_mem_152784, struct memblock wqry_mem_152785, struct memblock wte_mem_152786, struct memblock wup_mem_152787, struct memblock wval_mem_152788, struct memblock wvoc_mem_152789, struct memblock tokens_mem_152790, struct memblock target_mem_152791, struct memblock mask_mem_152792);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_155759, struct memblock wdown_mem_152781, struct memblock wkey_mem_152782, struct memblock wout_mem_152783, struct memblock wpe_mem_152784, struct memblock wqry_mem_152785, struct memblock wte_mem_152786, struct memblock wup_mem_152787, struct memblock wval_mem_152788, struct memblock wvoc_mem_152789, struct memblock tokens_mem_152790, struct memblock mask_mem_152791);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_155816, struct memblock *mem_out_p_155817, struct memblock *mem_out_p_155818, struct memblock *mem_out_p_155819, struct memblock *mem_out_p_155820, struct memblock *mem_out_p_155821, struct memblock *mem_out_p_155822, struct memblock *mem_out_p_155823, struct memblock *mem_out_p_155824, struct memblock wte_mem_152781, struct memblock wpe_mem_152782, struct memblock wqry_mem_152783, struct memblock wkey_mem_152784, struct memblock wval_mem_152785, struct memblock wout_mem_152786, struct memblock wup_mem_152787, struct memblock wdown_mem_152788, struct memblock wvoc_mem_152789);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_155825, struct memblock *mem_out_p_155826, struct memblock *mem_out_p_155827, struct memblock *mem_out_p_155828, struct memblock *mem_out_p_155829, struct memblock *mem_out_p_155830, struct memblock *mem_out_p_155831, struct memblock *mem_out_p_155832, struct memblock *mem_out_p_155833, struct memblock *mem_out_p_155834, struct memblock *mem_out_p_155835, struct memblock *mem_out_p_155836, struct memblock *mem_out_p_155837, struct memblock *mem_out_p_155838, struct memblock *mem_out_p_155839, struct memblock *mem_out_p_155840, struct memblock *mem_out_p_155841, struct memblock *mem_out_p_155842, struct memblock *mem_out_p_155843, struct memblock *mem_out_p_155844, struct memblock *mem_out_p_155845, struct memblock *mem_out_p_155846, struct memblock *mem_out_p_155847, struct memblock *mem_out_p_155848, struct memblock *mem_out_p_155849, struct memblock *mem_out_p_155850, struct memblock *mem_out_p_155851, struct memblock wdown_mem_152781, struct memblock wkey_mem_152782, struct memblock wout_mem_152783, struct memblock wpe_mem_152784, struct memblock wqry_mem_152785, struct memblock wte_mem_152786, struct memblock wup_mem_152787, struct memblock wval_mem_152788, struct memblock wvoc_mem_152789, struct memblock wdown_mem_152790, struct memblock wkey_mem_152791, struct memblock wout_mem_152792, struct memblock wpe_mem_152793, struct memblock wqry_mem_152794, struct memblock wte_mem_152795, struct memblock wup_mem_152796, struct memblock wval_mem_152797, struct memblock wvoc_mem_152798, struct memblock wdown_mem_152799, struct memblock wkey_mem_152800, struct memblock wout_mem_152801, struct memblock wpe_mem_152802, struct memblock wqry_mem_152803, struct memblock wte_mem_152804, struct memblock wup_mem_152805, struct memblock wval_mem_152806, struct memblock wvoc_mem_152807, struct memblock masks_mem_152808, struct memblock dls_mem_152809, struct memblock seqs_mem_152810, int64_t num_steps_112428);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_156099, struct memblock *mem_out_p_156100, struct memblock *mem_out_p_156101, struct memblock *mem_out_p_156102, struct memblock *mem_out_p_156103, struct memblock *mem_out_p_156104, struct memblock *mem_out_p_156105, struct memblock *mem_out_p_156106, struct memblock *mem_out_p_156107);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_152772 (ctx->constants->mem_152772)
    #define mem_152773 (ctx->constants->mem_152773)
    #define mem_152774 (ctx->constants->mem_152774)
    #define mem_152775 (ctx->constants->mem_152775)
    #define mem_152776 (ctx->constants->mem_152776)
    #define mem_152777 (ctx->constants->mem_152777)
    #define mem_152778 (ctx->constants->mem_152778)
    #define mem_152779 (ctx->constants->mem_152779)
    #define mem_152780 (ctx->constants->mem_152780)
    mem_152772.references = NULL;
    mem_152773.references = NULL;
    mem_152774.references = NULL;
    mem_152775.references = NULL;
    mem_152776.references = NULL;
    mem_152777.references = NULL;
    mem_152778.references = NULL;
    mem_152779.references = NULL;
    mem_152780.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152772, (int64_t) 3456, "mem_152772")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_155672 = 0; nest_i_155672 < (int64_t) 27; nest_i_155672++) {
        for (int64_t nest_i_155673 = 0; nest_i_155673 < (int64_t) 16; nest_i_155673++) {
            ((double *) mem_152772.mem)[nest_i_155672 * (int64_t) 16 + nest_i_155673] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152773, (int64_t) 2048, "mem_152773")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_155674 = 0; nest_i_155674 < (int64_t) 16; nest_i_155674++) {
        for (int64_t nest_i_155675 = 0; nest_i_155675 < (int64_t) 16; nest_i_155675++) {
            ((double *) mem_152773.mem)[nest_i_155674 * (int64_t) 16 + nest_i_155675] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152774, (int64_t) 2048, "mem_152774")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_155676 = 0; nest_i_155676 < (int64_t) 16; nest_i_155676++) {
        for (int64_t nest_i_155677 = 0; nest_i_155677 < (int64_t) 16; nest_i_155677++) {
            ((double *) mem_152774.mem)[nest_i_155676 * (int64_t) 16 + nest_i_155677] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152775, (int64_t) 2048, "mem_152775")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_155678 = 0; nest_i_155678 < (int64_t) 16; nest_i_155678++) {
        for (int64_t nest_i_155679 = 0; nest_i_155679 < (int64_t) 16; nest_i_155679++) {
            ((double *) mem_152775.mem)[nest_i_155678 * (int64_t) 16 + nest_i_155679] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152776, (int64_t) 2048, "mem_152776")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_155680 = 0; nest_i_155680 < (int64_t) 16; nest_i_155680++) {
        for (int64_t nest_i_155681 = 0; nest_i_155681 < (int64_t) 16; nest_i_155681++) {
            ((double *) mem_152776.mem)[nest_i_155680 * (int64_t) 16 + nest_i_155681] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152777, (int64_t) 2048, "mem_152777")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_155682 = 0; nest_i_155682 < (int64_t) 16; nest_i_155682++) {
        for (int64_t nest_i_155683 = 0; nest_i_155683 < (int64_t) 16; nest_i_155683++) {
            ((double *) mem_152777.mem)[nest_i_155682 * (int64_t) 16 + nest_i_155683] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152778, (int64_t) 8192, "mem_152778")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_155684 = 0; nest_i_155684 < (int64_t) 64; nest_i_155684++) {
        for (int64_t nest_i_155685 = 0; nest_i_155685 < (int64_t) 16; nest_i_155685++) {
            ((double *) mem_152778.mem)[nest_i_155684 * (int64_t) 16 + nest_i_155685] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152779, (int64_t) 8192, "mem_152779")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_155686 = 0; nest_i_155686 < (int64_t) 16; nest_i_155686++) {
        for (int64_t nest_i_155687 = 0; nest_i_155687 < (int64_t) 64; nest_i_155687++) {
            ((double *) mem_152779.mem)[nest_i_155686 * (int64_t) 64 + nest_i_155687] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152780, (int64_t) 3456, "mem_152780")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_155688 = 0; nest_i_155688 < (int64_t) 27; nest_i_155688++) {
        for (int64_t nest_i_155689 = 0; nest_i_155689 < (int64_t) 16; nest_i_155689++) {
            ((double *) mem_152780.mem)[nest_i_155688 * (int64_t) 16 + nest_i_155689] = 0.0;
        }
    }
    #undef mem_152772
    #undef mem_152773
    #undef mem_152774
    #undef mem_152775
    #undef mem_152776
    #undef mem_152777
    #undef mem_152778
    #undef mem_152779
    #undef mem_152780
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_152772, "ctx->constants->mem_152772") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_152773, "ctx->constants->mem_152773") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_152774, "ctx->constants->mem_152774") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_152775, "ctx->constants->mem_152775") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_152776, "ctx->constants->mem_152776") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_152777, "ctx->constants->mem_152777") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_152778, "ctx->constants->mem_152778") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_152779, "ctx->constants->mem_152779") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_152780, "ctx->constants->mem_152780") != 0)
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_12919(struct futhark_context *ctx, struct memblock *mem_out_p_155690, struct memblock *mem_out_p_155691, struct memblock *mem_out_p_155692, struct memblock w_mem_152781, struct memblock mw_mem_152782, struct memblock vw_mem_152783, struct memblock dw_mem_152784, int64_t n_110599, int64_t m_110600, int64_t step_110605, double lt_r_110606, double beta1_110607, double beta2_110608, double eps_adam_110609)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_152825_cached_sizze_155693 = 0;
    unsigned char *mem_152825 = NULL;
    int64_t mem_152828_cached_sizze_155694 = 0;
    unsigned char *mem_152828 = NULL;
    struct memblock mem_152863;
    
    mem_152863.references = NULL;
    
    struct memblock mem_152790;
    
    mem_152790.references = NULL;
    
    struct memblock mem_152787;
    
    mem_152787.references = NULL;
    
    struct memblock mem_out_155271;
    
    mem_out_155271.references = NULL;
    
    struct memblock mem_out_155270;
    
    mem_out_155270.references = NULL;
    
    struct memblock mem_out_155269;
    
    mem_out_155269.references = NULL;
    
    struct memblock mem_152772 = ctx->constants->mem_152772;
    struct memblock mem_152773 = ctx->constants->mem_152773;
    struct memblock mem_152774 = ctx->constants->mem_152774;
    struct memblock mem_152775 = ctx->constants->mem_152775;
    struct memblock mem_152776 = ctx->constants->mem_152776;
    struct memblock mem_152777 = ctx->constants->mem_152777;
    struct memblock mem_152778 = ctx->constants->mem_152778;
    struct memblock mem_152779 = ctx->constants->mem_152779;
    struct memblock mem_152780 = ctx->constants->mem_152780;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_152785 = (int64_t) 8 * n_110599;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_152786 = m_110600 * binop_x_152785;
    
    // futhark/microgpt.fut:468:28-35
    
    double zt_lhs_114943 = 1.0 - beta1_110607;
    
    // futhark/microgpt.fut:470:28-35
    
    double zt_lhs_114991 = 1.0 - beta2_110608;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152787, bytes_152786, "mem_152787")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152790, bytes_152786, "mem_152790")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151640 = 0; i_151640 < n_110599; i_151640++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151633 = 0; i_151633 < m_110600; i_151633++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_141488 = ((double *) mw_mem_152782.mem)[i_151640 * m_110600 + i_151633];
            
            // futhark/microgpt.fut:468:11-21
            
            double zp_lhs_141489 = beta1_110607 * zt_rhs_141488;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_141490 = ((double *) dw_mem_152784.mem)[i_151640 * m_110600 + i_151633];
            
            // futhark/microgpt.fut:468:37-47
            
            double zp_rhs_141491 = zt_lhs_114943 * zt_rhs_141490;
            
            // futhark/microgpt.fut:468:22-47
            
            double lifted_lambda_res_141492 = zp_lhs_141489 + zp_rhs_141491;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_141499 = ((double *) vw_mem_152783.mem)[i_151640 * m_110600 + i_151633];
            
            // futhark/microgpt.fut:470:11-21
            
            double zp_lhs_141500 = beta2_110608 * zt_rhs_141499;
            
            // futhark/microgpt.fut:470:37-47
            
            double zt_lhs_141502 = zt_lhs_114991 * zt_rhs_141490;
            
            // futhark/microgpt.fut:470:48-58
            
            double zp_rhs_141503 = zt_rhs_141490 * zt_lhs_141502;
            
            // futhark/microgpt.fut:470:22-58
            
            double lifted_lambda_res_141504 = zp_lhs_141500 + zp_rhs_141503;
            
            ((double *) mem_152787.mem)[i_151640 * m_110600 + i_151633] = lifted_lambda_res_141504;
            ((double *) mem_152790.mem)[i_151640 * m_110600 + i_151633] = lifted_lambda_res_141492;
        }
    }
    // futhark/microgpt.fut:56:26-45
    
    double i64_res_115739 = sitofp_i64_f64(step_110605);
    
    // futhark/microgpt.fut:472:55-58
    
    double ztzt_rhs_115740 = 1.0 + i64_res_115739;
    
    // futhark/microgpt.fut:472:31-58
    
    double zm_rhs_115741 = fpow64(beta1_110607, ztzt_rhs_115740);
    
    // futhark/microgpt.fut:472:23-58
    
    double zs_rhs_115742 = 1.0 - zm_rhs_115741;
    
    // futhark/microgpt.fut:474:32-59
    
    double zm_rhs_115781 = fpow64(beta2_110608, ztzt_rhs_115740);
    
    // futhark/microgpt.fut:474:23-59
    
    double zs_rhs_115782 = 1.0 - zm_rhs_115781;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_152825_cached_sizze_155693 < bytes_152786) {
        err = lexical_realloc(ctx, &mem_152825, &mem_152825_cached_sizze_155693, bytes_152786);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152828_cached_sizze_155694 < bytes_152786) {
        err = lexical_realloc(ctx, &mem_152828, &mem_152828_cached_sizze_155694, bytes_152786);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151654 = 0; i_151654 < n_110599; i_151654++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151647 = 0; i_151647 < m_110600; i_151647++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_141524 = ((double *) mem_152790.mem)[i_151654 * m_110600 + i_151647];
            
            // futhark/microgpt.fut:472:18-58
            
            double lifted_lambda_res_141525 = zs_lhs_141524 / zs_rhs_115742;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_141532 = ((double *) mem_152787.mem)[i_151654 * m_110600 + i_151647];
            
            // futhark/microgpt.fut:474:18-59
            
            double lifted_lambda_res_141533 = zs_lhs_141532 / zs_rhs_115782;
            
            ((double *) mem_152825)[i_151654 * m_110600 + i_151647] = lifted_lambda_res_141533;
            ((double *) mem_152828)[i_151654 * m_110600 + i_151647] = lifted_lambda_res_141525;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152863, bytes_152786, "mem_152863")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151663 = 0; i_151663 < n_110599; i_151663++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151659 = 0; i_151659 < m_110600; i_151659++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_115054 = ((double *) w_mem_152781.mem)[i_151663 * m_110600 + i_151659];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_115055 = ((double *) mem_152828)[i_151663 * m_110600 + i_151659];
            
            // futhark/microgpt.fut:476:21-34
            
            double zs_lhs_115056 = lt_r_110606 * zt_rhs_115055;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_115057 = ((double *) mem_152825)[i_151663 * m_110600 + i_151659];
            
            // futhark/microgpt.fut:476:51-57
            
            double zp_lhs_115058 = fpow64(ztzt_lhs_115057, 0.5);
            
            // futhark/microgpt.fut:476:59-69
            
            double zs_rhs_115059 = eps_adam_110609 + zp_lhs_115058;
            
            // futhark/microgpt.fut:476:35-69
            
            double zm_rhs_115060 = zs_lhs_115056 / zs_rhs_115059;
            
            // futhark/microgpt.fut:476:13-69
            
            double lifted_lambda_res_115061 = zm_lhs_115054 - zm_rhs_115060;
            
            ((double *) mem_152863.mem)[i_151663 * m_110600 + i_151659] = lifted_lambda_res_115061;
        }
    }
    if (memblock_set(ctx, &mem_out_155269, &mem_152863, "mem_152863") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155270, &mem_152790, "mem_152790") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155271, &mem_152787, "mem_152787") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155690, &mem_out_155269, "mem_out_155269") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155691, &mem_out_155270, "mem_out_155270") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155692, &mem_out_155271, "mem_out_155271") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_152825);
        free(mem_152828);
        if (memblock_unref(ctx, &mem_152863, "mem_152863") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_152790, "mem_152790") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_152787, "mem_152787") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155271, "mem_out_155271") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155270, "mem_out_155270") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155269, "mem_out_155269") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_12920(struct futhark_context *ctx, struct memblock *mem_out_p_155695, struct memblock *mem_out_p_155696, struct memblock *mem_out_p_155697, struct memblock w_mem_152781, struct memblock mw_mem_152782, struct memblock vw_mem_152783, struct memblock dw_mem_152784, int64_t n_111788, int64_t m_111789, int64_t step_111794, double lt_r_111795, double beta1_111796, double beta2_111797, double eps_adam_111798)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_152825_cached_sizze_155698 = 0;
    unsigned char *mem_152825 = NULL;
    int64_t mem_152828_cached_sizze_155699 = 0;
    unsigned char *mem_152828 = NULL;
    struct memblock mem_152863;
    
    mem_152863.references = NULL;
    
    struct memblock mem_152790;
    
    mem_152790.references = NULL;
    
    struct memblock mem_152787;
    
    mem_152787.references = NULL;
    
    struct memblock mem_out_155271;
    
    mem_out_155271.references = NULL;
    
    struct memblock mem_out_155270;
    
    mem_out_155270.references = NULL;
    
    struct memblock mem_out_155269;
    
    mem_out_155269.references = NULL;
    
    struct memblock mem_152772 = ctx->constants->mem_152772;
    struct memblock mem_152773 = ctx->constants->mem_152773;
    struct memblock mem_152774 = ctx->constants->mem_152774;
    struct memblock mem_152775 = ctx->constants->mem_152775;
    struct memblock mem_152776 = ctx->constants->mem_152776;
    struct memblock mem_152777 = ctx->constants->mem_152777;
    struct memblock mem_152778 = ctx->constants->mem_152778;
    struct memblock mem_152779 = ctx->constants->mem_152779;
    struct memblock mem_152780 = ctx->constants->mem_152780;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_152785 = (int64_t) 8 * n_111788;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_152786 = m_111789 * binop_x_152785;
    
    // futhark/microgpt.fut:468:28-35
    
    double zt_lhs_114943 = 1.0 - beta1_111796;
    
    // futhark/microgpt.fut:470:28-35
    
    double zt_lhs_114991 = 1.0 - beta2_111797;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152787, bytes_152786, "mem_152787")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152790, bytes_152786, "mem_152790")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151640 = 0; i_151640 < n_111788; i_151640++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151633 = 0; i_151633 < m_111789; i_151633++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_141488 = ((double *) mw_mem_152782.mem)[i_151640 * m_111789 + i_151633];
            
            // futhark/microgpt.fut:468:11-21
            
            double zp_lhs_141489 = beta1_111796 * zt_rhs_141488;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_141490 = ((double *) dw_mem_152784.mem)[i_151640 * m_111789 + i_151633];
            
            // futhark/microgpt.fut:468:37-47
            
            double zp_rhs_141491 = zt_lhs_114943 * zt_rhs_141490;
            
            // futhark/microgpt.fut:468:22-47
            
            double lifted_lambda_res_141492 = zp_lhs_141489 + zp_rhs_141491;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_141499 = ((double *) vw_mem_152783.mem)[i_151640 * m_111789 + i_151633];
            
            // futhark/microgpt.fut:470:11-21
            
            double zp_lhs_141500 = beta2_111797 * zt_rhs_141499;
            
            // futhark/microgpt.fut:470:37-47
            
            double zt_lhs_141502 = zt_lhs_114991 * zt_rhs_141490;
            
            // futhark/microgpt.fut:470:48-58
            
            double zp_rhs_141503 = zt_rhs_141490 * zt_lhs_141502;
            
            // futhark/microgpt.fut:470:22-58
            
            double lifted_lambda_res_141504 = zp_lhs_141500 + zp_rhs_141503;
            
            ((double *) mem_152787.mem)[i_151640 * m_111789 + i_151633] = lifted_lambda_res_141504;
            ((double *) mem_152790.mem)[i_151640 * m_111789 + i_151633] = lifted_lambda_res_141492;
        }
    }
    // futhark/microgpt.fut:56:26-45
    
    double i64_res_115739 = sitofp_i64_f64(step_111794);
    
    // futhark/microgpt.fut:472:55-58
    
    double ztzt_rhs_115740 = 1.0 + i64_res_115739;
    
    // futhark/microgpt.fut:472:31-58
    
    double zm_rhs_115741 = fpow64(beta1_111796, ztzt_rhs_115740);
    
    // futhark/microgpt.fut:472:23-58
    
    double zs_rhs_115742 = 1.0 - zm_rhs_115741;
    
    // futhark/microgpt.fut:474:32-59
    
    double zm_rhs_115781 = fpow64(beta2_111797, ztzt_rhs_115740);
    
    // futhark/microgpt.fut:474:23-59
    
    double zs_rhs_115782 = 1.0 - zm_rhs_115781;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_152825_cached_sizze_155698 < bytes_152786) {
        err = lexical_realloc(ctx, &mem_152825, &mem_152825_cached_sizze_155698, bytes_152786);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152828_cached_sizze_155699 < bytes_152786) {
        err = lexical_realloc(ctx, &mem_152828, &mem_152828_cached_sizze_155699, bytes_152786);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151654 = 0; i_151654 < n_111788; i_151654++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151647 = 0; i_151647 < m_111789; i_151647++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_141524 = ((double *) mem_152790.mem)[i_151654 * m_111789 + i_151647];
            
            // futhark/microgpt.fut:472:18-58
            
            double lifted_lambda_res_141525 = zs_lhs_141524 / zs_rhs_115742;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_141532 = ((double *) mem_152787.mem)[i_151654 * m_111789 + i_151647];
            
            // futhark/microgpt.fut:474:18-59
            
            double lifted_lambda_res_141533 = zs_lhs_141532 / zs_rhs_115782;
            
            ((double *) mem_152825)[i_151654 * m_111789 + i_151647] = lifted_lambda_res_141533;
            ((double *) mem_152828)[i_151654 * m_111789 + i_151647] = lifted_lambda_res_141525;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152863, bytes_152786, "mem_152863")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151663 = 0; i_151663 < n_111788; i_151663++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151659 = 0; i_151659 < m_111789; i_151659++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_115054 = ((double *) w_mem_152781.mem)[i_151663 * m_111789 + i_151659];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_115055 = ((double *) mem_152828)[i_151663 * m_111789 + i_151659];
            
            // futhark/microgpt.fut:476:21-34
            
            double zs_lhs_115056 = lt_r_111795 * zt_rhs_115055;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_115057 = ((double *) mem_152825)[i_151663 * m_111789 + i_151659];
            
            // futhark/microgpt.fut:476:51-57
            
            double zp_lhs_115058 = fpow64(ztzt_lhs_115057, 0.5);
            
            // futhark/microgpt.fut:476:59-69
            
            double zs_rhs_115059 = eps_adam_111798 + zp_lhs_115058;
            
            // futhark/microgpt.fut:476:35-69
            
            double zm_rhs_115060 = zs_lhs_115056 / zs_rhs_115059;
            
            // futhark/microgpt.fut:476:13-69
            
            double lifted_lambda_res_115061 = zm_lhs_115054 - zm_rhs_115060;
            
            ((double *) mem_152863.mem)[i_151663 * m_111789 + i_151659] = lifted_lambda_res_115061;
        }
    }
    if (memblock_set(ctx, &mem_out_155269, &mem_152863, "mem_152863") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155270, &mem_152790, "mem_152790") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155271, &mem_152787, "mem_152787") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155695, &mem_out_155269, "mem_out_155269") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155696, &mem_out_155270, "mem_out_155270") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155697, &mem_out_155271, "mem_out_155271") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_152825);
        free(mem_152828);
        if (memblock_unref(ctx, &mem_152863, "mem_152863") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_152790, "mem_152790") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_152787, "mem_152787") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155271, "mem_out_155271") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155270, "mem_out_155270") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155269, "mem_out_155269") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_155700, double *out_prim_out_155701, struct memblock wdown_mem_152781, struct memblock wkey_mem_152782, struct memblock wout_mem_152783, struct memblock wpe_mem_152784, struct memblock wqry_mem_152785, struct memblock wte_mem_152786, struct memblock wup_mem_152787, struct memblock wval_mem_152788, struct memblock wvoc_mem_152789, struct memblock tokens_mem_152790, struct memblock target_mem_152791, struct memblock mask_mem_152792)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_152793_cached_sizze_155702 = 0;
    unsigned char *mem_152793 = NULL;
    int64_t mem_152798_cached_sizze_155703 = 0;
    unsigned char *mem_152798 = NULL;
    int64_t mem_152809_cached_sizze_155704 = 0;
    unsigned char *mem_152809 = NULL;
    int64_t mem_152814_cached_sizze_155705 = 0;
    unsigned char *mem_152814 = NULL;
    int64_t mem_152821_cached_sizze_155706 = 0;
    unsigned char *mem_152821 = NULL;
    int64_t mem_152832_cached_sizze_155707 = 0;
    unsigned char *mem_152832 = NULL;
    int64_t mem_152837_cached_sizze_155708 = 0;
    unsigned char *mem_152837 = NULL;
    int64_t mem_152844_cached_sizze_155709 = 0;
    unsigned char *mem_152844 = NULL;
    int64_t mem_152855_cached_sizze_155710 = 0;
    unsigned char *mem_152855 = NULL;
    int64_t mem_152856_cached_sizze_155711 = 0;
    unsigned char *mem_152856 = NULL;
    int64_t mem_152857_cached_sizze_155712 = 0;
    unsigned char *mem_152857 = NULL;
    int64_t mem_152870_cached_sizze_155713 = 0;
    unsigned char *mem_152870 = NULL;
    int64_t mem_152871_cached_sizze_155714 = 0;
    unsigned char *mem_152871 = NULL;
    int64_t mem_152872_cached_sizze_155715 = 0;
    unsigned char *mem_152872 = NULL;
    int64_t mem_152903_cached_sizze_155716 = 0;
    unsigned char *mem_152903 = NULL;
    int64_t mem_152904_cached_sizze_155717 = 0;
    unsigned char *mem_152904 = NULL;
    int64_t mem_152905_cached_sizze_155718 = 0;
    unsigned char *mem_152905 = NULL;
    int64_t mem_152921_cached_sizze_155719 = 0;
    unsigned char *mem_152921 = NULL;
    int64_t mem_152922_cached_sizze_155720 = 0;
    unsigned char *mem_152922 = NULL;
    int64_t mem_152923_cached_sizze_155721 = 0;
    unsigned char *mem_152923 = NULL;
    int64_t mem_152936_cached_sizze_155722 = 0;
    unsigned char *mem_152936 = NULL;
    int64_t mem_152937_cached_sizze_155723 = 0;
    unsigned char *mem_152937 = NULL;
    int64_t mem_152938_cached_sizze_155724 = 0;
    unsigned char *mem_152938 = NULL;
    int64_t mem_152984_cached_sizze_155725 = 0;
    unsigned char *mem_152984 = NULL;
    int64_t mem_152990_cached_sizze_155726 = 0;
    unsigned char *mem_152990 = NULL;
    int64_t mem_152995_cached_sizze_155727 = 0;
    unsigned char *mem_152995 = NULL;
    int64_t mem_153006_cached_sizze_155728 = 0;
    unsigned char *mem_153006 = NULL;
    int64_t mem_153011_cached_sizze_155729 = 0;
    unsigned char *mem_153011 = NULL;
    int64_t mem_153022_cached_sizze_155730 = 0;
    unsigned char *mem_153022 = NULL;
    int64_t mem_153027_cached_sizze_155731 = 0;
    unsigned char *mem_153027 = NULL;
    int64_t mem_153034_cached_sizze_155732 = 0;
    unsigned char *mem_153034 = NULL;
    int64_t mem_153041_cached_sizze_155733 = 0;
    unsigned char *mem_153041 = NULL;
    int64_t mem_153052_cached_sizze_155734 = 0;
    unsigned char *mem_153052 = NULL;
    int64_t mem_153057_cached_sizze_155735 = 0;
    unsigned char *mem_153057 = NULL;
    int64_t mem_153068_cached_sizze_155736 = 0;
    unsigned char *mem_153068 = NULL;
    int64_t mem_153073_cached_sizze_155737 = 0;
    unsigned char *mem_153073 = NULL;
    int64_t mem_153089_cached_sizze_155738 = 0;
    unsigned char *mem_153089 = NULL;
    int64_t mem_153094_cached_sizze_155739 = 0;
    unsigned char *mem_153094 = NULL;
    int64_t mem_153105_cached_sizze_155740 = 0;
    unsigned char *mem_153105 = NULL;
    int64_t mem_153110_cached_sizze_155741 = 0;
    unsigned char *mem_153110 = NULL;
    int64_t mem_153121_cached_sizze_155742 = 0;
    unsigned char *mem_153121 = NULL;
    int64_t mem_153126_cached_sizze_155743 = 0;
    unsigned char *mem_153126 = NULL;
    int64_t mem_153137_cached_sizze_155744 = 0;
    unsigned char *mem_153137 = NULL;
    int64_t mem_153142_cached_sizze_155745 = 0;
    unsigned char *mem_153142 = NULL;
    int64_t mem_153149_cached_sizze_155746 = 0;
    unsigned char *mem_153149 = NULL;
    int64_t mem_153160_cached_sizze_155747 = 0;
    unsigned char *mem_153160 = NULL;
    int64_t mem_153165_cached_sizze_155748 = 0;
    unsigned char *mem_153165 = NULL;
    int64_t mem_153176_cached_sizze_155749 = 0;
    unsigned char *mem_153176 = NULL;
    int64_t mem_153181_cached_sizze_155750 = 0;
    unsigned char *mem_153181 = NULL;
    int64_t mem_153192_cached_sizze_155751 = 0;
    unsigned char *mem_153192 = NULL;
    int64_t mem_153197_cached_sizze_155752 = 0;
    unsigned char *mem_153197 = NULL;
    int64_t mem_153208_cached_sizze_155753 = 0;
    unsigned char *mem_153208 = NULL;
    int64_t mem_153213_cached_sizze_155754 = 0;
    unsigned char *mem_153213 = NULL;
    int64_t mem_153224_cached_sizze_155755 = 0;
    unsigned char *mem_153224 = NULL;
    int64_t mem_153229_cached_sizze_155756 = 0;
    unsigned char *mem_153229 = NULL;
    int64_t mem_153244_cached_sizze_155757 = 0;
    unsigned char *mem_153244 = NULL;
    int64_t mem_153251_cached_sizze_155758 = 0;
    unsigned char *mem_153251 = NULL;
    struct memblock mem_153240;
    
    mem_153240.references = NULL;
    
    struct memblock mem_out_155269;
    
    mem_out_155269.references = NULL;
    
    struct memblock mem_152772 = ctx->constants->mem_152772;
    struct memblock mem_152773 = ctx->constants->mem_152773;
    struct memblock mem_152774 = ctx->constants->mem_152774;
    struct memblock mem_152775 = ctx->constants->mem_152775;
    struct memblock mem_152776 = ctx->constants->mem_152776;
    struct memblock mem_152777 = ctx->constants->mem_152777;
    struct memblock mem_152778 = ctx->constants->mem_152778;
    struct memblock mem_152779 = ctx->constants->mem_152779;
    struct memblock mem_152780 = ctx->constants->mem_152780;
    double prim_out_155270;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_152793_cached_sizze_155702 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152793, &mem_152793_cached_sizze_155702, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152798_cached_sizze_155703 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152798, &mem_152798_cached_sizze_155703, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151635 = 0; i_151635 < (int64_t) 16; i_151635++) {
        // futhark/microgpt.fut:457:41-50
        
        int64_t tmp_140769 = ((int64_t *) tokens_mem_152790.mem)[i_151635];
        
        // futhark/microgpt.fut:457:37-51
        
        bool x_140770 = sle64((int64_t) 0, tmp_140769);
        
        // futhark/microgpt.fut:457:37-51
        
        bool y_140771 = slt64(tmp_140769, (int64_t) 27);
        
        // futhark/microgpt.fut:457:37-51
        
        bool bounds_check_140772 = x_140770 && y_140771;
        
        // futhark/microgpt.fut:457:37-51
        
        bool index_certs_140773;
        
        if (!bounds_check_140772) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_140769, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:457:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:457:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151631 = 0; i_151631 < (int64_t) 16; i_151631++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_140780 = ((double *) wte_mem_152786.mem)[tmp_140769 * (int64_t) 16 + i_151631];
            
            ((double *) mem_152798)[i_151631] = lifted_lambda_res_140780;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152793, i_151635 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152798, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152809_cached_sizze_155704 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152809, &mem_152809_cached_sizze_155704, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152814_cached_sizze_155705 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152814, &mem_152814_cached_sizze_155705, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152821_cached_sizze_155706 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152821, &mem_152821_cached_sizze_155706, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151647 = 0; i_151647 < (int64_t) 16; i_151647++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_140806;
        double r_140808 = 0.0;
        
        for (int64_t i_140807 = 0; i_140807 < (int64_t) 16; i_140807++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_140809 = ((double *) wpe_mem_152784.mem)[i_151647 * (int64_t) 16 + i_140807];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_140810 = ((double *) mem_152793)[i_151647 * (int64_t) 16 + i_140807];
            
            // futhark/microgpt.fut:193:76-116
            
            double zp_res_140811 = zp_lhs_140809 + zp_rhs_140810;
            
            // futhark/microgpt.fut:193:94-163
            
            double zt_res_140812 = zp_res_140811 * zp_res_140811;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_140813 = r_140808 + zt_res_140812;
            double r_tmp_155274 = zp_res_140813;
            
            r_140808 = r_tmp_155274;
        }
        defunc_0_lifted_lambda_res_140806 = r_140808;
        // futhark/microgpt.fut:193:54-182
        
        double zs_res_140814 = defunc_0_lifted_lambda_res_140806 / 16.0;
        
        // futhark/microgpt.fut:194:24-55
        
        double zp_res_140815 = 1.0e-5 + zs_res_140814;
        
        // futhark/microgpt.fut:194:16-55
        
        double sqrt_res_140816 = futrts_sqrt64(zp_res_140815);
        
        // futhark/microgpt.fut:195:85-96
        
        double zs_res_140817 = 1.0 / sqrt_res_140816;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151639 = 0; i_151639 < (int64_t) 16; i_151639++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_140824 = ((double *) wpe_mem_152784.mem)[i_151647 * (int64_t) 16 + i_151639];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_140825 = ((double *) mem_152793)[i_151647 * (int64_t) 16 + i_151639];
            
            // futhark/microgpt.fut:195:38-78
            
            double zp_res_140826 = zp_lhs_140824 + zp_rhs_140825;
            
            // futhark/microgpt.fut:195:56-96
            
            double zt_res_140827 = zs_res_140817 * zp_res_140826;
            
            ((double *) mem_152814)[i_151639] = zt_res_140827;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151643 = 0; i_151643 < (int64_t) 16; i_151643++) {
            // futhark/microgpt.fut:196:4-14
            
            double lifted_lambda_res_140835 = ((double *) mem_152814)[i_151643];
            
            ((double *) mem_152821)[i_151643] = lifted_lambda_res_140835;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152809, i_151647 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152821, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152832_cached_sizze_155707 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152832, &mem_152832_cached_sizze_155707, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152837_cached_sizze_155708 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152837, &mem_152837_cached_sizze_155708, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152844_cached_sizze_155709 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152844, &mem_152844_cached_sizze_155709, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151659 = 0; i_151659 < (int64_t) 16; i_151659++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_140844;
        double r_140846 = 0.0;
        
        for (int64_t i_140845 = 0; i_140845 < (int64_t) 16; i_140845++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_140847 = ((double *) mem_152809)[i_151659 * (int64_t) 16 + i_140845];
            
            // futhark/microgpt.fut:197:78-115
            
            double zt_res_140848 = zt_lhs_140847 * zt_lhs_140847;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_140849 = r_140846 + zt_res_140848;
            double r_tmp_155278 = zp_res_140849;
            
            r_140846 = r_tmp_155278;
        }
        defunc_0_lifted_lambda_res_140844 = r_140846;
        // futhark/microgpt.fut:197:57-133
        
        double zs_res_140850 = defunc_0_lifted_lambda_res_140844 / 16.0;
        
        // futhark/microgpt.fut:198:24-55
        
        double zp_res_140851 = 1.0e-5 + zs_res_140850;
        
        // futhark/microgpt.fut:198:16-55
        
        double sqrt_res_140852 = futrts_sqrt64(zp_res_140851);
        
        // futhark/microgpt.fut:199:59-70
        
        double zs_res_140853 = 1.0 / sqrt_res_140852;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151651 = 0; i_151651 < (int64_t) 16; i_151651++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_140860 = ((double *) mem_152809)[i_151659 * (int64_t) 16 + i_151651];
            
            // futhark/microgpt.fut:199:37-70
            
            double zt_res_140861 = zs_res_140853 * zt_lhs_140860;
            
            ((double *) mem_152837)[i_151651] = zt_res_140861;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151655 = 0; i_151655 < (int64_t) 16; i_151655++) {
            // futhark/microgpt.fut:200:4-14
            
            double lifted_lambda_res_140869 = ((double *) mem_152837)[i_151655];
            
            ((double *) mem_152844)[i_151655] = lifted_lambda_res_140869;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152832, i_151659 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152844, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152855_cached_sizze_155710 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152855, &mem_152855_cached_sizze_155710, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152856_cached_sizze_155711 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152856, &mem_152856_cached_sizze_155711, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152857_cached_sizze_155712 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152857, &mem_152857_cached_sizze_155712, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152870_cached_sizze_155713 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152870, &mem_152870_cached_sizze_155713, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152871_cached_sizze_155714 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152871, &mem_152871_cached_sizze_155714, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152872_cached_sizze_155715 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152872, &mem_152872_cached_sizze_155715, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151677 = 0; i_151677 < (int64_t) 16; i_151677++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151667 = 0; i_151667 < (int64_t) 16; i_151667++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141706;
            double r_141708 = 0.0;
            
            for (int64_t i_141707 = 0; i_141707 < (int64_t) 16; i_141707++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_141709 = ((double *) wqry_mem_152785.mem)[i_151667 * (int64_t) 16 + i_141707];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_141710 = ((double *) mem_152832)[i_151677 * (int64_t) 16 + i_141707];
                
                // futhark/microgpt.fut:201:66-105
                
                double zt_res_141711 = zt_lhs_141709 * zt_rhs_141710;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141712 = r_141708 + zt_res_141711;
                double r_tmp_155287 = zp_res_141712;
                
                r_141708 = r_tmp_155287;
            }
            defunc_0_lifted_lambda_res_141706 = r_141708;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141719;
            double r_141721 = 0.0;
            
            for (int64_t i_141720 = 0; i_141720 < (int64_t) 16; i_141720++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_141722 = ((double *) wkey_mem_152782.mem)[i_151667 * (int64_t) 16 + i_141720];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_141723 = ((double *) mem_152832)[i_151677 * (int64_t) 16 + i_141720];
                
                // futhark/microgpt.fut:202:66-105
                
                double zt_res_141724 = zt_lhs_141722 * zt_rhs_141723;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141725 = r_141721 + zt_res_141724;
                double r_tmp_155288 = zp_res_141725;
                
                r_141721 = r_tmp_155288;
            }
            defunc_0_lifted_lambda_res_141719 = r_141721;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141735;
            double r_141737 = 0.0;
            
            for (int64_t i_141736 = 0; i_141736 < (int64_t) 16; i_141736++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_141738 = ((double *) wval_mem_152788.mem)[i_151667 * (int64_t) 16 + i_141736];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_141739 = ((double *) mem_152832)[i_151677 * (int64_t) 16 + i_141736];
                
                // futhark/microgpt.fut:203:66-105
                
                double zt_res_141740 = zt_lhs_141738 * zt_rhs_141739;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141741 = r_141737 + zt_res_141740;
                double r_tmp_155289 = zp_res_141741;
                
                r_141737 = r_tmp_155289;
            }
            defunc_0_lifted_lambda_res_141735 = r_141737;
            ((double *) mem_152870)[i_151667] = defunc_0_lifted_lambda_res_141735;
            ((double *) mem_152871)[i_151667] = defunc_0_lifted_lambda_res_141719;
            ((double *) mem_152872)[i_151667] = defunc_0_lifted_lambda_res_141706;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152855, i_151677 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152870, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152856, i_151677 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152871, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152857, i_151677 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152872, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152903_cached_sizze_155716 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152903, &mem_152903_cached_sizze_155716, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152904_cached_sizze_155717 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152904, &mem_152904_cached_sizze_155717, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152905_cached_sizze_155718 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152905, &mem_152905_cached_sizze_155718, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152921_cached_sizze_155719 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152921, &mem_152921_cached_sizze_155719, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152922_cached_sizze_155720 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152922, &mem_152922_cached_sizze_155720, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152923_cached_sizze_155721 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152923, &mem_152923_cached_sizze_155721, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152936_cached_sizze_155722 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152936, &mem_152936_cached_sizze_155722, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152937_cached_sizze_155723 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152937, &mem_152937_cached_sizze_155723, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152938_cached_sizze_155724 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152938, &mem_152938_cached_sizze_155724, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151707 = 0; i_151707 < (int64_t) 4; i_151707++) {
        // futhark/microgpt.fut:204:69-72
        
        int64_t zp_lhs_141582 = mul64((int64_t) 4, i_151707);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151697 = 0; i_151697 < (int64_t) 16; i_151697++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151687 = 0; i_151687 < (int64_t) 4; i_151687++) {
                // futhark/microgpt.fut:204:74-81
                
                int64_t tmp_141899 = add64(zp_lhs_141582, i_151687);
                
                // futhark/microgpt.fut:204:51-83
                
                bool x_141900 = sle64((int64_t) 0, tmp_141899);
                
                // futhark/microgpt.fut:204:51-83
                
                bool y_141901 = slt64(tmp_141899, (int64_t) 16);
                
                // futhark/microgpt.fut:204:51-83
                
                bool bounds_check_141902 = x_141900 && y_141901;
                
                // futhark/microgpt.fut:204:51-83
                
                bool index_certs_141903;
                
                if (!bounds_check_141902) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_141899, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:204:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:204:15-84\n   #9  futhark/microgpt.fut:458:7-76\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141904 = ((double *) mem_152857)[i_151697 * (int64_t) 16 + tmp_141899];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141912 = ((double *) mem_152856)[i_151697 * (int64_t) 16 + tmp_141899];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141923 = ((double *) mem_152855)[i_151697 * (int64_t) 16 + tmp_141899];
                
                ((double *) mem_152936)[i_151687] = lifted_lambda_res_141923;
                ((double *) mem_152937)[i_151687] = lifted_lambda_res_141912;
                ((double *) mem_152938)[i_151687] = lifted_lambda_res_141904;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152921, i_151697 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152936, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152922, i_151697 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152937, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152923, i_151697 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152938, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_152903, i_151707 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_152921, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_152904, i_151707 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_152922, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_152905, i_151707 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_152923, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152984_cached_sizze_155725 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152984, &mem_152984_cached_sizze_155725, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152990_cached_sizze_155726 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152990, &mem_152990_cached_sizze_155726, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152995_cached_sizze_155727 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152995, &mem_152995_cached_sizze_155727, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153006_cached_sizze_155728 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153006, &mem_153006_cached_sizze_155728, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153011_cached_sizze_155729 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153011, &mem_153011_cached_sizze_155729, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153022_cached_sizze_155730 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153022, &mem_153022_cached_sizze_155730, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153027_cached_sizze_155731 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153027, &mem_153027_cached_sizze_155731, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153034_cached_sizze_155732 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153034, &mem_153034_cached_sizze_155732, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153041_cached_sizze_155733 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153041, &mem_153041_cached_sizze_155733, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153052_cached_sizze_155734 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153052, &mem_153052_cached_sizze_155734, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153057_cached_sizze_155735 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_153057, &mem_153057_cached_sizze_155735, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153068_cached_sizze_155736 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153068, &mem_153068_cached_sizze_155736, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153073_cached_sizze_155737 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_153073, &mem_153073_cached_sizze_155737, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151763 = 0; i_151763 < (int64_t) 4; i_151763++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151717 = 0; i_151717 < (int64_t) 16; i_151717++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151713 = 0; i_151713 < (int64_t) 16; i_151713++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_141014;
                double r_141016 = 0.0;
                
                for (int64_t i_141015 = 0; i_141015 < (int64_t) 4; i_141015++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_141017 = ((double *) mem_152905)[i_151763 * (int64_t) 64 + i_151717 * (int64_t) 4 + i_141015];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_141018 = ((double *) mem_152904)[i_151763 * (int64_t) 64 + i_151713 * (int64_t) 4 + i_141015];
                    
                    // futhark/microgpt.fut:207:113-164
                    
                    double zt_res_141019 = zt_lhs_141017 * zt_rhs_141018;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_141020 = r_141016 + zt_res_141019;
                    double r_tmp_155302 = zp_res_141020;
                    
                    r_141016 = r_tmp_155302;
                }
                defunc_0_lifted_lambda_res_141014 = r_141016;
                ((double *) mem_152995)[i_151713] = defunc_0_lifted_lambda_res_141014;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152990, i_151717 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152995, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151725 = 0; i_151725 < (int64_t) 16; i_151725++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151721 = 0; i_151721 < (int64_t) 16; i_151721++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_141035 = ((double *) mem_152990)[i_151725 * (int64_t) 16 + i_151721];
                
                // futhark/microgpt.fut:208:47-78
                
                double zs_res_141036 = zs_lhs_141035 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_141037 = ((double *) mask_mem_152792.mem)[i_151725 * (int64_t) 16 + i_151721];
                
                // futhark/microgpt.fut:208:65-102
                
                double zp_res_141038 = zs_res_141036 + zp_rhs_141037;
                
                ((double *) mem_153011)[i_151721] = zp_res_141038;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153006, i_151725 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153011, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151743 = 0; i_151743 < (int64_t) 16; i_151743++) {
            // futhark/microgpt.fut:105:13-33
            
            double defunc_0_reduce_res_142026;
            double redout_151727 = -INFINITY;
            
            for (int64_t i_151728 = 0; i_151728 < (int64_t) 16; i_151728++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141950 = ((double *) mem_153006)[i_151743 * (int64_t) 16 + i_151728];
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_141059 = fmax64(lifted_lambda_res_141950, redout_151727);
                double redout_tmp_155306 = max_res_141059;
                
                redout_151727 = redout_tmp_155306;
            }
            defunc_0_reduce_res_142026 = redout_151727;
            // futhark/microgpt.fut:210:67-76
            
            double neg_res_141060 = -defunc_0_reduce_res_142026;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151731 = 0; i_151731 < (int64_t) 16; i_151731++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_141067 = ((double *) mem_153006)[i_151743 * (int64_t) 16 + i_151731];
                
                // futhark/microgpt.fut:210:44-76
                
                double zp_res_141068 = neg_res_141060 + zp_lhs_141067;
                
                // futhark/microgpt.fut:210:37-76
                
                double exp_res_141069 = futrts_exp64(zp_res_141068);
                
                ((double *) mem_153027)[i_151731] = exp_res_141069;
            }
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141071;
            double r_141073 = 0.0;
            
            for (int64_t i_141072 = 0; i_141072 < (int64_t) 16; i_141072++) {
                // futhark/microgpt.fut:211:36-46
                
                double lifted_lambda_res_141074 = ((double *) mem_153027)[i_141072];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141075 = r_141073 + lifted_lambda_res_141074;
                double r_tmp_155308 = zp_res_141075;
                
                r_141073 = r_tmp_155308;
            }
            defunc_0_lifted_lambda_res_141071 = r_141073;
            // futhark/microgpt.fut:212:53-64
            
            double zs_res_141076 = 1.0 / defunc_0_lifted_lambda_res_141071;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151735 = 0; i_151735 < (int64_t) 16; i_151735++) {
                // futhark/microgpt.fut:212:37-47
                
                double zt_lhs_141083 = ((double *) mem_153027)[i_151735];
                
                // futhark/microgpt.fut:212:37-64
                
                double zt_res_141084 = zs_res_141076 * zt_lhs_141083;
                
                ((double *) mem_153034)[i_151735] = zt_res_141084;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151739 = 0; i_151739 < (int64_t) 16; i_151739++) {
                // futhark/microgpt.fut:213:4-14
                
                double lifted_lambda_res_141092 = ((double *) mem_153034)[i_151739];
                
                ((double *) mem_153041)[i_151739] = lifted_lambda_res_141092;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153022, i_151743 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153041, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151751 = 0; i_151751 < (int64_t) 16; i_151751++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151747 = 0; i_151747 < (int64_t) 4; i_151747++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_141107;
                double r_141109 = 0.0;
                
                for (int64_t i_141108 = 0; i_141108 < (int64_t) 16; i_141108++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_141110 = ((double *) mem_153022)[i_151751 * (int64_t) 16 + i_141108];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_141111 = ((double *) mem_152903)[i_151763 * (int64_t) 64 + i_141108 * (int64_t) 4 + i_151747];
                    
                    // futhark/microgpt.fut:214:66-111
                    
                    double zt_res_141112 = zt_lhs_141110 * zt_rhs_141111;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_141113 = r_141109 + zt_res_141112;
                    double r_tmp_155313 = zp_res_141113;
                    
                    r_141109 = r_tmp_155313;
                }
                defunc_0_lifted_lambda_res_141107 = r_141109;
                ((double *) mem_153057)[i_151747] = defunc_0_lifted_lambda_res_141107;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153052, i_151751 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153057, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151759 = 0; i_151759 < (int64_t) 16; i_151759++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151755 = 0; i_151755 < (int64_t) 4; i_151755++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141128 = ((double *) mem_153052)[i_151759 * (int64_t) 4 + i_151755];
                
                ((double *) mem_153073)[i_151755] = lifted_lambda_res_141128;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153068, i_151759 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153073, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_152984, i_151763 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_153068, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153089_cached_sizze_155738 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153089, &mem_153089_cached_sizze_155738, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153094_cached_sizze_155739 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153094, &mem_153094_cached_sizze_155739, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151771 = 0; i_151771 < (int64_t) 16; i_151771++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151767 = 0; i_151767 < (int64_t) 16; i_151767++) {
            // futhark/microgpt.fut:216:54-57
            
            int64_t tmp_141140 = sdiv64(i_151767, (int64_t) 4);
            
            // futhark/microgpt.fut:216:44-59
            
            bool x_141141 = sle64((int64_t) 0, tmp_141140);
            
            // futhark/microgpt.fut:216:44-59
            
            bool y_141142 = slt64(tmp_141140, (int64_t) 4);
            
            // futhark/microgpt.fut:216:44-59
            
            bool bounds_check_141143 = x_141141 && y_141142;
            
            // futhark/microgpt.fut:216:44-59
            
            bool index_certs_141144;
            
            if (!bounds_check_141143) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_141140, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:216:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:216:15-80\n   #6  futhark/microgpt.fut:458:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:216:74-77
            
            int64_t tmp_141145 = smod64(i_151767, (int64_t) 4);
            
            // futhark/microgpt.fut:216:44-79
            
            bool x_141146 = sle64((int64_t) 0, tmp_141145);
            
            // futhark/microgpt.fut:216:44-79
            
            bool y_141147 = slt64(tmp_141145, (int64_t) 4);
            
            // futhark/microgpt.fut:216:44-79
            
            bool bounds_check_141148 = x_141146 && y_141147;
            
            // futhark/microgpt.fut:216:44-79
            
            bool index_certs_141149;
            
            if (!bounds_check_141148) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_141145, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:216:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:216:15-80\n   #6  futhark/microgpt.fut:458:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_141150 = ((double *) mem_152984)[tmp_141140 * (int64_t) 64 + i_151771 * (int64_t) 4 + tmp_141145];
            
            ((double *) mem_153094)[i_151767] = lifted_lambda_res_141150;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153089, i_151771 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153094, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153105_cached_sizze_155740 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153105, &mem_153105_cached_sizze_155740, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153110_cached_sizze_155741 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153110, &mem_153110_cached_sizze_155741, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151779 = 0; i_151779 < (int64_t) 16; i_151779++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151775 = 0; i_151775 < (int64_t) 16; i_151775++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141165;
            double r_141167 = 0.0;
            
            for (int64_t i_141166 = 0; i_141166 < (int64_t) 16; i_141166++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_141168 = ((double *) wout_mem_152783.mem)[i_151775 * (int64_t) 16 + i_141166];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_141169 = ((double *) mem_153089)[i_151779 * (int64_t) 16 + i_141166];
                
                // futhark/microgpt.fut:217:67-106
                
                double zt_res_141170 = zt_lhs_141168 * zt_rhs_141169;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141171 = r_141167 + zt_res_141170;
                double r_tmp_155320 = zp_res_141171;
                
                r_141167 = r_tmp_155320;
            }
            defunc_0_lifted_lambda_res_141165 = r_141167;
            ((double *) mem_153110)[i_151775] = defunc_0_lifted_lambda_res_141165;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153105, i_151779 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153110, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153121_cached_sizze_155742 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153121, &mem_153121_cached_sizze_155742, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153126_cached_sizze_155743 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153126, &mem_153126_cached_sizze_155743, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151787 = 0; i_151787 < (int64_t) 16; i_151787++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151783 = 0; i_151783 < (int64_t) 16; i_151783++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_141186 = ((double *) mem_153105)[i_151787 * (int64_t) 16 + i_151783];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_141187 = ((double *) mem_152809)[i_151787 * (int64_t) 16 + i_151783];
            
            // futhark/microgpt.fut:218:46-84
            
            double zp_res_141188 = zp_lhs_141186 + zp_rhs_141187;
            
            ((double *) mem_153126)[i_151783] = zp_res_141188;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153121, i_151787 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153126, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153137_cached_sizze_155744 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153137, &mem_153137_cached_sizze_155744, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153142_cached_sizze_155745 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153142, &mem_153142_cached_sizze_155745, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153149_cached_sizze_155746 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153149, &mem_153149_cached_sizze_155746, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151799 = 0; i_151799 < (int64_t) 16; i_151799++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_141197;
        double r_141199 = 0.0;
        
        for (int64_t i_141198 = 0; i_141198 < (int64_t) 16; i_141198++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_141200 = ((double *) mem_153121)[i_151799 * (int64_t) 16 + i_141198];
            
            // futhark/microgpt.fut:219:79-118
            
            double zt_res_141201 = zt_lhs_141200 * zt_lhs_141200;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_141202 = r_141199 + zt_res_141201;
            double r_tmp_155324 = zp_res_141202;
            
            r_141199 = r_tmp_155324;
        }
        defunc_0_lifted_lambda_res_141197 = r_141199;
        // futhark/microgpt.fut:219:58-136
        
        double zs_res_141203 = defunc_0_lifted_lambda_res_141197 / 16.0;
        
        // futhark/microgpt.fut:220:24-55
        
        double zp_res_141204 = 1.0e-5 + zs_res_141203;
        
        // futhark/microgpt.fut:220:16-55
        
        double sqrt_res_141205 = futrts_sqrt64(zp_res_141204);
        
        // futhark/microgpt.fut:221:60-71
        
        double zs_res_141206 = 1.0 / sqrt_res_141205;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151791 = 0; i_151791 < (int64_t) 16; i_151791++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_141213 = ((double *) mem_153121)[i_151799 * (int64_t) 16 + i_151791];
            
            // futhark/microgpt.fut:221:37-71
            
            double zt_res_141214 = zs_res_141206 * zt_lhs_141213;
            
            ((double *) mem_153142)[i_151791] = zt_res_141214;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151795 = 0; i_151795 < (int64_t) 16; i_151795++) {
            // futhark/microgpt.fut:222:4-14
            
            double lifted_lambda_res_141222 = ((double *) mem_153142)[i_151795];
            
            ((double *) mem_153149)[i_151795] = lifted_lambda_res_141222;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153137, i_151799 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153149, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153160_cached_sizze_155747 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153160, &mem_153160_cached_sizze_155747, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153165_cached_sizze_155748 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153165, &mem_153165_cached_sizze_155748, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151807 = 0; i_151807 < (int64_t) 16; i_151807++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151803 = 0; i_151803 < (int64_t) 64; i_151803++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141238;
            double r_141240 = 0.0;
            
            for (int64_t i_141239 = 0; i_141239 < (int64_t) 16; i_141239++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_141241 = ((double *) wup_mem_152787.mem)[i_151803 * (int64_t) 16 + i_141239];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_141242 = ((double *) mem_153137)[i_151807 * (int64_t) 16 + i_141239];
                
                // futhark/microgpt.fut:223:67-106
                
                double zt_res_141243 = zt_lhs_141241 * zt_rhs_141242;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141244 = r_141240 + zt_res_141243;
                double r_tmp_155329 = zp_res_141244;
                
                r_141240 = r_tmp_155329;
            }
            defunc_0_lifted_lambda_res_141238 = r_141240;
            ((double *) mem_153165)[i_151803] = defunc_0_lifted_lambda_res_141238;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153160, i_151807 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153165, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153176_cached_sizze_155749 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153176, &mem_153176_cached_sizze_155749, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153181_cached_sizze_155750 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153181, &mem_153181_cached_sizze_155750, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151815 = 0; i_151815 < (int64_t) 16; i_151815++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151811 = 0; i_151811 < (int64_t) 64; i_151811++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_141259 = ((double *) mem_153160)[i_151815 * (int64_t) 64 + i_151811];
            
            // futhark/microgpt.fut:224:45-73
            
            double max_res_141260 = fmax64(0.0, max_arg0_141259);
            
            ((double *) mem_153181)[i_151811] = max_res_141260;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153176, i_151815 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153181, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153192_cached_sizze_155751 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153192, &mem_153192_cached_sizze_155751, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153197_cached_sizze_155752 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153197, &mem_153197_cached_sizze_155752, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151823 = 0; i_151823 < (int64_t) 16; i_151823++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151819 = 0; i_151819 < (int64_t) 16; i_151819++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141275;
            double r_141277 = 0.0;
            
            for (int64_t i_141276 = 0; i_141276 < (int64_t) 64; i_141276++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_141278 = ((double *) wdown_mem_152781.mem)[i_151819 * (int64_t) 64 + i_141276];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_141279 = ((double *) mem_153176)[i_151823 * (int64_t) 64 + i_141276];
                
                // futhark/microgpt.fut:225:67-108
                
                double zt_res_141280 = zt_lhs_141278 * zt_rhs_141279;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141281 = r_141277 + zt_res_141280;
                double r_tmp_155334 = zp_res_141281;
                
                r_141277 = r_tmp_155334;
            }
            defunc_0_lifted_lambda_res_141275 = r_141277;
            ((double *) mem_153197)[i_151819] = defunc_0_lifted_lambda_res_141275;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153192, i_151823 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153197, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153208_cached_sizze_155753 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153208, &mem_153208_cached_sizze_155753, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153213_cached_sizze_155754 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153213, &mem_153213_cached_sizze_155754, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151831 = 0; i_151831 < (int64_t) 16; i_151831++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151827 = 0; i_151827 < (int64_t) 16; i_151827++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_141296 = ((double *) mem_153192)[i_151831 * (int64_t) 16 + i_151827];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_141297 = ((double *) mem_153121)[i_151831 * (int64_t) 16 + i_151827];
            
            // futhark/microgpt.fut:226:46-85
            
            double zp_res_141298 = zp_lhs_141296 + zp_rhs_141297;
            
            ((double *) mem_153213)[i_151827] = zp_res_141298;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153208, i_151831 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153213, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153224_cached_sizze_155755 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_153224, &mem_153224_cached_sizze_155755, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153229_cached_sizze_155756 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_153229, &mem_153229_cached_sizze_155756, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151839 = 0; i_151839 < (int64_t) 16; i_151839++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151835 = 0; i_151835 < (int64_t) 27; i_151835++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141314;
            double r_141316 = 0.0;
            
            for (int64_t i_141315 = 0; i_141315 < (int64_t) 16; i_141315++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_141317 = ((double *) wvoc_mem_152789.mem)[i_151835 * (int64_t) 16 + i_141315];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_141318 = ((double *) mem_153208)[i_151839 * (int64_t) 16 + i_141315];
                
                // futhark/microgpt.fut:227:67-107
                
                double zt_res_141319 = zt_lhs_141317 * zt_rhs_141318;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141320 = r_141316 + zt_res_141319;
                double r_tmp_155339 = zp_res_141320;
                
                r_141316 = r_tmp_155339;
            }
            defunc_0_lifted_lambda_res_141314 = r_141316;
            ((double *) mem_153229)[i_151835] = defunc_0_lifted_lambda_res_141314;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153224, i_151839 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153229, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_153240, (int64_t) 128, "mem_153240")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153244_cached_sizze_155757 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_153244, &mem_153244_cached_sizze_155757, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153251_cached_sizze_155758 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_153251, &mem_153251_cached_sizze_155758, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151853 = 0; i_151853 < (int64_t) 16; i_151853++) {
        double x_142049;
        double redout_151841 = -INFINITY;
        
        for (int64_t i_151842 = 0; i_151842 < (int64_t) 27; i_151842++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_141996 = ((double *) mem_153224)[i_151853 * (int64_t) 27 + i_151842];
            
            // futhark/microgpt.fut:105:13-33
            
            double max_res_141344 = fmax64(lifted_lambda_res_141996, redout_151841);
            double redout_tmp_155341 = max_res_141344;
            
            redout_151841 = redout_tmp_155341;
        }
        x_142049 = redout_151841;
        // futhark/microgpt.fut:229:67-76
        
        double neg_res_141345 = -x_142049;
        
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_141329;
        double r_141331 = 0.0;
        
        for (int64_t i_141330 = 0; i_141330 < (int64_t) 27; i_141330++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151845 = 0; i_151845 < (int64_t) 27; i_151845++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_141352 = ((double *) mem_153224)[i_151853 * (int64_t) 27 + i_151845];
                
                // futhark/microgpt.fut:229:44-76
                
                double zp_res_141353 = neg_res_141345 + zp_lhs_141352;
                
                // futhark/microgpt.fut:229:37-76
                
                double exp_res_141354 = futrts_exp64(zp_res_141353);
                
                ((double *) mem_153244)[i_151845] = exp_res_141354;
            }
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141356;
            double r_141358 = 0.0;
            
            for (int64_t i_141357 = 0; i_141357 < (int64_t) 27; i_141357++) {
                // futhark/microgpt.fut:230:36-46
                
                double lifted_lambda_res_141359 = ((double *) mem_153244)[i_141357];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141360 = r_141358 + lifted_lambda_res_141359;
                double r_tmp_155344 = zp_res_141360;
                
                r_141358 = r_tmp_155344;
            }
            defunc_0_lifted_lambda_res_141356 = r_141358;
            // futhark/microgpt.fut:231:53-64
            
            double zs_res_141361 = 1.0 / defunc_0_lifted_lambda_res_141356;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151849 = 0; i_151849 < (int64_t) 27; i_151849++) {
                // futhark/microgpt.fut:231:37-47
                
                double zt_lhs_141368 = ((double *) mem_153244)[i_151849];
                
                // futhark/microgpt.fut:231:37-64
                
                double zt_res_141369 = zs_res_141361 * zt_lhs_141368;
                
                ((double *) mem_153251)[i_151849] = zt_res_141369;
            }
            // futhark/microgpt.fut:232:12-22
            
            double log_arg0_141371 = ((double *) mem_153251)[i_141330];
            
            // futhark/microgpt.fut:232:6-22
            
            double log_res_141372 = futrts_log64(log_arg0_141371);
            
            // futhark/microgpt.fut:61:46-49
            
            double zt_rhs_141373 = ((double *) target_mem_152791.mem)[i_151853 * (int64_t) 27 + i_141330];
            
            // futhark/microgpt.fut:232:6-48
            
            double zt_res_141374 = log_res_141372 * zt_rhs_141373;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_141375 = r_141331 + zt_res_141374;
            double r_tmp_155342 = zp_res_141375;
            
            r_141331 = r_tmp_155342;
        }
        defunc_0_lifted_lambda_res_141329 = r_141331;
        // futhark/microgpt.fut:228:37-232:54
        
        double neg_res_141376 = -defunc_0_lifted_lambda_res_141329;
        
        ((double *) mem_153240.mem)[i_151853] = neg_res_141376;
    }
    // futhark/microgpt.fut:61:13-49
    
    double defunc_0_lifted_lambda_res_141378;
    double r_141380 = 0.0;
    
    for (int64_t i_141379 = 0; i_141379 < (int64_t) 16; i_141379++) {
        // futhark/microgpt.fut:233:37-47
        
        double lifted_lambda_res_141381 = ((double *) mem_153240.mem)[i_141379];
        
        // futhark/microgpt.fut:61:40-49
        
        double zp_res_141382 = r_141380 + lifted_lambda_res_141381;
        double r_tmp_155346 = zp_res_141382;
        
        r_141380 = r_tmp_155346;
    }
    defunc_0_lifted_lambda_res_141378 = r_141380;
    // futhark/microgpt.fut:233:17-64
    
    double zs_res_141383 = defunc_0_lifted_lambda_res_141378 / 16.0;
    
    if (memblock_set(ctx, &mem_out_155269, &mem_153240, "mem_153240") != 0)
        return 1;
    prim_out_155270 = zs_res_141383;
    if (memblock_set(ctx, &*mem_out_p_155700, &mem_out_155269, "mem_out_155269") != 0)
        return 1;
    *out_prim_out_155701 = prim_out_155270;
    
  cleanup:
    {
        free(mem_152793);
        free(mem_152798);
        free(mem_152809);
        free(mem_152814);
        free(mem_152821);
        free(mem_152832);
        free(mem_152837);
        free(mem_152844);
        free(mem_152855);
        free(mem_152856);
        free(mem_152857);
        free(mem_152870);
        free(mem_152871);
        free(mem_152872);
        free(mem_152903);
        free(mem_152904);
        free(mem_152905);
        free(mem_152921);
        free(mem_152922);
        free(mem_152923);
        free(mem_152936);
        free(mem_152937);
        free(mem_152938);
        free(mem_152984);
        free(mem_152990);
        free(mem_152995);
        free(mem_153006);
        free(mem_153011);
        free(mem_153022);
        free(mem_153027);
        free(mem_153034);
        free(mem_153041);
        free(mem_153052);
        free(mem_153057);
        free(mem_153068);
        free(mem_153073);
        free(mem_153089);
        free(mem_153094);
        free(mem_153105);
        free(mem_153110);
        free(mem_153121);
        free(mem_153126);
        free(mem_153137);
        free(mem_153142);
        free(mem_153149);
        free(mem_153160);
        free(mem_153165);
        free(mem_153176);
        free(mem_153181);
        free(mem_153192);
        free(mem_153197);
        free(mem_153208);
        free(mem_153213);
        free(mem_153224);
        free(mem_153229);
        free(mem_153244);
        free(mem_153251);
        if (memblock_unref(ctx, &mem_153240, "mem_153240") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155269, "mem_out_155269") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_155759, struct memblock wdown_mem_152781, struct memblock wkey_mem_152782, struct memblock wout_mem_152783, struct memblock wpe_mem_152784, struct memblock wqry_mem_152785, struct memblock wte_mem_152786, struct memblock wup_mem_152787, struct memblock wval_mem_152788, struct memblock wvoc_mem_152789, struct memblock tokens_mem_152790, struct memblock mask_mem_152791)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_152792_cached_sizze_155760 = 0;
    unsigned char *mem_152792 = NULL;
    int64_t mem_152797_cached_sizze_155761 = 0;
    unsigned char *mem_152797 = NULL;
    int64_t mem_152808_cached_sizze_155762 = 0;
    unsigned char *mem_152808 = NULL;
    int64_t mem_152813_cached_sizze_155763 = 0;
    unsigned char *mem_152813 = NULL;
    int64_t mem_152820_cached_sizze_155764 = 0;
    unsigned char *mem_152820 = NULL;
    int64_t mem_152831_cached_sizze_155765 = 0;
    unsigned char *mem_152831 = NULL;
    int64_t mem_152836_cached_sizze_155766 = 0;
    unsigned char *mem_152836 = NULL;
    int64_t mem_152843_cached_sizze_155767 = 0;
    unsigned char *mem_152843 = NULL;
    int64_t mem_152854_cached_sizze_155768 = 0;
    unsigned char *mem_152854 = NULL;
    int64_t mem_152855_cached_sizze_155769 = 0;
    unsigned char *mem_152855 = NULL;
    int64_t mem_152856_cached_sizze_155770 = 0;
    unsigned char *mem_152856 = NULL;
    int64_t mem_152869_cached_sizze_155771 = 0;
    unsigned char *mem_152869 = NULL;
    int64_t mem_152870_cached_sizze_155772 = 0;
    unsigned char *mem_152870 = NULL;
    int64_t mem_152871_cached_sizze_155773 = 0;
    unsigned char *mem_152871 = NULL;
    int64_t mem_152902_cached_sizze_155774 = 0;
    unsigned char *mem_152902 = NULL;
    int64_t mem_152903_cached_sizze_155775 = 0;
    unsigned char *mem_152903 = NULL;
    int64_t mem_152904_cached_sizze_155776 = 0;
    unsigned char *mem_152904 = NULL;
    int64_t mem_152920_cached_sizze_155777 = 0;
    unsigned char *mem_152920 = NULL;
    int64_t mem_152921_cached_sizze_155778 = 0;
    unsigned char *mem_152921 = NULL;
    int64_t mem_152922_cached_sizze_155779 = 0;
    unsigned char *mem_152922 = NULL;
    int64_t mem_152935_cached_sizze_155780 = 0;
    unsigned char *mem_152935 = NULL;
    int64_t mem_152936_cached_sizze_155781 = 0;
    unsigned char *mem_152936 = NULL;
    int64_t mem_152937_cached_sizze_155782 = 0;
    unsigned char *mem_152937 = NULL;
    int64_t mem_152983_cached_sizze_155783 = 0;
    unsigned char *mem_152983 = NULL;
    int64_t mem_152989_cached_sizze_155784 = 0;
    unsigned char *mem_152989 = NULL;
    int64_t mem_152994_cached_sizze_155785 = 0;
    unsigned char *mem_152994 = NULL;
    int64_t mem_153005_cached_sizze_155786 = 0;
    unsigned char *mem_153005 = NULL;
    int64_t mem_153010_cached_sizze_155787 = 0;
    unsigned char *mem_153010 = NULL;
    int64_t mem_153021_cached_sizze_155788 = 0;
    unsigned char *mem_153021 = NULL;
    int64_t mem_153026_cached_sizze_155789 = 0;
    unsigned char *mem_153026 = NULL;
    int64_t mem_153033_cached_sizze_155790 = 0;
    unsigned char *mem_153033 = NULL;
    int64_t mem_153040_cached_sizze_155791 = 0;
    unsigned char *mem_153040 = NULL;
    int64_t mem_153051_cached_sizze_155792 = 0;
    unsigned char *mem_153051 = NULL;
    int64_t mem_153056_cached_sizze_155793 = 0;
    unsigned char *mem_153056 = NULL;
    int64_t mem_153067_cached_sizze_155794 = 0;
    unsigned char *mem_153067 = NULL;
    int64_t mem_153072_cached_sizze_155795 = 0;
    unsigned char *mem_153072 = NULL;
    int64_t mem_153088_cached_sizze_155796 = 0;
    unsigned char *mem_153088 = NULL;
    int64_t mem_153093_cached_sizze_155797 = 0;
    unsigned char *mem_153093 = NULL;
    int64_t mem_153104_cached_sizze_155798 = 0;
    unsigned char *mem_153104 = NULL;
    int64_t mem_153109_cached_sizze_155799 = 0;
    unsigned char *mem_153109 = NULL;
    int64_t mem_153120_cached_sizze_155800 = 0;
    unsigned char *mem_153120 = NULL;
    int64_t mem_153125_cached_sizze_155801 = 0;
    unsigned char *mem_153125 = NULL;
    int64_t mem_153136_cached_sizze_155802 = 0;
    unsigned char *mem_153136 = NULL;
    int64_t mem_153141_cached_sizze_155803 = 0;
    unsigned char *mem_153141 = NULL;
    int64_t mem_153148_cached_sizze_155804 = 0;
    unsigned char *mem_153148 = NULL;
    int64_t mem_153159_cached_sizze_155805 = 0;
    unsigned char *mem_153159 = NULL;
    int64_t mem_153164_cached_sizze_155806 = 0;
    unsigned char *mem_153164 = NULL;
    int64_t mem_153175_cached_sizze_155807 = 0;
    unsigned char *mem_153175 = NULL;
    int64_t mem_153180_cached_sizze_155808 = 0;
    unsigned char *mem_153180 = NULL;
    int64_t mem_153191_cached_sizze_155809 = 0;
    unsigned char *mem_153191 = NULL;
    int64_t mem_153196_cached_sizze_155810 = 0;
    unsigned char *mem_153196 = NULL;
    int64_t mem_153207_cached_sizze_155811 = 0;
    unsigned char *mem_153207 = NULL;
    int64_t mem_153212_cached_sizze_155812 = 0;
    unsigned char *mem_153212 = NULL;
    int64_t mem_153223_cached_sizze_155813 = 0;
    unsigned char *mem_153223 = NULL;
    int64_t mem_153228_cached_sizze_155814 = 0;
    unsigned char *mem_153228 = NULL;
    int64_t mem_153244_cached_sizze_155815 = 0;
    unsigned char *mem_153244 = NULL;
    struct memblock mem_153239;
    
    mem_153239.references = NULL;
    
    struct memblock mem_out_155269;
    
    mem_out_155269.references = NULL;
    
    struct memblock mem_152772 = ctx->constants->mem_152772;
    struct memblock mem_152773 = ctx->constants->mem_152773;
    struct memblock mem_152774 = ctx->constants->mem_152774;
    struct memblock mem_152775 = ctx->constants->mem_152775;
    struct memblock mem_152776 = ctx->constants->mem_152776;
    struct memblock mem_152777 = ctx->constants->mem_152777;
    struct memblock mem_152778 = ctx->constants->mem_152778;
    struct memblock mem_152779 = ctx->constants->mem_152779;
    struct memblock mem_152780 = ctx->constants->mem_152780;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_152792_cached_sizze_155760 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152792, &mem_152792_cached_sizze_155760, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152797_cached_sizze_155761 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152797, &mem_152797_cached_sizze_155761, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151635 = 0; i_151635 < (int64_t) 16; i_151635++) {
        // futhark/microgpt.fut:452:41-50
        
        int64_t tmp_140768 = ((int64_t *) tokens_mem_152790.mem)[i_151635];
        
        // futhark/microgpt.fut:452:37-51
        
        bool x_140769 = sle64((int64_t) 0, tmp_140768);
        
        // futhark/microgpt.fut:452:37-51
        
        bool y_140770 = slt64(tmp_140768, (int64_t) 27);
        
        // futhark/microgpt.fut:452:37-51
        
        bool bounds_check_140771 = x_140769 && y_140770;
        
        // futhark/microgpt.fut:452:37-51
        
        bool index_certs_140772;
        
        if (!bounds_check_140771) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_140768, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:452:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:452:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151631 = 0; i_151631 < (int64_t) 16; i_151631++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_140779 = ((double *) wte_mem_152786.mem)[tmp_140768 * (int64_t) 16 + i_151631];
            
            ((double *) mem_152797)[i_151631] = lifted_lambda_res_140779;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152792, i_151635 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152797, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152808_cached_sizze_155762 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152808, &mem_152808_cached_sizze_155762, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152813_cached_sizze_155763 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152813, &mem_152813_cached_sizze_155763, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152820_cached_sizze_155764 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152820, &mem_152820_cached_sizze_155764, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151647 = 0; i_151647 < (int64_t) 16; i_151647++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_140805;
        double r_140807 = 0.0;
        
        for (int64_t i_140806 = 0; i_140806 < (int64_t) 16; i_140806++) {
            // futhark/microgpt.fut:61:46-49
            
            double zp_lhs_140808 = ((double *) wpe_mem_152784.mem)[i_151647 * (int64_t) 16 + i_140806];
            
            // futhark/microgpt.fut:61:46-49
            
            double zp_rhs_140809 = ((double *) mem_152792)[i_151647 * (int64_t) 16 + i_140806];
            
            // futhark/microgpt.fut:138:76-116
            
            double zp_res_140810 = zp_lhs_140808 + zp_rhs_140809;
            
            // futhark/microgpt.fut:138:94-163
            
            double zt_res_140811 = zp_res_140810 * zp_res_140810;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_140812 = r_140807 + zt_res_140811;
            double r_tmp_155273 = zp_res_140812;
            
            r_140807 = r_tmp_155273;
        }
        defunc_0_lifted_lambda_res_140805 = r_140807;
        // futhark/microgpt.fut:138:54-182
        
        double zs_res_140813 = defunc_0_lifted_lambda_res_140805 / 16.0;
        
        // futhark/microgpt.fut:139:24-55
        
        double zp_res_140814 = 1.0e-5 + zs_res_140813;
        
        // futhark/microgpt.fut:139:16-55
        
        double sqrt_res_140815 = futrts_sqrt64(zp_res_140814);
        
        // futhark/microgpt.fut:140:85-96
        
        double zs_res_140816 = 1.0 / sqrt_res_140815;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151639 = 0; i_151639 < (int64_t) 16; i_151639++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_140823 = ((double *) wpe_mem_152784.mem)[i_151647 * (int64_t) 16 + i_151639];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_140824 = ((double *) mem_152792)[i_151647 * (int64_t) 16 + i_151639];
            
            // futhark/microgpt.fut:140:38-78
            
            double zp_res_140825 = zp_lhs_140823 + zp_rhs_140824;
            
            // futhark/microgpt.fut:140:56-96
            
            double zt_res_140826 = zs_res_140816 * zp_res_140825;
            
            ((double *) mem_152813)[i_151639] = zt_res_140826;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151643 = 0; i_151643 < (int64_t) 16; i_151643++) {
            // futhark/microgpt.fut:141:4-14
            
            double lifted_lambda_res_140834 = ((double *) mem_152813)[i_151643];
            
            ((double *) mem_152820)[i_151643] = lifted_lambda_res_140834;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152808, i_151647 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152820, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152831_cached_sizze_155765 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152831, &mem_152831_cached_sizze_155765, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152836_cached_sizze_155766 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152836, &mem_152836_cached_sizze_155766, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152843_cached_sizze_155767 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152843, &mem_152843_cached_sizze_155767, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151659 = 0; i_151659 < (int64_t) 16; i_151659++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_140843;
        double r_140845 = 0.0;
        
        for (int64_t i_140844 = 0; i_140844 < (int64_t) 16; i_140844++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_140846 = ((double *) mem_152808)[i_151659 * (int64_t) 16 + i_140844];
            
            // futhark/microgpt.fut:142:78-115
            
            double zt_res_140847 = zt_lhs_140846 * zt_lhs_140846;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_140848 = r_140845 + zt_res_140847;
            double r_tmp_155277 = zp_res_140848;
            
            r_140845 = r_tmp_155277;
        }
        defunc_0_lifted_lambda_res_140843 = r_140845;
        // futhark/microgpt.fut:142:57-133
        
        double zs_res_140849 = defunc_0_lifted_lambda_res_140843 / 16.0;
        
        // futhark/microgpt.fut:143:24-55
        
        double zp_res_140850 = 1.0e-5 + zs_res_140849;
        
        // futhark/microgpt.fut:143:16-55
        
        double sqrt_res_140851 = futrts_sqrt64(zp_res_140850);
        
        // futhark/microgpt.fut:144:59-70
        
        double zs_res_140852 = 1.0 / sqrt_res_140851;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151651 = 0; i_151651 < (int64_t) 16; i_151651++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_140859 = ((double *) mem_152808)[i_151659 * (int64_t) 16 + i_151651];
            
            // futhark/microgpt.fut:144:37-70
            
            double zt_res_140860 = zs_res_140852 * zt_lhs_140859;
            
            ((double *) mem_152836)[i_151651] = zt_res_140860;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151655 = 0; i_151655 < (int64_t) 16; i_151655++) {
            // futhark/microgpt.fut:145:4-14
            
            double lifted_lambda_res_140868 = ((double *) mem_152836)[i_151655];
            
            ((double *) mem_152843)[i_151655] = lifted_lambda_res_140868;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152831, i_151659 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152843, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152854_cached_sizze_155768 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152854, &mem_152854_cached_sizze_155768, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152855_cached_sizze_155769 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152855, &mem_152855_cached_sizze_155769, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152856_cached_sizze_155770 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152856, &mem_152856_cached_sizze_155770, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152869_cached_sizze_155771 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152869, &mem_152869_cached_sizze_155771, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152870_cached_sizze_155772 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152870, &mem_152870_cached_sizze_155772, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152871_cached_sizze_155773 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152871, &mem_152871_cached_sizze_155773, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151677 = 0; i_151677 < (int64_t) 16; i_151677++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151667 = 0; i_151667 < (int64_t) 16; i_151667++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141706;
            double r_141708 = 0.0;
            
            for (int64_t i_141707 = 0; i_141707 < (int64_t) 16; i_141707++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_141709 = ((double *) wqry_mem_152785.mem)[i_151667 * (int64_t) 16 + i_141707];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_141710 = ((double *) mem_152831)[i_151677 * (int64_t) 16 + i_141707];
                
                // futhark/microgpt.fut:146:66-105
                
                double zt_res_141711 = zt_lhs_141709 * zt_rhs_141710;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141712 = r_141708 + zt_res_141711;
                double r_tmp_155286 = zp_res_141712;
                
                r_141708 = r_tmp_155286;
            }
            defunc_0_lifted_lambda_res_141706 = r_141708;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141719;
            double r_141721 = 0.0;
            
            for (int64_t i_141720 = 0; i_141720 < (int64_t) 16; i_141720++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_141722 = ((double *) wkey_mem_152782.mem)[i_151667 * (int64_t) 16 + i_141720];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_141723 = ((double *) mem_152831)[i_151677 * (int64_t) 16 + i_141720];
                
                // futhark/microgpt.fut:147:66-105
                
                double zt_res_141724 = zt_lhs_141722 * zt_rhs_141723;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141725 = r_141721 + zt_res_141724;
                double r_tmp_155287 = zp_res_141725;
                
                r_141721 = r_tmp_155287;
            }
            defunc_0_lifted_lambda_res_141719 = r_141721;
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141735;
            double r_141737 = 0.0;
            
            for (int64_t i_141736 = 0; i_141736 < (int64_t) 16; i_141736++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_141738 = ((double *) wval_mem_152788.mem)[i_151667 * (int64_t) 16 + i_141736];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_141739 = ((double *) mem_152831)[i_151677 * (int64_t) 16 + i_141736];
                
                // futhark/microgpt.fut:148:66-105
                
                double zt_res_141740 = zt_lhs_141738 * zt_rhs_141739;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141741 = r_141737 + zt_res_141740;
                double r_tmp_155288 = zp_res_141741;
                
                r_141737 = r_tmp_155288;
            }
            defunc_0_lifted_lambda_res_141735 = r_141737;
            ((double *) mem_152869)[i_151667] = defunc_0_lifted_lambda_res_141735;
            ((double *) mem_152870)[i_151667] = defunc_0_lifted_lambda_res_141719;
            ((double *) mem_152871)[i_151667] = defunc_0_lifted_lambda_res_141706;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152854, i_151677 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152869, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152855, i_151677 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152870, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152856, i_151677 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152871, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152902_cached_sizze_155774 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152902, &mem_152902_cached_sizze_155774, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152903_cached_sizze_155775 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152903, &mem_152903_cached_sizze_155775, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152904_cached_sizze_155776 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152904, &mem_152904_cached_sizze_155776, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152920_cached_sizze_155777 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152920, &mem_152920_cached_sizze_155777, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152921_cached_sizze_155778 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152921, &mem_152921_cached_sizze_155778, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152922_cached_sizze_155779 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152922, &mem_152922_cached_sizze_155779, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152935_cached_sizze_155780 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152935, &mem_152935_cached_sizze_155780, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152936_cached_sizze_155781 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152936, &mem_152936_cached_sizze_155781, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152937_cached_sizze_155782 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152937, &mem_152937_cached_sizze_155782, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151707 = 0; i_151707 < (int64_t) 4; i_151707++) {
        // futhark/microgpt.fut:149:69-72
        
        int64_t zp_lhs_141582 = mul64((int64_t) 4, i_151707);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151697 = 0; i_151697 < (int64_t) 16; i_151697++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151687 = 0; i_151687 < (int64_t) 4; i_151687++) {
                // futhark/microgpt.fut:149:74-81
                
                int64_t tmp_141899 = add64(zp_lhs_141582, i_151687);
                
                // futhark/microgpt.fut:149:51-83
                
                bool x_141900 = sle64((int64_t) 0, tmp_141899);
                
                // futhark/microgpt.fut:149:51-83
                
                bool y_141901 = slt64(tmp_141899, (int64_t) 16);
                
                // futhark/microgpt.fut:149:51-83
                
                bool bounds_check_141902 = x_141900 && y_141901;
                
                // futhark/microgpt.fut:149:51-83
                
                bool index_certs_141903;
                
                if (!bounds_check_141902) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_141899, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:149:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:149:15-84\n   #9  futhark/microgpt.fut:453:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141904 = ((double *) mem_152856)[i_151697 * (int64_t) 16 + tmp_141899];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141912 = ((double *) mem_152855)[i_151697 * (int64_t) 16 + tmp_141899];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141923 = ((double *) mem_152854)[i_151697 * (int64_t) 16 + tmp_141899];
                
                ((double *) mem_152935)[i_151687] = lifted_lambda_res_141923;
                ((double *) mem_152936)[i_151687] = lifted_lambda_res_141912;
                ((double *) mem_152937)[i_151687] = lifted_lambda_res_141904;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152920, i_151697 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152935, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152921, i_151697 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152936, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152922, i_151697 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152937, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_152902, i_151707 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_152920, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_152903, i_151707 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_152921, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_152904, i_151707 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_152922, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152983_cached_sizze_155783 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152983, &mem_152983_cached_sizze_155783, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152989_cached_sizze_155784 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152989, &mem_152989_cached_sizze_155784, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152994_cached_sizze_155785 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152994, &mem_152994_cached_sizze_155785, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153005_cached_sizze_155786 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153005, &mem_153005_cached_sizze_155786, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153010_cached_sizze_155787 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153010, &mem_153010_cached_sizze_155787, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153021_cached_sizze_155788 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153021, &mem_153021_cached_sizze_155788, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153026_cached_sizze_155789 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153026, &mem_153026_cached_sizze_155789, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153033_cached_sizze_155790 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153033, &mem_153033_cached_sizze_155790, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153040_cached_sizze_155791 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153040, &mem_153040_cached_sizze_155791, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153051_cached_sizze_155792 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153051, &mem_153051_cached_sizze_155792, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153056_cached_sizze_155793 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_153056, &mem_153056_cached_sizze_155793, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153067_cached_sizze_155794 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153067, &mem_153067_cached_sizze_155794, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153072_cached_sizze_155795 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_153072, &mem_153072_cached_sizze_155795, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151763 = 0; i_151763 < (int64_t) 4; i_151763++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151717 = 0; i_151717 < (int64_t) 16; i_151717++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151713 = 0; i_151713 < (int64_t) 16; i_151713++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_141013;
                double r_141015 = 0.0;
                
                for (int64_t i_141014 = 0; i_141014 < (int64_t) 4; i_141014++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_141016 = ((double *) mem_152904)[i_151763 * (int64_t) 64 + i_151717 * (int64_t) 4 + i_141014];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_141017 = ((double *) mem_152903)[i_151763 * (int64_t) 64 + i_151713 * (int64_t) 4 + i_141014];
                    
                    // futhark/microgpt.fut:152:113-164
                    
                    double zt_res_141018 = zt_lhs_141016 * zt_rhs_141017;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_141019 = r_141015 + zt_res_141018;
                    double r_tmp_155301 = zp_res_141019;
                    
                    r_141015 = r_tmp_155301;
                }
                defunc_0_lifted_lambda_res_141013 = r_141015;
                ((double *) mem_152994)[i_151713] = defunc_0_lifted_lambda_res_141013;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152989, i_151717 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152994, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151725 = 0; i_151725 < (int64_t) 16; i_151725++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151721 = 0; i_151721 < (int64_t) 16; i_151721++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_141034 = ((double *) mem_152989)[i_151725 * (int64_t) 16 + i_151721];
                
                // futhark/microgpt.fut:153:47-78
                
                double zs_res_141035 = zs_lhs_141034 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_141036 = ((double *) mask_mem_152791.mem)[i_151725 * (int64_t) 16 + i_151721];
                
                // futhark/microgpt.fut:153:65-102
                
                double zp_res_141037 = zs_res_141035 + zp_rhs_141036;
                
                ((double *) mem_153010)[i_151721] = zp_res_141037;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153005, i_151725 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153010, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151743 = 0; i_151743 < (int64_t) 16; i_151743++) {
            // futhark/microgpt.fut:105:13-33
            
            double defunc_0_reduce_res_142001;
            double redout_151727 = -INFINITY;
            
            for (int64_t i_151728 = 0; i_151728 < (int64_t) 16; i_151728++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141950 = ((double *) mem_153005)[i_151743 * (int64_t) 16 + i_151728];
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_141058 = fmax64(lifted_lambda_res_141950, redout_151727);
                double redout_tmp_155305 = max_res_141058;
                
                redout_151727 = redout_tmp_155305;
            }
            defunc_0_reduce_res_142001 = redout_151727;
            // futhark/microgpt.fut:155:67-76
            
            double neg_res_141059 = -defunc_0_reduce_res_142001;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151731 = 0; i_151731 < (int64_t) 16; i_151731++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_141066 = ((double *) mem_153005)[i_151743 * (int64_t) 16 + i_151731];
                
                // futhark/microgpt.fut:155:44-76
                
                double zp_res_141067 = neg_res_141059 + zp_lhs_141066;
                
                // futhark/microgpt.fut:155:37-76
                
                double exp_res_141068 = futrts_exp64(zp_res_141067);
                
                ((double *) mem_153026)[i_151731] = exp_res_141068;
            }
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141070;
            double r_141072 = 0.0;
            
            for (int64_t i_141071 = 0; i_141071 < (int64_t) 16; i_141071++) {
                // futhark/microgpt.fut:156:36-46
                
                double lifted_lambda_res_141073 = ((double *) mem_153026)[i_141071];
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141074 = r_141072 + lifted_lambda_res_141073;
                double r_tmp_155307 = zp_res_141074;
                
                r_141072 = r_tmp_155307;
            }
            defunc_0_lifted_lambda_res_141070 = r_141072;
            // futhark/microgpt.fut:157:53-64
            
            double zs_res_141075 = 1.0 / defunc_0_lifted_lambda_res_141070;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151735 = 0; i_151735 < (int64_t) 16; i_151735++) {
                // futhark/microgpt.fut:157:37-47
                
                double zt_lhs_141082 = ((double *) mem_153026)[i_151735];
                
                // futhark/microgpt.fut:157:37-64
                
                double zt_res_141083 = zs_res_141075 * zt_lhs_141082;
                
                ((double *) mem_153033)[i_151735] = zt_res_141083;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151739 = 0; i_151739 < (int64_t) 16; i_151739++) {
                // futhark/microgpt.fut:158:4-14
                
                double lifted_lambda_res_141091 = ((double *) mem_153033)[i_151739];
                
                ((double *) mem_153040)[i_151739] = lifted_lambda_res_141091;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153021, i_151743 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153040, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151751 = 0; i_151751 < (int64_t) 16; i_151751++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151747 = 0; i_151747 < (int64_t) 4; i_151747++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_141106;
                double r_141108 = 0.0;
                
                for (int64_t i_141107 = 0; i_141107 < (int64_t) 16; i_141107++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_141109 = ((double *) mem_153021)[i_151751 * (int64_t) 16 + i_141107];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_141110 = ((double *) mem_152902)[i_151763 * (int64_t) 64 + i_141107 * (int64_t) 4 + i_151747];
                    
                    // futhark/microgpt.fut:159:66-111
                    
                    double zt_res_141111 = zt_lhs_141109 * zt_rhs_141110;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_141112 = r_141108 + zt_res_141111;
                    double r_tmp_155312 = zp_res_141112;
                    
                    r_141108 = r_tmp_155312;
                }
                defunc_0_lifted_lambda_res_141106 = r_141108;
                ((double *) mem_153056)[i_151747] = defunc_0_lifted_lambda_res_141106;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153051, i_151751 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153056, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151759 = 0; i_151759 < (int64_t) 16; i_151759++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151755 = 0; i_151755 < (int64_t) 4; i_151755++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141127 = ((double *) mem_153051)[i_151759 * (int64_t) 4 + i_151755];
                
                ((double *) mem_153072)[i_151755] = lifted_lambda_res_141127;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153067, i_151759 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153072, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_152983, i_151763 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_153067, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153088_cached_sizze_155796 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153088, &mem_153088_cached_sizze_155796, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153093_cached_sizze_155797 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153093, &mem_153093_cached_sizze_155797, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151771 = 0; i_151771 < (int64_t) 16; i_151771++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151767 = 0; i_151767 < (int64_t) 16; i_151767++) {
            // futhark/microgpt.fut:161:54-57
            
            int64_t tmp_141139 = sdiv64(i_151767, (int64_t) 4);
            
            // futhark/microgpt.fut:161:44-59
            
            bool x_141140 = sle64((int64_t) 0, tmp_141139);
            
            // futhark/microgpt.fut:161:44-59
            
            bool y_141141 = slt64(tmp_141139, (int64_t) 4);
            
            // futhark/microgpt.fut:161:44-59
            
            bool bounds_check_141142 = x_141140 && y_141141;
            
            // futhark/microgpt.fut:161:44-59
            
            bool index_certs_141143;
            
            if (!bounds_check_141142) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_141139, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:161:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:161:15-80\n   #6  futhark/microgpt.fut:453:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:161:74-77
            
            int64_t tmp_141144 = smod64(i_151767, (int64_t) 4);
            
            // futhark/microgpt.fut:161:44-79
            
            bool x_141145 = sle64((int64_t) 0, tmp_141144);
            
            // futhark/microgpt.fut:161:44-79
            
            bool y_141146 = slt64(tmp_141144, (int64_t) 4);
            
            // futhark/microgpt.fut:161:44-79
            
            bool bounds_check_141147 = x_141145 && y_141146;
            
            // futhark/microgpt.fut:161:44-79
            
            bool index_certs_141148;
            
            if (!bounds_check_141147) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_141144, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:161:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:161:15-80\n   #6  futhark/microgpt.fut:453:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_141149 = ((double *) mem_152983)[tmp_141139 * (int64_t) 64 + i_151771 * (int64_t) 4 + tmp_141144];
            
            ((double *) mem_153093)[i_151767] = lifted_lambda_res_141149;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153088, i_151771 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153093, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153104_cached_sizze_155798 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153104, &mem_153104_cached_sizze_155798, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153109_cached_sizze_155799 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153109, &mem_153109_cached_sizze_155799, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151779 = 0; i_151779 < (int64_t) 16; i_151779++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151775 = 0; i_151775 < (int64_t) 16; i_151775++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141164;
            double r_141166 = 0.0;
            
            for (int64_t i_141165 = 0; i_141165 < (int64_t) 16; i_141165++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_141167 = ((double *) wout_mem_152783.mem)[i_151775 * (int64_t) 16 + i_141165];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_141168 = ((double *) mem_153088)[i_151779 * (int64_t) 16 + i_141165];
                
                // futhark/microgpt.fut:162:67-106
                
                double zt_res_141169 = zt_lhs_141167 * zt_rhs_141168;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141170 = r_141166 + zt_res_141169;
                double r_tmp_155319 = zp_res_141170;
                
                r_141166 = r_tmp_155319;
            }
            defunc_0_lifted_lambda_res_141164 = r_141166;
            ((double *) mem_153109)[i_151775] = defunc_0_lifted_lambda_res_141164;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153104, i_151779 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153109, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153120_cached_sizze_155800 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153120, &mem_153120_cached_sizze_155800, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153125_cached_sizze_155801 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153125, &mem_153125_cached_sizze_155801, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151787 = 0; i_151787 < (int64_t) 16; i_151787++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151783 = 0; i_151783 < (int64_t) 16; i_151783++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_141185 = ((double *) mem_153104)[i_151787 * (int64_t) 16 + i_151783];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_141186 = ((double *) mem_152808)[i_151787 * (int64_t) 16 + i_151783];
            
            // futhark/microgpt.fut:163:46-84
            
            double zp_res_141187 = zp_lhs_141185 + zp_rhs_141186;
            
            ((double *) mem_153125)[i_151783] = zp_res_141187;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153120, i_151787 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153125, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153136_cached_sizze_155802 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153136, &mem_153136_cached_sizze_155802, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153141_cached_sizze_155803 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153141, &mem_153141_cached_sizze_155803, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153148_cached_sizze_155804 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153148, &mem_153148_cached_sizze_155804, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151799 = 0; i_151799 < (int64_t) 16; i_151799++) {
        // futhark/microgpt.fut:61:13-49
        
        double defunc_0_lifted_lambda_res_141196;
        double r_141198 = 0.0;
        
        for (int64_t i_141197 = 0; i_141197 < (int64_t) 16; i_141197++) {
            // futhark/microgpt.fut:61:46-49
            
            double zt_lhs_141199 = ((double *) mem_153120)[i_151799 * (int64_t) 16 + i_141197];
            
            // futhark/microgpt.fut:164:79-118
            
            double zt_res_141200 = zt_lhs_141199 * zt_lhs_141199;
            
            // futhark/microgpt.fut:61:40-49
            
            double zp_res_141201 = r_141198 + zt_res_141200;
            double r_tmp_155323 = zp_res_141201;
            
            r_141198 = r_tmp_155323;
        }
        defunc_0_lifted_lambda_res_141196 = r_141198;
        // futhark/microgpt.fut:164:58-136
        
        double zs_res_141202 = defunc_0_lifted_lambda_res_141196 / 16.0;
        
        // futhark/microgpt.fut:165:24-55
        
        double zp_res_141203 = 1.0e-5 + zs_res_141202;
        
        // futhark/microgpt.fut:165:16-55
        
        double sqrt_res_141204 = futrts_sqrt64(zp_res_141203);
        
        // futhark/microgpt.fut:166:60-71
        
        double zs_res_141205 = 1.0 / sqrt_res_141204;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151791 = 0; i_151791 < (int64_t) 16; i_151791++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_141212 = ((double *) mem_153120)[i_151799 * (int64_t) 16 + i_151791];
            
            // futhark/microgpt.fut:166:37-71
            
            double zt_res_141213 = zs_res_141205 * zt_lhs_141212;
            
            ((double *) mem_153141)[i_151791] = zt_res_141213;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151795 = 0; i_151795 < (int64_t) 16; i_151795++) {
            // futhark/microgpt.fut:167:4-14
            
            double lifted_lambda_res_141221 = ((double *) mem_153141)[i_151795];
            
            ((double *) mem_153148)[i_151795] = lifted_lambda_res_141221;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153136, i_151799 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153148, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153159_cached_sizze_155805 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153159, &mem_153159_cached_sizze_155805, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153164_cached_sizze_155806 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153164, &mem_153164_cached_sizze_155806, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151807 = 0; i_151807 < (int64_t) 16; i_151807++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151803 = 0; i_151803 < (int64_t) 64; i_151803++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141237;
            double r_141239 = 0.0;
            
            for (int64_t i_141238 = 0; i_141238 < (int64_t) 16; i_141238++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_141240 = ((double *) wup_mem_152787.mem)[i_151803 * (int64_t) 16 + i_141238];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_141241 = ((double *) mem_153136)[i_151807 * (int64_t) 16 + i_141238];
                
                // futhark/microgpt.fut:168:67-106
                
                double zt_res_141242 = zt_lhs_141240 * zt_rhs_141241;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141243 = r_141239 + zt_res_141242;
                double r_tmp_155328 = zp_res_141243;
                
                r_141239 = r_tmp_155328;
            }
            defunc_0_lifted_lambda_res_141237 = r_141239;
            ((double *) mem_153164)[i_151803] = defunc_0_lifted_lambda_res_141237;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153159, i_151807 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153164, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153175_cached_sizze_155807 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153175, &mem_153175_cached_sizze_155807, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153180_cached_sizze_155808 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153180, &mem_153180_cached_sizze_155808, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151815 = 0; i_151815 < (int64_t) 16; i_151815++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151811 = 0; i_151811 < (int64_t) 64; i_151811++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_141258 = ((double *) mem_153159)[i_151815 * (int64_t) 64 + i_151811];
            
            // futhark/microgpt.fut:169:45-73
            
            double max_res_141259 = fmax64(0.0, max_arg0_141258);
            
            ((double *) mem_153180)[i_151811] = max_res_141259;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153175, i_151815 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153180, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153191_cached_sizze_155809 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153191, &mem_153191_cached_sizze_155809, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153196_cached_sizze_155810 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153196, &mem_153196_cached_sizze_155810, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151823 = 0; i_151823 < (int64_t) 16; i_151823++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151819 = 0; i_151819 < (int64_t) 16; i_151819++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141274;
            double r_141276 = 0.0;
            
            for (int64_t i_141275 = 0; i_141275 < (int64_t) 64; i_141275++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_141277 = ((double *) wdown_mem_152781.mem)[i_151819 * (int64_t) 64 + i_141275];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_141278 = ((double *) mem_153175)[i_151823 * (int64_t) 64 + i_141275];
                
                // futhark/microgpt.fut:170:67-108
                
                double zt_res_141279 = zt_lhs_141277 * zt_rhs_141278;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141280 = r_141276 + zt_res_141279;
                double r_tmp_155333 = zp_res_141280;
                
                r_141276 = r_tmp_155333;
            }
            defunc_0_lifted_lambda_res_141274 = r_141276;
            ((double *) mem_153196)[i_151819] = defunc_0_lifted_lambda_res_141274;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153191, i_151823 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153196, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153207_cached_sizze_155811 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153207, &mem_153207_cached_sizze_155811, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153212_cached_sizze_155812 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153212, &mem_153212_cached_sizze_155812, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151831 = 0; i_151831 < (int64_t) 16; i_151831++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151827 = 0; i_151827 < (int64_t) 16; i_151827++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_141295 = ((double *) mem_153191)[i_151831 * (int64_t) 16 + i_151827];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_141296 = ((double *) mem_153120)[i_151831 * (int64_t) 16 + i_151827];
            
            // futhark/microgpt.fut:171:46-85
            
            double zp_res_141297 = zp_lhs_141295 + zp_rhs_141296;
            
            ((double *) mem_153212)[i_151827] = zp_res_141297;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153207, i_151831 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153212, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153223_cached_sizze_155813 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_153223, &mem_153223_cached_sizze_155813, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153228_cached_sizze_155814 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_153228, &mem_153228_cached_sizze_155814, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151839 = 0; i_151839 < (int64_t) 16; i_151839++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151835 = 0; i_151835 < (int64_t) 27; i_151835++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141313;
            double r_141315 = 0.0;
            
            for (int64_t i_141314 = 0; i_141314 < (int64_t) 16; i_141314++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_141316 = ((double *) wvoc_mem_152789.mem)[i_151835 * (int64_t) 16 + i_141314];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_141317 = ((double *) mem_153207)[i_151839 * (int64_t) 16 + i_141314];
                
                // futhark/microgpt.fut:172:67-107
                
                double zt_res_141318 = zt_lhs_141316 * zt_rhs_141317;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141319 = r_141315 + zt_res_141318;
                double r_tmp_155338 = zp_res_141319;
                
                r_141315 = r_tmp_155338;
            }
            defunc_0_lifted_lambda_res_141313 = r_141315;
            ((double *) mem_153228)[i_151835] = defunc_0_lifted_lambda_res_141313;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153223, i_151839 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153228, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_153239, (int64_t) 3456, "mem_153239")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153244_cached_sizze_155815 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_153244, &mem_153244_cached_sizze_155815, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151847 = 0; i_151847 < (int64_t) 16; i_151847++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151843 = 0; i_151843 < (int64_t) 27; i_151843++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_141334 = ((double *) mem_153223)[i_151847 * (int64_t) 27 + i_151843];
            
            ((double *) mem_153244)[i_151843] = lifted_lambda_res_141334;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_153239.mem, i_151847 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153244, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_155269, &mem_153239, "mem_153239") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155759, &mem_out_155269, "mem_out_155269") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_152792);
        free(mem_152797);
        free(mem_152808);
        free(mem_152813);
        free(mem_152820);
        free(mem_152831);
        free(mem_152836);
        free(mem_152843);
        free(mem_152854);
        free(mem_152855);
        free(mem_152856);
        free(mem_152869);
        free(mem_152870);
        free(mem_152871);
        free(mem_152902);
        free(mem_152903);
        free(mem_152904);
        free(mem_152920);
        free(mem_152921);
        free(mem_152922);
        free(mem_152935);
        free(mem_152936);
        free(mem_152937);
        free(mem_152983);
        free(mem_152989);
        free(mem_152994);
        free(mem_153005);
        free(mem_153010);
        free(mem_153021);
        free(mem_153026);
        free(mem_153033);
        free(mem_153040);
        free(mem_153051);
        free(mem_153056);
        free(mem_153067);
        free(mem_153072);
        free(mem_153088);
        free(mem_153093);
        free(mem_153104);
        free(mem_153109);
        free(mem_153120);
        free(mem_153125);
        free(mem_153136);
        free(mem_153141);
        free(mem_153148);
        free(mem_153159);
        free(mem_153164);
        free(mem_153175);
        free(mem_153180);
        free(mem_153191);
        free(mem_153196);
        free(mem_153207);
        free(mem_153212);
        free(mem_153223);
        free(mem_153228);
        free(mem_153244);
        if (memblock_unref(ctx, &mem_153239, "mem_153239") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155269, "mem_out_155269") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_155816, struct memblock *mem_out_p_155817, struct memblock *mem_out_p_155818, struct memblock *mem_out_p_155819, struct memblock *mem_out_p_155820, struct memblock *mem_out_p_155821, struct memblock *mem_out_p_155822, struct memblock *mem_out_p_155823, struct memblock *mem_out_p_155824, struct memblock wte_mem_152781, struct memblock wpe_mem_152782, struct memblock wqry_mem_152783, struct memblock wkey_mem_152784, struct memblock wval_mem_152785, struct memblock wout_mem_152786, struct memblock wup_mem_152787, struct memblock wdown_mem_152788, struct memblock wvoc_mem_152789)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_155277;
    
    mem_out_155277.references = NULL;
    
    struct memblock mem_out_155276;
    
    mem_out_155276.references = NULL;
    
    struct memblock mem_out_155275;
    
    mem_out_155275.references = NULL;
    
    struct memblock mem_out_155274;
    
    mem_out_155274.references = NULL;
    
    struct memblock mem_out_155273;
    
    mem_out_155273.references = NULL;
    
    struct memblock mem_out_155272;
    
    mem_out_155272.references = NULL;
    
    struct memblock mem_out_155271;
    
    mem_out_155271.references = NULL;
    
    struct memblock mem_out_155270;
    
    mem_out_155270.references = NULL;
    
    struct memblock mem_out_155269;
    
    mem_out_155269.references = NULL;
    
    struct memblock mem_152772 = ctx->constants->mem_152772;
    struct memblock mem_152773 = ctx->constants->mem_152773;
    struct memblock mem_152774 = ctx->constants->mem_152774;
    struct memblock mem_152775 = ctx->constants->mem_152775;
    struct memblock mem_152776 = ctx->constants->mem_152776;
    struct memblock mem_152777 = ctx->constants->mem_152777;
    struct memblock mem_152778 = ctx->constants->mem_152778;
    struct memblock mem_152779 = ctx->constants->mem_152779;
    struct memblock mem_152780 = ctx->constants->mem_152780;
    
    if (memblock_set(ctx, &mem_out_155269, &wdown_mem_152788, "wdown_mem_152788") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155270, &wkey_mem_152784, "wkey_mem_152784") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155271, &wout_mem_152786, "wout_mem_152786") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155272, &wpe_mem_152782, "wpe_mem_152782") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155273, &wqry_mem_152783, "wqry_mem_152783") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155274, &wte_mem_152781, "wte_mem_152781") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155275, &wup_mem_152787, "wup_mem_152787") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155276, &wval_mem_152785, "wval_mem_152785") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155277, &wvoc_mem_152789, "wvoc_mem_152789") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155816, &mem_out_155269, "mem_out_155269") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155817, &mem_out_155270, "mem_out_155270") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155818, &mem_out_155271, "mem_out_155271") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155819, &mem_out_155272, "mem_out_155272") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155820, &mem_out_155273, "mem_out_155273") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155821, &mem_out_155274, "mem_out_155274") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155822, &mem_out_155275, "mem_out_155275") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155823, &mem_out_155276, "mem_out_155276") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155824, &mem_out_155277, "mem_out_155277") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_155277, "mem_out_155277") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155276, "mem_out_155276") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155275, "mem_out_155275") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155274, "mem_out_155274") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155273, "mem_out_155273") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155272, "mem_out_155272") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155271, "mem_out_155271") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155270, "mem_out_155270") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155269, "mem_out_155269") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_155825, struct memblock *mem_out_p_155826, struct memblock *mem_out_p_155827, struct memblock *mem_out_p_155828, struct memblock *mem_out_p_155829, struct memblock *mem_out_p_155830, struct memblock *mem_out_p_155831, struct memblock *mem_out_p_155832, struct memblock *mem_out_p_155833, struct memblock *mem_out_p_155834, struct memblock *mem_out_p_155835, struct memblock *mem_out_p_155836, struct memblock *mem_out_p_155837, struct memblock *mem_out_p_155838, struct memblock *mem_out_p_155839, struct memblock *mem_out_p_155840, struct memblock *mem_out_p_155841, struct memblock *mem_out_p_155842, struct memblock *mem_out_p_155843, struct memblock *mem_out_p_155844, struct memblock *mem_out_p_155845, struct memblock *mem_out_p_155846, struct memblock *mem_out_p_155847, struct memblock *mem_out_p_155848, struct memblock *mem_out_p_155849, struct memblock *mem_out_p_155850, struct memblock *mem_out_p_155851, struct memblock wdown_mem_152781, struct memblock wkey_mem_152782, struct memblock wout_mem_152783, struct memblock wpe_mem_152784, struct memblock wqry_mem_152785, struct memblock wte_mem_152786, struct memblock wup_mem_152787, struct memblock wval_mem_152788, struct memblock wvoc_mem_152789, struct memblock wdown_mem_152790, struct memblock wkey_mem_152791, struct memblock wout_mem_152792, struct memblock wpe_mem_152793, struct memblock wqry_mem_152794, struct memblock wte_mem_152795, struct memblock wup_mem_152796, struct memblock wval_mem_152797, struct memblock wvoc_mem_152798, struct memblock wdown_mem_152799, struct memblock wkey_mem_152800, struct memblock wout_mem_152801, struct memblock wpe_mem_152802, struct memblock wqry_mem_152803, struct memblock wte_mem_152804, struct memblock wup_mem_152805, struct memblock wval_mem_152806, struct memblock wvoc_mem_152807, struct memblock masks_mem_152808, struct memblock dls_mem_152809, struct memblock seqs_mem_152810, int64_t num_steps_112428)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_152919_cached_sizze_155852 = 0;
    unsigned char *mem_152919 = NULL;
    int64_t mem_152920_cached_sizze_155853 = 0;
    unsigned char *mem_152920 = NULL;
    int64_t mem_152929_cached_sizze_155854 = 0;
    unsigned char *mem_152929 = NULL;
    int64_t mem_152936_cached_sizze_155855 = 0;
    unsigned char *mem_152936 = NULL;
    int64_t mem_152951_cached_sizze_155856 = 0;
    unsigned char *mem_152951 = NULL;
    int64_t mem_152952_cached_sizze_155857 = 0;
    unsigned char *mem_152952 = NULL;
    int64_t mem_152953_cached_sizze_155858 = 0;
    unsigned char *mem_152953 = NULL;
    int64_t mem_152972_cached_sizze_155859 = 0;
    unsigned char *mem_152972 = NULL;
    int64_t mem_152979_cached_sizze_155860 = 0;
    unsigned char *mem_152979 = NULL;
    int64_t mem_152984_cached_sizze_155861 = 0;
    unsigned char *mem_152984 = NULL;
    int64_t mem_152995_cached_sizze_155862 = 0;
    unsigned char *mem_152995 = NULL;
    int64_t mem_153000_cached_sizze_155863 = 0;
    unsigned char *mem_153000 = NULL;
    int64_t mem_153011_cached_sizze_155864 = 0;
    unsigned char *mem_153011 = NULL;
    int64_t mem_153012_cached_sizze_155865 = 0;
    unsigned char *mem_153012 = NULL;
    int64_t mem_153025_cached_sizze_155866 = 0;
    unsigned char *mem_153025 = NULL;
    int64_t mem_153032_cached_sizze_155867 = 0;
    unsigned char *mem_153032 = NULL;
    int64_t mem_153037_cached_sizze_155868 = 0;
    unsigned char *mem_153037 = NULL;
    int64_t mem_153048_cached_sizze_155869 = 0;
    unsigned char *mem_153048 = NULL;
    int64_t mem_153053_cached_sizze_155870 = 0;
    unsigned char *mem_153053 = NULL;
    int64_t mem_153064_cached_sizze_155871 = 0;
    unsigned char *mem_153064 = NULL;
    int64_t mem_153065_cached_sizze_155872 = 0;
    unsigned char *mem_153065 = NULL;
    int64_t mem_153066_cached_sizze_155873 = 0;
    unsigned char *mem_153066 = NULL;
    int64_t mem_153082_cached_sizze_155874 = 0;
    unsigned char *mem_153082 = NULL;
    int64_t mem_153083_cached_sizze_155875 = 0;
    unsigned char *mem_153083 = NULL;
    int64_t mem_153084_cached_sizze_155876 = 0;
    unsigned char *mem_153084 = NULL;
    int64_t mem_153097_cached_sizze_155877 = 0;
    unsigned char *mem_153097 = NULL;
    int64_t mem_153098_cached_sizze_155878 = 0;
    unsigned char *mem_153098 = NULL;
    int64_t mem_153099_cached_sizze_155879 = 0;
    unsigned char *mem_153099 = NULL;
    int64_t mem_153145_cached_sizze_155880 = 0;
    unsigned char *mem_153145 = NULL;
    int64_t mem_153146_cached_sizze_155881 = 0;
    unsigned char *mem_153146 = NULL;
    int64_t mem_153147_cached_sizze_155882 = 0;
    unsigned char *mem_153147 = NULL;
    int64_t mem_153148_cached_sizze_155883 = 0;
    unsigned char *mem_153148 = NULL;
    int64_t mem_153169_cached_sizze_155884 = 0;
    unsigned char *mem_153169 = NULL;
    int64_t mem_153170_cached_sizze_155885 = 0;
    unsigned char *mem_153170 = NULL;
    int64_t mem_153171_cached_sizze_155886 = 0;
    unsigned char *mem_153171 = NULL;
    int64_t mem_153172_cached_sizze_155887 = 0;
    unsigned char *mem_153172 = NULL;
    int64_t mem_153189_cached_sizze_155888 = 0;
    unsigned char *mem_153189 = NULL;
    int64_t mem_153190_cached_sizze_155889 = 0;
    unsigned char *mem_153190 = NULL;
    int64_t mem_153191_cached_sizze_155890 = 0;
    unsigned char *mem_153191 = NULL;
    int64_t mem_153192_cached_sizze_155891 = 0;
    unsigned char *mem_153192 = NULL;
    int64_t mem_153253_cached_sizze_155892 = 0;
    unsigned char *mem_153253 = NULL;
    int64_t mem_153254_cached_sizze_155893 = 0;
    unsigned char *mem_153254 = NULL;
    int64_t mem_153255_cached_sizze_155894 = 0;
    unsigned char *mem_153255 = NULL;
    int64_t mem_153256_cached_sizze_155895 = 0;
    unsigned char *mem_153256 = NULL;
    int64_t mem_153277_cached_sizze_155896 = 0;
    unsigned char *mem_153277 = NULL;
    int64_t mem_153278_cached_sizze_155897 = 0;
    unsigned char *mem_153278 = NULL;
    int64_t mem_153279_cached_sizze_155898 = 0;
    unsigned char *mem_153279 = NULL;
    int64_t mem_153280_cached_sizze_155899 = 0;
    unsigned char *mem_153280 = NULL;
    int64_t mem_153297_cached_sizze_155900 = 0;
    unsigned char *mem_153297 = NULL;
    int64_t mem_153298_cached_sizze_155901 = 0;
    unsigned char *mem_153298 = NULL;
    int64_t mem_153299_cached_sizze_155902 = 0;
    unsigned char *mem_153299 = NULL;
    int64_t mem_153300_cached_sizze_155903 = 0;
    unsigned char *mem_153300 = NULL;
    int64_t mem_153361_cached_sizze_155904 = 0;
    unsigned char *mem_153361 = NULL;
    int64_t mem_153362_cached_sizze_155905 = 0;
    unsigned char *mem_153362 = NULL;
    int64_t mem_153363_cached_sizze_155906 = 0;
    unsigned char *mem_153363 = NULL;
    int64_t mem_153364_cached_sizze_155907 = 0;
    unsigned char *mem_153364 = NULL;
    int64_t mem_153365_cached_sizze_155908 = 0;
    unsigned char *mem_153365 = NULL;
    int64_t mem_153366_cached_sizze_155909 = 0;
    unsigned char *mem_153366 = NULL;
    int64_t mem_153367_cached_sizze_155910 = 0;
    unsigned char *mem_153367 = NULL;
    int64_t mem_153368_cached_sizze_155911 = 0;
    unsigned char *mem_153368 = NULL;
    int64_t mem_153401_cached_sizze_155912 = 0;
    unsigned char *mem_153401 = NULL;
    int64_t mem_153402_cached_sizze_155913 = 0;
    unsigned char *mem_153402 = NULL;
    int64_t mem_153403_cached_sizze_155914 = 0;
    unsigned char *mem_153403 = NULL;
    int64_t mem_153404_cached_sizze_155915 = 0;
    unsigned char *mem_153404 = NULL;
    int64_t mem_153405_cached_sizze_155916 = 0;
    unsigned char *mem_153405 = NULL;
    int64_t mem_153406_cached_sizze_155917 = 0;
    unsigned char *mem_153406 = NULL;
    int64_t mem_153407_cached_sizze_155918 = 0;
    unsigned char *mem_153407 = NULL;
    int64_t mem_153408_cached_sizze_155919 = 0;
    unsigned char *mem_153408 = NULL;
    int64_t mem_153489_cached_sizze_155920 = 0;
    unsigned char *mem_153489 = NULL;
    int64_t mem_153490_cached_sizze_155921 = 0;
    unsigned char *mem_153490 = NULL;
    int64_t mem_153491_cached_sizze_155922 = 0;
    unsigned char *mem_153491 = NULL;
    int64_t mem_153492_cached_sizze_155923 = 0;
    unsigned char *mem_153492 = NULL;
    int64_t mem_153513_cached_sizze_155924 = 0;
    unsigned char *mem_153513 = NULL;
    int64_t mem_153514_cached_sizze_155925 = 0;
    unsigned char *mem_153514 = NULL;
    int64_t mem_153515_cached_sizze_155926 = 0;
    unsigned char *mem_153515 = NULL;
    int64_t mem_153516_cached_sizze_155927 = 0;
    unsigned char *mem_153516 = NULL;
    int64_t mem_153533_cached_sizze_155928 = 0;
    unsigned char *mem_153533 = NULL;
    int64_t mem_153534_cached_sizze_155929 = 0;
    unsigned char *mem_153534 = NULL;
    int64_t mem_153535_cached_sizze_155930 = 0;
    unsigned char *mem_153535 = NULL;
    int64_t mem_153536_cached_sizze_155931 = 0;
    unsigned char *mem_153536 = NULL;
    int64_t mem_153597_cached_sizze_155932 = 0;
    unsigned char *mem_153597 = NULL;
    int64_t mem_153598_cached_sizze_155933 = 0;
    unsigned char *mem_153598 = NULL;
    int64_t mem_153607_cached_sizze_155934 = 0;
    unsigned char *mem_153607 = NULL;
    int64_t mem_153608_cached_sizze_155935 = 0;
    unsigned char *mem_153608 = NULL;
    int64_t mem_153629_cached_sizze_155936 = 0;
    unsigned char *mem_153629 = NULL;
    int64_t mem_153630_cached_sizze_155937 = 0;
    unsigned char *mem_153630 = NULL;
    int64_t mem_153641_cached_sizze_155938 = 0;
    unsigned char *mem_153641 = NULL;
    int64_t mem_153642_cached_sizze_155939 = 0;
    unsigned char *mem_153642 = NULL;
    int64_t mem_153651_cached_sizze_155940 = 0;
    unsigned char *mem_153651 = NULL;
    int64_t mem_153652_cached_sizze_155941 = 0;
    unsigned char *mem_153652 = NULL;
    int64_t mem_153683_cached_sizze_155942 = 0;
    unsigned char *mem_153683 = NULL;
    int64_t mem_153684_cached_sizze_155943 = 0;
    unsigned char *mem_153684 = NULL;
    int64_t mem_153695_cached_sizze_155944 = 0;
    unsigned char *mem_153695 = NULL;
    int64_t mem_153696_cached_sizze_155945 = 0;
    unsigned char *mem_153696 = NULL;
    int64_t mem_153705_cached_sizze_155946 = 0;
    unsigned char *mem_153705 = NULL;
    int64_t mem_153706_cached_sizze_155947 = 0;
    unsigned char *mem_153706 = NULL;
    int64_t mem_153737_cached_sizze_155948 = 0;
    unsigned char *mem_153737 = NULL;
    int64_t mem_153743_cached_sizze_155949 = 0;
    unsigned char *mem_153743 = NULL;
    int64_t mem_153748_cached_sizze_155950 = 0;
    unsigned char *mem_153748 = NULL;
    int64_t mem_153764_cached_sizze_155951 = 0;
    unsigned char *mem_153764 = NULL;
    int64_t mem_153769_cached_sizze_155952 = 0;
    unsigned char *mem_153769 = NULL;
    int64_t mem_153780_cached_sizze_155953 = 0;
    unsigned char *mem_153780 = NULL;
    int64_t mem_153785_cached_sizze_155954 = 0;
    unsigned char *mem_153785 = NULL;
    int64_t mem_153796_cached_sizze_155955 = 0;
    unsigned char *mem_153796 = NULL;
    int64_t mem_153797_cached_sizze_155956 = 0;
    unsigned char *mem_153797 = NULL;
    int64_t mem_153810_cached_sizze_155957 = 0;
    unsigned char *mem_153810 = NULL;
    int64_t mem_153817_cached_sizze_155958 = 0;
    unsigned char *mem_153817 = NULL;
    int64_t mem_153822_cached_sizze_155959 = 0;
    unsigned char *mem_153822 = NULL;
    int64_t mem_153833_cached_sizze_155960 = 0;
    unsigned char *mem_153833 = NULL;
    int64_t mem_153838_cached_sizze_155961 = 0;
    unsigned char *mem_153838 = NULL;
    int64_t mem_153849_cached_sizze_155962 = 0;
    unsigned char *mem_153849 = NULL;
    int64_t mem_153854_cached_sizze_155963 = 0;
    unsigned char *mem_153854 = NULL;
    int64_t mem_153865_cached_sizze_155964 = 0;
    unsigned char *mem_153865 = NULL;
    int64_t mem_153870_cached_sizze_155965 = 0;
    unsigned char *mem_153870 = NULL;
    int64_t mem_153881_cached_sizze_155966 = 0;
    unsigned char *mem_153881 = NULL;
    int64_t mem_153886_cached_sizze_155967 = 0;
    unsigned char *mem_153886 = NULL;
    int64_t mem_153897_cached_sizze_155968 = 0;
    unsigned char *mem_153897 = NULL;
    int64_t mem_153902_cached_sizze_155969 = 0;
    unsigned char *mem_153902 = NULL;
    int64_t mem_153913_cached_sizze_155970 = 0;
    unsigned char *mem_153913 = NULL;
    int64_t mem_153914_cached_sizze_155971 = 0;
    unsigned char *mem_153914 = NULL;
    int64_t mem_153915_cached_sizze_155972 = 0;
    unsigned char *mem_153915 = NULL;
    int64_t mem_153916_cached_sizze_155973 = 0;
    unsigned char *mem_153916 = NULL;
    int64_t mem_153934_cached_sizze_155974 = 0;
    unsigned char *mem_153934 = NULL;
    int64_t mem_153939_cached_sizze_155975 = 0;
    unsigned char *mem_153939 = NULL;
    int64_t mem_153943_cached_sizze_155976 = 0;
    unsigned char *mem_153943 = NULL;
    int64_t mem_153950_cached_sizze_155977 = 0;
    unsigned char *mem_153950 = NULL;
    int64_t mem_153984_cached_sizze_155978 = 0;
    unsigned char *mem_153984 = NULL;
    int64_t mem_153990_cached_sizze_155979 = 0;
    unsigned char *mem_153990 = NULL;
    int64_t mem_153995_cached_sizze_155980 = 0;
    unsigned char *mem_153995 = NULL;
    int64_t mem_154011_cached_sizze_155981 = 0;
    unsigned char *mem_154011 = NULL;
    int64_t mem_154012_cached_sizze_155982 = 0;
    unsigned char *mem_154012 = NULL;
    int64_t mem_154021_cached_sizze_155983 = 0;
    unsigned char *mem_154021 = NULL;
    int64_t mem_154022_cached_sizze_155984 = 0;
    unsigned char *mem_154022 = NULL;
    int64_t mem_154043_cached_sizze_155985 = 0;
    unsigned char *mem_154043 = NULL;
    int64_t mem_154049_cached_sizze_155986 = 0;
    unsigned char *mem_154049 = NULL;
    int64_t mem_154054_cached_sizze_155987 = 0;
    unsigned char *mem_154054 = NULL;
    int64_t mem_154070_cached_sizze_155988 = 0;
    unsigned char *mem_154070 = NULL;
    int64_t mem_154075_cached_sizze_155989 = 0;
    unsigned char *mem_154075 = NULL;
    int64_t mem_154086_cached_sizze_155990 = 0;
    unsigned char *mem_154086 = NULL;
    int64_t mem_154091_cached_sizze_155991 = 0;
    unsigned char *mem_154091 = NULL;
    int64_t mem_154102_cached_sizze_155992 = 0;
    unsigned char *mem_154102 = NULL;
    int64_t mem_154107_cached_sizze_155993 = 0;
    unsigned char *mem_154107 = NULL;
    int64_t mem_154118_cached_sizze_155994 = 0;
    unsigned char *mem_154118 = NULL;
    int64_t mem_154119_cached_sizze_155995 = 0;
    unsigned char *mem_154119 = NULL;
    int64_t mem_154128_cached_sizze_155996 = 0;
    unsigned char *mem_154128 = NULL;
    int64_t mem_154129_cached_sizze_155997 = 0;
    unsigned char *mem_154129 = NULL;
    int64_t mem_154150_cached_sizze_155998 = 0;
    unsigned char *mem_154150 = NULL;
    int64_t mem_154155_cached_sizze_155999 = 0;
    unsigned char *mem_154155 = NULL;
    int64_t mem_154166_cached_sizze_156000 = 0;
    unsigned char *mem_154166 = NULL;
    int64_t mem_154167_cached_sizze_156001 = 0;
    unsigned char *mem_154167 = NULL;
    int64_t mem_154180_cached_sizze_156002 = 0;
    unsigned char *mem_154180 = NULL;
    int64_t mem_154187_cached_sizze_156003 = 0;
    unsigned char *mem_154187 = NULL;
    int64_t mem_154192_cached_sizze_156004 = 0;
    unsigned char *mem_154192 = NULL;
    int64_t mem_154203_cached_sizze_156005 = 0;
    unsigned char *mem_154203 = NULL;
    int64_t mem_154209_cached_sizze_156006 = 0;
    unsigned char *mem_154209 = NULL;
    int64_t mem_154214_cached_sizze_156007 = 0;
    unsigned char *mem_154214 = NULL;
    int64_t mem_154230_cached_sizze_156008 = 0;
    unsigned char *mem_154230 = NULL;
    int64_t mem_154231_cached_sizze_156009 = 0;
    unsigned char *mem_154231 = NULL;
    int64_t mem_154232_cached_sizze_156010 = 0;
    unsigned char *mem_154232 = NULL;
    int64_t mem_154248_cached_sizze_156011 = 0;
    unsigned char *mem_154248 = NULL;
    int64_t mem_154249_cached_sizze_156012 = 0;
    unsigned char *mem_154249 = NULL;
    int64_t mem_154250_cached_sizze_156013 = 0;
    unsigned char *mem_154250 = NULL;
    int64_t mem_154263_cached_sizze_156014 = 0;
    unsigned char *mem_154263 = NULL;
    int64_t mem_154264_cached_sizze_156015 = 0;
    unsigned char *mem_154264 = NULL;
    int64_t mem_154305_cached_sizze_156016 = 0;
    unsigned char *mem_154305 = NULL;
    int64_t mem_154306_cached_sizze_156017 = 0;
    unsigned char *mem_154306 = NULL;
    int64_t mem_154317_cached_sizze_156018 = 0;
    unsigned char *mem_154317 = NULL;
    int64_t mem_154318_cached_sizze_156019 = 0;
    unsigned char *mem_154318 = NULL;
    int64_t mem_154327_cached_sizze_156020 = 0;
    unsigned char *mem_154327 = NULL;
    int64_t mem_154328_cached_sizze_156021 = 0;
    unsigned char *mem_154328 = NULL;
    int64_t mem_154359_cached_sizze_156022 = 0;
    unsigned char *mem_154359 = NULL;
    int64_t mem_154360_cached_sizze_156023 = 0;
    unsigned char *mem_154360 = NULL;
    int64_t mem_154371_cached_sizze_156024 = 0;
    unsigned char *mem_154371 = NULL;
    int64_t mem_154372_cached_sizze_156025 = 0;
    unsigned char *mem_154372 = NULL;
    int64_t mem_154381_cached_sizze_156026 = 0;
    unsigned char *mem_154381 = NULL;
    int64_t mem_154382_cached_sizze_156027 = 0;
    unsigned char *mem_154382 = NULL;
    int64_t mem_154413_cached_sizze_156028 = 0;
    unsigned char *mem_154413 = NULL;
    int64_t mem_154414_cached_sizze_156029 = 0;
    unsigned char *mem_154414 = NULL;
    int64_t mem_154415_cached_sizze_156030 = 0;
    unsigned char *mem_154415 = NULL;
    int64_t mem_154416_cached_sizze_156031 = 0;
    unsigned char *mem_154416 = NULL;
    int64_t mem_154433_cached_sizze_156032 = 0;
    unsigned char *mem_154433 = NULL;
    int64_t mem_154434_cached_sizze_156033 = 0;
    unsigned char *mem_154434 = NULL;
    int64_t mem_154435_cached_sizze_156034 = 0;
    unsigned char *mem_154435 = NULL;
    int64_t mem_154436_cached_sizze_156035 = 0;
    unsigned char *mem_154436 = NULL;
    int64_t mem_154477_cached_sizze_156036 = 0;
    unsigned char *mem_154477 = NULL;
    int64_t mem_154478_cached_sizze_156037 = 0;
    unsigned char *mem_154478 = NULL;
    int64_t mem_154489_cached_sizze_156038 = 0;
    unsigned char *mem_154489 = NULL;
    int64_t mem_154490_cached_sizze_156039 = 0;
    unsigned char *mem_154490 = NULL;
    int64_t mem_154499_cached_sizze_156040 = 0;
    unsigned char *mem_154499 = NULL;
    int64_t mem_154500_cached_sizze_156041 = 0;
    unsigned char *mem_154500 = NULL;
    int64_t mem_154531_cached_sizze_156042 = 0;
    unsigned char *mem_154531 = NULL;
    int64_t mem_154532_cached_sizze_156043 = 0;
    unsigned char *mem_154532 = NULL;
    int64_t mem_154541_cached_sizze_156044 = 0;
    unsigned char *mem_154541 = NULL;
    int64_t mem_154542_cached_sizze_156045 = 0;
    unsigned char *mem_154542 = NULL;
    int64_t mem_154563_cached_sizze_156046 = 0;
    unsigned char *mem_154563 = NULL;
    int64_t mem_154564_cached_sizze_156047 = 0;
    unsigned char *mem_154564 = NULL;
    int64_t mem_154575_cached_sizze_156048 = 0;
    unsigned char *mem_154575 = NULL;
    int64_t mem_154576_cached_sizze_156049 = 0;
    unsigned char *mem_154576 = NULL;
    int64_t mem_154585_cached_sizze_156050 = 0;
    unsigned char *mem_154585 = NULL;
    int64_t mem_154586_cached_sizze_156051 = 0;
    unsigned char *mem_154586 = NULL;
    int64_t mem_154617_cached_sizze_156052 = 0;
    unsigned char *mem_154617 = NULL;
    int64_t mem_154618_cached_sizze_156053 = 0;
    unsigned char *mem_154618 = NULL;
    int64_t mem_154629_cached_sizze_156054 = 0;
    unsigned char *mem_154629 = NULL;
    int64_t mem_154630_cached_sizze_156055 = 0;
    unsigned char *mem_154630 = NULL;
    int64_t mem_154639_cached_sizze_156056 = 0;
    unsigned char *mem_154639 = NULL;
    int64_t mem_154640_cached_sizze_156057 = 0;
    unsigned char *mem_154640 = NULL;
    int64_t mem_154671_cached_sizze_156058 = 0;
    unsigned char *mem_154671 = NULL;
    int64_t mem_154672_cached_sizze_156059 = 0;
    unsigned char *mem_154672 = NULL;
    int64_t mem_154673_cached_sizze_156060 = 0;
    unsigned char *mem_154673 = NULL;
    int64_t mem_154674_cached_sizze_156061 = 0;
    unsigned char *mem_154674 = NULL;
    int64_t mem_154691_cached_sizze_156062 = 0;
    unsigned char *mem_154691 = NULL;
    int64_t mem_154692_cached_sizze_156063 = 0;
    unsigned char *mem_154692 = NULL;
    int64_t mem_154693_cached_sizze_156064 = 0;
    unsigned char *mem_154693 = NULL;
    int64_t mem_154694_cached_sizze_156065 = 0;
    unsigned char *mem_154694 = NULL;
    int64_t mem_154735_cached_sizze_156066 = 0;
    unsigned char *mem_154735 = NULL;
    int64_t mem_154740_cached_sizze_156067 = 0;
    unsigned char *mem_154740 = NULL;
    int64_t mem_154751_cached_sizze_156068 = 0;
    unsigned char *mem_154751 = NULL;
    int64_t mem_154752_cached_sizze_156069 = 0;
    unsigned char *mem_154752 = NULL;
    int64_t mem_154753_cached_sizze_156070 = 0;
    unsigned char *mem_154753 = NULL;
    int64_t mem_154754_cached_sizze_156071 = 0;
    unsigned char *mem_154754 = NULL;
    int64_t mem_154755_cached_sizze_156072 = 0;
    unsigned char *mem_154755 = NULL;
    int64_t mem_154774_cached_sizze_156073 = 0;
    unsigned char *mem_154774 = NULL;
    int64_t mem_154775_cached_sizze_156074 = 0;
    unsigned char *mem_154775 = NULL;
    int64_t mem_154776_cached_sizze_156075 = 0;
    unsigned char *mem_154776 = NULL;
    int64_t mem_154813_cached_sizze_156076 = 0;
    unsigned char *mem_154813 = NULL;
    int64_t mem_154820_cached_sizze_156077 = 0;
    unsigned char *mem_154820 = NULL;
    int64_t mem_154825_cached_sizze_156078 = 0;
    unsigned char *mem_154825 = NULL;
    int64_t mem_154836_cached_sizze_156079 = 0;
    unsigned char *mem_154836 = NULL;
    int64_t mem_154837_cached_sizze_156080 = 0;
    unsigned char *mem_154837 = NULL;
    int64_t mem_154846_cached_sizze_156081 = 0;
    unsigned char *mem_154846 = NULL;
    int64_t mem_154847_cached_sizze_156082 = 0;
    unsigned char *mem_154847 = NULL;
    int64_t mem_154868_cached_sizze_156083 = 0;
    unsigned char *mem_154868 = NULL;
    int64_t mem_154869_cached_sizze_156084 = 0;
    unsigned char *mem_154869 = NULL;
    int64_t mem_154870_cached_sizze_156085 = 0;
    unsigned char *mem_154870 = NULL;
    int64_t mem_154871_cached_sizze_156086 = 0;
    unsigned char *mem_154871 = NULL;
    int64_t mem_154896_cached_sizze_156087 = 0;
    unsigned char *mem_154896 = NULL;
    int64_t mem_154897_cached_sizze_156088 = 0;
    unsigned char *mem_154897 = NULL;
    int64_t mem_154910_cached_sizze_156089 = 0;
    unsigned char *mem_154910 = NULL;
    int64_t mem_154911_cached_sizze_156090 = 0;
    unsigned char *mem_154911 = NULL;
    int64_t mem_154920_cached_sizze_156091 = 0;
    unsigned char *mem_154920 = NULL;
    int64_t mem_154921_cached_sizze_156092 = 0;
    unsigned char *mem_154921 = NULL;
    int64_t mem_154942_cached_sizze_156093 = 0;
    unsigned char *mem_154942 = NULL;
    int64_t mem_154947_cached_sizze_156094 = 0;
    unsigned char *mem_154947 = NULL;
    int64_t mem_154958_cached_sizze_156095 = 0;
    unsigned char *mem_154958 = NULL;
    int64_t mem_154959_cached_sizze_156096 = 0;
    unsigned char *mem_154959 = NULL;
    int64_t mem_154968_cached_sizze_156097 = 0;
    unsigned char *mem_154968 = NULL;
    int64_t mem_154969_cached_sizze_156098 = 0;
    unsigned char *mem_154969 = NULL;
    struct memblock mem_param_tmp_155325;
    
    mem_param_tmp_155325.references = NULL;
    
    struct memblock mem_param_tmp_155324;
    
    mem_param_tmp_155324.references = NULL;
    
    struct memblock mem_param_tmp_155323;
    
    mem_param_tmp_155323.references = NULL;
    
    struct memblock mem_param_tmp_155322;
    
    mem_param_tmp_155322.references = NULL;
    
    struct memblock mem_param_tmp_155321;
    
    mem_param_tmp_155321.references = NULL;
    
    struct memblock mem_param_tmp_155320;
    
    mem_param_tmp_155320.references = NULL;
    
    struct memblock mem_param_tmp_155319;
    
    mem_param_tmp_155319.references = NULL;
    
    struct memblock mem_param_tmp_155318;
    
    mem_param_tmp_155318.references = NULL;
    
    struct memblock mem_param_tmp_155317;
    
    mem_param_tmp_155317.references = NULL;
    
    struct memblock mem_param_tmp_155316;
    
    mem_param_tmp_155316.references = NULL;
    
    struct memblock mem_param_tmp_155315;
    
    mem_param_tmp_155315.references = NULL;
    
    struct memblock mem_param_tmp_155314;
    
    mem_param_tmp_155314.references = NULL;
    
    struct memblock mem_param_tmp_155313;
    
    mem_param_tmp_155313.references = NULL;
    
    struct memblock mem_param_tmp_155312;
    
    mem_param_tmp_155312.references = NULL;
    
    struct memblock mem_param_tmp_155311;
    
    mem_param_tmp_155311.references = NULL;
    
    struct memblock mem_param_tmp_155310;
    
    mem_param_tmp_155310.references = NULL;
    
    struct memblock mem_param_tmp_155309;
    
    mem_param_tmp_155309.references = NULL;
    
    struct memblock mem_param_tmp_155308;
    
    mem_param_tmp_155308.references = NULL;
    
    struct memblock mem_param_tmp_155307;
    
    mem_param_tmp_155307.references = NULL;
    
    struct memblock mem_param_tmp_155306;
    
    mem_param_tmp_155306.references = NULL;
    
    struct memblock mem_param_tmp_155305;
    
    mem_param_tmp_155305.references = NULL;
    
    struct memblock mem_param_tmp_155304;
    
    mem_param_tmp_155304.references = NULL;
    
    struct memblock mem_param_tmp_155303;
    
    mem_param_tmp_155303.references = NULL;
    
    struct memblock mem_param_tmp_155302;
    
    mem_param_tmp_155302.references = NULL;
    
    struct memblock mem_param_tmp_155301;
    
    mem_param_tmp_155301.references = NULL;
    
    struct memblock mem_param_tmp_155300;
    
    mem_param_tmp_155300.references = NULL;
    
    struct memblock mem_param_tmp_155299;
    
    mem_param_tmp_155299.references = NULL;
    
    struct memblock ext_mem_155086;
    
    ext_mem_155086.references = NULL;
    
    struct memblock ext_mem_155087;
    
    ext_mem_155087.references = NULL;
    
    struct memblock ext_mem_155088;
    
    ext_mem_155088.references = NULL;
    
    struct memblock mem_155084;
    
    mem_155084.references = NULL;
    
    struct memblock mem_155082;
    
    mem_155082.references = NULL;
    
    struct memblock mem_155080;
    
    mem_155080.references = NULL;
    
    struct memblock mem_155078;
    
    mem_155078.references = NULL;
    
    struct memblock ext_mem_155075;
    
    ext_mem_155075.references = NULL;
    
    struct memblock ext_mem_155076;
    
    ext_mem_155076.references = NULL;
    
    struct memblock ext_mem_155077;
    
    ext_mem_155077.references = NULL;
    
    struct memblock mem_155073;
    
    mem_155073.references = NULL;
    
    struct memblock mem_155071;
    
    mem_155071.references = NULL;
    
    struct memblock mem_155069;
    
    mem_155069.references = NULL;
    
    struct memblock mem_155067;
    
    mem_155067.references = NULL;
    
    struct memblock ext_mem_155064;
    
    ext_mem_155064.references = NULL;
    
    struct memblock ext_mem_155065;
    
    ext_mem_155065.references = NULL;
    
    struct memblock ext_mem_155066;
    
    ext_mem_155066.references = NULL;
    
    struct memblock mem_155062;
    
    mem_155062.references = NULL;
    
    struct memblock mem_155060;
    
    mem_155060.references = NULL;
    
    struct memblock mem_155058;
    
    mem_155058.references = NULL;
    
    struct memblock mem_155056;
    
    mem_155056.references = NULL;
    
    struct memblock ext_mem_155053;
    
    ext_mem_155053.references = NULL;
    
    struct memblock ext_mem_155054;
    
    ext_mem_155054.references = NULL;
    
    struct memblock ext_mem_155055;
    
    ext_mem_155055.references = NULL;
    
    struct memblock mem_155051;
    
    mem_155051.references = NULL;
    
    struct memblock mem_155049;
    
    mem_155049.references = NULL;
    
    struct memblock mem_155047;
    
    mem_155047.references = NULL;
    
    struct memblock mem_155045;
    
    mem_155045.references = NULL;
    
    struct memblock ext_mem_155042;
    
    ext_mem_155042.references = NULL;
    
    struct memblock ext_mem_155043;
    
    ext_mem_155043.references = NULL;
    
    struct memblock ext_mem_155044;
    
    ext_mem_155044.references = NULL;
    
    struct memblock mem_155040;
    
    mem_155040.references = NULL;
    
    struct memblock mem_155038;
    
    mem_155038.references = NULL;
    
    struct memblock mem_155036;
    
    mem_155036.references = NULL;
    
    struct memblock mem_155034;
    
    mem_155034.references = NULL;
    
    struct memblock ext_mem_155031;
    
    ext_mem_155031.references = NULL;
    
    struct memblock ext_mem_155032;
    
    ext_mem_155032.references = NULL;
    
    struct memblock ext_mem_155033;
    
    ext_mem_155033.references = NULL;
    
    struct memblock mem_155029;
    
    mem_155029.references = NULL;
    
    struct memblock mem_155027;
    
    mem_155027.references = NULL;
    
    struct memblock mem_155025;
    
    mem_155025.references = NULL;
    
    struct memblock mem_155023;
    
    mem_155023.references = NULL;
    
    struct memblock ext_mem_155020;
    
    ext_mem_155020.references = NULL;
    
    struct memblock ext_mem_155021;
    
    ext_mem_155021.references = NULL;
    
    struct memblock ext_mem_155022;
    
    ext_mem_155022.references = NULL;
    
    struct memblock mem_155018;
    
    mem_155018.references = NULL;
    
    struct memblock mem_155016;
    
    mem_155016.references = NULL;
    
    struct memblock mem_155014;
    
    mem_155014.references = NULL;
    
    struct memblock mem_155012;
    
    mem_155012.references = NULL;
    
    struct memblock ext_mem_155009;
    
    ext_mem_155009.references = NULL;
    
    struct memblock ext_mem_155010;
    
    ext_mem_155010.references = NULL;
    
    struct memblock ext_mem_155011;
    
    ext_mem_155011.references = NULL;
    
    struct memblock mem_155007;
    
    mem_155007.references = NULL;
    
    struct memblock mem_155005;
    
    mem_155005.references = NULL;
    
    struct memblock mem_155003;
    
    mem_155003.references = NULL;
    
    struct memblock mem_155001;
    
    mem_155001.references = NULL;
    
    struct memblock ext_mem_154998;
    
    ext_mem_154998.references = NULL;
    
    struct memblock ext_mem_154999;
    
    ext_mem_154999.references = NULL;
    
    struct memblock ext_mem_155000;
    
    ext_mem_155000.references = NULL;
    
    struct memblock mem_154996;
    
    mem_154996.references = NULL;
    
    struct memblock mem_154994;
    
    mem_154994.references = NULL;
    
    struct memblock mem_154992;
    
    mem_154992.references = NULL;
    
    struct memblock mem_154990;
    
    mem_154990.references = NULL;
    
    struct memblock mem_param_152918;
    
    mem_param_152918.references = NULL;
    
    struct memblock mem_param_152914;
    
    mem_param_152914.references = NULL;
    
    struct memblock mem_param_152910;
    
    mem_param_152910.references = NULL;
    
    struct memblock mem_param_152906;
    
    mem_param_152906.references = NULL;
    
    struct memblock mem_param_152902;
    
    mem_param_152902.references = NULL;
    
    struct memblock mem_param_152898;
    
    mem_param_152898.references = NULL;
    
    struct memblock mem_param_152894;
    
    mem_param_152894.references = NULL;
    
    struct memblock mem_param_152890;
    
    mem_param_152890.references = NULL;
    
    struct memblock mem_param_152886;
    
    mem_param_152886.references = NULL;
    
    struct memblock mem_param_152882;
    
    mem_param_152882.references = NULL;
    
    struct memblock mem_param_152878;
    
    mem_param_152878.references = NULL;
    
    struct memblock mem_param_152874;
    
    mem_param_152874.references = NULL;
    
    struct memblock mem_param_152870;
    
    mem_param_152870.references = NULL;
    
    struct memblock mem_param_152866;
    
    mem_param_152866.references = NULL;
    
    struct memblock mem_param_152862;
    
    mem_param_152862.references = NULL;
    
    struct memblock mem_param_152858;
    
    mem_param_152858.references = NULL;
    
    struct memblock mem_param_152854;
    
    mem_param_152854.references = NULL;
    
    struct memblock mem_param_152850;
    
    mem_param_152850.references = NULL;
    
    struct memblock mem_param_152846;
    
    mem_param_152846.references = NULL;
    
    struct memblock mem_param_152842;
    
    mem_param_152842.references = NULL;
    
    struct memblock mem_param_152838;
    
    mem_param_152838.references = NULL;
    
    struct memblock mem_param_152834;
    
    mem_param_152834.references = NULL;
    
    struct memblock mem_param_152830;
    
    mem_param_152830.references = NULL;
    
    struct memblock mem_param_152826;
    
    mem_param_152826.references = NULL;
    
    struct memblock mem_param_152822;
    
    mem_param_152822.references = NULL;
    
    struct memblock mem_param_152818;
    
    mem_param_152818.references = NULL;
    
    struct memblock mem_param_152814;
    
    mem_param_152814.references = NULL;
    
    struct memblock ext_mem_155170;
    
    ext_mem_155170.references = NULL;
    
    struct memblock ext_mem_155171;
    
    ext_mem_155171.references = NULL;
    
    struct memblock ext_mem_155172;
    
    ext_mem_155172.references = NULL;
    
    struct memblock ext_mem_155173;
    
    ext_mem_155173.references = NULL;
    
    struct memblock ext_mem_155174;
    
    ext_mem_155174.references = NULL;
    
    struct memblock ext_mem_155175;
    
    ext_mem_155175.references = NULL;
    
    struct memblock ext_mem_155176;
    
    ext_mem_155176.references = NULL;
    
    struct memblock ext_mem_155177;
    
    ext_mem_155177.references = NULL;
    
    struct memblock ext_mem_155178;
    
    ext_mem_155178.references = NULL;
    
    struct memblock ext_mem_155179;
    
    ext_mem_155179.references = NULL;
    
    struct memblock ext_mem_155180;
    
    ext_mem_155180.references = NULL;
    
    struct memblock ext_mem_155181;
    
    ext_mem_155181.references = NULL;
    
    struct memblock ext_mem_155182;
    
    ext_mem_155182.references = NULL;
    
    struct memblock ext_mem_155183;
    
    ext_mem_155183.references = NULL;
    
    struct memblock ext_mem_155184;
    
    ext_mem_155184.references = NULL;
    
    struct memblock ext_mem_155185;
    
    ext_mem_155185.references = NULL;
    
    struct memblock ext_mem_155186;
    
    ext_mem_155186.references = NULL;
    
    struct memblock ext_mem_155187;
    
    ext_mem_155187.references = NULL;
    
    struct memblock ext_mem_155188;
    
    ext_mem_155188.references = NULL;
    
    struct memblock ext_mem_155189;
    
    ext_mem_155189.references = NULL;
    
    struct memblock ext_mem_155190;
    
    ext_mem_155190.references = NULL;
    
    struct memblock ext_mem_155191;
    
    ext_mem_155191.references = NULL;
    
    struct memblock ext_mem_155192;
    
    ext_mem_155192.references = NULL;
    
    struct memblock ext_mem_155193;
    
    ext_mem_155193.references = NULL;
    
    struct memblock ext_mem_155194;
    
    ext_mem_155194.references = NULL;
    
    struct memblock ext_mem_155195;
    
    ext_mem_155195.references = NULL;
    
    struct memblock ext_mem_155196;
    
    ext_mem_155196.references = NULL;
    
    struct memblock mem_out_155295;
    
    mem_out_155295.references = NULL;
    
    struct memblock mem_out_155294;
    
    mem_out_155294.references = NULL;
    
    struct memblock mem_out_155293;
    
    mem_out_155293.references = NULL;
    
    struct memblock mem_out_155292;
    
    mem_out_155292.references = NULL;
    
    struct memblock mem_out_155291;
    
    mem_out_155291.references = NULL;
    
    struct memblock mem_out_155290;
    
    mem_out_155290.references = NULL;
    
    struct memblock mem_out_155289;
    
    mem_out_155289.references = NULL;
    
    struct memblock mem_out_155288;
    
    mem_out_155288.references = NULL;
    
    struct memblock mem_out_155287;
    
    mem_out_155287.references = NULL;
    
    struct memblock mem_out_155286;
    
    mem_out_155286.references = NULL;
    
    struct memblock mem_out_155285;
    
    mem_out_155285.references = NULL;
    
    struct memblock mem_out_155284;
    
    mem_out_155284.references = NULL;
    
    struct memblock mem_out_155283;
    
    mem_out_155283.references = NULL;
    
    struct memblock mem_out_155282;
    
    mem_out_155282.references = NULL;
    
    struct memblock mem_out_155281;
    
    mem_out_155281.references = NULL;
    
    struct memblock mem_out_155280;
    
    mem_out_155280.references = NULL;
    
    struct memblock mem_out_155279;
    
    mem_out_155279.references = NULL;
    
    struct memblock mem_out_155278;
    
    mem_out_155278.references = NULL;
    
    struct memblock mem_out_155277;
    
    mem_out_155277.references = NULL;
    
    struct memblock mem_out_155276;
    
    mem_out_155276.references = NULL;
    
    struct memblock mem_out_155275;
    
    mem_out_155275.references = NULL;
    
    struct memblock mem_out_155274;
    
    mem_out_155274.references = NULL;
    
    struct memblock mem_out_155273;
    
    mem_out_155273.references = NULL;
    
    struct memblock mem_out_155272;
    
    mem_out_155272.references = NULL;
    
    struct memblock mem_out_155271;
    
    mem_out_155271.references = NULL;
    
    struct memblock mem_out_155270;
    
    mem_out_155270.references = NULL;
    
    struct memblock mem_out_155269;
    
    mem_out_155269.references = NULL;
    
    struct memblock mem_152772 = ctx->constants->mem_152772;
    struct memblock mem_152773 = ctx->constants->mem_152773;
    struct memblock mem_152774 = ctx->constants->mem_152774;
    struct memblock mem_152775 = ctx->constants->mem_152775;
    struct memblock mem_152776 = ctx->constants->mem_152776;
    struct memblock mem_152777 = ctx->constants->mem_152777;
    struct memblock mem_152778 = ctx->constants->mem_152778;
    struct memblock mem_152779 = ctx->constants->mem_152779;
    struct memblock mem_152780 = ctx->constants->mem_152780;
    
    // futhark/microgpt.fut:61:13-49
    
    double defunc_0_lifted_lambda_res_138894;
    double r_138896 = 0.0;
    
    for (int64_t i_138895 = 0; i_138895 < (int64_t) 27; i_138895++) {
        // futhark/microgpt.fut:61:40-49
        
        double zp_res_138897 = 1.0 + r_138896;
        double r_tmp_155296 = zp_res_138897;
        
        r_138896 = r_tmp_155296;
    }
    defunc_0_lifted_lambda_res_138894 = r_138896;
    // futhark/microgpt.fut:61:13-49
    
    double defunc_0_lifted_lambda_res_139667;
    double r_139669 = 0.0;
    
    for (int64_t i_139668 = 0; i_139668 < (int64_t) 16; i_139668++) {
        // futhark/microgpt.fut:61:40-49
        
        double zp_res_139670 = 1.0 + r_139669;
        double r_tmp_155297 = zp_res_139670;
        
        r_139669 = r_tmp_155297;
    }
    defunc_0_lifted_lambda_res_139667 = r_139669;
    // futhark/microgpt.fut:61:13-49
    
    double defunc_0_lifted_lambda_res_140096;
    double r_140098 = 0.0;
    
    for (int64_t i_140097 = 0; i_140097 < (int64_t) 16; i_140097++) {
        // futhark/microgpt.fut:61:40-49
        
        double zp_res_140099 = 1.0 + r_140098;
        double r_tmp_155298 = zp_res_140099;
        
        r_140098 = r_tmp_155298;
    }
    defunc_0_lifted_lambda_res_140096 = r_140098;
    // futhark/microgpt.fut:56:26-45
    
    double i64_res_140710 = sitofp_i64_f64(num_steps_112428);
    
    // futhark/microgpt.fut:4:11-25
    if (mem_152919_cached_sizze_155852 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152919, &mem_152919_cached_sizze_155852, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152920_cached_sizze_155853 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_152920, &mem_152920_cached_sizze_155853, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152929_cached_sizze_155854 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_152929, &mem_152929_cached_sizze_155854, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152936_cached_sizze_155855 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152936, &mem_152936_cached_sizze_155855, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152951_cached_sizze_155856 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152951, &mem_152951_cached_sizze_155856, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152952_cached_sizze_155857 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152952, &mem_152952_cached_sizze_155857, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152953_cached_sizze_155858 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152953, &mem_152953_cached_sizze_155858, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152972_cached_sizze_155859 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152972, &mem_152972_cached_sizze_155859, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152979_cached_sizze_155860 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152979, &mem_152979_cached_sizze_155860, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152984_cached_sizze_155861 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152984, &mem_152984_cached_sizze_155861, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152995_cached_sizze_155862 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152995, &mem_152995_cached_sizze_155862, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153000_cached_sizze_155863 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153000, &mem_153000_cached_sizze_155863, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153011_cached_sizze_155864 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153011, &mem_153011_cached_sizze_155864, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153012_cached_sizze_155865 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153012, &mem_153012_cached_sizze_155865, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153025_cached_sizze_155866 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153025, &mem_153025_cached_sizze_155866, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153032_cached_sizze_155867 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153032, &mem_153032_cached_sizze_155867, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153037_cached_sizze_155868 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153037, &mem_153037_cached_sizze_155868, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153048_cached_sizze_155869 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153048, &mem_153048_cached_sizze_155869, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153053_cached_sizze_155870 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153053, &mem_153053_cached_sizze_155870, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153064_cached_sizze_155871 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153064, &mem_153064_cached_sizze_155871, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153065_cached_sizze_155872 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153065, &mem_153065_cached_sizze_155872, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153066_cached_sizze_155873 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153066, &mem_153066_cached_sizze_155873, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153082_cached_sizze_155874 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153082, &mem_153082_cached_sizze_155874, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153083_cached_sizze_155875 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153083, &mem_153083_cached_sizze_155875, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153084_cached_sizze_155876 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153084, &mem_153084_cached_sizze_155876, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153097_cached_sizze_155877 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_153097, &mem_153097_cached_sizze_155877, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153098_cached_sizze_155878 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_153098, &mem_153098_cached_sizze_155878, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153099_cached_sizze_155879 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_153099, &mem_153099_cached_sizze_155879, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153145_cached_sizze_155880 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153145, &mem_153145_cached_sizze_155880, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153146_cached_sizze_155881 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153146, &mem_153146_cached_sizze_155881, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153147_cached_sizze_155882 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153147, &mem_153147_cached_sizze_155882, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153148_cached_sizze_155883 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153148, &mem_153148_cached_sizze_155883, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153169_cached_sizze_155884 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153169, &mem_153169_cached_sizze_155884, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153170_cached_sizze_155885 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153170, &mem_153170_cached_sizze_155885, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153171_cached_sizze_155886 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153171, &mem_153171_cached_sizze_155886, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153172_cached_sizze_155887 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153172, &mem_153172_cached_sizze_155887, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153189_cached_sizze_155888 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153189, &mem_153189_cached_sizze_155888, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153190_cached_sizze_155889 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153190, &mem_153190_cached_sizze_155889, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153191_cached_sizze_155890 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153191, &mem_153191_cached_sizze_155890, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153192_cached_sizze_155891 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153192, &mem_153192_cached_sizze_155891, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153253_cached_sizze_155892 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153253, &mem_153253_cached_sizze_155892, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153254_cached_sizze_155893 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153254, &mem_153254_cached_sizze_155893, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153255_cached_sizze_155894 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153255, &mem_153255_cached_sizze_155894, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153256_cached_sizze_155895 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153256, &mem_153256_cached_sizze_155895, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153277_cached_sizze_155896 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153277, &mem_153277_cached_sizze_155896, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153278_cached_sizze_155897 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153278, &mem_153278_cached_sizze_155897, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153279_cached_sizze_155898 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153279, &mem_153279_cached_sizze_155898, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153280_cached_sizze_155899 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153280, &mem_153280_cached_sizze_155899, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153297_cached_sizze_155900 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153297, &mem_153297_cached_sizze_155900, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153298_cached_sizze_155901 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153298, &mem_153298_cached_sizze_155901, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153299_cached_sizze_155902 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153299, &mem_153299_cached_sizze_155902, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153300_cached_sizze_155903 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153300, &mem_153300_cached_sizze_155903, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153361_cached_sizze_155904 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153361, &mem_153361_cached_sizze_155904, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153362_cached_sizze_155905 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153362, &mem_153362_cached_sizze_155905, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153363_cached_sizze_155906 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153363, &mem_153363_cached_sizze_155906, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153364_cached_sizze_155907 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153364, &mem_153364_cached_sizze_155907, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153365_cached_sizze_155908 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153365, &mem_153365_cached_sizze_155908, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153366_cached_sizze_155909 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153366, &mem_153366_cached_sizze_155909, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153367_cached_sizze_155910 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153367, &mem_153367_cached_sizze_155910, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153368_cached_sizze_155911 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153368, &mem_153368_cached_sizze_155911, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153401_cached_sizze_155912 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153401, &mem_153401_cached_sizze_155912, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153402_cached_sizze_155913 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153402, &mem_153402_cached_sizze_155913, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153403_cached_sizze_155914 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153403, &mem_153403_cached_sizze_155914, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153404_cached_sizze_155915 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153404, &mem_153404_cached_sizze_155915, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153405_cached_sizze_155916 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153405, &mem_153405_cached_sizze_155916, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153406_cached_sizze_155917 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153406, &mem_153406_cached_sizze_155917, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153407_cached_sizze_155918 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153407, &mem_153407_cached_sizze_155918, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153408_cached_sizze_155919 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153408, &mem_153408_cached_sizze_155919, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153489_cached_sizze_155920 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153489, &mem_153489_cached_sizze_155920, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153490_cached_sizze_155921 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153490, &mem_153490_cached_sizze_155921, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153491_cached_sizze_155922 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153491, &mem_153491_cached_sizze_155922, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153492_cached_sizze_155923 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153492, &mem_153492_cached_sizze_155923, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153513_cached_sizze_155924 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153513, &mem_153513_cached_sizze_155924, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153514_cached_sizze_155925 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153514, &mem_153514_cached_sizze_155925, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153515_cached_sizze_155926 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153515, &mem_153515_cached_sizze_155926, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153516_cached_sizze_155927 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153516, &mem_153516_cached_sizze_155927, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153533_cached_sizze_155928 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153533, &mem_153533_cached_sizze_155928, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153534_cached_sizze_155929 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153534, &mem_153534_cached_sizze_155929, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153535_cached_sizze_155930 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153535, &mem_153535_cached_sizze_155930, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153536_cached_sizze_155931 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153536, &mem_153536_cached_sizze_155931, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153597_cached_sizze_155932 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153597, &mem_153597_cached_sizze_155932, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153598_cached_sizze_155933 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153598, &mem_153598_cached_sizze_155933, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153607_cached_sizze_155934 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153607, &mem_153607_cached_sizze_155934, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153608_cached_sizze_155935 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153608, &mem_153608_cached_sizze_155935, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153629_cached_sizze_155936 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153629, &mem_153629_cached_sizze_155936, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153630_cached_sizze_155937 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153630, &mem_153630_cached_sizze_155937, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153641_cached_sizze_155938 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153641, &mem_153641_cached_sizze_155938, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153642_cached_sizze_155939 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153642, &mem_153642_cached_sizze_155939, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153651_cached_sizze_155940 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153651, &mem_153651_cached_sizze_155940, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153652_cached_sizze_155941 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153652, &mem_153652_cached_sizze_155941, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153683_cached_sizze_155942 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153683, &mem_153683_cached_sizze_155942, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153684_cached_sizze_155943 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153684, &mem_153684_cached_sizze_155943, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153695_cached_sizze_155944 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153695, &mem_153695_cached_sizze_155944, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153696_cached_sizze_155945 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153696, &mem_153696_cached_sizze_155945, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153705_cached_sizze_155946 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153705, &mem_153705_cached_sizze_155946, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153706_cached_sizze_155947 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153706, &mem_153706_cached_sizze_155947, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153737_cached_sizze_155948 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153737, &mem_153737_cached_sizze_155948, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153743_cached_sizze_155949 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153743, &mem_153743_cached_sizze_155949, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153748_cached_sizze_155950 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_153748, &mem_153748_cached_sizze_155950, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153764_cached_sizze_155951 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153764, &mem_153764_cached_sizze_155951, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153769_cached_sizze_155952 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153769, &mem_153769_cached_sizze_155952, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153780_cached_sizze_155953 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153780, &mem_153780_cached_sizze_155953, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153785_cached_sizze_155954 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153785, &mem_153785_cached_sizze_155954, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153796_cached_sizze_155955 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153796, &mem_153796_cached_sizze_155955, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153797_cached_sizze_155956 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153797, &mem_153797_cached_sizze_155956, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153810_cached_sizze_155957 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153810, &mem_153810_cached_sizze_155957, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153817_cached_sizze_155958 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153817, &mem_153817_cached_sizze_155958, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153822_cached_sizze_155959 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153822, &mem_153822_cached_sizze_155959, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153833_cached_sizze_155960 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153833, &mem_153833_cached_sizze_155960, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153838_cached_sizze_155961 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153838, &mem_153838_cached_sizze_155961, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153849_cached_sizze_155962 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153849, &mem_153849_cached_sizze_155962, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153854_cached_sizze_155963 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153854, &mem_153854_cached_sizze_155963, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153865_cached_sizze_155964 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153865, &mem_153865_cached_sizze_155964, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153870_cached_sizze_155965 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153870, &mem_153870_cached_sizze_155965, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153881_cached_sizze_155966 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153881, &mem_153881_cached_sizze_155966, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153886_cached_sizze_155967 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153886, &mem_153886_cached_sizze_155967, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153897_cached_sizze_155968 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_153897, &mem_153897_cached_sizze_155968, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153902_cached_sizze_155969 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_153902, &mem_153902_cached_sizze_155969, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153913_cached_sizze_155970 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_153913, &mem_153913_cached_sizze_155970, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153914_cached_sizze_155971 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_153914, &mem_153914_cached_sizze_155971, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153915_cached_sizze_155972 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_153915, &mem_153915_cached_sizze_155972, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153916_cached_sizze_155973 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_153916, &mem_153916_cached_sizze_155973, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:105:13-33
    if (mem_153934_cached_sizze_155974 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_153934, &mem_153934_cached_sizze_155974, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153939_cached_sizze_155975 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_153939, &mem_153939_cached_sizze_155975, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153984_cached_sizze_155978 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_153984, &mem_153984_cached_sizze_155978, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153990_cached_sizze_155979 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_153990, &mem_153990_cached_sizze_155979, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153995_cached_sizze_155980 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_153995, &mem_153995_cached_sizze_155980, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154011_cached_sizze_155981 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_154011, &mem_154011_cached_sizze_155981, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154012_cached_sizze_155982 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_154012, &mem_154012_cached_sizze_155982, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154021_cached_sizze_155983 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_154021, &mem_154021_cached_sizze_155983, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154022_cached_sizze_155984 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_154022, &mem_154022_cached_sizze_155984, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154043_cached_sizze_155985 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_154043, &mem_154043_cached_sizze_155985, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154049_cached_sizze_155986 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_154049, &mem_154049_cached_sizze_155986, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154054_cached_sizze_155987 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_154054, &mem_154054_cached_sizze_155987, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154070_cached_sizze_155988 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_154070, &mem_154070_cached_sizze_155988, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154075_cached_sizze_155989 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_154075, &mem_154075_cached_sizze_155989, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154086_cached_sizze_155990 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_154086, &mem_154086_cached_sizze_155990, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154091_cached_sizze_155991 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_154091, &mem_154091_cached_sizze_155991, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154102_cached_sizze_155992 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154102, &mem_154102_cached_sizze_155992, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154107_cached_sizze_155993 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154107, &mem_154107_cached_sizze_155993, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154118_cached_sizze_155994 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_154118, &mem_154118_cached_sizze_155994, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154119_cached_sizze_155995 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_154119, &mem_154119_cached_sizze_155995, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154128_cached_sizze_155996 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_154128, &mem_154128_cached_sizze_155996, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154129_cached_sizze_155997 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_154129, &mem_154129_cached_sizze_155997, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154150_cached_sizze_155998 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154150, &mem_154150_cached_sizze_155998, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154155_cached_sizze_155999 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154155, &mem_154155_cached_sizze_155999, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154166_cached_sizze_156000 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154166, &mem_154166_cached_sizze_156000, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154167_cached_sizze_156001 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154167, &mem_154167_cached_sizze_156001, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154180_cached_sizze_156002 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154180, &mem_154180_cached_sizze_156002, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154187_cached_sizze_156003 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154187, &mem_154187_cached_sizze_156003, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154192_cached_sizze_156004 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154192, &mem_154192_cached_sizze_156004, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154203_cached_sizze_156005 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154203, &mem_154203_cached_sizze_156005, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154209_cached_sizze_156006 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_154209, &mem_154209_cached_sizze_156006, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154214_cached_sizze_156007 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_154214, &mem_154214_cached_sizze_156007, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154230_cached_sizze_156008 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154230, &mem_154230_cached_sizze_156008, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154231_cached_sizze_156009 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154231, &mem_154231_cached_sizze_156009, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154232_cached_sizze_156010 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154232, &mem_154232_cached_sizze_156010, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154248_cached_sizze_156011 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_154248, &mem_154248_cached_sizze_156011, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154249_cached_sizze_156012 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_154249, &mem_154249_cached_sizze_156012, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154250_cached_sizze_156013 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_154250, &mem_154250_cached_sizze_156013, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154263_cached_sizze_156014 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_154263, &mem_154263_cached_sizze_156014, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154264_cached_sizze_156015 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_154264, &mem_154264_cached_sizze_156015, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154305_cached_sizze_156016 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_154305, &mem_154305_cached_sizze_156016, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154306_cached_sizze_156017 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_154306, &mem_154306_cached_sizze_156017, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154317_cached_sizze_156018 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154317, &mem_154317_cached_sizze_156018, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154318_cached_sizze_156019 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154318, &mem_154318_cached_sizze_156019, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154327_cached_sizze_156020 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154327, &mem_154327_cached_sizze_156020, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154328_cached_sizze_156021 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154328, &mem_154328_cached_sizze_156021, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154359_cached_sizze_156022 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_154359, &mem_154359_cached_sizze_156022, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154360_cached_sizze_156023 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_154360, &mem_154360_cached_sizze_156023, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154371_cached_sizze_156024 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154371, &mem_154371_cached_sizze_156024, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154372_cached_sizze_156025 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154372, &mem_154372_cached_sizze_156025, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154381_cached_sizze_156026 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154381, &mem_154381_cached_sizze_156026, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154382_cached_sizze_156027 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154382, &mem_154382_cached_sizze_156027, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154413_cached_sizze_156028 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_154413, &mem_154413_cached_sizze_156028, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154414_cached_sizze_156029 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_154414, &mem_154414_cached_sizze_156029, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154415_cached_sizze_156030 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_154415, &mem_154415_cached_sizze_156030, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154416_cached_sizze_156031 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_154416, &mem_154416_cached_sizze_156031, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154433_cached_sizze_156032 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154433, &mem_154433_cached_sizze_156032, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154434_cached_sizze_156033 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154434, &mem_154434_cached_sizze_156033, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154435_cached_sizze_156034 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154435, &mem_154435_cached_sizze_156034, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154436_cached_sizze_156035 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154436, &mem_154436_cached_sizze_156035, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154477_cached_sizze_156036 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_154477, &mem_154477_cached_sizze_156036, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154478_cached_sizze_156037 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_154478, &mem_154478_cached_sizze_156037, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154489_cached_sizze_156038 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154489, &mem_154489_cached_sizze_156038, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154490_cached_sizze_156039 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154490, &mem_154490_cached_sizze_156039, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154499_cached_sizze_156040 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154499, &mem_154499_cached_sizze_156040, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154500_cached_sizze_156041 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154500, &mem_154500_cached_sizze_156041, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154531_cached_sizze_156042 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_154531, &mem_154531_cached_sizze_156042, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154532_cached_sizze_156043 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_154532, &mem_154532_cached_sizze_156043, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154541_cached_sizze_156044 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154541, &mem_154541_cached_sizze_156044, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154542_cached_sizze_156045 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154542, &mem_154542_cached_sizze_156045, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154563_cached_sizze_156046 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_154563, &mem_154563_cached_sizze_156046, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154564_cached_sizze_156047 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_154564, &mem_154564_cached_sizze_156047, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154575_cached_sizze_156048 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154575, &mem_154575_cached_sizze_156048, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154576_cached_sizze_156049 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154576, &mem_154576_cached_sizze_156049, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154585_cached_sizze_156050 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154585, &mem_154585_cached_sizze_156050, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154586_cached_sizze_156051 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154586, &mem_154586_cached_sizze_156051, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154617_cached_sizze_156052 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_154617, &mem_154617_cached_sizze_156052, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154618_cached_sizze_156053 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_154618, &mem_154618_cached_sizze_156053, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154629_cached_sizze_156054 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154629, &mem_154629_cached_sizze_156054, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154630_cached_sizze_156055 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154630, &mem_154630_cached_sizze_156055, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154639_cached_sizze_156056 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154639, &mem_154639_cached_sizze_156056, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154640_cached_sizze_156057 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154640, &mem_154640_cached_sizze_156057, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154671_cached_sizze_156058 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154671, &mem_154671_cached_sizze_156058, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154672_cached_sizze_156059 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154672, &mem_154672_cached_sizze_156059, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154673_cached_sizze_156060 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154673, &mem_154673_cached_sizze_156060, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154674_cached_sizze_156061 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154674, &mem_154674_cached_sizze_156061, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154691_cached_sizze_156062 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154691, &mem_154691_cached_sizze_156062, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154692_cached_sizze_156063 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154692, &mem_154692_cached_sizze_156063, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154693_cached_sizze_156064 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154693, &mem_154693_cached_sizze_156064, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154694_cached_sizze_156065 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154694, &mem_154694_cached_sizze_156065, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154735_cached_sizze_156066 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154735, &mem_154735_cached_sizze_156066, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154740_cached_sizze_156067 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154740, &mem_154740_cached_sizze_156067, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154751_cached_sizze_156068 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154751, &mem_154751_cached_sizze_156068, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154752_cached_sizze_156069 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154752, &mem_154752_cached_sizze_156069, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154753_cached_sizze_156070 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154753, &mem_154753_cached_sizze_156070, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154754_cached_sizze_156071 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154754, &mem_154754_cached_sizze_156071, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154755_cached_sizze_156072 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154755, &mem_154755_cached_sizze_156072, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154774_cached_sizze_156073 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154774, &mem_154774_cached_sizze_156073, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154775_cached_sizze_156074 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154775, &mem_154775_cached_sizze_156074, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154776_cached_sizze_156075 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154776, &mem_154776_cached_sizze_156075, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154813_cached_sizze_156076 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154813, &mem_154813_cached_sizze_156076, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154820_cached_sizze_156077 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154820, &mem_154820_cached_sizze_156077, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154825_cached_sizze_156078 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154825, &mem_154825_cached_sizze_156078, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154836_cached_sizze_156079 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154836, &mem_154836_cached_sizze_156079, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154837_cached_sizze_156080 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154837, &mem_154837_cached_sizze_156080, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154846_cached_sizze_156081 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154846, &mem_154846_cached_sizze_156081, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154847_cached_sizze_156082 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154847, &mem_154847_cached_sizze_156082, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154868_cached_sizze_156083 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154868, &mem_154868_cached_sizze_156083, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154869_cached_sizze_156084 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154869, &mem_154869_cached_sizze_156084, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154870_cached_sizze_156085 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154870, &mem_154870_cached_sizze_156085, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154871_cached_sizze_156086 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154871, &mem_154871_cached_sizze_156086, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154896_cached_sizze_156087 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154896, &mem_154896_cached_sizze_156087, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154897_cached_sizze_156088 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154897, &mem_154897_cached_sizze_156088, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154910_cached_sizze_156089 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154910, &mem_154910_cached_sizze_156089, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154911_cached_sizze_156090 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154911, &mem_154911_cached_sizze_156090, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154920_cached_sizze_156091 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154920, &mem_154920_cached_sizze_156091, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154921_cached_sizze_156092 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154921, &mem_154921_cached_sizze_156092, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154942_cached_sizze_156093 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_154942, &mem_154942_cached_sizze_156093, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154947_cached_sizze_156094 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154947, &mem_154947_cached_sizze_156094, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154958_cached_sizze_156095 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_154958, &mem_154958_cached_sizze_156095, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154959_cached_sizze_156096 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_154959, &mem_154959_cached_sizze_156096, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154968_cached_sizze_156097 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154968, &mem_154968_cached_sizze_156097, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154969_cached_sizze_156098 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154969, &mem_154969_cached_sizze_156098, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:585:5-590:58
    if (memblock_set(ctx, &mem_param_152814, &wdown_mem_152781, "wdown_mem_152781") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152818, &wkey_mem_152782, "wkey_mem_152782") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152822, &wout_mem_152783, "wout_mem_152783") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152826, &wpe_mem_152784, "wpe_mem_152784") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152830, &wqry_mem_152785, "wqry_mem_152785") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152834, &wte_mem_152786, "wte_mem_152786") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152838, &wup_mem_152787, "wup_mem_152787") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152842, &wval_mem_152788, "wval_mem_152788") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152846, &wvoc_mem_152789, "wvoc_mem_152789") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152850, &wdown_mem_152790, "wdown_mem_152790") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152854, &wkey_mem_152791, "wkey_mem_152791") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152858, &wout_mem_152792, "wout_mem_152792") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152862, &wpe_mem_152793, "wpe_mem_152793") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152866, &wqry_mem_152794, "wqry_mem_152794") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152870, &wte_mem_152795, "wte_mem_152795") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152874, &wup_mem_152796, "wup_mem_152796") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152878, &wval_mem_152797, "wval_mem_152797") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152882, &wvoc_mem_152798, "wvoc_mem_152798") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152886, &wdown_mem_152799, "wdown_mem_152799") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152890, &wkey_mem_152800, "wkey_mem_152800") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152894, &wout_mem_152801, "wout_mem_152801") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152898, &wpe_mem_152802, "wpe_mem_152802") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152902, &wqry_mem_152803, "wqry_mem_152803") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152906, &wte_mem_152804, "wte_mem_152804") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152910, &wup_mem_152805, "wup_mem_152805") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152914, &wval_mem_152806, "wval_mem_152806") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152918, &wvoc_mem_152807, "wvoc_mem_152807") != 0)
        return 1;
    for (int64_t step_138005 = 0; step_138005 < num_steps_112428; step_138005++) {
        // futhark/microgpt.fut:587:16-25
        
        int64_t dl_138033 = ((int64_t *) dls_mem_152809.mem)[step_138005];
        
        // futhark/microgpt.fut:461:37-40
        
        int64_t zl_rhs_138038 = sub64(dl_138033, (int64_t) 1);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151641 = 0; i_151641 < (int64_t) 16; i_151641++) {
            // futhark/microgpt.fut:461:25-81
            
            bool cond_141900 = slt64(i_151641, zl_rhs_138038);
            
            // futhark/microgpt.fut:461:56-59
            
            int64_t zeze_lhs_141901 = add64((int64_t) 1, i_151641);
            
            // futhark/microgpt.fut:461:47-60
            
            bool x_141902 = sle64((int64_t) 0, zeze_lhs_141901);
            
            // futhark/microgpt.fut:461:47-60
            
            bool y_141903 = slt64(zeze_lhs_141901, (int64_t) 16);
            
            // futhark/microgpt.fut:461:47-60
            
            bool bounds_check_141904 = x_141902 && y_141903;
            
            // futhark/microgpt.fut:9:27-39
            
            bool loop_not_taken_141905 = !cond_141900;
            
            // futhark/microgpt.fut:9:27-39
            
            bool protect_assert_disj_141906 = bounds_check_141904 || loop_not_taken_141905;
            
            // futhark/microgpt.fut:461:47-60
            
            bool index_certs_141907;
            
            if (!protect_assert_disj_141906) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_141901, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:461:47-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:461:3-83\n   #6  futhark/microgpt.fut:542:18-38\n   #7  futhark/microgpt.fut:551:26-557:31\n   #8  futhark/microgpt.fut:590:11-57\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            int64_t tmp_141922 = ((int64_t *) seqs_mem_152810.mem)[step_138005 * (int64_t) 16 + i_151641];
            
            // futhark/microgpt.fut:544:37-51
            
            bool x_141923 = sle64((int64_t) 0, tmp_141922);
            
            // futhark/microgpt.fut:544:37-51
            
            bool y_141924 = slt64(tmp_141922, (int64_t) 27);
            
            // futhark/microgpt.fut:544:37-51
            
            bool bounds_check_141925 = x_141923 && y_141924;
            
            // futhark/microgpt.fut:544:37-51
            
            bool index_certs_141926;
            
            if (!bounds_check_141925) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_141922, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:544:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:544:16-55\n   #6  futhark/microgpt.fut:551:26-557:31\n   #7  futhark/microgpt.fut:590:11-57\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:461:47-60
            
            int64_t zeze_lhs_141908;
            
            if (cond_141900) {
                int64_t x_151307 = ((int64_t *) seqs_mem_152810.mem)[step_138005 * (int64_t) 16 + zeze_lhs_141901];
                
                zeze_lhs_141908 = x_151307;
            } else {
                zeze_lhs_141908 = (int64_t) 0;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151631 = 0; i_151631 < (int64_t) 27; i_151631++) {
                // futhark/microgpt.fut:461:61-65
                
                bool cond_t_res_141912 = zeze_lhs_141908 == i_151631;
                
                // futhark/microgpt.fut:9:27-39
                
                bool x_141913 = cond_141900 && cond_t_res_141912;
                
                // futhark/microgpt.fut:461:25-81
                
                double lifted_lambda_res_141914;
                
                if (x_141913) {
                    lifted_lambda_res_141914 = 1.0;
                } else {
                    lifted_lambda_res_141914 = 0.0;
                }
                ((double *) mem_152929)[i_151631] = lifted_lambda_res_141914;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151635 = 0; i_151635 < (int64_t) 16; i_151635++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141933 = ((double *) mem_param_152834.mem)[tmp_141922 * (int64_t) 16 + i_151635];
                
                ((double *) mem_152936)[i_151635] = lifted_lambda_res_141933;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152919, i_151641 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152936, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152920, i_151641 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152929, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151650 = 0; i_151650 < (int64_t) 16; i_151650++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141999;
            double r_142001 = 0.0;
            
            for (int64_t i_142000 = 0; i_142000 < (int64_t) 16; i_142000++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_142002 = ((double *) mem_param_152826.mem)[i_151650 * (int64_t) 16 + i_142000];
                
                // futhark/microgpt.fut:61:46-49
                
                double zp_rhs_142003 = ((double *) mem_152919)[i_151650 * (int64_t) 16 + i_142000];
                
                // futhark/microgpt.fut:269:63-99
                
                double zp_res_142004 = zp_lhs_142002 + zp_rhs_142003;
                
                // futhark/microgpt.fut:269:79-142
                
                double zt_res_142005 = zp_res_142004 * zp_res_142004;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_142006 = r_142001 + zt_res_142005;
                double r_tmp_155360 = zp_res_142006;
                
                r_142001 = r_tmp_155360;
            }
            defunc_0_lifted_lambda_res_141999 = r_142001;
            // futhark/microgpt.fut:269:42-161
            
            double zs_res_142007 = defunc_0_lifted_lambda_res_141999 / 16.0;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_142014;
            double r_142016 = 0.0;
            
            for (int64_t i_142015 = 0; i_142015 < (int64_t) 16; i_142015++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_142017 = ((double *) mem_param_152826.mem)[i_151650 * (int64_t) 16 + i_142015];
                
                // futhark/microgpt.fut:61:46-49
                
                double zp_rhs_142018 = ((double *) mem_152919)[i_151650 * (int64_t) 16 + i_142015];
                
                // futhark/microgpt.fut:385:71-115
                
                double zp_res_142019 = zp_lhs_142017 + zp_rhs_142018;
                
                // futhark/microgpt.fut:385:91-166
                
                double zt_res_142020 = zp_res_142019 * zp_res_142019;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_142021 = r_142016 + zt_res_142020;
                double r_tmp_155361 = zp_res_142021;
                
                r_142016 = r_tmp_155361;
            }
            defunc_0_lifted_lambda_res_142014 = r_142016;
            // futhark/microgpt.fut:385:48-185
            
            double zs_res_142022 = defunc_0_lifted_lambda_res_142014 / 16.0;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_142032;
            double r_142034 = 0.0;
            
            for (int64_t i_142033 = 0; i_142033 < (int64_t) 16; i_142033++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_142035 = ((double *) mem_param_152826.mem)[i_151650 * (int64_t) 16 + i_142033];
                
                // futhark/microgpt.fut:61:46-49
                
                double zp_rhs_142036 = ((double *) mem_152919)[i_151650 * (int64_t) 16 + i_142033];
                
                // futhark/microgpt.fut:398:72-116
                
                double zp_res_142037 = zp_lhs_142035 + zp_rhs_142036;
                
                // futhark/microgpt.fut:398:92-167
                
                double zt_res_142038 = zp_res_142037 * zp_res_142037;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_142039 = r_142034 + zt_res_142038;
                double r_tmp_155362 = zp_res_142039;
                
                r_142034 = r_tmp_155362;
            }
            defunc_0_lifted_lambda_res_142032 = r_142034;
            // futhark/microgpt.fut:398:49-186
            
            double zs_res_142040 = defunc_0_lifted_lambda_res_142032 / 16.0;
            
            ((double *) mem_152951)[i_151650] = zs_res_142040;
            ((double *) mem_152952)[i_151650] = zs_res_142022;
            ((double *) mem_152953)[i_151650] = zs_res_142007;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151656 = 0; i_151656 < (int64_t) 16; i_151656++) {
            // futhark/microgpt.fut:270:43-51
            
            double zp_lhs_138100 = ((double *) mem_152953)[i_151656];
            
            // futhark/microgpt.fut:270:43-79
            
            double zp_res_138101 = 1.0e-5 + zp_lhs_138100;
            
            // futhark/microgpt.fut:270:35-79
            
            double sqrt_res_138102 = futrts_sqrt64(zp_res_138101);
            
            ((double *) mem_152972)[i_151656] = sqrt_res_138102;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151664 = 0; i_151664 < (int64_t) 16; i_151664++) {
            // futhark/microgpt.fut:271:95-103
            
            double zs_rhs_138110 = ((double *) mem_152972)[i_151664];
            
            // futhark/microgpt.fut:271:87-103
            
            double zs_res_138111 = 1.0 / zs_rhs_138110;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151660 = 0; i_151660 < (int64_t) 16; i_151660++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_138118 = ((double *) mem_param_152826.mem)[i_151664 * (int64_t) 16 + i_151660];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_138119 = ((double *) mem_152919)[i_151664 * (int64_t) 16 + i_151660];
                
                // futhark/microgpt.fut:271:44-80
                
                double zp_res_138120 = zp_lhs_138118 + zp_rhs_138119;
                
                // futhark/microgpt.fut:271:60-103
                
                double zt_res_138121 = zs_res_138111 * zp_res_138120;
                
                ((double *) mem_152984)[i_151660] = zt_res_138121;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152979, i_151664 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152984, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151672 = 0; i_151672 < (int64_t) 16; i_151672++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151668 = 0; i_151668 < (int64_t) 16; i_151668++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_138136 = ((double *) mem_152979)[i_151672 * (int64_t) 16 + i_151668];
                
                ((double *) mem_153000)[i_151668] = lifted_lambda_res_138136;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152995, i_151672 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153000, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151678 = 0; i_151678 < (int64_t) 16; i_151678++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_142059;
            double r_142061 = 0.0;
            
            for (int64_t i_142060 = 0; i_142060 < (int64_t) 16; i_142060++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_142062 = ((double *) mem_152995)[i_151678 * (int64_t) 16 + i_142060];
                
                // futhark/microgpt.fut:273:65-102
                
                double zt_res_142063 = zt_lhs_142062 * zt_lhs_142062;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_142064 = r_142061 + zt_res_142063;
                double r_tmp_155370 = zp_res_142064;
                
                r_142061 = r_tmp_155370;
            }
            defunc_0_lifted_lambda_res_142059 = r_142061;
            // futhark/microgpt.fut:273:44-120
            
            double zs_res_142065 = defunc_0_lifted_lambda_res_142059 / 16.0;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_142072;
            double r_142074 = 0.0;
            
            for (int64_t i_142073 = 0; i_142073 < (int64_t) 16; i_142073++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_142075 = ((double *) mem_152995)[i_151678 * (int64_t) 16 + i_142073];
                
                // futhark/microgpt.fut:363:70-111
                
                double zt_res_142076 = zt_lhs_142075 * zt_lhs_142075;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_142077 = r_142074 + zt_res_142076;
                double r_tmp_155371 = zp_res_142077;
                
                r_142074 = r_tmp_155371;
            }
            defunc_0_lifted_lambda_res_142072 = r_142074;
            // futhark/microgpt.fut:363:48-129
            
            double zs_res_142078 = defunc_0_lifted_lambda_res_142072 / 16.0;
            
            ((double *) mem_153011)[i_151678] = zs_res_142078;
            ((double *) mem_153012)[i_151678] = zs_res_142065;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151683 = 0; i_151683 < (int64_t) 16; i_151683++) {
            // futhark/microgpt.fut:274:45-55
            
            double zp_lhs_138159 = ((double *) mem_153012)[i_151683];
            
            // futhark/microgpt.fut:274:45-83
            
            double zp_res_138160 = 1.0e-5 + zp_lhs_138159;
            
            // futhark/microgpt.fut:274:37-83
            
            double sqrt_res_138161 = futrts_sqrt64(zp_res_138160);
            
            ((double *) mem_153025)[i_151683] = sqrt_res_138161;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151691 = 0; i_151691 < (int64_t) 16; i_151691++) {
            // futhark/microgpt.fut:275:76-86
            
            double zs_rhs_138169 = ((double *) mem_153025)[i_151691];
            
            // futhark/microgpt.fut:275:68-86
            
            double zs_res_138170 = 1.0 / zs_rhs_138169;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151687 = 0; i_151687 < (int64_t) 16; i_151687++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_138177 = ((double *) mem_152995)[i_151691 * (int64_t) 16 + i_151687];
                
                // futhark/microgpt.fut:275:46-86
                
                double zt_res_138178 = zs_res_138170 * zt_lhs_138177;
                
                ((double *) mem_153037)[i_151687] = zt_res_138178;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153032, i_151691 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153037, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151699 = 0; i_151699 < (int64_t) 16; i_151699++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151695 = 0; i_151695 < (int64_t) 16; i_151695++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_138193 = ((double *) mem_153032)[i_151699 * (int64_t) 16 + i_151695];
                
                ((double *) mem_153053)[i_151695] = lifted_lambda_res_138193;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153048, i_151699 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153053, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151727 = 0; i_151727 < (int64_t) 4; i_151727++) {
            // futhark/microgpt.fut:277:83-86
            
            int64_t zp_lhs_142159 = mul64((int64_t) 4, i_151727);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151717 = 0; i_151717 < (int64_t) 16; i_151717++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151707 = 0; i_151707 < (int64_t) 4; i_151707++) {
                    // futhark/microgpt.fut:277:88-95
                    
                    int64_t zt_lhs_146180 = add64(zp_lhs_142159, i_151707);
                    
                    // futhark/microgpt.fut:277:70-97
                    
                    bool x_146181 = sle64((int64_t) 0, zt_lhs_146180);
                    
                    // futhark/microgpt.fut:277:70-97
                    
                    bool y_146182 = slt64(zt_lhs_146180, (int64_t) 16);
                    
                    // futhark/microgpt.fut:277:70-97
                    
                    bool bounds_check_146183 = x_146181 && y_146182;
                    
                    // futhark/microgpt.fut:277:70-97
                    
                    bool index_certs_146184;
                    
                    if (!bounds_check_146183) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_146180, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:277:70-97\n   #1  futhark/microgpt.fut:61:46-49\n   #2  futhark/microgpt.fut:277:49-127\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:15:29-44\n   #8  futhark/microgpt.fut:4:11-25\n   #9  futhark/microgpt.fut:15:15-45\n   #10 futhark/microgpt.fut:277:12-129\n   #11 futhark/microgpt.fut:547:5-76\n   #12 futhark/microgpt.fut:551:26-557:31\n   #13 futhark/microgpt.fut:590:11-57\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:61:13-49
                    
                    double defunc_0_lifted_lambda_res_146185;
                    double r_146187 = 0.0;
                    
                    for (int64_t i_146186 = 0; i_146186 < (int64_t) 16; i_146186++) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_146188 = ((double *) mem_param_152830.mem)[zt_lhs_146180 * (int64_t) 16 + i_146186];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_146189 = ((double *) mem_153048)[i_151717 * (int64_t) 16 + i_146186];
                        
                        // futhark/microgpt.fut:277:70-125
                        
                        double zt_res_146190 = zt_lhs_146188 * zt_rhs_146189;
                        
                        // futhark/microgpt.fut:61:40-49
                        
                        double zp_res_146191 = r_146187 + zt_res_146190;
                        double r_tmp_155386 = zp_res_146191;
                        
                        r_146187 = r_tmp_155386;
                    }
                    defunc_0_lifted_lambda_res_146185 = r_146187;
                    // futhark/microgpt.fut:61:13-49
                    
                    double defunc_0_lifted_lambda_res_146199;
                    double r_146201 = 0.0;
                    
                    for (int64_t i_146200 = 0; i_146200 < (int64_t) 16; i_146200++) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_146202 = ((double *) mem_param_152818.mem)[zt_lhs_146180 * (int64_t) 16 + i_146200];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_146203 = ((double *) mem_153048)[i_151717 * (int64_t) 16 + i_146200];
                        
                        // futhark/microgpt.fut:278:70-125
                        
                        double zt_res_146204 = zt_lhs_146202 * zt_rhs_146203;
                        
                        // futhark/microgpt.fut:61:40-49
                        
                        double zp_res_146205 = r_146201 + zt_res_146204;
                        double r_tmp_155387 = zp_res_146205;
                        
                        r_146201 = r_tmp_155387;
                    }
                    defunc_0_lifted_lambda_res_146199 = r_146201;
                    // futhark/microgpt.fut:61:13-49
                    
                    double defunc_0_lifted_lambda_res_146216;
                    double r_146218 = 0.0;
                    
                    for (int64_t i_146217 = 0; i_146217 < (int64_t) 16; i_146217++) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_146219 = ((double *) mem_param_152842.mem)[zt_lhs_146180 * (int64_t) 16 + i_146217];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_146220 = ((double *) mem_153048)[i_151717 * (int64_t) 16 + i_146217];
                        
                        // futhark/microgpt.fut:279:70-125
                        
                        double zt_res_146221 = zt_lhs_146219 * zt_rhs_146220;
                        
                        // futhark/microgpt.fut:61:40-49
                        
                        double zp_res_146222 = r_146218 + zt_res_146221;
                        double r_tmp_155388 = zp_res_146222;
                        
                        r_146218 = r_tmp_155388;
                    }
                    defunc_0_lifted_lambda_res_146216 = r_146218;
                    ((double *) mem_153097)[i_151707] = defunc_0_lifted_lambda_res_146216;
                    ((double *) mem_153098)[i_151707] = defunc_0_lifted_lambda_res_146199;
                    ((double *) mem_153099)[i_151707] = defunc_0_lifted_lambda_res_146185;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153082, i_151717 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153097, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153083, i_151717 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153098, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153084, i_151717 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153099, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153064, i_151727 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_153082, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153065, i_151727 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_153083, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153066, i_151727 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_153084, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151765 = 0; i_151765 < (int64_t) 4; i_151765++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151752 = 0; i_151752 < (int64_t) 16; i_151752++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151739 = 0; i_151739 < (int64_t) 16; i_151739++) {
                    // futhark/microgpt.fut:61:13-49
                    
                    double defunc_0_lifted_lambda_res_146604;
                    double r_146606 = 0.0;
                    
                    for (int64_t i_146605 = 0; i_146605 < (int64_t) 4; i_146605++) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_146607 = ((double *) mem_153066)[i_151765 * (int64_t) 64 + i_151752 * (int64_t) 4 + i_146605];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_146608 = ((double *) mem_153065)[i_151765 * (int64_t) 64 + i_151739 * (int64_t) 4 + i_146605];
                        
                        // futhark/microgpt.fut:280:111-164
                        
                        double zt_res_146609 = zt_lhs_146607 * zt_rhs_146608;
                        
                        // futhark/microgpt.fut:61:40-49
                        
                        double zp_res_146610 = r_146606 + zt_res_146609;
                        double r_tmp_155401 = zp_res_146610;
                        
                        r_146606 = r_tmp_155401;
                    }
                    defunc_0_lifted_lambda_res_146604 = r_146606;
                    // futhark/microgpt.fut:61:13-49
                    
                    double defunc_0_lifted_lambda_res_146617;
                    double r_146619 = 0.0;
                    
                    for (int64_t i_146618 = 0; i_146618 < (int64_t) 4; i_146618++) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_146620 = ((double *) mem_153066)[i_151765 * (int64_t) 64 + i_151752 * (int64_t) 4 + i_146618];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_146621 = ((double *) mem_153065)[i_151765 * (int64_t) 64 + i_151739 * (int64_t) 4 + i_146618];
                        
                        // futhark/microgpt.fut:322:119-178
                        
                        double zt_res_146622 = zt_lhs_146620 * zt_rhs_146621;
                        
                        // futhark/microgpt.fut:61:40-49
                        
                        double zp_res_146623 = r_146619 + zt_res_146622;
                        double r_tmp_155402 = zp_res_146623;
                        
                        r_146619 = r_tmp_155402;
                    }
                    defunc_0_lifted_lambda_res_146617 = r_146619;
                    // futhark/microgpt.fut:61:13-49
                    
                    double defunc_0_lifted_lambda_res_146633;
                    double r_146635 = 0.0;
                    
                    for (int64_t i_146634 = 0; i_146634 < (int64_t) 4; i_146634++) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_146636 = ((double *) mem_153066)[i_151765 * (int64_t) 64 + i_151752 * (int64_t) 4 + i_146634];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_146637 = ((double *) mem_153065)[i_151765 * (int64_t) 64 + i_151739 * (int64_t) 4 + i_146634];
                        
                        // futhark/microgpt.fut:331:119-178
                        
                        double zt_res_146638 = zt_lhs_146636 * zt_rhs_146637;
                        
                        // futhark/microgpt.fut:61:40-49
                        
                        double zp_res_146639 = r_146635 + zt_res_146638;
                        double r_tmp_155403 = zp_res_146639;
                        
                        r_146635 = r_tmp_155403;
                    }
                    defunc_0_lifted_lambda_res_146633 = r_146635;
                    // futhark/microgpt.fut:61:13-49
                    
                    double defunc_0_lifted_lambda_res_146651;
                    double r_146653 = 0.0;
                    
                    for (int64_t i_146652 = 0; i_146652 < (int64_t) 4; i_146652++) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_146654 = ((double *) mem_153066)[i_151765 * (int64_t) 64 + i_151752 * (int64_t) 4 + i_146652];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_146655 = ((double *) mem_153065)[i_151765 * (int64_t) 64 + i_151739 * (int64_t) 4 + i_146652];
                        
                        // futhark/microgpt.fut:347:119-178
                        
                        double zt_res_146656 = zt_lhs_146654 * zt_rhs_146655;
                        
                        // futhark/microgpt.fut:61:40-49
                        
                        double zp_res_146657 = r_146653 + zt_res_146656;
                        double r_tmp_155404 = zp_res_146657;
                        
                        r_146653 = r_tmp_155404;
                    }
                    defunc_0_lifted_lambda_res_146651 = r_146653;
                    ((double *) mem_153189)[i_151739] = defunc_0_lifted_lambda_res_146651;
                    ((double *) mem_153190)[i_151739] = defunc_0_lifted_lambda_res_146633;
                    ((double *) mem_153191)[i_151739] = defunc_0_lifted_lambda_res_146617;
                    ((double *) mem_153192)[i_151739] = defunc_0_lifted_lambda_res_146604;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153169, i_151752 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153189, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153170, i_151752 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153190, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153171, i_151752 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153191, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153172, i_151752 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153192, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153145, i_151765 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153169, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153146, i_151765 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153170, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153147, i_151765 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153171, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153148, i_151765 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153172, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151804 = 0; i_151804 < (int64_t) 4; i_151804++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151791 = 0; i_151791 < (int64_t) 16; i_151791++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151778 = 0; i_151778 < (int64_t) 16; i_151778++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_147001 = ((double *) mem_153148)[i_151804 * (int64_t) 256 + i_151791 * (int64_t) 16 + i_151778];
                    
                    // futhark/microgpt.fut:281:55-93
                    
                    double zs_res_147002 = zs_lhs_147001 / 2.0;
                    double zp_rhs_147003 = ((double *) masks_mem_152808.mem)[step_138005 * (int64_t) 256 + i_151791 * (int64_t) 16 + i_151778];
                    
                    // futhark/microgpt.fut:281:80-117
                    
                    double zp_res_147004 = zs_res_147002 + zp_rhs_147003;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_147011 = ((double *) mem_153147)[i_151804 * (int64_t) 256 + i_151791 * (int64_t) 16 + i_151778];
                    
                    // futhark/microgpt.fut:323:59-101
                    
                    double zs_res_147012 = zs_lhs_147011 / 2.0;
                    
                    // futhark/microgpt.fut:323:88-127
                    
                    double zp_res_147014 = zp_rhs_147003 + zs_res_147012;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_147024 = ((double *) mem_153146)[i_151804 * (int64_t) 256 + i_151791 * (int64_t) 16 + i_151778];
                    
                    // futhark/microgpt.fut:332:59-101
                    
                    double zs_res_147025 = zs_lhs_147024 / 2.0;
                    
                    // futhark/microgpt.fut:332:88-127
                    
                    double zp_res_147027 = zp_rhs_147003 + zs_res_147025;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_147039 = ((double *) mem_153145)[i_151804 * (int64_t) 256 + i_151791 * (int64_t) 16 + i_151778];
                    
                    // futhark/microgpt.fut:348:59-101
                    
                    double zs_res_147040 = zs_lhs_147039 / 2.0;
                    
                    // futhark/microgpt.fut:348:88-127
                    
                    double zp_res_147042 = zp_rhs_147003 + zs_res_147040;
                    
                    ((double *) mem_153297)[i_151778] = zp_res_147042;
                    ((double *) mem_153298)[i_151778] = zp_res_147027;
                    ((double *) mem_153299)[i_151778] = zp_res_147014;
                    ((double *) mem_153300)[i_151778] = zp_res_147004;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153277, i_151791 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153297, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153278, i_151791 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153298, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153279, i_151791 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153299, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153280, i_151791 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153300, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153253, i_151804 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153277, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153254, i_151804 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153278, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153255, i_151804 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153279, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153256, i_151804 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153280, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151857 = 0; i_151857 < (int64_t) 4; i_151857++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151832 = 0; i_151832 < (int64_t) 16; i_151832++) {
                // futhark/microgpt.fut:105:13-33
                
                double defunc_0_reduce_res_151336;
                double defunc_0_reduce_res_151337;
                double defunc_0_reduce_res_151338;
                double defunc_0_reduce_res_151339;
                double defunc_0_reduce_res_151340;
                double defunc_0_reduce_res_151341;
                double redout_151809;
                double redout_151810;
                double redout_151811;
                double redout_151812;
                double redout_151813;
                double redout_151814;
                
                redout_151809 = -INFINITY;
                redout_151810 = -INFINITY;
                redout_151811 = -INFINITY;
                redout_151812 = -INFINITY;
                redout_151813 = -INFINITY;
                redout_151814 = -INFINITY;
                for (int64_t i_151815 = 0; i_151815 < (int64_t) 16; i_151815++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_148354 = ((double *) mem_153256)[i_151857 * (int64_t) 256 + i_151832 * (int64_t) 16 + i_151815];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_148364 = ((double *) mem_153255)[i_151857 * (int64_t) 256 + i_151832 * (int64_t) 16 + i_151815];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_148383 = ((double *) mem_153254)[i_151857 * (int64_t) 256 + i_151832 * (int64_t) 16 + i_151815];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_148427 = ((double *) mem_153253)[i_151857 * (int64_t) 256 + i_151832 * (int64_t) 16 + i_151815];
                    
                    // futhark/microgpt.fut:105:13-33
                    
                    double max_res_147654 = fmax64(lifted_lambda_res_148354, redout_151809);
                    
                    // futhark/microgpt.fut:105:13-33
                    
                    double max_res_147673 = fmax64(lifted_lambda_res_148364, redout_151810);
                    
                    // futhark/microgpt.fut:105:13-33
                    
                    double max_res_147695 = fmax64(lifted_lambda_res_148383, redout_151811);
                    
                    // futhark/microgpt.fut:105:13-33
                    
                    double max_res_147720 = fmax64(lifted_lambda_res_148383, redout_151812);
                    
                    // futhark/microgpt.fut:105:13-33
                    
                    double max_res_147770 = fmax64(lifted_lambda_res_148427, redout_151813);
                    
                    // futhark/microgpt.fut:105:13-33
                    
                    double max_res_147801 = fmax64(lifted_lambda_res_148427, redout_151814);
                    double redout_tmp_155433 = max_res_147654;
                    double redout_tmp_155434 = max_res_147673;
                    double redout_tmp_155435 = max_res_147695;
                    double redout_tmp_155436 = max_res_147720;
                    double redout_tmp_155437 = max_res_147770;
                    double redout_tmp_155438 = max_res_147801;
                    
                    redout_151809 = redout_tmp_155433;
                    redout_151810 = redout_tmp_155434;
                    redout_151811 = redout_tmp_155435;
                    redout_151812 = redout_tmp_155436;
                    redout_151813 = redout_tmp_155437;
                    redout_151814 = redout_tmp_155438;
                }
                defunc_0_reduce_res_151336 = redout_151809;
                defunc_0_reduce_res_151337 = redout_151810;
                defunc_0_reduce_res_151338 = redout_151811;
                defunc_0_reduce_res_151339 = redout_151812;
                defunc_0_reduce_res_151340 = redout_151813;
                defunc_0_reduce_res_151341 = redout_151814;
                // futhark/microgpt.fut:343:172-198
                
                double neg_res_147728 = -defunc_0_reduce_res_151339;
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_147729;
                double r_147731 = 0.0;
                
                for (int64_t i_147730 = 0; i_147730 < (int64_t) 16; i_147730++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zp_lhs_147732 = ((double *) mem_153254)[i_151857 * (int64_t) 256 + i_151832 * (int64_t) 16 + i_147730];
                    
                    // futhark/microgpt.fut:343:138-198
                    
                    double zp_res_147733 = neg_res_147728 + zp_lhs_147732;
                    
                    // futhark/microgpt.fut:343:131-198
                    
                    double neg_res_147734 = -zp_res_147733;
                    
                    // futhark/microgpt.fut:100:42-54
                    
                    double max_res_147735 = fmax64(0.0, neg_res_147734);
                    
                    // futhark/microgpt.fut:100:35-54
                    
                    double sgn_res_147736 = fsignum64(max_res_147735);
                    
                    // futhark/microgpt.fut:343:112-201
                    
                    double neg_res_147737 = -sgn_res_147736;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_147738 = r_147731 + neg_res_147737;
                    double r_tmp_155439 = zp_res_147738;
                    
                    r_147731 = r_tmp_155439;
                }
                defunc_0_lifted_lambda_res_147729 = r_147731;
                // futhark/microgpt.fut:343:58-204
                
                double zp_res_147739 = defunc_0_lifted_lambda_res_139667 + defunc_0_lifted_lambda_res_147729;
                
                // futhark/microgpt.fut:343:48-204
                
                double zs_res_147740 = 1.0 / zp_res_147739;
                
                // futhark/microgpt.fut:359:172-198
                
                double neg_res_147809 = -defunc_0_reduce_res_151341;
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_147810;
                double r_147812 = 0.0;
                
                for (int64_t i_147811 = 0; i_147811 < (int64_t) 16; i_147811++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zp_lhs_147813 = ((double *) mem_153253)[i_151857 * (int64_t) 256 + i_151832 * (int64_t) 16 + i_147811];
                    
                    // futhark/microgpt.fut:359:138-198
                    
                    double zp_res_147814 = neg_res_147809 + zp_lhs_147813;
                    
                    // futhark/microgpt.fut:359:131-198
                    
                    double neg_res_147815 = -zp_res_147814;
                    
                    // futhark/microgpt.fut:100:42-54
                    
                    double max_res_147816 = fmax64(0.0, neg_res_147815);
                    
                    // futhark/microgpt.fut:100:35-54
                    
                    double sgn_res_147817 = fsignum64(max_res_147816);
                    
                    // futhark/microgpt.fut:359:112-201
                    
                    double neg_res_147818 = -sgn_res_147817;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_147819 = r_147812 + neg_res_147818;
                    double r_tmp_155440 = zp_res_147819;
                    
                    r_147812 = r_tmp_155440;
                }
                defunc_0_lifted_lambda_res_147810 = r_147812;
                // futhark/microgpt.fut:359:58-204
                
                double zp_res_147820 = defunc_0_lifted_lambda_res_140096 + defunc_0_lifted_lambda_res_147810;
                
                // futhark/microgpt.fut:359:48-204
                
                double zs_res_147821 = 1.0 / zp_res_147820;
                
                ((double *) mem_153401)[i_151832] = zs_res_147821;
                ((double *) mem_153402)[i_151832] = defunc_0_reduce_res_151341;
                ((double *) mem_153403)[i_151832] = defunc_0_reduce_res_151340;
                ((double *) mem_153404)[i_151832] = zs_res_147740;
                ((double *) mem_153405)[i_151832] = defunc_0_reduce_res_151339;
                ((double *) mem_153406)[i_151832] = defunc_0_reduce_res_151338;
                ((double *) mem_153407)[i_151832] = defunc_0_reduce_res_151337;
                ((double *) mem_153408)[i_151832] = defunc_0_reduce_res_151336;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153361, i_151857 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153401, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153362, i_151857 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153402, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153363, i_151857 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153403, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153364, i_151857 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153404, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153365, i_151857 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153405, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153366, i_151857 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153406, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153367, i_151857 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153407, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153368, i_151857 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153408, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151900 = 0; i_151900 < (int64_t) 4; i_151900++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151887 = 0; i_151887 < (int64_t) 16; i_151887++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_148643 = ((double *) mem_153368)[i_151900 * (int64_t) 16 + i_151887];
                
                // futhark/microgpt.fut:283:91-114
                
                double neg_res_148644 = -neg_arg0_148643;
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_148705 = ((double *) mem_153363)[i_151900 * (int64_t) 16 + i_151887];
                
                // futhark/microgpt.fut:352:99-125
                
                double neg_res_148706 = -neg_arg0_148705;
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_148682 = ((double *) mem_153366)[i_151900 * (int64_t) 16 + i_151887];
                
                // futhark/microgpt.fut:336:99-125
                
                double neg_res_148683 = -neg_arg0_148682;
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_148661 = ((double *) mem_153367)[i_151900 * (int64_t) 16 + i_151887];
                
                // futhark/microgpt.fut:325:99-125
                
                double neg_res_148662 = -neg_arg0_148661;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151874 = 0; i_151874 < (int64_t) 16; i_151874++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_148825 = ((double *) mem_153256)[i_151900 * (int64_t) 256 + i_151887 * (int64_t) 16 + i_151874];
                    
                    // futhark/microgpt.fut:283:61-114
                    
                    double zp_res_148826 = neg_res_148644 + zp_lhs_148825;
                    
                    // futhark/microgpt.fut:283:54-114
                    
                    double exp_res_148827 = futrts_exp64(zp_res_148826);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_148834 = ((double *) mem_153255)[i_151900 * (int64_t) 256 + i_151887 * (int64_t) 16 + i_151874];
                    
                    // futhark/microgpt.fut:325:65-125
                    
                    double zp_res_148835 = neg_res_148662 + zp_lhs_148834;
                    
                    // futhark/microgpt.fut:325:58-125
                    
                    double exp_res_148836 = futrts_exp64(zp_res_148835);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_148846 = ((double *) mem_153254)[i_151900 * (int64_t) 256 + i_151887 * (int64_t) 16 + i_151874];
                    
                    // futhark/microgpt.fut:336:65-125
                    
                    double zp_res_148847 = neg_res_148683 + zp_lhs_148846;
                    
                    // futhark/microgpt.fut:336:58-125
                    
                    double exp_res_148848 = futrts_exp64(zp_res_148847);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_148860 = ((double *) mem_153253)[i_151900 * (int64_t) 256 + i_151887 * (int64_t) 16 + i_151874];
                    
                    // futhark/microgpt.fut:352:65-125
                    
                    double zp_res_148861 = neg_res_148706 + zp_lhs_148860;
                    
                    // futhark/microgpt.fut:352:58-125
                    
                    double exp_res_148862 = futrts_exp64(zp_res_148861);
                    
                    ((double *) mem_153533)[i_151874] = exp_res_148862;
                    ((double *) mem_153534)[i_151874] = exp_res_148848;
                    ((double *) mem_153535)[i_151874] = exp_res_148836;
                    ((double *) mem_153536)[i_151874] = exp_res_148827;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153513, i_151887 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153533, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153514, i_151887 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153534, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153515, i_151887 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153535, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153516, i_151887 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153536, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153489, i_151900 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153513, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153490, i_151900 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153514, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153491, i_151900 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153515, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153492, i_151900 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153516, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151916 = 0; i_151916 < (int64_t) 4; i_151916++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151909 = 0; i_151909 < (int64_t) 16; i_151909++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_148894;
                double r_148896 = 0.0;
                
                for (int64_t i_148895 = 0; i_148895 < (int64_t) 16; i_148895++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double lifted_lambda_res_148897 = ((double *) mem_153492)[i_151916 * (int64_t) 256 + i_151909 * (int64_t) 16 + i_148895];
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_148898 = r_148896 + lifted_lambda_res_148897;
                    double r_tmp_155457 = zp_res_148898;
                    
                    r_148896 = r_tmp_155457;
                }
                defunc_0_lifted_lambda_res_148894 = r_148896;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_148905;
                double r_148907 = 0.0;
                
                for (int64_t i_148906 = 0; i_148906 < (int64_t) 16; i_148906++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double lifted_lambda_res_148908 = ((double *) mem_153491)[i_151916 * (int64_t) 256 + i_151909 * (int64_t) 16 + i_148906];
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_148909 = r_148907 + lifted_lambda_res_148908;
                    double r_tmp_155458 = zp_res_148909;
                    
                    r_148907 = r_tmp_155458;
                }
                defunc_0_lifted_lambda_res_148905 = r_148907;
                ((double *) mem_153607)[i_151909] = defunc_0_lifted_lambda_res_148905;
                ((double *) mem_153608)[i_151909] = defunc_0_lifted_lambda_res_148894;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153597, i_151916 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153607, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153598, i_151916 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153608, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151937 = 0; i_151937 < (int64_t) 4; i_151937++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151930 = 0; i_151930 < (int64_t) 16; i_151930++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_148929 = ((double *) mem_153598)[i_151937 * (int64_t) 16 + i_151930];
                
                // futhark/microgpt.fut:285:84-109
                
                double zs_res_148930 = 1.0 / zs_rhs_148929;
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_148946 = ((double *) mem_153597)[i_151937 * (int64_t) 16 + i_151930];
                
                // futhark/microgpt.fut:327:92-120
                
                double zs_res_148947 = 1.0 / zs_rhs_148946;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151923 = 0; i_151923 < (int64_t) 16; i_151923++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_148974 = ((double *) mem_153492)[i_151937 * (int64_t) 256 + i_151930 * (int64_t) 16 + i_151923];
                    
                    // futhark/microgpt.fut:285:54-109
                    
                    double zt_res_148975 = zs_res_148930 * zt_lhs_148974;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_148982 = ((double *) mem_153491)[i_151937 * (int64_t) 256 + i_151930 * (int64_t) 16 + i_151923];
                    
                    // futhark/microgpt.fut:327:58-120
                    
                    double zt_res_148983 = zs_res_148947 * zt_lhs_148982;
                    
                    ((double *) mem_153651)[i_151923] = zt_res_148983;
                    ((double *) mem_153652)[i_151923] = zt_res_148975;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153641, i_151930 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153651, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153642, i_151930 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153652, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153629, i_151937 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153641, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153630, i_151937 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153642, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151958 = 0; i_151958 < (int64_t) 4; i_151958++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151951 = 0; i_151951 < (int64_t) 16; i_151951++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151944 = 0; i_151944 < (int64_t) 16; i_151944++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_149046 = ((double *) mem_153630)[i_151958 * (int64_t) 256 + i_151951 * (int64_t) 16 + i_151944];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_149053 = ((double *) mem_153629)[i_151958 * (int64_t) 256 + i_151951 * (int64_t) 16 + i_151944];
                    
                    ((double *) mem_153705)[i_151944] = lifted_lambda_res_149053;
                    ((double *) mem_153706)[i_151944] = lifted_lambda_res_149046;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153695, i_151951 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153705, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153696, i_151951 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153706, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153683, i_151958 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153695, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153684, i_151958 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153696, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151971 = 0; i_151971 < (int64_t) 4; i_151971++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151967 = 0; i_151967 < (int64_t) 16; i_151967++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151963 = 0; i_151963 < (int64_t) 4; i_151963++) {
                    // futhark/microgpt.fut:61:13-49
                    
                    double defunc_0_lifted_lambda_res_138465;
                    double r_138467 = 0.0;
                    
                    for (int64_t i_138466 = 0; i_138466 < (int64_t) 16; i_138466++) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_138468 = ((double *) mem_153684)[i_151971 * (int64_t) 256 + i_151967 * (int64_t) 16 + i_138466];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_138469 = ((double *) mem_153064)[i_151971 * (int64_t) 64 + i_138466 * (int64_t) 4 + i_151963];
                        
                        // futhark/microgpt.fut:287:74-127
                        
                        double zt_res_138470 = zt_lhs_138468 * zt_rhs_138469;
                        
                        // futhark/microgpt.fut:61:40-49
                        
                        double zp_res_138471 = r_138467 + zt_res_138470;
                        double r_tmp_155474 = zp_res_138471;
                        
                        r_138467 = r_tmp_155474;
                    }
                    defunc_0_lifted_lambda_res_138465 = r_138467;
                    ((double *) mem_153748)[i_151963] = defunc_0_lifted_lambda_res_138465;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153743, i_151967 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153748, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153737, i_151971 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_153743, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151979 = 0; i_151979 < (int64_t) 16; i_151979++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151975 = 0; i_151975 < (int64_t) 16; i_151975++) {
                // futhark/microgpt.fut:288:15-18
                
                int64_t tmp_138483 = sdiv64(i_151975, (int64_t) 4);
                
                // futhark/microgpt.fut:288:4-20
                
                bool x_138484 = sle64((int64_t) 0, tmp_138483);
                
                // futhark/microgpt.fut:288:4-20
                
                bool y_138485 = slt64(tmp_138483, (int64_t) 4);
                
                // futhark/microgpt.fut:288:4-20
                
                bool bounds_check_138486 = x_138484 && y_138485;
                
                // futhark/microgpt.fut:288:4-20
                
                bool index_certs_138487;
                
                if (!bounds_check_138486) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_138483, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:288:4-20\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:280:12-288:49\n   #6  futhark/microgpt.fut:547:5-76\n   #7  futhark/microgpt.fut:551:26-557:31\n   #8  futhark/microgpt.fut:590:11-57\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:288:35-38
                
                int64_t tmp_138488 = smod64(i_151975, (int64_t) 4);
                
                // futhark/microgpt.fut:288:4-40
                
                bool x_138489 = sle64((int64_t) 0, tmp_138488);
                
                // futhark/microgpt.fut:288:4-40
                
                bool y_138490 = slt64(tmp_138488, (int64_t) 4);
                
                // futhark/microgpt.fut:288:4-40
                
                bool bounds_check_138491 = x_138489 && y_138490;
                
                // futhark/microgpt.fut:288:4-40
                
                bool index_certs_138492;
                
                if (!bounds_check_138491) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_138488, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:288:4-40\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:280:12-288:49\n   #6  futhark/microgpt.fut:547:5-76\n   #7  futhark/microgpt.fut:551:26-557:31\n   #8  futhark/microgpt.fut:590:11-57\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_138493 = ((double *) mem_153737)[tmp_138483 * (int64_t) 64 + i_151979 * (int64_t) 4 + tmp_138488];
                
                ((double *) mem_153769)[i_151975] = lifted_lambda_res_138493;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153764, i_151979 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153769, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151987 = 0; i_151987 < (int64_t) 16; i_151987++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151983 = 0; i_151983 < (int64_t) 16; i_151983++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_138508;
                double r_138510 = 0.0;
                
                for (int64_t i_138509 = 0; i_138509 < (int64_t) 16; i_138509++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_138511 = ((double *) mem_param_152822.mem)[i_151983 * (int64_t) 16 + i_138509];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_138512 = ((double *) mem_153764)[i_151987 * (int64_t) 16 + i_138509];
                    
                    // futhark/microgpt.fut:289:64-104
                    
                    double zt_res_138513 = zt_lhs_138511 * zt_rhs_138512;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_138514 = r_138510 + zt_res_138513;
                    double r_tmp_155479 = zp_res_138514;
                    
                    r_138510 = r_tmp_155479;
                }
                defunc_0_lifted_lambda_res_138508 = r_138510;
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_138515 = ((double *) mem_152995)[i_151987 * (int64_t) 16 + i_151983];
                
                // futhark/microgpt.fut:289:43-128
                
                double zp_res_138516 = defunc_0_lifted_lambda_res_138508 + zp_rhs_138515;
                
                ((double *) mem_153785)[i_151983] = zp_res_138516;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153780, i_151987 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153785, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151993 = 0; i_151993 < (int64_t) 16; i_151993++) {
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_144136;
            double r_144138 = 0.0;
            
            for (int64_t i_144137 = 0; i_144137 < (int64_t) 16; i_144137++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_144139 = ((double *) mem_153780)[i_151993 * (int64_t) 16 + i_144137];
                
                // futhark/microgpt.fut:290:66-105
                
                double zt_res_144140 = zt_lhs_144139 * zt_lhs_144139;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_144141 = r_144138 + zt_res_144140;
                double r_tmp_155482 = zp_res_144141;
                
                r_144138 = r_tmp_155482;
            }
            defunc_0_lifted_lambda_res_144136 = r_144138;
            // futhark/microgpt.fut:290:45-123
            
            double zs_res_144142 = defunc_0_lifted_lambda_res_144136 / 16.0;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_144149;
            double r_144151 = 0.0;
            
            for (int64_t i_144150 = 0; i_144150 < (int64_t) 16; i_144150++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_144152 = ((double *) mem_153780)[i_151993 * (int64_t) 16 + i_144150];
                
                // futhark/microgpt.fut:315:70-113
                
                double zt_res_144153 = zt_lhs_144152 * zt_lhs_144152;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_144154 = r_144151 + zt_res_144153;
                double r_tmp_155483 = zp_res_144154;
                
                r_144151 = r_tmp_155483;
            }
            defunc_0_lifted_lambda_res_144149 = r_144151;
            // futhark/microgpt.fut:315:48-131
            
            double zs_res_144155 = defunc_0_lifted_lambda_res_144149 / 16.0;
            
            ((double *) mem_153796)[i_151993] = zs_res_144155;
            ((double *) mem_153797)[i_151993] = zs_res_144142;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151998 = 0; i_151998 < (int64_t) 16; i_151998++) {
            // futhark/microgpt.fut:291:45-55
            
            double zp_lhs_138539 = ((double *) mem_153797)[i_151998];
            
            // futhark/microgpt.fut:291:45-83
            
            double zp_res_138540 = 1.0e-5 + zp_lhs_138539;
            
            // futhark/microgpt.fut:291:37-83
            
            double sqrt_res_138541 = futrts_sqrt64(zp_res_138540);
            
            ((double *) mem_153810)[i_151998] = sqrt_res_138541;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152006 = 0; i_152006 < (int64_t) 16; i_152006++) {
            // futhark/microgpt.fut:292:77-87
            
            double zs_rhs_138549 = ((double *) mem_153810)[i_152006];
            
            // futhark/microgpt.fut:292:69-87
            
            double zs_res_138550 = 1.0 / zs_rhs_138549;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152002 = 0; i_152002 < (int64_t) 16; i_152002++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_138557 = ((double *) mem_153780)[i_152006 * (int64_t) 16 + i_152002];
                
                // futhark/microgpt.fut:292:46-87
                
                double zt_res_138558 = zs_res_138550 * zt_lhs_138557;
                
                ((double *) mem_153822)[i_152002] = zt_res_138558;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153817, i_152006 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153822, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152014 = 0; i_152014 < (int64_t) 16; i_152014++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152010 = 0; i_152010 < (int64_t) 16; i_152010++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_138573 = ((double *) mem_153817)[i_152014 * (int64_t) 16 + i_152010];
                
                ((double *) mem_153838)[i_152010] = lifted_lambda_res_138573;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153833, i_152014 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153838, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152022 = 0; i_152022 < (int64_t) 16; i_152022++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152018 = 0; i_152018 < (int64_t) 64; i_152018++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_138589;
                double r_138591 = 0.0;
                
                for (int64_t i_138590 = 0; i_138590 < (int64_t) 16; i_138590++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_138592 = ((double *) mem_param_152838.mem)[i_152018 * (int64_t) 16 + i_138590];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_138593 = ((double *) mem_153833)[i_152022 * (int64_t) 16 + i_138590];
                    
                    // futhark/microgpt.fut:294:63-102
                    
                    double zt_res_138594 = zt_lhs_138592 * zt_rhs_138593;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_138595 = r_138591 + zt_res_138594;
                    double r_tmp_155491 = zp_res_138595;
                    
                    r_138591 = r_tmp_155491;
                }
                defunc_0_lifted_lambda_res_138589 = r_138591;
                ((double *) mem_153854)[i_152018] = defunc_0_lifted_lambda_res_138589;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153849, i_152022 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153854, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152030 = 0; i_152030 < (int64_t) 16; i_152030++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152026 = 0; i_152026 < (int64_t) 64; i_152026++) {
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_138610 = ((double *) mem_153849)[i_152030 * (int64_t) 64 + i_152026];
                
                // futhark/microgpt.fut:295:41-69
                
                double max_res_138611 = fmax64(0.0, max_arg0_138610);
                
                ((double *) mem_153870)[i_152026] = max_res_138611;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153865, i_152030 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153870, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152038 = 0; i_152038 < (int64_t) 16; i_152038++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152034 = 0; i_152034 < (int64_t) 16; i_152034++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_138626;
                double r_138628 = 0.0;
                
                for (int64_t i_138627 = 0; i_138627 < (int64_t) 64; i_138627++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_138629 = ((double *) mem_param_152814.mem)[i_152034 * (int64_t) 64 + i_138627];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_138630 = ((double *) mem_153865)[i_152038 * (int64_t) 64 + i_138627];
                    
                    // futhark/microgpt.fut:296:64-105
                    
                    double zt_res_138631 = zt_lhs_138629 * zt_rhs_138630;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_138632 = r_138628 + zt_res_138631;
                    double r_tmp_155496 = zp_res_138632;
                    
                    r_138628 = r_tmp_155496;
                }
                defunc_0_lifted_lambda_res_138626 = r_138628;
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_138633 = ((double *) mem_153780)[i_152038 * (int64_t) 16 + i_152034];
                
                // futhark/microgpt.fut:296:43-130
                
                double zp_res_138634 = defunc_0_lifted_lambda_res_138626 + zp_rhs_138633;
                
                ((double *) mem_153886)[i_152034] = zp_res_138634;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153881, i_152038 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153886, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152046 = 0; i_152046 < (int64_t) 16; i_152046++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152042 = 0; i_152042 < (int64_t) 27; i_152042++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_138649;
                double r_138651 = 0.0;
                
                for (int64_t i_138650 = 0; i_138650 < (int64_t) 16; i_138650++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_138652 = ((double *) mem_param_152846.mem)[i_152042 * (int64_t) 16 + i_138650];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_138653 = ((double *) mem_153881)[i_152046 * (int64_t) 16 + i_138650];
                    
                    // futhark/microgpt.fut:297:63-103
                    
                    double zt_res_138654 = zt_lhs_138652 * zt_rhs_138653;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_138655 = r_138651 + zt_res_138654;
                    double r_tmp_155499 = zp_res_138655;
                    
                    r_138651 = r_tmp_155499;
                }
                defunc_0_lifted_lambda_res_138649 = r_138651;
                ((double *) mem_153902)[i_152042] = defunc_0_lifted_lambda_res_138649;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153897, i_152046 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153902, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152076 = 0; i_152076 < (int64_t) 16; i_152076++) {
            // futhark/microgpt.fut:105:13-33
            
            double defunc_0_reduce_res_151439;
            double defunc_0_reduce_res_151440;
            double redout_152063;
            double redout_152064;
            
            redout_152063 = -INFINITY;
            redout_152064 = -INFINITY;
            for (int64_t i_152066 = 0; i_152066 < (int64_t) 27; i_152066++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_149224 = ((double *) mem_153897)[i_152076 * (int64_t) 27 + i_152066];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_152060 = 0; i_152060 < (int64_t) 27; i_152060++) {
                    // futhark/microgpt.fut:302:55-306:90
                    
                    bool cond_149233 = i_152060 == i_152066;
                    
                    // futhark/microgpt.fut:302:55-306:90
                    
                    double lifted_lambda_res_149234;
                    
                    if (cond_149233) {
                        // futhark/microgpt.fut:105:13-33
                        
                        double defunc_0_reduce_res_151386;
                        double redout_152048 = -INFINITY;
                        
                        for (int64_t i_152049 = 0; i_152049 < (int64_t) 27; i_152049++) {
                            // futhark/microgpt.fut:4:11-25
                            
                            double lifted_lambda_res_151392 = ((double *) mem_153897)[i_152076 * (int64_t) 27 + i_152049];
                            
                            // futhark/microgpt.fut:105:13-33
                            
                            double max_res_151395 = fmax64(lifted_lambda_res_151392, redout_152048);
                            double redout_tmp_155508 = max_res_151395;
                            
                            redout_152048 = redout_tmp_155508;
                        }
                        defunc_0_reduce_res_151386 = redout_152048;
                        // futhark/microgpt.fut:303:67-76
                        
                        double neg_res_151397 = -defunc_0_reduce_res_151386;
                        
                        // futhark/microgpt.fut:4:11-25
                        if (mem_153943_cached_sizze_155976 < (int64_t) 216) {
                            err = lexical_realloc(ctx, &mem_153943, &mem_153943_cached_sizze_155976, (int64_t) 216);
                            if (err != FUTHARK_SUCCESS)
                                goto cleanup;
                        }
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_152052 = 0; i_152052 < (int64_t) 27; i_152052++) {
                            // futhark/microgpt.fut:4:11-25
                            
                            double zp_lhs_151404 = ((double *) mem_153897)[i_152076 * (int64_t) 27 + i_152052];
                            
                            // futhark/microgpt.fut:303:44-76
                            
                            double zp_res_151405 = neg_res_151397 + zp_lhs_151404;
                            
                            // futhark/microgpt.fut:303:37-76
                            
                            double exp_res_151406 = futrts_exp64(zp_res_151405);
                            
                            ((double *) mem_153943)[i_152052] = exp_res_151406;
                        }
                        // futhark/microgpt.fut:61:13-49
                        
                        double defunc_0_lifted_lambda_res_151409;
                        double r_151411 = 0.0;
                        
                        for (int64_t i_151410 = 0; i_151410 < (int64_t) 27; i_151410++) {
                            // futhark/microgpt.fut:304:36-46
                            
                            double lifted_lambda_res_151412 = ((double *) mem_153943)[i_151410];
                            
                            // futhark/microgpt.fut:61:40-49
                            
                            double zp_res_151413 = r_151411 + lifted_lambda_res_151412;
                            double r_tmp_155510 = zp_res_151413;
                            
                            r_151411 = r_tmp_155510;
                        }
                        defunc_0_lifted_lambda_res_151409 = r_151411;
                        // futhark/microgpt.fut:305:53-64
                        
                        double zs_res_151414 = 1.0 / defunc_0_lifted_lambda_res_151409;
                        
                        // futhark/microgpt.fut:4:11-25
                        if (mem_153950_cached_sizze_155977 < (int64_t) 216) {
                            err = lexical_realloc(ctx, &mem_153950, &mem_153950_cached_sizze_155977, (int64_t) 216);
                            if (err != FUTHARK_SUCCESS)
                                goto cleanup;
                        }
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_152056 = 0; i_152056 < (int64_t) 27; i_152056++) {
                            // futhark/microgpt.fut:305:37-47
                            
                            double zt_lhs_151421 = ((double *) mem_153943)[i_152056];
                            
                            // futhark/microgpt.fut:305:37-64
                            
                            double zt_res_151422 = zs_res_151414 * zt_lhs_151421;
                            
                            ((double *) mem_153950)[i_152056] = zt_res_151422;
                        }
                        // futhark/microgpt.fut:4:11-25
                        
                        double zt_rhs_151429 = ((double *) mem_152920)[i_152076 * (int64_t) 27 + i_152066];
                        
                        // futhark/microgpt.fut:306:7-49
                        
                        double zt_res_151430 = -6.25e-2 * zt_rhs_151429;
                        
                        // futhark/microgpt.fut:306:64-74
                        
                        double zs_rhs_151435 = ((double *) mem_153950)[i_152060];
                        
                        // futhark/microgpt.fut:306:56-74
                        
                        double zs_res_151436 = 1.0 / zs_rhs_151435;
                        
                        // futhark/microgpt.fut:306:25-74
                        
                        double zt_res_151437 = zt_res_151430 * zs_res_151436;
                        
                        lifted_lambda_res_149234 = zt_res_151437;
                    } else {
                        lifted_lambda_res_149234 = 0.0;
                    }
                    ((double *) mem_153939)[i_152060] = lifted_lambda_res_149234;
                }
                // futhark/microgpt.fut:105:13-33
                
                double max_res_144292 = fmax64(lifted_lambda_res_149224, redout_152063);
                
                // futhark/microgpt.fut:105:13-33
                
                double max_res_144383 = fmax64(lifted_lambda_res_149224, redout_152064);
                
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153934, i_152066 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153939, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
                
                double redout_tmp_155504 = max_res_144292;
                double redout_tmp_155505 = max_res_144383;
                
                redout_152063 = redout_tmp_155504;
                redout_152064 = redout_tmp_155505;
            }
            defunc_0_reduce_res_151439 = redout_152063;
            defunc_0_reduce_res_151440 = redout_152064;
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_155512 = 0; nest_i_155512 < (int64_t) 27; nest_i_155512++) {
                ((double *) mem_153916)[i_152076 * (int64_t) 27 + nest_i_155512] = defunc_0_reduce_res_151439;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_155513 = 0; nest_i_155513 < (int64_t) 27; nest_i_155513++) {
                ((double *) mem_153914)[i_152076 * (int64_t) 27 + nest_i_155513] = defunc_0_reduce_res_151440;
            }
            // futhark/microgpt.fut:311:163-188
            
            double neg_res_144394 = -defunc_0_reduce_res_151440;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_144395;
            double r_144397 = 0.0;
            
            for (int64_t i_144396 = 0; i_144396 < (int64_t) 27; i_144396++) {
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_144398 = ((double *) mem_153897)[i_152076 * (int64_t) 27 + i_144396];
                
                // futhark/microgpt.fut:311:138-188
                
                double zp_res_144399 = neg_res_144394 + zp_lhs_144398;
                
                // futhark/microgpt.fut:311:131-188
                
                double neg_res_144400 = -zp_res_144399;
                
                // futhark/microgpt.fut:100:42-54
                
                double max_res_144401 = fmax64(0.0, neg_res_144400);
                
                // futhark/microgpt.fut:100:35-54
                
                double sgn_res_144402 = fsignum64(max_res_144401);
                
                // futhark/microgpt.fut:311:112-191
                
                double neg_res_144403 = -sgn_res_144402;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_144404 = r_144397 + neg_res_144403;
                double r_tmp_155514 = zp_res_144404;
                
                r_144397 = r_tmp_155514;
            }
            defunc_0_lifted_lambda_res_144395 = r_144397;
            // futhark/microgpt.fut:311:58-194
            
            double zp_res_144405 = defunc_0_lifted_lambda_res_138894 + defunc_0_lifted_lambda_res_144395;
            
            // futhark/microgpt.fut:311:48-194
            
            double zs_res_144406 = 1.0 / zp_res_144405;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_155515 = 0; nest_i_155515 < (int64_t) 27; nest_i_155515++) {
                ((double *) mem_153913)[i_152076 * (int64_t) 27 + nest_i_155515] = zs_res_144406;
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153915, i_152076 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_153934, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152091 = 0; i_152091 < (int64_t) 16; i_152091++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152087 = 0; i_152087 < (int64_t) 27; i_152087++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_138691 = ((double *) mem_153916)[i_152091 * (int64_t) 27 + i_152087];
                
                // futhark/microgpt.fut:300:85-108
                
                double neg_res_138692 = -neg_arg0_138691;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_152083 = 0; i_152083 < (int64_t) 27; i_152083++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_138699 = ((double *) mem_153897)[i_152091 * (int64_t) 27 + i_152083];
                    
                    // futhark/microgpt.fut:300:62-108
                    
                    double zp_res_138700 = neg_res_138692 + zp_lhs_138699;
                    
                    // futhark/microgpt.fut:300:55-108
                    
                    double exp_res_138701 = futrts_exp64(zp_res_138700);
                    
                    ((double *) mem_153995)[i_152083] = exp_res_138701;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153990, i_152087 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153995, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153984, i_152091 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_153990, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152104 = 0; i_152104 < (int64_t) 16; i_152104++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152097 = 0; i_152097 < (int64_t) 27; i_152097++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_149598;
                double r_149600 = 0.0;
                
                for (int64_t i_149599 = 0; i_149599 < (int64_t) 27; i_149599++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double lifted_lambda_res_149601 = ((double *) mem_153984)[i_152104 * (int64_t) 729 + i_152097 * (int64_t) 27 + i_149599];
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_149602 = r_149600 + lifted_lambda_res_149601;
                    double r_tmp_155523 = zp_res_149602;
                    
                    r_149600 = r_tmp_155523;
                }
                defunc_0_lifted_lambda_res_149598 = r_149600;
                // futhark/microgpt.fut:307:147-186
                
                double zt_res_149610 = defunc_0_lifted_lambda_res_149598 * defunc_0_lifted_lambda_res_149598;
                
                // futhark/microgpt.fut:307:138-186
                
                double zs_res_149611 = 1.0 / zt_res_149610;
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_149612;
                double r_149614 = 0.0;
                
                for (int64_t i_149613 = 0; i_149613 < (int64_t) 27; i_149613++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_149615 = ((double *) mem_153915)[i_152104 * (int64_t) 729 + i_152097 * (int64_t) 27 + i_149613];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_149616 = ((double *) mem_153984)[i_152104 * (int64_t) 729 + i_152097 * (int64_t) 27 + i_149613];
                    
                    // futhark/microgpt.fut:307:76-131
                    
                    double zt_res_149617 = zt_lhs_149615 * zt_rhs_149616;
                    
                    // futhark/microgpt.fut:307:102-186
                    
                    double zt_res_149618 = zs_res_149611 * zt_res_149617;
                    
                    // futhark/microgpt.fut:307:68-186
                    
                    double neg_res_149619 = -zt_res_149618;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_149620 = r_149614 + neg_res_149619;
                    double r_tmp_155524 = zp_res_149620;
                    
                    r_149614 = r_tmp_155524;
                }
                defunc_0_lifted_lambda_res_149612 = r_149614;
                ((double *) mem_154021)[i_152097] = defunc_0_lifted_lambda_res_149612;
                ((double *) mem_154022)[i_152097] = defunc_0_lifted_lambda_res_149598;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154011, i_152104 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154021, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154012, i_152104 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154022, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152117 = 0; i_152117 < (int64_t) 16; i_152117++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152113 = 0; i_152113 < (int64_t) 27; i_152113++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_138831 = ((double *) mem_154012)[i_152117 * (int64_t) 27 + i_152113];
                
                // futhark/microgpt.fut:308:92-119
                
                double zs_res_138832 = 1.0 / zs_rhs_138831;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_138833 = ((double *) mem_154011)[i_152117 * (int64_t) 27 + i_152113];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_152109 = 0; i_152109 < (int64_t) 27; i_152109++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_138840 = ((double *) mem_153915)[i_152117 * (int64_t) 729 + i_152113 * (int64_t) 27 + i_152109];
                    
                    // futhark/microgpt.fut:308:59-119
                    
                    double zt_res_138841 = zs_res_138832 * zt_lhs_138840;
                    
                    // futhark/microgpt.fut:308:87-145
                    
                    double zp_res_138842 = zp_rhs_138833 + zt_res_138841;
                    
                    ((double *) mem_154054)[i_152109] = zp_res_138842;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154049, i_152113 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154054, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154043, i_152117 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_154049, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152125 = 0; i_152125 < (int64_t) 16; i_152125++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152121 = 0; i_152121 < (int64_t) 27; i_152121++) {
                double f_elem_138855 = ((double *) mem_153916)[i_152125 * (int64_t) 27 + i_152121];
                
                // futhark/microgpt.fut:309:110-135
                
                double neg_res_138860 = -f_elem_138855;
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_138861;
                double r_138863 = 0.0;
                
                for (int64_t i_138862 = 0; i_138862 < (int64_t) 27; i_138862++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zp_lhs_138864 = ((double *) mem_153897)[i_152125 * (int64_t) 27 + i_138862];
                    
                    // futhark/microgpt.fut:309:85-135
                    
                    double zp_res_138865 = neg_res_138860 + zp_lhs_138864;
                    
                    // futhark/microgpt.fut:309:78-135
                    
                    double exp_res_138866 = futrts_exp64(zp_res_138865);
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_138867 = ((double *) mem_154043)[i_152125 * (int64_t) 729 + i_152121 * (int64_t) 27 + i_138862];
                    
                    // futhark/microgpt.fut:309:78-170
                    
                    double zt_res_138868 = exp_res_138866 * zt_rhs_138867;
                    
                    // futhark/microgpt.fut:309:70-170
                    
                    double neg_res_138869 = -zt_res_138868;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_138870 = r_138863 + neg_res_138869;
                    double r_tmp_155530 = zp_res_138870;
                    
                    r_138863 = r_tmp_155530;
                }
                defunc_0_lifted_lambda_res_138861 = r_138863;
                ((double *) mem_154075)[i_152121] = defunc_0_lifted_lambda_res_138861;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154070, i_152125 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154075, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152133 = 0; i_152133 < (int64_t) 16; i_152133++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152129 = 0; i_152129 < (int64_t) 27; i_152129++) {
                double f_elem_138931 = ((double *) mem_153897)[i_152133 * (int64_t) 27 + i_152129];
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_138936;
                double r_138938 = 0.0;
                
                for (int64_t i_138937 = 0; i_138937 < (int64_t) 27; i_138937++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double neg_arg0_138939 = ((double *) mem_153916)[i_152133 * (int64_t) 27 + i_138937];
                    
                    // futhark/microgpt.fut:312:89-113
                    
                    double neg_res_138940 = -neg_arg0_138939;
                    
                    // futhark/microgpt.fut:312:66-113
                    
                    double zp_res_138941 = f_elem_138931 + neg_res_138940;
                    
                    // futhark/microgpt.fut:312:59-113
                    
                    double exp_res_138942 = futrts_exp64(zp_res_138941);
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_138943 = ((double *) mem_154043)[i_152133 * (int64_t) 729 + i_138937 * (int64_t) 27 + i_152129];
                    
                    // futhark/microgpt.fut:312:59-146
                    
                    double zt_res_138944 = exp_res_138942 * zt_rhs_138943;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_138945 = r_138938 + zt_res_138944;
                    double r_tmp_155533 = zp_res_138945;
                    
                    r_138938 = r_tmp_155533;
                }
                defunc_0_lifted_lambda_res_138936 = r_138938;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_138946;
                double r_138948 = 0.0;
                
                for (int64_t i_138947 = 0; i_138947 < (int64_t) 27; i_138947++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_138949 = ((double *) mem_154070)[i_152133 * (int64_t) 27 + i_138947];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double neg_arg0_138950 = ((double *) mem_153914)[i_152133 * (int64_t) 27 + i_138947];
                    
                    // futhark/microgpt.fut:312:260-284
                    
                    double neg_res_138951 = -neg_arg0_138950;
                    
                    // futhark/microgpt.fut:312:237-284
                    
                    double zp_res_138952 = f_elem_138931 + neg_res_138951;
                    
                    // futhark/microgpt.fut:312:230-284
                    
                    double neg_res_138953 = -zp_res_138952;
                    
                    // futhark/microgpt.fut:100:42-54
                    
                    double max_res_138954 = fmax64(0.0, neg_res_138953);
                    
                    // futhark/microgpt.fut:100:35-54
                    
                    double sgn_res_138955 = fsignum64(max_res_138954);
                    
                    // futhark/microgpt.fut:312:211-287
                    
                    double neg_res_138956 = -sgn_res_138955;
                    
                    // futhark/microgpt.fut:312:202-288
                    
                    double zp_res_138957 = 1.0 + neg_res_138956;
                    
                    // futhark/microgpt.fut:312:178-288
                    
                    double zt_res_138958 = zt_lhs_138949 * zp_res_138957;
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_138959 = ((double *) mem_153913)[i_152133 * (int64_t) 27 + i_138947];
                    
                    // futhark/microgpt.fut:312:197-314
                    
                    double zt_res_138960 = zt_res_138958 * zt_rhs_138959;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_138961 = r_138948 + zt_res_138960;
                    double r_tmp_155534 = zp_res_138961;
                    
                    r_138948 = r_tmp_155534;
                }
                defunc_0_lifted_lambda_res_138946 = r_138948;
                // futhark/microgpt.fut:312:36-316
                
                double zp_res_138962 = defunc_0_lifted_lambda_res_138936 + defunc_0_lifted_lambda_res_138946;
                
                ((double *) mem_154091)[i_152129] = zp_res_138962;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154086, i_152133 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154091, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152141 = 0; i_152141 < (int64_t) 16; i_152141++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152137 = 0; i_152137 < (int64_t) 16; i_152137++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_138977;
                double r_138979 = 0.0;
                
                for (int64_t i_138978 = 0; i_138978 < (int64_t) 27; i_138978++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_138980 = ((double *) mem_154086)[i_152141 * (int64_t) 27 + i_138978];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_138981 = ((double *) mem_param_152846.mem)[i_138978 * (int64_t) 16 + i_152137];
                    
                    // futhark/microgpt.fut:313:67-111
                    
                    double zt_res_138982 = zt_lhs_138980 * zt_rhs_138981;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_138983 = r_138979 + zt_res_138982;
                    double r_tmp_155537 = zp_res_138983;
                    
                    r_138979 = r_tmp_155537;
                }
                defunc_0_lifted_lambda_res_138977 = r_138979;
                ((double *) mem_154107)[i_152137] = defunc_0_lifted_lambda_res_138977;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154102, i_152141 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154107, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152154 = 0; i_152154 < (int64_t) 16; i_152154++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152147 = 0; i_152147 < (int64_t) 64; i_152147++) {
                // futhark/microgpt.fut:4:11-25
                
                double indicatorp_arg0_149645 = ((double *) mem_153849)[i_152154 * (int64_t) 64 + i_152147];
                
                // futhark/microgpt.fut:100:42-54
                
                double max_res_149646 = fmax64(0.0, indicatorp_arg0_149645);
                
                // futhark/microgpt.fut:100:35-54
                
                double sgn_res_149647 = fsignum64(max_res_149646);
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_149648;
                double r_149650 = 0.0;
                
                for (int64_t i_149649 = 0; i_149649 < (int64_t) 16; i_149649++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_149651 = ((double *) mem_154102)[i_152154 * (int64_t) 16 + i_149649];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_149652 = ((double *) mem_param_152814.mem)[i_149649 * (int64_t) 64 + i_152147];
                    
                    // futhark/microgpt.fut:314:105-151
                    
                    double zt_res_149653 = zt_lhs_149651 * zt_rhs_149652;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_149654 = r_149650 + zt_res_149653;
                    double r_tmp_155542 = zp_res_149654;
                    
                    r_149650 = r_tmp_155542;
                }
                defunc_0_lifted_lambda_res_149648 = r_149650;
                // futhark/microgpt.fut:314:46-153
                
                double zt_res_149655 = sgn_res_149647 * defunc_0_lifted_lambda_res_149648;
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_149662;
                double r_149664 = 0.0;
                
                for (int64_t i_149663 = 0; i_149663 < (int64_t) 16; i_149663++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_149665 = ((double *) mem_154102)[i_149663 * (int64_t) 16 + i_152154];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_149666 = ((double *) mem_153865)[i_149663 * (int64_t) 64 + i_152147];
                    
                    // futhark/microgpt.fut:396:69-113
                    
                    double zt_res_149667 = zt_lhs_149665 * zt_rhs_149666;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_149668 = r_149664 + zt_res_149667;
                    double r_tmp_155543 = zp_res_149668;
                    
                    r_149664 = r_tmp_155543;
                }
                defunc_0_lifted_lambda_res_149662 = r_149664;
                ((double *) mem_154128)[i_152147] = defunc_0_lifted_lambda_res_149662;
                ((double *) mem_154129)[i_152147] = zt_res_149655;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154118, i_152154 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154128, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154119, i_152154 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154129, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152163 = 0; i_152163 < (int64_t) 16; i_152163++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152159 = 0; i_152159 < (int64_t) 16; i_152159++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_139047;
                double r_139049 = 0.0;
                
                for (int64_t i_139048 = 0; i_139048 < (int64_t) 64; i_139048++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_139050 = ((double *) mem_154119)[i_152163 * (int64_t) 64 + i_139048];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_139051 = ((double *) mem_param_152838.mem)[i_139048 * (int64_t) 16 + i_152159];
                    
                    // futhark/microgpt.fut:317:71-115
                    
                    double zt_res_139052 = zt_lhs_139050 * zt_rhs_139051;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_139053 = r_139049 + zt_res_139052;
                    double r_tmp_155546 = zp_res_139053;
                    
                    r_139049 = r_tmp_155546;
                }
                defunc_0_lifted_lambda_res_139047 = r_139049;
                ((double *) mem_154155)[i_152159] = defunc_0_lifted_lambda_res_139047;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154150, i_152163 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154155, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152169 = 0; i_152169 < (int64_t) 16; i_152169++) {
            // futhark/microgpt.fut:316:47-59
            
            double zp_lhs_141757 = ((double *) mem_153796)[i_152169];
            
            // futhark/microgpt.fut:316:47-87
            
            double zp_res_141758 = 1.0e-5 + zp_lhs_141757;
            
            // futhark/microgpt.fut:316:39-87
            
            double sqrt_res_141759 = futrts_sqrt64(zp_res_141758);
            
            // futhark/microgpt.fut:318:129-158
            
            double zt_res_141767 = sqrt_res_141759 * sqrt_res_141759;
            
            // futhark/microgpt.fut:318:120-158
            
            double zs_res_141768 = 1.0 / zt_res_141767;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_141769;
            double r_141771 = 0.0;
            
            for (int64_t i_141770 = 0; i_141770 < (int64_t) 16; i_141770++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_141772 = ((double *) mem_154150)[i_152169 * (int64_t) 16 + i_141770];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_141773 = ((double *) mem_153780)[i_152169 * (int64_t) 16 + i_141770];
                
                // futhark/microgpt.fut:318:69-113
                
                double zt_res_141774 = zt_lhs_141772 * zt_rhs_141773;
                
                // futhark/microgpt.fut:318:90-158
                
                double zt_res_141775 = zs_res_141768 * zt_res_141774;
                
                // futhark/microgpt.fut:318:61-158
                
                double neg_res_141776 = -zt_res_141775;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_141777 = r_141771 + neg_res_141776;
                double r_tmp_155549 = zp_res_141777;
                
                r_141771 = r_tmp_155549;
            }
            defunc_0_lifted_lambda_res_141769 = r_141771;
            ((double *) mem_154166)[i_152169] = defunc_0_lifted_lambda_res_141769;
            ((double *) mem_154167)[i_152169] = sqrt_res_141759;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152174 = 0; i_152174 < (int64_t) 16; i_152174++) {
            // futhark/microgpt.fut:319:39-51
            
            double zt_lhs_139081 = ((double *) mem_154166)[i_152174];
            
            // futhark/microgpt.fut:319:93-105
            
            double zp_lhs_139082 = ((double *) mem_153796)[i_152174];
            
            // futhark/microgpt.fut:319:93-133
            
            double zp_res_139083 = 1.0e-5 + zp_lhs_139082;
            
            // futhark/microgpt.fut:319:85-133
            
            double sqrt_res_139084 = futrts_sqrt64(zp_res_139083);
            
            // futhark/microgpt.fut:319:71-135
            
            double zt_res_139085 = 2.0 * sqrt_res_139084;
            
            // futhark/microgpt.fut:319:57-135
            
            double zs_res_139086 = 1.0 / zt_res_139085;
            
            // futhark/microgpt.fut:319:39-135
            
            double zt_res_139087 = zt_lhs_139081 * zs_res_139086;
            
            ((double *) mem_154180)[i_152174] = zt_res_139087;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152182 = 0; i_152182 < (int64_t) 16; i_152182++) {
            // futhark/microgpt.fut:320:98-110
            
            double zs_rhs_139095 = ((double *) mem_154167)[i_152182];
            
            // futhark/microgpt.fut:320:90-110
            
            double zs_res_139096 = 1.0 / zs_rhs_139095;
            
            // futhark/microgpt.fut:320:120-132
            
            double zs_lhs_139097 = ((double *) mem_154180)[i_152182];
            
            // futhark/microgpt.fut:320:120-147
            
            double zs_res_139098 = zs_lhs_139097 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152178 = 0; i_152178 < (int64_t) 16; i_152178++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_139105 = ((double *) mem_154102)[i_152182 * (int64_t) 16 + i_152178];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_139106 = ((double *) mem_154150)[i_152182 * (int64_t) 16 + i_152178];
                
                // futhark/microgpt.fut:320:64-110
                
                double zt_res_139107 = zs_res_139096 * zt_lhs_139106;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_139108 = ((double *) mem_153780)[i_152182 * (int64_t) 16 + i_152178];
                
                // futhark/microgpt.fut:320:133-172
                
                double zt_res_139109 = zs_res_139098 * zt_rhs_139108;
                
                // futhark/microgpt.fut:320:149-232
                
                double zp_res_139110 = zt_res_139109 + zt_res_139109;
                
                // futhark/microgpt.fut:320:85-232
                
                double zp_res_139111 = zt_res_139107 + zp_res_139110;
                
                // futhark/microgpt.fut:320:37-232
                
                double zp_res_139112 = zp_lhs_139105 + zp_res_139111;
                
                ((double *) mem_154192)[i_152178] = zp_res_139112;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154187, i_152182 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154192, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152194 = 0; i_152194 < (int64_t) 4; i_152194++) {
            // futhark/microgpt.fut:321:122-125
            
            int64_t zp_lhs_139117 = mul64((int64_t) 4, i_152194);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152190 = 0; i_152190 < (int64_t) 16; i_152190++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_152186 = 0; i_152186 < (int64_t) 4; i_152186++) {
                    // futhark/microgpt.fut:321:127-135
                    
                    int64_t zt_rhs_139126 = add64(zp_lhs_139117, i_152186);
                    
                    // futhark/microgpt.fut:321:100-137
                    
                    bool x_139127 = sle64((int64_t) 0, zt_rhs_139126);
                    
                    // futhark/microgpt.fut:321:100-137
                    
                    bool y_139128 = slt64(zt_rhs_139126, (int64_t) 16);
                    
                    // futhark/microgpt.fut:321:100-137
                    
                    bool bounds_check_139129 = x_139127 && y_139128;
                    
                    // futhark/microgpt.fut:321:100-137
                    
                    bool index_certs_139130;
                    
                    if (!bounds_check_139129) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_rhs_139126, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:321:100-137\n   #1  futhark/microgpt.fut:61:46-49\n   #2  futhark/microgpt.fut:321:53-139\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:15:29-44\n   #8  futhark/microgpt.fut:4:11-25\n   #9  futhark/microgpt.fut:15:15-45\n   #10 futhark/microgpt.fut:321:13-141\n   #11 futhark/microgpt.fut:547:5-76\n   #12 futhark/microgpt.fut:551:26-557:31\n   #13 futhark/microgpt.fut:590:11-57\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:61:13-49
                    
                    double defunc_0_lifted_lambda_res_139131;
                    double r_139133 = 0.0;
                    
                    for (int64_t i_139132 = 0; i_139132 < (int64_t) 16; i_139132++) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_139134 = ((double *) mem_154187)[i_152190 * (int64_t) 16 + i_139132];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_139135 = ((double *) mem_param_152822.mem)[i_139132 * (int64_t) 16 + zt_rhs_139126];
                        
                        // futhark/microgpt.fut:321:75-137
                        
                        double zt_res_139136 = zt_lhs_139134 * zt_rhs_139135;
                        
                        // futhark/microgpt.fut:61:40-49
                        
                        double zp_res_139137 = r_139133 + zt_res_139136;
                        double r_tmp_155556 = zp_res_139137;
                        
                        r_139133 = r_tmp_155556;
                    }
                    defunc_0_lifted_lambda_res_139131 = r_139133;
                    ((double *) mem_154214)[i_152186] = defunc_0_lifted_lambda_res_139131;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154209, i_152190 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154214, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154203, i_152194 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_154209, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152219 = 0; i_152219 < (int64_t) 4; i_152219++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152209 = 0; i_152209 < (int64_t) 16; i_152209++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_152200 = 0; i_152200 < (int64_t) 4; i_152200++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_149825 = ((double *) mem_154203)[i_152219 * (int64_t) 64 + i_152209 * (int64_t) 4 + i_152200];
                    
                    ((double *) mem_154263)[i_152200] = lifted_lambda_res_149825;
                    ((double *) mem_154264)[i_152200] = lifted_lambda_res_149825;
                }
                // futhark/microgpt.fut:4:11-25
                // futhark/microgpt.fut:4:11-25
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154250, i_152209 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154264, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154248, i_152209 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154263, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154249, i_152209 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154264, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154230, i_152219 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_154248, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154231, i_152219 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_154249, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154232, i_152219 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_154250, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152241 = 0; i_152241 < (int64_t) 4; i_152241++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152234 = 0; i_152234 < (int64_t) 16; i_152234++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_152227 = 0; i_152227 < (int64_t) 16; i_152227++) {
                    // futhark/microgpt.fut:61:13-49
                    
                    double defunc_0_lifted_lambda_res_150155;
                    double r_150157 = 0.0;
                    
                    for (int64_t i_150156 = 0; i_150156 < (int64_t) 4; i_150156++) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_150158 = ((double *) mem_154231)[i_152241 * (int64_t) 64 + i_152234 * (int64_t) 4 + i_150156];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_150159 = ((double *) mem_153064)[i_152241 * (int64_t) 64 + i_152227 * (int64_t) 4 + i_150156];
                        
                        // futhark/microgpt.fut:334:79-139
                        
                        double zt_res_150160 = zt_lhs_150158 * zt_rhs_150159;
                        
                        // futhark/microgpt.fut:61:40-49
                        
                        double zp_res_150161 = r_150157 + zt_res_150160;
                        double r_tmp_155571 = zp_res_150161;
                        
                        r_150157 = r_tmp_155571;
                    }
                    defunc_0_lifted_lambda_res_150155 = r_150157;
                    // futhark/microgpt.fut:61:13-49
                    
                    double defunc_0_lifted_lambda_res_150168;
                    double r_150170 = 0.0;
                    
                    for (int64_t i_150169 = 0; i_150169 < (int64_t) 4; i_150169++) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_lhs_150171 = ((double *) mem_154230)[i_152241 * (int64_t) 64 + i_152234 * (int64_t) 4 + i_150169];
                        
                        // futhark/microgpt.fut:61:46-49
                        
                        double zt_rhs_150172 = ((double *) mem_153064)[i_152241 * (int64_t) 64 + i_152227 * (int64_t) 4 + i_150169];
                        
                        // futhark/microgpt.fut:350:79-139
                        
                        double zt_res_150173 = zt_lhs_150171 * zt_rhs_150172;
                        
                        // futhark/microgpt.fut:61:40-49
                        
                        double zp_res_150174 = r_150170 + zt_res_150173;
                        double r_tmp_155572 = zp_res_150174;
                        
                        r_150170 = r_tmp_155572;
                    }
                    defunc_0_lifted_lambda_res_150168 = r_150170;
                    ((double *) mem_154327)[i_152227] = defunc_0_lifted_lambda_res_150168;
                    ((double *) mem_154328)[i_152227] = defunc_0_lifted_lambda_res_150155;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154317, i_152234 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154327, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154318, i_152234 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154328, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154305, i_152241 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154317, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154306, i_152241 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154318, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152262 = 0; i_152262 < (int64_t) 4; i_152262++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152255 = 0; i_152255 < (int64_t) 16; i_152255++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_152248 = 0; i_152248 < (int64_t) 16; i_152248++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_150407 = ((double *) mem_154306)[i_152262 * (int64_t) 256 + i_152255 * (int64_t) 16 + i_152248];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_150414 = ((double *) mem_154305)[i_152262 * (int64_t) 256 + i_152255 * (int64_t) 16 + i_152248];
                    
                    ((double *) mem_154381)[i_152248] = lifted_lambda_res_150414;
                    ((double *) mem_154382)[i_152248] = lifted_lambda_res_150407;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154371, i_152255 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154381, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154372, i_152255 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154382, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154359, i_152262 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154371, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154360, i_152262 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154372, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152286 = 0; i_152286 < (int64_t) 4; i_152286++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152273 = 0; i_152273 < (int64_t) 16; i_152273++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_150287;
                double r_150289 = 0.0;
                
                for (int64_t i_150288 = 0; i_150288 < (int64_t) 16; i_150288++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double lifted_lambda_res_150290 = ((double *) mem_153490)[i_152286 * (int64_t) 256 + i_152273 * (int64_t) 16 + i_150288];
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_150291 = r_150289 + lifted_lambda_res_150290;
                    double r_tmp_155587 = zp_res_150291;
                    
                    r_150289 = r_tmp_155587;
                }
                defunc_0_lifted_lambda_res_150287 = r_150289;
                // futhark/microgpt.fut:339:155-200
                
                double zt_res_150299 = defunc_0_lifted_lambda_res_150287 * defunc_0_lifted_lambda_res_150287;
                
                // futhark/microgpt.fut:339:146-200
                
                double zs_res_150300 = 1.0 / zt_res_150299;
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_150301;
                double r_150303 = 0.0;
                
                for (int64_t i_150302 = 0; i_150302 < (int64_t) 16; i_150302++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_150304 = ((double *) mem_154360)[i_152286 * (int64_t) 256 + i_152273 * (int64_t) 16 + i_150302];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_150305 = ((double *) mem_153490)[i_152286 * (int64_t) 256 + i_152273 * (int64_t) 16 + i_150302];
                    
                    // futhark/microgpt.fut:339:78-139
                    
                    double zt_res_150306 = zt_lhs_150304 * zt_rhs_150305;
                    
                    // futhark/microgpt.fut:339:107-200
                    
                    double zt_res_150307 = zs_res_150300 * zt_res_150306;
                    
                    // futhark/microgpt.fut:339:70-200
                    
                    double neg_res_150308 = -zt_res_150307;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_150309 = r_150303 + neg_res_150308;
                    double r_tmp_155588 = zp_res_150309;
                    
                    r_150303 = r_tmp_155588;
                }
                defunc_0_lifted_lambda_res_150301 = r_150303;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_150320;
                double r_150322 = 0.0;
                
                for (int64_t i_150321 = 0; i_150321 < (int64_t) 16; i_150321++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double lifted_lambda_res_150323 = ((double *) mem_153489)[i_152286 * (int64_t) 256 + i_152273 * (int64_t) 16 + i_150321];
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_150324 = r_150322 + lifted_lambda_res_150323;
                    double r_tmp_155589 = zp_res_150324;
                    
                    r_150322 = r_tmp_155589;
                }
                defunc_0_lifted_lambda_res_150320 = r_150322;
                // futhark/microgpt.fut:355:155-200
                
                double zt_res_150332 = defunc_0_lifted_lambda_res_150320 * defunc_0_lifted_lambda_res_150320;
                
                // futhark/microgpt.fut:355:146-200
                
                double zs_res_150333 = 1.0 / zt_res_150332;
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_150334;
                double r_150336 = 0.0;
                
                for (int64_t i_150335 = 0; i_150335 < (int64_t) 16; i_150335++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_150337 = ((double *) mem_154359)[i_152286 * (int64_t) 256 + i_152273 * (int64_t) 16 + i_150335];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_150338 = ((double *) mem_153489)[i_152286 * (int64_t) 256 + i_152273 * (int64_t) 16 + i_150335];
                    
                    // futhark/microgpt.fut:355:78-139
                    
                    double zt_res_150339 = zt_lhs_150337 * zt_rhs_150338;
                    
                    // futhark/microgpt.fut:355:107-200
                    
                    double zt_res_150340 = zs_res_150333 * zt_res_150339;
                    
                    // futhark/microgpt.fut:355:70-200
                    
                    double neg_res_150341 = -zt_res_150340;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_150342 = r_150336 + neg_res_150341;
                    double r_tmp_155590 = zp_res_150342;
                    
                    r_150336 = r_tmp_155590;
                }
                defunc_0_lifted_lambda_res_150334 = r_150336;
                ((double *) mem_154433)[i_152273] = defunc_0_lifted_lambda_res_150334;
                ((double *) mem_154434)[i_152273] = defunc_0_lifted_lambda_res_150320;
                ((double *) mem_154435)[i_152273] = defunc_0_lifted_lambda_res_150301;
                ((double *) mem_154436)[i_152273] = defunc_0_lifted_lambda_res_150287;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154413, i_152286 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154433, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154414, i_152286 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154434, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154415, i_152286 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154435, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154416, i_152286 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154436, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152309 = 0; i_152309 < (int64_t) 4; i_152309++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152302 = 0; i_152302 < (int64_t) 16; i_152302++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_150438 = ((double *) mem_154416)[i_152309 * (int64_t) 16 + i_152302];
                
                // futhark/microgpt.fut:340:93-121
                
                double zs_res_150439 = 1.0 / zs_rhs_150438;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_150440 = ((double *) mem_154415)[i_152309 * (int64_t) 16 + i_152302];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_150459 = ((double *) mem_154413)[i_152309 * (int64_t) 16 + i_152302];
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_150457 = ((double *) mem_154414)[i_152309 * (int64_t) 16 + i_152302];
                
                // futhark/microgpt.fut:356:93-121
                
                double zs_res_150458 = 1.0 / zs_rhs_150457;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_152295 = 0; i_152295 < (int64_t) 16; i_152295++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_150487 = ((double *) mem_154360)[i_152309 * (int64_t) 256 + i_152302 * (int64_t) 16 + i_152295];
                    
                    // futhark/microgpt.fut:340:59-121
                    
                    double zt_res_150488 = zs_res_150439 * zt_lhs_150487;
                    
                    // futhark/microgpt.fut:340:88-148
                    
                    double zp_res_150489 = zp_rhs_150440 + zt_res_150488;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_150496 = ((double *) mem_154359)[i_152309 * (int64_t) 256 + i_152302 * (int64_t) 16 + i_152295];
                    
                    // futhark/microgpt.fut:356:59-121
                    
                    double zt_res_150497 = zs_res_150458 * zt_lhs_150496;
                    
                    // futhark/microgpt.fut:356:88-148
                    
                    double zp_res_150498 = zp_rhs_150459 + zt_res_150497;
                    
                    ((double *) mem_154499)[i_152295] = zp_res_150498;
                    ((double *) mem_154500)[i_152295] = zp_res_150489;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154489, i_152302 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154499, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154490, i_152302 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154500, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154477, i_152309 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154489, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154478, i_152309 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154490, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152323 = 0; i_152323 < (int64_t) 4; i_152323++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152316 = 0; i_152316 < (int64_t) 16; i_152316++) {
                double f_elem_150518 = ((double *) mem_153366)[i_152323 * (int64_t) 16 + i_152316];
                double f_elem_150520 = ((double *) mem_153363)[i_152323 * (int64_t) 16 + i_152316];
                
                // futhark/microgpt.fut:341:119-145
                
                double neg_res_150525 = -f_elem_150518;
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_150526;
                double r_150528 = 0.0;
                
                for (int64_t i_150527 = 0; i_150527 < (int64_t) 16; i_150527++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zp_lhs_150529 = ((double *) mem_153254)[i_152323 * (int64_t) 256 + i_152316 * (int64_t) 16 + i_150527];
                    
                    // futhark/microgpt.fut:341:85-145
                    
                    double zp_res_150530 = neg_res_150525 + zp_lhs_150529;
                    
                    // futhark/microgpt.fut:341:78-145
                    
                    double exp_res_150531 = futrts_exp64(zp_res_150530);
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_150532 = ((double *) mem_154478)[i_152323 * (int64_t) 256 + i_152316 * (int64_t) 16 + i_150527];
                    
                    // futhark/microgpt.fut:341:78-181
                    
                    double zt_res_150533 = exp_res_150531 * zt_rhs_150532;
                    
                    // futhark/microgpt.fut:341:70-181
                    
                    double neg_res_150534 = -zt_res_150533;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_150535 = r_150528 + neg_res_150534;
                    double r_tmp_155601 = zp_res_150535;
                    
                    r_150528 = r_tmp_155601;
                }
                defunc_0_lifted_lambda_res_150526 = r_150528;
                // futhark/microgpt.fut:357:119-145
                
                double neg_res_150543 = -f_elem_150520;
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_150544;
                double r_150546 = 0.0;
                
                for (int64_t i_150545 = 0; i_150545 < (int64_t) 16; i_150545++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zp_lhs_150547 = ((double *) mem_153253)[i_152323 * (int64_t) 256 + i_152316 * (int64_t) 16 + i_150545];
                    
                    // futhark/microgpt.fut:357:85-145
                    
                    double zp_res_150548 = neg_res_150543 + zp_lhs_150547;
                    
                    // futhark/microgpt.fut:357:78-145
                    
                    double exp_res_150549 = futrts_exp64(zp_res_150548);
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_150550 = ((double *) mem_154477)[i_152323 * (int64_t) 256 + i_152316 * (int64_t) 16 + i_150545];
                    
                    // futhark/microgpt.fut:357:78-181
                    
                    double zt_res_150551 = exp_res_150549 * zt_rhs_150550;
                    
                    // futhark/microgpt.fut:357:70-181
                    
                    double neg_res_150552 = -zt_res_150551;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_150553 = r_150546 + neg_res_150552;
                    double r_tmp_155602 = zp_res_150553;
                    
                    r_150546 = r_tmp_155602;
                }
                defunc_0_lifted_lambda_res_150544 = r_150546;
                ((double *) mem_154541)[i_152316] = defunc_0_lifted_lambda_res_150544;
                ((double *) mem_154542)[i_152316] = defunc_0_lifted_lambda_res_150526;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154531, i_152323 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154541, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154532, i_152323 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154542, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152344 = 0; i_152344 < (int64_t) 4; i_152344++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152337 = 0; i_152337 < (int64_t) 16; i_152337++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_150573 = ((double *) mem_153366)[i_152344 * (int64_t) 16 + i_152337];
                
                // futhark/microgpt.fut:344:101-127
                
                double neg_res_150574 = -neg_arg0_150573;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_150575 = ((double *) mem_154532)[i_152344 * (int64_t) 16 + i_152337];
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_150576 = ((double *) mem_153365)[i_152344 * (int64_t) 16 + i_152337];
                
                // futhark/microgpt.fut:344:266-292
                
                double neg_res_150577 = -neg_arg0_150576;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_150578 = ((double *) mem_153364)[i_152344 * (int64_t) 16 + i_152337];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_150611 = ((double *) mem_153361)[i_152344 * (int64_t) 16 + i_152337];
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_150609 = ((double *) mem_153362)[i_152344 * (int64_t) 16 + i_152337];
                
                // futhark/microgpt.fut:360:266-292
                
                double neg_res_150610 = -neg_arg0_150609;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_150608 = ((double *) mem_154531)[i_152344 * (int64_t) 16 + i_152337];
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_150606 = ((double *) mem_153363)[i_152344 * (int64_t) 16 + i_152337];
                
                // futhark/microgpt.fut:360:101-127
                
                double neg_res_150607 = -neg_arg0_150606;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_152330 = 0; i_152330 < (int64_t) 16; i_152330++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_150650 = ((double *) mem_153254)[i_152344 * (int64_t) 256 + i_152337 * (int64_t) 16 + i_152330];
                    
                    // futhark/microgpt.fut:344:67-127
                    
                    double zp_res_150651 = neg_res_150574 + zp_lhs_150650;
                    
                    // futhark/microgpt.fut:344:60-127
                    
                    double exp_res_150652 = futrts_exp64(zp_res_150651);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_150653 = ((double *) mem_154478)[i_152344 * (int64_t) 256 + i_152337 * (int64_t) 16 + i_152330];
                    
                    // futhark/microgpt.fut:344:60-163
                    
                    double zt_res_150654 = exp_res_150652 * zt_rhs_150653;
                    
                    // futhark/microgpt.fut:344:232-292
                    
                    double zp_res_150655 = neg_res_150577 + zp_lhs_150650;
                    
                    // futhark/microgpt.fut:344:225-292
                    
                    double neg_res_150656 = -zp_res_150655;
                    
                    // futhark/microgpt.fut:100:42-54
                    
                    double max_res_150657 = fmax64(0.0, neg_res_150656);
                    
                    // futhark/microgpt.fut:100:35-54
                    
                    double sgn_res_150658 = fsignum64(max_res_150657);
                    
                    // futhark/microgpt.fut:344:206-295
                    
                    double neg_res_150659 = -sgn_res_150658;
                    
                    // futhark/microgpt.fut:344:197-296
                    
                    double zp_res_150660 = 1.0 + neg_res_150659;
                    
                    // futhark/microgpt.fut:344:171-296
                    
                    double zt_res_150661 = zt_lhs_150575 * zp_res_150660;
                    
                    // futhark/microgpt.fut:344:192-324
                    
                    double zt_res_150662 = zt_rhs_150578 * zt_res_150661;
                    
                    // futhark/microgpt.fut:344:131-324
                    
                    double zp_res_150663 = zt_res_150654 + zt_res_150662;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_150670 = ((double *) mem_153253)[i_152344 * (int64_t) 256 + i_152337 * (int64_t) 16 + i_152330];
                    
                    // futhark/microgpt.fut:360:67-127
                    
                    double zp_res_150671 = neg_res_150607 + zp_lhs_150670;
                    
                    // futhark/microgpt.fut:360:60-127
                    
                    double exp_res_150672 = futrts_exp64(zp_res_150671);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_150673 = ((double *) mem_154477)[i_152344 * (int64_t) 256 + i_152337 * (int64_t) 16 + i_152330];
                    
                    // futhark/microgpt.fut:360:60-163
                    
                    double zt_res_150674 = exp_res_150672 * zt_rhs_150673;
                    
                    // futhark/microgpt.fut:360:232-292
                    
                    double zp_res_150675 = neg_res_150610 + zp_lhs_150670;
                    
                    // futhark/microgpt.fut:360:225-292
                    
                    double neg_res_150676 = -zp_res_150675;
                    
                    // futhark/microgpt.fut:100:42-54
                    
                    double max_res_150677 = fmax64(0.0, neg_res_150676);
                    
                    // futhark/microgpt.fut:100:35-54
                    
                    double sgn_res_150678 = fsignum64(max_res_150677);
                    
                    // futhark/microgpt.fut:360:206-295
                    
                    double neg_res_150679 = -sgn_res_150678;
                    
                    // futhark/microgpt.fut:360:197-296
                    
                    double zp_res_150680 = 1.0 + neg_res_150679;
                    
                    // futhark/microgpt.fut:360:171-296
                    
                    double zt_res_150681 = zt_lhs_150608 * zp_res_150680;
                    
                    // futhark/microgpt.fut:360:192-324
                    
                    double zt_res_150682 = zt_rhs_150611 * zt_res_150681;
                    
                    // futhark/microgpt.fut:360:131-324
                    
                    double zp_res_150683 = zt_res_150674 + zt_res_150682;
                    
                    ((double *) mem_154585)[i_152330] = zp_res_150683;
                    ((double *) mem_154586)[i_152330] = zp_res_150663;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154575, i_152337 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154585, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154576, i_152337 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154586, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154563, i_152344 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154575, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154564, i_152344 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154576, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152365 = 0; i_152365 < (int64_t) 4; i_152365++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152358 = 0; i_152358 < (int64_t) 16; i_152358++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_152351 = 0; i_152351 < (int64_t) 16; i_152351++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_150748 = ((double *) mem_154564)[i_152365 * (int64_t) 256 + i_152358 * (int64_t) 16 + i_152351];
                    
                    // futhark/microgpt.fut:345:58-100
                    
                    double zs_res_150749 = zs_lhs_150748 / 2.0;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_150756 = ((double *) mem_154563)[i_152365 * (int64_t) 256 + i_152358 * (int64_t) 16 + i_152351];
                    
                    // futhark/microgpt.fut:361:58-100
                    
                    double zs_res_150757 = zs_lhs_150756 / 2.0;
                    
                    ((double *) mem_154639)[i_152351] = zs_res_150757;
                    ((double *) mem_154640)[i_152351] = zs_res_150749;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154629, i_152358 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154639, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154630, i_152358 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154640, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154617, i_152365 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154629, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154618, i_152365 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154630, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152389 = 0; i_152389 < (int64_t) 16; i_152389++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152376 = 0; i_152376 < (int64_t) 16; i_152376++) {
                // futhark/microgpt.fut:330:40-43
                
                int64_t zt_lhs_150005 = sdiv64(i_152376, (int64_t) 4);
                
                // futhark/microgpt.fut:330:27-45
                
                bool x_150006 = sle64((int64_t) 0, zt_lhs_150005);
                
                // futhark/microgpt.fut:330:27-45
                
                bool y_150007 = slt64(zt_lhs_150005, (int64_t) 4);
                
                // futhark/microgpt.fut:330:27-45
                
                bool bounds_check_150008 = x_150006 && y_150007;
                
                // futhark/microgpt.fut:330:27-45
                
                bool index_certs_150009;
                
                if (!bounds_check_150008) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_150005, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:330:27-45\n   #1  futhark/microgpt.fut:61:46-49\n   #2  futhark/microgpt.fut:330:5-108\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:322:13-330:118\n   #8  futhark/microgpt.fut:547:5-76\n   #9  futhark/microgpt.fut:551:26-557:31\n   #10 futhark/microgpt.fut:590:11-57\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:330:62-65
                
                int64_t zt_lhs_150010 = smod64(i_152376, (int64_t) 4);
                
                // futhark/microgpt.fut:330:27-67
                
                bool x_150011 = sle64((int64_t) 0, zt_lhs_150010);
                
                // futhark/microgpt.fut:330:27-67
                
                bool y_150012 = slt64(zt_lhs_150010, (int64_t) 4);
                
                // futhark/microgpt.fut:330:27-67
                
                bool bounds_check_150013 = x_150011 && y_150012;
                
                // futhark/microgpt.fut:330:27-67
                
                bool index_certs_150014;
                
                if (!bounds_check_150013) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_150010, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:330:27-67\n   #1  futhark/microgpt.fut:61:46-49\n   #2  futhark/microgpt.fut:330:5-108\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:322:13-330:118\n   #8  futhark/microgpt.fut:547:5-76\n   #9  futhark/microgpt.fut:551:26-557:31\n   #10 futhark/microgpt.fut:590:11-57\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_150015;
                double r_150017 = 0.0;
                
                for (int64_t i_150016 = 0; i_150016 < (int64_t) 16; i_150016++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_150018 = ((double *) mem_154232)[zt_lhs_150005 * (int64_t) 64 + i_150016 * (int64_t) 4 + zt_lhs_150010];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_150019 = ((double *) mem_153683)[zt_lhs_150005 * (int64_t) 256 + i_150016 * (int64_t) 16 + i_152389];
                    
                    // futhark/microgpt.fut:330:27-106
                    
                    double zt_res_150020 = zt_lhs_150018 * zt_rhs_150019;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_150021 = r_150017 + zt_res_150020;
                    double r_tmp_155623 = zp_res_150021;
                    
                    r_150017 = r_tmp_155623;
                }
                defunc_0_lifted_lambda_res_150015 = r_150017;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_150034;
                double r_150036 = 0.0;
                
                for (int64_t i_150035 = 0; i_150035 < (int64_t) 16; i_150035++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_150037 = ((double *) mem_154618)[zt_lhs_150005 * (int64_t) 256 + i_150035 * (int64_t) 16 + i_152389];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_150038 = ((double *) mem_153066)[zt_lhs_150005 * (int64_t) 64 + i_150035 * (int64_t) 4 + zt_lhs_150010];
                    
                    // futhark/microgpt.fut:346:27-105
                    
                    double zt_res_150039 = zt_lhs_150037 * zt_rhs_150038;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_150040 = r_150036 + zt_res_150039;
                    double r_tmp_155624 = zp_res_150040;
                    
                    r_150036 = r_tmp_155624;
                }
                defunc_0_lifted_lambda_res_150034 = r_150036;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_150056;
                double r_150058 = 0.0;
                
                for (int64_t i_150057 = 0; i_150057 < (int64_t) 16; i_150057++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_150059 = ((double *) mem_154617)[zt_lhs_150005 * (int64_t) 256 + i_152389 * (int64_t) 16 + i_150057];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_150060 = ((double *) mem_153065)[zt_lhs_150005 * (int64_t) 64 + i_150057 * (int64_t) 4 + zt_lhs_150010];
                    
                    // futhark/microgpt.fut:362:27-105
                    
                    double zt_res_150061 = zt_lhs_150059 * zt_rhs_150060;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_150062 = r_150058 + zt_res_150061;
                    double r_tmp_155625 = zp_res_150062;
                    
                    r_150058 = r_tmp_155625;
                }
                defunc_0_lifted_lambda_res_150056 = r_150058;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_150074;
                double r_150076 = 0.0;
                
                for (int64_t i_150075 = 0; i_150075 < (int64_t) 16; i_150075++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_150077 = ((double *) mem_154187)[i_150075 * (int64_t) 16 + i_152389];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_150078 = ((double *) mem_153764)[i_150075 * (int64_t) 16 + i_152376];
                    
                    // futhark/microgpt.fut:394:68-112
                    
                    double zt_res_150079 = zt_lhs_150077 * zt_rhs_150078;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_150080 = r_150076 + zt_res_150079;
                    double r_tmp_155626 = zp_res_150080;
                    
                    r_150076 = r_tmp_155626;
                }
                defunc_0_lifted_lambda_res_150074 = r_150076;
                ((double *) mem_154691)[i_152376] = defunc_0_lifted_lambda_res_150074;
                ((double *) mem_154692)[i_152376] = defunc_0_lifted_lambda_res_150056;
                ((double *) mem_154693)[i_152376] = defunc_0_lifted_lambda_res_150034;
                ((double *) mem_154694)[i_152376] = defunc_0_lifted_lambda_res_150015;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154671, i_152389 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154691, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154672, i_152389 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154692, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154673, i_152389 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154693, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154674, i_152389 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154694, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152400 = 0; i_152400 < (int64_t) 16; i_152400++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152396 = 0; i_152396 < (int64_t) 16; i_152396++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_140256;
                double r_140258 = 0.0;
                
                for (int64_t i_140257 = 0; i_140257 < (int64_t) 16; i_140257++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_140259 = ((double *) mem_154674)[i_152400 * (int64_t) 16 + i_140257];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_140260 = ((double *) mem_param_152842.mem)[i_140257 * (int64_t) 16 + i_152396];
                    
                    // futhark/microgpt.fut:365:73-118
                    
                    double zt_res_140261 = zt_lhs_140259 * zt_rhs_140260;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_140262 = r_140258 + zt_res_140261;
                    double r_tmp_155629 = zp_res_140262;
                    
                    r_140258 = r_tmp_155629;
                }
                defunc_0_lifted_lambda_res_140256 = r_140258;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_140263;
                double r_140265 = 0.0;
                
                for (int64_t i_140264 = 0; i_140264 < (int64_t) 16; i_140264++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_140266 = ((double *) mem_154673)[i_152400 * (int64_t) 16 + i_140264];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_140267 = ((double *) mem_param_152818.mem)[i_140264 * (int64_t) 16 + i_152396];
                    
                    // futhark/microgpt.fut:365:149-194
                    
                    double zt_res_140268 = zt_lhs_140266 * zt_rhs_140267;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_140269 = r_140265 + zt_res_140268;
                    double r_tmp_155630 = zp_res_140269;
                    
                    r_140265 = r_tmp_155630;
                }
                defunc_0_lifted_lambda_res_140263 = r_140265;
                // futhark/microgpt.fut:365:51-196
                
                double zp_res_140270 = defunc_0_lifted_lambda_res_140256 + defunc_0_lifted_lambda_res_140263;
                
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_140271;
                double r_140273 = 0.0;
                
                for (int64_t i_140272 = 0; i_140272 < (int64_t) 16; i_140272++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_140274 = ((double *) mem_154672)[i_152400 * (int64_t) 16 + i_140272];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_140275 = ((double *) mem_param_152830.mem)[i_140272 * (int64_t) 16 + i_152396];
                    
                    // futhark/microgpt.fut:365:226-271
                    
                    double zt_res_140276 = zt_lhs_140274 * zt_rhs_140275;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_140277 = r_140273 + zt_res_140276;
                    double r_tmp_155631 = zp_res_140277;
                    
                    r_140273 = r_tmp_155631;
                }
                defunc_0_lifted_lambda_res_140271 = r_140273;
                // futhark/microgpt.fut:365:122-273
                
                double zp_res_140278 = zp_res_140270 + defunc_0_lifted_lambda_res_140271;
                
                ((double *) mem_154740)[i_152396] = zp_res_140278;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154735, i_152400 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154740, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152422 = 0; i_152422 < (int64_t) 16; i_152422++) {
            // futhark/microgpt.fut:364:47-59
            
            double zp_lhs_145608 = ((double *) mem_153011)[i_152422];
            
            // futhark/microgpt.fut:364:47-87
            
            double zp_res_145609 = 1.0e-5 + zp_lhs_145608;
            
            // futhark/microgpt.fut:364:39-87
            
            double sqrt_res_145610 = futrts_sqrt64(zp_res_145609);
            
            // futhark/microgpt.fut:366:128-157
            
            double zt_res_145618 = sqrt_res_145610 * sqrt_res_145610;
            
            // futhark/microgpt.fut:366:119-157
            
            double zs_res_145619 = 1.0 / zt_res_145618;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_145620;
            double r_145622 = 0.0;
            
            for (int64_t i_145621 = 0; i_145621 < (int64_t) 16; i_145621++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_145623 = ((double *) mem_154735)[i_152422 * (int64_t) 16 + i_145621];
                
                // futhark/microgpt.fut:61:46-49
                
                double zt_rhs_145624 = ((double *) mem_152995)[i_152422 * (int64_t) 16 + i_145621];
                
                // futhark/microgpt.fut:366:69-112
                
                double zt_res_145625 = zt_lhs_145623 * zt_rhs_145624;
                
                // futhark/microgpt.fut:366:90-157
                
                double zt_res_145626 = zs_res_145619 * zt_res_145625;
                
                // futhark/microgpt.fut:366:61-157
                
                double neg_res_145627 = -zt_res_145626;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_145628 = r_145622 + neg_res_145627;
                double r_tmp_155637 = zp_res_145628;
                
                r_145622 = r_tmp_155637;
            }
            defunc_0_lifted_lambda_res_145620 = r_145622;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152408 = 0; i_152408 < (int64_t) 16; i_152408++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_150824;
                double r_150826 = 0.0;
                
                for (int64_t i_150825 = 0; i_150825 < (int64_t) 16; i_150825++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_150827 = ((double *) mem_154672)[i_150825 * (int64_t) 16 + i_152422];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_150828 = ((double *) mem_153048)[i_150825 * (int64_t) 16 + i_152408];
                    
                    // futhark/microgpt.fut:391:68-111
                    
                    double zt_res_150829 = zt_lhs_150827 * zt_rhs_150828;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_150830 = r_150826 + zt_res_150829;
                    double r_tmp_155641 = zp_res_150830;
                    
                    r_150826 = r_tmp_155641;
                }
                defunc_0_lifted_lambda_res_150824 = r_150826;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_150837;
                double r_150839 = 0.0;
                
                for (int64_t i_150838 = 0; i_150838 < (int64_t) 16; i_150838++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_150840 = ((double *) mem_154673)[i_150838 * (int64_t) 16 + i_152422];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_150841 = ((double *) mem_153048)[i_150838 * (int64_t) 16 + i_152408];
                    
                    // futhark/microgpt.fut:392:68-111
                    
                    double zt_res_150842 = zt_lhs_150840 * zt_rhs_150841;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_150843 = r_150839 + zt_res_150842;
                    double r_tmp_155642 = zp_res_150843;
                    
                    r_150839 = r_tmp_155642;
                }
                defunc_0_lifted_lambda_res_150837 = r_150839;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_150853;
                double r_150855 = 0.0;
                
                for (int64_t i_150854 = 0; i_150854 < (int64_t) 16; i_150854++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_150856 = ((double *) mem_154674)[i_150854 * (int64_t) 16 + i_152422];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_150857 = ((double *) mem_153048)[i_150854 * (int64_t) 16 + i_152408];
                    
                    // futhark/microgpt.fut:393:68-111
                    
                    double zt_res_150858 = zt_lhs_150856 * zt_rhs_150857;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_150859 = r_150855 + zt_res_150858;
                    double r_tmp_155643 = zp_res_150859;
                    
                    r_150855 = r_tmp_155643;
                }
                defunc_0_lifted_lambda_res_150853 = r_150855;
                ((double *) mem_154774)[i_152408] = defunc_0_lifted_lambda_res_150853;
                ((double *) mem_154775)[i_152408] = defunc_0_lifted_lambda_res_150837;
                ((double *) mem_154776)[i_152408] = defunc_0_lifted_lambda_res_150824;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154751, i_152422 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154774, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154752, i_152422 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154775, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154753, i_152422 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154776, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            ((double *) mem_154754)[i_152422] = defunc_0_lifted_lambda_res_145620;
            ((double *) mem_154755)[i_152422] = sqrt_res_145610;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152430 = 0; i_152430 < (int64_t) 16; i_152430++) {
            // futhark/microgpt.fut:367:39-51
            
            double zt_lhs_140306 = ((double *) mem_154754)[i_152430];
            
            // futhark/microgpt.fut:367:93-105
            
            double zp_lhs_140307 = ((double *) mem_153011)[i_152430];
            
            // futhark/microgpt.fut:367:93-133
            
            double zp_res_140308 = 1.0e-5 + zp_lhs_140307;
            
            // futhark/microgpt.fut:367:85-133
            
            double sqrt_res_140309 = futrts_sqrt64(zp_res_140308);
            
            // futhark/microgpt.fut:367:71-135
            
            double zt_res_140310 = 2.0 * sqrt_res_140309;
            
            // futhark/microgpt.fut:367:57-135
            
            double zs_res_140311 = 1.0 / zt_res_140310;
            
            // futhark/microgpt.fut:367:39-135
            
            double zt_res_140312 = zt_lhs_140306 * zs_res_140311;
            
            ((double *) mem_154813)[i_152430] = zt_res_140312;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152438 = 0; i_152438 < (int64_t) 16; i_152438++) {
            // futhark/microgpt.fut:368:98-110
            
            double zs_rhs_140320 = ((double *) mem_154755)[i_152438];
            
            // futhark/microgpt.fut:368:90-110
            
            double zs_res_140321 = 1.0 / zs_rhs_140320;
            
            // futhark/microgpt.fut:368:120-132
            
            double zs_lhs_140322 = ((double *) mem_154813)[i_152438];
            
            // futhark/microgpt.fut:368:120-147
            
            double zs_res_140323 = zs_lhs_140322 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152434 = 0; i_152434 < (int64_t) 16; i_152434++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_140330 = ((double *) mem_154187)[i_152438 * (int64_t) 16 + i_152434];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_140331 = ((double *) mem_154735)[i_152438 * (int64_t) 16 + i_152434];
                
                // futhark/microgpt.fut:368:64-110
                
                double zt_res_140332 = zs_res_140321 * zt_lhs_140331;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_140333 = ((double *) mem_152995)[i_152438 * (int64_t) 16 + i_152434];
                
                // futhark/microgpt.fut:368:133-171
                
                double zt_res_140334 = zs_res_140323 * zt_rhs_140333;
                
                // futhark/microgpt.fut:368:149-230
                
                double zp_res_140335 = zt_res_140334 + zt_res_140334;
                
                // futhark/microgpt.fut:368:85-230
                
                double zp_res_140336 = zt_res_140332 + zp_res_140335;
                
                // futhark/microgpt.fut:368:37-230
                
                double zp_res_140337 = zp_lhs_140330 + zp_res_140336;
                
                ((double *) mem_154825)[i_152434] = zp_res_140337;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154820, i_152438 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154825, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152451 = 0; i_152451 < (int64_t) 16; i_152451++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152444 = 0; i_152444 < (int64_t) 16; i_152444++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_150883 = ((double *) mem_154820)[i_152451 * (int64_t) 16 + i_152444];
                
                ((double *) mem_154846)[i_152444] = lifted_lambda_res_150883;
                ((double *) mem_154847)[i_152444] = lifted_lambda_res_150883;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154836, i_152451 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154846, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154837, i_152451 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154847, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152462 = 0; i_152462 < (int64_t) 16; i_152462++) {
            // futhark/microgpt.fut:386:47-59
            
            double zp_lhs_145733 = ((double *) mem_152952)[i_152462];
            
            // futhark/microgpt.fut:386:47-87
            
            double zp_res_145734 = 1.0e-5 + zp_lhs_145733;
            
            // futhark/microgpt.fut:386:39-87
            
            double sqrt_res_145735 = futrts_sqrt64(zp_res_145734);
            
            // futhark/microgpt.fut:388:156-185
            
            double zt_res_145743 = sqrt_res_145735 * sqrt_res_145735;
            
            // futhark/microgpt.fut:388:147-185
            
            double zs_res_145744 = 1.0 / zt_res_145743;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_145745;
            double r_145747 = 0.0;
            
            for (int64_t i_145746 = 0; i_145746 < (int64_t) 16; i_145746++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_145748 = ((double *) mem_154837)[i_152462 * (int64_t) 16 + i_145746];
                
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_145749 = ((double *) mem_param_152826.mem)[i_152462 * (int64_t) 16 + i_145746];
                
                // futhark/microgpt.fut:61:46-49
                
                double zp_rhs_145750 = ((double *) mem_152919)[i_152462 * (int64_t) 16 + i_145746];
                
                // futhark/microgpt.fut:388:95-139
                
                double zp_res_145751 = zp_lhs_145749 + zp_rhs_145750;
                
                // futhark/microgpt.fut:388:69-139
                
                double zt_res_145752 = zt_lhs_145748 * zp_res_145751;
                
                // futhark/microgpt.fut:388:90-185
                
                double zt_res_145753 = zs_res_145744 * zt_res_145752;
                
                // futhark/microgpt.fut:388:61-185
                
                double neg_res_145754 = -zt_res_145753;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_145755 = r_145747 + neg_res_145754;
                double r_tmp_155655 = zp_res_145755;
                
                r_145747 = r_tmp_155655;
            }
            defunc_0_lifted_lambda_res_145745 = r_145747;
            // futhark/microgpt.fut:399:47-59
            
            double zp_lhs_145766 = ((double *) mem_152951)[i_152462];
            
            // futhark/microgpt.fut:399:47-87
            
            double zp_res_145767 = 1.0e-5 + zp_lhs_145766;
            
            // futhark/microgpt.fut:399:39-87
            
            double sqrt_res_145768 = futrts_sqrt64(zp_res_145767);
            
            // futhark/microgpt.fut:401:156-185
            
            double zt_res_145776 = sqrt_res_145768 * sqrt_res_145768;
            
            // futhark/microgpt.fut:401:147-185
            
            double zs_res_145777 = 1.0 / zt_res_145776;
            
            // futhark/microgpt.fut:61:13-49
            
            double defunc_0_lifted_lambda_res_145778;
            double r_145780 = 0.0;
            
            for (int64_t i_145779 = 0; i_145779 < (int64_t) 16; i_145779++) {
                // futhark/microgpt.fut:61:46-49
                
                double zt_lhs_145781 = ((double *) mem_154836)[i_152462 * (int64_t) 16 + i_145779];
                
                // futhark/microgpt.fut:61:46-49
                
                double zp_lhs_145782 = ((double *) mem_param_152826.mem)[i_152462 * (int64_t) 16 + i_145779];
                
                // futhark/microgpt.fut:61:46-49
                
                double zp_rhs_145783 = ((double *) mem_152919)[i_152462 * (int64_t) 16 + i_145779];
                
                // futhark/microgpt.fut:401:95-139
                
                double zp_res_145784 = zp_lhs_145782 + zp_rhs_145783;
                
                // futhark/microgpt.fut:401:69-139
                
                double zt_res_145785 = zt_lhs_145781 * zp_res_145784;
                
                // futhark/microgpt.fut:401:90-185
                
                double zt_res_145786 = zs_res_145777 * zt_res_145785;
                
                // futhark/microgpt.fut:401:61-185
                
                double neg_res_145787 = -zt_res_145786;
                
                // futhark/microgpt.fut:61:40-49
                
                double zp_res_145788 = r_145780 + neg_res_145787;
                double r_tmp_155656 = zp_res_145788;
                
                r_145780 = r_tmp_155656;
            }
            defunc_0_lifted_lambda_res_145778 = r_145780;
            ((double *) mem_154868)[i_152462] = defunc_0_lifted_lambda_res_145778;
            ((double *) mem_154869)[i_152462] = sqrt_res_145768;
            ((double *) mem_154870)[i_152462] = defunc_0_lifted_lambda_res_145745;
            ((double *) mem_154871)[i_152462] = sqrt_res_145735;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152471 = 0; i_152471 < (int64_t) 16; i_152471++) {
            // futhark/microgpt.fut:389:39-51
            
            double zt_lhs_145849 = ((double *) mem_154870)[i_152471];
            
            // futhark/microgpt.fut:389:93-105
            
            double zp_lhs_145850 = ((double *) mem_152952)[i_152471];
            
            // futhark/microgpt.fut:389:93-133
            
            double zp_res_145851 = 1.0e-5 + zp_lhs_145850;
            
            // futhark/microgpt.fut:389:85-133
            
            double sqrt_res_145852 = futrts_sqrt64(zp_res_145851);
            
            // futhark/microgpt.fut:389:71-135
            
            double zt_res_145853 = 2.0 * sqrt_res_145852;
            
            // futhark/microgpt.fut:389:57-135
            
            double zs_res_145854 = 1.0 / zt_res_145853;
            
            // futhark/microgpt.fut:389:39-135
            
            double zt_res_145855 = zt_lhs_145849 * zs_res_145854;
            
            // futhark/microgpt.fut:402:39-51
            
            double zt_lhs_145862 = ((double *) mem_154868)[i_152471];
            
            // futhark/microgpt.fut:402:93-105
            
            double zp_lhs_145863 = ((double *) mem_152951)[i_152471];
            
            // futhark/microgpt.fut:402:93-133
            
            double zp_res_145864 = 1.0e-5 + zp_lhs_145863;
            
            // futhark/microgpt.fut:402:85-133
            
            double sqrt_res_145865 = futrts_sqrt64(zp_res_145864);
            
            // futhark/microgpt.fut:402:71-135
            
            double zt_res_145866 = 2.0 * sqrt_res_145865;
            
            // futhark/microgpt.fut:402:57-135
            
            double zs_res_145867 = 1.0 / zt_res_145866;
            
            // futhark/microgpt.fut:402:39-135
            
            double zt_res_145868 = zt_lhs_145862 * zs_res_145867;
            
            ((double *) mem_154896)[i_152471] = zt_res_145868;
            ((double *) mem_154897)[i_152471] = zt_res_145855;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152485 = 0; i_152485 < (int64_t) 16; i_152485++) {
            // futhark/microgpt.fut:390:72-84
            
            double zs_rhs_145886 = ((double *) mem_154871)[i_152485];
            
            // futhark/microgpt.fut:390:64-84
            
            double zs_res_145887 = 1.0 / zs_rhs_145886;
            
            // futhark/microgpt.fut:390:94-106
            
            double zs_lhs_145888 = ((double *) mem_154897)[i_152485];
            
            // futhark/microgpt.fut:390:94-121
            
            double zs_res_145889 = zs_lhs_145888 / 16.0;
            
            // futhark/microgpt.fut:403:94-106
            
            double zs_lhs_145913 = ((double *) mem_154896)[i_152485];
            
            // futhark/microgpt.fut:403:94-121
            
            double zs_res_145914 = zs_lhs_145913 / 16.0;
            
            // futhark/microgpt.fut:403:72-84
            
            double zs_rhs_145911 = ((double *) mem_154869)[i_152485];
            
            // futhark/microgpt.fut:403:64-84
            
            double zs_res_145912 = 1.0 / zs_rhs_145911;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152478 = 0; i_152478 < (int64_t) 16; i_152478++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_150910 = ((double *) mem_154837)[i_152485 * (int64_t) 16 + i_152478];
                
                // futhark/microgpt.fut:390:38-84
                
                double zt_res_150911 = zs_res_145887 * zt_lhs_150910;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_150912 = ((double *) mem_param_152826.mem)[i_152485 * (int64_t) 16 + i_152478];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_150913 = ((double *) mem_152919)[i_152485 * (int64_t) 16 + i_152478];
                
                // futhark/microgpt.fut:390:128-172
                
                double zp_res_150914 = zp_lhs_150912 + zp_rhs_150913;
                
                // futhark/microgpt.fut:390:107-172
                
                double zt_res_150915 = zs_res_145889 * zp_res_150914;
                
                // futhark/microgpt.fut:390:123-259
                
                double zp_res_150916 = zt_res_150915 + zt_res_150915;
                
                // futhark/microgpt.fut:390:59-259
                
                double zp_res_150917 = zt_res_150911 + zp_res_150916;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_150924 = ((double *) mem_154836)[i_152485 * (int64_t) 16 + i_152478];
                
                // futhark/microgpt.fut:403:38-84
                
                double zt_res_150925 = zs_res_145912 * zt_lhs_150924;
                
                // futhark/microgpt.fut:403:107-172
                
                double zt_res_150929 = zs_res_145914 * zp_res_150914;
                
                // futhark/microgpt.fut:403:123-259
                
                double zp_res_150930 = zt_res_150929 + zt_res_150929;
                
                // futhark/microgpt.fut:403:59-259
                
                double zp_res_150931 = zt_res_150925 + zp_res_150930;
                
                ((double *) mem_154920)[i_152478] = zp_res_150931;
                ((double *) mem_154921)[i_152478] = zp_res_150917;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154910, i_152485 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154920, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154911, i_152485 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154921, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152494 = 0; i_152494 < (int64_t) 64; i_152494++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152490 = 0; i_152490 < (int64_t) 16; i_152490++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_140538;
                double r_140540 = 0.0;
                
                for (int64_t i_140539 = 0; i_140539 < (int64_t) 16; i_140539++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_140541 = ((double *) mem_154119)[i_140539 * (int64_t) 64 + i_152494];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_140542 = ((double *) mem_153833)[i_140539 * (int64_t) 16 + i_152490];
                    
                    // futhark/microgpt.fut:395:67-111
                    
                    double zt_res_140543 = zt_lhs_140541 * zt_rhs_140542;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_140544 = r_140540 + zt_res_140543;
                    double r_tmp_155665 = zp_res_140544;
                    
                    r_140540 = r_tmp_155665;
                }
                defunc_0_lifted_lambda_res_140538 = r_140540;
                ((double *) mem_154947)[i_152490] = defunc_0_lifted_lambda_res_140538;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154942, i_152494 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154947, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_152507 = 0; i_152507 < (int64_t) 27; i_152507++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_152500 = 0; i_152500 < (int64_t) 16; i_152500++) {
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_150952;
                double r_150954 = 0.0;
                
                for (int64_t i_150953 = 0; i_150953 < (int64_t) 16; i_150953++) {
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_lhs_150955 = ((double *) mem_154086)[i_150953 * (int64_t) 27 + i_152507];
                    
                    // futhark/microgpt.fut:61:46-49
                    
                    double zt_rhs_150956 = ((double *) mem_153881)[i_150953 * (int64_t) 16 + i_152500];
                    
                    // futhark/microgpt.fut:397:68-111
                    
                    double zt_res_150957 = zt_lhs_150955 * zt_rhs_150956;
                    
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_150958 = r_150954 + zt_res_150957;
                    double r_tmp_155670 = zp_res_150958;
                    
                    r_150954 = r_tmp_155670;
                }
                defunc_0_lifted_lambda_res_150952 = r_150954;
                // futhark/microgpt.fut:61:13-49
                
                double defunc_0_lifted_lambda_res_150961;
                double r_150963 = 0.0;
                
                for (int64_t i_150962 = 0; i_150962 < (int64_t) 16; i_150962++) {
                    int64_t zeze_lhs_150964 = ((int64_t *) seqs_mem_152810.mem)[step_138005 * (int64_t) 16 + i_150962];
                    
                    // futhark/microgpt.fut:548:58-109
                    
                    bool cond_150965 = zeze_lhs_150964 == i_152507;
                    
                    // futhark/microgpt.fut:548:58-109
                    
                    double lifted_lambda_res_150966;
                    
                    if (cond_150965) {
                        // futhark/microgpt.fut:61:46-49
                        
                        double lifted_lambda_res_t_res_151507 = ((double *) mem_154910)[i_150962 * (int64_t) 16 + i_152500];
                        
                        lifted_lambda_res_150966 = lifted_lambda_res_t_res_151507;
                    } else {
                        lifted_lambda_res_150966 = 0.0;
                    }
                    // futhark/microgpt.fut:61:40-49
                    
                    double zp_res_150972 = r_150963 + lifted_lambda_res_150966;
                    double r_tmp_155671 = zp_res_150972;
                    
                    r_150963 = r_tmp_155671;
                }
                defunc_0_lifted_lambda_res_150961 = r_150963;
                ((double *) mem_154968)[i_152500] = defunc_0_lifted_lambda_res_150961;
                ((double *) mem_154969)[i_152500] = defunc_0_lifted_lambda_res_150952;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154958, i_152507 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154968, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154959, i_152507 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154969, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:56:26-45
        
        double i64_res_140709 = sitofp_i64_f64(step_138005);
        
        // futhark/microgpt.fut:483:44-69
        
        double zm_rhs_140711 = i64_res_140709 / i64_res_140710;
        
        // futhark/microgpt.fut:483:22-69
        
        double zt_rhs_140712 = 1.0 - zm_rhs_140711;
        
        // futhark/microgpt.fut:483:17-69
        
        double lt_r_140713 = 1.0e-2 * zt_rhs_140712;
        
        // futhark/microgpt.fut:485:5-73
        if (memblock_alloc(ctx, &mem_154990, (int64_t) 3456, "mem_154990")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:485:5-73
        // futhark/microgpt.fut:485:5-73
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154990.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152834.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:485:5-73
        if (memblock_alloc(ctx, &mem_154992, (int64_t) 3456, "mem_154992")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:485:5-73
        // futhark/microgpt.fut:485:5-73
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154992.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152870.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:485:5-73
        if (memblock_alloc(ctx, &mem_154994, (int64_t) 3456, "mem_154994")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:485:5-73
        // futhark/microgpt.fut:485:5-73
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154994.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152906.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:485:5-73
        if (memblock_alloc(ctx, &mem_154996, (int64_t) 3456, "mem_154996")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:485:5-73
        // futhark/microgpt.fut:485:5-73
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154996.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154958, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:485:5-73
        if (futrts_adam_opt_w_12919(ctx, &ext_mem_155000, &ext_mem_154999, &ext_mem_154998, mem_154990, mem_154992, mem_154994, mem_154996, (int64_t) 27, (int64_t) 16, step_138005, lt_r_140713, 0.85, 0.99, 1.0e-8) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_154990, "mem_154990") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154992, "mem_154992") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154994, "mem_154994") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154996, "mem_154996") != 0)
            return 1;
        // futhark/microgpt.fut:487:5-73
        if (memblock_alloc(ctx, &mem_155001, (int64_t) 2048, "mem_155001")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:487:5-73
        // futhark/microgpt.fut:487:5-73
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155001.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152826.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:487:5-73
        if (memblock_alloc(ctx, &mem_155003, (int64_t) 2048, "mem_155003")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:487:5-73
        // futhark/microgpt.fut:487:5-73
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155003.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152862.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:487:5-73
        if (memblock_alloc(ctx, &mem_155005, (int64_t) 2048, "mem_155005")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:487:5-73
        // futhark/microgpt.fut:487:5-73
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155005.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152898.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:487:5-73
        if (memblock_alloc(ctx, &mem_155007, (int64_t) 2048, "mem_155007")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:487:5-73
        // futhark/microgpt.fut:487:5-73
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155007.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154911, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:487:5-73
        if (futrts_adam_opt_w_12920(ctx, &ext_mem_155011, &ext_mem_155010, &ext_mem_155009, mem_155001, mem_155003, mem_155005, mem_155007, (int64_t) 16, (int64_t) 16, step_138005, lt_r_140713, 0.85, 0.99, 1.0e-8) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_155001, "mem_155001") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155003, "mem_155003") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155005, "mem_155005") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155007, "mem_155007") != 0)
            return 1;
        // futhark/microgpt.fut:489:5-77
        if (memblock_alloc(ctx, &mem_155012, (int64_t) 2048, "mem_155012")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:489:5-77
        // futhark/microgpt.fut:489:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155012.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152830.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:489:5-77
        if (memblock_alloc(ctx, &mem_155014, (int64_t) 2048, "mem_155014")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:489:5-77
        // futhark/microgpt.fut:489:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155014.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152866.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:489:5-77
        if (memblock_alloc(ctx, &mem_155016, (int64_t) 2048, "mem_155016")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:489:5-77
        // futhark/microgpt.fut:489:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155016.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152902.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:489:5-77
        if (memblock_alloc(ctx, &mem_155018, (int64_t) 2048, "mem_155018")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:489:5-77
        // futhark/microgpt.fut:489:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155018.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154753, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:489:5-77
        if (futrts_adam_opt_w_12920(ctx, &ext_mem_155022, &ext_mem_155021, &ext_mem_155020, mem_155012, mem_155014, mem_155016, mem_155018, (int64_t) 16, (int64_t) 16, step_138005, lt_r_140713, 0.85, 0.99, 1.0e-8) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_155012, "mem_155012") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155014, "mem_155014") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155016, "mem_155016") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155018, "mem_155018") != 0)
            return 1;
        // futhark/microgpt.fut:491:5-77
        if (memblock_alloc(ctx, &mem_155023, (int64_t) 2048, "mem_155023")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:491:5-77
        // futhark/microgpt.fut:491:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155023.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152818.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:491:5-77
        if (memblock_alloc(ctx, &mem_155025, (int64_t) 2048, "mem_155025")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:491:5-77
        // futhark/microgpt.fut:491:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155025.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152854.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:491:5-77
        if (memblock_alloc(ctx, &mem_155027, (int64_t) 2048, "mem_155027")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:491:5-77
        // futhark/microgpt.fut:491:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155027.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152890.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:491:5-77
        if (memblock_alloc(ctx, &mem_155029, (int64_t) 2048, "mem_155029")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:491:5-77
        // futhark/microgpt.fut:491:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155029.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154752, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:491:5-77
        if (futrts_adam_opt_w_12920(ctx, &ext_mem_155033, &ext_mem_155032, &ext_mem_155031, mem_155023, mem_155025, mem_155027, mem_155029, (int64_t) 16, (int64_t) 16, step_138005, lt_r_140713, 0.85, 0.99, 1.0e-8) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_155023, "mem_155023") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155025, "mem_155025") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155027, "mem_155027") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155029, "mem_155029") != 0)
            return 1;
        // futhark/microgpt.fut:493:5-77
        if (memblock_alloc(ctx, &mem_155034, (int64_t) 2048, "mem_155034")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:493:5-77
        // futhark/microgpt.fut:493:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155034.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152842.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:493:5-77
        if (memblock_alloc(ctx, &mem_155036, (int64_t) 2048, "mem_155036")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:493:5-77
        // futhark/microgpt.fut:493:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155036.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152878.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:493:5-77
        if (memblock_alloc(ctx, &mem_155038, (int64_t) 2048, "mem_155038")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:493:5-77
        // futhark/microgpt.fut:493:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155038.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152914.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:493:5-77
        if (memblock_alloc(ctx, &mem_155040, (int64_t) 2048, "mem_155040")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:493:5-77
        // futhark/microgpt.fut:493:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155040.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154751, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:493:5-77
        if (futrts_adam_opt_w_12920(ctx, &ext_mem_155044, &ext_mem_155043, &ext_mem_155042, mem_155034, mem_155036, mem_155038, mem_155040, (int64_t) 16, (int64_t) 16, step_138005, lt_r_140713, 0.85, 0.99, 1.0e-8) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_155034, "mem_155034") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155036, "mem_155036") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155038, "mem_155038") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155040, "mem_155040") != 0)
            return 1;
        // futhark/microgpt.fut:495:5-77
        if (memblock_alloc(ctx, &mem_155045, (int64_t) 2048, "mem_155045")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:495:5-77
        // futhark/microgpt.fut:495:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155045.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152822.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:495:5-77
        if (memblock_alloc(ctx, &mem_155047, (int64_t) 2048, "mem_155047")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:495:5-77
        // futhark/microgpt.fut:495:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155047.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152858.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:495:5-77
        if (memblock_alloc(ctx, &mem_155049, (int64_t) 2048, "mem_155049")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:495:5-77
        // futhark/microgpt.fut:495:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155049.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152894.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:495:5-77
        if (memblock_alloc(ctx, &mem_155051, (int64_t) 2048, "mem_155051")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:495:5-77
        // futhark/microgpt.fut:495:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155051.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154671, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:495:5-77
        if (futrts_adam_opt_w_12920(ctx, &ext_mem_155055, &ext_mem_155054, &ext_mem_155053, mem_155045, mem_155047, mem_155049, mem_155051, (int64_t) 16, (int64_t) 16, step_138005, lt_r_140713, 0.85, 0.99, 1.0e-8) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_155045, "mem_155045") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155047, "mem_155047") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155049, "mem_155049") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155051, "mem_155051") != 0)
            return 1;
        // futhark/microgpt.fut:497:5-73
        if (memblock_alloc(ctx, &mem_155056, (int64_t) 8192, "mem_155056")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:497:5-73
        // futhark/microgpt.fut:497:5-73
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155056.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152838.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:497:5-73
        if (memblock_alloc(ctx, &mem_155058, (int64_t) 8192, "mem_155058")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:497:5-73
        // futhark/microgpt.fut:497:5-73
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155058.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152874.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:497:5-73
        if (memblock_alloc(ctx, &mem_155060, (int64_t) 8192, "mem_155060")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:497:5-73
        // futhark/microgpt.fut:497:5-73
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155060.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152910.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:497:5-73
        if (memblock_alloc(ctx, &mem_155062, (int64_t) 8192, "mem_155062")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:497:5-73
        // futhark/microgpt.fut:497:5-73
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155062.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154942, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:497:5-73
        if (futrts_adam_opt_w_12919(ctx, &ext_mem_155066, &ext_mem_155065, &ext_mem_155064, mem_155056, mem_155058, mem_155060, mem_155062, (int64_t) 64, (int64_t) 16, step_138005, lt_r_140713, 0.85, 0.99, 1.0e-8) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_155056, "mem_155056") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155058, "mem_155058") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155060, "mem_155060") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155062, "mem_155062") != 0)
            return 1;
        // futhark/microgpt.fut:499:5-81
        if (memblock_alloc(ctx, &mem_155067, (int64_t) 8192, "mem_155067")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:499:5-81
        // futhark/microgpt.fut:499:5-81
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155067.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_152814.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:499:5-81
        if (memblock_alloc(ctx, &mem_155069, (int64_t) 8192, "mem_155069")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:499:5-81
        // futhark/microgpt.fut:499:5-81
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155069.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_152850.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:499:5-81
        if (memblock_alloc(ctx, &mem_155071, (int64_t) 8192, "mem_155071")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:499:5-81
        // futhark/microgpt.fut:499:5-81
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155071.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_152886.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:499:5-81
        if (memblock_alloc(ctx, &mem_155073, (int64_t) 8192, "mem_155073")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:499:5-81
        // futhark/microgpt.fut:499:5-81
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155073.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_154118, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:499:5-81
        if (futrts_adam_opt_w_12919(ctx, &ext_mem_155077, &ext_mem_155076, &ext_mem_155075, mem_155067, mem_155069, mem_155071, mem_155073, (int64_t) 16, (int64_t) 64, step_138005, lt_r_140713, 0.85, 0.99, 1.0e-8) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_155067, "mem_155067") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155069, "mem_155069") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155071, "mem_155071") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155073, "mem_155073") != 0)
            return 1;
        // futhark/microgpt.fut:501:5-77
        if (memblock_alloc(ctx, &mem_155078, (int64_t) 3456, "mem_155078")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:501:5-77
        // futhark/microgpt.fut:501:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155078.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152846.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:501:5-77
        if (memblock_alloc(ctx, &mem_155080, (int64_t) 3456, "mem_155080")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:501:5-77
        // futhark/microgpt.fut:501:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155080.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152882.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:501:5-77
        if (memblock_alloc(ctx, &mem_155082, (int64_t) 3456, "mem_155082")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:501:5-77
        // futhark/microgpt.fut:501:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155082.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152918.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:501:5-77
        if (memblock_alloc(ctx, &mem_155084, (int64_t) 3456, "mem_155084")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:501:5-77
        // futhark/microgpt.fut:501:5-77
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_155084.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154959, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:501:5-77
        if (futrts_adam_opt_w_12919(ctx, &ext_mem_155088, &ext_mem_155087, &ext_mem_155086, mem_155078, mem_155080, mem_155082, mem_155084, (int64_t) 27, (int64_t) 16, step_138005, lt_r_140713, 0.85, 0.99, 1.0e-8) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_155078, "mem_155078") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155080, "mem_155080") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155082, "mem_155082") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155084, "mem_155084") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155299, &ext_mem_155077, "ext_mem_155077") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155300, &ext_mem_155033, "ext_mem_155033") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155301, &ext_mem_155055, "ext_mem_155055") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155302, &ext_mem_155011, "ext_mem_155011") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155303, &ext_mem_155022, "ext_mem_155022") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155304, &ext_mem_155000, "ext_mem_155000") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155305, &ext_mem_155066, "ext_mem_155066") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155306, &ext_mem_155044, "ext_mem_155044") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155307, &ext_mem_155088, "ext_mem_155088") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155308, &ext_mem_155076, "ext_mem_155076") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155309, &ext_mem_155032, "ext_mem_155032") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155310, &ext_mem_155054, "ext_mem_155054") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155311, &ext_mem_155010, "ext_mem_155010") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155312, &ext_mem_155021, "ext_mem_155021") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155313, &ext_mem_154999, "ext_mem_154999") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155314, &ext_mem_155065, "ext_mem_155065") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155315, &ext_mem_155043, "ext_mem_155043") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155316, &ext_mem_155087, "ext_mem_155087") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155317, &ext_mem_155075, "ext_mem_155075") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155318, &ext_mem_155031, "ext_mem_155031") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155319, &ext_mem_155053, "ext_mem_155053") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155320, &ext_mem_155009, "ext_mem_155009") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155321, &ext_mem_155020, "ext_mem_155020") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155322, &ext_mem_154998, "ext_mem_154998") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155323, &ext_mem_155064, "ext_mem_155064") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155324, &ext_mem_155042, "ext_mem_155042") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_155325, &ext_mem_155086, "ext_mem_155086") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152814, &mem_param_tmp_155299, "mem_param_tmp_155299") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152818, &mem_param_tmp_155300, "mem_param_tmp_155300") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152822, &mem_param_tmp_155301, "mem_param_tmp_155301") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152826, &mem_param_tmp_155302, "mem_param_tmp_155302") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152830, &mem_param_tmp_155303, "mem_param_tmp_155303") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152834, &mem_param_tmp_155304, "mem_param_tmp_155304") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152838, &mem_param_tmp_155305, "mem_param_tmp_155305") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152842, &mem_param_tmp_155306, "mem_param_tmp_155306") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152846, &mem_param_tmp_155307, "mem_param_tmp_155307") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152850, &mem_param_tmp_155308, "mem_param_tmp_155308") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152854, &mem_param_tmp_155309, "mem_param_tmp_155309") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152858, &mem_param_tmp_155310, "mem_param_tmp_155310") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152862, &mem_param_tmp_155311, "mem_param_tmp_155311") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152866, &mem_param_tmp_155312, "mem_param_tmp_155312") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152870, &mem_param_tmp_155313, "mem_param_tmp_155313") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152874, &mem_param_tmp_155314, "mem_param_tmp_155314") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152878, &mem_param_tmp_155315, "mem_param_tmp_155315") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152882, &mem_param_tmp_155316, "mem_param_tmp_155316") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152886, &mem_param_tmp_155317, "mem_param_tmp_155317") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152890, &mem_param_tmp_155318, "mem_param_tmp_155318") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152894, &mem_param_tmp_155319, "mem_param_tmp_155319") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152898, &mem_param_tmp_155320, "mem_param_tmp_155320") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152902, &mem_param_tmp_155321, "mem_param_tmp_155321") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152906, &mem_param_tmp_155322, "mem_param_tmp_155322") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152910, &mem_param_tmp_155323, "mem_param_tmp_155323") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152914, &mem_param_tmp_155324, "mem_param_tmp_155324") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152918, &mem_param_tmp_155325, "mem_param_tmp_155325") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_155196, &mem_param_152814, "mem_param_152814") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155195, &mem_param_152818, "mem_param_152818") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155194, &mem_param_152822, "mem_param_152822") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155193, &mem_param_152826, "mem_param_152826") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155192, &mem_param_152830, "mem_param_152830") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155191, &mem_param_152834, "mem_param_152834") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155190, &mem_param_152838, "mem_param_152838") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155189, &mem_param_152842, "mem_param_152842") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155188, &mem_param_152846, "mem_param_152846") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155187, &mem_param_152850, "mem_param_152850") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155186, &mem_param_152854, "mem_param_152854") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155185, &mem_param_152858, "mem_param_152858") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155184, &mem_param_152862, "mem_param_152862") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155183, &mem_param_152866, "mem_param_152866") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155182, &mem_param_152870, "mem_param_152870") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155181, &mem_param_152874, "mem_param_152874") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155180, &mem_param_152878, "mem_param_152878") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155179, &mem_param_152882, "mem_param_152882") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155178, &mem_param_152886, "mem_param_152886") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155177, &mem_param_152890, "mem_param_152890") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155176, &mem_param_152894, "mem_param_152894") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155175, &mem_param_152898, "mem_param_152898") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155174, &mem_param_152902, "mem_param_152902") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155173, &mem_param_152906, "mem_param_152906") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155172, &mem_param_152910, "mem_param_152910") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155171, &mem_param_152914, "mem_param_152914") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_155170, &mem_param_152918, "mem_param_152918") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155269, &ext_mem_155191, "ext_mem_155191") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155270, &ext_mem_155193, "ext_mem_155193") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155271, &ext_mem_155192, "ext_mem_155192") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155272, &ext_mem_155195, "ext_mem_155195") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155273, &ext_mem_155189, "ext_mem_155189") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155274, &ext_mem_155194, "ext_mem_155194") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155275, &ext_mem_155190, "ext_mem_155190") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155276, &ext_mem_155196, "ext_mem_155196") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155277, &ext_mem_155188, "ext_mem_155188") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155278, &ext_mem_155182, "ext_mem_155182") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155279, &ext_mem_155184, "ext_mem_155184") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155280, &ext_mem_155183, "ext_mem_155183") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155281, &ext_mem_155186, "ext_mem_155186") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155282, &ext_mem_155180, "ext_mem_155180") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155283, &ext_mem_155185, "ext_mem_155185") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155284, &ext_mem_155181, "ext_mem_155181") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155285, &ext_mem_155187, "ext_mem_155187") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155286, &ext_mem_155179, "ext_mem_155179") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155287, &ext_mem_155173, "ext_mem_155173") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155288, &ext_mem_155175, "ext_mem_155175") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155289, &ext_mem_155174, "ext_mem_155174") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155290, &ext_mem_155177, "ext_mem_155177") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155291, &ext_mem_155171, "ext_mem_155171") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155292, &ext_mem_155176, "ext_mem_155176") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155293, &ext_mem_155172, "ext_mem_155172") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155294, &ext_mem_155178, "ext_mem_155178") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155295, &ext_mem_155170, "ext_mem_155170") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155825, &mem_out_155269, "mem_out_155269") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155826, &mem_out_155270, "mem_out_155270") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155827, &mem_out_155271, "mem_out_155271") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155828, &mem_out_155272, "mem_out_155272") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155829, &mem_out_155273, "mem_out_155273") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155830, &mem_out_155274, "mem_out_155274") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155831, &mem_out_155275, "mem_out_155275") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155832, &mem_out_155276, "mem_out_155276") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155833, &mem_out_155277, "mem_out_155277") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155834, &mem_out_155278, "mem_out_155278") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155835, &mem_out_155279, "mem_out_155279") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155836, &mem_out_155280, "mem_out_155280") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155837, &mem_out_155281, "mem_out_155281") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155838, &mem_out_155282, "mem_out_155282") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155839, &mem_out_155283, "mem_out_155283") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155840, &mem_out_155284, "mem_out_155284") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155841, &mem_out_155285, "mem_out_155285") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155842, &mem_out_155286, "mem_out_155286") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155843, &mem_out_155287, "mem_out_155287") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155844, &mem_out_155288, "mem_out_155288") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155845, &mem_out_155289, "mem_out_155289") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155846, &mem_out_155290, "mem_out_155290") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155847, &mem_out_155291, "mem_out_155291") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155848, &mem_out_155292, "mem_out_155292") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155849, &mem_out_155293, "mem_out_155293") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155850, &mem_out_155294, "mem_out_155294") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155851, &mem_out_155295, "mem_out_155295") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_152919);
        free(mem_152920);
        free(mem_152929);
        free(mem_152936);
        free(mem_152951);
        free(mem_152952);
        free(mem_152953);
        free(mem_152972);
        free(mem_152979);
        free(mem_152984);
        free(mem_152995);
        free(mem_153000);
        free(mem_153011);
        free(mem_153012);
        free(mem_153025);
        free(mem_153032);
        free(mem_153037);
        free(mem_153048);
        free(mem_153053);
        free(mem_153064);
        free(mem_153065);
        free(mem_153066);
        free(mem_153082);
        free(mem_153083);
        free(mem_153084);
        free(mem_153097);
        free(mem_153098);
        free(mem_153099);
        free(mem_153145);
        free(mem_153146);
        free(mem_153147);
        free(mem_153148);
        free(mem_153169);
        free(mem_153170);
        free(mem_153171);
        free(mem_153172);
        free(mem_153189);
        free(mem_153190);
        free(mem_153191);
        free(mem_153192);
        free(mem_153253);
        free(mem_153254);
        free(mem_153255);
        free(mem_153256);
        free(mem_153277);
        free(mem_153278);
        free(mem_153279);
        free(mem_153280);
        free(mem_153297);
        free(mem_153298);
        free(mem_153299);
        free(mem_153300);
        free(mem_153361);
        free(mem_153362);
        free(mem_153363);
        free(mem_153364);
        free(mem_153365);
        free(mem_153366);
        free(mem_153367);
        free(mem_153368);
        free(mem_153401);
        free(mem_153402);
        free(mem_153403);
        free(mem_153404);
        free(mem_153405);
        free(mem_153406);
        free(mem_153407);
        free(mem_153408);
        free(mem_153489);
        free(mem_153490);
        free(mem_153491);
        free(mem_153492);
        free(mem_153513);
        free(mem_153514);
        free(mem_153515);
        free(mem_153516);
        free(mem_153533);
        free(mem_153534);
        free(mem_153535);
        free(mem_153536);
        free(mem_153597);
        free(mem_153598);
        free(mem_153607);
        free(mem_153608);
        free(mem_153629);
        free(mem_153630);
        free(mem_153641);
        free(mem_153642);
        free(mem_153651);
        free(mem_153652);
        free(mem_153683);
        free(mem_153684);
        free(mem_153695);
        free(mem_153696);
        free(mem_153705);
        free(mem_153706);
        free(mem_153737);
        free(mem_153743);
        free(mem_153748);
        free(mem_153764);
        free(mem_153769);
        free(mem_153780);
        free(mem_153785);
        free(mem_153796);
        free(mem_153797);
        free(mem_153810);
        free(mem_153817);
        free(mem_153822);
        free(mem_153833);
        free(mem_153838);
        free(mem_153849);
        free(mem_153854);
        free(mem_153865);
        free(mem_153870);
        free(mem_153881);
        free(mem_153886);
        free(mem_153897);
        free(mem_153902);
        free(mem_153913);
        free(mem_153914);
        free(mem_153915);
        free(mem_153916);
        free(mem_153934);
        free(mem_153939);
        free(mem_153943);
        free(mem_153950);
        free(mem_153984);
        free(mem_153990);
        free(mem_153995);
        free(mem_154011);
        free(mem_154012);
        free(mem_154021);
        free(mem_154022);
        free(mem_154043);
        free(mem_154049);
        free(mem_154054);
        free(mem_154070);
        free(mem_154075);
        free(mem_154086);
        free(mem_154091);
        free(mem_154102);
        free(mem_154107);
        free(mem_154118);
        free(mem_154119);
        free(mem_154128);
        free(mem_154129);
        free(mem_154150);
        free(mem_154155);
        free(mem_154166);
        free(mem_154167);
        free(mem_154180);
        free(mem_154187);
        free(mem_154192);
        free(mem_154203);
        free(mem_154209);
        free(mem_154214);
        free(mem_154230);
        free(mem_154231);
        free(mem_154232);
        free(mem_154248);
        free(mem_154249);
        free(mem_154250);
        free(mem_154263);
        free(mem_154264);
        free(mem_154305);
        free(mem_154306);
        free(mem_154317);
        free(mem_154318);
        free(mem_154327);
        free(mem_154328);
        free(mem_154359);
        free(mem_154360);
        free(mem_154371);
        free(mem_154372);
        free(mem_154381);
        free(mem_154382);
        free(mem_154413);
        free(mem_154414);
        free(mem_154415);
        free(mem_154416);
        free(mem_154433);
        free(mem_154434);
        free(mem_154435);
        free(mem_154436);
        free(mem_154477);
        free(mem_154478);
        free(mem_154489);
        free(mem_154490);
        free(mem_154499);
        free(mem_154500);
        free(mem_154531);
        free(mem_154532);
        free(mem_154541);
        free(mem_154542);
        free(mem_154563);
        free(mem_154564);
        free(mem_154575);
        free(mem_154576);
        free(mem_154585);
        free(mem_154586);
        free(mem_154617);
        free(mem_154618);
        free(mem_154629);
        free(mem_154630);
        free(mem_154639);
        free(mem_154640);
        free(mem_154671);
        free(mem_154672);
        free(mem_154673);
        free(mem_154674);
        free(mem_154691);
        free(mem_154692);
        free(mem_154693);
        free(mem_154694);
        free(mem_154735);
        free(mem_154740);
        free(mem_154751);
        free(mem_154752);
        free(mem_154753);
        free(mem_154754);
        free(mem_154755);
        free(mem_154774);
        free(mem_154775);
        free(mem_154776);
        free(mem_154813);
        free(mem_154820);
        free(mem_154825);
        free(mem_154836);
        free(mem_154837);
        free(mem_154846);
        free(mem_154847);
        free(mem_154868);
        free(mem_154869);
        free(mem_154870);
        free(mem_154871);
        free(mem_154896);
        free(mem_154897);
        free(mem_154910);
        free(mem_154911);
        free(mem_154920);
        free(mem_154921);
        free(mem_154942);
        free(mem_154947);
        free(mem_154958);
        free(mem_154959);
        free(mem_154968);
        free(mem_154969);
        if (memblock_unref(ctx, &mem_param_tmp_155325, "mem_param_tmp_155325") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155324, "mem_param_tmp_155324") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155323, "mem_param_tmp_155323") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155322, "mem_param_tmp_155322") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155321, "mem_param_tmp_155321") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155320, "mem_param_tmp_155320") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155319, "mem_param_tmp_155319") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155318, "mem_param_tmp_155318") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155317, "mem_param_tmp_155317") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155316, "mem_param_tmp_155316") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155315, "mem_param_tmp_155315") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155314, "mem_param_tmp_155314") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155313, "mem_param_tmp_155313") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155312, "mem_param_tmp_155312") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155311, "mem_param_tmp_155311") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155310, "mem_param_tmp_155310") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155309, "mem_param_tmp_155309") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155308, "mem_param_tmp_155308") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155307, "mem_param_tmp_155307") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155306, "mem_param_tmp_155306") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155305, "mem_param_tmp_155305") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155304, "mem_param_tmp_155304") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155303, "mem_param_tmp_155303") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155302, "mem_param_tmp_155302") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155301, "mem_param_tmp_155301") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155300, "mem_param_tmp_155300") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_155299, "mem_param_tmp_155299") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155086, "ext_mem_155086") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155087, "ext_mem_155087") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155088, "ext_mem_155088") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155084, "mem_155084") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155082, "mem_155082") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155080, "mem_155080") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155078, "mem_155078") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155075, "ext_mem_155075") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155076, "ext_mem_155076") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155077, "ext_mem_155077") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155073, "mem_155073") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155071, "mem_155071") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155069, "mem_155069") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155067, "mem_155067") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155064, "ext_mem_155064") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155065, "ext_mem_155065") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155066, "ext_mem_155066") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155062, "mem_155062") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155060, "mem_155060") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155058, "mem_155058") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155056, "mem_155056") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155053, "ext_mem_155053") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155054, "ext_mem_155054") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155055, "ext_mem_155055") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155051, "mem_155051") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155049, "mem_155049") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155047, "mem_155047") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155045, "mem_155045") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155042, "ext_mem_155042") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155043, "ext_mem_155043") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155044, "ext_mem_155044") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155040, "mem_155040") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155038, "mem_155038") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155036, "mem_155036") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155034, "mem_155034") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155031, "ext_mem_155031") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155032, "ext_mem_155032") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155033, "ext_mem_155033") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155029, "mem_155029") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155027, "mem_155027") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155025, "mem_155025") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155023, "mem_155023") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155020, "ext_mem_155020") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155021, "ext_mem_155021") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155022, "ext_mem_155022") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155018, "mem_155018") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155016, "mem_155016") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155014, "mem_155014") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155012, "mem_155012") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155009, "ext_mem_155009") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155010, "ext_mem_155010") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155011, "ext_mem_155011") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155007, "mem_155007") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155005, "mem_155005") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155003, "mem_155003") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_155001, "mem_155001") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154998, "ext_mem_154998") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154999, "ext_mem_154999") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155000, "ext_mem_155000") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154996, "mem_154996") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154994, "mem_154994") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154992, "mem_154992") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154990, "mem_154990") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152918, "mem_param_152918") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152914, "mem_param_152914") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152910, "mem_param_152910") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152906, "mem_param_152906") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152902, "mem_param_152902") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152898, "mem_param_152898") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152894, "mem_param_152894") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152890, "mem_param_152890") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152886, "mem_param_152886") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152882, "mem_param_152882") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152878, "mem_param_152878") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152874, "mem_param_152874") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152870, "mem_param_152870") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152866, "mem_param_152866") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152862, "mem_param_152862") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152858, "mem_param_152858") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152854, "mem_param_152854") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152850, "mem_param_152850") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152846, "mem_param_152846") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152842, "mem_param_152842") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152838, "mem_param_152838") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152834, "mem_param_152834") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152830, "mem_param_152830") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152826, "mem_param_152826") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152822, "mem_param_152822") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152818, "mem_param_152818") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152814, "mem_param_152814") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155170, "ext_mem_155170") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155171, "ext_mem_155171") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155172, "ext_mem_155172") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155173, "ext_mem_155173") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155174, "ext_mem_155174") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155175, "ext_mem_155175") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155176, "ext_mem_155176") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155177, "ext_mem_155177") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155178, "ext_mem_155178") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155179, "ext_mem_155179") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155180, "ext_mem_155180") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155181, "ext_mem_155181") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155182, "ext_mem_155182") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155183, "ext_mem_155183") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155184, "ext_mem_155184") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155185, "ext_mem_155185") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155186, "ext_mem_155186") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155187, "ext_mem_155187") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155188, "ext_mem_155188") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155189, "ext_mem_155189") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155190, "ext_mem_155190") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155191, "ext_mem_155191") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155192, "ext_mem_155192") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155193, "ext_mem_155193") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155194, "ext_mem_155194") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155195, "ext_mem_155195") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_155196, "ext_mem_155196") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155295, "mem_out_155295") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155294, "mem_out_155294") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155293, "mem_out_155293") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155292, "mem_out_155292") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155291, "mem_out_155291") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155290, "mem_out_155290") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155289, "mem_out_155289") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155288, "mem_out_155288") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155287, "mem_out_155287") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155286, "mem_out_155286") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155285, "mem_out_155285") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155284, "mem_out_155284") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155283, "mem_out_155283") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155282, "mem_out_155282") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155281, "mem_out_155281") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155280, "mem_out_155280") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155279, "mem_out_155279") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155278, "mem_out_155278") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155277, "mem_out_155277") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155276, "mem_out_155276") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155275, "mem_out_155275") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155274, "mem_out_155274") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155273, "mem_out_155273") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155272, "mem_out_155272") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155271, "mem_out_155271") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155270, "mem_out_155270") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155269, "mem_out_155269") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_156099, struct memblock *mem_out_p_156100, struct memblock *mem_out_p_156101, struct memblock *mem_out_p_156102, struct memblock *mem_out_p_156103, struct memblock *mem_out_p_156104, struct memblock *mem_out_p_156105, struct memblock *mem_out_p_156106, struct memblock *mem_out_p_156107)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_155277;
    
    mem_out_155277.references = NULL;
    
    struct memblock mem_out_155276;
    
    mem_out_155276.references = NULL;
    
    struct memblock mem_out_155275;
    
    mem_out_155275.references = NULL;
    
    struct memblock mem_out_155274;
    
    mem_out_155274.references = NULL;
    
    struct memblock mem_out_155273;
    
    mem_out_155273.references = NULL;
    
    struct memblock mem_out_155272;
    
    mem_out_155272.references = NULL;
    
    struct memblock mem_out_155271;
    
    mem_out_155271.references = NULL;
    
    struct memblock mem_out_155270;
    
    mem_out_155270.references = NULL;
    
    struct memblock mem_out_155269;
    
    mem_out_155269.references = NULL;
    
    struct memblock mem_152772 = ctx->constants->mem_152772;
    struct memblock mem_152773 = ctx->constants->mem_152773;
    struct memblock mem_152774 = ctx->constants->mem_152774;
    struct memblock mem_152775 = ctx->constants->mem_152775;
    struct memblock mem_152776 = ctx->constants->mem_152776;
    struct memblock mem_152777 = ctx->constants->mem_152777;
    struct memblock mem_152778 = ctx->constants->mem_152778;
    struct memblock mem_152779 = ctx->constants->mem_152779;
    struct memblock mem_152780 = ctx->constants->mem_152780;
    
    if (memblock_set(ctx, &mem_out_155269, &mem_152779, "mem_152779") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155270, &mem_152775, "mem_152775") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155271, &mem_152777, "mem_152777") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155272, &mem_152773, "mem_152773") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155273, &mem_152774, "mem_152774") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155274, &mem_152772, "mem_152772") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155275, &mem_152778, "mem_152778") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155276, &mem_152776, "mem_152776") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_155277, &mem_152780, "mem_152780") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_156099, &mem_out_155269, "mem_out_155269") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_156100, &mem_out_155270, "mem_out_155270") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_156101, &mem_out_155271, "mem_out_155271") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_156102, &mem_out_155272, "mem_out_155272") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_156103, &mem_out_155273, "mem_out_155273") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_156104, &mem_out_155274, "mem_out_155274") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_156105, &mem_out_155275, "mem_out_155275") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_156106, &mem_out_155276, "mem_out_155276") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_156107, &mem_out_155277, "mem_out_155277") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_155277, "mem_out_155277") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155276, "mem_out_155276") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155275, "mem_out_155275") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155274, "mem_out_155274") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155273, "mem_out_155273") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155272, "mem_out_155272") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155271, "mem_out_155271") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155270, "mem_out_155270") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_155269, "mem_out_155269") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_cal_loss(struct futhark_context *ctx, struct futhark_opaque_tup2_f64_arr1d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3)
{
    double prim_out_155270 = 0.0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_155269;
    
    mem_out_155269.references = NULL;
    
    struct memblock mask_mem_152792;
    
    mask_mem_152792.references = NULL;
    
    struct memblock target_mem_152791;
    
    target_mem_152791.references = NULL;
    
    struct memblock tokens_mem_152790;
    
    tokens_mem_152790.references = NULL;
    
    struct memblock wvoc_mem_152789;
    
    wvoc_mem_152789.references = NULL;
    
    struct memblock wval_mem_152788;
    
    wval_mem_152788.references = NULL;
    
    struct memblock wup_mem_152787;
    
    wup_mem_152787.references = NULL;
    
    struct memblock wte_mem_152786;
    
    wte_mem_152786.references = NULL;
    
    struct memblock wqry_mem_152785;
    
    wqry_mem_152785.references = NULL;
    
    struct memblock wpe_mem_152784;
    
    wpe_mem_152784.references = NULL;
    
    struct memblock wout_mem_152783;
    
    wout_mem_152783.references = NULL;
    
    struct memblock wkey_mem_152782;
    
    wkey_mem_152782.references = NULL;
    
    struct memblock wdown_mem_152781;
    
    wdown_mem_152781.references = NULL;
    wdown_mem_152781 = in0->v0->mem;
    wkey_mem_152782 = in0->v1->mem;
    wout_mem_152783 = in0->v2->mem;
    wpe_mem_152784 = in0->v3->mem;
    wqry_mem_152785 = in0->v4->mem;
    wte_mem_152786 = in0->v5->mem;
    wup_mem_152787 = in0->v6->mem;
    wval_mem_152788 = in0->v7->mem;
    wvoc_mem_152789 = in0->v8->mem;
    tokens_mem_152790 = in1->mem;
    target_mem_152791 = in2->mem;
    mask_mem_152792 = in3->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && (((int64_t) 16 == in2->shape[0] && (int64_t) 27 == in2->shape[1]) && ((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_cal_loss(ctx, &mem_out_155269, &prim_out_155270, wdown_mem_152781, wkey_mem_152782, wout_mem_152783, wpe_mem_152784, wqry_mem_152785, wte_mem_152786, wup_mem_152787, wval_mem_152788, wvoc_mem_152789, tokens_mem_152790, target_mem_152791, mask_mem_152792);
        if (ret == 0) {
            struct memblock mem_152772 = ctx->constants->mem_152772;
            struct memblock mem_152773 = ctx->constants->mem_152773;
            struct memblock mem_152774 = ctx->constants->mem_152774;
            struct memblock mem_152775 = ctx->constants->mem_152775;
            struct memblock mem_152776 = ctx->constants->mem_152776;
            struct memblock mem_152777 = ctx->constants->mem_152777;
            struct memblock mem_152778 = ctx->constants->mem_152778;
            struct memblock mem_152779 = ctx->constants->mem_152779;
            struct memblock mem_152780 = ctx->constants->mem_152780;
            
            assert((*out = (struct futhark_opaque_tup2_f64_arr1d_f64 *) malloc(sizeof(struct futhark_opaque_tup2_f64_arr1d_f64))) != NULL);
            (*out)->v0 = prim_out_155270;
            assert(((*out)->v1 = (struct futhark_f64_1d *) malloc(sizeof(struct futhark_f64_1d))) != NULL);
            (*out)->v1->mem = mem_out_155269;
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
    
    struct memblock mem_out_155269;
    
    mem_out_155269.references = NULL;
    
    struct memblock mask_mem_152791;
    
    mask_mem_152791.references = NULL;
    
    struct memblock tokens_mem_152790;
    
    tokens_mem_152790.references = NULL;
    
    struct memblock wvoc_mem_152789;
    
    wvoc_mem_152789.references = NULL;
    
    struct memblock wval_mem_152788;
    
    wval_mem_152788.references = NULL;
    
    struct memblock wup_mem_152787;
    
    wup_mem_152787.references = NULL;
    
    struct memblock wte_mem_152786;
    
    wte_mem_152786.references = NULL;
    
    struct memblock wqry_mem_152785;
    
    wqry_mem_152785.references = NULL;
    
    struct memblock wpe_mem_152784;
    
    wpe_mem_152784.references = NULL;
    
    struct memblock wout_mem_152783;
    
    wout_mem_152783.references = NULL;
    
    struct memblock wkey_mem_152782;
    
    wkey_mem_152782.references = NULL;
    
    struct memblock wdown_mem_152781;
    
    wdown_mem_152781.references = NULL;
    wdown_mem_152781 = in0->v0->mem;
    wkey_mem_152782 = in0->v1->mem;
    wout_mem_152783 = in0->v2->mem;
    wpe_mem_152784 = in0->v3->mem;
    wqry_mem_152785 = in0->v4->mem;
    wte_mem_152786 = in0->v5->mem;
    wup_mem_152787 = in0->v6->mem;
    wval_mem_152788 = in0->v7->mem;
    wvoc_mem_152789 = in0->v8->mem;
    tokens_mem_152790 = in1->mem;
    mask_mem_152791 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_155269, wdown_mem_152781, wkey_mem_152782, wout_mem_152783, wpe_mem_152784, wqry_mem_152785, wte_mem_152786, wup_mem_152787, wval_mem_152788, wvoc_mem_152789, tokens_mem_152790, mask_mem_152791);
        if (ret == 0) {
            struct memblock mem_152772 = ctx->constants->mem_152772;
            struct memblock mem_152773 = ctx->constants->mem_152773;
            struct memblock mem_152774 = ctx->constants->mem_152774;
            struct memblock mem_152775 = ctx->constants->mem_152775;
            struct memblock mem_152776 = ctx->constants->mem_152776;
            struct memblock mem_152777 = ctx->constants->mem_152777;
            struct memblock mem_152778 = ctx->constants->mem_152778;
            struct memblock mem_152779 = ctx->constants->mem_152779;
            struct memblock mem_152780 = ctx->constants->mem_152780;
            
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_155269;
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
    
    struct memblock mem_out_155277;
    
    mem_out_155277.references = NULL;
    
    struct memblock mem_out_155276;
    
    mem_out_155276.references = NULL;
    
    struct memblock mem_out_155275;
    
    mem_out_155275.references = NULL;
    
    struct memblock mem_out_155274;
    
    mem_out_155274.references = NULL;
    
    struct memblock mem_out_155273;
    
    mem_out_155273.references = NULL;
    
    struct memblock mem_out_155272;
    
    mem_out_155272.references = NULL;
    
    struct memblock mem_out_155271;
    
    mem_out_155271.references = NULL;
    
    struct memblock mem_out_155270;
    
    mem_out_155270.references = NULL;
    
    struct memblock mem_out_155269;
    
    mem_out_155269.references = NULL;
    
    struct memblock wvoc_mem_152789;
    
    wvoc_mem_152789.references = NULL;
    
    struct memblock wdown_mem_152788;
    
    wdown_mem_152788.references = NULL;
    
    struct memblock wup_mem_152787;
    
    wup_mem_152787.references = NULL;
    
    struct memblock wout_mem_152786;
    
    wout_mem_152786.references = NULL;
    
    struct memblock wval_mem_152785;
    
    wval_mem_152785.references = NULL;
    
    struct memblock wkey_mem_152784;
    
    wkey_mem_152784.references = NULL;
    
    struct memblock wqry_mem_152783;
    
    wqry_mem_152783.references = NULL;
    
    struct memblock wpe_mem_152782;
    
    wpe_mem_152782.references = NULL;
    
    struct memblock wte_mem_152781;
    
    wte_mem_152781.references = NULL;
    wte_mem_152781 = in0->mem;
    wpe_mem_152782 = in1->mem;
    wqry_mem_152783 = in2->mem;
    wkey_mem_152784 = in3->mem;
    wval_mem_152785 = in4->mem;
    wout_mem_152786 = in5->mem;
    wup_mem_152787 = in6->mem;
    wdown_mem_152788 = in7->mem;
    wvoc_mem_152789 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_155269, &mem_out_155270, &mem_out_155271, &mem_out_155272, &mem_out_155273, &mem_out_155274, &mem_out_155275, &mem_out_155276, &mem_out_155277, wte_mem_152781, wpe_mem_152782, wqry_mem_152783, wkey_mem_152784, wval_mem_152785, wout_mem_152786, wup_mem_152787, wdown_mem_152788, wvoc_mem_152789);
        if (ret == 0) {
            struct memblock mem_152772 = ctx->constants->mem_152772;
            struct memblock mem_152773 = ctx->constants->mem_152773;
            struct memblock mem_152774 = ctx->constants->mem_152774;
            struct memblock mem_152775 = ctx->constants->mem_152775;
            struct memblock mem_152776 = ctx->constants->mem_152776;
            struct memblock mem_152777 = ctx->constants->mem_152777;
            struct memblock mem_152778 = ctx->constants->mem_152778;
            struct memblock mem_152779 = ctx->constants->mem_152779;
            struct memblock mem_152780 = ctx->constants->mem_152780;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_155269;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_155270;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_155271;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_155272;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_155273;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_155274;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_155275;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_155276;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_155277;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_train(struct futhark_context *ctx, struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const int64_t in0, const struct futhark_opaque_params *in1, const struct futhark_opaque_params *in2, const struct futhark_opaque_params *in3, const struct futhark_f64_3d *in4, const struct futhark_i64_1d *in5, const struct futhark_i64_2d *in6)
{
    int64_t num_steps_112428 = (int64_t) 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_155295;
    
    mem_out_155295.references = NULL;
    
    struct memblock mem_out_155294;
    
    mem_out_155294.references = NULL;
    
    struct memblock mem_out_155293;
    
    mem_out_155293.references = NULL;
    
    struct memblock mem_out_155292;
    
    mem_out_155292.references = NULL;
    
    struct memblock mem_out_155291;
    
    mem_out_155291.references = NULL;
    
    struct memblock mem_out_155290;
    
    mem_out_155290.references = NULL;
    
    struct memblock mem_out_155289;
    
    mem_out_155289.references = NULL;
    
    struct memblock mem_out_155288;
    
    mem_out_155288.references = NULL;
    
    struct memblock mem_out_155287;
    
    mem_out_155287.references = NULL;
    
    struct memblock mem_out_155286;
    
    mem_out_155286.references = NULL;
    
    struct memblock mem_out_155285;
    
    mem_out_155285.references = NULL;
    
    struct memblock mem_out_155284;
    
    mem_out_155284.references = NULL;
    
    struct memblock mem_out_155283;
    
    mem_out_155283.references = NULL;
    
    struct memblock mem_out_155282;
    
    mem_out_155282.references = NULL;
    
    struct memblock mem_out_155281;
    
    mem_out_155281.references = NULL;
    
    struct memblock mem_out_155280;
    
    mem_out_155280.references = NULL;
    
    struct memblock mem_out_155279;
    
    mem_out_155279.references = NULL;
    
    struct memblock mem_out_155278;
    
    mem_out_155278.references = NULL;
    
    struct memblock mem_out_155277;
    
    mem_out_155277.references = NULL;
    
    struct memblock mem_out_155276;
    
    mem_out_155276.references = NULL;
    
    struct memblock mem_out_155275;
    
    mem_out_155275.references = NULL;
    
    struct memblock mem_out_155274;
    
    mem_out_155274.references = NULL;
    
    struct memblock mem_out_155273;
    
    mem_out_155273.references = NULL;
    
    struct memblock mem_out_155272;
    
    mem_out_155272.references = NULL;
    
    struct memblock mem_out_155271;
    
    mem_out_155271.references = NULL;
    
    struct memblock mem_out_155270;
    
    mem_out_155270.references = NULL;
    
    struct memblock mem_out_155269;
    
    mem_out_155269.references = NULL;
    
    struct memblock seqs_mem_152810;
    
    seqs_mem_152810.references = NULL;
    
    struct memblock dls_mem_152809;
    
    dls_mem_152809.references = NULL;
    
    struct memblock masks_mem_152808;
    
    masks_mem_152808.references = NULL;
    
    struct memblock wvoc_mem_152807;
    
    wvoc_mem_152807.references = NULL;
    
    struct memblock wval_mem_152806;
    
    wval_mem_152806.references = NULL;
    
    struct memblock wup_mem_152805;
    
    wup_mem_152805.references = NULL;
    
    struct memblock wte_mem_152804;
    
    wte_mem_152804.references = NULL;
    
    struct memblock wqry_mem_152803;
    
    wqry_mem_152803.references = NULL;
    
    struct memblock wpe_mem_152802;
    
    wpe_mem_152802.references = NULL;
    
    struct memblock wout_mem_152801;
    
    wout_mem_152801.references = NULL;
    
    struct memblock wkey_mem_152800;
    
    wkey_mem_152800.references = NULL;
    
    struct memblock wdown_mem_152799;
    
    wdown_mem_152799.references = NULL;
    
    struct memblock wvoc_mem_152798;
    
    wvoc_mem_152798.references = NULL;
    
    struct memblock wval_mem_152797;
    
    wval_mem_152797.references = NULL;
    
    struct memblock wup_mem_152796;
    
    wup_mem_152796.references = NULL;
    
    struct memblock wte_mem_152795;
    
    wte_mem_152795.references = NULL;
    
    struct memblock wqry_mem_152794;
    
    wqry_mem_152794.references = NULL;
    
    struct memblock wpe_mem_152793;
    
    wpe_mem_152793.references = NULL;
    
    struct memblock wout_mem_152792;
    
    wout_mem_152792.references = NULL;
    
    struct memblock wkey_mem_152791;
    
    wkey_mem_152791.references = NULL;
    
    struct memblock wdown_mem_152790;
    
    wdown_mem_152790.references = NULL;
    
    struct memblock wvoc_mem_152789;
    
    wvoc_mem_152789.references = NULL;
    
    struct memblock wval_mem_152788;
    
    wval_mem_152788.references = NULL;
    
    struct memblock wup_mem_152787;
    
    wup_mem_152787.references = NULL;
    
    struct memblock wte_mem_152786;
    
    wte_mem_152786.references = NULL;
    
    struct memblock wqry_mem_152785;
    
    wqry_mem_152785.references = NULL;
    
    struct memblock wpe_mem_152784;
    
    wpe_mem_152784.references = NULL;
    
    struct memblock wout_mem_152783;
    
    wout_mem_152783.references = NULL;
    
    struct memblock wkey_mem_152782;
    
    wkey_mem_152782.references = NULL;
    
    struct memblock wdown_mem_152781;
    
    wdown_mem_152781.references = NULL;
    num_steps_112428 = in0;
    wdown_mem_152781 = in1->v0->mem;
    wkey_mem_152782 = in1->v1->mem;
    wout_mem_152783 = in1->v2->mem;
    wpe_mem_152784 = in1->v3->mem;
    wqry_mem_152785 = in1->v4->mem;
    wte_mem_152786 = in1->v5->mem;
    wup_mem_152787 = in1->v6->mem;
    wval_mem_152788 = in1->v7->mem;
    wvoc_mem_152789 = in1->v8->mem;
    wdown_mem_152790 = in2->v0->mem;
    wkey_mem_152791 = in2->v1->mem;
    wout_mem_152792 = in2->v2->mem;
    wpe_mem_152793 = in2->v3->mem;
    wqry_mem_152794 = in2->v4->mem;
    wte_mem_152795 = in2->v5->mem;
    wup_mem_152796 = in2->v6->mem;
    wval_mem_152797 = in2->v7->mem;
    wvoc_mem_152798 = in2->v8->mem;
    wdown_mem_152799 = in3->v0->mem;
    wkey_mem_152800 = in3->v1->mem;
    wout_mem_152801 = in3->v2->mem;
    wpe_mem_152802 = in3->v3->mem;
    wqry_mem_152803 = in3->v4->mem;
    wte_mem_152804 = in3->v5->mem;
    wup_mem_152805 = in3->v6->mem;
    wval_mem_152806 = in3->v7->mem;
    wvoc_mem_152807 = in3->v8->mem;
    masks_mem_152808 = in4->mem;
    dls_mem_152809 = in5->mem;
    seqs_mem_152810 = in6->mem;
    if (!(((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in3->v0->shape[0] && ((int64_t) 64 == in3->v0->shape[1] && ((int64_t) 16 == in3->v1->shape[0] && ((int64_t) 16 == in3->v1->shape[1] && ((int64_t) 16 == in3->v2->shape[0] && ((int64_t) 16 == in3->v2->shape[1] && ((int64_t) 16 == in3->v3->shape[0] && ((int64_t) 16 == in3->v3->shape[1] && ((int64_t) 16 == in3->v4->shape[0] && ((int64_t) 16 == in3->v4->shape[1] && ((int64_t) 27 == in3->v5->shape[0] && ((int64_t) 16 == in3->v5->shape[1] && ((int64_t) 64 == in3->v6->shape[0] && ((int64_t) 16 == in3->v6->shape[1] && ((int64_t) 16 == in3->v7->shape[0] && ((int64_t) 16 == in3->v7->shape[1] && ((int64_t) 27 == in3->v8->shape[0] && (int64_t) 16 == in3->v8->shape[1]))))))))))))))))) && ((num_steps_112428 == in4->shape[0] && ((int64_t) 16 == in4->shape[1] && (int64_t) 16 == in4->shape[2])) && (num_steps_112428 == in5->shape[0] && (num_steps_112428 == in6->shape[0] && (int64_t) 16 == in6->shape[1]))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_155269, &mem_out_155270, &mem_out_155271, &mem_out_155272, &mem_out_155273, &mem_out_155274, &mem_out_155275, &mem_out_155276, &mem_out_155277, &mem_out_155278, &mem_out_155279, &mem_out_155280, &mem_out_155281, &mem_out_155282, &mem_out_155283, &mem_out_155284, &mem_out_155285, &mem_out_155286, &mem_out_155287, &mem_out_155288, &mem_out_155289, &mem_out_155290, &mem_out_155291, &mem_out_155292, &mem_out_155293, &mem_out_155294, &mem_out_155295, wdown_mem_152781, wkey_mem_152782, wout_mem_152783, wpe_mem_152784, wqry_mem_152785, wte_mem_152786, wup_mem_152787, wval_mem_152788, wvoc_mem_152789, wdown_mem_152790, wkey_mem_152791, wout_mem_152792, wpe_mem_152793, wqry_mem_152794, wte_mem_152795, wup_mem_152796, wval_mem_152797, wvoc_mem_152798, wdown_mem_152799, wkey_mem_152800, wout_mem_152801, wpe_mem_152802, wqry_mem_152803, wte_mem_152804, wup_mem_152805, wval_mem_152806, wvoc_mem_152807, masks_mem_152808, dls_mem_152809, seqs_mem_152810, num_steps_112428);
        if (ret == 0) {
            struct memblock mem_152772 = ctx->constants->mem_152772;
            struct memblock mem_152773 = ctx->constants->mem_152773;
            struct memblock mem_152774 = ctx->constants->mem_152774;
            struct memblock mem_152775 = ctx->constants->mem_152775;
            struct memblock mem_152776 = ctx->constants->mem_152776;
            struct memblock mem_152777 = ctx->constants->mem_152777;
            struct memblock mem_152778 = ctx->constants->mem_152778;
            struct memblock mem_152779 = ctx->constants->mem_152779;
            struct memblock mem_152780 = ctx->constants->mem_152780;
            
            assert((*out = (struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_155269;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_155270;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_155271;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_155272;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_155273;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_155274;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_155275;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_155276;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_155277;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v9->mem = mem_out_155278;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v10->mem = mem_out_155279;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v11->mem = mem_out_155280;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v12->mem = mem_out_155281;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v13->mem = mem_out_155282;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v14->mem = mem_out_155283;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v15->mem = mem_out_155284;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v16->mem = mem_out_155285;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v17->mem = mem_out_155286;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v18->mem = mem_out_155287;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v19->mem = mem_out_155288;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v20->mem = mem_out_155289;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v21->mem = mem_out_155290;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v22->mem = mem_out_155291;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v23->mem = mem_out_155292;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v24->mem = mem_out_155293;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v25->mem = mem_out_155294;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v26->mem = mem_out_155295;
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
    
    struct memblock mem_out_155277;
    
    mem_out_155277.references = NULL;
    
    struct memblock mem_out_155276;
    
    mem_out_155276.references = NULL;
    
    struct memblock mem_out_155275;
    
    mem_out_155275.references = NULL;
    
    struct memblock mem_out_155274;
    
    mem_out_155274.references = NULL;
    
    struct memblock mem_out_155273;
    
    mem_out_155273.references = NULL;
    
    struct memblock mem_out_155272;
    
    mem_out_155272.references = NULL;
    
    struct memblock mem_out_155271;
    
    mem_out_155271.references = NULL;
    
    struct memblock mem_out_155270;
    
    mem_out_155270.references = NULL;
    
    struct memblock mem_out_155269;
    
    mem_out_155269.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_155269, &mem_out_155270, &mem_out_155271, &mem_out_155272, &mem_out_155273, &mem_out_155274, &mem_out_155275, &mem_out_155276, &mem_out_155277);
        if (ret == 0) {
            struct memblock mem_152772 = ctx->constants->mem_152772;
            struct memblock mem_152773 = ctx->constants->mem_152773;
            struct memblock mem_152774 = ctx->constants->mem_152774;
            struct memblock mem_152775 = ctx->constants->mem_152775;
            struct memblock mem_152776 = ctx->constants->mem_152776;
            struct memblock mem_152777 = ctx->constants->mem_152777;
            struct memblock mem_152778 = ctx->constants->mem_152778;
            struct memblock mem_152779 = ctx->constants->mem_152779;
            struct memblock mem_152780 = ctx->constants->mem_152780;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_155269;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_155270;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_155271;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_155272;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_155273;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_155274;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_155275;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_155276;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_155277;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
