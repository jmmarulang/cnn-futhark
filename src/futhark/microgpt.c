
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
    struct memblock mem_152234;
    struct memblock mem_152235;
    struct memblock mem_152236;
    struct memblock mem_152237;
    struct memblock mem_152238;
    struct memblock mem_152239;
    struct memblock mem_152240;
    struct memblock mem_152241;
    struct memblock mem_152242;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_12952(struct futhark_context *ctx, struct memblock *mem_out_p_155152, struct memblock *mem_out_p_155153, struct memblock *mem_out_p_155154, struct memblock w_mem_152243, struct memblock mw_mem_152244, struct memblock vw_mem_152245, struct memblock dw_mem_152246, int64_t n_110258, int64_t m_110259, int64_t step_110264, double lt_r_110265);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_12953(struct futhark_context *ctx, struct memblock *mem_out_p_155157, struct memblock *mem_out_p_155158, struct memblock *mem_out_p_155159, struct memblock w_mem_152243, struct memblock mw_mem_152244, struct memblock vw_mem_152245, struct memblock dw_mem_152246, int64_t n_111291, int64_t m_111292, int64_t step_111297, double lt_r_111298);
FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_155162, double *out_prim_out_155163, struct memblock wdown_mem_152243, struct memblock wkey_mem_152244, struct memblock wout_mem_152245, struct memblock wpe_mem_152246, struct memblock wqry_mem_152247, struct memblock wte_mem_152248, struct memblock wup_mem_152249, struct memblock wval_mem_152250, struct memblock wvoc_mem_152251, struct memblock tokens_mem_152252, struct memblock target_mem_152253, struct memblock mask_mem_152254);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_155221, struct memblock wdown_mem_152243, struct memblock wkey_mem_152244, struct memblock wout_mem_152245, struct memblock wpe_mem_152246, struct memblock wqry_mem_152247, struct memblock wte_mem_152248, struct memblock wup_mem_152249, struct memblock wval_mem_152250, struct memblock wvoc_mem_152251, struct memblock tokens_mem_152252, struct memblock mask_mem_152253);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_155278, struct memblock *mem_out_p_155279, struct memblock *mem_out_p_155280, struct memblock *mem_out_p_155281, struct memblock *mem_out_p_155282, struct memblock *mem_out_p_155283, struct memblock *mem_out_p_155284, struct memblock *mem_out_p_155285, struct memblock *mem_out_p_155286, struct memblock wte_mem_152243, struct memblock wpe_mem_152244, struct memblock wqry_mem_152245, struct memblock wkey_mem_152246, struct memblock wval_mem_152247, struct memblock wout_mem_152248, struct memblock wup_mem_152249, struct memblock wdown_mem_152250, struct memblock wvoc_mem_152251);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_155287, struct memblock *mem_out_p_155288, struct memblock *mem_out_p_155289, struct memblock *mem_out_p_155290, struct memblock *mem_out_p_155291, struct memblock *mem_out_p_155292, struct memblock *mem_out_p_155293, struct memblock *mem_out_p_155294, struct memblock *mem_out_p_155295, struct memblock *mem_out_p_155296, struct memblock *mem_out_p_155297, struct memblock *mem_out_p_155298, struct memblock *mem_out_p_155299, struct memblock *mem_out_p_155300, struct memblock *mem_out_p_155301, struct memblock *mem_out_p_155302, struct memblock *mem_out_p_155303, struct memblock *mem_out_p_155304, struct memblock *mem_out_p_155305, struct memblock *mem_out_p_155306, struct memblock *mem_out_p_155307, struct memblock *mem_out_p_155308, struct memblock *mem_out_p_155309, struct memblock *mem_out_p_155310, struct memblock *mem_out_p_155311, struct memblock *mem_out_p_155312, struct memblock *mem_out_p_155313, struct memblock wdown_mem_152243, struct memblock wkey_mem_152244, struct memblock wout_mem_152245, struct memblock wpe_mem_152246, struct memblock wqry_mem_152247, struct memblock wte_mem_152248, struct memblock wup_mem_152249, struct memblock wval_mem_152250, struct memblock wvoc_mem_152251, struct memblock wdown_mem_152252, struct memblock wkey_mem_152253, struct memblock wout_mem_152254, struct memblock wpe_mem_152255, struct memblock wqry_mem_152256, struct memblock wte_mem_152257, struct memblock wup_mem_152258, struct memblock wval_mem_152259, struct memblock wvoc_mem_152260, struct memblock wdown_mem_152261, struct memblock wkey_mem_152262, struct memblock wout_mem_152263, struct memblock wpe_mem_152264, struct memblock wqry_mem_152265, struct memblock wte_mem_152266, struct memblock wup_mem_152267, struct memblock wval_mem_152268, struct memblock wvoc_mem_152269, struct memblock masks_mem_152270, struct memblock dls_mem_152271, struct memblock seqs_mem_152272);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_155561, struct memblock *mem_out_p_155562, struct memblock *mem_out_p_155563, struct memblock *mem_out_p_155564, struct memblock *mem_out_p_155565, struct memblock *mem_out_p_155566, struct memblock *mem_out_p_155567, struct memblock *mem_out_p_155568, struct memblock *mem_out_p_155569);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_152234 (ctx->constants->mem_152234)
    #define mem_152235 (ctx->constants->mem_152235)
    #define mem_152236 (ctx->constants->mem_152236)
    #define mem_152237 (ctx->constants->mem_152237)
    #define mem_152238 (ctx->constants->mem_152238)
    #define mem_152239 (ctx->constants->mem_152239)
    #define mem_152240 (ctx->constants->mem_152240)
    #define mem_152241 (ctx->constants->mem_152241)
    #define mem_152242 (ctx->constants->mem_152242)
    mem_152234.references = NULL;
    mem_152235.references = NULL;
    mem_152236.references = NULL;
    mem_152237.references = NULL;
    mem_152238.references = NULL;
    mem_152239.references = NULL;
    mem_152240.references = NULL;
    mem_152241.references = NULL;
    mem_152242.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152234, (int64_t) 3456, "mem_152234")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_155134 = 0; nest_i_155134 < (int64_t) 27; nest_i_155134++) {
        for (int64_t nest_i_155135 = 0; nest_i_155135 < (int64_t) 16; nest_i_155135++) {
            ((double *) mem_152234.mem)[nest_i_155134 * (int64_t) 16 + nest_i_155135] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152235, (int64_t) 2048, "mem_152235")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_155136 = 0; nest_i_155136 < (int64_t) 16; nest_i_155136++) {
        for (int64_t nest_i_155137 = 0; nest_i_155137 < (int64_t) 16; nest_i_155137++) {
            ((double *) mem_152235.mem)[nest_i_155136 * (int64_t) 16 + nest_i_155137] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152236, (int64_t) 2048, "mem_152236")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_155138 = 0; nest_i_155138 < (int64_t) 16; nest_i_155138++) {
        for (int64_t nest_i_155139 = 0; nest_i_155139 < (int64_t) 16; nest_i_155139++) {
            ((double *) mem_152236.mem)[nest_i_155138 * (int64_t) 16 + nest_i_155139] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152237, (int64_t) 2048, "mem_152237")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_155140 = 0; nest_i_155140 < (int64_t) 16; nest_i_155140++) {
        for (int64_t nest_i_155141 = 0; nest_i_155141 < (int64_t) 16; nest_i_155141++) {
            ((double *) mem_152237.mem)[nest_i_155140 * (int64_t) 16 + nest_i_155141] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152238, (int64_t) 2048, "mem_152238")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_155142 = 0; nest_i_155142 < (int64_t) 16; nest_i_155142++) {
        for (int64_t nest_i_155143 = 0; nest_i_155143 < (int64_t) 16; nest_i_155143++) {
            ((double *) mem_152238.mem)[nest_i_155142 * (int64_t) 16 + nest_i_155143] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152239, (int64_t) 2048, "mem_152239")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_155144 = 0; nest_i_155144 < (int64_t) 16; nest_i_155144++) {
        for (int64_t nest_i_155145 = 0; nest_i_155145 < (int64_t) 16; nest_i_155145++) {
            ((double *) mem_152239.mem)[nest_i_155144 * (int64_t) 16 + nest_i_155145] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152240, (int64_t) 8192, "mem_152240")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_155146 = 0; nest_i_155146 < (int64_t) 64; nest_i_155146++) {
        for (int64_t nest_i_155147 = 0; nest_i_155147 < (int64_t) 16; nest_i_155147++) {
            ((double *) mem_152240.mem)[nest_i_155146 * (int64_t) 16 + nest_i_155147] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152241, (int64_t) 8192, "mem_152241")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_155148 = 0; nest_i_155148 < (int64_t) 16; nest_i_155148++) {
        for (int64_t nest_i_155149 = 0; nest_i_155149 < (int64_t) 64; nest_i_155149++) {
            ((double *) mem_152241.mem)[nest_i_155148 * (int64_t) 64 + nest_i_155149] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152242, (int64_t) 3456, "mem_152242")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_155150 = 0; nest_i_155150 < (int64_t) 27; nest_i_155150++) {
        for (int64_t nest_i_155151 = 0; nest_i_155151 < (int64_t) 16; nest_i_155151++) {
            ((double *) mem_152242.mem)[nest_i_155150 * (int64_t) 16 + nest_i_155151] = 0.0;
        }
    }
    #undef mem_152234
    #undef mem_152235
    #undef mem_152236
    #undef mem_152237
    #undef mem_152238
    #undef mem_152239
    #undef mem_152240
    #undef mem_152241
    #undef mem_152242
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_152234, "ctx->constants->mem_152234") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_152235, "ctx->constants->mem_152235") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_152236, "ctx->constants->mem_152236") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_152237, "ctx->constants->mem_152237") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_152238, "ctx->constants->mem_152238") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_152239, "ctx->constants->mem_152239") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_152240, "ctx->constants->mem_152240") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_152241, "ctx->constants->mem_152241") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_152242, "ctx->constants->mem_152242") != 0)
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_12952(struct futhark_context *ctx, struct memblock *mem_out_p_155152, struct memblock *mem_out_p_155153, struct memblock *mem_out_p_155154, struct memblock w_mem_152243, struct memblock mw_mem_152244, struct memblock vw_mem_152245, struct memblock dw_mem_152246, int64_t n_110258, int64_t m_110259, int64_t step_110264, double lt_r_110265)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_152287_cached_sizze_155155 = 0;
    unsigned char *mem_152287 = NULL;
    int64_t mem_152290_cached_sizze_155156 = 0;
    unsigned char *mem_152290 = NULL;
    struct memblock mem_152325;
    
    mem_152325.references = NULL;
    
    struct memblock mem_152252;
    
    mem_152252.references = NULL;
    
    struct memblock mem_152249;
    
    mem_152249.references = NULL;
    
    struct memblock mem_out_154733;
    
    mem_out_154733.references = NULL;
    
    struct memblock mem_out_154732;
    
    mem_out_154732.references = NULL;
    
    struct memblock mem_out_154731;
    
    mem_out_154731.references = NULL;
    
    struct memblock mem_152234 = ctx->constants->mem_152234;
    struct memblock mem_152235 = ctx->constants->mem_152235;
    struct memblock mem_152236 = ctx->constants->mem_152236;
    struct memblock mem_152237 = ctx->constants->mem_152237;
    struct memblock mem_152238 = ctx->constants->mem_152238;
    struct memblock mem_152239 = ctx->constants->mem_152239;
    struct memblock mem_152240 = ctx->constants->mem_152240;
    struct memblock mem_152241 = ctx->constants->mem_152241;
    struct memblock mem_152242 = ctx->constants->mem_152242;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_152247 = (int64_t) 8 * n_110258;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_152248 = m_110259 * binop_x_152247;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152249, bytes_152248, "mem_152249")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152252, bytes_152248, "mem_152252")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151102 = 0; i_151102 < n_110258; i_151102++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151095 = 0; i_151095 < m_110259; i_151095++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_140950 = ((double *) mw_mem_152244.mem)[i_151102 * m_110259 + i_151095];
            
            // futhark/microgpt.fut:476:10-20
            
            double zp_lhs_140951 = 0.85 * zt_rhs_140950;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_140952 = ((double *) dw_mem_152246.mem)[i_151102 * m_110259 + i_151095];
            
            // futhark/microgpt.fut:476:35-45
            
            double zp_rhs_140953 = 0.15000000000000002 * zt_rhs_140952;
            
            // futhark/microgpt.fut:476:21-45
            
            double lifted_lambda_res_140954 = zp_lhs_140951 + zp_rhs_140953;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_140961 = ((double *) vw_mem_152245.mem)[i_151102 * m_110259 + i_151095];
            
            // futhark/microgpt.fut:478:10-20
            
            double zp_lhs_140962 = 0.99 * zt_rhs_140961;
            
            // futhark/microgpt.fut:478:35-45
            
            double zt_lhs_140964 = 1.0000000000000009e-2 * zt_rhs_140952;
            
            // futhark/microgpt.fut:478:46-56
            
            double zp_rhs_140965 = zt_rhs_140952 * zt_lhs_140964;
            
            // futhark/microgpt.fut:478:21-56
            
            double lifted_lambda_res_140966 = zp_lhs_140962 + zp_rhs_140965;
            
            ((double *) mem_152249.mem)[i_151102 * m_110259 + i_151095] = lifted_lambda_res_140966;
            ((double *) mem_152252.mem)[i_151102 * m_110259 + i_151095] = lifted_lambda_res_140954;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_115204 = sitofp_i64_f64(step_110264);
    
    // futhark/microgpt.fut:480:54-57
    
    double ztzt_rhs_115205 = 1.0 + i64_res_115204;
    
    // futhark/microgpt.fut:480:30-57
    
    double zm_rhs_115206 = fpow64(0.85, ztzt_rhs_115205);
    
    // futhark/microgpt.fut:480:23-57
    
    double zs_rhs_115207 = 1.0 - zm_rhs_115206;
    
    // futhark/microgpt.fut:482:31-58
    
    double zm_rhs_115245 = fpow64(0.99, ztzt_rhs_115205);
    
    // futhark/microgpt.fut:482:23-58
    
    double zs_rhs_115246 = 1.0 - zm_rhs_115245;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_152287_cached_sizze_155155 < bytes_152248) {
        err = lexical_realloc(ctx, &mem_152287, &mem_152287_cached_sizze_155155, bytes_152248);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152290_cached_sizze_155156 < bytes_152248) {
        err = lexical_realloc(ctx, &mem_152290, &mem_152290_cached_sizze_155156, bytes_152248);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151116 = 0; i_151116 < n_110258; i_151116++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151109 = 0; i_151109 < m_110259; i_151109++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_140986 = ((double *) mem_152252.mem)[i_151116 * m_110259 + i_151109];
            
            // futhark/microgpt.fut:480:18-57
            
            double lifted_lambda_res_140987 = zs_lhs_140986 / zs_rhs_115207;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_140994 = ((double *) mem_152249.mem)[i_151116 * m_110259 + i_151109];
            
            // futhark/microgpt.fut:482:18-58
            
            double lifted_lambda_res_140995 = zs_lhs_140994 / zs_rhs_115246;
            
            ((double *) mem_152287)[i_151116 * m_110259 + i_151109] = lifted_lambda_res_140995;
            ((double *) mem_152290)[i_151116 * m_110259 + i_151109] = lifted_lambda_res_140987;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152325, bytes_152248, "mem_152325")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151125 = 0; i_151125 < n_110258; i_151125++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151121 = 0; i_151121 < m_110259; i_151121++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_114515 = ((double *) w_mem_152243.mem)[i_151125 * m_110259 + i_151121];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_114516 = ((double *) mem_152290)[i_151125 * m_110259 + i_151121];
            
            // futhark/microgpt.fut:484:21-34
            
            double zs_lhs_114517 = lt_r_110265 * zt_rhs_114516;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_114518 = ((double *) mem_152287)[i_151125 * m_110259 + i_151121];
            
            // futhark/microgpt.fut:484:51-57
            
            double zp_lhs_114519 = fpow64(ztzt_lhs_114518, 0.5);
            
            // futhark/microgpt.fut:484:59-71
            
            double zs_rhs_114520 = 1.0e-8 + zp_lhs_114519;
            
            // futhark/microgpt.fut:484:35-71
            
            double zm_rhs_114521 = zs_lhs_114517 / zs_rhs_114520;
            
            // futhark/microgpt.fut:484:13-71
            
            double lifted_lambda_res_114522 = zm_lhs_114515 - zm_rhs_114521;
            
            ((double *) mem_152325.mem)[i_151125 * m_110259 + i_151121] = lifted_lambda_res_114522;
        }
    }
    if (memblock_set(ctx, &mem_out_154731, &mem_152325, "mem_152325") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154732, &mem_152252, "mem_152252") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154733, &mem_152249, "mem_152249") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155152, &mem_out_154731, "mem_out_154731") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155153, &mem_out_154732, "mem_out_154732") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155154, &mem_out_154733, "mem_out_154733") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_152287);
        free(mem_152290);
        if (memblock_unref(ctx, &mem_152325, "mem_152325") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_152252, "mem_152252") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_152249, "mem_152249") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154733, "mem_out_154733") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154732, "mem_out_154732") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154731, "mem_out_154731") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_12953(struct futhark_context *ctx, struct memblock *mem_out_p_155157, struct memblock *mem_out_p_155158, struct memblock *mem_out_p_155159, struct memblock w_mem_152243, struct memblock mw_mem_152244, struct memblock vw_mem_152245, struct memblock dw_mem_152246, int64_t n_111291, int64_t m_111292, int64_t step_111297, double lt_r_111298)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_152287_cached_sizze_155160 = 0;
    unsigned char *mem_152287 = NULL;
    int64_t mem_152290_cached_sizze_155161 = 0;
    unsigned char *mem_152290 = NULL;
    struct memblock mem_152325;
    
    mem_152325.references = NULL;
    
    struct memblock mem_152252;
    
    mem_152252.references = NULL;
    
    struct memblock mem_152249;
    
    mem_152249.references = NULL;
    
    struct memblock mem_out_154733;
    
    mem_out_154733.references = NULL;
    
    struct memblock mem_out_154732;
    
    mem_out_154732.references = NULL;
    
    struct memblock mem_out_154731;
    
    mem_out_154731.references = NULL;
    
    struct memblock mem_152234 = ctx->constants->mem_152234;
    struct memblock mem_152235 = ctx->constants->mem_152235;
    struct memblock mem_152236 = ctx->constants->mem_152236;
    struct memblock mem_152237 = ctx->constants->mem_152237;
    struct memblock mem_152238 = ctx->constants->mem_152238;
    struct memblock mem_152239 = ctx->constants->mem_152239;
    struct memblock mem_152240 = ctx->constants->mem_152240;
    struct memblock mem_152241 = ctx->constants->mem_152241;
    struct memblock mem_152242 = ctx->constants->mem_152242;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_152247 = (int64_t) 8 * n_111291;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_152248 = m_111292 * binop_x_152247;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152249, bytes_152248, "mem_152249")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152252, bytes_152248, "mem_152252")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151102 = 0; i_151102 < n_111291; i_151102++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151095 = 0; i_151095 < m_111292; i_151095++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_140950 = ((double *) mw_mem_152244.mem)[i_151102 * m_111292 + i_151095];
            
            // futhark/microgpt.fut:476:10-20
            
            double zp_lhs_140951 = 0.85 * zt_rhs_140950;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_140952 = ((double *) dw_mem_152246.mem)[i_151102 * m_111292 + i_151095];
            
            // futhark/microgpt.fut:476:35-45
            
            double zp_rhs_140953 = 0.15000000000000002 * zt_rhs_140952;
            
            // futhark/microgpt.fut:476:21-45
            
            double lifted_lambda_res_140954 = zp_lhs_140951 + zp_rhs_140953;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_140961 = ((double *) vw_mem_152245.mem)[i_151102 * m_111292 + i_151095];
            
            // futhark/microgpt.fut:478:10-20
            
            double zp_lhs_140962 = 0.99 * zt_rhs_140961;
            
            // futhark/microgpt.fut:478:35-45
            
            double zt_lhs_140964 = 1.0000000000000009e-2 * zt_rhs_140952;
            
            // futhark/microgpt.fut:478:46-56
            
            double zp_rhs_140965 = zt_rhs_140952 * zt_lhs_140964;
            
            // futhark/microgpt.fut:478:21-56
            
            double lifted_lambda_res_140966 = zp_lhs_140962 + zp_rhs_140965;
            
            ((double *) mem_152249.mem)[i_151102 * m_111292 + i_151095] = lifted_lambda_res_140966;
            ((double *) mem_152252.mem)[i_151102 * m_111292 + i_151095] = lifted_lambda_res_140954;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_115204 = sitofp_i64_f64(step_111297);
    
    // futhark/microgpt.fut:480:54-57
    
    double ztzt_rhs_115205 = 1.0 + i64_res_115204;
    
    // futhark/microgpt.fut:480:30-57
    
    double zm_rhs_115206 = fpow64(0.85, ztzt_rhs_115205);
    
    // futhark/microgpt.fut:480:23-57
    
    double zs_rhs_115207 = 1.0 - zm_rhs_115206;
    
    // futhark/microgpt.fut:482:31-58
    
    double zm_rhs_115245 = fpow64(0.99, ztzt_rhs_115205);
    
    // futhark/microgpt.fut:482:23-58
    
    double zs_rhs_115246 = 1.0 - zm_rhs_115245;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_152287_cached_sizze_155160 < bytes_152248) {
        err = lexical_realloc(ctx, &mem_152287, &mem_152287_cached_sizze_155160, bytes_152248);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152290_cached_sizze_155161 < bytes_152248) {
        err = lexical_realloc(ctx, &mem_152290, &mem_152290_cached_sizze_155161, bytes_152248);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151116 = 0; i_151116 < n_111291; i_151116++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151109 = 0; i_151109 < m_111292; i_151109++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_140986 = ((double *) mem_152252.mem)[i_151116 * m_111292 + i_151109];
            
            // futhark/microgpt.fut:480:18-57
            
            double lifted_lambda_res_140987 = zs_lhs_140986 / zs_rhs_115207;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_140994 = ((double *) mem_152249.mem)[i_151116 * m_111292 + i_151109];
            
            // futhark/microgpt.fut:482:18-58
            
            double lifted_lambda_res_140995 = zs_lhs_140994 / zs_rhs_115246;
            
            ((double *) mem_152287)[i_151116 * m_111292 + i_151109] = lifted_lambda_res_140995;
            ((double *) mem_152290)[i_151116 * m_111292 + i_151109] = lifted_lambda_res_140987;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152325, bytes_152248, "mem_152325")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151125 = 0; i_151125 < n_111291; i_151125++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151121 = 0; i_151121 < m_111292; i_151121++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_114515 = ((double *) w_mem_152243.mem)[i_151125 * m_111292 + i_151121];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_114516 = ((double *) mem_152290)[i_151125 * m_111292 + i_151121];
            
            // futhark/microgpt.fut:484:21-34
            
            double zs_lhs_114517 = lt_r_111298 * zt_rhs_114516;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_114518 = ((double *) mem_152287)[i_151125 * m_111292 + i_151121];
            
            // futhark/microgpt.fut:484:51-57
            
            double zp_lhs_114519 = fpow64(ztzt_lhs_114518, 0.5);
            
            // futhark/microgpt.fut:484:59-71
            
            double zs_rhs_114520 = 1.0e-8 + zp_lhs_114519;
            
            // futhark/microgpt.fut:484:35-71
            
            double zm_rhs_114521 = zs_lhs_114517 / zs_rhs_114520;
            
            // futhark/microgpt.fut:484:13-71
            
            double lifted_lambda_res_114522 = zm_lhs_114515 - zm_rhs_114521;
            
            ((double *) mem_152325.mem)[i_151125 * m_111292 + i_151121] = lifted_lambda_res_114522;
        }
    }
    if (memblock_set(ctx, &mem_out_154731, &mem_152325, "mem_152325") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154732, &mem_152252, "mem_152252") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154733, &mem_152249, "mem_152249") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155157, &mem_out_154731, "mem_out_154731") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155158, &mem_out_154732, "mem_out_154732") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155159, &mem_out_154733, "mem_out_154733") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_152287);
        free(mem_152290);
        if (memblock_unref(ctx, &mem_152325, "mem_152325") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_152252, "mem_152252") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_152249, "mem_152249") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154733, "mem_out_154733") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154732, "mem_out_154732") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154731, "mem_out_154731") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_155162, double *out_prim_out_155163, struct memblock wdown_mem_152243, struct memblock wkey_mem_152244, struct memblock wout_mem_152245, struct memblock wpe_mem_152246, struct memblock wqry_mem_152247, struct memblock wte_mem_152248, struct memblock wup_mem_152249, struct memblock wval_mem_152250, struct memblock wvoc_mem_152251, struct memblock tokens_mem_152252, struct memblock target_mem_152253, struct memblock mask_mem_152254)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_152255_cached_sizze_155164 = 0;
    unsigned char *mem_152255 = NULL;
    int64_t mem_152260_cached_sizze_155165 = 0;
    unsigned char *mem_152260 = NULL;
    int64_t mem_152271_cached_sizze_155166 = 0;
    unsigned char *mem_152271 = NULL;
    int64_t mem_152276_cached_sizze_155167 = 0;
    unsigned char *mem_152276 = NULL;
    int64_t mem_152283_cached_sizze_155168 = 0;
    unsigned char *mem_152283 = NULL;
    int64_t mem_152294_cached_sizze_155169 = 0;
    unsigned char *mem_152294 = NULL;
    int64_t mem_152299_cached_sizze_155170 = 0;
    unsigned char *mem_152299 = NULL;
    int64_t mem_152306_cached_sizze_155171 = 0;
    unsigned char *mem_152306 = NULL;
    int64_t mem_152317_cached_sizze_155172 = 0;
    unsigned char *mem_152317 = NULL;
    int64_t mem_152318_cached_sizze_155173 = 0;
    unsigned char *mem_152318 = NULL;
    int64_t mem_152319_cached_sizze_155174 = 0;
    unsigned char *mem_152319 = NULL;
    int64_t mem_152332_cached_sizze_155175 = 0;
    unsigned char *mem_152332 = NULL;
    int64_t mem_152333_cached_sizze_155176 = 0;
    unsigned char *mem_152333 = NULL;
    int64_t mem_152334_cached_sizze_155177 = 0;
    unsigned char *mem_152334 = NULL;
    int64_t mem_152365_cached_sizze_155178 = 0;
    unsigned char *mem_152365 = NULL;
    int64_t mem_152366_cached_sizze_155179 = 0;
    unsigned char *mem_152366 = NULL;
    int64_t mem_152367_cached_sizze_155180 = 0;
    unsigned char *mem_152367 = NULL;
    int64_t mem_152383_cached_sizze_155181 = 0;
    unsigned char *mem_152383 = NULL;
    int64_t mem_152384_cached_sizze_155182 = 0;
    unsigned char *mem_152384 = NULL;
    int64_t mem_152385_cached_sizze_155183 = 0;
    unsigned char *mem_152385 = NULL;
    int64_t mem_152398_cached_sizze_155184 = 0;
    unsigned char *mem_152398 = NULL;
    int64_t mem_152399_cached_sizze_155185 = 0;
    unsigned char *mem_152399 = NULL;
    int64_t mem_152400_cached_sizze_155186 = 0;
    unsigned char *mem_152400 = NULL;
    int64_t mem_152446_cached_sizze_155187 = 0;
    unsigned char *mem_152446 = NULL;
    int64_t mem_152452_cached_sizze_155188 = 0;
    unsigned char *mem_152452 = NULL;
    int64_t mem_152457_cached_sizze_155189 = 0;
    unsigned char *mem_152457 = NULL;
    int64_t mem_152468_cached_sizze_155190 = 0;
    unsigned char *mem_152468 = NULL;
    int64_t mem_152473_cached_sizze_155191 = 0;
    unsigned char *mem_152473 = NULL;
    int64_t mem_152484_cached_sizze_155192 = 0;
    unsigned char *mem_152484 = NULL;
    int64_t mem_152489_cached_sizze_155193 = 0;
    unsigned char *mem_152489 = NULL;
    int64_t mem_152496_cached_sizze_155194 = 0;
    unsigned char *mem_152496 = NULL;
    int64_t mem_152503_cached_sizze_155195 = 0;
    unsigned char *mem_152503 = NULL;
    int64_t mem_152514_cached_sizze_155196 = 0;
    unsigned char *mem_152514 = NULL;
    int64_t mem_152519_cached_sizze_155197 = 0;
    unsigned char *mem_152519 = NULL;
    int64_t mem_152530_cached_sizze_155198 = 0;
    unsigned char *mem_152530 = NULL;
    int64_t mem_152535_cached_sizze_155199 = 0;
    unsigned char *mem_152535 = NULL;
    int64_t mem_152551_cached_sizze_155200 = 0;
    unsigned char *mem_152551 = NULL;
    int64_t mem_152556_cached_sizze_155201 = 0;
    unsigned char *mem_152556 = NULL;
    int64_t mem_152567_cached_sizze_155202 = 0;
    unsigned char *mem_152567 = NULL;
    int64_t mem_152572_cached_sizze_155203 = 0;
    unsigned char *mem_152572 = NULL;
    int64_t mem_152583_cached_sizze_155204 = 0;
    unsigned char *mem_152583 = NULL;
    int64_t mem_152588_cached_sizze_155205 = 0;
    unsigned char *mem_152588 = NULL;
    int64_t mem_152599_cached_sizze_155206 = 0;
    unsigned char *mem_152599 = NULL;
    int64_t mem_152604_cached_sizze_155207 = 0;
    unsigned char *mem_152604 = NULL;
    int64_t mem_152611_cached_sizze_155208 = 0;
    unsigned char *mem_152611 = NULL;
    int64_t mem_152622_cached_sizze_155209 = 0;
    unsigned char *mem_152622 = NULL;
    int64_t mem_152627_cached_sizze_155210 = 0;
    unsigned char *mem_152627 = NULL;
    int64_t mem_152638_cached_sizze_155211 = 0;
    unsigned char *mem_152638 = NULL;
    int64_t mem_152643_cached_sizze_155212 = 0;
    unsigned char *mem_152643 = NULL;
    int64_t mem_152654_cached_sizze_155213 = 0;
    unsigned char *mem_152654 = NULL;
    int64_t mem_152659_cached_sizze_155214 = 0;
    unsigned char *mem_152659 = NULL;
    int64_t mem_152670_cached_sizze_155215 = 0;
    unsigned char *mem_152670 = NULL;
    int64_t mem_152675_cached_sizze_155216 = 0;
    unsigned char *mem_152675 = NULL;
    int64_t mem_152686_cached_sizze_155217 = 0;
    unsigned char *mem_152686 = NULL;
    int64_t mem_152691_cached_sizze_155218 = 0;
    unsigned char *mem_152691 = NULL;
    int64_t mem_152706_cached_sizze_155219 = 0;
    unsigned char *mem_152706 = NULL;
    int64_t mem_152713_cached_sizze_155220 = 0;
    unsigned char *mem_152713 = NULL;
    struct memblock mem_152702;
    
    mem_152702.references = NULL;
    
    struct memblock mem_out_154731;
    
    mem_out_154731.references = NULL;
    
    struct memblock mem_152234 = ctx->constants->mem_152234;
    struct memblock mem_152235 = ctx->constants->mem_152235;
    struct memblock mem_152236 = ctx->constants->mem_152236;
    struct memblock mem_152237 = ctx->constants->mem_152237;
    struct memblock mem_152238 = ctx->constants->mem_152238;
    struct memblock mem_152239 = ctx->constants->mem_152239;
    struct memblock mem_152240 = ctx->constants->mem_152240;
    struct memblock mem_152241 = ctx->constants->mem_152241;
    struct memblock mem_152242 = ctx->constants->mem_152242;
    double prim_out_154732;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_152255_cached_sizze_155164 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152255, &mem_152255_cached_sizze_155164, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152260_cached_sizze_155165 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152260, &mem_152260_cached_sizze_155165, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151097 = 0; i_151097 < (int64_t) 16; i_151097++) {
        // futhark/microgpt.fut:466:41-50
        
        int64_t tmp_140231 = ((int64_t *) tokens_mem_152252.mem)[i_151097];
        
        // futhark/microgpt.fut:466:37-51
        
        bool x_140232 = sle64((int64_t) 0, tmp_140231);
        
        // futhark/microgpt.fut:466:37-51
        
        bool y_140233 = slt64(tmp_140231, (int64_t) 27);
        
        // futhark/microgpt.fut:466:37-51
        
        bool bounds_check_140234 = x_140232 && y_140233;
        
        // futhark/microgpt.fut:466:37-51
        
        bool index_certs_140235;
        
        if (!bounds_check_140234) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_140231, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:466:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:466:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151093 = 0; i_151093 < (int64_t) 16; i_151093++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_140242 = ((double *) wte_mem_152248.mem)[tmp_140231 * (int64_t) 16 + i_151093];
            
            ((double *) mem_152260)[i_151093] = lifted_lambda_res_140242;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152255, i_151097 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152260, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152271_cached_sizze_155166 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152271, &mem_152271_cached_sizze_155166, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152276_cached_sizze_155167 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152276, &mem_152276_cached_sizze_155167, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152283_cached_sizze_155168 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152283, &mem_152283_cached_sizze_155168, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151109 = 0; i_151109 < (int64_t) 16; i_151109++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_140268;
        double r_140270 = 0.0;
        
        for (int64_t i_140269 = 0; i_140269 < (int64_t) 16; i_140269++) {
            // futhark/microgpt.fut:71:46-49
            
            double zp_lhs_140271 = ((double *) wpe_mem_152246.mem)[i_151109 * (int64_t) 16 + i_140269];
            
            // futhark/microgpt.fut:71:46-49
            
            double zp_rhs_140272 = ((double *) mem_152255)[i_151109 * (int64_t) 16 + i_140269];
            
            // futhark/microgpt.fut:203:76-116
            
            double zp_res_140273 = zp_lhs_140271 + zp_rhs_140272;
            
            // futhark/microgpt.fut:203:94-163
            
            double zt_res_140274 = zp_res_140273 * zp_res_140273;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_140275 = r_140270 + zt_res_140274;
            double r_tmp_154736 = zp_res_140275;
            
            r_140270 = r_tmp_154736;
        }
        defunc_0_lifted_lambda_res_140268 = r_140270;
        // futhark/microgpt.fut:203:54-182
        
        double zs_res_140276 = defunc_0_lifted_lambda_res_140268 / 16.0;
        
        // futhark/microgpt.fut:204:24-55
        
        double zp_res_140277 = 1.0e-5 + zs_res_140276;
        
        // futhark/microgpt.fut:204:16-55
        
        double sqrt_res_140278 = futrts_sqrt64(zp_res_140277);
        
        // futhark/microgpt.fut:205:85-96
        
        double zs_res_140279 = 1.0 / sqrt_res_140278;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151101 = 0; i_151101 < (int64_t) 16; i_151101++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_140286 = ((double *) wpe_mem_152246.mem)[i_151109 * (int64_t) 16 + i_151101];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_140287 = ((double *) mem_152255)[i_151109 * (int64_t) 16 + i_151101];
            
            // futhark/microgpt.fut:205:38-78
            
            double zp_res_140288 = zp_lhs_140286 + zp_rhs_140287;
            
            // futhark/microgpt.fut:205:56-96
            
            double zt_res_140289 = zs_res_140279 * zp_res_140288;
            
            ((double *) mem_152276)[i_151101] = zt_res_140289;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151105 = 0; i_151105 < (int64_t) 16; i_151105++) {
            // futhark/microgpt.fut:206:4-14
            
            double lifted_lambda_res_140297 = ((double *) mem_152276)[i_151105];
            
            ((double *) mem_152283)[i_151105] = lifted_lambda_res_140297;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152271, i_151109 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152283, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152294_cached_sizze_155169 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152294, &mem_152294_cached_sizze_155169, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152299_cached_sizze_155170 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152299, &mem_152299_cached_sizze_155170, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152306_cached_sizze_155171 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152306, &mem_152306_cached_sizze_155171, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151121 = 0; i_151121 < (int64_t) 16; i_151121++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_140306;
        double r_140308 = 0.0;
        
        for (int64_t i_140307 = 0; i_140307 < (int64_t) 16; i_140307++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_140309 = ((double *) mem_152271)[i_151121 * (int64_t) 16 + i_140307];
            
            // futhark/microgpt.fut:207:78-115
            
            double zt_res_140310 = zt_lhs_140309 * zt_lhs_140309;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_140311 = r_140308 + zt_res_140310;
            double r_tmp_154740 = zp_res_140311;
            
            r_140308 = r_tmp_154740;
        }
        defunc_0_lifted_lambda_res_140306 = r_140308;
        // futhark/microgpt.fut:207:57-133
        
        double zs_res_140312 = defunc_0_lifted_lambda_res_140306 / 16.0;
        
        // futhark/microgpt.fut:208:24-55
        
        double zp_res_140313 = 1.0e-5 + zs_res_140312;
        
        // futhark/microgpt.fut:208:16-55
        
        double sqrt_res_140314 = futrts_sqrt64(zp_res_140313);
        
        // futhark/microgpt.fut:209:59-70
        
        double zs_res_140315 = 1.0 / sqrt_res_140314;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151113 = 0; i_151113 < (int64_t) 16; i_151113++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_140322 = ((double *) mem_152271)[i_151121 * (int64_t) 16 + i_151113];
            
            // futhark/microgpt.fut:209:37-70
            
            double zt_res_140323 = zs_res_140315 * zt_lhs_140322;
            
            ((double *) mem_152299)[i_151113] = zt_res_140323;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151117 = 0; i_151117 < (int64_t) 16; i_151117++) {
            // futhark/microgpt.fut:210:4-14
            
            double lifted_lambda_res_140331 = ((double *) mem_152299)[i_151117];
            
            ((double *) mem_152306)[i_151117] = lifted_lambda_res_140331;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152294, i_151121 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152306, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152317_cached_sizze_155172 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152317, &mem_152317_cached_sizze_155172, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152318_cached_sizze_155173 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152318, &mem_152318_cached_sizze_155173, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152319_cached_sizze_155174 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152319, &mem_152319_cached_sizze_155174, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152332_cached_sizze_155175 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152332, &mem_152332_cached_sizze_155175, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152333_cached_sizze_155176 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152333, &mem_152333_cached_sizze_155176, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152334_cached_sizze_155177 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152334, &mem_152334_cached_sizze_155177, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151139 = 0; i_151139 < (int64_t) 16; i_151139++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151129 = 0; i_151129 < (int64_t) 16; i_151129++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_141168;
            double r_141170 = 0.0;
            
            for (int64_t i_141169 = 0; i_141169 < (int64_t) 16; i_141169++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_141171 = ((double *) wqry_mem_152247.mem)[i_151129 * (int64_t) 16 + i_141169];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_141172 = ((double *) mem_152294)[i_151139 * (int64_t) 16 + i_141169];
                
                // futhark/microgpt.fut:211:66-105
                
                double zt_res_141173 = zt_lhs_141171 * zt_rhs_141172;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_141174 = r_141170 + zt_res_141173;
                double r_tmp_154749 = zp_res_141174;
                
                r_141170 = r_tmp_154749;
            }
            defunc_0_lifted_lambda_res_141168 = r_141170;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_141181;
            double r_141183 = 0.0;
            
            for (int64_t i_141182 = 0; i_141182 < (int64_t) 16; i_141182++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_141184 = ((double *) wkey_mem_152244.mem)[i_151129 * (int64_t) 16 + i_141182];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_141185 = ((double *) mem_152294)[i_151139 * (int64_t) 16 + i_141182];
                
                // futhark/microgpt.fut:212:66-105
                
                double zt_res_141186 = zt_lhs_141184 * zt_rhs_141185;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_141187 = r_141183 + zt_res_141186;
                double r_tmp_154750 = zp_res_141187;
                
                r_141183 = r_tmp_154750;
            }
            defunc_0_lifted_lambda_res_141181 = r_141183;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_141197;
            double r_141199 = 0.0;
            
            for (int64_t i_141198 = 0; i_141198 < (int64_t) 16; i_141198++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_141200 = ((double *) wval_mem_152250.mem)[i_151129 * (int64_t) 16 + i_141198];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_141201 = ((double *) mem_152294)[i_151139 * (int64_t) 16 + i_141198];
                
                // futhark/microgpt.fut:213:66-105
                
                double zt_res_141202 = zt_lhs_141200 * zt_rhs_141201;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_141203 = r_141199 + zt_res_141202;
                double r_tmp_154751 = zp_res_141203;
                
                r_141199 = r_tmp_154751;
            }
            defunc_0_lifted_lambda_res_141197 = r_141199;
            ((double *) mem_152332)[i_151129] = defunc_0_lifted_lambda_res_141197;
            ((double *) mem_152333)[i_151129] = defunc_0_lifted_lambda_res_141181;
            ((double *) mem_152334)[i_151129] = defunc_0_lifted_lambda_res_141168;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152317, i_151139 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152332, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152318, i_151139 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152333, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152319, i_151139 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152334, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152365_cached_sizze_155178 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152365, &mem_152365_cached_sizze_155178, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152366_cached_sizze_155179 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152366, &mem_152366_cached_sizze_155179, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152367_cached_sizze_155180 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152367, &mem_152367_cached_sizze_155180, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152383_cached_sizze_155181 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152383, &mem_152383_cached_sizze_155181, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152384_cached_sizze_155182 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152384, &mem_152384_cached_sizze_155182, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152385_cached_sizze_155183 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152385, &mem_152385_cached_sizze_155183, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152398_cached_sizze_155184 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152398, &mem_152398_cached_sizze_155184, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152399_cached_sizze_155185 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152399, &mem_152399_cached_sizze_155185, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152400_cached_sizze_155186 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152400, &mem_152400_cached_sizze_155186, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151169 = 0; i_151169 < (int64_t) 4; i_151169++) {
        // futhark/microgpt.fut:214:69-72
        
        int64_t zp_lhs_141044 = mul64((int64_t) 4, i_151169);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151159 = 0; i_151159 < (int64_t) 16; i_151159++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151149 = 0; i_151149 < (int64_t) 4; i_151149++) {
                // futhark/microgpt.fut:214:74-81
                
                int64_t tmp_141361 = add64(zp_lhs_141044, i_151149);
                
                // futhark/microgpt.fut:214:51-83
                
                bool x_141362 = sle64((int64_t) 0, tmp_141361);
                
                // futhark/microgpt.fut:214:51-83
                
                bool y_141363 = slt64(tmp_141361, (int64_t) 16);
                
                // futhark/microgpt.fut:214:51-83
                
                bool bounds_check_141364 = x_141362 && y_141363;
                
                // futhark/microgpt.fut:214:51-83
                
                bool index_certs_141365;
                
                if (!bounds_check_141364) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_141361, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:214:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:214:15-84\n   #9  futhark/microgpt.fut:467:7-76\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141366 = ((double *) mem_152319)[i_151159 * (int64_t) 16 + tmp_141361];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141374 = ((double *) mem_152318)[i_151159 * (int64_t) 16 + tmp_141361];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141385 = ((double *) mem_152317)[i_151159 * (int64_t) 16 + tmp_141361];
                
                ((double *) mem_152398)[i_151149] = lifted_lambda_res_141385;
                ((double *) mem_152399)[i_151149] = lifted_lambda_res_141374;
                ((double *) mem_152400)[i_151149] = lifted_lambda_res_141366;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152383, i_151159 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152398, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152384, i_151159 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152399, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152385, i_151159 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152400, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_152365, i_151169 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_152383, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_152366, i_151169 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_152384, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_152367, i_151169 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_152385, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152446_cached_sizze_155187 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152446, &mem_152446_cached_sizze_155187, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152452_cached_sizze_155188 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152452, &mem_152452_cached_sizze_155188, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152457_cached_sizze_155189 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152457, &mem_152457_cached_sizze_155189, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152468_cached_sizze_155190 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152468, &mem_152468_cached_sizze_155190, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152473_cached_sizze_155191 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152473, &mem_152473_cached_sizze_155191, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152484_cached_sizze_155192 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152484, &mem_152484_cached_sizze_155192, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152489_cached_sizze_155193 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152489, &mem_152489_cached_sizze_155193, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152496_cached_sizze_155194 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152496, &mem_152496_cached_sizze_155194, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152503_cached_sizze_155195 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152503, &mem_152503_cached_sizze_155195, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152514_cached_sizze_155196 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152514, &mem_152514_cached_sizze_155196, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152519_cached_sizze_155197 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152519, &mem_152519_cached_sizze_155197, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152530_cached_sizze_155198 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152530, &mem_152530_cached_sizze_155198, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152535_cached_sizze_155199 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152535, &mem_152535_cached_sizze_155199, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151225 = 0; i_151225 < (int64_t) 4; i_151225++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151179 = 0; i_151179 < (int64_t) 16; i_151179++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151175 = 0; i_151175 < (int64_t) 16; i_151175++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_140476;
                double r_140478 = 0.0;
                
                for (int64_t i_140477 = 0; i_140477 < (int64_t) 4; i_140477++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_140479 = ((double *) mem_152367)[i_151225 * (int64_t) 64 + i_151179 * (int64_t) 4 + i_140477];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_140480 = ((double *) mem_152366)[i_151225 * (int64_t) 64 + i_151175 * (int64_t) 4 + i_140477];
                    
                    // futhark/microgpt.fut:217:113-164
                    
                    double zt_res_140481 = zt_lhs_140479 * zt_rhs_140480;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_140482 = r_140478 + zt_res_140481;
                    double r_tmp_154764 = zp_res_140482;
                    
                    r_140478 = r_tmp_154764;
                }
                defunc_0_lifted_lambda_res_140476 = r_140478;
                ((double *) mem_152457)[i_151175] = defunc_0_lifted_lambda_res_140476;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152452, i_151179 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152457, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151187 = 0; i_151187 < (int64_t) 16; i_151187++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151183 = 0; i_151183 < (int64_t) 16; i_151183++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_140497 = ((double *) mem_152452)[i_151187 * (int64_t) 16 + i_151183];
                
                // futhark/microgpt.fut:218:47-78
                
                double zs_res_140498 = zs_lhs_140497 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_140499 = ((double *) mask_mem_152254.mem)[i_151187 * (int64_t) 16 + i_151183];
                
                // futhark/microgpt.fut:218:65-102
                
                double zp_res_140500 = zs_res_140498 + zp_rhs_140499;
                
                ((double *) mem_152473)[i_151183] = zp_res_140500;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152468, i_151187 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152473, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151205 = 0; i_151205 < (int64_t) 16; i_151205++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_141488;
            double redout_151189 = -INFINITY;
            
            for (int64_t i_151190 = 0; i_151190 < (int64_t) 16; i_151190++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141412 = ((double *) mem_152468)[i_151205 * (int64_t) 16 + i_151190];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_140521 = fmax64(lifted_lambda_res_141412, redout_151189);
                double redout_tmp_154768 = max_res_140521;
                
                redout_151189 = redout_tmp_154768;
            }
            defunc_0_reduce_res_141488 = redout_151189;
            // futhark/microgpt.fut:220:67-76
            
            double neg_res_140522 = -defunc_0_reduce_res_141488;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151193 = 0; i_151193 < (int64_t) 16; i_151193++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_140529 = ((double *) mem_152468)[i_151205 * (int64_t) 16 + i_151193];
                
                // futhark/microgpt.fut:220:44-76
                
                double zp_res_140530 = neg_res_140522 + zp_lhs_140529;
                
                // futhark/microgpt.fut:220:37-76
                
                double exp_res_140531 = futrts_exp64(zp_res_140530);
                
                ((double *) mem_152489)[i_151193] = exp_res_140531;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_140533;
            double r_140535 = 0.0;
            
            for (int64_t i_140534 = 0; i_140534 < (int64_t) 16; i_140534++) {
                // futhark/microgpt.fut:221:36-46
                
                double lifted_lambda_res_140536 = ((double *) mem_152489)[i_140534];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_140537 = r_140535 + lifted_lambda_res_140536;
                double r_tmp_154770 = zp_res_140537;
                
                r_140535 = r_tmp_154770;
            }
            defunc_0_lifted_lambda_res_140533 = r_140535;
            // futhark/microgpt.fut:222:53-64
            
            double zs_res_140538 = 1.0 / defunc_0_lifted_lambda_res_140533;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151197 = 0; i_151197 < (int64_t) 16; i_151197++) {
                // futhark/microgpt.fut:222:37-47
                
                double zt_lhs_140545 = ((double *) mem_152489)[i_151197];
                
                // futhark/microgpt.fut:222:37-64
                
                double zt_res_140546 = zs_res_140538 * zt_lhs_140545;
                
                ((double *) mem_152496)[i_151197] = zt_res_140546;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151201 = 0; i_151201 < (int64_t) 16; i_151201++) {
                // futhark/microgpt.fut:223:4-14
                
                double lifted_lambda_res_140554 = ((double *) mem_152496)[i_151201];
                
                ((double *) mem_152503)[i_151201] = lifted_lambda_res_140554;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152484, i_151205 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152503, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151213 = 0; i_151213 < (int64_t) 16; i_151213++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151209 = 0; i_151209 < (int64_t) 4; i_151209++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_140569;
                double r_140571 = 0.0;
                
                for (int64_t i_140570 = 0; i_140570 < (int64_t) 16; i_140570++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_140572 = ((double *) mem_152484)[i_151213 * (int64_t) 16 + i_140570];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_140573 = ((double *) mem_152365)[i_151225 * (int64_t) 64 + i_140570 * (int64_t) 4 + i_151209];
                    
                    // futhark/microgpt.fut:224:66-111
                    
                    double zt_res_140574 = zt_lhs_140572 * zt_rhs_140573;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_140575 = r_140571 + zt_res_140574;
                    double r_tmp_154775 = zp_res_140575;
                    
                    r_140571 = r_tmp_154775;
                }
                defunc_0_lifted_lambda_res_140569 = r_140571;
                ((double *) mem_152519)[i_151209] = defunc_0_lifted_lambda_res_140569;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152514, i_151213 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152519, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151221 = 0; i_151221 < (int64_t) 16; i_151221++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151217 = 0; i_151217 < (int64_t) 4; i_151217++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_140590 = ((double *) mem_152514)[i_151221 * (int64_t) 4 + i_151217];
                
                ((double *) mem_152535)[i_151217] = lifted_lambda_res_140590;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152530, i_151221 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152535, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_152446, i_151225 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_152530, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152551_cached_sizze_155200 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152551, &mem_152551_cached_sizze_155200, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152556_cached_sizze_155201 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152556, &mem_152556_cached_sizze_155201, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151233 = 0; i_151233 < (int64_t) 16; i_151233++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151229 = 0; i_151229 < (int64_t) 16; i_151229++) {
            // futhark/microgpt.fut:226:54-57
            
            int64_t tmp_140602 = sdiv64(i_151229, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-59
            
            bool x_140603 = sle64((int64_t) 0, tmp_140602);
            
            // futhark/microgpt.fut:226:44-59
            
            bool y_140604 = slt64(tmp_140602, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-59
            
            bool bounds_check_140605 = x_140603 && y_140604;
            
            // futhark/microgpt.fut:226:44-59
            
            bool index_certs_140606;
            
            if (!bounds_check_140605) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_140602, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:226:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:226:15-80\n   #6  futhark/microgpt.fut:467:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:226:74-77
            
            int64_t tmp_140607 = smod64(i_151229, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-79
            
            bool x_140608 = sle64((int64_t) 0, tmp_140607);
            
            // futhark/microgpt.fut:226:44-79
            
            bool y_140609 = slt64(tmp_140607, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-79
            
            bool bounds_check_140610 = x_140608 && y_140609;
            
            // futhark/microgpt.fut:226:44-79
            
            bool index_certs_140611;
            
            if (!bounds_check_140610) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_140607, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:226:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:226:15-80\n   #6  futhark/microgpt.fut:467:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_140612 = ((double *) mem_152446)[tmp_140602 * (int64_t) 64 + i_151233 * (int64_t) 4 + tmp_140607];
            
            ((double *) mem_152556)[i_151229] = lifted_lambda_res_140612;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152551, i_151233 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152556, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152567_cached_sizze_155202 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152567, &mem_152567_cached_sizze_155202, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152572_cached_sizze_155203 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152572, &mem_152572_cached_sizze_155203, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151241 = 0; i_151241 < (int64_t) 16; i_151241++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151237 = 0; i_151237 < (int64_t) 16; i_151237++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_140627;
            double r_140629 = 0.0;
            
            for (int64_t i_140628 = 0; i_140628 < (int64_t) 16; i_140628++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_140630 = ((double *) wout_mem_152245.mem)[i_151237 * (int64_t) 16 + i_140628];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_140631 = ((double *) mem_152551)[i_151241 * (int64_t) 16 + i_140628];
                
                // futhark/microgpt.fut:227:67-106
                
                double zt_res_140632 = zt_lhs_140630 * zt_rhs_140631;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_140633 = r_140629 + zt_res_140632;
                double r_tmp_154782 = zp_res_140633;
                
                r_140629 = r_tmp_154782;
            }
            defunc_0_lifted_lambda_res_140627 = r_140629;
            ((double *) mem_152572)[i_151237] = defunc_0_lifted_lambda_res_140627;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152567, i_151241 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152572, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152583_cached_sizze_155204 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152583, &mem_152583_cached_sizze_155204, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152588_cached_sizze_155205 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152588, &mem_152588_cached_sizze_155205, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151249 = 0; i_151249 < (int64_t) 16; i_151249++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151245 = 0; i_151245 < (int64_t) 16; i_151245++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_140648 = ((double *) mem_152567)[i_151249 * (int64_t) 16 + i_151245];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_140649 = ((double *) mem_152271)[i_151249 * (int64_t) 16 + i_151245];
            
            // futhark/microgpt.fut:228:46-84
            
            double zp_res_140650 = zp_lhs_140648 + zp_rhs_140649;
            
            ((double *) mem_152588)[i_151245] = zp_res_140650;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152583, i_151249 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152588, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152599_cached_sizze_155206 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152599, &mem_152599_cached_sizze_155206, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152604_cached_sizze_155207 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152604, &mem_152604_cached_sizze_155207, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152611_cached_sizze_155208 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152611, &mem_152611_cached_sizze_155208, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151261 = 0; i_151261 < (int64_t) 16; i_151261++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_140659;
        double r_140661 = 0.0;
        
        for (int64_t i_140660 = 0; i_140660 < (int64_t) 16; i_140660++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_140662 = ((double *) mem_152583)[i_151261 * (int64_t) 16 + i_140660];
            
            // futhark/microgpt.fut:229:79-118
            
            double zt_res_140663 = zt_lhs_140662 * zt_lhs_140662;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_140664 = r_140661 + zt_res_140663;
            double r_tmp_154786 = zp_res_140664;
            
            r_140661 = r_tmp_154786;
        }
        defunc_0_lifted_lambda_res_140659 = r_140661;
        // futhark/microgpt.fut:229:58-136
        
        double zs_res_140665 = defunc_0_lifted_lambda_res_140659 / 16.0;
        
        // futhark/microgpt.fut:230:24-55
        
        double zp_res_140666 = 1.0e-5 + zs_res_140665;
        
        // futhark/microgpt.fut:230:16-55
        
        double sqrt_res_140667 = futrts_sqrt64(zp_res_140666);
        
        // futhark/microgpt.fut:231:60-71
        
        double zs_res_140668 = 1.0 / sqrt_res_140667;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151253 = 0; i_151253 < (int64_t) 16; i_151253++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_140675 = ((double *) mem_152583)[i_151261 * (int64_t) 16 + i_151253];
            
            // futhark/microgpt.fut:231:37-71
            
            double zt_res_140676 = zs_res_140668 * zt_lhs_140675;
            
            ((double *) mem_152604)[i_151253] = zt_res_140676;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151257 = 0; i_151257 < (int64_t) 16; i_151257++) {
            // futhark/microgpt.fut:232:4-14
            
            double lifted_lambda_res_140684 = ((double *) mem_152604)[i_151257];
            
            ((double *) mem_152611)[i_151257] = lifted_lambda_res_140684;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152599, i_151261 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152611, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152622_cached_sizze_155209 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_152622, &mem_152622_cached_sizze_155209, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152627_cached_sizze_155210 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152627, &mem_152627_cached_sizze_155210, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151269 = 0; i_151269 < (int64_t) 16; i_151269++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151265 = 0; i_151265 < (int64_t) 64; i_151265++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_140700;
            double r_140702 = 0.0;
            
            for (int64_t i_140701 = 0; i_140701 < (int64_t) 16; i_140701++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_140703 = ((double *) wup_mem_152249.mem)[i_151265 * (int64_t) 16 + i_140701];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_140704 = ((double *) mem_152599)[i_151269 * (int64_t) 16 + i_140701];
                
                // futhark/microgpt.fut:233:67-106
                
                double zt_res_140705 = zt_lhs_140703 * zt_rhs_140704;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_140706 = r_140702 + zt_res_140705;
                double r_tmp_154791 = zp_res_140706;
                
                r_140702 = r_tmp_154791;
            }
            defunc_0_lifted_lambda_res_140700 = r_140702;
            ((double *) mem_152627)[i_151265] = defunc_0_lifted_lambda_res_140700;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152622, i_151269 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152627, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152638_cached_sizze_155211 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_152638, &mem_152638_cached_sizze_155211, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152643_cached_sizze_155212 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152643, &mem_152643_cached_sizze_155212, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151277 = 0; i_151277 < (int64_t) 16; i_151277++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151273 = 0; i_151273 < (int64_t) 64; i_151273++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_140721 = ((double *) mem_152622)[i_151277 * (int64_t) 64 + i_151273];
            
            // futhark/microgpt.fut:234:45-73
            
            double max_res_140722 = fmax64(0.0, max_arg0_140721);
            
            ((double *) mem_152643)[i_151273] = max_res_140722;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152638, i_151277 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152643, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152654_cached_sizze_155213 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152654, &mem_152654_cached_sizze_155213, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152659_cached_sizze_155214 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152659, &mem_152659_cached_sizze_155214, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151285 = 0; i_151285 < (int64_t) 16; i_151285++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151281 = 0; i_151281 < (int64_t) 16; i_151281++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_140737;
            double r_140739 = 0.0;
            
            for (int64_t i_140738 = 0; i_140738 < (int64_t) 64; i_140738++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_140740 = ((double *) wdown_mem_152243.mem)[i_151281 * (int64_t) 64 + i_140738];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_140741 = ((double *) mem_152638)[i_151285 * (int64_t) 64 + i_140738];
                
                // futhark/microgpt.fut:235:67-108
                
                double zt_res_140742 = zt_lhs_140740 * zt_rhs_140741;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_140743 = r_140739 + zt_res_140742;
                double r_tmp_154796 = zp_res_140743;
                
                r_140739 = r_tmp_154796;
            }
            defunc_0_lifted_lambda_res_140737 = r_140739;
            ((double *) mem_152659)[i_151281] = defunc_0_lifted_lambda_res_140737;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152654, i_151285 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152659, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152670_cached_sizze_155215 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152670, &mem_152670_cached_sizze_155215, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152675_cached_sizze_155216 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152675, &mem_152675_cached_sizze_155216, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151293 = 0; i_151293 < (int64_t) 16; i_151293++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151289 = 0; i_151289 < (int64_t) 16; i_151289++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_140758 = ((double *) mem_152654)[i_151293 * (int64_t) 16 + i_151289];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_140759 = ((double *) mem_152583)[i_151293 * (int64_t) 16 + i_151289];
            
            // futhark/microgpt.fut:236:46-85
            
            double zp_res_140760 = zp_lhs_140758 + zp_rhs_140759;
            
            ((double *) mem_152675)[i_151289] = zp_res_140760;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152670, i_151293 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152675, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152686_cached_sizze_155217 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_152686, &mem_152686_cached_sizze_155217, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152691_cached_sizze_155218 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_152691, &mem_152691_cached_sizze_155218, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151301 = 0; i_151301 < (int64_t) 16; i_151301++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151297 = 0; i_151297 < (int64_t) 27; i_151297++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_140776;
            double r_140778 = 0.0;
            
            for (int64_t i_140777 = 0; i_140777 < (int64_t) 16; i_140777++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_140779 = ((double *) wvoc_mem_152251.mem)[i_151297 * (int64_t) 16 + i_140777];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_140780 = ((double *) mem_152670)[i_151301 * (int64_t) 16 + i_140777];
                
                // futhark/microgpt.fut:237:67-107
                
                double zt_res_140781 = zt_lhs_140779 * zt_rhs_140780;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_140782 = r_140778 + zt_res_140781;
                double r_tmp_154801 = zp_res_140782;
                
                r_140778 = r_tmp_154801;
            }
            defunc_0_lifted_lambda_res_140776 = r_140778;
            ((double *) mem_152691)[i_151297] = defunc_0_lifted_lambda_res_140776;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152686, i_151301 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152691, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152702, (int64_t) 128, "mem_152702")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152706_cached_sizze_155219 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_152706, &mem_152706_cached_sizze_155219, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152713_cached_sizze_155220 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_152713, &mem_152713_cached_sizze_155220, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151315 = 0; i_151315 < (int64_t) 16; i_151315++) {
        double x_141511;
        double redout_151303 = -INFINITY;
        
        for (int64_t i_151304 = 0; i_151304 < (int64_t) 27; i_151304++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_141458 = ((double *) mem_152686)[i_151315 * (int64_t) 27 + i_151304];
            
            // futhark/microgpt.fut:115:13-33
            
            double max_res_140806 = fmax64(lifted_lambda_res_141458, redout_151303);
            double redout_tmp_154803 = max_res_140806;
            
            redout_151303 = redout_tmp_154803;
        }
        x_141511 = redout_151303;
        // futhark/microgpt.fut:239:67-76
        
        double neg_res_140807 = -x_141511;
        
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_140791;
        double r_140793 = 0.0;
        
        for (int64_t i_140792 = 0; i_140792 < (int64_t) 27; i_140792++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151307 = 0; i_151307 < (int64_t) 27; i_151307++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_140814 = ((double *) mem_152686)[i_151315 * (int64_t) 27 + i_151307];
                
                // futhark/microgpt.fut:239:44-76
                
                double zp_res_140815 = neg_res_140807 + zp_lhs_140814;
                
                // futhark/microgpt.fut:239:37-76
                
                double exp_res_140816 = futrts_exp64(zp_res_140815);
                
                ((double *) mem_152706)[i_151307] = exp_res_140816;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_140818;
            double r_140820 = 0.0;
            
            for (int64_t i_140819 = 0; i_140819 < (int64_t) 27; i_140819++) {
                // futhark/microgpt.fut:240:36-46
                
                double lifted_lambda_res_140821 = ((double *) mem_152706)[i_140819];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_140822 = r_140820 + lifted_lambda_res_140821;
                double r_tmp_154806 = zp_res_140822;
                
                r_140820 = r_tmp_154806;
            }
            defunc_0_lifted_lambda_res_140818 = r_140820;
            // futhark/microgpt.fut:241:53-64
            
            double zs_res_140823 = 1.0 / defunc_0_lifted_lambda_res_140818;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151311 = 0; i_151311 < (int64_t) 27; i_151311++) {
                // futhark/microgpt.fut:241:37-47
                
                double zt_lhs_140830 = ((double *) mem_152706)[i_151311];
                
                // futhark/microgpt.fut:241:37-64
                
                double zt_res_140831 = zs_res_140823 * zt_lhs_140830;
                
                ((double *) mem_152713)[i_151311] = zt_res_140831;
            }
            // futhark/microgpt.fut:242:12-22
            
            double log_arg0_140833 = ((double *) mem_152713)[i_140792];
            
            // futhark/microgpt.fut:242:6-22
            
            double log_res_140834 = futrts_log64(log_arg0_140833);
            
            // futhark/microgpt.fut:71:46-49
            
            double zt_rhs_140835 = ((double *) target_mem_152253.mem)[i_151315 * (int64_t) 27 + i_140792];
            
            // futhark/microgpt.fut:242:6-48
            
            double zt_res_140836 = log_res_140834 * zt_rhs_140835;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_140837 = r_140793 + zt_res_140836;
            double r_tmp_154804 = zp_res_140837;
            
            r_140793 = r_tmp_154804;
        }
        defunc_0_lifted_lambda_res_140791 = r_140793;
        // futhark/microgpt.fut:238:37-242:54
        
        double neg_res_140838 = -defunc_0_lifted_lambda_res_140791;
        
        ((double *) mem_152702.mem)[i_151315] = neg_res_140838;
    }
    // futhark/microgpt.fut:71:13-49
    
    double defunc_0_lifted_lambda_res_140840;
    double r_140842 = 0.0;
    
    for (int64_t i_140841 = 0; i_140841 < (int64_t) 16; i_140841++) {
        // futhark/microgpt.fut:243:37-47
        
        double lifted_lambda_res_140843 = ((double *) mem_152702.mem)[i_140841];
        
        // futhark/microgpt.fut:71:40-49
        
        double zp_res_140844 = r_140842 + lifted_lambda_res_140843;
        double r_tmp_154808 = zp_res_140844;
        
        r_140842 = r_tmp_154808;
    }
    defunc_0_lifted_lambda_res_140840 = r_140842;
    // futhark/microgpt.fut:243:17-64
    
    double zs_res_140845 = defunc_0_lifted_lambda_res_140840 / 16.0;
    
    if (memblock_set(ctx, &mem_out_154731, &mem_152702, "mem_152702") != 0)
        return 1;
    prim_out_154732 = zs_res_140845;
    if (memblock_set(ctx, &*mem_out_p_155162, &mem_out_154731, "mem_out_154731") != 0)
        return 1;
    *out_prim_out_155163 = prim_out_154732;
    
  cleanup:
    {
        free(mem_152255);
        free(mem_152260);
        free(mem_152271);
        free(mem_152276);
        free(mem_152283);
        free(mem_152294);
        free(mem_152299);
        free(mem_152306);
        free(mem_152317);
        free(mem_152318);
        free(mem_152319);
        free(mem_152332);
        free(mem_152333);
        free(mem_152334);
        free(mem_152365);
        free(mem_152366);
        free(mem_152367);
        free(mem_152383);
        free(mem_152384);
        free(mem_152385);
        free(mem_152398);
        free(mem_152399);
        free(mem_152400);
        free(mem_152446);
        free(mem_152452);
        free(mem_152457);
        free(mem_152468);
        free(mem_152473);
        free(mem_152484);
        free(mem_152489);
        free(mem_152496);
        free(mem_152503);
        free(mem_152514);
        free(mem_152519);
        free(mem_152530);
        free(mem_152535);
        free(mem_152551);
        free(mem_152556);
        free(mem_152567);
        free(mem_152572);
        free(mem_152583);
        free(mem_152588);
        free(mem_152599);
        free(mem_152604);
        free(mem_152611);
        free(mem_152622);
        free(mem_152627);
        free(mem_152638);
        free(mem_152643);
        free(mem_152654);
        free(mem_152659);
        free(mem_152670);
        free(mem_152675);
        free(mem_152686);
        free(mem_152691);
        free(mem_152706);
        free(mem_152713);
        if (memblock_unref(ctx, &mem_152702, "mem_152702") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154731, "mem_out_154731") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_155221, struct memblock wdown_mem_152243, struct memblock wkey_mem_152244, struct memblock wout_mem_152245, struct memblock wpe_mem_152246, struct memblock wqry_mem_152247, struct memblock wte_mem_152248, struct memblock wup_mem_152249, struct memblock wval_mem_152250, struct memblock wvoc_mem_152251, struct memblock tokens_mem_152252, struct memblock mask_mem_152253)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_152254_cached_sizze_155222 = 0;
    unsigned char *mem_152254 = NULL;
    int64_t mem_152259_cached_sizze_155223 = 0;
    unsigned char *mem_152259 = NULL;
    int64_t mem_152270_cached_sizze_155224 = 0;
    unsigned char *mem_152270 = NULL;
    int64_t mem_152275_cached_sizze_155225 = 0;
    unsigned char *mem_152275 = NULL;
    int64_t mem_152282_cached_sizze_155226 = 0;
    unsigned char *mem_152282 = NULL;
    int64_t mem_152293_cached_sizze_155227 = 0;
    unsigned char *mem_152293 = NULL;
    int64_t mem_152298_cached_sizze_155228 = 0;
    unsigned char *mem_152298 = NULL;
    int64_t mem_152305_cached_sizze_155229 = 0;
    unsigned char *mem_152305 = NULL;
    int64_t mem_152316_cached_sizze_155230 = 0;
    unsigned char *mem_152316 = NULL;
    int64_t mem_152317_cached_sizze_155231 = 0;
    unsigned char *mem_152317 = NULL;
    int64_t mem_152318_cached_sizze_155232 = 0;
    unsigned char *mem_152318 = NULL;
    int64_t mem_152331_cached_sizze_155233 = 0;
    unsigned char *mem_152331 = NULL;
    int64_t mem_152332_cached_sizze_155234 = 0;
    unsigned char *mem_152332 = NULL;
    int64_t mem_152333_cached_sizze_155235 = 0;
    unsigned char *mem_152333 = NULL;
    int64_t mem_152364_cached_sizze_155236 = 0;
    unsigned char *mem_152364 = NULL;
    int64_t mem_152365_cached_sizze_155237 = 0;
    unsigned char *mem_152365 = NULL;
    int64_t mem_152366_cached_sizze_155238 = 0;
    unsigned char *mem_152366 = NULL;
    int64_t mem_152382_cached_sizze_155239 = 0;
    unsigned char *mem_152382 = NULL;
    int64_t mem_152383_cached_sizze_155240 = 0;
    unsigned char *mem_152383 = NULL;
    int64_t mem_152384_cached_sizze_155241 = 0;
    unsigned char *mem_152384 = NULL;
    int64_t mem_152397_cached_sizze_155242 = 0;
    unsigned char *mem_152397 = NULL;
    int64_t mem_152398_cached_sizze_155243 = 0;
    unsigned char *mem_152398 = NULL;
    int64_t mem_152399_cached_sizze_155244 = 0;
    unsigned char *mem_152399 = NULL;
    int64_t mem_152445_cached_sizze_155245 = 0;
    unsigned char *mem_152445 = NULL;
    int64_t mem_152451_cached_sizze_155246 = 0;
    unsigned char *mem_152451 = NULL;
    int64_t mem_152456_cached_sizze_155247 = 0;
    unsigned char *mem_152456 = NULL;
    int64_t mem_152467_cached_sizze_155248 = 0;
    unsigned char *mem_152467 = NULL;
    int64_t mem_152472_cached_sizze_155249 = 0;
    unsigned char *mem_152472 = NULL;
    int64_t mem_152483_cached_sizze_155250 = 0;
    unsigned char *mem_152483 = NULL;
    int64_t mem_152488_cached_sizze_155251 = 0;
    unsigned char *mem_152488 = NULL;
    int64_t mem_152495_cached_sizze_155252 = 0;
    unsigned char *mem_152495 = NULL;
    int64_t mem_152502_cached_sizze_155253 = 0;
    unsigned char *mem_152502 = NULL;
    int64_t mem_152513_cached_sizze_155254 = 0;
    unsigned char *mem_152513 = NULL;
    int64_t mem_152518_cached_sizze_155255 = 0;
    unsigned char *mem_152518 = NULL;
    int64_t mem_152529_cached_sizze_155256 = 0;
    unsigned char *mem_152529 = NULL;
    int64_t mem_152534_cached_sizze_155257 = 0;
    unsigned char *mem_152534 = NULL;
    int64_t mem_152550_cached_sizze_155258 = 0;
    unsigned char *mem_152550 = NULL;
    int64_t mem_152555_cached_sizze_155259 = 0;
    unsigned char *mem_152555 = NULL;
    int64_t mem_152566_cached_sizze_155260 = 0;
    unsigned char *mem_152566 = NULL;
    int64_t mem_152571_cached_sizze_155261 = 0;
    unsigned char *mem_152571 = NULL;
    int64_t mem_152582_cached_sizze_155262 = 0;
    unsigned char *mem_152582 = NULL;
    int64_t mem_152587_cached_sizze_155263 = 0;
    unsigned char *mem_152587 = NULL;
    int64_t mem_152598_cached_sizze_155264 = 0;
    unsigned char *mem_152598 = NULL;
    int64_t mem_152603_cached_sizze_155265 = 0;
    unsigned char *mem_152603 = NULL;
    int64_t mem_152610_cached_sizze_155266 = 0;
    unsigned char *mem_152610 = NULL;
    int64_t mem_152621_cached_sizze_155267 = 0;
    unsigned char *mem_152621 = NULL;
    int64_t mem_152626_cached_sizze_155268 = 0;
    unsigned char *mem_152626 = NULL;
    int64_t mem_152637_cached_sizze_155269 = 0;
    unsigned char *mem_152637 = NULL;
    int64_t mem_152642_cached_sizze_155270 = 0;
    unsigned char *mem_152642 = NULL;
    int64_t mem_152653_cached_sizze_155271 = 0;
    unsigned char *mem_152653 = NULL;
    int64_t mem_152658_cached_sizze_155272 = 0;
    unsigned char *mem_152658 = NULL;
    int64_t mem_152669_cached_sizze_155273 = 0;
    unsigned char *mem_152669 = NULL;
    int64_t mem_152674_cached_sizze_155274 = 0;
    unsigned char *mem_152674 = NULL;
    int64_t mem_152685_cached_sizze_155275 = 0;
    unsigned char *mem_152685 = NULL;
    int64_t mem_152690_cached_sizze_155276 = 0;
    unsigned char *mem_152690 = NULL;
    int64_t mem_152706_cached_sizze_155277 = 0;
    unsigned char *mem_152706 = NULL;
    struct memblock mem_152701;
    
    mem_152701.references = NULL;
    
    struct memblock mem_out_154731;
    
    mem_out_154731.references = NULL;
    
    struct memblock mem_152234 = ctx->constants->mem_152234;
    struct memblock mem_152235 = ctx->constants->mem_152235;
    struct memblock mem_152236 = ctx->constants->mem_152236;
    struct memblock mem_152237 = ctx->constants->mem_152237;
    struct memblock mem_152238 = ctx->constants->mem_152238;
    struct memblock mem_152239 = ctx->constants->mem_152239;
    struct memblock mem_152240 = ctx->constants->mem_152240;
    struct memblock mem_152241 = ctx->constants->mem_152241;
    struct memblock mem_152242 = ctx->constants->mem_152242;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_152254_cached_sizze_155222 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152254, &mem_152254_cached_sizze_155222, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152259_cached_sizze_155223 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152259, &mem_152259_cached_sizze_155223, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151097 = 0; i_151097 < (int64_t) 16; i_151097++) {
        // futhark/microgpt.fut:461:41-50
        
        int64_t tmp_140230 = ((int64_t *) tokens_mem_152252.mem)[i_151097];
        
        // futhark/microgpt.fut:461:37-51
        
        bool x_140231 = sle64((int64_t) 0, tmp_140230);
        
        // futhark/microgpt.fut:461:37-51
        
        bool y_140232 = slt64(tmp_140230, (int64_t) 27);
        
        // futhark/microgpt.fut:461:37-51
        
        bool bounds_check_140233 = x_140231 && y_140232;
        
        // futhark/microgpt.fut:461:37-51
        
        bool index_certs_140234;
        
        if (!bounds_check_140233) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_140230, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:461:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:461:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151093 = 0; i_151093 < (int64_t) 16; i_151093++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_140241 = ((double *) wte_mem_152248.mem)[tmp_140230 * (int64_t) 16 + i_151093];
            
            ((double *) mem_152259)[i_151093] = lifted_lambda_res_140241;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152254, i_151097 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152259, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152270_cached_sizze_155224 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152270, &mem_152270_cached_sizze_155224, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152275_cached_sizze_155225 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152275, &mem_152275_cached_sizze_155225, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152282_cached_sizze_155226 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152282, &mem_152282_cached_sizze_155226, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151109 = 0; i_151109 < (int64_t) 16; i_151109++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_140267;
        double r_140269 = 0.0;
        
        for (int64_t i_140268 = 0; i_140268 < (int64_t) 16; i_140268++) {
            // futhark/microgpt.fut:71:46-49
            
            double zp_lhs_140270 = ((double *) wpe_mem_152246.mem)[i_151109 * (int64_t) 16 + i_140268];
            
            // futhark/microgpt.fut:71:46-49
            
            double zp_rhs_140271 = ((double *) mem_152254)[i_151109 * (int64_t) 16 + i_140268];
            
            // futhark/microgpt.fut:148:76-116
            
            double zp_res_140272 = zp_lhs_140270 + zp_rhs_140271;
            
            // futhark/microgpt.fut:148:94-163
            
            double zt_res_140273 = zp_res_140272 * zp_res_140272;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_140274 = r_140269 + zt_res_140273;
            double r_tmp_154735 = zp_res_140274;
            
            r_140269 = r_tmp_154735;
        }
        defunc_0_lifted_lambda_res_140267 = r_140269;
        // futhark/microgpt.fut:148:54-182
        
        double zs_res_140275 = defunc_0_lifted_lambda_res_140267 / 16.0;
        
        // futhark/microgpt.fut:149:24-55
        
        double zp_res_140276 = 1.0e-5 + zs_res_140275;
        
        // futhark/microgpt.fut:149:16-55
        
        double sqrt_res_140277 = futrts_sqrt64(zp_res_140276);
        
        // futhark/microgpt.fut:150:85-96
        
        double zs_res_140278 = 1.0 / sqrt_res_140277;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151101 = 0; i_151101 < (int64_t) 16; i_151101++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_140285 = ((double *) wpe_mem_152246.mem)[i_151109 * (int64_t) 16 + i_151101];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_140286 = ((double *) mem_152254)[i_151109 * (int64_t) 16 + i_151101];
            
            // futhark/microgpt.fut:150:38-78
            
            double zp_res_140287 = zp_lhs_140285 + zp_rhs_140286;
            
            // futhark/microgpt.fut:150:56-96
            
            double zt_res_140288 = zs_res_140278 * zp_res_140287;
            
            ((double *) mem_152275)[i_151101] = zt_res_140288;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151105 = 0; i_151105 < (int64_t) 16; i_151105++) {
            // futhark/microgpt.fut:151:4-14
            
            double lifted_lambda_res_140296 = ((double *) mem_152275)[i_151105];
            
            ((double *) mem_152282)[i_151105] = lifted_lambda_res_140296;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152270, i_151109 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152282, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152293_cached_sizze_155227 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152293, &mem_152293_cached_sizze_155227, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152298_cached_sizze_155228 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152298, &mem_152298_cached_sizze_155228, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152305_cached_sizze_155229 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152305, &mem_152305_cached_sizze_155229, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151121 = 0; i_151121 < (int64_t) 16; i_151121++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_140305;
        double r_140307 = 0.0;
        
        for (int64_t i_140306 = 0; i_140306 < (int64_t) 16; i_140306++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_140308 = ((double *) mem_152270)[i_151121 * (int64_t) 16 + i_140306];
            
            // futhark/microgpt.fut:152:78-115
            
            double zt_res_140309 = zt_lhs_140308 * zt_lhs_140308;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_140310 = r_140307 + zt_res_140309;
            double r_tmp_154739 = zp_res_140310;
            
            r_140307 = r_tmp_154739;
        }
        defunc_0_lifted_lambda_res_140305 = r_140307;
        // futhark/microgpt.fut:152:57-133
        
        double zs_res_140311 = defunc_0_lifted_lambda_res_140305 / 16.0;
        
        // futhark/microgpt.fut:153:24-55
        
        double zp_res_140312 = 1.0e-5 + zs_res_140311;
        
        // futhark/microgpt.fut:153:16-55
        
        double sqrt_res_140313 = futrts_sqrt64(zp_res_140312);
        
        // futhark/microgpt.fut:154:59-70
        
        double zs_res_140314 = 1.0 / sqrt_res_140313;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151113 = 0; i_151113 < (int64_t) 16; i_151113++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_140321 = ((double *) mem_152270)[i_151121 * (int64_t) 16 + i_151113];
            
            // futhark/microgpt.fut:154:37-70
            
            double zt_res_140322 = zs_res_140314 * zt_lhs_140321;
            
            ((double *) mem_152298)[i_151113] = zt_res_140322;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151117 = 0; i_151117 < (int64_t) 16; i_151117++) {
            // futhark/microgpt.fut:155:4-14
            
            double lifted_lambda_res_140330 = ((double *) mem_152298)[i_151117];
            
            ((double *) mem_152305)[i_151117] = lifted_lambda_res_140330;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152293, i_151121 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152305, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152316_cached_sizze_155230 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152316, &mem_152316_cached_sizze_155230, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152317_cached_sizze_155231 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152317, &mem_152317_cached_sizze_155231, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152318_cached_sizze_155232 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152318, &mem_152318_cached_sizze_155232, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152331_cached_sizze_155233 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152331, &mem_152331_cached_sizze_155233, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152332_cached_sizze_155234 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152332, &mem_152332_cached_sizze_155234, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152333_cached_sizze_155235 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152333, &mem_152333_cached_sizze_155235, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151139 = 0; i_151139 < (int64_t) 16; i_151139++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151129 = 0; i_151129 < (int64_t) 16; i_151129++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_141168;
            double r_141170 = 0.0;
            
            for (int64_t i_141169 = 0; i_141169 < (int64_t) 16; i_141169++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_141171 = ((double *) wqry_mem_152247.mem)[i_151129 * (int64_t) 16 + i_141169];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_141172 = ((double *) mem_152293)[i_151139 * (int64_t) 16 + i_141169];
                
                // futhark/microgpt.fut:156:66-105
                
                double zt_res_141173 = zt_lhs_141171 * zt_rhs_141172;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_141174 = r_141170 + zt_res_141173;
                double r_tmp_154748 = zp_res_141174;
                
                r_141170 = r_tmp_154748;
            }
            defunc_0_lifted_lambda_res_141168 = r_141170;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_141181;
            double r_141183 = 0.0;
            
            for (int64_t i_141182 = 0; i_141182 < (int64_t) 16; i_141182++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_141184 = ((double *) wkey_mem_152244.mem)[i_151129 * (int64_t) 16 + i_141182];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_141185 = ((double *) mem_152293)[i_151139 * (int64_t) 16 + i_141182];
                
                // futhark/microgpt.fut:157:66-105
                
                double zt_res_141186 = zt_lhs_141184 * zt_rhs_141185;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_141187 = r_141183 + zt_res_141186;
                double r_tmp_154749 = zp_res_141187;
                
                r_141183 = r_tmp_154749;
            }
            defunc_0_lifted_lambda_res_141181 = r_141183;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_141197;
            double r_141199 = 0.0;
            
            for (int64_t i_141198 = 0; i_141198 < (int64_t) 16; i_141198++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_141200 = ((double *) wval_mem_152250.mem)[i_151129 * (int64_t) 16 + i_141198];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_141201 = ((double *) mem_152293)[i_151139 * (int64_t) 16 + i_141198];
                
                // futhark/microgpt.fut:158:66-105
                
                double zt_res_141202 = zt_lhs_141200 * zt_rhs_141201;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_141203 = r_141199 + zt_res_141202;
                double r_tmp_154750 = zp_res_141203;
                
                r_141199 = r_tmp_154750;
            }
            defunc_0_lifted_lambda_res_141197 = r_141199;
            ((double *) mem_152331)[i_151129] = defunc_0_lifted_lambda_res_141197;
            ((double *) mem_152332)[i_151129] = defunc_0_lifted_lambda_res_141181;
            ((double *) mem_152333)[i_151129] = defunc_0_lifted_lambda_res_141168;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152316, i_151139 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152331, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152317, i_151139 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152332, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152318, i_151139 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152333, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152364_cached_sizze_155236 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152364, &mem_152364_cached_sizze_155236, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152365_cached_sizze_155237 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152365, &mem_152365_cached_sizze_155237, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152366_cached_sizze_155238 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152366, &mem_152366_cached_sizze_155238, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152382_cached_sizze_155239 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152382, &mem_152382_cached_sizze_155239, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152383_cached_sizze_155240 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152383, &mem_152383_cached_sizze_155240, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152384_cached_sizze_155241 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152384, &mem_152384_cached_sizze_155241, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152397_cached_sizze_155242 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152397, &mem_152397_cached_sizze_155242, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152398_cached_sizze_155243 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152398, &mem_152398_cached_sizze_155243, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152399_cached_sizze_155244 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152399, &mem_152399_cached_sizze_155244, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151169 = 0; i_151169 < (int64_t) 4; i_151169++) {
        // futhark/microgpt.fut:159:69-72
        
        int64_t zp_lhs_141044 = mul64((int64_t) 4, i_151169);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151159 = 0; i_151159 < (int64_t) 16; i_151159++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151149 = 0; i_151149 < (int64_t) 4; i_151149++) {
                // futhark/microgpt.fut:159:74-81
                
                int64_t tmp_141361 = add64(zp_lhs_141044, i_151149);
                
                // futhark/microgpt.fut:159:51-83
                
                bool x_141362 = sle64((int64_t) 0, tmp_141361);
                
                // futhark/microgpt.fut:159:51-83
                
                bool y_141363 = slt64(tmp_141361, (int64_t) 16);
                
                // futhark/microgpt.fut:159:51-83
                
                bool bounds_check_141364 = x_141362 && y_141363;
                
                // futhark/microgpt.fut:159:51-83
                
                bool index_certs_141365;
                
                if (!bounds_check_141364) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_141361, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:159:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:159:15-84\n   #9  futhark/microgpt.fut:462:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141366 = ((double *) mem_152318)[i_151159 * (int64_t) 16 + tmp_141361];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141374 = ((double *) mem_152317)[i_151159 * (int64_t) 16 + tmp_141361];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141385 = ((double *) mem_152316)[i_151159 * (int64_t) 16 + tmp_141361];
                
                ((double *) mem_152397)[i_151149] = lifted_lambda_res_141385;
                ((double *) mem_152398)[i_151149] = lifted_lambda_res_141374;
                ((double *) mem_152399)[i_151149] = lifted_lambda_res_141366;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152382, i_151159 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152397, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152383, i_151159 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152398, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152384, i_151159 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152399, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_152364, i_151169 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_152382, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_152365, i_151169 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_152383, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_152366, i_151169 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_152384, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152445_cached_sizze_155245 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152445, &mem_152445_cached_sizze_155245, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152451_cached_sizze_155246 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152451, &mem_152451_cached_sizze_155246, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152456_cached_sizze_155247 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152456, &mem_152456_cached_sizze_155247, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152467_cached_sizze_155248 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152467, &mem_152467_cached_sizze_155248, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152472_cached_sizze_155249 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152472, &mem_152472_cached_sizze_155249, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152483_cached_sizze_155250 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152483, &mem_152483_cached_sizze_155250, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152488_cached_sizze_155251 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152488, &mem_152488_cached_sizze_155251, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152495_cached_sizze_155252 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152495, &mem_152495_cached_sizze_155252, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152502_cached_sizze_155253 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152502, &mem_152502_cached_sizze_155253, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152513_cached_sizze_155254 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152513, &mem_152513_cached_sizze_155254, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152518_cached_sizze_155255 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152518, &mem_152518_cached_sizze_155255, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152529_cached_sizze_155256 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152529, &mem_152529_cached_sizze_155256, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152534_cached_sizze_155257 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152534, &mem_152534_cached_sizze_155257, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151225 = 0; i_151225 < (int64_t) 4; i_151225++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151179 = 0; i_151179 < (int64_t) 16; i_151179++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151175 = 0; i_151175 < (int64_t) 16; i_151175++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_140475;
                double r_140477 = 0.0;
                
                for (int64_t i_140476 = 0; i_140476 < (int64_t) 4; i_140476++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_140478 = ((double *) mem_152366)[i_151225 * (int64_t) 64 + i_151179 * (int64_t) 4 + i_140476];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_140479 = ((double *) mem_152365)[i_151225 * (int64_t) 64 + i_151175 * (int64_t) 4 + i_140476];
                    
                    // futhark/microgpt.fut:162:113-164
                    
                    double zt_res_140480 = zt_lhs_140478 * zt_rhs_140479;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_140481 = r_140477 + zt_res_140480;
                    double r_tmp_154763 = zp_res_140481;
                    
                    r_140477 = r_tmp_154763;
                }
                defunc_0_lifted_lambda_res_140475 = r_140477;
                ((double *) mem_152456)[i_151175] = defunc_0_lifted_lambda_res_140475;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152451, i_151179 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152456, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151187 = 0; i_151187 < (int64_t) 16; i_151187++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151183 = 0; i_151183 < (int64_t) 16; i_151183++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_140496 = ((double *) mem_152451)[i_151187 * (int64_t) 16 + i_151183];
                
                // futhark/microgpt.fut:163:47-78
                
                double zs_res_140497 = zs_lhs_140496 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_140498 = ((double *) mask_mem_152253.mem)[i_151187 * (int64_t) 16 + i_151183];
                
                // futhark/microgpt.fut:163:65-102
                
                double zp_res_140499 = zs_res_140497 + zp_rhs_140498;
                
                ((double *) mem_152472)[i_151183] = zp_res_140499;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152467, i_151187 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152472, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151205 = 0; i_151205 < (int64_t) 16; i_151205++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_141463;
            double redout_151189 = -INFINITY;
            
            for (int64_t i_151190 = 0; i_151190 < (int64_t) 16; i_151190++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141412 = ((double *) mem_152467)[i_151205 * (int64_t) 16 + i_151190];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_140520 = fmax64(lifted_lambda_res_141412, redout_151189);
                double redout_tmp_154767 = max_res_140520;
                
                redout_151189 = redout_tmp_154767;
            }
            defunc_0_reduce_res_141463 = redout_151189;
            // futhark/microgpt.fut:165:67-76
            
            double neg_res_140521 = -defunc_0_reduce_res_141463;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151193 = 0; i_151193 < (int64_t) 16; i_151193++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_140528 = ((double *) mem_152467)[i_151205 * (int64_t) 16 + i_151193];
                
                // futhark/microgpt.fut:165:44-76
                
                double zp_res_140529 = neg_res_140521 + zp_lhs_140528;
                
                // futhark/microgpt.fut:165:37-76
                
                double exp_res_140530 = futrts_exp64(zp_res_140529);
                
                ((double *) mem_152488)[i_151193] = exp_res_140530;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_140532;
            double r_140534 = 0.0;
            
            for (int64_t i_140533 = 0; i_140533 < (int64_t) 16; i_140533++) {
                // futhark/microgpt.fut:166:36-46
                
                double lifted_lambda_res_140535 = ((double *) mem_152488)[i_140533];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_140536 = r_140534 + lifted_lambda_res_140535;
                double r_tmp_154769 = zp_res_140536;
                
                r_140534 = r_tmp_154769;
            }
            defunc_0_lifted_lambda_res_140532 = r_140534;
            // futhark/microgpt.fut:167:53-64
            
            double zs_res_140537 = 1.0 / defunc_0_lifted_lambda_res_140532;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151197 = 0; i_151197 < (int64_t) 16; i_151197++) {
                // futhark/microgpt.fut:167:37-47
                
                double zt_lhs_140544 = ((double *) mem_152488)[i_151197];
                
                // futhark/microgpt.fut:167:37-64
                
                double zt_res_140545 = zs_res_140537 * zt_lhs_140544;
                
                ((double *) mem_152495)[i_151197] = zt_res_140545;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151201 = 0; i_151201 < (int64_t) 16; i_151201++) {
                // futhark/microgpt.fut:168:4-14
                
                double lifted_lambda_res_140553 = ((double *) mem_152495)[i_151201];
                
                ((double *) mem_152502)[i_151201] = lifted_lambda_res_140553;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152483, i_151205 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152502, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151213 = 0; i_151213 < (int64_t) 16; i_151213++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151209 = 0; i_151209 < (int64_t) 4; i_151209++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_140568;
                double r_140570 = 0.0;
                
                for (int64_t i_140569 = 0; i_140569 < (int64_t) 16; i_140569++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_140571 = ((double *) mem_152483)[i_151213 * (int64_t) 16 + i_140569];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_140572 = ((double *) mem_152364)[i_151225 * (int64_t) 64 + i_140569 * (int64_t) 4 + i_151209];
                    
                    // futhark/microgpt.fut:169:66-111
                    
                    double zt_res_140573 = zt_lhs_140571 * zt_rhs_140572;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_140574 = r_140570 + zt_res_140573;
                    double r_tmp_154774 = zp_res_140574;
                    
                    r_140570 = r_tmp_154774;
                }
                defunc_0_lifted_lambda_res_140568 = r_140570;
                ((double *) mem_152518)[i_151209] = defunc_0_lifted_lambda_res_140568;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152513, i_151213 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152518, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151221 = 0; i_151221 < (int64_t) 16; i_151221++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151217 = 0; i_151217 < (int64_t) 4; i_151217++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_140589 = ((double *) mem_152513)[i_151221 * (int64_t) 4 + i_151217];
                
                ((double *) mem_152534)[i_151217] = lifted_lambda_res_140589;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152529, i_151221 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152534, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_152445, i_151225 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_152529, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152550_cached_sizze_155258 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152550, &mem_152550_cached_sizze_155258, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152555_cached_sizze_155259 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152555, &mem_152555_cached_sizze_155259, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151233 = 0; i_151233 < (int64_t) 16; i_151233++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151229 = 0; i_151229 < (int64_t) 16; i_151229++) {
            // futhark/microgpt.fut:171:54-57
            
            int64_t tmp_140601 = sdiv64(i_151229, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-59
            
            bool x_140602 = sle64((int64_t) 0, tmp_140601);
            
            // futhark/microgpt.fut:171:44-59
            
            bool y_140603 = slt64(tmp_140601, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-59
            
            bool bounds_check_140604 = x_140602 && y_140603;
            
            // futhark/microgpt.fut:171:44-59
            
            bool index_certs_140605;
            
            if (!bounds_check_140604) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_140601, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:171:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:171:15-80\n   #6  futhark/microgpt.fut:462:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:171:74-77
            
            int64_t tmp_140606 = smod64(i_151229, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-79
            
            bool x_140607 = sle64((int64_t) 0, tmp_140606);
            
            // futhark/microgpt.fut:171:44-79
            
            bool y_140608 = slt64(tmp_140606, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-79
            
            bool bounds_check_140609 = x_140607 && y_140608;
            
            // futhark/microgpt.fut:171:44-79
            
            bool index_certs_140610;
            
            if (!bounds_check_140609) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_140606, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:171:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:171:15-80\n   #6  futhark/microgpt.fut:462:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_140611 = ((double *) mem_152445)[tmp_140601 * (int64_t) 64 + i_151233 * (int64_t) 4 + tmp_140606];
            
            ((double *) mem_152555)[i_151229] = lifted_lambda_res_140611;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152550, i_151233 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152555, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152566_cached_sizze_155260 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152566, &mem_152566_cached_sizze_155260, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152571_cached_sizze_155261 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152571, &mem_152571_cached_sizze_155261, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151241 = 0; i_151241 < (int64_t) 16; i_151241++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151237 = 0; i_151237 < (int64_t) 16; i_151237++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_140626;
            double r_140628 = 0.0;
            
            for (int64_t i_140627 = 0; i_140627 < (int64_t) 16; i_140627++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_140629 = ((double *) wout_mem_152245.mem)[i_151237 * (int64_t) 16 + i_140627];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_140630 = ((double *) mem_152550)[i_151241 * (int64_t) 16 + i_140627];
                
                // futhark/microgpt.fut:172:67-106
                
                double zt_res_140631 = zt_lhs_140629 * zt_rhs_140630;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_140632 = r_140628 + zt_res_140631;
                double r_tmp_154781 = zp_res_140632;
                
                r_140628 = r_tmp_154781;
            }
            defunc_0_lifted_lambda_res_140626 = r_140628;
            ((double *) mem_152571)[i_151237] = defunc_0_lifted_lambda_res_140626;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152566, i_151241 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152571, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152582_cached_sizze_155262 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152582, &mem_152582_cached_sizze_155262, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152587_cached_sizze_155263 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152587, &mem_152587_cached_sizze_155263, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151249 = 0; i_151249 < (int64_t) 16; i_151249++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151245 = 0; i_151245 < (int64_t) 16; i_151245++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_140647 = ((double *) mem_152566)[i_151249 * (int64_t) 16 + i_151245];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_140648 = ((double *) mem_152270)[i_151249 * (int64_t) 16 + i_151245];
            
            // futhark/microgpt.fut:173:46-84
            
            double zp_res_140649 = zp_lhs_140647 + zp_rhs_140648;
            
            ((double *) mem_152587)[i_151245] = zp_res_140649;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152582, i_151249 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152587, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152598_cached_sizze_155264 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152598, &mem_152598_cached_sizze_155264, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152603_cached_sizze_155265 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152603, &mem_152603_cached_sizze_155265, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152610_cached_sizze_155266 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152610, &mem_152610_cached_sizze_155266, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151261 = 0; i_151261 < (int64_t) 16; i_151261++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_140658;
        double r_140660 = 0.0;
        
        for (int64_t i_140659 = 0; i_140659 < (int64_t) 16; i_140659++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_140661 = ((double *) mem_152582)[i_151261 * (int64_t) 16 + i_140659];
            
            // futhark/microgpt.fut:174:79-118
            
            double zt_res_140662 = zt_lhs_140661 * zt_lhs_140661;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_140663 = r_140660 + zt_res_140662;
            double r_tmp_154785 = zp_res_140663;
            
            r_140660 = r_tmp_154785;
        }
        defunc_0_lifted_lambda_res_140658 = r_140660;
        // futhark/microgpt.fut:174:58-136
        
        double zs_res_140664 = defunc_0_lifted_lambda_res_140658 / 16.0;
        
        // futhark/microgpt.fut:175:24-55
        
        double zp_res_140665 = 1.0e-5 + zs_res_140664;
        
        // futhark/microgpt.fut:175:16-55
        
        double sqrt_res_140666 = futrts_sqrt64(zp_res_140665);
        
        // futhark/microgpt.fut:176:60-71
        
        double zs_res_140667 = 1.0 / sqrt_res_140666;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151253 = 0; i_151253 < (int64_t) 16; i_151253++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_140674 = ((double *) mem_152582)[i_151261 * (int64_t) 16 + i_151253];
            
            // futhark/microgpt.fut:176:37-71
            
            double zt_res_140675 = zs_res_140667 * zt_lhs_140674;
            
            ((double *) mem_152603)[i_151253] = zt_res_140675;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151257 = 0; i_151257 < (int64_t) 16; i_151257++) {
            // futhark/microgpt.fut:177:4-14
            
            double lifted_lambda_res_140683 = ((double *) mem_152603)[i_151257];
            
            ((double *) mem_152610)[i_151257] = lifted_lambda_res_140683;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152598, i_151261 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152610, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152621_cached_sizze_155267 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_152621, &mem_152621_cached_sizze_155267, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152626_cached_sizze_155268 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152626, &mem_152626_cached_sizze_155268, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151269 = 0; i_151269 < (int64_t) 16; i_151269++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151265 = 0; i_151265 < (int64_t) 64; i_151265++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_140699;
            double r_140701 = 0.0;
            
            for (int64_t i_140700 = 0; i_140700 < (int64_t) 16; i_140700++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_140702 = ((double *) wup_mem_152249.mem)[i_151265 * (int64_t) 16 + i_140700];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_140703 = ((double *) mem_152598)[i_151269 * (int64_t) 16 + i_140700];
                
                // futhark/microgpt.fut:178:67-106
                
                double zt_res_140704 = zt_lhs_140702 * zt_rhs_140703;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_140705 = r_140701 + zt_res_140704;
                double r_tmp_154790 = zp_res_140705;
                
                r_140701 = r_tmp_154790;
            }
            defunc_0_lifted_lambda_res_140699 = r_140701;
            ((double *) mem_152626)[i_151265] = defunc_0_lifted_lambda_res_140699;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152621, i_151269 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152626, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152637_cached_sizze_155269 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_152637, &mem_152637_cached_sizze_155269, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152642_cached_sizze_155270 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152642, &mem_152642_cached_sizze_155270, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151277 = 0; i_151277 < (int64_t) 16; i_151277++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151273 = 0; i_151273 < (int64_t) 64; i_151273++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_140720 = ((double *) mem_152621)[i_151277 * (int64_t) 64 + i_151273];
            
            // futhark/microgpt.fut:179:45-73
            
            double max_res_140721 = fmax64(0.0, max_arg0_140720);
            
            ((double *) mem_152642)[i_151273] = max_res_140721;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152637, i_151277 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152642, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152653_cached_sizze_155271 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152653, &mem_152653_cached_sizze_155271, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152658_cached_sizze_155272 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152658, &mem_152658_cached_sizze_155272, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151285 = 0; i_151285 < (int64_t) 16; i_151285++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151281 = 0; i_151281 < (int64_t) 16; i_151281++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_140736;
            double r_140738 = 0.0;
            
            for (int64_t i_140737 = 0; i_140737 < (int64_t) 64; i_140737++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_140739 = ((double *) wdown_mem_152243.mem)[i_151281 * (int64_t) 64 + i_140737];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_140740 = ((double *) mem_152637)[i_151285 * (int64_t) 64 + i_140737];
                
                // futhark/microgpt.fut:180:67-108
                
                double zt_res_140741 = zt_lhs_140739 * zt_rhs_140740;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_140742 = r_140738 + zt_res_140741;
                double r_tmp_154795 = zp_res_140742;
                
                r_140738 = r_tmp_154795;
            }
            defunc_0_lifted_lambda_res_140736 = r_140738;
            ((double *) mem_152658)[i_151281] = defunc_0_lifted_lambda_res_140736;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152653, i_151285 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152658, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152669_cached_sizze_155273 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152669, &mem_152669_cached_sizze_155273, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152674_cached_sizze_155274 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152674, &mem_152674_cached_sizze_155274, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151293 = 0; i_151293 < (int64_t) 16; i_151293++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151289 = 0; i_151289 < (int64_t) 16; i_151289++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_140757 = ((double *) mem_152653)[i_151293 * (int64_t) 16 + i_151289];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_140758 = ((double *) mem_152582)[i_151293 * (int64_t) 16 + i_151289];
            
            // futhark/microgpt.fut:181:46-85
            
            double zp_res_140759 = zp_lhs_140757 + zp_rhs_140758;
            
            ((double *) mem_152674)[i_151289] = zp_res_140759;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152669, i_151293 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152674, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152685_cached_sizze_155275 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_152685, &mem_152685_cached_sizze_155275, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152690_cached_sizze_155276 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_152690, &mem_152690_cached_sizze_155276, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151301 = 0; i_151301 < (int64_t) 16; i_151301++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151297 = 0; i_151297 < (int64_t) 27; i_151297++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_140775;
            double r_140777 = 0.0;
            
            for (int64_t i_140776 = 0; i_140776 < (int64_t) 16; i_140776++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_140778 = ((double *) wvoc_mem_152251.mem)[i_151297 * (int64_t) 16 + i_140776];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_140779 = ((double *) mem_152669)[i_151301 * (int64_t) 16 + i_140776];
                
                // futhark/microgpt.fut:182:67-107
                
                double zt_res_140780 = zt_lhs_140778 * zt_rhs_140779;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_140781 = r_140777 + zt_res_140780;
                double r_tmp_154800 = zp_res_140781;
                
                r_140777 = r_tmp_154800;
            }
            defunc_0_lifted_lambda_res_140775 = r_140777;
            ((double *) mem_152690)[i_151297] = defunc_0_lifted_lambda_res_140775;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152685, i_151301 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152690, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_152701, (int64_t) 3456, "mem_152701")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152706_cached_sizze_155277 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_152706, &mem_152706_cached_sizze_155277, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_151309 = 0; i_151309 < (int64_t) 16; i_151309++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151305 = 0; i_151305 < (int64_t) 27; i_151305++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_140796 = ((double *) mem_152685)[i_151309 * (int64_t) 27 + i_151305];
            
            ((double *) mem_152706)[i_151305] = lifted_lambda_res_140796;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_152701.mem, i_151309 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152706, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_154731, &mem_152701, "mem_152701") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155221, &mem_out_154731, "mem_out_154731") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_152254);
        free(mem_152259);
        free(mem_152270);
        free(mem_152275);
        free(mem_152282);
        free(mem_152293);
        free(mem_152298);
        free(mem_152305);
        free(mem_152316);
        free(mem_152317);
        free(mem_152318);
        free(mem_152331);
        free(mem_152332);
        free(mem_152333);
        free(mem_152364);
        free(mem_152365);
        free(mem_152366);
        free(mem_152382);
        free(mem_152383);
        free(mem_152384);
        free(mem_152397);
        free(mem_152398);
        free(mem_152399);
        free(mem_152445);
        free(mem_152451);
        free(mem_152456);
        free(mem_152467);
        free(mem_152472);
        free(mem_152483);
        free(mem_152488);
        free(mem_152495);
        free(mem_152502);
        free(mem_152513);
        free(mem_152518);
        free(mem_152529);
        free(mem_152534);
        free(mem_152550);
        free(mem_152555);
        free(mem_152566);
        free(mem_152571);
        free(mem_152582);
        free(mem_152587);
        free(mem_152598);
        free(mem_152603);
        free(mem_152610);
        free(mem_152621);
        free(mem_152626);
        free(mem_152637);
        free(mem_152642);
        free(mem_152653);
        free(mem_152658);
        free(mem_152669);
        free(mem_152674);
        free(mem_152685);
        free(mem_152690);
        free(mem_152706);
        if (memblock_unref(ctx, &mem_152701, "mem_152701") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154731, "mem_out_154731") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_155278, struct memblock *mem_out_p_155279, struct memblock *mem_out_p_155280, struct memblock *mem_out_p_155281, struct memblock *mem_out_p_155282, struct memblock *mem_out_p_155283, struct memblock *mem_out_p_155284, struct memblock *mem_out_p_155285, struct memblock *mem_out_p_155286, struct memblock wte_mem_152243, struct memblock wpe_mem_152244, struct memblock wqry_mem_152245, struct memblock wkey_mem_152246, struct memblock wval_mem_152247, struct memblock wout_mem_152248, struct memblock wup_mem_152249, struct memblock wdown_mem_152250, struct memblock wvoc_mem_152251)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_154739;
    
    mem_out_154739.references = NULL;
    
    struct memblock mem_out_154738;
    
    mem_out_154738.references = NULL;
    
    struct memblock mem_out_154737;
    
    mem_out_154737.references = NULL;
    
    struct memblock mem_out_154736;
    
    mem_out_154736.references = NULL;
    
    struct memblock mem_out_154735;
    
    mem_out_154735.references = NULL;
    
    struct memblock mem_out_154734;
    
    mem_out_154734.references = NULL;
    
    struct memblock mem_out_154733;
    
    mem_out_154733.references = NULL;
    
    struct memblock mem_out_154732;
    
    mem_out_154732.references = NULL;
    
    struct memblock mem_out_154731;
    
    mem_out_154731.references = NULL;
    
    struct memblock mem_152234 = ctx->constants->mem_152234;
    struct memblock mem_152235 = ctx->constants->mem_152235;
    struct memblock mem_152236 = ctx->constants->mem_152236;
    struct memblock mem_152237 = ctx->constants->mem_152237;
    struct memblock mem_152238 = ctx->constants->mem_152238;
    struct memblock mem_152239 = ctx->constants->mem_152239;
    struct memblock mem_152240 = ctx->constants->mem_152240;
    struct memblock mem_152241 = ctx->constants->mem_152241;
    struct memblock mem_152242 = ctx->constants->mem_152242;
    
    if (memblock_set(ctx, &mem_out_154731, &wdown_mem_152250, "wdown_mem_152250") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154732, &wkey_mem_152246, "wkey_mem_152246") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154733, &wout_mem_152248, "wout_mem_152248") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154734, &wpe_mem_152244, "wpe_mem_152244") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154735, &wqry_mem_152245, "wqry_mem_152245") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154736, &wte_mem_152243, "wte_mem_152243") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154737, &wup_mem_152249, "wup_mem_152249") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154738, &wval_mem_152247, "wval_mem_152247") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154739, &wvoc_mem_152251, "wvoc_mem_152251") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155278, &mem_out_154731, "mem_out_154731") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155279, &mem_out_154732, "mem_out_154732") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155280, &mem_out_154733, "mem_out_154733") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155281, &mem_out_154734, "mem_out_154734") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155282, &mem_out_154735, "mem_out_154735") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155283, &mem_out_154736, "mem_out_154736") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155284, &mem_out_154737, "mem_out_154737") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155285, &mem_out_154738, "mem_out_154738") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155286, &mem_out_154739, "mem_out_154739") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_154739, "mem_out_154739") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154738, "mem_out_154738") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154737, "mem_out_154737") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154736, "mem_out_154736") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154735, "mem_out_154735") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154734, "mem_out_154734") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154733, "mem_out_154733") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154732, "mem_out_154732") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154731, "mem_out_154731") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_155287, struct memblock *mem_out_p_155288, struct memblock *mem_out_p_155289, struct memblock *mem_out_p_155290, struct memblock *mem_out_p_155291, struct memblock *mem_out_p_155292, struct memblock *mem_out_p_155293, struct memblock *mem_out_p_155294, struct memblock *mem_out_p_155295, struct memblock *mem_out_p_155296, struct memblock *mem_out_p_155297, struct memblock *mem_out_p_155298, struct memblock *mem_out_p_155299, struct memblock *mem_out_p_155300, struct memblock *mem_out_p_155301, struct memblock *mem_out_p_155302, struct memblock *mem_out_p_155303, struct memblock *mem_out_p_155304, struct memblock *mem_out_p_155305, struct memblock *mem_out_p_155306, struct memblock *mem_out_p_155307, struct memblock *mem_out_p_155308, struct memblock *mem_out_p_155309, struct memblock *mem_out_p_155310, struct memblock *mem_out_p_155311, struct memblock *mem_out_p_155312, struct memblock *mem_out_p_155313, struct memblock wdown_mem_152243, struct memblock wkey_mem_152244, struct memblock wout_mem_152245, struct memblock wpe_mem_152246, struct memblock wqry_mem_152247, struct memblock wte_mem_152248, struct memblock wup_mem_152249, struct memblock wval_mem_152250, struct memblock wvoc_mem_152251, struct memblock wdown_mem_152252, struct memblock wkey_mem_152253, struct memblock wout_mem_152254, struct memblock wpe_mem_152255, struct memblock wqry_mem_152256, struct memblock wte_mem_152257, struct memblock wup_mem_152258, struct memblock wval_mem_152259, struct memblock wvoc_mem_152260, struct memblock wdown_mem_152261, struct memblock wkey_mem_152262, struct memblock wout_mem_152263, struct memblock wpe_mem_152264, struct memblock wqry_mem_152265, struct memblock wte_mem_152266, struct memblock wup_mem_152267, struct memblock wval_mem_152268, struct memblock wvoc_mem_152269, struct memblock masks_mem_152270, struct memblock dls_mem_152271, struct memblock seqs_mem_152272)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_152381_cached_sizze_155314 = 0;
    unsigned char *mem_152381 = NULL;
    int64_t mem_152382_cached_sizze_155315 = 0;
    unsigned char *mem_152382 = NULL;
    int64_t mem_152391_cached_sizze_155316 = 0;
    unsigned char *mem_152391 = NULL;
    int64_t mem_152398_cached_sizze_155317 = 0;
    unsigned char *mem_152398 = NULL;
    int64_t mem_152413_cached_sizze_155318 = 0;
    unsigned char *mem_152413 = NULL;
    int64_t mem_152414_cached_sizze_155319 = 0;
    unsigned char *mem_152414 = NULL;
    int64_t mem_152415_cached_sizze_155320 = 0;
    unsigned char *mem_152415 = NULL;
    int64_t mem_152434_cached_sizze_155321 = 0;
    unsigned char *mem_152434 = NULL;
    int64_t mem_152441_cached_sizze_155322 = 0;
    unsigned char *mem_152441 = NULL;
    int64_t mem_152446_cached_sizze_155323 = 0;
    unsigned char *mem_152446 = NULL;
    int64_t mem_152457_cached_sizze_155324 = 0;
    unsigned char *mem_152457 = NULL;
    int64_t mem_152462_cached_sizze_155325 = 0;
    unsigned char *mem_152462 = NULL;
    int64_t mem_152473_cached_sizze_155326 = 0;
    unsigned char *mem_152473 = NULL;
    int64_t mem_152474_cached_sizze_155327 = 0;
    unsigned char *mem_152474 = NULL;
    int64_t mem_152487_cached_sizze_155328 = 0;
    unsigned char *mem_152487 = NULL;
    int64_t mem_152494_cached_sizze_155329 = 0;
    unsigned char *mem_152494 = NULL;
    int64_t mem_152499_cached_sizze_155330 = 0;
    unsigned char *mem_152499 = NULL;
    int64_t mem_152510_cached_sizze_155331 = 0;
    unsigned char *mem_152510 = NULL;
    int64_t mem_152515_cached_sizze_155332 = 0;
    unsigned char *mem_152515 = NULL;
    int64_t mem_152526_cached_sizze_155333 = 0;
    unsigned char *mem_152526 = NULL;
    int64_t mem_152527_cached_sizze_155334 = 0;
    unsigned char *mem_152527 = NULL;
    int64_t mem_152528_cached_sizze_155335 = 0;
    unsigned char *mem_152528 = NULL;
    int64_t mem_152544_cached_sizze_155336 = 0;
    unsigned char *mem_152544 = NULL;
    int64_t mem_152545_cached_sizze_155337 = 0;
    unsigned char *mem_152545 = NULL;
    int64_t mem_152546_cached_sizze_155338 = 0;
    unsigned char *mem_152546 = NULL;
    int64_t mem_152559_cached_sizze_155339 = 0;
    unsigned char *mem_152559 = NULL;
    int64_t mem_152560_cached_sizze_155340 = 0;
    unsigned char *mem_152560 = NULL;
    int64_t mem_152561_cached_sizze_155341 = 0;
    unsigned char *mem_152561 = NULL;
    int64_t mem_152607_cached_sizze_155342 = 0;
    unsigned char *mem_152607 = NULL;
    int64_t mem_152608_cached_sizze_155343 = 0;
    unsigned char *mem_152608 = NULL;
    int64_t mem_152609_cached_sizze_155344 = 0;
    unsigned char *mem_152609 = NULL;
    int64_t mem_152610_cached_sizze_155345 = 0;
    unsigned char *mem_152610 = NULL;
    int64_t mem_152631_cached_sizze_155346 = 0;
    unsigned char *mem_152631 = NULL;
    int64_t mem_152632_cached_sizze_155347 = 0;
    unsigned char *mem_152632 = NULL;
    int64_t mem_152633_cached_sizze_155348 = 0;
    unsigned char *mem_152633 = NULL;
    int64_t mem_152634_cached_sizze_155349 = 0;
    unsigned char *mem_152634 = NULL;
    int64_t mem_152651_cached_sizze_155350 = 0;
    unsigned char *mem_152651 = NULL;
    int64_t mem_152652_cached_sizze_155351 = 0;
    unsigned char *mem_152652 = NULL;
    int64_t mem_152653_cached_sizze_155352 = 0;
    unsigned char *mem_152653 = NULL;
    int64_t mem_152654_cached_sizze_155353 = 0;
    unsigned char *mem_152654 = NULL;
    int64_t mem_152715_cached_sizze_155354 = 0;
    unsigned char *mem_152715 = NULL;
    int64_t mem_152716_cached_sizze_155355 = 0;
    unsigned char *mem_152716 = NULL;
    int64_t mem_152717_cached_sizze_155356 = 0;
    unsigned char *mem_152717 = NULL;
    int64_t mem_152718_cached_sizze_155357 = 0;
    unsigned char *mem_152718 = NULL;
    int64_t mem_152739_cached_sizze_155358 = 0;
    unsigned char *mem_152739 = NULL;
    int64_t mem_152740_cached_sizze_155359 = 0;
    unsigned char *mem_152740 = NULL;
    int64_t mem_152741_cached_sizze_155360 = 0;
    unsigned char *mem_152741 = NULL;
    int64_t mem_152742_cached_sizze_155361 = 0;
    unsigned char *mem_152742 = NULL;
    int64_t mem_152759_cached_sizze_155362 = 0;
    unsigned char *mem_152759 = NULL;
    int64_t mem_152760_cached_sizze_155363 = 0;
    unsigned char *mem_152760 = NULL;
    int64_t mem_152761_cached_sizze_155364 = 0;
    unsigned char *mem_152761 = NULL;
    int64_t mem_152762_cached_sizze_155365 = 0;
    unsigned char *mem_152762 = NULL;
    int64_t mem_152823_cached_sizze_155366 = 0;
    unsigned char *mem_152823 = NULL;
    int64_t mem_152824_cached_sizze_155367 = 0;
    unsigned char *mem_152824 = NULL;
    int64_t mem_152825_cached_sizze_155368 = 0;
    unsigned char *mem_152825 = NULL;
    int64_t mem_152826_cached_sizze_155369 = 0;
    unsigned char *mem_152826 = NULL;
    int64_t mem_152827_cached_sizze_155370 = 0;
    unsigned char *mem_152827 = NULL;
    int64_t mem_152828_cached_sizze_155371 = 0;
    unsigned char *mem_152828 = NULL;
    int64_t mem_152829_cached_sizze_155372 = 0;
    unsigned char *mem_152829 = NULL;
    int64_t mem_152830_cached_sizze_155373 = 0;
    unsigned char *mem_152830 = NULL;
    int64_t mem_152863_cached_sizze_155374 = 0;
    unsigned char *mem_152863 = NULL;
    int64_t mem_152864_cached_sizze_155375 = 0;
    unsigned char *mem_152864 = NULL;
    int64_t mem_152865_cached_sizze_155376 = 0;
    unsigned char *mem_152865 = NULL;
    int64_t mem_152866_cached_sizze_155377 = 0;
    unsigned char *mem_152866 = NULL;
    int64_t mem_152867_cached_sizze_155378 = 0;
    unsigned char *mem_152867 = NULL;
    int64_t mem_152868_cached_sizze_155379 = 0;
    unsigned char *mem_152868 = NULL;
    int64_t mem_152869_cached_sizze_155380 = 0;
    unsigned char *mem_152869 = NULL;
    int64_t mem_152870_cached_sizze_155381 = 0;
    unsigned char *mem_152870 = NULL;
    int64_t mem_152951_cached_sizze_155382 = 0;
    unsigned char *mem_152951 = NULL;
    int64_t mem_152952_cached_sizze_155383 = 0;
    unsigned char *mem_152952 = NULL;
    int64_t mem_152953_cached_sizze_155384 = 0;
    unsigned char *mem_152953 = NULL;
    int64_t mem_152954_cached_sizze_155385 = 0;
    unsigned char *mem_152954 = NULL;
    int64_t mem_152975_cached_sizze_155386 = 0;
    unsigned char *mem_152975 = NULL;
    int64_t mem_152976_cached_sizze_155387 = 0;
    unsigned char *mem_152976 = NULL;
    int64_t mem_152977_cached_sizze_155388 = 0;
    unsigned char *mem_152977 = NULL;
    int64_t mem_152978_cached_sizze_155389 = 0;
    unsigned char *mem_152978 = NULL;
    int64_t mem_152995_cached_sizze_155390 = 0;
    unsigned char *mem_152995 = NULL;
    int64_t mem_152996_cached_sizze_155391 = 0;
    unsigned char *mem_152996 = NULL;
    int64_t mem_152997_cached_sizze_155392 = 0;
    unsigned char *mem_152997 = NULL;
    int64_t mem_152998_cached_sizze_155393 = 0;
    unsigned char *mem_152998 = NULL;
    int64_t mem_153059_cached_sizze_155394 = 0;
    unsigned char *mem_153059 = NULL;
    int64_t mem_153060_cached_sizze_155395 = 0;
    unsigned char *mem_153060 = NULL;
    int64_t mem_153069_cached_sizze_155396 = 0;
    unsigned char *mem_153069 = NULL;
    int64_t mem_153070_cached_sizze_155397 = 0;
    unsigned char *mem_153070 = NULL;
    int64_t mem_153091_cached_sizze_155398 = 0;
    unsigned char *mem_153091 = NULL;
    int64_t mem_153092_cached_sizze_155399 = 0;
    unsigned char *mem_153092 = NULL;
    int64_t mem_153103_cached_sizze_155400 = 0;
    unsigned char *mem_153103 = NULL;
    int64_t mem_153104_cached_sizze_155401 = 0;
    unsigned char *mem_153104 = NULL;
    int64_t mem_153113_cached_sizze_155402 = 0;
    unsigned char *mem_153113 = NULL;
    int64_t mem_153114_cached_sizze_155403 = 0;
    unsigned char *mem_153114 = NULL;
    int64_t mem_153145_cached_sizze_155404 = 0;
    unsigned char *mem_153145 = NULL;
    int64_t mem_153146_cached_sizze_155405 = 0;
    unsigned char *mem_153146 = NULL;
    int64_t mem_153157_cached_sizze_155406 = 0;
    unsigned char *mem_153157 = NULL;
    int64_t mem_153158_cached_sizze_155407 = 0;
    unsigned char *mem_153158 = NULL;
    int64_t mem_153167_cached_sizze_155408 = 0;
    unsigned char *mem_153167 = NULL;
    int64_t mem_153168_cached_sizze_155409 = 0;
    unsigned char *mem_153168 = NULL;
    int64_t mem_153199_cached_sizze_155410 = 0;
    unsigned char *mem_153199 = NULL;
    int64_t mem_153205_cached_sizze_155411 = 0;
    unsigned char *mem_153205 = NULL;
    int64_t mem_153210_cached_sizze_155412 = 0;
    unsigned char *mem_153210 = NULL;
    int64_t mem_153226_cached_sizze_155413 = 0;
    unsigned char *mem_153226 = NULL;
    int64_t mem_153231_cached_sizze_155414 = 0;
    unsigned char *mem_153231 = NULL;
    int64_t mem_153242_cached_sizze_155415 = 0;
    unsigned char *mem_153242 = NULL;
    int64_t mem_153247_cached_sizze_155416 = 0;
    unsigned char *mem_153247 = NULL;
    int64_t mem_153258_cached_sizze_155417 = 0;
    unsigned char *mem_153258 = NULL;
    int64_t mem_153259_cached_sizze_155418 = 0;
    unsigned char *mem_153259 = NULL;
    int64_t mem_153272_cached_sizze_155419 = 0;
    unsigned char *mem_153272 = NULL;
    int64_t mem_153279_cached_sizze_155420 = 0;
    unsigned char *mem_153279 = NULL;
    int64_t mem_153284_cached_sizze_155421 = 0;
    unsigned char *mem_153284 = NULL;
    int64_t mem_153295_cached_sizze_155422 = 0;
    unsigned char *mem_153295 = NULL;
    int64_t mem_153300_cached_sizze_155423 = 0;
    unsigned char *mem_153300 = NULL;
    int64_t mem_153311_cached_sizze_155424 = 0;
    unsigned char *mem_153311 = NULL;
    int64_t mem_153316_cached_sizze_155425 = 0;
    unsigned char *mem_153316 = NULL;
    int64_t mem_153327_cached_sizze_155426 = 0;
    unsigned char *mem_153327 = NULL;
    int64_t mem_153332_cached_sizze_155427 = 0;
    unsigned char *mem_153332 = NULL;
    int64_t mem_153343_cached_sizze_155428 = 0;
    unsigned char *mem_153343 = NULL;
    int64_t mem_153348_cached_sizze_155429 = 0;
    unsigned char *mem_153348 = NULL;
    int64_t mem_153359_cached_sizze_155430 = 0;
    unsigned char *mem_153359 = NULL;
    int64_t mem_153364_cached_sizze_155431 = 0;
    unsigned char *mem_153364 = NULL;
    int64_t mem_153375_cached_sizze_155432 = 0;
    unsigned char *mem_153375 = NULL;
    int64_t mem_153376_cached_sizze_155433 = 0;
    unsigned char *mem_153376 = NULL;
    int64_t mem_153377_cached_sizze_155434 = 0;
    unsigned char *mem_153377 = NULL;
    int64_t mem_153378_cached_sizze_155435 = 0;
    unsigned char *mem_153378 = NULL;
    int64_t mem_153396_cached_sizze_155436 = 0;
    unsigned char *mem_153396 = NULL;
    int64_t mem_153401_cached_sizze_155437 = 0;
    unsigned char *mem_153401 = NULL;
    int64_t mem_153405_cached_sizze_155438 = 0;
    unsigned char *mem_153405 = NULL;
    int64_t mem_153412_cached_sizze_155439 = 0;
    unsigned char *mem_153412 = NULL;
    int64_t mem_153446_cached_sizze_155440 = 0;
    unsigned char *mem_153446 = NULL;
    int64_t mem_153452_cached_sizze_155441 = 0;
    unsigned char *mem_153452 = NULL;
    int64_t mem_153457_cached_sizze_155442 = 0;
    unsigned char *mem_153457 = NULL;
    int64_t mem_153473_cached_sizze_155443 = 0;
    unsigned char *mem_153473 = NULL;
    int64_t mem_153474_cached_sizze_155444 = 0;
    unsigned char *mem_153474 = NULL;
    int64_t mem_153483_cached_sizze_155445 = 0;
    unsigned char *mem_153483 = NULL;
    int64_t mem_153484_cached_sizze_155446 = 0;
    unsigned char *mem_153484 = NULL;
    int64_t mem_153505_cached_sizze_155447 = 0;
    unsigned char *mem_153505 = NULL;
    int64_t mem_153511_cached_sizze_155448 = 0;
    unsigned char *mem_153511 = NULL;
    int64_t mem_153516_cached_sizze_155449 = 0;
    unsigned char *mem_153516 = NULL;
    int64_t mem_153532_cached_sizze_155450 = 0;
    unsigned char *mem_153532 = NULL;
    int64_t mem_153537_cached_sizze_155451 = 0;
    unsigned char *mem_153537 = NULL;
    int64_t mem_153548_cached_sizze_155452 = 0;
    unsigned char *mem_153548 = NULL;
    int64_t mem_153553_cached_sizze_155453 = 0;
    unsigned char *mem_153553 = NULL;
    int64_t mem_153564_cached_sizze_155454 = 0;
    unsigned char *mem_153564 = NULL;
    int64_t mem_153569_cached_sizze_155455 = 0;
    unsigned char *mem_153569 = NULL;
    int64_t mem_153580_cached_sizze_155456 = 0;
    unsigned char *mem_153580 = NULL;
    int64_t mem_153581_cached_sizze_155457 = 0;
    unsigned char *mem_153581 = NULL;
    int64_t mem_153590_cached_sizze_155458 = 0;
    unsigned char *mem_153590 = NULL;
    int64_t mem_153591_cached_sizze_155459 = 0;
    unsigned char *mem_153591 = NULL;
    int64_t mem_153612_cached_sizze_155460 = 0;
    unsigned char *mem_153612 = NULL;
    int64_t mem_153617_cached_sizze_155461 = 0;
    unsigned char *mem_153617 = NULL;
    int64_t mem_153628_cached_sizze_155462 = 0;
    unsigned char *mem_153628 = NULL;
    int64_t mem_153629_cached_sizze_155463 = 0;
    unsigned char *mem_153629 = NULL;
    int64_t mem_153642_cached_sizze_155464 = 0;
    unsigned char *mem_153642 = NULL;
    int64_t mem_153649_cached_sizze_155465 = 0;
    unsigned char *mem_153649 = NULL;
    int64_t mem_153654_cached_sizze_155466 = 0;
    unsigned char *mem_153654 = NULL;
    int64_t mem_153665_cached_sizze_155467 = 0;
    unsigned char *mem_153665 = NULL;
    int64_t mem_153671_cached_sizze_155468 = 0;
    unsigned char *mem_153671 = NULL;
    int64_t mem_153676_cached_sizze_155469 = 0;
    unsigned char *mem_153676 = NULL;
    int64_t mem_153692_cached_sizze_155470 = 0;
    unsigned char *mem_153692 = NULL;
    int64_t mem_153693_cached_sizze_155471 = 0;
    unsigned char *mem_153693 = NULL;
    int64_t mem_153694_cached_sizze_155472 = 0;
    unsigned char *mem_153694 = NULL;
    int64_t mem_153710_cached_sizze_155473 = 0;
    unsigned char *mem_153710 = NULL;
    int64_t mem_153711_cached_sizze_155474 = 0;
    unsigned char *mem_153711 = NULL;
    int64_t mem_153712_cached_sizze_155475 = 0;
    unsigned char *mem_153712 = NULL;
    int64_t mem_153725_cached_sizze_155476 = 0;
    unsigned char *mem_153725 = NULL;
    int64_t mem_153726_cached_sizze_155477 = 0;
    unsigned char *mem_153726 = NULL;
    int64_t mem_153767_cached_sizze_155478 = 0;
    unsigned char *mem_153767 = NULL;
    int64_t mem_153768_cached_sizze_155479 = 0;
    unsigned char *mem_153768 = NULL;
    int64_t mem_153779_cached_sizze_155480 = 0;
    unsigned char *mem_153779 = NULL;
    int64_t mem_153780_cached_sizze_155481 = 0;
    unsigned char *mem_153780 = NULL;
    int64_t mem_153789_cached_sizze_155482 = 0;
    unsigned char *mem_153789 = NULL;
    int64_t mem_153790_cached_sizze_155483 = 0;
    unsigned char *mem_153790 = NULL;
    int64_t mem_153821_cached_sizze_155484 = 0;
    unsigned char *mem_153821 = NULL;
    int64_t mem_153822_cached_sizze_155485 = 0;
    unsigned char *mem_153822 = NULL;
    int64_t mem_153833_cached_sizze_155486 = 0;
    unsigned char *mem_153833 = NULL;
    int64_t mem_153834_cached_sizze_155487 = 0;
    unsigned char *mem_153834 = NULL;
    int64_t mem_153843_cached_sizze_155488 = 0;
    unsigned char *mem_153843 = NULL;
    int64_t mem_153844_cached_sizze_155489 = 0;
    unsigned char *mem_153844 = NULL;
    int64_t mem_153875_cached_sizze_155490 = 0;
    unsigned char *mem_153875 = NULL;
    int64_t mem_153876_cached_sizze_155491 = 0;
    unsigned char *mem_153876 = NULL;
    int64_t mem_153877_cached_sizze_155492 = 0;
    unsigned char *mem_153877 = NULL;
    int64_t mem_153878_cached_sizze_155493 = 0;
    unsigned char *mem_153878 = NULL;
    int64_t mem_153895_cached_sizze_155494 = 0;
    unsigned char *mem_153895 = NULL;
    int64_t mem_153896_cached_sizze_155495 = 0;
    unsigned char *mem_153896 = NULL;
    int64_t mem_153897_cached_sizze_155496 = 0;
    unsigned char *mem_153897 = NULL;
    int64_t mem_153898_cached_sizze_155497 = 0;
    unsigned char *mem_153898 = NULL;
    int64_t mem_153939_cached_sizze_155498 = 0;
    unsigned char *mem_153939 = NULL;
    int64_t mem_153940_cached_sizze_155499 = 0;
    unsigned char *mem_153940 = NULL;
    int64_t mem_153951_cached_sizze_155500 = 0;
    unsigned char *mem_153951 = NULL;
    int64_t mem_153952_cached_sizze_155501 = 0;
    unsigned char *mem_153952 = NULL;
    int64_t mem_153961_cached_sizze_155502 = 0;
    unsigned char *mem_153961 = NULL;
    int64_t mem_153962_cached_sizze_155503 = 0;
    unsigned char *mem_153962 = NULL;
    int64_t mem_153993_cached_sizze_155504 = 0;
    unsigned char *mem_153993 = NULL;
    int64_t mem_153994_cached_sizze_155505 = 0;
    unsigned char *mem_153994 = NULL;
    int64_t mem_154003_cached_sizze_155506 = 0;
    unsigned char *mem_154003 = NULL;
    int64_t mem_154004_cached_sizze_155507 = 0;
    unsigned char *mem_154004 = NULL;
    int64_t mem_154025_cached_sizze_155508 = 0;
    unsigned char *mem_154025 = NULL;
    int64_t mem_154026_cached_sizze_155509 = 0;
    unsigned char *mem_154026 = NULL;
    int64_t mem_154037_cached_sizze_155510 = 0;
    unsigned char *mem_154037 = NULL;
    int64_t mem_154038_cached_sizze_155511 = 0;
    unsigned char *mem_154038 = NULL;
    int64_t mem_154047_cached_sizze_155512 = 0;
    unsigned char *mem_154047 = NULL;
    int64_t mem_154048_cached_sizze_155513 = 0;
    unsigned char *mem_154048 = NULL;
    int64_t mem_154079_cached_sizze_155514 = 0;
    unsigned char *mem_154079 = NULL;
    int64_t mem_154080_cached_sizze_155515 = 0;
    unsigned char *mem_154080 = NULL;
    int64_t mem_154091_cached_sizze_155516 = 0;
    unsigned char *mem_154091 = NULL;
    int64_t mem_154092_cached_sizze_155517 = 0;
    unsigned char *mem_154092 = NULL;
    int64_t mem_154101_cached_sizze_155518 = 0;
    unsigned char *mem_154101 = NULL;
    int64_t mem_154102_cached_sizze_155519 = 0;
    unsigned char *mem_154102 = NULL;
    int64_t mem_154133_cached_sizze_155520 = 0;
    unsigned char *mem_154133 = NULL;
    int64_t mem_154134_cached_sizze_155521 = 0;
    unsigned char *mem_154134 = NULL;
    int64_t mem_154135_cached_sizze_155522 = 0;
    unsigned char *mem_154135 = NULL;
    int64_t mem_154136_cached_sizze_155523 = 0;
    unsigned char *mem_154136 = NULL;
    int64_t mem_154153_cached_sizze_155524 = 0;
    unsigned char *mem_154153 = NULL;
    int64_t mem_154154_cached_sizze_155525 = 0;
    unsigned char *mem_154154 = NULL;
    int64_t mem_154155_cached_sizze_155526 = 0;
    unsigned char *mem_154155 = NULL;
    int64_t mem_154156_cached_sizze_155527 = 0;
    unsigned char *mem_154156 = NULL;
    int64_t mem_154197_cached_sizze_155528 = 0;
    unsigned char *mem_154197 = NULL;
    int64_t mem_154202_cached_sizze_155529 = 0;
    unsigned char *mem_154202 = NULL;
    int64_t mem_154213_cached_sizze_155530 = 0;
    unsigned char *mem_154213 = NULL;
    int64_t mem_154214_cached_sizze_155531 = 0;
    unsigned char *mem_154214 = NULL;
    int64_t mem_154215_cached_sizze_155532 = 0;
    unsigned char *mem_154215 = NULL;
    int64_t mem_154216_cached_sizze_155533 = 0;
    unsigned char *mem_154216 = NULL;
    int64_t mem_154217_cached_sizze_155534 = 0;
    unsigned char *mem_154217 = NULL;
    int64_t mem_154236_cached_sizze_155535 = 0;
    unsigned char *mem_154236 = NULL;
    int64_t mem_154237_cached_sizze_155536 = 0;
    unsigned char *mem_154237 = NULL;
    int64_t mem_154238_cached_sizze_155537 = 0;
    unsigned char *mem_154238 = NULL;
    int64_t mem_154275_cached_sizze_155538 = 0;
    unsigned char *mem_154275 = NULL;
    int64_t mem_154282_cached_sizze_155539 = 0;
    unsigned char *mem_154282 = NULL;
    int64_t mem_154287_cached_sizze_155540 = 0;
    unsigned char *mem_154287 = NULL;
    int64_t mem_154298_cached_sizze_155541 = 0;
    unsigned char *mem_154298 = NULL;
    int64_t mem_154299_cached_sizze_155542 = 0;
    unsigned char *mem_154299 = NULL;
    int64_t mem_154308_cached_sizze_155543 = 0;
    unsigned char *mem_154308 = NULL;
    int64_t mem_154309_cached_sizze_155544 = 0;
    unsigned char *mem_154309 = NULL;
    int64_t mem_154330_cached_sizze_155545 = 0;
    unsigned char *mem_154330 = NULL;
    int64_t mem_154331_cached_sizze_155546 = 0;
    unsigned char *mem_154331 = NULL;
    int64_t mem_154332_cached_sizze_155547 = 0;
    unsigned char *mem_154332 = NULL;
    int64_t mem_154333_cached_sizze_155548 = 0;
    unsigned char *mem_154333 = NULL;
    int64_t mem_154358_cached_sizze_155549 = 0;
    unsigned char *mem_154358 = NULL;
    int64_t mem_154359_cached_sizze_155550 = 0;
    unsigned char *mem_154359 = NULL;
    int64_t mem_154372_cached_sizze_155551 = 0;
    unsigned char *mem_154372 = NULL;
    int64_t mem_154373_cached_sizze_155552 = 0;
    unsigned char *mem_154373 = NULL;
    int64_t mem_154382_cached_sizze_155553 = 0;
    unsigned char *mem_154382 = NULL;
    int64_t mem_154383_cached_sizze_155554 = 0;
    unsigned char *mem_154383 = NULL;
    int64_t mem_154404_cached_sizze_155555 = 0;
    unsigned char *mem_154404 = NULL;
    int64_t mem_154409_cached_sizze_155556 = 0;
    unsigned char *mem_154409 = NULL;
    int64_t mem_154420_cached_sizze_155557 = 0;
    unsigned char *mem_154420 = NULL;
    int64_t mem_154421_cached_sizze_155558 = 0;
    unsigned char *mem_154421 = NULL;
    int64_t mem_154430_cached_sizze_155559 = 0;
    unsigned char *mem_154430 = NULL;
    int64_t mem_154431_cached_sizze_155560 = 0;
    unsigned char *mem_154431 = NULL;
    struct memblock mem_param_tmp_154787;
    
    mem_param_tmp_154787.references = NULL;
    
    struct memblock mem_param_tmp_154786;
    
    mem_param_tmp_154786.references = NULL;
    
    struct memblock mem_param_tmp_154785;
    
    mem_param_tmp_154785.references = NULL;
    
    struct memblock mem_param_tmp_154784;
    
    mem_param_tmp_154784.references = NULL;
    
    struct memblock mem_param_tmp_154783;
    
    mem_param_tmp_154783.references = NULL;
    
    struct memblock mem_param_tmp_154782;
    
    mem_param_tmp_154782.references = NULL;
    
    struct memblock mem_param_tmp_154781;
    
    mem_param_tmp_154781.references = NULL;
    
    struct memblock mem_param_tmp_154780;
    
    mem_param_tmp_154780.references = NULL;
    
    struct memblock mem_param_tmp_154779;
    
    mem_param_tmp_154779.references = NULL;
    
    struct memblock mem_param_tmp_154778;
    
    mem_param_tmp_154778.references = NULL;
    
    struct memblock mem_param_tmp_154777;
    
    mem_param_tmp_154777.references = NULL;
    
    struct memblock mem_param_tmp_154776;
    
    mem_param_tmp_154776.references = NULL;
    
    struct memblock mem_param_tmp_154775;
    
    mem_param_tmp_154775.references = NULL;
    
    struct memblock mem_param_tmp_154774;
    
    mem_param_tmp_154774.references = NULL;
    
    struct memblock mem_param_tmp_154773;
    
    mem_param_tmp_154773.references = NULL;
    
    struct memblock mem_param_tmp_154772;
    
    mem_param_tmp_154772.references = NULL;
    
    struct memblock mem_param_tmp_154771;
    
    mem_param_tmp_154771.references = NULL;
    
    struct memblock mem_param_tmp_154770;
    
    mem_param_tmp_154770.references = NULL;
    
    struct memblock mem_param_tmp_154769;
    
    mem_param_tmp_154769.references = NULL;
    
    struct memblock mem_param_tmp_154768;
    
    mem_param_tmp_154768.references = NULL;
    
    struct memblock mem_param_tmp_154767;
    
    mem_param_tmp_154767.references = NULL;
    
    struct memblock mem_param_tmp_154766;
    
    mem_param_tmp_154766.references = NULL;
    
    struct memblock mem_param_tmp_154765;
    
    mem_param_tmp_154765.references = NULL;
    
    struct memblock mem_param_tmp_154764;
    
    mem_param_tmp_154764.references = NULL;
    
    struct memblock mem_param_tmp_154763;
    
    mem_param_tmp_154763.references = NULL;
    
    struct memblock mem_param_tmp_154762;
    
    mem_param_tmp_154762.references = NULL;
    
    struct memblock mem_param_tmp_154761;
    
    mem_param_tmp_154761.references = NULL;
    
    struct memblock ext_mem_154548;
    
    ext_mem_154548.references = NULL;
    
    struct memblock ext_mem_154549;
    
    ext_mem_154549.references = NULL;
    
    struct memblock ext_mem_154550;
    
    ext_mem_154550.references = NULL;
    
    struct memblock mem_154546;
    
    mem_154546.references = NULL;
    
    struct memblock mem_154544;
    
    mem_154544.references = NULL;
    
    struct memblock mem_154542;
    
    mem_154542.references = NULL;
    
    struct memblock mem_154540;
    
    mem_154540.references = NULL;
    
    struct memblock ext_mem_154537;
    
    ext_mem_154537.references = NULL;
    
    struct memblock ext_mem_154538;
    
    ext_mem_154538.references = NULL;
    
    struct memblock ext_mem_154539;
    
    ext_mem_154539.references = NULL;
    
    struct memblock mem_154535;
    
    mem_154535.references = NULL;
    
    struct memblock mem_154533;
    
    mem_154533.references = NULL;
    
    struct memblock mem_154531;
    
    mem_154531.references = NULL;
    
    struct memblock mem_154529;
    
    mem_154529.references = NULL;
    
    struct memblock ext_mem_154526;
    
    ext_mem_154526.references = NULL;
    
    struct memblock ext_mem_154527;
    
    ext_mem_154527.references = NULL;
    
    struct memblock ext_mem_154528;
    
    ext_mem_154528.references = NULL;
    
    struct memblock mem_154524;
    
    mem_154524.references = NULL;
    
    struct memblock mem_154522;
    
    mem_154522.references = NULL;
    
    struct memblock mem_154520;
    
    mem_154520.references = NULL;
    
    struct memblock mem_154518;
    
    mem_154518.references = NULL;
    
    struct memblock ext_mem_154515;
    
    ext_mem_154515.references = NULL;
    
    struct memblock ext_mem_154516;
    
    ext_mem_154516.references = NULL;
    
    struct memblock ext_mem_154517;
    
    ext_mem_154517.references = NULL;
    
    struct memblock mem_154513;
    
    mem_154513.references = NULL;
    
    struct memblock mem_154511;
    
    mem_154511.references = NULL;
    
    struct memblock mem_154509;
    
    mem_154509.references = NULL;
    
    struct memblock mem_154507;
    
    mem_154507.references = NULL;
    
    struct memblock ext_mem_154504;
    
    ext_mem_154504.references = NULL;
    
    struct memblock ext_mem_154505;
    
    ext_mem_154505.references = NULL;
    
    struct memblock ext_mem_154506;
    
    ext_mem_154506.references = NULL;
    
    struct memblock mem_154502;
    
    mem_154502.references = NULL;
    
    struct memblock mem_154500;
    
    mem_154500.references = NULL;
    
    struct memblock mem_154498;
    
    mem_154498.references = NULL;
    
    struct memblock mem_154496;
    
    mem_154496.references = NULL;
    
    struct memblock ext_mem_154493;
    
    ext_mem_154493.references = NULL;
    
    struct memblock ext_mem_154494;
    
    ext_mem_154494.references = NULL;
    
    struct memblock ext_mem_154495;
    
    ext_mem_154495.references = NULL;
    
    struct memblock mem_154491;
    
    mem_154491.references = NULL;
    
    struct memblock mem_154489;
    
    mem_154489.references = NULL;
    
    struct memblock mem_154487;
    
    mem_154487.references = NULL;
    
    struct memblock mem_154485;
    
    mem_154485.references = NULL;
    
    struct memblock ext_mem_154482;
    
    ext_mem_154482.references = NULL;
    
    struct memblock ext_mem_154483;
    
    ext_mem_154483.references = NULL;
    
    struct memblock ext_mem_154484;
    
    ext_mem_154484.references = NULL;
    
    struct memblock mem_154480;
    
    mem_154480.references = NULL;
    
    struct memblock mem_154478;
    
    mem_154478.references = NULL;
    
    struct memblock mem_154476;
    
    mem_154476.references = NULL;
    
    struct memblock mem_154474;
    
    mem_154474.references = NULL;
    
    struct memblock ext_mem_154471;
    
    ext_mem_154471.references = NULL;
    
    struct memblock ext_mem_154472;
    
    ext_mem_154472.references = NULL;
    
    struct memblock ext_mem_154473;
    
    ext_mem_154473.references = NULL;
    
    struct memblock mem_154469;
    
    mem_154469.references = NULL;
    
    struct memblock mem_154467;
    
    mem_154467.references = NULL;
    
    struct memblock mem_154465;
    
    mem_154465.references = NULL;
    
    struct memblock mem_154463;
    
    mem_154463.references = NULL;
    
    struct memblock ext_mem_154460;
    
    ext_mem_154460.references = NULL;
    
    struct memblock ext_mem_154461;
    
    ext_mem_154461.references = NULL;
    
    struct memblock ext_mem_154462;
    
    ext_mem_154462.references = NULL;
    
    struct memblock mem_154458;
    
    mem_154458.references = NULL;
    
    struct memblock mem_154456;
    
    mem_154456.references = NULL;
    
    struct memblock mem_154454;
    
    mem_154454.references = NULL;
    
    struct memblock mem_154452;
    
    mem_154452.references = NULL;
    
    struct memblock mem_param_152380;
    
    mem_param_152380.references = NULL;
    
    struct memblock mem_param_152376;
    
    mem_param_152376.references = NULL;
    
    struct memblock mem_param_152372;
    
    mem_param_152372.references = NULL;
    
    struct memblock mem_param_152368;
    
    mem_param_152368.references = NULL;
    
    struct memblock mem_param_152364;
    
    mem_param_152364.references = NULL;
    
    struct memblock mem_param_152360;
    
    mem_param_152360.references = NULL;
    
    struct memblock mem_param_152356;
    
    mem_param_152356.references = NULL;
    
    struct memblock mem_param_152352;
    
    mem_param_152352.references = NULL;
    
    struct memblock mem_param_152348;
    
    mem_param_152348.references = NULL;
    
    struct memblock mem_param_152344;
    
    mem_param_152344.references = NULL;
    
    struct memblock mem_param_152340;
    
    mem_param_152340.references = NULL;
    
    struct memblock mem_param_152336;
    
    mem_param_152336.references = NULL;
    
    struct memblock mem_param_152332;
    
    mem_param_152332.references = NULL;
    
    struct memblock mem_param_152328;
    
    mem_param_152328.references = NULL;
    
    struct memblock mem_param_152324;
    
    mem_param_152324.references = NULL;
    
    struct memblock mem_param_152320;
    
    mem_param_152320.references = NULL;
    
    struct memblock mem_param_152316;
    
    mem_param_152316.references = NULL;
    
    struct memblock mem_param_152312;
    
    mem_param_152312.references = NULL;
    
    struct memblock mem_param_152308;
    
    mem_param_152308.references = NULL;
    
    struct memblock mem_param_152304;
    
    mem_param_152304.references = NULL;
    
    struct memblock mem_param_152300;
    
    mem_param_152300.references = NULL;
    
    struct memblock mem_param_152296;
    
    mem_param_152296.references = NULL;
    
    struct memblock mem_param_152292;
    
    mem_param_152292.references = NULL;
    
    struct memblock mem_param_152288;
    
    mem_param_152288.references = NULL;
    
    struct memblock mem_param_152284;
    
    mem_param_152284.references = NULL;
    
    struct memblock mem_param_152280;
    
    mem_param_152280.references = NULL;
    
    struct memblock mem_param_152276;
    
    mem_param_152276.references = NULL;
    
    struct memblock ext_mem_154632;
    
    ext_mem_154632.references = NULL;
    
    struct memblock ext_mem_154633;
    
    ext_mem_154633.references = NULL;
    
    struct memblock ext_mem_154634;
    
    ext_mem_154634.references = NULL;
    
    struct memblock ext_mem_154635;
    
    ext_mem_154635.references = NULL;
    
    struct memblock ext_mem_154636;
    
    ext_mem_154636.references = NULL;
    
    struct memblock ext_mem_154637;
    
    ext_mem_154637.references = NULL;
    
    struct memblock ext_mem_154638;
    
    ext_mem_154638.references = NULL;
    
    struct memblock ext_mem_154639;
    
    ext_mem_154639.references = NULL;
    
    struct memblock ext_mem_154640;
    
    ext_mem_154640.references = NULL;
    
    struct memblock ext_mem_154641;
    
    ext_mem_154641.references = NULL;
    
    struct memblock ext_mem_154642;
    
    ext_mem_154642.references = NULL;
    
    struct memblock ext_mem_154643;
    
    ext_mem_154643.references = NULL;
    
    struct memblock ext_mem_154644;
    
    ext_mem_154644.references = NULL;
    
    struct memblock ext_mem_154645;
    
    ext_mem_154645.references = NULL;
    
    struct memblock ext_mem_154646;
    
    ext_mem_154646.references = NULL;
    
    struct memblock ext_mem_154647;
    
    ext_mem_154647.references = NULL;
    
    struct memblock ext_mem_154648;
    
    ext_mem_154648.references = NULL;
    
    struct memblock ext_mem_154649;
    
    ext_mem_154649.references = NULL;
    
    struct memblock ext_mem_154650;
    
    ext_mem_154650.references = NULL;
    
    struct memblock ext_mem_154651;
    
    ext_mem_154651.references = NULL;
    
    struct memblock ext_mem_154652;
    
    ext_mem_154652.references = NULL;
    
    struct memblock ext_mem_154653;
    
    ext_mem_154653.references = NULL;
    
    struct memblock ext_mem_154654;
    
    ext_mem_154654.references = NULL;
    
    struct memblock ext_mem_154655;
    
    ext_mem_154655.references = NULL;
    
    struct memblock ext_mem_154656;
    
    ext_mem_154656.references = NULL;
    
    struct memblock ext_mem_154657;
    
    ext_mem_154657.references = NULL;
    
    struct memblock ext_mem_154658;
    
    ext_mem_154658.references = NULL;
    
    struct memblock mem_out_154757;
    
    mem_out_154757.references = NULL;
    
    struct memblock mem_out_154756;
    
    mem_out_154756.references = NULL;
    
    struct memblock mem_out_154755;
    
    mem_out_154755.references = NULL;
    
    struct memblock mem_out_154754;
    
    mem_out_154754.references = NULL;
    
    struct memblock mem_out_154753;
    
    mem_out_154753.references = NULL;
    
    struct memblock mem_out_154752;
    
    mem_out_154752.references = NULL;
    
    struct memblock mem_out_154751;
    
    mem_out_154751.references = NULL;
    
    struct memblock mem_out_154750;
    
    mem_out_154750.references = NULL;
    
    struct memblock mem_out_154749;
    
    mem_out_154749.references = NULL;
    
    struct memblock mem_out_154748;
    
    mem_out_154748.references = NULL;
    
    struct memblock mem_out_154747;
    
    mem_out_154747.references = NULL;
    
    struct memblock mem_out_154746;
    
    mem_out_154746.references = NULL;
    
    struct memblock mem_out_154745;
    
    mem_out_154745.references = NULL;
    
    struct memblock mem_out_154744;
    
    mem_out_154744.references = NULL;
    
    struct memblock mem_out_154743;
    
    mem_out_154743.references = NULL;
    
    struct memblock mem_out_154742;
    
    mem_out_154742.references = NULL;
    
    struct memblock mem_out_154741;
    
    mem_out_154741.references = NULL;
    
    struct memblock mem_out_154740;
    
    mem_out_154740.references = NULL;
    
    struct memblock mem_out_154739;
    
    mem_out_154739.references = NULL;
    
    struct memblock mem_out_154738;
    
    mem_out_154738.references = NULL;
    
    struct memblock mem_out_154737;
    
    mem_out_154737.references = NULL;
    
    struct memblock mem_out_154736;
    
    mem_out_154736.references = NULL;
    
    struct memblock mem_out_154735;
    
    mem_out_154735.references = NULL;
    
    struct memblock mem_out_154734;
    
    mem_out_154734.references = NULL;
    
    struct memblock mem_out_154733;
    
    mem_out_154733.references = NULL;
    
    struct memblock mem_out_154732;
    
    mem_out_154732.references = NULL;
    
    struct memblock mem_out_154731;
    
    mem_out_154731.references = NULL;
    
    struct memblock mem_152234 = ctx->constants->mem_152234;
    struct memblock mem_152235 = ctx->constants->mem_152235;
    struct memblock mem_152236 = ctx->constants->mem_152236;
    struct memblock mem_152237 = ctx->constants->mem_152237;
    struct memblock mem_152238 = ctx->constants->mem_152238;
    struct memblock mem_152239 = ctx->constants->mem_152239;
    struct memblock mem_152240 = ctx->constants->mem_152240;
    struct memblock mem_152241 = ctx->constants->mem_152241;
    struct memblock mem_152242 = ctx->constants->mem_152242;
    
    // futhark/microgpt.fut:71:13-49
    
    double defunc_0_lifted_lambda_res_138357;
    double r_138359 = 0.0;
    
    for (int64_t i_138358 = 0; i_138358 < (int64_t) 27; i_138358++) {
        // futhark/microgpt.fut:71:40-49
        
        double zp_res_138360 = 1.0 + r_138359;
        double r_tmp_154758 = zp_res_138360;
        
        r_138359 = r_tmp_154758;
    }
    defunc_0_lifted_lambda_res_138357 = r_138359;
    // futhark/microgpt.fut:71:13-49
    
    double defunc_0_lifted_lambda_res_139130;
    double r_139132 = 0.0;
    
    for (int64_t i_139131 = 0; i_139131 < (int64_t) 16; i_139131++) {
        // futhark/microgpt.fut:71:40-49
        
        double zp_res_139133 = 1.0 + r_139132;
        double r_tmp_154759 = zp_res_139133;
        
        r_139132 = r_tmp_154759;
    }
    defunc_0_lifted_lambda_res_139130 = r_139132;
    // futhark/microgpt.fut:71:13-49
    
    double defunc_0_lifted_lambda_res_139559;
    double r_139561 = 0.0;
    
    for (int64_t i_139560 = 0; i_139560 < (int64_t) 16; i_139560++) {
        // futhark/microgpt.fut:71:40-49
        
        double zp_res_139562 = 1.0 + r_139561;
        double r_tmp_154760 = zp_res_139562;
        
        r_139561 = r_tmp_154760;
    }
    defunc_0_lifted_lambda_res_139559 = r_139561;
    // futhark/microgpt.fut:4:11-25
    if (mem_152381_cached_sizze_155314 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152381, &mem_152381_cached_sizze_155314, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152382_cached_sizze_155315 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_152382, &mem_152382_cached_sizze_155315, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152391_cached_sizze_155316 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_152391, &mem_152391_cached_sizze_155316, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152398_cached_sizze_155317 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152398, &mem_152398_cached_sizze_155317, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152413_cached_sizze_155318 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152413, &mem_152413_cached_sizze_155318, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152414_cached_sizze_155319 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152414, &mem_152414_cached_sizze_155319, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152415_cached_sizze_155320 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152415, &mem_152415_cached_sizze_155320, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152434_cached_sizze_155321 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152434, &mem_152434_cached_sizze_155321, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152441_cached_sizze_155322 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152441, &mem_152441_cached_sizze_155322, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152446_cached_sizze_155323 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152446, &mem_152446_cached_sizze_155323, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152457_cached_sizze_155324 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152457, &mem_152457_cached_sizze_155324, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152462_cached_sizze_155325 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152462, &mem_152462_cached_sizze_155325, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152473_cached_sizze_155326 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152473, &mem_152473_cached_sizze_155326, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152474_cached_sizze_155327 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152474, &mem_152474_cached_sizze_155327, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152487_cached_sizze_155328 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152487, &mem_152487_cached_sizze_155328, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152494_cached_sizze_155329 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152494, &mem_152494_cached_sizze_155329, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152499_cached_sizze_155330 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152499, &mem_152499_cached_sizze_155330, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152510_cached_sizze_155331 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152510, &mem_152510_cached_sizze_155331, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152515_cached_sizze_155332 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152515, &mem_152515_cached_sizze_155332, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152526_cached_sizze_155333 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152526, &mem_152526_cached_sizze_155333, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152527_cached_sizze_155334 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152527, &mem_152527_cached_sizze_155334, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152528_cached_sizze_155335 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152528, &mem_152528_cached_sizze_155335, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152544_cached_sizze_155336 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152544, &mem_152544_cached_sizze_155336, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152545_cached_sizze_155337 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152545, &mem_152545_cached_sizze_155337, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152546_cached_sizze_155338 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152546, &mem_152546_cached_sizze_155338, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152559_cached_sizze_155339 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152559, &mem_152559_cached_sizze_155339, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152560_cached_sizze_155340 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152560, &mem_152560_cached_sizze_155340, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152561_cached_sizze_155341 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_152561, &mem_152561_cached_sizze_155341, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152607_cached_sizze_155342 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_152607, &mem_152607_cached_sizze_155342, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152608_cached_sizze_155343 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_152608, &mem_152608_cached_sizze_155343, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152609_cached_sizze_155344 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_152609, &mem_152609_cached_sizze_155344, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152610_cached_sizze_155345 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_152610, &mem_152610_cached_sizze_155345, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152631_cached_sizze_155346 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152631, &mem_152631_cached_sizze_155346, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152632_cached_sizze_155347 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152632, &mem_152632_cached_sizze_155347, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152633_cached_sizze_155348 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152633, &mem_152633_cached_sizze_155348, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152634_cached_sizze_155349 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152634, &mem_152634_cached_sizze_155349, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152651_cached_sizze_155350 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152651, &mem_152651_cached_sizze_155350, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152652_cached_sizze_155351 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152652, &mem_152652_cached_sizze_155351, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152653_cached_sizze_155352 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152653, &mem_152653_cached_sizze_155352, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152654_cached_sizze_155353 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152654, &mem_152654_cached_sizze_155353, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152715_cached_sizze_155354 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_152715, &mem_152715_cached_sizze_155354, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152716_cached_sizze_155355 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_152716, &mem_152716_cached_sizze_155355, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152717_cached_sizze_155356 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_152717, &mem_152717_cached_sizze_155356, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152718_cached_sizze_155357 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_152718, &mem_152718_cached_sizze_155357, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152739_cached_sizze_155358 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152739, &mem_152739_cached_sizze_155358, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152740_cached_sizze_155359 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152740, &mem_152740_cached_sizze_155359, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152741_cached_sizze_155360 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152741, &mem_152741_cached_sizze_155360, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152742_cached_sizze_155361 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152742, &mem_152742_cached_sizze_155361, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152759_cached_sizze_155362 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152759, &mem_152759_cached_sizze_155362, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152760_cached_sizze_155363 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152760, &mem_152760_cached_sizze_155363, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152761_cached_sizze_155364 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152761, &mem_152761_cached_sizze_155364, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152762_cached_sizze_155365 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152762, &mem_152762_cached_sizze_155365, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152823_cached_sizze_155366 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152823, &mem_152823_cached_sizze_155366, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152824_cached_sizze_155367 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152824, &mem_152824_cached_sizze_155367, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152825_cached_sizze_155368 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152825, &mem_152825_cached_sizze_155368, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152826_cached_sizze_155369 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152826, &mem_152826_cached_sizze_155369, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152827_cached_sizze_155370 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152827, &mem_152827_cached_sizze_155370, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152828_cached_sizze_155371 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152828, &mem_152828_cached_sizze_155371, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152829_cached_sizze_155372 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152829, &mem_152829_cached_sizze_155372, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152830_cached_sizze_155373 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_152830, &mem_152830_cached_sizze_155373, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152863_cached_sizze_155374 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152863, &mem_152863_cached_sizze_155374, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152864_cached_sizze_155375 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152864, &mem_152864_cached_sizze_155375, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152865_cached_sizze_155376 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152865, &mem_152865_cached_sizze_155376, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152866_cached_sizze_155377 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152866, &mem_152866_cached_sizze_155377, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152867_cached_sizze_155378 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152867, &mem_152867_cached_sizze_155378, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152868_cached_sizze_155379 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152868, &mem_152868_cached_sizze_155379, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152869_cached_sizze_155380 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152869, &mem_152869_cached_sizze_155380, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152870_cached_sizze_155381 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152870, &mem_152870_cached_sizze_155381, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152951_cached_sizze_155382 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_152951, &mem_152951_cached_sizze_155382, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152952_cached_sizze_155383 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_152952, &mem_152952_cached_sizze_155383, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152953_cached_sizze_155384 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_152953, &mem_152953_cached_sizze_155384, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152954_cached_sizze_155385 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_152954, &mem_152954_cached_sizze_155385, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152975_cached_sizze_155386 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152975, &mem_152975_cached_sizze_155386, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152976_cached_sizze_155387 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152976, &mem_152976_cached_sizze_155387, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152977_cached_sizze_155388 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152977, &mem_152977_cached_sizze_155388, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152978_cached_sizze_155389 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_152978, &mem_152978_cached_sizze_155389, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152995_cached_sizze_155390 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152995, &mem_152995_cached_sizze_155390, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152996_cached_sizze_155391 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152996, &mem_152996_cached_sizze_155391, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152997_cached_sizze_155392 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152997, &mem_152997_cached_sizze_155392, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_152998_cached_sizze_155393 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_152998, &mem_152998_cached_sizze_155393, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153059_cached_sizze_155394 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153059, &mem_153059_cached_sizze_155394, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153060_cached_sizze_155395 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153060, &mem_153060_cached_sizze_155395, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153069_cached_sizze_155396 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153069, &mem_153069_cached_sizze_155396, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153070_cached_sizze_155397 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153070, &mem_153070_cached_sizze_155397, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153091_cached_sizze_155398 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153091, &mem_153091_cached_sizze_155398, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153092_cached_sizze_155399 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153092, &mem_153092_cached_sizze_155399, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153103_cached_sizze_155400 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153103, &mem_153103_cached_sizze_155400, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153104_cached_sizze_155401 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153104, &mem_153104_cached_sizze_155401, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153113_cached_sizze_155402 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153113, &mem_153113_cached_sizze_155402, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153114_cached_sizze_155403 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153114, &mem_153114_cached_sizze_155403, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153145_cached_sizze_155404 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153145, &mem_153145_cached_sizze_155404, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153146_cached_sizze_155405 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153146, &mem_153146_cached_sizze_155405, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153157_cached_sizze_155406 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153157, &mem_153157_cached_sizze_155406, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153158_cached_sizze_155407 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153158, &mem_153158_cached_sizze_155407, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153167_cached_sizze_155408 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153167, &mem_153167_cached_sizze_155408, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153168_cached_sizze_155409 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153168, &mem_153168_cached_sizze_155409, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153199_cached_sizze_155410 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153199, &mem_153199_cached_sizze_155410, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153205_cached_sizze_155411 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153205, &mem_153205_cached_sizze_155411, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153210_cached_sizze_155412 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_153210, &mem_153210_cached_sizze_155412, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153226_cached_sizze_155413 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153226, &mem_153226_cached_sizze_155413, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153231_cached_sizze_155414 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153231, &mem_153231_cached_sizze_155414, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153242_cached_sizze_155415 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153242, &mem_153242_cached_sizze_155415, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153247_cached_sizze_155416 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153247, &mem_153247_cached_sizze_155416, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153258_cached_sizze_155417 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153258, &mem_153258_cached_sizze_155417, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153259_cached_sizze_155418 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153259, &mem_153259_cached_sizze_155418, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153272_cached_sizze_155419 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153272, &mem_153272_cached_sizze_155419, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153279_cached_sizze_155420 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153279, &mem_153279_cached_sizze_155420, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153284_cached_sizze_155421 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153284, &mem_153284_cached_sizze_155421, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153295_cached_sizze_155422 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153295, &mem_153295_cached_sizze_155422, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153300_cached_sizze_155423 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153300, &mem_153300_cached_sizze_155423, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153311_cached_sizze_155424 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153311, &mem_153311_cached_sizze_155424, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153316_cached_sizze_155425 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153316, &mem_153316_cached_sizze_155425, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153327_cached_sizze_155426 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153327, &mem_153327_cached_sizze_155426, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153332_cached_sizze_155427 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153332, &mem_153332_cached_sizze_155427, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153343_cached_sizze_155428 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153343, &mem_153343_cached_sizze_155428, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153348_cached_sizze_155429 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153348, &mem_153348_cached_sizze_155429, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153359_cached_sizze_155430 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_153359, &mem_153359_cached_sizze_155430, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153364_cached_sizze_155431 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_153364, &mem_153364_cached_sizze_155431, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153375_cached_sizze_155432 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_153375, &mem_153375_cached_sizze_155432, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153376_cached_sizze_155433 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_153376, &mem_153376_cached_sizze_155433, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153377_cached_sizze_155434 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_153377, &mem_153377_cached_sizze_155434, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153378_cached_sizze_155435 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_153378, &mem_153378_cached_sizze_155435, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:115:13-33
    if (mem_153396_cached_sizze_155436 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_153396, &mem_153396_cached_sizze_155436, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153401_cached_sizze_155437 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_153401, &mem_153401_cached_sizze_155437, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153446_cached_sizze_155440 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_153446, &mem_153446_cached_sizze_155440, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153452_cached_sizze_155441 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_153452, &mem_153452_cached_sizze_155441, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153457_cached_sizze_155442 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_153457, &mem_153457_cached_sizze_155442, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153473_cached_sizze_155443 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_153473, &mem_153473_cached_sizze_155443, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153474_cached_sizze_155444 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_153474, &mem_153474_cached_sizze_155444, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153483_cached_sizze_155445 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_153483, &mem_153483_cached_sizze_155445, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153484_cached_sizze_155446 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_153484, &mem_153484_cached_sizze_155446, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153505_cached_sizze_155447 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_153505, &mem_153505_cached_sizze_155447, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153511_cached_sizze_155448 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_153511, &mem_153511_cached_sizze_155448, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153516_cached_sizze_155449 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_153516, &mem_153516_cached_sizze_155449, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153532_cached_sizze_155450 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_153532, &mem_153532_cached_sizze_155450, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153537_cached_sizze_155451 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_153537, &mem_153537_cached_sizze_155451, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153548_cached_sizze_155452 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_153548, &mem_153548_cached_sizze_155452, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153553_cached_sizze_155453 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_153553, &mem_153553_cached_sizze_155453, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153564_cached_sizze_155454 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153564, &mem_153564_cached_sizze_155454, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153569_cached_sizze_155455 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153569, &mem_153569_cached_sizze_155455, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153580_cached_sizze_155456 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153580, &mem_153580_cached_sizze_155456, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153581_cached_sizze_155457 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153581, &mem_153581_cached_sizze_155457, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153590_cached_sizze_155458 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153590, &mem_153590_cached_sizze_155458, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153591_cached_sizze_155459 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153591, &mem_153591_cached_sizze_155459, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153612_cached_sizze_155460 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153612, &mem_153612_cached_sizze_155460, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153617_cached_sizze_155461 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153617, &mem_153617_cached_sizze_155461, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153628_cached_sizze_155462 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153628, &mem_153628_cached_sizze_155462, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153629_cached_sizze_155463 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153629, &mem_153629_cached_sizze_155463, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153642_cached_sizze_155464 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153642, &mem_153642_cached_sizze_155464, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153649_cached_sizze_155465 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153649, &mem_153649_cached_sizze_155465, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153654_cached_sizze_155466 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153654, &mem_153654_cached_sizze_155466, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153665_cached_sizze_155467 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153665, &mem_153665_cached_sizze_155467, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153671_cached_sizze_155468 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153671, &mem_153671_cached_sizze_155468, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153676_cached_sizze_155469 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_153676, &mem_153676_cached_sizze_155469, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153692_cached_sizze_155470 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153692, &mem_153692_cached_sizze_155470, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153693_cached_sizze_155471 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153693, &mem_153693_cached_sizze_155471, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153694_cached_sizze_155472 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153694, &mem_153694_cached_sizze_155472, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153710_cached_sizze_155473 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153710, &mem_153710_cached_sizze_155473, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153711_cached_sizze_155474 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153711, &mem_153711_cached_sizze_155474, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153712_cached_sizze_155475 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153712, &mem_153712_cached_sizze_155475, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153725_cached_sizze_155476 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_153725, &mem_153725_cached_sizze_155476, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153726_cached_sizze_155477 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_153726, &mem_153726_cached_sizze_155477, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153767_cached_sizze_155478 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153767, &mem_153767_cached_sizze_155478, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153768_cached_sizze_155479 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153768, &mem_153768_cached_sizze_155479, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153779_cached_sizze_155480 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153779, &mem_153779_cached_sizze_155480, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153780_cached_sizze_155481 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153780, &mem_153780_cached_sizze_155481, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153789_cached_sizze_155482 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153789, &mem_153789_cached_sizze_155482, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153790_cached_sizze_155483 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153790, &mem_153790_cached_sizze_155483, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153821_cached_sizze_155484 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153821, &mem_153821_cached_sizze_155484, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153822_cached_sizze_155485 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153822, &mem_153822_cached_sizze_155485, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153833_cached_sizze_155486 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153833, &mem_153833_cached_sizze_155486, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153834_cached_sizze_155487 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153834, &mem_153834_cached_sizze_155487, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153843_cached_sizze_155488 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153843, &mem_153843_cached_sizze_155488, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153844_cached_sizze_155489 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153844, &mem_153844_cached_sizze_155489, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153875_cached_sizze_155490 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153875, &mem_153875_cached_sizze_155490, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153876_cached_sizze_155491 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153876, &mem_153876_cached_sizze_155491, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153877_cached_sizze_155492 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153877, &mem_153877_cached_sizze_155492, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153878_cached_sizze_155493 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153878, &mem_153878_cached_sizze_155493, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153895_cached_sizze_155494 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153895, &mem_153895_cached_sizze_155494, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153896_cached_sizze_155495 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153896, &mem_153896_cached_sizze_155495, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153897_cached_sizze_155496 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153897, &mem_153897_cached_sizze_155496, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153898_cached_sizze_155497 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153898, &mem_153898_cached_sizze_155497, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153939_cached_sizze_155498 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153939, &mem_153939_cached_sizze_155498, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153940_cached_sizze_155499 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_153940, &mem_153940_cached_sizze_155499, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153951_cached_sizze_155500 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153951, &mem_153951_cached_sizze_155500, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153952_cached_sizze_155501 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_153952, &mem_153952_cached_sizze_155501, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153961_cached_sizze_155502 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153961, &mem_153961_cached_sizze_155502, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153962_cached_sizze_155503 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_153962, &mem_153962_cached_sizze_155503, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153993_cached_sizze_155504 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153993, &mem_153993_cached_sizze_155504, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_153994_cached_sizze_155505 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_153994, &mem_153994_cached_sizze_155505, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154003_cached_sizze_155506 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154003, &mem_154003_cached_sizze_155506, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154004_cached_sizze_155507 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154004, &mem_154004_cached_sizze_155507, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154025_cached_sizze_155508 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_154025, &mem_154025_cached_sizze_155508, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154026_cached_sizze_155509 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_154026, &mem_154026_cached_sizze_155509, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154037_cached_sizze_155510 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154037, &mem_154037_cached_sizze_155510, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154038_cached_sizze_155511 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154038, &mem_154038_cached_sizze_155511, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154047_cached_sizze_155512 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154047, &mem_154047_cached_sizze_155512, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154048_cached_sizze_155513 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154048, &mem_154048_cached_sizze_155513, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154079_cached_sizze_155514 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_154079, &mem_154079_cached_sizze_155514, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154080_cached_sizze_155515 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_154080, &mem_154080_cached_sizze_155515, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154091_cached_sizze_155516 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154091, &mem_154091_cached_sizze_155516, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154092_cached_sizze_155517 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154092, &mem_154092_cached_sizze_155517, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154101_cached_sizze_155518 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154101, &mem_154101_cached_sizze_155518, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154102_cached_sizze_155519 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154102, &mem_154102_cached_sizze_155519, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154133_cached_sizze_155520 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154133, &mem_154133_cached_sizze_155520, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154134_cached_sizze_155521 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154134, &mem_154134_cached_sizze_155521, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154135_cached_sizze_155522 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154135, &mem_154135_cached_sizze_155522, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154136_cached_sizze_155523 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154136, &mem_154136_cached_sizze_155523, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154153_cached_sizze_155524 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154153, &mem_154153_cached_sizze_155524, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154154_cached_sizze_155525 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154154, &mem_154154_cached_sizze_155525, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154155_cached_sizze_155526 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154155, &mem_154155_cached_sizze_155526, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154156_cached_sizze_155527 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154156, &mem_154156_cached_sizze_155527, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154197_cached_sizze_155528 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154197, &mem_154197_cached_sizze_155528, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154202_cached_sizze_155529 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154202, &mem_154202_cached_sizze_155529, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154213_cached_sizze_155530 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154213, &mem_154213_cached_sizze_155530, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154214_cached_sizze_155531 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154214, &mem_154214_cached_sizze_155531, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154215_cached_sizze_155532 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154215, &mem_154215_cached_sizze_155532, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154216_cached_sizze_155533 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154216, &mem_154216_cached_sizze_155533, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154217_cached_sizze_155534 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154217, &mem_154217_cached_sizze_155534, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154236_cached_sizze_155535 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154236, &mem_154236_cached_sizze_155535, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154237_cached_sizze_155536 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154237, &mem_154237_cached_sizze_155536, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154238_cached_sizze_155537 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154238, &mem_154238_cached_sizze_155537, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154275_cached_sizze_155538 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154275, &mem_154275_cached_sizze_155538, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154282_cached_sizze_155539 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154282, &mem_154282_cached_sizze_155539, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154287_cached_sizze_155540 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154287, &mem_154287_cached_sizze_155540, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154298_cached_sizze_155541 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154298, &mem_154298_cached_sizze_155541, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154299_cached_sizze_155542 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154299, &mem_154299_cached_sizze_155542, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154308_cached_sizze_155543 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154308, &mem_154308_cached_sizze_155543, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154309_cached_sizze_155544 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154309, &mem_154309_cached_sizze_155544, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154330_cached_sizze_155545 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154330, &mem_154330_cached_sizze_155545, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154331_cached_sizze_155546 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154331, &mem_154331_cached_sizze_155546, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154332_cached_sizze_155547 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154332, &mem_154332_cached_sizze_155547, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154333_cached_sizze_155548 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154333, &mem_154333_cached_sizze_155548, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154358_cached_sizze_155549 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154358, &mem_154358_cached_sizze_155549, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154359_cached_sizze_155550 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154359, &mem_154359_cached_sizze_155550, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154372_cached_sizze_155551 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154372, &mem_154372_cached_sizze_155551, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154373_cached_sizze_155552 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_154373, &mem_154373_cached_sizze_155552, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154382_cached_sizze_155553 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154382, &mem_154382_cached_sizze_155553, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154383_cached_sizze_155554 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154383, &mem_154383_cached_sizze_155554, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154404_cached_sizze_155555 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_154404, &mem_154404_cached_sizze_155555, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154409_cached_sizze_155556 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154409, &mem_154409_cached_sizze_155556, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154420_cached_sizze_155557 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_154420, &mem_154420_cached_sizze_155557, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154421_cached_sizze_155558 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_154421, &mem_154421_cached_sizze_155558, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154430_cached_sizze_155559 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154430, &mem_154430_cached_sizze_155559, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_154431_cached_sizze_155560 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_154431, &mem_154431_cached_sizze_155560, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:628:5-633:51
    if (memblock_set(ctx, &mem_param_152276, &wdown_mem_152243, "wdown_mem_152243") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152280, &wkey_mem_152244, "wkey_mem_152244") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152284, &wout_mem_152245, "wout_mem_152245") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152288, &wpe_mem_152246, "wpe_mem_152246") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152292, &wqry_mem_152247, "wqry_mem_152247") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152296, &wte_mem_152248, "wte_mem_152248") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152300, &wup_mem_152249, "wup_mem_152249") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152304, &wval_mem_152250, "wval_mem_152250") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152308, &wvoc_mem_152251, "wvoc_mem_152251") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152312, &wdown_mem_152252, "wdown_mem_152252") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152316, &wkey_mem_152253, "wkey_mem_152253") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152320, &wout_mem_152254, "wout_mem_152254") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152324, &wpe_mem_152255, "wpe_mem_152255") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152328, &wqry_mem_152256, "wqry_mem_152256") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152332, &wte_mem_152257, "wte_mem_152257") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152336, &wup_mem_152258, "wup_mem_152258") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152340, &wval_mem_152259, "wval_mem_152259") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152344, &wvoc_mem_152260, "wvoc_mem_152260") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152348, &wdown_mem_152261, "wdown_mem_152261") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152352, &wkey_mem_152262, "wkey_mem_152262") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152356, &wout_mem_152263, "wout_mem_152263") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152360, &wpe_mem_152264, "wpe_mem_152264") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152364, &wqry_mem_152265, "wqry_mem_152265") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152368, &wte_mem_152266, "wte_mem_152266") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152372, &wup_mem_152267, "wup_mem_152267") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152376, &wval_mem_152268, "wval_mem_152268") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_152380, &wvoc_mem_152269, "wvoc_mem_152269") != 0)
        return 1;
    for (int64_t step_137468 = 0; step_137468 < (int64_t) 5; step_137468++) {
        // futhark/microgpt.fut:630:16-25
        
        int64_t dl_137496 = ((int64_t *) dls_mem_152271.mem)[step_137468];
        
        // futhark/microgpt.fut:470:37-40
        
        int64_t zl_rhs_137501 = sub64(dl_137496, (int64_t) 1);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151103 = 0; i_151103 < (int64_t) 16; i_151103++) {
            // futhark/microgpt.fut:470:25-81
            
            bool cond_141362 = slt64(i_151103, zl_rhs_137501);
            
            // futhark/microgpt.fut:470:56-59
            
            int64_t zeze_lhs_141363 = add64((int64_t) 1, i_151103);
            
            // futhark/microgpt.fut:470:47-60
            
            bool x_141364 = sle64((int64_t) 0, zeze_lhs_141363);
            
            // futhark/microgpt.fut:470:47-60
            
            bool y_141365 = slt64(zeze_lhs_141363, (int64_t) 16);
            
            // futhark/microgpt.fut:470:47-60
            
            bool bounds_check_141366 = x_141364 && y_141365;
            
            // futhark/microgpt.fut:9:27-39
            
            bool loop_not_taken_141367 = !cond_141362;
            
            // futhark/microgpt.fut:9:27-39
            
            bool protect_assert_disj_141368 = bounds_check_141366 || loop_not_taken_141367;
            
            // futhark/microgpt.fut:470:47-60
            
            bool index_certs_141369;
            
            if (!protect_assert_disj_141368) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_141363, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:470:47-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:470:3-83\n   #6  futhark/microgpt.fut:577:18-38\n   #7  futhark/microgpt.fut:599:26-605:31\n   #8  futhark/microgpt.fut:633:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            int64_t tmp_141384 = ((int64_t *) seqs_mem_152272.mem)[step_137468 * (int64_t) 16 + i_151103];
            
            // futhark/microgpt.fut:579:37-51
            
            bool x_141385 = sle64((int64_t) 0, tmp_141384);
            
            // futhark/microgpt.fut:579:37-51
            
            bool y_141386 = slt64(tmp_141384, (int64_t) 27);
            
            // futhark/microgpt.fut:579:37-51
            
            bool bounds_check_141387 = x_141385 && y_141386;
            
            // futhark/microgpt.fut:579:37-51
            
            bool index_certs_141388;
            
            if (!bounds_check_141387) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_141384, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:579:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:579:16-55\n   #6  futhark/microgpt.fut:599:26-605:31\n   #7  futhark/microgpt.fut:633:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:470:47-60
            
            int64_t zeze_lhs_141370;
            
            if (cond_141362) {
                int64_t x_150769 = ((int64_t *) seqs_mem_152272.mem)[step_137468 * (int64_t) 16 + zeze_lhs_141363];
                
                zeze_lhs_141370 = x_150769;
            } else {
                zeze_lhs_141370 = (int64_t) 0;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151093 = 0; i_151093 < (int64_t) 27; i_151093++) {
                // futhark/microgpt.fut:470:61-65
                
                bool cond_t_res_141374 = zeze_lhs_141370 == i_151093;
                
                // futhark/microgpt.fut:9:27-39
                
                bool x_141375 = cond_141362 && cond_t_res_141374;
                
                // futhark/microgpt.fut:470:25-81
                
                double lifted_lambda_res_141376;
                
                if (x_141375) {
                    lifted_lambda_res_141376 = 1.0;
                } else {
                    lifted_lambda_res_141376 = 0.0;
                }
                ((double *) mem_152391)[i_151093] = lifted_lambda_res_141376;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151097 = 0; i_151097 < (int64_t) 16; i_151097++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_141395 = ((double *) mem_param_152296.mem)[tmp_141384 * (int64_t) 16 + i_151097];
                
                ((double *) mem_152398)[i_151097] = lifted_lambda_res_141395;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152381, i_151103 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152398, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152382, i_151103 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152391, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151112 = 0; i_151112 < (int64_t) 16; i_151112++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_141461;
            double r_141463 = 0.0;
            
            for (int64_t i_141462 = 0; i_141462 < (int64_t) 16; i_141462++) {
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_141464 = ((double *) mem_param_152288.mem)[i_151112 * (int64_t) 16 + i_141462];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_rhs_141465 = ((double *) mem_152381)[i_151112 * (int64_t) 16 + i_141462];
                
                // futhark/microgpt.fut:279:63-99
                
                double zp_res_141466 = zp_lhs_141464 + zp_rhs_141465;
                
                // futhark/microgpt.fut:279:79-142
                
                double zt_res_141467 = zp_res_141466 * zp_res_141466;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_141468 = r_141463 + zt_res_141467;
                double r_tmp_154822 = zp_res_141468;
                
                r_141463 = r_tmp_154822;
            }
            defunc_0_lifted_lambda_res_141461 = r_141463;
            // futhark/microgpt.fut:279:42-161
            
            double zs_res_141469 = defunc_0_lifted_lambda_res_141461 / 16.0;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_141476;
            double r_141478 = 0.0;
            
            for (int64_t i_141477 = 0; i_141477 < (int64_t) 16; i_141477++) {
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_141479 = ((double *) mem_param_152288.mem)[i_151112 * (int64_t) 16 + i_141477];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_rhs_141480 = ((double *) mem_152381)[i_151112 * (int64_t) 16 + i_141477];
                
                // futhark/microgpt.fut:395:71-115
                
                double zp_res_141481 = zp_lhs_141479 + zp_rhs_141480;
                
                // futhark/microgpt.fut:395:91-166
                
                double zt_res_141482 = zp_res_141481 * zp_res_141481;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_141483 = r_141478 + zt_res_141482;
                double r_tmp_154823 = zp_res_141483;
                
                r_141478 = r_tmp_154823;
            }
            defunc_0_lifted_lambda_res_141476 = r_141478;
            // futhark/microgpt.fut:395:48-185
            
            double zs_res_141484 = defunc_0_lifted_lambda_res_141476 / 16.0;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_141494;
            double r_141496 = 0.0;
            
            for (int64_t i_141495 = 0; i_141495 < (int64_t) 16; i_141495++) {
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_141497 = ((double *) mem_param_152288.mem)[i_151112 * (int64_t) 16 + i_141495];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_rhs_141498 = ((double *) mem_152381)[i_151112 * (int64_t) 16 + i_141495];
                
                // futhark/microgpt.fut:408:72-116
                
                double zp_res_141499 = zp_lhs_141497 + zp_rhs_141498;
                
                // futhark/microgpt.fut:408:92-167
                
                double zt_res_141500 = zp_res_141499 * zp_res_141499;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_141501 = r_141496 + zt_res_141500;
                double r_tmp_154824 = zp_res_141501;
                
                r_141496 = r_tmp_154824;
            }
            defunc_0_lifted_lambda_res_141494 = r_141496;
            // futhark/microgpt.fut:408:49-186
            
            double zs_res_141502 = defunc_0_lifted_lambda_res_141494 / 16.0;
            
            ((double *) mem_152413)[i_151112] = zs_res_141502;
            ((double *) mem_152414)[i_151112] = zs_res_141484;
            ((double *) mem_152415)[i_151112] = zs_res_141469;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151118 = 0; i_151118 < (int64_t) 16; i_151118++) {
            // futhark/microgpt.fut:280:43-51
            
            double zp_lhs_137563 = ((double *) mem_152415)[i_151118];
            
            // futhark/microgpt.fut:280:43-79
            
            double zp_res_137564 = 1.0e-5 + zp_lhs_137563;
            
            // futhark/microgpt.fut:280:35-79
            
            double sqrt_res_137565 = futrts_sqrt64(zp_res_137564);
            
            ((double *) mem_152434)[i_151118] = sqrt_res_137565;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151126 = 0; i_151126 < (int64_t) 16; i_151126++) {
            // futhark/microgpt.fut:281:95-103
            
            double zs_rhs_137573 = ((double *) mem_152434)[i_151126];
            
            // futhark/microgpt.fut:281:87-103
            
            double zs_res_137574 = 1.0 / zs_rhs_137573;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151122 = 0; i_151122 < (int64_t) 16; i_151122++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_137581 = ((double *) mem_param_152288.mem)[i_151126 * (int64_t) 16 + i_151122];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_137582 = ((double *) mem_152381)[i_151126 * (int64_t) 16 + i_151122];
                
                // futhark/microgpt.fut:281:44-80
                
                double zp_res_137583 = zp_lhs_137581 + zp_rhs_137582;
                
                // futhark/microgpt.fut:281:60-103
                
                double zt_res_137584 = zs_res_137574 * zp_res_137583;
                
                ((double *) mem_152446)[i_151122] = zt_res_137584;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152441, i_151126 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152446, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151134 = 0; i_151134 < (int64_t) 16; i_151134++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151130 = 0; i_151130 < (int64_t) 16; i_151130++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_137599 = ((double *) mem_152441)[i_151134 * (int64_t) 16 + i_151130];
                
                ((double *) mem_152462)[i_151130] = lifted_lambda_res_137599;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152457, i_151134 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152462, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151140 = 0; i_151140 < (int64_t) 16; i_151140++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_141521;
            double r_141523 = 0.0;
            
            for (int64_t i_141522 = 0; i_141522 < (int64_t) 16; i_141522++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_141524 = ((double *) mem_152457)[i_151140 * (int64_t) 16 + i_141522];
                
                // futhark/microgpt.fut:283:65-102
                
                double zt_res_141525 = zt_lhs_141524 * zt_lhs_141524;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_141526 = r_141523 + zt_res_141525;
                double r_tmp_154832 = zp_res_141526;
                
                r_141523 = r_tmp_154832;
            }
            defunc_0_lifted_lambda_res_141521 = r_141523;
            // futhark/microgpt.fut:283:44-120
            
            double zs_res_141527 = defunc_0_lifted_lambda_res_141521 / 16.0;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_141534;
            double r_141536 = 0.0;
            
            for (int64_t i_141535 = 0; i_141535 < (int64_t) 16; i_141535++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_141537 = ((double *) mem_152457)[i_151140 * (int64_t) 16 + i_141535];
                
                // futhark/microgpt.fut:373:70-111
                
                double zt_res_141538 = zt_lhs_141537 * zt_lhs_141537;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_141539 = r_141536 + zt_res_141538;
                double r_tmp_154833 = zp_res_141539;
                
                r_141536 = r_tmp_154833;
            }
            defunc_0_lifted_lambda_res_141534 = r_141536;
            // futhark/microgpt.fut:373:48-129
            
            double zs_res_141540 = defunc_0_lifted_lambda_res_141534 / 16.0;
            
            ((double *) mem_152473)[i_151140] = zs_res_141540;
            ((double *) mem_152474)[i_151140] = zs_res_141527;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151145 = 0; i_151145 < (int64_t) 16; i_151145++) {
            // futhark/microgpt.fut:284:45-55
            
            double zp_lhs_137622 = ((double *) mem_152474)[i_151145];
            
            // futhark/microgpt.fut:284:45-83
            
            double zp_res_137623 = 1.0e-5 + zp_lhs_137622;
            
            // futhark/microgpt.fut:284:37-83
            
            double sqrt_res_137624 = futrts_sqrt64(zp_res_137623);
            
            ((double *) mem_152487)[i_151145] = sqrt_res_137624;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151153 = 0; i_151153 < (int64_t) 16; i_151153++) {
            // futhark/microgpt.fut:285:76-86
            
            double zs_rhs_137632 = ((double *) mem_152487)[i_151153];
            
            // futhark/microgpt.fut:285:68-86
            
            double zs_res_137633 = 1.0 / zs_rhs_137632;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151149 = 0; i_151149 < (int64_t) 16; i_151149++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_137640 = ((double *) mem_152457)[i_151153 * (int64_t) 16 + i_151149];
                
                // futhark/microgpt.fut:285:46-86
                
                double zt_res_137641 = zs_res_137633 * zt_lhs_137640;
                
                ((double *) mem_152499)[i_151149] = zt_res_137641;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152494, i_151153 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152499, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151161 = 0; i_151161 < (int64_t) 16; i_151161++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151157 = 0; i_151157 < (int64_t) 16; i_151157++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_137656 = ((double *) mem_152494)[i_151161 * (int64_t) 16 + i_151157];
                
                ((double *) mem_152515)[i_151157] = lifted_lambda_res_137656;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152510, i_151161 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152515, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151189 = 0; i_151189 < (int64_t) 4; i_151189++) {
            // futhark/microgpt.fut:287:83-86
            
            int64_t zp_lhs_141621 = mul64((int64_t) 4, i_151189);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151179 = 0; i_151179 < (int64_t) 16; i_151179++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151169 = 0; i_151169 < (int64_t) 4; i_151169++) {
                    // futhark/microgpt.fut:287:88-95
                    
                    int64_t zt_lhs_145642 = add64(zp_lhs_141621, i_151169);
                    
                    // futhark/microgpt.fut:287:70-97
                    
                    bool x_145643 = sle64((int64_t) 0, zt_lhs_145642);
                    
                    // futhark/microgpt.fut:287:70-97
                    
                    bool y_145644 = slt64(zt_lhs_145642, (int64_t) 16);
                    
                    // futhark/microgpt.fut:287:70-97
                    
                    bool bounds_check_145645 = x_145643 && y_145644;
                    
                    // futhark/microgpt.fut:287:70-97
                    
                    bool index_certs_145646;
                    
                    if (!bounds_check_145645) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_145642, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:287:70-97\n   #1  futhark/microgpt.fut:71:46-49\n   #2  futhark/microgpt.fut:287:49-127\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:15:29-44\n   #8  futhark/microgpt.fut:4:11-25\n   #9  futhark/microgpt.fut:15:15-45\n   #10 futhark/microgpt.fut:287:12-129\n   #11 futhark/microgpt.fut:582:5-76\n   #12 futhark/microgpt.fut:599:26-605:31\n   #13 futhark/microgpt.fut:633:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_145647;
                    double r_145649 = 0.0;
                    
                    for (int64_t i_145648 = 0; i_145648 < (int64_t) 16; i_145648++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_145650 = ((double *) mem_param_152292.mem)[zt_lhs_145642 * (int64_t) 16 + i_145648];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_145651 = ((double *) mem_152510)[i_151179 * (int64_t) 16 + i_145648];
                        
                        // futhark/microgpt.fut:287:70-125
                        
                        double zt_res_145652 = zt_lhs_145650 * zt_rhs_145651;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_145653 = r_145649 + zt_res_145652;
                        double r_tmp_154848 = zp_res_145653;
                        
                        r_145649 = r_tmp_154848;
                    }
                    defunc_0_lifted_lambda_res_145647 = r_145649;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_145661;
                    double r_145663 = 0.0;
                    
                    for (int64_t i_145662 = 0; i_145662 < (int64_t) 16; i_145662++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_145664 = ((double *) mem_param_152280.mem)[zt_lhs_145642 * (int64_t) 16 + i_145662];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_145665 = ((double *) mem_152510)[i_151179 * (int64_t) 16 + i_145662];
                        
                        // futhark/microgpt.fut:288:70-125
                        
                        double zt_res_145666 = zt_lhs_145664 * zt_rhs_145665;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_145667 = r_145663 + zt_res_145666;
                        double r_tmp_154849 = zp_res_145667;
                        
                        r_145663 = r_tmp_154849;
                    }
                    defunc_0_lifted_lambda_res_145661 = r_145663;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_145678;
                    double r_145680 = 0.0;
                    
                    for (int64_t i_145679 = 0; i_145679 < (int64_t) 16; i_145679++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_145681 = ((double *) mem_param_152304.mem)[zt_lhs_145642 * (int64_t) 16 + i_145679];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_145682 = ((double *) mem_152510)[i_151179 * (int64_t) 16 + i_145679];
                        
                        // futhark/microgpt.fut:289:70-125
                        
                        double zt_res_145683 = zt_lhs_145681 * zt_rhs_145682;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_145684 = r_145680 + zt_res_145683;
                        double r_tmp_154850 = zp_res_145684;
                        
                        r_145680 = r_tmp_154850;
                    }
                    defunc_0_lifted_lambda_res_145678 = r_145680;
                    ((double *) mem_152559)[i_151169] = defunc_0_lifted_lambda_res_145678;
                    ((double *) mem_152560)[i_151169] = defunc_0_lifted_lambda_res_145661;
                    ((double *) mem_152561)[i_151169] = defunc_0_lifted_lambda_res_145647;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_152544, i_151179 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152559, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_152545, i_151179 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152560, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_152546, i_151179 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152561, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_152526, i_151189 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_152544, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_152527, i_151189 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_152545, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_152528, i_151189 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_152546, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151227 = 0; i_151227 < (int64_t) 4; i_151227++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151214 = 0; i_151214 < (int64_t) 16; i_151214++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151201 = 0; i_151201 < (int64_t) 16; i_151201++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_146066;
                    double r_146068 = 0.0;
                    
                    for (int64_t i_146067 = 0; i_146067 < (int64_t) 4; i_146067++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_146069 = ((double *) mem_152528)[i_151227 * (int64_t) 64 + i_151214 * (int64_t) 4 + i_146067];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_146070 = ((double *) mem_152527)[i_151227 * (int64_t) 64 + i_151201 * (int64_t) 4 + i_146067];
                        
                        // futhark/microgpt.fut:290:111-164
                        
                        double zt_res_146071 = zt_lhs_146069 * zt_rhs_146070;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_146072 = r_146068 + zt_res_146071;
                        double r_tmp_154863 = zp_res_146072;
                        
                        r_146068 = r_tmp_154863;
                    }
                    defunc_0_lifted_lambda_res_146066 = r_146068;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_146079;
                    double r_146081 = 0.0;
                    
                    for (int64_t i_146080 = 0; i_146080 < (int64_t) 4; i_146080++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_146082 = ((double *) mem_152528)[i_151227 * (int64_t) 64 + i_151214 * (int64_t) 4 + i_146080];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_146083 = ((double *) mem_152527)[i_151227 * (int64_t) 64 + i_151201 * (int64_t) 4 + i_146080];
                        
                        // futhark/microgpt.fut:332:119-178
                        
                        double zt_res_146084 = zt_lhs_146082 * zt_rhs_146083;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_146085 = r_146081 + zt_res_146084;
                        double r_tmp_154864 = zp_res_146085;
                        
                        r_146081 = r_tmp_154864;
                    }
                    defunc_0_lifted_lambda_res_146079 = r_146081;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_146095;
                    double r_146097 = 0.0;
                    
                    for (int64_t i_146096 = 0; i_146096 < (int64_t) 4; i_146096++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_146098 = ((double *) mem_152528)[i_151227 * (int64_t) 64 + i_151214 * (int64_t) 4 + i_146096];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_146099 = ((double *) mem_152527)[i_151227 * (int64_t) 64 + i_151201 * (int64_t) 4 + i_146096];
                        
                        // futhark/microgpt.fut:341:119-178
                        
                        double zt_res_146100 = zt_lhs_146098 * zt_rhs_146099;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_146101 = r_146097 + zt_res_146100;
                        double r_tmp_154865 = zp_res_146101;
                        
                        r_146097 = r_tmp_154865;
                    }
                    defunc_0_lifted_lambda_res_146095 = r_146097;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_146113;
                    double r_146115 = 0.0;
                    
                    for (int64_t i_146114 = 0; i_146114 < (int64_t) 4; i_146114++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_146116 = ((double *) mem_152528)[i_151227 * (int64_t) 64 + i_151214 * (int64_t) 4 + i_146114];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_146117 = ((double *) mem_152527)[i_151227 * (int64_t) 64 + i_151201 * (int64_t) 4 + i_146114];
                        
                        // futhark/microgpt.fut:357:119-178
                        
                        double zt_res_146118 = zt_lhs_146116 * zt_rhs_146117;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_146119 = r_146115 + zt_res_146118;
                        double r_tmp_154866 = zp_res_146119;
                        
                        r_146115 = r_tmp_154866;
                    }
                    defunc_0_lifted_lambda_res_146113 = r_146115;
                    ((double *) mem_152651)[i_151201] = defunc_0_lifted_lambda_res_146113;
                    ((double *) mem_152652)[i_151201] = defunc_0_lifted_lambda_res_146095;
                    ((double *) mem_152653)[i_151201] = defunc_0_lifted_lambda_res_146079;
                    ((double *) mem_152654)[i_151201] = defunc_0_lifted_lambda_res_146066;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_152631, i_151214 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152651, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_152632, i_151214 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152652, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_152633, i_151214 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152653, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_152634, i_151214 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152654, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_152607, i_151227 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_152631, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_152608, i_151227 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_152632, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_152609, i_151227 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_152633, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_152610, i_151227 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_152634, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151266 = 0; i_151266 < (int64_t) 4; i_151266++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151253 = 0; i_151253 < (int64_t) 16; i_151253++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151240 = 0; i_151240 < (int64_t) 16; i_151240++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_146463 = ((double *) mem_152610)[i_151266 * (int64_t) 256 + i_151253 * (int64_t) 16 + i_151240];
                    
                    // futhark/microgpt.fut:291:55-93
                    
                    double zs_res_146464 = zs_lhs_146463 / 2.0;
                    double zp_rhs_146465 = ((double *) masks_mem_152270.mem)[step_137468 * (int64_t) 256 + i_151253 * (int64_t) 16 + i_151240];
                    
                    // futhark/microgpt.fut:291:80-117
                    
                    double zp_res_146466 = zs_res_146464 + zp_rhs_146465;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_146473 = ((double *) mem_152609)[i_151266 * (int64_t) 256 + i_151253 * (int64_t) 16 + i_151240];
                    
                    // futhark/microgpt.fut:333:59-101
                    
                    double zs_res_146474 = zs_lhs_146473 / 2.0;
                    
                    // futhark/microgpt.fut:333:88-127
                    
                    double zp_res_146476 = zp_rhs_146465 + zs_res_146474;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_146486 = ((double *) mem_152608)[i_151266 * (int64_t) 256 + i_151253 * (int64_t) 16 + i_151240];
                    
                    // futhark/microgpt.fut:342:59-101
                    
                    double zs_res_146487 = zs_lhs_146486 / 2.0;
                    
                    // futhark/microgpt.fut:342:88-127
                    
                    double zp_res_146489 = zp_rhs_146465 + zs_res_146487;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_146501 = ((double *) mem_152607)[i_151266 * (int64_t) 256 + i_151253 * (int64_t) 16 + i_151240];
                    
                    // futhark/microgpt.fut:358:59-101
                    
                    double zs_res_146502 = zs_lhs_146501 / 2.0;
                    
                    // futhark/microgpt.fut:358:88-127
                    
                    double zp_res_146504 = zp_rhs_146465 + zs_res_146502;
                    
                    ((double *) mem_152759)[i_151240] = zp_res_146504;
                    ((double *) mem_152760)[i_151240] = zp_res_146489;
                    ((double *) mem_152761)[i_151240] = zp_res_146476;
                    ((double *) mem_152762)[i_151240] = zp_res_146466;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_152739, i_151253 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152759, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_152740, i_151253 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152760, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_152741, i_151253 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152761, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_152742, i_151253 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152762, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_152715, i_151266 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_152739, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_152716, i_151266 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_152740, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_152717, i_151266 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_152741, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_152718, i_151266 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_152742, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151319 = 0; i_151319 < (int64_t) 4; i_151319++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151294 = 0; i_151294 < (int64_t) 16; i_151294++) {
                // futhark/microgpt.fut:115:13-33
                
                double defunc_0_reduce_res_150798;
                double defunc_0_reduce_res_150799;
                double defunc_0_reduce_res_150800;
                double defunc_0_reduce_res_150801;
                double defunc_0_reduce_res_150802;
                double defunc_0_reduce_res_150803;
                double redout_151271;
                double redout_151272;
                double redout_151273;
                double redout_151274;
                double redout_151275;
                double redout_151276;
                
                redout_151271 = -INFINITY;
                redout_151272 = -INFINITY;
                redout_151273 = -INFINITY;
                redout_151274 = -INFINITY;
                redout_151275 = -INFINITY;
                redout_151276 = -INFINITY;
                for (int64_t i_151277 = 0; i_151277 < (int64_t) 16; i_151277++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_147816 = ((double *) mem_152718)[i_151319 * (int64_t) 256 + i_151294 * (int64_t) 16 + i_151277];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_147826 = ((double *) mem_152717)[i_151319 * (int64_t) 256 + i_151294 * (int64_t) 16 + i_151277];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_147845 = ((double *) mem_152716)[i_151319 * (int64_t) 256 + i_151294 * (int64_t) 16 + i_151277];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_147889 = ((double *) mem_152715)[i_151319 * (int64_t) 256 + i_151294 * (int64_t) 16 + i_151277];
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_147116 = fmax64(lifted_lambda_res_147816, redout_151271);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_147135 = fmax64(lifted_lambda_res_147826, redout_151272);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_147157 = fmax64(lifted_lambda_res_147845, redout_151273);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_147182 = fmax64(lifted_lambda_res_147845, redout_151274);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_147232 = fmax64(lifted_lambda_res_147889, redout_151275);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_147263 = fmax64(lifted_lambda_res_147889, redout_151276);
                    double redout_tmp_154895 = max_res_147116;
                    double redout_tmp_154896 = max_res_147135;
                    double redout_tmp_154897 = max_res_147157;
                    double redout_tmp_154898 = max_res_147182;
                    double redout_tmp_154899 = max_res_147232;
                    double redout_tmp_154900 = max_res_147263;
                    
                    redout_151271 = redout_tmp_154895;
                    redout_151272 = redout_tmp_154896;
                    redout_151273 = redout_tmp_154897;
                    redout_151274 = redout_tmp_154898;
                    redout_151275 = redout_tmp_154899;
                    redout_151276 = redout_tmp_154900;
                }
                defunc_0_reduce_res_150798 = redout_151271;
                defunc_0_reduce_res_150799 = redout_151272;
                defunc_0_reduce_res_150800 = redout_151273;
                defunc_0_reduce_res_150801 = redout_151274;
                defunc_0_reduce_res_150802 = redout_151275;
                defunc_0_reduce_res_150803 = redout_151276;
                // futhark/microgpt.fut:353:172-198
                
                double neg_res_147190 = -defunc_0_reduce_res_150801;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_147191;
                double r_147193 = 0.0;
                
                for (int64_t i_147192 = 0; i_147192 < (int64_t) 16; i_147192++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_147194 = ((double *) mem_152716)[i_151319 * (int64_t) 256 + i_151294 * (int64_t) 16 + i_147192];
                    
                    // futhark/microgpt.fut:353:138-198
                    
                    double zp_res_147195 = neg_res_147190 + zp_lhs_147194;
                    
                    // futhark/microgpt.fut:353:131-198
                    
                    double neg_res_147196 = -zp_res_147195;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_147197 = fmax64(0.0, neg_res_147196);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_147198 = fsignum64(max_res_147197);
                    
                    // futhark/microgpt.fut:353:112-201
                    
                    double neg_res_147199 = -sgn_res_147198;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_147200 = r_147193 + neg_res_147199;
                    double r_tmp_154901 = zp_res_147200;
                    
                    r_147193 = r_tmp_154901;
                }
                defunc_0_lifted_lambda_res_147191 = r_147193;
                // futhark/microgpt.fut:353:58-204
                
                double zp_res_147201 = defunc_0_lifted_lambda_res_139130 + defunc_0_lifted_lambda_res_147191;
                
                // futhark/microgpt.fut:353:48-204
                
                double zs_res_147202 = 1.0 / zp_res_147201;
                
                // futhark/microgpt.fut:369:172-198
                
                double neg_res_147271 = -defunc_0_reduce_res_150803;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_147272;
                double r_147274 = 0.0;
                
                for (int64_t i_147273 = 0; i_147273 < (int64_t) 16; i_147273++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_147275 = ((double *) mem_152715)[i_151319 * (int64_t) 256 + i_151294 * (int64_t) 16 + i_147273];
                    
                    // futhark/microgpt.fut:369:138-198
                    
                    double zp_res_147276 = neg_res_147271 + zp_lhs_147275;
                    
                    // futhark/microgpt.fut:369:131-198
                    
                    double neg_res_147277 = -zp_res_147276;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_147278 = fmax64(0.0, neg_res_147277);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_147279 = fsignum64(max_res_147278);
                    
                    // futhark/microgpt.fut:369:112-201
                    
                    double neg_res_147280 = -sgn_res_147279;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_147281 = r_147274 + neg_res_147280;
                    double r_tmp_154902 = zp_res_147281;
                    
                    r_147274 = r_tmp_154902;
                }
                defunc_0_lifted_lambda_res_147272 = r_147274;
                // futhark/microgpt.fut:369:58-204
                
                double zp_res_147282 = defunc_0_lifted_lambda_res_139559 + defunc_0_lifted_lambda_res_147272;
                
                // futhark/microgpt.fut:369:48-204
                
                double zs_res_147283 = 1.0 / zp_res_147282;
                
                ((double *) mem_152863)[i_151294] = zs_res_147283;
                ((double *) mem_152864)[i_151294] = defunc_0_reduce_res_150803;
                ((double *) mem_152865)[i_151294] = defunc_0_reduce_res_150802;
                ((double *) mem_152866)[i_151294] = zs_res_147202;
                ((double *) mem_152867)[i_151294] = defunc_0_reduce_res_150801;
                ((double *) mem_152868)[i_151294] = defunc_0_reduce_res_150800;
                ((double *) mem_152869)[i_151294] = defunc_0_reduce_res_150799;
                ((double *) mem_152870)[i_151294] = defunc_0_reduce_res_150798;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152823, i_151319 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152863, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152824, i_151319 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152864, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152825, i_151319 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152865, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152826, i_151319 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152866, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152827, i_151319 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152867, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152828, i_151319 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152868, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152829, i_151319 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152869, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_152830, i_151319 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152870, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151362 = 0; i_151362 < (int64_t) 4; i_151362++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151349 = 0; i_151349 < (int64_t) 16; i_151349++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_148105 = ((double *) mem_152830)[i_151362 * (int64_t) 16 + i_151349];
                
                // futhark/microgpt.fut:293:91-114
                
                double neg_res_148106 = -neg_arg0_148105;
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_148167 = ((double *) mem_152825)[i_151362 * (int64_t) 16 + i_151349];
                
                // futhark/microgpt.fut:362:99-125
                
                double neg_res_148168 = -neg_arg0_148167;
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_148144 = ((double *) mem_152828)[i_151362 * (int64_t) 16 + i_151349];
                
                // futhark/microgpt.fut:346:99-125
                
                double neg_res_148145 = -neg_arg0_148144;
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_148123 = ((double *) mem_152829)[i_151362 * (int64_t) 16 + i_151349];
                
                // futhark/microgpt.fut:335:99-125
                
                double neg_res_148124 = -neg_arg0_148123;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151336 = 0; i_151336 < (int64_t) 16; i_151336++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_148287 = ((double *) mem_152718)[i_151362 * (int64_t) 256 + i_151349 * (int64_t) 16 + i_151336];
                    
                    // futhark/microgpt.fut:293:61-114
                    
                    double zp_res_148288 = neg_res_148106 + zp_lhs_148287;
                    
                    // futhark/microgpt.fut:293:54-114
                    
                    double exp_res_148289 = futrts_exp64(zp_res_148288);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_148296 = ((double *) mem_152717)[i_151362 * (int64_t) 256 + i_151349 * (int64_t) 16 + i_151336];
                    
                    // futhark/microgpt.fut:335:65-125
                    
                    double zp_res_148297 = neg_res_148124 + zp_lhs_148296;
                    
                    // futhark/microgpt.fut:335:58-125
                    
                    double exp_res_148298 = futrts_exp64(zp_res_148297);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_148308 = ((double *) mem_152716)[i_151362 * (int64_t) 256 + i_151349 * (int64_t) 16 + i_151336];
                    
                    // futhark/microgpt.fut:346:65-125
                    
                    double zp_res_148309 = neg_res_148145 + zp_lhs_148308;
                    
                    // futhark/microgpt.fut:346:58-125
                    
                    double exp_res_148310 = futrts_exp64(zp_res_148309);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_148322 = ((double *) mem_152715)[i_151362 * (int64_t) 256 + i_151349 * (int64_t) 16 + i_151336];
                    
                    // futhark/microgpt.fut:362:65-125
                    
                    double zp_res_148323 = neg_res_148168 + zp_lhs_148322;
                    
                    // futhark/microgpt.fut:362:58-125
                    
                    double exp_res_148324 = futrts_exp64(zp_res_148323);
                    
                    ((double *) mem_152995)[i_151336] = exp_res_148324;
                    ((double *) mem_152996)[i_151336] = exp_res_148310;
                    ((double *) mem_152997)[i_151336] = exp_res_148298;
                    ((double *) mem_152998)[i_151336] = exp_res_148289;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_152975, i_151349 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152995, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_152976, i_151349 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152996, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_152977, i_151349 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152997, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_152978, i_151349 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_152998, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_152951, i_151362 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_152975, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_152952, i_151362 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_152976, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_152953, i_151362 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_152977, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_152954, i_151362 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_152978, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151378 = 0; i_151378 < (int64_t) 4; i_151378++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151371 = 0; i_151371 < (int64_t) 16; i_151371++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_148356;
                double r_148358 = 0.0;
                
                for (int64_t i_148357 = 0; i_148357 < (int64_t) 16; i_148357++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_148359 = ((double *) mem_152954)[i_151378 * (int64_t) 256 + i_151371 * (int64_t) 16 + i_148357];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_148360 = r_148358 + lifted_lambda_res_148359;
                    double r_tmp_154919 = zp_res_148360;
                    
                    r_148358 = r_tmp_154919;
                }
                defunc_0_lifted_lambda_res_148356 = r_148358;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_148367;
                double r_148369 = 0.0;
                
                for (int64_t i_148368 = 0; i_148368 < (int64_t) 16; i_148368++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_148370 = ((double *) mem_152953)[i_151378 * (int64_t) 256 + i_151371 * (int64_t) 16 + i_148368];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_148371 = r_148369 + lifted_lambda_res_148370;
                    double r_tmp_154920 = zp_res_148371;
                    
                    r_148369 = r_tmp_154920;
                }
                defunc_0_lifted_lambda_res_148367 = r_148369;
                ((double *) mem_153069)[i_151371] = defunc_0_lifted_lambda_res_148367;
                ((double *) mem_153070)[i_151371] = defunc_0_lifted_lambda_res_148356;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153059, i_151378 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153069, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153060, i_151378 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153070, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151399 = 0; i_151399 < (int64_t) 4; i_151399++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151392 = 0; i_151392 < (int64_t) 16; i_151392++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_148391 = ((double *) mem_153060)[i_151399 * (int64_t) 16 + i_151392];
                
                // futhark/microgpt.fut:295:84-109
                
                double zs_res_148392 = 1.0 / zs_rhs_148391;
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_148408 = ((double *) mem_153059)[i_151399 * (int64_t) 16 + i_151392];
                
                // futhark/microgpt.fut:337:92-120
                
                double zs_res_148409 = 1.0 / zs_rhs_148408;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151385 = 0; i_151385 < (int64_t) 16; i_151385++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_148436 = ((double *) mem_152954)[i_151399 * (int64_t) 256 + i_151392 * (int64_t) 16 + i_151385];
                    
                    // futhark/microgpt.fut:295:54-109
                    
                    double zt_res_148437 = zs_res_148392 * zt_lhs_148436;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_148444 = ((double *) mem_152953)[i_151399 * (int64_t) 256 + i_151392 * (int64_t) 16 + i_151385];
                    
                    // futhark/microgpt.fut:337:58-120
                    
                    double zt_res_148445 = zs_res_148409 * zt_lhs_148444;
                    
                    ((double *) mem_153113)[i_151385] = zt_res_148445;
                    ((double *) mem_153114)[i_151385] = zt_res_148437;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153103, i_151392 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153113, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153104, i_151392 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153114, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153091, i_151399 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153103, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153092, i_151399 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153104, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151420 = 0; i_151420 < (int64_t) 4; i_151420++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151413 = 0; i_151413 < (int64_t) 16; i_151413++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151406 = 0; i_151406 < (int64_t) 16; i_151406++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_148508 = ((double *) mem_153092)[i_151420 * (int64_t) 256 + i_151413 * (int64_t) 16 + i_151406];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_148515 = ((double *) mem_153091)[i_151420 * (int64_t) 256 + i_151413 * (int64_t) 16 + i_151406];
                    
                    ((double *) mem_153167)[i_151406] = lifted_lambda_res_148515;
                    ((double *) mem_153168)[i_151406] = lifted_lambda_res_148508;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153157, i_151413 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153167, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153158, i_151413 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153168, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153145, i_151420 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153157, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153146, i_151420 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153158, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151433 = 0; i_151433 < (int64_t) 4; i_151433++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151429 = 0; i_151429 < (int64_t) 16; i_151429++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151425 = 0; i_151425 < (int64_t) 4; i_151425++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_137928;
                    double r_137930 = 0.0;
                    
                    for (int64_t i_137929 = 0; i_137929 < (int64_t) 16; i_137929++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_137931 = ((double *) mem_153146)[i_151433 * (int64_t) 256 + i_151429 * (int64_t) 16 + i_137929];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_137932 = ((double *) mem_152526)[i_151433 * (int64_t) 64 + i_137929 * (int64_t) 4 + i_151425];
                        
                        // futhark/microgpt.fut:297:74-127
                        
                        double zt_res_137933 = zt_lhs_137931 * zt_rhs_137932;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_137934 = r_137930 + zt_res_137933;
                        double r_tmp_154936 = zp_res_137934;
                        
                        r_137930 = r_tmp_154936;
                    }
                    defunc_0_lifted_lambda_res_137928 = r_137930;
                    ((double *) mem_153210)[i_151425] = defunc_0_lifted_lambda_res_137928;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153205, i_151429 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153210, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153199, i_151433 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_153205, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151441 = 0; i_151441 < (int64_t) 16; i_151441++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151437 = 0; i_151437 < (int64_t) 16; i_151437++) {
                // futhark/microgpt.fut:298:15-18
                
                int64_t tmp_137946 = sdiv64(i_151437, (int64_t) 4);
                
                // futhark/microgpt.fut:298:4-20
                
                bool x_137947 = sle64((int64_t) 0, tmp_137946);
                
                // futhark/microgpt.fut:298:4-20
                
                bool y_137948 = slt64(tmp_137946, (int64_t) 4);
                
                // futhark/microgpt.fut:298:4-20
                
                bool bounds_check_137949 = x_137947 && y_137948;
                
                // futhark/microgpt.fut:298:4-20
                
                bool index_certs_137950;
                
                if (!bounds_check_137949) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_137946, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:298:4-20\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:290:12-298:49\n   #6  futhark/microgpt.fut:582:5-76\n   #7  futhark/microgpt.fut:599:26-605:31\n   #8  futhark/microgpt.fut:633:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:298:35-38
                
                int64_t tmp_137951 = smod64(i_151437, (int64_t) 4);
                
                // futhark/microgpt.fut:298:4-40
                
                bool x_137952 = sle64((int64_t) 0, tmp_137951);
                
                // futhark/microgpt.fut:298:4-40
                
                bool y_137953 = slt64(tmp_137951, (int64_t) 4);
                
                // futhark/microgpt.fut:298:4-40
                
                bool bounds_check_137954 = x_137952 && y_137953;
                
                // futhark/microgpt.fut:298:4-40
                
                bool index_certs_137955;
                
                if (!bounds_check_137954) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_137951, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:298:4-40\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:290:12-298:49\n   #6  futhark/microgpt.fut:582:5-76\n   #7  futhark/microgpt.fut:599:26-605:31\n   #8  futhark/microgpt.fut:633:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_137956 = ((double *) mem_153199)[tmp_137946 * (int64_t) 64 + i_151441 * (int64_t) 4 + tmp_137951];
                
                ((double *) mem_153231)[i_151437] = lifted_lambda_res_137956;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153226, i_151441 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153231, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151449 = 0; i_151449 < (int64_t) 16; i_151449++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151445 = 0; i_151445 < (int64_t) 16; i_151445++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_137971;
                double r_137973 = 0.0;
                
                for (int64_t i_137972 = 0; i_137972 < (int64_t) 16; i_137972++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_137974 = ((double *) mem_param_152284.mem)[i_151445 * (int64_t) 16 + i_137972];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_137975 = ((double *) mem_153226)[i_151449 * (int64_t) 16 + i_137972];
                    
                    // futhark/microgpt.fut:299:64-104
                    
                    double zt_res_137976 = zt_lhs_137974 * zt_rhs_137975;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_137977 = r_137973 + zt_res_137976;
                    double r_tmp_154941 = zp_res_137977;
                    
                    r_137973 = r_tmp_154941;
                }
                defunc_0_lifted_lambda_res_137971 = r_137973;
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_137978 = ((double *) mem_152457)[i_151449 * (int64_t) 16 + i_151445];
                
                // futhark/microgpt.fut:299:43-128
                
                double zp_res_137979 = defunc_0_lifted_lambda_res_137971 + zp_rhs_137978;
                
                ((double *) mem_153247)[i_151445] = zp_res_137979;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153242, i_151449 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153247, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151455 = 0; i_151455 < (int64_t) 16; i_151455++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_143598;
            double r_143600 = 0.0;
            
            for (int64_t i_143599 = 0; i_143599 < (int64_t) 16; i_143599++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_143601 = ((double *) mem_153242)[i_151455 * (int64_t) 16 + i_143599];
                
                // futhark/microgpt.fut:300:66-105
                
                double zt_res_143602 = zt_lhs_143601 * zt_lhs_143601;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_143603 = r_143600 + zt_res_143602;
                double r_tmp_154944 = zp_res_143603;
                
                r_143600 = r_tmp_154944;
            }
            defunc_0_lifted_lambda_res_143598 = r_143600;
            // futhark/microgpt.fut:300:45-123
            
            double zs_res_143604 = defunc_0_lifted_lambda_res_143598 / 16.0;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_143611;
            double r_143613 = 0.0;
            
            for (int64_t i_143612 = 0; i_143612 < (int64_t) 16; i_143612++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_143614 = ((double *) mem_153242)[i_151455 * (int64_t) 16 + i_143612];
                
                // futhark/microgpt.fut:325:70-113
                
                double zt_res_143615 = zt_lhs_143614 * zt_lhs_143614;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_143616 = r_143613 + zt_res_143615;
                double r_tmp_154945 = zp_res_143616;
                
                r_143613 = r_tmp_154945;
            }
            defunc_0_lifted_lambda_res_143611 = r_143613;
            // futhark/microgpt.fut:325:48-131
            
            double zs_res_143617 = defunc_0_lifted_lambda_res_143611 / 16.0;
            
            ((double *) mem_153258)[i_151455] = zs_res_143617;
            ((double *) mem_153259)[i_151455] = zs_res_143604;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151460 = 0; i_151460 < (int64_t) 16; i_151460++) {
            // futhark/microgpt.fut:301:45-55
            
            double zp_lhs_138002 = ((double *) mem_153259)[i_151460];
            
            // futhark/microgpt.fut:301:45-83
            
            double zp_res_138003 = 1.0e-5 + zp_lhs_138002;
            
            // futhark/microgpt.fut:301:37-83
            
            double sqrt_res_138004 = futrts_sqrt64(zp_res_138003);
            
            ((double *) mem_153272)[i_151460] = sqrt_res_138004;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151468 = 0; i_151468 < (int64_t) 16; i_151468++) {
            // futhark/microgpt.fut:302:77-87
            
            double zs_rhs_138012 = ((double *) mem_153272)[i_151468];
            
            // futhark/microgpt.fut:302:69-87
            
            double zs_res_138013 = 1.0 / zs_rhs_138012;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151464 = 0; i_151464 < (int64_t) 16; i_151464++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_138020 = ((double *) mem_153242)[i_151468 * (int64_t) 16 + i_151464];
                
                // futhark/microgpt.fut:302:46-87
                
                double zt_res_138021 = zs_res_138013 * zt_lhs_138020;
                
                ((double *) mem_153284)[i_151464] = zt_res_138021;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153279, i_151468 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153284, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151476 = 0; i_151476 < (int64_t) 16; i_151476++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151472 = 0; i_151472 < (int64_t) 16; i_151472++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_138036 = ((double *) mem_153279)[i_151476 * (int64_t) 16 + i_151472];
                
                ((double *) mem_153300)[i_151472] = lifted_lambda_res_138036;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153295, i_151476 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153300, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151484 = 0; i_151484 < (int64_t) 16; i_151484++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151480 = 0; i_151480 < (int64_t) 64; i_151480++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_138052;
                double r_138054 = 0.0;
                
                for (int64_t i_138053 = 0; i_138053 < (int64_t) 16; i_138053++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_138055 = ((double *) mem_param_152300.mem)[i_151480 * (int64_t) 16 + i_138053];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_138056 = ((double *) mem_153295)[i_151484 * (int64_t) 16 + i_138053];
                    
                    // futhark/microgpt.fut:304:63-102
                    
                    double zt_res_138057 = zt_lhs_138055 * zt_rhs_138056;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_138058 = r_138054 + zt_res_138057;
                    double r_tmp_154953 = zp_res_138058;
                    
                    r_138054 = r_tmp_154953;
                }
                defunc_0_lifted_lambda_res_138052 = r_138054;
                ((double *) mem_153316)[i_151480] = defunc_0_lifted_lambda_res_138052;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153311, i_151484 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153316, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151492 = 0; i_151492 < (int64_t) 16; i_151492++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151488 = 0; i_151488 < (int64_t) 64; i_151488++) {
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_138073 = ((double *) mem_153311)[i_151492 * (int64_t) 64 + i_151488];
                
                // futhark/microgpt.fut:305:41-69
                
                double max_res_138074 = fmax64(0.0, max_arg0_138073);
                
                ((double *) mem_153332)[i_151488] = max_res_138074;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153327, i_151492 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153332, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151500 = 0; i_151500 < (int64_t) 16; i_151500++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151496 = 0; i_151496 < (int64_t) 16; i_151496++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_138089;
                double r_138091 = 0.0;
                
                for (int64_t i_138090 = 0; i_138090 < (int64_t) 64; i_138090++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_138092 = ((double *) mem_param_152276.mem)[i_151496 * (int64_t) 64 + i_138090];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_138093 = ((double *) mem_153327)[i_151500 * (int64_t) 64 + i_138090];
                    
                    // futhark/microgpt.fut:306:64-105
                    
                    double zt_res_138094 = zt_lhs_138092 * zt_rhs_138093;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_138095 = r_138091 + zt_res_138094;
                    double r_tmp_154958 = zp_res_138095;
                    
                    r_138091 = r_tmp_154958;
                }
                defunc_0_lifted_lambda_res_138089 = r_138091;
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_138096 = ((double *) mem_153242)[i_151500 * (int64_t) 16 + i_151496];
                
                // futhark/microgpt.fut:306:43-130
                
                double zp_res_138097 = defunc_0_lifted_lambda_res_138089 + zp_rhs_138096;
                
                ((double *) mem_153348)[i_151496] = zp_res_138097;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153343, i_151500 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153348, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151508 = 0; i_151508 < (int64_t) 16; i_151508++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151504 = 0; i_151504 < (int64_t) 27; i_151504++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_138112;
                double r_138114 = 0.0;
                
                for (int64_t i_138113 = 0; i_138113 < (int64_t) 16; i_138113++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_138115 = ((double *) mem_param_152308.mem)[i_151504 * (int64_t) 16 + i_138113];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_138116 = ((double *) mem_153343)[i_151508 * (int64_t) 16 + i_138113];
                    
                    // futhark/microgpt.fut:307:63-103
                    
                    double zt_res_138117 = zt_lhs_138115 * zt_rhs_138116;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_138118 = r_138114 + zt_res_138117;
                    double r_tmp_154961 = zp_res_138118;
                    
                    r_138114 = r_tmp_154961;
                }
                defunc_0_lifted_lambda_res_138112 = r_138114;
                ((double *) mem_153364)[i_151504] = defunc_0_lifted_lambda_res_138112;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153359, i_151508 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153364, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151538 = 0; i_151538 < (int64_t) 16; i_151538++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_150901;
            double defunc_0_reduce_res_150902;
            double redout_151525;
            double redout_151526;
            
            redout_151525 = -INFINITY;
            redout_151526 = -INFINITY;
            for (int64_t i_151528 = 0; i_151528 < (int64_t) 27; i_151528++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_148686 = ((double *) mem_153359)[i_151538 * (int64_t) 27 + i_151528];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151522 = 0; i_151522 < (int64_t) 27; i_151522++) {
                    // futhark/microgpt.fut:312:55-316:90
                    
                    bool cond_148695 = i_151522 == i_151528;
                    
                    // futhark/microgpt.fut:312:55-316:90
                    
                    double lifted_lambda_res_148696;
                    
                    if (cond_148695) {
                        // futhark/microgpt.fut:115:13-33
                        
                        double defunc_0_reduce_res_150848;
                        double redout_151510 = -INFINITY;
                        
                        for (int64_t i_151511 = 0; i_151511 < (int64_t) 27; i_151511++) {
                            // futhark/microgpt.fut:4:11-25
                            
                            double lifted_lambda_res_150854 = ((double *) mem_153359)[i_151538 * (int64_t) 27 + i_151511];
                            
                            // futhark/microgpt.fut:115:13-33
                            
                            double max_res_150857 = fmax64(lifted_lambda_res_150854, redout_151510);
                            double redout_tmp_154970 = max_res_150857;
                            
                            redout_151510 = redout_tmp_154970;
                        }
                        defunc_0_reduce_res_150848 = redout_151510;
                        // futhark/microgpt.fut:313:67-76
                        
                        double neg_res_150859 = -defunc_0_reduce_res_150848;
                        
                        // futhark/microgpt.fut:4:11-25
                        if (mem_153405_cached_sizze_155438 < (int64_t) 216) {
                            err = lexical_realloc(ctx, &mem_153405, &mem_153405_cached_sizze_155438, (int64_t) 216);
                            if (err != FUTHARK_SUCCESS)
                                goto cleanup;
                        }
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_151514 = 0; i_151514 < (int64_t) 27; i_151514++) {
                            // futhark/microgpt.fut:4:11-25
                            
                            double zp_lhs_150866 = ((double *) mem_153359)[i_151538 * (int64_t) 27 + i_151514];
                            
                            // futhark/microgpt.fut:313:44-76
                            
                            double zp_res_150867 = neg_res_150859 + zp_lhs_150866;
                            
                            // futhark/microgpt.fut:313:37-76
                            
                            double exp_res_150868 = futrts_exp64(zp_res_150867);
                            
                            ((double *) mem_153405)[i_151514] = exp_res_150868;
                        }
                        // futhark/microgpt.fut:71:13-49
                        
                        double defunc_0_lifted_lambda_res_150871;
                        double r_150873 = 0.0;
                        
                        for (int64_t i_150872 = 0; i_150872 < (int64_t) 27; i_150872++) {
                            // futhark/microgpt.fut:314:36-46
                            
                            double lifted_lambda_res_150874 = ((double *) mem_153405)[i_150872];
                            
                            // futhark/microgpt.fut:71:40-49
                            
                            double zp_res_150875 = r_150873 + lifted_lambda_res_150874;
                            double r_tmp_154972 = zp_res_150875;
                            
                            r_150873 = r_tmp_154972;
                        }
                        defunc_0_lifted_lambda_res_150871 = r_150873;
                        // futhark/microgpt.fut:315:53-64
                        
                        double zs_res_150876 = 1.0 / defunc_0_lifted_lambda_res_150871;
                        
                        // futhark/microgpt.fut:4:11-25
                        if (mem_153412_cached_sizze_155439 < (int64_t) 216) {
                            err = lexical_realloc(ctx, &mem_153412, &mem_153412_cached_sizze_155439, (int64_t) 216);
                            if (err != FUTHARK_SUCCESS)
                                goto cleanup;
                        }
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_151518 = 0; i_151518 < (int64_t) 27; i_151518++) {
                            // futhark/microgpt.fut:315:37-47
                            
                            double zt_lhs_150883 = ((double *) mem_153405)[i_151518];
                            
                            // futhark/microgpt.fut:315:37-64
                            
                            double zt_res_150884 = zs_res_150876 * zt_lhs_150883;
                            
                            ((double *) mem_153412)[i_151518] = zt_res_150884;
                        }
                        // futhark/microgpt.fut:4:11-25
                        
                        double zt_rhs_150891 = ((double *) mem_152382)[i_151538 * (int64_t) 27 + i_151528];
                        
                        // futhark/microgpt.fut:316:7-49
                        
                        double zt_res_150892 = -6.25e-2 * zt_rhs_150891;
                        
                        // futhark/microgpt.fut:316:64-74
                        
                        double zs_rhs_150897 = ((double *) mem_153412)[i_151522];
                        
                        // futhark/microgpt.fut:316:56-74
                        
                        double zs_res_150898 = 1.0 / zs_rhs_150897;
                        
                        // futhark/microgpt.fut:316:25-74
                        
                        double zt_res_150899 = zt_res_150892 * zs_res_150898;
                        
                        lifted_lambda_res_148696 = zt_res_150899;
                    } else {
                        lifted_lambda_res_148696 = 0.0;
                    }
                    ((double *) mem_153401)[i_151522] = lifted_lambda_res_148696;
                }
                // futhark/microgpt.fut:115:13-33
                
                double max_res_143754 = fmax64(lifted_lambda_res_148686, redout_151525);
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_143845 = fmax64(lifted_lambda_res_148686, redout_151526);
                
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153396, i_151528 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153401, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
                
                double redout_tmp_154966 = max_res_143754;
                double redout_tmp_154967 = max_res_143845;
                
                redout_151525 = redout_tmp_154966;
                redout_151526 = redout_tmp_154967;
            }
            defunc_0_reduce_res_150901 = redout_151525;
            defunc_0_reduce_res_150902 = redout_151526;
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_154974 = 0; nest_i_154974 < (int64_t) 27; nest_i_154974++) {
                ((double *) mem_153378)[i_151538 * (int64_t) 27 + nest_i_154974] = defunc_0_reduce_res_150901;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_154975 = 0; nest_i_154975 < (int64_t) 27; nest_i_154975++) {
                ((double *) mem_153376)[i_151538 * (int64_t) 27 + nest_i_154975] = defunc_0_reduce_res_150902;
            }
            // futhark/microgpt.fut:321:163-188
            
            double neg_res_143856 = -defunc_0_reduce_res_150902;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_143857;
            double r_143859 = 0.0;
            
            for (int64_t i_143858 = 0; i_143858 < (int64_t) 27; i_143858++) {
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_143860 = ((double *) mem_153359)[i_151538 * (int64_t) 27 + i_143858];
                
                // futhark/microgpt.fut:321:138-188
                
                double zp_res_143861 = neg_res_143856 + zp_lhs_143860;
                
                // futhark/microgpt.fut:321:131-188
                
                double neg_res_143862 = -zp_res_143861;
                
                // futhark/microgpt.fut:110:42-54
                
                double max_res_143863 = fmax64(0.0, neg_res_143862);
                
                // futhark/microgpt.fut:110:35-54
                
                double sgn_res_143864 = fsignum64(max_res_143863);
                
                // futhark/microgpt.fut:321:112-191
                
                double neg_res_143865 = -sgn_res_143864;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_143866 = r_143859 + neg_res_143865;
                double r_tmp_154976 = zp_res_143866;
                
                r_143859 = r_tmp_154976;
            }
            defunc_0_lifted_lambda_res_143857 = r_143859;
            // futhark/microgpt.fut:321:58-194
            
            double zp_res_143867 = defunc_0_lifted_lambda_res_138357 + defunc_0_lifted_lambda_res_143857;
            
            // futhark/microgpt.fut:321:48-194
            
            double zs_res_143868 = 1.0 / zp_res_143867;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_154977 = 0; nest_i_154977 < (int64_t) 27; nest_i_154977++) {
                ((double *) mem_153375)[i_151538 * (int64_t) 27 + nest_i_154977] = zs_res_143868;
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153377, i_151538 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_153396, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151553 = 0; i_151553 < (int64_t) 16; i_151553++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151549 = 0; i_151549 < (int64_t) 27; i_151549++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_138154 = ((double *) mem_153378)[i_151553 * (int64_t) 27 + i_151549];
                
                // futhark/microgpt.fut:310:85-108
                
                double neg_res_138155 = -neg_arg0_138154;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151545 = 0; i_151545 < (int64_t) 27; i_151545++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_138162 = ((double *) mem_153359)[i_151553 * (int64_t) 27 + i_151545];
                    
                    // futhark/microgpt.fut:310:62-108
                    
                    double zp_res_138163 = neg_res_138155 + zp_lhs_138162;
                    
                    // futhark/microgpt.fut:310:55-108
                    
                    double exp_res_138164 = futrts_exp64(zp_res_138163);
                    
                    ((double *) mem_153457)[i_151545] = exp_res_138164;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153452, i_151549 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153457, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153446, i_151553 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_153452, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151566 = 0; i_151566 < (int64_t) 16; i_151566++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151559 = 0; i_151559 < (int64_t) 27; i_151559++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_149060;
                double r_149062 = 0.0;
                
                for (int64_t i_149061 = 0; i_149061 < (int64_t) 27; i_149061++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_149063 = ((double *) mem_153446)[i_151566 * (int64_t) 729 + i_151559 * (int64_t) 27 + i_149061];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_149064 = r_149062 + lifted_lambda_res_149063;
                    double r_tmp_154985 = zp_res_149064;
                    
                    r_149062 = r_tmp_154985;
                }
                defunc_0_lifted_lambda_res_149060 = r_149062;
                // futhark/microgpt.fut:317:147-186
                
                double zt_res_149072 = defunc_0_lifted_lambda_res_149060 * defunc_0_lifted_lambda_res_149060;
                
                // futhark/microgpt.fut:317:138-186
                
                double zs_res_149073 = 1.0 / zt_res_149072;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_149074;
                double r_149076 = 0.0;
                
                for (int64_t i_149075 = 0; i_149075 < (int64_t) 27; i_149075++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_149077 = ((double *) mem_153377)[i_151566 * (int64_t) 729 + i_151559 * (int64_t) 27 + i_149075];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_149078 = ((double *) mem_153446)[i_151566 * (int64_t) 729 + i_151559 * (int64_t) 27 + i_149075];
                    
                    // futhark/microgpt.fut:317:76-131
                    
                    double zt_res_149079 = zt_lhs_149077 * zt_rhs_149078;
                    
                    // futhark/microgpt.fut:317:102-186
                    
                    double zt_res_149080 = zs_res_149073 * zt_res_149079;
                    
                    // futhark/microgpt.fut:317:68-186
                    
                    double neg_res_149081 = -zt_res_149080;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_149082 = r_149076 + neg_res_149081;
                    double r_tmp_154986 = zp_res_149082;
                    
                    r_149076 = r_tmp_154986;
                }
                defunc_0_lifted_lambda_res_149074 = r_149076;
                ((double *) mem_153483)[i_151559] = defunc_0_lifted_lambda_res_149074;
                ((double *) mem_153484)[i_151559] = defunc_0_lifted_lambda_res_149060;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153473, i_151566 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153483, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153474, i_151566 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153484, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151579 = 0; i_151579 < (int64_t) 16; i_151579++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151575 = 0; i_151575 < (int64_t) 27; i_151575++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_138294 = ((double *) mem_153474)[i_151579 * (int64_t) 27 + i_151575];
                
                // futhark/microgpt.fut:318:92-119
                
                double zs_res_138295 = 1.0 / zs_rhs_138294;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_138296 = ((double *) mem_153473)[i_151579 * (int64_t) 27 + i_151575];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151571 = 0; i_151571 < (int64_t) 27; i_151571++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_138303 = ((double *) mem_153377)[i_151579 * (int64_t) 729 + i_151575 * (int64_t) 27 + i_151571];
                    
                    // futhark/microgpt.fut:318:59-119
                    
                    double zt_res_138304 = zs_res_138295 * zt_lhs_138303;
                    
                    // futhark/microgpt.fut:318:87-145
                    
                    double zp_res_138305 = zp_rhs_138296 + zt_res_138304;
                    
                    ((double *) mem_153516)[i_151571] = zp_res_138305;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153511, i_151575 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153516, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153505, i_151579 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_153511, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151587 = 0; i_151587 < (int64_t) 16; i_151587++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151583 = 0; i_151583 < (int64_t) 27; i_151583++) {
                double f_elem_138318 = ((double *) mem_153378)[i_151587 * (int64_t) 27 + i_151583];
                
                // futhark/microgpt.fut:319:110-135
                
                double neg_res_138323 = -f_elem_138318;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_138324;
                double r_138326 = 0.0;
                
                for (int64_t i_138325 = 0; i_138325 < (int64_t) 27; i_138325++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_138327 = ((double *) mem_153359)[i_151587 * (int64_t) 27 + i_138325];
                    
                    // futhark/microgpt.fut:319:85-135
                    
                    double zp_res_138328 = neg_res_138323 + zp_lhs_138327;
                    
                    // futhark/microgpt.fut:319:78-135
                    
                    double exp_res_138329 = futrts_exp64(zp_res_138328);
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_138330 = ((double *) mem_153505)[i_151587 * (int64_t) 729 + i_151583 * (int64_t) 27 + i_138325];
                    
                    // futhark/microgpt.fut:319:78-170
                    
                    double zt_res_138331 = exp_res_138329 * zt_rhs_138330;
                    
                    // futhark/microgpt.fut:319:70-170
                    
                    double neg_res_138332 = -zt_res_138331;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_138333 = r_138326 + neg_res_138332;
                    double r_tmp_154992 = zp_res_138333;
                    
                    r_138326 = r_tmp_154992;
                }
                defunc_0_lifted_lambda_res_138324 = r_138326;
                ((double *) mem_153537)[i_151583] = defunc_0_lifted_lambda_res_138324;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153532, i_151587 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153537, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151595 = 0; i_151595 < (int64_t) 16; i_151595++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151591 = 0; i_151591 < (int64_t) 27; i_151591++) {
                double f_elem_138394 = ((double *) mem_153359)[i_151595 * (int64_t) 27 + i_151591];
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_138399;
                double r_138401 = 0.0;
                
                for (int64_t i_138400 = 0; i_138400 < (int64_t) 27; i_138400++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double neg_arg0_138402 = ((double *) mem_153378)[i_151595 * (int64_t) 27 + i_138400];
                    
                    // futhark/microgpt.fut:322:89-113
                    
                    double neg_res_138403 = -neg_arg0_138402;
                    
                    // futhark/microgpt.fut:322:66-113
                    
                    double zp_res_138404 = f_elem_138394 + neg_res_138403;
                    
                    // futhark/microgpt.fut:322:59-113
                    
                    double exp_res_138405 = futrts_exp64(zp_res_138404);
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_138406 = ((double *) mem_153505)[i_151595 * (int64_t) 729 + i_138400 * (int64_t) 27 + i_151591];
                    
                    // futhark/microgpt.fut:322:59-146
                    
                    double zt_res_138407 = exp_res_138405 * zt_rhs_138406;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_138408 = r_138401 + zt_res_138407;
                    double r_tmp_154995 = zp_res_138408;
                    
                    r_138401 = r_tmp_154995;
                }
                defunc_0_lifted_lambda_res_138399 = r_138401;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_138409;
                double r_138411 = 0.0;
                
                for (int64_t i_138410 = 0; i_138410 < (int64_t) 27; i_138410++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_138412 = ((double *) mem_153532)[i_151595 * (int64_t) 27 + i_138410];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double neg_arg0_138413 = ((double *) mem_153376)[i_151595 * (int64_t) 27 + i_138410];
                    
                    // futhark/microgpt.fut:322:260-284
                    
                    double neg_res_138414 = -neg_arg0_138413;
                    
                    // futhark/microgpt.fut:322:237-284
                    
                    double zp_res_138415 = f_elem_138394 + neg_res_138414;
                    
                    // futhark/microgpt.fut:322:230-284
                    
                    double neg_res_138416 = -zp_res_138415;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_138417 = fmax64(0.0, neg_res_138416);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_138418 = fsignum64(max_res_138417);
                    
                    // futhark/microgpt.fut:322:211-287
                    
                    double neg_res_138419 = -sgn_res_138418;
                    
                    // futhark/microgpt.fut:322:202-288
                    
                    double zp_res_138420 = 1.0 + neg_res_138419;
                    
                    // futhark/microgpt.fut:322:178-288
                    
                    double zt_res_138421 = zt_lhs_138412 * zp_res_138420;
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_138422 = ((double *) mem_153375)[i_151595 * (int64_t) 27 + i_138410];
                    
                    // futhark/microgpt.fut:322:197-314
                    
                    double zt_res_138423 = zt_res_138421 * zt_rhs_138422;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_138424 = r_138411 + zt_res_138423;
                    double r_tmp_154996 = zp_res_138424;
                    
                    r_138411 = r_tmp_154996;
                }
                defunc_0_lifted_lambda_res_138409 = r_138411;
                // futhark/microgpt.fut:322:36-316
                
                double zp_res_138425 = defunc_0_lifted_lambda_res_138399 + defunc_0_lifted_lambda_res_138409;
                
                ((double *) mem_153553)[i_151591] = zp_res_138425;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153548, i_151595 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153553, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151603 = 0; i_151603 < (int64_t) 16; i_151603++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151599 = 0; i_151599 < (int64_t) 16; i_151599++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_138440;
                double r_138442 = 0.0;
                
                for (int64_t i_138441 = 0; i_138441 < (int64_t) 27; i_138441++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_138443 = ((double *) mem_153548)[i_151603 * (int64_t) 27 + i_138441];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_138444 = ((double *) mem_param_152308.mem)[i_138441 * (int64_t) 16 + i_151599];
                    
                    // futhark/microgpt.fut:323:67-111
                    
                    double zt_res_138445 = zt_lhs_138443 * zt_rhs_138444;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_138446 = r_138442 + zt_res_138445;
                    double r_tmp_154999 = zp_res_138446;
                    
                    r_138442 = r_tmp_154999;
                }
                defunc_0_lifted_lambda_res_138440 = r_138442;
                ((double *) mem_153569)[i_151599] = defunc_0_lifted_lambda_res_138440;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153564, i_151603 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153569, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151616 = 0; i_151616 < (int64_t) 16; i_151616++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151609 = 0; i_151609 < (int64_t) 64; i_151609++) {
                // futhark/microgpt.fut:4:11-25
                
                double indicatorp_arg0_149107 = ((double *) mem_153311)[i_151616 * (int64_t) 64 + i_151609];
                
                // futhark/microgpt.fut:110:42-54
                
                double max_res_149108 = fmax64(0.0, indicatorp_arg0_149107);
                
                // futhark/microgpt.fut:110:35-54
                
                double sgn_res_149109 = fsignum64(max_res_149108);
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_149110;
                double r_149112 = 0.0;
                
                for (int64_t i_149111 = 0; i_149111 < (int64_t) 16; i_149111++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_149113 = ((double *) mem_153564)[i_151616 * (int64_t) 16 + i_149111];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_149114 = ((double *) mem_param_152276.mem)[i_149111 * (int64_t) 64 + i_151609];
                    
                    // futhark/microgpt.fut:324:105-151
                    
                    double zt_res_149115 = zt_lhs_149113 * zt_rhs_149114;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_149116 = r_149112 + zt_res_149115;
                    double r_tmp_155004 = zp_res_149116;
                    
                    r_149112 = r_tmp_155004;
                }
                defunc_0_lifted_lambda_res_149110 = r_149112;
                // futhark/microgpt.fut:324:46-153
                
                double zt_res_149117 = sgn_res_149109 * defunc_0_lifted_lambda_res_149110;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_149124;
                double r_149126 = 0.0;
                
                for (int64_t i_149125 = 0; i_149125 < (int64_t) 16; i_149125++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_149127 = ((double *) mem_153564)[i_149125 * (int64_t) 16 + i_151616];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_149128 = ((double *) mem_153327)[i_149125 * (int64_t) 64 + i_151609];
                    
                    // futhark/microgpt.fut:406:69-113
                    
                    double zt_res_149129 = zt_lhs_149127 * zt_rhs_149128;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_149130 = r_149126 + zt_res_149129;
                    double r_tmp_155005 = zp_res_149130;
                    
                    r_149126 = r_tmp_155005;
                }
                defunc_0_lifted_lambda_res_149124 = r_149126;
                ((double *) mem_153590)[i_151609] = defunc_0_lifted_lambda_res_149124;
                ((double *) mem_153591)[i_151609] = zt_res_149117;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153580, i_151616 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153590, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153581, i_151616 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153591, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151625 = 0; i_151625 < (int64_t) 16; i_151625++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151621 = 0; i_151621 < (int64_t) 16; i_151621++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_138510;
                double r_138512 = 0.0;
                
                for (int64_t i_138511 = 0; i_138511 < (int64_t) 64; i_138511++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_138513 = ((double *) mem_153581)[i_151625 * (int64_t) 64 + i_138511];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_138514 = ((double *) mem_param_152300.mem)[i_138511 * (int64_t) 16 + i_151621];
                    
                    // futhark/microgpt.fut:327:71-115
                    
                    double zt_res_138515 = zt_lhs_138513 * zt_rhs_138514;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_138516 = r_138512 + zt_res_138515;
                    double r_tmp_155008 = zp_res_138516;
                    
                    r_138512 = r_tmp_155008;
                }
                defunc_0_lifted_lambda_res_138510 = r_138512;
                ((double *) mem_153617)[i_151621] = defunc_0_lifted_lambda_res_138510;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153612, i_151625 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153617, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151631 = 0; i_151631 < (int64_t) 16; i_151631++) {
            // futhark/microgpt.fut:326:47-59
            
            double zp_lhs_141219 = ((double *) mem_153258)[i_151631];
            
            // futhark/microgpt.fut:326:47-87
            
            double zp_res_141220 = 1.0e-5 + zp_lhs_141219;
            
            // futhark/microgpt.fut:326:39-87
            
            double sqrt_res_141221 = futrts_sqrt64(zp_res_141220);
            
            // futhark/microgpt.fut:328:129-158
            
            double zt_res_141229 = sqrt_res_141221 * sqrt_res_141221;
            
            // futhark/microgpt.fut:328:120-158
            
            double zs_res_141230 = 1.0 / zt_res_141229;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_141231;
            double r_141233 = 0.0;
            
            for (int64_t i_141232 = 0; i_141232 < (int64_t) 16; i_141232++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_141234 = ((double *) mem_153612)[i_151631 * (int64_t) 16 + i_141232];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_141235 = ((double *) mem_153242)[i_151631 * (int64_t) 16 + i_141232];
                
                // futhark/microgpt.fut:328:69-113
                
                double zt_res_141236 = zt_lhs_141234 * zt_rhs_141235;
                
                // futhark/microgpt.fut:328:90-158
                
                double zt_res_141237 = zs_res_141230 * zt_res_141236;
                
                // futhark/microgpt.fut:328:61-158
                
                double neg_res_141238 = -zt_res_141237;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_141239 = r_141233 + neg_res_141238;
                double r_tmp_155011 = zp_res_141239;
                
                r_141233 = r_tmp_155011;
            }
            defunc_0_lifted_lambda_res_141231 = r_141233;
            ((double *) mem_153628)[i_151631] = defunc_0_lifted_lambda_res_141231;
            ((double *) mem_153629)[i_151631] = sqrt_res_141221;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151636 = 0; i_151636 < (int64_t) 16; i_151636++) {
            // futhark/microgpt.fut:329:39-51
            
            double zt_lhs_138544 = ((double *) mem_153628)[i_151636];
            
            // futhark/microgpt.fut:329:93-105
            
            double zp_lhs_138545 = ((double *) mem_153258)[i_151636];
            
            // futhark/microgpt.fut:329:93-133
            
            double zp_res_138546 = 1.0e-5 + zp_lhs_138545;
            
            // futhark/microgpt.fut:329:85-133
            
            double sqrt_res_138547 = futrts_sqrt64(zp_res_138546);
            
            // futhark/microgpt.fut:329:71-135
            
            double zt_res_138548 = 2.0 * sqrt_res_138547;
            
            // futhark/microgpt.fut:329:57-135
            
            double zs_res_138549 = 1.0 / zt_res_138548;
            
            // futhark/microgpt.fut:329:39-135
            
            double zt_res_138550 = zt_lhs_138544 * zs_res_138549;
            
            ((double *) mem_153642)[i_151636] = zt_res_138550;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151644 = 0; i_151644 < (int64_t) 16; i_151644++) {
            // futhark/microgpt.fut:330:98-110
            
            double zs_rhs_138558 = ((double *) mem_153629)[i_151644];
            
            // futhark/microgpt.fut:330:90-110
            
            double zs_res_138559 = 1.0 / zs_rhs_138558;
            
            // futhark/microgpt.fut:330:120-132
            
            double zs_lhs_138560 = ((double *) mem_153642)[i_151644];
            
            // futhark/microgpt.fut:330:120-147
            
            double zs_res_138561 = zs_lhs_138560 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151640 = 0; i_151640 < (int64_t) 16; i_151640++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_138568 = ((double *) mem_153564)[i_151644 * (int64_t) 16 + i_151640];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_138569 = ((double *) mem_153612)[i_151644 * (int64_t) 16 + i_151640];
                
                // futhark/microgpt.fut:330:64-110
                
                double zt_res_138570 = zs_res_138559 * zt_lhs_138569;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_138571 = ((double *) mem_153242)[i_151644 * (int64_t) 16 + i_151640];
                
                // futhark/microgpt.fut:330:133-172
                
                double zt_res_138572 = zs_res_138561 * zt_rhs_138571;
                
                // futhark/microgpt.fut:330:149-232
                
                double zp_res_138573 = zt_res_138572 + zt_res_138572;
                
                // futhark/microgpt.fut:330:85-232
                
                double zp_res_138574 = zt_res_138570 + zp_res_138573;
                
                // futhark/microgpt.fut:330:37-232
                
                double zp_res_138575 = zp_lhs_138568 + zp_res_138574;
                
                ((double *) mem_153654)[i_151640] = zp_res_138575;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153649, i_151644 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153654, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151656 = 0; i_151656 < (int64_t) 4; i_151656++) {
            // futhark/microgpt.fut:331:122-125
            
            int64_t zp_lhs_138580 = mul64((int64_t) 4, i_151656);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151652 = 0; i_151652 < (int64_t) 16; i_151652++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151648 = 0; i_151648 < (int64_t) 4; i_151648++) {
                    // futhark/microgpt.fut:331:127-135
                    
                    int64_t zt_rhs_138589 = add64(zp_lhs_138580, i_151648);
                    
                    // futhark/microgpt.fut:331:100-137
                    
                    bool x_138590 = sle64((int64_t) 0, zt_rhs_138589);
                    
                    // futhark/microgpt.fut:331:100-137
                    
                    bool y_138591 = slt64(zt_rhs_138589, (int64_t) 16);
                    
                    // futhark/microgpt.fut:331:100-137
                    
                    bool bounds_check_138592 = x_138590 && y_138591;
                    
                    // futhark/microgpt.fut:331:100-137
                    
                    bool index_certs_138593;
                    
                    if (!bounds_check_138592) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_rhs_138589, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:331:100-137\n   #1  futhark/microgpt.fut:71:46-49\n   #2  futhark/microgpt.fut:331:53-139\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:15:29-44\n   #8  futhark/microgpt.fut:4:11-25\n   #9  futhark/microgpt.fut:15:15-45\n   #10 futhark/microgpt.fut:331:13-141\n   #11 futhark/microgpt.fut:582:5-76\n   #12 futhark/microgpt.fut:599:26-605:31\n   #13 futhark/microgpt.fut:633:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_138594;
                    double r_138596 = 0.0;
                    
                    for (int64_t i_138595 = 0; i_138595 < (int64_t) 16; i_138595++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_138597 = ((double *) mem_153649)[i_151652 * (int64_t) 16 + i_138595];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_138598 = ((double *) mem_param_152284.mem)[i_138595 * (int64_t) 16 + zt_rhs_138589];
                        
                        // futhark/microgpt.fut:331:75-137
                        
                        double zt_res_138599 = zt_lhs_138597 * zt_rhs_138598;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_138600 = r_138596 + zt_res_138599;
                        double r_tmp_155018 = zp_res_138600;
                        
                        r_138596 = r_tmp_155018;
                    }
                    defunc_0_lifted_lambda_res_138594 = r_138596;
                    ((double *) mem_153676)[i_151648] = defunc_0_lifted_lambda_res_138594;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153671, i_151652 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153676, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153665, i_151656 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_153671, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151681 = 0; i_151681 < (int64_t) 4; i_151681++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151671 = 0; i_151671 < (int64_t) 16; i_151671++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151662 = 0; i_151662 < (int64_t) 4; i_151662++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_149287 = ((double *) mem_153665)[i_151681 * (int64_t) 64 + i_151671 * (int64_t) 4 + i_151662];
                    
                    ((double *) mem_153725)[i_151662] = lifted_lambda_res_149287;
                    ((double *) mem_153726)[i_151662] = lifted_lambda_res_149287;
                }
                // futhark/microgpt.fut:4:11-25
                // futhark/microgpt.fut:4:11-25
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153712, i_151671 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153726, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153710, i_151671 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153725, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153711, i_151671 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153726, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153692, i_151681 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_153710, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153693, i_151681 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_153711, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153694, i_151681 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_153712, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151703 = 0; i_151703 < (int64_t) 4; i_151703++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151696 = 0; i_151696 < (int64_t) 16; i_151696++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151689 = 0; i_151689 < (int64_t) 16; i_151689++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_149617;
                    double r_149619 = 0.0;
                    
                    for (int64_t i_149618 = 0; i_149618 < (int64_t) 4; i_149618++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_149620 = ((double *) mem_153693)[i_151703 * (int64_t) 64 + i_151696 * (int64_t) 4 + i_149618];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_149621 = ((double *) mem_152526)[i_151703 * (int64_t) 64 + i_151689 * (int64_t) 4 + i_149618];
                        
                        // futhark/microgpt.fut:344:79-139
                        
                        double zt_res_149622 = zt_lhs_149620 * zt_rhs_149621;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_149623 = r_149619 + zt_res_149622;
                        double r_tmp_155033 = zp_res_149623;
                        
                        r_149619 = r_tmp_155033;
                    }
                    defunc_0_lifted_lambda_res_149617 = r_149619;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_149630;
                    double r_149632 = 0.0;
                    
                    for (int64_t i_149631 = 0; i_149631 < (int64_t) 4; i_149631++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_149633 = ((double *) mem_153692)[i_151703 * (int64_t) 64 + i_151696 * (int64_t) 4 + i_149631];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_149634 = ((double *) mem_152526)[i_151703 * (int64_t) 64 + i_151689 * (int64_t) 4 + i_149631];
                        
                        // futhark/microgpt.fut:360:79-139
                        
                        double zt_res_149635 = zt_lhs_149633 * zt_rhs_149634;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_149636 = r_149632 + zt_res_149635;
                        double r_tmp_155034 = zp_res_149636;
                        
                        r_149632 = r_tmp_155034;
                    }
                    defunc_0_lifted_lambda_res_149630 = r_149632;
                    ((double *) mem_153789)[i_151689] = defunc_0_lifted_lambda_res_149630;
                    ((double *) mem_153790)[i_151689] = defunc_0_lifted_lambda_res_149617;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153779, i_151696 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153789, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153780, i_151696 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153790, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153767, i_151703 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153779, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153768, i_151703 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153780, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151724 = 0; i_151724 < (int64_t) 4; i_151724++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151717 = 0; i_151717 < (int64_t) 16; i_151717++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151710 = 0; i_151710 < (int64_t) 16; i_151710++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_149869 = ((double *) mem_153768)[i_151724 * (int64_t) 256 + i_151717 * (int64_t) 16 + i_151710];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_149876 = ((double *) mem_153767)[i_151724 * (int64_t) 256 + i_151717 * (int64_t) 16 + i_151710];
                    
                    ((double *) mem_153843)[i_151710] = lifted_lambda_res_149876;
                    ((double *) mem_153844)[i_151710] = lifted_lambda_res_149869;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153833, i_151717 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153843, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153834, i_151717 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153844, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153821, i_151724 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153833, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153822, i_151724 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153834, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151748 = 0; i_151748 < (int64_t) 4; i_151748++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151735 = 0; i_151735 < (int64_t) 16; i_151735++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_149749;
                double r_149751 = 0.0;
                
                for (int64_t i_149750 = 0; i_149750 < (int64_t) 16; i_149750++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_149752 = ((double *) mem_152952)[i_151748 * (int64_t) 256 + i_151735 * (int64_t) 16 + i_149750];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_149753 = r_149751 + lifted_lambda_res_149752;
                    double r_tmp_155049 = zp_res_149753;
                    
                    r_149751 = r_tmp_155049;
                }
                defunc_0_lifted_lambda_res_149749 = r_149751;
                // futhark/microgpt.fut:349:155-200
                
                double zt_res_149761 = defunc_0_lifted_lambda_res_149749 * defunc_0_lifted_lambda_res_149749;
                
                // futhark/microgpt.fut:349:146-200
                
                double zs_res_149762 = 1.0 / zt_res_149761;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_149763;
                double r_149765 = 0.0;
                
                for (int64_t i_149764 = 0; i_149764 < (int64_t) 16; i_149764++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_149766 = ((double *) mem_153822)[i_151748 * (int64_t) 256 + i_151735 * (int64_t) 16 + i_149764];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_149767 = ((double *) mem_152952)[i_151748 * (int64_t) 256 + i_151735 * (int64_t) 16 + i_149764];
                    
                    // futhark/microgpt.fut:349:78-139
                    
                    double zt_res_149768 = zt_lhs_149766 * zt_rhs_149767;
                    
                    // futhark/microgpt.fut:349:107-200
                    
                    double zt_res_149769 = zs_res_149762 * zt_res_149768;
                    
                    // futhark/microgpt.fut:349:70-200
                    
                    double neg_res_149770 = -zt_res_149769;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_149771 = r_149765 + neg_res_149770;
                    double r_tmp_155050 = zp_res_149771;
                    
                    r_149765 = r_tmp_155050;
                }
                defunc_0_lifted_lambda_res_149763 = r_149765;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_149782;
                double r_149784 = 0.0;
                
                for (int64_t i_149783 = 0; i_149783 < (int64_t) 16; i_149783++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_149785 = ((double *) mem_152951)[i_151748 * (int64_t) 256 + i_151735 * (int64_t) 16 + i_149783];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_149786 = r_149784 + lifted_lambda_res_149785;
                    double r_tmp_155051 = zp_res_149786;
                    
                    r_149784 = r_tmp_155051;
                }
                defunc_0_lifted_lambda_res_149782 = r_149784;
                // futhark/microgpt.fut:365:155-200
                
                double zt_res_149794 = defunc_0_lifted_lambda_res_149782 * defunc_0_lifted_lambda_res_149782;
                
                // futhark/microgpt.fut:365:146-200
                
                double zs_res_149795 = 1.0 / zt_res_149794;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_149796;
                double r_149798 = 0.0;
                
                for (int64_t i_149797 = 0; i_149797 < (int64_t) 16; i_149797++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_149799 = ((double *) mem_153821)[i_151748 * (int64_t) 256 + i_151735 * (int64_t) 16 + i_149797];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_149800 = ((double *) mem_152951)[i_151748 * (int64_t) 256 + i_151735 * (int64_t) 16 + i_149797];
                    
                    // futhark/microgpt.fut:365:78-139
                    
                    double zt_res_149801 = zt_lhs_149799 * zt_rhs_149800;
                    
                    // futhark/microgpt.fut:365:107-200
                    
                    double zt_res_149802 = zs_res_149795 * zt_res_149801;
                    
                    // futhark/microgpt.fut:365:70-200
                    
                    double neg_res_149803 = -zt_res_149802;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_149804 = r_149798 + neg_res_149803;
                    double r_tmp_155052 = zp_res_149804;
                    
                    r_149798 = r_tmp_155052;
                }
                defunc_0_lifted_lambda_res_149796 = r_149798;
                ((double *) mem_153895)[i_151735] = defunc_0_lifted_lambda_res_149796;
                ((double *) mem_153896)[i_151735] = defunc_0_lifted_lambda_res_149782;
                ((double *) mem_153897)[i_151735] = defunc_0_lifted_lambda_res_149763;
                ((double *) mem_153898)[i_151735] = defunc_0_lifted_lambda_res_149749;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153875, i_151748 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153895, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153876, i_151748 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153896, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153877, i_151748 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153897, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153878, i_151748 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153898, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151771 = 0; i_151771 < (int64_t) 4; i_151771++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151764 = 0; i_151764 < (int64_t) 16; i_151764++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_149900 = ((double *) mem_153878)[i_151771 * (int64_t) 16 + i_151764];
                
                // futhark/microgpt.fut:350:93-121
                
                double zs_res_149901 = 1.0 / zs_rhs_149900;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_149902 = ((double *) mem_153877)[i_151771 * (int64_t) 16 + i_151764];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_149921 = ((double *) mem_153875)[i_151771 * (int64_t) 16 + i_151764];
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_149919 = ((double *) mem_153876)[i_151771 * (int64_t) 16 + i_151764];
                
                // futhark/microgpt.fut:366:93-121
                
                double zs_res_149920 = 1.0 / zs_rhs_149919;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151757 = 0; i_151757 < (int64_t) 16; i_151757++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_149949 = ((double *) mem_153822)[i_151771 * (int64_t) 256 + i_151764 * (int64_t) 16 + i_151757];
                    
                    // futhark/microgpt.fut:350:59-121
                    
                    double zt_res_149950 = zs_res_149901 * zt_lhs_149949;
                    
                    // futhark/microgpt.fut:350:88-148
                    
                    double zp_res_149951 = zp_rhs_149902 + zt_res_149950;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_149958 = ((double *) mem_153821)[i_151771 * (int64_t) 256 + i_151764 * (int64_t) 16 + i_151757];
                    
                    // futhark/microgpt.fut:366:59-121
                    
                    double zt_res_149959 = zs_res_149920 * zt_lhs_149958;
                    
                    // futhark/microgpt.fut:366:88-148
                    
                    double zp_res_149960 = zp_rhs_149921 + zt_res_149959;
                    
                    ((double *) mem_153961)[i_151757] = zp_res_149960;
                    ((double *) mem_153962)[i_151757] = zp_res_149951;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153951, i_151764 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153961, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_153952, i_151764 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_153962, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153939, i_151771 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153951, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_153940, i_151771 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_153952, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151785 = 0; i_151785 < (int64_t) 4; i_151785++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151778 = 0; i_151778 < (int64_t) 16; i_151778++) {
                double f_elem_149980 = ((double *) mem_152828)[i_151785 * (int64_t) 16 + i_151778];
                double f_elem_149982 = ((double *) mem_152825)[i_151785 * (int64_t) 16 + i_151778];
                
                // futhark/microgpt.fut:351:119-145
                
                double neg_res_149987 = -f_elem_149980;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_149988;
                double r_149990 = 0.0;
                
                for (int64_t i_149989 = 0; i_149989 < (int64_t) 16; i_149989++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_149991 = ((double *) mem_152716)[i_151785 * (int64_t) 256 + i_151778 * (int64_t) 16 + i_149989];
                    
                    // futhark/microgpt.fut:351:85-145
                    
                    double zp_res_149992 = neg_res_149987 + zp_lhs_149991;
                    
                    // futhark/microgpt.fut:351:78-145
                    
                    double exp_res_149993 = futrts_exp64(zp_res_149992);
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_149994 = ((double *) mem_153940)[i_151785 * (int64_t) 256 + i_151778 * (int64_t) 16 + i_149989];
                    
                    // futhark/microgpt.fut:351:78-181
                    
                    double zt_res_149995 = exp_res_149993 * zt_rhs_149994;
                    
                    // futhark/microgpt.fut:351:70-181
                    
                    double neg_res_149996 = -zt_res_149995;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_149997 = r_149990 + neg_res_149996;
                    double r_tmp_155063 = zp_res_149997;
                    
                    r_149990 = r_tmp_155063;
                }
                defunc_0_lifted_lambda_res_149988 = r_149990;
                // futhark/microgpt.fut:367:119-145
                
                double neg_res_150005 = -f_elem_149982;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_150006;
                double r_150008 = 0.0;
                
                for (int64_t i_150007 = 0; i_150007 < (int64_t) 16; i_150007++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_150009 = ((double *) mem_152715)[i_151785 * (int64_t) 256 + i_151778 * (int64_t) 16 + i_150007];
                    
                    // futhark/microgpt.fut:367:85-145
                    
                    double zp_res_150010 = neg_res_150005 + zp_lhs_150009;
                    
                    // futhark/microgpt.fut:367:78-145
                    
                    double exp_res_150011 = futrts_exp64(zp_res_150010);
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_150012 = ((double *) mem_153939)[i_151785 * (int64_t) 256 + i_151778 * (int64_t) 16 + i_150007];
                    
                    // futhark/microgpt.fut:367:78-181
                    
                    double zt_res_150013 = exp_res_150011 * zt_rhs_150012;
                    
                    // futhark/microgpt.fut:367:70-181
                    
                    double neg_res_150014 = -zt_res_150013;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_150015 = r_150008 + neg_res_150014;
                    double r_tmp_155064 = zp_res_150015;
                    
                    r_150008 = r_tmp_155064;
                }
                defunc_0_lifted_lambda_res_150006 = r_150008;
                ((double *) mem_154003)[i_151778] = defunc_0_lifted_lambda_res_150006;
                ((double *) mem_154004)[i_151778] = defunc_0_lifted_lambda_res_149988;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153993, i_151785 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154003, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_153994, i_151785 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154004, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151806 = 0; i_151806 < (int64_t) 4; i_151806++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151799 = 0; i_151799 < (int64_t) 16; i_151799++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_150035 = ((double *) mem_152828)[i_151806 * (int64_t) 16 + i_151799];
                
                // futhark/microgpt.fut:354:101-127
                
                double neg_res_150036 = -neg_arg0_150035;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_150037 = ((double *) mem_153994)[i_151806 * (int64_t) 16 + i_151799];
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_150038 = ((double *) mem_152827)[i_151806 * (int64_t) 16 + i_151799];
                
                // futhark/microgpt.fut:354:266-292
                
                double neg_res_150039 = -neg_arg0_150038;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_150040 = ((double *) mem_152826)[i_151806 * (int64_t) 16 + i_151799];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_150073 = ((double *) mem_152823)[i_151806 * (int64_t) 16 + i_151799];
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_150071 = ((double *) mem_152824)[i_151806 * (int64_t) 16 + i_151799];
                
                // futhark/microgpt.fut:370:266-292
                
                double neg_res_150072 = -neg_arg0_150071;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_150070 = ((double *) mem_153993)[i_151806 * (int64_t) 16 + i_151799];
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_150068 = ((double *) mem_152825)[i_151806 * (int64_t) 16 + i_151799];
                
                // futhark/microgpt.fut:370:101-127
                
                double neg_res_150069 = -neg_arg0_150068;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151792 = 0; i_151792 < (int64_t) 16; i_151792++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_150112 = ((double *) mem_152716)[i_151806 * (int64_t) 256 + i_151799 * (int64_t) 16 + i_151792];
                    
                    // futhark/microgpt.fut:354:67-127
                    
                    double zp_res_150113 = neg_res_150036 + zp_lhs_150112;
                    
                    // futhark/microgpt.fut:354:60-127
                    
                    double exp_res_150114 = futrts_exp64(zp_res_150113);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_150115 = ((double *) mem_153940)[i_151806 * (int64_t) 256 + i_151799 * (int64_t) 16 + i_151792];
                    
                    // futhark/microgpt.fut:354:60-163
                    
                    double zt_res_150116 = exp_res_150114 * zt_rhs_150115;
                    
                    // futhark/microgpt.fut:354:232-292
                    
                    double zp_res_150117 = neg_res_150039 + zp_lhs_150112;
                    
                    // futhark/microgpt.fut:354:225-292
                    
                    double neg_res_150118 = -zp_res_150117;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_150119 = fmax64(0.0, neg_res_150118);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_150120 = fsignum64(max_res_150119);
                    
                    // futhark/microgpt.fut:354:206-295
                    
                    double neg_res_150121 = -sgn_res_150120;
                    
                    // futhark/microgpt.fut:354:197-296
                    
                    double zp_res_150122 = 1.0 + neg_res_150121;
                    
                    // futhark/microgpt.fut:354:171-296
                    
                    double zt_res_150123 = zt_lhs_150037 * zp_res_150122;
                    
                    // futhark/microgpt.fut:354:192-324
                    
                    double zt_res_150124 = zt_rhs_150040 * zt_res_150123;
                    
                    // futhark/microgpt.fut:354:131-324
                    
                    double zp_res_150125 = zt_res_150116 + zt_res_150124;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_150132 = ((double *) mem_152715)[i_151806 * (int64_t) 256 + i_151799 * (int64_t) 16 + i_151792];
                    
                    // futhark/microgpt.fut:370:67-127
                    
                    double zp_res_150133 = neg_res_150069 + zp_lhs_150132;
                    
                    // futhark/microgpt.fut:370:60-127
                    
                    double exp_res_150134 = futrts_exp64(zp_res_150133);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_150135 = ((double *) mem_153939)[i_151806 * (int64_t) 256 + i_151799 * (int64_t) 16 + i_151792];
                    
                    // futhark/microgpt.fut:370:60-163
                    
                    double zt_res_150136 = exp_res_150134 * zt_rhs_150135;
                    
                    // futhark/microgpt.fut:370:232-292
                    
                    double zp_res_150137 = neg_res_150072 + zp_lhs_150132;
                    
                    // futhark/microgpt.fut:370:225-292
                    
                    double neg_res_150138 = -zp_res_150137;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_150139 = fmax64(0.0, neg_res_150138);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_150140 = fsignum64(max_res_150139);
                    
                    // futhark/microgpt.fut:370:206-295
                    
                    double neg_res_150141 = -sgn_res_150140;
                    
                    // futhark/microgpt.fut:370:197-296
                    
                    double zp_res_150142 = 1.0 + neg_res_150141;
                    
                    // futhark/microgpt.fut:370:171-296
                    
                    double zt_res_150143 = zt_lhs_150070 * zp_res_150142;
                    
                    // futhark/microgpt.fut:370:192-324
                    
                    double zt_res_150144 = zt_rhs_150073 * zt_res_150143;
                    
                    // futhark/microgpt.fut:370:131-324
                    
                    double zp_res_150145 = zt_res_150136 + zt_res_150144;
                    
                    ((double *) mem_154047)[i_151792] = zp_res_150145;
                    ((double *) mem_154048)[i_151792] = zp_res_150125;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154037, i_151799 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154047, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154038, i_151799 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154048, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154025, i_151806 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154037, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154026, i_151806 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154038, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151827 = 0; i_151827 < (int64_t) 4; i_151827++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151820 = 0; i_151820 < (int64_t) 16; i_151820++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_151813 = 0; i_151813 < (int64_t) 16; i_151813++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_150210 = ((double *) mem_154026)[i_151827 * (int64_t) 256 + i_151820 * (int64_t) 16 + i_151813];
                    
                    // futhark/microgpt.fut:355:58-100
                    
                    double zs_res_150211 = zs_lhs_150210 / 2.0;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_150218 = ((double *) mem_154025)[i_151827 * (int64_t) 256 + i_151820 * (int64_t) 16 + i_151813];
                    
                    // futhark/microgpt.fut:371:58-100
                    
                    double zs_res_150219 = zs_lhs_150218 / 2.0;
                    
                    ((double *) mem_154101)[i_151813] = zs_res_150219;
                    ((double *) mem_154102)[i_151813] = zs_res_150211;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154091, i_151820 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154101, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_154092, i_151820 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154102, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154079, i_151827 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154091, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_154080, i_151827 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154092, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151851 = 0; i_151851 < (int64_t) 16; i_151851++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151838 = 0; i_151838 < (int64_t) 16; i_151838++) {
                // futhark/microgpt.fut:340:40-43
                
                int64_t zt_lhs_149467 = sdiv64(i_151838, (int64_t) 4);
                
                // futhark/microgpt.fut:340:27-45
                
                bool x_149468 = sle64((int64_t) 0, zt_lhs_149467);
                
                // futhark/microgpt.fut:340:27-45
                
                bool y_149469 = slt64(zt_lhs_149467, (int64_t) 4);
                
                // futhark/microgpt.fut:340:27-45
                
                bool bounds_check_149470 = x_149468 && y_149469;
                
                // futhark/microgpt.fut:340:27-45
                
                bool index_certs_149471;
                
                if (!bounds_check_149470) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_149467, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:340:27-45\n   #1  futhark/microgpt.fut:71:46-49\n   #2  futhark/microgpt.fut:340:5-108\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:332:13-340:118\n   #8  futhark/microgpt.fut:582:5-76\n   #9  futhark/microgpt.fut:599:26-605:31\n   #10 futhark/microgpt.fut:633:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:340:62-65
                
                int64_t zt_lhs_149472 = smod64(i_151838, (int64_t) 4);
                
                // futhark/microgpt.fut:340:27-67
                
                bool x_149473 = sle64((int64_t) 0, zt_lhs_149472);
                
                // futhark/microgpt.fut:340:27-67
                
                bool y_149474 = slt64(zt_lhs_149472, (int64_t) 4);
                
                // futhark/microgpt.fut:340:27-67
                
                bool bounds_check_149475 = x_149473 && y_149474;
                
                // futhark/microgpt.fut:340:27-67
                
                bool index_certs_149476;
                
                if (!bounds_check_149475) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_149472, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:340:27-67\n   #1  futhark/microgpt.fut:71:46-49\n   #2  futhark/microgpt.fut:340:5-108\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:332:13-340:118\n   #8  futhark/microgpt.fut:582:5-76\n   #9  futhark/microgpt.fut:599:26-605:31\n   #10 futhark/microgpt.fut:633:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_149477;
                double r_149479 = 0.0;
                
                for (int64_t i_149478 = 0; i_149478 < (int64_t) 16; i_149478++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_149480 = ((double *) mem_153694)[zt_lhs_149467 * (int64_t) 64 + i_149478 * (int64_t) 4 + zt_lhs_149472];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_149481 = ((double *) mem_153145)[zt_lhs_149467 * (int64_t) 256 + i_149478 * (int64_t) 16 + i_151851];
                    
                    // futhark/microgpt.fut:340:27-106
                    
                    double zt_res_149482 = zt_lhs_149480 * zt_rhs_149481;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_149483 = r_149479 + zt_res_149482;
                    double r_tmp_155085 = zp_res_149483;
                    
                    r_149479 = r_tmp_155085;
                }
                defunc_0_lifted_lambda_res_149477 = r_149479;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_149496;
                double r_149498 = 0.0;
                
                for (int64_t i_149497 = 0; i_149497 < (int64_t) 16; i_149497++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_149499 = ((double *) mem_154080)[zt_lhs_149467 * (int64_t) 256 + i_149497 * (int64_t) 16 + i_151851];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_149500 = ((double *) mem_152528)[zt_lhs_149467 * (int64_t) 64 + i_149497 * (int64_t) 4 + zt_lhs_149472];
                    
                    // futhark/microgpt.fut:356:27-105
                    
                    double zt_res_149501 = zt_lhs_149499 * zt_rhs_149500;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_149502 = r_149498 + zt_res_149501;
                    double r_tmp_155086 = zp_res_149502;
                    
                    r_149498 = r_tmp_155086;
                }
                defunc_0_lifted_lambda_res_149496 = r_149498;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_149518;
                double r_149520 = 0.0;
                
                for (int64_t i_149519 = 0; i_149519 < (int64_t) 16; i_149519++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_149521 = ((double *) mem_154079)[zt_lhs_149467 * (int64_t) 256 + i_151851 * (int64_t) 16 + i_149519];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_149522 = ((double *) mem_152527)[zt_lhs_149467 * (int64_t) 64 + i_149519 * (int64_t) 4 + zt_lhs_149472];
                    
                    // futhark/microgpt.fut:372:27-105
                    
                    double zt_res_149523 = zt_lhs_149521 * zt_rhs_149522;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_149524 = r_149520 + zt_res_149523;
                    double r_tmp_155087 = zp_res_149524;
                    
                    r_149520 = r_tmp_155087;
                }
                defunc_0_lifted_lambda_res_149518 = r_149520;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_149536;
                double r_149538 = 0.0;
                
                for (int64_t i_149537 = 0; i_149537 < (int64_t) 16; i_149537++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_149539 = ((double *) mem_153649)[i_149537 * (int64_t) 16 + i_151851];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_149540 = ((double *) mem_153226)[i_149537 * (int64_t) 16 + i_151838];
                    
                    // futhark/microgpt.fut:404:68-112
                    
                    double zt_res_149541 = zt_lhs_149539 * zt_rhs_149540;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_149542 = r_149538 + zt_res_149541;
                    double r_tmp_155088 = zp_res_149542;
                    
                    r_149538 = r_tmp_155088;
                }
                defunc_0_lifted_lambda_res_149536 = r_149538;
                ((double *) mem_154153)[i_151838] = defunc_0_lifted_lambda_res_149536;
                ((double *) mem_154154)[i_151838] = defunc_0_lifted_lambda_res_149518;
                ((double *) mem_154155)[i_151838] = defunc_0_lifted_lambda_res_149496;
                ((double *) mem_154156)[i_151838] = defunc_0_lifted_lambda_res_149477;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154133, i_151851 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154153, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154134, i_151851 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154154, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154135, i_151851 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154155, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154136, i_151851 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154156, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151862 = 0; i_151862 < (int64_t) 16; i_151862++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151858 = 0; i_151858 < (int64_t) 16; i_151858++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_139719;
                double r_139721 = 0.0;
                
                for (int64_t i_139720 = 0; i_139720 < (int64_t) 16; i_139720++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_139722 = ((double *) mem_154136)[i_151862 * (int64_t) 16 + i_139720];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_139723 = ((double *) mem_param_152304.mem)[i_139720 * (int64_t) 16 + i_151858];
                    
                    // futhark/microgpt.fut:375:73-118
                    
                    double zt_res_139724 = zt_lhs_139722 * zt_rhs_139723;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_139725 = r_139721 + zt_res_139724;
                    double r_tmp_155091 = zp_res_139725;
                    
                    r_139721 = r_tmp_155091;
                }
                defunc_0_lifted_lambda_res_139719 = r_139721;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_139726;
                double r_139728 = 0.0;
                
                for (int64_t i_139727 = 0; i_139727 < (int64_t) 16; i_139727++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_139729 = ((double *) mem_154135)[i_151862 * (int64_t) 16 + i_139727];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_139730 = ((double *) mem_param_152280.mem)[i_139727 * (int64_t) 16 + i_151858];
                    
                    // futhark/microgpt.fut:375:149-194
                    
                    double zt_res_139731 = zt_lhs_139729 * zt_rhs_139730;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_139732 = r_139728 + zt_res_139731;
                    double r_tmp_155092 = zp_res_139732;
                    
                    r_139728 = r_tmp_155092;
                }
                defunc_0_lifted_lambda_res_139726 = r_139728;
                // futhark/microgpt.fut:375:51-196
                
                double zp_res_139733 = defunc_0_lifted_lambda_res_139719 + defunc_0_lifted_lambda_res_139726;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_139734;
                double r_139736 = 0.0;
                
                for (int64_t i_139735 = 0; i_139735 < (int64_t) 16; i_139735++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_139737 = ((double *) mem_154134)[i_151862 * (int64_t) 16 + i_139735];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_139738 = ((double *) mem_param_152292.mem)[i_139735 * (int64_t) 16 + i_151858];
                    
                    // futhark/microgpt.fut:375:226-271
                    
                    double zt_res_139739 = zt_lhs_139737 * zt_rhs_139738;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_139740 = r_139736 + zt_res_139739;
                    double r_tmp_155093 = zp_res_139740;
                    
                    r_139736 = r_tmp_155093;
                }
                defunc_0_lifted_lambda_res_139734 = r_139736;
                // futhark/microgpt.fut:375:122-273
                
                double zp_res_139741 = zp_res_139733 + defunc_0_lifted_lambda_res_139734;
                
                ((double *) mem_154202)[i_151858] = zp_res_139741;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154197, i_151862 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154202, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151884 = 0; i_151884 < (int64_t) 16; i_151884++) {
            // futhark/microgpt.fut:374:47-59
            
            double zp_lhs_145070 = ((double *) mem_152473)[i_151884];
            
            // futhark/microgpt.fut:374:47-87
            
            double zp_res_145071 = 1.0e-5 + zp_lhs_145070;
            
            // futhark/microgpt.fut:374:39-87
            
            double sqrt_res_145072 = futrts_sqrt64(zp_res_145071);
            
            // futhark/microgpt.fut:376:128-157
            
            double zt_res_145080 = sqrt_res_145072 * sqrt_res_145072;
            
            // futhark/microgpt.fut:376:119-157
            
            double zs_res_145081 = 1.0 / zt_res_145080;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_145082;
            double r_145084 = 0.0;
            
            for (int64_t i_145083 = 0; i_145083 < (int64_t) 16; i_145083++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_145085 = ((double *) mem_154197)[i_151884 * (int64_t) 16 + i_145083];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_145086 = ((double *) mem_152457)[i_151884 * (int64_t) 16 + i_145083];
                
                // futhark/microgpt.fut:376:69-112
                
                double zt_res_145087 = zt_lhs_145085 * zt_rhs_145086;
                
                // futhark/microgpt.fut:376:90-157
                
                double zt_res_145088 = zs_res_145081 * zt_res_145087;
                
                // futhark/microgpt.fut:376:61-157
                
                double neg_res_145089 = -zt_res_145088;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_145090 = r_145084 + neg_res_145089;
                double r_tmp_155099 = zp_res_145090;
                
                r_145084 = r_tmp_155099;
            }
            defunc_0_lifted_lambda_res_145082 = r_145084;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151870 = 0; i_151870 < (int64_t) 16; i_151870++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_150286;
                double r_150288 = 0.0;
                
                for (int64_t i_150287 = 0; i_150287 < (int64_t) 16; i_150287++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_150289 = ((double *) mem_154134)[i_150287 * (int64_t) 16 + i_151884];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_150290 = ((double *) mem_152510)[i_150287 * (int64_t) 16 + i_151870];
                    
                    // futhark/microgpt.fut:401:68-111
                    
                    double zt_res_150291 = zt_lhs_150289 * zt_rhs_150290;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_150292 = r_150288 + zt_res_150291;
                    double r_tmp_155103 = zp_res_150292;
                    
                    r_150288 = r_tmp_155103;
                }
                defunc_0_lifted_lambda_res_150286 = r_150288;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_150299;
                double r_150301 = 0.0;
                
                for (int64_t i_150300 = 0; i_150300 < (int64_t) 16; i_150300++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_150302 = ((double *) mem_154135)[i_150300 * (int64_t) 16 + i_151884];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_150303 = ((double *) mem_152510)[i_150300 * (int64_t) 16 + i_151870];
                    
                    // futhark/microgpt.fut:402:68-111
                    
                    double zt_res_150304 = zt_lhs_150302 * zt_rhs_150303;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_150305 = r_150301 + zt_res_150304;
                    double r_tmp_155104 = zp_res_150305;
                    
                    r_150301 = r_tmp_155104;
                }
                defunc_0_lifted_lambda_res_150299 = r_150301;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_150315;
                double r_150317 = 0.0;
                
                for (int64_t i_150316 = 0; i_150316 < (int64_t) 16; i_150316++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_150318 = ((double *) mem_154136)[i_150316 * (int64_t) 16 + i_151884];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_150319 = ((double *) mem_152510)[i_150316 * (int64_t) 16 + i_151870];
                    
                    // futhark/microgpt.fut:403:68-111
                    
                    double zt_res_150320 = zt_lhs_150318 * zt_rhs_150319;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_150321 = r_150317 + zt_res_150320;
                    double r_tmp_155105 = zp_res_150321;
                    
                    r_150317 = r_tmp_155105;
                }
                defunc_0_lifted_lambda_res_150315 = r_150317;
                ((double *) mem_154236)[i_151870] = defunc_0_lifted_lambda_res_150315;
                ((double *) mem_154237)[i_151870] = defunc_0_lifted_lambda_res_150299;
                ((double *) mem_154238)[i_151870] = defunc_0_lifted_lambda_res_150286;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154213, i_151884 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154236, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154214, i_151884 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154237, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154215, i_151884 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154238, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            ((double *) mem_154216)[i_151884] = defunc_0_lifted_lambda_res_145082;
            ((double *) mem_154217)[i_151884] = sqrt_res_145072;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151892 = 0; i_151892 < (int64_t) 16; i_151892++) {
            // futhark/microgpt.fut:377:39-51
            
            double zt_lhs_139769 = ((double *) mem_154216)[i_151892];
            
            // futhark/microgpt.fut:377:93-105
            
            double zp_lhs_139770 = ((double *) mem_152473)[i_151892];
            
            // futhark/microgpt.fut:377:93-133
            
            double zp_res_139771 = 1.0e-5 + zp_lhs_139770;
            
            // futhark/microgpt.fut:377:85-133
            
            double sqrt_res_139772 = futrts_sqrt64(zp_res_139771);
            
            // futhark/microgpt.fut:377:71-135
            
            double zt_res_139773 = 2.0 * sqrt_res_139772;
            
            // futhark/microgpt.fut:377:57-135
            
            double zs_res_139774 = 1.0 / zt_res_139773;
            
            // futhark/microgpt.fut:377:39-135
            
            double zt_res_139775 = zt_lhs_139769 * zs_res_139774;
            
            ((double *) mem_154275)[i_151892] = zt_res_139775;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151900 = 0; i_151900 < (int64_t) 16; i_151900++) {
            // futhark/microgpt.fut:378:98-110
            
            double zs_rhs_139783 = ((double *) mem_154217)[i_151900];
            
            // futhark/microgpt.fut:378:90-110
            
            double zs_res_139784 = 1.0 / zs_rhs_139783;
            
            // futhark/microgpt.fut:378:120-132
            
            double zs_lhs_139785 = ((double *) mem_154275)[i_151900];
            
            // futhark/microgpt.fut:378:120-147
            
            double zs_res_139786 = zs_lhs_139785 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151896 = 0; i_151896 < (int64_t) 16; i_151896++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_139793 = ((double *) mem_153649)[i_151900 * (int64_t) 16 + i_151896];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_139794 = ((double *) mem_154197)[i_151900 * (int64_t) 16 + i_151896];
                
                // futhark/microgpt.fut:378:64-110
                
                double zt_res_139795 = zs_res_139784 * zt_lhs_139794;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_139796 = ((double *) mem_152457)[i_151900 * (int64_t) 16 + i_151896];
                
                // futhark/microgpt.fut:378:133-171
                
                double zt_res_139797 = zs_res_139786 * zt_rhs_139796;
                
                // futhark/microgpt.fut:378:149-230
                
                double zp_res_139798 = zt_res_139797 + zt_res_139797;
                
                // futhark/microgpt.fut:378:85-230
                
                double zp_res_139799 = zt_res_139795 + zp_res_139798;
                
                // futhark/microgpt.fut:378:37-230
                
                double zp_res_139800 = zp_lhs_139793 + zp_res_139799;
                
                ((double *) mem_154287)[i_151896] = zp_res_139800;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154282, i_151900 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154287, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151913 = 0; i_151913 < (int64_t) 16; i_151913++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151906 = 0; i_151906 < (int64_t) 16; i_151906++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_150345 = ((double *) mem_154282)[i_151913 * (int64_t) 16 + i_151906];
                
                ((double *) mem_154308)[i_151906] = lifted_lambda_res_150345;
                ((double *) mem_154309)[i_151906] = lifted_lambda_res_150345;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154298, i_151913 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154308, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154299, i_151913 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154309, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151924 = 0; i_151924 < (int64_t) 16; i_151924++) {
            // futhark/microgpt.fut:396:47-59
            
            double zp_lhs_145195 = ((double *) mem_152414)[i_151924];
            
            // futhark/microgpt.fut:396:47-87
            
            double zp_res_145196 = 1.0e-5 + zp_lhs_145195;
            
            // futhark/microgpt.fut:396:39-87
            
            double sqrt_res_145197 = futrts_sqrt64(zp_res_145196);
            
            // futhark/microgpt.fut:398:156-185
            
            double zt_res_145205 = sqrt_res_145197 * sqrt_res_145197;
            
            // futhark/microgpt.fut:398:147-185
            
            double zs_res_145206 = 1.0 / zt_res_145205;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_145207;
            double r_145209 = 0.0;
            
            for (int64_t i_145208 = 0; i_145208 < (int64_t) 16; i_145208++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_145210 = ((double *) mem_154299)[i_151924 * (int64_t) 16 + i_145208];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_145211 = ((double *) mem_param_152288.mem)[i_151924 * (int64_t) 16 + i_145208];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_rhs_145212 = ((double *) mem_152381)[i_151924 * (int64_t) 16 + i_145208];
                
                // futhark/microgpt.fut:398:95-139
                
                double zp_res_145213 = zp_lhs_145211 + zp_rhs_145212;
                
                // futhark/microgpt.fut:398:69-139
                
                double zt_res_145214 = zt_lhs_145210 * zp_res_145213;
                
                // futhark/microgpt.fut:398:90-185
                
                double zt_res_145215 = zs_res_145206 * zt_res_145214;
                
                // futhark/microgpt.fut:398:61-185
                
                double neg_res_145216 = -zt_res_145215;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_145217 = r_145209 + neg_res_145216;
                double r_tmp_155117 = zp_res_145217;
                
                r_145209 = r_tmp_155117;
            }
            defunc_0_lifted_lambda_res_145207 = r_145209;
            // futhark/microgpt.fut:409:47-59
            
            double zp_lhs_145228 = ((double *) mem_152413)[i_151924];
            
            // futhark/microgpt.fut:409:47-87
            
            double zp_res_145229 = 1.0e-5 + zp_lhs_145228;
            
            // futhark/microgpt.fut:409:39-87
            
            double sqrt_res_145230 = futrts_sqrt64(zp_res_145229);
            
            // futhark/microgpt.fut:411:156-185
            
            double zt_res_145238 = sqrt_res_145230 * sqrt_res_145230;
            
            // futhark/microgpt.fut:411:147-185
            
            double zs_res_145239 = 1.0 / zt_res_145238;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_145240;
            double r_145242 = 0.0;
            
            for (int64_t i_145241 = 0; i_145241 < (int64_t) 16; i_145241++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_145243 = ((double *) mem_154298)[i_151924 * (int64_t) 16 + i_145241];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_145244 = ((double *) mem_param_152288.mem)[i_151924 * (int64_t) 16 + i_145241];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_rhs_145245 = ((double *) mem_152381)[i_151924 * (int64_t) 16 + i_145241];
                
                // futhark/microgpt.fut:411:95-139
                
                double zp_res_145246 = zp_lhs_145244 + zp_rhs_145245;
                
                // futhark/microgpt.fut:411:69-139
                
                double zt_res_145247 = zt_lhs_145243 * zp_res_145246;
                
                // futhark/microgpt.fut:411:90-185
                
                double zt_res_145248 = zs_res_145239 * zt_res_145247;
                
                // futhark/microgpt.fut:411:61-185
                
                double neg_res_145249 = -zt_res_145248;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_145250 = r_145242 + neg_res_145249;
                double r_tmp_155118 = zp_res_145250;
                
                r_145242 = r_tmp_155118;
            }
            defunc_0_lifted_lambda_res_145240 = r_145242;
            ((double *) mem_154330)[i_151924] = defunc_0_lifted_lambda_res_145240;
            ((double *) mem_154331)[i_151924] = sqrt_res_145230;
            ((double *) mem_154332)[i_151924] = defunc_0_lifted_lambda_res_145207;
            ((double *) mem_154333)[i_151924] = sqrt_res_145197;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151933 = 0; i_151933 < (int64_t) 16; i_151933++) {
            // futhark/microgpt.fut:399:39-51
            
            double zt_lhs_145311 = ((double *) mem_154332)[i_151933];
            
            // futhark/microgpt.fut:399:93-105
            
            double zp_lhs_145312 = ((double *) mem_152414)[i_151933];
            
            // futhark/microgpt.fut:399:93-133
            
            double zp_res_145313 = 1.0e-5 + zp_lhs_145312;
            
            // futhark/microgpt.fut:399:85-133
            
            double sqrt_res_145314 = futrts_sqrt64(zp_res_145313);
            
            // futhark/microgpt.fut:399:71-135
            
            double zt_res_145315 = 2.0 * sqrt_res_145314;
            
            // futhark/microgpt.fut:399:57-135
            
            double zs_res_145316 = 1.0 / zt_res_145315;
            
            // futhark/microgpt.fut:399:39-135
            
            double zt_res_145317 = zt_lhs_145311 * zs_res_145316;
            
            // futhark/microgpt.fut:412:39-51
            
            double zt_lhs_145324 = ((double *) mem_154330)[i_151933];
            
            // futhark/microgpt.fut:412:93-105
            
            double zp_lhs_145325 = ((double *) mem_152413)[i_151933];
            
            // futhark/microgpt.fut:412:93-133
            
            double zp_res_145326 = 1.0e-5 + zp_lhs_145325;
            
            // futhark/microgpt.fut:412:85-133
            
            double sqrt_res_145327 = futrts_sqrt64(zp_res_145326);
            
            // futhark/microgpt.fut:412:71-135
            
            double zt_res_145328 = 2.0 * sqrt_res_145327;
            
            // futhark/microgpt.fut:412:57-135
            
            double zs_res_145329 = 1.0 / zt_res_145328;
            
            // futhark/microgpt.fut:412:39-135
            
            double zt_res_145330 = zt_lhs_145324 * zs_res_145329;
            
            ((double *) mem_154358)[i_151933] = zt_res_145330;
            ((double *) mem_154359)[i_151933] = zt_res_145317;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151947 = 0; i_151947 < (int64_t) 16; i_151947++) {
            // futhark/microgpt.fut:400:72-84
            
            double zs_rhs_145348 = ((double *) mem_154333)[i_151947];
            
            // futhark/microgpt.fut:400:64-84
            
            double zs_res_145349 = 1.0 / zs_rhs_145348;
            
            // futhark/microgpt.fut:400:94-106
            
            double zs_lhs_145350 = ((double *) mem_154359)[i_151947];
            
            // futhark/microgpt.fut:400:94-121
            
            double zs_res_145351 = zs_lhs_145350 / 16.0;
            
            // futhark/microgpt.fut:413:94-106
            
            double zs_lhs_145375 = ((double *) mem_154358)[i_151947];
            
            // futhark/microgpt.fut:413:94-121
            
            double zs_res_145376 = zs_lhs_145375 / 16.0;
            
            // futhark/microgpt.fut:413:72-84
            
            double zs_rhs_145373 = ((double *) mem_154331)[i_151947];
            
            // futhark/microgpt.fut:413:64-84
            
            double zs_res_145374 = 1.0 / zs_rhs_145373;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151940 = 0; i_151940 < (int64_t) 16; i_151940++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_150372 = ((double *) mem_154299)[i_151947 * (int64_t) 16 + i_151940];
                
                // futhark/microgpt.fut:400:38-84
                
                double zt_res_150373 = zs_res_145349 * zt_lhs_150372;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_150374 = ((double *) mem_param_152288.mem)[i_151947 * (int64_t) 16 + i_151940];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_150375 = ((double *) mem_152381)[i_151947 * (int64_t) 16 + i_151940];
                
                // futhark/microgpt.fut:400:128-172
                
                double zp_res_150376 = zp_lhs_150374 + zp_rhs_150375;
                
                // futhark/microgpt.fut:400:107-172
                
                double zt_res_150377 = zs_res_145351 * zp_res_150376;
                
                // futhark/microgpt.fut:400:123-259
                
                double zp_res_150378 = zt_res_150377 + zt_res_150377;
                
                // futhark/microgpt.fut:400:59-259
                
                double zp_res_150379 = zt_res_150373 + zp_res_150378;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_150386 = ((double *) mem_154298)[i_151947 * (int64_t) 16 + i_151940];
                
                // futhark/microgpt.fut:413:38-84
                
                double zt_res_150387 = zs_res_145374 * zt_lhs_150386;
                
                // futhark/microgpt.fut:413:107-172
                
                double zt_res_150391 = zs_res_145376 * zp_res_150376;
                
                // futhark/microgpt.fut:413:123-259
                
                double zp_res_150392 = zt_res_150391 + zt_res_150391;
                
                // futhark/microgpt.fut:413:59-259
                
                double zp_res_150393 = zt_res_150387 + zp_res_150392;
                
                ((double *) mem_154382)[i_151940] = zp_res_150393;
                ((double *) mem_154383)[i_151940] = zp_res_150379;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154372, i_151947 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154382, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154373, i_151947 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154383, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151956 = 0; i_151956 < (int64_t) 64; i_151956++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151952 = 0; i_151952 < (int64_t) 16; i_151952++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_140001;
                double r_140003 = 0.0;
                
                for (int64_t i_140002 = 0; i_140002 < (int64_t) 16; i_140002++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_140004 = ((double *) mem_153581)[i_140002 * (int64_t) 64 + i_151956];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_140005 = ((double *) mem_153295)[i_140002 * (int64_t) 16 + i_151952];
                    
                    // futhark/microgpt.fut:405:67-111
                    
                    double zt_res_140006 = zt_lhs_140004 * zt_rhs_140005;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_140007 = r_140003 + zt_res_140006;
                    double r_tmp_155127 = zp_res_140007;
                    
                    r_140003 = r_tmp_155127;
                }
                defunc_0_lifted_lambda_res_140001 = r_140003;
                ((double *) mem_154409)[i_151952] = defunc_0_lifted_lambda_res_140001;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154404, i_151956 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154409, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_151969 = 0; i_151969 < (int64_t) 27; i_151969++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_151962 = 0; i_151962 < (int64_t) 16; i_151962++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_150414;
                double r_150416 = 0.0;
                
                for (int64_t i_150415 = 0; i_150415 < (int64_t) 16; i_150415++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_150417 = ((double *) mem_153548)[i_150415 * (int64_t) 27 + i_151969];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_150418 = ((double *) mem_153343)[i_150415 * (int64_t) 16 + i_151962];
                    
                    // futhark/microgpt.fut:407:68-111
                    
                    double zt_res_150419 = zt_lhs_150417 * zt_rhs_150418;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_150420 = r_150416 + zt_res_150419;
                    double r_tmp_155132 = zp_res_150420;
                    
                    r_150416 = r_tmp_155132;
                }
                defunc_0_lifted_lambda_res_150414 = r_150416;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_150423;
                double r_150425 = 0.0;
                
                for (int64_t i_150424 = 0; i_150424 < (int64_t) 16; i_150424++) {
                    int64_t zeze_lhs_150426 = ((int64_t *) seqs_mem_152272.mem)[step_137468 * (int64_t) 16 + i_150424];
                    
                    // futhark/microgpt.fut:583:58-109
                    
                    bool cond_150427 = zeze_lhs_150426 == i_151969;
                    
                    // futhark/microgpt.fut:583:58-109
                    
                    double lifted_lambda_res_150428;
                    
                    if (cond_150427) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double lifted_lambda_res_t_res_150969 = ((double *) mem_154372)[i_150424 * (int64_t) 16 + i_151962];
                        
                        lifted_lambda_res_150428 = lifted_lambda_res_t_res_150969;
                    } else {
                        lifted_lambda_res_150428 = 0.0;
                    }
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_150434 = r_150425 + lifted_lambda_res_150428;
                    double r_tmp_155133 = zp_res_150434;
                    
                    r_150425 = r_tmp_155133;
                }
                defunc_0_lifted_lambda_res_150423 = r_150425;
                ((double *) mem_154430)[i_151962] = defunc_0_lifted_lambda_res_150423;
                ((double *) mem_154431)[i_151962] = defunc_0_lifted_lambda_res_150414;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154420, i_151969 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154430, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_154421, i_151969 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_154431, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:66:26-45
        
        double i64_res_140172 = sitofp_i64_f64(step_137468);
        
        // futhark/microgpt.fut:518:46-65
        
        double zm_rhs_140173 = i64_res_140172 / 500.0;
        
        // futhark/microgpt.fut:518:24-65
        
        double zt_rhs_140174 = 1.0 - zm_rhs_140173;
        
        // futhark/microgpt.fut:518:19-65
        
        double lt_r_140175 = 1.0e-2 * zt_rhs_140174;
        
        // futhark/microgpt.fut:520:5-52
        if (memblock_alloc(ctx, &mem_154452, (int64_t) 3456, "mem_154452")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:520:5-52
        // futhark/microgpt.fut:520:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154452.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152296.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:520:5-52
        if (memblock_alloc(ctx, &mem_154454, (int64_t) 3456, "mem_154454")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:520:5-52
        // futhark/microgpt.fut:520:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154454.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152332.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:520:5-52
        if (memblock_alloc(ctx, &mem_154456, (int64_t) 3456, "mem_154456")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:520:5-52
        // futhark/microgpt.fut:520:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154456.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152368.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:520:5-52
        if (memblock_alloc(ctx, &mem_154458, (int64_t) 3456, "mem_154458")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:520:5-52
        // futhark/microgpt.fut:520:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154458.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154420, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:520:5-52
        if (futrts_adam_opt_w_12952(ctx, &ext_mem_154462, &ext_mem_154461, &ext_mem_154460, mem_154452, mem_154454, mem_154456, mem_154458, (int64_t) 27, (int64_t) 16, step_137468, lt_r_140175) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_154452, "mem_154452") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154454, "mem_154454") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154456, "mem_154456") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154458, "mem_154458") != 0)
            return 1;
        // futhark/microgpt.fut:522:5-52
        if (memblock_alloc(ctx, &mem_154463, (int64_t) 2048, "mem_154463")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:522:5-52
        // futhark/microgpt.fut:522:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154463.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152288.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:522:5-52
        if (memblock_alloc(ctx, &mem_154465, (int64_t) 2048, "mem_154465")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:522:5-52
        // futhark/microgpt.fut:522:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154465.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152324.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:522:5-52
        if (memblock_alloc(ctx, &mem_154467, (int64_t) 2048, "mem_154467")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:522:5-52
        // futhark/microgpt.fut:522:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154467.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152360.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:522:5-52
        if (memblock_alloc(ctx, &mem_154469, (int64_t) 2048, "mem_154469")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:522:5-52
        // futhark/microgpt.fut:522:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154469.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154373, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:522:5-52
        if (futrts_adam_opt_w_12953(ctx, &ext_mem_154473, &ext_mem_154472, &ext_mem_154471, mem_154463, mem_154465, mem_154467, mem_154469, (int64_t) 16, (int64_t) 16, step_137468, lt_r_140175) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_154463, "mem_154463") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154465, "mem_154465") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154467, "mem_154467") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154469, "mem_154469") != 0)
            return 1;
        // futhark/microgpt.fut:524:5-56
        if (memblock_alloc(ctx, &mem_154474, (int64_t) 2048, "mem_154474")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:524:5-56
        // futhark/microgpt.fut:524:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154474.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152292.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:524:5-56
        if (memblock_alloc(ctx, &mem_154476, (int64_t) 2048, "mem_154476")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:524:5-56
        // futhark/microgpt.fut:524:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154476.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152328.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:524:5-56
        if (memblock_alloc(ctx, &mem_154478, (int64_t) 2048, "mem_154478")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:524:5-56
        // futhark/microgpt.fut:524:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154478.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152364.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:524:5-56
        if (memblock_alloc(ctx, &mem_154480, (int64_t) 2048, "mem_154480")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:524:5-56
        // futhark/microgpt.fut:524:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154480.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154215, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:524:5-56
        if (futrts_adam_opt_w_12953(ctx, &ext_mem_154484, &ext_mem_154483, &ext_mem_154482, mem_154474, mem_154476, mem_154478, mem_154480, (int64_t) 16, (int64_t) 16, step_137468, lt_r_140175) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_154474, "mem_154474") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154476, "mem_154476") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154478, "mem_154478") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154480, "mem_154480") != 0)
            return 1;
        // futhark/microgpt.fut:526:5-56
        if (memblock_alloc(ctx, &mem_154485, (int64_t) 2048, "mem_154485")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:526:5-56
        // futhark/microgpt.fut:526:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154485.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152280.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:526:5-56
        if (memblock_alloc(ctx, &mem_154487, (int64_t) 2048, "mem_154487")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:526:5-56
        // futhark/microgpt.fut:526:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154487.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152316.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:526:5-56
        if (memblock_alloc(ctx, &mem_154489, (int64_t) 2048, "mem_154489")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:526:5-56
        // futhark/microgpt.fut:526:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154489.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152352.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:526:5-56
        if (memblock_alloc(ctx, &mem_154491, (int64_t) 2048, "mem_154491")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:526:5-56
        // futhark/microgpt.fut:526:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154491.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154214, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:526:5-56
        if (futrts_adam_opt_w_12953(ctx, &ext_mem_154495, &ext_mem_154494, &ext_mem_154493, mem_154485, mem_154487, mem_154489, mem_154491, (int64_t) 16, (int64_t) 16, step_137468, lt_r_140175) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_154485, "mem_154485") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154487, "mem_154487") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154489, "mem_154489") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154491, "mem_154491") != 0)
            return 1;
        // futhark/microgpt.fut:528:5-56
        if (memblock_alloc(ctx, &mem_154496, (int64_t) 2048, "mem_154496")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:528:5-56
        // futhark/microgpt.fut:528:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154496.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152304.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:528:5-56
        if (memblock_alloc(ctx, &mem_154498, (int64_t) 2048, "mem_154498")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:528:5-56
        // futhark/microgpt.fut:528:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154498.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152340.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:528:5-56
        if (memblock_alloc(ctx, &mem_154500, (int64_t) 2048, "mem_154500")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:528:5-56
        // futhark/microgpt.fut:528:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154500.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152376.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:528:5-56
        if (memblock_alloc(ctx, &mem_154502, (int64_t) 2048, "mem_154502")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:528:5-56
        // futhark/microgpt.fut:528:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154502.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154213, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:528:5-56
        if (futrts_adam_opt_w_12953(ctx, &ext_mem_154506, &ext_mem_154505, &ext_mem_154504, mem_154496, mem_154498, mem_154500, mem_154502, (int64_t) 16, (int64_t) 16, step_137468, lt_r_140175) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_154496, "mem_154496") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154498, "mem_154498") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154500, "mem_154500") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154502, "mem_154502") != 0)
            return 1;
        // futhark/microgpt.fut:530:5-56
        if (memblock_alloc(ctx, &mem_154507, (int64_t) 2048, "mem_154507")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:530:5-56
        // futhark/microgpt.fut:530:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154507.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152284.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:530:5-56
        if (memblock_alloc(ctx, &mem_154509, (int64_t) 2048, "mem_154509")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:530:5-56
        // futhark/microgpt.fut:530:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154509.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152320.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:530:5-56
        if (memblock_alloc(ctx, &mem_154511, (int64_t) 2048, "mem_154511")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:530:5-56
        // futhark/microgpt.fut:530:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154511.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152356.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:530:5-56
        if (memblock_alloc(ctx, &mem_154513, (int64_t) 2048, "mem_154513")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:530:5-56
        // futhark/microgpt.fut:530:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154513.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154133, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:530:5-56
        if (futrts_adam_opt_w_12953(ctx, &ext_mem_154517, &ext_mem_154516, &ext_mem_154515, mem_154507, mem_154509, mem_154511, mem_154513, (int64_t) 16, (int64_t) 16, step_137468, lt_r_140175) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_154507, "mem_154507") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154509, "mem_154509") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154511, "mem_154511") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154513, "mem_154513") != 0)
            return 1;
        // futhark/microgpt.fut:532:5-52
        if (memblock_alloc(ctx, &mem_154518, (int64_t) 8192, "mem_154518")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:532:5-52
        // futhark/microgpt.fut:532:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154518.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152300.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:532:5-52
        if (memblock_alloc(ctx, &mem_154520, (int64_t) 8192, "mem_154520")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:532:5-52
        // futhark/microgpt.fut:532:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154520.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152336.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:532:5-52
        if (memblock_alloc(ctx, &mem_154522, (int64_t) 8192, "mem_154522")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:532:5-52
        // futhark/microgpt.fut:532:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154522.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152372.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:532:5-52
        if (memblock_alloc(ctx, &mem_154524, (int64_t) 8192, "mem_154524")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:532:5-52
        // futhark/microgpt.fut:532:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154524.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154404, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:532:5-52
        if (futrts_adam_opt_w_12952(ctx, &ext_mem_154528, &ext_mem_154527, &ext_mem_154526, mem_154518, mem_154520, mem_154522, mem_154524, (int64_t) 64, (int64_t) 16, step_137468, lt_r_140175) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_154518, "mem_154518") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154520, "mem_154520") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154522, "mem_154522") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154524, "mem_154524") != 0)
            return 1;
        // futhark/microgpt.fut:534:5-60
        if (memblock_alloc(ctx, &mem_154529, (int64_t) 8192, "mem_154529")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:534:5-60
        // futhark/microgpt.fut:534:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154529.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_152276.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:534:5-60
        if (memblock_alloc(ctx, &mem_154531, (int64_t) 8192, "mem_154531")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:534:5-60
        // futhark/microgpt.fut:534:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154531.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_152312.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:534:5-60
        if (memblock_alloc(ctx, &mem_154533, (int64_t) 8192, "mem_154533")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:534:5-60
        // futhark/microgpt.fut:534:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154533.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_152348.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:534:5-60
        if (memblock_alloc(ctx, &mem_154535, (int64_t) 8192, "mem_154535")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:534:5-60
        // futhark/microgpt.fut:534:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154535.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_153580, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:534:5-60
        if (futrts_adam_opt_w_12952(ctx, &ext_mem_154539, &ext_mem_154538, &ext_mem_154537, mem_154529, mem_154531, mem_154533, mem_154535, (int64_t) 16, (int64_t) 64, step_137468, lt_r_140175) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_154529, "mem_154529") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154531, "mem_154531") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154533, "mem_154533") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154535, "mem_154535") != 0)
            return 1;
        // futhark/microgpt.fut:536:5-56
        if (memblock_alloc(ctx, &mem_154540, (int64_t) 3456, "mem_154540")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:536:5-56
        // futhark/microgpt.fut:536:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154540.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152308.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:536:5-56
        if (memblock_alloc(ctx, &mem_154542, (int64_t) 3456, "mem_154542")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:536:5-56
        // futhark/microgpt.fut:536:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154542.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152344.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:536:5-56
        if (memblock_alloc(ctx, &mem_154544, (int64_t) 3456, "mem_154544")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:536:5-56
        // futhark/microgpt.fut:536:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154544.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_152380.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:536:5-56
        if (memblock_alloc(ctx, &mem_154546, (int64_t) 3456, "mem_154546")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:536:5-56
        // futhark/microgpt.fut:536:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_154546.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_154421, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:536:5-56
        if (futrts_adam_opt_w_12952(ctx, &ext_mem_154550, &ext_mem_154549, &ext_mem_154548, mem_154540, mem_154542, mem_154544, mem_154546, (int64_t) 27, (int64_t) 16, step_137468, lt_r_140175) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_154540, "mem_154540") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154542, "mem_154542") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154544, "mem_154544") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154546, "mem_154546") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154761, &ext_mem_154539, "ext_mem_154539") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154762, &ext_mem_154495, "ext_mem_154495") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154763, &ext_mem_154517, "ext_mem_154517") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154764, &ext_mem_154473, "ext_mem_154473") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154765, &ext_mem_154484, "ext_mem_154484") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154766, &ext_mem_154462, "ext_mem_154462") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154767, &ext_mem_154528, "ext_mem_154528") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154768, &ext_mem_154506, "ext_mem_154506") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154769, &ext_mem_154550, "ext_mem_154550") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154770, &ext_mem_154538, "ext_mem_154538") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154771, &ext_mem_154494, "ext_mem_154494") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154772, &ext_mem_154516, "ext_mem_154516") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154773, &ext_mem_154472, "ext_mem_154472") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154774, &ext_mem_154483, "ext_mem_154483") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154775, &ext_mem_154461, "ext_mem_154461") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154776, &ext_mem_154527, "ext_mem_154527") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154777, &ext_mem_154505, "ext_mem_154505") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154778, &ext_mem_154549, "ext_mem_154549") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154779, &ext_mem_154537, "ext_mem_154537") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154780, &ext_mem_154493, "ext_mem_154493") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154781, &ext_mem_154515, "ext_mem_154515") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154782, &ext_mem_154471, "ext_mem_154471") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154783, &ext_mem_154482, "ext_mem_154482") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154784, &ext_mem_154460, "ext_mem_154460") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154785, &ext_mem_154526, "ext_mem_154526") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154786, &ext_mem_154504, "ext_mem_154504") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_154787, &ext_mem_154548, "ext_mem_154548") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152276, &mem_param_tmp_154761, "mem_param_tmp_154761") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152280, &mem_param_tmp_154762, "mem_param_tmp_154762") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152284, &mem_param_tmp_154763, "mem_param_tmp_154763") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152288, &mem_param_tmp_154764, "mem_param_tmp_154764") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152292, &mem_param_tmp_154765, "mem_param_tmp_154765") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152296, &mem_param_tmp_154766, "mem_param_tmp_154766") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152300, &mem_param_tmp_154767, "mem_param_tmp_154767") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152304, &mem_param_tmp_154768, "mem_param_tmp_154768") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152308, &mem_param_tmp_154769, "mem_param_tmp_154769") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152312, &mem_param_tmp_154770, "mem_param_tmp_154770") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152316, &mem_param_tmp_154771, "mem_param_tmp_154771") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152320, &mem_param_tmp_154772, "mem_param_tmp_154772") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152324, &mem_param_tmp_154773, "mem_param_tmp_154773") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152328, &mem_param_tmp_154774, "mem_param_tmp_154774") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152332, &mem_param_tmp_154775, "mem_param_tmp_154775") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152336, &mem_param_tmp_154776, "mem_param_tmp_154776") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152340, &mem_param_tmp_154777, "mem_param_tmp_154777") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152344, &mem_param_tmp_154778, "mem_param_tmp_154778") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152348, &mem_param_tmp_154779, "mem_param_tmp_154779") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152352, &mem_param_tmp_154780, "mem_param_tmp_154780") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152356, &mem_param_tmp_154781, "mem_param_tmp_154781") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152360, &mem_param_tmp_154782, "mem_param_tmp_154782") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152364, &mem_param_tmp_154783, "mem_param_tmp_154783") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152368, &mem_param_tmp_154784, "mem_param_tmp_154784") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152372, &mem_param_tmp_154785, "mem_param_tmp_154785") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152376, &mem_param_tmp_154786, "mem_param_tmp_154786") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_152380, &mem_param_tmp_154787, "mem_param_tmp_154787") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_154658, &mem_param_152276, "mem_param_152276") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154657, &mem_param_152280, "mem_param_152280") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154656, &mem_param_152284, "mem_param_152284") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154655, &mem_param_152288, "mem_param_152288") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154654, &mem_param_152292, "mem_param_152292") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154653, &mem_param_152296, "mem_param_152296") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154652, &mem_param_152300, "mem_param_152300") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154651, &mem_param_152304, "mem_param_152304") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154650, &mem_param_152308, "mem_param_152308") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154649, &mem_param_152312, "mem_param_152312") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154648, &mem_param_152316, "mem_param_152316") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154647, &mem_param_152320, "mem_param_152320") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154646, &mem_param_152324, "mem_param_152324") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154645, &mem_param_152328, "mem_param_152328") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154644, &mem_param_152332, "mem_param_152332") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154643, &mem_param_152336, "mem_param_152336") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154642, &mem_param_152340, "mem_param_152340") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154641, &mem_param_152344, "mem_param_152344") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154640, &mem_param_152348, "mem_param_152348") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154639, &mem_param_152352, "mem_param_152352") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154638, &mem_param_152356, "mem_param_152356") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154637, &mem_param_152360, "mem_param_152360") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154636, &mem_param_152364, "mem_param_152364") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154635, &mem_param_152368, "mem_param_152368") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154634, &mem_param_152372, "mem_param_152372") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154633, &mem_param_152376, "mem_param_152376") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_154632, &mem_param_152380, "mem_param_152380") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154731, &ext_mem_154653, "ext_mem_154653") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154732, &ext_mem_154655, "ext_mem_154655") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154733, &ext_mem_154654, "ext_mem_154654") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154734, &ext_mem_154657, "ext_mem_154657") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154735, &ext_mem_154651, "ext_mem_154651") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154736, &ext_mem_154656, "ext_mem_154656") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154737, &ext_mem_154652, "ext_mem_154652") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154738, &ext_mem_154658, "ext_mem_154658") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154739, &ext_mem_154650, "ext_mem_154650") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154740, &ext_mem_154644, "ext_mem_154644") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154741, &ext_mem_154646, "ext_mem_154646") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154742, &ext_mem_154645, "ext_mem_154645") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154743, &ext_mem_154648, "ext_mem_154648") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154744, &ext_mem_154642, "ext_mem_154642") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154745, &ext_mem_154647, "ext_mem_154647") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154746, &ext_mem_154643, "ext_mem_154643") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154747, &ext_mem_154649, "ext_mem_154649") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154748, &ext_mem_154641, "ext_mem_154641") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154749, &ext_mem_154635, "ext_mem_154635") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154750, &ext_mem_154637, "ext_mem_154637") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154751, &ext_mem_154636, "ext_mem_154636") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154752, &ext_mem_154639, "ext_mem_154639") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154753, &ext_mem_154633, "ext_mem_154633") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154754, &ext_mem_154638, "ext_mem_154638") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154755, &ext_mem_154634, "ext_mem_154634") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154756, &ext_mem_154640, "ext_mem_154640") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154757, &ext_mem_154632, "ext_mem_154632") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155287, &mem_out_154731, "mem_out_154731") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155288, &mem_out_154732, "mem_out_154732") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155289, &mem_out_154733, "mem_out_154733") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155290, &mem_out_154734, "mem_out_154734") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155291, &mem_out_154735, "mem_out_154735") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155292, &mem_out_154736, "mem_out_154736") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155293, &mem_out_154737, "mem_out_154737") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155294, &mem_out_154738, "mem_out_154738") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155295, &mem_out_154739, "mem_out_154739") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155296, &mem_out_154740, "mem_out_154740") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155297, &mem_out_154741, "mem_out_154741") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155298, &mem_out_154742, "mem_out_154742") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155299, &mem_out_154743, "mem_out_154743") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155300, &mem_out_154744, "mem_out_154744") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155301, &mem_out_154745, "mem_out_154745") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155302, &mem_out_154746, "mem_out_154746") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155303, &mem_out_154747, "mem_out_154747") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155304, &mem_out_154748, "mem_out_154748") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155305, &mem_out_154749, "mem_out_154749") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155306, &mem_out_154750, "mem_out_154750") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155307, &mem_out_154751, "mem_out_154751") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155308, &mem_out_154752, "mem_out_154752") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155309, &mem_out_154753, "mem_out_154753") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155310, &mem_out_154754, "mem_out_154754") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155311, &mem_out_154755, "mem_out_154755") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155312, &mem_out_154756, "mem_out_154756") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155313, &mem_out_154757, "mem_out_154757") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_152381);
        free(mem_152382);
        free(mem_152391);
        free(mem_152398);
        free(mem_152413);
        free(mem_152414);
        free(mem_152415);
        free(mem_152434);
        free(mem_152441);
        free(mem_152446);
        free(mem_152457);
        free(mem_152462);
        free(mem_152473);
        free(mem_152474);
        free(mem_152487);
        free(mem_152494);
        free(mem_152499);
        free(mem_152510);
        free(mem_152515);
        free(mem_152526);
        free(mem_152527);
        free(mem_152528);
        free(mem_152544);
        free(mem_152545);
        free(mem_152546);
        free(mem_152559);
        free(mem_152560);
        free(mem_152561);
        free(mem_152607);
        free(mem_152608);
        free(mem_152609);
        free(mem_152610);
        free(mem_152631);
        free(mem_152632);
        free(mem_152633);
        free(mem_152634);
        free(mem_152651);
        free(mem_152652);
        free(mem_152653);
        free(mem_152654);
        free(mem_152715);
        free(mem_152716);
        free(mem_152717);
        free(mem_152718);
        free(mem_152739);
        free(mem_152740);
        free(mem_152741);
        free(mem_152742);
        free(mem_152759);
        free(mem_152760);
        free(mem_152761);
        free(mem_152762);
        free(mem_152823);
        free(mem_152824);
        free(mem_152825);
        free(mem_152826);
        free(mem_152827);
        free(mem_152828);
        free(mem_152829);
        free(mem_152830);
        free(mem_152863);
        free(mem_152864);
        free(mem_152865);
        free(mem_152866);
        free(mem_152867);
        free(mem_152868);
        free(mem_152869);
        free(mem_152870);
        free(mem_152951);
        free(mem_152952);
        free(mem_152953);
        free(mem_152954);
        free(mem_152975);
        free(mem_152976);
        free(mem_152977);
        free(mem_152978);
        free(mem_152995);
        free(mem_152996);
        free(mem_152997);
        free(mem_152998);
        free(mem_153059);
        free(mem_153060);
        free(mem_153069);
        free(mem_153070);
        free(mem_153091);
        free(mem_153092);
        free(mem_153103);
        free(mem_153104);
        free(mem_153113);
        free(mem_153114);
        free(mem_153145);
        free(mem_153146);
        free(mem_153157);
        free(mem_153158);
        free(mem_153167);
        free(mem_153168);
        free(mem_153199);
        free(mem_153205);
        free(mem_153210);
        free(mem_153226);
        free(mem_153231);
        free(mem_153242);
        free(mem_153247);
        free(mem_153258);
        free(mem_153259);
        free(mem_153272);
        free(mem_153279);
        free(mem_153284);
        free(mem_153295);
        free(mem_153300);
        free(mem_153311);
        free(mem_153316);
        free(mem_153327);
        free(mem_153332);
        free(mem_153343);
        free(mem_153348);
        free(mem_153359);
        free(mem_153364);
        free(mem_153375);
        free(mem_153376);
        free(mem_153377);
        free(mem_153378);
        free(mem_153396);
        free(mem_153401);
        free(mem_153405);
        free(mem_153412);
        free(mem_153446);
        free(mem_153452);
        free(mem_153457);
        free(mem_153473);
        free(mem_153474);
        free(mem_153483);
        free(mem_153484);
        free(mem_153505);
        free(mem_153511);
        free(mem_153516);
        free(mem_153532);
        free(mem_153537);
        free(mem_153548);
        free(mem_153553);
        free(mem_153564);
        free(mem_153569);
        free(mem_153580);
        free(mem_153581);
        free(mem_153590);
        free(mem_153591);
        free(mem_153612);
        free(mem_153617);
        free(mem_153628);
        free(mem_153629);
        free(mem_153642);
        free(mem_153649);
        free(mem_153654);
        free(mem_153665);
        free(mem_153671);
        free(mem_153676);
        free(mem_153692);
        free(mem_153693);
        free(mem_153694);
        free(mem_153710);
        free(mem_153711);
        free(mem_153712);
        free(mem_153725);
        free(mem_153726);
        free(mem_153767);
        free(mem_153768);
        free(mem_153779);
        free(mem_153780);
        free(mem_153789);
        free(mem_153790);
        free(mem_153821);
        free(mem_153822);
        free(mem_153833);
        free(mem_153834);
        free(mem_153843);
        free(mem_153844);
        free(mem_153875);
        free(mem_153876);
        free(mem_153877);
        free(mem_153878);
        free(mem_153895);
        free(mem_153896);
        free(mem_153897);
        free(mem_153898);
        free(mem_153939);
        free(mem_153940);
        free(mem_153951);
        free(mem_153952);
        free(mem_153961);
        free(mem_153962);
        free(mem_153993);
        free(mem_153994);
        free(mem_154003);
        free(mem_154004);
        free(mem_154025);
        free(mem_154026);
        free(mem_154037);
        free(mem_154038);
        free(mem_154047);
        free(mem_154048);
        free(mem_154079);
        free(mem_154080);
        free(mem_154091);
        free(mem_154092);
        free(mem_154101);
        free(mem_154102);
        free(mem_154133);
        free(mem_154134);
        free(mem_154135);
        free(mem_154136);
        free(mem_154153);
        free(mem_154154);
        free(mem_154155);
        free(mem_154156);
        free(mem_154197);
        free(mem_154202);
        free(mem_154213);
        free(mem_154214);
        free(mem_154215);
        free(mem_154216);
        free(mem_154217);
        free(mem_154236);
        free(mem_154237);
        free(mem_154238);
        free(mem_154275);
        free(mem_154282);
        free(mem_154287);
        free(mem_154298);
        free(mem_154299);
        free(mem_154308);
        free(mem_154309);
        free(mem_154330);
        free(mem_154331);
        free(mem_154332);
        free(mem_154333);
        free(mem_154358);
        free(mem_154359);
        free(mem_154372);
        free(mem_154373);
        free(mem_154382);
        free(mem_154383);
        free(mem_154404);
        free(mem_154409);
        free(mem_154420);
        free(mem_154421);
        free(mem_154430);
        free(mem_154431);
        if (memblock_unref(ctx, &mem_param_tmp_154787, "mem_param_tmp_154787") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154786, "mem_param_tmp_154786") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154785, "mem_param_tmp_154785") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154784, "mem_param_tmp_154784") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154783, "mem_param_tmp_154783") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154782, "mem_param_tmp_154782") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154781, "mem_param_tmp_154781") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154780, "mem_param_tmp_154780") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154779, "mem_param_tmp_154779") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154778, "mem_param_tmp_154778") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154777, "mem_param_tmp_154777") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154776, "mem_param_tmp_154776") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154775, "mem_param_tmp_154775") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154774, "mem_param_tmp_154774") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154773, "mem_param_tmp_154773") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154772, "mem_param_tmp_154772") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154771, "mem_param_tmp_154771") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154770, "mem_param_tmp_154770") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154769, "mem_param_tmp_154769") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154768, "mem_param_tmp_154768") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154767, "mem_param_tmp_154767") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154766, "mem_param_tmp_154766") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154765, "mem_param_tmp_154765") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154764, "mem_param_tmp_154764") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154763, "mem_param_tmp_154763") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154762, "mem_param_tmp_154762") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_154761, "mem_param_tmp_154761") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154548, "ext_mem_154548") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154549, "ext_mem_154549") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154550, "ext_mem_154550") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154546, "mem_154546") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154544, "mem_154544") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154542, "mem_154542") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154540, "mem_154540") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154537, "ext_mem_154537") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154538, "ext_mem_154538") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154539, "ext_mem_154539") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154535, "mem_154535") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154533, "mem_154533") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154531, "mem_154531") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154529, "mem_154529") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154526, "ext_mem_154526") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154527, "ext_mem_154527") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154528, "ext_mem_154528") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154524, "mem_154524") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154522, "mem_154522") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154520, "mem_154520") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154518, "mem_154518") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154515, "ext_mem_154515") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154516, "ext_mem_154516") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154517, "ext_mem_154517") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154513, "mem_154513") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154511, "mem_154511") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154509, "mem_154509") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154507, "mem_154507") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154504, "ext_mem_154504") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154505, "ext_mem_154505") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154506, "ext_mem_154506") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154502, "mem_154502") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154500, "mem_154500") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154498, "mem_154498") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154496, "mem_154496") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154493, "ext_mem_154493") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154494, "ext_mem_154494") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154495, "ext_mem_154495") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154491, "mem_154491") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154489, "mem_154489") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154487, "mem_154487") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154485, "mem_154485") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154482, "ext_mem_154482") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154483, "ext_mem_154483") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154484, "ext_mem_154484") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154480, "mem_154480") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154478, "mem_154478") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154476, "mem_154476") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154474, "mem_154474") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154471, "ext_mem_154471") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154472, "ext_mem_154472") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154473, "ext_mem_154473") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154469, "mem_154469") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154467, "mem_154467") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154465, "mem_154465") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154463, "mem_154463") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154460, "ext_mem_154460") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154461, "ext_mem_154461") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154462, "ext_mem_154462") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154458, "mem_154458") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154456, "mem_154456") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154454, "mem_154454") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_154452, "mem_154452") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152380, "mem_param_152380") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152376, "mem_param_152376") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152372, "mem_param_152372") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152368, "mem_param_152368") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152364, "mem_param_152364") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152360, "mem_param_152360") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152356, "mem_param_152356") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152352, "mem_param_152352") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152348, "mem_param_152348") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152344, "mem_param_152344") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152340, "mem_param_152340") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152336, "mem_param_152336") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152332, "mem_param_152332") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152328, "mem_param_152328") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152324, "mem_param_152324") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152320, "mem_param_152320") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152316, "mem_param_152316") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152312, "mem_param_152312") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152308, "mem_param_152308") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152304, "mem_param_152304") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152300, "mem_param_152300") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152296, "mem_param_152296") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152292, "mem_param_152292") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152288, "mem_param_152288") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152284, "mem_param_152284") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152280, "mem_param_152280") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_152276, "mem_param_152276") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154632, "ext_mem_154632") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154633, "ext_mem_154633") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154634, "ext_mem_154634") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154635, "ext_mem_154635") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154636, "ext_mem_154636") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154637, "ext_mem_154637") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154638, "ext_mem_154638") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154639, "ext_mem_154639") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154640, "ext_mem_154640") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154641, "ext_mem_154641") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154642, "ext_mem_154642") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154643, "ext_mem_154643") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154644, "ext_mem_154644") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154645, "ext_mem_154645") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154646, "ext_mem_154646") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154647, "ext_mem_154647") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154648, "ext_mem_154648") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154649, "ext_mem_154649") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154650, "ext_mem_154650") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154651, "ext_mem_154651") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154652, "ext_mem_154652") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154653, "ext_mem_154653") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154654, "ext_mem_154654") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154655, "ext_mem_154655") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154656, "ext_mem_154656") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154657, "ext_mem_154657") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_154658, "ext_mem_154658") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154757, "mem_out_154757") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154756, "mem_out_154756") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154755, "mem_out_154755") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154754, "mem_out_154754") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154753, "mem_out_154753") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154752, "mem_out_154752") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154751, "mem_out_154751") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154750, "mem_out_154750") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154749, "mem_out_154749") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154748, "mem_out_154748") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154747, "mem_out_154747") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154746, "mem_out_154746") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154745, "mem_out_154745") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154744, "mem_out_154744") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154743, "mem_out_154743") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154742, "mem_out_154742") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154741, "mem_out_154741") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154740, "mem_out_154740") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154739, "mem_out_154739") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154738, "mem_out_154738") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154737, "mem_out_154737") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154736, "mem_out_154736") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154735, "mem_out_154735") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154734, "mem_out_154734") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154733, "mem_out_154733") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154732, "mem_out_154732") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154731, "mem_out_154731") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_155561, struct memblock *mem_out_p_155562, struct memblock *mem_out_p_155563, struct memblock *mem_out_p_155564, struct memblock *mem_out_p_155565, struct memblock *mem_out_p_155566, struct memblock *mem_out_p_155567, struct memblock *mem_out_p_155568, struct memblock *mem_out_p_155569)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_154739;
    
    mem_out_154739.references = NULL;
    
    struct memblock mem_out_154738;
    
    mem_out_154738.references = NULL;
    
    struct memblock mem_out_154737;
    
    mem_out_154737.references = NULL;
    
    struct memblock mem_out_154736;
    
    mem_out_154736.references = NULL;
    
    struct memblock mem_out_154735;
    
    mem_out_154735.references = NULL;
    
    struct memblock mem_out_154734;
    
    mem_out_154734.references = NULL;
    
    struct memblock mem_out_154733;
    
    mem_out_154733.references = NULL;
    
    struct memblock mem_out_154732;
    
    mem_out_154732.references = NULL;
    
    struct memblock mem_out_154731;
    
    mem_out_154731.references = NULL;
    
    struct memblock mem_152234 = ctx->constants->mem_152234;
    struct memblock mem_152235 = ctx->constants->mem_152235;
    struct memblock mem_152236 = ctx->constants->mem_152236;
    struct memblock mem_152237 = ctx->constants->mem_152237;
    struct memblock mem_152238 = ctx->constants->mem_152238;
    struct memblock mem_152239 = ctx->constants->mem_152239;
    struct memblock mem_152240 = ctx->constants->mem_152240;
    struct memblock mem_152241 = ctx->constants->mem_152241;
    struct memblock mem_152242 = ctx->constants->mem_152242;
    
    if (memblock_set(ctx, &mem_out_154731, &mem_152241, "mem_152241") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154732, &mem_152237, "mem_152237") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154733, &mem_152239, "mem_152239") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154734, &mem_152235, "mem_152235") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154735, &mem_152236, "mem_152236") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154736, &mem_152234, "mem_152234") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154737, &mem_152240, "mem_152240") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154738, &mem_152238, "mem_152238") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_154739, &mem_152242, "mem_152242") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155561, &mem_out_154731, "mem_out_154731") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155562, &mem_out_154732, "mem_out_154732") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155563, &mem_out_154733, "mem_out_154733") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155564, &mem_out_154734, "mem_out_154734") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155565, &mem_out_154735, "mem_out_154735") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155566, &mem_out_154736, "mem_out_154736") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155567, &mem_out_154737, "mem_out_154737") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155568, &mem_out_154738, "mem_out_154738") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_155569, &mem_out_154739, "mem_out_154739") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_154739, "mem_out_154739") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154738, "mem_out_154738") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154737, "mem_out_154737") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154736, "mem_out_154736") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154735, "mem_out_154735") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154734, "mem_out_154734") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154733, "mem_out_154733") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154732, "mem_out_154732") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_154731, "mem_out_154731") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_cal_loss(struct futhark_context *ctx, struct futhark_opaque_tup2_f64_arr1d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3)
{
    double prim_out_154732 = 0.0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_154731;
    
    mem_out_154731.references = NULL;
    
    struct memblock mask_mem_152254;
    
    mask_mem_152254.references = NULL;
    
    struct memblock target_mem_152253;
    
    target_mem_152253.references = NULL;
    
    struct memblock tokens_mem_152252;
    
    tokens_mem_152252.references = NULL;
    
    struct memblock wvoc_mem_152251;
    
    wvoc_mem_152251.references = NULL;
    
    struct memblock wval_mem_152250;
    
    wval_mem_152250.references = NULL;
    
    struct memblock wup_mem_152249;
    
    wup_mem_152249.references = NULL;
    
    struct memblock wte_mem_152248;
    
    wte_mem_152248.references = NULL;
    
    struct memblock wqry_mem_152247;
    
    wqry_mem_152247.references = NULL;
    
    struct memblock wpe_mem_152246;
    
    wpe_mem_152246.references = NULL;
    
    struct memblock wout_mem_152245;
    
    wout_mem_152245.references = NULL;
    
    struct memblock wkey_mem_152244;
    
    wkey_mem_152244.references = NULL;
    
    struct memblock wdown_mem_152243;
    
    wdown_mem_152243.references = NULL;
    wdown_mem_152243 = in0->v0->mem;
    wkey_mem_152244 = in0->v1->mem;
    wout_mem_152245 = in0->v2->mem;
    wpe_mem_152246 = in0->v3->mem;
    wqry_mem_152247 = in0->v4->mem;
    wte_mem_152248 = in0->v5->mem;
    wup_mem_152249 = in0->v6->mem;
    wval_mem_152250 = in0->v7->mem;
    wvoc_mem_152251 = in0->v8->mem;
    tokens_mem_152252 = in1->mem;
    target_mem_152253 = in2->mem;
    mask_mem_152254 = in3->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && (((int64_t) 16 == in2->shape[0] && (int64_t) 27 == in2->shape[1]) && ((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_cal_loss(ctx, &mem_out_154731, &prim_out_154732, wdown_mem_152243, wkey_mem_152244, wout_mem_152245, wpe_mem_152246, wqry_mem_152247, wte_mem_152248, wup_mem_152249, wval_mem_152250, wvoc_mem_152251, tokens_mem_152252, target_mem_152253, mask_mem_152254);
        if (ret == 0) {
            struct memblock mem_152234 = ctx->constants->mem_152234;
            struct memblock mem_152235 = ctx->constants->mem_152235;
            struct memblock mem_152236 = ctx->constants->mem_152236;
            struct memblock mem_152237 = ctx->constants->mem_152237;
            struct memblock mem_152238 = ctx->constants->mem_152238;
            struct memblock mem_152239 = ctx->constants->mem_152239;
            struct memblock mem_152240 = ctx->constants->mem_152240;
            struct memblock mem_152241 = ctx->constants->mem_152241;
            struct memblock mem_152242 = ctx->constants->mem_152242;
            
            assert((*out = (struct futhark_opaque_tup2_f64_arr1d_f64 *) malloc(sizeof(struct futhark_opaque_tup2_f64_arr1d_f64))) != NULL);
            (*out)->v0 = prim_out_154732;
            assert(((*out)->v1 = (struct futhark_f64_1d *) malloc(sizeof(struct futhark_f64_1d))) != NULL);
            (*out)->v1->mem = mem_out_154731;
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
    
    struct memblock mem_out_154731;
    
    mem_out_154731.references = NULL;
    
    struct memblock mask_mem_152253;
    
    mask_mem_152253.references = NULL;
    
    struct memblock tokens_mem_152252;
    
    tokens_mem_152252.references = NULL;
    
    struct memblock wvoc_mem_152251;
    
    wvoc_mem_152251.references = NULL;
    
    struct memblock wval_mem_152250;
    
    wval_mem_152250.references = NULL;
    
    struct memblock wup_mem_152249;
    
    wup_mem_152249.references = NULL;
    
    struct memblock wte_mem_152248;
    
    wte_mem_152248.references = NULL;
    
    struct memblock wqry_mem_152247;
    
    wqry_mem_152247.references = NULL;
    
    struct memblock wpe_mem_152246;
    
    wpe_mem_152246.references = NULL;
    
    struct memblock wout_mem_152245;
    
    wout_mem_152245.references = NULL;
    
    struct memblock wkey_mem_152244;
    
    wkey_mem_152244.references = NULL;
    
    struct memblock wdown_mem_152243;
    
    wdown_mem_152243.references = NULL;
    wdown_mem_152243 = in0->v0->mem;
    wkey_mem_152244 = in0->v1->mem;
    wout_mem_152245 = in0->v2->mem;
    wpe_mem_152246 = in0->v3->mem;
    wqry_mem_152247 = in0->v4->mem;
    wte_mem_152248 = in0->v5->mem;
    wup_mem_152249 = in0->v6->mem;
    wval_mem_152250 = in0->v7->mem;
    wvoc_mem_152251 = in0->v8->mem;
    tokens_mem_152252 = in1->mem;
    mask_mem_152253 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_154731, wdown_mem_152243, wkey_mem_152244, wout_mem_152245, wpe_mem_152246, wqry_mem_152247, wte_mem_152248, wup_mem_152249, wval_mem_152250, wvoc_mem_152251, tokens_mem_152252, mask_mem_152253);
        if (ret == 0) {
            struct memblock mem_152234 = ctx->constants->mem_152234;
            struct memblock mem_152235 = ctx->constants->mem_152235;
            struct memblock mem_152236 = ctx->constants->mem_152236;
            struct memblock mem_152237 = ctx->constants->mem_152237;
            struct memblock mem_152238 = ctx->constants->mem_152238;
            struct memblock mem_152239 = ctx->constants->mem_152239;
            struct memblock mem_152240 = ctx->constants->mem_152240;
            struct memblock mem_152241 = ctx->constants->mem_152241;
            struct memblock mem_152242 = ctx->constants->mem_152242;
            
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_154731;
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
    
    struct memblock mem_out_154739;
    
    mem_out_154739.references = NULL;
    
    struct memblock mem_out_154738;
    
    mem_out_154738.references = NULL;
    
    struct memblock mem_out_154737;
    
    mem_out_154737.references = NULL;
    
    struct memblock mem_out_154736;
    
    mem_out_154736.references = NULL;
    
    struct memblock mem_out_154735;
    
    mem_out_154735.references = NULL;
    
    struct memblock mem_out_154734;
    
    mem_out_154734.references = NULL;
    
    struct memblock mem_out_154733;
    
    mem_out_154733.references = NULL;
    
    struct memblock mem_out_154732;
    
    mem_out_154732.references = NULL;
    
    struct memblock mem_out_154731;
    
    mem_out_154731.references = NULL;
    
    struct memblock wvoc_mem_152251;
    
    wvoc_mem_152251.references = NULL;
    
    struct memblock wdown_mem_152250;
    
    wdown_mem_152250.references = NULL;
    
    struct memblock wup_mem_152249;
    
    wup_mem_152249.references = NULL;
    
    struct memblock wout_mem_152248;
    
    wout_mem_152248.references = NULL;
    
    struct memblock wval_mem_152247;
    
    wval_mem_152247.references = NULL;
    
    struct memblock wkey_mem_152246;
    
    wkey_mem_152246.references = NULL;
    
    struct memblock wqry_mem_152245;
    
    wqry_mem_152245.references = NULL;
    
    struct memblock wpe_mem_152244;
    
    wpe_mem_152244.references = NULL;
    
    struct memblock wte_mem_152243;
    
    wte_mem_152243.references = NULL;
    wte_mem_152243 = in0->mem;
    wpe_mem_152244 = in1->mem;
    wqry_mem_152245 = in2->mem;
    wkey_mem_152246 = in3->mem;
    wval_mem_152247 = in4->mem;
    wout_mem_152248 = in5->mem;
    wup_mem_152249 = in6->mem;
    wdown_mem_152250 = in7->mem;
    wvoc_mem_152251 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_154731, &mem_out_154732, &mem_out_154733, &mem_out_154734, &mem_out_154735, &mem_out_154736, &mem_out_154737, &mem_out_154738, &mem_out_154739, wte_mem_152243, wpe_mem_152244, wqry_mem_152245, wkey_mem_152246, wval_mem_152247, wout_mem_152248, wup_mem_152249, wdown_mem_152250, wvoc_mem_152251);
        if (ret == 0) {
            struct memblock mem_152234 = ctx->constants->mem_152234;
            struct memblock mem_152235 = ctx->constants->mem_152235;
            struct memblock mem_152236 = ctx->constants->mem_152236;
            struct memblock mem_152237 = ctx->constants->mem_152237;
            struct memblock mem_152238 = ctx->constants->mem_152238;
            struct memblock mem_152239 = ctx->constants->mem_152239;
            struct memblock mem_152240 = ctx->constants->mem_152240;
            struct memblock mem_152241 = ctx->constants->mem_152241;
            struct memblock mem_152242 = ctx->constants->mem_152242;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_154731;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_154732;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_154733;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_154734;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_154735;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_154736;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_154737;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_154738;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_154739;
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
    
    struct memblock mem_out_154757;
    
    mem_out_154757.references = NULL;
    
    struct memblock mem_out_154756;
    
    mem_out_154756.references = NULL;
    
    struct memblock mem_out_154755;
    
    mem_out_154755.references = NULL;
    
    struct memblock mem_out_154754;
    
    mem_out_154754.references = NULL;
    
    struct memblock mem_out_154753;
    
    mem_out_154753.references = NULL;
    
    struct memblock mem_out_154752;
    
    mem_out_154752.references = NULL;
    
    struct memblock mem_out_154751;
    
    mem_out_154751.references = NULL;
    
    struct memblock mem_out_154750;
    
    mem_out_154750.references = NULL;
    
    struct memblock mem_out_154749;
    
    mem_out_154749.references = NULL;
    
    struct memblock mem_out_154748;
    
    mem_out_154748.references = NULL;
    
    struct memblock mem_out_154747;
    
    mem_out_154747.references = NULL;
    
    struct memblock mem_out_154746;
    
    mem_out_154746.references = NULL;
    
    struct memblock mem_out_154745;
    
    mem_out_154745.references = NULL;
    
    struct memblock mem_out_154744;
    
    mem_out_154744.references = NULL;
    
    struct memblock mem_out_154743;
    
    mem_out_154743.references = NULL;
    
    struct memblock mem_out_154742;
    
    mem_out_154742.references = NULL;
    
    struct memblock mem_out_154741;
    
    mem_out_154741.references = NULL;
    
    struct memblock mem_out_154740;
    
    mem_out_154740.references = NULL;
    
    struct memblock mem_out_154739;
    
    mem_out_154739.references = NULL;
    
    struct memblock mem_out_154738;
    
    mem_out_154738.references = NULL;
    
    struct memblock mem_out_154737;
    
    mem_out_154737.references = NULL;
    
    struct memblock mem_out_154736;
    
    mem_out_154736.references = NULL;
    
    struct memblock mem_out_154735;
    
    mem_out_154735.references = NULL;
    
    struct memblock mem_out_154734;
    
    mem_out_154734.references = NULL;
    
    struct memblock mem_out_154733;
    
    mem_out_154733.references = NULL;
    
    struct memblock mem_out_154732;
    
    mem_out_154732.references = NULL;
    
    struct memblock mem_out_154731;
    
    mem_out_154731.references = NULL;
    
    struct memblock seqs_mem_152272;
    
    seqs_mem_152272.references = NULL;
    
    struct memblock dls_mem_152271;
    
    dls_mem_152271.references = NULL;
    
    struct memblock masks_mem_152270;
    
    masks_mem_152270.references = NULL;
    
    struct memblock wvoc_mem_152269;
    
    wvoc_mem_152269.references = NULL;
    
    struct memblock wval_mem_152268;
    
    wval_mem_152268.references = NULL;
    
    struct memblock wup_mem_152267;
    
    wup_mem_152267.references = NULL;
    
    struct memblock wte_mem_152266;
    
    wte_mem_152266.references = NULL;
    
    struct memblock wqry_mem_152265;
    
    wqry_mem_152265.references = NULL;
    
    struct memblock wpe_mem_152264;
    
    wpe_mem_152264.references = NULL;
    
    struct memblock wout_mem_152263;
    
    wout_mem_152263.references = NULL;
    
    struct memblock wkey_mem_152262;
    
    wkey_mem_152262.references = NULL;
    
    struct memblock wdown_mem_152261;
    
    wdown_mem_152261.references = NULL;
    
    struct memblock wvoc_mem_152260;
    
    wvoc_mem_152260.references = NULL;
    
    struct memblock wval_mem_152259;
    
    wval_mem_152259.references = NULL;
    
    struct memblock wup_mem_152258;
    
    wup_mem_152258.references = NULL;
    
    struct memblock wte_mem_152257;
    
    wte_mem_152257.references = NULL;
    
    struct memblock wqry_mem_152256;
    
    wqry_mem_152256.references = NULL;
    
    struct memblock wpe_mem_152255;
    
    wpe_mem_152255.references = NULL;
    
    struct memblock wout_mem_152254;
    
    wout_mem_152254.references = NULL;
    
    struct memblock wkey_mem_152253;
    
    wkey_mem_152253.references = NULL;
    
    struct memblock wdown_mem_152252;
    
    wdown_mem_152252.references = NULL;
    
    struct memblock wvoc_mem_152251;
    
    wvoc_mem_152251.references = NULL;
    
    struct memblock wval_mem_152250;
    
    wval_mem_152250.references = NULL;
    
    struct memblock wup_mem_152249;
    
    wup_mem_152249.references = NULL;
    
    struct memblock wte_mem_152248;
    
    wte_mem_152248.references = NULL;
    
    struct memblock wqry_mem_152247;
    
    wqry_mem_152247.references = NULL;
    
    struct memblock wpe_mem_152246;
    
    wpe_mem_152246.references = NULL;
    
    struct memblock wout_mem_152245;
    
    wout_mem_152245.references = NULL;
    
    struct memblock wkey_mem_152244;
    
    wkey_mem_152244.references = NULL;
    
    struct memblock wdown_mem_152243;
    
    wdown_mem_152243.references = NULL;
    wdown_mem_152243 = in0->v0->mem;
    wkey_mem_152244 = in0->v1->mem;
    wout_mem_152245 = in0->v2->mem;
    wpe_mem_152246 = in0->v3->mem;
    wqry_mem_152247 = in0->v4->mem;
    wte_mem_152248 = in0->v5->mem;
    wup_mem_152249 = in0->v6->mem;
    wval_mem_152250 = in0->v7->mem;
    wvoc_mem_152251 = in0->v8->mem;
    wdown_mem_152252 = in1->v0->mem;
    wkey_mem_152253 = in1->v1->mem;
    wout_mem_152254 = in1->v2->mem;
    wpe_mem_152255 = in1->v3->mem;
    wqry_mem_152256 = in1->v4->mem;
    wte_mem_152257 = in1->v5->mem;
    wup_mem_152258 = in1->v6->mem;
    wval_mem_152259 = in1->v7->mem;
    wvoc_mem_152260 = in1->v8->mem;
    wdown_mem_152261 = in2->v0->mem;
    wkey_mem_152262 = in2->v1->mem;
    wout_mem_152263 = in2->v2->mem;
    wpe_mem_152264 = in2->v3->mem;
    wqry_mem_152265 = in2->v4->mem;
    wte_mem_152266 = in2->v5->mem;
    wup_mem_152267 = in2->v6->mem;
    wval_mem_152268 = in2->v7->mem;
    wvoc_mem_152269 = in2->v8->mem;
    masks_mem_152270 = in3->mem;
    dls_mem_152271 = in4->mem;
    seqs_mem_152272 = in5->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && (((int64_t) 5 == in3->shape[0] && ((int64_t) 16 == in3->shape[1] && (int64_t) 16 == in3->shape[2])) && ((int64_t) 5 == in4->shape[0] && ((int64_t) 5 == in5->shape[0] && (int64_t) 16 == in5->shape[1]))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_154731, &mem_out_154732, &mem_out_154733, &mem_out_154734, &mem_out_154735, &mem_out_154736, &mem_out_154737, &mem_out_154738, &mem_out_154739, &mem_out_154740, &mem_out_154741, &mem_out_154742, &mem_out_154743, &mem_out_154744, &mem_out_154745, &mem_out_154746, &mem_out_154747, &mem_out_154748, &mem_out_154749, &mem_out_154750, &mem_out_154751, &mem_out_154752, &mem_out_154753, &mem_out_154754, &mem_out_154755, &mem_out_154756, &mem_out_154757, wdown_mem_152243, wkey_mem_152244, wout_mem_152245, wpe_mem_152246, wqry_mem_152247, wte_mem_152248, wup_mem_152249, wval_mem_152250, wvoc_mem_152251, wdown_mem_152252, wkey_mem_152253, wout_mem_152254, wpe_mem_152255, wqry_mem_152256, wte_mem_152257, wup_mem_152258, wval_mem_152259, wvoc_mem_152260, wdown_mem_152261, wkey_mem_152262, wout_mem_152263, wpe_mem_152264, wqry_mem_152265, wte_mem_152266, wup_mem_152267, wval_mem_152268, wvoc_mem_152269, masks_mem_152270, dls_mem_152271, seqs_mem_152272);
        if (ret == 0) {
            struct memblock mem_152234 = ctx->constants->mem_152234;
            struct memblock mem_152235 = ctx->constants->mem_152235;
            struct memblock mem_152236 = ctx->constants->mem_152236;
            struct memblock mem_152237 = ctx->constants->mem_152237;
            struct memblock mem_152238 = ctx->constants->mem_152238;
            struct memblock mem_152239 = ctx->constants->mem_152239;
            struct memblock mem_152240 = ctx->constants->mem_152240;
            struct memblock mem_152241 = ctx->constants->mem_152241;
            struct memblock mem_152242 = ctx->constants->mem_152242;
            
            assert((*out = (struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_154731;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_154732;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_154733;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_154734;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_154735;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_154736;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_154737;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_154738;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_154739;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v9->mem = mem_out_154740;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v10->mem = mem_out_154741;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v11->mem = mem_out_154742;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v12->mem = mem_out_154743;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v13->mem = mem_out_154744;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v14->mem = mem_out_154745;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v15->mem = mem_out_154746;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v16->mem = mem_out_154747;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v17->mem = mem_out_154748;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v18->mem = mem_out_154749;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v19->mem = mem_out_154750;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v20->mem = mem_out_154751;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v21->mem = mem_out_154752;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v22->mem = mem_out_154753;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v23->mem = mem_out_154754;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v24->mem = mem_out_154755;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v25->mem = mem_out_154756;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v26->mem = mem_out_154757;
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
    
    struct memblock mem_out_154739;
    
    mem_out_154739.references = NULL;
    
    struct memblock mem_out_154738;
    
    mem_out_154738.references = NULL;
    
    struct memblock mem_out_154737;
    
    mem_out_154737.references = NULL;
    
    struct memblock mem_out_154736;
    
    mem_out_154736.references = NULL;
    
    struct memblock mem_out_154735;
    
    mem_out_154735.references = NULL;
    
    struct memblock mem_out_154734;
    
    mem_out_154734.references = NULL;
    
    struct memblock mem_out_154733;
    
    mem_out_154733.references = NULL;
    
    struct memblock mem_out_154732;
    
    mem_out_154732.references = NULL;
    
    struct memblock mem_out_154731;
    
    mem_out_154731.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_154731, &mem_out_154732, &mem_out_154733, &mem_out_154734, &mem_out_154735, &mem_out_154736, &mem_out_154737, &mem_out_154738, &mem_out_154739);
        if (ret == 0) {
            struct memblock mem_152234 = ctx->constants->mem_152234;
            struct memblock mem_152235 = ctx->constants->mem_152235;
            struct memblock mem_152236 = ctx->constants->mem_152236;
            struct memblock mem_152237 = ctx->constants->mem_152237;
            struct memblock mem_152238 = ctx->constants->mem_152238;
            struct memblock mem_152239 = ctx->constants->mem_152239;
            struct memblock mem_152240 = ctx->constants->mem_152240;
            struct memblock mem_152241 = ctx->constants->mem_152241;
            struct memblock mem_152242 = ctx->constants->mem_152242;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_154731;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_154732;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_154733;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_154734;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_154735;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_154736;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_154737;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_154738;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_154739;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
