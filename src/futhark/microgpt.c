
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
    struct memblock mem_146715;
    struct memblock mem_146716;
    struct memblock mem_146717;
    struct memblock mem_146718;
    struct memblock mem_146719;
    struct memblock mem_146720;
    struct memblock mem_146721;
    struct memblock mem_146722;
    struct memblock mem_146723;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_12815(struct futhark_context *ctx, struct memblock *mem_out_p_149567, struct memblock *mem_out_p_149568, struct memblock *mem_out_p_149569, struct memblock w_mem_146724, struct memblock mw_mem_146725, struct memblock vw_mem_146726, struct memblock dw_mem_146727, int64_t n_107782, int64_t m_107783, int64_t step_107788, double lt_r_107789);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_12816(struct futhark_context *ctx, struct memblock *mem_out_p_149572, struct memblock *mem_out_p_149573, struct memblock *mem_out_p_149574, struct memblock w_mem_146724, struct memblock mw_mem_146725, struct memblock vw_mem_146726, struct memblock dw_mem_146727, int64_t n_108815, int64_t m_108816, int64_t step_108821, double lt_r_108822);
FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_149577, double *out_prim_out_149578, struct memblock wdown_mem_146724, struct memblock wkey_mem_146725, struct memblock wout_mem_146726, struct memblock wpe_mem_146727, struct memblock wqry_mem_146728, struct memblock wte_mem_146729, struct memblock wup_mem_146730, struct memblock wval_mem_146731, struct memblock wvoc_mem_146732, struct memblock tokens_mem_146733, struct memblock target_mem_146734, struct memblock mask_mem_146735);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_149636, struct memblock wdown_mem_146724, struct memblock wkey_mem_146725, struct memblock wout_mem_146726, struct memblock wpe_mem_146727, struct memblock wqry_mem_146728, struct memblock wte_mem_146729, struct memblock wup_mem_146730, struct memblock wval_mem_146731, struct memblock wvoc_mem_146732, struct memblock tokens_mem_146733, struct memblock mask_mem_146734);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_149693, struct memblock *mem_out_p_149694, struct memblock *mem_out_p_149695, struct memblock *mem_out_p_149696, struct memblock *mem_out_p_149697, struct memblock *mem_out_p_149698, struct memblock *mem_out_p_149699, struct memblock *mem_out_p_149700, struct memblock *mem_out_p_149701, struct memblock wte_mem_146724, struct memblock wpe_mem_146725, struct memblock wqry_mem_146726, struct memblock wkey_mem_146727, struct memblock wval_mem_146728, struct memblock wout_mem_146729, struct memblock wup_mem_146730, struct memblock wdown_mem_146731, struct memblock wvoc_mem_146732);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_149702, struct memblock *mem_out_p_149703, struct memblock *mem_out_p_149704, struct memblock *mem_out_p_149705, struct memblock *mem_out_p_149706, struct memblock *mem_out_p_149707, struct memblock *mem_out_p_149708, struct memblock *mem_out_p_149709, struct memblock *mem_out_p_149710, struct memblock *mem_out_p_149711, struct memblock *mem_out_p_149712, struct memblock *mem_out_p_149713, struct memblock *mem_out_p_149714, struct memblock *mem_out_p_149715, struct memblock *mem_out_p_149716, struct memblock *mem_out_p_149717, struct memblock *mem_out_p_149718, struct memblock *mem_out_p_149719, struct memblock *mem_out_p_149720, struct memblock *mem_out_p_149721, struct memblock *mem_out_p_149722, struct memblock *mem_out_p_149723, struct memblock *mem_out_p_149724, struct memblock *mem_out_p_149725, struct memblock *mem_out_p_149726, struct memblock *mem_out_p_149727, struct memblock *mem_out_p_149728, struct memblock wdown_mem_146724, struct memblock wkey_mem_146725, struct memblock wout_mem_146726, struct memblock wpe_mem_146727, struct memblock wqry_mem_146728, struct memblock wte_mem_146729, struct memblock wup_mem_146730, struct memblock wval_mem_146731, struct memblock wvoc_mem_146732, struct memblock wdown_mem_146733, struct memblock wkey_mem_146734, struct memblock wout_mem_146735, struct memblock wpe_mem_146736, struct memblock wqry_mem_146737, struct memblock wte_mem_146738, struct memblock wup_mem_146739, struct memblock wval_mem_146740, struct memblock wvoc_mem_146741, struct memblock wdown_mem_146742, struct memblock wkey_mem_146743, struct memblock wout_mem_146744, struct memblock wpe_mem_146745, struct memblock wqry_mem_146746, struct memblock wte_mem_146747, struct memblock wup_mem_146748, struct memblock wval_mem_146749, struct memblock wvoc_mem_146750, struct memblock masks_mem_146751, struct memblock dls_mem_146752, struct memblock seqs_mem_146753);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_149969, struct memblock *mem_out_p_149970, struct memblock *mem_out_p_149971, struct memblock *mem_out_p_149972, struct memblock *mem_out_p_149973, struct memblock *mem_out_p_149974, struct memblock *mem_out_p_149975, struct memblock *mem_out_p_149976, struct memblock *mem_out_p_149977);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_146715 (ctx->constants->mem_146715)
    #define mem_146716 (ctx->constants->mem_146716)
    #define mem_146717 (ctx->constants->mem_146717)
    #define mem_146718 (ctx->constants->mem_146718)
    #define mem_146719 (ctx->constants->mem_146719)
    #define mem_146720 (ctx->constants->mem_146720)
    #define mem_146721 (ctx->constants->mem_146721)
    #define mem_146722 (ctx->constants->mem_146722)
    #define mem_146723 (ctx->constants->mem_146723)
    mem_146715.references = NULL;
    mem_146716.references = NULL;
    mem_146717.references = NULL;
    mem_146718.references = NULL;
    mem_146719.references = NULL;
    mem_146720.references = NULL;
    mem_146721.references = NULL;
    mem_146722.references = NULL;
    mem_146723.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_146715, (int64_t) 3456, "mem_146715")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_149549 = 0; nest_i_149549 < (int64_t) 27; nest_i_149549++) {
        for (int64_t nest_i_149550 = 0; nest_i_149550 < (int64_t) 16; nest_i_149550++) {
            ((double *) mem_146715.mem)[nest_i_149549 * (int64_t) 16 + nest_i_149550] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_146716, (int64_t) 2048, "mem_146716")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_149551 = 0; nest_i_149551 < (int64_t) 16; nest_i_149551++) {
        for (int64_t nest_i_149552 = 0; nest_i_149552 < (int64_t) 16; nest_i_149552++) {
            ((double *) mem_146716.mem)[nest_i_149551 * (int64_t) 16 + nest_i_149552] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_146717, (int64_t) 2048, "mem_146717")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_149553 = 0; nest_i_149553 < (int64_t) 16; nest_i_149553++) {
        for (int64_t nest_i_149554 = 0; nest_i_149554 < (int64_t) 16; nest_i_149554++) {
            ((double *) mem_146717.mem)[nest_i_149553 * (int64_t) 16 + nest_i_149554] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_146718, (int64_t) 2048, "mem_146718")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_149555 = 0; nest_i_149555 < (int64_t) 16; nest_i_149555++) {
        for (int64_t nest_i_149556 = 0; nest_i_149556 < (int64_t) 16; nest_i_149556++) {
            ((double *) mem_146718.mem)[nest_i_149555 * (int64_t) 16 + nest_i_149556] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_146719, (int64_t) 2048, "mem_146719")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_149557 = 0; nest_i_149557 < (int64_t) 16; nest_i_149557++) {
        for (int64_t nest_i_149558 = 0; nest_i_149558 < (int64_t) 16; nest_i_149558++) {
            ((double *) mem_146719.mem)[nest_i_149557 * (int64_t) 16 + nest_i_149558] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_146720, (int64_t) 2048, "mem_146720")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_149559 = 0; nest_i_149559 < (int64_t) 16; nest_i_149559++) {
        for (int64_t nest_i_149560 = 0; nest_i_149560 < (int64_t) 16; nest_i_149560++) {
            ((double *) mem_146720.mem)[nest_i_149559 * (int64_t) 16 + nest_i_149560] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_146721, (int64_t) 8192, "mem_146721")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_149561 = 0; nest_i_149561 < (int64_t) 64; nest_i_149561++) {
        for (int64_t nest_i_149562 = 0; nest_i_149562 < (int64_t) 16; nest_i_149562++) {
            ((double *) mem_146721.mem)[nest_i_149561 * (int64_t) 16 + nest_i_149562] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_146722, (int64_t) 8192, "mem_146722")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_149563 = 0; nest_i_149563 < (int64_t) 16; nest_i_149563++) {
        for (int64_t nest_i_149564 = 0; nest_i_149564 < (int64_t) 64; nest_i_149564++) {
            ((double *) mem_146722.mem)[nest_i_149563 * (int64_t) 64 + nest_i_149564] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_146723, (int64_t) 3456, "mem_146723")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_149565 = 0; nest_i_149565 < (int64_t) 27; nest_i_149565++) {
        for (int64_t nest_i_149566 = 0; nest_i_149566 < (int64_t) 16; nest_i_149566++) {
            ((double *) mem_146723.mem)[nest_i_149565 * (int64_t) 16 + nest_i_149566] = 0.0;
        }
    }
    #undef mem_146715
    #undef mem_146716
    #undef mem_146717
    #undef mem_146718
    #undef mem_146719
    #undef mem_146720
    #undef mem_146721
    #undef mem_146722
    #undef mem_146723
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_146715, "ctx->constants->mem_146715") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_146716, "ctx->constants->mem_146716") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_146717, "ctx->constants->mem_146717") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_146718, "ctx->constants->mem_146718") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_146719, "ctx->constants->mem_146719") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_146720, "ctx->constants->mem_146720") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_146721, "ctx->constants->mem_146721") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_146722, "ctx->constants->mem_146722") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_146723, "ctx->constants->mem_146723") != 0)
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_12815(struct futhark_context *ctx, struct memblock *mem_out_p_149567, struct memblock *mem_out_p_149568, struct memblock *mem_out_p_149569, struct memblock w_mem_146724, struct memblock mw_mem_146725, struct memblock vw_mem_146726, struct memblock dw_mem_146727, int64_t n_107782, int64_t m_107783, int64_t step_107788, double lt_r_107789)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_146768_cached_sizze_149570 = 0;
    unsigned char *mem_146768 = NULL;
    int64_t mem_146771_cached_sizze_149571 = 0;
    unsigned char *mem_146771 = NULL;
    struct memblock mem_146806;
    
    mem_146806.references = NULL;
    
    struct memblock mem_146733;
    
    mem_146733.references = NULL;
    
    struct memblock mem_146730;
    
    mem_146730.references = NULL;
    
    struct memblock mem_out_149152;
    
    mem_out_149152.references = NULL;
    
    struct memblock mem_out_149151;
    
    mem_out_149151.references = NULL;
    
    struct memblock mem_out_149150;
    
    mem_out_149150.references = NULL;
    
    struct memblock mem_146715 = ctx->constants->mem_146715;
    struct memblock mem_146716 = ctx->constants->mem_146716;
    struct memblock mem_146717 = ctx->constants->mem_146717;
    struct memblock mem_146718 = ctx->constants->mem_146718;
    struct memblock mem_146719 = ctx->constants->mem_146719;
    struct memblock mem_146720 = ctx->constants->mem_146720;
    struct memblock mem_146721 = ctx->constants->mem_146721;
    struct memblock mem_146722 = ctx->constants->mem_146722;
    struct memblock mem_146723 = ctx->constants->mem_146723;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_146728 = (int64_t) 8 * n_107782;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_146729 = m_107783 * binop_x_146728;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_146730, bytes_146729, "mem_146730")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_146733, bytes_146729, "mem_146733")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145637 = 0; i_145637 < n_107782; i_145637++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145630 = 0; i_145630 < m_107783; i_145630++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_134163 = ((double *) mw_mem_146725.mem)[i_145637 * m_107783 + i_145630];
            
            // futhark/microgpt.fut:483:10-20
            
            double zp_lhs_134164 = 0.85 * zt_rhs_134163;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_134165 = ((double *) dw_mem_146727.mem)[i_145637 * m_107783 + i_145630];
            
            // futhark/microgpt.fut:483:35-45
            
            double zp_rhs_134166 = 0.15000000000000002 * zt_rhs_134165;
            
            // futhark/microgpt.fut:483:21-45
            
            double lifted_lambda_res_134167 = zp_lhs_134164 + zp_rhs_134166;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_134174 = ((double *) vw_mem_146726.mem)[i_145637 * m_107783 + i_145630];
            
            // futhark/microgpt.fut:485:10-20
            
            double zp_lhs_134175 = 0.99 * zt_rhs_134174;
            
            // futhark/microgpt.fut:485:35-45
            
            double zt_lhs_134177 = 1.0000000000000009e-2 * zt_rhs_134165;
            
            // futhark/microgpt.fut:485:46-56
            
            double zp_rhs_134178 = zt_rhs_134165 * zt_lhs_134177;
            
            // futhark/microgpt.fut:485:21-56
            
            double lifted_lambda_res_134179 = zp_lhs_134175 + zp_rhs_134178;
            
            ((double *) mem_146730.mem)[i_145637 * m_107783 + i_145630] = lifted_lambda_res_134179;
            ((double *) mem_146733.mem)[i_145637 * m_107783 + i_145630] = lifted_lambda_res_134167;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_113199 = sitofp_i64_f64(step_107788);
    
    // futhark/microgpt.fut:487:54-57
    
    double ztzt_rhs_113200 = 1.0 + i64_res_113199;
    
    // futhark/microgpt.fut:487:30-57
    
    double zm_rhs_113201 = fpow64(0.85, ztzt_rhs_113200);
    
    // futhark/microgpt.fut:487:23-57
    
    double zs_rhs_113202 = 1.0 - zm_rhs_113201;
    
    // futhark/microgpt.fut:489:31-58
    
    double zm_rhs_113240 = fpow64(0.99, ztzt_rhs_113200);
    
    // futhark/microgpt.fut:489:23-58
    
    double zs_rhs_113241 = 1.0 - zm_rhs_113240;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_146768_cached_sizze_149570 < bytes_146729) {
        err = lexical_realloc(ctx, &mem_146768, &mem_146768_cached_sizze_149570, bytes_146729);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146771_cached_sizze_149571 < bytes_146729) {
        err = lexical_realloc(ctx, &mem_146771, &mem_146771_cached_sizze_149571, bytes_146729);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145651 = 0; i_145651 < n_107782; i_145651++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145644 = 0; i_145644 < m_107783; i_145644++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_134199 = ((double *) mem_146733.mem)[i_145651 * m_107783 + i_145644];
            
            // futhark/microgpt.fut:487:18-57
            
            double lifted_lambda_res_134200 = zs_lhs_134199 / zs_rhs_113202;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_134207 = ((double *) mem_146730.mem)[i_145651 * m_107783 + i_145644];
            
            // futhark/microgpt.fut:489:18-58
            
            double lifted_lambda_res_134208 = zs_lhs_134207 / zs_rhs_113241;
            
            ((double *) mem_146768)[i_145651 * m_107783 + i_145644] = lifted_lambda_res_134208;
            ((double *) mem_146771)[i_145651 * m_107783 + i_145644] = lifted_lambda_res_134200;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_146806, bytes_146729, "mem_146806")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145660 = 0; i_145660 < n_107782; i_145660++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145656 = 0; i_145656 < m_107783; i_145656++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_112192 = ((double *) w_mem_146724.mem)[i_145660 * m_107783 + i_145656];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_112193 = ((double *) mem_146771)[i_145660 * m_107783 + i_145656];
            
            // futhark/microgpt.fut:491:21-34
            
            double zs_lhs_112194 = lt_r_107789 * zt_rhs_112193;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_112195 = ((double *) mem_146768)[i_145660 * m_107783 + i_145656];
            
            // futhark/microgpt.fut:491:51-57
            
            double zp_lhs_112196 = fpow64(ztzt_lhs_112195, 0.5);
            
            // futhark/microgpt.fut:491:59-71
            
            double zs_rhs_112197 = 1.0e-8 + zp_lhs_112196;
            
            // futhark/microgpt.fut:491:35-71
            
            double zm_rhs_112198 = zs_lhs_112194 / zs_rhs_112197;
            
            // futhark/microgpt.fut:491:13-71
            
            double lifted_lambda_res_112199 = zm_lhs_112192 - zm_rhs_112198;
            
            ((double *) mem_146806.mem)[i_145660 * m_107783 + i_145656] = lifted_lambda_res_112199;
        }
    }
    if (memblock_set(ctx, &mem_out_149150, &mem_146806, "mem_146806") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149151, &mem_146733, "mem_146733") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149152, &mem_146730, "mem_146730") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149567, &mem_out_149150, "mem_out_149150") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149568, &mem_out_149151, "mem_out_149151") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149569, &mem_out_149152, "mem_out_149152") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_146768);
        free(mem_146771);
        if (memblock_unref(ctx, &mem_146806, "mem_146806") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146733, "mem_146733") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146730, "mem_146730") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149152, "mem_out_149152") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149151, "mem_out_149151") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149150, "mem_out_149150") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_12816(struct futhark_context *ctx, struct memblock *mem_out_p_149572, struct memblock *mem_out_p_149573, struct memblock *mem_out_p_149574, struct memblock w_mem_146724, struct memblock mw_mem_146725, struct memblock vw_mem_146726, struct memblock dw_mem_146727, int64_t n_108815, int64_t m_108816, int64_t step_108821, double lt_r_108822)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_146768_cached_sizze_149575 = 0;
    unsigned char *mem_146768 = NULL;
    int64_t mem_146771_cached_sizze_149576 = 0;
    unsigned char *mem_146771 = NULL;
    struct memblock mem_146806;
    
    mem_146806.references = NULL;
    
    struct memblock mem_146733;
    
    mem_146733.references = NULL;
    
    struct memblock mem_146730;
    
    mem_146730.references = NULL;
    
    struct memblock mem_out_149152;
    
    mem_out_149152.references = NULL;
    
    struct memblock mem_out_149151;
    
    mem_out_149151.references = NULL;
    
    struct memblock mem_out_149150;
    
    mem_out_149150.references = NULL;
    
    struct memblock mem_146715 = ctx->constants->mem_146715;
    struct memblock mem_146716 = ctx->constants->mem_146716;
    struct memblock mem_146717 = ctx->constants->mem_146717;
    struct memblock mem_146718 = ctx->constants->mem_146718;
    struct memblock mem_146719 = ctx->constants->mem_146719;
    struct memblock mem_146720 = ctx->constants->mem_146720;
    struct memblock mem_146721 = ctx->constants->mem_146721;
    struct memblock mem_146722 = ctx->constants->mem_146722;
    struct memblock mem_146723 = ctx->constants->mem_146723;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_146728 = (int64_t) 8 * n_108815;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_146729 = m_108816 * binop_x_146728;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_146730, bytes_146729, "mem_146730")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_146733, bytes_146729, "mem_146733")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145637 = 0; i_145637 < n_108815; i_145637++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145630 = 0; i_145630 < m_108816; i_145630++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_134163 = ((double *) mw_mem_146725.mem)[i_145637 * m_108816 + i_145630];
            
            // futhark/microgpt.fut:483:10-20
            
            double zp_lhs_134164 = 0.85 * zt_rhs_134163;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_134165 = ((double *) dw_mem_146727.mem)[i_145637 * m_108816 + i_145630];
            
            // futhark/microgpt.fut:483:35-45
            
            double zp_rhs_134166 = 0.15000000000000002 * zt_rhs_134165;
            
            // futhark/microgpt.fut:483:21-45
            
            double lifted_lambda_res_134167 = zp_lhs_134164 + zp_rhs_134166;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_134174 = ((double *) vw_mem_146726.mem)[i_145637 * m_108816 + i_145630];
            
            // futhark/microgpt.fut:485:10-20
            
            double zp_lhs_134175 = 0.99 * zt_rhs_134174;
            
            // futhark/microgpt.fut:485:35-45
            
            double zt_lhs_134177 = 1.0000000000000009e-2 * zt_rhs_134165;
            
            // futhark/microgpt.fut:485:46-56
            
            double zp_rhs_134178 = zt_rhs_134165 * zt_lhs_134177;
            
            // futhark/microgpt.fut:485:21-56
            
            double lifted_lambda_res_134179 = zp_lhs_134175 + zp_rhs_134178;
            
            ((double *) mem_146730.mem)[i_145637 * m_108816 + i_145630] = lifted_lambda_res_134179;
            ((double *) mem_146733.mem)[i_145637 * m_108816 + i_145630] = lifted_lambda_res_134167;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_113199 = sitofp_i64_f64(step_108821);
    
    // futhark/microgpt.fut:487:54-57
    
    double ztzt_rhs_113200 = 1.0 + i64_res_113199;
    
    // futhark/microgpt.fut:487:30-57
    
    double zm_rhs_113201 = fpow64(0.85, ztzt_rhs_113200);
    
    // futhark/microgpt.fut:487:23-57
    
    double zs_rhs_113202 = 1.0 - zm_rhs_113201;
    
    // futhark/microgpt.fut:489:31-58
    
    double zm_rhs_113240 = fpow64(0.99, ztzt_rhs_113200);
    
    // futhark/microgpt.fut:489:23-58
    
    double zs_rhs_113241 = 1.0 - zm_rhs_113240;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_146768_cached_sizze_149575 < bytes_146729) {
        err = lexical_realloc(ctx, &mem_146768, &mem_146768_cached_sizze_149575, bytes_146729);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146771_cached_sizze_149576 < bytes_146729) {
        err = lexical_realloc(ctx, &mem_146771, &mem_146771_cached_sizze_149576, bytes_146729);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145651 = 0; i_145651 < n_108815; i_145651++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145644 = 0; i_145644 < m_108816; i_145644++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_134199 = ((double *) mem_146733.mem)[i_145651 * m_108816 + i_145644];
            
            // futhark/microgpt.fut:487:18-57
            
            double lifted_lambda_res_134200 = zs_lhs_134199 / zs_rhs_113202;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_134207 = ((double *) mem_146730.mem)[i_145651 * m_108816 + i_145644];
            
            // futhark/microgpt.fut:489:18-58
            
            double lifted_lambda_res_134208 = zs_lhs_134207 / zs_rhs_113241;
            
            ((double *) mem_146768)[i_145651 * m_108816 + i_145644] = lifted_lambda_res_134208;
            ((double *) mem_146771)[i_145651 * m_108816 + i_145644] = lifted_lambda_res_134200;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_146806, bytes_146729, "mem_146806")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145660 = 0; i_145660 < n_108815; i_145660++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145656 = 0; i_145656 < m_108816; i_145656++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_112192 = ((double *) w_mem_146724.mem)[i_145660 * m_108816 + i_145656];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_112193 = ((double *) mem_146771)[i_145660 * m_108816 + i_145656];
            
            // futhark/microgpt.fut:491:21-34
            
            double zs_lhs_112194 = lt_r_108822 * zt_rhs_112193;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_112195 = ((double *) mem_146768)[i_145660 * m_108816 + i_145656];
            
            // futhark/microgpt.fut:491:51-57
            
            double zp_lhs_112196 = fpow64(ztzt_lhs_112195, 0.5);
            
            // futhark/microgpt.fut:491:59-71
            
            double zs_rhs_112197 = 1.0e-8 + zp_lhs_112196;
            
            // futhark/microgpt.fut:491:35-71
            
            double zm_rhs_112198 = zs_lhs_112194 / zs_rhs_112197;
            
            // futhark/microgpt.fut:491:13-71
            
            double lifted_lambda_res_112199 = zm_lhs_112192 - zm_rhs_112198;
            
            ((double *) mem_146806.mem)[i_145660 * m_108816 + i_145656] = lifted_lambda_res_112199;
        }
    }
    if (memblock_set(ctx, &mem_out_149150, &mem_146806, "mem_146806") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149151, &mem_146733, "mem_146733") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149152, &mem_146730, "mem_146730") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149572, &mem_out_149150, "mem_out_149150") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149573, &mem_out_149151, "mem_out_149151") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149574, &mem_out_149152, "mem_out_149152") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_146768);
        free(mem_146771);
        if (memblock_unref(ctx, &mem_146806, "mem_146806") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146733, "mem_146733") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146730, "mem_146730") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149152, "mem_out_149152") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149151, "mem_out_149151") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149150, "mem_out_149150") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_149577, double *out_prim_out_149578, struct memblock wdown_mem_146724, struct memblock wkey_mem_146725, struct memblock wout_mem_146726, struct memblock wpe_mem_146727, struct memblock wqry_mem_146728, struct memblock wte_mem_146729, struct memblock wup_mem_146730, struct memblock wval_mem_146731, struct memblock wvoc_mem_146732, struct memblock tokens_mem_146733, struct memblock target_mem_146734, struct memblock mask_mem_146735)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_146736_cached_sizze_149579 = 0;
    unsigned char *mem_146736 = NULL;
    int64_t mem_146741_cached_sizze_149580 = 0;
    unsigned char *mem_146741 = NULL;
    int64_t mem_146752_cached_sizze_149581 = 0;
    unsigned char *mem_146752 = NULL;
    int64_t mem_146757_cached_sizze_149582 = 0;
    unsigned char *mem_146757 = NULL;
    int64_t mem_146764_cached_sizze_149583 = 0;
    unsigned char *mem_146764 = NULL;
    int64_t mem_146775_cached_sizze_149584 = 0;
    unsigned char *mem_146775 = NULL;
    int64_t mem_146780_cached_sizze_149585 = 0;
    unsigned char *mem_146780 = NULL;
    int64_t mem_146787_cached_sizze_149586 = 0;
    unsigned char *mem_146787 = NULL;
    int64_t mem_146798_cached_sizze_149587 = 0;
    unsigned char *mem_146798 = NULL;
    int64_t mem_146799_cached_sizze_149588 = 0;
    unsigned char *mem_146799 = NULL;
    int64_t mem_146800_cached_sizze_149589 = 0;
    unsigned char *mem_146800 = NULL;
    int64_t mem_146813_cached_sizze_149590 = 0;
    unsigned char *mem_146813 = NULL;
    int64_t mem_146814_cached_sizze_149591 = 0;
    unsigned char *mem_146814 = NULL;
    int64_t mem_146815_cached_sizze_149592 = 0;
    unsigned char *mem_146815 = NULL;
    int64_t mem_146846_cached_sizze_149593 = 0;
    unsigned char *mem_146846 = NULL;
    int64_t mem_146847_cached_sizze_149594 = 0;
    unsigned char *mem_146847 = NULL;
    int64_t mem_146848_cached_sizze_149595 = 0;
    unsigned char *mem_146848 = NULL;
    int64_t mem_146864_cached_sizze_149596 = 0;
    unsigned char *mem_146864 = NULL;
    int64_t mem_146865_cached_sizze_149597 = 0;
    unsigned char *mem_146865 = NULL;
    int64_t mem_146866_cached_sizze_149598 = 0;
    unsigned char *mem_146866 = NULL;
    int64_t mem_146879_cached_sizze_149599 = 0;
    unsigned char *mem_146879 = NULL;
    int64_t mem_146880_cached_sizze_149600 = 0;
    unsigned char *mem_146880 = NULL;
    int64_t mem_146881_cached_sizze_149601 = 0;
    unsigned char *mem_146881 = NULL;
    int64_t mem_146927_cached_sizze_149602 = 0;
    unsigned char *mem_146927 = NULL;
    int64_t mem_146933_cached_sizze_149603 = 0;
    unsigned char *mem_146933 = NULL;
    int64_t mem_146938_cached_sizze_149604 = 0;
    unsigned char *mem_146938 = NULL;
    int64_t mem_146949_cached_sizze_149605 = 0;
    unsigned char *mem_146949 = NULL;
    int64_t mem_146954_cached_sizze_149606 = 0;
    unsigned char *mem_146954 = NULL;
    int64_t mem_146965_cached_sizze_149607 = 0;
    unsigned char *mem_146965 = NULL;
    int64_t mem_146970_cached_sizze_149608 = 0;
    unsigned char *mem_146970 = NULL;
    int64_t mem_146977_cached_sizze_149609 = 0;
    unsigned char *mem_146977 = NULL;
    int64_t mem_146984_cached_sizze_149610 = 0;
    unsigned char *mem_146984 = NULL;
    int64_t mem_146995_cached_sizze_149611 = 0;
    unsigned char *mem_146995 = NULL;
    int64_t mem_147000_cached_sizze_149612 = 0;
    unsigned char *mem_147000 = NULL;
    int64_t mem_147011_cached_sizze_149613 = 0;
    unsigned char *mem_147011 = NULL;
    int64_t mem_147016_cached_sizze_149614 = 0;
    unsigned char *mem_147016 = NULL;
    int64_t mem_147032_cached_sizze_149615 = 0;
    unsigned char *mem_147032 = NULL;
    int64_t mem_147037_cached_sizze_149616 = 0;
    unsigned char *mem_147037 = NULL;
    int64_t mem_147048_cached_sizze_149617 = 0;
    unsigned char *mem_147048 = NULL;
    int64_t mem_147053_cached_sizze_149618 = 0;
    unsigned char *mem_147053 = NULL;
    int64_t mem_147064_cached_sizze_149619 = 0;
    unsigned char *mem_147064 = NULL;
    int64_t mem_147069_cached_sizze_149620 = 0;
    unsigned char *mem_147069 = NULL;
    int64_t mem_147080_cached_sizze_149621 = 0;
    unsigned char *mem_147080 = NULL;
    int64_t mem_147085_cached_sizze_149622 = 0;
    unsigned char *mem_147085 = NULL;
    int64_t mem_147092_cached_sizze_149623 = 0;
    unsigned char *mem_147092 = NULL;
    int64_t mem_147103_cached_sizze_149624 = 0;
    unsigned char *mem_147103 = NULL;
    int64_t mem_147108_cached_sizze_149625 = 0;
    unsigned char *mem_147108 = NULL;
    int64_t mem_147119_cached_sizze_149626 = 0;
    unsigned char *mem_147119 = NULL;
    int64_t mem_147124_cached_sizze_149627 = 0;
    unsigned char *mem_147124 = NULL;
    int64_t mem_147135_cached_sizze_149628 = 0;
    unsigned char *mem_147135 = NULL;
    int64_t mem_147140_cached_sizze_149629 = 0;
    unsigned char *mem_147140 = NULL;
    int64_t mem_147151_cached_sizze_149630 = 0;
    unsigned char *mem_147151 = NULL;
    int64_t mem_147156_cached_sizze_149631 = 0;
    unsigned char *mem_147156 = NULL;
    int64_t mem_147167_cached_sizze_149632 = 0;
    unsigned char *mem_147167 = NULL;
    int64_t mem_147172_cached_sizze_149633 = 0;
    unsigned char *mem_147172 = NULL;
    int64_t mem_147187_cached_sizze_149634 = 0;
    unsigned char *mem_147187 = NULL;
    int64_t mem_147194_cached_sizze_149635 = 0;
    unsigned char *mem_147194 = NULL;
    struct memblock mem_147183;
    
    mem_147183.references = NULL;
    
    struct memblock mem_out_149150;
    
    mem_out_149150.references = NULL;
    
    struct memblock mem_146715 = ctx->constants->mem_146715;
    struct memblock mem_146716 = ctx->constants->mem_146716;
    struct memblock mem_146717 = ctx->constants->mem_146717;
    struct memblock mem_146718 = ctx->constants->mem_146718;
    struct memblock mem_146719 = ctx->constants->mem_146719;
    struct memblock mem_146720 = ctx->constants->mem_146720;
    struct memblock mem_146721 = ctx->constants->mem_146721;
    struct memblock mem_146722 = ctx->constants->mem_146722;
    struct memblock mem_146723 = ctx->constants->mem_146723;
    double prim_out_149151;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_146736_cached_sizze_149579 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146736, &mem_146736_cached_sizze_149579, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146741_cached_sizze_149580 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146741, &mem_146741_cached_sizze_149580, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145632 = 0; i_145632 < (int64_t) 16; i_145632++) {
        // futhark/microgpt.fut:473:41-50
        
        int64_t tmp_126275 = ((int64_t *) tokens_mem_146733.mem)[i_145632];
        
        // futhark/microgpt.fut:473:37-51
        
        bool x_126276 = sle64((int64_t) 0, tmp_126275);
        
        // futhark/microgpt.fut:473:37-51
        
        bool y_126277 = slt64(tmp_126275, (int64_t) 27);
        
        // futhark/microgpt.fut:473:37-51
        
        bool bounds_check_126278 = x_126276 && y_126277;
        
        // futhark/microgpt.fut:473:37-51
        
        bool index_certs_126279;
        
        if (!bounds_check_126278) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_126275, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:473:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:473:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145628 = 0; i_145628 < (int64_t) 16; i_145628++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_126286 = ((double *) wte_mem_146729.mem)[tmp_126275 * (int64_t) 16 + i_145628];
            
            ((double *) mem_146741)[i_145628] = lifted_lambda_res_126286;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_146736, i_145632 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146741, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146752_cached_sizze_149581 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146752, &mem_146752_cached_sizze_149581, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146757_cached_sizze_149582 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146757, &mem_146757_cached_sizze_149582, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146764_cached_sizze_149583 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146764, &mem_146764_cached_sizze_149583, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145644 = 0; i_145644 < (int64_t) 16; i_145644++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_126312;
        double r_126314 = 0.0;
        
        for (int64_t i_126313 = 0; i_126313 < (int64_t) 16; i_126313++) {
            // futhark/microgpt.fut:71:46-49
            
            double zp_lhs_126315 = ((double *) wpe_mem_146727.mem)[i_145644 * (int64_t) 16 + i_126313];
            
            // futhark/microgpt.fut:71:46-49
            
            double zp_rhs_126316 = ((double *) mem_146736)[i_145644 * (int64_t) 16 + i_126313];
            
            // futhark/microgpt.fut:203:76-116
            
            double zp_res_126317 = zp_lhs_126315 + zp_rhs_126316;
            
            // futhark/microgpt.fut:203:94-163
            
            double zt_res_126318 = zp_res_126317 * zp_res_126317;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_126319 = r_126314 + zt_res_126318;
            double r_tmp_149155 = zp_res_126319;
            
            r_126314 = r_tmp_149155;
        }
        defunc_0_lifted_lambda_res_126312 = r_126314;
        // futhark/microgpt.fut:203:54-182
        
        double zs_res_126320 = defunc_0_lifted_lambda_res_126312 / 16.0;
        
        // futhark/microgpt.fut:204:24-55
        
        double zp_res_126321 = 1.0e-5 + zs_res_126320;
        
        // futhark/microgpt.fut:204:16-55
        
        double sqrt_res_126322 = futrts_sqrt64(zp_res_126321);
        
        // futhark/microgpt.fut:205:85-96
        
        double zs_res_126323 = 1.0 / sqrt_res_126322;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145636 = 0; i_145636 < (int64_t) 16; i_145636++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_126330 = ((double *) wpe_mem_146727.mem)[i_145644 * (int64_t) 16 + i_145636];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_126331 = ((double *) mem_146736)[i_145644 * (int64_t) 16 + i_145636];
            
            // futhark/microgpt.fut:205:38-78
            
            double zp_res_126332 = zp_lhs_126330 + zp_rhs_126331;
            
            // futhark/microgpt.fut:205:56-96
            
            double zt_res_126333 = zs_res_126323 * zp_res_126332;
            
            ((double *) mem_146757)[i_145636] = zt_res_126333;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145640 = 0; i_145640 < (int64_t) 16; i_145640++) {
            // futhark/microgpt.fut:206:4-14
            
            double lifted_lambda_res_126341 = ((double *) mem_146757)[i_145640];
            
            ((double *) mem_146764)[i_145640] = lifted_lambda_res_126341;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_146752, i_145644 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146764, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146775_cached_sizze_149584 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146775, &mem_146775_cached_sizze_149584, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146780_cached_sizze_149585 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146780, &mem_146780_cached_sizze_149585, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146787_cached_sizze_149586 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146787, &mem_146787_cached_sizze_149586, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145656 = 0; i_145656 < (int64_t) 16; i_145656++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_126350;
        double r_126352 = 0.0;
        
        for (int64_t i_126351 = 0; i_126351 < (int64_t) 16; i_126351++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_126353 = ((double *) mem_146752)[i_145656 * (int64_t) 16 + i_126351];
            
            // futhark/microgpt.fut:207:78-115
            
            double zt_res_126354 = zt_lhs_126353 * zt_lhs_126353;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_126355 = r_126352 + zt_res_126354;
            double r_tmp_149159 = zp_res_126355;
            
            r_126352 = r_tmp_149159;
        }
        defunc_0_lifted_lambda_res_126350 = r_126352;
        // futhark/microgpt.fut:207:57-133
        
        double zs_res_126356 = defunc_0_lifted_lambda_res_126350 / 16.0;
        
        // futhark/microgpt.fut:208:24-55
        
        double zp_res_126357 = 1.0e-5 + zs_res_126356;
        
        // futhark/microgpt.fut:208:16-55
        
        double sqrt_res_126358 = futrts_sqrt64(zp_res_126357);
        
        // futhark/microgpt.fut:209:59-70
        
        double zs_res_126359 = 1.0 / sqrt_res_126358;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145648 = 0; i_145648 < (int64_t) 16; i_145648++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_126366 = ((double *) mem_146752)[i_145656 * (int64_t) 16 + i_145648];
            
            // futhark/microgpt.fut:209:37-70
            
            double zt_res_126367 = zs_res_126359 * zt_lhs_126366;
            
            ((double *) mem_146780)[i_145648] = zt_res_126367;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145652 = 0; i_145652 < (int64_t) 16; i_145652++) {
            // futhark/microgpt.fut:210:4-14
            
            double lifted_lambda_res_126375 = ((double *) mem_146780)[i_145652];
            
            ((double *) mem_146787)[i_145652] = lifted_lambda_res_126375;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_146775, i_145656 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146787, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146798_cached_sizze_149587 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146798, &mem_146798_cached_sizze_149587, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146799_cached_sizze_149588 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146799, &mem_146799_cached_sizze_149588, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146800_cached_sizze_149589 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146800, &mem_146800_cached_sizze_149589, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146813_cached_sizze_149590 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146813, &mem_146813_cached_sizze_149590, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146814_cached_sizze_149591 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146814, &mem_146814_cached_sizze_149591, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146815_cached_sizze_149592 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146815, &mem_146815_cached_sizze_149592, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145674 = 0; i_145674 < (int64_t) 16; i_145674++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145664 = 0; i_145664 < (int64_t) 16; i_145664++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_134381;
            double r_134383 = 0.0;
            
            for (int64_t i_134382 = 0; i_134382 < (int64_t) 16; i_134382++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_134384 = ((double *) wqry_mem_146728.mem)[i_145664 * (int64_t) 16 + i_134382];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_134385 = ((double *) mem_146775)[i_145674 * (int64_t) 16 + i_134382];
                
                // futhark/microgpt.fut:211:66-105
                
                double zt_res_134386 = zt_lhs_134384 * zt_rhs_134385;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_134387 = r_134383 + zt_res_134386;
                double r_tmp_149168 = zp_res_134387;
                
                r_134383 = r_tmp_149168;
            }
            defunc_0_lifted_lambda_res_134381 = r_134383;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_134394;
            double r_134396 = 0.0;
            
            for (int64_t i_134395 = 0; i_134395 < (int64_t) 16; i_134395++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_134397 = ((double *) wkey_mem_146725.mem)[i_145664 * (int64_t) 16 + i_134395];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_134398 = ((double *) mem_146775)[i_145674 * (int64_t) 16 + i_134395];
                
                // futhark/microgpt.fut:212:66-105
                
                double zt_res_134399 = zt_lhs_134397 * zt_rhs_134398;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_134400 = r_134396 + zt_res_134399;
                double r_tmp_149169 = zp_res_134400;
                
                r_134396 = r_tmp_149169;
            }
            defunc_0_lifted_lambda_res_134394 = r_134396;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_134410;
            double r_134412 = 0.0;
            
            for (int64_t i_134411 = 0; i_134411 < (int64_t) 16; i_134411++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_134413 = ((double *) wval_mem_146731.mem)[i_145664 * (int64_t) 16 + i_134411];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_134414 = ((double *) mem_146775)[i_145674 * (int64_t) 16 + i_134411];
                
                // futhark/microgpt.fut:213:66-105
                
                double zt_res_134415 = zt_lhs_134413 * zt_rhs_134414;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_134416 = r_134412 + zt_res_134415;
                double r_tmp_149170 = zp_res_134416;
                
                r_134412 = r_tmp_149170;
            }
            defunc_0_lifted_lambda_res_134410 = r_134412;
            ((double *) mem_146813)[i_145664] = defunc_0_lifted_lambda_res_134410;
            ((double *) mem_146814)[i_145664] = defunc_0_lifted_lambda_res_134394;
            ((double *) mem_146815)[i_145664] = defunc_0_lifted_lambda_res_134381;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_146798, i_145674 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146813, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_146799, i_145674 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146814, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_146800, i_145674 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146815, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146846_cached_sizze_149593 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146846, &mem_146846_cached_sizze_149593, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146847_cached_sizze_149594 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146847, &mem_146847_cached_sizze_149594, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146848_cached_sizze_149595 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146848, &mem_146848_cached_sizze_149595, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146864_cached_sizze_149596 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_146864, &mem_146864_cached_sizze_149596, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146865_cached_sizze_149597 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_146865, &mem_146865_cached_sizze_149597, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146866_cached_sizze_149598 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_146866, &mem_146866_cached_sizze_149598, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146879_cached_sizze_149599 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_146879, &mem_146879_cached_sizze_149599, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146880_cached_sizze_149600 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_146880, &mem_146880_cached_sizze_149600, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146881_cached_sizze_149601 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_146881, &mem_146881_cached_sizze_149601, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145704 = 0; i_145704 < (int64_t) 4; i_145704++) {
        // futhark/microgpt.fut:214:69-72
        
        int64_t zp_lhs_134257 = mul64((int64_t) 4, i_145704);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145694 = 0; i_145694 < (int64_t) 16; i_145694++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145684 = 0; i_145684 < (int64_t) 4; i_145684++) {
                // futhark/microgpt.fut:214:74-81
                
                int64_t tmp_134574 = add64(zp_lhs_134257, i_145684);
                
                // futhark/microgpt.fut:214:51-83
                
                bool x_134575 = sle64((int64_t) 0, tmp_134574);
                
                // futhark/microgpt.fut:214:51-83
                
                bool y_134576 = slt64(tmp_134574, (int64_t) 16);
                
                // futhark/microgpt.fut:214:51-83
                
                bool bounds_check_134577 = x_134575 && y_134576;
                
                // futhark/microgpt.fut:214:51-83
                
                bool index_certs_134578;
                
                if (!bounds_check_134577) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_134574, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:214:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:214:15-84\n   #9  futhark/microgpt.fut:474:7-76\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_134579 = ((double *) mem_146800)[i_145694 * (int64_t) 16 + tmp_134574];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_134587 = ((double *) mem_146799)[i_145694 * (int64_t) 16 + tmp_134574];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_134598 = ((double *) mem_146798)[i_145694 * (int64_t) 16 + tmp_134574];
                
                ((double *) mem_146879)[i_145684] = lifted_lambda_res_134598;
                ((double *) mem_146880)[i_145684] = lifted_lambda_res_134587;
                ((double *) mem_146881)[i_145684] = lifted_lambda_res_134579;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146864, i_145694 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146879, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146865, i_145694 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146880, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146866, i_145694 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146881, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_146846, i_145704 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_146864, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_146847, i_145704 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_146865, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_146848, i_145704 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_146866, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146927_cached_sizze_149602 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146927, &mem_146927_cached_sizze_149602, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146933_cached_sizze_149603 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146933, &mem_146933_cached_sizze_149603, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146938_cached_sizze_149604 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146938, &mem_146938_cached_sizze_149604, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146949_cached_sizze_149605 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146949, &mem_146949_cached_sizze_149605, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146954_cached_sizze_149606 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146954, &mem_146954_cached_sizze_149606, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146965_cached_sizze_149607 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146965, &mem_146965_cached_sizze_149607, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146970_cached_sizze_149608 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146970, &mem_146970_cached_sizze_149608, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146977_cached_sizze_149609 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146977, &mem_146977_cached_sizze_149609, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146984_cached_sizze_149610 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146984, &mem_146984_cached_sizze_149610, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146995_cached_sizze_149611 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_146995, &mem_146995_cached_sizze_149611, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147000_cached_sizze_149612 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_147000, &mem_147000_cached_sizze_149612, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147011_cached_sizze_149613 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147011, &mem_147011_cached_sizze_149613, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147016_cached_sizze_149614 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_147016, &mem_147016_cached_sizze_149614, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145760 = 0; i_145760 < (int64_t) 4; i_145760++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145714 = 0; i_145714 < (int64_t) 16; i_145714++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145710 = 0; i_145710 < (int64_t) 16; i_145710++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_126520;
                double r_126522 = 0.0;
                
                for (int64_t i_126521 = 0; i_126521 < (int64_t) 4; i_126521++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_126523 = ((double *) mem_146848)[i_145760 * (int64_t) 64 + i_145714 * (int64_t) 4 + i_126521];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_126524 = ((double *) mem_146847)[i_145760 * (int64_t) 64 + i_145710 * (int64_t) 4 + i_126521];
                    
                    // futhark/microgpt.fut:217:113-164
                    
                    double zt_res_126525 = zt_lhs_126523 * zt_rhs_126524;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_126526 = r_126522 + zt_res_126525;
                    double r_tmp_149183 = zp_res_126526;
                    
                    r_126522 = r_tmp_149183;
                }
                defunc_0_lifted_lambda_res_126520 = r_126522;
                ((double *) mem_146938)[i_145710] = defunc_0_lifted_lambda_res_126520;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146933, i_145714 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146938, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145722 = 0; i_145722 < (int64_t) 16; i_145722++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145718 = 0; i_145718 < (int64_t) 16; i_145718++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_126541 = ((double *) mem_146933)[i_145722 * (int64_t) 16 + i_145718];
                
                // futhark/microgpt.fut:218:47-78
                
                double zs_res_126542 = zs_lhs_126541 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_126543 = ((double *) mask_mem_146735.mem)[i_145722 * (int64_t) 16 + i_145718];
                
                // futhark/microgpt.fut:218:65-102
                
                double zp_res_126544 = zs_res_126542 + zp_rhs_126543;
                
                ((double *) mem_146954)[i_145718] = zp_res_126544;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146949, i_145722 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146954, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145740 = 0; i_145740 < (int64_t) 16; i_145740++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_134701;
            double redout_145724 = -INFINITY;
            
            for (int64_t i_145725 = 0; i_145725 < (int64_t) 16; i_145725++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_134625 = ((double *) mem_146949)[i_145740 * (int64_t) 16 + i_145725];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_126565 = fmax64(lifted_lambda_res_134625, redout_145724);
                double redout_tmp_149187 = max_res_126565;
                
                redout_145724 = redout_tmp_149187;
            }
            defunc_0_reduce_res_134701 = redout_145724;
            // futhark/microgpt.fut:220:67-76
            
            double neg_res_126566 = -defunc_0_reduce_res_134701;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145728 = 0; i_145728 < (int64_t) 16; i_145728++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_126573 = ((double *) mem_146949)[i_145740 * (int64_t) 16 + i_145728];
                
                // futhark/microgpt.fut:220:44-76
                
                double zp_res_126574 = neg_res_126566 + zp_lhs_126573;
                
                // futhark/microgpt.fut:220:37-76
                
                double exp_res_126575 = futrts_exp64(zp_res_126574);
                
                ((double *) mem_146970)[i_145728] = exp_res_126575;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_126577;
            double r_126579 = 0.0;
            
            for (int64_t i_126578 = 0; i_126578 < (int64_t) 16; i_126578++) {
                // futhark/microgpt.fut:221:36-46
                
                double lifted_lambda_res_126580 = ((double *) mem_146970)[i_126578];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_126581 = r_126579 + lifted_lambda_res_126580;
                double r_tmp_149189 = zp_res_126581;
                
                r_126579 = r_tmp_149189;
            }
            defunc_0_lifted_lambda_res_126577 = r_126579;
            // futhark/microgpt.fut:222:53-64
            
            double zs_res_126582 = 1.0 / defunc_0_lifted_lambda_res_126577;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145732 = 0; i_145732 < (int64_t) 16; i_145732++) {
                // futhark/microgpt.fut:222:37-47
                
                double zt_lhs_126589 = ((double *) mem_146970)[i_145732];
                
                // futhark/microgpt.fut:222:37-64
                
                double zt_res_126590 = zs_res_126582 * zt_lhs_126589;
                
                ((double *) mem_146977)[i_145732] = zt_res_126590;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145736 = 0; i_145736 < (int64_t) 16; i_145736++) {
                // futhark/microgpt.fut:223:4-14
                
                double lifted_lambda_res_126598 = ((double *) mem_146977)[i_145736];
                
                ((double *) mem_146984)[i_145736] = lifted_lambda_res_126598;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146965, i_145740 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146984, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145748 = 0; i_145748 < (int64_t) 16; i_145748++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145744 = 0; i_145744 < (int64_t) 4; i_145744++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_126613;
                double r_126615 = 0.0;
                
                for (int64_t i_126614 = 0; i_126614 < (int64_t) 16; i_126614++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_126616 = ((double *) mem_146965)[i_145748 * (int64_t) 16 + i_126614];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_126617 = ((double *) mem_146846)[i_145760 * (int64_t) 64 + i_126614 * (int64_t) 4 + i_145744];
                    
                    // futhark/microgpt.fut:224:66-111
                    
                    double zt_res_126618 = zt_lhs_126616 * zt_rhs_126617;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_126619 = r_126615 + zt_res_126618;
                    double r_tmp_149194 = zp_res_126619;
                    
                    r_126615 = r_tmp_149194;
                }
                defunc_0_lifted_lambda_res_126613 = r_126615;
                ((double *) mem_147000)[i_145744] = defunc_0_lifted_lambda_res_126613;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146995, i_145748 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147000, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145756 = 0; i_145756 < (int64_t) 16; i_145756++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145752 = 0; i_145752 < (int64_t) 4; i_145752++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_126634 = ((double *) mem_146995)[i_145756 * (int64_t) 4 + i_145752];
                
                ((double *) mem_147016)[i_145752] = lifted_lambda_res_126634;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147011, i_145756 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147016, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_146927, i_145760 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_147011, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147032_cached_sizze_149615 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147032, &mem_147032_cached_sizze_149615, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147037_cached_sizze_149616 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147037, &mem_147037_cached_sizze_149616, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145768 = 0; i_145768 < (int64_t) 16; i_145768++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145764 = 0; i_145764 < (int64_t) 16; i_145764++) {
            // futhark/microgpt.fut:226:54-57
            
            int64_t tmp_126646 = sdiv64(i_145764, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-59
            
            bool x_126647 = sle64((int64_t) 0, tmp_126646);
            
            // futhark/microgpt.fut:226:44-59
            
            bool y_126648 = slt64(tmp_126646, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-59
            
            bool bounds_check_126649 = x_126647 && y_126648;
            
            // futhark/microgpt.fut:226:44-59
            
            bool index_certs_126650;
            
            if (!bounds_check_126649) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_126646, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:226:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:226:15-80\n   #6  futhark/microgpt.fut:474:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:226:74-77
            
            int64_t tmp_126651 = smod64(i_145764, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-79
            
            bool x_126652 = sle64((int64_t) 0, tmp_126651);
            
            // futhark/microgpt.fut:226:44-79
            
            bool y_126653 = slt64(tmp_126651, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-79
            
            bool bounds_check_126654 = x_126652 && y_126653;
            
            // futhark/microgpt.fut:226:44-79
            
            bool index_certs_126655;
            
            if (!bounds_check_126654) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_126651, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:226:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:226:15-80\n   #6  futhark/microgpt.fut:474:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_126656 = ((double *) mem_146927)[tmp_126646 * (int64_t) 64 + i_145768 * (int64_t) 4 + tmp_126651];
            
            ((double *) mem_147037)[i_145764] = lifted_lambda_res_126656;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147032, i_145768 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147037, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147048_cached_sizze_149617 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147048, &mem_147048_cached_sizze_149617, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147053_cached_sizze_149618 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147053, &mem_147053_cached_sizze_149618, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145776 = 0; i_145776 < (int64_t) 16; i_145776++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145772 = 0; i_145772 < (int64_t) 16; i_145772++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_126671;
            double r_126673 = 0.0;
            
            for (int64_t i_126672 = 0; i_126672 < (int64_t) 16; i_126672++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_126674 = ((double *) wout_mem_146726.mem)[i_145772 * (int64_t) 16 + i_126672];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_126675 = ((double *) mem_147032)[i_145776 * (int64_t) 16 + i_126672];
                
                // futhark/microgpt.fut:227:67-106
                
                double zt_res_126676 = zt_lhs_126674 * zt_rhs_126675;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_126677 = r_126673 + zt_res_126676;
                double r_tmp_149201 = zp_res_126677;
                
                r_126673 = r_tmp_149201;
            }
            defunc_0_lifted_lambda_res_126671 = r_126673;
            ((double *) mem_147053)[i_145772] = defunc_0_lifted_lambda_res_126671;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147048, i_145776 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147053, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147064_cached_sizze_149619 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147064, &mem_147064_cached_sizze_149619, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147069_cached_sizze_149620 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147069, &mem_147069_cached_sizze_149620, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145784 = 0; i_145784 < (int64_t) 16; i_145784++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145780 = 0; i_145780 < (int64_t) 16; i_145780++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_126692 = ((double *) mem_147048)[i_145784 * (int64_t) 16 + i_145780];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_126693 = ((double *) mem_146752)[i_145784 * (int64_t) 16 + i_145780];
            
            // futhark/microgpt.fut:228:46-84
            
            double zp_res_126694 = zp_lhs_126692 + zp_rhs_126693;
            
            ((double *) mem_147069)[i_145780] = zp_res_126694;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147064, i_145784 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147069, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147080_cached_sizze_149621 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147080, &mem_147080_cached_sizze_149621, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147085_cached_sizze_149622 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147085, &mem_147085_cached_sizze_149622, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147092_cached_sizze_149623 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147092, &mem_147092_cached_sizze_149623, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145796 = 0; i_145796 < (int64_t) 16; i_145796++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_126703;
        double r_126705 = 0.0;
        
        for (int64_t i_126704 = 0; i_126704 < (int64_t) 16; i_126704++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_126706 = ((double *) mem_147064)[i_145796 * (int64_t) 16 + i_126704];
            
            // futhark/microgpt.fut:229:79-118
            
            double zt_res_126707 = zt_lhs_126706 * zt_lhs_126706;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_126708 = r_126705 + zt_res_126707;
            double r_tmp_149205 = zp_res_126708;
            
            r_126705 = r_tmp_149205;
        }
        defunc_0_lifted_lambda_res_126703 = r_126705;
        // futhark/microgpt.fut:229:58-136
        
        double zs_res_126709 = defunc_0_lifted_lambda_res_126703 / 16.0;
        
        // futhark/microgpt.fut:230:24-55
        
        double zp_res_126710 = 1.0e-5 + zs_res_126709;
        
        // futhark/microgpt.fut:230:16-55
        
        double sqrt_res_126711 = futrts_sqrt64(zp_res_126710);
        
        // futhark/microgpt.fut:231:60-71
        
        double zs_res_126712 = 1.0 / sqrt_res_126711;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145788 = 0; i_145788 < (int64_t) 16; i_145788++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_126719 = ((double *) mem_147064)[i_145796 * (int64_t) 16 + i_145788];
            
            // futhark/microgpt.fut:231:37-71
            
            double zt_res_126720 = zs_res_126712 * zt_lhs_126719;
            
            ((double *) mem_147085)[i_145788] = zt_res_126720;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145792 = 0; i_145792 < (int64_t) 16; i_145792++) {
            // futhark/microgpt.fut:232:4-14
            
            double lifted_lambda_res_126728 = ((double *) mem_147085)[i_145792];
            
            ((double *) mem_147092)[i_145792] = lifted_lambda_res_126728;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147080, i_145796 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147092, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147103_cached_sizze_149624 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147103, &mem_147103_cached_sizze_149624, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147108_cached_sizze_149625 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147108, &mem_147108_cached_sizze_149625, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145804 = 0; i_145804 < (int64_t) 16; i_145804++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145800 = 0; i_145800 < (int64_t) 64; i_145800++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_126744;
            double r_126746 = 0.0;
            
            for (int64_t i_126745 = 0; i_126745 < (int64_t) 16; i_126745++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_126747 = ((double *) wup_mem_146730.mem)[i_145800 * (int64_t) 16 + i_126745];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_126748 = ((double *) mem_147080)[i_145804 * (int64_t) 16 + i_126745];
                
                // futhark/microgpt.fut:233:67-106
                
                double zt_res_126749 = zt_lhs_126747 * zt_rhs_126748;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_126750 = r_126746 + zt_res_126749;
                double r_tmp_149210 = zp_res_126750;
                
                r_126746 = r_tmp_149210;
            }
            defunc_0_lifted_lambda_res_126744 = r_126746;
            ((double *) mem_147108)[i_145800] = defunc_0_lifted_lambda_res_126744;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147103, i_145804 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147108, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147119_cached_sizze_149626 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147119, &mem_147119_cached_sizze_149626, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147124_cached_sizze_149627 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147124, &mem_147124_cached_sizze_149627, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145812 = 0; i_145812 < (int64_t) 16; i_145812++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145808 = 0; i_145808 < (int64_t) 64; i_145808++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_126765 = ((double *) mem_147103)[i_145812 * (int64_t) 64 + i_145808];
            
            // futhark/microgpt.fut:234:45-73
            
            double max_res_126766 = fmax64(0.0, max_arg0_126765);
            
            ((double *) mem_147124)[i_145808] = max_res_126766;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147119, i_145812 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147124, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147135_cached_sizze_149628 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147135, &mem_147135_cached_sizze_149628, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147140_cached_sizze_149629 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147140, &mem_147140_cached_sizze_149629, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145820 = 0; i_145820 < (int64_t) 16; i_145820++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145816 = 0; i_145816 < (int64_t) 16; i_145816++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_126781;
            double r_126783 = 0.0;
            
            for (int64_t i_126782 = 0; i_126782 < (int64_t) 64; i_126782++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_126784 = ((double *) wdown_mem_146724.mem)[i_145816 * (int64_t) 64 + i_126782];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_126785 = ((double *) mem_147119)[i_145820 * (int64_t) 64 + i_126782];
                
                // futhark/microgpt.fut:235:67-108
                
                double zt_res_126786 = zt_lhs_126784 * zt_rhs_126785;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_126787 = r_126783 + zt_res_126786;
                double r_tmp_149215 = zp_res_126787;
                
                r_126783 = r_tmp_149215;
            }
            defunc_0_lifted_lambda_res_126781 = r_126783;
            ((double *) mem_147140)[i_145816] = defunc_0_lifted_lambda_res_126781;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147135, i_145820 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147140, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147151_cached_sizze_149630 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147151, &mem_147151_cached_sizze_149630, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147156_cached_sizze_149631 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147156, &mem_147156_cached_sizze_149631, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145828 = 0; i_145828 < (int64_t) 16; i_145828++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145824 = 0; i_145824 < (int64_t) 16; i_145824++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_126802 = ((double *) mem_147135)[i_145828 * (int64_t) 16 + i_145824];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_126803 = ((double *) mem_147064)[i_145828 * (int64_t) 16 + i_145824];
            
            // futhark/microgpt.fut:236:46-85
            
            double zp_res_126804 = zp_lhs_126802 + zp_rhs_126803;
            
            ((double *) mem_147156)[i_145824] = zp_res_126804;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147151, i_145828 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147156, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147167_cached_sizze_149632 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_147167, &mem_147167_cached_sizze_149632, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147172_cached_sizze_149633 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_147172, &mem_147172_cached_sizze_149633, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145836 = 0; i_145836 < (int64_t) 16; i_145836++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145832 = 0; i_145832 < (int64_t) 27; i_145832++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_126820;
            double r_126822 = 0.0;
            
            for (int64_t i_126821 = 0; i_126821 < (int64_t) 16; i_126821++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_126823 = ((double *) wvoc_mem_146732.mem)[i_145832 * (int64_t) 16 + i_126821];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_126824 = ((double *) mem_147151)[i_145836 * (int64_t) 16 + i_126821];
                
                // futhark/microgpt.fut:237:67-107
                
                double zt_res_126825 = zt_lhs_126823 * zt_rhs_126824;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_126826 = r_126822 + zt_res_126825;
                double r_tmp_149220 = zp_res_126826;
                
                r_126822 = r_tmp_149220;
            }
            defunc_0_lifted_lambda_res_126820 = r_126822;
            ((double *) mem_147172)[i_145832] = defunc_0_lifted_lambda_res_126820;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147167, i_145836 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147172, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_147183, (int64_t) 128, "mem_147183")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147187_cached_sizze_149634 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_147187, &mem_147187_cached_sizze_149634, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147194_cached_sizze_149635 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_147194, &mem_147194_cached_sizze_149635, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145850 = 0; i_145850 < (int64_t) 16; i_145850++) {
        double x_134724;
        double redout_145838 = -INFINITY;
        
        for (int64_t i_145839 = 0; i_145839 < (int64_t) 27; i_145839++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_134671 = ((double *) mem_147167)[i_145850 * (int64_t) 27 + i_145839];
            
            // futhark/microgpt.fut:115:13-33
            
            double max_res_126850 = fmax64(lifted_lambda_res_134671, redout_145838);
            double redout_tmp_149222 = max_res_126850;
            
            redout_145838 = redout_tmp_149222;
        }
        x_134724 = redout_145838;
        // futhark/microgpt.fut:239:67-76
        
        double neg_res_126851 = -x_134724;
        
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_126835;
        double r_126837 = 0.0;
        
        for (int64_t i_126836 = 0; i_126836 < (int64_t) 27; i_126836++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145842 = 0; i_145842 < (int64_t) 27; i_145842++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_126858 = ((double *) mem_147167)[i_145850 * (int64_t) 27 + i_145842];
                
                // futhark/microgpt.fut:239:44-76
                
                double zp_res_126859 = neg_res_126851 + zp_lhs_126858;
                
                // futhark/microgpt.fut:239:37-76
                
                double exp_res_126860 = futrts_exp64(zp_res_126859);
                
                ((double *) mem_147187)[i_145842] = exp_res_126860;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_126862;
            double r_126864 = 0.0;
            
            for (int64_t i_126863 = 0; i_126863 < (int64_t) 27; i_126863++) {
                // futhark/microgpt.fut:240:36-46
                
                double lifted_lambda_res_126865 = ((double *) mem_147187)[i_126863];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_126866 = r_126864 + lifted_lambda_res_126865;
                double r_tmp_149225 = zp_res_126866;
                
                r_126864 = r_tmp_149225;
            }
            defunc_0_lifted_lambda_res_126862 = r_126864;
            // futhark/microgpt.fut:241:53-64
            
            double zs_res_126867 = 1.0 / defunc_0_lifted_lambda_res_126862;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145846 = 0; i_145846 < (int64_t) 27; i_145846++) {
                // futhark/microgpt.fut:241:37-47
                
                double zt_lhs_126874 = ((double *) mem_147187)[i_145846];
                
                // futhark/microgpt.fut:241:37-64
                
                double zt_res_126875 = zs_res_126867 * zt_lhs_126874;
                
                ((double *) mem_147194)[i_145846] = zt_res_126875;
            }
            // futhark/microgpt.fut:242:12-22
            
            double log_arg0_126877 = ((double *) mem_147194)[i_126836];
            
            // futhark/microgpt.fut:242:6-22
            
            double log_res_126878 = futrts_log64(log_arg0_126877);
            
            // futhark/microgpt.fut:71:46-49
            
            double zt_rhs_126879 = ((double *) target_mem_146734.mem)[i_145850 * (int64_t) 27 + i_126836];
            
            // futhark/microgpt.fut:242:6-48
            
            double zt_res_126880 = log_res_126878 * zt_rhs_126879;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_126881 = r_126837 + zt_res_126880;
            double r_tmp_149223 = zp_res_126881;
            
            r_126837 = r_tmp_149223;
        }
        defunc_0_lifted_lambda_res_126835 = r_126837;
        // futhark/microgpt.fut:238:37-242:54
        
        double neg_res_126882 = -defunc_0_lifted_lambda_res_126835;
        
        ((double *) mem_147183.mem)[i_145850] = neg_res_126882;
    }
    // futhark/microgpt.fut:71:13-49
    
    double defunc_0_lifted_lambda_res_126884;
    double r_126886 = 0.0;
    
    for (int64_t i_126885 = 0; i_126885 < (int64_t) 16; i_126885++) {
        // futhark/microgpt.fut:243:37-47
        
        double lifted_lambda_res_126887 = ((double *) mem_147183.mem)[i_126885];
        
        // futhark/microgpt.fut:71:40-49
        
        double zp_res_126888 = r_126886 + lifted_lambda_res_126887;
        double r_tmp_149227 = zp_res_126888;
        
        r_126886 = r_tmp_149227;
    }
    defunc_0_lifted_lambda_res_126884 = r_126886;
    // futhark/microgpt.fut:243:17-64
    
    double zs_res_126889 = defunc_0_lifted_lambda_res_126884 / 16.0;
    
    if (memblock_set(ctx, &mem_out_149150, &mem_147183, "mem_147183") != 0)
        return 1;
    prim_out_149151 = zs_res_126889;
    if (memblock_set(ctx, &*mem_out_p_149577, &mem_out_149150, "mem_out_149150") != 0)
        return 1;
    *out_prim_out_149578 = prim_out_149151;
    
  cleanup:
    {
        free(mem_146736);
        free(mem_146741);
        free(mem_146752);
        free(mem_146757);
        free(mem_146764);
        free(mem_146775);
        free(mem_146780);
        free(mem_146787);
        free(mem_146798);
        free(mem_146799);
        free(mem_146800);
        free(mem_146813);
        free(mem_146814);
        free(mem_146815);
        free(mem_146846);
        free(mem_146847);
        free(mem_146848);
        free(mem_146864);
        free(mem_146865);
        free(mem_146866);
        free(mem_146879);
        free(mem_146880);
        free(mem_146881);
        free(mem_146927);
        free(mem_146933);
        free(mem_146938);
        free(mem_146949);
        free(mem_146954);
        free(mem_146965);
        free(mem_146970);
        free(mem_146977);
        free(mem_146984);
        free(mem_146995);
        free(mem_147000);
        free(mem_147011);
        free(mem_147016);
        free(mem_147032);
        free(mem_147037);
        free(mem_147048);
        free(mem_147053);
        free(mem_147064);
        free(mem_147069);
        free(mem_147080);
        free(mem_147085);
        free(mem_147092);
        free(mem_147103);
        free(mem_147108);
        free(mem_147119);
        free(mem_147124);
        free(mem_147135);
        free(mem_147140);
        free(mem_147151);
        free(mem_147156);
        free(mem_147167);
        free(mem_147172);
        free(mem_147187);
        free(mem_147194);
        if (memblock_unref(ctx, &mem_147183, "mem_147183") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149150, "mem_out_149150") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_149636, struct memblock wdown_mem_146724, struct memblock wkey_mem_146725, struct memblock wout_mem_146726, struct memblock wpe_mem_146727, struct memblock wqry_mem_146728, struct memblock wte_mem_146729, struct memblock wup_mem_146730, struct memblock wval_mem_146731, struct memblock wvoc_mem_146732, struct memblock tokens_mem_146733, struct memblock mask_mem_146734)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_146735_cached_sizze_149637 = 0;
    unsigned char *mem_146735 = NULL;
    int64_t mem_146740_cached_sizze_149638 = 0;
    unsigned char *mem_146740 = NULL;
    int64_t mem_146751_cached_sizze_149639 = 0;
    unsigned char *mem_146751 = NULL;
    int64_t mem_146756_cached_sizze_149640 = 0;
    unsigned char *mem_146756 = NULL;
    int64_t mem_146763_cached_sizze_149641 = 0;
    unsigned char *mem_146763 = NULL;
    int64_t mem_146774_cached_sizze_149642 = 0;
    unsigned char *mem_146774 = NULL;
    int64_t mem_146779_cached_sizze_149643 = 0;
    unsigned char *mem_146779 = NULL;
    int64_t mem_146786_cached_sizze_149644 = 0;
    unsigned char *mem_146786 = NULL;
    int64_t mem_146797_cached_sizze_149645 = 0;
    unsigned char *mem_146797 = NULL;
    int64_t mem_146798_cached_sizze_149646 = 0;
    unsigned char *mem_146798 = NULL;
    int64_t mem_146799_cached_sizze_149647 = 0;
    unsigned char *mem_146799 = NULL;
    int64_t mem_146812_cached_sizze_149648 = 0;
    unsigned char *mem_146812 = NULL;
    int64_t mem_146813_cached_sizze_149649 = 0;
    unsigned char *mem_146813 = NULL;
    int64_t mem_146814_cached_sizze_149650 = 0;
    unsigned char *mem_146814 = NULL;
    int64_t mem_146845_cached_sizze_149651 = 0;
    unsigned char *mem_146845 = NULL;
    int64_t mem_146846_cached_sizze_149652 = 0;
    unsigned char *mem_146846 = NULL;
    int64_t mem_146847_cached_sizze_149653 = 0;
    unsigned char *mem_146847 = NULL;
    int64_t mem_146863_cached_sizze_149654 = 0;
    unsigned char *mem_146863 = NULL;
    int64_t mem_146864_cached_sizze_149655 = 0;
    unsigned char *mem_146864 = NULL;
    int64_t mem_146865_cached_sizze_149656 = 0;
    unsigned char *mem_146865 = NULL;
    int64_t mem_146878_cached_sizze_149657 = 0;
    unsigned char *mem_146878 = NULL;
    int64_t mem_146879_cached_sizze_149658 = 0;
    unsigned char *mem_146879 = NULL;
    int64_t mem_146880_cached_sizze_149659 = 0;
    unsigned char *mem_146880 = NULL;
    int64_t mem_146926_cached_sizze_149660 = 0;
    unsigned char *mem_146926 = NULL;
    int64_t mem_146932_cached_sizze_149661 = 0;
    unsigned char *mem_146932 = NULL;
    int64_t mem_146937_cached_sizze_149662 = 0;
    unsigned char *mem_146937 = NULL;
    int64_t mem_146948_cached_sizze_149663 = 0;
    unsigned char *mem_146948 = NULL;
    int64_t mem_146953_cached_sizze_149664 = 0;
    unsigned char *mem_146953 = NULL;
    int64_t mem_146964_cached_sizze_149665 = 0;
    unsigned char *mem_146964 = NULL;
    int64_t mem_146969_cached_sizze_149666 = 0;
    unsigned char *mem_146969 = NULL;
    int64_t mem_146976_cached_sizze_149667 = 0;
    unsigned char *mem_146976 = NULL;
    int64_t mem_146983_cached_sizze_149668 = 0;
    unsigned char *mem_146983 = NULL;
    int64_t mem_146994_cached_sizze_149669 = 0;
    unsigned char *mem_146994 = NULL;
    int64_t mem_146999_cached_sizze_149670 = 0;
    unsigned char *mem_146999 = NULL;
    int64_t mem_147010_cached_sizze_149671 = 0;
    unsigned char *mem_147010 = NULL;
    int64_t mem_147015_cached_sizze_149672 = 0;
    unsigned char *mem_147015 = NULL;
    int64_t mem_147031_cached_sizze_149673 = 0;
    unsigned char *mem_147031 = NULL;
    int64_t mem_147036_cached_sizze_149674 = 0;
    unsigned char *mem_147036 = NULL;
    int64_t mem_147047_cached_sizze_149675 = 0;
    unsigned char *mem_147047 = NULL;
    int64_t mem_147052_cached_sizze_149676 = 0;
    unsigned char *mem_147052 = NULL;
    int64_t mem_147063_cached_sizze_149677 = 0;
    unsigned char *mem_147063 = NULL;
    int64_t mem_147068_cached_sizze_149678 = 0;
    unsigned char *mem_147068 = NULL;
    int64_t mem_147079_cached_sizze_149679 = 0;
    unsigned char *mem_147079 = NULL;
    int64_t mem_147084_cached_sizze_149680 = 0;
    unsigned char *mem_147084 = NULL;
    int64_t mem_147091_cached_sizze_149681 = 0;
    unsigned char *mem_147091 = NULL;
    int64_t mem_147102_cached_sizze_149682 = 0;
    unsigned char *mem_147102 = NULL;
    int64_t mem_147107_cached_sizze_149683 = 0;
    unsigned char *mem_147107 = NULL;
    int64_t mem_147118_cached_sizze_149684 = 0;
    unsigned char *mem_147118 = NULL;
    int64_t mem_147123_cached_sizze_149685 = 0;
    unsigned char *mem_147123 = NULL;
    int64_t mem_147134_cached_sizze_149686 = 0;
    unsigned char *mem_147134 = NULL;
    int64_t mem_147139_cached_sizze_149687 = 0;
    unsigned char *mem_147139 = NULL;
    int64_t mem_147150_cached_sizze_149688 = 0;
    unsigned char *mem_147150 = NULL;
    int64_t mem_147155_cached_sizze_149689 = 0;
    unsigned char *mem_147155 = NULL;
    int64_t mem_147166_cached_sizze_149690 = 0;
    unsigned char *mem_147166 = NULL;
    int64_t mem_147171_cached_sizze_149691 = 0;
    unsigned char *mem_147171 = NULL;
    int64_t mem_147187_cached_sizze_149692 = 0;
    unsigned char *mem_147187 = NULL;
    struct memblock mem_147182;
    
    mem_147182.references = NULL;
    
    struct memblock mem_out_149150;
    
    mem_out_149150.references = NULL;
    
    struct memblock mem_146715 = ctx->constants->mem_146715;
    struct memblock mem_146716 = ctx->constants->mem_146716;
    struct memblock mem_146717 = ctx->constants->mem_146717;
    struct memblock mem_146718 = ctx->constants->mem_146718;
    struct memblock mem_146719 = ctx->constants->mem_146719;
    struct memblock mem_146720 = ctx->constants->mem_146720;
    struct memblock mem_146721 = ctx->constants->mem_146721;
    struct memblock mem_146722 = ctx->constants->mem_146722;
    struct memblock mem_146723 = ctx->constants->mem_146723;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_146735_cached_sizze_149637 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146735, &mem_146735_cached_sizze_149637, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146740_cached_sizze_149638 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146740, &mem_146740_cached_sizze_149638, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145632 = 0; i_145632 < (int64_t) 16; i_145632++) {
        // futhark/microgpt.fut:468:41-50
        
        int64_t tmp_126274 = ((int64_t *) tokens_mem_146733.mem)[i_145632];
        
        // futhark/microgpt.fut:468:37-51
        
        bool x_126275 = sle64((int64_t) 0, tmp_126274);
        
        // futhark/microgpt.fut:468:37-51
        
        bool y_126276 = slt64(tmp_126274, (int64_t) 27);
        
        // futhark/microgpt.fut:468:37-51
        
        bool bounds_check_126277 = x_126275 && y_126276;
        
        // futhark/microgpt.fut:468:37-51
        
        bool index_certs_126278;
        
        if (!bounds_check_126277) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_126274, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:468:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:468:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145628 = 0; i_145628 < (int64_t) 16; i_145628++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_126285 = ((double *) wte_mem_146729.mem)[tmp_126274 * (int64_t) 16 + i_145628];
            
            ((double *) mem_146740)[i_145628] = lifted_lambda_res_126285;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_146735, i_145632 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146740, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146751_cached_sizze_149639 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146751, &mem_146751_cached_sizze_149639, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146756_cached_sizze_149640 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146756, &mem_146756_cached_sizze_149640, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146763_cached_sizze_149641 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146763, &mem_146763_cached_sizze_149641, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145644 = 0; i_145644 < (int64_t) 16; i_145644++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_126311;
        double r_126313 = 0.0;
        
        for (int64_t i_126312 = 0; i_126312 < (int64_t) 16; i_126312++) {
            // futhark/microgpt.fut:71:46-49
            
            double zp_lhs_126314 = ((double *) wpe_mem_146727.mem)[i_145644 * (int64_t) 16 + i_126312];
            
            // futhark/microgpt.fut:71:46-49
            
            double zp_rhs_126315 = ((double *) mem_146735)[i_145644 * (int64_t) 16 + i_126312];
            
            // futhark/microgpt.fut:148:76-116
            
            double zp_res_126316 = zp_lhs_126314 + zp_rhs_126315;
            
            // futhark/microgpt.fut:148:94-163
            
            double zt_res_126317 = zp_res_126316 * zp_res_126316;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_126318 = r_126313 + zt_res_126317;
            double r_tmp_149154 = zp_res_126318;
            
            r_126313 = r_tmp_149154;
        }
        defunc_0_lifted_lambda_res_126311 = r_126313;
        // futhark/microgpt.fut:148:54-182
        
        double zs_res_126319 = defunc_0_lifted_lambda_res_126311 / 16.0;
        
        // futhark/microgpt.fut:149:24-55
        
        double zp_res_126320 = 1.0e-5 + zs_res_126319;
        
        // futhark/microgpt.fut:149:16-55
        
        double sqrt_res_126321 = futrts_sqrt64(zp_res_126320);
        
        // futhark/microgpt.fut:150:85-96
        
        double zs_res_126322 = 1.0 / sqrt_res_126321;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145636 = 0; i_145636 < (int64_t) 16; i_145636++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_126329 = ((double *) wpe_mem_146727.mem)[i_145644 * (int64_t) 16 + i_145636];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_126330 = ((double *) mem_146735)[i_145644 * (int64_t) 16 + i_145636];
            
            // futhark/microgpt.fut:150:38-78
            
            double zp_res_126331 = zp_lhs_126329 + zp_rhs_126330;
            
            // futhark/microgpt.fut:150:56-96
            
            double zt_res_126332 = zs_res_126322 * zp_res_126331;
            
            ((double *) mem_146756)[i_145636] = zt_res_126332;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145640 = 0; i_145640 < (int64_t) 16; i_145640++) {
            // futhark/microgpt.fut:151:4-14
            
            double lifted_lambda_res_126340 = ((double *) mem_146756)[i_145640];
            
            ((double *) mem_146763)[i_145640] = lifted_lambda_res_126340;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_146751, i_145644 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146763, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146774_cached_sizze_149642 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146774, &mem_146774_cached_sizze_149642, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146779_cached_sizze_149643 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146779, &mem_146779_cached_sizze_149643, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146786_cached_sizze_149644 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146786, &mem_146786_cached_sizze_149644, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145656 = 0; i_145656 < (int64_t) 16; i_145656++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_126349;
        double r_126351 = 0.0;
        
        for (int64_t i_126350 = 0; i_126350 < (int64_t) 16; i_126350++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_126352 = ((double *) mem_146751)[i_145656 * (int64_t) 16 + i_126350];
            
            // futhark/microgpt.fut:152:78-115
            
            double zt_res_126353 = zt_lhs_126352 * zt_lhs_126352;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_126354 = r_126351 + zt_res_126353;
            double r_tmp_149158 = zp_res_126354;
            
            r_126351 = r_tmp_149158;
        }
        defunc_0_lifted_lambda_res_126349 = r_126351;
        // futhark/microgpt.fut:152:57-133
        
        double zs_res_126355 = defunc_0_lifted_lambda_res_126349 / 16.0;
        
        // futhark/microgpt.fut:153:24-55
        
        double zp_res_126356 = 1.0e-5 + zs_res_126355;
        
        // futhark/microgpt.fut:153:16-55
        
        double sqrt_res_126357 = futrts_sqrt64(zp_res_126356);
        
        // futhark/microgpt.fut:154:59-70
        
        double zs_res_126358 = 1.0 / sqrt_res_126357;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145648 = 0; i_145648 < (int64_t) 16; i_145648++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_126365 = ((double *) mem_146751)[i_145656 * (int64_t) 16 + i_145648];
            
            // futhark/microgpt.fut:154:37-70
            
            double zt_res_126366 = zs_res_126358 * zt_lhs_126365;
            
            ((double *) mem_146779)[i_145648] = zt_res_126366;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145652 = 0; i_145652 < (int64_t) 16; i_145652++) {
            // futhark/microgpt.fut:155:4-14
            
            double lifted_lambda_res_126374 = ((double *) mem_146779)[i_145652];
            
            ((double *) mem_146786)[i_145652] = lifted_lambda_res_126374;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_146774, i_145656 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146786, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146797_cached_sizze_149645 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146797, &mem_146797_cached_sizze_149645, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146798_cached_sizze_149646 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146798, &mem_146798_cached_sizze_149646, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146799_cached_sizze_149647 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146799, &mem_146799_cached_sizze_149647, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146812_cached_sizze_149648 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146812, &mem_146812_cached_sizze_149648, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146813_cached_sizze_149649 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146813, &mem_146813_cached_sizze_149649, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146814_cached_sizze_149650 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146814, &mem_146814_cached_sizze_149650, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145674 = 0; i_145674 < (int64_t) 16; i_145674++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145664 = 0; i_145664 < (int64_t) 16; i_145664++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_134381;
            double r_134383 = 0.0;
            
            for (int64_t i_134382 = 0; i_134382 < (int64_t) 16; i_134382++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_134384 = ((double *) wqry_mem_146728.mem)[i_145664 * (int64_t) 16 + i_134382];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_134385 = ((double *) mem_146774)[i_145674 * (int64_t) 16 + i_134382];
                
                // futhark/microgpt.fut:156:66-105
                
                double zt_res_134386 = zt_lhs_134384 * zt_rhs_134385;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_134387 = r_134383 + zt_res_134386;
                double r_tmp_149167 = zp_res_134387;
                
                r_134383 = r_tmp_149167;
            }
            defunc_0_lifted_lambda_res_134381 = r_134383;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_134394;
            double r_134396 = 0.0;
            
            for (int64_t i_134395 = 0; i_134395 < (int64_t) 16; i_134395++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_134397 = ((double *) wkey_mem_146725.mem)[i_145664 * (int64_t) 16 + i_134395];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_134398 = ((double *) mem_146774)[i_145674 * (int64_t) 16 + i_134395];
                
                // futhark/microgpt.fut:157:66-105
                
                double zt_res_134399 = zt_lhs_134397 * zt_rhs_134398;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_134400 = r_134396 + zt_res_134399;
                double r_tmp_149168 = zp_res_134400;
                
                r_134396 = r_tmp_149168;
            }
            defunc_0_lifted_lambda_res_134394 = r_134396;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_134410;
            double r_134412 = 0.0;
            
            for (int64_t i_134411 = 0; i_134411 < (int64_t) 16; i_134411++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_134413 = ((double *) wval_mem_146731.mem)[i_145664 * (int64_t) 16 + i_134411];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_134414 = ((double *) mem_146774)[i_145674 * (int64_t) 16 + i_134411];
                
                // futhark/microgpt.fut:158:66-105
                
                double zt_res_134415 = zt_lhs_134413 * zt_rhs_134414;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_134416 = r_134412 + zt_res_134415;
                double r_tmp_149169 = zp_res_134416;
                
                r_134412 = r_tmp_149169;
            }
            defunc_0_lifted_lambda_res_134410 = r_134412;
            ((double *) mem_146812)[i_145664] = defunc_0_lifted_lambda_res_134410;
            ((double *) mem_146813)[i_145664] = defunc_0_lifted_lambda_res_134394;
            ((double *) mem_146814)[i_145664] = defunc_0_lifted_lambda_res_134381;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_146797, i_145674 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146812, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_146798, i_145674 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146813, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_146799, i_145674 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146814, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146845_cached_sizze_149651 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146845, &mem_146845_cached_sizze_149651, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146846_cached_sizze_149652 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146846, &mem_146846_cached_sizze_149652, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146847_cached_sizze_149653 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146847, &mem_146847_cached_sizze_149653, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146863_cached_sizze_149654 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_146863, &mem_146863_cached_sizze_149654, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146864_cached_sizze_149655 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_146864, &mem_146864_cached_sizze_149655, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146865_cached_sizze_149656 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_146865, &mem_146865_cached_sizze_149656, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146878_cached_sizze_149657 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_146878, &mem_146878_cached_sizze_149657, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146879_cached_sizze_149658 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_146879, &mem_146879_cached_sizze_149658, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146880_cached_sizze_149659 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_146880, &mem_146880_cached_sizze_149659, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145704 = 0; i_145704 < (int64_t) 4; i_145704++) {
        // futhark/microgpt.fut:159:69-72
        
        int64_t zp_lhs_134257 = mul64((int64_t) 4, i_145704);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145694 = 0; i_145694 < (int64_t) 16; i_145694++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145684 = 0; i_145684 < (int64_t) 4; i_145684++) {
                // futhark/microgpt.fut:159:74-81
                
                int64_t tmp_134574 = add64(zp_lhs_134257, i_145684);
                
                // futhark/microgpt.fut:159:51-83
                
                bool x_134575 = sle64((int64_t) 0, tmp_134574);
                
                // futhark/microgpt.fut:159:51-83
                
                bool y_134576 = slt64(tmp_134574, (int64_t) 16);
                
                // futhark/microgpt.fut:159:51-83
                
                bool bounds_check_134577 = x_134575 && y_134576;
                
                // futhark/microgpt.fut:159:51-83
                
                bool index_certs_134578;
                
                if (!bounds_check_134577) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_134574, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:159:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:159:15-84\n   #9  futhark/microgpt.fut:469:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_134579 = ((double *) mem_146799)[i_145694 * (int64_t) 16 + tmp_134574];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_134587 = ((double *) mem_146798)[i_145694 * (int64_t) 16 + tmp_134574];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_134598 = ((double *) mem_146797)[i_145694 * (int64_t) 16 + tmp_134574];
                
                ((double *) mem_146878)[i_145684] = lifted_lambda_res_134598;
                ((double *) mem_146879)[i_145684] = lifted_lambda_res_134587;
                ((double *) mem_146880)[i_145684] = lifted_lambda_res_134579;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146863, i_145694 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146878, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146864, i_145694 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146879, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146865, i_145694 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146880, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_146845, i_145704 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_146863, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_146846, i_145704 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_146864, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_146847, i_145704 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_146865, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146926_cached_sizze_149660 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146926, &mem_146926_cached_sizze_149660, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146932_cached_sizze_149661 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146932, &mem_146932_cached_sizze_149661, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146937_cached_sizze_149662 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146937, &mem_146937_cached_sizze_149662, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146948_cached_sizze_149663 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146948, &mem_146948_cached_sizze_149663, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146953_cached_sizze_149664 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146953, &mem_146953_cached_sizze_149664, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146964_cached_sizze_149665 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146964, &mem_146964_cached_sizze_149665, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146969_cached_sizze_149666 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146969, &mem_146969_cached_sizze_149666, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146976_cached_sizze_149667 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146976, &mem_146976_cached_sizze_149667, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146983_cached_sizze_149668 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146983, &mem_146983_cached_sizze_149668, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146994_cached_sizze_149669 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_146994, &mem_146994_cached_sizze_149669, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146999_cached_sizze_149670 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_146999, &mem_146999_cached_sizze_149670, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147010_cached_sizze_149671 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147010, &mem_147010_cached_sizze_149671, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147015_cached_sizze_149672 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_147015, &mem_147015_cached_sizze_149672, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145760 = 0; i_145760 < (int64_t) 4; i_145760++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145714 = 0; i_145714 < (int64_t) 16; i_145714++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145710 = 0; i_145710 < (int64_t) 16; i_145710++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_126519;
                double r_126521 = 0.0;
                
                for (int64_t i_126520 = 0; i_126520 < (int64_t) 4; i_126520++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_126522 = ((double *) mem_146847)[i_145760 * (int64_t) 64 + i_145714 * (int64_t) 4 + i_126520];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_126523 = ((double *) mem_146846)[i_145760 * (int64_t) 64 + i_145710 * (int64_t) 4 + i_126520];
                    
                    // futhark/microgpt.fut:162:113-164
                    
                    double zt_res_126524 = zt_lhs_126522 * zt_rhs_126523;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_126525 = r_126521 + zt_res_126524;
                    double r_tmp_149182 = zp_res_126525;
                    
                    r_126521 = r_tmp_149182;
                }
                defunc_0_lifted_lambda_res_126519 = r_126521;
                ((double *) mem_146937)[i_145710] = defunc_0_lifted_lambda_res_126519;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146932, i_145714 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146937, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145722 = 0; i_145722 < (int64_t) 16; i_145722++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145718 = 0; i_145718 < (int64_t) 16; i_145718++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_126540 = ((double *) mem_146932)[i_145722 * (int64_t) 16 + i_145718];
                
                // futhark/microgpt.fut:163:47-78
                
                double zs_res_126541 = zs_lhs_126540 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_126542 = ((double *) mask_mem_146734.mem)[i_145722 * (int64_t) 16 + i_145718];
                
                // futhark/microgpt.fut:163:65-102
                
                double zp_res_126543 = zs_res_126541 + zp_rhs_126542;
                
                ((double *) mem_146953)[i_145718] = zp_res_126543;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146948, i_145722 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146953, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145740 = 0; i_145740 < (int64_t) 16; i_145740++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_134676;
            double redout_145724 = -INFINITY;
            
            for (int64_t i_145725 = 0; i_145725 < (int64_t) 16; i_145725++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_134625 = ((double *) mem_146948)[i_145740 * (int64_t) 16 + i_145725];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_126564 = fmax64(lifted_lambda_res_134625, redout_145724);
                double redout_tmp_149186 = max_res_126564;
                
                redout_145724 = redout_tmp_149186;
            }
            defunc_0_reduce_res_134676 = redout_145724;
            // futhark/microgpt.fut:165:67-76
            
            double neg_res_126565 = -defunc_0_reduce_res_134676;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145728 = 0; i_145728 < (int64_t) 16; i_145728++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_126572 = ((double *) mem_146948)[i_145740 * (int64_t) 16 + i_145728];
                
                // futhark/microgpt.fut:165:44-76
                
                double zp_res_126573 = neg_res_126565 + zp_lhs_126572;
                
                // futhark/microgpt.fut:165:37-76
                
                double exp_res_126574 = futrts_exp64(zp_res_126573);
                
                ((double *) mem_146969)[i_145728] = exp_res_126574;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_126576;
            double r_126578 = 0.0;
            
            for (int64_t i_126577 = 0; i_126577 < (int64_t) 16; i_126577++) {
                // futhark/microgpt.fut:166:36-46
                
                double lifted_lambda_res_126579 = ((double *) mem_146969)[i_126577];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_126580 = r_126578 + lifted_lambda_res_126579;
                double r_tmp_149188 = zp_res_126580;
                
                r_126578 = r_tmp_149188;
            }
            defunc_0_lifted_lambda_res_126576 = r_126578;
            // futhark/microgpt.fut:167:53-64
            
            double zs_res_126581 = 1.0 / defunc_0_lifted_lambda_res_126576;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145732 = 0; i_145732 < (int64_t) 16; i_145732++) {
                // futhark/microgpt.fut:167:37-47
                
                double zt_lhs_126588 = ((double *) mem_146969)[i_145732];
                
                // futhark/microgpt.fut:167:37-64
                
                double zt_res_126589 = zs_res_126581 * zt_lhs_126588;
                
                ((double *) mem_146976)[i_145732] = zt_res_126589;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145736 = 0; i_145736 < (int64_t) 16; i_145736++) {
                // futhark/microgpt.fut:168:4-14
                
                double lifted_lambda_res_126597 = ((double *) mem_146976)[i_145736];
                
                ((double *) mem_146983)[i_145736] = lifted_lambda_res_126597;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146964, i_145740 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146983, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145748 = 0; i_145748 < (int64_t) 16; i_145748++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145744 = 0; i_145744 < (int64_t) 4; i_145744++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_126612;
                double r_126614 = 0.0;
                
                for (int64_t i_126613 = 0; i_126613 < (int64_t) 16; i_126613++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_126615 = ((double *) mem_146964)[i_145748 * (int64_t) 16 + i_126613];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_126616 = ((double *) mem_146845)[i_145760 * (int64_t) 64 + i_126613 * (int64_t) 4 + i_145744];
                    
                    // futhark/microgpt.fut:169:66-111
                    
                    double zt_res_126617 = zt_lhs_126615 * zt_rhs_126616;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_126618 = r_126614 + zt_res_126617;
                    double r_tmp_149193 = zp_res_126618;
                    
                    r_126614 = r_tmp_149193;
                }
                defunc_0_lifted_lambda_res_126612 = r_126614;
                ((double *) mem_146999)[i_145744] = defunc_0_lifted_lambda_res_126612;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146994, i_145748 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146999, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145756 = 0; i_145756 < (int64_t) 16; i_145756++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145752 = 0; i_145752 < (int64_t) 4; i_145752++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_126633 = ((double *) mem_146994)[i_145756 * (int64_t) 4 + i_145752];
                
                ((double *) mem_147015)[i_145752] = lifted_lambda_res_126633;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147010, i_145756 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147015, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_146926, i_145760 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_147010, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147031_cached_sizze_149673 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147031, &mem_147031_cached_sizze_149673, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147036_cached_sizze_149674 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147036, &mem_147036_cached_sizze_149674, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145768 = 0; i_145768 < (int64_t) 16; i_145768++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145764 = 0; i_145764 < (int64_t) 16; i_145764++) {
            // futhark/microgpt.fut:171:54-57
            
            int64_t tmp_126645 = sdiv64(i_145764, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-59
            
            bool x_126646 = sle64((int64_t) 0, tmp_126645);
            
            // futhark/microgpt.fut:171:44-59
            
            bool y_126647 = slt64(tmp_126645, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-59
            
            bool bounds_check_126648 = x_126646 && y_126647;
            
            // futhark/microgpt.fut:171:44-59
            
            bool index_certs_126649;
            
            if (!bounds_check_126648) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_126645, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:171:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:171:15-80\n   #6  futhark/microgpt.fut:469:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:171:74-77
            
            int64_t tmp_126650 = smod64(i_145764, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-79
            
            bool x_126651 = sle64((int64_t) 0, tmp_126650);
            
            // futhark/microgpt.fut:171:44-79
            
            bool y_126652 = slt64(tmp_126650, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-79
            
            bool bounds_check_126653 = x_126651 && y_126652;
            
            // futhark/microgpt.fut:171:44-79
            
            bool index_certs_126654;
            
            if (!bounds_check_126653) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_126650, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:171:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:171:15-80\n   #6  futhark/microgpt.fut:469:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_126655 = ((double *) mem_146926)[tmp_126645 * (int64_t) 64 + i_145768 * (int64_t) 4 + tmp_126650];
            
            ((double *) mem_147036)[i_145764] = lifted_lambda_res_126655;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147031, i_145768 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147036, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147047_cached_sizze_149675 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147047, &mem_147047_cached_sizze_149675, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147052_cached_sizze_149676 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147052, &mem_147052_cached_sizze_149676, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145776 = 0; i_145776 < (int64_t) 16; i_145776++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145772 = 0; i_145772 < (int64_t) 16; i_145772++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_126670;
            double r_126672 = 0.0;
            
            for (int64_t i_126671 = 0; i_126671 < (int64_t) 16; i_126671++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_126673 = ((double *) wout_mem_146726.mem)[i_145772 * (int64_t) 16 + i_126671];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_126674 = ((double *) mem_147031)[i_145776 * (int64_t) 16 + i_126671];
                
                // futhark/microgpt.fut:172:67-106
                
                double zt_res_126675 = zt_lhs_126673 * zt_rhs_126674;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_126676 = r_126672 + zt_res_126675;
                double r_tmp_149200 = zp_res_126676;
                
                r_126672 = r_tmp_149200;
            }
            defunc_0_lifted_lambda_res_126670 = r_126672;
            ((double *) mem_147052)[i_145772] = defunc_0_lifted_lambda_res_126670;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147047, i_145776 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147052, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147063_cached_sizze_149677 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147063, &mem_147063_cached_sizze_149677, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147068_cached_sizze_149678 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147068, &mem_147068_cached_sizze_149678, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145784 = 0; i_145784 < (int64_t) 16; i_145784++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145780 = 0; i_145780 < (int64_t) 16; i_145780++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_126691 = ((double *) mem_147047)[i_145784 * (int64_t) 16 + i_145780];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_126692 = ((double *) mem_146751)[i_145784 * (int64_t) 16 + i_145780];
            
            // futhark/microgpt.fut:173:46-84
            
            double zp_res_126693 = zp_lhs_126691 + zp_rhs_126692;
            
            ((double *) mem_147068)[i_145780] = zp_res_126693;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147063, i_145784 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147068, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147079_cached_sizze_149679 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147079, &mem_147079_cached_sizze_149679, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147084_cached_sizze_149680 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147084, &mem_147084_cached_sizze_149680, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147091_cached_sizze_149681 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147091, &mem_147091_cached_sizze_149681, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145796 = 0; i_145796 < (int64_t) 16; i_145796++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_126702;
        double r_126704 = 0.0;
        
        for (int64_t i_126703 = 0; i_126703 < (int64_t) 16; i_126703++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_126705 = ((double *) mem_147063)[i_145796 * (int64_t) 16 + i_126703];
            
            // futhark/microgpt.fut:174:79-118
            
            double zt_res_126706 = zt_lhs_126705 * zt_lhs_126705;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_126707 = r_126704 + zt_res_126706;
            double r_tmp_149204 = zp_res_126707;
            
            r_126704 = r_tmp_149204;
        }
        defunc_0_lifted_lambda_res_126702 = r_126704;
        // futhark/microgpt.fut:174:58-136
        
        double zs_res_126708 = defunc_0_lifted_lambda_res_126702 / 16.0;
        
        // futhark/microgpt.fut:175:24-55
        
        double zp_res_126709 = 1.0e-5 + zs_res_126708;
        
        // futhark/microgpt.fut:175:16-55
        
        double sqrt_res_126710 = futrts_sqrt64(zp_res_126709);
        
        // futhark/microgpt.fut:176:60-71
        
        double zs_res_126711 = 1.0 / sqrt_res_126710;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145788 = 0; i_145788 < (int64_t) 16; i_145788++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_126718 = ((double *) mem_147063)[i_145796 * (int64_t) 16 + i_145788];
            
            // futhark/microgpt.fut:176:37-71
            
            double zt_res_126719 = zs_res_126711 * zt_lhs_126718;
            
            ((double *) mem_147084)[i_145788] = zt_res_126719;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145792 = 0; i_145792 < (int64_t) 16; i_145792++) {
            // futhark/microgpt.fut:177:4-14
            
            double lifted_lambda_res_126727 = ((double *) mem_147084)[i_145792];
            
            ((double *) mem_147091)[i_145792] = lifted_lambda_res_126727;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147079, i_145796 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147091, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147102_cached_sizze_149682 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147102, &mem_147102_cached_sizze_149682, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147107_cached_sizze_149683 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147107, &mem_147107_cached_sizze_149683, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145804 = 0; i_145804 < (int64_t) 16; i_145804++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145800 = 0; i_145800 < (int64_t) 64; i_145800++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_126743;
            double r_126745 = 0.0;
            
            for (int64_t i_126744 = 0; i_126744 < (int64_t) 16; i_126744++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_126746 = ((double *) wup_mem_146730.mem)[i_145800 * (int64_t) 16 + i_126744];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_126747 = ((double *) mem_147079)[i_145804 * (int64_t) 16 + i_126744];
                
                // futhark/microgpt.fut:178:67-106
                
                double zt_res_126748 = zt_lhs_126746 * zt_rhs_126747;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_126749 = r_126745 + zt_res_126748;
                double r_tmp_149209 = zp_res_126749;
                
                r_126745 = r_tmp_149209;
            }
            defunc_0_lifted_lambda_res_126743 = r_126745;
            ((double *) mem_147107)[i_145800] = defunc_0_lifted_lambda_res_126743;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147102, i_145804 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147107, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147118_cached_sizze_149684 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147118, &mem_147118_cached_sizze_149684, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147123_cached_sizze_149685 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147123, &mem_147123_cached_sizze_149685, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145812 = 0; i_145812 < (int64_t) 16; i_145812++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145808 = 0; i_145808 < (int64_t) 64; i_145808++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_126764 = ((double *) mem_147102)[i_145812 * (int64_t) 64 + i_145808];
            
            // futhark/microgpt.fut:179:45-73
            
            double max_res_126765 = fmax64(0.0, max_arg0_126764);
            
            ((double *) mem_147123)[i_145808] = max_res_126765;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147118, i_145812 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147123, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147134_cached_sizze_149686 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147134, &mem_147134_cached_sizze_149686, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147139_cached_sizze_149687 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147139, &mem_147139_cached_sizze_149687, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145820 = 0; i_145820 < (int64_t) 16; i_145820++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145816 = 0; i_145816 < (int64_t) 16; i_145816++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_126780;
            double r_126782 = 0.0;
            
            for (int64_t i_126781 = 0; i_126781 < (int64_t) 64; i_126781++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_126783 = ((double *) wdown_mem_146724.mem)[i_145816 * (int64_t) 64 + i_126781];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_126784 = ((double *) mem_147118)[i_145820 * (int64_t) 64 + i_126781];
                
                // futhark/microgpt.fut:180:67-108
                
                double zt_res_126785 = zt_lhs_126783 * zt_rhs_126784;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_126786 = r_126782 + zt_res_126785;
                double r_tmp_149214 = zp_res_126786;
                
                r_126782 = r_tmp_149214;
            }
            defunc_0_lifted_lambda_res_126780 = r_126782;
            ((double *) mem_147139)[i_145816] = defunc_0_lifted_lambda_res_126780;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147134, i_145820 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147139, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147150_cached_sizze_149688 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147150, &mem_147150_cached_sizze_149688, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147155_cached_sizze_149689 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147155, &mem_147155_cached_sizze_149689, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145828 = 0; i_145828 < (int64_t) 16; i_145828++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145824 = 0; i_145824 < (int64_t) 16; i_145824++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_126801 = ((double *) mem_147134)[i_145828 * (int64_t) 16 + i_145824];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_126802 = ((double *) mem_147063)[i_145828 * (int64_t) 16 + i_145824];
            
            // futhark/microgpt.fut:181:46-85
            
            double zp_res_126803 = zp_lhs_126801 + zp_rhs_126802;
            
            ((double *) mem_147155)[i_145824] = zp_res_126803;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147150, i_145828 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147155, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147166_cached_sizze_149690 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_147166, &mem_147166_cached_sizze_149690, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147171_cached_sizze_149691 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_147171, &mem_147171_cached_sizze_149691, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145836 = 0; i_145836 < (int64_t) 16; i_145836++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145832 = 0; i_145832 < (int64_t) 27; i_145832++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_126819;
            double r_126821 = 0.0;
            
            for (int64_t i_126820 = 0; i_126820 < (int64_t) 16; i_126820++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_126822 = ((double *) wvoc_mem_146732.mem)[i_145832 * (int64_t) 16 + i_126820];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_126823 = ((double *) mem_147150)[i_145836 * (int64_t) 16 + i_126820];
                
                // futhark/microgpt.fut:182:67-107
                
                double zt_res_126824 = zt_lhs_126822 * zt_rhs_126823;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_126825 = r_126821 + zt_res_126824;
                double r_tmp_149219 = zp_res_126825;
                
                r_126821 = r_tmp_149219;
            }
            defunc_0_lifted_lambda_res_126819 = r_126821;
            ((double *) mem_147171)[i_145832] = defunc_0_lifted_lambda_res_126819;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147166, i_145836 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147171, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_147182, (int64_t) 3456, "mem_147182")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147187_cached_sizze_149692 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_147187, &mem_147187_cached_sizze_149692, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_145844 = 0; i_145844 < (int64_t) 16; i_145844++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145840 = 0; i_145840 < (int64_t) 27; i_145840++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_126840 = ((double *) mem_147166)[i_145844 * (int64_t) 27 + i_145840];
            
            ((double *) mem_147187)[i_145840] = lifted_lambda_res_126840;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_147182.mem, i_145844 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147187, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_149150, &mem_147182, "mem_147182") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149636, &mem_out_149150, "mem_out_149150") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_146735);
        free(mem_146740);
        free(mem_146751);
        free(mem_146756);
        free(mem_146763);
        free(mem_146774);
        free(mem_146779);
        free(mem_146786);
        free(mem_146797);
        free(mem_146798);
        free(mem_146799);
        free(mem_146812);
        free(mem_146813);
        free(mem_146814);
        free(mem_146845);
        free(mem_146846);
        free(mem_146847);
        free(mem_146863);
        free(mem_146864);
        free(mem_146865);
        free(mem_146878);
        free(mem_146879);
        free(mem_146880);
        free(mem_146926);
        free(mem_146932);
        free(mem_146937);
        free(mem_146948);
        free(mem_146953);
        free(mem_146964);
        free(mem_146969);
        free(mem_146976);
        free(mem_146983);
        free(mem_146994);
        free(mem_146999);
        free(mem_147010);
        free(mem_147015);
        free(mem_147031);
        free(mem_147036);
        free(mem_147047);
        free(mem_147052);
        free(mem_147063);
        free(mem_147068);
        free(mem_147079);
        free(mem_147084);
        free(mem_147091);
        free(mem_147102);
        free(mem_147107);
        free(mem_147118);
        free(mem_147123);
        free(mem_147134);
        free(mem_147139);
        free(mem_147150);
        free(mem_147155);
        free(mem_147166);
        free(mem_147171);
        free(mem_147187);
        if (memblock_unref(ctx, &mem_147182, "mem_147182") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149150, "mem_out_149150") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_149693, struct memblock *mem_out_p_149694, struct memblock *mem_out_p_149695, struct memblock *mem_out_p_149696, struct memblock *mem_out_p_149697, struct memblock *mem_out_p_149698, struct memblock *mem_out_p_149699, struct memblock *mem_out_p_149700, struct memblock *mem_out_p_149701, struct memblock wte_mem_146724, struct memblock wpe_mem_146725, struct memblock wqry_mem_146726, struct memblock wkey_mem_146727, struct memblock wval_mem_146728, struct memblock wout_mem_146729, struct memblock wup_mem_146730, struct memblock wdown_mem_146731, struct memblock wvoc_mem_146732)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_149158;
    
    mem_out_149158.references = NULL;
    
    struct memblock mem_out_149157;
    
    mem_out_149157.references = NULL;
    
    struct memblock mem_out_149156;
    
    mem_out_149156.references = NULL;
    
    struct memblock mem_out_149155;
    
    mem_out_149155.references = NULL;
    
    struct memblock mem_out_149154;
    
    mem_out_149154.references = NULL;
    
    struct memblock mem_out_149153;
    
    mem_out_149153.references = NULL;
    
    struct memblock mem_out_149152;
    
    mem_out_149152.references = NULL;
    
    struct memblock mem_out_149151;
    
    mem_out_149151.references = NULL;
    
    struct memblock mem_out_149150;
    
    mem_out_149150.references = NULL;
    
    struct memblock mem_146715 = ctx->constants->mem_146715;
    struct memblock mem_146716 = ctx->constants->mem_146716;
    struct memblock mem_146717 = ctx->constants->mem_146717;
    struct memblock mem_146718 = ctx->constants->mem_146718;
    struct memblock mem_146719 = ctx->constants->mem_146719;
    struct memblock mem_146720 = ctx->constants->mem_146720;
    struct memblock mem_146721 = ctx->constants->mem_146721;
    struct memblock mem_146722 = ctx->constants->mem_146722;
    struct memblock mem_146723 = ctx->constants->mem_146723;
    
    if (memblock_set(ctx, &mem_out_149150, &wdown_mem_146731, "wdown_mem_146731") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149151, &wkey_mem_146727, "wkey_mem_146727") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149152, &wout_mem_146729, "wout_mem_146729") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149153, &wpe_mem_146725, "wpe_mem_146725") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149154, &wqry_mem_146726, "wqry_mem_146726") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149155, &wte_mem_146724, "wte_mem_146724") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149156, &wup_mem_146730, "wup_mem_146730") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149157, &wval_mem_146728, "wval_mem_146728") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149158, &wvoc_mem_146732, "wvoc_mem_146732") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149693, &mem_out_149150, "mem_out_149150") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149694, &mem_out_149151, "mem_out_149151") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149695, &mem_out_149152, "mem_out_149152") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149696, &mem_out_149153, "mem_out_149153") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149697, &mem_out_149154, "mem_out_149154") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149698, &mem_out_149155, "mem_out_149155") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149699, &mem_out_149156, "mem_out_149156") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149700, &mem_out_149157, "mem_out_149157") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149701, &mem_out_149158, "mem_out_149158") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_149158, "mem_out_149158") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149157, "mem_out_149157") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149156, "mem_out_149156") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149155, "mem_out_149155") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149154, "mem_out_149154") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149153, "mem_out_149153") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149152, "mem_out_149152") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149151, "mem_out_149151") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149150, "mem_out_149150") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_149702, struct memblock *mem_out_p_149703, struct memblock *mem_out_p_149704, struct memblock *mem_out_p_149705, struct memblock *mem_out_p_149706, struct memblock *mem_out_p_149707, struct memblock *mem_out_p_149708, struct memblock *mem_out_p_149709, struct memblock *mem_out_p_149710, struct memblock *mem_out_p_149711, struct memblock *mem_out_p_149712, struct memblock *mem_out_p_149713, struct memblock *mem_out_p_149714, struct memblock *mem_out_p_149715, struct memblock *mem_out_p_149716, struct memblock *mem_out_p_149717, struct memblock *mem_out_p_149718, struct memblock *mem_out_p_149719, struct memblock *mem_out_p_149720, struct memblock *mem_out_p_149721, struct memblock *mem_out_p_149722, struct memblock *mem_out_p_149723, struct memblock *mem_out_p_149724, struct memblock *mem_out_p_149725, struct memblock *mem_out_p_149726, struct memblock *mem_out_p_149727, struct memblock *mem_out_p_149728, struct memblock wdown_mem_146724, struct memblock wkey_mem_146725, struct memblock wout_mem_146726, struct memblock wpe_mem_146727, struct memblock wqry_mem_146728, struct memblock wte_mem_146729, struct memblock wup_mem_146730, struct memblock wval_mem_146731, struct memblock wvoc_mem_146732, struct memblock wdown_mem_146733, struct memblock wkey_mem_146734, struct memblock wout_mem_146735, struct memblock wpe_mem_146736, struct memblock wqry_mem_146737, struct memblock wte_mem_146738, struct memblock wup_mem_146739, struct memblock wval_mem_146740, struct memblock wvoc_mem_146741, struct memblock wdown_mem_146742, struct memblock wkey_mem_146743, struct memblock wout_mem_146744, struct memblock wpe_mem_146745, struct memblock wqry_mem_146746, struct memblock wte_mem_146747, struct memblock wup_mem_146748, struct memblock wval_mem_146749, struct memblock wvoc_mem_146750, struct memblock masks_mem_146751, struct memblock dls_mem_146752, struct memblock seqs_mem_146753)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_146862_cached_sizze_149729 = 0;
    unsigned char *mem_146862 = NULL;
    int64_t mem_146863_cached_sizze_149730 = 0;
    unsigned char *mem_146863 = NULL;
    int64_t mem_146872_cached_sizze_149731 = 0;
    unsigned char *mem_146872 = NULL;
    int64_t mem_146879_cached_sizze_149732 = 0;
    unsigned char *mem_146879 = NULL;
    int64_t mem_146894_cached_sizze_149733 = 0;
    unsigned char *mem_146894 = NULL;
    int64_t mem_146895_cached_sizze_149734 = 0;
    unsigned char *mem_146895 = NULL;
    int64_t mem_146904_cached_sizze_149735 = 0;
    unsigned char *mem_146904 = NULL;
    int64_t mem_146911_cached_sizze_149736 = 0;
    unsigned char *mem_146911 = NULL;
    int64_t mem_146926_cached_sizze_149737 = 0;
    unsigned char *mem_146926 = NULL;
    int64_t mem_146927_cached_sizze_149738 = 0;
    unsigned char *mem_146927 = NULL;
    int64_t mem_146936_cached_sizze_149739 = 0;
    unsigned char *mem_146936 = NULL;
    int64_t mem_146937_cached_sizze_149740 = 0;
    unsigned char *mem_146937 = NULL;
    int64_t mem_146950_cached_sizze_149741 = 0;
    unsigned char *mem_146950 = NULL;
    int64_t mem_146965_cached_sizze_149742 = 0;
    unsigned char *mem_146965 = NULL;
    int64_t mem_146966_cached_sizze_149743 = 0;
    unsigned char *mem_146966 = NULL;
    int64_t mem_146967_cached_sizze_149744 = 0;
    unsigned char *mem_146967 = NULL;
    int64_t mem_146979_cached_sizze_149745 = 0;
    unsigned char *mem_146979 = NULL;
    int64_t mem_146980_cached_sizze_149746 = 0;
    unsigned char *mem_146980 = NULL;
    int64_t mem_146993_cached_sizze_149747 = 0;
    unsigned char *mem_146993 = NULL;
    int64_t mem_147011_cached_sizze_149748 = 0;
    unsigned char *mem_147011 = NULL;
    int64_t mem_147012_cached_sizze_149749 = 0;
    unsigned char *mem_147012 = NULL;
    int64_t mem_147013_cached_sizze_149750 = 0;
    unsigned char *mem_147013 = NULL;
    int64_t mem_147014_cached_sizze_149751 = 0;
    unsigned char *mem_147014 = NULL;
    int64_t mem_147015_cached_sizze_149752 = 0;
    unsigned char *mem_147015 = NULL;
    int64_t mem_147034_cached_sizze_149753 = 0;
    unsigned char *mem_147034 = NULL;
    int64_t mem_147035_cached_sizze_149754 = 0;
    unsigned char *mem_147035 = NULL;
    int64_t mem_147036_cached_sizze_149755 = 0;
    unsigned char *mem_147036 = NULL;
    int64_t mem_147073_cached_sizze_149756 = 0;
    unsigned char *mem_147073 = NULL;
    int64_t mem_147074_cached_sizze_149757 = 0;
    unsigned char *mem_147074 = NULL;
    int64_t mem_147075_cached_sizze_149758 = 0;
    unsigned char *mem_147075 = NULL;
    int64_t mem_147091_cached_sizze_149759 = 0;
    unsigned char *mem_147091 = NULL;
    int64_t mem_147092_cached_sizze_149760 = 0;
    unsigned char *mem_147092 = NULL;
    int64_t mem_147093_cached_sizze_149761 = 0;
    unsigned char *mem_147093 = NULL;
    int64_t mem_147106_cached_sizze_149762 = 0;
    unsigned char *mem_147106 = NULL;
    int64_t mem_147107_cached_sizze_149763 = 0;
    unsigned char *mem_147107 = NULL;
    int64_t mem_147108_cached_sizze_149764 = 0;
    unsigned char *mem_147108 = NULL;
    int64_t mem_147154_cached_sizze_149765 = 0;
    unsigned char *mem_147154 = NULL;
    int64_t mem_147155_cached_sizze_149766 = 0;
    unsigned char *mem_147155 = NULL;
    int64_t mem_147156_cached_sizze_149767 = 0;
    unsigned char *mem_147156 = NULL;
    int64_t mem_147157_cached_sizze_149768 = 0;
    unsigned char *mem_147157 = NULL;
    int64_t mem_147178_cached_sizze_149769 = 0;
    unsigned char *mem_147178 = NULL;
    int64_t mem_147179_cached_sizze_149770 = 0;
    unsigned char *mem_147179 = NULL;
    int64_t mem_147180_cached_sizze_149771 = 0;
    unsigned char *mem_147180 = NULL;
    int64_t mem_147181_cached_sizze_149772 = 0;
    unsigned char *mem_147181 = NULL;
    int64_t mem_147198_cached_sizze_149773 = 0;
    unsigned char *mem_147198 = NULL;
    int64_t mem_147199_cached_sizze_149774 = 0;
    unsigned char *mem_147199 = NULL;
    int64_t mem_147200_cached_sizze_149775 = 0;
    unsigned char *mem_147200 = NULL;
    int64_t mem_147201_cached_sizze_149776 = 0;
    unsigned char *mem_147201 = NULL;
    int64_t mem_147242_cached_sizze_149777 = 0;
    unsigned char *mem_147242 = NULL;
    int64_t mem_147247_cached_sizze_149778 = 0;
    unsigned char *mem_147247 = NULL;
    int64_t mem_147258_cached_sizze_149779 = 0;
    unsigned char *mem_147258 = NULL;
    int64_t mem_147268_cached_sizze_149780 = 0;
    unsigned char *mem_147268 = NULL;
    int64_t mem_147273_cached_sizze_149781 = 0;
    unsigned char *mem_147273 = NULL;
    int64_t mem_147280_cached_sizze_149782 = 0;
    unsigned char *mem_147280 = NULL;
    int64_t mem_147291_cached_sizze_149783 = 0;
    unsigned char *mem_147291 = NULL;
    int64_t mem_147296_cached_sizze_149784 = 0;
    unsigned char *mem_147296 = NULL;
    int64_t mem_147327_cached_sizze_149785 = 0;
    unsigned char *mem_147327 = NULL;
    int64_t mem_147328_cached_sizze_149786 = 0;
    unsigned char *mem_147328 = NULL;
    int64_t mem_147336_cached_sizze_149787 = 0;
    unsigned char *mem_147336 = NULL;
    int64_t mem_147350_cached_sizze_149788 = 0;
    unsigned char *mem_147350 = NULL;
    int64_t mem_147355_cached_sizze_149789 = 0;
    unsigned char *mem_147355 = NULL;
    int64_t mem_147366_cached_sizze_149790 = 0;
    unsigned char *mem_147366 = NULL;
    int64_t mem_147371_cached_sizze_149791 = 0;
    unsigned char *mem_147371 = NULL;
    int64_t mem_147382_cached_sizze_149792 = 0;
    unsigned char *mem_147382 = NULL;
    int64_t mem_147383_cached_sizze_149793 = 0;
    unsigned char *mem_147383 = NULL;
    int64_t mem_147392_cached_sizze_149794 = 0;
    unsigned char *mem_147392 = NULL;
    int64_t mem_147393_cached_sizze_149795 = 0;
    unsigned char *mem_147393 = NULL;
    int64_t mem_147406_cached_sizze_149796 = 0;
    unsigned char *mem_147406 = NULL;
    int64_t mem_147421_cached_sizze_149797 = 0;
    unsigned char *mem_147421 = NULL;
    int64_t mem_147422_cached_sizze_149798 = 0;
    unsigned char *mem_147422 = NULL;
    int64_t mem_147430_cached_sizze_149799 = 0;
    unsigned char *mem_147430 = NULL;
    int64_t mem_147444_cached_sizze_149800 = 0;
    unsigned char *mem_147444 = NULL;
    int64_t mem_147445_cached_sizze_149801 = 0;
    unsigned char *mem_147445 = NULL;
    int64_t mem_147453_cached_sizze_149802 = 0;
    unsigned char *mem_147453 = NULL;
    int64_t mem_147467_cached_sizze_149803 = 0;
    unsigned char *mem_147467 = NULL;
    int64_t mem_147472_cached_sizze_149804 = 0;
    unsigned char *mem_147472 = NULL;
    int64_t mem_147483_cached_sizze_149805 = 0;
    unsigned char *mem_147483 = NULL;
    int64_t mem_147488_cached_sizze_149806 = 0;
    unsigned char *mem_147488 = NULL;
    int64_t mem_147499_cached_sizze_149807 = 0;
    unsigned char *mem_147499 = NULL;
    int64_t mem_147504_cached_sizze_149808 = 0;
    unsigned char *mem_147504 = NULL;
    int64_t mem_147515_cached_sizze_149809 = 0;
    unsigned char *mem_147515 = NULL;
    int64_t mem_147520_cached_sizze_149810 = 0;
    unsigned char *mem_147520 = NULL;
    int64_t mem_147531_cached_sizze_149811 = 0;
    unsigned char *mem_147531 = NULL;
    int64_t mem_147532_cached_sizze_149812 = 0;
    unsigned char *mem_147532 = NULL;
    int64_t mem_147539_cached_sizze_149813 = 0;
    unsigned char *mem_147539 = NULL;
    int64_t mem_147552_cached_sizze_149814 = 0;
    unsigned char *mem_147552 = NULL;
    int64_t mem_147557_cached_sizze_149815 = 0;
    unsigned char *mem_147557 = NULL;
    int64_t mem_147564_cached_sizze_149816 = 0;
    unsigned char *mem_147564 = NULL;
    int64_t mem_147575_cached_sizze_149817 = 0;
    unsigned char *mem_147575 = NULL;
    int64_t mem_147580_cached_sizze_149818 = 0;
    unsigned char *mem_147580 = NULL;
    int64_t mem_147591_cached_sizze_149819 = 0;
    unsigned char *mem_147591 = NULL;
    int64_t mem_147596_cached_sizze_149820 = 0;
    unsigned char *mem_147596 = NULL;
    int64_t mem_147607_cached_sizze_149821 = 0;
    unsigned char *mem_147607 = NULL;
    int64_t mem_147608_cached_sizze_149822 = 0;
    unsigned char *mem_147608 = NULL;
    int64_t mem_147617_cached_sizze_149823 = 0;
    unsigned char *mem_147617 = NULL;
    int64_t mem_147618_cached_sizze_149824 = 0;
    unsigned char *mem_147618 = NULL;
    int64_t mem_147639_cached_sizze_149825 = 0;
    unsigned char *mem_147639 = NULL;
    int64_t mem_147644_cached_sizze_149826 = 0;
    unsigned char *mem_147644 = NULL;
    int64_t mem_147655_cached_sizze_149827 = 0;
    unsigned char *mem_147655 = NULL;
    int64_t mem_147660_cached_sizze_149828 = 0;
    unsigned char *mem_147660 = NULL;
    int64_t mem_147671_cached_sizze_149829 = 0;
    unsigned char *mem_147671 = NULL;
    int64_t mem_147678_cached_sizze_149830 = 0;
    unsigned char *mem_147678 = NULL;
    int64_t mem_147685_cached_sizze_149831 = 0;
    unsigned char *mem_147685 = NULL;
    int64_t mem_147695_cached_sizze_149832 = 0;
    unsigned char *mem_147695 = NULL;
    int64_t mem_147700_cached_sizze_149833 = 0;
    unsigned char *mem_147700 = NULL;
    int64_t mem_147711_cached_sizze_149834 = 0;
    unsigned char *mem_147711 = NULL;
    int64_t mem_147712_cached_sizze_149835 = 0;
    unsigned char *mem_147712 = NULL;
    int64_t mem_147721_cached_sizze_149836 = 0;
    unsigned char *mem_147721 = NULL;
    int64_t mem_147722_cached_sizze_149837 = 0;
    unsigned char *mem_147722 = NULL;
    int64_t mem_147743_cached_sizze_149838 = 0;
    unsigned char *mem_147743 = NULL;
    int64_t mem_147744_cached_sizze_149839 = 0;
    unsigned char *mem_147744 = NULL;
    int64_t mem_147745_cached_sizze_149840 = 0;
    unsigned char *mem_147745 = NULL;
    int64_t mem_147746_cached_sizze_149841 = 0;
    unsigned char *mem_147746 = NULL;
    int64_t mem_147767_cached_sizze_149842 = 0;
    unsigned char *mem_147767 = NULL;
    int64_t mem_147768_cached_sizze_149843 = 0;
    unsigned char *mem_147768 = NULL;
    int64_t mem_147769_cached_sizze_149844 = 0;
    unsigned char *mem_147769 = NULL;
    int64_t mem_147770_cached_sizze_149845 = 0;
    unsigned char *mem_147770 = NULL;
    int64_t mem_147787_cached_sizze_149846 = 0;
    unsigned char *mem_147787 = NULL;
    int64_t mem_147794_cached_sizze_149847 = 0;
    unsigned char *mem_147794 = NULL;
    int64_t mem_147795_cached_sizze_149848 = 0;
    unsigned char *mem_147795 = NULL;
    int64_t mem_147796_cached_sizze_149849 = 0;
    unsigned char *mem_147796 = NULL;
    int64_t mem_147851_cached_sizze_149850 = 0;
    unsigned char *mem_147851 = NULL;
    int64_t mem_147852_cached_sizze_149851 = 0;
    unsigned char *mem_147852 = NULL;
    int64_t mem_147853_cached_sizze_149852 = 0;
    unsigned char *mem_147853 = NULL;
    int64_t mem_147854_cached_sizze_149853 = 0;
    unsigned char *mem_147854 = NULL;
    int64_t mem_147855_cached_sizze_149854 = 0;
    unsigned char *mem_147855 = NULL;
    int64_t mem_147856_cached_sizze_149855 = 0;
    unsigned char *mem_147856 = NULL;
    int64_t mem_147857_cached_sizze_149856 = 0;
    unsigned char *mem_147857 = NULL;
    int64_t mem_147858_cached_sizze_149857 = 0;
    unsigned char *mem_147858 = NULL;
    int64_t mem_147859_cached_sizze_149858 = 0;
    unsigned char *mem_147859 = NULL;
    int64_t mem_147901_cached_sizze_149859 = 0;
    unsigned char *mem_147901 = NULL;
    int64_t mem_147902_cached_sizze_149860 = 0;
    unsigned char *mem_147902 = NULL;
    int64_t mem_147903_cached_sizze_149861 = 0;
    unsigned char *mem_147903 = NULL;
    int64_t mem_147904_cached_sizze_149862 = 0;
    unsigned char *mem_147904 = NULL;
    int64_t mem_147905_cached_sizze_149863 = 0;
    unsigned char *mem_147905 = NULL;
    int64_t mem_147906_cached_sizze_149864 = 0;
    unsigned char *mem_147906 = NULL;
    int64_t mem_147907_cached_sizze_149865 = 0;
    unsigned char *mem_147907 = NULL;
    int64_t mem_147908_cached_sizze_149866 = 0;
    unsigned char *mem_147908 = NULL;
    int64_t mem_147909_cached_sizze_149867 = 0;
    unsigned char *mem_147909 = NULL;
    int64_t mem_147942_cached_sizze_149868 = 0;
    unsigned char *mem_147942 = NULL;
    int64_t mem_147943_cached_sizze_149869 = 0;
    unsigned char *mem_147943 = NULL;
    int64_t mem_148032_cached_sizze_149870 = 0;
    unsigned char *mem_148032 = NULL;
    int64_t mem_148033_cached_sizze_149871 = 0;
    unsigned char *mem_148033 = NULL;
    int64_t mem_148034_cached_sizze_149872 = 0;
    unsigned char *mem_148034 = NULL;
    int64_t mem_148050_cached_sizze_149873 = 0;
    unsigned char *mem_148050 = NULL;
    int64_t mem_148051_cached_sizze_149874 = 0;
    unsigned char *mem_148051 = NULL;
    int64_t mem_148052_cached_sizze_149875 = 0;
    unsigned char *mem_148052 = NULL;
    int64_t mem_148065_cached_sizze_149876 = 0;
    unsigned char *mem_148065 = NULL;
    int64_t mem_148066_cached_sizze_149877 = 0;
    unsigned char *mem_148066 = NULL;
    int64_t mem_148067_cached_sizze_149878 = 0;
    unsigned char *mem_148067 = NULL;
    int64_t mem_148086_cached_sizze_149879 = 0;
    unsigned char *mem_148086 = NULL;
    int64_t mem_148120_cached_sizze_149880 = 0;
    unsigned char *mem_148120 = NULL;
    int64_t mem_148121_cached_sizze_149881 = 0;
    unsigned char *mem_148121 = NULL;
    int64_t mem_148122_cached_sizze_149882 = 0;
    unsigned char *mem_148122 = NULL;
    int64_t mem_148123_cached_sizze_149883 = 0;
    unsigned char *mem_148123 = NULL;
    int64_t mem_148124_cached_sizze_149884 = 0;
    unsigned char *mem_148124 = NULL;
    int64_t mem_148146_cached_sizze_149885 = 0;
    unsigned char *mem_148146 = NULL;
    int64_t mem_148147_cached_sizze_149886 = 0;
    unsigned char *mem_148147 = NULL;
    int64_t mem_148148_cached_sizze_149887 = 0;
    unsigned char *mem_148148 = NULL;
    int64_t mem_148149_cached_sizze_149888 = 0;
    unsigned char *mem_148149 = NULL;
    int64_t mem_148150_cached_sizze_149889 = 0;
    unsigned char *mem_148150 = NULL;
    int64_t mem_148167_cached_sizze_149890 = 0;
    unsigned char *mem_148167 = NULL;
    int64_t mem_148211_cached_sizze_149891 = 0;
    unsigned char *mem_148211 = NULL;
    int64_t mem_148212_cached_sizze_149892 = 0;
    unsigned char *mem_148212 = NULL;
    int64_t mem_148213_cached_sizze_149893 = 0;
    unsigned char *mem_148213 = NULL;
    int64_t mem_148214_cached_sizze_149894 = 0;
    unsigned char *mem_148214 = NULL;
    int64_t mem_148215_cached_sizze_149895 = 0;
    unsigned char *mem_148215 = NULL;
    int64_t mem_148216_cached_sizze_149896 = 0;
    unsigned char *mem_148216 = NULL;
    int64_t mem_148243_cached_sizze_149897 = 0;
    unsigned char *mem_148243 = NULL;
    int64_t mem_148244_cached_sizze_149898 = 0;
    unsigned char *mem_148244 = NULL;
    int64_t mem_148245_cached_sizze_149899 = 0;
    unsigned char *mem_148245 = NULL;
    int64_t mem_148246_cached_sizze_149900 = 0;
    unsigned char *mem_148246 = NULL;
    int64_t mem_148247_cached_sizze_149901 = 0;
    unsigned char *mem_148247 = NULL;
    int64_t mem_148248_cached_sizze_149902 = 0;
    unsigned char *mem_148248 = NULL;
    int64_t mem_148269_cached_sizze_149903 = 0;
    unsigned char *mem_148269 = NULL;
    int64_t mem_148270_cached_sizze_149904 = 0;
    unsigned char *mem_148270 = NULL;
    int64_t mem_148329_cached_sizze_149905 = 0;
    unsigned char *mem_148329 = NULL;
    int64_t mem_148330_cached_sizze_149906 = 0;
    unsigned char *mem_148330 = NULL;
    int64_t mem_148331_cached_sizze_149907 = 0;
    unsigned char *mem_148331 = NULL;
    int64_t mem_148332_cached_sizze_149908 = 0;
    unsigned char *mem_148332 = NULL;
    int64_t mem_148353_cached_sizze_149909 = 0;
    unsigned char *mem_148353 = NULL;
    int64_t mem_148354_cached_sizze_149910 = 0;
    unsigned char *mem_148354 = NULL;
    int64_t mem_148355_cached_sizze_149911 = 0;
    unsigned char *mem_148355 = NULL;
    int64_t mem_148356_cached_sizze_149912 = 0;
    unsigned char *mem_148356 = NULL;
    int64_t mem_148373_cached_sizze_149913 = 0;
    unsigned char *mem_148373 = NULL;
    int64_t mem_148374_cached_sizze_149914 = 0;
    unsigned char *mem_148374 = NULL;
    int64_t mem_148375_cached_sizze_149915 = 0;
    unsigned char *mem_148375 = NULL;
    int64_t mem_148376_cached_sizze_149916 = 0;
    unsigned char *mem_148376 = NULL;
    int64_t mem_148437_cached_sizze_149917 = 0;
    unsigned char *mem_148437 = NULL;
    int64_t mem_148438_cached_sizze_149918 = 0;
    unsigned char *mem_148438 = NULL;
    int64_t mem_148449_cached_sizze_149919 = 0;
    unsigned char *mem_148449 = NULL;
    int64_t mem_148450_cached_sizze_149920 = 0;
    unsigned char *mem_148450 = NULL;
    int64_t mem_148459_cached_sizze_149921 = 0;
    unsigned char *mem_148459 = NULL;
    int64_t mem_148460_cached_sizze_149922 = 0;
    unsigned char *mem_148460 = NULL;
    int64_t mem_148491_cached_sizze_149923 = 0;
    unsigned char *mem_148491 = NULL;
    int64_t mem_148492_cached_sizze_149924 = 0;
    unsigned char *mem_148492 = NULL;
    int64_t mem_148503_cached_sizze_149925 = 0;
    unsigned char *mem_148503 = NULL;
    int64_t mem_148504_cached_sizze_149926 = 0;
    unsigned char *mem_148504 = NULL;
    int64_t mem_148513_cached_sizze_149927 = 0;
    unsigned char *mem_148513 = NULL;
    int64_t mem_148514_cached_sizze_149928 = 0;
    unsigned char *mem_148514 = NULL;
    int64_t mem_148545_cached_sizze_149929 = 0;
    unsigned char *mem_148545 = NULL;
    int64_t mem_148546_cached_sizze_149930 = 0;
    unsigned char *mem_148546 = NULL;
    int64_t mem_148557_cached_sizze_149931 = 0;
    unsigned char *mem_148557 = NULL;
    int64_t mem_148558_cached_sizze_149932 = 0;
    unsigned char *mem_148558 = NULL;
    int64_t mem_148567_cached_sizze_149933 = 0;
    unsigned char *mem_148567 = NULL;
    int64_t mem_148568_cached_sizze_149934 = 0;
    unsigned char *mem_148568 = NULL;
    int64_t mem_148599_cached_sizze_149935 = 0;
    unsigned char *mem_148599 = NULL;
    int64_t mem_148600_cached_sizze_149936 = 0;
    unsigned char *mem_148600 = NULL;
    int64_t mem_148601_cached_sizze_149937 = 0;
    unsigned char *mem_148601 = NULL;
    int64_t mem_148614_cached_sizze_149938 = 0;
    unsigned char *mem_148614 = NULL;
    int64_t mem_148615_cached_sizze_149939 = 0;
    unsigned char *mem_148615 = NULL;
    int64_t mem_148616_cached_sizze_149940 = 0;
    unsigned char *mem_148616 = NULL;
    int64_t mem_148647_cached_sizze_149941 = 0;
    unsigned char *mem_148647 = NULL;
    int64_t mem_148648_cached_sizze_149942 = 0;
    unsigned char *mem_148648 = NULL;
    int64_t mem_148649_cached_sizze_149943 = 0;
    unsigned char *mem_148649 = NULL;
    int64_t mem_148650_cached_sizze_149944 = 0;
    unsigned char *mem_148650 = NULL;
    int64_t mem_148667_cached_sizze_149945 = 0;
    unsigned char *mem_148667 = NULL;
    int64_t mem_148668_cached_sizze_149946 = 0;
    unsigned char *mem_148668 = NULL;
    int64_t mem_148669_cached_sizze_149947 = 0;
    unsigned char *mem_148669 = NULL;
    int64_t mem_148670_cached_sizze_149948 = 0;
    unsigned char *mem_148670 = NULL;
    int64_t mem_148711_cached_sizze_149949 = 0;
    unsigned char *mem_148711 = NULL;
    int64_t mem_148718_cached_sizze_149950 = 0;
    unsigned char *mem_148718 = NULL;
    int64_t mem_148725_cached_sizze_149951 = 0;
    unsigned char *mem_148725 = NULL;
    int64_t mem_148735_cached_sizze_149952 = 0;
    unsigned char *mem_148735 = NULL;
    int64_t mem_148740_cached_sizze_149953 = 0;
    unsigned char *mem_148740 = NULL;
    int64_t mem_148751_cached_sizze_149954 = 0;
    unsigned char *mem_148751 = NULL;
    int64_t mem_148758_cached_sizze_149955 = 0;
    unsigned char *mem_148758 = NULL;
    int64_t mem_148765_cached_sizze_149956 = 0;
    unsigned char *mem_148765 = NULL;
    int64_t mem_148775_cached_sizze_149957 = 0;
    unsigned char *mem_148775 = NULL;
    int64_t mem_148780_cached_sizze_149958 = 0;
    unsigned char *mem_148780 = NULL;
    int64_t mem_148791_cached_sizze_149959 = 0;
    unsigned char *mem_148791 = NULL;
    int64_t mem_148792_cached_sizze_149960 = 0;
    unsigned char *mem_148792 = NULL;
    int64_t mem_148801_cached_sizze_149961 = 0;
    unsigned char *mem_148801 = NULL;
    int64_t mem_148802_cached_sizze_149962 = 0;
    unsigned char *mem_148802 = NULL;
    int64_t mem_148823_cached_sizze_149963 = 0;
    unsigned char *mem_148823 = NULL;
    int64_t mem_148828_cached_sizze_149964 = 0;
    unsigned char *mem_148828 = NULL;
    int64_t mem_148839_cached_sizze_149965 = 0;
    unsigned char *mem_148839 = NULL;
    int64_t mem_148840_cached_sizze_149966 = 0;
    unsigned char *mem_148840 = NULL;
    int64_t mem_148849_cached_sizze_149967 = 0;
    unsigned char *mem_148849 = NULL;
    int64_t mem_148850_cached_sizze_149968 = 0;
    unsigned char *mem_148850 = NULL;
    struct memblock mem_param_tmp_149203;
    
    mem_param_tmp_149203.references = NULL;
    
    struct memblock mem_param_tmp_149202;
    
    mem_param_tmp_149202.references = NULL;
    
    struct memblock mem_param_tmp_149201;
    
    mem_param_tmp_149201.references = NULL;
    
    struct memblock mem_param_tmp_149200;
    
    mem_param_tmp_149200.references = NULL;
    
    struct memblock mem_param_tmp_149199;
    
    mem_param_tmp_149199.references = NULL;
    
    struct memblock mem_param_tmp_149198;
    
    mem_param_tmp_149198.references = NULL;
    
    struct memblock mem_param_tmp_149197;
    
    mem_param_tmp_149197.references = NULL;
    
    struct memblock mem_param_tmp_149196;
    
    mem_param_tmp_149196.references = NULL;
    
    struct memblock mem_param_tmp_149195;
    
    mem_param_tmp_149195.references = NULL;
    
    struct memblock mem_param_tmp_149194;
    
    mem_param_tmp_149194.references = NULL;
    
    struct memblock mem_param_tmp_149193;
    
    mem_param_tmp_149193.references = NULL;
    
    struct memblock mem_param_tmp_149192;
    
    mem_param_tmp_149192.references = NULL;
    
    struct memblock mem_param_tmp_149191;
    
    mem_param_tmp_149191.references = NULL;
    
    struct memblock mem_param_tmp_149190;
    
    mem_param_tmp_149190.references = NULL;
    
    struct memblock mem_param_tmp_149189;
    
    mem_param_tmp_149189.references = NULL;
    
    struct memblock mem_param_tmp_149188;
    
    mem_param_tmp_149188.references = NULL;
    
    struct memblock mem_param_tmp_149187;
    
    mem_param_tmp_149187.references = NULL;
    
    struct memblock mem_param_tmp_149186;
    
    mem_param_tmp_149186.references = NULL;
    
    struct memblock mem_param_tmp_149185;
    
    mem_param_tmp_149185.references = NULL;
    
    struct memblock mem_param_tmp_149184;
    
    mem_param_tmp_149184.references = NULL;
    
    struct memblock mem_param_tmp_149183;
    
    mem_param_tmp_149183.references = NULL;
    
    struct memblock mem_param_tmp_149182;
    
    mem_param_tmp_149182.references = NULL;
    
    struct memblock mem_param_tmp_149181;
    
    mem_param_tmp_149181.references = NULL;
    
    struct memblock mem_param_tmp_149180;
    
    mem_param_tmp_149180.references = NULL;
    
    struct memblock mem_param_tmp_149179;
    
    mem_param_tmp_149179.references = NULL;
    
    struct memblock mem_param_tmp_149178;
    
    mem_param_tmp_149178.references = NULL;
    
    struct memblock mem_param_tmp_149177;
    
    mem_param_tmp_149177.references = NULL;
    
    struct memblock ext_mem_148967;
    
    ext_mem_148967.references = NULL;
    
    struct memblock ext_mem_148968;
    
    ext_mem_148968.references = NULL;
    
    struct memblock ext_mem_148969;
    
    ext_mem_148969.references = NULL;
    
    struct memblock mem_148965;
    
    mem_148965.references = NULL;
    
    struct memblock mem_148963;
    
    mem_148963.references = NULL;
    
    struct memblock mem_148961;
    
    mem_148961.references = NULL;
    
    struct memblock mem_148959;
    
    mem_148959.references = NULL;
    
    struct memblock ext_mem_148956;
    
    ext_mem_148956.references = NULL;
    
    struct memblock ext_mem_148957;
    
    ext_mem_148957.references = NULL;
    
    struct memblock ext_mem_148958;
    
    ext_mem_148958.references = NULL;
    
    struct memblock mem_148954;
    
    mem_148954.references = NULL;
    
    struct memblock mem_148952;
    
    mem_148952.references = NULL;
    
    struct memblock mem_148950;
    
    mem_148950.references = NULL;
    
    struct memblock mem_148948;
    
    mem_148948.references = NULL;
    
    struct memblock ext_mem_148945;
    
    ext_mem_148945.references = NULL;
    
    struct memblock ext_mem_148946;
    
    ext_mem_148946.references = NULL;
    
    struct memblock ext_mem_148947;
    
    ext_mem_148947.references = NULL;
    
    struct memblock mem_148943;
    
    mem_148943.references = NULL;
    
    struct memblock mem_148941;
    
    mem_148941.references = NULL;
    
    struct memblock mem_148939;
    
    mem_148939.references = NULL;
    
    struct memblock mem_148937;
    
    mem_148937.references = NULL;
    
    struct memblock ext_mem_148934;
    
    ext_mem_148934.references = NULL;
    
    struct memblock ext_mem_148935;
    
    ext_mem_148935.references = NULL;
    
    struct memblock ext_mem_148936;
    
    ext_mem_148936.references = NULL;
    
    struct memblock mem_148932;
    
    mem_148932.references = NULL;
    
    struct memblock mem_148930;
    
    mem_148930.references = NULL;
    
    struct memblock mem_148928;
    
    mem_148928.references = NULL;
    
    struct memblock mem_148926;
    
    mem_148926.references = NULL;
    
    struct memblock ext_mem_148923;
    
    ext_mem_148923.references = NULL;
    
    struct memblock ext_mem_148924;
    
    ext_mem_148924.references = NULL;
    
    struct memblock ext_mem_148925;
    
    ext_mem_148925.references = NULL;
    
    struct memblock mem_148921;
    
    mem_148921.references = NULL;
    
    struct memblock mem_148919;
    
    mem_148919.references = NULL;
    
    struct memblock mem_148917;
    
    mem_148917.references = NULL;
    
    struct memblock mem_148915;
    
    mem_148915.references = NULL;
    
    struct memblock ext_mem_148912;
    
    ext_mem_148912.references = NULL;
    
    struct memblock ext_mem_148913;
    
    ext_mem_148913.references = NULL;
    
    struct memblock ext_mem_148914;
    
    ext_mem_148914.references = NULL;
    
    struct memblock mem_148910;
    
    mem_148910.references = NULL;
    
    struct memblock mem_148908;
    
    mem_148908.references = NULL;
    
    struct memblock mem_148906;
    
    mem_148906.references = NULL;
    
    struct memblock mem_148904;
    
    mem_148904.references = NULL;
    
    struct memblock ext_mem_148901;
    
    ext_mem_148901.references = NULL;
    
    struct memblock ext_mem_148902;
    
    ext_mem_148902.references = NULL;
    
    struct memblock ext_mem_148903;
    
    ext_mem_148903.references = NULL;
    
    struct memblock mem_148899;
    
    mem_148899.references = NULL;
    
    struct memblock mem_148897;
    
    mem_148897.references = NULL;
    
    struct memblock mem_148895;
    
    mem_148895.references = NULL;
    
    struct memblock mem_148893;
    
    mem_148893.references = NULL;
    
    struct memblock ext_mem_148890;
    
    ext_mem_148890.references = NULL;
    
    struct memblock ext_mem_148891;
    
    ext_mem_148891.references = NULL;
    
    struct memblock ext_mem_148892;
    
    ext_mem_148892.references = NULL;
    
    struct memblock mem_148888;
    
    mem_148888.references = NULL;
    
    struct memblock mem_148886;
    
    mem_148886.references = NULL;
    
    struct memblock mem_148884;
    
    mem_148884.references = NULL;
    
    struct memblock mem_148882;
    
    mem_148882.references = NULL;
    
    struct memblock ext_mem_148879;
    
    ext_mem_148879.references = NULL;
    
    struct memblock ext_mem_148880;
    
    ext_mem_148880.references = NULL;
    
    struct memblock ext_mem_148881;
    
    ext_mem_148881.references = NULL;
    
    struct memblock mem_148877;
    
    mem_148877.references = NULL;
    
    struct memblock mem_148875;
    
    mem_148875.references = NULL;
    
    struct memblock mem_148873;
    
    mem_148873.references = NULL;
    
    struct memblock mem_148871;
    
    mem_148871.references = NULL;
    
    struct memblock mem_param_146861;
    
    mem_param_146861.references = NULL;
    
    struct memblock mem_param_146857;
    
    mem_param_146857.references = NULL;
    
    struct memblock mem_param_146853;
    
    mem_param_146853.references = NULL;
    
    struct memblock mem_param_146849;
    
    mem_param_146849.references = NULL;
    
    struct memblock mem_param_146845;
    
    mem_param_146845.references = NULL;
    
    struct memblock mem_param_146841;
    
    mem_param_146841.references = NULL;
    
    struct memblock mem_param_146837;
    
    mem_param_146837.references = NULL;
    
    struct memblock mem_param_146833;
    
    mem_param_146833.references = NULL;
    
    struct memblock mem_param_146829;
    
    mem_param_146829.references = NULL;
    
    struct memblock mem_param_146825;
    
    mem_param_146825.references = NULL;
    
    struct memblock mem_param_146821;
    
    mem_param_146821.references = NULL;
    
    struct memblock mem_param_146817;
    
    mem_param_146817.references = NULL;
    
    struct memblock mem_param_146813;
    
    mem_param_146813.references = NULL;
    
    struct memblock mem_param_146809;
    
    mem_param_146809.references = NULL;
    
    struct memblock mem_param_146805;
    
    mem_param_146805.references = NULL;
    
    struct memblock mem_param_146801;
    
    mem_param_146801.references = NULL;
    
    struct memblock mem_param_146797;
    
    mem_param_146797.references = NULL;
    
    struct memblock mem_param_146793;
    
    mem_param_146793.references = NULL;
    
    struct memblock mem_param_146789;
    
    mem_param_146789.references = NULL;
    
    struct memblock mem_param_146785;
    
    mem_param_146785.references = NULL;
    
    struct memblock mem_param_146781;
    
    mem_param_146781.references = NULL;
    
    struct memblock mem_param_146777;
    
    mem_param_146777.references = NULL;
    
    struct memblock mem_param_146773;
    
    mem_param_146773.references = NULL;
    
    struct memblock mem_param_146769;
    
    mem_param_146769.references = NULL;
    
    struct memblock mem_param_146765;
    
    mem_param_146765.references = NULL;
    
    struct memblock mem_param_146761;
    
    mem_param_146761.references = NULL;
    
    struct memblock mem_param_146757;
    
    mem_param_146757.references = NULL;
    
    struct memblock ext_mem_149051;
    
    ext_mem_149051.references = NULL;
    
    struct memblock ext_mem_149052;
    
    ext_mem_149052.references = NULL;
    
    struct memblock ext_mem_149053;
    
    ext_mem_149053.references = NULL;
    
    struct memblock ext_mem_149054;
    
    ext_mem_149054.references = NULL;
    
    struct memblock ext_mem_149055;
    
    ext_mem_149055.references = NULL;
    
    struct memblock ext_mem_149056;
    
    ext_mem_149056.references = NULL;
    
    struct memblock ext_mem_149057;
    
    ext_mem_149057.references = NULL;
    
    struct memblock ext_mem_149058;
    
    ext_mem_149058.references = NULL;
    
    struct memblock ext_mem_149059;
    
    ext_mem_149059.references = NULL;
    
    struct memblock ext_mem_149060;
    
    ext_mem_149060.references = NULL;
    
    struct memblock ext_mem_149061;
    
    ext_mem_149061.references = NULL;
    
    struct memblock ext_mem_149062;
    
    ext_mem_149062.references = NULL;
    
    struct memblock ext_mem_149063;
    
    ext_mem_149063.references = NULL;
    
    struct memblock ext_mem_149064;
    
    ext_mem_149064.references = NULL;
    
    struct memblock ext_mem_149065;
    
    ext_mem_149065.references = NULL;
    
    struct memblock ext_mem_149066;
    
    ext_mem_149066.references = NULL;
    
    struct memblock ext_mem_149067;
    
    ext_mem_149067.references = NULL;
    
    struct memblock ext_mem_149068;
    
    ext_mem_149068.references = NULL;
    
    struct memblock ext_mem_149069;
    
    ext_mem_149069.references = NULL;
    
    struct memblock ext_mem_149070;
    
    ext_mem_149070.references = NULL;
    
    struct memblock ext_mem_149071;
    
    ext_mem_149071.references = NULL;
    
    struct memblock ext_mem_149072;
    
    ext_mem_149072.references = NULL;
    
    struct memblock ext_mem_149073;
    
    ext_mem_149073.references = NULL;
    
    struct memblock ext_mem_149074;
    
    ext_mem_149074.references = NULL;
    
    struct memblock ext_mem_149075;
    
    ext_mem_149075.references = NULL;
    
    struct memblock ext_mem_149076;
    
    ext_mem_149076.references = NULL;
    
    struct memblock ext_mem_149077;
    
    ext_mem_149077.references = NULL;
    
    struct memblock mem_out_149176;
    
    mem_out_149176.references = NULL;
    
    struct memblock mem_out_149175;
    
    mem_out_149175.references = NULL;
    
    struct memblock mem_out_149174;
    
    mem_out_149174.references = NULL;
    
    struct memblock mem_out_149173;
    
    mem_out_149173.references = NULL;
    
    struct memblock mem_out_149172;
    
    mem_out_149172.references = NULL;
    
    struct memblock mem_out_149171;
    
    mem_out_149171.references = NULL;
    
    struct memblock mem_out_149170;
    
    mem_out_149170.references = NULL;
    
    struct memblock mem_out_149169;
    
    mem_out_149169.references = NULL;
    
    struct memblock mem_out_149168;
    
    mem_out_149168.references = NULL;
    
    struct memblock mem_out_149167;
    
    mem_out_149167.references = NULL;
    
    struct memblock mem_out_149166;
    
    mem_out_149166.references = NULL;
    
    struct memblock mem_out_149165;
    
    mem_out_149165.references = NULL;
    
    struct memblock mem_out_149164;
    
    mem_out_149164.references = NULL;
    
    struct memblock mem_out_149163;
    
    mem_out_149163.references = NULL;
    
    struct memblock mem_out_149162;
    
    mem_out_149162.references = NULL;
    
    struct memblock mem_out_149161;
    
    mem_out_149161.references = NULL;
    
    struct memblock mem_out_149160;
    
    mem_out_149160.references = NULL;
    
    struct memblock mem_out_149159;
    
    mem_out_149159.references = NULL;
    
    struct memblock mem_out_149158;
    
    mem_out_149158.references = NULL;
    
    struct memblock mem_out_149157;
    
    mem_out_149157.references = NULL;
    
    struct memblock mem_out_149156;
    
    mem_out_149156.references = NULL;
    
    struct memblock mem_out_149155;
    
    mem_out_149155.references = NULL;
    
    struct memblock mem_out_149154;
    
    mem_out_149154.references = NULL;
    
    struct memblock mem_out_149153;
    
    mem_out_149153.references = NULL;
    
    struct memblock mem_out_149152;
    
    mem_out_149152.references = NULL;
    
    struct memblock mem_out_149151;
    
    mem_out_149151.references = NULL;
    
    struct memblock mem_out_149150;
    
    mem_out_149150.references = NULL;
    
    struct memblock mem_146715 = ctx->constants->mem_146715;
    struct memblock mem_146716 = ctx->constants->mem_146716;
    struct memblock mem_146717 = ctx->constants->mem_146717;
    struct memblock mem_146718 = ctx->constants->mem_146718;
    struct memblock mem_146719 = ctx->constants->mem_146719;
    struct memblock mem_146720 = ctx->constants->mem_146720;
    struct memblock mem_146721 = ctx->constants->mem_146721;
    struct memblock mem_146722 = ctx->constants->mem_146722;
    struct memblock mem_146723 = ctx->constants->mem_146723;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_146862_cached_sizze_149729 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146862, &mem_146862_cached_sizze_149729, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146863_cached_sizze_149730 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_146863, &mem_146863_cached_sizze_149730, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146872_cached_sizze_149731 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_146872, &mem_146872_cached_sizze_149731, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146879_cached_sizze_149732 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146879, &mem_146879_cached_sizze_149732, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146894_cached_sizze_149733 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_146894, &mem_146894_cached_sizze_149733, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146895_cached_sizze_149734 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146895, &mem_146895_cached_sizze_149734, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146904_cached_sizze_149735 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146904, &mem_146904_cached_sizze_149735, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146911_cached_sizze_149736 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_146911, &mem_146911_cached_sizze_149736, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146926_cached_sizze_149737 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146926, &mem_146926_cached_sizze_149737, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146927_cached_sizze_149738 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146927, &mem_146927_cached_sizze_149738, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146936_cached_sizze_149739 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146936, &mem_146936_cached_sizze_149739, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146937_cached_sizze_149740 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146937, &mem_146937_cached_sizze_149740, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146950_cached_sizze_149741 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146950, &mem_146950_cached_sizze_149741, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146965_cached_sizze_149742 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146965, &mem_146965_cached_sizze_149742, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146966_cached_sizze_149743 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146966, &mem_146966_cached_sizze_149743, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146967_cached_sizze_149744 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_146967, &mem_146967_cached_sizze_149744, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146979_cached_sizze_149745 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146979, &mem_146979_cached_sizze_149745, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146980_cached_sizze_149746 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146980, &mem_146980_cached_sizze_149746, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146993_cached_sizze_149747 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_146993, &mem_146993_cached_sizze_149747, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147011_cached_sizze_149748 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147011, &mem_147011_cached_sizze_149748, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147012_cached_sizze_149749 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147012, &mem_147012_cached_sizze_149749, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147013_cached_sizze_149750 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147013, &mem_147013_cached_sizze_149750, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147014_cached_sizze_149751 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147014, &mem_147014_cached_sizze_149751, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147015_cached_sizze_149752 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147015, &mem_147015_cached_sizze_149752, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147034_cached_sizze_149753 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147034, &mem_147034_cached_sizze_149753, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147035_cached_sizze_149754 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147035, &mem_147035_cached_sizze_149754, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147036_cached_sizze_149755 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147036, &mem_147036_cached_sizze_149755, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147073_cached_sizze_149756 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147073, &mem_147073_cached_sizze_149756, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147074_cached_sizze_149757 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147074, &mem_147074_cached_sizze_149757, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147075_cached_sizze_149758 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147075, &mem_147075_cached_sizze_149758, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147091_cached_sizze_149759 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147091, &mem_147091_cached_sizze_149759, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147092_cached_sizze_149760 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147092, &mem_147092_cached_sizze_149760, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147093_cached_sizze_149761 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147093, &mem_147093_cached_sizze_149761, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147106_cached_sizze_149762 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_147106, &mem_147106_cached_sizze_149762, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147107_cached_sizze_149763 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_147107, &mem_147107_cached_sizze_149763, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147108_cached_sizze_149764 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_147108, &mem_147108_cached_sizze_149764, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147154_cached_sizze_149765 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147154, &mem_147154_cached_sizze_149765, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147155_cached_sizze_149766 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147155, &mem_147155_cached_sizze_149766, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147156_cached_sizze_149767 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147156, &mem_147156_cached_sizze_149767, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147157_cached_sizze_149768 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147157, &mem_147157_cached_sizze_149768, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147178_cached_sizze_149769 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147178, &mem_147178_cached_sizze_149769, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147179_cached_sizze_149770 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147179, &mem_147179_cached_sizze_149770, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147180_cached_sizze_149771 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147180, &mem_147180_cached_sizze_149771, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147181_cached_sizze_149772 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147181, &mem_147181_cached_sizze_149772, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147198_cached_sizze_149773 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147198, &mem_147198_cached_sizze_149773, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147199_cached_sizze_149774 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147199, &mem_147199_cached_sizze_149774, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147200_cached_sizze_149775 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147200, &mem_147200_cached_sizze_149775, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147201_cached_sizze_149776 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147201, &mem_147201_cached_sizze_149776, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147242_cached_sizze_149777 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147242, &mem_147242_cached_sizze_149777, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147247_cached_sizze_149778 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147247, &mem_147247_cached_sizze_149778, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147258_cached_sizze_149779 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147258, &mem_147258_cached_sizze_149779, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147268_cached_sizze_149780 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147268, &mem_147268_cached_sizze_149780, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147273_cached_sizze_149781 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147273, &mem_147273_cached_sizze_149781, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147280_cached_sizze_149782 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147280, &mem_147280_cached_sizze_149782, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147291_cached_sizze_149783 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147291, &mem_147291_cached_sizze_149783, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147296_cached_sizze_149784 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_147296, &mem_147296_cached_sizze_149784, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147327_cached_sizze_149785 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147327, &mem_147327_cached_sizze_149785, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147328_cached_sizze_149786 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147328, &mem_147328_cached_sizze_149786, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147336_cached_sizze_149787 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147336, &mem_147336_cached_sizze_149787, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147350_cached_sizze_149788 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147350, &mem_147350_cached_sizze_149788, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147355_cached_sizze_149789 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147355, &mem_147355_cached_sizze_149789, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147366_cached_sizze_149790 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147366, &mem_147366_cached_sizze_149790, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147371_cached_sizze_149791 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147371, &mem_147371_cached_sizze_149791, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147382_cached_sizze_149792 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147382, &mem_147382_cached_sizze_149792, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147383_cached_sizze_149793 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147383, &mem_147383_cached_sizze_149793, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147392_cached_sizze_149794 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147392, &mem_147392_cached_sizze_149794, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147393_cached_sizze_149795 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147393, &mem_147393_cached_sizze_149795, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147406_cached_sizze_149796 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147406, &mem_147406_cached_sizze_149796, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147421_cached_sizze_149797 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147421, &mem_147421_cached_sizze_149797, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147422_cached_sizze_149798 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147422, &mem_147422_cached_sizze_149798, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147430_cached_sizze_149799 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147430, &mem_147430_cached_sizze_149799, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147444_cached_sizze_149800 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147444, &mem_147444_cached_sizze_149800, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147445_cached_sizze_149801 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147445, &mem_147445_cached_sizze_149801, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147453_cached_sizze_149802 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147453, &mem_147453_cached_sizze_149802, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147467_cached_sizze_149803 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147467, &mem_147467_cached_sizze_149803, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147472_cached_sizze_149804 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147472, &mem_147472_cached_sizze_149804, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147483_cached_sizze_149805 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147483, &mem_147483_cached_sizze_149805, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147488_cached_sizze_149806 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147488, &mem_147488_cached_sizze_149806, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147499_cached_sizze_149807 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_147499, &mem_147499_cached_sizze_149807, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147504_cached_sizze_149808 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_147504, &mem_147504_cached_sizze_149808, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147515_cached_sizze_149809 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_147515, &mem_147515_cached_sizze_149809, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147520_cached_sizze_149810 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_147520, &mem_147520_cached_sizze_149810, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147531_cached_sizze_149811 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147531, &mem_147531_cached_sizze_149811, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147532_cached_sizze_149812 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147532, &mem_147532_cached_sizze_149812, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147539_cached_sizze_149813 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_147539, &mem_147539_cached_sizze_149813, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147552_cached_sizze_149814 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_147552, &mem_147552_cached_sizze_149814, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147557_cached_sizze_149815 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_147557, &mem_147557_cached_sizze_149815, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147564_cached_sizze_149816 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_147564, &mem_147564_cached_sizze_149816, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147575_cached_sizze_149817 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_147575, &mem_147575_cached_sizze_149817, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147580_cached_sizze_149818 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_147580, &mem_147580_cached_sizze_149818, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147591_cached_sizze_149819 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147591, &mem_147591_cached_sizze_149819, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147596_cached_sizze_149820 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147596, &mem_147596_cached_sizze_149820, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147607_cached_sizze_149821 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147607, &mem_147607_cached_sizze_149821, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147608_cached_sizze_149822 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147608, &mem_147608_cached_sizze_149822, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147617_cached_sizze_149823 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147617, &mem_147617_cached_sizze_149823, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147618_cached_sizze_149824 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147618, &mem_147618_cached_sizze_149824, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147639_cached_sizze_149825 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147639, &mem_147639_cached_sizze_149825, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147644_cached_sizze_149826 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147644, &mem_147644_cached_sizze_149826, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147655_cached_sizze_149827 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147655, &mem_147655_cached_sizze_149827, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147660_cached_sizze_149828 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147660, &mem_147660_cached_sizze_149828, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147671_cached_sizze_149829 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147671, &mem_147671_cached_sizze_149829, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147678_cached_sizze_149830 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147678, &mem_147678_cached_sizze_149830, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147685_cached_sizze_149831 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147685, &mem_147685_cached_sizze_149831, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147695_cached_sizze_149832 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147695, &mem_147695_cached_sizze_149832, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147700_cached_sizze_149833 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147700, &mem_147700_cached_sizze_149833, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147711_cached_sizze_149834 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147711, &mem_147711_cached_sizze_149834, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147712_cached_sizze_149835 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147712, &mem_147712_cached_sizze_149835, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147721_cached_sizze_149836 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147721, &mem_147721_cached_sizze_149836, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147722_cached_sizze_149837 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147722, &mem_147722_cached_sizze_149837, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147743_cached_sizze_149838 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147743, &mem_147743_cached_sizze_149838, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147744_cached_sizze_149839 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147744, &mem_147744_cached_sizze_149839, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147745_cached_sizze_149840 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147745, &mem_147745_cached_sizze_149840, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147746_cached_sizze_149841 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147746, &mem_147746_cached_sizze_149841, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147767_cached_sizze_149842 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147767, &mem_147767_cached_sizze_149842, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147768_cached_sizze_149843 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147768, &mem_147768_cached_sizze_149843, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147769_cached_sizze_149844 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147769, &mem_147769_cached_sizze_149844, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147770_cached_sizze_149845 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147770, &mem_147770_cached_sizze_149845, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147787_cached_sizze_149846 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_147787, &mem_147787_cached_sizze_149846, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147794_cached_sizze_149847 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147794, &mem_147794_cached_sizze_149847, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147795_cached_sizze_149848 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147795, &mem_147795_cached_sizze_149848, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147796_cached_sizze_149849 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147796, &mem_147796_cached_sizze_149849, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147851_cached_sizze_149850 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147851, &mem_147851_cached_sizze_149850, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147852_cached_sizze_149851 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147852, &mem_147852_cached_sizze_149851, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147853_cached_sizze_149852 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147853, &mem_147853_cached_sizze_149852, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147854_cached_sizze_149853 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147854, &mem_147854_cached_sizze_149853, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147855_cached_sizze_149854 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147855, &mem_147855_cached_sizze_149854, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147856_cached_sizze_149855 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_147856, &mem_147856_cached_sizze_149855, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147857_cached_sizze_149856 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147857, &mem_147857_cached_sizze_149856, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147858_cached_sizze_149857 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147858, &mem_147858_cached_sizze_149857, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147859_cached_sizze_149858 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_147859, &mem_147859_cached_sizze_149858, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147901_cached_sizze_149859 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147901, &mem_147901_cached_sizze_149859, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147902_cached_sizze_149860 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147902, &mem_147902_cached_sizze_149860, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147903_cached_sizze_149861 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147903, &mem_147903_cached_sizze_149861, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147904_cached_sizze_149862 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147904, &mem_147904_cached_sizze_149862, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147905_cached_sizze_149863 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147905, &mem_147905_cached_sizze_149863, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147906_cached_sizze_149864 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147906, &mem_147906_cached_sizze_149864, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147907_cached_sizze_149865 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147907, &mem_147907_cached_sizze_149865, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147908_cached_sizze_149866 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147908, &mem_147908_cached_sizze_149866, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_147909_cached_sizze_149867 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_147909, &mem_147909_cached_sizze_149867, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:115:13-33
    if (mem_147942_cached_sizze_149868 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147942, &mem_147942_cached_sizze_149868, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:115:13-33
    if (mem_147943_cached_sizze_149869 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_147943, &mem_147943_cached_sizze_149869, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148032_cached_sizze_149870 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_148032, &mem_148032_cached_sizze_149870, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148033_cached_sizze_149871 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_148033, &mem_148033_cached_sizze_149871, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148034_cached_sizze_149872 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_148034, &mem_148034_cached_sizze_149872, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148050_cached_sizze_149873 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148050, &mem_148050_cached_sizze_149873, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148051_cached_sizze_149874 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148051, &mem_148051_cached_sizze_149874, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148052_cached_sizze_149875 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148052, &mem_148052_cached_sizze_149875, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148065_cached_sizze_149876 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148065, &mem_148065_cached_sizze_149876, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148066_cached_sizze_149877 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148066, &mem_148066_cached_sizze_149877, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148067_cached_sizze_149878 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148067, &mem_148067_cached_sizze_149878, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148086_cached_sizze_149879 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148086, &mem_148086_cached_sizze_149879, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148120_cached_sizze_149880 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_148120, &mem_148120_cached_sizze_149880, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148121_cached_sizze_149881 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_148121, &mem_148121_cached_sizze_149881, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148122_cached_sizze_149882 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_148122, &mem_148122_cached_sizze_149882, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148123_cached_sizze_149883 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_148123, &mem_148123_cached_sizze_149883, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148124_cached_sizze_149884 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148124, &mem_148124_cached_sizze_149884, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148146_cached_sizze_149885 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148146, &mem_148146_cached_sizze_149885, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148147_cached_sizze_149886 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148147, &mem_148147_cached_sizze_149886, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148148_cached_sizze_149887 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148148, &mem_148148_cached_sizze_149887, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148149_cached_sizze_149888 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148149, &mem_148149_cached_sizze_149888, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148150_cached_sizze_149889 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_148150, &mem_148150_cached_sizze_149889, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148167_cached_sizze_149890 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_148167, &mem_148167_cached_sizze_149890, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148211_cached_sizze_149891 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_148211, &mem_148211_cached_sizze_149891, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148212_cached_sizze_149892 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_148212, &mem_148212_cached_sizze_149892, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148213_cached_sizze_149893 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_148213, &mem_148213_cached_sizze_149893, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148214_cached_sizze_149894 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_148214, &mem_148214_cached_sizze_149894, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148215_cached_sizze_149895 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_148215, &mem_148215_cached_sizze_149895, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148216_cached_sizze_149896 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_148216, &mem_148216_cached_sizze_149896, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148243_cached_sizze_149897 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148243, &mem_148243_cached_sizze_149897, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148244_cached_sizze_149898 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148244, &mem_148244_cached_sizze_149898, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148245_cached_sizze_149899 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148245, &mem_148245_cached_sizze_149899, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148246_cached_sizze_149900 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148246, &mem_148246_cached_sizze_149900, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148247_cached_sizze_149901 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148247, &mem_148247_cached_sizze_149901, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148248_cached_sizze_149902 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148248, &mem_148248_cached_sizze_149902, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148269_cached_sizze_149903 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148269, &mem_148269_cached_sizze_149903, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148270_cached_sizze_149904 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148270, &mem_148270_cached_sizze_149904, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148329_cached_sizze_149905 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_148329, &mem_148329_cached_sizze_149905, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148330_cached_sizze_149906 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_148330, &mem_148330_cached_sizze_149906, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148331_cached_sizze_149907 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_148331, &mem_148331_cached_sizze_149907, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148332_cached_sizze_149908 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_148332, &mem_148332_cached_sizze_149908, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148353_cached_sizze_149909 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148353, &mem_148353_cached_sizze_149909, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148354_cached_sizze_149910 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148354, &mem_148354_cached_sizze_149910, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148355_cached_sizze_149911 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148355, &mem_148355_cached_sizze_149911, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148356_cached_sizze_149912 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148356, &mem_148356_cached_sizze_149912, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148373_cached_sizze_149913 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148373, &mem_148373_cached_sizze_149913, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148374_cached_sizze_149914 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148374, &mem_148374_cached_sizze_149914, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148375_cached_sizze_149915 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148375, &mem_148375_cached_sizze_149915, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148376_cached_sizze_149916 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148376, &mem_148376_cached_sizze_149916, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148437_cached_sizze_149917 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_148437, &mem_148437_cached_sizze_149917, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148438_cached_sizze_149918 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_148438, &mem_148438_cached_sizze_149918, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148449_cached_sizze_149919 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148449, &mem_148449_cached_sizze_149919, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148450_cached_sizze_149920 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148450, &mem_148450_cached_sizze_149920, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148459_cached_sizze_149921 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148459, &mem_148459_cached_sizze_149921, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148460_cached_sizze_149922 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148460, &mem_148460_cached_sizze_149922, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148491_cached_sizze_149923 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_148491, &mem_148491_cached_sizze_149923, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148492_cached_sizze_149924 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_148492, &mem_148492_cached_sizze_149924, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148503_cached_sizze_149925 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148503, &mem_148503_cached_sizze_149925, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148504_cached_sizze_149926 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148504, &mem_148504_cached_sizze_149926, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148513_cached_sizze_149927 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148513, &mem_148513_cached_sizze_149927, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148514_cached_sizze_149928 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148514, &mem_148514_cached_sizze_149928, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148545_cached_sizze_149929 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148545, &mem_148545_cached_sizze_149929, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148546_cached_sizze_149930 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148546, &mem_148546_cached_sizze_149930, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148557_cached_sizze_149931 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_148557, &mem_148557_cached_sizze_149931, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148558_cached_sizze_149932 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_148558, &mem_148558_cached_sizze_149932, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148567_cached_sizze_149933 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_148567, &mem_148567_cached_sizze_149933, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148568_cached_sizze_149934 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_148568, &mem_148568_cached_sizze_149934, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148599_cached_sizze_149935 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148599, &mem_148599_cached_sizze_149935, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148600_cached_sizze_149936 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148600, &mem_148600_cached_sizze_149936, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148601_cached_sizze_149937 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148601, &mem_148601_cached_sizze_149937, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148614_cached_sizze_149938 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148614, &mem_148614_cached_sizze_149938, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148615_cached_sizze_149939 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148615, &mem_148615_cached_sizze_149939, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148616_cached_sizze_149940 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148616, &mem_148616_cached_sizze_149940, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148647_cached_sizze_149941 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148647, &mem_148647_cached_sizze_149941, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148648_cached_sizze_149942 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148648, &mem_148648_cached_sizze_149942, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148649_cached_sizze_149943 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148649, &mem_148649_cached_sizze_149943, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148650_cached_sizze_149944 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148650, &mem_148650_cached_sizze_149944, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148667_cached_sizze_149945 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148667, &mem_148667_cached_sizze_149945, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148668_cached_sizze_149946 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148668, &mem_148668_cached_sizze_149946, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148669_cached_sizze_149947 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148669, &mem_148669_cached_sizze_149947, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148670_cached_sizze_149948 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148670, &mem_148670_cached_sizze_149948, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148711_cached_sizze_149949 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148711, &mem_148711_cached_sizze_149949, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148718_cached_sizze_149950 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148718, &mem_148718_cached_sizze_149950, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148725_cached_sizze_149951 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148725, &mem_148725_cached_sizze_149951, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148735_cached_sizze_149952 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148735, &mem_148735_cached_sizze_149952, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148740_cached_sizze_149953 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148740, &mem_148740_cached_sizze_149953, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148751_cached_sizze_149954 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148751, &mem_148751_cached_sizze_149954, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148758_cached_sizze_149955 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148758, &mem_148758_cached_sizze_149955, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148765_cached_sizze_149956 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148765, &mem_148765_cached_sizze_149956, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148775_cached_sizze_149957 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148775, &mem_148775_cached_sizze_149957, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148780_cached_sizze_149958 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148780, &mem_148780_cached_sizze_149958, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148791_cached_sizze_149959 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148791, &mem_148791_cached_sizze_149959, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148792_cached_sizze_149960 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_148792, &mem_148792_cached_sizze_149960, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148801_cached_sizze_149961 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148801, &mem_148801_cached_sizze_149961, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148802_cached_sizze_149962 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148802, &mem_148802_cached_sizze_149962, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148823_cached_sizze_149963 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_148823, &mem_148823_cached_sizze_149963, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148828_cached_sizze_149964 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148828, &mem_148828_cached_sizze_149964, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148839_cached_sizze_149965 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_148839, &mem_148839_cached_sizze_149965, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148840_cached_sizze_149966 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_148840, &mem_148840_cached_sizze_149966, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148849_cached_sizze_149967 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148849, &mem_148849_cached_sizze_149967, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_148850_cached_sizze_149968 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_148850, &mem_148850_cached_sizze_149968, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:635:5-640:51
    if (memblock_set(ctx, &mem_param_146757, &wdown_mem_146724, "wdown_mem_146724") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146761, &wkey_mem_146725, "wkey_mem_146725") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146765, &wout_mem_146726, "wout_mem_146726") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146769, &wpe_mem_146727, "wpe_mem_146727") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146773, &wqry_mem_146728, "wqry_mem_146728") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146777, &wte_mem_146729, "wte_mem_146729") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146781, &wup_mem_146730, "wup_mem_146730") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146785, &wval_mem_146731, "wval_mem_146731") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146789, &wvoc_mem_146732, "wvoc_mem_146732") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146793, &wdown_mem_146733, "wdown_mem_146733") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146797, &wkey_mem_146734, "wkey_mem_146734") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146801, &wout_mem_146735, "wout_mem_146735") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146805, &wpe_mem_146736, "wpe_mem_146736") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146809, &wqry_mem_146737, "wqry_mem_146737") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146813, &wte_mem_146738, "wte_mem_146738") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146817, &wup_mem_146739, "wup_mem_146739") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146821, &wval_mem_146740, "wval_mem_146740") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146825, &wvoc_mem_146741, "wvoc_mem_146741") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146829, &wdown_mem_146742, "wdown_mem_146742") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146833, &wkey_mem_146743, "wkey_mem_146743") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146837, &wout_mem_146744, "wout_mem_146744") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146841, &wpe_mem_146745, "wpe_mem_146745") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146845, &wqry_mem_146746, "wqry_mem_146746") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146849, &wte_mem_146747, "wte_mem_146747") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146853, &wup_mem_146748, "wup_mem_146748") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146857, &wval_mem_146749, "wval_mem_146749") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_146861, &wvoc_mem_146750, "wvoc_mem_146750") != 0)
        return 1;
    for (int64_t step_131465 = 0; step_131465 < (int64_t) 50; step_131465++) {
        // futhark/microgpt.fut:637:16-25
        
        int64_t dl_131493 = ((int64_t *) dls_mem_146752.mem)[step_131465];
        
        // futhark/microgpt.fut:477:37-40
        
        int64_t zl_rhs_131498 = sub64(dl_131493, (int64_t) 1);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145638 = 0; i_145638 < (int64_t) 16; i_145638++) {
            // futhark/microgpt.fut:477:25-81
            
            bool cond_134200 = slt64(i_145638, zl_rhs_131498);
            
            // futhark/microgpt.fut:477:56-59
            
            int64_t zeze_lhs_134201 = add64((int64_t) 1, i_145638);
            
            // futhark/microgpt.fut:477:47-60
            
            bool x_134202 = sle64((int64_t) 0, zeze_lhs_134201);
            
            // futhark/microgpt.fut:477:47-60
            
            bool y_134203 = slt64(zeze_lhs_134201, (int64_t) 16);
            
            // futhark/microgpt.fut:477:47-60
            
            bool bounds_check_134204 = x_134202 && y_134203;
            
            // futhark/microgpt.fut:9:27-39
            
            bool loop_not_taken_134205 = !cond_134200;
            
            // futhark/microgpt.fut:9:27-39
            
            bool protect_assert_disj_134206 = bounds_check_134204 || loop_not_taken_134205;
            
            // futhark/microgpt.fut:477:47-60
            
            bool index_certs_134207;
            
            if (!protect_assert_disj_134206) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_134201, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:477:47-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:477:3-83\n   #6  futhark/microgpt.fut:584:18-38\n   #7  futhark/microgpt.fut:606:26-612:31\n   #8  futhark/microgpt.fut:640:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            int64_t tmp_134222 = ((int64_t *) seqs_mem_146753.mem)[step_131465 * (int64_t) 16 + i_145638];
            
            // futhark/microgpt.fut:586:37-51
            
            bool x_134223 = sle64((int64_t) 0, tmp_134222);
            
            // futhark/microgpt.fut:586:37-51
            
            bool y_134224 = slt64(tmp_134222, (int64_t) 27);
            
            // futhark/microgpt.fut:586:37-51
            
            bool bounds_check_134225 = x_134223 && y_134224;
            
            // futhark/microgpt.fut:586:37-51
            
            bool index_certs_134226;
            
            if (!bounds_check_134225) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_134222, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:586:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:586:16-55\n   #6  futhark/microgpt.fut:606:26-612:31\n   #7  futhark/microgpt.fut:640:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:477:47-60
            
            int64_t zeze_lhs_134208;
            
            if (cond_134200) {
                int64_t x_145374 = ((int64_t *) seqs_mem_146753.mem)[step_131465 * (int64_t) 16 + zeze_lhs_134201];
                
                zeze_lhs_134208 = x_145374;
            } else {
                zeze_lhs_134208 = (int64_t) 0;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145628 = 0; i_145628 < (int64_t) 27; i_145628++) {
                // futhark/microgpt.fut:477:61-65
                
                bool cond_t_res_134212 = zeze_lhs_134208 == i_145628;
                
                // futhark/microgpt.fut:9:27-39
                
                bool x_134213 = cond_134200 && cond_t_res_134212;
                
                // futhark/microgpt.fut:477:25-81
                
                double lifted_lambda_res_134214;
                
                if (x_134213) {
                    lifted_lambda_res_134214 = 1.0;
                } else {
                    lifted_lambda_res_134214 = 0.0;
                }
                ((double *) mem_146872)[i_145628] = lifted_lambda_res_134214;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145632 = 0; i_145632 < (int64_t) 16; i_145632++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_134233 = ((double *) mem_param_146777.mem)[tmp_134222 * (int64_t) 16 + i_145632];
                
                ((double *) mem_146879)[i_145632] = lifted_lambda_res_134233;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146862, i_145638 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146879, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146863, i_145638 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146872, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145653 = 0; i_145653 < (int64_t) 16; i_145653++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145643 = 0; i_145643 < (int64_t) 16; i_145643++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_134258 = ((double *) mem_param_146769.mem)[i_145653 * (int64_t) 16 + i_145643];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_134259 = ((double *) mem_146862)[i_145653 * (int64_t) 16 + i_145643];
                
                // futhark/microgpt.fut:279:39-75
                
                double zp_res_134260 = zp_lhs_134258 + zp_rhs_134259;
                
                ((double *) mem_146904)[i_145643] = zp_res_134260;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145647 = 0; i_145647 < (int64_t) 27; i_145647++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_134274 = ((double *) mem_146863)[i_145653 * (int64_t) 27 + i_145647];
                
                // futhark/microgpt.fut:314:54-96
                
                double zt_res_134275 = -6.25e-2 * zt_rhs_134274;
                
                ((double *) mem_146911)[i_145647] = zt_res_134275;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146894, i_145653 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146911, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146895, i_145653 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146904, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145671 = 0; i_145671 < (int64_t) 16; i_145671++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145660 = 0; i_145660 < (int64_t) 16; i_145660++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_139701 = ((double *) mem_146895)[i_145671 * (int64_t) 16 + i_145660];
                
                // futhark/microgpt.fut:280:69-102
                
                double zt_res_139702 = zt_lhs_139701 * zt_lhs_139701;
                
                ((double *) mem_146936)[i_145660] = zt_res_139702;
                ((double *) mem_146937)[i_145660] = zt_res_139702;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_134303;
            double r_134305 = 0.0;
            
            for (int64_t i_134304 = 0; i_134304 < (int64_t) 16; i_134304++) {
                // futhark/microgpt.fut:281:35-43
                
                double lifted_lambda_res_134306 = ((double *) mem_146937)[i_134304];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_134307 = r_134305 + lifted_lambda_res_134306;
                double r_tmp_149243 = zp_res_134307;
                
                r_134305 = r_tmp_149243;
            }
            defunc_0_lifted_lambda_res_134303 = r_134305;
            // futhark/microgpt.fut:281:16-60
            
            double zs_res_134308 = defunc_0_lifted_lambda_res_134303 / 16.0;
            
            // futhark/microgpt.fut:282:23-53
            
            double zp_res_134309 = 1.0e-5 + zs_res_134308;
            
            // futhark/microgpt.fut:282:15-53
            
            double sqrt_res_134310 = futrts_sqrt64(zp_res_134309);
            
            // futhark/microgpt.fut:283:25-35
            
            double zs_res_134311 = 1.0 / sqrt_res_134310;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145665 = 0; i_145665 < (int64_t) 16; i_145665++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_134318 = ((double *) mem_146895)[i_145671 * (int64_t) 16 + i_145665];
                
                // futhark/microgpt.fut:283:5-35
                
                double zt_res_134319 = zs_res_134311 * zt_lhs_134318;
                
                ((double *) mem_146950)[i_145665] = zt_res_134319;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146926, i_145671 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146936, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146927, i_145671 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146950, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145691 = 0; i_145691 < (int64_t) 16; i_145691++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145678 = 0; i_145678 < (int64_t) 16; i_145678++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_139731 = ((double *) mem_146927)[i_145691 * (int64_t) 16 + i_145678];
                
                // futhark/microgpt.fut:284:73-110
                
                double zt_res_139732 = zt_lhs_139731 * zt_lhs_139731;
                
                ((double *) mem_146979)[i_145678] = zt_res_139732;
                ((double *) mem_146980)[i_145678] = zt_res_139732;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_134427;
            double r_134429 = 0.0;
            
            for (int64_t i_134428 = 0; i_134428 < (int64_t) 16; i_134428++) {
                // futhark/microgpt.fut:285:37-47
                
                double lifted_lambda_res_134430 = ((double *) mem_146980)[i_134428];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_134431 = r_134429 + lifted_lambda_res_134430;
                double r_tmp_149250 = zp_res_134431;
                
                r_134429 = r_tmp_149250;
            }
            defunc_0_lifted_lambda_res_134427 = r_134429;
            // futhark/microgpt.fut:285:17-64
            
            double zs_res_134432 = defunc_0_lifted_lambda_res_134427 / 16.0;
            
            // futhark/microgpt.fut:286:24-55
            
            double zp_res_134433 = 1.0e-5 + zs_res_134432;
            
            // futhark/microgpt.fut:286:16-55
            
            double sqrt_res_134434 = futrts_sqrt64(zp_res_134433);
            
            // futhark/microgpt.fut:287:27-38
            
            double zs_res_134435 = 1.0 / sqrt_res_134434;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145683 = 0; i_145683 < (int64_t) 16; i_145683++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_134442 = ((double *) mem_146927)[i_145691 * (int64_t) 16 + i_145683];
                
                // futhark/microgpt.fut:287:5-38
                
                double zt_res_134443 = zs_res_134435 * zt_lhs_134442;
                
                ((double *) mem_146993)[i_145683] = zt_res_134443;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_134469;
            double r_134471 = 0.0;
            
            for (int64_t i_134470 = 0; i_134470 < (int64_t) 16; i_134470++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_134472 = ((double *) mem_146926)[i_145691 * (int64_t) 16 + i_134470];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_134473 = r_134471 + lifted_lambda_res_134472;
                double r_tmp_149252 = zp_res_134473;
                
                r_134471 = r_tmp_149252;
            }
            defunc_0_lifted_lambda_res_134469 = r_134471;
            // futhark/microgpt.fut:390:40-98
            
            double zs_res_134474 = defunc_0_lifted_lambda_res_134469 / 16.0;
            
            ((double *) mem_146965)[i_145691] = zs_res_134474;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146966, i_145691 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146979, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_146967, i_145691 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_146993, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145715 = 0; i_145715 < (int64_t) 16; i_145715++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145701 = 0; i_145701 < (int64_t) 16; i_145701++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_139804;
                double r_139806 = 0.0;
                
                for (int64_t i_139805 = 0; i_139805 < (int64_t) 16; i_139805++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_139807 = ((double *) mem_param_146773.mem)[i_145701 * (int64_t) 16 + i_139805];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_139808 = ((double *) mem_146967)[i_145715 * (int64_t) 16 + i_139805];
                    
                    // futhark/microgpt.fut:288:63-102
                    
                    double zt_res_139809 = zt_lhs_139807 * zt_rhs_139808;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_139810 = r_139806 + zt_res_139809;
                    double r_tmp_149261 = zp_res_139810;
                    
                    r_139806 = r_tmp_149261;
                }
                defunc_0_lifted_lambda_res_139804 = r_139806;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_139817;
                double r_139819 = 0.0;
                
                for (int64_t i_139818 = 0; i_139818 < (int64_t) 16; i_139818++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_139820 = ((double *) mem_param_146761.mem)[i_145701 * (int64_t) 16 + i_139818];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_139821 = ((double *) mem_146967)[i_145715 * (int64_t) 16 + i_139818];
                    
                    // futhark/microgpt.fut:289:63-102
                    
                    double zt_res_139822 = zt_lhs_139820 * zt_rhs_139821;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_139823 = r_139819 + zt_res_139822;
                    double r_tmp_149262 = zp_res_139823;
                    
                    r_139819 = r_tmp_149262;
                }
                defunc_0_lifted_lambda_res_139817 = r_139819;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_139833;
                double r_139835 = 0.0;
                
                for (int64_t i_139834 = 0; i_139834 < (int64_t) 16; i_139834++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_139836 = ((double *) mem_param_146785.mem)[i_145701 * (int64_t) 16 + i_139834];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_139837 = ((double *) mem_146967)[i_145715 * (int64_t) 16 + i_139834];
                    
                    // futhark/microgpt.fut:290:63-102
                    
                    double zt_res_139838 = zt_lhs_139836 * zt_rhs_139837;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_139839 = r_139835 + zt_res_139838;
                    double r_tmp_149263 = zp_res_139839;
                    
                    r_139835 = r_tmp_149263;
                }
                defunc_0_lifted_lambda_res_139833 = r_139835;
                ((double *) mem_147034)[i_145701] = defunc_0_lifted_lambda_res_139833;
                ((double *) mem_147035)[i_145701] = defunc_0_lifted_lambda_res_139817;
                ((double *) mem_147036)[i_145701] = defunc_0_lifted_lambda_res_139804;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_134816;
            double r_134818 = 0.0;
            
            for (int64_t i_134817 = 0; i_134817 < (int64_t) 16; i_134817++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_134819 = ((double *) mem_146966)[i_145715 * (int64_t) 16 + i_134817];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_134820 = r_134818 + lifted_lambda_res_134819;
                double r_tmp_149264 = zp_res_134820;
                
                r_134818 = r_tmp_149264;
            }
            defunc_0_lifted_lambda_res_134816 = r_134818;
            // futhark/microgpt.fut:383:40-98
            
            double zs_res_134821 = defunc_0_lifted_lambda_res_134816 / 16.0;
            
            // futhark/microgpt.fut:391:47-59
            
            double zp_lhs_134835 = ((double *) mem_146965)[i_145715];
            
            // futhark/microgpt.fut:391:47-87
            
            double zp_res_134836 = 1.0e-5 + zp_lhs_134835;
            
            // futhark/microgpt.fut:391:39-87
            
            double sqrt_res_134837 = futrts_sqrt64(zp_res_134836);
            
            ((double *) mem_147011)[i_145715] = sqrt_res_134837;
            ((double *) mem_147012)[i_145715] = zs_res_134821;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147013, i_145715 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147034, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147014, i_145715 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147035, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147015, i_145715 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147036, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145747 = 0; i_145747 < (int64_t) 4; i_145747++) {
            // futhark/microgpt.fut:291:67-70
            
            int64_t zp_lhs_134909 = mul64((int64_t) 4, i_145747);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145737 = 0; i_145737 < (int64_t) 16; i_145737++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_145727 = 0; i_145727 < (int64_t) 4; i_145727++) {
                    // futhark/microgpt.fut:291:72-79
                    
                    int64_t tmp_139997 = add64(zp_lhs_134909, i_145727);
                    
                    // futhark/microgpt.fut:291:48-81
                    
                    bool x_139998 = sle64((int64_t) 0, tmp_139997);
                    
                    // futhark/microgpt.fut:291:48-81
                    
                    bool y_139999 = slt64(tmp_139997, (int64_t) 16);
                    
                    // futhark/microgpt.fut:291:48-81
                    
                    bool bounds_check_140000 = x_139998 && y_139999;
                    
                    // futhark/microgpt.fut:291:48-81
                    
                    bool index_certs_140001;
                    
                    if (!bounds_check_140000) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_139997, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:291:48-81\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:291:12-82\n   #9  futhark/microgpt.fut:589:5-76\n   #10 futhark/microgpt.fut:606:26-612:31\n   #11 futhark/microgpt.fut:640:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_140002 = ((double *) mem_147015)[i_145737 * (int64_t) 16 + tmp_139997];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_140010 = ((double *) mem_147014)[i_145737 * (int64_t) 16 + tmp_139997];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_140021 = ((double *) mem_147013)[i_145737 * (int64_t) 16 + tmp_139997];
                    
                    ((double *) mem_147106)[i_145727] = lifted_lambda_res_140021;
                    ((double *) mem_147107)[i_145727] = lifted_lambda_res_140010;
                    ((double *) mem_147108)[i_145727] = lifted_lambda_res_140002;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_147091, i_145737 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147106, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_147092, i_145737 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147107, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_147093, i_145737 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147108, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_147073, i_145747 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_147091, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_147074, i_145747 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_147092, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_147075, i_145747 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_147093, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145819 = 0; i_145819 < (int64_t) 4; i_145819++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145772 = 0; i_145772 < (int64_t) 16; i_145772++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_145759 = 0; i_145759 < (int64_t) 16; i_145759++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_140403;
                    double r_140405 = 0.0;
                    
                    for (int64_t i_140404 = 0; i_140404 < (int64_t) 4; i_140404++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_140406 = ((double *) mem_147075)[i_145819 * (int64_t) 64 + i_145772 * (int64_t) 4 + i_140404];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_140407 = ((double *) mem_147074)[i_145819 * (int64_t) 64 + i_145759 * (int64_t) 4 + i_140404];
                        
                        // futhark/microgpt.fut:294:110-163
                        
                        double zt_res_140408 = zt_lhs_140406 * zt_rhs_140407;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_140409 = r_140405 + zt_res_140408;
                        double r_tmp_149286 = zp_res_140409;
                        
                        r_140405 = r_tmp_149286;
                    }
                    defunc_0_lifted_lambda_res_140403 = r_140405;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_140416;
                    double r_140418 = 0.0;
                    
                    for (int64_t i_140417 = 0; i_140417 < (int64_t) 4; i_140417++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_140419 = ((double *) mem_147075)[i_145819 * (int64_t) 64 + i_145772 * (int64_t) 4 + i_140417];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_140420 = ((double *) mem_147074)[i_145819 * (int64_t) 64 + i_145759 * (int64_t) 4 + i_140417];
                        
                        // futhark/microgpt.fut:337:87-146
                        
                        double zt_res_140421 = zt_lhs_140419 * zt_rhs_140420;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_140422 = r_140418 + zt_res_140421;
                        double r_tmp_149287 = zp_res_140422;
                        
                        r_140418 = r_tmp_149287;
                    }
                    defunc_0_lifted_lambda_res_140416 = r_140418;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_140432;
                    double r_140434 = 0.0;
                    
                    for (int64_t i_140433 = 0; i_140433 < (int64_t) 4; i_140433++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_140435 = ((double *) mem_147075)[i_145819 * (int64_t) 64 + i_145772 * (int64_t) 4 + i_140433];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_140436 = ((double *) mem_147074)[i_145819 * (int64_t) 64 + i_145759 * (int64_t) 4 + i_140433];
                        
                        // futhark/microgpt.fut:344:87-146
                        
                        double zt_res_140437 = zt_lhs_140435 * zt_rhs_140436;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_140438 = r_140434 + zt_res_140437;
                        double r_tmp_149288 = zp_res_140438;
                        
                        r_140434 = r_tmp_149288;
                    }
                    defunc_0_lifted_lambda_res_140432 = r_140434;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_140450;
                    double r_140452 = 0.0;
                    
                    for (int64_t i_140451 = 0; i_140451 < (int64_t) 4; i_140451++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_140453 = ((double *) mem_147075)[i_145819 * (int64_t) 64 + i_145772 * (int64_t) 4 + i_140451];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_140454 = ((double *) mem_147074)[i_145819 * (int64_t) 64 + i_145759 * (int64_t) 4 + i_140451];
                        
                        // futhark/microgpt.fut:361:87-146
                        
                        double zt_res_140455 = zt_lhs_140453 * zt_rhs_140454;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_140456 = r_140452 + zt_res_140455;
                        double r_tmp_149289 = zp_res_140456;
                        
                        r_140452 = r_tmp_149289;
                    }
                    defunc_0_lifted_lambda_res_140450 = r_140452;
                    ((double *) mem_147198)[i_145759] = defunc_0_lifted_lambda_res_140450;
                    ((double *) mem_147199)[i_145759] = defunc_0_lifted_lambda_res_140432;
                    ((double *) mem_147200)[i_145759] = defunc_0_lifted_lambda_res_140416;
                    ((double *) mem_147201)[i_145759] = defunc_0_lifted_lambda_res_140403;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_147178, i_145772 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147198, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_147179, i_145772 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147199, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_147180, i_145772 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147200, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_147181, i_145772 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147201, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145783 = 0; i_145783 < (int64_t) 16; i_145783++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_145779 = 0; i_145779 < (int64_t) 16; i_145779++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_135385 = ((double *) mem_147181)[i_145783 * (int64_t) 16 + i_145779];
                    
                    // futhark/microgpt.fut:295:47-78
                    
                    double zs_res_135386 = zs_lhs_135385 / 2.0;
                    double zp_rhs_135387 = ((double *) masks_mem_146751.mem)[step_131465 * (int64_t) 256 + i_145783 * (int64_t) 16 + i_145779];
                    
                    // futhark/microgpt.fut:295:65-102
                    
                    double zp_res_135388 = zs_res_135386 + zp_rhs_135387;
                    
                    ((double *) mem_147247)[i_145779] = zp_res_135388;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_147242, i_145783 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147247, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145789 = 0; i_145789 < (int64_t) 16; i_145789++) {
                // futhark/microgpt.fut:115:13-33
                
                double defunc_0_reduce_res_145399;
                double redout_145785 = -INFINITY;
                
                for (int64_t i_145786 = 0; i_145786 < (int64_t) 16; i_145786++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_140478 = ((double *) mem_147242)[i_145789 * (int64_t) 16 + i_145786];
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_135409 = fmax64(lifted_lambda_res_140478, redout_145785);
                    double redout_tmp_149293 = max_res_135409;
                    
                    redout_145785 = redout_tmp_149293;
                }
                defunc_0_reduce_res_145399 = redout_145785;
                // futhark/microgpt.fut:4:11-25
                for (int64_t nest_i_149294 = 0; nest_i_149294 < (int64_t) 16; nest_i_149294++) {
                    ((double *) mem_147258)[i_145789 * (int64_t) 16 + nest_i_149294] = defunc_0_reduce_res_145399;
                }
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145801 = 0; i_145801 < (int64_t) 16; i_145801++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_145793 = 0; i_145793 < (int64_t) 16; i_145793++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_135424 = ((double *) mem_147242)[i_145801 * (int64_t) 16 + i_145793];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double neg_arg0_135425 = ((double *) mem_147258)[i_145801 * (int64_t) 16 + i_145793];
                    
                    // futhark/microgpt.fut:297:108-131
                    
                    double neg_res_135426 = -neg_arg0_135425;
                    
                    // futhark/microgpt.fut:297:85-131
                    
                    double zp_res_135427 = zp_lhs_135424 + neg_res_135426;
                    
                    // futhark/microgpt.fut:297:78-131
                    
                    double exp_res_135428 = futrts_exp64(zp_res_135427);
                    
                    ((double *) mem_147273)[i_145793] = exp_res_135428;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_135430;
                double r_135432 = 0.0;
                
                for (int64_t i_135431 = 0; i_135431 < (int64_t) 16; i_135431++) {
                    // futhark/microgpt.fut:298:45-55
                    
                    double lifted_lambda_res_135433 = ((double *) mem_147273)[i_135431];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_135434 = r_135432 + lifted_lambda_res_135433;
                    double r_tmp_149297 = zp_res_135434;
                    
                    r_135432 = r_tmp_149297;
                }
                defunc_0_lifted_lambda_res_135430 = r_135432;
                // futhark/microgpt.fut:298:16-56
                
                double zs_res_135435 = 1.0 / defunc_0_lifted_lambda_res_135430;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_145797 = 0; i_145797 < (int64_t) 16; i_145797++) {
                    // futhark/microgpt.fut:299:5-15
                    
                    double zt_lhs_135442 = ((double *) mem_147273)[i_145797];
                    
                    // futhark/microgpt.fut:299:5-23
                    
                    double zt_res_135443 = zs_res_135435 * zt_lhs_135442;
                    
                    ((double *) mem_147280)[i_145797] = zt_res_135443;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_147268, i_145801 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147280, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145809 = 0; i_145809 < (int64_t) 16; i_145809++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_145805 = 0; i_145805 < (int64_t) 4; i_145805++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_135458;
                    double r_135460 = 0.0;
                    
                    for (int64_t i_135459 = 0; i_135459 < (int64_t) 16; i_135459++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_135461 = ((double *) mem_147268)[i_145809 * (int64_t) 16 + i_135459];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_135462 = ((double *) mem_147073)[i_145819 * (int64_t) 64 + i_135459 * (int64_t) 4 + i_145805];
                        
                        // futhark/microgpt.fut:300:26-72
                        
                        double zt_res_135463 = zt_lhs_135461 * zt_rhs_135462;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_135464 = r_135460 + zt_res_135463;
                        double r_tmp_149301 = zp_res_135464;
                        
                        r_135460 = r_tmp_149301;
                    }
                    defunc_0_lifted_lambda_res_135458 = r_135460;
                    ((double *) mem_147296)[i_145805] = defunc_0_lifted_lambda_res_135458;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_147291, i_145809 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147296, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_147154, i_145819 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_147178, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_147155, i_145819 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_147179, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_147156, i_145819 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_147180, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_147157, i_145819 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_147291, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145832 = 0; i_145832 < (int64_t) 16; i_145832++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145826 = 0; i_145826 < (int64_t) 16; i_145826++) {
                // futhark/microgpt.fut:301:52-55
                
                int64_t tmp_135577 = sdiv64(i_145826, (int64_t) 4);
                
                // futhark/microgpt.fut:301:41-57
                
                bool x_135578 = sle64((int64_t) 0, tmp_135577);
                
                // futhark/microgpt.fut:301:41-57
                
                bool y_135579 = slt64(tmp_135577, (int64_t) 4);
                
                // futhark/microgpt.fut:301:41-57
                
                bool bounds_check_135580 = x_135578 && y_135579;
                
                // futhark/microgpt.fut:301:41-57
                
                bool index_certs_135581;
                
                if (!bounds_check_135580) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_135577, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:301:41-57\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:301:12-78\n   #6  futhark/microgpt.fut:589:5-76\n   #7  futhark/microgpt.fut:606:26-612:31\n   #8  futhark/microgpt.fut:640:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:301:72-75
                
                int64_t tmp_135582 = smod64(i_145826, (int64_t) 4);
                
                // futhark/microgpt.fut:301:41-77
                
                bool x_135583 = sle64((int64_t) 0, tmp_135582);
                
                // futhark/microgpt.fut:301:41-77
                
                bool y_135584 = slt64(tmp_135582, (int64_t) 4);
                
                // futhark/microgpt.fut:301:41-77
                
                bool bounds_check_135585 = x_135583 && y_135584;
                
                // futhark/microgpt.fut:301:41-77
                
                bool index_certs_135586;
                
                if (!bounds_check_135585) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_135582, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:301:41-77\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:301:12-78\n   #6  futhark/microgpt.fut:589:5-76\n   #7  futhark/microgpt.fut:606:26-612:31\n   #8  futhark/microgpt.fut:640:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_135587 = ((double *) mem_147157)[tmp_135577 * (int64_t) 64 + i_145832 * (int64_t) 4 + tmp_135582];
                
                ((double *) mem_147336)[i_145826] = lifted_lambda_res_135587;
            }
            // futhark/microgpt.fut:384:47-59
            
            double zp_lhs_135595 = ((double *) mem_147012)[i_145832];
            
            // futhark/microgpt.fut:384:47-87
            
            double zp_res_135596 = 1.0e-5 + zp_lhs_135595;
            
            // futhark/microgpt.fut:384:39-87
            
            double sqrt_res_135597 = futrts_sqrt64(zp_res_135596);
            
            ((double *) mem_147327)[i_145832] = sqrt_res_135597;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147328, i_145832 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147336, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145841 = 0; i_145841 < (int64_t) 16; i_145841++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145837 = 0; i_145837 < (int64_t) 16; i_145837++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_131902;
                double r_131904 = 0.0;
                
                for (int64_t i_131903 = 0; i_131903 < (int64_t) 16; i_131903++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_131905 = ((double *) mem_param_146765.mem)[i_145837 * (int64_t) 16 + i_131903];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_131906 = ((double *) mem_147328)[i_145841 * (int64_t) 16 + i_131903];
                    
                    // futhark/microgpt.fut:302:63-103
                    
                    double zt_res_131907 = zt_lhs_131905 * zt_rhs_131906;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_131908 = r_131904 + zt_res_131907;
                    double r_tmp_149307 = zp_res_131908;
                    
                    r_131904 = r_tmp_149307;
                }
                defunc_0_lifted_lambda_res_131902 = r_131904;
                ((double *) mem_147355)[i_145837] = defunc_0_lifted_lambda_res_131902;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147350, i_145841 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147355, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145849 = 0; i_145849 < (int64_t) 16; i_145849++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145845 = 0; i_145845 < (int64_t) 16; i_145845++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_131923 = ((double *) mem_147350)[i_145849 * (int64_t) 16 + i_145845];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_131924 = ((double *) mem_146927)[i_145849 * (int64_t) 16 + i_145845];
                
                // futhark/microgpt.fut:303:42-80
                
                double zp_res_131925 = zp_lhs_131923 + zp_rhs_131924;
                
                ((double *) mem_147371)[i_145845] = zp_res_131925;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147366, i_145849 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147371, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145866 = 0; i_145866 < (int64_t) 16; i_145866++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145855 = 0; i_145855 < (int64_t) 16; i_145855++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_140522 = ((double *) mem_147366)[i_145866 * (int64_t) 16 + i_145855];
                
                // futhark/microgpt.fut:304:74-113
                
                double zt_res_140523 = zt_lhs_140522 * zt_lhs_140522;
                
                ((double *) mem_147392)[i_145855] = zt_res_140523;
                ((double *) mem_147393)[i_145855] = zt_res_140523;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_135624;
            double r_135626 = 0.0;
            
            for (int64_t i_135625 = 0; i_135625 < (int64_t) 16; i_135625++) {
                // futhark/microgpt.fut:305:37-47
                
                double lifted_lambda_res_135627 = ((double *) mem_147393)[i_135625];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_135628 = r_135626 + lifted_lambda_res_135627;
                double r_tmp_149314 = zp_res_135628;
                
                r_135626 = r_tmp_149314;
            }
            defunc_0_lifted_lambda_res_135624 = r_135626;
            // futhark/microgpt.fut:305:17-64
            
            double zs_res_135629 = defunc_0_lifted_lambda_res_135624 / 16.0;
            
            // futhark/microgpt.fut:306:24-55
            
            double zp_res_135630 = 1.0e-5 + zs_res_135629;
            
            // futhark/microgpt.fut:306:16-55
            
            double sqrt_res_135631 = futrts_sqrt64(zp_res_135630);
            
            // futhark/microgpt.fut:307:28-39
            
            double zs_res_135632 = 1.0 / sqrt_res_135631;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145860 = 0; i_145860 < (int64_t) 16; i_145860++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_135639 = ((double *) mem_147366)[i_145866 * (int64_t) 16 + i_145860];
                
                // futhark/microgpt.fut:307:5-39
                
                double zt_res_135640 = zs_res_135632 * zt_lhs_135639;
                
                ((double *) mem_147406)[i_145860] = zt_res_135640;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147382, i_145866 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147392, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147383, i_145866 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147406, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145877 = 0; i_145877 < (int64_t) 16; i_145877++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145871 = 0; i_145871 < (int64_t) 64; i_145871++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_135680;
                double r_135682 = 0.0;
                
                for (int64_t i_135681 = 0; i_135681 < (int64_t) 16; i_135681++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_135683 = ((double *) mem_param_146781.mem)[i_145871 * (int64_t) 16 + i_135681];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_135684 = ((double *) mem_147383)[i_145877 * (int64_t) 16 + i_135681];
                    
                    // futhark/microgpt.fut:308:63-102
                    
                    double zt_res_135685 = zt_lhs_135683 * zt_rhs_135684;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_135686 = r_135682 + zt_res_135685;
                    double r_tmp_149319 = zp_res_135686;
                    
                    r_135682 = r_tmp_149319;
                }
                defunc_0_lifted_lambda_res_135680 = r_135682;
                ((double *) mem_147430)[i_145871] = defunc_0_lifted_lambda_res_135680;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_135694;
            double r_135696 = 0.0;
            
            for (int64_t i_135695 = 0; i_135695 < (int64_t) 16; i_135695++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_135697 = ((double *) mem_147382)[i_145877 * (int64_t) 16 + i_135695];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_135698 = r_135696 + lifted_lambda_res_135697;
                double r_tmp_149320 = zp_res_135698;
                
                r_135696 = r_tmp_149320;
            }
            defunc_0_lifted_lambda_res_135694 = r_135696;
            // futhark/microgpt.fut:329:40-98
            
            double zs_res_135699 = defunc_0_lifted_lambda_res_135694 / 16.0;
            
            ((double *) mem_147421)[i_145877] = zs_res_135699;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147422, i_145877 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147430, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145888 = 0; i_145888 < (int64_t) 16; i_145888++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145882 = 0; i_145882 < (int64_t) 64; i_145882++) {
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_135723 = ((double *) mem_147422)[i_145888 * (int64_t) 64 + i_145882];
                
                // futhark/microgpt.fut:309:41-69
                
                double max_res_135724 = fmax64(0.0, max_arg0_135723);
                
                ((double *) mem_147453)[i_145882] = max_res_135724;
            }
            // futhark/microgpt.fut:330:47-59
            
            double zp_lhs_135732 = ((double *) mem_147421)[i_145888];
            
            // futhark/microgpt.fut:330:47-87
            
            double zp_res_135733 = 1.0e-5 + zp_lhs_135732;
            
            // futhark/microgpt.fut:330:39-87
            
            double sqrt_res_135734 = futrts_sqrt64(zp_res_135733);
            
            ((double *) mem_147444)[i_145888] = sqrt_res_135734;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147445, i_145888 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147453, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145897 = 0; i_145897 < (int64_t) 16; i_145897++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145893 = 0; i_145893 < (int64_t) 16; i_145893++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_132012;
                double r_132014 = 0.0;
                
                for (int64_t i_132013 = 0; i_132013 < (int64_t) 64; i_132013++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_132015 = ((double *) mem_param_146757.mem)[i_145893 * (int64_t) 64 + i_132013];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_132016 = ((double *) mem_147445)[i_145897 * (int64_t) 64 + i_132013];
                    
                    // futhark/microgpt.fut:310:63-104
                    
                    double zt_res_132017 = zt_lhs_132015 * zt_rhs_132016;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_132018 = r_132014 + zt_res_132017;
                    double r_tmp_149326 = zp_res_132018;
                    
                    r_132014 = r_tmp_149326;
                }
                defunc_0_lifted_lambda_res_132012 = r_132014;
                ((double *) mem_147472)[i_145893] = defunc_0_lifted_lambda_res_132012;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147467, i_145897 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147472, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145905 = 0; i_145905 < (int64_t) 16; i_145905++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145901 = 0; i_145901 < (int64_t) 16; i_145901++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_132033 = ((double *) mem_147467)[i_145905 * (int64_t) 16 + i_145901];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_132034 = ((double *) mem_147366)[i_145905 * (int64_t) 16 + i_145901];
                
                // futhark/microgpt.fut:311:42-81
                
                double zp_res_132035 = zp_lhs_132033 + zp_rhs_132034;
                
                ((double *) mem_147488)[i_145901] = zp_res_132035;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147483, i_145905 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147488, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145913 = 0; i_145913 < (int64_t) 16; i_145913++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145909 = 0; i_145909 < (int64_t) 27; i_145909++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_132050;
                double r_132052 = 0.0;
                
                for (int64_t i_132051 = 0; i_132051 < (int64_t) 16; i_132051++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_132053 = ((double *) mem_param_146789.mem)[i_145909 * (int64_t) 16 + i_132051];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_132054 = ((double *) mem_147483)[i_145913 * (int64_t) 16 + i_132051];
                    
                    // futhark/microgpt.fut:312:63-103
                    
                    double zt_res_132055 = zt_lhs_132053 * zt_rhs_132054;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_132056 = r_132052 + zt_res_132055;
                    double r_tmp_149331 = zp_res_132056;
                    
                    r_132052 = r_tmp_149331;
                }
                defunc_0_lifted_lambda_res_132050 = r_132052;
                ((double *) mem_147504)[i_145909] = defunc_0_lifted_lambda_res_132050;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147499, i_145913 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147504, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145921 = 0; i_145921 < (int64_t) 16; i_145921++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145917 = 0; i_145917 < (int64_t) 27; i_145917++) {
                // futhark/microgpt.fut:4:11-25
                
                double exp_arg0_132087 = ((double *) mem_147499)[i_145921 * (int64_t) 27 + i_145917];
                
                // futhark/microgpt.fut:315:46-69
                
                double exp_res_132088 = futrts_exp64(exp_arg0_132087);
                
                ((double *) mem_147520)[i_145917] = exp_res_132088;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147515, i_145921 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147520, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145931 = 0; i_145931 < (int64_t) 16; i_145931++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_135752;
            double r_135754 = 0.0;
            
            for (int64_t i_135753 = 0; i_135753 < (int64_t) 27; i_135753++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_135755 = ((double *) mem_147515)[i_145931 * (int64_t) 27 + i_135753];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_135756 = r_135754 + lifted_lambda_res_135755;
                double r_tmp_149336 = zp_res_135756;
                
                r_135754 = r_tmp_149336;
            }
            defunc_0_lifted_lambda_res_135752 = r_135754;
            // futhark/microgpt.fut:316:37-84
            
            double zs_res_135757 = 1.0 / defunc_0_lifted_lambda_res_135752;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_135764;
            double r_135766 = 0.0;
            
            for (int64_t i_135765 = 0; i_135765 < (int64_t) 27; i_135765++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_145925 = 0; i_145925 < (int64_t) 27; i_145925++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double exp_arg0_135773 = ((double *) mem_147499)[i_145931 * (int64_t) 27 + i_145925];
                    
                    // futhark/microgpt.fut:317:90-113
                    
                    double exp_res_135774 = futrts_exp64(exp_arg0_135773);
                    
                    ((double *) mem_147539)[i_145925] = exp_res_135774;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_135776;
                double r_135778 = 0.0;
                
                for (int64_t i_135777 = 0; i_135777 < (int64_t) 27; i_135777++) {
                    // futhark/microgpt.fut:318:45-55
                    
                    double lifted_lambda_res_135779 = ((double *) mem_147539)[i_135777];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_135780 = r_135778 + lifted_lambda_res_135779;
                    double r_tmp_149339 = zp_res_135780;
                    
                    r_135778 = r_tmp_149339;
                }
                defunc_0_lifted_lambda_res_135776 = r_135778;
                // futhark/microgpt.fut:318:16-56
                
                double zs_res_135781 = 1.0 / defunc_0_lifted_lambda_res_135776;
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_135782 = ((double *) mem_146894)[i_145931 * (int64_t) 27 + i_135765];
                
                // futhark/microgpt.fut:319:38-48
                
                double zt_lhs_135783 = ((double *) mem_147539)[i_135765];
                
                // futhark/microgpt.fut:319:38-56
                
                double zt_res_135784 = zs_res_135781 * zt_lhs_135783;
                
                // futhark/microgpt.fut:319:29-56
                
                double zs_res_135785 = 1.0 / zt_res_135784;
                
                // futhark/microgpt.fut:319:6-56
                
                double zt_res_135786 = zt_lhs_135782 * zs_res_135785;
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_135787 = ((double *) mem_147515)[i_145931 * (int64_t) 27 + i_135765];
                
                // futhark/microgpt.fut:319:24-81
                
                double zt_res_135788 = zt_res_135786 * zt_rhs_135787;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_135789 = r_135766 + zt_res_135788;
                double r_tmp_149337 = zp_res_135789;
                
                r_135766 = r_tmp_149337;
            }
            defunc_0_lifted_lambda_res_135764 = r_135766;
            ((double *) mem_147531)[i_145931] = defunc_0_lifted_lambda_res_135764;
            ((double *) mem_147532)[i_145931] = zs_res_135757;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145944 = 0; i_145944 < (int64_t) 16; i_145944++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145936 = 0; i_145936 < (int64_t) 27; i_145936++) {
                // futhark/microgpt.fut:4:11-25
                
                double exp_arg0_132149 = ((double *) mem_147499)[i_145944 * (int64_t) 27 + i_145936];
                
                // futhark/microgpt.fut:320:78-101
                
                double exp_res_132150 = futrts_exp64(exp_arg0_132149);
                
                ((double *) mem_147557)[i_145936] = exp_res_132150;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_132152;
            double r_132154 = 0.0;
            
            for (int64_t i_132153 = 0; i_132153 < (int64_t) 27; i_132153++) {
                // futhark/microgpt.fut:321:46-57
                
                double lifted_lambda_res_132155 = ((double *) mem_147557)[i_132153];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_132156 = r_132154 + lifted_lambda_res_132155;
                double r_tmp_149342 = zp_res_132156;
                
                r_132154 = r_tmp_149342;
            }
            defunc_0_lifted_lambda_res_132152 = r_132154;
            // futhark/microgpt.fut:321:16-58
            
            double zs_res_132157 = 1.0 / defunc_0_lifted_lambda_res_132152;
            
            // futhark/microgpt.fut:322:65-75
            
            double zt_rhs_132158 = ((double *) mem_147532)[i_145944];
            
            // futhark/microgpt.fut:322:89-99
            
            double zt_lhs_132159 = ((double *) mem_147531)[i_145944];
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_132160;
            double r_132162 = 0.0;
            
            for (int64_t i_132161 = 0; i_132161 < (int64_t) 27; i_132161++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_132163 = ((double *) mem_147515)[i_145944 * (int64_t) 27 + i_132161];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_132164 = r_132162 + lifted_lambda_res_132163;
                double r_tmp_149343 = zp_res_132164;
                
                r_132162 = r_tmp_149343;
            }
            defunc_0_lifted_lambda_res_132160 = r_132162;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_132165;
            double r_132167 = 0.0;
            
            for (int64_t i_132166 = 0; i_132166 < (int64_t) 27; i_132166++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_132168 = ((double *) mem_147515)[i_145944 * (int64_t) 27 + i_132166];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_132169 = r_132167 + lifted_lambda_res_132168;
                double r_tmp_149344 = zp_res_132169;
                
                r_132167 = r_tmp_149344;
            }
            defunc_0_lifted_lambda_res_132165 = r_132167;
            // futhark/microgpt.fut:322:115-198
            
            double zt_res_132170 = defunc_0_lifted_lambda_res_132160 * defunc_0_lifted_lambda_res_132165;
            
            // futhark/microgpt.fut:322:105-198
            
            double zs_res_132171 = 1.0 / zt_res_132170;
            
            // futhark/microgpt.fut:322:89-198
            
            double zt_res_132172 = zt_lhs_132159 * zs_res_132171;
            
            // futhark/microgpt.fut:322:82-198
            
            double neg_res_132173 = -zt_res_132172;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145940 = 0; i_145940 < (int64_t) 27; i_145940++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_132180 = ((double *) mem_146894)[i_145944 * (int64_t) 27 + i_145940];
                
                // futhark/microgpt.fut:322:39-49
                
                double zt_lhs_132181 = ((double *) mem_147557)[i_145940];
                
                // futhark/microgpt.fut:322:39-57
                
                double zt_res_132182 = zs_res_132157 * zt_lhs_132181;
                
                // futhark/microgpt.fut:322:30-57
                
                double zs_res_132183 = 1.0 / zt_res_132182;
                
                // futhark/microgpt.fut:322:7-57
                
                double zt_res_132184 = zt_lhs_132180 * zs_res_132183;
                
                // futhark/microgpt.fut:322:25-75
                
                double zt_res_132185 = zt_rhs_132158 * zt_res_132184;
                
                // futhark/microgpt.fut:322:61-202
                
                double zp_res_132186 = neg_res_132173 + zt_res_132185;
                
                ((double *) mem_147564)[i_145940] = zp_res_132186;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147552, i_145944 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147564, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145952 = 0; i_145952 < (int64_t) 16; i_145952++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145948 = 0; i_145948 < (int64_t) 27; i_145948++) {
                // futhark/microgpt.fut:4:11-25
                
                double exp_arg0_132201 = ((double *) mem_147499)[i_145952 * (int64_t) 27 + i_145948];
                
                // futhark/microgpt.fut:323:36-59
                
                double exp_res_132202 = futrts_exp64(exp_arg0_132201);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_132203 = ((double *) mem_147552)[i_145952 * (int64_t) 27 + i_145948];
                
                // futhark/microgpt.fut:323:36-82
                
                double zt_res_132204 = exp_res_132202 * zt_rhs_132203;
                
                ((double *) mem_147580)[i_145948] = zt_res_132204;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147575, i_145952 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147580, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145960 = 0; i_145960 < (int64_t) 16; i_145960++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145956 = 0; i_145956 < (int64_t) 16; i_145956++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_132219;
                double r_132221 = 0.0;
                
                for (int64_t i_132220 = 0; i_132220 < (int64_t) 27; i_132220++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_132222 = ((double *) mem_147575)[i_145960 * (int64_t) 27 + i_132220];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_132223 = ((double *) mem_param_146789.mem)[i_132220 * (int64_t) 16 + i_145956];
                    
                    // futhark/microgpt.fut:324:67-111
                    
                    double zt_res_132224 = zt_lhs_132222 * zt_rhs_132223;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_132225 = r_132221 + zt_res_132224;
                    double r_tmp_149350 = zp_res_132225;
                    
                    r_132221 = r_tmp_149350;
                }
                defunc_0_lifted_lambda_res_132219 = r_132221;
                ((double *) mem_147596)[i_145956] = defunc_0_lifted_lambda_res_132219;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147591, i_145960 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147596, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145973 = 0; i_145973 < (int64_t) 16; i_145973++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145966 = 0; i_145966 < (int64_t) 64; i_145966++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_140563;
                double r_140565 = 0.0;
                
                for (int64_t i_140564 = 0; i_140564 < (int64_t) 16; i_140564++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_140566 = ((double *) mem_147591)[i_145973 * (int64_t) 16 + i_140564];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_140567 = ((double *) mem_param_146757.mem)[i_140564 * (int64_t) 64 + i_145966];
                    
                    // futhark/microgpt.fut:325:67-113
                    
                    double zt_res_140568 = zt_lhs_140566 * zt_rhs_140567;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_140569 = r_140565 + zt_res_140568;
                    double r_tmp_149355 = zp_res_140569;
                    
                    r_140565 = r_tmp_149355;
                }
                defunc_0_lifted_lambda_res_140563 = r_140565;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_140576;
                double r_140578 = 0.0;
                
                for (int64_t i_140577 = 0; i_140577 < (int64_t) 16; i_140577++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_140579 = ((double *) mem_147591)[i_140577 * (int64_t) 16 + i_145973];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_140580 = ((double *) mem_147445)[i_140577 * (int64_t) 64 + i_145966];
                    
                    // futhark/microgpt.fut:419:69-113
                    
                    double zt_res_140581 = zt_lhs_140579 * zt_rhs_140580;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_140582 = r_140578 + zt_res_140581;
                    double r_tmp_149356 = zp_res_140582;
                    
                    r_140578 = r_tmp_149356;
                }
                defunc_0_lifted_lambda_res_140576 = r_140578;
                ((double *) mem_147617)[i_145966] = defunc_0_lifted_lambda_res_140576;
                ((double *) mem_147618)[i_145966] = defunc_0_lifted_lambda_res_140563;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147607, i_145973 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147617, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147608, i_145973 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147618, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145982 = 0; i_145982 < (int64_t) 16; i_145982++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145978 = 0; i_145978 < (int64_t) 64; i_145978++) {
                // futhark/microgpt.fut:4:11-25
                
                double indicatorp_arg0_132261 = ((double *) mem_147422)[i_145982 * (int64_t) 64 + i_145978];
                
                // futhark/microgpt.fut:110:42-54
                
                double max_res_132262 = fmax64(0.0, indicatorp_arg0_132261);
                
                // futhark/microgpt.fut:110:35-54
                
                double sgn_res_132263 = fsignum64(max_res_132262);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_132264 = ((double *) mem_147608)[i_145982 * (int64_t) 64 + i_145978];
                
                // futhark/microgpt.fut:326:46-102
                
                double zt_res_132265 = sgn_res_132263 * zt_rhs_132264;
                
                ((double *) mem_147644)[i_145978] = zt_res_132265;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147639, i_145982 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147644, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145990 = 0; i_145990 < (int64_t) 16; i_145990++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_145986 = 0; i_145986 < (int64_t) 16; i_145986++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_132280;
                double r_132282 = 0.0;
                
                for (int64_t i_132281 = 0; i_132281 < (int64_t) 64; i_132281++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_132283 = ((double *) mem_147639)[i_145990 * (int64_t) 64 + i_132281];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_132284 = ((double *) mem_param_146781.mem)[i_132281 * (int64_t) 16 + i_145986];
                    
                    // futhark/microgpt.fut:327:67-111
                    
                    double zt_res_132285 = zt_lhs_132283 * zt_rhs_132284;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_132286 = r_132282 + zt_res_132285;
                    double r_tmp_149361 = zp_res_132286;
                    
                    r_132282 = r_tmp_149361;
                }
                defunc_0_lifted_lambda_res_132280 = r_132282;
                ((double *) mem_147660)[i_145986] = defunc_0_lifted_lambda_res_132280;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147655, i_145990 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147660, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145994 = 0; i_145994 < (int64_t) 16; i_145994++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_132334;
            double r_132336 = 0.0;
            
            for (int64_t i_132335 = 0; i_132335 < (int64_t) 16; i_132335++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_132337 = ((double *) mem_147655)[i_145994 * (int64_t) 16 + i_132335];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_132338 = ((double *) mem_147366)[i_145994 * (int64_t) 16 + i_132335];
                
                // futhark/microgpt.fut:331:69-113
                
                double zt_res_132339 = zt_lhs_132337 * zt_rhs_132338;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_132340 = r_132336 + zt_res_132339;
                double r_tmp_149363 = zp_res_132340;
                
                r_132336 = r_tmp_149363;
            }
            defunc_0_lifted_lambda_res_132334 = r_132336;
            // futhark/microgpt.fut:331:131-143
            
            double zt_lhs_132341 = ((double *) mem_147444)[i_145994];
            
            // futhark/microgpt.fut:331:131-160
            
            double zt_res_132342 = zt_lhs_132341 * zt_lhs_132341;
            
            // futhark/microgpt.fut:331:122-160
            
            double zs_res_132343 = 1.0 / zt_res_132342;
            
            // futhark/microgpt.fut:331:47-160
            
            double zt_res_132344 = defunc_0_lifted_lambda_res_132334 * zs_res_132343;
            
            // futhark/microgpt.fut:331:39-160
            
            double neg_res_132345 = -zt_res_132344;
            
            ((double *) mem_147671)[i_145994] = neg_res_132345;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_145998 = 0; i_145998 < (int64_t) 16; i_145998++) {
            // futhark/microgpt.fut:332:39-51
            
            double zt_lhs_132353 = ((double *) mem_147671)[i_145998];
            
            // futhark/microgpt.fut:332:93-105
            
            double zp_lhs_132354 = ((double *) mem_147421)[i_145998];
            
            // futhark/microgpt.fut:332:93-133
            
            double zp_res_132355 = 1.0e-5 + zp_lhs_132354;
            
            // futhark/microgpt.fut:332:85-133
            
            double sqrt_res_132356 = futrts_sqrt64(zp_res_132355);
            
            // futhark/microgpt.fut:332:71-135
            
            double zt_res_132357 = 2.0 * sqrt_res_132356;
            
            // futhark/microgpt.fut:332:57-135
            
            double zs_res_132358 = 1.0 / zt_res_132357;
            
            // futhark/microgpt.fut:332:39-135
            
            double zt_res_132359 = zt_lhs_132353 * zs_res_132358;
            
            ((double *) mem_147678)[i_145998] = zt_res_132359;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146002 = 0; i_146002 < (int64_t) 16; i_146002++) {
            // futhark/microgpt.fut:333:49-61
            
            double zs_lhs_132367 = ((double *) mem_147678)[i_146002];
            
            // futhark/microgpt.fut:333:49-76
            
            double zs_res_132368 = zs_lhs_132367 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_149366 = 0; nest_i_149366 < (int64_t) 16; nest_i_149366++) {
                ((double *) mem_147685)[i_146002 * (int64_t) 16 + nest_i_149366] = zs_res_132368;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146010 = 0; i_146010 < (int64_t) 16; i_146010++) {
            // futhark/microgpt.fut:334:99-111
            
            double zs_rhs_132377 = ((double *) mem_147444)[i_146010];
            
            // futhark/microgpt.fut:334:91-111
            
            double zs_res_132378 = 1.0 / zs_rhs_132377;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_146006 = 0; i_146006 < (int64_t) 16; i_146006++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_132385 = ((double *) mem_147591)[i_146010 * (int64_t) 16 + i_146006];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_132386 = ((double *) mem_147655)[i_146010 * (int64_t) 16 + i_146006];
                
                // futhark/microgpt.fut:334:65-111
                
                double zt_res_132387 = zs_res_132378 * zt_lhs_132386;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_132388 = ((double *) mem_147685)[i_146010 * (int64_t) 16 + i_146006];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_132389 = ((double *) mem_147366)[i_146010 * (int64_t) 16 + i_146006];
                
                // futhark/microgpt.fut:334:119-163
                
                double zt_res_132390 = zt_lhs_132388 * zt_rhs_132389;
                
                // futhark/microgpt.fut:334:86-163
                
                double zp_res_132391 = zt_res_132387 + zt_res_132390;
                
                // futhark/microgpt.fut:334:114-215
                
                double zp_res_132392 = zt_res_132390 + zp_res_132391;
                
                // futhark/microgpt.fut:334:37-215
                
                double zp_res_132393 = zp_lhs_132385 + zp_res_132392;
                
                ((double *) mem_147700)[i_146006] = zp_res_132393;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147695, i_146010 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147700, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146023 = 0; i_146023 < (int64_t) 16; i_146023++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_146016 = 0; i_146016 < (int64_t) 16; i_146016++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_140605;
                double r_140607 = 0.0;
                
                for (int64_t i_140606 = 0; i_140606 < (int64_t) 16; i_140606++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_140608 = ((double *) mem_147695)[i_146023 * (int64_t) 16 + i_140606];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_140609 = ((double *) mem_param_146765.mem)[i_140606 * (int64_t) 16 + i_146016];
                    
                    // futhark/microgpt.fut:335:67-112
                    
                    double zt_res_140610 = zt_lhs_140608 * zt_rhs_140609;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_140611 = r_140607 + zt_res_140610;
                    double r_tmp_149373 = zp_res_140611;
                    
                    r_140607 = r_tmp_149373;
                }
                defunc_0_lifted_lambda_res_140605 = r_140607;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_140618;
                double r_140620 = 0.0;
                
                for (int64_t i_140619 = 0; i_140619 < (int64_t) 16; i_140619++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_140621 = ((double *) mem_147695)[i_140619 * (int64_t) 16 + i_146023];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_140622 = ((double *) mem_147328)[i_140619 * (int64_t) 16 + i_146016];
                    
                    // futhark/microgpt.fut:417:68-112
                    
                    double zt_res_140623 = zt_lhs_140621 * zt_rhs_140622;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_140624 = r_140620 + zt_res_140623;
                    double r_tmp_149374 = zp_res_140624;
                    
                    r_140620 = r_tmp_149374;
                }
                defunc_0_lifted_lambda_res_140618 = r_140620;
                ((double *) mem_147721)[i_146016] = defunc_0_lifted_lambda_res_140618;
                ((double *) mem_147722)[i_146016] = defunc_0_lifted_lambda_res_140605;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147711, i_146023 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147721, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147712, i_146023 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147722, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146061 = 0; i_146061 < (int64_t) 4; i_146061++) {
            // futhark/microgpt.fut:336:74-77
            
            int64_t zp_lhs_136059 = mul64((int64_t) 4, i_146061);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_146048 = 0; i_146048 < (int64_t) 16; i_146048++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_146028 = 0; i_146028 < (int64_t) 4; i_146028++) {
                    // futhark/microgpt.fut:336:79-87
                    
                    int64_t tmp_140771 = add64(zp_lhs_136059, i_146028);
                    
                    // futhark/microgpt.fut:336:52-89
                    
                    bool x_140772 = sle64((int64_t) 0, tmp_140771);
                    
                    // futhark/microgpt.fut:336:52-89
                    
                    bool y_140773 = slt64(tmp_140771, (int64_t) 16);
                    
                    // futhark/microgpt.fut:336:52-89
                    
                    bool bounds_check_140774 = x_140772 && y_140773;
                    
                    // futhark/microgpt.fut:336:52-89
                    
                    bool index_certs_140775;
                    
                    if (!bounds_check_140774) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_140771, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:336:52-89\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:336:13-90\n   #9  futhark/microgpt.fut:589:5-76\n   #10 futhark/microgpt.fut:606:26-612:31\n   #11 futhark/microgpt.fut:640:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_140776 = ((double *) mem_147712)[i_146048 * (int64_t) 16 + tmp_140771];
                    
                    ((double *) mem_147787)[i_146028] = lifted_lambda_res_140776;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_146036 = 0; i_146036 < (int64_t) 16; i_146036++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_140893 = ((double *) mem_147156)[i_146061 * (int64_t) 256 + i_146048 * (int64_t) 16 + i_146036];
                    
                    // futhark/microgpt.fut:338:59-101
                    
                    double zs_res_140894 = zs_lhs_140893 / 2.0;
                    double zp_rhs_140895 = ((double *) masks_mem_146751.mem)[step_131465 * (int64_t) 256 + i_146048 * (int64_t) 16 + i_146036];
                    
                    // futhark/microgpt.fut:338:88-127
                    
                    double zp_res_140896 = zs_res_140894 + zp_rhs_140895;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_140903 = ((double *) mem_147155)[i_146061 * (int64_t) 256 + i_146048 * (int64_t) 16 + i_146036];
                    
                    // futhark/microgpt.fut:345:59-101
                    
                    double zs_res_140904 = zs_lhs_140903 / 2.0;
                    
                    // futhark/microgpt.fut:345:88-127
                    
                    double zp_res_140906 = zp_rhs_140895 + zs_res_140904;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_140916 = ((double *) mem_147154)[i_146061 * (int64_t) 256 + i_146048 * (int64_t) 16 + i_146036];
                    
                    // futhark/microgpt.fut:362:59-101
                    
                    double zs_res_140917 = zs_lhs_140916 / 2.0;
                    
                    // futhark/microgpt.fut:362:88-127
                    
                    double zp_res_140919 = zp_rhs_140895 + zs_res_140917;
                    
                    ((double *) mem_147794)[i_146036] = zp_res_140919;
                    ((double *) mem_147795)[i_146036] = zp_res_140906;
                    ((double *) mem_147796)[i_146036] = zp_res_140896;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_147767, i_146048 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147794, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_147768, i_146048 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147795, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_147769, i_146048 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147796, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_147770, i_146048 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147787, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_147743, i_146061 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_147767, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_147744, i_146061 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_147768, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_147745, i_146061 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_147769, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_147746, i_146061 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_147770, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146124 = 0; i_146124 < (int64_t) 4; i_146124++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_146096 = 0; i_146096 < (int64_t) 16; i_146096++) {
                // futhark/microgpt.fut:115:13-33
                
                double defunc_0_reduce_res_145443;
                double defunc_0_reduce_res_145444;
                double defunc_0_reduce_res_145445;
                double defunc_0_reduce_res_145446;
                double defunc_0_reduce_res_145447;
                double redout_146068;
                double redout_146069;
                double redout_146070;
                double redout_146071;
                double redout_146072;
                
                redout_146068 = -INFINITY;
                redout_146069 = -INFINITY;
                redout_146070 = -INFINITY;
                redout_146071 = -INFINITY;
                redout_146072 = -INFINITY;
                for (int64_t i_146075 = 0; i_146075 < (int64_t) 16; i_146075++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_142622 = ((double *) mem_147745)[i_146124 * (int64_t) 256 + i_146096 * (int64_t) 16 + i_146075];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_142632 = ((double *) mem_147744)[i_146124 * (int64_t) 256 + i_146096 * (int64_t) 16 + i_146075];
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_142653;
                    double r_142655 = 0.0;
                    
                    for (int64_t i_142654 = 0; i_142654 < (int64_t) 4; i_142654++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_142656 = ((double *) mem_147746)[i_146124 * (int64_t) 64 + i_146096 * (int64_t) 4 + i_142654];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_142657 = ((double *) mem_147073)[i_146124 * (int64_t) 64 + i_146075 * (int64_t) 4 + i_142654];
                        
                        // futhark/microgpt.fut:347:79-139
                        
                        double zt_res_142658 = zt_lhs_142656 * zt_rhs_142657;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_142659 = r_142655 + zt_res_142658;
                        double r_tmp_149412 = zp_res_142659;
                        
                        r_142655 = r_tmp_149412;
                    }
                    defunc_0_lifted_lambda_res_142653 = r_142655;
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_142691 = ((double *) mem_147743)[i_146124 * (int64_t) 256 + i_146096 * (int64_t) 16 + i_146075];
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_142721;
                    double r_142723 = 0.0;
                    
                    for (int64_t i_142722 = 0; i_142722 < (int64_t) 4; i_142722++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_142724 = ((double *) mem_147746)[i_146124 * (int64_t) 64 + i_146096 * (int64_t) 4 + i_142722];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_142725 = ((double *) mem_147073)[i_146124 * (int64_t) 64 + i_146075 * (int64_t) 4 + i_142722];
                        
                        // futhark/microgpt.fut:364:79-139
                        
                        double zt_res_142726 = zt_lhs_142724 * zt_rhs_142725;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_142727 = r_142723 + zt_res_142726;
                        double r_tmp_149413 = zp_res_142727;
                        
                        r_142723 = r_tmp_149413;
                    }
                    defunc_0_lifted_lambda_res_142721 = r_142723;
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_141763 = fmax64(lifted_lambda_res_142622, redout_146068);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_141783 = fmax64(lifted_lambda_res_142632, redout_146069);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_141832 = fmax64(lifted_lambda_res_142632, redout_146070);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_141882 = fmax64(lifted_lambda_res_142691, redout_146071);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_141947 = fmax64(lifted_lambda_res_142691, redout_146072);
                    
                    ((double *) mem_147942)[i_146075] = defunc_0_lifted_lambda_res_142721;
                    ((double *) mem_147943)[i_146075] = defunc_0_lifted_lambda_res_142653;
                    
                    double redout_tmp_149405 = max_res_141763;
                    double redout_tmp_149406 = max_res_141783;
                    double redout_tmp_149407 = max_res_141832;
                    double redout_tmp_149408 = max_res_141882;
                    double redout_tmp_149409 = max_res_141947;
                    
                    redout_146068 = redout_tmp_149405;
                    redout_146069 = redout_tmp_149406;
                    redout_146070 = redout_tmp_149407;
                    redout_146071 = redout_tmp_149408;
                    redout_146072 = redout_tmp_149409;
                }
                defunc_0_reduce_res_145443 = redout_146068;
                defunc_0_reduce_res_145444 = redout_146069;
                defunc_0_reduce_res_145445 = redout_146070;
                defunc_0_reduce_res_145446 = redout_146071;
                defunc_0_reduce_res_145447 = redout_146072;
                // futhark/microgpt.fut:4:11-25
                for (int64_t nest_i_149414 = 0; nest_i_149414 < (int64_t) 16; nest_i_149414++) {
                    ((double *) mem_147909)[i_146096 * (int64_t) 16 + nest_i_149414] = defunc_0_reduce_res_145443;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t nest_i_149415 = 0; nest_i_149415 < (int64_t) 16; nest_i_149415++) {
                    ((double *) mem_147908)[i_146096 * (int64_t) 16 + nest_i_149415] = defunc_0_reduce_res_145444;
                }
                // futhark/microgpt.fut:357:148-174
                
                double neg_res_141840 = -defunc_0_reduce_res_145445;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_141841;
                double r_141843 = 0.0;
                
                for (int64_t i_141842 = 0; i_141842 < (int64_t) 16; i_141842++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_141844 = ((double *) mem_147744)[i_146124 * (int64_t) 256 + i_146096 * (int64_t) 16 + i_141842];
                    
                    // futhark/microgpt.fut:357:114-174
                    
                    double zp_res_141845 = neg_res_141840 + zp_lhs_141844;
                    
                    // futhark/microgpt.fut:357:107-174
                    
                    double neg_res_141846 = -zp_res_141845;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_141847 = fmax64(0.0, neg_res_141846);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_141848 = fsignum64(max_res_141847);
                    
                    // futhark/microgpt.fut:357:88-177
                    
                    double neg_res_141849 = -sgn_res_141848;
                    
                    // futhark/microgpt.fut:357:79-178
                    
                    double zp_res_141850 = 1.0 + neg_res_141849;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_141851 = r_141843 + zp_res_141850;
                    double r_tmp_149416 = zp_res_141851;
                    
                    r_141843 = r_tmp_149416;
                }
                defunc_0_lifted_lambda_res_141841 = r_141843;
                // futhark/microgpt.fut:357:48-181
                
                double zs_res_141852 = 1.0 / defunc_0_lifted_lambda_res_141841;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t nest_i_149417 = 0; nest_i_149417 < (int64_t) 16; nest_i_149417++) {
                    ((double *) mem_147904)[i_146096 * (int64_t) 16 + nest_i_149417] = defunc_0_reduce_res_145446;
                }
                // futhark/microgpt.fut:374:148-174
                
                double neg_res_141955 = -defunc_0_reduce_res_145447;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_141956;
                double r_141958 = 0.0;
                
                for (int64_t i_141957 = 0; i_141957 < (int64_t) 16; i_141957++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_141959 = ((double *) mem_147743)[i_146124 * (int64_t) 256 + i_146096 * (int64_t) 16 + i_141957];
                    
                    // futhark/microgpt.fut:374:114-174
                    
                    double zp_res_141960 = neg_res_141955 + zp_lhs_141959;
                    
                    // futhark/microgpt.fut:374:107-174
                    
                    double neg_res_141961 = -zp_res_141960;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_141962 = fmax64(0.0, neg_res_141961);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_141963 = fsignum64(max_res_141962);
                    
                    // futhark/microgpt.fut:374:88-177
                    
                    double neg_res_141964 = -sgn_res_141963;
                    
                    // futhark/microgpt.fut:374:79-178
                    
                    double zp_res_141965 = 1.0 + neg_res_141964;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_141966 = r_141958 + zp_res_141965;
                    double r_tmp_149418 = zp_res_141966;
                    
                    r_141958 = r_tmp_149418;
                }
                defunc_0_lifted_lambda_res_141956 = r_141958;
                // futhark/microgpt.fut:374:48-181
                
                double zs_res_141967 = 1.0 / defunc_0_lifted_lambda_res_141956;
                
                ((double *) mem_147901)[i_146096] = zs_res_141967;
                ((double *) mem_147902)[i_146096] = defunc_0_reduce_res_145447;
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_147903, i_146096 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147942, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                ((double *) mem_147905)[i_146096] = zs_res_141852;
                ((double *) mem_147906)[i_146096] = defunc_0_reduce_res_145445;
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_147907, i_146096 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147943, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147851, i_146124 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147901, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147852, i_146124 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147902, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_147853, i_146124 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_147903, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_147854, i_146124 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_147904, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147855, i_146124 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147905, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_147856, i_146124 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_147906, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_147857, i_146124 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_147907, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_147858, i_146124 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_147908, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_147859, i_146124 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_147909, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146164 = 0; i_146164 < (int64_t) 4; i_146164++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_146154 = 0; i_146154 < (int64_t) 16; i_146154++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_146140 = 0; i_146140 < (int64_t) 16; i_146140++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_142992 = ((double *) mem_147745)[i_146164 * (int64_t) 256 + i_146154 * (int64_t) 16 + i_146140];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double neg_arg0_142993 = ((double *) mem_147859)[i_146164 * (int64_t) 256 + i_146154 * (int64_t) 16 + i_146140];
                    
                    // futhark/microgpt.fut:340:133-167
                    
                    double neg_res_142994 = -neg_arg0_142993;
                    
                    // futhark/microgpt.fut:340:99-167
                    
                    double zp_res_142995 = zp_lhs_142992 + neg_res_142994;
                    
                    // futhark/microgpt.fut:340:92-167
                    
                    double exp_res_142996 = futrts_exp64(zp_res_142995);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_143003 = ((double *) mem_147744)[i_146164 * (int64_t) 256 + i_146154 * (int64_t) 16 + i_146140];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double neg_arg0_143004 = ((double *) mem_147858)[i_146164 * (int64_t) 256 + i_146154 * (int64_t) 16 + i_146140];
                    
                    // futhark/microgpt.fut:348:99-133
                    
                    double neg_res_143005 = -neg_arg0_143004;
                    
                    // futhark/microgpt.fut:348:65-133
                    
                    double zp_res_143006 = zp_lhs_143003 + neg_res_143005;
                    
                    // futhark/microgpt.fut:348:58-133
                    
                    double exp_res_143007 = futrts_exp64(zp_res_143006);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_143017 = ((double *) mem_147743)[i_146164 * (int64_t) 256 + i_146154 * (int64_t) 16 + i_146140];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double neg_arg0_143018 = ((double *) mem_147854)[i_146164 * (int64_t) 256 + i_146154 * (int64_t) 16 + i_146140];
                    
                    // futhark/microgpt.fut:365:99-133
                    
                    double neg_res_143019 = -neg_arg0_143018;
                    
                    // futhark/microgpt.fut:365:65-133
                    
                    double zp_res_143020 = zp_lhs_143017 + neg_res_143019;
                    
                    // futhark/microgpt.fut:365:58-133
                    
                    double exp_res_143021 = futrts_exp64(zp_res_143020);
                    
                    ((double *) mem_148065)[i_146140] = exp_res_143021;
                    ((double *) mem_148066)[i_146140] = exp_res_143007;
                    ((double *) mem_148067)[i_146140] = exp_res_142996;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_142881;
                double r_142883 = 0.0;
                
                for (int64_t i_142882 = 0; i_142882 < (int64_t) 16; i_142882++) {
                    // futhark/microgpt.fut:341:47-59
                    
                    double lifted_lambda_res_142884 = ((double *) mem_148067)[i_142882];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_142885 = r_142883 + lifted_lambda_res_142884;
                    double r_tmp_149428 = zp_res_142885;
                    
                    r_142883 = r_tmp_149428;
                }
                defunc_0_lifted_lambda_res_142881 = r_142883;
                // futhark/microgpt.fut:341:17-60
                
                double zs_res_142886 = 1.0 / defunc_0_lifted_lambda_res_142881;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_146146 = 0; i_146146 < (int64_t) 16; i_146146++) {
                    // futhark/microgpt.fut:342:5-17
                    
                    double zt_lhs_142893 = ((double *) mem_148067)[i_146146];
                    
                    // futhark/microgpt.fut:342:5-26
                    
                    double zt_res_142894 = zs_res_142886 * zt_lhs_142893;
                    
                    ((double *) mem_148086)[i_146146] = zt_res_142894;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_148050, i_146154 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148065, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_148051, i_146154 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148066, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_148052, i_146154 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148086, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_148032, i_146164 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148050, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_148033, i_146164 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148051, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_148034, i_146164 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148052, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146198 = 0; i_146198 < (int64_t) 4; i_146198++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_146182 = 0; i_146182 < (int64_t) 16; i_146182++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_146170 = 0; i_146170 < (int64_t) 4; i_146170++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_143274;
                    double r_143276 = 0.0;
                    
                    for (int64_t i_143275 = 0; i_143275 < (int64_t) 16; i_143275++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_143277 = ((double *) mem_147746)[i_146198 * (int64_t) 64 + i_143275 * (int64_t) 4 + i_146170];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_143278 = ((double *) mem_148034)[i_146198 * (int64_t) 256 + i_143275 * (int64_t) 16 + i_146182];
                        
                        // futhark/microgpt.fut:343:67-128
                        
                        double zt_res_143279 = zt_lhs_143277 * zt_rhs_143278;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_143280 = r_143276 + zt_res_143279;
                        double r_tmp_149441 = zp_res_143280;
                        
                        r_143276 = r_tmp_149441;
                    }
                    defunc_0_lifted_lambda_res_143274 = r_143276;
                    ((double *) mem_148167)[i_146170] = defunc_0_lifted_lambda_res_143274;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_143288;
                double r_143290 = 0.0;
                
                for (int64_t i_143289 = 0; i_143289 < (int64_t) 16; i_143289++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_143291 = ((double *) mem_148033)[i_146198 * (int64_t) 256 + i_146182 * (int64_t) 16 + i_143289];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_143292 = r_143290 + lifted_lambda_res_143291;
                    double r_tmp_149442 = zp_res_143292;
                    
                    r_143290 = r_tmp_149442;
                }
                defunc_0_lifted_lambda_res_143288 = r_143290;
                // futhark/microgpt.fut:349:48-107
                
                double zs_res_143293 = 1.0 / defunc_0_lifted_lambda_res_143288;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_143303;
                double r_143305 = 0.0;
                
                for (int64_t i_143304 = 0; i_143304 < (int64_t) 16; i_143304++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_143306 = ((double *) mem_147857)[i_146198 * (int64_t) 256 + i_146182 * (int64_t) 16 + i_143304];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_143307 = ((double *) mem_148033)[i_146198 * (int64_t) 256 + i_146182 * (int64_t) 16 + i_143304];
                    
                    // futhark/microgpt.fut:350:70-131
                    
                    double zt_res_143308 = zt_lhs_143306 * zt_rhs_143307;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_143309 = r_143305 + zt_res_143308;
                    double r_tmp_149443 = zp_res_143309;
                    
                    r_143305 = r_tmp_149443;
                }
                defunc_0_lifted_lambda_res_143303 = r_143305;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_143321;
                double r_143323 = 0.0;
                
                for (int64_t i_143322 = 0; i_143322 < (int64_t) 16; i_143322++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_143324 = ((double *) mem_148032)[i_146198 * (int64_t) 256 + i_146182 * (int64_t) 16 + i_143322];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_143325 = r_143323 + lifted_lambda_res_143324;
                    double r_tmp_149444 = zp_res_143325;
                    
                    r_143323 = r_tmp_149444;
                }
                defunc_0_lifted_lambda_res_143321 = r_143323;
                // futhark/microgpt.fut:366:48-107
                
                double zs_res_143326 = 1.0 / defunc_0_lifted_lambda_res_143321;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_143340;
                double r_143342 = 0.0;
                
                for (int64_t i_143341 = 0; i_143341 < (int64_t) 16; i_143341++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_143343 = ((double *) mem_147853)[i_146198 * (int64_t) 256 + i_146182 * (int64_t) 16 + i_143341];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_143344 = ((double *) mem_148032)[i_146198 * (int64_t) 256 + i_146182 * (int64_t) 16 + i_143341];
                    
                    // futhark/microgpt.fut:367:70-131
                    
                    double zt_res_143345 = zt_lhs_143343 * zt_rhs_143344;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_143346 = r_143342 + zt_res_143345;
                    double r_tmp_149445 = zp_res_143346;
                    
                    r_143342 = r_tmp_149445;
                }
                defunc_0_lifted_lambda_res_143340 = r_143342;
                ((double *) mem_148146)[i_146182] = defunc_0_lifted_lambda_res_143340;
                ((double *) mem_148147)[i_146182] = zs_res_143326;
                ((double *) mem_148148)[i_146182] = defunc_0_lifted_lambda_res_143303;
                ((double *) mem_148149)[i_146182] = zs_res_143293;
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_148150, i_146182 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148167, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148120, i_146198 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148146, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148121, i_146198 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148147, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148122, i_146198 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148148, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148123, i_146198 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148149, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_148124, i_146198 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_148150, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146242 = 0; i_146242 < (int64_t) 4; i_146242++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_146223 = 0; i_146223 < (int64_t) 16; i_146223++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_143758 = ((double *) mem_148123)[i_146242 * (int64_t) 16 + i_146223];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_143759 = ((double *) mem_148122)[i_146242 * (int64_t) 16 + i_146223];
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_143760;
                double r_143762 = 0.0;
                
                for (int64_t i_143761 = 0; i_143761 < (int64_t) 16; i_143761++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_143763 = ((double *) mem_148033)[i_146242 * (int64_t) 256 + i_146223 * (int64_t) 16 + i_143761];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_143764 = r_143762 + lifted_lambda_res_143763;
                    double r_tmp_149458 = zp_res_143764;
                    
                    r_143762 = r_tmp_149458;
                }
                defunc_0_lifted_lambda_res_143760 = r_143762;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_143765;
                double r_143767 = 0.0;
                
                for (int64_t i_143766 = 0; i_143766 < (int64_t) 16; i_143766++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_143768 = ((double *) mem_148033)[i_146242 * (int64_t) 256 + i_146223 * (int64_t) 16 + i_143766];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_143769 = r_143767 + lifted_lambda_res_143768;
                    double r_tmp_149459 = zp_res_143769;
                    
                    r_143767 = r_tmp_149459;
                }
                defunc_0_lifted_lambda_res_143765 = r_143767;
                // futhark/microgpt.fut:351:162-269
                
                double zt_res_143770 = defunc_0_lifted_lambda_res_143760 * defunc_0_lifted_lambda_res_143765;
                
                // futhark/microgpt.fut:351:152-269
                
                double zs_res_143771 = 1.0 / zt_res_143770;
                
                // futhark/microgpt.fut:351:126-269
                
                double zt_res_143772 = zt_lhs_143759 * zs_res_143771;
                
                // futhark/microgpt.fut:351:119-269
                
                double neg_res_143773 = -zt_res_143772;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_143819;
                double r_143821 = 0.0;
                
                for (int64_t i_143820 = 0; i_143820 < (int64_t) 16; i_143820++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_143822 = ((double *) mem_148032)[i_146242 * (int64_t) 256 + i_146223 * (int64_t) 16 + i_143820];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_143823 = r_143821 + lifted_lambda_res_143822;
                    double r_tmp_149460 = zp_res_143823;
                    
                    r_143821 = r_tmp_149460;
                }
                defunc_0_lifted_lambda_res_143819 = r_143821;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_143814;
                double r_143816 = 0.0;
                
                for (int64_t i_143815 = 0; i_143815 < (int64_t) 16; i_143815++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_143817 = ((double *) mem_148032)[i_146242 * (int64_t) 256 + i_146223 * (int64_t) 16 + i_143815];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_143818 = r_143816 + lifted_lambda_res_143817;
                    double r_tmp_149461 = zp_res_143818;
                    
                    r_143816 = r_tmp_149461;
                }
                defunc_0_lifted_lambda_res_143814 = r_143816;
                // futhark/microgpt.fut:368:162-269
                
                double zt_res_143824 = defunc_0_lifted_lambda_res_143814 * defunc_0_lifted_lambda_res_143819;
                
                // futhark/microgpt.fut:368:152-269
                
                double zs_res_143825 = 1.0 / zt_res_143824;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_143813 = ((double *) mem_148120)[i_146242 * (int64_t) 16 + i_146223];
                
                // futhark/microgpt.fut:368:126-269
                
                double zt_res_143826 = zt_lhs_143813 * zs_res_143825;
                
                // futhark/microgpt.fut:368:119-269
                
                double neg_res_143827 = -zt_res_143826;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_143812 = ((double *) mem_148121)[i_146242 * (int64_t) 16 + i_146223];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_146208 = 0; i_146208 < (int64_t) 16; i_146208++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_143889 = ((double *) mem_147857)[i_146242 * (int64_t) 256 + i_146223 * (int64_t) 16 + i_146208];
                    
                    // futhark/microgpt.fut:351:59-112
                    
                    double zt_res_143890 = zt_rhs_143758 * zt_lhs_143889;
                    
                    // futhark/microgpt.fut:351:88-273
                    
                    double zp_res_143891 = neg_res_143773 + zt_res_143890;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_143898 = ((double *) mem_147853)[i_146242 * (int64_t) 256 + i_146223 * (int64_t) 16 + i_146208];
                    
                    // futhark/microgpt.fut:368:59-112
                    
                    double zt_res_143899 = zt_rhs_143812 * zt_lhs_143898;
                    
                    // futhark/microgpt.fut:368:88-273
                    
                    double zp_res_143900 = neg_res_143827 + zt_res_143899;
                    
                    ((double *) mem_148269)[i_146208] = zp_res_143900;
                    ((double *) mem_148270)[i_146208] = zp_res_143891;
                }
                ((double *) mem_148243)[i_146223] = zt_lhs_143813;
                ((double *) mem_148244)[i_146223] = zt_rhs_143812;
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_148245, i_146223 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148269, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                ((double *) mem_148246)[i_146223] = zt_lhs_143759;
                ((double *) mem_148247)[i_146223] = zt_rhs_143758;
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_148248, i_146223 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148270, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148211, i_146242 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148243, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148212, i_146242 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148244, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_148213, i_146242 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148245, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148214, i_146242 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148246, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148215, i_146242 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148247, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_148216, i_146242 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148248, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146283 = 0; i_146283 < (int64_t) 4; i_146283++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_146270 = 0; i_146270 < (int64_t) 16; i_146270++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_144188;
                double r_144190 = 0.0;
                
                for (int64_t i_144189 = 0; i_144189 < (int64_t) 16; i_144189++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_144191 = ((double *) mem_148032)[i_146283 * (int64_t) 256 + i_146270 * (int64_t) 16 + i_144189];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_144192 = r_144190 + lifted_lambda_res_144191;
                    double r_tmp_149472 = zp_res_144192;
                    
                    r_144190 = r_tmp_149472;
                }
                defunc_0_lifted_lambda_res_144188 = r_144190;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_144183;
                double r_144185 = 0.0;
                
                for (int64_t i_144184 = 0; i_144184 < (int64_t) 16; i_144184++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_144186 = ((double *) mem_148032)[i_146283 * (int64_t) 256 + i_146270 * (int64_t) 16 + i_144184];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_144187 = r_144185 + lifted_lambda_res_144186;
                    double r_tmp_149473 = zp_res_144187;
                    
                    r_144185 = r_tmp_149473;
                }
                defunc_0_lifted_lambda_res_144183 = r_144185;
                // futhark/microgpt.fut:372:162-269
                
                double zt_res_144193 = defunc_0_lifted_lambda_res_144183 * defunc_0_lifted_lambda_res_144188;
                
                // futhark/microgpt.fut:372:152-269
                
                double zs_res_144194 = 1.0 / zt_res_144193;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_144182 = ((double *) mem_148211)[i_146283 * (int64_t) 16 + i_146270];
                
                // futhark/microgpt.fut:372:126-269
                
                double zt_res_144195 = zt_lhs_144182 * zs_res_144194;
                
                // futhark/microgpt.fut:372:119-269
                
                double neg_res_144196 = -zt_res_144195;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_144181 = ((double *) mem_148212)[i_146283 * (int64_t) 16 + i_146270];
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_144127;
                double r_144129 = 0.0;
                
                for (int64_t i_144128 = 0; i_144128 < (int64_t) 16; i_144128++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_144130 = ((double *) mem_148033)[i_146283 * (int64_t) 256 + i_146270 * (int64_t) 16 + i_144128];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_144131 = r_144129 + lifted_lambda_res_144130;
                    double r_tmp_149474 = zp_res_144131;
                    
                    r_144129 = r_tmp_149474;
                }
                defunc_0_lifted_lambda_res_144127 = r_144129;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_144122;
                double r_144124 = 0.0;
                
                for (int64_t i_144123 = 0; i_144123 < (int64_t) 16; i_144123++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_144125 = ((double *) mem_148033)[i_146283 * (int64_t) 256 + i_146270 * (int64_t) 16 + i_144123];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_144126 = r_144124 + lifted_lambda_res_144125;
                    double r_tmp_149475 = zp_res_144126;
                    
                    r_144124 = r_tmp_149475;
                }
                defunc_0_lifted_lambda_res_144122 = r_144124;
                // futhark/microgpt.fut:355:162-269
                
                double zt_res_144132 = defunc_0_lifted_lambda_res_144122 * defunc_0_lifted_lambda_res_144127;
                
                // futhark/microgpt.fut:355:152-269
                
                double zs_res_144133 = 1.0 / zt_res_144132;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_144121 = ((double *) mem_148214)[i_146283 * (int64_t) 16 + i_146270];
                
                // futhark/microgpt.fut:355:126-269
                
                double zt_res_144134 = zt_lhs_144121 * zs_res_144133;
                
                // futhark/microgpt.fut:355:119-269
                
                double neg_res_144135 = -zt_res_144134;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_144120 = ((double *) mem_148215)[i_146283 * (int64_t) 16 + i_146270];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_146257 = 0; i_146257 < (int64_t) 16; i_146257++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_144330 = ((double *) mem_147744)[i_146283 * (int64_t) 256 + i_146270 * (int64_t) 16 + i_146257];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double neg_arg0_144331 = ((double *) mem_147858)[i_146283 * (int64_t) 256 + i_146270 * (int64_t) 16 + i_146257];
                    
                    // futhark/microgpt.fut:352:107-141
                    
                    double neg_res_144332 = -neg_arg0_144331;
                    
                    // futhark/microgpt.fut:352:73-141
                    
                    double zp_res_144333 = zp_lhs_144330 + neg_res_144332;
                    
                    // futhark/microgpt.fut:352:66-141
                    
                    double exp_res_144334 = futrts_exp64(zp_res_144333);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_144335 = ((double *) mem_148216)[i_146283 * (int64_t) 256 + i_146270 * (int64_t) 16 + i_146257];
                    
                    // futhark/microgpt.fut:352:66-177
                    
                    double zt_res_144336 = exp_res_144334 * zt_rhs_144335;
                    
                    // futhark/microgpt.fut:352:58-177
                    
                    double neg_res_144337 = -zt_res_144336;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_144344 = ((double *) mem_147857)[i_146283 * (int64_t) 256 + i_146270 * (int64_t) 16 + i_146257];
                    
                    // futhark/microgpt.fut:355:59-112
                    
                    double zt_res_144345 = zt_rhs_144120 * zt_lhs_144344;
                    
                    // futhark/microgpt.fut:355:88-273
                    
                    double zp_res_144346 = neg_res_144135 + zt_res_144345;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_144356 = ((double *) mem_147743)[i_146283 * (int64_t) 256 + i_146270 * (int64_t) 16 + i_146257];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double neg_arg0_144357 = ((double *) mem_147854)[i_146283 * (int64_t) 256 + i_146270 * (int64_t) 16 + i_146257];
                    
                    // futhark/microgpt.fut:369:107-141
                    
                    double neg_res_144358 = -neg_arg0_144357;
                    
                    // futhark/microgpt.fut:369:73-141
                    
                    double zp_res_144359 = zp_lhs_144356 + neg_res_144358;
                    
                    // futhark/microgpt.fut:369:66-141
                    
                    double exp_res_144360 = futrts_exp64(zp_res_144359);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_144361 = ((double *) mem_148213)[i_146283 * (int64_t) 256 + i_146270 * (int64_t) 16 + i_146257];
                    
                    // futhark/microgpt.fut:369:66-177
                    
                    double zt_res_144362 = exp_res_144360 * zt_rhs_144361;
                    
                    // futhark/microgpt.fut:369:58-177
                    
                    double neg_res_144363 = -zt_res_144362;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_144375 = ((double *) mem_147853)[i_146283 * (int64_t) 256 + i_146270 * (int64_t) 16 + i_146257];
                    
                    // futhark/microgpt.fut:372:59-112
                    
                    double zt_res_144376 = zt_rhs_144181 * zt_lhs_144375;
                    
                    // futhark/microgpt.fut:372:88-273
                    
                    double zp_res_144377 = neg_res_144196 + zt_res_144376;
                    
                    ((double *) mem_148373)[i_146257] = zp_res_144377;
                    ((double *) mem_148374)[i_146257] = neg_res_144363;
                    ((double *) mem_148375)[i_146257] = zp_res_144346;
                    ((double *) mem_148376)[i_146257] = neg_res_144337;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_148353, i_146270 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148373, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_148354, i_146270 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148374, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_148355, i_146270 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148375, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_148356, i_146270 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148376, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_148329, i_146283 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148353, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_148330, i_146283 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148354, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_148331, i_146283 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148355, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_148332, i_146283 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148356, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146306 = 0; i_146306 < (int64_t) 4; i_146306++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_146299 = 0; i_146299 < (int64_t) 16; i_146299++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_144409;
                double r_144411 = 0.0;
                
                for (int64_t i_144410 = 0; i_144410 < (int64_t) 16; i_144410++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_144412 = ((double *) mem_148332)[i_146306 * (int64_t) 256 + i_146299 * (int64_t) 16 + i_144410];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_144413 = r_144411 + lifted_lambda_res_144412;
                    double r_tmp_149484 = zp_res_144413;
                    
                    r_144411 = r_tmp_149484;
                }
                defunc_0_lifted_lambda_res_144409 = r_144411;
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_144414 = ((double *) mem_147856)[i_146306 * (int64_t) 16 + i_146299];
                
                // futhark/microgpt.fut:358:306-332
                
                double neg_res_144415 = -neg_arg0_144414;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_144416 = ((double *) mem_147855)[i_146306 * (int64_t) 16 + i_146299];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_144453 = ((double *) mem_147851)[i_146306 * (int64_t) 16 + i_146299];
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_144451 = ((double *) mem_147852)[i_146306 * (int64_t) 16 + i_146299];
                
                // futhark/microgpt.fut:375:306-332
                
                double neg_res_144452 = -neg_arg0_144451;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_144446;
                double r_144448 = 0.0;
                
                for (int64_t i_144447 = 0; i_144447 < (int64_t) 16; i_144447++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_144449 = ((double *) mem_148330)[i_146306 * (int64_t) 256 + i_146299 * (int64_t) 16 + i_144447];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_144450 = r_144448 + lifted_lambda_res_144449;
                    double r_tmp_149485 = zp_res_144450;
                    
                    r_144448 = r_tmp_149485;
                }
                defunc_0_lifted_lambda_res_144446 = r_144448;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_146292 = 0; i_146292 < (int64_t) 16; i_146292++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_144494 = ((double *) mem_147744)[i_146306 * (int64_t) 256 + i_146299 * (int64_t) 16 + i_146292];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double neg_arg0_144495 = ((double *) mem_147858)[i_146306 * (int64_t) 256 + i_146299 * (int64_t) 16 + i_146292];
                    
                    // futhark/microgpt.fut:358:101-135
                    
                    double neg_res_144496 = -neg_arg0_144495;
                    
                    // futhark/microgpt.fut:358:67-135
                    
                    double zp_res_144497 = zp_lhs_144494 + neg_res_144496;
                    
                    // futhark/microgpt.fut:358:60-135
                    
                    double exp_res_144498 = futrts_exp64(zp_res_144497);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_144499 = ((double *) mem_148331)[i_146306 * (int64_t) 256 + i_146299 * (int64_t) 16 + i_146292];
                    
                    // futhark/microgpt.fut:358:60-171
                    
                    double zt_res_144500 = exp_res_144498 * zt_rhs_144499;
                    
                    // futhark/microgpt.fut:358:272-332
                    
                    double zp_res_144501 = neg_res_144415 + zp_lhs_144494;
                    
                    // futhark/microgpt.fut:358:265-332
                    
                    double neg_res_144502 = -zp_res_144501;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_144503 = fmax64(0.0, neg_res_144502);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_144504 = fsignum64(max_res_144503);
                    
                    // futhark/microgpt.fut:358:246-335
                    
                    double neg_res_144505 = -sgn_res_144504;
                    
                    // futhark/microgpt.fut:358:237-336
                    
                    double zp_res_144506 = 1.0 + neg_res_144505;
                    
                    // futhark/microgpt.fut:358:180-336
                    
                    double zt_res_144507 = defunc_0_lifted_lambda_res_144409 * zp_res_144506;
                    
                    // futhark/microgpt.fut:358:232-364
                    
                    double zt_res_144508 = zt_rhs_144416 * zt_res_144507;
                    
                    // futhark/microgpt.fut:358:139-364
                    
                    double zp_res_144509 = zt_res_144500 + zt_res_144508;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_144516 = ((double *) mem_147743)[i_146306 * (int64_t) 256 + i_146299 * (int64_t) 16 + i_146292];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double neg_arg0_144517 = ((double *) mem_147854)[i_146306 * (int64_t) 256 + i_146299 * (int64_t) 16 + i_146292];
                    
                    // futhark/microgpt.fut:375:101-135
                    
                    double neg_res_144518 = -neg_arg0_144517;
                    
                    // futhark/microgpt.fut:375:67-135
                    
                    double zp_res_144519 = zp_lhs_144516 + neg_res_144518;
                    
                    // futhark/microgpt.fut:375:60-135
                    
                    double exp_res_144520 = futrts_exp64(zp_res_144519);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_144521 = ((double *) mem_148329)[i_146306 * (int64_t) 256 + i_146299 * (int64_t) 16 + i_146292];
                    
                    // futhark/microgpt.fut:375:60-171
                    
                    double zt_res_144522 = exp_res_144520 * zt_rhs_144521;
                    
                    // futhark/microgpt.fut:375:272-332
                    
                    double zp_res_144523 = neg_res_144452 + zp_lhs_144516;
                    
                    // futhark/microgpt.fut:375:265-332
                    
                    double neg_res_144524 = -zp_res_144523;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_144525 = fmax64(0.0, neg_res_144524);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_144526 = fsignum64(max_res_144525);
                    
                    // futhark/microgpt.fut:375:246-335
                    
                    double neg_res_144527 = -sgn_res_144526;
                    
                    // futhark/microgpt.fut:375:237-336
                    
                    double zp_res_144528 = 1.0 + neg_res_144527;
                    
                    // futhark/microgpt.fut:375:180-336
                    
                    double zt_res_144529 = defunc_0_lifted_lambda_res_144446 * zp_res_144528;
                    
                    // futhark/microgpt.fut:375:232-364
                    
                    double zt_res_144530 = zt_rhs_144453 * zt_res_144529;
                    
                    // futhark/microgpt.fut:375:139-364
                    
                    double zp_res_144531 = zt_res_144522 + zt_res_144530;
                    
                    ((double *) mem_148459)[i_146292] = zp_res_144531;
                    ((double *) mem_148460)[i_146292] = zp_res_144509;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_148449, i_146299 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148459, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_148450, i_146299 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148460, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_148437, i_146306 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148449, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_148438, i_146306 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148450, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146327 = 0; i_146327 < (int64_t) 4; i_146327++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_146320 = 0; i_146320 < (int64_t) 16; i_146320++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_146313 = 0; i_146313 < (int64_t) 16; i_146313++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_144596 = ((double *) mem_148438)[i_146327 * (int64_t) 256 + i_146320 * (int64_t) 16 + i_146313];
                    
                    // futhark/microgpt.fut:359:58-100
                    
                    double zs_res_144597 = zs_lhs_144596 / 2.0;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_144604 = ((double *) mem_148437)[i_146327 * (int64_t) 256 + i_146320 * (int64_t) 16 + i_146313];
                    
                    // futhark/microgpt.fut:376:58-100
                    
                    double zs_res_144605 = zs_lhs_144604 / 2.0;
                    
                    ((double *) mem_148513)[i_146313] = zs_res_144605;
                    ((double *) mem_148514)[i_146313] = zs_res_144597;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_148503, i_146320 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148513, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_148504, i_146320 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148514, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_148491, i_146327 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148503, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_148492, i_146327 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148504, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146348 = 0; i_146348 < (int64_t) 4; i_146348++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_146341 = 0; i_146341 < (int64_t) 16; i_146341++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_146334 = 0; i_146334 < (int64_t) 4; i_146334++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_144680;
                    double r_144682 = 0.0;
                    
                    for (int64_t i_144681 = 0; i_144681 < (int64_t) 16; i_144681++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_144683 = ((double *) mem_148492)[i_146348 * (int64_t) 256 + i_144681 * (int64_t) 16 + i_146341];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_144684 = ((double *) mem_147075)[i_146348 * (int64_t) 64 + i_144681 * (int64_t) 4 + i_146334];
                        
                        // futhark/microgpt.fut:360:67-127
                        
                        double zt_res_144685 = zt_lhs_144683 * zt_rhs_144684;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_144686 = r_144682 + zt_res_144685;
                        double r_tmp_149500 = zp_res_144686;
                        
                        r_144682 = r_tmp_149500;
                    }
                    defunc_0_lifted_lambda_res_144680 = r_144682;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_144693;
                    double r_144695 = 0.0;
                    
                    for (int64_t i_144694 = 0; i_144694 < (int64_t) 16; i_144694++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_144696 = ((double *) mem_148491)[i_146348 * (int64_t) 256 + i_146341 * (int64_t) 16 + i_144694];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_144697 = ((double *) mem_147074)[i_146348 * (int64_t) 64 + i_144694 * (int64_t) 4 + i_146334];
                        
                        // futhark/microgpt.fut:377:67-127
                        
                        double zt_res_144698 = zt_lhs_144696 * zt_rhs_144697;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_144699 = r_144695 + zt_res_144698;
                        double r_tmp_149501 = zp_res_144699;
                        
                        r_144695 = r_tmp_149501;
                    }
                    defunc_0_lifted_lambda_res_144693 = r_144695;
                    ((double *) mem_148567)[i_146334] = defunc_0_lifted_lambda_res_144693;
                    ((double *) mem_148568)[i_146334] = defunc_0_lifted_lambda_res_144680;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_148557, i_146341 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148567, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_148558, i_146341 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148568, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_148545, i_146348 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_148557, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_148546, i_146348 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_148558, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146367 = 0; i_146367 < (int64_t) 16; i_146367++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_146357 = 0; i_146357 < (int64_t) 16; i_146357++) {
                // futhark/microgpt.fut:378:57-60
                
                int64_t tmp_144762 = sdiv64(i_146357, (int64_t) 4);
                
                // futhark/microgpt.fut:378:44-62
                
                bool x_144763 = sle64((int64_t) 0, tmp_144762);
                
                // futhark/microgpt.fut:378:44-62
                
                bool y_144764 = slt64(tmp_144762, (int64_t) 4);
                
                // futhark/microgpt.fut:378:44-62
                
                bool bounds_check_144765 = x_144763 && y_144764;
                
                // futhark/microgpt.fut:378:44-62
                
                bool index_certs_144766;
                
                if (!bounds_check_144765) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_144762, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:378:44-62\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:378:13-85\n   #6  futhark/microgpt.fut:589:5-76\n   #7  futhark/microgpt.fut:606:26-612:31\n   #8  futhark/microgpt.fut:640:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:378:79-82
                
                int64_t tmp_144767 = smod64(i_146357, (int64_t) 4);
                
                // futhark/microgpt.fut:378:44-84
                
                bool x_144768 = sle64((int64_t) 0, tmp_144767);
                
                // futhark/microgpt.fut:378:44-84
                
                bool y_144769 = slt64(tmp_144767, (int64_t) 4);
                
                // futhark/microgpt.fut:378:44-84
                
                bool bounds_check_144770 = x_144768 && y_144769;
                
                // futhark/microgpt.fut:378:44-84
                
                bool index_certs_144771;
                
                if (!bounds_check_144770) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_144767, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:378:44-84\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:378:13-85\n   #6  futhark/microgpt.fut:589:5-76\n   #7  futhark/microgpt.fut:606:26-612:31\n   #8  futhark/microgpt.fut:640:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_144772 = ((double *) mem_148124)[tmp_144762 * (int64_t) 64 + i_146367 * (int64_t) 4 + tmp_144767];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_144785 = ((double *) mem_148546)[tmp_144762 * (int64_t) 64 + i_146367 * (int64_t) 4 + tmp_144767];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_144801 = ((double *) mem_148545)[tmp_144762 * (int64_t) 64 + i_146367 * (int64_t) 4 + tmp_144767];
                
                ((double *) mem_148614)[i_146357] = lifted_lambda_res_144801;
                ((double *) mem_148615)[i_146357] = lifted_lambda_res_144785;
                ((double *) mem_148616)[i_146357] = lifted_lambda_res_144772;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148599, i_146367 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148614, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148600, i_146367 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148615, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148601, i_146367 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148616, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146392 = 0; i_146392 < (int64_t) 16; i_146392++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_146379 = 0; i_146379 < (int64_t) 16; i_146379++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_144964;
                double r_144966 = 0.0;
                
                for (int64_t i_144965 = 0; i_144965 < (int64_t) 16; i_144965++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_144967 = ((double *) mem_148601)[i_146392 * (int64_t) 16 + i_144965];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_144968 = ((double *) mem_param_146785.mem)[i_144965 * (int64_t) 16 + i_146379];
                    
                    // futhark/microgpt.fut:381:69-114
                    
                    double zt_res_144969 = zt_lhs_144967 * zt_rhs_144968;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_144970 = r_144966 + zt_res_144969;
                    double r_tmp_149516 = zp_res_144970;
                    
                    r_144966 = r_tmp_149516;
                }
                defunc_0_lifted_lambda_res_144964 = r_144966;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_144971;
                double r_144973 = 0.0;
                
                for (int64_t i_144972 = 0; i_144972 < (int64_t) 16; i_144972++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_144974 = ((double *) mem_148600)[i_146392 * (int64_t) 16 + i_144972];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_144975 = ((double *) mem_param_146761.mem)[i_144972 * (int64_t) 16 + i_146379];
                    
                    // futhark/microgpt.fut:381:145-190
                    
                    double zt_res_144976 = zt_lhs_144974 * zt_rhs_144975;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_144977 = r_144973 + zt_res_144976;
                    double r_tmp_149517 = zp_res_144977;
                    
                    r_144973 = r_tmp_149517;
                }
                defunc_0_lifted_lambda_res_144971 = r_144973;
                // futhark/microgpt.fut:381:47-192
                
                double zp_res_144978 = defunc_0_lifted_lambda_res_144964 + defunc_0_lifted_lambda_res_144971;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_144979;
                double r_144981 = 0.0;
                
                for (int64_t i_144980 = 0; i_144980 < (int64_t) 16; i_144980++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_144982 = ((double *) mem_148599)[i_146392 * (int64_t) 16 + i_144980];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_144983 = ((double *) mem_param_146773.mem)[i_144980 * (int64_t) 16 + i_146379];
                    
                    // futhark/microgpt.fut:381:222-267
                    
                    double zt_res_144984 = zt_lhs_144982 * zt_rhs_144983;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_144985 = r_144981 + zt_res_144984;
                    double r_tmp_149518 = zp_res_144985;
                    
                    r_144981 = r_tmp_149518;
                }
                defunc_0_lifted_lambda_res_144979 = r_144981;
                // futhark/microgpt.fut:381:118-269
                
                double zp_res_144986 = zp_res_144978 + defunc_0_lifted_lambda_res_144979;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_144993;
                double r_144995 = 0.0;
                
                for (int64_t i_144994 = 0; i_144994 < (int64_t) 16; i_144994++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_144996 = ((double *) mem_148599)[i_144994 * (int64_t) 16 + i_146392];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_144997 = ((double *) mem_146967)[i_144994 * (int64_t) 16 + i_146379];
                    
                    // futhark/microgpt.fut:414:68-111
                    
                    double zt_res_144998 = zt_lhs_144996 * zt_rhs_144997;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_144999 = r_144995 + zt_res_144998;
                    double r_tmp_149519 = zp_res_144999;
                    
                    r_144995 = r_tmp_149519;
                }
                defunc_0_lifted_lambda_res_144993 = r_144995;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_145009;
                double r_145011 = 0.0;
                
                for (int64_t i_145010 = 0; i_145010 < (int64_t) 16; i_145010++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_145012 = ((double *) mem_148600)[i_145010 * (int64_t) 16 + i_146392];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_145013 = ((double *) mem_146967)[i_145010 * (int64_t) 16 + i_146379];
                    
                    // futhark/microgpt.fut:415:68-111
                    
                    double zt_res_145014 = zt_lhs_145012 * zt_rhs_145013;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_145015 = r_145011 + zt_res_145014;
                    double r_tmp_149520 = zp_res_145015;
                    
                    r_145011 = r_tmp_149520;
                }
                defunc_0_lifted_lambda_res_145009 = r_145011;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_145027;
                double r_145029 = 0.0;
                
                for (int64_t i_145028 = 0; i_145028 < (int64_t) 16; i_145028++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_145030 = ((double *) mem_148601)[i_145028 * (int64_t) 16 + i_146392];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_145031 = ((double *) mem_146967)[i_145028 * (int64_t) 16 + i_146379];
                    
                    // futhark/microgpt.fut:416:68-111
                    
                    double zt_res_145032 = zt_lhs_145030 * zt_rhs_145031;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_145033 = r_145029 + zt_res_145032;
                    double r_tmp_149521 = zp_res_145033;
                    
                    r_145029 = r_tmp_149521;
                }
                defunc_0_lifted_lambda_res_145027 = r_145029;
                ((double *) mem_148667)[i_146379] = defunc_0_lifted_lambda_res_145027;
                ((double *) mem_148668)[i_146379] = defunc_0_lifted_lambda_res_145009;
                ((double *) mem_148669)[i_146379] = defunc_0_lifted_lambda_res_144993;
                ((double *) mem_148670)[i_146379] = zp_res_144986;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148647, i_146392 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148667, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148648, i_146392 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148668, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148649, i_146392 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148669, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148650, i_146392 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148670, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146399 = 0; i_146399 < (int64_t) 16; i_146399++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_133666;
            double r_133668 = 0.0;
            
            for (int64_t i_133667 = 0; i_133667 < (int64_t) 16; i_133667++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_133669 = ((double *) mem_148650)[i_146399 * (int64_t) 16 + i_133667];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_133670 = ((double *) mem_146927)[i_146399 * (int64_t) 16 + i_133667];
                
                // futhark/microgpt.fut:385:69-112
                
                double zt_res_133671 = zt_lhs_133669 * zt_rhs_133670;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_133672 = r_133668 + zt_res_133671;
                double r_tmp_149523 = zp_res_133672;
                
                r_133668 = r_tmp_149523;
            }
            defunc_0_lifted_lambda_res_133666 = r_133668;
            // futhark/microgpt.fut:385:130-142
            
            double zt_lhs_133673 = ((double *) mem_147327)[i_146399];
            
            // futhark/microgpt.fut:385:130-159
            
            double zt_res_133674 = zt_lhs_133673 * zt_lhs_133673;
            
            // futhark/microgpt.fut:385:121-159
            
            double zs_res_133675 = 1.0 / zt_res_133674;
            
            // futhark/microgpt.fut:385:47-159
            
            double zt_res_133676 = defunc_0_lifted_lambda_res_133666 * zs_res_133675;
            
            // futhark/microgpt.fut:385:39-159
            
            double neg_res_133677 = -zt_res_133676;
            
            ((double *) mem_148711)[i_146399] = neg_res_133677;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146403 = 0; i_146403 < (int64_t) 16; i_146403++) {
            // futhark/microgpt.fut:386:39-51
            
            double zt_lhs_133685 = ((double *) mem_148711)[i_146403];
            
            // futhark/microgpt.fut:386:93-105
            
            double zp_lhs_133686 = ((double *) mem_147012)[i_146403];
            
            // futhark/microgpt.fut:386:93-133
            
            double zp_res_133687 = 1.0e-5 + zp_lhs_133686;
            
            // futhark/microgpt.fut:386:85-133
            
            double sqrt_res_133688 = futrts_sqrt64(zp_res_133687);
            
            // futhark/microgpt.fut:386:71-135
            
            double zt_res_133689 = 2.0 * sqrt_res_133688;
            
            // futhark/microgpt.fut:386:57-135
            
            double zs_res_133690 = 1.0 / zt_res_133689;
            
            // futhark/microgpt.fut:386:39-135
            
            double zt_res_133691 = zt_lhs_133685 * zs_res_133690;
            
            ((double *) mem_148718)[i_146403] = zt_res_133691;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146407 = 0; i_146407 < (int64_t) 16; i_146407++) {
            // futhark/microgpt.fut:387:49-61
            
            double zs_lhs_133699 = ((double *) mem_148718)[i_146407];
            
            // futhark/microgpt.fut:387:49-76
            
            double zs_res_133700 = zs_lhs_133699 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_149526 = 0; nest_i_149526 < (int64_t) 16; nest_i_149526++) {
                ((double *) mem_148725)[i_146407 * (int64_t) 16 + nest_i_149526] = zs_res_133700;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146415 = 0; i_146415 < (int64_t) 16; i_146415++) {
            // futhark/microgpt.fut:388:99-111
            
            double zs_rhs_133709 = ((double *) mem_147327)[i_146415];
            
            // futhark/microgpt.fut:388:91-111
            
            double zs_res_133710 = 1.0 / zs_rhs_133709;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_146411 = 0; i_146411 < (int64_t) 16; i_146411++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_133717 = ((double *) mem_147695)[i_146415 * (int64_t) 16 + i_146411];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_133718 = ((double *) mem_148650)[i_146415 * (int64_t) 16 + i_146411];
                
                // futhark/microgpt.fut:388:65-111
                
                double zt_res_133719 = zs_res_133710 * zt_lhs_133718;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_133720 = ((double *) mem_148725)[i_146415 * (int64_t) 16 + i_146411];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_133721 = ((double *) mem_146927)[i_146415 * (int64_t) 16 + i_146411];
                
                // futhark/microgpt.fut:388:119-162
                
                double zt_res_133722 = zt_lhs_133720 * zt_rhs_133721;
                
                // futhark/microgpt.fut:388:86-162
                
                double zp_res_133723 = zt_res_133719 + zt_res_133722;
                
                // futhark/microgpt.fut:388:114-213
                
                double zp_res_133724 = zt_res_133722 + zp_res_133723;
                
                // futhark/microgpt.fut:388:37-213
                
                double zp_res_133725 = zp_lhs_133717 + zp_res_133724;
                
                ((double *) mem_148740)[i_146411] = zp_res_133725;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148735, i_146415 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148740, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146419 = 0; i_146419 < (int64_t) 16; i_146419++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_133773;
            double r_133775 = 0.0;
            
            for (int64_t i_133774 = 0; i_133774 < (int64_t) 16; i_133774++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_133776 = ((double *) mem_148735)[i_146419 * (int64_t) 16 + i_133774];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_133777 = ((double *) mem_146895)[i_146419 * (int64_t) 16 + i_133774];
                
                // futhark/microgpt.fut:392:69-112
                
                double zt_res_133778 = zt_lhs_133776 * zt_rhs_133777;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_133779 = r_133775 + zt_res_133778;
                double r_tmp_149530 = zp_res_133779;
                
                r_133775 = r_tmp_149530;
            }
            defunc_0_lifted_lambda_res_133773 = r_133775;
            // futhark/microgpt.fut:392:130-142
            
            double zt_lhs_133780 = ((double *) mem_147011)[i_146419];
            
            // futhark/microgpt.fut:392:130-159
            
            double zt_res_133781 = zt_lhs_133780 * zt_lhs_133780;
            
            // futhark/microgpt.fut:392:121-159
            
            double zs_res_133782 = 1.0 / zt_res_133781;
            
            // futhark/microgpt.fut:392:47-159
            
            double zt_res_133783 = defunc_0_lifted_lambda_res_133773 * zs_res_133782;
            
            // futhark/microgpt.fut:392:39-159
            
            double neg_res_133784 = -zt_res_133783;
            
            ((double *) mem_148751)[i_146419] = neg_res_133784;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146423 = 0; i_146423 < (int64_t) 16; i_146423++) {
            // futhark/microgpt.fut:393:39-51
            
            double zt_lhs_133792 = ((double *) mem_148751)[i_146423];
            
            // futhark/microgpt.fut:393:93-105
            
            double zp_lhs_133793 = ((double *) mem_146965)[i_146423];
            
            // futhark/microgpt.fut:393:93-133
            
            double zp_res_133794 = 1.0e-5 + zp_lhs_133793;
            
            // futhark/microgpt.fut:393:85-133
            
            double sqrt_res_133795 = futrts_sqrt64(zp_res_133794);
            
            // futhark/microgpt.fut:393:71-135
            
            double zt_res_133796 = 2.0 * sqrt_res_133795;
            
            // futhark/microgpt.fut:393:57-135
            
            double zs_res_133797 = 1.0 / zt_res_133796;
            
            // futhark/microgpt.fut:393:39-135
            
            double zt_res_133798 = zt_lhs_133792 * zs_res_133797;
            
            ((double *) mem_148758)[i_146423] = zt_res_133798;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146427 = 0; i_146427 < (int64_t) 16; i_146427++) {
            // futhark/microgpt.fut:394:49-61
            
            double zs_lhs_133806 = ((double *) mem_148758)[i_146427];
            
            // futhark/microgpt.fut:394:49-76
            
            double zs_res_133807 = zs_lhs_133806 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_149533 = 0; nest_i_149533 < (int64_t) 16; nest_i_149533++) {
                ((double *) mem_148765)[i_146427 * (int64_t) 16 + nest_i_149533] = zs_res_133807;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146435 = 0; i_146435 < (int64_t) 16; i_146435++) {
            // futhark/microgpt.fut:395:73-85
            
            double zs_rhs_133816 = ((double *) mem_147011)[i_146435];
            
            // futhark/microgpt.fut:395:65-85
            
            double zs_res_133817 = 1.0 / zs_rhs_133816;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_146431 = 0; i_146431 < (int64_t) 16; i_146431++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_133824 = ((double *) mem_148735)[i_146435 * (int64_t) 16 + i_146431];
                
                // futhark/microgpt.fut:395:39-85
                
                double zt_res_133825 = zs_res_133817 * zt_lhs_133824;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_133826 = ((double *) mem_148765)[i_146435 * (int64_t) 16 + i_146431];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_133827 = ((double *) mem_146895)[i_146435 * (int64_t) 16 + i_146431];
                
                // futhark/microgpt.fut:395:93-136
                
                double zt_res_133828 = zt_lhs_133826 * zt_rhs_133827;
                
                // futhark/microgpt.fut:395:60-136
                
                double zp_res_133829 = zt_res_133825 + zt_res_133828;
                
                // futhark/microgpt.fut:395:88-187
                
                double zp_res_133830 = zt_res_133828 + zp_res_133829;
                
                ((double *) mem_148780)[i_146431] = zp_res_133830;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148775, i_146435 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148780, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146448 = 0; i_146448 < (int64_t) 16; i_146448++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_146441 = 0; i_146441 < (int64_t) 16; i_146441++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_145059 = ((double *) mem_148775)[i_146448 * (int64_t) 16 + i_146441];
                
                ((double *) mem_148801)[i_146441] = lifted_lambda_res_145059;
                ((double *) mem_148802)[i_146441] = lifted_lambda_res_145059;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148791, i_146448 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148801, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148792, i_146448 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148802, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146457 = 0; i_146457 < (int64_t) 64; i_146457++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_146453 = 0; i_146453 < (int64_t) 16; i_146453++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_133944;
                double r_133946 = 0.0;
                
                for (int64_t i_133945 = 0; i_133945 < (int64_t) 16; i_133945++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_133947 = ((double *) mem_147639)[i_133945 * (int64_t) 64 + i_146457];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_133948 = ((double *) mem_147383)[i_133945 * (int64_t) 16 + i_146453];
                    
                    // futhark/microgpt.fut:418:67-111
                    
                    double zt_res_133949 = zt_lhs_133947 * zt_rhs_133948;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_133950 = r_133946 + zt_res_133949;
                    double r_tmp_149542 = zp_res_133950;
                    
                    r_133946 = r_tmp_149542;
                }
                defunc_0_lifted_lambda_res_133944 = r_133946;
                ((double *) mem_148828)[i_146453] = defunc_0_lifted_lambda_res_133944;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148823, i_146457 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148828, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_146470 = 0; i_146470 < (int64_t) 27; i_146470++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_146463 = 0; i_146463 < (int64_t) 16; i_146463++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_145087;
                double r_145089 = 0.0;
                
                for (int64_t i_145088 = 0; i_145088 < (int64_t) 16; i_145088++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_145090 = ((double *) mem_147575)[i_145088 * (int64_t) 27 + i_146470];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_145091 = ((double *) mem_147483)[i_145088 * (int64_t) 16 + i_146463];
                    
                    // futhark/microgpt.fut:420:68-111
                    
                    double zt_res_145092 = zt_lhs_145090 * zt_rhs_145091;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_145093 = r_145089 + zt_res_145092;
                    double r_tmp_149547 = zp_res_145093;
                    
                    r_145089 = r_tmp_149547;
                }
                defunc_0_lifted_lambda_res_145087 = r_145089;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_145096;
                double r_145098 = 0.0;
                
                for (int64_t i_145097 = 0; i_145097 < (int64_t) 16; i_145097++) {
                    int64_t zeze_lhs_145099 = ((int64_t *) seqs_mem_146753.mem)[step_131465 * (int64_t) 16 + i_145097];
                    
                    // futhark/microgpt.fut:590:58-109
                    
                    bool cond_145100 = zeze_lhs_145099 == i_146470;
                    
                    // futhark/microgpt.fut:590:58-109
                    
                    double lifted_lambda_res_145101;
                    
                    if (cond_145100) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double lifted_lambda_res_t_res_145516 = ((double *) mem_148791)[i_145097 * (int64_t) 16 + i_146463];
                        
                        lifted_lambda_res_145101 = lifted_lambda_res_t_res_145516;
                    } else {
                        lifted_lambda_res_145101 = 0.0;
                    }
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_145107 = r_145098 + lifted_lambda_res_145101;
                    double r_tmp_149548 = zp_res_145107;
                    
                    r_145098 = r_tmp_149548;
                }
                defunc_0_lifted_lambda_res_145096 = r_145098;
                ((double *) mem_148849)[i_146463] = defunc_0_lifted_lambda_res_145096;
                ((double *) mem_148850)[i_146463] = defunc_0_lifted_lambda_res_145087;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148839, i_146470 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148849, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_148840, i_146470 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_148850, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:66:26-45
        
        double i64_res_134028 = sitofp_i64_f64(step_131465);
        
        // futhark/microgpt.fut:525:46-66
        
        double zm_rhs_134029 = i64_res_134028 / 50.0;
        
        // futhark/microgpt.fut:525:24-66
        
        double zt_rhs_134030 = 1.0 - zm_rhs_134029;
        
        // futhark/microgpt.fut:525:19-66
        
        double lt_r_134031 = 1.0e-2 * zt_rhs_134030;
        
        // futhark/microgpt.fut:527:5-52
        if (memblock_alloc(ctx, &mem_148871, (int64_t) 3456, "mem_148871")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:527:5-52
        // futhark/microgpt.fut:527:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148871.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146777.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:527:5-52
        if (memblock_alloc(ctx, &mem_148873, (int64_t) 3456, "mem_148873")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:527:5-52
        // futhark/microgpt.fut:527:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148873.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146813.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:527:5-52
        if (memblock_alloc(ctx, &mem_148875, (int64_t) 3456, "mem_148875")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:527:5-52
        // futhark/microgpt.fut:527:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148875.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146849.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:527:5-52
        if (memblock_alloc(ctx, &mem_148877, (int64_t) 3456, "mem_148877")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:527:5-52
        // futhark/microgpt.fut:527:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148877.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148839, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:527:5-52
        if (futrts_adam_opt_w_12815(ctx, &ext_mem_148881, &ext_mem_148880, &ext_mem_148879, mem_148871, mem_148873, mem_148875, mem_148877, (int64_t) 27, (int64_t) 16, step_131465, lt_r_134031) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_148871, "mem_148871") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148873, "mem_148873") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148875, "mem_148875") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148877, "mem_148877") != 0)
            return 1;
        // futhark/microgpt.fut:529:5-52
        if (memblock_alloc(ctx, &mem_148882, (int64_t) 2048, "mem_148882")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:529:5-52
        // futhark/microgpt.fut:529:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148882.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146769.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:529:5-52
        if (memblock_alloc(ctx, &mem_148884, (int64_t) 2048, "mem_148884")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:529:5-52
        // futhark/microgpt.fut:529:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148884.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146805.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:529:5-52
        if (memblock_alloc(ctx, &mem_148886, (int64_t) 2048, "mem_148886")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:529:5-52
        // futhark/microgpt.fut:529:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148886.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146841.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:529:5-52
        if (memblock_alloc(ctx, &mem_148888, (int64_t) 2048, "mem_148888")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:529:5-52
        // futhark/microgpt.fut:529:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148888.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148792, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:529:5-52
        if (futrts_adam_opt_w_12816(ctx, &ext_mem_148892, &ext_mem_148891, &ext_mem_148890, mem_148882, mem_148884, mem_148886, mem_148888, (int64_t) 16, (int64_t) 16, step_131465, lt_r_134031) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_148882, "mem_148882") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148884, "mem_148884") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148886, "mem_148886") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148888, "mem_148888") != 0)
            return 1;
        // futhark/microgpt.fut:531:5-56
        if (memblock_alloc(ctx, &mem_148893, (int64_t) 2048, "mem_148893")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:531:5-56
        // futhark/microgpt.fut:531:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148893.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146773.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:531:5-56
        if (memblock_alloc(ctx, &mem_148895, (int64_t) 2048, "mem_148895")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:531:5-56
        // futhark/microgpt.fut:531:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148895.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146809.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:531:5-56
        if (memblock_alloc(ctx, &mem_148897, (int64_t) 2048, "mem_148897")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:531:5-56
        // futhark/microgpt.fut:531:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148897.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146845.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:531:5-56
        if (memblock_alloc(ctx, &mem_148899, (int64_t) 2048, "mem_148899")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:531:5-56
        // futhark/microgpt.fut:531:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148899.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148649, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:531:5-56
        if (futrts_adam_opt_w_12816(ctx, &ext_mem_148903, &ext_mem_148902, &ext_mem_148901, mem_148893, mem_148895, mem_148897, mem_148899, (int64_t) 16, (int64_t) 16, step_131465, lt_r_134031) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_148893, "mem_148893") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148895, "mem_148895") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148897, "mem_148897") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148899, "mem_148899") != 0)
            return 1;
        // futhark/microgpt.fut:533:5-56
        if (memblock_alloc(ctx, &mem_148904, (int64_t) 2048, "mem_148904")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:533:5-56
        // futhark/microgpt.fut:533:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148904.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146761.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:533:5-56
        if (memblock_alloc(ctx, &mem_148906, (int64_t) 2048, "mem_148906")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:533:5-56
        // futhark/microgpt.fut:533:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148906.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146797.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:533:5-56
        if (memblock_alloc(ctx, &mem_148908, (int64_t) 2048, "mem_148908")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:533:5-56
        // futhark/microgpt.fut:533:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148908.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146833.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:533:5-56
        if (memblock_alloc(ctx, &mem_148910, (int64_t) 2048, "mem_148910")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:533:5-56
        // futhark/microgpt.fut:533:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148910.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148648, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:533:5-56
        if (futrts_adam_opt_w_12816(ctx, &ext_mem_148914, &ext_mem_148913, &ext_mem_148912, mem_148904, mem_148906, mem_148908, mem_148910, (int64_t) 16, (int64_t) 16, step_131465, lt_r_134031) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_148904, "mem_148904") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148906, "mem_148906") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148908, "mem_148908") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148910, "mem_148910") != 0)
            return 1;
        // futhark/microgpt.fut:535:5-56
        if (memblock_alloc(ctx, &mem_148915, (int64_t) 2048, "mem_148915")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:535:5-56
        // futhark/microgpt.fut:535:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148915.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146785.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:535:5-56
        if (memblock_alloc(ctx, &mem_148917, (int64_t) 2048, "mem_148917")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:535:5-56
        // futhark/microgpt.fut:535:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148917.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146821.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:535:5-56
        if (memblock_alloc(ctx, &mem_148919, (int64_t) 2048, "mem_148919")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:535:5-56
        // futhark/microgpt.fut:535:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148919.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146857.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:535:5-56
        if (memblock_alloc(ctx, &mem_148921, (int64_t) 2048, "mem_148921")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:535:5-56
        // futhark/microgpt.fut:535:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148921.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148647, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:535:5-56
        if (futrts_adam_opt_w_12816(ctx, &ext_mem_148925, &ext_mem_148924, &ext_mem_148923, mem_148915, mem_148917, mem_148919, mem_148921, (int64_t) 16, (int64_t) 16, step_131465, lt_r_134031) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_148915, "mem_148915") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148917, "mem_148917") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148919, "mem_148919") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148921, "mem_148921") != 0)
            return 1;
        // futhark/microgpt.fut:537:5-56
        if (memblock_alloc(ctx, &mem_148926, (int64_t) 2048, "mem_148926")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:537:5-56
        // futhark/microgpt.fut:537:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148926.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146765.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:537:5-56
        if (memblock_alloc(ctx, &mem_148928, (int64_t) 2048, "mem_148928")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:537:5-56
        // futhark/microgpt.fut:537:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148928.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146801.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:537:5-56
        if (memblock_alloc(ctx, &mem_148930, (int64_t) 2048, "mem_148930")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:537:5-56
        // futhark/microgpt.fut:537:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148930.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146837.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:537:5-56
        if (memblock_alloc(ctx, &mem_148932, (int64_t) 2048, "mem_148932")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:537:5-56
        // futhark/microgpt.fut:537:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148932.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_147711, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:537:5-56
        if (futrts_adam_opt_w_12816(ctx, &ext_mem_148936, &ext_mem_148935, &ext_mem_148934, mem_148926, mem_148928, mem_148930, mem_148932, (int64_t) 16, (int64_t) 16, step_131465, lt_r_134031) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_148926, "mem_148926") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148928, "mem_148928") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148930, "mem_148930") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148932, "mem_148932") != 0)
            return 1;
        // futhark/microgpt.fut:539:5-52
        if (memblock_alloc(ctx, &mem_148937, (int64_t) 8192, "mem_148937")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:539:5-52
        // futhark/microgpt.fut:539:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148937.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146781.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:539:5-52
        if (memblock_alloc(ctx, &mem_148939, (int64_t) 8192, "mem_148939")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:539:5-52
        // futhark/microgpt.fut:539:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148939.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146817.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:539:5-52
        if (memblock_alloc(ctx, &mem_148941, (int64_t) 8192, "mem_148941")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:539:5-52
        // futhark/microgpt.fut:539:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148941.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146853.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:539:5-52
        if (memblock_alloc(ctx, &mem_148943, (int64_t) 8192, "mem_148943")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:539:5-52
        // futhark/microgpt.fut:539:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148943.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148823, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:539:5-52
        if (futrts_adam_opt_w_12815(ctx, &ext_mem_148947, &ext_mem_148946, &ext_mem_148945, mem_148937, mem_148939, mem_148941, mem_148943, (int64_t) 64, (int64_t) 16, step_131465, lt_r_134031) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_148937, "mem_148937") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148939, "mem_148939") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148941, "mem_148941") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148943, "mem_148943") != 0)
            return 1;
        // futhark/microgpt.fut:541:5-60
        if (memblock_alloc(ctx, &mem_148948, (int64_t) 8192, "mem_148948")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:541:5-60
        // futhark/microgpt.fut:541:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148948.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_146757.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:541:5-60
        if (memblock_alloc(ctx, &mem_148950, (int64_t) 8192, "mem_148950")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:541:5-60
        // futhark/microgpt.fut:541:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148950.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_146793.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:541:5-60
        if (memblock_alloc(ctx, &mem_148952, (int64_t) 8192, "mem_148952")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:541:5-60
        // futhark/microgpt.fut:541:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148952.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_146829.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:541:5-60
        if (memblock_alloc(ctx, &mem_148954, (int64_t) 8192, "mem_148954")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:541:5-60
        // futhark/microgpt.fut:541:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148954.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_147607, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:541:5-60
        if (futrts_adam_opt_w_12815(ctx, &ext_mem_148958, &ext_mem_148957, &ext_mem_148956, mem_148948, mem_148950, mem_148952, mem_148954, (int64_t) 16, (int64_t) 64, step_131465, lt_r_134031) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_148948, "mem_148948") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148950, "mem_148950") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148952, "mem_148952") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148954, "mem_148954") != 0)
            return 1;
        // futhark/microgpt.fut:543:5-56
        if (memblock_alloc(ctx, &mem_148959, (int64_t) 3456, "mem_148959")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:543:5-56
        // futhark/microgpt.fut:543:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148959.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146789.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:543:5-56
        if (memblock_alloc(ctx, &mem_148961, (int64_t) 3456, "mem_148961")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:543:5-56
        // futhark/microgpt.fut:543:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148961.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146825.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:543:5-56
        if (memblock_alloc(ctx, &mem_148963, (int64_t) 3456, "mem_148963")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:543:5-56
        // futhark/microgpt.fut:543:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148963.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_146861.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:543:5-56
        if (memblock_alloc(ctx, &mem_148965, (int64_t) 3456, "mem_148965")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:543:5-56
        // futhark/microgpt.fut:543:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_148965.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_148840, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:543:5-56
        if (futrts_adam_opt_w_12815(ctx, &ext_mem_148969, &ext_mem_148968, &ext_mem_148967, mem_148959, mem_148961, mem_148963, mem_148965, (int64_t) 27, (int64_t) 16, step_131465, lt_r_134031) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_148959, "mem_148959") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148961, "mem_148961") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148963, "mem_148963") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148965, "mem_148965") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149177, &ext_mem_148958, "ext_mem_148958") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149178, &ext_mem_148914, "ext_mem_148914") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149179, &ext_mem_148936, "ext_mem_148936") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149180, &ext_mem_148892, "ext_mem_148892") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149181, &ext_mem_148903, "ext_mem_148903") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149182, &ext_mem_148881, "ext_mem_148881") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149183, &ext_mem_148947, "ext_mem_148947") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149184, &ext_mem_148925, "ext_mem_148925") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149185, &ext_mem_148969, "ext_mem_148969") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149186, &ext_mem_148957, "ext_mem_148957") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149187, &ext_mem_148913, "ext_mem_148913") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149188, &ext_mem_148935, "ext_mem_148935") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149189, &ext_mem_148891, "ext_mem_148891") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149190, &ext_mem_148902, "ext_mem_148902") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149191, &ext_mem_148880, "ext_mem_148880") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149192, &ext_mem_148946, "ext_mem_148946") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149193, &ext_mem_148924, "ext_mem_148924") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149194, &ext_mem_148968, "ext_mem_148968") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149195, &ext_mem_148956, "ext_mem_148956") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149196, &ext_mem_148912, "ext_mem_148912") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149197, &ext_mem_148934, "ext_mem_148934") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149198, &ext_mem_148890, "ext_mem_148890") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149199, &ext_mem_148901, "ext_mem_148901") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149200, &ext_mem_148879, "ext_mem_148879") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149201, &ext_mem_148945, "ext_mem_148945") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149202, &ext_mem_148923, "ext_mem_148923") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_149203, &ext_mem_148967, "ext_mem_148967") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146757, &mem_param_tmp_149177, "mem_param_tmp_149177") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146761, &mem_param_tmp_149178, "mem_param_tmp_149178") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146765, &mem_param_tmp_149179, "mem_param_tmp_149179") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146769, &mem_param_tmp_149180, "mem_param_tmp_149180") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146773, &mem_param_tmp_149181, "mem_param_tmp_149181") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146777, &mem_param_tmp_149182, "mem_param_tmp_149182") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146781, &mem_param_tmp_149183, "mem_param_tmp_149183") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146785, &mem_param_tmp_149184, "mem_param_tmp_149184") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146789, &mem_param_tmp_149185, "mem_param_tmp_149185") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146793, &mem_param_tmp_149186, "mem_param_tmp_149186") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146797, &mem_param_tmp_149187, "mem_param_tmp_149187") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146801, &mem_param_tmp_149188, "mem_param_tmp_149188") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146805, &mem_param_tmp_149189, "mem_param_tmp_149189") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146809, &mem_param_tmp_149190, "mem_param_tmp_149190") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146813, &mem_param_tmp_149191, "mem_param_tmp_149191") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146817, &mem_param_tmp_149192, "mem_param_tmp_149192") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146821, &mem_param_tmp_149193, "mem_param_tmp_149193") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146825, &mem_param_tmp_149194, "mem_param_tmp_149194") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146829, &mem_param_tmp_149195, "mem_param_tmp_149195") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146833, &mem_param_tmp_149196, "mem_param_tmp_149196") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146837, &mem_param_tmp_149197, "mem_param_tmp_149197") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146841, &mem_param_tmp_149198, "mem_param_tmp_149198") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146845, &mem_param_tmp_149199, "mem_param_tmp_149199") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146849, &mem_param_tmp_149200, "mem_param_tmp_149200") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146853, &mem_param_tmp_149201, "mem_param_tmp_149201") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146857, &mem_param_tmp_149202, "mem_param_tmp_149202") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_146861, &mem_param_tmp_149203, "mem_param_tmp_149203") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_149077, &mem_param_146757, "mem_param_146757") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149076, &mem_param_146761, "mem_param_146761") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149075, &mem_param_146765, "mem_param_146765") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149074, &mem_param_146769, "mem_param_146769") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149073, &mem_param_146773, "mem_param_146773") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149072, &mem_param_146777, "mem_param_146777") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149071, &mem_param_146781, "mem_param_146781") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149070, &mem_param_146785, "mem_param_146785") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149069, &mem_param_146789, "mem_param_146789") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149068, &mem_param_146793, "mem_param_146793") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149067, &mem_param_146797, "mem_param_146797") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149066, &mem_param_146801, "mem_param_146801") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149065, &mem_param_146805, "mem_param_146805") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149064, &mem_param_146809, "mem_param_146809") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149063, &mem_param_146813, "mem_param_146813") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149062, &mem_param_146817, "mem_param_146817") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149061, &mem_param_146821, "mem_param_146821") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149060, &mem_param_146825, "mem_param_146825") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149059, &mem_param_146829, "mem_param_146829") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149058, &mem_param_146833, "mem_param_146833") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149057, &mem_param_146837, "mem_param_146837") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149056, &mem_param_146841, "mem_param_146841") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149055, &mem_param_146845, "mem_param_146845") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149054, &mem_param_146849, "mem_param_146849") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149053, &mem_param_146853, "mem_param_146853") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149052, &mem_param_146857, "mem_param_146857") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_149051, &mem_param_146861, "mem_param_146861") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149150, &ext_mem_149072, "ext_mem_149072") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149151, &ext_mem_149074, "ext_mem_149074") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149152, &ext_mem_149073, "ext_mem_149073") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149153, &ext_mem_149076, "ext_mem_149076") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149154, &ext_mem_149070, "ext_mem_149070") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149155, &ext_mem_149075, "ext_mem_149075") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149156, &ext_mem_149071, "ext_mem_149071") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149157, &ext_mem_149077, "ext_mem_149077") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149158, &ext_mem_149069, "ext_mem_149069") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149159, &ext_mem_149063, "ext_mem_149063") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149160, &ext_mem_149065, "ext_mem_149065") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149161, &ext_mem_149064, "ext_mem_149064") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149162, &ext_mem_149067, "ext_mem_149067") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149163, &ext_mem_149061, "ext_mem_149061") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149164, &ext_mem_149066, "ext_mem_149066") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149165, &ext_mem_149062, "ext_mem_149062") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149166, &ext_mem_149068, "ext_mem_149068") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149167, &ext_mem_149060, "ext_mem_149060") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149168, &ext_mem_149054, "ext_mem_149054") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149169, &ext_mem_149056, "ext_mem_149056") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149170, &ext_mem_149055, "ext_mem_149055") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149171, &ext_mem_149058, "ext_mem_149058") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149172, &ext_mem_149052, "ext_mem_149052") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149173, &ext_mem_149057, "ext_mem_149057") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149174, &ext_mem_149053, "ext_mem_149053") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149175, &ext_mem_149059, "ext_mem_149059") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149176, &ext_mem_149051, "ext_mem_149051") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149702, &mem_out_149150, "mem_out_149150") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149703, &mem_out_149151, "mem_out_149151") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149704, &mem_out_149152, "mem_out_149152") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149705, &mem_out_149153, "mem_out_149153") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149706, &mem_out_149154, "mem_out_149154") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149707, &mem_out_149155, "mem_out_149155") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149708, &mem_out_149156, "mem_out_149156") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149709, &mem_out_149157, "mem_out_149157") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149710, &mem_out_149158, "mem_out_149158") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149711, &mem_out_149159, "mem_out_149159") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149712, &mem_out_149160, "mem_out_149160") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149713, &mem_out_149161, "mem_out_149161") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149714, &mem_out_149162, "mem_out_149162") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149715, &mem_out_149163, "mem_out_149163") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149716, &mem_out_149164, "mem_out_149164") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149717, &mem_out_149165, "mem_out_149165") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149718, &mem_out_149166, "mem_out_149166") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149719, &mem_out_149167, "mem_out_149167") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149720, &mem_out_149168, "mem_out_149168") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149721, &mem_out_149169, "mem_out_149169") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149722, &mem_out_149170, "mem_out_149170") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149723, &mem_out_149171, "mem_out_149171") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149724, &mem_out_149172, "mem_out_149172") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149725, &mem_out_149173, "mem_out_149173") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149726, &mem_out_149174, "mem_out_149174") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149727, &mem_out_149175, "mem_out_149175") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149728, &mem_out_149176, "mem_out_149176") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_146862);
        free(mem_146863);
        free(mem_146872);
        free(mem_146879);
        free(mem_146894);
        free(mem_146895);
        free(mem_146904);
        free(mem_146911);
        free(mem_146926);
        free(mem_146927);
        free(mem_146936);
        free(mem_146937);
        free(mem_146950);
        free(mem_146965);
        free(mem_146966);
        free(mem_146967);
        free(mem_146979);
        free(mem_146980);
        free(mem_146993);
        free(mem_147011);
        free(mem_147012);
        free(mem_147013);
        free(mem_147014);
        free(mem_147015);
        free(mem_147034);
        free(mem_147035);
        free(mem_147036);
        free(mem_147073);
        free(mem_147074);
        free(mem_147075);
        free(mem_147091);
        free(mem_147092);
        free(mem_147093);
        free(mem_147106);
        free(mem_147107);
        free(mem_147108);
        free(mem_147154);
        free(mem_147155);
        free(mem_147156);
        free(mem_147157);
        free(mem_147178);
        free(mem_147179);
        free(mem_147180);
        free(mem_147181);
        free(mem_147198);
        free(mem_147199);
        free(mem_147200);
        free(mem_147201);
        free(mem_147242);
        free(mem_147247);
        free(mem_147258);
        free(mem_147268);
        free(mem_147273);
        free(mem_147280);
        free(mem_147291);
        free(mem_147296);
        free(mem_147327);
        free(mem_147328);
        free(mem_147336);
        free(mem_147350);
        free(mem_147355);
        free(mem_147366);
        free(mem_147371);
        free(mem_147382);
        free(mem_147383);
        free(mem_147392);
        free(mem_147393);
        free(mem_147406);
        free(mem_147421);
        free(mem_147422);
        free(mem_147430);
        free(mem_147444);
        free(mem_147445);
        free(mem_147453);
        free(mem_147467);
        free(mem_147472);
        free(mem_147483);
        free(mem_147488);
        free(mem_147499);
        free(mem_147504);
        free(mem_147515);
        free(mem_147520);
        free(mem_147531);
        free(mem_147532);
        free(mem_147539);
        free(mem_147552);
        free(mem_147557);
        free(mem_147564);
        free(mem_147575);
        free(mem_147580);
        free(mem_147591);
        free(mem_147596);
        free(mem_147607);
        free(mem_147608);
        free(mem_147617);
        free(mem_147618);
        free(mem_147639);
        free(mem_147644);
        free(mem_147655);
        free(mem_147660);
        free(mem_147671);
        free(mem_147678);
        free(mem_147685);
        free(mem_147695);
        free(mem_147700);
        free(mem_147711);
        free(mem_147712);
        free(mem_147721);
        free(mem_147722);
        free(mem_147743);
        free(mem_147744);
        free(mem_147745);
        free(mem_147746);
        free(mem_147767);
        free(mem_147768);
        free(mem_147769);
        free(mem_147770);
        free(mem_147787);
        free(mem_147794);
        free(mem_147795);
        free(mem_147796);
        free(mem_147851);
        free(mem_147852);
        free(mem_147853);
        free(mem_147854);
        free(mem_147855);
        free(mem_147856);
        free(mem_147857);
        free(mem_147858);
        free(mem_147859);
        free(mem_147901);
        free(mem_147902);
        free(mem_147903);
        free(mem_147904);
        free(mem_147905);
        free(mem_147906);
        free(mem_147907);
        free(mem_147908);
        free(mem_147909);
        free(mem_147942);
        free(mem_147943);
        free(mem_148032);
        free(mem_148033);
        free(mem_148034);
        free(mem_148050);
        free(mem_148051);
        free(mem_148052);
        free(mem_148065);
        free(mem_148066);
        free(mem_148067);
        free(mem_148086);
        free(mem_148120);
        free(mem_148121);
        free(mem_148122);
        free(mem_148123);
        free(mem_148124);
        free(mem_148146);
        free(mem_148147);
        free(mem_148148);
        free(mem_148149);
        free(mem_148150);
        free(mem_148167);
        free(mem_148211);
        free(mem_148212);
        free(mem_148213);
        free(mem_148214);
        free(mem_148215);
        free(mem_148216);
        free(mem_148243);
        free(mem_148244);
        free(mem_148245);
        free(mem_148246);
        free(mem_148247);
        free(mem_148248);
        free(mem_148269);
        free(mem_148270);
        free(mem_148329);
        free(mem_148330);
        free(mem_148331);
        free(mem_148332);
        free(mem_148353);
        free(mem_148354);
        free(mem_148355);
        free(mem_148356);
        free(mem_148373);
        free(mem_148374);
        free(mem_148375);
        free(mem_148376);
        free(mem_148437);
        free(mem_148438);
        free(mem_148449);
        free(mem_148450);
        free(mem_148459);
        free(mem_148460);
        free(mem_148491);
        free(mem_148492);
        free(mem_148503);
        free(mem_148504);
        free(mem_148513);
        free(mem_148514);
        free(mem_148545);
        free(mem_148546);
        free(mem_148557);
        free(mem_148558);
        free(mem_148567);
        free(mem_148568);
        free(mem_148599);
        free(mem_148600);
        free(mem_148601);
        free(mem_148614);
        free(mem_148615);
        free(mem_148616);
        free(mem_148647);
        free(mem_148648);
        free(mem_148649);
        free(mem_148650);
        free(mem_148667);
        free(mem_148668);
        free(mem_148669);
        free(mem_148670);
        free(mem_148711);
        free(mem_148718);
        free(mem_148725);
        free(mem_148735);
        free(mem_148740);
        free(mem_148751);
        free(mem_148758);
        free(mem_148765);
        free(mem_148775);
        free(mem_148780);
        free(mem_148791);
        free(mem_148792);
        free(mem_148801);
        free(mem_148802);
        free(mem_148823);
        free(mem_148828);
        free(mem_148839);
        free(mem_148840);
        free(mem_148849);
        free(mem_148850);
        if (memblock_unref(ctx, &mem_param_tmp_149203, "mem_param_tmp_149203") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149202, "mem_param_tmp_149202") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149201, "mem_param_tmp_149201") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149200, "mem_param_tmp_149200") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149199, "mem_param_tmp_149199") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149198, "mem_param_tmp_149198") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149197, "mem_param_tmp_149197") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149196, "mem_param_tmp_149196") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149195, "mem_param_tmp_149195") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149194, "mem_param_tmp_149194") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149193, "mem_param_tmp_149193") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149192, "mem_param_tmp_149192") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149191, "mem_param_tmp_149191") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149190, "mem_param_tmp_149190") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149189, "mem_param_tmp_149189") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149188, "mem_param_tmp_149188") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149187, "mem_param_tmp_149187") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149186, "mem_param_tmp_149186") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149185, "mem_param_tmp_149185") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149184, "mem_param_tmp_149184") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149183, "mem_param_tmp_149183") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149182, "mem_param_tmp_149182") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149181, "mem_param_tmp_149181") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149180, "mem_param_tmp_149180") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149179, "mem_param_tmp_149179") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149178, "mem_param_tmp_149178") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_149177, "mem_param_tmp_149177") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148967, "ext_mem_148967") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148968, "ext_mem_148968") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148969, "ext_mem_148969") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148965, "mem_148965") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148963, "mem_148963") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148961, "mem_148961") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148959, "mem_148959") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148956, "ext_mem_148956") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148957, "ext_mem_148957") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148958, "ext_mem_148958") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148954, "mem_148954") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148952, "mem_148952") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148950, "mem_148950") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148948, "mem_148948") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148945, "ext_mem_148945") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148946, "ext_mem_148946") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148947, "ext_mem_148947") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148943, "mem_148943") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148941, "mem_148941") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148939, "mem_148939") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148937, "mem_148937") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148934, "ext_mem_148934") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148935, "ext_mem_148935") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148936, "ext_mem_148936") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148932, "mem_148932") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148930, "mem_148930") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148928, "mem_148928") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148926, "mem_148926") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148923, "ext_mem_148923") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148924, "ext_mem_148924") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148925, "ext_mem_148925") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148921, "mem_148921") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148919, "mem_148919") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148917, "mem_148917") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148915, "mem_148915") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148912, "ext_mem_148912") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148913, "ext_mem_148913") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148914, "ext_mem_148914") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148910, "mem_148910") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148908, "mem_148908") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148906, "mem_148906") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148904, "mem_148904") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148901, "ext_mem_148901") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148902, "ext_mem_148902") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148903, "ext_mem_148903") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148899, "mem_148899") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148897, "mem_148897") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148895, "mem_148895") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148893, "mem_148893") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148890, "ext_mem_148890") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148891, "ext_mem_148891") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148892, "ext_mem_148892") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148888, "mem_148888") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148886, "mem_148886") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148884, "mem_148884") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148882, "mem_148882") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148879, "ext_mem_148879") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148880, "ext_mem_148880") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_148881, "ext_mem_148881") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148877, "mem_148877") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148875, "mem_148875") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148873, "mem_148873") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_148871, "mem_148871") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146861, "mem_param_146861") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146857, "mem_param_146857") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146853, "mem_param_146853") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146849, "mem_param_146849") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146845, "mem_param_146845") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146841, "mem_param_146841") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146837, "mem_param_146837") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146833, "mem_param_146833") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146829, "mem_param_146829") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146825, "mem_param_146825") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146821, "mem_param_146821") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146817, "mem_param_146817") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146813, "mem_param_146813") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146809, "mem_param_146809") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146805, "mem_param_146805") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146801, "mem_param_146801") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146797, "mem_param_146797") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146793, "mem_param_146793") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146789, "mem_param_146789") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146785, "mem_param_146785") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146781, "mem_param_146781") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146777, "mem_param_146777") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146773, "mem_param_146773") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146769, "mem_param_146769") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146765, "mem_param_146765") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146761, "mem_param_146761") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_146757, "mem_param_146757") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149051, "ext_mem_149051") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149052, "ext_mem_149052") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149053, "ext_mem_149053") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149054, "ext_mem_149054") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149055, "ext_mem_149055") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149056, "ext_mem_149056") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149057, "ext_mem_149057") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149058, "ext_mem_149058") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149059, "ext_mem_149059") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149060, "ext_mem_149060") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149061, "ext_mem_149061") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149062, "ext_mem_149062") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149063, "ext_mem_149063") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149064, "ext_mem_149064") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149065, "ext_mem_149065") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149066, "ext_mem_149066") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149067, "ext_mem_149067") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149068, "ext_mem_149068") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149069, "ext_mem_149069") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149070, "ext_mem_149070") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149071, "ext_mem_149071") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149072, "ext_mem_149072") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149073, "ext_mem_149073") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149074, "ext_mem_149074") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149075, "ext_mem_149075") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149076, "ext_mem_149076") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_149077, "ext_mem_149077") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149176, "mem_out_149176") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149175, "mem_out_149175") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149174, "mem_out_149174") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149173, "mem_out_149173") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149172, "mem_out_149172") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149171, "mem_out_149171") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149170, "mem_out_149170") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149169, "mem_out_149169") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149168, "mem_out_149168") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149167, "mem_out_149167") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149166, "mem_out_149166") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149165, "mem_out_149165") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149164, "mem_out_149164") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149163, "mem_out_149163") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149162, "mem_out_149162") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149161, "mem_out_149161") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149160, "mem_out_149160") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149159, "mem_out_149159") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149158, "mem_out_149158") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149157, "mem_out_149157") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149156, "mem_out_149156") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149155, "mem_out_149155") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149154, "mem_out_149154") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149153, "mem_out_149153") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149152, "mem_out_149152") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149151, "mem_out_149151") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149150, "mem_out_149150") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_149969, struct memblock *mem_out_p_149970, struct memblock *mem_out_p_149971, struct memblock *mem_out_p_149972, struct memblock *mem_out_p_149973, struct memblock *mem_out_p_149974, struct memblock *mem_out_p_149975, struct memblock *mem_out_p_149976, struct memblock *mem_out_p_149977)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_149158;
    
    mem_out_149158.references = NULL;
    
    struct memblock mem_out_149157;
    
    mem_out_149157.references = NULL;
    
    struct memblock mem_out_149156;
    
    mem_out_149156.references = NULL;
    
    struct memblock mem_out_149155;
    
    mem_out_149155.references = NULL;
    
    struct memblock mem_out_149154;
    
    mem_out_149154.references = NULL;
    
    struct memblock mem_out_149153;
    
    mem_out_149153.references = NULL;
    
    struct memblock mem_out_149152;
    
    mem_out_149152.references = NULL;
    
    struct memblock mem_out_149151;
    
    mem_out_149151.references = NULL;
    
    struct memblock mem_out_149150;
    
    mem_out_149150.references = NULL;
    
    struct memblock mem_146715 = ctx->constants->mem_146715;
    struct memblock mem_146716 = ctx->constants->mem_146716;
    struct memblock mem_146717 = ctx->constants->mem_146717;
    struct memblock mem_146718 = ctx->constants->mem_146718;
    struct memblock mem_146719 = ctx->constants->mem_146719;
    struct memblock mem_146720 = ctx->constants->mem_146720;
    struct memblock mem_146721 = ctx->constants->mem_146721;
    struct memblock mem_146722 = ctx->constants->mem_146722;
    struct memblock mem_146723 = ctx->constants->mem_146723;
    
    if (memblock_set(ctx, &mem_out_149150, &mem_146722, "mem_146722") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149151, &mem_146718, "mem_146718") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149152, &mem_146720, "mem_146720") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149153, &mem_146716, "mem_146716") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149154, &mem_146717, "mem_146717") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149155, &mem_146715, "mem_146715") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149156, &mem_146721, "mem_146721") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149157, &mem_146719, "mem_146719") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_149158, &mem_146723, "mem_146723") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149969, &mem_out_149150, "mem_out_149150") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149970, &mem_out_149151, "mem_out_149151") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149971, &mem_out_149152, "mem_out_149152") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149972, &mem_out_149153, "mem_out_149153") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149973, &mem_out_149154, "mem_out_149154") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149974, &mem_out_149155, "mem_out_149155") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149975, &mem_out_149156, "mem_out_149156") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149976, &mem_out_149157, "mem_out_149157") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_149977, &mem_out_149158, "mem_out_149158") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_149158, "mem_out_149158") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149157, "mem_out_149157") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149156, "mem_out_149156") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149155, "mem_out_149155") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149154, "mem_out_149154") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149153, "mem_out_149153") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149152, "mem_out_149152") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149151, "mem_out_149151") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_149150, "mem_out_149150") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_cal_loss(struct futhark_context *ctx, struct futhark_opaque_tup2_f64_arr1d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3)
{
    double prim_out_149151 = 0.0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_149150;
    
    mem_out_149150.references = NULL;
    
    struct memblock mask_mem_146735;
    
    mask_mem_146735.references = NULL;
    
    struct memblock target_mem_146734;
    
    target_mem_146734.references = NULL;
    
    struct memblock tokens_mem_146733;
    
    tokens_mem_146733.references = NULL;
    
    struct memblock wvoc_mem_146732;
    
    wvoc_mem_146732.references = NULL;
    
    struct memblock wval_mem_146731;
    
    wval_mem_146731.references = NULL;
    
    struct memblock wup_mem_146730;
    
    wup_mem_146730.references = NULL;
    
    struct memblock wte_mem_146729;
    
    wte_mem_146729.references = NULL;
    
    struct memblock wqry_mem_146728;
    
    wqry_mem_146728.references = NULL;
    
    struct memblock wpe_mem_146727;
    
    wpe_mem_146727.references = NULL;
    
    struct memblock wout_mem_146726;
    
    wout_mem_146726.references = NULL;
    
    struct memblock wkey_mem_146725;
    
    wkey_mem_146725.references = NULL;
    
    struct memblock wdown_mem_146724;
    
    wdown_mem_146724.references = NULL;
    wdown_mem_146724 = in0->v0->mem;
    wkey_mem_146725 = in0->v1->mem;
    wout_mem_146726 = in0->v2->mem;
    wpe_mem_146727 = in0->v3->mem;
    wqry_mem_146728 = in0->v4->mem;
    wte_mem_146729 = in0->v5->mem;
    wup_mem_146730 = in0->v6->mem;
    wval_mem_146731 = in0->v7->mem;
    wvoc_mem_146732 = in0->v8->mem;
    tokens_mem_146733 = in1->mem;
    target_mem_146734 = in2->mem;
    mask_mem_146735 = in3->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && (((int64_t) 16 == in2->shape[0] && (int64_t) 27 == in2->shape[1]) && ((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_cal_loss(ctx, &mem_out_149150, &prim_out_149151, wdown_mem_146724, wkey_mem_146725, wout_mem_146726, wpe_mem_146727, wqry_mem_146728, wte_mem_146729, wup_mem_146730, wval_mem_146731, wvoc_mem_146732, tokens_mem_146733, target_mem_146734, mask_mem_146735);
        if (ret == 0) {
            struct memblock mem_146715 = ctx->constants->mem_146715;
            struct memblock mem_146716 = ctx->constants->mem_146716;
            struct memblock mem_146717 = ctx->constants->mem_146717;
            struct memblock mem_146718 = ctx->constants->mem_146718;
            struct memblock mem_146719 = ctx->constants->mem_146719;
            struct memblock mem_146720 = ctx->constants->mem_146720;
            struct memblock mem_146721 = ctx->constants->mem_146721;
            struct memblock mem_146722 = ctx->constants->mem_146722;
            struct memblock mem_146723 = ctx->constants->mem_146723;
            
            assert((*out = (struct futhark_opaque_tup2_f64_arr1d_f64 *) malloc(sizeof(struct futhark_opaque_tup2_f64_arr1d_f64))) != NULL);
            (*out)->v0 = prim_out_149151;
            assert(((*out)->v1 = (struct futhark_f64_1d *) malloc(sizeof(struct futhark_f64_1d))) != NULL);
            (*out)->v1->mem = mem_out_149150;
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
    
    struct memblock mem_out_149150;
    
    mem_out_149150.references = NULL;
    
    struct memblock mask_mem_146734;
    
    mask_mem_146734.references = NULL;
    
    struct memblock tokens_mem_146733;
    
    tokens_mem_146733.references = NULL;
    
    struct memblock wvoc_mem_146732;
    
    wvoc_mem_146732.references = NULL;
    
    struct memblock wval_mem_146731;
    
    wval_mem_146731.references = NULL;
    
    struct memblock wup_mem_146730;
    
    wup_mem_146730.references = NULL;
    
    struct memblock wte_mem_146729;
    
    wte_mem_146729.references = NULL;
    
    struct memblock wqry_mem_146728;
    
    wqry_mem_146728.references = NULL;
    
    struct memblock wpe_mem_146727;
    
    wpe_mem_146727.references = NULL;
    
    struct memblock wout_mem_146726;
    
    wout_mem_146726.references = NULL;
    
    struct memblock wkey_mem_146725;
    
    wkey_mem_146725.references = NULL;
    
    struct memblock wdown_mem_146724;
    
    wdown_mem_146724.references = NULL;
    wdown_mem_146724 = in0->v0->mem;
    wkey_mem_146725 = in0->v1->mem;
    wout_mem_146726 = in0->v2->mem;
    wpe_mem_146727 = in0->v3->mem;
    wqry_mem_146728 = in0->v4->mem;
    wte_mem_146729 = in0->v5->mem;
    wup_mem_146730 = in0->v6->mem;
    wval_mem_146731 = in0->v7->mem;
    wvoc_mem_146732 = in0->v8->mem;
    tokens_mem_146733 = in1->mem;
    mask_mem_146734 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_149150, wdown_mem_146724, wkey_mem_146725, wout_mem_146726, wpe_mem_146727, wqry_mem_146728, wte_mem_146729, wup_mem_146730, wval_mem_146731, wvoc_mem_146732, tokens_mem_146733, mask_mem_146734);
        if (ret == 0) {
            struct memblock mem_146715 = ctx->constants->mem_146715;
            struct memblock mem_146716 = ctx->constants->mem_146716;
            struct memblock mem_146717 = ctx->constants->mem_146717;
            struct memblock mem_146718 = ctx->constants->mem_146718;
            struct memblock mem_146719 = ctx->constants->mem_146719;
            struct memblock mem_146720 = ctx->constants->mem_146720;
            struct memblock mem_146721 = ctx->constants->mem_146721;
            struct memblock mem_146722 = ctx->constants->mem_146722;
            struct memblock mem_146723 = ctx->constants->mem_146723;
            
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_149150;
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
    
    struct memblock mem_out_149158;
    
    mem_out_149158.references = NULL;
    
    struct memblock mem_out_149157;
    
    mem_out_149157.references = NULL;
    
    struct memblock mem_out_149156;
    
    mem_out_149156.references = NULL;
    
    struct memblock mem_out_149155;
    
    mem_out_149155.references = NULL;
    
    struct memblock mem_out_149154;
    
    mem_out_149154.references = NULL;
    
    struct memblock mem_out_149153;
    
    mem_out_149153.references = NULL;
    
    struct memblock mem_out_149152;
    
    mem_out_149152.references = NULL;
    
    struct memblock mem_out_149151;
    
    mem_out_149151.references = NULL;
    
    struct memblock mem_out_149150;
    
    mem_out_149150.references = NULL;
    
    struct memblock wvoc_mem_146732;
    
    wvoc_mem_146732.references = NULL;
    
    struct memblock wdown_mem_146731;
    
    wdown_mem_146731.references = NULL;
    
    struct memblock wup_mem_146730;
    
    wup_mem_146730.references = NULL;
    
    struct memblock wout_mem_146729;
    
    wout_mem_146729.references = NULL;
    
    struct memblock wval_mem_146728;
    
    wval_mem_146728.references = NULL;
    
    struct memblock wkey_mem_146727;
    
    wkey_mem_146727.references = NULL;
    
    struct memblock wqry_mem_146726;
    
    wqry_mem_146726.references = NULL;
    
    struct memblock wpe_mem_146725;
    
    wpe_mem_146725.references = NULL;
    
    struct memblock wte_mem_146724;
    
    wte_mem_146724.references = NULL;
    wte_mem_146724 = in0->mem;
    wpe_mem_146725 = in1->mem;
    wqry_mem_146726 = in2->mem;
    wkey_mem_146727 = in3->mem;
    wval_mem_146728 = in4->mem;
    wout_mem_146729 = in5->mem;
    wup_mem_146730 = in6->mem;
    wdown_mem_146731 = in7->mem;
    wvoc_mem_146732 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_149150, &mem_out_149151, &mem_out_149152, &mem_out_149153, &mem_out_149154, &mem_out_149155, &mem_out_149156, &mem_out_149157, &mem_out_149158, wte_mem_146724, wpe_mem_146725, wqry_mem_146726, wkey_mem_146727, wval_mem_146728, wout_mem_146729, wup_mem_146730, wdown_mem_146731, wvoc_mem_146732);
        if (ret == 0) {
            struct memblock mem_146715 = ctx->constants->mem_146715;
            struct memblock mem_146716 = ctx->constants->mem_146716;
            struct memblock mem_146717 = ctx->constants->mem_146717;
            struct memblock mem_146718 = ctx->constants->mem_146718;
            struct memblock mem_146719 = ctx->constants->mem_146719;
            struct memblock mem_146720 = ctx->constants->mem_146720;
            struct memblock mem_146721 = ctx->constants->mem_146721;
            struct memblock mem_146722 = ctx->constants->mem_146722;
            struct memblock mem_146723 = ctx->constants->mem_146723;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_149150;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_149151;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_149152;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_149153;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_149154;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_149155;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_149156;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_149157;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_149158;
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
    
    struct memblock mem_out_149176;
    
    mem_out_149176.references = NULL;
    
    struct memblock mem_out_149175;
    
    mem_out_149175.references = NULL;
    
    struct memblock mem_out_149174;
    
    mem_out_149174.references = NULL;
    
    struct memblock mem_out_149173;
    
    mem_out_149173.references = NULL;
    
    struct memblock mem_out_149172;
    
    mem_out_149172.references = NULL;
    
    struct memblock mem_out_149171;
    
    mem_out_149171.references = NULL;
    
    struct memblock mem_out_149170;
    
    mem_out_149170.references = NULL;
    
    struct memblock mem_out_149169;
    
    mem_out_149169.references = NULL;
    
    struct memblock mem_out_149168;
    
    mem_out_149168.references = NULL;
    
    struct memblock mem_out_149167;
    
    mem_out_149167.references = NULL;
    
    struct memblock mem_out_149166;
    
    mem_out_149166.references = NULL;
    
    struct memblock mem_out_149165;
    
    mem_out_149165.references = NULL;
    
    struct memblock mem_out_149164;
    
    mem_out_149164.references = NULL;
    
    struct memblock mem_out_149163;
    
    mem_out_149163.references = NULL;
    
    struct memblock mem_out_149162;
    
    mem_out_149162.references = NULL;
    
    struct memblock mem_out_149161;
    
    mem_out_149161.references = NULL;
    
    struct memblock mem_out_149160;
    
    mem_out_149160.references = NULL;
    
    struct memblock mem_out_149159;
    
    mem_out_149159.references = NULL;
    
    struct memblock mem_out_149158;
    
    mem_out_149158.references = NULL;
    
    struct memblock mem_out_149157;
    
    mem_out_149157.references = NULL;
    
    struct memblock mem_out_149156;
    
    mem_out_149156.references = NULL;
    
    struct memblock mem_out_149155;
    
    mem_out_149155.references = NULL;
    
    struct memblock mem_out_149154;
    
    mem_out_149154.references = NULL;
    
    struct memblock mem_out_149153;
    
    mem_out_149153.references = NULL;
    
    struct memblock mem_out_149152;
    
    mem_out_149152.references = NULL;
    
    struct memblock mem_out_149151;
    
    mem_out_149151.references = NULL;
    
    struct memblock mem_out_149150;
    
    mem_out_149150.references = NULL;
    
    struct memblock seqs_mem_146753;
    
    seqs_mem_146753.references = NULL;
    
    struct memblock dls_mem_146752;
    
    dls_mem_146752.references = NULL;
    
    struct memblock masks_mem_146751;
    
    masks_mem_146751.references = NULL;
    
    struct memblock wvoc_mem_146750;
    
    wvoc_mem_146750.references = NULL;
    
    struct memblock wval_mem_146749;
    
    wval_mem_146749.references = NULL;
    
    struct memblock wup_mem_146748;
    
    wup_mem_146748.references = NULL;
    
    struct memblock wte_mem_146747;
    
    wte_mem_146747.references = NULL;
    
    struct memblock wqry_mem_146746;
    
    wqry_mem_146746.references = NULL;
    
    struct memblock wpe_mem_146745;
    
    wpe_mem_146745.references = NULL;
    
    struct memblock wout_mem_146744;
    
    wout_mem_146744.references = NULL;
    
    struct memblock wkey_mem_146743;
    
    wkey_mem_146743.references = NULL;
    
    struct memblock wdown_mem_146742;
    
    wdown_mem_146742.references = NULL;
    
    struct memblock wvoc_mem_146741;
    
    wvoc_mem_146741.references = NULL;
    
    struct memblock wval_mem_146740;
    
    wval_mem_146740.references = NULL;
    
    struct memblock wup_mem_146739;
    
    wup_mem_146739.references = NULL;
    
    struct memblock wte_mem_146738;
    
    wte_mem_146738.references = NULL;
    
    struct memblock wqry_mem_146737;
    
    wqry_mem_146737.references = NULL;
    
    struct memblock wpe_mem_146736;
    
    wpe_mem_146736.references = NULL;
    
    struct memblock wout_mem_146735;
    
    wout_mem_146735.references = NULL;
    
    struct memblock wkey_mem_146734;
    
    wkey_mem_146734.references = NULL;
    
    struct memblock wdown_mem_146733;
    
    wdown_mem_146733.references = NULL;
    
    struct memblock wvoc_mem_146732;
    
    wvoc_mem_146732.references = NULL;
    
    struct memblock wval_mem_146731;
    
    wval_mem_146731.references = NULL;
    
    struct memblock wup_mem_146730;
    
    wup_mem_146730.references = NULL;
    
    struct memblock wte_mem_146729;
    
    wte_mem_146729.references = NULL;
    
    struct memblock wqry_mem_146728;
    
    wqry_mem_146728.references = NULL;
    
    struct memblock wpe_mem_146727;
    
    wpe_mem_146727.references = NULL;
    
    struct memblock wout_mem_146726;
    
    wout_mem_146726.references = NULL;
    
    struct memblock wkey_mem_146725;
    
    wkey_mem_146725.references = NULL;
    
    struct memblock wdown_mem_146724;
    
    wdown_mem_146724.references = NULL;
    wdown_mem_146724 = in0->v0->mem;
    wkey_mem_146725 = in0->v1->mem;
    wout_mem_146726 = in0->v2->mem;
    wpe_mem_146727 = in0->v3->mem;
    wqry_mem_146728 = in0->v4->mem;
    wte_mem_146729 = in0->v5->mem;
    wup_mem_146730 = in0->v6->mem;
    wval_mem_146731 = in0->v7->mem;
    wvoc_mem_146732 = in0->v8->mem;
    wdown_mem_146733 = in1->v0->mem;
    wkey_mem_146734 = in1->v1->mem;
    wout_mem_146735 = in1->v2->mem;
    wpe_mem_146736 = in1->v3->mem;
    wqry_mem_146737 = in1->v4->mem;
    wte_mem_146738 = in1->v5->mem;
    wup_mem_146739 = in1->v6->mem;
    wval_mem_146740 = in1->v7->mem;
    wvoc_mem_146741 = in1->v8->mem;
    wdown_mem_146742 = in2->v0->mem;
    wkey_mem_146743 = in2->v1->mem;
    wout_mem_146744 = in2->v2->mem;
    wpe_mem_146745 = in2->v3->mem;
    wqry_mem_146746 = in2->v4->mem;
    wte_mem_146747 = in2->v5->mem;
    wup_mem_146748 = in2->v6->mem;
    wval_mem_146749 = in2->v7->mem;
    wvoc_mem_146750 = in2->v8->mem;
    masks_mem_146751 = in3->mem;
    dls_mem_146752 = in4->mem;
    seqs_mem_146753 = in5->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && (((int64_t) 50 == in3->shape[0] && ((int64_t) 16 == in3->shape[1] && (int64_t) 16 == in3->shape[2])) && ((int64_t) 50 == in4->shape[0] && ((int64_t) 50 == in5->shape[0] && (int64_t) 16 == in5->shape[1]))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_149150, &mem_out_149151, &mem_out_149152, &mem_out_149153, &mem_out_149154, &mem_out_149155, &mem_out_149156, &mem_out_149157, &mem_out_149158, &mem_out_149159, &mem_out_149160, &mem_out_149161, &mem_out_149162, &mem_out_149163, &mem_out_149164, &mem_out_149165, &mem_out_149166, &mem_out_149167, &mem_out_149168, &mem_out_149169, &mem_out_149170, &mem_out_149171, &mem_out_149172, &mem_out_149173, &mem_out_149174, &mem_out_149175, &mem_out_149176, wdown_mem_146724, wkey_mem_146725, wout_mem_146726, wpe_mem_146727, wqry_mem_146728, wte_mem_146729, wup_mem_146730, wval_mem_146731, wvoc_mem_146732, wdown_mem_146733, wkey_mem_146734, wout_mem_146735, wpe_mem_146736, wqry_mem_146737, wte_mem_146738, wup_mem_146739, wval_mem_146740, wvoc_mem_146741, wdown_mem_146742, wkey_mem_146743, wout_mem_146744, wpe_mem_146745, wqry_mem_146746, wte_mem_146747, wup_mem_146748, wval_mem_146749, wvoc_mem_146750, masks_mem_146751, dls_mem_146752, seqs_mem_146753);
        if (ret == 0) {
            struct memblock mem_146715 = ctx->constants->mem_146715;
            struct memblock mem_146716 = ctx->constants->mem_146716;
            struct memblock mem_146717 = ctx->constants->mem_146717;
            struct memblock mem_146718 = ctx->constants->mem_146718;
            struct memblock mem_146719 = ctx->constants->mem_146719;
            struct memblock mem_146720 = ctx->constants->mem_146720;
            struct memblock mem_146721 = ctx->constants->mem_146721;
            struct memblock mem_146722 = ctx->constants->mem_146722;
            struct memblock mem_146723 = ctx->constants->mem_146723;
            
            assert((*out = (struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_149150;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_149151;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_149152;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_149153;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_149154;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_149155;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_149156;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_149157;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_149158;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v9->mem = mem_out_149159;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v10->mem = mem_out_149160;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v11->mem = mem_out_149161;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v12->mem = mem_out_149162;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v13->mem = mem_out_149163;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v14->mem = mem_out_149164;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v15->mem = mem_out_149165;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v16->mem = mem_out_149166;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v17->mem = mem_out_149167;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v18->mem = mem_out_149168;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v19->mem = mem_out_149169;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v20->mem = mem_out_149170;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v21->mem = mem_out_149171;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v22->mem = mem_out_149172;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v23->mem = mem_out_149173;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v24->mem = mem_out_149174;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v25->mem = mem_out_149175;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v26->mem = mem_out_149176;
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
    
    struct memblock mem_out_149158;
    
    mem_out_149158.references = NULL;
    
    struct memblock mem_out_149157;
    
    mem_out_149157.references = NULL;
    
    struct memblock mem_out_149156;
    
    mem_out_149156.references = NULL;
    
    struct memblock mem_out_149155;
    
    mem_out_149155.references = NULL;
    
    struct memblock mem_out_149154;
    
    mem_out_149154.references = NULL;
    
    struct memblock mem_out_149153;
    
    mem_out_149153.references = NULL;
    
    struct memblock mem_out_149152;
    
    mem_out_149152.references = NULL;
    
    struct memblock mem_out_149151;
    
    mem_out_149151.references = NULL;
    
    struct memblock mem_out_149150;
    
    mem_out_149150.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_149150, &mem_out_149151, &mem_out_149152, &mem_out_149153, &mem_out_149154, &mem_out_149155, &mem_out_149156, &mem_out_149157, &mem_out_149158);
        if (ret == 0) {
            struct memblock mem_146715 = ctx->constants->mem_146715;
            struct memblock mem_146716 = ctx->constants->mem_146716;
            struct memblock mem_146717 = ctx->constants->mem_146717;
            struct memblock mem_146718 = ctx->constants->mem_146718;
            struct memblock mem_146719 = ctx->constants->mem_146719;
            struct memblock mem_146720 = ctx->constants->mem_146720;
            struct memblock mem_146721 = ctx->constants->mem_146721;
            struct memblock mem_146722 = ctx->constants->mem_146722;
            struct memblock mem_146723 = ctx->constants->mem_146723;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_149150;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_149151;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_149152;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_149153;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_149154;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_149155;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_149156;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_149157;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_149158;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
