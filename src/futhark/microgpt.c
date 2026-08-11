
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
    struct memblock mem_114414;
    struct memblock mem_114415;
    struct memblock mem_114416;
    struct memblock mem_114417;
    struct memblock mem_114418;
    struct memblock mem_114419;
    struct memblock mem_114420;
    struct memblock mem_114421;
    struct memblock mem_114422;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_11669(struct futhark_context *ctx, struct memblock *mem_out_p_116786, struct memblock *mem_out_p_116787, struct memblock *mem_out_p_116788, struct memblock w_mem_114423, struct memblock mw_mem_114424, struct memblock vw_mem_114425, struct memblock dw_mem_114426, int64_t n_85478, int64_t m_85479, int64_t step_85484, double lt_r_85485);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_11670(struct futhark_context *ctx, struct memblock *mem_out_p_116791, struct memblock *mem_out_p_116792, struct memblock *mem_out_p_116793, struct memblock w_mem_114423, struct memblock mw_mem_114424, struct memblock vw_mem_114425, struct memblock dw_mem_114426, int64_t n_86511, int64_t m_86512, int64_t step_86517, double lt_r_86518);
FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_116796, double *out_prim_out_116797, struct memblock wdown_mem_114423, struct memblock wkey_mem_114424, struct memblock wout_mem_114425, struct memblock wpe_mem_114426, struct memblock wqry_mem_114427, struct memblock wte_mem_114428, struct memblock wup_mem_114429, struct memblock wval_mem_114430, struct memblock wvoc_mem_114431, struct memblock tokens_mem_114432, struct memblock target_mem_114433, struct memblock mask_mem_114434);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_116855, struct memblock wdown_mem_114423, struct memblock wkey_mem_114424, struct memblock wout_mem_114425, struct memblock wpe_mem_114426, struct memblock wqry_mem_114427, struct memblock wte_mem_114428, struct memblock wup_mem_114429, struct memblock wval_mem_114430, struct memblock wvoc_mem_114431, struct memblock tokens_mem_114432, struct memblock mask_mem_114433);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_116912, struct memblock *mem_out_p_116913, struct memblock *mem_out_p_116914, struct memblock *mem_out_p_116915, struct memblock *mem_out_p_116916, struct memblock *mem_out_p_116917, struct memblock *mem_out_p_116918, struct memblock *mem_out_p_116919, struct memblock *mem_out_p_116920, struct memblock wte_mem_114423, struct memblock wpe_mem_114424, struct memblock wqry_mem_114425, struct memblock wkey_mem_114426, struct memblock wval_mem_114427, struct memblock wout_mem_114428, struct memblock wup_mem_114429, struct memblock wdown_mem_114430, struct memblock wvoc_mem_114431);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_116921, struct memblock *mem_out_p_116922, struct memblock *mem_out_p_116923, struct memblock *mem_out_p_116924, struct memblock *mem_out_p_116925, struct memblock *mem_out_p_116926, struct memblock *mem_out_p_116927, struct memblock *mem_out_p_116928, struct memblock *mem_out_p_116929, struct memblock *mem_out_p_116930, struct memblock *mem_out_p_116931, struct memblock *mem_out_p_116932, struct memblock *mem_out_p_116933, struct memblock *mem_out_p_116934, struct memblock *mem_out_p_116935, struct memblock *mem_out_p_116936, struct memblock *mem_out_p_116937, struct memblock *mem_out_p_116938, struct memblock *mem_out_p_116939, struct memblock *mem_out_p_116940, struct memblock *mem_out_p_116941, struct memblock *mem_out_p_116942, struct memblock *mem_out_p_116943, struct memblock *mem_out_p_116944, struct memblock *mem_out_p_116945, struct memblock *mem_out_p_116946, struct memblock *mem_out_p_116947, struct memblock wdown_mem_114423, struct memblock wkey_mem_114424, struct memblock wout_mem_114425, struct memblock wpe_mem_114426, struct memblock wqry_mem_114427, struct memblock wte_mem_114428, struct memblock wup_mem_114429, struct memblock wval_mem_114430, struct memblock wvoc_mem_114431, struct memblock wdown_mem_114432, struct memblock wkey_mem_114433, struct memblock wout_mem_114434, struct memblock wpe_mem_114435, struct memblock wqry_mem_114436, struct memblock wte_mem_114437, struct memblock wup_mem_114438, struct memblock wval_mem_114439, struct memblock wvoc_mem_114440, struct memblock wdown_mem_114441, struct memblock wkey_mem_114442, struct memblock wout_mem_114443, struct memblock wpe_mem_114444, struct memblock wqry_mem_114445, struct memblock wte_mem_114446, struct memblock wup_mem_114447, struct memblock wval_mem_114448, struct memblock wvoc_mem_114449, struct memblock masks_mem_114450, struct memblock dls_mem_114451, struct memblock seqs_mem_114452);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_117143, struct memblock *mem_out_p_117144, struct memblock *mem_out_p_117145, struct memblock *mem_out_p_117146, struct memblock *mem_out_p_117147, struct memblock *mem_out_p_117148, struct memblock *mem_out_p_117149, struct memblock *mem_out_p_117150, struct memblock *mem_out_p_117151);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_114414 (ctx->constants->mem_114414)
    #define mem_114415 (ctx->constants->mem_114415)
    #define mem_114416 (ctx->constants->mem_114416)
    #define mem_114417 (ctx->constants->mem_114417)
    #define mem_114418 (ctx->constants->mem_114418)
    #define mem_114419 (ctx->constants->mem_114419)
    #define mem_114420 (ctx->constants->mem_114420)
    #define mem_114421 (ctx->constants->mem_114421)
    #define mem_114422 (ctx->constants->mem_114422)
    mem_114414.references = NULL;
    mem_114415.references = NULL;
    mem_114416.references = NULL;
    mem_114417.references = NULL;
    mem_114418.references = NULL;
    mem_114419.references = NULL;
    mem_114420.references = NULL;
    mem_114421.references = NULL;
    mem_114422.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_114414, (int64_t) 3456, "mem_114414")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_116768 = 0; nest_i_116768 < (int64_t) 27; nest_i_116768++) {
        for (int64_t nest_i_116769 = 0; nest_i_116769 < (int64_t) 16; nest_i_116769++) {
            ((double *) mem_114414.mem)[nest_i_116768 * (int64_t) 16 + nest_i_116769] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_114415, (int64_t) 2048, "mem_114415")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_116770 = 0; nest_i_116770 < (int64_t) 16; nest_i_116770++) {
        for (int64_t nest_i_116771 = 0; nest_i_116771 < (int64_t) 16; nest_i_116771++) {
            ((double *) mem_114415.mem)[nest_i_116770 * (int64_t) 16 + nest_i_116771] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_114416, (int64_t) 2048, "mem_114416")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_116772 = 0; nest_i_116772 < (int64_t) 16; nest_i_116772++) {
        for (int64_t nest_i_116773 = 0; nest_i_116773 < (int64_t) 16; nest_i_116773++) {
            ((double *) mem_114416.mem)[nest_i_116772 * (int64_t) 16 + nest_i_116773] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_114417, (int64_t) 2048, "mem_114417")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_116774 = 0; nest_i_116774 < (int64_t) 16; nest_i_116774++) {
        for (int64_t nest_i_116775 = 0; nest_i_116775 < (int64_t) 16; nest_i_116775++) {
            ((double *) mem_114417.mem)[nest_i_116774 * (int64_t) 16 + nest_i_116775] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_114418, (int64_t) 2048, "mem_114418")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_116776 = 0; nest_i_116776 < (int64_t) 16; nest_i_116776++) {
        for (int64_t nest_i_116777 = 0; nest_i_116777 < (int64_t) 16; nest_i_116777++) {
            ((double *) mem_114418.mem)[nest_i_116776 * (int64_t) 16 + nest_i_116777] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_114419, (int64_t) 2048, "mem_114419")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_116778 = 0; nest_i_116778 < (int64_t) 16; nest_i_116778++) {
        for (int64_t nest_i_116779 = 0; nest_i_116779 < (int64_t) 16; nest_i_116779++) {
            ((double *) mem_114419.mem)[nest_i_116778 * (int64_t) 16 + nest_i_116779] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_114420, (int64_t) 8192, "mem_114420")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_116780 = 0; nest_i_116780 < (int64_t) 64; nest_i_116780++) {
        for (int64_t nest_i_116781 = 0; nest_i_116781 < (int64_t) 16; nest_i_116781++) {
            ((double *) mem_114420.mem)[nest_i_116780 * (int64_t) 16 + nest_i_116781] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_114421, (int64_t) 8192, "mem_114421")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_116782 = 0; nest_i_116782 < (int64_t) 16; nest_i_116782++) {
        for (int64_t nest_i_116783 = 0; nest_i_116783 < (int64_t) 64; nest_i_116783++) {
            ((double *) mem_114421.mem)[nest_i_116782 * (int64_t) 64 + nest_i_116783] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_114422, (int64_t) 3456, "mem_114422")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_116784 = 0; nest_i_116784 < (int64_t) 27; nest_i_116784++) {
        for (int64_t nest_i_116785 = 0; nest_i_116785 < (int64_t) 16; nest_i_116785++) {
            ((double *) mem_114422.mem)[nest_i_116784 * (int64_t) 16 + nest_i_116785] = 0.0;
        }
    }
    #undef mem_114414
    #undef mem_114415
    #undef mem_114416
    #undef mem_114417
    #undef mem_114418
    #undef mem_114419
    #undef mem_114420
    #undef mem_114421
    #undef mem_114422
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_114414, "ctx->constants->mem_114414") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_114415, "ctx->constants->mem_114415") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_114416, "ctx->constants->mem_114416") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_114417, "ctx->constants->mem_114417") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_114418, "ctx->constants->mem_114418") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_114419, "ctx->constants->mem_114419") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_114420, "ctx->constants->mem_114420") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_114421, "ctx->constants->mem_114421") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_114422, "ctx->constants->mem_114422") != 0)
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_11669(struct futhark_context *ctx, struct memblock *mem_out_p_116786, struct memblock *mem_out_p_116787, struct memblock *mem_out_p_116788, struct memblock w_mem_114423, struct memblock mw_mem_114424, struct memblock vw_mem_114425, struct memblock dw_mem_114426, int64_t n_85478, int64_t m_85479, int64_t step_85484, double lt_r_85485)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_114467_cached_sizze_116789 = 0;
    unsigned char *mem_114467 = NULL;
    int64_t mem_114470_cached_sizze_116790 = 0;
    unsigned char *mem_114470 = NULL;
    struct memblock mem_114505;
    
    mem_114505.references = NULL;
    
    struct memblock mem_114432;
    
    mem_114432.references = NULL;
    
    struct memblock mem_114429;
    
    mem_114429.references = NULL;
    
    struct memblock mem_out_116439;
    
    mem_out_116439.references = NULL;
    
    struct memblock mem_out_116438;
    
    mem_out_116438.references = NULL;
    
    struct memblock mem_out_116437;
    
    mem_out_116437.references = NULL;
    
    struct memblock mem_114414 = ctx->constants->mem_114414;
    struct memblock mem_114415 = ctx->constants->mem_114415;
    struct memblock mem_114416 = ctx->constants->mem_114416;
    struct memblock mem_114417 = ctx->constants->mem_114417;
    struct memblock mem_114418 = ctx->constants->mem_114418;
    struct memblock mem_114419 = ctx->constants->mem_114419;
    struct memblock mem_114420 = ctx->constants->mem_114420;
    struct memblock mem_114421 = ctx->constants->mem_114421;
    struct memblock mem_114422 = ctx->constants->mem_114422;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_114427 = (int64_t) 8 * n_85478;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_114428 = m_85479 * binop_x_114427;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_114429, bytes_114428, "mem_114429")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_114432, bytes_114428, "mem_114432")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113421 = 0; i_113421 < n_85478; i_113421++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113414 = 0; i_113414 < m_85479; i_113414++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_108322 = ((double *) mw_mem_114424.mem)[i_113421 * m_85479 + i_113414];
            
            // futhark/microgpt.fut:455:10-20
            
            double zp_lhs_108323 = 0.85 * zt_rhs_108322;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_108324 = ((double *) dw_mem_114426.mem)[i_113421 * m_85479 + i_113414];
            
            // futhark/microgpt.fut:455:35-45
            
            double zp_rhs_108325 = 0.15000000000000002 * zt_rhs_108324;
            
            // futhark/microgpt.fut:455:21-45
            
            double lifted_lambda_res_108326 = zp_lhs_108323 + zp_rhs_108325;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_108333 = ((double *) vw_mem_114425.mem)[i_113421 * m_85479 + i_113414];
            
            // futhark/microgpt.fut:457:10-20
            
            double zp_lhs_108334 = 0.99 * zt_rhs_108333;
            
            // futhark/microgpt.fut:457:35-45
            
            double zt_lhs_108336 = 1.0000000000000009e-2 * zt_rhs_108324;
            
            // futhark/microgpt.fut:457:46-56
            
            double zp_rhs_108337 = zt_rhs_108324 * zt_lhs_108336;
            
            // futhark/microgpt.fut:457:21-56
            
            double lifted_lambda_res_108338 = zp_lhs_108334 + zp_rhs_108337;
            
            ((double *) mem_114429.mem)[i_113421 * m_85479 + i_113414] = lifted_lambda_res_108338;
            ((double *) mem_114432.mem)[i_113421 * m_85479 + i_113414] = lifted_lambda_res_108326;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_90486 = sitofp_i64_f64(step_85484);
    
    // futhark/microgpt.fut:459:54-57
    
    double ztzt_rhs_90487 = 1.0 + i64_res_90486;
    
    // futhark/microgpt.fut:459:30-57
    
    double zm_rhs_90488 = fpow64(0.85, ztzt_rhs_90487);
    
    // futhark/microgpt.fut:459:23-57
    
    double zs_rhs_90489 = 1.0 - zm_rhs_90488;
    
    // futhark/microgpt.fut:461:31-58
    
    double zm_rhs_90527 = fpow64(0.99, ztzt_rhs_90487);
    
    // futhark/microgpt.fut:461:23-58
    
    double zs_rhs_90528 = 1.0 - zm_rhs_90527;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_114467_cached_sizze_116789 < bytes_114428) {
        err = lexical_realloc(ctx, &mem_114467, &mem_114467_cached_sizze_116789, bytes_114428);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114470_cached_sizze_116790 < bytes_114428) {
        err = lexical_realloc(ctx, &mem_114470, &mem_114470_cached_sizze_116790, bytes_114428);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113435 = 0; i_113435 < n_85478; i_113435++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113428 = 0; i_113428 < m_85479; i_113428++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_108358 = ((double *) mem_114432.mem)[i_113435 * m_85479 + i_113428];
            
            // futhark/microgpt.fut:459:18-57
            
            double lifted_lambda_res_108359 = zs_lhs_108358 / zs_rhs_90489;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_108366 = ((double *) mem_114429.mem)[i_113435 * m_85479 + i_113428];
            
            // futhark/microgpt.fut:461:18-58
            
            double lifted_lambda_res_108367 = zs_lhs_108366 / zs_rhs_90528;
            
            ((double *) mem_114467)[i_113435 * m_85479 + i_113428] = lifted_lambda_res_108367;
            ((double *) mem_114470)[i_113435 * m_85479 + i_113428] = lifted_lambda_res_108359;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_114505, bytes_114428, "mem_114505")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113444 = 0; i_113444 < n_85478; i_113444++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113440 = 0; i_113440 < m_85479; i_113440++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_89779 = ((double *) w_mem_114423.mem)[i_113444 * m_85479 + i_113440];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_89780 = ((double *) mem_114470)[i_113444 * m_85479 + i_113440];
            
            // futhark/microgpt.fut:463:21-34
            
            double zs_lhs_89781 = lt_r_85485 * zt_rhs_89780;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_89782 = ((double *) mem_114467)[i_113444 * m_85479 + i_113440];
            
            // futhark/microgpt.fut:463:51-57
            
            double zp_lhs_89783 = fpow64(ztzt_lhs_89782, 0.5);
            
            // futhark/microgpt.fut:463:59-71
            
            double zs_rhs_89784 = 1.0e-8 + zp_lhs_89783;
            
            // futhark/microgpt.fut:463:35-71
            
            double zm_rhs_89785 = zs_lhs_89781 / zs_rhs_89784;
            
            // futhark/microgpt.fut:463:13-71
            
            double lifted_lambda_res_89786 = zm_lhs_89779 - zm_rhs_89785;
            
            ((double *) mem_114505.mem)[i_113444 * m_85479 + i_113440] = lifted_lambda_res_89786;
        }
    }
    if (memblock_set(ctx, &mem_out_116437, &mem_114505, "mem_114505") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116438, &mem_114432, "mem_114432") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116439, &mem_114429, "mem_114429") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116786, &mem_out_116437, "mem_out_116437") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116787, &mem_out_116438, "mem_out_116438") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116788, &mem_out_116439, "mem_out_116439") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_114467);
        free(mem_114470);
        if (memblock_unref(ctx, &mem_114505, "mem_114505") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_114432, "mem_114432") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_114429, "mem_114429") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116439, "mem_out_116439") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116438, "mem_out_116438") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116437, "mem_out_116437") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_11670(struct futhark_context *ctx, struct memblock *mem_out_p_116791, struct memblock *mem_out_p_116792, struct memblock *mem_out_p_116793, struct memblock w_mem_114423, struct memblock mw_mem_114424, struct memblock vw_mem_114425, struct memblock dw_mem_114426, int64_t n_86511, int64_t m_86512, int64_t step_86517, double lt_r_86518)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_114467_cached_sizze_116794 = 0;
    unsigned char *mem_114467 = NULL;
    int64_t mem_114470_cached_sizze_116795 = 0;
    unsigned char *mem_114470 = NULL;
    struct memblock mem_114505;
    
    mem_114505.references = NULL;
    
    struct memblock mem_114432;
    
    mem_114432.references = NULL;
    
    struct memblock mem_114429;
    
    mem_114429.references = NULL;
    
    struct memblock mem_out_116439;
    
    mem_out_116439.references = NULL;
    
    struct memblock mem_out_116438;
    
    mem_out_116438.references = NULL;
    
    struct memblock mem_out_116437;
    
    mem_out_116437.references = NULL;
    
    struct memblock mem_114414 = ctx->constants->mem_114414;
    struct memblock mem_114415 = ctx->constants->mem_114415;
    struct memblock mem_114416 = ctx->constants->mem_114416;
    struct memblock mem_114417 = ctx->constants->mem_114417;
    struct memblock mem_114418 = ctx->constants->mem_114418;
    struct memblock mem_114419 = ctx->constants->mem_114419;
    struct memblock mem_114420 = ctx->constants->mem_114420;
    struct memblock mem_114421 = ctx->constants->mem_114421;
    struct memblock mem_114422 = ctx->constants->mem_114422;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_114427 = (int64_t) 8 * n_86511;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_114428 = m_86512 * binop_x_114427;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_114429, bytes_114428, "mem_114429")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_114432, bytes_114428, "mem_114432")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113421 = 0; i_113421 < n_86511; i_113421++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113414 = 0; i_113414 < m_86512; i_113414++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_108322 = ((double *) mw_mem_114424.mem)[i_113421 * m_86512 + i_113414];
            
            // futhark/microgpt.fut:455:10-20
            
            double zp_lhs_108323 = 0.85 * zt_rhs_108322;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_108324 = ((double *) dw_mem_114426.mem)[i_113421 * m_86512 + i_113414];
            
            // futhark/microgpt.fut:455:35-45
            
            double zp_rhs_108325 = 0.15000000000000002 * zt_rhs_108324;
            
            // futhark/microgpt.fut:455:21-45
            
            double lifted_lambda_res_108326 = zp_lhs_108323 + zp_rhs_108325;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_108333 = ((double *) vw_mem_114425.mem)[i_113421 * m_86512 + i_113414];
            
            // futhark/microgpt.fut:457:10-20
            
            double zp_lhs_108334 = 0.99 * zt_rhs_108333;
            
            // futhark/microgpt.fut:457:35-45
            
            double zt_lhs_108336 = 1.0000000000000009e-2 * zt_rhs_108324;
            
            // futhark/microgpt.fut:457:46-56
            
            double zp_rhs_108337 = zt_rhs_108324 * zt_lhs_108336;
            
            // futhark/microgpt.fut:457:21-56
            
            double lifted_lambda_res_108338 = zp_lhs_108334 + zp_rhs_108337;
            
            ((double *) mem_114429.mem)[i_113421 * m_86512 + i_113414] = lifted_lambda_res_108338;
            ((double *) mem_114432.mem)[i_113421 * m_86512 + i_113414] = lifted_lambda_res_108326;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_90486 = sitofp_i64_f64(step_86517);
    
    // futhark/microgpt.fut:459:54-57
    
    double ztzt_rhs_90487 = 1.0 + i64_res_90486;
    
    // futhark/microgpt.fut:459:30-57
    
    double zm_rhs_90488 = fpow64(0.85, ztzt_rhs_90487);
    
    // futhark/microgpt.fut:459:23-57
    
    double zs_rhs_90489 = 1.0 - zm_rhs_90488;
    
    // futhark/microgpt.fut:461:31-58
    
    double zm_rhs_90527 = fpow64(0.99, ztzt_rhs_90487);
    
    // futhark/microgpt.fut:461:23-58
    
    double zs_rhs_90528 = 1.0 - zm_rhs_90527;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_114467_cached_sizze_116794 < bytes_114428) {
        err = lexical_realloc(ctx, &mem_114467, &mem_114467_cached_sizze_116794, bytes_114428);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114470_cached_sizze_116795 < bytes_114428) {
        err = lexical_realloc(ctx, &mem_114470, &mem_114470_cached_sizze_116795, bytes_114428);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113435 = 0; i_113435 < n_86511; i_113435++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113428 = 0; i_113428 < m_86512; i_113428++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_108358 = ((double *) mem_114432.mem)[i_113435 * m_86512 + i_113428];
            
            // futhark/microgpt.fut:459:18-57
            
            double lifted_lambda_res_108359 = zs_lhs_108358 / zs_rhs_90489;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_108366 = ((double *) mem_114429.mem)[i_113435 * m_86512 + i_113428];
            
            // futhark/microgpt.fut:461:18-58
            
            double lifted_lambda_res_108367 = zs_lhs_108366 / zs_rhs_90528;
            
            ((double *) mem_114467)[i_113435 * m_86512 + i_113428] = lifted_lambda_res_108367;
            ((double *) mem_114470)[i_113435 * m_86512 + i_113428] = lifted_lambda_res_108359;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_114505, bytes_114428, "mem_114505")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113444 = 0; i_113444 < n_86511; i_113444++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113440 = 0; i_113440 < m_86512; i_113440++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_89779 = ((double *) w_mem_114423.mem)[i_113444 * m_86512 + i_113440];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_89780 = ((double *) mem_114470)[i_113444 * m_86512 + i_113440];
            
            // futhark/microgpt.fut:463:21-34
            
            double zs_lhs_89781 = lt_r_86518 * zt_rhs_89780;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_89782 = ((double *) mem_114467)[i_113444 * m_86512 + i_113440];
            
            // futhark/microgpt.fut:463:51-57
            
            double zp_lhs_89783 = fpow64(ztzt_lhs_89782, 0.5);
            
            // futhark/microgpt.fut:463:59-71
            
            double zs_rhs_89784 = 1.0e-8 + zp_lhs_89783;
            
            // futhark/microgpt.fut:463:35-71
            
            double zm_rhs_89785 = zs_lhs_89781 / zs_rhs_89784;
            
            // futhark/microgpt.fut:463:13-71
            
            double lifted_lambda_res_89786 = zm_lhs_89779 - zm_rhs_89785;
            
            ((double *) mem_114505.mem)[i_113444 * m_86512 + i_113440] = lifted_lambda_res_89786;
        }
    }
    if (memblock_set(ctx, &mem_out_116437, &mem_114505, "mem_114505") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116438, &mem_114432, "mem_114432") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116439, &mem_114429, "mem_114429") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116791, &mem_out_116437, "mem_out_116437") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116792, &mem_out_116438, "mem_out_116438") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116793, &mem_out_116439, "mem_out_116439") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_114467);
        free(mem_114470);
        if (memblock_unref(ctx, &mem_114505, "mem_114505") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_114432, "mem_114432") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_114429, "mem_114429") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116439, "mem_out_116439") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116438, "mem_out_116438") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116437, "mem_out_116437") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_116796, double *out_prim_out_116797, struct memblock wdown_mem_114423, struct memblock wkey_mem_114424, struct memblock wout_mem_114425, struct memblock wpe_mem_114426, struct memblock wqry_mem_114427, struct memblock wte_mem_114428, struct memblock wup_mem_114429, struct memblock wval_mem_114430, struct memblock wvoc_mem_114431, struct memblock tokens_mem_114432, struct memblock target_mem_114433, struct memblock mask_mem_114434)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_114435_cached_sizze_116798 = 0;
    unsigned char *mem_114435 = NULL;
    int64_t mem_114440_cached_sizze_116799 = 0;
    unsigned char *mem_114440 = NULL;
    int64_t mem_114451_cached_sizze_116800 = 0;
    unsigned char *mem_114451 = NULL;
    int64_t mem_114456_cached_sizze_116801 = 0;
    unsigned char *mem_114456 = NULL;
    int64_t mem_114463_cached_sizze_116802 = 0;
    unsigned char *mem_114463 = NULL;
    int64_t mem_114474_cached_sizze_116803 = 0;
    unsigned char *mem_114474 = NULL;
    int64_t mem_114479_cached_sizze_116804 = 0;
    unsigned char *mem_114479 = NULL;
    int64_t mem_114486_cached_sizze_116805 = 0;
    unsigned char *mem_114486 = NULL;
    int64_t mem_114497_cached_sizze_116806 = 0;
    unsigned char *mem_114497 = NULL;
    int64_t mem_114498_cached_sizze_116807 = 0;
    unsigned char *mem_114498 = NULL;
    int64_t mem_114499_cached_sizze_116808 = 0;
    unsigned char *mem_114499 = NULL;
    int64_t mem_114512_cached_sizze_116809 = 0;
    unsigned char *mem_114512 = NULL;
    int64_t mem_114513_cached_sizze_116810 = 0;
    unsigned char *mem_114513 = NULL;
    int64_t mem_114514_cached_sizze_116811 = 0;
    unsigned char *mem_114514 = NULL;
    int64_t mem_114545_cached_sizze_116812 = 0;
    unsigned char *mem_114545 = NULL;
    int64_t mem_114546_cached_sizze_116813 = 0;
    unsigned char *mem_114546 = NULL;
    int64_t mem_114547_cached_sizze_116814 = 0;
    unsigned char *mem_114547 = NULL;
    int64_t mem_114563_cached_sizze_116815 = 0;
    unsigned char *mem_114563 = NULL;
    int64_t mem_114564_cached_sizze_116816 = 0;
    unsigned char *mem_114564 = NULL;
    int64_t mem_114565_cached_sizze_116817 = 0;
    unsigned char *mem_114565 = NULL;
    int64_t mem_114578_cached_sizze_116818 = 0;
    unsigned char *mem_114578 = NULL;
    int64_t mem_114579_cached_sizze_116819 = 0;
    unsigned char *mem_114579 = NULL;
    int64_t mem_114580_cached_sizze_116820 = 0;
    unsigned char *mem_114580 = NULL;
    int64_t mem_114626_cached_sizze_116821 = 0;
    unsigned char *mem_114626 = NULL;
    int64_t mem_114632_cached_sizze_116822 = 0;
    unsigned char *mem_114632 = NULL;
    int64_t mem_114637_cached_sizze_116823 = 0;
    unsigned char *mem_114637 = NULL;
    int64_t mem_114648_cached_sizze_116824 = 0;
    unsigned char *mem_114648 = NULL;
    int64_t mem_114653_cached_sizze_116825 = 0;
    unsigned char *mem_114653 = NULL;
    int64_t mem_114664_cached_sizze_116826 = 0;
    unsigned char *mem_114664 = NULL;
    int64_t mem_114669_cached_sizze_116827 = 0;
    unsigned char *mem_114669 = NULL;
    int64_t mem_114676_cached_sizze_116828 = 0;
    unsigned char *mem_114676 = NULL;
    int64_t mem_114683_cached_sizze_116829 = 0;
    unsigned char *mem_114683 = NULL;
    int64_t mem_114694_cached_sizze_116830 = 0;
    unsigned char *mem_114694 = NULL;
    int64_t mem_114699_cached_sizze_116831 = 0;
    unsigned char *mem_114699 = NULL;
    int64_t mem_114710_cached_sizze_116832 = 0;
    unsigned char *mem_114710 = NULL;
    int64_t mem_114715_cached_sizze_116833 = 0;
    unsigned char *mem_114715 = NULL;
    int64_t mem_114731_cached_sizze_116834 = 0;
    unsigned char *mem_114731 = NULL;
    int64_t mem_114736_cached_sizze_116835 = 0;
    unsigned char *mem_114736 = NULL;
    int64_t mem_114747_cached_sizze_116836 = 0;
    unsigned char *mem_114747 = NULL;
    int64_t mem_114752_cached_sizze_116837 = 0;
    unsigned char *mem_114752 = NULL;
    int64_t mem_114763_cached_sizze_116838 = 0;
    unsigned char *mem_114763 = NULL;
    int64_t mem_114768_cached_sizze_116839 = 0;
    unsigned char *mem_114768 = NULL;
    int64_t mem_114779_cached_sizze_116840 = 0;
    unsigned char *mem_114779 = NULL;
    int64_t mem_114784_cached_sizze_116841 = 0;
    unsigned char *mem_114784 = NULL;
    int64_t mem_114791_cached_sizze_116842 = 0;
    unsigned char *mem_114791 = NULL;
    int64_t mem_114802_cached_sizze_116843 = 0;
    unsigned char *mem_114802 = NULL;
    int64_t mem_114807_cached_sizze_116844 = 0;
    unsigned char *mem_114807 = NULL;
    int64_t mem_114818_cached_sizze_116845 = 0;
    unsigned char *mem_114818 = NULL;
    int64_t mem_114823_cached_sizze_116846 = 0;
    unsigned char *mem_114823 = NULL;
    int64_t mem_114834_cached_sizze_116847 = 0;
    unsigned char *mem_114834 = NULL;
    int64_t mem_114839_cached_sizze_116848 = 0;
    unsigned char *mem_114839 = NULL;
    int64_t mem_114850_cached_sizze_116849 = 0;
    unsigned char *mem_114850 = NULL;
    int64_t mem_114855_cached_sizze_116850 = 0;
    unsigned char *mem_114855 = NULL;
    int64_t mem_114866_cached_sizze_116851 = 0;
    unsigned char *mem_114866 = NULL;
    int64_t mem_114871_cached_sizze_116852 = 0;
    unsigned char *mem_114871 = NULL;
    int64_t mem_114886_cached_sizze_116853 = 0;
    unsigned char *mem_114886 = NULL;
    int64_t mem_114893_cached_sizze_116854 = 0;
    unsigned char *mem_114893 = NULL;
    struct memblock mem_114882;
    
    mem_114882.references = NULL;
    
    struct memblock mem_out_116437;
    
    mem_out_116437.references = NULL;
    
    struct memblock mem_114414 = ctx->constants->mem_114414;
    struct memblock mem_114415 = ctx->constants->mem_114415;
    struct memblock mem_114416 = ctx->constants->mem_114416;
    struct memblock mem_114417 = ctx->constants->mem_114417;
    struct memblock mem_114418 = ctx->constants->mem_114418;
    struct memblock mem_114419 = ctx->constants->mem_114419;
    struct memblock mem_114420 = ctx->constants->mem_114420;
    struct memblock mem_114421 = ctx->constants->mem_114421;
    struct memblock mem_114422 = ctx->constants->mem_114422;
    double prim_out_116438;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_114435_cached_sizze_116798 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114435, &mem_114435_cached_sizze_116798, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114440_cached_sizze_116799 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114440, &mem_114440_cached_sizze_116799, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113416 = 0; i_113416 < (int64_t) 16; i_113416++) {
        // futhark/microgpt.fut:445:41-50
        
        int64_t tmp_101721 = ((int64_t *) tokens_mem_114432.mem)[i_113416];
        
        // futhark/microgpt.fut:445:37-51
        
        bool x_101722 = sle64((int64_t) 0, tmp_101721);
        
        // futhark/microgpt.fut:445:37-51
        
        bool y_101723 = slt64(tmp_101721, (int64_t) 27);
        
        // futhark/microgpt.fut:445:37-51
        
        bool bounds_check_101724 = x_101722 && y_101723;
        
        // futhark/microgpt.fut:445:37-51
        
        bool index_certs_101725;
        
        if (!bounds_check_101724) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_101721, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:445:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:445:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113412 = 0; i_113412 < (int64_t) 16; i_113412++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_101732 = ((double *) wte_mem_114428.mem)[tmp_101721 * (int64_t) 16 + i_113412];
            
            ((double *) mem_114440)[i_113412] = lifted_lambda_res_101732;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114435, i_113416 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114440, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114451_cached_sizze_116800 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114451, &mem_114451_cached_sizze_116800, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114456_cached_sizze_116801 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114456, &mem_114456_cached_sizze_116801, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114463_cached_sizze_116802 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114463, &mem_114463_cached_sizze_116802, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113428 = 0; i_113428 < (int64_t) 16; i_113428++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_101758;
        double r_101760 = 0.0;
        
        for (int64_t i_101759 = 0; i_101759 < (int64_t) 16; i_101759++) {
            // futhark/microgpt.fut:71:46-49
            
            double zp_lhs_101761 = ((double *) wpe_mem_114426.mem)[i_113428 * (int64_t) 16 + i_101759];
            
            // futhark/microgpt.fut:71:46-49
            
            double zp_rhs_101762 = ((double *) mem_114435)[i_113428 * (int64_t) 16 + i_101759];
            
            // futhark/microgpt.fut:203:76-116
            
            double zp_res_101763 = zp_lhs_101761 + zp_rhs_101762;
            
            // futhark/microgpt.fut:203:94-163
            
            double zt_res_101764 = zp_res_101763 * zp_res_101763;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_101765 = r_101760 + zt_res_101764;
            double r_tmp_116442 = zp_res_101765;
            
            r_101760 = r_tmp_116442;
        }
        defunc_0_lifted_lambda_res_101758 = r_101760;
        // futhark/microgpt.fut:203:54-182
        
        double zs_res_101766 = defunc_0_lifted_lambda_res_101758 / 16.0;
        
        // futhark/microgpt.fut:204:24-55
        
        double zp_res_101767 = 1.0e-5 + zs_res_101766;
        
        // futhark/microgpt.fut:204:16-55
        
        double sqrt_res_101768 = futrts_sqrt64(zp_res_101767);
        
        // futhark/microgpt.fut:205:85-96
        
        double zs_res_101769 = 1.0 / sqrt_res_101768;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113420 = 0; i_113420 < (int64_t) 16; i_113420++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_101776 = ((double *) wpe_mem_114426.mem)[i_113428 * (int64_t) 16 + i_113420];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_101777 = ((double *) mem_114435)[i_113428 * (int64_t) 16 + i_113420];
            
            // futhark/microgpt.fut:205:38-78
            
            double zp_res_101778 = zp_lhs_101776 + zp_rhs_101777;
            
            // futhark/microgpt.fut:205:56-96
            
            double zt_res_101779 = zs_res_101769 * zp_res_101778;
            
            ((double *) mem_114456)[i_113420] = zt_res_101779;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113424 = 0; i_113424 < (int64_t) 16; i_113424++) {
            // futhark/microgpt.fut:206:4-14
            
            double lifted_lambda_res_101787 = ((double *) mem_114456)[i_113424];
            
            ((double *) mem_114463)[i_113424] = lifted_lambda_res_101787;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114451, i_113428 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114463, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114474_cached_sizze_116803 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114474, &mem_114474_cached_sizze_116803, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114479_cached_sizze_116804 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114479, &mem_114479_cached_sizze_116804, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114486_cached_sizze_116805 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114486, &mem_114486_cached_sizze_116805, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113440 = 0; i_113440 < (int64_t) 16; i_113440++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_101796;
        double r_101798 = 0.0;
        
        for (int64_t i_101797 = 0; i_101797 < (int64_t) 16; i_101797++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_101799 = ((double *) mem_114451)[i_113440 * (int64_t) 16 + i_101797];
            
            // futhark/microgpt.fut:207:78-115
            
            double zt_res_101800 = zt_lhs_101799 * zt_lhs_101799;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_101801 = r_101798 + zt_res_101800;
            double r_tmp_116446 = zp_res_101801;
            
            r_101798 = r_tmp_116446;
        }
        defunc_0_lifted_lambda_res_101796 = r_101798;
        // futhark/microgpt.fut:207:57-133
        
        double zs_res_101802 = defunc_0_lifted_lambda_res_101796 / 16.0;
        
        // futhark/microgpt.fut:208:24-55
        
        double zp_res_101803 = 1.0e-5 + zs_res_101802;
        
        // futhark/microgpt.fut:208:16-55
        
        double sqrt_res_101804 = futrts_sqrt64(zp_res_101803);
        
        // futhark/microgpt.fut:209:59-70
        
        double zs_res_101805 = 1.0 / sqrt_res_101804;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113432 = 0; i_113432 < (int64_t) 16; i_113432++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_101812 = ((double *) mem_114451)[i_113440 * (int64_t) 16 + i_113432];
            
            // futhark/microgpt.fut:209:37-70
            
            double zt_res_101813 = zs_res_101805 * zt_lhs_101812;
            
            ((double *) mem_114479)[i_113432] = zt_res_101813;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113436 = 0; i_113436 < (int64_t) 16; i_113436++) {
            // futhark/microgpt.fut:210:4-14
            
            double lifted_lambda_res_101821 = ((double *) mem_114479)[i_113436];
            
            ((double *) mem_114486)[i_113436] = lifted_lambda_res_101821;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114474, i_113440 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114486, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114497_cached_sizze_116806 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114497, &mem_114497_cached_sizze_116806, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114498_cached_sizze_116807 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114498, &mem_114498_cached_sizze_116807, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114499_cached_sizze_116808 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114499, &mem_114499_cached_sizze_116808, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114512_cached_sizze_116809 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114512, &mem_114512_cached_sizze_116809, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114513_cached_sizze_116810 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114513, &mem_114513_cached_sizze_116810, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114514_cached_sizze_116811 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114514, &mem_114514_cached_sizze_116811, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113458 = 0; i_113458 < (int64_t) 16; i_113458++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113448 = 0; i_113448 < (int64_t) 16; i_113448++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_108540;
            double r_108542 = 0.0;
            
            for (int64_t i_108541 = 0; i_108541 < (int64_t) 16; i_108541++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_108543 = ((double *) wqry_mem_114427.mem)[i_113448 * (int64_t) 16 + i_108541];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_108544 = ((double *) mem_114474)[i_113458 * (int64_t) 16 + i_108541];
                
                // futhark/microgpt.fut:211:66-105
                
                double zt_res_108545 = zt_lhs_108543 * zt_rhs_108544;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_108546 = r_108542 + zt_res_108545;
                double r_tmp_116455 = zp_res_108546;
                
                r_108542 = r_tmp_116455;
            }
            defunc_0_lifted_lambda_res_108540 = r_108542;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_108553;
            double r_108555 = 0.0;
            
            for (int64_t i_108554 = 0; i_108554 < (int64_t) 16; i_108554++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_108556 = ((double *) wkey_mem_114424.mem)[i_113448 * (int64_t) 16 + i_108554];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_108557 = ((double *) mem_114474)[i_113458 * (int64_t) 16 + i_108554];
                
                // futhark/microgpt.fut:212:66-105
                
                double zt_res_108558 = zt_lhs_108556 * zt_rhs_108557;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_108559 = r_108555 + zt_res_108558;
                double r_tmp_116456 = zp_res_108559;
                
                r_108555 = r_tmp_116456;
            }
            defunc_0_lifted_lambda_res_108553 = r_108555;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_108569;
            double r_108571 = 0.0;
            
            for (int64_t i_108570 = 0; i_108570 < (int64_t) 16; i_108570++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_108572 = ((double *) wval_mem_114430.mem)[i_113448 * (int64_t) 16 + i_108570];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_108573 = ((double *) mem_114474)[i_113458 * (int64_t) 16 + i_108570];
                
                // futhark/microgpt.fut:213:66-105
                
                double zt_res_108574 = zt_lhs_108572 * zt_rhs_108573;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_108575 = r_108571 + zt_res_108574;
                double r_tmp_116457 = zp_res_108575;
                
                r_108571 = r_tmp_116457;
            }
            defunc_0_lifted_lambda_res_108569 = r_108571;
            ((double *) mem_114512)[i_113448] = defunc_0_lifted_lambda_res_108569;
            ((double *) mem_114513)[i_113448] = defunc_0_lifted_lambda_res_108553;
            ((double *) mem_114514)[i_113448] = defunc_0_lifted_lambda_res_108540;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114497, i_113458 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114512, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114498, i_113458 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114513, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114499, i_113458 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114514, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114545_cached_sizze_116812 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114545, &mem_114545_cached_sizze_116812, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114546_cached_sizze_116813 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114546, &mem_114546_cached_sizze_116813, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114547_cached_sizze_116814 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114547, &mem_114547_cached_sizze_116814, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114563_cached_sizze_116815 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114563, &mem_114563_cached_sizze_116815, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114564_cached_sizze_116816 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114564, &mem_114564_cached_sizze_116816, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114565_cached_sizze_116817 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114565, &mem_114565_cached_sizze_116817, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114578_cached_sizze_116818 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_114578, &mem_114578_cached_sizze_116818, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114579_cached_sizze_116819 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_114579, &mem_114579_cached_sizze_116819, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114580_cached_sizze_116820 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_114580, &mem_114580_cached_sizze_116820, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113488 = 0; i_113488 < (int64_t) 4; i_113488++) {
        // futhark/microgpt.fut:214:69-72
        
        int64_t zp_lhs_108416 = mul64((int64_t) 4, i_113488);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113478 = 0; i_113478 < (int64_t) 16; i_113478++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113468 = 0; i_113468 < (int64_t) 4; i_113468++) {
                // futhark/microgpt.fut:214:74-81
                
                int64_t tmp_108733 = add64(zp_lhs_108416, i_113468);
                
                // futhark/microgpt.fut:214:51-83
                
                bool x_108734 = sle64((int64_t) 0, tmp_108733);
                
                // futhark/microgpt.fut:214:51-83
                
                bool y_108735 = slt64(tmp_108733, (int64_t) 16);
                
                // futhark/microgpt.fut:214:51-83
                
                bool bounds_check_108736 = x_108734 && y_108735;
                
                // futhark/microgpt.fut:214:51-83
                
                bool index_certs_108737;
                
                if (!bounds_check_108736) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_108733, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:214:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:214:15-84\n   #9  futhark/microgpt.fut:446:7-76\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_108738 = ((double *) mem_114499)[i_113478 * (int64_t) 16 + tmp_108733];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_108746 = ((double *) mem_114498)[i_113478 * (int64_t) 16 + tmp_108733];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_108757 = ((double *) mem_114497)[i_113478 * (int64_t) 16 + tmp_108733];
                
                ((double *) mem_114578)[i_113468] = lifted_lambda_res_108757;
                ((double *) mem_114579)[i_113468] = lifted_lambda_res_108746;
                ((double *) mem_114580)[i_113468] = lifted_lambda_res_108738;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114563, i_113478 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114578, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114564, i_113478 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114579, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114565, i_113478 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114580, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_114545, i_113488 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_114563, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_114546, i_113488 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_114564, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_114547, i_113488 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_114565, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114626_cached_sizze_116821 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114626, &mem_114626_cached_sizze_116821, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114632_cached_sizze_116822 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114632, &mem_114632_cached_sizze_116822, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114637_cached_sizze_116823 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114637, &mem_114637_cached_sizze_116823, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114648_cached_sizze_116824 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114648, &mem_114648_cached_sizze_116824, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114653_cached_sizze_116825 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114653, &mem_114653_cached_sizze_116825, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114664_cached_sizze_116826 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114664, &mem_114664_cached_sizze_116826, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114669_cached_sizze_116827 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114669, &mem_114669_cached_sizze_116827, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114676_cached_sizze_116828 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114676, &mem_114676_cached_sizze_116828, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114683_cached_sizze_116829 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114683, &mem_114683_cached_sizze_116829, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114694_cached_sizze_116830 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114694, &mem_114694_cached_sizze_116830, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114699_cached_sizze_116831 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_114699, &mem_114699_cached_sizze_116831, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114710_cached_sizze_116832 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114710, &mem_114710_cached_sizze_116832, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114715_cached_sizze_116833 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_114715, &mem_114715_cached_sizze_116833, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113544 = 0; i_113544 < (int64_t) 4; i_113544++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113498 = 0; i_113498 < (int64_t) 16; i_113498++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113494 = 0; i_113494 < (int64_t) 16; i_113494++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_101966;
                double r_101968 = 0.0;
                
                for (int64_t i_101967 = 0; i_101967 < (int64_t) 4; i_101967++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_101969 = ((double *) mem_114547)[i_113544 * (int64_t) 64 + i_113498 * (int64_t) 4 + i_101967];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_101970 = ((double *) mem_114546)[i_113544 * (int64_t) 64 + i_113494 * (int64_t) 4 + i_101967];
                    
                    // futhark/microgpt.fut:217:113-164
                    
                    double zt_res_101971 = zt_lhs_101969 * zt_rhs_101970;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_101972 = r_101968 + zt_res_101971;
                    double r_tmp_116470 = zp_res_101972;
                    
                    r_101968 = r_tmp_116470;
                }
                defunc_0_lifted_lambda_res_101966 = r_101968;
                ((double *) mem_114637)[i_113494] = defunc_0_lifted_lambda_res_101966;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114632, i_113498 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114637, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113506 = 0; i_113506 < (int64_t) 16; i_113506++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113502 = 0; i_113502 < (int64_t) 16; i_113502++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_101987 = ((double *) mem_114632)[i_113506 * (int64_t) 16 + i_113502];
                
                // futhark/microgpt.fut:218:47-78
                
                double zs_res_101988 = zs_lhs_101987 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_101989 = ((double *) mask_mem_114434.mem)[i_113506 * (int64_t) 16 + i_113502];
                
                // futhark/microgpt.fut:218:65-102
                
                double zp_res_101990 = zs_res_101988 + zp_rhs_101989;
                
                ((double *) mem_114653)[i_113502] = zp_res_101990;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114648, i_113506 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114653, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113524 = 0; i_113524 < (int64_t) 16; i_113524++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_108860;
            double redout_113508 = -INFINITY;
            
            for (int64_t i_113509 = 0; i_113509 < (int64_t) 16; i_113509++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_108784 = ((double *) mem_114648)[i_113524 * (int64_t) 16 + i_113509];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_102011 = fmax64(lifted_lambda_res_108784, redout_113508);
                double redout_tmp_116474 = max_res_102011;
                
                redout_113508 = redout_tmp_116474;
            }
            defunc_0_reduce_res_108860 = redout_113508;
            // futhark/microgpt.fut:220:67-76
            
            double neg_res_102012 = -defunc_0_reduce_res_108860;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113512 = 0; i_113512 < (int64_t) 16; i_113512++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_102019 = ((double *) mem_114648)[i_113524 * (int64_t) 16 + i_113512];
                
                // futhark/microgpt.fut:220:44-76
                
                double zp_res_102020 = neg_res_102012 + zp_lhs_102019;
                
                // futhark/microgpt.fut:220:37-76
                
                double exp_res_102021 = futrts_exp64(zp_res_102020);
                
                ((double *) mem_114669)[i_113512] = exp_res_102021;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_102023;
            double r_102025 = 0.0;
            
            for (int64_t i_102024 = 0; i_102024 < (int64_t) 16; i_102024++) {
                // futhark/microgpt.fut:221:36-46
                
                double lifted_lambda_res_102026 = ((double *) mem_114669)[i_102024];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_102027 = r_102025 + lifted_lambda_res_102026;
                double r_tmp_116476 = zp_res_102027;
                
                r_102025 = r_tmp_116476;
            }
            defunc_0_lifted_lambda_res_102023 = r_102025;
            // futhark/microgpt.fut:222:53-64
            
            double zs_res_102028 = 1.0 / defunc_0_lifted_lambda_res_102023;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113516 = 0; i_113516 < (int64_t) 16; i_113516++) {
                // futhark/microgpt.fut:222:37-47
                
                double zt_lhs_102035 = ((double *) mem_114669)[i_113516];
                
                // futhark/microgpt.fut:222:37-64
                
                double zt_res_102036 = zs_res_102028 * zt_lhs_102035;
                
                ((double *) mem_114676)[i_113516] = zt_res_102036;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113520 = 0; i_113520 < (int64_t) 16; i_113520++) {
                // futhark/microgpt.fut:223:4-14
                
                double lifted_lambda_res_102044 = ((double *) mem_114676)[i_113520];
                
                ((double *) mem_114683)[i_113520] = lifted_lambda_res_102044;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114664, i_113524 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114683, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113532 = 0; i_113532 < (int64_t) 16; i_113532++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113528 = 0; i_113528 < (int64_t) 4; i_113528++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_102059;
                double r_102061 = 0.0;
                
                for (int64_t i_102060 = 0; i_102060 < (int64_t) 16; i_102060++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_102062 = ((double *) mem_114664)[i_113532 * (int64_t) 16 + i_102060];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_102063 = ((double *) mem_114545)[i_113544 * (int64_t) 64 + i_102060 * (int64_t) 4 + i_113528];
                    
                    // futhark/microgpt.fut:224:66-111
                    
                    double zt_res_102064 = zt_lhs_102062 * zt_rhs_102063;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_102065 = r_102061 + zt_res_102064;
                    double r_tmp_116481 = zp_res_102065;
                    
                    r_102061 = r_tmp_116481;
                }
                defunc_0_lifted_lambda_res_102059 = r_102061;
                ((double *) mem_114699)[i_113528] = defunc_0_lifted_lambda_res_102059;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114694, i_113532 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114699, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113540 = 0; i_113540 < (int64_t) 16; i_113540++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113536 = 0; i_113536 < (int64_t) 4; i_113536++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_102080 = ((double *) mem_114694)[i_113540 * (int64_t) 4 + i_113536];
                
                ((double *) mem_114715)[i_113536] = lifted_lambda_res_102080;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114710, i_113540 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114715, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_114626, i_113544 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_114710, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114731_cached_sizze_116834 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114731, &mem_114731_cached_sizze_116834, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114736_cached_sizze_116835 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114736, &mem_114736_cached_sizze_116835, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113552 = 0; i_113552 < (int64_t) 16; i_113552++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113548 = 0; i_113548 < (int64_t) 16; i_113548++) {
            // futhark/microgpt.fut:226:54-57
            
            int64_t tmp_102092 = sdiv64(i_113548, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-59
            
            bool x_102093 = sle64((int64_t) 0, tmp_102092);
            
            // futhark/microgpt.fut:226:44-59
            
            bool y_102094 = slt64(tmp_102092, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-59
            
            bool bounds_check_102095 = x_102093 && y_102094;
            
            // futhark/microgpt.fut:226:44-59
            
            bool index_certs_102096;
            
            if (!bounds_check_102095) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_102092, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:226:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:226:15-80\n   #6  futhark/microgpt.fut:446:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:226:74-77
            
            int64_t tmp_102097 = smod64(i_113548, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-79
            
            bool x_102098 = sle64((int64_t) 0, tmp_102097);
            
            // futhark/microgpt.fut:226:44-79
            
            bool y_102099 = slt64(tmp_102097, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-79
            
            bool bounds_check_102100 = x_102098 && y_102099;
            
            // futhark/microgpt.fut:226:44-79
            
            bool index_certs_102101;
            
            if (!bounds_check_102100) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_102097, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:226:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:226:15-80\n   #6  futhark/microgpt.fut:446:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_102102 = ((double *) mem_114626)[tmp_102092 * (int64_t) 64 + i_113552 * (int64_t) 4 + tmp_102097];
            
            ((double *) mem_114736)[i_113548] = lifted_lambda_res_102102;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114731, i_113552 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114736, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114747_cached_sizze_116836 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114747, &mem_114747_cached_sizze_116836, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114752_cached_sizze_116837 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114752, &mem_114752_cached_sizze_116837, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113560 = 0; i_113560 < (int64_t) 16; i_113560++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113556 = 0; i_113556 < (int64_t) 16; i_113556++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_102117;
            double r_102119 = 0.0;
            
            for (int64_t i_102118 = 0; i_102118 < (int64_t) 16; i_102118++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_102120 = ((double *) wout_mem_114425.mem)[i_113556 * (int64_t) 16 + i_102118];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_102121 = ((double *) mem_114731)[i_113560 * (int64_t) 16 + i_102118];
                
                // futhark/microgpt.fut:227:67-106
                
                double zt_res_102122 = zt_lhs_102120 * zt_rhs_102121;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_102123 = r_102119 + zt_res_102122;
                double r_tmp_116488 = zp_res_102123;
                
                r_102119 = r_tmp_116488;
            }
            defunc_0_lifted_lambda_res_102117 = r_102119;
            ((double *) mem_114752)[i_113556] = defunc_0_lifted_lambda_res_102117;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114747, i_113560 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114752, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114763_cached_sizze_116838 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114763, &mem_114763_cached_sizze_116838, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114768_cached_sizze_116839 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114768, &mem_114768_cached_sizze_116839, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113568 = 0; i_113568 < (int64_t) 16; i_113568++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113564 = 0; i_113564 < (int64_t) 16; i_113564++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_102138 = ((double *) mem_114747)[i_113568 * (int64_t) 16 + i_113564];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_102139 = ((double *) mem_114451)[i_113568 * (int64_t) 16 + i_113564];
            
            // futhark/microgpt.fut:228:46-84
            
            double zp_res_102140 = zp_lhs_102138 + zp_rhs_102139;
            
            ((double *) mem_114768)[i_113564] = zp_res_102140;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114763, i_113568 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114768, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114779_cached_sizze_116840 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114779, &mem_114779_cached_sizze_116840, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114784_cached_sizze_116841 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114784, &mem_114784_cached_sizze_116841, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114791_cached_sizze_116842 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114791, &mem_114791_cached_sizze_116842, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113580 = 0; i_113580 < (int64_t) 16; i_113580++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_102149;
        double r_102151 = 0.0;
        
        for (int64_t i_102150 = 0; i_102150 < (int64_t) 16; i_102150++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_102152 = ((double *) mem_114763)[i_113580 * (int64_t) 16 + i_102150];
            
            // futhark/microgpt.fut:229:79-118
            
            double zt_res_102153 = zt_lhs_102152 * zt_lhs_102152;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_102154 = r_102151 + zt_res_102153;
            double r_tmp_116492 = zp_res_102154;
            
            r_102151 = r_tmp_116492;
        }
        defunc_0_lifted_lambda_res_102149 = r_102151;
        // futhark/microgpt.fut:229:58-136
        
        double zs_res_102155 = defunc_0_lifted_lambda_res_102149 / 16.0;
        
        // futhark/microgpt.fut:230:24-55
        
        double zp_res_102156 = 1.0e-5 + zs_res_102155;
        
        // futhark/microgpt.fut:230:16-55
        
        double sqrt_res_102157 = futrts_sqrt64(zp_res_102156);
        
        // futhark/microgpt.fut:231:60-71
        
        double zs_res_102158 = 1.0 / sqrt_res_102157;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113572 = 0; i_113572 < (int64_t) 16; i_113572++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_102165 = ((double *) mem_114763)[i_113580 * (int64_t) 16 + i_113572];
            
            // futhark/microgpt.fut:231:37-71
            
            double zt_res_102166 = zs_res_102158 * zt_lhs_102165;
            
            ((double *) mem_114784)[i_113572] = zt_res_102166;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113576 = 0; i_113576 < (int64_t) 16; i_113576++) {
            // futhark/microgpt.fut:232:4-14
            
            double lifted_lambda_res_102174 = ((double *) mem_114784)[i_113576];
            
            ((double *) mem_114791)[i_113576] = lifted_lambda_res_102174;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114779, i_113580 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114791, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114802_cached_sizze_116843 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_114802, &mem_114802_cached_sizze_116843, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114807_cached_sizze_116844 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114807, &mem_114807_cached_sizze_116844, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113588 = 0; i_113588 < (int64_t) 16; i_113588++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113584 = 0; i_113584 < (int64_t) 64; i_113584++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_102190;
            double r_102192 = 0.0;
            
            for (int64_t i_102191 = 0; i_102191 < (int64_t) 16; i_102191++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_102193 = ((double *) wup_mem_114429.mem)[i_113584 * (int64_t) 16 + i_102191];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_102194 = ((double *) mem_114779)[i_113588 * (int64_t) 16 + i_102191];
                
                // futhark/microgpt.fut:233:67-106
                
                double zt_res_102195 = zt_lhs_102193 * zt_rhs_102194;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_102196 = r_102192 + zt_res_102195;
                double r_tmp_116497 = zp_res_102196;
                
                r_102192 = r_tmp_116497;
            }
            defunc_0_lifted_lambda_res_102190 = r_102192;
            ((double *) mem_114807)[i_113584] = defunc_0_lifted_lambda_res_102190;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114802, i_113588 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114807, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114818_cached_sizze_116845 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_114818, &mem_114818_cached_sizze_116845, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114823_cached_sizze_116846 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114823, &mem_114823_cached_sizze_116846, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113596 = 0; i_113596 < (int64_t) 16; i_113596++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113592 = 0; i_113592 < (int64_t) 64; i_113592++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_102211 = ((double *) mem_114802)[i_113596 * (int64_t) 64 + i_113592];
            
            // futhark/microgpt.fut:234:45-73
            
            double max_res_102212 = fmax64(0.0, max_arg0_102211);
            
            ((double *) mem_114823)[i_113592] = max_res_102212;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114818, i_113596 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114823, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114834_cached_sizze_116847 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114834, &mem_114834_cached_sizze_116847, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114839_cached_sizze_116848 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114839, &mem_114839_cached_sizze_116848, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113604 = 0; i_113604 < (int64_t) 16; i_113604++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113600 = 0; i_113600 < (int64_t) 16; i_113600++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_102227;
            double r_102229 = 0.0;
            
            for (int64_t i_102228 = 0; i_102228 < (int64_t) 64; i_102228++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_102230 = ((double *) wdown_mem_114423.mem)[i_113600 * (int64_t) 64 + i_102228];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_102231 = ((double *) mem_114818)[i_113604 * (int64_t) 64 + i_102228];
                
                // futhark/microgpt.fut:235:67-108
                
                double zt_res_102232 = zt_lhs_102230 * zt_rhs_102231;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_102233 = r_102229 + zt_res_102232;
                double r_tmp_116502 = zp_res_102233;
                
                r_102229 = r_tmp_116502;
            }
            defunc_0_lifted_lambda_res_102227 = r_102229;
            ((double *) mem_114839)[i_113600] = defunc_0_lifted_lambda_res_102227;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114834, i_113604 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114839, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114850_cached_sizze_116849 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114850, &mem_114850_cached_sizze_116849, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114855_cached_sizze_116850 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114855, &mem_114855_cached_sizze_116850, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113612 = 0; i_113612 < (int64_t) 16; i_113612++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113608 = 0; i_113608 < (int64_t) 16; i_113608++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_102248 = ((double *) mem_114834)[i_113612 * (int64_t) 16 + i_113608];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_102249 = ((double *) mem_114763)[i_113612 * (int64_t) 16 + i_113608];
            
            // futhark/microgpt.fut:236:46-85
            
            double zp_res_102250 = zp_lhs_102248 + zp_rhs_102249;
            
            ((double *) mem_114855)[i_113608] = zp_res_102250;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114850, i_113612 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114855, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114866_cached_sizze_116851 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_114866, &mem_114866_cached_sizze_116851, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114871_cached_sizze_116852 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_114871, &mem_114871_cached_sizze_116852, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113620 = 0; i_113620 < (int64_t) 16; i_113620++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113616 = 0; i_113616 < (int64_t) 27; i_113616++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_102266;
            double r_102268 = 0.0;
            
            for (int64_t i_102267 = 0; i_102267 < (int64_t) 16; i_102267++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_102269 = ((double *) wvoc_mem_114431.mem)[i_113616 * (int64_t) 16 + i_102267];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_102270 = ((double *) mem_114850)[i_113620 * (int64_t) 16 + i_102267];
                
                // futhark/microgpt.fut:237:67-107
                
                double zt_res_102271 = zt_lhs_102269 * zt_rhs_102270;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_102272 = r_102268 + zt_res_102271;
                double r_tmp_116507 = zp_res_102272;
                
                r_102268 = r_tmp_116507;
            }
            defunc_0_lifted_lambda_res_102266 = r_102268;
            ((double *) mem_114871)[i_113616] = defunc_0_lifted_lambda_res_102266;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114866, i_113620 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114871, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_114882, (int64_t) 128, "mem_114882")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114886_cached_sizze_116853 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_114886, &mem_114886_cached_sizze_116853, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114893_cached_sizze_116854 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_114893, &mem_114893_cached_sizze_116854, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113634 = 0; i_113634 < (int64_t) 16; i_113634++) {
        double x_108883;
        double redout_113622 = -INFINITY;
        
        for (int64_t i_113623 = 0; i_113623 < (int64_t) 27; i_113623++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_108830 = ((double *) mem_114866)[i_113634 * (int64_t) 27 + i_113623];
            
            // futhark/microgpt.fut:115:13-33
            
            double max_res_102296 = fmax64(lifted_lambda_res_108830, redout_113622);
            double redout_tmp_116509 = max_res_102296;
            
            redout_113622 = redout_tmp_116509;
        }
        x_108883 = redout_113622;
        // futhark/microgpt.fut:239:67-76
        
        double neg_res_102297 = -x_108883;
        
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_102281;
        double r_102283 = 0.0;
        
        for (int64_t i_102282 = 0; i_102282 < (int64_t) 27; i_102282++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113626 = 0; i_113626 < (int64_t) 27; i_113626++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_102304 = ((double *) mem_114866)[i_113634 * (int64_t) 27 + i_113626];
                
                // futhark/microgpt.fut:239:44-76
                
                double zp_res_102305 = neg_res_102297 + zp_lhs_102304;
                
                // futhark/microgpt.fut:239:37-76
                
                double exp_res_102306 = futrts_exp64(zp_res_102305);
                
                ((double *) mem_114886)[i_113626] = exp_res_102306;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_102308;
            double r_102310 = 0.0;
            
            for (int64_t i_102309 = 0; i_102309 < (int64_t) 27; i_102309++) {
                // futhark/microgpt.fut:240:36-46
                
                double lifted_lambda_res_102311 = ((double *) mem_114886)[i_102309];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_102312 = r_102310 + lifted_lambda_res_102311;
                double r_tmp_116512 = zp_res_102312;
                
                r_102310 = r_tmp_116512;
            }
            defunc_0_lifted_lambda_res_102308 = r_102310;
            // futhark/microgpt.fut:241:53-64
            
            double zs_res_102313 = 1.0 / defunc_0_lifted_lambda_res_102308;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113630 = 0; i_113630 < (int64_t) 27; i_113630++) {
                // futhark/microgpt.fut:241:37-47
                
                double zt_lhs_102320 = ((double *) mem_114886)[i_113630];
                
                // futhark/microgpt.fut:241:37-64
                
                double zt_res_102321 = zs_res_102313 * zt_lhs_102320;
                
                ((double *) mem_114893)[i_113630] = zt_res_102321;
            }
            // futhark/microgpt.fut:242:12-22
            
            double log_arg0_102323 = ((double *) mem_114893)[i_102282];
            
            // futhark/microgpt.fut:242:6-22
            
            double log_res_102324 = futrts_log64(log_arg0_102323);
            
            // futhark/microgpt.fut:71:46-49
            
            double zt_rhs_102325 = ((double *) target_mem_114433.mem)[i_113634 * (int64_t) 27 + i_102282];
            
            // futhark/microgpt.fut:242:6-48
            
            double zt_res_102326 = log_res_102324 * zt_rhs_102325;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_102327 = r_102283 + zt_res_102326;
            double r_tmp_116510 = zp_res_102327;
            
            r_102283 = r_tmp_116510;
        }
        defunc_0_lifted_lambda_res_102281 = r_102283;
        // futhark/microgpt.fut:238:37-242:54
        
        double neg_res_102328 = -defunc_0_lifted_lambda_res_102281;
        
        ((double *) mem_114882.mem)[i_113634] = neg_res_102328;
    }
    // futhark/microgpt.fut:71:13-49
    
    double defunc_0_lifted_lambda_res_102330;
    double r_102332 = 0.0;
    
    for (int64_t i_102331 = 0; i_102331 < (int64_t) 16; i_102331++) {
        // futhark/microgpt.fut:243:37-47
        
        double lifted_lambda_res_102333 = ((double *) mem_114882.mem)[i_102331];
        
        // futhark/microgpt.fut:71:40-49
        
        double zp_res_102334 = r_102332 + lifted_lambda_res_102333;
        double r_tmp_116514 = zp_res_102334;
        
        r_102332 = r_tmp_116514;
    }
    defunc_0_lifted_lambda_res_102330 = r_102332;
    // futhark/microgpt.fut:243:17-64
    
    double zs_res_102335 = defunc_0_lifted_lambda_res_102330 / 16.0;
    
    if (memblock_set(ctx, &mem_out_116437, &mem_114882, "mem_114882") != 0)
        return 1;
    prim_out_116438 = zs_res_102335;
    if (memblock_set(ctx, &*mem_out_p_116796, &mem_out_116437, "mem_out_116437") != 0)
        return 1;
    *out_prim_out_116797 = prim_out_116438;
    
  cleanup:
    {
        free(mem_114435);
        free(mem_114440);
        free(mem_114451);
        free(mem_114456);
        free(mem_114463);
        free(mem_114474);
        free(mem_114479);
        free(mem_114486);
        free(mem_114497);
        free(mem_114498);
        free(mem_114499);
        free(mem_114512);
        free(mem_114513);
        free(mem_114514);
        free(mem_114545);
        free(mem_114546);
        free(mem_114547);
        free(mem_114563);
        free(mem_114564);
        free(mem_114565);
        free(mem_114578);
        free(mem_114579);
        free(mem_114580);
        free(mem_114626);
        free(mem_114632);
        free(mem_114637);
        free(mem_114648);
        free(mem_114653);
        free(mem_114664);
        free(mem_114669);
        free(mem_114676);
        free(mem_114683);
        free(mem_114694);
        free(mem_114699);
        free(mem_114710);
        free(mem_114715);
        free(mem_114731);
        free(mem_114736);
        free(mem_114747);
        free(mem_114752);
        free(mem_114763);
        free(mem_114768);
        free(mem_114779);
        free(mem_114784);
        free(mem_114791);
        free(mem_114802);
        free(mem_114807);
        free(mem_114818);
        free(mem_114823);
        free(mem_114834);
        free(mem_114839);
        free(mem_114850);
        free(mem_114855);
        free(mem_114866);
        free(mem_114871);
        free(mem_114886);
        free(mem_114893);
        if (memblock_unref(ctx, &mem_114882, "mem_114882") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116437, "mem_out_116437") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_116855, struct memblock wdown_mem_114423, struct memblock wkey_mem_114424, struct memblock wout_mem_114425, struct memblock wpe_mem_114426, struct memblock wqry_mem_114427, struct memblock wte_mem_114428, struct memblock wup_mem_114429, struct memblock wval_mem_114430, struct memblock wvoc_mem_114431, struct memblock tokens_mem_114432, struct memblock mask_mem_114433)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_114434_cached_sizze_116856 = 0;
    unsigned char *mem_114434 = NULL;
    int64_t mem_114439_cached_sizze_116857 = 0;
    unsigned char *mem_114439 = NULL;
    int64_t mem_114450_cached_sizze_116858 = 0;
    unsigned char *mem_114450 = NULL;
    int64_t mem_114455_cached_sizze_116859 = 0;
    unsigned char *mem_114455 = NULL;
    int64_t mem_114462_cached_sizze_116860 = 0;
    unsigned char *mem_114462 = NULL;
    int64_t mem_114473_cached_sizze_116861 = 0;
    unsigned char *mem_114473 = NULL;
    int64_t mem_114478_cached_sizze_116862 = 0;
    unsigned char *mem_114478 = NULL;
    int64_t mem_114485_cached_sizze_116863 = 0;
    unsigned char *mem_114485 = NULL;
    int64_t mem_114496_cached_sizze_116864 = 0;
    unsigned char *mem_114496 = NULL;
    int64_t mem_114497_cached_sizze_116865 = 0;
    unsigned char *mem_114497 = NULL;
    int64_t mem_114498_cached_sizze_116866 = 0;
    unsigned char *mem_114498 = NULL;
    int64_t mem_114511_cached_sizze_116867 = 0;
    unsigned char *mem_114511 = NULL;
    int64_t mem_114512_cached_sizze_116868 = 0;
    unsigned char *mem_114512 = NULL;
    int64_t mem_114513_cached_sizze_116869 = 0;
    unsigned char *mem_114513 = NULL;
    int64_t mem_114544_cached_sizze_116870 = 0;
    unsigned char *mem_114544 = NULL;
    int64_t mem_114545_cached_sizze_116871 = 0;
    unsigned char *mem_114545 = NULL;
    int64_t mem_114546_cached_sizze_116872 = 0;
    unsigned char *mem_114546 = NULL;
    int64_t mem_114562_cached_sizze_116873 = 0;
    unsigned char *mem_114562 = NULL;
    int64_t mem_114563_cached_sizze_116874 = 0;
    unsigned char *mem_114563 = NULL;
    int64_t mem_114564_cached_sizze_116875 = 0;
    unsigned char *mem_114564 = NULL;
    int64_t mem_114577_cached_sizze_116876 = 0;
    unsigned char *mem_114577 = NULL;
    int64_t mem_114578_cached_sizze_116877 = 0;
    unsigned char *mem_114578 = NULL;
    int64_t mem_114579_cached_sizze_116878 = 0;
    unsigned char *mem_114579 = NULL;
    int64_t mem_114625_cached_sizze_116879 = 0;
    unsigned char *mem_114625 = NULL;
    int64_t mem_114631_cached_sizze_116880 = 0;
    unsigned char *mem_114631 = NULL;
    int64_t mem_114636_cached_sizze_116881 = 0;
    unsigned char *mem_114636 = NULL;
    int64_t mem_114647_cached_sizze_116882 = 0;
    unsigned char *mem_114647 = NULL;
    int64_t mem_114652_cached_sizze_116883 = 0;
    unsigned char *mem_114652 = NULL;
    int64_t mem_114663_cached_sizze_116884 = 0;
    unsigned char *mem_114663 = NULL;
    int64_t mem_114668_cached_sizze_116885 = 0;
    unsigned char *mem_114668 = NULL;
    int64_t mem_114675_cached_sizze_116886 = 0;
    unsigned char *mem_114675 = NULL;
    int64_t mem_114682_cached_sizze_116887 = 0;
    unsigned char *mem_114682 = NULL;
    int64_t mem_114693_cached_sizze_116888 = 0;
    unsigned char *mem_114693 = NULL;
    int64_t mem_114698_cached_sizze_116889 = 0;
    unsigned char *mem_114698 = NULL;
    int64_t mem_114709_cached_sizze_116890 = 0;
    unsigned char *mem_114709 = NULL;
    int64_t mem_114714_cached_sizze_116891 = 0;
    unsigned char *mem_114714 = NULL;
    int64_t mem_114730_cached_sizze_116892 = 0;
    unsigned char *mem_114730 = NULL;
    int64_t mem_114735_cached_sizze_116893 = 0;
    unsigned char *mem_114735 = NULL;
    int64_t mem_114746_cached_sizze_116894 = 0;
    unsigned char *mem_114746 = NULL;
    int64_t mem_114751_cached_sizze_116895 = 0;
    unsigned char *mem_114751 = NULL;
    int64_t mem_114762_cached_sizze_116896 = 0;
    unsigned char *mem_114762 = NULL;
    int64_t mem_114767_cached_sizze_116897 = 0;
    unsigned char *mem_114767 = NULL;
    int64_t mem_114778_cached_sizze_116898 = 0;
    unsigned char *mem_114778 = NULL;
    int64_t mem_114783_cached_sizze_116899 = 0;
    unsigned char *mem_114783 = NULL;
    int64_t mem_114790_cached_sizze_116900 = 0;
    unsigned char *mem_114790 = NULL;
    int64_t mem_114801_cached_sizze_116901 = 0;
    unsigned char *mem_114801 = NULL;
    int64_t mem_114806_cached_sizze_116902 = 0;
    unsigned char *mem_114806 = NULL;
    int64_t mem_114817_cached_sizze_116903 = 0;
    unsigned char *mem_114817 = NULL;
    int64_t mem_114822_cached_sizze_116904 = 0;
    unsigned char *mem_114822 = NULL;
    int64_t mem_114833_cached_sizze_116905 = 0;
    unsigned char *mem_114833 = NULL;
    int64_t mem_114838_cached_sizze_116906 = 0;
    unsigned char *mem_114838 = NULL;
    int64_t mem_114849_cached_sizze_116907 = 0;
    unsigned char *mem_114849 = NULL;
    int64_t mem_114854_cached_sizze_116908 = 0;
    unsigned char *mem_114854 = NULL;
    int64_t mem_114865_cached_sizze_116909 = 0;
    unsigned char *mem_114865 = NULL;
    int64_t mem_114870_cached_sizze_116910 = 0;
    unsigned char *mem_114870 = NULL;
    int64_t mem_114886_cached_sizze_116911 = 0;
    unsigned char *mem_114886 = NULL;
    struct memblock mem_114881;
    
    mem_114881.references = NULL;
    
    struct memblock mem_out_116437;
    
    mem_out_116437.references = NULL;
    
    struct memblock mem_114414 = ctx->constants->mem_114414;
    struct memblock mem_114415 = ctx->constants->mem_114415;
    struct memblock mem_114416 = ctx->constants->mem_114416;
    struct memblock mem_114417 = ctx->constants->mem_114417;
    struct memblock mem_114418 = ctx->constants->mem_114418;
    struct memblock mem_114419 = ctx->constants->mem_114419;
    struct memblock mem_114420 = ctx->constants->mem_114420;
    struct memblock mem_114421 = ctx->constants->mem_114421;
    struct memblock mem_114422 = ctx->constants->mem_114422;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_114434_cached_sizze_116856 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114434, &mem_114434_cached_sizze_116856, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114439_cached_sizze_116857 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114439, &mem_114439_cached_sizze_116857, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113416 = 0; i_113416 < (int64_t) 16; i_113416++) {
        // futhark/microgpt.fut:440:41-50
        
        int64_t tmp_101720 = ((int64_t *) tokens_mem_114432.mem)[i_113416];
        
        // futhark/microgpt.fut:440:37-51
        
        bool x_101721 = sle64((int64_t) 0, tmp_101720);
        
        // futhark/microgpt.fut:440:37-51
        
        bool y_101722 = slt64(tmp_101720, (int64_t) 27);
        
        // futhark/microgpt.fut:440:37-51
        
        bool bounds_check_101723 = x_101721 && y_101722;
        
        // futhark/microgpt.fut:440:37-51
        
        bool index_certs_101724;
        
        if (!bounds_check_101723) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_101720, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:440:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:440:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113412 = 0; i_113412 < (int64_t) 16; i_113412++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_101731 = ((double *) wte_mem_114428.mem)[tmp_101720 * (int64_t) 16 + i_113412];
            
            ((double *) mem_114439)[i_113412] = lifted_lambda_res_101731;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114434, i_113416 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114439, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114450_cached_sizze_116858 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114450, &mem_114450_cached_sizze_116858, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114455_cached_sizze_116859 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114455, &mem_114455_cached_sizze_116859, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114462_cached_sizze_116860 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114462, &mem_114462_cached_sizze_116860, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113428 = 0; i_113428 < (int64_t) 16; i_113428++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_101757;
        double r_101759 = 0.0;
        
        for (int64_t i_101758 = 0; i_101758 < (int64_t) 16; i_101758++) {
            // futhark/microgpt.fut:71:46-49
            
            double zp_lhs_101760 = ((double *) wpe_mem_114426.mem)[i_113428 * (int64_t) 16 + i_101758];
            
            // futhark/microgpt.fut:71:46-49
            
            double zp_rhs_101761 = ((double *) mem_114434)[i_113428 * (int64_t) 16 + i_101758];
            
            // futhark/microgpt.fut:148:76-116
            
            double zp_res_101762 = zp_lhs_101760 + zp_rhs_101761;
            
            // futhark/microgpt.fut:148:94-163
            
            double zt_res_101763 = zp_res_101762 * zp_res_101762;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_101764 = r_101759 + zt_res_101763;
            double r_tmp_116441 = zp_res_101764;
            
            r_101759 = r_tmp_116441;
        }
        defunc_0_lifted_lambda_res_101757 = r_101759;
        // futhark/microgpt.fut:148:54-182
        
        double zs_res_101765 = defunc_0_lifted_lambda_res_101757 / 16.0;
        
        // futhark/microgpt.fut:149:24-55
        
        double zp_res_101766 = 1.0e-5 + zs_res_101765;
        
        // futhark/microgpt.fut:149:16-55
        
        double sqrt_res_101767 = futrts_sqrt64(zp_res_101766);
        
        // futhark/microgpt.fut:150:85-96
        
        double zs_res_101768 = 1.0 / sqrt_res_101767;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113420 = 0; i_113420 < (int64_t) 16; i_113420++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_101775 = ((double *) wpe_mem_114426.mem)[i_113428 * (int64_t) 16 + i_113420];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_101776 = ((double *) mem_114434)[i_113428 * (int64_t) 16 + i_113420];
            
            // futhark/microgpt.fut:150:38-78
            
            double zp_res_101777 = zp_lhs_101775 + zp_rhs_101776;
            
            // futhark/microgpt.fut:150:56-96
            
            double zt_res_101778 = zs_res_101768 * zp_res_101777;
            
            ((double *) mem_114455)[i_113420] = zt_res_101778;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113424 = 0; i_113424 < (int64_t) 16; i_113424++) {
            // futhark/microgpt.fut:151:4-14
            
            double lifted_lambda_res_101786 = ((double *) mem_114455)[i_113424];
            
            ((double *) mem_114462)[i_113424] = lifted_lambda_res_101786;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114450, i_113428 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114462, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114473_cached_sizze_116861 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114473, &mem_114473_cached_sizze_116861, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114478_cached_sizze_116862 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114478, &mem_114478_cached_sizze_116862, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114485_cached_sizze_116863 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114485, &mem_114485_cached_sizze_116863, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113440 = 0; i_113440 < (int64_t) 16; i_113440++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_101795;
        double r_101797 = 0.0;
        
        for (int64_t i_101796 = 0; i_101796 < (int64_t) 16; i_101796++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_101798 = ((double *) mem_114450)[i_113440 * (int64_t) 16 + i_101796];
            
            // futhark/microgpt.fut:152:78-115
            
            double zt_res_101799 = zt_lhs_101798 * zt_lhs_101798;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_101800 = r_101797 + zt_res_101799;
            double r_tmp_116445 = zp_res_101800;
            
            r_101797 = r_tmp_116445;
        }
        defunc_0_lifted_lambda_res_101795 = r_101797;
        // futhark/microgpt.fut:152:57-133
        
        double zs_res_101801 = defunc_0_lifted_lambda_res_101795 / 16.0;
        
        // futhark/microgpt.fut:153:24-55
        
        double zp_res_101802 = 1.0e-5 + zs_res_101801;
        
        // futhark/microgpt.fut:153:16-55
        
        double sqrt_res_101803 = futrts_sqrt64(zp_res_101802);
        
        // futhark/microgpt.fut:154:59-70
        
        double zs_res_101804 = 1.0 / sqrt_res_101803;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113432 = 0; i_113432 < (int64_t) 16; i_113432++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_101811 = ((double *) mem_114450)[i_113440 * (int64_t) 16 + i_113432];
            
            // futhark/microgpt.fut:154:37-70
            
            double zt_res_101812 = zs_res_101804 * zt_lhs_101811;
            
            ((double *) mem_114478)[i_113432] = zt_res_101812;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113436 = 0; i_113436 < (int64_t) 16; i_113436++) {
            // futhark/microgpt.fut:155:4-14
            
            double lifted_lambda_res_101820 = ((double *) mem_114478)[i_113436];
            
            ((double *) mem_114485)[i_113436] = lifted_lambda_res_101820;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114473, i_113440 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114485, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114496_cached_sizze_116864 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114496, &mem_114496_cached_sizze_116864, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114497_cached_sizze_116865 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114497, &mem_114497_cached_sizze_116865, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114498_cached_sizze_116866 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114498, &mem_114498_cached_sizze_116866, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114511_cached_sizze_116867 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114511, &mem_114511_cached_sizze_116867, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114512_cached_sizze_116868 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114512, &mem_114512_cached_sizze_116868, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114513_cached_sizze_116869 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114513, &mem_114513_cached_sizze_116869, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113458 = 0; i_113458 < (int64_t) 16; i_113458++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113448 = 0; i_113448 < (int64_t) 16; i_113448++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_108540;
            double r_108542 = 0.0;
            
            for (int64_t i_108541 = 0; i_108541 < (int64_t) 16; i_108541++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_108543 = ((double *) wqry_mem_114427.mem)[i_113448 * (int64_t) 16 + i_108541];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_108544 = ((double *) mem_114473)[i_113458 * (int64_t) 16 + i_108541];
                
                // futhark/microgpt.fut:156:66-105
                
                double zt_res_108545 = zt_lhs_108543 * zt_rhs_108544;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_108546 = r_108542 + zt_res_108545;
                double r_tmp_116454 = zp_res_108546;
                
                r_108542 = r_tmp_116454;
            }
            defunc_0_lifted_lambda_res_108540 = r_108542;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_108553;
            double r_108555 = 0.0;
            
            for (int64_t i_108554 = 0; i_108554 < (int64_t) 16; i_108554++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_108556 = ((double *) wkey_mem_114424.mem)[i_113448 * (int64_t) 16 + i_108554];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_108557 = ((double *) mem_114473)[i_113458 * (int64_t) 16 + i_108554];
                
                // futhark/microgpt.fut:157:66-105
                
                double zt_res_108558 = zt_lhs_108556 * zt_rhs_108557;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_108559 = r_108555 + zt_res_108558;
                double r_tmp_116455 = zp_res_108559;
                
                r_108555 = r_tmp_116455;
            }
            defunc_0_lifted_lambda_res_108553 = r_108555;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_108569;
            double r_108571 = 0.0;
            
            for (int64_t i_108570 = 0; i_108570 < (int64_t) 16; i_108570++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_108572 = ((double *) wval_mem_114430.mem)[i_113448 * (int64_t) 16 + i_108570];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_108573 = ((double *) mem_114473)[i_113458 * (int64_t) 16 + i_108570];
                
                // futhark/microgpt.fut:158:66-105
                
                double zt_res_108574 = zt_lhs_108572 * zt_rhs_108573;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_108575 = r_108571 + zt_res_108574;
                double r_tmp_116456 = zp_res_108575;
                
                r_108571 = r_tmp_116456;
            }
            defunc_0_lifted_lambda_res_108569 = r_108571;
            ((double *) mem_114511)[i_113448] = defunc_0_lifted_lambda_res_108569;
            ((double *) mem_114512)[i_113448] = defunc_0_lifted_lambda_res_108553;
            ((double *) mem_114513)[i_113448] = defunc_0_lifted_lambda_res_108540;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114496, i_113458 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114511, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114497, i_113458 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114512, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114498, i_113458 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114513, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114544_cached_sizze_116870 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114544, &mem_114544_cached_sizze_116870, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114545_cached_sizze_116871 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114545, &mem_114545_cached_sizze_116871, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114546_cached_sizze_116872 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114546, &mem_114546_cached_sizze_116872, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114562_cached_sizze_116873 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114562, &mem_114562_cached_sizze_116873, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114563_cached_sizze_116874 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114563, &mem_114563_cached_sizze_116874, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114564_cached_sizze_116875 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114564, &mem_114564_cached_sizze_116875, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114577_cached_sizze_116876 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_114577, &mem_114577_cached_sizze_116876, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114578_cached_sizze_116877 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_114578, &mem_114578_cached_sizze_116877, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114579_cached_sizze_116878 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_114579, &mem_114579_cached_sizze_116878, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113488 = 0; i_113488 < (int64_t) 4; i_113488++) {
        // futhark/microgpt.fut:159:69-72
        
        int64_t zp_lhs_108416 = mul64((int64_t) 4, i_113488);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113478 = 0; i_113478 < (int64_t) 16; i_113478++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113468 = 0; i_113468 < (int64_t) 4; i_113468++) {
                // futhark/microgpt.fut:159:74-81
                
                int64_t tmp_108733 = add64(zp_lhs_108416, i_113468);
                
                // futhark/microgpt.fut:159:51-83
                
                bool x_108734 = sle64((int64_t) 0, tmp_108733);
                
                // futhark/microgpt.fut:159:51-83
                
                bool y_108735 = slt64(tmp_108733, (int64_t) 16);
                
                // futhark/microgpt.fut:159:51-83
                
                bool bounds_check_108736 = x_108734 && y_108735;
                
                // futhark/microgpt.fut:159:51-83
                
                bool index_certs_108737;
                
                if (!bounds_check_108736) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_108733, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:159:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:159:15-84\n   #9  futhark/microgpt.fut:441:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_108738 = ((double *) mem_114498)[i_113478 * (int64_t) 16 + tmp_108733];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_108746 = ((double *) mem_114497)[i_113478 * (int64_t) 16 + tmp_108733];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_108757 = ((double *) mem_114496)[i_113478 * (int64_t) 16 + tmp_108733];
                
                ((double *) mem_114577)[i_113468] = lifted_lambda_res_108757;
                ((double *) mem_114578)[i_113468] = lifted_lambda_res_108746;
                ((double *) mem_114579)[i_113468] = lifted_lambda_res_108738;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114562, i_113478 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114577, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114563, i_113478 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114578, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114564, i_113478 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114579, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_114544, i_113488 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_114562, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_114545, i_113488 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_114563, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_114546, i_113488 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_114564, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114625_cached_sizze_116879 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114625, &mem_114625_cached_sizze_116879, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114631_cached_sizze_116880 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114631, &mem_114631_cached_sizze_116880, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114636_cached_sizze_116881 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114636, &mem_114636_cached_sizze_116881, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114647_cached_sizze_116882 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114647, &mem_114647_cached_sizze_116882, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114652_cached_sizze_116883 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114652, &mem_114652_cached_sizze_116883, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114663_cached_sizze_116884 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114663, &mem_114663_cached_sizze_116884, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114668_cached_sizze_116885 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114668, &mem_114668_cached_sizze_116885, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114675_cached_sizze_116886 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114675, &mem_114675_cached_sizze_116886, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114682_cached_sizze_116887 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114682, &mem_114682_cached_sizze_116887, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114693_cached_sizze_116888 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114693, &mem_114693_cached_sizze_116888, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114698_cached_sizze_116889 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_114698, &mem_114698_cached_sizze_116889, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114709_cached_sizze_116890 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114709, &mem_114709_cached_sizze_116890, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114714_cached_sizze_116891 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_114714, &mem_114714_cached_sizze_116891, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113544 = 0; i_113544 < (int64_t) 4; i_113544++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113498 = 0; i_113498 < (int64_t) 16; i_113498++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113494 = 0; i_113494 < (int64_t) 16; i_113494++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_101965;
                double r_101967 = 0.0;
                
                for (int64_t i_101966 = 0; i_101966 < (int64_t) 4; i_101966++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_101968 = ((double *) mem_114546)[i_113544 * (int64_t) 64 + i_113498 * (int64_t) 4 + i_101966];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_101969 = ((double *) mem_114545)[i_113544 * (int64_t) 64 + i_113494 * (int64_t) 4 + i_101966];
                    
                    // futhark/microgpt.fut:162:113-164
                    
                    double zt_res_101970 = zt_lhs_101968 * zt_rhs_101969;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_101971 = r_101967 + zt_res_101970;
                    double r_tmp_116469 = zp_res_101971;
                    
                    r_101967 = r_tmp_116469;
                }
                defunc_0_lifted_lambda_res_101965 = r_101967;
                ((double *) mem_114636)[i_113494] = defunc_0_lifted_lambda_res_101965;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114631, i_113498 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114636, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113506 = 0; i_113506 < (int64_t) 16; i_113506++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113502 = 0; i_113502 < (int64_t) 16; i_113502++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_101986 = ((double *) mem_114631)[i_113506 * (int64_t) 16 + i_113502];
                
                // futhark/microgpt.fut:163:47-78
                
                double zs_res_101987 = zs_lhs_101986 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_101988 = ((double *) mask_mem_114433.mem)[i_113506 * (int64_t) 16 + i_113502];
                
                // futhark/microgpt.fut:163:65-102
                
                double zp_res_101989 = zs_res_101987 + zp_rhs_101988;
                
                ((double *) mem_114652)[i_113502] = zp_res_101989;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114647, i_113506 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114652, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113524 = 0; i_113524 < (int64_t) 16; i_113524++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_108835;
            double redout_113508 = -INFINITY;
            
            for (int64_t i_113509 = 0; i_113509 < (int64_t) 16; i_113509++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_108784 = ((double *) mem_114647)[i_113524 * (int64_t) 16 + i_113509];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_102010 = fmax64(lifted_lambda_res_108784, redout_113508);
                double redout_tmp_116473 = max_res_102010;
                
                redout_113508 = redout_tmp_116473;
            }
            defunc_0_reduce_res_108835 = redout_113508;
            // futhark/microgpt.fut:165:67-76
            
            double neg_res_102011 = -defunc_0_reduce_res_108835;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113512 = 0; i_113512 < (int64_t) 16; i_113512++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_102018 = ((double *) mem_114647)[i_113524 * (int64_t) 16 + i_113512];
                
                // futhark/microgpt.fut:165:44-76
                
                double zp_res_102019 = neg_res_102011 + zp_lhs_102018;
                
                // futhark/microgpt.fut:165:37-76
                
                double exp_res_102020 = futrts_exp64(zp_res_102019);
                
                ((double *) mem_114668)[i_113512] = exp_res_102020;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_102022;
            double r_102024 = 0.0;
            
            for (int64_t i_102023 = 0; i_102023 < (int64_t) 16; i_102023++) {
                // futhark/microgpt.fut:166:36-46
                
                double lifted_lambda_res_102025 = ((double *) mem_114668)[i_102023];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_102026 = r_102024 + lifted_lambda_res_102025;
                double r_tmp_116475 = zp_res_102026;
                
                r_102024 = r_tmp_116475;
            }
            defunc_0_lifted_lambda_res_102022 = r_102024;
            // futhark/microgpt.fut:167:53-64
            
            double zs_res_102027 = 1.0 / defunc_0_lifted_lambda_res_102022;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113516 = 0; i_113516 < (int64_t) 16; i_113516++) {
                // futhark/microgpt.fut:167:37-47
                
                double zt_lhs_102034 = ((double *) mem_114668)[i_113516];
                
                // futhark/microgpt.fut:167:37-64
                
                double zt_res_102035 = zs_res_102027 * zt_lhs_102034;
                
                ((double *) mem_114675)[i_113516] = zt_res_102035;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113520 = 0; i_113520 < (int64_t) 16; i_113520++) {
                // futhark/microgpt.fut:168:4-14
                
                double lifted_lambda_res_102043 = ((double *) mem_114675)[i_113520];
                
                ((double *) mem_114682)[i_113520] = lifted_lambda_res_102043;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114663, i_113524 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114682, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113532 = 0; i_113532 < (int64_t) 16; i_113532++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113528 = 0; i_113528 < (int64_t) 4; i_113528++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_102058;
                double r_102060 = 0.0;
                
                for (int64_t i_102059 = 0; i_102059 < (int64_t) 16; i_102059++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_102061 = ((double *) mem_114663)[i_113532 * (int64_t) 16 + i_102059];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_102062 = ((double *) mem_114544)[i_113544 * (int64_t) 64 + i_102059 * (int64_t) 4 + i_113528];
                    
                    // futhark/microgpt.fut:169:66-111
                    
                    double zt_res_102063 = zt_lhs_102061 * zt_rhs_102062;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_102064 = r_102060 + zt_res_102063;
                    double r_tmp_116480 = zp_res_102064;
                    
                    r_102060 = r_tmp_116480;
                }
                defunc_0_lifted_lambda_res_102058 = r_102060;
                ((double *) mem_114698)[i_113528] = defunc_0_lifted_lambda_res_102058;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114693, i_113532 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114698, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113540 = 0; i_113540 < (int64_t) 16; i_113540++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113536 = 0; i_113536 < (int64_t) 4; i_113536++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_102079 = ((double *) mem_114693)[i_113540 * (int64_t) 4 + i_113536];
                
                ((double *) mem_114714)[i_113536] = lifted_lambda_res_102079;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114709, i_113540 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114714, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_114625, i_113544 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_114709, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114730_cached_sizze_116892 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114730, &mem_114730_cached_sizze_116892, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114735_cached_sizze_116893 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114735, &mem_114735_cached_sizze_116893, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113552 = 0; i_113552 < (int64_t) 16; i_113552++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113548 = 0; i_113548 < (int64_t) 16; i_113548++) {
            // futhark/microgpt.fut:171:54-57
            
            int64_t tmp_102091 = sdiv64(i_113548, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-59
            
            bool x_102092 = sle64((int64_t) 0, tmp_102091);
            
            // futhark/microgpt.fut:171:44-59
            
            bool y_102093 = slt64(tmp_102091, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-59
            
            bool bounds_check_102094 = x_102092 && y_102093;
            
            // futhark/microgpt.fut:171:44-59
            
            bool index_certs_102095;
            
            if (!bounds_check_102094) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_102091, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:171:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:171:15-80\n   #6  futhark/microgpt.fut:441:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:171:74-77
            
            int64_t tmp_102096 = smod64(i_113548, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-79
            
            bool x_102097 = sle64((int64_t) 0, tmp_102096);
            
            // futhark/microgpt.fut:171:44-79
            
            bool y_102098 = slt64(tmp_102096, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-79
            
            bool bounds_check_102099 = x_102097 && y_102098;
            
            // futhark/microgpt.fut:171:44-79
            
            bool index_certs_102100;
            
            if (!bounds_check_102099) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_102096, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:171:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:171:15-80\n   #6  futhark/microgpt.fut:441:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_102101 = ((double *) mem_114625)[tmp_102091 * (int64_t) 64 + i_113552 * (int64_t) 4 + tmp_102096];
            
            ((double *) mem_114735)[i_113548] = lifted_lambda_res_102101;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114730, i_113552 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114735, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114746_cached_sizze_116894 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114746, &mem_114746_cached_sizze_116894, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114751_cached_sizze_116895 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114751, &mem_114751_cached_sizze_116895, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113560 = 0; i_113560 < (int64_t) 16; i_113560++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113556 = 0; i_113556 < (int64_t) 16; i_113556++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_102116;
            double r_102118 = 0.0;
            
            for (int64_t i_102117 = 0; i_102117 < (int64_t) 16; i_102117++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_102119 = ((double *) wout_mem_114425.mem)[i_113556 * (int64_t) 16 + i_102117];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_102120 = ((double *) mem_114730)[i_113560 * (int64_t) 16 + i_102117];
                
                // futhark/microgpt.fut:172:67-106
                
                double zt_res_102121 = zt_lhs_102119 * zt_rhs_102120;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_102122 = r_102118 + zt_res_102121;
                double r_tmp_116487 = zp_res_102122;
                
                r_102118 = r_tmp_116487;
            }
            defunc_0_lifted_lambda_res_102116 = r_102118;
            ((double *) mem_114751)[i_113556] = defunc_0_lifted_lambda_res_102116;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114746, i_113560 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114751, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114762_cached_sizze_116896 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114762, &mem_114762_cached_sizze_116896, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114767_cached_sizze_116897 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114767, &mem_114767_cached_sizze_116897, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113568 = 0; i_113568 < (int64_t) 16; i_113568++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113564 = 0; i_113564 < (int64_t) 16; i_113564++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_102137 = ((double *) mem_114746)[i_113568 * (int64_t) 16 + i_113564];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_102138 = ((double *) mem_114450)[i_113568 * (int64_t) 16 + i_113564];
            
            // futhark/microgpt.fut:173:46-84
            
            double zp_res_102139 = zp_lhs_102137 + zp_rhs_102138;
            
            ((double *) mem_114767)[i_113564] = zp_res_102139;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114762, i_113568 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114767, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114778_cached_sizze_116898 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114778, &mem_114778_cached_sizze_116898, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114783_cached_sizze_116899 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114783, &mem_114783_cached_sizze_116899, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114790_cached_sizze_116900 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114790, &mem_114790_cached_sizze_116900, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113580 = 0; i_113580 < (int64_t) 16; i_113580++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_102148;
        double r_102150 = 0.0;
        
        for (int64_t i_102149 = 0; i_102149 < (int64_t) 16; i_102149++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_102151 = ((double *) mem_114762)[i_113580 * (int64_t) 16 + i_102149];
            
            // futhark/microgpt.fut:174:79-118
            
            double zt_res_102152 = zt_lhs_102151 * zt_lhs_102151;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_102153 = r_102150 + zt_res_102152;
            double r_tmp_116491 = zp_res_102153;
            
            r_102150 = r_tmp_116491;
        }
        defunc_0_lifted_lambda_res_102148 = r_102150;
        // futhark/microgpt.fut:174:58-136
        
        double zs_res_102154 = defunc_0_lifted_lambda_res_102148 / 16.0;
        
        // futhark/microgpt.fut:175:24-55
        
        double zp_res_102155 = 1.0e-5 + zs_res_102154;
        
        // futhark/microgpt.fut:175:16-55
        
        double sqrt_res_102156 = futrts_sqrt64(zp_res_102155);
        
        // futhark/microgpt.fut:176:60-71
        
        double zs_res_102157 = 1.0 / sqrt_res_102156;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113572 = 0; i_113572 < (int64_t) 16; i_113572++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_102164 = ((double *) mem_114762)[i_113580 * (int64_t) 16 + i_113572];
            
            // futhark/microgpt.fut:176:37-71
            
            double zt_res_102165 = zs_res_102157 * zt_lhs_102164;
            
            ((double *) mem_114783)[i_113572] = zt_res_102165;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113576 = 0; i_113576 < (int64_t) 16; i_113576++) {
            // futhark/microgpt.fut:177:4-14
            
            double lifted_lambda_res_102173 = ((double *) mem_114783)[i_113576];
            
            ((double *) mem_114790)[i_113576] = lifted_lambda_res_102173;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114778, i_113580 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114790, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114801_cached_sizze_116901 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_114801, &mem_114801_cached_sizze_116901, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114806_cached_sizze_116902 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114806, &mem_114806_cached_sizze_116902, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113588 = 0; i_113588 < (int64_t) 16; i_113588++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113584 = 0; i_113584 < (int64_t) 64; i_113584++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_102189;
            double r_102191 = 0.0;
            
            for (int64_t i_102190 = 0; i_102190 < (int64_t) 16; i_102190++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_102192 = ((double *) wup_mem_114429.mem)[i_113584 * (int64_t) 16 + i_102190];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_102193 = ((double *) mem_114778)[i_113588 * (int64_t) 16 + i_102190];
                
                // futhark/microgpt.fut:178:67-106
                
                double zt_res_102194 = zt_lhs_102192 * zt_rhs_102193;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_102195 = r_102191 + zt_res_102194;
                double r_tmp_116496 = zp_res_102195;
                
                r_102191 = r_tmp_116496;
            }
            defunc_0_lifted_lambda_res_102189 = r_102191;
            ((double *) mem_114806)[i_113584] = defunc_0_lifted_lambda_res_102189;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114801, i_113588 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114806, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114817_cached_sizze_116903 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_114817, &mem_114817_cached_sizze_116903, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114822_cached_sizze_116904 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114822, &mem_114822_cached_sizze_116904, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113596 = 0; i_113596 < (int64_t) 16; i_113596++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113592 = 0; i_113592 < (int64_t) 64; i_113592++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_102210 = ((double *) mem_114801)[i_113596 * (int64_t) 64 + i_113592];
            
            // futhark/microgpt.fut:179:45-73
            
            double max_res_102211 = fmax64(0.0, max_arg0_102210);
            
            ((double *) mem_114822)[i_113592] = max_res_102211;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114817, i_113596 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114822, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114833_cached_sizze_116905 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114833, &mem_114833_cached_sizze_116905, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114838_cached_sizze_116906 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114838, &mem_114838_cached_sizze_116906, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113604 = 0; i_113604 < (int64_t) 16; i_113604++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113600 = 0; i_113600 < (int64_t) 16; i_113600++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_102226;
            double r_102228 = 0.0;
            
            for (int64_t i_102227 = 0; i_102227 < (int64_t) 64; i_102227++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_102229 = ((double *) wdown_mem_114423.mem)[i_113600 * (int64_t) 64 + i_102227];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_102230 = ((double *) mem_114817)[i_113604 * (int64_t) 64 + i_102227];
                
                // futhark/microgpt.fut:180:67-108
                
                double zt_res_102231 = zt_lhs_102229 * zt_rhs_102230;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_102232 = r_102228 + zt_res_102231;
                double r_tmp_116501 = zp_res_102232;
                
                r_102228 = r_tmp_116501;
            }
            defunc_0_lifted_lambda_res_102226 = r_102228;
            ((double *) mem_114838)[i_113600] = defunc_0_lifted_lambda_res_102226;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114833, i_113604 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114838, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114849_cached_sizze_116907 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114849, &mem_114849_cached_sizze_116907, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114854_cached_sizze_116908 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114854, &mem_114854_cached_sizze_116908, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113612 = 0; i_113612 < (int64_t) 16; i_113612++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113608 = 0; i_113608 < (int64_t) 16; i_113608++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_102247 = ((double *) mem_114833)[i_113612 * (int64_t) 16 + i_113608];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_102248 = ((double *) mem_114762)[i_113612 * (int64_t) 16 + i_113608];
            
            // futhark/microgpt.fut:181:46-85
            
            double zp_res_102249 = zp_lhs_102247 + zp_rhs_102248;
            
            ((double *) mem_114854)[i_113608] = zp_res_102249;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114849, i_113612 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114854, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114865_cached_sizze_116909 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_114865, &mem_114865_cached_sizze_116909, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114870_cached_sizze_116910 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_114870, &mem_114870_cached_sizze_116910, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113620 = 0; i_113620 < (int64_t) 16; i_113620++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113616 = 0; i_113616 < (int64_t) 27; i_113616++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_102265;
            double r_102267 = 0.0;
            
            for (int64_t i_102266 = 0; i_102266 < (int64_t) 16; i_102266++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_102268 = ((double *) wvoc_mem_114431.mem)[i_113616 * (int64_t) 16 + i_102266];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_102269 = ((double *) mem_114849)[i_113620 * (int64_t) 16 + i_102266];
                
                // futhark/microgpt.fut:182:67-107
                
                double zt_res_102270 = zt_lhs_102268 * zt_rhs_102269;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_102271 = r_102267 + zt_res_102270;
                double r_tmp_116506 = zp_res_102271;
                
                r_102267 = r_tmp_116506;
            }
            defunc_0_lifted_lambda_res_102265 = r_102267;
            ((double *) mem_114870)[i_113616] = defunc_0_lifted_lambda_res_102265;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114865, i_113620 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114870, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_114881, (int64_t) 3456, "mem_114881")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114886_cached_sizze_116911 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_114886, &mem_114886_cached_sizze_116911, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_113628 = 0; i_113628 < (int64_t) 16; i_113628++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113624 = 0; i_113624 < (int64_t) 27; i_113624++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_102286 = ((double *) mem_114865)[i_113628 * (int64_t) 27 + i_113624];
            
            ((double *) mem_114886)[i_113624] = lifted_lambda_res_102286;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_114881.mem, i_113628 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114886, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_116437, &mem_114881, "mem_114881") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116855, &mem_out_116437, "mem_out_116437") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_114434);
        free(mem_114439);
        free(mem_114450);
        free(mem_114455);
        free(mem_114462);
        free(mem_114473);
        free(mem_114478);
        free(mem_114485);
        free(mem_114496);
        free(mem_114497);
        free(mem_114498);
        free(mem_114511);
        free(mem_114512);
        free(mem_114513);
        free(mem_114544);
        free(mem_114545);
        free(mem_114546);
        free(mem_114562);
        free(mem_114563);
        free(mem_114564);
        free(mem_114577);
        free(mem_114578);
        free(mem_114579);
        free(mem_114625);
        free(mem_114631);
        free(mem_114636);
        free(mem_114647);
        free(mem_114652);
        free(mem_114663);
        free(mem_114668);
        free(mem_114675);
        free(mem_114682);
        free(mem_114693);
        free(mem_114698);
        free(mem_114709);
        free(mem_114714);
        free(mem_114730);
        free(mem_114735);
        free(mem_114746);
        free(mem_114751);
        free(mem_114762);
        free(mem_114767);
        free(mem_114778);
        free(mem_114783);
        free(mem_114790);
        free(mem_114801);
        free(mem_114806);
        free(mem_114817);
        free(mem_114822);
        free(mem_114833);
        free(mem_114838);
        free(mem_114849);
        free(mem_114854);
        free(mem_114865);
        free(mem_114870);
        free(mem_114886);
        if (memblock_unref(ctx, &mem_114881, "mem_114881") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116437, "mem_out_116437") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_116912, struct memblock *mem_out_p_116913, struct memblock *mem_out_p_116914, struct memblock *mem_out_p_116915, struct memblock *mem_out_p_116916, struct memblock *mem_out_p_116917, struct memblock *mem_out_p_116918, struct memblock *mem_out_p_116919, struct memblock *mem_out_p_116920, struct memblock wte_mem_114423, struct memblock wpe_mem_114424, struct memblock wqry_mem_114425, struct memblock wkey_mem_114426, struct memblock wval_mem_114427, struct memblock wout_mem_114428, struct memblock wup_mem_114429, struct memblock wdown_mem_114430, struct memblock wvoc_mem_114431)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_116445;
    
    mem_out_116445.references = NULL;
    
    struct memblock mem_out_116444;
    
    mem_out_116444.references = NULL;
    
    struct memblock mem_out_116443;
    
    mem_out_116443.references = NULL;
    
    struct memblock mem_out_116442;
    
    mem_out_116442.references = NULL;
    
    struct memblock mem_out_116441;
    
    mem_out_116441.references = NULL;
    
    struct memblock mem_out_116440;
    
    mem_out_116440.references = NULL;
    
    struct memblock mem_out_116439;
    
    mem_out_116439.references = NULL;
    
    struct memblock mem_out_116438;
    
    mem_out_116438.references = NULL;
    
    struct memblock mem_out_116437;
    
    mem_out_116437.references = NULL;
    
    struct memblock mem_114414 = ctx->constants->mem_114414;
    struct memblock mem_114415 = ctx->constants->mem_114415;
    struct memblock mem_114416 = ctx->constants->mem_114416;
    struct memblock mem_114417 = ctx->constants->mem_114417;
    struct memblock mem_114418 = ctx->constants->mem_114418;
    struct memblock mem_114419 = ctx->constants->mem_114419;
    struct memblock mem_114420 = ctx->constants->mem_114420;
    struct memblock mem_114421 = ctx->constants->mem_114421;
    struct memblock mem_114422 = ctx->constants->mem_114422;
    
    if (memblock_set(ctx, &mem_out_116437, &wdown_mem_114430, "wdown_mem_114430") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116438, &wkey_mem_114426, "wkey_mem_114426") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116439, &wout_mem_114428, "wout_mem_114428") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116440, &wpe_mem_114424, "wpe_mem_114424") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116441, &wqry_mem_114425, "wqry_mem_114425") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116442, &wte_mem_114423, "wte_mem_114423") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116443, &wup_mem_114429, "wup_mem_114429") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116444, &wval_mem_114427, "wval_mem_114427") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116445, &wvoc_mem_114431, "wvoc_mem_114431") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116912, &mem_out_116437, "mem_out_116437") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116913, &mem_out_116438, "mem_out_116438") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116914, &mem_out_116439, "mem_out_116439") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116915, &mem_out_116440, "mem_out_116440") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116916, &mem_out_116441, "mem_out_116441") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116917, &mem_out_116442, "mem_out_116442") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116918, &mem_out_116443, "mem_out_116443") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116919, &mem_out_116444, "mem_out_116444") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116920, &mem_out_116445, "mem_out_116445") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_116445, "mem_out_116445") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116444, "mem_out_116444") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116443, "mem_out_116443") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116442, "mem_out_116442") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116441, "mem_out_116441") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116440, "mem_out_116440") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116439, "mem_out_116439") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116438, "mem_out_116438") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116437, "mem_out_116437") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_116921, struct memblock *mem_out_p_116922, struct memblock *mem_out_p_116923, struct memblock *mem_out_p_116924, struct memblock *mem_out_p_116925, struct memblock *mem_out_p_116926, struct memblock *mem_out_p_116927, struct memblock *mem_out_p_116928, struct memblock *mem_out_p_116929, struct memblock *mem_out_p_116930, struct memblock *mem_out_p_116931, struct memblock *mem_out_p_116932, struct memblock *mem_out_p_116933, struct memblock *mem_out_p_116934, struct memblock *mem_out_p_116935, struct memblock *mem_out_p_116936, struct memblock *mem_out_p_116937, struct memblock *mem_out_p_116938, struct memblock *mem_out_p_116939, struct memblock *mem_out_p_116940, struct memblock *mem_out_p_116941, struct memblock *mem_out_p_116942, struct memblock *mem_out_p_116943, struct memblock *mem_out_p_116944, struct memblock *mem_out_p_116945, struct memblock *mem_out_p_116946, struct memblock *mem_out_p_116947, struct memblock wdown_mem_114423, struct memblock wkey_mem_114424, struct memblock wout_mem_114425, struct memblock wpe_mem_114426, struct memblock wqry_mem_114427, struct memblock wte_mem_114428, struct memblock wup_mem_114429, struct memblock wval_mem_114430, struct memblock wvoc_mem_114431, struct memblock wdown_mem_114432, struct memblock wkey_mem_114433, struct memblock wout_mem_114434, struct memblock wpe_mem_114435, struct memblock wqry_mem_114436, struct memblock wte_mem_114437, struct memblock wup_mem_114438, struct memblock wval_mem_114439, struct memblock wvoc_mem_114440, struct memblock wdown_mem_114441, struct memblock wkey_mem_114442, struct memblock wout_mem_114443, struct memblock wpe_mem_114444, struct memblock wqry_mem_114445, struct memblock wte_mem_114446, struct memblock wup_mem_114447, struct memblock wval_mem_114448, struct memblock wvoc_mem_114449, struct memblock masks_mem_114450, struct memblock dls_mem_114451, struct memblock seqs_mem_114452)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_114561_cached_sizze_116948 = 0;
    unsigned char *mem_114561 = NULL;
    int64_t mem_114562_cached_sizze_116949 = 0;
    unsigned char *mem_114562 = NULL;
    int64_t mem_114571_cached_sizze_116950 = 0;
    unsigned char *mem_114571 = NULL;
    int64_t mem_114578_cached_sizze_116951 = 0;
    unsigned char *mem_114578 = NULL;
    int64_t mem_114593_cached_sizze_116952 = 0;
    unsigned char *mem_114593 = NULL;
    int64_t mem_114594_cached_sizze_116953 = 0;
    unsigned char *mem_114594 = NULL;
    int64_t mem_114602_cached_sizze_116954 = 0;
    unsigned char *mem_114602 = NULL;
    int64_t mem_114609_cached_sizze_116955 = 0;
    unsigned char *mem_114609 = NULL;
    int64_t mem_114623_cached_sizze_116956 = 0;
    unsigned char *mem_114623 = NULL;
    int64_t mem_114624_cached_sizze_116957 = 0;
    unsigned char *mem_114624 = NULL;
    int64_t mem_114632_cached_sizze_116958 = 0;
    unsigned char *mem_114632 = NULL;
    int64_t mem_114639_cached_sizze_116959 = 0;
    unsigned char *mem_114639 = NULL;
    int64_t mem_114653_cached_sizze_116960 = 0;
    unsigned char *mem_114653 = NULL;
    int64_t mem_114654_cached_sizze_116961 = 0;
    unsigned char *mem_114654 = NULL;
    int64_t mem_114655_cached_sizze_116962 = 0;
    unsigned char *mem_114655 = NULL;
    int64_t mem_114668_cached_sizze_116963 = 0;
    unsigned char *mem_114668 = NULL;
    int64_t mem_114669_cached_sizze_116964 = 0;
    unsigned char *mem_114669 = NULL;
    int64_t mem_114670_cached_sizze_116965 = 0;
    unsigned char *mem_114670 = NULL;
    int64_t mem_114701_cached_sizze_116966 = 0;
    unsigned char *mem_114701 = NULL;
    int64_t mem_114702_cached_sizze_116967 = 0;
    unsigned char *mem_114702 = NULL;
    int64_t mem_114703_cached_sizze_116968 = 0;
    unsigned char *mem_114703 = NULL;
    int64_t mem_114719_cached_sizze_116969 = 0;
    unsigned char *mem_114719 = NULL;
    int64_t mem_114720_cached_sizze_116970 = 0;
    unsigned char *mem_114720 = NULL;
    int64_t mem_114721_cached_sizze_116971 = 0;
    unsigned char *mem_114721 = NULL;
    int64_t mem_114734_cached_sizze_116972 = 0;
    unsigned char *mem_114734 = NULL;
    int64_t mem_114735_cached_sizze_116973 = 0;
    unsigned char *mem_114735 = NULL;
    int64_t mem_114736_cached_sizze_116974 = 0;
    unsigned char *mem_114736 = NULL;
    int64_t mem_114782_cached_sizze_116975 = 0;
    unsigned char *mem_114782 = NULL;
    int64_t mem_114783_cached_sizze_116976 = 0;
    unsigned char *mem_114783 = NULL;
    int64_t mem_114794_cached_sizze_116977 = 0;
    unsigned char *mem_114794 = NULL;
    int64_t mem_114795_cached_sizze_116978 = 0;
    unsigned char *mem_114795 = NULL;
    int64_t mem_114804_cached_sizze_116979 = 0;
    unsigned char *mem_114804 = NULL;
    int64_t mem_114805_cached_sizze_116980 = 0;
    unsigned char *mem_114805 = NULL;
    int64_t mem_114826_cached_sizze_116981 = 0;
    unsigned char *mem_114826 = NULL;
    int64_t mem_114831_cached_sizze_116982 = 0;
    unsigned char *mem_114831 = NULL;
    int64_t mem_114842_cached_sizze_116983 = 0;
    unsigned char *mem_114842 = NULL;
    int64_t mem_114847_cached_sizze_116984 = 0;
    unsigned char *mem_114847 = NULL;
    int64_t mem_114854_cached_sizze_116985 = 0;
    unsigned char *mem_114854 = NULL;
    int64_t mem_114861_cached_sizze_116986 = 0;
    unsigned char *mem_114861 = NULL;
    int64_t mem_114872_cached_sizze_116987 = 0;
    unsigned char *mem_114872 = NULL;
    int64_t mem_114877_cached_sizze_116988 = 0;
    unsigned char *mem_114877 = NULL;
    int64_t mem_114888_cached_sizze_116989 = 0;
    unsigned char *mem_114888 = NULL;
    int64_t mem_114893_cached_sizze_116990 = 0;
    unsigned char *mem_114893 = NULL;
    int64_t mem_114914_cached_sizze_116991 = 0;
    unsigned char *mem_114914 = NULL;
    int64_t mem_114919_cached_sizze_116992 = 0;
    unsigned char *mem_114919 = NULL;
    int64_t mem_114930_cached_sizze_116993 = 0;
    unsigned char *mem_114930 = NULL;
    int64_t mem_114935_cached_sizze_116994 = 0;
    unsigned char *mem_114935 = NULL;
    int64_t mem_114946_cached_sizze_116995 = 0;
    unsigned char *mem_114946 = NULL;
    int64_t mem_114951_cached_sizze_116996 = 0;
    unsigned char *mem_114951 = NULL;
    int64_t mem_114962_cached_sizze_116997 = 0;
    unsigned char *mem_114962 = NULL;
    int64_t mem_114963_cached_sizze_116998 = 0;
    unsigned char *mem_114963 = NULL;
    int64_t mem_114971_cached_sizze_116999 = 0;
    unsigned char *mem_114971 = NULL;
    int64_t mem_114978_cached_sizze_117000 = 0;
    unsigned char *mem_114978 = NULL;
    int64_t mem_114992_cached_sizze_117001 = 0;
    unsigned char *mem_114992 = NULL;
    int64_t mem_114997_cached_sizze_117002 = 0;
    unsigned char *mem_114997 = NULL;
    int64_t mem_115008_cached_sizze_117003 = 0;
    unsigned char *mem_115008 = NULL;
    int64_t mem_115013_cached_sizze_117004 = 0;
    unsigned char *mem_115013 = NULL;
    int64_t mem_115024_cached_sizze_117005 = 0;
    unsigned char *mem_115024 = NULL;
    int64_t mem_115029_cached_sizze_117006 = 0;
    unsigned char *mem_115029 = NULL;
    int64_t mem_115040_cached_sizze_117007 = 0;
    unsigned char *mem_115040 = NULL;
    int64_t mem_115045_cached_sizze_117008 = 0;
    unsigned char *mem_115045 = NULL;
    int64_t mem_115056_cached_sizze_117009 = 0;
    unsigned char *mem_115056 = NULL;
    int64_t mem_115061_cached_sizze_117010 = 0;
    unsigned char *mem_115061 = NULL;
    int64_t mem_115072_cached_sizze_117011 = 0;
    unsigned char *mem_115072 = NULL;
    int64_t mem_115073_cached_sizze_117012 = 0;
    unsigned char *mem_115073 = NULL;
    int64_t mem_115082_cached_sizze_117013 = 0;
    unsigned char *mem_115082 = NULL;
    int64_t mem_115087_cached_sizze_117014 = 0;
    unsigned char *mem_115087 = NULL;
    int64_t mem_115091_cached_sizze_117015 = 0;
    unsigned char *mem_115091 = NULL;
    int64_t mem_115098_cached_sizze_117016 = 0;
    unsigned char *mem_115098 = NULL;
    int64_t mem_115120_cached_sizze_117017 = 0;
    unsigned char *mem_115120 = NULL;
    int64_t mem_115125_cached_sizze_117018 = 0;
    unsigned char *mem_115125 = NULL;
    int64_t mem_115136_cached_sizze_117019 = 0;
    unsigned char *mem_115136 = NULL;
    int64_t mem_115137_cached_sizze_117020 = 0;
    unsigned char *mem_115137 = NULL;
    int64_t mem_115145_cached_sizze_117021 = 0;
    unsigned char *mem_115145 = NULL;
    int64_t mem_115159_cached_sizze_117022 = 0;
    unsigned char *mem_115159 = NULL;
    int64_t mem_115165_cached_sizze_117023 = 0;
    unsigned char *mem_115165 = NULL;
    int64_t mem_115170_cached_sizze_117024 = 0;
    unsigned char *mem_115170 = NULL;
    int64_t mem_115186_cached_sizze_117025 = 0;
    unsigned char *mem_115186 = NULL;
    int64_t mem_115191_cached_sizze_117026 = 0;
    unsigned char *mem_115191 = NULL;
    int64_t mem_115202_cached_sizze_117027 = 0;
    unsigned char *mem_115202 = NULL;
    int64_t mem_115207_cached_sizze_117028 = 0;
    unsigned char *mem_115207 = NULL;
    int64_t mem_115218_cached_sizze_117029 = 0;
    unsigned char *mem_115218 = NULL;
    int64_t mem_115223_cached_sizze_117030 = 0;
    unsigned char *mem_115223 = NULL;
    int64_t mem_115234_cached_sizze_117031 = 0;
    unsigned char *mem_115234 = NULL;
    int64_t mem_115239_cached_sizze_117032 = 0;
    unsigned char *mem_115239 = NULL;
    int64_t mem_115250_cached_sizze_117033 = 0;
    unsigned char *mem_115250 = NULL;
    int64_t mem_115251_cached_sizze_117034 = 0;
    unsigned char *mem_115251 = NULL;
    int64_t mem_115260_cached_sizze_117035 = 0;
    unsigned char *mem_115260 = NULL;
    int64_t mem_115261_cached_sizze_117036 = 0;
    unsigned char *mem_115261 = NULL;
    int64_t mem_115282_cached_sizze_117037 = 0;
    unsigned char *mem_115282 = NULL;
    int64_t mem_115287_cached_sizze_117038 = 0;
    unsigned char *mem_115287 = NULL;
    int64_t mem_115298_cached_sizze_117039 = 0;
    unsigned char *mem_115298 = NULL;
    int64_t mem_115303_cached_sizze_117040 = 0;
    unsigned char *mem_115303 = NULL;
    int64_t mem_115314_cached_sizze_117041 = 0;
    unsigned char *mem_115314 = NULL;
    int64_t mem_115319_cached_sizze_117042 = 0;
    unsigned char *mem_115319 = NULL;
    int64_t mem_115330_cached_sizze_117043 = 0;
    unsigned char *mem_115330 = NULL;
    int64_t mem_115331_cached_sizze_117044 = 0;
    unsigned char *mem_115331 = NULL;
    int64_t mem_115344_cached_sizze_117045 = 0;
    unsigned char *mem_115344 = NULL;
    int64_t mem_115351_cached_sizze_117046 = 0;
    unsigned char *mem_115351 = NULL;
    int64_t mem_115356_cached_sizze_117047 = 0;
    unsigned char *mem_115356 = NULL;
    int64_t mem_115367_cached_sizze_117048 = 0;
    unsigned char *mem_115367 = NULL;
    int64_t mem_115372_cached_sizze_117049 = 0;
    unsigned char *mem_115372 = NULL;
    int64_t mem_115383_cached_sizze_117050 = 0;
    unsigned char *mem_115383 = NULL;
    int64_t mem_115384_cached_sizze_117051 = 0;
    unsigned char *mem_115384 = NULL;
    int64_t mem_115393_cached_sizze_117052 = 0;
    unsigned char *mem_115393 = NULL;
    int64_t mem_115394_cached_sizze_117053 = 0;
    unsigned char *mem_115394 = NULL;
    int64_t mem_115415_cached_sizze_117054 = 0;
    unsigned char *mem_115415 = NULL;
    int64_t mem_115416_cached_sizze_117055 = 0;
    unsigned char *mem_115416 = NULL;
    int64_t mem_115427_cached_sizze_117056 = 0;
    unsigned char *mem_115427 = NULL;
    int64_t mem_115428_cached_sizze_117057 = 0;
    unsigned char *mem_115428 = NULL;
    int64_t mem_115437_cached_sizze_117058 = 0;
    unsigned char *mem_115437 = NULL;
    int64_t mem_115444_cached_sizze_117059 = 0;
    unsigned char *mem_115444 = NULL;
    int64_t mem_115469_cached_sizze_117060 = 0;
    unsigned char *mem_115469 = NULL;
    int64_t mem_115470_cached_sizze_117061 = 0;
    unsigned char *mem_115470 = NULL;
    int64_t mem_115471_cached_sizze_117062 = 0;
    unsigned char *mem_115471 = NULL;
    int64_t mem_115486_cached_sizze_117063 = 0;
    unsigned char *mem_115486 = NULL;
    int64_t mem_115487_cached_sizze_117064 = 0;
    unsigned char *mem_115487 = NULL;
    int64_t mem_115488_cached_sizze_117065 = 0;
    unsigned char *mem_115488 = NULL;
    int64_t mem_115500_cached_sizze_117066 = 0;
    unsigned char *mem_115500 = NULL;
    int64_t mem_115507_cached_sizze_117067 = 0;
    unsigned char *mem_115507 = NULL;
    int64_t mem_115514_cached_sizze_117068 = 0;
    unsigned char *mem_115514 = NULL;
    int64_t mem_115521_cached_sizze_117069 = 0;
    unsigned char *mem_115521 = NULL;
    int64_t mem_115553_cached_sizze_117070 = 0;
    unsigned char *mem_115553 = NULL;
    int64_t mem_115554_cached_sizze_117071 = 0;
    unsigned char *mem_115554 = NULL;
    int64_t mem_115555_cached_sizze_117072 = 0;
    unsigned char *mem_115555 = NULL;
    int64_t mem_115556_cached_sizze_117073 = 0;
    unsigned char *mem_115556 = NULL;
    int64_t mem_115557_cached_sizze_117074 = 0;
    unsigned char *mem_115557 = NULL;
    int64_t mem_115581_cached_sizze_117075 = 0;
    unsigned char *mem_115581 = NULL;
    int64_t mem_115582_cached_sizze_117076 = 0;
    unsigned char *mem_115582 = NULL;
    int64_t mem_115583_cached_sizze_117077 = 0;
    unsigned char *mem_115583 = NULL;
    int64_t mem_115584_cached_sizze_117078 = 0;
    unsigned char *mem_115584 = NULL;
    int64_t mem_115585_cached_sizze_117079 = 0;
    unsigned char *mem_115585 = NULL;
    int64_t mem_115604_cached_sizze_117080 = 0;
    unsigned char *mem_115604 = NULL;
    int64_t mem_115605_cached_sizze_117081 = 0;
    unsigned char *mem_115605 = NULL;
    int64_t mem_115618_cached_sizze_117082 = 0;
    unsigned char *mem_115618 = NULL;
    int64_t mem_115666_cached_sizze_117083 = 0;
    unsigned char *mem_115666 = NULL;
    int64_t mem_115672_cached_sizze_117084 = 0;
    unsigned char *mem_115672 = NULL;
    int64_t mem_115677_cached_sizze_117085 = 0;
    unsigned char *mem_115677 = NULL;
    int64_t mem_115693_cached_sizze_117086 = 0;
    unsigned char *mem_115693 = NULL;
    int64_t mem_115694_cached_sizze_117087 = 0;
    unsigned char *mem_115694 = NULL;
    int64_t mem_115703_cached_sizze_117088 = 0;
    unsigned char *mem_115703 = NULL;
    int64_t mem_115704_cached_sizze_117089 = 0;
    unsigned char *mem_115704 = NULL;
    int64_t mem_115725_cached_sizze_117090 = 0;
    unsigned char *mem_115725 = NULL;
    int64_t mem_115731_cached_sizze_117091 = 0;
    unsigned char *mem_115731 = NULL;
    int64_t mem_115736_cached_sizze_117092 = 0;
    unsigned char *mem_115736 = NULL;
    int64_t mem_115752_cached_sizze_117093 = 0;
    unsigned char *mem_115752 = NULL;
    int64_t mem_115757_cached_sizze_117094 = 0;
    unsigned char *mem_115757 = NULL;
    int64_t mem_115768_cached_sizze_117095 = 0;
    unsigned char *mem_115768 = NULL;
    int64_t mem_115774_cached_sizze_117096 = 0;
    unsigned char *mem_115774 = NULL;
    int64_t mem_115779_cached_sizze_117097 = 0;
    unsigned char *mem_115779 = NULL;
    int64_t mem_115795_cached_sizze_117098 = 0;
    unsigned char *mem_115795 = NULL;
    int64_t mem_115801_cached_sizze_117099 = 0;
    unsigned char *mem_115801 = NULL;
    int64_t mem_115806_cached_sizze_117100 = 0;
    unsigned char *mem_115806 = NULL;
    int64_t mem_115822_cached_sizze_117101 = 0;
    unsigned char *mem_115822 = NULL;
    int64_t mem_115823_cached_sizze_117102 = 0;
    unsigned char *mem_115823 = NULL;
    int64_t mem_115834_cached_sizze_117103 = 0;
    unsigned char *mem_115834 = NULL;
    int64_t mem_115835_cached_sizze_117104 = 0;
    unsigned char *mem_115835 = NULL;
    int64_t mem_115844_cached_sizze_117105 = 0;
    unsigned char *mem_115844 = NULL;
    int64_t mem_115845_cached_sizze_117106 = 0;
    unsigned char *mem_115845 = NULL;
    int64_t mem_115876_cached_sizze_117107 = 0;
    unsigned char *mem_115876 = NULL;
    int64_t mem_115877_cached_sizze_117108 = 0;
    unsigned char *mem_115877 = NULL;
    int64_t mem_115878_cached_sizze_117109 = 0;
    unsigned char *mem_115878 = NULL;
    int64_t mem_115891_cached_sizze_117110 = 0;
    unsigned char *mem_115891 = NULL;
    int64_t mem_115892_cached_sizze_117111 = 0;
    unsigned char *mem_115892 = NULL;
    int64_t mem_115893_cached_sizze_117112 = 0;
    unsigned char *mem_115893 = NULL;
    int64_t mem_115924_cached_sizze_117113 = 0;
    unsigned char *mem_115924 = NULL;
    int64_t mem_115925_cached_sizze_117114 = 0;
    unsigned char *mem_115925 = NULL;
    int64_t mem_115926_cached_sizze_117115 = 0;
    unsigned char *mem_115926 = NULL;
    int64_t mem_115927_cached_sizze_117116 = 0;
    unsigned char *mem_115927 = NULL;
    int64_t mem_115944_cached_sizze_117117 = 0;
    unsigned char *mem_115944 = NULL;
    int64_t mem_115945_cached_sizze_117118 = 0;
    unsigned char *mem_115945 = NULL;
    int64_t mem_115946_cached_sizze_117119 = 0;
    unsigned char *mem_115946 = NULL;
    int64_t mem_115947_cached_sizze_117120 = 0;
    unsigned char *mem_115947 = NULL;
    int64_t mem_115988_cached_sizze_117121 = 0;
    unsigned char *mem_115988 = NULL;
    int64_t mem_115993_cached_sizze_117122 = 0;
    unsigned char *mem_115993 = NULL;
    int64_t mem_116004_cached_sizze_117123 = 0;
    unsigned char *mem_116004 = NULL;
    int64_t mem_116005_cached_sizze_117124 = 0;
    unsigned char *mem_116005 = NULL;
    int64_t mem_116018_cached_sizze_117125 = 0;
    unsigned char *mem_116018 = NULL;
    int64_t mem_116025_cached_sizze_117126 = 0;
    unsigned char *mem_116025 = NULL;
    int64_t mem_116030_cached_sizze_117127 = 0;
    unsigned char *mem_116030 = NULL;
    int64_t mem_116041_cached_sizze_117128 = 0;
    unsigned char *mem_116041 = NULL;
    int64_t mem_116046_cached_sizze_117129 = 0;
    unsigned char *mem_116046 = NULL;
    int64_t mem_116057_cached_sizze_117130 = 0;
    unsigned char *mem_116057 = NULL;
    int64_t mem_116058_cached_sizze_117131 = 0;
    unsigned char *mem_116058 = NULL;
    int64_t mem_116071_cached_sizze_117132 = 0;
    unsigned char *mem_116071 = NULL;
    int64_t mem_116078_cached_sizze_117133 = 0;
    unsigned char *mem_116078 = NULL;
    int64_t mem_116079_cached_sizze_117134 = 0;
    unsigned char *mem_116079 = NULL;
    int64_t mem_116088_cached_sizze_117135 = 0;
    unsigned char *mem_116088 = NULL;
    int64_t mem_116089_cached_sizze_117136 = 0;
    unsigned char *mem_116089 = NULL;
    int64_t mem_116110_cached_sizze_117137 = 0;
    unsigned char *mem_116110 = NULL;
    int64_t mem_116115_cached_sizze_117138 = 0;
    unsigned char *mem_116115 = NULL;
    int64_t mem_116126_cached_sizze_117139 = 0;
    unsigned char *mem_116126 = NULL;
    int64_t mem_116127_cached_sizze_117140 = 0;
    unsigned char *mem_116127 = NULL;
    int64_t mem_116136_cached_sizze_117141 = 0;
    unsigned char *mem_116136 = NULL;
    int64_t mem_116137_cached_sizze_117142 = 0;
    unsigned char *mem_116137 = NULL;
    struct memblock mem_param_tmp_116490;
    
    mem_param_tmp_116490.references = NULL;
    
    struct memblock mem_param_tmp_116489;
    
    mem_param_tmp_116489.references = NULL;
    
    struct memblock mem_param_tmp_116488;
    
    mem_param_tmp_116488.references = NULL;
    
    struct memblock mem_param_tmp_116487;
    
    mem_param_tmp_116487.references = NULL;
    
    struct memblock mem_param_tmp_116486;
    
    mem_param_tmp_116486.references = NULL;
    
    struct memblock mem_param_tmp_116485;
    
    mem_param_tmp_116485.references = NULL;
    
    struct memblock mem_param_tmp_116484;
    
    mem_param_tmp_116484.references = NULL;
    
    struct memblock mem_param_tmp_116483;
    
    mem_param_tmp_116483.references = NULL;
    
    struct memblock mem_param_tmp_116482;
    
    mem_param_tmp_116482.references = NULL;
    
    struct memblock mem_param_tmp_116481;
    
    mem_param_tmp_116481.references = NULL;
    
    struct memblock mem_param_tmp_116480;
    
    mem_param_tmp_116480.references = NULL;
    
    struct memblock mem_param_tmp_116479;
    
    mem_param_tmp_116479.references = NULL;
    
    struct memblock mem_param_tmp_116478;
    
    mem_param_tmp_116478.references = NULL;
    
    struct memblock mem_param_tmp_116477;
    
    mem_param_tmp_116477.references = NULL;
    
    struct memblock mem_param_tmp_116476;
    
    mem_param_tmp_116476.references = NULL;
    
    struct memblock mem_param_tmp_116475;
    
    mem_param_tmp_116475.references = NULL;
    
    struct memblock mem_param_tmp_116474;
    
    mem_param_tmp_116474.references = NULL;
    
    struct memblock mem_param_tmp_116473;
    
    mem_param_tmp_116473.references = NULL;
    
    struct memblock mem_param_tmp_116472;
    
    mem_param_tmp_116472.references = NULL;
    
    struct memblock mem_param_tmp_116471;
    
    mem_param_tmp_116471.references = NULL;
    
    struct memblock mem_param_tmp_116470;
    
    mem_param_tmp_116470.references = NULL;
    
    struct memblock mem_param_tmp_116469;
    
    mem_param_tmp_116469.references = NULL;
    
    struct memblock mem_param_tmp_116468;
    
    mem_param_tmp_116468.references = NULL;
    
    struct memblock mem_param_tmp_116467;
    
    mem_param_tmp_116467.references = NULL;
    
    struct memblock mem_param_tmp_116466;
    
    mem_param_tmp_116466.references = NULL;
    
    struct memblock mem_param_tmp_116465;
    
    mem_param_tmp_116465.references = NULL;
    
    struct memblock mem_param_tmp_116464;
    
    mem_param_tmp_116464.references = NULL;
    
    struct memblock ext_mem_116254;
    
    ext_mem_116254.references = NULL;
    
    struct memblock ext_mem_116255;
    
    ext_mem_116255.references = NULL;
    
    struct memblock ext_mem_116256;
    
    ext_mem_116256.references = NULL;
    
    struct memblock mem_116252;
    
    mem_116252.references = NULL;
    
    struct memblock mem_116250;
    
    mem_116250.references = NULL;
    
    struct memblock mem_116248;
    
    mem_116248.references = NULL;
    
    struct memblock mem_116246;
    
    mem_116246.references = NULL;
    
    struct memblock ext_mem_116243;
    
    ext_mem_116243.references = NULL;
    
    struct memblock ext_mem_116244;
    
    ext_mem_116244.references = NULL;
    
    struct memblock ext_mem_116245;
    
    ext_mem_116245.references = NULL;
    
    struct memblock mem_116241;
    
    mem_116241.references = NULL;
    
    struct memblock mem_116239;
    
    mem_116239.references = NULL;
    
    struct memblock mem_116237;
    
    mem_116237.references = NULL;
    
    struct memblock mem_116235;
    
    mem_116235.references = NULL;
    
    struct memblock ext_mem_116232;
    
    ext_mem_116232.references = NULL;
    
    struct memblock ext_mem_116233;
    
    ext_mem_116233.references = NULL;
    
    struct memblock ext_mem_116234;
    
    ext_mem_116234.references = NULL;
    
    struct memblock mem_116230;
    
    mem_116230.references = NULL;
    
    struct memblock mem_116228;
    
    mem_116228.references = NULL;
    
    struct memblock mem_116226;
    
    mem_116226.references = NULL;
    
    struct memblock mem_116224;
    
    mem_116224.references = NULL;
    
    struct memblock ext_mem_116221;
    
    ext_mem_116221.references = NULL;
    
    struct memblock ext_mem_116222;
    
    ext_mem_116222.references = NULL;
    
    struct memblock ext_mem_116223;
    
    ext_mem_116223.references = NULL;
    
    struct memblock mem_116219;
    
    mem_116219.references = NULL;
    
    struct memblock mem_116217;
    
    mem_116217.references = NULL;
    
    struct memblock mem_116215;
    
    mem_116215.references = NULL;
    
    struct memblock mem_116213;
    
    mem_116213.references = NULL;
    
    struct memblock ext_mem_116210;
    
    ext_mem_116210.references = NULL;
    
    struct memblock ext_mem_116211;
    
    ext_mem_116211.references = NULL;
    
    struct memblock ext_mem_116212;
    
    ext_mem_116212.references = NULL;
    
    struct memblock mem_116208;
    
    mem_116208.references = NULL;
    
    struct memblock mem_116206;
    
    mem_116206.references = NULL;
    
    struct memblock mem_116204;
    
    mem_116204.references = NULL;
    
    struct memblock mem_116202;
    
    mem_116202.references = NULL;
    
    struct memblock ext_mem_116199;
    
    ext_mem_116199.references = NULL;
    
    struct memblock ext_mem_116200;
    
    ext_mem_116200.references = NULL;
    
    struct memblock ext_mem_116201;
    
    ext_mem_116201.references = NULL;
    
    struct memblock mem_116197;
    
    mem_116197.references = NULL;
    
    struct memblock mem_116195;
    
    mem_116195.references = NULL;
    
    struct memblock mem_116193;
    
    mem_116193.references = NULL;
    
    struct memblock mem_116191;
    
    mem_116191.references = NULL;
    
    struct memblock ext_mem_116188;
    
    ext_mem_116188.references = NULL;
    
    struct memblock ext_mem_116189;
    
    ext_mem_116189.references = NULL;
    
    struct memblock ext_mem_116190;
    
    ext_mem_116190.references = NULL;
    
    struct memblock mem_116186;
    
    mem_116186.references = NULL;
    
    struct memblock mem_116184;
    
    mem_116184.references = NULL;
    
    struct memblock mem_116182;
    
    mem_116182.references = NULL;
    
    struct memblock mem_116180;
    
    mem_116180.references = NULL;
    
    struct memblock ext_mem_116177;
    
    ext_mem_116177.references = NULL;
    
    struct memblock ext_mem_116178;
    
    ext_mem_116178.references = NULL;
    
    struct memblock ext_mem_116179;
    
    ext_mem_116179.references = NULL;
    
    struct memblock mem_116175;
    
    mem_116175.references = NULL;
    
    struct memblock mem_116173;
    
    mem_116173.references = NULL;
    
    struct memblock mem_116171;
    
    mem_116171.references = NULL;
    
    struct memblock mem_116169;
    
    mem_116169.references = NULL;
    
    struct memblock ext_mem_116166;
    
    ext_mem_116166.references = NULL;
    
    struct memblock ext_mem_116167;
    
    ext_mem_116167.references = NULL;
    
    struct memblock ext_mem_116168;
    
    ext_mem_116168.references = NULL;
    
    struct memblock mem_116164;
    
    mem_116164.references = NULL;
    
    struct memblock mem_116162;
    
    mem_116162.references = NULL;
    
    struct memblock mem_116160;
    
    mem_116160.references = NULL;
    
    struct memblock mem_116158;
    
    mem_116158.references = NULL;
    
    struct memblock mem_param_114560;
    
    mem_param_114560.references = NULL;
    
    struct memblock mem_param_114556;
    
    mem_param_114556.references = NULL;
    
    struct memblock mem_param_114552;
    
    mem_param_114552.references = NULL;
    
    struct memblock mem_param_114548;
    
    mem_param_114548.references = NULL;
    
    struct memblock mem_param_114544;
    
    mem_param_114544.references = NULL;
    
    struct memblock mem_param_114540;
    
    mem_param_114540.references = NULL;
    
    struct memblock mem_param_114536;
    
    mem_param_114536.references = NULL;
    
    struct memblock mem_param_114532;
    
    mem_param_114532.references = NULL;
    
    struct memblock mem_param_114528;
    
    mem_param_114528.references = NULL;
    
    struct memblock mem_param_114524;
    
    mem_param_114524.references = NULL;
    
    struct memblock mem_param_114520;
    
    mem_param_114520.references = NULL;
    
    struct memblock mem_param_114516;
    
    mem_param_114516.references = NULL;
    
    struct memblock mem_param_114512;
    
    mem_param_114512.references = NULL;
    
    struct memblock mem_param_114508;
    
    mem_param_114508.references = NULL;
    
    struct memblock mem_param_114504;
    
    mem_param_114504.references = NULL;
    
    struct memblock mem_param_114500;
    
    mem_param_114500.references = NULL;
    
    struct memblock mem_param_114496;
    
    mem_param_114496.references = NULL;
    
    struct memblock mem_param_114492;
    
    mem_param_114492.references = NULL;
    
    struct memblock mem_param_114488;
    
    mem_param_114488.references = NULL;
    
    struct memblock mem_param_114484;
    
    mem_param_114484.references = NULL;
    
    struct memblock mem_param_114480;
    
    mem_param_114480.references = NULL;
    
    struct memblock mem_param_114476;
    
    mem_param_114476.references = NULL;
    
    struct memblock mem_param_114472;
    
    mem_param_114472.references = NULL;
    
    struct memblock mem_param_114468;
    
    mem_param_114468.references = NULL;
    
    struct memblock mem_param_114464;
    
    mem_param_114464.references = NULL;
    
    struct memblock mem_param_114460;
    
    mem_param_114460.references = NULL;
    
    struct memblock mem_param_114456;
    
    mem_param_114456.references = NULL;
    
    struct memblock ext_mem_116338;
    
    ext_mem_116338.references = NULL;
    
    struct memblock ext_mem_116339;
    
    ext_mem_116339.references = NULL;
    
    struct memblock ext_mem_116340;
    
    ext_mem_116340.references = NULL;
    
    struct memblock ext_mem_116341;
    
    ext_mem_116341.references = NULL;
    
    struct memblock ext_mem_116342;
    
    ext_mem_116342.references = NULL;
    
    struct memblock ext_mem_116343;
    
    ext_mem_116343.references = NULL;
    
    struct memblock ext_mem_116344;
    
    ext_mem_116344.references = NULL;
    
    struct memblock ext_mem_116345;
    
    ext_mem_116345.references = NULL;
    
    struct memblock ext_mem_116346;
    
    ext_mem_116346.references = NULL;
    
    struct memblock ext_mem_116347;
    
    ext_mem_116347.references = NULL;
    
    struct memblock ext_mem_116348;
    
    ext_mem_116348.references = NULL;
    
    struct memblock ext_mem_116349;
    
    ext_mem_116349.references = NULL;
    
    struct memblock ext_mem_116350;
    
    ext_mem_116350.references = NULL;
    
    struct memblock ext_mem_116351;
    
    ext_mem_116351.references = NULL;
    
    struct memblock ext_mem_116352;
    
    ext_mem_116352.references = NULL;
    
    struct memblock ext_mem_116353;
    
    ext_mem_116353.references = NULL;
    
    struct memblock ext_mem_116354;
    
    ext_mem_116354.references = NULL;
    
    struct memblock ext_mem_116355;
    
    ext_mem_116355.references = NULL;
    
    struct memblock ext_mem_116356;
    
    ext_mem_116356.references = NULL;
    
    struct memblock ext_mem_116357;
    
    ext_mem_116357.references = NULL;
    
    struct memblock ext_mem_116358;
    
    ext_mem_116358.references = NULL;
    
    struct memblock ext_mem_116359;
    
    ext_mem_116359.references = NULL;
    
    struct memblock ext_mem_116360;
    
    ext_mem_116360.references = NULL;
    
    struct memblock ext_mem_116361;
    
    ext_mem_116361.references = NULL;
    
    struct memblock ext_mem_116362;
    
    ext_mem_116362.references = NULL;
    
    struct memblock ext_mem_116363;
    
    ext_mem_116363.references = NULL;
    
    struct memblock ext_mem_116364;
    
    ext_mem_116364.references = NULL;
    
    struct memblock mem_out_116463;
    
    mem_out_116463.references = NULL;
    
    struct memblock mem_out_116462;
    
    mem_out_116462.references = NULL;
    
    struct memblock mem_out_116461;
    
    mem_out_116461.references = NULL;
    
    struct memblock mem_out_116460;
    
    mem_out_116460.references = NULL;
    
    struct memblock mem_out_116459;
    
    mem_out_116459.references = NULL;
    
    struct memblock mem_out_116458;
    
    mem_out_116458.references = NULL;
    
    struct memblock mem_out_116457;
    
    mem_out_116457.references = NULL;
    
    struct memblock mem_out_116456;
    
    mem_out_116456.references = NULL;
    
    struct memblock mem_out_116455;
    
    mem_out_116455.references = NULL;
    
    struct memblock mem_out_116454;
    
    mem_out_116454.references = NULL;
    
    struct memblock mem_out_116453;
    
    mem_out_116453.references = NULL;
    
    struct memblock mem_out_116452;
    
    mem_out_116452.references = NULL;
    
    struct memblock mem_out_116451;
    
    mem_out_116451.references = NULL;
    
    struct memblock mem_out_116450;
    
    mem_out_116450.references = NULL;
    
    struct memblock mem_out_116449;
    
    mem_out_116449.references = NULL;
    
    struct memblock mem_out_116448;
    
    mem_out_116448.references = NULL;
    
    struct memblock mem_out_116447;
    
    mem_out_116447.references = NULL;
    
    struct memblock mem_out_116446;
    
    mem_out_116446.references = NULL;
    
    struct memblock mem_out_116445;
    
    mem_out_116445.references = NULL;
    
    struct memblock mem_out_116444;
    
    mem_out_116444.references = NULL;
    
    struct memblock mem_out_116443;
    
    mem_out_116443.references = NULL;
    
    struct memblock mem_out_116442;
    
    mem_out_116442.references = NULL;
    
    struct memblock mem_out_116441;
    
    mem_out_116441.references = NULL;
    
    struct memblock mem_out_116440;
    
    mem_out_116440.references = NULL;
    
    struct memblock mem_out_116439;
    
    mem_out_116439.references = NULL;
    
    struct memblock mem_out_116438;
    
    mem_out_116438.references = NULL;
    
    struct memblock mem_out_116437;
    
    mem_out_116437.references = NULL;
    
    struct memblock mem_114414 = ctx->constants->mem_114414;
    struct memblock mem_114415 = ctx->constants->mem_114415;
    struct memblock mem_114416 = ctx->constants->mem_114416;
    struct memblock mem_114417 = ctx->constants->mem_114417;
    struct memblock mem_114418 = ctx->constants->mem_114418;
    struct memblock mem_114419 = ctx->constants->mem_114419;
    struct memblock mem_114420 = ctx->constants->mem_114420;
    struct memblock mem_114421 = ctx->constants->mem_114421;
    struct memblock mem_114422 = ctx->constants->mem_114422;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_114561_cached_sizze_116948 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114561, &mem_114561_cached_sizze_116948, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114562_cached_sizze_116949 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_114562, &mem_114562_cached_sizze_116949, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114571_cached_sizze_116950 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_114571, &mem_114571_cached_sizze_116950, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114578_cached_sizze_116951 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114578, &mem_114578_cached_sizze_116951, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114593_cached_sizze_116952 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114593, &mem_114593_cached_sizze_116952, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114594_cached_sizze_116953 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114594, &mem_114594_cached_sizze_116953, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114602_cached_sizze_116954 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114602, &mem_114602_cached_sizze_116954, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114609_cached_sizze_116955 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114609, &mem_114609_cached_sizze_116955, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114623_cached_sizze_116956 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114623, &mem_114623_cached_sizze_116956, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114624_cached_sizze_116957 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114624, &mem_114624_cached_sizze_116957, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114632_cached_sizze_116958 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114632, &mem_114632_cached_sizze_116958, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114639_cached_sizze_116959 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114639, &mem_114639_cached_sizze_116959, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114653_cached_sizze_116960 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114653, &mem_114653_cached_sizze_116960, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114654_cached_sizze_116961 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114654, &mem_114654_cached_sizze_116961, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114655_cached_sizze_116962 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114655, &mem_114655_cached_sizze_116962, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114668_cached_sizze_116963 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114668, &mem_114668_cached_sizze_116963, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114669_cached_sizze_116964 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114669, &mem_114669_cached_sizze_116964, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114670_cached_sizze_116965 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114670, &mem_114670_cached_sizze_116965, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114701_cached_sizze_116966 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114701, &mem_114701_cached_sizze_116966, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114702_cached_sizze_116967 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114702, &mem_114702_cached_sizze_116967, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114703_cached_sizze_116968 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114703, &mem_114703_cached_sizze_116968, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114719_cached_sizze_116969 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114719, &mem_114719_cached_sizze_116969, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114720_cached_sizze_116970 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114720, &mem_114720_cached_sizze_116970, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114721_cached_sizze_116971 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114721, &mem_114721_cached_sizze_116971, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114734_cached_sizze_116972 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_114734, &mem_114734_cached_sizze_116972, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114735_cached_sizze_116973 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_114735, &mem_114735_cached_sizze_116973, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114736_cached_sizze_116974 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_114736, &mem_114736_cached_sizze_116974, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114782_cached_sizze_116975 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_114782, &mem_114782_cached_sizze_116975, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114783_cached_sizze_116976 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114783, &mem_114783_cached_sizze_116976, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114794_cached_sizze_116977 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114794, &mem_114794_cached_sizze_116977, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114795_cached_sizze_116978 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114795, &mem_114795_cached_sizze_116978, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114804_cached_sizze_116979 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114804, &mem_114804_cached_sizze_116979, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114805_cached_sizze_116980 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114805, &mem_114805_cached_sizze_116980, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114826_cached_sizze_116981 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114826, &mem_114826_cached_sizze_116981, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114831_cached_sizze_116982 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114831, &mem_114831_cached_sizze_116982, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114842_cached_sizze_116983 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114842, &mem_114842_cached_sizze_116983, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114847_cached_sizze_116984 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114847, &mem_114847_cached_sizze_116984, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114854_cached_sizze_116985 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114854, &mem_114854_cached_sizze_116985, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114861_cached_sizze_116986 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114861, &mem_114861_cached_sizze_116986, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114872_cached_sizze_116987 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114872, &mem_114872_cached_sizze_116987, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114877_cached_sizze_116988 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_114877, &mem_114877_cached_sizze_116988, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114888_cached_sizze_116989 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114888, &mem_114888_cached_sizze_116989, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114893_cached_sizze_116990 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_114893, &mem_114893_cached_sizze_116990, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114914_cached_sizze_116991 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114914, &mem_114914_cached_sizze_116991, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114919_cached_sizze_116992 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114919, &mem_114919_cached_sizze_116992, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114930_cached_sizze_116993 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114930, &mem_114930_cached_sizze_116993, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114935_cached_sizze_116994 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114935, &mem_114935_cached_sizze_116994, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114946_cached_sizze_116995 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114946, &mem_114946_cached_sizze_116995, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114951_cached_sizze_116996 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114951, &mem_114951_cached_sizze_116996, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114962_cached_sizze_116997 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114962, &mem_114962_cached_sizze_116997, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114963_cached_sizze_116998 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_114963, &mem_114963_cached_sizze_116998, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114971_cached_sizze_116999 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114971, &mem_114971_cached_sizze_116999, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114978_cached_sizze_117000 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_114978, &mem_114978_cached_sizze_117000, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114992_cached_sizze_117001 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_114992, &mem_114992_cached_sizze_117001, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_114997_cached_sizze_117002 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_114997, &mem_114997_cached_sizze_117002, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115008_cached_sizze_117003 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_115008, &mem_115008_cached_sizze_117003, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115013_cached_sizze_117004 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_115013, &mem_115013_cached_sizze_117004, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115024_cached_sizze_117005 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115024, &mem_115024_cached_sizze_117005, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115029_cached_sizze_117006 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115029, &mem_115029_cached_sizze_117006, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115040_cached_sizze_117007 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115040, &mem_115040_cached_sizze_117007, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115045_cached_sizze_117008 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115045, &mem_115045_cached_sizze_117008, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115056_cached_sizze_117009 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_115056, &mem_115056_cached_sizze_117009, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115061_cached_sizze_117010 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_115061, &mem_115061_cached_sizze_117010, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115072_cached_sizze_117011 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_115072, &mem_115072_cached_sizze_117011, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115073_cached_sizze_117012 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115073, &mem_115073_cached_sizze_117012, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115082_cached_sizze_117013 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_115082, &mem_115082_cached_sizze_117013, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115087_cached_sizze_117014 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_115087, &mem_115087_cached_sizze_117014, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115120_cached_sizze_117017 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_115120, &mem_115120_cached_sizze_117017, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115125_cached_sizze_117018 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_115125, &mem_115125_cached_sizze_117018, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115136_cached_sizze_117019 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_115136, &mem_115136_cached_sizze_117019, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115137_cached_sizze_117020 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115137, &mem_115137_cached_sizze_117020, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115145_cached_sizze_117021 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_115145, &mem_115145_cached_sizze_117021, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115159_cached_sizze_117022 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_115159, &mem_115159_cached_sizze_117022, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115165_cached_sizze_117023 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_115165, &mem_115165_cached_sizze_117023, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115170_cached_sizze_117024 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_115170, &mem_115170_cached_sizze_117024, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115186_cached_sizze_117025 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_115186, &mem_115186_cached_sizze_117025, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115191_cached_sizze_117026 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_115191, &mem_115191_cached_sizze_117026, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115202_cached_sizze_117027 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_115202, &mem_115202_cached_sizze_117027, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115207_cached_sizze_117028 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_115207, &mem_115207_cached_sizze_117028, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115218_cached_sizze_117029 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115218, &mem_115218_cached_sizze_117029, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115223_cached_sizze_117030 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115223, &mem_115223_cached_sizze_117030, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115234_cached_sizze_117031 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115234, &mem_115234_cached_sizze_117031, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115239_cached_sizze_117032 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115239, &mem_115239_cached_sizze_117032, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115250_cached_sizze_117033 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_115250, &mem_115250_cached_sizze_117033, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115251_cached_sizze_117034 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_115251, &mem_115251_cached_sizze_117034, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115260_cached_sizze_117035 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_115260, &mem_115260_cached_sizze_117035, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115261_cached_sizze_117036 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_115261, &mem_115261_cached_sizze_117036, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115282_cached_sizze_117037 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_115282, &mem_115282_cached_sizze_117037, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115287_cached_sizze_117038 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_115287, &mem_115287_cached_sizze_117038, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115298_cached_sizze_117039 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115298, &mem_115298_cached_sizze_117039, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115303_cached_sizze_117040 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115303, &mem_115303_cached_sizze_117040, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115314_cached_sizze_117041 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115314, &mem_115314_cached_sizze_117041, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115319_cached_sizze_117042 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115319, &mem_115319_cached_sizze_117042, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115330_cached_sizze_117043 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115330, &mem_115330_cached_sizze_117043, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115331_cached_sizze_117044 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115331, &mem_115331_cached_sizze_117044, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115344_cached_sizze_117045 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115344, &mem_115344_cached_sizze_117045, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115351_cached_sizze_117046 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115351, &mem_115351_cached_sizze_117046, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115356_cached_sizze_117047 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115356, &mem_115356_cached_sizze_117047, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115367_cached_sizze_117048 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115367, &mem_115367_cached_sizze_117048, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115372_cached_sizze_117049 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115372, &mem_115372_cached_sizze_117049, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115383_cached_sizze_117050 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115383, &mem_115383_cached_sizze_117050, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115384_cached_sizze_117051 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115384, &mem_115384_cached_sizze_117051, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115393_cached_sizze_117052 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115393, &mem_115393_cached_sizze_117052, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115394_cached_sizze_117053 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115394, &mem_115394_cached_sizze_117053, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115415_cached_sizze_117054 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_115415, &mem_115415_cached_sizze_117054, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115416_cached_sizze_117055 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115416, &mem_115416_cached_sizze_117055, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115427_cached_sizze_117056 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115427, &mem_115427_cached_sizze_117056, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115428_cached_sizze_117057 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_115428, &mem_115428_cached_sizze_117057, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115437_cached_sizze_117058 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_115437, &mem_115437_cached_sizze_117058, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115444_cached_sizze_117059 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115444, &mem_115444_cached_sizze_117059, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115469_cached_sizze_117060 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_115469, &mem_115469_cached_sizze_117060, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115470_cached_sizze_117061 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115470, &mem_115470_cached_sizze_117061, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115471_cached_sizze_117062 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_115471, &mem_115471_cached_sizze_117062, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115486_cached_sizze_117063 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115486, &mem_115486_cached_sizze_117063, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115487_cached_sizze_117064 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_115487, &mem_115487_cached_sizze_117064, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115488_cached_sizze_117065 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115488, &mem_115488_cached_sizze_117065, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115500_cached_sizze_117066 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115500, &mem_115500_cached_sizze_117066, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115507_cached_sizze_117067 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115507, &mem_115507_cached_sizze_117067, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115514_cached_sizze_117068 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115514, &mem_115514_cached_sizze_117068, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115521_cached_sizze_117069 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_115521, &mem_115521_cached_sizze_117069, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115553_cached_sizze_117070 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115553, &mem_115553_cached_sizze_117070, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115554_cached_sizze_117071 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_115554, &mem_115554_cached_sizze_117071, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115555_cached_sizze_117072 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_115555, &mem_115555_cached_sizze_117072, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115556_cached_sizze_117073 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_115556, &mem_115556_cached_sizze_117073, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115557_cached_sizze_117074 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_115557, &mem_115557_cached_sizze_117074, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115581_cached_sizze_117075 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_115581, &mem_115581_cached_sizze_117075, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115582_cached_sizze_117076 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115582, &mem_115582_cached_sizze_117076, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115583_cached_sizze_117077 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115583, &mem_115583_cached_sizze_117077, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115584_cached_sizze_117078 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115584, &mem_115584_cached_sizze_117078, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115585_cached_sizze_117079 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115585, &mem_115585_cached_sizze_117079, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115604_cached_sizze_117080 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115604, &mem_115604_cached_sizze_117080, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115605_cached_sizze_117081 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115605, &mem_115605_cached_sizze_117081, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115618_cached_sizze_117082 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_115618, &mem_115618_cached_sizze_117082, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115666_cached_sizze_117083 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_115666, &mem_115666_cached_sizze_117083, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115672_cached_sizze_117084 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115672, &mem_115672_cached_sizze_117084, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115677_cached_sizze_117085 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115677, &mem_115677_cached_sizze_117085, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115693_cached_sizze_117086 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_115693, &mem_115693_cached_sizze_117086, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115694_cached_sizze_117087 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_115694, &mem_115694_cached_sizze_117087, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115703_cached_sizze_117088 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115703, &mem_115703_cached_sizze_117088, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115704_cached_sizze_117089 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115704, &mem_115704_cached_sizze_117089, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115725_cached_sizze_117090 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_115725, &mem_115725_cached_sizze_117090, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115731_cached_sizze_117091 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115731, &mem_115731_cached_sizze_117091, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115736_cached_sizze_117092 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115736, &mem_115736_cached_sizze_117092, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115752_cached_sizze_117093 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_115752, &mem_115752_cached_sizze_117093, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115757_cached_sizze_117094 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115757, &mem_115757_cached_sizze_117094, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115768_cached_sizze_117095 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_115768, &mem_115768_cached_sizze_117095, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115774_cached_sizze_117096 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115774, &mem_115774_cached_sizze_117096, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115779_cached_sizze_117097 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115779, &mem_115779_cached_sizze_117097, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115795_cached_sizze_117098 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_115795, &mem_115795_cached_sizze_117098, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115801_cached_sizze_117099 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115801, &mem_115801_cached_sizze_117099, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115806_cached_sizze_117100 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115806, &mem_115806_cached_sizze_117100, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115822_cached_sizze_117101 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115822, &mem_115822_cached_sizze_117101, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115823_cached_sizze_117102 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115823, &mem_115823_cached_sizze_117102, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115834_cached_sizze_117103 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_115834, &mem_115834_cached_sizze_117103, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115835_cached_sizze_117104 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_115835, &mem_115835_cached_sizze_117104, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115844_cached_sizze_117105 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_115844, &mem_115844_cached_sizze_117105, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115845_cached_sizze_117106 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_115845, &mem_115845_cached_sizze_117106, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115876_cached_sizze_117107 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115876, &mem_115876_cached_sizze_117107, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115877_cached_sizze_117108 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115877, &mem_115877_cached_sizze_117108, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115878_cached_sizze_117109 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115878, &mem_115878_cached_sizze_117109, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115891_cached_sizze_117110 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115891, &mem_115891_cached_sizze_117110, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115892_cached_sizze_117111 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115892, &mem_115892_cached_sizze_117111, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115893_cached_sizze_117112 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115893, &mem_115893_cached_sizze_117112, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115924_cached_sizze_117113 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115924, &mem_115924_cached_sizze_117113, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115925_cached_sizze_117114 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115925, &mem_115925_cached_sizze_117114, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115926_cached_sizze_117115 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115926, &mem_115926_cached_sizze_117115, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115927_cached_sizze_117116 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115927, &mem_115927_cached_sizze_117116, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115944_cached_sizze_117117 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115944, &mem_115944_cached_sizze_117117, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115945_cached_sizze_117118 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115945, &mem_115945_cached_sizze_117118, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115946_cached_sizze_117119 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115946, &mem_115946_cached_sizze_117119, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115947_cached_sizze_117120 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115947, &mem_115947_cached_sizze_117120, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115988_cached_sizze_117121 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_115988, &mem_115988_cached_sizze_117121, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_115993_cached_sizze_117122 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_115993, &mem_115993_cached_sizze_117122, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116004_cached_sizze_117123 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_116004, &mem_116004_cached_sizze_117123, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116005_cached_sizze_117124 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_116005, &mem_116005_cached_sizze_117124, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116018_cached_sizze_117125 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_116018, &mem_116018_cached_sizze_117125, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116025_cached_sizze_117126 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_116025, &mem_116025_cached_sizze_117126, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116030_cached_sizze_117127 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_116030, &mem_116030_cached_sizze_117127, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116041_cached_sizze_117128 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_116041, &mem_116041_cached_sizze_117128, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116046_cached_sizze_117129 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_116046, &mem_116046_cached_sizze_117129, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116057_cached_sizze_117130 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_116057, &mem_116057_cached_sizze_117130, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116058_cached_sizze_117131 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_116058, &mem_116058_cached_sizze_117131, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116071_cached_sizze_117132 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_116071, &mem_116071_cached_sizze_117132, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116078_cached_sizze_117133 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_116078, &mem_116078_cached_sizze_117133, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116079_cached_sizze_117134 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_116079, &mem_116079_cached_sizze_117134, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116088_cached_sizze_117135 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_116088, &mem_116088_cached_sizze_117135, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116089_cached_sizze_117136 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_116089, &mem_116089_cached_sizze_117136, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116110_cached_sizze_117137 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_116110, &mem_116110_cached_sizze_117137, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116115_cached_sizze_117138 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_116115, &mem_116115_cached_sizze_117138, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116126_cached_sizze_117139 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_116126, &mem_116126_cached_sizze_117139, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116127_cached_sizze_117140 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_116127, &mem_116127_cached_sizze_117140, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116136_cached_sizze_117141 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_116136, &mem_116136_cached_sizze_117141, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_116137_cached_sizze_117142 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_116137, &mem_116137_cached_sizze_117142, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:607:5-612:51
    if (memblock_set(ctx, &mem_param_114456, &wdown_mem_114423, "wdown_mem_114423") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114460, &wkey_mem_114424, "wkey_mem_114424") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114464, &wout_mem_114425, "wout_mem_114425") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114468, &wpe_mem_114426, "wpe_mem_114426") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114472, &wqry_mem_114427, "wqry_mem_114427") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114476, &wte_mem_114428, "wte_mem_114428") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114480, &wup_mem_114429, "wup_mem_114429") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114484, &wval_mem_114430, "wval_mem_114430") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114488, &wvoc_mem_114431, "wvoc_mem_114431") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114492, &wdown_mem_114432, "wdown_mem_114432") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114496, &wkey_mem_114433, "wkey_mem_114433") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114500, &wout_mem_114434, "wout_mem_114434") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114504, &wpe_mem_114435, "wpe_mem_114435") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114508, &wqry_mem_114436, "wqry_mem_114436") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114512, &wte_mem_114437, "wte_mem_114437") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114516, &wup_mem_114438, "wup_mem_114438") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114520, &wval_mem_114439, "wval_mem_114439") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114524, &wvoc_mem_114440, "wvoc_mem_114440") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114528, &wdown_mem_114441, "wdown_mem_114441") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114532, &wkey_mem_114442, "wkey_mem_114442") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114536, &wout_mem_114443, "wout_mem_114443") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114540, &wpe_mem_114444, "wpe_mem_114444") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114544, &wqry_mem_114445, "wqry_mem_114445") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114548, &wte_mem_114446, "wte_mem_114446") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114552, &wup_mem_114447, "wup_mem_114447") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114556, &wval_mem_114448, "wval_mem_114448") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_114560, &wvoc_mem_114449, "wvoc_mem_114449") != 0)
        return 1;
    for (int64_t step_106053 = 0; step_106053 < (int64_t) 500; step_106053++) {
        // futhark/microgpt.fut:609:16-25
        
        int64_t dl_106081 = ((int64_t *) dls_mem_114451.mem)[step_106053];
        
        // futhark/microgpt.fut:449:37-40
        
        int64_t zl_rhs_106086 = sub64(dl_106081, (int64_t) 1);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113422 = 0; i_113422 < (int64_t) 16; i_113422++) {
            // futhark/microgpt.fut:449:25-81
            
            bool cond_108665 = slt64(i_113422, zl_rhs_106086);
            
            // futhark/microgpt.fut:449:56-59
            
            int64_t zeze_lhs_108666 = add64((int64_t) 1, i_113422);
            
            // futhark/microgpt.fut:449:47-60
            
            bool x_108667 = sle64((int64_t) 0, zeze_lhs_108666);
            
            // futhark/microgpt.fut:449:47-60
            
            bool y_108668 = slt64(zeze_lhs_108666, (int64_t) 16);
            
            // futhark/microgpt.fut:449:47-60
            
            bool bounds_check_108669 = x_108667 && y_108668;
            
            // futhark/microgpt.fut:9:27-39
            
            bool loop_not_taken_108670 = !cond_108665;
            
            // futhark/microgpt.fut:9:27-39
            
            bool protect_assert_disj_108671 = bounds_check_108669 || loop_not_taken_108670;
            
            // futhark/microgpt.fut:449:47-60
            
            bool index_certs_108672;
            
            if (!protect_assert_disj_108671) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_108666, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:449:47-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:449:3-83\n   #6  futhark/microgpt.fut:556:18-38\n   #7  futhark/microgpt.fut:578:26-584:31\n   #8  futhark/microgpt.fut:612:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            int64_t tmp_108687 = ((int64_t *) seqs_mem_114452.mem)[step_106053 * (int64_t) 16 + i_113422];
            
            // futhark/microgpt.fut:558:37-51
            
            bool x_108688 = sle64((int64_t) 0, tmp_108687);
            
            // futhark/microgpt.fut:558:37-51
            
            bool y_108689 = slt64(tmp_108687, (int64_t) 27);
            
            // futhark/microgpt.fut:558:37-51
            
            bool bounds_check_108690 = x_108688 && y_108689;
            
            // futhark/microgpt.fut:558:37-51
            
            bool index_certs_108691;
            
            if (!bounds_check_108690) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_108687, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:558:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:558:16-55\n   #6  futhark/microgpt.fut:578:26-584:31\n   #7  futhark/microgpt.fut:612:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:449:47-60
            
            int64_t zeze_lhs_108673;
            
            if (cond_108665) {
                int64_t x_113147 = ((int64_t *) seqs_mem_114452.mem)[step_106053 * (int64_t) 16 + zeze_lhs_108666];
                
                zeze_lhs_108673 = x_113147;
            } else {
                zeze_lhs_108673 = (int64_t) 0;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113412 = 0; i_113412 < (int64_t) 27; i_113412++) {
                // futhark/microgpt.fut:449:61-65
                
                bool cond_t_res_108677 = zeze_lhs_108673 == i_113412;
                
                // futhark/microgpt.fut:9:27-39
                
                bool x_108678 = cond_108665 && cond_t_res_108677;
                
                // futhark/microgpt.fut:449:25-81
                
                double lifted_lambda_res_108679;
                
                if (x_108678) {
                    lifted_lambda_res_108679 = 1.0;
                } else {
                    lifted_lambda_res_108679 = 0.0;
                }
                ((double *) mem_114571)[i_113412] = lifted_lambda_res_108679;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113416 = 0; i_113416 < (int64_t) 16; i_113416++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_108698 = ((double *) mem_param_114476.mem)[tmp_108687 * (int64_t) 16 + i_113416];
                
                ((double *) mem_114578)[i_113416] = lifted_lambda_res_108698;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114561, i_113422 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114578, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114562, i_113422 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114571, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113437 = 0; i_113437 < (int64_t) 16; i_113437++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_108717;
            double r_108719 = 0.0;
            
            for (int64_t i_108718 = 0; i_108718 < (int64_t) 16; i_108718++) {
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_108720 = ((double *) mem_param_114468.mem)[i_113437 * (int64_t) 16 + i_108718];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_rhs_108721 = ((double *) mem_114561)[i_113437 * (int64_t) 16 + i_108718];
                
                // futhark/microgpt.fut:279:71-107
                
                double zp_res_108722 = zp_lhs_108720 + zp_rhs_108721;
                
                // futhark/microgpt.fut:279:87-150
                
                double zt_res_108723 = zp_res_108722 * zp_res_108722;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_108724 = r_108719 + zt_res_108723;
                double r_tmp_116524 = zp_res_108724;
                
                r_108719 = r_tmp_116524;
            }
            defunc_0_lifted_lambda_res_108717 = r_108719;
            // futhark/microgpt.fut:279:50-169
            
            double zs_res_108725 = defunc_0_lifted_lambda_res_108717 / 16.0;
            
            // futhark/microgpt.fut:280:23-53
            
            double zp_res_108726 = 1.0e-5 + zs_res_108725;
            
            // futhark/microgpt.fut:280:15-53
            
            double sqrt_res_108727 = futrts_sqrt64(zp_res_108726);
            
            // futhark/microgpt.fut:281:79-89
            
            double zs_res_108728 = 1.0 / sqrt_res_108727;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113427 = 0; i_113427 < (int64_t) 16; i_113427++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_108735 = ((double *) mem_param_114468.mem)[i_113437 * (int64_t) 16 + i_113427];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_108736 = ((double *) mem_114561)[i_113437 * (int64_t) 16 + i_113427];
                
                // futhark/microgpt.fut:281:36-72
                
                double zp_res_108737 = zp_lhs_108735 + zp_rhs_108736;
                
                // futhark/microgpt.fut:281:52-89
                
                double zt_res_108738 = zs_res_108728 * zp_res_108737;
                
                ((double *) mem_114602)[i_113427] = zt_res_108738;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113431 = 0; i_113431 < (int64_t) 16; i_113431++) {
                // futhark/microgpt.fut:282:4-12
                
                double lifted_lambda_res_108746 = ((double *) mem_114602)[i_113431];
                
                ((double *) mem_114609)[i_113431] = lifted_lambda_res_108746;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_108754;
            double r_108756 = 0.0;
            
            for (int64_t i_108755 = 0; i_108755 < (int64_t) 16; i_108755++) {
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_108757 = ((double *) mem_param_114468.mem)[i_113437 * (int64_t) 16 + i_108755];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_rhs_108758 = ((double *) mem_114561)[i_113437 * (int64_t) 16 + i_108755];
                
                // futhark/microgpt.fut:377:59-103
                
                double zp_res_108759 = zp_lhs_108757 + zp_rhs_108758;
                
                // futhark/microgpt.fut:377:79-154
                
                double zt_res_108760 = zp_res_108759 * zp_res_108759;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_108761 = r_108756 + zt_res_108760;
                double r_tmp_116527 = zp_res_108761;
                
                r_108756 = r_tmp_116527;
            }
            defunc_0_lifted_lambda_res_108754 = r_108756;
            // futhark/microgpt.fut:377:36-173
            
            double zs_res_108762 = defunc_0_lifted_lambda_res_108754 / 16.0;
            
            ((double *) mem_114593)[i_113437] = zs_res_108762;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114594, i_113437 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114609, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113452 = 0; i_113452 < (int64_t) 16; i_113452++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_108780;
            double r_108782 = 0.0;
            
            for (int64_t i_108781 = 0; i_108781 < (int64_t) 16; i_108781++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_108783 = ((double *) mem_114594)[i_113452 * (int64_t) 16 + i_108781];
                
                // futhark/microgpt.fut:283:71-106
                
                double zt_res_108784 = zt_lhs_108783 * zt_lhs_108783;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_108785 = r_108782 + zt_res_108784;
                double r_tmp_116530 = zp_res_108785;
                
                r_108782 = r_tmp_116530;
            }
            defunc_0_lifted_lambda_res_108780 = r_108782;
            // futhark/microgpt.fut:283:50-124
            
            double zs_res_108786 = defunc_0_lifted_lambda_res_108780 / 16.0;
            
            // futhark/microgpt.fut:284:24-54
            
            double zp_res_108787 = 1.0e-5 + zs_res_108786;
            
            // futhark/microgpt.fut:284:16-54
            
            double sqrt_res_108788 = futrts_sqrt64(zp_res_108787);
            
            // futhark/microgpt.fut:285:58-69
            
            double zs_res_108789 = 1.0 / sqrt_res_108788;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113442 = 0; i_113442 < (int64_t) 16; i_113442++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_108796 = ((double *) mem_114594)[i_113452 * (int64_t) 16 + i_113442];
                
                // futhark/microgpt.fut:285:37-69
                
                double zt_res_108797 = zs_res_108789 * zt_lhs_108796;
                
                ((double *) mem_114632)[i_113442] = zt_res_108797;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113446 = 0; i_113446 < (int64_t) 16; i_113446++) {
                // futhark/microgpt.fut:286:4-13
                
                double lifted_lambda_res_108805 = ((double *) mem_114632)[i_113446];
                
                ((double *) mem_114639)[i_113446] = lifted_lambda_res_108805;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_108813;
            double r_108815 = 0.0;
            
            for (int64_t i_108814 = 0; i_108814 < (int64_t) 16; i_108814++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_108816 = ((double *) mem_114594)[i_113452 * (int64_t) 16 + i_108814];
                
                // futhark/microgpt.fut:371:58-99
                
                double zt_res_108817 = zt_lhs_108816 * zt_lhs_108816;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_108818 = r_108815 + zt_res_108817;
                double r_tmp_116533 = zp_res_108818;
                
                r_108815 = r_tmp_116533;
            }
            defunc_0_lifted_lambda_res_108813 = r_108815;
            // futhark/microgpt.fut:371:36-117
            
            double zs_res_108819 = defunc_0_lifted_lambda_res_108813 / 16.0;
            
            ((double *) mem_114623)[i_113452] = zs_res_108819;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114624, i_113452 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114639, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113471 = 0; i_113471 < (int64_t) 16; i_113471++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113461 = 0; i_113461 < (int64_t) 16; i_113461++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_110798;
                double r_110800 = 0.0;
                
                for (int64_t i_110799 = 0; i_110799 < (int64_t) 16; i_110799++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_110801 = ((double *) mem_param_114472.mem)[i_113461 * (int64_t) 16 + i_110799];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_110802 = ((double *) mem_114624)[i_113471 * (int64_t) 16 + i_110799];
                    
                    // futhark/microgpt.fut:287:63-102
                    
                    double zt_res_110803 = zt_lhs_110801 * zt_rhs_110802;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_110804 = r_110800 + zt_res_110803;
                    double r_tmp_116540 = zp_res_110804;
                    
                    r_110800 = r_tmp_116540;
                }
                defunc_0_lifted_lambda_res_110798 = r_110800;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_110811;
                double r_110813 = 0.0;
                
                for (int64_t i_110812 = 0; i_110812 < (int64_t) 16; i_110812++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_110814 = ((double *) mem_param_114460.mem)[i_113461 * (int64_t) 16 + i_110812];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_110815 = ((double *) mem_114624)[i_113471 * (int64_t) 16 + i_110812];
                    
                    // futhark/microgpt.fut:288:63-102
                    
                    double zt_res_110816 = zt_lhs_110814 * zt_rhs_110815;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_110817 = r_110813 + zt_res_110816;
                    double r_tmp_116541 = zp_res_110817;
                    
                    r_110813 = r_tmp_116541;
                }
                defunc_0_lifted_lambda_res_110811 = r_110813;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_110827;
                double r_110829 = 0.0;
                
                for (int64_t i_110828 = 0; i_110828 < (int64_t) 16; i_110828++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_110830 = ((double *) mem_param_114484.mem)[i_113461 * (int64_t) 16 + i_110828];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_110831 = ((double *) mem_114624)[i_113471 * (int64_t) 16 + i_110828];
                    
                    // futhark/microgpt.fut:289:63-102
                    
                    double zt_res_110832 = zt_lhs_110830 * zt_rhs_110831;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_110833 = r_110829 + zt_res_110832;
                    double r_tmp_116542 = zp_res_110833;
                    
                    r_110829 = r_tmp_116542;
                }
                defunc_0_lifted_lambda_res_110827 = r_110829;
                ((double *) mem_114668)[i_113461] = defunc_0_lifted_lambda_res_110827;
                ((double *) mem_114669)[i_113461] = defunc_0_lifted_lambda_res_110811;
                ((double *) mem_114670)[i_113461] = defunc_0_lifted_lambda_res_110798;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114653, i_113471 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114668, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114654, i_113471 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114669, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114655, i_113471 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114670, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113501 = 0; i_113501 < (int64_t) 4; i_113501++) {
            // futhark/microgpt.fut:290:67-70
            
            int64_t zp_lhs_109020 = mul64((int64_t) 4, i_113501);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113491 = 0; i_113491 < (int64_t) 16; i_113491++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113481 = 0; i_113481 < (int64_t) 4; i_113481++) {
                    // futhark/microgpt.fut:290:72-79
                    
                    int64_t tmp_110991 = add64(zp_lhs_109020, i_113481);
                    
                    // futhark/microgpt.fut:290:48-81
                    
                    bool x_110992 = sle64((int64_t) 0, tmp_110991);
                    
                    // futhark/microgpt.fut:290:48-81
                    
                    bool y_110993 = slt64(tmp_110991, (int64_t) 16);
                    
                    // futhark/microgpt.fut:290:48-81
                    
                    bool bounds_check_110994 = x_110992 && y_110993;
                    
                    // futhark/microgpt.fut:290:48-81
                    
                    bool index_certs_110995;
                    
                    if (!bounds_check_110994) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_110991, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:290:48-81\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:290:12-82\n   #9  futhark/microgpt.fut:561:5-76\n   #10 futhark/microgpt.fut:578:26-584:31\n   #11 futhark/microgpt.fut:612:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_110996 = ((double *) mem_114655)[i_113491 * (int64_t) 16 + tmp_110991];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_111004 = ((double *) mem_114654)[i_113491 * (int64_t) 16 + tmp_110991];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_111015 = ((double *) mem_114653)[i_113491 * (int64_t) 16 + tmp_110991];
                    
                    ((double *) mem_114734)[i_113481] = lifted_lambda_res_111015;
                    ((double *) mem_114735)[i_113481] = lifted_lambda_res_111004;
                    ((double *) mem_114736)[i_113481] = lifted_lambda_res_110996;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_114719, i_113491 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114734, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_114720, i_113491 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114735, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_114721, i_113491 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114736, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_114701, i_113501 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_114719, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_114702, i_113501 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_114720, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_114703, i_113501 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_114721, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113565 = 0; i_113565 < (int64_t) 4; i_113565++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113516 = 0; i_113516 < (int64_t) 16; i_113516++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113509 = 0; i_113509 < (int64_t) 16; i_113509++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_111094;
                    double r_111096 = 0.0;
                    
                    for (int64_t i_111095 = 0; i_111095 < (int64_t) 4; i_111095++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_111097 = ((double *) mem_114703)[i_113565 * (int64_t) 64 + i_113516 * (int64_t) 4 + i_111095];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_111098 = ((double *) mem_114702)[i_113565 * (int64_t) 64 + i_113509 * (int64_t) 4 + i_111095];
                        
                        // futhark/microgpt.fut:293:110-163
                        
                        double zt_res_111099 = zt_lhs_111097 * zt_rhs_111098;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_111100 = r_111096 + zt_res_111099;
                        double r_tmp_116558 = zp_res_111100;
                        
                        r_111096 = r_tmp_116558;
                    }
                    defunc_0_lifted_lambda_res_111094 = r_111096;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_111107;
                    double r_111109 = 0.0;
                    
                    for (int64_t i_111108 = 0; i_111108 < (int64_t) 4; i_111108++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_111110 = ((double *) mem_114703)[i_113565 * (int64_t) 64 + i_113516 * (int64_t) 4 + i_111108];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_111111 = ((double *) mem_114702)[i_113565 * (int64_t) 64 + i_113509 * (int64_t) 4 + i_111108];
                        
                        // futhark/microgpt.fut:344:75-134
                        
                        double zt_res_111112 = zt_lhs_111110 * zt_rhs_111111;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_111113 = r_111109 + zt_res_111112;
                        double r_tmp_116559 = zp_res_111113;
                        
                        r_111109 = r_tmp_116559;
                    }
                    defunc_0_lifted_lambda_res_111107 = r_111109;
                    ((double *) mem_114804)[i_113509] = defunc_0_lifted_lambda_res_111107;
                    ((double *) mem_114805)[i_113509] = defunc_0_lifted_lambda_res_111094;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_114794, i_113516 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114804, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_114795, i_113516 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114805, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113525 = 0; i_113525 < (int64_t) 16; i_113525++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113521 = 0; i_113521 < (int64_t) 16; i_113521++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_109129 = ((double *) mem_114795)[i_113525 * (int64_t) 16 + i_113521];
                    
                    // futhark/microgpt.fut:294:47-78
                    
                    double zs_res_109130 = zs_lhs_109129 / 2.0;
                    double zp_rhs_109131 = ((double *) masks_mem_114450.mem)[step_106053 * (int64_t) 256 + i_113525 * (int64_t) 16 + i_113521];
                    
                    // futhark/microgpt.fut:294:65-102
                    
                    double zp_res_109132 = zs_res_109130 + zp_rhs_109131;
                    
                    ((double *) mem_114831)[i_113521] = zp_res_109132;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_114826, i_113525 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114831, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113543 = 0; i_113543 < (int64_t) 16; i_113543++) {
                // futhark/microgpt.fut:115:13-33
                
                double defunc_0_reduce_res_113166;
                double redout_113527 = -INFINITY;
                
                for (int64_t i_113528 = 0; i_113528 < (int64_t) 16; i_113528++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_111131 = ((double *) mem_114826)[i_113543 * (int64_t) 16 + i_113528];
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_109153 = fmax64(lifted_lambda_res_111131, redout_113527);
                    double redout_tmp_116563 = max_res_109153;
                    
                    redout_113527 = redout_tmp_116563;
                }
                defunc_0_reduce_res_113166 = redout_113527;
                // futhark/microgpt.fut:296:67-76
                
                double neg_res_109154 = -defunc_0_reduce_res_113166;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113531 = 0; i_113531 < (int64_t) 16; i_113531++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_109161 = ((double *) mem_114826)[i_113543 * (int64_t) 16 + i_113531];
                    
                    // futhark/microgpt.fut:296:44-76
                    
                    double zp_res_109162 = neg_res_109154 + zp_lhs_109161;
                    
                    // futhark/microgpt.fut:296:37-76
                    
                    double exp_res_109163 = futrts_exp64(zp_res_109162);
                    
                    ((double *) mem_114847)[i_113531] = exp_res_109163;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_109165;
                double r_109167 = 0.0;
                
                for (int64_t i_109166 = 0; i_109166 < (int64_t) 16; i_109166++) {
                    // futhark/microgpt.fut:297:36-46
                    
                    double lifted_lambda_res_109168 = ((double *) mem_114847)[i_109166];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_109169 = r_109167 + lifted_lambda_res_109168;
                    double r_tmp_116565 = zp_res_109169;
                    
                    r_109167 = r_tmp_116565;
                }
                defunc_0_lifted_lambda_res_109165 = r_109167;
                // futhark/microgpt.fut:298:53-64
                
                double zs_res_109170 = 1.0 / defunc_0_lifted_lambda_res_109165;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113535 = 0; i_113535 < (int64_t) 16; i_113535++) {
                    // futhark/microgpt.fut:298:37-47
                    
                    double zt_lhs_109177 = ((double *) mem_114847)[i_113535];
                    
                    // futhark/microgpt.fut:298:37-64
                    
                    double zt_res_109178 = zs_res_109170 * zt_lhs_109177;
                    
                    ((double *) mem_114854)[i_113535] = zt_res_109178;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113539 = 0; i_113539 < (int64_t) 16; i_113539++) {
                    // futhark/microgpt.fut:299:4-14
                    
                    double lifted_lambda_res_109186 = ((double *) mem_114854)[i_113539];
                    
                    ((double *) mem_114861)[i_113539] = lifted_lambda_res_109186;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_114842, i_113543 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114861, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113551 = 0; i_113551 < (int64_t) 16; i_113551++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113547 = 0; i_113547 < (int64_t) 4; i_113547++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_109201;
                    double r_109203 = 0.0;
                    
                    for (int64_t i_109202 = 0; i_109202 < (int64_t) 16; i_109202++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_109204 = ((double *) mem_114842)[i_113551 * (int64_t) 16 + i_109202];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_109205 = ((double *) mem_114701)[i_113565 * (int64_t) 64 + i_109202 * (int64_t) 4 + i_113547];
                        
                        // futhark/microgpt.fut:300:66-112
                        
                        double zt_res_109206 = zt_lhs_109204 * zt_rhs_109205;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_109207 = r_109203 + zt_res_109206;
                        double r_tmp_116570 = zp_res_109207;
                        
                        r_109203 = r_tmp_116570;
                    }
                    defunc_0_lifted_lambda_res_109201 = r_109203;
                    ((double *) mem_114877)[i_113547] = defunc_0_lifted_lambda_res_109201;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_114872, i_113551 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114877, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113559 = 0; i_113559 < (int64_t) 16; i_113559++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113555 = 0; i_113555 < (int64_t) 4; i_113555++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_109222 = ((double *) mem_114872)[i_113559 * (int64_t) 4 + i_113555];
                    
                    ((double *) mem_114893)[i_113555] = lifted_lambda_res_109222;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_114888, i_113559 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114893, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_114782, i_113565 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_114794, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_114783, i_113565 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_114888, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113574 = 0; i_113574 < (int64_t) 16; i_113574++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113570 = 0; i_113570 < (int64_t) 16; i_113570++) {
                // futhark/microgpt.fut:302:52-55
                
                int64_t tmp_106466 = sdiv64(i_113570, (int64_t) 4);
                
                // futhark/microgpt.fut:302:41-57
                
                bool x_106467 = sle64((int64_t) 0, tmp_106466);
                
                // futhark/microgpt.fut:302:41-57
                
                bool y_106468 = slt64(tmp_106466, (int64_t) 4);
                
                // futhark/microgpt.fut:302:41-57
                
                bool bounds_check_106469 = x_106467 && y_106468;
                
                // futhark/microgpt.fut:302:41-57
                
                bool index_certs_106470;
                
                if (!bounds_check_106469) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_106466, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:302:41-57\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:302:12-78\n   #6  futhark/microgpt.fut:561:5-76\n   #7  futhark/microgpt.fut:578:26-584:31\n   #8  futhark/microgpt.fut:612:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:302:72-75
                
                int64_t tmp_106471 = smod64(i_113570, (int64_t) 4);
                
                // futhark/microgpt.fut:302:41-77
                
                bool x_106472 = sle64((int64_t) 0, tmp_106471);
                
                // futhark/microgpt.fut:302:41-77
                
                bool y_106473 = slt64(tmp_106471, (int64_t) 4);
                
                // futhark/microgpt.fut:302:41-77
                
                bool bounds_check_106474 = x_106472 && y_106473;
                
                // futhark/microgpt.fut:302:41-77
                
                bool index_certs_106475;
                
                if (!bounds_check_106474) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_106471, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:302:41-77\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:302:12-78\n   #6  futhark/microgpt.fut:561:5-76\n   #7  futhark/microgpt.fut:578:26-584:31\n   #8  futhark/microgpt.fut:612:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_106476 = ((double *) mem_114783)[tmp_106466 * (int64_t) 64 + i_113574 * (int64_t) 4 + tmp_106471];
                
                ((double *) mem_114919)[i_113570] = lifted_lambda_res_106476;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114914, i_113574 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114919, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113582 = 0; i_113582 < (int64_t) 16; i_113582++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113578 = 0; i_113578 < (int64_t) 16; i_113578++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_106491;
                double r_106493 = 0.0;
                
                for (int64_t i_106492 = 0; i_106492 < (int64_t) 16; i_106492++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_106494 = ((double *) mem_param_114464.mem)[i_113578 * (int64_t) 16 + i_106492];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_106495 = ((double *) mem_114914)[i_113582 * (int64_t) 16 + i_106492];
                    
                    // futhark/microgpt.fut:303:63-103
                    
                    double zt_res_106496 = zt_lhs_106494 * zt_rhs_106495;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_106497 = r_106493 + zt_res_106496;
                    double r_tmp_116577 = zp_res_106497;
                    
                    r_106493 = r_tmp_116577;
                }
                defunc_0_lifted_lambda_res_106491 = r_106493;
                ((double *) mem_114935)[i_113578] = defunc_0_lifted_lambda_res_106491;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114930, i_113582 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114935, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113590 = 0; i_113590 < (int64_t) 16; i_113590++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113586 = 0; i_113586 < (int64_t) 16; i_113586++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_106512 = ((double *) mem_114930)[i_113590 * (int64_t) 16 + i_113586];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_106513 = ((double *) mem_114594)[i_113590 * (int64_t) 16 + i_113586];
                
                // futhark/microgpt.fut:304:42-80
                
                double zp_res_106514 = zp_lhs_106512 + zp_rhs_106513;
                
                ((double *) mem_114951)[i_113586] = zp_res_106514;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114946, i_113590 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114951, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113604 = 0; i_113604 < (int64_t) 16; i_113604++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_109269;
            double r_109271 = 0.0;
            
            for (int64_t i_109270 = 0; i_109270 < (int64_t) 16; i_109270++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_109272 = ((double *) mem_114946)[i_113604 * (int64_t) 16 + i_109270];
                
                // futhark/microgpt.fut:305:75-114
                
                double zt_res_109273 = zt_lhs_109272 * zt_lhs_109272;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_109274 = r_109271 + zt_res_109273;
                double r_tmp_116582 = zp_res_109274;
                
                r_109271 = r_tmp_116582;
            }
            defunc_0_lifted_lambda_res_109269 = r_109271;
            // futhark/microgpt.fut:305:54-132
            
            double zs_res_109275 = defunc_0_lifted_lambda_res_109269 / 16.0;
            
            // futhark/microgpt.fut:306:24-55
            
            double zp_res_109276 = 1.0e-5 + zs_res_109275;
            
            // futhark/microgpt.fut:306:16-55
            
            double sqrt_res_109277 = futrts_sqrt64(zp_res_109276);
            
            // futhark/microgpt.fut:307:60-71
            
            double zs_res_109278 = 1.0 / sqrt_res_109277;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113594 = 0; i_113594 < (int64_t) 16; i_113594++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_109285 = ((double *) mem_114946)[i_113604 * (int64_t) 16 + i_113594];
                
                // futhark/microgpt.fut:307:37-71
                
                double zt_res_109286 = zs_res_109278 * zt_lhs_109285;
                
                ((double *) mem_114971)[i_113594] = zt_res_109286;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113598 = 0; i_113598 < (int64_t) 16; i_113598++) {
                // futhark/microgpt.fut:308:4-14
                
                double lifted_lambda_res_109294 = ((double *) mem_114971)[i_113598];
                
                ((double *) mem_114978)[i_113598] = lifted_lambda_res_109294;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_109302;
            double r_109304 = 0.0;
            
            for (int64_t i_109303 = 0; i_109303 < (int64_t) 16; i_109303++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_109305 = ((double *) mem_114946)[i_113604 * (int64_t) 16 + i_109303];
                
                // futhark/microgpt.fut:335:58-101
                
                double zt_res_109306 = zt_lhs_109305 * zt_lhs_109305;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_109307 = r_109304 + zt_res_109306;
                double r_tmp_116585 = zp_res_109307;
                
                r_109304 = r_tmp_116585;
            }
            defunc_0_lifted_lambda_res_109302 = r_109304;
            // futhark/microgpt.fut:335:36-119
            
            double zs_res_109308 = defunc_0_lifted_lambda_res_109302 / 16.0;
            
            ((double *) mem_114962)[i_113604] = zs_res_109308;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114963, i_113604 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114978, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113613 = 0; i_113613 < (int64_t) 16; i_113613++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113609 = 0; i_113609 < (int64_t) 64; i_113609++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_106564;
                double r_106566 = 0.0;
                
                for (int64_t i_106565 = 0; i_106565 < (int64_t) 16; i_106565++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_106567 = ((double *) mem_param_114480.mem)[i_113609 * (int64_t) 16 + i_106565];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_106568 = ((double *) mem_114963)[i_113613 * (int64_t) 16 + i_106565];
                    
                    // futhark/microgpt.fut:309:63-102
                    
                    double zt_res_106569 = zt_lhs_106567 * zt_rhs_106568;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_106570 = r_106566 + zt_res_106569;
                    double r_tmp_116588 = zp_res_106570;
                    
                    r_106566 = r_tmp_116588;
                }
                defunc_0_lifted_lambda_res_106564 = r_106566;
                ((double *) mem_114997)[i_113609] = defunc_0_lifted_lambda_res_106564;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_114992, i_113613 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_114997, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113621 = 0; i_113621 < (int64_t) 16; i_113621++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113617 = 0; i_113617 < (int64_t) 64; i_113617++) {
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_106585 = ((double *) mem_114992)[i_113621 * (int64_t) 64 + i_113617];
                
                // futhark/microgpt.fut:310:41-69
                
                double max_res_106586 = fmax64(0.0, max_arg0_106585);
                
                ((double *) mem_115013)[i_113617] = max_res_106586;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115008, i_113621 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115013, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113629 = 0; i_113629 < (int64_t) 16; i_113629++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113625 = 0; i_113625 < (int64_t) 16; i_113625++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_106601;
                double r_106603 = 0.0;
                
                for (int64_t i_106602 = 0; i_106602 < (int64_t) 64; i_106602++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_106604 = ((double *) mem_param_114456.mem)[i_113625 * (int64_t) 64 + i_106602];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_106605 = ((double *) mem_115008)[i_113629 * (int64_t) 64 + i_106602];
                    
                    // futhark/microgpt.fut:311:63-104
                    
                    double zt_res_106606 = zt_lhs_106604 * zt_rhs_106605;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_106607 = r_106603 + zt_res_106606;
                    double r_tmp_116593 = zp_res_106607;
                    
                    r_106603 = r_tmp_116593;
                }
                defunc_0_lifted_lambda_res_106601 = r_106603;
                ((double *) mem_115029)[i_113625] = defunc_0_lifted_lambda_res_106601;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115024, i_113629 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115029, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113637 = 0; i_113637 < (int64_t) 16; i_113637++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113633 = 0; i_113633 < (int64_t) 16; i_113633++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_106622 = ((double *) mem_115024)[i_113637 * (int64_t) 16 + i_113633];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_106623 = ((double *) mem_114946)[i_113637 * (int64_t) 16 + i_113633];
                
                // futhark/microgpt.fut:312:42-81
                
                double zp_res_106624 = zp_lhs_106622 + zp_rhs_106623;
                
                ((double *) mem_115045)[i_113633] = zp_res_106624;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115040, i_113637 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115045, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113645 = 0; i_113645 < (int64_t) 16; i_113645++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113641 = 0; i_113641 < (int64_t) 27; i_113641++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_106639;
                double r_106641 = 0.0;
                
                for (int64_t i_106640 = 0; i_106640 < (int64_t) 16; i_106640++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_106642 = ((double *) mem_param_114488.mem)[i_113641 * (int64_t) 16 + i_106640];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_106643 = ((double *) mem_115040)[i_113645 * (int64_t) 16 + i_106640];
                    
                    // futhark/microgpt.fut:313:63-103
                    
                    double zt_res_106644 = zt_lhs_106642 * zt_rhs_106643;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_106645 = r_106641 + zt_res_106644;
                    double r_tmp_116598 = zp_res_106645;
                    
                    r_106641 = r_tmp_116598;
                }
                defunc_0_lifted_lambda_res_106639 = r_106641;
                ((double *) mem_115061)[i_113641] = defunc_0_lifted_lambda_res_106639;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115056, i_113645 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115061, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113670 = 0; i_113670 < (int64_t) 16; i_113670++) {
            // futhark/microgpt.fut:4:11-25
            
            double defunc_0_reduce_res_113249;
            double redout_113662 = -INFINITY;
            
            for (int64_t i_113664 = 0; i_113664 < (int64_t) 27; i_113664++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_111200 = ((double *) mem_115056)[i_113670 * (int64_t) 27 + i_113664];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113659 = 0; i_113659 < (int64_t) 27; i_113659++) {
                    // futhark/microgpt.fut:319:51-323:90
                    
                    bool cond_111209 = i_113659 == i_113664;
                    
                    // futhark/microgpt.fut:319:51-323:90
                    
                    double lifted_lambda_res_111210;
                    
                    if (cond_111209) {
                        // futhark/microgpt.fut:115:13-33
                        
                        double defunc_0_reduce_res_113196;
                        double redout_113647 = -INFINITY;
                        
                        for (int64_t i_113648 = 0; i_113648 < (int64_t) 27; i_113648++) {
                            // futhark/microgpt.fut:4:11-25
                            
                            double lifted_lambda_res_113202 = ((double *) mem_115056)[i_113670 * (int64_t) 27 + i_113648];
                            
                            // futhark/microgpt.fut:115:13-33
                            
                            double max_res_113205 = fmax64(lifted_lambda_res_113202, redout_113647);
                            double redout_tmp_116604 = max_res_113205;
                            
                            redout_113647 = redout_tmp_116604;
                        }
                        defunc_0_reduce_res_113196 = redout_113647;
                        // futhark/microgpt.fut:320:67-76
                        
                        double neg_res_113207 = -defunc_0_reduce_res_113196;
                        
                        // futhark/microgpt.fut:4:11-25
                        if (mem_115091_cached_sizze_117015 < (int64_t) 216) {
                            err = lexical_realloc(ctx, &mem_115091, &mem_115091_cached_sizze_117015, (int64_t) 216);
                            if (err != FUTHARK_SUCCESS)
                                goto cleanup;
                        }
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_113651 = 0; i_113651 < (int64_t) 27; i_113651++) {
                            // futhark/microgpt.fut:4:11-25
                            
                            double zp_lhs_113214 = ((double *) mem_115056)[i_113670 * (int64_t) 27 + i_113651];
                            
                            // futhark/microgpt.fut:320:44-76
                            
                            double zp_res_113215 = neg_res_113207 + zp_lhs_113214;
                            
                            // futhark/microgpt.fut:320:37-76
                            
                            double exp_res_113216 = futrts_exp64(zp_res_113215);
                            
                            ((double *) mem_115091)[i_113651] = exp_res_113216;
                        }
                        // futhark/microgpt.fut:71:13-49
                        
                        double defunc_0_lifted_lambda_res_113219;
                        double r_113221 = 0.0;
                        
                        for (int64_t i_113220 = 0; i_113220 < (int64_t) 27; i_113220++) {
                            // futhark/microgpt.fut:321:36-46
                            
                            double lifted_lambda_res_113222 = ((double *) mem_115091)[i_113220];
                            
                            // futhark/microgpt.fut:71:40-49
                            
                            double zp_res_113223 = r_113221 + lifted_lambda_res_113222;
                            double r_tmp_116606 = zp_res_113223;
                            
                            r_113221 = r_tmp_116606;
                        }
                        defunc_0_lifted_lambda_res_113219 = r_113221;
                        // futhark/microgpt.fut:322:53-64
                        
                        double zs_res_113224 = 1.0 / defunc_0_lifted_lambda_res_113219;
                        
                        // futhark/microgpt.fut:4:11-25
                        if (mem_115098_cached_sizze_117016 < (int64_t) 216) {
                            err = lexical_realloc(ctx, &mem_115098, &mem_115098_cached_sizze_117016, (int64_t) 216);
                            if (err != FUTHARK_SUCCESS)
                                goto cleanup;
                        }
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_113655 = 0; i_113655 < (int64_t) 27; i_113655++) {
                            // futhark/microgpt.fut:322:37-47
                            
                            double zt_lhs_113231 = ((double *) mem_115091)[i_113655];
                            
                            // futhark/microgpt.fut:322:37-64
                            
                            double zt_res_113232 = zs_res_113224 * zt_lhs_113231;
                            
                            ((double *) mem_115098)[i_113655] = zt_res_113232;
                        }
                        // futhark/microgpt.fut:4:11-25
                        
                        double zt_rhs_113239 = ((double *) mem_114562)[i_113670 * (int64_t) 27 + i_113664];
                        
                        // futhark/microgpt.fut:323:7-49
                        
                        double zt_res_113240 = -6.25e-2 * zt_rhs_113239;
                        
                        // futhark/microgpt.fut:323:64-74
                        
                        double zs_rhs_113245 = ((double *) mem_115098)[i_113659];
                        
                        // futhark/microgpt.fut:323:56-74
                        
                        double zs_res_113246 = 1.0 / zs_rhs_113245;
                        
                        // futhark/microgpt.fut:323:25-74
                        
                        double zt_res_113247 = zt_res_113240 * zs_res_113246;
                        
                        lifted_lambda_res_111210 = zt_res_113247;
                    } else {
                        lifted_lambda_res_111210 = 0.0;
                    }
                    ((double *) mem_115087)[i_113659] = lifted_lambda_res_111210;
                }
                // futhark/microgpt.fut:115:13-33
                
                double max_res_109338 = fmax64(lifted_lambda_res_111200, redout_113662);
                
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_115082, i_113664 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115087, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
                
                double redout_tmp_116601 = max_res_109338;
                
                redout_113662 = redout_tmp_116601;
            }
            defunc_0_reduce_res_113249 = redout_113662;
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_115072, i_113670 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_115082, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
            ((double *) mem_115073)[i_113670] = defunc_0_reduce_res_113249;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113679 = 0; i_113679 < (int64_t) 16; i_113679++) {
            // futhark/microgpt.fut:317:78-88
            
            double neg_arg0_106674 = ((double *) mem_115073)[i_113679];
            
            // futhark/microgpt.fut:317:72-88
            
            double neg_res_106675 = -neg_arg0_106674;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113675 = 0; i_113675 < (int64_t) 27; i_113675++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_106682 = ((double *) mem_115056)[i_113679 * (int64_t) 27 + i_113675];
                
                // futhark/microgpt.fut:317:49-88
                
                double zp_res_106683 = neg_res_106675 + zp_lhs_106682;
                
                // futhark/microgpt.fut:317:42-88
                
                double exp_res_106684 = futrts_exp64(zp_res_106683);
                
                ((double *) mem_115125)[i_113675] = exp_res_106684;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115120, i_113679 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115125, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113689 = 0; i_113689 < (int64_t) 16; i_113689++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_108622;
            double r_108624 = 0.0;
            
            for (int64_t i_108623 = 0; i_108623 < (int64_t) 27; i_108623++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_108625 = ((double *) mem_115120)[i_113689 * (int64_t) 27 + i_108623];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_108626 = r_108624 + lifted_lambda_res_108625;
                double r_tmp_116612 = zp_res_108626;
                
                r_108624 = r_tmp_116612;
            }
            defunc_0_lifted_lambda_res_108622 = r_108624;
            // futhark/microgpt.fut:324:133-158
            
            double zt_res_108634 = defunc_0_lifted_lambda_res_108622 * defunc_0_lifted_lambda_res_108622;
            
            // futhark/microgpt.fut:324:124-158
            
            double zs_res_108635 = 1.0 / zt_res_108634;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113683 = 0; i_113683 < (int64_t) 27; i_113683++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_108642;
                double r_108644 = 0.0;
                
                for (int64_t i_108643 = 0; i_108643 < (int64_t) 27; i_108643++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_108645 = ((double *) mem_115072)[i_113689 * (int64_t) 729 + i_113683 * (int64_t) 27 + i_108643];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_108646 = ((double *) mem_115120)[i_113689 * (int64_t) 27 + i_108643];
                    
                    // futhark/microgpt.fut:324:71-117
                    
                    double zt_res_108647 = zt_lhs_108645 * zt_rhs_108646;
                    
                    // futhark/microgpt.fut:324:96-158
                    
                    double zt_res_108648 = zs_res_108635 * zt_res_108647;
                    
                    // futhark/microgpt.fut:324:63-158
                    
                    double neg_res_108649 = -zt_res_108648;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_108650 = r_108644 + neg_res_108649;
                    double r_tmp_116614 = zp_res_108650;
                    
                    r_108644 = r_tmp_116614;
                }
                defunc_0_lifted_lambda_res_108642 = r_108644;
                ((double *) mem_115145)[i_113683] = defunc_0_lifted_lambda_res_108642;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115136, i_113689 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115145, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            ((double *) mem_115137)[i_113689] = defunc_0_lifted_lambda_res_108622;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113702 = 0; i_113702 < (int64_t) 16; i_113702++) {
            // futhark/microgpt.fut:325:97-108
            
            double zs_rhs_106799 = ((double *) mem_115137)[i_113702];
            
            // futhark/microgpt.fut:325:89-108
            
            double zs_res_106800 = 1.0 / zs_rhs_106799;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113698 = 0; i_113698 < (int64_t) 27; i_113698++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_106807 = ((double *) mem_115136)[i_113702 * (int64_t) 27 + i_113698];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113694 = 0; i_113694 < (int64_t) 27; i_113694++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_106814 = ((double *) mem_115072)[i_113702 * (int64_t) 729 + i_113698 * (int64_t) 27 + i_113694];
                    
                    // futhark/microgpt.fut:325:56-108
                    
                    double zt_res_106815 = zs_res_106800 * zt_lhs_106814;
                    
                    // futhark/microgpt.fut:325:84-134
                    
                    double zp_res_106816 = zp_rhs_106807 + zt_res_106815;
                    
                    ((double *) mem_115170)[i_113694] = zp_res_106816;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_115165, i_113698 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115170, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_115159, i_113702 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_115165, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113710 = 0; i_113710 < (int64_t) 16; i_113710++) {
            double f_elem_106822 = ((double *) mem_115073)[i_113710];
            
            // futhark/microgpt.fut:326:107-124
            
            double neg_res_106827 = -f_elem_106822;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113706 = 0; i_113706 < (int64_t) 27; i_113706++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_106834;
                double r_106836 = 0.0;
                
                for (int64_t i_106835 = 0; i_106835 < (int64_t) 27; i_106835++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_106837 = ((double *) mem_115056)[i_113710 * (int64_t) 27 + i_106835];
                    
                    // futhark/microgpt.fut:326:82-124
                    
                    double zp_res_106838 = neg_res_106827 + zp_lhs_106837;
                    
                    // futhark/microgpt.fut:326:75-124
                    
                    double exp_res_106839 = futrts_exp64(zp_res_106838);
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_106840 = ((double *) mem_115159)[i_113710 * (int64_t) 729 + i_113706 * (int64_t) 27 + i_106835];
                    
                    // futhark/microgpt.fut:326:75-160
                    
                    double zt_res_106841 = exp_res_106839 * zt_rhs_106840;
                    
                    // futhark/microgpt.fut:326:67-160
                    
                    double neg_res_106842 = -zt_res_106841;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_106843 = r_106836 + neg_res_106842;
                    double r_tmp_116620 = zp_res_106843;
                    
                    r_106836 = r_tmp_116620;
                }
                defunc_0_lifted_lambda_res_106834 = r_106836;
                ((double *) mem_115191)[i_113706] = defunc_0_lifted_lambda_res_106834;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115186, i_113710 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115191, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113718 = 0; i_113718 < (int64_t) 16; i_113718++) {
            double f_elem_108533 = ((double *) mem_115073)[i_113718];
            
            // futhark/microgpt.fut:328:126-144
            
            double neg_res_108551 = -f_elem_108533;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_108552;
            double r_108554 = 0.0;
            
            for (int64_t i_108553 = 0; i_108553 < (int64_t) 27; i_108553++) {
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_108555 = ((double *) mem_115056)[i_113718 * (int64_t) 27 + i_108553];
                
                // futhark/microgpt.fut:328:101-144
                
                double zp_res_108556 = neg_res_108551 + zp_lhs_108555;
                
                // futhark/microgpt.fut:328:94-144
                
                double neg_res_108557 = -zp_res_108556;
                
                // futhark/microgpt.fut:110:42-54
                
                double max_res_108558 = fmax64(0.0, neg_res_108557);
                
                // futhark/microgpt.fut:110:35-54
                
                double sgn_res_108559 = fsignum64(max_res_108558);
                
                // futhark/microgpt.fut:328:75-147
                
                double neg_res_108560 = -sgn_res_108559;
                
                // futhark/microgpt.fut:328:66-148
                
                double zp_res_108561 = 1.0 + neg_res_108560;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_108562 = r_108554 + zp_res_108561;
                double r_tmp_116622 = zp_res_108562;
                
                r_108554 = r_tmp_116622;
            }
            defunc_0_lifted_lambda_res_108552 = r_108554;
            // futhark/microgpt.fut:328:35-151
            
            double zs_res_108563 = 1.0 / defunc_0_lifted_lambda_res_108552;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113714 = 0; i_113714 < (int64_t) 27; i_113714++) {
                double f_elem_108578 = ((double *) mem_115056)[i_113718 * (int64_t) 27 + i_113714];
                
                // futhark/microgpt.fut:329:76-118
                
                double zp_res_108583 = neg_res_108551 + f_elem_108578;
                
                // futhark/microgpt.fut:329:69-118
                
                double exp_res_108584 = futrts_exp64(zp_res_108583);
                
                // futhark/microgpt.fut:329:216-266
                
                double neg_res_108586 = -zp_res_108583;
                
                // futhark/microgpt.fut:110:42-54
                
                double max_res_108587 = fmax64(0.0, neg_res_108586);
                
                // futhark/microgpt.fut:110:35-54
                
                double sgn_res_108588 = fsignum64(max_res_108587);
                
                // futhark/microgpt.fut:329:197-269
                
                double neg_res_108589 = -sgn_res_108588;
                
                // futhark/microgpt.fut:329:188-270
                
                double zp_res_108590 = 1.0 + neg_res_108589;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_108591;
                double r_108593 = 0.0;
                
                for (int64_t i_108592 = 0; i_108592 < (int64_t) 27; i_108592++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_108594 = ((double *) mem_115159)[i_113718 * (int64_t) 729 + i_108592 * (int64_t) 27 + i_113714];
                    
                    // futhark/microgpt.fut:329:69-154
                    
                    double zt_res_108595 = exp_res_108584 * zt_rhs_108594;
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_108596 = ((double *) mem_115186)[i_113718 * (int64_t) 27 + i_108592];
                    
                    // futhark/microgpt.fut:329:162-270
                    
                    double zt_res_108597 = zp_res_108590 * zt_lhs_108596;
                    
                    // futhark/microgpt.fut:329:183-290
                    
                    double zt_res_108598 = zs_res_108563 * zt_res_108597;
                    
                    // futhark/microgpt.fut:329:122-290
                    
                    double zp_res_108599 = zt_res_108595 + zt_res_108598;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_108600 = r_108593 + zp_res_108599;
                    double r_tmp_116624 = zp_res_108600;
                    
                    r_108593 = r_tmp_116624;
                }
                defunc_0_lifted_lambda_res_108591 = r_108593;
                ((double *) mem_115207)[i_113714] = defunc_0_lifted_lambda_res_108591;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115202, i_113718 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115207, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113726 = 0; i_113726 < (int64_t) 16; i_113726++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113722 = 0; i_113722 < (int64_t) 16; i_113722++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_106926;
                double r_106928 = 0.0;
                
                for (int64_t i_106927 = 0; i_106927 < (int64_t) 27; i_106927++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_106929 = ((double *) mem_115202)[i_113726 * (int64_t) 27 + i_106927];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_106930 = ((double *) mem_param_114488.mem)[i_106927 * (int64_t) 16 + i_113722];
                    
                    // futhark/microgpt.fut:330:67-112
                    
                    double zt_res_106931 = zt_lhs_106929 * zt_rhs_106930;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_106932 = r_106928 + zt_res_106931;
                    double r_tmp_116627 = zp_res_106932;
                    
                    r_106928 = r_tmp_116627;
                }
                defunc_0_lifted_lambda_res_106926 = r_106928;
                ((double *) mem_115223)[i_113722] = defunc_0_lifted_lambda_res_106926;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115218, i_113726 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115223, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113734 = 0; i_113734 < (int64_t) 16; i_113734++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113730 = 0; i_113730 < (int64_t) 16; i_113730++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_106947 = ((double *) mem_115218)[i_113734 * (int64_t) 16 + i_113730];
                
                ((double *) mem_115239)[i_113730] = lifted_lambda_res_106947;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115234, i_113734 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115239, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113747 = 0; i_113747 < (int64_t) 16; i_113747++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113740 = 0; i_113740 < (int64_t) 64; i_113740++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_111558;
                double r_111560 = 0.0;
                
                for (int64_t i_111559 = 0; i_111559 < (int64_t) 16; i_111559++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_111561 = ((double *) mem_115234)[i_113747 * (int64_t) 16 + i_111559];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_111562 = ((double *) mem_param_114456.mem)[i_111559 * (int64_t) 64 + i_113740];
                    
                    // futhark/microgpt.fut:332:67-113
                    
                    double zt_res_111563 = zt_lhs_111561 * zt_rhs_111562;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_111564 = r_111560 + zt_res_111563;
                    double r_tmp_116634 = zp_res_111564;
                    
                    r_111560 = r_tmp_116634;
                }
                defunc_0_lifted_lambda_res_111558 = r_111560;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_111571;
                double r_111573 = 0.0;
                
                for (int64_t i_111572 = 0; i_111572 < (int64_t) 16; i_111572++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_111574 = ((double *) mem_115234)[i_111572 * (int64_t) 16 + i_113747];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_111575 = ((double *) mem_115008)[i_111572 * (int64_t) 64 + i_113740];
                    
                    // futhark/microgpt.fut:390:69-113
                    
                    double zt_res_111576 = zt_lhs_111574 * zt_rhs_111575;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_111577 = r_111573 + zt_res_111576;
                    double r_tmp_116635 = zp_res_111577;
                    
                    r_111573 = r_tmp_116635;
                }
                defunc_0_lifted_lambda_res_111571 = r_111573;
                ((double *) mem_115260)[i_113740] = defunc_0_lifted_lambda_res_111571;
                ((double *) mem_115261)[i_113740] = defunc_0_lifted_lambda_res_111558;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115250, i_113747 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115260, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115251, i_113747 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115261, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113756 = 0; i_113756 < (int64_t) 16; i_113756++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113752 = 0; i_113752 < (int64_t) 64; i_113752++) {
                // futhark/microgpt.fut:4:11-25
                
                double indicatorp_arg0_106983 = ((double *) mem_114992)[i_113756 * (int64_t) 64 + i_113752];
                
                // futhark/microgpt.fut:110:42-54
                
                double max_res_106984 = fmax64(0.0, indicatorp_arg0_106983);
                
                // futhark/microgpt.fut:110:35-54
                
                double sgn_res_106985 = fsignum64(max_res_106984);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_106986 = ((double *) mem_115251)[i_113756 * (int64_t) 64 + i_113752];
                
                // futhark/microgpt.fut:333:46-102
                
                double zt_res_106987 = sgn_res_106985 * zt_rhs_106986;
                
                ((double *) mem_115287)[i_113752] = zt_res_106987;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115282, i_113756 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115287, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113764 = 0; i_113764 < (int64_t) 16; i_113764++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113760 = 0; i_113760 < (int64_t) 16; i_113760++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_107002;
                double r_107004 = 0.0;
                
                for (int64_t i_107003 = 0; i_107003 < (int64_t) 64; i_107003++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_107005 = ((double *) mem_115282)[i_113764 * (int64_t) 64 + i_107003];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_107006 = ((double *) mem_param_114480.mem)[i_107003 * (int64_t) 16 + i_113760];
                    
                    // futhark/microgpt.fut:334:67-111
                    
                    double zt_res_107007 = zt_lhs_107005 * zt_rhs_107006;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_107008 = r_107004 + zt_res_107007;
                    double r_tmp_116640 = zp_res_107008;
                    
                    r_107004 = r_tmp_116640;
                }
                defunc_0_lifted_lambda_res_107002 = r_107004;
                ((double *) mem_115303)[i_113760] = defunc_0_lifted_lambda_res_107002;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115298, i_113764 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115303, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113772 = 0; i_113772 < (int64_t) 16; i_113772++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113768 = 0; i_113768 < (int64_t) 16; i_113768++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_107047 = ((double *) mem_115298)[i_113772 * (int64_t) 16 + i_113768];
                
                ((double *) mem_115319)[i_113768] = lifted_lambda_res_107047;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115314, i_113772 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115319, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113778 = 0; i_113778 < (int64_t) 16; i_113778++) {
            // futhark/microgpt.fut:336:43-55
            
            double zp_lhs_108419 = ((double *) mem_114962)[i_113778];
            
            // futhark/microgpt.fut:336:43-83
            
            double zp_res_108420 = 1.0e-5 + zp_lhs_108419;
            
            // futhark/microgpt.fut:336:35-83
            
            double sqrt_res_108421 = futrts_sqrt64(zp_res_108420);
            
            // futhark/microgpt.fut:338:125-154
            
            double zt_res_108429 = sqrt_res_108421 * sqrt_res_108421;
            
            // futhark/microgpt.fut:338:116-154
            
            double zs_res_108430 = 1.0 / zt_res_108429;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_108431;
            double r_108433 = 0.0;
            
            for (int64_t i_108432 = 0; i_108432 < (int64_t) 16; i_108432++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_108434 = ((double *) mem_115314)[i_113778 * (int64_t) 16 + i_108432];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_108435 = ((double *) mem_114946)[i_113778 * (int64_t) 16 + i_108432];
                
                // futhark/microgpt.fut:338:65-109
                
                double zt_res_108436 = zt_lhs_108434 * zt_rhs_108435;
                
                // futhark/microgpt.fut:338:86-154
                
                double zt_res_108437 = zs_res_108430 * zt_res_108436;
                
                // futhark/microgpt.fut:338:57-154
                
                double neg_res_108438 = -zt_res_108437;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_108439 = r_108433 + neg_res_108438;
                double r_tmp_116645 = zp_res_108439;
                
                r_108433 = r_tmp_116645;
            }
            defunc_0_lifted_lambda_res_108431 = r_108433;
            ((double *) mem_115330)[i_113778] = defunc_0_lifted_lambda_res_108431;
            ((double *) mem_115331)[i_113778] = sqrt_res_108421;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113783 = 0; i_113783 < (int64_t) 16; i_113783++) {
            // futhark/microgpt.fut:339:35-47
            
            double zt_lhs_107075 = ((double *) mem_115330)[i_113783];
            
            // futhark/microgpt.fut:339:89-101
            
            double zp_lhs_107076 = ((double *) mem_114962)[i_113783];
            
            // futhark/microgpt.fut:339:89-129
            
            double zp_res_107077 = 1.0e-5 + zp_lhs_107076;
            
            // futhark/microgpt.fut:339:81-129
            
            double sqrt_res_107078 = futrts_sqrt64(zp_res_107077);
            
            // futhark/microgpt.fut:339:67-131
            
            double zt_res_107079 = 2.0 * sqrt_res_107078;
            
            // futhark/microgpt.fut:339:53-131
            
            double zs_res_107080 = 1.0 / zt_res_107079;
            
            // futhark/microgpt.fut:339:35-131
            
            double zt_res_107081 = zt_lhs_107075 * zs_res_107080;
            
            ((double *) mem_115344)[i_113783] = zt_res_107081;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113791 = 0; i_113791 < (int64_t) 16; i_113791++) {
            // futhark/microgpt.fut:340:106-118
            
            double zs_rhs_107089 = ((double *) mem_115331)[i_113791];
            
            // futhark/microgpt.fut:340:98-118
            
            double zs_res_107090 = 1.0 / zs_rhs_107089;
            
            // futhark/microgpt.fut:340:128-140
            
            double zs_lhs_107091 = ((double *) mem_115344)[i_113791];
            
            // futhark/microgpt.fut:340:128-155
            
            double zs_res_107092 = zs_lhs_107091 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113787 = 0; i_113787 < (int64_t) 16; i_113787++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_107099 = ((double *) mem_115234)[i_113791 * (int64_t) 16 + i_113787];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_107100 = ((double *) mem_115314)[i_113791 * (int64_t) 16 + i_113787];
                
                // futhark/microgpt.fut:340:72-118
                
                double zt_res_107101 = zs_res_107090 * zt_lhs_107100;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_107102 = ((double *) mem_114946)[i_113791 * (int64_t) 16 + i_113787];
                
                // futhark/microgpt.fut:340:141-180
                
                double zt_res_107103 = zs_res_107092 * zt_rhs_107102;
                
                // futhark/microgpt.fut:340:157-240
                
                double zp_res_107104 = zt_res_107103 + zt_res_107103;
                
                // futhark/microgpt.fut:340:93-240
                
                double zp_res_107105 = zt_res_107101 + zp_res_107104;
                
                // futhark/microgpt.fut:340:45-240
                
                double zp_res_107106 = zp_lhs_107099 + zp_res_107105;
                
                ((double *) mem_115356)[i_113787] = zp_res_107106;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115351, i_113791 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115356, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113799 = 0; i_113799 < (int64_t) 16; i_113799++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113795 = 0; i_113795 < (int64_t) 16; i_113795++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_107121 = ((double *) mem_115351)[i_113799 * (int64_t) 16 + i_113795];
                
                ((double *) mem_115372)[i_113795] = lifted_lambda_res_107121;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115367, i_113799 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115372, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113812 = 0; i_113812 < (int64_t) 16; i_113812++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113805 = 0; i_113805 < (int64_t) 16; i_113805++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_111602;
                double r_111604 = 0.0;
                
                for (int64_t i_111603 = 0; i_111603 < (int64_t) 16; i_111603++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_111605 = ((double *) mem_115367)[i_113812 * (int64_t) 16 + i_111603];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_111606 = ((double *) mem_param_114464.mem)[i_111603 * (int64_t) 16 + i_113805];
                    
                    // futhark/microgpt.fut:342:67-112
                    
                    double zt_res_111607 = zt_lhs_111605 * zt_rhs_111606;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_111608 = r_111604 + zt_res_111607;
                    double r_tmp_116655 = zp_res_111608;
                    
                    r_111604 = r_tmp_116655;
                }
                defunc_0_lifted_lambda_res_111602 = r_111604;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_111615;
                double r_111617 = 0.0;
                
                for (int64_t i_111616 = 0; i_111616 < (int64_t) 16; i_111616++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_111618 = ((double *) mem_115367)[i_111616 * (int64_t) 16 + i_113812];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_111619 = ((double *) mem_114914)[i_111616 * (int64_t) 16 + i_113805];
                    
                    // futhark/microgpt.fut:388:68-112
                    
                    double zt_res_111620 = zt_lhs_111618 * zt_rhs_111619;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_111621 = r_111617 + zt_res_111620;
                    double r_tmp_116656 = zp_res_111621;
                    
                    r_111617 = r_tmp_116656;
                }
                defunc_0_lifted_lambda_res_111615 = r_111617;
                ((double *) mem_115393)[i_113805] = defunc_0_lifted_lambda_res_111615;
                ((double *) mem_115394)[i_113805] = defunc_0_lifted_lambda_res_111602;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115383, i_113812 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115393, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115384, i_113812 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115394, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113834 = 0; i_113834 < (int64_t) 4; i_113834++) {
            // futhark/microgpt.fut:343:74-77
            
            int64_t zp_lhs_109521 = mul64((int64_t) 4, i_113834);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113827 = 0; i_113827 < (int64_t) 16; i_113827++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113817 = 0; i_113817 < (int64_t) 4; i_113817++) {
                    // futhark/microgpt.fut:343:79-87
                    
                    int64_t tmp_111643 = add64(zp_lhs_109521, i_113817);
                    
                    // futhark/microgpt.fut:343:52-89
                    
                    bool x_111644 = sle64((int64_t) 0, tmp_111643);
                    
                    // futhark/microgpt.fut:343:52-89
                    
                    bool y_111645 = slt64(tmp_111643, (int64_t) 16);
                    
                    // futhark/microgpt.fut:343:52-89
                    
                    bool bounds_check_111646 = x_111644 && y_111645;
                    
                    // futhark/microgpt.fut:343:52-89
                    
                    bool index_certs_111647;
                    
                    if (!bounds_check_111646) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_111643, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:343:52-89\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:343:13-90\n   #9  futhark/microgpt.fut:561:5-76\n   #10 futhark/microgpt.fut:578:26-584:31\n   #11 futhark/microgpt.fut:612:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_111648 = ((double *) mem_115384)[i_113827 * (int64_t) 16 + tmp_111643];
                    
                    ((double *) mem_115437)[i_113817] = lifted_lambda_res_111648;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113821 = 0; i_113821 < (int64_t) 16; i_113821++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_111662 = ((double *) mem_114782)[i_113834 * (int64_t) 256 + i_113827 * (int64_t) 16 + i_113821];
                    
                    // futhark/microgpt.fut:345:55-97
                    
                    double zs_res_111663 = zs_lhs_111662 / 2.0;
                    double zp_rhs_111664 = ((double *) masks_mem_114450.mem)[step_106053 * (int64_t) 256 + i_113827 * (int64_t) 16 + i_113821];
                    
                    // futhark/microgpt.fut:345:84-123
                    
                    double zp_res_111665 = zs_res_111663 + zp_rhs_111664;
                    
                    ((double *) mem_115444)[i_113821] = zp_res_111665;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_115427, i_113827 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115444, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_115428, i_113827 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115437, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_115415, i_113834 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_115427, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_115416, i_113834 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_115428, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113872 = 0; i_113872 < (int64_t) 4; i_113872++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113862 = 0; i_113862 < (int64_t) 16; i_113862++) {
                // futhark/microgpt.fut:115:13-33
                
                double defunc_0_reduce_res_113272;
                double defunc_0_reduce_res_113273;
                double redout_113837;
                double redout_113838;
                
                redout_113837 = -INFINITY;
                redout_113838 = -INFINITY;
                for (int64_t i_113839 = 0; i_113839 < (int64_t) 16; i_113839++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_111918 = ((double *) mem_115415)[i_113872 * (int64_t) 256 + i_113862 * (int64_t) 16 + i_113839];
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_111786 = fmax64(lifted_lambda_res_111918, redout_113837);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_111856 = fmax64(lifted_lambda_res_111918, redout_113838);
                    double redout_tmp_116669 = max_res_111786;
                    double redout_tmp_116670 = max_res_111856;
                    
                    redout_113837 = redout_tmp_116669;
                    redout_113838 = redout_tmp_116670;
                }
                defunc_0_reduce_res_113272 = redout_113837;
                defunc_0_reduce_res_113273 = redout_113838;
                // futhark/microgpt.fut:347:80-90
                
                double neg_res_111787 = -defunc_0_reduce_res_113272;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113842 = 0; i_113842 < (int64_t) 16; i_113842++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_111794 = ((double *) mem_115415)[i_113872 * (int64_t) 256 + i_113862 * (int64_t) 16 + i_113842];
                    
                    // futhark/microgpt.fut:347:46-90
                    
                    double zp_res_111795 = neg_res_111787 + zp_lhs_111794;
                    
                    // futhark/microgpt.fut:347:39-90
                    
                    double exp_res_111796 = futrts_exp64(zp_res_111795);
                    
                    ((double *) mem_115500)[i_113842] = exp_res_111796;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_111798;
                double r_111800 = 0.0;
                
                for (int64_t i_111799 = 0; i_111799 < (int64_t) 16; i_111799++) {
                    // futhark/microgpt.fut:348:38-50
                    
                    double lifted_lambda_res_111801 = ((double *) mem_115500)[i_111799];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_111802 = r_111800 + lifted_lambda_res_111801;
                    double r_tmp_116672 = zp_res_111802;
                    
                    r_111800 = r_tmp_116672;
                }
                defunc_0_lifted_lambda_res_111798 = r_111800;
                // futhark/microgpt.fut:349:57-69
                
                double zs_res_111803 = 1.0 / defunc_0_lifted_lambda_res_111798;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113846 = 0; i_113846 < (int64_t) 16; i_113846++) {
                    // futhark/microgpt.fut:349:39-51
                    
                    double zt_lhs_111810 = ((double *) mem_115500)[i_113846];
                    
                    // futhark/microgpt.fut:349:39-69
                    
                    double zt_res_111811 = zs_res_111803 * zt_lhs_111810;
                    
                    ((double *) mem_115507)[i_113846] = zt_res_111811;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113850 = 0; i_113850 < (int64_t) 16; i_113850++) {
                    // futhark/microgpt.fut:350:4-16
                    
                    double lifted_lambda_res_111819 = ((double *) mem_115507)[i_113850];
                    
                    ((double *) mem_115514)[i_113850] = lifted_lambda_res_111819;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113854 = 0; i_113854 < (int64_t) 4; i_113854++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_111833 = ((double *) mem_115416)[i_113872 * (int64_t) 64 + i_113862 * (int64_t) 4 + i_113854];
                    
                    ((double *) mem_115521)[i_113854] = lifted_lambda_res_111833;
                }
                ((double *) mem_115486)[i_113862] = defunc_0_reduce_res_113273;
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_115487, i_113862 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115521, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_115488, i_113862 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115514, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115469, i_113872 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115486, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_115470, i_113872 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_115487, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_115471, i_113872 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_115488, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113913 = 0; i_113913 < (int64_t) 4; i_113913++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113897 = 0; i_113897 < (int64_t) 16; i_113897++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_112189 = ((double *) mem_115469)[i_113913 * (int64_t) 16 + i_113897];
                
                // futhark/microgpt.fut:354:95-121
                
                double neg_res_112190 = -neg_arg0_112189;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113880 = 0; i_113880 < (int64_t) 16; i_113880++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_112281;
                    double r_112283 = 0.0;
                    
                    for (int64_t i_112282 = 0; i_112282 < (int64_t) 4; i_112282++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_112284 = ((double *) mem_115470)[i_113913 * (int64_t) 64 + i_113897 * (int64_t) 4 + i_112282];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_112285 = ((double *) mem_114701)[i_113913 * (int64_t) 64 + i_113880 * (int64_t) 4 + i_112282];
                        
                        // futhark/microgpt.fut:352:75-135
                        
                        double zt_res_112286 = zt_lhs_112284 * zt_rhs_112285;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_112287 = r_112283 + zt_res_112286;
                        double r_tmp_116688 = zp_res_112287;
                        
                        r_112283 = r_tmp_116688;
                    }
                    defunc_0_lifted_lambda_res_112281 = r_112283;
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_112294 = ((double *) mem_115415)[i_113913 * (int64_t) 256 + i_113897 * (int64_t) 16 + i_113880];
                    
                    // futhark/microgpt.fut:354:61-121
                    
                    double zp_res_112295 = neg_res_112190 + zp_lhs_112294;
                    
                    // futhark/microgpt.fut:354:54-121
                    
                    double exp_res_112296 = futrts_exp64(zp_res_112295);
                    
                    ((double *) mem_115604)[i_113880] = exp_res_112296;
                    ((double *) mem_115605)[i_113880] = defunc_0_lifted_lambda_res_112281;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_112220;
                double r_112222 = 0.0;
                
                for (int64_t i_112221 = 0; i_112221 < (int64_t) 16; i_112221++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_112223 = ((double *) mem_115415)[i_113913 * (int64_t) 256 + i_113897 * (int64_t) 16 + i_112221];
                    
                    // futhark/microgpt.fut:361:110-170
                    
                    double zp_res_112224 = neg_res_112190 + zp_lhs_112223;
                    
                    // futhark/microgpt.fut:361:103-170
                    
                    double neg_res_112225 = -zp_res_112224;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_112226 = fmax64(0.0, neg_res_112225);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_112227 = fsignum64(max_res_112226);
                    
                    // futhark/microgpt.fut:361:84-173
                    
                    double neg_res_112228 = -sgn_res_112227;
                    
                    // futhark/microgpt.fut:361:75-174
                    
                    double zp_res_112229 = 1.0 + neg_res_112228;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_112230 = r_112222 + zp_res_112229;
                    double r_tmp_116689 = zp_res_112230;
                    
                    r_112222 = r_tmp_116689;
                }
                defunc_0_lifted_lambda_res_112220 = r_112222;
                // futhark/microgpt.fut:361:44-177
                
                double zs_res_112231 = 1.0 / defunc_0_lifted_lambda_res_112220;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113885 = 0; i_113885 < (int64_t) 4; i_113885++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_112253;
                    double r_112255 = 0.0;
                    
                    for (int64_t i_112254 = 0; i_112254 < (int64_t) 16; i_112254++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_112256 = ((double *) mem_115470)[i_113913 * (int64_t) 64 + i_112254 * (int64_t) 4 + i_113885];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_112257 = ((double *) mem_115471)[i_113913 * (int64_t) 256 + i_112254 * (int64_t) 16 + i_113897];
                        
                        // futhark/microgpt.fut:364:75-136
                        
                        double zt_res_112258 = zt_lhs_112256 * zt_rhs_112257;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_112259 = r_112255 + zt_res_112258;
                        double r_tmp_116691 = zp_res_112259;
                        
                        r_112255 = r_tmp_116691;
                    }
                    defunc_0_lifted_lambda_res_112253 = r_112255;
                    ((double *) mem_115618)[i_113885] = defunc_0_lifted_lambda_res_112253;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_115581, i_113897 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115618, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                ((double *) mem_115582)[i_113897] = zs_res_112231;
                ((double *) mem_115583)[i_113897] = neg_arg0_112189;
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_115584, i_113897 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115604, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_115585, i_113897 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115605, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_115553, i_113913 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_115581, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115554, i_113913 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115582, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115555, i_113913 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115583, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_115556, i_113913 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_115584, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_115557, i_113913 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_115585, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113929 = 0; i_113929 < (int64_t) 4; i_113929++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113925 = 0; i_113925 < (int64_t) 16; i_113925++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113921 = 0; i_113921 < (int64_t) 16; i_113921++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_107419 = ((double *) mem_115557)[i_113929 * (int64_t) 256 + i_113925 * (int64_t) 16 + i_113921];
                    
                    ((double *) mem_115677)[i_113921] = lifted_lambda_res_107419;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_115672, i_113925 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115677, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_115666, i_113929 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_115672, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113942 = 0; i_113942 < (int64_t) 4; i_113942++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113935 = 0; i_113935 < (int64_t) 16; i_113935++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_112325;
                double r_112327 = 0.0;
                
                for (int64_t i_112326 = 0; i_112326 < (int64_t) 16; i_112326++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_112328 = ((double *) mem_115556)[i_113942 * (int64_t) 256 + i_113935 * (int64_t) 16 + i_112326];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_112329 = r_112327 + lifted_lambda_res_112328;
                    double r_tmp_116699 = zp_res_112329;
                    
                    r_112327 = r_tmp_116699;
                }
                defunc_0_lifted_lambda_res_112325 = r_112327;
                // futhark/microgpt.fut:357:151-196
                
                double zt_res_112337 = defunc_0_lifted_lambda_res_112325 * defunc_0_lifted_lambda_res_112325;
                
                // futhark/microgpt.fut:357:142-196
                
                double zs_res_112338 = 1.0 / zt_res_112337;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_112339;
                double r_112341 = 0.0;
                
                for (int64_t i_112340 = 0; i_112340 < (int64_t) 16; i_112340++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_112342 = ((double *) mem_115666)[i_113942 * (int64_t) 256 + i_113935 * (int64_t) 16 + i_112340];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_112343 = ((double *) mem_115556)[i_113942 * (int64_t) 256 + i_113935 * (int64_t) 16 + i_112340];
                    
                    // futhark/microgpt.fut:357:74-135
                    
                    double zt_res_112344 = zt_lhs_112342 * zt_rhs_112343;
                    
                    // futhark/microgpt.fut:357:103-196
                    
                    double zt_res_112345 = zs_res_112338 * zt_res_112344;
                    
                    // futhark/microgpt.fut:357:66-196
                    
                    double neg_res_112346 = -zt_res_112345;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_112347 = r_112341 + neg_res_112346;
                    double r_tmp_116700 = zp_res_112347;
                    
                    r_112341 = r_tmp_116700;
                }
                defunc_0_lifted_lambda_res_112339 = r_112341;
                ((double *) mem_115703)[i_113935] = defunc_0_lifted_lambda_res_112339;
                ((double *) mem_115704)[i_113935] = defunc_0_lifted_lambda_res_112325;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115693, i_113942 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115703, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115694, i_113942 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115704, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113955 = 0; i_113955 < (int64_t) 4; i_113955++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113951 = 0; i_113951 < (int64_t) 16; i_113951++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_107462 = ((double *) mem_115694)[i_113955 * (int64_t) 16 + i_113951];
                
                // futhark/microgpt.fut:358:89-117
                
                double zs_res_107463 = 1.0 / zs_rhs_107462;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_107464 = ((double *) mem_115693)[i_113955 * (int64_t) 16 + i_113951];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113947 = 0; i_113947 < (int64_t) 16; i_113947++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_107471 = ((double *) mem_115666)[i_113955 * (int64_t) 256 + i_113951 * (int64_t) 16 + i_113947];
                    
                    // futhark/microgpt.fut:358:55-117
                    
                    double zt_res_107472 = zs_res_107463 * zt_lhs_107471;
                    
                    // futhark/microgpt.fut:358:84-144
                    
                    double zp_res_107473 = zp_rhs_107464 + zt_res_107472;
                    
                    ((double *) mem_115736)[i_113947] = zp_res_107473;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_115731, i_113951 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115736, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_115725, i_113955 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_115731, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113963 = 0; i_113963 < (int64_t) 4; i_113963++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113959 = 0; i_113959 < (int64_t) 16; i_113959++) {
                double f_elem_107486 = ((double *) mem_115469)[i_113963 * (int64_t) 16 + i_113959];
                
                // futhark/microgpt.fut:359:115-141
                
                double neg_res_107491 = -f_elem_107486;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_107492;
                double r_107494 = 0.0;
                
                for (int64_t i_107493 = 0; i_107493 < (int64_t) 16; i_107493++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_107495 = ((double *) mem_115415)[i_113963 * (int64_t) 256 + i_113959 * (int64_t) 16 + i_107493];
                    
                    // futhark/microgpt.fut:359:81-141
                    
                    double zp_res_107496 = neg_res_107491 + zp_lhs_107495;
                    
                    // futhark/microgpt.fut:359:74-141
                    
                    double exp_res_107497 = futrts_exp64(zp_res_107496);
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_107498 = ((double *) mem_115725)[i_113963 * (int64_t) 256 + i_113959 * (int64_t) 16 + i_107493];
                    
                    // futhark/microgpt.fut:359:74-177
                    
                    double zt_res_107499 = exp_res_107497 * zt_rhs_107498;
                    
                    // futhark/microgpt.fut:359:66-177
                    
                    double neg_res_107500 = -zt_res_107499;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_107501 = r_107494 + neg_res_107500;
                    double r_tmp_116706 = zp_res_107501;
                    
                    r_107494 = r_tmp_116706;
                }
                defunc_0_lifted_lambda_res_107492 = r_107494;
                ((double *) mem_115757)[i_113959] = defunc_0_lifted_lambda_res_107492;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115752, i_113963 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115757, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113975 = 0; i_113975 < (int64_t) 4; i_113975++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113971 = 0; i_113971 < (int64_t) 16; i_113971++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_107560 = ((double *) mem_115469)[i_113975 * (int64_t) 16 + i_113971];
                
                // futhark/microgpt.fut:362:97-123
                
                double neg_res_107561 = -neg_arg0_107560;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_107562 = ((double *) mem_115752)[i_113975 * (int64_t) 16 + i_113971];
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_107563 = ((double *) mem_115555)[i_113975 * (int64_t) 16 + i_113971];
                
                // futhark/microgpt.fut:362:262-288
                
                double neg_res_107564 = -neg_arg0_107563;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_107565 = ((double *) mem_115554)[i_113975 * (int64_t) 16 + i_113971];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113967 = 0; i_113967 < (int64_t) 16; i_113967++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_107572 = ((double *) mem_115415)[i_113975 * (int64_t) 256 + i_113971 * (int64_t) 16 + i_113967];
                    
                    // futhark/microgpt.fut:362:63-123
                    
                    double zp_res_107573 = neg_res_107561 + zp_lhs_107572;
                    
                    // futhark/microgpt.fut:362:56-123
                    
                    double exp_res_107574 = futrts_exp64(zp_res_107573);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_107575 = ((double *) mem_115725)[i_113975 * (int64_t) 256 + i_113971 * (int64_t) 16 + i_113967];
                    
                    // futhark/microgpt.fut:362:56-159
                    
                    double zt_res_107576 = exp_res_107574 * zt_rhs_107575;
                    
                    // futhark/microgpt.fut:362:228-288
                    
                    double zp_res_107577 = neg_res_107564 + zp_lhs_107572;
                    
                    // futhark/microgpt.fut:362:221-288
                    
                    double neg_res_107578 = -zp_res_107577;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_107579 = fmax64(0.0, neg_res_107578);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_107580 = fsignum64(max_res_107579);
                    
                    // futhark/microgpt.fut:362:202-291
                    
                    double neg_res_107581 = -sgn_res_107580;
                    
                    // futhark/microgpt.fut:362:193-292
                    
                    double zp_res_107582 = 1.0 + neg_res_107581;
                    
                    // futhark/microgpt.fut:362:167-292
                    
                    double zt_res_107583 = zt_lhs_107562 * zp_res_107582;
                    
                    // futhark/microgpt.fut:362:188-320
                    
                    double zt_res_107584 = zt_rhs_107565 * zt_res_107583;
                    
                    // futhark/microgpt.fut:362:127-320
                    
                    double zp_res_107585 = zt_res_107576 + zt_res_107584;
                    
                    ((double *) mem_115779)[i_113967] = zp_res_107585;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_115774, i_113971 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115779, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_115768, i_113975 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_115774, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_113987 = 0; i_113987 < (int64_t) 4; i_113987++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_113983 = 0; i_113983 < (int64_t) 16; i_113983++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113979 = 0; i_113979 < (int64_t) 16; i_113979++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_107607 = ((double *) mem_115768)[i_113987 * (int64_t) 256 + i_113983 * (int64_t) 16 + i_113979];
                    
                    // futhark/microgpt.fut:363:54-96
                    
                    double zs_res_107608 = zs_lhs_107607 / 2.0;
                    
                    ((double *) mem_115806)[i_113979] = zs_res_107608;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_115801, i_113983 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115806, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_115795, i_113987 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_115801, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_114007 = 0; i_114007 < (int64_t) 4; i_114007++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_114000 = 0; i_114000 < (int64_t) 16; i_114000++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_113993 = 0; i_113993 < (int64_t) 4; i_113993++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_112427;
                    double r_112429 = 0.0;
                    
                    for (int64_t i_112428 = 0; i_112428 < (int64_t) 16; i_112428++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_112430 = ((double *) mem_115795)[i_114007 * (int64_t) 256 + i_112428 * (int64_t) 16 + i_114000];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_112431 = ((double *) mem_114703)[i_114007 * (int64_t) 64 + i_112428 * (int64_t) 4 + i_113993];
                        
                        // futhark/microgpt.fut:365:75-135
                        
                        double zt_res_112432 = zt_lhs_112430 * zt_rhs_112431;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_112433 = r_112429 + zt_res_112432;
                        double r_tmp_116719 = zp_res_112433;
                        
                        r_112429 = r_tmp_116719;
                    }
                    defunc_0_lifted_lambda_res_112427 = r_112429;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_112440;
                    double r_112442 = 0.0;
                    
                    for (int64_t i_112441 = 0; i_112441 < (int64_t) 16; i_112441++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_112443 = ((double *) mem_115795)[i_114007 * (int64_t) 256 + i_114000 * (int64_t) 16 + i_112441];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_112444 = ((double *) mem_114702)[i_114007 * (int64_t) 64 + i_112441 * (int64_t) 4 + i_113993];
                        
                        // futhark/microgpt.fut:366:75-135
                        
                        double zt_res_112445 = zt_lhs_112443 * zt_rhs_112444;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_112446 = r_112442 + zt_res_112445;
                        double r_tmp_116720 = zp_res_112446;
                        
                        r_112442 = r_tmp_116720;
                    }
                    defunc_0_lifted_lambda_res_112440 = r_112442;
                    ((double *) mem_115844)[i_113993] = defunc_0_lifted_lambda_res_112440;
                    ((double *) mem_115845)[i_113993] = defunc_0_lifted_lambda_res_112427;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_115834, i_114000 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115844, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_115835, i_114000 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115845, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_115822, i_114007 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_115834, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_115823, i_114007 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_115835, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_114026 = 0; i_114026 < (int64_t) 16; i_114026++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_114016 = 0; i_114016 < (int64_t) 16; i_114016++) {
                // futhark/microgpt.fut:367:57-60
                
                int64_t tmp_112509 = sdiv64(i_114016, (int64_t) 4);
                
                // futhark/microgpt.fut:367:44-62
                
                bool x_112510 = sle64((int64_t) 0, tmp_112509);
                
                // futhark/microgpt.fut:367:44-62
                
                bool y_112511 = slt64(tmp_112509, (int64_t) 4);
                
                // futhark/microgpt.fut:367:44-62
                
                bool bounds_check_112512 = x_112510 && y_112511;
                
                // futhark/microgpt.fut:367:44-62
                
                bool index_certs_112513;
                
                if (!bounds_check_112512) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_112509, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:367:44-62\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:367:13-85\n   #6  futhark/microgpt.fut:561:5-76\n   #7  futhark/microgpt.fut:578:26-584:31\n   #8  futhark/microgpt.fut:612:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:367:79-82
                
                int64_t tmp_112514 = smod64(i_114016, (int64_t) 4);
                
                // futhark/microgpt.fut:367:44-84
                
                bool x_112515 = sle64((int64_t) 0, tmp_112514);
                
                // futhark/microgpt.fut:367:44-84
                
                bool y_112516 = slt64(tmp_112514, (int64_t) 4);
                
                // futhark/microgpt.fut:367:44-84
                
                bool bounds_check_112517 = x_112515 && y_112516;
                
                // futhark/microgpt.fut:367:44-84
                
                bool index_certs_112518;
                
                if (!bounds_check_112517) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_112514, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:367:44-84\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:367:13-85\n   #6  futhark/microgpt.fut:561:5-76\n   #7  futhark/microgpt.fut:578:26-584:31\n   #8  futhark/microgpt.fut:612:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_112519 = ((double *) mem_115553)[tmp_112509 * (int64_t) 64 + i_114026 * (int64_t) 4 + tmp_112514];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_112532 = ((double *) mem_115823)[tmp_112509 * (int64_t) 64 + i_114026 * (int64_t) 4 + tmp_112514];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_112548 = ((double *) mem_115822)[tmp_112509 * (int64_t) 64 + i_114026 * (int64_t) 4 + tmp_112514];
                
                ((double *) mem_115891)[i_114016] = lifted_lambda_res_112548;
                ((double *) mem_115892)[i_114016] = lifted_lambda_res_112532;
                ((double *) mem_115893)[i_114016] = lifted_lambda_res_112519;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115876, i_114026 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115891, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115877, i_114026 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115892, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115878, i_114026 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115893, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_114051 = 0; i_114051 < (int64_t) 16; i_114051++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_114038 = 0; i_114038 < (int64_t) 16; i_114038++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_112711;
                double r_112713 = 0.0;
                
                for (int64_t i_112712 = 0; i_112712 < (int64_t) 16; i_112712++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_112714 = ((double *) mem_115878)[i_114051 * (int64_t) 16 + i_112712];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_112715 = ((double *) mem_param_114484.mem)[i_112712 * (int64_t) 16 + i_114038];
                    
                    // futhark/microgpt.fut:370:69-114
                    
                    double zt_res_112716 = zt_lhs_112714 * zt_rhs_112715;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_112717 = r_112713 + zt_res_112716;
                    double r_tmp_116735 = zp_res_112717;
                    
                    r_112713 = r_tmp_116735;
                }
                defunc_0_lifted_lambda_res_112711 = r_112713;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_112718;
                double r_112720 = 0.0;
                
                for (int64_t i_112719 = 0; i_112719 < (int64_t) 16; i_112719++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_112721 = ((double *) mem_115877)[i_114051 * (int64_t) 16 + i_112719];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_112722 = ((double *) mem_param_114460.mem)[i_112719 * (int64_t) 16 + i_114038];
                    
                    // futhark/microgpt.fut:370:145-190
                    
                    double zt_res_112723 = zt_lhs_112721 * zt_rhs_112722;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_112724 = r_112720 + zt_res_112723;
                    double r_tmp_116736 = zp_res_112724;
                    
                    r_112720 = r_tmp_116736;
                }
                defunc_0_lifted_lambda_res_112718 = r_112720;
                // futhark/microgpt.fut:370:47-192
                
                double zp_res_112725 = defunc_0_lifted_lambda_res_112711 + defunc_0_lifted_lambda_res_112718;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_112726;
                double r_112728 = 0.0;
                
                for (int64_t i_112727 = 0; i_112727 < (int64_t) 16; i_112727++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_112729 = ((double *) mem_115876)[i_114051 * (int64_t) 16 + i_112727];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_112730 = ((double *) mem_param_114472.mem)[i_112727 * (int64_t) 16 + i_114038];
                    
                    // futhark/microgpt.fut:370:222-267
                    
                    double zt_res_112731 = zt_lhs_112729 * zt_rhs_112730;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_112732 = r_112728 + zt_res_112731;
                    double r_tmp_116737 = zp_res_112732;
                    
                    r_112728 = r_tmp_116737;
                }
                defunc_0_lifted_lambda_res_112726 = r_112728;
                // futhark/microgpt.fut:370:118-269
                
                double zp_res_112733 = zp_res_112725 + defunc_0_lifted_lambda_res_112726;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_112740;
                double r_112742 = 0.0;
                
                for (int64_t i_112741 = 0; i_112741 < (int64_t) 16; i_112741++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_112743 = ((double *) mem_115876)[i_112741 * (int64_t) 16 + i_114051];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_112744 = ((double *) mem_114624)[i_112741 * (int64_t) 16 + i_114038];
                    
                    // futhark/microgpt.fut:385:68-111
                    
                    double zt_res_112745 = zt_lhs_112743 * zt_rhs_112744;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_112746 = r_112742 + zt_res_112745;
                    double r_tmp_116738 = zp_res_112746;
                    
                    r_112742 = r_tmp_116738;
                }
                defunc_0_lifted_lambda_res_112740 = r_112742;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_112756;
                double r_112758 = 0.0;
                
                for (int64_t i_112757 = 0; i_112757 < (int64_t) 16; i_112757++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_112759 = ((double *) mem_115877)[i_112757 * (int64_t) 16 + i_114051];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_112760 = ((double *) mem_114624)[i_112757 * (int64_t) 16 + i_114038];
                    
                    // futhark/microgpt.fut:386:68-111
                    
                    double zt_res_112761 = zt_lhs_112759 * zt_rhs_112760;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_112762 = r_112758 + zt_res_112761;
                    double r_tmp_116739 = zp_res_112762;
                    
                    r_112758 = r_tmp_116739;
                }
                defunc_0_lifted_lambda_res_112756 = r_112758;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_112774;
                double r_112776 = 0.0;
                
                for (int64_t i_112775 = 0; i_112775 < (int64_t) 16; i_112775++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_112777 = ((double *) mem_115878)[i_112775 * (int64_t) 16 + i_114051];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_112778 = ((double *) mem_114624)[i_112775 * (int64_t) 16 + i_114038];
                    
                    // futhark/microgpt.fut:387:68-111
                    
                    double zt_res_112779 = zt_lhs_112777 * zt_rhs_112778;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_112780 = r_112776 + zt_res_112779;
                    double r_tmp_116740 = zp_res_112780;
                    
                    r_112776 = r_tmp_116740;
                }
                defunc_0_lifted_lambda_res_112774 = r_112776;
                ((double *) mem_115944)[i_114038] = defunc_0_lifted_lambda_res_112774;
                ((double *) mem_115945)[i_114038] = defunc_0_lifted_lambda_res_112756;
                ((double *) mem_115946)[i_114038] = defunc_0_lifted_lambda_res_112740;
                ((double *) mem_115947)[i_114038] = zp_res_112733;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115924, i_114051 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115944, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115925, i_114051 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115945, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115926, i_114051 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115946, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115927, i_114051 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115947, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_114062 = 0; i_114062 < (int64_t) 16; i_114062++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_114058 = 0; i_114058 < (int64_t) 16; i_114058++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_107832 = ((double *) mem_115927)[i_114062 * (int64_t) 16 + i_114058];
                
                ((double *) mem_115993)[i_114058] = lifted_lambda_res_107832;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_115988, i_114062 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_115993, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_114068 = 0; i_114068 < (int64_t) 16; i_114068++) {
            // futhark/microgpt.fut:372:43-55
            
            double zp_lhs_108273 = ((double *) mem_114623)[i_114068];
            
            // futhark/microgpt.fut:372:43-83
            
            double zp_res_108274 = 1.0e-5 + zp_lhs_108273;
            
            // futhark/microgpt.fut:372:35-83
            
            double sqrt_res_108275 = futrts_sqrt64(zp_res_108274);
            
            // futhark/microgpt.fut:374:124-153
            
            double zt_res_108283 = sqrt_res_108275 * sqrt_res_108275;
            
            // futhark/microgpt.fut:374:115-153
            
            double zs_res_108284 = 1.0 / zt_res_108283;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_108285;
            double r_108287 = 0.0;
            
            for (int64_t i_108286 = 0; i_108286 < (int64_t) 16; i_108286++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_108288 = ((double *) mem_115988)[i_114068 * (int64_t) 16 + i_108286];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_108289 = ((double *) mem_114594)[i_114068 * (int64_t) 16 + i_108286];
                
                // futhark/microgpt.fut:374:65-108
                
                double zt_res_108290 = zt_lhs_108288 * zt_rhs_108289;
                
                // futhark/microgpt.fut:374:86-153
                
                double zt_res_108291 = zs_res_108284 * zt_res_108290;
                
                // futhark/microgpt.fut:374:57-153
                
                double neg_res_108292 = -zt_res_108291;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_108293 = r_108287 + neg_res_108292;
                double r_tmp_116745 = zp_res_108293;
                
                r_108287 = r_tmp_116745;
            }
            defunc_0_lifted_lambda_res_108285 = r_108287;
            ((double *) mem_116004)[i_114068] = defunc_0_lifted_lambda_res_108285;
            ((double *) mem_116005)[i_114068] = sqrt_res_108275;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_114073 = 0; i_114073 < (int64_t) 16; i_114073++) {
            // futhark/microgpt.fut:375:35-47
            
            double zt_lhs_107860 = ((double *) mem_116004)[i_114073];
            
            // futhark/microgpt.fut:375:89-101
            
            double zp_lhs_107861 = ((double *) mem_114623)[i_114073];
            
            // futhark/microgpt.fut:375:89-129
            
            double zp_res_107862 = 1.0e-5 + zp_lhs_107861;
            
            // futhark/microgpt.fut:375:81-129
            
            double sqrt_res_107863 = futrts_sqrt64(zp_res_107862);
            
            // futhark/microgpt.fut:375:67-131
            
            double zt_res_107864 = 2.0 * sqrt_res_107863;
            
            // futhark/microgpt.fut:375:53-131
            
            double zs_res_107865 = 1.0 / zt_res_107864;
            
            // futhark/microgpt.fut:375:35-131
            
            double zt_res_107866 = zt_lhs_107860 * zs_res_107865;
            
            ((double *) mem_116018)[i_114073] = zt_res_107866;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_114081 = 0; i_114081 < (int64_t) 16; i_114081++) {
            // futhark/microgpt.fut:376:106-118
            
            double zs_rhs_107874 = ((double *) mem_116005)[i_114081];
            
            // futhark/microgpt.fut:376:98-118
            
            double zs_res_107875 = 1.0 / zs_rhs_107874;
            
            // futhark/microgpt.fut:376:128-140
            
            double zs_lhs_107876 = ((double *) mem_116018)[i_114081];
            
            // futhark/microgpt.fut:376:128-155
            
            double zs_res_107877 = zs_lhs_107876 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_114077 = 0; i_114077 < (int64_t) 16; i_114077++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_107884 = ((double *) mem_115367)[i_114081 * (int64_t) 16 + i_114077];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_107885 = ((double *) mem_115988)[i_114081 * (int64_t) 16 + i_114077];
                
                // futhark/microgpt.fut:376:72-118
                
                double zt_res_107886 = zs_res_107875 * zt_lhs_107885;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_107887 = ((double *) mem_114594)[i_114081 * (int64_t) 16 + i_114077];
                
                // futhark/microgpt.fut:376:141-179
                
                double zt_res_107888 = zs_res_107877 * zt_rhs_107887;
                
                // futhark/microgpt.fut:376:157-238
                
                double zp_res_107889 = zt_res_107888 + zt_res_107888;
                
                // futhark/microgpt.fut:376:93-238
                
                double zp_res_107890 = zt_res_107886 + zp_res_107889;
                
                // futhark/microgpt.fut:376:45-238
                
                double zp_res_107891 = zp_lhs_107884 + zp_res_107890;
                
                ((double *) mem_116030)[i_114077] = zp_res_107891;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_116025, i_114081 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_116030, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_114089 = 0; i_114089 < (int64_t) 16; i_114089++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_114085 = 0; i_114085 < (int64_t) 16; i_114085++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_107932 = ((double *) mem_116025)[i_114089 * (int64_t) 16 + i_114085];
                
                ((double *) mem_116046)[i_114085] = lifted_lambda_res_107932;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_116041, i_114089 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_116046, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_114095 = 0; i_114095 < (int64_t) 16; i_114095++) {
            // futhark/microgpt.fut:378:43-55
            
            double zp_lhs_108233 = ((double *) mem_114593)[i_114095];
            
            // futhark/microgpt.fut:378:43-83
            
            double zp_res_108234 = 1.0e-5 + zp_lhs_108233;
            
            // futhark/microgpt.fut:378:35-83
            
            double sqrt_res_108235 = futrts_sqrt64(zp_res_108234);
            
            // futhark/microgpt.fut:380:152-181
            
            double zt_res_108243 = sqrt_res_108235 * sqrt_res_108235;
            
            // futhark/microgpt.fut:380:143-181
            
            double zs_res_108244 = 1.0 / zt_res_108243;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_108245;
            double r_108247 = 0.0;
            
            for (int64_t i_108246 = 0; i_108246 < (int64_t) 16; i_108246++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_108248 = ((double *) mem_116041)[i_114095 * (int64_t) 16 + i_108246];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_108249 = ((double *) mem_param_114468.mem)[i_114095 * (int64_t) 16 + i_108246];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_rhs_108250 = ((double *) mem_114561)[i_114095 * (int64_t) 16 + i_108246];
                
                // futhark/microgpt.fut:380:91-135
                
                double zp_res_108251 = zp_lhs_108249 + zp_rhs_108250;
                
                // futhark/microgpt.fut:380:65-135
                
                double zt_res_108252 = zt_lhs_108248 * zp_res_108251;
                
                // futhark/microgpt.fut:380:86-181
                
                double zt_res_108253 = zs_res_108244 * zt_res_108252;
                
                // futhark/microgpt.fut:380:57-181
                
                double neg_res_108254 = -zt_res_108253;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_108255 = r_108247 + neg_res_108254;
                double r_tmp_116753 = zp_res_108255;
                
                r_108247 = r_tmp_116753;
            }
            defunc_0_lifted_lambda_res_108245 = r_108247;
            ((double *) mem_116057)[i_114095] = defunc_0_lifted_lambda_res_108245;
            ((double *) mem_116058)[i_114095] = sqrt_res_108235;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_114100 = 0; i_114100 < (int64_t) 16; i_114100++) {
            // futhark/microgpt.fut:381:35-47
            
            double zt_lhs_107962 = ((double *) mem_116057)[i_114100];
            
            // futhark/microgpt.fut:381:89-101
            
            double zp_lhs_107963 = ((double *) mem_114593)[i_114100];
            
            // futhark/microgpt.fut:381:89-129
            
            double zp_res_107964 = 1.0e-5 + zp_lhs_107963;
            
            // futhark/microgpt.fut:381:81-129
            
            double sqrt_res_107965 = futrts_sqrt64(zp_res_107964);
            
            // futhark/microgpt.fut:381:67-131
            
            double zt_res_107966 = 2.0 * sqrt_res_107965;
            
            // futhark/microgpt.fut:381:53-131
            
            double zs_res_107967 = 1.0 / zt_res_107966;
            
            // futhark/microgpt.fut:381:35-131
            
            double zt_res_107968 = zt_lhs_107962 * zs_res_107967;
            
            ((double *) mem_116071)[i_114100] = zt_res_107968;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_114113 = 0; i_114113 < (int64_t) 16; i_114113++) {
            // futhark/microgpt.fut:384:80-92
            
            double zs_rhs_110639 = ((double *) mem_116058)[i_114113];
            
            // futhark/microgpt.fut:384:72-92
            
            double zs_res_110640 = 1.0 / zs_rhs_110639;
            
            // futhark/microgpt.fut:384:102-114
            
            double zs_lhs_110641 = ((double *) mem_116071)[i_114113];
            
            // futhark/microgpt.fut:384:102-129
            
            double zs_res_110642 = zs_lhs_110641 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_114106 = 0; i_114106 < (int64_t) 16; i_114106++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_112807 = ((double *) mem_116041)[i_114113 * (int64_t) 16 + i_114106];
                
                // futhark/microgpt.fut:384:46-92
                
                double zt_res_112808 = zs_res_110640 * zt_lhs_112807;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_112809 = ((double *) mem_param_114468.mem)[i_114113 * (int64_t) 16 + i_114106];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_112810 = ((double *) mem_114561)[i_114113 * (int64_t) 16 + i_114106];
                
                // futhark/microgpt.fut:384:136-180
                
                double zp_res_112811 = zp_lhs_112809 + zp_rhs_112810;
                
                // futhark/microgpt.fut:384:115-180
                
                double zt_res_112812 = zs_res_110642 * zp_res_112811;
                
                // futhark/microgpt.fut:384:131-267
                
                double zp_res_112813 = zt_res_112812 + zt_res_112812;
                
                // futhark/microgpt.fut:384:67-267
                
                double zp_res_112814 = zt_res_112808 + zp_res_112813;
                
                ((double *) mem_116088)[i_114106] = zp_res_112814;
                ((double *) mem_116089)[i_114106] = zp_res_112814;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_116078, i_114113 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_116088, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_116079, i_114113 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_116089, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_114122 = 0; i_114122 < (int64_t) 64; i_114122++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_114118 = 0; i_114118 < (int64_t) 16; i_114118++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_108092;
                double r_108094 = 0.0;
                
                for (int64_t i_108093 = 0; i_108093 < (int64_t) 16; i_108093++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_108095 = ((double *) mem_115282)[i_108093 * (int64_t) 64 + i_114122];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_108096 = ((double *) mem_114963)[i_108093 * (int64_t) 16 + i_114118];
                    
                    // futhark/microgpt.fut:389:67-111
                    
                    double zt_res_108097 = zt_lhs_108095 * zt_rhs_108096;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_108098 = r_108094 + zt_res_108097;
                    double r_tmp_116761 = zp_res_108098;
                    
                    r_108094 = r_tmp_116761;
                }
                defunc_0_lifted_lambda_res_108092 = r_108094;
                ((double *) mem_116115)[i_114118] = defunc_0_lifted_lambda_res_108092;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_116110, i_114122 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_116115, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_114135 = 0; i_114135 < (int64_t) 27; i_114135++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_114128 = 0; i_114128 < (int64_t) 16; i_114128++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_112849;
                double r_112851 = 0.0;
                
                for (int64_t i_112850 = 0; i_112850 < (int64_t) 16; i_112850++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_112852 = ((double *) mem_115202)[i_112850 * (int64_t) 27 + i_114135];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_112853 = ((double *) mem_115040)[i_112850 * (int64_t) 16 + i_114128];
                    
                    // futhark/microgpt.fut:391:68-112
                    
                    double zt_res_112854 = zt_lhs_112852 * zt_rhs_112853;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_112855 = r_112851 + zt_res_112854;
                    double r_tmp_116766 = zp_res_112855;
                    
                    r_112851 = r_tmp_116766;
                }
                defunc_0_lifted_lambda_res_112849 = r_112851;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_112858;
                double r_112860 = 0.0;
                
                for (int64_t i_112859 = 0; i_112859 < (int64_t) 16; i_112859++) {
                    int64_t zeze_lhs_112861 = ((int64_t *) seqs_mem_114452.mem)[step_106053 * (int64_t) 16 + i_112859];
                    
                    // futhark/microgpt.fut:562:58-109
                    
                    bool cond_112862 = zeze_lhs_112861 == i_114135;
                    
                    // futhark/microgpt.fut:562:58-109
                    
                    double lifted_lambda_res_112863;
                    
                    if (cond_112862) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double lifted_lambda_res_t_res_113321 = ((double *) mem_116078)[i_112859 * (int64_t) 16 + i_114128];
                        
                        lifted_lambda_res_112863 = lifted_lambda_res_t_res_113321;
                    } else {
                        lifted_lambda_res_112863 = 0.0;
                    }
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_112869 = r_112860 + lifted_lambda_res_112863;
                    double r_tmp_116767 = zp_res_112869;
                    
                    r_112860 = r_tmp_116767;
                }
                defunc_0_lifted_lambda_res_112858 = r_112860;
                ((double *) mem_116136)[i_114128] = defunc_0_lifted_lambda_res_112858;
                ((double *) mem_116137)[i_114128] = defunc_0_lifted_lambda_res_112849;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_116126, i_114135 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_116136, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_116127, i_114135 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_116137, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:66:26-45
        
        double i64_res_108187 = sitofp_i64_f64(step_106053);
        
        // futhark/microgpt.fut:497:46-65
        
        double zm_rhs_108188 = i64_res_108187 / 500.0;
        
        // futhark/microgpt.fut:497:24-65
        
        double zt_rhs_108189 = 1.0 - zm_rhs_108188;
        
        // futhark/microgpt.fut:497:19-65
        
        double lt_r_108190 = 1.0e-2 * zt_rhs_108189;
        
        // futhark/microgpt.fut:499:5-52
        if (memblock_alloc(ctx, &mem_116158, (int64_t) 3456, "mem_116158")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:499:5-52
        // futhark/microgpt.fut:499:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116158.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114476.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:499:5-52
        if (memblock_alloc(ctx, &mem_116160, (int64_t) 3456, "mem_116160")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:499:5-52
        // futhark/microgpt.fut:499:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116160.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114512.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:499:5-52
        if (memblock_alloc(ctx, &mem_116162, (int64_t) 3456, "mem_116162")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:499:5-52
        // futhark/microgpt.fut:499:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116162.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114548.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:499:5-52
        if (memblock_alloc(ctx, &mem_116164, (int64_t) 3456, "mem_116164")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:499:5-52
        // futhark/microgpt.fut:499:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116164.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_116126, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:499:5-52
        if (futrts_adam_opt_w_11669(ctx, &ext_mem_116168, &ext_mem_116167, &ext_mem_116166, mem_116158, mem_116160, mem_116162, mem_116164, (int64_t) 27, (int64_t) 16, step_106053, lt_r_108190) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_116158, "mem_116158") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116160, "mem_116160") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116162, "mem_116162") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116164, "mem_116164") != 0)
            return 1;
        // futhark/microgpt.fut:501:5-52
        if (memblock_alloc(ctx, &mem_116169, (int64_t) 2048, "mem_116169")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:501:5-52
        // futhark/microgpt.fut:501:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116169.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114468.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:501:5-52
        if (memblock_alloc(ctx, &mem_116171, (int64_t) 2048, "mem_116171")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:501:5-52
        // futhark/microgpt.fut:501:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116171.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114504.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:501:5-52
        if (memblock_alloc(ctx, &mem_116173, (int64_t) 2048, "mem_116173")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:501:5-52
        // futhark/microgpt.fut:501:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116173.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114540.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:501:5-52
        if (memblock_alloc(ctx, &mem_116175, (int64_t) 2048, "mem_116175")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:501:5-52
        // futhark/microgpt.fut:501:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116175.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_116079, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:501:5-52
        if (futrts_adam_opt_w_11670(ctx, &ext_mem_116179, &ext_mem_116178, &ext_mem_116177, mem_116169, mem_116171, mem_116173, mem_116175, (int64_t) 16, (int64_t) 16, step_106053, lt_r_108190) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_116169, "mem_116169") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116171, "mem_116171") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116173, "mem_116173") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116175, "mem_116175") != 0)
            return 1;
        // futhark/microgpt.fut:503:5-56
        if (memblock_alloc(ctx, &mem_116180, (int64_t) 2048, "mem_116180")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:503:5-56
        // futhark/microgpt.fut:503:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116180.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114472.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:503:5-56
        if (memblock_alloc(ctx, &mem_116182, (int64_t) 2048, "mem_116182")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:503:5-56
        // futhark/microgpt.fut:503:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116182.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114508.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:503:5-56
        if (memblock_alloc(ctx, &mem_116184, (int64_t) 2048, "mem_116184")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:503:5-56
        // futhark/microgpt.fut:503:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116184.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114544.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:503:5-56
        if (memblock_alloc(ctx, &mem_116186, (int64_t) 2048, "mem_116186")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:503:5-56
        // futhark/microgpt.fut:503:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116186.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_115926, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:503:5-56
        if (futrts_adam_opt_w_11670(ctx, &ext_mem_116190, &ext_mem_116189, &ext_mem_116188, mem_116180, mem_116182, mem_116184, mem_116186, (int64_t) 16, (int64_t) 16, step_106053, lt_r_108190) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_116180, "mem_116180") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116182, "mem_116182") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116184, "mem_116184") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116186, "mem_116186") != 0)
            return 1;
        // futhark/microgpt.fut:505:5-56
        if (memblock_alloc(ctx, &mem_116191, (int64_t) 2048, "mem_116191")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:505:5-56
        // futhark/microgpt.fut:505:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116191.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114460.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:505:5-56
        if (memblock_alloc(ctx, &mem_116193, (int64_t) 2048, "mem_116193")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:505:5-56
        // futhark/microgpt.fut:505:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116193.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114496.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:505:5-56
        if (memblock_alloc(ctx, &mem_116195, (int64_t) 2048, "mem_116195")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:505:5-56
        // futhark/microgpt.fut:505:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116195.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114532.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:505:5-56
        if (memblock_alloc(ctx, &mem_116197, (int64_t) 2048, "mem_116197")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:505:5-56
        // futhark/microgpt.fut:505:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116197.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_115925, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:505:5-56
        if (futrts_adam_opt_w_11670(ctx, &ext_mem_116201, &ext_mem_116200, &ext_mem_116199, mem_116191, mem_116193, mem_116195, mem_116197, (int64_t) 16, (int64_t) 16, step_106053, lt_r_108190) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_116191, "mem_116191") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116193, "mem_116193") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116195, "mem_116195") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116197, "mem_116197") != 0)
            return 1;
        // futhark/microgpt.fut:507:5-56
        if (memblock_alloc(ctx, &mem_116202, (int64_t) 2048, "mem_116202")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:507:5-56
        // futhark/microgpt.fut:507:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116202.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114484.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:507:5-56
        if (memblock_alloc(ctx, &mem_116204, (int64_t) 2048, "mem_116204")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:507:5-56
        // futhark/microgpt.fut:507:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116204.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114520.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:507:5-56
        if (memblock_alloc(ctx, &mem_116206, (int64_t) 2048, "mem_116206")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:507:5-56
        // futhark/microgpt.fut:507:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116206.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114556.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:507:5-56
        if (memblock_alloc(ctx, &mem_116208, (int64_t) 2048, "mem_116208")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:507:5-56
        // futhark/microgpt.fut:507:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116208.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_115924, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:507:5-56
        if (futrts_adam_opt_w_11670(ctx, &ext_mem_116212, &ext_mem_116211, &ext_mem_116210, mem_116202, mem_116204, mem_116206, mem_116208, (int64_t) 16, (int64_t) 16, step_106053, lt_r_108190) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_116202, "mem_116202") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116204, "mem_116204") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116206, "mem_116206") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116208, "mem_116208") != 0)
            return 1;
        // futhark/microgpt.fut:509:5-56
        if (memblock_alloc(ctx, &mem_116213, (int64_t) 2048, "mem_116213")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:509:5-56
        // futhark/microgpt.fut:509:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116213.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114464.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:509:5-56
        if (memblock_alloc(ctx, &mem_116215, (int64_t) 2048, "mem_116215")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:509:5-56
        // futhark/microgpt.fut:509:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116215.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114500.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:509:5-56
        if (memblock_alloc(ctx, &mem_116217, (int64_t) 2048, "mem_116217")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:509:5-56
        // futhark/microgpt.fut:509:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116217.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114536.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:509:5-56
        if (memblock_alloc(ctx, &mem_116219, (int64_t) 2048, "mem_116219")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:509:5-56
        // futhark/microgpt.fut:509:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116219.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_115383, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:509:5-56
        if (futrts_adam_opt_w_11670(ctx, &ext_mem_116223, &ext_mem_116222, &ext_mem_116221, mem_116213, mem_116215, mem_116217, mem_116219, (int64_t) 16, (int64_t) 16, step_106053, lt_r_108190) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_116213, "mem_116213") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116215, "mem_116215") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116217, "mem_116217") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116219, "mem_116219") != 0)
            return 1;
        // futhark/microgpt.fut:511:5-52
        if (memblock_alloc(ctx, &mem_116224, (int64_t) 8192, "mem_116224")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:511:5-52
        // futhark/microgpt.fut:511:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116224.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114480.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:511:5-52
        if (memblock_alloc(ctx, &mem_116226, (int64_t) 8192, "mem_116226")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:511:5-52
        // futhark/microgpt.fut:511:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116226.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114516.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:511:5-52
        if (memblock_alloc(ctx, &mem_116228, (int64_t) 8192, "mem_116228")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:511:5-52
        // futhark/microgpt.fut:511:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116228.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114552.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:511:5-52
        if (memblock_alloc(ctx, &mem_116230, (int64_t) 8192, "mem_116230")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:511:5-52
        // futhark/microgpt.fut:511:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116230.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_116110, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:511:5-52
        if (futrts_adam_opt_w_11669(ctx, &ext_mem_116234, &ext_mem_116233, &ext_mem_116232, mem_116224, mem_116226, mem_116228, mem_116230, (int64_t) 64, (int64_t) 16, step_106053, lt_r_108190) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_116224, "mem_116224") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116226, "mem_116226") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116228, "mem_116228") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116230, "mem_116230") != 0)
            return 1;
        // futhark/microgpt.fut:513:5-60
        if (memblock_alloc(ctx, &mem_116235, (int64_t) 8192, "mem_116235")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:513:5-60
        // futhark/microgpt.fut:513:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116235.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_114456.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:513:5-60
        if (memblock_alloc(ctx, &mem_116237, (int64_t) 8192, "mem_116237")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:513:5-60
        // futhark/microgpt.fut:513:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116237.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_114492.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:513:5-60
        if (memblock_alloc(ctx, &mem_116239, (int64_t) 8192, "mem_116239")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:513:5-60
        // futhark/microgpt.fut:513:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116239.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_114528.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:513:5-60
        if (memblock_alloc(ctx, &mem_116241, (int64_t) 8192, "mem_116241")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:513:5-60
        // futhark/microgpt.fut:513:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116241.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_115250, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:513:5-60
        if (futrts_adam_opt_w_11669(ctx, &ext_mem_116245, &ext_mem_116244, &ext_mem_116243, mem_116235, mem_116237, mem_116239, mem_116241, (int64_t) 16, (int64_t) 64, step_106053, lt_r_108190) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_116235, "mem_116235") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116237, "mem_116237") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116239, "mem_116239") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116241, "mem_116241") != 0)
            return 1;
        // futhark/microgpt.fut:515:5-56
        if (memblock_alloc(ctx, &mem_116246, (int64_t) 3456, "mem_116246")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:515:5-56
        // futhark/microgpt.fut:515:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116246.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114488.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:515:5-56
        if (memblock_alloc(ctx, &mem_116248, (int64_t) 3456, "mem_116248")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:515:5-56
        // futhark/microgpt.fut:515:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116248.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114524.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:515:5-56
        if (memblock_alloc(ctx, &mem_116250, (int64_t) 3456, "mem_116250")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:515:5-56
        // futhark/microgpt.fut:515:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116250.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_114560.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:515:5-56
        if (memblock_alloc(ctx, &mem_116252, (int64_t) 3456, "mem_116252")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:515:5-56
        // futhark/microgpt.fut:515:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_116252.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_116127, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:515:5-56
        if (futrts_adam_opt_w_11669(ctx, &ext_mem_116256, &ext_mem_116255, &ext_mem_116254, mem_116246, mem_116248, mem_116250, mem_116252, (int64_t) 27, (int64_t) 16, step_106053, lt_r_108190) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_116246, "mem_116246") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116248, "mem_116248") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116250, "mem_116250") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116252, "mem_116252") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116464, &ext_mem_116245, "ext_mem_116245") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116465, &ext_mem_116201, "ext_mem_116201") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116466, &ext_mem_116223, "ext_mem_116223") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116467, &ext_mem_116179, "ext_mem_116179") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116468, &ext_mem_116190, "ext_mem_116190") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116469, &ext_mem_116168, "ext_mem_116168") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116470, &ext_mem_116234, "ext_mem_116234") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116471, &ext_mem_116212, "ext_mem_116212") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116472, &ext_mem_116256, "ext_mem_116256") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116473, &ext_mem_116244, "ext_mem_116244") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116474, &ext_mem_116200, "ext_mem_116200") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116475, &ext_mem_116222, "ext_mem_116222") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116476, &ext_mem_116178, "ext_mem_116178") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116477, &ext_mem_116189, "ext_mem_116189") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116478, &ext_mem_116167, "ext_mem_116167") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116479, &ext_mem_116233, "ext_mem_116233") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116480, &ext_mem_116211, "ext_mem_116211") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116481, &ext_mem_116255, "ext_mem_116255") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116482, &ext_mem_116243, "ext_mem_116243") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116483, &ext_mem_116199, "ext_mem_116199") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116484, &ext_mem_116221, "ext_mem_116221") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116485, &ext_mem_116177, "ext_mem_116177") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116486, &ext_mem_116188, "ext_mem_116188") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116487, &ext_mem_116166, "ext_mem_116166") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116488, &ext_mem_116232, "ext_mem_116232") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116489, &ext_mem_116210, "ext_mem_116210") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_116490, &ext_mem_116254, "ext_mem_116254") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114456, &mem_param_tmp_116464, "mem_param_tmp_116464") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114460, &mem_param_tmp_116465, "mem_param_tmp_116465") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114464, &mem_param_tmp_116466, "mem_param_tmp_116466") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114468, &mem_param_tmp_116467, "mem_param_tmp_116467") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114472, &mem_param_tmp_116468, "mem_param_tmp_116468") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114476, &mem_param_tmp_116469, "mem_param_tmp_116469") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114480, &mem_param_tmp_116470, "mem_param_tmp_116470") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114484, &mem_param_tmp_116471, "mem_param_tmp_116471") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114488, &mem_param_tmp_116472, "mem_param_tmp_116472") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114492, &mem_param_tmp_116473, "mem_param_tmp_116473") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114496, &mem_param_tmp_116474, "mem_param_tmp_116474") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114500, &mem_param_tmp_116475, "mem_param_tmp_116475") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114504, &mem_param_tmp_116476, "mem_param_tmp_116476") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114508, &mem_param_tmp_116477, "mem_param_tmp_116477") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114512, &mem_param_tmp_116478, "mem_param_tmp_116478") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114516, &mem_param_tmp_116479, "mem_param_tmp_116479") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114520, &mem_param_tmp_116480, "mem_param_tmp_116480") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114524, &mem_param_tmp_116481, "mem_param_tmp_116481") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114528, &mem_param_tmp_116482, "mem_param_tmp_116482") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114532, &mem_param_tmp_116483, "mem_param_tmp_116483") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114536, &mem_param_tmp_116484, "mem_param_tmp_116484") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114540, &mem_param_tmp_116485, "mem_param_tmp_116485") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114544, &mem_param_tmp_116486, "mem_param_tmp_116486") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114548, &mem_param_tmp_116487, "mem_param_tmp_116487") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114552, &mem_param_tmp_116488, "mem_param_tmp_116488") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114556, &mem_param_tmp_116489, "mem_param_tmp_116489") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_114560, &mem_param_tmp_116490, "mem_param_tmp_116490") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_116364, &mem_param_114456, "mem_param_114456") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116363, &mem_param_114460, "mem_param_114460") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116362, &mem_param_114464, "mem_param_114464") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116361, &mem_param_114468, "mem_param_114468") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116360, &mem_param_114472, "mem_param_114472") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116359, &mem_param_114476, "mem_param_114476") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116358, &mem_param_114480, "mem_param_114480") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116357, &mem_param_114484, "mem_param_114484") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116356, &mem_param_114488, "mem_param_114488") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116355, &mem_param_114492, "mem_param_114492") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116354, &mem_param_114496, "mem_param_114496") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116353, &mem_param_114500, "mem_param_114500") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116352, &mem_param_114504, "mem_param_114504") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116351, &mem_param_114508, "mem_param_114508") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116350, &mem_param_114512, "mem_param_114512") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116349, &mem_param_114516, "mem_param_114516") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116348, &mem_param_114520, "mem_param_114520") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116347, &mem_param_114524, "mem_param_114524") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116346, &mem_param_114528, "mem_param_114528") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116345, &mem_param_114532, "mem_param_114532") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116344, &mem_param_114536, "mem_param_114536") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116343, &mem_param_114540, "mem_param_114540") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116342, &mem_param_114544, "mem_param_114544") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116341, &mem_param_114548, "mem_param_114548") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116340, &mem_param_114552, "mem_param_114552") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116339, &mem_param_114556, "mem_param_114556") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_116338, &mem_param_114560, "mem_param_114560") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116437, &ext_mem_116359, "ext_mem_116359") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116438, &ext_mem_116361, "ext_mem_116361") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116439, &ext_mem_116360, "ext_mem_116360") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116440, &ext_mem_116363, "ext_mem_116363") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116441, &ext_mem_116357, "ext_mem_116357") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116442, &ext_mem_116362, "ext_mem_116362") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116443, &ext_mem_116358, "ext_mem_116358") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116444, &ext_mem_116364, "ext_mem_116364") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116445, &ext_mem_116356, "ext_mem_116356") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116446, &ext_mem_116350, "ext_mem_116350") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116447, &ext_mem_116352, "ext_mem_116352") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116448, &ext_mem_116351, "ext_mem_116351") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116449, &ext_mem_116354, "ext_mem_116354") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116450, &ext_mem_116348, "ext_mem_116348") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116451, &ext_mem_116353, "ext_mem_116353") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116452, &ext_mem_116349, "ext_mem_116349") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116453, &ext_mem_116355, "ext_mem_116355") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116454, &ext_mem_116347, "ext_mem_116347") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116455, &ext_mem_116341, "ext_mem_116341") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116456, &ext_mem_116343, "ext_mem_116343") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116457, &ext_mem_116342, "ext_mem_116342") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116458, &ext_mem_116345, "ext_mem_116345") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116459, &ext_mem_116339, "ext_mem_116339") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116460, &ext_mem_116344, "ext_mem_116344") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116461, &ext_mem_116340, "ext_mem_116340") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116462, &ext_mem_116346, "ext_mem_116346") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116463, &ext_mem_116338, "ext_mem_116338") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116921, &mem_out_116437, "mem_out_116437") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116922, &mem_out_116438, "mem_out_116438") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116923, &mem_out_116439, "mem_out_116439") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116924, &mem_out_116440, "mem_out_116440") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116925, &mem_out_116441, "mem_out_116441") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116926, &mem_out_116442, "mem_out_116442") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116927, &mem_out_116443, "mem_out_116443") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116928, &mem_out_116444, "mem_out_116444") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116929, &mem_out_116445, "mem_out_116445") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116930, &mem_out_116446, "mem_out_116446") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116931, &mem_out_116447, "mem_out_116447") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116932, &mem_out_116448, "mem_out_116448") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116933, &mem_out_116449, "mem_out_116449") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116934, &mem_out_116450, "mem_out_116450") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116935, &mem_out_116451, "mem_out_116451") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116936, &mem_out_116452, "mem_out_116452") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116937, &mem_out_116453, "mem_out_116453") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116938, &mem_out_116454, "mem_out_116454") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116939, &mem_out_116455, "mem_out_116455") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116940, &mem_out_116456, "mem_out_116456") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116941, &mem_out_116457, "mem_out_116457") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116942, &mem_out_116458, "mem_out_116458") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116943, &mem_out_116459, "mem_out_116459") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116944, &mem_out_116460, "mem_out_116460") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116945, &mem_out_116461, "mem_out_116461") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116946, &mem_out_116462, "mem_out_116462") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_116947, &mem_out_116463, "mem_out_116463") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_114561);
        free(mem_114562);
        free(mem_114571);
        free(mem_114578);
        free(mem_114593);
        free(mem_114594);
        free(mem_114602);
        free(mem_114609);
        free(mem_114623);
        free(mem_114624);
        free(mem_114632);
        free(mem_114639);
        free(mem_114653);
        free(mem_114654);
        free(mem_114655);
        free(mem_114668);
        free(mem_114669);
        free(mem_114670);
        free(mem_114701);
        free(mem_114702);
        free(mem_114703);
        free(mem_114719);
        free(mem_114720);
        free(mem_114721);
        free(mem_114734);
        free(mem_114735);
        free(mem_114736);
        free(mem_114782);
        free(mem_114783);
        free(mem_114794);
        free(mem_114795);
        free(mem_114804);
        free(mem_114805);
        free(mem_114826);
        free(mem_114831);
        free(mem_114842);
        free(mem_114847);
        free(mem_114854);
        free(mem_114861);
        free(mem_114872);
        free(mem_114877);
        free(mem_114888);
        free(mem_114893);
        free(mem_114914);
        free(mem_114919);
        free(mem_114930);
        free(mem_114935);
        free(mem_114946);
        free(mem_114951);
        free(mem_114962);
        free(mem_114963);
        free(mem_114971);
        free(mem_114978);
        free(mem_114992);
        free(mem_114997);
        free(mem_115008);
        free(mem_115013);
        free(mem_115024);
        free(mem_115029);
        free(mem_115040);
        free(mem_115045);
        free(mem_115056);
        free(mem_115061);
        free(mem_115072);
        free(mem_115073);
        free(mem_115082);
        free(mem_115087);
        free(mem_115091);
        free(mem_115098);
        free(mem_115120);
        free(mem_115125);
        free(mem_115136);
        free(mem_115137);
        free(mem_115145);
        free(mem_115159);
        free(mem_115165);
        free(mem_115170);
        free(mem_115186);
        free(mem_115191);
        free(mem_115202);
        free(mem_115207);
        free(mem_115218);
        free(mem_115223);
        free(mem_115234);
        free(mem_115239);
        free(mem_115250);
        free(mem_115251);
        free(mem_115260);
        free(mem_115261);
        free(mem_115282);
        free(mem_115287);
        free(mem_115298);
        free(mem_115303);
        free(mem_115314);
        free(mem_115319);
        free(mem_115330);
        free(mem_115331);
        free(mem_115344);
        free(mem_115351);
        free(mem_115356);
        free(mem_115367);
        free(mem_115372);
        free(mem_115383);
        free(mem_115384);
        free(mem_115393);
        free(mem_115394);
        free(mem_115415);
        free(mem_115416);
        free(mem_115427);
        free(mem_115428);
        free(mem_115437);
        free(mem_115444);
        free(mem_115469);
        free(mem_115470);
        free(mem_115471);
        free(mem_115486);
        free(mem_115487);
        free(mem_115488);
        free(mem_115500);
        free(mem_115507);
        free(mem_115514);
        free(mem_115521);
        free(mem_115553);
        free(mem_115554);
        free(mem_115555);
        free(mem_115556);
        free(mem_115557);
        free(mem_115581);
        free(mem_115582);
        free(mem_115583);
        free(mem_115584);
        free(mem_115585);
        free(mem_115604);
        free(mem_115605);
        free(mem_115618);
        free(mem_115666);
        free(mem_115672);
        free(mem_115677);
        free(mem_115693);
        free(mem_115694);
        free(mem_115703);
        free(mem_115704);
        free(mem_115725);
        free(mem_115731);
        free(mem_115736);
        free(mem_115752);
        free(mem_115757);
        free(mem_115768);
        free(mem_115774);
        free(mem_115779);
        free(mem_115795);
        free(mem_115801);
        free(mem_115806);
        free(mem_115822);
        free(mem_115823);
        free(mem_115834);
        free(mem_115835);
        free(mem_115844);
        free(mem_115845);
        free(mem_115876);
        free(mem_115877);
        free(mem_115878);
        free(mem_115891);
        free(mem_115892);
        free(mem_115893);
        free(mem_115924);
        free(mem_115925);
        free(mem_115926);
        free(mem_115927);
        free(mem_115944);
        free(mem_115945);
        free(mem_115946);
        free(mem_115947);
        free(mem_115988);
        free(mem_115993);
        free(mem_116004);
        free(mem_116005);
        free(mem_116018);
        free(mem_116025);
        free(mem_116030);
        free(mem_116041);
        free(mem_116046);
        free(mem_116057);
        free(mem_116058);
        free(mem_116071);
        free(mem_116078);
        free(mem_116079);
        free(mem_116088);
        free(mem_116089);
        free(mem_116110);
        free(mem_116115);
        free(mem_116126);
        free(mem_116127);
        free(mem_116136);
        free(mem_116137);
        if (memblock_unref(ctx, &mem_param_tmp_116490, "mem_param_tmp_116490") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116489, "mem_param_tmp_116489") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116488, "mem_param_tmp_116488") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116487, "mem_param_tmp_116487") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116486, "mem_param_tmp_116486") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116485, "mem_param_tmp_116485") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116484, "mem_param_tmp_116484") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116483, "mem_param_tmp_116483") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116482, "mem_param_tmp_116482") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116481, "mem_param_tmp_116481") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116480, "mem_param_tmp_116480") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116479, "mem_param_tmp_116479") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116478, "mem_param_tmp_116478") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116477, "mem_param_tmp_116477") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116476, "mem_param_tmp_116476") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116475, "mem_param_tmp_116475") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116474, "mem_param_tmp_116474") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116473, "mem_param_tmp_116473") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116472, "mem_param_tmp_116472") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116471, "mem_param_tmp_116471") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116470, "mem_param_tmp_116470") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116469, "mem_param_tmp_116469") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116468, "mem_param_tmp_116468") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116467, "mem_param_tmp_116467") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116466, "mem_param_tmp_116466") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116465, "mem_param_tmp_116465") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_116464, "mem_param_tmp_116464") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116254, "ext_mem_116254") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116255, "ext_mem_116255") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116256, "ext_mem_116256") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116252, "mem_116252") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116250, "mem_116250") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116248, "mem_116248") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116246, "mem_116246") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116243, "ext_mem_116243") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116244, "ext_mem_116244") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116245, "ext_mem_116245") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116241, "mem_116241") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116239, "mem_116239") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116237, "mem_116237") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116235, "mem_116235") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116232, "ext_mem_116232") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116233, "ext_mem_116233") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116234, "ext_mem_116234") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116230, "mem_116230") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116228, "mem_116228") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116226, "mem_116226") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116224, "mem_116224") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116221, "ext_mem_116221") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116222, "ext_mem_116222") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116223, "ext_mem_116223") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116219, "mem_116219") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116217, "mem_116217") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116215, "mem_116215") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116213, "mem_116213") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116210, "ext_mem_116210") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116211, "ext_mem_116211") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116212, "ext_mem_116212") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116208, "mem_116208") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116206, "mem_116206") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116204, "mem_116204") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116202, "mem_116202") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116199, "ext_mem_116199") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116200, "ext_mem_116200") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116201, "ext_mem_116201") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116197, "mem_116197") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116195, "mem_116195") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116193, "mem_116193") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116191, "mem_116191") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116188, "ext_mem_116188") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116189, "ext_mem_116189") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116190, "ext_mem_116190") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116186, "mem_116186") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116184, "mem_116184") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116182, "mem_116182") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116180, "mem_116180") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116177, "ext_mem_116177") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116178, "ext_mem_116178") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116179, "ext_mem_116179") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116175, "mem_116175") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116173, "mem_116173") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116171, "mem_116171") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116169, "mem_116169") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116166, "ext_mem_116166") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116167, "ext_mem_116167") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116168, "ext_mem_116168") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116164, "mem_116164") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116162, "mem_116162") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116160, "mem_116160") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_116158, "mem_116158") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114560, "mem_param_114560") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114556, "mem_param_114556") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114552, "mem_param_114552") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114548, "mem_param_114548") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114544, "mem_param_114544") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114540, "mem_param_114540") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114536, "mem_param_114536") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114532, "mem_param_114532") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114528, "mem_param_114528") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114524, "mem_param_114524") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114520, "mem_param_114520") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114516, "mem_param_114516") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114512, "mem_param_114512") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114508, "mem_param_114508") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114504, "mem_param_114504") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114500, "mem_param_114500") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114496, "mem_param_114496") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114492, "mem_param_114492") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114488, "mem_param_114488") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114484, "mem_param_114484") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114480, "mem_param_114480") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114476, "mem_param_114476") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114472, "mem_param_114472") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114468, "mem_param_114468") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114464, "mem_param_114464") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114460, "mem_param_114460") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_114456, "mem_param_114456") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116338, "ext_mem_116338") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116339, "ext_mem_116339") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116340, "ext_mem_116340") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116341, "ext_mem_116341") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116342, "ext_mem_116342") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116343, "ext_mem_116343") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116344, "ext_mem_116344") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116345, "ext_mem_116345") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116346, "ext_mem_116346") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116347, "ext_mem_116347") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116348, "ext_mem_116348") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116349, "ext_mem_116349") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116350, "ext_mem_116350") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116351, "ext_mem_116351") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116352, "ext_mem_116352") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116353, "ext_mem_116353") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116354, "ext_mem_116354") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116355, "ext_mem_116355") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116356, "ext_mem_116356") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116357, "ext_mem_116357") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116358, "ext_mem_116358") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116359, "ext_mem_116359") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116360, "ext_mem_116360") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116361, "ext_mem_116361") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116362, "ext_mem_116362") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116363, "ext_mem_116363") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_116364, "ext_mem_116364") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116463, "mem_out_116463") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116462, "mem_out_116462") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116461, "mem_out_116461") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116460, "mem_out_116460") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116459, "mem_out_116459") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116458, "mem_out_116458") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116457, "mem_out_116457") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116456, "mem_out_116456") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116455, "mem_out_116455") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116454, "mem_out_116454") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116453, "mem_out_116453") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116452, "mem_out_116452") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116451, "mem_out_116451") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116450, "mem_out_116450") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116449, "mem_out_116449") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116448, "mem_out_116448") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116447, "mem_out_116447") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116446, "mem_out_116446") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116445, "mem_out_116445") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116444, "mem_out_116444") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116443, "mem_out_116443") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116442, "mem_out_116442") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116441, "mem_out_116441") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116440, "mem_out_116440") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116439, "mem_out_116439") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116438, "mem_out_116438") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116437, "mem_out_116437") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_117143, struct memblock *mem_out_p_117144, struct memblock *mem_out_p_117145, struct memblock *mem_out_p_117146, struct memblock *mem_out_p_117147, struct memblock *mem_out_p_117148, struct memblock *mem_out_p_117149, struct memblock *mem_out_p_117150, struct memblock *mem_out_p_117151)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_116445;
    
    mem_out_116445.references = NULL;
    
    struct memblock mem_out_116444;
    
    mem_out_116444.references = NULL;
    
    struct memblock mem_out_116443;
    
    mem_out_116443.references = NULL;
    
    struct memblock mem_out_116442;
    
    mem_out_116442.references = NULL;
    
    struct memblock mem_out_116441;
    
    mem_out_116441.references = NULL;
    
    struct memblock mem_out_116440;
    
    mem_out_116440.references = NULL;
    
    struct memblock mem_out_116439;
    
    mem_out_116439.references = NULL;
    
    struct memblock mem_out_116438;
    
    mem_out_116438.references = NULL;
    
    struct memblock mem_out_116437;
    
    mem_out_116437.references = NULL;
    
    struct memblock mem_114414 = ctx->constants->mem_114414;
    struct memblock mem_114415 = ctx->constants->mem_114415;
    struct memblock mem_114416 = ctx->constants->mem_114416;
    struct memblock mem_114417 = ctx->constants->mem_114417;
    struct memblock mem_114418 = ctx->constants->mem_114418;
    struct memblock mem_114419 = ctx->constants->mem_114419;
    struct memblock mem_114420 = ctx->constants->mem_114420;
    struct memblock mem_114421 = ctx->constants->mem_114421;
    struct memblock mem_114422 = ctx->constants->mem_114422;
    
    if (memblock_set(ctx, &mem_out_116437, &mem_114421, "mem_114421") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116438, &mem_114417, "mem_114417") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116439, &mem_114419, "mem_114419") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116440, &mem_114415, "mem_114415") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116441, &mem_114416, "mem_114416") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116442, &mem_114414, "mem_114414") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116443, &mem_114420, "mem_114420") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116444, &mem_114418, "mem_114418") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_116445, &mem_114422, "mem_114422") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_117143, &mem_out_116437, "mem_out_116437") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_117144, &mem_out_116438, "mem_out_116438") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_117145, &mem_out_116439, "mem_out_116439") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_117146, &mem_out_116440, "mem_out_116440") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_117147, &mem_out_116441, "mem_out_116441") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_117148, &mem_out_116442, "mem_out_116442") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_117149, &mem_out_116443, "mem_out_116443") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_117150, &mem_out_116444, "mem_out_116444") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_117151, &mem_out_116445, "mem_out_116445") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_116445, "mem_out_116445") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116444, "mem_out_116444") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116443, "mem_out_116443") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116442, "mem_out_116442") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116441, "mem_out_116441") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116440, "mem_out_116440") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116439, "mem_out_116439") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116438, "mem_out_116438") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_116437, "mem_out_116437") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_cal_loss(struct futhark_context *ctx, struct futhark_opaque_tup2_f64_arr1d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3)
{
    double prim_out_116438 = 0.0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_116437;
    
    mem_out_116437.references = NULL;
    
    struct memblock mask_mem_114434;
    
    mask_mem_114434.references = NULL;
    
    struct memblock target_mem_114433;
    
    target_mem_114433.references = NULL;
    
    struct memblock tokens_mem_114432;
    
    tokens_mem_114432.references = NULL;
    
    struct memblock wvoc_mem_114431;
    
    wvoc_mem_114431.references = NULL;
    
    struct memblock wval_mem_114430;
    
    wval_mem_114430.references = NULL;
    
    struct memblock wup_mem_114429;
    
    wup_mem_114429.references = NULL;
    
    struct memblock wte_mem_114428;
    
    wte_mem_114428.references = NULL;
    
    struct memblock wqry_mem_114427;
    
    wqry_mem_114427.references = NULL;
    
    struct memblock wpe_mem_114426;
    
    wpe_mem_114426.references = NULL;
    
    struct memblock wout_mem_114425;
    
    wout_mem_114425.references = NULL;
    
    struct memblock wkey_mem_114424;
    
    wkey_mem_114424.references = NULL;
    
    struct memblock wdown_mem_114423;
    
    wdown_mem_114423.references = NULL;
    wdown_mem_114423 = in0->v0->mem;
    wkey_mem_114424 = in0->v1->mem;
    wout_mem_114425 = in0->v2->mem;
    wpe_mem_114426 = in0->v3->mem;
    wqry_mem_114427 = in0->v4->mem;
    wte_mem_114428 = in0->v5->mem;
    wup_mem_114429 = in0->v6->mem;
    wval_mem_114430 = in0->v7->mem;
    wvoc_mem_114431 = in0->v8->mem;
    tokens_mem_114432 = in1->mem;
    target_mem_114433 = in2->mem;
    mask_mem_114434 = in3->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && (((int64_t) 16 == in2->shape[0] && (int64_t) 27 == in2->shape[1]) && ((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_cal_loss(ctx, &mem_out_116437, &prim_out_116438, wdown_mem_114423, wkey_mem_114424, wout_mem_114425, wpe_mem_114426, wqry_mem_114427, wte_mem_114428, wup_mem_114429, wval_mem_114430, wvoc_mem_114431, tokens_mem_114432, target_mem_114433, mask_mem_114434);
        if (ret == 0) {
            struct memblock mem_114414 = ctx->constants->mem_114414;
            struct memblock mem_114415 = ctx->constants->mem_114415;
            struct memblock mem_114416 = ctx->constants->mem_114416;
            struct memblock mem_114417 = ctx->constants->mem_114417;
            struct memblock mem_114418 = ctx->constants->mem_114418;
            struct memblock mem_114419 = ctx->constants->mem_114419;
            struct memblock mem_114420 = ctx->constants->mem_114420;
            struct memblock mem_114421 = ctx->constants->mem_114421;
            struct memblock mem_114422 = ctx->constants->mem_114422;
            
            assert((*out = (struct futhark_opaque_tup2_f64_arr1d_f64 *) malloc(sizeof(struct futhark_opaque_tup2_f64_arr1d_f64))) != NULL);
            (*out)->v0 = prim_out_116438;
            assert(((*out)->v1 = (struct futhark_f64_1d *) malloc(sizeof(struct futhark_f64_1d))) != NULL);
            (*out)->v1->mem = mem_out_116437;
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
    
    struct memblock mem_out_116437;
    
    mem_out_116437.references = NULL;
    
    struct memblock mask_mem_114433;
    
    mask_mem_114433.references = NULL;
    
    struct memblock tokens_mem_114432;
    
    tokens_mem_114432.references = NULL;
    
    struct memblock wvoc_mem_114431;
    
    wvoc_mem_114431.references = NULL;
    
    struct memblock wval_mem_114430;
    
    wval_mem_114430.references = NULL;
    
    struct memblock wup_mem_114429;
    
    wup_mem_114429.references = NULL;
    
    struct memblock wte_mem_114428;
    
    wte_mem_114428.references = NULL;
    
    struct memblock wqry_mem_114427;
    
    wqry_mem_114427.references = NULL;
    
    struct memblock wpe_mem_114426;
    
    wpe_mem_114426.references = NULL;
    
    struct memblock wout_mem_114425;
    
    wout_mem_114425.references = NULL;
    
    struct memblock wkey_mem_114424;
    
    wkey_mem_114424.references = NULL;
    
    struct memblock wdown_mem_114423;
    
    wdown_mem_114423.references = NULL;
    wdown_mem_114423 = in0->v0->mem;
    wkey_mem_114424 = in0->v1->mem;
    wout_mem_114425 = in0->v2->mem;
    wpe_mem_114426 = in0->v3->mem;
    wqry_mem_114427 = in0->v4->mem;
    wte_mem_114428 = in0->v5->mem;
    wup_mem_114429 = in0->v6->mem;
    wval_mem_114430 = in0->v7->mem;
    wvoc_mem_114431 = in0->v8->mem;
    tokens_mem_114432 = in1->mem;
    mask_mem_114433 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_116437, wdown_mem_114423, wkey_mem_114424, wout_mem_114425, wpe_mem_114426, wqry_mem_114427, wte_mem_114428, wup_mem_114429, wval_mem_114430, wvoc_mem_114431, tokens_mem_114432, mask_mem_114433);
        if (ret == 0) {
            struct memblock mem_114414 = ctx->constants->mem_114414;
            struct memblock mem_114415 = ctx->constants->mem_114415;
            struct memblock mem_114416 = ctx->constants->mem_114416;
            struct memblock mem_114417 = ctx->constants->mem_114417;
            struct memblock mem_114418 = ctx->constants->mem_114418;
            struct memblock mem_114419 = ctx->constants->mem_114419;
            struct memblock mem_114420 = ctx->constants->mem_114420;
            struct memblock mem_114421 = ctx->constants->mem_114421;
            struct memblock mem_114422 = ctx->constants->mem_114422;
            
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_116437;
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
    
    struct memblock mem_out_116445;
    
    mem_out_116445.references = NULL;
    
    struct memblock mem_out_116444;
    
    mem_out_116444.references = NULL;
    
    struct memblock mem_out_116443;
    
    mem_out_116443.references = NULL;
    
    struct memblock mem_out_116442;
    
    mem_out_116442.references = NULL;
    
    struct memblock mem_out_116441;
    
    mem_out_116441.references = NULL;
    
    struct memblock mem_out_116440;
    
    mem_out_116440.references = NULL;
    
    struct memblock mem_out_116439;
    
    mem_out_116439.references = NULL;
    
    struct memblock mem_out_116438;
    
    mem_out_116438.references = NULL;
    
    struct memblock mem_out_116437;
    
    mem_out_116437.references = NULL;
    
    struct memblock wvoc_mem_114431;
    
    wvoc_mem_114431.references = NULL;
    
    struct memblock wdown_mem_114430;
    
    wdown_mem_114430.references = NULL;
    
    struct memblock wup_mem_114429;
    
    wup_mem_114429.references = NULL;
    
    struct memblock wout_mem_114428;
    
    wout_mem_114428.references = NULL;
    
    struct memblock wval_mem_114427;
    
    wval_mem_114427.references = NULL;
    
    struct memblock wkey_mem_114426;
    
    wkey_mem_114426.references = NULL;
    
    struct memblock wqry_mem_114425;
    
    wqry_mem_114425.references = NULL;
    
    struct memblock wpe_mem_114424;
    
    wpe_mem_114424.references = NULL;
    
    struct memblock wte_mem_114423;
    
    wte_mem_114423.references = NULL;
    wte_mem_114423 = in0->mem;
    wpe_mem_114424 = in1->mem;
    wqry_mem_114425 = in2->mem;
    wkey_mem_114426 = in3->mem;
    wval_mem_114427 = in4->mem;
    wout_mem_114428 = in5->mem;
    wup_mem_114429 = in6->mem;
    wdown_mem_114430 = in7->mem;
    wvoc_mem_114431 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_116437, &mem_out_116438, &mem_out_116439, &mem_out_116440, &mem_out_116441, &mem_out_116442, &mem_out_116443, &mem_out_116444, &mem_out_116445, wte_mem_114423, wpe_mem_114424, wqry_mem_114425, wkey_mem_114426, wval_mem_114427, wout_mem_114428, wup_mem_114429, wdown_mem_114430, wvoc_mem_114431);
        if (ret == 0) {
            struct memblock mem_114414 = ctx->constants->mem_114414;
            struct memblock mem_114415 = ctx->constants->mem_114415;
            struct memblock mem_114416 = ctx->constants->mem_114416;
            struct memblock mem_114417 = ctx->constants->mem_114417;
            struct memblock mem_114418 = ctx->constants->mem_114418;
            struct memblock mem_114419 = ctx->constants->mem_114419;
            struct memblock mem_114420 = ctx->constants->mem_114420;
            struct memblock mem_114421 = ctx->constants->mem_114421;
            struct memblock mem_114422 = ctx->constants->mem_114422;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_116437;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_116438;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_116439;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_116440;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_116441;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_116442;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_116443;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_116444;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_116445;
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
    
    struct memblock mem_out_116463;
    
    mem_out_116463.references = NULL;
    
    struct memblock mem_out_116462;
    
    mem_out_116462.references = NULL;
    
    struct memblock mem_out_116461;
    
    mem_out_116461.references = NULL;
    
    struct memblock mem_out_116460;
    
    mem_out_116460.references = NULL;
    
    struct memblock mem_out_116459;
    
    mem_out_116459.references = NULL;
    
    struct memblock mem_out_116458;
    
    mem_out_116458.references = NULL;
    
    struct memblock mem_out_116457;
    
    mem_out_116457.references = NULL;
    
    struct memblock mem_out_116456;
    
    mem_out_116456.references = NULL;
    
    struct memblock mem_out_116455;
    
    mem_out_116455.references = NULL;
    
    struct memblock mem_out_116454;
    
    mem_out_116454.references = NULL;
    
    struct memblock mem_out_116453;
    
    mem_out_116453.references = NULL;
    
    struct memblock mem_out_116452;
    
    mem_out_116452.references = NULL;
    
    struct memblock mem_out_116451;
    
    mem_out_116451.references = NULL;
    
    struct memblock mem_out_116450;
    
    mem_out_116450.references = NULL;
    
    struct memblock mem_out_116449;
    
    mem_out_116449.references = NULL;
    
    struct memblock mem_out_116448;
    
    mem_out_116448.references = NULL;
    
    struct memblock mem_out_116447;
    
    mem_out_116447.references = NULL;
    
    struct memblock mem_out_116446;
    
    mem_out_116446.references = NULL;
    
    struct memblock mem_out_116445;
    
    mem_out_116445.references = NULL;
    
    struct memblock mem_out_116444;
    
    mem_out_116444.references = NULL;
    
    struct memblock mem_out_116443;
    
    mem_out_116443.references = NULL;
    
    struct memblock mem_out_116442;
    
    mem_out_116442.references = NULL;
    
    struct memblock mem_out_116441;
    
    mem_out_116441.references = NULL;
    
    struct memblock mem_out_116440;
    
    mem_out_116440.references = NULL;
    
    struct memblock mem_out_116439;
    
    mem_out_116439.references = NULL;
    
    struct memblock mem_out_116438;
    
    mem_out_116438.references = NULL;
    
    struct memblock mem_out_116437;
    
    mem_out_116437.references = NULL;
    
    struct memblock seqs_mem_114452;
    
    seqs_mem_114452.references = NULL;
    
    struct memblock dls_mem_114451;
    
    dls_mem_114451.references = NULL;
    
    struct memblock masks_mem_114450;
    
    masks_mem_114450.references = NULL;
    
    struct memblock wvoc_mem_114449;
    
    wvoc_mem_114449.references = NULL;
    
    struct memblock wval_mem_114448;
    
    wval_mem_114448.references = NULL;
    
    struct memblock wup_mem_114447;
    
    wup_mem_114447.references = NULL;
    
    struct memblock wte_mem_114446;
    
    wte_mem_114446.references = NULL;
    
    struct memblock wqry_mem_114445;
    
    wqry_mem_114445.references = NULL;
    
    struct memblock wpe_mem_114444;
    
    wpe_mem_114444.references = NULL;
    
    struct memblock wout_mem_114443;
    
    wout_mem_114443.references = NULL;
    
    struct memblock wkey_mem_114442;
    
    wkey_mem_114442.references = NULL;
    
    struct memblock wdown_mem_114441;
    
    wdown_mem_114441.references = NULL;
    
    struct memblock wvoc_mem_114440;
    
    wvoc_mem_114440.references = NULL;
    
    struct memblock wval_mem_114439;
    
    wval_mem_114439.references = NULL;
    
    struct memblock wup_mem_114438;
    
    wup_mem_114438.references = NULL;
    
    struct memblock wte_mem_114437;
    
    wte_mem_114437.references = NULL;
    
    struct memblock wqry_mem_114436;
    
    wqry_mem_114436.references = NULL;
    
    struct memblock wpe_mem_114435;
    
    wpe_mem_114435.references = NULL;
    
    struct memblock wout_mem_114434;
    
    wout_mem_114434.references = NULL;
    
    struct memblock wkey_mem_114433;
    
    wkey_mem_114433.references = NULL;
    
    struct memblock wdown_mem_114432;
    
    wdown_mem_114432.references = NULL;
    
    struct memblock wvoc_mem_114431;
    
    wvoc_mem_114431.references = NULL;
    
    struct memblock wval_mem_114430;
    
    wval_mem_114430.references = NULL;
    
    struct memblock wup_mem_114429;
    
    wup_mem_114429.references = NULL;
    
    struct memblock wte_mem_114428;
    
    wte_mem_114428.references = NULL;
    
    struct memblock wqry_mem_114427;
    
    wqry_mem_114427.references = NULL;
    
    struct memblock wpe_mem_114426;
    
    wpe_mem_114426.references = NULL;
    
    struct memblock wout_mem_114425;
    
    wout_mem_114425.references = NULL;
    
    struct memblock wkey_mem_114424;
    
    wkey_mem_114424.references = NULL;
    
    struct memblock wdown_mem_114423;
    
    wdown_mem_114423.references = NULL;
    wdown_mem_114423 = in0->v0->mem;
    wkey_mem_114424 = in0->v1->mem;
    wout_mem_114425 = in0->v2->mem;
    wpe_mem_114426 = in0->v3->mem;
    wqry_mem_114427 = in0->v4->mem;
    wte_mem_114428 = in0->v5->mem;
    wup_mem_114429 = in0->v6->mem;
    wval_mem_114430 = in0->v7->mem;
    wvoc_mem_114431 = in0->v8->mem;
    wdown_mem_114432 = in1->v0->mem;
    wkey_mem_114433 = in1->v1->mem;
    wout_mem_114434 = in1->v2->mem;
    wpe_mem_114435 = in1->v3->mem;
    wqry_mem_114436 = in1->v4->mem;
    wte_mem_114437 = in1->v5->mem;
    wup_mem_114438 = in1->v6->mem;
    wval_mem_114439 = in1->v7->mem;
    wvoc_mem_114440 = in1->v8->mem;
    wdown_mem_114441 = in2->v0->mem;
    wkey_mem_114442 = in2->v1->mem;
    wout_mem_114443 = in2->v2->mem;
    wpe_mem_114444 = in2->v3->mem;
    wqry_mem_114445 = in2->v4->mem;
    wte_mem_114446 = in2->v5->mem;
    wup_mem_114447 = in2->v6->mem;
    wval_mem_114448 = in2->v7->mem;
    wvoc_mem_114449 = in2->v8->mem;
    masks_mem_114450 = in3->mem;
    dls_mem_114451 = in4->mem;
    seqs_mem_114452 = in5->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && (((int64_t) 500 == in3->shape[0] && ((int64_t) 16 == in3->shape[1] && (int64_t) 16 == in3->shape[2])) && ((int64_t) 500 == in4->shape[0] && ((int64_t) 500 == in5->shape[0] && (int64_t) 16 == in5->shape[1]))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_116437, &mem_out_116438, &mem_out_116439, &mem_out_116440, &mem_out_116441, &mem_out_116442, &mem_out_116443, &mem_out_116444, &mem_out_116445, &mem_out_116446, &mem_out_116447, &mem_out_116448, &mem_out_116449, &mem_out_116450, &mem_out_116451, &mem_out_116452, &mem_out_116453, &mem_out_116454, &mem_out_116455, &mem_out_116456, &mem_out_116457, &mem_out_116458, &mem_out_116459, &mem_out_116460, &mem_out_116461, &mem_out_116462, &mem_out_116463, wdown_mem_114423, wkey_mem_114424, wout_mem_114425, wpe_mem_114426, wqry_mem_114427, wte_mem_114428, wup_mem_114429, wval_mem_114430, wvoc_mem_114431, wdown_mem_114432, wkey_mem_114433, wout_mem_114434, wpe_mem_114435, wqry_mem_114436, wte_mem_114437, wup_mem_114438, wval_mem_114439, wvoc_mem_114440, wdown_mem_114441, wkey_mem_114442, wout_mem_114443, wpe_mem_114444, wqry_mem_114445, wte_mem_114446, wup_mem_114447, wval_mem_114448, wvoc_mem_114449, masks_mem_114450, dls_mem_114451, seqs_mem_114452);
        if (ret == 0) {
            struct memblock mem_114414 = ctx->constants->mem_114414;
            struct memblock mem_114415 = ctx->constants->mem_114415;
            struct memblock mem_114416 = ctx->constants->mem_114416;
            struct memblock mem_114417 = ctx->constants->mem_114417;
            struct memblock mem_114418 = ctx->constants->mem_114418;
            struct memblock mem_114419 = ctx->constants->mem_114419;
            struct memblock mem_114420 = ctx->constants->mem_114420;
            struct memblock mem_114421 = ctx->constants->mem_114421;
            struct memblock mem_114422 = ctx->constants->mem_114422;
            
            assert((*out = (struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_116437;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_116438;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_116439;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_116440;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_116441;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_116442;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_116443;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_116444;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_116445;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v9->mem = mem_out_116446;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v10->mem = mem_out_116447;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v11->mem = mem_out_116448;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v12->mem = mem_out_116449;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v13->mem = mem_out_116450;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v14->mem = mem_out_116451;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v15->mem = mem_out_116452;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v16->mem = mem_out_116453;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v17->mem = mem_out_116454;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v18->mem = mem_out_116455;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v19->mem = mem_out_116456;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v20->mem = mem_out_116457;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v21->mem = mem_out_116458;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v22->mem = mem_out_116459;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v23->mem = mem_out_116460;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v24->mem = mem_out_116461;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v25->mem = mem_out_116462;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v26->mem = mem_out_116463;
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
    
    struct memblock mem_out_116445;
    
    mem_out_116445.references = NULL;
    
    struct memblock mem_out_116444;
    
    mem_out_116444.references = NULL;
    
    struct memblock mem_out_116443;
    
    mem_out_116443.references = NULL;
    
    struct memblock mem_out_116442;
    
    mem_out_116442.references = NULL;
    
    struct memblock mem_out_116441;
    
    mem_out_116441.references = NULL;
    
    struct memblock mem_out_116440;
    
    mem_out_116440.references = NULL;
    
    struct memblock mem_out_116439;
    
    mem_out_116439.references = NULL;
    
    struct memblock mem_out_116438;
    
    mem_out_116438.references = NULL;
    
    struct memblock mem_out_116437;
    
    mem_out_116437.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_116437, &mem_out_116438, &mem_out_116439, &mem_out_116440, &mem_out_116441, &mem_out_116442, &mem_out_116443, &mem_out_116444, &mem_out_116445);
        if (ret == 0) {
            struct memblock mem_114414 = ctx->constants->mem_114414;
            struct memblock mem_114415 = ctx->constants->mem_114415;
            struct memblock mem_114416 = ctx->constants->mem_114416;
            struct memblock mem_114417 = ctx->constants->mem_114417;
            struct memblock mem_114418 = ctx->constants->mem_114418;
            struct memblock mem_114419 = ctx->constants->mem_114419;
            struct memblock mem_114420 = ctx->constants->mem_114420;
            struct memblock mem_114421 = ctx->constants->mem_114421;
            struct memblock mem_114422 = ctx->constants->mem_114422;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_116437;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_116438;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_116439;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_116440;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_116441;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_116442;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_116443;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_116444;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_116445;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
