
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
    struct memblock mem_143922;
    struct memblock mem_143923;
    struct memblock mem_143924;
    struct memblock mem_143925;
    struct memblock mem_143926;
    struct memblock mem_143927;
    struct memblock mem_143928;
    struct memblock mem_143929;
    struct memblock mem_143930;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_12710(struct futhark_context *ctx, struct memblock *mem_out_p_146550, struct memblock *mem_out_p_146551, struct memblock *mem_out_p_146552, struct memblock w_mem_143931, struct memblock mw_mem_143932, struct memblock vw_mem_143933, struct memblock dw_mem_143934, int64_t n_104410, int64_t m_104411, int64_t step_104416, double lt_r_104417);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_12711(struct futhark_context *ctx, struct memblock *mem_out_p_146555, struct memblock *mem_out_p_146556, struct memblock *mem_out_p_146557, struct memblock w_mem_143931, struct memblock mw_mem_143932, struct memblock vw_mem_143933, struct memblock dw_mem_143934, int64_t n_105443, int64_t m_105444, int64_t step_105449, double lt_r_105450);
FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_146560, double *out_prim_out_146561, struct memblock wdown_mem_143931, struct memblock wkey_mem_143932, struct memblock wout_mem_143933, struct memblock wpe_mem_143934, struct memblock wqry_mem_143935, struct memblock wte_mem_143936, struct memblock wup_mem_143937, struct memblock wval_mem_143938, struct memblock wvoc_mem_143939, struct memblock tokens_mem_143940, struct memblock target_mem_143941, struct memblock mask_mem_143942);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_146619, struct memblock wdown_mem_143931, struct memblock wkey_mem_143932, struct memblock wout_mem_143933, struct memblock wpe_mem_143934, struct memblock wqry_mem_143935, struct memblock wte_mem_143936, struct memblock wup_mem_143937, struct memblock wval_mem_143938, struct memblock wvoc_mem_143939, struct memblock tokens_mem_143940, struct memblock mask_mem_143941);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_146676, struct memblock *mem_out_p_146677, struct memblock *mem_out_p_146678, struct memblock *mem_out_p_146679, struct memblock *mem_out_p_146680, struct memblock *mem_out_p_146681, struct memblock *mem_out_p_146682, struct memblock *mem_out_p_146683, struct memblock *mem_out_p_146684, struct memblock wte_mem_143931, struct memblock wpe_mem_143932, struct memblock wqry_mem_143933, struct memblock wkey_mem_143934, struct memblock wval_mem_143935, struct memblock wout_mem_143936, struct memblock wup_mem_143937, struct memblock wdown_mem_143938, struct memblock wvoc_mem_143939);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_146685, struct memblock *mem_out_p_146686, struct memblock *mem_out_p_146687, struct memblock *mem_out_p_146688, struct memblock *mem_out_p_146689, struct memblock *mem_out_p_146690, struct memblock *mem_out_p_146691, struct memblock *mem_out_p_146692, struct memblock *mem_out_p_146693, struct memblock *mem_out_p_146694, struct memblock *mem_out_p_146695, struct memblock *mem_out_p_146696, struct memblock *mem_out_p_146697, struct memblock *mem_out_p_146698, struct memblock *mem_out_p_146699, struct memblock *mem_out_p_146700, struct memblock *mem_out_p_146701, struct memblock *mem_out_p_146702, struct memblock *mem_out_p_146703, struct memblock *mem_out_p_146704, struct memblock *mem_out_p_146705, struct memblock *mem_out_p_146706, struct memblock *mem_out_p_146707, struct memblock *mem_out_p_146708, struct memblock *mem_out_p_146709, struct memblock *mem_out_p_146710, struct memblock *mem_out_p_146711, struct memblock wdown_mem_143931, struct memblock wkey_mem_143932, struct memblock wout_mem_143933, struct memblock wpe_mem_143934, struct memblock wqry_mem_143935, struct memblock wte_mem_143936, struct memblock wup_mem_143937, struct memblock wval_mem_143938, struct memblock wvoc_mem_143939, struct memblock wdown_mem_143940, struct memblock wkey_mem_143941, struct memblock wout_mem_143942, struct memblock wpe_mem_143943, struct memblock wqry_mem_143944, struct memblock wte_mem_143945, struct memblock wup_mem_143946, struct memblock wval_mem_143947, struct memblock wvoc_mem_143948, struct memblock wdown_mem_143949, struct memblock wkey_mem_143950, struct memblock wout_mem_143951, struct memblock wpe_mem_143952, struct memblock wqry_mem_143953, struct memblock wte_mem_143954, struct memblock wup_mem_143955, struct memblock wval_mem_143956, struct memblock wvoc_mem_143957, struct memblock masks_mem_143958, struct memblock dls_mem_143959, struct memblock seqs_mem_143960);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_146930, struct memblock *mem_out_p_146931, struct memblock *mem_out_p_146932, struct memblock *mem_out_p_146933, struct memblock *mem_out_p_146934, struct memblock *mem_out_p_146935, struct memblock *mem_out_p_146936, struct memblock *mem_out_p_146937, struct memblock *mem_out_p_146938);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_143922 (ctx->constants->mem_143922)
    #define mem_143923 (ctx->constants->mem_143923)
    #define mem_143924 (ctx->constants->mem_143924)
    #define mem_143925 (ctx->constants->mem_143925)
    #define mem_143926 (ctx->constants->mem_143926)
    #define mem_143927 (ctx->constants->mem_143927)
    #define mem_143928 (ctx->constants->mem_143928)
    #define mem_143929 (ctx->constants->mem_143929)
    #define mem_143930 (ctx->constants->mem_143930)
    mem_143922.references = NULL;
    mem_143923.references = NULL;
    mem_143924.references = NULL;
    mem_143925.references = NULL;
    mem_143926.references = NULL;
    mem_143927.references = NULL;
    mem_143928.references = NULL;
    mem_143929.references = NULL;
    mem_143930.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143922, (int64_t) 3456, "mem_143922")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_146532 = 0; nest_i_146532 < (int64_t) 27; nest_i_146532++) {
        for (int64_t nest_i_146533 = 0; nest_i_146533 < (int64_t) 16; nest_i_146533++) {
            ((double *) mem_143922.mem)[nest_i_146532 * (int64_t) 16 + nest_i_146533] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143923, (int64_t) 2048, "mem_143923")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_146534 = 0; nest_i_146534 < (int64_t) 16; nest_i_146534++) {
        for (int64_t nest_i_146535 = 0; nest_i_146535 < (int64_t) 16; nest_i_146535++) {
            ((double *) mem_143923.mem)[nest_i_146534 * (int64_t) 16 + nest_i_146535] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143924, (int64_t) 2048, "mem_143924")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_146536 = 0; nest_i_146536 < (int64_t) 16; nest_i_146536++) {
        for (int64_t nest_i_146537 = 0; nest_i_146537 < (int64_t) 16; nest_i_146537++) {
            ((double *) mem_143924.mem)[nest_i_146536 * (int64_t) 16 + nest_i_146537] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143925, (int64_t) 2048, "mem_143925")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_146538 = 0; nest_i_146538 < (int64_t) 16; nest_i_146538++) {
        for (int64_t nest_i_146539 = 0; nest_i_146539 < (int64_t) 16; nest_i_146539++) {
            ((double *) mem_143925.mem)[nest_i_146538 * (int64_t) 16 + nest_i_146539] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143926, (int64_t) 2048, "mem_143926")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_146540 = 0; nest_i_146540 < (int64_t) 16; nest_i_146540++) {
        for (int64_t nest_i_146541 = 0; nest_i_146541 < (int64_t) 16; nest_i_146541++) {
            ((double *) mem_143926.mem)[nest_i_146540 * (int64_t) 16 + nest_i_146541] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143927, (int64_t) 2048, "mem_143927")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_146542 = 0; nest_i_146542 < (int64_t) 16; nest_i_146542++) {
        for (int64_t nest_i_146543 = 0; nest_i_146543 < (int64_t) 16; nest_i_146543++) {
            ((double *) mem_143927.mem)[nest_i_146542 * (int64_t) 16 + nest_i_146543] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143928, (int64_t) 8192, "mem_143928")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_146544 = 0; nest_i_146544 < (int64_t) 64; nest_i_146544++) {
        for (int64_t nest_i_146545 = 0; nest_i_146545 < (int64_t) 16; nest_i_146545++) {
            ((double *) mem_143928.mem)[nest_i_146544 * (int64_t) 16 + nest_i_146545] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143929, (int64_t) 8192, "mem_143929")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_146546 = 0; nest_i_146546 < (int64_t) 16; nest_i_146546++) {
        for (int64_t nest_i_146547 = 0; nest_i_146547 < (int64_t) 64; nest_i_146547++) {
            ((double *) mem_143929.mem)[nest_i_146546 * (int64_t) 64 + nest_i_146547] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143930, (int64_t) 3456, "mem_143930")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_146548 = 0; nest_i_146548 < (int64_t) 27; nest_i_146548++) {
        for (int64_t nest_i_146549 = 0; nest_i_146549 < (int64_t) 16; nest_i_146549++) {
            ((double *) mem_143930.mem)[nest_i_146548 * (int64_t) 16 + nest_i_146549] = 0.0;
        }
    }
    #undef mem_143922
    #undef mem_143923
    #undef mem_143924
    #undef mem_143925
    #undef mem_143926
    #undef mem_143927
    #undef mem_143928
    #undef mem_143929
    #undef mem_143930
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_143922, "ctx->constants->mem_143922") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_143923, "ctx->constants->mem_143923") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_143924, "ctx->constants->mem_143924") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_143925, "ctx->constants->mem_143925") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_143926, "ctx->constants->mem_143926") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_143927, "ctx->constants->mem_143927") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_143928, "ctx->constants->mem_143928") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_143929, "ctx->constants->mem_143929") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_143930, "ctx->constants->mem_143930") != 0)
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_12710(struct futhark_context *ctx, struct memblock *mem_out_p_146550, struct memblock *mem_out_p_146551, struct memblock *mem_out_p_146552, struct memblock w_mem_143931, struct memblock mw_mem_143932, struct memblock vw_mem_143933, struct memblock dw_mem_143934, int64_t n_104410, int64_t m_104411, int64_t step_104416, double lt_r_104417)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_143975_cached_sizze_146553 = 0;
    unsigned char *mem_143975 = NULL;
    int64_t mem_143978_cached_sizze_146554 = 0;
    unsigned char *mem_143978 = NULL;
    struct memblock mem_144013;
    
    mem_144013.references = NULL;
    
    struct memblock mem_143940;
    
    mem_143940.references = NULL;
    
    struct memblock mem_143937;
    
    mem_143937.references = NULL;
    
    struct memblock mem_out_146164;
    
    mem_out_146164.references = NULL;
    
    struct memblock mem_out_146163;
    
    mem_out_146163.references = NULL;
    
    struct memblock mem_out_146162;
    
    mem_out_146162.references = NULL;
    
    struct memblock mem_143922 = ctx->constants->mem_143922;
    struct memblock mem_143923 = ctx->constants->mem_143923;
    struct memblock mem_143924 = ctx->constants->mem_143924;
    struct memblock mem_143925 = ctx->constants->mem_143925;
    struct memblock mem_143926 = ctx->constants->mem_143926;
    struct memblock mem_143927 = ctx->constants->mem_143927;
    struct memblock mem_143928 = ctx->constants->mem_143928;
    struct memblock mem_143929 = ctx->constants->mem_143929;
    struct memblock mem_143930 = ctx->constants->mem_143930;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_143935 = (int64_t) 8 * n_104410;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_143936 = m_104411 * binop_x_143935;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143937, bytes_143936, "mem_143937")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143940, bytes_143936, "mem_143940")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142925 = 0; i_142925 < n_104410; i_142925++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142918 = 0; i_142918 < m_104411; i_142918++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_132033 = ((double *) mw_mem_143932.mem)[i_142925 * m_104411 + i_142918];
            
            // futhark/microgpt.fut:476:10-20
            
            double zp_lhs_132034 = 0.85 * zt_rhs_132033;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_132035 = ((double *) dw_mem_143934.mem)[i_142925 * m_104411 + i_142918];
            
            // futhark/microgpt.fut:476:35-45
            
            double zp_rhs_132036 = 0.15000000000000002 * zt_rhs_132035;
            
            // futhark/microgpt.fut:476:21-45
            
            double lifted_lambda_res_132037 = zp_lhs_132034 + zp_rhs_132036;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_132044 = ((double *) vw_mem_143933.mem)[i_142925 * m_104411 + i_142918];
            
            // futhark/microgpt.fut:478:10-20
            
            double zp_lhs_132045 = 0.99 * zt_rhs_132044;
            
            // futhark/microgpt.fut:478:35-45
            
            double zt_lhs_132047 = 1.0000000000000009e-2 * zt_rhs_132035;
            
            // futhark/microgpt.fut:478:46-56
            
            double zp_rhs_132048 = zt_rhs_132035 * zt_lhs_132047;
            
            // futhark/microgpt.fut:478:21-56
            
            double lifted_lambda_res_132049 = zp_lhs_132045 + zp_rhs_132048;
            
            ((double *) mem_143937.mem)[i_142925 * m_104411 + i_142918] = lifted_lambda_res_132049;
            ((double *) mem_143940.mem)[i_142925 * m_104411 + i_142918] = lifted_lambda_res_132037;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_109018 = sitofp_i64_f64(step_104416);
    
    // futhark/microgpt.fut:480:54-57
    
    double ztzt_rhs_109019 = 1.0 + i64_res_109018;
    
    // futhark/microgpt.fut:480:30-57
    
    double zm_rhs_109020 = fpow64(0.85, ztzt_rhs_109019);
    
    // futhark/microgpt.fut:480:23-57
    
    double zs_rhs_109021 = 1.0 - zm_rhs_109020;
    
    // futhark/microgpt.fut:482:31-58
    
    double zm_rhs_109059 = fpow64(0.99, ztzt_rhs_109019);
    
    // futhark/microgpt.fut:482:23-58
    
    double zs_rhs_109060 = 1.0 - zm_rhs_109059;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_143975_cached_sizze_146553 < bytes_143936) {
        err = lexical_realloc(ctx, &mem_143975, &mem_143975_cached_sizze_146553, bytes_143936);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143978_cached_sizze_146554 < bytes_143936) {
        err = lexical_realloc(ctx, &mem_143978, &mem_143978_cached_sizze_146554, bytes_143936);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142939 = 0; i_142939 < n_104410; i_142939++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142932 = 0; i_142932 < m_104411; i_142932++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_132069 = ((double *) mem_143940.mem)[i_142939 * m_104411 + i_142932];
            
            // futhark/microgpt.fut:480:18-57
            
            double lifted_lambda_res_132070 = zs_lhs_132069 / zs_rhs_109021;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_132077 = ((double *) mem_143937.mem)[i_142939 * m_104411 + i_142932];
            
            // futhark/microgpt.fut:482:18-58
            
            double lifted_lambda_res_132078 = zs_lhs_132077 / zs_rhs_109060;
            
            ((double *) mem_143975)[i_142939 * m_104411 + i_142932] = lifted_lambda_res_132078;
            ((double *) mem_143978)[i_142939 * m_104411 + i_142932] = lifted_lambda_res_132070;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_144013, bytes_143936, "mem_144013")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142948 = 0; i_142948 < n_104410; i_142948++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142944 = 0; i_142944 < m_104411; i_142944++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_108580 = ((double *) w_mem_143931.mem)[i_142948 * m_104411 + i_142944];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_108581 = ((double *) mem_143978)[i_142948 * m_104411 + i_142944];
            
            // futhark/microgpt.fut:484:21-34
            
            double zs_lhs_108582 = lt_r_104417 * zt_rhs_108581;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_108583 = ((double *) mem_143975)[i_142948 * m_104411 + i_142944];
            
            // futhark/microgpt.fut:484:51-57
            
            double zp_lhs_108584 = fpow64(ztzt_lhs_108583, 0.5);
            
            // futhark/microgpt.fut:484:59-71
            
            double zs_rhs_108585 = 1.0e-8 + zp_lhs_108584;
            
            // futhark/microgpt.fut:484:35-71
            
            double zm_rhs_108586 = zs_lhs_108582 / zs_rhs_108585;
            
            // futhark/microgpt.fut:484:13-71
            
            double lifted_lambda_res_108587 = zm_lhs_108580 - zm_rhs_108586;
            
            ((double *) mem_144013.mem)[i_142948 * m_104411 + i_142944] = lifted_lambda_res_108587;
        }
    }
    if (memblock_set(ctx, &mem_out_146162, &mem_144013, "mem_144013") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146163, &mem_143940, "mem_143940") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146164, &mem_143937, "mem_143937") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146550, &mem_out_146162, "mem_out_146162") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146551, &mem_out_146163, "mem_out_146163") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146552, &mem_out_146164, "mem_out_146164") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_143975);
        free(mem_143978);
        if (memblock_unref(ctx, &mem_144013, "mem_144013") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_143940, "mem_143940") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_143937, "mem_143937") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146164, "mem_out_146164") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146163, "mem_out_146163") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146162, "mem_out_146162") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_12711(struct futhark_context *ctx, struct memblock *mem_out_p_146555, struct memblock *mem_out_p_146556, struct memblock *mem_out_p_146557, struct memblock w_mem_143931, struct memblock mw_mem_143932, struct memblock vw_mem_143933, struct memblock dw_mem_143934, int64_t n_105443, int64_t m_105444, int64_t step_105449, double lt_r_105450)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_143975_cached_sizze_146558 = 0;
    unsigned char *mem_143975 = NULL;
    int64_t mem_143978_cached_sizze_146559 = 0;
    unsigned char *mem_143978 = NULL;
    struct memblock mem_144013;
    
    mem_144013.references = NULL;
    
    struct memblock mem_143940;
    
    mem_143940.references = NULL;
    
    struct memblock mem_143937;
    
    mem_143937.references = NULL;
    
    struct memblock mem_out_146164;
    
    mem_out_146164.references = NULL;
    
    struct memblock mem_out_146163;
    
    mem_out_146163.references = NULL;
    
    struct memblock mem_out_146162;
    
    mem_out_146162.references = NULL;
    
    struct memblock mem_143922 = ctx->constants->mem_143922;
    struct memblock mem_143923 = ctx->constants->mem_143923;
    struct memblock mem_143924 = ctx->constants->mem_143924;
    struct memblock mem_143925 = ctx->constants->mem_143925;
    struct memblock mem_143926 = ctx->constants->mem_143926;
    struct memblock mem_143927 = ctx->constants->mem_143927;
    struct memblock mem_143928 = ctx->constants->mem_143928;
    struct memblock mem_143929 = ctx->constants->mem_143929;
    struct memblock mem_143930 = ctx->constants->mem_143930;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_143935 = (int64_t) 8 * n_105443;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_143936 = m_105444 * binop_x_143935;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143937, bytes_143936, "mem_143937")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_143940, bytes_143936, "mem_143940")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142925 = 0; i_142925 < n_105443; i_142925++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142918 = 0; i_142918 < m_105444; i_142918++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_132033 = ((double *) mw_mem_143932.mem)[i_142925 * m_105444 + i_142918];
            
            // futhark/microgpt.fut:476:10-20
            
            double zp_lhs_132034 = 0.85 * zt_rhs_132033;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_132035 = ((double *) dw_mem_143934.mem)[i_142925 * m_105444 + i_142918];
            
            // futhark/microgpt.fut:476:35-45
            
            double zp_rhs_132036 = 0.15000000000000002 * zt_rhs_132035;
            
            // futhark/microgpt.fut:476:21-45
            
            double lifted_lambda_res_132037 = zp_lhs_132034 + zp_rhs_132036;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_132044 = ((double *) vw_mem_143933.mem)[i_142925 * m_105444 + i_142918];
            
            // futhark/microgpt.fut:478:10-20
            
            double zp_lhs_132045 = 0.99 * zt_rhs_132044;
            
            // futhark/microgpt.fut:478:35-45
            
            double zt_lhs_132047 = 1.0000000000000009e-2 * zt_rhs_132035;
            
            // futhark/microgpt.fut:478:46-56
            
            double zp_rhs_132048 = zt_rhs_132035 * zt_lhs_132047;
            
            // futhark/microgpt.fut:478:21-56
            
            double lifted_lambda_res_132049 = zp_lhs_132045 + zp_rhs_132048;
            
            ((double *) mem_143937.mem)[i_142925 * m_105444 + i_142918] = lifted_lambda_res_132049;
            ((double *) mem_143940.mem)[i_142925 * m_105444 + i_142918] = lifted_lambda_res_132037;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_109018 = sitofp_i64_f64(step_105449);
    
    // futhark/microgpt.fut:480:54-57
    
    double ztzt_rhs_109019 = 1.0 + i64_res_109018;
    
    // futhark/microgpt.fut:480:30-57
    
    double zm_rhs_109020 = fpow64(0.85, ztzt_rhs_109019);
    
    // futhark/microgpt.fut:480:23-57
    
    double zs_rhs_109021 = 1.0 - zm_rhs_109020;
    
    // futhark/microgpt.fut:482:31-58
    
    double zm_rhs_109059 = fpow64(0.99, ztzt_rhs_109019);
    
    // futhark/microgpt.fut:482:23-58
    
    double zs_rhs_109060 = 1.0 - zm_rhs_109059;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_143975_cached_sizze_146558 < bytes_143936) {
        err = lexical_realloc(ctx, &mem_143975, &mem_143975_cached_sizze_146558, bytes_143936);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143978_cached_sizze_146559 < bytes_143936) {
        err = lexical_realloc(ctx, &mem_143978, &mem_143978_cached_sizze_146559, bytes_143936);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142939 = 0; i_142939 < n_105443; i_142939++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142932 = 0; i_142932 < m_105444; i_142932++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_132069 = ((double *) mem_143940.mem)[i_142939 * m_105444 + i_142932];
            
            // futhark/microgpt.fut:480:18-57
            
            double lifted_lambda_res_132070 = zs_lhs_132069 / zs_rhs_109021;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_132077 = ((double *) mem_143937.mem)[i_142939 * m_105444 + i_142932];
            
            // futhark/microgpt.fut:482:18-58
            
            double lifted_lambda_res_132078 = zs_lhs_132077 / zs_rhs_109060;
            
            ((double *) mem_143975)[i_142939 * m_105444 + i_142932] = lifted_lambda_res_132078;
            ((double *) mem_143978)[i_142939 * m_105444 + i_142932] = lifted_lambda_res_132070;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_144013, bytes_143936, "mem_144013")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142948 = 0; i_142948 < n_105443; i_142948++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142944 = 0; i_142944 < m_105444; i_142944++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_108580 = ((double *) w_mem_143931.mem)[i_142948 * m_105444 + i_142944];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_108581 = ((double *) mem_143978)[i_142948 * m_105444 + i_142944];
            
            // futhark/microgpt.fut:484:21-34
            
            double zs_lhs_108582 = lt_r_105450 * zt_rhs_108581;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_108583 = ((double *) mem_143975)[i_142948 * m_105444 + i_142944];
            
            // futhark/microgpt.fut:484:51-57
            
            double zp_lhs_108584 = fpow64(ztzt_lhs_108583, 0.5);
            
            // futhark/microgpt.fut:484:59-71
            
            double zs_rhs_108585 = 1.0e-8 + zp_lhs_108584;
            
            // futhark/microgpt.fut:484:35-71
            
            double zm_rhs_108586 = zs_lhs_108582 / zs_rhs_108585;
            
            // futhark/microgpt.fut:484:13-71
            
            double lifted_lambda_res_108587 = zm_lhs_108580 - zm_rhs_108586;
            
            ((double *) mem_144013.mem)[i_142948 * m_105444 + i_142944] = lifted_lambda_res_108587;
        }
    }
    if (memblock_set(ctx, &mem_out_146162, &mem_144013, "mem_144013") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146163, &mem_143940, "mem_143940") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146164, &mem_143937, "mem_143937") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146555, &mem_out_146162, "mem_out_146162") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146556, &mem_out_146163, "mem_out_146163") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146557, &mem_out_146164, "mem_out_146164") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_143975);
        free(mem_143978);
        if (memblock_unref(ctx, &mem_144013, "mem_144013") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_143940, "mem_143940") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_143937, "mem_143937") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146164, "mem_out_146164") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146163, "mem_out_146163") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146162, "mem_out_146162") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_146560, double *out_prim_out_146561, struct memblock wdown_mem_143931, struct memblock wkey_mem_143932, struct memblock wout_mem_143933, struct memblock wpe_mem_143934, struct memblock wqry_mem_143935, struct memblock wte_mem_143936, struct memblock wup_mem_143937, struct memblock wval_mem_143938, struct memblock wvoc_mem_143939, struct memblock tokens_mem_143940, struct memblock target_mem_143941, struct memblock mask_mem_143942)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_143943_cached_sizze_146562 = 0;
    unsigned char *mem_143943 = NULL;
    int64_t mem_143948_cached_sizze_146563 = 0;
    unsigned char *mem_143948 = NULL;
    int64_t mem_143959_cached_sizze_146564 = 0;
    unsigned char *mem_143959 = NULL;
    int64_t mem_143964_cached_sizze_146565 = 0;
    unsigned char *mem_143964 = NULL;
    int64_t mem_143971_cached_sizze_146566 = 0;
    unsigned char *mem_143971 = NULL;
    int64_t mem_143982_cached_sizze_146567 = 0;
    unsigned char *mem_143982 = NULL;
    int64_t mem_143987_cached_sizze_146568 = 0;
    unsigned char *mem_143987 = NULL;
    int64_t mem_143994_cached_sizze_146569 = 0;
    unsigned char *mem_143994 = NULL;
    int64_t mem_144005_cached_sizze_146570 = 0;
    unsigned char *mem_144005 = NULL;
    int64_t mem_144006_cached_sizze_146571 = 0;
    unsigned char *mem_144006 = NULL;
    int64_t mem_144007_cached_sizze_146572 = 0;
    unsigned char *mem_144007 = NULL;
    int64_t mem_144020_cached_sizze_146573 = 0;
    unsigned char *mem_144020 = NULL;
    int64_t mem_144021_cached_sizze_146574 = 0;
    unsigned char *mem_144021 = NULL;
    int64_t mem_144022_cached_sizze_146575 = 0;
    unsigned char *mem_144022 = NULL;
    int64_t mem_144053_cached_sizze_146576 = 0;
    unsigned char *mem_144053 = NULL;
    int64_t mem_144054_cached_sizze_146577 = 0;
    unsigned char *mem_144054 = NULL;
    int64_t mem_144055_cached_sizze_146578 = 0;
    unsigned char *mem_144055 = NULL;
    int64_t mem_144071_cached_sizze_146579 = 0;
    unsigned char *mem_144071 = NULL;
    int64_t mem_144072_cached_sizze_146580 = 0;
    unsigned char *mem_144072 = NULL;
    int64_t mem_144073_cached_sizze_146581 = 0;
    unsigned char *mem_144073 = NULL;
    int64_t mem_144086_cached_sizze_146582 = 0;
    unsigned char *mem_144086 = NULL;
    int64_t mem_144087_cached_sizze_146583 = 0;
    unsigned char *mem_144087 = NULL;
    int64_t mem_144088_cached_sizze_146584 = 0;
    unsigned char *mem_144088 = NULL;
    int64_t mem_144134_cached_sizze_146585 = 0;
    unsigned char *mem_144134 = NULL;
    int64_t mem_144140_cached_sizze_146586 = 0;
    unsigned char *mem_144140 = NULL;
    int64_t mem_144145_cached_sizze_146587 = 0;
    unsigned char *mem_144145 = NULL;
    int64_t mem_144156_cached_sizze_146588 = 0;
    unsigned char *mem_144156 = NULL;
    int64_t mem_144161_cached_sizze_146589 = 0;
    unsigned char *mem_144161 = NULL;
    int64_t mem_144172_cached_sizze_146590 = 0;
    unsigned char *mem_144172 = NULL;
    int64_t mem_144177_cached_sizze_146591 = 0;
    unsigned char *mem_144177 = NULL;
    int64_t mem_144184_cached_sizze_146592 = 0;
    unsigned char *mem_144184 = NULL;
    int64_t mem_144191_cached_sizze_146593 = 0;
    unsigned char *mem_144191 = NULL;
    int64_t mem_144202_cached_sizze_146594 = 0;
    unsigned char *mem_144202 = NULL;
    int64_t mem_144207_cached_sizze_146595 = 0;
    unsigned char *mem_144207 = NULL;
    int64_t mem_144218_cached_sizze_146596 = 0;
    unsigned char *mem_144218 = NULL;
    int64_t mem_144223_cached_sizze_146597 = 0;
    unsigned char *mem_144223 = NULL;
    int64_t mem_144239_cached_sizze_146598 = 0;
    unsigned char *mem_144239 = NULL;
    int64_t mem_144244_cached_sizze_146599 = 0;
    unsigned char *mem_144244 = NULL;
    int64_t mem_144255_cached_sizze_146600 = 0;
    unsigned char *mem_144255 = NULL;
    int64_t mem_144260_cached_sizze_146601 = 0;
    unsigned char *mem_144260 = NULL;
    int64_t mem_144271_cached_sizze_146602 = 0;
    unsigned char *mem_144271 = NULL;
    int64_t mem_144276_cached_sizze_146603 = 0;
    unsigned char *mem_144276 = NULL;
    int64_t mem_144287_cached_sizze_146604 = 0;
    unsigned char *mem_144287 = NULL;
    int64_t mem_144292_cached_sizze_146605 = 0;
    unsigned char *mem_144292 = NULL;
    int64_t mem_144299_cached_sizze_146606 = 0;
    unsigned char *mem_144299 = NULL;
    int64_t mem_144310_cached_sizze_146607 = 0;
    unsigned char *mem_144310 = NULL;
    int64_t mem_144315_cached_sizze_146608 = 0;
    unsigned char *mem_144315 = NULL;
    int64_t mem_144326_cached_sizze_146609 = 0;
    unsigned char *mem_144326 = NULL;
    int64_t mem_144331_cached_sizze_146610 = 0;
    unsigned char *mem_144331 = NULL;
    int64_t mem_144342_cached_sizze_146611 = 0;
    unsigned char *mem_144342 = NULL;
    int64_t mem_144347_cached_sizze_146612 = 0;
    unsigned char *mem_144347 = NULL;
    int64_t mem_144358_cached_sizze_146613 = 0;
    unsigned char *mem_144358 = NULL;
    int64_t mem_144363_cached_sizze_146614 = 0;
    unsigned char *mem_144363 = NULL;
    int64_t mem_144374_cached_sizze_146615 = 0;
    unsigned char *mem_144374 = NULL;
    int64_t mem_144379_cached_sizze_146616 = 0;
    unsigned char *mem_144379 = NULL;
    int64_t mem_144394_cached_sizze_146617 = 0;
    unsigned char *mem_144394 = NULL;
    int64_t mem_144401_cached_sizze_146618 = 0;
    unsigned char *mem_144401 = NULL;
    struct memblock mem_144390;
    
    mem_144390.references = NULL;
    
    struct memblock mem_out_146162;
    
    mem_out_146162.references = NULL;
    
    struct memblock mem_143922 = ctx->constants->mem_143922;
    struct memblock mem_143923 = ctx->constants->mem_143923;
    struct memblock mem_143924 = ctx->constants->mem_143924;
    struct memblock mem_143925 = ctx->constants->mem_143925;
    struct memblock mem_143926 = ctx->constants->mem_143926;
    struct memblock mem_143927 = ctx->constants->mem_143927;
    struct memblock mem_143928 = ctx->constants->mem_143928;
    struct memblock mem_143929 = ctx->constants->mem_143929;
    struct memblock mem_143930 = ctx->constants->mem_143930;
    double prim_out_146163;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_143943_cached_sizze_146562 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143943, &mem_143943_cached_sizze_146562, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143948_cached_sizze_146563 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143948, &mem_143948_cached_sizze_146563, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142920 = 0; i_142920 < (int64_t) 16; i_142920++) {
        // futhark/microgpt.fut:466:41-50
        
        int64_t tmp_124409 = ((int64_t *) tokens_mem_143940.mem)[i_142920];
        
        // futhark/microgpt.fut:466:37-51
        
        bool x_124410 = sle64((int64_t) 0, tmp_124409);
        
        // futhark/microgpt.fut:466:37-51
        
        bool y_124411 = slt64(tmp_124409, (int64_t) 27);
        
        // futhark/microgpt.fut:466:37-51
        
        bool bounds_check_124412 = x_124410 && y_124411;
        
        // futhark/microgpt.fut:466:37-51
        
        bool index_certs_124413;
        
        if (!bounds_check_124412) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_124409, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:466:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:466:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142916 = 0; i_142916 < (int64_t) 16; i_142916++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_124420 = ((double *) wte_mem_143936.mem)[tmp_124409 * (int64_t) 16 + i_142916];
            
            ((double *) mem_143948)[i_142916] = lifted_lambda_res_124420;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143943, i_142920 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143948, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143959_cached_sizze_146564 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143959, &mem_143959_cached_sizze_146564, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143964_cached_sizze_146565 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143964, &mem_143964_cached_sizze_146565, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143971_cached_sizze_146566 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143971, &mem_143971_cached_sizze_146566, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142932 = 0; i_142932 < (int64_t) 16; i_142932++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_124446;
        double r_124448 = 0.0;
        
        for (int64_t i_124447 = 0; i_124447 < (int64_t) 16; i_124447++) {
            // futhark/microgpt.fut:71:46-49
            
            double zp_lhs_124449 = ((double *) wpe_mem_143934.mem)[i_142932 * (int64_t) 16 + i_124447];
            
            // futhark/microgpt.fut:71:46-49
            
            double zp_rhs_124450 = ((double *) mem_143943)[i_142932 * (int64_t) 16 + i_124447];
            
            // futhark/microgpt.fut:203:76-116
            
            double zp_res_124451 = zp_lhs_124449 + zp_rhs_124450;
            
            // futhark/microgpt.fut:203:94-163
            
            double zt_res_124452 = zp_res_124451 * zp_res_124451;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_124453 = r_124448 + zt_res_124452;
            double r_tmp_146167 = zp_res_124453;
            
            r_124448 = r_tmp_146167;
        }
        defunc_0_lifted_lambda_res_124446 = r_124448;
        // futhark/microgpt.fut:203:54-182
        
        double zs_res_124454 = defunc_0_lifted_lambda_res_124446 / 16.0;
        
        // futhark/microgpt.fut:204:24-55
        
        double zp_res_124455 = 1.0e-5 + zs_res_124454;
        
        // futhark/microgpt.fut:204:16-55
        
        double sqrt_res_124456 = futrts_sqrt64(zp_res_124455);
        
        // futhark/microgpt.fut:205:85-96
        
        double zs_res_124457 = 1.0 / sqrt_res_124456;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142924 = 0; i_142924 < (int64_t) 16; i_142924++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_124464 = ((double *) wpe_mem_143934.mem)[i_142932 * (int64_t) 16 + i_142924];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_124465 = ((double *) mem_143943)[i_142932 * (int64_t) 16 + i_142924];
            
            // futhark/microgpt.fut:205:38-78
            
            double zp_res_124466 = zp_lhs_124464 + zp_rhs_124465;
            
            // futhark/microgpt.fut:205:56-96
            
            double zt_res_124467 = zs_res_124457 * zp_res_124466;
            
            ((double *) mem_143964)[i_142924] = zt_res_124467;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142928 = 0; i_142928 < (int64_t) 16; i_142928++) {
            // futhark/microgpt.fut:206:4-14
            
            double lifted_lambda_res_124475 = ((double *) mem_143964)[i_142928];
            
            ((double *) mem_143971)[i_142928] = lifted_lambda_res_124475;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143959, i_142932 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143971, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143982_cached_sizze_146567 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143982, &mem_143982_cached_sizze_146567, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143987_cached_sizze_146568 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143987, &mem_143987_cached_sizze_146568, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143994_cached_sizze_146569 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143994, &mem_143994_cached_sizze_146569, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142944 = 0; i_142944 < (int64_t) 16; i_142944++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_124484;
        double r_124486 = 0.0;
        
        for (int64_t i_124485 = 0; i_124485 < (int64_t) 16; i_124485++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_124487 = ((double *) mem_143959)[i_142944 * (int64_t) 16 + i_124485];
            
            // futhark/microgpt.fut:207:78-115
            
            double zt_res_124488 = zt_lhs_124487 * zt_lhs_124487;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_124489 = r_124486 + zt_res_124488;
            double r_tmp_146171 = zp_res_124489;
            
            r_124486 = r_tmp_146171;
        }
        defunc_0_lifted_lambda_res_124484 = r_124486;
        // futhark/microgpt.fut:207:57-133
        
        double zs_res_124490 = defunc_0_lifted_lambda_res_124484 / 16.0;
        
        // futhark/microgpt.fut:208:24-55
        
        double zp_res_124491 = 1.0e-5 + zs_res_124490;
        
        // futhark/microgpt.fut:208:16-55
        
        double sqrt_res_124492 = futrts_sqrt64(zp_res_124491);
        
        // futhark/microgpt.fut:209:59-70
        
        double zs_res_124493 = 1.0 / sqrt_res_124492;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142936 = 0; i_142936 < (int64_t) 16; i_142936++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_124500 = ((double *) mem_143959)[i_142944 * (int64_t) 16 + i_142936];
            
            // futhark/microgpt.fut:209:37-70
            
            double zt_res_124501 = zs_res_124493 * zt_lhs_124500;
            
            ((double *) mem_143987)[i_142936] = zt_res_124501;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142940 = 0; i_142940 < (int64_t) 16; i_142940++) {
            // futhark/microgpt.fut:210:4-14
            
            double lifted_lambda_res_124509 = ((double *) mem_143987)[i_142940];
            
            ((double *) mem_143994)[i_142940] = lifted_lambda_res_124509;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143982, i_142944 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143994, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144005_cached_sizze_146570 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144005, &mem_144005_cached_sizze_146570, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144006_cached_sizze_146571 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144006, &mem_144006_cached_sizze_146571, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144007_cached_sizze_146572 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144007, &mem_144007_cached_sizze_146572, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144020_cached_sizze_146573 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144020, &mem_144020_cached_sizze_146573, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144021_cached_sizze_146574 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144021, &mem_144021_cached_sizze_146574, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144022_cached_sizze_146575 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144022, &mem_144022_cached_sizze_146575, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142962 = 0; i_142962 < (int64_t) 16; i_142962++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142952 = 0; i_142952 < (int64_t) 16; i_142952++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_132251;
            double r_132253 = 0.0;
            
            for (int64_t i_132252 = 0; i_132252 < (int64_t) 16; i_132252++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_132254 = ((double *) wqry_mem_143935.mem)[i_142952 * (int64_t) 16 + i_132252];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_132255 = ((double *) mem_143982)[i_142962 * (int64_t) 16 + i_132252];
                
                // futhark/microgpt.fut:211:66-105
                
                double zt_res_132256 = zt_lhs_132254 * zt_rhs_132255;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_132257 = r_132253 + zt_res_132256;
                double r_tmp_146180 = zp_res_132257;
                
                r_132253 = r_tmp_146180;
            }
            defunc_0_lifted_lambda_res_132251 = r_132253;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_132264;
            double r_132266 = 0.0;
            
            for (int64_t i_132265 = 0; i_132265 < (int64_t) 16; i_132265++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_132267 = ((double *) wkey_mem_143932.mem)[i_142952 * (int64_t) 16 + i_132265];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_132268 = ((double *) mem_143982)[i_142962 * (int64_t) 16 + i_132265];
                
                // futhark/microgpt.fut:212:66-105
                
                double zt_res_132269 = zt_lhs_132267 * zt_rhs_132268;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_132270 = r_132266 + zt_res_132269;
                double r_tmp_146181 = zp_res_132270;
                
                r_132266 = r_tmp_146181;
            }
            defunc_0_lifted_lambda_res_132264 = r_132266;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_132280;
            double r_132282 = 0.0;
            
            for (int64_t i_132281 = 0; i_132281 < (int64_t) 16; i_132281++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_132283 = ((double *) wval_mem_143938.mem)[i_142952 * (int64_t) 16 + i_132281];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_132284 = ((double *) mem_143982)[i_142962 * (int64_t) 16 + i_132281];
                
                // futhark/microgpt.fut:213:66-105
                
                double zt_res_132285 = zt_lhs_132283 * zt_rhs_132284;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_132286 = r_132282 + zt_res_132285;
                double r_tmp_146182 = zp_res_132286;
                
                r_132282 = r_tmp_146182;
            }
            defunc_0_lifted_lambda_res_132280 = r_132282;
            ((double *) mem_144020)[i_142952] = defunc_0_lifted_lambda_res_132280;
            ((double *) mem_144021)[i_142952] = defunc_0_lifted_lambda_res_132264;
            ((double *) mem_144022)[i_142952] = defunc_0_lifted_lambda_res_132251;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144005, i_142962 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144020, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144006, i_142962 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144021, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144007, i_142962 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144022, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144053_cached_sizze_146576 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144053, &mem_144053_cached_sizze_146576, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144054_cached_sizze_146577 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144054, &mem_144054_cached_sizze_146577, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144055_cached_sizze_146578 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144055, &mem_144055_cached_sizze_146578, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144071_cached_sizze_146579 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144071, &mem_144071_cached_sizze_146579, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144072_cached_sizze_146580 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144072, &mem_144072_cached_sizze_146580, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144073_cached_sizze_146581 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144073, &mem_144073_cached_sizze_146581, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144086_cached_sizze_146582 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144086, &mem_144086_cached_sizze_146582, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144087_cached_sizze_146583 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144087, &mem_144087_cached_sizze_146583, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144088_cached_sizze_146584 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144088, &mem_144088_cached_sizze_146584, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142992 = 0; i_142992 < (int64_t) 4; i_142992++) {
        // futhark/microgpt.fut:214:69-72
        
        int64_t zp_lhs_132127 = mul64((int64_t) 4, i_142992);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142982 = 0; i_142982 < (int64_t) 16; i_142982++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142972 = 0; i_142972 < (int64_t) 4; i_142972++) {
                // futhark/microgpt.fut:214:74-81
                
                int64_t tmp_132444 = add64(zp_lhs_132127, i_142972);
                
                // futhark/microgpt.fut:214:51-83
                
                bool x_132445 = sle64((int64_t) 0, tmp_132444);
                
                // futhark/microgpt.fut:214:51-83
                
                bool y_132446 = slt64(tmp_132444, (int64_t) 16);
                
                // futhark/microgpt.fut:214:51-83
                
                bool bounds_check_132447 = x_132445 && y_132446;
                
                // futhark/microgpt.fut:214:51-83
                
                bool index_certs_132448;
                
                if (!bounds_check_132447) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_132444, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:214:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:214:15-84\n   #9  futhark/microgpt.fut:467:7-76\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_132449 = ((double *) mem_144007)[i_142982 * (int64_t) 16 + tmp_132444];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_132457 = ((double *) mem_144006)[i_142982 * (int64_t) 16 + tmp_132444];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_132468 = ((double *) mem_144005)[i_142982 * (int64_t) 16 + tmp_132444];
                
                ((double *) mem_144086)[i_142972] = lifted_lambda_res_132468;
                ((double *) mem_144087)[i_142972] = lifted_lambda_res_132457;
                ((double *) mem_144088)[i_142972] = lifted_lambda_res_132449;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144071, i_142982 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144086, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144072, i_142982 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144087, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144073, i_142982 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144088, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144053, i_142992 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144071, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144054, i_142992 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144072, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144055, i_142992 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144073, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144134_cached_sizze_146585 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144134, &mem_144134_cached_sizze_146585, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144140_cached_sizze_146586 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144140, &mem_144140_cached_sizze_146586, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144145_cached_sizze_146587 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144145, &mem_144145_cached_sizze_146587, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144156_cached_sizze_146588 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144156, &mem_144156_cached_sizze_146588, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144161_cached_sizze_146589 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144161, &mem_144161_cached_sizze_146589, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144172_cached_sizze_146590 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144172, &mem_144172_cached_sizze_146590, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144177_cached_sizze_146591 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144177, &mem_144177_cached_sizze_146591, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144184_cached_sizze_146592 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144184, &mem_144184_cached_sizze_146592, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144191_cached_sizze_146593 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144191, &mem_144191_cached_sizze_146593, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144202_cached_sizze_146594 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144202, &mem_144202_cached_sizze_146594, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144207_cached_sizze_146595 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144207, &mem_144207_cached_sizze_146595, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144218_cached_sizze_146596 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144218, &mem_144218_cached_sizze_146596, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144223_cached_sizze_146597 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144223, &mem_144223_cached_sizze_146597, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143048 = 0; i_143048 < (int64_t) 4; i_143048++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143002 = 0; i_143002 < (int64_t) 16; i_143002++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142998 = 0; i_142998 < (int64_t) 16; i_142998++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_124654;
                double r_124656 = 0.0;
                
                for (int64_t i_124655 = 0; i_124655 < (int64_t) 4; i_124655++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_124657 = ((double *) mem_144055)[i_143048 * (int64_t) 64 + i_143002 * (int64_t) 4 + i_124655];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_124658 = ((double *) mem_144054)[i_143048 * (int64_t) 64 + i_142998 * (int64_t) 4 + i_124655];
                    
                    // futhark/microgpt.fut:217:113-164
                    
                    double zt_res_124659 = zt_lhs_124657 * zt_rhs_124658;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_124660 = r_124656 + zt_res_124659;
                    double r_tmp_146195 = zp_res_124660;
                    
                    r_124656 = r_tmp_146195;
                }
                defunc_0_lifted_lambda_res_124654 = r_124656;
                ((double *) mem_144145)[i_142998] = defunc_0_lifted_lambda_res_124654;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144140, i_143002 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144145, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143010 = 0; i_143010 < (int64_t) 16; i_143010++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143006 = 0; i_143006 < (int64_t) 16; i_143006++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_124675 = ((double *) mem_144140)[i_143010 * (int64_t) 16 + i_143006];
                
                // futhark/microgpt.fut:218:47-78
                
                double zs_res_124676 = zs_lhs_124675 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_124677 = ((double *) mask_mem_143942.mem)[i_143010 * (int64_t) 16 + i_143006];
                
                // futhark/microgpt.fut:218:65-102
                
                double zp_res_124678 = zs_res_124676 + zp_rhs_124677;
                
                ((double *) mem_144161)[i_143006] = zp_res_124678;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144156, i_143010 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144161, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143028 = 0; i_143028 < (int64_t) 16; i_143028++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_132571;
            double redout_143012 = -INFINITY;
            
            for (int64_t i_143013 = 0; i_143013 < (int64_t) 16; i_143013++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_132495 = ((double *) mem_144156)[i_143028 * (int64_t) 16 + i_143013];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_124699 = fmax64(lifted_lambda_res_132495, redout_143012);
                double redout_tmp_146199 = max_res_124699;
                
                redout_143012 = redout_tmp_146199;
            }
            defunc_0_reduce_res_132571 = redout_143012;
            // futhark/microgpt.fut:220:67-76
            
            double neg_res_124700 = -defunc_0_reduce_res_132571;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143016 = 0; i_143016 < (int64_t) 16; i_143016++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_124707 = ((double *) mem_144156)[i_143028 * (int64_t) 16 + i_143016];
                
                // futhark/microgpt.fut:220:44-76
                
                double zp_res_124708 = neg_res_124700 + zp_lhs_124707;
                
                // futhark/microgpt.fut:220:37-76
                
                double exp_res_124709 = futrts_exp64(zp_res_124708);
                
                ((double *) mem_144177)[i_143016] = exp_res_124709;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124711;
            double r_124713 = 0.0;
            
            for (int64_t i_124712 = 0; i_124712 < (int64_t) 16; i_124712++) {
                // futhark/microgpt.fut:221:36-46
                
                double lifted_lambda_res_124714 = ((double *) mem_144177)[i_124712];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_124715 = r_124713 + lifted_lambda_res_124714;
                double r_tmp_146201 = zp_res_124715;
                
                r_124713 = r_tmp_146201;
            }
            defunc_0_lifted_lambda_res_124711 = r_124713;
            // futhark/microgpt.fut:222:53-64
            
            double zs_res_124716 = 1.0 / defunc_0_lifted_lambda_res_124711;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143020 = 0; i_143020 < (int64_t) 16; i_143020++) {
                // futhark/microgpt.fut:222:37-47
                
                double zt_lhs_124723 = ((double *) mem_144177)[i_143020];
                
                // futhark/microgpt.fut:222:37-64
                
                double zt_res_124724 = zs_res_124716 * zt_lhs_124723;
                
                ((double *) mem_144184)[i_143020] = zt_res_124724;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143024 = 0; i_143024 < (int64_t) 16; i_143024++) {
                // futhark/microgpt.fut:223:4-14
                
                double lifted_lambda_res_124732 = ((double *) mem_144184)[i_143024];
                
                ((double *) mem_144191)[i_143024] = lifted_lambda_res_124732;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144172, i_143028 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144191, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143036 = 0; i_143036 < (int64_t) 16; i_143036++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143032 = 0; i_143032 < (int64_t) 4; i_143032++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_124747;
                double r_124749 = 0.0;
                
                for (int64_t i_124748 = 0; i_124748 < (int64_t) 16; i_124748++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_124750 = ((double *) mem_144172)[i_143036 * (int64_t) 16 + i_124748];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_124751 = ((double *) mem_144053)[i_143048 * (int64_t) 64 + i_124748 * (int64_t) 4 + i_143032];
                    
                    // futhark/microgpt.fut:224:66-111
                    
                    double zt_res_124752 = zt_lhs_124750 * zt_rhs_124751;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_124753 = r_124749 + zt_res_124752;
                    double r_tmp_146206 = zp_res_124753;
                    
                    r_124749 = r_tmp_146206;
                }
                defunc_0_lifted_lambda_res_124747 = r_124749;
                ((double *) mem_144207)[i_143032] = defunc_0_lifted_lambda_res_124747;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144202, i_143036 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144207, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143044 = 0; i_143044 < (int64_t) 16; i_143044++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143040 = 0; i_143040 < (int64_t) 4; i_143040++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_124768 = ((double *) mem_144202)[i_143044 * (int64_t) 4 + i_143040];
                
                ((double *) mem_144223)[i_143040] = lifted_lambda_res_124768;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144218, i_143044 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144223, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144134, i_143048 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144218, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144239_cached_sizze_146598 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144239, &mem_144239_cached_sizze_146598, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144244_cached_sizze_146599 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144244, &mem_144244_cached_sizze_146599, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143056 = 0; i_143056 < (int64_t) 16; i_143056++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143052 = 0; i_143052 < (int64_t) 16; i_143052++) {
            // futhark/microgpt.fut:226:54-57
            
            int64_t tmp_124780 = sdiv64(i_143052, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-59
            
            bool x_124781 = sle64((int64_t) 0, tmp_124780);
            
            // futhark/microgpt.fut:226:44-59
            
            bool y_124782 = slt64(tmp_124780, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-59
            
            bool bounds_check_124783 = x_124781 && y_124782;
            
            // futhark/microgpt.fut:226:44-59
            
            bool index_certs_124784;
            
            if (!bounds_check_124783) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_124780, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:226:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:226:15-80\n   #6  futhark/microgpt.fut:467:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:226:74-77
            
            int64_t tmp_124785 = smod64(i_143052, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-79
            
            bool x_124786 = sle64((int64_t) 0, tmp_124785);
            
            // futhark/microgpt.fut:226:44-79
            
            bool y_124787 = slt64(tmp_124785, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-79
            
            bool bounds_check_124788 = x_124786 && y_124787;
            
            // futhark/microgpt.fut:226:44-79
            
            bool index_certs_124789;
            
            if (!bounds_check_124788) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_124785, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:226:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:226:15-80\n   #6  futhark/microgpt.fut:467:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_124790 = ((double *) mem_144134)[tmp_124780 * (int64_t) 64 + i_143056 * (int64_t) 4 + tmp_124785];
            
            ((double *) mem_144244)[i_143052] = lifted_lambda_res_124790;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144239, i_143056 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144244, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144255_cached_sizze_146600 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144255, &mem_144255_cached_sizze_146600, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144260_cached_sizze_146601 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144260, &mem_144260_cached_sizze_146601, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143064 = 0; i_143064 < (int64_t) 16; i_143064++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143060 = 0; i_143060 < (int64_t) 16; i_143060++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124805;
            double r_124807 = 0.0;
            
            for (int64_t i_124806 = 0; i_124806 < (int64_t) 16; i_124806++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_124808 = ((double *) wout_mem_143933.mem)[i_143060 * (int64_t) 16 + i_124806];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_124809 = ((double *) mem_144239)[i_143064 * (int64_t) 16 + i_124806];
                
                // futhark/microgpt.fut:227:67-106
                
                double zt_res_124810 = zt_lhs_124808 * zt_rhs_124809;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_124811 = r_124807 + zt_res_124810;
                double r_tmp_146213 = zp_res_124811;
                
                r_124807 = r_tmp_146213;
            }
            defunc_0_lifted_lambda_res_124805 = r_124807;
            ((double *) mem_144260)[i_143060] = defunc_0_lifted_lambda_res_124805;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144255, i_143064 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144260, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144271_cached_sizze_146602 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144271, &mem_144271_cached_sizze_146602, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144276_cached_sizze_146603 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144276, &mem_144276_cached_sizze_146603, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143072 = 0; i_143072 < (int64_t) 16; i_143072++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143068 = 0; i_143068 < (int64_t) 16; i_143068++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_124826 = ((double *) mem_144255)[i_143072 * (int64_t) 16 + i_143068];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_124827 = ((double *) mem_143959)[i_143072 * (int64_t) 16 + i_143068];
            
            // futhark/microgpt.fut:228:46-84
            
            double zp_res_124828 = zp_lhs_124826 + zp_rhs_124827;
            
            ((double *) mem_144276)[i_143068] = zp_res_124828;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144271, i_143072 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144276, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144287_cached_sizze_146604 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144287, &mem_144287_cached_sizze_146604, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144292_cached_sizze_146605 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144292, &mem_144292_cached_sizze_146605, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144299_cached_sizze_146606 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144299, &mem_144299_cached_sizze_146606, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143084 = 0; i_143084 < (int64_t) 16; i_143084++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_124837;
        double r_124839 = 0.0;
        
        for (int64_t i_124838 = 0; i_124838 < (int64_t) 16; i_124838++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_124840 = ((double *) mem_144271)[i_143084 * (int64_t) 16 + i_124838];
            
            // futhark/microgpt.fut:229:79-118
            
            double zt_res_124841 = zt_lhs_124840 * zt_lhs_124840;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_124842 = r_124839 + zt_res_124841;
            double r_tmp_146217 = zp_res_124842;
            
            r_124839 = r_tmp_146217;
        }
        defunc_0_lifted_lambda_res_124837 = r_124839;
        // futhark/microgpt.fut:229:58-136
        
        double zs_res_124843 = defunc_0_lifted_lambda_res_124837 / 16.0;
        
        // futhark/microgpt.fut:230:24-55
        
        double zp_res_124844 = 1.0e-5 + zs_res_124843;
        
        // futhark/microgpt.fut:230:16-55
        
        double sqrt_res_124845 = futrts_sqrt64(zp_res_124844);
        
        // futhark/microgpt.fut:231:60-71
        
        double zs_res_124846 = 1.0 / sqrt_res_124845;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143076 = 0; i_143076 < (int64_t) 16; i_143076++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_124853 = ((double *) mem_144271)[i_143084 * (int64_t) 16 + i_143076];
            
            // futhark/microgpt.fut:231:37-71
            
            double zt_res_124854 = zs_res_124846 * zt_lhs_124853;
            
            ((double *) mem_144292)[i_143076] = zt_res_124854;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143080 = 0; i_143080 < (int64_t) 16; i_143080++) {
            // futhark/microgpt.fut:232:4-14
            
            double lifted_lambda_res_124862 = ((double *) mem_144292)[i_143080];
            
            ((double *) mem_144299)[i_143080] = lifted_lambda_res_124862;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144287, i_143084 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144299, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144310_cached_sizze_146607 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144310, &mem_144310_cached_sizze_146607, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144315_cached_sizze_146608 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144315, &mem_144315_cached_sizze_146608, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143092 = 0; i_143092 < (int64_t) 16; i_143092++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143088 = 0; i_143088 < (int64_t) 64; i_143088++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124878;
            double r_124880 = 0.0;
            
            for (int64_t i_124879 = 0; i_124879 < (int64_t) 16; i_124879++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_124881 = ((double *) wup_mem_143937.mem)[i_143088 * (int64_t) 16 + i_124879];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_124882 = ((double *) mem_144287)[i_143092 * (int64_t) 16 + i_124879];
                
                // futhark/microgpt.fut:233:67-106
                
                double zt_res_124883 = zt_lhs_124881 * zt_rhs_124882;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_124884 = r_124880 + zt_res_124883;
                double r_tmp_146222 = zp_res_124884;
                
                r_124880 = r_tmp_146222;
            }
            defunc_0_lifted_lambda_res_124878 = r_124880;
            ((double *) mem_144315)[i_143088] = defunc_0_lifted_lambda_res_124878;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144310, i_143092 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144315, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144326_cached_sizze_146609 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144326, &mem_144326_cached_sizze_146609, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144331_cached_sizze_146610 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144331, &mem_144331_cached_sizze_146610, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143100 = 0; i_143100 < (int64_t) 16; i_143100++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143096 = 0; i_143096 < (int64_t) 64; i_143096++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_124899 = ((double *) mem_144310)[i_143100 * (int64_t) 64 + i_143096];
            
            // futhark/microgpt.fut:234:45-73
            
            double max_res_124900 = fmax64(0.0, max_arg0_124899);
            
            ((double *) mem_144331)[i_143096] = max_res_124900;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144326, i_143100 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144331, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144342_cached_sizze_146611 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144342, &mem_144342_cached_sizze_146611, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144347_cached_sizze_146612 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144347, &mem_144347_cached_sizze_146612, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143108 = 0; i_143108 < (int64_t) 16; i_143108++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143104 = 0; i_143104 < (int64_t) 16; i_143104++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124915;
            double r_124917 = 0.0;
            
            for (int64_t i_124916 = 0; i_124916 < (int64_t) 64; i_124916++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_124918 = ((double *) wdown_mem_143931.mem)[i_143104 * (int64_t) 64 + i_124916];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_124919 = ((double *) mem_144326)[i_143108 * (int64_t) 64 + i_124916];
                
                // futhark/microgpt.fut:235:67-108
                
                double zt_res_124920 = zt_lhs_124918 * zt_rhs_124919;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_124921 = r_124917 + zt_res_124920;
                double r_tmp_146227 = zp_res_124921;
                
                r_124917 = r_tmp_146227;
            }
            defunc_0_lifted_lambda_res_124915 = r_124917;
            ((double *) mem_144347)[i_143104] = defunc_0_lifted_lambda_res_124915;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144342, i_143108 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144347, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144358_cached_sizze_146613 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144358, &mem_144358_cached_sizze_146613, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144363_cached_sizze_146614 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144363, &mem_144363_cached_sizze_146614, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143116 = 0; i_143116 < (int64_t) 16; i_143116++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143112 = 0; i_143112 < (int64_t) 16; i_143112++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_124936 = ((double *) mem_144342)[i_143116 * (int64_t) 16 + i_143112];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_124937 = ((double *) mem_144271)[i_143116 * (int64_t) 16 + i_143112];
            
            // futhark/microgpt.fut:236:46-85
            
            double zp_res_124938 = zp_lhs_124936 + zp_rhs_124937;
            
            ((double *) mem_144363)[i_143112] = zp_res_124938;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144358, i_143116 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144363, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144374_cached_sizze_146615 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144374, &mem_144374_cached_sizze_146615, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144379_cached_sizze_146616 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144379, &mem_144379_cached_sizze_146616, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143124 = 0; i_143124 < (int64_t) 16; i_143124++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143120 = 0; i_143120 < (int64_t) 27; i_143120++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124954;
            double r_124956 = 0.0;
            
            for (int64_t i_124955 = 0; i_124955 < (int64_t) 16; i_124955++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_124957 = ((double *) wvoc_mem_143939.mem)[i_143120 * (int64_t) 16 + i_124955];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_124958 = ((double *) mem_144358)[i_143124 * (int64_t) 16 + i_124955];
                
                // futhark/microgpt.fut:237:67-107
                
                double zt_res_124959 = zt_lhs_124957 * zt_rhs_124958;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_124960 = r_124956 + zt_res_124959;
                double r_tmp_146232 = zp_res_124960;
                
                r_124956 = r_tmp_146232;
            }
            defunc_0_lifted_lambda_res_124954 = r_124956;
            ((double *) mem_144379)[i_143120] = defunc_0_lifted_lambda_res_124954;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144374, i_143124 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144379, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_144390, (int64_t) 128, "mem_144390")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144394_cached_sizze_146617 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144394, &mem_144394_cached_sizze_146617, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144401_cached_sizze_146618 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144401, &mem_144401_cached_sizze_146618, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143138 = 0; i_143138 < (int64_t) 16; i_143138++) {
        double x_132594;
        double redout_143126 = -INFINITY;
        
        for (int64_t i_143127 = 0; i_143127 < (int64_t) 27; i_143127++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_132541 = ((double *) mem_144374)[i_143138 * (int64_t) 27 + i_143127];
            
            // futhark/microgpt.fut:115:13-33
            
            double max_res_124984 = fmax64(lifted_lambda_res_132541, redout_143126);
            double redout_tmp_146234 = max_res_124984;
            
            redout_143126 = redout_tmp_146234;
        }
        x_132594 = redout_143126;
        // futhark/microgpt.fut:239:67-76
        
        double neg_res_124985 = -x_132594;
        
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_124969;
        double r_124971 = 0.0;
        
        for (int64_t i_124970 = 0; i_124970 < (int64_t) 27; i_124970++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143130 = 0; i_143130 < (int64_t) 27; i_143130++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_124992 = ((double *) mem_144374)[i_143138 * (int64_t) 27 + i_143130];
                
                // futhark/microgpt.fut:239:44-76
                
                double zp_res_124993 = neg_res_124985 + zp_lhs_124992;
                
                // futhark/microgpt.fut:239:37-76
                
                double exp_res_124994 = futrts_exp64(zp_res_124993);
                
                ((double *) mem_144394)[i_143130] = exp_res_124994;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124996;
            double r_124998 = 0.0;
            
            for (int64_t i_124997 = 0; i_124997 < (int64_t) 27; i_124997++) {
                // futhark/microgpt.fut:240:36-46
                
                double lifted_lambda_res_124999 = ((double *) mem_144394)[i_124997];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_125000 = r_124998 + lifted_lambda_res_124999;
                double r_tmp_146237 = zp_res_125000;
                
                r_124998 = r_tmp_146237;
            }
            defunc_0_lifted_lambda_res_124996 = r_124998;
            // futhark/microgpt.fut:241:53-64
            
            double zs_res_125001 = 1.0 / defunc_0_lifted_lambda_res_124996;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143134 = 0; i_143134 < (int64_t) 27; i_143134++) {
                // futhark/microgpt.fut:241:37-47
                
                double zt_lhs_125008 = ((double *) mem_144394)[i_143134];
                
                // futhark/microgpt.fut:241:37-64
                
                double zt_res_125009 = zs_res_125001 * zt_lhs_125008;
                
                ((double *) mem_144401)[i_143134] = zt_res_125009;
            }
            // futhark/microgpt.fut:242:12-22
            
            double log_arg0_125011 = ((double *) mem_144401)[i_124970];
            
            // futhark/microgpt.fut:242:6-22
            
            double log_res_125012 = futrts_log64(log_arg0_125011);
            
            // futhark/microgpt.fut:71:46-49
            
            double zt_rhs_125013 = ((double *) target_mem_143941.mem)[i_143138 * (int64_t) 27 + i_124970];
            
            // futhark/microgpt.fut:242:6-48
            
            double zt_res_125014 = log_res_125012 * zt_rhs_125013;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_125015 = r_124971 + zt_res_125014;
            double r_tmp_146235 = zp_res_125015;
            
            r_124971 = r_tmp_146235;
        }
        defunc_0_lifted_lambda_res_124969 = r_124971;
        // futhark/microgpt.fut:238:37-242:54
        
        double neg_res_125016 = -defunc_0_lifted_lambda_res_124969;
        
        ((double *) mem_144390.mem)[i_143138] = neg_res_125016;
    }
    // futhark/microgpt.fut:71:13-49
    
    double defunc_0_lifted_lambda_res_125018;
    double r_125020 = 0.0;
    
    for (int64_t i_125019 = 0; i_125019 < (int64_t) 16; i_125019++) {
        // futhark/microgpt.fut:243:37-47
        
        double lifted_lambda_res_125021 = ((double *) mem_144390.mem)[i_125019];
        
        // futhark/microgpt.fut:71:40-49
        
        double zp_res_125022 = r_125020 + lifted_lambda_res_125021;
        double r_tmp_146239 = zp_res_125022;
        
        r_125020 = r_tmp_146239;
    }
    defunc_0_lifted_lambda_res_125018 = r_125020;
    // futhark/microgpt.fut:243:17-64
    
    double zs_res_125023 = defunc_0_lifted_lambda_res_125018 / 16.0;
    
    if (memblock_set(ctx, &mem_out_146162, &mem_144390, "mem_144390") != 0)
        return 1;
    prim_out_146163 = zs_res_125023;
    if (memblock_set(ctx, &*mem_out_p_146560, &mem_out_146162, "mem_out_146162") != 0)
        return 1;
    *out_prim_out_146561 = prim_out_146163;
    
  cleanup:
    {
        free(mem_143943);
        free(mem_143948);
        free(mem_143959);
        free(mem_143964);
        free(mem_143971);
        free(mem_143982);
        free(mem_143987);
        free(mem_143994);
        free(mem_144005);
        free(mem_144006);
        free(mem_144007);
        free(mem_144020);
        free(mem_144021);
        free(mem_144022);
        free(mem_144053);
        free(mem_144054);
        free(mem_144055);
        free(mem_144071);
        free(mem_144072);
        free(mem_144073);
        free(mem_144086);
        free(mem_144087);
        free(mem_144088);
        free(mem_144134);
        free(mem_144140);
        free(mem_144145);
        free(mem_144156);
        free(mem_144161);
        free(mem_144172);
        free(mem_144177);
        free(mem_144184);
        free(mem_144191);
        free(mem_144202);
        free(mem_144207);
        free(mem_144218);
        free(mem_144223);
        free(mem_144239);
        free(mem_144244);
        free(mem_144255);
        free(mem_144260);
        free(mem_144271);
        free(mem_144276);
        free(mem_144287);
        free(mem_144292);
        free(mem_144299);
        free(mem_144310);
        free(mem_144315);
        free(mem_144326);
        free(mem_144331);
        free(mem_144342);
        free(mem_144347);
        free(mem_144358);
        free(mem_144363);
        free(mem_144374);
        free(mem_144379);
        free(mem_144394);
        free(mem_144401);
        if (memblock_unref(ctx, &mem_144390, "mem_144390") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146162, "mem_out_146162") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_146619, struct memblock wdown_mem_143931, struct memblock wkey_mem_143932, struct memblock wout_mem_143933, struct memblock wpe_mem_143934, struct memblock wqry_mem_143935, struct memblock wte_mem_143936, struct memblock wup_mem_143937, struct memblock wval_mem_143938, struct memblock wvoc_mem_143939, struct memblock tokens_mem_143940, struct memblock mask_mem_143941)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_143942_cached_sizze_146620 = 0;
    unsigned char *mem_143942 = NULL;
    int64_t mem_143947_cached_sizze_146621 = 0;
    unsigned char *mem_143947 = NULL;
    int64_t mem_143958_cached_sizze_146622 = 0;
    unsigned char *mem_143958 = NULL;
    int64_t mem_143963_cached_sizze_146623 = 0;
    unsigned char *mem_143963 = NULL;
    int64_t mem_143970_cached_sizze_146624 = 0;
    unsigned char *mem_143970 = NULL;
    int64_t mem_143981_cached_sizze_146625 = 0;
    unsigned char *mem_143981 = NULL;
    int64_t mem_143986_cached_sizze_146626 = 0;
    unsigned char *mem_143986 = NULL;
    int64_t mem_143993_cached_sizze_146627 = 0;
    unsigned char *mem_143993 = NULL;
    int64_t mem_144004_cached_sizze_146628 = 0;
    unsigned char *mem_144004 = NULL;
    int64_t mem_144005_cached_sizze_146629 = 0;
    unsigned char *mem_144005 = NULL;
    int64_t mem_144006_cached_sizze_146630 = 0;
    unsigned char *mem_144006 = NULL;
    int64_t mem_144019_cached_sizze_146631 = 0;
    unsigned char *mem_144019 = NULL;
    int64_t mem_144020_cached_sizze_146632 = 0;
    unsigned char *mem_144020 = NULL;
    int64_t mem_144021_cached_sizze_146633 = 0;
    unsigned char *mem_144021 = NULL;
    int64_t mem_144052_cached_sizze_146634 = 0;
    unsigned char *mem_144052 = NULL;
    int64_t mem_144053_cached_sizze_146635 = 0;
    unsigned char *mem_144053 = NULL;
    int64_t mem_144054_cached_sizze_146636 = 0;
    unsigned char *mem_144054 = NULL;
    int64_t mem_144070_cached_sizze_146637 = 0;
    unsigned char *mem_144070 = NULL;
    int64_t mem_144071_cached_sizze_146638 = 0;
    unsigned char *mem_144071 = NULL;
    int64_t mem_144072_cached_sizze_146639 = 0;
    unsigned char *mem_144072 = NULL;
    int64_t mem_144085_cached_sizze_146640 = 0;
    unsigned char *mem_144085 = NULL;
    int64_t mem_144086_cached_sizze_146641 = 0;
    unsigned char *mem_144086 = NULL;
    int64_t mem_144087_cached_sizze_146642 = 0;
    unsigned char *mem_144087 = NULL;
    int64_t mem_144133_cached_sizze_146643 = 0;
    unsigned char *mem_144133 = NULL;
    int64_t mem_144139_cached_sizze_146644 = 0;
    unsigned char *mem_144139 = NULL;
    int64_t mem_144144_cached_sizze_146645 = 0;
    unsigned char *mem_144144 = NULL;
    int64_t mem_144155_cached_sizze_146646 = 0;
    unsigned char *mem_144155 = NULL;
    int64_t mem_144160_cached_sizze_146647 = 0;
    unsigned char *mem_144160 = NULL;
    int64_t mem_144171_cached_sizze_146648 = 0;
    unsigned char *mem_144171 = NULL;
    int64_t mem_144176_cached_sizze_146649 = 0;
    unsigned char *mem_144176 = NULL;
    int64_t mem_144183_cached_sizze_146650 = 0;
    unsigned char *mem_144183 = NULL;
    int64_t mem_144190_cached_sizze_146651 = 0;
    unsigned char *mem_144190 = NULL;
    int64_t mem_144201_cached_sizze_146652 = 0;
    unsigned char *mem_144201 = NULL;
    int64_t mem_144206_cached_sizze_146653 = 0;
    unsigned char *mem_144206 = NULL;
    int64_t mem_144217_cached_sizze_146654 = 0;
    unsigned char *mem_144217 = NULL;
    int64_t mem_144222_cached_sizze_146655 = 0;
    unsigned char *mem_144222 = NULL;
    int64_t mem_144238_cached_sizze_146656 = 0;
    unsigned char *mem_144238 = NULL;
    int64_t mem_144243_cached_sizze_146657 = 0;
    unsigned char *mem_144243 = NULL;
    int64_t mem_144254_cached_sizze_146658 = 0;
    unsigned char *mem_144254 = NULL;
    int64_t mem_144259_cached_sizze_146659 = 0;
    unsigned char *mem_144259 = NULL;
    int64_t mem_144270_cached_sizze_146660 = 0;
    unsigned char *mem_144270 = NULL;
    int64_t mem_144275_cached_sizze_146661 = 0;
    unsigned char *mem_144275 = NULL;
    int64_t mem_144286_cached_sizze_146662 = 0;
    unsigned char *mem_144286 = NULL;
    int64_t mem_144291_cached_sizze_146663 = 0;
    unsigned char *mem_144291 = NULL;
    int64_t mem_144298_cached_sizze_146664 = 0;
    unsigned char *mem_144298 = NULL;
    int64_t mem_144309_cached_sizze_146665 = 0;
    unsigned char *mem_144309 = NULL;
    int64_t mem_144314_cached_sizze_146666 = 0;
    unsigned char *mem_144314 = NULL;
    int64_t mem_144325_cached_sizze_146667 = 0;
    unsigned char *mem_144325 = NULL;
    int64_t mem_144330_cached_sizze_146668 = 0;
    unsigned char *mem_144330 = NULL;
    int64_t mem_144341_cached_sizze_146669 = 0;
    unsigned char *mem_144341 = NULL;
    int64_t mem_144346_cached_sizze_146670 = 0;
    unsigned char *mem_144346 = NULL;
    int64_t mem_144357_cached_sizze_146671 = 0;
    unsigned char *mem_144357 = NULL;
    int64_t mem_144362_cached_sizze_146672 = 0;
    unsigned char *mem_144362 = NULL;
    int64_t mem_144373_cached_sizze_146673 = 0;
    unsigned char *mem_144373 = NULL;
    int64_t mem_144378_cached_sizze_146674 = 0;
    unsigned char *mem_144378 = NULL;
    int64_t mem_144394_cached_sizze_146675 = 0;
    unsigned char *mem_144394 = NULL;
    struct memblock mem_144389;
    
    mem_144389.references = NULL;
    
    struct memblock mem_out_146162;
    
    mem_out_146162.references = NULL;
    
    struct memblock mem_143922 = ctx->constants->mem_143922;
    struct memblock mem_143923 = ctx->constants->mem_143923;
    struct memblock mem_143924 = ctx->constants->mem_143924;
    struct memblock mem_143925 = ctx->constants->mem_143925;
    struct memblock mem_143926 = ctx->constants->mem_143926;
    struct memblock mem_143927 = ctx->constants->mem_143927;
    struct memblock mem_143928 = ctx->constants->mem_143928;
    struct memblock mem_143929 = ctx->constants->mem_143929;
    struct memblock mem_143930 = ctx->constants->mem_143930;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_143942_cached_sizze_146620 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143942, &mem_143942_cached_sizze_146620, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143947_cached_sizze_146621 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143947, &mem_143947_cached_sizze_146621, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142920 = 0; i_142920 < (int64_t) 16; i_142920++) {
        // futhark/microgpt.fut:461:41-50
        
        int64_t tmp_124408 = ((int64_t *) tokens_mem_143940.mem)[i_142920];
        
        // futhark/microgpt.fut:461:37-51
        
        bool x_124409 = sle64((int64_t) 0, tmp_124408);
        
        // futhark/microgpt.fut:461:37-51
        
        bool y_124410 = slt64(tmp_124408, (int64_t) 27);
        
        // futhark/microgpt.fut:461:37-51
        
        bool bounds_check_124411 = x_124409 && y_124410;
        
        // futhark/microgpt.fut:461:37-51
        
        bool index_certs_124412;
        
        if (!bounds_check_124411) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_124408, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:461:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:461:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142916 = 0; i_142916 < (int64_t) 16; i_142916++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_124419 = ((double *) wte_mem_143936.mem)[tmp_124408 * (int64_t) 16 + i_142916];
            
            ((double *) mem_143947)[i_142916] = lifted_lambda_res_124419;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143942, i_142920 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143947, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143958_cached_sizze_146622 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143958, &mem_143958_cached_sizze_146622, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143963_cached_sizze_146623 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143963, &mem_143963_cached_sizze_146623, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143970_cached_sizze_146624 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143970, &mem_143970_cached_sizze_146624, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142932 = 0; i_142932 < (int64_t) 16; i_142932++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_124445;
        double r_124447 = 0.0;
        
        for (int64_t i_124446 = 0; i_124446 < (int64_t) 16; i_124446++) {
            // futhark/microgpt.fut:71:46-49
            
            double zp_lhs_124448 = ((double *) wpe_mem_143934.mem)[i_142932 * (int64_t) 16 + i_124446];
            
            // futhark/microgpt.fut:71:46-49
            
            double zp_rhs_124449 = ((double *) mem_143942)[i_142932 * (int64_t) 16 + i_124446];
            
            // futhark/microgpt.fut:148:76-116
            
            double zp_res_124450 = zp_lhs_124448 + zp_rhs_124449;
            
            // futhark/microgpt.fut:148:94-163
            
            double zt_res_124451 = zp_res_124450 * zp_res_124450;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_124452 = r_124447 + zt_res_124451;
            double r_tmp_146166 = zp_res_124452;
            
            r_124447 = r_tmp_146166;
        }
        defunc_0_lifted_lambda_res_124445 = r_124447;
        // futhark/microgpt.fut:148:54-182
        
        double zs_res_124453 = defunc_0_lifted_lambda_res_124445 / 16.0;
        
        // futhark/microgpt.fut:149:24-55
        
        double zp_res_124454 = 1.0e-5 + zs_res_124453;
        
        // futhark/microgpt.fut:149:16-55
        
        double sqrt_res_124455 = futrts_sqrt64(zp_res_124454);
        
        // futhark/microgpt.fut:150:85-96
        
        double zs_res_124456 = 1.0 / sqrt_res_124455;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142924 = 0; i_142924 < (int64_t) 16; i_142924++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_124463 = ((double *) wpe_mem_143934.mem)[i_142932 * (int64_t) 16 + i_142924];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_124464 = ((double *) mem_143942)[i_142932 * (int64_t) 16 + i_142924];
            
            // futhark/microgpt.fut:150:38-78
            
            double zp_res_124465 = zp_lhs_124463 + zp_rhs_124464;
            
            // futhark/microgpt.fut:150:56-96
            
            double zt_res_124466 = zs_res_124456 * zp_res_124465;
            
            ((double *) mem_143963)[i_142924] = zt_res_124466;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142928 = 0; i_142928 < (int64_t) 16; i_142928++) {
            // futhark/microgpt.fut:151:4-14
            
            double lifted_lambda_res_124474 = ((double *) mem_143963)[i_142928];
            
            ((double *) mem_143970)[i_142928] = lifted_lambda_res_124474;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143958, i_142932 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143970, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143981_cached_sizze_146625 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_143981, &mem_143981_cached_sizze_146625, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143986_cached_sizze_146626 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143986, &mem_143986_cached_sizze_146626, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_143993_cached_sizze_146627 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_143993, &mem_143993_cached_sizze_146627, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142944 = 0; i_142944 < (int64_t) 16; i_142944++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_124483;
        double r_124485 = 0.0;
        
        for (int64_t i_124484 = 0; i_124484 < (int64_t) 16; i_124484++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_124486 = ((double *) mem_143958)[i_142944 * (int64_t) 16 + i_124484];
            
            // futhark/microgpt.fut:152:78-115
            
            double zt_res_124487 = zt_lhs_124486 * zt_lhs_124486;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_124488 = r_124485 + zt_res_124487;
            double r_tmp_146170 = zp_res_124488;
            
            r_124485 = r_tmp_146170;
        }
        defunc_0_lifted_lambda_res_124483 = r_124485;
        // futhark/microgpt.fut:152:57-133
        
        double zs_res_124489 = defunc_0_lifted_lambda_res_124483 / 16.0;
        
        // futhark/microgpt.fut:153:24-55
        
        double zp_res_124490 = 1.0e-5 + zs_res_124489;
        
        // futhark/microgpt.fut:153:16-55
        
        double sqrt_res_124491 = futrts_sqrt64(zp_res_124490);
        
        // futhark/microgpt.fut:154:59-70
        
        double zs_res_124492 = 1.0 / sqrt_res_124491;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142936 = 0; i_142936 < (int64_t) 16; i_142936++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_124499 = ((double *) mem_143958)[i_142944 * (int64_t) 16 + i_142936];
            
            // futhark/microgpt.fut:154:37-70
            
            double zt_res_124500 = zs_res_124492 * zt_lhs_124499;
            
            ((double *) mem_143986)[i_142936] = zt_res_124500;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142940 = 0; i_142940 < (int64_t) 16; i_142940++) {
            // futhark/microgpt.fut:155:4-14
            
            double lifted_lambda_res_124508 = ((double *) mem_143986)[i_142940];
            
            ((double *) mem_143993)[i_142940] = lifted_lambda_res_124508;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_143981, i_142944 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_143993, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144004_cached_sizze_146628 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144004, &mem_144004_cached_sizze_146628, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144005_cached_sizze_146629 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144005, &mem_144005_cached_sizze_146629, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144006_cached_sizze_146630 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144006, &mem_144006_cached_sizze_146630, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144019_cached_sizze_146631 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144019, &mem_144019_cached_sizze_146631, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144020_cached_sizze_146632 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144020, &mem_144020_cached_sizze_146632, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144021_cached_sizze_146633 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144021, &mem_144021_cached_sizze_146633, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142962 = 0; i_142962 < (int64_t) 16; i_142962++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142952 = 0; i_142952 < (int64_t) 16; i_142952++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_132251;
            double r_132253 = 0.0;
            
            for (int64_t i_132252 = 0; i_132252 < (int64_t) 16; i_132252++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_132254 = ((double *) wqry_mem_143935.mem)[i_142952 * (int64_t) 16 + i_132252];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_132255 = ((double *) mem_143981)[i_142962 * (int64_t) 16 + i_132252];
                
                // futhark/microgpt.fut:156:66-105
                
                double zt_res_132256 = zt_lhs_132254 * zt_rhs_132255;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_132257 = r_132253 + zt_res_132256;
                double r_tmp_146179 = zp_res_132257;
                
                r_132253 = r_tmp_146179;
            }
            defunc_0_lifted_lambda_res_132251 = r_132253;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_132264;
            double r_132266 = 0.0;
            
            for (int64_t i_132265 = 0; i_132265 < (int64_t) 16; i_132265++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_132267 = ((double *) wkey_mem_143932.mem)[i_142952 * (int64_t) 16 + i_132265];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_132268 = ((double *) mem_143981)[i_142962 * (int64_t) 16 + i_132265];
                
                // futhark/microgpt.fut:157:66-105
                
                double zt_res_132269 = zt_lhs_132267 * zt_rhs_132268;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_132270 = r_132266 + zt_res_132269;
                double r_tmp_146180 = zp_res_132270;
                
                r_132266 = r_tmp_146180;
            }
            defunc_0_lifted_lambda_res_132264 = r_132266;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_132280;
            double r_132282 = 0.0;
            
            for (int64_t i_132281 = 0; i_132281 < (int64_t) 16; i_132281++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_132283 = ((double *) wval_mem_143938.mem)[i_142952 * (int64_t) 16 + i_132281];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_132284 = ((double *) mem_143981)[i_142962 * (int64_t) 16 + i_132281];
                
                // futhark/microgpt.fut:158:66-105
                
                double zt_res_132285 = zt_lhs_132283 * zt_rhs_132284;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_132286 = r_132282 + zt_res_132285;
                double r_tmp_146181 = zp_res_132286;
                
                r_132282 = r_tmp_146181;
            }
            defunc_0_lifted_lambda_res_132280 = r_132282;
            ((double *) mem_144019)[i_142952] = defunc_0_lifted_lambda_res_132280;
            ((double *) mem_144020)[i_142952] = defunc_0_lifted_lambda_res_132264;
            ((double *) mem_144021)[i_142952] = defunc_0_lifted_lambda_res_132251;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144004, i_142962 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144019, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144005, i_142962 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144020, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144006, i_142962 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144021, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144052_cached_sizze_146634 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144052, &mem_144052_cached_sizze_146634, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144053_cached_sizze_146635 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144053, &mem_144053_cached_sizze_146635, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144054_cached_sizze_146636 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144054, &mem_144054_cached_sizze_146636, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144070_cached_sizze_146637 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144070, &mem_144070_cached_sizze_146637, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144071_cached_sizze_146638 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144071, &mem_144071_cached_sizze_146638, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144072_cached_sizze_146639 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144072, &mem_144072_cached_sizze_146639, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144085_cached_sizze_146640 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144085, &mem_144085_cached_sizze_146640, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144086_cached_sizze_146641 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144086, &mem_144086_cached_sizze_146641, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144087_cached_sizze_146642 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144087, &mem_144087_cached_sizze_146642, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_142992 = 0; i_142992 < (int64_t) 4; i_142992++) {
        // futhark/microgpt.fut:159:69-72
        
        int64_t zp_lhs_132127 = mul64((int64_t) 4, i_142992);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142982 = 0; i_142982 < (int64_t) 16; i_142982++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142972 = 0; i_142972 < (int64_t) 4; i_142972++) {
                // futhark/microgpt.fut:159:74-81
                
                int64_t tmp_132444 = add64(zp_lhs_132127, i_142972);
                
                // futhark/microgpt.fut:159:51-83
                
                bool x_132445 = sle64((int64_t) 0, tmp_132444);
                
                // futhark/microgpt.fut:159:51-83
                
                bool y_132446 = slt64(tmp_132444, (int64_t) 16);
                
                // futhark/microgpt.fut:159:51-83
                
                bool bounds_check_132447 = x_132445 && y_132446;
                
                // futhark/microgpt.fut:159:51-83
                
                bool index_certs_132448;
                
                if (!bounds_check_132447) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_132444, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:159:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:159:15-84\n   #9  futhark/microgpt.fut:462:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_132449 = ((double *) mem_144006)[i_142982 * (int64_t) 16 + tmp_132444];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_132457 = ((double *) mem_144005)[i_142982 * (int64_t) 16 + tmp_132444];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_132468 = ((double *) mem_144004)[i_142982 * (int64_t) 16 + tmp_132444];
                
                ((double *) mem_144085)[i_142972] = lifted_lambda_res_132468;
                ((double *) mem_144086)[i_142972] = lifted_lambda_res_132457;
                ((double *) mem_144087)[i_142972] = lifted_lambda_res_132449;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144070, i_142982 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144085, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144071, i_142982 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144086, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144072, i_142982 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144087, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144052, i_142992 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144070, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144053, i_142992 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144071, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144054, i_142992 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144072, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144133_cached_sizze_146643 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144133, &mem_144133_cached_sizze_146643, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144139_cached_sizze_146644 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144139, &mem_144139_cached_sizze_146644, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144144_cached_sizze_146645 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144144, &mem_144144_cached_sizze_146645, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144155_cached_sizze_146646 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144155, &mem_144155_cached_sizze_146646, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144160_cached_sizze_146647 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144160, &mem_144160_cached_sizze_146647, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144171_cached_sizze_146648 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144171, &mem_144171_cached_sizze_146648, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144176_cached_sizze_146649 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144176, &mem_144176_cached_sizze_146649, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144183_cached_sizze_146650 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144183, &mem_144183_cached_sizze_146650, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144190_cached_sizze_146651 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144190, &mem_144190_cached_sizze_146651, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144201_cached_sizze_146652 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144201, &mem_144201_cached_sizze_146652, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144206_cached_sizze_146653 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144206, &mem_144206_cached_sizze_146653, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144217_cached_sizze_146654 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144217, &mem_144217_cached_sizze_146654, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144222_cached_sizze_146655 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144222, &mem_144222_cached_sizze_146655, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143048 = 0; i_143048 < (int64_t) 4; i_143048++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143002 = 0; i_143002 < (int64_t) 16; i_143002++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142998 = 0; i_142998 < (int64_t) 16; i_142998++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_124653;
                double r_124655 = 0.0;
                
                for (int64_t i_124654 = 0; i_124654 < (int64_t) 4; i_124654++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_124656 = ((double *) mem_144054)[i_143048 * (int64_t) 64 + i_143002 * (int64_t) 4 + i_124654];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_124657 = ((double *) mem_144053)[i_143048 * (int64_t) 64 + i_142998 * (int64_t) 4 + i_124654];
                    
                    // futhark/microgpt.fut:162:113-164
                    
                    double zt_res_124658 = zt_lhs_124656 * zt_rhs_124657;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_124659 = r_124655 + zt_res_124658;
                    double r_tmp_146194 = zp_res_124659;
                    
                    r_124655 = r_tmp_146194;
                }
                defunc_0_lifted_lambda_res_124653 = r_124655;
                ((double *) mem_144144)[i_142998] = defunc_0_lifted_lambda_res_124653;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144139, i_143002 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144144, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143010 = 0; i_143010 < (int64_t) 16; i_143010++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143006 = 0; i_143006 < (int64_t) 16; i_143006++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_124674 = ((double *) mem_144139)[i_143010 * (int64_t) 16 + i_143006];
                
                // futhark/microgpt.fut:163:47-78
                
                double zs_res_124675 = zs_lhs_124674 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_124676 = ((double *) mask_mem_143941.mem)[i_143010 * (int64_t) 16 + i_143006];
                
                // futhark/microgpt.fut:163:65-102
                
                double zp_res_124677 = zs_res_124675 + zp_rhs_124676;
                
                ((double *) mem_144160)[i_143006] = zp_res_124677;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144155, i_143010 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144160, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143028 = 0; i_143028 < (int64_t) 16; i_143028++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_132546;
            double redout_143012 = -INFINITY;
            
            for (int64_t i_143013 = 0; i_143013 < (int64_t) 16; i_143013++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_132495 = ((double *) mem_144155)[i_143028 * (int64_t) 16 + i_143013];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_124698 = fmax64(lifted_lambda_res_132495, redout_143012);
                double redout_tmp_146198 = max_res_124698;
                
                redout_143012 = redout_tmp_146198;
            }
            defunc_0_reduce_res_132546 = redout_143012;
            // futhark/microgpt.fut:165:67-76
            
            double neg_res_124699 = -defunc_0_reduce_res_132546;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143016 = 0; i_143016 < (int64_t) 16; i_143016++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_124706 = ((double *) mem_144155)[i_143028 * (int64_t) 16 + i_143016];
                
                // futhark/microgpt.fut:165:44-76
                
                double zp_res_124707 = neg_res_124699 + zp_lhs_124706;
                
                // futhark/microgpt.fut:165:37-76
                
                double exp_res_124708 = futrts_exp64(zp_res_124707);
                
                ((double *) mem_144176)[i_143016] = exp_res_124708;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124710;
            double r_124712 = 0.0;
            
            for (int64_t i_124711 = 0; i_124711 < (int64_t) 16; i_124711++) {
                // futhark/microgpt.fut:166:36-46
                
                double lifted_lambda_res_124713 = ((double *) mem_144176)[i_124711];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_124714 = r_124712 + lifted_lambda_res_124713;
                double r_tmp_146200 = zp_res_124714;
                
                r_124712 = r_tmp_146200;
            }
            defunc_0_lifted_lambda_res_124710 = r_124712;
            // futhark/microgpt.fut:167:53-64
            
            double zs_res_124715 = 1.0 / defunc_0_lifted_lambda_res_124710;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143020 = 0; i_143020 < (int64_t) 16; i_143020++) {
                // futhark/microgpt.fut:167:37-47
                
                double zt_lhs_124722 = ((double *) mem_144176)[i_143020];
                
                // futhark/microgpt.fut:167:37-64
                
                double zt_res_124723 = zs_res_124715 * zt_lhs_124722;
                
                ((double *) mem_144183)[i_143020] = zt_res_124723;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143024 = 0; i_143024 < (int64_t) 16; i_143024++) {
                // futhark/microgpt.fut:168:4-14
                
                double lifted_lambda_res_124731 = ((double *) mem_144183)[i_143024];
                
                ((double *) mem_144190)[i_143024] = lifted_lambda_res_124731;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144171, i_143028 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144190, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143036 = 0; i_143036 < (int64_t) 16; i_143036++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143032 = 0; i_143032 < (int64_t) 4; i_143032++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_124746;
                double r_124748 = 0.0;
                
                for (int64_t i_124747 = 0; i_124747 < (int64_t) 16; i_124747++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_124749 = ((double *) mem_144171)[i_143036 * (int64_t) 16 + i_124747];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_124750 = ((double *) mem_144052)[i_143048 * (int64_t) 64 + i_124747 * (int64_t) 4 + i_143032];
                    
                    // futhark/microgpt.fut:169:66-111
                    
                    double zt_res_124751 = zt_lhs_124749 * zt_rhs_124750;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_124752 = r_124748 + zt_res_124751;
                    double r_tmp_146205 = zp_res_124752;
                    
                    r_124748 = r_tmp_146205;
                }
                defunc_0_lifted_lambda_res_124746 = r_124748;
                ((double *) mem_144206)[i_143032] = defunc_0_lifted_lambda_res_124746;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144201, i_143036 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144206, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143044 = 0; i_143044 < (int64_t) 16; i_143044++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143040 = 0; i_143040 < (int64_t) 4; i_143040++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_124767 = ((double *) mem_144201)[i_143044 * (int64_t) 4 + i_143040];
                
                ((double *) mem_144222)[i_143040] = lifted_lambda_res_124767;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144217, i_143044 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144222, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_144133, i_143048 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144217, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144238_cached_sizze_146656 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144238, &mem_144238_cached_sizze_146656, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144243_cached_sizze_146657 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144243, &mem_144243_cached_sizze_146657, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143056 = 0; i_143056 < (int64_t) 16; i_143056++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143052 = 0; i_143052 < (int64_t) 16; i_143052++) {
            // futhark/microgpt.fut:171:54-57
            
            int64_t tmp_124779 = sdiv64(i_143052, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-59
            
            bool x_124780 = sle64((int64_t) 0, tmp_124779);
            
            // futhark/microgpt.fut:171:44-59
            
            bool y_124781 = slt64(tmp_124779, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-59
            
            bool bounds_check_124782 = x_124780 && y_124781;
            
            // futhark/microgpt.fut:171:44-59
            
            bool index_certs_124783;
            
            if (!bounds_check_124782) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_124779, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:171:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:171:15-80\n   #6  futhark/microgpt.fut:462:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:171:74-77
            
            int64_t tmp_124784 = smod64(i_143052, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-79
            
            bool x_124785 = sle64((int64_t) 0, tmp_124784);
            
            // futhark/microgpt.fut:171:44-79
            
            bool y_124786 = slt64(tmp_124784, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-79
            
            bool bounds_check_124787 = x_124785 && y_124786;
            
            // futhark/microgpt.fut:171:44-79
            
            bool index_certs_124788;
            
            if (!bounds_check_124787) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_124784, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:171:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:171:15-80\n   #6  futhark/microgpt.fut:462:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_124789 = ((double *) mem_144133)[tmp_124779 * (int64_t) 64 + i_143056 * (int64_t) 4 + tmp_124784];
            
            ((double *) mem_144243)[i_143052] = lifted_lambda_res_124789;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144238, i_143056 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144243, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144254_cached_sizze_146658 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144254, &mem_144254_cached_sizze_146658, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144259_cached_sizze_146659 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144259, &mem_144259_cached_sizze_146659, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143064 = 0; i_143064 < (int64_t) 16; i_143064++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143060 = 0; i_143060 < (int64_t) 16; i_143060++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124804;
            double r_124806 = 0.0;
            
            for (int64_t i_124805 = 0; i_124805 < (int64_t) 16; i_124805++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_124807 = ((double *) wout_mem_143933.mem)[i_143060 * (int64_t) 16 + i_124805];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_124808 = ((double *) mem_144238)[i_143064 * (int64_t) 16 + i_124805];
                
                // futhark/microgpt.fut:172:67-106
                
                double zt_res_124809 = zt_lhs_124807 * zt_rhs_124808;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_124810 = r_124806 + zt_res_124809;
                double r_tmp_146212 = zp_res_124810;
                
                r_124806 = r_tmp_146212;
            }
            defunc_0_lifted_lambda_res_124804 = r_124806;
            ((double *) mem_144259)[i_143060] = defunc_0_lifted_lambda_res_124804;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144254, i_143064 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144259, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144270_cached_sizze_146660 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144270, &mem_144270_cached_sizze_146660, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144275_cached_sizze_146661 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144275, &mem_144275_cached_sizze_146661, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143072 = 0; i_143072 < (int64_t) 16; i_143072++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143068 = 0; i_143068 < (int64_t) 16; i_143068++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_124825 = ((double *) mem_144254)[i_143072 * (int64_t) 16 + i_143068];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_124826 = ((double *) mem_143958)[i_143072 * (int64_t) 16 + i_143068];
            
            // futhark/microgpt.fut:173:46-84
            
            double zp_res_124827 = zp_lhs_124825 + zp_rhs_124826;
            
            ((double *) mem_144275)[i_143068] = zp_res_124827;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144270, i_143072 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144275, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144286_cached_sizze_146662 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144286, &mem_144286_cached_sizze_146662, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144291_cached_sizze_146663 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144291, &mem_144291_cached_sizze_146663, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144298_cached_sizze_146664 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144298, &mem_144298_cached_sizze_146664, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143084 = 0; i_143084 < (int64_t) 16; i_143084++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_124836;
        double r_124838 = 0.0;
        
        for (int64_t i_124837 = 0; i_124837 < (int64_t) 16; i_124837++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_124839 = ((double *) mem_144270)[i_143084 * (int64_t) 16 + i_124837];
            
            // futhark/microgpt.fut:174:79-118
            
            double zt_res_124840 = zt_lhs_124839 * zt_lhs_124839;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_124841 = r_124838 + zt_res_124840;
            double r_tmp_146216 = zp_res_124841;
            
            r_124838 = r_tmp_146216;
        }
        defunc_0_lifted_lambda_res_124836 = r_124838;
        // futhark/microgpt.fut:174:58-136
        
        double zs_res_124842 = defunc_0_lifted_lambda_res_124836 / 16.0;
        
        // futhark/microgpt.fut:175:24-55
        
        double zp_res_124843 = 1.0e-5 + zs_res_124842;
        
        // futhark/microgpt.fut:175:16-55
        
        double sqrt_res_124844 = futrts_sqrt64(zp_res_124843);
        
        // futhark/microgpt.fut:176:60-71
        
        double zs_res_124845 = 1.0 / sqrt_res_124844;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143076 = 0; i_143076 < (int64_t) 16; i_143076++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_124852 = ((double *) mem_144270)[i_143084 * (int64_t) 16 + i_143076];
            
            // futhark/microgpt.fut:176:37-71
            
            double zt_res_124853 = zs_res_124845 * zt_lhs_124852;
            
            ((double *) mem_144291)[i_143076] = zt_res_124853;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143080 = 0; i_143080 < (int64_t) 16; i_143080++) {
            // futhark/microgpt.fut:177:4-14
            
            double lifted_lambda_res_124861 = ((double *) mem_144291)[i_143080];
            
            ((double *) mem_144298)[i_143080] = lifted_lambda_res_124861;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144286, i_143084 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144298, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144309_cached_sizze_146665 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144309, &mem_144309_cached_sizze_146665, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144314_cached_sizze_146666 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144314, &mem_144314_cached_sizze_146666, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143092 = 0; i_143092 < (int64_t) 16; i_143092++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143088 = 0; i_143088 < (int64_t) 64; i_143088++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124877;
            double r_124879 = 0.0;
            
            for (int64_t i_124878 = 0; i_124878 < (int64_t) 16; i_124878++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_124880 = ((double *) wup_mem_143937.mem)[i_143088 * (int64_t) 16 + i_124878];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_124881 = ((double *) mem_144286)[i_143092 * (int64_t) 16 + i_124878];
                
                // futhark/microgpt.fut:178:67-106
                
                double zt_res_124882 = zt_lhs_124880 * zt_rhs_124881;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_124883 = r_124879 + zt_res_124882;
                double r_tmp_146221 = zp_res_124883;
                
                r_124879 = r_tmp_146221;
            }
            defunc_0_lifted_lambda_res_124877 = r_124879;
            ((double *) mem_144314)[i_143088] = defunc_0_lifted_lambda_res_124877;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144309, i_143092 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144314, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144325_cached_sizze_146667 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144325, &mem_144325_cached_sizze_146667, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144330_cached_sizze_146668 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144330, &mem_144330_cached_sizze_146668, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143100 = 0; i_143100 < (int64_t) 16; i_143100++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143096 = 0; i_143096 < (int64_t) 64; i_143096++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_124898 = ((double *) mem_144309)[i_143100 * (int64_t) 64 + i_143096];
            
            // futhark/microgpt.fut:179:45-73
            
            double max_res_124899 = fmax64(0.0, max_arg0_124898);
            
            ((double *) mem_144330)[i_143096] = max_res_124899;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144325, i_143100 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144330, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144341_cached_sizze_146669 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144341, &mem_144341_cached_sizze_146669, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144346_cached_sizze_146670 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144346, &mem_144346_cached_sizze_146670, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143108 = 0; i_143108 < (int64_t) 16; i_143108++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143104 = 0; i_143104 < (int64_t) 16; i_143104++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124914;
            double r_124916 = 0.0;
            
            for (int64_t i_124915 = 0; i_124915 < (int64_t) 64; i_124915++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_124917 = ((double *) wdown_mem_143931.mem)[i_143104 * (int64_t) 64 + i_124915];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_124918 = ((double *) mem_144325)[i_143108 * (int64_t) 64 + i_124915];
                
                // futhark/microgpt.fut:180:67-108
                
                double zt_res_124919 = zt_lhs_124917 * zt_rhs_124918;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_124920 = r_124916 + zt_res_124919;
                double r_tmp_146226 = zp_res_124920;
                
                r_124916 = r_tmp_146226;
            }
            defunc_0_lifted_lambda_res_124914 = r_124916;
            ((double *) mem_144346)[i_143104] = defunc_0_lifted_lambda_res_124914;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144341, i_143108 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144346, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144357_cached_sizze_146671 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144357, &mem_144357_cached_sizze_146671, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144362_cached_sizze_146672 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144362, &mem_144362_cached_sizze_146672, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143116 = 0; i_143116 < (int64_t) 16; i_143116++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143112 = 0; i_143112 < (int64_t) 16; i_143112++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_124935 = ((double *) mem_144341)[i_143116 * (int64_t) 16 + i_143112];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_124936 = ((double *) mem_144270)[i_143116 * (int64_t) 16 + i_143112];
            
            // futhark/microgpt.fut:181:46-85
            
            double zp_res_124937 = zp_lhs_124935 + zp_rhs_124936;
            
            ((double *) mem_144362)[i_143112] = zp_res_124937;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144357, i_143116 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144362, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144373_cached_sizze_146673 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144373, &mem_144373_cached_sizze_146673, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144378_cached_sizze_146674 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144378, &mem_144378_cached_sizze_146674, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143124 = 0; i_143124 < (int64_t) 16; i_143124++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143120 = 0; i_143120 < (int64_t) 27; i_143120++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_124953;
            double r_124955 = 0.0;
            
            for (int64_t i_124954 = 0; i_124954 < (int64_t) 16; i_124954++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_124956 = ((double *) wvoc_mem_143939.mem)[i_143120 * (int64_t) 16 + i_124954];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_124957 = ((double *) mem_144357)[i_143124 * (int64_t) 16 + i_124954];
                
                // futhark/microgpt.fut:182:67-107
                
                double zt_res_124958 = zt_lhs_124956 * zt_rhs_124957;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_124959 = r_124955 + zt_res_124958;
                double r_tmp_146231 = zp_res_124959;
                
                r_124955 = r_tmp_146231;
            }
            defunc_0_lifted_lambda_res_124953 = r_124955;
            ((double *) mem_144378)[i_143120] = defunc_0_lifted_lambda_res_124953;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144373, i_143124 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144378, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_144389, (int64_t) 3456, "mem_144389")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144394_cached_sizze_146675 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144394, &mem_144394_cached_sizze_146675, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_143132 = 0; i_143132 < (int64_t) 16; i_143132++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143128 = 0; i_143128 < (int64_t) 27; i_143128++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_124974 = ((double *) mem_144373)[i_143132 * (int64_t) 27 + i_143128];
            
            ((double *) mem_144394)[i_143128] = lifted_lambda_res_124974;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_144389.mem, i_143132 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144394, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_146162, &mem_144389, "mem_144389") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146619, &mem_out_146162, "mem_out_146162") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_143942);
        free(mem_143947);
        free(mem_143958);
        free(mem_143963);
        free(mem_143970);
        free(mem_143981);
        free(mem_143986);
        free(mem_143993);
        free(mem_144004);
        free(mem_144005);
        free(mem_144006);
        free(mem_144019);
        free(mem_144020);
        free(mem_144021);
        free(mem_144052);
        free(mem_144053);
        free(mem_144054);
        free(mem_144070);
        free(mem_144071);
        free(mem_144072);
        free(mem_144085);
        free(mem_144086);
        free(mem_144087);
        free(mem_144133);
        free(mem_144139);
        free(mem_144144);
        free(mem_144155);
        free(mem_144160);
        free(mem_144171);
        free(mem_144176);
        free(mem_144183);
        free(mem_144190);
        free(mem_144201);
        free(mem_144206);
        free(mem_144217);
        free(mem_144222);
        free(mem_144238);
        free(mem_144243);
        free(mem_144254);
        free(mem_144259);
        free(mem_144270);
        free(mem_144275);
        free(mem_144286);
        free(mem_144291);
        free(mem_144298);
        free(mem_144309);
        free(mem_144314);
        free(mem_144325);
        free(mem_144330);
        free(mem_144341);
        free(mem_144346);
        free(mem_144357);
        free(mem_144362);
        free(mem_144373);
        free(mem_144378);
        free(mem_144394);
        if (memblock_unref(ctx, &mem_144389, "mem_144389") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146162, "mem_out_146162") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_146676, struct memblock *mem_out_p_146677, struct memblock *mem_out_p_146678, struct memblock *mem_out_p_146679, struct memblock *mem_out_p_146680, struct memblock *mem_out_p_146681, struct memblock *mem_out_p_146682, struct memblock *mem_out_p_146683, struct memblock *mem_out_p_146684, struct memblock wte_mem_143931, struct memblock wpe_mem_143932, struct memblock wqry_mem_143933, struct memblock wkey_mem_143934, struct memblock wval_mem_143935, struct memblock wout_mem_143936, struct memblock wup_mem_143937, struct memblock wdown_mem_143938, struct memblock wvoc_mem_143939)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_146170;
    
    mem_out_146170.references = NULL;
    
    struct memblock mem_out_146169;
    
    mem_out_146169.references = NULL;
    
    struct memblock mem_out_146168;
    
    mem_out_146168.references = NULL;
    
    struct memblock mem_out_146167;
    
    mem_out_146167.references = NULL;
    
    struct memblock mem_out_146166;
    
    mem_out_146166.references = NULL;
    
    struct memblock mem_out_146165;
    
    mem_out_146165.references = NULL;
    
    struct memblock mem_out_146164;
    
    mem_out_146164.references = NULL;
    
    struct memblock mem_out_146163;
    
    mem_out_146163.references = NULL;
    
    struct memblock mem_out_146162;
    
    mem_out_146162.references = NULL;
    
    struct memblock mem_143922 = ctx->constants->mem_143922;
    struct memblock mem_143923 = ctx->constants->mem_143923;
    struct memblock mem_143924 = ctx->constants->mem_143924;
    struct memblock mem_143925 = ctx->constants->mem_143925;
    struct memblock mem_143926 = ctx->constants->mem_143926;
    struct memblock mem_143927 = ctx->constants->mem_143927;
    struct memblock mem_143928 = ctx->constants->mem_143928;
    struct memblock mem_143929 = ctx->constants->mem_143929;
    struct memblock mem_143930 = ctx->constants->mem_143930;
    
    if (memblock_set(ctx, &mem_out_146162, &wdown_mem_143938, "wdown_mem_143938") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146163, &wkey_mem_143934, "wkey_mem_143934") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146164, &wout_mem_143936, "wout_mem_143936") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146165, &wpe_mem_143932, "wpe_mem_143932") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146166, &wqry_mem_143933, "wqry_mem_143933") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146167, &wte_mem_143931, "wte_mem_143931") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146168, &wup_mem_143937, "wup_mem_143937") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146169, &wval_mem_143935, "wval_mem_143935") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146170, &wvoc_mem_143939, "wvoc_mem_143939") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146676, &mem_out_146162, "mem_out_146162") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146677, &mem_out_146163, "mem_out_146163") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146678, &mem_out_146164, "mem_out_146164") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146679, &mem_out_146165, "mem_out_146165") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146680, &mem_out_146166, "mem_out_146166") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146681, &mem_out_146167, "mem_out_146167") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146682, &mem_out_146168, "mem_out_146168") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146683, &mem_out_146169, "mem_out_146169") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146684, &mem_out_146170, "mem_out_146170") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_146170, "mem_out_146170") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146169, "mem_out_146169") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146168, "mem_out_146168") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146167, "mem_out_146167") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146166, "mem_out_146166") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146165, "mem_out_146165") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146164, "mem_out_146164") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146163, "mem_out_146163") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146162, "mem_out_146162") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_146685, struct memblock *mem_out_p_146686, struct memblock *mem_out_p_146687, struct memblock *mem_out_p_146688, struct memblock *mem_out_p_146689, struct memblock *mem_out_p_146690, struct memblock *mem_out_p_146691, struct memblock *mem_out_p_146692, struct memblock *mem_out_p_146693, struct memblock *mem_out_p_146694, struct memblock *mem_out_p_146695, struct memblock *mem_out_p_146696, struct memblock *mem_out_p_146697, struct memblock *mem_out_p_146698, struct memblock *mem_out_p_146699, struct memblock *mem_out_p_146700, struct memblock *mem_out_p_146701, struct memblock *mem_out_p_146702, struct memblock *mem_out_p_146703, struct memblock *mem_out_p_146704, struct memblock *mem_out_p_146705, struct memblock *mem_out_p_146706, struct memblock *mem_out_p_146707, struct memblock *mem_out_p_146708, struct memblock *mem_out_p_146709, struct memblock *mem_out_p_146710, struct memblock *mem_out_p_146711, struct memblock wdown_mem_143931, struct memblock wkey_mem_143932, struct memblock wout_mem_143933, struct memblock wpe_mem_143934, struct memblock wqry_mem_143935, struct memblock wte_mem_143936, struct memblock wup_mem_143937, struct memblock wval_mem_143938, struct memblock wvoc_mem_143939, struct memblock wdown_mem_143940, struct memblock wkey_mem_143941, struct memblock wout_mem_143942, struct memblock wpe_mem_143943, struct memblock wqry_mem_143944, struct memblock wte_mem_143945, struct memblock wup_mem_143946, struct memblock wval_mem_143947, struct memblock wvoc_mem_143948, struct memblock wdown_mem_143949, struct memblock wkey_mem_143950, struct memblock wout_mem_143951, struct memblock wpe_mem_143952, struct memblock wqry_mem_143953, struct memblock wte_mem_143954, struct memblock wup_mem_143955, struct memblock wval_mem_143956, struct memblock wvoc_mem_143957, struct memblock masks_mem_143958, struct memblock dls_mem_143959, struct memblock seqs_mem_143960)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_144069_cached_sizze_146712 = 0;
    unsigned char *mem_144069 = NULL;
    int64_t mem_144070_cached_sizze_146713 = 0;
    unsigned char *mem_144070 = NULL;
    int64_t mem_144079_cached_sizze_146714 = 0;
    unsigned char *mem_144079 = NULL;
    int64_t mem_144086_cached_sizze_146715 = 0;
    unsigned char *mem_144086 = NULL;
    int64_t mem_144101_cached_sizze_146716 = 0;
    unsigned char *mem_144101 = NULL;
    int64_t mem_144102_cached_sizze_146717 = 0;
    unsigned char *mem_144102 = NULL;
    int64_t mem_144103_cached_sizze_146718 = 0;
    unsigned char *mem_144103 = NULL;
    int64_t mem_144114_cached_sizze_146719 = 0;
    unsigned char *mem_144114 = NULL;
    int64_t mem_144121_cached_sizze_146720 = 0;
    unsigned char *mem_144121 = NULL;
    int64_t mem_144138_cached_sizze_146721 = 0;
    unsigned char *mem_144138 = NULL;
    int64_t mem_144139_cached_sizze_146722 = 0;
    unsigned char *mem_144139 = NULL;
    int64_t mem_144147_cached_sizze_146723 = 0;
    unsigned char *mem_144147 = NULL;
    int64_t mem_144154_cached_sizze_146724 = 0;
    unsigned char *mem_144154 = NULL;
    int64_t mem_144168_cached_sizze_146725 = 0;
    unsigned char *mem_144168 = NULL;
    int64_t mem_144169_cached_sizze_146726 = 0;
    unsigned char *mem_144169 = NULL;
    int64_t mem_144170_cached_sizze_146727 = 0;
    unsigned char *mem_144170 = NULL;
    int64_t mem_144186_cached_sizze_146728 = 0;
    unsigned char *mem_144186 = NULL;
    int64_t mem_144187_cached_sizze_146729 = 0;
    unsigned char *mem_144187 = NULL;
    int64_t mem_144188_cached_sizze_146730 = 0;
    unsigned char *mem_144188 = NULL;
    int64_t mem_144201_cached_sizze_146731 = 0;
    unsigned char *mem_144201 = NULL;
    int64_t mem_144202_cached_sizze_146732 = 0;
    unsigned char *mem_144202 = NULL;
    int64_t mem_144203_cached_sizze_146733 = 0;
    unsigned char *mem_144203 = NULL;
    int64_t mem_144249_cached_sizze_146734 = 0;
    unsigned char *mem_144249 = NULL;
    int64_t mem_144254_cached_sizze_146735 = 0;
    unsigned char *mem_144254 = NULL;
    int64_t mem_144258_cached_sizze_146736 = 0;
    unsigned char *mem_144258 = NULL;
    int64_t mem_144263_cached_sizze_146737 = 0;
    unsigned char *mem_144263 = NULL;
    int64_t mem_144274_cached_sizze_146738 = 0;
    unsigned char *mem_144274 = NULL;
    int64_t mem_144279_cached_sizze_146739 = 0;
    unsigned char *mem_144279 = NULL;
    int64_t mem_144290_cached_sizze_146740 = 0;
    unsigned char *mem_144290 = NULL;
    int64_t mem_144295_cached_sizze_146741 = 0;
    unsigned char *mem_144295 = NULL;
    int64_t mem_144302_cached_sizze_146742 = 0;
    unsigned char *mem_144302 = NULL;
    int64_t mem_144309_cached_sizze_146743 = 0;
    unsigned char *mem_144309 = NULL;
    int64_t mem_144320_cached_sizze_146744 = 0;
    unsigned char *mem_144320 = NULL;
    int64_t mem_144325_cached_sizze_146745 = 0;
    unsigned char *mem_144325 = NULL;
    int64_t mem_144343_cached_sizze_146746 = 0;
    unsigned char *mem_144343 = NULL;
    int64_t mem_144348_cached_sizze_146747 = 0;
    unsigned char *mem_144348 = NULL;
    int64_t mem_144359_cached_sizze_146748 = 0;
    unsigned char *mem_144359 = NULL;
    int64_t mem_144360_cached_sizze_146749 = 0;
    unsigned char *mem_144360 = NULL;
    int64_t mem_144368_cached_sizze_146750 = 0;
    unsigned char *mem_144368 = NULL;
    int64_t mem_144375_cached_sizze_146751 = 0;
    unsigned char *mem_144375 = NULL;
    int64_t mem_144389_cached_sizze_146752 = 0;
    unsigned char *mem_144389 = NULL;
    int64_t mem_144394_cached_sizze_146753 = 0;
    unsigned char *mem_144394 = NULL;
    int64_t mem_144405_cached_sizze_146754 = 0;
    unsigned char *mem_144405 = NULL;
    int64_t mem_144410_cached_sizze_146755 = 0;
    unsigned char *mem_144410 = NULL;
    int64_t mem_144421_cached_sizze_146756 = 0;
    unsigned char *mem_144421 = NULL;
    int64_t mem_144426_cached_sizze_146757 = 0;
    unsigned char *mem_144426 = NULL;
    int64_t mem_144437_cached_sizze_146758 = 0;
    unsigned char *mem_144437 = NULL;
    int64_t mem_144442_cached_sizze_146759 = 0;
    unsigned char *mem_144442 = NULL;
    int64_t mem_144453_cached_sizze_146760 = 0;
    unsigned char *mem_144453 = NULL;
    int64_t mem_144454_cached_sizze_146761 = 0;
    unsigned char *mem_144454 = NULL;
    int64_t mem_144455_cached_sizze_146762 = 0;
    unsigned char *mem_144455 = NULL;
    int64_t mem_144456_cached_sizze_146763 = 0;
    unsigned char *mem_144456 = NULL;
    int64_t mem_144474_cached_sizze_146764 = 0;
    unsigned char *mem_144474 = NULL;
    int64_t mem_144479_cached_sizze_146765 = 0;
    unsigned char *mem_144479 = NULL;
    int64_t mem_144483_cached_sizze_146766 = 0;
    unsigned char *mem_144483 = NULL;
    int64_t mem_144490_cached_sizze_146767 = 0;
    unsigned char *mem_144490 = NULL;
    int64_t mem_144524_cached_sizze_146768 = 0;
    unsigned char *mem_144524 = NULL;
    int64_t mem_144530_cached_sizze_146769 = 0;
    unsigned char *mem_144530 = NULL;
    int64_t mem_144535_cached_sizze_146770 = 0;
    unsigned char *mem_144535 = NULL;
    int64_t mem_144551_cached_sizze_146771 = 0;
    unsigned char *mem_144551 = NULL;
    int64_t mem_144552_cached_sizze_146772 = 0;
    unsigned char *mem_144552 = NULL;
    int64_t mem_144561_cached_sizze_146773 = 0;
    unsigned char *mem_144561 = NULL;
    int64_t mem_144562_cached_sizze_146774 = 0;
    unsigned char *mem_144562 = NULL;
    int64_t mem_144583_cached_sizze_146775 = 0;
    unsigned char *mem_144583 = NULL;
    int64_t mem_144589_cached_sizze_146776 = 0;
    unsigned char *mem_144589 = NULL;
    int64_t mem_144594_cached_sizze_146777 = 0;
    unsigned char *mem_144594 = NULL;
    int64_t mem_144610_cached_sizze_146778 = 0;
    unsigned char *mem_144610 = NULL;
    int64_t mem_144615_cached_sizze_146779 = 0;
    unsigned char *mem_144615 = NULL;
    int64_t mem_144626_cached_sizze_146780 = 0;
    unsigned char *mem_144626 = NULL;
    int64_t mem_144631_cached_sizze_146781 = 0;
    unsigned char *mem_144631 = NULL;
    int64_t mem_144642_cached_sizze_146782 = 0;
    unsigned char *mem_144642 = NULL;
    int64_t mem_144647_cached_sizze_146783 = 0;
    unsigned char *mem_144647 = NULL;
    int64_t mem_144658_cached_sizze_146784 = 0;
    unsigned char *mem_144658 = NULL;
    int64_t mem_144659_cached_sizze_146785 = 0;
    unsigned char *mem_144659 = NULL;
    int64_t mem_144668_cached_sizze_146786 = 0;
    unsigned char *mem_144668 = NULL;
    int64_t mem_144669_cached_sizze_146787 = 0;
    unsigned char *mem_144669 = NULL;
    int64_t mem_144690_cached_sizze_146788 = 0;
    unsigned char *mem_144690 = NULL;
    int64_t mem_144695_cached_sizze_146789 = 0;
    unsigned char *mem_144695 = NULL;
    int64_t mem_144706_cached_sizze_146790 = 0;
    unsigned char *mem_144706 = NULL;
    int64_t mem_144707_cached_sizze_146791 = 0;
    unsigned char *mem_144707 = NULL;
    int64_t mem_144720_cached_sizze_146792 = 0;
    unsigned char *mem_144720 = NULL;
    int64_t mem_144727_cached_sizze_146793 = 0;
    unsigned char *mem_144727 = NULL;
    int64_t mem_144732_cached_sizze_146794 = 0;
    unsigned char *mem_144732 = NULL;
    int64_t mem_144743_cached_sizze_146795 = 0;
    unsigned char *mem_144743 = NULL;
    int64_t mem_144744_cached_sizze_146796 = 0;
    unsigned char *mem_144744 = NULL;
    int64_t mem_144745_cached_sizze_146797 = 0;
    unsigned char *mem_144745 = NULL;
    int64_t mem_144746_cached_sizze_146798 = 0;
    unsigned char *mem_144746 = NULL;
    int64_t mem_144767_cached_sizze_146799 = 0;
    unsigned char *mem_144767 = NULL;
    int64_t mem_144768_cached_sizze_146800 = 0;
    unsigned char *mem_144768 = NULL;
    int64_t mem_144769_cached_sizze_146801 = 0;
    unsigned char *mem_144769 = NULL;
    int64_t mem_144770_cached_sizze_146802 = 0;
    unsigned char *mem_144770 = NULL;
    int64_t mem_144787_cached_sizze_146803 = 0;
    unsigned char *mem_144787 = NULL;
    int64_t mem_144794_cached_sizze_146804 = 0;
    unsigned char *mem_144794 = NULL;
    int64_t mem_144795_cached_sizze_146805 = 0;
    unsigned char *mem_144795 = NULL;
    int64_t mem_144796_cached_sizze_146806 = 0;
    unsigned char *mem_144796 = NULL;
    int64_t mem_144851_cached_sizze_146807 = 0;
    unsigned char *mem_144851 = NULL;
    int64_t mem_144852_cached_sizze_146808 = 0;
    unsigned char *mem_144852 = NULL;
    int64_t mem_144853_cached_sizze_146809 = 0;
    unsigned char *mem_144853 = NULL;
    int64_t mem_144854_cached_sizze_146810 = 0;
    unsigned char *mem_144854 = NULL;
    int64_t mem_144855_cached_sizze_146811 = 0;
    unsigned char *mem_144855 = NULL;
    int64_t mem_144856_cached_sizze_146812 = 0;
    unsigned char *mem_144856 = NULL;
    int64_t mem_144887_cached_sizze_146813 = 0;
    unsigned char *mem_144887 = NULL;
    int64_t mem_144888_cached_sizze_146814 = 0;
    unsigned char *mem_144888 = NULL;
    int64_t mem_144889_cached_sizze_146815 = 0;
    unsigned char *mem_144889 = NULL;
    int64_t mem_144890_cached_sizze_146816 = 0;
    unsigned char *mem_144890 = NULL;
    int64_t mem_144891_cached_sizze_146817 = 0;
    unsigned char *mem_144891 = NULL;
    int64_t mem_144892_cached_sizze_146818 = 0;
    unsigned char *mem_144892 = NULL;
    int64_t mem_144917_cached_sizze_146819 = 0;
    unsigned char *mem_144917 = NULL;
    int64_t mem_144918_cached_sizze_146820 = 0;
    unsigned char *mem_144918 = NULL;
    int64_t mem_144919_cached_sizze_146821 = 0;
    unsigned char *mem_144919 = NULL;
    int64_t mem_144938_cached_sizze_146822 = 0;
    unsigned char *mem_144938 = NULL;
    int64_t mem_144939_cached_sizze_146823 = 0;
    unsigned char *mem_144939 = NULL;
    int64_t mem_145007_cached_sizze_146824 = 0;
    unsigned char *mem_145007 = NULL;
    int64_t mem_145008_cached_sizze_146825 = 0;
    unsigned char *mem_145008 = NULL;
    int64_t mem_145009_cached_sizze_146826 = 0;
    unsigned char *mem_145009 = NULL;
    int64_t mem_145010_cached_sizze_146827 = 0;
    unsigned char *mem_145010 = NULL;
    int64_t mem_145011_cached_sizze_146828 = 0;
    unsigned char *mem_145011 = NULL;
    int64_t mem_145012_cached_sizze_146829 = 0;
    unsigned char *mem_145012 = NULL;
    int64_t mem_145013_cached_sizze_146830 = 0;
    unsigned char *mem_145013 = NULL;
    int64_t mem_145014_cached_sizze_146831 = 0;
    unsigned char *mem_145014 = NULL;
    int64_t mem_145015_cached_sizze_146832 = 0;
    unsigned char *mem_145015 = NULL;
    int64_t mem_145055_cached_sizze_146833 = 0;
    unsigned char *mem_145055 = NULL;
    int64_t mem_145056_cached_sizze_146834 = 0;
    unsigned char *mem_145056 = NULL;
    int64_t mem_145057_cached_sizze_146835 = 0;
    unsigned char *mem_145057 = NULL;
    int64_t mem_145058_cached_sizze_146836 = 0;
    unsigned char *mem_145058 = NULL;
    int64_t mem_145059_cached_sizze_146837 = 0;
    unsigned char *mem_145059 = NULL;
    int64_t mem_145060_cached_sizze_146838 = 0;
    unsigned char *mem_145060 = NULL;
    int64_t mem_145061_cached_sizze_146839 = 0;
    unsigned char *mem_145061 = NULL;
    int64_t mem_145062_cached_sizze_146840 = 0;
    unsigned char *mem_145062 = NULL;
    int64_t mem_145063_cached_sizze_146841 = 0;
    unsigned char *mem_145063 = NULL;
    int64_t mem_145094_cached_sizze_146842 = 0;
    unsigned char *mem_145094 = NULL;
    int64_t mem_145095_cached_sizze_146843 = 0;
    unsigned char *mem_145095 = NULL;
    int64_t mem_145108_cached_sizze_146844 = 0;
    unsigned char *mem_145108 = NULL;
    int64_t mem_145115_cached_sizze_146845 = 0;
    unsigned char *mem_145115 = NULL;
    int64_t mem_145122_cached_sizze_146846 = 0;
    unsigned char *mem_145122 = NULL;
    int64_t mem_145198_cached_sizze_146847 = 0;
    unsigned char *mem_145198 = NULL;
    int64_t mem_145199_cached_sizze_146848 = 0;
    unsigned char *mem_145199 = NULL;
    int64_t mem_145200_cached_sizze_146849 = 0;
    unsigned char *mem_145200 = NULL;
    int64_t mem_145201_cached_sizze_146850 = 0;
    unsigned char *mem_145201 = NULL;
    int64_t mem_145222_cached_sizze_146851 = 0;
    unsigned char *mem_145222 = NULL;
    int64_t mem_145223_cached_sizze_146852 = 0;
    unsigned char *mem_145223 = NULL;
    int64_t mem_145224_cached_sizze_146853 = 0;
    unsigned char *mem_145224 = NULL;
    int64_t mem_145225_cached_sizze_146854 = 0;
    unsigned char *mem_145225 = NULL;
    int64_t mem_145242_cached_sizze_146855 = 0;
    unsigned char *mem_145242 = NULL;
    int64_t mem_145243_cached_sizze_146856 = 0;
    unsigned char *mem_145243 = NULL;
    int64_t mem_145244_cached_sizze_146857 = 0;
    unsigned char *mem_145244 = NULL;
    int64_t mem_145245_cached_sizze_146858 = 0;
    unsigned char *mem_145245 = NULL;
    int64_t mem_145306_cached_sizze_146859 = 0;
    unsigned char *mem_145306 = NULL;
    int64_t mem_145307_cached_sizze_146860 = 0;
    unsigned char *mem_145307 = NULL;
    int64_t mem_145308_cached_sizze_146861 = 0;
    unsigned char *mem_145308 = NULL;
    int64_t mem_145309_cached_sizze_146862 = 0;
    unsigned char *mem_145309 = NULL;
    int64_t mem_145326_cached_sizze_146863 = 0;
    unsigned char *mem_145326 = NULL;
    int64_t mem_145327_cached_sizze_146864 = 0;
    unsigned char *mem_145327 = NULL;
    int64_t mem_145328_cached_sizze_146865 = 0;
    unsigned char *mem_145328 = NULL;
    int64_t mem_145329_cached_sizze_146866 = 0;
    unsigned char *mem_145329 = NULL;
    int64_t mem_145370_cached_sizze_146867 = 0;
    unsigned char *mem_145370 = NULL;
    int64_t mem_145371_cached_sizze_146868 = 0;
    unsigned char *mem_145371 = NULL;
    int64_t mem_145382_cached_sizze_146869 = 0;
    unsigned char *mem_145382 = NULL;
    int64_t mem_145383_cached_sizze_146870 = 0;
    unsigned char *mem_145383 = NULL;
    int64_t mem_145392_cached_sizze_146871 = 0;
    unsigned char *mem_145392 = NULL;
    int64_t mem_145393_cached_sizze_146872 = 0;
    unsigned char *mem_145393 = NULL;
    int64_t mem_145424_cached_sizze_146873 = 0;
    unsigned char *mem_145424 = NULL;
    int64_t mem_145425_cached_sizze_146874 = 0;
    unsigned char *mem_145425 = NULL;
    int64_t mem_145434_cached_sizze_146875 = 0;
    unsigned char *mem_145434 = NULL;
    int64_t mem_145435_cached_sizze_146876 = 0;
    unsigned char *mem_145435 = NULL;
    int64_t mem_145456_cached_sizze_146877 = 0;
    unsigned char *mem_145456 = NULL;
    int64_t mem_145457_cached_sizze_146878 = 0;
    unsigned char *mem_145457 = NULL;
    int64_t mem_145468_cached_sizze_146879 = 0;
    unsigned char *mem_145468 = NULL;
    int64_t mem_145469_cached_sizze_146880 = 0;
    unsigned char *mem_145469 = NULL;
    int64_t mem_145478_cached_sizze_146881 = 0;
    unsigned char *mem_145478 = NULL;
    int64_t mem_145479_cached_sizze_146882 = 0;
    unsigned char *mem_145479 = NULL;
    int64_t mem_145510_cached_sizze_146883 = 0;
    unsigned char *mem_145510 = NULL;
    int64_t mem_145511_cached_sizze_146884 = 0;
    unsigned char *mem_145511 = NULL;
    int64_t mem_145522_cached_sizze_146885 = 0;
    unsigned char *mem_145522 = NULL;
    int64_t mem_145523_cached_sizze_146886 = 0;
    unsigned char *mem_145523 = NULL;
    int64_t mem_145532_cached_sizze_146887 = 0;
    unsigned char *mem_145532 = NULL;
    int64_t mem_145533_cached_sizze_146888 = 0;
    unsigned char *mem_145533 = NULL;
    int64_t mem_145564_cached_sizze_146889 = 0;
    unsigned char *mem_145564 = NULL;
    int64_t mem_145565_cached_sizze_146890 = 0;
    unsigned char *mem_145565 = NULL;
    int64_t mem_145566_cached_sizze_146891 = 0;
    unsigned char *mem_145566 = NULL;
    int64_t mem_145567_cached_sizze_146892 = 0;
    unsigned char *mem_145567 = NULL;
    int64_t mem_145584_cached_sizze_146893 = 0;
    unsigned char *mem_145584 = NULL;
    int64_t mem_145585_cached_sizze_146894 = 0;
    unsigned char *mem_145585 = NULL;
    int64_t mem_145586_cached_sizze_146895 = 0;
    unsigned char *mem_145586 = NULL;
    int64_t mem_145587_cached_sizze_146896 = 0;
    unsigned char *mem_145587 = NULL;
    int64_t mem_145628_cached_sizze_146897 = 0;
    unsigned char *mem_145628 = NULL;
    int64_t mem_145633_cached_sizze_146898 = 0;
    unsigned char *mem_145633 = NULL;
    int64_t mem_145644_cached_sizze_146899 = 0;
    unsigned char *mem_145644 = NULL;
    int64_t mem_145645_cached_sizze_146900 = 0;
    unsigned char *mem_145645 = NULL;
    int64_t mem_145646_cached_sizze_146901 = 0;
    unsigned char *mem_145646 = NULL;
    int64_t mem_145647_cached_sizze_146902 = 0;
    unsigned char *mem_145647 = NULL;
    int64_t mem_145648_cached_sizze_146903 = 0;
    unsigned char *mem_145648 = NULL;
    int64_t mem_145667_cached_sizze_146904 = 0;
    unsigned char *mem_145667 = NULL;
    int64_t mem_145668_cached_sizze_146905 = 0;
    unsigned char *mem_145668 = NULL;
    int64_t mem_145669_cached_sizze_146906 = 0;
    unsigned char *mem_145669 = NULL;
    int64_t mem_145706_cached_sizze_146907 = 0;
    unsigned char *mem_145706 = NULL;
    int64_t mem_145713_cached_sizze_146908 = 0;
    unsigned char *mem_145713 = NULL;
    int64_t mem_145718_cached_sizze_146909 = 0;
    unsigned char *mem_145718 = NULL;
    int64_t mem_145729_cached_sizze_146910 = 0;
    unsigned char *mem_145729 = NULL;
    int64_t mem_145730_cached_sizze_146911 = 0;
    unsigned char *mem_145730 = NULL;
    int64_t mem_145739_cached_sizze_146912 = 0;
    unsigned char *mem_145739 = NULL;
    int64_t mem_145740_cached_sizze_146913 = 0;
    unsigned char *mem_145740 = NULL;
    int64_t mem_145761_cached_sizze_146914 = 0;
    unsigned char *mem_145761 = NULL;
    int64_t mem_145762_cached_sizze_146915 = 0;
    unsigned char *mem_145762 = NULL;
    int64_t mem_145763_cached_sizze_146916 = 0;
    unsigned char *mem_145763 = NULL;
    int64_t mem_145764_cached_sizze_146917 = 0;
    unsigned char *mem_145764 = NULL;
    int64_t mem_145789_cached_sizze_146918 = 0;
    unsigned char *mem_145789 = NULL;
    int64_t mem_145790_cached_sizze_146919 = 0;
    unsigned char *mem_145790 = NULL;
    int64_t mem_145803_cached_sizze_146920 = 0;
    unsigned char *mem_145803 = NULL;
    int64_t mem_145804_cached_sizze_146921 = 0;
    unsigned char *mem_145804 = NULL;
    int64_t mem_145813_cached_sizze_146922 = 0;
    unsigned char *mem_145813 = NULL;
    int64_t mem_145814_cached_sizze_146923 = 0;
    unsigned char *mem_145814 = NULL;
    int64_t mem_145835_cached_sizze_146924 = 0;
    unsigned char *mem_145835 = NULL;
    int64_t mem_145840_cached_sizze_146925 = 0;
    unsigned char *mem_145840 = NULL;
    int64_t mem_145851_cached_sizze_146926 = 0;
    unsigned char *mem_145851 = NULL;
    int64_t mem_145852_cached_sizze_146927 = 0;
    unsigned char *mem_145852 = NULL;
    int64_t mem_145861_cached_sizze_146928 = 0;
    unsigned char *mem_145861 = NULL;
    int64_t mem_145862_cached_sizze_146929 = 0;
    unsigned char *mem_145862 = NULL;
    struct memblock mem_param_tmp_146215;
    
    mem_param_tmp_146215.references = NULL;
    
    struct memblock mem_param_tmp_146214;
    
    mem_param_tmp_146214.references = NULL;
    
    struct memblock mem_param_tmp_146213;
    
    mem_param_tmp_146213.references = NULL;
    
    struct memblock mem_param_tmp_146212;
    
    mem_param_tmp_146212.references = NULL;
    
    struct memblock mem_param_tmp_146211;
    
    mem_param_tmp_146211.references = NULL;
    
    struct memblock mem_param_tmp_146210;
    
    mem_param_tmp_146210.references = NULL;
    
    struct memblock mem_param_tmp_146209;
    
    mem_param_tmp_146209.references = NULL;
    
    struct memblock mem_param_tmp_146208;
    
    mem_param_tmp_146208.references = NULL;
    
    struct memblock mem_param_tmp_146207;
    
    mem_param_tmp_146207.references = NULL;
    
    struct memblock mem_param_tmp_146206;
    
    mem_param_tmp_146206.references = NULL;
    
    struct memblock mem_param_tmp_146205;
    
    mem_param_tmp_146205.references = NULL;
    
    struct memblock mem_param_tmp_146204;
    
    mem_param_tmp_146204.references = NULL;
    
    struct memblock mem_param_tmp_146203;
    
    mem_param_tmp_146203.references = NULL;
    
    struct memblock mem_param_tmp_146202;
    
    mem_param_tmp_146202.references = NULL;
    
    struct memblock mem_param_tmp_146201;
    
    mem_param_tmp_146201.references = NULL;
    
    struct memblock mem_param_tmp_146200;
    
    mem_param_tmp_146200.references = NULL;
    
    struct memblock mem_param_tmp_146199;
    
    mem_param_tmp_146199.references = NULL;
    
    struct memblock mem_param_tmp_146198;
    
    mem_param_tmp_146198.references = NULL;
    
    struct memblock mem_param_tmp_146197;
    
    mem_param_tmp_146197.references = NULL;
    
    struct memblock mem_param_tmp_146196;
    
    mem_param_tmp_146196.references = NULL;
    
    struct memblock mem_param_tmp_146195;
    
    mem_param_tmp_146195.references = NULL;
    
    struct memblock mem_param_tmp_146194;
    
    mem_param_tmp_146194.references = NULL;
    
    struct memblock mem_param_tmp_146193;
    
    mem_param_tmp_146193.references = NULL;
    
    struct memblock mem_param_tmp_146192;
    
    mem_param_tmp_146192.references = NULL;
    
    struct memblock mem_param_tmp_146191;
    
    mem_param_tmp_146191.references = NULL;
    
    struct memblock mem_param_tmp_146190;
    
    mem_param_tmp_146190.references = NULL;
    
    struct memblock mem_param_tmp_146189;
    
    mem_param_tmp_146189.references = NULL;
    
    struct memblock ext_mem_145979;
    
    ext_mem_145979.references = NULL;
    
    struct memblock ext_mem_145980;
    
    ext_mem_145980.references = NULL;
    
    struct memblock ext_mem_145981;
    
    ext_mem_145981.references = NULL;
    
    struct memblock mem_145977;
    
    mem_145977.references = NULL;
    
    struct memblock mem_145975;
    
    mem_145975.references = NULL;
    
    struct memblock mem_145973;
    
    mem_145973.references = NULL;
    
    struct memblock mem_145971;
    
    mem_145971.references = NULL;
    
    struct memblock ext_mem_145968;
    
    ext_mem_145968.references = NULL;
    
    struct memblock ext_mem_145969;
    
    ext_mem_145969.references = NULL;
    
    struct memblock ext_mem_145970;
    
    ext_mem_145970.references = NULL;
    
    struct memblock mem_145966;
    
    mem_145966.references = NULL;
    
    struct memblock mem_145964;
    
    mem_145964.references = NULL;
    
    struct memblock mem_145962;
    
    mem_145962.references = NULL;
    
    struct memblock mem_145960;
    
    mem_145960.references = NULL;
    
    struct memblock ext_mem_145957;
    
    ext_mem_145957.references = NULL;
    
    struct memblock ext_mem_145958;
    
    ext_mem_145958.references = NULL;
    
    struct memblock ext_mem_145959;
    
    ext_mem_145959.references = NULL;
    
    struct memblock mem_145955;
    
    mem_145955.references = NULL;
    
    struct memblock mem_145953;
    
    mem_145953.references = NULL;
    
    struct memblock mem_145951;
    
    mem_145951.references = NULL;
    
    struct memblock mem_145949;
    
    mem_145949.references = NULL;
    
    struct memblock ext_mem_145946;
    
    ext_mem_145946.references = NULL;
    
    struct memblock ext_mem_145947;
    
    ext_mem_145947.references = NULL;
    
    struct memblock ext_mem_145948;
    
    ext_mem_145948.references = NULL;
    
    struct memblock mem_145944;
    
    mem_145944.references = NULL;
    
    struct memblock mem_145942;
    
    mem_145942.references = NULL;
    
    struct memblock mem_145940;
    
    mem_145940.references = NULL;
    
    struct memblock mem_145938;
    
    mem_145938.references = NULL;
    
    struct memblock ext_mem_145935;
    
    ext_mem_145935.references = NULL;
    
    struct memblock ext_mem_145936;
    
    ext_mem_145936.references = NULL;
    
    struct memblock ext_mem_145937;
    
    ext_mem_145937.references = NULL;
    
    struct memblock mem_145933;
    
    mem_145933.references = NULL;
    
    struct memblock mem_145931;
    
    mem_145931.references = NULL;
    
    struct memblock mem_145929;
    
    mem_145929.references = NULL;
    
    struct memblock mem_145927;
    
    mem_145927.references = NULL;
    
    struct memblock ext_mem_145924;
    
    ext_mem_145924.references = NULL;
    
    struct memblock ext_mem_145925;
    
    ext_mem_145925.references = NULL;
    
    struct memblock ext_mem_145926;
    
    ext_mem_145926.references = NULL;
    
    struct memblock mem_145922;
    
    mem_145922.references = NULL;
    
    struct memblock mem_145920;
    
    mem_145920.references = NULL;
    
    struct memblock mem_145918;
    
    mem_145918.references = NULL;
    
    struct memblock mem_145916;
    
    mem_145916.references = NULL;
    
    struct memblock ext_mem_145913;
    
    ext_mem_145913.references = NULL;
    
    struct memblock ext_mem_145914;
    
    ext_mem_145914.references = NULL;
    
    struct memblock ext_mem_145915;
    
    ext_mem_145915.references = NULL;
    
    struct memblock mem_145911;
    
    mem_145911.references = NULL;
    
    struct memblock mem_145909;
    
    mem_145909.references = NULL;
    
    struct memblock mem_145907;
    
    mem_145907.references = NULL;
    
    struct memblock mem_145905;
    
    mem_145905.references = NULL;
    
    struct memblock ext_mem_145902;
    
    ext_mem_145902.references = NULL;
    
    struct memblock ext_mem_145903;
    
    ext_mem_145903.references = NULL;
    
    struct memblock ext_mem_145904;
    
    ext_mem_145904.references = NULL;
    
    struct memblock mem_145900;
    
    mem_145900.references = NULL;
    
    struct memblock mem_145898;
    
    mem_145898.references = NULL;
    
    struct memblock mem_145896;
    
    mem_145896.references = NULL;
    
    struct memblock mem_145894;
    
    mem_145894.references = NULL;
    
    struct memblock ext_mem_145891;
    
    ext_mem_145891.references = NULL;
    
    struct memblock ext_mem_145892;
    
    ext_mem_145892.references = NULL;
    
    struct memblock ext_mem_145893;
    
    ext_mem_145893.references = NULL;
    
    struct memblock mem_145889;
    
    mem_145889.references = NULL;
    
    struct memblock mem_145887;
    
    mem_145887.references = NULL;
    
    struct memblock mem_145885;
    
    mem_145885.references = NULL;
    
    struct memblock mem_145883;
    
    mem_145883.references = NULL;
    
    struct memblock mem_param_144068;
    
    mem_param_144068.references = NULL;
    
    struct memblock mem_param_144064;
    
    mem_param_144064.references = NULL;
    
    struct memblock mem_param_144060;
    
    mem_param_144060.references = NULL;
    
    struct memblock mem_param_144056;
    
    mem_param_144056.references = NULL;
    
    struct memblock mem_param_144052;
    
    mem_param_144052.references = NULL;
    
    struct memblock mem_param_144048;
    
    mem_param_144048.references = NULL;
    
    struct memblock mem_param_144044;
    
    mem_param_144044.references = NULL;
    
    struct memblock mem_param_144040;
    
    mem_param_144040.references = NULL;
    
    struct memblock mem_param_144036;
    
    mem_param_144036.references = NULL;
    
    struct memblock mem_param_144032;
    
    mem_param_144032.references = NULL;
    
    struct memblock mem_param_144028;
    
    mem_param_144028.references = NULL;
    
    struct memblock mem_param_144024;
    
    mem_param_144024.references = NULL;
    
    struct memblock mem_param_144020;
    
    mem_param_144020.references = NULL;
    
    struct memblock mem_param_144016;
    
    mem_param_144016.references = NULL;
    
    struct memblock mem_param_144012;
    
    mem_param_144012.references = NULL;
    
    struct memblock mem_param_144008;
    
    mem_param_144008.references = NULL;
    
    struct memblock mem_param_144004;
    
    mem_param_144004.references = NULL;
    
    struct memblock mem_param_144000;
    
    mem_param_144000.references = NULL;
    
    struct memblock mem_param_143996;
    
    mem_param_143996.references = NULL;
    
    struct memblock mem_param_143992;
    
    mem_param_143992.references = NULL;
    
    struct memblock mem_param_143988;
    
    mem_param_143988.references = NULL;
    
    struct memblock mem_param_143984;
    
    mem_param_143984.references = NULL;
    
    struct memblock mem_param_143980;
    
    mem_param_143980.references = NULL;
    
    struct memblock mem_param_143976;
    
    mem_param_143976.references = NULL;
    
    struct memblock mem_param_143972;
    
    mem_param_143972.references = NULL;
    
    struct memblock mem_param_143968;
    
    mem_param_143968.references = NULL;
    
    struct memblock mem_param_143964;
    
    mem_param_143964.references = NULL;
    
    struct memblock ext_mem_146063;
    
    ext_mem_146063.references = NULL;
    
    struct memblock ext_mem_146064;
    
    ext_mem_146064.references = NULL;
    
    struct memblock ext_mem_146065;
    
    ext_mem_146065.references = NULL;
    
    struct memblock ext_mem_146066;
    
    ext_mem_146066.references = NULL;
    
    struct memblock ext_mem_146067;
    
    ext_mem_146067.references = NULL;
    
    struct memblock ext_mem_146068;
    
    ext_mem_146068.references = NULL;
    
    struct memblock ext_mem_146069;
    
    ext_mem_146069.references = NULL;
    
    struct memblock ext_mem_146070;
    
    ext_mem_146070.references = NULL;
    
    struct memblock ext_mem_146071;
    
    ext_mem_146071.references = NULL;
    
    struct memblock ext_mem_146072;
    
    ext_mem_146072.references = NULL;
    
    struct memblock ext_mem_146073;
    
    ext_mem_146073.references = NULL;
    
    struct memblock ext_mem_146074;
    
    ext_mem_146074.references = NULL;
    
    struct memblock ext_mem_146075;
    
    ext_mem_146075.references = NULL;
    
    struct memblock ext_mem_146076;
    
    ext_mem_146076.references = NULL;
    
    struct memblock ext_mem_146077;
    
    ext_mem_146077.references = NULL;
    
    struct memblock ext_mem_146078;
    
    ext_mem_146078.references = NULL;
    
    struct memblock ext_mem_146079;
    
    ext_mem_146079.references = NULL;
    
    struct memblock ext_mem_146080;
    
    ext_mem_146080.references = NULL;
    
    struct memblock ext_mem_146081;
    
    ext_mem_146081.references = NULL;
    
    struct memblock ext_mem_146082;
    
    ext_mem_146082.references = NULL;
    
    struct memblock ext_mem_146083;
    
    ext_mem_146083.references = NULL;
    
    struct memblock ext_mem_146084;
    
    ext_mem_146084.references = NULL;
    
    struct memblock ext_mem_146085;
    
    ext_mem_146085.references = NULL;
    
    struct memblock ext_mem_146086;
    
    ext_mem_146086.references = NULL;
    
    struct memblock ext_mem_146087;
    
    ext_mem_146087.references = NULL;
    
    struct memblock ext_mem_146088;
    
    ext_mem_146088.references = NULL;
    
    struct memblock ext_mem_146089;
    
    ext_mem_146089.references = NULL;
    
    struct memblock mem_out_146188;
    
    mem_out_146188.references = NULL;
    
    struct memblock mem_out_146187;
    
    mem_out_146187.references = NULL;
    
    struct memblock mem_out_146186;
    
    mem_out_146186.references = NULL;
    
    struct memblock mem_out_146185;
    
    mem_out_146185.references = NULL;
    
    struct memblock mem_out_146184;
    
    mem_out_146184.references = NULL;
    
    struct memblock mem_out_146183;
    
    mem_out_146183.references = NULL;
    
    struct memblock mem_out_146182;
    
    mem_out_146182.references = NULL;
    
    struct memblock mem_out_146181;
    
    mem_out_146181.references = NULL;
    
    struct memblock mem_out_146180;
    
    mem_out_146180.references = NULL;
    
    struct memblock mem_out_146179;
    
    mem_out_146179.references = NULL;
    
    struct memblock mem_out_146178;
    
    mem_out_146178.references = NULL;
    
    struct memblock mem_out_146177;
    
    mem_out_146177.references = NULL;
    
    struct memblock mem_out_146176;
    
    mem_out_146176.references = NULL;
    
    struct memblock mem_out_146175;
    
    mem_out_146175.references = NULL;
    
    struct memblock mem_out_146174;
    
    mem_out_146174.references = NULL;
    
    struct memblock mem_out_146173;
    
    mem_out_146173.references = NULL;
    
    struct memblock mem_out_146172;
    
    mem_out_146172.references = NULL;
    
    struct memblock mem_out_146171;
    
    mem_out_146171.references = NULL;
    
    struct memblock mem_out_146170;
    
    mem_out_146170.references = NULL;
    
    struct memblock mem_out_146169;
    
    mem_out_146169.references = NULL;
    
    struct memblock mem_out_146168;
    
    mem_out_146168.references = NULL;
    
    struct memblock mem_out_146167;
    
    mem_out_146167.references = NULL;
    
    struct memblock mem_out_146166;
    
    mem_out_146166.references = NULL;
    
    struct memblock mem_out_146165;
    
    mem_out_146165.references = NULL;
    
    struct memblock mem_out_146164;
    
    mem_out_146164.references = NULL;
    
    struct memblock mem_out_146163;
    
    mem_out_146163.references = NULL;
    
    struct memblock mem_out_146162;
    
    mem_out_146162.references = NULL;
    
    struct memblock mem_143922 = ctx->constants->mem_143922;
    struct memblock mem_143923 = ctx->constants->mem_143923;
    struct memblock mem_143924 = ctx->constants->mem_143924;
    struct memblock mem_143925 = ctx->constants->mem_143925;
    struct memblock mem_143926 = ctx->constants->mem_143926;
    struct memblock mem_143927 = ctx->constants->mem_143927;
    struct memblock mem_143928 = ctx->constants->mem_143928;
    struct memblock mem_143929 = ctx->constants->mem_143929;
    struct memblock mem_143930 = ctx->constants->mem_143930;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_144069_cached_sizze_146712 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144069, &mem_144069_cached_sizze_146712, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144070_cached_sizze_146713 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144070, &mem_144070_cached_sizze_146713, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144079_cached_sizze_146714 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144079, &mem_144079_cached_sizze_146714, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144086_cached_sizze_146715 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144086, &mem_144086_cached_sizze_146715, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144101_cached_sizze_146716 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144101, &mem_144101_cached_sizze_146716, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144102_cached_sizze_146717 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144102, &mem_144102_cached_sizze_146717, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144103_cached_sizze_146718 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144103, &mem_144103_cached_sizze_146718, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144114_cached_sizze_146719 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144114, &mem_144114_cached_sizze_146719, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144121_cached_sizze_146720 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144121, &mem_144121_cached_sizze_146720, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144138_cached_sizze_146721 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144138, &mem_144138_cached_sizze_146721, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144139_cached_sizze_146722 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144139, &mem_144139_cached_sizze_146722, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144147_cached_sizze_146723 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144147, &mem_144147_cached_sizze_146723, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144154_cached_sizze_146724 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144154, &mem_144154_cached_sizze_146724, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144168_cached_sizze_146725 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144168, &mem_144168_cached_sizze_146725, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144169_cached_sizze_146726 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144169, &mem_144169_cached_sizze_146726, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144170_cached_sizze_146727 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144170, &mem_144170_cached_sizze_146727, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144186_cached_sizze_146728 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144186, &mem_144186_cached_sizze_146728, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144187_cached_sizze_146729 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144187, &mem_144187_cached_sizze_146729, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144188_cached_sizze_146730 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144188, &mem_144188_cached_sizze_146730, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144201_cached_sizze_146731 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144201, &mem_144201_cached_sizze_146731, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144202_cached_sizze_146732 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144202, &mem_144202_cached_sizze_146732, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144203_cached_sizze_146733 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144203, &mem_144203_cached_sizze_146733, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144249_cached_sizze_146734 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144249, &mem_144249_cached_sizze_146734, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144254_cached_sizze_146735 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144254, &mem_144254_cached_sizze_146735, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144258_cached_sizze_146736 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144258, &mem_144258_cached_sizze_146736, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144263_cached_sizze_146737 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144263, &mem_144263_cached_sizze_146737, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144274_cached_sizze_146738 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144274, &mem_144274_cached_sizze_146738, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144279_cached_sizze_146739 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144279, &mem_144279_cached_sizze_146739, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144290_cached_sizze_146740 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144290, &mem_144290_cached_sizze_146740, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144295_cached_sizze_146741 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144295, &mem_144295_cached_sizze_146741, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144302_cached_sizze_146742 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144302, &mem_144302_cached_sizze_146742, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144309_cached_sizze_146743 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144309, &mem_144309_cached_sizze_146743, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144320_cached_sizze_146744 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144320, &mem_144320_cached_sizze_146744, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144325_cached_sizze_146745 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144325, &mem_144325_cached_sizze_146745, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144343_cached_sizze_146746 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144343, &mem_144343_cached_sizze_146746, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144348_cached_sizze_146747 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144348, &mem_144348_cached_sizze_146747, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144359_cached_sizze_146748 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144359, &mem_144359_cached_sizze_146748, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144360_cached_sizze_146749 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144360, &mem_144360_cached_sizze_146749, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144368_cached_sizze_146750 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144368, &mem_144368_cached_sizze_146750, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144375_cached_sizze_146751 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144375, &mem_144375_cached_sizze_146751, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144389_cached_sizze_146752 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144389, &mem_144389_cached_sizze_146752, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144394_cached_sizze_146753 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144394, &mem_144394_cached_sizze_146753, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144405_cached_sizze_146754 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144405, &mem_144405_cached_sizze_146754, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144410_cached_sizze_146755 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144410, &mem_144410_cached_sizze_146755, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144421_cached_sizze_146756 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144421, &mem_144421_cached_sizze_146756, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144426_cached_sizze_146757 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144426, &mem_144426_cached_sizze_146757, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144437_cached_sizze_146758 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144437, &mem_144437_cached_sizze_146758, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144442_cached_sizze_146759 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144442, &mem_144442_cached_sizze_146759, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144453_cached_sizze_146760 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144453, &mem_144453_cached_sizze_146760, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144454_cached_sizze_146761 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144454, &mem_144454_cached_sizze_146761, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144455_cached_sizze_146762 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_144455, &mem_144455_cached_sizze_146762, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144456_cached_sizze_146763 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144456, &mem_144456_cached_sizze_146763, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:115:13-33
    if (mem_144474_cached_sizze_146764 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_144474, &mem_144474_cached_sizze_146764, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144479_cached_sizze_146765 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144479, &mem_144479_cached_sizze_146765, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144524_cached_sizze_146768 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_144524, &mem_144524_cached_sizze_146768, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144530_cached_sizze_146769 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_144530, &mem_144530_cached_sizze_146769, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144535_cached_sizze_146770 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144535, &mem_144535_cached_sizze_146770, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144551_cached_sizze_146771 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144551, &mem_144551_cached_sizze_146771, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144552_cached_sizze_146772 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144552, &mem_144552_cached_sizze_146772, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144561_cached_sizze_146773 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144561, &mem_144561_cached_sizze_146773, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144562_cached_sizze_146774 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144562, &mem_144562_cached_sizze_146774, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144583_cached_sizze_146775 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_144583, &mem_144583_cached_sizze_146775, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144589_cached_sizze_146776 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_144589, &mem_144589_cached_sizze_146776, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144594_cached_sizze_146777 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144594, &mem_144594_cached_sizze_146777, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144610_cached_sizze_146778 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144610, &mem_144610_cached_sizze_146778, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144615_cached_sizze_146779 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144615, &mem_144615_cached_sizze_146779, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144626_cached_sizze_146780 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_144626, &mem_144626_cached_sizze_146780, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144631_cached_sizze_146781 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_144631, &mem_144631_cached_sizze_146781, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144642_cached_sizze_146782 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144642, &mem_144642_cached_sizze_146782, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144647_cached_sizze_146783 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144647, &mem_144647_cached_sizze_146783, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144658_cached_sizze_146784 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144658, &mem_144658_cached_sizze_146784, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144659_cached_sizze_146785 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144659, &mem_144659_cached_sizze_146785, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144668_cached_sizze_146786 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144668, &mem_144668_cached_sizze_146786, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144669_cached_sizze_146787 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144669, &mem_144669_cached_sizze_146787, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144690_cached_sizze_146788 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144690, &mem_144690_cached_sizze_146788, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144695_cached_sizze_146789 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144695, &mem_144695_cached_sizze_146789, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144706_cached_sizze_146790 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144706, &mem_144706_cached_sizze_146790, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144707_cached_sizze_146791 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144707, &mem_144707_cached_sizze_146791, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144720_cached_sizze_146792 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144720, &mem_144720_cached_sizze_146792, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144727_cached_sizze_146793 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144727, &mem_144727_cached_sizze_146793, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144732_cached_sizze_146794 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144732, &mem_144732_cached_sizze_146794, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144743_cached_sizze_146795 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144743, &mem_144743_cached_sizze_146795, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144744_cached_sizze_146796 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144744, &mem_144744_cached_sizze_146796, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144745_cached_sizze_146797 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144745, &mem_144745_cached_sizze_146797, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144746_cached_sizze_146798 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144746, &mem_144746_cached_sizze_146798, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144767_cached_sizze_146799 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144767, &mem_144767_cached_sizze_146799, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144768_cached_sizze_146800 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144768, &mem_144768_cached_sizze_146800, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144769_cached_sizze_146801 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144769, &mem_144769_cached_sizze_146801, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144770_cached_sizze_146802 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144770, &mem_144770_cached_sizze_146802, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144787_cached_sizze_146803 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144787, &mem_144787_cached_sizze_146803, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144794_cached_sizze_146804 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144794, &mem_144794_cached_sizze_146804, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144795_cached_sizze_146805 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144795, &mem_144795_cached_sizze_146805, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144796_cached_sizze_146806 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144796, &mem_144796_cached_sizze_146806, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144851_cached_sizze_146807 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144851, &mem_144851_cached_sizze_146807, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144852_cached_sizze_146808 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144852, &mem_144852_cached_sizze_146808, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144853_cached_sizze_146809 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144853, &mem_144853_cached_sizze_146809, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144854_cached_sizze_146810 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144854, &mem_144854_cached_sizze_146810, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144855_cached_sizze_146811 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144855, &mem_144855_cached_sizze_146811, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144856_cached_sizze_146812 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_144856, &mem_144856_cached_sizze_146812, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144887_cached_sizze_146813 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144887, &mem_144887_cached_sizze_146813, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144888_cached_sizze_146814 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144888, &mem_144888_cached_sizze_146814, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144889_cached_sizze_146815 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144889, &mem_144889_cached_sizze_146815, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144890_cached_sizze_146816 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144890, &mem_144890_cached_sizze_146816, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144891_cached_sizze_146817 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_144891, &mem_144891_cached_sizze_146817, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144892_cached_sizze_146818 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_144892, &mem_144892_cached_sizze_146818, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144917_cached_sizze_146819 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144917, &mem_144917_cached_sizze_146819, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144918_cached_sizze_146820 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144918, &mem_144918_cached_sizze_146820, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144919_cached_sizze_146821 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_144919, &mem_144919_cached_sizze_146821, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144938_cached_sizze_146822 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144938, &mem_144938_cached_sizze_146822, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_144939_cached_sizze_146823 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_144939, &mem_144939_cached_sizze_146823, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145007_cached_sizze_146824 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_145007, &mem_145007_cached_sizze_146824, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145008_cached_sizze_146825 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_145008, &mem_145008_cached_sizze_146825, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145009_cached_sizze_146826 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_145009, &mem_145009_cached_sizze_146826, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145010_cached_sizze_146827 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_145010, &mem_145010_cached_sizze_146827, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145011_cached_sizze_146828 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_145011, &mem_145011_cached_sizze_146828, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145012_cached_sizze_146829 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_145012, &mem_145012_cached_sizze_146829, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145013_cached_sizze_146830 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_145013, &mem_145013_cached_sizze_146830, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145014_cached_sizze_146831 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_145014, &mem_145014_cached_sizze_146831, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145015_cached_sizze_146832 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_145015, &mem_145015_cached_sizze_146832, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145055_cached_sizze_146833 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145055, &mem_145055_cached_sizze_146833, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145056_cached_sizze_146834 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145056, &mem_145056_cached_sizze_146834, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145057_cached_sizze_146835 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145057, &mem_145057_cached_sizze_146835, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145058_cached_sizze_146836 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145058, &mem_145058_cached_sizze_146836, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145059_cached_sizze_146837 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145059, &mem_145059_cached_sizze_146837, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145060_cached_sizze_146838 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145060, &mem_145060_cached_sizze_146838, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145061_cached_sizze_146839 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145061, &mem_145061_cached_sizze_146839, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145062_cached_sizze_146840 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145062, &mem_145062_cached_sizze_146840, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145063_cached_sizze_146841 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145063, &mem_145063_cached_sizze_146841, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:115:13-33
    if (mem_145094_cached_sizze_146842 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145094, &mem_145094_cached_sizze_146842, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:115:13-33
    if (mem_145095_cached_sizze_146843 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145095, &mem_145095_cached_sizze_146843, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145108_cached_sizze_146844 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145108, &mem_145108_cached_sizze_146844, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145115_cached_sizze_146845 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145115, &mem_145115_cached_sizze_146845, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145122_cached_sizze_146846 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145122, &mem_145122_cached_sizze_146846, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145198_cached_sizze_146847 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_145198, &mem_145198_cached_sizze_146847, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145199_cached_sizze_146848 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_145199, &mem_145199_cached_sizze_146848, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145200_cached_sizze_146849 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_145200, &mem_145200_cached_sizze_146849, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145201_cached_sizze_146850 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_145201, &mem_145201_cached_sizze_146850, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145222_cached_sizze_146851 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145222, &mem_145222_cached_sizze_146851, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145223_cached_sizze_146852 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145223, &mem_145223_cached_sizze_146852, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145224_cached_sizze_146853 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145224, &mem_145224_cached_sizze_146853, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145225_cached_sizze_146854 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145225, &mem_145225_cached_sizze_146854, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145242_cached_sizze_146855 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145242, &mem_145242_cached_sizze_146855, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145243_cached_sizze_146856 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145243, &mem_145243_cached_sizze_146856, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145244_cached_sizze_146857 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145244, &mem_145244_cached_sizze_146857, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145245_cached_sizze_146858 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145245, &mem_145245_cached_sizze_146858, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145306_cached_sizze_146859 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_145306, &mem_145306_cached_sizze_146859, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145307_cached_sizze_146860 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_145307, &mem_145307_cached_sizze_146860, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145308_cached_sizze_146861 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_145308, &mem_145308_cached_sizze_146861, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145309_cached_sizze_146862 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_145309, &mem_145309_cached_sizze_146862, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145326_cached_sizze_146863 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145326, &mem_145326_cached_sizze_146863, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145327_cached_sizze_146864 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145327, &mem_145327_cached_sizze_146864, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145328_cached_sizze_146865 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145328, &mem_145328_cached_sizze_146865, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145329_cached_sizze_146866 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145329, &mem_145329_cached_sizze_146866, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145370_cached_sizze_146867 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_145370, &mem_145370_cached_sizze_146867, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145371_cached_sizze_146868 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_145371, &mem_145371_cached_sizze_146868, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145382_cached_sizze_146869 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145382, &mem_145382_cached_sizze_146869, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145383_cached_sizze_146870 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145383, &mem_145383_cached_sizze_146870, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145392_cached_sizze_146871 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145392, &mem_145392_cached_sizze_146871, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145393_cached_sizze_146872 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145393, &mem_145393_cached_sizze_146872, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145424_cached_sizze_146873 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_145424, &mem_145424_cached_sizze_146873, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145425_cached_sizze_146874 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_145425, &mem_145425_cached_sizze_146874, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145434_cached_sizze_146875 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145434, &mem_145434_cached_sizze_146875, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145435_cached_sizze_146876 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145435, &mem_145435_cached_sizze_146876, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145456_cached_sizze_146877 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_145456, &mem_145456_cached_sizze_146877, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145457_cached_sizze_146878 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_145457, &mem_145457_cached_sizze_146878, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145468_cached_sizze_146879 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145468, &mem_145468_cached_sizze_146879, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145469_cached_sizze_146880 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145469, &mem_145469_cached_sizze_146880, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145478_cached_sizze_146881 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145478, &mem_145478_cached_sizze_146881, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145479_cached_sizze_146882 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145479, &mem_145479_cached_sizze_146882, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145510_cached_sizze_146883 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_145510, &mem_145510_cached_sizze_146883, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145511_cached_sizze_146884 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_145511, &mem_145511_cached_sizze_146884, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145522_cached_sizze_146885 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145522, &mem_145522_cached_sizze_146885, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145523_cached_sizze_146886 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145523, &mem_145523_cached_sizze_146886, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145532_cached_sizze_146887 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145532, &mem_145532_cached_sizze_146887, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145533_cached_sizze_146888 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145533, &mem_145533_cached_sizze_146888, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145564_cached_sizze_146889 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145564, &mem_145564_cached_sizze_146889, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145565_cached_sizze_146890 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145565, &mem_145565_cached_sizze_146890, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145566_cached_sizze_146891 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145566, &mem_145566_cached_sizze_146891, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145567_cached_sizze_146892 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145567, &mem_145567_cached_sizze_146892, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145584_cached_sizze_146893 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145584, &mem_145584_cached_sizze_146893, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145585_cached_sizze_146894 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145585, &mem_145585_cached_sizze_146894, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145586_cached_sizze_146895 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145586, &mem_145586_cached_sizze_146895, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145587_cached_sizze_146896 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145587, &mem_145587_cached_sizze_146896, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145628_cached_sizze_146897 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145628, &mem_145628_cached_sizze_146897, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145633_cached_sizze_146898 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145633, &mem_145633_cached_sizze_146898, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145644_cached_sizze_146899 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145644, &mem_145644_cached_sizze_146899, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145645_cached_sizze_146900 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145645, &mem_145645_cached_sizze_146900, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145646_cached_sizze_146901 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145646, &mem_145646_cached_sizze_146901, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145647_cached_sizze_146902 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145647, &mem_145647_cached_sizze_146902, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145648_cached_sizze_146903 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145648, &mem_145648_cached_sizze_146903, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145667_cached_sizze_146904 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145667, &mem_145667_cached_sizze_146904, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145668_cached_sizze_146905 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145668, &mem_145668_cached_sizze_146905, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145669_cached_sizze_146906 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145669, &mem_145669_cached_sizze_146906, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145706_cached_sizze_146907 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145706, &mem_145706_cached_sizze_146907, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145713_cached_sizze_146908 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145713, &mem_145713_cached_sizze_146908, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145718_cached_sizze_146909 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145718, &mem_145718_cached_sizze_146909, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145729_cached_sizze_146910 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145729, &mem_145729_cached_sizze_146910, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145730_cached_sizze_146911 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145730, &mem_145730_cached_sizze_146911, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145739_cached_sizze_146912 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145739, &mem_145739_cached_sizze_146912, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145740_cached_sizze_146913 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145740, &mem_145740_cached_sizze_146913, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145761_cached_sizze_146914 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145761, &mem_145761_cached_sizze_146914, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145762_cached_sizze_146915 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145762, &mem_145762_cached_sizze_146915, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145763_cached_sizze_146916 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145763, &mem_145763_cached_sizze_146916, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145764_cached_sizze_146917 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145764, &mem_145764_cached_sizze_146917, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145789_cached_sizze_146918 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145789, &mem_145789_cached_sizze_146918, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145790_cached_sizze_146919 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145790, &mem_145790_cached_sizze_146919, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145803_cached_sizze_146920 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145803, &mem_145803_cached_sizze_146920, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145804_cached_sizze_146921 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_145804, &mem_145804_cached_sizze_146921, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145813_cached_sizze_146922 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145813, &mem_145813_cached_sizze_146922, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145814_cached_sizze_146923 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145814, &mem_145814_cached_sizze_146923, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145835_cached_sizze_146924 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_145835, &mem_145835_cached_sizze_146924, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145840_cached_sizze_146925 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145840, &mem_145840_cached_sizze_146925, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145851_cached_sizze_146926 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_145851, &mem_145851_cached_sizze_146926, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145852_cached_sizze_146927 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_145852, &mem_145852_cached_sizze_146927, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145861_cached_sizze_146928 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145861, &mem_145861_cached_sizze_146928, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145862_cached_sizze_146929 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_145862, &mem_145862_cached_sizze_146929, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:628:5-633:51
    if (memblock_set(ctx, &mem_param_143964, &wdown_mem_143931, "wdown_mem_143931") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_143968, &wkey_mem_143932, "wkey_mem_143932") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_143972, &wout_mem_143933, "wout_mem_143933") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_143976, &wpe_mem_143934, "wpe_mem_143934") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_143980, &wqry_mem_143935, "wqry_mem_143935") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_143984, &wte_mem_143936, "wte_mem_143936") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_143988, &wup_mem_143937, "wup_mem_143937") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_143992, &wval_mem_143938, "wval_mem_143938") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_143996, &wvoc_mem_143939, "wvoc_mem_143939") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_144000, &wdown_mem_143940, "wdown_mem_143940") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_144004, &wkey_mem_143941, "wkey_mem_143941") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_144008, &wout_mem_143942, "wout_mem_143942") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_144012, &wpe_mem_143943, "wpe_mem_143943") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_144016, &wqry_mem_143944, "wqry_mem_143944") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_144020, &wte_mem_143945, "wte_mem_143945") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_144024, &wup_mem_143946, "wup_mem_143946") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_144028, &wval_mem_143947, "wval_mem_143947") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_144032, &wvoc_mem_143948, "wvoc_mem_143948") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_144036, &wdown_mem_143949, "wdown_mem_143949") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_144040, &wkey_mem_143950, "wkey_mem_143950") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_144044, &wout_mem_143951, "wout_mem_143951") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_144048, &wpe_mem_143952, "wpe_mem_143952") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_144052, &wqry_mem_143953, "wqry_mem_143953") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_144056, &wte_mem_143954, "wte_mem_143954") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_144060, &wup_mem_143955, "wup_mem_143955") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_144064, &wval_mem_143956, "wval_mem_143956") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_144068, &wvoc_mem_143957, "wvoc_mem_143957") != 0)
        return 1;
    for (int64_t step_129423 = 0; step_129423 < (int64_t) 500; step_129423++) {
        // futhark/microgpt.fut:630:16-25
        
        int64_t dl_129451 = ((int64_t *) dls_mem_143959.mem)[step_129423];
        
        // futhark/microgpt.fut:470:37-40
        
        int64_t zl_rhs_129456 = sub64(dl_129451, (int64_t) 1);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142926 = 0; i_142926 < (int64_t) 16; i_142926++) {
            // futhark/microgpt.fut:470:25-81
            
            bool cond_132445 = slt64(i_142926, zl_rhs_129456);
            
            // futhark/microgpt.fut:470:56-59
            
            int64_t zeze_lhs_132446 = add64((int64_t) 1, i_142926);
            
            // futhark/microgpt.fut:470:47-60
            
            bool x_132447 = sle64((int64_t) 0, zeze_lhs_132446);
            
            // futhark/microgpt.fut:470:47-60
            
            bool y_132448 = slt64(zeze_lhs_132446, (int64_t) 16);
            
            // futhark/microgpt.fut:470:47-60
            
            bool bounds_check_132449 = x_132447 && y_132448;
            
            // futhark/microgpt.fut:9:27-39
            
            bool loop_not_taken_132450 = !cond_132445;
            
            // futhark/microgpt.fut:9:27-39
            
            bool protect_assert_disj_132451 = bounds_check_132449 || loop_not_taken_132450;
            
            // futhark/microgpt.fut:470:47-60
            
            bool index_certs_132452;
            
            if (!protect_assert_disj_132451) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_132446, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:470:47-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:470:3-83\n   #6  futhark/microgpt.fut:577:18-38\n   #7  futhark/microgpt.fut:599:26-605:31\n   #8  futhark/microgpt.fut:633:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            int64_t tmp_132467 = ((int64_t *) seqs_mem_143960.mem)[step_129423 * (int64_t) 16 + i_142926];
            
            // futhark/microgpt.fut:579:37-51
            
            bool x_132468 = sle64((int64_t) 0, tmp_132467);
            
            // futhark/microgpt.fut:579:37-51
            
            bool y_132469 = slt64(tmp_132467, (int64_t) 27);
            
            // futhark/microgpt.fut:579:37-51
            
            bool bounds_check_132470 = x_132468 && y_132469;
            
            // futhark/microgpt.fut:579:37-51
            
            bool index_certs_132471;
            
            if (!bounds_check_132470) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_132467, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:579:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:579:16-55\n   #6  futhark/microgpt.fut:599:26-605:31\n   #7  futhark/microgpt.fut:633:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:470:47-60
            
            int64_t zeze_lhs_132453;
            
            if (cond_132445) {
                int64_t x_142621 = ((int64_t *) seqs_mem_143960.mem)[step_129423 * (int64_t) 16 + zeze_lhs_132446];
                
                zeze_lhs_132453 = x_142621;
            } else {
                zeze_lhs_132453 = (int64_t) 0;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142916 = 0; i_142916 < (int64_t) 27; i_142916++) {
                // futhark/microgpt.fut:470:61-65
                
                bool cond_t_res_132457 = zeze_lhs_132453 == i_142916;
                
                // futhark/microgpt.fut:9:27-39
                
                bool x_132458 = cond_132445 && cond_t_res_132457;
                
                // futhark/microgpt.fut:470:25-81
                
                double lifted_lambda_res_132459;
                
                if (x_132458) {
                    lifted_lambda_res_132459 = 1.0;
                } else {
                    lifted_lambda_res_132459 = 0.0;
                }
                ((double *) mem_144079)[i_142916] = lifted_lambda_res_132459;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142920 = 0; i_142920 < (int64_t) 16; i_142920++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_132478 = ((double *) mem_param_143984.mem)[tmp_132467 * (int64_t) 16 + i_142920];
                
                ((double *) mem_144086)[i_142920] = lifted_lambda_res_132478;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144069, i_142926 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144086, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144070, i_142926 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144079, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142943 = 0; i_142943 < (int64_t) 16; i_142943++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_132566;
            double r_132568 = 0.0;
            
            for (int64_t i_132567 = 0; i_132567 < (int64_t) 16; i_132567++) {
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_132569 = ((double *) mem_param_143976.mem)[i_142943 * (int64_t) 16 + i_132567];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_rhs_132570 = ((double *) mem_144069)[i_142943 * (int64_t) 16 + i_132567];
                
                // futhark/microgpt.fut:279:71-107
                
                double zp_res_132571 = zp_lhs_132569 + zp_rhs_132570;
                
                // futhark/microgpt.fut:279:87-150
                
                double zt_res_132572 = zp_res_132571 * zp_res_132571;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_132573 = r_132568 + zt_res_132572;
                double r_tmp_146250 = zp_res_132573;
                
                r_132568 = r_tmp_146250;
            }
            defunc_0_lifted_lambda_res_132566 = r_132568;
            // futhark/microgpt.fut:279:50-169
            
            double zs_res_132574 = defunc_0_lifted_lambda_res_132566 / 16.0;
            
            // futhark/microgpt.fut:280:23-53
            
            double zp_res_132575 = 1.0e-5 + zs_res_132574;
            
            // futhark/microgpt.fut:280:15-53
            
            double sqrt_res_132576 = futrts_sqrt64(zp_res_132575);
            
            // futhark/microgpt.fut:281:79-89
            
            double zs_res_132577 = 1.0 / sqrt_res_132576;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142931 = 0; i_142931 < (int64_t) 16; i_142931++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_132584 = ((double *) mem_param_143976.mem)[i_142943 * (int64_t) 16 + i_142931];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_132585 = ((double *) mem_144069)[i_142943 * (int64_t) 16 + i_142931];
                
                // futhark/microgpt.fut:281:36-72
                
                double zp_res_132586 = zp_lhs_132584 + zp_rhs_132585;
                
                // futhark/microgpt.fut:281:52-89
                
                double zt_res_132587 = zs_res_132577 * zp_res_132586;
                
                ((double *) mem_144114)[i_142931] = zt_res_132587;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142935 = 0; i_142935 < (int64_t) 16; i_142935++) {
                // futhark/microgpt.fut:282:4-12
                
                double lifted_lambda_res_132595 = ((double *) mem_144114)[i_142935];
                
                ((double *) mem_144121)[i_142935] = lifted_lambda_res_132595;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_132603;
            double r_132605 = 0.0;
            
            for (int64_t i_132604 = 0; i_132604 < (int64_t) 16; i_132604++) {
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_132606 = ((double *) mem_param_143976.mem)[i_142943 * (int64_t) 16 + i_132604];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_rhs_132607 = ((double *) mem_144069)[i_142943 * (int64_t) 16 + i_132604];
                
                // futhark/microgpt.fut:395:71-115
                
                double zp_res_132608 = zp_lhs_132606 + zp_rhs_132607;
                
                // futhark/microgpt.fut:395:91-166
                
                double zt_res_132609 = zp_res_132608 * zp_res_132608;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_132610 = r_132605 + zt_res_132609;
                double r_tmp_146253 = zp_res_132610;
                
                r_132605 = r_tmp_146253;
            }
            defunc_0_lifted_lambda_res_132603 = r_132605;
            // futhark/microgpt.fut:395:48-185
            
            double zs_res_132611 = defunc_0_lifted_lambda_res_132603 / 16.0;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_132621;
            double r_132623 = 0.0;
            
            for (int64_t i_132622 = 0; i_132622 < (int64_t) 16; i_132622++) {
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_132624 = ((double *) mem_param_143976.mem)[i_142943 * (int64_t) 16 + i_132622];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_rhs_132625 = ((double *) mem_144069)[i_142943 * (int64_t) 16 + i_132622];
                
                // futhark/microgpt.fut:408:72-116
                
                double zp_res_132626 = zp_lhs_132624 + zp_rhs_132625;
                
                // futhark/microgpt.fut:408:92-167
                
                double zt_res_132627 = zp_res_132626 * zp_res_132626;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_132628 = r_132623 + zt_res_132627;
                double r_tmp_146254 = zp_res_132628;
                
                r_132623 = r_tmp_146254;
            }
            defunc_0_lifted_lambda_res_132621 = r_132623;
            // futhark/microgpt.fut:408:49-186
            
            double zs_res_132629 = defunc_0_lifted_lambda_res_132621 / 16.0;
            
            ((double *) mem_144101)[i_142943] = zs_res_132629;
            ((double *) mem_144102)[i_142943] = zs_res_132611;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144103, i_142943 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144121, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142959 = 0; i_142959 < (int64_t) 16; i_142959++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_132648;
            double r_132650 = 0.0;
            
            for (int64_t i_132649 = 0; i_132649 < (int64_t) 16; i_132649++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_132651 = ((double *) mem_144103)[i_142959 * (int64_t) 16 + i_132649];
                
                // futhark/microgpt.fut:283:71-106
                
                double zt_res_132652 = zt_lhs_132651 * zt_lhs_132651;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_132653 = r_132650 + zt_res_132652;
                double r_tmp_146257 = zp_res_132653;
                
                r_132650 = r_tmp_146257;
            }
            defunc_0_lifted_lambda_res_132648 = r_132650;
            // futhark/microgpt.fut:283:50-124
            
            double zs_res_132654 = defunc_0_lifted_lambda_res_132648 / 16.0;
            
            // futhark/microgpt.fut:284:24-54
            
            double zp_res_132655 = 1.0e-5 + zs_res_132654;
            
            // futhark/microgpt.fut:284:16-54
            
            double sqrt_res_132656 = futrts_sqrt64(zp_res_132655);
            
            // futhark/microgpt.fut:285:58-69
            
            double zs_res_132657 = 1.0 / sqrt_res_132656;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142949 = 0; i_142949 < (int64_t) 16; i_142949++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_132664 = ((double *) mem_144103)[i_142959 * (int64_t) 16 + i_142949];
                
                // futhark/microgpt.fut:285:37-69
                
                double zt_res_132665 = zs_res_132657 * zt_lhs_132664;
                
                ((double *) mem_144147)[i_142949] = zt_res_132665;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142953 = 0; i_142953 < (int64_t) 16; i_142953++) {
                // futhark/microgpt.fut:286:4-13
                
                double lifted_lambda_res_132673 = ((double *) mem_144147)[i_142953];
                
                ((double *) mem_144154)[i_142953] = lifted_lambda_res_132673;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_132681;
            double r_132683 = 0.0;
            
            for (int64_t i_132682 = 0; i_132682 < (int64_t) 16; i_132682++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_132684 = ((double *) mem_144103)[i_142959 * (int64_t) 16 + i_132682];
                
                // futhark/microgpt.fut:373:70-111
                
                double zt_res_132685 = zt_lhs_132684 * zt_lhs_132684;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_132686 = r_132683 + zt_res_132685;
                double r_tmp_146260 = zp_res_132686;
                
                r_132683 = r_tmp_146260;
            }
            defunc_0_lifted_lambda_res_132681 = r_132683;
            // futhark/microgpt.fut:373:48-129
            
            double zs_res_132687 = defunc_0_lifted_lambda_res_132681 / 16.0;
            
            ((double *) mem_144138)[i_142959] = zs_res_132687;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144139, i_142959 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144154, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_142988 = 0; i_142988 < (int64_t) 4; i_142988++) {
            // futhark/microgpt.fut:287:83-86
            
            int64_t zp_lhs_132768 = mul64((int64_t) 4, i_142988);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_142978 = 0; i_142978 < (int64_t) 16; i_142978++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_142968 = 0; i_142968 < (int64_t) 4; i_142968++) {
                    // futhark/microgpt.fut:287:88-95
                    
                    int64_t zt_lhs_137239 = add64(zp_lhs_132768, i_142968);
                    
                    // futhark/microgpt.fut:287:70-97
                    
                    bool x_137240 = sle64((int64_t) 0, zt_lhs_137239);
                    
                    // futhark/microgpt.fut:287:70-97
                    
                    bool y_137241 = slt64(zt_lhs_137239, (int64_t) 16);
                    
                    // futhark/microgpt.fut:287:70-97
                    
                    bool bounds_check_137242 = x_137240 && y_137241;
                    
                    // futhark/microgpt.fut:287:70-97
                    
                    bool index_certs_137243;
                    
                    if (!bounds_check_137242) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_137239, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:287:70-97\n   #1  futhark/microgpt.fut:71:46-49\n   #2  futhark/microgpt.fut:287:49-127\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:15:29-44\n   #8  futhark/microgpt.fut:4:11-25\n   #9  futhark/microgpt.fut:15:15-45\n   #10 futhark/microgpt.fut:287:12-129\n   #11 futhark/microgpt.fut:582:5-76\n   #12 futhark/microgpt.fut:599:26-605:31\n   #13 futhark/microgpt.fut:633:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_137244;
                    double r_137246 = 0.0;
                    
                    for (int64_t i_137245 = 0; i_137245 < (int64_t) 16; i_137245++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_137247 = ((double *) mem_param_143980.mem)[zt_lhs_137239 * (int64_t) 16 + i_137245];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_137248 = ((double *) mem_144139)[i_142978 * (int64_t) 16 + i_137245];
                        
                        // futhark/microgpt.fut:287:70-125
                        
                        double zt_res_137249 = zt_lhs_137247 * zt_rhs_137248;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_137250 = r_137246 + zt_res_137249;
                        double r_tmp_146270 = zp_res_137250;
                        
                        r_137246 = r_tmp_146270;
                    }
                    defunc_0_lifted_lambda_res_137244 = r_137246;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_137258;
                    double r_137260 = 0.0;
                    
                    for (int64_t i_137259 = 0; i_137259 < (int64_t) 16; i_137259++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_137261 = ((double *) mem_param_143968.mem)[zt_lhs_137239 * (int64_t) 16 + i_137259];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_137262 = ((double *) mem_144139)[i_142978 * (int64_t) 16 + i_137259];
                        
                        // futhark/microgpt.fut:288:70-125
                        
                        double zt_res_137263 = zt_lhs_137261 * zt_rhs_137262;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_137264 = r_137260 + zt_res_137263;
                        double r_tmp_146271 = zp_res_137264;
                        
                        r_137260 = r_tmp_146271;
                    }
                    defunc_0_lifted_lambda_res_137258 = r_137260;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_137275;
                    double r_137277 = 0.0;
                    
                    for (int64_t i_137276 = 0; i_137276 < (int64_t) 16; i_137276++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_137278 = ((double *) mem_param_143992.mem)[zt_lhs_137239 * (int64_t) 16 + i_137276];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_137279 = ((double *) mem_144139)[i_142978 * (int64_t) 16 + i_137276];
                        
                        // futhark/microgpt.fut:289:70-125
                        
                        double zt_res_137280 = zt_lhs_137278 * zt_rhs_137279;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_137281 = r_137277 + zt_res_137280;
                        double r_tmp_146272 = zp_res_137281;
                        
                        r_137277 = r_tmp_146272;
                    }
                    defunc_0_lifted_lambda_res_137275 = r_137277;
                    ((double *) mem_144201)[i_142968] = defunc_0_lifted_lambda_res_137275;
                    ((double *) mem_144202)[i_142968] = defunc_0_lifted_lambda_res_137258;
                    ((double *) mem_144203)[i_142968] = defunc_0_lifted_lambda_res_137244;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_144186, i_142978 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144201, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_144187, i_142978 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144202, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_144188, i_142978 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144203, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_144168, i_142988 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144186, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_144169, i_142988 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144187, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_144170, i_142988 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144188, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143040 = 0; i_143040 < (int64_t) 16; i_143040++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143036 = 0; i_143036 < (int64_t) 16; i_143036++) {
                // futhark/microgpt.fut:290:114-117
                
                int64_t zt_lhs_129655 = sdiv64(i_143036, (int64_t) 4);
                
                // futhark/microgpt.fut:290:103-119
                
                bool x_129656 = sle64((int64_t) 0, zt_lhs_129655);
                
                // futhark/microgpt.fut:290:103-119
                
                bool y_129657 = slt64(zt_lhs_129655, (int64_t) 4);
                
                // futhark/microgpt.fut:290:103-119
                
                bool bounds_check_129658 = x_129656 && y_129657;
                
                // futhark/microgpt.fut:290:103-119
                
                bool index_certs_129659;
                
                if (!bounds_check_129658) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_129655, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:290:103-119\n   #1  futhark/microgpt.fut:71:46-49\n   #2  futhark/microgpt.fut:290:83-170\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:290:53-172\n   #8  futhark/microgpt.fut:4:11-25\n   #9  futhark/microgpt.fut:9:27-39\n   #10 futhark/microgpt.fut:4:11-25\n   #11 futhark/microgpt.fut:9:13-40\n   #12 futhark/microgpt.fut:290:12-298:32\n   #13 futhark/microgpt.fut:582:5-76\n   #14 futhark/microgpt.fut:599:26-605:31\n   #15 futhark/microgpt.fut:633:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:298:22-25
                
                int64_t tmp_129774 = smod64(i_143036, (int64_t) 4);
                
                // futhark/microgpt.fut:298:4-27
                
                bool x_129775 = sle64((int64_t) 0, tmp_129774);
                
                // futhark/microgpt.fut:298:4-27
                
                bool y_129776 = slt64(tmp_129774, (int64_t) 4);
                
                // futhark/microgpt.fut:298:4-27
                
                bool bounds_check_129777 = x_129775 && y_129776;
                
                // futhark/microgpt.fut:298:4-27
                
                bool index_certs_129778;
                
                if (!bounds_check_129777) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_129774, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:298:4-27\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:290:12-298:32\n   #6  futhark/microgpt.fut:582:5-76\n   #7  futhark/microgpt.fut:599:26-605:31\n   #8  futhark/microgpt.fut:633:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_142998 = 0; i_142998 < (int64_t) 16; i_142998++) {
                    // futhark/microgpt.fut:4:11-25
                    for (int64_t i_142994 = 0; i_142994 < (int64_t) 16; i_142994++) {
                        // futhark/microgpt.fut:71:13-49
                        
                        double defunc_0_lifted_lambda_res_129672;
                        double r_129674 = 0.0;
                        
                        for (int64_t i_129673 = 0; i_129673 < (int64_t) 4; i_129673++) {
                            // futhark/microgpt.fut:71:46-49
                            
                            double zt_lhs_129675 = ((double *) mem_144170)[zt_lhs_129655 * (int64_t) 64 + i_142998 * (int64_t) 4 + i_129673];
                            
                            // futhark/microgpt.fut:71:46-49
                            
                            double zt_rhs_129676 = ((double *) mem_144169)[zt_lhs_129655 * (int64_t) 64 + i_142994 * (int64_t) 4 + i_129673];
                            
                            // futhark/microgpt.fut:290:103-168
                            
                            double zt_res_129677 = zt_lhs_129675 * zt_rhs_129676;
                            
                            // futhark/microgpt.fut:71:40-49
                            
                            double zp_res_129678 = r_129674 + zt_res_129677;
                            double r_tmp_146277 = zp_res_129678;
                            
                            r_129674 = r_tmp_146277;
                        }
                        defunc_0_lifted_lambda_res_129672 = r_129674;
                        ((double *) mem_144263)[i_142994] = defunc_0_lifted_lambda_res_129672;
                    }
                    lmad_copy_8b(ctx, 1, (uint64_t *) mem_144258, i_142998 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144263, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_143006 = 0; i_143006 < (int64_t) 16; i_143006++) {
                    // futhark/microgpt.fut:4:11-25
                    for (int64_t i_143002 = 0; i_143002 < (int64_t) 16; i_143002++) {
                        // futhark/microgpt.fut:4:11-25
                        
                        double zs_lhs_129693 = ((double *) mem_144258)[i_143006 * (int64_t) 16 + i_143002];
                        
                        // futhark/microgpt.fut:291:47-78
                        
                        double zs_res_129694 = zs_lhs_129693 / 2.0;
                        double zp_rhs_129695 = ((double *) masks_mem_143958.mem)[step_129423 * (int64_t) 256 + i_143006 * (int64_t) 16 + i_143002];
                        
                        // futhark/microgpt.fut:291:65-102
                        
                        double zp_res_129696 = zs_res_129694 + zp_rhs_129695;
                        
                        ((double *) mem_144279)[i_143002] = zp_res_129696;
                    }
                    lmad_copy_8b(ctx, 1, (uint64_t *) mem_144274, i_143006 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144279, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_143024 = 0; i_143024 < (int64_t) 16; i_143024++) {
                    // futhark/microgpt.fut:115:13-33
                    
                    double defunc_0_reduce_res_142636;
                    double redout_143008 = -INFINITY;
                    
                    for (int64_t i_143009 = 0; i_143009 < (int64_t) 16; i_143009++) {
                        // futhark/microgpt.fut:4:11-25
                        
                        double lifted_lambda_res_137308 = ((double *) mem_144274)[i_143024 * (int64_t) 16 + i_143009];
                        
                        // futhark/microgpt.fut:115:13-33
                        
                        double max_res_129717 = fmax64(lifted_lambda_res_137308, redout_143008);
                        double redout_tmp_146281 = max_res_129717;
                        
                        redout_143008 = redout_tmp_146281;
                    }
                    defunc_0_reduce_res_142636 = redout_143008;
                    // futhark/microgpt.fut:293:67-76
                    
                    double neg_res_129718 = -defunc_0_reduce_res_142636;
                    
                    // futhark/microgpt.fut:4:11-25
                    for (int64_t i_143012 = 0; i_143012 < (int64_t) 16; i_143012++) {
                        // futhark/microgpt.fut:4:11-25
                        
                        double zp_lhs_129725 = ((double *) mem_144274)[i_143024 * (int64_t) 16 + i_143012];
                        
                        // futhark/microgpt.fut:293:44-76
                        
                        double zp_res_129726 = neg_res_129718 + zp_lhs_129725;
                        
                        // futhark/microgpt.fut:293:37-76
                        
                        double exp_res_129727 = futrts_exp64(zp_res_129726);
                        
                        ((double *) mem_144295)[i_143012] = exp_res_129727;
                    }
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_129729;
                    double r_129731 = 0.0;
                    
                    for (int64_t i_129730 = 0; i_129730 < (int64_t) 16; i_129730++) {
                        // futhark/microgpt.fut:294:36-46
                        
                        double lifted_lambda_res_129732 = ((double *) mem_144295)[i_129730];
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_129733 = r_129731 + lifted_lambda_res_129732;
                        double r_tmp_146283 = zp_res_129733;
                        
                        r_129731 = r_tmp_146283;
                    }
                    defunc_0_lifted_lambda_res_129729 = r_129731;
                    // futhark/microgpt.fut:295:53-64
                    
                    double zs_res_129734 = 1.0 / defunc_0_lifted_lambda_res_129729;
                    
                    // futhark/microgpt.fut:4:11-25
                    for (int64_t i_143016 = 0; i_143016 < (int64_t) 16; i_143016++) {
                        // futhark/microgpt.fut:295:37-47
                        
                        double zt_lhs_129741 = ((double *) mem_144295)[i_143016];
                        
                        // futhark/microgpt.fut:295:37-64
                        
                        double zt_res_129742 = zs_res_129734 * zt_lhs_129741;
                        
                        ((double *) mem_144302)[i_143016] = zt_res_129742;
                    }
                    // futhark/microgpt.fut:4:11-25
                    for (int64_t i_143020 = 0; i_143020 < (int64_t) 16; i_143020++) {
                        // futhark/microgpt.fut:296:4-14
                        
                        double lifted_lambda_res_129750 = ((double *) mem_144302)[i_143020];
                        
                        ((double *) mem_144309)[i_143020] = lifted_lambda_res_129750;
                    }
                    lmad_copy_8b(ctx, 1, (uint64_t *) mem_144290, i_143024 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144309, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_143032 = 0; i_143032 < (int64_t) 16; i_143032++) {
                    // futhark/microgpt.fut:4:11-25
                    for (int64_t i_143028 = 0; i_143028 < (int64_t) 4; i_143028++) {
                        // futhark/microgpt.fut:71:13-49
                        
                        double defunc_0_lifted_lambda_res_129765;
                        double r_129767 = 0.0;
                        
                        for (int64_t i_129766 = 0; i_129766 < (int64_t) 16; i_129766++) {
                            // futhark/microgpt.fut:71:46-49
                            
                            double zt_lhs_129768 = ((double *) mem_144290)[i_143032 * (int64_t) 16 + i_129766];
                            
                            // futhark/microgpt.fut:71:46-49
                            
                            double zt_rhs_129769 = ((double *) mem_144168)[zt_lhs_129655 * (int64_t) 64 + i_129766 * (int64_t) 4 + i_143028];
                            
                            // futhark/microgpt.fut:297:66-118
                            
                            double zt_res_129770 = zt_lhs_129768 * zt_rhs_129769;
                            
                            // futhark/microgpt.fut:71:40-49
                            
                            double zp_res_129771 = r_129767 + zt_res_129770;
                            double r_tmp_146288 = zp_res_129771;
                            
                            r_129767 = r_tmp_146288;
                        }
                        defunc_0_lifted_lambda_res_129765 = r_129767;
                        ((double *) mem_144325)[i_143028] = defunc_0_lifted_lambda_res_129765;
                    }
                    lmad_copy_8b(ctx, 1, (uint64_t *) mem_144320, i_143032 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144325, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_129779 = ((double *) mem_144320)[i_143040 * (int64_t) 4 + tmp_129774];
                
                ((double *) mem_144254)[i_143036] = lifted_lambda_res_129779;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144249, i_143040 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144254, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143048 = 0; i_143048 < (int64_t) 16; i_143048++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143044 = 0; i_143044 < (int64_t) 16; i_143044++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_129794;
                double r_129796 = 0.0;
                
                for (int64_t i_129795 = 0; i_129795 < (int64_t) 16; i_129795++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_129797 = ((double *) mem_param_143972.mem)[i_143044 * (int64_t) 16 + i_129795];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_129798 = ((double *) mem_144249)[i_143048 * (int64_t) 16 + i_129795];
                    
                    // futhark/microgpt.fut:299:64-104
                    
                    double zt_res_129799 = zt_lhs_129797 * zt_rhs_129798;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_129800 = r_129796 + zt_res_129799;
                    double r_tmp_146291 = zp_res_129800;
                    
                    r_129796 = r_tmp_146291;
                }
                defunc_0_lifted_lambda_res_129794 = r_129796;
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_129801 = ((double *) mem_144103)[i_143048 * (int64_t) 16 + i_143044];
                
                // futhark/microgpt.fut:299:43-128
                
                double zp_res_129802 = defunc_0_lifted_lambda_res_129794 + zp_rhs_129801;
                
                ((double *) mem_144348)[i_143044] = zp_res_129802;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144343, i_143048 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144348, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143062 = 0; i_143062 < (int64_t) 16; i_143062++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_132862;
            double r_132864 = 0.0;
            
            for (int64_t i_132863 = 0; i_132863 < (int64_t) 16; i_132863++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_132865 = ((double *) mem_144343)[i_143062 * (int64_t) 16 + i_132863];
                
                // futhark/microgpt.fut:300:75-114
                
                double zt_res_132866 = zt_lhs_132865 * zt_lhs_132865;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_132867 = r_132864 + zt_res_132866;
                double r_tmp_146294 = zp_res_132867;
                
                r_132864 = r_tmp_146294;
            }
            defunc_0_lifted_lambda_res_132862 = r_132864;
            // futhark/microgpt.fut:300:54-132
            
            double zs_res_132868 = defunc_0_lifted_lambda_res_132862 / 16.0;
            
            // futhark/microgpt.fut:301:24-55
            
            double zp_res_132869 = 1.0e-5 + zs_res_132868;
            
            // futhark/microgpt.fut:301:16-55
            
            double sqrt_res_132870 = futrts_sqrt64(zp_res_132869);
            
            // futhark/microgpt.fut:302:60-71
            
            double zs_res_132871 = 1.0 / sqrt_res_132870;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143052 = 0; i_143052 < (int64_t) 16; i_143052++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_132878 = ((double *) mem_144343)[i_143062 * (int64_t) 16 + i_143052];
                
                // futhark/microgpt.fut:302:37-71
                
                double zt_res_132879 = zs_res_132871 * zt_lhs_132878;
                
                ((double *) mem_144368)[i_143052] = zt_res_132879;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143056 = 0; i_143056 < (int64_t) 16; i_143056++) {
                // futhark/microgpt.fut:303:4-14
                
                double lifted_lambda_res_132887 = ((double *) mem_144368)[i_143056];
                
                ((double *) mem_144375)[i_143056] = lifted_lambda_res_132887;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_132895;
            double r_132897 = 0.0;
            
            for (int64_t i_132896 = 0; i_132896 < (int64_t) 16; i_132896++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_132898 = ((double *) mem_144343)[i_143062 * (int64_t) 16 + i_132896];
                
                // futhark/microgpt.fut:325:70-113
                
                double zt_res_132899 = zt_lhs_132898 * zt_lhs_132898;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_132900 = r_132897 + zt_res_132899;
                double r_tmp_146297 = zp_res_132900;
                
                r_132897 = r_tmp_146297;
            }
            defunc_0_lifted_lambda_res_132895 = r_132897;
            // futhark/microgpt.fut:325:48-131
            
            double zs_res_132901 = defunc_0_lifted_lambda_res_132895 / 16.0;
            
            ((double *) mem_144359)[i_143062] = zs_res_132901;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144360, i_143062 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144375, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143071 = 0; i_143071 < (int64_t) 16; i_143071++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143067 = 0; i_143067 < (int64_t) 64; i_143067++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_129852;
                double r_129854 = 0.0;
                
                for (int64_t i_129853 = 0; i_129853 < (int64_t) 16; i_129853++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_129855 = ((double *) mem_param_143988.mem)[i_143067 * (int64_t) 16 + i_129853];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_129856 = ((double *) mem_144360)[i_143071 * (int64_t) 16 + i_129853];
                    
                    // futhark/microgpt.fut:304:63-102
                    
                    double zt_res_129857 = zt_lhs_129855 * zt_rhs_129856;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_129858 = r_129854 + zt_res_129857;
                    double r_tmp_146300 = zp_res_129858;
                    
                    r_129854 = r_tmp_146300;
                }
                defunc_0_lifted_lambda_res_129852 = r_129854;
                ((double *) mem_144394)[i_143067] = defunc_0_lifted_lambda_res_129852;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144389, i_143071 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144394, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143079 = 0; i_143079 < (int64_t) 16; i_143079++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143075 = 0; i_143075 < (int64_t) 64; i_143075++) {
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_129873 = ((double *) mem_144389)[i_143079 * (int64_t) 64 + i_143075];
                
                // futhark/microgpt.fut:305:41-69
                
                double max_res_129874 = fmax64(0.0, max_arg0_129873);
                
                ((double *) mem_144410)[i_143075] = max_res_129874;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144405, i_143079 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144410, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143087 = 0; i_143087 < (int64_t) 16; i_143087++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143083 = 0; i_143083 < (int64_t) 16; i_143083++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_129889;
                double r_129891 = 0.0;
                
                for (int64_t i_129890 = 0; i_129890 < (int64_t) 64; i_129890++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_129892 = ((double *) mem_param_143964.mem)[i_143083 * (int64_t) 64 + i_129890];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_129893 = ((double *) mem_144405)[i_143087 * (int64_t) 64 + i_129890];
                    
                    // futhark/microgpt.fut:306:64-105
                    
                    double zt_res_129894 = zt_lhs_129892 * zt_rhs_129893;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_129895 = r_129891 + zt_res_129894;
                    double r_tmp_146305 = zp_res_129895;
                    
                    r_129891 = r_tmp_146305;
                }
                defunc_0_lifted_lambda_res_129889 = r_129891;
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_129896 = ((double *) mem_144343)[i_143087 * (int64_t) 16 + i_143083];
                
                // futhark/microgpt.fut:306:43-130
                
                double zp_res_129897 = defunc_0_lifted_lambda_res_129889 + zp_rhs_129896;
                
                ((double *) mem_144426)[i_143083] = zp_res_129897;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144421, i_143087 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144426, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143095 = 0; i_143095 < (int64_t) 16; i_143095++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143091 = 0; i_143091 < (int64_t) 27; i_143091++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_129912;
                double r_129914 = 0.0;
                
                for (int64_t i_129913 = 0; i_129913 < (int64_t) 16; i_129913++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_129915 = ((double *) mem_param_143996.mem)[i_143091 * (int64_t) 16 + i_129913];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_129916 = ((double *) mem_144421)[i_143095 * (int64_t) 16 + i_129913];
                    
                    // futhark/microgpt.fut:307:63-103
                    
                    double zt_res_129917 = zt_lhs_129915 * zt_rhs_129916;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_129918 = r_129914 + zt_res_129917;
                    double r_tmp_146308 = zp_res_129918;
                    
                    r_129914 = r_tmp_146308;
                }
                defunc_0_lifted_lambda_res_129912 = r_129914;
                ((double *) mem_144442)[i_143091] = defunc_0_lifted_lambda_res_129912;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144437, i_143095 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144442, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143125 = 0; i_143125 < (int64_t) 16; i_143125++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_142714;
            double defunc_0_reduce_res_142715;
            double redout_143112;
            double redout_143113;
            
            redout_143112 = -INFINITY;
            redout_143113 = -INFINITY;
            for (int64_t i_143115 = 0; i_143115 < (int64_t) 27; i_143115++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_137500 = ((double *) mem_144437)[i_143125 * (int64_t) 27 + i_143115];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_143109 = 0; i_143109 < (int64_t) 27; i_143109++) {
                    // futhark/microgpt.fut:312:55-316:90
                    
                    bool cond_137509 = i_143109 == i_143115;
                    
                    // futhark/microgpt.fut:312:55-316:90
                    
                    double lifted_lambda_res_137510;
                    
                    if (cond_137509) {
                        // futhark/microgpt.fut:115:13-33
                        
                        double defunc_0_reduce_res_142661;
                        double redout_143097 = -INFINITY;
                        
                        for (int64_t i_143098 = 0; i_143098 < (int64_t) 27; i_143098++) {
                            // futhark/microgpt.fut:4:11-25
                            
                            double lifted_lambda_res_142667 = ((double *) mem_144437)[i_143125 * (int64_t) 27 + i_143098];
                            
                            // futhark/microgpt.fut:115:13-33
                            
                            double max_res_142670 = fmax64(lifted_lambda_res_142667, redout_143097);
                            double redout_tmp_146317 = max_res_142670;
                            
                            redout_143097 = redout_tmp_146317;
                        }
                        defunc_0_reduce_res_142661 = redout_143097;
                        // futhark/microgpt.fut:313:67-76
                        
                        double neg_res_142672 = -defunc_0_reduce_res_142661;
                        
                        // futhark/microgpt.fut:4:11-25
                        if (mem_144483_cached_sizze_146766 < (int64_t) 216) {
                            err = lexical_realloc(ctx, &mem_144483, &mem_144483_cached_sizze_146766, (int64_t) 216);
                            if (err != FUTHARK_SUCCESS)
                                goto cleanup;
                        }
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_143101 = 0; i_143101 < (int64_t) 27; i_143101++) {
                            // futhark/microgpt.fut:4:11-25
                            
                            double zp_lhs_142679 = ((double *) mem_144437)[i_143125 * (int64_t) 27 + i_143101];
                            
                            // futhark/microgpt.fut:313:44-76
                            
                            double zp_res_142680 = neg_res_142672 + zp_lhs_142679;
                            
                            // futhark/microgpt.fut:313:37-76
                            
                            double exp_res_142681 = futrts_exp64(zp_res_142680);
                            
                            ((double *) mem_144483)[i_143101] = exp_res_142681;
                        }
                        // futhark/microgpt.fut:71:13-49
                        
                        double defunc_0_lifted_lambda_res_142684;
                        double r_142686 = 0.0;
                        
                        for (int64_t i_142685 = 0; i_142685 < (int64_t) 27; i_142685++) {
                            // futhark/microgpt.fut:314:36-46
                            
                            double lifted_lambda_res_142687 = ((double *) mem_144483)[i_142685];
                            
                            // futhark/microgpt.fut:71:40-49
                            
                            double zp_res_142688 = r_142686 + lifted_lambda_res_142687;
                            double r_tmp_146319 = zp_res_142688;
                            
                            r_142686 = r_tmp_146319;
                        }
                        defunc_0_lifted_lambda_res_142684 = r_142686;
                        // futhark/microgpt.fut:315:53-64
                        
                        double zs_res_142689 = 1.0 / defunc_0_lifted_lambda_res_142684;
                        
                        // futhark/microgpt.fut:4:11-25
                        if (mem_144490_cached_sizze_146767 < (int64_t) 216) {
                            err = lexical_realloc(ctx, &mem_144490, &mem_144490_cached_sizze_146767, (int64_t) 216);
                            if (err != FUTHARK_SUCCESS)
                                goto cleanup;
                        }
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_143105 = 0; i_143105 < (int64_t) 27; i_143105++) {
                            // futhark/microgpt.fut:315:37-47
                            
                            double zt_lhs_142696 = ((double *) mem_144483)[i_143105];
                            
                            // futhark/microgpt.fut:315:37-64
                            
                            double zt_res_142697 = zs_res_142689 * zt_lhs_142696;
                            
                            ((double *) mem_144490)[i_143105] = zt_res_142697;
                        }
                        // futhark/microgpt.fut:4:11-25
                        
                        double zt_rhs_142704 = ((double *) mem_144070)[i_143125 * (int64_t) 27 + i_143115];
                        
                        // futhark/microgpt.fut:316:7-49
                        
                        double zt_res_142705 = -6.25e-2 * zt_rhs_142704;
                        
                        // futhark/microgpt.fut:316:64-74
                        
                        double zs_rhs_142710 = ((double *) mem_144490)[i_143109];
                        
                        // futhark/microgpt.fut:316:56-74
                        
                        double zs_res_142711 = 1.0 / zs_rhs_142710;
                        
                        // futhark/microgpt.fut:316:25-74
                        
                        double zt_res_142712 = zt_res_142705 * zs_res_142711;
                        
                        lifted_lambda_res_137510 = zt_res_142712;
                    } else {
                        lifted_lambda_res_137510 = 0.0;
                    }
                    ((double *) mem_144479)[i_143109] = lifted_lambda_res_137510;
                }
                // futhark/microgpt.fut:115:13-33
                
                double max_res_133038 = fmax64(lifted_lambda_res_137500, redout_143112);
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_133129 = fmax64(lifted_lambda_res_137500, redout_143113);
                
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_144474, i_143115 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144479, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
                
                double redout_tmp_146313 = max_res_133038;
                double redout_tmp_146314 = max_res_133129;
                
                redout_143112 = redout_tmp_146313;
                redout_143113 = redout_tmp_146314;
            }
            defunc_0_reduce_res_142714 = redout_143112;
            defunc_0_reduce_res_142715 = redout_143113;
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_146321 = 0; nest_i_146321 < (int64_t) 27; nest_i_146321++) {
                ((double *) mem_144456)[i_143125 * (int64_t) 27 + nest_i_146321] = defunc_0_reduce_res_142714;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_146322 = 0; nest_i_146322 < (int64_t) 27; nest_i_146322++) {
                ((double *) mem_144454)[i_143125 * (int64_t) 27 + nest_i_146322] = defunc_0_reduce_res_142715;
            }
            // futhark/microgpt.fut:321:134-157
            
            double neg_res_133140 = -defunc_0_reduce_res_142715;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_133141;
            double r_133143 = 0.0;
            
            for (int64_t i_133142 = 0; i_133142 < (int64_t) 27; i_133142++) {
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_133144 = ((double *) mem_144437)[i_143125 * (int64_t) 27 + i_133142];
                
                // futhark/microgpt.fut:321:111-157
                
                double zp_res_133145 = neg_res_133140 + zp_lhs_133144;
                
                // futhark/microgpt.fut:321:104-157
                
                double neg_res_133146 = -zp_res_133145;
                
                // futhark/microgpt.fut:110:42-54
                
                double max_res_133147 = fmax64(0.0, neg_res_133146);
                
                // futhark/microgpt.fut:110:35-54
                
                double sgn_res_133148 = fsignum64(max_res_133147);
                
                // futhark/microgpt.fut:321:85-160
                
                double neg_res_133149 = -sgn_res_133148;
                
                // futhark/microgpt.fut:321:76-161
                
                double zp_res_133150 = 1.0 + neg_res_133149;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_133151 = r_133143 + zp_res_133150;
                double r_tmp_146323 = zp_res_133151;
                
                r_133143 = r_tmp_146323;
            }
            defunc_0_lifted_lambda_res_133141 = r_133143;
            // futhark/microgpt.fut:321:46-164
            
            double zs_res_133152 = 1.0 / defunc_0_lifted_lambda_res_133141;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_146324 = 0; nest_i_146324 < (int64_t) 27; nest_i_146324++) {
                ((double *) mem_144453)[i_143125 * (int64_t) 27 + nest_i_146324] = zs_res_133152;
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_144455, i_143125 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_144474, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143140 = 0; i_143140 < (int64_t) 16; i_143140++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143136 = 0; i_143136 < (int64_t) 27; i_143136++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_129954 = ((double *) mem_144456)[i_143140 * (int64_t) 27 + i_143136];
                
                // futhark/microgpt.fut:310:85-108
                
                double neg_res_129955 = -neg_arg0_129954;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_143132 = 0; i_143132 < (int64_t) 27; i_143132++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_129962 = ((double *) mem_144437)[i_143140 * (int64_t) 27 + i_143132];
                    
                    // futhark/microgpt.fut:310:62-108
                    
                    double zp_res_129963 = neg_res_129955 + zp_lhs_129962;
                    
                    // futhark/microgpt.fut:310:55-108
                    
                    double exp_res_129964 = futrts_exp64(zp_res_129963);
                    
                    ((double *) mem_144535)[i_143132] = exp_res_129964;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_144530, i_143136 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144535, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_144524, i_143140 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_144530, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143153 = 0; i_143153 < (int64_t) 16; i_143153++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143146 = 0; i_143146 < (int64_t) 27; i_143146++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_137874;
                double r_137876 = 0.0;
                
                for (int64_t i_137875 = 0; i_137875 < (int64_t) 27; i_137875++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_137877 = ((double *) mem_144524)[i_143153 * (int64_t) 729 + i_143146 * (int64_t) 27 + i_137875];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_137878 = r_137876 + lifted_lambda_res_137877;
                    double r_tmp_146332 = zp_res_137878;
                    
                    r_137876 = r_tmp_146332;
                }
                defunc_0_lifted_lambda_res_137874 = r_137876;
                // futhark/microgpt.fut:317:144-183
                
                double zt_res_137886 = defunc_0_lifted_lambda_res_137874 * defunc_0_lifted_lambda_res_137874;
                
                // futhark/microgpt.fut:317:135-183
                
                double zs_res_137887 = 1.0 / zt_res_137886;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_137888;
                double r_137890 = 0.0;
                
                for (int64_t i_137889 = 0; i_137889 < (int64_t) 27; i_137889++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_137891 = ((double *) mem_144455)[i_143153 * (int64_t) 729 + i_143146 * (int64_t) 27 + i_137889];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_137892 = ((double *) mem_144524)[i_143153 * (int64_t) 729 + i_143146 * (int64_t) 27 + i_137889];
                    
                    // futhark/microgpt.fut:317:75-128
                    
                    double zt_res_137893 = zt_lhs_137891 * zt_rhs_137892;
                    
                    // futhark/microgpt.fut:317:100-183
                    
                    double zt_res_137894 = zs_res_137887 * zt_res_137893;
                    
                    // futhark/microgpt.fut:317:67-183
                    
                    double neg_res_137895 = -zt_res_137894;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_137896 = r_137890 + neg_res_137895;
                    double r_tmp_146333 = zp_res_137896;
                    
                    r_137890 = r_tmp_146333;
                }
                defunc_0_lifted_lambda_res_137888 = r_137890;
                ((double *) mem_144561)[i_143146] = defunc_0_lifted_lambda_res_137888;
                ((double *) mem_144562)[i_143146] = defunc_0_lifted_lambda_res_137874;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144551, i_143153 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144561, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144552, i_143153 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144562, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143166 = 0; i_143166 < (int64_t) 16; i_143166++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143162 = 0; i_143162 < (int64_t) 27; i_143162++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_130094 = ((double *) mem_144552)[i_143166 * (int64_t) 27 + i_143162];
                
                // futhark/microgpt.fut:318:86-111
                
                double zs_res_130095 = 1.0 / zs_rhs_130094;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_130096 = ((double *) mem_144551)[i_143166 * (int64_t) 27 + i_143162];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_143158 = 0; i_143158 < (int64_t) 27; i_143158++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_130103 = ((double *) mem_144455)[i_143166 * (int64_t) 729 + i_143162 * (int64_t) 27 + i_143158];
                    
                    // futhark/microgpt.fut:318:56-111
                    
                    double zt_res_130104 = zs_res_130095 * zt_lhs_130103;
                    
                    // futhark/microgpt.fut:318:81-135
                    
                    double zp_res_130105 = zp_rhs_130096 + zt_res_130104;
                    
                    ((double *) mem_144594)[i_143158] = zp_res_130105;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_144589, i_143162 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144594, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_144583, i_143166 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_144589, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143174 = 0; i_143174 < (int64_t) 16; i_143174++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143170 = 0; i_143170 < (int64_t) 27; i_143170++) {
                double f_elem_130118 = ((double *) mem_144456)[i_143174 * (int64_t) 27 + i_143170];
                
                // futhark/microgpt.fut:319:105-128
                
                double neg_res_130123 = -f_elem_130118;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_130124;
                double r_130126 = 0.0;
                
                for (int64_t i_130125 = 0; i_130125 < (int64_t) 27; i_130125++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_130127 = ((double *) mem_144437)[i_143174 * (int64_t) 27 + i_130125];
                    
                    // futhark/microgpt.fut:319:82-128
                    
                    double zp_res_130128 = neg_res_130123 + zp_lhs_130127;
                    
                    // futhark/microgpt.fut:319:75-128
                    
                    double exp_res_130129 = futrts_exp64(zp_res_130128);
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_130130 = ((double *) mem_144583)[i_143174 * (int64_t) 729 + i_143170 * (int64_t) 27 + i_130125];
                    
                    // futhark/microgpt.fut:319:75-160
                    
                    double zt_res_130131 = exp_res_130129 * zt_rhs_130130;
                    
                    // futhark/microgpt.fut:319:67-160
                    
                    double neg_res_130132 = -zt_res_130131;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_130133 = r_130126 + neg_res_130132;
                    double r_tmp_146339 = zp_res_130133;
                    
                    r_130126 = r_tmp_146339;
                }
                defunc_0_lifted_lambda_res_130124 = r_130126;
                ((double *) mem_144615)[i_143170] = defunc_0_lifted_lambda_res_130124;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144610, i_143174 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144615, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143182 = 0; i_143182 < (int64_t) 16; i_143182++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143178 = 0; i_143178 < (int64_t) 27; i_143178++) {
                double f_elem_130190 = ((double *) mem_144437)[i_143182 * (int64_t) 27 + i_143178];
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_130195;
                double r_130197 = 0.0;
                
                for (int64_t i_130196 = 0; i_130196 < (int64_t) 27; i_130196++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double neg_arg0_130198 = ((double *) mem_144456)[i_143182 * (int64_t) 27 + i_130196];
                    
                    // futhark/microgpt.fut:322:89-113
                    
                    double neg_res_130199 = -neg_arg0_130198;
                    
                    // futhark/microgpt.fut:322:66-113
                    
                    double zp_res_130200 = f_elem_130190 + neg_res_130199;
                    
                    // futhark/microgpt.fut:322:59-113
                    
                    double exp_res_130201 = futrts_exp64(zp_res_130200);
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_130202 = ((double *) mem_144583)[i_143182 * (int64_t) 729 + i_130196 * (int64_t) 27 + i_143178];
                    
                    // futhark/microgpt.fut:322:59-146
                    
                    double zt_res_130203 = exp_res_130201 * zt_rhs_130202;
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_130204 = ((double *) mem_144610)[i_143182 * (int64_t) 27 + i_130196];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double neg_arg0_130205 = ((double *) mem_144454)[i_143182 * (int64_t) 27 + i_130196];
                    
                    // futhark/microgpt.fut:322:236-260
                    
                    double neg_res_130206 = -neg_arg0_130205;
                    
                    // futhark/microgpt.fut:322:213-260
                    
                    double zp_res_130207 = f_elem_130190 + neg_res_130206;
                    
                    // futhark/microgpt.fut:322:206-260
                    
                    double neg_res_130208 = -zp_res_130207;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_130209 = fmax64(0.0, neg_res_130208);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_130210 = fsignum64(max_res_130209);
                    
                    // futhark/microgpt.fut:322:187-263
                    
                    double neg_res_130211 = -sgn_res_130210;
                    
                    // futhark/microgpt.fut:322:178-264
                    
                    double zp_res_130212 = 1.0 + neg_res_130211;
                    
                    // futhark/microgpt.fut:322:154-264
                    
                    double zt_res_130213 = zt_lhs_130204 * zp_res_130212;
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_130214 = ((double *) mem_144453)[i_143182 * (int64_t) 27 + i_130196];
                    
                    // futhark/microgpt.fut:322:173-290
                    
                    double zt_res_130215 = zt_res_130213 * zt_rhs_130214;
                    
                    // futhark/microgpt.fut:322:117-290
                    
                    double zp_res_130216 = zt_res_130203 + zt_res_130215;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_130217 = r_130197 + zp_res_130216;
                    double r_tmp_146342 = zp_res_130217;
                    
                    r_130197 = r_tmp_146342;
                }
                defunc_0_lifted_lambda_res_130195 = r_130197;
                ((double *) mem_144631)[i_143178] = defunc_0_lifted_lambda_res_130195;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144626, i_143182 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144631, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143190 = 0; i_143190 < (int64_t) 16; i_143190++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143186 = 0; i_143186 < (int64_t) 16; i_143186++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_130232;
                double r_130234 = 0.0;
                
                for (int64_t i_130233 = 0; i_130233 < (int64_t) 27; i_130233++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_130235 = ((double *) mem_144626)[i_143190 * (int64_t) 27 + i_130233];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_130236 = ((double *) mem_param_143996.mem)[i_130233 * (int64_t) 16 + i_143186];
                    
                    // futhark/microgpt.fut:323:67-111
                    
                    double zt_res_130237 = zt_lhs_130235 * zt_rhs_130236;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_130238 = r_130234 + zt_res_130237;
                    double r_tmp_146345 = zp_res_130238;
                    
                    r_130234 = r_tmp_146345;
                }
                defunc_0_lifted_lambda_res_130232 = r_130234;
                ((double *) mem_144647)[i_143186] = defunc_0_lifted_lambda_res_130232;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144642, i_143190 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144647, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143203 = 0; i_143203 < (int64_t) 16; i_143203++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143196 = 0; i_143196 < (int64_t) 64; i_143196++) {
                // futhark/microgpt.fut:4:11-25
                
                double indicatorp_arg0_137921 = ((double *) mem_144389)[i_143203 * (int64_t) 64 + i_143196];
                
                // futhark/microgpt.fut:110:42-54
                
                double max_res_137922 = fmax64(0.0, indicatorp_arg0_137921);
                
                // futhark/microgpt.fut:110:35-54
                
                double sgn_res_137923 = fsignum64(max_res_137922);
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_137924;
                double r_137926 = 0.0;
                
                for (int64_t i_137925 = 0; i_137925 < (int64_t) 16; i_137925++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_137927 = ((double *) mem_144642)[i_143203 * (int64_t) 16 + i_137925];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_137928 = ((double *) mem_param_143964.mem)[i_137925 * (int64_t) 64 + i_143196];
                    
                    // futhark/microgpt.fut:324:105-151
                    
                    double zt_res_137929 = zt_lhs_137927 * zt_rhs_137928;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_137930 = r_137926 + zt_res_137929;
                    double r_tmp_146350 = zp_res_137930;
                    
                    r_137926 = r_tmp_146350;
                }
                defunc_0_lifted_lambda_res_137924 = r_137926;
                // futhark/microgpt.fut:324:46-153
                
                double zt_res_137931 = sgn_res_137923 * defunc_0_lifted_lambda_res_137924;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_137938;
                double r_137940 = 0.0;
                
                for (int64_t i_137939 = 0; i_137939 < (int64_t) 16; i_137939++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_137941 = ((double *) mem_144642)[i_137939 * (int64_t) 16 + i_143203];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_137942 = ((double *) mem_144405)[i_137939 * (int64_t) 64 + i_143196];
                    
                    // futhark/microgpt.fut:406:69-113
                    
                    double zt_res_137943 = zt_lhs_137941 * zt_rhs_137942;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_137944 = r_137940 + zt_res_137943;
                    double r_tmp_146351 = zp_res_137944;
                    
                    r_137940 = r_tmp_146351;
                }
                defunc_0_lifted_lambda_res_137938 = r_137940;
                ((double *) mem_144668)[i_143196] = defunc_0_lifted_lambda_res_137938;
                ((double *) mem_144669)[i_143196] = zt_res_137931;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144658, i_143203 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144668, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144659, i_143203 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144669, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143212 = 0; i_143212 < (int64_t) 16; i_143212++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143208 = 0; i_143208 < (int64_t) 16; i_143208++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_130302;
                double r_130304 = 0.0;
                
                for (int64_t i_130303 = 0; i_130303 < (int64_t) 64; i_130303++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_130305 = ((double *) mem_144659)[i_143212 * (int64_t) 64 + i_130303];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_130306 = ((double *) mem_param_143988.mem)[i_130303 * (int64_t) 16 + i_143208];
                    
                    // futhark/microgpt.fut:327:71-115
                    
                    double zt_res_130307 = zt_lhs_130305 * zt_rhs_130306;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_130308 = r_130304 + zt_res_130307;
                    double r_tmp_146354 = zp_res_130308;
                    
                    r_130304 = r_tmp_146354;
                }
                defunc_0_lifted_lambda_res_130302 = r_130304;
                ((double *) mem_144695)[i_143208] = defunc_0_lifted_lambda_res_130302;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144690, i_143212 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144695, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143218 = 0; i_143218 < (int64_t) 16; i_143218++) {
            // futhark/microgpt.fut:326:47-59
            
            double zp_lhs_132302 = ((double *) mem_144359)[i_143218];
            
            // futhark/microgpt.fut:326:47-87
            
            double zp_res_132303 = 1.0e-5 + zp_lhs_132302;
            
            // futhark/microgpt.fut:326:39-87
            
            double sqrt_res_132304 = futrts_sqrt64(zp_res_132303);
            
            // futhark/microgpt.fut:328:129-158
            
            double zt_res_132312 = sqrt_res_132304 * sqrt_res_132304;
            
            // futhark/microgpt.fut:328:120-158
            
            double zs_res_132313 = 1.0 / zt_res_132312;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_132314;
            double r_132316 = 0.0;
            
            for (int64_t i_132315 = 0; i_132315 < (int64_t) 16; i_132315++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_132317 = ((double *) mem_144690)[i_143218 * (int64_t) 16 + i_132315];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_132318 = ((double *) mem_144343)[i_143218 * (int64_t) 16 + i_132315];
                
                // futhark/microgpt.fut:328:69-113
                
                double zt_res_132319 = zt_lhs_132317 * zt_rhs_132318;
                
                // futhark/microgpt.fut:328:90-158
                
                double zt_res_132320 = zs_res_132313 * zt_res_132319;
                
                // futhark/microgpt.fut:328:61-158
                
                double neg_res_132321 = -zt_res_132320;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_132322 = r_132316 + neg_res_132321;
                double r_tmp_146357 = zp_res_132322;
                
                r_132316 = r_tmp_146357;
            }
            defunc_0_lifted_lambda_res_132314 = r_132316;
            ((double *) mem_144706)[i_143218] = defunc_0_lifted_lambda_res_132314;
            ((double *) mem_144707)[i_143218] = sqrt_res_132304;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143223 = 0; i_143223 < (int64_t) 16; i_143223++) {
            // futhark/microgpt.fut:329:39-51
            
            double zt_lhs_130336 = ((double *) mem_144706)[i_143223];
            
            // futhark/microgpt.fut:329:93-105
            
            double zp_lhs_130337 = ((double *) mem_144359)[i_143223];
            
            // futhark/microgpt.fut:329:93-133
            
            double zp_res_130338 = 1.0e-5 + zp_lhs_130337;
            
            // futhark/microgpt.fut:329:85-133
            
            double sqrt_res_130339 = futrts_sqrt64(zp_res_130338);
            
            // futhark/microgpt.fut:329:71-135
            
            double zt_res_130340 = 2.0 * sqrt_res_130339;
            
            // futhark/microgpt.fut:329:57-135
            
            double zs_res_130341 = 1.0 / zt_res_130340;
            
            // futhark/microgpt.fut:329:39-135
            
            double zt_res_130342 = zt_lhs_130336 * zs_res_130341;
            
            ((double *) mem_144720)[i_143223] = zt_res_130342;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143231 = 0; i_143231 < (int64_t) 16; i_143231++) {
            // futhark/microgpt.fut:330:98-110
            
            double zs_rhs_130350 = ((double *) mem_144707)[i_143231];
            
            // futhark/microgpt.fut:330:90-110
            
            double zs_res_130351 = 1.0 / zs_rhs_130350;
            
            // futhark/microgpt.fut:330:120-132
            
            double zs_lhs_130352 = ((double *) mem_144720)[i_143231];
            
            // futhark/microgpt.fut:330:120-147
            
            double zs_res_130353 = zs_lhs_130352 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143227 = 0; i_143227 < (int64_t) 16; i_143227++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_130360 = ((double *) mem_144642)[i_143231 * (int64_t) 16 + i_143227];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_130361 = ((double *) mem_144690)[i_143231 * (int64_t) 16 + i_143227];
                
                // futhark/microgpt.fut:330:64-110
                
                double zt_res_130362 = zs_res_130351 * zt_lhs_130361;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_130363 = ((double *) mem_144343)[i_143231 * (int64_t) 16 + i_143227];
                
                // futhark/microgpt.fut:330:133-172
                
                double zt_res_130364 = zs_res_130353 * zt_rhs_130363;
                
                // futhark/microgpt.fut:330:149-232
                
                double zp_res_130365 = zt_res_130364 + zt_res_130364;
                
                // futhark/microgpt.fut:330:85-232
                
                double zp_res_130366 = zt_res_130362 + zp_res_130365;
                
                // futhark/microgpt.fut:330:37-232
                
                double zp_res_130367 = zp_lhs_130360 + zp_res_130366;
                
                ((double *) mem_144732)[i_143227] = zp_res_130367;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_144727, i_143231 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144732, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143268 = 0; i_143268 < (int64_t) 4; i_143268++) {
            // futhark/microgpt.fut:331:122-125
            
            int64_t zp_lhs_133401 = mul64((int64_t) 4, i_143268);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143255 = 0; i_143255 < (int64_t) 16; i_143255++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_143235 = 0; i_143235 < (int64_t) 4; i_143235++) {
                    // futhark/microgpt.fut:331:127-135
                    
                    int64_t zt_rhs_138114 = add64(zp_lhs_133401, i_143235);
                    
                    // futhark/microgpt.fut:331:100-137
                    
                    bool x_138115 = sle64((int64_t) 0, zt_rhs_138114);
                    
                    // futhark/microgpt.fut:331:100-137
                    
                    bool y_138116 = slt64(zt_rhs_138114, (int64_t) 16);
                    
                    // futhark/microgpt.fut:331:100-137
                    
                    bool bounds_check_138117 = x_138115 && y_138116;
                    
                    // futhark/microgpt.fut:331:100-137
                    
                    bool index_certs_138118;
                    
                    if (!bounds_check_138117) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_rhs_138114, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:331:100-137\n   #1  futhark/microgpt.fut:71:46-49\n   #2  futhark/microgpt.fut:331:53-139\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:15:29-44\n   #8  futhark/microgpt.fut:4:11-25\n   #9  futhark/microgpt.fut:15:15-45\n   #10 futhark/microgpt.fut:331:13-141\n   #11 futhark/microgpt.fut:582:5-76\n   #12 futhark/microgpt.fut:599:26-605:31\n   #13 futhark/microgpt.fut:633:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_138119;
                    double r_138121 = 0.0;
                    
                    for (int64_t i_138120 = 0; i_138120 < (int64_t) 16; i_138120++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_138122 = ((double *) mem_144727)[i_143255 * (int64_t) 16 + i_138120];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_138123 = ((double *) mem_param_143972.mem)[i_138120 * (int64_t) 16 + zt_rhs_138114];
                        
                        // futhark/microgpt.fut:331:75-137
                        
                        double zt_res_138124 = zt_lhs_138122 * zt_rhs_138123;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_138125 = r_138121 + zt_res_138124;
                        double r_tmp_146370 = zp_res_138125;
                        
                        r_138121 = r_tmp_146370;
                    }
                    defunc_0_lifted_lambda_res_138119 = r_138121;
                    ((double *) mem_144787)[i_143235] = defunc_0_lifted_lambda_res_138119;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_143243 = 0; i_143243 < (int64_t) 16; i_143243++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_138257;
                    double r_138259 = 0.0;
                    
                    for (int64_t i_138258 = 0; i_138258 < (int64_t) 4; i_138258++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_138260 = ((double *) mem_144170)[i_143268 * (int64_t) 64 + i_143255 * (int64_t) 4 + i_138258];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_138261 = ((double *) mem_144169)[i_143268 * (int64_t) 64 + i_143243 * (int64_t) 4 + i_138258];
                        
                        // futhark/microgpt.fut:332:119-178
                        
                        double zt_res_138262 = zt_lhs_138260 * zt_rhs_138261;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_138263 = r_138259 + zt_res_138262;
                        double r_tmp_146374 = zp_res_138263;
                        
                        r_138259 = r_tmp_146374;
                    }
                    defunc_0_lifted_lambda_res_138257 = r_138259;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_138270;
                    double r_138272 = 0.0;
                    
                    for (int64_t i_138271 = 0; i_138271 < (int64_t) 4; i_138271++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_138273 = ((double *) mem_144170)[i_143268 * (int64_t) 64 + i_143255 * (int64_t) 4 + i_138271];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_138274 = ((double *) mem_144169)[i_143268 * (int64_t) 64 + i_143243 * (int64_t) 4 + i_138271];
                        
                        // futhark/microgpt.fut:341:119-178
                        
                        double zt_res_138275 = zt_lhs_138273 * zt_rhs_138274;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_138276 = r_138272 + zt_res_138275;
                        double r_tmp_146375 = zp_res_138276;
                        
                        r_138272 = r_tmp_146375;
                    }
                    defunc_0_lifted_lambda_res_138270 = r_138272;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_138286;
                    double r_138288 = 0.0;
                    
                    for (int64_t i_138287 = 0; i_138287 < (int64_t) 4; i_138287++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_138289 = ((double *) mem_144170)[i_143268 * (int64_t) 64 + i_143255 * (int64_t) 4 + i_138287];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_138290 = ((double *) mem_144169)[i_143268 * (int64_t) 64 + i_143243 * (int64_t) 4 + i_138287];
                        
                        // futhark/microgpt.fut:357:119-178
                        
                        double zt_res_138291 = zt_lhs_138289 * zt_rhs_138290;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_138292 = r_138288 + zt_res_138291;
                        double r_tmp_146376 = zp_res_138292;
                        
                        r_138288 = r_tmp_146376;
                    }
                    defunc_0_lifted_lambda_res_138286 = r_138288;
                    ((double *) mem_144794)[i_143243] = defunc_0_lifted_lambda_res_138286;
                    ((double *) mem_144795)[i_143243] = defunc_0_lifted_lambda_res_138270;
                    ((double *) mem_144796)[i_143243] = defunc_0_lifted_lambda_res_138257;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_144767, i_143255 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144794, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_144768, i_143255 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144795, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_144769, i_143255 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144796, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_144770, i_143255 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144787, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_144743, i_143268 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_144767, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_144744, i_143268 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_144768, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_144745, i_143268 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_144769, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_144746, i_143268 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144770, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143321 = 0; i_143321 < (int64_t) 4; i_143321++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143302 = 0; i_143302 < (int64_t) 16; i_143302++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_143279 = 0; i_143279 < (int64_t) 16; i_143279++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_138848 = ((double *) mem_144745)[i_143321 * (int64_t) 256 + i_143302 * (int64_t) 16 + i_143279];
                    
                    // futhark/microgpt.fut:333:59-101
                    
                    double zs_res_138849 = zs_lhs_138848 / 2.0;
                    double zp_rhs_138850 = ((double *) masks_mem_143958.mem)[step_129423 * (int64_t) 256 + i_143302 * (int64_t) 16 + i_143279];
                    
                    // futhark/microgpt.fut:333:88-127
                    
                    double zp_res_138851 = zs_res_138849 + zp_rhs_138850;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_138858 = ((double *) mem_144744)[i_143321 * (int64_t) 256 + i_143302 * (int64_t) 16 + i_143279];
                    
                    // futhark/microgpt.fut:342:59-101
                    
                    double zs_res_138859 = zs_lhs_138858 / 2.0;
                    
                    // futhark/microgpt.fut:342:88-127
                    
                    double zp_res_138861 = zp_rhs_138850 + zs_res_138859;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_138871 = ((double *) mem_144743)[i_143321 * (int64_t) 256 + i_143302 * (int64_t) 16 + i_143279];
                    
                    // futhark/microgpt.fut:358:59-101
                    
                    double zs_res_138872 = zs_lhs_138871 / 2.0;
                    
                    // futhark/microgpt.fut:358:88-127
                    
                    double zp_res_138874 = zp_rhs_138850 + zs_res_138872;
                    
                    ((double *) mem_144917)[i_143279] = zp_res_138874;
                    ((double *) mem_144918)[i_143279] = zp_res_138861;
                    ((double *) mem_144919)[i_143279] = zp_res_138851;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_143287 = 0; i_143287 < (int64_t) 4; i_143287++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_138924 = ((double *) mem_144746)[i_143321 * (int64_t) 64 + i_143302 * (int64_t) 4 + i_143287];
                    
                    ((double *) mem_144938)[i_143287] = lifted_lambda_res_138924;
                    ((double *) mem_144939)[i_143287] = lifted_lambda_res_138924;
                }
                // futhark/microgpt.fut:4:11-25
                // futhark/microgpt.fut:4:11-25
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_144891, i_143302 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144939, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_144887, i_143302 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144938, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_144888, i_143302 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144917, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_144889, i_143302 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144939, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_144890, i_143302 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144918, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_144892, i_143302 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_144919, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_144851, i_143321 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144887, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_144852, i_143321 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_144888, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_144853, i_143321 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144889, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_144854, i_143321 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_144890, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_144855, i_143321 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_144891, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_144856, i_143321 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_144892, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143398 = 0; i_143398 < (int64_t) 4; i_143398++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143370 = 0; i_143370 < (int64_t) 16; i_143370++) {
                // futhark/microgpt.fut:115:13-33
                
                double defunc_0_reduce_res_142750;
                double defunc_0_reduce_res_142751;
                double defunc_0_reduce_res_142752;
                double defunc_0_reduce_res_142753;
                double defunc_0_reduce_res_142754;
                double redout_143330;
                double redout_143331;
                double redout_143332;
                double redout_143333;
                double redout_143334;
                
                redout_143330 = -INFINITY;
                redout_143331 = -INFINITY;
                redout_143332 = -INFINITY;
                redout_143333 = -INFINITY;
                redout_143334 = -INFINITY;
                for (int64_t i_143337 = 0; i_143337 < (int64_t) 16; i_143337++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_140831 = ((double *) mem_144856)[i_143398 * (int64_t) 256 + i_143370 * (int64_t) 16 + i_143337];
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_140842;
                    double r_140844 = 0.0;
                    
                    for (int64_t i_140843 = 0; i_140843 < (int64_t) 4; i_140843++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_140845 = ((double *) mem_144853)[i_143398 * (int64_t) 64 + i_143370 * (int64_t) 4 + i_140843];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_140846 = ((double *) mem_144168)[i_143398 * (int64_t) 64 + i_143337 * (int64_t) 4 + i_140843];
                        
                        // futhark/microgpt.fut:344:79-139
                        
                        double zt_res_140847 = zt_lhs_140845 * zt_rhs_140846;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_140848 = r_140844 + zt_res_140847;
                        double r_tmp_146419 = zp_res_140848;
                        
                        r_140844 = r_tmp_146419;
                    }
                    defunc_0_lifted_lambda_res_140842 = r_140844;
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_140856 = ((double *) mem_144854)[i_143398 * (int64_t) 256 + i_143370 * (int64_t) 16 + i_143337];
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_140904;
                    double r_140906 = 0.0;
                    
                    for (int64_t i_140905 = 0; i_140905 < (int64_t) 4; i_140905++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_140907 = ((double *) mem_144851)[i_143398 * (int64_t) 64 + i_143370 * (int64_t) 4 + i_140905];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_140908 = ((double *) mem_144168)[i_143398 * (int64_t) 64 + i_143337 * (int64_t) 4 + i_140905];
                        
                        // futhark/microgpt.fut:360:79-139
                        
                        double zt_res_140909 = zt_lhs_140907 * zt_rhs_140908;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_140910 = r_140906 + zt_res_140909;
                        double r_tmp_146420 = zp_res_140910;
                        
                        r_140906 = r_tmp_146420;
                    }
                    defunc_0_lifted_lambda_res_140904 = r_140906;
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_140921 = ((double *) mem_144852)[i_143398 * (int64_t) 256 + i_143370 * (int64_t) 16 + i_143337];
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_139951 = fmax64(lifted_lambda_res_140831, redout_143330);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_140027 = fmax64(lifted_lambda_res_140856, redout_143331);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_140052 = fmax64(lifted_lambda_res_140856, redout_143332);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_140133 = fmax64(lifted_lambda_res_140921, redout_143333);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_140166 = fmax64(lifted_lambda_res_140921, redout_143334);
                    
                    ((double *) mem_145094)[i_143337] = defunc_0_lifted_lambda_res_140904;
                    ((double *) mem_145095)[i_143337] = defunc_0_lifted_lambda_res_140842;
                    
                    double redout_tmp_146412 = max_res_139951;
                    double redout_tmp_146413 = max_res_140027;
                    double redout_tmp_146414 = max_res_140052;
                    double redout_tmp_146415 = max_res_140133;
                    double redout_tmp_146416 = max_res_140166;
                    
                    redout_143330 = redout_tmp_146412;
                    redout_143331 = redout_tmp_146413;
                    redout_143332 = redout_tmp_146414;
                    redout_143333 = redout_tmp_146415;
                    redout_143334 = redout_tmp_146416;
                }
                defunc_0_reduce_res_142750 = redout_143330;
                defunc_0_reduce_res_142751 = redout_143331;
                defunc_0_reduce_res_142752 = redout_143332;
                defunc_0_reduce_res_142753 = redout_143333;
                defunc_0_reduce_res_142754 = redout_143334;
                // futhark/microgpt.fut:335:80-90
                
                double neg_res_139952 = -defunc_0_reduce_res_142750;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_143342 = 0; i_143342 < (int64_t) 16; i_143342++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_139959 = ((double *) mem_144856)[i_143398 * (int64_t) 256 + i_143370 * (int64_t) 16 + i_143342];
                    
                    // futhark/microgpt.fut:335:46-90
                    
                    double zp_res_139960 = neg_res_139952 + zp_lhs_139959;
                    
                    // futhark/microgpt.fut:335:39-90
                    
                    double exp_res_139961 = futrts_exp64(zp_res_139960);
                    
                    ((double *) mem_145108)[i_143342] = exp_res_139961;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_139963;
                double r_139965 = 0.0;
                
                for (int64_t i_139964 = 0; i_139964 < (int64_t) 16; i_139964++) {
                    // futhark/microgpt.fut:336:38-50
                    
                    double lifted_lambda_res_139966 = ((double *) mem_145108)[i_139964];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_139967 = r_139965 + lifted_lambda_res_139966;
                    double r_tmp_146422 = zp_res_139967;
                    
                    r_139965 = r_tmp_146422;
                }
                defunc_0_lifted_lambda_res_139963 = r_139965;
                // futhark/microgpt.fut:337:57-69
                
                double zs_res_139968 = 1.0 / defunc_0_lifted_lambda_res_139963;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_143346 = 0; i_143346 < (int64_t) 16; i_143346++) {
                    // futhark/microgpt.fut:337:39-51
                    
                    double zt_lhs_139975 = ((double *) mem_145108)[i_143346];
                    
                    // futhark/microgpt.fut:337:39-69
                    
                    double zt_res_139976 = zs_res_139968 * zt_lhs_139975;
                    
                    ((double *) mem_145115)[i_143346] = zt_res_139976;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_143350 = 0; i_143350 < (int64_t) 16; i_143350++) {
                    // futhark/microgpt.fut:338:4-16
                    
                    double lifted_lambda_res_139984 = ((double *) mem_145115)[i_143350];
                    
                    ((double *) mem_145122)[i_143350] = lifted_lambda_res_139984;
                }
                // futhark/microgpt.fut:353:148-174
                
                double neg_res_140060 = -defunc_0_reduce_res_142752;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_140061;
                double r_140063 = 0.0;
                
                for (int64_t i_140062 = 0; i_140062 < (int64_t) 16; i_140062++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_140064 = ((double *) mem_144854)[i_143398 * (int64_t) 256 + i_143370 * (int64_t) 16 + i_140062];
                    
                    // futhark/microgpt.fut:353:114-174
                    
                    double zp_res_140065 = neg_res_140060 + zp_lhs_140064;
                    
                    // futhark/microgpt.fut:353:107-174
                    
                    double neg_res_140066 = -zp_res_140065;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_140067 = fmax64(0.0, neg_res_140066);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_140068 = fsignum64(max_res_140067);
                    
                    // futhark/microgpt.fut:353:88-177
                    
                    double neg_res_140069 = -sgn_res_140068;
                    
                    // futhark/microgpt.fut:353:79-178
                    
                    double zp_res_140070 = 1.0 + neg_res_140069;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_140071 = r_140063 + zp_res_140070;
                    double r_tmp_146425 = zp_res_140071;
                    
                    r_140063 = r_tmp_146425;
                }
                defunc_0_lifted_lambda_res_140061 = r_140063;
                // futhark/microgpt.fut:353:48-181
                
                double zs_res_140072 = 1.0 / defunc_0_lifted_lambda_res_140061;
                
                // futhark/microgpt.fut:369:148-174
                
                double neg_res_140174 = -defunc_0_reduce_res_142754;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_140175;
                double r_140177 = 0.0;
                
                for (int64_t i_140176 = 0; i_140176 < (int64_t) 16; i_140176++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_140178 = ((double *) mem_144852)[i_143398 * (int64_t) 256 + i_143370 * (int64_t) 16 + i_140176];
                    
                    // futhark/microgpt.fut:369:114-174
                    
                    double zp_res_140179 = neg_res_140174 + zp_lhs_140178;
                    
                    // futhark/microgpt.fut:369:107-174
                    
                    double neg_res_140180 = -zp_res_140179;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_140181 = fmax64(0.0, neg_res_140180);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_140182 = fsignum64(max_res_140181);
                    
                    // futhark/microgpt.fut:369:88-177
                    
                    double neg_res_140183 = -sgn_res_140182;
                    
                    // futhark/microgpt.fut:369:79-178
                    
                    double zp_res_140184 = 1.0 + neg_res_140183;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_140185 = r_140177 + zp_res_140184;
                    double r_tmp_146426 = zp_res_140185;
                    
                    r_140177 = r_tmp_146426;
                }
                defunc_0_lifted_lambda_res_140175 = r_140177;
                // futhark/microgpt.fut:369:48-181
                
                double zs_res_140186 = 1.0 / defunc_0_lifted_lambda_res_140175;
                
                ((double *) mem_145055)[i_143370] = zs_res_140186;
                ((double *) mem_145056)[i_143370] = defunc_0_reduce_res_142754;
                ((double *) mem_145057)[i_143370] = defunc_0_reduce_res_142753;
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_145058, i_143370 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145094, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                ((double *) mem_145059)[i_143370] = zs_res_140072;
                ((double *) mem_145060)[i_143370] = defunc_0_reduce_res_142752;
                ((double *) mem_145061)[i_143370] = defunc_0_reduce_res_142751;
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_145062, i_143370 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145095, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_145063, i_143370 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145122, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145007, i_143398 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145055, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145008, i_143398 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145056, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145009, i_143398 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145057, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_145010, i_143398 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145058, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145011, i_143398 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145059, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145012, i_143398 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145060, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145013, i_143398 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145061, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_145014, i_143398 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145062, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_145015, i_143398 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145063, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143442 = 0; i_143442 < (int64_t) 4; i_143442++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143429 = 0; i_143429 < (int64_t) 16; i_143429++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_141376 = ((double *) mem_145013)[i_143442 * (int64_t) 16 + i_143429];
                
                // futhark/microgpt.fut:346:99-125
                
                double neg_res_141377 = -neg_arg0_141376;
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_141411 = ((double *) mem_145009)[i_143442 * (int64_t) 16 + i_143429];
                
                // futhark/microgpt.fut:362:99-125
                
                double neg_res_141412 = -neg_arg0_141411;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_143416 = 0; i_143416 < (int64_t) 16; i_143416++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_141546 = ((double *) mem_144854)[i_143442 * (int64_t) 256 + i_143429 * (int64_t) 16 + i_143416];
                    
                    // futhark/microgpt.fut:346:65-125
                    
                    double zp_res_141547 = neg_res_141377 + zp_lhs_141546;
                    
                    // futhark/microgpt.fut:346:58-125
                    
                    double exp_res_141548 = futrts_exp64(zp_res_141547);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_141555 = ((double *) mem_145014)[i_143442 * (int64_t) 256 + i_143429 * (int64_t) 16 + i_143416];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_141565 = ((double *) mem_144852)[i_143442 * (int64_t) 256 + i_143429 * (int64_t) 16 + i_143416];
                    
                    // futhark/microgpt.fut:362:65-125
                    
                    double zp_res_141566 = neg_res_141412 + zp_lhs_141565;
                    
                    // futhark/microgpt.fut:362:58-125
                    
                    double exp_res_141567 = futrts_exp64(zp_res_141566);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_141579 = ((double *) mem_145010)[i_143442 * (int64_t) 256 + i_143429 * (int64_t) 16 + i_143416];
                    
                    ((double *) mem_145242)[i_143416] = lifted_lambda_res_141579;
                    ((double *) mem_145243)[i_143416] = exp_res_141567;
                    ((double *) mem_145244)[i_143416] = lifted_lambda_res_141555;
                    ((double *) mem_145245)[i_143416] = exp_res_141548;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_145222, i_143429 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145242, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_145223, i_143429 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145243, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_145224, i_143429 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145244, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_145225, i_143429 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145245, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_145198, i_143442 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145222, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_145199, i_143442 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145223, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_145200, i_143442 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145224, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_145201, i_143442 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145225, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143468 = 0; i_143468 < (int64_t) 4; i_143468++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143455 = 0; i_143455 < (int64_t) 16; i_143455++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_141700;
                double r_141702 = 0.0;
                
                for (int64_t i_141701 = 0; i_141701 < (int64_t) 16; i_141701++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_141703 = ((double *) mem_145201)[i_143468 * (int64_t) 256 + i_143455 * (int64_t) 16 + i_141701];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_141704 = r_141702 + lifted_lambda_res_141703;
                    double r_tmp_146447 = zp_res_141704;
                    
                    r_141702 = r_tmp_146447;
                }
                defunc_0_lifted_lambda_res_141700 = r_141702;
                // futhark/microgpt.fut:349:155-200
                
                double zt_res_141712 = defunc_0_lifted_lambda_res_141700 * defunc_0_lifted_lambda_res_141700;
                
                // futhark/microgpt.fut:349:146-200
                
                double zs_res_141713 = 1.0 / zt_res_141712;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_141714;
                double r_141716 = 0.0;
                
                for (int64_t i_141715 = 0; i_141715 < (int64_t) 16; i_141715++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_141717 = ((double *) mem_145200)[i_143468 * (int64_t) 256 + i_143455 * (int64_t) 16 + i_141715];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_141718 = ((double *) mem_145201)[i_143468 * (int64_t) 256 + i_143455 * (int64_t) 16 + i_141715];
                    
                    // futhark/microgpt.fut:349:78-139
                    
                    double zt_res_141719 = zt_lhs_141717 * zt_rhs_141718;
                    
                    // futhark/microgpt.fut:349:107-200
                    
                    double zt_res_141720 = zs_res_141713 * zt_res_141719;
                    
                    // futhark/microgpt.fut:349:70-200
                    
                    double neg_res_141721 = -zt_res_141720;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_141722 = r_141716 + neg_res_141721;
                    double r_tmp_146448 = zp_res_141722;
                    
                    r_141716 = r_tmp_146448;
                }
                defunc_0_lifted_lambda_res_141714 = r_141716;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_141733;
                double r_141735 = 0.0;
                
                for (int64_t i_141734 = 0; i_141734 < (int64_t) 16; i_141734++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_141736 = ((double *) mem_145199)[i_143468 * (int64_t) 256 + i_143455 * (int64_t) 16 + i_141734];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_141737 = r_141735 + lifted_lambda_res_141736;
                    double r_tmp_146449 = zp_res_141737;
                    
                    r_141735 = r_tmp_146449;
                }
                defunc_0_lifted_lambda_res_141733 = r_141735;
                // futhark/microgpt.fut:365:155-200
                
                double zt_res_141745 = defunc_0_lifted_lambda_res_141733 * defunc_0_lifted_lambda_res_141733;
                
                // futhark/microgpt.fut:365:146-200
                
                double zs_res_141746 = 1.0 / zt_res_141745;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_141747;
                double r_141749 = 0.0;
                
                for (int64_t i_141748 = 0; i_141748 < (int64_t) 16; i_141748++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_141750 = ((double *) mem_145198)[i_143468 * (int64_t) 256 + i_143455 * (int64_t) 16 + i_141748];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_141751 = ((double *) mem_145199)[i_143468 * (int64_t) 256 + i_143455 * (int64_t) 16 + i_141748];
                    
                    // futhark/microgpt.fut:365:78-139
                    
                    double zt_res_141752 = zt_lhs_141750 * zt_rhs_141751;
                    
                    // futhark/microgpt.fut:365:107-200
                    
                    double zt_res_141753 = zs_res_141746 * zt_res_141752;
                    
                    // futhark/microgpt.fut:365:70-200
                    
                    double neg_res_141754 = -zt_res_141753;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_141755 = r_141749 + neg_res_141754;
                    double r_tmp_146450 = zp_res_141755;
                    
                    r_141749 = r_tmp_146450;
                }
                defunc_0_lifted_lambda_res_141747 = r_141749;
                ((double *) mem_145326)[i_143455] = defunc_0_lifted_lambda_res_141747;
                ((double *) mem_145327)[i_143455] = defunc_0_lifted_lambda_res_141733;
                ((double *) mem_145328)[i_143455] = defunc_0_lifted_lambda_res_141714;
                ((double *) mem_145329)[i_143455] = defunc_0_lifted_lambda_res_141700;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145306, i_143468 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145326, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145307, i_143468 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145327, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145308, i_143468 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145328, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145309, i_143468 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145329, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143491 = 0; i_143491 < (int64_t) 4; i_143491++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143484 = 0; i_143484 < (int64_t) 16; i_143484++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_141781 = ((double *) mem_145309)[i_143491 * (int64_t) 16 + i_143484];
                
                // futhark/microgpt.fut:350:93-121
                
                double zs_res_141782 = 1.0 / zs_rhs_141781;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_141783 = ((double *) mem_145308)[i_143491 * (int64_t) 16 + i_143484];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_141802 = ((double *) mem_145306)[i_143491 * (int64_t) 16 + i_143484];
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_141800 = ((double *) mem_145307)[i_143491 * (int64_t) 16 + i_143484];
                
                // futhark/microgpt.fut:366:93-121
                
                double zs_res_141801 = 1.0 / zs_rhs_141800;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_143477 = 0; i_143477 < (int64_t) 16; i_143477++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_141830 = ((double *) mem_145200)[i_143491 * (int64_t) 256 + i_143484 * (int64_t) 16 + i_143477];
                    
                    // futhark/microgpt.fut:350:59-121
                    
                    double zt_res_141831 = zs_res_141782 * zt_lhs_141830;
                    
                    // futhark/microgpt.fut:350:88-148
                    
                    double zp_res_141832 = zp_rhs_141783 + zt_res_141831;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_141839 = ((double *) mem_145198)[i_143491 * (int64_t) 256 + i_143484 * (int64_t) 16 + i_143477];
                    
                    // futhark/microgpt.fut:366:59-121
                    
                    double zt_res_141840 = zs_res_141801 * zt_lhs_141839;
                    
                    // futhark/microgpt.fut:366:88-148
                    
                    double zp_res_141841 = zp_rhs_141802 + zt_res_141840;
                    
                    ((double *) mem_145392)[i_143477] = zp_res_141841;
                    ((double *) mem_145393)[i_143477] = zp_res_141832;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_145382, i_143484 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145392, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_145383, i_143484 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145393, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_145370, i_143491 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145382, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_145371, i_143491 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145383, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143505 = 0; i_143505 < (int64_t) 4; i_143505++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143498 = 0; i_143498 < (int64_t) 16; i_143498++) {
                double f_elem_141861 = ((double *) mem_145013)[i_143505 * (int64_t) 16 + i_143498];
                double f_elem_141863 = ((double *) mem_145009)[i_143505 * (int64_t) 16 + i_143498];
                
                // futhark/microgpt.fut:351:119-145
                
                double neg_res_141868 = -f_elem_141861;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_141869;
                double r_141871 = 0.0;
                
                for (int64_t i_141870 = 0; i_141870 < (int64_t) 16; i_141870++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_141872 = ((double *) mem_144854)[i_143505 * (int64_t) 256 + i_143498 * (int64_t) 16 + i_141870];
                    
                    // futhark/microgpt.fut:351:85-145
                    
                    double zp_res_141873 = neg_res_141868 + zp_lhs_141872;
                    
                    // futhark/microgpt.fut:351:78-145
                    
                    double exp_res_141874 = futrts_exp64(zp_res_141873);
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_141875 = ((double *) mem_145371)[i_143505 * (int64_t) 256 + i_143498 * (int64_t) 16 + i_141870];
                    
                    // futhark/microgpt.fut:351:78-181
                    
                    double zt_res_141876 = exp_res_141874 * zt_rhs_141875;
                    
                    // futhark/microgpt.fut:351:70-181
                    
                    double neg_res_141877 = -zt_res_141876;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_141878 = r_141871 + neg_res_141877;
                    double r_tmp_146461 = zp_res_141878;
                    
                    r_141871 = r_tmp_146461;
                }
                defunc_0_lifted_lambda_res_141869 = r_141871;
                // futhark/microgpt.fut:367:119-145
                
                double neg_res_141886 = -f_elem_141863;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_141887;
                double r_141889 = 0.0;
                
                for (int64_t i_141888 = 0; i_141888 < (int64_t) 16; i_141888++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_141890 = ((double *) mem_144852)[i_143505 * (int64_t) 256 + i_143498 * (int64_t) 16 + i_141888];
                    
                    // futhark/microgpt.fut:367:85-145
                    
                    double zp_res_141891 = neg_res_141886 + zp_lhs_141890;
                    
                    // futhark/microgpt.fut:367:78-145
                    
                    double exp_res_141892 = futrts_exp64(zp_res_141891);
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_141893 = ((double *) mem_145370)[i_143505 * (int64_t) 256 + i_143498 * (int64_t) 16 + i_141888];
                    
                    // futhark/microgpt.fut:367:78-181
                    
                    double zt_res_141894 = exp_res_141892 * zt_rhs_141893;
                    
                    // futhark/microgpt.fut:367:70-181
                    
                    double neg_res_141895 = -zt_res_141894;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_141896 = r_141889 + neg_res_141895;
                    double r_tmp_146462 = zp_res_141896;
                    
                    r_141889 = r_tmp_146462;
                }
                defunc_0_lifted_lambda_res_141887 = r_141889;
                ((double *) mem_145434)[i_143498] = defunc_0_lifted_lambda_res_141887;
                ((double *) mem_145435)[i_143498] = defunc_0_lifted_lambda_res_141869;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145424, i_143505 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145434, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145425, i_143505 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145435, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143526 = 0; i_143526 < (int64_t) 4; i_143526++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143519 = 0; i_143519 < (int64_t) 16; i_143519++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_141916 = ((double *) mem_145013)[i_143526 * (int64_t) 16 + i_143519];
                
                // futhark/microgpt.fut:354:101-127
                
                double neg_res_141917 = -neg_arg0_141916;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_141918 = ((double *) mem_145425)[i_143526 * (int64_t) 16 + i_143519];
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_141919 = ((double *) mem_145012)[i_143526 * (int64_t) 16 + i_143519];
                
                // futhark/microgpt.fut:354:266-292
                
                double neg_res_141920 = -neg_arg0_141919;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_141921 = ((double *) mem_145011)[i_143526 * (int64_t) 16 + i_143519];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_141954 = ((double *) mem_145007)[i_143526 * (int64_t) 16 + i_143519];
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_141952 = ((double *) mem_145008)[i_143526 * (int64_t) 16 + i_143519];
                
                // futhark/microgpt.fut:370:266-292
                
                double neg_res_141953 = -neg_arg0_141952;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_141951 = ((double *) mem_145424)[i_143526 * (int64_t) 16 + i_143519];
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_141949 = ((double *) mem_145009)[i_143526 * (int64_t) 16 + i_143519];
                
                // futhark/microgpt.fut:370:101-127
                
                double neg_res_141950 = -neg_arg0_141949;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_143512 = 0; i_143512 < (int64_t) 16; i_143512++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_141993 = ((double *) mem_144854)[i_143526 * (int64_t) 256 + i_143519 * (int64_t) 16 + i_143512];
                    
                    // futhark/microgpt.fut:354:67-127
                    
                    double zp_res_141994 = neg_res_141917 + zp_lhs_141993;
                    
                    // futhark/microgpt.fut:354:60-127
                    
                    double exp_res_141995 = futrts_exp64(zp_res_141994);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_141996 = ((double *) mem_145371)[i_143526 * (int64_t) 256 + i_143519 * (int64_t) 16 + i_143512];
                    
                    // futhark/microgpt.fut:354:60-163
                    
                    double zt_res_141997 = exp_res_141995 * zt_rhs_141996;
                    
                    // futhark/microgpt.fut:354:232-292
                    
                    double zp_res_141998 = neg_res_141920 + zp_lhs_141993;
                    
                    // futhark/microgpt.fut:354:225-292
                    
                    double neg_res_141999 = -zp_res_141998;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_142000 = fmax64(0.0, neg_res_141999);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_142001 = fsignum64(max_res_142000);
                    
                    // futhark/microgpt.fut:354:206-295
                    
                    double neg_res_142002 = -sgn_res_142001;
                    
                    // futhark/microgpt.fut:354:197-296
                    
                    double zp_res_142003 = 1.0 + neg_res_142002;
                    
                    // futhark/microgpt.fut:354:171-296
                    
                    double zt_res_142004 = zt_lhs_141918 * zp_res_142003;
                    
                    // futhark/microgpt.fut:354:192-324
                    
                    double zt_res_142005 = zt_rhs_141921 * zt_res_142004;
                    
                    // futhark/microgpt.fut:354:131-324
                    
                    double zp_res_142006 = zt_res_141997 + zt_res_142005;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_142013 = ((double *) mem_144852)[i_143526 * (int64_t) 256 + i_143519 * (int64_t) 16 + i_143512];
                    
                    // futhark/microgpt.fut:370:67-127
                    
                    double zp_res_142014 = neg_res_141950 + zp_lhs_142013;
                    
                    // futhark/microgpt.fut:370:60-127
                    
                    double exp_res_142015 = futrts_exp64(zp_res_142014);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_142016 = ((double *) mem_145370)[i_143526 * (int64_t) 256 + i_143519 * (int64_t) 16 + i_143512];
                    
                    // futhark/microgpt.fut:370:60-163
                    
                    double zt_res_142017 = exp_res_142015 * zt_rhs_142016;
                    
                    // futhark/microgpt.fut:370:232-292
                    
                    double zp_res_142018 = neg_res_141953 + zp_lhs_142013;
                    
                    // futhark/microgpt.fut:370:225-292
                    
                    double neg_res_142019 = -zp_res_142018;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_142020 = fmax64(0.0, neg_res_142019);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_142021 = fsignum64(max_res_142020);
                    
                    // futhark/microgpt.fut:370:206-295
                    
                    double neg_res_142022 = -sgn_res_142021;
                    
                    // futhark/microgpt.fut:370:197-296
                    
                    double zp_res_142023 = 1.0 + neg_res_142022;
                    
                    // futhark/microgpt.fut:370:171-296
                    
                    double zt_res_142024 = zt_lhs_141951 * zp_res_142023;
                    
                    // futhark/microgpt.fut:370:192-324
                    
                    double zt_res_142025 = zt_rhs_141954 * zt_res_142024;
                    
                    // futhark/microgpt.fut:370:131-324
                    
                    double zp_res_142026 = zt_res_142017 + zt_res_142025;
                    
                    ((double *) mem_145478)[i_143512] = zp_res_142026;
                    ((double *) mem_145479)[i_143512] = zp_res_142006;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_145468, i_143519 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145478, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_145469, i_143519 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145479, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_145456, i_143526 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145468, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_145457, i_143526 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145469, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143547 = 0; i_143547 < (int64_t) 4; i_143547++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143540 = 0; i_143540 < (int64_t) 16; i_143540++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_143533 = 0; i_143533 < (int64_t) 16; i_143533++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_142091 = ((double *) mem_145457)[i_143547 * (int64_t) 256 + i_143540 * (int64_t) 16 + i_143533];
                    
                    // futhark/microgpt.fut:355:58-100
                    
                    double zs_res_142092 = zs_lhs_142091 / 2.0;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_142099 = ((double *) mem_145456)[i_143547 * (int64_t) 256 + i_143540 * (int64_t) 16 + i_143533];
                    
                    // futhark/microgpt.fut:371:58-100
                    
                    double zs_res_142100 = zs_lhs_142099 / 2.0;
                    
                    ((double *) mem_145532)[i_143533] = zs_res_142100;
                    ((double *) mem_145533)[i_143533] = zs_res_142092;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_145522, i_143540 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145532, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_145523, i_143540 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145533, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_145510, i_143547 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145522, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_145511, i_143547 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145523, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143571 = 0; i_143571 < (int64_t) 16; i_143571++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143558 = 0; i_143558 < (int64_t) 16; i_143558++) {
                // futhark/microgpt.fut:340:40-43
                
                int64_t zt_lhs_141151 = sdiv64(i_143558, (int64_t) 4);
                
                // futhark/microgpt.fut:340:27-45
                
                bool x_141152 = sle64((int64_t) 0, zt_lhs_141151);
                
                // futhark/microgpt.fut:340:27-45
                
                bool y_141153 = slt64(zt_lhs_141151, (int64_t) 4);
                
                // futhark/microgpt.fut:340:27-45
                
                bool bounds_check_141154 = x_141152 && y_141153;
                
                // futhark/microgpt.fut:340:27-45
                
                bool index_certs_141155;
                
                if (!bounds_check_141154) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_141151, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:340:27-45\n   #1  futhark/microgpt.fut:71:46-49\n   #2  futhark/microgpt.fut:340:5-108\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:332:13-340:114\n   #8  futhark/microgpt.fut:582:5-76\n   #9  futhark/microgpt.fut:599:26-605:31\n   #10 futhark/microgpt.fut:633:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:340:62-65
                
                int64_t zt_lhs_141156 = smod64(i_143558, (int64_t) 4);
                
                // futhark/microgpt.fut:340:27-67
                
                bool x_141157 = sle64((int64_t) 0, zt_lhs_141156);
                
                // futhark/microgpt.fut:340:27-67
                
                bool y_141158 = slt64(zt_lhs_141156, (int64_t) 4);
                
                // futhark/microgpt.fut:340:27-67
                
                bool bounds_check_141159 = x_141157 && y_141158;
                
                // futhark/microgpt.fut:340:27-67
                
                bool index_certs_141160;
                
                if (!bounds_check_141159) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zt_lhs_141156, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:340:27-67\n   #1  futhark/microgpt.fut:71:46-49\n   #2  futhark/microgpt.fut:340:5-108\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:27-39\n   #5  futhark/microgpt.fut:4:11-25\n   #6  futhark/microgpt.fut:9:13-40\n   #7  futhark/microgpt.fut:332:13-340:114\n   #8  futhark/microgpt.fut:582:5-76\n   #9  futhark/microgpt.fut:599:26-605:31\n   #10 futhark/microgpt.fut:633:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_141161;
                double r_141163 = 0.0;
                
                for (int64_t i_141162 = 0; i_141162 < (int64_t) 16; i_141162++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_141164 = ((double *) mem_144855)[zt_lhs_141151 * (int64_t) 64 + i_141162 * (int64_t) 4 + zt_lhs_141156];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_141165 = ((double *) mem_145015)[zt_lhs_141151 * (int64_t) 256 + i_141162 * (int64_t) 16 + i_143571];
                    
                    // futhark/microgpt.fut:340:27-106
                    
                    double zt_res_141166 = zt_lhs_141164 * zt_rhs_141165;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_141167 = r_141163 + zt_res_141166;
                    double r_tmp_146483 = zp_res_141167;
                    
                    r_141163 = r_tmp_146483;
                }
                defunc_0_lifted_lambda_res_141161 = r_141163;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_141180;
                double r_141182 = 0.0;
                
                for (int64_t i_141181 = 0; i_141181 < (int64_t) 16; i_141181++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_141183 = ((double *) mem_145511)[zt_lhs_141151 * (int64_t) 256 + i_141181 * (int64_t) 16 + i_143571];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_141184 = ((double *) mem_144170)[zt_lhs_141151 * (int64_t) 64 + i_141181 * (int64_t) 4 + zt_lhs_141156];
                    
                    // futhark/microgpt.fut:356:27-105
                    
                    double zt_res_141185 = zt_lhs_141183 * zt_rhs_141184;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_141186 = r_141182 + zt_res_141185;
                    double r_tmp_146484 = zp_res_141186;
                    
                    r_141182 = r_tmp_146484;
                }
                defunc_0_lifted_lambda_res_141180 = r_141182;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_141202;
                double r_141204 = 0.0;
                
                for (int64_t i_141203 = 0; i_141203 < (int64_t) 16; i_141203++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_141205 = ((double *) mem_145510)[zt_lhs_141151 * (int64_t) 256 + i_143571 * (int64_t) 16 + i_141203];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_141206 = ((double *) mem_144169)[zt_lhs_141151 * (int64_t) 64 + i_141203 * (int64_t) 4 + zt_lhs_141156];
                    
                    // futhark/microgpt.fut:372:27-105
                    
                    double zt_res_141207 = zt_lhs_141205 * zt_rhs_141206;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_141208 = r_141204 + zt_res_141207;
                    double r_tmp_146485 = zp_res_141208;
                    
                    r_141204 = r_tmp_146485;
                }
                defunc_0_lifted_lambda_res_141202 = r_141204;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_141220;
                double r_141222 = 0.0;
                
                for (int64_t i_141221 = 0; i_141221 < (int64_t) 16; i_141221++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_141223 = ((double *) mem_144727)[i_141221 * (int64_t) 16 + i_143571];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_141224 = ((double *) mem_144249)[i_141221 * (int64_t) 16 + i_143558];
                    
                    // futhark/microgpt.fut:404:68-112
                    
                    double zt_res_141225 = zt_lhs_141223 * zt_rhs_141224;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_141226 = r_141222 + zt_res_141225;
                    double r_tmp_146486 = zp_res_141226;
                    
                    r_141222 = r_tmp_146486;
                }
                defunc_0_lifted_lambda_res_141220 = r_141222;
                ((double *) mem_145584)[i_143558] = defunc_0_lifted_lambda_res_141220;
                ((double *) mem_145585)[i_143558] = defunc_0_lifted_lambda_res_141202;
                ((double *) mem_145586)[i_143558] = defunc_0_lifted_lambda_res_141180;
                ((double *) mem_145587)[i_143558] = defunc_0_lifted_lambda_res_141161;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145564, i_143571 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145584, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145565, i_143571 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145585, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145566, i_143571 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145586, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145567, i_143571 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145587, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143582 = 0; i_143582 < (int64_t) 16; i_143582++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143578 = 0; i_143578 < (int64_t) 16; i_143578++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_131445;
                double r_131447 = 0.0;
                
                for (int64_t i_131446 = 0; i_131446 < (int64_t) 16; i_131446++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_131448 = ((double *) mem_145567)[i_143582 * (int64_t) 16 + i_131446];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_131449 = ((double *) mem_param_143992.mem)[i_131446 * (int64_t) 16 + i_143578];
                    
                    // futhark/microgpt.fut:375:73-118
                    
                    double zt_res_131450 = zt_lhs_131448 * zt_rhs_131449;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_131451 = r_131447 + zt_res_131450;
                    double r_tmp_146489 = zp_res_131451;
                    
                    r_131447 = r_tmp_146489;
                }
                defunc_0_lifted_lambda_res_131445 = r_131447;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_131452;
                double r_131454 = 0.0;
                
                for (int64_t i_131453 = 0; i_131453 < (int64_t) 16; i_131453++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_131455 = ((double *) mem_145566)[i_143582 * (int64_t) 16 + i_131453];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_131456 = ((double *) mem_param_143968.mem)[i_131453 * (int64_t) 16 + i_143578];
                    
                    // futhark/microgpt.fut:375:149-194
                    
                    double zt_res_131457 = zt_lhs_131455 * zt_rhs_131456;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_131458 = r_131454 + zt_res_131457;
                    double r_tmp_146490 = zp_res_131458;
                    
                    r_131454 = r_tmp_146490;
                }
                defunc_0_lifted_lambda_res_131452 = r_131454;
                // futhark/microgpt.fut:375:51-196
                
                double zp_res_131459 = defunc_0_lifted_lambda_res_131445 + defunc_0_lifted_lambda_res_131452;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_131460;
                double r_131462 = 0.0;
                
                for (int64_t i_131461 = 0; i_131461 < (int64_t) 16; i_131461++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_131463 = ((double *) mem_145565)[i_143582 * (int64_t) 16 + i_131461];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_131464 = ((double *) mem_param_143980.mem)[i_131461 * (int64_t) 16 + i_143578];
                    
                    // futhark/microgpt.fut:375:226-271
                    
                    double zt_res_131465 = zt_lhs_131463 * zt_rhs_131464;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_131466 = r_131462 + zt_res_131465;
                    double r_tmp_146491 = zp_res_131466;
                    
                    r_131462 = r_tmp_146491;
                }
                defunc_0_lifted_lambda_res_131460 = r_131462;
                // futhark/microgpt.fut:375:122-273
                
                double zp_res_131467 = zp_res_131459 + defunc_0_lifted_lambda_res_131460;
                
                ((double *) mem_145633)[i_143578] = zp_res_131467;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145628, i_143582 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145633, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143604 = 0; i_143604 < (int64_t) 16; i_143604++) {
            // futhark/microgpt.fut:374:47-59
            
            double zp_lhs_136667 = ((double *) mem_144138)[i_143604];
            
            // futhark/microgpt.fut:374:47-87
            
            double zp_res_136668 = 1.0e-5 + zp_lhs_136667;
            
            // futhark/microgpt.fut:374:39-87
            
            double sqrt_res_136669 = futrts_sqrt64(zp_res_136668);
            
            // futhark/microgpt.fut:376:128-157
            
            double zt_res_136677 = sqrt_res_136669 * sqrt_res_136669;
            
            // futhark/microgpt.fut:376:119-157
            
            double zs_res_136678 = 1.0 / zt_res_136677;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_136679;
            double r_136681 = 0.0;
            
            for (int64_t i_136680 = 0; i_136680 < (int64_t) 16; i_136680++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_136682 = ((double *) mem_145628)[i_143604 * (int64_t) 16 + i_136680];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_136683 = ((double *) mem_144103)[i_143604 * (int64_t) 16 + i_136680];
                
                // futhark/microgpt.fut:376:69-112
                
                double zt_res_136684 = zt_lhs_136682 * zt_rhs_136683;
                
                // futhark/microgpt.fut:376:90-157
                
                double zt_res_136685 = zs_res_136678 * zt_res_136684;
                
                // futhark/microgpt.fut:376:61-157
                
                double neg_res_136686 = -zt_res_136685;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_136687 = r_136681 + neg_res_136686;
                double r_tmp_146497 = zp_res_136687;
                
                r_136681 = r_tmp_146497;
            }
            defunc_0_lifted_lambda_res_136679 = r_136681;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143590 = 0; i_143590 < (int64_t) 16; i_143590++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_142167;
                double r_142169 = 0.0;
                
                for (int64_t i_142168 = 0; i_142168 < (int64_t) 16; i_142168++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_142170 = ((double *) mem_145565)[i_142168 * (int64_t) 16 + i_143604];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_142171 = ((double *) mem_144139)[i_142168 * (int64_t) 16 + i_143590];
                    
                    // futhark/microgpt.fut:401:68-111
                    
                    double zt_res_142172 = zt_lhs_142170 * zt_rhs_142171;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_142173 = r_142169 + zt_res_142172;
                    double r_tmp_146501 = zp_res_142173;
                    
                    r_142169 = r_tmp_146501;
                }
                defunc_0_lifted_lambda_res_142167 = r_142169;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_142180;
                double r_142182 = 0.0;
                
                for (int64_t i_142181 = 0; i_142181 < (int64_t) 16; i_142181++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_142183 = ((double *) mem_145566)[i_142181 * (int64_t) 16 + i_143604];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_142184 = ((double *) mem_144139)[i_142181 * (int64_t) 16 + i_143590];
                    
                    // futhark/microgpt.fut:402:68-111
                    
                    double zt_res_142185 = zt_lhs_142183 * zt_rhs_142184;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_142186 = r_142182 + zt_res_142185;
                    double r_tmp_146502 = zp_res_142186;
                    
                    r_142182 = r_tmp_146502;
                }
                defunc_0_lifted_lambda_res_142180 = r_142182;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_142196;
                double r_142198 = 0.0;
                
                for (int64_t i_142197 = 0; i_142197 < (int64_t) 16; i_142197++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_142199 = ((double *) mem_145567)[i_142197 * (int64_t) 16 + i_143604];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_142200 = ((double *) mem_144139)[i_142197 * (int64_t) 16 + i_143590];
                    
                    // futhark/microgpt.fut:403:68-111
                    
                    double zt_res_142201 = zt_lhs_142199 * zt_rhs_142200;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_142202 = r_142198 + zt_res_142201;
                    double r_tmp_146503 = zp_res_142202;
                    
                    r_142198 = r_tmp_146503;
                }
                defunc_0_lifted_lambda_res_142196 = r_142198;
                ((double *) mem_145667)[i_143590] = defunc_0_lifted_lambda_res_142196;
                ((double *) mem_145668)[i_143590] = defunc_0_lifted_lambda_res_142180;
                ((double *) mem_145669)[i_143590] = defunc_0_lifted_lambda_res_142167;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145644, i_143604 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145667, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145645, i_143604 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145668, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145646, i_143604 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145669, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            ((double *) mem_145647)[i_143604] = defunc_0_lifted_lambda_res_136679;
            ((double *) mem_145648)[i_143604] = sqrt_res_136669;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143612 = 0; i_143612 < (int64_t) 16; i_143612++) {
            // futhark/microgpt.fut:377:39-51
            
            double zt_lhs_131495 = ((double *) mem_145647)[i_143612];
            
            // futhark/microgpt.fut:377:93-105
            
            double zp_lhs_131496 = ((double *) mem_144138)[i_143612];
            
            // futhark/microgpt.fut:377:93-133
            
            double zp_res_131497 = 1.0e-5 + zp_lhs_131496;
            
            // futhark/microgpt.fut:377:85-133
            
            double sqrt_res_131498 = futrts_sqrt64(zp_res_131497);
            
            // futhark/microgpt.fut:377:71-135
            
            double zt_res_131499 = 2.0 * sqrt_res_131498;
            
            // futhark/microgpt.fut:377:57-135
            
            double zs_res_131500 = 1.0 / zt_res_131499;
            
            // futhark/microgpt.fut:377:39-135
            
            double zt_res_131501 = zt_lhs_131495 * zs_res_131500;
            
            ((double *) mem_145706)[i_143612] = zt_res_131501;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143620 = 0; i_143620 < (int64_t) 16; i_143620++) {
            // futhark/microgpt.fut:378:98-110
            
            double zs_rhs_131509 = ((double *) mem_145648)[i_143620];
            
            // futhark/microgpt.fut:378:90-110
            
            double zs_res_131510 = 1.0 / zs_rhs_131509;
            
            // futhark/microgpt.fut:378:120-132
            
            double zs_lhs_131511 = ((double *) mem_145706)[i_143620];
            
            // futhark/microgpt.fut:378:120-147
            
            double zs_res_131512 = zs_lhs_131511 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143616 = 0; i_143616 < (int64_t) 16; i_143616++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_131519 = ((double *) mem_144727)[i_143620 * (int64_t) 16 + i_143616];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_131520 = ((double *) mem_145628)[i_143620 * (int64_t) 16 + i_143616];
                
                // futhark/microgpt.fut:378:64-110
                
                double zt_res_131521 = zs_res_131510 * zt_lhs_131520;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_131522 = ((double *) mem_144103)[i_143620 * (int64_t) 16 + i_143616];
                
                // futhark/microgpt.fut:378:133-171
                
                double zt_res_131523 = zs_res_131512 * zt_rhs_131522;
                
                // futhark/microgpt.fut:378:149-230
                
                double zp_res_131524 = zt_res_131523 + zt_res_131523;
                
                // futhark/microgpt.fut:378:85-230
                
                double zp_res_131525 = zt_res_131521 + zp_res_131524;
                
                // futhark/microgpt.fut:378:37-230
                
                double zp_res_131526 = zp_lhs_131519 + zp_res_131525;
                
                ((double *) mem_145718)[i_143616] = zp_res_131526;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145713, i_143620 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145718, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143633 = 0; i_143633 < (int64_t) 16; i_143633++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143626 = 0; i_143626 < (int64_t) 16; i_143626++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_142226 = ((double *) mem_145713)[i_143633 * (int64_t) 16 + i_143626];
                
                ((double *) mem_145739)[i_143626] = lifted_lambda_res_142226;
                ((double *) mem_145740)[i_143626] = lifted_lambda_res_142226;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145729, i_143633 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145739, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145730, i_143633 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145740, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143644 = 0; i_143644 < (int64_t) 16; i_143644++) {
            // futhark/microgpt.fut:396:47-59
            
            double zp_lhs_136792 = ((double *) mem_144102)[i_143644];
            
            // futhark/microgpt.fut:396:47-87
            
            double zp_res_136793 = 1.0e-5 + zp_lhs_136792;
            
            // futhark/microgpt.fut:396:39-87
            
            double sqrt_res_136794 = futrts_sqrt64(zp_res_136793);
            
            // futhark/microgpt.fut:398:156-185
            
            double zt_res_136802 = sqrt_res_136794 * sqrt_res_136794;
            
            // futhark/microgpt.fut:398:147-185
            
            double zs_res_136803 = 1.0 / zt_res_136802;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_136804;
            double r_136806 = 0.0;
            
            for (int64_t i_136805 = 0; i_136805 < (int64_t) 16; i_136805++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_136807 = ((double *) mem_145730)[i_143644 * (int64_t) 16 + i_136805];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_136808 = ((double *) mem_param_143976.mem)[i_143644 * (int64_t) 16 + i_136805];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_rhs_136809 = ((double *) mem_144069)[i_143644 * (int64_t) 16 + i_136805];
                
                // futhark/microgpt.fut:398:95-139
                
                double zp_res_136810 = zp_lhs_136808 + zp_rhs_136809;
                
                // futhark/microgpt.fut:398:69-139
                
                double zt_res_136811 = zt_lhs_136807 * zp_res_136810;
                
                // futhark/microgpt.fut:398:90-185
                
                double zt_res_136812 = zs_res_136803 * zt_res_136811;
                
                // futhark/microgpt.fut:398:61-185
                
                double neg_res_136813 = -zt_res_136812;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_136814 = r_136806 + neg_res_136813;
                double r_tmp_146515 = zp_res_136814;
                
                r_136806 = r_tmp_146515;
            }
            defunc_0_lifted_lambda_res_136804 = r_136806;
            // futhark/microgpt.fut:409:47-59
            
            double zp_lhs_136825 = ((double *) mem_144101)[i_143644];
            
            // futhark/microgpt.fut:409:47-87
            
            double zp_res_136826 = 1.0e-5 + zp_lhs_136825;
            
            // futhark/microgpt.fut:409:39-87
            
            double sqrt_res_136827 = futrts_sqrt64(zp_res_136826);
            
            // futhark/microgpt.fut:411:156-185
            
            double zt_res_136835 = sqrt_res_136827 * sqrt_res_136827;
            
            // futhark/microgpt.fut:411:147-185
            
            double zs_res_136836 = 1.0 / zt_res_136835;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_136837;
            double r_136839 = 0.0;
            
            for (int64_t i_136838 = 0; i_136838 < (int64_t) 16; i_136838++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_136840 = ((double *) mem_145729)[i_143644 * (int64_t) 16 + i_136838];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_136841 = ((double *) mem_param_143976.mem)[i_143644 * (int64_t) 16 + i_136838];
                
                // futhark/microgpt.fut:71:46-49
                
                double zp_rhs_136842 = ((double *) mem_144069)[i_143644 * (int64_t) 16 + i_136838];
                
                // futhark/microgpt.fut:411:95-139
                
                double zp_res_136843 = zp_lhs_136841 + zp_rhs_136842;
                
                // futhark/microgpt.fut:411:69-139
                
                double zt_res_136844 = zt_lhs_136840 * zp_res_136843;
                
                // futhark/microgpt.fut:411:90-185
                
                double zt_res_136845 = zs_res_136836 * zt_res_136844;
                
                // futhark/microgpt.fut:411:61-185
                
                double neg_res_136846 = -zt_res_136845;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_136847 = r_136839 + neg_res_136846;
                double r_tmp_146516 = zp_res_136847;
                
                r_136839 = r_tmp_146516;
            }
            defunc_0_lifted_lambda_res_136837 = r_136839;
            ((double *) mem_145761)[i_143644] = defunc_0_lifted_lambda_res_136837;
            ((double *) mem_145762)[i_143644] = sqrt_res_136827;
            ((double *) mem_145763)[i_143644] = defunc_0_lifted_lambda_res_136804;
            ((double *) mem_145764)[i_143644] = sqrt_res_136794;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143653 = 0; i_143653 < (int64_t) 16; i_143653++) {
            // futhark/microgpt.fut:399:39-51
            
            double zt_lhs_136908 = ((double *) mem_145763)[i_143653];
            
            // futhark/microgpt.fut:399:93-105
            
            double zp_lhs_136909 = ((double *) mem_144102)[i_143653];
            
            // futhark/microgpt.fut:399:93-133
            
            double zp_res_136910 = 1.0e-5 + zp_lhs_136909;
            
            // futhark/microgpt.fut:399:85-133
            
            double sqrt_res_136911 = futrts_sqrt64(zp_res_136910);
            
            // futhark/microgpt.fut:399:71-135
            
            double zt_res_136912 = 2.0 * sqrt_res_136911;
            
            // futhark/microgpt.fut:399:57-135
            
            double zs_res_136913 = 1.0 / zt_res_136912;
            
            // futhark/microgpt.fut:399:39-135
            
            double zt_res_136914 = zt_lhs_136908 * zs_res_136913;
            
            // futhark/microgpt.fut:412:39-51
            
            double zt_lhs_136921 = ((double *) mem_145761)[i_143653];
            
            // futhark/microgpt.fut:412:93-105
            
            double zp_lhs_136922 = ((double *) mem_144101)[i_143653];
            
            // futhark/microgpt.fut:412:93-133
            
            double zp_res_136923 = 1.0e-5 + zp_lhs_136922;
            
            // futhark/microgpt.fut:412:85-133
            
            double sqrt_res_136924 = futrts_sqrt64(zp_res_136923);
            
            // futhark/microgpt.fut:412:71-135
            
            double zt_res_136925 = 2.0 * sqrt_res_136924;
            
            // futhark/microgpt.fut:412:57-135
            
            double zs_res_136926 = 1.0 / zt_res_136925;
            
            // futhark/microgpt.fut:412:39-135
            
            double zt_res_136927 = zt_lhs_136921 * zs_res_136926;
            
            ((double *) mem_145789)[i_143653] = zt_res_136927;
            ((double *) mem_145790)[i_143653] = zt_res_136914;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143667 = 0; i_143667 < (int64_t) 16; i_143667++) {
            // futhark/microgpt.fut:400:72-84
            
            double zs_rhs_136945 = ((double *) mem_145764)[i_143667];
            
            // futhark/microgpt.fut:400:64-84
            
            double zs_res_136946 = 1.0 / zs_rhs_136945;
            
            // futhark/microgpt.fut:400:94-106
            
            double zs_lhs_136947 = ((double *) mem_145790)[i_143667];
            
            // futhark/microgpt.fut:400:94-121
            
            double zs_res_136948 = zs_lhs_136947 / 16.0;
            
            // futhark/microgpt.fut:413:94-106
            
            double zs_lhs_136972 = ((double *) mem_145789)[i_143667];
            
            // futhark/microgpt.fut:413:94-121
            
            double zs_res_136973 = zs_lhs_136972 / 16.0;
            
            // futhark/microgpt.fut:413:72-84
            
            double zs_rhs_136970 = ((double *) mem_145762)[i_143667];
            
            // futhark/microgpt.fut:413:64-84
            
            double zs_res_136971 = 1.0 / zs_rhs_136970;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143660 = 0; i_143660 < (int64_t) 16; i_143660++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_142253 = ((double *) mem_145730)[i_143667 * (int64_t) 16 + i_143660];
                
                // futhark/microgpt.fut:400:38-84
                
                double zt_res_142254 = zs_res_136946 * zt_lhs_142253;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_142255 = ((double *) mem_param_143976.mem)[i_143667 * (int64_t) 16 + i_143660];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_142256 = ((double *) mem_144069)[i_143667 * (int64_t) 16 + i_143660];
                
                // futhark/microgpt.fut:400:128-172
                
                double zp_res_142257 = zp_lhs_142255 + zp_rhs_142256;
                
                // futhark/microgpt.fut:400:107-172
                
                double zt_res_142258 = zs_res_136948 * zp_res_142257;
                
                // futhark/microgpt.fut:400:123-259
                
                double zp_res_142259 = zt_res_142258 + zt_res_142258;
                
                // futhark/microgpt.fut:400:59-259
                
                double zp_res_142260 = zt_res_142254 + zp_res_142259;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_142267 = ((double *) mem_145729)[i_143667 * (int64_t) 16 + i_143660];
                
                // futhark/microgpt.fut:413:38-84
                
                double zt_res_142268 = zs_res_136971 * zt_lhs_142267;
                
                // futhark/microgpt.fut:413:107-172
                
                double zt_res_142272 = zs_res_136973 * zp_res_142257;
                
                // futhark/microgpt.fut:413:123-259
                
                double zp_res_142273 = zt_res_142272 + zt_res_142272;
                
                // futhark/microgpt.fut:413:59-259
                
                double zp_res_142274 = zt_res_142268 + zp_res_142273;
                
                ((double *) mem_145813)[i_143660] = zp_res_142274;
                ((double *) mem_145814)[i_143660] = zp_res_142260;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145803, i_143667 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145813, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145804, i_143667 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145814, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143676 = 0; i_143676 < (int64_t) 64; i_143676++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143672 = 0; i_143672 < (int64_t) 16; i_143672++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_131727;
                double r_131729 = 0.0;
                
                for (int64_t i_131728 = 0; i_131728 < (int64_t) 16; i_131728++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_131730 = ((double *) mem_144659)[i_131728 * (int64_t) 64 + i_143676];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_131731 = ((double *) mem_144360)[i_131728 * (int64_t) 16 + i_143672];
                    
                    // futhark/microgpt.fut:405:67-111
                    
                    double zt_res_131732 = zt_lhs_131730 * zt_rhs_131731;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_131733 = r_131729 + zt_res_131732;
                    double r_tmp_146525 = zp_res_131733;
                    
                    r_131729 = r_tmp_146525;
                }
                defunc_0_lifted_lambda_res_131727 = r_131729;
                ((double *) mem_145840)[i_143672] = defunc_0_lifted_lambda_res_131727;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145835, i_143676 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145840, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_143689 = 0; i_143689 < (int64_t) 27; i_143689++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_143682 = 0; i_143682 < (int64_t) 16; i_143682++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_142295;
                double r_142297 = 0.0;
                
                for (int64_t i_142296 = 0; i_142296 < (int64_t) 16; i_142296++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_142298 = ((double *) mem_144626)[i_142296 * (int64_t) 27 + i_143689];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_142299 = ((double *) mem_144421)[i_142296 * (int64_t) 16 + i_143682];
                    
                    // futhark/microgpt.fut:407:68-111
                    
                    double zt_res_142300 = zt_lhs_142298 * zt_rhs_142299;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_142301 = r_142297 + zt_res_142300;
                    double r_tmp_146530 = zp_res_142301;
                    
                    r_142297 = r_tmp_146530;
                }
                defunc_0_lifted_lambda_res_142295 = r_142297;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_142304;
                double r_142306 = 0.0;
                
                for (int64_t i_142305 = 0; i_142305 < (int64_t) 16; i_142305++) {
                    int64_t zeze_lhs_142307 = ((int64_t *) seqs_mem_143960.mem)[step_129423 * (int64_t) 16 + i_142305];
                    
                    // futhark/microgpt.fut:583:58-109
                    
                    bool cond_142308 = zeze_lhs_142307 == i_143689;
                    
                    // futhark/microgpt.fut:583:58-109
                    
                    double lifted_lambda_res_142309;
                    
                    if (cond_142308) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double lifted_lambda_res_t_res_142813 = ((double *) mem_145803)[i_142305 * (int64_t) 16 + i_143682];
                        
                        lifted_lambda_res_142309 = lifted_lambda_res_t_res_142813;
                    } else {
                        lifted_lambda_res_142309 = 0.0;
                    }
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_142315 = r_142306 + lifted_lambda_res_142309;
                    double r_tmp_146531 = zp_res_142315;
                    
                    r_142306 = r_tmp_146531;
                }
                defunc_0_lifted_lambda_res_142304 = r_142306;
                ((double *) mem_145861)[i_143682] = defunc_0_lifted_lambda_res_142304;
                ((double *) mem_145862)[i_143682] = defunc_0_lifted_lambda_res_142295;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145851, i_143689 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145861, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_145852, i_143689 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_145862, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:66:26-45
        
        double i64_res_131898 = sitofp_i64_f64(step_129423);
        
        // futhark/microgpt.fut:518:46-65
        
        double zm_rhs_131899 = i64_res_131898 / 500.0;
        
        // futhark/microgpt.fut:518:24-65
        
        double zt_rhs_131900 = 1.0 - zm_rhs_131899;
        
        // futhark/microgpt.fut:518:19-65
        
        double lt_r_131901 = 1.0e-2 * zt_rhs_131900;
        
        // futhark/microgpt.fut:520:5-52
        if (memblock_alloc(ctx, &mem_145883, (int64_t) 3456, "mem_145883")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:520:5-52
        // futhark/microgpt.fut:520:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145883.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_143984.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:520:5-52
        if (memblock_alloc(ctx, &mem_145885, (int64_t) 3456, "mem_145885")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:520:5-52
        // futhark/microgpt.fut:520:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145885.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_144020.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:520:5-52
        if (memblock_alloc(ctx, &mem_145887, (int64_t) 3456, "mem_145887")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:520:5-52
        // futhark/microgpt.fut:520:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145887.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_144056.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:520:5-52
        if (memblock_alloc(ctx, &mem_145889, (int64_t) 3456, "mem_145889")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:520:5-52
        // futhark/microgpt.fut:520:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145889.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145851, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:520:5-52
        if (futrts_adam_opt_w_12710(ctx, &ext_mem_145893, &ext_mem_145892, &ext_mem_145891, mem_145883, mem_145885, mem_145887, mem_145889, (int64_t) 27, (int64_t) 16, step_129423, lt_r_131901) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_145883, "mem_145883") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145885, "mem_145885") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145887, "mem_145887") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145889, "mem_145889") != 0)
            return 1;
        // futhark/microgpt.fut:522:5-52
        if (memblock_alloc(ctx, &mem_145894, (int64_t) 2048, "mem_145894")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:522:5-52
        // futhark/microgpt.fut:522:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145894.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_143976.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:522:5-52
        if (memblock_alloc(ctx, &mem_145896, (int64_t) 2048, "mem_145896")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:522:5-52
        // futhark/microgpt.fut:522:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145896.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_144012.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:522:5-52
        if (memblock_alloc(ctx, &mem_145898, (int64_t) 2048, "mem_145898")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:522:5-52
        // futhark/microgpt.fut:522:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145898.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_144048.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:522:5-52
        if (memblock_alloc(ctx, &mem_145900, (int64_t) 2048, "mem_145900")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:522:5-52
        // futhark/microgpt.fut:522:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145900.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145804, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:522:5-52
        if (futrts_adam_opt_w_12711(ctx, &ext_mem_145904, &ext_mem_145903, &ext_mem_145902, mem_145894, mem_145896, mem_145898, mem_145900, (int64_t) 16, (int64_t) 16, step_129423, lt_r_131901) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_145894, "mem_145894") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145896, "mem_145896") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145898, "mem_145898") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145900, "mem_145900") != 0)
            return 1;
        // futhark/microgpt.fut:524:5-56
        if (memblock_alloc(ctx, &mem_145905, (int64_t) 2048, "mem_145905")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:524:5-56
        // futhark/microgpt.fut:524:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145905.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_143980.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:524:5-56
        if (memblock_alloc(ctx, &mem_145907, (int64_t) 2048, "mem_145907")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:524:5-56
        // futhark/microgpt.fut:524:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145907.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_144016.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:524:5-56
        if (memblock_alloc(ctx, &mem_145909, (int64_t) 2048, "mem_145909")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:524:5-56
        // futhark/microgpt.fut:524:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145909.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_144052.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:524:5-56
        if (memblock_alloc(ctx, &mem_145911, (int64_t) 2048, "mem_145911")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:524:5-56
        // futhark/microgpt.fut:524:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145911.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145646, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:524:5-56
        if (futrts_adam_opt_w_12711(ctx, &ext_mem_145915, &ext_mem_145914, &ext_mem_145913, mem_145905, mem_145907, mem_145909, mem_145911, (int64_t) 16, (int64_t) 16, step_129423, lt_r_131901) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_145905, "mem_145905") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145907, "mem_145907") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145909, "mem_145909") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145911, "mem_145911") != 0)
            return 1;
        // futhark/microgpt.fut:526:5-56
        if (memblock_alloc(ctx, &mem_145916, (int64_t) 2048, "mem_145916")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:526:5-56
        // futhark/microgpt.fut:526:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145916.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_143968.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:526:5-56
        if (memblock_alloc(ctx, &mem_145918, (int64_t) 2048, "mem_145918")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:526:5-56
        // futhark/microgpt.fut:526:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145918.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_144004.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:526:5-56
        if (memblock_alloc(ctx, &mem_145920, (int64_t) 2048, "mem_145920")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:526:5-56
        // futhark/microgpt.fut:526:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145920.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_144040.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:526:5-56
        if (memblock_alloc(ctx, &mem_145922, (int64_t) 2048, "mem_145922")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:526:5-56
        // futhark/microgpt.fut:526:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145922.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145645, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:526:5-56
        if (futrts_adam_opt_w_12711(ctx, &ext_mem_145926, &ext_mem_145925, &ext_mem_145924, mem_145916, mem_145918, mem_145920, mem_145922, (int64_t) 16, (int64_t) 16, step_129423, lt_r_131901) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_145916, "mem_145916") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145918, "mem_145918") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145920, "mem_145920") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145922, "mem_145922") != 0)
            return 1;
        // futhark/microgpt.fut:528:5-56
        if (memblock_alloc(ctx, &mem_145927, (int64_t) 2048, "mem_145927")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:528:5-56
        // futhark/microgpt.fut:528:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145927.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_143992.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:528:5-56
        if (memblock_alloc(ctx, &mem_145929, (int64_t) 2048, "mem_145929")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:528:5-56
        // futhark/microgpt.fut:528:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145929.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_144028.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:528:5-56
        if (memblock_alloc(ctx, &mem_145931, (int64_t) 2048, "mem_145931")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:528:5-56
        // futhark/microgpt.fut:528:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145931.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_144064.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:528:5-56
        if (memblock_alloc(ctx, &mem_145933, (int64_t) 2048, "mem_145933")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:528:5-56
        // futhark/microgpt.fut:528:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145933.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145644, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:528:5-56
        if (futrts_adam_opt_w_12711(ctx, &ext_mem_145937, &ext_mem_145936, &ext_mem_145935, mem_145927, mem_145929, mem_145931, mem_145933, (int64_t) 16, (int64_t) 16, step_129423, lt_r_131901) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_145927, "mem_145927") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145929, "mem_145929") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145931, "mem_145931") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145933, "mem_145933") != 0)
            return 1;
        // futhark/microgpt.fut:530:5-56
        if (memblock_alloc(ctx, &mem_145938, (int64_t) 2048, "mem_145938")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:530:5-56
        // futhark/microgpt.fut:530:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145938.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_143972.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:530:5-56
        if (memblock_alloc(ctx, &mem_145940, (int64_t) 2048, "mem_145940")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:530:5-56
        // futhark/microgpt.fut:530:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145940.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_144008.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:530:5-56
        if (memblock_alloc(ctx, &mem_145942, (int64_t) 2048, "mem_145942")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:530:5-56
        // futhark/microgpt.fut:530:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145942.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_144044.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:530:5-56
        if (memblock_alloc(ctx, &mem_145944, (int64_t) 2048, "mem_145944")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:530:5-56
        // futhark/microgpt.fut:530:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145944.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145564, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:530:5-56
        if (futrts_adam_opt_w_12711(ctx, &ext_mem_145948, &ext_mem_145947, &ext_mem_145946, mem_145938, mem_145940, mem_145942, mem_145944, (int64_t) 16, (int64_t) 16, step_129423, lt_r_131901) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_145938, "mem_145938") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145940, "mem_145940") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145942, "mem_145942") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145944, "mem_145944") != 0)
            return 1;
        // futhark/microgpt.fut:532:5-52
        if (memblock_alloc(ctx, &mem_145949, (int64_t) 8192, "mem_145949")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:532:5-52
        // futhark/microgpt.fut:532:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145949.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_143988.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:532:5-52
        if (memblock_alloc(ctx, &mem_145951, (int64_t) 8192, "mem_145951")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:532:5-52
        // futhark/microgpt.fut:532:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145951.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_144024.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:532:5-52
        if (memblock_alloc(ctx, &mem_145953, (int64_t) 8192, "mem_145953")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:532:5-52
        // futhark/microgpt.fut:532:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145953.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_144060.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:532:5-52
        if (memblock_alloc(ctx, &mem_145955, (int64_t) 8192, "mem_145955")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:532:5-52
        // futhark/microgpt.fut:532:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145955.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145835, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:532:5-52
        if (futrts_adam_opt_w_12710(ctx, &ext_mem_145959, &ext_mem_145958, &ext_mem_145957, mem_145949, mem_145951, mem_145953, mem_145955, (int64_t) 64, (int64_t) 16, step_129423, lt_r_131901) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_145949, "mem_145949") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145951, "mem_145951") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145953, "mem_145953") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145955, "mem_145955") != 0)
            return 1;
        // futhark/microgpt.fut:534:5-60
        if (memblock_alloc(ctx, &mem_145960, (int64_t) 8192, "mem_145960")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:534:5-60
        // futhark/microgpt.fut:534:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145960.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_143964.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:534:5-60
        if (memblock_alloc(ctx, &mem_145962, (int64_t) 8192, "mem_145962")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:534:5-60
        // futhark/microgpt.fut:534:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145962.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_144000.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:534:5-60
        if (memblock_alloc(ctx, &mem_145964, (int64_t) 8192, "mem_145964")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:534:5-60
        // futhark/microgpt.fut:534:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145964.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_144036.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:534:5-60
        if (memblock_alloc(ctx, &mem_145966, (int64_t) 8192, "mem_145966")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:534:5-60
        // futhark/microgpt.fut:534:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145966.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_144658, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:534:5-60
        if (futrts_adam_opt_w_12710(ctx, &ext_mem_145970, &ext_mem_145969, &ext_mem_145968, mem_145960, mem_145962, mem_145964, mem_145966, (int64_t) 16, (int64_t) 64, step_129423, lt_r_131901) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_145960, "mem_145960") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145962, "mem_145962") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145964, "mem_145964") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145966, "mem_145966") != 0)
            return 1;
        // futhark/microgpt.fut:536:5-56
        if (memblock_alloc(ctx, &mem_145971, (int64_t) 3456, "mem_145971")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:536:5-56
        // futhark/microgpt.fut:536:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145971.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_143996.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:536:5-56
        if (memblock_alloc(ctx, &mem_145973, (int64_t) 3456, "mem_145973")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:536:5-56
        // futhark/microgpt.fut:536:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145973.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_144032.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:536:5-56
        if (memblock_alloc(ctx, &mem_145975, (int64_t) 3456, "mem_145975")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:536:5-56
        // futhark/microgpt.fut:536:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145975.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_144068.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:536:5-56
        if (memblock_alloc(ctx, &mem_145977, (int64_t) 3456, "mem_145977")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:536:5-56
        // futhark/microgpt.fut:536:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_145977.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_145852, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:536:5-56
        if (futrts_adam_opt_w_12710(ctx, &ext_mem_145981, &ext_mem_145980, &ext_mem_145979, mem_145971, mem_145973, mem_145975, mem_145977, (int64_t) 27, (int64_t) 16, step_129423, lt_r_131901) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_145971, "mem_145971") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145973, "mem_145973") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145975, "mem_145975") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145977, "mem_145977") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146189, &ext_mem_145970, "ext_mem_145970") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146190, &ext_mem_145926, "ext_mem_145926") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146191, &ext_mem_145948, "ext_mem_145948") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146192, &ext_mem_145904, "ext_mem_145904") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146193, &ext_mem_145915, "ext_mem_145915") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146194, &ext_mem_145893, "ext_mem_145893") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146195, &ext_mem_145959, "ext_mem_145959") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146196, &ext_mem_145937, "ext_mem_145937") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146197, &ext_mem_145981, "ext_mem_145981") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146198, &ext_mem_145969, "ext_mem_145969") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146199, &ext_mem_145925, "ext_mem_145925") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146200, &ext_mem_145947, "ext_mem_145947") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146201, &ext_mem_145903, "ext_mem_145903") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146202, &ext_mem_145914, "ext_mem_145914") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146203, &ext_mem_145892, "ext_mem_145892") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146204, &ext_mem_145958, "ext_mem_145958") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146205, &ext_mem_145936, "ext_mem_145936") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146206, &ext_mem_145980, "ext_mem_145980") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146207, &ext_mem_145968, "ext_mem_145968") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146208, &ext_mem_145924, "ext_mem_145924") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146209, &ext_mem_145946, "ext_mem_145946") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146210, &ext_mem_145902, "ext_mem_145902") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146211, &ext_mem_145913, "ext_mem_145913") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146212, &ext_mem_145891, "ext_mem_145891") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146213, &ext_mem_145957, "ext_mem_145957") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146214, &ext_mem_145935, "ext_mem_145935") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_146215, &ext_mem_145979, "ext_mem_145979") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_143964, &mem_param_tmp_146189, "mem_param_tmp_146189") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_143968, &mem_param_tmp_146190, "mem_param_tmp_146190") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_143972, &mem_param_tmp_146191, "mem_param_tmp_146191") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_143976, &mem_param_tmp_146192, "mem_param_tmp_146192") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_143980, &mem_param_tmp_146193, "mem_param_tmp_146193") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_143984, &mem_param_tmp_146194, "mem_param_tmp_146194") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_143988, &mem_param_tmp_146195, "mem_param_tmp_146195") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_143992, &mem_param_tmp_146196, "mem_param_tmp_146196") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_143996, &mem_param_tmp_146197, "mem_param_tmp_146197") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_144000, &mem_param_tmp_146198, "mem_param_tmp_146198") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_144004, &mem_param_tmp_146199, "mem_param_tmp_146199") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_144008, &mem_param_tmp_146200, "mem_param_tmp_146200") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_144012, &mem_param_tmp_146201, "mem_param_tmp_146201") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_144016, &mem_param_tmp_146202, "mem_param_tmp_146202") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_144020, &mem_param_tmp_146203, "mem_param_tmp_146203") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_144024, &mem_param_tmp_146204, "mem_param_tmp_146204") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_144028, &mem_param_tmp_146205, "mem_param_tmp_146205") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_144032, &mem_param_tmp_146206, "mem_param_tmp_146206") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_144036, &mem_param_tmp_146207, "mem_param_tmp_146207") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_144040, &mem_param_tmp_146208, "mem_param_tmp_146208") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_144044, &mem_param_tmp_146209, "mem_param_tmp_146209") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_144048, &mem_param_tmp_146210, "mem_param_tmp_146210") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_144052, &mem_param_tmp_146211, "mem_param_tmp_146211") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_144056, &mem_param_tmp_146212, "mem_param_tmp_146212") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_144060, &mem_param_tmp_146213, "mem_param_tmp_146213") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_144064, &mem_param_tmp_146214, "mem_param_tmp_146214") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_144068, &mem_param_tmp_146215, "mem_param_tmp_146215") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_146089, &mem_param_143964, "mem_param_143964") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146088, &mem_param_143968, "mem_param_143968") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146087, &mem_param_143972, "mem_param_143972") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146086, &mem_param_143976, "mem_param_143976") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146085, &mem_param_143980, "mem_param_143980") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146084, &mem_param_143984, "mem_param_143984") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146083, &mem_param_143988, "mem_param_143988") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146082, &mem_param_143992, "mem_param_143992") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146081, &mem_param_143996, "mem_param_143996") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146080, &mem_param_144000, "mem_param_144000") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146079, &mem_param_144004, "mem_param_144004") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146078, &mem_param_144008, "mem_param_144008") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146077, &mem_param_144012, "mem_param_144012") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146076, &mem_param_144016, "mem_param_144016") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146075, &mem_param_144020, "mem_param_144020") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146074, &mem_param_144024, "mem_param_144024") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146073, &mem_param_144028, "mem_param_144028") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146072, &mem_param_144032, "mem_param_144032") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146071, &mem_param_144036, "mem_param_144036") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146070, &mem_param_144040, "mem_param_144040") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146069, &mem_param_144044, "mem_param_144044") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146068, &mem_param_144048, "mem_param_144048") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146067, &mem_param_144052, "mem_param_144052") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146066, &mem_param_144056, "mem_param_144056") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146065, &mem_param_144060, "mem_param_144060") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146064, &mem_param_144064, "mem_param_144064") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_146063, &mem_param_144068, "mem_param_144068") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146162, &ext_mem_146084, "ext_mem_146084") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146163, &ext_mem_146086, "ext_mem_146086") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146164, &ext_mem_146085, "ext_mem_146085") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146165, &ext_mem_146088, "ext_mem_146088") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146166, &ext_mem_146082, "ext_mem_146082") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146167, &ext_mem_146087, "ext_mem_146087") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146168, &ext_mem_146083, "ext_mem_146083") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146169, &ext_mem_146089, "ext_mem_146089") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146170, &ext_mem_146081, "ext_mem_146081") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146171, &ext_mem_146075, "ext_mem_146075") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146172, &ext_mem_146077, "ext_mem_146077") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146173, &ext_mem_146076, "ext_mem_146076") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146174, &ext_mem_146079, "ext_mem_146079") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146175, &ext_mem_146073, "ext_mem_146073") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146176, &ext_mem_146078, "ext_mem_146078") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146177, &ext_mem_146074, "ext_mem_146074") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146178, &ext_mem_146080, "ext_mem_146080") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146179, &ext_mem_146072, "ext_mem_146072") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146180, &ext_mem_146066, "ext_mem_146066") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146181, &ext_mem_146068, "ext_mem_146068") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146182, &ext_mem_146067, "ext_mem_146067") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146183, &ext_mem_146070, "ext_mem_146070") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146184, &ext_mem_146064, "ext_mem_146064") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146185, &ext_mem_146069, "ext_mem_146069") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146186, &ext_mem_146065, "ext_mem_146065") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146187, &ext_mem_146071, "ext_mem_146071") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146188, &ext_mem_146063, "ext_mem_146063") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146685, &mem_out_146162, "mem_out_146162") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146686, &mem_out_146163, "mem_out_146163") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146687, &mem_out_146164, "mem_out_146164") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146688, &mem_out_146165, "mem_out_146165") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146689, &mem_out_146166, "mem_out_146166") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146690, &mem_out_146167, "mem_out_146167") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146691, &mem_out_146168, "mem_out_146168") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146692, &mem_out_146169, "mem_out_146169") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146693, &mem_out_146170, "mem_out_146170") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146694, &mem_out_146171, "mem_out_146171") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146695, &mem_out_146172, "mem_out_146172") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146696, &mem_out_146173, "mem_out_146173") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146697, &mem_out_146174, "mem_out_146174") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146698, &mem_out_146175, "mem_out_146175") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146699, &mem_out_146176, "mem_out_146176") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146700, &mem_out_146177, "mem_out_146177") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146701, &mem_out_146178, "mem_out_146178") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146702, &mem_out_146179, "mem_out_146179") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146703, &mem_out_146180, "mem_out_146180") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146704, &mem_out_146181, "mem_out_146181") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146705, &mem_out_146182, "mem_out_146182") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146706, &mem_out_146183, "mem_out_146183") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146707, &mem_out_146184, "mem_out_146184") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146708, &mem_out_146185, "mem_out_146185") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146709, &mem_out_146186, "mem_out_146186") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146710, &mem_out_146187, "mem_out_146187") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146711, &mem_out_146188, "mem_out_146188") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_144069);
        free(mem_144070);
        free(mem_144079);
        free(mem_144086);
        free(mem_144101);
        free(mem_144102);
        free(mem_144103);
        free(mem_144114);
        free(mem_144121);
        free(mem_144138);
        free(mem_144139);
        free(mem_144147);
        free(mem_144154);
        free(mem_144168);
        free(mem_144169);
        free(mem_144170);
        free(mem_144186);
        free(mem_144187);
        free(mem_144188);
        free(mem_144201);
        free(mem_144202);
        free(mem_144203);
        free(mem_144249);
        free(mem_144254);
        free(mem_144258);
        free(mem_144263);
        free(mem_144274);
        free(mem_144279);
        free(mem_144290);
        free(mem_144295);
        free(mem_144302);
        free(mem_144309);
        free(mem_144320);
        free(mem_144325);
        free(mem_144343);
        free(mem_144348);
        free(mem_144359);
        free(mem_144360);
        free(mem_144368);
        free(mem_144375);
        free(mem_144389);
        free(mem_144394);
        free(mem_144405);
        free(mem_144410);
        free(mem_144421);
        free(mem_144426);
        free(mem_144437);
        free(mem_144442);
        free(mem_144453);
        free(mem_144454);
        free(mem_144455);
        free(mem_144456);
        free(mem_144474);
        free(mem_144479);
        free(mem_144483);
        free(mem_144490);
        free(mem_144524);
        free(mem_144530);
        free(mem_144535);
        free(mem_144551);
        free(mem_144552);
        free(mem_144561);
        free(mem_144562);
        free(mem_144583);
        free(mem_144589);
        free(mem_144594);
        free(mem_144610);
        free(mem_144615);
        free(mem_144626);
        free(mem_144631);
        free(mem_144642);
        free(mem_144647);
        free(mem_144658);
        free(mem_144659);
        free(mem_144668);
        free(mem_144669);
        free(mem_144690);
        free(mem_144695);
        free(mem_144706);
        free(mem_144707);
        free(mem_144720);
        free(mem_144727);
        free(mem_144732);
        free(mem_144743);
        free(mem_144744);
        free(mem_144745);
        free(mem_144746);
        free(mem_144767);
        free(mem_144768);
        free(mem_144769);
        free(mem_144770);
        free(mem_144787);
        free(mem_144794);
        free(mem_144795);
        free(mem_144796);
        free(mem_144851);
        free(mem_144852);
        free(mem_144853);
        free(mem_144854);
        free(mem_144855);
        free(mem_144856);
        free(mem_144887);
        free(mem_144888);
        free(mem_144889);
        free(mem_144890);
        free(mem_144891);
        free(mem_144892);
        free(mem_144917);
        free(mem_144918);
        free(mem_144919);
        free(mem_144938);
        free(mem_144939);
        free(mem_145007);
        free(mem_145008);
        free(mem_145009);
        free(mem_145010);
        free(mem_145011);
        free(mem_145012);
        free(mem_145013);
        free(mem_145014);
        free(mem_145015);
        free(mem_145055);
        free(mem_145056);
        free(mem_145057);
        free(mem_145058);
        free(mem_145059);
        free(mem_145060);
        free(mem_145061);
        free(mem_145062);
        free(mem_145063);
        free(mem_145094);
        free(mem_145095);
        free(mem_145108);
        free(mem_145115);
        free(mem_145122);
        free(mem_145198);
        free(mem_145199);
        free(mem_145200);
        free(mem_145201);
        free(mem_145222);
        free(mem_145223);
        free(mem_145224);
        free(mem_145225);
        free(mem_145242);
        free(mem_145243);
        free(mem_145244);
        free(mem_145245);
        free(mem_145306);
        free(mem_145307);
        free(mem_145308);
        free(mem_145309);
        free(mem_145326);
        free(mem_145327);
        free(mem_145328);
        free(mem_145329);
        free(mem_145370);
        free(mem_145371);
        free(mem_145382);
        free(mem_145383);
        free(mem_145392);
        free(mem_145393);
        free(mem_145424);
        free(mem_145425);
        free(mem_145434);
        free(mem_145435);
        free(mem_145456);
        free(mem_145457);
        free(mem_145468);
        free(mem_145469);
        free(mem_145478);
        free(mem_145479);
        free(mem_145510);
        free(mem_145511);
        free(mem_145522);
        free(mem_145523);
        free(mem_145532);
        free(mem_145533);
        free(mem_145564);
        free(mem_145565);
        free(mem_145566);
        free(mem_145567);
        free(mem_145584);
        free(mem_145585);
        free(mem_145586);
        free(mem_145587);
        free(mem_145628);
        free(mem_145633);
        free(mem_145644);
        free(mem_145645);
        free(mem_145646);
        free(mem_145647);
        free(mem_145648);
        free(mem_145667);
        free(mem_145668);
        free(mem_145669);
        free(mem_145706);
        free(mem_145713);
        free(mem_145718);
        free(mem_145729);
        free(mem_145730);
        free(mem_145739);
        free(mem_145740);
        free(mem_145761);
        free(mem_145762);
        free(mem_145763);
        free(mem_145764);
        free(mem_145789);
        free(mem_145790);
        free(mem_145803);
        free(mem_145804);
        free(mem_145813);
        free(mem_145814);
        free(mem_145835);
        free(mem_145840);
        free(mem_145851);
        free(mem_145852);
        free(mem_145861);
        free(mem_145862);
        if (memblock_unref(ctx, &mem_param_tmp_146215, "mem_param_tmp_146215") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146214, "mem_param_tmp_146214") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146213, "mem_param_tmp_146213") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146212, "mem_param_tmp_146212") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146211, "mem_param_tmp_146211") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146210, "mem_param_tmp_146210") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146209, "mem_param_tmp_146209") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146208, "mem_param_tmp_146208") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146207, "mem_param_tmp_146207") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146206, "mem_param_tmp_146206") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146205, "mem_param_tmp_146205") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146204, "mem_param_tmp_146204") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146203, "mem_param_tmp_146203") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146202, "mem_param_tmp_146202") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146201, "mem_param_tmp_146201") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146200, "mem_param_tmp_146200") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146199, "mem_param_tmp_146199") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146198, "mem_param_tmp_146198") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146197, "mem_param_tmp_146197") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146196, "mem_param_tmp_146196") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146195, "mem_param_tmp_146195") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146194, "mem_param_tmp_146194") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146193, "mem_param_tmp_146193") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146192, "mem_param_tmp_146192") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146191, "mem_param_tmp_146191") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146190, "mem_param_tmp_146190") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_146189, "mem_param_tmp_146189") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145979, "ext_mem_145979") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145980, "ext_mem_145980") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145981, "ext_mem_145981") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145977, "mem_145977") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145975, "mem_145975") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145973, "mem_145973") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145971, "mem_145971") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145968, "ext_mem_145968") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145969, "ext_mem_145969") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145970, "ext_mem_145970") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145966, "mem_145966") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145964, "mem_145964") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145962, "mem_145962") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145960, "mem_145960") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145957, "ext_mem_145957") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145958, "ext_mem_145958") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145959, "ext_mem_145959") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145955, "mem_145955") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145953, "mem_145953") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145951, "mem_145951") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145949, "mem_145949") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145946, "ext_mem_145946") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145947, "ext_mem_145947") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145948, "ext_mem_145948") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145944, "mem_145944") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145942, "mem_145942") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145940, "mem_145940") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145938, "mem_145938") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145935, "ext_mem_145935") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145936, "ext_mem_145936") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145937, "ext_mem_145937") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145933, "mem_145933") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145931, "mem_145931") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145929, "mem_145929") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145927, "mem_145927") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145924, "ext_mem_145924") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145925, "ext_mem_145925") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145926, "ext_mem_145926") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145922, "mem_145922") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145920, "mem_145920") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145918, "mem_145918") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145916, "mem_145916") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145913, "ext_mem_145913") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145914, "ext_mem_145914") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145915, "ext_mem_145915") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145911, "mem_145911") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145909, "mem_145909") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145907, "mem_145907") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145905, "mem_145905") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145902, "ext_mem_145902") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145903, "ext_mem_145903") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145904, "ext_mem_145904") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145900, "mem_145900") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145898, "mem_145898") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145896, "mem_145896") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145894, "mem_145894") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145891, "ext_mem_145891") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145892, "ext_mem_145892") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145893, "ext_mem_145893") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145889, "mem_145889") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145887, "mem_145887") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145885, "mem_145885") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145883, "mem_145883") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_144068, "mem_param_144068") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_144064, "mem_param_144064") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_144060, "mem_param_144060") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_144056, "mem_param_144056") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_144052, "mem_param_144052") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_144048, "mem_param_144048") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_144044, "mem_param_144044") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_144040, "mem_param_144040") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_144036, "mem_param_144036") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_144032, "mem_param_144032") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_144028, "mem_param_144028") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_144024, "mem_param_144024") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_144020, "mem_param_144020") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_144016, "mem_param_144016") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_144012, "mem_param_144012") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_144008, "mem_param_144008") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_144004, "mem_param_144004") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_144000, "mem_param_144000") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_143996, "mem_param_143996") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_143992, "mem_param_143992") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_143988, "mem_param_143988") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_143984, "mem_param_143984") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_143980, "mem_param_143980") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_143976, "mem_param_143976") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_143972, "mem_param_143972") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_143968, "mem_param_143968") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_143964, "mem_param_143964") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146063, "ext_mem_146063") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146064, "ext_mem_146064") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146065, "ext_mem_146065") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146066, "ext_mem_146066") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146067, "ext_mem_146067") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146068, "ext_mem_146068") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146069, "ext_mem_146069") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146070, "ext_mem_146070") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146071, "ext_mem_146071") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146072, "ext_mem_146072") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146073, "ext_mem_146073") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146074, "ext_mem_146074") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146075, "ext_mem_146075") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146076, "ext_mem_146076") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146077, "ext_mem_146077") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146078, "ext_mem_146078") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146079, "ext_mem_146079") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146080, "ext_mem_146080") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146081, "ext_mem_146081") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146082, "ext_mem_146082") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146083, "ext_mem_146083") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146084, "ext_mem_146084") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146085, "ext_mem_146085") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146086, "ext_mem_146086") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146087, "ext_mem_146087") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146088, "ext_mem_146088") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146089, "ext_mem_146089") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146188, "mem_out_146188") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146187, "mem_out_146187") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146186, "mem_out_146186") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146185, "mem_out_146185") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146184, "mem_out_146184") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146183, "mem_out_146183") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146182, "mem_out_146182") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146181, "mem_out_146181") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146180, "mem_out_146180") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146179, "mem_out_146179") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146178, "mem_out_146178") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146177, "mem_out_146177") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146176, "mem_out_146176") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146175, "mem_out_146175") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146174, "mem_out_146174") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146173, "mem_out_146173") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146172, "mem_out_146172") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146171, "mem_out_146171") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146170, "mem_out_146170") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146169, "mem_out_146169") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146168, "mem_out_146168") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146167, "mem_out_146167") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146166, "mem_out_146166") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146165, "mem_out_146165") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146164, "mem_out_146164") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146163, "mem_out_146163") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146162, "mem_out_146162") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_146930, struct memblock *mem_out_p_146931, struct memblock *mem_out_p_146932, struct memblock *mem_out_p_146933, struct memblock *mem_out_p_146934, struct memblock *mem_out_p_146935, struct memblock *mem_out_p_146936, struct memblock *mem_out_p_146937, struct memblock *mem_out_p_146938)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_146170;
    
    mem_out_146170.references = NULL;
    
    struct memblock mem_out_146169;
    
    mem_out_146169.references = NULL;
    
    struct memblock mem_out_146168;
    
    mem_out_146168.references = NULL;
    
    struct memblock mem_out_146167;
    
    mem_out_146167.references = NULL;
    
    struct memblock mem_out_146166;
    
    mem_out_146166.references = NULL;
    
    struct memblock mem_out_146165;
    
    mem_out_146165.references = NULL;
    
    struct memblock mem_out_146164;
    
    mem_out_146164.references = NULL;
    
    struct memblock mem_out_146163;
    
    mem_out_146163.references = NULL;
    
    struct memblock mem_out_146162;
    
    mem_out_146162.references = NULL;
    
    struct memblock mem_143922 = ctx->constants->mem_143922;
    struct memblock mem_143923 = ctx->constants->mem_143923;
    struct memblock mem_143924 = ctx->constants->mem_143924;
    struct memblock mem_143925 = ctx->constants->mem_143925;
    struct memblock mem_143926 = ctx->constants->mem_143926;
    struct memblock mem_143927 = ctx->constants->mem_143927;
    struct memblock mem_143928 = ctx->constants->mem_143928;
    struct memblock mem_143929 = ctx->constants->mem_143929;
    struct memblock mem_143930 = ctx->constants->mem_143930;
    
    if (memblock_set(ctx, &mem_out_146162, &mem_143929, "mem_143929") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146163, &mem_143925, "mem_143925") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146164, &mem_143927, "mem_143927") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146165, &mem_143923, "mem_143923") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146166, &mem_143924, "mem_143924") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146167, &mem_143922, "mem_143922") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146168, &mem_143928, "mem_143928") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146169, &mem_143926, "mem_143926") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_146170, &mem_143930, "mem_143930") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146930, &mem_out_146162, "mem_out_146162") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146931, &mem_out_146163, "mem_out_146163") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146932, &mem_out_146164, "mem_out_146164") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146933, &mem_out_146165, "mem_out_146165") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146934, &mem_out_146166, "mem_out_146166") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146935, &mem_out_146167, "mem_out_146167") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146936, &mem_out_146168, "mem_out_146168") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146937, &mem_out_146169, "mem_out_146169") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_146938, &mem_out_146170, "mem_out_146170") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_146170, "mem_out_146170") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146169, "mem_out_146169") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146168, "mem_out_146168") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146167, "mem_out_146167") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146166, "mem_out_146166") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146165, "mem_out_146165") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146164, "mem_out_146164") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146163, "mem_out_146163") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_146162, "mem_out_146162") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_cal_loss(struct futhark_context *ctx, struct futhark_opaque_tup2_f64_arr1d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3)
{
    double prim_out_146163 = 0.0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_146162;
    
    mem_out_146162.references = NULL;
    
    struct memblock mask_mem_143942;
    
    mask_mem_143942.references = NULL;
    
    struct memblock target_mem_143941;
    
    target_mem_143941.references = NULL;
    
    struct memblock tokens_mem_143940;
    
    tokens_mem_143940.references = NULL;
    
    struct memblock wvoc_mem_143939;
    
    wvoc_mem_143939.references = NULL;
    
    struct memblock wval_mem_143938;
    
    wval_mem_143938.references = NULL;
    
    struct memblock wup_mem_143937;
    
    wup_mem_143937.references = NULL;
    
    struct memblock wte_mem_143936;
    
    wte_mem_143936.references = NULL;
    
    struct memblock wqry_mem_143935;
    
    wqry_mem_143935.references = NULL;
    
    struct memblock wpe_mem_143934;
    
    wpe_mem_143934.references = NULL;
    
    struct memblock wout_mem_143933;
    
    wout_mem_143933.references = NULL;
    
    struct memblock wkey_mem_143932;
    
    wkey_mem_143932.references = NULL;
    
    struct memblock wdown_mem_143931;
    
    wdown_mem_143931.references = NULL;
    wdown_mem_143931 = in0->v0->mem;
    wkey_mem_143932 = in0->v1->mem;
    wout_mem_143933 = in0->v2->mem;
    wpe_mem_143934 = in0->v3->mem;
    wqry_mem_143935 = in0->v4->mem;
    wte_mem_143936 = in0->v5->mem;
    wup_mem_143937 = in0->v6->mem;
    wval_mem_143938 = in0->v7->mem;
    wvoc_mem_143939 = in0->v8->mem;
    tokens_mem_143940 = in1->mem;
    target_mem_143941 = in2->mem;
    mask_mem_143942 = in3->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && (((int64_t) 16 == in2->shape[0] && (int64_t) 27 == in2->shape[1]) && ((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_cal_loss(ctx, &mem_out_146162, &prim_out_146163, wdown_mem_143931, wkey_mem_143932, wout_mem_143933, wpe_mem_143934, wqry_mem_143935, wte_mem_143936, wup_mem_143937, wval_mem_143938, wvoc_mem_143939, tokens_mem_143940, target_mem_143941, mask_mem_143942);
        if (ret == 0) {
            struct memblock mem_143922 = ctx->constants->mem_143922;
            struct memblock mem_143923 = ctx->constants->mem_143923;
            struct memblock mem_143924 = ctx->constants->mem_143924;
            struct memblock mem_143925 = ctx->constants->mem_143925;
            struct memblock mem_143926 = ctx->constants->mem_143926;
            struct memblock mem_143927 = ctx->constants->mem_143927;
            struct memblock mem_143928 = ctx->constants->mem_143928;
            struct memblock mem_143929 = ctx->constants->mem_143929;
            struct memblock mem_143930 = ctx->constants->mem_143930;
            
            assert((*out = (struct futhark_opaque_tup2_f64_arr1d_f64 *) malloc(sizeof(struct futhark_opaque_tup2_f64_arr1d_f64))) != NULL);
            (*out)->v0 = prim_out_146163;
            assert(((*out)->v1 = (struct futhark_f64_1d *) malloc(sizeof(struct futhark_f64_1d))) != NULL);
            (*out)->v1->mem = mem_out_146162;
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
    
    struct memblock mem_out_146162;
    
    mem_out_146162.references = NULL;
    
    struct memblock mask_mem_143941;
    
    mask_mem_143941.references = NULL;
    
    struct memblock tokens_mem_143940;
    
    tokens_mem_143940.references = NULL;
    
    struct memblock wvoc_mem_143939;
    
    wvoc_mem_143939.references = NULL;
    
    struct memblock wval_mem_143938;
    
    wval_mem_143938.references = NULL;
    
    struct memblock wup_mem_143937;
    
    wup_mem_143937.references = NULL;
    
    struct memblock wte_mem_143936;
    
    wte_mem_143936.references = NULL;
    
    struct memblock wqry_mem_143935;
    
    wqry_mem_143935.references = NULL;
    
    struct memblock wpe_mem_143934;
    
    wpe_mem_143934.references = NULL;
    
    struct memblock wout_mem_143933;
    
    wout_mem_143933.references = NULL;
    
    struct memblock wkey_mem_143932;
    
    wkey_mem_143932.references = NULL;
    
    struct memblock wdown_mem_143931;
    
    wdown_mem_143931.references = NULL;
    wdown_mem_143931 = in0->v0->mem;
    wkey_mem_143932 = in0->v1->mem;
    wout_mem_143933 = in0->v2->mem;
    wpe_mem_143934 = in0->v3->mem;
    wqry_mem_143935 = in0->v4->mem;
    wte_mem_143936 = in0->v5->mem;
    wup_mem_143937 = in0->v6->mem;
    wval_mem_143938 = in0->v7->mem;
    wvoc_mem_143939 = in0->v8->mem;
    tokens_mem_143940 = in1->mem;
    mask_mem_143941 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_146162, wdown_mem_143931, wkey_mem_143932, wout_mem_143933, wpe_mem_143934, wqry_mem_143935, wte_mem_143936, wup_mem_143937, wval_mem_143938, wvoc_mem_143939, tokens_mem_143940, mask_mem_143941);
        if (ret == 0) {
            struct memblock mem_143922 = ctx->constants->mem_143922;
            struct memblock mem_143923 = ctx->constants->mem_143923;
            struct memblock mem_143924 = ctx->constants->mem_143924;
            struct memblock mem_143925 = ctx->constants->mem_143925;
            struct memblock mem_143926 = ctx->constants->mem_143926;
            struct memblock mem_143927 = ctx->constants->mem_143927;
            struct memblock mem_143928 = ctx->constants->mem_143928;
            struct memblock mem_143929 = ctx->constants->mem_143929;
            struct memblock mem_143930 = ctx->constants->mem_143930;
            
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_146162;
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
    
    struct memblock mem_out_146170;
    
    mem_out_146170.references = NULL;
    
    struct memblock mem_out_146169;
    
    mem_out_146169.references = NULL;
    
    struct memblock mem_out_146168;
    
    mem_out_146168.references = NULL;
    
    struct memblock mem_out_146167;
    
    mem_out_146167.references = NULL;
    
    struct memblock mem_out_146166;
    
    mem_out_146166.references = NULL;
    
    struct memblock mem_out_146165;
    
    mem_out_146165.references = NULL;
    
    struct memblock mem_out_146164;
    
    mem_out_146164.references = NULL;
    
    struct memblock mem_out_146163;
    
    mem_out_146163.references = NULL;
    
    struct memblock mem_out_146162;
    
    mem_out_146162.references = NULL;
    
    struct memblock wvoc_mem_143939;
    
    wvoc_mem_143939.references = NULL;
    
    struct memblock wdown_mem_143938;
    
    wdown_mem_143938.references = NULL;
    
    struct memblock wup_mem_143937;
    
    wup_mem_143937.references = NULL;
    
    struct memblock wout_mem_143936;
    
    wout_mem_143936.references = NULL;
    
    struct memblock wval_mem_143935;
    
    wval_mem_143935.references = NULL;
    
    struct memblock wkey_mem_143934;
    
    wkey_mem_143934.references = NULL;
    
    struct memblock wqry_mem_143933;
    
    wqry_mem_143933.references = NULL;
    
    struct memblock wpe_mem_143932;
    
    wpe_mem_143932.references = NULL;
    
    struct memblock wte_mem_143931;
    
    wte_mem_143931.references = NULL;
    wte_mem_143931 = in0->mem;
    wpe_mem_143932 = in1->mem;
    wqry_mem_143933 = in2->mem;
    wkey_mem_143934 = in3->mem;
    wval_mem_143935 = in4->mem;
    wout_mem_143936 = in5->mem;
    wup_mem_143937 = in6->mem;
    wdown_mem_143938 = in7->mem;
    wvoc_mem_143939 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_146162, &mem_out_146163, &mem_out_146164, &mem_out_146165, &mem_out_146166, &mem_out_146167, &mem_out_146168, &mem_out_146169, &mem_out_146170, wte_mem_143931, wpe_mem_143932, wqry_mem_143933, wkey_mem_143934, wval_mem_143935, wout_mem_143936, wup_mem_143937, wdown_mem_143938, wvoc_mem_143939);
        if (ret == 0) {
            struct memblock mem_143922 = ctx->constants->mem_143922;
            struct memblock mem_143923 = ctx->constants->mem_143923;
            struct memblock mem_143924 = ctx->constants->mem_143924;
            struct memblock mem_143925 = ctx->constants->mem_143925;
            struct memblock mem_143926 = ctx->constants->mem_143926;
            struct memblock mem_143927 = ctx->constants->mem_143927;
            struct memblock mem_143928 = ctx->constants->mem_143928;
            struct memblock mem_143929 = ctx->constants->mem_143929;
            struct memblock mem_143930 = ctx->constants->mem_143930;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_146162;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_146163;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_146164;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_146165;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_146166;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_146167;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_146168;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_146169;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_146170;
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
    
    struct memblock mem_out_146188;
    
    mem_out_146188.references = NULL;
    
    struct memblock mem_out_146187;
    
    mem_out_146187.references = NULL;
    
    struct memblock mem_out_146186;
    
    mem_out_146186.references = NULL;
    
    struct memblock mem_out_146185;
    
    mem_out_146185.references = NULL;
    
    struct memblock mem_out_146184;
    
    mem_out_146184.references = NULL;
    
    struct memblock mem_out_146183;
    
    mem_out_146183.references = NULL;
    
    struct memblock mem_out_146182;
    
    mem_out_146182.references = NULL;
    
    struct memblock mem_out_146181;
    
    mem_out_146181.references = NULL;
    
    struct memblock mem_out_146180;
    
    mem_out_146180.references = NULL;
    
    struct memblock mem_out_146179;
    
    mem_out_146179.references = NULL;
    
    struct memblock mem_out_146178;
    
    mem_out_146178.references = NULL;
    
    struct memblock mem_out_146177;
    
    mem_out_146177.references = NULL;
    
    struct memblock mem_out_146176;
    
    mem_out_146176.references = NULL;
    
    struct memblock mem_out_146175;
    
    mem_out_146175.references = NULL;
    
    struct memblock mem_out_146174;
    
    mem_out_146174.references = NULL;
    
    struct memblock mem_out_146173;
    
    mem_out_146173.references = NULL;
    
    struct memblock mem_out_146172;
    
    mem_out_146172.references = NULL;
    
    struct memblock mem_out_146171;
    
    mem_out_146171.references = NULL;
    
    struct memblock mem_out_146170;
    
    mem_out_146170.references = NULL;
    
    struct memblock mem_out_146169;
    
    mem_out_146169.references = NULL;
    
    struct memblock mem_out_146168;
    
    mem_out_146168.references = NULL;
    
    struct memblock mem_out_146167;
    
    mem_out_146167.references = NULL;
    
    struct memblock mem_out_146166;
    
    mem_out_146166.references = NULL;
    
    struct memblock mem_out_146165;
    
    mem_out_146165.references = NULL;
    
    struct memblock mem_out_146164;
    
    mem_out_146164.references = NULL;
    
    struct memblock mem_out_146163;
    
    mem_out_146163.references = NULL;
    
    struct memblock mem_out_146162;
    
    mem_out_146162.references = NULL;
    
    struct memblock seqs_mem_143960;
    
    seqs_mem_143960.references = NULL;
    
    struct memblock dls_mem_143959;
    
    dls_mem_143959.references = NULL;
    
    struct memblock masks_mem_143958;
    
    masks_mem_143958.references = NULL;
    
    struct memblock wvoc_mem_143957;
    
    wvoc_mem_143957.references = NULL;
    
    struct memblock wval_mem_143956;
    
    wval_mem_143956.references = NULL;
    
    struct memblock wup_mem_143955;
    
    wup_mem_143955.references = NULL;
    
    struct memblock wte_mem_143954;
    
    wte_mem_143954.references = NULL;
    
    struct memblock wqry_mem_143953;
    
    wqry_mem_143953.references = NULL;
    
    struct memblock wpe_mem_143952;
    
    wpe_mem_143952.references = NULL;
    
    struct memblock wout_mem_143951;
    
    wout_mem_143951.references = NULL;
    
    struct memblock wkey_mem_143950;
    
    wkey_mem_143950.references = NULL;
    
    struct memblock wdown_mem_143949;
    
    wdown_mem_143949.references = NULL;
    
    struct memblock wvoc_mem_143948;
    
    wvoc_mem_143948.references = NULL;
    
    struct memblock wval_mem_143947;
    
    wval_mem_143947.references = NULL;
    
    struct memblock wup_mem_143946;
    
    wup_mem_143946.references = NULL;
    
    struct memblock wte_mem_143945;
    
    wte_mem_143945.references = NULL;
    
    struct memblock wqry_mem_143944;
    
    wqry_mem_143944.references = NULL;
    
    struct memblock wpe_mem_143943;
    
    wpe_mem_143943.references = NULL;
    
    struct memblock wout_mem_143942;
    
    wout_mem_143942.references = NULL;
    
    struct memblock wkey_mem_143941;
    
    wkey_mem_143941.references = NULL;
    
    struct memblock wdown_mem_143940;
    
    wdown_mem_143940.references = NULL;
    
    struct memblock wvoc_mem_143939;
    
    wvoc_mem_143939.references = NULL;
    
    struct memblock wval_mem_143938;
    
    wval_mem_143938.references = NULL;
    
    struct memblock wup_mem_143937;
    
    wup_mem_143937.references = NULL;
    
    struct memblock wte_mem_143936;
    
    wte_mem_143936.references = NULL;
    
    struct memblock wqry_mem_143935;
    
    wqry_mem_143935.references = NULL;
    
    struct memblock wpe_mem_143934;
    
    wpe_mem_143934.references = NULL;
    
    struct memblock wout_mem_143933;
    
    wout_mem_143933.references = NULL;
    
    struct memblock wkey_mem_143932;
    
    wkey_mem_143932.references = NULL;
    
    struct memblock wdown_mem_143931;
    
    wdown_mem_143931.references = NULL;
    wdown_mem_143931 = in0->v0->mem;
    wkey_mem_143932 = in0->v1->mem;
    wout_mem_143933 = in0->v2->mem;
    wpe_mem_143934 = in0->v3->mem;
    wqry_mem_143935 = in0->v4->mem;
    wte_mem_143936 = in0->v5->mem;
    wup_mem_143937 = in0->v6->mem;
    wval_mem_143938 = in0->v7->mem;
    wvoc_mem_143939 = in0->v8->mem;
    wdown_mem_143940 = in1->v0->mem;
    wkey_mem_143941 = in1->v1->mem;
    wout_mem_143942 = in1->v2->mem;
    wpe_mem_143943 = in1->v3->mem;
    wqry_mem_143944 = in1->v4->mem;
    wte_mem_143945 = in1->v5->mem;
    wup_mem_143946 = in1->v6->mem;
    wval_mem_143947 = in1->v7->mem;
    wvoc_mem_143948 = in1->v8->mem;
    wdown_mem_143949 = in2->v0->mem;
    wkey_mem_143950 = in2->v1->mem;
    wout_mem_143951 = in2->v2->mem;
    wpe_mem_143952 = in2->v3->mem;
    wqry_mem_143953 = in2->v4->mem;
    wte_mem_143954 = in2->v5->mem;
    wup_mem_143955 = in2->v6->mem;
    wval_mem_143956 = in2->v7->mem;
    wvoc_mem_143957 = in2->v8->mem;
    masks_mem_143958 = in3->mem;
    dls_mem_143959 = in4->mem;
    seqs_mem_143960 = in5->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && (((int64_t) 500 == in3->shape[0] && ((int64_t) 16 == in3->shape[1] && (int64_t) 16 == in3->shape[2])) && ((int64_t) 500 == in4->shape[0] && ((int64_t) 500 == in5->shape[0] && (int64_t) 16 == in5->shape[1]))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_146162, &mem_out_146163, &mem_out_146164, &mem_out_146165, &mem_out_146166, &mem_out_146167, &mem_out_146168, &mem_out_146169, &mem_out_146170, &mem_out_146171, &mem_out_146172, &mem_out_146173, &mem_out_146174, &mem_out_146175, &mem_out_146176, &mem_out_146177, &mem_out_146178, &mem_out_146179, &mem_out_146180, &mem_out_146181, &mem_out_146182, &mem_out_146183, &mem_out_146184, &mem_out_146185, &mem_out_146186, &mem_out_146187, &mem_out_146188, wdown_mem_143931, wkey_mem_143932, wout_mem_143933, wpe_mem_143934, wqry_mem_143935, wte_mem_143936, wup_mem_143937, wval_mem_143938, wvoc_mem_143939, wdown_mem_143940, wkey_mem_143941, wout_mem_143942, wpe_mem_143943, wqry_mem_143944, wte_mem_143945, wup_mem_143946, wval_mem_143947, wvoc_mem_143948, wdown_mem_143949, wkey_mem_143950, wout_mem_143951, wpe_mem_143952, wqry_mem_143953, wte_mem_143954, wup_mem_143955, wval_mem_143956, wvoc_mem_143957, masks_mem_143958, dls_mem_143959, seqs_mem_143960);
        if (ret == 0) {
            struct memblock mem_143922 = ctx->constants->mem_143922;
            struct memblock mem_143923 = ctx->constants->mem_143923;
            struct memblock mem_143924 = ctx->constants->mem_143924;
            struct memblock mem_143925 = ctx->constants->mem_143925;
            struct memblock mem_143926 = ctx->constants->mem_143926;
            struct memblock mem_143927 = ctx->constants->mem_143927;
            struct memblock mem_143928 = ctx->constants->mem_143928;
            struct memblock mem_143929 = ctx->constants->mem_143929;
            struct memblock mem_143930 = ctx->constants->mem_143930;
            
            assert((*out = (struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_146162;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_146163;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_146164;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_146165;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_146166;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_146167;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_146168;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_146169;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_146170;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v9->mem = mem_out_146171;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v10->mem = mem_out_146172;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v11->mem = mem_out_146173;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v12->mem = mem_out_146174;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v13->mem = mem_out_146175;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v14->mem = mem_out_146176;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v15->mem = mem_out_146177;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v16->mem = mem_out_146178;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v17->mem = mem_out_146179;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v18->mem = mem_out_146180;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v19->mem = mem_out_146181;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v20->mem = mem_out_146182;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v21->mem = mem_out_146183;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v22->mem = mem_out_146184;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v23->mem = mem_out_146185;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v24->mem = mem_out_146186;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v25->mem = mem_out_146187;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v26->mem = mem_out_146188;
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
    
    struct memblock mem_out_146170;
    
    mem_out_146170.references = NULL;
    
    struct memblock mem_out_146169;
    
    mem_out_146169.references = NULL;
    
    struct memblock mem_out_146168;
    
    mem_out_146168.references = NULL;
    
    struct memblock mem_out_146167;
    
    mem_out_146167.references = NULL;
    
    struct memblock mem_out_146166;
    
    mem_out_146166.references = NULL;
    
    struct memblock mem_out_146165;
    
    mem_out_146165.references = NULL;
    
    struct memblock mem_out_146164;
    
    mem_out_146164.references = NULL;
    
    struct memblock mem_out_146163;
    
    mem_out_146163.references = NULL;
    
    struct memblock mem_out_146162;
    
    mem_out_146162.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_146162, &mem_out_146163, &mem_out_146164, &mem_out_146165, &mem_out_146166, &mem_out_146167, &mem_out_146168, &mem_out_146169, &mem_out_146170);
        if (ret == 0) {
            struct memblock mem_143922 = ctx->constants->mem_143922;
            struct memblock mem_143923 = ctx->constants->mem_143923;
            struct memblock mem_143924 = ctx->constants->mem_143924;
            struct memblock mem_143925 = ctx->constants->mem_143925;
            struct memblock mem_143926 = ctx->constants->mem_143926;
            struct memblock mem_143927 = ctx->constants->mem_143927;
            struct memblock mem_143928 = ctx->constants->mem_143928;
            struct memblock mem_143929 = ctx->constants->mem_143929;
            struct memblock mem_143930 = ctx->constants->mem_143930;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_146162;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_146163;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_146164;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_146165;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_146166;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_146167;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_146168;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_146169;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_146170;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
